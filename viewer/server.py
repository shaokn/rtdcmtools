"""Local, read-only DICOM-RT research viewer. Run with pyauto Python."""
import argparse
from collections import OrderedDict
import csv
from functools import wraps
import io
import json
from pathlib import Path
import threading
import secrets
import time

import numpy as np
import pydicom
import SimpleITK as sitk
from flask import Flask, Response, jsonify, request, send_from_directory, has_request_context
import nifti_backend
import quick_backend
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from scipy.ndimage import binary_erosion
from rt_utils import RTStructBuilder

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent.parent
ROOT = OUTPUT_ROOT / 'organized_dicom'
NIFTI_ROOT = OUTPUT_ROOT / 'nifti_data'
app = Flask(__name__, static_folder=str(HERE / 'static'))
LOCK = threading.RLock()
CACHE = OrderedDict()
QUICK = OrderedDict()
QUICK_TEMP = OUTPUT_ROOT.parent / 'temp'
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 ** 2
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)


def catalog():
    return {p.parent.name: p for p in sorted(ROOT.glob('*/relationships.json'))}


def records(case):
    paths = catalog()
    if case not in paths:
        raise ValueError('病例不存在')
    return json.loads(paths[case].read_text())['files']


def path_for(case, record):
    base = (ROOT / case).resolve()
    path = (base / record['path']).resolve()
    if base not in path.parents:
        raise ValueError('无效文件路径')
    return path


def image_from_array(array, origin, spacing, direction):
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetOrigin(tuple(float(x) for x in origin))
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetDirection(tuple(float(x) for x in np.asarray(direction).ravel()))
    return image


def load_ct(paths):
    data = [pydicom.dcmread(p) for p in paths]
    first = data[0]
    orientation = np.array(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    data.sort(key=lambda d: np.dot(np.asarray(d.ImagePositionPatient, float), normal))
    first = data[0]
    positions = np.array([d.ImagePositionPatient for d in data], float)
    for d in data:
        if (not np.allclose(d.ImageOrientationPatient, orientation, atol=1e-5)
                or not np.allclose(d.PixelSpacing, first.PixelSpacing)
                or d.Rows != first.Rows or d.Columns != first.Columns):
            raise ValueError('CT 序列的几何不一致，暂不显示')
    if len(data) < 2:
        raise ValueError('三切面浏览至少需要两张 CT')
    steps = np.diff(positions @ normal)
    dz = float(np.median(steps))
    if dz <= 0 or not np.allclose(np.diff(positions, axis=0), normal * dz, atol=.02):
        raise ValueError('CT 层间距不规则或存在倾斜，暂不显示')
    direction = np.column_stack([orientation[:3], orientation[3:], normal])
    if not np.allclose(direction, np.eye(3), atol=1e-5):
        raise ValueError('当前版本仅支持 LPS 正向轴位 CT；未对斜位扫描进行隐式重采样')
    array = np.stack([d.pixel_array.astype(np.float32) * float(d.get('RescaleSlope', 1))
                      + float(d.get('RescaleIntercept', 0)) for d in data])
    spacing = [float(first.PixelSpacing[1]), float(first.PixelSpacing[0]), dz]
    image = image_from_array(array, positions[0], spacing, direction)
    return image, array, data


def load_dose(path):
    ds = pydicom.dcmread(path)
    if ds.get('DoseUnits') != 'GY':
        raise ValueError('当前仅显示绝对剂量 GY')
    arr = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError('当前剂量必须是多帧三维剂量')
    o = np.asarray(ds.ImageOrientationPatient, float)
    normal = np.cross(o[:3], o[3:])
    pos = np.asarray(ds.ImagePositionPatient, float)
    offsets = np.asarray(ds.GridFrameOffsetVector, float)
    if len(offsets) != arr.shape[0]:
        raise ValueError('剂量帧数与位置数不一致')
    if np.isclose(offsets[0], 0, atol=.001):
        origin = pos
    elif np.allclose(o, [1, 0, 0, 0, 1, 0]) and np.isclose(offsets[0], pos[2], atol=.001):
        origin = pos
        offsets = offsets - pos[2]
    else:
        raise ValueError('无法解释剂量 GridFrameOffsetVector')
    steps = np.diff(offsets)
    dz = float(np.median(steps))
    if abs(dz) < 1e-8 or not np.allclose(steps, dz, atol=.001):
        raise ValueError('不支持非等间距剂量网格')
    direction = np.column_stack([o[:3], o[3:], normal * np.sign(dz)])
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('剂量方向无效')
    image = image_from_array(arr, origin,
                             [ds.PixelSpacing[1], ds.PixelSpacing[0], abs(dz)], direction)
    return image, ds, float(arr.max())


class Volume:
    def __init__(self, case, series):
        self.case = case
        self.files = records(case)
        self.ct_records = [r for r in self.files if r['modality'] == 'CT' and r['series_uid'] == series]
        if not self.ct_records:
            raise ValueError('CT 序列不存在')
        self.image, self.array, self.datasets = load_ct([path_for(case, r) for r in self.ct_records])
        self.series = series
        self.frame = str(self.datasets[0].FrameOfReferenceUID)
        self.masks = OrderedDict()
        self.builders = {}
        self.doses = OrderedDict()

    def record(self, uid, modality):
        found = next((r for r in self.files if r.get('sop_uid') == uid and r['modality'] == modality), None)
        if found is None:
            raise ValueError(f'{modality} 对象不存在')
        return found

    def colors(self, struct):
        ds = pydicom.dcmread(path_for(self.case, self.record(struct, 'RTSTRUCT')), stop_before_pixels=True)
        return {int(c.ReferencedROINumber): list(c.get('ROIDisplayColor', [30, 210, 170]))
                for c in ds.get('ROIContourSequence', [])}

    def mask(self, struct, roi):
        key = (struct, int(roi))
        if key in self.masks:
            self.masks.move_to_end(key)
            return self.masks[key]
        record = self.record(struct, 'RTSTRUCT')
        if self.series not in record.get('ct_series_uids', []):
            raise ValueError('所选结构没有引用当前 CT，不能叠加')
        # rt-utils 1.2.7 assumes square pixels and square slices in its mask buffer.
        first = self.datasets[0]
        if first.Rows != first.Columns or not np.isclose(*first.PixelSpacing):
            raise ValueError('当前结构栅格化仅支持方形切片与等距平面像素')
        if struct not in self.builders:
            self.builders[struct] = RTStructBuilder.create_from(
                str(path_for(self.case, self.ct_records[0]).parent),
                str(path_for(self.case, record)))
        builder = self.builders[struct]
        if [str(d.SOPInstanceUID) for d in builder.series_data] != [str(d.SOPInstanceUID) for d in self.datasets]:
            raise ValueError('结构栅格化的 CT 层序不一致')
        name = next((r['name'] for r in record['rois'] if r['number'] == int(roi)), None)
        if name is None:
            raise ValueError('ROI 不存在')
        mask = builder.get_roi_mask_by_name(name).transpose(2, 0, 1)
        if mask.shape != self.array.shape:
            raise ValueError('结构网格与 CT 不一致')
        self.masks[key] = mask
        while len(self.masks) > 8:
            self.masks.popitem(last=False)
        return mask

    def dose(self, uid):
        if uid in self.doses:
            self.doses.move_to_end(uid)
            return self.doses[uid]
        record = self.record(uid, 'RTDOSE')
        image, ds, maximum = load_dose(path_for(self.case, record))
        if str(ds.get('FrameOfReferenceUID', '')) != self.frame:
            raise ValueError('剂量与 CT 的 FrameOfReferenceUID 不一致，不能直接叠加')
        plans = [self.record(u, 'RTPLAN') for u in record.get('plan_uids', [])]
        structures = [self.record(u, 'RTSTRUCT') for p in plans for u in p.get('structure_uids', [])]
        if not any(self.series in s.get('ct_series_uids', []) for s in structures):
            raise ValueError('所选剂量没有通过计划和结构关联到当前 CT')
        mapped = sitk.Resample(image, self.image, sitk.Transform(), sitk.sitkLinear,
                               float('nan'), sitk.sitkFloat32)
        value = (sitk.GetArrayFromImage(mapped), maximum)
        self.doses[uid] = value
        while len(self.doses) > 1:
            self.doses.popitem(last=False)
        return value


def volume(case, series):
    source = data_source()
    if source == 'quick':
        if series != 'ct':
            raise ValueError('快速查看影像不存在')
        return quick_volume(case)
    key = (source, case, series)
    if key not in CACHE:
        CACHE.clear()
        CACHE[key] = nifti_backend.Volume(NIFTI_ROOT, case, series) if source == 'nifti' else Volume(case, series)
    return CACHE[key]


def data_source():
    source = request.args.get('source', 'dicom') if has_request_context() else 'dicom'
    if source not in ('dicom', 'nifti', 'quick'):
        raise ValueError('未知数据源')
    return source


def api(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        try:
            with LOCK:
                expire_quick()
                return fn(*args, **kwargs)
        except RequestEntityTooLarge:
            return jsonify(error='所选文件总大小不能超过 512 MiB'), 413
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Image request failed')
            return jsonify(error='影像处理失败，请检查本地服务日志'), 500
    return call


def expire_quick():
    for token, (_, touched) in list(QUICK.items()):
        if time.monotonic() - touched > 3600:
            del QUICK[token]


def quick_volume(token):
    if token not in QUICK:
        raise ValueError('快速查看会话已清除或过期，请重新打开文件')
    value, _ = QUICK[token]
    QUICK[token] = value, time.monotonic()
    QUICK.move_to_end(token)
    return value


@app.post('/api/quick')
@api
def quick_open():
    if request.headers.get('X-Quick-View') != '1':
        raise ValueError('请通过本地快速查看面板打开文件')
    value = quick_backend.Volume(request.files, request.form, QUICK_TEMP)
    token = secrets.token_urlsafe(24)
    value.info['case'] = token
    QUICK[token] = value, time.monotonic()
    while len(QUICK) > 2:
        QUICK.popitem(last=False)
    return jsonify(value.info)


@app.delete('/api/quick')
@api
def quick_clear():
    if request.headers.get('X-Quick-View') != '1':
        raise ValueError('无效清除请求')
    QUICK.pop(request.args.get('case', ''), None)
    return jsonify(cleared=True)


@app.after_request
def private(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/')
def home():
    return send_from_directory(HERE / 'static', 'index.html')


@app.get('/api/cases')
@api
def cases():
    if data_source() == 'quick':
        return jsonify([])
    if data_source() == 'nifti':
        items = []
        for case in nifti_backend.catalog(NIFTI_ROOT):
            data = nifti_backend.info(NIFTI_ROOT, case)
            items.append(dict(id=case, ct_slices=sum(s['count'] for s in data['series']),
                              plans=[p['name'] for p in data['plans']],
                              structures=len(data['structures']), doses=len(data['doses'])))
        return jsonify(items)
    items = []
    for case in catalog():
        files = records(case)
        items.append({'id': case, 'ct_slices': sum(r['modality'] == 'CT' for r in files),
                      'plans': [r.get('plan_label') or r.get('plan_name') for r in files if r['modality'] == 'RTPLAN'],
                      'structures': sum(r['modality'] == 'RTSTRUCT' for r in files),
                      'doses': sum(r['modality'] == 'RTDOSE' for r in files)})
    return jsonify(items)


@app.get('/api/config')
def config():
    return jsonify(default_source=app.config.get('DEFAULT_SOURCE', 'dicom'))


@app.get('/api/info')
@api
def info():
    case = request.args['case']
    if data_source() == 'quick':
        return jsonify(quick_volume(case).info)
    if data_source() == 'nifti':
        return jsonify(nifti_backend.info(NIFTI_ROOT, case))
    files = records(case)
    series = {}
    plans, structures, doses = [], [], []
    for r in files:
        if r['modality'] == 'CT':
            s = series.setdefault(r['series_uid'], {'uid': r['series_uid'], 'name': r['folder'],
                                    'description': r['series_description'], 'date': r['date_time'], 'count': 0})
            s['count'] += 1
        elif r['modality'] == 'RTPLAN':
            ds = pydicom.dcmread(path_for(case, r), stop_before_pixels=True)
            plans.append({'uid': r['sop_uid'], 'name': r['plan_label'] or r['plan_name'],
                          'structs': r['structure_uids'], 'date': str(ds.get('RTPlanDate', '')),
                          'fractions': [int(g.NumberOfFractionsPlanned) for g in ds.get('FractionGroupSequence', [])
                                        if g.get('NumberOfFractionsPlanned') is not None],
                          'beams': [{'number': int(b.BeamNumber), 'name': str(b.BeamName),
                                     'type': str(b.get('RadiationType', ''))} for b in ds.get('BeamSequence', [])]})
        elif r['modality'] == 'RTSTRUCT':
            ds = pydicom.dcmread(path_for(case, r), stop_before_pixels=True)
            colors = {int(c.ReferencedROINumber): list(c.get('ROIDisplayColor', [30, 210, 170]))
                      for c in ds.get('ROIContourSequence', [])}
            structures.append({'uid': r['sop_uid'], 'name': r['folder'], 'series': r['ct_series_uids'],
                               'rois': [{**x, 'color': colors.get(x['number'], [30, 210, 170])} for x in r['rois']]})
        elif r['modality'] == 'RTDOSE':
            doses.append({'uid': r['sop_uid'], 'type': r['dose_type'], 'plans': r['plan_uids'],
                          'units': r['dose_units'], 'beams': r['beams']})
    return jsonify(case=case, series=list(series.values()), plans=plans, structures=structures, doses=doses)


@app.get('/api/volume')
@api
def meta():
    v = volume(request.args['case'], request.args['series'])
    return jsonify(size=list(v.image.GetSize()), spacing=list(v.image.GetSpacing()),
                   origin=list(v.image.GetOrigin()), range=[float(v.array.min()), float(v.array.max())])


def plane(array, axis, index):
    if axis == 'axial':
        return array[index], (1, 0)
    if axis == 'coronal':
        return array[::-1, index, :], (2, 0)
    if axis == 'sagittal':
        return array[::-1, :, index], (2, 1)
    raise ValueError('无效切面')


def color_dose(values, maximum):
    stops = np.array([[30, 100, 245], [0, 200, 215], [70, 215, 110],
                      [250, 222, 75], [255, 126, 44], [243, 55, 75]], float)
    t = np.nan_to_num(values / max(maximum, .001), nan=0).clip(0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    return stops[i] * (1 - (t - i)[..., None]) + stops[i + 1] * (t - i)[..., None]


@app.get('/api/slice')
@api
def slice_image():
    a = request.args
    v = volume(a['case'], a['series'])
    axis, index = a.get('axis', 'axial'), int(a.get('index', 0))
    dim = {'axial': 2, 'coronal': 1, 'sagittal': 0}.get(axis)
    if dim is None or not 0 <= index < v.image.GetSize()[dim]:
        raise ValueError('切片位置超出范围')
    ww, wl = float(a.get('ww', 400)), float(a.get('wl', 40))
    if not np.isfinite([ww, wl]).all() or ww <= 0:
        raise ValueError('窗宽必须大于零')
    image, axes = plane(v.array, axis, index)
    gray = np.clip((image - wl + ww / 2) / ww, 0, 1) * 255
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if a.get('dose'):
        dose, maximum = v.dose(a['dose'])
        dv, _ = plane(dose, axis, index)
        opacity = np.clip(float(a.get('opacity', .35)), 0, 1)
        mode = a.get('dose_mode', 'wash')
        levels = [float(s) for s in a.get('levels', '5,10,20,30,40').split(',') if s]
        if len(levels) > 16 or not np.isfinite(levels).all() or any(x < 0 for x in levels):
            raise ValueError('等剂量级别无效')
        levels = sorted(set(levels))
        if mode not in ('wash', 'lines', 'filled') or (mode != 'wash' and not levels):
            raise ValueError('请选择显示方式并填写等剂量级别')
        if mode == 'wash':
            weight = (np.isfinite(dv) & (dv >= float(a.get('threshold', 1))))[..., None] * opacity
            rgb = rgb * (1 - weight) + color_dose(dv, maximum) * weight
        elif mode == 'filled':
            # Disjoint dose bands avoid repeatedly blending nested isodose regions.
            bands = np.searchsorted(levels, dv, side='right') - 1
            for band, level in enumerate(levels):
                inside = np.isfinite(dv) & (bands == band)
                color = color_dose(np.array([level]), maximum)[0]
                rgb[inside] = rgb[inside] * (1 - opacity) + color * opacity
        else:
            for level in levels:
                inside = np.isfinite(dv) & (dv >= level)
                edge = inside & ~binary_erosion(inside)
                rgb[edge] = rgb[edge] * (1 - opacity) + color_dose(np.array([level]), maximum)[0] * opacity
    selected = [int(s) for s in a.get('rois', '').split(',') if s]
    if len(selected) > 12:
        raise ValueError('同时显示最多 12 个结构')
    if selected:
        colors = v.colors(a['struct'])
        for number in selected:
            mask, _ = plane(v.mask(a['struct'], number), axis, index)
            edge = mask & ~binary_erosion(mask)
            color = np.array(colors.get(number, [30, 210, 170]))
            if a.get('fill') == '1':
                rgb[mask] = rgb[mask] * .8 + color * .2
            rgb[edge] = color
    spacing = v.image.GetSpacing()
    physical_h = image.shape[0] * spacing[axes[0]]
    physical_w = image.shape[1] * spacing[axes[1]]
    scale = min(900 / physical_w, 900 / physical_h)
    output = Image.fromarray(rgb.clip(0, 255).astype('uint8')).resize(
        (max(1, round(physical_w * scale)), max(1, round(physical_h * scale))), Image.Resampling.BILINEAR)
    stream = io.BytesIO()
    output.save(stream, format='PNG')
    return Response(stream.getvalue(), mimetype='image/png')


@app.get('/api/dose_meta')
@api
def dose_meta():
    v = volume(request.args['case'], request.args['series'])
    dose, maximum = v.dose(request.args['dose'])
    return jsonify(maximum=maximum, coverage=float(np.isfinite(dose).mean()))


@app.get('/api/dvh')
@api
def dvh():
    if data_source() == 'quick':
        raise ValueError('快速查看未验证计划关联与结构完整性，不提供 DVH')
    a = request.args
    v = volume(a['case'], a['series'])
    dose, maximum = v.dose(a['dose'])
    result = []
    rois = [int(s) for s in a.get('rois', '').split(',') if s]
    if not rois or len(rois) > 8:
        raise ValueError('DVH 请选择 1 至 8 个结构')
    thresholds = np.linspace(0, maximum, 201)
    struct = v.record(a['struct'], 'RTSTRUCT')
    for number in rois:
        mask = v.mask(a['struct'], number)
        values = dose[mask]
        count = len(values)
        coverage = float(np.isfinite(values).mean()) if count else 0
        name = next(x['name'] for x in struct['rois'] if x['number'] == number)
        row = {'roi': number, 'name': name, 'volume_cc': count * np.prod(v.image.GetSpacing()) / 1000,
               'coverage': coverage}
        clipped = next(x for x in struct['rois'] if x['number'] == number).get('clipped', False)
        if clipped:
            row['error'] = '结构已裁剪到 CT 视野；仅显示剩余体积，不计算 DVH 指标'
        elif count and np.isfinite(values).all():
            values.sort()
            row.update(mean=float(values.mean()), d95=float(np.percentile(values, 5)),
                       d2=float(np.percentile(values, 98)), v20=float((values >= 20).mean() * 100),
                       curve=((count - np.searchsorted(values, thresholds, side='left')) / count * 100).tolist())
        else:
            row['error'] = '结构为空或部分位于剂量网格外，未计算指标'
        result.append(row)
    if a.get('format') == 'csv':
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['ROI', 'volume_cc', 'coverage', 'Dmean_Gy', 'D95_Gy', 'D2_Gy', 'V20_percent', 'note'])
        for row in result:
            writer.writerow([row.get(k, '') for k in ['name', 'volume_cc', 'coverage', 'mean', 'd95', 'd2', 'v20', 'error']])
        return Response('\ufeff' + stream.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment; filename="dvh_metrics.csv"'})
    return jsonify(doses=thresholds.tolist(), structures=result,
                   method='CT 网格结构栅格化；剂量线性插值；未覆盖结构不计算。研究预览，需与 TPS 核对。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--nifti-root', type=Path, default=NIFTI_ROOT)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    ROOT = args.data_root.resolve()
    NIFTI_ROOT = args.nifti_root.resolve()
    app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)
