"""Read converter datasets without reopening the source DICOM files."""
from collections import OrderedDict
from functools import lru_cache
import json
import re

import numpy as np
import pydicom
import SimpleITK as sitk


@lru_cache(maxsize=512)
def catalog_kind(path, modified_ns, size):
    data = json.loads(path.read_text(encoding='utf-8'))
    return 'fraction' if data.get('ct') else 'legacy' if 'images' in data else 'collection'


def catalog(root):
    root = root.resolve()
    candidates = set(root.glob('*/index.json')) | set(root.glob('*/*/index.json'))
    if (root / 'index.json').is_file():
        candidates.add(root / 'index.json')
    result = {}
    for path in candidates:
        if any(part.startswith('.') for part in path.relative_to(root).parts):
            continue
        stat = path.stat()
        kind = catalog_kind(path, stat.st_mtime_ns, stat.st_size)
        if kind == 'fraction':
            key = f'{path.parent.parent.name}/{path.parent.name}'
        elif kind == 'legacy':
            key = path.parent.name
        else:
            continue
        if key in result:
            raise ValueError(f'Duplicate NIfTI case: {key}')
        result[key] = path
    def order(key):
        parent, _, leaf = key.rpartition('/')
        match = re.fullmatch(r'fraction_fbct(\d+)', leaf)
        return (parent, 0 if leaf == 'plan_ct1' else 1,
                int(match.group(1)) if match else -1, key)
    return dict(sorted(result.items(), key=lambda item: order(item[0])))


def path_for(root, case, relative):
    paths = catalog(root)
    if case not in paths:
        raise ValueError('NIfTI 病例不存在')
    base = paths[case].parent.resolve()
    path = (base / relative).resolve()
    if base not in path.parents:
        raise ValueError('无效 NIfTI 文件路径')
    return path


def read_json(root, case, relative='index.json'):
    data = json.loads(path_for(root, case, relative).read_text(encoding='utf-8'))
    if relative == 'index.json' and data.get('ct'):
        ct = data['ct']
        data['images'] = [dict(series_uid=ct['series_uid'], name=data['package'],
                               nifti=ct['nifti'], _metadata=ct)]
        for structure in data['structures']:
            structure.setdefault('name', structure['metadata'].split('/')[-2])
            structure.setdefault('json', structure['metadata'])
        for dose in data['doses']:
            dose.setdefault('json', dose['metadata'])
    return data


def metadata(root, case, entry):
    if '_metadata' in entry:
        return entry['_metadata']
    return read_json(root, case, entry['json'])


def header(root, case, entry):
    return pydicom.Dataset.from_json(metadata(root, case, entry).get('dicom', {}))


def info(root, case):
    index = read_json(root, case)
    series, plans, structs, doses = [], [], [], []
    for entry in index['images']:
        meta = metadata(root, case, entry)
        series.append(dict(uid=entry['series_uid'], name=entry['name'], description='', date='',
                           count=meta['geometry']['shape_xyz'][2]))
    for entry in index['structures']:
        rois = [r for r in entry['rois'] if r['status'] == 'converted']
        structs.append(dict(uid=entry['sop_uid'], name=entry['name'],
            series=sorted({r['reference_ct_series_uid'] for r in rois}),
            rois=[dict(number=r['roi_number'], name=r['roi_name'],
                       series=r['reference_ct_series_uid'],
                       clipped=r.get('clipped_to_ct_grid', False),
                       color=r.get('color_rgb') or [30, 210, 170]) for r in rois]))
    for entry in index['plans']:
        ds = header(root, case, entry)
        plans.append(dict(uid=entry['sop_uid'], name=entry['name'], structs=entry['structure_uids'],
            date=str(ds.get('RTPlanDate', '')),
            fractions=[int(g.NumberOfFractionsPlanned) for g in ds.get('FractionGroupSequence', [])
                       if g.get('NumberOfFractionsPlanned') is not None],
            beams=[dict(number=int(b.BeamNumber), name=str(b.get('BeamName', b.BeamNumber)),
                        type=str(b.get('RadiationType', ''))) for b in ds.get('BeamSequence', [])]))
    for entry in index['doses']:
        if entry['status'] != 'converted':
            continue
        doses.append(dict(uid=entry['sop_uid'], type=entry['dose_summation_type'],
                          plans=entry['plan_uids'], units=entry['dose_units'], beams=[]))
    return dict(case=case, series=series, plans=plans, structures=structs, doses=doses)


def read_volume(path):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('仅支持三维标量 NIfTI')
    if min(image.GetSize()) < 2:
        raise ValueError('三切面影像各轴至少需要两个体素')
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('NIfTI 方向矩阵无效')
    return image


class Volume:
    def __init__(self, root, case, series):
        self.root, self.case, self.series = root, case, series
        self.index = read_json(root, case)
        self.info = info(root, case)
        entry = next((e for e in self.index['images'] if e['series_uid'] == series), None)
        if entry is None:
            raise ValueError('NIfTI CT 序列不存在')
        meta = metadata(root, case, entry)
        if meta.get('intensity_units') != 'HU':
            raise ValueError('CT 未声明 HU 单位')
        self.frame = str(header(root, case, entry).get('FrameOfReferenceUID', ''))
        self.image = read_volume(path_for(root, case, entry['nifti']))
        # Current MPR labels assume LPS-positive axial CT, as in the DICOM viewer.
        if not np.allclose(np.asarray(self.image.GetDirection()).reshape(3, 3), np.eye(3), atol=1e-5):
            raise ValueError('当前三切面仅支持 LPS 正向轴位 CT；斜位 NIfTI 未隐式重采样')
        self.array = sitk.GetArrayFromImage(self.image).astype(np.float32)
        if not np.isfinite(self.array).all():
            raise ValueError('CT 包含无效数值')
        self.masks, self.doses = OrderedDict(), OrderedDict()

    def record(self, uid, modality):
        key = {'RTSTRUCT': 'structures', 'RTPLAN': 'plans', 'RTDOSE': 'doses'}[modality]
        record = next((r for r in self.info[key] if r['uid'] == uid), None)
        if record is None:
            raise ValueError(f'{modality} 对象不可用')
        return record

    def colors(self, struct):
        return {r['number']: r['color'] for r in self.record(struct, 'RTSTRUCT')['rois']}

    def mask(self, struct, roi):
        key = (struct, int(roi))
        if key in self.masks:
            self.masks.move_to_end(key)
            return self.masks[key]
        entry = next((e for e in self.index['structures'] if e['sop_uid'] == struct), None)
        if entry is None:
            raise ValueError('结构集不存在')
        r = next((r for r in entry['rois'] if r['roi_number'] == int(roi) and r['status'] == 'converted'), None)
        if r is None or r['reference_ct_series_uid'] != self.series:
            raise ValueError('mask 未引用当前 CT 或未成功转换')
        if not self.frame or r.get('frame_of_reference_uid') != self.frame:
            raise ValueError('mask 与 CT 坐标参考系不一致')
        image = read_volume(path_for(self.root, self.case, r['nifti']))
        if image.GetSize() != self.image.GetSize() or any(not np.allclose(
                getattr(image, getter)(), getattr(self.image, getter)(), atol=1e-5, rtol=0)
                for getter in ('GetOrigin', 'GetSpacing', 'GetDirection')):
            raise ValueError('mask 与 CT 网格不一致，不进行隐式重采样')
        array = sitk.GetArrayFromImage(image)
        if not np.isin(array, [0, 1]).all():
            raise ValueError('结构 mask 必须为 0/1 二值数据')
        mask = array.astype(bool)
        self.masks[key] = mask
        while len(self.masks) > 8:
            self.masks.popitem(last=False)
        return mask

    def dose(self, uid):
        if uid in self.doses:
            return self.doses[uid]
        entry = next((e for e in self.index['doses'] if e['sop_uid'] == uid and e['status'] == 'converted'), None)
        if entry is None:
            raise ValueError('该剂量没有可用 NIfTI')
        if entry.get('dose_units') != 'Gy' or entry.get('dose_grid_scaling_applied') is not True:
            raise ValueError('剂量未声明已换算为 Gy，拒绝猜测缩放')
        frame = str(header(self.root, self.case, entry).get('FrameOfReferenceUID', ''))
        if not frame or frame != self.frame:
            raise ValueError('剂量与 CT 的 FrameOfReferenceUID 不一致')
        plans = [self.record(uid, 'RTPLAN') for uid in entry['plan_uids']]
        structs = [self.record(uid, 'RTSTRUCT') for p in plans for uid in p['structs']]
        if not any(self.series in s['series'] for s in structs):
            raise ValueError('剂量未通过计划和结构关联到当前 CT')
        image = read_volume(path_for(self.root, self.case, entry['nifti']))
        array = sitk.GetArrayFromImage(image)
        if not np.isfinite(array).all():
            raise ValueError('剂量包含无效数值')
        maximum = float(array.max())
        mapped = sitk.Resample(image, self.image, sitk.Transform(), sitk.sitkLinear,
                               float('nan'), sitk.sitkFloat32)
        self.doses.clear()
        self.doses[uid] = (sitk.GetArrayFromImage(mapped), maximum)
        return self.doses[uid]
