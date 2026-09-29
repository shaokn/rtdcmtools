"""Temporary, explicitly selected NIfTI overlays; no inferred RT relationships."""
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

COLORS = [[238, 70, 90], [20, 195, 185], [155, 220, 65], [245, 190, 30],
          [125, 100, 245], [240, 110, 200], [55, 155, 245]]
MAX_VOXELS = 64_000_000


def read_upload(upload, folder, number):
    name = upload.filename.replace('\\', '/').split('/')[-1]
    if not name.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError('仅接受 .nii 或 .nii.gz 文件')
    path = folder / (str(number) + ('.nii.gz' if name.lower().endswith('.gz') else '.nii'))
    upload.save(path)
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
            raise ValueError('仅支持三维标量 NIfTI，不支持 4D 或向量图像')
        size = reader.GetSize()
        if min(size) < 2 or int(np.prod(size, dtype=np.int64)) > MAX_VOXELS:
            raise ValueError('图像各轴至少 2 个体素，单文件最多 6400 万体素')
        image = reader.Execute()
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        if not np.isfinite(image.GetOrigin()).all() or not np.isfinite(image.GetSpacing()).all():
            raise ValueError('NIfTI 空间坐标无效')
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError('NIfTI 方向矩阵无效')
        return image, name
    except RuntimeError as exc:
        raise ValueError(f'无法读取 NIfTI：{name}') from exc


def axial(image):
    # Axis permutations/flips preserve voxels. Oblique images are not resampled here.
    image = sitk.DICOMOrient(image, 'LPS')
    if not np.allclose(np.asarray(image.GetDirection()).reshape(3, 3), np.eye(3), atol=1e-5):
        raise ValueError('暂不支持斜位 CT/mask；请先在专用工具中重采样到轴位')
    return image


class Volume:
    def __init__(self, files, form, temp_root):
        ct_files = files.getlist('ct')
        masks = [f for f in files.getlist('masks') if f.filename]
        doses = [f for f in files.getlist('dose') if f.filename]
        if len(ct_files) != 1 or not ct_files[0].filename:
            raise ValueError('请选择一份 CT NIfTI')
        if len(masks) > 12 or len(doses) > 1:
            raise ValueError('最多添加 12 份 mask 和 1 份剂量')
        if (masks or doses) and form.get('same_space') != 'yes':
            raise ValueError('请确认叠加文件属于同一患者、同一物理坐标系')
        if doses and form.get('dose_units') not in ('Gy', 'cGy'):
            raise ValueError('请明确选择剂量原始单位 Gy 或 cGy')
        self.maps, self.rois, self.dose_data = [], [], None
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='viewer-quick-', dir=temp_root) as tmp:
            folder = Path(tmp)
            image, self.name = read_upload(ct_files[0], folder, 0)
            self.image = axial(sitk.Cast(image, sitk.sitkFloat32))
            self.array = sitk.GetArrayFromImage(self.image)
            if not np.isfinite(self.array).all():
                raise ValueError('CT 含 NaN/Inf')
            bytes_used = self.array.nbytes * 2
            for i, upload in enumerate(masks, 1):
                image, name = read_upload(upload, folder, i)
                image = axial(image)
                if image.GetSize() != self.image.GetSize() or any(not np.allclose(
                        getattr(image, method)(), getattr(self.image, method)(), atol=1e-4, rtol=0)
                        for method in ('GetOrigin', 'GetSpacing', 'GetDirection')):
                    raise ValueError(f'mask 与 CT 网格不一致：{name}；未自动重采样')
                array = sitk.GetArrayFromImage(image)
                labels = np.unique(array)
                if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels != np.floor(labels)):
                    raise ValueError(f'mask 必须是非负整数标签图：{name}')
                labels = labels[labels > 0]
                if not len(labels) or len(labels) + len(self.rois) > 128:
                    raise ValueError('mask 不能为空，合计最多支持 128 个非零标签')
                bytes_used += array.nbytes
                if bytes_used > 512 * 1024 ** 2:
                    raise ValueError('本次解压后数据过大，请减少 mask')
                self.maps.append(array)
                for label in labels:
                    number = len(self.rois) + 1
                    self.rois.append(dict(number=number, name=name + (f' · 标签 {int(label)}' if len(labels) > 1 else ''),
                        color=COLORS[(number - 1) % len(COLORS)], map_index=i - 1, label=int(label)))
            if doses:
                image, dose_name = read_upload(doses[0], folder, 99)
                image = sitk.Cast(image, sitk.sitkFloat32)
                array = sitk.GetArrayFromImage(image)
                if not np.isfinite(array).all() or np.any(array < 0):
                    raise ValueError('剂量必须为有限非负数值')
                factor = .01 if form['dose_units'] == 'cGy' else 1.
                maximum = float(array.max()) * factor
                mapped = sitk.Resample(image, self.image, sitk.Transform(), sitk.sitkLinear,
                                       float('nan'), sitk.sitkFloat32)
                values = sitk.GetArrayFromImage(mapped) * factor
                if not np.isfinite(values).any():
                    raise ValueError('剂量与 CT 没有可显示的空间重叠')
                if bytes_used + values.nbytes > 512 * 1024 ** 2:
                    raise ValueError('本次解压后数据过大，请减少 mask')
                self.dose_data = values, maximum
        structures = [dict(uid='masks', name='手动选择', series=['ct'], rois=self.rois)] if self.rois else []
        self.info = dict(case='', title=self.name, series=[dict(uid='ct', name=self.name,
            count=self.image.GetSize()[2], date='', description='')], plans=[], structures=structures,
            doses=[dict(uid='dose', type=dose_name, plans=[], units='Gy', beams=[])] if doses else [])

    def record(self, uid, modality):
        if modality == 'RTSTRUCT' and uid == 'masks' and self.rois:
            return self.info['structures'][0]
        raise ValueError('快速查看对象不存在')

    def colors(self, struct):
        return {r['number']: r['color'] for r in self.record(struct, 'RTSTRUCT')['rois']}

    def mask(self, struct, roi):
        entry = next((r for r in self.record(struct, 'RTSTRUCT')['rois'] if r['number'] == int(roi)), None)
        if entry is None:
            raise ValueError('mask 标签不存在')
        return self.maps[entry['map_index']] == entry['label']

    def dose(self, uid):
        if uid != 'dose' or self.dose_data is None:
            raise ValueError('未加载剂量')
        return self.dose_data
