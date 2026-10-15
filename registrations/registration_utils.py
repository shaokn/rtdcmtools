"""Shared registration geometry, resampling and REG helpers."""
import json
import numpy as np
import SimpleITK as sitk


def single(items, name):
    if len(items) != 1:
        raise ValueError(f'{name}: expected one, found {len(items)}; select explicitly')
    return items[0]

def registration_matrices(ds):
    """DICOM entry maps its frame into the REG common reference frame (LPS mm)."""
    result = {}
    if str(ds.get('SOPClassUID', '')) != '1.2.840.10008.5.1.4.1.1.66.1':
        raise ValueError('Expected DICOM Spatial Registration Storage')
    for entry in ds.RegistrationSequence:
        uid = str(entry.FrameOfReferenceUID)
        if uid in result:
            raise ValueError('Duplicate registration frame')
        registration = single(list(entry.MatrixRegistrationSequence), 'matrix registration')
        # Reject chains in v1 rather than silently guessing multiplication order.
        item = single(list(registration.MatrixSequence), 'matrix')
        if item.FrameOfReferenceTransformationMatrixType != 'RIGID':
            raise ValueError('Only rigid IGRT matrices accepted')
        matrix = np.asarray(item.FrameOfReferenceTransformationMatrix, float).reshape(4, 4)
        rotation = matrix[:3, :3]
        if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError('Invalid rigid matrix')
        result[uid] = matrix
    return result

def pull_matrix(ds, fixed_frame, moving_frame):
    matrices = registration_matrices(ds)
    # Both entries map into the same common frame: inv(M_moving) @ M_fixed.
    return np.linalg.inv(matrices[moving_frame]) @ matrices[fixed_frame]

def affine_transform(matrix):
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix[:3, :3].ravel().tolist())
    transform.SetTranslation(matrix[:3, 3].tolist())
    return transform

def ones(image):
    result = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
    result.CopyInformation(image)
    return result

def map_image(image, fixed, transform, mask=False):
    return sitk.Resample(image, fixed, transform,
                         sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,
                         0 if mask else -1000,
                         sitk.sitkUInt8 if mask else sitk.sitkFloat32)

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def work_reference(fixed, factor):
    """Preserve both endpoint voxel centers, origin, and direction."""
    old_size = np.asarray(fixed.GetSize())
    size = np.ceil((old_size - 1) / factor).astype(int) + 1
    if np.any(size < 4):
        raise ValueError('Working grid too small; reduce --downsample')
    spacing = (old_size - 1) * np.asarray(fixed.GetSpacing()) / (size - 1)
    result = sitk.Image(size.tolist(), sitk.sitkFloat32)
    result.SetSpacing(spacing.tolist())
    result.SetOrigin(fixed.GetOrigin())
    result.SetDirection(fixed.GetDirection())
    return result
