#!/usr/bin/env python3
"""Convert indexed organized DICOM exports to NIfTI and per-object JSON."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pydicom
from rt_utils import image_helper

VERSION = '1.0.2'
LPS_TO_RAS = np.diag([-1., -1., 1., 1.])
COMMON = ['SOPClassUID', 'SOPInstanceUID', 'Modality', 'StudyInstanceUID',
          'SeriesInstanceUID', 'FrameOfReferenceUID', 'SeriesDescription']
CT_TAGS = COMMON + ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing',
    'SliceThickness', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'RescaleType',
    'AcquisitionDateTime', 'AcquisitionDate', 'AcquisitionTime', 'SeriesDate', 'SeriesTime',
    'KVP', 'ConvolutionKernel', 'ReconstructionDiameter', 'ProtocolName', 'PatientPosition']
RS_TAGS = COMMON + ['StructureSetLabel', 'StructureSetName', 'StructureSetDate',
    'ReferencedFrameOfReferenceSequence', 'StructureSetROISequence', 'ROIContourSequence',
    'RTROIObservationsSequence']
RP_TAGS = COMMON + ['RTPlanLabel', 'RTPlanName', 'RTPlanDate', 'RTPlanTime', 'RTPlanGeometry',
    'ReferencedStructureSetSequence', 'ReferencedRTPlanSequence', 'DoseReferenceSequence',
    'FractionGroupSequence', 'BeamSequence', 'PatientSetupSequence', 'ToleranceTableSequence',
    'ApprovalStatus', 'ApplicationSetupSequence']
RD_TAGS = COMMON + ['DoseUnits', 'DoseType', 'DoseSummationType', 'DoseGridScaling',
    'ReferencedRTPlanSequence', 'ReferencedInstanceSequence', 'DVHSequence',
    'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'GridFrameOffsetVector',
    'Rows', 'Columns', 'NumberOfFrames', 'SliceThickness', 'TissueHeterogeneityCorrection']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(text):
    name = re.sub(r'[^\w.-]+', '_', str(text)).strip('._')[:100] or 'unnamed'
    if re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', name.split('.')[0], re.I):
        name = '_' + name
    return name


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def dicom_metadata(ds, keywords):
    selected = pydicom.Dataset()
    for keyword in keywords:
        if keyword in ds:
            selected.add(ds.data_element(keyword))
    return selected.to_json_dict()


def reference_uids(ds, sequence):
    return [str(x.ReferencedSOPInstanceUID) for x in ds.get(sequence, [])
            if x.get('ReferencedSOPInstanceUID')]


def source_path(patient, record):
    path = (patient / record['path']).resolve()
    if patient.resolve() not in path.parents:
        raise ValueError('Indexed DICOM path escapes patient folder')
    return path


def basis(ds):
    o = np.asarray(ds.ImageOrientationPatient, dtype=float)
    if o.shape != (6,) or not np.isfinite(o).all():
        raise ValueError('Invalid ImageOrientationPatient')
    matrix = np.column_stack([o[:3], o[3:], np.cross(o[:3], o[3:])])
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError('Image directions are not orthonormal')
    spacing = np.asarray(ds.PixelSpacing, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid PixelSpacing')
    return matrix


def make_affine(ds, origin, z_step):
    matrix = basis(ds)
    affine = np.eye(4)
    affine[:3, 0] = matrix[:, 0] * float(ds.PixelSpacing[1])
    affine[:3, 1] = matrix[:, 1] * float(ds.PixelSpacing[0])
    affine[:3, 2] = matrix[:, 2] * z_step
    affine[:3, 3] = origin
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise ValueError('Invalid image affine')
    return LPS_TO_RAS @ affine


def ct_volume(datasets):
    if len(datasets) < 2:
        raise ValueError('At least two single-frame CT slices are required')
    normal = basis(datasets[0])[:, 2]
    data = sorted(datasets, key=lambda d: np.dot(np.asarray(d.ImagePositionPatient, float), normal))
    first = data[0]
    for ds in data:
        if (str(ds.SeriesInstanceUID) != str(first.SeriesInstanceUID)
                or str(ds.get('FrameOfReferenceUID', '')) != str(first.get('FrameOfReferenceUID', ''))
                or ds.Rows != first.Rows or ds.Columns != first.Columns
                or not np.allclose(ds.PixelSpacing, first.PixelSpacing)
                or not np.allclose(ds.ImageOrientationPatient, first.ImageOrientationPatient, atol=1e-5)
                or int(ds.get('NumberOfFrames', 1)) != 1):
            raise ValueError('Mixed CT series or inconsistent slice geometry')
        if ds.get('ModalityLUTSequence') or ds.get('RescaleType', 'HU') not in ('HU', ''):
            raise ValueError('CT intensity mapping is not supported as HU')
    positions = np.asarray([ds.ImagePositionPatient for ds in data], float)
    dz = float(np.median(np.diff(positions @ normal)))
    if dz <= 0 or not np.allclose(np.diff(positions, axis=0), normal * dz, atol=.02):
        raise ValueError('Nonuniform, duplicate or sheared CT slice positions; conversion stopped')
    slices = []
    for ds in data:
        slope, intercept = float(ds.get('RescaleSlope', 1)), float(ds.get('RescaleIntercept', 0))
        if not np.isfinite([slope, intercept]).all():
            raise ValueError('Invalid CT rescale mapping')
        slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)
    # pydicom is row/column; NIfTI axes here are column/row/slice.
    array = np.stack(slices, axis=2).transpose(1, 0, 2)
    affine = make_affine(first, positions[0], dz)
    return array, affine, data


def dose_volume(ds):
    if ds.get('DoseUnits') != 'GY':
        raise ValueError('DoseUnits is not GY; relative dose needs an explicit absolute reference')
    scaling = float(ds.DoseGridScaling)
    if not np.isfinite(scaling) or scaling <= 0:
        raise ValueError('Invalid DoseGridScaling')
    array = ds.pixel_array
    if array.ndim == 2:
        raise ValueError('Single-frame dose has no verified inter-plane spacing')
    offsets = np.asarray(ds.GridFrameOffsetVector, float)
    if len(offsets) != array.shape[0] or len(offsets) < 2 or not np.isfinite(offsets).all():
        raise ValueError('Invalid dose frame positions')
    origin = np.asarray(ds.ImagePositionPatient, float)
    if np.isclose(offsets[0], 0, atol=1e-6):
        interpretation = 'relative_to_ImagePositionPatient'
    elif (np.allclose(ds.ImageOrientationPatient, [1, 0, 0, 0, 1, 0])
          and np.isclose(offsets[0], origin[2], atol=1e-4)):
        offsets = offsets - origin[2]
        interpretation = 'absolute_patient_z'
    else:
        raise ValueError('Unsupported GridFrameOffsetVector origin convention')
    dz = float(np.median(np.diff(offsets)))
    if abs(dz) < 1e-8 or not np.allclose(np.diff(offsets), dz, atol=.001):
        raise ValueError('Nonuniform dose frame spacing cannot be represented by one NIfTI affine')
    values = (array.astype(np.float32) * scaling).transpose(2, 1, 0)
    return values, make_affine(ds, origin, dz), interpretation


def write_nifti(path, array, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units('mm')
    image.header.set_slope_inter(1., 0.)
    nib.save(image, path)
    loaded = nib.load(path)
    if not np.allclose(loaded.affine, affine, atol=1e-4, rtol=1e-7):
        raise ValueError('NIfTI affine round-trip failed')
    if not np.array_equal(np.asanyarray(loaded.dataobj), array):
        raise ValueError('NIfTI voxel round-trip failed')
    return {'shape_xyz': list(array.shape), 'dtype': str(array.dtype), 'coordinate_system': 'RAS',
            'affine_ras_mm': affine.tolist(), 'spacing_mm': nib.affines.voxel_sizes(affine).tolist(),
            'axis_codes': list(nib.aff2axcodes(affine)), 'qform_code': 1, 'sform_code': 1,
            'nifti_scaling': {'slope': 1., 'intercept': 0.},
            'range': [float(array.min()), float(array.max())], 'round_trip_verified': True}


def rasterize_roi(ct, contour_sequence, frame_uid, clip_rois=False):
    datasets, affine, shape = ct['datasets'], ct['affine'], ct['shape']
    first = datasets[0]
    if frame_uid != str(first.get('FrameOfReferenceUID', '')):
        raise ValueError('ROI and CT FrameOfReferenceUID differ')
    if first.Rows != first.Columns or not np.isclose(*first.PixelSpacing):
        raise ValueError('rt-utils 1.2.7 rasterization requires square slices and square in-plane pixels')
    types = {str(c.ContourGeometricType) for c in contour_sequence}
    if not types <= {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'} or len(types) > 1:
        raise ValueError('Unsupported or mixed volumetric contour types')
    lookup = {str(ds.SOPInstanceUID): k for k, ds in enumerate(datasets)}
    inverse_lps = np.linalg.inv(LPS_TO_RAS @ affine)
    outside_points = 0
    for contour in contour_sequence:
        points = np.asarray(contour.ContourData, float).reshape(-1, 3)
        if len(points) < 3 or len(points) != int(contour.NumberOfContourPoints):
            raise ValueError('Invalid closed contour point count')
        refs = reference_uids(contour, 'ContourImageSequence')
        if len(refs) != 1 or refs[0] not in lookup:
            raise ValueError('Contour does not uniquely reference a loaded CT slice')
        indices = nib.affines.apply_affine(inverse_lps, points)
        if not np.isfinite(indices).all() or not np.allclose(indices[:, 2], lookup[refs[0]], atol=.05):
            raise ValueError('Contour plane disagrees with its referenced CT slice')
        outside_points += int(np.count_nonzero(np.any(
            (indices[:, :2] < -.5) | (indices[:, :2] > np.array(shape[:2]) - .5), axis=1)))
    if outside_points and not clip_rois:
        raise ValueError('Contour exceeds CT field of view; use --clip-rois-to-ct to explicitly permit clipping')
    mask = image_helper.create_series_mask_from_contour_sequence(datasets, contour_sequence)
    # rt-utils uses OpenCV's row/column array layout despite its square buffer naming.
    mask = mask.transpose(1, 0, 2).astype(np.uint8)
    if mask.shape != shape:
        raise ValueError('Mask shape does not match reference CT')
    return mask, {'clipped_to_ct_grid': bool(outside_points),
                  'contour_points_outside_ct_fov': outside_points}


def convert_patient(patient, output_root, include_beam=False, list_only=False, clip_rois=False):
    patient = patient.resolve()
    destination = (output_root / patient.name).resolve()
    if patient == destination or patient in destination.parents or destination in patient.parents:
        raise ValueError('Input and output must be separate directory trees')
    manifest = json.loads((patient / 'relationships.json').read_text(encoding='utf-8'))
    records = manifest['files']
    supported = [r for r in records if r['modality'] in ('CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE')]
    if list_only:
        return {'patient': patient.name, 'status': 'preview', 'indexed_files': len(supported),
                'output': str(destination), 'dose_selection': 'PLAN+BEAM' if include_beam else 'PLAN'}
    sources = []
    by_uid = {}
    for r in supported:
        digest = sha256(source_path(patient, r))
        if r.get('sha256') and digest != r['sha256']:
            raise ValueError(f'Original index hash mismatch: {r["path"]}')
        sources.append({'path': r['path'], 'sop_uid': r['sop_uid'], 'sha256': digest})
        if r['sop_uid'] in by_uid:
            if by_uid[r['sop_uid']]['sha256'] != digest:
                raise ValueError('Duplicate SOPInstanceUID with different content')
        else:
            by_uid[r['sop_uid']] = {**r, 'sha256': digest}
    options = {'include_beam': include_beam, 'clip_rois_to_ct': clip_rois,
               'resampling': 'none', 'dose_units': 'Gy'}
    signature = hashlib.sha256(json.dumps({'version': VERSION, 'sources': sources,
        'manifest_sha256': sha256(patient / 'relationships.json'), 'options': options}, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        index = json.loads((destination / 'index.json').read_text(encoding='utf-8'))
        if index.get('signature') != signature:
            raise ValueError('Output exists with different input/options; choose a new --output-root')
        for r in index['outputs']:
            if sha256(destination / r['path']) != r['sha256']:
                raise ValueError(f'Output modified or corrupt: {r["path"]}')
        return {'patient': patient.name, 'status': 'reused', 'summary': index['summary'], 'output': str(destination)}
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nifti-stage-', dir=output_root) as temp:
        stage = Path(temp) / patient.name
        stage.mkdir()
        index = {'schema_version': 1, 'converter_version': VERSION, 'patient_folder': patient.name,
                 'signature': signature, 'options': options, 'sources': sources,
                 'software': {p: version(p) for p in ('numpy', 'nibabel', 'pydicom', 'rt-utils')},
                 'images': [], 'structures': [], 'plans': [], 'doses': [], 'issues': []}
        headers = {uid: pydicom.dcmread(source_path(patient, r), stop_before_pixels=True)
                   for uid, r in by_uid.items()}
        for uid, ds in headers.items():
            if str(ds.SOPInstanceUID) != uid or str(ds.Modality) != by_uid[uid]['modality']:
                raise ValueError('DICOM header does not match organized index')
        groups = defaultdict(list)
        for r in by_uid.values():
            if r['modality'] == 'CT':
                ds = headers[r['sop_uid']]
                if str(ds.SeriesInstanceUID) != r['series_uid']:
                    raise ValueError('CT series UID disagrees with index')
                groups[r['series_uid']].append(r)
        ct_lookup = {}
        used = set()

        def name_unique(label, uid):
            name = safe(label)
            if name.casefold() in used:
                name += '_' + hashlib.sha256(uid.encode()).hexdigest()[:10]
            if name.casefold() in used:
                raise ValueError('Output name collision')
            used.add(name.casefold())
            return name

        for series, group in sorted(groups.items()):
            name = name_unique(Path(group[0]['folder']).name, series)
            datasets = [pydicom.dcmread(source_path(patient, r)) for r in group]
            arr, affine, ordered = ct_volume(datasets)
            image_rel, json_rel = f'images/{name}.nii.gz', f'images/{name}.json'
            geometry = write_nifti(stage / image_rel, arr, affine)
            slice_info = [{'sop_uid': str(ds.SOPInstanceUID),
                           'position_lps_mm': [float(x) for x in ds.ImagePositionPatient],
                           'slope': float(ds.get('RescaleSlope', 1)),
                           'intercept': float(ds.get('RescaleIntercept', 0))} for ds in ordered]
            info = {'kind': 'CT', 'name': name, 'series_uid': series, 'nifti': image_rel,
                    'intensity_units': 'HU', 'rescale_applied': True, 'resampled': False,
                    'geometry': geometry, 'slices_in_output_order': slice_info,
                    'dicom': dicom_metadata(ordered[0], CT_TAGS)}
            json_write(stage / json_rel, info)
            index['images'].append({'series_uid': series, 'name': name, 'nifti': image_rel, 'json': json_rel})
            ct_lookup[series] = {'name': name, 'datasets': ordered, 'affine': affine,
                                 'shape': arr.shape, 'nifti': image_rel}
            del arr

        used.clear()
        for uid, r in by_uid.items():
            if r['modality'] != 'RTSTRUCT':
                continue
            ds = headers[uid]
            name = name_unique(Path(r['folder']).name, uid)
            root = f'structures/{name}'
            rs = {'kind': 'RTSTRUCT', 'sop_uid': uid, 'source_path': r['path'],
                  'dicom': dicom_metadata(ds, RS_TAGS), 'rois': []}
            contours = {int(c.ReferencedROINumber): c for c in ds.get('ROIContourSequence', [])}
            roi_used = {name.casefold()}
            sop_to_series = {str(d.SOPInstanceUID): s for s, ct in ct_lookup.items() for d in ct['datasets']}
            for roi in ds.StructureSetROISequence:
                number, roi_name = int(roi.ROINumber), str(roi.ROIName)
                filename = safe(roi_name)
                if filename.casefold() in roi_used:
                    filename += f'_ROI{number}'
                if filename.casefold() in roi_used:
                    raise ValueError('ROI filename collision')
                roi_used.add(filename.casefold())
                entry = {'roi_number': number, 'roi_name': roi_name, 'source_struct_uid': uid,
                         'json': f'{root}/{filename}.json', 'mask_label': 1, 'background_label': 0}
                contour = contours.get(number)
                seq = contour.get('ContourSequence', []) if contour is not None else []
                types = sorted({str(c.ContourGeometricType) for c in seq})
                entry['contour_types'] = types
                entry['color_rgb'] = [int(x) for x in contour.get('ROIDisplayColor', [])] if contour is not None else []
                refs = {ref for c in seq for ref in reference_uids(c, 'ContourImageSequence')}
                entry['referenced_ct_sop_uids'] = sorted(refs)
                entry['frame_of_reference_uid'] = str(roi.get('ReferencedFrameOfReferenceUID', ''))
                if not seq or not set(types) <= {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'}:
                    entry.update(status='metadata_only', reason='Empty or non-volumetric contours; not a 3D mask')
                else:
                    try:
                        candidates = {sop_to_series[x] for x in refs if x in sop_to_series}
                        if len(candidates) != 1 or not refs or any(x not in sop_to_series for x in refs):
                            raise ValueError('ROI does not uniquely reference one available CT series')
                        series = next(iter(candidates))
                        ct = ct_lookup[series]
                        mask, clipping = rasterize_roi(ct, seq, entry['frame_of_reference_uid'], clip_rois)
                        if not mask.any():
                            raise ValueError('Closed contours produced an empty mask')
                        mask_rel = f'{root}/{filename}.nii.gz'
                        geometry = write_nifti(stage / mask_rel, mask, ct['affine'])
                        entry.update(status='converted', nifti=mask_rel, reference_ct_series_uid=series,
                                     **clipping,
                                     reference_ct_nifti=ct['nifti'], geometry=geometry,
                                     voxel_count=int(mask.sum()),
                                     volume_cc=float(mask.sum() * abs(np.linalg.det(ct['affine'][:3, :3])) / 1000),
                                     rasterization='rt-utils 1.2.7 / OpenCV fillPoly; CT voxel grid; no interpolation')
                    except Exception as exc:
                        entry.update(status='failed', reason=str(exc))
                        index['issues'].append({'kind': 'ROI', 'uid': uid, 'roi': number, 'error': str(exc)})
                json_write(stage / entry['json'], entry)
                rs['rois'].append(entry)
            json_write(stage / f'{root}/{name}.json', rs)
            index['structures'].append({'sop_uid': uid, 'name': name, 'json': f'{root}/{name}.json',
                                        'rois': rs['rois']})

        used.clear()
        plan_names = {}
        for uid, r in by_uid.items():
            if r['modality'] != 'RTPLAN':
                continue
            ds = headers[uid]
            name = name_unique(str(ds.get('RTPlanLabel') or ds.get('RTPlanName') or 'Plan'), uid)
            plan_names[uid] = name
            refs = reference_uids(ds, 'ReferencedStructureSetSequence')
            rel = f'plans/{name}.json'
            dcm_rel = f'plans/{name}.dcm'
            (stage / dcm_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path(patient, r), stage / dcm_rel)
            if sha256(stage / dcm_rel) != r['sha256']:
                raise ValueError('RTPLAN copy hash mismatch')
            json_write(stage / rel, {'kind': 'RTPLAN', 'sop_uid': uid, 'name': name,
                'dicom_file': dcm_rel, 'dicom_sha256': r['sha256'],
                'source_path': r['path'], 'structure_uids': refs, 'dicom': dicom_metadata(ds, RP_TAGS)})
            index['plans'].append({'sop_uid': uid, 'name': name, 'json': rel, 'structure_uids': refs,
                                   'dicom_file': dcm_rel, 'dicom_sha256': r['sha256']})
            for ref in refs:
                if ref not in headers or headers[ref].Modality != 'RTSTRUCT':
                    index['issues'].append({'kind': 'reference', 'uid': uid, 'missing_struct': ref})

        used.clear()
        for uid, r in by_uid.items():
            if r['modality'] != 'RTDOSE':
                continue
            ds = headers[uid]
            refs = reference_uids(ds, 'ReferencedRTPlanSequence')
            summation = str(ds.get('DoseSummationType', 'UNKNOWN'))
            name = plan_names.get(refs[0], 'UNMATCHED') if len(refs) == 1 else 'MULTI_PLAN'
            label = name + '_' + summation
            beams = [str(b.ReferencedBeamNumber) for p in ds.get('ReferencedRTPlanSequence', [])
                     for g in p.get('ReferencedFractionGroupSequence', [])
                     for b in g.get('ReferencedBeamSequence', [])]
            if beams:
                label += '_beam' + '-'.join(beams)
            label = name_unique(label, uid)
            meta = {'kind': 'RTDOSE', 'sop_uid': uid, 'source_path': r['path'], 'plan_uids': refs,
                    'dose_summation_type': summation, 'json': f'dose/{label}.json',
                    'source_dose_units': str(ds.get('DoseUnits', '')),
                    'source_dose_grid_scaling': float(ds.get('DoseGridScaling', 1)),
                    'dicom': dicom_metadata(ds, RD_TAGS)}
            if summation != 'PLAN' and not (include_beam and summation == 'BEAM'):
                meta.update(status='metadata_only', reason='Dose summation type not selected for NIfTI')
            else:
                try:
                    full = pydicom.dcmread(source_path(patient, r))
                    arr, affine, convention = dose_volume(full)
                    rel = f'dose/{label}.nii.gz'
                    geometry = write_nifti(stage / rel, arr, affine)
                    meta.update(status='converted', nifti=rel, dose_units='Gy',
                                dose_grid_scaling_applied=True, resampled=False,
                                grid_frame_offset_interpretation=convention, geometry=geometry)
                except Exception as exc:
                    meta.update(status='failed', reason=str(exc))
                    index['issues'].append({'kind': 'RTDOSE', 'uid': uid, 'error': str(exc)})
            for ref in refs:
                if ref not in plan_names:
                    index['issues'].append({'kind': 'reference', 'uid': uid, 'missing_plan': ref})
            json_write(stage / meta['json'], meta)
            index['doses'].append({k: v for k, v in meta.items() if k != 'dicom'})

        index['summary'] = {'ct': len(index['images']), 'rs': len(index['structures']),
            'masks': sum(r['status'] == 'converted' for s in index['structures'] for r in s['rois']),
            'nonvolume_rois': sum(r['status'] == 'metadata_only' for s in index['structures'] for r in s['rois']),
            'clipped_rois': sum(r.get('clipped_to_ct_grid', False) for s in index['structures'] for r in s['rois']),
            'plans': len(index['plans']), 'dose_nifti': sum(d['status'] == 'converted' for d in index['doses']),
            'dose_metadata_only': sum(d['status'] == 'metadata_only' for d in index['doses']),
            'issues': len(index['issues'])}
        index['outputs'] = [{'path': str(p.relative_to(stage)).replace('\\', '/'), 'sha256': sha256(p)}
                            for p in sorted(stage.rglob('*')) if p.is_file()]
        index['status'] = 'partial' if index['issues'] else 'ok'
        json_write(stage / 'index.json', index)
        if destination.exists():
            raise ValueError('Destination appeared during conversion; refusing overwrite')
        stage.rename(destination)
        return {'patient': patient.name, 'status': index['status'], 'summary': index['summary'],
                'output': str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Organized patient folder or its parent collection')
    parser.add_argument('--mode', choices=['auto', 'patient', 'batch'], default='auto')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent.parent / 'nifti_data')
    parser.add_argument('--include-beam', action='store_true', help='Also convert BEAM dose to NIfTI')
    parser.add_argument('--clip-rois-to-ct', action='store_true',
                        help='Allow ROI clipping at CT field of view; record clipping in each ROI JSON')
    parser.add_argument('--list-only', action='store_true', help='Preview indexed patients without writing')
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Input and output roots must be separate')
    mode = args.mode
    if mode == 'auto':
        mode = 'patient' if (source / 'relationships.json').is_file() else 'batch'
    patients = [source] if mode == 'patient' else sorted(p for p in source.iterdir()
                    if p.is_dir() and not p.name.startswith('.'))
    if not patients:
        parser.error('No patient folders')
    results = []
    for i, patient in enumerate(patients, 1):
        print(f'[{i}/{len(patients)}] {patient.name}', flush=True)
        try:
            result = convert_patient(patient, output, args.include_beam, args.list_only, args.clip_rois_to_ct)
        except Exception as exc:
            result = {'patient': patient.name, 'status': 'failed', 'error': str(exc)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if not args.list_only:
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        report = output / f'conversion_summary_{stamp}.json'
        json_write(report, {'converter_version': VERSION, 'results': results})
        print(f'Report: {report}')
    return int(any(r['status'] in ('failed', 'partial') or r.get('summary', {}).get('issues', 0) for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
