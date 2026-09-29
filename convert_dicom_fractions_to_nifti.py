#!/usr/bin/env python3
"""Convert plan_ct1/fraction_fbctX DICOM packages to matching NIfTI packages.

CT and binary ROI masks use the CT grid. RTDOSE uses its native grid in Gy.
RTPLAN and REG are exported as JSON metadata, not image volumes.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pydicom

import convert_dicom_to_nifti as base


VERSION = '1.0.2'
PACKAGE_ORDER = {'plan_ct1': 0}
MODALITY_FOLDERS = {'CT': 'CT', 'RTSTRUCT': 'RS', 'RTPLAN': 'RP', 'RTDOSE': 'RD', 'REG': 'REG'}


def package_sort_key(path):
    name = path.name
    if name in PACKAGE_ORDER:
        return PACKAGE_ORDER[name], 0
    match = __import__('re').fullmatch(r'fraction_fbct(\d+)', name)
    return (1, int(match.group(1))) if match else (2, name)


def read_dicom_files(package):
    records = []
    for folder, expected in MODALITY_FOLDERS.items():
        directory = package / expected
        for path in sorted(directory.glob('*.dcm')):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.get('Modality', '')) != folder:
                raise ValueError(f'{path}: expected {folder}, got {ds.get("Modality", "")!r}')
            if not ds.get('SOPInstanceUID'):
                raise ValueError(f'{path}: missing SOPInstanceUID')
            records.append({'path': path, 'relative_path': str(path.relative_to(package)),
                            'modality': folder, 'sop_uid': str(ds.SOPInstanceUID), 'header': ds})
    if not records:
        raise ValueError(f'No DICOM files in {package}')
    if len({record['sop_uid'] for record in records}) != len(records):
        raise ValueError(f'Duplicate SOPInstanceUID in {package}')
    return records


def unique_name(label, uid, used):
    name = base.safe(label)
    if name.casefold() in used:
        name += '_' + hashlib.sha256(uid.encode()).hexdigest()[:10]
    if name.casefold() in used:
        raise ValueError(f'Output name collision: {name}')
    used.add(name.casefold())
    return name


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def references(ds, sequence):
    return [str(item.ReferencedSOPInstanceUID) for item in ds.get(sequence, [])
            if item.get('ReferencedSOPInstanceUID')]


def struct_series(ds):
    values = []
    for frame in ds.get('ReferencedFrameOfReferenceSequence', []):
        for study in frame.get('RTReferencedStudySequence', []):
            for series in study.get('RTReferencedSeriesSequence', []):
                if series.get('SeriesInstanceUID'):
                    values.append(str(series.SeriesInstanceUID))
    return sorted(set(values))


def registration_frames(ds):
    return [str(item.FrameOfReferenceUID) for item in ds.get('RegistrationSequence', [])
            if item.get('FrameOfReferenceUID')]


def dose_name(ds, uid, used):
    summation = str(ds.get('DoseSummationType', 'UNKNOWN'))
    beam_numbers = [str(beam.ReferencedBeamNumber)
                    for plan in ds.get('ReferencedRTPlanSequence', [])
                    for group in plan.get('ReferencedFractionGroupSequence', [])
                    for beam in group.get('ReferencedBeamSequence', [])]
    label = summation + (('_beam' + '-'.join(beam_numbers)) if beam_numbers else '')
    return unique_name(label, uid, used)


def convert_package(package, stage, clip_rois_to_ct=False):
    records = read_dicom_files(package)
    by_modality = {modality: [r for r in records if r['modality'] == modality]
                   for modality in MODALITY_FOLDERS}
    summary = Counter(r['modality'] for r in records)
    index = {'schema_version': 1, 'converter_version': VERSION, 'package': package.name,
             'source_package': str(package), 'summary': dict(summary), 'ct': None,
             'structures': [], 'plans': [], 'doses': [], 'registrations': [], 'issues': []}

    if len(by_modality['CT']) < 2:
        raise ValueError(f'{package.name}: at least two CT slices required')
    ct_series = {str(r['header'].get('SeriesInstanceUID', '')) for r in by_modality['CT']}
    if len(ct_series) != 1 or '' in ct_series:
        raise ValueError(f'{package.name}: exactly one CT series required')
    ct_datasets = [pydicom.dcmread(r['path']) for r in by_modality['CT']]
    ct_array, ct_affine, ordered_ct = base.ct_volume(ct_datasets)
    ct_geometry = base.write_nifti(stage / 'CT' / 'ct.nii.gz', ct_array, ct_affine)
    ct_series_uid = next(iter(ct_series))
    ct = {'datasets': ordered_ct, 'affine': ct_affine, 'shape': ct_array.shape,
          'series_uid': ct_series_uid, 'nifti': 'CT/ct.nii.gz'}
    index['ct'] = {'series_uid': ct_series_uid, 'nifti': ct['nifti'], 'intensity_units': 'HU',
                   'rescale_applied': True, 'resampled': False, 'geometry': ct_geometry,
                   'slices_in_output_order': [str(ds.SOPInstanceUID) for ds in ordered_ct],
                   'dicom': base.dicom_metadata(ordered_ct[0], base.CT_TAGS)}
    del ct_array

    rs_uids = set()
    for number, record in enumerate(by_modality['RTSTRUCT'], 1):
        ds = record['header']
        uid = record['sop_uid']
        rs_uids.add(uid)
        root = Path('RS') / f'RS{number}'
        item = {'sop_uid': uid, 'source_dicom_path': record['relative_path'],
                'metadata': str(root / 'metadata.json'),
                'referenced_ct_series_uids': struct_series(ds),
                'rois': []}
        if set(item['referenced_ct_series_uids']) != {ct_series_uid}:
            index['issues'].append({'kind': 'RTSTRUCT', 'uid': uid,
                                    'error': 'Referenced CT series differs from package CT'})
        contours = {int(c.ReferencedROINumber): c for c in ds.get('ROIContourSequence', [])}
        used = set()
        for roi in ds.get('StructureSetROISequence', []):
            roi_number, roi_name = int(roi.ROINumber), str(roi.ROIName)
            filename = unique_name(roi_name, f'{uid}:{roi_number}', used)
            contour = contours.get(roi_number)
            sequence = contour.get('ContourSequence', []) if contour is not None else []
            entry = {'roi_number': roi_number, 'roi_name': roi_name,
                     'mask_label': 1, 'background_label': 0,
                     'color_rgb': [int(x) for x in contour.get('ROIDisplayColor', [])] if contour else [],
                     'contour_types': sorted({str(c.ContourGeometricType) for c in sequence}),
                     'referenced_ct_sop_uids': sorted({ref for c in sequence
                         for ref in references(c, 'ContourImageSequence')}),
                     'frame_of_reference_uid': str(roi.get('ReferencedFrameOfReferenceUID', ''))}
            if not sequence or not set(entry['contour_types']) <= {'CLOSED_PLANAR', 'CLOSEDPLANAR_XOR'}:
                entry.update(status='metadata_only', reason='Empty or non-volumetric contours')
            else:
                try:
                    mask, clipping = base.rasterize_roi(ct, sequence, entry['frame_of_reference_uid'],
                                                        clip_rois=clip_rois_to_ct)
                    if not mask.any():
                        raise ValueError('Closed contours produced an empty mask')
                    nifti = str(root / 'masks' / f'{filename}.nii.gz')
                    geometry = base.write_nifti(stage / nifti, mask, ct_affine)
                    entry.update(status='converted', nifti=nifti, reference_ct_nifti=ct['nifti'],
                                 reference_ct_series_uid=ct_series_uid, geometry=geometry, **clipping,
                                 voxel_count=int(mask.sum()),
                                 volume_cc=float(mask.sum() * abs(np.linalg.det(ct_affine[:3, :3])) / 1000))
                except Exception as exc:
                    entry.update(status='failed', reason=str(exc))
                    index['issues'].append({'kind': 'ROI', 'uid': uid, 'roi': roi_number, 'error': str(exc)})
            item['rois'].append(entry)
        item['dicom'] = base.dicom_metadata(ds, base.RS_TAGS)
        write_json(stage / item['metadata'], item)
        index['structures'].append(item)

    plan_uids = set()
    for number, record in enumerate(by_modality['RTPLAN'], 1):
        ds = record['header']
        uid = record['sop_uid']
        plan_uids.add(uid)
        rel = f'RP/RP{number}.json'
        dcm_rel = f'RP/RP{number}.dcm'
        (stage / dcm_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record['path'], stage / dcm_rel)
        digest = base.sha256(record['path'])
        if base.sha256(stage / dcm_rel) != digest:
            raise ValueError('RTPLAN copy hash mismatch')
        refs = references(ds, 'ReferencedStructureSetSequence')
        item = {'sop_uid': uid, 'source_dicom_path': record['relative_path'], 'json': rel,
                'dicom_file': dcm_rel, 'dicom_sha256': digest,
                'name': str(ds.get('RTPlanLabel') or ds.get('RTPlanName') or f'RP{number}'),
                'structure_uids': refs, 'dicom': base.dicom_metadata(ds, base.RP_TAGS)}
        if set(refs) != rs_uids:
            index['issues'].append({'kind': 'RTPLAN', 'uid': uid,
                                    'error': 'Referenced RTSTRUCT differs from package RS'})
        write_json(stage / rel, item)
        index['plans'].append(item)

    used_dose = set()
    for record in by_modality['RTDOSE']:
        ds = record['header']
        uid = record['sop_uid']
        name = dose_name(ds, uid, used_dose)
        rel = f'RD/{name}.nii.gz'
        item = {'sop_uid': uid, 'source_dicom_path': record['relative_path'],
                'dose_summation_type': str(ds.get('DoseSummationType', 'UNKNOWN')),
                'plan_uids': references(ds, 'ReferencedRTPlanSequence'),
                'nifti': rel, 'metadata': f'RD/{name}.json',
                'source_dose_units': str(ds.get('DoseUnits', '')),
                'source_dose_grid_scaling': float(ds.get('DoseGridScaling', 1))}
        if set(item['plan_uids']) != plan_uids:
            index['issues'].append({'kind': 'RTDOSE', 'uid': uid,
                                    'error': 'Referenced RTPLAN differs from package RP'})
        try:
            full = pydicom.dcmread(record['path'])
            dose_array, dose_affine, convention = base.dose_volume(full)
            geometry = base.write_nifti(stage / rel, dose_array, dose_affine)
            item.update(status='converted', dose_units='Gy', dose_grid_scaling_applied=True,
                        resampled=False, grid_frame_offset_interpretation=convention, geometry=geometry)
        except Exception as exc:
            item.update(status='failed', reason=str(exc))
            index['issues'].append({'kind': 'RTDOSE', 'uid': uid, 'error': str(exc)})
        item['dicom'] = base.dicom_metadata(ds, base.RD_TAGS)
        write_json(stage / item['metadata'], item)
        index['doses'].append(item)

    for number, record in enumerate(by_modality['REG'], 1):
        ds = record['header']
        rel = f'REG/REG{number}.json'
        item = {'sop_uid': record['sop_uid'], 'source_dicom_path': record['relative_path'],
                'json': rel, 'frame_of_reference_uids': registration_frames(ds),
                'dicom': base.dicom_metadata(ds, base.COMMON + ['RegistrationSequence'])}
        write_json(stage / rel, item)
        index['registrations'].append(item)

    index['summary'].update(masks=sum(roi['status'] == 'converted'
        for struct in index['structures'] for roi in struct['rois']),
        dose_nifti=sum(dose['status'] == 'converted' for dose in index['doses']),
        issues=len(index['issues']))
    index['status'] = 'partial' if index['issues'] else 'ok'
    write_json(stage / 'index.json', index)
    return index


def convert_patient(patient, output_root, list_only=False, clip_rois_to_ct=False):
    patient = patient.resolve()
    if not (patient / 'relationships.json').is_file():
        raise ValueError(f'Missing fraction relationships.json: {patient}')
    manifest = json.loads((patient / 'relationships.json').read_text(encoding='utf-8'))
    patient_id = str(manifest['patient_id'])
    packages = sorted([p for p in patient.iterdir() if p.is_dir() and
                       (p.name == 'plan_ct1' or p.name.startswith('fraction_fbct'))], key=package_sort_key)
    if not packages:
        raise ValueError('No plan_ct1 or fraction_fbctX folders')
    destination = output_root.resolve() / patient_id
    if list_only:
        return {'patient': patient_id, 'status': 'preview', 'packages': [p.name for p in packages],
                'output': str(destination)}
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nifti-fractions-', dir=output_root) as temp:
        stage = Path(temp) / patient_id
        stage.mkdir()
        top = {'schema_version': 1, 'converter_version': VERSION, 'source': str(patient),
               'patient_id': patient_id, 'layout': 'NIfTI mirrors plan_ct1/fraction_fbctX DICOM packages.',
               'software': {name: version(name) for name in ('numpy', 'nibabel', 'pydicom', 'rt-utils')},
               'packages': [], 'status': 'ok'}
        for package in packages:
            package_stage = stage / f'.{package.name}.staging'
            try:
                # A package is staged independently: a failed conversion never looks complete.
                index = convert_package(package, package_stage, clip_rois_to_ct=clip_rois_to_ct)
                package_stage.rename(stage / package.name)
                top['packages'].append({'package': package.name, 'status': index['status'],
                                        'summary': index['summary']})
                if index['status'] != 'ok':
                    top['status'] = 'partial'
            except Exception as exc:
                shutil.rmtree(package_stage, ignore_errors=True)
                top['packages'].append({'package': package.name, 'status': 'failed', 'error': str(exc)})
                top['status'] = 'partial'
        write_json(stage / 'index.json', top)
        if destination.exists():
            raise ValueError('Destination appeared during conversion')
        stage.rename(destination)
    return {'patient': patient_id, 'status': top['status'], 'packages': len(packages),
            'output': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='One organized fraction patient folder')
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'nifti_fractions')
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--clip-rois-to-ct', action='store_true',
                        help='Allow masks to be clipped to the CT field of view (recorded in metadata).')
    args = parser.parse_args()
    try:
        result = convert_patient(args.source, args.output_root, args.list_only,
                                 clip_rois_to_ct=args.clip_rois_to_ct)
    except Exception as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('ok', 'preview') else 1


if __name__ == '__main__':
    sys.exit(main())
