#!/usr/bin/env python3
"""Copy one anonymized RT patient into planning/fraction packages using DICOM references."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

import pydicom


MODALITY_FOLDERS = {
    'CT': 'CT', 'RTSTRUCT': 'RS', 'RTPLAN': 'RP', 'RTDOSE': 'RD', 'REG': 'REG',
}


def only(values, description):
    values = sorted(set(values))
    if len(values) != 1:
        raise ValueError(f'{description}: expected one value, found {values}')
    return values[0]


def referenced_sops(dataset, sequence):
    return [str(item.ReferencedSOPInstanceUID) for item in dataset.get(sequence, [])
            if item.get('ReferencedSOPInstanceUID')]


def struct_series(dataset):
    values = []
    for frame in dataset.get('ReferencedFrameOfReferenceSequence', []):
        for study in frame.get('RTReferencedStudySequence', []):
            for series in study.get('RTReferencedSeriesSequence', []):
                if series.get('SeriesInstanceUID'):
                    values.append(str(series.SeriesInstanceUID))
    return values


def registration_frames(dataset):
    return [str(item.FrameOfReferenceUID) for item in dataset.get('RegistrationSequence', [])
            if item.get('FrameOfReferenceUID')]


def package_hint(path):
    """Use CT file name/its direct source directory as the explicit CT/FBC T label."""
    evidence = f'{path.name} {path.parent.name}'
    fbct = re.findall(r'FBCT\s*[_-]?(\d+)', evidence, flags=re.IGNORECASE)
    if fbct:
        return f'fraction_fbct{int(only(fbct, "FBCT number"))}'
    if re.search(r'(?:^|[._ -])CT1(?:[._ -]|$)', evidence, re.IGNORECASE):
        return 'plan_ct1'
    return None


def load_records(source):
    records, patient_ids = [], set()
    for path in sorted(source.rglob('*.dcm')):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        modality = str(dataset.get('Modality', ''))
        sop = str(dataset.get('SOPInstanceUID', ''))
        if not modality or not sop:
            raise ValueError(f'Missing Modality or SOPInstanceUID: {path}')
        patient_id = str(dataset.get('PatientID', ''))
        if not patient_id:
            raise ValueError(f'Missing PatientID: {path}')
        patient_ids.add(patient_id)
        record = {
            'source_path': str(path.relative_to(source)), 'path': path,
            'modality': modality, 'sop_uid': sop,
            'series_uid': str(dataset.get('SeriesInstanceUID', '')),
            'frame_uid': str(dataset.get('FrameOfReferenceUID', '')),
            'package': None, 'assignment_basis': None,
            'rs_uids': referenced_sops(dataset, 'ReferencedStructureSetSequence'),
            'plan_uids': referenced_sops(dataset, 'ReferencedRTPlanSequence'),
            'ct_series_uids': struct_series(dataset),
            'reg_frame_uids': registration_frames(dataset),
        }
        records.append(record)
    return records, only(patient_ids, 'PatientID')


def assign(record, package, basis):
    if record['package'] is None:
        record['package'], record['assignment_basis'] = package, basis
        return True
    if record['package'] != package:
        raise ValueError(f'{record["source_path"]}: {record["package"]} conflicts with {package} ({basis})')
    return False


def resolve(records):
    by_sop = {record['sop_uid']: record for record in records}
    if len(by_sop) != len(records):
        raise ValueError('Duplicate SOPInstanceUIDs are not supported')
    ct_by_series, ct_by_frame = {}, defaultdict(set)
    for record in records:
        if record['modality'] != 'CT':
            continue
        package = package_hint(record['path'])
        if not package:
            raise ValueError(f'Cannot identify CT1/FBCT number from: {record["source_path"]}')
        assign(record, package, 'CT filename/direct parent directory')
        if record['series_uid']:
            existing = ct_by_series.setdefault(record['series_uid'], package)
            if existing != package:
                raise ValueError(f'CT series maps to multiple packages: {record["series_uid"]}')
        if record['frame_uid']:
            ct_by_frame[record['frame_uid']].add(package)

    changed = True
    while changed:
        changed = False
        for record in records:
            if record['package']:
                continue
            if record['modality'] == 'RTSTRUCT':
                packages = [ct_by_series[uid] for uid in record['ct_series_uids'] if uid in ct_by_series]
                if packages:
                    changed |= assign(record, only(packages, f'RTSTRUCT {record["source_path"]} CT reference'),
                                      'RTSTRUCT -> CT SeriesInstanceUID')
            elif record['modality'] == 'RTPLAN':
                packages = [by_sop[uid]['package'] for uid in record['rs_uids']
                            if uid in by_sop and by_sop[uid]['package']]
                if packages:
                    changed |= assign(record, only(packages, f'RTPLAN {record["source_path"]} RS reference'),
                                      'RTPLAN -> RTSTRUCT')
            elif record['modality'] == 'RTDOSE':
                packages = [by_sop[uid]['package'] for uid in record['plan_uids']
                            if uid in by_sop and by_sop[uid]['package']]
                if packages:
                    changed |= assign(record, only(packages, f'RTDOSE {record["source_path"]} RP reference'),
                                      'RTDOSE -> RTPLAN')
            elif record['modality'] == 'REG':
                packages = {package for frame in record['reg_frame_uids'] for package in ct_by_frame.get(frame, set())}
                fractions = sorted(package for package in packages if package.startswith('fraction_'))
                if len(fractions) == 1:
                    changed |= assign(record, fractions[0], 'REG FrameOfReferenceUID (fraction target)')
                elif packages == {'plan_ct1'}:
                    changed |= assign(record, 'plan_ct1', 'REG FrameOfReferenceUID')

    unresolved = [record for record in records if record['package'] is None]
    if unresolved:
        paths = '\n'.join(record['source_path'] for record in unresolved[:20])
        raise ValueError(f'Unresolved DICOM package assignments ({len(unresolved)}):\n{paths}')
    return by_sop


def output_path(destination, record):
    folder = MODALITY_FOLDERS.get(record['modality'], 'OTHER')
    return destination / record['package'] / folder / record['path'].name


def manifest_record(record):
    return {key: value for key, value in record.items() if key != 'path'}


def organize(source, output_root, list_only=False):
    source = source.resolve()
    records, patient_id = load_records(source)
    resolve(records)
    destination = (output_root / patient_id).resolve()
    if destination.exists() and not list_only:
        raise ValueError(f'Output already exists: {destination}')
    targets = {}
    for record in records:
        target = output_path(destination, record)
        if target in targets:
            raise ValueError(f'Filename collision: {target}')
        targets[target] = record
    summary = defaultdict(Counter)
    for record in records:
        summary[record['package']][record['modality']] += 1
    print(json.dumps({'patient': patient_id, 'source': str(source), 'output': str(destination),
                      'packages': {key: dict(value) for key, value in sorted(summary.items())},
                      'files': len(records)}, ensure_ascii=False, indent=2))
    if list_only:
        return
    for target, record in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record['path'], target)
        if target.stat().st_size != record['path'].stat().st_size:
            raise IOError(f'Copy size mismatch: {target}')
    manifest = {
        'schema_version': 1, 'source': str(source), 'patient_id': patient_id,
        'layout': 'plan_ct1 and fraction_fbctX packages; files are copied, source unchanged.',
        'packages': {key: dict(value) for key, value in sorted(summary.items())},
        'files': [manifest_record(record) for record in records],
    }
    (destination / 'relationships.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [f'# {patient_id}', '', '按计划CT与治疗次归档的DICOM副本；源数据未改动。', '',
             '| 包 | CT | RS | RP | RD | REG |', '|---|---:|---:|---:|---:|---:|']
    for package, counts in sorted(summary.items()):
        lines.append('| ' + ' | '.join([package] + [str(counts.get(modality, 0))
                     for modality in ('CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE', 'REG')]) + ' |')
    lines += ['', '归属规则：CT从文件名和直接父目录的CT1/FBCT编号识别；RS→CT、RP→RS、RD→RP、REG→FrameOfReferenceUID。',
              '详细 UID 引用、来源路径和每个文件的归属依据见 [relationships.json](relationships.json)。', '']
    (destination / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Single anonymized patient export root')
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'organized_fractions')
    parser.add_argument('--list-only', action='store_true', help='Resolve and print package counts without copying')
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f'Not a directory: {args.source}')
    try:
        organize(args.source, args.output_root, args.list_only)
    except Exception as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
