#!/usr/bin/env python3
"""Copy DICOM exports into indexed patient folders without changing sources."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone
from urllib.parse import quote

import pydicom


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(value):
    value = re.sub(r'[^\w.-]+', '_', str(value)).strip('._')
    return value[:90] or 'unnamed'


def referenced(ds, sequence):
    return [str(x.ReferencedSOPInstanceUID) for x in ds.get(sequence, [])
            if x.get('ReferencedSOPInstanceUID')]


def struct_series(ds):
    result = set()
    for frame in ds.get('ReferencedFrameOfReferenceSequence', []):
        for study in frame.get('RTReferencedStudySequence', []):
            for series in study.get('RTReferencedSeriesSequence', []):
                if series.get('SeriesInstanceUID'):
                    result.add(str(series.SeriesInstanceUID))
    return sorted(result)


def classify(ds, source, overrides):
    uid = str(ds.get('SeriesInstanceUID', ''))
    if uid in overrides:
        kind = overrides[uid].upper()
        if kind not in ('CT', 'FBCT', 'CBCT', 'MR'):
            raise ValueError(f'Unsupported series override: {kind}')
        return kind, 'manual override'
    modality = str(ds.get('Modality', 'UNKNOWN'))
    if modality != 'CT':
        return safe(modality), 'DICOM Modality'
    description = ' '.join(str(ds.get(k, '')) for k in
                           ('SeriesDescription', 'ProtocolName'))
    evidence = description + ' ' + ' '.join(source.parts[-3:-1])
    matches = set()
    if re.search(r'CBCT|CONE[ _-]*BEAM', evidence, re.I):
        matches.add('CBCT')
    if re.search(r'FBCT|FAN[ _-]*BEAM', evidence, re.I):
        matches.add('FBCT')
    if len(matches) == 1:
        return matches.pop(), f'description/path: {evidence}'
    if matches:
        return 'CT_unclassified', f'conflicting hints: {evidence}'
    return 'CT', 'Modality=CT; acquisition purpose unconfirmed'


def organize(source, output_root, overrides):
    source = source.resolve()
    destination = (output_root / source.name).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate directory trees')
    manifest_path = destination / 'relationships.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old and old.get('source') != str(source):
        raise ValueError('This output folder belongs to a different source')
    assignments = old.get('assignments', {})
    used = set(assignments.values())

    def allocate(key, parent, preferred, numbered=False):
        if key in assignments:
            return assignments[key]
        number = 1
        candidate = f'{parent}{preferred}{number}' if numbered else f'{parent}{preferred}'
        while candidate in used:
            number += 1
            candidate = f'{parent}{preferred}{number}' if numbered else f'{parent}{preferred}_{number}'
        assignments[key] = candidate
        used.add(candidate)
        return candidate

    records = []
    datasets = {}
    warnings = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if not ds.get('SOPInstanceUID') or not ds.get('Modality'):
                raise ValueError('Missing SOPInstanceUID or Modality')
        except Exception as exc:
            warnings.append(f'{path.relative_to(source)}: kept in other; DICOM read failed: {exc}')
            records.append({'source_path': str(path.relative_to(source)),
                            'modality': 'OTHER', 'read_error': str(exc)})
            continue
        uid = str(ds.SOPInstanceUID)
        record = {'source_path': str(path.relative_to(source)), 'sop_uid': uid,
                  'series_uid': str(ds.get('SeriesInstanceUID', '')),
                  'modality': str(ds.Modality),
                  'frame_of_reference_uid': str(ds.get('FrameOfReferenceUID', '')),
                  'series_description': str(ds.get('SeriesDescription', '')),
                  'date_time': str(ds.get('AcquisitionDateTime', '')) or
                  (str(ds.get('SeriesDate', ds.get('StudyDate', ''))) +
                   str(ds.get('SeriesTime', ds.get('StudyTime', ''))))}
        if record['modality'] == 'RTPLAN':
            record.update(plan_label=str(ds.get('RTPlanLabel', '')),
                          plan_name=str(ds.get('RTPlanName', '')),
                          structure_uids=referenced(ds, 'ReferencedStructureSetSequence'))
        elif record['modality'] == 'RTSTRUCT':
            record.update(structure_label=str(ds.get('StructureSetLabel', '')),
                          ct_series_uids=struct_series(ds),
                          rois=[{'number': int(x.ROINumber), 'name': str(x.ROIName)}
                                for x in ds.get('StructureSetROISequence', [])])
        elif record['modality'] == 'RTDOSE':
            beams = []
            for plan in ds.get('ReferencedRTPlanSequence', []):
                for group in plan.get('ReferencedFractionGroupSequence', []):
                    for beam in group.get('ReferencedBeamSequence', []):
                        beams.append({'plan_uid': str(plan.ReferencedSOPInstanceUID),
                                      'fraction_group': int(group.ReferencedFractionGroupNumber),
                                      'beam_number': int(beam.ReferencedBeamNumber)})
            record.update(plan_uids=referenced(ds, 'ReferencedRTPlanSequence'),
                          dose_type=str(ds.get('DoseSummationType', 'UNKNOWN')),
                          dose_units=str(ds.get('DoseUnits', '')), beams=beams)
        datasets[str(path.relative_to(source))] = ds
        records.append(record)
    if not records:
        raise ValueError(f'Empty source: {source}')

    index = {r['sop_uid']: r for r in records if 'sop_uid' in r}
    series = defaultdict(list)
    for r in records:
        if r['modality'] in ('CT', 'MR', 'PT', 'NM'):
            series[r['series_uid'] or r['sop_uid']].append(r)
    for uid, group in sorted(series.items(), key=lambda x: (x[1][0]['date_time'], x[0])):
        kinds = [classify(datasets[r['source_path']], source / r['source_path'], overrides)
                 for r in group]
        kind, evidence = kinds[0]
        if len({x[0] for x in kinds}) > 1:
            kind, evidence = 'CT_unclassified', 'conflicting series descriptions'
        folder = allocate('series:' + uid, '', kind, numbered=True)
        for r in group:
            r.update(folder=folder, image_type=kind, classification_basis=evidence)

    for r in sorted(records, key=lambda x: x.get('sop_uid', x['source_path'])):
        uid = r.get('sop_uid', '')
        if r['modality'] == 'RTSTRUCT':
            r['folder'] = allocate('struct:' + uid, 'RTstruct/', 'RS', numbered=True)
        elif r['modality'] == 'RTPLAN':
            label = safe(r['plan_label'] or r['plan_name'] or 'Plan')
            r['folder'] = allocate('plan:' + uid, 'plan/', label)
    for r in records:
        if r['modality'] == 'RTDOSE':
            refs = r['plan_uids']
            if len(refs) == 1 and refs[0] in index and index[refs[0]]['modality'] == 'RTPLAN':
                name = Path(index[refs[0]]['folder']).name
            else:
                name = 'MULTI_PLAN' if len(refs) > 1 else 'UNMATCHED'
            r['folder'] = f'dose/{name}/{safe(r["dose_type"])}'
        r.setdefault('folder', 'other/' + safe(r['modality']))
        for key, expected in [('plan_uids', 'RTPLAN'), ('structure_uids', 'RTSTRUCT')]:
            for uid in r.get(key, []):
                if uid not in index or index[uid]['modality'] != expected:
                    warnings.append(f'{r["source_path"]}: unresolved {key}: {uid}')
        for uid in r.get('ct_series_uids', []):
            if uid not in series:
                warnings.append(f'{r["source_path"]}: missing image series: {uid}')

    # Validate every target before copying any input; names and bytes stay intact.
    targets = {}
    uid_hashes = {}
    for r in records:
        path = source / r['source_path']
        r['sha256'] = digest(path)
        uid = r.get('sop_uid')
        if uid and uid in uid_hashes and uid_hashes[uid] != r['sha256']:
            raise ValueError(f'Same SOPInstanceUID has different file contents: {uid}')
        if uid:
            uid_hashes[uid] = r['sha256']
        relative = str(Path(r['folder']) / path.name)
        if relative in targets and targets[relative] != r['sha256']:
            raise ValueError(f'Conflicting source filenames: {relative}')
        targets[relative] = r['sha256']
        r['path'] = relative
        target = destination / relative
        if target.exists() and digest(target) != r['sha256']:
            raise ValueError(f'Refusing to overwrite changed file: {target}')
    for r in records:
        target = destination / r['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source / r['source_path'], target)
        if digest(target) != r['sha256']:
            raise IOError(f'Copy verification failed: {target}')

    manifest = {'schema_version': 1, 'source': str(source),
                'assignments': assignments, 'files': records, 'warnings': warnings,
                'notes': ['References describe DICOM links, not geometric or clinical validation.',
                          'CT without FBCT/CBCT hints does not establish planning purpose.']}
    destination.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(manifest_path)

    def link(folder):
        return f'[{folder}]({quote(folder)}/)'

    lines = [f'# {source.name}', '', '原始文件仅复制；副本 SHA-256 已核对。', '',
             '## 影像序列', '', '| 目录 | 类型 | 时间 | 文件数 | 原始描述 |',
             '|---|---|---|---:|---|']
    for group in series.values():
        r = group[0]
        desc = r['series_description'].replace('|', '/')
        lines.append(f'| {link(r["folder"])} | {r["image_type"]} | {r["date_time"]} | {len(group)} | {desc} |')
    lines += ['', 'CT 表示 DICOM 模态；没有明确分类线索时，不认定它一定是定位 CT。', '',
              '## 计划关联', '', '| 计划 | 结构集 | 结构引用的影像 | 剂量 |', '|---|---|---|---|']
    for r in records:
        if r['modality'] != 'RTPLAN':
            continue
        structs = [index[u] for u in r['structure_uids'] if u in index]
        images = sorted({series[u][0]['folder'] for s in structs
                         for u in s.get('ct_series_uids', []) if u in series})
        doses = [d for d in records if r['sop_uid'] in d.get('plan_uids', [])]
        dose_links = sorted({link(d['folder']) for d in doses})
        lines.append('| ' + ' | '.join([link(r['folder']),
                     ', '.join(link(s['folder']) for s in structs) or '未解析',
                     ', '.join(link(i) for i in images) or '未解析',
                     ', '.join(dose_links) or '无']) + ' |')
    lines += ['', '## 核查说明', '',
              '关联依据：RD → RP → RS → 影像序列的 DICOM UID 引用。未执行空间几何验证或剂量累加。',
              '详细来源、ROI 名称、射野编号、文件路径与哈希见 [relationships.json](relationships.json)。', '']
    lines += ['- ' + w for w in warnings] or ['未发现缺失的显式计划、结构或影像序列引用。']
    (destination / 'README.md').write_text('\n'.join(lines) + '\n')
    summary = {'output': str(destination), 'files': len(records),
               'unique_copies': len(targets), 'warnings': len(warnings)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def select_sources(source, mode):
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f'Source is not a directory: {source}')
    if mode == 'auto':
        # Flat exports have DICOM objects directly inside each patient folder.
        direct_dicom = False
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True,
                                    specific_tags=['SOPInstanceUID', 'Modality'])
                if ds.get('SOPInstanceUID') and ds.get('Modality'):
                    direct_dicom = True
                    break
            except Exception:
                continue
        if direct_dicom:
            mode = 'patient'
        else:
            identities = defaultdict(set)
            incomplete_identity = False
            for path in sorted(source.rglob('*')):
                if not path.is_file() or path.suffix.lower() != '.dcm':
                    continue
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True,
                                        specific_tags=['PatientID', 'IssuerOfPatientID', 'SOPInstanceUID'])
                    if not ds.get('SOPInstanceUID'):
                        continue
                    if not ds.get('PatientID'):
                        incomplete_identity = True
                        continue
                    identity = (str(ds.PatientID), str(ds.get('IssuerOfPatientID', '')))
                    identities[path.relative_to(source).parts[0]].add(identity)
                except Exception:
                    continue
            patients = set().union(*identities.values()) if identities else set()
            if not incomplete_identity and len(patients) == 1:
                mode = 'patient'
            elif not incomplete_identity and len(patients) > 1 and all(
                    len(group) == 1 for group in identities.values()):
                mode = 'batch'
            else:
                raise ValueError('Ambiguous nested input; specify --mode patient or --mode batch')
    sources = [source] if mode == 'patient' else sorted(
        p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not sources:
        raise ValueError('No patient directories found; nested single patients need --mode patient')
    return mode, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'organized_dicom')
    parser.add_argument('--mode', choices=['auto', 'patient', 'batch'], default='auto',
                        help='Auto uses direct files or nested patient identities; use explicit mode for ambiguous exports')
    parser.add_argument('--batch', action='store_true', help='Alias for --mode batch')
    parser.add_argument('--list-only', action='store_true', help='List patient folders without writing files')
    parser.add_argument('--series-types', type=Path, help='JSON mapping SeriesInstanceUID to CT/FBCT/CBCT/MR')
    args = parser.parse_args()
    overrides = json.loads(args.series_types.read_text()) if args.series_types else {}
    source_root = args.source.resolve()
    output_root = args.output_root.resolve()
    if (source_root == output_root or source_root in output_root.parents
            or output_root in source_root.parents):
        parser.error('Input and output roots must be separate directory trees')
    mode, sources = select_sources(source_root, 'batch' if args.batch else args.mode)
    print(f'Mode: {mode}; patients: {len(sources)}; output: {output_root}')
    if args.list_only:
        for source in sources:
            print(f'{source.name} -> {output_root / source.name}')
        return 0
    results = []
    for number, source in enumerate(sources, 1):
        print(f'[{number}/{len(sources)}] {source.name}', flush=True)
        try:
            summary = organize(source, output_root, overrides)
            results.append({'patient': source.name, 'status': 'ok', **summary})
        except Exception as exc:
            results.append({'patient': source.name, 'status': 'failed', 'error': str(exc)})
            print(f'FAILED {source.name}: {exc}', file=sys.stderr, flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = output_root / f'batch_summary_{stamp}.json'
    report.write_text(json.dumps({'source': str(source_root), 'mode': mode,
                                 'results': results}, ensure_ascii=False, indent=2) + '\n')
    failures = sum(r['status'] == 'failed' for r in results)
    print(f'Finished: {len(results)-failures} succeeded, {failures} failed. Report: {report}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
