"""Session-based RT viewer with add-folder and close-case controls.

The original server.py is intentionally unchanged. Browser-selected folders are
copied into a temporary session workspace. Closing a case only removes its
session link; source data is never deleted.
"""
from __future__ import annotations

import argparse
import atexit
from collections import defaultdict
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

import sys
HERE = Path(__file__).resolve().parent
VIEWER = HERE / "viewer"
sys.path.insert(0, str(VIEWER))

from flask import jsonify, request, send_from_directory
import pydicom

import nifti_backend
import server as core


from organize_dicom import organize  # noqa: E402


SESSION = Path(tempfile.mkdtemp(prefix="rtdcmviewer-"))
SESSION_DICOM = SESSION / "dicom"
SESSION_NIFTI = SESSION / "nifti"
STAGING = SESSION / "staging"
HIDDEN = SESSION / "closed"
CASE_META: dict[tuple[str, str], dict[str, str]] = {}
for directory in (SESSION_DICOM, SESSION_NIFTI, STAGING, HIDDEN):
    directory.mkdir(parents=True, exist_ok=True)
atexit.register(lambda: shutil.rmtree(SESSION, ignore_errors=True))

core.app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3


def safe_name(value: str, fallback: str = "unnamed") -> str:
    result = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")
    return (result[:90] or fallback)


def unique_path(parent: Path, preferred: str) -> Path:
    candidate = parent / safe_name(preferred)
    number = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = parent / f"{safe_name(preferred)}_{number}"
        number += 1
    return candidate


def safe_upload_relative(filename: str) -> Path:
    value = filename.replace("\\", "/")
    pure = PurePosixPath(value)
    if (pure.is_absolute() or not pure.parts
            or any(part in ("", ".", "..") or "\x00" in part for part in pure.parts)):
        raise ValueError("上传目录包含无效相对路径")
    return Path(*pure.parts)


def save_uploads() -> tuple[Path, str, int]:
    files = request.files.getlist("files")
    if not files:
        raise ValueError("没有收到文件")
    token = next(tempfile._get_candidate_names())
    staging = STAGING / token
    staging.mkdir()
    count = 0
    roots = []
    for upload in files:
        if not upload.filename:
            continue
        relative = safe_upload_relative(upload.filename)
        roots.append(relative.parts[0])
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        upload.save(target)
        count += 1
    if not count:
        raise ValueError("所选目录没有可读取的文件")
    root_name = safe_name(request.form.get("root_name") or (roots[0] if roots else "folder"), "folder")
    return staging, root_name, count


def install_symlink(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination.parent, destination.name)
    destination.symlink_to(source.resolve(), target_is_directory=True)
    return destination


def add_initial_dicom(root: Path):
    if not root.is_dir():
        return
    for manifest in sorted(root.glob("*/relationships.json")):
        installed = install_symlink(manifest.parent, SESSION_DICOM / manifest.parent.name)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        source_folder = Path(data.get("source") or manifest.parent).name
        patient_id = manifest.parent.name
        for record in data.get("files", []):
            if record.get("modality") not in ("CT", "RTSTRUCT", "RTPLAN", "RTDOSE"):
                continue
            try:
                dataset = pydicom.dcmread(manifest.parent / record["path"], stop_before_pixels=True,
                                          specific_tags=["PatientID", "PatientName"])
                patient_id = dicom_identity(dataset)
                break
            except Exception:
                continue
        CASE_META[("dicom", installed.name)] = {
            "patient_id": patient_id,
            "source_folder": source_folder,
            "source_path": str(Path(data.get("source") or manifest.parent).resolve()),
        }


def add_initial_nifti(root: Path):
    if not root.is_dir():
        return
    for case, index_path in nifti_backend.catalog(root).items():
        parts = case.split("/", 1)
        destination = SESSION_NIFTI.joinpath(*parts)
        installed = install_symlink(index_path.parent, destination)
        installed_case = f"{installed.parent.name}/{installed.name}" if len(parts) == 2 else installed.name
        folder = index_path.parent.parent.name if len(parts) == 2 else index_path.parent.name
        CASE_META[("nifti", installed_case)] = {
            "patient_id": parts[0] if len(parts) == 2 else installed.name,
            "source_folder": folder,
            "source_path": str((index_path.parent.parent if len(parts) == 2 else index_path.parent).resolve()),
        }


def dicom_identity(dataset) -> str:
    patient = str(dataset.get("PatientID", "")).strip()
    if not patient:
        patient = str(dataset.get("PatientName", "")).strip()
    if not patient:
        patient = str(dataset.get("StudyInstanceUID", "")).strip()
    return safe_name(patient, "unknown_patient")


def add_dicom_folder(staging: Path, selected_name: str) -> tuple[list[str], list[str]]:
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    warnings = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(
                path, stop_before_pixels=True,
                specific_tags=["PatientID", "PatientName", "StudyInstanceUID",
                               "SOPInstanceUID", "Modality"],
            )
            if not dataset.get("SOPInstanceUID") or not dataset.get("Modality"):
                continue
        except Exception:
            continue
        groups[dicom_identity(dataset)][str(dataset.SOPInstanceUID)] = path
    if not groups:
        raise ValueError("所选目录中未识别到 DICOM 对象")

    added = []
    group_root = staging / "_patients"
    group_root.mkdir()
    for patient, objects in sorted(groups.items()):
        case_source = unique_path(group_root, patient)
        case_source.mkdir()
        for uid, source in objects.items():
            target = case_source / f"{safe_name(uid)}.dcm"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        destination = unique_path(SESSION_DICOM, patient)
        if destination.name != case_source.name:
            renamed = case_source.with_name(destination.name)
            case_source.rename(renamed)
            case_source = renamed
        result = organize(case_source, SESSION_DICOM, {})
        added.append(Path(result["output"]).name)
        CASE_META[("dicom", added[-1])] = {
            "patient_id": patient,
            "source_folder": selected_name,
            "source_path": "",
        }
        if result.get("warnings"):
            warnings.append(f"{added[-1]}：{result['warnings']} 条索引警告")
    return added, warnings


def index_kind(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return "fraction" if data.get("ct") else "legacy" if "images" in data else None


def add_nifti_folder(staging: Path, selected_name: str) -> tuple[list[str], list[str]]:
    before = set(nifti_backend.catalog(SESSION_NIFTI))
    found = []
    for index_path in sorted(staging.rglob("index.json")):
        kind = index_kind(index_path)
        if kind:
            found.append((kind, index_path.parent))
    if not found:
        raise ValueError("未找到可用的 NIfTI index.json；请添加转换后的病例目录")

    installed_sources = set()
    for kind, package in found:
        resolved = package.resolve()
        if resolved in installed_sources:
            continue
        installed_sources.add(resolved)
        if kind == "fraction":
            parent = package.parent
            patient = selected_name if parent == staging else parent.name
            destination = SESSION_NIFTI / safe_name(patient) / safe_name(package.name)
        else:
            destination = SESSION_NIFTI / safe_name(package.name)
        installed = install_symlink(package, destination)
        installed_case = (f"{installed.parent.name}/{installed.name}"
                          if kind == "fraction" else installed.name)
        folder_name = package.parent.name if kind == "fraction" else package.name
        CASE_META[("nifti", installed_case)] = {
            "patient_id": installed.parent.name if kind == "fraction" else installed.name,
            "source_folder": safe_name(folder_name, selected_name),
            "source_path": "",
        }

    nifti_backend.catalog_kind.cache_clear()
    after = set(nifti_backend.catalog(SESSION_NIFTI))
    added = sorted(after - before)
    if not added:
        raise ValueError("NIfTI 病例与列表中已有条目重复")
    return added, []


def move_closed(path: Path):
    if path.is_symlink():
        path.unlink()
    else:
        target = unique_path(HIDDEN, path.name)
        shutil.move(str(path), str(target))
    parent = path.parent
    if parent != SESSION_NIFTI and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


@core.app.post("/api/library/add")
@core.api
def library_add():
    source = request.args.get("source")
    if source not in ("dicom", "nifti"):
        raise ValueError("只能向 DICOM 或 NIfTI 病例库添加文件夹")
    staging, selected_name, uploaded = save_uploads()
    if source == "dicom":
        added, warnings = add_dicom_folder(staging, selected_name)
    else:
        added, warnings = add_nifti_folder(staging, selected_name)
    core.CACHE.clear()
    return jsonify(added=added, uploaded_files=uploaded, warnings=warnings)


@core.app.delete("/api/library/case")
@core.api
def library_close():
    source = request.args.get("source")
    case = request.args.get("case", "")
    if source == "dicom":
        target = (SESSION_DICOM / case).resolve(strict=False)
        listed = core.catalog().get(case)
        if listed is None:
            raise ValueError("DICOM 病例不在当前列表")
        target = listed.parent
    elif source == "nifti":
        paths = nifti_backend.catalog(SESSION_NIFTI)
        if case not in paths:
            raise ValueError("NIfTI 病例不在当前列表")
        target = paths[case].parent
    else:
        raise ValueError("快速查看不使用病例列表")
    move_closed(target)
    CASE_META.pop((source, case), None)
    nifti_backend.catalog_kind.cache_clear()
    core.CACHE.clear()
    return jsonify(closed=case)


@core.app.get("/api/library/meta")
@core.api
def library_meta():
    source = request.args.get("source")
    case = request.args.get("case", "")
    if source not in ("dicom", "nifti"):
        return jsonify(folder_name="", case_name=case)
    meta = CASE_META.get((source, case), {})
    patient_id = meta.get("patient_id") or case.split("/", 1)[0]
    source_folder = meta.get("source_folder") or case.split("/", 1)[0]
    return jsonify(patient_id=patient_id, source_folder=source_folder,
                   source_path=meta.get("source_path", ""),
                   list_item=case.rsplit("/", 1)[-1])


def new_home():
    return send_from_directory(VIEWER / "static", "index_new.html")


core.app.view_functions["home"] = new_home


def configure(dicom_root: Path, nifti_root: Path, default_source: str):
    add_initial_dicom(dicom_root.resolve())
    add_initial_nifti(nifti_root.resolve())
    core.ROOT = SESSION_DICOM
    core.NIFTI_ROOT = SESSION_NIFTI
    core.app.config["DEFAULT_SOURCE"] = default_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=core.OUTPUT_ROOT / "organized_dicom")
    parser.add_argument("--nifti-root", type=Path, default=core.OUTPUT_ROOT / "nifti_fractions")
    parser.add_argument("--default-source", choices=("dicom", "nifti"), default="nifti")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    configure(args.data_root, args.nifti_root, args.default_source)
    print(f"New viewer: http://127.0.0.1:{args.port}/?source={args.default_source}", flush=True)
    print(f"Temporary session: {SESSION}", flush=True)
    core.app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
