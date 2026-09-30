"""Session-based RT viewer with add-folder and close-case controls.

The original server.py is intentionally unchanged. Browser-selected folders are
copied into a temporary session workspace. Closing a case only removes its
session link; source data is never deleted.

This module also overrides ``/api/dvh`` with an absolute-dose metric spec so
research protocols can request their own indicators without editing the core
viewer.
"""
from __future__ import annotations

import argparse
import atexit
from collections import defaultdict
import csv
import io
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

from flask import jsonify, request, Response, send_from_directory
import numpy as np
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


def read_index(path: Path) -> tuple[str | None, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    kind = ("fraction" if data.get("ct") else "legacy" if "images" in data
            else "collection" if data.get("patient_id") and "packages" in data else None)
    return kind, data


def fraction_patient_id(package: Path, index: dict, staging: Path,
                        selected_name: str) -> str:
    for parent in package.parents:
        if parent == staging:
            break
        kind, data = read_index(parent / "index.json")
        if kind == "collection" and data.get("patient_id"):
            return safe_name(data["patient_id"], "unknown_patient")

    source_package = index.get("source_package")
    if source_package:
        source_patient = Path(str(source_package)).parent.name
        if source_patient:
            return safe_name(source_patient, "unknown_patient")

    if package.parent != staging and package.parent.name != selected_name:
        return safe_name(package.parent.name, "unknown_patient")
    if not re.fullmatch(r"(?:plan_ct\d+|fraction_(?:fbct|cbct|ct)\d+)", selected_name,
                        flags=re.IGNORECASE):
        return safe_name(selected_name, "unknown_patient")
    return "unknown_patient"


def add_nifti_folder(staging: Path, selected_name: str) -> tuple[list[str], list[str]]:
    before = set(nifti_backend.catalog(SESSION_NIFTI))
    found = []
    for index_path in sorted(staging.rglob("index.json")):
        kind, data = read_index(index_path)
        if kind in ("fraction", "legacy"):
            found.append((kind, index_path.parent, data))
    if not found:
        raise ValueError("未找到可用的 NIfTI index.json；请添加转换后的病例目录")

    installed_sources = set()
    for kind, package, index in found:
        resolved = package.resolve()
        if resolved in installed_sources:
            continue
        installed_sources.add(resolved)
        if kind == "fraction":
            patient = fraction_patient_id(package, index, staging, selected_name)
            destination = SESSION_NIFTI / safe_name(patient) / safe_name(package.name)
        else:
            destination = SESSION_NIFTI / safe_name(package.name)
        installed = install_symlink(package, destination)
        installed_case = (f"{installed.parent.name}/{installed.name}"
                          if kind == "fraction" else installed.name)
        CASE_META[("nifti", installed_case)] = {
            "patient_id": patient if kind == "fraction" else installed.name,
            "source_folder": selected_name,
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


# --- DVH metrics -----------------------------------------------------------
#
# The core viewer exposes a fixed Dmean / D95 / D2 / V20 table. This override
# keeps that response shape and adds a metric spec on top, so the front end can
# render whatever columns the current protocol asks for.
#
# Supported tokens (case-insensitive, separated by comma, semicolon or space):
#   D95       dose to 95% of the volume
#   D2cc      lowest dose inside the hottest 2 cc
#   V20Gy     percentage of the volume receiving at least 20 Gy
#   Dmean, Dmax, Dmin, volume
#
# Relative-dose forms (D95%, V107%) need a prescription dose, which the current
# pipeline does not carry, so they are rejected with an explicit message rather
# than silently resolved against the maximum dose.

MAX_METRICS = 12
DEFAULT_METRICS = "Dmean,D95,D2,V20Gy"
_METRIC_TOKEN = re.compile(r"^[A-Za-z0-9.%]+$")
_RELATIVE_HINT = ("相对剂量型需要处方剂量，当前链路未提供；"
                  "请改用绝对剂量型，如 D95、D2cc、V20Gy、Dmax")

# (key, kind, parameter, label, unit)
Metric = tuple[str, str, float | None, str, str]


def parse_metric(token: str) -> Metric:
    if len(token) > 24 or not _METRIC_TOKEN.match(token):
        raise ValueError(f"无法识别的指标 “{token}”")
    upper = token.upper()
    if re.fullmatch(r"[DV]\d+(?:\.\d+)?%", upper):
        raise ValueError(f"“{token}”：{_RELATIVE_HINT}")
    if upper in ("MEAN", "DMEAN"):
        return "Dmean", "mean", None, "Dmean", "Gy"
    if upper in ("MAX", "DMAX"):
        return "Dmax", "max", None, "Dmax", "Gy"
    if upper in ("MIN", "DMIN"):
        return "Dmin", "min", None, "Dmin", "Gy"
    if upper in ("VOL", "VOLUME", "VCC"):
        return "Volume", "volume", None, "体积", "cm³"

    percent = re.fullmatch(r"D(\d+(?:\.\d+)?)(CC)?", upper)
    if percent:
        value = float(percent.group(1))
        if not 0 < value < 100:
            raise ValueError(f"“{token}”：D 的百分比必须在 0 到 100 之间（不含端点）")
        if percent.group(2):
            return f"D{percent.group(1)}cc", "dxcc", value, f"D{percent.group(1)}cc", "Gy"
        return f"D{percent.group(1)}", "dx", value, f"D{percent.group(1)}", "Gy"

    level = re.fullmatch(r"V(\d+(?:\.\d+)?)(?:GY)?", upper)
    if level:
        return f"V{level.group(1)}Gy", "vx", float(level.group(1)), f"V{level.group(1)}Gy", "%"

    raise ValueError(f"无法识别的指标 “{token}”；支持 D_x、D_xcc、V_xGy、Dmean、Dmax、Dmin、volume")


def parse_metrics(spec: str) -> list[Metric]:
    tokens = [token for token in re.split(r"[,;\s]+", (spec or "").strip()) if token]
    if not tokens:
        tokens = DEFAULT_METRICS.split(",")
    if len(tokens) > MAX_METRICS:
        raise ValueError(f"自定义指标最多 {MAX_METRICS} 项，当前 {len(tokens)} 项")
    metrics, seen = [], set()
    for token in tokens:
        metric = parse_metric(token)
        if metric[0] not in seen:
            seen.add(metric[0])
            metrics.append(metric)
    return metrics


def evaluate_metric(kind: str, parameter: float | None, values, voxel_cc: float) -> float:
    """Evaluate one metric. ``values`` must be ascending (sorted in place)."""
    count = len(values)
    if kind == "mean":
        return float(values.mean())
    if kind == "max":
        return float(values[-1])
    if kind == "min":
        return float(values[0])
    if kind == "volume":
        return float(count * voxel_cc)
    if kind == "vx":
        return float((values >= parameter).mean() * 100)
    if kind == "dx":
        # Same estimator as the built-in D95 / D2 columns, so typing D95 in the
        # metric box returns exactly the number the fixed column used to show.
        return float(np.percentile(values, 100 - parameter))
    if kind == "dxcc":
        hottest = max(1, min(count, int(round(parameter / voxel_cc))))
        return float(values[-hottest])
    raise ValueError(f"未知指标类型 {kind}")


def metric_notes(kind: str, parameter: float | None, values, voxel_cc: float) -> list[str]:
    if kind != "dxcc":
        return []
    hottest = int(round(parameter / voxel_cc))
    if hottest < 1:
        return [f"{parameter:g} cc 小于单个体素体积 {voxel_cc:.4f} cc，已退化为最热体素剂量"]
    if hottest >= len(values):
        return [f"{parameter:g} cc 超过结构体积 {len(values) * voxel_cc:.2f} cc，结果等于 Dmin"]
    return []


def dvh_with_metrics():
    if core.data_source() == "quick":
        raise ValueError("快速查看未验证计划关联与结构完整性，不提供 DVH")
    a = request.args
    metrics = parse_metrics(a.get("metrics", ""))
    v = core.volume(a["case"], a["series"])
    dose, maximum = v.dose(a["dose"])
    rois = [int(s) for s in a.get("rois", "").split(",") if s]
    if not rois or len(rois) > 8:
        raise ValueError("DVH 请选择 1 至 8 个结构")
    thresholds = np.linspace(0, maximum, 201)
    struct = v.record(a["struct"], "RTSTRUCT")
    voxel_cc = float(np.prod(v.image.GetSpacing())) / 1000
    result = []
    for number in rois:
        roi = next(x for x in struct["rois"] if x["number"] == number)
        raw = dose[v.mask(a["struct"], number)]
        count = len(raw)
        row = {"roi": number, "name": roi["name"],
               "volume_cc": float(count * np.prod(v.image.GetSpacing()) / 1000),
               "coverage": float(np.isfinite(raw).mean()) if count else 0.0,
               "values": {}, "notes": []}
        if roi.get("clipped"):
            row["error"] = "结构已裁剪到 CT 视野；仅显示剩余体积，不计算 DVH 指标"
        elif count and np.isfinite(raw).all():
            values = np.sort(raw)
            row["mean"] = float(values.mean())
            row["d95"] = float(np.percentile(values, 5))
            row["d2"] = float(np.percentile(values, 98))
            row["v20"] = float((values >= 20).mean() * 100)
            row["curve"] = ((count - np.searchsorted(values, thresholds, side="left"))
                            / count * 100).tolist()
            for key, kind, parameter, _, _ in metrics:
                row["values"][key] = evaluate_metric(kind, parameter, values, voxel_cc)
                row["notes"].extend(metric_notes(kind, parameter, values, voxel_cc))
        else:
            row["error"] = "结构为空或部分位于剂量网格外，未计算指标"
        result.append(row)
    if a.get("format") == "csv":
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["ROI", "volume_cc", "coverage"]
                        + [f"{label}_{unit}" for _, _, _, label, unit in metrics] + ["note"])
        for row in result:
            cells = [f"{row['values'][key]:.4f}" if key in row["values"] else ""
                     for key, *_ in metrics]
            writer.writerow([row["name"], f"{row['volume_cc']:.4f}", f"{row['coverage']:.4f}"]
                            + cells + [row.get("error", "")])
        return Response("\ufeff" + stream.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="dvh_metrics.csv"'})
    return jsonify(doses=thresholds.tolist(), structures=result,
                   metrics=[{"key": key, "label": label, "unit": unit}
                            for key, _, _, label, unit in metrics],
                   method="CT 网格结构栅格化；剂量线性插值；未覆盖结构不计算。研究预览，需与 TPS 核对。")


core.app.view_functions["dvh"] = core.api(dvh_with_metrics)


def new_home():
    return send_from_directory(VIEWER / "static", "index_new.html")


core.app.view_functions["home"] = new_home


def configure(dicom_root: Path | None = None, nifti_root: Path | None = None,
              default_source: str = "dicom"):
    if dicom_root is not None:
        add_initial_dicom(dicom_root.resolve())
    if nifti_root is not None:
        add_initial_nifti(nifti_root.resolve())
    core.ROOT = SESSION_DICOM
    core.NIFTI_ROOT = SESSION_NIFTI
    core.app.config["DEFAULT_SOURCE"] = default_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="Optional organized DICOM root to preload")
    parser.add_argument("--nifti-root", type=Path,
                        help="Optional converted NIfTI root to preload")
    parser.add_argument("--default-source", choices=("dicom", "nifti"), default="dicom")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    configure(args.data_root, args.nifti_root, args.default_source)
    print(f"New viewer: http://127.0.0.1:{args.port}/?source={args.default_source}", flush=True)
    print(f"Temporary session: {SESSION}", flush=True)
    core.app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
