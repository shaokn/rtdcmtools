#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${RTDCMTOOLS_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in "$PACKAGE_ROOT/.venv/bin/python" "$PACKAGE_ROOT/.pyauto/bin/python" \
        "$PACKAGE_ROOT/../../../.pyauto/bin/python"; do
        if [[ -x "$candidate" ]]; then PYTHON_BIN="$candidate"; break; fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3 || true)"; fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Python not found. Create .venv or set RTDCMTOOLS_PYTHON.\n' >&2
    exit 1
fi

[[ -d "$PACKAGE_ROOT/../organized_dicom" ]] && DEFAULT_DICOM="$PACKAGE_ROOT/../organized_dicom" || DEFAULT_DICOM="$PACKAGE_ROOT/data/organized_dicom"
[[ -d "$PACKAGE_ROOT/../nifti_fractions" ]] && DEFAULT_NIFTI="$PACKAGE_ROOT/../nifti_fractions" || DEFAULT_NIFTI="$PACKAGE_ROOT/data/nifti_fractions"

PORT="${1:-8768}"
DICOM_ROOT="${2:-$DEFAULT_DICOM}"
NIFTI_ROOT="${3:-$DEFAULT_NIFTI}"

printf 'New viewer URL: http://127.0.0.1:%s/?source=nifti\n' "$PORT"
printf 'Initial DICOM root: %s\n' "$DICOM_ROOT"
printf 'Initial NIfTI root: %s\n' "$NIFTI_ROOT"
printf 'Added folders are temporary for this server session. Ctrl+C stops the server.\n\n'

exec "$PYTHON_BIN" -u "$PACKAGE_ROOT/server_new.py" \
    --port "$PORT" --data-root "$DICOM_ROOT" --nifti-root "$NIFTI_ROOT" \
    --default-source nifti
