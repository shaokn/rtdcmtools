#!/usr/bin/env bash
# Local-only RT DICOM viewer. Keep this terminal open while using the browser.
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
[[ -d "$PACKAGE_ROOT/../nifti_data" ]] && DEFAULT_NIFTI="$PACKAGE_ROOT/../nifti_data" || DEFAULT_NIFTI="$PACKAGE_ROOT/data/nifti_data"

PORT="${1:-8765}"
DATA_ROOT="${2:-$DEFAULT_DICOM}"
NIFTI_ROOT="${3:-$DEFAULT_NIFTI}"

printf 'Viewer URL: http://127.0.0.1:%s\n' "$PORT"
printf 'DICOM root: %s\n' "$DATA_ROOT"
printf 'NIfTI root: %s\n' "$NIFTI_ROOT"
printf 'Ctrl+C stops the server.\n\n'

exec "$PYTHON_BIN" -u "$PACKAGE_ROOT/viewer/server.py" \
    --port "$PORT" --data-root "$DATA_ROOT" --nifti-root "$NIFTI_ROOT"
