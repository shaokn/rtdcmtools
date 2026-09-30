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

PORT="${1:-8768}"
# Bind address. Keep the default for this machine only; set RTDCMTOOLS_HOST to
# 0.0.0.0 to accept connections from the local network.
HOST="${RTDCMTOOLS_HOST:-127.0.0.1}"
ARGS=(--host "$HOST" --port "$PORT" --default-source dicom)
if [[ -n "${2:-}" ]]; then
    ARGS+=(--data-root "$2")
fi
if [[ -n "${3:-}" ]]; then
    ARGS+=(--nifti-root "$3")
fi

if [[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]]; then
    printf 'New viewer URL: http://127.0.0.1:%s/?source=dicom\n' "$PORT"
else
    # 0.0.0.0 and :: are bind keywords, not browsing addresses; the concrete URLs
    # are printed by server_new.py right below.
    printf 'Bind address: %s. Concrete URLs are printed below by the server.\n' "$HOST"
    printf 'Other machines must use the LAN address of this computer, and the firewall\n'
    printf 'has to allow port %s. The viewer has no authentication.\n' "$PORT"
fi
if [[ -z "${2:-}${3:-}" ]]; then
    printf 'Case library starts empty. Use + in the browser to add folders.\n'
else
    [[ -n "${2:-}" ]] && printf 'Initial DICOM root: %s\n' "$2"
    [[ -n "${3:-}" ]] && printf 'Initial NIfTI root: %s\n' "$3"
fi
printf 'Added folders are temporary for this server session. Ctrl+C stops the server.\n\n'

exec "$PYTHON_BIN" -u "$PACKAGE_ROOT/server_new.py" "${ARGS[@]}"
