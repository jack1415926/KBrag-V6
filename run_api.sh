#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$PYTHON_BIN" -m uvicorn api_server:app --host "$HOST" --port "$PORT" --workers 1
