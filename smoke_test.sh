#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TOKEN="${KAFU_API_TOKEN:-$(cd "$(dirname "$0")" && "$PYTHON_BIN" -c 'from dotenv import load_dotenv; import os; load_dotenv(".env"); print(os.getenv("KAFU_API_TOKEN", ""))')}"
: "${TOKEN:?Set KAFU_API_TOKEN in the environment or .env}"

echo "== /health =="
HEALTH_JSON="$(curl -sS "$BASE_URL/health")"
echo "$HEALTH_JSON"
printf '%s' "$HEALTH_JSON" | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status")=="ok", d'
echo

echo "== /chat smoke =="
CHAT_JSON="$(curl -sS -X POST "$BASE_URL/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"椅子的扶手使用一段时间后为什么会松动？","session_id":"demo"}')"
echo "$CHAT_JSON"
printf '%s' "$CHAT_JSON" | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); assert d.get("code")==0 and d.get("data",{}).get("answer"), d'
echo

echo "smoke_test=OK"
