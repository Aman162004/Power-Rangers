#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SERVICE_NAME="${SERVICE_NAME:-power-rangers}"
REGION="${REGION:-asia-south1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python3 or python is required to auto-generate JWT_SECRET_KEY" >&2
    exit 1
  fi
fi

if [[ -z "${JWT_SECRET_KEY:-}" ]]; then
  JWT_SECRET_KEY="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(48))')"
fi

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 4 \
  --concurrency 1 \
  --max-instances 1 \
  --min-instances 1 \
  --timeout 900 \
  --set-env-vars "JWT_SECRET_KEY=$JWT_SECRET_KEY,AUTH_DB_PATH=/tmp/auth.db,ENABLE_DUMMY_FALLBACK=false,OPERATIONAL_DIR=/tmp/power-rangers/operational,SLDC_LOAD_CACHE_DIR=/tmp/power-rangers/operational/raw,OPENMETEO_CACHE_NAME=/tmp/power-rangers/openmeteo"
