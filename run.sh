#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "venv not found. Run ./setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Avoid HF telemetry / transformers warnings
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"

exec uvicorn scribe.server:app --host "$HOST" --port "$PORT" --log-level info
