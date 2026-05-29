#!/usr/bin/env bash
# Start the Scribe server.
#
# Run this to launch (or re-launch) the local UI. Stops anything already
# listening on $PORT so you don't end up with a stale uvicorn serving an
# old copy of the templates — the symptom is "I edited the page but the
# button isn't there." Watches scribe/ + scribe/templates/ via --reload
# so template + Python edits land on the next page load.
#
# Usage:
#   ./run.sh                      # default: 0.0.0.0:8765 (LAN-reachable)
#   HOST=127.0.0.1 ./run.sh       # localhost-only
#   PORT=8000 ./run.sh
#
# Default binds 0.0.0.0 so anything else on the same network — your
# phone, another machine — can reach the UI at
# http://<this-host's-LAN-ip>:8765. There is no auth on the app: anyone
# who can route to this port can read every project + transcript on
# disk. If that's not what you want, set HOST=127.0.0.1 (or run inside
# an SSH tunnel, etc.).
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
HOST="${HOST:-0.0.0.0}"

# Stop anything already listening on the port. Without this, an old
# uvicorn from a previous session keeps serving stale templates and the
# new instance fails with "Address already in use" — the user then sees
# their old UI and assumes the edits didn't take.
stop_existing() {
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti tcp:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v fuser >/dev/null 2>&1; then
    pids=$(fuser -n tcp "${PORT}" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)
  fi
  if [ -n "${pids}" ]; then
    echo "Stopping existing listener on port ${PORT} (pid: ${pids})…" >&2
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    # Give it a moment to release the port.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      sleep 0.2
      if command -v lsof >/dev/null 2>&1; then
        if ! lsof -ti tcp:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
          return 0
        fi
      else
        return 0
      fi
    done
    # Last resort: SIGKILL.
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    sleep 0.3
  fi
}
stop_existing

echo "Scribe → http://${HOST}:${PORT}"
if [ "${HOST}" = "0.0.0.0" ]; then
  # Surface the LAN URLs as well so the user knows what to type into
  # a phone / another laptop. ``hostname -I`` exists on Linux;
  # ``ipconfig getifaddr en0`` is the macOS equivalent. Either way
  # we just print whatever we can find — the headline still says
  # 0.0.0.0 so anyone scripting against this output isn't surprised.
  if command -v hostname >/dev/null 2>&1; then
    lan_ips="$(hostname -I 2>/dev/null || true)"
  else
    lan_ips=""
  fi
  if [ -z "${lan_ips}" ] && command -v ipconfig >/dev/null 2>&1; then
    # macOS: try the usual interface names.
    for iface in en0 en1 en2; do
      ip="$(ipconfig getifaddr "${iface}" 2>/dev/null || true)"
      if [ -n "${ip}" ]; then
        lan_ips="${lan_ips:+${lan_ips} }${ip}"
      fi
    done
  fi
  if [ -n "${lan_ips}" ]; then
    echo "        Reachable on the LAN at:"
    for ip in ${lan_ips}; do
      echo "          http://${ip}:${PORT}"
    done
  fi
  echo "        (no auth — anyone on the network can use the app.)"
fi
exec uvicorn scribe.server:app \
  --host "$HOST" --port "$PORT" \
  --log-level info \
  --reload \
  --reload-dir scribe \
  --reload-include "*.html"
