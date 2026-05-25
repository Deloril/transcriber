#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo ">> Detected: $OS $ARCH"

# --- python check ---
# WhisperX requires Python 3.10 or 3.11. If a venv already exists we honor its
# interpreter; otherwise we hunt for a supported python3.X on PATH.
pick_python() {
  for candidate in python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

print_python_install_hint() {
  case "$OS" in
    Darwin) echo "  brew install python@3.11" >&2 ;;
    Linux)  echo "  sudo apt install python3.11 python3.11-venv  (Debian/Ubuntu)" >&2
            echo "  sudo dnf install python3.11                    (Fedora/RHEL)" >&2 ;;
  esac
}

if [ -d .venv ]; then
  PYTHON=.venv/bin/python
else
  if PYTHON=$(pick_python); then
    :
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "python3 not found." >&2
    print_python_install_hint
    exit 1
  fi
fi

PYV=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$PYV" in
  3.10|3.11) ;;
  *)
    echo "Python $PYV detected ($PYTHON). WhisperX requires Python 3.10 or 3.11." >&2
    if [ -d .venv ]; then
      echo "Your existing .venv was built with $PYV. Delete it and rerun:" >&2
      echo "  rm -rf .venv && ./setup.sh" >&2
    else
      echo "Install a supported Python and rerun:" >&2
      print_python_install_hint
    fi
    exit 1
    ;;
esac

# --- ffmpeg check ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found." >&2
  case "$OS" in
    Darwin) echo "  Install: brew install ffmpeg" >&2 ;;
    Linux)  echo "  Install: sudo apt install ffmpeg          (Debian/Ubuntu)" >&2
            echo "       or: sudo dnf install ffmpeg          (Fedora/RHEL)" >&2 ;;
  esac
  exit 1
fi

# --- venv ---
if [ ! -d .venv ]; then
  echo ">> Creating venv with $PYTHON (Python $PYV)"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Upgrading pip"
pip install --upgrade pip wheel setuptools

# --- platform-specific PyTorch index ---
# By default, pip on Linux x86_64 grabs the CUDA wheel from PyPI. That's fine
# if you have an NVIDIA GPU. For CPU-only Linux machines, set SCRIBE_TORCH=cpu.
TORCH_INDEX_FLAGS=()
if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
  case "${SCRIBE_TORCH:-auto}" in
    cpu)
      echo ">> Installing CPU-only PyTorch (SCRIBE_TORCH=cpu)"
      TORCH_INDEX_FLAGS=(--index-url https://download.pytorch.org/whl/cpu)
      ;;
    cu121)
      echo ">> Installing PyTorch built for CUDA 12.1 (SCRIBE_TORCH=cu121)"
      TORCH_INDEX_FLAGS=(--index-url https://download.pytorch.org/whl/cu121)
      ;;
    cu124)
      echo ">> Installing PyTorch built for CUDA 12.4 (SCRIBE_TORCH=cu124)"
      TORCH_INDEX_FLAGS=(--index-url https://download.pytorch.org/whl/cu124)
      ;;
    auto|*) ;;  # use whatever pip resolves from PyPI
  esac
fi

if [ "${#TORCH_INDEX_FLAGS[@]}" -gt 0 ]; then
  echo ">> Installing torch/torchaudio first"
  pip install "${TORCH_INDEX_FLAGS[@]}" torch torchaudio
fi

echo ">> Installing requirements (this takes a few minutes)"
pip install -r requirements.txt

mkdir -p uploads outputs models

echo
echo ">> Verifying device configuration"
python -m scribe.devices || true

cat <<EOF

>> Done.
>> Run with: ./run.sh
>> Then open http://localhost:8765
>> Check device config any time with: source .venv/bin/activate && python -m scribe.devices

Optional: enable NVIDIA Parakeet (English-only, ~30× faster than Whisper):
  source .venv/bin/activate
  pip install -r requirements-parakeet.txt
Then select a parakeet-* model in the UI's model dropdown.
EOF
