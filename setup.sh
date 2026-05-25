#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- --realign: bring an existing venv back in line with requirements.txt ---
# Use this when `python -m scribe.devices` shows package versions outside the
# requirements.txt pins (typically because a third-party install upgraded
# huggingface_hub / transformers / ctranslate2 transitively).
if [ "${1:-}" = "--realign" ]; then
  if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh (without --realign) first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo ">> Realigning drift-prone packages against requirements.txt"
  # Only force the three packages that actually drift in practice. We skip
  # the full `pip install -r requirements.txt` because whisperx's stale
  # metadata (ctranslate2<4.5) makes pip's resolver refuse the wider set —
  # so we install these three with --no-deps to avoid the resolver chasing
  # transitive conflicts that don't apply to runtime behaviour.
  #   - huggingface_hub: gets bumped to 1.x by NeMo / new transformers,
  #     breaking pyannote 3.4's use_auth_token kwarg
  #   - transformers 5.x: requires hf_hub>=1.5 (incompatible with pyannote)
  #   - ctranslate2 4.4: links cuDNN 8 (gone from torch 2.6+ wheels);
  #     4.7.x ships cuDNN 9 and keeps the Whisper APIs whisperx uses
  pip install --upgrade --force-reinstall --no-deps \
    "huggingface_hub>=0.30,<1.0" \
    "transformers>=4.40,<5.0" \
    "ctranslate2>=4.6,<4.8"
  echo
  echo ">> Verifying device configuration and versions"
  python -m scribe.devices || true
  echo
  echo ">> Realign complete. Run ./run.sh and try the transcription again."
  echo ">> If something still looks off, paste the version report above."
  exit 0
fi

# --- --rocm: switch an existing venv to AMD ROCm acceleration ---
# Run AFTER ./setup.sh has created the venv. Replaces the CUDA torch wheels
# with the ROCm 6.3 build and installs the CTranslate2 ROCm wheel from
# the v4.7.2 GitHub release (it's not on PyPI). Linux + RDNA 3/4 are
# first-class; RDNA 2 works with auto-applied env-var workarounds.
if [ "${1:-}" = "--rocm" ]; then
  if [ "$OS" != "Linux" ]; then
    echo "AMD ROCm support is Linux-only at the moment." >&2
    echo "On Windows AMD, the path forward is whisper.cpp Vulkan, which" >&2
    echo "Scribe doesn't integrate yet. See docs/research/amd-rocm-research.md." >&2
    exit 1
  fi
  if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh first to create it." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  CT2_VERSION="${SCRIBE_CT2_ROCM_VERSION:-4.7.2}"
  ROCM_TORCH_INDEX="${SCRIBE_ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.3}"

  echo ">> Installing PyTorch ROCm 6.3 wheel (replacing CUDA build)"
  pip install --upgrade --force-reinstall --index-url "$ROCM_TORCH_INDEX" torch torchaudio

  echo
  echo ">> Fetching CTranslate2 ROCm wheel v$CT2_VERSION"
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT
  WHEEL_ZIP_URL="https://github.com/OpenNMT/CTranslate2/releases/download/v${CT2_VERSION}/rocm-python-wheels-Linux.zip"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; install it and retry." >&2
    exit 1
  fi
  if ! curl -L --fail -o "$TMPDIR/ct2-rocm.zip" "$WHEEL_ZIP_URL"; then
    echo "Could not download $WHEEL_ZIP_URL" >&2
    echo "Check the version (SCRIBE_CT2_ROCM_VERSION) or your network." >&2
    exit 1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip not found; install it (e.g. sudo apt install unzip) and retry." >&2
    exit 1
  fi
  unzip -q "$TMPDIR/ct2-rocm.zip" -d "$TMPDIR/ct2"
  PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
  WHEEL_FILE="$(ls "$TMPDIR/ct2"/ctranslate2-*-${PY_TAG}-*-linux_x86_64.whl 2>/dev/null | head -n1)"
  if [ -z "$WHEEL_FILE" ]; then
    echo "Could not find a CT2 ROCm wheel for $PY_TAG in the release zip." >&2
    echo "Available files:" >&2
    ls "$TMPDIR/ct2" >&2
    exit 1
  fi
  echo ">> Installing $(basename "$WHEEL_FILE")"
  pip install --upgrade --force-reinstall --no-deps "$WHEEL_FILE"

  echo
  echo ">> Verifying device configuration"
  python -m scribe.devices || true

  cat <<EOF

>> AMD ROCm setup complete.
>> Tier 1 (RDNA 3/4): you should be ready to go.
>> Tier 2 (RDNA 2 / RX 6000):  Scribe will auto-apply
>>   CT2_CUDA_ALLOCATOR=cub_caching for you.
>>   You may also need:
>>     export HSA_OVERRIDE_GFX_VERSION=10.3.0
>>   See docs/research/amd-rocm-research.md for hardware notes.
>> Run ./run.sh and try a transcription. If you see errors, paste the
>> output of \`python -m scribe.devices\` plus the failure log.
EOF
  exit 0
fi

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
