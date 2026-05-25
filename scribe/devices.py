"""Print the device configuration Scribe will use. `python -m scribe.devices`."""

from __future__ import annotations

import platform
import sys

import torch

from .engine import (
    _cuda_vram_gb,
    _diarization_device,
    _torch_device,
    _whisper_device_and_compute,
)


def main() -> int:
    print(f"OS:                 {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"  Device 0:         {torch.cuda.get_device_name(0)} ({_cuda_vram_gb():.1f} GB)")
        except Exception as e:  # noqa: BLE001
            print(f"  (could not query device 0: {e})")
    print(f"MPS available:      {torch.backends.mps.is_available()}")

    w_dev, w_compute = _whisper_device_and_compute()
    print()
    print("Selected backends:")
    print(f"  Whisper (CTranslate2):  device={w_dev:<5} compute={w_compute}")
    print(f"  Alignment (torch):      device={_torch_device()}")
    print(f"  Diarization (pyannote): device={_diarization_device()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
