"""Print the device configuration Scribe will use. `python -m scribe.devices`."""

from __future__ import annotations

import platform
import sys

import torch

from .engine import (
    _cuda_vram_gb,
    _diarization_device,
    _is_rdna2,
    _torch_device,
    _whisper_device_and_compute,
    gpu_backend,
)


def main() -> int:
    backend = gpu_backend()
    print(f"OS:                 {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"GPU backend:        {backend}")

    cuda_ver = getattr(torch.version, "cuda", None)
    hip_ver = getattr(torch.version, "hip", None)
    if cuda_ver:
        print(f"  CUDA build:       {cuda_ver}")
    if hip_ver:
        print(f"  HIP build:        {hip_ver}")

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            print(f"  Device 0:         {name} ({_cuda_vram_gb():.1f} GB)")
        except Exception as e:  # noqa: BLE001
            print(f"  (could not query device 0: {e})")
        if backend == "rocm" and _is_rdna2():
            print("  Note: RDNA 2 detected — auto-applying CT2_CUDA_ALLOCATOR=cub_caching")
    print(f"MPS available:      {torch.backends.mps.is_available()}")

    w_dev, w_compute = _whisper_device_and_compute()
    print()
    print("Selected backends:")
    print(f"  Whisper (CTranslate2):  device={w_dev:<5} compute={w_compute}")
    print(f"  Alignment (torch):      device={_torch_device()}")
    print(f"  Diarization (pyannote): device={_diarization_device()}")

    # Package versions matter when something breaks; pyannote/whisperx are
    # particularly sensitive to huggingface_hub and transformers majors.
    print()
    print("Package versions:")
    for name in (
        "whisperx",
        "faster_whisper",
        "ctranslate2",
        "transformers",
        "huggingface_hub",
        "pyannote.audio",
        "torchaudio",
        "nemo_toolkit",
    ):
        try:
            from importlib.metadata import version, PackageNotFoundError
            try:
                v = version(name)
            except PackageNotFoundError:
                v = "(not installed)"
        except Exception as e:  # noqa: BLE001
            v = f"(error: {e})"
        print(f"  {name:<20} {v}")

    # Optional Parakeet engine probe.
    try:
        from .parakeet import nemo_available
        ok, err = nemo_available()
        print()
        print(f"Parakeet engine:    {'available' if ok else 'unavailable'}")
        if not ok and err:
            print(f"  reason:           {err}")
    except Exception as e:  # noqa: BLE001
        print(f"Parakeet engine:    error probing — {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
