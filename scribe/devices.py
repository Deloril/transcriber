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
    gpu_arch_name,
    gpu_backend,
)
from .rocm_install import (
    ct2_drift_message,
    installed_ct2_version,
    pinned_ct2_rocm_version,
    rocm_wheel_fallback_urls,
)


def _linux_distro() -> str | None:
    """Linux distribution pretty-name (e.g. ``Ubuntu 24.04.4 LTS``) or None.

    Used in the device report so AMD/ROCm support tickets show the distro
    — ROCm support varies meaningfully across Ubuntu / RHEL / Fedora /
    Arch (G1.3); also useful on CUDA when reporting kernel-driver issues.

    Returns None on non-Linux systems, when ``platform.freedesktop_os_release``
    is unavailable (Python < 3.10), or when ``/etc/os-release`` can't be
    read. Empty pretty-names normalise to None.
    """
    if platform.system() != "Linux":
        return None
    fn = getattr(platform, "freedesktop_os_release", None)
    if fn is None:
        return None
    try:
        info = fn()
    except (OSError, FileNotFoundError):
        return None
    pretty = info.get("PRETTY_NAME") or info.get("NAME")
    return pretty or None


def main() -> int:
    backend = gpu_backend()
    print(f"OS:                 {platform.system()} {platform.release()} ({platform.machine()})")
    distro = _linux_distro()
    if distro:
        print(f"  Distro:           {distro}")
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
        # G1.3: surface the gfx target on AMD/ROCm — it's the one identifier
        # CTranslate2 / ROCm upstream maintainers ask for first when triaging.
        if backend == "rocm":
            arch = gpu_arch_name()
            if arch:
                print(f"  GFX target:       {arch}")
        if backend == "rocm" and _is_rdna2():
            print("  Note: RDNA 2 detected — auto-applying CT2_CUDA_ALLOCATOR=cub_caching")
        # G2.1: surface the pinned CT2 ROCm wheel version (and warn on drift)
        # only when the active backend is ROCm — on CUDA / MPS / CPU the
        # ctranslate2 build that's installed is unrelated to the ROCm pin.
        if backend == "rocm":
            pinned = pinned_ct2_rocm_version()
            installed = installed_ct2_version()
            shown = installed or "not installed"
            print(f"  CT2 ROCm pin:     v{pinned} (installed: {shown})")
            drift = ct2_drift_message(installed=installed, pinned=pinned)
            if drift:
                print(f"  ⚠  drift:          {drift}")
            # G2.2: when the user has configured fallback mirrors,
            # surface them so they can verify their list before re-running
            # ./setup.sh --rocm. Empty by default; we never ship a default
            # mirror in this repo (no infrastructure for one yet).
            fallbacks = rocm_wheel_fallback_urls()
            if fallbacks:
                print(f"  CT2 wheel mirrors: {len(fallbacks)} configured (fallbacks)")
                for u in fallbacks:
                    print(f"    - {u}")
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
