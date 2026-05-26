"""Print the device configuration Scribe will use. `python -m scribe.devices`."""

from __future__ import annotations

import platform
import sys

import torch

from .engine import (
    HSA_OVERRIDE_RDNA2_VALUE,
    _cuda_vram_gb,
    _diarization_device,
    _is_rdna2,
    _torch_device,
    _whisper_device_and_compute,
    gpu_arch_name,
    gpu_backend,
    rocm_lstm_dropout_patch_active,
    rocm_lstm_dropout_patch_explanation,
)
from .rocm_install import (
    ct2_drift_message,
    installed_ct2_version,
    pinned_ct2_rocm_version,
    rocm_wheel_fallback_urls,
)
from .rocm_distro import (
    detected_tier_line,
    tier_for_system,
)


def _os_release_info() -> dict[str, str] | None:
    """Return the parsed ``/etc/os-release`` mapping, or None.

    Wraps :func:`platform.freedesktop_os_release` with the same defensive
    catches as :func:`_linux_distro` (Python < 3.10, missing file, OSError).
    Returns None on non-Linux. Used to feed the G2.3 distro-tier classifier
    *and* the legacy pretty-name display.
    """
    if platform.system() != "Linux":
        return None
    fn = getattr(platform, "freedesktop_os_release", None)
    if fn is None:
        return None
    try:
        return dict(fn())
    except (OSError, FileNotFoundError):
        return None


def _linux_distro() -> str | None:
    """Linux distribution pretty-name (e.g. ``Ubuntu 24.04.4 LTS``) or None.

    Used in the device report so AMD/ROCm support tickets show the distro
    — ROCm support varies meaningfully across Ubuntu / RHEL / Fedora /
    Arch (G1.3); also useful on CUDA when reporting kernel-driver issues.

    Returns None on non-Linux systems, when ``platform.freedesktop_os_release``
    is unavailable (Python < 3.10), or when ``/etc/os-release`` can't be
    read. Empty pretty-names normalise to None.
    """
    info = _os_release_info()
    if not info:
        return None
    pretty = info.get("PRETTY_NAME") or info.get("NAME")
    return pretty or None


def _rocm_distro_tier() -> tuple[str, str | None]:
    """G2.3: classify the active distro into a ROCm support tier.

    Returns ``(tier, pretty_name_or_None)``. On non-Linux this returns
    ``("unsupported", None)`` — surfaced only when the active backend is
    ROCm (callers gate on ``backend == "rocm"`` so it doesn't appear on
    CUDA / MPS / CPU). Pure-ish wrapper around :func:`tier_for_system`.
    """
    info = _os_release_info()
    pretty = (info.get("PRETTY_NAME") if info else None) or (
        info.get("NAME") if info else None
    )
    tier = tier_for_system(platform.system(), info)
    return tier, pretty or None


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
        # G4.1: surface the cub_caching workaround state for RDNA 2 owners
        # so support tickets show whether the env var was auto-applied,
        # user-overridden, or still unset (worker race condition).
        if backend == "rocm" and _is_rdna2():
            import os as _os
            allocator = _os.environ.get("CT2_CUDA_ALLOCATOR")
            if allocator == "cub_caching":
                print("  Allocator:        CT2_CUDA_ALLOCATOR=cub_caching (auto, RDNA 2 workaround)")
            elif allocator:
                print(f"  Allocator:        CT2_CUDA_ALLOCATOR={allocator} (user-overridden)")
            else:
                print("  Allocator:        CT2_CUDA_ALLOCATOR unset — call apply_rocm_runtime_workarounds() before CT2 import")
        # G4.2: surface the HSA_OVERRIDE_GFX_VERSION state on ROCm. RDNA 2
        # dies that aren't gfx1030 (i.e. RX 6700 / 6600 / 6500 / 6400 and
        # the RDNA 2 APUs) need ``HSA_OVERRIDE_GFX_VERSION=10.3.0`` exported
        # *before* HIP initialises, so we surface it as a recommendation
        # (we can't auto-set it from Python — by the time this module loads,
        # torch has already brought the HIP runtime up). When the user has
        # already set the variable we just echo the value so support
        # bundles show the active configuration; on gfx1030 (the one
        # officially-supported RDNA 2 target) the line is omitted.
        if backend == "rocm":
            import os as _os
            hsa_value = _os.environ.get("HSA_OVERRIDE_GFX_VERSION")
            arch_for_hsa = gpu_arch_name()
            if hsa_value:
                print(
                    f"  HSA override:     HSA_OVERRIDE_GFX_VERSION={hsa_value} (user-set)"
                )
            elif _is_rdna2() and arch_for_hsa and arch_for_hsa != "gfx1030":
                print(
                    f"  HSA override:     HSA_OVERRIDE_GFX_VERSION unset — "
                    f"recommend export HSA_OVERRIDE_GFX_VERSION={HSA_OVERRIDE_RDNA2_VALUE} for {arch_for_hsa}"
                )
        # G2.3: classify the Linux distro against AMD's official matrix.
        # Only meaningful on ROCm — on CUDA / MPS / CPU the user doesn't
        # care which Radeon-supporting distro they're on.
        if backend == "rocm":
            tier, pretty = _rocm_distro_tier()
            print(f"  {detected_tier_line(pretty, tier)}")
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
            # G3.1: confirm the pyannote LSTM dropout MIOpen workaround
            # is in the install (it fires automatically on diarization
            # load — surfacing the line gives a ROCm user something
            # actionable to copy into a pyannote-audio #1995 support
            # thread instead of "I think the patch is there").
            if rocm_lstm_dropout_patch_active():
                print(
                    f"  LSTM dropout patch: active ({rocm_lstm_dropout_patch_explanation()})"
                )
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
