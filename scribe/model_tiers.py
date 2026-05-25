"""Model-tier picker with hardware autodetection (F8.11).

Per PLANNING.md F8.11:

  > Model-tier picker with hardware autodetection. Tiers: small (3B /
  > laptop / 8GB GPU or CPU), mid (8–14B / 16GB GPU), large (32–70B /
  > 24GB GPU). Includes a download manager.

This module is **the picker**: it defines the three named tiers as
data, snapshots the local hardware (GPU backend, VRAM, system RAM,
CPU count) using only stdlib + the existing ``scribe.engine``
helpers, and returns a recommendation. It does *not* download
models — that lives in ``scribe.ai_backend.OllamaBackend.pull_model``
(see :func:`scribe.ai_backend.parse_pull_event`), wired to the same
download manager surface.

Concrete model-name recommendations (Llama 3.2 3B, Phi-4 14B, Qwen
2.5 32B, etc.) are deliberately *not* baked in here — that's F8.12.
F8.11 only commits to the tier shapes and the autodetection logic.

Design notes
------------

* **Pure logic, no I/O.** ``detect_hardware`` reads from
  ``scribe.engine`` (which already caches its own gpu_backend probe)
  and from ``os.sysconf`` for system RAM. No model loads, no network.
* **Hardware can be injected.** Every public function takes an
  optional ``HardwareSnapshot``; tests synthesise one with the exact
  VRAM / RAM combination they want to assert about, no monkeypatching
  ``torch`` required.
* **Three-state fit verdict.** Each tier evaluates to "comfortable"
  (recommended), "marginal" (will run but slow / tight), or
  "infeasible" (will OOM or be uselessly slow). The recommendation
  picks the largest comfortable tier; if nothing's comfortable, the
  largest marginal tier; otherwise ``small`` as the floor (the user
  always has *some* option).
* **Apple Silicon nuance.** MPS exposes the unified-memory RAM as
  effective VRAM, so we route MPS through the system-RAM path with a
  ~75% headroom factor (the OS keeps the rest).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Tier vocabulary
# --------------------------------------------------------------------------- #

TIER_SMALL = "small"
TIER_MID = "mid"
TIER_LARGE = "large"

KNOWN_TIER_IDS: tuple[str, ...] = (TIER_SMALL, TIER_MID, TIER_LARGE)

# Apple unified-memory headroom: how much of system RAM is actually
# usable for a single MPS model load. Keeping 25% for the OS / other
# processes is the rough convention.
MPS_USABLE_RAM_FRACTION = 0.75

# Floor for "infeasible" — anything below this much RAM/VRAM and the
# tier won't even load. 0.0 means we don't know (no GPU + sysconf
# failed); we treat that as "can't recommend GPU; hope CPU works".
NO_VRAM = 0.0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelTier:
    """One tier in the small / mid / large picker.

    Sizes are quoted in **billions of parameters** (``parameter_min_b``
    / ``parameter_max_b``) and **GiB** (memory). The ranges intentionally
    overlap a little so a 7B model and an 8B model both find a home.
    """

    id: str
    display_name: str
    description: str
    parameter_label: str         # human-facing string e.g. "3–4B"
    parameter_min_b: float       # inclusive lower bound, billions
    parameter_max_b: float       # inclusive upper bound, billions
    recommended_vram_gb: float   # VRAM for "comfortable" GPU fit
    minimum_vram_gb: float       # VRAM below which the model won't load
    recommended_ram_gb_cpu: float  # RAM for usable CPU-only inference

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Tier definitions, ordered small → large. Keep the list short and
# stable; F8.12 will tack model-name recommendations onto each tier
# without touching this shape.
MODEL_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        id=TIER_SMALL,
        display_name="Small (≈3–4B)",
        description=(
            "Fits comfortably on laptops and 8 GB GPUs. Workable on CPU "
            "with 8 GB+ RAM, though slower. The grounded-theory inductive "
            "opening is fine on this tier — code suggestions are short."
        ),
        parameter_label="3–4B",
        parameter_min_b=1.0,
        parameter_max_b=4.5,
        recommended_vram_gb=8.0,
        minimum_vram_gb=4.0,
        recommended_ram_gb_cpu=8.0,
    ),
    ModelTier(
        id=TIER_MID,
        display_name="Mid (8–14B)",
        description=(
            "Strong analysis on a 16 GB GPU. Feasible on CPU with 32 GB+ "
            "RAM but materially slower; suitable for batch reviews left "
            "running overnight. Usable 'second-coder' quality."
        ),
        parameter_label="8–14B",
        parameter_min_b=7.0,
        parameter_max_b=15.0,
        recommended_vram_gb=16.0,
        minimum_vram_gb=10.0,
        recommended_ram_gb_cpu=32.0,
    ),
    ModelTier(
        id=TIER_LARGE,
        display_name="Large (32–70B)",
        description=(
            "Best quality. Requires a 24 GB+ GPU; CPU-only is "
            "impractical except for tiny corpora with 64 GB+ RAM. "
            "Use for whole-transcript review and the AI second-coder pass."
        ),
        parameter_label="32–70B",
        parameter_min_b=30.0,
        parameter_max_b=80.0,
        recommended_vram_gb=24.0,
        minimum_vram_gb=20.0,
        recommended_ram_gb_cpu=64.0,
    ),
)


@dataclass(frozen=True)
class HardwareSnapshot:
    """A point-in-time snapshot of what hardware Scribe can see.

    All fields are populated by :func:`detect_hardware`; tests build
    snapshots directly to assert tier behaviour at exact VRAM/RAM
    combinations without monkeypatching torch.
    """

    gpu_backend: str             # "cuda" | "rocm" | "mps" | "cpu"
    gpu_name: str                # human label, may be empty
    vram_gb: float               # 0.0 when no GPU / unknown
    system_ram_gb: float         # 0.0 when sysconf failed
    cpu_count: int               # 0 when unknown

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Verdict strings exposed to the UI; stable values written to JSON.
FIT_COMFORTABLE = "comfortable"
FIT_MARGINAL = "marginal"
FIT_INFEASIBLE = "infeasible"

KNOWN_FITS: tuple[str, ...] = (FIT_COMFORTABLE, FIT_MARGINAL, FIT_INFEASIBLE)


@dataclass(frozen=True)
class TierFit:
    """A single tier's fit verdict for a given hardware snapshot.

    Carries enough explanation that the UI can render a full sentence
    without having to re-derive any of the logic on the JS side.
    """

    tier_id: str
    fit: str                     # one of KNOWN_FITS
    reason: str                  # short human sentence
    effective_vram_gb: float = 0.0  # VRAM used for the decision (incl. MPS)


# --------------------------------------------------------------------------- #
# Hardware autodetection
# --------------------------------------------------------------------------- #


def system_ram_gb() -> float:
    """Best-effort total system RAM in GiB. Returns 0.0 when unknown.

    Uses ``os.sysconf`` on POSIX (Linux + macOS); a stdlib-only ctypes
    fallback covers Windows. We deliberately don't depend on
    ``psutil`` — Scribe's hard requirements stay slim.
    """
    # POSIX path: Linux + macOS both expose SC_PHYS_PAGES.
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in getattr(os, "sysconf_names", {}):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 ** 3)
    except (OSError, ValueError):
        pass
    # Windows path via ctypes. Best-effort; if anything goes wrong we
    # bail to 0.0 rather than crashing the whole tier-picker call.
    try:  # pragma: no cover - exercised only on Windows
        import ctypes  # noqa: WPS433

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def detect_hardware() -> HardwareSnapshot:
    """Snapshot the local hardware. Cheap; safe to call repeatedly.

    Pulls ``gpu_backend`` and VRAM via :mod:`scribe.engine` so we share
    the same backend taxonomy ("cuda" / "rocm" / "mps" / "cpu") the
    rest of the app uses. RAM and CPU count come from stdlib.
    """
    # Lazy import: scribe.engine pulls in torch at import time, but
    # this module wants to be safe to import in environments (tests,
    # docs builds) that haven't installed it. Engine is already loaded
    # by the time anyone calls detect_hardware in normal use.
    from . import engine  # noqa: WPS433

    backend = engine.gpu_backend()
    gpu_name = engine._gpu_device_name()

    if backend in ("cuda", "rocm"):
        vram = engine._cuda_vram_gb()
    else:
        vram = 0.0

    return HardwareSnapshot(
        gpu_backend=backend,
        gpu_name=gpu_name,
        vram_gb=float(vram),
        system_ram_gb=system_ram_gb(),
        cpu_count=os.cpu_count() or 0,
    )


# --------------------------------------------------------------------------- #
# Tier fit + recommendation
# --------------------------------------------------------------------------- #


def _effective_vram_gb(snapshot: HardwareSnapshot) -> float:
    """The amount of GPU memory we treat as available for a model load.

    For ``cuda`` / ``rocm`` it's the queried VRAM. For ``mps`` it's a
    fraction of system RAM (Apple unified memory). For ``cpu`` it's
    zero — fall back to RAM-only logic.
    """
    if snapshot.gpu_backend in ("cuda", "rocm"):
        return snapshot.vram_gb
    if snapshot.gpu_backend == "mps":
        return snapshot.system_ram_gb * MPS_USABLE_RAM_FRACTION
    return 0.0


def evaluate_tier(tier: ModelTier, snapshot: HardwareSnapshot) -> TierFit:
    """Return the fit verdict + a short reason for one tier."""
    eff_vram = _effective_vram_gb(snapshot)
    has_gpu = snapshot.gpu_backend in ("cuda", "rocm", "mps")

    if has_gpu and eff_vram >= tier.recommended_vram_gb:
        return TierFit(
            tier_id=tier.id,
            fit=FIT_COMFORTABLE,
            reason=(
                f"{eff_vram:.0f} GB available "
                f"(≥ {tier.recommended_vram_gb:.0f} GB recommended)."
            ),
            effective_vram_gb=eff_vram,
        )

    if has_gpu and eff_vram >= tier.minimum_vram_gb:
        return TierFit(
            tier_id=tier.id,
            fit=FIT_MARGINAL,
            reason=(
                f"{eff_vram:.0f} GB available "
                f"(≥ {tier.minimum_vram_gb:.0f} GB minimum but "
                f"< {tier.recommended_vram_gb:.0f} GB recommended)."
            ),
            effective_vram_gb=eff_vram,
        )

    # GPU absent or too small — see if RAM gives us a CPU fallback.
    if snapshot.system_ram_gb >= tier.recommended_ram_gb_cpu:
        return TierFit(
            tier_id=tier.id,
            fit=FIT_MARGINAL,
            reason=(
                f"GPU unavailable or undersized; "
                f"{snapshot.system_ram_gb:.0f} GB RAM is enough for "
                f"CPU-only inference but will be slow."
            ),
            effective_vram_gb=eff_vram,
        )

    return TierFit(
        tier_id=tier.id,
        fit=FIT_INFEASIBLE,
        reason=(
            f"Needs ≥ {tier.minimum_vram_gb:.0f} GB GPU "
            f"or ≥ {tier.recommended_ram_gb_cpu:.0f} GB RAM; "
            f"have {eff_vram:.0f} GB GPU / "
            f"{snapshot.system_ram_gb:.0f} GB RAM."
        ),
        effective_vram_gb=eff_vram,
    )


def evaluate_all_tiers(snapshot: HardwareSnapshot) -> list[TierFit]:
    """Evaluate every known tier against ``snapshot``, ordered small→large."""
    return [evaluate_tier(t, snapshot) for t in MODEL_TIERS]


def recommend_tier(snapshot: HardwareSnapshot) -> str:
    """Pick the largest tier the hardware can run.

    Decision rule:

      1. Prefer the largest "comfortable" tier.
      2. If nothing's comfortable, pick the largest "marginal" tier.
      3. Otherwise return ``small`` — the user always has *some* option,
         and we'd rather suggest a tier that might be slow than say "no".
    """
    fits = evaluate_all_tiers(snapshot)
    # Walk largest → smallest looking for COMFORTABLE.
    for fit, tier in zip(reversed(fits), reversed(MODEL_TIERS)):
        if fit.fit == FIT_COMFORTABLE:
            return tier.id
    for fit, tier in zip(reversed(fits), reversed(MODEL_TIERS)):
        if fit.fit == FIT_MARGINAL:
            return tier.id
    return TIER_SMALL


def tier_by_id(tier_id: str) -> ModelTier:
    """Look up a tier by id or raise ``ValueError``."""
    for t in MODEL_TIERS:
        if t.id == tier_id:
            return t
    raise ValueError(
        f"Unknown tier {tier_id!r}; known: {KNOWN_TIER_IDS}"
    )


def summarise(snapshot: HardwareSnapshot) -> dict[str, Any]:
    """Build the JSON-serialisable shape the UI / server endpoints want.

    Keeps the wire format colocated with the logic that produces it so
    server tests don't have to reach into private dataclass internals.
    """
    fits = evaluate_all_tiers(snapshot)
    return {
        "hardware": snapshot.to_dict(),
        "tiers": [
            {
                **tier.to_dict(),
                "fit": fit.fit,
                "reason": fit.reason,
                "effective_vram_gb": fit.effective_vram_gb,
            }
            for tier, fit in zip(MODEL_TIERS, fits)
        ],
        "recommended": recommend_tier(snapshot),
    }
