"""Linux distro support tiers for the AMD/ROCm path (G2.3).

CTranslate2's ROCm wheels and PyTorch's ROCm 6.3 build are *Linux only*,
and AMD only officially supports a narrow distro matrix for consumer
Radeon: **Ubuntu 22.04 + 24.04** and **RHEL 9 + 10**. Everything else
(Fedora, Arch, Debian, openSUSE, Alpine, NixOS) works in practice via
upstream packages and ``HSA_OVERRIDE_GFX_VERSION``, but isn't in AMD's
support matrix — when something breaks, you're on your own with the
community.

This module is the *single source of truth* for which distro lands in
which tier. We surface the classification in two places:

* ``scribe.devices`` shows the user their tier next to the detected
  distro so support-ticket triage is a one-liner instead of
  "what distro / version / architecture / GPU?".
* ``setup.sh --rocm`` reads it (via the same Python helper) to decide
  whether to print a "you're on a best-effort distro, please help us
  triage if it breaks" warning before downloading the wheel.

Pure functions only — no I/O, no platform calls — so the tests can
exhaustively cover the matrix without monkeypatching ``/etc/os-release``.
The platform-call side ("which distro am I actually on?") lives in
:func:`scribe.devices._linux_distro` (G1.3) and is kept separate so
that whichever os-release dict shape you give us, we can classify it.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional


#: Support-tier vocabulary. The values are stable and may be persisted in
#: log lines / support tickets, so don't rename them lightly.
#:
#: * ``"first-class"`` — AMD officially supports this distro for ROCm and
#:   we test on it. RDNA 3/4 Tier 1.
#: * ``"supported"`` — AMD officially supports this distro but we don't
#:   actively test on it. Wheels and the broad install path should work.
#: * ``"best-effort"`` — Not in AMD's matrix but works in practice via
#:   upstream packages. Users should expect to need
#:   ``HSA_OVERRIDE_GFX_VERSION`` and may hit distro-specific edge cases.
#: * ``"unsupported"`` — Tier 3 (Windows AMD, macOS) or unrecognised OS.
#:   ``setup.sh --rocm`` already refuses to run on non-Linux; this
#:   classification is for reporting only.
#: * ``"unknown"`` — Linux, but the distro id is missing or unrecognised.
#:   Treated as best-effort by callers but flagged separately so we can
#:   prompt for the actual id in support tickets.
Tier = Literal[
    "first-class",
    "supported",
    "best-effort",
    "unsupported",
    "unknown",
]


# Distro IDs are the lowercase ``ID`` field from ``/etc/os-release``.
# ``ID_LIKE`` is *not* consulted automatically — Pop!_OS reports
# ``ID_LIKE=ubuntu debian`` but we'd rather classify it as best-effort
# than implicitly call it first-class. Callers can pass an explicit
# fallback via :func:`classify_os_release`.
_FIRST_CLASS_DISTROS: frozenset[str] = frozenset({"ubuntu"})
_FIRST_CLASS_UBUNTU_VERSIONS: frozenset[str] = frozenset({"22.04", "24.04"})

# RHEL ships as ``ID=rhel``; clones (Rocky, AlmaLinux) are best-effort
# because AMD doesn't list them. CentOS Stream is also best-effort.
_SUPPORTED_RHEL_VERSIONS: frozenset[str] = frozenset({"9", "10"})

# Best-effort: well-known Linux distros that work but aren't in AMD's
# matrix. We include the major derivatives explicitly rather than relying
# on ID_LIKE to keep classification deterministic.
_BEST_EFFORT_DISTROS: frozenset[str] = frozenset({
    "fedora",
    "arch",
    "debian",
    "opensuse",
    "opensuse-leap",
    "opensuse-tumbleweed",
    "manjaro",
    "endeavouros",
    "pop",        # Pop!_OS
    "linuxmint",
    "elementary",
    "nixos",
    "alpine",
    "rocky",
    "almalinux",
    "centos",
})


def normalise_id(value: str | None) -> str:
    """Lowercase, strip, and tolerate ``None`` for an os-release field.

    ``/etc/os-release`` allows quoted values; ``platform.freedesktop_os_release``
    already strips the quotes for us, but we still normalise case + whitespace
    to keep the classifier branch-free. Returns ``""`` for missing/blank input.
    """
    if not value:
        return ""
    return value.strip().lower()


def normalise_version(value: str | None) -> str:
    """Lowercase + strip an os-release VERSION_ID field.

    Returns ``""`` for missing/blank input. Doesn't try to parse semver —
    Ubuntu reports ``"24.04"``, RHEL reports ``"9.7"``, Arch reports
    nothing at all (rolling release). The caller compares against the
    relevant set.
    """
    if not value:
        return ""
    return value.strip().lower()


def _ubuntu_major_minor(version_id: str) -> str:
    """``"24.04.4"`` → ``"24.04"``; tolerant of extra dot-segments."""
    parts = version_id.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version_id


def _rhel_major(version_id: str) -> str:
    """``"9.7"`` → ``"9"``; ``"10"`` → ``"10"``."""
    return version_id.split(".", 1)[0] if version_id else ""


def classify(os_id: str | None, version_id: str | None = None) -> Tier:
    """Return the support tier for a Linux distro id + optional version.

    Pure function. Empty / unknown ids → ``"unknown"`` (Linux but we
    couldn't identify it). Use :func:`classify_os_release` if you have a
    full os-release dict and want the same logic with sensible fallbacks.

    >>> classify("ubuntu", "24.04")
    'first-class'
    >>> classify("ubuntu", "20.04")
    'best-effort'
    >>> classify("rhel", "9.7")
    'supported'
    >>> classify("fedora", "41")
    'best-effort'
    >>> classify("", None)
    'unknown'
    """
    nid = normalise_id(os_id)
    nver = normalise_version(version_id)

    if not nid:
        return "unknown"

    if nid in _FIRST_CLASS_DISTROS:
        # Ubuntu + supported version → first-class; otherwise best-effort
        # (we don't refuse to run; we just note that it isn't in AMD's
        # matrix).
        if _ubuntu_major_minor(nver) in _FIRST_CLASS_UBUNTU_VERSIONS:
            return "first-class"
        return "best-effort"

    if nid == "rhel":
        if _rhel_major(nver) in _SUPPORTED_RHEL_VERSIONS:
            return "supported"
        return "best-effort"

    if nid in _BEST_EFFORT_DISTROS:
        return "best-effort"

    return "unknown"


def classify_os_release(info: Mapping[str, str] | None) -> Tier:
    """Classify a freedesktop os-release mapping.

    Accepts the dict returned by ``platform.freedesktop_os_release()``
    directly, or any mapping with the same shape. Returns ``"unsupported"``
    for an empty / None mapping (treated as "we couldn't read os-release
    at all" — i.e. probably not Linux). For a Linux-shaped dict with an
    unrecognised ``ID``, returns ``"unknown"``.

    Falls back to ``ID_LIKE`` for derivatives that aren't in our explicit
    set: this lets, e.g., ``ID=foo ID_LIKE=debian`` classify as
    best-effort instead of unknown. ``ID_LIKE`` is *space-separated* per
    the spec.
    """
    if not info:
        return "unsupported"

    direct = classify(info.get("ID"), info.get("VERSION_ID"))
    if direct != "unknown":
        return direct

    id_like = normalise_id(info.get("ID_LIKE"))
    if not id_like:
        return "unknown"

    for token in id_like.split():
        token = token.strip()
        if not token:
            continue
        # ID_LIKE is matched without VERSION_ID — derivatives don't
        # share the parent's version semantics.
        derived = classify(token, None)
        if derived in ("first-class", "supported", "best-effort"):
            # Even Ubuntu-derived distros (Pop!_OS, Mint) only get
            # best-effort, never first-class — reserve first-class for
            # actual Ubuntu point releases that AMD certifies.
            return "best-effort"
    return "unknown"


def tier_for_system(
    system: str,
    os_release: Mapping[str, str] | None = None,
) -> Tier:
    """Top-level classifier including the OS-name short-circuit.

    ``system`` is the value of :func:`platform.system` —
    ``"Linux"`` / ``"Darwin"`` / ``"Windows"``. Non-Linux returns
    ``"unsupported"`` regardless of ``os_release`` because ROCm wheels
    are Linux-only on the official PyTorch index.

    On Linux, defers to :func:`classify_os_release`.
    """
    if normalise_id(system) != "linux":
        return "unsupported"
    return classify_os_release(os_release)


def tier_explanation(tier: Tier) -> str:
    """One-line human-readable explanation for a tier label.

    Stable strings; intended for the ``scribe.devices`` report and the
    ``setup.sh --rocm`` warning. Don't translate to f-strings without
    locking the wording — support docs reference these phrases.
    """
    if tier == "first-class":
        return "AMD officially supports this distro for ROCm; tested by Scribe"
    if tier == "supported":
        return "AMD officially supports this distro for ROCm"
    if tier == "best-effort":
        return (
            "not in AMD's official ROCm matrix; works in practice via "
            "upstream packages"
        )
    if tier == "unsupported":
        return "no ROCm wheels available for this OS (Linux only)"
    # tier == "unknown"
    return "Linux distro not recognised — treat as best-effort"


def is_rocm_capable(tier: Tier) -> bool:
    """True if the user can plausibly run the ROCm install path.

    ``"unsupported"`` callers — Windows / macOS — should not even attempt
    ``setup.sh --rocm`` (the script already refuses on non-Linux). Every
    other tier is "go for it; here's how confident we are".
    """
    return tier != "unsupported"


# Quick lookup the README + setup.sh can render. Order matters — first
# entry is the tier we want users to land on if they have a choice. The
# strings come straight from the support-matrix discussion in
# docs/research/amd-rocm-research.md.
SUPPORT_MATRIX: tuple[tuple[Tier, str, tuple[str, ...]], ...] = (
    ("first-class", "Ubuntu 22.04, Ubuntu 24.04",
     ("ubuntu 22.04", "ubuntu 24.04")),
    ("supported", "RHEL 9, RHEL 10",
     ("rhel 9", "rhel 10")),
    ("best-effort", "Fedora, Arch, Debian, openSUSE, derivatives",
     ("fedora", "arch", "debian", "opensuse")),
    ("unsupported", "Windows, macOS (no ROCm wheels exist)",
     ("windows", "darwin")),
)


def support_matrix_lines() -> list[str]:
    """Render the support matrix as printable lines (one per tier).

    Used by ``python -m scribe.devices --rocm-distros`` if/when we add
    that flag, and by the README generator. Pure helper so that the
    table is in one place.
    """
    lines: list[str] = []
    for tier, distros, _ids in SUPPORT_MATRIX:
        lines.append(f"  {tier:<12} {distros}")
    return lines


def detected_tier_line(distro_pretty: Optional[str], tier: Tier) -> str:
    """Render the user's detected distro + tier as a one-liner.

    Used in :func:`scribe.devices.main` only when the active backend is
    ROCm — the line is meaningless on CUDA / MPS / CPU. Includes the
    distro pretty-name when we have one so users can copy-paste it
    into a support ticket.
    """
    label = distro_pretty or "(distro not detected)"
    return f"Distro support:    {tier} — {label} ({tier_explanation(tier)})"
