"""ROCm installer metadata — pinned wheel version + drift detection.

CTranslate2's ROCm wheels aren't published on PyPI; they live as zipped
wheels on each `OpenNMT/CTranslate2` GitHub release. ``setup.sh --rocm``
fetches and pip-installs one of them. This module is the *single source
of truth* for which version that should be, so:

* ``setup.sh --rocm`` can read the pin from here instead of carrying its
  own hard-coded literal that's free to drift,
* ``scribe.devices`` can show users which version is pinned and warn when
  the actually-installed ``ctranslate2`` doesn't match (which is the
  most common failure mode after an unrelated ``pip install`` upgrades
  the package transitively).

Implements the "pin a known-good version; surface it in scribe.devices
so we can spot drift" half of G2.1.

G2.2 — fallback mirrors. The GitHub release URL is the *primary* source
but it's a single point of failure: a GitHub Releases outage, a corporate
firewall, or an air-gapped install all break ``setup.sh --rocm`` today.
The fix is a configurable fallback-URL list (env var
``SCRIBE_CT2_ROCM_FALLBACK_URLS``, comma-separated) that ``setup.sh``
walks in order if the primary download fails. We don't ship a default
mirror in this repo — there's no infrastructure to host one yet — but
the *mechanism* lets ops, lab admins, or a future Scribe-hosted mirror
slot in trivially.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Mapping


# The pinned CTranslate2 ROCm wheel version. Bump this in lockstep with
# setup.sh's expectations and re-test on a representative AMD card before
# shipping. Override at install time with ``SCRIBE_CT2_ROCM_VERSION=...``.
PINNED_CT2_ROCM_VERSION: str = "4.7.2"

# GitHub release artefact path. The release tag is ``v<version>`` and the
# wheel zip is always named ``rocm-python-wheels-Linux.zip``.
_RELEASE_URL_TEMPLATE: str = (
    "https://github.com/OpenNMT/CTranslate2/releases/download/v{version}/"
    "rocm-python-wheels-Linux.zip"
)


def pinned_ct2_rocm_version() -> str:
    """Return the pinned CT2 ROCm wheel version."""
    return PINNED_CT2_ROCM_VERSION


def rocm_wheel_zip_url(version_str: str | None = None) -> str:
    """Construct the CT2 ROCm wheel-zip URL for the given (or pinned) version."""
    v = version_str or PINNED_CT2_ROCM_VERSION
    return _RELEASE_URL_TEMPLATE.format(version=v)


# G2.2 — fallback-URL plumbing.
#
# Users override the primary GitHub-release URL with one or more comma-
# separated mirrors via ``SCRIBE_CT2_ROCM_FALLBACK_URLS``. Empty entries
# are silently dropped (so trailing commas / whitespace don't bite). The
# values are taken as full, ready-to-fetch URLs — we don't try to
# template ``{version}`` for the user; if a mirror is version-specific
# they can rebuild the env var as part of the same shell invocation.
ROCM_FALLBACK_ENV_VAR: str = "SCRIBE_CT2_ROCM_FALLBACK_URLS"


def _split_fallback_csv(raw: str) -> list[str]:
    """Split + strip a comma-separated URL string; drop empty entries."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def rocm_wheel_fallback_urls(env: Mapping[str, str] | None = None) -> list[str]:
    """User-configured fallback URLs for the CT2 ROCm wheel zip.

    Reads :data:`ROCM_FALLBACK_ENV_VAR` (``SCRIBE_CT2_ROCM_FALLBACK_URLS``)
    from the supplied ``env`` mapping (or :data:`os.environ` by default).
    Returns an empty list when the env var is unset or empty.

    Each entry is treated as a full URL; the GitHub release URL is
    *not* included here — this is fallbacks only. See
    :func:`rocm_wheel_zip_urls` for the full ordered list.
    """
    e = os.environ if env is None else env
    return _split_fallback_csv(e.get(ROCM_FALLBACK_ENV_VAR, ""))


def rocm_wheel_zip_urls(
    version_str: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Ordered URL list to try for the CT2 ROCm wheel zip.

    Returns ``[primary_github_url, *user_configured_fallbacks]``. The
    primary URL is always first — fallbacks only fire when the primary
    download fails (the actual fall-through is implemented in
    ``setup.sh --rocm``).

    Tests pass ``env={...}`` to inject a synthetic environment without
    touching the real :data:`os.environ`.
    """
    return [rocm_wheel_zip_url(version_str), *rocm_wheel_fallback_urls(env=env)]


def installed_ct2_version() -> str | None:
    """Installed ``ctranslate2`` package version, or None if not installed.

    Uses :mod:`importlib.metadata` so it doesn't require importing the
    package — safe to call on a CPU-only machine where the ROCm wheel
    isn't installed at all.
    """
    try:
        return version("ctranslate2")
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None


_UNSET = object()


def ct2_drift_message(installed=_UNSET, pinned=_UNSET) -> str | None:
    """Return a one-line drift warning, or None when CT2 matches the pin.

    Only meaningful when the active backend is ROCm — caller decides when
    to call. Three states:

    * matched → ``None``
    * not installed → ``"ctranslate2 not found …"`` (run setup again)
    * mismatch → ``"ctranslate2 v<actual> installed; pinned ROCm wheel is v<pinned> …"``

    ``installed`` and ``pinned`` are injectable for tests. Passing
    ``installed=None`` explicitly means "the package isn't installed";
    omitting the argument means "look it up". A private sentinel is used
    so the two cases are distinguishable.
    """
    p = PINNED_CT2_ROCM_VERSION if pinned is _UNSET else pinned
    i = installed_ct2_version() if installed is _UNSET else installed
    if i is None:
        return (
            f"ctranslate2 not found; pinned ROCm wheel is v{p} "
            "(run ./setup.sh --rocm)"
        )
    if i != p:
        return (
            f"ctranslate2 v{i} installed; pinned ROCm wheel is v{p} "
            "(run ./setup.sh --rocm to realign)"
        )
    return None
