"""Tiny smoke tests so the runner has something to run before the real
test files land. These also serve as canaries: if they ever fail, the
test infrastructure itself is broken."""

from __future__ import annotations

import scribe


def test_package_importable() -> None:
    assert hasattr(scribe, "__version__")


def test_silent_wav_fixture_works(silent_wav) -> None:
    assert silent_wav.exists()
    assert silent_wav.stat().st_size > 0
