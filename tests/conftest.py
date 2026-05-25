"""
Shared pytest fixtures for the Scribe test suite.

Anything ML-heavy is skipped from the default invocation via the
`slow` / `gpu` markers (see pytest.ini). Fixtures here build the
plumbing the fast unit tests need without touching real models.
"""

from __future__ import annotations

import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Ensure the project package is importable when pytest is run from any cwd.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Generic fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_outputs(tmp_path: Path) -> Path:
    """Empty `outputs/<id>/` style directory."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def tmp_uploads(tmp_path: Path) -> Path:
    """Empty `uploads/<id>/` style directory."""
    up = tmp_path / "uploads"
    up.mkdir()
    return up


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 1-second mono 16-bit 16 kHz WAV of silence — cheap fake input."""
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return p


@pytest.fixture
def fake_segment_dict() -> dict[str, Any]:
    """Whisperx-shaped segment dict for writer tests."""
    return {
        "start": 1.0,
        "end": 3.0,
        "text": "Hello world.",
        "speaker": "SPEAKER_00",
        "words": [
            {"text": "Hello", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_00", "score": 0.99},
            {"text": "world.", "start": 1.6, "end": 3.0, "speaker": "SPEAKER_00", "score": 0.95},
        ],
    }


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Default test environment: no HF token, no SCRIBE_* device overrides leaking
    in from the developer's shell. Tests can re-set these explicitly.
    """
    for var in (
        "HF_TOKEN",
        "SCRIBE_DEVICE",
        "SCRIBE_WHISPER_DEVICE",
        "SCRIBE_DIARIZE_DEVICE",
        "SCRIBE_COMPUTE_TYPE",
        "SCRIBE_STRICT_TORCH_LOAD",
    ):
        monkeypatch.delenv(var, raising=False)
