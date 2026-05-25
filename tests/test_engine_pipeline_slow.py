"""Real-model integration tests for the transcription engine.

Marked @pytest.mark.slow @pytest.mark.gpu — excluded from the default
suite. Run explicitly with:
    pytest -m slow tests/test_engine_pipeline_slow.py
or to require GPU:
    pytest -m "slow and gpu" tests/test_engine_pipeline_slow.py

These exist to catch regressions in the wiring around whisperx,
pyannote, and our progress callbacks. They do *not* try to validate
transcription quality (silence in, no text out is a fine result).
"""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import pytest


def _make_silent_wav(path: Path, seconds: float = 5.0, sr: int = 16000) -> None:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def _make_dual_track_mkv(path: Path, seconds: float = 3.0) -> None:
    """Build a tiny MKV with two silent audio streams via ffmpeg, so the
    multi-track path has something to map against."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # Two `anullsrc` filters → two audio streams, no video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=16000:cl=mono",
        "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=16000:cl=mono",
        "-map", "0:a", "-map", "1:a",
        "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)


@pytest.mark.slow
class TestDiarizePipeline:
    """The diarize path needs an HF token to load the pyannote pipeline.
    If HF_TOKEN isn't set we skip — there's no way to load the diarization
    model otherwise."""

    @pytest.mark.gpu
    def test_diarize_runs_to_completion(self, tmp_path: Path) -> None:
        if not os.environ.get("HF_TOKEN"):
            pytest.skip("HF_TOKEN not set; can't load pyannote pipeline")

        wav = tmp_path / "silent.wav"
        _make_silent_wav(wav, seconds=5.0)

        from scribe.engine import AdvancedOptions, transcribe_diarize
        progress_calls: list[tuple[str, float]] = []

        result = transcribe_diarize(
            wav,
            work_dir=tmp_path / "work",
            hf_token=os.environ["HF_TOKEN"],
            model_name="tiny",
            language="en",
            batch_size=1,
            options=AdvancedOptions(),
            progress=lambda msg, frac: progress_calls.append((msg, frac)),
        )

        # Pipeline reached terminal state.
        assert progress_calls, "no progress callbacks fired"
        assert progress_calls[-1] == ("Done", 1.0)
        # Result has the right shape even on silence.
        assert result.language in ("en", "auto")
        assert result.mode == "diarize"
        assert isinstance(result.segments, list)


@pytest.mark.slow
class TestMultiTrackPipeline:
    @pytest.mark.gpu
    def test_multi_track_runs_to_completion(self, tmp_path: Path) -> None:
        # ffmpeg is required to build the dual-track input.
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

        mkv = tmp_path / "dual.mkv"
        _make_dual_track_mkv(mkv, seconds=3.0)

        from scribe.engine import AdvancedOptions, transcribe_multi_track
        progress_calls: list[tuple[str, float]] = []

        result = transcribe_multi_track(
            mkv,
            work_dir=tmp_path / "work",
            speaker_labels=["A", "B"],
            model_name="tiny",
            language="en",
            batch_size=1,
            options=AdvancedOptions(),
            progress=lambda msg, frac: progress_calls.append((msg, frac)),
        )

        assert progress_calls
        assert progress_calls[-1] == ("Done", 1.0)
        assert result.mode == "multi-track"
        # Both per-track speaker labels survive into the canonical list.
        assert sorted(result.speaker_labels) == ["A", "B"]


@pytest.mark.slow
class TestTranscribeAuto:
    """transcribe(mode='auto') picks multi-track when ≥2 streams, diarize otherwise."""

    @pytest.mark.gpu
    def test_auto_picks_multi_track_on_dual_stream(self, tmp_path: Path) -> None:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

        mkv = tmp_path / "dual.mkv"
        _make_dual_track_mkv(mkv)

        from scribe.engine import AdvancedOptions, transcribe
        result = transcribe(
            mkv,
            work_dir=tmp_path / "work",
            mode="auto",
            speaker_labels=["A", "B"],
            model_name="tiny",
            language="en",
            batch_size=1,
            options=AdvancedOptions(),
        )
        assert result.mode == "multi-track"

    @pytest.mark.gpu
    def test_auto_picks_diarize_on_single_stream(self, tmp_path: Path) -> None:
        if not os.environ.get("HF_TOKEN"):
            pytest.skip("HF_TOKEN not set; auto path falls into diarize")

        wav = tmp_path / "single.wav"
        _make_silent_wav(wav)
        from scribe.engine import AdvancedOptions, transcribe
        result = transcribe(
            wav,
            work_dir=tmp_path / "work",
            mode="auto",
            model_name="tiny",
            language="en",
            batch_size=1,
            hf_token=os.environ["HF_TOKEN"],
            options=AdvancedOptions(),
        )
        assert result.mode == "diarize"
