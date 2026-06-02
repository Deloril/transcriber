"""Tests for scribe.audio — ffprobe / ffmpeg wrappers and waveform builder."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scribe import audio
from scribe.audio import (
    AudioStream,
    _parse_fraction,
    _safe_float,
    _safe_int,
    compute_waveform,
    extract_track_to_wav,
    probe_audio_streams,
    probe_media_info,
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


class TestSafeInt:
    @pytest.mark.parametrize("v,expected", [
        ("12", 12),
        (12, 12),
        ("0", 0),
        ("12.5", None),  # not an int literal
        (None, None),
        ("", None),
        ("garbage", None),
    ])
    def test_cases(self, v: object, expected: int | None) -> None:
        assert _safe_int(v) == expected


class TestSafeFloat:
    @pytest.mark.parametrize("v,expected", [
        ("1.5", 1.5),
        (1.5, 1.5),
        ("0", 0.0),
        (None, None),
        ("", None),
        ("garbage", None),
    ])
    def test_cases(self, v: object, expected: float | None) -> None:
        assert _safe_float(v) == expected


class TestParseFraction:
    def test_simple(self) -> None:
        assert _parse_fraction("25/1") == 25.0
        assert _parse_fraction("30000/1001") == pytest.approx(29.97002997)

    def test_zero_denominator(self) -> None:
        assert _parse_fraction("30/0") is None
        assert _parse_fraction("0/0") is None

    def test_plain_numeric(self) -> None:
        assert _parse_fraction("25.0") == 25.0
        assert _parse_fraction(24.0) == 24.0

    def test_garbage(self) -> None:
        assert _parse_fraction("not/a/fraction") is None
        assert _parse_fraction(None) is None
        assert _parse_fraction("") is None


# --------------------------------------------------------------------------- #
# probe_audio_streams (mocks ffprobe)
# --------------------------------------------------------------------------- #


class TestProbeAudioStreams:
    def test_parses_streams(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        payload = {
            "streams": [
                {
                    "index": 1,
                    "channels": 2,
                    "codec_name": "aac",
                    "tags": {"title": "Mic", "language": "eng"},
                },
                {
                    "index": 2,
                    "channels": 1,
                    "codec_name": "pcm_s16le",
                },
            ]
        }
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: json.dumps(payload).encode("utf-8"),
        )
        out = probe_audio_streams(f)
        assert len(out) == 2
        assert isinstance(out[0], AudioStream)
        assert out[0].index == 1
        assert out[0].channels == 2
        assert out[0].title == "Mic"
        assert out[0].language == "eng"
        assert out[0].codec == "aac"
        assert out[1].title is None
        assert out[1].codec == "pcm_s16le"

    def test_empty_streams(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: b'{"streams": []}',
        )
        assert probe_audio_streams(f) == []

    def test_handles_missing_tags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: b'{"streams":[{"index":1,"codec_name":"aac"}]}',
        )
        out = probe_audio_streams(f)
        assert len(out) == 1
        assert out[0].channels == 1  # default
        assert out[0].title is None
        assert out[0].language is None


# --------------------------------------------------------------------------- #
# probe_media_info
# --------------------------------------------------------------------------- #


class TestProbeMediaInfo:
    def test_full_payload(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"x" * 1024)
        payload = {
            "format": {
                "duration": "120.5",
                "bit_rate": "1000000",
                "size": "1024",
                "format_long_name": "QuickTime / MOV",
                "format_name": "mov,mp4",
            },
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "pix_fmt": "yuv420p",
                    "bit_rate": "5000000",
                    "duration": "120.5",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "channel_layout": "stereo",
                    "sample_rate": "48000",
                    "bit_rate": "128000",
                    "duration": "120.5",
                    "tags": {"title": "Main", "language": "eng"},
                },
            ],
        }
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: json.dumps(payload).encode("utf-8"),
        )
        info = probe_media_info(f)
        assert info["filename"] == "in.mp4"
        assert info["duration_seconds"] == 120.5
        assert info["bit_rate"] == 1000000
        assert info["size_bytes"] == 1024
        assert info["format_name"] == "QuickTime / MOV"
        assert len(info["video"]) == 1
        v = info["video"][0]
        assert v["codec"] == "h264"
        assert v["width"] == 1920
        assert v["fps"] == pytest.approx(29.97003)
        assert len(info["audio"]) == 1
        a = info["audio"][0]
        assert a["channels"] == 2
        assert a["sample_rate"] == 48000
        assert a["title"] == "Main"
        assert a["language"] == "eng"

    def test_falls_back_to_stat_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # When ffprobe omits size, we use the file's actual size.
        f = tmp_path / "in.mp4"
        f.write_bytes(b"abcdefghij")  # 10 bytes
        payload = {"format": {}, "streams": []}
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: json.dumps(payload).encode("utf-8"),
        )
        info = probe_media_info(f)
        assert info["size_bytes"] == 10

    def test_format_name_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        payload = {"format": {"format_name": "mov,mp4"}, "streams": []}
        monkeypatch.setattr(
            audio.subprocess, "check_output",
            lambda *a, **kw: json.dumps(payload).encode("utf-8"),
        )
        info = probe_media_info(f)
        assert info["format_name"] == "mov,mp4"


# --------------------------------------------------------------------------- #
# extract_track_to_wav
# --------------------------------------------------------------------------- #


class TestExtractTrackToWav:
    def test_default_args(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        out = tmp_path / "out.wav"

        captured: list[list[str]] = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))
            return MagicMock(returncode=0)

        monkeypatch.setattr(audio.subprocess, "run", fake_run)
        result = extract_track_to_wav(f, out)
        assert result == out
        cmd = captured[0]
        # Pulls from default audio stream.
        assert "-map" in cmd
        i = cmd.index("-map")
        assert cmd[i + 1] == "0:a:0"
        # Mono 16 kHz PCM.
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"

    def test_explicit_stream_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        out = tmp_path / "out.wav"

        captured: list[list[str]] = []
        monkeypatch.setattr(
            audio.subprocess, "run",
            lambda cmd, **kw: (captured.append(list(cmd)), MagicMock(returncode=0))[1],
        )
        extract_track_to_wav(f, out, stream_index=2)
        cmd = captured[0]
        i = cmd.index("-map")
        assert cmd[i + 1] == "0:2"

    def test_creates_parent_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        out = tmp_path / "nested" / "out.wav"
        monkeypatch.setattr(audio.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
        extract_track_to_wav(f, out)
        assert out.parent.is_dir()

    def test_uses_copyts_and_start_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """The extraction must preserve the stream's container-level
        start time so multi-track segment timestamps are absolute,
        not relative to each track's first non-silent moment.

        Without ``-copyts -start_at_zero``, ffmpeg rebases the
        stream's internal time so the first audio sample lands at
        0:00 in the output WAV — Whisper then transcribes timestamps
        relative to the stream's first sample, the user sees every
        speaker "start at 00:00", and the multi-track sort produces
        an out-of-order transcript.
        """
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        out = tmp_path / "out.wav"
        captured: list[list[str]] = []
        monkeypatch.setattr(
            audio.subprocess, "run",
            lambda cmd, **kw: (captured.append(list(cmd)), MagicMock(returncode=0))[1],
        )
        extract_track_to_wav(f, out, stream_index=2)
        cmd = captured[0]
        # Both flags must be present, AND they must come before the
        # ``-i`` so they apply to the input demuxer (not the output).
        assert "-copyts" in cmd
        assert "-start_at_zero" in cmd
        i_input = cmd.index("-i")
        assert cmd.index("-copyts") < i_input
        assert cmd.index("-start_at_zero") < i_input


# --------------------------------------------------------------------------- #
# compute_waveform
# --------------------------------------------------------------------------- #


class _FakeProc:
    """Stand-in for subprocess.Popen when we want to feed scripted PCM data."""

    def __init__(self, samples: list[int]) -> None:
        self._buf = struct.pack(f"<{len(samples)}h", *samples)
        self._read_offset = 0
        self.returncode: int | None = None
        # `proc.stdout` is the read end.
        outer = self

        class _Stdout:
            def read(self, n: int) -> bytes:
                start = outer._read_offset
                end = min(start + n, len(outer._buf))
                outer._read_offset = end
                return outer._buf[start:end]

        self.stdout = _Stdout()

    def wait(self) -> int:
        self.returncode = 0
        return 0


class TestComputeWaveform:
    def test_silent_returns_zeros(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.wav"
        f.write_bytes(b"")
        # 1024 samples of silence
        proc = _FakeProc([0] * 1024)
        monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: proc)
        out = compute_waveform(f, bins=10)
        assert len(out) == 10
        assert all(v == 0.0 for v in out)

    def test_full_scale_signal_normalised_to_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.wav"
        f.write_bytes(b"")
        # 1024 samples at the full negative scale (most extreme abs() value).
        # The bucketer divides by 32768 so we'd get exactly 1.0.
        proc = _FakeProc([-32768] * 1024)
        monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: proc)
        out = compute_waveform(f, bins=4)
        assert all(abs(v - 1.0) < 1e-6 for v in out)

    def test_bin_count_exact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.wav"
        f.write_bytes(b"")
        # 65536 samples (one chunk_samples block) at varying levels.
        samples = [(-1) ** i * (i % 32768) for i in range(65536)]
        proc = _FakeProc(samples)
        monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: proc)
        out = compute_waveform(f, bins=200)
        assert len(out) == 200
        # All values in [0, 1].
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_zero_bins_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "in.wav"
        f.write_bytes(b"")
        with pytest.raises(ValueError):
            compute_waveform(f, bins=0)

    def test_no_audio_returns_zeros(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # ffmpeg returns no PCM (silence in the input or empty stream).
        f = tmp_path / "in.wav"
        f.write_bytes(b"")
        proc = _FakeProc([])
        monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: proc)
        out = compute_waveform(f, bins=50)
        assert out == [0.0] * 50

    def test_propagates_ffmpeg_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "in.wav"
        f.write_bytes(b"")

        class _FailProc:
            stdout = MagicMock()
            returncode = 1

            def __init__(self) -> None:
                self.stdout.read = MagicMock(return_value=b"")

            def wait(self) -> int:
                return 1

        monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: _FailProc())
        with pytest.raises(RuntimeError):
            compute_waveform(f, bins=10)


# --------------------------------------------------------------------------- #
# build_playback_mix — merge multi-track audio for browser playback.
# --------------------------------------------------------------------------- #


class TestBuildPlaybackMix:
    """Real ffmpeg integration tests; small synthetic inputs keep them
    fast. Skipped if ffmpeg isn't on PATH."""

    def _have_ffmpeg(self) -> bool:
        import shutil
        return bool(shutil.which("ffmpeg"))

    def _make_dual_audio_mka(self, out: Path, *, seconds: float = 0.5) -> Path:
        """Two-audio-track MKA: 440 Hz + 880 Hz sines."""
        if not self._have_ffmpeg():
            pytest.skip("ffmpeg not on PATH")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}",
            "-map", "0:a", "-map", "1:a",
            "-c:a", "pcm_s16le",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        return out

    def test_mix_collapses_two_tracks_to_one(self, tmp_path: Path) -> None:
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        out = tmp_path / "mix.mp3"
        result = audio.build_playback_mix(src, out)
        assert result == out
        assert out.is_file()
        # The output file has exactly one audio stream (the mix).
        streams = audio.probe_audio_streams(out)
        assert len(streams) == 1

    def test_mix_is_audible_on_both_inputs(self, tmp_path: Path) -> None:
        """Decode the mixed output to PCM and confirm it contains
        non-trivial audio across the duration. Earlier ``-map 0:a:0``
        bugs would silently drop everything past the first track; this
        asserts the mix actually has signal."""
        np = pytest.importorskip("numpy")
        src = self._make_dual_audio_mka(tmp_path / "two.mka", seconds=0.5)
        out = tmp_path / "mix.mp3"
        audio.build_playback_mix(src, out)
        # Decode the mix to f32le PCM and check it's not silent.
        proc = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error",
                "-i", str(out),
                "-f", "f32le", "-ac", "1", "-ar", "16000",
                "-",
            ],
            check=True, capture_output=True,
        )
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        assert samples.size > 0
        assert float(samples.std()) > 0.01, "mix appears silent"

    def test_idempotent_when_cache_is_fresh(self, tmp_path: Path) -> None:
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        out = tmp_path / "mix.mp3"
        audio.build_playback_mix(src, out)
        first_mtime = out.stat().st_mtime
        # Second call returns immediately without re-running ffmpeg.
        # We assert by stamping the source backwards so the cache is
        # newer than it.
        import os as _os, time as _time
        old = first_mtime - 60
        _os.utime(src, (old, old))
        # Ensure mtime granularity ticks.
        _time.sleep(0.01)
        result = audio.build_playback_mix(src, out)
        assert result == out
        assert out.stat().st_mtime == first_mtime

    def test_cache_busts_when_source_is_newer(self, tmp_path: Path) -> None:
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        out = tmp_path / "mix.mp3"
        audio.build_playback_mix(src, out)
        first_mtime = out.stat().st_mtime
        # Fake a fresher source.
        import os as _os, time as _time
        _time.sleep(0.05)
        _os.utime(src, None)  # touch
        audio.build_playback_mix(src, out)
        assert out.stat().st_mtime >= first_mtime

    def test_selected_indices_filter(self, tmp_path: Path) -> None:
        """Selecting only one of two tracks should produce a mix
        that's non-silent (it contains that one track) but has the
        same single-stream output shape."""
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        # Probe to find absolute indices.
        streams = audio.probe_audio_streams(src)
        assert len(streams) == 2
        out = tmp_path / "filtered.mp3"
        audio.build_playback_mix(
            src, out, selected_stream_indices=[streams[1].index],
        )
        assert out.is_file()

    def test_unknown_indices_raise(self, tmp_path: Path) -> None:
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        out = tmp_path / "x.mp3"
        with pytest.raises(ValueError):
            audio.build_playback_mix(
                src, out, selected_stream_indices=[99],
            )

    def test_empty_selection_raises(self, tmp_path: Path) -> None:
        src = self._make_dual_audio_mka(tmp_path / "two.mka")
        out = tmp_path / "x.mp3"
        # An empty selection is rejected — a player with zero audio
        # tracks isn't useful.
        streams = audio.probe_audio_streams(src)
        bogus_filter = [i for i in range(100) if i not in {s.index for s in streams}]
        with pytest.raises(ValueError):
            audio.build_playback_mix(
                src, out, selected_stream_indices=bogus_filter,
            )

    def test_single_track_is_fast_path(self, tmp_path: Path) -> None:
        """One-track input shouldn't go through amix — but the helper
        still works on it (the ``/media`` endpoint won't normally call
        the helper for single-track files, but defensive callers might)."""
        if not self._have_ffmpeg():
            pytest.skip("ffmpeg not on PATH")
        # Mono file — no need for the dual-input fixture.
        src = tmp_path / "mono.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                "-c:a", "pcm_s16le",
                str(src),
            ],
            check=True,
        )
        out = tmp_path / "mix.mp3"
        audio.build_playback_mix(src, out)
        assert out.is_file()
