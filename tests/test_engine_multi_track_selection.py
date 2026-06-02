"""Tests for ``transcribe_multi_track``'s ``selected_stream_indices``.

The full multi-track pipeline is gated behind real model loads; this
test stays in pure-Python land by stubbing ``probe_audio_streams`` and
the inner ``_transcribe_with_alignment`` so we can exercise the
selection-and-validation logic without spinning Whisper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe import engine
from scribe.audio import AudioStream


def _stub_probe(streams: list[AudioStream]):
    return lambda _path: list(streams)


def _stub_align(transcribed: list[Path]):
    """Replacement for ``_transcribe_with_alignment`` that records each
    track WAV path it was asked to process and returns a tiny fake
    segment list."""
    def _inner(audio_path: Path, **kwargs):
        transcribed.append(audio_path)
        seg = {
            "start": 0.0, "end": 1.0, "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}],
        }
        return [seg], "en"
    return _inner


def _stub_extract(monkeypatch: pytest.MonkeyPatch):
    """Replace extract_track_to_wav with a no-op that creates the
    target file so downstream open()s don't blow up."""
    def _inner(input_path: Path, out_path: Path, stream_index: int | None = None,
               sample_rate: int = 16000) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF" + b"\x00" * 60)
        return out_path
    monkeypatch.setattr(engine, "extract_track_to_wav", _inner)


@pytest.fixture
def streams() -> list[AudioStream]:
    return [
        AudioStream(index=0, channels=1, title="Luke", language="eng", codec="pcm_s16le"),
        AudioStream(index=1, channels=1, title="Maria", language="eng", codec="pcm_s16le"),
        AudioStream(index=2, channels=1, title=None, language=None, codec="pcm_s16le"),
    ]


class TestSelectedStreamIndices:
    def test_none_transcribes_every_track(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        _stub_extract(monkeypatch)
        seen: list[Path] = []
        monkeypatch.setattr(engine, "_transcribe_with_alignment", _stub_align(seen))
        result = engine.transcribe_multi_track(
            tmp_path / "in.wav",
            work_dir=tmp_path / "work",
            selected_stream_indices=None,
        )
        assert len(seen) == 3
        # Every track produced segments; speaker labels appear in roster.
        # ``_label_for_track`` uppercases the title-derived labels.
        assert {seg.speaker for seg in result.segments} == {
            "LUKE", "MARIA", "SPEAKER_03",
        }

    def test_selection_filters_streams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        _stub_extract(monkeypatch)
        seen: list[Path] = []
        monkeypatch.setattr(engine, "_transcribe_with_alignment", _stub_align(seen))
        result = engine.transcribe_multi_track(
            tmp_path / "in.wav",
            work_dir=tmp_path / "work",
            selected_stream_indices=[0, 2],
        )
        # Two tracks transcribed; the middle one was skipped entirely.
        assert len(seen) == 2
        speakers = {seg.speaker for seg in result.segments}
        assert "MARIA" not in speakers
        assert "LUKE" in speakers

    def test_empty_list_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        with pytest.raises(ValueError, match="empty"):
            engine.transcribe_multi_track(
                tmp_path / "in.wav",
                work_dir=tmp_path / "work",
                selected_stream_indices=[],
            )

    def test_unknown_index_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        with pytest.raises(ValueError, match="aren't in the file"):
            engine.transcribe_multi_track(
                tmp_path / "in.wav",
                work_dir=tmp_path / "work",
                selected_stream_indices=[99],
            )

    def test_speaker_labels_indexed_by_original_position(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        """speaker_labels is positional over the *full* stream list, so
        when we filter to a subset the picker's labels for the kept
        tracks still land in the right slot."""
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        _stub_extract(monkeypatch)
        seen: list[Path] = []
        monkeypatch.setattr(engine, "_transcribe_with_alignment", _stub_align(seen))
        # Three positional labels for three streams; we only transcribe
        # streams 0 and 2 — the engine should pick "Renamed-Luke" for
        # stream 0 and "Renamed-Anon" for stream 2 (the third position).
        result = engine.transcribe_multi_track(
            tmp_path / "in.wav",
            work_dir=tmp_path / "work",
            selected_stream_indices=[0, 2],
            speaker_labels=["Renamed-Luke", "Renamed-Maria", "Renamed-Anon"],
        )
        speakers = {seg.speaker for seg in result.segments}
        assert "RENAMED-LUKE" in speakers
        assert "RENAMED-ANON" in speakers
        # Stream 1's label doesn't show up — that track wasn't transcribed.
        assert "RENAMED-MARIA" not in speakers


# --------------------------------------------------------------------------- #
# transcribe()'s mode auto-detection — fixes the "I picked one channel and
# it transcribed all six" report. A single-track selection on an N-track
# file must route through diarize on that one stream, not multi-track on
# every stream.
# --------------------------------------------------------------------------- #


class TestSingleSelectionRoutesToDiarize:
    def test_one_selected_stream_picks_diarize_under_auto(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        # ``transcribe()``'s auto-detection used to call
        # ``probe_audio_streams`` and pick multi-track whenever
        # the file had ≥2 streams — even if the user only selected
        # one. That was the bug: 6-track file + selection=[3] still
        # spawned 6 transcription passes. Now a 1-element selection
        # forces diarize.
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        called: dict[str, bool] = {"diarize": False, "multi_track": False}

        def fake_diarize(input_path, *, work_dir, hf_token, **kwargs):
            called["diarize"] = True
            assert kwargs.get("selected_stream_indices") == [1]
            return engine.TranscriptionResult(
                segments=[],
                language="en",
                mode="diarize",
                speaker_labels=[],
                audio_path=input_path,
            )

        def fake_multi_track(*a, **kw):
            called["multi_track"] = True
            raise AssertionError(
                "transcribe() must not route a single-stream selection "
                "through transcribe_multi_track"
            )

        monkeypatch.setattr(engine, "transcribe_diarize", fake_diarize)
        monkeypatch.setattr(engine, "transcribe_multi_track", fake_multi_track)

        engine.transcribe(
            tmp_path / "in.wav",
            work_dir=tmp_path / "work",
            mode="auto",
            selected_stream_indices=[1],
            hf_token="fake-token-for-test",
        )
        assert called["diarize"] is True
        assert called["multi_track"] is False

    def test_multiple_selected_streams_routes_to_multi_track(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        streams: list[AudioStream],
    ) -> None:
        monkeypatch.setattr(engine, "probe_audio_streams", _stub_probe(streams))
        called: dict[str, bool] = {"diarize": False, "multi_track": False}

        def fake_multi_track(input_path, *, work_dir, **kwargs):
            called["multi_track"] = True
            assert kwargs.get("selected_stream_indices") == [0, 2]
            return engine.TranscriptionResult(
                segments=[],
                language="en",
                mode="multi-track",
                speaker_labels=[],
                audio_path=input_path,
            )

        def fake_diarize(*a, **kw):
            called["diarize"] = True
            raise AssertionError(
                "transcribe() must route ≥2-stream selections through "
                "transcribe_multi_track, not transcribe_diarize"
            )

        monkeypatch.setattr(engine, "transcribe_multi_track", fake_multi_track)
        monkeypatch.setattr(engine, "transcribe_diarize", fake_diarize)

        engine.transcribe(
            tmp_path / "in.wav",
            work_dir=tmp_path / "work",
            mode="auto",
            selected_stream_indices=[0, 2],
        )
        assert called["multi_track"] is True
        assert called["diarize"] is False


class TestDiarizeHonoursSelection:
    def test_diarize_rejects_multi_stream_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A multi-stream selection should land in
        ``transcribe_multi_track``; ``transcribe_diarize`` rejects
        ≥2 entries so a routing bug doesn't silently degrade."""
        with pytest.raises(ValueError, match="at most one stream"):
            engine.transcribe_diarize(
                tmp_path / "in.wav",
                work_dir=tmp_path / "work",
                hf_token="t",
                selected_stream_indices=[0, 1],
            )

    def test_diarize_uses_selected_index_for_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The selected index must reach ``extract_track_to_wav`` so the
        WAV is the user's chosen stream, not ffmpeg's default. We
        intercept the function early — extract_track_to_wav raising
        StopIteration short-circuits the rest of the diarize body."""
        seen: list = []

        class _Stop(Exception):
            pass

        def fake_extract(input_path, out_path, stream_index=None, sample_rate=16000):
            seen.append(stream_index)
            raise _Stop("expected — short-circuits the rest of diarize")

        monkeypatch.setattr(engine, "extract_track_to_wav", fake_extract)

        with pytest.raises(_Stop):
            engine.transcribe_diarize(
                tmp_path / "in.wav",
                work_dir=tmp_path / "work",
                hf_token="t",
                selected_stream_indices=[3],
            )
        assert seen == [3]

    def test_diarize_no_selection_falls_through_to_default_stream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No selection → extract uses ``stream_index=None`` so ffmpeg
        picks the default audio stream — backwards-compatible with
        every job that pre-dates the picker."""
        seen: list = []

        class _Stop(Exception):
            pass

        def fake_extract(input_path, out_path, stream_index=None, sample_rate=16000):
            seen.append(stream_index)
            raise _Stop()

        monkeypatch.setattr(engine, "extract_track_to_wav", fake_extract)
        with pytest.raises(_Stop):
            engine.transcribe_diarize(
                tmp_path / "in.wav",
                work_dir=tmp_path / "work",
                hf_token="t",
            )
        assert seen == [None]
