"""Tests for scribe.writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.engine import Segment, TranscriptionResult, Word
from scribe.writers import (
    _fmt_clock,
    _fmt_time,
    write_all,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)


# --------------------------------------------------------------------------- #
# Time formatters
# --------------------------------------------------------------------------- #


class TestFmtTime:
    @pytest.mark.parametrize("secs,expected", [
        (0.0, "00:00:00,000"),
        (1.234, "00:00:01,234"),
        (61.5, "00:01:01,500"),
        (3661.0, "01:01:01,000"),
        (3600 * 12 + 34 * 60 + 56.789, "12:34:56,789"),
    ])
    def test_default_separator(self, secs: float, expected: str) -> None:
        assert _fmt_time(secs) == expected

    def test_vtt_separator(self) -> None:
        assert _fmt_time(1.5, sep=".") == "00:00:01.500"

    def test_negative_clamped_to_zero(self) -> None:
        assert _fmt_time(-3.0) == "00:00:00,000"

    def test_999_5_ms_rounds_into_next_second(self) -> None:
        # 0.9995s rounds to 1000 ms — the formatter must roll the second.
        assert _fmt_time(0.9995) == "00:00:01,000"


class TestFmtClock:
    @pytest.mark.parametrize("secs,expected", [
        (0, "00:00"),
        (5, "00:05"),
        (65, "01:05"),
        (3661, "1:01:01"),
        (7325, "2:02:05"),
    ])
    def test_durations(self, secs: int, expected: str) -> None:
        assert _fmt_clock(secs) == expected


# --------------------------------------------------------------------------- #
# Result fixtures
# --------------------------------------------------------------------------- #


def _build_result(segments: list[Segment]) -> TranscriptionResult:
    speakers = []
    for s in segments:
        if s.speaker not in speakers:
            speakers.append(s.speaker)
    return TranscriptionResult(
        segments=segments,
        language="en",
        mode="diarize",
        speaker_labels=speakers,
        audio_path=Path("/tmp/dummy.wav"),
    )


@pytest.fixture
def two_speaker_result() -> TranscriptionResult:
    return _build_result([
        Segment(text="Hello there.", start=0.0, end=2.0, speaker="LUKE", words=[
            Word("Hello", 0.0, 0.5, "LUKE", 0.99),
            Word("there.", 0.6, 2.0, "LUKE", 0.97),
        ]),
        Segment(text="General Kenobi.", start=2.5, end=4.0, speaker="GUEST", words=[
            Word("General", 2.5, 3.0, "GUEST", 0.99),
            Word("Kenobi.", 3.1, 4.0, "GUEST", 0.99),
        ]),
        Segment(text="Nice meme.", start=4.5, end=5.5, speaker="LUKE", words=[]),
    ])


@pytest.fixture
def single_speaker_consecutive() -> TranscriptionResult:
    return _build_result([
        Segment(text="One two", start=0.0, end=1.0, speaker="A", words=[]),
        Segment(text="three four", start=1.0, end=2.0, speaker="A", words=[]),
        Segment(text="", start=2.0, end=2.5, speaker="A", words=[]),  # empty segment
    ])


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


class TestWriteJson:
    def test_round_trip(self, tmp_path: Path, two_speaker_result: TranscriptionResult) -> None:
        out = tmp_path / "x.json"
        write_json(two_speaker_result, out)
        loaded = json.loads(out.read_text())
        assert loaded["language"] == "en"
        assert loaded["mode"] == "diarize"
        assert loaded["speakers"] == ["LUKE", "GUEST"]
        assert len(loaded["segments"]) == 3
        assert loaded["segments"][0]["text"] == "Hello there."
        assert loaded["segments"][0]["words"][0]["text"] == "Hello"

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        result = _build_result([
            Segment(text="Café — résumé naïve", start=0.0, end=1.0, speaker="A", words=[]),
        ])
        out = tmp_path / "x.json"
        write_json(result, out)
        text = out.read_text(encoding="utf-8")
        # ensure_ascii=False so the bytes contain the literal characters.
        assert "Café" in text
        assert "résumé" in text


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #


class TestWriteTxt:
    def test_groups_consecutive_speaker(
        self, tmp_path: Path, single_speaker_consecutive: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.txt"
        write_txt(single_speaker_consecutive, out)
        text = out.read_text()
        # All non-empty content collapses into one paragraph.
        assert text == "[00:00] A: One two three four\n"

    def test_separates_speaker_changes(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.txt"
        write_txt(two_speaker_result, out)
        text = out.read_text()
        # Each speaker turn is its own block separated by a blank line.
        blocks = [b for b in text.split("\n\n") if b.strip()]
        assert len(blocks) == 3
        assert blocks[0].startswith("[00:00] LUKE:")
        assert "Hello there." in blocks[0]
        assert blocks[1].startswith("[00:02] GUEST:")
        assert blocks[2].startswith("[00:04] LUKE:")

    def test_empty_segments_skipped(self, tmp_path: Path) -> None:
        result = _build_result([
            Segment(text="", start=0.0, end=1.0, speaker="A", words=[]),
            Segment(text="real", start=1.0, end=2.0, speaker="A", words=[]),
        ])
        out = tmp_path / "x.txt"
        write_txt(result, out)
        # The block-start for the first non-empty content is at 0.0 because
        # the speaker hadn't changed between the two segments.
        assert out.read_text() == "[00:00] A: real\n"

    def test_empty_result(self, tmp_path: Path) -> None:
        out = tmp_path / "x.txt"
        write_txt(_build_result([]), out)
        assert out.read_text() == ""


# --------------------------------------------------------------------------- #
# SRT
# --------------------------------------------------------------------------- #


class TestWriteSrt:
    def test_basic_shape(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.srt"
        write_srt(two_speaker_result, out)
        text = out.read_text()
        assert text.startswith("1\n")
        assert "00:00:00,000 --> 00:00:02,000" in text
        assert "LUKE: Hello there." in text
        assert "00:00:02,500 --> 00:00:04,000" in text
        assert "GUEST: General Kenobi." in text

    def test_skips_empty_segments(self, tmp_path: Path) -> None:
        result = _build_result([
            Segment(text="", start=0.0, end=1.0, speaker="A", words=[]),
            Segment(text="real", start=1.0, end=2.0, speaker="A", words=[]),
        ])
        out = tmp_path / "x.srt"
        write_srt(result, out)
        text = out.read_text()
        # Only one cue.
        assert text.count(" --> ") == 1
        # Numbering starts at 2 because we still increment the counter
        # before checking for empty text — that's the current behaviour.
        # If this changes, this test will catch it.
        assert text.startswith("2\n")


# --------------------------------------------------------------------------- #
# VTT
# --------------------------------------------------------------------------- #


class TestWriteVtt:
    def test_starts_with_webvtt_header(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.vtt"
        write_vtt(two_speaker_result, out)
        text = out.read_text()
        assert text.startswith("WEBVTT\n")

    def test_uses_voice_tag(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.vtt"
        write_vtt(two_speaker_result, out)
        text = out.read_text()
        assert "<v LUKE>Hello there." in text
        assert "<v GUEST>General Kenobi." in text

    def test_uses_dot_separator(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        out = tmp_path / "x.vtt"
        write_vtt(two_speaker_result, out)
        text = out.read_text()
        # WebVTT uses a period before milliseconds, not a comma.
        assert "00:00:00.000 --> 00:00:02.000" in text
        assert "00:00:00,000" not in text


# --------------------------------------------------------------------------- #
# write_all
# --------------------------------------------------------------------------- #


class TestWriteAll:
    def test_writes_all_four_formats(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        base = tmp_path / "out"
        paths = write_all(two_speaker_result, base)
        assert set(paths.keys()) == {"json", "txt", "srt", "vtt"}
        for kind, path in paths.items():
            assert path.exists()
            assert path.suffix == f".{kind}"
            assert path.stat().st_size > 0

    def test_creates_parent_dirs(
        self, tmp_path: Path, two_speaker_result: TranscriptionResult
    ) -> None:
        base = tmp_path / "nested" / "deeply" / "out"
        paths = write_all(two_speaker_result, base)
        assert paths["json"].parent.is_dir()
