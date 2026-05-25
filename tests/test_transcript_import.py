"""Tests for ``scribe.transcript_import`` — the pure parsers that
turn already-finished transcripts (TXT / SRT / VTT / Scribe JSON)
into the standard ``TranscriptionResult.to_dict()`` shape.

F10.3. Pure functions only — no FastAPI here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scribe import transcript_import as ti


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #


class TestSniffFormat:
    def test_extension_is_authoritative_for_srt(self) -> None:
        assert ti.sniff_format("clip.srt", "1\n00:00:00,000 --> 00:00:01,000\nHi\n") == "srt"

    def test_extension_is_authoritative_for_vtt(self) -> None:
        assert ti.sniff_format("clip.VTT", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi") == "vtt"

    def test_extension_is_authoritative_for_json(self) -> None:
        assert ti.sniff_format("clip.json", '{"segments": []}') == "scribe-json"

    def test_txt_extension_with_json_content_still_picks_json(self) -> None:
        # The user saved Scribe JSON as ``transcript.txt``; we sniff
        # the ``{`` to recover.
        assert ti.sniff_format("foo.txt", '{"segments": []}') == "scribe-json"

    def test_txt_extension_with_webvtt_content_picks_vtt(self) -> None:
        assert ti.sniff_format("foo.txt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi") == "vtt"

    def test_no_filename_sniffs_from_content(self) -> None:
        assert ti.sniff_format(None, "WEBVTT\n\n") == "vtt"
        assert ti.sniff_format(None, '{"segments":[]}') == "scribe-json"
        assert ti.sniff_format(None, "1\n00:00:00,000 --> 00:00:01,000\nHi\n") == "srt"

    def test_falls_back_to_txt(self) -> None:
        assert ti.sniff_format(None, "[00:01] LUKE: hi") == "txt"
        assert ti.sniff_format("foo.bin", "just words") == "txt"

    def test_strips_bom_when_sniffing(self) -> None:
        body = "﻿" + '{"segments": []}'
        assert ti.sniff_format(None, body) == "scribe-json"


# --------------------------------------------------------------------------- #
# Plain-text parser
# --------------------------------------------------------------------------- #


class TestParseTxt:
    def test_round_trips_writer_format(self) -> None:
        # Mirrors what scribe.writers.write_txt produces.
        body = (
            "[00:00] LUKE: Hello there.\n\n"
            "[00:05] GUEST: Hi back.\n"
        )
        out = ti.parse_txt(body)
        assert out["mode"] == "diarize"
        assert out["language"] == "en"
        assert out["speakers"] == ["LUKE", "GUEST"]
        assert len(out["segments"]) == 2
        assert out["segments"][0]["speaker"] == "LUKE"
        assert out["segments"][0]["text"] == "Hello there."
        assert out["segments"][0]["start"] == 0.0
        # Stitched: ends at next start.
        assert out["segments"][0]["end"] == pytest.approx(5.0)
        assert out["segments"][1]["start"] == pytest.approx(5.0)

    def test_synthesises_word_timestamps(self) -> None:
        out = ti.parse_txt("[00:00] LUKE: one two three\n\n[00:03] LUKE: four\n")
        words = out["segments"][0]["words"]
        assert [w["text"] for w in words] == ["one", "two", "three"]
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[-1]["end"] == pytest.approx(3.0)
        # Word starts strictly increase.
        starts = [w["start"] for w in words]
        assert starts == sorted(starts)
        assert all(w["speaker"] == "LUKE" for w in words)

    def test_speaker_label_can_be_omitted(self) -> None:
        out = ti.parse_txt("[00:00] First line.\n\n[00:02] Second line.\n")
        assert all(s["speaker"] == "Speaker 1" for s in out["segments"])

    def test_inherits_speaker_when_label_dropped_mid_transcript(self) -> None:
        body = (
            "[00:00] LUKE: First.\n\n"
            "[00:02] continuing.\n\n"
            "[00:04] GUEST: switching.\n"
        )
        out = ti.parse_txt(body)
        assert out["segments"][0]["speaker"] == "LUKE"
        assert out["segments"][1]["speaker"] == "LUKE"
        assert out["segments"][2]["speaker"] == "GUEST"

    def test_no_timestamps_uses_default_4s_spacing(self) -> None:
        out = ti.parse_txt("LUKE: hello\n\nGUEST: hi\n")
        starts = [s["start"] for s in out["segments"]]
        assert starts == [0.0, 4.0]

    def test_handles_crlf_line_endings(self) -> None:
        out = ti.parse_txt("[00:00] LUKE: hi\r\n\r\n[00:02] LUKE: there\r\n")
        assert len(out["segments"]) == 2

    def test_blank_input_yields_empty_segments(self) -> None:
        # An empty transcript file is technically parseable but
        # we surface it as "nothing to import" via the dispatcher;
        # the underlying parser still returns the empty envelope.
        out = ti.parse_txt("")
        assert out["segments"] == []
        assert out["speakers"] == []

    def test_h_mm_ss_clock_prefix(self) -> None:
        out = ti.parse_txt("[1:02:03] LUKE: long interview.\n")
        assert out["segments"][0]["start"] == pytest.approx(3723.0)

    def test_speaker_with_spaces_and_dash(self) -> None:
        out = ti.parse_txt("[00:00] Speaker A-1: hi\n")
        assert out["segments"][0]["speaker"] == "Speaker A-1"


# --------------------------------------------------------------------------- #
# SRT parser
# --------------------------------------------------------------------------- #


class TestParseSrt:
    def test_basic_two_cue_file(self) -> None:
        body = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "LUKE: Hello there.\n"
            "\n"
            "2\n"
            "00:00:02,500 --> 00:00:04,000\n"
            "GUEST: Hi back.\n"
        )
        out = ti.parse_srt(body)
        assert len(out["segments"]) == 2
        assert out["segments"][0]["speaker"] == "LUKE"
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["end"] == pytest.approx(2.5)
        assert out["segments"][0]["text"] == "Hello there."
        assert out["segments"][1]["speaker"] == "GUEST"

    def test_synthesises_word_timestamps(self) -> None:
        body = (
            "1\n00:00:00,000 --> 00:00:02,000\nThis has four words.\n"
        )
        out = ti.parse_srt(body)
        words = out["segments"][0]["words"]
        assert len(words) == 4
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[-1]["end"] == pytest.approx(2.0)

    def test_no_speaker_prefix_uses_default(self) -> None:
        body = "1\n00:00:00,000 --> 00:00:01,000\nNo speaker label.\n"
        out = ti.parse_srt(body)
        assert out["segments"][0]["speaker"] == "Speaker 1"

    def test_skips_blank_cues(self) -> None:
        body = "1\n00:00:00,000 --> 00:00:01,000\n\n\n2\n00:00:01,000 --> 00:00:02,000\nReal\n"
        out = ti.parse_srt(body)
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "Real"

    def test_multiline_cue_body_is_joined(self) -> None:
        body = "1\n00:00:00,000 --> 00:00:03,000\nLUKE: This\nspans two lines.\n"
        out = ti.parse_srt(body)
        assert out["segments"][0]["text"] == "This spans two lines."

    def test_handles_dot_separator_too(self) -> None:
        # Some "SRT" files in the wild use dots instead of commas.
        body = "1\n00:00:00.000 --> 00:00:01.000\nHi\n"
        out = ti.parse_srt(body)
        assert out["segments"][0]["end"] == pytest.approx(1.0)

    def test_zero_length_cue_is_padded(self) -> None:
        body = "1\n00:00:00,500 --> 00:00:00,500\nQuick\n"
        out = ti.parse_srt(body)
        # _segment bumps end to start+0.05 so word-spread doesn't
        # divide by zero.
        assert out["segments"][0]["end"] > out["segments"][0]["start"]


# --------------------------------------------------------------------------- #
# VTT parser
# --------------------------------------------------------------------------- #


class TestParseVtt:
    def test_basic_file(self) -> None:
        body = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "<v LUKE>Hello there.\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "<v GUEST>Hi back.\n"
        )
        out = ti.parse_vtt(body)
        assert len(out["segments"]) == 2
        assert out["segments"][0]["speaker"] == "LUKE"
        assert out["segments"][1]["speaker"] == "GUEST"
        assert out["segments"][0]["text"] == "Hello there."

    def test_voice_tag_with_close_tag(self) -> None:
        body = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "<v LUKE>Hello.</v>\n"
        )
        out = ti.parse_vtt(body)
        assert out["segments"][0]["speaker"] == "LUKE"
        assert out["segments"][0]["text"] == "Hello."

    def test_inline_speaker_prefix_when_no_voice_tag(self) -> None:
        body = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "GUEST: hi\n"
        )
        out = ti.parse_vtt(body)
        assert out["segments"][0]["speaker"] == "GUEST"
        assert out["segments"][0]["text"] == "hi"

    def test_skips_note_and_style_blocks(self) -> None:
        body = (
            "WEBVTT\n\n"
            "NOTE this is a comment\n\n"
            "STYLE\n::cue { color: red }\n\n"
            "00:00:00.000 --> 00:00:01.000\nHi\n"
        )
        out = ti.parse_vtt(body)
        assert len(out["segments"]) == 1

    def test_cue_identifier_is_skipped(self) -> None:
        body = (
            "WEBVTT\n\n"
            "intro-cue\n"
            "00:00:00.000 --> 00:00:01.000\nHi\n"
        )
        out = ti.parse_vtt(body)
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "Hi"

    def test_missing_signature_raises(self) -> None:
        with pytest.raises(ValueError, match="WEBVTT"):
            ti.parse_vtt("00:00:00.000 --> 00:00:01.000\nHi\n")


# --------------------------------------------------------------------------- #
# Scribe JSON parser
# --------------------------------------------------------------------------- #


def _scribe_json_payload(**override: Any) -> dict[str, Any]:
    base = {
        "language": "en",
        "mode": "diarize",
        "speakers": ["LUKE", "GUEST"],
        "segments": [
            {
                "text": "Hello there.",
                "start": 0.0,
                "end": 2.0,
                "speaker": "LUKE",
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 1.0,
                     "speaker": "LUKE", "score": 0.9},
                    {"text": "there.", "start": 1.0, "end": 2.0,
                     "speaker": "LUKE", "score": 0.95},
                ],
            },
            {
                "text": "Hi back.",
                "start": 2.0,
                "end": 4.0,
                "speaker": "GUEST",
                "words": [],
            },
        ],
    }
    base.update(override)
    return base


class TestParseScribeJson:
    def test_round_trips_full_payload(self) -> None:
        payload = _scribe_json_payload()
        out = ti.parse_scribe_json(json.dumps(payload))
        assert out["language"] == "en"
        assert out["mode"] == "diarize"
        assert out["speakers"] == ["LUKE", "GUEST"]
        assert len(out["segments"]) == 2
        # Word timings preserved when present.
        assert out["segments"][0]["words"][0]["score"] == 0.9
        # Synthesised when missing.
        assert len(out["segments"][1]["words"]) == 2

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            ti.parse_scribe_json("{not valid")

    def test_missing_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="segments"):
            ti.parse_scribe_json("{}")

    def test_top_level_must_be_object(self) -> None:
        with pytest.raises(ValueError, match="object"):
            ti.parse_scribe_json("[]")

    def test_unknown_mode_falls_back_to_diarize(self) -> None:
        out = ti.parse_scribe_json(json.dumps(_scribe_json_payload(mode="weird")))
        assert out["mode"] == "diarize"

    def test_skips_segments_with_blank_text(self) -> None:
        payload = _scribe_json_payload(segments=[
            {"text": "real", "start": 0.0, "end": 1.0, "speaker": "X", "words": []},
            {"text": "", "start": 1.0, "end": 2.0, "speaker": "Y", "words": []},
        ])
        out = ti.parse_scribe_json(json.dumps(payload))
        assert len(out["segments"]) == 1


# --------------------------------------------------------------------------- #
# Top-level dispatcher
# --------------------------------------------------------------------------- #


class TestParseTranscript:
    def test_picks_srt_by_extension(self) -> None:
        body = "1\n00:00:00,000 --> 00:00:01,000\nHi\n"
        out = ti.parse_transcript("foo.srt", body)
        assert len(out["segments"]) == 1

    def test_picks_vtt_by_extension(self) -> None:
        body = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n"
        out = ti.parse_transcript("foo.vtt", body)
        assert len(out["segments"]) == 1

    def test_picks_scribe_json(self) -> None:
        out = ti.parse_transcript("foo.json", json.dumps(_scribe_json_payload()))
        assert out["speakers"] == ["LUKE", "GUEST"]

    def test_explicit_format_overrides_sniff(self) -> None:
        # File looks like JSON, but caller says it's TXT.  We honour
        # the override even though it'll produce one weird segment.
        out = ti.parse_transcript("foo.json", "[00:00] LUKE: hi\n", fmt="txt")
        assert out["segments"][0]["speaker"] == "LUKE"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            ti.parse_transcript("foo.txt", "x", fmt="docx")

    def test_empty_parse_raises_no_segments(self) -> None:
        with pytest.raises(ValueError, match="No transcript segments"):
            ti.parse_transcript("foo.txt", "\n\n\n")


# --------------------------------------------------------------------------- #
# Internal helpers — the small but important stuff
# --------------------------------------------------------------------------- #


class TestSpreadWords:
    def test_empty_text_returns_empty(self) -> None:
        assert ti._spread_words("", 0.0, 1.0, "LUKE") == []

    def test_evenly_distributes(self) -> None:
        out = ti._spread_words("a b c d", 0.0, 4.0, "LUKE")
        assert [w["text"] for w in out] == ["a", "b", "c", "d"]
        assert all(w["start"] >= 0 and w["end"] > w["start"] for w in out)
        assert out[-1]["end"] == pytest.approx(4.0)

    def test_clamps_minimum_span(self) -> None:
        out = ti._spread_words("a b", 5.0, 5.0, "X")
        # _spread_words enforces ≥0.05 span so each word is non-zero
        # length even on a degenerate cue.
        assert out[0]["end"] > out[0]["start"]


class TestParseClockStr:
    def test_hms_with_comma(self) -> None:
        assert ti._parse_clock_str("00:01:02,500") == pytest.approx(62.5)

    def test_hms_with_dot(self) -> None:
        assert ti._parse_clock_str("00:01:02.250") == pytest.approx(62.25)

    def test_ms_only(self) -> None:
        assert ti._parse_clock_str("01:02,000") == pytest.approx(62.0)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            ti._parse_clock_str("not a time")


class TestParseSpeakerPrefix:
    def test_extracts_speaker(self) -> None:
        assert ti._parse_speaker_prefix("LUKE: hello") == ("LUKE", "hello")

    def test_speaker_with_dash_and_space(self) -> None:
        assert ti._parse_speaker_prefix("Speaker A-1: x") == ("Speaker A-1", "x")

    def test_no_prefix(self) -> None:
        assert ti._parse_speaker_prefix("just text") == (None, "just text")

    def test_does_not_match_normal_sentence(self) -> None:
        # Capitalised first word followed by a colon is rare in
        # English but if you have one ("Wait: I want to clarify")
        # we'd pull "Wait" out.  Document that expectation rather
        # than fight it.
        assert ti._parse_speaker_prefix("Wait: hi") == ("Wait", "hi")


class TestParseClockPrefix:
    def test_square_brackets_mm_ss(self) -> None:
        secs, rest = ti._parse_clock_prefix("[01:30] hi")
        assert secs == pytest.approx(90.0)
        assert rest.strip() == "hi"

    def test_paren_h_mm_ss(self) -> None:
        secs, rest = ti._parse_clock_prefix("(1:00:30) hi")
        assert secs == pytest.approx(3630.0)
        assert rest.strip() == "hi"

    def test_no_prefix(self) -> None:
        secs, rest = ti._parse_clock_prefix("hi there")
        assert secs is None
        assert rest == "hi there"


class TestFinalise:
    def test_collects_speakers_in_first_use_order(self) -> None:
        segs = [
            ti._segment("hi", 0.0, 1.0, "B"),
            ti._segment("there", 1.0, 2.0, "A"),
            ti._segment("more", 2.0, 3.0, "B"),
        ]
        out = ti._finalise(segs)
        assert out["speakers"] == ["B", "A"]

    def test_caps_max_segments(self) -> None:
        too_many = [{"text": "x", "start": 0, "end": 1, "speaker": "S", "words": []}] * (
            ti.MAX_SEGMENTS + 1
        )
        with pytest.raises(ValueError, match="max is"):
            ti._finalise(too_many)


class TestStripVoiceTag:
    def test_passthrough_when_absent(self) -> None:
        assert ti._strip_voice_tag_speaker("plain") == "plain"

    def test_concatenates_multiple_tags(self) -> None:
        body = "<v A>hello</v><v B>world</v>"
        out = ti._strip_voice_tag_speaker(body)
        assert "A: hello" in out
        assert "B: world" in out
