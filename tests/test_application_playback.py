"""Tests for scribe.application_playback (F4.6).

Per PLANNING.md F4.6:

  > One-click playback from any coded segment (reuse the editor's
  > word→time map).

This suite covers:

* :func:`build_word_time_map` — flatten segments to a word-id map,
  skip untimed words, reject malformed segments.
* :func:`playback_range_for_application` — whole-word and sub-word
  anchors, segment-fallback when timing is missing, validation
  paths.
* :func:`playback_ranges_for_applications` — bulk lookup, multi-
  source bucketing, missing-source skip.

All tests are pure Python — no FastAPI, no engine.
"""

from __future__ import annotations

import math

import pytest

from scribe.applications import Application, ProjectValidationError
from scribe.application_playback import (
    PlaybackRange,
    WordTime,
    build_word_time_map,
    playback_range_for_application,
    playback_ranges_for_applications,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE_1 = "1" * 12
_HEX_SOURCE_2 = "2" * 12
_HEX_CODER = "d" * 12
_HEX_VERSION = "e" * 12


def _hex_id(seed: int) -> str:
    return f"{seed:012x}"


def _word(text: str, start: float | None, end: float | None, speaker: str = "S0") -> dict:
    """Build a word dict with optional timing."""
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def _seg(words, start: float | None = None, end: float | None = None, speaker: str = "S0") -> dict:
    """Build a segment dict; ``start`` / ``end`` default to the first/last word's timing."""
    if start is None and words and words[0].get("start") is not None:
        start = words[0]["start"]
    if end is None and words and words[-1].get("end") is not None:
        end = words[-1]["end"]
    return {"speaker": speaker, "start": start, "end": end, "words": words}


def _app(
    *,
    source_id: str = _HEX_SOURCE_1,
    start_id: str = "s0w0",
    end_id: str = "s0w2",
    start_offset: int | None = None,
    end_offset: int | None = None,
    application_id: str | None = None,
) -> Application:
    return Application.new(
        project_id=_HEX_PROJECT,
        code_id=_HEX_CODE,
        source_id=source_id,
        coder_id=_HEX_CODER,
        anchor_start_word_id=start_id,
        anchor_end_word_id=end_id,
        definition_version_id_at_apply=_HEX_VERSION,
        start_char_offset=start_offset,
        end_char_offset=end_offset,
        application_id=application_id,
    )


# A simple two-segment transcript reused across tests.
def _basic_transcript() -> list[dict]:
    return [
        _seg(
            [
                _word("Hello,", 0.0, 0.5),
                _word("world.", 0.5, 1.0),
                _word("How", 1.2, 1.4),
                _word("are", 1.4, 1.6),
                _word("you?", 1.6, 2.0),
            ],
            speaker="A",
        ),
        _seg(
            [
                _word("I'm", 3.0, 3.2),
                _word("fine,", 3.2, 3.5),
                _word("thanks.", 3.5, 4.0),
            ],
            speaker="B",
        ),
    ]


# --------------------------------------------------------------------------- #
# build_word_time_map
# --------------------------------------------------------------------------- #


class TestBuildWordTimeMap:
    def test_builds_map_for_simple_transcript(self):
        segs = _basic_transcript()
        m = build_word_time_map(segs)
        assert "s0w0" in m
        assert m["s0w0"] == WordTime(0.0, 0.5, "Hello,")
        assert m["s0w4"] == WordTime(1.6, 2.0, "you?")
        assert m["s1w0"] == WordTime(3.0, 3.2, "I'm")
        assert m["s1w2"] == WordTime(3.5, 4.0, "thanks.")

    def test_preserves_document_order(self):
        segs = _basic_transcript()
        keys = list(build_word_time_map(segs).keys())
        assert keys == [
            "s0w0", "s0w1", "s0w2", "s0w3", "s0w4",
            "s1w0", "s1w1", "s1w2",
        ]

    def test_empty_segments_yields_empty_map(self):
        assert build_word_time_map([]) == {}

    def test_skips_words_with_missing_timing(self):
        segs = [
            _seg(
                [
                    _word("first", 0.0, 0.5),
                    _word("untimed", None, None),
                    _word("third", 1.0, 1.2),
                ],
            )
        ]
        m = build_word_time_map(segs)
        assert "s0w0" in m
        assert "s0w1" not in m
        assert "s0w2" in m

    def test_skips_words_with_partial_timing(self):
        # Either side missing → skip.
        segs = [
            _seg(
                [
                    _word("a", 0.0, None),
                    _word("b", None, 1.0),
                    _word("c", 1.0, 2.0),
                ]
            )
        ]
        m = build_word_time_map(segs)
        assert list(m.keys()) == ["s0w2"]

    def test_skips_words_with_reversed_timing(self):
        segs = [_seg([_word("backwards", 1.0, 0.5)])]
        assert build_word_time_map(segs) == {}

    def test_skips_words_with_nan_or_inf(self):
        segs = [
            _seg(
                [
                    _word("a", float("nan"), 0.5),
                    _word("b", 0.0, float("inf")),
                    _word("c", 0.0, 0.5),
                ],
                start=0.0,
                end=0.5,
            )
        ]
        m = build_word_time_map(segs)
        assert list(m.keys()) == ["s0w2"]

    def test_skips_words_with_bool_timing(self):
        # bool is an int subclass; we explicitly reject it so a
        # stray ``True`` doesn't seek to 1.0s.
        segs = [_seg([{"text": "x", "start": True, "end": 0.5}])]
        assert build_word_time_map(segs) == {}

    def test_non_dict_word_skipped(self):
        segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "words": ["raw-string", _word("ok", 0.0, 0.5)]}]
        m = build_word_time_map(segs)
        # The string is at index 0; the dict is at index 1.
        assert "s0w0" not in m
        assert "s0w1" in m

    def test_segment_without_words_list_is_skipped(self):
        segs = [
            {"speaker": "A", "start": 0.0, "end": 1.0},  # no "words" key
            _seg([_word("ok", 1.0, 2.0)]),
        ]
        m = build_word_time_map(segs)
        assert list(m.keys()) == ["s1w0"]

    def test_non_sequence_segments_raises(self):
        with pytest.raises(ProjectValidationError):
            build_word_time_map(None)  # type: ignore[arg-type]
        with pytest.raises(ProjectValidationError):
            build_word_time_map("not a list")  # type: ignore[arg-type]

    def test_text_coerced_to_string_when_present(self):
        segs = [_seg([{"text": None, "start": 0.0, "end": 0.5}])]
        m = build_word_time_map(segs)
        assert m["s0w0"].text == ""


# --------------------------------------------------------------------------- #
# playback_range_for_application — whole-word
# --------------------------------------------------------------------------- #


class TestPlaybackRangeWholeWord:
    def test_single_word_anchor(self):
        segs = _basic_transcript()
        a = _app(start_id="s0w0", end_id="s0w0")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.application_id == a.id
        assert r.source_id == _HEX_SOURCE_1
        assert r.start == pytest.approx(0.0)
        assert r.end == pytest.approx(0.5)

    def test_multi_word_anchor_within_segment(self):
        segs = _basic_transcript()
        a = _app(start_id="s0w0", end_id="s0w2")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.0)
        assert r.end == pytest.approx(1.4)

    def test_anchor_crossing_segments(self):
        segs = _basic_transcript()
        a = _app(start_id="s0w3", end_id="s1w1")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(1.4)
        assert r.end == pytest.approx(3.5)

    def test_returns_none_when_no_timing_anywhere(self):
        segs = [
            {
                "speaker": None,
                # no segment-level start/end
                "words": [
                    _word("a", None, None),
                    _word("b", None, None),
                ],
            }
        ]
        a = _app(start_id="s0w0", end_id="s0w1")
        assert playback_range_for_application(a, segs) is None

    def test_falls_back_to_segment_start_when_word_untimed(self):
        # First word has no timing; use segment.start (which has).
        segs = [
            _seg(
                [
                    _word("a", None, None),
                    _word("b", 0.5, 1.0),
                    _word("c", 1.0, 1.5),
                ],
                start=0.1,
                end=1.5,
            )
        ]
        a = _app(start_id="s0w0", end_id="s0w2")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.1)  # seg.start
        assert r.end == pytest.approx(1.5)

    def test_falls_back_to_segment_end_when_last_word_untimed(self):
        segs = [
            _seg(
                [
                    _word("a", 0.0, 0.5),
                    _word("b", None, None),
                ],
                start=0.0,
                end=2.0,
            )
        ]
        a = _app(start_id="s0w0", end_id="s0w1")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.0)
        assert r.end == pytest.approx(2.0)

    def test_clamps_to_non_negative_interval(self):
        # If end_time computes to less than start_time (e.g. weird
        # transcript edit), clamp end to start rather than seek
        # backwards.
        segs = [
            _seg(
                [
                    _word("a", 5.0, 6.0),
                    _word("b", 1.0, 2.0),  # earlier than the previous word
                ],
                start=1.0,
                end=6.0,
            )
        ]
        a = _app(start_id="s0w0", end_id="s0w1")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(5.0)
        assert r.end >= r.start


# --------------------------------------------------------------------------- #
# playback_range_for_application — sub-word
# --------------------------------------------------------------------------- #


class TestPlaybackRangeSubWord:
    def test_start_offset_interpolates_proportionally(self):
        # "Hello," is 6 chars, [0.0, 0.5] → offset 3 → 0.25.
        segs = _basic_transcript()
        a = _app(start_id="s0w0", end_id="s0w0", start_offset=3, end_offset=6)
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.25)
        assert r.end == pytest.approx(0.5)

    def test_end_offset_interpolates_proportionally(self):
        # "world." length 6, [0.5, 1.0] → end_offset 3 → 0.75.
        segs = _basic_transcript()
        a = _app(start_id="s0w1", end_id="s0w1", start_offset=0, end_offset=3)
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.5)
        assert r.end == pytest.approx(0.75)

    def test_offset_clamps_to_word_bounds(self):
        # Negative-side clamping is handled by Application.validate
        # already, but a too-large offset just clamps to the word
        # length.
        segs = _basic_transcript()
        a = _app(start_id="s0w0", end_id="s0w0", start_offset=0, end_offset=200)
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(0.0)
        assert r.end == pytest.approx(0.5)  # clamped to word.end

    def test_zero_length_word_text_falls_back_to_word_bounds(self):
        # An empty-text word can't interpolate (no chars to divide
        # over). With no offsets set, the range covers the full
        # word interval. (Sub-word offsets on an empty-text word
        # would already be ruled out by Application.validate, so
        # this is the only realistic case.)
        segs = [_seg([{"text": "", "start": 1.0, "end": 2.0}])]
        a = _app(start_id="s0w0", end_id="s0w0")
        r = playback_range_for_application(a, segs)
        assert r is not None
        assert r.start == pytest.approx(1.0)
        assert r.end == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# playback_range_for_application — validation
# --------------------------------------------------------------------------- #


class TestPlaybackRangeValidation:
    def test_anchor_segment_out_of_range_raises(self):
        segs = _basic_transcript()  # 2 segments
        a = _app(start_id="s5w0", end_id="s5w1")
        with pytest.raises(ProjectValidationError):
            playback_range_for_application(a, segs)

    def test_end_anchor_segment_out_of_range_raises(self):
        segs = _basic_transcript()
        a = _app(start_id="s0w0", end_id="s9w0")
        with pytest.raises(ProjectValidationError):
            playback_range_for_application(a, segs)

    def test_non_application_raises(self):
        with pytest.raises(ProjectValidationError):
            playback_range_for_application(
                "not-an-application",  # type: ignore[arg-type]
                _basic_transcript(),
            )

    def test_external_word_time_map_is_used(self):
        # If the caller passes a custom map, that map wins (so
        # the bulk helper can pre-build).
        segs = _basic_transcript()
        wmap = {
            "s0w0": WordTime(10.0, 11.0, "Hello,"),
            "s0w2": WordTime(12.0, 13.0, "How"),
        }
        a = _app(start_id="s0w0", end_id="s0w2")
        r = playback_range_for_application(a, segs, word_time_map=wmap)
        assert r is not None
        assert r.start == pytest.approx(10.0)
        assert r.end == pytest.approx(13.0)


# --------------------------------------------------------------------------- #
# playback_ranges_for_applications
# --------------------------------------------------------------------------- #


class TestPlaybackRangesBulk:
    def test_buckets_per_source(self):
        segs1 = _basic_transcript()
        segs2 = [
            _seg(
                [_word("alpha", 0.0, 0.4), _word("beta", 0.4, 0.8)],
                speaker="X",
            )
        ]
        a1 = _app(source_id=_HEX_SOURCE_1, start_id="s0w0", end_id="s0w0",
                  application_id=_hex_id(1))
        a2 = _app(source_id=_HEX_SOURCE_2, start_id="s0w0", end_id="s0w1",
                  application_id=_hex_id(2))
        out = playback_ranges_for_applications(
            [a1, a2],
            {_HEX_SOURCE_1: segs1, _HEX_SOURCE_2: segs2},
        )
        assert set(out.keys()) == {a1.id, a2.id}
        assert out[a1.id].source_id == _HEX_SOURCE_1
        assert out[a1.id].start == pytest.approx(0.0)
        assert out[a1.id].end == pytest.approx(0.5)
        assert out[a2.id].source_id == _HEX_SOURCE_2
        assert out[a2.id].start == pytest.approx(0.0)
        assert out[a2.id].end == pytest.approx(0.8)

    def test_skips_applications_with_unknown_source(self):
        segs = _basic_transcript()
        a = _app(source_id=_HEX_SOURCE_2, start_id="s0w0", end_id="s0w0")
        out = playback_ranges_for_applications([a], {_HEX_SOURCE_1: segs})
        assert out == {}

    def test_skips_applications_returning_none(self):
        # Untimed transcript → individual lookup returns None → skipped.
        segs = [
            {
                "speaker": None,
                "words": [_word("a", None, None)],
            }
        ]
        a = _app(start_id="s0w0", end_id="s0w0")
        out = playback_ranges_for_applications([a], {_HEX_SOURCE_1: segs})
        assert out == {}

    def test_word_time_map_built_once_per_source(self):
        # Two applications on the same source should both resolve
        # to ranges; we don't have a direct way to assert "built
        # once", but this exercises the cache path.
        segs = _basic_transcript()
        a1 = _app(start_id="s0w0", end_id="s0w0", application_id=_hex_id(1))
        a2 = _app(start_id="s0w2", end_id="s0w4", application_id=_hex_id(2))
        out = playback_ranges_for_applications([a1, a2], {_HEX_SOURCE_1: segs})
        assert set(out.keys()) == {a1.id, a2.id}
        assert out[a1.id].start == pytest.approx(0.0)
        assert out[a2.id].start == pytest.approx(1.2)

    def test_empty_input(self):
        assert playback_ranges_for_applications([], {}) == {}
