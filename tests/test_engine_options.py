"""Tests for AdvancedOptions and Word/Segment dataclasses in scribe.engine."""

from __future__ import annotations

import pytest

from scribe.engine import AdvancedOptions, Segment, Word


class TestAdvancedOptions:
    def test_defaults(self) -> None:
        opts = AdvancedOptions()
        assert opts.beam_size == 5
        assert opts.best_of == 5
        assert opts.temperature == 0.0
        assert opts.no_speech_threshold == 0.45
        assert opts.compression_ratio_threshold == 2.4
        assert opts.condition_on_previous_text is False
        assert opts.chunk_size == 30
        assert opts.vad_onset == 0.5
        assert opts.vad_offset == 0.363
        assert opts.initial_prompt == ""
        assert opts.hotwords == ""

    def test_from_dict_empty(self) -> None:
        # Empty / None inputs return defaults.
        assert AdvancedOptions.from_dict(None) == AdvancedOptions()
        assert AdvancedOptions.from_dict({}) == AdvancedOptions()

    def test_from_dict_partial(self) -> None:
        opts = AdvancedOptions.from_dict({"beam_size": 10, "initial_prompt": "hi"})
        assert opts.beam_size == 10
        assert opts.initial_prompt == "hi"
        # Untouched fields still default.
        assert opts.chunk_size == 30

    def test_from_dict_drops_none_and_empty(self) -> None:
        # None and "" are treated as "use default."
        opts = AdvancedOptions.from_dict({
            "beam_size": None,
            "initial_prompt": "",
            "best_of": 7,
        })
        assert opts.beam_size == 5
        assert opts.initial_prompt == ""
        assert opts.best_of == 7

    def test_from_dict_ignores_unknown_keys(self) -> None:
        # Unknown keys must not trigger a TypeError on construction.
        opts = AdvancedOptions.from_dict({"beam_size": 3, "garbage": "x"})
        assert opts.beam_size == 3

    def test_asr_options_shape(self) -> None:
        opts = AdvancedOptions(
            beam_size=8,
            best_of=4,
            temperature=0.2,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.0,
            condition_on_previous_text=True,
            initial_prompt="hello",
            hotwords="foo, bar",
        )
        out = opts.asr_options()
        assert out["beam_size"] == 8
        assert out["best_of"] == 4
        # temperatures is a single-element tuple — we suppress faster-whisper's
        # default 6-step fallback ladder
        assert out["temperatures"] == (0.2,)
        assert out["no_speech_threshold"] == 0.5
        assert out["compression_ratio_threshold"] == 2.0
        assert out["condition_on_previous_text"] is True
        assert out["initial_prompt"] == "hello"
        assert out["hotwords"] == "foo, bar"

    def test_asr_options_empty_strings_become_none(self) -> None:
        # faster-whisper expects None, not empty string, to mean "no prompt".
        opts = AdvancedOptions()
        out = opts.asr_options()
        assert out["initial_prompt"] is None
        assert out["hotwords"] is None

    def test_vad_options(self) -> None:
        opts = AdvancedOptions(chunk_size=15, vad_onset=0.4, vad_offset=0.3)
        out = opts.vad_options()
        assert out == {"chunk_size": 15, "vad_onset": 0.4, "vad_offset": 0.3}

    def test_temperatures_zero_default(self) -> None:
        # Default temperature=0.0 must still produce a tuple (not bare 0.0).
        opts = AdvancedOptions()
        assert opts.asr_options()["temperatures"] == (0.0,)


class TestWordSegmentDataclasses:
    def test_word_to_dict(self) -> None:
        w = Word(text="hi", start=0.0, end=0.5, speaker="S0", score=0.9)
        d = w.to_dict()
        assert d == {"text": "hi", "start": 0.0, "end": 0.5, "speaker": "S0", "score": 0.9}

    def test_word_score_optional(self) -> None:
        w = Word(text="hi", start=0.0, end=0.5, speaker="S0")
        assert w.to_dict()["score"] is None

    def test_segment_to_dict_includes_words(self) -> None:
        seg = Segment(
            text="hi there",
            start=0.0,
            end=1.0,
            speaker="S0",
            words=[Word("hi", 0.0, 0.4, "S0", 0.9), Word("there", 0.5, 1.0, "S0", 0.95)],
        )
        d = seg.to_dict()
        assert d["text"] == "hi there"
        assert len(d["words"]) == 2
        assert d["words"][0]["text"] == "hi"
        assert d["speaker"] == "S0"
