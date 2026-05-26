"""Tests for ``scribe.transcript_tidy`` — the pure helpers behind the
editor's "Tidy speech with AI" feature.

The module's job is twofold:

1. Group consecutive same-speaker segments into a *run* the LLM can
   rewrite as a unit.
2. Realign the LLM's word-level output back onto the original audio
   timestamps so playback word highlighting still tracks.

Both pieces must be deterministic, never raise on hand-edited
transcripts, and produce monotone-non-decreasing word timestamps.
"""

from __future__ import annotations

from typing import Any

import pytest

from scribe import transcript_tidy as tidy


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _seg(*, speaker: str, start: float, end: float, text: str,
         words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if words is None:
        words = _words_for(text, speaker, start, end)
    return {"speaker": speaker, "start": start, "end": end, "text": text, "words": words}


def _words_for(text: str, speaker: str, start: float, end: float) -> list[dict[str, Any]]:
    """Build word records spread evenly across [start, end]."""
    toks = text.split()
    if not toks:
        return []
    span = (end - start) / len(toks)
    return [
        {
            "text": tok,
            "start": start + i * span,
            "end": start + (i + 1) * span,
            "speaker": speaker,
        }
        for i, tok in enumerate(toks)
    ]


# --------------------------------------------------------------------------- #
# Run grouping
# --------------------------------------------------------------------------- #


class TestGroupRuns:
    def test_groups_consecutive_same_speaker(self) -> None:
        segs = [
            _seg(speaker="A", start=0.0, end=2.0, text="hello there"),
            _seg(speaker="A", start=2.0, end=4.0, text="how are you"),
            _seg(speaker="B", start=4.0, end=6.0, text="i am fine"),
            _seg(speaker="A", start=6.0, end=8.0, text="great great"),
            _seg(speaker="A", start=8.0, end=10.0, text="lovely day"),
        ]
        runs = tidy.group_runs(segs)
        # Two runs (A's first pair and A's second pair). B's lone
        # segment is dropped — singletons aren't tidy candidates.
        assert len(runs) == 2
        assert runs[0].speaker == "A"
        assert runs[0].segment_indices == (0, 1)
        assert runs[0].start == 0.0 and runs[0].end == 4.0
        assert runs[0].text == "hello there how are you"
        assert runs[1].speaker == "A"
        assert runs[1].segment_indices == (3, 4)

    def test_singleton_runs_dropped(self) -> None:
        runs = tidy.group_runs([
            _seg(speaker="A", start=0, end=1, text="a"),
            _seg(speaker="B", start=1, end=2, text="b"),
            _seg(speaker="A", start=2, end=3, text="c"),
        ])
        assert runs == []

    def test_empty_input(self) -> None:
        assert tidy.group_runs([]) == []

    def test_speaker_normalisation(self) -> None:
        # Whitespace + None speakers normalise so they group correctly.
        segs = [
            _seg(speaker="  A  ", start=0, end=1, text="x"),
            _seg(speaker="A", start=1, end=2, text="y"),
        ]
        runs = tidy.group_runs(segs)
        assert len(runs) == 1
        assert runs[0].speaker == "A"

    def test_word_count_caps_drop_run(self) -> None:
        # A run that exceeds MAX_RUN_WORDS is dropped.
        big_text = " ".join(["word"] * (tidy.MAX_RUN_WORDS + 50))
        segs = [
            _seg(speaker="A", start=0, end=10, text=big_text),
            _seg(speaker="A", start=10, end=20, text="more"),
        ]
        assert tidy.group_runs(segs) == []

    def test_duration_cap_drops_run(self) -> None:
        segs = [
            _seg(speaker="A", start=0.0, end=tidy.MAX_RUN_DURATION_S - 1, text="x"),
            _seg(speaker="A", start=tidy.MAX_RUN_DURATION_S - 1,
                 end=tidy.MAX_RUN_DURATION_S + 60, text="y"),
        ]
        assert tidy.group_runs(segs) == []

    def test_run_to_dict_shape(self) -> None:
        segs = [
            _seg(speaker="A", start=0, end=1, text="hello"),
            _seg(speaker="A", start=1, end=2, text="there"),
        ]
        d = tidy.group_runs(segs)[0].to_dict()
        assert set(d.keys()) == {
            "speaker", "segment_indices", "start", "end",
            "text", "word_count", "duration_s",
        }
        assert d["segment_indices"] == [0, 1]


# --------------------------------------------------------------------------- #
# Prompt + parse
# --------------------------------------------------------------------------- #


class TestPrompt:
    def test_prompt_contains_run_text(self) -> None:
        p = tidy.build_tidy_prompt("hello world")
        assert "hello world" in p
        # System prompt keywords the editor depends on.
        assert "verbatim" in p.lower()
        assert "paragraph" in p.lower()


class TestParseTidiedParagraphs:
    def test_splits_on_blank_lines(self) -> None:
        body = "First paragraph.\n\nSecond paragraph.\n\nThird."
        assert tidy.parse_tidied_paragraphs(body) == [
            "First paragraph.", "Second paragraph.", "Third.",
        ]

    def test_collapses_internal_newlines(self) -> None:
        body = "Line one\nstill line one.\n\nNew paragraph."
        ps = tidy.parse_tidied_paragraphs(body)
        assert ps[0] == "Line one still line one."
        assert ps[1] == "New paragraph."

    def test_strips_fenced_block_wrappers(self) -> None:
        for body in ("```\nhello\n```", "```text\nhello\n```"):
            assert tidy.parse_tidied_paragraphs(body) == ["hello"]

    def test_strips_bullet_prefixes(self) -> None:
        body = "- one\n\n* two\n\n• three"
        assert tidy.parse_tidied_paragraphs(body) == ["one", "two", "three"]

    def test_empty_input_returns_empty(self) -> None:
        assert tidy.parse_tidied_paragraphs("") == []
        assert tidy.parse_tidied_paragraphs("   \n\n  ") == []


# --------------------------------------------------------------------------- #
# Realignment
# --------------------------------------------------------------------------- #


class TestRealignWords:
    def test_identical_text_preserves_timestamps(self) -> None:
        # Words match 1:1 → every proposed word keeps its source timestamp.
        old = [
            {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "A"},
            {"text": "world", "start": 0.5, "end": 1.0, "speaker": "A"},
        ]
        para_words = tidy.realign_words(
            old, ["Hello world."],
            fallback_start=0.0, fallback_end=1.0,
        )
        assert len(para_words) == 1
        flat = para_words[0]
        # Two tokens, capitalisation/punct stripped on the diff path.
        assert [w["text"] for w in flat] == ["Hello", "world."]
        assert flat[0]["start"] == pytest.approx(0.0)
        assert flat[0]["end"] == pytest.approx(0.5)
        assert flat[1]["start"] == pytest.approx(0.5)
        assert flat[1]["end"] == pytest.approx(1.0)

    def test_inserted_word_is_interpolated_into_gap(self) -> None:
        # Source: "hello world"   (0.0–0.5, 0.5–1.0)
        # Tidied: "hello there world"  — "there" gets timestamps in the
        # gap between "hello" end (0.5) and "world" start (0.5). Since
        # there's no gap, the new word lands at that pinch point.
        old = [
            {"text": "hello", "start": 0.0, "end": 0.5, "speaker": "A"},
            {"text": "world", "start": 0.5, "end": 1.0, "speaker": "A"},
        ]
        out = tidy.realign_words(
            old, ["hello there world"],
            fallback_start=0.0, fallback_end=1.0,
        )
        words = out[0]
        assert [w["text"] for w in words] == ["hello", "there", "world"]
        # Timestamps must be monotone non-decreasing.
        starts = [w["start"] for w in words]
        ends = [w["end"] for w in words]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1]
            assert ends[i] >= ends[i - 1]
        # And bounded by the original anchors.
        assert words[0]["start"] == pytest.approx(0.0)
        assert words[2]["end"] == pytest.approx(1.0)

    def test_inserted_word_uses_real_gap(self) -> None:
        # Source has a 1.0s gap between "hello" (ends 1.0) and "world"
        # (starts 2.0). An inserted "there" should land somewhere in that
        # gap, not at the pinch points.
        old = [
            {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "A"},
            {"text": "world", "start": 2.0, "end": 3.0, "speaker": "A"},
        ]
        out = tidy.realign_words(
            old, ["hello there world"],
            fallback_start=0.0, fallback_end=3.0,
        )
        there = out[0][1]
        assert there["text"] == "there"
        # Lies strictly between the surrounding anchors.
        assert 1.0 <= there["start"] < 2.0
        assert 1.0 < there["end"] <= 2.0

    def test_completely_new_text_falls_back_to_even_spread(self) -> None:
        # The proposed text shares no tokens with the original. Every
        # word is interpolated across [fallback_start, fallback_end].
        old = [{"text": "alpha", "start": 0.0, "end": 1.0, "speaker": "A"}]
        out = tidy.realign_words(
            old, ["totally different stuff entirely"],
            fallback_start=0.0, fallback_end=4.0,
        )
        words = out[0]
        assert [w["text"] for w in words] == ["totally", "different", "stuff", "entirely"]
        # Even spread across [0, 4]: each word gets a 1.0s window.
        for i, w in enumerate(words):
            assert w["start"] == pytest.approx(i * 1.0)
            assert w["end"] == pytest.approx((i + 1) * 1.0)

    def test_paragraph_split_is_preserved(self) -> None:
        # Two paragraphs out → two output lists.
        old = [
            {"text": "one", "start": 0.0, "end": 1.0, "speaker": "A"},
            {"text": "two", "start": 1.0, "end": 2.0, "speaker": "A"},
            {"text": "three", "start": 2.0, "end": 3.0, "speaker": "A"},
        ]
        out = tidy.realign_words(
            old, ["one two.", "three."],
            fallback_start=0.0, fallback_end=3.0,
        )
        assert len(out) == 2
        assert [w["text"] for w in out[0]] == ["one", "two."]
        assert [w["text"] for w in out[1]] == ["three."]
        # Paragraph boundary doesn't break monotone time.
        assert out[1][0]["start"] >= out[0][-1]["end"]

    def test_empty_old_words_uses_fallback_only(self) -> None:
        out = tidy.realign_words(
            [], ["a b c"],
            fallback_start=10.0, fallback_end=13.0,
        )
        words = out[0]
        assert [w["text"] for w in words] == ["a", "b", "c"]
        # Even spread across [10, 13].
        assert words[0]["start"] == pytest.approx(10.0)
        assert words[-1]["end"] == pytest.approx(13.0)

    def test_empty_paragraphs_returns_empty(self) -> None:
        out = tidy.realign_words(
            [{"text": "hello", "start": 0, "end": 1, "speaker": "A"}],
            [], fallback_start=0, fallback_end=1,
        )
        assert out == []

    def test_speaker_propagates_from_old(self) -> None:
        old = [{"text": "hello", "start": 0.0, "end": 1.0, "speaker": "A"}]
        out = tidy.realign_words(
            old, ["Hello!"], fallback_start=0.0, fallback_end=1.0,
        )
        # The matched token picks up the source speaker.
        assert out[0][0]["speaker"] == "A"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


class TestAssembleTidiedSegments:
    def test_one_segment_per_paragraph(self) -> None:
        paragraphs = ["Hello world.", "Goodbye."]
        para_words = [
            [{"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "A"},
             {"text": "world.", "start": 0.5, "end": 1.0, "speaker": "A"}],
            [{"text": "Goodbye.", "start": 1.0, "end": 1.5, "speaker": "A"}],
        ]
        segs = tidy.assemble_tidied_segments(
            paragraphs=paragraphs,
            paragraph_words=para_words,
            speaker="A",
            fallback_start=0.0,
            fallback_end=1.5,
        )
        assert len(segs) == 2
        assert segs[0].speaker == "A"
        assert segs[0].text == "Hello world."
        assert segs[0].start == 0.0
        assert segs[0].end == 1.0
        assert segs[1].text == "Goodbye."

    def test_empty_paragraph_dropped(self) -> None:
        segs = tidy.assemble_tidied_segments(
            paragraphs=["Hello.", "  ", "Bye."],
            paragraph_words=[
                [{"text": "Hello.", "start": 0.0, "end": 1.0, "speaker": "A"}],
                [],
                [{"text": "Bye.", "start": 1.0, "end": 2.0, "speaker": "A"}],
            ],
            speaker="A",
            fallback_start=0.0,
            fallback_end=2.0,
        )
        assert len(segs) == 2

    def test_paragraph_with_no_words_uses_fallback_bounds(self) -> None:
        segs = tidy.assemble_tidied_segments(
            paragraphs=["lone"],
            paragraph_words=[[]],
            speaker="A",
            fallback_start=0.0, fallback_end=5.0,
        )
        assert len(segs) == 1
        assert segs[0].start == 0.0
        assert segs[0].end == 5.0

    def test_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            tidy.assemble_tidied_segments(
                paragraphs=["a", "b"],
                paragraph_words=[[]],
                speaker="A",
                fallback_start=0.0, fallback_end=1.0,
            )

    def test_to_dict_round_trip(self) -> None:
        seg = tidy.TidiedSegment(
            speaker="A", start=0.0, end=1.0, text="hi",
            words=[{"text": "hi", "start": 0, "end": 1, "speaker": "A"}],
        )
        d = seg.to_dict()
        assert d["speaker"] == "A"
        assert d["text"] == "hi"
        assert d["words"][0]["text"] == "hi"


# --------------------------------------------------------------------------- #
# Splice
# --------------------------------------------------------------------------- #


class TestSpliceRun:
    def test_replaces_contiguous_range(self) -> None:
        original = {"segments": [
            _seg(speaker="A", start=0, end=1, text="x"),
            _seg(speaker="A", start=1, end=2, text="y"),
            _seg(speaker="A", start=2, end=3, text="z"),
            _seg(speaker="B", start=3, end=4, text="b"),
        ]}
        new = [_seg(speaker="A", start=0, end=3, text="x y z combined")]
        out = tidy.splice_run(original, segment_indices=[0, 1, 2], new_segments=new)
        assert [s["text"] for s in out["segments"]] == ["x y z combined", "b"]

    def test_does_not_mutate_input(self) -> None:
        original = {"segments": [
            _seg(speaker="A", start=0, end=1, text="x"),
            _seg(speaker="A", start=1, end=2, text="y"),
        ]}
        out = tidy.splice_run(
            original,
            segment_indices=[0, 1],
            new_segments=[_seg(speaker="A", start=0, end=2, text="combined")],
        )
        # Original survives untouched.
        assert [s["text"] for s in original["segments"]] == ["x", "y"]
        assert [s["text"] for s in out["segments"]] == ["combined"]

    def test_non_contiguous_raises(self) -> None:
        original = {"segments": [
            _seg(speaker="A", start=0, end=1, text="x"),
            _seg(speaker="A", start=1, end=2, text="y"),
            _seg(speaker="A", start=2, end=3, text="z"),
        ]}
        with pytest.raises(ValueError):
            tidy.splice_run(
                original, segment_indices=[0, 2], new_segments=[],
            )

    def test_out_of_range_raises(self) -> None:
        original = {"segments": [_seg(speaker="A", start=0, end=1, text="x")]}
        with pytest.raises(ValueError):
            tidy.splice_run(
                original, segment_indices=[5, 6], new_segments=[],
            )

    def test_empty_indices_raises(self) -> None:
        with pytest.raises(ValueError):
            tidy.splice_run(
                {"segments": []}, segment_indices=[], new_segments=[],
            )

    def test_non_dict_raises(self) -> None:
        with pytest.raises(TypeError):
            tidy.splice_run("not a dict", segment_indices=[0], new_segments=[])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end: realistic round trip via the public surface
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_full_pipeline_tidy_a_messy_run(self) -> None:
        """Simulate: model returns clean text; we realign onto original
        timestamps; we splice into the transcript.

        This is the test that proves "playback word highlighting still
        works after a tidy" — every output word has a finite, monotone
        timestamp inside the original run's wall-clock window.
        """
        transcript = {"segments": [
            # The speaker says "i, i think... um, this is great."
            # in three segments.
            _seg(speaker="A", start=0.0, end=1.0, text="i i think"),
            _seg(speaker="A", start=1.0, end=2.0, text="um this is"),
            _seg(speaker="A", start=2.0, end=3.0, text="great"),
            # Second speaker stays untouched.
            _seg(speaker="B", start=3.0, end=4.0, text="agreed"),
        ]}
        runs = tidy.group_runs(transcript["segments"])
        assert len(runs) == 1
        run = runs[0]

        # Pretend the LLM returns one paragraph with disfluencies removed.
        model_response = "I think this is great."
        paragraphs = tidy.parse_tidied_paragraphs(model_response)
        assert paragraphs == ["I think this is great."]

        old_words = tidy._flatten_old_words(run, transcript["segments"])
        para_words = tidy.realign_words(
            old_words, paragraphs,
            fallback_start=run.start, fallback_end=run.end,
        )
        new_segs = tidy.assemble_tidied_segments(
            paragraphs=paragraphs,
            paragraph_words=para_words,
            speaker=run.speaker,
            fallback_start=run.start, fallback_end=run.end,
        )

        # Every output word has a timestamp inside [run.start, run.end].
        for seg in new_segs:
            for w in seg.words:
                assert run.start <= w["start"] <= run.end
                assert run.start <= w["end"] <= run.end
                assert w["start"] <= w["end"]

        # Splice the run back into the transcript and check B's segment
        # survives untouched.
        out = tidy.splice_run(
            transcript,
            segment_indices=run.segment_indices,
            new_segments=[s.to_dict() for s in new_segs],
        )
        # Last segment is still B's "agreed".
        assert out["segments"][-1]["speaker"] == "B"
        assert out["segments"][-1]["text"] == "agreed"
        # The A run shrank from 3 segments to 1.
        a_segs = [s for s in out["segments"] if s["speaker"] == "A"]
        assert len(a_segs) == 1
        assert a_segs[0]["text"] == "I think this is great."
