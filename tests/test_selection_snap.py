"""Tests for scribe.selection_snap (F4.4).

F4.4 layers a small set of pure functions over the F4.1 anchor format
that snap a candidate selection to word / sentence / paragraph
boundaries. These tests cover:

* Selection record shape
* _is_sentence_final / sentence_ranges_in_segment
* paragraph_ranges
* snap_to_word / snap_to_sentence / snap_to_paragraph (including
  cross-segment, idempotency, and validation paths)

The module is stand-alone (no FastAPI, no engine), so the tests are
pure-Python.
"""

from __future__ import annotations

import pytest

from scribe.applications import ProjectValidationError
from scribe.selection_snap import (
    Selection,
    _is_sentence_final,
    paragraph_ranges,
    sentence_ranges_in_segment,
    snap_to_paragraph,
    snap_to_sentence,
    snap_to_word,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _seg(words, speaker=None):
    return {"speaker": speaker, "words": [{"text": w} for w in words]}


def _sel(start, end, so=None, eo=None):
    return Selection(
        start_word_id=start,
        end_word_id=end,
        start_char_offset=so,
        end_char_offset=eo,
    )


# --------------------------------------------------------------------------- #
# _is_sentence_final
# --------------------------------------------------------------------------- #


class TestIsSentenceFinal:
    def test_period_question_exclaim(self):
        assert _is_sentence_final("hello.")
        assert _is_sentence_final("really?")
        assert _is_sentence_final("stop!")

    def test_strips_closing_quotes_and_brackets(self):
        assert _is_sentence_final('hello."')
        assert _is_sentence_final("(stop!)")
        assert _is_sentence_final("done.]")
        assert _is_sentence_final("quiet?'")

    def test_ellipsis_counts_as_final(self):
        # We deliberately treat trailing dots as sentence-final.
        assert _is_sentence_final("trailing...")

    def test_non_final_words(self):
        assert not _is_sentence_final("hello")
        assert not _is_sentence_final("comma,")
        assert not _is_sentence_final("dash-")
        assert not _is_sentence_final("colon:")

    def test_empty_and_whitespace(self):
        assert not _is_sentence_final("")
        assert not _is_sentence_final("   ")
        assert not _is_sentence_final('"')
        assert not _is_sentence_final(")")

    def test_non_string_inputs(self):
        # Defensive: junk in word.text shouldn't crash.
        assert not _is_sentence_final(None)
        assert not _is_sentence_final(42)


# --------------------------------------------------------------------------- #
# sentence_ranges_in_segment
# --------------------------------------------------------------------------- #


class TestSentenceRanges:
    def test_single_sentence(self):
        words = [{"text": "Hello"}, {"text": "world."}]
        assert sentence_ranges_in_segment(words) == [(0, 1)]

    def test_multiple_sentences(self):
        words = [
            {"text": "Hello."},
            {"text": "How"},
            {"text": "are"},
            {"text": "you?"},
            {"text": "Goodbye!"},
        ]
        assert sentence_ranges_in_segment(words) == [(0, 0), (1, 3), (4, 4)]

    def test_unterminated_trailing_words(self):
        # A segment that doesn't end with sentence punctuation still
        # gets a final sentence covering the trailing words.
        words = [
            {"text": "Hello."},
            {"text": "And"},
            {"text": "then"},
        ]
        assert sentence_ranges_in_segment(words) == [(0, 0), (1, 2)]

    def test_no_terminators_at_all(self):
        words = [{"text": "no"}, {"text": "punctuation"}, {"text": "here"}]
        assert sentence_ranges_in_segment(words) == [(0, 2)]

    def test_empty(self):
        assert sentence_ranges_in_segment([]) == []

    def test_partition_invariant(self):
        # The ranges must partition [0, len(words)) exactly.
        words = [
            {"text": "a."}, {"text": "b"}, {"text": "c?"}, {"text": "d"},
            {"text": "e"}, {"text": "f!"},
        ]
        ranges = sentence_ranges_in_segment(words)
        flat = [i for r in ranges for i in range(r[0], r[1] + 1)]
        assert flat == list(range(len(words)))


# --------------------------------------------------------------------------- #
# paragraph_ranges
# --------------------------------------------------------------------------- #


class TestParagraphRanges:
    def test_single_speaker_run(self):
        segs = [
            _seg(["a"], speaker="S0"),
            _seg(["b"], speaker="S0"),
            _seg(["c"], speaker="S0"),
        ]
        assert paragraph_ranges(segs) == [(0, 2)]

    def test_speaker_change_breaks_paragraph(self):
        segs = [
            _seg(["a"], speaker="S0"),
            _seg(["b"], speaker="S1"),
            _seg(["c"], speaker="S1"),
            _seg(["d"], speaker="S0"),
        ]
        assert paragraph_ranges(segs) == [(0, 0), (1, 2), (3, 3)]

    def test_missing_speaker_is_own_paragraph(self):
        segs = [
            _seg(["a"], speaker="S0"),
            _seg(["b"], speaker=None),
            _seg(["c"], speaker=None),
            _seg(["d"], speaker="S0"),
        ]
        # None never groups with anything (not even another None).
        assert paragraph_ranges(segs) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_empty(self):
        assert paragraph_ranges([]) == []

    def test_single_segment(self):
        assert paragraph_ranges([_seg(["a"], speaker="S0")]) == [(0, 0)]

    def test_partition_invariant(self):
        segs = [
            _seg(["a"], speaker="S0"),
            _seg(["b"], speaker="S0"),
            _seg(["c"], speaker="S1"),
            _seg(["d"], speaker="S2"),
            _seg(["e"], speaker="S2"),
        ]
        ranges = paragraph_ranges(segs)
        flat = [i for r in ranges for i in range(r[0], r[1] + 1)]
        assert flat == list(range(len(segs)))


# --------------------------------------------------------------------------- #
# snap_to_word
# --------------------------------------------------------------------------- #


class TestSnapToWord:
    def test_drops_offsets(self):
        sel = _sel("s0w0", "s0w5", so=2, eo=4)
        out = snap_to_word(sel)
        assert out.start_word_id == "s0w0"
        assert out.end_word_id == "s0w5"
        assert out.start_char_offset is None
        assert out.end_char_offset is None

    def test_idempotent(self):
        sel = _sel("s0w0", "s0w5")
        once = snap_to_word(sel)
        twice = snap_to_word(once)
        assert once == twice
        assert once is sel  # no copy when already snapped

    def test_validates_word_ids(self):
        with pytest.raises(ProjectValidationError):
            snap_to_word(_sel("invalid", "s0w0"))
        with pytest.raises(ProjectValidationError):
            snap_to_word(_sel("s0w0", "S0W0"))  # uppercase rejected

    def test_rejects_non_selection(self):
        with pytest.raises(ProjectValidationError):
            snap_to_word("not a selection")  # type: ignore[arg-type]

    def test_partial_offsets_dropped(self):
        sel = _sel("s0w0", "s0w5", so=3, eo=None)
        out = snap_to_word(sel)
        assert out.start_char_offset is None
        assert out.end_char_offset is None


# --------------------------------------------------------------------------- #
# snap_to_sentence
# --------------------------------------------------------------------------- #


class TestSnapToSentence:
    def _transcript(self):
        # Segment 0: "Hello. How are you? Goodbye!"   (3 sentences)
        # Segment 1: "I think so."                    (1 sentence)
        # Segment 2: "And then we left and went home" (1 sentence, no terminator)
        return [
            _seg(["Hello.", "How", "are", "you?", "Goodbye!"], speaker="S0"),
            _seg(["I", "think", "so."], speaker="S0"),
            _seg(["And", "then", "we", "left", "and", "went", "home"], speaker="S0"),
        ]

    def test_extends_to_sentence_within_segment(self):
        sel = _sel("s0w2", "s0w2")  # word "are"
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_word_id == "s0w1"  # "How"
        assert out.end_word_id == "s0w3"    # "you?"

    def test_endpoint_already_at_sentence_boundary(self):
        sel = _sel("s0w1", "s0w3")  # exactly "How are you?"
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_word_id == "s0w1"
        assert out.end_word_id == "s0w3"

    def test_drops_offsets(self):
        sel = _sel("s0w2", "s0w2", so=1, eo=2)
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_char_offset is None
        assert out.end_char_offset is None

    def test_cross_segment(self):
        # Selection from "are" in seg0 to "think" in seg1.
        sel = _sel("s0w2", "s1w1")
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_word_id == "s0w1"  # "How"
        assert out.end_word_id == "s1w2"    # "so."

    def test_unterminated_segment(self):
        # Selection in segment 2 (no terminator) snaps to whole seg.
        sel = _sel("s2w3", "s2w3")
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_word_id == "s2w0"
        assert out.end_word_id == "s2w6"

    def test_idempotent(self):
        sel = _sel("s0w2", "s0w4")
        once = snap_to_sentence(sel, self._transcript())
        twice = snap_to_sentence(once, self._transcript())
        assert once == twice

    def test_word_out_of_range(self):
        with pytest.raises(ProjectValidationError):
            snap_to_sentence(_sel("s0w99", "s0w99"), self._transcript())

    def test_segment_out_of_range(self):
        with pytest.raises(ProjectValidationError):
            snap_to_sentence(_sel("s9w0", "s9w0"), self._transcript())

    def test_start_after_end(self):
        with pytest.raises(ProjectValidationError):
            snap_to_sentence(_sel("s0w3", "s0w1"), self._transcript())

    def test_rejects_bad_segments(self):
        with pytest.raises(ProjectValidationError):
            snap_to_sentence(_sel("s0w0", "s0w0"), "not a list")  # type: ignore[arg-type]

    def test_rejects_non_selection(self):
        with pytest.raises(ProjectValidationError):
            snap_to_sentence("nope", self._transcript())  # type: ignore[arg-type]

    def test_first_sentence_starts_at_word_0(self):
        sel = _sel("s0w0", "s0w0")  # "Hello."
        out = snap_to_sentence(sel, self._transcript())
        assert out.start_word_id == "s0w0"
        assert out.end_word_id == "s0w0"


# --------------------------------------------------------------------------- #
# snap_to_paragraph
# --------------------------------------------------------------------------- #


class TestSnapToParagraph:
    def _transcript(self):
        # Speakers:    S0    S0    S1    S1    S0
        # Segments:    0     1     2     3     4
        # Words:       3     2     2     3     2
        return [
            _seg(["Hello.", "How", "are"], speaker="S0"),
            _seg(["you?", "Yes."], speaker="S0"),
            _seg(["I'm", "fine."], speaker="S1"),
            _seg(["Are", "you", "sure?"], speaker="S1"),
            _seg(["OK", "good."], speaker="S0"),
        ]

    def test_extends_to_paragraph_in_speaker_run(self):
        sel = _sel("s0w1", "s0w2")
        out = snap_to_paragraph(sel, self._transcript())
        # paragraph 0–1 (S0), first word s0w0, last word s1w1
        assert out.start_word_id == "s0w0"
        assert out.end_word_id == "s1w1"

    def test_single_segment_paragraph(self):
        sel = _sel("s4w0", "s4w0")
        out = snap_to_paragraph(sel, self._transcript())
        assert out.start_word_id == "s4w0"
        assert out.end_word_id == "s4w1"

    def test_cross_paragraph(self):
        # Start in S0 paragraph (segs 0-1), end in S1 paragraph (segs 2-3).
        sel = _sel("s1w0", "s2w0")
        out = snap_to_paragraph(sel, self._transcript())
        assert out.start_word_id == "s0w0"
        assert out.end_word_id == "s3w2"

    def test_drops_offsets(self):
        sel = _sel("s0w1", "s0w2", so=2, eo=3)
        out = snap_to_paragraph(sel, self._transcript())
        assert out.start_char_offset is None
        assert out.end_char_offset is None

    def test_missing_speaker_is_own_paragraph(self):
        segs = [
            _seg(["a", "b"], speaker="S0"),
            _seg(["c", "d"], speaker=None),
            _seg(["e", "f"], speaker="S0"),
        ]
        sel = _sel("s1w0", "s1w0")
        out = snap_to_paragraph(sel, segs)
        # The None-speaker segment is its own paragraph.
        assert out.start_word_id == "s1w0"
        assert out.end_word_id == "s1w1"

    def test_idempotent(self):
        sel = _sel("s0w1", "s0w2")
        once = snap_to_paragraph(sel, self._transcript())
        twice = snap_to_paragraph(once, self._transcript())
        assert once == twice

    def test_word_out_of_range(self):
        with pytest.raises(ProjectValidationError):
            snap_to_paragraph(_sel("s0w99", "s0w99"), self._transcript())

    def test_start_after_end(self):
        with pytest.raises(ProjectValidationError):
            snap_to_paragraph(_sel("s2w0", "s0w0"), self._transcript())

    def test_rejects_non_selection(self):
        with pytest.raises(ProjectValidationError):
            snap_to_paragraph("nope", self._transcript())  # type: ignore[arg-type]

    def test_rejects_bad_segments(self):
        with pytest.raises(ProjectValidationError):
            snap_to_paragraph(_sel("s0w0", "s0w0"), "not a list")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Snap composition
# --------------------------------------------------------------------------- #


class TestSnapComposition:
    """Snap helpers compose meaningfully: word ⊆ sentence ⊆ paragraph."""

    def test_widening_chain(self):
        # Segment 0: "Hello. How are you? Goodbye!"
        # Segment 1: "I'm fine."  (different speaker)
        segs = [
            _seg(["Hello.", "How", "are", "you?", "Goodbye!"], speaker="S0"),
            _seg(["I'm", "fine."], speaker="S1"),
        ]
        sel = _sel("s0w2", "s0w2", so=1, eo=2)

        word_snapped = snap_to_word(sel)
        sent_snapped = snap_to_sentence(sel, segs)
        para_snapped = snap_to_paragraph(sel, segs)

        # word: same single word, no offsets
        assert (word_snapped.start_word_id, word_snapped.end_word_id) == ("s0w2", "s0w2")
        # sentence: "How are you?"
        assert (sent_snapped.start_word_id, sent_snapped.end_word_id) == ("s0w1", "s0w3")
        # paragraph: whole segment 0 (S0 paragraph is just seg 0)
        assert (para_snapped.start_word_id, para_snapped.end_word_id) == ("s0w0", "s0w4")
