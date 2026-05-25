"""Snap-to-word / sentence / paragraph selection helpers (F4.4).

Per PLANNING.md F4.4:

  > Snap-to-word / sentence / paragraph selection helpers.

When a coder drags a highlight across a transcript, the raw selection
endpoints rarely fall exactly where the analytic unit ends — the
mouse-up lands inside a word, a few characters before the comma, half-
way through a thought. F4.4 gives the editor (and any other surface
that constructs an :class:`scribe.applications.Application`) a small
toolkit of pure functions that *snap* a candidate selection to a
clean boundary:

* :func:`snap_to_word` — drop sub-word character offsets entirely;
  the result covers full words from start to end.
* :func:`snap_to_sentence` — extend the selection so its endpoints
  land on sentence boundaries within the transcript. Sentence ends
  are detected by trailing ``.``, ``?``, or ``!`` (after stripping
  closing quote / bracket punctuation).
* :func:`snap_to_paragraph` — extend to the entire speaker turn
  (paragraph). Paragraphs are runs of consecutive segments that share
  the same ``speaker`` value; segments without speaker info are each
  their own paragraph.

The transcript is passed in the canonical Scribe shape::

    segments: [
      {
        "speaker": "SPEAKER_00" | None,
        "words": [{"text": "Hello,"}, ...],
        ...
      },
      ...
    ]

Only the ``speaker`` and ``words[].text`` keys are read; timing and
score fields are ignored. This keeps the helpers usable on raw ASR
output, edited transcripts, and imported transcripts (F10.3) alike.

Design choices, all defensible:

* The helpers are **idempotent**: snapping an already-snapped
  selection returns an equivalent selection.
* Snap **never narrows**: the result always covers a span at least as
  large as the input. A selection that crosses three sentences snaps
  to the union of those three sentences, not the middle one.
* Sub-word character offsets are **dropped** by all three snap
  functions — F4.4 boundaries are word-level by definition. Callers
  that want sub-word precision should not snap.
* Empty / mis-shaped transcripts surface as
  :class:`scribe.applications.ProjectValidationError`. Out-of-range
  word ids do too — silently clamping would mask real bugs.

This module is stand-alone (no FastAPI, no engine) and matches the
conventions of :mod:`scribe.application_spans`,
:mod:`scribe.application_gutter`, :mod:`scribe.code_lifecycle`, and
:mod:`scribe.matrix`. A JS mirror lives in
``scribe/static/js/helpers.mjs`` (``snapToWord``, ``snapToSentence``,
``snapToParagraph``) and must produce identical results for any
shared input — there is a parallel test suite in
``tests/js/selection-snap.test.mjs``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .applications import make_word_id, parse_word_id
from .projects import ProjectValidationError


# --------------------------------------------------------------------------- #
# Selection record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Selection:
    """A candidate text selection over a Scribe transcript.

    Mirrors the anchor fields of
    :class:`scribe.applications.Application` but without the
    cross-entity ids — the snap helpers only care about *where* the
    span lies, not which code / source / coder it belongs to.

    The endpoints follow the same closed-interval convention as
    Application: ``[start_word_id, end_word_id]`` is inclusive on
    both ends, with optional sub-word offsets (``None`` = whole word).
    """

    start_word_id: str
    end_word_id: str
    start_char_offset: int | None = None
    end_char_offset: int | None = None


# --------------------------------------------------------------------------- #
# Sentence detection
# --------------------------------------------------------------------------- #


# Closing punctuation characters to strip *off the end* of a token
# before testing for sentence-final punctuation. ASR output sometimes
# emits ``"hello."`` or ``(yes!)`` — we want both to count as
# sentence-final.
_CLOSING_TRIM = '"\')]}»›”’'

# Sentence-final markers. We deliberately keep this small: ``.``,
# ``?``, ``!``. Ellipsis (``...``) ends with a ``.`` and so is treated
# as sentence-final too — that's a defensible choice for transcripts
# (a trailing-off thought is still the end of the unit), and avoiding
# false-negatives on quoted dialogue ("…", "...") matters more than
# the rare false-positive on abbreviations like "Mr.".
_SENTENCE_FINAL = ".?!"


def _is_sentence_final(text: str) -> bool:
    """Return True iff ``text`` (a single token) ends a sentence.

    Strips closing quotes / brackets first so ``hello."`` and
    ``(stop!)`` both count. Empty strings, whitespace-only strings,
    and pure-punctuation tokens that do not contain a sentence-final
    char return False.
    """
    if not isinstance(text, str):
        return False
    s = text.rstrip()
    while s and s[-1] in _CLOSING_TRIM:
        s = s[:-1]
    if not s:
        return False
    return s[-1] in _SENTENCE_FINAL


def sentence_ranges_in_segment(
    words: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """Return inclusive (start_word_idx, end_word_idx) ranges per sentence.

    A sentence ends at the first sentence-final word at or after its
    start. The trailing words of a segment that lack any sentence-
    final punctuation form one final sentence whose end is the last
    word of the segment.

    Empty ``words`` returns ``[]``. The returned list partitions the
    word indices ``[0, len(words))`` exactly: the union of the ranges
    is the whole segment, with no overlaps and no gaps.
    """
    if not words:
        return []
    ranges: list[tuple[int, int]] = []
    sent_start = 0
    for i, w in enumerate(words):
        text = w.get("text", "") if isinstance(w, Mapping) else ""
        if _is_sentence_final(text):
            ranges.append((sent_start, i))
            sent_start = i + 1
    # Trailing words without a sentence-final marker: one final
    # sentence covering them.
    if sent_start <= len(words) - 1:
        ranges.append((sent_start, len(words) - 1))
    return ranges


def _sentence_for_word(
    sentences: Sequence[tuple[int, int]],
    word_idx: int,
) -> tuple[int, int]:
    """Return the (start, end) sentence range that contains ``word_idx``.

    Linear scan; sentence counts per segment are tiny in practice
    (typically < 10), so any cleverer search is wasted effort.
    Raises :class:`ProjectValidationError` if ``word_idx`` falls in
    no range — that means the caller passed a word id beyond the
    segment, which is a real bug worth surfacing.
    """
    for s_start, s_end in sentences:
        if s_start <= word_idx <= s_end:
            return (s_start, s_end)
    raise ProjectValidationError(
        f"word index {word_idx} is not in any sentence range"
    )


# --------------------------------------------------------------------------- #
# Paragraph detection
# --------------------------------------------------------------------------- #


def paragraph_ranges(
    segments: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """Return inclusive (start_seg_idx, end_seg_idx) ranges per paragraph.

    Two consecutive segments share a paragraph iff they share a
    non-None ``speaker`` value. A segment with ``speaker`` missing or
    None is its own paragraph (we don't roll an unknown speaker into
    a known one — that would silently merge data).

    Empty ``segments`` returns ``[]``. The output partitions the
    segment indices exactly.
    """
    n = len(segments)
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    para_start = 0
    for i in range(1, n):
        prev_sp = segments[i - 1].get("speaker") if isinstance(segments[i - 1], Mapping) else None
        cur_sp = segments[i].get("speaker") if isinstance(segments[i], Mapping) else None
        if prev_sp is None or cur_sp is None or prev_sp != cur_sp:
            ranges.append((para_start, i - 1))
            para_start = i
    ranges.append((para_start, n - 1))
    return ranges


def _paragraph_for_segment(
    paragraphs: Sequence[tuple[int, int]],
    seg_idx: int,
) -> tuple[int, int]:
    """Return the paragraph (start, end) range that contains ``seg_idx``."""
    for p_start, p_end in paragraphs:
        if p_start <= seg_idx <= p_end:
            return (p_start, p_end)
    raise ProjectValidationError(
        f"segment index {seg_idx} is not in any paragraph range"
    )


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _segment_word_count(segments: Sequence[Mapping[str, Any]], seg_idx: int) -> int:
    """Number of words in ``segments[seg_idx]``; 0 if missing/empty."""
    if seg_idx < 0 or seg_idx >= len(segments):
        raise ProjectValidationError(
            f"segment index {seg_idx} out of range [0, {len(segments)})"
        )
    seg = segments[seg_idx]
    if not isinstance(seg, Mapping):
        raise ProjectValidationError(
            f"segment {seg_idx} must be a mapping; got {type(seg).__name__}"
        )
    words = seg.get("words")
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        return 0
    return len(words)


def _validate_word_ref(
    segments: Sequence[Mapping[str, Any]],
    word_id: str,
) -> tuple[int, int]:
    """Parse ``word_id`` and check it lies in ``segments``.

    Returns the parsed ``(seg_idx, word_idx)``. Raises
    :class:`ProjectValidationError` for malformed ids and for ids that
    point outside the transcript.
    """
    seg_idx, word_idx = parse_word_id(word_id)
    n_words = _segment_word_count(segments, seg_idx)
    if n_words == 0:
        raise ProjectValidationError(
            f"segment {seg_idx} has no words; cannot resolve {word_id!r}"
        )
    if word_idx >= n_words:
        raise ProjectValidationError(
            f"word index {word_idx} out of range "
            f"[0, {n_words}) in segment {seg_idx}"
        )
    return (seg_idx, word_idx)


def _segment_words(segments: Sequence[Mapping[str, Any]], seg_idx: int) -> Sequence[Mapping[str, Any]]:
    """Return the words list of ``segments[seg_idx]``; empty seq if absent."""
    seg = segments[seg_idx]
    words = seg.get("words") if isinstance(seg, Mapping) else None
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        return []
    return words


# --------------------------------------------------------------------------- #
# Snap helpers
# --------------------------------------------------------------------------- #


def snap_to_word(selection: Selection) -> Selection:
    """Return a copy of ``selection`` with sub-word offsets dropped.

    The result spans whole words from ``start_word_id`` to
    ``end_word_id`` inclusive. Idempotent — calling twice yields
    the same selection.

    Does not need a transcript: the transformation is purely on the
    selection's offset fields.
    """
    if not isinstance(selection, Selection):
        raise ProjectValidationError(
            f"selection must be a Selection; got {type(selection).__name__}"
        )
    # Validate the word-id shapes so a malformed selection doesn't
    # silently pass through.
    parse_word_id(selection.start_word_id)
    parse_word_id(selection.end_word_id)
    if selection.start_char_offset is None and selection.end_char_offset is None:
        return selection  # already snapped
    return replace(
        selection,
        start_char_offset=None,
        end_char_offset=None,
    )


def snap_to_sentence(
    selection: Selection,
    segments: Sequence[Mapping[str, Any]],
) -> Selection:
    """Snap to sentence boundaries within each endpoint's segment.

    Sentence boundaries are detected per-segment by trailing
    sentence-final punctuation (``.``, ``?``, ``!``) on a word's
    text, after stripping closing quotes / brackets. The first word
    of a segment always begins a sentence.

    The result:

    * Drops both sub-word offsets (sentence boundaries are word-level).
    * Extends the start to the first word of the sentence containing
      the original start word.
    * Extends the end to the last word of the sentence containing the
      original end word.
    * If the original start was *after* the original end (callers
      should not do this, but we don't want to mask it), raises
      :class:`ProjectValidationError`.

    Cross-segment selections snap each endpoint within its own
    segment; nothing in between is touched, because sentence
    boundaries don't cross segments in our data model. The natural
    consequence: a selection that starts mid-sentence in segment 3
    and ends mid-sentence in segment 5 expands to the start of its
    sentence in segment 3 and the end of its sentence in segment 5.
    """
    if not isinstance(selection, Selection):
        raise ProjectValidationError(
            f"selection must be a Selection; got {type(selection).__name__}"
        )
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ProjectValidationError(
            "segments must be a sequence of segment objects"
        )

    s_seg, s_word = _validate_word_ref(segments, selection.start_word_id)
    e_seg, e_word = _validate_word_ref(segments, selection.end_word_id)
    if (s_seg, s_word) > (e_seg, e_word):
        raise ProjectValidationError(
            "selection start is after selection end; cannot snap"
        )

    start_sent = sentence_ranges_in_segment(_segment_words(segments, s_seg))
    end_sent = sentence_ranges_in_segment(_segment_words(segments, e_seg))
    new_s_word, _ = _sentence_for_word(start_sent, s_word)
    _, new_e_word = _sentence_for_word(end_sent, e_word)

    return Selection(
        start_word_id=make_word_id(s_seg, new_s_word),
        end_word_id=make_word_id(e_seg, new_e_word),
        start_char_offset=None,
        end_char_offset=None,
    )


def snap_to_paragraph(
    selection: Selection,
    segments: Sequence[Mapping[str, Any]],
) -> Selection:
    """Snap to whole-paragraph boundaries.

    A *paragraph* is a maximal run of consecutive segments that share
    a non-None ``speaker`` value. Segments without a speaker (None /
    missing) are each their own paragraph.

    The result:

    * Drops both sub-word offsets.
    * Extends the start to the first word of the first segment of the
      paragraph containing the original start segment.
    * Extends the end to the last word of the last segment of the
      paragraph containing the original end segment.
    * Selections spanning multiple paragraphs are extended to cover
      from the start of the first paragraph touched to the end of the
      last paragraph touched (the union of all paragraphs touched).

    Empty paragraphs (segments with zero words) are not skipped — the
    selection's endpoint stays anchored to the same segment, but at
    word index 0 / last-word respectively. Callers should validate
    transcripts before calling.
    """
    if not isinstance(selection, Selection):
        raise ProjectValidationError(
            f"selection must be a Selection; got {type(selection).__name__}"
        )
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ProjectValidationError(
            "segments must be a sequence of segment objects"
        )

    s_seg, s_word = _validate_word_ref(segments, selection.start_word_id)
    e_seg, e_word = _validate_word_ref(segments, selection.end_word_id)
    if (s_seg, s_word) > (e_seg, e_word):
        raise ProjectValidationError(
            "selection start is after selection end; cannot snap"
        )

    paragraphs = paragraph_ranges(segments)
    p_start_seg, _ = _paragraph_for_segment(paragraphs, s_seg)
    _, p_end_seg = _paragraph_for_segment(paragraphs, e_seg)

    end_words = _segment_word_count(segments, p_end_seg)
    if end_words == 0:
        raise ProjectValidationError(
            f"paragraph end segment {p_end_seg} has no words; "
            "cannot snap"
        )

    return Selection(
        start_word_id=make_word_id(p_start_seg, 0),
        end_word_id=make_word_id(p_end_seg, end_words - 1),
        start_char_offset=None,
        end_char_offset=None,
    )
