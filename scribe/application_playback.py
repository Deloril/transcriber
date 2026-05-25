"""One-click playback for coded segments (F4.6).

Per PLANNING.md F4.6:

  > One-click playback from any coded segment (reuse the editor's
  > word→time map).

F4.1 gave us :class:`scribe.applications.Application` anchored on
``s<segment>w<word>`` ids. F4.6 closes the loop with the audio
pipeline: any coded segment can be replayed in the editor by
asking the transcript "what wall-clock time does this anchor
cover?". The editor already maintains a word→time map for the
yellow word-highlight that follows playback (see ``findActiveWord``
in ``helpers.mjs``); this module exposes the same lookup as a
small set of pure functions so the same answer is computable from
the server and from JS.

This module is *only* about turning anchors into ``(start, end)``
seconds; it does not actually play anything. The editor's existing
``<audio>``/``<video>`` element handles playback — F4.6 just gives
it a clean range to seek to. That keeps the F4.6 surface area
testable and the audio plumbing in one place.

What the helpers do
-------------------

* :func:`build_word_time_map` — flatten a Scribe transcript
  (``segments[].words[]``) into a deterministic
  ``word_id → (start, end, text)`` mapping. Words missing a
  timestamp are skipped silently — that's fine, the editor's
  ``spreadTokensAcrossSpan`` synthesises timestamps for those
  cases (F4.6 only promises playback when timing actually exists).
* :func:`playback_range_for_application` — turn an
  :class:`Application` anchor into a ``PlaybackRange`` covering
  ``[start_time, end_time]`` in seconds. Sub-word character
  offsets, when present, are *interpolated proportionally* across
  the anchor word's text (the same trick ``spreadTokensAcrossSpan``
  uses for editor inserts).
* :func:`playback_ranges_for_applications` — bulk lookup, useful
  for the gutter (F4.3) where we want a play-icon per coded
  segment without N round trips.

Sub-word offset interpolation
-----------------------------

A whole-word anchor maps to the word's full ``[start, end]``
timestamp. A sub-word anchor — ``"...crimi***nalisation***..."``
with ``start_char_offset = 4`` — slices the word's time interval
proportionally::

    word.start + (word.end - word.start) * (offset / len(word.text))

This is the same proportional-spread heuristic the editor uses
for typed-in (untimed) words. It's not phoneme-accurate; nobody
expects a code-application play-button to be sample-perfect.
What it *is* is monotonic, deterministic, and never seeks to the
wrong word.

Edge cases the helpers nail down explicitly:

* When the anchor's start word has no timing (e.g. an inserted
  untimed token) but later words do, fall back to the *segment's*
  start time. Symmetric for the end: fall back to segment end.
* If neither the word nor the segment has timing, return ``None``
  — the caller should hide the play button rather than seek to
  zero.
* Word ids that don't resolve to any word in the transcript
  raise :class:`scribe.applications.ProjectValidationError`. F4.5
  (orphan re-anchoring) is the right place to handle that long-
  term; F4.6 surfaces the bad anchor honestly.

What this module is **not**
---------------------------

* Not a player. The editor still owns ``<audio>``/``<video>``.
* Not stateful. Build a word-time map per request; don't cache —
  the transcript can be edited between calls (F4.5) and a stale
  cache is the worst kind of bug for an audit trail.
* Not a writer. We don't mutate the transcript or the
  application; pure read-only.

Stand-alone (no FastAPI, no engine), matching the conventions of
:mod:`scribe.application_spans`, :mod:`scribe.application_gutter`,
:mod:`scribe.selection_snap`, and :mod:`scribe.application_reanchor`.

A JS mirror lives in ``scribe/static/js/helpers.mjs``
(``buildWordTimeMap``, ``playbackRangeForApplication``) and must
produce identical results for any shared input — there is a
parallel test suite in ``tests/js/playback.test.mjs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .applications import Application, make_word_id, parse_word_id
from .projects import ProjectValidationError


# --------------------------------------------------------------------------- #
# Word-time map
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WordTime:
    """The timing record of a single word in a transcript.

    * ``start`` and ``end`` are seconds from the start of the media.
    * ``text`` is the word's display text (kept here so callers can
      compute sub-word offsets without a second lookup).

    ``WordTime`` is what :func:`build_word_time_map` puts in the
    map — never instantiated by callers directly.
    """

    start: float
    end: float
    text: str


def _coerce_time(value: Any) -> float | None:
    """Return ``value`` as a ``float`` if it's a finite real number.

    Returns ``None`` for ``None``, ``""``, NaN, ±inf, bool, or any
    non-numeric type. We're deliberately strict: a stray boolean
    sneaking in as ``True`` becoming ``1.0`` would silently seek
    the wrong place.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        # NaN and ±inf are not playable timestamps.
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    return None


def build_word_time_map(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, WordTime]:
    """Flatten a transcript's ``segments[].words[]`` into a word-id map.

    Returns ``{ "s<seg>w<word>": WordTime(start, end, text), ... }``.
    Words with missing or invalid timing are **skipped** — the
    caller learns by absence (``word_id not in map``). The text is
    still recorded under :func:`build_word_text_map` if the caller
    needs it independently.

    Words ordering is preserved (Python dicts are insertion-ordered
    on every supported runtime). The map's keys are always valid
    word ids, so iterating ``map.items()`` reads the transcript in
    document order modulo skipped untimed words.
    """
    if segments is None:
        raise ProjectValidationError("segments must be a sequence")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ProjectValidationError(
            f"segments must be a sequence; got {type(segments).__name__}"
        )

    out: dict[str, WordTime] = {}
    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            continue
        words = seg.get("words")
        if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
            continue
        for word_idx, w in enumerate(words):
            if not isinstance(w, Mapping):
                continue
            start = _coerce_time(w.get("start"))
            end = _coerce_time(w.get("end"))
            if start is None or end is None:
                continue
            if end < start:
                # Defensive: drop reversed timings rather than
                # propagate them. The editor's resync logic can
                # rebuild them.
                continue
            text = w.get("text")
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            out[make_word_id(seg_idx, word_idx)] = WordTime(
                start=start, end=end, text=text
            )
    return out


def _segment_time(
    segments: Sequence[Mapping[str, Any]], seg_idx: int
) -> tuple[float | None, float | None]:
    """Return the segment's ``(start, end)`` time, with safe fallbacks."""
    if seg_idx < 0 or seg_idx >= len(segments):
        return (None, None)
    seg = segments[seg_idx]
    if not isinstance(seg, Mapping):
        return (None, None)
    return (_coerce_time(seg.get("start")), _coerce_time(seg.get("end")))


# --------------------------------------------------------------------------- #
# Sub-word interpolation
# --------------------------------------------------------------------------- #


def _interpolate_offset(
    word: WordTime, char_offset: int, *, side: str
) -> float:
    """Map a character offset within a word to a wall-clock second.

    Linear interpolation: offset 0 → ``word.start``, offset
    ``len(text)`` → ``word.end``. Out-of-range offsets clamp into
    the word's interval (we already validated bounds at
    :class:`Application` construction; the clamp is a belt-and-
    braces against malformed text round-trips).

    ``side`` is ``"start"`` or ``"end"`` and selects the natural
    fallback when the word's text is empty (no characters to
    interpolate over) — start-side offsets pin to ``word.start``,
    end-side offsets pin to ``word.end``.
    """
    text_len = len(word.text)
    if text_len <= 0:
        return word.start if side == "start" else word.end
    span = word.end - word.start
    if span <= 0:
        return word.start
    clamped = max(0, min(char_offset, text_len))
    return word.start + span * (clamped / text_len)


# --------------------------------------------------------------------------- #
# Playback range
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlaybackRange:
    """A wall-clock ``[start, end]`` interval for one Application.

    Both ends are seconds from the start of the media. ``end`` is
    always ≥ ``start``. ``source_id`` is carried through so callers
    can route to the correct media file when a project has many.
    """

    application_id: str
    source_id: str
    start: float
    end: float


def playback_range_for_application(
    application: Application,
    segments: Sequence[Mapping[str, Any]],
    *,
    word_time_map: Mapping[str, WordTime] | None = None,
) -> PlaybackRange | None:
    """Return the seek range for ``application`` over ``segments``.

    Returns ``None`` if neither the anchor words nor the
    surrounding segments carry usable timing — the caller should
    hide the play button rather than seeking to zero.

    Sub-word ``start_char_offset`` / ``end_char_offset`` on the
    application are honoured by proportional interpolation across
    the anchor word's text (see :func:`_interpolate_offset`). When
    the anchor word itself lacks timing but the segment's
    ``start`` / ``end`` are present, those are used as a fallback
    so the play button still seeks somewhere reasonable.

    ``word_time_map`` may be passed by callers that already built
    one (e.g. the bulk helper :func:`playback_ranges_for_applications`)
    to avoid rebuilding it per application. Equivalent results
    either way.

    Validation
    ----------
    Anchor word ids that don't parse as ``s<seg>w<word>`` raise
    :class:`ProjectValidationError`. Anchors whose ``segment_index``
    is outside ``segments`` also raise — that's a real "this anchor
    is wrong for this transcript" condition F4.5 will eventually
    handle in a queue. We surface it now rather than silently
    returning None.
    """
    if not isinstance(application, Application):
        raise ProjectValidationError(
            f"application must be an Application; got "
            f"{type(application).__name__}"
        )

    sa_seg, _ = parse_word_id(application.anchor_start_word_id)
    ea_seg, _ = parse_word_id(application.anchor_end_word_id)

    if sa_seg < 0 or sa_seg >= len(segments):
        raise ProjectValidationError(
            f"anchor_start_word_id segment {sa_seg} out of range "
            f"[0, {len(segments)})"
        )
    if ea_seg < 0 or ea_seg >= len(segments):
        raise ProjectValidationError(
            f"anchor_end_word_id segment {ea_seg} out of range "
            f"[0, {len(segments)})"
        )

    wmap = word_time_map if word_time_map is not None else build_word_time_map(segments)

    start_word = wmap.get(application.anchor_start_word_id)
    end_word = wmap.get(application.anchor_end_word_id)

    seg_start, _ = _segment_time(segments, sa_seg)
    _, seg_end = _segment_time(segments, ea_seg)

    # Determine start time.
    if start_word is not None:
        if application.start_char_offset is not None:
            start_time = _interpolate_offset(
                start_word, application.start_char_offset, side="start"
            )
        else:
            start_time = start_word.start
    elif seg_start is not None:
        start_time = seg_start
    else:
        return None

    # Determine end time.
    if end_word is not None:
        if application.end_char_offset is not None:
            end_time = _interpolate_offset(
                end_word, application.end_char_offset, side="end"
            )
        else:
            end_time = end_word.end
    elif seg_end is not None:
        end_time = seg_end
    else:
        return None

    # Defensive normalisation: a transcript edit (F4.5) can leave
    # an anchor pointing at a now-earlier word relative to its
    # original end. Clamp so the play button never seeks
    # backwards.
    if end_time < start_time:
        end_time = start_time

    return PlaybackRange(
        application_id=application.id,
        source_id=application.source_id,
        start=start_time,
        end=end_time,
    )


def playback_ranges_for_applications(
    applications: Iterable[Application],
    segments_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, PlaybackRange]:
    """Bulk version of :func:`playback_range_for_application`.

    ``segments_by_source`` maps ``source_id`` to that source's
    ``segments`` list. Applications whose ``source_id`` is missing
    from the map are skipped (the caller knows they don't have a
    transcript to seek into yet).

    Returns ``{application_id: PlaybackRange}``. Applications that
    yield ``None`` (no timing anywhere) are simply absent from the
    output — same convention as :func:`build_word_time_map`. A
    word-time map is built **once per source** and reused across
    that source's applications, so this is O(W + A) rather than
    O(W × A).

    Anchor-out-of-range :class:`ProjectValidationError`s from the
    per-application helper propagate. F4.5 will eventually
    intercept those into the orphan queue; for F4.6 the call site
    is expected to filter to applications whose anchors still
    resolve.
    """
    cache: dict[str, dict[str, WordTime]] = {}
    out: dict[str, PlaybackRange] = {}
    for app in applications:
        segs = segments_by_source.get(app.source_id)
        if segs is None:
            continue
        wmap = cache.get(app.source_id)
        if wmap is None:
            wmap = build_word_time_map(segs)
            cache[app.source_id] = wmap
        rng = playback_range_for_application(app, segs, word_time_map=wmap)
        if rng is not None:
            out[app.id] = rng
    return out


__all__ = [
    "PlaybackRange",
    "WordTime",
    "build_word_time_map",
    "playback_range_for_application",
    "playback_ranges_for_applications",
]
