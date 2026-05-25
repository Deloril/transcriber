"""Span operations on Applications (F4.2).

Per PLANNING.md F4.2:

  > Multiple non-contiguous applications per (code, source).

F4.1 gave us the :class:`scribe.applications.Application` data model
and per-application persistence. F4.2 layers the *operations* a
researcher actually performs once a (code, source) pair carries more
than one application:

* enumerate the applications in document order;
* tell whether two applications **overlap**, are **adjacent**, or are
  **fully disjoint**;
* group all of a project's applications by ``(code_id, source_id)``;
* spot **exact-duplicate anchors** (same code, same source, identical
  span and offsets — almost always a UX or import bug);
* compute **non-contiguous components** for a (code, source) pair:
  the maximal clusters of pairwise overlap-or-adjacent applications.
  The number of components is the F4.2 "how many places in this
  source did I apply this code?" count — and crucially, it can be
  > 1, which is the *normal* and supported case.

What this module is **not**:

* It never modifies persisted Applications. There's no auto-merge of
  adjacent quotes — researchers explicitly told us in the §4 design
  notes that the gap is part of the analytic point.
* It does not deal with *different* codes overlapping each other —
  the gutter renderer and overlap UX is F4.3.
* It does not snap selections to word/sentence/paragraph boundaries
  (F4.4) and does not re-anchor on transcript edit (F4.5).
* It does not depend on the transcript JSON itself; everything is a
  pure function over :class:`Application` instances. Cross-segment
  adjacency, which would need per-segment word counts, is parameter-
  ised (``segment_word_counts``) so callers can opt in when they have
  that information; without it, only **within-segment** adjacency is
  detected.

Stand-alone (no FastAPI, no engine imports), matching the conventions
of :mod:`scribe.code_lifecycle`, :mod:`scribe.icr`, :mod:`scribe.query`,
and :mod:`scribe.matrix`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

from .applications import (
    Application,
    parse_word_id,
)


# --------------------------------------------------------------------------- #
# Anchor representation & ordering
# --------------------------------------------------------------------------- #


# Sentinel used as the "right edge" of a word when ``end_char_offset`` is
# unset, so that a None-offset end always sorts *after* any explicit
# integer offset on the same word. Using ``math.inf`` keeps the anchor
# tuple JSON-friendly (we never serialise it) and makes overlap/adjacency
# checks trivially correct under standard tuple ordering.
_END_OF_WORD: float = math.inf


def anchor_key(application: Application) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int | float],
]:
    """Return a sortable key describing an application's text span.

    The key is ``((seg_start, word_start, start_offset_or_0),
    (seg_end, word_end, end_offset_or_inf))``. Comparing two keys
    compares applications in document order; equality means identical
    span on identical word boundaries.

    ``None`` for ``start_char_offset`` is normalised to ``0`` ("from the
    start of the word") and ``None`` for ``end_char_offset`` is
    normalised to ``+inf`` ("to the end of the word"), so a whole-word
    anchor always brackets any sub-word anchor on the same words.
    """
    sa_seg, sa_word = parse_word_id(application.anchor_start_word_id)
    ea_seg, ea_word = parse_word_id(application.anchor_end_word_id)
    so = application.start_char_offset if application.start_char_offset is not None else 0
    eo: int | float
    if application.end_char_offset is None:
        eo = _END_OF_WORD
    else:
        eo = application.end_char_offset
    return ((sa_seg, sa_word, so), (ea_seg, ea_word, eo))


def sort_by_anchor(applications: Iterable[Application]) -> list[Application]:
    """Return applications sorted by anchor position, ties broken by id.

    Stable: ties on anchor position are broken by ``application.id`` so
    repeated runs always produce the same order even when two coders
    have applied the same code at exactly the same span.
    """
    return sorted(applications, key=lambda a: (anchor_key(a), a.id))


# --------------------------------------------------------------------------- #
# Pairwise relations
# --------------------------------------------------------------------------- #


def _start_position(a: Application) -> tuple[int, int, int]:
    """The "leftmost" sortable position of an application's span."""
    seg, word = parse_word_id(a.anchor_start_word_id)
    so = a.start_char_offset if a.start_char_offset is not None else 0
    return (seg, word, so)


def _end_position(a: Application) -> tuple[int, int, int | float]:
    """The "rightmost" sortable position of an application's span.

    Whole-word ends use ``+inf`` so any sub-word offset on the same
    word sorts before them. This matches :func:`anchor_key`.
    """
    seg, word = parse_word_id(a.anchor_end_word_id)
    if a.end_char_offset is None:
        return (seg, word, _END_OF_WORD)
    return (seg, word, a.end_char_offset)


def applications_overlap(a: Application, b: Application) -> bool:
    """Return True iff ``a`` and ``b`` cover at least one common position.

    Word-id ranges are inclusive on both ends, so two applications
    sharing a single word boundary at whole-word resolution count as
    overlapping. Sub-word offsets refine the boundary: ``[s0w0:0,
    s0w0:5]`` and ``[s0w0:5, s0w0:10]`` *touch* but do not overlap
    (the end of the first is the strict left edge of the second).

    Different ``source_id`` always returns False — overlap is a
    same-source notion. Different ``code_id`` is allowed; this module
    is agnostic about which code each application carries.
    """
    if a.source_id != b.source_id:
        return False
    a_lo = _start_position(a)
    a_hi = _end_position(a)
    b_lo = _start_position(b)
    b_hi = _end_position(b)
    # Intervals [a_lo, a_hi] and [b_lo, b_hi] overlap iff
    # a_lo < b_hi and b_lo < a_hi (strict — touching at a single point
    # is not overlap; that's adjacency).
    return a_lo < b_hi and b_lo < a_hi


def applications_disjoint(a: Application, b: Application) -> bool:
    """Return True iff ``a`` and ``b`` are on the same source and don't overlap.

    Different sources are *not* "disjoint" in this F4.2 sense — they're
    not comparable. The function returns False so callers can use it as
    a positive same-source-and-separate predicate without an explicit
    pre-check.
    """
    if a.source_id != b.source_id:
        return False
    return not applications_overlap(a, b)


def applications_adjacent(
    a: Application,
    b: Application,
    *,
    segment_word_counts: Mapping[int, int] | None = None,
) -> bool:
    """Return True iff ``a`` and ``b`` touch at a clean word boundary.

    "Adjacent" means: same source, no overlap, and one application
    ends exactly where the other begins, at whole-word resolution.

    The two adjacency cases:

    1. **Within-segment**: ``end = (seg, w, end_of_word)`` and
       ``start = (seg, w + 1, 0)``. Detected unconditionally.
    2. **Across-segment**: ``end = (seg, last_word, end_of_word)`` and
       ``start = (seg + 1, 0, 0)``. Requires ``segment_word_counts`` —
       a mapping of ``segment_index → number_of_words_in_that_segment``
       — so we know what the last word of a segment is. Without it, we
       conservatively say False (rather than guess). Pass
       ``segment_word_counts`` when you can derive it from the
       transcript JSON.

    Sub-word offsets break adjacency: a span ending at ``s0w5:8`` is
    *not* adjacent to one starting at ``s0w5:8`` (that's overlap-by-
    touching; this module reports it as not-overlap-not-adjacent —
    both halves of the same word, but with offsets on the boundary,
    are an explicit choice of the coder to *split* a word).
    """
    if a.source_id != b.source_id:
        return False
    if applications_overlap(a, b):
        return False

    # Order so ``first`` is the earlier-starting application.
    first, second = (a, b) if anchor_key(a) <= anchor_key(b) else (b, a)

    # Adjacency requires whole-word boundaries on the two touching
    # ends (end_char_offset is None on the earlier; start_char_offset
    # is None on the later). Sub-word offsets at the touch point are
    # a deliberate split, not an adjacency.
    if first.end_char_offset is not None:
        return False
    if second.start_char_offset is not None:
        return False

    e_seg, e_word = parse_word_id(first.anchor_end_word_id)
    s_seg, s_word = parse_word_id(second.anchor_start_word_id)

    # Case 1: within-segment.
    if s_seg == e_seg and s_word == e_word + 1:
        return True

    # Case 2: across-segment, conditional on caller-supplied counts.
    if (
        segment_word_counts is not None
        and s_seg == e_seg + 1
        and s_word == 0
    ):
        last_word_count = segment_word_counts.get(e_seg)
        if isinstance(last_word_count, int) and last_word_count > 0:
            return e_word == last_word_count - 1

    return False


# --------------------------------------------------------------------------- #
# Group queries
# --------------------------------------------------------------------------- #


def applications_for_code_source(
    applications: Iterable[Application],
    code_id: str,
    source_id: str,
) -> list[Application]:
    """Return the applications matching ``(code_id, source_id)``, ordered.

    Document order; ties broken by application id (see
    :func:`sort_by_anchor`). Pass the full project's applications and
    let this filter — at F4.2 scale this is cheap and avoids leaking
    persistence concerns into the call site.
    """
    matched = (
        a for a in applications
        if a.code_id == code_id and a.source_id == source_id
    )
    return sort_by_anchor(matched)


def group_by_code_source(
    applications: Iterable[Application],
) -> dict[tuple[str, str], list[Application]]:
    """Bucket applications by ``(code_id, source_id)``; each bucket sorted.

    The returned dict's value lists are in document order. Code-source
    pairs with no applications simply don't appear in the dict.
    """
    buckets: dict[tuple[str, str], list[Application]] = defaultdict(list)
    for a in applications:
        buckets[(a.code_id, a.source_id)].append(a)
    return {k: sort_by_anchor(v) for k, v in buckets.items()}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def find_duplicate_anchors(
    applications: Iterable[Application],
) -> list[list[Application]]:
    """Return groups of 2+ applications that share an identical anchor.

    "Identical" means the same ``code_id``, ``source_id``,
    ``anchor_start_word_id``, ``anchor_end_word_id``,
    ``start_char_offset`` (None and 0 are NOT collapsed; offsets are
    compared as-is), and ``end_char_offset``. Different ``coder_id``s
    on otherwise-identical anchors *do* group: that's the multi-coder
    case (F2.5 / F4.2 territory) and worth surfacing as a duplicate so
    the reconciliation UI can highlight it.

    Single-application groups are not returned. Each group is sorted
    by application id for stable output.

    Why "duplicate" rather than "merge candidate"? F4.2's principle is
    to never auto-merge — exact duplicates are the only case where the
    intent is unambiguously "this should be one record, not two", and
    even then we surface, not delete. F9.1's event log will capture
    the resolution.
    """
    by_key: dict[tuple, list[Application]] = defaultdict(list)
    for a in applications:
        key = (
            a.code_id,
            a.source_id,
            a.anchor_start_word_id,
            a.anchor_end_word_id,
            a.start_char_offset,
            a.end_char_offset,
        )
        by_key[key].append(a)
    out: list[list[Application]] = []
    for group in by_key.values():
        if len(group) >= 2:
            out.append(sorted(group, key=lambda a: a.id))
    # Stable order across runs: sort the outer list by the first id in
    # each group.
    out.sort(key=lambda g: g[0].id)
    return out


def overlap_clusters(
    applications: Iterable[Application],
) -> list[list[Application]]:
    """Return transitively-overlapping clusters of applications.

    Two applications are connected if they overlap (same source).
    Clusters of size 1 are *not* returned — only the genuinely
    overlapping ones, since a list-of-singletons is just the original
    input. Inside each cluster, applications are sorted by anchor.

    "Transitively" means: if A overlaps B and B overlaps C but A and
    C don't overlap directly, all three still cluster together. This
    matches the F4.3 gutter renderer's intuition (a "stack" of overlap
    occupies the same column even if its endpoints don't).

    Different code ids overlap freely — a single span can carry many
    codes. F4.3 will lean on this to render the gutter; F4.2 just
    surfaces the structure.
    """
    apps = list(applications)
    n = len(apps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # O(n^2) is fine: a transcript with > a few thousand applications
    # is not on the F4.2 radar, and the constant factor on parse_word_id
    # is small. F8.x's embedding index is the right place to optimise
    # if we ever hit a real bottleneck.
    for i in range(n):
        for j in range(i + 1, n):
            if applications_overlap(apps[i], apps[j]):
                union(i, j)

    clusters: dict[int, list[Application]] = defaultdict(list)
    for i, a in enumerate(apps):
        clusters[find(i)].append(a)

    out: list[list[Application]] = []
    for group in clusters.values():
        if len(group) >= 2:
            out.append(sort_by_anchor(group))
    # Stable outer order: sort by the first member's anchor key.
    out.sort(key=lambda g: (anchor_key(g[0]), g[0].id))
    return out


def non_contiguous_components(
    applications: Iterable[Application],
    code_id: str,
    source_id: str,
    *,
    segment_word_counts: Mapping[int, int] | None = None,
) -> list[list[Application]]:
    """Return the maximal overlap-or-adjacent clusters for a ``(code, source)``.

    This is the F4.2 headline operation. Given all of a project's
    applications, restrict to those carrying ``code_id`` on
    ``source_id``, then bucket them so that two applications land in
    the same bucket iff they (transitively) overlap or are adjacent.

    The ``len(...)`` of the return value is the number of "places in
    this source where this code applies" — the non-contiguous span
    count. ``len(...) > 1`` is the *normal* F4.2 case: a researcher
    coded the same idea three times in three different places.

    Inside each component the applications are anchor-sorted; the
    components themselves are sorted by their earliest member's
    anchor.

    ``segment_word_counts`` is forwarded to :func:`applications_adjacent`
    so cross-segment adjacency is detected when the caller can
    compute it from the transcript. Without it, only within-segment
    adjacency is recognised.
    """
    matched = applications_for_code_source(applications, code_id, source_id)
    n = len(matched)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if applications_overlap(matched[i], matched[j]) or applications_adjacent(
                matched[i],
                matched[j],
                segment_word_counts=segment_word_counts,
            ):
                union(i, j)

    components: dict[int, list[Application]] = defaultdict(list)
    for i, a in enumerate(matched):
        components[find(i)].append(a)

    out = [sort_by_anchor(g) for g in components.values()]
    out.sort(key=lambda g: (anchor_key(g[0]), g[0].id))
    return out


def count_non_contiguous_components(
    applications: Iterable[Application],
    code_id: str,
    source_id: str,
    *,
    segment_word_counts: Mapping[int, int] | None = None,
) -> int:
    """How many distinct, non-contiguous places does this code appear?

    Convenience wrapper around :func:`non_contiguous_components` for
    callers that only need the count (e.g. matrix views, tooltips).
    Returns 0 when there are no applications for the (code, source)
    pair.
    """
    return len(
        non_contiguous_components(
            applications,
            code_id,
            source_id,
            segment_word_counts=segment_word_counts,
        )
    )
