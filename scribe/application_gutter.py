"""Gutter / margin layout for overlapping applications (F4.3).

Per PLANNING.md F4.3:

  > Unlimited overlapping codes on a span; gutter/margin renderer.

Reading the design notes (`docs/research/coding-engine-research.md`
§4):

  > A 30-word participant utterance routinely picks up 4–6 codes
  > (topic, emotion, action, in-vivo phrase, reflexive note). … The
  > gutter approach scales to many overlapping codes; in-text
  > highlights stop being readable past ~3 layers.

This module is the **layout brain** behind the gutter. It does not
draw anything: it takes a list of :class:`Application` instances and
returns a deterministic assignment of each application to a *lane*
(a vertical track in the gutter) such that no two applications in the
same lane overlap. The renderer (HTML / CSS in the editor; or any
other surface) consumes the assignment and paints coloured bars at
the anchor offsets.

The algorithm is the textbook **interval-graph greedy colouring**:

1. Sort applications by anchor start (ties by anchor end, then by id
   for stability).
2. Scan left to right. For each application, place it in the
   lowest-indexed lane whose previously-placed end position is
   strictly to the left of this application's start position. If no
   lane is free, open a new one.
3. The total number of lanes used equals the maximum *clique size*
   of the interval graph — i.e. the largest count of mutually
   overlapping applications anywhere in the source. This is optimal:
   no smaller lane count can lay these intervals out without an
   overlap collision.

What this module is **not**:

* It does not render HTML or SVG. ``scribe/static/js/helpers.mjs``
  exposes a JS mirror of :func:`assign_lanes` for the in-browser
  renderer; both must agree on lane numbers for any given input.
* It does not depend on persistence. Like
  :mod:`scribe.application_spans`, it operates on
  :class:`Application` instances directly.
* It does not group across sources. Lane assignments are per-source;
  passing a mixed-source list partitions on ``source_id`` first.
* It does not deduplicate overlapping anchors that carry the *same*
  code — F4.2 (:func:`scribe.application_spans.find_duplicate_anchors`)
  surfaces those; this module places them in adjacent lanes so the
  duplicate is visible rather than hidden.

This module is stand-alone — no FastAPI, no engine — matching the
conventions of :mod:`scribe.application_spans`,
:mod:`scribe.code_lifecycle`, :mod:`scribe.icr`, :mod:`scribe.query`,
and :mod:`scribe.matrix`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .applications import Application
from .application_spans import (
    _end_position,  # noqa: PLC2701  — module-internal helper, deliberate reuse
    _start_position,  # noqa: PLC2701
    anchor_key,
    sort_by_anchor,
)


# --------------------------------------------------------------------------- #
# Lane assignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LanePlacement:
    """Where in the gutter one :class:`Application` lives.

    * ``application_id`` — the F4.1 ``Application.id``.
    * ``lane`` — zero-indexed lane number; ``0`` is the lane closest
      to the transcript text.
    * ``stack_depth`` — how many other applications cover at least
      one position in common with this one (i.e. the size of the
      overlap clique this application belongs to, minus 1 for itself).
      ``stack_depth == 0`` means a solo, non-overlapped span.

    The renderer draws the coloured bar at column ``lane`` and can
    use ``stack_depth`` to tag the segment as "deeply stacked" for a
    UI hint (e.g. dim the background, show a count badge).
    """

    application_id: str
    lane: int
    stack_depth: int


@dataclass(frozen=True)
class GutterLayout:
    """The result of laying out one source's applications.

    * ``source_id`` — the source these placements belong to.
    * ``placements`` — one :class:`LanePlacement` per input
      application, in the same document order returned by
      :func:`scribe.application_spans.sort_by_anchor`.
    * ``lane_count`` — total number of lanes opened. Equals the
      maximum overlap clique size at any point in the source. Zero
      when there are no applications.
    * ``max_stack_depth`` — the largest ``stack_depth`` across all
      placements. Useful as a "this transcript is heavily coded"
      hint without iterating placements.

    Two layouts on equal input produce equal output (by-value
    equality). The CLI / server can therefore cache the layout
    keyed on a hash of the input application ids without worrying
    about ordering instability.
    """

    source_id: str
    placements: tuple[LanePlacement, ...] = field(default_factory=tuple)
    lane_count: int = 0
    max_stack_depth: int = 0

    def lane_for(self, application_id: str) -> int | None:
        """Return the lane assigned to ``application_id`` or ``None``.

        Linear scan — fine for the F4.3 scale (a single source with
        a few thousand applications at most). If we ever exceed that,
        precompute a dict in ``__post_init__``.
        """
        for p in self.placements:
            if p.application_id == application_id:
                return p.lane
        return None


def assign_lanes(applications: Iterable[Application]) -> GutterLayout:
    """Place a single source's applications into non-overlapping lanes.

    All input applications must share the same ``source_id`` (raises
    :class:`ValueError` otherwise — passing mixed sources is almost
    always a bug at the call site; use :func:`assign_lanes_per_source`
    to bucket by source first).

    The algorithm is a left-to-right sweep:

    * Sort by :func:`anchor_key`, ties broken by application id.
    * Track each lane's "last end position" — the end-of-span
      position of the most recently placed application in that lane.
    * For each application, pick the lowest-index lane whose last
      end position is **less than or equal to** this application's
      start position. (``<=`` rather than ``<``, because two
      applications that meet at a single point — F4.2's "touching"
      case — do not *overlap*, so they can share a lane.) If no
      lane qualifies, open a new one.

    Stack depth is computed in a second pass: for each application,
    count how many lanes are simultaneously "active" (cover this
    application's full span) when it is placed.

    Empty input returns an empty layout with ``source_id`` set to
    the empty string — callers that need a deterministic source id
    on empty layouts can use :func:`assign_lanes_per_source` and
    pre-seed an empty list under that id.
    """
    apps = list(applications)
    if not apps:
        return GutterLayout(source_id="", placements=(), lane_count=0, max_stack_depth=0)

    source_ids = {a.source_id for a in apps}
    if len(source_ids) > 1:
        raise ValueError(
            "assign_lanes requires single-source input; "
            f"got {len(source_ids)} distinct source_ids "
            "(use assign_lanes_per_source to bucket first)"
        )
    (source_id,) = source_ids

    ordered = sort_by_anchor(apps)

    # Lane index → end position of the last application in that lane.
    # We pick the *lowest-indexed* free lane on each placement, so
    # the bar closest to the text is always reused first.
    lane_ends: list[tuple[int, int, int | float]] = []
    placements_by_id: dict[str, tuple[int, int, int | float]] = {}
    lane_of: dict[str, int] = {}

    for app in ordered:
        start = _start_position(app)
        end = _end_position(app)
        chosen: int | None = None
        for i, last_end in enumerate(lane_ends):
            if last_end <= start:
                chosen = i
                break
        if chosen is None:
            chosen = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[chosen] = end
        lane_of[app.id] = chosen
        placements_by_id[app.id] = (chosen, 0, 0)  # populated below

    # Compute stack depth: how many other applications overlap each
    # one. We iterate every pair once; F4.3 scale is fine for O(n^2).
    # An "active" overlap is the F4.2 strict overlap (touching at a
    # single point doesn't count, matching :func:`applications_overlap`).
    stack_depth: dict[str, int] = {a.id: 0 for a in ordered}
    for i, a in enumerate(ordered):
        a_lo = _start_position(a)
        a_hi = _end_position(a)
        for j in range(len(ordered)):
            if i == j:
                continue
            b = ordered[j]
            b_lo = _start_position(b)
            b_hi = _end_position(b)
            if a_lo < b_hi and b_lo < a_hi:
                stack_depth[a.id] += 1

    placements = tuple(
        LanePlacement(
            application_id=a.id,
            lane=lane_of[a.id],
            stack_depth=stack_depth[a.id],
        )
        for a in ordered
    )

    lane_count = len(lane_ends)
    max_depth = max(stack_depth.values()) if stack_depth else 0

    return GutterLayout(
        source_id=source_id,
        placements=placements,
        lane_count=lane_count,
        max_stack_depth=max_depth,
    )


def assign_lanes_per_source(
    applications: Iterable[Application],
) -> dict[str, GutterLayout]:
    """Bucket applications by source and lay each source out independently.

    Returns a dict mapping ``source_id → GutterLayout``. Sources with
    no applications do not appear (callers know which sources they
    care about). Each :class:`GutterLayout` is computed independently;
    lane numbers are not comparable across sources.
    """
    buckets: dict[str, list[Application]] = defaultdict(list)
    for a in applications:
        buckets[a.source_id].append(a)
    return {sid: assign_lanes(apps) for sid, apps in buckets.items()}


# --------------------------------------------------------------------------- #
# Per-position queries
# --------------------------------------------------------------------------- #


def applications_at_word(
    applications: Iterable[Application],
    source_id: str,
    word_id: str,
) -> list[Application]:
    """Return applications covering ``word_id`` in ``source_id``, sorted.

    "Covers" means the closed [start, end] word-id interval includes
    ``word_id`` at whole-word resolution. Sub-word offsets are
    deliberately ignored here: a renderer asking "which codes are on
    word X?" wants the answer at word granularity. Use
    :func:`scribe.application_spans.applications_overlap` if you need
    sub-word precision.

    Sorted in document order with ties broken by application id, so
    the returned list lines up with the gutter lane order produced
    by :func:`assign_lanes`.

    Returns an empty list when no applications match. ``source_id``
    is required: cross-source lookups by word id make no sense (word
    ids are scoped per source).
    """
    from .applications import parse_word_id  # local import: cheap, avoids cycle risk

    target = parse_word_id(word_id)
    matched: list[Application] = []
    for a in applications:
        if a.source_id != source_id:
            continue
        start = parse_word_id(a.anchor_start_word_id)
        end = parse_word_id(a.anchor_end_word_id)
        if start <= target <= end:
            matched.append(a)
    return sort_by_anchor(matched)


def lane_envelope(layout: GutterLayout) -> list[int]:
    """Return how many applications occupy each lane in ``layout``.

    The result has length ``layout.lane_count``; index ``i`` is the
    number of applications placed in lane ``i``. A "tall" lane is
    rare in normal coding and usually indicates a workhorse code
    applied many times in the same source.

    Returns an empty list for an empty layout.
    """
    counts = [0] * layout.lane_count
    for p in layout.placements:
        counts[p.lane] += 1
    return counts


def serialise_layout(layout: GutterLayout) -> dict:
    """Produce a JSON-friendly representation of a :class:`GutterLayout`.

    Used by the HTTP layer and by tests that want to round-trip a
    layout through JSON. Keeps the shape small and stable: future
    additions append new keys, existing keys never change meaning.
    """
    return {
        "source_id": layout.source_id,
        "lane_count": layout.lane_count,
        "max_stack_depth": layout.max_stack_depth,
        "placements": [
            {
                "application_id": p.application_id,
                "lane": p.lane,
                "stack_depth": p.stack_depth,
            }
            for p in layout.placements
        ],
    }


# Re-export :func:`anchor_key` so callers importing from this module
# can stay inside the F4.3 namespace; the gutter renderer needs a way
# to ask "which application starts first?" and not having to know
# whether to import from spans vs gutter is friction we don't need.
__all__ = [
    "GutterLayout",
    "LanePlacement",
    "anchor_key",
    "applications_at_word",
    "assign_lanes",
    "assign_lanes_per_source",
    "lane_envelope",
    "serialise_layout",
]
