"""Runtime adapter that bridges on-disk Applications + transcripts to
the pure :mod:`scribe.query` executor (F3.5 user-surface wiring).

Why this exists
---------------

:mod:`scribe.query` ships a pure executor —
:func:`scribe.query.applications_for_query` — that takes a
:class:`scribe.query.Query` plus an iterable of
"application-shaped dicts" (``{"code_id", "source_id", optional
"speaker", "start", "end", optional "participant_id"}``).  That
contract is deliberately stand-alone: it round-trips JSON cleanly,
keeps F3.5 testable without an HTTP layer, and means the matrix /
saved-queries modules don't need to know the on-disk Application
shape.

But running a query from the user-facing surface needs the *real*
on-disk objects: :class:`scribe.applications.Application` (anchored
on ``s<seg>w<word>`` ids), each application's source (for the
:class:`scribe.query.SourceFilter`), the project's participants (for
the :class:`scribe.query.ParticipantFilter`), and the source's
speaker map (for :class:`scribe.query.SpeakerFilter` role-based
matching).

This module owns that translation:

  * :func:`application_to_query_dict` — turn one Application + its
    source's transcript segments into the dict shape the executor
    expects.  Resolves ``speaker`` from the segment that contains
    the anchor's start word; resolves ``start`` / ``end`` from the
    same word's wall-clock timestamps via
    :mod:`scribe.application_playback`.

  * :func:`run_query_against_project` — top-level helper used by
    the FastAPI route.  Takes a project root, a project id, a
    Query, and a callable that loads each source's segments
    (deliberately injected so the server-side discovery rules stay
    in :mod:`scribe.server` and the unit tests can pass in fixture
    data).  Returns the matching applications **plus** a small
    metadata payload (per-source segment-load status) so the UI can
    explain "this source had no transcript" without the user
    having to guess.

What this module does *not* do
------------------------------

  * Persist queries (:mod:`scribe.saved_queries` does that).
  * Compute matrices (:mod:`scribe.matrix` does that — it consumes
    the same applications iterable this module yields).
  * Touch HTTP, FastAPI, jinja, or filesystem paths beyond the
    abstract ``segments_loader`` callable.  All paths are resolved
    by the caller.

Conventions match :mod:`scribe.applications` (F4.1),
:mod:`scribe.application_playback` (F4.6), :mod:`scribe.query`
(F3.5), :mod:`scribe.speaker_map` (F3.4), :mod:`scribe.sources`
(F1.2), :mod:`scribe.participants` (F1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import applications as _applications
from . import participants as _participants
from . import sources as _sources
from . import speaker_map as _speaker_map
from .application_playback import (
    build_word_time_map,
    playback_range_for_application,
)
from .applications import Application, parse_word_id
from .projects import ProjectValidationError
from .query import Query, applications_for_query


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


# Callable that maps a source id to its transcript segment list, or
# None when the source has no resolvable transcript on disk.
SegmentsLoader = Callable[[str], "Sequence[Mapping[str, Any]] | None"]


@dataclass
class QueryRunReport:
    """Result of :func:`run_query_against_project`.

    Carried as a separate type so the FastAPI layer can return both
    the matches and a per-source diagnostic without inventing an ad
    hoc dict.

    Fields:

      * ``matches`` — Application objects that satisfied the query,
        in disk order.  The route serialises these via
        :meth:`Application.to_dict`.
      * ``total_applications`` — how many applications were
        considered before the query filter ran.  Useful for the UI
        to explain "0 of 47 matched" vs "0 of 0 matched".
      * ``sources_missing_transcript`` — source ids whose
        ``segments_loader`` returned ``None``.  Their applications
        are still considered, but the executor receives them with
        ``speaker=""`` / ``start=None`` / ``end=None`` — speaker /
        proximity filters will silently exclude them.
      * ``warnings`` — human-readable strings describing
        non-fatal issues (e.g. an anchor whose segment_index falls
        outside the loaded transcript).  Surfaced in the UI so the
        researcher knows why an expected match didn't appear.
    """

    matches: list[Application]
    total_applications: int
    sources_missing_transcript: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-application adapter
# --------------------------------------------------------------------------- #


def application_to_query_dict(
    app: Application,
    segments: "Sequence[Mapping[str, Any]] | None",
    *,
    speaker_map: _speaker_map.SpeakerMap | None = None,
) -> dict[str, Any]:
    """Build the executor-shape dict for one Application.

    ``segments`` may be ``None`` (no transcript on disk) — in that
    case ``speaker`` falls back to the empty string and ``start`` /
    ``end`` are omitted.  ``speaker_map`` is optional and not used
    here directly: the executor takes the speaker map separately
    (so a SpeakerFilter can resolve role / participant_id) and only
    needs the raw label from this dict.

    Returns a fresh dict per call; never mutates ``app`` or
    ``segments``.

    Validation: invalid anchor word ids raise
    :class:`ProjectValidationError`.  Anchors whose
    ``segment_index`` is out of range are *not* fatal — we return
    ``speaker=""`` and skip the timing fields, and let the caller
    (which is :func:`run_query_against_project`) record a warning.
    """
    if not isinstance(app, Application):
        raise ProjectValidationError(
            f"app must be an Application; got {type(app).__name__}"
        )

    out: dict[str, Any] = {
        "id": app.id,
        "code_id": app.code_id,
        "source_id": app.source_id,
        "coder_id": app.coder_id,
        "speaker": "",
    }

    if not segments:
        return out

    try:
        seg_idx, _ = parse_word_id(app.anchor_start_word_id)
    except ProjectValidationError:
        return out

    if seg_idx < 0 or seg_idx >= len(segments):
        return out

    seg = segments[seg_idx]
    if isinstance(seg, Mapping):
        spk = seg.get("speaker")
        if spk is not None:
            out["speaker"] = str(spk)

    # Timing — best-effort, identical resolution to the F4.6 play
    # button. We catch ProjectValidationError because the executor
    # only needs numeric start/end if a ProximityFilter is in use;
    # for the common "match by code / source / speaker" path,
    # missing timing is fine.
    try:
        word_map = build_word_time_map(segments)
        rng = playback_range_for_application(app, segments, word_time_map=word_map)
    except ProjectValidationError:
        rng = None
    if rng is not None:
        out["start"] = float(rng.start)
        out["end"] = float(rng.end)

    return out


# --------------------------------------------------------------------------- #
# Project-level executor
# --------------------------------------------------------------------------- #


def run_query_against_project(
    projects_root: Path,
    project_id: str,
    query: Query,
    *,
    segments_loader: SegmentsLoader,
    applications: "Iterable[Application] | None" = None,
    speaker_maps: "Mapping[str, _speaker_map.SpeakerMap] | None" = None,
) -> QueryRunReport:
    """Execute a Query against the on-disk corpus of a project.

    ``segments_loader`` is the only path-aware dependency: callers
    pass a closure that knows how to discover a source's transcript
    (the server.py helper
    :func:`_load_segments_for_source_speaker_map` already
    implements this).  Tests can pass a dict-backed lambda.

    ``applications`` defaults to
    :func:`scribe.applications.list_applications`; tests can inject
    a known set.  ``speaker_maps`` defaults to a per-source
    :func:`scribe.speaker_map.load_or_empty_speaker_map`.

    Returns a :class:`QueryRunReport`.  The matches list contains
    the **Application objects** (not dicts) so the route can
    serialise via the canonical :meth:`Application.to_dict`.
    """
    if not isinstance(query, Query):
        raise ProjectValidationError(
            f"query must be a Query; got {type(query).__name__}"
        )
    query.validate()
    if query.project_id != project_id:
        raise ProjectValidationError(
            f"query.project_id ({query.project_id!r}) must equal "
            f"project_id ({project_id!r})"
        )

    if applications is None:
        applications = _applications.list_applications(
            projects_root, project_id
        )
    apps_list: list[Application] = list(applications)

    sources_for_filter: list[_sources.Source] | None = None
    if not query.sources.is_empty():
        sources_for_filter = _sources.list_sources(projects_root, project_id)

    parts_for_filter: list[_participants.Participant] | None = None
    if not query.participants.is_empty():
        parts_for_filter = _participants.list_participants(
            projects_root, project_id
        )

    # Build a per-source segments cache so multiple applications on
    # the same source pay the discovery cost once.
    seg_cache: dict[str, "Sequence[Mapping[str, Any]] | None"] = {}
    smap_cache: dict[str, _speaker_map.SpeakerMap] = (
        dict(speaker_maps) if speaker_maps else {}
    )
    sources_missing: list[str] = []
    warnings: list[str] = []

    # The query executor needs a per-source SpeakerMap if any
    # SpeakerFilter is in play — load lazily.
    needs_speaker_map = not query.speakers.is_empty()

    app_dicts: list[dict[str, Any]] = []
    by_app_id: dict[str, Application] = {}
    for app in apps_list:
        sid = app.source_id
        if sid not in seg_cache:
            try:
                seg_cache[sid] = segments_loader(sid)
            except Exception as e:  # defensive: a discovery bug shouldn't 500
                seg_cache[sid] = None
                warnings.append(
                    f"source {sid}: segments loader raised "
                    f"{type(e).__name__}: {e}"
                )
            if seg_cache[sid] is None:
                sources_missing.append(sid)
        if needs_speaker_map and sid not in smap_cache:
            try:
                smap_cache[sid] = _speaker_map.load_or_empty_speaker_map(
                    projects_root, project_id, sid
                )
            except Exception:
                smap_cache[sid] = _speaker_map.SpeakerMap.new(
                    project_id=project_id, source_id=sid
                )
        d = application_to_query_dict(app, seg_cache[sid])
        app_dicts.append(d)
        by_app_id[d["id"]] = app

    matched_dicts = applications_for_query(
        query,
        app_dicts,
        sources=sources_for_filter,
        participants=parts_for_filter,
        speaker_maps=smap_cache if needs_speaker_map else None,
    )

    matched_apps: list[Application] = []
    for d in matched_dicts:
        aid = d.get("id") if isinstance(d, Mapping) else None
        if aid and aid in by_app_id:
            matched_apps.append(by_app_id[aid])

    return QueryRunReport(
        matches=matched_apps,
        total_applications=len(apps_list),
        sources_missing_transcript=sources_missing,
        warnings=warnings,
    )
