"""Participant ↔ source mapping (F3.3).

Per PLANNING.md F3.3:

  > Participant ↔ source mapping; one participant can have multiple
  > sources.

F1.3 already lets each :class:`scribe.participants.Participant` carry a
``source_ids`` list — that's the **forward** direction (one participant,
many sources, the longitudinal-interview pattern). F3.3 closes the
mapping by exposing:

  * the **inverse** direction (one source, many participants — the
    focus-group / multi-speaker pattern, called out in
    ``docs/research/coding-engine-research.md``);
  * **set-style** helpers so the UI can declare "the participants in
    this focus group are X, Y, Z" in a single call rather than
    add/remove fiddling per participant;
  * **integrity checks** so a participant whose ``source_ids`` points
    at a deleted source is surfaced rather than silently lost.

The on-disk layout is unchanged — the mapping lives entirely on the
participant side as a list of source IDs (F1.3). What's new here is the
algorithmic layer that walks the project's participant + source files
and computes the inverse / set-difference / orphan views without ever
mutating data outside the participants directory.

Like the rest of the F1.* / F2.* / F3.* foundation modules this is
stand-alone — no FastAPI, no engine imports — so it's testable in pure
Python and reusable from the CLI later. Conventions match
``scribe.projects`` (F1.1), ``scribe.sources`` (F1.2),
``scribe.participants`` (F1.3), and ``scribe.source_schema`` (F3.2).

Design notes:

* **Source side stays read-only.** A focus group's participants are
  recorded by mutating each *participant's* ``source_ids`` (adding /
  removing the source's id). Sources don't get a sister ``participant_ids``
  field; we'd then have two places to keep in sync. Inverse navigation
  is computed by scanning participant files, which is fast at the
  scales researchers actually have (10s–100s of participants per
  project).
* **Idempotent set operations.** ``set_participants_for_source``
  computes the diff against the on-disk truth and only writes the
  participants whose lists actually change. No spurious ``modified_at``
  bumps. Mirrors how ``Participant.add_source`` is already idempotent.
* **Orphan detection is non-destructive.** ``find_orphan_links``
  reports broken edges; it doesn't auto-clean. Researchers reviewing
  a project state want to *see* "P03 references source 0123… which no
  longer exists" before deciding whether that's a fixable typo or a
  legitimately deleted source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .participants import (
    PARTICIPANT_ID_RE,
    Participant,
    list_participants,
    load_participant,
    save_participant,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    utcnow_iso,
)
from .sources import (
    SOURCE_ID_RE,
    Source,
    list_sources,
)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _check_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")


def _check_source_id(source_id: str) -> None:
    if not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(f"Invalid source id: {source_id!r}")


def _check_participant_id(participant_id: str) -> None:
    if not PARTICIPANT_ID_RE.match(participant_id):
        raise ProjectValidationError(
            f"Invalid participant id: {participant_id!r}"
        )


# --------------------------------------------------------------------------- #
# Inverse navigation: participants for a source / sources for a participant
# --------------------------------------------------------------------------- #


def list_participants_for_source(
    projects_root: Path, project_id: str, source_id: str
) -> list[Participant]:
    """Return every participant whose ``source_ids`` includes ``source_id``.

    Returns an empty list when nothing is linked. Order follows
    :func:`scribe.participants.list_participants` (created_at ascending,
    then participant id) so the result is stable.

    Validates ``source_id`` shape so a typo or path-traversal attempt
    fails loudly rather than returning an empty list.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    return [
        p
        for p in list_participants(projects_root, project_id)
        if source_id in p.source_ids
    ]


def list_sources_for_participant(
    projects_root: Path, project_id: str, participant_id: str
) -> list[Source]:
    """Return the actual :class:`Source` objects this participant references.

    Resolves each id in the participant's ``source_ids`` to a real
    Source on disk; missing references are silently skipped (use
    :func:`find_orphan_links` for the auditor's view). Order matches
    the participant's own ``source_ids`` so the researcher's preferred
    ordering (earliest interview first, etc.) is preserved.
    """
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    p = load_participant(projects_root, project_id, participant_id)
    by_id: dict[str, Source] = {
        s.id: s for s in list_sources(projects_root, project_id)
    }
    out: list[Source] = []
    for sid in p.source_ids:
        s = by_id.get(sid)
        if s is not None:
            out.append(s)
    return out


def participant_source_map(
    projects_root: Path, project_id: str
) -> dict[str, list[str]]:
    """Return ``{source_id: [participant_id, ...]}`` for the whole project.

    A snapshot of the inverse mapping useful to the UI (rendering a
    focus-group's roster, building a source-by-participant matrix for
    F3.6). Sources with no linked participants are still present as
    keys (with an empty list) so the caller can iterate every source
    in the project. Sources referenced by a participant but not present
    on disk show up as keys too — the auditor's view — so callers can
    detect orphans inline; use :func:`find_orphan_links` if a clean
    distinction matters.

    Participant ids inside each list are sorted by participant
    ``created_at`` then id (matches ``list_participants``).
    """
    _check_project_id(project_id)
    parts = list_participants(projects_root, project_id)
    sources = list_sources(projects_root, project_id)

    out: dict[str, list[str]] = {s.id: [] for s in sources}
    for p in parts:
        for sid in p.source_ids:
            out.setdefault(sid, []).append(p.id)
    return out


# --------------------------------------------------------------------------- #
# Orphan detection
# --------------------------------------------------------------------------- #


@dataclass
class OrphanLink:
    """One broken participant→source edge.

    ``participant_id`` references ``source_id`` in its ``source_ids``
    list, but no Source with that id exists in the project. Used by
    :func:`find_orphan_links` and surfaced in audit / cleanup UIs.
    """

    participant_id: str
    source_id: str


def find_orphan_links(
    projects_root: Path, project_id: str
) -> list[OrphanLink]:
    """List every ``source_ids`` reference that points at a missing source.

    Order: by participant ``created_at`` / id (outer), then by the
    order the missing source ids appear in the participant's list
    (inner). Stable across runs.
    """
    _check_project_id(project_id)
    parts = list_participants(projects_root, project_id)
    known_sids = {s.id for s in list_sources(projects_root, project_id)}
    out: list[OrphanLink] = []
    for p in parts:
        for sid in p.source_ids:
            if sid not in known_sids:
                out.append(OrphanLink(participant_id=p.id, source_id=sid))
    return out


# --------------------------------------------------------------------------- #
# Single-edge mutation
# --------------------------------------------------------------------------- #


def link_participant_to_source(
    projects_root: Path,
    project_id: str,
    participant_id: str,
    source_id: str,
    *,
    require_source_exists: bool = True,
    now: str | None = None,
) -> bool:
    """Attach a source to a participant. Persists on success.

    Returns True if a new edge was added, False if it was already
    present (no write, no clock advance — mirrors
    :meth:`Participant.add_source`).

    By default the source must exist on disk; pass
    ``require_source_exists=False`` to permit a forward reference (e.g.
    during import where sources land later in the same batch).
    """
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    _check_source_id(source_id)
    if require_source_exists:
        known_sids = {s.id for s in list_sources(projects_root, project_id)}
        if source_id not in known_sids:
            raise ProjectValidationError(
                f"Source {source_id!r} not found in project {project_id!r}"
            )
    p = load_participant(projects_root, project_id, participant_id)
    added = p.add_source(source_id, now=now)
    if added:
        save_participant(projects_root, p)
    return added


def unlink_participant_from_source(
    projects_root: Path,
    project_id: str,
    participant_id: str,
    source_id: str,
    *,
    now: str | None = None,
) -> bool:
    """Detach a source from a participant. Persists on success.

    Returns True if an edge was removed, False if it wasn't there to
    begin with. ``source_id`` shape is validated; we don't insist the
    *source* itself exists, since the common case for unlinking is
    cleaning up after a deleted source.
    """
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    _check_source_id(source_id)
    p = load_participant(projects_root, project_id, participant_id)
    removed = p.remove_source(source_id, now=now)
    if removed:
        save_participant(projects_root, p)
    return removed


# --------------------------------------------------------------------------- #
# Focus-group / set-style mutation
# --------------------------------------------------------------------------- #


@dataclass
class ParticipantSourceChange:
    """Diff produced by :func:`set_participants_for_source`.

    ``added`` and ``removed`` are participant ids. The persisted
    ``Participant`` objects can be reloaded via the standard
    ``load_participant`` helper if a caller needs the full record;
    for the UI confirmation toast / audit log entry the id list is
    enough.
    """

    source_id: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def set_participants_for_source(
    projects_root: Path,
    project_id: str,
    source_id: str,
    participant_ids: Iterable[str],
    *,
    require_source_exists: bool = True,
    now: str | None = None,
) -> ParticipantSourceChange:
    """Declare exactly which participants are linked to ``source_id``.

    The focus-group editor pattern: the UI presents a checklist and
    the user clicks Save. This function:

      1. validates ids,
      2. checks each participant exists,
      3. computes the diff against the on-disk truth (which
         participants currently have ``source_id`` in their
         ``source_ids`` list),
      4. writes only the participants whose list actually changes.

    Idempotent: calling twice with the same desired set is a no-op on
    the second call (no clock bumps, no orphan writes). Stamp
    ``modified_at`` on each touched participant from ``now`` (or
    :func:`scribe.projects.utcnow_iso` if absent).

    Forward-references (``source_id`` not yet on disk) are rejected
    by default; pass ``require_source_exists=False`` to allow them
    (used by importers that stage participants before sources).
    """
    _check_project_id(project_id)
    _check_source_id(source_id)

    # Validate participant ids up front and de-dupe while preserving
    # insertion order. We don't accept duplicates silently because the
    # caller almost certainly has a UI bug if it submitted ["P03","P03"].
    desired_order: list[str] = []
    desired_set: set[str] = set()
    for pid in participant_ids:
        s = str(pid)
        _check_participant_id(s)
        if s in desired_set:
            continue
        desired_set.add(s)
        desired_order.append(s)

    if require_source_exists:
        known_sids = {s.id for s in list_sources(projects_root, project_id)}
        if source_id not in known_sids:
            raise ProjectValidationError(
                f"Source {source_id!r} not found in project {project_id!r}"
            )

    # Snapshot every participant in the project. We need both:
    #   * the desired set (to know who must end up linked), and
    #   * the existing inverse (to know who must end up unlinked),
    # so iterating the full list once is the cleanest path.
    parts = list_participants(projects_root, project_id)
    by_id: dict[str, Participant] = {p.id: p for p in parts}

    # Every desired id must correspond to an existing participant.
    missing = [pid for pid in desired_order if pid not in by_id]
    if missing:
        raise ProjectValidationError(
            f"Unknown participant ids in project {project_id!r}: "
            f"{', '.join(missing)}"
        )

    ts = now or utcnow_iso()
    change = ParticipantSourceChange(source_id=source_id)

    for p in parts:
        already_linked = source_id in p.source_ids
        should_be_linked = p.id in desired_set
        if should_be_linked and not already_linked:
            p.add_source(source_id, now=ts)
            save_participant(projects_root, p)
            change.added.append(p.id)
        elif already_linked and not should_be_linked:
            p.remove_source(source_id, now=ts)
            save_participant(projects_root, p)
            change.removed.append(p.id)
        elif should_be_linked and already_linked:
            change.unchanged.append(p.id)

    return change
