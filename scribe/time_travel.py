"""Time-travel view (F9.8).

Per PLANNING.md F9.8:

  > "Time-travel" view — display the project read-only as it was on
  > date Y.

The reproducibility story Scribe is building (F9.1 event log, F9.2
code-versioning, F9.3 codebook snapshots, F9.4 project checkpoints,
F9.5 lock-with-reason) culminates here: given an ISO-8601 timestamp
``as_of``, this module reconstructs **what the project looked like at
that moment** so a researcher (or thesis examiner) can show
"the codebook as of 2026-04-12" or "applications that existed on the
day the codebook was locked".

Design principles
-----------------

* **Read-only.** No writes. The reconstruction never touches on-disk
  state; it composes existing append-only logs and current entity
  files into an in-memory :class:`ProjectStateAtTime` value.
* **Pure where possible.** Each per-entity reconstructor takes a
  loaded list (or live store) and returns the at-time state. The
  module-level :func:`reconstruct_state_at` is the convenience wrapper
  that walks the on-disk tree.
* **Honest about precision.** Different entity types have different
  histories on disk:

    * **Codes** — F2.2 ``code_versions`` is an append-only per-code
      JSONL log of every definition edit. Picking the latest version
      with ``created_at <= as_of`` reconstructs the *exact* definition
      that was in force. ``deleted`` codes are detected via F9.1
      ``delete`` events (when emitted) **and** via the live "code is
      missing from ``codes/<id>.json`` but a version log existed for
      it" signal.

    * **Codebook lock + stage** — F2.4 ``codebook_lock_log.jsonl`` is
      an append-only log of every lock toggle with timestamps and the
      ``new_stage`` set by each toggle. The latest event up to
      ``as_of`` defines the lock state and stage at that time. With
      no lock events, the project's ``codebook_stage`` is treated as
      the steady state since ``project.created_at`` (best-effort:
      stage edits that bypass the lock helper aren't captured).

    * **Project metadata** — when F9.1 emits ``create`` / ``update``
      events on the project entity, we replay them. Otherwise we fall
      back to the live ``project.json`` if its ``created_at <= as_of``
      (the project itself existed); the metadata may have drifted
      since.

    * **Sources / Participants / Applications / Memos** — best-effort:
      we list the current on-disk entities and filter by
      ``created_at <= as_of``. This is exact about *which entities
      existed* but not their state-at-time if they were later edited.
      Each entity carries ``modified_at``; when ``modified_at >
      as_of`` we flag the row in :attr:`ProjectStateAtTime.warnings`
      so the consumer can surface a "this entity was modified after
      the as-of moment; we're showing its current state" notice.

* **No HTTP / FastAPI surface.** The reconstruction is the data
  layer; routes (``/api/projects/<id>/time-travel?as_of=...``) are a
  later iteration if needed. Same staged approach as F9.1, F9.6,
  F9.7.

On-disk dependencies
--------------------

This module reads from (but never writes):

* ``projects/<pid>/code_versions/<cid>.jsonl`` (F2.2).
* ``projects/<pid>/codebook_lock_log.jsonl`` (F2.4).
* ``projects/<pid>/events/<eid>.json`` (F9.1).
* ``projects/<pid>/codes/<cid>.json`` (F2.1) — to discover live code
  ids when reconstructing codes that have no version history.
* ``projects/<pid>/applications/<aid>.json`` (F4.1).
* ``projects/<pid>/memos/<mid>.json`` (F5.1).
* ``projects/<pid>/sources/<sid>.json`` (F1.2).
* ``projects/<pid>/participants/<pid>.json`` (F1.3).
* ``projects/<pid>/project.json`` (F1.1).

Pure helpers (:func:`replay_entity_states`, :func:`pick_version_at`,
:func:`lock_state_from_log`, :func:`filter_by_created_at`) operate on
already-loaded data structures so unit tests don't need a temp dir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .applications import Application, list_applications
from .codebook_lock import (
    LOCK_ACTIONS,
    LOCKED_STAGE,
    LockEvent,
    read_lock_log,
)
from .code_versions import (
    CodeVersion,
    read_code_versions,
)
from .codes import Code, list_codes
from .event_log import (
    EVENT_ACTION_CREATE,
    EVENT_ACTION_DELETE,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_PROJECT,
    Event,
    list_events,
)
from .memos import Memo, list_memos
from .participants import Participant, list_participants
from .projects import (
    CODEBOOK_STAGES,
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
    load_project,
)
from .sources import Source, list_sources


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Per-entity-kind labels used in :attr:`ProjectStateAtTime.warnings`
# and other diagnostic surfaces. Closed set so callers can switch on
# them without inventing new strings.
ENTITY_KIND_PROJECT = "project"
ENTITY_KIND_CODE = "code"
ENTITY_KIND_APPLICATION = "application"
ENTITY_KIND_MEMO = "memo"
ENTITY_KIND_SOURCE = "source"
ENTITY_KIND_PARTICIPANT = "participant"
ENTITY_KIND_CODEBOOK = "codebook"

ENTITY_KINDS: tuple[str, ...] = (
    ENTITY_KIND_PROJECT,
    ENTITY_KIND_CODE,
    ENTITY_KIND_APPLICATION,
    ENTITY_KIND_MEMO,
    ENTITY_KIND_SOURCE,
    ENTITY_KIND_PARTICIPANT,
    ENTITY_KIND_CODEBOOK,
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _validate_as_of(as_of: str) -> str:
    """Normalise / validate an ``as_of`` argument.

    We don't parse ISO-8601 strictly here — Scribe writes
    ``utcnow_iso()`` strings everywhere (Z-suffixed, microseconds), and
    those are lexically comparable as plain strings. We just enforce
    that the value is a non-empty string. Callers that need stricter
    parsing can layer it on top.
    """
    if not isinstance(as_of, str):
        raise ProjectValidationError(
            f"as_of must be an ISO-8601 string; got {type(as_of).__name__}"
        )
    s = as_of.strip()
    if not s:
        raise ProjectValidationError("as_of must be a non-empty timestamp")
    return s


def pick_version_at(
    versions: Sequence[CodeVersion], as_of: str
) -> CodeVersion | None:
    """Return the latest :class:`CodeVersion` with ``created_at <= as_of``.

    ``versions`` may be in any order — the function inspects every entry
    and returns the one with the largest ``created_at`` that does not
    exceed ``as_of``. Returns ``None`` if no version qualifies (the code
    didn't exist yet at ``as_of``).

    Pure helper; no I/O.
    """
    as_of = _validate_as_of(as_of)
    best: CodeVersion | None = None
    for v in versions:
        if not v.created_at:
            # Defensive: refuse to compare empty timestamps. Skip rather
            # than raise so a single bad row doesn't break replay.
            continue
        if v.created_at > as_of:
            continue
        if best is None or v.created_at > best.created_at or (
            v.created_at == best.created_at and v.version > best.version
        ):
            best = v
    return best


def replay_entity_states(
    events: Iterable[Event],
    *,
    as_of: str | None = None,
    entity_type: str | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Replay events to produce the latest *after-state* per entity-id.

    Walks ``events`` in chronological (timestamp, id) order. For each
    event whose ``entity_id`` is non-empty, the entity's reconstructed
    state is set to:

      * the event's ``after`` payload for any non-delete action (
        ``create`` / ``update`` / ``rename`` / ``merge`` / ``split`` /
        ``retire`` / ``promote`` / ``lock`` / ``unlock`` / ``snapshot``
        / ``checkpoint`` / ``import`` / ``export`` / ``other``), or
      * ``None`` for ``delete`` actions (signalling the entity no longer
        exists at this point).

    Events with empty ``entity_id`` are ignored — they describe the
    project as a whole rather than a specific record. (For project-
    level state, callers should use :func:`reconstruct_project_at`.)

    Filtering:

      * ``as_of`` (optional ISO-8601) — events strictly after ``as_of``
        are skipped.
      * ``entity_type`` (optional) — only events of that type contribute
        to the replay.

    Pure helper: takes already-loaded events and returns a dict.
    """
    if as_of is not None:
        as_of = _validate_as_of(as_of)
    sorted_events = sorted(
        events, key=lambda e: (e.created_at, e.id)
    )
    out: dict[str, dict[str, Any] | None] = {}
    for ev in sorted_events:
        if entity_type is not None and ev.entity_type != entity_type:
            continue
        if as_of is not None and ev.created_at > as_of:
            continue
        if not ev.entity_id:
            continue
        if ev.action == EVENT_ACTION_DELETE:
            out[ev.entity_id] = None
        else:
            # Defensive copy via ``dict`` so callers can mutate without
            # disturbing the next replay.
            out[ev.entity_id] = (
                dict(ev.after) if ev.after is not None else None
            )
    return out


def lock_state_from_log(
    events: Sequence[LockEvent], as_of: str, *, fallback_stage: str = "initial"
) -> tuple[bool, str]:
    """Derive ``(is_locked, stage)`` from a lock log at ``as_of``.

    Returns the state defined by the latest :class:`LockEvent` with
    ``created_at <= as_of``. If no event qualifies, returns
    ``(False, fallback_stage)`` — the project pre-dates the first lock
    toggle, so we treat it as unlocked at the project's own stage.

    The stage returned is the event's ``new_stage`` when set, falling
    back to :data:`LOCKED_STAGE` for ``lock`` actions and to
    ``fallback_stage`` for ``unlock`` actions whose ``new_stage`` was
    not recorded (older log entries; F2.4 always sets it but defensive
    here).

    Pure helper; no I/O.
    """
    as_of = _validate_as_of(as_of)
    if fallback_stage and fallback_stage not in CODEBOOK_STAGES:
        # Tolerate unknown fallback by coercing to ``initial`` — the
        # caller passing garbage shouldn't propagate it into the audit
        # surface.
        fallback_stage = "initial"
    best: LockEvent | None = None
    for ev in events:
        if not ev.created_at:
            continue
        if ev.created_at > as_of:
            continue
        if best is None or ev.created_at > best.created_at or (
            ev.created_at == best.created_at and ev.id > best.id
        ):
            best = ev
    if best is None:
        return (False, fallback_stage)
    if best.action == "lock":
        stage = best.new_stage or LOCKED_STAGE
        return (stage == LOCKED_STAGE, stage)
    # unlock
    stage = best.new_stage or fallback_stage
    return (False, stage)


def filter_by_created_at(
    items: Iterable[Any], as_of: str, *, attr: str = "created_at"
) -> list[Any]:
    """Return items whose ``attr`` (a timestamp string) is ``<= as_of``.

    Order-preserving filter. Items missing or with an empty timestamp
    are skipped (we can't place them on the timeline). Items where
    ``attr`` is not a string raise :class:`ProjectValidationError`.
    """
    as_of = _validate_as_of(as_of)
    out: list[Any] = []
    for item in items:
        ts = getattr(item, attr, "")
        if ts is None:
            continue
        if not isinstance(ts, str):
            raise ProjectValidationError(
                f"filter_by_created_at: {attr} must be a string, "
                f"got {type(ts).__name__}"
            )
        if not ts:
            continue
        if ts <= as_of:
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Per-entity reconstructors (touch disk)
# --------------------------------------------------------------------------- #


def reconstruct_codes_at(
    projects_root: Path, project_id: str, as_of: str
) -> list[Code]:
    """Return the codebook as it was at ``as_of``.

    Strategy:

      1. Discover the universe of code ids: the union of the live
         ``codes/`` directory and code ids that appear in any
         ``code_versions/<cid>.jsonl`` (so a code that has since been
         hard-deleted is still reconstructable from its version log).
      2. For each candidate code, read its version log via F2.2's
         :func:`read_code_versions` and pick the latest version with
         ``created_at <= as_of`` via :func:`pick_version_at`. If no
         version qualifies, the code didn't exist yet — skip.
      3. Replay F9.1 ``delete`` events on entity_type=``code`` up to
         ``as_of``: any code with a delete event in that window is
         excluded from the result.
      4. Build a :class:`Code` from the chosen snapshot.

    Codes without a version log fall back to filtering by
    ``Code.created_at`` (legacy data, or imports that bypassed
    :func:`save_code_with_version`). They appear in the result if
    ``code.created_at <= as_of``.

    Sorted by ``name`` for stable display.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)

    # Discover candidate code ids: live codes + code-version logs.
    live_codes = list_codes(projects_root, project_id)
    candidate_ids: dict[str, Code | None] = {c.id: c for c in live_codes}

    from .code_versions import code_versions_dir

    cv_dir = code_versions_dir(projects_root, project_id)
    if cv_dir.exists():
        for f in cv_dir.iterdir():
            if not f.is_file() or not f.name.endswith(".jsonl"):
                continue
            cid = f.stem
            if cid not in candidate_ids:
                candidate_ids[cid] = None  # version log only, no live code

    # Replay F9.1 delete events for codes up to as_of so a code that
    # was created and later deleted before as_of disappears correctly.
    try:
        events = list_events(
            projects_root,
            project_id,
            entity_type=EVENT_ENTITY_CODE,
            until=as_of,
        )
    except (FileNotFoundError, ProjectValidationError):
        events = []
    deleted_ids: set[str] = set()
    # Walk in chrono order; a delete then a re-create flips state back
    # to alive. We respect the latest action.
    for ev in events:
        if not ev.entity_id:
            continue
        if ev.action == EVENT_ACTION_DELETE:
            deleted_ids.add(ev.entity_id)
        elif ev.action == EVENT_ACTION_CREATE:
            deleted_ids.discard(ev.entity_id)

    out: list[Code] = []
    for cid, live in candidate_ids.items():
        if cid in deleted_ids:
            continue
        try:
            versions = read_code_versions(projects_root, project_id, cid)
        except ProjectValidationError:
            versions = []
        chosen = pick_version_at(versions, as_of) if versions else None
        if chosen is not None:
            try:
                out.append(Code.from_dict(dict(chosen.snapshot)))
            except ProjectValidationError:
                # Malformed snapshot — skip rather than break the view.
                continue
            continue
        # No version history: fall back to the live entity's
        # ``created_at`` if it pre-dates ``as_of``.
        if live is None:
            continue
        if live.created_at and live.created_at <= as_of:
            out.append(live)
    out.sort(key=lambda c: (c.name.lower(), c.id))
    return out


def reconstruct_codebook_lock_state_at(
    projects_root: Path, project_id: str, as_of: str
) -> tuple[bool, str]:
    """Return ``(is_locked, codebook_stage)`` at ``as_of``.

    Reads F2.4's lock log and applies :func:`lock_state_from_log`. The
    fallback stage (when no lock events have happened yet) is the
    project's *current* ``codebook_stage`` — best-effort: if the user
    edited the stage outside the lock helper, that history isn't
    captured. The :class:`ProjectStateAtTime.warnings` surface flags
    this case.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)
    try:
        log = read_lock_log(projects_root, project_id)
    except (FileNotFoundError, ProjectValidationError):
        log = []
    fallback = "initial"
    try:
        proj = load_project(projects_root, project_id)
        if proj.codebook_stage in CODEBOOK_STAGES:
            fallback = proj.codebook_stage
    except (FileNotFoundError, ProjectValidationError):
        pass
    return lock_state_from_log(log, as_of, fallback_stage=fallback)


def reconstruct_project_at(
    projects_root: Path, project_id: str, as_of: str
) -> Project | None:
    """Return the :class:`Project` entity as it was at ``as_of``.

    Strategy:

      1. Replay F9.1 events on entity_type=``project`` up to ``as_of``.
         If a non-empty ``after`` payload is left for any project id
         (typically the project itself), build a :class:`Project` from
         it.
      2. If no events apply, fall back to the live ``project.json``
         when its ``created_at <= as_of``.
      3. Return ``None`` if the project didn't exist yet at ``as_of``.

    The returned Project is a *fresh* dataclass; mutating it does not
    affect on-disk state.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)

    try:
        events = list_events(
            projects_root,
            project_id,
            entity_type=EVENT_ENTITY_PROJECT,
            until=as_of,
        )
    except (FileNotFoundError, ProjectValidationError):
        events = []
    states = replay_entity_states(
        events, as_of=as_of, entity_type=EVENT_ENTITY_PROJECT
    )
    # Prefer the entry keyed by ``project_id`` itself.
    after = states.get(project_id)
    if after:
        try:
            return Project.from_dict(dict(after))
        except ProjectValidationError:
            pass
    # If the latest project event ended in a delete (or a malformed
    # state), refuse to forge a Project from the live disk.
    if project_id in states and states[project_id] is None:
        return None

    # Fall back to live project if its created_at <= as_of.
    try:
        live = load_project(projects_root, project_id)
    except (FileNotFoundError, ProjectValidationError):
        return None
    if live.created_at and live.created_at <= as_of:
        return live
    return None


def reconstruct_applications_at(
    projects_root: Path, project_id: str, as_of: str
) -> list[Application]:
    """Return applications that *existed* at ``as_of``.

    Best-effort: filters live applications by ``created_at <= as_of``.
    Applications modified after ``as_of`` are returned in their *current*
    state (we don't have an application-version log). Callers should
    treat the returned list as "which applications were live at the
    moment", not "their exact state at the moment".
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)
    return filter_by_created_at(
        list_applications(projects_root, project_id), as_of
    )


def reconstruct_memos_at(
    projects_root: Path, project_id: str, as_of: str
) -> list[Memo]:
    """Return memos that *existed* at ``as_of``. Best-effort, like
    :func:`reconstruct_applications_at`."""
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)
    return filter_by_created_at(list_memos(projects_root, project_id), as_of)


def reconstruct_sources_at(
    projects_root: Path, project_id: str, as_of: str
) -> list[Source]:
    """Return sources that *existed* at ``as_of``. Best-effort."""
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)
    return filter_by_created_at(list_sources(projects_root, project_id), as_of)


def reconstruct_participants_at(
    projects_root: Path, project_id: str, as_of: str
) -> list[Participant]:
    """Return participants that *existed* at ``as_of``. Best-effort."""
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)
    return filter_by_created_at(
        list_participants(projects_root, project_id), as_of
    )


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


@dataclass
class ProjectStateAtTime:
    """Read-only snapshot of a project's state at a given moment.

    Built by :func:`reconstruct_state_at`. Fields:

      project_id
        12-char hex project id this snapshot is for.
      as_of
        ISO-8601 timestamp the state was reconstructed against.
      project
        The :class:`Project` entity at ``as_of`` (or ``None`` if the
        project didn't exist yet).
      codes
        :class:`Code` instances at ``as_of``, with definitions pinned
        from F2.2 version snapshots.
      applications
        Applications live at ``as_of`` (best-effort: see
        :func:`reconstruct_applications_at`).
      memos
        Memos live at ``as_of`` (best-effort).
      sources
        Sources live at ``as_of`` (best-effort).
      participants
        Participants live at ``as_of`` (best-effort).
      codebook_stage
        Stage at ``as_of`` from the lock log.
      codebook_locked
        Whether the codebook was locked at ``as_of``.
      best_effort
        ``True`` when the reconstruction included any best-effort
        signal (i.e. any of applications / memos / sources /
        participants exist; or codebook_stage came from the live
        project rather than the lock log). The codebook itself is
        always exact when there's a version log.
      warnings
        List of human-readable strings describing degraded data
        (e.g. ``"memo abcdef0123 was modified after as_of"``). The
        UI should surface these.
    """

    project_id: str
    as_of: str
    project: Project | None = None
    codes: list[Code] = field(default_factory=list)
    applications: list[Application] = field(default_factory=list)
    memos: list[Memo] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    participants: list[Participant] = field(default_factory=list)
    codebook_stage: str = "initial"
    codebook_locked: bool = False
    best_effort: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Embedded entities use their own ``to_dict()`` so the result is
        plain JSON. Mainly for HTTP responses + audit-trail exports.
        """
        return {
            "project_id": self.project_id,
            "as_of": self.as_of,
            "project": (
                self.project.to_dict() if self.project is not None else None
            ),
            "codes": [c.to_dict() for c in self.codes],
            "applications": [a.to_dict() for a in self.applications],
            "memos": [m.to_dict() for m in self.memos],
            "sources": [s.to_dict() for s in self.sources],
            "participants": [p.to_dict() for p in self.participants],
            "codebook_stage": self.codebook_stage,
            "codebook_locked": self.codebook_locked,
            "best_effort": self.best_effort,
            "warnings": list(self.warnings),
        }


def reconstruct_state_at(
    projects_root: Path,
    project_id: str,
    as_of: str,
    *,
    include_applications: bool = True,
    include_memos: bool = True,
    include_sources: bool = True,
    include_participants: bool = True,
) -> ProjectStateAtTime:
    """Reconstruct the full project state at ``as_of``.

    Composes every per-entity reconstructor above into one
    :class:`ProjectStateAtTime` value. The optional ``include_*`` flags
    let callers skip the best-effort sections when they only need the
    exact-history parts (project + codebook).

    Raises :class:`ProjectValidationError` for invalid arguments;
    propagates :class:`FileNotFoundError` from the underlying loaders
    only when the project directory itself is missing.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    as_of = _validate_as_of(as_of)

    state = ProjectStateAtTime(project_id=project_id, as_of=as_of)
    state.project = reconstruct_project_at(projects_root, project_id, as_of)
    state.codes = reconstruct_codes_at(projects_root, project_id, as_of)

    locked, stage = reconstruct_codebook_lock_state_at(
        projects_root, project_id, as_of
    )
    state.codebook_locked = locked
    state.codebook_stage = stage

    # Cross-check: did the lock state come from the lock log, or from
    # the project fallback? read_lock_log is cheap enough to call again
    # for the diagnostic.
    try:
        lock_log = read_lock_log(projects_root, project_id)
    except (FileNotFoundError, ProjectValidationError):
        lock_log = []
    in_window = [
        ev for ev in lock_log if ev.created_at and ev.created_at <= as_of
    ]
    if not in_window:
        # Used the live project's stage as fallback — flag it.
        state.best_effort = True
        state.warnings.append(
            f"codebook_stage at {as_of} taken from current project; "
            "no lock log events occurred in the window"
        )

    if include_applications:
        apps = reconstruct_applications_at(projects_root, project_id, as_of)
        state.applications = apps
        if apps:
            state.best_effort = True
            for a in apps:
                if a.modified_at and a.modified_at > as_of:
                    state.warnings.append(
                        f"application {a.id} modified after as_of "
                        f"({a.modified_at}); showing current state"
                    )

    if include_memos:
        memos = reconstruct_memos_at(projects_root, project_id, as_of)
        state.memos = memos
        if memos:
            state.best_effort = True
            for m in memos:
                if m.modified_at and m.modified_at > as_of:
                    state.warnings.append(
                        f"memo {m.id} modified after as_of "
                        f"({m.modified_at}); showing current state"
                    )

    if include_sources:
        sources = reconstruct_sources_at(projects_root, project_id, as_of)
        state.sources = sources
        if sources:
            state.best_effort = True
            for s in sources:
                if s.modified_at and s.modified_at > as_of:
                    state.warnings.append(
                        f"source {s.id} modified after as_of "
                        f"({s.modified_at}); showing current state"
                    )

    if include_participants:
        parts = reconstruct_participants_at(projects_root, project_id, as_of)
        state.participants = parts
        if parts:
            state.best_effort = True
            for p in parts:
                if p.modified_at and p.modified_at > as_of:
                    state.warnings.append(
                        f"participant {p.id} modified after as_of "
                        f"({p.modified_at}); showing current state"
                    )

    return state


__all__ = [
    "ENTITY_KIND_APPLICATION",
    "ENTITY_KIND_CODE",
    "ENTITY_KIND_CODEBOOK",
    "ENTITY_KIND_MEMO",
    "ENTITY_KIND_PARTICIPANT",
    "ENTITY_KIND_PROJECT",
    "ENTITY_KIND_SOURCE",
    "ENTITY_KINDS",
    "ProjectStateAtTime",
    "filter_by_created_at",
    "lock_state_from_log",
    "pick_version_at",
    "reconstruct_applications_at",
    "reconstruct_codebook_lock_state_at",
    "reconstruct_codes_at",
    "reconstruct_memos_at",
    "reconstruct_participants_at",
    "reconstruct_project_at",
    "reconstruct_sources_at",
    "reconstruct_state_at",
    "replay_entity_states",
]
