"""F9.5 — Locked-codebook mode with reason-to-unlock memo (audit integration).

Per PLANNING.md F9.5:

  > Locked-codebook mode with reason-to-unlock memo.

F2.4 (``scribe.codebook_lock``) shipped the *primitive*: toggle the
project's ``codebook_stage`` to ``"locked"``, write an entry to a
dedicated ``codebook_lock_log.jsonl``, require a non-empty
``methodological_memo`` on unlock. Methodologically, that's already a
real lock. What F2.4 did **not** do — and what this module adds — is
make the unlock memo and the lock event **visible to the rest of
Scribe's audit machinery**:

  1. **Unlock memos become first-class** :class:`scribe.memos.Memo`
     records of type ``methodological``, linked to the project with
     role ``codebook_unlock``. That's what surfaces them in the F5.4
     memo export, the F6.4 REFI-QDA bundle, and any future memo-
     centric UI. Without F9.5 they'd live only in the lock-log JSONL,
     invisible to those exports.
  2. **Lock and unlock land in the F9.1 event log** alongside every
     other project-level operation. F9.7 (audit-trail export) and
     F9.8 (time-travel view) walk the F9.1 log; without an entry there
     the lock toggles would be missing from any unified audit.

This is deliberately a thin wrapper module — F2.4 stays the single
source of truth for the lock state itself, and F5.1 / F9.1 stay the
sole writers of memos and events. F9.5 just composes them so a UI or
CLI caller gets all three artefacts (lock log entry, methodological
memo, F9.1 event) in one transactional call.

What stays in F2.4
------------------

The lock log file (``codebook_lock_log.jsonl``) remains. It's a
self-contained record of lock toggles with the methodological memo
inline; F9.5 *does not* delete it. The redundancy is intentional:
the lock log is a small, append-only file that survives even if the
F9.1 events directory or the Memo files are corrupted, and inversely
the F9.1 view is the unified one researchers will reach for in
practice. Two views, one source of truth in the project's
``codebook_stage`` field.

Boundaries
----------

* **No HTTP / FastAPI surface here.** The HTTP endpoints
  (``/api/projects/<id>/codebook/lock`` etc.) are added by a later
  iteration if we wire a UI; F9.5 is the data composition layer.
* **Stand-alone, pure Python.** Imports F2.4 (``codebook_lock``),
  F5.1 (``memos``), F9.1 (``event_log``) only. No engine imports.

On-disk layout (no new files)
-----------------------------

After ``unlock_codebook_with_memo``::

    projects/<pid>/
      project.json                       # codebook_stage updated (F1.1)
      codebook_lock_log.jsonl            # +1 line (F2.4)
      memos/<memo_id>.json               # +1 file (F5.1, type=methodological)
      events/<event_id>.json             # +1 file (F9.1, action=unlock)

After ``lock_codebook_with_audit``::

    projects/<pid>/
      project.json                       # codebook_stage = locked (F1.1)
      codebook_lock_log.jsonl            # +1 line (F2.4)
      events/<event_id>.json             # +1 file (F9.1, action=lock)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .codebook_lock import (
    LockEvent,
    lock_codebook,
    read_lock_log,
    unlock_codebook,
)
from .coders import CODER_ID_RE
from .event_log import (
    EVENT_ACTION_LOCK,
    EVENT_ACTION_UNLOCK,
    EVENT_ENTITY_CODEBOOK,
    Event,
    list_events,
    record_event,
)
from .memos import (
    MAX_PROVENANCE_VALUE_LEN,
    MAX_TITLE_LEN,
    Memo,
    MemoLink,
    list_memos,
    save_memo,
)
from .projects import (
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Role used on the methodological memo's project link so the F9.5
# helpers can find the unlock memos again later (and so any UI can
# style "this memo justified an unlock" specially). The value is a
# valid LINK_ROLE_RE token (letters / digits / underscore / hyphen /
# space, must start with a letter — ``codebook_unlock`` qualifies).
UNLOCK_LINK_ROLE = "codebook_unlock"

# Provenance source recorded on the methodological memo. ``other`` is
# the closest match in :data:`scribe.memos.MEMO_PROVENANCE_SOURCES`
# for "this memo was emitted by a system action that wraps a human
# decision" — neither pure ``human`` (no free-text editor pass) nor
# ``imported`` (it isn't from another file). The accompanying
# ``codebook_lock_event_id`` provenance value points at the F2.4 log
# entry so a reader can cross-reference the JSONL line.
UNLOCK_PROVENANCE_SOURCE = "other"

# Cap on how much of the unlock reason is folded into the memo title.
# We hard-truncate to MAX_TITLE_LEN minus the prefix so a 2000-char
# reason (the F2.4 cap) doesn't blow the title constraint.
_TITLE_PREFIX = "Codebook unlock: "
_MAX_REASON_IN_TITLE = MAX_TITLE_LEN - len(_TITLE_PREFIX)
_LOCK_TITLE_PREFIX = "Codebook lock: "
_MAX_REASON_IN_LOCK_TITLE = MAX_TITLE_LEN - len(_LOCK_TITLE_PREFIX)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class LockResult:
    """Triple of artefacts produced by :func:`lock_codebook_with_audit`.

    * ``project`` — the post-lock :class:`Project` (codebook_stage =
      ``"locked"``).
    * ``lock_event`` — the F2.4 :class:`LockEvent` appended to the
      per-project ``codebook_lock_log.jsonl``.
    * ``event`` — the F9.1 :class:`Event` recorded under
      ``events/<event_id>.json``. Notes carries the reason; ``before``
      / ``after`` capture the stage transition.
    """

    project: Project
    lock_event: LockEvent
    event: Event


@dataclass
class UnlockResult:
    """Quadruple of artefacts produced by :func:`unlock_codebook_with_memo`.

    * ``project`` — the post-unlock :class:`Project` (codebook_stage
      restored to a non-locked stage).
    * ``lock_event`` — the F2.4 :class:`LockEvent` (action ``unlock``)
      appended to the per-project ``codebook_lock_log.jsonl``. Carries
      the methodological memo body inline.
    * ``memo`` — the first-class :class:`Memo` (type
      ``methodological``) created for the unlock justification, with a
      ``project`` link bearing role ``codebook_unlock``. Provenance
      includes ``codebook_lock_event_id`` and ``codebook_unlock_reason``
      so a reader can reconstruct the full context.
    * ``event`` — the F9.1 :class:`Event` (action ``unlock``) recorded
      under ``events/<event_id>.json``. ``after`` payload includes
      ``memo_id`` and ``lock_event_id`` so a unified audit trail can
      jump from the event to either of the two sidecar artefacts.
    """

    project: Project
    lock_event: LockEvent
    memo: Memo
    event: Event


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _validate_actor_coder_id(actor_coder_id: str) -> str:
    """Return ``actor_coder_id`` after light validation.

    Empty / falsy actor is allowed (system / anonymous events) and
    returned as ``""``. A non-empty value must match the canonical
    coder-id shape; F9.1's :class:`Event` validates this too, but
    surfacing the rejection here gives a clearer call-site error.
    """
    if not actor_coder_id:
        return ""
    if not CODER_ID_RE.match(actor_coder_id):
        raise ProjectValidationError(
            f"Invalid actor_coder_id: {actor_coder_id!r}"
        )
    return actor_coder_id


def _truncate_for_provenance(value: str) -> str:
    """Cap a provenance value at :data:`MAX_PROVENANCE_VALUE_LEN`.

    F2.4 allows ``reason`` up to 2000 chars and ``methodological_memo``
    up to 8000 chars; F5.1's provenance values cap at 1000. Long
    reasons get hard-truncated with a trailing ellipsis so the F9.5
    memo still validates. The full text remains in the memo body and
    in the F2.4 lock-log line — provenance only carries a summary.
    """
    if len(value) <= MAX_PROVENANCE_VALUE_LEN:
        return value
    # Reserve one char for the ellipsis marker.
    return value[: MAX_PROVENANCE_VALUE_LEN - 1].rstrip() + "…"


def _truncate_for_title(reason: str, *, prefix: str, max_reason: int) -> str:
    """Build a memo title from a prefix + (possibly truncated) reason.

    Memo titles cap at :data:`scribe.memos.MAX_TITLE_LEN`; long
    F2.4-style reasons (≤2000 chars) get abridged with a trailing
    ellipsis. Falls back to the prefix alone when ``max_reason``
    leaves no room.
    """
    reason = reason.strip()
    if max_reason <= 1:
        return prefix.rstrip(": ")
    if len(reason) <= max_reason:
        return f"{prefix}{reason}"
    return f"{prefix}{reason[: max_reason - 1].rstrip()}…"


# --------------------------------------------------------------------------- #
# Public API — lock with audit
# --------------------------------------------------------------------------- #


def lock_codebook_with_audit(
    projects_root: Path,
    project_id: str,
    *,
    reason: str,
    actor_coder_id: str = "",
    now: str | None = None,
) -> LockResult:
    """Lock the project's codebook and record an F9.1 event.

    Composes:

    * F2.4 :func:`scribe.codebook_lock.lock_codebook` — sets
      ``project.codebook_stage = "locked"`` and appends a lock-log
      entry.
    * F9.1 :func:`scribe.event_log.record_event` — writes a
      ``events/<event_id>.json`` with ``action="lock"``,
      ``entity_type="codebook"``, before/after ``codebook_stage``,
      and ``notes=reason``.

    The two writes are sequential, not transactional: a crash between
    them leaves the F2.4 lock log up to date but the F9.1 events
    directory missing the entry. F9.7's audit export tolerates that
    shape (it reads both sources). Errors raised by the F2.4 layer
    propagate without writing the F9.1 event.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    actor = _validate_actor_coder_id(actor_coder_id)

    project, lock_event = lock_codebook(
        projects_root, project_id, reason=reason, now=now
    )

    event = record_event(
        projects_root,
        project_id=project_id,
        action=EVENT_ACTION_LOCK,
        entity_type=EVENT_ENTITY_CODEBOOK,
        entity_id="",
        actor_coder_id=actor,
        before={"codebook_stage": lock_event.prior_stage},
        after={
            "codebook_stage": lock_event.new_stage,
            "lock_event_id": lock_event.id,
        },
        notes=lock_event.reason,
        now=now,
    )
    return LockResult(project=project, lock_event=lock_event, event=event)


# --------------------------------------------------------------------------- #
# Public API — unlock with memo + audit
# --------------------------------------------------------------------------- #


def unlock_codebook_with_memo(
    projects_root: Path,
    project_id: str,
    *,
    reason: str,
    methodological_memo: str,
    new_stage: str | None = None,
    author_coder_id: str | None = None,
    actor_coder_id: str = "",
    now: str | None = None,
) -> UnlockResult:
    """Unlock the codebook, persist the memo as F5.1, and record F9.1.

    Composes three writes:

    1. F2.4 :func:`scribe.codebook_lock.unlock_codebook` — flips
       ``project.codebook_stage`` back to ``new_stage`` (or the most
       recent prior stage from the lock log) and appends an unlock
       entry to ``codebook_lock_log.jsonl`` with the
       ``methodological_memo`` inline.
    2. F5.1 :func:`scribe.memos.save_memo` — creates a new
       :class:`Memo` of type ``methodological`` with the unlock memo
       body, a ``project`` :class:`MemoLink` bearing role
       ``codebook_unlock``, and provenance pointing at the lock-log
       entry id. The memo is now visible to F5.4 / F6.4 exports.
    3. F9.1 :func:`scribe.event_log.record_event` — records an
       ``unlock`` event whose ``after`` payload includes both the
       memo id and the lock-log event id so the audit trail can
       cross-reference all three artefacts.

    Both ``reason`` and ``methodological_memo`` must be non-empty
    (per F2.4); validation errors propagate from the F2.4 layer and no
    memo / event is written.

    ``author_coder_id`` is optional. When supplied it lands on the
    Memo as its ``author_coder_id``, mirroring how ``Application``
    carries human authorship. ``actor_coder_id`` lands on the F9.1
    event; the two are deliberately separate so a system-triggered
    unlock (e.g. a CLI script) can have an actor of "" while the
    memo records the human researcher behind the decision.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    actor = _validate_actor_coder_id(actor_coder_id)

    # Light validation of author_coder_id ahead of time (Memo would
    # validate too; surfacing here gives a clearer call site).
    if author_coder_id is not None and author_coder_id != "":
        if not CODER_ID_RE.match(author_coder_id):
            raise ProjectValidationError(
                f"Invalid author_coder_id: {author_coder_id!r}"
            )

    # Step 1 — F2.4 unlock. If this raises (codebook not locked,
    # missing memo, etc.), we exit before writing the Memo or the
    # F9.1 event.
    project, lock_event = unlock_codebook(
        projects_root,
        project_id,
        reason=reason,
        methodological_memo=methodological_memo,
        new_stage=new_stage,
        now=now,
    )

    # Step 2 — F5.1 methodological memo.
    title = _truncate_for_title(
        lock_event.reason,
        prefix=_TITLE_PREFIX,
        max_reason=_MAX_REASON_IN_TITLE,
    )
    link = MemoLink(
        target_type="project",
        target_id=project_id,
        role=UNLOCK_LINK_ROLE,
    )
    provenance = {
        "source": UNLOCK_PROVENANCE_SOURCE,
        "codebook_lock_event_id": lock_event.id,
        "codebook_unlock_reason": _truncate_for_provenance(
            lock_event.reason
        ),
        "codebook_unlock_prior_stage": lock_event.prior_stage,
        "codebook_unlock_new_stage": lock_event.new_stage,
    }
    memo = Memo.new(
        project_id=project_id,
        type="methodological",
        title=title,
        body=lock_event.methodological_memo,
        author_coder_id=author_coder_id if author_coder_id else None,
        links=[link],
        provenance=provenance,
        now=now,
    )
    save_memo(projects_root, memo)

    # Step 3 — F9.1 event with cross-references.
    event = record_event(
        projects_root,
        project_id=project_id,
        action=EVENT_ACTION_UNLOCK,
        entity_type=EVENT_ENTITY_CODEBOOK,
        entity_id="",
        actor_coder_id=actor,
        before={"codebook_stage": lock_event.prior_stage},
        after={
            "codebook_stage": lock_event.new_stage,
            "lock_event_id": lock_event.id,
            "memo_id": memo.id,
        },
        notes=lock_event.reason,
        now=now,
    )

    return UnlockResult(
        project=project,
        lock_event=lock_event,
        memo=memo,
        event=event,
    )


# --------------------------------------------------------------------------- #
# Public API — read-side helpers
# --------------------------------------------------------------------------- #


def find_unlock_memos(
    projects_root: Path,
    project_id: str,
) -> list[Memo]:
    """Return all methodological memos that justify a codebook unlock.

    Filters for Memos with type ``methodological`` *and* a project
    link carrying role ``codebook_unlock``. Sorted oldest-first
    (matches :func:`scribe.memos.list_memos`'s ordering).

    Methodologically, this is the answer to "show me every time the
    codebook was reopened, and why" — a key piece of any methods
    section that has to defend a non-trivial coding workflow.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    candidates = list_memos(
        projects_root,
        project_id,
        type="methodological",
        target_type="project",
        target_id=project_id,
    )
    out: list[Memo] = []
    for m in candidates:
        for link in m.links:
            if (
                link.target_type == "project"
                and link.target_id == project_id
                and link.role == UNLOCK_LINK_ROLE
            ):
                out.append(m)
                break
    return out


def latest_unlock_memo(
    projects_root: Path,
    project_id: str,
) -> Memo | None:
    """Return the most recent codebook-unlock memo, or ``None``."""
    memos = find_unlock_memos(projects_root, project_id)
    return memos[-1] if memos else None


def find_codebook_lock_events(
    projects_root: Path,
    project_id: str,
) -> list[Event]:
    """Return F9.1 events recording a codebook lock or unlock.

    Walks the F9.1 events directory (``projects/<pid>/events/``) and
    keeps entries whose ``entity_type`` is ``codebook`` and whose
    ``action`` is one of ``lock`` / ``unlock``. Order matches
    :func:`scribe.event_log.list_events` (oldest-first by ``created_at``,
    then by ``id`` for stable tie-breaking).

    This is the audit-export-ready timeline of locking activity; F9.7
    will eventually consume it. Note that legacy projects may have a
    populated F2.4 lock log without F9.1 events (lock toggles emitted
    before F9.5 landed); the lock log remains the canonical fallback.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    out: list[Event] = []
    for ev in list_events(projects_root, project_id):
        if ev.entity_type != EVENT_ENTITY_CODEBOOK:
            continue
        if ev.action not in (EVENT_ACTION_LOCK, EVENT_ACTION_UNLOCK):
            continue
        out.append(ev)
    return out


def reconcile_unlock_artefacts(
    projects_root: Path,
    project_id: str,
) -> list[dict[str, str]]:
    """Cross-reference F2.4 lock-log entries against F9.5 sidecars.

    Returns a list of ``{lock_event_id, action, memo_id, event_id}``
    rows, one per F2.4 lock-log entry. ``memo_id`` and ``event_id``
    are populated when the corresponding F5.1 memo / F9.1 event is
    found (matched via the lock event id stored in their payloads),
    or empty strings otherwise.

    The use case is "this project pre-dates F9.5; have all unlocks
    been mirrored?" — and as the data shape an F9.7 audit export
    walks to render the per-toggle row.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    lock_log = read_lock_log(projects_root, project_id)
    events = find_codebook_lock_events(projects_root, project_id)
    memos = find_unlock_memos(projects_root, project_id)

    event_by_lock_id: dict[str, str] = {}
    for ev in events:
        after = ev.after or {}
        lock_id = str(after.get("lock_event_id") or "")
        if lock_id:
            event_by_lock_id.setdefault(lock_id, ev.id)

    memo_by_lock_id: dict[str, str] = {}
    for m in memos:
        lock_id = str(m.provenance.get("codebook_lock_event_id") or "")
        if lock_id:
            memo_by_lock_id.setdefault(lock_id, m.id)

    out: list[dict[str, str]] = []
    for ev in lock_log:
        out.append(
            {
                "lock_event_id": ev.id,
                "action": ev.action,
                "memo_id": memo_by_lock_id.get(ev.id, "")
                if ev.action == "unlock"
                else "",
                "event_id": event_by_lock_id.get(ev.id, ""),
            }
        )
    return out
