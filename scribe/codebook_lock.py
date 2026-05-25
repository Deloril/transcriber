"""Locked-codebook stage marker (F2.4).

Per PLANNING.md F2.4:

  > Locked-codebook stage marker. Toggle prevents new codes / edits but
  > allows applications. Unlock requires a methodological memo with a
  > reason.

Methodologically, locking a codebook is a **deliberate analytic move**:
the researcher has finished iterating on the codebook itself and is
about to run final coding (or release a frozen codebook to a co-coder
team for ICR). After that point, every accidental tweak corrodes the
audit trail — applications captured against "v3 of the definition" no
longer line up with "v3" if v3 silently got patched. Lock prevents
that; unlock requires a written justification so the methodological
break is visible in the audit log.

What lock blocks
----------------

Lock prevents writes to **codebook structure**: new codes, edits to
existing codes, lifecycle ops (rename / retire / merge / split /
re-parent). It does **not** block:

  * **Applications** of codes to source segments (F4.1+, future) —
    that is the whole point: a locked codebook is precisely what you
    want when you're applying codes for real.
  * **Memos** attached to codes (F5.x) — analytic note-taking continues
    even with a frozen codebook.
  * **Reading** anything in the project.

Unlock requires a methodological memo
-------------------------------------

The point of a lock is to make breaking it visible. ``unlock_codebook``
requires both a short ``reason`` (what changed) and a
``methodological_memo`` (the longer "why this is justified") — both
non-empty. The memo lands in the lock log as evidence; ``F5.4`` will
later be able to surface these alongside other memos.

On-disk layout
--------------

Lock state lives in two places:

  * ``project.codebook_stage = "locked"`` on the project's ``project.json``
    (round-trips through the F1.1 entity, no schema change).
  * ``projects/<project_id>/codebook_lock_log.jsonl`` — append-only
    JSONL log of every lock and unlock event, with reasons and memos.

Append-only mirrors the F1.4 sampling log and F2.2 code-version log;
F9.1's project-wide event log will eventually subsume this, but the
on-disk shape is forward-compatible.

Guarded helpers
---------------

This module exposes :func:`assert_codebook_unlocked` for callers that
want to enforce the lock at the right boundary. Two convenience
wrappers — :func:`guarded_save_code` and
:func:`guarded_save_code_with_version` — compose the guard with the
F2.1 / F2.2 primitives so HTTP and CLI layers can opt into enforcement
without re-implementing the check. The underlying primitives in
``scribe.codes`` and ``scribe.code_versions`` deliberately stay
lock-unaware so importers and migration scripts can write through them
without first having to negotiate lock state.

This module is stand-alone — no FastAPI, no engine imports — matching
the conventions of every other F-feature.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .codes import Code, save_code
from .code_versions import CodeVersion, save_code_with_version
from .projects import (
    CODEBOOK_STAGES,
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
    load_project,
    project_dir,
    save_project,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Lock-event IDs follow the same 12-char hex shape as every other id in
# Scribe; consistent with project / source / code / version IDs.
LOCK_EVENT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Two actions land in the lock log: locking and unlocking. The shape is
# the same in both directions; the action distinguishes them in reports.
LOCK_ACTIONS: tuple[str, ...] = ("lock", "unlock")

# Field length / cardinality limits. The methodological memo gets a
# generous cap because a real unlock justification ("we encountered an
# unexpected category in source 23 and need to revise the boundary of
# 'pacing'") is naturally a paragraph, not a sentence.
MIN_REASON_LEN = 1
MAX_REASON_LEN = 2000
MIN_METHODOLOGICAL_MEMO_LEN = 1
MAX_METHODOLOGICAL_MEMO_LEN = 8000

# The sentinel stage value used by the project entity for "locked".
# Imported from ``scribe.projects.CODEBOOK_STAGES`` to stay in sync.
LOCKED_STAGE = "locked"


class LockedCodebookError(ProjectValidationError):
    """Raised when a write to the codebook is attempted while it is locked.

    Subclasses ``ProjectValidationError`` so the existing HTTP error
    handling (which already maps ``ProjectValidationError`` to a 400
    in the F1.1 endpoints) can render this as a sensible error without
    new wiring. The message string is the canonical user-facing text.
    """


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class LockEvent:
    """One entry in the codebook lock log.

    A lock event records *who toggled the codebook lock when, and why*.
    For ``action='lock'``, ``reason`` is the human-readable rationale;
    ``methodological_memo`` is empty (locking is the safe direction —
    no extra friction). For ``action='unlock'``, both ``reason`` and
    ``methodological_memo`` are required and non-empty: an unlock is
    a breaking-of-the-seal moment and the memo is the audit trail.

    ``prior_stage`` records the project's ``codebook_stage`` *before*
    the toggle happened; ``new_stage`` is the stage set by this event.
    For an unlock, ``new_stage`` defaults to whatever the user picked
    (typically ``"theoretical"`` — the last working stage before the
    lock — but the F2.4 helper accepts any non-locked stage so a
    researcher unlocking after a long pause can land in any stage).
    """

    id: str
    project_id: str
    action: str
    reason: str
    created_at: str
    prior_stage: str = ""
    new_stage: str = ""
    methodological_memo: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        action: str,
        reason: str,
        prior_stage: str = "",
        new_stage: str = "",
        methodological_memo: str = "",
        event_id: str | None = None,
        now: str | None = None,
    ) -> "LockEvent":
        e = cls(
            id=event_id or new_lock_event_id(),
            project_id=project_id,
            action=action,
            reason=reason,
            created_at=now or utcnow_iso(),
            prior_stage=prior_stage,
            new_stage=new_stage,
            methodological_memo=methodological_memo,
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LockEvent":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "LockEvent payload must be an object"
            )
        for required in ("id", "project_id", "action", "created_at"):
            if required not in d:
                raise ProjectValidationError(
                    f"LockEvent payload missing required key: {required}"
                )
        e = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            action=str(d["action"]),
            reason=str(d.get("reason", "") or ""),
            created_at=str(d["created_at"]),
            prior_stage=str(d.get("prior_stage", "") or ""),
            new_stage=str(d.get("new_stage", "") or ""),
            methodological_memo=str(
                d.get("methodological_memo", "") or ""
            ),
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not LOCK_EVENT_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid lock-event id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if self.action not in LOCK_ACTIONS:
            raise ProjectValidationError(
                f"action must be one of {LOCK_ACTIONS}; got {self.action!r}"
            )
        if not self.created_at:
            raise ProjectValidationError("created_at is required")

        # Stage values, when supplied, must be drawn from the canonical
        # vocabulary so log readers don't have to second-guess them.
        if self.prior_stage and self.prior_stage not in CODEBOOK_STAGES:
            raise ProjectValidationError(
                f"prior_stage must be one of {CODEBOOK_STAGES}; "
                f"got {self.prior_stage!r}"
            )
        if self.new_stage and self.new_stage not in CODEBOOK_STAGES:
            raise ProjectValidationError(
                f"new_stage must be one of {CODEBOOK_STAGES}; "
                f"got {self.new_stage!r}"
            )

        reason = self.reason.strip()
        if len(reason) < MIN_REASON_LEN:
            raise ProjectValidationError(
                "reason must not be empty"
            )
        if len(reason) > MAX_REASON_LEN:
            raise ProjectValidationError(
                f"reason must be ≤ {MAX_REASON_LEN} chars"
            )
        # Persist trimmed so on-disk state is canonical.
        self.reason = reason

        memo = self.methodological_memo.strip()
        if self.action == "unlock":
            if len(memo) < MIN_METHODOLOGICAL_MEMO_LEN:
                raise ProjectValidationError(
                    "unlock requires a non-empty methodological_memo"
                )
        if len(memo) > MAX_METHODOLOGICAL_MEMO_LEN:
            raise ProjectValidationError(
                f"methodological_memo must be ≤ "
                f"{MAX_METHODOLOGICAL_MEMO_LEN} chars"
            )
        self.methodological_memo = memo


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_lock_event_id() -> str:
    """Mint a new 12-char hex lock-event id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence (append-only JSONL)
# --------------------------------------------------------------------------- #


def lock_log_path(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk path of a project's codebook lock log.

    Validates ``project_id`` to prevent traversal. Does not create the
    file — readers handle the "missing log" case as "empty log".
    """
    return project_dir(projects_root, project_id) / "codebook_lock_log.jsonl"


def append_lock_event(projects_root: Path, event: LockEvent) -> Path:
    """Append a lock event to the project's lock log.

    The parent ``projects/<id>`` directory must already exist (the
    project itself must have been saved). Mirrors
    :func:`scribe.sampling_log.append_sampling_entry`.
    """
    event.validate()
    parent = project_dir(projects_root, event.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before logging lock events."
        )
    target = lock_log_path(projects_root, event.project_id)
    line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)
    return target


def read_lock_log(
    projects_root: Path, project_id: str
) -> list[LockEvent]:
    """Read all lock events for a project, in stored (chronological) order.

    Skips lines that don't parse as a valid ``LockEvent`` so a single
    corrupt entry doesn't lock the user out of their own lock log.
    Empty / missing file returns ``[]``.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    path = lock_log_path(projects_root, project_id)
    if not path.exists():
        return []
    out: list[LockEvent] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            out.append(LockEvent.from_dict(payload))
        except ProjectValidationError:
            continue
    return out


def latest_lock_event(
    projects_root: Path, project_id: str
) -> LockEvent | None:
    """Return the most-recently-recorded lock event, or ``None``."""
    log = read_lock_log(projects_root, project_id)
    return log[-1] if log else None


def count_lock_events(
    projects_root: Path, project_id: str
) -> int:
    """Return the number of valid lock events for a project."""
    return len(read_lock_log(projects_root, project_id))


# --------------------------------------------------------------------------- #
# State accessors
# --------------------------------------------------------------------------- #


def is_codebook_locked(
    projects_root: Path, project_id: str
) -> bool:
    """Return True if the project's codebook is currently locked.

    Source of truth is the project's ``codebook_stage`` field. A missing
    project raises :class:`FileNotFoundError`; callers usually want that
    to surface as a 404, not silently fall through.
    """
    project = load_project(projects_root, project_id)
    return project.codebook_stage == LOCKED_STAGE


def assert_codebook_unlocked(
    projects_root: Path, project_id: str
) -> None:
    """Raise :class:`LockedCodebookError` if the codebook is locked.

    The canonical guard. Call this at the boundary of any operation
    that mutates codebook structure (new codes, code edits, lifecycle
    ops). Read paths and application paths must **not** call it.
    """
    if is_codebook_locked(projects_root, project_id):
        raise LockedCodebookError(
            f"Project {project_id!r} codebook is locked; "
            "unlock with a methodological memo before editing codes."
        )


# --------------------------------------------------------------------------- #
# Lock / unlock toggles
# --------------------------------------------------------------------------- #


def lock_codebook(
    projects_root: Path,
    project_id: str,
    *,
    reason: str,
    now: str | None = None,
) -> tuple[Project, LockEvent]:
    """Lock the project's codebook.

    Sets ``project.codebook_stage = "locked"`` and appends a
    ``LockEvent`` with ``action='lock'`` and the supplied ``reason``.
    Re-locking an already-locked codebook is rejected — the
    methodologically interesting move is the *unlock*, and silently
    accepting a no-op lock would clutter the audit log without changing
    state.

    A project's prior stage is recorded on the event so a future
    ``unlock_codebook`` (or a UI "restore previous stage" flow) can
    suggest the right resumption stage.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")

    if not reason or not reason.strip():
        raise ProjectValidationError(
            "lock_codebook requires a non-empty reason"
        )

    project = load_project(projects_root, project_id)
    if project.codebook_stage == LOCKED_STAGE:
        raise LockedCodebookError(
            f"Project {project_id!r} codebook is already locked"
        )

    prior_stage = project.codebook_stage
    project.apply_update({"codebook_stage": LOCKED_STAGE}, now=now)
    save_project(projects_root, project)

    event = LockEvent.new(
        project_id=project_id,
        action="lock",
        reason=reason,
        prior_stage=prior_stage,
        new_stage=LOCKED_STAGE,
        now=now,
    )
    append_lock_event(projects_root, event)
    return project, event


def unlock_codebook(
    projects_root: Path,
    project_id: str,
    *,
    reason: str,
    methodological_memo: str,
    new_stage: str | None = None,
    now: str | None = None,
) -> tuple[Project, LockEvent]:
    """Unlock the project's codebook.

    The "breaking the seal" operation. Both ``reason`` and
    ``methodological_memo`` are required and non-empty (per F2.4 spec).

    ``new_stage`` defaults to the most recent non-locked stage from the
    lock log (i.e. the stage the project was in just before the most
    recent lock), or ``"theoretical"`` when there is no prior history.
    Callers can pass an explicit stage; ``"locked"`` is rejected.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")

    if not reason or not reason.strip():
        raise ProjectValidationError(
            "unlock_codebook requires a non-empty reason"
        )
    if not methodological_memo or not methodological_memo.strip():
        raise ProjectValidationError(
            "unlock_codebook requires a non-empty methodological_memo"
        )

    project = load_project(projects_root, project_id)
    if project.codebook_stage != LOCKED_STAGE:
        raise ProjectValidationError(
            f"Project {project_id!r} codebook is not locked "
            f"(current stage: {project.codebook_stage!r})"
        )

    # Pick a default new_stage if the caller didn't specify one. We walk
    # back through the lock log looking for the most recent
    # ``prior_stage`` recorded on a lock event — that's the stage the
    # project was in just before the most recent lock toggle.
    resolved_stage = new_stage
    if not resolved_stage:
        for ev in reversed(read_lock_log(projects_root, project_id)):
            if ev.action == "lock" and ev.prior_stage and ev.prior_stage != LOCKED_STAGE:
                resolved_stage = ev.prior_stage
                break
    if not resolved_stage:
        resolved_stage = "theoretical"

    if resolved_stage == LOCKED_STAGE:
        raise ProjectValidationError(
            "unlock_codebook: new_stage cannot be 'locked'"
        )
    if resolved_stage not in CODEBOOK_STAGES:
        raise ProjectValidationError(
            f"unlock_codebook: new_stage must be one of {CODEBOOK_STAGES}; "
            f"got {resolved_stage!r}"
        )

    project.apply_update({"codebook_stage": resolved_stage}, now=now)
    save_project(projects_root, project)

    event = LockEvent.new(
        project_id=project_id,
        action="unlock",
        reason=reason,
        methodological_memo=methodological_memo,
        prior_stage=LOCKED_STAGE,
        new_stage=resolved_stage,
        now=now,
    )
    append_lock_event(projects_root, event)
    return project, event


# --------------------------------------------------------------------------- #
# Guarded helpers (lock-aware wrappers around F2.1 / F2.2 primitives)
# --------------------------------------------------------------------------- #


def guarded_save_code(projects_root: Path, code: Code) -> Path:
    """Like :func:`scribe.codes.save_code` but refuses to write when locked.

    The thin wrapper HTTP / CLI callers should prefer once F2.4 is
    wired into a UI. Importers that need to seed a codebook before the
    project is ever locked can keep using ``save_code`` directly.
    """
    assert_codebook_unlocked(projects_root, code.project_id)
    return save_code(projects_root, code)


def guarded_save_code_with_version(
    projects_root: Path,
    code: Code,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Path, CodeVersion | None]:
    """Lock-aware wrapper around :func:`save_code_with_version`."""
    assert_codebook_unlocked(projects_root, code.project_id)
    return save_code_with_version(
        projects_root, code, change_note=change_note, now=now
    )
