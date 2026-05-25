"""Append-only event log for the academic-coding workflow (F9.1).

Per PLANNING.md F9.1:

  > Append-only event log. Every operation becomes an event with
  > timestamp, actor, and full payload diff. Never deletable.

This is the **generic** event log — distinct from F8.9's AI event log:

  * F8.9 records AI invocations (request / decision / application /
    error) at ``projects/<pid>/ai_events/<eid>.json``.
  * F9.1 records *every* operation on the project — code created,
    application added, memo edited, codebook locked, source attached,
    snapshot taken. Lives at ``projects/<pid>/events/<eid>.json``.

Both are append-only; both use 12-char hex ids; both are JSON
sidecars. F9.1 is the foundation the rest of the F9 trust-and-
reproducibility stack builds on:

  * F9.2 — code definition versioning (events store before/after
    snapshots so we can reconstruct any code's state at a moment).
  * F9.3 — named codebook snapshots (snapshot ids reference the event
    that produced them).
  * F9.6 — AI invocation log (the AI event log is replayed into this
    one for unified audit trail export).
  * F9.7 — audit trail export (chronological Markdown / Word).
  * F9.8 — time-travel view (replay events up to a timestamp).

Boundaries
----------

* **No HTTP / FastAPI surface here.** F9.1 is the data model + writer
  + reader. Routes (``/api/projects/<id>/events``) are added by a
  later iteration if needed.
* **No automatic emit from existing F1–F6 entities.** F9.1 ships the
  schema + helpers; wiring each existing mutator to call
  :func:`record_event` is a follow-on. Same staged approach as F8.9.
  Callers can use the convenience emitters
  :func:`record_create` / :func:`record_update` / :func:`record_delete`
  to capture before/after state with minimal ceremony.
* **Stand-alone, pure Python.** No FastAPI, no engine imports. Mirrors
  :mod:`scribe.ai_provenance`, :mod:`scribe.applications`, etc.

On-disk layout
--------------

::

    projects/<project_id>/
      events/
        <event_id>.json     # one file per event; never modified

Files are named by event id (12-char hex). Listing relies on the
``created_at`` field inside each file rather than mtime, so backups
that re-touch files don't reorder the audit trail.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .coders import CODER_ID_RE
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Event ids share the 12-char hex shape used everywhere in Scribe.
EVENT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory for the generic event log. Sibling of the AI
# event log directory (``ai_events``) so they don't collide.
EVENTS_DIRNAME = "events"


# Action vocabulary — what the operation *did*. Closed set with
# ``other`` for forward-compat (a new feature can ship without
# touching this module).
EVENT_ACTION_CREATE = "create"
EVENT_ACTION_UPDATE = "update"
EVENT_ACTION_DELETE = "delete"
EVENT_ACTION_RENAME = "rename"
EVENT_ACTION_MERGE = "merge"
EVENT_ACTION_SPLIT = "split"
EVENT_ACTION_RETIRE = "retire"
EVENT_ACTION_PROMOTE = "promote"
EVENT_ACTION_LOCK = "lock"
EVENT_ACTION_UNLOCK = "unlock"
EVENT_ACTION_SNAPSHOT = "snapshot"
EVENT_ACTION_IMPORT = "import"
EVENT_ACTION_EXPORT = "export"
EVENT_ACTION_OTHER = "other"
EVENT_ACTIONS: tuple[str, ...] = (
    EVENT_ACTION_CREATE,
    EVENT_ACTION_UPDATE,
    EVENT_ACTION_DELETE,
    EVENT_ACTION_RENAME,
    EVENT_ACTION_MERGE,
    EVENT_ACTION_SPLIT,
    EVENT_ACTION_RETIRE,
    EVENT_ACTION_PROMOTE,
    EVENT_ACTION_LOCK,
    EVENT_ACTION_UNLOCK,
    EVENT_ACTION_SNAPSHOT,
    EVENT_ACTION_IMPORT,
    EVENT_ACTION_EXPORT,
    EVENT_ACTION_OTHER,
)


# Entity type vocabulary — *what kind of thing* the event is about.
# Closed set + ``other`` for forward-compat. Mirrors the F1–F8 modules.
EVENT_ENTITY_PROJECT = "project"
EVENT_ENTITY_SOURCE = "source"
EVENT_ENTITY_PARTICIPANT = "participant"
EVENT_ENTITY_CODE = "code"
EVENT_ENTITY_CODE_VERSION = "code_version"
EVENT_ENTITY_APPLICATION = "application"
EVENT_ENTITY_MEMO = "memo"
EVENT_ENTITY_CODER = "coder"
EVENT_ENTITY_SAVED_QUERY = "saved_query"
EVENT_ENTITY_SAMPLING_LOG = "sampling_log"
EVENT_ENTITY_CODEBOOK = "codebook"
EVENT_ENTITY_SNAPSHOT = "snapshot"
EVENT_ENTITY_OTHER = "other"
EVENT_ENTITY_TYPES: tuple[str, ...] = (
    EVENT_ENTITY_PROJECT,
    EVENT_ENTITY_SOURCE,
    EVENT_ENTITY_PARTICIPANT,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_CODE_VERSION,
    EVENT_ENTITY_APPLICATION,
    EVENT_ENTITY_MEMO,
    EVENT_ENTITY_CODER,
    EVENT_ENTITY_SAVED_QUERY,
    EVENT_ENTITY_SAMPLING_LOG,
    EVENT_ENTITY_CODEBOOK,
    EVENT_ENTITY_SNAPSHOT,
    EVENT_ENTITY_OTHER,
)


# Diff op vocabulary. ``added`` / ``removed`` / ``changed`` cover the
# top-level field changes between two payload dicts.
DIFF_OP_ADDED = "added"
DIFF_OP_REMOVED = "removed"
DIFF_OP_CHANGED = "changed"
DIFF_OPS: tuple[str, ...] = (
    DIFF_OP_ADDED,
    DIFF_OP_REMOVED,
    DIFF_OP_CHANGED,
)


# Field-length / cardinality caps. Generous, but bounded so a stray
# upstream bug can't write a 50 MB event record. The before/after
# snapshots are bigger than the AI-event payload cap because we
# routinely store whole entity records (a Code with exemplars or a
# Source with full attribute schema can run several KiB).
MAX_NOTES_LEN = 4000
MAX_ENTITY_TYPE_LEN = 64       # belt-and-braces over the closed set
MAX_PAYLOAD_KEYS = 256         # one per top-level field of a Code etc.
MAX_PAYLOAD_KEY_LEN = 128
MAX_PAYLOAD_STRING_LEN = 16_000
MAX_PAYLOAD_LIST_LEN = 256
MAX_PAYLOAD_DEPTH = 4
MAX_PAYLOAD_BYTES = 256 * 1024  # 256 KiB serialised before+after combined
MAX_DIFF_ENTRIES = 256


_PAYLOAD_KEY_RE = re.compile(r"^[A-Za-z_][\w\-.]{0,127}$")


# --------------------------------------------------------------------------- #
# Event dataclass
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    """One entry in a project's append-only event log (F9.1).

    Fields
    ------
    id
        12-char hex event id. Mint with :func:`new_event_id`.
    project_id
        12-char hex project id this event belongs to.
    created_at
        ISO-8601 UTC timestamp; set at construction time.
    actor_coder_id
        12-char hex coder id of the human who triggered the operation,
        or empty string for system / anonymous events. Same shape as
        :class:`scribe.coders.Coder.id`.
    action
        One of :data:`EVENT_ACTIONS`. Required.
    entity_type
        One of :data:`EVENT_ENTITY_TYPES`. Required. Says *what kind of
        thing* the event is about.
    entity_id
        Optional 12-char hex id of the affected entity. Empty for
        events that aren't about a specific record (e.g. a project-wide
        ``snapshot`` action). Note: not validated against any specific
        entity table — the event log doesn't enforce referential
        integrity, since deletes are real and we still need to be able
        to retain the audit trail.
    before
        Optional dict snapshot of the entity *before* the operation.
        ``None`` for ``create`` events.
    after
        Optional dict snapshot of the entity *after* the operation.
        ``None`` for ``delete`` events.
    diff
        Optional structured diff between ``before`` and ``after``.
        Computed by :func:`compute_diff` when not supplied. Always a
        list of ``{path, op, before, after}`` rows.
    notes
        Free-form short text (e.g. the methodological reason supplied
        when unlocking a codebook). Bounded by :data:`MAX_NOTES_LEN`.
    """

    id: str
    project_id: str
    action: str
    entity_type: str
    entity_id: str = ""
    actor_coder_id: str = ""
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    diff: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        action: str,
        entity_type: str,
        entity_id: str = "",
        actor_coder_id: str = "",
        before: Mapping[str, Any] | None = None,
        after: Mapping[str, Any] | None = None,
        diff: Iterable[Mapping[str, Any]] | None = None,
        notes: str = "",
        event_id: str | None = None,
        now: str | None = None,
    ) -> "Event":
        """Construct and validate a new :class:`Event`.

        ``diff`` is *not* auto-computed here — callers may want to
        supply a pre-computed diff (from a higher-level API) or pass
        ``before`` / ``after`` and let :meth:`with_computed_diff` fill
        in the gap. The convenience emitters below
        (:func:`record_create` / :func:`record_update` / :func:`record_delete`)
        do the right thing for the common cases.
        """
        b = _normalise_payload(before, label="before")
        a = _normalise_payload(after, label="after")
        d = _normalise_diff(diff)
        ts = now or utcnow_iso()
        ev = cls(
            id=event_id or new_event_id(),
            project_id=project_id,
            action=str(action),
            entity_type=str(entity_type),
            entity_id=str(entity_id or ""),
            actor_coder_id=str(actor_coder_id or ""),
            before=b,
            after=a,
            diff=d,
            notes=str(notes or ""),
            created_at=ts,
        )
        ev.validate()
        return ev

    def with_computed_diff(self) -> "Event":
        """Return a copy with ``diff`` filled from ``before`` / ``after``.

        If a diff is already set, it's preserved (callers may want a
        custom domain-specific diff that's richer than field-level).
        """
        if self.diff:
            return self
        d = compute_diff(self.before, self.after)
        return Event(
            id=self.id,
            project_id=self.project_id,
            action=self.action,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            actor_coder_id=self.actor_coder_id,
            before=self.before,
            after=self.after,
            diff=d,
            notes=self.notes,
            created_at=self.created_at,
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "actor_coder_id": self.actor_coder_id,
            "before": _clone_payload(self.before),
            "after": _clone_payload(self.after),
            "diff": [dict(row) for row in self.diff],
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Event":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("Event payload must be an object")
        for required in ("id", "project_id", "action", "entity_type"):
            if required not in d:
                raise ProjectValidationError(
                    f"Event payload missing required key: {required}"
                )
        ev = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            action=str(d.get("action", "") or ""),
            entity_type=str(d.get("entity_type", "") or ""),
            entity_id=str(d.get("entity_id", "") or ""),
            actor_coder_id=str(d.get("actor_coder_id", "") or ""),
            before=_normalise_payload(d.get("before"), label="before"),
            after=_normalise_payload(d.get("after"), label="after"),
            diff=_normalise_diff(d.get("diff")),
            notes=str(d.get("notes", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
        )
        ev.validate()
        return ev

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not EVENT_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid event id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if self.action not in EVENT_ACTIONS:
            raise ProjectValidationError(
                f"Event.action must be one of {EVENT_ACTIONS}; "
                f"got {self.action!r}"
            )
        if self.entity_type not in EVENT_ENTITY_TYPES:
            raise ProjectValidationError(
                f"Event.entity_type must be one of {EVENT_ENTITY_TYPES}; "
                f"got {self.entity_type!r}"
            )
        if self.entity_id and not EVENT_ID_RE.match(self.entity_id):
            raise ProjectValidationError(
                f"Event.entity_id must be 12-char hex or empty; "
                f"got {self.entity_id!r}"
            )
        if self.actor_coder_id and not CODER_ID_RE.match(self.actor_coder_id):
            raise ProjectValidationError(
                "Event.actor_coder_id must be 12-char hex or empty; "
                f"got {self.actor_coder_id!r}"
            )
        if not isinstance(self.notes, str):
            raise ProjectValidationError("Event.notes must be a string")
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"Event.notes exceeds {MAX_NOTES_LEN} characters"
            )
        if self.created_at and not isinstance(self.created_at, str):
            raise ProjectValidationError("Event.created_at must be a string")
        # Re-normalise after construction so callers who poked the
        # in-memory value still see a validated payload.
        self.before = _normalise_payload(self.before, label="before")
        self.after = _normalise_payload(self.after, label="after")
        self.diff = _normalise_diff(self.diff)
        # Combined size ceiling so a runaway dict doesn't blow up
        # the audit trail.
        encoded = json.dumps(
            {
                "before": self.before,
                "after": self.after,
                "diff": self.diff,
            },
            ensure_ascii=False,
        )
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ProjectValidationError(
                f"Event payload exceeds {MAX_PAYLOAD_BYTES} bytes "
                "(before + after + diff combined)"
            )


# --------------------------------------------------------------------------- #
# Payload normalisation
# --------------------------------------------------------------------------- #


def _normalise_payload(
    raw: Any, *, label: str
) -> dict[str, Any] | None:
    """Coerce a payload value into a flat dict (or None).

    Accepts ``None`` (no payload), ``Mapping``s (recursively bounded),
    and rejects anything else. Returns a fresh dict each call so
    callers can't mutate the on-disk payload by holding a reference.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProjectValidationError(
            f"Event.{label} must be an object or null; got {type(raw).__name__}"
        )
    return _validate_payload_dict(raw, depth=0, path=label)


def _validate_payload_dict(
    raw: Mapping[str, Any], *, depth: int, path: str
) -> dict[str, Any]:
    if depth > MAX_PAYLOAD_DEPTH:
        raise ProjectValidationError(
            f"Event payload at {path}: nested too deep "
            f"(>{MAX_PAYLOAD_DEPTH} levels)"
        )
    if len(raw) > MAX_PAYLOAD_KEYS:
        raise ProjectValidationError(
            f"Event payload at {path}: more than {MAX_PAYLOAD_KEYS} keys"
        )
    out: dict[str, Any] = {}
    for raw_k, raw_v in raw.items():
        k = str(raw_k)
        if not _PAYLOAD_KEY_RE.match(k):
            raise ProjectValidationError(
                f"Event payload key {path}.{k!r} must match "
                "letters/digits/underscore/hyphen/dot, "
                f"≤{MAX_PAYLOAD_KEY_LEN} chars"
            )
        out[k] = _validate_payload_value(
            raw_v, depth=depth + 1, path=f"{path}.{k}"
        )
    return out


def _validate_payload_value(v: Any, *, depth: int, path: str) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        if len(v) > MAX_PAYLOAD_STRING_LEN:
            raise ProjectValidationError(
                f"Event payload at {path}: string exceeds "
                f"{MAX_PAYLOAD_STRING_LEN} chars"
            )
        return v
    if isinstance(v, Mapping):
        return _validate_payload_dict(v, depth=depth, path=path)
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_PAYLOAD_LIST_LEN:
            raise ProjectValidationError(
                f"Event payload at {path}: list exceeds "
                f"{MAX_PAYLOAD_LIST_LEN} entries"
            )
        return [
            _validate_payload_value(item, depth=depth + 1, path=f"{path}[{i}]")
            for i, item in enumerate(v)
        ]
    raise ProjectValidationError(
        f"Event payload at {path}: unsupported type {type(v).__name__}"
    )


def _clone_payload(p: dict[str, Any] | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return json.loads(json.dumps(p, ensure_ascii=False))


def _normalise_diff(
    raw: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Coerce a diff iterable into a normalised list of rows.

    Each row must be a Mapping with at minimum a ``path`` and ``op``.
    ``before`` / ``after`` are optional and validated as payload values.
    """
    if raw is None:
        return []
    out: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        # Allow any non-string iterable.
        try:
            raw = list(raw)
        except TypeError as e:
            raise ProjectValidationError(
                "Event.diff must be a list of {path, op, before, after}"
            ) from e
    if len(raw) > MAX_DIFF_ENTRIES:
        raise ProjectValidationError(
            f"Event.diff exceeds {MAX_DIFF_ENTRIES} entries"
        )
    for i, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ProjectValidationError(
                f"Event.diff[{i}] must be an object; got {type(row).__name__}"
            )
        path = str(row.get("path", "") or "")
        if not path:
            raise ProjectValidationError(
                f"Event.diff[{i}].path must be a non-empty string"
            )
        if len(path) > MAX_PAYLOAD_KEY_LEN:
            raise ProjectValidationError(
                f"Event.diff[{i}].path exceeds {MAX_PAYLOAD_KEY_LEN} chars"
            )
        op = str(row.get("op", "") or "")
        if op not in DIFF_OPS:
            raise ProjectValidationError(
                f"Event.diff[{i}].op must be one of {DIFF_OPS}; got {op!r}"
            )
        entry: dict[str, Any] = {"path": path, "op": op}
        if "before" in row:
            entry["before"] = _validate_payload_value(
                row["before"], depth=0, path=f"diff[{i}].before"
            )
        if "after" in row:
            entry["after"] = _validate_payload_value(
                row["after"], depth=0, path=f"diff[{i}].after"
            )
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# compute_diff — structured top-level diff between two payload dicts
# --------------------------------------------------------------------------- #


def compute_diff(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return a structured top-level diff between two payload dicts.

    The diff is a list of ``{path, op, before, after}`` rows, one per
    changed top-level field, with these semantics:

      * ``op="added"`` — the key exists only in ``after`` (and the
        value is non-null). Emits ``after`` only.
      * ``op="removed"`` — the key exists only in ``before`` (and the
        value is non-null). Emits ``before`` only.
      * ``op="changed"`` — the key exists in both and the values
        differ by deep-equality. Emits both ``before`` and ``after``.

    Equal values are skipped (no row). Both inputs may be ``None``;
    if so, ``added`` rows are emitted for every key in the non-None
    side. The diff is intentionally *shallow* (top-level keys only) —
    if a caller wants nested diffs they can compute their own and
    pass it as the ``diff`` argument to :meth:`Event.new`. Pure helper;
    no I/O.
    """
    if before is not None and not isinstance(before, Mapping):
        raise ProjectValidationError(
            "compute_diff requires Mapping or None for both sides"
        )
    if after is not None and not isinstance(after, Mapping):
        raise ProjectValidationError(
            "compute_diff requires Mapping or None for both sides"
        )
    b: Mapping[str, Any] = before if before is not None else {}
    a: Mapping[str, Any] = after if after is not None else {}
    out: list[dict[str, Any]] = []
    keys = sorted(set(b.keys()) | set(a.keys()))
    for k in keys:
        in_b = k in b
        in_a = k in a
        bv = b.get(k)
        av = a.get(k)
        if in_b and in_a:
            if bv == av:
                continue
            out.append(
                {"path": str(k), "op": DIFF_OP_CHANGED, "before": bv, "after": av}
            )
        elif in_a:
            out.append({"path": str(k), "op": DIFF_OP_ADDED, "after": av})
        else:
            out.append({"path": str(k), "op": DIFF_OP_REMOVED, "before": bv})
    return out


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_event_id() -> str:
    """Mint a new 12-char hex event id (matches Scribe's standard shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def events_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's event log."""
    return project_dir(projects_root, project_id) / EVENTS_DIRNAME


def event_state_path(
    projects_root: Path, project_id: str, event_id: str
) -> Path:
    """Return the path for a single event JSON file."""
    if not EVENT_ID_RE.match(event_id):
        raise ProjectValidationError(f"Invalid event id: {event_id!r}")
    return events_dir(projects_root, project_id) / f"{event_id}.json"


def save_event(projects_root: Path, event: Event) -> Path:
    """Persist an event atomically; refuses to overwrite an existing id.

    Events are append-only — once written, the on-disk file *is* the
    audit trail. Re-saving an existing id raises :class:`FileExistsError`;
    callers should mint a fresh id with :func:`new_event_id` for every
    new operation. There is no ``delete_event`` helper, by design:
    deleting events would defeat the audit-trail purpose.
    """
    event.validate()
    parent = project_dir(projects_root, event.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving events."
        )
    ed = events_dir(projects_root, event.project_id)
    ed.mkdir(parents=True, exist_ok=True)
    target = event_state_path(projects_root, event.project_id, event.id)
    if target.exists():
        raise FileExistsError(
            f"Event {event.id} already exists; events are append-only"
        )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(event.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_event(
    projects_root: Path, project_id: str, event_id: str
) -> Event:
    """Load an event by id. Raises ``FileNotFoundError`` if missing."""
    p = event_state_path(projects_root, project_id, event_id)
    if not p.exists():
        raise FileNotFoundError(f"No event at {p}")
    return Event.from_dict(json.loads(p.read_text()))


def list_events(
    projects_root: Path,
    project_id: str,
    *,
    action: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    actor_coder_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[Event]:
    """List events in a project, optionally filtered.

    Filters AND-combine. ``since`` / ``until`` are inclusive ISO-8601
    strings compared lexically (which works for the Z-suffixed UTC
    timestamps Scribe uses everywhere). Skips files that don't parse
    so a single corrupt event doesn't break the view (matches the rest
    of the F-feature stack). Sorted by ``created_at`` ascending then
    by ``id`` for stability — natural reading order is the order the
    operations were emitted.
    """
    if action is not None and action not in EVENT_ACTIONS:
        raise ProjectValidationError(f"Invalid action filter: {action!r}")
    if entity_type is not None and entity_type not in EVENT_ENTITY_TYPES:
        raise ProjectValidationError(
            f"Invalid entity_type filter: {entity_type!r}"
        )
    if entity_id is not None and entity_id and not EVENT_ID_RE.match(entity_id):
        raise ProjectValidationError(f"Invalid entity_id filter: {entity_id!r}")
    if actor_coder_id is not None and actor_coder_id and not CODER_ID_RE.match(
        actor_coder_id
    ):
        raise ProjectValidationError(
            f"Invalid actor_coder_id filter: {actor_coder_id!r}"
        )
    ed = events_dir(projects_root, project_id)
    if not ed.exists():
        return []
    out: list[Event] = []
    for f in sorted(ed.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        eid = f.stem
        if not EVENT_ID_RE.match(eid):
            continue
        try:
            ev = Event.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if action is not None and ev.action != action:
            continue
        if entity_type is not None and ev.entity_type != entity_type:
            continue
        if entity_id is not None and ev.entity_id != entity_id:
            continue
        if actor_coder_id is not None and ev.actor_coder_id != actor_coder_id:
            continue
        if since is not None and ev.created_at < since:
            continue
        if until is not None and ev.created_at > until:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.created_at, e.id))
    return out


def count_events(projects_root: Path, project_id: str) -> int:
    """Return the number of well-formed events in a project's log.

    Cheap counter — useful for the UI badge ("12 audit-log entries")
    without paying the parse cost of :func:`list_events`.
    """
    ed = events_dir(projects_root, project_id)
    if not ed.exists():
        return 0
    n = 0
    for f in ed.iterdir():
        if not f.is_file():
            continue
        if not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        if EVENT_ID_RE.match(f.stem):
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Convenience emitters — capture the common shapes of operations.
# --------------------------------------------------------------------------- #


def record_event(
    projects_root: Path,
    *,
    project_id: str,
    action: str,
    entity_type: str,
    entity_id: str = "",
    actor_coder_id: str = "",
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    diff: Iterable[Mapping[str, Any]] | None = None,
    notes: str = "",
    event_id: str | None = None,
    now: str | None = None,
    auto_diff: bool = True,
) -> Event:
    """Construct, persist and return an :class:`Event` in one call.

    When ``diff`` is not supplied and ``auto_diff`` is true, a top-level
    diff is computed from ``before`` / ``after`` via :func:`compute_diff`.
    Set ``auto_diff=False`` to skip — useful when the change is large
    and the caller doesn't want the diff column blown out.
    """
    if diff is None and auto_diff:
        diff = compute_diff(before, after)
    ev = Event.new(
        project_id=project_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_coder_id=actor_coder_id,
        before=before,
        after=after,
        diff=diff,
        notes=notes,
        event_id=event_id,
        now=now,
    )
    save_event(projects_root, ev)
    return ev


def record_create(
    projects_root: Path,
    *,
    project_id: str,
    entity_type: str,
    entity_id: str = "",
    after: Mapping[str, Any],
    actor_coder_id: str = "",
    notes: str = "",
    now: str | None = None,
) -> Event:
    """Record a ``create`` event. ``after`` is the new entity payload."""
    return record_event(
        projects_root,
        project_id=project_id,
        action=EVENT_ACTION_CREATE,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_coder_id=actor_coder_id,
        before=None,
        after=after,
        notes=notes,
        now=now,
    )


def record_update(
    projects_root: Path,
    *,
    project_id: str,
    entity_type: str,
    entity_id: str = "",
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    actor_coder_id: str = "",
    notes: str = "",
    now: str | None = None,
) -> Event:
    """Record an ``update`` event with both states + computed diff."""
    return record_event(
        projects_root,
        project_id=project_id,
        action=EVENT_ACTION_UPDATE,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_coder_id=actor_coder_id,
        before=before,
        after=after,
        notes=notes,
        now=now,
    )


def record_delete(
    projects_root: Path,
    *,
    project_id: str,
    entity_type: str,
    entity_id: str = "",
    before: Mapping[str, Any],
    actor_coder_id: str = "",
    notes: str = "",
    now: str | None = None,
) -> Event:
    """Record a ``delete`` event. ``before`` is the entity's last state."""
    return record_event(
        projects_root,
        project_id=project_id,
        action=EVENT_ACTION_DELETE,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_coder_id=actor_coder_id,
        before=before,
        after=None,
        notes=notes,
        now=now,
    )
