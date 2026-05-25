"""Code revision history (F2.2).

Per PLANNING.md F2.2:

  > Code revision history. Every definition edit creates a new immutable
  > version; applications record version-at-apply.

A *code* (F2.1) is the labelled analytic concept at the heart of
qualitative coding. Its **definition evolves**: line-by-line names get
sharpened in focused coding, exclusions emerge once the boundary is
tested, exemplars are added as more transcripts come in. Methodological
transparency requires that an *application* of a code (F4.1, planned)
can be reported against the **definition that was in force at the time
the application was made** — not whatever the definition has since
become. That is the audit-trail role this module exists to play.

Design choices
--------------

* **Append-only JSON-Lines log per code.** Same shape as the F1.4
  sampling log: one file per code, one line per immutable version, no
  in-place edits. Corrupt lines are silently skipped on read so a single
  bad entry can't lock the user out of their version history.

* **Snapshot the entire serialisable code state.** Forward-compatibility
  matters more than disk-byte frugality here. If F2.3 (lifecycle ops:
  merge / split / rename / retire) adds fields, old snapshots still
  parse; new snapshots capture the new fields automatically. Disk cost
  is bounded by the number of definition edits, which a typical
  research project counts in the hundreds, not millions.

* **Definition vs. metadata.** Not every field on a Code is part of its
  *definition*. ``stage``, ``colour``, ``status`` and ``provenance`` are
  organisational metadata that shouldn't trigger a new version when
  toggled — promoting a code from ``draft`` to ``active`` doesn't change
  its analytic meaning. The closed set of definition-bearing fields is
  ``DEFINITION_FIELDS`` below; only changes to those fields cause
  ``save_code_with_version`` to record a new revision.

* **Version IDs are 12-char hex.** Mirrors project / source / code /
  participant / job IDs so F4.1's ``definition_version_id_at_apply``
  has the same shape as every other foreign key in the system.

* **One-based version numbering.** The first save records ``version=1``;
  every subsequent definition change increments. Convenient for UI
  ("v3"), unambiguous in reports.

On-disk layout::

    projects/<project_id>/code_versions/<code_id>.jsonl

Lives next to ``codes/<code_id>.json`` (F2.1) inside the project
directory, so the existing ``delete_project`` cascade picks the version
log up for free, and so F1.5's bundle exporter can include version logs
later by adding one component path — no migration needed.

This module is stand-alone (no FastAPI, no engine imports), matching
the conventions of F1.* and F2.1.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .codes import (
    CODE_ID_RE,
    Code,
    code_state_path,
    save_code,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Code-version IDs follow the same 12-char hex shape as every other id
# in Scribe; F4.1's ``definition_version_id_at_apply`` will reference
# this directly.
CODE_VERSION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# The closed set of fields that constitute a code's *definition*. Only
# changes to one or more of these trigger a new version when
# ``save_code_with_version`` is used. Order is documentation, not
# semantics; the comparison is a per-field equality check.
#
# Why each field is included:
#   - ``name`` — renaming a code changes its analytic meaning.
#   - ``definition`` — the obvious one.
#   - ``inclusion_criteria`` / ``exclusion_criteria`` — the *boundary*
#     of the category; methodologically central in Charmaz.
#   - ``exemplars`` — the worked examples that ground the definition;
#     adding/removing an exemplar shifts how a coder applies the code.
#   - ``theoretical_memo`` — the analytic notes attached to the code;
#     part of the audit-trail story (F9.x).
#   - ``parent_code_id`` — moving a code in the hierarchy changes its
#     meaning by changing what it's a *kind of* (or a sub-category of).
#   - ``related_codes`` — typed links express "what this code stands
#     in relation to"; changing them changes what the code means.
#
# Deliberately *excluded* (treated as metadata, no version on change):
#   - ``stage`` — analytic-stage tag, not a definitional change.
#   - ``colour`` — purely cosmetic.
#   - ``status`` — lifecycle (active / draft / retired) — F2.3 will
#     extend this; status changes are recorded in the F9.1 event log,
#     not as definition revisions.
#   - ``provenance`` — origin metadata (who minted the code, AI model
#     id, etc.) — should not recur on every edit.
#   - ``id`` / ``project_id`` / ``created_at`` / ``modified_at`` —
#     entity-managed fields, never part of the definition.
DEFINITION_FIELDS: tuple[str, ...] = (
    "name",
    "definition",
    "inclusion_criteria",
    "exclusion_criteria",
    "exemplars",
    "theoretical_memo",
    "parent_code_id",
    "related_codes",
)

# Field length cap on the optional human-readable change note. Generous
# — researchers may write a sentence or two — but bounded so a runaway
# script can't write a 50 MB versions.jsonl.
MAX_CHANGE_NOTE_LEN = 4000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class CodeVersion:
    """One immutable version snapshot of a Code's definition.

    A version captures the *full* serialised Code state at a moment in
    time, plus a 1-based ``version`` ordinal and an optional human
    ``change_note``. Snapshots are written append-only; the entity has
    no ``apply_update`` because there is nothing to update — making a
    new version is the only way to change the audit log.

    ``snapshot`` is the dictionary form of ``Code.to_dict()`` at the
    moment ``record_code_version`` was called. Storing the whole dict
    (rather than a delta) keeps readers simple and means F2.3's future
    field additions parse cleanly out of older logs.
    """

    id: str
    code_id: str
    project_id: str
    version: int
    created_at: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    change_note: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        code: Code,
        version: int,
        change_note: str = "",
        version_id: str | None = None,
        now: str | None = None,
    ) -> "CodeVersion":
        """Build a fresh CodeVersion from a Code's current state.

        The caller supplies the ordinal ``version`` (typically the
        return of ``len(read_code_versions(...)) + 1``) so this class
        stays a pure value type and tests stay deterministic.
        """
        # Snapshot via to_dict() to capture exactly what would land on
        # disk if ``save_code`` were called now. Defensive copy ensures
        # later mutation of ``code`` doesn't bleed into the snapshot.
        snapshot = json.loads(json.dumps(code.to_dict()))
        v = cls(
            id=version_id or new_code_version_id(),
            code_id=code.id,
            project_id=code.project_id,
            version=version,
            created_at=now or utcnow_iso(),
            snapshot=snapshot,
            change_note=change_note,
        )
        v.validate()
        return v

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodeVersion":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "CodeVersion payload must be an object"
            )
        for required in ("id", "code_id", "project_id", "version", "created_at"):
            if required not in d:
                raise ProjectValidationError(
                    f"CodeVersion payload missing required key: {required}"
                )
        # ``version`` may arrive as a JSON number or string; coerce
        # defensively but raise on garbage.
        try:
            version_num = int(d["version"])
        except (TypeError, ValueError):
            raise ProjectValidationError(
                f"CodeVersion.version must be an integer; got {d['version']!r}"
            )
        snapshot = d.get("snapshot") or {}
        if not isinstance(snapshot, dict):
            raise ProjectValidationError(
                "CodeVersion.snapshot must be an object"
            )
        v = cls(
            id=str(d["id"]),
            code_id=str(d["code_id"]),
            project_id=str(d["project_id"]),
            version=version_num,
            created_at=str(d["created_at"]),
            snapshot=snapshot,
            change_note=str(d.get("change_note", "") or ""),
        )
        v.validate()
        return v

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not CODE_VERSION_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid code-version id: {self.id!r}"
            )
        if not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"Invalid code id: {self.code_id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not isinstance(self.version, int) or self.version < 1:
            raise ProjectValidationError(
                f"version must be a positive integer; got {self.version!r}"
            )
        if not self.created_at:
            raise ProjectValidationError("created_at is required")
        if not isinstance(self.snapshot, dict):
            raise ProjectValidationError("snapshot must be an object")
        # Cross-check: snapshot's ids should agree with the version's
        # ids when present. We tolerate missing keys (older logs may
        # predate stricter rules), but if both sides specify, they must
        # match — otherwise a stale or hand-edited line could quietly
        # mis-attribute a snapshot.
        snap_code = self.snapshot.get("id")
        if snap_code and snap_code != self.code_id:
            raise ProjectValidationError(
                f"snapshot.id {snap_code!r} does not match "
                f"code_id {self.code_id!r}"
            )
        snap_project = self.snapshot.get("project_id")
        if snap_project and snap_project != self.project_id:
            raise ProjectValidationError(
                f"snapshot.project_id {snap_project!r} does not match "
                f"project_id {self.project_id!r}"
            )
        if len(self.change_note) > MAX_CHANGE_NOTE_LEN:
            raise ProjectValidationError(
                f"change_note must be ≤ {MAX_CHANGE_NOTE_LEN} chars"
            )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_code_version_id() -> str:
    """Mint a new 12-char hex code-version id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Helpers — definition signatures
# --------------------------------------------------------------------------- #


def definition_signature(code_or_dict: Code | dict[str, Any]) -> dict[str, Any]:
    """Project a Code (or its to_dict()) onto the definition fields only.

    The return is a JSON-normalised dict of just ``DEFINITION_FIELDS``,
    suitable for direct equality comparison between a previous snapshot
    and the current code state. Fields absent from a snapshot (e.g. an
    older version recorded before a new field was added) are filled
    with their dataclass defaults via ``Code.from_dict``-style coercion
    — but we go light here and just default missing keys to ``None``
    or empty containers; the equality check is symmetric.
    """
    if isinstance(code_or_dict, Code):
        d = code_or_dict.to_dict()
    elif isinstance(code_or_dict, dict):
        d = code_or_dict
    else:
        raise TypeError(
            "definition_signature expects a Code or its to_dict() output"
        )
    out: dict[str, Any] = {}
    for f in DEFINITION_FIELDS:
        v = d.get(f)
        # Normalise list-typed fields so a missing key compares equal
        # to an explicit empty list (older snapshots may omit defaults).
        if f in ("exemplars", "related_codes") and v is None:
            v = []
        out[f] = v
    # Round-trip through JSON to neutralise dataclass / list-of-dataclass
    # subtleties (CodeRelation vs dict). The on-disk format is JSON, so
    # this is the canonical comparison form.
    return json.loads(json.dumps(out))


def definition_changed(
    previous: CodeVersion | dict[str, Any] | None, current: Code
) -> bool:
    """Has the code's definition changed since ``previous``?

    Returns ``True`` if there is no previous version (the very first
    save always counts as a change), or if any of ``DEFINITION_FIELDS``
    differ between ``previous`` and ``current``. Metadata-only changes
    (stage, colour, status, provenance) return ``False``.
    """
    if previous is None:
        return True
    if isinstance(previous, CodeVersion):
        prev_sig = definition_signature(previous.snapshot)
    elif isinstance(previous, dict):
        prev_sig = definition_signature(previous)
    else:
        raise TypeError(
            "definition_changed: previous must be a CodeVersion, dict, or None"
        )
    return prev_sig != definition_signature(current)


# --------------------------------------------------------------------------- #
# On-disk persistence (append-only JSONL)
# --------------------------------------------------------------------------- #


def code_versions_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's code-version logs.

    Validates ``project_id`` to prevent traversal. Does not create the
    directory — callers / readers handle the missing case.
    """
    return project_dir(projects_root, project_id) / "code_versions"


def code_versions_path(
    projects_root: Path, project_id: str, code_id: str
) -> Path:
    """Return the path of a code's append-only version log."""
    if not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(f"Invalid code id: {code_id!r}")
    return code_versions_dir(projects_root, project_id) / f"{code_id}.jsonl"


def read_code_versions(
    projects_root: Path, project_id: str, code_id: str
) -> list[CodeVersion]:
    """Read all versions for a code, in stored (chronological) order.

    Skips lines that don't parse as a valid ``CodeVersion`` so a single
    corrupt line doesn't break the history view. Empty / missing file
    returns ``[]``.

    Order is preserved as written, not sorted by ``created_at`` — clock
    skew or backfilled entries should be visible in the order the
    researcher actually edited the code.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    path = code_versions_path(projects_root, project_id, code_id)
    if not path.exists():
        return []
    out: list[CodeVersion] = []
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
            out.append(CodeVersion.from_dict(payload))
        except ProjectValidationError:
            continue
    return out


def latest_code_version(
    projects_root: Path, project_id: str, code_id: str
) -> CodeVersion | None:
    """Return the most-recently-recorded version for a code, or None."""
    versions = read_code_versions(projects_root, project_id, code_id)
    return versions[-1] if versions else None


def find_code_version(
    projects_root: Path,
    project_id: str,
    code_id: str,
    version_id: str,
) -> CodeVersion | None:
    """Return the version with the given id, or None if not found.

    Used by F4.1's ``definition_version_id_at_apply`` lookup: given an
    application's recorded version id, fetch the snapshot to display
    "definition at the time of coding" in reports.
    """
    if not CODE_VERSION_ID_RE.match(version_id):
        raise ProjectValidationError(
            f"Invalid code-version id: {version_id!r}"
        )
    for v in read_code_versions(projects_root, project_id, code_id):
        if v.id == version_id:
            return v
    return None


def count_code_versions(
    projects_root: Path, project_id: str, code_id: str
) -> int:
    """Return the number of valid versions logged for a code."""
    return len(read_code_versions(projects_root, project_id, code_id))


def record_code_version(
    projects_root: Path,
    code: Code,
    *,
    change_note: str = "",
    now: str | None = None,
) -> CodeVersion:
    """Append a new version snapshot for ``code`` to its version log.

    Always records — does *not* check whether the definition has
    changed. Callers that want change-detection should use
    ``save_code_with_version``. This lower-level primitive exists so
    a hypothetical future "force re-snapshot" or "import historical
    versions" path has a hook.

    The parent ``projects/<id>`` directory must already exist (the
    project itself must have been saved). The ``code_versions/``
    subdirectory is created on demand.
    """
    code.validate()
    parent = project_dir(projects_root, code.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before recording code versions."
        )
    cvd = code_versions_dir(projects_root, code.project_id)
    cvd.mkdir(parents=True, exist_ok=True)

    if len(change_note) > MAX_CHANGE_NOTE_LEN:
        raise ProjectValidationError(
            f"change_note must be ≤ {MAX_CHANGE_NOTE_LEN} chars"
        )

    existing = read_code_versions(projects_root, code.project_id, code.id)
    next_version = (existing[-1].version + 1) if existing else 1

    v = CodeVersion.new(
        code=code,
        version=next_version,
        change_note=change_note,
        now=now,
    )
    target = code_versions_path(projects_root, code.project_id, code.id)
    line = json.dumps(v.to_dict(), ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)
    return v


def save_code_with_version(
    projects_root: Path,
    code: Code,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Path, CodeVersion | None]:
    """Persist a code and record a new version *if* its definition changed.

    The convenience wrapper most callers (and the F4.1 application
    workflow) will use. Behaviour:

      * Always writes the current ``code`` state via ``save_code``.
      * If there is no prior version, records ``version=1`` (so that
        the very first application has a version id to point at).
      * Otherwise compares the latest version's ``snapshot`` to the
        current ``code`` along ``DEFINITION_FIELDS``; if they differ,
        records a new version with ``version = latest.version + 1``.
        If they don't, leaves the version log untouched and returns
        the existing latest version as the second tuple element.

    The returned tuple is ``(saved_path, version_recorded_or_existing)``:
    the second element is never ``None`` after a successful save —
    callers always have a version they can pin an application to. The
    return type allows ``None`` so future callers (e.g. a "save without
    versioning" debug path) have an obvious extension point.
    """
    target = save_code(projects_root, code)

    latest = latest_code_version(projects_root, code.project_id, code.id)
    if latest is None or definition_changed(latest, code):
        recorded = record_code_version(
            projects_root, code, change_note=change_note, now=now
        )
        return target, recorded
    # No definition change — return the existing latest so the caller
    # has something to anchor an application to.
    return target, latest
