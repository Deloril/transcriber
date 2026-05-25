"""Participant entity for the academic-coding workflow (F1.3).

A Participant is the human voice (or author) behind one or more
``Source`` records inside a Project. Per PLANNING.md F1.3:

  > Participant entity (one participant ↔ many sources;
  > demographic columns user-defined).

Participants live alongside their project on disk:

    projects/<project_id>/participants/<participant_id>.json

so ``delete_project`` cleans them up for free, mirroring how Sources
work in F1.2.

Demographics are user-defined: every project picks its own column set
(age band, gender identity, role, organisation, …) so we store them as
a free-form ``dict[str, str]``. F3.2 / F3.3 will layer a project-level
schema on top to drive a consistent UI table — but the on-disk shape
already supports it.

Source linkage is stored as a list of source IDs on the participant
side. Today this models "one participant ↔ many sources" cleanly. F3.3
will add the inverse navigation (and focus-group support, where one
source has many participants) without breaking this layout: a single
source ID can appear on more than one participant's list.

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused by
the CLI later. It mirrors ``scribe.projects`` (F1.1) and
``scribe.sources`` (F1.2).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .sources import SOURCE_ID_RE


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Participant IDs follow the same 12-char hex shape as project / source /
# job IDs; keeps URL routing and path-traversal guards uniform.
PARTICIPANT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Demographic keys: same shape as Source.custom_attributes. Short
# identifiers, letters / digits / underscore / hyphen / space, so the
# UI can render them as table column headers.
DEMOGRAPHIC_KEY_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB participant.json.
MAX_NAME_LEN = 200
MAX_PSEUDONYM_LEN = 200
MAX_NOTES_LEN = 4000
MAX_DEMOGRAPHICS = 32
MAX_DEMOGRAPHIC_VALUE_LEN = 500
MAX_SOURCE_LINKS = 1000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Participant:
    """One participant attached to a project.

    ``name`` is the working label used inside the tool (often a short
    code like "P03" or the participant's first name). ``pseudonym`` is
    the published / exported label used for anonymisation; F6.7 will
    use it when bundling anonymised exports. Both default to free-form
    text — nothing here decides what's "real" and what's anonymised;
    that's a project-level convention.

    ``demographics`` carries the project-defined column values for this
    participant (age band, role, etc.). Free-form ``dict[str, str]``
    today, schema'd via F3.2 later.

    ``source_ids`` lists the sources in which this participant
    appears. We don't enforce referential integrity on the source side
    — a source might exist or not — so a stale link doesn't break load.
    The ``list_sources`` API can be used to filter in the UI.
    """

    id: str
    project_id: str
    name: str
    pseudonym: str = ""
    demographics: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    source_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        name: str,
        pseudonym: str = "",
        demographics: dict[str, Any] | None = None,
        notes: str = "",
        source_ids: Iterable[str] | None = None,
        participant_id: str | None = None,
        now: str | None = None,
    ) -> "Participant":
        """Build a fresh Participant, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        p = cls(
            id=participant_id or new_participant_id(),
            project_id=project_id,
            name=name,
            pseudonym=pseudonym,
            demographics=dict(demographics or {}),
            notes=notes,
            source_ids=list(source_ids or []),
            created_at=ts,
            modified_at=ts,
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Participant":
        if not isinstance(d, dict):
            raise ProjectValidationError("Participant payload must be an object")
        if "id" not in d or "project_id" not in d or "name" not in d:
            raise ProjectValidationError(
                "Participant payload missing required keys"
            )
        p = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "")),
            pseudonym=str(d.get("pseudonym", "") or ""),
            demographics={
                str(k): str(v)
                for k, v in (d.get("demographics") or {}).items()
            },
            notes=str(d.get("notes", "") or ""),
            source_ids=[str(s) for s in (d.get("source_ids") or [])],
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Mirrors ``Source.apply_update``.

        ``id``, ``project_id``, ``created_at``, and ``modified_at`` are
        ignored if present — they're managed by the entity, not the
        user.
        """
        if not isinstance(patch, dict):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS - _IGNORED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "name" in patch:
            self.name = str(patch["name"] or "")
        if "pseudonym" in patch:
            self.pseudonym = str(patch["pseudonym"] or "")
        if "demographics" in patch:
            demo = patch["demographics"] or {}
            if not isinstance(demo, dict):
                raise ProjectValidationError(
                    "demographics must be an object of string→string"
                )
            self.demographics = {str(k): str(v) for k, v in demo.items()}
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")
        if "source_ids" in patch:
            sids = patch["source_ids"] or []
            if not isinstance(sids, list):
                raise ProjectValidationError(
                    "source_ids must be a list of source ids"
                )
            self.source_ids = [str(s) for s in sids]

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Source-link helpers
    # ------------------------------------------------------------------ #

    def add_source(self, source_id: str, *, now: str | None = None) -> bool:
        """Link this participant to a source. Idempotent.

        Returns True if a new link was added, False if it was already
        there. Validates id shape; raises on bad shape.
        """
        sid = str(source_id)
        if not SOURCE_ID_RE.match(sid):
            raise ProjectValidationError(f"Invalid source id: {sid!r}")
        if sid in self.source_ids:
            return False
        self.source_ids.append(sid)
        self.validate()
        self.modified_at = now or utcnow_iso()
        return True

    def remove_source(self, source_id: str, *, now: str | None = None) -> bool:
        """Unlink a source. Returns True if removed, False if not linked."""
        sid = str(source_id)
        if sid not in self.source_ids:
            return False
        self.source_ids = [s for s in self.source_ids if s != sid]
        self.modified_at = now or utcnow_iso()
        return True

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PARTICIPANT_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid participant id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        name = self.name.strip()
        if not name:
            raise ProjectValidationError("Participant name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Participant name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist the trimmed name so on-disk state is canonical.
        self.name = name

        pseudonym = self.pseudonym.strip()
        if len(pseudonym) > MAX_PSEUDONYM_LEN:
            raise ProjectValidationError(
                f"pseudonym must be ≤ {MAX_PSEUDONYM_LEN} chars"
            )
        self.pseudonym = pseudonym

        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes must be ≤ {MAX_NOTES_LEN} chars"
            )

        if not isinstance(self.demographics, dict):
            raise ProjectValidationError(
                "demographics must be an object of string→string"
            )
        if len(self.demographics) > MAX_DEMOGRAPHICS:
            raise ProjectValidationError(
                f"At most {MAX_DEMOGRAPHICS} demographic fields allowed"
            )
        cleaned: dict[str, str] = {}
        for raw_k, raw_v in self.demographics.items():
            k = str(raw_k).strip()
            if not k:
                continue  # silently drop empty keys; less friction in UI
            if not DEMOGRAPHIC_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"demographics key {k!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            v = str(raw_v)
            if len(v) > MAX_DEMOGRAPHIC_VALUE_LEN:
                raise ProjectValidationError(
                    f"demographics[{k!r}] value too long "
                    f"(>{MAX_DEMOGRAPHIC_VALUE_LEN})"
                )
            cleaned[k] = v
        self.demographics = cleaned

        if not isinstance(self.source_ids, list):
            raise ProjectValidationError(
                "source_ids must be a list of source ids"
            )
        if len(self.source_ids) > MAX_SOURCE_LINKS:
            raise ProjectValidationError(
                f"At most {MAX_SOURCE_LINKS} source links allowed"
            )
        # De-dupe while preserving insertion order; reject any link that
        # isn't a 12-char hex id (matches Source.SOURCE_ID_RE).
        seen: set[str] = set()
        deduped: list[str] = []
        for raw in self.source_ids:
            sid = str(raw).strip()
            if not sid:
                continue
            if not SOURCE_ID_RE.match(sid):
                raise ProjectValidationError(
                    f"Invalid source id in source_ids: {sid!r}"
                )
            if sid in seen:
                continue
            seen.add(sid)
            deduped.append(sid)
        self.source_ids = deduped


# Fields a PATCH may set. id/project_id/created_at/modified_at are
# managed by the entity itself; passing them is allowed (and ignored)
# so a client can round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "name",
    "pseudonym",
    "demographics",
    "notes",
    "source_ids",
}
_IGNORED_PATCH_KEYS = {"id", "project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_participant_id() -> str:
    """Mint a new 12-char hex participant id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def participants_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's participants.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "participants"


def participant_state_path(
    projects_root: Path, project_id: str, participant_id: str
) -> Path:
    if not PARTICIPANT_ID_RE.match(participant_id):
        raise ProjectValidationError(
            f"Invalid participant id: {participant_id!r}"
        )
    return participants_dir(projects_root, project_id) / f"{participant_id}.json"


def save_participant(projects_root: Path, participant: Participant) -> Path:
    """Persist a participant to
    ``<projects_root>/<pid>/participants/<part_id>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). Like ``save_source``, a
    participant without a project is meaningless and we surface that
    early.
    """
    participant.validate()
    parent = project_dir(projects_root, participant.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its participants."
        )
    pd = participants_dir(projects_root, participant.project_id)
    pd.mkdir(parents=True, exist_ok=True)
    target = participant_state_path(
        projects_root, participant.project_id, participant.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(participant.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_participant(
    projects_root: Path, project_id: str, participant_id: str
) -> Participant:
    """Load a participant by id. Raises ``FileNotFoundError`` if missing."""
    p = participant_state_path(projects_root, project_id, participant_id)
    if not p.exists():
        raise FileNotFoundError(f"No participant at {p}")
    return Participant.from_dict(json.loads(p.read_text()))


def list_participants(
    projects_root: Path, project_id: str
) -> list[Participant]:
    """List all participants in a project.

    Skips files that don't parse as a valid Participant so a single
    corrupt file doesn't break the project view (audit log will
    eventually surface this — F9.7). Sorted by ``created_at`` ascending
    so the order matches how the researcher built up the corpus
    (mirrors ``list_sources``).
    """
    pd = participants_dir(projects_root, project_id)
    if not pd.exists():
        return []
    out: list[Participant] = []
    for f in sorted(pd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        pid = f.stem
        if not PARTICIPANT_ID_RE.match(pid):
            continue
        try:
            out.append(Participant.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda p: (p.created_at, p.id))
    return out


def delete_participant(
    projects_root: Path, project_id: str, participant_id: str
) -> bool:
    """Remove a participant file. Returns False if it didn't exist."""
    p = participant_state_path(projects_root, project_id, participant_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
