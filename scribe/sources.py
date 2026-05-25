"""Source entity for the academic-coding workflow (F1.2).

A Source is a single primary-data item attached to a Project: most
commonly a transcript produced by Scribe's existing ASR pipeline, but
the field set is forward-compatible with field notes, documents, and
images that later F-features will plug in.

Per PLANNING.md F1.2 the Source captures:
  - the linkage to the underlying transcript job (``transcript_job_id``)
  - source ``type`` (transcript / field_notes / document / image)
  - ``language``
  - ``recording_date``
  - user-defined ``custom_attributes``

Sources live under ``projects/<project_id>/sources/<source_id>.json``,
inside the parent project's directory, so deleting a project cleans up
its sources for free.

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused
by the CLI later. It mirrors the conventions established by
``scribe.projects`` in F1.1.
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


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Source kinds we recognise. ``transcript`` is the only one the rest of
# the pipeline produces today; the others are listed so future phases
# (importing field notes / scanned documents / images) can land without
# breaking forward-compat on the on-disk format.
SOURCE_TYPES: tuple[str, ...] = (
    "transcript",
    "field_notes",
    "document",
    "image",
)

# Source IDs follow the same 12-char hex shape as project + job IDs.
SOURCE_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# A reference to a transcription job lives under ``outputs/<job_id>/``;
# we don't validate existence here (a Source can outlive its job, e.g.
# during reorganisation), but the shape must match the project's job-id
# convention so we never end up with a path-traversal in the field.
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# YYYY-MM-DD; empty allowed. We don't insist on full ISO-8601 because
# recording metadata is often imprecise (the participant said "Tuesday"
# and that's all we know).
RECORDING_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Custom-attribute keys: user-defined columns per source (F3.2 will
# formalise the schema). We only constrain shape: short identifiers,
# letters / digits / underscore / hyphen / space, so the UI can render
# them as column headers and exports stay sensible.
CUSTOM_ATTR_KEY_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB source.json.
MAX_NAME_LEN = 200
MAX_LANGUAGE_LEN = 16          # BCP-47 codes ("en", "en-US", "zh-Hant")
MAX_NOTES_LEN = 4000           # free-form notes about the source
MAX_CUSTOM_ATTRS = 32
MAX_CUSTOM_ATTR_VALUE_LEN = 500


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    """One source attached to a project.

    Today this is overwhelmingly a Scribe transcript (linked through
    ``transcript_job_id``), but the entity is structured so other
    primary-data types — field notes typed in by the researcher,
    imported documents, photographs — can be added later without an
    on-disk migration.

    ``custom_attributes`` is a free-form dict of short strings; F3.2
    will layer a project-level *schema* on top so the UI can render
    consistent columns across sources, but the storage shape is
    already in place.
    """

    id: str
    project_id: str
    name: str
    source_type: str = "transcript"
    transcript_job_id: str | None = None
    language: str = ""
    recording_date: str = ""
    notes: str = ""
    custom_attributes: dict[str, str] = field(default_factory=dict)
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
        source_type: str = "transcript",
        transcript_job_id: str | None = None,
        language: str = "",
        recording_date: str = "",
        notes: str = "",
        custom_attributes: dict[str, Any] | None = None,
        source_id: str | None = None,
        now: str | None = None,
    ) -> "Source":
        """Build a fresh Source, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        # Normalise falsy ("", None) → None so callers can pass either.
        normalised_job_id = transcript_job_id if transcript_job_id else None
        s = cls(
            id=source_id or new_source_id(),
            project_id=project_id,
            name=name,
            source_type=source_type,
            transcript_job_id=normalised_job_id,
            language=language,
            recording_date=recording_date,
            notes=notes,
            custom_attributes=dict(custom_attributes or {}),
            created_at=ts,
            modified_at=ts,
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        if not isinstance(d, dict):
            raise ProjectValidationError("Source payload must be an object")
        if "id" not in d or "project_id" not in d or "name" not in d:
            raise ProjectValidationError("Source payload missing required keys")
        s = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "")),
            source_type=str(d.get("source_type", "transcript") or "transcript"),
            transcript_job_id=(
                str(d["transcript_job_id"])
                if d.get("transcript_job_id")
                else None
            ),
            language=str(d.get("language", "") or ""),
            recording_date=str(d.get("recording_date", "") or ""),
            notes=str(d.get("notes", "") or ""),
            custom_attributes={
                str(k): str(v) for k, v in (d.get("custom_attributes") or {}).items()
            },
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Mirrors ``Project.apply_update``.

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
        if "source_type" in patch:
            self.source_type = str(patch["source_type"] or "")
        if "transcript_job_id" in patch:
            v = patch["transcript_job_id"]
            self.transcript_job_id = str(v) if v else None
        if "language" in patch:
            self.language = str(patch["language"] or "")
        if "recording_date" in patch:
            self.recording_date = str(patch["recording_date"] or "")
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")
        if "custom_attributes" in patch:
            ca = patch["custom_attributes"] or {}
            if not isinstance(ca, dict):
                raise ProjectValidationError(
                    "custom_attributes must be an object of string→string"
                )
            self.custom_attributes = {str(k): str(v) for k, v in ca.items()}

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not SOURCE_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid source id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        name = self.name.strip()
        if not name:
            raise ProjectValidationError("Source name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Source name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist the trimmed name so on-disk state is canonical.
        self.name = name

        if self.source_type not in SOURCE_TYPES:
            raise ProjectValidationError(
                f"source_type must be one of {SOURCE_TYPES}; "
                f"got {self.source_type!r}"
            )

        if self.transcript_job_id is not None:
            if not JOB_ID_RE.match(self.transcript_job_id):
                raise ProjectValidationError(
                    f"transcript_job_id must be 12-char hex; "
                    f"got {self.transcript_job_id!r}"
                )

        if len(self.language) > MAX_LANGUAGE_LEN:
            raise ProjectValidationError(
                f"language must be ≤ {MAX_LANGUAGE_LEN} chars"
            )

        if self.recording_date:
            if not RECORDING_DATE_RE.match(self.recording_date):
                raise ProjectValidationError(
                    f"recording_date must be YYYY-MM-DD; "
                    f"got {self.recording_date!r}"
                )
            # Range-check the components — catches "2024-13-40" which
            # the regex permits.
            y, m, d = (int(x) for x in self.recording_date.split("-"))
            if not (1 <= m <= 12 and 1 <= d <= 31):
                raise ProjectValidationError(
                    f"recording_date components out of range: {self.recording_date!r}"
                )

        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes must be ≤ {MAX_NOTES_LEN} chars"
            )

        if not isinstance(self.custom_attributes, dict):
            raise ProjectValidationError(
                "custom_attributes must be an object of string→string"
            )
        if len(self.custom_attributes) > MAX_CUSTOM_ATTRS:
            raise ProjectValidationError(
                f"At most {MAX_CUSTOM_ATTRS} custom attributes allowed"
            )
        cleaned: dict[str, str] = {}
        for raw_k, raw_v in self.custom_attributes.items():
            k = str(raw_k).strip()
            if not k:
                continue  # silently drop empty keys; less friction in UI
            if not CUSTOM_ATTR_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"custom_attributes key {k!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            v = str(raw_v)
            if len(v) > MAX_CUSTOM_ATTR_VALUE_LEN:
                raise ProjectValidationError(
                    f"custom_attributes[{k!r}] value too long "
                    f"(>{MAX_CUSTOM_ATTR_VALUE_LEN})"
                )
            cleaned[k] = v
        self.custom_attributes = cleaned


# Fields a PATCH may set. id/project_id/created_at/modified_at are
# managed by the entity itself; passing them is allowed (and ignored)
# so a client can round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "name",
    "source_type",
    "transcript_job_id",
    "language",
    "recording_date",
    "notes",
    "custom_attributes",
}
_IGNORED_PATCH_KEYS = {"id", "project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_source_id() -> str:
    """Mint a new 12-char hex source id (matches project + job id shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def sources_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's sources.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "sources"


def source_state_path(
    projects_root: Path, project_id: str, source_id: str
) -> Path:
    if not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(f"Invalid source id: {source_id!r}")
    return sources_dir(projects_root, project_id) / f"{source_id}.json"


def save_source(projects_root: Path, source: Source) -> Path:
    """Persist a source to ``<projects_root>/<pid>/sources/<sid>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). We don't auto-create it
    because a source without a project is meaningless and would just
    hide a programming error.
    """
    source.validate()
    parent = project_dir(projects_root, source.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its sources."
        )
    sd = sources_dir(projects_root, source.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = source_state_path(projects_root, source.project_id, source.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(source.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_source(
    projects_root: Path, project_id: str, source_id: str
) -> Source:
    """Load a source by id. Raises ``FileNotFoundError`` if missing."""
    p = source_state_path(projects_root, project_id, source_id)
    if not p.exists():
        raise FileNotFoundError(f"No source at {p}")
    return Source.from_dict(json.loads(p.read_text()))


def list_sources(projects_root: Path, project_id: str) -> list[Source]:
    """List all sources of a project.

    Skips files that don't parse as a valid Source so a single corrupt
    source doesn't break the project view (audit log will eventually
    surface this — F9.7). Sorted by ``created_at`` ascending so the
    natural reading order matches the order sources were added (mirrors
    how a researcher builds up the corpus).
    """
    sd = sources_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[Source] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not SOURCE_ID_RE.match(sid):
            continue
        try:
            out.append(Source.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def delete_source(
    projects_root: Path, project_id: str, source_id: str
) -> bool:
    """Remove a source file. Returns False if it didn't exist."""
    p = source_state_path(projects_root, project_id, source_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
