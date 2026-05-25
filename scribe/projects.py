"""Project entity for the academic-coding workflow (F1.1).

A Scribe project is a research corpus: a name, a research question, a
chosen methodology, optional sensitising concepts, and a current
codebook stage. It owns a directory under ``projects/<id>/`` that will
later hold the codebook, applications, memos, and audit trail.

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused
by the CLI later.

Subsequent features (F1.2 sources, F1.3 participants, F2.x codebook,
F9.x audit trail) extend this entity but should not need to break the
on-disk format.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# The canonical codebook stages from PLANNING.md "Glossary" section.
# `locked` is a terminal stage where new codes/edits are blocked (F2.4).
CODEBOOK_STAGES: tuple[str, ...] = (
    "initial",
    "focused",
    "axial",
    "theoretical",
    "locked",
)

# Methodologies we explicitly recognise. Free-form `other` is allowed
# so we don't lock users into Charmaz; F2 axial coding is optional, etc.
KNOWN_METHODOLOGIES: tuple[str, ...] = (
    "charmaz",
    "strauss-corbin",
    "glaser",
    "other",
    "",
)

# Field length / cardinality limits. Generous, but bounded so a
# typo in the UI can't write a 50 MB project.json.
MAX_NAME_LEN = 200
MAX_RESEARCH_QUESTION_LEN = 4000
MAX_METHODOLOGY_LEN = 64
MAX_SENSITISING_CONCEPT_LEN = 200
MAX_SENSITISING_CONCEPTS = 64
MAX_DESCRIPTION_LEN = 4000

# Project IDs follow the same shape as job IDs: 12-char lowercase hex.
# Keeps URL routing rules consistent across the app.
PROJECT_ID_RE = re.compile(r"^[a-f0-9]{12}$")


class ProjectValidationError(ValueError):
    """Raised when a Project payload fails validation."""


# --------------------------------------------------------------------------- #
# Time helpers — keep all timestamps as ISO-8601 UTC strings on disk
# --------------------------------------------------------------------------- #


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a Z suffix.

    We avoid ``datetime.utcnow()`` (deprecated in 3.12) and stick to
    timezone-aware values so round-tripping through other tools is
    unambiguous.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond:06d}Z"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Project:
    """A research project: corpus + codebook + audit trail anchor.

    Fields here track the F1.1 spec: name, research question,
    methodology, sensitising concepts, created/modified timestamps,
    and current codebook stage. ``description`` is included as an
    optional free-form field; it pairs naturally with the research
    question and costs nothing to support.

    The on-disk format is a single ``project.json`` per project,
    written to ``<projects_root>/<id>/project.json``. Future features
    (sources, codebook, memos) will sit alongside as sibling JSON
    files or subdirectories.
    """

    id: str
    name: str
    research_question: str = ""
    methodology: str = ""
    sensitising_concepts: list[str] = field(default_factory=list)
    codebook_stage: str = "initial"
    description: str = ""
    created_at: str = ""
    modified_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        name: str,
        research_question: str = "",
        methodology: str = "",
        sensitising_concepts: Iterable[str] | None = None,
        description: str = "",
        codebook_stage: str = "initial",
        project_id: str | None = None,
        now: str | None = None,
    ) -> "Project":
        """Build a fresh Project, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        p = cls(
            id=project_id or new_project_id(),
            name=name,
            research_question=research_question,
            methodology=methodology,
            sensitising_concepts=list(sensitising_concepts or []),
            codebook_stage=codebook_stage,
            description=description,
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
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        if not isinstance(d, dict):
            raise ProjectValidationError("Project payload must be an object")
        if "id" not in d or "name" not in d:
            raise ProjectValidationError("Project payload missing required keys")
        p = cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            research_question=str(d.get("research_question", "") or ""),
            methodology=str(d.get("methodology", "") or ""),
            sensitising_concepts=[str(s) for s in (d.get("sensitising_concepts") or [])],
            codebook_stage=str(d.get("codebook_stage", "initial") or "initial"),
            description=str(d.get("description", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Only known fields are
        accepted; anything else raises so the caller (or HTTP layer)
        gets a clear 400.

        ``id``, ``created_at``, and ``modified_at`` are ignored if
        present — they're managed by the entity, not the user.
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
        if "research_question" in patch:
            self.research_question = str(patch["research_question"] or "")
        if "methodology" in patch:
            self.methodology = str(patch["methodology"] or "")
        if "sensitising_concepts" in patch:
            sc = patch["sensitising_concepts"] or []
            if not isinstance(sc, list):
                raise ProjectValidationError(
                    "sensitising_concepts must be a list of strings"
                )
            self.sensitising_concepts = [str(s) for s in sc]
        if "codebook_stage" in patch:
            self.codebook_stage = str(patch["codebook_stage"] or "")
        if "description" in patch:
            self.description = str(patch["description"] or "")

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PROJECT_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid project id: {self.id!r}")
        name = self.name.strip()
        if not name:
            raise ProjectValidationError("Project name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Project name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist the trimmed name so on-disk state is canonical.
        self.name = name

        if len(self.research_question) > MAX_RESEARCH_QUESTION_LEN:
            raise ProjectValidationError(
                f"research_question must be ≤ {MAX_RESEARCH_QUESTION_LEN} chars"
            )
        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise ProjectValidationError(
                f"description must be ≤ {MAX_DESCRIPTION_LEN} chars"
            )
        if len(self.methodology) > MAX_METHODOLOGY_LEN:
            raise ProjectValidationError(
                f"methodology must be ≤ {MAX_METHODOLOGY_LEN} chars"
            )

        if self.codebook_stage not in CODEBOOK_STAGES:
            raise ProjectValidationError(
                f"codebook_stage must be one of {CODEBOOK_STAGES}; "
                f"got {self.codebook_stage!r}"
            )

        if not isinstance(self.sensitising_concepts, list):
            raise ProjectValidationError(
                "sensitising_concepts must be a list of strings"
            )
        if len(self.sensitising_concepts) > MAX_SENSITISING_CONCEPTS:
            raise ProjectValidationError(
                f"At most {MAX_SENSITISING_CONCEPTS} sensitising concepts allowed"
            )
        cleaned: list[str] = []
        for raw in self.sensitising_concepts:
            s = str(raw).strip()
            if not s:
                continue  # silently drop empties; less friction in UI
            if len(s) > MAX_SENSITISING_CONCEPT_LEN:
                raise ProjectValidationError(
                    f"Sensitising concept too long (>{MAX_SENSITISING_CONCEPT_LEN}): {s[:40]!r}…"
                )
            cleaned.append(s)
        self.sensitising_concepts = cleaned


# Fields a PATCH may set. id/created_at/modified_at are managed by the
# entity itself; passing them is allowed (and ignored) so a client can
# round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "name",
    "research_question",
    "methodology",
    "sensitising_concepts",
    "codebook_stage",
    "description",
}
_IGNORED_PATCH_KEYS = {"id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_project_id() -> str:
    """Mint a new 12-char hex project id (matches job-id shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def project_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory for a project. Does not create it."""
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    return projects_root / project_id


def project_state_path(projects_root: Path, project_id: str) -> Path:
    return project_dir(projects_root, project_id) / "project.json"


def save_project(projects_root: Path, project: Project) -> Path:
    """Persist a project to ``<projects_root>/<id>/project.json``.

    Atomic-ish: writes to a temp file in the same directory, then
    renames. Mirrors the pattern in ``server._save_profiles``.
    """
    project.validate()
    d = project_dir(projects_root, project.id)
    d.mkdir(parents=True, exist_ok=True)
    target = project_state_path(projects_root, project.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_project(projects_root: Path, project_id: str) -> Project:
    """Load a project by id. Raises ``FileNotFoundError`` if missing."""
    p = project_state_path(projects_root, project_id)
    if not p.exists():
        raise FileNotFoundError(f"No project at {p}")
    return Project.from_dict(json.loads(p.read_text()))


def list_projects(projects_root: Path) -> list[Project]:
    """List all projects under ``projects_root``.

    Skips directories that don't have a valid ``project.json``;
    surfaces no errors so a single corrupt project doesn't break the
    list view. Sorted by ``modified_at`` desc, then by id for
    stability.
    """
    if not projects_root.exists():
        return []
    out: list[Project] = []
    for d in sorted(projects_root.iterdir()):
        if not d.is_dir() or not PROJECT_ID_RE.match(d.name):
            continue
        sp = d / "project.json"
        if not sp.exists():
            continue
        try:
            out.append(Project.from_dict(json.loads(sp.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            # Don't let one bad project break the list; F9.7 will
            # eventually surface this through the audit log.
            continue
    out.sort(key=lambda p: (p.modified_at, p.id), reverse=True)
    return out


def delete_project(projects_root: Path, project_id: str) -> bool:
    """Remove a project's entire directory. Returns False if it didn't
    exist. Used by the API; tests use it for cleanup.
    """
    d = project_dir(projects_root, project_id)
    if not d.exists():
        return False
    # rmtree-equivalent done manually so we never traverse a symlink
    # outside the projects root.
    import shutil
    real_root = projects_root.resolve()
    real_d = d.resolve()
    if not str(real_d).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {d}")
    shutil.rmtree(d)
    return True
