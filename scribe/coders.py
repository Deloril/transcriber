"""Coder entity for the academic-coding workflow (F2.5, part 1).

A Coder is a **person who applies codes to source segments**. Per
PLANNING.md F2.5:

  > Multi-coder mode. Per-coder application; ICR computation
  > (Cohen's kappa first, Krippendorff's alpha later); reconciliation
  > UI.

Most projects have one coder (the researcher themselves). Multi-coder
mode is when a project tracks two or more coders so their work can be
compared — for inter-coder reliability (ICR), reconciliation, or
simply to record who coded what for the audit trail.

Coders live alongside their project on disk:

    projects/<project_id>/coders/<coder_id>.json

so ``delete_project`` cleans them up for free, mirroring how Sources
(F1.2), Participants (F1.3), and Codes (F2.1) work.

Why not Participants?
---------------------

A *participant* is the human voice **on** a transcript (interviewee,
focus-group member). A *coder* is the human **analysing** transcripts
(researcher, second coder, reviewer). They share a similar shape but
have completely different roles in the audit trail: applications carry
``coder_id`` (F4.1, future), not ``participant_id``. Keeping the two
concepts separate matches every other QDA tool's vocabulary and avoids
sprinkling ``role="researcher"`` flags through participant rows.

What a coder records
--------------------

Minimal, on purpose:

  * ``name`` — display label ("Luke", "Coder B", "Reviewer 1").
  * ``role`` — short tag ("researcher", "second coder", "reviewer", or
    free-form "other"). Drives report grouping.
  * ``email`` — optional contact (for multi-team projects).
  * ``colour`` — display tint for highlighting their applications;
    same hex format as Code.colour.
  * ``status`` — ``active`` / ``inactive``. Inactive coders stay in
    history but are hidden in pickers.
  * ``notes`` — free-form ("trained on initial codebook 2026-04-12").

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused by
the CLI later. Conventions match ``scribe.projects`` (F1.1),
``scribe.sources`` (F1.2), ``scribe.participants`` (F1.3), and
``scribe.codes`` (F2.1).

Per-application coder linkage
-----------------------------

F4.1 specifies that an Application carries ``coder_id``. That field
references the Coder entity persisted here. F4.1 is not yet built;
once it lands, applications will read this directory to populate the
"who coded this" provenance display (F9.9) and to drive the multi-
coder ICR computation in :mod:`scribe.icr`.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Coder IDs follow the same 12-char hex shape as project / source /
# participant / code IDs; keeps URL routing and path-traversal guards
# uniform.
CODER_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Role vocabulary. Closed set with an ``other`` escape hatch so reports
# can still group sensibly. ``researcher`` is the project-PI / lead
# author; ``second_coder`` is the parallel coder used for ICR;
# ``reviewer`` adjudicates disagreements; ``trainee`` is a coder being
# onboarded whose work isn't yet being trusted in headline numbers.
CODER_ROLES: tuple[str, ...] = (
    "researcher",
    "second_coder",
    "reviewer",
    "trainee",
    "other",
)

# Lifecycle status. ``active`` is the default; ``inactive`` keeps the
# coder in history (their applications are still attributed) but hides
# them from pickers — analogous to the ``retired`` status on a Code.
CODER_STATUSES: tuple[str, ...] = (
    "active",
    "inactive",
)

# Colour: a CSS hex colour, either ``#RGB`` or ``#RRGGBB``. Empty
# string allowed (UI picks a default). Same shape as Code.colour so the
# UI's colour-picker component can be reused.
CODER_COLOUR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Email validation: deliberately conservative — we just need a "looks
# like a contact address" check, not a full RFC 5322 parser. Empty
# string is allowed (most coders don't need an email recorded).
CODER_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB coder.json.
MAX_NAME_LEN = 200
MAX_EMAIL_LEN = 320  # RFC 5321 cap
MAX_NOTES_LEN = 4000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Coder:
    """One person attached to a project as a coder.

    A coder records *who is doing the analytic work*. F4.1 will add
    ``coder_id`` to the Application entity so every coded segment
    points back here. F2.5's ICR helpers in :mod:`scribe.icr` consume
    coder-keyed application sets to compute Cohen's kappa.
    """

    id: str
    project_id: str
    name: str
    role: str = "researcher"
    email: str = ""
    colour: str = ""
    status: str = "active"
    notes: str = ""
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
        role: str = "researcher",
        email: str = "",
        colour: str = "",
        status: str = "active",
        notes: str = "",
        coder_id: str | None = None,
        now: str | None = None,
    ) -> "Coder":
        """Build a fresh Coder, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        c = cls(
            id=coder_id or new_coder_id(),
            project_id=project_id,
            name=name,
            role=role,
            email=email,
            colour=colour,
            status=status,
            notes=notes,
            created_at=ts,
            modified_at=ts,
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Coder":
        if not isinstance(d, dict):
            raise ProjectValidationError("Coder payload must be an object")
        if "id" not in d or "project_id" not in d or "name" not in d:
            raise ProjectValidationError(
                "Coder payload missing required keys"
            )
        c = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "")),
            role=str(d.get("role", "researcher") or "researcher"),
            email=str(d.get("email", "") or ""),
            colour=str(d.get("colour", "") or ""),
            status=str(d.get("status", "active") or "active"),
            notes=str(d.get("notes", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Mirrors ``Participant.apply_update``.

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
        if "role" in patch:
            self.role = str(patch["role"] or "")
        if "email" in patch:
            self.email = str(patch["email"] or "")
        if "colour" in patch:
            self.colour = str(patch["colour"] or "")
        if "status" in patch:
            self.status = str(patch["status"] or "")
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not CODER_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid coder id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        name = self.name.strip()
        if not name:
            raise ProjectValidationError("Coder name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Coder name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist the trimmed name so on-disk state is canonical.
        self.name = name

        if self.role not in CODER_ROLES:
            raise ProjectValidationError(
                f"role must be one of {CODER_ROLES}; got {self.role!r}"
            )

        email = self.email.strip()
        if email:
            if len(email) > MAX_EMAIL_LEN:
                raise ProjectValidationError(
                    f"email must be ≤ {MAX_EMAIL_LEN} chars"
                )
            if not CODER_EMAIL_RE.match(email):
                raise ProjectValidationError(
                    f"email does not look like an address: {email!r}"
                )
        self.email = email

        if self.colour:
            if not CODER_COLOUR_RE.match(self.colour):
                raise ProjectValidationError(
                    f"colour must be #RGB or #RRGGBB hex; got {self.colour!r}"
                )

        if self.status not in CODER_STATUSES:
            raise ProjectValidationError(
                f"status must be one of {CODER_STATUSES}; got {self.status!r}"
            )

        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes must be ≤ {MAX_NOTES_LEN} chars"
            )


# Fields a PATCH may set. id/project_id/created_at/modified_at are
# managed by the entity itself; passing them is allowed (and ignored)
# so a client can round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "name",
    "role",
    "email",
    "colour",
    "status",
    "notes",
}
_IGNORED_PATCH_KEYS = {"id", "project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_coder_id() -> str:
    """Mint a new 12-char hex coder id (matches project / source / job id shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def coders_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's coders.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "coders"


def coder_state_path(
    projects_root: Path, project_id: str, coder_id: str
) -> Path:
    if not CODER_ID_RE.match(coder_id):
        raise ProjectValidationError(f"Invalid coder id: {coder_id!r}")
    return coders_dir(projects_root, project_id) / f"{coder_id}.json"


def save_coder(projects_root: Path, coder: Coder) -> Path:
    """Persist a coder to ``<projects_root>/<pid>/coders/<cid>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). Like ``save_participant``,
    a coder without a project is meaningless and we surface that early.
    """
    coder.validate()
    parent = project_dir(projects_root, coder.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its coders."
        )
    cd = coders_dir(projects_root, coder.project_id)
    cd.mkdir(parents=True, exist_ok=True)
    target = coder_state_path(projects_root, coder.project_id, coder.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(coder.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_coder(
    projects_root: Path, project_id: str, coder_id: str
) -> Coder:
    """Load a coder by id. Raises ``FileNotFoundError`` if missing."""
    p = coder_state_path(projects_root, project_id, coder_id)
    if not p.exists():
        raise FileNotFoundError(f"No coder at {p}")
    return Coder.from_dict(json.loads(p.read_text()))


def list_coders(projects_root: Path, project_id: str) -> list[Coder]:
    """List all coders in a project.

    Skips files that don't parse as a valid Coder so a single corrupt
    file doesn't break the team view (audit log will eventually surface
    this — F9.7). Sorted by ``created_at`` ascending so the natural
    order matches how the team was assembled.
    """
    cd = coders_dir(projects_root, project_id)
    if not cd.exists():
        return []
    out: list[Coder] = []
    for f in sorted(cd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        cid = f.stem
        if not CODER_ID_RE.match(cid):
            continue
        try:
            out.append(Coder.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda c: (c.created_at, c.id))
    return out


def delete_coder(
    projects_root: Path, project_id: str, coder_id: str
) -> bool:
    """Remove a coder file. Returns False if it didn't exist.

    Note: deleting a coder does **not** retro-actively orphan their
    applications (F4.1 will record ``coder_id`` as a stable string
    reference, not a foreign key). The audit trail keeps the id around
    even if the Coder record itself is gone — which is the right
    behaviour: a researcher leaving the project doesn't erase the
    methodological record of who did what.
    """
    p = coder_state_path(projects_root, project_id, coder_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
