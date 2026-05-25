"""Sampling log on the project (F1.4).

Per PLANNING.md F1.4:

  > Sampling log on the project (which sources added when, why —
  > theoretical-sampling justification).

Grounded theory — and Charmaz's constructivist variant in particular —
treats *theoretical sampling* as a first-class methodological move:
once categories start forming, the researcher chooses what to look at
next in order to fill out properties of those categories, *not* for
representativeness. The sampling log captures that decision trail:
which source was added when, what kind of sampling decision drove it,
and the rationale (the "why this one, why now"). It's also where a
researcher records sources they *removed* (e.g. unusable audio) or
sources they *plan* to recruit but haven't yet.

The on-disk format is **append-only JSON Lines** at:

    projects/<project_id>/sampling_log.jsonl

Append-only is deliberate — entries are evidence, not editable state.
To "correct" an earlier entry, append a new entry that references it
(``amends`` field, free-form). This is the same shape F9.1's project-
wide event log will use; we can fold the sampling log into that later
without an on-disk migration.

This module is stand-alone — no FastAPI, no engine imports — mirroring
the conventions of ``scribe.projects`` (F1.1), ``scribe.sources`` (F1.2),
and ``scribe.participants`` (F1.3).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .sources import SOURCE_ID_RE
from .participants import PARTICIPANT_ID_RE


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Sampling-log entry IDs follow the same 12-char hex shape as project /
# source / participant / job IDs.
SAMPLING_ENTRY_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# What kind of action this entry records. ``added`` is the common case
# (a source has been added to the corpus); ``planned`` lets the
# researcher document a recruitment decision before the source exists;
# ``removed`` records an exclusion (e.g. unusable audio); ``noted`` is
# a free-form sampling reflection (e.g. "saturation reached for
# category X") that doesn't fit the other actions.
SAMPLING_ACTIONS: tuple[str, ...] = (
    "added",
    "planned",
    "removed",
    "noted",
)

# The methodological flavour of the sampling decision. "" is allowed —
# not every entry needs a strategy label, and we don't lock users into
# one vocabulary. The values come from standard qualitative-methods
# textbooks (Charmaz, Patton): "theoretical" for grounded-theory
# sampling driven by emerging categories; "purposive" for selecting on
# specific criteria; "convenience" for who's available; "snowball" for
# referrals; "criterion" for inclusion-criteria-driven; "extreme_case"
# / "typical_case" / "negative_case" / "deviant_case" for the various
# case-selection strategies; "opportunistic" for serendipitous adds;
# "maximum_variation" for diversity-driven; "other" as the catch-all.
SAMPLING_DECISION_TYPES: tuple[str, ...] = (
    "",
    "theoretical",
    "purposive",
    "convenience",
    "snowball",
    "criterion",
    "extreme_case",
    "typical_case",
    "negative_case",
    "deviant_case",
    "opportunistic",
    "maximum_variation",
    "other",
)

# Field length / cardinality limits.
MAX_TARGET_CATEGORY_LEN = 200
MAX_RATIONALE_LEN = 4000
MAX_NOTES_LEN = 4000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class SamplingEntry:
    """One immutable entry in a project's sampling log.

    A sampling entry says "on date D, the researcher took action A
    relative to source S (or participant P, or just a planned slot)
    because of rationale R, framed as decision type T". It's the
    methodologically-transparent paper trail for theoretical sampling.

    ``source_id`` and ``participant_id`` are both optional: a "planned"
    entry typically has neither (the source hasn't been recorded yet,
    the participant hasn't been recruited), while a "noted" reflection
    might also stand alone. An "added" entry will normally reference at
    least one of them.

    ``target_category`` captures what the researcher hoped this sample
    would illuminate ("what category they were meant to fill" — direct
    quote from PLANNING.md). It's optional because not every decision
    is category-driven.
    """

    id: str
    project_id: str
    created_at: str
    action: str = "added"
    decision_type: str = ""
    source_id: str | None = None
    participant_id: str | None = None
    target_category: str = ""
    rationale: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        action: str = "added",
        decision_type: str = "",
        source_id: str | None = None,
        participant_id: str | None = None,
        target_category: str = "",
        rationale: str = "",
        notes: str = "",
        entry_id: str | None = None,
        now: str | None = None,
    ) -> "SamplingEntry":
        """Build a fresh SamplingEntry, validate, and stamp ``created_at``."""
        e = cls(
            id=entry_id or new_sampling_entry_id(),
            project_id=project_id,
            created_at=now or utcnow_iso(),
            action=action,
            decision_type=decision_type,
            source_id=source_id if source_id else None,
            participant_id=participant_id if participant_id else None,
            target_category=target_category,
            rationale=rationale,
            notes=notes,
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SamplingEntry":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "SamplingEntry payload must be an object"
            )
        for required in ("id", "project_id", "created_at"):
            if required not in d:
                raise ProjectValidationError(
                    f"SamplingEntry payload missing required key: {required}"
                )
        e = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            created_at=str(d["created_at"]),
            action=str(d.get("action", "added") or "added"),
            decision_type=str(d.get("decision_type", "") or ""),
            source_id=(
                str(d["source_id"]) if d.get("source_id") else None
            ),
            participant_id=(
                str(d["participant_id"]) if d.get("participant_id") else None
            ),
            target_category=str(d.get("target_category", "") or ""),
            rationale=str(d.get("rationale", "") or ""),
            notes=str(d.get("notes", "") or ""),
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not SAMPLING_ENTRY_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid sampling entry id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not self.created_at:
            raise ProjectValidationError("created_at is required")

        if self.action not in SAMPLING_ACTIONS:
            raise ProjectValidationError(
                f"action must be one of {SAMPLING_ACTIONS}; "
                f"got {self.action!r}"
            )
        if self.decision_type not in SAMPLING_DECISION_TYPES:
            raise ProjectValidationError(
                f"decision_type must be one of {SAMPLING_DECISION_TYPES}; "
                f"got {self.decision_type!r}"
            )

        if self.source_id is not None:
            if not SOURCE_ID_RE.match(self.source_id):
                raise ProjectValidationError(
                    f"source_id must be 12-char hex; got {self.source_id!r}"
                )
        if self.participant_id is not None:
            if not PARTICIPANT_ID_RE.match(self.participant_id):
                raise ProjectValidationError(
                    f"participant_id must be 12-char hex; "
                    f"got {self.participant_id!r}"
                )

        target = self.target_category.strip()
        if len(target) > MAX_TARGET_CATEGORY_LEN:
            raise ProjectValidationError(
                f"target_category must be ≤ {MAX_TARGET_CATEGORY_LEN} chars"
            )
        # Persist trimmed so on-disk state is canonical (matches the
        # ``Project.name`` / ``Source.name`` pattern).
        self.target_category = target

        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise ProjectValidationError(
                f"rationale must be ≤ {MAX_RATIONALE_LEN} chars"
            )
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes must be ≤ {MAX_NOTES_LEN} chars"
            )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_sampling_entry_id() -> str:
    """Mint a new 12-char hex sampling-entry id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence (append-only JSONL)
# --------------------------------------------------------------------------- #


def sampling_log_path(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk path of a project's sampling log.

    Validates ``project_id`` to prevent traversal. Does not create the
    file — readers handle the "missing log" case as "empty log".
    """
    return project_dir(projects_root, project_id) / "sampling_log.jsonl"


def append_sampling_entry(
    projects_root: Path, entry: SamplingEntry
) -> Path:
    """Append an entry to the project's sampling log.

    The parent ``projects/<id>`` directory must already exist (the
    project itself must have been saved). Append-only: the on-disk file
    is opened in ``"a"`` mode, which is atomic for line-sized writes on
    POSIX systems. We don't expose an "edit" or "delete" — corrections
    are made by appending a follow-up entry that references the prior
    one in its ``notes``.
    """
    entry.validate()
    parent = project_dir(projects_root, entry.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before logging sampling entries."
        )
    target = sampling_log_path(projects_root, entry.project_id)
    # ``ensure_ascii=False`` keeps non-ASCII rationale text readable on
    # disk; ``json.dumps`` doesn't emit newlines so the line-per-entry
    # invariant is maintained.
    line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)
    return target


def read_sampling_log(
    projects_root: Path, project_id: str
) -> list[SamplingEntry]:
    """Read all sampling-log entries for a project, in stored order.

    Skips lines that don't parse as a valid ``SamplingEntry`` so a
    single corrupt line doesn't break the view (mirrors the resilience
    of ``list_sources`` / ``list_participants``). Empty file or missing
    file returns ``[]``.

    Note: the returned list preserves the on-disk order, which is the
    chronological order entries were appended. We deliberately do **not**
    sort by ``created_at`` — clock skew or backfilled entries should
    be visible in the order the researcher actually wrote them.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    path = sampling_log_path(projects_root, project_id)
    if not path.exists():
        return []
    out: list[SamplingEntry] = []
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
            out.append(SamplingEntry.from_dict(payload))
        except ProjectValidationError:
            continue
    return out


def count_sampling_entries(
    projects_root: Path, project_id: str
) -> int:
    """Return the number of valid entries in the sampling log.

    Convenience for UI badges ("Sampling log: 12 entries"). Reads the
    file once; for very large logs callers should switch to streaming,
    but a research project is unlikely to have >10k sampling decisions.
    """
    return len(read_sampling_log(projects_root, project_id))
