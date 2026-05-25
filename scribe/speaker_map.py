"""Speaker awareness for queries (F3.4).

Per PLANNING.md F3.4:

  > Speaker awareness in queries (separate interviewer from
  > interviewee; focus group support).

Scribe's ASR pipeline produces transcripts whose segments each carry a
``speaker`` string label — either the diarisation algorithm's
auto-generated tag (``"SPEAKER_00"``, …) or the operator-supplied
multi-track name (``"Luke"``, ``"Guest"``, …). Those labels are useful
for *display* but they don't tell the analyst which utterances are the
interviewer's, which are the interviewee's, or which participant in a
focus group is talking. Researchers asking "show every quote where the
speaker is the interviewee" (research note 8 in
``docs/research/coding-engine-research.md``) need a layer that maps
raw transcript labels → semantic role and (optionally) → a Project
participant.

This module is that layer. It provides:

  * a per-source **SpeakerMap** entity (one JSON file at
    ``projects/<pid>/speaker_maps/<sid>.json``) storing one
    :class:`SpeakerEntry` per distinct transcript label, with
    ``role`` ∈ :data:`SPEAKER_ROLES` and an optional
    ``participant_id`` linking to a :class:`scribe.participants.Participant`;
  * pure-Python helpers for **populating** the map from transcript
    segments (``speaker_labels_in_segments`` /
    ``speaker_map_from_segments`` / ``merge_segments_into_map``);
  * pure-Python helpers for **querying** transcript segments by role,
    by participant, or by raw label, plus role / participant
    distributions for the matrix views F3.6 will build on top.

F3.5's query builder is the natural caller of these helpers. F3.4 is
the data + algorithm layer that makes "interviewer vs interviewee"
filterable in the first place.

Design notes:

* **Per source, not project.** The same raw label (``"SPEAKER_00"``)
  in two different sources almost never refers to the same human, so
  the map lives next to the source. Cross-source identity is recorded
  via ``participant_id`` (one Participant linked from many sources'
  speaker maps).
* **Stored as a list, not a dict.** Insertion order matters for the
  UI ("interviewer is the first row"), and JSON dicts don't preserve
  that contract on the wire as cleanly as ordered lists do. Lookups
  inside this module go through ``SpeakerMap.get`` so callers don't
  pay the linear-scan cost.
* **Forward-compat.** Unknown ``role`` strings are rejected
  (vocabulary is small and stable), but unknown raw labels appearing
  in segments are allowed — :func:`merge_segments_into_map` will
  silently add them with ``role="unknown"``, so re-running on a new
  transcript doesn't trample existing role assignments.
* **Stand-alone.** No FastAPI, no engine imports. Mirrors
  :mod:`scribe.projects` (F1.1), :mod:`scribe.sources` (F1.2),
  :mod:`scribe.participants` (F1.3), and :mod:`scribe.source_schema`
  (F3.2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

# Canonical roles a speaker can play in a Source.
#
# ``unknown`` is the on-create default — a brand-new SpeakerMap built
# from a transcript shouldn't claim a role for any speaker until the
# researcher confirms one. ``other`` is the escape hatch for cases
# that don't fit the named roles (a translator joining a call, an
# audio source bleed-through), so the vocabulary stays small.
SPEAKER_ROLES: tuple[str, ...] = (
    "interviewer",
    "interviewee",
    "facilitator",
    "observer",
    "other",
    "unknown",
)

# Roles that, in the typical interview / focus-group setup, contain
# the *participant's* voice. Used by helpers like
# :func:`participant_voice_segments` so callers don't have to spell
# out the set every time.
PARTICIPANT_VOICE_ROLES: frozenset[str] = frozenset(
    {"interviewee", "facilitator"}
)

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB speaker_map.json.
MAX_LABEL_LEN = 200
MAX_DISPLAY_NAME_LEN = 200
MAX_NOTES_LEN = 1000
MAX_ENTRIES = 256

# Directory name relative to ``projects/<project_id>/``. One JSON file
# per source.
SPEAKER_MAPS_DIRNAME = "speaker_maps"


# --------------------------------------------------------------------------- #
# SpeakerEntry
# --------------------------------------------------------------------------- #


@dataclass
class SpeakerEntry:
    """One row in a :class:`SpeakerMap`.

    ``label`` is the raw transcript-side string (``"SPEAKER_00"`` or
    ``"Luke"``) — a verbatim copy of the ``speaker`` field on Scribe
    segments.  ``role`` selects the semantic role from
    :data:`SPEAKER_ROLES`. ``participant_id``, when set, points at a
    :class:`scribe.participants.Participant` — the cross-source
    identity. ``display_name`` is the optional UI override; falsy
    means "fall back to label".
    """

    label: str
    role: str = "unknown"
    participant_id: str | None = None
    display_name: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role,
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpeakerEntry":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SpeakerEntry payload must be an object"
            )
        if "label" not in d:
            raise ProjectValidationError(
                "SpeakerEntry payload missing 'label'"
            )
        pid_raw = d.get("participant_id")
        e = cls(
            label=str(d["label"]),
            role=str(d.get("role", "unknown") or "unknown"),
            participant_id=str(pid_raw) if pid_raw else None,
            display_name=str(d.get("display_name", "") or ""),
            notes=str(d.get("notes", "") or ""),
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        # Labels are user-visible strings — we strip whitespace and
        # forbid empties, but allow any character set otherwise (the
        # diariser may produce labels we can't anticipate, and
        # multi-track operator names are arbitrary).
        label = self.label.strip()
        if not label:
            raise ProjectValidationError("SpeakerEntry label is required")
        if len(label) > MAX_LABEL_LEN:
            raise ProjectValidationError(
                f"SpeakerEntry label must be ≤ {MAX_LABEL_LEN} chars"
            )
        self.label = label

        if self.role not in SPEAKER_ROLES:
            raise ProjectValidationError(
                f"SpeakerEntry role must be one of {SPEAKER_ROLES}; "
                f"got {self.role!r}"
            )

        if self.participant_id is not None:
            if not PARTICIPANT_ID_RE.match(self.participant_id):
                raise ProjectValidationError(
                    "SpeakerEntry participant_id must be 12-char hex; "
                    f"got {self.participant_id!r}"
                )

        dn = self.display_name.strip() if self.display_name else ""
        if len(dn) > MAX_DISPLAY_NAME_LEN:
            raise ProjectValidationError(
                f"SpeakerEntry display_name must be ≤ "
                f"{MAX_DISPLAY_NAME_LEN} chars"
            )
        self.display_name = dn

        notes = self.notes if self.notes else ""
        if len(notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"SpeakerEntry notes must be ≤ {MAX_NOTES_LEN} chars"
            )
        self.notes = notes


# --------------------------------------------------------------------------- #
# SpeakerMap
# --------------------------------------------------------------------------- #


@dataclass
class SpeakerMap:
    """Per-source map of raw speaker labels → role + participant.

    Entries are stored in insertion order so a UI listing them gets a
    stable column order. Labels are unique within the map.
    """

    project_id: str
    source_id: str
    entries: list[SpeakerEntry] = field(default_factory=list)
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
        source_id: str,
        entries: Iterable[SpeakerEntry | Mapping[str, Any]] | None = None,
        now: str | None = None,
    ) -> "SpeakerMap":
        ts = now or utcnow_iso()
        normalised: list[SpeakerEntry] = []
        for raw in entries or []:
            if isinstance(raw, SpeakerEntry):
                normalised.append(raw)
            elif isinstance(raw, Mapping):
                normalised.append(SpeakerEntry.from_dict(raw))
            else:
                raise ProjectValidationError(
                    "entries must be SpeakerEntry or dict objects"
                )
        m = cls(
            project_id=project_id,
            source_id=source_id,
            entries=normalised,
            created_at=ts,
            modified_at=ts,
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_id": self.source_id,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpeakerMap":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SpeakerMap payload must be an object"
            )
        for key in ("project_id", "source_id"):
            if key not in d:
                raise ProjectValidationError(
                    f"SpeakerMap payload missing {key!r}"
                )
        raw_entries = d.get("entries")
        if raw_entries is None:
            raw_entries = []
        if not isinstance(raw_entries, list):
            raise ProjectValidationError(
                "SpeakerMap entries must be a list"
            )
        entries = [SpeakerEntry.from_dict(e) for e in raw_entries]
        m = cls(
            project_id=str(d["project_id"]),
            source_id=str(d["source_id"]),
            entries=entries,
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def get(self, label: str) -> SpeakerEntry | None:
        """Return the entry for ``label`` or None."""
        for e in self.entries:
            if e.label == label:
                return e
        return None

    def has(self, label: str) -> bool:
        return self.get(label) is not None

    def role_for(self, label: str) -> str:
        """Return the role for ``label``; ``"unknown"`` if absent."""
        e = self.get(label)
        return e.role if e is not None else "unknown"

    def participant_for(self, label: str) -> str | None:
        """Return the linked participant id for ``label``, or None."""
        e = self.get(label)
        return e.participant_id if e is not None else None

    def display_name_for(self, label: str) -> str:
        """Return the display name for ``label``, falling back to label."""
        e = self.get(label)
        if e is None:
            return label
        return e.display_name or e.label

    def labels(self) -> list[str]:
        """All labels in insertion order."""
        return [e.label for e in self.entries]

    def labels_for_role(self, role: str | Iterable[str]) -> list[str]:
        """All labels whose role matches one of ``role``.

        ``role`` can be a single string or an iterable of role strings.
        Insertion order is preserved.
        """
        roles = _coerce_role_set(role)
        return [e.label for e in self.entries if e.role in roles]

    def labels_for_participant(self, participant_id: str) -> list[str]:
        """All labels linked to ``participant_id``. Insertion order preserved."""
        if not PARTICIPANT_ID_RE.match(participant_id):
            raise ProjectValidationError(
                f"Invalid participant id: {participant_id!r}"
            )
        return [
            e.label for e in self.entries if e.participant_id == participant_id
        ]

    def participants(self) -> list[str]:
        """Distinct participant ids referenced by entries (insertion order)."""
        seen: set[str] = set()
        out: list[str] = []
        for e in self.entries:
            pid = e.participant_id
            if pid and pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def upsert_entry(
        self,
        label: str,
        *,
        role: str | None = None,
        participant_id: str | None = None,
        display_name: str | None = None,
        notes: str | None = None,
        now: str | None = None,
    ) -> SpeakerEntry:
        """Insert or update an entry by ``label``. Returns the entry.

        Existing fields are preserved when the corresponding kwarg is
        ``None`` — this lets the UI patch one field without resending
        the whole row. Pass an explicit empty string to clear a
        free-form field; pass ``""`` for ``participant_id`` to unlink
        (kept as ``None`` on disk).
        """
        existing = self.get(label)
        if existing is None:
            entry = SpeakerEntry(
                label=label,
                role=role if role is not None else "unknown",
                participant_id=participant_id or None,
                display_name=display_name or "",
                notes=notes or "",
            )
            entry.validate()
            # Pre-check the cardinality so a rejected insert doesn't
            # leave the list temporarily over the limit (a follow-up
            # call would then trip the validator on its own state).
            if len(self.entries) >= MAX_ENTRIES:
                raise ProjectValidationError(
                    f"At most {MAX_ENTRIES} entries allowed"
                )
            self.entries.append(entry)
        else:
            # Patch in place. Keep validation atomic by mutating a
            # copy first and only swapping if it passes.
            patched = SpeakerEntry(
                label=existing.label,
                role=role if role is not None else existing.role,
                participant_id=(
                    participant_id or None
                    if participant_id is not None
                    else existing.participant_id
                ),
                display_name=(
                    display_name
                    if display_name is not None
                    else existing.display_name
                ),
                notes=(
                    notes
                    if notes is not None
                    else existing.notes
                ),
            )
            patched.validate()
            existing.role = patched.role
            existing.participant_id = patched.participant_id
            existing.display_name = patched.display_name
            existing.notes = patched.notes
            entry = existing

        # Re-validate the whole map to enforce the cardinality limit.
        self.validate()
        self.modified_at = now or utcnow_iso()
        return entry

    def remove_entry(self, label: str, *, now: str | None = None) -> bool:
        """Drop the entry for ``label``. Returns False if absent."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.label != label]
        if len(self.entries) == before:
            return False
        self.modified_at = now or utcnow_iso()
        return True

    def set_role(
        self, label: str, role: str, *, now: str | None = None
    ) -> SpeakerEntry:
        """Convenience: set the role of one label."""
        return self.upsert_entry(label, role=role, now=now)

    def link_participant(
        self,
        label: str,
        participant_id: str,
        *,
        now: str | None = None,
    ) -> SpeakerEntry:
        """Convenience: link ``label`` to ``participant_id``."""
        return self.upsert_entry(
            label, participant_id=participant_id, now=now
        )

    def unlink_participant(
        self, label: str, *, now: str | None = None
    ) -> SpeakerEntry:
        """Convenience: clear the participant link on ``label``.

        Raises if no entry for ``label`` exists.
        """
        if not self.has(label):
            raise ProjectValidationError(
                f"No entry for label {label!r}; "
                "use upsert_entry to create one"
            )
        # Pass an empty string sentinel so the patch sees "explicitly
        # clear" rather than "leave alone".
        return self.upsert_entry(label, participant_id="", now=now)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        if not isinstance(self.entries, list):
            raise ProjectValidationError(
                "SpeakerMap entries must be a list"
            )
        if len(self.entries) > MAX_ENTRIES:
            raise ProjectValidationError(
                f"At most {MAX_ENTRIES} entries allowed"
            )
        seen: set[str] = set()
        for e in self.entries:
            e.validate()
            if e.label in seen:
                raise ProjectValidationError(
                    f"Duplicate label in SpeakerMap: {e.label!r}"
                )
            seen.add(e.label)


# --------------------------------------------------------------------------- #
# Pure helpers — segment introspection
# --------------------------------------------------------------------------- #


def _speaker_of(segment: Any) -> str:
    """Pull the speaker label off a segment-shaped object.

    Accepts dicts (the on-disk transcript shape) and any object with a
    ``speaker`` attribute (the engine's ``Segment`` dataclass). Returns
    ``""`` if absent — callers treat that as "unlabelled".
    """
    if isinstance(segment, Mapping):
        v = segment.get("speaker")
    else:
        v = getattr(segment, "speaker", None)
    if v is None:
        return ""
    return str(v)


def speaker_labels_in_segments(
    segments: Iterable[Any],
) -> list[str]:
    """Return distinct speaker labels appearing in ``segments``.

    Order is **first-occurrence** — the same order the labels appear
    in the transcript. Empty / missing labels are dropped (a speakerless
    segment isn't a "speaker").
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            continue
        if lbl in seen:
            continue
        seen.add(lbl)
        out.append(lbl)
    return out


def speaker_map_from_segments(
    *,
    project_id: str,
    source_id: str,
    segments: Iterable[Any],
    default_role: str = "unknown",
    now: str | None = None,
) -> SpeakerMap:
    """Build a :class:`SpeakerMap` seeded from a transcript's segments.

    Every distinct label gets one entry; ``role`` defaults to
    ``default_role`` (which must be in :data:`SPEAKER_ROLES`).
    Researchers using diarised mode commonly want a fresh map with
    every speaker as ``"unknown"`` so they can assign roles in the UI;
    multi-track-mode users may want to seed with ``"interviewee"`` and
    just flip the interviewer row.
    """
    if default_role not in SPEAKER_ROLES:
        raise ProjectValidationError(
            f"default_role must be one of {SPEAKER_ROLES}; "
            f"got {default_role!r}"
        )
    labels = speaker_labels_in_segments(segments)
    entries = [SpeakerEntry(label=lbl, role=default_role) for lbl in labels]
    return SpeakerMap.new(
        project_id=project_id,
        source_id=source_id,
        entries=entries,
        now=now,
    )


def merge_segments_into_map(
    speaker_map: SpeakerMap,
    segments: Iterable[Any],
    *,
    new_role: str = "unknown",
    now: str | None = None,
) -> list[str]:
    """Append entries for any new labels found in ``segments``.

    Existing entries are left untouched (so a re-run after the
    researcher set roles doesn't trample those choices). Returns the
    list of newly added labels in first-occurrence order.

    ``new_role`` is the role assigned to the freshly-discovered
    labels; defaults to ``"unknown"``.
    """
    if new_role not in SPEAKER_ROLES:
        raise ProjectValidationError(
            f"new_role must be one of {SPEAKER_ROLES}; got {new_role!r}"
        )
    existing = {e.label for e in speaker_map.entries}
    added: list[str] = []
    for lbl in speaker_labels_in_segments(segments):
        if lbl in existing:
            continue
        existing.add(lbl)
        speaker_map.entries.append(SpeakerEntry(label=lbl, role=new_role))
        added.append(lbl)
    if added:
        speaker_map.validate()
        speaker_map.modified_at = now or utcnow_iso()
    return added


# --------------------------------------------------------------------------- #
# Pure helpers — segment filtering / counting
# --------------------------------------------------------------------------- #


def _coerce_role_set(roles: str | Iterable[str]) -> frozenset[str]:
    if isinstance(roles, str):
        items = [roles]
    else:
        items = [str(r) for r in roles]
    for r in items:
        if r not in SPEAKER_ROLES:
            raise ProjectValidationError(
                f"Unknown role {r!r}; must be one of {SPEAKER_ROLES}"
            )
    return frozenset(items)


def filter_segments_by_label(
    segments: Iterable[Any],
    labels: str | Iterable[str],
) -> list[Any]:
    """Return segments whose speaker label is in ``labels``.

    Empty/missing speaker labels never match — even if the caller
    passes ``""`` in ``labels``. (An unlabelled segment carries no
    speaker information; matching against the empty string would
    surface them in a way the caller almost never wants.)
    """
    if isinstance(labels, str):
        wanted = {labels}
    else:
        wanted = {str(s) for s in labels}
    out: list[Any] = []
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            continue
        if lbl in wanted:
            out.append(s)
    return out


def filter_segments_by_role(
    segments: Iterable[Any],
    speaker_map: SpeakerMap,
    roles: str | Iterable[str],
    *,
    include_unmapped: bool = False,
) -> list[Any]:
    """Return segments whose speaker has one of the given roles.

    Segments with a label not in ``speaker_map`` are treated as role
    ``"unknown"`` — so they match ``filter_segments_by_role(..., "unknown")``
    by default, mirroring how the rest of the module treats absent
    entries. Pass ``include_unmapped=True`` to additionally include
    those unmapped-but-labelled segments regardless of the requested
    role set (useful for an "everything plus orphans" audit pass).

    Segments with no speaker label at all are always dropped — they
    carry no speaker information for any role to match.
    """
    wanted = _coerce_role_set(roles)
    out: list[Any] = []
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            # Unlabelled segments carry no speaker info; nothing to
            # match against. (``include_unmapped`` targets the
            # *labelled-but-missing-from-map* case, not this one.)
            continue
        entry = speaker_map.get(lbl)
        role = entry.role if entry is not None else "unknown"
        if role in wanted:
            out.append(s)
        elif include_unmapped and entry is None:
            out.append(s)
    return out


def filter_segments_by_participant(
    segments: Iterable[Any],
    speaker_map: SpeakerMap,
    participant_ids: str | Iterable[str],
) -> list[Any]:
    """Return segments whose speaker is linked to one of ``participant_ids``.

    Segments whose label has no participant link (or whose label is
    absent from the map) are dropped — matching "show me only this
    participant's words" semantics.
    """
    if isinstance(participant_ids, str):
        ids = [participant_ids]
    else:
        ids = [str(p) for p in participant_ids]
    for pid in ids:
        if not PARTICIPANT_ID_RE.match(pid):
            raise ProjectValidationError(
                f"Invalid participant id: {pid!r}"
            )
    wanted = set(ids)
    # Pre-compute label → participant_id for O(1) lookups in the loop.
    label_to_pid: dict[str, str | None] = {
        e.label: e.participant_id for e in speaker_map.entries
    }
    out: list[Any] = []
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            continue
        pid = label_to_pid.get(lbl)
        if pid and pid in wanted:
            out.append(s)
    return out


def role_distribution(
    segments: Iterable[Any],
    speaker_map: SpeakerMap,
) -> dict[str, int]:
    """Count segments per role.

    Returns a dict whose keys are every role in :data:`SPEAKER_ROLES`
    (including those with zero count) plus ``""`` for unlabelled
    segments. Labels in the transcript that aren't in ``speaker_map``
    are bucketed as ``"unknown"`` (matching :func:`filter_segments_by_role`).
    """
    counts: dict[str, int] = {r: 0 for r in SPEAKER_ROLES}
    counts[""] = 0
    label_to_role: dict[str, str] = {
        e.label: e.role for e in speaker_map.entries
    }
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            counts[""] += 1
            continue
        counts[label_to_role.get(lbl, "unknown")] += 1
    return counts


def participant_distribution(
    segments: Iterable[Any],
    speaker_map: SpeakerMap,
) -> dict[str, int]:
    """Count segments per linked participant id.

    Keys are participant ids that actually appear in the transcript;
    segments whose speaker label has no participant link are bucketed
    under the empty string ``""``. Useful for the F3.6 matrix views
    ("how much did each participant talk in this focus group?").
    """
    counts: dict[str, int] = {}
    label_to_pid: dict[str, str | None] = {
        e.label: e.participant_id for e in speaker_map.entries
    }
    for s in segments:
        lbl = _speaker_of(s)
        if not lbl:
            counts[""] = counts.get("", 0) + 1
            continue
        pid = label_to_pid.get(lbl) or ""
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def participant_voice_segments(
    segments: Iterable[Any],
    speaker_map: SpeakerMap,
    *,
    voice_roles: Iterable[str] | None = None,
) -> list[Any]:
    """Return segments whose speaker is "the participant" by role.

    Default roles are :data:`PARTICIPANT_VOICE_ROLES` —
    ``interviewee`` and ``facilitator``. The interviewer's words are
    excluded, which matches the most common analytical query
    ("show me only what participants said"). Override ``voice_roles``
    if your project's role assignments differ.
    """
    roles = (
        list(voice_roles) if voice_roles is not None
        else list(PARTICIPANT_VOICE_ROLES)
    )
    return filter_segments_by_role(segments, speaker_map, roles)


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def speaker_maps_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's speaker maps.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / SPEAKER_MAPS_DIRNAME


def speaker_map_state_path(
    projects_root: Path, project_id: str, source_id: str
) -> Path:
    if not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id: {source_id!r}"
        )
    return speaker_maps_dir(projects_root, project_id) / f"{source_id}.json"


def save_speaker_map(
    projects_root: Path, speaker_map: SpeakerMap
) -> Path:
    """Persist a speaker map to ``<root>/<pid>/speaker_maps/<sid>.json``.

    The parent project directory must already exist. We don't insist
    that the source itself exists — a researcher can build a map for
    a forthcoming source during import — but the project must.
    """
    speaker_map.validate()
    parent = project_dir(projects_root, speaker_map.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its speaker maps."
        )
    sd = speaker_maps_dir(projects_root, speaker_map.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = speaker_map_state_path(
        projects_root, speaker_map.project_id, speaker_map.source_id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(speaker_map.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_speaker_map(
    projects_root: Path, project_id: str, source_id: str
) -> SpeakerMap:
    """Load a speaker map by (project, source) id pair.

    Raises ``FileNotFoundError`` if missing.
    """
    p = speaker_map_state_path(projects_root, project_id, source_id)
    if not p.exists():
        raise FileNotFoundError(f"No speaker map at {p}")
    return SpeakerMap.from_dict(json.loads(p.read_text()))


def load_or_empty_speaker_map(
    projects_root: Path, project_id: str, source_id: str
) -> SpeakerMap:
    """Load a speaker map if present; otherwise return an empty one.

    Useful for UI / export code paths that want to treat absence as
    "no roles assigned yet".
    """
    try:
        return load_speaker_map(projects_root, project_id, source_id)
    except FileNotFoundError:
        return SpeakerMap.new(project_id=project_id, source_id=source_id)


def list_speaker_maps(
    projects_root: Path, project_id: str
) -> list[SpeakerMap]:
    """List all speaker maps in a project. Sorted by source id.

    Skips files that don't parse as a valid SpeakerMap so a single
    corrupt file doesn't break the project view (audit log will
    eventually surface this — F9.7).
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(
            f"Invalid project id: {project_id!r}"
        )
    sd = speaker_maps_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[SpeakerMap] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not SOURCE_ID_RE.match(sid):
            continue
        try:
            out.append(SpeakerMap.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda m: m.source_id)
    return out


def delete_speaker_map(
    projects_root: Path, project_id: str, source_id: str
) -> bool:
    """Remove a source's speaker map file. Returns False if missing."""
    p = speaker_map_state_path(projects_root, project_id, source_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
