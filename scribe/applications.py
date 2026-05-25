"""Code Application entity for the academic-coding workflow (F4.1).

Per PLANNING.md F4.1:

  > Application = (code_id, source_id, anchor_start_word_id,
  > anchor_end_word_id, optional sub-word char offsets, coder_id,
  > created_at, optional confidence/provenance,
  > definition_version_id_at_apply).

An *application* is one specific instance of a code attached to a span
of text in a source. Other QDA tools call this a "coded segment" or a
"quotation"; we adopt the methodologically-precise name. Per PLANNING
core principles (§2 "AI suggests, the human applies") and §4
("Word-level anchoring"):

* Every application has a **human author** (``coder_id``) — even AI
  suggestions only become applications via an explicit accept event,
  which carries the accepting human's ``coder_id``.

* The anchor is a **word-id span**, not a raw character offset. The
  word IDs are stable across transcript edits (within reason) so the
  audit trail survives typo fixes and anonymisation passes (F4.5 will
  formalise the orphan-review queue).

* A **sub-word character offset** at each end is optional. Researchers
  do code mid-word ("…cri-***minal***isation…"); the start/end offsets
  let an application narrow the span without inventing a new word ID.

* Every application records the **code-version-at-apply** (F2.2). A
  report can later show "this application was made when the code
  definition said X", regardless of how the definition has since
  evolved. ``definition_version_id_at_apply`` references the
  ``CodeVersion.id`` from :mod:`scribe.code_versions`.

* ``provenance`` is a small string→string dict modelled on the same
  field on :class:`scribe.codes.Code`. It carries origin metadata —
  ``source = "human" / "ai_accepted" / "imported" / ...``, plus
  optional ``model_id`` / ``suggestion_id`` / ``accepted_at`` keys for
  AI-derived applications. F8.x and F9.6 will use this; F4.1 just
  records and validates the shape.

* ``confidence`` is an optional float in [0, 1]. AI-derived applications
  populate it; manual applications usually leave it ``None``.

Word-ID format
--------------

Scribe's ASR pipeline produces ``segments[].words[]``. We anchor on a
deterministic synthetic id::

    s<segment_index>w<word_index>

so anchors carry no information not already in the transcript JSON, but
have a fixed, validatable shape that survives serialisation. Helpers
below (:func:`make_word_id`, :func:`parse_word_id`,
:func:`compare_word_ids`) keep the format in one place. Future
features (F4.4 selection helpers, F4.5 re-anchoring) will reuse them.

On-disk layout::

    projects/<project_id>/applications/<application_id>.json

So ``delete_project`` cleans applications up for free, mirroring how
Sources (F1.2), Participants (F1.3), Codes (F2.1), and Coders (F2.5)
are stored.

This module is stand-alone — no FastAPI, no engine imports — so the
data model can be tested in pure Python and reused by the CLI later.
Conventions match the rest of the F-feature stack.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .ai_provenance import AIProvenance
from .codes import CODE_ID_RE
from .code_versions import CODE_VERSION_ID_RE
from .coders import CODER_ID_RE
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

# Application IDs follow the same 12-char hex shape as every other id
# in Scribe.
APPLICATION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Word IDs encode (segment_index, word_index) as ``s<si>w<wi>`` with
# both as non-negative integers. The format is fixed so anchors can be
# parsed and compared deterministically; F4.5 (orphan re-anchoring) will
# need both pieces.
#
# Why not the raw ``words[].text`` value? Two reasons:
#   1. Texts collide ("the" appears 80 times in a transcript).
#   2. Texts mutate when the user edits the transcript; the index does
#      not until words are inserted/deleted, and that operation is
#      visible (so F4.5 can detect and queue an orphan review).
WORD_ID_RE = re.compile(r"^s(\d+)w(\d+)$")

# Provenance ``source`` vocabulary for an application. Closed set with
# an "other" escape hatch; mirrors :data:`scribe.codes.CODE_PROVENANCE_SOURCES`
# but tuned for application-time semantics:
#   - human: a coder selected text and applied a code.
#   - ai_accepted: AI suggested, a human pressed accept.
#   - ai_modified: AI suggested, a human accepted with span / code edits.
#   - imported: came in via REFI-QDA / QDPX import (F6.6).
#   - other: escape hatch for unusual flows (e.g. promoted from a memo).
APPLICATION_PROVENANCE_SOURCES: tuple[str, ...] = (
    "human",
    "ai_accepted",
    "ai_modified",
    "imported",
    "other",
)

# Provenance keys are short identifiers (same shape as Code.provenance
# and Source.custom_attributes). Letters/digits/underscore/hyphen/space,
# 1–64 chars, must start with a letter.
PROVENANCE_KEY_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")

# Field length / cardinality limits.
MAX_NOTE_LEN = 4000  # free-form per-application analytic note
MAX_PROVENANCE_KEYS = 16
MAX_PROVENANCE_VALUE_LEN = 1000

# Sub-word character offsets are bounded so a typo can't write a 4-GiB
# offset. Word texts in Scribe transcripts are short (longest reasonable
# tokenised word is well under 200 chars); we allow up to 2_000 to leave
# room for unusual corpora (long German compound nouns, URLs, etc.).
MAX_CHAR_OFFSET = 2_000


# --------------------------------------------------------------------------- #
# Word-ID helpers
# --------------------------------------------------------------------------- #


def make_word_id(segment_index: int, word_index: int) -> str:
    """Compose a canonical word id from (segment_index, word_index).

    Both indices must be non-negative integers; a TypeError or
    ProjectValidationError is raised otherwise. The composed id is
    guaranteed to round-trip through :func:`parse_word_id`.
    """
    if not isinstance(segment_index, int) or isinstance(segment_index, bool):
        raise ProjectValidationError(
            f"segment_index must be int; got {type(segment_index).__name__}"
        )
    if not isinstance(word_index, int) or isinstance(word_index, bool):
        raise ProjectValidationError(
            f"word_index must be int; got {type(word_index).__name__}"
        )
    if segment_index < 0:
        raise ProjectValidationError(
            f"segment_index must be ≥ 0; got {segment_index}"
        )
    if word_index < 0:
        raise ProjectValidationError(
            f"word_index must be ≥ 0; got {word_index}"
        )
    return f"s{segment_index}w{word_index}"


def parse_word_id(word_id: str) -> tuple[int, int]:
    """Parse a word id into (segment_index, word_index).

    Raises :class:`ProjectValidationError` on malformed input. The
    return is the canonical pair expected by anchor-comparison helpers.
    """
    if not isinstance(word_id, str):
        raise ProjectValidationError(
            f"word_id must be a string; got {type(word_id).__name__}"
        )
    m = WORD_ID_RE.match(word_id)
    if not m:
        raise ProjectValidationError(
            f"word_id must match s<segment>w<word>; got {word_id!r}"
        )
    return int(m.group(1)), int(m.group(2))


def compare_word_ids(a: str, b: str) -> int:
    """Return -1 / 0 / +1 ordering on two word ids.

    Compares lexicographically by (segment_index, word_index) — *not*
    by raw string, because ``"s10w0"`` should sort *after* ``"s2w0"``.
    Used by :meth:`Application.validate` to require start ≤ end.
    """
    pa = parse_word_id(a)
    pb = parse_word_id(b)
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Application:
    """One application of a code to a span of text in a source.

    All cross-entity references (``code_id``, ``source_id``,
    ``coder_id``, ``definition_version_id_at_apply``) are validated for
    *shape* only — F4.1 is the data model layer, and verifying that
    e.g. the referenced code exists is the HTTP / service layer's job
    (the HTTP layer can return 404 with context). Decoupling shape from
    existence keeps this module testable without a fully-built project.

    The anchor is a closed [start, end] interval over word ids, with
    optional sub-word character offsets at each end:

    * ``start_char_offset`` is interpreted relative to the *start* of
      the start word's text (so 0 means "from the first character").
    * ``end_char_offset`` is interpreted relative to the *end* of the
      end word's text — i.e. the offset is *from the right* — but to
      keep the on-disk format intuitive we store it *from the left* of
      the end word, and the editor renders the highlight up to (but not
      including) that index. ``None`` means "to the end of the word",
      which is the common case.

    The two char offsets are independent; either, both, or neither may
    be set. ``None`` everywhere is the natural default for whole-word
    snapping (F4.4 will lift this into convenience helpers).
    """

    id: str
    project_id: str
    code_id: str
    source_id: str
    coder_id: str
    anchor_start_word_id: str
    anchor_end_word_id: str
    definition_version_id_at_apply: str
    start_char_offset: int | None = None
    end_char_offset: int | None = None
    confidence: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    # F8.9: structured AI provenance. Optional — manual applications
    # leave it ``None``. When set, ``provenance['source']`` should be
    # one of the ``ai_*`` values for consistency with the legacy dict;
    # the helper :meth:`AIProvenance.to_application_provenance_dict`
    # produces an aligned dict.
    ai_provenance: "AIProvenance | None" = None
    note: str = ""
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
        code_id: str,
        source_id: str,
        coder_id: str,
        anchor_start_word_id: str,
        anchor_end_word_id: str,
        definition_version_id_at_apply: str,
        start_char_offset: int | None = None,
        end_char_offset: int | None = None,
        confidence: float | None = None,
        provenance: dict[str, Any] | None = None,
        ai_provenance: "AIProvenance | dict[str, Any] | None" = None,
        note: str = "",
        application_id: str | None = None,
        now: str | None = None,
    ) -> "Application":
        """Build a fresh Application, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        # Coerce provenance up-front so a non-dict surfaces as a clean
        # ProjectValidationError (matching apply_update / from_dict)
        # rather than a raw ``dict()`` ValueError out of the constructor.
        if provenance is None:
            coerced_provenance: dict[str, Any] = {}
        elif isinstance(provenance, dict):
            coerced_provenance = dict(provenance)
        else:
            raise ProjectValidationError(
                "provenance must be an object of string→string"
            )
        coerced_ai_prov = _coerce_ai_provenance(ai_provenance)
        a = cls(
            id=application_id or new_application_id(),
            project_id=project_id,
            code_id=code_id,
            source_id=source_id,
            coder_id=coder_id,
            anchor_start_word_id=anchor_start_word_id,
            anchor_end_word_id=anchor_end_word_id,
            definition_version_id_at_apply=definition_version_id_at_apply,
            start_char_offset=start_char_offset,
            end_char_offset=end_char_offset,
            confidence=confidence,
            provenance=coerced_provenance,
            ai_provenance=coerced_ai_prov,
            note=note,
            created_at=ts,
            modified_at=ts,
        )
        a.validate()
        return a

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        # ``asdict`` recurses into nested dataclasses, but AIProvenance
        # has its own to_dict that omits stray internal fields and keeps
        # field order canonical — call it explicitly instead.
        d = asdict(self)
        if self.ai_provenance is None:
            d["ai_provenance"] = None
        else:
            d["ai_provenance"] = self.ai_provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Application":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "Application payload must be an object"
            )
        for required in (
            "id",
            "project_id",
            "code_id",
            "source_id",
            "coder_id",
            "anchor_start_word_id",
            "anchor_end_word_id",
            "definition_version_id_at_apply",
        ):
            if required not in d:
                raise ProjectValidationError(
                    f"Application payload missing required key: {required}"
                )
        a = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            code_id=str(d["code_id"]),
            source_id=str(d["source_id"]),
            coder_id=str(d["coder_id"]),
            anchor_start_word_id=str(d["anchor_start_word_id"]),
            anchor_end_word_id=str(d["anchor_end_word_id"]),
            definition_version_id_at_apply=str(
                d["definition_version_id_at_apply"]
            ),
            start_char_offset=_optional_int(
                d.get("start_char_offset"), "start_char_offset"
            ),
            end_char_offset=_optional_int(
                d.get("end_char_offset"), "end_char_offset"
            ),
            confidence=_optional_float(d.get("confidence"), "confidence"),
            provenance={
                str(k): str(v)
                for k, v in (d.get("provenance") or {}).items()
            },
            ai_provenance=_coerce_ai_provenance(d.get("ai_provenance")),
            note=str(d.get("note", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        a.validate()
        return a

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Mirrors ``Code.apply_update``.

        The cross-entity ids (``code_id``, ``source_id``, ``coder_id``,
        ``definition_version_id_at_apply``) are not updatable here — an
        application that points at a different code is a *new*
        application, audit-trail-wise. F4.5 (re-anchoring) will let the
        anchor word-ids change in place when transcript edits shift
        their meaning; the F4.1 base allows this as well so existing
        callers don't need a separate helper before F4.5 lands.

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
        if "anchor_start_word_id" in patch:
            self.anchor_start_word_id = str(patch["anchor_start_word_id"] or "")
        if "anchor_end_word_id" in patch:
            self.anchor_end_word_id = str(patch["anchor_end_word_id"] or "")
        if "start_char_offset" in patch:
            self.start_char_offset = _optional_int(
                patch["start_char_offset"], "start_char_offset"
            )
        if "end_char_offset" in patch:
            self.end_char_offset = _optional_int(
                patch["end_char_offset"], "end_char_offset"
            )
        if "confidence" in patch:
            self.confidence = _optional_float(patch["confidence"], "confidence")
        if "provenance" in patch:
            prov = patch["provenance"] or {}
            if not isinstance(prov, dict):
                raise ProjectValidationError(
                    "provenance must be an object of string→string"
                )
            self.provenance = {str(k): str(v) for k, v in prov.items()}
        if "ai_provenance" in patch:
            self.ai_provenance = _coerce_ai_provenance(patch["ai_provenance"])
        if "note" in patch:
            self.note = str(patch["note"] or "")

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not APPLICATION_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid application id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"Invalid code id: {self.code_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        if not CODER_ID_RE.match(self.coder_id):
            raise ProjectValidationError(
                f"Invalid coder id: {self.coder_id!r}"
            )
        if not CODE_VERSION_ID_RE.match(self.definition_version_id_at_apply):
            raise ProjectValidationError(
                f"Invalid definition_version_id_at_apply: "
                f"{self.definition_version_id_at_apply!r}"
            )

        # Anchor word ids: shape + ordering. parse_word_id raises on
        # malformed input; compare_word_ids enforces start ≤ end.
        sa_seg, sa_word = parse_word_id(self.anchor_start_word_id)
        ea_seg, ea_word = parse_word_id(self.anchor_end_word_id)
        if (sa_seg, sa_word) > (ea_seg, ea_word):
            raise ProjectValidationError(
                f"anchor_start_word_id must be ≤ anchor_end_word_id; "
                f"got {self.anchor_start_word_id!r} > "
                f"{self.anchor_end_word_id!r}"
            )

        # Sub-word offsets: optional non-negative ints, bounded.
        if self.start_char_offset is not None:
            if (
                not isinstance(self.start_char_offset, int)
                or isinstance(self.start_char_offset, bool)
            ):
                raise ProjectValidationError(
                    "start_char_offset must be an integer or null"
                )
            if self.start_char_offset < 0:
                raise ProjectValidationError(
                    f"start_char_offset must be ≥ 0; "
                    f"got {self.start_char_offset}"
                )
            if self.start_char_offset > MAX_CHAR_OFFSET:
                raise ProjectValidationError(
                    f"start_char_offset must be ≤ {MAX_CHAR_OFFSET}; "
                    f"got {self.start_char_offset}"
                )
        if self.end_char_offset is not None:
            if (
                not isinstance(self.end_char_offset, int)
                or isinstance(self.end_char_offset, bool)
            ):
                raise ProjectValidationError(
                    "end_char_offset must be an integer or null"
                )
            if self.end_char_offset < 0:
                raise ProjectValidationError(
                    f"end_char_offset must be ≥ 0; "
                    f"got {self.end_char_offset}"
                )
            if self.end_char_offset > MAX_CHAR_OFFSET:
                raise ProjectValidationError(
                    f"end_char_offset must be ≤ {MAX_CHAR_OFFSET}; "
                    f"got {self.end_char_offset}"
                )
        # If start and end ids point at the *same* word and both
        # offsets are set, start must be < end (an empty span has no
        # meaning).
        if (
            self.anchor_start_word_id == self.anchor_end_word_id
            and self.start_char_offset is not None
            and self.end_char_offset is not None
            and self.start_char_offset >= self.end_char_offset
        ):
            raise ProjectValidationError(
                "On a single-word anchor, start_char_offset must be < "
                f"end_char_offset; got {self.start_char_offset} >= "
                f"{self.end_char_offset}"
            )

        # Confidence: optional float in [0, 1].
        if self.confidence is not None:
            if (
                not isinstance(self.confidence, (int, float))
                or isinstance(self.confidence, bool)
            ):
                raise ProjectValidationError(
                    "confidence must be a number in [0, 1] or null"
                )
            self.confidence = float(self.confidence)
            if not (0.0 <= self.confidence <= 1.0):
                raise ProjectValidationError(
                    f"confidence must be in [0, 1]; got {self.confidence}"
                )

        # Provenance: free-form dict bounded by shape limits.
        if not isinstance(self.provenance, dict):
            raise ProjectValidationError(
                "provenance must be an object of string→string"
            )
        if len(self.provenance) > MAX_PROVENANCE_KEYS:
            raise ProjectValidationError(
                f"At most {MAX_PROVENANCE_KEYS} provenance keys allowed"
            )
        cleaned_prov: dict[str, str] = {}
        for raw_k, raw_v in self.provenance.items():
            k = str(raw_k).strip()
            if not k:
                continue
            if not PROVENANCE_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"provenance key {k!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            v = str(raw_v)
            if len(v) > MAX_PROVENANCE_VALUE_LEN:
                raise ProjectValidationError(
                    f"provenance[{k!r}] value too long "
                    f"(>{MAX_PROVENANCE_VALUE_LEN})"
                )
            cleaned_prov[k] = v
        if "source" in cleaned_prov:
            if cleaned_prov["source"] not in APPLICATION_PROVENANCE_SOURCES:
                raise ProjectValidationError(
                    f"provenance.source must be one of "
                    f"{APPLICATION_PROVENANCE_SOURCES}; "
                    f"got {cleaned_prov['source']!r}"
                )
        self.provenance = cleaned_prov

        # Structured AI provenance (F8.9): optional dataclass.
        if self.ai_provenance is not None:
            if not isinstance(self.ai_provenance, AIProvenance):
                raise ProjectValidationError(
                    "ai_provenance must be an AIProvenance instance or null"
                )
            self.ai_provenance.validate()

        if len(self.note) > MAX_NOTE_LEN:
            raise ProjectValidationError(
                f"note must be ≤ {MAX_NOTE_LEN} chars"
            )


# Fields a PATCH may set. Cross-entity ids and ``id``/timestamps are
# entity-managed; passing them is allowed (and ignored) so a client can
# round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "anchor_start_word_id",
    "anchor_end_word_id",
    "start_char_offset",
    "end_char_offset",
    "confidence",
    "provenance",
    "ai_provenance",
    "note",
}
_IGNORED_PATCH_KEYS = {
    "id",
    "project_id",
    "code_id",
    "source_id",
    "coder_id",
    "definition_version_id_at_apply",
    "created_at",
    "modified_at",
}


# --------------------------------------------------------------------------- #
# Optional-numeric coercion helpers
# --------------------------------------------------------------------------- #


def _coerce_ai_provenance(
    v: "AIProvenance | dict[str, Any] | None",
) -> "AIProvenance | None":
    """Coerce ``v`` into an :class:`AIProvenance` or ``None``.

    Accepts the dataclass (returned as-is after validation), a dict
    (passed to ``AIProvenance.from_dict``), or ``None`` / ``""``.
    Anything else is a clean :class:`ProjectValidationError`.
    """
    if v is None or v == "":
        return None
    if isinstance(v, AIProvenance):
        v.validate()
        return v
    if isinstance(v, dict):
        return AIProvenance.from_dict(v)
    raise ProjectValidationError(
        "ai_provenance must be an AIProvenance, an object, or null; got "
        f"{type(v).__name__}"
    )


def _optional_int(v: Any, field_name: str) -> int | None:
    """Coerce ``v`` to ``int | None``; raise on garbage.

    JSON numbers arrive as int or float; we accept both as long as the
    float has an integer value, matching what the in-memory dataclass
    expects.
    """
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        # bool is an int subclass; rule it out so we don't quietly turn
        # ``True`` into ``1`` for a char-offset.
        raise ProjectValidationError(
            f"{field_name} must be an integer or null; got bool"
        )
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ProjectValidationError(
                f"{field_name} must be an integer or null; got {v!r}"
            )
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as e:
            raise ProjectValidationError(
                f"{field_name} must be an integer or null; got {v!r}"
            ) from e
    raise ProjectValidationError(
        f"{field_name} must be an integer or null; got "
        f"{type(v).__name__}"
    )


def _optional_float(v: Any, field_name: str) -> float | None:
    """Coerce ``v`` to ``float | None``; raise on garbage."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise ProjectValidationError(
            f"{field_name} must be a number or null; got bool"
        )
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError as e:
            raise ProjectValidationError(
                f"{field_name} must be a number or null; got {v!r}"
            ) from e
    raise ProjectValidationError(
        f"{field_name} must be a number or null; got {type(v).__name__}"
    )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_application_id() -> str:
    """Mint a new 12-char hex application id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def applications_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's applications.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "applications"


def application_state_path(
    projects_root: Path, project_id: str, application_id: str
) -> Path:
    """Return the path of a single application's JSON file."""
    if not APPLICATION_ID_RE.match(application_id):
        raise ProjectValidationError(
            f"Invalid application id: {application_id!r}"
        )
    return applications_dir(projects_root, project_id) / f"{application_id}.json"


def save_application(projects_root: Path, application: Application) -> Path:
    """Persist an application to ``<projects_root>/<pid>/applications/<aid>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). Mirrors the convention of
    ``save_source`` / ``save_code`` / ``save_coder`` — an application
    without a project is meaningless and we surface that early.
    """
    application.validate()
    parent = project_dir(projects_root, application.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its applications."
        )
    ad = applications_dir(projects_root, application.project_id)
    ad.mkdir(parents=True, exist_ok=True)
    target = application_state_path(
        projects_root, application.project_id, application.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(application.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_application(
    projects_root: Path, project_id: str, application_id: str
) -> Application:
    """Load an application by id. Raises ``FileNotFoundError`` if missing."""
    p = application_state_path(projects_root, project_id, application_id)
    if not p.exists():
        raise FileNotFoundError(f"No application at {p}")
    return Application.from_dict(json.loads(p.read_text()))


def list_applications(
    projects_root: Path,
    project_id: str,
    *,
    code_id: str | None = None,
    source_id: str | None = None,
    coder_id: str | None = None,
) -> list[Application]:
    """List all applications in a project, optionally filtered.

    ``code_id`` / ``source_id`` / ``coder_id`` are AND-combined when
    multiple are passed. Skips files that don't parse as a valid
    Application so a single corrupt file doesn't break the view (audit
    log will eventually surface this — F9.7). Sorted by ``created_at``
    ascending so the natural reading order matches when applications
    were made (matching the audit-trail story).
    """
    if code_id is not None and not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(
            f"Invalid code id filter: {code_id!r}"
        )
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    if coder_id is not None and not CODER_ID_RE.match(coder_id):
        raise ProjectValidationError(
            f"Invalid coder id filter: {coder_id!r}"
        )
    ad = applications_dir(projects_root, project_id)
    if not ad.exists():
        return []
    out: list[Application] = []
    for f in sorted(ad.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        aid = f.stem
        if not APPLICATION_ID_RE.match(aid):
            continue
        try:
            a = Application.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if code_id is not None and a.code_id != code_id:
            continue
        if source_id is not None and a.source_id != source_id:
            continue
        if coder_id is not None and a.coder_id != coder_id:
            continue
        out.append(a)
    out.sort(key=lambda a: (a.created_at, a.id))
    return out


def delete_application(
    projects_root: Path, project_id: str, application_id: str
) -> bool:
    """Remove an application file. Returns False if it didn't exist.

    Hard delete is exposed because applications, unlike Codes (F2.3),
    don't have a separate retire-vs-delete distinction at the F4.1
    layer. F9.1's event log will record the deletion for audit. Future
    features may add a soft-delete; F4.1 keeps it simple.
    """
    p = application_state_path(projects_root, project_id, application_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
