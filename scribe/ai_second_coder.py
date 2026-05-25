"""AI second-coder pass on a locked codebook (F8.7).

Per PLANNING.md F8.7:

  > AI second-coder pass on a locked codebook. Diffs against human
  > coding; ICR view.

Methodologically, this feature exists for a single, narrow purpose: once
a codebook is **locked** (F2.4 — the researcher has declared the
codebook stable enough for final coding), the AI is asked to *re-code*
the same transcript in parallel, and we compute inter-coder reliability
(F2.5 Cohen's kappa) between the AI's coding and the human's coding.
The output is a diagnostic, not a coding action: it surfaces where the
AI and the human disagree, which is exactly the conversation a
methodologically-careful coder wants to have with their second coder.

What this module does
---------------------

1. **Guard the lock.** :func:`start_second_coder_pass` refuses to run
   on an unlocked codebook. The whole point of F8.7 is to validate a
   *locked* codebook against a parallel coder; running it on a still-
   evolving codebook would produce numbers that cannot be interpreted
   as agreement against a stable specification.

2. **Reuse F8.6.** A second-coder pass is, mechanically, an F8.6 review
   pass with ``skip_already_coded=False`` (we want a *parallel* coding,
   not a gap-fill). We build the inner :class:`scribe.transcript_review.ReviewPass`,
   reference it by id on the outer :class:`SecondCoderPass`, and let
   F8.6 do the per-item suggestion work. F8.7 layers diff + ICR on top.

3. **Diff against a chosen human coder.** A project may have several
   human coders (F2.5); the second-coder pass scores against *one*
   designated coder's applications. We take their applications,
   overlap-test each one against each review item's anchor range, and
   produce a per-item set of "human-applied codes". The AI's
   counterpart set is the suggestion's top-N candidates with
   ``combined_score >= min_score``.

4. **Compute Cohen's kappa per code** via :mod:`scribe.icr`. Each review
   item is one ICR "item"; for every code in the universe (AI ∪ human),
   we ask "did AI apply this code here? did human?". That's a binary
   per-code decision per item — the canonical input to per-code kappa.
   We also compute a flattened overall kappa over the multi-label
   encoding for an at-a-glance summary.

5. **Persist the diff + ICR.** A :class:`SecondCoderPass` record
   accumulates the metadata (which review pass it wrapped, who the
   human coder was, top_n / min_score thresholds) and the computed
   ``icr_results`` once the inner review pass completes. The diff
   itself is recomputed lazily from the suggestions + applications
   each time it's needed — the suggestions are the immutable record;
   re-running the diff after, say, a human-coder edit lets the
   researcher see how the picture has shifted.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.7 is the engine; the
  ``/api/projects/<id>/second-coder-passes`` routes will be a thin
  shell over this module, mirroring the F8.6 split.
* **No automatic application creation.** The AI's "applied codes" are
  *interpretations of the suggestion record*; no new
  :class:`scribe.applications.Application` is ever written by F8.7.
  Suggestions are themselves audit-grade records of an AI invocation
  (F8.3); F8.7 is a diff layer over those records.
* **Lock check happens once, at start.** If the codebook gets unlocked
  mid-pass, computing the diff still works (it's a function over the
  suggestions + applications); only :func:`start_second_coder_pass`
  refuses on an unlocked codebook. Audit-trail integrity stays with
  the F2.4 unlock memo.

Conventions match the rest of the F-feature stack
(:mod:`scribe.transcript_review`, :mod:`scribe.code_suggestions`,
:mod:`scribe.codebook_lock`, :mod:`scribe.icr`).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .applications import Application, parse_word_id
from .code_suggestions import (
    SUGGESTION_ID_RE,
    CodeSuggestion,
    EmbedFn,
    GenerateFn,
    load_suggestion,
)
from .codebook_lock import is_codebook_locked
from .coders import CODER_ID_RE
from .codes import Code
from .icr import (
    ICRError,
    cohens_kappa,
    expected_agreement,
    interpret_kappa,
    observed_agreement,
    per_code_kappa,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .sources import SOURCE_ID_RE
from .transcript_review import (
    REVIEW_GRANULARITIES,
    REVIEW_GRANULARITY_PARAGRAPH,
    REVIEW_PASS_ID_RE,
    REVIEW_STATUS_COMPLETED,
    REVIEW_TERMINAL_STATUSES,
    ReviewPass,
    cancel_review_pass,
    mark_review_pass_failed,
    process_next_review_item,
    start_review_pass,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# 12-char hex, same shape as every other id in Scribe.
SECOND_CODER_PASS_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding pass records.
SECOND_CODER_PASSES_DIRNAME = "second_coder_passes"

# Pass-level lifecycle states. We mirror ReviewPass closely so the
# state machine reads identically; the wrapper just adds a few
# bookkeeping fields.
SECOND_CODER_STATUS_PENDING = "pending"
SECOND_CODER_STATUS_RUNNING = "running"
SECOND_CODER_STATUS_COMPLETED = "completed"
SECOND_CODER_STATUS_CANCELLED = "cancelled"
SECOND_CODER_STATUS_FAILED = "failed"
SECOND_CODER_STATUSES: tuple[str, ...] = (
    SECOND_CODER_STATUS_PENDING,
    SECOND_CODER_STATUS_RUNNING,
    SECOND_CODER_STATUS_COMPLETED,
    SECOND_CODER_STATUS_CANCELLED,
    SECOND_CODER_STATUS_FAILED,
)
SECOND_CODER_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        SECOND_CODER_STATUS_COMPLETED,
        SECOND_CODER_STATUS_CANCELLED,
        SECOND_CODER_STATUS_FAILED,
    }
)

# Defaults for the diff thresholds. ``top_n=1`` matches the most
# common ICR convention ("the AI's pick"). ``min_score=0.0`` accepts
# any candidate the engine returned; tighten via the kwarg.
DEFAULT_TOP_N = 1
DEFAULT_MIN_SCORE = 0.0

# Bounds. Generous, but bounded so a stray upstream bug can't write a
# 50 MB pass record.
MAX_NOTES_LEN = 4000
MAX_ERROR_MESSAGE_LEN = 4000


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class CodebookNotLockedError(ProjectValidationError):
    """Raised when a second-coder pass is started on an unlocked codebook.

    Subclasses :class:`ProjectValidationError` so HTTP layers map it to
    400 by default. The message string is the canonical user-facing
    text — methodologically explicit about *why* the operation is
    refused.
    """


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_second_coder_pass_id() -> str:
    """Mint a fresh 12-char hex id for a second-coder pass record."""
    return uuid.uuid4().hex[:12]


def is_terminal_second_coder_status(status: str) -> bool:
    """Return True if ``status`` is one of the closed-set terminal states."""
    return status in SECOND_CODER_TERMINAL_STATUSES


def _coerce_int(v: Any, label: str, *, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be an integer; got {v!r}"
        ) from e


def _coerce_float(v: Any, label: str, *, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be numeric; got {v!r}"
        ) from e


def _word_position(word_id: str) -> tuple[int, int]:
    """Return the (segment_index, word_index) tuple for an anchor."""
    return parse_word_id(word_id)


def _anchor_overlap(
    a_start: str, a_end: str, b_start: str, b_end: str
) -> bool:
    """Return True iff anchor [a_start, a_end] overlaps [b_start, b_end].

    Both ranges are inclusive on both ends. Comparison is by
    ``(segment_index, word_index)`` tuple ordering. Different sources
    are checked by the caller; this helper is purely positional.
    """
    a_lo = _word_position(a_start)
    a_hi = _word_position(a_end)
    b_lo = _word_position(b_start)
    b_hi = _word_position(b_end)
    if a_lo > a_hi:
        a_lo, a_hi = a_hi, a_lo
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo
    # Inclusive overlap: max(a_lo, b_lo) <= min(a_hi, b_hi).
    return max(a_lo, b_lo) <= min(a_hi, b_hi)


# --------------------------------------------------------------------------- #
# Diff data model
# --------------------------------------------------------------------------- #


@dataclass
class SecondCoderItemDiff:
    """One review item's AI-vs-human diff.

    ``ai_code_ids`` is the set of code ids the AI "applied" at this
    span (top-N candidates with combined_score >= min_score, ordered by
    score). ``human_code_ids`` is the set of code ids the designated
    human coder applied to applications that overlap this span.

    The two are stored as **lists** to keep on-disk order deterministic
    (sorted by code id) but semantically represent sets — duplicates
    are dropped on construction.
    """

    item_index: int
    anchor_start_word_id: str
    anchor_end_word_id: str
    paragraph_start_segment: int
    paragraph_end_segment: int
    suggestion_id: str
    ai_code_ids: list[str] = field(default_factory=list)
    human_code_ids: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_index": int(self.item_index),
            "anchor_start_word_id": self.anchor_start_word_id,
            "anchor_end_word_id": self.anchor_end_word_id,
            "paragraph_start_segment": int(self.paragraph_start_segment),
            "paragraph_end_segment": int(self.paragraph_end_segment),
            "suggestion_id": self.suggestion_id,
            "ai_code_ids": list(self.ai_code_ids),
            "human_code_ids": list(self.human_code_ids),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SecondCoderItemDiff":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SecondCoderItemDiff payload must be an object"
            )
        item = cls(
            item_index=_coerce_int(d.get("item_index"), "item_index"),
            anchor_start_word_id=str(d.get("anchor_start_word_id", "") or ""),
            anchor_end_word_id=str(d.get("anchor_end_word_id", "") or ""),
            paragraph_start_segment=_coerce_int(
                d.get("paragraph_start_segment"), "paragraph_start_segment"
            ),
            paragraph_end_segment=_coerce_int(
                d.get("paragraph_end_segment"), "paragraph_end_segment"
            ),
            suggestion_id=str(d.get("suggestion_id", "") or ""),
            ai_code_ids=[str(c) for c in (d.get("ai_code_ids") or [])],
            human_code_ids=[str(c) for c in (d.get("human_code_ids") or [])],
            error=str(d.get("error", "") or ""),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.item_index < 0:
            raise ProjectValidationError("item_index must be ≥ 0")
        # Anchor shape, when set, must parse. (Empty anchors are allowed
        # for pass-level error rows that never produced a suggestion.)
        if self.anchor_start_word_id:
            parse_word_id(self.anchor_start_word_id)
        if self.anchor_end_word_id:
            parse_word_id(self.anchor_end_word_id)
        if self.paragraph_start_segment < 0:
            raise ProjectValidationError("paragraph_start_segment must be ≥ 0")
        if self.paragraph_end_segment < self.paragraph_start_segment:
            raise ProjectValidationError(
                "paragraph_end_segment must be ≥ paragraph_start_segment"
            )
        if self.suggestion_id and not SUGGESTION_ID_RE.match(self.suggestion_id):
            raise ProjectValidationError(
                f"suggestion_id must be 12-char hex or empty; "
                f"got {self.suggestion_id!r}"
            )
        for c in self.ai_code_ids:
            if not isinstance(c, str) or not c:
                raise ProjectValidationError(
                    "ai_code_ids entries must be non-empty strings"
                )
        for c in self.human_code_ids:
            if not isinstance(c, str) or not c:
                raise ProjectValidationError(
                    "human_code_ids entries must be non-empty strings"
                )

    @property
    def agreement_codes(self) -> list[str]:
        """Codes both sides applied here (sorted)."""
        return sorted(set(self.ai_code_ids) & set(self.human_code_ids))

    @property
    def ai_only_codes(self) -> list[str]:
        """Codes the AI applied that the human didn't (sorted)."""
        return sorted(set(self.ai_code_ids) - set(self.human_code_ids))

    @property
    def human_only_codes(self) -> list[str]:
        """Codes the human applied that the AI didn't (sorted)."""
        return sorted(set(self.human_code_ids) - set(self.ai_code_ids))


@dataclass
class SecondCoderDiff:
    """The full diff for a second-coder pass: one entry per review item."""

    items: list[SecondCoderItemDiff] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [it.to_dict() for it in self.items]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SecondCoderDiff":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SecondCoderDiff payload must be an object"
            )
        return cls(
            items=[
                SecondCoderItemDiff.from_dict(it)
                for it in (d.get("items") or [])
            ]
        )


# --------------------------------------------------------------------------- #
# ICR data model
# --------------------------------------------------------------------------- #


@dataclass
class CodeICR:
    """Per-code ICR breakdown.

    All counts are item-level (one increment per review item where the
    relevant condition holds — multi-application of the same code on
    the same item is collapsed since the underlying ICR is binary
    "applied / not-applied").
    """

    code_id: str
    ai_count: int
    human_count: int
    both_count: int
    kappa: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "ai_count": int(self.ai_count),
            "human_count": int(self.human_count),
            "both_count": int(self.both_count),
            "kappa": float(self.kappa),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CodeICR":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "CodeICR payload must be an object"
            )
        return cls(
            code_id=str(d.get("code_id", "")),
            ai_count=_coerce_int(d.get("ai_count"), "ai_count"),
            human_count=_coerce_int(d.get("human_count"), "human_count"),
            both_count=_coerce_int(d.get("both_count"), "both_count"),
            kappa=_coerce_float(d.get("kappa"), "kappa"),
            interpretation=str(d.get("interpretation", "")),
        )


@dataclass
class SecondCoderICR:
    """Aggregate ICR results for a second-coder pass.

    ``per_code`` lists each code that either side applied at least
    once, with item-level counts and Cohen's kappa for that code's
    binary applied/not-applied decision.

    ``overall_kappa`` is computed by flattening every ``(item, code)``
    pair into a binary AI / human label and computing Cohen's kappa
    on the two parallel binary lists. It's a single-number summary
    suitable for "did AI and human agree overall?" but the per-code
    breakdown is what a methodologically careful researcher will look
    at first.

    ``n_items`` is the count of items that contributed to the ICR
    (items with a successful suggestion). Items with errors are not
    included.
    """

    n_items: int
    n_codes: int
    overall_observed_agreement: float
    overall_expected_agreement: float
    overall_kappa: float
    overall_interpretation: str
    items_with_full_agreement: int
    items_with_any_disagreement: int
    per_code: list[CodeICR] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": int(self.n_items),
            "n_codes": int(self.n_codes),
            "overall_observed_agreement": float(self.overall_observed_agreement),
            "overall_expected_agreement": float(self.overall_expected_agreement),
            "overall_kappa": float(self.overall_kappa),
            "overall_interpretation": self.overall_interpretation,
            "items_with_full_agreement": int(self.items_with_full_agreement),
            "items_with_any_disagreement": int(self.items_with_any_disagreement),
            "per_code": [c.to_dict() for c in self.per_code],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SecondCoderICR":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SecondCoderICR payload must be an object"
            )
        return cls(
            n_items=_coerce_int(d.get("n_items"), "n_items"),
            n_codes=_coerce_int(d.get("n_codes"), "n_codes"),
            overall_observed_agreement=_coerce_float(
                d.get("overall_observed_agreement"),
                "overall_observed_agreement",
            ),
            overall_expected_agreement=_coerce_float(
                d.get("overall_expected_agreement"),
                "overall_expected_agreement",
            ),
            overall_kappa=_coerce_float(
                d.get("overall_kappa"), "overall_kappa"
            ),
            overall_interpretation=str(
                d.get("overall_interpretation", "") or ""
            ),
            items_with_full_agreement=_coerce_int(
                d.get("items_with_full_agreement"),
                "items_with_full_agreement",
            ),
            items_with_any_disagreement=_coerce_int(
                d.get("items_with_any_disagreement"),
                "items_with_any_disagreement",
            ),
            per_code=[
                CodeICR.from_dict(c) for c in (d.get("per_code") or [])
            ],
        )


# --------------------------------------------------------------------------- #
# SecondCoderPass dataclass
# --------------------------------------------------------------------------- #


@dataclass
class SecondCoderPass:
    """One AI second-coder run, persisted as the audit record.

    A second-coder pass references a single F8.6 :class:`ReviewPass`
    by id (``review_pass_id``) and records the human side of the
    diff (``human_coder_id``) plus the thresholds used to decide
    which AI candidates count as "applied" (``top_n``, ``min_score``).

    Status moves ``pending → running → completed | cancelled |
    failed`` and never flows out of a terminal state. Computed ICR
    results land in ``icr_results`` once the inner review pass
    completes; until then ``icr_results`` is an empty dict.
    """

    id: str
    project_id: str
    source_id: str
    human_coder_id: str
    review_pass_id: str = ""
    status: str = SECOND_CODER_STATUS_PENDING
    granularity: str = REVIEW_GRANULARITY_PARAGRAPH
    top_n: int = DEFAULT_TOP_N
    min_score: float = DEFAULT_MIN_SCORE
    embedding_model: str = ""
    generation_model: str = ""
    error_message: str = ""
    notes: str = ""
    icr_results: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    modified_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        source_id: str,
        human_coder_id: str,
        review_pass_id: str = "",
        granularity: str = REVIEW_GRANULARITY_PARAGRAPH,
        top_n: int = DEFAULT_TOP_N,
        min_score: float = DEFAULT_MIN_SCORE,
        embedding_model: str = "",
        generation_model: str = "",
        notes: str = "",
        pass_id: str | None = None,
        now: str | None = None,
    ) -> "SecondCoderPass":
        ts = now or utcnow_iso()
        p = cls(
            id=pass_id or new_second_coder_pass_id(),
            project_id=str(project_id),
            source_id=str(source_id),
            human_coder_id=str(human_coder_id),
            review_pass_id=str(review_pass_id or ""),
            status=SECOND_CODER_STATUS_PENDING,
            granularity=str(granularity),
            top_n=int(top_n),
            min_score=float(min_score),
            embedding_model=str(embedding_model or ""),
            generation_model=str(generation_model or ""),
            error_message="",
            notes=str(notes or ""),
            icr_results={},
            created_at=ts,
            modified_at=ts,
            started_at="",
            completed_at="",
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_id": self.source_id,
            "human_coder_id": self.human_coder_id,
            "review_pass_id": self.review_pass_id,
            "status": self.status,
            "granularity": self.granularity,
            "top_n": int(self.top_n),
            "min_score": float(self.min_score),
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "error_message": self.error_message,
            "notes": self.notes,
            "icr_results": dict(self.icr_results),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SecondCoderPass":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SecondCoderPass payload must be an object"
            )
        for required in ("id", "project_id", "source_id", "human_coder_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"SecondCoderPass payload missing required key: "
                    f"{required}"
                )
        raw_icr = d.get("icr_results")
        if raw_icr is None:
            raw_icr = {}
        elif not isinstance(raw_icr, Mapping):
            raise ProjectValidationError(
                "icr_results must be an object"
            )
        p = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            source_id=str(d["source_id"]),
            human_coder_id=str(d["human_coder_id"]),
            review_pass_id=str(d.get("review_pass_id", "") or ""),
            status=str(
                d.get("status", SECOND_CODER_STATUS_PENDING)
                or SECOND_CODER_STATUS_PENDING
            ),
            granularity=str(
                d.get("granularity", REVIEW_GRANULARITY_PARAGRAPH)
                or REVIEW_GRANULARITY_PARAGRAPH
            ),
            top_n=_coerce_int(
                d.get("top_n", DEFAULT_TOP_N), "top_n", default=DEFAULT_TOP_N
            ),
            min_score=_coerce_float(
                d.get("min_score", DEFAULT_MIN_SCORE),
                "min_score",
                default=DEFAULT_MIN_SCORE,
            ),
            embedding_model=str(d.get("embedding_model", "") or ""),
            generation_model=str(d.get("generation_model", "") or ""),
            error_message=str(d.get("error_message", "") or ""),
            notes=str(d.get("notes", "") or ""),
            icr_results=dict(raw_icr),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
            started_at=str(d.get("started_at", "") or ""),
            completed_at=str(d.get("completed_at", "") or ""),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not SECOND_CODER_PASS_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid second-coder pass id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        if not CODER_ID_RE.match(self.human_coder_id):
            raise ProjectValidationError(
                f"Invalid human coder id: {self.human_coder_id!r}"
            )
        if self.review_pass_id and not REVIEW_PASS_ID_RE.match(
            self.review_pass_id
        ):
            raise ProjectValidationError(
                f"review_pass_id must be 12-char hex or empty; "
                f"got {self.review_pass_id!r}"
            )
        if self.status not in SECOND_CODER_STATUSES:
            raise ProjectValidationError(
                f"status must be one of {SECOND_CODER_STATUSES}; "
                f"got {self.status!r}"
            )
        if self.granularity not in REVIEW_GRANULARITIES:
            raise ProjectValidationError(
                f"granularity must be one of {REVIEW_GRANULARITIES}; "
                f"got {self.granularity!r}"
            )
        if self.top_n < 1:
            raise ProjectValidationError(
                f"top_n must be ≥ 1; got {self.top_n}"
            )
        if not (-1.0 <= self.min_score <= 1.0):
            raise ProjectValidationError(
                f"min_score must be in [-1, 1]; got {self.min_score}"
            )
        if not isinstance(self.notes, str) or len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes must be a string ≤ {MAX_NOTES_LEN} chars"
            )
        if (
            not isinstance(self.error_message, str)
            or len(self.error_message) > MAX_ERROR_MESSAGE_LEN
        ):
            raise ProjectValidationError(
                f"error_message must be a string ≤ "
                f"{MAX_ERROR_MESSAGE_LEN} chars"
            )
        # Status / timestamp invariants — match ReviewPass.
        if self.status == SECOND_CODER_STATUS_PENDING:
            if self.started_at:
                raise ProjectValidationError(
                    "pending pass must not have started_at"
                )
            if self.completed_at:
                raise ProjectValidationError(
                    "pending pass must not have completed_at"
                )
        if self.status == SECOND_CODER_STATUS_RUNNING:
            if not self.started_at:
                raise ProjectValidationError(
                    "running pass must have started_at"
                )
            if self.completed_at:
                raise ProjectValidationError(
                    "running pass must not have completed_at"
                )
        if self.status in SECOND_CODER_TERMINAL_STATUSES:
            if not self.started_at:
                raise ProjectValidationError(
                    f"{self.status!r} pass must have started_at"
                )
            if not self.completed_at:
                raise ProjectValidationError(
                    f"{self.status!r} pass must have completed_at"
                )
        if (
            self.status != SECOND_CODER_STATUS_FAILED
            and self.error_message
        ):
            raise ProjectValidationError(
                "error_message must only be set when status='failed'"
            )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: Mapping[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update — currently only ``notes``.

        Status transitions go through dedicated lifecycle functions
        (:func:`cancel_second_coder_pass`, :func:`mark_second_coder_pass_failed`)
        so the invariants stay enforced; ``icr_results`` is set by
        :func:`compute_and_store_icr` after the inner review pass
        completes.
        """
        if not isinstance(patch, Mapping):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - {"notes"}
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")
        self.modified_at = now or utcnow_iso()
        self.validate()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def second_coder_passes_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's second-coder passes.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / SECOND_CODER_PASSES_DIRNAME


def second_coder_pass_state_path(
    projects_root: Path, project_id: str, pass_id: str
) -> Path:
    if not SECOND_CODER_PASS_ID_RE.match(pass_id):
        raise ProjectValidationError(
            f"Invalid second-coder pass id: {pass_id!r}"
        )
    return (
        second_coder_passes_dir(projects_root, project_id) / f"{pass_id}.json"
    )


def save_second_coder_pass(
    projects_root: Path, pass_record: SecondCoderPass
) -> Path:
    """Persist a second-coder pass atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    pass_record.validate()
    parent = project_dir(projects_root, pass_record.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving second-coder passes."
        )
    pd = second_coder_passes_dir(projects_root, pass_record.project_id)
    pd.mkdir(parents=True, exist_ok=True)
    target = second_coder_pass_state_path(
        projects_root, pass_record.project_id, pass_record.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(pass_record.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_second_coder_pass(
    projects_root: Path, project_id: str, pass_id: str
) -> SecondCoderPass:
    """Load a second-coder pass by id. Raises ``FileNotFoundError`` if missing."""
    p = second_coder_pass_state_path(projects_root, project_id, pass_id)
    if not p.exists():
        raise FileNotFoundError(f"No second-coder pass at {p}")
    return SecondCoderPass.from_dict(json.loads(p.read_text()))


def list_second_coder_passes(
    projects_root: Path,
    project_id: str,
    *,
    source_id: str | None = None,
    human_coder_id: str | None = None,
    status: str | None = None,
) -> list[SecondCoderPass]:
    """List second-coder passes in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse — a single
    corrupt file does not break the listing. Sorted by ``created_at``
    ascending so the natural reading order is "the order in which
    passes were started".
    """
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    if human_coder_id is not None and not CODER_ID_RE.match(human_coder_id):
        raise ProjectValidationError(
            f"Invalid human coder id filter: {human_coder_id!r}"
        )
    if status is not None and status not in SECOND_CODER_STATUSES:
        raise ProjectValidationError(
            f"Invalid status filter: {status!r}"
        )
    pd = second_coder_passes_dir(projects_root, project_id)
    if not pd.exists():
        return []
    out: list[SecondCoderPass] = []
    for f in sorted(pd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        rid = f.stem
        if not SECOND_CODER_PASS_ID_RE.match(rid):
            continue
        try:
            sp = SecondCoderPass.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if source_id is not None and sp.source_id != source_id:
            continue
        if human_coder_id is not None and sp.human_coder_id != human_coder_id:
            continue
        if status is not None and sp.status != status:
            continue
        out.append(sp)
    out.sort(key=lambda p: (p.created_at, p.id))
    return out


def delete_second_coder_pass(
    projects_root: Path, project_id: str, pass_id: str
) -> bool:
    """Remove a second-coder pass file. Returns False if it didn't exist.

    Production code should prefer keeping passes for the audit trail;
    deletion is exposed for tests and for the REFI-QDA import path
    (where a clean slate matters).
    """
    p = second_coder_pass_state_path(projects_root, project_id, pass_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True


# --------------------------------------------------------------------------- #
# Diff computation
# --------------------------------------------------------------------------- #


def _select_ai_codes_from_suggestion(
    suggestion: CodeSuggestion,
    *,
    top_n: int,
    min_score: float,
) -> list[str]:
    """Pick the AI's "applied codes" out of a suggestion's candidates.

    We sort by ``combined_score`` descending (stable in candidate
    order) and take up to ``top_n`` whose score is at least
    ``min_score``. Returns a list of code ids in score-descending
    order; duplicates (which shouldn't occur in a well-formed
    suggestion but are guarded against here) are dropped.
    """
    if top_n < 1:
        return []
    sorted_candidates = sorted(
        suggestion.candidates,
        key=lambda c: (-float(c.combined_score), c.code_id),
    )
    out: list[str] = []
    seen: set[str] = set()
    for c in sorted_candidates:
        if len(out) >= top_n:
            break
        if float(c.combined_score) < float(min_score):
            continue
        if c.code_id in seen:
            continue
        seen.add(c.code_id)
        out.append(c.code_id)
    return out


def _human_codes_for_anchor(
    *,
    applications: Sequence[Application],
    source_id: str,
    human_coder_id: str,
    anchor_start: str,
    anchor_end: str,
) -> list[str]:
    """Return the unique code ids the human coder applied that overlap.

    Same source only; same coder only; whole-word inclusive overlap on
    the anchor range. Sub-word char offsets are ignored at this layer
    — F8.7 ICR is computed at item granularity, not character
    granularity. Returns sorted unique code ids.
    """
    out: set[str] = set()
    for a in applications:
        if a.source_id != source_id:
            continue
        if a.coder_id != human_coder_id:
            continue
        if not _anchor_overlap(
            anchor_start, anchor_end,
            a.anchor_start_word_id, a.anchor_end_word_id,
        ):
            continue
        out.add(a.code_id)
    return sorted(out)


def compute_second_coder_diff(
    *,
    projects_root: Path,
    pass_record: SecondCoderPass,
    review_pass: ReviewPass,
    applications: Sequence[Application],
) -> SecondCoderDiff:
    """Build a :class:`SecondCoderDiff` from a completed review pass.

    For every review item:

    * Loads the persisted :class:`CodeSuggestion` (if any) and picks
      the AI's "applied codes" via :func:`_select_ai_codes_from_suggestion`.
    * Finds applications by ``pass_record.human_coder_id`` that
      overlap the item's anchor range and collects their code ids.
    * Records both sets on a :class:`SecondCoderItemDiff`.

    Items with errors (no suggestion produced) carry an ``error``
    note and empty AI-code list; the human side is still populated
    for transparency. Items in pending state are skipped — diffing a
    not-yet-processed item produces noise.
    """
    if review_pass.project_id != pass_record.project_id:
        raise ProjectValidationError(
            "review_pass.project_id does not match pass_record.project_id"
        )
    if review_pass.source_id != pass_record.source_id:
        raise ProjectValidationError(
            "review_pass.source_id does not match pass_record.source_id"
        )
    diff = SecondCoderDiff()
    for idx, item in enumerate(review_pass.items):
        # Skip not-yet-processed items entirely (they have no
        # suggestion to diff against; keeping them would hurt the
        # ICR's chance-agreement baseline by pretending the AI made
        # an "absent" choice on every code it never saw).
        if item.is_pending:
            continue
        ai_codes: list[str] = []
        suggestion_id = item.suggestion_id or ""
        if item.error:
            ai_codes = []
        elif suggestion_id:
            try:
                suggestion = load_suggestion(
                    projects_root, pass_record.project_id, suggestion_id
                )
            except (FileNotFoundError, json.JSONDecodeError):
                ai_codes = []
            else:
                ai_codes = _select_ai_codes_from_suggestion(
                    suggestion,
                    top_n=pass_record.top_n,
                    min_score=pass_record.min_score,
                )
        human_codes = _human_codes_for_anchor(
            applications=applications,
            source_id=pass_record.source_id,
            human_coder_id=pass_record.human_coder_id,
            anchor_start=item.anchor_start_word_id,
            anchor_end=item.anchor_end_word_id,
        )
        diff.items.append(
            SecondCoderItemDiff(
                item_index=idx,
                anchor_start_word_id=item.anchor_start_word_id,
                anchor_end_word_id=item.anchor_end_word_id,
                paragraph_start_segment=item.paragraph_start_segment,
                paragraph_end_segment=item.paragraph_end_segment,
                suggestion_id=suggestion_id,
                ai_code_ids=sorted(set(ai_codes)),
                human_code_ids=human_codes,
                error=item.error,
            )
        )
    return diff


# --------------------------------------------------------------------------- #
# ICR computation
# --------------------------------------------------------------------------- #


def compute_second_coder_icr(diff: SecondCoderDiff) -> SecondCoderICR:
    """Compute the ICR statistics for a :class:`SecondCoderDiff`.

    Strategy:

    1. Drop items with errors (no AI coding to diff against).
    2. Build per-item code sets for each side.
    3. Per code, treat each item as a binary decision and run
       Cohen's kappa via :func:`scribe.icr.per_code_kappa`.
    4. For the overall summary, flatten the universe to all
       ``(item, code)`` binary decisions and compute kappa over the
       parallel binary lists.

    Returns an empty result (zero items, zero codes) when the diff
    has no usable items — kappa requires at least one decision.
    """
    usable = [it for it in diff.items if not it.error]
    if not usable:
        return SecondCoderICR(
            n_items=0,
            n_codes=0,
            overall_observed_agreement=1.0,
            overall_expected_agreement=1.0,
            overall_kappa=1.0,
            overall_interpretation=interpret_kappa(1.0),
            items_with_full_agreement=0,
            items_with_any_disagreement=0,
            per_code=[],
        )

    # Build the {item_id: set(code_id)} dicts for per_code_kappa.
    ai_apps: dict[str, set[str]] = {}
    human_apps: dict[str, set[str]] = {}
    for it in usable:
        key = f"item-{it.item_index}"
        ai_apps[key] = set(it.ai_code_ids)
        human_apps[key] = set(it.human_code_ids)

    # Universe of codes either side touched.
    universe: set[str] = set()
    for s in ai_apps.values():
        universe |= s
    for s in human_apps.values():
        universe |= s
    code_list = sorted(universe)

    # Per-code kappa breakdown.
    item_keys = sorted(ai_apps.keys())
    per_code_results = per_code_kappa(
        ai_apps,
        human_apps,
        items=item_keys,
        codes=code_list,
    )
    per_code: list[CodeICR] = []
    for code_id in code_list:
        ai_count = sum(1 for k in item_keys if code_id in ai_apps[k])
        human_count = sum(1 for k in item_keys if code_id in human_apps[k])
        both_count = sum(
            1
            for k in item_keys
            if code_id in ai_apps[k] and code_id in human_apps[k]
        )
        kappa_val = float(per_code_results.get(code_id, 1.0))
        per_code.append(
            CodeICR(
                code_id=str(code_id),
                ai_count=ai_count,
                human_count=human_count,
                both_count=both_count,
                kappa=kappa_val,
                interpretation=interpret_kappa(kappa_val),
            )
        )

    # Overall (flattened) kappa: every (item, code) pair becomes one
    # binary decision per side. With no codes in the universe we skip
    # straight to "perfect agreement" (vacuously: nothing was applied
    # by either side, so every pair agrees on absence).
    if not code_list:
        return SecondCoderICR(
            n_items=len(usable),
            n_codes=0,
            overall_observed_agreement=1.0,
            overall_expected_agreement=1.0,
            overall_kappa=1.0,
            overall_interpretation=interpret_kappa(1.0),
            items_with_full_agreement=len(usable),
            items_with_any_disagreement=0,
            per_code=[],
        )

    a_flat: list[bool] = []
    b_flat: list[bool] = []
    for k in item_keys:
        a_set = ai_apps[k]
        b_set = human_apps[k]
        for code_id in code_list:
            a_flat.append(code_id in a_set)
            b_flat.append(code_id in b_set)

    try:
        obs = observed_agreement(a_flat, b_flat)
        exp = expected_agreement(a_flat, b_flat)
        overall_kappa = cohens_kappa(a_flat, b_flat)
    except ICRError as e:
        # Defensive: same lengths by construction; fall back to
        # neutral values rather than raising out of an analytic helper.
        raise ProjectValidationError(f"Overall kappa computation failed: {e}") from e

    # Item-level agreement: did every code agree at this item?
    full = 0
    disagree = 0
    for k in item_keys:
        if ai_apps[k] == human_apps[k]:
            full += 1
        else:
            disagree += 1

    return SecondCoderICR(
        n_items=len(usable),
        n_codes=len(code_list),
        overall_observed_agreement=float(obs),
        overall_expected_agreement=float(exp),
        overall_kappa=float(overall_kappa),
        overall_interpretation=interpret_kappa(overall_kappa),
        items_with_full_agreement=full,
        items_with_any_disagreement=disagree,
        per_code=per_code,
    )


def compute_and_store_icr(
    pass_record: SecondCoderPass,
    *,
    projects_root: Path,
    review_pass: ReviewPass,
    applications: Sequence[Application],
    now: str | None = None,
) -> SecondCoderICR:
    """Compute ICR for a pass, persist it onto the record, return the result.

    Calls :func:`compute_second_coder_diff` then
    :func:`compute_second_coder_icr` and stamps the JSON-serialised
    result onto ``pass_record.icr_results``. Caller saves the pass
    afterwards (or via :func:`run_second_coder_pass` which does it
    automatically).
    """
    diff = compute_second_coder_diff(
        projects_root=projects_root,
        pass_record=pass_record,
        review_pass=review_pass,
        applications=applications,
    )
    icr = compute_second_coder_icr(diff)
    pass_record.icr_results = icr.to_dict()
    pass_record.modified_at = now or utcnow_iso()
    return icr


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def start_second_coder_pass(
    *,
    projects_root: Path,
    project_id: str,
    source_id: str,
    human_coder_id: str,
    segments: Sequence[Mapping[str, Any]],
    applications: Sequence[Application] = (),
    granularity: str = REVIEW_GRANULARITY_PARAGRAPH,
    embedding_model: str = "",
    generation_model: str = "",
    top_n: int = DEFAULT_TOP_N,
    min_score: float = DEFAULT_MIN_SCORE,
    notes: str = "",
    pass_id: str | None = None,
    now: str | None = None,
) -> SecondCoderPass:
    """Build, persist, and return a fresh :class:`SecondCoderPass`.

    Refuses to run on an unlocked codebook — :class:`CodebookNotLockedError`
    surfaces the methodological constraint. Internally creates an
    F8.6 :class:`ReviewPass` with ``skip_already_coded=False`` (we
    want a parallel coding for the diff, not gap-filling) and
    references it by id on the outer pass record.

    The two pass records (review pass + second-coder pass) share
    timestamps via the ``now`` argument so audit views can correlate
    them.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    if not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(f"Invalid source id: {source_id!r}")
    if not CODER_ID_RE.match(human_coder_id):
        raise ProjectValidationError(
            f"Invalid human coder id: {human_coder_id!r}"
        )
    if granularity not in REVIEW_GRANULARITIES:
        raise ProjectValidationError(
            f"granularity must be one of {REVIEW_GRANULARITIES}; "
            f"got {granularity!r}"
        )
    # Lock guard. Per F8.7 spec, second-coder passes only run on
    # locked codebooks — agreement against an evolving codebook is
    # not a meaningful number.
    if not is_codebook_locked(projects_root, project_id):
        raise CodebookNotLockedError(
            f"Project {project_id!r} codebook is not locked. "
            "Lock the codebook (F2.4) before running an AI second-coder "
            "pass — agreement against an evolving codebook is not a "
            "meaningful number."
        )

    ts = now or utcnow_iso()

    # Inner review pass. We pass embedding/generation model names
    # through so persisted records share the same audit trail.
    review_pass = start_review_pass(
        projects_root=projects_root,
        project_id=project_id,
        source_id=source_id,
        segments=segments,
        applications=applications,
        granularity=granularity,
        skip_already_coded=False,
        embedding_model=embedding_model,
        generation_model=generation_model,
        notes=f"AI second-coder pass (F8.7) — human coder {human_coder_id}",
        now=ts,
    )

    pass_record = SecondCoderPass.new(
        project_id=project_id,
        source_id=source_id,
        human_coder_id=human_coder_id,
        review_pass_id=review_pass.id,
        granularity=granularity,
        top_n=top_n,
        min_score=min_score,
        embedding_model=embedding_model,
        generation_model=generation_model,
        notes=notes,
        pass_id=pass_id,
        now=ts,
    )
    save_second_coder_pass(projects_root, pass_record)
    return pass_record


def process_next_second_coder_item(
    pass_record: SecondCoderPass,
    *,
    projects_root: Path,
    review_pass: ReviewPass,
    codes: Sequence[Code],
    applications: Sequence[Application],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn | None = None,
    now: str | None = None,
) -> tuple[int, CodeSuggestion | None]:
    """Process the next pending item via the inner F8.6 review pass.

    Mutates both ``pass_record`` and ``review_pass`` in place.

    * On the first call from ``"pending"``, flips
      ``pass_record.status`` to ``"running"`` and stamps
      ``started_at`` (mirrors the inner review pass).
    * Delegates to :func:`scribe.transcript_review.process_next_review_item`
      for the per-item engine work.
    * When the inner review pass completes, computes ICR via
      :func:`compute_and_store_icr` and flips ``pass_record.status``
      to ``"completed"``.

    Raises :class:`ProjectValidationError` if called on a
    second-coder pass already in a terminal state.
    """
    if pass_record.status in SECOND_CODER_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot process: second-coder pass {pass_record.id} is in "
            f"terminal state {pass_record.status!r}"
        )
    ts = now or utcnow_iso()
    if pass_record.status == SECOND_CODER_STATUS_PENDING:
        pass_record.status = SECOND_CODER_STATUS_RUNNING
        pass_record.started_at = ts

    idx, suggestion = process_next_review_item(
        review_pass,
        projects_root=projects_root,
        codes=codes,
        applications=applications,
        embed_fn=embed_fn,
        generate_fn=generate_fn,
        now=ts,
    )

    # When the inner review pass completes, finalise the second-coder
    # pass too.
    if review_pass.status == REVIEW_STATUS_COMPLETED:
        compute_and_store_icr(
            pass_record,
            projects_root=projects_root,
            review_pass=review_pass,
            applications=applications,
            now=ts,
        )
        pass_record.status = SECOND_CODER_STATUS_COMPLETED
        pass_record.completed_at = ts

    pass_record.modified_at = ts
    save_second_coder_pass(projects_root, pass_record)
    return idx, suggestion


def run_second_coder_pass(
    pass_record: SecondCoderPass,
    *,
    projects_root: Path,
    review_pass: ReviewPass,
    codes: Sequence[Code],
    applications: Sequence[Application],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn | None = None,
    on_step: Any = None,
    max_steps: int | None = None,
    now: str | None = None,
) -> SecondCoderPass:
    """Drive a second-coder pass to completion (or up to ``max_steps``).

    Calls :func:`process_next_second_coder_item` repeatedly. ``on_step``
    is invoked after each item with ``(item_index, suggestion_or_None)``
    so a server can stream progress to a polling client.

    Returns the (mutated) ``pass_record``. Stops early if the pass
    enters a terminal state.
    """
    steps = 0
    while pass_record.status not in SECOND_CODER_TERMINAL_STATUSES:
        if max_steps is not None and steps >= max_steps:
            break
        if not review_pass.pending_indices():
            # No pending items left in the inner pass; let
            # process_next_review_item finish bookkeeping if it
            # hasn't, otherwise nothing more to do here.
            if review_pass.status == REVIEW_STATUS_COMPLETED:
                # Make sure ICR + outer status are updated even if the
                # last call landed on a stale ``running`` outer state.
                if pass_record.status != SECOND_CODER_STATUS_COMPLETED:
                    ts = now or utcnow_iso()
                    compute_and_store_icr(
                        pass_record,
                        projects_root=projects_root,
                        review_pass=review_pass,
                        applications=applications,
                        now=ts,
                    )
                    if pass_record.status == SECOND_CODER_STATUS_PENDING:
                        pass_record.started_at = ts
                        pass_record.status = SECOND_CODER_STATUS_RUNNING
                    pass_record.status = SECOND_CODER_STATUS_COMPLETED
                    pass_record.completed_at = ts
                    pass_record.modified_at = ts
                    save_second_coder_pass(projects_root, pass_record)
            break
        idx, suggestion = process_next_second_coder_item(
            pass_record,
            projects_root=projects_root,
            review_pass=review_pass,
            codes=codes,
            applications=applications,
            embed_fn=embed_fn,
            generate_fn=generate_fn,
            now=now,
        )
        if on_step is not None:
            on_step(idx, suggestion)
        steps += 1
    return pass_record


def cancel_second_coder_pass(
    pass_record: SecondCoderPass,
    *,
    projects_root: Path,
    review_pass: ReviewPass | None = None,
    now: str | None = None,
) -> None:
    """Move a non-terminal second-coder pass to ``"cancelled"`` and persist.

    No-op if already cancelled. Raises
    :class:`ProjectValidationError` for other terminal states (you
    can't cancel a completed or failed pass — that would rewrite the
    audit trail).

    When ``review_pass`` is supplied and it is not yet terminal, the
    inner review pass is cancelled too so worker threads polling it
    stop scheduling new items.
    """
    if pass_record.status == SECOND_CODER_STATUS_CANCELLED:
        return
    if pass_record.status in SECOND_CODER_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot cancel pass {pass_record.id}: already in terminal "
            f"state {pass_record.status!r}"
        )
    ts = now or utcnow_iso()
    if review_pass is not None and review_pass.status not in REVIEW_TERMINAL_STATUSES:
        cancel_review_pass(review_pass, projects_root=projects_root, now=ts)
    if not pass_record.started_at:
        pass_record.started_at = ts
    pass_record.status = SECOND_CODER_STATUS_CANCELLED
    pass_record.completed_at = ts
    pass_record.modified_at = ts
    save_second_coder_pass(projects_root, pass_record)


def mark_second_coder_pass_failed(
    pass_record: SecondCoderPass,
    *,
    projects_root: Path,
    error_message: str,
    review_pass: ReviewPass | None = None,
    now: str | None = None,
) -> None:
    """Move a non-terminal pass to ``"failed"`` and persist with reason.

    Use this for *pass-level* failures (the project disappeared
    mid-pass; the embedder backend cannot be reached at all).
    Per-item errors land on the inner :class:`ReviewItem` and don't
    abort the pass.
    """
    if pass_record.status in SECOND_CODER_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot fail pass {pass_record.id}: already in terminal "
            f"state {pass_record.status!r}"
        )
    msg = str(error_message or "")[:MAX_ERROR_MESSAGE_LEN]
    if not msg:
        raise ProjectValidationError(
            "error_message is required when failing a second-coder pass"
        )
    ts = now or utcnow_iso()
    if review_pass is not None and review_pass.status not in REVIEW_TERMINAL_STATUSES:
        mark_review_pass_failed(
            review_pass,
            projects_root=projects_root,
            error_message=msg,
            now=ts,
        )
    if not pass_record.started_at:
        pass_record.started_at = ts
    pass_record.status = SECOND_CODER_STATUS_FAILED
    pass_record.error_message = msg
    pass_record.completed_at = ts
    pass_record.modified_at = ts
    save_second_coder_pass(projects_root, pass_record)
