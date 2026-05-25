"""New-code suggestion engine for the academic-coding workflow (F8.4).

Per PLANNING.md F8.4:

  > "Suggest a *new* code" action — separate command, requires
  > explicit invocation. The phrasing in the UI nudges toward
  > gerund-form Charmaz-style names.

Where F8.3 ranks **existing** codes against a span, F8.4 generates
**new** code proposals — names, definitions, rationales — that the
codebook does *not* yet have. The two engines deliberately live in
separate modules and are surfaced as separate commands so a researcher
never accidentally invents a duplicate of an existing code with one
click.

What this module does
---------------------

1. **Find existing codes that are similar.** Embed the query span,
   then score every code in the codebook (via definitions + exemplars
   the same way F8.3 does, but here only to *avoid duplicates*). The
   short-list of "near-by" codes is included in the prompt as
   "**don't** propose any of these — they already exist".

2. **Ask the model for new code proposals.** A structured prompt
   requests a small JSON array. Each entry has a name (gerund-form
   strongly preferred — "negotiating identity" rather than "identity
   negotiation"), a one-sentence definition, a one-sentence rationale,
   and up to a few short quote excerpts from the span supporting the
   suggestion. The prompt makes the gerund nudge explicit but does
   *not* enforce it — Strauss/Corbin and Glaser users may name codes
   differently and we don't lock them out.

3. **Score how novel each proposal looks.** For each parsed proposal,
   embed its name + definition; compare to the project's existing
   codes; record ``nearest_existing_code_id`` and the cosine score.
   The UI can warn "this looks 0.92 similar to ``NavigatingChange``;
   are you sure it's a new code?".

4. **Persist the suggestion.** A :class:`NewCodeSuggestion` record
   captures the query, all proposals, the raw LLM response, and a
   decision lifecycle ``pending → accepted | modified | rejected``.
   Even rejected suggestions are kept — the AI invocation log (F9.6)
   needs them.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.4 is the engine; the
  ``/api/projects/<id>/new-code-suggestions`` routes are deferred
  and will be a thin shell over this module, mirroring the F8.3 split.
* **No automatic code creation.** Accepting a suggestion records
  ``accepted_proposal_index`` and (optionally) ``created_code_id`` —
  the actual ``Code`` is built by the caller via the F2.1 path. The
  decision recorder just stamps the audit trail.
* **Pure callables.** ``embed_fn`` and ``generate_fn`` are arbitrary
  callables (the F8.1 backend adapter wraps OllamaBackend into both).
  Tests stub them with deterministic functions; production wires them
  to the registered backend.

This module is stand-alone — no FastAPI, no engine imports — so the
data model can be tested in pure Python and reused by the CLI later.
Conventions match the rest of the F-feature stack
(:mod:`scribe.code_suggestions`, :mod:`scribe.codes`,
:mod:`scribe.embedding_index`).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codes import CODE_ID_RE, Code
from .coders import CODER_ID_RE
from .embedding_index import canonical_text, cosine_similarity
from .applications import parse_word_id
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


# Suggestion IDs follow the same 12-char hex shape as every other id.
NEW_CODE_SUGGESTION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding new-code suggestions.
NEW_CODE_SUGGESTIONS_DIRNAME = "new_code_suggestions"

# Decision lifecycle states. Same shape as F8.3 so a UI can render
# both lists with the same column widget.
NEW_CODE_DECISION_PENDING = "pending"
NEW_CODE_DECISION_ACCEPTED = "accepted"
NEW_CODE_DECISION_MODIFIED = "modified"
NEW_CODE_DECISION_REJECTED = "rejected"
NEW_CODE_DECISIONS: tuple[str, ...] = (
    NEW_CODE_DECISION_PENDING,
    NEW_CODE_DECISION_ACCEPTED,
    NEW_CODE_DECISION_MODIFIED,
    NEW_CODE_DECISION_REJECTED,
)
TERMINAL_NEW_CODE_DECISIONS: frozenset[str] = frozenset(
    {
        NEW_CODE_DECISION_ACCEPTED,
        NEW_CODE_DECISION_MODIFIED,
        NEW_CODE_DECISION_REJECTED,
    }
)

# Defaults. Tuned for the typical "one paragraph → 2-4 new code ideas"
# call. The model is asked for at most ``DEFAULT_NUM_PROPOSALS`` ideas
# so prompt + response stay small.
DEFAULT_NUM_PROPOSALS = 4
DEFAULT_NEAR_DUPLICATE_TOP_K = 5      # how many existing codes to show as "avoid duplicates"
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.5  # only show codes scoring ≥ this against the span
DEFAULT_DUPLICATE_WARN_THRESHOLD = 0.85  # nearest_existing_similarity ≥ this → UI warns

# Bounds. Generous, but bounded so a stray upstream bug can't write
# a 50 MB suggestion record.
MAX_QUERY_TEXT_LEN = 8000
MAX_NAME_LEN = 200
MAX_DEFINITION_LEN = 4000
MAX_RATIONALE_LEN = 2000
MAX_QUOTE_EXCERPT_LEN = 1000
MAX_QUOTE_EXCERPTS = 8
MAX_PROPOSALS_PERSISTED = 32
MAX_RAW_LLM_RESPONSE_LEN = 16 * 1024
MAX_REJECTION_REASON_LEN = 2000
MAX_NOTES_LEN = 4000

# Allowed callable signatures. Match the F8.3 / F8.2 shapes exactly so
# the same backend adapter can drive both engines.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]
GenerateFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Proposal data model
# --------------------------------------------------------------------------- #


# Heuristic gerund detector. We deliberately keep this simple: the
# *first* word ends in ``ing`` and is at least four characters
# (so "ring" / "thing" / "sing" / "king" don't count). We also skip
# words ending in "thing"/"ling"/"sting" only if they're the entire
# word — very defensive. False positives are fine; this is a UI hint,
# not a gate.
_GERUND_RE = re.compile(r"^[A-Za-z]{4,}ing\b", re.IGNORECASE)


def looks_like_gerund(name: str) -> bool:
    """Return True if ``name``'s first word is plausibly a gerund.

    Charmaz suggests gerund-form code names ("negotiating identity",
    "managing uncertainty"). This helper is a *hint* — the engine
    surfaces it on each proposal so a UI can render a small badge or
    nudge text. It's not a validator: a code named "Identity work"
    is perfectly legal and will simply have ``is_gerund=False``.
    """
    if not isinstance(name, str):
        return False
    s = name.strip()
    if not s:
        return False
    return bool(_GERUND_RE.match(s))


@dataclass
class NewCodeProposal:
    """One proposed new code, output by the LLM and post-processed.

    ``confidence`` is a number in ``[0, 1]`` reflecting how strongly
    the model thinks the proposal applies to the span; the engine
    clamps the model's raw value to this range and defaults to ``0.0``
    when the model omits it.

    ``nearest_existing_code_id`` / ``nearest_existing_similarity``
    record the best embedding match against the project's existing
    codes (definition + exemplars). High similarity is a duplicate
    warning, not an error — the engine never drops a proposal on
    similarity alone.
    """

    name: str
    definition: str = ""
    rationale: str = ""
    quote_excerpts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    nearest_existing_code_id: str | None = None
    nearest_existing_similarity: float = 0.0
    is_gerund: bool = False

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "definition": self.definition,
            "rationale": self.rationale,
            "quote_excerpts": list(self.quote_excerpts),
            "confidence": float(self.confidence),
            "nearest_existing_code_id": self.nearest_existing_code_id,
            "nearest_existing_similarity": float(
                self.nearest_existing_similarity
            ),
            "is_gerund": bool(self.is_gerund),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NewCodeProposal":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "NewCodeProposal payload must be an object"
            )
        if "name" not in d:
            raise ProjectValidationError(
                "NewCodeProposal payload missing required key: name"
            )
        excerpts_raw = d.get("quote_excerpts") or []
        if not isinstance(excerpts_raw, list):
            raise ProjectValidationError(
                "quote_excerpts must be a list of strings"
            )
        nearest_raw = d.get("nearest_existing_code_id")
        nearest = (
            str(nearest_raw)
            if (nearest_raw is not None and nearest_raw != "")
            else None
        )
        p = cls(
            name=str(d["name"] or ""),
            definition=str(d.get("definition", "") or ""),
            rationale=str(d.get("rationale", "") or ""),
            quote_excerpts=[str(e) for e in excerpts_raw],
            confidence=_coerce_score(d.get("confidence", 0.0), "confidence"),
            nearest_existing_code_id=nearest,
            nearest_existing_similarity=_coerce_score(
                d.get("nearest_existing_similarity", 0.0),
                "nearest_existing_similarity",
                lo=-1.0,
                hi=1.0,
            ),
            is_gerund=bool(d.get("is_gerund", False)),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ProjectValidationError("name must be a string")
        name = self.name.strip()
        if not name:
            raise ProjectValidationError("NewCodeProposal.name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"NewCodeProposal.name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist trimmed name so on-disk state is canonical.
        self.name = name

        if not isinstance(self.definition, str):
            raise ProjectValidationError("definition must be a string")
        if len(self.definition) > MAX_DEFINITION_LEN:
            raise ProjectValidationError(
                f"definition must be ≤ {MAX_DEFINITION_LEN} chars"
            )

        if not isinstance(self.rationale, str):
            raise ProjectValidationError("rationale must be a string")
        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise ProjectValidationError(
                f"rationale must be ≤ {MAX_RATIONALE_LEN} chars"
            )

        if not isinstance(self.quote_excerpts, list):
            raise ProjectValidationError(
                "quote_excerpts must be a list of strings"
            )
        if len(self.quote_excerpts) > MAX_QUOTE_EXCERPTS:
            raise ProjectValidationError(
                f"At most {MAX_QUOTE_EXCERPTS} quote_excerpts allowed"
            )
        cleaned: list[str] = []
        for raw in self.quote_excerpts:
            t = str(raw).strip()
            if not t:
                continue   # silently drop empties; less friction
            if len(t) > MAX_QUOTE_EXCERPT_LEN:
                raise ProjectValidationError(
                    f"quote excerpt too long (>{MAX_QUOTE_EXCERPT_LEN}): "
                    f"{t[:40]!r}…"
                )
            cleaned.append(t)
        self.quote_excerpts = cleaned

        if not _finite(self.confidence) or self.confidence < -0.0001 or self.confidence > 1.0001:
            raise ProjectValidationError(
                f"confidence must be in [0, 1]; got {self.confidence!r}"
            )
        # Clamp to canonical range to absorb rounding.
        self.confidence = max(0.0, min(1.0, self.confidence))

        if (
            self.nearest_existing_code_id is not None
            and not CODE_ID_RE.match(self.nearest_existing_code_id)
        ):
            raise ProjectValidationError(
                f"nearest_existing_code_id must be 12-char hex or null; "
                f"got {self.nearest_existing_code_id!r}"
            )
        if (
            not _finite(self.nearest_existing_similarity)
            or self.nearest_existing_similarity < -1.0001
            or self.nearest_existing_similarity > 1.0001
        ):
            raise ProjectValidationError(
                f"nearest_existing_similarity must be in [-1, 1]; "
                f"got {self.nearest_existing_similarity!r}"
            )

        # Recompute gerund hint from the current name, ignoring whatever
        # the caller stored. The on-disk flag is *derived* from the name
        # and we want it to stay consistent across name edits.
        self.is_gerund = looks_like_gerund(self.name)


# --------------------------------------------------------------------------- #
# Suggestion record
# --------------------------------------------------------------------------- #


@dataclass
class NewCodeSuggestion:
    """One new-code-suggestion invocation, persisted as the audit record.

    The anchor fields mirror :class:`scribe.applications.Application`
    so the span the user highlighted is preserved alongside the
    generated proposals. Decision lifecycle::

        pending → accepted | modified | rejected

    * ``accepted`` — the researcher picked one of the proposals as-is.
      ``accepted_proposal_index`` records which one. ``created_code_id``
      is the resulting :class:`scribe.codes.Code` (set by the caller).
    * ``modified`` — the researcher took a proposal as a starting point
      but edited the name / definition before saving it. Same fields
      as ``accepted``; the audit trail can show "AI proposed X, human
      saved Y".
    * ``rejected`` — none of the proposals were useful. Forbids
      ``accepted_proposal_index`` and ``created_code_id``.
    """

    id: str
    project_id: str
    source_id: str
    anchor_start_word_id: str
    anchor_end_word_id: str
    start_char_offset: int | None
    end_char_offset: int | None
    query_text: str
    embedding_model: str
    generation_model: str
    proposals: list[NewCodeProposal] = field(default_factory=list)
    decision: str = NEW_CODE_DECISION_PENDING
    decided_at: str = ""
    decided_by_coder_id: str = ""
    accepted_proposal_index: int | None = None
    created_code_id: str | None = None
    rejection_reason: str = ""
    notes: str = ""
    raw_llm_response: str = ""
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
        anchor_start_word_id: str,
        anchor_end_word_id: str,
        query_text: str,
        embedding_model: str = "",
        generation_model: str = "",
        proposals: Iterable[NewCodeProposal] | None = None,
        start_char_offset: int | None = None,
        end_char_offset: int | None = None,
        notes: str = "",
        raw_llm_response: str = "",
        suggestion_id: str | None = None,
        now: str | None = None,
    ) -> "NewCodeSuggestion":
        ts = now or utcnow_iso()
        coerced: list[NewCodeProposal] = []
        for p in proposals or ():
            if isinstance(p, NewCodeProposal):
                coerced.append(p)
            else:
                coerced.append(NewCodeProposal.from_dict(p))
        s = cls(
            id=suggestion_id or new_new_code_suggestion_id(),
            project_id=project_id,
            source_id=source_id,
            anchor_start_word_id=anchor_start_word_id,
            anchor_end_word_id=anchor_end_word_id,
            start_char_offset=start_char_offset,
            end_char_offset=end_char_offset,
            query_text=str(query_text or ""),
            embedding_model=str(embedding_model or ""),
            generation_model=str(generation_model or ""),
            proposals=coerced,
            decision=NEW_CODE_DECISION_PENDING,
            decided_at="",
            decided_by_coder_id="",
            accepted_proposal_index=None,
            created_code_id=None,
            rejection_reason="",
            notes=str(notes or ""),
            raw_llm_response=str(raw_llm_response or ""),
            created_at=ts,
            modified_at=ts,
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_id": self.source_id,
            "anchor_start_word_id": self.anchor_start_word_id,
            "anchor_end_word_id": self.anchor_end_word_id,
            "start_char_offset": self.start_char_offset,
            "end_char_offset": self.end_char_offset,
            "query_text": self.query_text,
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "proposals": [p.to_dict() for p in self.proposals],
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by_coder_id": self.decided_by_coder_id,
            "accepted_proposal_index": self.accepted_proposal_index,
            "created_code_id": self.created_code_id,
            "rejection_reason": self.rejection_reason,
            "notes": self.notes,
            "raw_llm_response": self.raw_llm_response,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NewCodeSuggestion":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "NewCodeSuggestion payload must be an object"
            )
        for required in (
            "id",
            "project_id",
            "source_id",
            "anchor_start_word_id",
            "anchor_end_word_id",
            "query_text",
        ):
            if required not in d:
                raise ProjectValidationError(
                    f"NewCodeSuggestion payload missing required key: {required}"
                )
        s = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            source_id=str(d["source_id"]),
            anchor_start_word_id=str(d["anchor_start_word_id"]),
            anchor_end_word_id=str(d["anchor_end_word_id"]),
            start_char_offset=_optional_int(
                d.get("start_char_offset"), "start_char_offset"
            ),
            end_char_offset=_optional_int(
                d.get("end_char_offset"), "end_char_offset"
            ),
            query_text=str(d.get("query_text", "") or ""),
            embedding_model=str(d.get("embedding_model", "") or ""),
            generation_model=str(d.get("generation_model", "") or ""),
            proposals=[
                NewCodeProposal.from_dict(p) for p in (d.get("proposals") or [])
            ],
            decision=str(
                d.get("decision", NEW_CODE_DECISION_PENDING)
                or NEW_CODE_DECISION_PENDING
            ),
            decided_at=str(d.get("decided_at", "") or ""),
            decided_by_coder_id=str(d.get("decided_by_coder_id", "") or ""),
            accepted_proposal_index=_optional_int(
                d.get("accepted_proposal_index"), "accepted_proposal_index"
            ),
            created_code_id=(
                str(d["created_code_id"])
                if d.get("created_code_id")
                else None
            ),
            rejection_reason=str(d.get("rejection_reason", "") or ""),
            notes=str(d.get("notes", "") or ""),
            raw_llm_response=str(d.get("raw_llm_response", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not NEW_CODE_SUGGESTION_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid new-code suggestion id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        sa_seg, sa_word = parse_word_id(self.anchor_start_word_id)
        ea_seg, ea_word = parse_word_id(self.anchor_end_word_id)
        if (sa_seg, sa_word) > (ea_seg, ea_word):
            raise ProjectValidationError(
                f"anchor_start_word_id must be ≤ anchor_end_word_id; "
                f"got {self.anchor_start_word_id!r} > "
                f"{self.anchor_end_word_id!r}"
            )
        if self.start_char_offset is not None and self.start_char_offset < 0:
            raise ProjectValidationError(
                "start_char_offset must be ≥ 0 if set"
            )
        if self.end_char_offset is not None and self.end_char_offset < 0:
            raise ProjectValidationError(
                "end_char_offset must be ≥ 0 if set"
            )
        if not isinstance(self.query_text, str):
            raise ProjectValidationError("query_text must be a string")
        if len(self.query_text) > MAX_QUERY_TEXT_LEN:
            raise ProjectValidationError(
                f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars"
            )
        if (
            not isinstance(self.embedding_model, str)
            or len(self.embedding_model) > 256
        ):
            raise ProjectValidationError(
                "embedding_model must be a string ≤ 256 chars"
            )
        if (
            not isinstance(self.generation_model, str)
            or len(self.generation_model) > 256
        ):
            raise ProjectValidationError(
                "generation_model must be a string ≤ 256 chars"
            )
        if len(self.proposals) > MAX_PROPOSALS_PERSISTED:
            raise ProjectValidationError(
                f"proposals exceeds {MAX_PROPOSALS_PERSISTED} entries"
            )
        for p in self.proposals:
            p.validate()
        if self.decision not in NEW_CODE_DECISIONS:
            raise ProjectValidationError(
                f"decision must be one of {NEW_CODE_DECISIONS}; "
                f"got {self.decision!r}"
            )
        if self.decision in TERMINAL_NEW_CODE_DECISIONS:
            if not self.decided_at:
                raise ProjectValidationError(
                    f"decided_at must be set when decision is "
                    f"{self.decision!r}"
                )
            if not self.decided_by_coder_id or not CODER_ID_RE.match(
                self.decided_by_coder_id
            ):
                raise ProjectValidationError(
                    "decided_by_coder_id must be a 12-char hex coder id "
                    f"when decision is {self.decision!r}"
                )
        if self.accepted_proposal_index is not None:
            if self.accepted_proposal_index < 0:
                raise ProjectValidationError(
                    "accepted_proposal_index must be ≥ 0 if set"
                )
            if self.accepted_proposal_index >= max(1, len(self.proposals)):
                # Allow == len-1 only when proposals is non-empty.
                if (
                    not self.proposals
                    or self.accepted_proposal_index >= len(self.proposals)
                ):
                    raise ProjectValidationError(
                        f"accepted_proposal_index {self.accepted_proposal_index} "
                        f"out of range for {len(self.proposals)} proposals"
                    )
        if (
            self.created_code_id is not None
            and not CODE_ID_RE.match(self.created_code_id)
        ):
            raise ProjectValidationError(
                f"created_code_id must be 12-char hex or null; "
                f"got {self.created_code_id!r}"
            )
        # Decision-specific invariants.
        if self.decision in (
            NEW_CODE_DECISION_ACCEPTED,
            NEW_CODE_DECISION_MODIFIED,
        ):
            if self.accepted_proposal_index is None:
                raise ProjectValidationError(
                    "accepted_proposal_index is required when decision "
                    f"is {self.decision!r}"
                )
        elif self.decision == NEW_CODE_DECISION_REJECTED:
            if (
                self.accepted_proposal_index is not None
                or self.created_code_id is not None
            ):
                raise ProjectValidationError(
                    "rejected suggestions must not record an accepted "
                    "proposal index or created code id"
                )
        if len(self.rejection_reason) > MAX_REJECTION_REASON_LEN:
            raise ProjectValidationError(
                f"rejection_reason exceeds {MAX_REJECTION_REASON_LEN} chars"
            )
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )
        if len(self.raw_llm_response) > MAX_RAW_LLM_RESPONSE_LEN:
            raise ProjectValidationError(
                f"raw_llm_response exceeds {MAX_RAW_LLM_RESPONSE_LEN} chars"
            )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: Mapping[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update in place. Mirrors F8.3's apply_update.

        Only ``notes``, ``rejection_reason``, ``created_code_id``, and
        ``accepted_proposal_index`` can be patched freely. Decision
        transitions go through :func:`record_new_code_decision` so the
        invariants stay enforced. Stamps ``modified_at`` on success.
        """
        if not isinstance(patch, Mapping):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")
        if "rejection_reason" in patch:
            self.rejection_reason = str(patch["rejection_reason"] or "")
        if "created_code_id" in patch:
            v = patch["created_code_id"]
            self.created_code_id = str(v) if v else None
        if "accepted_proposal_index" in patch:
            v = patch["accepted_proposal_index"]
            self.accepted_proposal_index = (
                None if v is None else int(v)
            )
        self.validate()
        self.modified_at = now or utcnow_iso()


_ALLOWED_PATCH_KEYS = {
    "notes",
    "rejection_reason",
    "created_code_id",
    "accepted_proposal_index",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _coerce_score(
    v: Any, label: str, *, lo: float = 0.0, hi: float = 1.0
) -> float:
    """Coerce ``v`` to float; raise on garbage; clamp to [lo, hi].

    The clamp is important because LLMs return values like ``1.05`` or
    ``-0.001`` due to rounding, and we don't want a validate() round-trip
    to fail on a third-decimal slip.
    """
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be numeric; got {v!r}"
        ) from e
    if not math.isfinite(f):
        raise ProjectValidationError(
            f"{label} must be finite; got {v!r}"
        )
    if f < lo - 0.01 or f > hi + 0.01:
        raise ProjectValidationError(
            f"{label} out of range [{lo}, {hi}]; got {v!r}"
        )
    return max(lo, min(hi, f))


def _optional_int(v: Any, label: str) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be an integer or null; got {v!r}"
        ) from e


def new_new_code_suggestion_id() -> str:
    """Mint a new 12-char hex new-code suggestion id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Existing-code similarity (the "don't propose duplicates" map)
# --------------------------------------------------------------------------- #


def _build_existing_corpus(
    codes: Sequence[Code],
) -> list[tuple[str, str, str]]:
    """Return ``(code_id, code_name, text)`` rows for similarity scoring.

    For each code we add one row with its name + definition concatenated
    (the most useful single-shot summary), and one row per non-empty
    exemplar. Empty / whitespace-only entries are skipped. The same row
    list drives both:

    * the prompt's "existing codes shortlist" (we group scores back to
      ``code_id`` and pick the max);
    * the proposal-side "nearest existing code" computation.
    """
    rows: list[tuple[str, str, str]] = []
    for c in codes:
        if c.status == "retired":
            continue   # retired codes don't count as duplicates
        defn = canonical_text(c.definition)
        name = canonical_text(c.name)
        # Always at least one row per code: the name (+definition if any).
        head = name
        if defn:
            head = f"{name}. {defn}" if name else defn
        if head:
            rows.append((c.id, c.name, head))
        for ex in c.exemplars:
            t = canonical_text(ex)
            if not t:
                continue
            rows.append((c.id, c.name, t))
    return rows


def find_near_duplicates(
    *,
    query_vector: Sequence[float],
    codes: Sequence[Code],
    embed_fn: EmbedFn,
    top_k: int = DEFAULT_NEAR_DUPLICATE_TOP_K,
    min_score: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Return existing codes most similar to ``query_vector``.

    Output is ``[(code_id, code_name, similarity)]`` sorted by
    similarity descending and capped at ``top_k``. Codes scoring below
    ``min_score`` are dropped. ``embed_fn`` is called once on the
    name/definition/exemplar corpus; dimensional mismatches are skipped
    (matches the F8.3 pattern).
    """
    qv = tuple(float(x) for x in query_vector)
    if not qv:
        return []
    rows = _build_existing_corpus(codes)
    if not rows:
        return []
    texts = [r[2] for r in rows]
    vectors = list(embed_fn(texts))
    if len(vectors) != len(rows):
        raise ProjectValidationError(
            f"embed_fn returned {len(vectors)} vectors for "
            f"{len(rows)} inputs"
        )
    best: dict[str, tuple[str, float]] = {}
    for (cid, name, _t), vec in zip(rows, vectors):
        v = tuple(float(x) for x in vec)
        if not v or len(v) != len(qv):
            continue
        s = cosine_similarity(qv, v)
        prev = best.get(cid)
        if prev is None or s > prev[1]:
            best[cid] = (name, s)
    out: list[tuple[str, str, float]] = []
    for cid, (name, s) in best.items():
        if s < min_score:
            continue
        out.append((cid, name, s))
    out.sort(key=lambda r: (-r[2], r[1].lower(), r[0]))
    return out[: max(1, top_k)]


def annotate_proposal_duplicates(
    proposals: Sequence[NewCodeProposal],
    *,
    codes: Sequence[Code],
    embed_fn: EmbedFn,
) -> list[NewCodeProposal]:
    """Fill ``nearest_existing_code_id`` / ``nearest_existing_similarity``.

    For each proposal, build the canonical "name. definition" string,
    embed it, then compare to each existing code's corpus (same rows as
    :func:`find_near_duplicates`, but per-proposal so each gets its own
    nearest match). Returns a new list of proposals; the input is not
    mutated. If no codes exist (or the embedding fails), proposals are
    returned with the duplicate fields untouched.
    """
    if not proposals:
        return []
    rows = _build_existing_corpus(codes)
    proposal_texts = [
        canonical_text(
            f"{p.name}. {p.definition}" if p.definition else p.name
        )
        or p.name
        for p in proposals
    ]
    # We do *one* batched embed call: proposal texts then existing corpus,
    # so the backend can amortise a single HTTP round-trip.
    all_texts = list(proposal_texts) + [r[2] for r in rows]
    if not all_texts:
        return [_clone_proposal(p) for p in proposals]
    vectors = list(embed_fn(all_texts))
    if len(vectors) != len(all_texts):
        raise ProjectValidationError(
            f"embed_fn returned {len(vectors)} vectors for "
            f"{len(all_texts)} inputs"
        )
    proposal_vecs = [tuple(float(x) for x in v) for v in vectors[: len(proposals)]]
    corpus_vecs = [tuple(float(x) for x in v) for v in vectors[len(proposals) :]]

    out: list[NewCodeProposal] = []
    for p, qv in zip(proposals, proposal_vecs):
        clone = _clone_proposal(p)
        if not qv:
            out.append(clone)
            continue
        best_score = -2.0
        best_cid: str | None = None
        for (cid, _name, _t), cv in zip(rows, corpus_vecs):
            if not cv or len(cv) != len(qv):
                continue
            s = cosine_similarity(qv, cv)
            if s > best_score:
                best_score = s
                best_cid = cid
        if best_cid is not None:
            clone.nearest_existing_code_id = best_cid
            clone.nearest_existing_similarity = max(0.0, min(1.0, best_score))
        out.append(clone)
    return out


def _clone_proposal(p: NewCodeProposal) -> NewCodeProposal:
    return NewCodeProposal(
        name=p.name,
        definition=p.definition,
        rationale=p.rationale,
        quote_excerpts=list(p.quote_excerpts),
        confidence=p.confidence,
        nearest_existing_code_id=p.nearest_existing_code_id,
        nearest_existing_similarity=p.nearest_existing_similarity,
        is_gerund=p.is_gerund,
    )


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


# Strict JSON, gerund-form-preferred. The "but Strauss/Corbin / Glaser
# users may use other forms" caveat is implicit — we don't *require* the
# gerund, we just nudge.
NEW_CODE_PROMPT_TEMPLATE = (
    "You are assisting a qualitative researcher with thematic coding "
    "of an interview transcript. Their methodology favours "
    "constructivist grounded theory (Charmaz). They have highlighted "
    "a span of text and want you to propose **NEW** code names — "
    "labels that do NOT yet exist in their codebook.\n"
    "\n"
    "Quoted span:\n"
    "\"\"\"\n{query_text}\n\"\"\"\n"
    "\n"
    "{existing_block}"
    "Guidance:\n"
    "- Prefer GERUND-form names (an -ing verb describing the action), "
    "e.g. \"negotiating identity\" rather than \"identity negotiation\". "
    "This is a strong nudge but not a hard rule.\n"
    "- Each name should be short (2–5 words) and analytic, not "
    "descriptive of the literal content.\n"
    "- Each definition should be one sentence describing what the code "
    "captures across cases (not just this quote).\n"
    "- Each rationale should be one sentence explaining why the span "
    "supports the proposed code.\n"
    "- Quote excerpts should be 1–3 short snippets from the span "
    "supporting the proposed code.\n"
    "\n"
    "Respond with strict JSON only — an array of at most {num_proposals} "
    "objects, ordered best-first, each with keys:\n"
    '  "name":           string (the proposed code label)\n'
    '  "definition":     string (one sentence)\n'
    '  "rationale":      string (one sentence; why it fits this span)\n'
    '  "quote_excerpts": array of strings (short snippets from the span)\n'
    '  "confidence":     number in [0, 1]\n'
    "\n"
    "If the span is too thin to ground any new code, respond with an "
    "empty array `[]`. Do not propose codes that duplicate ones in "
    "the existing codebook above."
)


def make_new_code_prompt(
    *,
    query_text: str,
    existing_codes_to_avoid: Sequence[tuple[str, str, float]] = (),
    num_proposals: int = DEFAULT_NUM_PROPOSALS,
) -> str:
    """Render the prompt sent to the generation backend.

    ``existing_codes_to_avoid`` is the output of
    :func:`find_near_duplicates`: a list of ``(code_id, code_name,
    similarity)`` tuples. We feed it back into the prompt so the model
    has explicit "don't duplicate" context. If the list is empty (no
    similar codes, or no codebook yet), the prompt simply omits the
    block.
    """
    if existing_codes_to_avoid:
        lines = ["Existing codes already in the codebook (do NOT duplicate these):"]
        for _cid, name, _score in existing_codes_to_avoid:
            n = (name or "").strip()
            if not n:
                continue
            lines.append(f"- {n}")
        existing_block = "\n".join(lines) + "\n\n"
    else:
        existing_block = ""
    return NEW_CODE_PROMPT_TEMPLATE.format(
        query_text=(query_text or "").strip(),
        existing_block=existing_block,
        num_proposals=int(num_proposals),
    )


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_proposals_response(
    response_text: str,
) -> list[NewCodeProposal]:
    """Parse the model's JSON response into :class:`NewCodeProposal` list.

    Tolerant of:

    * Plain JSON arrays.
    * JSON wrapped in a ``​```json ... ``​``​`` fence.
    * Models that prefix the JSON with a "Sure! Here you go:" line —
      the helper extracts the largest ``[ ... ]`` slice and parses
      that, falling back to the empty list on hard failure.

    Garbage entries are dropped (rather than raising) so a misbehaving
    model degrades gracefully into "no proposals" instead of breaking
    the whole flow. Entries with empty / missing names are dropped.
    Confidence is clamped to ``[0, 1]``; non-numeric confidence
    becomes ``0.0``. Long fields are truncated to module-level limits.
    """
    text = (response_text or "").strip()
    if not text:
        return []
    candidates: list[str] = []
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    lb = text.find("[")
    rb = text.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        candidates.append(text[lb : rb + 1])
    candidates.append(text)
    parsed: list[Any] | None = None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            parsed = obj
            break
    if parsed is None:
        return []
    out: list[NewCodeProposal] = []
    for entry in parsed:
        if not isinstance(entry, Mapping):
            continue
        name_raw = entry.get("name")
        if not isinstance(name_raw, str):
            continue
        name = name_raw.strip()
        if not name:
            continue
        if len(name) > MAX_NAME_LEN:
            name = name[:MAX_NAME_LEN]
        definition = str(entry.get("definition", "") or "")
        if len(definition) > MAX_DEFINITION_LEN:
            definition = definition[:MAX_DEFINITION_LEN]
        rationale = str(entry.get("rationale", "") or "")
        if len(rationale) > MAX_RATIONALE_LEN:
            rationale = rationale[:MAX_RATIONALE_LEN]
        excerpts_raw = entry.get("quote_excerpts") or []
        if not isinstance(excerpts_raw, list):
            excerpts_raw = []
        excerpts: list[str] = []
        for e in excerpts_raw:
            t = str(e or "").strip()
            if not t:
                continue
            if len(t) > MAX_QUOTE_EXCERPT_LEN:
                t = t[:MAX_QUOTE_EXCERPT_LEN]
            excerpts.append(t)
            if len(excerpts) >= MAX_QUOTE_EXCERPTS:
                break
        try:
            conf = float(entry.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if not _finite(conf):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        try:
            proposal = NewCodeProposal(
                name=name,
                definition=definition,
                rationale=rationale,
                quote_excerpts=excerpts,
                confidence=conf,
            )
            proposal.validate()
        except ProjectValidationError:
            # Last-ditch: skip this entry rather than crashing the
            # whole call.
            continue
        out.append(proposal)
    return out


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def suggest_new_codes_for_span(
    *,
    project_id: str,
    source_id: str,
    anchor_start_word_id: str,
    anchor_end_word_id: str,
    query_text: str,
    codes: Sequence[Code],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn,
    start_char_offset: int | None = None,
    end_char_offset: int | None = None,
    embedding_model: str = "",
    generation_model: str = "",
    num_proposals: int = DEFAULT_NUM_PROPOSALS,
    near_duplicate_top_k: int = DEFAULT_NEAR_DUPLICATE_TOP_K,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    now: str | None = None,
) -> NewCodeSuggestion:
    """End-to-end: build a :class:`NewCodeSuggestion` for a query span.

    Workflow:

      1. Embed the canonicalised ``query_text`` once.
      2. Find existing codes the LLM should be told *not* to duplicate
         (:func:`find_near_duplicates`).
      3. Build the prompt with the gerund-form nudge and the
         "don't duplicate" list.
      4. Call ``generate_fn``; parse the response into proposals.
      5. Annotate each proposal with its nearest existing code
         (:func:`annotate_proposal_duplicates`).
      6. Wrap in a ``NewCodeSuggestion`` with ``decision="pending"``.

    The caller is responsible for persisting the result via
    :func:`save_new_code_suggestion`.
    """
    qt = canonical_text(query_text)
    if not qt:
        raise ProjectValidationError(
            "query_text is empty after canonicalisation"
        )
    if len(qt) > MAX_QUERY_TEXT_LEN:
        raise ProjectValidationError(
            f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars"
        )

    # 1. embed the query
    qv_list = list(embed_fn([qt]))
    if not qv_list:
        raise ProjectValidationError("embed_fn returned no vectors")
    query_vector = tuple(float(x) for x in qv_list[0])
    if not query_vector:
        raise ProjectValidationError("embed_fn returned an empty vector")

    # 2. find existing-code shortlist for the prompt
    near_codes = find_near_duplicates(
        query_vector=query_vector,
        codes=codes,
        embed_fn=embed_fn,
        top_k=near_duplicate_top_k,
        min_score=near_duplicate_threshold,
    )

    # 3. build prompt + 4. call model + parse
    prompt = make_new_code_prompt(
        query_text=qt,
        existing_codes_to_avoid=near_codes,
        num_proposals=num_proposals,
    )
    raw_response = str(generate_fn(prompt) or "")
    if len(raw_response) > MAX_RAW_LLM_RESPONSE_LEN:
        raw_response = raw_response[:MAX_RAW_LLM_RESPONSE_LEN]
    proposals = parse_proposals_response(raw_response)

    # Cap to num_proposals (model may exceed; we trust the contract less
    # than the requested bound).
    proposals = proposals[: max(1, int(num_proposals))]

    # 5. annotate each proposal with nearest-existing-code (best-effort)
    if proposals and codes:
        try:
            proposals = annotate_proposal_duplicates(
                proposals,
                codes=codes,
                embed_fn=embed_fn,
            )
        except ProjectValidationError:
            # Don't lose the suggestion just because the duplicate-check
            # embedding broke; the LLM response is still useful.
            pass

    # Cap to MAX_PROPOSALS_PERSISTED (defence in depth — a buggy model
    # that returned 1000 entries and num_proposals was huge).
    if len(proposals) > MAX_PROPOSALS_PERSISTED:
        proposals = proposals[:MAX_PROPOSALS_PERSISTED]

    return NewCodeSuggestion.new(
        project_id=project_id,
        source_id=source_id,
        anchor_start_word_id=anchor_start_word_id,
        anchor_end_word_id=anchor_end_word_id,
        start_char_offset=start_char_offset,
        end_char_offset=end_char_offset,
        query_text=qt,
        embedding_model=embedding_model,
        generation_model=generation_model,
        proposals=proposals,
        raw_llm_response=raw_response,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Decision lifecycle
# --------------------------------------------------------------------------- #


def record_new_code_decision(
    suggestion: NewCodeSuggestion,
    *,
    decision: str,
    coder_id: str,
    accepted_proposal_index: int | None = None,
    created_code_id: str | None = None,
    rejection_reason: str = "",
    notes: str | None = None,
    now: str | None = None,
) -> None:
    """Move a suggestion from ``pending`` into a terminal state.

    Mutates ``suggestion`` in place and re-runs ``validate``. Decision
    transitions out of a terminal state are not allowed — that would
    rewrite the audit trail; create a new suggestion instead.

    * ``accepted`` and ``modified`` require ``accepted_proposal_index``
      to be set and to be a valid index into ``suggestion.proposals``.
      ``created_code_id`` is optional (the caller may save the
      :class:`Code` after this call) but must be 12-hex if provided.
    * ``rejected`` forbids ``accepted_proposal_index`` /
      ``created_code_id`` and accepts a free-text reason.
    """
    if decision not in TERMINAL_NEW_CODE_DECISIONS:
        raise ProjectValidationError(
            f"decision must be one of {sorted(TERMINAL_NEW_CODE_DECISIONS)}; "
            f"got {decision!r}"
        )
    if suggestion.decision in TERMINAL_NEW_CODE_DECISIONS:
        raise ProjectValidationError(
            f"Suggestion {suggestion.id} already has decision "
            f"{suggestion.decision!r}; create a new suggestion instead "
            "of overwriting the audit trail."
        )
    if not isinstance(coder_id, str) or not CODER_ID_RE.match(coder_id):
        raise ProjectValidationError(
            f"coder_id must be a 12-char hex coder id; got {coder_id!r}"
        )

    if decision in (NEW_CODE_DECISION_ACCEPTED, NEW_CODE_DECISION_MODIFIED):
        if accepted_proposal_index is None:
            raise ProjectValidationError(
                "accepted_proposal_index is required when decision "
                f"is {decision!r}"
            )
        try:
            idx = int(accepted_proposal_index)
        except (TypeError, ValueError) as e:
            raise ProjectValidationError(
                f"accepted_proposal_index must be an integer; "
                f"got {accepted_proposal_index!r}"
            ) from e
        if idx < 0 or idx >= len(suggestion.proposals):
            raise ProjectValidationError(
                f"accepted_proposal_index {idx} out of range for "
                f"{len(suggestion.proposals)} proposals"
            )
        suggestion.accepted_proposal_index = idx
        if created_code_id is not None:
            if not CODE_ID_RE.match(str(created_code_id)):
                raise ProjectValidationError(
                    "created_code_id must be 12-char hex if set"
                )
            suggestion.created_code_id = str(created_code_id)
    else:  # rejected
        if accepted_proposal_index is not None or created_code_id is not None:
            raise ProjectValidationError(
                "rejected suggestions must not record an accepted "
                "proposal index or created code id"
            )
        suggestion.accepted_proposal_index = None
        suggestion.created_code_id = None

    suggestion.decision = decision
    suggestion.decided_at = now or utcnow_iso()
    suggestion.decided_by_coder_id = coder_id
    if rejection_reason:
        if len(rejection_reason) > MAX_REJECTION_REASON_LEN:
            raise ProjectValidationError(
                f"rejection_reason exceeds {MAX_REJECTION_REASON_LEN} chars"
            )
        suggestion.rejection_reason = rejection_reason
    elif decision != NEW_CODE_DECISION_REJECTED:
        suggestion.rejection_reason = ""
    if notes is not None:
        if len(notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )
        suggestion.notes = notes
    suggestion.modified_at = now or utcnow_iso()
    suggestion.validate()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def new_code_suggestions_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's new-code suggestions.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / NEW_CODE_SUGGESTIONS_DIRNAME


def new_code_suggestion_state_path(
    projects_root: Path, project_id: str, suggestion_id: str
) -> Path:
    if not NEW_CODE_SUGGESTION_ID_RE.match(suggestion_id):
        raise ProjectValidationError(
            f"Invalid new-code suggestion id: {suggestion_id!r}"
        )
    return (
        new_code_suggestions_dir(projects_root, project_id)
        / f"{suggestion_id}.json"
    )


def save_new_code_suggestion(
    projects_root: Path, suggestion: NewCodeSuggestion
) -> Path:
    """Persist a new-code suggestion atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    suggestion.validate()
    parent = project_dir(projects_root, suggestion.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving new-code suggestions."
        )
    sd = new_code_suggestions_dir(projects_root, suggestion.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = new_code_suggestion_state_path(
        projects_root, suggestion.project_id, suggestion.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(suggestion.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_new_code_suggestion(
    projects_root: Path, project_id: str, suggestion_id: str
) -> NewCodeSuggestion:
    """Load a new-code suggestion by id. Raises ``FileNotFoundError`` if missing."""
    p = new_code_suggestion_state_path(
        projects_root, project_id, suggestion_id
    )
    if not p.exists():
        raise FileNotFoundError(f"No new-code suggestion at {p}")
    return NewCodeSuggestion.from_dict(json.loads(p.read_text()))


def list_new_code_suggestions(
    projects_root: Path,
    project_id: str,
    *,
    source_id: str | None = None,
    decision: str | None = None,
) -> list[NewCodeSuggestion]:
    """List all new-code suggestions in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse. Sorted by
    ``created_at`` ascending so the natural reading order is
    "the order in which suggestions were requested" — matches the
    audit-trail story. Files whose stem isn't a valid suggestion id
    are skipped.
    """
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    if decision is not None and decision not in NEW_CODE_DECISIONS:
        raise ProjectValidationError(
            f"Invalid decision filter: {decision!r}"
        )
    sd = new_code_suggestions_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[NewCodeSuggestion] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not NEW_CODE_SUGGESTION_ID_RE.match(sid):
            continue
        try:
            s = NewCodeSuggestion.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if source_id is not None and s.source_id != source_id:
            continue
        if decision is not None and s.decision != decision:
            continue
        out.append(s)
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def delete_new_code_suggestion(
    projects_root: Path, project_id: str, suggestion_id: str
) -> bool:
    """Remove a new-code suggestion file. Returns False if it didn't exist.

    Production code should prefer keeping suggestions for the audit
    trail; deletion is exposed for tests and for the REFI-QDA import
    path (where a clean slate matters).
    """
    p = new_code_suggestion_state_path(
        projects_root, project_id, suggestion_id
    )
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(
            f"Refusing to delete outside root: {p}"
        )
    p.unlink()
    return True
