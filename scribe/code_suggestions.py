"""Code suggestion engine for the academic-coding workflow (F8.3).

Per PLANNING.md F8.3:

  > "Suggest codes from existing codebook" action on a highlighted span.
  > Ranked list combining embedding similarity to existing exemplars +
  > LLM-driven analysis. Explicit accept / modify / reject buttons.

The engine never *applies* a code — it produces a ranked
:class:`CodeSuggestion` whose decision starts as ``"pending"``. A human
coder later calls :func:`record_decision` with one of ``"accepted"``,
``"modified"``, or ``"rejected"``. Even rejections are persisted so
the AI invocation log (F9.6) carries the full picture.

What this module does
---------------------

1. **Score candidates from the embedding index.** Given a query span,
   embed its text once and use cosine similarity against the F8.2
   index to find the most similar coded segments. Group by
   ``code_id``; the per-code embedding score is the *max* similarity
   across that code's matching segments. (We deliberately use max not
   mean — a single strong analogue is more diagnostic than a sea of
   weak ones for the suggestion ranking.)

2. **Score candidates by definition + exemplars.** Codes that have not
   yet been applied still need to be candidates. We embed each code's
   ``definition`` and each ``exemplar`` on the fly (one batched
   :func:`embed_fn` call) and fold the max into the per-code score.

3. **Optional LLM rerank / rationale.** When a generation callable is
   supplied, the engine builds a structured prompt that lists the
   short-listed codes with their definitions, asks for a JSON-shaped
   ranking with one-line rationales, and folds the LLM scores into the
   final ranking via a configurable weight. The raw response is stored
   alongside the parsed candidates so a researcher can audit what the
   model actually said.

4. **Persist the suggestion.** Each invocation produces a
   :class:`CodeSuggestion` saved at
   ``projects/<pid>/code_suggestions/<sid>.json``. Decisions update the
   record in place; the suggestion file is the audit trail.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.3 is the engine; the
  ``/api/projects/<id>/code-suggestions`` routes are deferred and will
  be a thin shell over this module.
* **No automatic application creation.** The decision recorder marks
  a suggestion ``"accepted"`` and notes the application id the caller
  created on the side. Applications are still F4.1 territory.
* **Pure callables.** ``embed_fn`` and ``generate_fn`` are arbitrary
  callables (the F8.1 backend adapter wraps OllamaBackend into both).
  Tests stub them with deterministic functions; production wires them
  to the registered backend.

This module is stand-alone — no FastAPI, no engine imports — so the
data model can be tested in pure Python and reused by the CLI later.
Conventions match the rest of the F-feature stack
(:mod:`scribe.applications`, :mod:`scribe.codes`,
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

from .applications import APPLICATION_ID_RE, Application, parse_word_id
from .codes import CODE_ID_RE, Code
from .coders import CODER_ID_RE
from .embedding_index import (
    EMBEDDING_KIND_CODED_SEGMENT,
    EmbeddingEntry,
    canonical_text,
    cosine_similarity,
    list_embedding_entries,
)
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
SUGGESTION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding suggestions.
SUGGESTIONS_DIRNAME = "code_suggestions"

# Decision lifecycle states for a suggestion. ``pending`` is the
# starting state; the other three are terminal but a researcher can
# still attach a note after the fact (apply_update covers that).
SUGGESTION_DECISION_PENDING = "pending"
SUGGESTION_DECISION_ACCEPTED = "accepted"
SUGGESTION_DECISION_MODIFIED = "modified"
SUGGESTION_DECISION_REJECTED = "rejected"
SUGGESTION_DECISIONS: tuple[str, ...] = (
    SUGGESTION_DECISION_PENDING,
    SUGGESTION_DECISION_ACCEPTED,
    SUGGESTION_DECISION_MODIFIED,
    SUGGESTION_DECISION_REJECTED,
)
TERMINAL_DECISIONS: frozenset[str] = frozenset(
    {
        SUGGESTION_DECISION_ACCEPTED,
        SUGGESTION_DECISION_MODIFIED,
        SUGGESTION_DECISION_REJECTED,
    }
)

# The "source" kind for a candidate's match: which embedding pool the
# top similarity came from. Useful when explaining why a code is
# being suggested.
CANDIDATE_MATCH_SEGMENT = "coded_segment"
CANDIDATE_MATCH_EXEMPLAR = "exemplar"
CANDIDATE_MATCH_DEFINITION = "definition"
CANDIDATE_MATCH_KINDS: tuple[str, ...] = (
    CANDIDATE_MATCH_SEGMENT,
    CANDIDATE_MATCH_EXEMPLAR,
    CANDIDATE_MATCH_DEFINITION,
)

# Defaults for the engine's tunables.
DEFAULT_TOP_K = 5
DEFAULT_MAX_CANDIDATES = 12        # cap before LLM rerank to keep prompts cheap
DEFAULT_EMBEDDING_WEIGHT = 0.6     # combined = α·embedding + (1-α)·llm
DEFAULT_TEMPERATURE = 0.2          # low — we want a ranked list, not creative writing
DEFAULT_MAX_TOKENS = 1024

# Field-length / cardinality caps. Generous, but bounded so a stray
# upstream bug can't write a 50 MB suggestion record.
MAX_QUERY_TEXT_LEN = 8000
MAX_RATIONALE_LEN = 2000
MAX_RAW_LLM_RESPONSE_LEN = 16 * 1024
MAX_CANDIDATES_PERSISTED = 50
MAX_REJECTION_REASON_LEN = 2000
MAX_NOTES_LEN = 4000

# Allowed callable signatures. ``embed_fn`` matches F8.2's
# ``EmbedFn`` (a single batched call returns a list of vectors,
# one per input). ``generate_fn`` is a small callable returning
# the raw text body.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]
GenerateFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Candidate data model
# --------------------------------------------------------------------------- #


@dataclass
class CandidateMatch:
    """One concrete piece of evidence supporting a candidate's score.

    ``ref`` is interpreted relative to ``kind``:

    * ``coded_segment`` — application id (a 12-char hex).
    * ``exemplar`` — the exemplar's index in the code's exemplars list,
      stored as a string (we keep the field free-shape so it can also
      hold a hash later).
    * ``definition`` — the literal string ``"definition"``.
    """

    kind: str
    ref: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "score": float(self.score)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CandidateMatch":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "CandidateMatch payload must be an object"
            )
        kind = str(d.get("kind", "") or "")
        ref = str(d.get("ref", "") or "")
        try:
            score = float(d.get("score", 0.0) or 0.0)
        except (TypeError, ValueError) as e:
            raise ProjectValidationError(
                f"CandidateMatch.score must be numeric: {d.get('score')!r}"
            ) from e
        m = cls(kind=kind, ref=ref, score=score)
        m.validate()
        return m

    def validate(self) -> None:
        if self.kind not in CANDIDATE_MATCH_KINDS:
            raise ProjectValidationError(
                f"CandidateMatch.kind must be one of {CANDIDATE_MATCH_KINDS}; "
                f"got {self.kind!r}"
            )
        if not isinstance(self.ref, str) or not self.ref:
            raise ProjectValidationError(
                "CandidateMatch.ref must be a non-empty string"
            )
        if self.kind == CANDIDATE_MATCH_SEGMENT and not APPLICATION_ID_RE.match(
            self.ref
        ):
            raise ProjectValidationError(
                f"CandidateMatch with kind=coded_segment requires a "
                f"12-char hex application id; got {self.ref!r}"
            )
        if not _finite(self.score) or self.score < -1.0001 or self.score > 1.0001:
            raise ProjectValidationError(
                f"CandidateMatch.score must be in [-1, 1]; got {self.score!r}"
            )


@dataclass
class CodeCandidate:
    """One ranked candidate code in a :class:`CodeSuggestion`.

    All three score fields live in ``[0, 1]``. The combined score is
    what callers should sort on; the components are kept so a UI can
    show "this came primarily from the LLM, not the index".
    """

    code_id: str
    code_name: str
    embedding_score: float
    llm_score: float
    combined_score: float
    rationale: str = ""
    matches: list[CandidateMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "code_name": self.code_name,
            "embedding_score": float(self.embedding_score),
            "llm_score": float(self.llm_score),
            "combined_score": float(self.combined_score),
            "rationale": self.rationale,
            "matches": [m.to_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CodeCandidate":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "CodeCandidate payload must be an object"
            )
        for required in ("code_id", "code_name"):
            if required not in d:
                raise ProjectValidationError(
                    f"CodeCandidate payload missing required key: {required}"
                )
        c = cls(
            code_id=str(d["code_id"]),
            code_name=str(d.get("code_name", "")),
            embedding_score=_coerce_score(
                d.get("embedding_score", 0.0), "embedding_score"
            ),
            llm_score=_coerce_score(d.get("llm_score", 0.0), "llm_score"),
            combined_score=_coerce_score(
                d.get("combined_score", 0.0), "combined_score"
            ),
            rationale=str(d.get("rationale", "") or "")[:MAX_RATIONALE_LEN],
            matches=[
                CandidateMatch.from_dict(m) for m in (d.get("matches") or [])
            ],
        )
        c.validate()
        return c

    def validate(self) -> None:
        if not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"CodeCandidate.code_id must be 12-char hex; "
                f"got {self.code_id!r}"
            )
        if not isinstance(self.code_name, str):
            raise ProjectValidationError("CodeCandidate.code_name must be a string")
        for label, val in (
            ("embedding_score", self.embedding_score),
            ("llm_score", self.llm_score),
            ("combined_score", self.combined_score),
        ):
            if not _finite(val) or val < -0.0001 or val > 1.0001:
                raise ProjectValidationError(
                    f"CodeCandidate.{label} must be in [0, 1]; got {val!r}"
                )
        if not isinstance(self.rationale, str):
            raise ProjectValidationError("CodeCandidate.rationale must be a string")
        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise ProjectValidationError(
                f"CodeCandidate.rationale exceeds {MAX_RATIONALE_LEN} chars"
            )
        for m in self.matches:
            m.validate()


# --------------------------------------------------------------------------- #
# Suggestion record
# --------------------------------------------------------------------------- #


@dataclass
class CodeSuggestion:
    """One code-suggestion invocation, persisted as the audit record.

    The anchor fields mirror :class:`scribe.applications.Application`
    so an accepted suggestion can be turned into an Application without
    re-deriving anchors. The decision lifecycle is::

        pending → accepted | modified | rejected

    Once a decision is recorded, ``decided_at`` and
    ``decided_by_coder_id`` are non-empty. ``modified`` is the case
    where the researcher took the suggestion as a starting point but
    changed the code or the span before applying — we keep both the
    original suggested code and the eventual ``accepted_code_id`` so
    reports can show "AI suggested X, human applied Y".
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
    candidates: list[CodeCandidate] = field(default_factory=list)
    decision: str = SUGGESTION_DECISION_PENDING
    decided_at: str = ""
    decided_by_coder_id: str = ""
    accepted_code_id: str | None = None
    accepted_application_id: str | None = None
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
        candidates: Iterable[CodeCandidate] | None = None,
        start_char_offset: int | None = None,
        end_char_offset: int | None = None,
        notes: str = "",
        raw_llm_response: str = "",
        suggestion_id: str | None = None,
        now: str | None = None,
    ) -> "CodeSuggestion":
        ts = now or utcnow_iso()
        coerced: list[CodeCandidate] = []
        for c in candidates or ():
            if isinstance(c, CodeCandidate):
                coerced.append(c)
            else:
                coerced.append(CodeCandidate.from_dict(c))
        s = cls(
            id=suggestion_id or new_suggestion_id(),
            project_id=project_id,
            source_id=source_id,
            anchor_start_word_id=anchor_start_word_id,
            anchor_end_word_id=anchor_end_word_id,
            start_char_offset=start_char_offset,
            end_char_offset=end_char_offset,
            query_text=str(query_text or ""),
            embedding_model=str(embedding_model or ""),
            generation_model=str(generation_model or ""),
            candidates=coerced,
            decision=SUGGESTION_DECISION_PENDING,
            decided_at="",
            decided_by_coder_id="",
            accepted_code_id=None,
            accepted_application_id=None,
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
            "candidates": [c.to_dict() for c in self.candidates],
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by_coder_id": self.decided_by_coder_id,
            "accepted_code_id": self.accepted_code_id,
            "accepted_application_id": self.accepted_application_id,
            "rejection_reason": self.rejection_reason,
            "notes": self.notes,
            "raw_llm_response": self.raw_llm_response,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CodeSuggestion":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "CodeSuggestion payload must be an object"
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
                    f"CodeSuggestion payload missing required key: {required}"
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
            candidates=[
                CodeCandidate.from_dict(c) for c in (d.get("candidates") or [])
            ],
            decision=str(d.get("decision", SUGGESTION_DECISION_PENDING) or
                         SUGGESTION_DECISION_PENDING),
            decided_at=str(d.get("decided_at", "") or ""),
            decided_by_coder_id=str(d.get("decided_by_coder_id", "") or ""),
            accepted_code_id=(
                str(d["accepted_code_id"])
                if d.get("accepted_code_id")
                else None
            ),
            accepted_application_id=(
                str(d["accepted_application_id"])
                if d.get("accepted_application_id")
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
        if not SUGGESTION_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid suggestion id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        # Anchors: shape and ordering. parse_word_id raises on garbage.
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
        if not isinstance(self.embedding_model, str) or len(self.embedding_model) > 256:
            raise ProjectValidationError(
                "embedding_model must be a string ≤ 256 chars"
            )
        if not isinstance(self.generation_model, str) or len(self.generation_model) > 256:
            raise ProjectValidationError(
                "generation_model must be a string ≤ 256 chars"
            )
        if len(self.candidates) > MAX_CANDIDATES_PERSISTED:
            raise ProjectValidationError(
                f"candidates exceeds {MAX_CANDIDATES_PERSISTED} entries"
            )
        for c in self.candidates:
            c.validate()
        if self.decision not in SUGGESTION_DECISIONS:
            raise ProjectValidationError(
                f"decision must be one of {SUGGESTION_DECISIONS}; "
                f"got {self.decision!r}"
            )
        if self.decision in TERMINAL_DECISIONS:
            if not self.decided_at:
                raise ProjectValidationError(
                    f"decided_at must be set when decision is {self.decision!r}"
                )
            if not self.decided_by_coder_id or not CODER_ID_RE.match(
                self.decided_by_coder_id
            ):
                raise ProjectValidationError(
                    f"decided_by_coder_id must be a 12-char hex coder id "
                    f"when decision is {self.decision!r}"
                )
        if self.accepted_code_id is not None and not CODE_ID_RE.match(
            self.accepted_code_id
        ):
            raise ProjectValidationError(
                f"accepted_code_id must be 12-char hex or null; "
                f"got {self.accepted_code_id!r}"
            )
        if self.accepted_application_id is not None and not APPLICATION_ID_RE.match(
            self.accepted_application_id
        ):
            raise ProjectValidationError(
                f"accepted_application_id must be 12-char hex or null; "
                f"got {self.accepted_application_id!r}"
            )
        # An "accepted" or "modified" decision must record which code was
        # ultimately applied; "rejected" must not.
        if self.decision in (
            SUGGESTION_DECISION_ACCEPTED,
            SUGGESTION_DECISION_MODIFIED,
        ):
            if not self.accepted_code_id:
                raise ProjectValidationError(
                    f"accepted_code_id is required when decision is "
                    f"{self.decision!r}"
                )
        elif self.decision == SUGGESTION_DECISION_REJECTED:
            if self.accepted_code_id or self.accepted_application_id:
                raise ProjectValidationError(
                    "rejected suggestions must not record an accepted "
                    "code or application id"
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
        """Apply a partial update in place. Mirrors ``Application.apply_update``.

        Only ``notes``, ``rejection_reason``, ``accepted_code_id``, and
        ``accepted_application_id`` can be patched freely. Decision
        transitions go through :func:`record_decision` so the
        invariants stay enforced. Stamps ``modified_at``.
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
        if "accepted_code_id" in patch:
            v = patch["accepted_code_id"]
            self.accepted_code_id = str(v) if v else None
        if "accepted_application_id" in patch:
            v = patch["accepted_application_id"]
            self.accepted_application_id = str(v) if v else None
        self.modified_at = now or utcnow_iso()
        self.validate()


_ALLOWED_PATCH_KEYS = {
    "notes",
    "rejection_reason",
    "accepted_code_id",
    "accepted_application_id",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _coerce_score(v: Any, label: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be numeric; got {v!r}"
        ) from e


def _optional_int(v: Any, label: str) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be an integer or null; got {v!r}"
        ) from e


def new_suggestion_id() -> str:
    """Mint a new 12-char hex suggestion id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Embedding-side scoring
# --------------------------------------------------------------------------- #


def _max_score_per_code_from_index(
    *,
    query_vector: Sequence[float],
    index_entries: Sequence[EmbeddingEntry],
    code_id_by_application: Mapping[str, str],
    embedding_model: str,
) -> dict[str, list[CandidateMatch]]:
    """Score every coded-segment entry against the query, group by code.

    Returns ``code_id → [CandidateMatch]`` sorted by score descending.
    Mismatched dimensionalities are silently skipped (matches
    :func:`scribe.embedding_index.search_similar`'s stance — different
    model families produce incomparable vectors and the right answer is
    "skip", not raise).

    ``embedding_model`` filters the index to entries from the same model
    so we don't try to compare bge-m3 vectors with nomic-embed vectors.
    Empty model name disables the filter (handy for tests with stub
    embedders that don't bother setting it).
    """
    qv = tuple(float(x) for x in query_vector)
    out: dict[str, list[CandidateMatch]] = {}
    for entry in index_entries:
        if entry.kind != EMBEDDING_KIND_CODED_SEGMENT:
            continue
        if not entry.application_id:
            continue
        code_id = code_id_by_application.get(entry.application_id)
        if not code_id:
            continue
        if entry.dim != len(qv):
            continue
        if embedding_model and entry.model_name and entry.model_name != embedding_model:
            continue
        score = cosine_similarity(qv, entry.vector)
        out.setdefault(code_id, []).append(
            CandidateMatch(
                kind=CANDIDATE_MATCH_SEGMENT,
                ref=entry.application_id,
                score=score,
            )
        )
    for matches in out.values():
        matches.sort(key=lambda m: -m.score)
    return out


def _build_definition_exemplar_corpus(
    codes: Sequence[Code],
) -> list[tuple[str, str, str, str]]:
    """Return ``(code_id, kind, ref, text)`` rows for definitions + exemplars.

    ``ref`` is what gets stored on a :class:`CandidateMatch`:

    * ``"definition"`` for the definition row.
    * ``"exemplar:<i>"`` where ``<i>`` is the *original* index of the
      exemplar in ``code.exemplars`` — so a UI can deep-link back to the
      exact entry. Empty / whitespace-only entries are skipped (they'd
      surface as embed-backend validation errors), but the surviving
      exemplars keep their original list indices.
    """
    rows: list[tuple[str, str, str, str]] = []
    for c in codes:
        d = canonical_text(c.definition)
        if d:
            rows.append(
                (c.id, CANDIDATE_MATCH_DEFINITION, CANDIDATE_MATCH_DEFINITION, d)
            )
        for i, ex in enumerate(c.exemplars):
            t = canonical_text(ex)
            if not t:
                continue
            rows.append((c.id, CANDIDATE_MATCH_EXEMPLAR, f"exemplar:{i}", t))
    return rows


def score_candidates(
    *,
    query_vector: Sequence[float],
    codes: Sequence[Code],
    code_id_by_application: Mapping[str, str],
    index_entries: Sequence[EmbeddingEntry],
    embed_fn: EmbedFn | None = None,
    embedding_model: str = "",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    min_score: float = 0.0,
) -> list[CodeCandidate]:
    """Score every code in ``codes`` against the query vector.

    Inputs:

    * ``query_vector`` — already-embedded query (call ``embed_fn`` to
      produce this; the caller does it once so the same vector can be
      reused for prompt construction or persistence).
    * ``codes`` — the project's full codebook. Used to resolve names,
      definitions, and exemplars.
    * ``code_id_by_application`` — ``application_id → code_id`` mapping
      so coded-segment entries from the F8.2 index can be grouped.
    * ``index_entries`` — F8.2 ``EmbeddingEntry`` list. Only
      ``coded_segment`` entries contribute; ``uncoded_paragraph``
      entries are ignored here (they're for the F8.5 / F8.6 features).
    * ``embed_fn`` — optional. If provided, definitions + exemplars are
      embedded via this callable so codes that have not yet been
      applied still produce a ranking. Caller supplies the same
      callable used for the query.
    * ``embedding_model`` — recorded on each candidate's matches so the
      audit trail shows which model was queried; also filters the index
      to comparable entries.
    * ``max_candidates`` — cap on the returned list.
    * ``min_score`` — drop candidates whose embedding_score is below
      this threshold (default 0.0; cosine ≤ 0 means "unrelated").

    Codes whose status is ``"retired"`` are excluded — F2.3 marks
    retired codes as not available for new applications. ``"draft"``
    codes are kept; researchers in active coding want to see them.

    Returns the candidates sorted by ``embedding_score`` descending,
    truncated to ``max_candidates``. ``llm_score`` is 0.0 and
    ``combined_score`` mirrors ``embedding_score`` until
    :func:`apply_llm_rerank` (or a manual caller) updates them.
    """
    qv = tuple(float(x) for x in query_vector)
    if not qv:
        return []

    # ---- index-side scoring (coded segments) -------------------------
    seg_scores = _max_score_per_code_from_index(
        query_vector=qv,
        index_entries=index_entries,
        code_id_by_application=code_id_by_application,
        embedding_model=embedding_model,
    )

    # ---- definition / exemplar scoring -------------------------------
    def_ex_scores: dict[str, list[CandidateMatch]] = {}
    if embed_fn is not None:
        rows = _build_definition_exemplar_corpus(codes)
        if rows:
            texts = [r[3] for r in rows]
            vectors = list(embed_fn(texts))
            if len(vectors) != len(rows):
                raise ProjectValidationError(
                    f"embed_fn returned {len(vectors)} vectors for "
                    f"{len(rows)} inputs"
                )
            for (code_id, kind, ref, _text), vec in zip(rows, vectors):
                v = tuple(float(x) for x in vec)
                if not v or len(v) != len(qv):
                    continue
                score = cosine_similarity(qv, v)
                def_ex_scores.setdefault(code_id, []).append(
                    CandidateMatch(kind=kind, ref=ref, score=score)
                )

    # ---- assemble candidates -----------------------------------------
    candidates: list[CodeCandidate] = []
    for code in codes:
        if code.status == "retired":
            continue
        seg_matches = seg_scores.get(code.id, [])
        de_matches = def_ex_scores.get(code.id, [])
        all_matches = seg_matches + de_matches
        if not all_matches:
            continue
        all_matches.sort(key=lambda m: -m.score)
        top_score = all_matches[0].score
        if top_score < min_score:
            continue
        # Clamp to [0, 1]; cosine on real embeddings is normally in
        # [-1, 1] but for "similarity" we treat negatives as 0.
        emb = max(0.0, min(1.0, top_score))
        candidates.append(
            CodeCandidate(
                code_id=code.id,
                code_name=code.name,
                embedding_score=emb,
                llm_score=0.0,
                combined_score=emb,
                rationale="",
                matches=all_matches[:8],   # keep most salient evidence
            )
        )
    candidates.sort(
        key=lambda c: (-c.embedding_score, c.code_name.lower(), c.code_id)
    )
    return candidates[: max(1, max_candidates)]


# --------------------------------------------------------------------------- #
# LLM rerank
# --------------------------------------------------------------------------- #


# A short, unambiguous prompt. Asks for strict JSON to make parsing
# robust; the parser also tolerates the model wrapping the JSON in a
# code fence.
SUGGEST_PROMPT_TEMPLATE = (
    "You are assisting a qualitative researcher with thematic coding "
    "of an interview transcript. Given a quoted span of text and a "
    "shortlist of candidate codes (each with a definition), rank the "
    "candidates that best apply.\n"
    "\n"
    "Quoted span:\n"
    "\"\"\"\n{query_text}\n\"\"\"\n"
    "\n"
    "Candidate codes:\n"
    "{codes_block}\n"
    "Respond with strict JSON only — an array of objects, ordered "
    "best-first, each with keys:\n"
    '  "code_id": string (one of the candidate ids above)\n'
    '  "score":   number in [0, 1] (your confidence the code applies)\n'
    '  "rationale": short string (one sentence; why it applies, '
    "concretely)\n"
    "\n"
    "Include at most {top_k} entries. If none of the candidates "
    'apply, respond with an empty array `[]`. Do not invent code ids.'
)


def make_suggestion_prompt(
    *,
    query_text: str,
    candidates: Sequence[CodeCandidate],
    codes: Sequence[Code],
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Render the prompt sent to the generation backend.

    ``codes`` is consulted so the prompt can include each candidate's
    full definition (and inclusion criteria when present), not just its
    name. We deliberately exclude the rationale / score the embedding
    layer assigned — telling the LLM "this scored 0.78" biases the
    rerank.
    """
    by_id = {c.id: c for c in codes}
    blocks: list[str] = []
    for cand in candidates:
        c = by_id.get(cand.code_id)
        if c is None:
            # Defensive: caller passed a candidate for a code we don't
            # have. Skip rather than crash.
            continue
        defn = canonical_text(c.definition) or "(no definition)"
        incl = canonical_text(c.inclusion_criteria)
        excl = canonical_text(c.exclusion_criteria)
        block = f"- id: {c.id}\n  name: {c.name}\n  definition: {defn}"
        if incl:
            block += f"\n  inclusion: {incl}"
        if excl:
            block += f"\n  exclusion: {excl}"
        blocks.append(block)
    codes_block = "\n".join(blocks) if blocks else "(none)"
    return SUGGEST_PROMPT_TEMPLATE.format(
        query_text=query_text.strip(),
        codes_block=codes_block,
        top_k=int(top_k),
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_llm_ranking(
    response_text: str,
) -> list[tuple[str, float, str]]:
    """Parse the model's JSON ranking. Returns ``[(code_id, score, rationale)]``.

    Tolerant of:

    * Plain JSON arrays.
    * JSON wrapped in a ``​```json ... ``​``` fence.
    * Models that prefix the JSON with a "Sure! Here you go:" line —
      the helper extracts the largest ``[ ... ]`` slice and parses
      that, falling back to the empty list on hard failure.

    Garbage / non-numeric scores → entries dropped (rather than
    raising) so a misbehaving model degrades gracefully into "no LLM
    contribution" instead of breaking the whole suggestion flow.
    """
    text = (response_text or "").strip()
    if not text:
        return []
    candidates: list[str] = []
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    # Look for the first '[' through the last ']' as a fallback slice.
    lb = text.find("[")
    rb = text.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        candidates.append(text[lb : rb + 1])
    # Last-ditch: maybe the whole response is JSON.
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
    out: list[tuple[str, float, str]] = []
    for entry in parsed:
        if not isinstance(entry, Mapping):
            continue
        code_id = entry.get("code_id")
        if not isinstance(code_id, str) or not CODE_ID_RE.match(code_id):
            continue
        score_raw = entry.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            continue
        if not _finite(score):
            continue
        score = max(0.0, min(1.0, score))
        rationale = str(entry.get("rationale", "") or "")[:MAX_RATIONALE_LEN]
        out.append((code_id, score, rationale))
    return out


def apply_llm_rerank(
    candidates: Sequence[CodeCandidate],
    llm_rankings: Sequence[tuple[str, float, str]],
    *,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
) -> list[CodeCandidate]:
    """Fold LLM scores + rationales into the candidate list.

    ``embedding_weight`` is α in ``combined = α·embedding + (1-α)·llm``.
    Codes the LLM didn't mention keep ``llm_score = 0`` and
    ``combined = α·embedding`` (so the LLM can lower a code's combined
    score by simply not naming it; that's intended).

    Returns a new list sorted by ``combined_score`` descending.
    """
    if not 0.0 <= embedding_weight <= 1.0:
        raise ProjectValidationError(
            f"embedding_weight must be in [0, 1]; got {embedding_weight}"
        )
    llm_weight = 1.0 - embedding_weight
    by_code: dict[str, tuple[float, str]] = {}
    for code_id, score, rationale in llm_rankings:
        # If the LLM mentions the same code twice, take the max.
        prev = by_code.get(code_id)
        if prev is None or score > prev[0]:
            by_code[code_id] = (score, rationale)
    out: list[CodeCandidate] = []
    for c in candidates:
        llm_score, rationale = by_code.get(c.code_id, (0.0, ""))
        combined = (
            embedding_weight * c.embedding_score + llm_weight * llm_score
        )
        out.append(
            CodeCandidate(
                code_id=c.code_id,
                code_name=c.code_name,
                embedding_score=c.embedding_score,
                llm_score=llm_score,
                combined_score=max(0.0, min(1.0, combined)),
                rationale=rationale or c.rationale,
                matches=list(c.matches),
            )
        )
    out.sort(
        key=lambda c: (
            -c.combined_score,
            -c.embedding_score,
            c.code_name.lower(),
            c.code_id,
        )
    )
    return out


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def suggest_codes_for_span(
    *,
    projects_root: Path,
    project_id: str,
    source_id: str,
    anchor_start_word_id: str,
    anchor_end_word_id: str,
    query_text: str,
    codes: Sequence[Code],
    applications: Sequence[Application],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn | None = None,
    start_char_offset: int | None = None,
    end_char_offset: int | None = None,
    embedding_model: str = "",
    generation_model: str = "",
    top_k: int = DEFAULT_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    min_score: float = 0.0,
    now: str | None = None,
) -> CodeSuggestion:
    """End-to-end: build a :class:`CodeSuggestion` for a query span.

    Workflow:

      1. Embed the canonicalised ``query_text`` once.
      2. Score every code via :func:`score_candidates`.
      3. Truncate to ``max_candidates`` and (if ``generate_fn`` is set)
         build the prompt, call the LLM, parse the response, and
         :func:`apply_llm_rerank`.
      4. Truncate to ``top_k``.
      5. Wrap in a ``CodeSuggestion`` with ``decision="pending"``.

    Caller is responsible for persisting the result via
    :func:`save_suggestion`.
    """
    qt = canonical_text(query_text)
    if not qt:
        raise ProjectValidationError("query_text is empty after canonicalisation")
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

    # 2. score candidates
    code_id_by_application = {a.id: a.code_id for a in applications}
    index_entries = list_embedding_entries(
        projects_root,
        project_id,
        kind=EMBEDDING_KIND_CODED_SEGMENT,
    )
    candidates = score_candidates(
        query_vector=query_vector,
        codes=codes,
        code_id_by_application=code_id_by_application,
        index_entries=index_entries,
        embed_fn=embed_fn,
        embedding_model=embedding_model,
        max_candidates=max_candidates,
        min_score=min_score,
    )

    raw_response = ""
    if generate_fn is not None and candidates:
        prompt = make_suggestion_prompt(
            query_text=qt,
            candidates=candidates,
            codes=codes,
            top_k=top_k,
        )
        raw_response = str(generate_fn(prompt) or "")
        if len(raw_response) > MAX_RAW_LLM_RESPONSE_LEN:
            raw_response = raw_response[:MAX_RAW_LLM_RESPONSE_LEN]
        rankings = parse_llm_ranking(raw_response)
        candidates = apply_llm_rerank(
            candidates,
            rankings,
            embedding_weight=embedding_weight,
        )

    return CodeSuggestion.new(
        project_id=project_id,
        source_id=source_id,
        anchor_start_word_id=anchor_start_word_id,
        anchor_end_word_id=anchor_end_word_id,
        start_char_offset=start_char_offset,
        end_char_offset=end_char_offset,
        query_text=qt,
        embedding_model=embedding_model,
        generation_model=generation_model,
        candidates=candidates[: max(1, top_k)],
        raw_llm_response=raw_response,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Decision lifecycle
# --------------------------------------------------------------------------- #


def record_decision(
    suggestion: CodeSuggestion,
    *,
    decision: str,
    coder_id: str,
    accepted_code_id: str | None = None,
    accepted_application_id: str | None = None,
    rejection_reason: str = "",
    notes: str | None = None,
    now: str | None = None,
) -> None:
    """Move a suggestion from ``pending`` into a terminal state.

    Mutates ``suggestion`` in place and re-runs ``validate``. Decision
    transitions out of a terminal state are not allowed — that would
    rewrite the audit trail; create a new suggestion instead.

    * ``accepted`` and ``modified`` require ``accepted_code_id``. For
      ``accepted`` the code id is one of the candidates'; for
      ``modified`` it is *any* code in the project (the human picked a
      different code). The validator only enforces shape, not
      membership; the caller can pre-check candidate membership if it
      cares.
    * ``rejected`` forbids ``accepted_code_id`` /
      ``accepted_application_id`` and accepts a free-text reason.
    """
    if decision not in TERMINAL_DECISIONS:
        raise ProjectValidationError(
            f"decision must be one of {sorted(TERMINAL_DECISIONS)}; "
            f"got {decision!r}"
        )
    if suggestion.decision in TERMINAL_DECISIONS:
        raise ProjectValidationError(
            f"Suggestion {suggestion.id} already has decision "
            f"{suggestion.decision!r}; create a new suggestion instead "
            "of overwriting the audit trail."
        )
    if not isinstance(coder_id, str) or not CODER_ID_RE.match(coder_id):
        raise ProjectValidationError(
            f"coder_id must be a 12-char hex coder id; got {coder_id!r}"
        )
    if decision in (SUGGESTION_DECISION_ACCEPTED, SUGGESTION_DECISION_MODIFIED):
        if not accepted_code_id or not CODE_ID_RE.match(accepted_code_id):
            raise ProjectValidationError(
                "accepted_code_id is required (12-char hex) when "
                f"decision is {decision!r}"
            )
        suggestion.accepted_code_id = accepted_code_id
        if accepted_application_id is not None:
            if not APPLICATION_ID_RE.match(accepted_application_id):
                raise ProjectValidationError(
                    "accepted_application_id must be 12-char hex if set"
                )
            suggestion.accepted_application_id = accepted_application_id
    else:  # rejected
        if accepted_code_id or accepted_application_id:
            raise ProjectValidationError(
                "rejected suggestions must not record an accepted "
                "code or application id"
            )
        suggestion.accepted_code_id = None
        suggestion.accepted_application_id = None
    suggestion.decision = decision
    suggestion.decided_at = now or utcnow_iso()
    suggestion.decided_by_coder_id = coder_id
    if rejection_reason:
        if len(rejection_reason) > MAX_REJECTION_REASON_LEN:
            raise ProjectValidationError(
                f"rejection_reason exceeds {MAX_REJECTION_REASON_LEN} chars"
            )
        suggestion.rejection_reason = rejection_reason
    elif decision != SUGGESTION_DECISION_REJECTED:
        # Clear any stale rejection_reason on a positive decision.
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


def suggestions_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's code suggestions.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / SUGGESTIONS_DIRNAME


def suggestion_state_path(
    projects_root: Path, project_id: str, suggestion_id: str
) -> Path:
    if not SUGGESTION_ID_RE.match(suggestion_id):
        raise ProjectValidationError(
            f"Invalid suggestion id: {suggestion_id!r}"
        )
    return (
        suggestions_dir(projects_root, project_id)
        / f"{suggestion_id}.json"
    )


def save_suggestion(projects_root: Path, suggestion: CodeSuggestion) -> Path:
    """Persist a suggestion atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    suggestion.validate()
    parent = project_dir(projects_root, suggestion.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving suggestions."
        )
    sd = suggestions_dir(projects_root, suggestion.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = suggestion_state_path(
        projects_root, suggestion.project_id, suggestion.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(suggestion.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_suggestion(
    projects_root: Path, project_id: str, suggestion_id: str
) -> CodeSuggestion:
    """Load a suggestion by id. Raises ``FileNotFoundError`` if missing."""
    p = suggestion_state_path(projects_root, project_id, suggestion_id)
    if not p.exists():
        raise FileNotFoundError(f"No suggestion at {p}")
    return CodeSuggestion.from_dict(json.loads(p.read_text()))


def list_suggestions(
    projects_root: Path,
    project_id: str,
    *,
    source_id: str | None = None,
    decision: str | None = None,
) -> list[CodeSuggestion]:
    """List all code suggestions in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse. Sorted by
    ``created_at`` ascending so the natural reading order is "the order
    in which suggestions were requested" — matches the audit-trail
    story. Files whose stem isn't a valid suggestion id are skipped.
    """
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    if decision is not None and decision not in SUGGESTION_DECISIONS:
        raise ProjectValidationError(
            f"Invalid decision filter: {decision!r}"
        )
    sd = suggestions_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[CodeSuggestion] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not SUGGESTION_ID_RE.match(sid):
            continue
        try:
            s = CodeSuggestion.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if source_id is not None and s.source_id != source_id:
            continue
        if decision is not None and s.decision != decision:
            continue
        out.append(s)
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def delete_suggestion(
    projects_root: Path, project_id: str, suggestion_id: str
) -> bool:
    """Remove a suggestion file. Returns False if it didn't exist.

    Production code should prefer keeping suggestions for the audit
    trail; deletion is exposed for tests and for the REFI-QDA import
    path (where a clean slate matters).
    """
    p = suggestion_state_path(projects_root, project_id, suggestion_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
