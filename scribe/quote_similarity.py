"""Find-similar-quotes engine for the academic-coding workflow (F8.5).

Per PLANNING.md F8.5:

  > "Find similar quotes" action on any quote (semantic search on
  > the embedding index).

Where F8.3 / F8.4 ask the model "what code(s) might apply here?",
F8.5 just asks the F8.2 index "what other quotes look like *this*
one?". No code is invoked, no LLM is called — pure embedding-space
nearest-neighbour search. That keeps it the safest of the AI
features (per PLANNING.md, Phase C ordering puts F8.5 ahead of the
suggestion features for exactly that reason: it makes no category
judgement).

What this module does
---------------------

1. **Resolve the query.** The caller can drive the search in two
   modes:

   * ``query_text=...`` — a free-form span. Embedded once via
     ``embed_fn``.
   * ``query_application_id=...`` (with the source it lives on) —
     re-uses the existing F8.2 entry for that application, so we
     don't burn an embedding call to look up something we already
     stored. Falls back to embedding the application's anchored
     text if the entry isn't in the index yet.

2. **Run the search.** Wraps :func:`scribe.embedding_index.search_similar`,
   with optional filters:

   * ``kind`` — restrict to ``coded_segment`` or
     ``uncoded_paragraph`` (default: both).
   * ``source_id`` — restrict to one source.
   * ``exclude_source_ids`` — drop matches from particular sources
     (e.g. exclude the seed's own source for cross-corpus comparison).
   * ``code_id`` / ``exclude_code_ids`` — restrict / exclude by code,
     resolved against the supplied ``applications`` list. Ignored
     for ``uncoded_paragraph`` matches (they have no code).
   * ``exclude_seed`` — drop the application/paragraph the search
     started from. Default ``True`` when seeded by application id.
   * ``min_score`` / ``top_k`` — the obvious knobs.

3. **Persist a :class:`QuoteSearch`.** Same idea as F8.3's
   :class:`CodeSuggestion` but lighter: there's no decision lifecycle
   (a search isn't a proposal, it's just a lookup). The persisted
   record carries the query, filters, and the ranked
   :class:`QuoteMatch` list. The AI invocation log (F9.6) reads from
   this directory.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.5 is the engine; the
  ``/api/projects/<id>/quote-searches`` routes are deferred and will
  be a thin shell over this module.
* **No LLM calls.** Embedding search only. (Future: re-rank could be
  bolted on the same way as F8.3, but the spec doesn't ask for it.)
* **Pure callables.** ``embed_fn`` matches the F8.2 / F8.3 shape so
  the same backend adapter drives all three engines.

This module is stand-alone — no FastAPI, no engine imports — so the
data model can be tested in pure Python and reused by the CLI later.
Conventions match the rest of the F-feature stack
(:mod:`scribe.code_suggestions`, :mod:`scribe.embedding_index`,
:mod:`scribe.applications`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .applications import APPLICATION_ID_RE, Application, parse_word_id
from .codes import CODE_ID_RE
from .coders import CODER_ID_RE
from .embedding_index import (
    EMBEDDING_KIND_CODED_SEGMENT,
    EMBEDDING_KIND_UNCODED_PARAGRAPH,
    EMBEDDING_KINDS,
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


# 12-char hex shape mirrors every other id in Scribe.
QUOTE_SEARCH_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding searches.
QUOTE_SEARCHES_DIRNAME = "quote_searches"

# How the search was driven.
QUERY_KIND_TEXT = "text"
QUERY_KIND_APPLICATION = "application"
QUERY_KINDS: tuple[str, ...] = (QUERY_KIND_TEXT, QUERY_KIND_APPLICATION)

# Defaults. Tuned for "show me ten things that look like this on a
# sidebar".
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.0
DEFAULT_EXCLUDE_SEED = True

# Bounds. Generous, but bounded so a stray upstream bug can't write
# a 50 MB record.
MAX_QUERY_TEXT_LEN = 8000
MAX_TEXT_PREVIEW_LEN = 500
MAX_MATCHES_PERSISTED = 200
MAX_NOTES_LEN = 4000
MAX_FILTER_LIST = 256          # max len of source/code id filter lists
MAX_MODEL_NAME_LEN = 256

# Allowed callable signature. Match F8.2's ``EmbedFn``.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_quote_search_id() -> str:
    """Generate a fresh 12-char hex id for a search record.

    Mirrors :func:`scribe.code_suggestions.new_suggestion_id`.
    """
    import uuid as _uuid

    return _uuid.uuid4().hex[:12]


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _coerce_score(v: Any, label: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be numeric; got {v!r}"
        ) from e
    if not _finite(f):
        raise ProjectValidationError(f"{label} must be finite; got {f!r}")
    if f < -1.0 or f > 1.0:
        raise ProjectValidationError(
            f"{label} must lie in [-1, 1]; got {f!r}"
        )
    return f


def _optional_int(v: Any, label: str) -> int | None:
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise ProjectValidationError(
            f"{label} must be an integer or null; got {v!r}"
        ) from e
    return i


def _validate_id_list(
    values: Sequence[str], pattern: re.Pattern[str], label: str
) -> tuple[str, ...]:
    out: list[str] = []
    for v in values:
        if not isinstance(v, str) or not pattern.match(v):
            raise ProjectValidationError(
                f"{label} entries must be 12-char hex; got {v!r}"
            )
        out.append(v)
    if len(out) > MAX_FILTER_LIST:
        raise ProjectValidationError(
            f"{label} has more than {MAX_FILTER_LIST} entries"
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# QuoteMatch — one row of search results
# --------------------------------------------------------------------------- #


@dataclass
class QuoteMatch:
    """One nearest-neighbour result row.

    The geometry mirrors :class:`scribe.embedding_index.EmbeddingEntry`
    so a UI can jump straight to the source span without re-querying
    the index. ``code_id`` is filled in for ``coded_segment`` matches
    when the caller passes the project's applications; otherwise
    ``None``.
    """

    embedding_id: str
    kind: str
    source_id: str
    application_id: str | None
    paragraph_start_segment: int | None
    paragraph_end_segment: int | None
    anchor_start_word_id: str
    anchor_end_word_id: str
    text_preview: str
    score: float
    code_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "kind": self.kind,
            "source_id": self.source_id,
            "application_id": self.application_id,
            "paragraph_start_segment": self.paragraph_start_segment,
            "paragraph_end_segment": self.paragraph_end_segment,
            "anchor_start_word_id": self.anchor_start_word_id,
            "anchor_end_word_id": self.anchor_end_word_id,
            "text_preview": self.text_preview,
            "score": float(self.score),
            "code_id": self.code_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuoteMatch":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("QuoteMatch payload must be an object")
        m = cls(
            embedding_id=str(d.get("embedding_id", "") or ""),
            kind=str(d.get("kind", "") or ""),
            source_id=str(d.get("source_id", "") or ""),
            application_id=(
                str(d["application_id"]) if d.get("application_id") else None
            ),
            paragraph_start_segment=_optional_int(
                d.get("paragraph_start_segment"), "paragraph_start_segment"
            ),
            paragraph_end_segment=_optional_int(
                d.get("paragraph_end_segment"), "paragraph_end_segment"
            ),
            anchor_start_word_id=str(d.get("anchor_start_word_id", "") or ""),
            anchor_end_word_id=str(d.get("anchor_end_word_id", "") or ""),
            text_preview=str(d.get("text_preview", "") or ""),
            score=_coerce_score(d.get("score", 0.0), "QuoteMatch.score"),
            code_id=(str(d["code_id"]) if d.get("code_id") else None),
        )
        m.validate()
        return m

    def validate(self) -> None:
        if not QUOTE_SEARCH_ID_RE.match(self.embedding_id):
            raise ProjectValidationError(
                f"Invalid embedding_id: {self.embedding_id!r}"
            )
        if self.kind not in EMBEDDING_KINDS:
            raise ProjectValidationError(
                f"kind must be one of {EMBEDDING_KINDS}; got {self.kind!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source_id: {self.source_id!r}"
            )
        if self.kind == EMBEDDING_KIND_CODED_SEGMENT:
            if not self.application_id or not APPLICATION_ID_RE.match(
                self.application_id
            ):
                raise ProjectValidationError(
                    "coded_segment matches require a 12-char hex application_id"
                )
            if (
                self.paragraph_start_segment is not None
                or self.paragraph_end_segment is not None
            ):
                raise ProjectValidationError(
                    "coded_segment matches must not set paragraph indices"
                )
        else:
            if self.application_id is not None:
                raise ProjectValidationError(
                    "uncoded_paragraph matches must not set application_id"
                )
            if (
                self.paragraph_start_segment is None
                or self.paragraph_end_segment is None
            ):
                raise ProjectValidationError(
                    "uncoded_paragraph matches require paragraph indices"
                )
            if self.paragraph_start_segment < 0:
                raise ProjectValidationError(
                    "paragraph_start_segment must be ≥ 0"
                )
            if self.paragraph_end_segment < self.paragraph_start_segment:
                raise ProjectValidationError(
                    "paragraph_end_segment must be ≥ paragraph_start_segment"
                )
        # Anchor shape + ordering. parse_word_id raises on garbage.
        sa_seg, sa_word = parse_word_id(self.anchor_start_word_id)
        ea_seg, ea_word = parse_word_id(self.anchor_end_word_id)
        if (sa_seg, sa_word) > (ea_seg, ea_word):
            raise ProjectValidationError(
                f"anchor_start_word_id must be ≤ anchor_end_word_id; "
                f"got {self.anchor_start_word_id!r} > "
                f"{self.anchor_end_word_id!r}"
            )
        if not isinstance(self.text_preview, str):
            raise ProjectValidationError("text_preview must be a string")
        if len(self.text_preview) > MAX_TEXT_PREVIEW_LEN:
            raise ProjectValidationError(
                f"text_preview exceeds {MAX_TEXT_PREVIEW_LEN} chars"
            )
        # Score range / finiteness already enforced via _coerce_score
        # but re-verify in case the dataclass was built directly.
        _coerce_score(self.score, "QuoteMatch.score")
        if self.code_id is not None and not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"code_id must be 12-char hex or null; got {self.code_id!r}"
            )

    @classmethod
    def from_entry(
        cls,
        entry: EmbeddingEntry,
        score: float,
        *,
        code_id: str | None = None,
    ) -> "QuoteMatch":
        """Build a :class:`QuoteMatch` from an :class:`EmbeddingEntry`."""
        return cls(
            embedding_id=entry.id,
            kind=entry.kind,
            source_id=entry.source_id,
            application_id=entry.application_id,
            paragraph_start_segment=entry.paragraph_start_segment,
            paragraph_end_segment=entry.paragraph_end_segment,
            anchor_start_word_id=entry.anchor_start_word_id,
            anchor_end_word_id=entry.anchor_end_word_id,
            text_preview=entry.text_preview,
            score=float(score),
            code_id=code_id,
        )


# --------------------------------------------------------------------------- #
# QuoteSearch — the persisted invocation
# --------------------------------------------------------------------------- #


@dataclass
class QuoteSearch:
    """One find-similar-quotes invocation, persisted as the audit record.

    A search has no decision lifecycle: results can be reviewed,
    bookmarked elsewhere, or simply read. The record stores the
    query plus the ranked match list so a researcher can re-open
    the panel later and see exactly what came back at the time —
    even if the index has since moved.
    """

    id: str
    project_id: str
    query_kind: str
    query_text: str
    query_source_id: str | None
    query_application_id: str | None
    embedding_model: str
    top_k: int
    min_score: float
    kind_filter: str | None
    source_id_filter: str | None
    exclude_source_ids: tuple[str, ...]
    code_id_filter: str | None
    exclude_code_ids: tuple[str, ...]
    exclude_seed: bool
    matches: list[QuoteMatch] = field(default_factory=list)
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
        query_kind: str,
        query_text: str,
        query_source_id: str | None = None,
        query_application_id: str | None = None,
        embedding_model: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        kind_filter: str | None = None,
        source_id_filter: str | None = None,
        exclude_source_ids: Iterable[str] = (),
        code_id_filter: str | None = None,
        exclude_code_ids: Iterable[str] = (),
        exclude_seed: bool = DEFAULT_EXCLUDE_SEED,
        matches: Iterable[QuoteMatch] | None = None,
        notes: str = "",
        search_id: str | None = None,
        now: str | None = None,
    ) -> "QuoteSearch":
        ts = now or utcnow_iso()
        coerced_matches: list[QuoteMatch] = []
        for m in matches or ():
            if isinstance(m, QuoteMatch):
                coerced_matches.append(m)
            else:
                coerced_matches.append(QuoteMatch.from_dict(m))
        s = cls(
            id=search_id or new_quote_search_id(),
            project_id=str(project_id),
            query_kind=str(query_kind),
            query_text=str(query_text or ""),
            query_source_id=(
                str(query_source_id) if query_source_id else None
            ),
            query_application_id=(
                str(query_application_id) if query_application_id else None
            ),
            embedding_model=str(embedding_model or ""),
            top_k=int(top_k),
            min_score=float(min_score),
            kind_filter=(str(kind_filter) if kind_filter else None),
            source_id_filter=(
                str(source_id_filter) if source_id_filter else None
            ),
            exclude_source_ids=tuple(str(s) for s in exclude_source_ids),
            code_id_filter=(
                str(code_id_filter) if code_id_filter else None
            ),
            exclude_code_ids=tuple(str(c) for c in exclude_code_ids),
            exclude_seed=bool(exclude_seed),
            matches=coerced_matches,
            notes=str(notes or ""),
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
            "query_kind": self.query_kind,
            "query_text": self.query_text,
            "query_source_id": self.query_source_id,
            "query_application_id": self.query_application_id,
            "embedding_model": self.embedding_model,
            "top_k": int(self.top_k),
            "min_score": float(self.min_score),
            "kind_filter": self.kind_filter,
            "source_id_filter": self.source_id_filter,
            "exclude_source_ids": list(self.exclude_source_ids),
            "code_id_filter": self.code_id_filter,
            "exclude_code_ids": list(self.exclude_code_ids),
            "exclude_seed": bool(self.exclude_seed),
            "matches": [m.to_dict() for m in self.matches],
            "notes": self.notes,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuoteSearch":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "QuoteSearch payload must be an object"
            )
        for required in ("id", "project_id", "query_kind"):
            if required not in d:
                raise ProjectValidationError(
                    f"QuoteSearch payload missing required key: {required}"
                )
        s = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            query_kind=str(d["query_kind"]),
            query_text=str(d.get("query_text", "") or ""),
            query_source_id=(
                str(d["query_source_id"]) if d.get("query_source_id") else None
            ),
            query_application_id=(
                str(d["query_application_id"])
                if d.get("query_application_id")
                else None
            ),
            embedding_model=str(d.get("embedding_model", "") or ""),
            top_k=int(d.get("top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K),
            min_score=float(
                d.get("min_score", DEFAULT_MIN_SCORE) or DEFAULT_MIN_SCORE
            ),
            kind_filter=(
                str(d["kind_filter"]) if d.get("kind_filter") else None
            ),
            source_id_filter=(
                str(d["source_id_filter"])
                if d.get("source_id_filter")
                else None
            ),
            exclude_source_ids=tuple(
                str(x) for x in (d.get("exclude_source_ids") or ())
            ),
            code_id_filter=(
                str(d["code_id_filter"]) if d.get("code_id_filter") else None
            ),
            exclude_code_ids=tuple(
                str(x) for x in (d.get("exclude_code_ids") or ())
            ),
            exclude_seed=bool(d.get("exclude_seed", DEFAULT_EXCLUDE_SEED)),
            matches=[QuoteMatch.from_dict(m) for m in (d.get("matches") or ())],
            notes=str(d.get("notes", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not QUOTE_SEARCH_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid quote-search id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if self.query_kind not in QUERY_KINDS:
            raise ProjectValidationError(
                f"query_kind must be one of {QUERY_KINDS}; "
                f"got {self.query_kind!r}"
            )
        if self.query_kind == QUERY_KIND_APPLICATION:
            if not self.query_application_id or not APPLICATION_ID_RE.match(
                self.query_application_id
            ):
                raise ProjectValidationError(
                    "application-mode searches require a 12-char hex "
                    "query_application_id"
                )
            if not self.query_source_id or not SOURCE_ID_RE.match(
                self.query_source_id
            ):
                raise ProjectValidationError(
                    "application-mode searches require a 12-char hex "
                    "query_source_id"
                )
        else:
            if self.query_application_id is not None:
                raise ProjectValidationError(
                    "text-mode searches must not set query_application_id"
                )
            if self.query_source_id is not None and not SOURCE_ID_RE.match(
                self.query_source_id
            ):
                raise ProjectValidationError(
                    f"query_source_id must be 12-char hex or null; "
                    f"got {self.query_source_id!r}"
                )
        if not isinstance(self.query_text, str):
            raise ProjectValidationError("query_text must be a string")
        if len(self.query_text) > MAX_QUERY_TEXT_LEN:
            raise ProjectValidationError(
                f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars"
            )
        if (
            not isinstance(self.embedding_model, str)
            or len(self.embedding_model) > MAX_MODEL_NAME_LEN
        ):
            raise ProjectValidationError(
                f"embedding_model must be a string ≤ {MAX_MODEL_NAME_LEN} chars"
            )
        if self.top_k < 1:
            raise ProjectValidationError(
                f"top_k must be ≥ 1; got {self.top_k}"
            )
        if not _finite(float(self.min_score)):
            raise ProjectValidationError("min_score must be finite")
        if self.min_score < -1.0 or self.min_score > 1.0:
            raise ProjectValidationError(
                "min_score must lie in [-1, 1]"
            )
        if self.kind_filter is not None and self.kind_filter not in EMBEDDING_KINDS:
            raise ProjectValidationError(
                f"kind_filter must be one of {EMBEDDING_KINDS} or null; "
                f"got {self.kind_filter!r}"
            )
        if self.source_id_filter is not None and not SOURCE_ID_RE.match(
            self.source_id_filter
        ):
            raise ProjectValidationError(
                f"source_id_filter must be 12-char hex or null; "
                f"got {self.source_id_filter!r}"
            )
        _validate_id_list(
            self.exclude_source_ids, SOURCE_ID_RE, "exclude_source_ids"
        )
        if self.code_id_filter is not None and not CODE_ID_RE.match(
            self.code_id_filter
        ):
            raise ProjectValidationError(
                f"code_id_filter must be 12-char hex or null; "
                f"got {self.code_id_filter!r}"
            )
        _validate_id_list(self.exclude_code_ids, CODE_ID_RE, "exclude_code_ids")
        if len(self.matches) > MAX_MATCHES_PERSISTED:
            raise ProjectValidationError(
                f"matches exceeds {MAX_MATCHES_PERSISTED} entries"
            )
        for m in self.matches:
            m.validate()
        if not isinstance(self.notes, str):
            raise ProjectValidationError("notes must be a string")
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def apply_update(
        self,
        *,
        notes: str | None = None,
        now: str | None = None,
    ) -> None:
        """Apply a small post-creation update (just notes for now).

        Mirrors :meth:`scribe.code_suggestions.CodeSuggestion.apply_update`
        in spirit — the matches list is immutable once written; only
        the researcher's annotation can grow.
        """
        if notes is not None:
            if not isinstance(notes, str):
                raise ProjectValidationError("notes must be a string")
            if len(notes) > MAX_NOTES_LEN:
                raise ProjectValidationError(
                    f"notes exceeds {MAX_NOTES_LEN} chars"
                )
            self.notes = notes
        self.modified_at = now or utcnow_iso()
        self.validate()


# --------------------------------------------------------------------------- #
# Search engine
# --------------------------------------------------------------------------- #


def _entry_for_application(
    entries: Sequence[EmbeddingEntry],
    *,
    source_id: str,
    application_id: str,
) -> EmbeddingEntry | None:
    for e in entries:
        if (
            e.kind == EMBEDDING_KIND_CODED_SEGMENT
            and e.source_id == source_id
            and e.application_id == application_id
        ):
            return e
    return None


def _resolve_query_vector(
    *,
    projects_root: Path,
    project_id: str,
    query_text: str,
    query_application_id: str | None,
    query_source_id: str | None,
    embed_fn: EmbedFn,
    embedding_model: str,
) -> tuple[tuple[float, ...], EmbeddingEntry | None, str]:
    """Resolve the query into (vector, seed_entry, canonical_text).

    * If ``query_application_id`` is given and the entry exists in
      the index, return its vector verbatim and the canonical text
      from its preview. No embed call.
    * Otherwise embed ``query_text`` with ``embed_fn``.
    """
    qt = canonical_text(query_text or "")
    if query_application_id:
        if not query_source_id:
            raise ProjectValidationError(
                "query_application_id requires query_source_id"
            )
        existing = list_embedding_entries(
            projects_root,
            project_id,
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=query_source_id,
        )
        seed = _entry_for_application(
            existing,
            source_id=query_source_id,
            application_id=query_application_id,
        )
        if seed is not None:
            # Optional: if caller supplied a model name and the seed
            # entry's model differs, surface that as a validation error
            # so we don't accidentally compare across model spaces.
            if (
                embedding_model
                and seed.model_name
                and embedding_model != seed.model_name
            ):
                raise ProjectValidationError(
                    f"seed entry was embedded with model "
                    f"{seed.model_name!r}; refusing to compare against "
                    f"{embedding_model!r}"
                )
            return seed.vector, seed, qt or seed.text_preview
    # Fall through: free-text embed
    if not qt:
        raise ProjectValidationError("query_text is empty after canonicalisation")
    if len(qt) > MAX_QUERY_TEXT_LEN:
        raise ProjectValidationError(
            f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars"
        )
    qv_list = list(embed_fn([qt]))
    if not qv_list:
        raise ProjectValidationError("embed_fn returned no vectors")
    qv = tuple(float(x) for x in qv_list[0])
    if not qv:
        raise ProjectValidationError("embed_fn returned an empty vector")
    return qv, None, qt


def _build_code_id_lookup(
    applications: Sequence[Application],
) -> dict[str, str]:
    """Map application_id → code_id for the supplied applications."""
    return {a.id: a.code_id for a in applications}


def find_similar_quotes(
    *,
    projects_root: Path,
    project_id: str,
    embed_fn: EmbedFn,
    applications: Sequence[Application] = (),
    query_text: str = "",
    query_application_id: str | None = None,
    query_source_id: str | None = None,
    embedding_model: str = "",
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    kind_filter: str | None = None,
    source_id_filter: str | None = None,
    exclude_source_ids: Sequence[str] = (),
    code_id_filter: str | None = None,
    exclude_code_ids: Sequence[str] = (),
    exclude_seed: bool = DEFAULT_EXCLUDE_SEED,
    notes: str = "",
    now: str | None = None,
) -> QuoteSearch:
    """End-to-end nearest-neighbour search over the embedding index.

    Workflow:

    1. Resolve the query vector. Application-id mode prefers the
       existing F8.2 entry; text mode embeds via ``embed_fn``.
    2. Load index entries (filtered by ``kind_filter`` /
       ``source_id_filter`` if set).
    3. Score every remaining entry; apply ``min_score``,
       ``exclude_source_ids``, ``exclude_seed``, ``code_id_filter``
       / ``exclude_code_ids``, dim mismatch.
    4. Sort by score descending (stable on entry id), truncate to
       ``top_k``.
    5. Wrap in a :class:`QuoteSearch`. Caller persists via
       :func:`save_quote_search`.

    The ``code_id_filter`` and ``exclude_code_ids`` knobs only narrow
    coded-segment matches. ``uncoded_paragraph`` matches always pass
    code-id filters (they have no code), but are dropped when
    ``code_id_filter`` is set — "find similar quotes coded as X" is
    by definition a coded-segment query.
    """
    if top_k < 1:
        raise ProjectValidationError(f"top_k must be ≥ 1; got {top_k}")
    if not _finite(float(min_score)) or min_score < -1.0 or min_score > 1.0:
        raise ProjectValidationError("min_score must be finite in [-1, 1]")
    if kind_filter is not None and kind_filter not in EMBEDDING_KINDS:
        raise ProjectValidationError(
            f"kind_filter must be one of {EMBEDDING_KINDS} or null; "
            f"got {kind_filter!r}"
        )
    if source_id_filter is not None and not SOURCE_ID_RE.match(source_id_filter):
        raise ProjectValidationError(
            f"source_id_filter must be 12-char hex or null; "
            f"got {source_id_filter!r}"
        )
    excl_sources = _validate_id_list(
        tuple(exclude_source_ids), SOURCE_ID_RE, "exclude_source_ids"
    )
    if code_id_filter is not None and not CODE_ID_RE.match(code_id_filter):
        raise ProjectValidationError(
            f"code_id_filter must be 12-char hex or null; "
            f"got {code_id_filter!r}"
        )
    excl_codes = _validate_id_list(
        tuple(exclude_code_ids), CODE_ID_RE, "exclude_code_ids"
    )
    # Decide query mode
    if query_application_id:
        if not APPLICATION_ID_RE.match(query_application_id):
            raise ProjectValidationError(
                f"query_application_id must be 12-char hex; "
                f"got {query_application_id!r}"
            )
        if not query_source_id or not SOURCE_ID_RE.match(query_source_id):
            raise ProjectValidationError(
                "query_application_id requires a valid query_source_id"
            )
        query_kind = QUERY_KIND_APPLICATION
    else:
        query_kind = QUERY_KIND_TEXT
        if query_source_id is not None and not SOURCE_ID_RE.match(query_source_id):
            raise ProjectValidationError(
                f"query_source_id must be 12-char hex or null; "
                f"got {query_source_id!r}"
            )

    # 1. Query vector
    qv, seed_entry, canonical_query = _resolve_query_vector(
        projects_root=projects_root,
        project_id=project_id,
        query_text=query_text,
        query_application_id=query_application_id,
        query_source_id=query_source_id,
        embed_fn=embed_fn,
        embedding_model=embedding_model,
    )

    # 2. Load entries (with kind/source filters honoured up front).
    entries = list_embedding_entries(
        projects_root,
        project_id,
        kind=kind_filter,
        source_id=source_id_filter,
    )

    # 3. Score + filter.
    code_by_app = _build_code_id_lookup(applications)
    excl_source_set = set(excl_sources)
    excl_code_set = set(excl_codes)
    scored: list[tuple[float, EmbeddingEntry, str | None]] = []
    for e in entries:
        if e.dim != len(qv):
            continue
        if exclude_seed and seed_entry is not None and e.id == seed_entry.id:
            continue
        if exclude_seed and query_application_id and (
            e.kind == EMBEDDING_KIND_CODED_SEGMENT
            and e.application_id == query_application_id
            and e.source_id == query_source_id
        ):
            continue
        if e.source_id in excl_source_set:
            continue
        # Code-id filtering — applies to coded_segment entries.
        match_code_id: str | None = None
        if e.kind == EMBEDDING_KIND_CODED_SEGMENT and e.application_id:
            match_code_id = code_by_app.get(e.application_id)
        if code_id_filter is not None:
            # "find similar quotes coded as X" — only coded segments
            # are eligible.
            if e.kind != EMBEDDING_KIND_CODED_SEGMENT:
                continue
            if match_code_id != code_id_filter:
                continue
        if match_code_id is not None and match_code_id in excl_code_set:
            continue
        score = cosine_similarity(qv, e.vector)
        if not _finite(score) or score < min_score:
            continue
        scored.append((score, e, match_code_id))

    # 4. Sort: score desc, tie-break by entry id asc.
    scored.sort(key=lambda t: (-t[0], t[1].id))
    top = scored[: max(1, int(top_k))]
    matches = [
        QuoteMatch.from_entry(e, score=s, code_id=cid)
        for (s, e, cid) in top
    ]

    # 5. Wrap.
    search = QuoteSearch.new(
        project_id=project_id,
        query_kind=query_kind,
        query_text=canonical_query,
        query_source_id=query_source_id,
        query_application_id=query_application_id,
        embedding_model=(
            embedding_model
            or (seed_entry.model_name if seed_entry is not None else "")
        ),
        top_k=int(top_k),
        min_score=float(min_score),
        kind_filter=kind_filter,
        source_id_filter=source_id_filter,
        exclude_source_ids=excl_sources,
        code_id_filter=code_id_filter,
        exclude_code_ids=excl_codes,
        exclude_seed=bool(exclude_seed),
        matches=matches,
        notes=notes,
        now=now,
    )
    return search


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def quote_searches_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's quote searches.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / QUOTE_SEARCHES_DIRNAME


def quote_search_state_path(
    projects_root: Path, project_id: str, search_id: str
) -> Path:
    if not QUOTE_SEARCH_ID_RE.match(search_id):
        raise ProjectValidationError(
            f"Invalid quote-search id: {search_id!r}"
        )
    return (
        quote_searches_dir(projects_root, project_id)
        / f"{search_id}.json"
    )


def save_quote_search(projects_root: Path, search: QuoteSearch) -> Path:
    """Persist a quote-search atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    search.validate()
    parent = project_dir(projects_root, search.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving quote searches."
        )
    sd = quote_searches_dir(projects_root, search.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = quote_search_state_path(
        projects_root, search.project_id, search.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(search.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_quote_search(
    projects_root: Path, project_id: str, search_id: str
) -> QuoteSearch:
    """Load a quote search by id. Raises ``FileNotFoundError`` if missing."""
    p = quote_search_state_path(projects_root, project_id, search_id)
    if not p.exists():
        raise FileNotFoundError(f"No quote search at {p}")
    return QuoteSearch.from_dict(json.loads(p.read_text()))


def list_quote_searches(
    projects_root: Path,
    project_id: str,
    *,
    query_kind: str | None = None,
    query_source_id: str | None = None,
) -> list[QuoteSearch]:
    """List all quote searches in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse. Sorted by
    ``created_at`` ascending so the natural reading order is "the
    order in which searches were requested" — matches the audit-trail
    story. Files whose stem isn't a valid id are skipped.
    """
    if query_kind is not None and query_kind not in QUERY_KINDS:
        raise ProjectValidationError(
            f"Invalid query_kind filter: {query_kind!r}"
        )
    if query_source_id is not None and not SOURCE_ID_RE.match(query_source_id):
        raise ProjectValidationError(
            f"Invalid query_source_id filter: {query_source_id!r}"
        )
    sd = quote_searches_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[QuoteSearch] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not QUOTE_SEARCH_ID_RE.match(sid):
            continue
        try:
            s = QuoteSearch.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if query_kind is not None and s.query_kind != query_kind:
            continue
        if query_source_id is not None and s.query_source_id != query_source_id:
            continue
        out.append(s)
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def delete_quote_search(
    projects_root: Path, project_id: str, search_id: str
) -> bool:
    """Remove a search file. Returns False if it didn't exist.

    Production code should prefer keeping searches for the audit trail
    (F9.6); deletion is exposed for tests and the REFI-QDA import path.
    """
    p = quote_search_state_path(projects_root, project_id, search_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
