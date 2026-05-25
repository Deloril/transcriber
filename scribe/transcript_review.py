"""Whole-transcript AI review pass (F8.6).

Per PLANNING.md F8.6:

  > Whole-transcript AI review pass as a background job. Produces a
  > list of suggestions for review; never auto-applies.

Where F8.3 / F8.4 ask for code suggestions on **one selected span**,
F8.6 walks the **entire transcript** and produces a stream of
:class:`scribe.code_suggestions.CodeSuggestion` records — one per
candidate span — each starting in the same ``"pending"`` state. A
researcher then reviews them in the UI, accepting / modifying /
rejecting individually. Nothing is auto-applied; F8.6 is plumbing
around the existing F8.3 engine.

What this module does
---------------------

1. **Enumerate review items.** Given the transcript's segments and
   the project's applications, decide which spans to review:

   * ``granularity="paragraph"`` — one item per speaker-turn
     (paragraph), as defined by :func:`scribe.selection_snap.paragraph_ranges`.
   * ``granularity="sentence"`` — one item per sentence inside each
     segment, via
     :func:`scribe.selection_snap.sentence_ranges_in_segment`.

   Items whose extracted text canonicalises to empty are skipped (an
   embedder will reject them anyway). When ``skip_already_coded=True``
   (the default), spans that overlap *any* existing application are
   skipped — the pass is meant to surface what the human missed.
   Setting ``skip_already_coded=False`` produces a full sweep, useful
   for second-coder-style passes.

2. **Persist a ReviewPass record.** :func:`start_review_pass` builds
   a :class:`ReviewPass` with the chosen items and saves it under
   ``projects/<pid>/review_passes/<rpid>.json``. The status starts at
   ``"pending"``. Items are part of the persisted record so the pass
   is **resumable**: if the worker dies, a future invocation can
   pick up exactly where it left off.

3. **Process items one at a time.**
   :func:`process_next_review_item` finds the first not-yet-processed
   item and:

     * delegates to :func:`scribe.code_suggestions.suggest_codes_for_span`;
     * persists the resulting :class:`CodeSuggestion`;
     * stamps the item with the suggestion id and the pass with
       ``status="running"``;
     * flips the pass to ``"completed"`` once every item has either
       a ``suggestion_id`` or an ``error`` recorded.

   Per-item errors (the embedder raised; the LLM call timed out)
   stop *that item* but **don't** abort the pass — the error string
   lands on the item and the pass continues. Pass-level failures
   (the project disappeared mid-pass; an invariant broke) are
   surfaced via :func:`mark_review_pass_failed`.

4. **Drive the loop.** :func:`run_review_pass` is a thin synchronous
   wrapper around :func:`process_next_review_item`. The "background
   job" framing in the spec refers to *how* the server schedules it
   (a worker thread); this engine just exposes a step function so a
   worker can call it repeatedly and the UI can poll progress.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.6 is the engine; the
  ``/api/projects/<id>/review-passes`` routes are deferred and will
  be a thin shell over this module, mirroring the F8.3 / F8.5 split.
* **No automatic application creation.** All decisions stay on the
  individual :class:`CodeSuggestion` records the pass produces.
  Accepting / modifying / rejecting goes through the F8.3 decision
  recorder.
* **Pure callables.** ``embed_fn`` and ``generate_fn`` match the
  F8.1 / F8.3 shapes so the same backend adapter drives all the
  engines.

This module is stand-alone — no FastAPI, no engine imports — so
the data model can be tested in pure Python and reused by the CLI
later. Conventions match the rest of the F-feature stack
(:mod:`scribe.code_suggestions`, :mod:`scribe.embedding_index`,
:mod:`scribe.quote_similarity`).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .applications import Application, parse_word_id
from .code_suggestions import (
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TOP_K,
    CodeSuggestion,
    EmbedFn,
    GenerateFn,
    save_suggestion,
    suggest_codes_for_span,
)
from .codes import Code
from .embedding_index import (
    canonical_text,
    extract_paragraph_text,
)
from .application_reanchor import collect_word_texts
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .selection_snap import paragraph_ranges, sentence_ranges_in_segment
from .sources import SOURCE_ID_RE


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# 12-char hex, same shape as every other id in Scribe.
REVIEW_PASS_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding pass records.
REVIEW_PASSES_DIRNAME = "review_passes"

# Pass-level lifecycle states.
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_RUNNING = "running"
REVIEW_STATUS_COMPLETED = "completed"
REVIEW_STATUS_CANCELLED = "cancelled"
REVIEW_STATUS_FAILED = "failed"
REVIEW_STATUSES: tuple[str, ...] = (
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_RUNNING,
    REVIEW_STATUS_COMPLETED,
    REVIEW_STATUS_CANCELLED,
    REVIEW_STATUS_FAILED,
)
REVIEW_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        REVIEW_STATUS_COMPLETED,
        REVIEW_STATUS_CANCELLED,
        REVIEW_STATUS_FAILED,
    }
)

# Granularity. Closed set — anything else is a validation error.
REVIEW_GRANULARITY_PARAGRAPH = "paragraph"
REVIEW_GRANULARITY_SENTENCE = "sentence"
REVIEW_GRANULARITIES: tuple[str, ...] = (
    REVIEW_GRANULARITY_PARAGRAPH,
    REVIEW_GRANULARITY_SENTENCE,
)

# Bounds. Generous, but bounded so a stray upstream bug can't write a
# 50 MB pass record.
MAX_TEXT_PREVIEW_LEN = 500
MAX_ITEMS = 5000              # whole-transcript review on a long interview
MAX_NOTES_LEN = 4000
MAX_ERROR_MESSAGE_LEN = 4000
MAX_ITEM_ERROR_LEN = 2000
MAX_MODEL_NAME_LEN = 256


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_review_pass_id() -> str:
    """Mint a fresh 12-char hex id for a pass record."""
    return uuid.uuid4().hex[:12]


def is_terminal_status(status: str) -> bool:
    """Return True if ``status`` is one of the closed-set terminal states."""
    return status in REVIEW_TERMINAL_STATUSES


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


# --------------------------------------------------------------------------- #
# ReviewItem — one span scheduled for review
# --------------------------------------------------------------------------- #


@dataclass
class ReviewItem:
    """One scheduled span in a :class:`ReviewPass`.

    ``suggestion_id`` is set on success; ``error`` on a per-item
    failure. Mutually exclusive — both empty means "still pending".

    The anchors are word ids (``s<segment>w<word>``) just like
    everywhere else in Scribe; the paragraph indices are stored
    redundantly so the UI can show "Paragraph 5/27" without
    re-running the snap helpers.
    """

    anchor_start_word_id: str
    anchor_end_word_id: str
    paragraph_start_segment: int
    paragraph_end_segment: int
    text_preview: str
    suggestion_id: str | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_start_word_id": self.anchor_start_word_id,
            "anchor_end_word_id": self.anchor_end_word_id,
            "paragraph_start_segment": int(self.paragraph_start_segment),
            "paragraph_end_segment": int(self.paragraph_end_segment),
            "text_preview": self.text_preview,
            "suggestion_id": self.suggestion_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReviewItem":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("ReviewItem payload must be an object")
        item = cls(
            anchor_start_word_id=str(d.get("anchor_start_word_id", "") or ""),
            anchor_end_word_id=str(d.get("anchor_end_word_id", "") or ""),
            paragraph_start_segment=_coerce_int(
                d.get("paragraph_start_segment"), "paragraph_start_segment"
            ),
            paragraph_end_segment=_coerce_int(
                d.get("paragraph_end_segment"), "paragraph_end_segment"
            ),
            text_preview=str(d.get("text_preview", "") or ""),
            suggestion_id=(
                str(d["suggestion_id"]) if d.get("suggestion_id") else None
            ),
            error=str(d.get("error", "") or ""),
        )
        item.validate()
        return item

    def validate(self) -> None:
        # Anchor shape + ordering. parse_word_id raises on garbage.
        sa = parse_word_id(self.anchor_start_word_id)
        ea = parse_word_id(self.anchor_end_word_id)
        if sa > ea:
            raise ProjectValidationError(
                f"anchor_start_word_id must be ≤ anchor_end_word_id; "
                f"got {self.anchor_start_word_id!r} > "
                f"{self.anchor_end_word_id!r}"
            )
        if self.paragraph_start_segment < 0:
            raise ProjectValidationError(
                "paragraph_start_segment must be ≥ 0"
            )
        if self.paragraph_end_segment < self.paragraph_start_segment:
            raise ProjectValidationError(
                "paragraph_end_segment must be ≥ paragraph_start_segment"
            )
        if not isinstance(self.text_preview, str):
            raise ProjectValidationError("text_preview must be a string")
        if len(self.text_preview) > MAX_TEXT_PREVIEW_LEN:
            raise ProjectValidationError(
                f"text_preview exceeds {MAX_TEXT_PREVIEW_LEN} chars"
            )
        if self.suggestion_id is not None:
            if not REVIEW_PASS_ID_RE.match(self.suggestion_id):
                raise ProjectValidationError(
                    f"suggestion_id must be 12-char hex or null; "
                    f"got {self.suggestion_id!r}"
                )
        if not isinstance(self.error, str):
            raise ProjectValidationError("error must be a string")
        if len(self.error) > MAX_ITEM_ERROR_LEN:
            raise ProjectValidationError(
                f"error exceeds {MAX_ITEM_ERROR_LEN} chars"
            )
        # success/failure are mutually exclusive.
        if self.suggestion_id and self.error:
            raise ProjectValidationError(
                "ReviewItem cannot have both suggestion_id and error set"
            )

    @property
    def is_pending(self) -> bool:
        """A pending item has neither a suggestion id nor an error."""
        return not self.suggestion_id and not self.error

    @property
    def is_processed(self) -> bool:
        """A processed item has either a suggestion_id or an error."""
        return bool(self.suggestion_id) or bool(self.error)


# --------------------------------------------------------------------------- #
# Span enumeration
# --------------------------------------------------------------------------- #


def _segments_touched_by_application(
    application: Application,
    segments: Sequence[Mapping[str, Any]],
) -> set[int]:
    """Return the set of segment indices the application's anchor spans."""
    sa_seg, _ = parse_word_id(application.anchor_start_word_id)
    ea_seg, _ = parse_word_id(application.anchor_end_word_id)
    if sa_seg < 0:
        sa_seg = 0
    if ea_seg >= len(segments):
        ea_seg = len(segments) - 1
    if ea_seg < sa_seg:
        return set()
    return set(range(sa_seg, ea_seg + 1))


def _word_indices_touched_by_application(
    application: Application,
    seg_idx: int,
    seg_word_count: int,
) -> set[int]:
    """Return word indices in ``seg_idx`` covered by the application.

    For multi-segment anchors, internal segments are fully covered;
    the start segment is covered from the start word onward; the end
    segment up to the end word inclusive.
    """
    sa_seg, sa_word = parse_word_id(application.anchor_start_word_id)
    ea_seg, ea_word = parse_word_id(application.anchor_end_word_id)
    if seg_idx < sa_seg or seg_idx > ea_seg:
        return set()
    if seg_word_count <= 0:
        return set()
    if sa_seg < seg_idx < ea_seg:
        return set(range(seg_word_count))
    lo = sa_word if seg_idx == sa_seg else 0
    hi = ea_word if seg_idx == ea_seg else seg_word_count - 1
    if lo < 0:
        lo = 0
    if hi >= seg_word_count:
        hi = seg_word_count - 1
    if hi < lo:
        return set()
    return set(range(lo, hi + 1))


def _extract_sentence_text(
    seg_words: Sequence[Mapping[str, Any]],
    word_start: int,
    word_end: int,
) -> str:
    """Return the canonicalised text spanning words[word_start..word_end] in a segment."""
    if not seg_words or word_start < 0 or word_end < word_start:
        return ""
    word_end = min(word_end, len(seg_words) - 1)
    pieces: list[str] = []
    for i in range(word_start, word_end + 1):
        w = seg_words[i]
        if isinstance(w, Mapping):
            t = str(w.get("text", "") or "")
        else:
            t = ""
        if t:
            pieces.append(t)
    return canonical_text(" ".join(pieces))


def enumerate_review_items(
    *,
    source_id: str,
    segments: Sequence[Mapping[str, Any]],
    applications: Sequence[Application] = (),
    granularity: str = REVIEW_GRANULARITY_PARAGRAPH,
    skip_already_coded: bool = True,
) -> list[ReviewItem]:
    """Compute the spans a whole-transcript review pass should visit.

    * ``granularity="paragraph"`` — one item per
      :func:`paragraph_ranges` entry. Skipped when
      ``skip_already_coded=True`` and any application touches any
      segment in the paragraph.
    * ``granularity="sentence"`` — one item per sentence inside each
      segment, using :func:`sentence_ranges_in_segment`. Skipped when
      ``skip_already_coded=True`` and any application word index
      falls inside the sentence.

    Empty paragraphs / sentences (no words after canonicalisation)
    are dropped — embedders reject empty input anyway and there is
    nothing to suggest about a silent stretch.

    Order of the returned list:

    * Paragraph mode: paragraph order in the transcript.
    * Sentence mode: segment index, then sentence index within that
      segment. Mirrors how a researcher would scroll through.
    """
    if not isinstance(source_id, str) or not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id: {source_id!r}"
        )
    if granularity not in REVIEW_GRANULARITIES:
        raise ProjectValidationError(
            f"granularity must be one of {REVIEW_GRANULARITIES}; "
            f"got {granularity!r}"
        )
    apps = [a for a in applications if a.source_id == source_id]
    items: list[ReviewItem] = []
    n = len(segments)
    if n == 0:
        return items
    words_2d = collect_word_texts(segments)

    if granularity == REVIEW_GRANULARITY_PARAGRAPH:
        # Pre-compute touched segments per source for skip filtering.
        touched: set[int] = set()
        for a in apps:
            touched.update(_segments_touched_by_application(a, segments))
        for p_start, p_end in paragraph_ranges(segments):
            if skip_already_coded and any(
                i in touched for i in range(p_start, p_end + 1)
            ):
                continue
            text = extract_paragraph_text(segments, p_start, p_end)
            if not text:
                continue
            # Find first/last segment with words, like F8.2 does.
            first = None
            last = None
            for si in range(p_start, p_end + 1):
                if words_2d[si]:
                    if first is None:
                        first = si
                    last = si
            if first is None or last is None:
                continue
            anchor_start = f"s{first}w0"
            anchor_end = f"s{last}w{len(words_2d[last]) - 1}"
            items.append(
                ReviewItem(
                    anchor_start_word_id=anchor_start,
                    anchor_end_word_id=anchor_end,
                    paragraph_start_segment=p_start,
                    paragraph_end_segment=p_end,
                    text_preview=text[:MAX_TEXT_PREVIEW_LEN],
                )
            )
    else:  # sentence
        # For sentence mode, paragraph_start/end per item is the
        # segment that hosts the sentence (a sentence never spans
        # speaker turns in our model — sentence_ranges_in_segment is
        # per-segment). The paragraph-segment fields therefore equal
        # the host segment index on both ends.
        for si in range(n):
            seg = segments[si]
            seg_words_raw: Sequence[Mapping[str, Any]]
            if isinstance(seg, Mapping):
                wlist = seg.get("words", [])
                seg_words_raw = wlist if isinstance(wlist, Sequence) else []
            else:
                seg_words_raw = []
            ranges = sentence_ranges_in_segment(seg_words_raw)
            if not ranges:
                continue
            # Compute touched word indices for this segment once.
            touched_words: set[int] = set()
            if skip_already_coded:
                for a in apps:
                    touched_words.update(
                        _word_indices_touched_by_application(
                            a, si, len(words_2d[si])
                        )
                    )
            for w_start, w_end in ranges:
                if skip_already_coded and any(
                    w in touched_words for w in range(w_start, w_end + 1)
                ):
                    continue
                text = _extract_sentence_text(
                    seg_words_raw, w_start, w_end
                )
                if not text:
                    continue
                items.append(
                    ReviewItem(
                        anchor_start_word_id=f"s{si}w{w_start}",
                        anchor_end_word_id=f"s{si}w{w_end}",
                        paragraph_start_segment=si,
                        paragraph_end_segment=si,
                        text_preview=text[:MAX_TEXT_PREVIEW_LEN],
                    )
                )
    if len(items) > MAX_ITEMS:
        raise ProjectValidationError(
            f"review pass would enumerate {len(items)} items "
            f"(max {MAX_ITEMS}); narrow the source or use sentence mode "
            f"selectively"
        )
    return items


# --------------------------------------------------------------------------- #
# ReviewPass dataclass
# --------------------------------------------------------------------------- #


@dataclass
class ReviewPass:
    """One whole-transcript review-pass invocation, persisted as audit.

    The ``items`` list is fixed at construction time and never reshaped;
    individual items mutate as work progresses. Status moves
    ``pending → running → completed | cancelled | failed`` and never
    flows out of a terminal state — that would rewrite the audit
    trail; create a new pass instead.
    """

    id: str
    project_id: str
    source_id: str
    status: str = REVIEW_STATUS_PENDING
    granularity: str = REVIEW_GRANULARITY_PARAGRAPH
    skip_already_coded: bool = True
    embedding_model: str = ""
    generation_model: str = ""
    top_k: int = DEFAULT_TOP_K
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT
    min_score: float = 0.0
    items: list[ReviewItem] = field(default_factory=list)
    error_message: str = ""
    notes: str = ""
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
        items: Iterable[ReviewItem | Mapping[str, Any]] = (),
        granularity: str = REVIEW_GRANULARITY_PARAGRAPH,
        skip_already_coded: bool = True,
        embedding_model: str = "",
        generation_model: str = "",
        top_k: int = DEFAULT_TOP_K,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        min_score: float = 0.0,
        notes: str = "",
        pass_id: str | None = None,
        now: str | None = None,
    ) -> "ReviewPass":
        ts = now or utcnow_iso()
        coerced: list[ReviewItem] = []
        for it in items:
            if isinstance(it, ReviewItem):
                coerced.append(it)
            else:
                coerced.append(ReviewItem.from_dict(it))
        p = cls(
            id=pass_id or new_review_pass_id(),
            project_id=str(project_id),
            source_id=str(source_id),
            status=REVIEW_STATUS_PENDING,
            granularity=str(granularity),
            skip_already_coded=bool(skip_already_coded),
            embedding_model=str(embedding_model or ""),
            generation_model=str(generation_model or ""),
            top_k=int(top_k),
            max_candidates=int(max_candidates),
            embedding_weight=float(embedding_weight),
            min_score=float(min_score),
            items=coerced,
            error_message="",
            notes=str(notes or ""),
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
            "status": self.status,
            "granularity": self.granularity,
            "skip_already_coded": bool(self.skip_already_coded),
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "top_k": int(self.top_k),
            "max_candidates": int(self.max_candidates),
            "embedding_weight": float(self.embedding_weight),
            "min_score": float(self.min_score),
            "items": [it.to_dict() for it in self.items],
            "error_message": self.error_message,
            "notes": self.notes,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReviewPass":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "ReviewPass payload must be an object"
            )
        for required in ("id", "project_id", "source_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"ReviewPass payload missing required key: {required}"
                )
        p = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            source_id=str(d["source_id"]),
            status=str(d.get("status", REVIEW_STATUS_PENDING) or REVIEW_STATUS_PENDING),
            granularity=str(
                d.get("granularity", REVIEW_GRANULARITY_PARAGRAPH)
                or REVIEW_GRANULARITY_PARAGRAPH
            ),
            skip_already_coded=bool(d.get("skip_already_coded", True)),
            embedding_model=str(d.get("embedding_model", "") or ""),
            generation_model=str(d.get("generation_model", "") or ""),
            top_k=_coerce_int(
                d.get("top_k", DEFAULT_TOP_K), "top_k", default=DEFAULT_TOP_K
            ),
            max_candidates=_coerce_int(
                d.get("max_candidates", DEFAULT_MAX_CANDIDATES),
                "max_candidates",
                default=DEFAULT_MAX_CANDIDATES,
            ),
            embedding_weight=_coerce_float(
                d.get("embedding_weight", DEFAULT_EMBEDDING_WEIGHT),
                "embedding_weight",
                default=DEFAULT_EMBEDDING_WEIGHT,
            ),
            min_score=_coerce_float(
                d.get("min_score", 0.0), "min_score", default=0.0
            ),
            items=[ReviewItem.from_dict(it) for it in (d.get("items") or ())],
            error_message=str(d.get("error_message", "") or ""),
            notes=str(d.get("notes", "") or ""),
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
        if not REVIEW_PASS_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid review-pass id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        if self.status not in REVIEW_STATUSES:
            raise ProjectValidationError(
                f"status must be one of {REVIEW_STATUSES}; "
                f"got {self.status!r}"
            )
        if self.granularity not in REVIEW_GRANULARITIES:
            raise ProjectValidationError(
                f"granularity must be one of {REVIEW_GRANULARITIES}; "
                f"got {self.granularity!r}"
            )
        if (
            not isinstance(self.embedding_model, str)
            or len(self.embedding_model) > MAX_MODEL_NAME_LEN
        ):
            raise ProjectValidationError(
                f"embedding_model must be a string ≤ {MAX_MODEL_NAME_LEN} chars"
            )
        if (
            not isinstance(self.generation_model, str)
            or len(self.generation_model) > MAX_MODEL_NAME_LEN
        ):
            raise ProjectValidationError(
                f"generation_model must be a string ≤ {MAX_MODEL_NAME_LEN} chars"
            )
        if self.top_k < 1:
            raise ProjectValidationError(
                f"top_k must be ≥ 1; got {self.top_k}"
            )
        if self.max_candidates < 1:
            raise ProjectValidationError(
                f"max_candidates must be ≥ 1; got {self.max_candidates}"
            )
        if not (0.0 <= self.embedding_weight <= 1.0):
            raise ProjectValidationError(
                f"embedding_weight must be in [0, 1]; got {self.embedding_weight}"
            )
        if self.min_score < -1.0 or self.min_score > 1.0:
            raise ProjectValidationError(
                f"min_score must be in [-1, 1]; got {self.min_score}"
            )
        if len(self.items) > MAX_ITEMS:
            raise ProjectValidationError(
                f"items exceeds {MAX_ITEMS} entries"
            )
        for it in self.items:
            it.validate()
        if not isinstance(self.error_message, str):
            raise ProjectValidationError("error_message must be a string")
        if len(self.error_message) > MAX_ERROR_MESSAGE_LEN:
            raise ProjectValidationError(
                f"error_message exceeds {MAX_ERROR_MESSAGE_LEN} chars"
            )
        if not isinstance(self.notes, str):
            raise ProjectValidationError("notes must be a string")
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )
        # Status / timestamp invariants.
        if self.status == REVIEW_STATUS_PENDING:
            if self.started_at:
                raise ProjectValidationError(
                    "pending pass must not have started_at"
                )
            if self.completed_at:
                raise ProjectValidationError(
                    "pending pass must not have completed_at"
                )
        if self.status == REVIEW_STATUS_RUNNING:
            if not self.started_at:
                raise ProjectValidationError(
                    "running pass must have started_at"
                )
            if self.completed_at:
                raise ProjectValidationError(
                    "running pass must not have completed_at"
                )
        if self.status in REVIEW_TERMINAL_STATUSES:
            if not self.started_at:
                raise ProjectValidationError(
                    f"{self.status!r} pass must have started_at"
                )
            if not self.completed_at:
                raise ProjectValidationError(
                    f"{self.status!r} pass must have completed_at"
                )
        if self.status == REVIEW_STATUS_FAILED and not self.error_message:
            # We allow empty error_message for legacy records but
            # warn-level invariants are: if not failed, error_message
            # should not be set.
            pass
        if self.status != REVIEW_STATUS_FAILED and self.error_message:
            raise ProjectValidationError(
                "error_message must only be set when status='failed'"
            )

    # ------------------------------------------------------------------ #
    # Progress
    # ------------------------------------------------------------------ #

    @property
    def total_spans(self) -> int:
        return len(self.items)

    @property
    def completed_spans(self) -> int:
        """Count items with either a suggestion id or a per-item error."""
        return sum(1 for it in self.items if it.is_processed)

    @property
    def succeeded_spans(self) -> int:
        return sum(1 for it in self.items if it.suggestion_id)

    @property
    def failed_spans(self) -> int:
        return sum(1 for it in self.items if it.error)

    def pending_indices(self) -> list[int]:
        """Return the indices of items still waiting to be processed."""
        return [i for i, it in enumerate(self.items) if it.is_pending]

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: Mapping[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update — currently only ``notes``.

        Status transitions go through dedicated functions so the
        invariants stay enforced; ``items`` mutation goes through
        :func:`process_next_review_item`.
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
# Lifecycle
# --------------------------------------------------------------------------- #


def start_review_pass(
    *,
    projects_root: Path,
    project_id: str,
    source_id: str,
    segments: Sequence[Mapping[str, Any]],
    applications: Sequence[Application] = (),
    granularity: str = REVIEW_GRANULARITY_PARAGRAPH,
    skip_already_coded: bool = True,
    embedding_model: str = "",
    generation_model: str = "",
    top_k: int = DEFAULT_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    min_score: float = 0.0,
    notes: str = "",
    pass_id: str | None = None,
    now: str | None = None,
) -> ReviewPass:
    """Build, persist, and return a fresh :class:`ReviewPass`.

    Items are enumerated once via :func:`enumerate_review_items` and
    frozen onto the pass record so subsequent steps are reproducible
    even if the transcript is later edited or new applications land.

    Status starts at ``"pending"``. ``started_at`` / ``completed_at``
    are blank.
    """
    items = enumerate_review_items(
        source_id=source_id,
        segments=segments,
        applications=applications,
        granularity=granularity,
        skip_already_coded=skip_already_coded,
    )
    rp = ReviewPass.new(
        project_id=project_id,
        source_id=source_id,
        items=items,
        granularity=granularity,
        skip_already_coded=skip_already_coded,
        embedding_model=embedding_model,
        generation_model=generation_model,
        top_k=top_k,
        max_candidates=max_candidates,
        embedding_weight=embedding_weight,
        min_score=min_score,
        notes=notes,
        pass_id=pass_id,
        now=now,
    )
    save_review_pass(projects_root, rp)
    return rp


def process_next_review_item(
    pass_record: ReviewPass,
    *,
    projects_root: Path,
    codes: Sequence[Code],
    applications: Sequence[Application],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn | None = None,
    now: str | None = None,
) -> tuple[int, CodeSuggestion | None]:
    """Process the next pending item via the F8.3 suggestion engine.

    Mutates ``pass_record`` in place:

    * On the first call from ``"pending"``, flips status to
      ``"running"`` and stamps ``started_at``.
    * On success, sets the item's ``suggestion_id`` and persists
      both the new :class:`CodeSuggestion` and the pass.
    * On a per-item failure (e.g. embed_fn raised, or
      ``suggest_codes_for_span`` raised
      :class:`ProjectValidationError`), records ``error`` on the
      item and keeps the pass running.
    * When no items remain pending after this step, flips status to
      ``"completed"`` and stamps ``completed_at``.

    Returns ``(item_index, suggestion)``; ``suggestion`` is ``None``
    on a per-item failure. Raises :class:`ProjectValidationError`
    when called on a pass that is already in a terminal state, or
    when the pass has no pending items at the start of the call.
    """
    if pass_record.status in REVIEW_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot process: pass {pass_record.id} is in terminal "
            f"state {pass_record.status!r}"
        )
    pending = pass_record.pending_indices()
    if not pending:
        # Defensive: caller should have flipped to completed already.
        # Do that now so the state is consistent.
        ts = now or utcnow_iso()
        pass_record.status = REVIEW_STATUS_COMPLETED
        if not pass_record.started_at:
            pass_record.started_at = ts
        pass_record.completed_at = ts
        pass_record.modified_at = ts
        save_review_pass(projects_root, pass_record)
        raise ProjectValidationError(
            f"Pass {pass_record.id} has no pending items to process"
        )

    idx = pending[0]
    item = pass_record.items[idx]
    ts = now or utcnow_iso()
    if pass_record.status == REVIEW_STATUS_PENDING:
        pass_record.status = REVIEW_STATUS_RUNNING
        pass_record.started_at = ts

    suggestion: CodeSuggestion | None = None
    try:
        suggestion = suggest_codes_for_span(
            projects_root=projects_root,
            project_id=pass_record.project_id,
            source_id=pass_record.source_id,
            anchor_start_word_id=item.anchor_start_word_id,
            anchor_end_word_id=item.anchor_end_word_id,
            query_text=item.text_preview,
            codes=codes,
            applications=applications,
            embed_fn=embed_fn,
            generate_fn=generate_fn,
            embedding_model=pass_record.embedding_model,
            generation_model=pass_record.generation_model,
            top_k=pass_record.top_k,
            max_candidates=pass_record.max_candidates,
            embedding_weight=pass_record.embedding_weight,
            min_score=pass_record.min_score,
            now=ts,
        )
    except ProjectValidationError as e:
        msg = str(e)[:MAX_ITEM_ERROR_LEN]
        item.error = msg or "ProjectValidationError"
    except Exception as e:  # noqa: BLE001  (engine boundary)
        msg = f"{type(e).__name__}: {e}"[:MAX_ITEM_ERROR_LEN]
        item.error = msg or type(e).__name__

    if suggestion is not None:
        save_suggestion(projects_root, suggestion)
        item.suggestion_id = suggestion.id

    item.validate()
    # Refresh per-pass progress / completion.
    if not pass_record.pending_indices():
        pass_record.status = REVIEW_STATUS_COMPLETED
        pass_record.completed_at = ts
    pass_record.modified_at = ts
    save_review_pass(projects_root, pass_record)
    return idx, suggestion


def run_review_pass(
    pass_record: ReviewPass,
    *,
    projects_root: Path,
    codes: Sequence[Code],
    applications: Sequence[Application],
    embed_fn: EmbedFn,
    generate_fn: GenerateFn | None = None,
    on_step: Callable[[int, CodeSuggestion | None], None] | None = None,
    max_steps: int | None = None,
    now: str | None = None,
) -> ReviewPass:
    """Drive a pass to completion (or up to ``max_steps``).

    Calls :func:`process_next_review_item` repeatedly. ``on_step`` is
    invoked after each item with ``(item_index, suggestion_or_None)``
    so a server can stream progress to a polling client.

    Returns the (mutated) ``pass_record``. Stops early if the pass
    enters a terminal state (e.g. via an external
    :func:`cancel_review_pass` call from another thread — though the
    current implementation is synchronous, the loop is defensive).
    """
    steps = 0
    while pass_record.status not in REVIEW_TERMINAL_STATUSES:
        if max_steps is not None and steps >= max_steps:
            break
        if not pass_record.pending_indices():
            # No pending items but also not yet flipped to completed:
            # let process_next_review_item finish the bookkeeping.
            break
        idx, suggestion = process_next_review_item(
            pass_record,
            projects_root=projects_root,
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


def cancel_review_pass(
    pass_record: ReviewPass,
    *,
    projects_root: Path,
    now: str | None = None,
) -> None:
    """Move a non-terminal pass to ``"cancelled"`` and persist.

    No-op if already cancelled. Raises
    :class:`ProjectValidationError` for other terminal states (you
    can't cancel a completed or failed pass — that would rewrite the
    audit trail).
    """
    if pass_record.status == REVIEW_STATUS_CANCELLED:
        return
    if pass_record.status in REVIEW_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot cancel pass {pass_record.id}: already in terminal "
            f"state {pass_record.status!r}"
        )
    ts = now or utcnow_iso()
    if not pass_record.started_at:
        pass_record.started_at = ts
    pass_record.status = REVIEW_STATUS_CANCELLED
    pass_record.completed_at = ts
    pass_record.modified_at = ts
    save_review_pass(projects_root, pass_record)


def mark_review_pass_failed(
    pass_record: ReviewPass,
    *,
    projects_root: Path,
    error_message: str,
    now: str | None = None,
) -> None:
    """Move a non-terminal pass to ``"failed"`` and persist with reason.

    Use this for *pass-level* failures (the project disappeared mid-
    pass; the embedder backend cannot be reached at all). Per-item
    errors land on the individual :class:`ReviewItem` and don't
    abort the pass.
    """
    if pass_record.status in REVIEW_TERMINAL_STATUSES:
        raise ProjectValidationError(
            f"Cannot fail pass {pass_record.id}: already in terminal "
            f"state {pass_record.status!r}"
        )
    msg = str(error_message or "")[:MAX_ERROR_MESSAGE_LEN]
    if not msg:
        raise ProjectValidationError(
            "error_message is required when failing a pass"
        )
    ts = now or utcnow_iso()
    if not pass_record.started_at:
        pass_record.started_at = ts
    pass_record.status = REVIEW_STATUS_FAILED
    pass_record.error_message = msg
    pass_record.completed_at = ts
    pass_record.modified_at = ts
    save_review_pass(projects_root, pass_record)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def review_passes_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's review passes.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / REVIEW_PASSES_DIRNAME


def review_pass_state_path(
    projects_root: Path, project_id: str, pass_id: str
) -> Path:
    if not REVIEW_PASS_ID_RE.match(pass_id):
        raise ProjectValidationError(
            f"Invalid review-pass id: {pass_id!r}"
        )
    return review_passes_dir(projects_root, project_id) / f"{pass_id}.json"


def save_review_pass(projects_root: Path, pass_record: ReviewPass) -> Path:
    """Persist a review-pass atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    pass_record.validate()
    parent = project_dir(projects_root, pass_record.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving review passes."
        )
    rd = review_passes_dir(projects_root, pass_record.project_id)
    rd.mkdir(parents=True, exist_ok=True)
    target = review_pass_state_path(
        projects_root, pass_record.project_id, pass_record.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(pass_record.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_review_pass(
    projects_root: Path, project_id: str, pass_id: str
) -> ReviewPass:
    """Load a review pass by id. Raises ``FileNotFoundError`` if missing."""
    p = review_pass_state_path(projects_root, project_id, pass_id)
    if not p.exists():
        raise FileNotFoundError(f"No review pass at {p}")
    return ReviewPass.from_dict(json.loads(p.read_text()))


def list_review_passes(
    projects_root: Path,
    project_id: str,
    *,
    source_id: str | None = None,
    status: str | None = None,
) -> list[ReviewPass]:
    """List review passes in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse — a single
    corrupt file does not break the listing. Sorted by ``created_at``
    ascending so the natural reading order is "the order in which
    passes were started".
    """
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    if status is not None and status not in REVIEW_STATUSES:
        raise ProjectValidationError(
            f"Invalid status filter: {status!r}"
        )
    rd = review_passes_dir(projects_root, project_id)
    if not rd.exists():
        return []
    out: list[ReviewPass] = []
    for f in sorted(rd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        rid = f.stem
        if not REVIEW_PASS_ID_RE.match(rid):
            continue
        try:
            rp = ReviewPass.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if source_id is not None and rp.source_id != source_id:
            continue
        if status is not None and rp.status != status:
            continue
        out.append(rp)
    out.sort(key=lambda r: (r.created_at, r.id))
    return out


def delete_review_pass(
    projects_root: Path, project_id: str, pass_id: str
) -> bool:
    """Remove a pass file. Returns False if it didn't exist.

    Production code should prefer keeping passes for the audit trail;
    deletion is exposed for tests and for the REFI-QDA import path.
    """
    p = review_pass_state_path(projects_root, project_id, pass_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
