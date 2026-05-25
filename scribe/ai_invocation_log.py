"""F9.6 — AI invocation log including rejected suggestions.

Per PLANNING.md F9.6:

  > AI invocation log including *rejected* suggestions (rejected
  > suggestions are evidence too).

F8.9 (:mod:`scribe.ai_provenance`) shipped the *schema* and
*persistence layer* for AI events: an :class:`AIEvent` dataclass,
``projects/<pid>/ai_events/<eid>.json`` files, and per-engine
extractors that lift a suggestion record into an
:class:`AIProvenance`. The F8.9 docstring noted that *wiring* each
engine's ``record_decision`` to also emit an :class:`AIEvent`,
together with the *report-side* aggregator, was a follow-on. F9.6 is
that follow-on: the canonical, unified read-side for the AI
invocation log, with first-class support for rejected suggestions.

What this module does
---------------------

1. **Emit decision events from per-engine records.** Each AI engine
   (F8.3 :class:`CodeSuggestion`, F8.4 :class:`NewCodeSuggestion`,
   F8.6 :class:`ReviewPass`, F8.7 :class:`SecondCoderPass`, F8.8
   :class:`MemoDraft`) keeps its own decision lifecycle
   (``pending → accepted | modified | rejected``) and writes to its
   own per-engine store. F9.6 provides
   :func:`record_decision_event_for_code_suggestion` and siblings
   that wrap the per-engine record into an :class:`AIEvent` of kind
   ``decision`` (or kind ``request`` for searches that have no
   decision lifecycle, like F8.5 :class:`QuoteSearch`) and persists
   it via :func:`scribe.ai_provenance.save_ai_event`. **Rejections
   land here verbatim** — that's the F9.6 commitment.

2. **Build the unified invocation log.** :func:`build_invocation_log`
   walks every per-engine store *and* the AI event log, and emits
   :class:`InvocationLogEntry` rows — one per suggestion / search /
   pass — with the decision pulled from the per-engine record
   (canonical) plus any cross-references to the AI event log
   (``ai_event_ids``). The result is a chronological, filterable
   audit trail that includes every rejected suggestion the project
   has ever seen.

3. **Counter helpers.** :func:`count_invocations` returns a small
   summary dict (``{"total": …, "accepted": …, "rejected": …, …}``)
   that's cheap enough to call on every page render of the project
   home / audit-log badge.

Boundaries
----------

* **Read-only with respect to the per-engine modules.** F9.6 does
  not modify the per-engine record_decision implementations; it
  reads the on-disk records and emits AIEvents on top.
* **No HTTP / FastAPI surface here.** Routes (``/api/projects/<id>/
  ai-invocations``) and CLI scripts are downstream of this module
  and are added when needed.
* **Pure-Python, deterministic, no engine imports beyond the
  per-engine record dataclasses.** Tests are unit-only.

On-disk shape
-------------

No new on-disk shape: the AI event log lives at
``projects/<pid>/ai_events/<eid>.json`` (F8.9), and the per-engine
suggestion stores already live at their own paths. F9.6 is purely a
read-side layer plus a thin write-side helper that goes through the
existing :func:`scribe.ai_provenance.save_ai_event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ai_provenance import (
    AI_DECISION_PENDING,
    AI_DECISIONS,
    AI_EVENT_KIND_DECISION,
    AI_EVENT_KIND_REQUEST,
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_MEMO_DRAFT,
    AI_FEATURE_NEW_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURE_SECOND_CODER,
    AI_FEATURE_TRANSCRIPT_REVIEW,
    AI_FEATURES,
    AIEvent,
    AIProvenance,
    list_ai_events,
    save_ai_event,
)
from .ai_second_coder import (
    SecondCoderPass,
    list_second_coder_passes,
)
from .code_suggestions import (
    CodeSuggestion,
    list_suggestions,
)
from .coders import CODER_ID_RE
from .memo_drafts import (
    MemoDraft,
    list_memo_drafts,
)
from .new_code_suggestions import (
    NewCodeSuggestion,
    list_new_code_suggestions,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
)
from .quote_similarity import (
    QuoteSearch,
    list_quote_searches,
)
from .transcript_review import (
    ReviewPass,
    list_review_passes,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# F8.5 QuoteSearches and F8.6/F8.7 *passes* don't carry a decision
# lifecycle of their own — the *items* they spawn (CodeSuggestions,
# review-pass items, second-coder per-application annotations) carry
# decisions individually. For those we emit a single ``request_only``
# entry into the unified invocation log so the audit-trail count of
# "AI invocations" matches the user's mental model of "how many times
# did I press a button that called a model".
DECISION_REQUEST_ONLY = "request_only"

# Closed set used by InvocationLogEntry.decision. Superset of
# AI_DECISIONS (which is per-suggestion) plus ``request_only`` for
# searches and passes.
INVOCATION_DECISIONS: tuple[str, ...] = AI_DECISIONS + (DECISION_REQUEST_ONLY,)


# Cardinality / size caps on the read-side entry.
MAX_SUMMARY_LEN = 240
MAX_RELATED_IDS = 8


# --------------------------------------------------------------------------- #
# InvocationLogEntry
# --------------------------------------------------------------------------- #


@dataclass
class InvocationLogEntry:
    """One row in the unified AI invocation log (F9.6).

    Each row corresponds to *one AI invocation* — one click of "suggest
    codes", one search, one whole-transcript review pass, one
    second-coder pass, one memo-draft request. The row carries the
    decision (or ``request_only`` for invocations that have no
    decision lifecycle of their own), enough metadata to render an
    audit-trail line, and back-references to:

    * the per-engine record (``suggestion_id``)
    * any :class:`AIEvent` rows that reference it (``ai_event_ids``).

    Fields
    ------
    feature
        One of :data:`AI_FEATURES`. Says which engine produced the
        invocation.
    suggestion_id
        12-char hex id of the per-engine record (CodeSuggestion,
        NewCodeSuggestion, MemoDraft, ReviewPass, SecondCoderPass,
        QuoteSearch). Always set — that's the natural key.
    project_id
        12-char hex project id.
    created_at
        ISO-8601 UTC; when the suggestion / search / pass was created.
    decision
        One of :data:`INVOCATION_DECISIONS`. ``pending`` /
        ``accepted`` / ``modified`` / ``rejected`` for invocations
        with a decision lifecycle; ``request_only`` for searches and
        passes.
    decided_at
        ISO-8601 UTC; when the decision was recorded. Empty for
        ``pending`` / ``request_only``.
    decided_by_coder_id
        12-char hex coder id; the human who pressed accept / modify /
        reject. Empty for ``pending`` / ``request_only``.
    requested_by_coder_id
        12-char hex coder id of the human who *triggered* the
        invocation. Most engines don't capture this directly today;
        for second-coder it's ``human_coder_id``. Empty when unknown.
    generation_model
        The chat / completion model id used. Empty if unset.
    embedding_model
        The embedding model id used. Empty if unset.
    rejection_reason
        Free-form short text supplied at rejection time. Only
        non-empty for ``decision == "rejected"``.
    summary
        Short human-readable summary (the *what*: span text, code
        name, search query). Bounded at :data:`MAX_SUMMARY_LEN`.
    related_entity_ids
        Up to :data:`MAX_RELATED_IDS` 12-char hex ids that contextualise
        the invocation: e.g. ``[source_id]`` for a CodeSuggestion,
        ``[code_id]`` for a memo draft, ``[application_id]`` for a
        quote search by application. Order is feature-specific.
    ai_event_ids
        12-char hex ids of any :class:`AIEvent` rows that reference
        this invocation (i.e. whose ``provenance.suggestion_id``
        equals this row's ``suggestion_id``). Empty if the engine
        never emitted an explicit AIEvent for this invocation.
    """

    feature: str
    suggestion_id: str
    project_id: str
    created_at: str
    decision: str = DECISION_REQUEST_ONLY
    decided_at: str = ""
    decided_by_coder_id: str = ""
    requested_by_coder_id: str = ""
    generation_model: str = ""
    embedding_model: str = ""
    rejection_reason: str = ""
    summary: str = ""
    related_entity_ids: list[str] = field(default_factory=list)
    ai_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "suggestion_id": self.suggestion_id,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by_coder_id": self.decided_by_coder_id,
            "requested_by_coder_id": self.requested_by_coder_id,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "rejection_reason": self.rejection_reason,
            "summary": self.summary,
            "related_entity_ids": list(self.related_entity_ids),
            "ai_event_ids": list(self.ai_event_ids),
        }


# --------------------------------------------------------------------------- #
# Per-engine entry extractors
# --------------------------------------------------------------------------- #


def _truncate_summary(s: str) -> str:
    """Trim a summary string to :data:`MAX_SUMMARY_LEN` with an ellipsis."""
    text = (s or "").strip()
    if len(text) <= MAX_SUMMARY_LEN:
        return text
    return text[: MAX_SUMMARY_LEN - 1].rstrip() + "…"


def _entry_from_code_suggestion(
    s: CodeSuggestion, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.3 CodeSuggestion onto an InvocationLogEntry."""
    return InvocationLogEntry(
        feature=AI_FEATURE_CODE_SUGGESTION,
        suggestion_id=s.id,
        project_id=s.project_id,
        created_at=s.created_at,
        decision=str(s.decision or AI_DECISION_PENDING),
        decided_at=str(s.decided_at or ""),
        decided_by_coder_id=str(s.decided_by_coder_id or ""),
        requested_by_coder_id="",  # F8.3 doesn't capture the requester
        generation_model=str(s.generation_model or ""),
        embedding_model=str(s.embedding_model or ""),
        rejection_reason=str(s.rejection_reason or ""),
        summary=_truncate_summary(s.query_text),
        related_entity_ids=[s.source_id] if s.source_id else [],
        ai_event_ids=list(ai_event_ids),
    )


def _entry_from_new_code_suggestion(
    s: NewCodeSuggestion, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.4 NewCodeSuggestion onto an InvocationLogEntry."""
    return InvocationLogEntry(
        feature=AI_FEATURE_NEW_CODE_SUGGESTION,
        suggestion_id=s.id,
        project_id=s.project_id,
        created_at=s.created_at,
        decision=str(s.decision or AI_DECISION_PENDING),
        decided_at=str(s.decided_at or ""),
        decided_by_coder_id=str(s.decided_by_coder_id or ""),
        requested_by_coder_id="",
        generation_model=str(s.generation_model or ""),
        embedding_model=str(s.embedding_model or ""),
        rejection_reason=str(s.rejection_reason or ""),
        summary=_truncate_summary(s.query_text),
        related_entity_ids=[s.source_id] if s.source_id else [],
        ai_event_ids=list(ai_event_ids),
    )


def _entry_from_memo_draft(
    d: MemoDraft, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.8 MemoDraft onto an InvocationLogEntry."""
    summary_bits = []
    if d.title:
        summary_bits.append(d.title)
    if d.memo_type:
        summary_bits.append(f"({d.memo_type})")
    return InvocationLogEntry(
        feature=AI_FEATURE_MEMO_DRAFT,
        suggestion_id=d.id,
        project_id=d.project_id,
        created_at=d.created_at,
        decision=str(d.decision or AI_DECISION_PENDING),
        decided_at=str(d.decided_at or ""),
        decided_by_coder_id=str(d.decided_by_coder_id or ""),
        requested_by_coder_id="",
        generation_model=str(d.generation_model or ""),
        embedding_model="",
        rejection_reason=str(d.rejection_reason or ""),
        summary=_truncate_summary(" ".join(summary_bits) or d.body),
        related_entity_ids=[d.code_id] if d.code_id else [],
        ai_event_ids=list(ai_event_ids),
    )


def _entry_from_review_pass(
    p: ReviewPass, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.6 ReviewPass onto an InvocationLogEntry.

    Review passes have no decision lifecycle of their own — they spawn
    individual :class:`CodeSuggestion` rows that have decisions. We
    emit a ``request_only`` entry per pass so the invocation count
    matches the user's mental model.
    """
    return InvocationLogEntry(
        feature=AI_FEATURE_TRANSCRIPT_REVIEW,
        suggestion_id=p.id,
        project_id=p.project_id,
        created_at=p.created_at,
        decision=DECISION_REQUEST_ONLY,
        decided_at="",
        decided_by_coder_id="",
        requested_by_coder_id="",
        generation_model=str(p.generation_model or ""),
        embedding_model=str(getattr(p, "embedding_model", "") or ""),
        rejection_reason="",
        summary=_truncate_summary(
            f"Whole-transcript review pass (status={p.status})"
        ),
        related_entity_ids=[p.source_id] if p.source_id else [],
        ai_event_ids=list(ai_event_ids),
    )


def _entry_from_second_coder_pass(
    p: SecondCoderPass, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.7 SecondCoderPass onto an InvocationLogEntry."""
    related: list[str] = []
    if p.source_id:
        related.append(p.source_id)
    if p.human_coder_id and p.human_coder_id not in related:
        related.append(p.human_coder_id)
    return InvocationLogEntry(
        feature=AI_FEATURE_SECOND_CODER,
        suggestion_id=p.id,
        project_id=p.project_id,
        created_at=p.created_at,
        decision=DECISION_REQUEST_ONLY,
        decided_at="",
        decided_by_coder_id="",
        requested_by_coder_id=str(p.human_coder_id or ""),
        generation_model=str(p.generation_model or ""),
        embedding_model=str(p.embedding_model or ""),
        rejection_reason="",
        summary=_truncate_summary(
            f"AI second-coder pass (status={p.status})"
        ),
        related_entity_ids=related,
        ai_event_ids=list(ai_event_ids),
    )


def _entry_from_quote_search(
    q: QuoteSearch, *, ai_event_ids: list[str]
) -> InvocationLogEntry:
    """Lift a F8.5 QuoteSearch onto an InvocationLogEntry.

    Quote searches are pure search invocations with no decision
    lifecycle — they're recorded as ``request_only``.
    """
    related: list[str] = []
    if q.query_application_id:
        related.append(q.query_application_id)
    if q.query_source_id:
        related.append(q.query_source_id)
    summary = q.query_text or f"quote search ({q.query_kind})"
    return InvocationLogEntry(
        feature=AI_FEATURE_QUOTE_SIMILARITY,
        suggestion_id=q.id,
        project_id=q.project_id,
        created_at=q.created_at,
        decision=DECISION_REQUEST_ONLY,
        decided_at="",
        decided_by_coder_id="",
        requested_by_coder_id="",
        generation_model="",
        embedding_model=str(q.embedding_model or ""),
        rejection_reason="",
        summary=_truncate_summary(summary),
        related_entity_ids=related,
        ai_event_ids=list(ai_event_ids),
    )


# --------------------------------------------------------------------------- #
# Aggregator: build_invocation_log
# --------------------------------------------------------------------------- #


def _index_ai_events_by_suggestion(
    events: list[AIEvent],
) -> dict[str, list[str]]:
    """Group AI event ids by the suggestion id they reference.

    ``events`` is the full list returned by
    :func:`scribe.ai_provenance.list_ai_events`. We bucket them by
    ``provenance.suggestion_id``; events without a suggestion id (rare,
    but allowed by F8.9) are skipped here — they show up via the
    ``loose_ai_events`` list returned alongside.
    """
    buckets: dict[str, list[str]] = {}
    for ev in events:
        sid = ev.provenance.suggestion_id
        if not sid:
            continue
        buckets.setdefault(sid, []).append(ev.id)
    return buckets


def build_invocation_log(
    projects_root: Path,
    project_id: str,
    *,
    feature: str | None = None,
    decision: str | None = None,
    actor_coder_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[InvocationLogEntry]:
    """Return every AI invocation in a project, filterable, chronological.

    Walks each per-engine store and the AI event log, producing one
    :class:`InvocationLogEntry` per per-engine record. Cross-references
    AI events by ``provenance.suggestion_id``. Sorted by ``created_at``
    ascending then by ``suggestion_id`` for stability — natural reading
    order is the order the operations were emitted.

    Filters
    -------

    All filters AND-combine.

    * ``feature`` — restrict to one of :data:`AI_FEATURES`.
    * ``decision`` — restrict to one of :data:`INVOCATION_DECISIONS`
      (i.e. ``pending`` / ``accepted`` / ``modified`` / ``rejected`` /
      ``request_only``). To list *only* the rejections — the F9.6
      headline use-case — pass ``decision="rejected"``.
    * ``actor_coder_id`` — restrict to entries decided by (or
      requested by, for ``request_only``) the given coder.
    * ``since`` / ``until`` — inclusive ISO-8601 bounds compared
      lexically (works for the Z-suffixed UTC timestamps Scribe uses).
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    if feature is not None and feature not in AI_FEATURES:
        raise ProjectValidationError(f"Invalid feature filter: {feature!r}")
    if decision is not None and decision not in INVOCATION_DECISIONS:
        raise ProjectValidationError(
            f"Invalid decision filter: {decision!r}"
        )
    if actor_coder_id is not None and actor_coder_id and not CODER_ID_RE.match(
        actor_coder_id
    ):
        raise ProjectValidationError(
            f"Invalid actor_coder_id filter: {actor_coder_id!r}"
        )

    # Pre-fetch AI events once and bucket by suggestion id.
    ai_events = list_ai_events(projects_root, project_id)
    by_sid = _index_ai_events_by_suggestion(ai_events)

    entries: list[InvocationLogEntry] = []

    # F8.3 CodeSuggestion (and the per-item suggestions spawned by F8.6
    # ReviewPass — they live in the same store).
    for s in list_suggestions(projects_root, project_id):
        entries.append(
            _entry_from_code_suggestion(s, ai_event_ids=by_sid.get(s.id, []))
        )

    # F8.4 NewCodeSuggestion.
    for ncs in list_new_code_suggestions(projects_root, project_id):
        entries.append(
            _entry_from_new_code_suggestion(
                ncs, ai_event_ids=by_sid.get(ncs.id, [])
            )
        )

    # F8.8 MemoDraft.
    for d in list_memo_drafts(projects_root, project_id):
        entries.append(
            _entry_from_memo_draft(d, ai_event_ids=by_sid.get(d.id, []))
        )

    # F8.6 ReviewPass — invocation-level row (request_only).
    for p in list_review_passes(projects_root, project_id):
        entries.append(
            _entry_from_review_pass(p, ai_event_ids=by_sid.get(p.id, []))
        )

    # F8.7 SecondCoderPass.
    for sp in list_second_coder_passes(projects_root, project_id):
        entries.append(
            _entry_from_second_coder_pass(
                sp, ai_event_ids=by_sid.get(sp.id, [])
            )
        )

    # F8.5 QuoteSearch.
    for qs in list_quote_searches(projects_root, project_id):
        entries.append(
            _entry_from_quote_search(qs, ai_event_ids=by_sid.get(qs.id, []))
        )

    # Apply filters.
    out: list[InvocationLogEntry] = []
    for e in entries:
        if feature is not None and e.feature != feature:
            continue
        if decision is not None and e.decision != decision:
            continue
        if actor_coder_id is not None and actor_coder_id:
            # Match against decided_by *or* requested_by — the
            # invocation log is "anything this coder touched".
            if (
                e.decided_by_coder_id != actor_coder_id
                and e.requested_by_coder_id != actor_coder_id
            ):
                continue
        if since is not None and e.created_at < since:
            continue
        if until is not None and e.created_at > until:
            continue
        out.append(e)

    out.sort(key=lambda e: (e.created_at, e.suggestion_id))
    return out


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #


def count_invocations(
    projects_root: Path,
    project_id: str,
    *,
    feature: str | None = None,
) -> dict[str, int]:
    """Return a small counter dict over the invocation log.

    Keys: ``"total"`` plus one per :data:`INVOCATION_DECISIONS` entry.
    All counts are integers ≥ 0. Optional ``feature`` filter restricts
    to one engine (e.g. count only F8.3 code suggestions).

    Cheap-ish — walks the same stores as :func:`build_invocation_log`
    once. Suitable for a UI badge but not a hot loop.
    """
    if feature is not None and feature not in AI_FEATURES:
        raise ProjectValidationError(f"Invalid feature filter: {feature!r}")
    log = build_invocation_log(projects_root, project_id, feature=feature)
    counts = {"total": len(log)}
    for d in INVOCATION_DECISIONS:
        counts[d] = 0
    for e in log:
        counts[e.decision] = counts.get(e.decision, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Write-side helpers — emit AIEvents from per-engine records.
# --------------------------------------------------------------------------- #


def _payload_for_code_suggestion(s: CodeSuggestion) -> dict[str, Any]:
    """Build a small AIEvent payload from a CodeSuggestion."""
    payload: dict[str, Any] = {
        "source_id": s.source_id,
        "anchor_start_word_id": s.anchor_start_word_id,
        "anchor_end_word_id": s.anchor_end_word_id,
    }
    if s.accepted_code_id:
        payload["accepted_code_id"] = s.accepted_code_id
    if s.accepted_application_id:
        payload["accepted_application_id"] = s.accepted_application_id
    if s.rejection_reason:
        payload["rejection_reason"] = s.rejection_reason[:1000]
    return payload


def _payload_for_new_code_suggestion(s: NewCodeSuggestion) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": s.source_id,
        "anchor_start_word_id": s.anchor_start_word_id,
        "anchor_end_word_id": s.anchor_end_word_id,
    }
    if s.accepted_proposal_index is not None:
        payload["accepted_proposal_index"] = int(s.accepted_proposal_index)
    if s.created_code_id:
        payload["created_code_id"] = s.created_code_id
    if s.rejection_reason:
        payload["rejection_reason"] = s.rejection_reason[:1000]
    return payload


def _payload_for_memo_draft(d: MemoDraft) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code_id": d.code_id,
        "memo_type": d.memo_type,
    }
    if d.accepted_memo_id:
        payload["accepted_memo_id"] = d.accepted_memo_id
    if d.rejection_reason:
        payload["rejection_reason"] = d.rejection_reason[:1000]
    return payload


def record_decision_event_for_code_suggestion(
    projects_root: Path,
    suggestion: CodeSuggestion,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``decision`` from a CodeSuggestion.

    Use after :func:`scribe.code_suggestions.record_decision` has moved
    a suggestion into a terminal state (``accepted`` / ``modified`` /
    ``rejected``). Rejections produce an event with
    ``provenance.decision == "rejected"`` and the ``rejection_reason``
    in the payload — that's the F9.6 commitment.

    Returns the persisted :class:`AIEvent`. Raises
    :class:`ProjectValidationError` if the suggestion is still pending
    (no decision to record).
    """
    if suggestion.decision == AI_DECISION_PENDING:
        raise ProjectValidationError(
            "Suggestion has no terminal decision yet; "
            "call record_decision first."
        )
    actor = actor_coder_id or suggestion.decided_by_coder_id or ""
    if actor and not CODER_ID_RE.match(actor):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_CODE_SUGGESTION,
        backend=backend,
        generation_model=str(suggestion.generation_model or ""),
        embedding_model=str(suggestion.embedding_model or ""),
        suggestion_id=suggestion.id,
        decision=str(suggestion.decision),
        decided_by_coder_id=str(suggestion.decided_by_coder_id or ""),
        decided_at=str(suggestion.decided_at or ""),
    )
    payload = _payload_for_code_suggestion(suggestion)
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=suggestion.project_id,
        feature=AI_FEATURE_CODE_SUGGESTION,
        kind=AI_EVENT_KIND_DECISION,
        actor_coder_id=actor,
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


def record_decision_event_for_new_code_suggestion(
    projects_root: Path,
    suggestion: NewCodeSuggestion,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``decision`` from a NewCodeSuggestion."""
    if suggestion.decision == AI_DECISION_PENDING:
        raise ProjectValidationError(
            "Suggestion has no terminal decision yet; "
            "call record_new_code_decision first."
        )
    actor = actor_coder_id or suggestion.decided_by_coder_id or ""
    if actor and not CODER_ID_RE.match(actor):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_NEW_CODE_SUGGESTION,
        backend=backend,
        generation_model=str(suggestion.generation_model or ""),
        embedding_model=str(suggestion.embedding_model or ""),
        suggestion_id=suggestion.id,
        decision=str(suggestion.decision),
        decided_by_coder_id=str(suggestion.decided_by_coder_id or ""),
        decided_at=str(suggestion.decided_at or ""),
    )
    payload = _payload_for_new_code_suggestion(suggestion)
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=suggestion.project_id,
        feature=AI_FEATURE_NEW_CODE_SUGGESTION,
        kind=AI_EVENT_KIND_DECISION,
        actor_coder_id=actor,
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


def record_decision_event_for_memo_draft(
    projects_root: Path,
    draft: MemoDraft,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``decision`` from a MemoDraft."""
    if draft.decision == AI_DECISION_PENDING:
        raise ProjectValidationError(
            "Draft has no terminal decision yet; "
            "call record_memo_draft_decision first."
        )
    actor = actor_coder_id or draft.decided_by_coder_id or ""
    if actor and not CODER_ID_RE.match(actor):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_MEMO_DRAFT,
        backend=backend,
        generation_model=str(draft.generation_model or ""),
        suggestion_id=draft.id,
        decision=str(draft.decision),
        decided_by_coder_id=str(draft.decided_by_coder_id or ""),
        decided_at=str(draft.decided_at or ""),
    )
    payload = _payload_for_memo_draft(draft)
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=draft.project_id,
        feature=AI_FEATURE_MEMO_DRAFT,
        kind=AI_EVENT_KIND_DECISION,
        actor_coder_id=actor,
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


def record_request_event_for_quote_search(
    projects_root: Path,
    search: QuoteSearch,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``request`` from a QuoteSearch.

    Quote searches have no decision lifecycle — they're search
    invocations. We log them as request events so the audit trail
    captures the "researcher pressed the search button at time X with
    embedding model Y" line.
    """
    if actor_coder_id and not CODER_ID_RE.match(actor_coder_id):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor_coder_id!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_QUOTE_SIMILARITY,
        backend=backend,
        embedding_model=str(search.embedding_model or ""),
        suggestion_id=search.id,
        decision=AI_DECISION_PENDING,
    )
    payload: dict[str, Any] = {
        "query_kind": search.query_kind,
        "top_k": int(search.top_k),
        "matches_count": len(search.matches),
    }
    if search.query_application_id:
        payload["query_application_id"] = search.query_application_id
    if search.query_source_id:
        payload["query_source_id"] = search.query_source_id
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=search.project_id,
        feature=AI_FEATURE_QUOTE_SIMILARITY,
        kind=AI_EVENT_KIND_REQUEST,
        actor_coder_id=str(actor_coder_id or ""),
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


def record_request_event_for_review_pass(
    projects_root: Path,
    pass_record: ReviewPass,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``request`` from a ReviewPass."""
    if actor_coder_id and not CODER_ID_RE.match(actor_coder_id):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor_coder_id!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_TRANSCRIPT_REVIEW,
        backend=backend,
        generation_model=str(pass_record.generation_model or ""),
        embedding_model=str(getattr(pass_record, "embedding_model", "") or ""),
        suggestion_id=pass_record.id,
        decision=AI_DECISION_PENDING,
    )
    payload: dict[str, Any] = {
        "source_id": pass_record.source_id,
        "status": pass_record.status,
    }
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=pass_record.project_id,
        feature=AI_FEATURE_TRANSCRIPT_REVIEW,
        kind=AI_EVENT_KIND_REQUEST,
        actor_coder_id=str(actor_coder_id or ""),
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


def record_request_event_for_second_coder_pass(
    projects_root: Path,
    pass_record: SecondCoderPass,
    *,
    actor_coder_id: str = "",
    backend: str = "",
    extra_payload: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist an :class:`AIEvent` of kind ``request`` from a SecondCoderPass."""
    actor = actor_coder_id or pass_record.human_coder_id or ""
    if actor and not CODER_ID_RE.match(actor):
        raise ProjectValidationError(
            f"actor_coder_id must be 12-char hex; got {actor!r}"
        )
    prov = AIProvenance.new(
        feature=AI_FEATURE_SECOND_CODER,
        backend=backend,
        generation_model=str(pass_record.generation_model or ""),
        embedding_model=str(pass_record.embedding_model or ""),
        suggestion_id=pass_record.id,
        decision=AI_DECISION_PENDING,
    )
    payload: dict[str, Any] = {
        "source_id": pass_record.source_id,
        "human_coder_id": pass_record.human_coder_id,
        "status": pass_record.status,
    }
    if extra_payload:
        payload.update(extra_payload)
    ev = AIEvent.new(
        project_id=pass_record.project_id,
        feature=AI_FEATURE_SECOND_CODER,
        kind=AI_EVENT_KIND_REQUEST,
        actor_coder_id=actor,
        provenance=prov,
        payload=payload,
        event_id=event_id,
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


# --------------------------------------------------------------------------- #
# Re-exports for the public API
# --------------------------------------------------------------------------- #


__all__ = [
    "DECISION_REQUEST_ONLY",
    "INVOCATION_DECISIONS",
    "InvocationLogEntry",
    "build_invocation_log",
    "count_invocations",
    "record_decision_event_for_code_suggestion",
    "record_decision_event_for_memo_draft",
    "record_decision_event_for_new_code_suggestion",
    "record_request_event_for_quote_search",
    "record_request_event_for_review_pass",
    "record_request_event_for_second_coder_pass",
]
