"""Tests for scribe.ai_invocation_log (F9.6).

F9.6 is the unified AI invocation log — including rejected suggestions,
which are evidence even when they're not applied. The module wraps the
F8.9 :class:`AIEvent` persistence layer with:

  * write-side helpers that emit decision/request events from per-engine
    records (CodeSuggestion, NewCodeSuggestion, MemoDraft, ReviewPass,
    SecondCoderPass, QuoteSearch);
  * a read-side aggregator (:func:`build_invocation_log`) that walks
    every per-engine store *and* the AI event log to produce a flat,
    chronological :class:`InvocationLogEntry` view, filterable by
    feature / decision / coder / time;
  * a small counter (:func:`count_invocations`).

These tests cover the dataclass round-trip, the per-engine extractors,
the write-side event emitters (including the rejection round-trip —
the F9.6 headline), the build_invocation_log aggregator across engines
and filters, and the counter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.ai_invocation_log import (
    DECISION_REQUEST_ONLY,
    INVOCATION_DECISIONS,
    InvocationLogEntry,
    build_invocation_log,
    count_invocations,
    record_decision_event_for_code_suggestion,
    record_decision_event_for_memo_draft,
    record_decision_event_for_new_code_suggestion,
    record_request_event_for_quote_search,
    record_request_event_for_review_pass,
    record_request_event_for_second_coder_pass,
)
from scribe.ai_provenance import (
    AI_DECISION_ACCEPTED,
    AI_DECISION_PENDING,
    AI_DECISION_REJECTED,
    AI_EVENT_KIND_DECISION,
    AI_EVENT_KIND_REQUEST,
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_MEMO_DRAFT,
    AI_FEATURE_NEW_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURE_SECOND_CODER,
    AI_FEATURE_TRANSCRIPT_REVIEW,
    list_ai_events,
)
from scribe.ai_second_coder import (
    SECOND_CODER_STATUS_PENDING,
    SecondCoderPass,
    save_second_coder_pass,
)
from scribe.code_suggestions import (
    CodeSuggestion,
    record_decision,
    save_suggestion,
)
from scribe.memo_drafts import (
    MemoDraft,
    record_memo_draft_decision,
    save_memo_draft,
)
from scribe.new_code_suggestions import (
    NewCodeSuggestion,
    record_new_code_decision,
    save_new_code_suggestion,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.quote_similarity import (
    QUERY_KIND_TEXT,
    QuoteSearch,
    save_quote_search,
)
from scribe.transcript_review import (
    REVIEW_STATUS_PENDING,
    ReviewPass,
    save_review_pass,
)


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_CODE_2 = "b" * 12
_HEX_SOURCE = "c" * 12
_HEX_CODER = "d" * 12
_HEX_CODER_2 = "e" * 12


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_code_suggestion(
    project_id: str,
    *,
    suggestion_id: str | None = None,
    source_id: str = _HEX_SOURCE,
    query_text: str = "I keep getting stuck on this part",
    generation_model: str = "llama3.2:3b",
    embedding_model: str = "bge-m3",
    now: str | None = None,
) -> CodeSuggestion:
    return CodeSuggestion.new(
        project_id=project_id,
        source_id=source_id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w5",
        query_text=query_text,
        embedding_model=embedding_model,
        generation_model=generation_model,
        suggestion_id=suggestion_id,
        now=now,
    )


def _make_new_code_suggestion(
    project_id: str,
    *,
    suggestion_id: str | None = None,
    source_id: str = _HEX_SOURCE,
    now: str | None = None,
) -> NewCodeSuggestion:
    return NewCodeSuggestion.new(
        project_id=project_id,
        source_id=source_id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w5",
        query_text="something going on with autonomy",
        embedding_model="bge-m3",
        generation_model="llama3.2:3b",
        suggestion_id=suggestion_id,
        now=now,
    )


def _make_memo_draft(
    project_id: str,
    *,
    draft_id: str | None = None,
    code_id: str = _HEX_CODE,
    now: str | None = None,
) -> MemoDraft:
    return MemoDraft.new(
        project_id=project_id,
        code_id=code_id,
        title="Negotiating autonomy",
        body="A first cut at the analytical memo for this code.",
        generation_model="llama3.2:3b",
        draft_id=draft_id,
        now=now,
    )


def _make_review_pass(
    project_id: str,
    *,
    pass_id: str | None = None,
    source_id: str = _HEX_SOURCE,
    now: str | None = None,
) -> ReviewPass:
    return ReviewPass.new(
        project_id=project_id,
        source_id=source_id,
        generation_model="llama3.2:3b",
        embedding_model="bge-m3",
        pass_id=pass_id,
        now=now,
    )


def _make_second_coder_pass(
    project_id: str,
    *,
    pass_id: str | None = None,
    source_id: str = _HEX_SOURCE,
    human_coder_id: str = _HEX_CODER,
    now: str | None = None,
) -> SecondCoderPass:
    return SecondCoderPass.new(
        project_id=project_id,
        source_id=source_id,
        human_coder_id=human_coder_id,
        generation_model="llama3.2:3b",
        embedding_model="bge-m3",
        pass_id=pass_id,
        now=now,
    )


def _make_quote_search(
    project_id: str,
    *,
    search_id: str | None = None,
    query_text: str = "feeling overwhelmed",
    now: str | None = None,
) -> QuoteSearch:
    return QuoteSearch.new(
        project_id=project_id,
        query_kind=QUERY_KIND_TEXT,
        query_text=query_text,
        embedding_model="bge-m3",
        search_id=search_id,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_invocation_decisions_superset_of_ai_decisions(self) -> None:
        # Pending/accepted/modified/rejected must all be reachable as
        # InvocationLogEntry.decision values.
        for d in ("pending", "accepted", "modified", "rejected"):
            assert d in INVOCATION_DECISIONS

    def test_request_only_is_distinct(self) -> None:
        # ``request_only`` is the F9.6 marker for invocations that have
        # no decision lifecycle of their own (searches, passes).
        assert DECISION_REQUEST_ONLY in INVOCATION_DECISIONS
        assert DECISION_REQUEST_ONLY not in (
            "pending",
            "accepted",
            "modified",
            "rejected",
        )


# --------------------------------------------------------------------------- #
# InvocationLogEntry
# --------------------------------------------------------------------------- #


class TestInvocationLogEntry:
    def test_round_trip_to_dict(self) -> None:
        e = InvocationLogEntry(
            feature=AI_FEATURE_CODE_SUGGESTION,
            suggestion_id="abcdef012345",
            project_id=_HEX_PROJECT,
            created_at="2026-05-01T12:00:00Z",
            decision=AI_DECISION_REJECTED,
            decided_at="2026-05-01T12:05:00Z",
            decided_by_coder_id=_HEX_CODER,
            generation_model="llama3.2:3b",
            embedding_model="bge-m3",
            rejection_reason="off-topic",
            summary="I keep getting stuck",
            related_entity_ids=[_HEX_SOURCE],
            ai_event_ids=["111111111111"],
        )
        d = e.to_dict()
        assert d["feature"] == AI_FEATURE_CODE_SUGGESTION
        assert d["suggestion_id"] == "abcdef012345"
        assert d["decision"] == "rejected"
        assert d["rejection_reason"] == "off-topic"
        assert d["related_entity_ids"] == [_HEX_SOURCE]
        assert d["ai_event_ids"] == ["111111111111"]


# --------------------------------------------------------------------------- #
# Write-side helpers
# --------------------------------------------------------------------------- #


class TestRecordDecisionEventForCodeSuggestion:
    def test_rejected_suggestion_round_trips(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        # Record a rejection — F9.6 must capture it.
        record_decision(
            s,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="None of the candidates fit; speaker was joking.",
        )
        save_suggestion(tmp_path, s)  # update the file
        ev = record_decision_event_for_code_suggestion(
            tmp_path, s, backend="ollama"
        )
        assert ev.kind == AI_EVENT_KIND_DECISION
        assert ev.feature == AI_FEATURE_CODE_SUGGESTION
        assert ev.provenance.decision == AI_DECISION_REJECTED
        assert ev.provenance.suggestion_id == s.id
        assert ev.provenance.decided_by_coder_id == _HEX_CODER
        assert ev.provenance.generation_model == "llama3.2:3b"
        assert ev.provenance.embedding_model == "bge-m3"
        assert ev.provenance.backend == "ollama"
        # Payload must capture the rejection reason as evidence.
        assert "rejection_reason" in ev.payload
        assert "joking" in ev.payload["rejection_reason"]
        # The actor on the event defaults to the decider.
        assert ev.actor_coder_id == _HEX_CODER
        # Event is on disk and round-trips via list.
        evs = list_ai_events(tmp_path, proj.id, kind=AI_EVENT_KIND_DECISION)
        assert any(x.id == ev.id for x in evs)

    def test_accepted_suggestion_records_event(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        record_decision(
            s,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
        )
        save_suggestion(tmp_path, s)
        ev = record_decision_event_for_code_suggestion(tmp_path, s)
        assert ev.provenance.decision == AI_DECISION_ACCEPTED
        assert ev.payload["accepted_code_id"] == _HEX_CODE

    def test_pending_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        with pytest.raises(ProjectValidationError):
            record_decision_event_for_code_suggestion(tmp_path, s)

    def test_explicit_actor_overrides_decider(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        record_decision(
            s,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
        )
        save_suggestion(tmp_path, s)
        ev = record_decision_event_for_code_suggestion(
            tmp_path, s, actor_coder_id=_HEX_CODER_2
        )
        assert ev.actor_coder_id == _HEX_CODER_2

    def test_bad_actor_id_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        record_decision(
            s,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
        )
        save_suggestion(tmp_path, s)
        with pytest.raises(ProjectValidationError):
            record_decision_event_for_code_suggestion(
                tmp_path, s, actor_coder_id="not-hex"
            )

    def test_extra_payload_merged(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(proj.id)
        save_suggestion(tmp_path, s)
        record_decision(
            s,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
        )
        save_suggestion(tmp_path, s)
        ev = record_decision_event_for_code_suggestion(
            tmp_path, s, extra_payload={"trigger": "ui"}
        )
        assert ev.payload["trigger"] == "ui"


class TestRecordDecisionEventForNewCodeSuggestion:
    def test_rejected_round_trips(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_new_code_suggestion(proj.id)
        # Add a single proposal so accept/modify decisions can target it.
        s.proposals = []  # rejected path doesn't need proposals
        save_new_code_suggestion(tmp_path, s)
        record_new_code_decision(
            s,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="not gerund-ish enough",
        )
        save_new_code_suggestion(tmp_path, s)
        ev = record_decision_event_for_new_code_suggestion(tmp_path, s)
        assert ev.kind == AI_EVENT_KIND_DECISION
        assert ev.feature == AI_FEATURE_NEW_CODE_SUGGESTION
        assert ev.provenance.decision == AI_DECISION_REJECTED
        assert ev.payload["rejection_reason"].startswith("not gerund")
        assert ev.actor_coder_id == _HEX_CODER

    def test_pending_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_new_code_suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        with pytest.raises(ProjectValidationError):
            record_decision_event_for_new_code_suggestion(tmp_path, s)


class TestRecordDecisionEventForMemoDraft:
    def test_rejected_round_trips(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        d = _make_memo_draft(proj.id)
        save_memo_draft(tmp_path, d)
        record_memo_draft_decision(
            d,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="memo missed the point",
        )
        save_memo_draft(tmp_path, d)
        ev = record_decision_event_for_memo_draft(tmp_path, d)
        assert ev.feature == AI_FEATURE_MEMO_DRAFT
        assert ev.kind == AI_EVENT_KIND_DECISION
        assert ev.provenance.decision == AI_DECISION_REJECTED
        assert ev.payload["code_id"] == _HEX_CODE
        assert "memo missed" in ev.payload["rejection_reason"]

    def test_pending_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        d = _make_memo_draft(proj.id)
        save_memo_draft(tmp_path, d)
        with pytest.raises(ProjectValidationError):
            record_decision_event_for_memo_draft(tmp_path, d)


class TestRecordRequestEventForQuoteSearch:
    def test_basic_request(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        q = _make_quote_search(proj.id)
        save_quote_search(tmp_path, q)
        ev = record_request_event_for_quote_search(
            tmp_path, q, actor_coder_id=_HEX_CODER, backend="ollama"
        )
        assert ev.feature == AI_FEATURE_QUOTE_SIMILARITY
        assert ev.kind == AI_EVENT_KIND_REQUEST
        assert ev.provenance.decision == AI_DECISION_PENDING
        assert ev.provenance.embedding_model == "bge-m3"
        assert ev.provenance.suggestion_id == q.id
        assert ev.payload["query_kind"] == QUERY_KIND_TEXT
        assert ev.actor_coder_id == _HEX_CODER

    def test_bad_actor(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        q = _make_quote_search(proj.id)
        save_quote_search(tmp_path, q)
        with pytest.raises(ProjectValidationError):
            record_request_event_for_quote_search(
                tmp_path, q, actor_coder_id="bad"
            )


class TestRecordRequestEventForReviewPass:
    def test_basic_request(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = _make_review_pass(proj.id)
        save_review_pass(tmp_path, rp)
        ev = record_request_event_for_review_pass(
            tmp_path, rp, actor_coder_id=_HEX_CODER
        )
        assert ev.feature == AI_FEATURE_TRANSCRIPT_REVIEW
        assert ev.kind == AI_EVENT_KIND_REQUEST
        assert ev.provenance.suggestion_id == rp.id
        assert ev.payload["status"] == REVIEW_STATUS_PENDING


class TestRecordRequestEventForSecondCoderPass:
    def test_basic_request(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sp = _make_second_coder_pass(proj.id)
        save_second_coder_pass(tmp_path, sp)
        ev = record_request_event_for_second_coder_pass(tmp_path, sp)
        assert ev.feature == AI_FEATURE_SECOND_CODER
        assert ev.kind == AI_EVENT_KIND_REQUEST
        assert ev.provenance.suggestion_id == sp.id
        assert ev.payload["human_coder_id"] == _HEX_CODER
        assert ev.payload["status"] == SECOND_CODER_STATUS_PENDING
        # Actor defaults to the human coder if unspecified.
        assert ev.actor_coder_id == _HEX_CODER


# --------------------------------------------------------------------------- #
# build_invocation_log — empty / per-engine / multi-engine
# --------------------------------------------------------------------------- #


class TestBuildInvocationLogEmpty:
    def test_empty_project_returns_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        out = build_invocation_log(tmp_path, proj.id)
        assert out == []

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            build_invocation_log(tmp_path, "not-hex")


class TestBuildInvocationLogCodeSuggestions:
    def test_pending_and_rejected_present(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s_pending = _make_code_suggestion(
            proj.id,
            suggestion_id="aa" * 6,
            now="2026-01-01T00:00:00Z",
        )
        save_suggestion(tmp_path, s_pending)
        s_rejected = _make_code_suggestion(
            proj.id,
            suggestion_id="bb" * 6,
            now="2026-01-02T00:00:00Z",
        )
        save_suggestion(tmp_path, s_rejected)
        record_decision(
            s_rejected,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="not it",
            now="2026-01-02T00:01:00Z",
        )
        save_suggestion(tmp_path, s_rejected)
        out = build_invocation_log(tmp_path, proj.id)
        assert len(out) == 2
        # Sorted by created_at ascending.
        assert out[0].suggestion_id == "aa" * 6
        assert out[1].suggestion_id == "bb" * 6
        # Pending entry has decision=pending; rejected one has rejection
        # reason captured.
        assert out[0].decision == AI_DECISION_PENDING
        assert out[1].decision == AI_DECISION_REJECTED
        assert out[1].rejection_reason == "not it"
        assert out[1].decided_by_coder_id == _HEX_CODER

    def test_decision_filter_rejected_only(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = _make_code_suggestion(
            proj.id, suggestion_id="11" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s1)
        s2 = _make_code_suggestion(
            proj.id, suggestion_id="22" * 6, now="2026-01-02T00:00:00Z"
        )
        save_suggestion(tmp_path, s2)
        record_decision(
            s2,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="off",
            now="2026-01-02T00:01:00Z",
        )
        save_suggestion(tmp_path, s2)
        s3 = _make_code_suggestion(
            proj.id, suggestion_id="33" * 6, now="2026-01-03T00:00:00Z"
        )
        save_suggestion(tmp_path, s3)
        record_decision(
            s3,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
            now="2026-01-03T00:01:00Z",
        )
        save_suggestion(tmp_path, s3)
        out = build_invocation_log(
            tmp_path, proj.id, decision=AI_DECISION_REJECTED
        )
        assert len(out) == 1
        assert out[0].suggestion_id == "22" * 6


class TestBuildInvocationLogMultiEngine:
    def test_all_engines_appear(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)

        # F8.3 CodeSuggestion
        cs = _make_code_suggestion(
            proj.id, suggestion_id="11" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, cs)

        # F8.4 NewCodeSuggestion
        ncs = _make_new_code_suggestion(
            proj.id, suggestion_id="22" * 6, now="2026-01-02T00:00:00Z"
        )
        save_new_code_suggestion(tmp_path, ncs)

        # F8.8 MemoDraft
        md = _make_memo_draft(
            proj.id, draft_id="33" * 6, now="2026-01-03T00:00:00Z"
        )
        save_memo_draft(tmp_path, md)

        # F8.6 ReviewPass
        rp = _make_review_pass(
            proj.id, pass_id="44" * 6, now="2026-01-04T00:00:00Z"
        )
        save_review_pass(tmp_path, rp)

        # F8.7 SecondCoderPass
        scp = _make_second_coder_pass(
            proj.id, pass_id="55" * 6, now="2026-01-05T00:00:00Z"
        )
        save_second_coder_pass(tmp_path, scp)

        # F8.5 QuoteSearch
        qs = _make_quote_search(
            proj.id, search_id="66" * 6, now="2026-01-06T00:00:00Z"
        )
        save_quote_search(tmp_path, qs)

        out = build_invocation_log(tmp_path, proj.id)
        # All six invocations should appear.
        features = [e.feature for e in out]
        assert features == [
            AI_FEATURE_CODE_SUGGESTION,
            AI_FEATURE_NEW_CODE_SUGGESTION,
            AI_FEATURE_MEMO_DRAFT,
            AI_FEATURE_TRANSCRIPT_REVIEW,
            AI_FEATURE_SECOND_CODER,
            AI_FEATURE_QUOTE_SIMILARITY,
        ]
        # Searches and passes are request_only.
        by_feat = {e.feature: e for e in out}
        assert by_feat[AI_FEATURE_QUOTE_SIMILARITY].decision == DECISION_REQUEST_ONLY
        assert by_feat[AI_FEATURE_TRANSCRIPT_REVIEW].decision == DECISION_REQUEST_ONLY
        assert by_feat[AI_FEATURE_SECOND_CODER].decision == DECISION_REQUEST_ONLY
        # Records with decision lifecycles default to pending while no
        # decision has been recorded.
        assert by_feat[AI_FEATURE_CODE_SUGGESTION].decision == AI_DECISION_PENDING
        assert by_feat[AI_FEATURE_NEW_CODE_SUGGESTION].decision == AI_DECISION_PENDING
        assert by_feat[AI_FEATURE_MEMO_DRAFT].decision == AI_DECISION_PENDING

    def test_feature_filter_restricts(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        cs = _make_code_suggestion(
            proj.id, suggestion_id="11" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, cs)
        md = _make_memo_draft(
            proj.id, draft_id="22" * 6, now="2026-01-02T00:00:00Z"
        )
        save_memo_draft(tmp_path, md)
        out = build_invocation_log(
            tmp_path, proj.id, feature=AI_FEATURE_MEMO_DRAFT
        )
        assert [e.feature for e in out] == [AI_FEATURE_MEMO_DRAFT]
        assert out[0].related_entity_ids == [_HEX_CODE]

    def test_invalid_feature_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            build_invocation_log(tmp_path, proj.id, feature="bogus-feature")

    def test_invalid_decision_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            build_invocation_log(tmp_path, proj.id, decision="maybe")

    def test_invalid_actor_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            build_invocation_log(
                tmp_path, proj.id, actor_coder_id="not-hex"
            )


class TestBuildInvocationLogActorAndTimeFilters:
    def test_actor_filter_matches_decided_by(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = _make_code_suggestion(
            proj.id, suggestion_id="aa" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s1)
        record_decision(
            s1,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="x",
            now="2026-01-01T00:01:00Z",
        )
        save_suggestion(tmp_path, s1)
        s2 = _make_code_suggestion(
            proj.id, suggestion_id="bb" * 6, now="2026-01-02T00:00:00Z"
        )
        save_suggestion(tmp_path, s2)
        record_decision(
            s2,
            decision="rejected",
            coder_id=_HEX_CODER_2,
            rejection_reason="y",
            now="2026-01-02T00:01:00Z",
        )
        save_suggestion(tmp_path, s2)
        out = build_invocation_log(
            tmp_path, proj.id, actor_coder_id=_HEX_CODER
        )
        assert [e.suggestion_id for e in out] == ["aa" * 6]

    def test_actor_filter_matches_requested_by(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # SecondCoderPass captures human_coder_id which becomes
        # requested_by_coder_id on the entry.
        scp = _make_second_coder_pass(
            proj.id,
            pass_id="55" * 6,
            human_coder_id=_HEX_CODER_2,
            now="2026-02-01T00:00:00Z",
        )
        save_second_coder_pass(tmp_path, scp)
        # Another pass with a different requester.
        scp2 = _make_second_coder_pass(
            proj.id,
            pass_id="66" * 6,
            human_coder_id=_HEX_CODER,
            now="2026-02-02T00:00:00Z",
        )
        save_second_coder_pass(tmp_path, scp2)
        out = build_invocation_log(
            tmp_path, proj.id, actor_coder_id=_HEX_CODER_2
        )
        assert [e.suggestion_id for e in out] == ["55" * 6]
        assert out[0].requested_by_coder_id == _HEX_CODER_2

    def test_since_until_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = _make_code_suggestion(
            proj.id, suggestion_id="aa" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s1)
        s2 = _make_code_suggestion(
            proj.id, suggestion_id="bb" * 6, now="2026-02-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s2)
        s3 = _make_code_suggestion(
            proj.id, suggestion_id="cc" * 6, now="2026-03-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s3)
        out = build_invocation_log(
            tmp_path,
            proj.id,
            since="2026-01-15T00:00:00Z",
            until="2026-02-15T00:00:00Z",
        )
        assert [e.suggestion_id for e in out] == ["bb" * 6]


# --------------------------------------------------------------------------- #
# AI event cross-references
# --------------------------------------------------------------------------- #


class TestAIEventCrossReferences:
    def test_decision_event_appears_in_entry(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(
            proj.id, suggestion_id="aa" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s)
        record_decision(
            s,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="r",
            now="2026-01-01T00:01:00Z",
        )
        save_suggestion(tmp_path, s)
        ev = record_decision_event_for_code_suggestion(tmp_path, s)
        out = build_invocation_log(tmp_path, proj.id)
        assert len(out) == 1
        assert out[0].suggestion_id == "aa" * 6
        assert ev.id in out[0].ai_event_ids


# --------------------------------------------------------------------------- #
# count_invocations
# --------------------------------------------------------------------------- #


class TestCountInvocations:
    def test_empty_project(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = count_invocations(tmp_path, proj.id)
        assert c["total"] == 0
        assert c["pending"] == 0
        assert c["accepted"] == 0
        assert c["rejected"] == 0
        assert c[DECISION_REQUEST_ONLY] == 0

    def test_mix_of_decisions(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Two pending CodeSuggestions
        for i, sid in enumerate(["aa" * 6, "bb" * 6]):
            s = _make_code_suggestion(
                proj.id,
                suggestion_id=sid,
                now=f"2026-01-0{i + 1}T00:00:00Z",
            )
            save_suggestion(tmp_path, s)
        # One rejected
        r = _make_code_suggestion(
            proj.id, suggestion_id="cc" * 6, now="2026-01-03T00:00:00Z"
        )
        save_suggestion(tmp_path, r)
        record_decision(
            r,
            decision="rejected",
            coder_id=_HEX_CODER,
            rejection_reason="not it",
            now="2026-01-03T00:01:00Z",
        )
        save_suggestion(tmp_path, r)
        # One accepted
        a = _make_code_suggestion(
            proj.id, suggestion_id="dd" * 6, now="2026-01-04T00:00:00Z"
        )
        save_suggestion(tmp_path, a)
        record_decision(
            a,
            decision="accepted",
            coder_id=_HEX_CODER,
            accepted_code_id=_HEX_CODE,
            now="2026-01-04T00:01:00Z",
        )
        save_suggestion(tmp_path, a)
        # One quote search (request_only)
        qs = _make_quote_search(
            proj.id, search_id="ee" * 6, now="2026-01-05T00:00:00Z"
        )
        save_quote_search(tmp_path, qs)

        c = count_invocations(tmp_path, proj.id)
        assert c["total"] == 5
        assert c["pending"] == 2
        assert c["accepted"] == 1
        assert c["rejected"] == 1
        assert c[DECISION_REQUEST_ONLY] == 1

    def test_feature_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _make_code_suggestion(
            proj.id, suggestion_id="aa" * 6, now="2026-01-01T00:00:00Z"
        )
        save_suggestion(tmp_path, s)
        qs = _make_quote_search(
            proj.id, search_id="ee" * 6, now="2026-01-05T00:00:00Z"
        )
        save_quote_search(tmp_path, qs)
        c = count_invocations(
            tmp_path, proj.id, feature=AI_FEATURE_QUOTE_SIMILARITY
        )
        assert c["total"] == 1
        assert c[DECISION_REQUEST_ONLY] == 1
        assert c["pending"] == 0

    def test_invalid_feature_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            count_invocations(tmp_path, proj.id, feature="bogus")


# --------------------------------------------------------------------------- #
# Summary truncation
# --------------------------------------------------------------------------- #


class TestSummaryTruncation:
    def test_long_query_text_is_truncated(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        long_text = "alpha " * 200  # ~1.2k chars
        s = _make_code_suggestion(
            proj.id, suggestion_id="aa" * 6, query_text=long_text
        )
        save_suggestion(tmp_path, s)
        out = build_invocation_log(tmp_path, proj.id)
        assert len(out) == 1
        # 240 cap → bounded length and should end with the ellipsis marker.
        assert len(out[0].summary) <= 240
        assert out[0].summary.endswith("…")
