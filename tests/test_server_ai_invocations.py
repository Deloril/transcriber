"""End-to-end reachability tests for F9.6 (AI invocation log).

The F9.6 module ``scribe.ai_invocation_log`` shipped the read-side
aggregator (``build_invocation_log`` / ``count_invocations``) and the
write-side helpers (``record_decision_event_for_*``,
``record_request_event_for_*``) in 1f5de63. The F8.5 / F8.6 / F8.7 /
F8.8 endpoints already call the write-side helpers when a per-engine
decision lands. Until these endpoints + the project_ai.html panel
landed, the *read* surface (a unified, filterable, joined view of
every AI invocation across every engine, with first-class
``decision=rejected`` support) was unreachable from the UI.

This file covers the F9.6 read surface:

  * ``GET /api/projects/<pid>/ai/invocations`` — list (filterable)
  * ``GET /api/projects/<pid>/ai/invocations/counts`` — counter dict
  * The ``project_ai.html`` template renders the F9.6 panel
    (``data-test-feature="F9.6"``) so the routes are reachable from
    the UI, not just curl.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.ai_invocation_log import (
    DECISION_REQUEST_ONLY,
    INVOCATION_DECISIONS,
)
from scribe.ai_provenance import (
    AI_DECISION_ACCEPTED,
    AI_DECISION_REJECTED,
    AI_FEATURES,
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
    NewCodeProposal,
    NewCodeSuggestion,
    record_new_code_decision,
    save_new_code_suggestion,
)
from scribe.quote_similarity import (
    QUERY_KIND_TEXT,
    QuoteSearch,
    save_quote_search,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up an isolated TestClient with tmp project / upload / output dirs."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    upload.mkdir()
    output.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects_dir)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _make_project(client: TestClient, name: str = "InvP") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


_HEX_CODER = "a" * 12
_HEX_CODER_2 = "b" * 12
_HEX_SOURCE = "c" * 12
_HEX_CODE = "d" * 12


def _seed_rejected_code_suggestion(
    projects_root: Path,
    project_id: str,
    *,
    coder: str = _HEX_CODER,
    reason: str = "Not the right code",
    now: str | None = None,
    decided_at: str | None = None,
) -> CodeSuggestion:
    """Persist a CodeSuggestion and reject it. The F9.6 headline scenario."""
    sug = CodeSuggestion.new(
        project_id=project_id,
        source_id=_HEX_SOURCE,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w5",
        query_text="A span the AI tried to suggest for",
        embedding_model="bge-m3",
        generation_model="llama3.2:3b",
        now=now,
    )
    save_suggestion(projects_root, sug)
    record_decision(
        sug,
        decision=AI_DECISION_REJECTED,
        coder_id=coder,
        rejection_reason=reason,
        now=decided_at,
    )
    save_suggestion(projects_root, sug)
    return sug


def _seed_accepted_new_code_suggestion(
    projects_root: Path,
    project_id: str,
    *,
    coder: str = _HEX_CODER_2,
    now: str | None = None,
    decided_at: str | None = None,
) -> NewCodeSuggestion:
    sug = NewCodeSuggestion.new(
        project_id=project_id,
        source_id=_HEX_SOURCE,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w5",
        query_text="autonomy is being negotiated",
        embedding_model="bge-m3",
        generation_model="llama3.2:3b",
        proposals=[
            NewCodeProposal(
                name="negotiating-autonomy",
                definition="How autonomy is bargained",
            )
        ],
        now=now,
    )
    save_new_code_suggestion(projects_root, sug)
    record_new_code_decision(
        sug,
        decision=AI_DECISION_ACCEPTED,
        coder_id=coder,
        accepted_proposal_index=0,
        created_code_id=_HEX_CODE,
        now=decided_at,
    )
    save_new_code_suggestion(projects_root, sug)
    return sug


def _seed_pending_quote_search(
    projects_root: Path,
    project_id: str,
    *,
    now: str | None = None,
) -> QuoteSearch:
    """Searches have no decision lifecycle — they emit a request_only row."""
    qs = QuoteSearch.new(
        project_id=project_id,
        query_kind=QUERY_KIND_TEXT,
        query_text="feeling overwhelmed",
        embedding_model="bge-m3",
        now=now,
    )
    save_quote_search(projects_root, qs)
    return qs


def _seed_rejected_memo_draft(
    projects_root: Path,
    project_id: str,
    *,
    coder: str = _HEX_CODER,
    now: str | None = None,
    decided_at: str | None = None,
) -> MemoDraft:
    d = MemoDraft.new(
        project_id=project_id,
        code_id=_HEX_CODE,
        title="A first-cut memo",
        body="A short body the AI proposed.",
        generation_model="llama3.2:3b",
        now=now,
    )
    save_memo_draft(projects_root, d)
    record_memo_draft_decision(
        d,
        decision=AI_DECISION_REJECTED,
        coder_id=coder,
        rejection_reason="Tone is off",
        now=decided_at,
    )
    save_memo_draft(projects_root, d)
    return d


# --------------------------------------------------------------------------- #
# project_ai.html template renders the F9.6 panel
# --------------------------------------------------------------------------- #


class TestProjectAIPageF96Panel:
    def test_panel_is_present(self, server_env) -> None:
        """The /projects/<pid>/ai page must surface the F9.6 panel so
        the F9.6 routes are reachable from the user-facing surface,
        not just curl."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-feature="F9.6"' in r.text
        assert 'data-test-id="ai-invocations-card"' in r.text

    def test_panel_renders_filter_controls(self, server_env) -> None:
        """The F9.6 headline filter (decision=rejected) must be a
        first-class UI affordance — it's the whole point of the
        feature ("rejected suggestions are evidence too")."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        body = r.text
        assert 'data-test-id="ai-invocations-list"' in body
        assert 'data-test-id="ai-invocations-filter-feature"' in body
        assert 'data-test-id="ai-invocations-filter-decision"' in body
        assert 'data-test-id="ai-invocations-filter-actor"' in body
        assert 'data-test-id="ai-invocations-refresh"' in body
        # F9.6 headline filter option must be present in the dropdown
        assert "rejected" in body

    def test_panel_renders_counter_badges(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        body = r.text
        for k in (
            "ai-invocations-count-total",
            "ai-invocations-count-pending",
            "ai-invocations-count-accepted",
            "ai-invocations-count-modified",
            "ai-invocations-count-rejected",
            "ai-invocations-count-request-only",
        ):
            assert f'data-test-id="{k}"' in body

    def test_panel_links_to_invocations_route(self, server_env) -> None:
        """The card's JS must wire up against the F9.6 route."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert "/ai/invocations" in r.text


# --------------------------------------------------------------------------- #
# GET /ai/invocations — listing
# --------------------------------------------------------------------------- #


class TestListInvocations:
    def test_empty_project_returns_empty_list_with_vocab(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/invocations")
        assert r.status_code == 200
        body = r.json()
        assert body["invocations"] == []
        assert body["total"] == 0
        assert body["returned"] == 0
        assert body["truncated"] is False
        assert body["order"] == "desc"
        # Filter-vocabulary contracts the UI relies on:
        assert set(body["available_features"]) == set(AI_FEATURES)
        assert set(body["available_decisions"]) == set(INVOCATION_DECISIONS)

    def test_lists_seeded_invocations_newest_first(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        # Old: rejected suggestion
        s1 = _seed_rejected_code_suggestion(
            projects_root, pid, now="2026-04-01T10:00:00Z",
            decided_at="2026-04-01T11:00:00Z",
        )
        # Newer: accepted new-code suggestion
        s2 = _seed_accepted_new_code_suggestion(
            projects_root, pid, now="2026-04-02T10:00:00Z",
            decided_at="2026-04-02T11:00:00Z",
        )
        r = client.get(f"/api/projects/{pid}/ai/invocations")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        ids = [e["suggestion_id"] for e in body["invocations"]]
        # Newest-first per default order=desc
        assert ids == [s2.id, s1.id]

    def test_filter_decision_rejected_is_f96_headline(self, server_env) -> None:
        """The F9.6 headline use-case: "show me every rejection"."""
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        rejected = _seed_rejected_code_suggestion(
            projects_root, pid, reason="Not a good fit",
        )
        _seed_accepted_new_code_suggestion(projects_root, pid)
        _seed_pending_quote_search(projects_root, pid)

        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"decision": "rejected"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        row = body["invocations"][0]
        assert row["suggestion_id"] == rejected.id
        assert row["decision"] == "rejected"
        assert row["rejection_reason"] == "Not a good fit"

    def test_filter_decision_request_only(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        qs = _seed_pending_quote_search(projects_root, pid)
        _seed_rejected_code_suggestion(projects_root, pid)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"decision": "request_only"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["invocations"][0]["suggestion_id"] == qs.id
        assert body["invocations"][0]["decision"] == "request_only"

    def test_filter_feature_quote_similarity(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        qs = _seed_pending_quote_search(projects_root, pid)
        _seed_rejected_code_suggestion(projects_root, pid)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"feature": "quote_similarity"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["invocations"][0]["feature"] == "quote_similarity"
        assert body["invocations"][0]["suggestion_id"] == qs.id

    def test_filter_feature_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"feature": "totally-bogus"},
        )
        assert r.status_code == 400

    def test_filter_decision_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"decision": "made-up"},
        )
        assert r.status_code == 400

    def test_filter_actor_matches_decided_or_requested(self, server_env) -> None:
        """``actor_coder_id`` matches decided_by *or* requested_by —
        "anything this coder touched"."""
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        # Rejected suggestion, decided_by=_HEX_CODER
        s1 = _seed_rejected_code_suggestion(
            projects_root, pid, coder=_HEX_CODER,
        )
        # Accepted new code, decided_by=_HEX_CODER_2
        _seed_accepted_new_code_suggestion(
            projects_root, pid, coder=_HEX_CODER_2,
        )
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"actor_coder_id": _HEX_CODER},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["invocations"][0]["suggestion_id"] == s1.id

    def test_order_asc(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        s1 = _seed_rejected_code_suggestion(
            projects_root, pid, now="2026-04-01T10:00:00Z",
        )
        s2 = _seed_accepted_new_code_suggestion(
            projects_root, pid, now="2026-04-02T10:00:00Z",
        )
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"order": "asc"},
        )
        assert r.status_code == 200
        ids = [e["suggestion_id"] for e in r.json()["invocations"]]
        assert ids == [s1.id, s2.id]

    def test_order_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"order": "weird"},
        )
        assert r.status_code == 400

    def test_limit_truncates(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        # Three rows
        _seed_rejected_code_suggestion(
            projects_root, pid, now="2026-04-01T10:00:00Z",
        )
        _seed_accepted_new_code_suggestion(
            projects_root, pid, now="2026-04-02T10:00:00Z",
        )
        _seed_pending_quote_search(
            projects_root, pid, now="2026-04-03T10:00:00Z",
        )
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"limit": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["returned"] == 2
        assert body["truncated"] is True

    def test_limit_zero_disables_truncation(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        for i in range(5):
            _seed_rejected_code_suggestion(
                projects_root, pid,
                now=f"2026-04-0{i+1}T10:00:00Z",
                decided_at=f"2026-04-0{i+1}T11:00:00Z",
            )
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"limit": 0},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 5
        assert body["returned"] == 5
        assert body["truncated"] is False

    def test_limit_negative_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations",
            params={"limit": -1},
        )
        assert r.status_code == 400

    def test_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        # 12-char hex but not a real project
        r = client.get("/api/projects/000000000000/ai/invocations")
        assert r.status_code == 404

    def test_invalid_project_id_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex/ai/invocations")
        assert r.status_code == 400

    def test_rejected_row_carries_summary_and_models(self, server_env) -> None:
        """A rejected CodeSuggestion lands as a full row with summary,
        generation_model, embedding_model, decided_by, rejection_reason."""
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        sug = _seed_rejected_code_suggestion(
            projects_root, pid, reason="Wrong concept",
        )
        r = client.get(f"/api/projects/{pid}/ai/invocations")
        assert r.status_code == 200
        rows = r.json()["invocations"]
        assert len(rows) == 1
        row = rows[0]
        assert row["suggestion_id"] == sug.id
        assert row["feature"] == "code_suggestion"
        assert row["decision"] == "rejected"
        assert row["rejection_reason"] == "Wrong concept"
        assert row["generation_model"] == "llama3.2:3b"
        assert row["embedding_model"] == "bge-m3"
        assert row["decided_by_coder_id"] == _HEX_CODER
        assert row["summary"]  # truncated query_text


# --------------------------------------------------------------------------- #
# GET /ai/invocations/counts — counters
# --------------------------------------------------------------------------- #


class TestCountInvocations:
    def test_empty_project_zero_counts(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/invocations/counts")
        assert r.status_code == 200
        body = r.json()
        c = body["counts"]
        assert c["total"] == 0
        for d in INVOCATION_DECISIONS:
            assert c[d] == 0
        assert body["feature"] == ""

    def test_counts_reflect_decisions(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        _seed_rejected_code_suggestion(projects_root, pid)
        _seed_rejected_memo_draft(projects_root, pid)
        _seed_accepted_new_code_suggestion(projects_root, pid)
        _seed_pending_quote_search(projects_root, pid)
        r = client.get(f"/api/projects/{pid}/ai/invocations/counts")
        assert r.status_code == 200
        c = r.json()["counts"]
        assert c["total"] == 4
        assert c["rejected"] == 2
        assert c["accepted"] == 1
        assert c["request_only"] == 1
        assert c["pending"] == 0
        assert c["modified"] == 0

    def test_counts_filter_by_feature(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        _seed_rejected_code_suggestion(projects_root, pid)
        _seed_rejected_memo_draft(projects_root, pid)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations/counts",
            params={"feature": "memo_draft"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["feature"] == "memo_draft"
        c = body["counts"]
        assert c["total"] == 1
        assert c["rejected"] == 1
        assert c["accepted"] == 0

    def test_counts_invalid_feature_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/invocations/counts",
            params={"feature": "nonsense"},
        )
        assert r.status_code == 400

    def test_counts_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/000000000000/ai/invocations/counts")
        assert r.status_code == 404
