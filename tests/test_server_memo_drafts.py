"""F8.8 — wiring the AI memo-draft engine to the user-facing surface.

The pure data plane (``scribe.memo_drafts``) shipped in d860e180 with
123 unit tests covering the SeedSnippet / MemoDraft data model,
:func:`collect_seed_snippets`, prompt + response helpers,
:func:`draft_memo_for_code` orchestration, the decision lifecycle
(pending → accepted | modified | rejected), and the
:func:`promote_memo_draft_to_memo` helper. What was missing — and what
this file proves — is the user-facing surface:

  * POST /api/projects/<pid>/codes/<cid>/draft-memo   — generate a draft
  * GET  /api/projects/<pid>/memo-drafts              — list (+ filters)
  * GET  /api/projects/<pid>/memo-drafts/<did>        — single fetch
  * POST /api/projects/<pid>/memo-drafts/<did>/accept — promote to Memo
  * POST /api/projects/<pid>/memo-drafts/<did>/reject — record rejection
  * DELETE /api/projects/<pid>/memo-drafts/<did>      — discard

Plus the codebook editor graduates a "✨ Draft memo with AI" button
into the per-code form (only visible when editing an existing code),
and the project AI page graduates a "Memo drafts" card listing every
draft across the project.

We bypass the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair — the same shim the F8.3 / F8.5 / F8.6 / F8.7 server tests use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv
from scribe.ai_backend import (
    BackendConfig,
    EmbeddingResponse,
    GenerationResponse,
    PROVIDER_OLLAMA,
)
from scribe.ai_gate import GATE_OVERRIDE_FORCE_ON


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    monkeypatch.setattr(srv, "_ai_suggest_backend_override", None)
    return TestClient(srv.app), projects, output


class FakeBackend:
    """Returns a deterministic JSON memo draft for every prompt.

    ``embed`` returns a constant 1-d vector; the F8.8 path doesn't use
    embeddings at all (it picks seed material by code identity), but
    the suggestion-backend resolver insists on a working ``embed_fn``
    so we provide one.
    """

    name = PROVIDER_OLLAMA

    def __init__(self) -> None:
        self.embed_calls: list = []
        self.generate_calls: list = []
        # The default response: a clean strict-JSON memo draft so the
        # parser doesn't have to fall back. Tests that need a non-JSON
        # response replace this with a plain string before driving.
        self.response_text = json.dumps({
            "title": "On managing pain",
            "body": (
                "I notice the participant's first instinct is to keep "
                "moving — \"I just keep moving the next day\". This is "
                "different from the language of acceptance the literature "
                "uses; closer to a stoic refusal to let pain organise the "
                "day. Tension: is this resilience, or avoidance? Worth "
                "comparing to other coded segments where the same "
                "participant talks about rest."
            ),
            "rationale": (
                "The exemplar quote anchors the code in a specific "
                "behavioural strategy rather than an emotional posture."
            ),
        })

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        vectors = [(1.0,)] * len(req.inputs)
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=req.model,
            provider=self.name,
        )

    def generate(self, cfg, req, *, transport=None):
        self.generate_calls.append((cfg, req))
        return GenerationResponse(
            text=self.response_text, model=req.model, provider=self.name,
        )


def _install_fake_backend(
    backend: FakeBackend | None = None,
    *,
    embedding_model: str = "bge-m3",
) -> FakeBackend:
    backend = backend or FakeBackend()
    cfg = BackendConfig(
        provider=PROVIDER_OLLAMA,
        base_url="http://test",
        default_model="llama3.2:3b",
        default_embedding_model=embedding_model,
    )
    srv._ai_suggest_backend_override = (cfg, backend)
    return backend


# --------------------------------------------------------------------------- #
# Project / code / coder helpers
# --------------------------------------------------------------------------- #


def _new_project(client: TestClient, name: str = "F8.8 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_code(
    client: TestClient,
    pid: str,
    name: str = "managing pain",
    *,
    exemplars: list[str] | None = None,
) -> str:
    payload: dict = {
        "name": name,
        "definition": "moments of coping with chronic pain",
    }
    if exemplars:
        payload["exemplars"] = exemplars
    r = client.post(f"/api/projects/{pid}/codes", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _force_gate_on(client: TestClient, pid: str) -> None:
    r = client.put(
        f"/api/projects/{pid}/ai/gate",
        json={"override": GATE_OVERRIDE_FORCE_ON},
    )
    assert r.status_code == 200, r.text


def _configure_backend(client: TestClient, pid: str) -> None:
    r = client.put(
        f"/api/projects/{pid}/ai/backend",
        json={
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "default_model": "llama3.2:3b",
            "default_embedding_model": "bge-m3",
        },
    )
    assert r.status_code == 200, r.text


def _seed_ready_project(env, *, exemplars: list[str] | None = None):
    """Project with one code (exemplars seeded), the gate forced on, and
    the fake backend installed. Returns the standard tuple."""
    client, projects, output = env
    pid = _new_project(client)
    cid = _new_code(
        client,
        pid,
        exemplars=exemplars or ["I just keep moving the next day"],
    )
    backend = _install_fake_backend()
    _configure_backend(client, pid)
    _force_gate_on(client, pid)
    return client, pid, cid, backend


# --------------------------------------------------------------------------- #
# A. UI render — the codebook editor + AI page must surface F8.8
# --------------------------------------------------------------------------- #


class TestCodebookEditorF88Button:
    def test_codebook_editor_renders_draft_memo_button(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        body = r.text
        assert 'id="cb-draft-memo-btn"' in body
        assert 'data-test-id="cb-draft-memo-btn"' in body
        # Button labelled per the F8.8 spec.
        assert "Draft memo with AI" in body

    def test_codebook_editor_renders_draft_memo_modal(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        body = r.text
        for marker in (
            'id="cb-draft-memo-modal"',
            'data-test-id="cb-draft-memo-modal"',
            'data-test-id="cb-draft-memo-start"',
            'data-test-id="cb-draft-memo-save"',
            'data-test-id="cb-draft-memo-reject"',
            'data-test-id="cb-draft-memo-result-title"',
            'data-test-id="cb-draft-memo-result-body"',
            'data-test-id="cb-draft-memo-result-rationale"',
        ):
            assert marker in body, marker

    def test_codebook_editor_advertises_route(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        # The JS template literal collapses the code-id placeholder;
        # the path stem must be present.
        assert "/draft-memo" in r.text
        assert "/memo-drafts/" in r.text


class TestProjectAiPageF88Card:
    def test_project_ai_page_renders_memo_drafts_card(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        body = r.text
        assert 'id="aiMemoDraftsCard"' in body
        assert 'data-test-id="ai-memo-drafts-card"' in body

    def test_card_has_filter_and_list_elements(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        body = r.text
        for marker in (
            'data-test-id="ai-memo-drafts-filter-decision"',
            'data-test-id="ai-memo-drafts-list"',
            'data-test-id="ai-memo-drafts-count"',
        ):
            assert marker in body, marker

    def test_suggestion_surfaces_lists_f88_anchor(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-surface-f8-8"' in r.text


# --------------------------------------------------------------------------- #
# B. POST /codes/<cid>/draft-memo — validation
# --------------------------------------------------------------------------- #


class TestDraftMemoValidation:
    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/!nope!/codes/aaaaaaaaaaaa/draft-memo",
            json={},
        )
        assert r.status_code == 400

    def test_invalid_code_id_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codes/!!!/draft-memo",
            json={},
        )
        assert r.status_code == 400

    def test_unknown_code_returns_404(self, env) -> None:
        client, pid, _, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/ffffffffffff/draft-memo",
            json={},
        )
        assert r.status_code == 404

    def test_invalid_memo_type_returns_400(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo",
            json={"memo_type": "bogus_type"},
        )
        assert r.status_code == 400

    def test_invalid_max_seed_snippets_returns_400(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo",
            json={"max_seed_snippets": 0},
        )
        assert r.status_code == 400

    def test_max_seed_snippets_too_large_returns_400(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo",
            json={"max_seed_snippets": 9999},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# C. AI gate guard (F8.10)
# --------------------------------------------------------------------------- #


class TestDraftMemoGate:
    def test_gate_blocks_when_not_satisfied(self, env) -> None:
        """A fresh project with one code and no hand-coding trips the
        F8.10 default thresholds (≥ 8 codes, ≥ 2 hand-coded
        transcripts). The route must surface 412 + structured gate
        body so the UI can act."""
        client, _, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        _install_fake_backend()
        _configure_backend(client, pid)
        # Note: gate NOT forced on — the threshold should block.
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo",
            json={},
        )
        assert r.status_code == 412, r.text
        body = r.json()["detail"]
        assert body["detail"] == "AI gate not satisfied"
        assert body["gate"]["allowed"] is False


# --------------------------------------------------------------------------- #
# D. Happy path: draft → list → fetch → accept
# --------------------------------------------------------------------------- #


class TestDraftMemoHappyPath:
    def test_post_creates_pending_draft(self, env) -> None:
        client, pid, cid, backend = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo",
            json={},
        )
        assert r.status_code == 201, r.text
        d = r.json()["draft"]
        assert d["project_id"] == pid
        assert d["code_id"] == cid
        assert d["decision"] == "pending"
        assert d["title"].startswith("On managing pain")
        # FakeBackend.generate was called exactly once.
        assert len(backend.generate_calls) == 1

    def test_draft_appears_in_list(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        client.post(f"/api/projects/{pid}/codes/{cid}/draft-memo", json={})
        r = client.get(f"/api/projects/{pid}/memo-drafts")
        assert r.status_code == 200
        drafts = r.json()["drafts"]
        assert len(drafts) == 1
        assert drafts[0]["code_id"] == cid

    def test_list_filters_by_code_and_decision(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        client.post(f"/api/projects/{pid}/codes/{cid}/draft-memo", json={})
        # Filter by code_id (matches) returns it.
        r = client.get(
            f"/api/projects/{pid}/memo-drafts?code_id={cid}"
        )
        assert r.status_code == 200
        assert len(r.json()["drafts"]) == 1
        # Filter by code_id (mismatch) returns empty.
        r2 = client.get(
            f"/api/projects/{pid}/memo-drafts?code_id=ffffffffffff"
        )
        assert r2.status_code == 200
        assert r2.json()["drafts"] == []
        # Filter by decision=accepted (none yet) returns empty.
        r3 = client.get(
            f"/api/projects/{pid}/memo-drafts?decision=accepted"
        )
        assert r3.status_code == 200
        assert r3.json()["drafts"] == []
        # Filter by decision=pending matches.
        r4 = client.get(
            f"/api/projects/{pid}/memo-drafts?decision=pending"
        )
        assert r4.status_code == 200
        assert len(r4.json()["drafts"]) == 1

    def test_single_fetch_returns_draft(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        r2 = client.get(f"/api/projects/{pid}/memo-drafts/{did}")
        assert r2.status_code == 200
        assert r2.json()["draft"]["id"] == did

    def test_accept_promotes_to_memo(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        r2 = client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/accept",
            json={},
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert body["draft"]["decision"] == "accepted"
        assert body["memo"]["project_id"] == pid
        assert body["memo"]["provenance"]["source"] == "ai_drafted"
        assert body["memo"]["provenance"]["draft_id"] == did
        # The memo carries a back-link to the source code.
        link_targets = [
            (l["target_type"], l["target_id"]) for l in body["memo"]["links"]
        ]
        assert ("code", cid) in link_targets

    def test_accept_with_edits_records_modified(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        # Override the body — should auto-detect modification.
        r2 = client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/accept",
            json={"body": "completely rewritten by hand"},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["draft"]["decision"] == "modified"
        assert r2.json()["memo"]["body"] == "completely rewritten by hand"

    def test_accept_after_terminal_returns_409(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/accept", json={}
        )
        # Second accept refused.
        r2 = client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/accept", json={}
        )
        assert r2.status_code == 409

    def test_reject_records_decision(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        r2 = client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/reject",
            json={"rejection_reason": "off topic"},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["draft"]["decision"] == "rejected"
        assert r2.json()["draft"]["rejection_reason"] == "off topic"

    def test_reject_after_terminal_returns_409(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/reject", json={}
        )
        r2 = client.post(
            f"/api/projects/{pid}/memo-drafts/{did}/reject", json={}
        )
        assert r2.status_code == 409


# --------------------------------------------------------------------------- #
# E. Fetch / delete edge cases
# --------------------------------------------------------------------------- #


class TestDraftMemoEdgeCases:
    def test_invalid_draft_id_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memo-drafts/!!!")
        assert r.status_code == 400

    def test_unknown_draft_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memo-drafts/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_delete_unknown_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.delete(
            f"/api/projects/{pid}/memo-drafts/aaaaaaaaaaaa"
        )
        assert r.status_code == 404

    def test_delete_existing_returns_ok(self, env) -> None:
        client, pid, cid, _ = _seed_ready_project(env)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/draft-memo", json={}
        )
        did = r.json()["draft"]["id"]
        r2 = client.delete(f"/api/projects/{pid}/memo-drafts/{did}")
        assert r2.status_code == 200
        # And it's gone.
        r3 = client.get(f"/api/projects/{pid}/memo-drafts/{did}")
        assert r3.status_code == 404

    def test_invalid_decision_filter_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/memo-drafts?decision=bogus"
        )
        assert r.status_code == 400
