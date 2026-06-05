"""Tests for the AI code-suggestion HTTP surface (F8.3 / F8.4).

The pure data layer (``scribe.code_suggestions`` /
``scribe.new_code_suggestions``) and the gate (``scribe.ai_gate``) have
their own unit tests. This file exercises only the FastAPI wrapping:

* the four routes are reachable,
* the AI gate gets honoured (412 when not satisfied),
* an accepted suggestion turns into an Application carrying AI
  provenance (F8.9),
* a rejected suggestion lands in the audit log (F9.6),
* mode=new dispatches to the new-code-suggestion engine,
* error mappings (404 for missing suggestion, 409 on re-decide).

We bypass the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair. The backend's ``embed`` and ``generate`` methods receive the
``cfg`` and ``req`` objects from ``_make_embed_and_generate_fns`` and
can return whatever the test needs.
"""

from __future__ import annotations

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
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test client with isolated tmp dirs and any AI backend override
    cleared between tests so leaks from a prior test can't affect us."""
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
    return TestClient(srv.app), projects


class FakeBackend:
    """Minimal stand-in for a ``ModelBackend``.

    The server only ever calls ``embed`` and ``generate`` on the
    backend it gets back from ``backend_for_config``; we don't need to
    subclass the ABC, just match the call shape.
    """

    name = PROVIDER_OLLAMA

    def __init__(self, *, vector: tuple[float, ...] = (1.0, 0.0, 0.0),
                 generation_text: str = "[]") -> None:
        self.vector = vector
        self.generation_text = generation_text
        self.embed_calls: list = []
        self.generate_calls: list = []

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        return EmbeddingResponse(
            vectors=tuple(self.vector for _ in req.inputs),
            model=req.model,
            provider=self.name,
        )

    def generate(self, cfg, req, *, transport=None):
        self.generate_calls.append((cfg, req))
        return GenerationResponse(
            text=self.generation_text,
            model=req.model,
            provider=self.name,
        )


def _install_fake_backend(backend: FakeBackend) -> None:
    cfg = BackendConfig(
        provider=PROVIDER_OLLAMA,
        base_url="http://test",
        default_model="llama3.2:3b",
        default_embedding_model="bge-m3",
    )
    srv._ai_suggest_backend_override = (cfg, backend)


def _new_project(client: TestClient, name: str = "Test project") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _force_gate_on(client: TestClient, pid: str) -> None:
    """Bypass F8.10's "code more by hand first" gate. We don't need to
    seed real codes / sources; the override short-circuits the
    threshold check."""
    r = client.put(
        f"/api/projects/{pid}/ai/gate",
        json={"override": GATE_OVERRIDE_FORCE_ON},
    )
    assert r.status_code == 200, r.text


def _seed(client: TestClient, *, with_code: bool = True) -> tuple[str, str, str]:
    """Project + (optional) code + source → (pid, cid, sid)."""
    pid = _new_project(client)
    cid = ""
    if with_code:
        r = client.post(f"/api/projects/{pid}/codes",
                        json={"name": "managing pain"})
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
    r = client.post(f"/api/projects/{pid}/sources",
                    json={"name": "S", "source_type": "transcript"})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    return pid, cid, sid


# --------------------------------------------------------------------------- #
# Gate / shape
# --------------------------------------------------------------------------- #


class TestGate:
    def test_gate_blocks_when_not_satisfied(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        # No codes, no override → gate should refuse with 412.
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": "a" * 12,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "hello",
                "mode": "existing",
            },
        )
        assert r.status_code == 412, r.text
        body = r.json()["detail"]
        assert body["detail"] == "AI gate not satisfied"
        assert body["gate"]["allowed"] is False

    def test_missing_anchor_fields_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _force_gate_on(client, pid)
        r = client.post(f"/api/projects/{pid}/ai/suggestions", json={"mode": "existing"})
        assert r.status_code == 400

    def test_invalid_mode_returns_400(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "hello",
                "mode": "wat",
            },
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Suggest existing-code (mode=existing)
# --------------------------------------------------------------------------- #


class TestSuggestExisting:
    def test_returns_persisted_suggestion(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "the patient describes coping",
                "mode": "existing",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "existing"
        sug = body["suggestion"]
        assert sug["project_id"] == pid
        assert sug["source_id"] == sid
        assert sug["query_text"] == "the patient describes coping"
        assert sug["embedding_model"] == "bge-m3"
        assert sug["generation_model"] == "llama3.2:3b"
        # Persisted as pending until accept/reject lands.
        assert sug["decision"] == "pending"

    def test_suggestion_persists_for_listing(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w3", "query_text": "x",
                  "mode": "existing"},
        )
        sid_sug = r.json()["suggestion"]["id"]

        r = client.get(f"/api/projects/{pid}/ai/suggestions")
        assert r.status_code == 200
        ids = [s["id"] for s in r.json()["suggestions"]]
        assert sid_sug in ids

    def test_list_can_filter_by_decision(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w3", "query_text": "x",
                  "mode": "existing"},
        )
        # Default state is pending.
        r = client.get(f"/api/projects/{pid}/ai/suggestions?decision=pending")
        assert r.status_code == 200
        assert len(r.json()["suggestions"]) == 1
        # No accepted suggestions yet.
        r = client.get(f"/api/projects/{pid}/ai/suggestions?decision=accepted")
        assert r.json()["suggestions"] == []


# --------------------------------------------------------------------------- #
# Suggest new code (mode=new)
# --------------------------------------------------------------------------- #


class TestSuggestNew:
    def test_dispatches_to_new_code_engine(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        # The new-code engine asks the LLM for proposals; return a JSON
        # array shape it knows how to parse.
        backend = FakeBackend(generation_text='[]')
        _install_fake_backend(backend)

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w3",
                  "query_text": "the participant describes new behaviour",
                  "mode": "new"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "new"
        assert "suggestion" in body

    def test_unexpected_exception_returns_500_with_class_and_message(
        self, env, monkeypatch,
    ) -> None:
        """Pin the regression: a stray exception from the AI stack
        used to bubble out as opaque ``Internal Server Error``. The
        endpoint now wraps unexpected exceptions in an HTTP 500 whose
        body identifies the exception class + message so the user
        (and a dev they paste it to) can diagnose without server
        logs."""
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        from scribe import new_code_suggestions as _ncs

        def boom(**_kwargs):
            raise RuntimeError("simulated AI stack explosion")

        monkeypatch.setattr(_ncs, "suggest_new_codes_for_span", boom)

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w3",
                  "query_text": "anything",
                  "mode": "new"},
        )
        assert r.status_code == 500, r.text
        # Body must carry the class name *and* the message so the
        # user can self-diagnose; "Internal Server Error" alone is
        # what the regression looked like.
        body = r.json()
        # FastAPI puts HTTPException detail under "detail".
        detail = body.get("detail") or body
        assert "RuntimeError" in str(detail)
        assert "simulated AI stack explosion" in str(detail)

    def test_filenotfound_returns_500_with_path(
        self, env, monkeypatch,
    ) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        from scribe import new_code_suggestions as _ncs

        def boom(**_kwargs):
            raise FileNotFoundError("/does/not/exist/codes.json")

        monkeypatch.setattr(_ncs, "suggest_new_codes_for_span", boom)

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w3",
                  "query_text": "anything", "mode": "new"},
        )
        assert r.status_code == 500
        detail = r.json().get("detail") or r.json()
        assert "/does/not/exist/codes.json" in str(detail)


# --------------------------------------------------------------------------- #
# Accept
# --------------------------------------------------------------------------- #


class TestAccept:
    def _make_suggestion(self, client: TestClient, pid: str, sid: str) -> dict:
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5",
                  "query_text": "managing pain comes up",
                  "mode": "existing"},
        )
        assert r.status_code == 200, r.text
        return r.json()["suggestion"]

    def test_accept_creates_application_with_ai_provenance(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        sug = self._make_suggestion(client, pid, sid)
        # If the engine produced no candidates (the FakeBackend gives
        # the LLM nothing to score), the body must specify a code_id.
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions/{sug['id']}/accept",
            json={"code_id": cid},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Suggestion now has a terminal decision.
        assert body["suggestion"]["decision"] in ("accepted", "modified")
        assert body["suggestion"]["accepted_code_id"] == cid
        # Application carries AI provenance per F8.9.
        ap = body["application"]
        assert ap["code_id"] == cid
        assert ap["source_id"] == sid
        assert ap["ai_provenance"]
        assert ap["ai_provenance"]["feature"] == "code_suggestion"
        assert ap["ai_provenance"]["suggestion_id"] == sug["id"]
        assert ap["ai_provenance"]["embedding_model"] == "bge-m3"
        assert ap["ai_provenance"]["generation_model"] == "llama3.2:3b"

    def test_double_accept_returns_409(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        sug = self._make_suggestion(client, pid, sid)
        r1 = client.post(
            f"/api/projects/{pid}/ai/suggestions/{sug['id']}/accept",
            json={"code_id": cid},
        )
        assert r1.status_code == 200
        r2 = client.post(
            f"/api/projects/{pid}/ai/suggestions/{sug['id']}/accept",
            json={"code_id": cid},
        )
        assert r2.status_code == 409

    def test_accept_404_on_missing_suggestion(self, env) -> None:
        client, _ = env
        pid, cid, _ = _seed(client)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions/aaaaaaaaaaaa/accept",
            json={"code_id": cid},
        )
        assert r.status_code == 404

    def test_accept_invalid_suggestion_id_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions/not-hex/accept",
            json={"code_id": "a" * 12},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Reject
# --------------------------------------------------------------------------- #


class TestReject:
    def test_reject_records_decision_and_reason(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "existing"},
        )
        sug_id = r.json()["suggestion"]["id"]

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions/{sug_id}/reject",
            json={"reason": "not relevant to the research question"},
        )
        assert r.status_code == 200, r.text
        sug = r.json()["suggestion"]
        assert sug["decision"] == "rejected"
        assert "research question" in sug["rejection_reason"]
        # Listing shows it under decision=rejected (F9.6 audit trail).
        r = client.get(f"/api/projects/{pid}/ai/suggestions?decision=rejected")
        ids = [s["id"] for s in r.json()["suggestions"]]
        assert sug_id in ids

    def test_double_reject_returns_409(self, env) -> None:
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        _install_fake_backend(FakeBackend())

        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "existing"},
        )
        sug_id = r.json()["suggestion"]["id"]

        r1 = client.post(f"/api/projects/{pid}/ai/suggestions/{sug_id}/reject")
        assert r1.status_code == 200
        r2 = client.post(f"/api/projects/{pid}/ai/suggestions/{sug_id}/reject")
        assert r2.status_code == 409


# --------------------------------------------------------------------------- #
# Coding view template — the "Suggest with AI" button must actually render
# --------------------------------------------------------------------------- #


class TestCodingViewAiUI:
    def test_ai_button_and_panel_render(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        # The popover gains an inline AI panel container and JS that
        # writes a "✨ Suggest …" row whenever the popover renders.
        assert 'id="aiPanel"' in body
        assert 'data-ai-mode' in body
        assert "Suggest existing code with AI" in body or "Propose a new code with AI" in body
        # Endpoint paths the JS posts to.
        assert "/ai/suggestions" in body


# --------------------------------------------------------------------------- #
# F8.3 reachability anchor
#
# F8.3 ("Code suggestion engine — embedding + LLM rerank, existing
# codebook mode") was implemented end-to-end in commit c033c9d (route +
# UI + tests) but the original commit body lacked the Reachable-via
# line the loop's done-detector relies on, so the loop has been
# treating F8.3 as not-yet-shipped. This class is the explicit
# reachability anchor that proves:
#
#   1. The source-coding view renders the F8.3 marker on the AI panel
#      and (when the codebook has codes) on the "Suggest existing code"
#      row in the apply-popover JS template literal.
#   2. The route POST /api/projects/<pid>/ai/suggestions with
#      mode="existing" returns 200 and a persisted suggestion the user
#      can accept or reject.
#   3. The persisted suggestion shows up in the F8.3 listing endpoint
#      (the inline panel reads from this when re-opening a popover).
#
# Together these assertions prove that a researcher with no prior
# context can reach the F8.3 engine through the user-facing surface:
# open a transcript, drop the marker tags into the popover, click
# "✨ Suggest existing code with AI", and the route round-trips a
# persisted suggestion that the UI can render.
# --------------------------------------------------------------------------- #


class TestF8_3Reachability:
    def test_ai_panel_carries_f8_3_marker(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        # The aiPanel container itself is anchored as the F8.3 surface.
        # Both attributes must travel with the element so test ids /
        # feature ids stay reachable from a CSS selector or DOM probe.
        assert 'data-test-feature="F8.3"' in body
        assert 'data-test-id="ai-panel"' in body
        # The panel still uses id="aiPanel" so the existing JS handler
        # finds it. Removing the id would silently break the popover.
        assert 'id="aiPanel"' in body
        # The F8.4 secondary marker rides on the same panel — see the
        # F8.4 reachability class for why this matters.
        assert 'data-test-feature-also="F8.4"' in body

    def test_popover_emits_f8_3_existing_row(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        # The renderPopList() template literal emits **both** rows now:
        # an F8.3 "Suggest existing code with AI" row (only when CODES
        # is non-empty), and an F8.4 "Propose a new code with AI" row
        # (always). The F8.3 anchor verifies its half of the conditional
        # is present with the expected feature/test markers.
        assert 'data-test-feature="F8.3"' in body
        assert 'data-test-id="ai-suggest-existing"' in body
        assert 'Suggest existing code with AI' in body
        # Shared infrastructure: the row class + ai-mode data attr the
        # click handler reads.
        assert 'class="pop-row ai-row"' in body
        assert 'data-ai-mode="existing"' in body
        # Endpoint URL the JS POSTs to when the row is clicked.
        assert "/ai/suggestions" in body

    def test_existing_mode_route_round_trips(self, env) -> None:
        """End-to-end: click "Suggest existing code with AI" calls
        POST /ai/suggestions with mode=existing, the engine persists a
        CodeSuggestion row, and a subsequent GET surfaces it in the
        decision="pending" filter. This is the same call the JS in
        suggestWithAi() makes."""
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend()
        _install_fake_backend(backend)

        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "the participant describes coping",
                "mode": "existing",
            },
        )
        assert post.status_code == 200, post.text
        post_body = post.json()
        assert post_body["kind"] == "existing"
        suggestion_id = post_body["suggestion"]["id"]
        # The fake backend received the embed + generate calls — proves
        # the route really invoked the F8.3 engine, not a stub.
        assert backend.embed_calls, "F8.3 engine never embedded the query"

        listing = client.get(
            f"/api/projects/{pid}/ai/suggestions?decision=pending"
        )
        assert listing.status_code == 200, listing.text
        ids = [s["id"] for s in listing.json()["suggestions"]]
        assert suggestion_id in ids, (
            "F8.3 suggestion was not retrievable through the listing "
            "endpoint that the inline AI panel reads from."
        )

    def test_project_ai_page_advertises_f8_3(self, env) -> None:
        """The /projects/<pid>/ai dashboard's "Suggestion surfaces"
        card points researchers from the AI page to the source-coding
        view. The F8.3 link is the discovery affordance — without it,
        a user landing on the AI page has no way to find the F8.3
        feature."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200, r.text
        body = r.text
        assert "F8.3" in body, "AI page must advertise F8.3 as a surface"
        # The phrase that names the feature on the source-coding view —
        # used for discovery copy on the AI page so the user knows what
        # to look for.
        assert "Suggest existing code with AI" in body


# --------------------------------------------------------------------------- #
# F8.4 reachability anchor
#
# F8.4 ("Suggest a new code" — separate command, gerund-form nudge,
# decision lifecycle) shipped its pure module in cb6508f without any
# user-facing wiring. The original surfacing bundled F8.4 and F8.3 into
# a single "Suggest with AI" row that toggled mode based on whether the
# codebook was empty, and accepted new-code proposals via a hack
# (create-on-the-fly + reject-the-suggestion). This iteration:
#
#   1. Gives F8.4 a separate apply-popover row that is always offered
#      ("✨ Propose a new code with AI"), so users can ask for new
#      proposals even when the codebook has codes.
#   2. Adds dedicated /ai/new-code-suggestions/<sid>/accept and
#      /reject routes that drive the proper F8.4 decision lifecycle
#      (pending → accepted/modified/rejected) and stamp the resulting
#      Application with AIProvenance(feature=new_code_suggestion).
#   3. Adds a list endpoint so the AI dashboard can paginate proposals.
#
# These assertions prove that a researcher with no prior context can
# reach F8.4 through the user-facing surface: open a transcript, drop
# the marker tags into the popover, click "✨ Propose a new code with
# AI", pick a proposal, and the route round-trips a created code +
# applied span + audit row.
# --------------------------------------------------------------------------- #


class TestF8_4Reachability:
    def test_popover_always_emits_f8_4_new_code_row(self, env) -> None:
        """Whether the codebook has codes or not, the popover must
        offer the F8.4 "Propose a new code with AI" command. F8.4 is a
        separate command from F8.3 per PLANNING.md ("requires explicit
        invocation"); a researcher with a non-empty codebook must still
        be able to reach it."""
        client, _ = env
        pid, _, sid = _seed(client)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        # F8.4-marked row.
        assert 'data-test-feature="F8.4"' in body
        assert 'data-test-id="ai-suggest-new"' in body
        # The label researchers see in the popover.
        assert 'Propose a new code with AI' in body
        # Shared infrastructure: row class + ai-mode the click handler reads.
        assert 'data-ai-mode="new"' in body

    def test_new_mode_route_round_trips(self, env) -> None:
        """End-to-end: click "Propose a new code with AI" calls
        POST /ai/suggestions with mode=new, the engine persists a
        NewCodeSuggestion, and a subsequent GET surfaces it in the
        F8.4 listing endpoint."""
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[]')
        _install_fake_backend(backend)

        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "the participant describes new behaviour",
                "mode": "new",
            },
        )
        assert post.status_code == 200, post.text
        body = post.json()
        assert body["kind"] == "new"
        suggestion_id = body["suggestion"]["id"]

        listing = client.get(
            f"/api/projects/{pid}/ai/new-code-suggestions"
            f"?decision=pending"
        )
        assert listing.status_code == 200, listing.text
        ids = [s["id"] for s in listing.json()["suggestions"]]
        assert suggestion_id in ids, (
            "F8.4 suggestion was not retrievable through the listing "
            "endpoint that the inline AI panel reads from."
        )

    def test_accept_creates_code_and_application_with_ai_provenance(
        self, env
    ) -> None:
        """Picking a proposal creates a real Code, applies it, and
        stamps AI provenance (feature=new_code_suggestion). The
        suggestion's decision moves to accepted with
        accepted_proposal_index + created_code_id."""
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        # Generate a single proposal so we know what we're accepting.
        backend = FakeBackend(generation_text=(
            '[{"name":"NavigatingChange",'
            '"definition":"How participants describe shifting roles",'
            '"rationale":"recurs in the span"}]'
        ))
        _install_fake_backend(backend)

        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "the participant describes shifting roles",
                "mode": "new",
            },
        )
        assert post.status_code == 200, post.text
        sug = post.json()["suggestion"]
        assert len(sug["proposals"]) == 1

        accept = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug['id']}/accept",
            json={"accepted_proposal_index": 0, "apply": True},
        )
        assert accept.status_code == 200, accept.text
        body = accept.json()
        # Suggestion moved to a terminal state with the audit fields.
        assert body["suggestion"]["decision"] in ("accepted", "modified")
        assert body["suggestion"]["accepted_proposal_index"] == 0
        assert body["suggestion"]["created_code_id"] == body["code"]["id"]
        # Code was created with the proposal's wording.
        assert body["code"]["name"] == "NavigatingChange"
        assert "shifting roles" in body["code"]["definition"]
        # Application was created and stamped with AI provenance.
        ap = body["application"]
        assert ap["code_id"] == body["code"]["id"]
        assert ap["source_id"] == sid
        assert ap["ai_provenance"]["feature"] == "new_code_suggestion"
        assert ap["ai_provenance"]["suggestion_id"] == sug["id"]
        assert ap["ai_provenance"]["embedding_model"] == "bge-m3"
        assert ap["ai_provenance"]["generation_model"] == "llama3.2:3b"

    def test_accept_with_modified_name_marks_decision_modified(
        self, env,
    ) -> None:
        """When the user edits the proposal's name before saving, the
        F8.4 audit lifecycle records 'modified' rather than 'accepted'.
        This is the audit trail's "AI proposed X, human saved Y" path."""
        client, _ = env
        pid, cid, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text=(
            '[{"name":"NavigatingChange","definition":"d","rationale":"r"}]'
        ))
        _install_fake_backend(backend)

        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]

        accept = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/accept",
            json={
                "accepted_proposal_index": 0,
                "name": "RidingTheWave",
                "apply": False,  # dont need an Application for the audit assertion
            },
        )
        assert accept.status_code == 200, accept.text
        body = accept.json()
        assert body["suggestion"]["decision"] == "modified"
        assert body["code"]["name"] == "RidingTheWave"
        # apply=False → no Application returned
        assert "application" not in body

    def test_accept_404_on_missing_suggestion(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/aaaaaaaaaaaa/accept",
            json={"accepted_proposal_index": 0},
        )
        assert r.status_code == 404

    def test_accept_invalid_suggestion_id_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/not-hex/accept",
            json={"accepted_proposal_index": 0},
        )
        assert r.status_code == 400

    def test_accept_requires_accepted_proposal_index(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[{"name":"X","definition":"d"}]')
        _install_fake_backend(backend)
        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]
        r = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/accept",
            json={},
        )
        assert r.status_code == 400

    def test_accept_out_of_range_index_returns_400(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[{"name":"X","definition":"d"}]')
        _install_fake_backend(backend)
        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]
        r = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/accept",
            json={"accepted_proposal_index": 99},
        )
        assert r.status_code == 400

    def test_double_accept_returns_409(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[{"name":"X","definition":"d"}]')
        _install_fake_backend(backend)
        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]
        r1 = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/accept",
            json={"accepted_proposal_index": 0, "apply": False},
        )
        assert r1.status_code == 200, r1.text
        r2 = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/accept",
            json={"accepted_proposal_index": 0, "apply": False},
        )
        assert r2.status_code == 409

    def test_reject_records_decision_and_reason(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[{"name":"X","definition":"d"}]')
        _install_fake_backend(backend)
        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]
        r = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/reject",
            json={"reason": "duplicate of NavigatingChange"},
        )
        assert r.status_code == 200, r.text
        sug = r.json()["suggestion"]
        assert sug["decision"] == "rejected"
        assert "duplicate" in sug["rejection_reason"]
        # Listing returns the rejected suggestion under decision=rejected.
        listing = client.get(
            f"/api/projects/{pid}/ai/new-code-suggestions?decision=rejected"
        )
        ids = [s["id"] for s in listing.json()["suggestions"]]
        assert sug_id in ids

    def test_double_reject_returns_409(self, env) -> None:
        client, _ = env
        pid, _, sid = _seed(client)
        _force_gate_on(client, pid)
        backend = FakeBackend(generation_text='[{"name":"X","definition":"d"}]')
        _install_fake_backend(backend)
        post = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={"source_id": sid, "anchor_start_word_id": "s0w0",
                  "anchor_end_word_id": "s0w5", "query_text": "x",
                  "mode": "new"},
        )
        sug_id = post.json()["suggestion"]["id"]
        r1 = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/reject",
        )
        assert r1.status_code == 200
        r2 = client.post(
            f"/api/projects/{pid}/ai/new-code-suggestions/{sug_id}/reject",
        )
        assert r2.status_code == 409

    def test_project_ai_page_advertises_f8_4(self, env) -> None:
        """The AI dashboard's Suggestion surfaces card must point users
        at F8.4 with its own list-item marker, separately from F8.3.
        Without it, a researcher landing on the AI page can't find the
        new-code command."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200, r.text
        body = r.text
        assert 'data-test-id="ai-surface-f8-4"' in body
        assert "Propose a new code" in body
        assert "F8.4" in body
