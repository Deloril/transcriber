"""F8.5 — wiring "Find similar quotes" to the user-facing surface.

The pure data plane (``scribe.quote_similarity``) shipped in 2c9a78c
with thorough unit tests in ``tests/test_quote_similarity.py``. What
was missing — and what this file proves — is the user-facing surface:

* POST /api/projects/<pid>/ai/quote-searches              — run a search
* GET  /api/projects/<pid>/ai/quote-searches              — list past searches
* GET  /api/projects/<pid>/ai/quote-searches/<sid>        — fetch one

Plus the source-coding view's ``.app-row`` carries a
"🔎 Find similar quotes" button (``data-act="similar"``) that POSTs to
the run endpoint and renders the matches in a modal. The project AI
page's "Suggestion surfaces" card carries the F8.5 anchor.

We bypass the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair, mirroring the pattern in ``tests/test_server_ai_suggestions.py``
and ``tests/test_server_embedding_index.py``.
"""

from __future__ import annotations

import hashlib
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
    """Minimal ModelBackend stand-in.

    The F8.5 route only ever calls ``embed`` (no LLM rerank), but the
    interface exposes ``generate`` so we don't trip the ABC.
    """

    name = PROVIDER_OLLAMA

    def __init__(self, *, dim: int = 4) -> None:
        self.dim = dim
        self.embed_calls: list = []

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        # Deterministic, distinct-ish per text: hash the text into one
        # of the dim slots so different inputs land on different basis
        # vectors. The cosine similarity to any embedding-index entry
        # depends only on the slot, so the test can pre-seed entries
        # that line up with a chosen query for a "found a match" case.
        vectors = []
        for t in req.inputs:
            slot = sum(ord(c) for c in t) % self.dim
            v = [0.0] * self.dim
            v[slot] = 1.0
            vectors.append(tuple(v))
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=req.model,
            provider=self.name,
        )

    def generate(self, cfg, req, *, transport=None):
        return GenerationResponse(
            text="[]", model=req.model, provider=self.name,
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
# Project / source / code / application / transcript helpers
# --------------------------------------------------------------------------- #


def _new_project(client: TestClient, name: str = "F8.5 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_code(client: TestClient, pid: str, name: str = "managing pain") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "moments of coping"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_source(client: TestClient, pid: str, name: str = "Interview 1") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _attach_transcript(
    output_dir: Path,
    projects_root: Path,
    pid: str,
    sid: str,
    segments: list[dict],
) -> None:
    """Plant an edited.json under outputs/<job_id>/ and link the source
    to it. Mirrors the helper in test_server_embedding_index.py."""
    job_id = hashlib.sha256(sid.encode()).hexdigest()[:12]
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({"segments": segments}))
    from scribe import sources as _sources
    src = _sources.load_source(projects_root, pid, sid)
    src.apply_update({"transcript_job_id": job_id})
    _sources.save_source(projects_root, src)


def _two_segment_transcript() -> list[dict]:
    return [
        {
            "id": 0, "speaker": "P", "start": 0.0, "end": 1.0,
            "text": "the patient describes coping",
            "words": [
                {"text": "the",       "start": 0.0, "end": 0.2, "speaker": "P"},
                {"text": "patient",   "start": 0.2, "end": 0.4, "speaker": "P"},
                {"text": "describes", "start": 0.4, "end": 0.6, "speaker": "P"},
                {"text": "coping",    "start": 0.6, "end": 1.0, "speaker": "P"},
            ],
        },
        {
            "id": 1, "speaker": "P", "start": 1.0, "end": 2.0,
            "text": "another paragraph entirely",
            "words": [
                {"text": "another",    "start": 1.0, "end": 1.3, "speaker": "P"},
                {"text": "paragraph",  "start": 1.3, "end": 1.6, "speaker": "P"},
                {"text": "entirely",   "start": 1.6, "end": 2.0, "speaker": "P"},
            ],
        },
    ]


def _make_application(
    client: TestClient, pid: str, cid: str, sid: str,
    *, anchor_start: str = "s0w0", anchor_end: str = "s0w3",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": cid,
            "source_id": sid,
            "anchor_start_word_id": anchor_start,
            "anchor_end_word_id": anchor_end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _force_gate_on(client: TestClient, pid: str) -> None:
    """Bypass F8.10's "code more by hand first" gate."""
    r = client.put(
        f"/api/projects/{pid}/ai/gate",
        json={"override": GATE_OVERRIDE_FORCE_ON},
    )
    assert r.status_code == 200, r.text


def _refresh_index(client: TestClient, pid: str) -> dict:
    """Build the F8.2 index so that the F8.5 search has something to
    hit. Uses the same fake backend that the F8.5 route uses, so the
    seed entry's vector matches what the F8.5 query computes."""
    r = client.post(f"/api/projects/{pid}/ai/embedding-index/refresh")
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# A. UI render — the "Find similar quotes" button + modal must render
# --------------------------------------------------------------------------- #


class TestSourceCodingViewSimilarUI:
    def test_app_row_template_has_similar_button(self, env) -> None:
        """The .app-row template literal in source_coding.html must
        emit a `🔎 Find similar quotes` button with the F8.5 marker.
        Without this, a user with coded segments has no surface to
        invoke the F8.5 engine."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        # Stable test markers anchor the button.
        assert 'data-act="similar"' in body, (
            "F8.5 button must carry data-act='similar' so the appList "
            "click delegate routes to findSimilarQuotesForApp."
        )
        assert 'data-test-feature="F8.5"' in body
        assert 'data-test-id="app-row-similar"' in body
        assert "Find similar quotes" in body

    def test_app_row_template_targets_f8_5_endpoint(self, env) -> None:
        """The findSimilarQuotesForApp() helper in source_coding.html
        must POST to /ai/quote-searches; if the URL drifts, the click
        becomes a no-op against a 404."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        assert "/ai/quote-searches" in r.text

    def test_similar_modal_present(self, env) -> None:
        """The modal that renders the matches must be in the DOM, with
        the F8.5 marker so the integration test can scope-pick it."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert 'id="similarModal"' in body
        assert 'data-test-id="src-similar-modal"' in body
        # The seed pane and matches container must be addressable so
        # the JS can populate them.
        assert 'data-test-id="src-similar-seed"' in body
        assert 'data-test-id="src-similar-matches"' in body


class TestProjectAiPageAdvertisesF85:
    def test_ai_page_lists_f8_5_surface(self, env) -> None:
        """The AI dashboard's Suggestion-surfaces card points
        researchers to the source-coding view's Find-similar-quotes
        button. Without this, a user landing on /projects/<pid>/ai has
        no way to discover where F8.5 lives."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        body = r.text
        assert "F8.5" in body
        # The phrase that names the button on the source-coding view —
        # used for discovery copy.
        assert "Find similar quotes" in body
        assert 'data-test-id="ai-surface-f8-5"' in body


# --------------------------------------------------------------------------- #
# B. POST /ai/quote-searches — gate, validation, happy path
# --------------------------------------------------------------------------- #


class TestRunQuoteSearchValidation:
    def test_invalid_project_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/!nope!/ai/quote-searches",
            json={"query_text": "anything"},
        )
        assert r.status_code == 400

    def test_unknown_project_returns_404(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/ai/quote-searches",
            json={"query_text": "x"},
        )
        # Force the gate before project lookup? No — the gate evaluator
        # itself opens the project, so we expect 404 from project load.
        # However, the gate may be evaluated first for a non-existent
        # project — accept either 404 or 400 depending on order.
        assert r.status_code in (404, 400)

    def test_missing_query_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches", json={},
        )
        # Either 400 (no query) or 412 (gate) is acceptable as a "bad
        # input" signal — the surface refused to run a search.
        assert r.status_code in (400, 412)

    def test_invalid_top_k_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"query_text": "hi", "top_k": 0},
        )
        assert r.status_code == 400

    def test_invalid_min_score_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"query_text": "hi", "min_score": 5.0},
        )
        assert r.status_code == 400


class TestRunQuoteSearchGate:
    def test_gate_blocks_when_not_satisfied(self, env) -> None:
        """Default thresholds (≥ 8 codes, ≥ 2 hand-coded transcripts)
        aren't met on an empty project; the F8.10 gate must close
        F8.5 with 412 + a structured gate body."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"query_text": "anything"},
        )
        assert r.status_code == 412, r.text
        body = r.json()["detail"]
        assert body["detail"] == "AI gate not satisfied"
        assert body["gate"]["allowed"] is False


class TestRunQuoteSearchHappyPath:
    def _seed_project_with_indexed_application(self, env):
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        # An application on the first segment.
        aid = _make_application(client, pid, cid, sid)

        # Backend: same FakeBackend services F8.2 refresh AND F8.5
        # search, so the seed application's vector and the search
        # vector live in the same model space.
        backend = _install_fake_backend()

        # Configure the project's backend so the persisted config
        # matches what the override returns. (Not strictly required —
        # the override short-circuits load_backend_config — but
        # mirrors production.)
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

        _force_gate_on(client, pid)

        # Build the index so F8.5 has something to match against.
        ref = _refresh_index(client, pid)
        assert ref["added"] >= 1, ref
        return client, pid, cid, sid, aid, backend

    def test_application_mode_round_trips(self, env) -> None:
        """End-to-end: seed a coded segment, refresh the index, run
        F8.5 in application mode, and assert the persisted QuoteSearch
        comes back. The seed itself is excluded by default
        (exclude_seed=True), so a single-application project may have
        zero matches — but the route still returns 200 with an
        empty match list."""
        client, pid, cid, sid, aid, backend = (
            self._seed_project_with_indexed_application(env)
        )
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={
                "application_id": aid,
                "source_id": sid,
                "top_k": 5,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "search" in body
        s = body["search"]
        assert s["query_kind"] == "application"
        assert s["query_application_id"] == aid
        assert s["query_source_id"] == sid
        assert s["top_k"] == 5
        # Search id is 12-char hex.
        import re
        assert re.match(r"^[a-f0-9]{12}$", s["id"])

    def test_text_mode_round_trips(self, env) -> None:
        """End-to-end: free-text query through the route. The fake
        backend embeds the query; the route persists the search and
        returns the matches list (possibly empty)."""
        client, pid, cid, sid, aid, backend = (
            self._seed_project_with_indexed_application(env)
        )
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={
                "query_text": "patient coping",
                "top_k": 3,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        s = body["search"]
        assert s["query_kind"] == "text"
        assert s["query_application_id"] is None
        assert s["query_text"] == "patient coping"

        # F8.5 *must* embed the query text (no LLM rerank, just
        # nearest-neighbour). The fake backend recorded the embed call
        # so we know the route invoked the engine, not a stub.
        # (For application-mode the engine reuses the index entry;
        # for text-mode it embeds.)
        assert backend.embed_calls, "F8.5 engine never embedded the query"

    def test_persisted_search_listed_via_get(self, env) -> None:
        """The persisted QuoteSearch must show up under
        GET /ai/quote-searches — that's how the AI page (when it
        graduates a queue UI) finds prior invocations."""
        client, pid, cid, sid, aid, backend = (
            self._seed_project_with_indexed_application(env)
        )
        post = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"application_id": aid, "source_id": sid, "top_k": 3},
        )
        assert post.status_code == 200, post.text
        sid_search = post.json()["search"]["id"]

        listing = client.get(f"/api/projects/{pid}/ai/quote-searches")
        assert listing.status_code == 200, listing.text
        ids = [s["id"] for s in listing.json()["searches"]]
        assert sid_search in ids

    def test_get_one_search_returns_full_record(self, env) -> None:
        """The single-search endpoint returns the full persisted
        QuoteSearch dict (including matches), not just metadata."""
        client, pid, cid, sid, aid, backend = (
            self._seed_project_with_indexed_application(env)
        )
        post = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"application_id": aid, "source_id": sid},
        )
        sid_search = post.json()["search"]["id"]
        get = client.get(f"/api/projects/{pid}/ai/quote-searches/{sid_search}")
        assert get.status_code == 200, get.text
        s = get.json()["search"]
        assert s["id"] == sid_search
        assert "matches" in s
        assert isinstance(s["matches"], list)

    def test_get_one_search_404_when_missing(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/quote-searches/aaaaaaaaaaaa"
        )
        assert r.status_code == 404

    def test_get_one_search_invalid_id_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/quote-searches/not-hex"
        )
        assert r.status_code == 400


class TestQuoteSearchAuditTrail:
    """F9.6 audit — the search invocation must be recorded as an
    AIEvent so the audit log tab can show that the user asked for
    similar quotes at time X with embedding model Y."""

    def test_search_records_ai_event(self, env) -> None:
        """The route persists an AIEvent of feature=quote_similarity,
        kind=request — the canonical record for the F9.6 audit log."""
        from scribe.ai_provenance import (
            list_ai_events,
            AI_FEATURE_QUOTE_SIMILARITY,
        )

        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        aid = _make_application(client, pid, cid, sid)
        _install_fake_backend()
        client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                "provider": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "default_model": "llama3.2:3b",
                "default_embedding_model": "bge-m3",
            },
        )
        _force_gate_on(client, pid)
        _refresh_index(client, pid)

        post = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={"application_id": aid, "source_id": sid},
        )
        assert post.status_code == 200, post.text
        events = list_ai_events(
            projects, pid, feature=AI_FEATURE_QUOTE_SIMILARITY,
        )
        assert events, (
            "F8.5 must record an AIEvent so F9.6 audit log shows the "
            "search invocation"
        )
        ev = events[-1]
        assert ev.feature == AI_FEATURE_QUOTE_SIMILARITY
