"""F8.6 — wiring the whole-transcript AI review pass to the user-facing surface.

The pure data plane (``scribe.transcript_review``) shipped in 167b8c6
with thorough unit tests in ``tests/test_transcript_review.py``. What
was missing — and what this file proves — is the user-facing surface:

  * POST /api/projects/<pid>/sources/<sid>/review            — start a pass
  * POST /api/projects/<pid>/review-passes/<rpid>/run        — drive forward
  * POST /api/projects/<pid>/review-passes/<rpid>/cancel     — abandon
  * GET  /api/projects/<pid>/review-passes                   — list
  * GET  /api/projects/<pid>/review-passes/<rpid>            — fetch one

Plus the source-coding view's page-actions bar carries a "✨ Review
whole transcript" button (data-test-feature=F8.6) that opens a modal.
The project AI page's "Suggestion surfaces" card carries an F8.6
discovery anchor.

We bypass the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair, mirroring the pattern in ``tests/test_server_ai_suggestions.py``,
``tests/test_server_embedding_index.py`` and
``tests/test_server_quote_similarity.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
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

    Mirrors the shape used by tests/test_server_quote_similarity.py:
    ``embed`` returns deterministic per-text basis vectors so the
    F8.2 index entries and F8.6's per-span query share the same model
    space (the FakeBackend services both). ``generate`` returns an
    empty JSON array so the F8.3 LLM rerank is a no-op — the engine
    falls back to embedding-only ranking.
    """

    name = PROVIDER_OLLAMA

    def __init__(self, *, dim: int = 4) -> None:
        self.dim = dim
        self.embed_calls: list = []
        self.generate_calls: list = []

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
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
        self.generate_calls.append((cfg, req))
        # Empty array triggers the embedding-only fallback in
        # code_suggestions.suggest_codes_for_span.
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


def _new_project(client: TestClient, name: str = "F8.6 holder") -> str:
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
    """Plant edited.json under outputs/<job_id>/ and link it to the source."""
    job_id = hashlib.sha256(sid.encode()).hexdigest()[:12]
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({"segments": segments}))
    from scribe import sources as _sources
    src = _sources.load_source(projects_root, pid, sid)
    src.apply_update({"transcript_job_id": job_id})
    _sources.save_source(projects_root, src)


def _multi_paragraph_transcript() -> list[dict]:
    """Two speaker turns → two paragraphs → two review items."""
    return [
        {
            "id": 0, "speaker": "I", "start": 0.0, "end": 1.5,
            "text": "tell me about coping with pain",
            "words": [
                {"text": "tell",   "start": 0.0, "end": 0.2, "speaker": "I"},
                {"text": "me",     "start": 0.2, "end": 0.4, "speaker": "I"},
                {"text": "about",  "start": 0.4, "end": 0.6, "speaker": "I"},
                {"text": "coping", "start": 0.6, "end": 1.0, "speaker": "I"},
                {"text": "with",   "start": 1.0, "end": 1.2, "speaker": "I"},
                {"text": "pain",   "start": 1.2, "end": 1.5, "speaker": "I"},
            ],
        },
        {
            "id": 1, "speaker": "P", "start": 1.5, "end": 3.0,
            "text": "I just keep moving the next day",
            "words": [
                {"text": "I",      "start": 1.5, "end": 1.6, "speaker": "P"},
                {"text": "just",   "start": 1.6, "end": 1.8, "speaker": "P"},
                {"text": "keep",   "start": 1.8, "end": 2.0, "speaker": "P"},
                {"text": "moving", "start": 2.0, "end": 2.3, "speaker": "P"},
                {"text": "the",    "start": 2.3, "end": 2.5, "speaker": "P"},
                {"text": "next",   "start": 2.5, "end": 2.7, "speaker": "P"},
                {"text": "day",    "start": 2.7, "end": 3.0, "speaker": "P"},
            ],
        },
    ]


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


# --------------------------------------------------------------------------- #
# A. UI render — the "Review whole transcript" button + modal must render
# --------------------------------------------------------------------------- #


class TestSourceCodingViewReviewUI:
    def test_page_actions_have_review_button(self, env) -> None:
        """The page-actions bar in source_coding.html must include a
        '✨ Review whole transcript' button. Without this the F8.6 engine
        is unreachable from the UI."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        body = r.text
        assert 'id="reviewPassBtn"' in body
        assert 'data-test-feature="F8.6"' in body
        assert 'data-test-id="src-review-pass"' in body
        assert "Review whole transcript" in body

    def test_review_modal_renders(self, env) -> None:
        """The modal that hosts the pass progress + suggestion list
        must be in the DOM with a stable test marker so the integration
        test can scope-pick it."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert 'id="reviewModal"' in body
        assert 'data-test-id="src-review-modal"' in body
        # Action / progress slots the JS populates.
        assert 'data-test-id="src-review-start"' in body
        assert 'data-test-id="src-review-status"' in body
        assert 'data-test-id="src-review-progress"' in body
        assert 'data-test-id="src-review-items"' in body
        assert 'data-test-id="src-review-granularity"' in body

    def test_review_endpoint_url_present(self, env) -> None:
        """The fetch URL the start-pass JS hits must be in the rendered
        HTML — if it drifts the click becomes a 404 no-op."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        # The JS template literal /sources/<sid>/review collapses to
        # /sources/${...}/review in the rendered text.
        assert "/review" in r.text


class TestProjectAiPageAdvertisesF86:
    def test_ai_page_lists_f8_6_surface(self, env) -> None:
        """The AI dashboard's Suggestion-surfaces card points researchers
        to the source-coding view's review-pass button. Without this a
        user landing on /projects/<pid>/ai has no way to discover F8.6."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        body = r.text
        assert "F8.6" in body
        assert "Review whole transcript" in body
        assert 'data-test-id="ai-surface-f8-6"' in body


# --------------------------------------------------------------------------- #
# B. POST /sources/<sid>/review — gate, validation, happy path
# --------------------------------------------------------------------------- #


class TestStartReviewValidation:
    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/!nope!/sources/aaaaaaaaaaaa/review",
            json={},
        )
        assert r.status_code == 400

    def test_unknown_project_returns_404(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/sources/bbbbbbbbbbbb/review",
            json={},
        )
        # Either 404 (not found) or 412 (gate evaluator complained
        # about the missing project) — both refuse to start a pass.
        assert r.status_code in (404, 400, 412)

    def test_unknown_source_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/bbbbbbbbbbbb/review",
            json={},
        )
        # When the gate is open we expect to reach the source-load
        # check and get a 404.
        assert r.status_code == 404, r.text

    def test_invalid_granularity_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"granularity": "chapter"},
        )
        assert r.status_code == 400

    def test_invalid_top_k_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"top_k": 0},
        )
        assert r.status_code == 400

    def test_invalid_max_steps_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 9999},
        )
        assert r.status_code == 400


class TestStartReviewGate:
    def test_gate_blocks_when_not_satisfied(self, env) -> None:
        """Default thresholds (≥ 8 codes, ≥ 2 hand-coded transcripts)
        aren't met on an empty project; F8.10 must close F8.6 with 412
        + a structured gate body."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review", json={},
        )
        assert r.status_code == 412, r.text
        body = r.json()["detail"]
        assert body["detail"] == "AI gate not satisfied"
        assert body["gate"]["allowed"] is False


class TestStartReviewHappyPath:
    def _seed(self, env):
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        backend = _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        return client, pid, cid, sid, backend, projects

    def test_post_starts_pass_and_returns_record(self, env) -> None:
        """POSTing /review enumerates items, persists a ReviewPass, and
        drives forward by max_steps; the response carries the pass dict
        with items + status."""
        client, pid, cid, sid, backend, projects = self._seed(env)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        p = body["pass"]
        assert p["project_id"] == pid
        assert p["source_id"] == sid
        assert p["granularity"] == "paragraph"
        assert re.match(r"^[a-f0-9]{12}$", p["id"])
        # Two paragraphs in the seed → two items.
        assert len(p["items"]) == 2
        # max_steps=5 ≥ 2 items → completed.
        assert p["status"] == "completed", p
        # Suggestions persisted: each item with no error has a
        # suggestion_id.
        for it in p["items"]:
            if not it.get("error"):
                assert it["suggestion_id"], it

    def test_max_steps_zero_starts_but_does_not_drive(self, env) -> None:
        """max_steps=0 lets the client kick off the pass and drive it
        itself via /run — useful for very long transcripts where the
        first request shouldn't block."""
        client, pid, cid, sid, backend, projects = self._seed(env)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 0},
        )
        assert r.status_code == 200, r.text
        p = r.json()["pass"]
        assert p["status"] == "pending"
        for it in p["items"]:
            assert not it["suggestion_id"]
            assert not it.get("error")

    def test_partial_run_then_resume(self, env) -> None:
        """max_steps=1 leaves the pass running with one item still
        pending; a second /run drives it to completion."""
        client, pid, cid, sid, backend, projects = self._seed(env)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 1},
        )
        assert r.status_code == 200, r.text
        p = r.json()["pass"]
        assert p["status"] == "running"
        processed = sum(1 for it in p["items"] if it.get("suggestion_id") or it.get("error"))
        assert processed == 1

        r2 = client.post(
            f"/api/projects/{pid}/review-passes/{p['id']}/run",
            json={"max_steps": 5},
        )
        assert r2.status_code == 200, r2.text
        p2 = r2.json()["pass"]
        assert p2["status"] == "completed"

    def test_persisted_pass_listed_via_get(self, env) -> None:
        """The persisted ReviewPass shows up under GET /review-passes —
        the AI page's queue UI reads from this listing."""
        client, pid, cid, sid, backend, projects = self._seed(env)
        post = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 0},
        )
        assert post.status_code == 200
        rid = post.json()["pass"]["id"]

        listing = client.get(f"/api/projects/{pid}/review-passes")
        assert listing.status_code == 200, listing.text
        ids = [p["id"] for p in listing.json()["passes"]]
        assert rid in ids

        # Optional source filter.
        scoped = client.get(
            f"/api/projects/{pid}/review-passes?source_id={sid}",
        )
        assert scoped.status_code == 200
        ids = [p["id"] for p in scoped.json()["passes"]]
        assert rid in ids

    def test_get_one_pass_returns_full_record(self, env) -> None:
        client, pid, cid, sid, backend, projects = self._seed(env)
        post = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 0},
        )
        rid = post.json()["pass"]["id"]
        r = client.get(f"/api/projects/{pid}/review-passes/{rid}")
        assert r.status_code == 200, r.text
        p = r.json()["pass"]
        assert p["id"] == rid
        assert "items" in p

    def test_get_one_pass_404_when_missing(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/review-passes/aaaaaaaaaaaa",
        )
        assert r.status_code == 404

    def test_get_one_pass_invalid_id_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/review-passes/not-hex",
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# C. POST /run — terminal-state guard
# --------------------------------------------------------------------------- #


class TestRunTerminalStateGuard:
    def test_run_on_completed_pass_returns_409(self, env) -> None:
        """Once a pass reaches a terminal state the /run route must
        refuse to step it further — that would corrupt the audit
        trail."""
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 5},
        )
        rid = r.json()["pass"]["id"]
        assert r.json()["pass"]["status"] == "completed"

        # /run should refuse.
        again = client.post(
            f"/api/projects/{pid}/review-passes/{rid}/run",
            json={"max_steps": 5},
        )
        assert again.status_code == 409


# --------------------------------------------------------------------------- #
# D. POST /cancel — happy path + idempotency + terminal guard
# --------------------------------------------------------------------------- #


class TestCancelReviewPass:
    def test_cancel_running_pass(self, env) -> None:
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        # Start with max_steps=0 → pending.
        post = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 0},
        )
        rid = post.json()["pass"]["id"]
        cancel = client.post(
            f"/api/projects/{pid}/review-passes/{rid}/cancel",
        )
        assert cancel.status_code == 200, cancel.text
        p = cancel.json()["pass"]
        assert p["status"] == "cancelled"

        # Cancel-after-cancel is idempotent (returns 200 with same state).
        again = client.post(
            f"/api/projects/{pid}/review-passes/{rid}/cancel",
        )
        assert again.status_code == 200

    def test_cancel_completed_pass_returns_409(self, env) -> None:
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        post = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 5},
        )
        rid = post.json()["pass"]["id"]
        assert post.json()["pass"]["status"] == "completed"
        cancel = client.post(
            f"/api/projects/{pid}/review-passes/{rid}/cancel",
        )
        assert cancel.status_code == 409


# --------------------------------------------------------------------------- #
# E. Audit trail — F9.6 records the pass start as an AIEvent
# --------------------------------------------------------------------------- #


class TestReviewAuditTrail:
    def test_pass_start_records_ai_event(self, env) -> None:
        """The route persists an AIEvent of feature=transcript_review,
        kind=request — the canonical record for the F9.6 audit log."""
        from scribe.ai_provenance import (
            list_ai_events,
            AI_FEATURE_TRANSCRIPT_REVIEW,
        )

        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/review",
            json={"max_steps": 0},
        )
        assert r.status_code == 200, r.text
        events = list_ai_events(
            projects, pid, feature=AI_FEATURE_TRANSCRIPT_REVIEW,
        )
        assert events, (
            "F8.6 must record an AIEvent so F9.6 audit log shows the "
            "pass invocation"
        )
        ev = events[-1]
        assert ev.feature == AI_FEATURE_TRANSCRIPT_REVIEW
