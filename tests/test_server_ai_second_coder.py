"""F8.7 — wiring the AI second-coder pass to the user-facing surface.

The pure data plane (``scribe.ai_second_coder``) shipped in a3be790
with thorough unit tests in ``tests/test_ai_second_coder.py``. What
was missing — and what this file proves — is the user-facing surface:

  * POST /api/projects/<pid>/sources/<sid>/second-coder            — start
  * POST /api/projects/<pid>/second-coder-passes/<rpid>/run        — drive
  * POST /api/projects/<pid>/second-coder-passes/<rpid>/cancel     — abandon
  * GET  /api/projects/<pid>/second-coder-passes                   — list
  * GET  /api/projects/<pid>/second-coder-passes/<rpid>            — fetch
  * GET  /api/projects/<pid>/second-coder-passes/<rpid>/diff       — diff

Plus the project AI page graduates the F8.7 stub into a real card with
a "Start second-coder pass" form, a lock-state row, and a passes list
that surfaces per-pass kappa + an inline diff disclosure.

We bypass the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair — the same shim the F8.3 / F8.5 / F8.6 server tests use.
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
    """Same shape as tests/test_server_transcript_review.py::FakeBackend.

    Deterministic per-text basis vectors so the F8.2 index entries and
    F8.6's per-span query share the same model space; ``generate``
    returns ``[]`` so the LLM rerank is a no-op and the engine falls
    back to embedding-only ranking.
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
# Project / source / coder / code / app / lock helpers
# --------------------------------------------------------------------------- #


def _new_project(client: TestClient, name: str = "F8.7 holder") -> str:
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


def _new_coder(client: TestClient, pid: str, name: str = "Alex") -> str:
    r = client.post(f"/api/projects/{pid}/coders", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _attach_transcript(
    output_dir: Path,
    projects_root: Path,
    pid: str,
    sid: str,
    segments: list[dict],
) -> None:
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


def _lock_codebook(client: TestClient, pid: str) -> None:
    r = client.post(
        f"/api/projects/{pid}/codebook/lock",
        json={"reason": "Locking for ICR"},
    )
    assert r.status_code == 200, r.text


def _seed_locked_project_with_transcript(env, *, with_coder: bool = True):
    """Standard happy-path seed: project + locked codebook + 1 source +
    transcript + 1 code + (optional) human coder + AI backend + open gate."""
    client, projects, output = env
    pid = _new_project(client)
    cid = _new_code(client, pid)
    sid = _new_source(client, pid)
    _attach_transcript(output, projects, pid, sid, _multi_paragraph_transcript())
    coder_id = _new_coder(client, pid) if with_coder else ""
    backend = _install_fake_backend()
    _configure_backend(client, pid)
    _force_gate_on(client, pid)
    _lock_codebook(client, pid)
    return client, pid, cid, sid, coder_id, backend, projects


# --------------------------------------------------------------------------- #
# A. UI render — the AI page must surface F8.7
# --------------------------------------------------------------------------- #


class TestProjectAiPageF87Card:
    def test_project_ai_page_renders_second_coder_card(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200, r.text
        body = r.text
        # Card anchor + test markers.
        assert 'id="aiSecondCoderCard"' in body
        assert 'data-test-feature="F8.7"' in body
        assert 'data-test-id="ai-second-coder-card"' in body

    def test_card_has_form_elements(self, env) -> None:
        """Source + coder + granularity selectors and the start button."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        body = r.text
        for marker in (
            'data-test-id="ai-second-coder-source"',
            'data-test-id="ai-second-coder-coder"',
            'data-test-id="ai-second-coder-granularity"',
            'data-test-id="ai-second-coder-top-n"',
            'data-test-id="ai-second-coder-start"',
            'data-test-id="ai-second-coder-lock-row"',
            'data-test-id="ai-second-coder-passes-list"',
        ):
            assert marker in body, marker

    def test_card_advertises_route(self, env) -> None:
        """The route URL the JS hits must be in the rendered HTML."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        # The JS template literal collapses the source-id placeholder
        # in a JS template string; what matters is the path stem.
        assert "/second-coder" in r.text

    def test_suggestion_surfaces_lists_f87_anchor(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-surface-f8-7"' in r.text


# --------------------------------------------------------------------------- #
# B. POST /sources/<sid>/second-coder — validation
# --------------------------------------------------------------------------- #


class TestStartSecondCoderValidation:
    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/!nope!/sources/aaaaaaaaaaaa/second-coder",
            json={"human_coder_id": "aaaaaaaaaaaa"},
        )
        assert r.status_code == 400

    def test_invalid_source_id_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/sources/!!!!/second-coder",
            json={"human_coder_id": "aaaaaaaaaaaa"},
        )
        assert r.status_code == 400

    def test_missing_human_coder_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={},
        )
        assert r.status_code == 400
        assert "human_coder_id" in r.text

    def test_invalid_human_coder_format_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": "not-hex"},
        )
        assert r.status_code == 400

    def test_unknown_human_coder_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        _force_gate_on(client, pid)
        # Well-formed but non-existent.
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": "ffffffffffff"},
        )
        assert r.status_code == 404, r.text

    def test_invalid_granularity_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        cid = _new_coder(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": cid, "granularity": "chapter"},
        )
        assert r.status_code == 400

    def test_invalid_top_n_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        cid = _new_coder(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": cid, "top_n": 0},
        )
        assert r.status_code == 400

    def test_invalid_max_steps_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        cid = _new_coder(client, pid)
        _force_gate_on(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": cid, "max_steps": 9999},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# C. Gate + lock guards
# --------------------------------------------------------------------------- #


class TestStartSecondCoderGate:
    def test_gate_blocks_when_not_satisfied(self, env) -> None:
        """Empty project trips the F8.10 default thresholds (≥ 8 codes,
        ≥ 2 hand-coded transcripts). Must surface as 412 + structured
        gate body so the UI can act."""
        client, _, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        cid = _new_coder(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": cid},
        )
        assert r.status_code == 412, r.text
        body = r.json()["detail"]
        assert body["detail"] == "AI gate not satisfied"
        assert body["gate"]["allowed"] is False


class TestStartSecondCoderLockGuard:
    def test_unlocked_codebook_returns_409(self, env) -> None:
        """The whole methodological point of F8.7: refuse on an
        evolving codebook. The server should map
        CodebookNotLockedError to 409 + a stable ``reason`` marker
        so the UI can render the lock-required notice."""
        client, projects, output = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        cid = _new_coder(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _multi_paragraph_transcript(),
        )
        _install_fake_backend()
        _configure_backend(client, pid)
        _force_gate_on(client, pid)
        # Note: codebook NOT locked.
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": cid},
        )
        assert r.status_code == 409, r.text
        body = r.json()["detail"]
        assert isinstance(body, dict), body
        assert body.get("reason") == "codebook_not_locked"


# --------------------------------------------------------------------------- #
# D. Happy path: start, list, fetch, diff, run, cancel
# --------------------------------------------------------------------------- #


class TestSecondCoderHappyPath:
    def test_post_starts_pass_and_returns_record(self, env) -> None:
        client, pid, cid_code, sid, coder_id, backend, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        assert r.status_code == 200, r.text
        p = r.json()["pass"]
        assert p["project_id"] == pid
        assert p["source_id"] == sid
        assert p["human_coder_id"] == coder_id
        assert p["granularity"] == "paragraph"
        assert re.match(r"^[a-f0-9]{12}$", p["id"])
        assert re.match(r"^[a-f0-9]{12}$", p["review_pass_id"])
        # Two paragraphs in the seed → two items inside the inner
        # review pass; max_steps=5 ≥ 2 → completed.
        assert p["status"] == "completed", p
        # ICR populated once the inner pass completes.
        assert "icr_results" in p
        assert p["icr_results"].get("n_items") is not None

    def test_pass_appears_in_listing_and_fetch(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        pass_id = r.json()["pass"]["id"]

        # List returns it.
        r2 = client.get(f"/api/projects/{pid}/second-coder-passes")
        assert r2.status_code == 200
        passes = r2.json()["passes"]
        assert any(p["id"] == pass_id for p in passes)

        # Single fetch returns it.
        r3 = client.get(
            f"/api/projects/{pid}/second-coder-passes/{pass_id}"
        )
        assert r3.status_code == 200
        assert r3.json()["pass"]["id"] == pass_id

    def test_list_filters(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        # Filter by source_id (matches) returns it.
        r = client.get(
            f"/api/projects/{pid}/second-coder-passes?source_id={sid}"
        )
        assert r.status_code == 200
        assert len(r.json()["passes"]) == 1
        # Filter by status=pending (no match — pass completed) returns
        # an empty list.
        r2 = client.get(
            f"/api/projects/{pid}/second-coder-passes?status=pending"
        )
        assert r2.status_code == 200
        assert r2.json()["passes"] == []

    def test_diff_endpoint_returns_diff_and_icr(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        pass_id = r.json()["pass"]["id"]

        d = client.get(
            f"/api/projects/{pid}/second-coder-passes/{pass_id}/diff"
        )
        assert d.status_code == 200, d.text
        body = d.json()
        assert "pass" in body
        assert "diff" in body
        assert "icr" in body
        # The ICR shape is the contract the UI consumes.
        for k in (
            "n_items", "n_codes", "overall_kappa",
            "overall_interpretation", "per_code",
        ):
            assert k in body["icr"], k

    def test_max_steps_zero_starts_but_does_not_drive(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 0},
        )
        assert r.status_code == 200
        p = r.json()["pass"]
        assert p["status"] == "pending"

    def test_partial_run_then_resume(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 1},
        )
        assert r.status_code == 200
        p = r.json()["pass"]
        # Inner review pass has 2 items; max_steps=1 leaves it running.
        assert p["status"] in ("running", "completed")
        pass_id = p["id"]
        if p["status"] == "running":
            r2 = client.post(
                f"/api/projects/{pid}/second-coder-passes/{pass_id}/run",
                json={"max_steps": 5},
            )
            assert r2.status_code == 200
            assert r2.json()["pass"]["status"] == "completed"

    def test_cancel_pass(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        # Start without driving so the pass is left pending.
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 0},
        )
        pass_id = r.json()["pass"]["id"]
        r2 = client.post(
            f"/api/projects/{pid}/second-coder-passes/{pass_id}/cancel"
        )
        assert r2.status_code == 200
        assert r2.json()["pass"]["status"] == "cancelled"

    def test_cancel_terminal_pass_returns_409_for_completed(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        pass_id = r.json()["pass"]["id"]
        # Cancelling a completed pass: 409.
        r2 = client.post(
            f"/api/projects/{pid}/second-coder-passes/{pass_id}/cancel"
        )
        assert r2.status_code == 409


# --------------------------------------------------------------------------- #
# E. Fetch / diff edge cases
# --------------------------------------------------------------------------- #


class TestSecondCoderFetchEdgeCases:
    def test_invalid_pass_id_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/second-coder-passes/!!!"
        )
        assert r.status_code == 400

    def test_unknown_pass_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/second-coder-passes/aaaaaaaaaaaa"
        )
        assert r.status_code == 404

    def test_diff_unknown_pass_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/second-coder-passes/aaaaaaaaaaaa/diff"
        )
        assert r.status_code == 404

    def test_run_unknown_pass_returns_404(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/second-coder-passes/aaaaaaaaaaaa/run",
            json={},
        )
        # 404 when the project exists, but pass record is gone.
        assert r.status_code in (404,)

    def test_run_terminal_pass_returns_409(self, env) -> None:
        client, pid, _, sid, coder_id, _, _ = (
            _seed_locked_project_with_transcript(env)
        )
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/second-coder",
            json={"human_coder_id": coder_id, "max_steps": 5},
        )
        pass_id = r.json()["pass"]["id"]
        # Pass is now completed; /run should refuse.
        r2 = client.post(
            f"/api/projects/{pid}/second-coder-passes/{pass_id}/run",
            json={"max_steps": 5},
        )
        assert r2.status_code == 409
