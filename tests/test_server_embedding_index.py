"""F8.2 — wiring the embedding index to the user-facing surface.

The pure data plane (``scribe.embedding_index``) shipped in 954c668
with thorough unit tests in ``tests/test_embedding_index.py``. What was
missing — and what this file proves — is the **user-facing surface**:

* GET  /api/projects/<pid>/ai/embedding-index            — stats
* POST /api/projects/<pid>/ai/embedding-index/refresh    — rebuild
* DELETE /api/projects/<pid>/ai/embedding-index          — clear

Plus the AI page (``project_ai.html``) renders the F8.2 card with a
"Refresh embedding index" button that hits the POST endpoint.

Bypasses the real Ollama daemon by setting
``server._ai_suggest_backend_override`` to a (BackendConfig, FakeBackend)
pair, mirroring the pattern in ``tests/test_server_ai_suggestions.py``.
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
    """Minimal ModelBackend stand-in that records embed() calls and
    returns deterministic vectors."""

    name = PROVIDER_OLLAMA

    def __init__(self, *, dim: int = 4) -> None:
        self.dim = dim
        self.embed_calls: list = []

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        # One distinct unit-ish vector per text so refresh detects no
        # spurious collisions.
        vectors = []
        for i, _t in enumerate(req.inputs):
            v = [0.0] * self.dim
            v[i % self.dim] = 1.0
            vectors.append(tuple(v))
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=req.model,
            provider=self.name,
        )

    def generate(self, cfg, req, *, transport=None):
        # F8.2 refresh never calls generate, but FakeBackend should
        # match the ABC.
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


def _new_project(client: TestClient, name: str = "F8.2 holder") -> str:
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
    to it. Mirrors tests/test_server_queries.py::_attach_transcript."""
    job_id = hashlib.sha256(sid.encode()).hexdigest()[:12]
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({"segments": segments}))

    from scribe import sources as _sources
    src = _sources.load_source(projects_root, pid, sid)
    src.apply_update({"transcript_job_id": job_id})
    _sources.save_source(projects_root, src)


def _two_segment_transcript() -> list[dict]:
    """A minimal speaker-paragraph-friendly transcript so the index has
    real spans to embed.

    Word dicts use the canonical ``text`` key (matches
    :class:`scribe.engine.Word.to_dict`), which is what
    :func:`scribe.application_reanchor.collect_word_texts` reads.
    """
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


# --------------------------------------------------------------------------- #
# Page render: the F8.2 card is on the AI page
# --------------------------------------------------------------------------- #


class TestAIPageRendersF82Card:
    def test_card_marker_present(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-feature="F8.2"' in r.text

    def test_refresh_button_present(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="embed-index-refresh"' in r.text

    def test_clear_button_present(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="embed-index-clear"' in r.text

    def test_stats_row_present(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="embed-index-stats"' in r.text

    def test_page_references_endpoint_paths(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/ai")
        # The JS issues fetches against these paths; if they're missing
        # the buttons can't reach the backend.
        assert "/ai/embedding-index" in r.text
        assert "/ai/embedding-index/refresh" in r.text


# --------------------------------------------------------------------------- #
# GET /ai/embedding-index — stats
# --------------------------------------------------------------------------- #


class TestEmbeddingIndexStats:
    def test_unknown_project_returns_404(self, env) -> None:
        client, _, _ = env
        r = client.get("/api/projects/aaaaaaaaaaaa/ai/embedding-index")
        assert r.status_code == 404

    def test_invalid_project_returns_400(self, env) -> None:
        client, _, _ = env
        r = client.get("/api/projects/!nope!/ai/embedding-index")
        assert r.status_code == 400

    def test_empty_project_returns_zero(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/ai/embedding-index")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 0
        assert body["by_kind"] == {}
        assert body["by_source"] == {}
        assert body["models"] == []
        assert body["last_modified_at"] is None

    def test_includes_configured_embedding_model(self, env) -> None:
        client, projects, _ = env
        pid = _new_project(client)
        # Persist a backend config so the stats endpoint can read it.
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
        r = client.get(f"/api/projects/{pid}/ai/embedding-index")
        assert r.status_code == 200, r.text
        assert r.json()["configured_embedding_model"] == "bge-m3"


# --------------------------------------------------------------------------- #
# POST /ai/embedding-index/refresh
# --------------------------------------------------------------------------- #


class TestEmbeddingIndexRefresh:
    def test_unknown_project_returns_404(self, env) -> None:
        client, _, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/ai/embedding-index/refresh"
        )
        assert r.status_code == 404

    def test_no_embedding_model_returns_400(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        # No backend configured → 400, not a server crash.
        r = client.post(
            f"/api/projects/{pid}/ai/embedding-index/refresh"
        )
        assert r.status_code == 400, r.text

    def test_refresh_embeds_paragraphs_and_applications(self, env) -> None:
        """End-to-end: seed a transcript + an application, refresh the
        index, check the RefreshResult counts and that the embed
        backend was called."""
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        _make_application(
            client, pid, cid, sid,
            anchor_start="s0w0", anchor_end="s0w3",
        )

        backend = _install_fake_backend()
        r = client.post(f"/api/projects/{pid}/ai/embedding-index/refresh")
        assert r.status_code == 200, r.text
        body = r.json()
        # First refresh: everything is "added".
        assert body["added"] >= 1, body
        assert body["updated"] == 0
        assert body["removed"] == 0
        assert body["model"] == "bge-m3"
        # The fake backend was actually invoked.
        assert backend.embed_calls

    def test_idempotent_second_refresh_is_unchanged(self, env) -> None:
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        _make_application(
            client, pid, cid, sid,
            anchor_start="s0w0", anchor_end="s0w3",
        )

        _install_fake_backend()
        r1 = client.post(f"/api/projects/{pid}/ai/embedding-index/refresh")
        assert r1.status_code == 200, r1.text
        added_first = r1.json()["added"]
        assert added_first >= 1

        r2 = client.post(f"/api/projects/{pid}/ai/embedding-index/refresh")
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["added"] == 0
        assert body2["updated"] == 0
        assert body2["removed"] == 0
        assert body2["unchanged"] >= added_first

    def test_stats_reflect_refresh(self, env) -> None:
        """After a refresh, GET /ai/embedding-index reports the entries
        that were just embedded — the round-trip the UI relies on."""
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        _make_application(
            client, pid, cid, sid,
            anchor_start="s0w0", anchor_end="s0w3",
        )

        _install_fake_backend()
        rrefresh = client.post(
            f"/api/projects/{pid}/ai/embedding-index/refresh"
        )
        assert rrefresh.status_code == 200
        added = rrefresh.json()["added"]

        rstats = client.get(f"/api/projects/{pid}/ai/embedding-index")
        assert rstats.status_code == 200
        body = rstats.json()
        assert body["total"] == added
        # Coded segment + at least one paragraph.
        assert "coded_segment" in body["by_kind"]
        # The embedding model that was used appears in models list.
        assert "bge-m3" in body["models"]
        # Per-source bucket includes our source.
        assert body["by_source"].get(sid, 0) >= 1
        # last_modified_at populated to a real ISO-ish string.
        assert isinstance(body["last_modified_at"], str)
        assert body["last_modified_at"]


# --------------------------------------------------------------------------- #
# DELETE /ai/embedding-index — clear
# --------------------------------------------------------------------------- #


class TestEmbeddingIndexClear:
    def test_clear_empty_returns_zero(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.delete(f"/api/projects/{pid}/ai/embedding-index")
        assert r.status_code == 200, r.text
        assert r.json()["removed"] == 0

    def test_clear_removes_persisted_entries(self, env) -> None:
        client, projects, output = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        sid = _new_source(client, pid)
        _attach_transcript(
            output, projects, pid, sid, _two_segment_transcript(),
        )
        _make_application(
            client, pid, cid, sid,
            anchor_start="s0w0", anchor_end="s0w3",
        )

        _install_fake_backend()
        client.post(f"/api/projects/{pid}/ai/embedding-index/refresh")
        # Sanity-check that there's something to clear.
        rstats = client.get(f"/api/projects/{pid}/ai/embedding-index")
        assert rstats.json()["total"] >= 1

        rclear = client.delete(f"/api/projects/{pid}/ai/embedding-index")
        assert rclear.status_code == 200, rclear.text
        assert rclear.json()["removed"] >= 1

        rstats = client.get(f"/api/projects/{pid}/ai/embedding-index")
        assert rstats.json()["total"] == 0

    def test_clear_unknown_project_returns_404(self, env) -> None:
        client, _, _ = env
        r = client.delete("/api/projects/aaaaaaaaaaaa/ai/embedding-index")
        assert r.status_code == 404
