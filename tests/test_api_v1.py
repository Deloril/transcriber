"""Tests for the public ``/api/v1/`` namespace.

We pin:

* every endpoint requires a valid bearer token (401 without one,
  401 on a wrong one, 200 with the right one);
* the discovery endpoint returns a stable shape;
* transcripts list / get / text round-trip a seeded job;
* search returns matching segments only;
* /ask wires through to the existing project_chat plumbing without
  persisting a Conversation to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import api_auth
from scribe import server as srv
from scribe.ai_backend import (
    BackendConfig,
    EmbeddingResponse,
    GenerationResponse,
    PROVIDER_OLLAMA,
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin SCRIBE_HOME + isolate every storage dir to a tmp tree.

    Critical: SCRIBE_HOME points at tmp_path so api_keys.json
    lands somewhere we control. Without this the test would
    write into the developer's real ~/.scribe.
    """
    monkeypatch.setenv(api_auth.ENV_HOME, str(tmp_path))
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "JOBS", {})
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    monkeypatch.setattr(srv, "_ai_suggest_backend_override", None)
    return TestClient(srv.app), tmp_path


def _mint_key(label: str = "test-client") -> str:
    _, plaintext = api_auth.mint_api_key(label)
    return plaintext


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_job(env, *, job_id: str = "abc123def456",
              transcript: dict | None = None) -> str:
    """Drop a finished Job into srv.JOBS for v1 endpoints to read."""
    _, tmp_path = env
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = srv.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / "raw.wav"
    input_path.write_bytes(b"\x00" * 64)
    if transcript is None:
        transcript = {
            "language": "en",
            "mode": "diarize",
            "speakers": ["SPEAKER_00"],
            "speaker_names": {"SPEAKER_00": "Maria"},
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 5.0,
                    "text": "I find it hard to ask for help.",
                    "words": [],
                },
                {
                    "speaker": "SPEAKER_00",
                    "start": 5.0,
                    "end": 10.0,
                    "text": "But sometimes you have to.",
                    "words": [],
                },
            ],
        }
    job = srv.Job(
        id=job_id,
        input_path=input_path,
        output_dir=out_dir,
        mode="diarize",
        speakers=None,
        num_speakers=None,
        language="en",
        model="large-v3",
        created_at="2026-06-04T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result=transcript,
        input_filename="raw.wav",
        display_name="Pilot interview",
        audio_streams=1,
    )
    srv.JOBS[job_id] = job
    return job_id


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


class TestAuth:
    def test_missing_header_401(self, env) -> None:
        client, _ = env
        r = client.get("/api/v1/")
        assert r.status_code == 401

    def test_wrong_scheme_401(self, env) -> None:
        client, _ = env
        _mint_key()
        r = client.get("/api/v1/", headers={"Authorization": "Basic abc"})
        assert r.status_code == 401

    def test_unknown_token_401(self, env) -> None:
        client, _ = env
        _mint_key()
        r = client.get("/api/v1/", headers={"Authorization": "Bearer sk_scribe_nope"})
        assert r.status_code == 401

    def test_valid_token_200(self, env) -> None:
        client, _ = env
        token = _mint_key()
        r = client.get("/api/v1/", headers=_auth(token))
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


class TestDiscovery:
    def test_lists_endpoints(self, env) -> None:
        client, _ = env
        token = _mint_key()
        body = client.get("/api/v1/", headers=_auth(token)).json()
        assert body["version"] == "v1"
        # Sanity: every endpoint shows up by path.
        paths = {e["path"] for e in body["endpoints"]}
        assert "/api/v1/transcripts" in paths
        assert "/api/v1/transcripts/{id}" in paths
        assert "/api/v1/projects" in paths
        assert "/api/v1/search" in paths
        assert "/api/v1/projects/{id}/ask" in paths


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


class TestTranscripts:
    def test_list_returns_seeded_job(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/transcripts", headers=_auth(token),
        ).json()
        assert body["total"] == 1
        row = body["transcripts"][0]
        assert row["id"] == "abc123def456"
        assert row["display_name"] == "Pilot interview"

    def test_list_filters_by_substring(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        # ``q`` matches the display name's "pilot" substring.
        hit = client.get(
            "/api/v1/transcripts?q=pilot", headers=_auth(token),
        ).json()
        assert hit["total"] == 1
        miss = client.get(
            "/api/v1/transcripts?q=zzzzz", headers=_auth(token),
        ).json()
        assert miss["total"] == 0

    def test_get_returns_full_transcript(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/transcripts/abc123def456", headers=_auth(token),
        ).json()
        assert body["id"] == "abc123def456"
        assert len(body["segments"]) == 2
        assert body["speaker_names"] == {"SPEAKER_00": "Maria"}

    def test_get_404(self, env) -> None:
        client, _ = env
        token = _mint_key()
        r = client.get(
            "/api/v1/transcripts/abc123def456", headers=_auth(token),
        )
        assert r.status_code == 404

    def test_text_renders_speakers_and_skips_timestamps_by_default(
        self, env,
    ) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/transcripts/abc123def456/text",
            headers=_auth(token),
        ).json()
        assert "Maria: I find it hard to ask for help." in body["text"]
        # Timestamps off by default.
        assert "[00:00]" not in body["text"]

    def test_text_can_include_timestamps(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/transcripts/abc123def456/text"
            "?include_timestamps=true&include_speakers=false",
            headers=_auth(token),
        ).json()
        assert "[00:00]" in body["text"]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_finds_matching_segment(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/search?q=hard%20to%20ask",
            headers=_auth(token),
        ).json()
        assert body["total"] == 1
        m = body["matches"][0]
        assert m["transcript_id"] == "abc123def456"
        assert m["segment_index"] == 0
        assert m["speaker"] == "Maria"

    def test_no_matches_returns_empty(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_job(env)
        body = client.get(
            "/api/v1/search?q=NOTHINGMATCHES",
            headers=_auth(token),
        ).json()
        assert body["total"] == 0
        assert body["matches"] == []

    def test_q_required(self, env) -> None:
        client, _ = env
        token = _mint_key()
        r = client.get("/api/v1/search?q=", headers=_auth(token))
        # FastAPI's Query(min_length=1) returns a 422.
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #


def _seed_project(client: TestClient, token: str, name: str = "P1") -> str:
    # Use the public projects endpoint to create one — same data
    # path the in-app UI uses.
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestProjects:
    def test_list(self, env) -> None:
        client, _ = env
        token = _mint_key()
        _seed_project(client, token)
        body = client.get("/api/v1/projects", headers=_auth(token)).json()
        assert len(body["projects"]) == 1
        assert body["projects"][0]["name"] == "P1"

    def test_get_one(self, env) -> None:
        client, _ = env
        token = _mint_key()
        pid = _seed_project(client, token)
        body = client.get(
            f"/api/v1/projects/{pid}", headers=_auth(token),
        ).json()
        assert body["id"] == pid
        assert body["sources"] == []
        assert body["code_count"] == 0


# --------------------------------------------------------------------------- #
# /ask — read-only LLM call
# --------------------------------------------------------------------------- #


class _FakeBackend:
    name = PROVIDER_OLLAMA
    def __init__(self, text: str) -> None:
        self.text = text
    def embed(self, cfg, req, *, transport=None):
        return EmbeddingResponse(
            vectors=tuple((1.0,) for _ in req.inputs),
            model=req.model,
            provider=self.name,
        )
    def generate(self, cfg, req, *, transport=None):
        return GenerationResponse(
            text=self.text, model=req.model, provider=self.name,
        )


class TestAsk:
    def _install(self, text: str = "An answer.") -> None:
        srv._ai_suggest_backend_override = (
            BackendConfig(
                provider=PROVIDER_OLLAMA,
                base_url="http://test",
                default_model="llama3.2:3b",
                default_embedding_model="bge-m3",
            ),
            _FakeBackend(text),
        )

    def test_returns_answer(self, env) -> None:
        client, _ = env
        token = _mint_key()
        pid = _seed_project(client, token)
        # Need a source on the project so the chat retrieval has
        # somewhere to look — even an empty embedding index just
        # means the answer gets no citations, not a 400.
        r = client.post(
            f"/api/projects/{pid}/sources",
            json={"name": "S", "source_type": "transcript"},
        )
        assert r.status_code == 201
        self._install("Pushback shows up at S1.")
        body = client.post(
            f"/api/v1/projects/{pid}/ask",
            headers=_auth(token),
            json={"question": "Where do they push back?"},
        ).json()
        assert body["question"] == "Where do they push back?"
        assert "Pushback" in body["answer"]
        assert "model" in body
        assert isinstance(body["citations"], list)

    def test_does_not_persist_a_conversation(self, env) -> None:
        """Read-only contract: the chats dir stays empty."""
        client, _ = env
        token = _mint_key()
        pid = _seed_project(client, token)
        client.post(
            f"/api/projects/{pid}/sources",
            json={"name": "S", "source_type": "transcript"},
        )
        self._install()
        client.post(
            f"/api/v1/projects/{pid}/ask",
            headers=_auth(token),
            json={"question": "x"},
        )
        chats_dir = srv.PROJECTS_DIR / pid / "chats"
        # If a conversation was persisted, the chats dir would exist
        # and contain a json file. Either it doesn't exist at all,
        # or it's empty.
        if chats_dir.is_dir():
            assert list(chats_dir.glob("*.json")) == []

    def test_400_on_missing_question(self, env) -> None:
        client, _ = env
        token = _mint_key()
        pid = _seed_project(client, token)
        self._install()
        r = client.post(
            f"/api/v1/projects/{pid}/ask",
            headers=_auth(token),
            json={},
        )
        assert r.status_code == 400

    def test_404_on_unknown_project(self, env) -> None:
        client, _ = env
        token = _mint_key()
        self._install()
        r = client.post(
            "/api/v1/projects/aaaaaaaaaaaa/ask",
            headers=_auth(token),
            json={"question": "x"},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Settings: /api/settings/api_keys (the management UI's backend)
# --------------------------------------------------------------------------- #


class TestSettingsApiKeysManagement:
    def test_list_initially_empty(self, env) -> None:
        client, _ = env
        body = client.get("/api/settings/api_keys").json()
        assert body == {"keys": []}

    def test_mint_returns_plaintext_once(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/settings/api_keys",
            json={"label": "claude-mcp"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["plaintext"].startswith(api_auth.TOKEN_PREFIX)
        # The list endpoint never echoes the plaintext or hash back.
        listing = client.get("/api/settings/api_keys").json()
        assert len(listing["keys"]) == 1
        assert "plaintext" not in listing["keys"][0]
        assert "hash" not in listing["keys"][0]

    def test_revoke_removes_key(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/settings/api_keys",
            json={"label": "doomed"},
        )
        kid = r.json()["key"]["id"]
        r2 = client.delete(f"/api/settings/api_keys/{kid}")
        assert r2.status_code == 200
        assert client.get("/api/settings/api_keys").json()["keys"] == []

    def test_mint_400_without_label(self, env) -> None:
        client, _ = env
        r = client.post("/api/settings/api_keys", json={})
        assert r.status_code == 400

    def test_revoke_404_on_unknown(self, env) -> None:
        client, _ = env
        r = client.delete("/api/settings/api_keys/key-deadbeef00")
        assert r.status_code == 404


class TestSettingsRendersApiKeysCard:
    def test_card_present(self, env) -> None:
        client, _ = env
        body = client.get("/settings").text
        assert 'data-test-id="settings-api-keys-card"' in body
        assert 'id="apiKeyMintBtn"' in body
        assert "/api/settings/api_keys" in body
        assert "/api/v1/" in body
