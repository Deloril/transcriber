"""Tests for the project-chat HTTP surface.

The pure data + prompt module (:mod:`scribe.project_chat`) has its own
tests; this file exercises FastAPI wrapping:

* The five routes are reachable and validate inputs.
* Conversation CRUD round-trips through disk.
* POST .../turn calls the embedding index + LLM, persists both turns,
  and surfaces structured errors when the backend isn't configured.
* The F8.10 gate is intentionally NOT applied (chat is exploration,
  not coding).

We bypass the real Ollama daemon via the same
``server._ai_suggest_backend_override`` hook the suggestion tests
already use.
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


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir(); output.mkdir(); projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    monkeypatch.setattr(srv, "_ai_suggest_backend_override", None)
    return TestClient(srv.app), projects


class FakeBackend:
    name = PROVIDER_OLLAMA

    def __init__(
        self,
        *,
        embed_vector: tuple[float, ...] = (1.0, 0.0, 0.0),
        generation_text: str = "An answer with a citation [S1].",
    ) -> None:
        self.embed_vector = embed_vector
        self.generation_text = generation_text
        self.embed_calls: list = []
        self.generate_calls: list = []

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        return EmbeddingResponse(
            vectors=tuple(self.embed_vector for _ in req.inputs),
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


def _install_fake(text: str = "Answer.") -> FakeBackend:
    backend = FakeBackend(generation_text=text)
    srv._ai_suggest_backend_override = (
        BackendConfig(
            provider=PROVIDER_OLLAMA,
            base_url="http://test",
            default_model="llama3.2:3b",
            default_embedding_model="bge-m3",
        ),
        backend,
    )
    return backend


def _new_project(client: TestClient, name: str = "Chat test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_source(client: TestClient, pid: str, name: str = "Source") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# CRUD: list / create / get / delete
# --------------------------------------------------------------------------- #


class TestCreate:
    def test_create_persists_and_returns_201(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/chats",
            json={"source_ids": [sid], "title": "Pilot"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["source_ids"] == [sid]
        assert body["title"] == "Pilot"
        assert body["messages"] == []

    def test_400_on_missing_source_ids(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/chats", json={"title": "x"})
        assert r.status_code == 400

    def test_400_on_unknown_source_id(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        # Source ID looks valid (12-char hex) but doesn't exist.
        r = client.post(
            f"/api/projects/{pid}/chats",
            json={"source_ids": ["a" * 12]},
        )
        assert r.status_code == 400

    def test_404_for_unknown_project(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/chats",
            json={"source_ids": ["b" * 12]},
        )
        assert r.status_code == 404

    def test_title_optional(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/chats", json={"source_ids": [sid]},
        )
        assert r.status_code == 201
        # Empty title is ok at creation; first user turn will
        # populate it via derive_title_from_first_question.
        assert r.json()["title"] == ""


class TestList:
    def test_empty_when_none_exist(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/chats")
        assert r.status_code == 200
        assert r.json()["conversations"] == []

    def test_list_omits_messages(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        client.post(
            f"/api/projects/{pid}/chats",
            json={"source_ids": [sid], "title": "Pilot"},
        )
        r = client.get(f"/api/projects/{pid}/chats")
        rows = r.json()["conversations"]
        assert len(rows) == 1
        assert "messages" not in rows[0]
        assert rows[0]["message_count"] == 0


class TestGet:
    def test_get_returns_full_thread(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        cid = client.post(
            f"/api/projects/{pid}/chats", json={"source_ids": [sid]},
        ).json()["id"]
        r = client.get(f"/api/projects/{pid}/chats/{cid}")
        assert r.status_code == 200
        assert r.json()["id"] == cid
        assert "messages" in r.json()

    def test_404_for_unknown_conversation(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/chats/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_400_for_invalid_id_shape(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/chats/not-hex")
        assert r.status_code == 400


class TestDelete:
    def test_deletes_and_404s_after(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        cid = client.post(
            f"/api/projects/{pid}/chats", json={"source_ids": [sid]},
        ).json()["id"]
        r = client.delete(f"/api/projects/{pid}/chats/{cid}")
        assert r.status_code == 200
        r2 = client.get(f"/api/projects/{pid}/chats/{cid}")
        assert r2.status_code == 404

    def test_404_for_unknown(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.delete(f"/api/projects/{pid}/chats/aaaaaaaaaaaa")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Turn — the actual chat round-trip
# --------------------------------------------------------------------------- #


class TestTurn:
    def _setup(self, client: TestClient) -> tuple[str, str, str]:
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        cid = client.post(
            f"/api/projects/{pid}/chats",
            json={"source_ids": [sid], "title": "T"},
        ).json()["id"]
        return pid, sid, cid

    def test_round_trip_persists_user_and_assistant(self, env) -> None:
        client, _ = env
        _install_fake(text="Pushback shows up in [S1].")
        pid, sid, cid = self._setup(client)
        r = client.post(
            f"/api/projects/{pid}/chats/{cid}/turn",
            json={"text": "Where do they push back?"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["messages"]) == 2
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant"]
        assert "push back" in body["messages"][0]["content"]
        # Empty source_ids in the conversation OR no embeddings on
        # disk → no citations. That's expected here.
        assert body["messages"][1]["citations"] == []

    def test_persisted_thread_survives_get(self, env) -> None:
        client, _ = env
        _install_fake()
        pid, sid, cid = self._setup(client)
        client.post(
            f"/api/projects/{pid}/chats/{cid}/turn",
            json={"text": "First question."},
        )
        # Re-GET the conversation — should have the same messages.
        r = client.get(f"/api/projects/{pid}/chats/{cid}")
        assert r.status_code == 200
        assert len(r.json()["messages"]) == 2

    def test_400_on_empty_text(self, env) -> None:
        client, _ = env
        _install_fake()
        pid, sid, cid = self._setup(client)
        for body in ({"text": ""}, {"text": "   "}, {}):
            r = client.post(
                f"/api/projects/{pid}/chats/{cid}/turn", json=body,
            )
            assert r.status_code == 400

    def test_400_on_missing_default_model(self, env) -> None:
        client, _ = env
        srv._ai_suggest_backend_override = (
            BackendConfig(
                provider=PROVIDER_OLLAMA,
                base_url="http://test",
                default_model="",   # not configured
            ),
            FakeBackend(),
        )
        pid, sid, cid = self._setup(client)
        r = client.post(
            f"/api/projects/{pid}/chats/{cid}/turn",
            json={"text": "anything"},
        )
        assert r.status_code == 400

    def test_404_for_unknown_conversation(self, env) -> None:
        client, _ = env
        _install_fake()
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/chats/aaaaaaaaaaaa/turn",
            json={"text": "hi"},
        )
        assert r.status_code == 404

    def test_400_for_invalid_conversation_id(self, env) -> None:
        client, _ = env
        _install_fake()
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/chats/not-hex/turn",
            json={"text": "hi"},
        )
        assert r.status_code == 400

    def test_does_not_pass_through_ai_gate(self, env) -> None:
        """F8.10 should not apply: a user can chat with their data on a
        fresh project that has no codes / no hand-coded transcripts.
        Asking 'what surprised you?' isn't coding."""
        client, _ = env
        _install_fake(text="Nothing yet.")
        pid, sid, cid = self._setup(client)
        # Project has zero codes and zero hand-coded transcripts —
        # the gate would reject AI suggestions outright. Chat is fine.
        r = client.post(
            f"/api/projects/{pid}/chats/{cid}/turn",
            json={"text": "What stood out?"},
        )
        assert r.status_code == 200, r.text

    def test_chat_page_renders(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/chat")
        assert r.status_code == 200
        body = r.text
        # Real working page (not the wireframe-stub template). The
        # wireframe-stub template renders <div class="stub"><strong>Wireframe.</strong>
        # at the top; its absence is our proof we're on the real template.
        assert "Chat with your data" in body
        assert "<strong>Wireframe.</strong>" not in body
        # JS hooks the page exercises.
        assert "/api/projects/" in body and "/chats" in body
        assert 'id="convList"' in body
        assert 'id="newBtn"' in body
        # Build-index affordance lives on the source picker so a
        # fresh project can populate the index without leaving the page.
        assert "embedding-index/refresh" in body
        assert 'id="buildBtn"' in body

    def test_first_question_populates_title(self, env) -> None:
        client, _ = env
        _install_fake()
        pid = _new_project(client)
        sid = _seed_source(client, pid)
        # Create with empty title.
        cid = client.post(
            f"/api/projects/{pid}/chats", json={"source_ids": [sid]},
        ).json()["id"]
        client.post(
            f"/api/projects/{pid}/chats/{cid}/turn",
            json={"text": "What recurring tensions appear?"},
        )
        r = client.get(f"/api/projects/{pid}/chats/{cid}")
        assert "recurring tensions" in r.json()["title"].lower()
