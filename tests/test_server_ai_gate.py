"""Tests for the AI-gate HTTP surface (F8.10).

Exercises GET / PUT ``/api/projects/{pid}/ai/gate``. Pure-Python
data-model coverage lives in ``tests/test_ai_gate.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.ai_gate import (
    DEFAULT_MIN_CODES,
    DEFAULT_MIN_HAND_CODED_SOURCES,
    GATE_OVERRIDE_AUTO,
    GATE_OVERRIDE_FORCE_OFF,
    GATE_OVERRIDE_FORCE_ON,
    GATE_OVERRIDES,
    REASON_FORCE_OFF,
    REASON_INSUFFICIENT_BOTH,
    SETTING_KEY_ENABLED,
    SETTING_KEY_MIN_CODES,
    SETTING_KEY_MIN_HAND_CODED_SOURCES,
    SETTING_KEY_OVERRIDE,
)
from scribe.ai_provenance import (
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a test client with isolated tmp dirs."""
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


def _make_project(client: TestClient, name: str = "P") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# GET /ai/gate
# --------------------------------------------------------------------------- #


class TestGetAIGate:
    def test_default_for_new_project(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/gate")
        assert r.status_code == 200, r.text
        body = r.json()

        # Status reflects the empty project.
        status = body["status"]
        assert status["allowed"] is False
        assert status["reason"] == REASON_INSUFFICIENT_BOTH
        assert status["code_count"] == 0
        assert status["hand_coded_source_count"] == 0
        assert status["min_codes"] == DEFAULT_MIN_CODES
        assert status["min_hand_coded_sources"] == DEFAULT_MIN_HAND_CODED_SOURCES
        assert status["override"] == GATE_OVERRIDE_AUTO
        assert status["enabled"] is True
        assert status["feature"] == ""
        assert status["feature_exempt"] is False

        # Config block reflects the spec defaults.
        cfg = body["config"]
        assert cfg[SETTING_KEY_MIN_CODES] == DEFAULT_MIN_CODES
        assert cfg[SETTING_KEY_MIN_HAND_CODED_SOURCES] == DEFAULT_MIN_HAND_CODED_SOURCES
        assert cfg[SETTING_KEY_OVERRIDE] == GATE_OVERRIDE_AUTO
        assert cfg[SETTING_KEY_ENABLED] is True
        assert cfg["exempt_features"] == []

        # Vocabulary blocks make the UI's job easier.
        assert set(body["available_overrides"]) == set(GATE_OVERRIDES)

    def test_404_for_missing_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/" + ("0" * 12) + "/ai/gate")
        assert r.status_code == 404

    def test_feature_query_param_passes_through(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Mark quote-similarity as exempt so the feature param actually
        # changes the answer.
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"exempt_features": [AI_FEATURE_QUOTE_SIMILARITY]},
        )
        assert r.status_code == 200, r.text

        # Without ?feature, the gate is still blocked (defaults).
        r = client.get(f"/api/projects/{pid}/ai/gate")
        assert r.json()["status"]["allowed"] is False

        # With ?feature=quote_similarity, the exemption fires.
        r = client.get(
            f"/api/projects/{pid}/ai/gate",
            params={"feature": AI_FEATURE_QUOTE_SIMILARITY},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"]["allowed"] is True
        assert body["status"]["feature_exempt"] is True
        assert body["status"]["feature"] == AI_FEATURE_QUOTE_SIMILARITY

        # Querying a non-exempt feature stays blocked.
        r = client.get(
            f"/api/projects/{pid}/ai/gate",
            params={"feature": AI_FEATURE_CODE_SUGGESTION},
        )
        assert r.json()["status"]["allowed"] is False

    def test_unknown_feature_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/gate", params={"feature": "bogus"}
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# PUT /ai/gate
# --------------------------------------------------------------------------- #


class TestPutAIGate:
    def test_persists_config(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 4,
                "min_hand_coded_sources": 1,
                "override": GATE_OVERRIDE_FORCE_ON,
                "enabled": True,
                "exempt_features": [AI_FEATURE_QUOTE_SIMILARITY],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["config"][SETTING_KEY_MIN_CODES] == 4
        assert body["config"][SETTING_KEY_OVERRIDE] == GATE_OVERRIDE_FORCE_ON
        assert body["config"]["exempt_features"] == [AI_FEATURE_QUOTE_SIMILARITY]
        # Status reflects the new force_on override.
        assert body["status"]["allowed"] is True
        assert body["status"]["override"] == GATE_OVERRIDE_FORCE_ON

        # GET round-trips identically.
        r2 = client.get(f"/api/projects/{pid}/ai/gate")
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["config"][SETTING_KEY_MIN_CODES] == 4
        assert body2["config"]["exempt_features"] == [AI_FEATURE_QUOTE_SIMILARITY]
        assert body2["status"]["override"] == GATE_OVERRIDE_FORCE_ON

    def test_force_off_blocks_after_save(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"override": GATE_OVERRIDE_FORCE_OFF},
        )
        assert r.status_code == 200
        assert r.json()["status"]["reason"] == REASON_FORCE_OFF

    def test_invalid_override_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"override": "maybe"},
        )
        assert r.status_code == 400

    def test_unknown_feature_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"exempt_features": ["nope"]},
        )
        assert r.status_code == 400

    def test_negative_threshold_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"min_codes": -1},
        )
        assert r.status_code == 400

    def test_non_object_body_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate", json=[1, 2, 3]
        )
        assert r.status_code == 400

    def test_invalid_exempt_features_type_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"exempt_features": "not-a-list"},
        )
        assert r.status_code == 400

    def test_404_for_missing_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.put(
            "/api/projects/" + ("0" * 12) + "/ai/gate",
            json={"min_codes": 1},
        )
        assert r.status_code == 404

    def test_partial_update_uses_defaults(self, server_env) -> None:
        # PUT replaces the config; an empty body resets to defaults.
        _, client, _ = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"min_codes": 99},
        )
        # Now PUT an empty object — defaults reapply.
        r = client.put(f"/api/projects/{pid}/ai/gate", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["config"][SETTING_KEY_MIN_CODES] == DEFAULT_MIN_CODES
        assert body["config"]["exempt_features"] == []
