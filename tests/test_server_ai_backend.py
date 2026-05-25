"""Tests for the AI-backend HTTP surface (F8.1).

These exercise the four endpoints under
``/api/projects/{pid}/ai/backend`` using a stub transport so no real
network calls fire. Pure-Python data-model coverage lives in
``tests/test_ai_backend.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from scribe.ai_backend import (
    DEFAULT_OLLAMA_BASE_URL,
    HTTPResponse,
    PROVIDER_OLLAMA,
    SETTING_AI_BACKEND,
    SETTING_AI_BACKEND_HEADERS,
    SETTING_KEY_BASE_URL,
    SETTING_KEY_DEFAULT_MODEL,
    SETTING_KEY_PROVIDER,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a test client with isolated tmp dirs (mirrors the
    existing fixture in test_server.py)."""
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
    # Reset transport override between tests.
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    # Reset model-tiers snapshot override (F8.11) so tests don't leak.
    monkeypatch.setattr(srv, "_model_tiers_snapshot_override", None)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _make_project(client: TestClient, name: str = "P") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


class StubTransport:
    """Stub transport for AI backend endpoint tests."""

    def __init__(self, routes: dict[tuple[str, str], HTTPResponse]) -> None:
        self.routes = dict(routes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HTTPResponse:
        # Strip the well-known base so routes can match by path alone.
        for base in (
            DEFAULT_OLLAMA_BASE_URL,
            "http://lan-box:11434",
            "http://nope:11434",
        ):
            if url.startswith(base):
                path = url[len(base):]
                break
        else:
            path = url
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        try:
            return self.routes[(method, path)]
        except KeyError as e:
            raise AssertionError(
                f"Unexpected transport call: {method} {url} (path={path}); "
                f"known: {sorted(self.routes.keys())}"
            ) from e


def _ok(body: dict[str, Any]) -> HTTPResponse:
    return HTTPResponse(status=200, body=json.dumps(body).encode("utf-8"))


# --------------------------------------------------------------------------- #
# GET /ai/backend
# --------------------------------------------------------------------------- #


class TestGetAIBackend:
    def test_returns_default_for_new_project(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/backend")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body[SETTING_KEY_PROVIDER] == PROVIDER_OLLAMA
        assert body[SETTING_KEY_BASE_URL] == DEFAULT_OLLAMA_BASE_URL
        assert body["extra_headers"] == {}
        assert PROVIDER_OLLAMA in body["available_providers"]

    def test_404_for_missing_project(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/api/projects/abcdef012345/ai/backend")
        assert r.status_code == 404

    def test_400_for_bad_project_id(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/api/projects/not-hex/ai/backend")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# PUT /ai/backend
# --------------------------------------------------------------------------- #


class TestPutAIBackend:
    def test_replaces_config(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                SETTING_KEY_PROVIDER: PROVIDER_OLLAMA,
                SETTING_KEY_BASE_URL: "http://lan-box:11434",
                SETTING_KEY_DEFAULT_MODEL: "llama3.2:3b",
                "extra_headers": {"X-Token": "abc"},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body[SETTING_KEY_BASE_URL] == "http://lan-box:11434"
        assert body[SETTING_KEY_DEFAULT_MODEL] == "llama3.2:3b"
        assert body["extra_headers"] == {"X-Token": "abc"}
        # And it persisted: a follow-up GET reads it back.
        body2 = client.get(f"/api/projects/{pid}/ai/backend").json()
        assert body2[SETTING_KEY_BASE_URL] == "http://lan-box:11434"
        assert body2["extra_headers"] == {"X-Token": "abc"}

    def test_rejects_invalid_provider(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json={SETTING_KEY_PROVIDER: "acme-magic"},
        )
        assert r.status_code == 400

    def test_rejects_non_object_body(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json=[1, 2, 3],
        )
        assert r.status_code == 400

    def test_rejects_invalid_json(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_404_for_missing_project(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.put(
            "/api/projects/abcdef012345/ai/backend",
            json={SETTING_KEY_PROVIDER: PROVIDER_OLLAMA},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# GET /ai/backend/health
# --------------------------------------------------------------------------- #


class TestAIBackendHealth:
    def test_returns_ok_when_daemon_reachable(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        transport = StubTransport(
            {("GET", "/api/version"): _ok({"version": "0.5.7"})}
        )
        srv._ai_backend_transport_override = transport

        r = client.get(f"/api/projects/{pid}/ai/backend/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["detail"] == "0.5.7"
        assert body["provider"] == PROVIDER_OLLAMA

    def test_returns_not_ok_when_unreachable(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        from scribe.ai_backend import BackendUnavailable

        def boom(*args, **kwargs):
            raise BackendUnavailable("connection refused")

        srv._ai_backend_transport_override = boom

        r = client.get(f"/api/projects/{pid}/ai/backend/health")
        # 200 with ok=False is a deliberate API choice (see endpoint
        # docstring): the frontend wants to render a "backend down"
        # banner, not throw a network error.
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "connection refused" in body["error"]


# --------------------------------------------------------------------------- #
# GET /ai/backend/models
# --------------------------------------------------------------------------- #


class TestAIBackendModels:
    def test_returns_model_list(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        payload = {
            "models": [
                {
                    "name": "llama3.2:3b",
                    "size": 2_000_000_000,
                    "details": {
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M",
                    },
                },
            ]
        }
        transport = StubTransport({("GET", "/api/tags"): _ok(payload)})
        srv._ai_backend_transport_override = transport

        r = client.get(f"/api/projects/{pid}/ai/backend/models")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["models"]) == 1
        assert body["models"][0]["name"] == "llama3.2:3b"
        assert body["models"][0]["kind"] == "generative"

    def test_502_when_unavailable(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        from scribe.ai_backend import BackendUnavailable

        def boom(*args, **kwargs):
            raise BackendUnavailable("connection refused")

        srv._ai_backend_transport_override = boom

        r = client.get(f"/api/projects/{pid}/ai/backend/models")
        assert r.status_code == 502
        assert "unavailable" in r.json().get("detail", "").lower()

    def test_400_when_provider_invalid_in_settings(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hand-craft a project.json with an invalid provider so the
        # configured backend fails resolution.
        srv, client, _ = server_env
        from scribe import projects as _projects

        p = _projects.Project.new(name="P")
        p.settings = {SETTING_AI_BACKEND: {SETTING_KEY_PROVIDER: "acme-magic"}}
        _projects.save_project(srv._projects_root(), p)

        r = client.get(f"/api/projects/{p.id}/ai/backend/models")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Headers persistence
# --------------------------------------------------------------------------- #


class TestHeadersPersistence:
    def test_headers_round_trip_via_disk(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        # Set headers via PUT.
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                SETTING_KEY_PROVIDER: PROVIDER_OLLAMA,
                SETTING_KEY_BASE_URL: "http://lan-box:11434",
                "extra_headers": {"X-Token": "abc"},
            },
        )
        assert r.status_code == 200

        # Peek at the on-disk project.json: headers must live under the
        # sibling top-level setting (depth-1 dict-of-scalars).
        project_path = srv._projects_root() / pid / "project.json"
        on_disk = json.loads(project_path.read_text())
        assert SETTING_AI_BACKEND_HEADERS in on_disk["settings"]
        assert on_disk["settings"][SETTING_AI_BACKEND_HEADERS] == {
            "X-Token": "abc"
        }

        # Headers come back through GET.
        body = client.get(f"/api/projects/{pid}/ai/backend").json()
        assert body["extra_headers"] == {"X-Token": "abc"}

    def test_clearing_headers_drops_sibling_key(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                SETTING_KEY_PROVIDER: PROVIDER_OLLAMA,
                "extra_headers": {"X-Token": "abc"},
            },
        )
        client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                SETTING_KEY_PROVIDER: PROVIDER_OLLAMA,
                "extra_headers": {},
            },
        )
        on_disk = json.loads(
            (srv._projects_root() / pid / "project.json").read_text()
        )
        assert SETTING_AI_BACKEND_HEADERS not in on_disk["settings"]


# --------------------------------------------------------------------------- #
# /api/projects/{id}/ai/backend/pull (F8.11)
# --------------------------------------------------------------------------- #


class TestProjectAiBackendPullEndpoint:
    OK_STREAM = (
        b'{"status": "pulling manifest"}\n'
        b'{"status": "pulling l1", "digest": "sha256:1", "total": 10, "completed": 10}\n'
        b'{"status": "writing manifest"}\n'
        b'{"status": "success"}\n'
    )

    def test_pull_success(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        transport = StubTransport(
            {("POST", "/api/pull"): HTTPResponse(200, self.OK_STREAM)}
        )
        srv._ai_backend_transport_override = transport

        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            json={"model": "llama3.2:3b"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert body["model"] == "llama3.2:3b"
        assert body["error"] == ""
        assert len(body["events"]) == 4
        # Each event shape includes the keys the UI cares about.
        for ev in body["events"]:
            for key in ("status", "digest", "total", "completed", "error", "percent"):
                assert key in ev

    def test_pull_failure_returns_200_with_success_false(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        transport = StubTransport(
            {("POST", "/api/pull"): HTTPResponse(
                200,
                b'{"status":"pulling manifest"}\n{"error":"model not found"}\n',
            )}
        )
        srv._ai_backend_transport_override = transport

        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            json={"model": "nope:99"},
        )
        # 200 — UI wants the per-event log even when the daemon errored.
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert "model not found" in body["error"]

    def test_pull_unreachable_daemon_returns_502(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)

        from scribe.ai_backend import BackendUnavailable

        def boom(*args, **kwargs):
            raise BackendUnavailable("connection refused")

        srv._ai_backend_transport_override = boom

        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            json={"model": "x:1b"},
        )
        assert r.status_code == 502

    def test_pull_missing_model_returns_400(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)

        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            json={},
        )
        assert r.status_code == 400

    def test_pull_invalid_json_returns_400(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)

        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_pull_unknown_project_returns_404(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.post(
            "/api/projects/" + ("a" * 12) + "/ai/backend/pull",
            json={"model": "x:1b"},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# /api/system/model-tiers (F8.11)
# --------------------------------------------------------------------------- #


class TestSystemModelTiersEndpoint:
    def test_returns_three_tiers_with_recommendation(self, server_env) -> None:
        srv, client, _ = server_env
        from scribe.model_tiers import (
            HardwareSnapshot,
            KNOWN_TIER_IDS,
            TIER_LARGE,
        )

        srv._model_tiers_snapshot_override = HardwareSnapshot(
            gpu_backend="cuda",
            gpu_name="Test GPU",
            vram_gb=24.0,
            system_ram_gb=64.0,
            cpu_count=16,
        )
        try:
            r = client.get("/api/system/model-tiers")
            assert r.status_code == 200
            body = r.json()
            assert body["recommended"] == TIER_LARGE
            assert body["hardware"]["gpu_backend"] == "cuda"
            assert body["hardware"]["vram_gb"] == 24.0
            assert [t["id"] for t in body["tiers"]] == list(KNOWN_TIER_IDS)
            for entry in body["tiers"]:
                assert entry["fit"] in ("comfortable", "marginal", "infeasible")
        finally:
            srv._model_tiers_snapshot_override = None

    def test_returns_small_recommendation_on_weak_box(self, server_env) -> None:
        srv, client, _ = server_env
        from scribe.model_tiers import HardwareSnapshot, TIER_SMALL

        srv._model_tiers_snapshot_override = HardwareSnapshot(
            gpu_backend="cpu",
            gpu_name="",
            vram_gb=0.0,
            system_ram_gb=8.0,
            cpu_count=4,
        )
        try:
            r = client.get("/api/system/model-tiers")
            assert r.status_code == 200
            body = r.json()
            assert body["recommended"] == TIER_SMALL
            assert body["hardware"]["gpu_backend"] == "cpu"
        finally:
            srv._model_tiers_snapshot_override = None

    def test_endpoint_works_with_no_override(self, server_env) -> None:
        # No override set — falls back to detect_hardware. Must return
        # 200 with a known tier id, regardless of the host's hardware.
        _srv, client, _ = server_env
        r = client.get("/api/system/model-tiers")
        assert r.status_code == 200
        body = r.json()
        assert body["recommended"] in ("small", "mid", "large")
        assert "hardware" in body
        assert "tiers" in body


# --------------------------------------------------------------------------- #
# /api/system/model-recommendations (F8.12)
# --------------------------------------------------------------------------- #


class TestSystemModelRecommendationsEndpoint:
    """The F8.12 endpoint surfaces concrete per-tier model picks plus
    embedding-model recommendations alongside the F8.11 hardware fit."""

    def test_returns_full_shape(self, server_env) -> None:
        srv, client, _ = server_env
        from scribe.model_tiers import HardwareSnapshot, TIER_LARGE

        srv._model_tiers_snapshot_override = HardwareSnapshot(
            gpu_backend="cuda",
            gpu_name="Test GPU",
            vram_gb=24.0,
            system_ram_gb=64.0,
            cpu_count=16,
        )
        try:
            r = client.get("/api/system/model-recommendations")
            assert r.status_code == 200
            body = r.json()
            assert set(body.keys()) == {
                "hardware",
                "tiers",
                "recommended",
                "embedding_models",
            }
            assert body["recommended"] == TIER_LARGE
            assert len(body["tiers"]) == 3
        finally:
            srv._model_tiers_snapshot_override = None

    def test_each_tier_carries_recommended_models(self, server_env) -> None:
        srv, client, _ = server_env
        from scribe.model_tiers import HardwareSnapshot

        srv._model_tiers_snapshot_override = HardwareSnapshot(
            gpu_backend="cuda",
            gpu_name="GPU",
            vram_gb=24.0,
            system_ram_gb=64.0,
            cpu_count=8,
        )
        try:
            r = client.get("/api/system/model-recommendations")
            assert r.status_code == 200
            body = r.json()
            for entry in body["tiers"]:
                assert "recommended_models" in entry
                models = entry["recommended_models"]
                assert isinstance(models, list)
                assert len(models) >= 1
                defaults = [m for m in models if m["is_default"]]
                assert len(defaults) == 1, entry["id"]
                # Tier-fit fields from F8.11 still present.
                assert entry["fit"] in (
                    "comfortable",
                    "marginal",
                    "infeasible",
                )
        finally:
            srv._model_tiers_snapshot_override = None

    def test_spec_models_present(self, server_env) -> None:
        # The F8.12 spec hard-codes these tags. The endpoint MUST surface
        # them or downstream UI will silently lose the recommendation.
        srv, client, _ = server_env
        from scribe.model_tiers import HardwareSnapshot

        srv._model_tiers_snapshot_override = HardwareSnapshot(
            gpu_backend="cuda",
            gpu_name="GPU",
            vram_gb=24.0,
            system_ram_gb=64.0,
            cpu_count=8,
        )
        try:
            r = client.get("/api/system/model-recommendations")
            body = r.json()
            tier_models = {
                t["id"]: {m["tag"] for m in t["recommended_models"]}
                for t in body["tiers"]
            }
            assert "llama3.2:3b" in tier_models["small"]
            assert "phi3.5:3.8b" in tier_models["small"]
            assert "phi4:14b" in tier_models["mid"]
            assert "mistral-nemo:12b" in tier_models["mid"]
            assert "qwen2.5:32b" in tier_models["large"]
            assert "llama3.3:70b" in tier_models["large"]
            embedding_tags = {m["tag"] for m in body["embedding_models"]}
            assert "bge-m3" in embedding_tags
            assert "nomic-embed-text:v1.5" in embedding_tags
        finally:
            srv._model_tiers_snapshot_override = None

    def test_endpoint_works_with_no_override(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.get("/api/system/model-recommendations")
        assert r.status_code == 200
        body = r.json()
        assert body["recommended"] in ("small", "mid", "large")
        assert "embedding_models" in body
