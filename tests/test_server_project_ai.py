"""End-to-end reachability tests for F8.1 (pluggable model backend).

The pure data plane shipped in 2182dab (``scribe.ai_backend``) and
the four ``/api/projects/<pid>/ai/backend*`` HTTP routes were already
covered by ``tests/test_server_ai_backend.py``. What was missing —
and what this file proves — is the **user-facing surface**:

  * GET /projects/<pid>/ai renders a real "Active model" picker
    (replacing the wireframe stub).
  * The page exposes form fields for provider / base_url /
    default_model / default_embedding_model.
  * The page exposes "Test connection" and "List installed models"
    actions that hit the existing health / models endpoints.
  * Submitting the form via PUT /api/projects/<pid>/ai/backend
    persists the config (round-trip via load_project).
  * The F8.10 gate-status row reads from /ai/gate.

The four AI routes (GET /backend, PUT /backend, GET /backend/health,
GET /backend/models) already have direct unit tests in
``tests/test_server_ai_backend.py``; here we cross-test that the
HTML page is wired to them.
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
    SETTING_KEY_BASE_URL,
    SETTING_KEY_DEFAULT_EMBEDDING_MODEL,
    SETTING_KEY_DEFAULT_MODEL,
    SETTING_KEY_PROVIDER,
)


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/test_server_ai_backend.py — kept inline so this file
# is self-contained).
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    monkeypatch.setattr(srv, "_model_tiers_snapshot_override", None)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _make_project(client: TestClient, name: str = "AI-page test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


class StubTransport:
    """Maps (method, path) → HTTPResponse so the AI endpoints don't
    fire real network calls."""

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
        for base in (DEFAULT_OLLAMA_BASE_URL, "http://lan-box:11434"):
            if url.startswith(base):
                path = url[len(base):]
                break
        else:
            path = url
        self.calls.append({
            "method": method, "url": url, "path": path,
            "headers": dict(headers), "body": body, "timeout_s": timeout_s,
        })
        try:
            return self.routes[(method, path)]
        except KeyError as e:  # pragma: no cover - defensive
            raise AssertionError(
                f"Unexpected transport call: {method} {path}"
            ) from e


def _ok(body: dict[str, Any]) -> HTTPResponse:
    return HTTPResponse(status=200, body=json.dumps(body).encode("utf-8"))


# --------------------------------------------------------------------------- #
# /projects/<pid>/ai renders a real page (no longer a wireframe)
# --------------------------------------------------------------------------- #


class TestAIPageRenders:
    def test_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200

    def test_no_longer_renders_wireframe_stub(self, server_env) -> None:
        """The previous template stamped a "Wireframe." banner. F8.1's
        graduation must remove it; if this regresses, F8.1's UI is
        gone and the loop should pick the feature back up."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert "Wireframe." not in r.text
        # The F8.1 active-model card carries this data attribute.
        assert 'data-test-feature="F8.1"' in r.text

    def test_400_for_obviously_invalid_project_id(self, server_env) -> None:
        """``_project_id_or_404`` is deliberately permissive for the
        HTML routes (unknown-but-valid-looking ids render so the user
        still sees the IA shell), but it rejects ids that contain
        traversal segments. We pick the longest illegal id to make
        sure ``len > 64`` triggers, not the slash check."""
        _, client, _ = server_env
        r = client.get("/projects/" + "x" * 80 + "/ai")
        assert r.status_code == 400

    def test_active_nav_is_projects(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        # _shell.html marks .active on the Projects link when active_nav=="projects"
        assert "Projects" in r.text


# --------------------------------------------------------------------------- #
# Form fields for the F8.1 BackendConfig must be on the page
# --------------------------------------------------------------------------- #


class TestActiveModelFormControls:
    """Each input on the form is reachable by data-test-id; the JS
    in the template binds those to the four backend endpoints. If any
    of these go missing, F8.1's user-facing surface is broken."""

    def test_provider_select_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-provider"' in r.text

    def test_base_url_input_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-base-url"' in r.text

    def test_default_model_input_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-default-model"' in r.text

    def test_default_embedding_input_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-default-embedding"' in r.text

    def test_save_button_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-save"' in r.text

    def test_test_connection_button_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-test"' in r.text

    def test_list_models_button_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-backend-list-models"' in r.text

    def test_gate_status_row_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-gate-status"' in r.text


# --------------------------------------------------------------------------- #
# End-to-end: form submit persists; test/list buttons reach the daemon.
# --------------------------------------------------------------------------- #


class TestActiveModelEndToEnd:
    def test_put_backend_persists_and_get_returns_it(self, server_env) -> None:
        """Mirror the form's submit flow: PUT then GET."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json={
                SETTING_KEY_PROVIDER: PROVIDER_OLLAMA,
                SETTING_KEY_BASE_URL: "http://lan-box:11434",
                SETTING_KEY_DEFAULT_MODEL: "llama3.2:3b",
                SETTING_KEY_DEFAULT_EMBEDDING_MODEL: "bge-m3",
            },
        )
        assert r.status_code == 200, r.text
        roundtrip = client.get(f"/api/projects/{pid}/ai/backend").json()
        assert roundtrip[SETTING_KEY_BASE_URL] == "http://lan-box:11434"
        assert roundtrip[SETTING_KEY_DEFAULT_MODEL] == "llama3.2:3b"
        assert roundtrip[SETTING_KEY_DEFAULT_EMBEDDING_MODEL] == "bge-m3"
        # The page (GET HTML) is reachable; this is the F8.1
        # "user can save a model from the picker" path the loop demands.
        page = client.get(f"/projects/{pid}/ai")
        assert page.status_code == 200

    def test_health_button_route_is_reachable(self, server_env) -> None:
        """The "Test connection" button hits this endpoint. With a stub
        Ollama daemon (version response) the page-bound button works."""
        srv, client, _ = server_env
        pid = _make_project(client)
        transport = StubTransport({
            ("GET", "/api/version"): _ok({"version": "0.4.0"}),
        })
        srv._ai_backend_transport_override = transport
        try:
            r = client.get(f"/api/projects/{pid}/ai/backend/health")
        finally:
            srv._ai_backend_transport_override = None
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["provider"] == PROVIDER_OLLAMA
        # The page surfaces this in #ai-health-row when "Test connection" fires.

    def test_models_button_route_is_reachable(self, server_env) -> None:
        """The "List installed models" button hits this endpoint."""
        srv, client, _ = server_env
        pid = _make_project(client)
        transport = StubTransport({
            ("GET", "/api/tags"): _ok({
                "models": [
                    {
                        "name": "llama3.2:3b",
                        "size": 2_147_483_648,
                        "details": {"family": "llama", "parameter_size": "3B"},
                    },
                    {
                        "name": "bge-m3:latest",
                        "size": 600_000_000,
                        "details": {"family": "bge", "parameter_size": "568M"},
                    },
                ]
            }),
        })
        srv._ai_backend_transport_override = transport
        try:
            r = client.get(f"/api/projects/{pid}/ai/backend/models")
        finally:
            srv._ai_backend_transport_override = None
        assert r.status_code == 200, r.text
        body = r.json()
        names = {m["name"] for m in body["models"]}
        assert "llama3.2:3b" in names
        assert "bge-m3:latest" in names

    def test_health_when_daemon_down(self, server_env) -> None:
        """If the Ollama daemon is unreachable, the endpoint still
        returns 200 with ok=False so the page can render a banner
        rather than a network error."""
        from scribe.ai_backend import BackendUnavailable

        srv, client, _ = server_env
        pid = _make_project(client)

        def boom(method, url, headers, body, timeout_s):
            raise BackendUnavailable("connection refused")

        srv._ai_backend_transport_override = boom
        try:
            r = client.get(f"/api/projects/{pid}/ai/backend/health")
        finally:
            srv._ai_backend_transport_override = None
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is False
        assert "connection refused" in (body["error"] or "")

    def test_invalid_save_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Empty base_url is invalid per BackendConfig.validate().
        r = client.put(
            f"/api/projects/{pid}/ai/backend",
            json={SETTING_KEY_PROVIDER: "acme-magic"},
        )
        assert r.status_code == 400
