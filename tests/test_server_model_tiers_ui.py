"""End-to-end reachability tests for F8.11 (model-tier picker + download
manager) and F8.12 (baked-in model recommendations) — UI surface.

The pure data plane shipped previously:

  * ``scribe.model_tiers`` (b5e6211) — tier definitions, hardware
    autodetection, recommend_tier; covered by tests/test_model_tiers.py.
  * ``scribe.model_recommendations`` (967e663) — concrete per-tier
    model picks + embedding picks; covered by
    tests/test_model_recommendations.py.
  * ``GET  /api/system/model-tiers`` (F8.11) — endpoint covered by
    tests/test_server_ai_backend.py::TestSystemModelTiersEndpoint.
  * ``GET  /api/system/model-recommendations`` (F8.12) — endpoint
    covered by TestSystemModelRecommendationsEndpoint in the same file.
  * ``POST /api/projects/<pid>/ai/backend/pull`` (F8.11) — endpoint
    covered by TestPullModel in tests/test_server_ai_backend.py.

Until this commit landed, none of the above were reachable from the
user-facing surface — the settings page was a wireframe with the words
"Backend / Tier / Model name" written in flat text, and the project AI
page only had the F8.1 active-model form (no tier verdict, no per-tier
recommendations, no Pull button).

This file proves the wiring:

  * GET /projects/<pid>/ai renders the F8.11 tier picker card with
    data-test-feature="F8.11" and the per-tier rows + embedding rows
    + Pull / Use-as-default buttons that the JS later populates.
  * GET /settings renders a graduated F8.11 tile (no longer wireframe)
    that consumes ``/api/system/model-recommendations`` to display the
    hardware-aware verdict + recommended model tags.
  * The pull button on the project AI page is wired to
    ``POST /api/projects/<pid>/ai/backend/pull``; we run a TestClient
    POST against that route here as the reachability proof.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from scribe.ai_backend import (
    HTTPResponse,
)


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/test_server_project_ai.py / test_server_ai_backend.py)
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


def _make_project(client: TestClient, name: str = "Tier UI test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


class StubTransport:
    """Stub transport that maps (METHOD, path) → HTTPResponse so the
    pull endpoint doesn't fire real network calls. Mirrors the stub
    pattern in tests/test_server_ai_backend.py."""

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
        # Same path-matching trick as test_server_ai_backend.py: match
        # on the trailing path so we don't have to know the host.
        for (m, path), resp in self.routes.items():
            if m == method and url.endswith(path):
                self.calls.append({"method": method, "url": url, "body": body})
                return resp
        raise AssertionError(f"Unrouted stub call: {method} {url}")


# --------------------------------------------------------------------------- #
# /projects/<pid>/ai — F8.11 / F8.12 tier picker card renders
# --------------------------------------------------------------------------- #


class TestTierPickerCardRenders:
    """The project AI page must surface the model-tier picker so users
    can see the hardware-aware verdict + recommended models without
    leaving the page or running curl."""

    def test_card_renders_with_feature_marker(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        # The card itself.
        assert 'data-test-id="ai-tier-picker-card"' in r.text
        assert 'data-test-feature="F8.11"' in r.text
        # The F8.11 / F8.12 pill labels both surface, so the card
        # makes the relationship visible.
        assert "F8.11" in r.text
        assert "F8.12" in r.text
        # Heading copy and intent.
        assert "Model tier picker" in r.text

    def test_card_has_hardware_row(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-id="ai-tier-hardware"' in r.text

    def test_card_has_tier_list_container(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-id="ai-tier-list"' in r.text

    def test_card_has_embedding_list_container(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-id="ai-embed-list"' in r.text

    def test_card_loader_calls_recommendations_endpoint(self, server_env) -> None:
        """The JS must hit /api/system/model-recommendations to populate
        the per-tier model picks (F8.12)."""
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert "/api/system/model-recommendations" in r.text

    def test_card_loader_calls_pull_endpoint(self, server_env) -> None:
        """The Pull button is wired to the project-scoped F8.11 pull
        endpoint."""
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert "/ai/backend/pull" in r.text

    def test_card_has_loadModelTiers_init(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        # The init call wires the card to the recommendations endpoint
        # on page load.
        assert "loadModelTiers()" in r.text


# --------------------------------------------------------------------------- #
# /api/system/model-recommendations — proves the data the UI consumes
# --------------------------------------------------------------------------- #


class TestRecommendationsEndpointShape:
    """The page can't render without the recommendations endpoint; this
    pins the fields the UI relies on (hardware, tiers, recommended,
    embedding_models)."""

    def test_recommendations_returns_full_shape(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.get("/api/system/model-recommendations")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "hardware",
            "tiers",
            "recommended",
            "embedding_models",
        }
        assert isinstance(body["tiers"], list) and len(body["tiers"]) == 3
        # Each tier carries the F8.12 recommended_models list with at
        # least one default-flagged entry.
        for tier in body["tiers"]:
            assert "recommended_models" in tier
            assert "fit" in tier
            assert tier["fit"] in ("comfortable", "marginal", "infeasible")
            models = tier["recommended_models"]
            assert isinstance(models, list)
            assert len(models) >= 1
            assert any(m.get("is_default") for m in models)
        # Embedding picks are not tier-bound.
        assert isinstance(body["embedding_models"], list)
        assert len(body["embedding_models"]) >= 1


# --------------------------------------------------------------------------- #
# Pull route — UI's "Pull" button has a working backend
# --------------------------------------------------------------------------- #


class TestPullButtonWiring:
    """The card's Pull button posts to
    /api/projects/<pid>/ai/backend/pull. We exercise the route here
    (with a stub Ollama transport) to prove the UI's action target
    really runs."""

    def test_pull_route_runs_with_recommended_tag(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        # Stub the Ollama /api/pull stream with a single "success"
        # event so the pull completes without a real daemon.
        from scribe.ai_backend import HTTPResponse
        ndjson = (
            json.dumps({"status": "pulling manifest"}) + "\n" +
            json.dumps({"status": "success"}) + "\n"
        ).encode("utf-8")
        srv._ai_backend_transport_override = StubTransport({
            ("POST", "/api/pull"): HTTPResponse(
                status=200,
                headers={"Content-Type": "application/x-ndjson"},
                body=ndjson,
            ),
        })
        try:
            r = client.post(
                f"/api/projects/{pid}/ai/backend/pull",
                json={"model": "llama3.2:3b"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["success"] is True
            assert body["model"] == "llama3.2:3b"
        finally:
            srv._ai_backend_transport_override = None

    def test_pull_route_400_on_missing_model(self, server_env) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/backend/pull",
            json={"model": ""},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# /settings — top-level settings page graduates from wireframe stub
# --------------------------------------------------------------------------- #


class TestSettingsTierPicker:
    """The top-level Settings page exposes a read-only tier verdict
    (F8.11) + per-tier recommended models (F8.12). The actual download
    manager lives on the project AI page (above) because the pull
    endpoint is project-scoped."""

    def test_settings_card_renders(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert 'data-test-id="settings-tier-picker"' in r.text
        assert 'data-test-feature="F8.11"' in r.text

    def test_settings_card_has_summary_and_rows(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert 'data-test-id="settings-tier-summary"' in r.text
        assert 'data-test-id="settings-tier-rows"' in r.text
        assert 'data-test-id="settings-tier-embed"' in r.text

    def test_settings_card_links_to_projects_for_download(self, server_env) -> None:
        """The download manager lives on the project AI page; the
        settings card must point users there so the workflow is
        discoverable."""
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert 'data-test-id="settings-tier-projects-link"' in r.text

    def test_settings_card_consumes_recommendations_endpoint(
        self, server_env
    ) -> None:
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert "/api/system/model-recommendations" in r.text

    def test_settings_no_longer_renders_wireframe_banner(self, server_env) -> None:
        """The page used to be flagged as a wireframe stub at the top.
        F8.11 graduates the Local AI card so the banner has to come
        off."""
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        # The page-level "Wireframe." stub banner is gone (the card-
        # level "Wireframe" labels for HF token / profile may remain
        # while those features stay un-graduated).
        assert "<strong>Wireframe.</strong>" not in r.text
