"""Tests for the new UI shell + wireframe pages.

Verifies that every new HTML route renders without crashing and that the
shell partial wires up correctly. Nothing here exercises real data — these
are placeholder pages today and the loop will graduate them per-feature.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Standalone TestClient with the app's storage redirected to tmp."""
    from scribe import server as srv

    # Redirect storage so test runs don't touch the developer's outputs/.
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    if hasattr(srv, "PROJECTS_DIR"):
        monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})

    return TestClient(srv.app)


VALID_PID = "abcdef012345"


# --------------------------------------------------------------------------- #
# Top-level pages
# --------------------------------------------------------------------------- #


class TestTopLevelPages:
    def test_projects_list_renders(self, client: TestClient) -> None:
        r = client.get("/projects")
        assert r.status_code == 200
        assert "<title>Scribe — Projects</title>" in r.text
        # Shell is wired in: brand link + nav.
        assert 'href="/library"' in r.text
        assert 'href="/projects"' in r.text
        assert 'href="/settings"' in r.text

    def test_project_new_renders(self, client: TestClient) -> None:
        r = client.get("/projects/new")
        assert r.status_code == 200
        assert "New project" in r.text

    def test_settings_renders(self, client: TestClient) -> None:
        r = client.get("/settings")
        assert r.status_code == 200
        assert "Settings" in r.text


# --------------------------------------------------------------------------- #
# Project subpages — every link from the project home renders
# --------------------------------------------------------------------------- #


class TestProjectSubpages:
    @pytest.mark.parametrize("path,marker", [
        (f"/projects/{VALID_PID}", "Project"),
        (f"/projects/{VALID_PID}/sources", "Sources"),
        (f"/projects/{VALID_PID}/sources/add", "Add source"),
        (f"/projects/{VALID_PID}/codebook", "Codebook"),
        (f"/projects/{VALID_PID}/queries", "Queries"),
        (f"/projects/{VALID_PID}/memos", "Memos"),
        (f"/projects/{VALID_PID}/ai", "AI suggestions"),
        (f"/projects/{VALID_PID}/audit", "Audit timeline"),
        (f"/projects/{VALID_PID}/settings", "Project settings"),
        (f"/projects/{VALID_PID}/sources/{VALID_PID}", "Coding"),
    ])
    def test_subpage_renders(self, client: TestClient, path: str, marker: str) -> None:
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert marker in r.text


# --------------------------------------------------------------------------- #
# Project-id validation — clear rejection of malicious paths
# --------------------------------------------------------------------------- #


class TestProjectIdValidation:
    @pytest.mark.parametrize("path", [
        "/projects/" + "x" * 200 + "/sources",   # over-long
        f"/projects/{VALID_PID}/sources/" + "x" * 200,
    ])
    def test_rejects_overly_long_id(self, client: TestClient, path: str) -> None:
        r = client.get(path)
        # Validator path returns 400; some extreme inputs may be rejected
        # by the routing layer with 404. Either is fine — neither leaks.
        assert r.status_code in (400, 404), (path, r.status_code)


# --------------------------------------------------------------------------- #
# Old routes still work (no regressions)
# --------------------------------------------------------------------------- #


class TestExistingRoutesIntact:
    def test_index_still_renders(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        # Both old and new pages should mention Scribe somewhere.
        assert "Scribe" in r.text

    def test_library_still_renders(self, client: TestClient) -> None:
        r = client.get("/library")
        assert r.status_code == 200

    def test_capabilities_endpoint_unchanged(self, client: TestClient) -> None:
        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Either the old or post-AMD shape is fine for this smoke test.
        assert "parakeet" in body or "gpu" in body
