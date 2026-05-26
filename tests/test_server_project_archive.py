"""End-to-end reachability tests for F1.5 (Project file format on disk +
bundle archive round-trip).

The pure data + format + zip-archive logic lives in
``scribe/project_format.py`` with unit tests in
``tests/test_project_format.py``. F1.5 also explicitly deferred the
HTTP + UI surface ("Deferred: wiring this into FastAPI endpoints
(project export/import HTTP handlers) and the CLI"). This file proves
the loop's Reachable-via gate is now satisfied:

  • GET /api/projects/<pid>/archive returns a .scribe.zip.
  • POST /api/projects/import-archive restores it (round-trip).
  • The project home page renders an "Export archive" button that
    points at the GET endpoint.
  • The projects index page renders an "Import archive" control
    that POSTs to the import endpoint via JS.

Why a separate file: matches the per-feature test naming pattern
established by F1.1–F1.4 (test_server_projects.py,
test_server_sources.py, test_server_participants.py,
test_server_sampling_log.py). Easier to grep when reconstructing the
audit trail from ``git log``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads/outputs/projects.

    Identical to the fixture in test_server_projects.py — kept inline
    rather than imported so this file stays self-contained.
    """
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


def _make_project(client: TestClient, name: str = "Cohort A interviews") -> dict:
    """Helper: POST /api/projects and return the JSON body."""
    r = client.post(
        "/api/projects",
        json={
            "name": name,
            "research_question": "How do midwives reason about risk?",
            "methodology": "constructivist-grounded-theory",
            "codebook_stage": "initial",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Project home renders the Export button
# --------------------------------------------------------------------------- #


class TestProjectHomeExportButton:
    """The project home page must surface the F1.5 archive download.

    Without a visible affordance the data-format work is technically
    reachable (``curl /api/projects/<id>/archive`` works) but the user
    has no way to find it. The Reachable-via gate requires UI proof.
    """

    def test_button_renders_on_project_home(self, server_env) -> None:
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/projects/{p['id']}")
        assert r.status_code == 200
        assert "Export archive" in r.text

    def test_button_links_to_archive_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/projects/{p['id']}")
        assert r.status_code == 200
        # Both the path and the include_outputs flag should be wired in.
        assert f"/api/projects/{p['id']}/archive" in r.text
        assert "include_outputs=1" in r.text

    def test_button_has_download_attribute(self, server_env) -> None:
        """``download`` triggers Save-As rather than rendering inline."""
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/projects/{p['id']}")
        # Look for a download attribute somewhere in the template;
        # exact placement is a UI detail but its presence is required
        # so the .scribe.zip doesn't try to render in the browser.
        assert "download" in r.text


# --------------------------------------------------------------------------- #
# Projects list renders the Import button
# --------------------------------------------------------------------------- #


class TestProjectsListImportButton:
    """The projects index page must surface the F1.5 archive import.

    Without an upload control the import endpoint is technically
    reachable but invisible. The button + hidden <input type="file">
    pattern keeps the New-project CTA visually primary while still
    putting Import one click away.
    """

    def test_button_renders_on_projects_index(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        assert r.status_code == 200
        assert "Import archive" in r.text

    def test_file_input_is_present(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        assert r.status_code == 200
        # The hidden file input is the actual upload control.
        assert 'id="importArchiveInput"' in r.text
        assert 'type="file"' in r.text

    def test_handler_posts_to_import_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        assert r.status_code == 200
        # The JS handler must POST to /api/projects/import-archive.
        assert "/api/projects/import-archive" in r.text


# --------------------------------------------------------------------------- #
# GET /api/projects/<pid>/archive — happy path
# --------------------------------------------------------------------------- #


class TestArchiveExportEndpoint:
    """The export endpoint must return a valid zip for an existing project."""

    def test_returns_200_with_zip_body(self, server_env) -> None:
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/api/projects/{p['id']}/archive")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        # Verify the bytes parse as a valid zip.
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n == f"{p['id']}/manifest.json" for n in names)
        assert any(n == f"{p['id']}/project.json" for n in names)

    def test_content_disposition_uses_slugified_name(self, server_env) -> None:
        _, client, _ = server_env
        p = _make_project(client, name="Cohort A — interviews")
        r = client.get(f"/api/projects/{p['id']}/archive")
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        # Punctuation collapses to dashes; .scribe.zip suffix is mandatory.
        assert "cohort-a-interviews.scribe.zip" in cd
        assert "attachment" in cd

    def test_404_for_missing_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/abcdef012345/archive")
        assert r.status_code == 404

    def test_400_for_malformed_id(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex!!!/archive")
        assert r.status_code == 400

    def test_archive_manifest_is_well_formed(self, server_env) -> None:
        """The exported manifest.json must be valid F1.5 format."""
        _, client, _ = server_env
        p = _make_project(client, name="Pilot study")
        r = client.get(f"/api/projects/{p['id']}/archive")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        manifest_bytes = zf.read(f"{p['id']}/manifest.json")
        manifest = json.loads(manifest_bytes)
        assert manifest["project_id"] == p["id"]
        assert manifest["name"] == "Pilot study"
        assert manifest["format"] == "scribe-project"
        assert manifest["format_version"] >= 1


# --------------------------------------------------------------------------- #
# POST /api/projects/import-archive — happy path + round-trip
# --------------------------------------------------------------------------- #


class TestArchiveImportRoundTrip:
    """Export → delete → import must restore the project bit-for-bit."""

    def test_round_trip_via_http(self, server_env) -> None:
        srv, client, _ = server_env
        p = _make_project(client, name="Cohort A interviews")

        # Export.
        r = client.get(f"/api/projects/{p['id']}/archive")
        assert r.status_code == 200
        archive_bytes = r.content

        # Delete the original.
        r = client.delete(f"/api/projects/{p['id']}")
        assert r.status_code == 200
        r = client.get(f"/api/projects/{p['id']}")
        assert r.status_code == 404

        # Import.
        files = {"archive": ("study.scribe.zip", archive_bytes, "application/zip")}
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["project_id"] == p["id"]
        assert body["name"] == "Cohort A interviews"
        assert body["redirect"] == f"/projects/{p['id']}"

        # The restored project is now reachable through the same UI
        # the original was: GET /api/projects/<id> + GET /projects/<id>.
        r = client.get(f"/api/projects/{p['id']}")
        assert r.status_code == 200
        assert r.json()["name"] == "Cohort A interviews"
        r = client.get(f"/projects/{p['id']}")
        assert r.status_code == 200

    def test_returns_summary_counts(self, server_env) -> None:
        """The import response includes counts so the UI can show a confirmation."""
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/api/projects/{p['id']}/archive")
        archive_bytes = r.content
        client.delete(f"/api/projects/{p['id']}")
        files = {"archive": ("p.scribe.zip", archive_bytes, "application/zip")}
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 201
        body = r.json()
        assert "sources" in body
        assert "participants" in body
        assert "codes" in body
        assert "sampling_entries" in body

    def test_refuses_overwrite_without_flag(self, server_env) -> None:
        """The default is conservative: importing onto an existing project 409s."""
        _, client, _ = server_env
        p = _make_project(client)
        r = client.get(f"/api/projects/{p['id']}/archive")
        archive_bytes = r.content
        # Project still exists. Import without overwrite → 409.
        files = {"archive": ("p.scribe.zip", archive_bytes, "application/zip")}
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 409

    def test_overwrite_flag_replaces_existing(self, server_env) -> None:
        """``overwrite=1`` is required to overwrite an existing project."""
        _, client, _ = server_env
        p = _make_project(client)
        # Snapshot, then mutate the live project, then re-import — the
        # imported manifest's name should win.
        r = client.get(f"/api/projects/{p['id']}/archive")
        archive_bytes = r.content
        client.patch(f"/api/projects/{p['id']}", json={"name": "Mutated locally"})
        files = {"archive": ("p.scribe.zip", archive_bytes, "application/zip")}
        r = client.post(
            "/api/projects/import-archive",
            files=files,
            data={"overwrite": "1"},
        )
        assert r.status_code == 201, r.text
        # Live project now matches the archive, not the local mutation.
        r = client.get(f"/api/projects/{p['id']}")
        assert r.json()["name"] == p["name"]


# --------------------------------------------------------------------------- #
# POST /api/projects/import-archive — defensive paths
# --------------------------------------------------------------------------- #


class TestArchiveImportDefences:
    """Bad uploads must produce 4xx errors with useful detail."""

    def test_400_for_non_zip_payload(self, server_env) -> None:
        _, client, _ = server_env
        files = {
            "archive": ("garbage.zip", b"not a zip", "application/zip"),
        }
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 400

    def test_400_for_empty_zip(self, server_env) -> None:
        _, client, _ = server_env
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w"):
            pass  # empty
        files = {"archive": ("empty.zip", buf.getvalue(), "application/zip")}
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 400

    def test_400_for_missing_manifest(self, server_env) -> None:
        """A zip with a project_id top-level dir but no manifest fails."""
        _, client, _ = server_env
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            # Need a syntactically valid project id top-level dir, otherwise
            # the format error fires earlier with a different message.
            zf.writestr("abcdef012345/junk.txt", "hello")
        files = {"archive": ("bad.zip", buf.getvalue(), "application/zip")}
        r = client.post("/api/projects/import-archive", files=files)
        assert r.status_code == 400
