"""F6.4 reachability verification — REFI-QDA / QDPX project export.

The pure builder + endpoint shipped in 8f8e218 (``scribe.refi_qda_project``,
``GET /api/projects/<pid>/qdpx``); the F6.4 commit body did not include a
``Reachable-via:`` line and no template surfaced the download. This file is
the explicit reachability anchor for the user-facing surface:

  1. the project home page renders the F6.4 download button next to the
     existing F1.5 archive button so a researcher can find the
     interoperable-export action without typing the URL by hand,
  2. the project settings page renders an "Interoperable export
     (REFI-QDA / QDPX)" card with the same download link so users who
     start in settings (where the .scribe archive lives) also see it,
  3. the underlying ``GET /api/projects/<pid>/qdpx`` endpoint that both
     surfaces target still returns the expected zip + headers.

The deeper coverage of the QDPX builder lives in
``tests/test_refi_qda_project.py`` and
``tests/test_server.py::TestExportQdpxAPI``; this file's job is purely
the F6.4 UI-reachability contract.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated TestClient + tmp uploads/outputs/projects roots."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
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

    client = TestClient(srv.app)
    yield srv, client, projects, output


def _new_project(client: TestClient, name: str = "Pilot study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Template render: the project home page surfaces the F6.4 download button
# --------------------------------------------------------------------------- #


class TestProjectHomeRendersF6_4Button:
    """The project home actions bar must render the F6.4 QDPX export
    button. Without it the user has no clickable path to the
    interoperable export — the whole point of REFI-QDA is "no
    lock-in", which fails if the download is unreachable from the UI.
    """

    def test_qdpx_button_present(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200, r.text
        # The button is tagged with a stable test feature attribute.
        assert 'data-test-feature="F6.4"' in r.text
        # The button id is stable for end-to-end UI tests.
        assert 'id="exportQdpxBtn"' in r.text

    def test_qdpx_button_targets_correct_endpoint(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}")
        # The href points at the project-scoped QDPX endpoint.
        assert f'/api/projects/{pid}/qdpx' in r.text

    def test_qdpx_button_carries_download_attribute(self, env) -> None:
        """``download`` makes the browser save the zip rather than
        attempt to render it — REFI-QDA QDPX is a binary format."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}")
        # Locate the button block and assert it carries ``download``.
        idx = r.text.find('id="exportQdpxBtn"')
        assert idx > 0
        # Look at the surrounding span around the anchor tag.
        snippet = r.text[max(0, idx - 200): idx + 400]
        assert "download" in snippet
        assert "Export QDPX" in snippet


# --------------------------------------------------------------------------- #
# Template render: the project settings page surfaces the same export
# --------------------------------------------------------------------------- #


class TestProjectSettingsRendersF6_4Card:
    """The settings page already exposes the F1.5 .scribe archive
    download. F6.4 piggy-backs on that affordance with a sibling card
    so users who start in settings find both export formats together.
    """

    def test_qdpx_card_rendered(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert r.status_code == 200, r.text
        # Card carries the F6.4 feature tag.
        assert 'data-test-feature="F6.4"' in r.text
        # The data-test-id link anchor is present on the settings page.
        assert 'data-test-id="ps-qdpx-link"' in r.text

    def test_qdpx_link_targets_correct_endpoint(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert f'/api/projects/{pid}/qdpx' in r.text

    def test_qdpx_card_explains_interop_value(self, env) -> None:
        """The card copy must mention REFI-QDA so a user who landed on
        the page via search engines / docs can confirm this is the
        right button to click."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        text = r.text
        assert "REFI-QDA" in text
        assert "QDPX" in text


# --------------------------------------------------------------------------- #
# Endpoint reachability: clicking the rendered hrefs lands on a real zip
# --------------------------------------------------------------------------- #


class TestQdpxEndpointReachableFromUi:
    """The hrefs the templates render must resolve to the live QDPX
    endpoint. This is the link between the UI surface and the pure
    builder — without this end-to-end check, the rendered button could
    point at a 404 and the F6.4 surface would still 'render'."""

    def test_button_href_resolves_to_qdpx_zip(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        # Hit the same endpoint the rendered button targets.
        r = client.get(f"/api/projects/{pid}/qdpx")
        assert r.status_code == 200, r.text
        # Body is a zip — REFI-QDA QDPX is a renamed .zip.
        assert r.content[:2] == b"PK"
        # Vendor MIME type per the F6.4 endpoint.
        assert r.headers["content-type"].startswith("application/x-qdpx")

    def test_archive_carries_project_qde_manifest(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/qdpx")
        assert r.status_code == 200, r.text
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            # project.qde is the REFI-QDA manifest required by every
            # importer (Atlas.ti / MAXQDA / NVivo / etc.). If this is
            # missing, the QDPX is unreadable.
            assert "project.qde" in zf.namelist()

    def test_filename_is_slugged_for_save_dialog(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Care work — interviews 2025")
        r = client.get(f"/api/projects/{pid}/qdpx")
        assert r.status_code == 200, r.text
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        # Filename ends in .qdpx and is slugified (no spaces / em-dashes).
        assert ".qdpx" in cd
        assert " — " not in cd

    def test_404_when_project_missing(self, env) -> None:
        _, client, _, _ = env
        # Valid 12-hex shape, no project on disk.
        r = client.get(f"/api/projects/{'0' * 12}/qdpx")
        assert r.status_code == 404
