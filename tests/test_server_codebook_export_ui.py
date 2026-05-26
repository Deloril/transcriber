"""Tests for the F2.6 codebook export UI surface.

The pure exporters (`scribe.codebook_export`) ship four formats —
CSV, Markdown, RTF, and REFI-QDA Codebook XML — and the FastAPI
download endpoints have lived at ``/api/projects/<pid>/codebook/export``
(F6.1) and ``/api/projects/<pid>/codebook/refi-qda-xml`` (F6.5) for a
while. The audit row in PLANNING.md (W2.5) called out that those
endpoints could "only be hit via curl." This test file proves the
codebook editor template now ships an Export dropdown that hits all
four targets, and that the dropdown's links exercise the routes
without 5xxs.

Concerns:

  1. The codebook editor page renders a Export menu that names all
     four formats (CSV / Markdown / Word / REFI-QDA XML) and points
     at the right URLs.
  2. Each href the menu surfaces resolves to a 200 against the live
     server, with the right ``Content-Type`` per format. (We already
     have deeper coverage of the export bodies in
     ``tests/test_server.py::TestExportCodebookAPI`` and
     ``::TestExportCodebookRefiQdaXmlAPI``; this file's job is purely
     to assert reachability through the UI surface.)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Test client + helpers
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
    return TestClient(srv.app), projects


def _new_project(client: TestClient, name: str = "Pilot study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_code(projects_dir: Path, project_id: str, name: str = "Pacing") -> str:
    """Persist one code so the export endpoint has something to render."""
    from scribe import codes as _codes
    from scribe.codes import Code

    code = Code.new(
        project_id=project_id,
        name=name,
        definition="Adjusting daily activity to manage limited energy.",
    )
    _codes.save_code(projects_dir, code)
    return code.id


# --------------------------------------------------------------------------- #
# Template render: the dropdown is in the page
# --------------------------------------------------------------------------- #


class TestCodebookEditorRendersExportMenu:
    """The codebook editor must surface F2.6's four export targets so the
    download routes are reachable without leaving the page.

    These assertions run against the rendered HTML so a refactor that
    quietly drops the menu fails here, not silently in a follow-up
    iteration."""

    def test_page_renders_with_export_button(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        text = r.text

        # The export button + the menu container.
        assert 'id="cb-export-btn"' in text
        assert 'id="cb-export-menu"' in text
        # Tagged with the F2.6 feature ref so future audits can find it.
        assert 'data-test-feature="F2.6"' in text
        # Header label visible in the page so a snapshot test would
        # catch a removal.
        assert "Export" in text

    def test_menu_lists_all_four_formats(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        text = r.text

        # All four format labels appear in the rendered dropdown.
        assert "CSV" in text
        assert "Markdown" in text
        # We label the RTF item "Word (RTF)" because that's how
        # researchers think about it; the hint clarifies.
        assert "Word (RTF)" in text or "Word" in text
        assert "REFI-QDA" in text

    def test_menu_links_target_correct_endpoints(self, env) -> None:
        """The four ``href``s must point at the live download URLs.

        We assert presence of each canonical URL form so a typo would
        fail here rather than leaking a 404 to the user."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        text = r.text

        assert f"/api/projects/{pid}/codebook/export?format=csv" in text
        assert f"/api/projects/{pid}/codebook/export?format=markdown" in text
        assert f"/api/projects/{pid}/codebook/export?format=rtf" in text
        # F6.5's REFI-QDA Codebook XML download lives on its own URL.
        assert f"/api/projects/{pid}/codebook/refi-qda-xml" in text

    def test_menu_items_have_download_attribute(self, env) -> None:
        """``download`` on the link tells the browser to save the body
        rather than render it inline (especially relevant for the XML
        target, which a browser would otherwise try to display)."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        text = r.text
        # The four ``download`` attributes are on the four <a> items
        # inside the dropdown. Use a coarse count to allow for future
        # additions without rewriting the assertion.
        assert text.count('class="cb-export-item"') >= 4
        # And the literal ``download`` attribute is present on items.
        # We don't pin the exact count to avoid being brittle if other
        # links elsewhere later add the attribute.
        assert 'download' in text


# --------------------------------------------------------------------------- #
# Reachability: the URLs from the menu actually resolve
# --------------------------------------------------------------------------- #


class TestCodebookExportUiHrefsAreLive:
    """Every URL the dropdown advertises must respond 200 with the
    expected content-type. If the menu shipped a typo or a stale path
    these assertions would catch it."""

    def test_csv_href_returns_csv(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid, name="Pacing")
        r = client.get(f"/api/projects/{pid}/codebook/export?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        # The body actually contains the seeded code.
        assert "Pacing" in r.text

    def test_markdown_href_returns_markdown(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid, name="Pacing")
        r = client.get(
            f"/api/projects/{pid}/codebook/export?format=markdown"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/markdown")
        assert r.text.startswith("# Codebook")
        assert "## Pacing" in r.text

    def test_rtf_href_returns_rtf(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid, name="Pacing")
        r = client.get(f"/api/projects/{pid}/codebook/export?format=rtf")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/rtf"
        # RTF preamble sanity check.
        assert r.text.startswith(r"{\rtf1")

    def test_refi_qda_xml_href_returns_xml(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid, name="Pacing")
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/xml")
        # REFI-QDA Codebook 1.0 root element + namespace.
        assert "<CodeBook" in r.text
        assert "urn:QDA-XML:codebook:1.0" in r.text

    def test_all_four_have_attachment_disposition(self, env) -> None:
        """Each download URL must set ``Content-Disposition: attachment``
        so the browser shows a save dialog rather than rendering."""
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid)
        urls = [
            f"/api/projects/{pid}/codebook/export?format=csv",
            f"/api/projects/{pid}/codebook/export?format=markdown",
            f"/api/projects/{pid}/codebook/export?format=rtf",
            f"/api/projects/{pid}/codebook/refi-qda-xml",
        ]
        for url in urls:
            r = client.get(url)
            assert r.status_code == 200, (url, r.text)
            cd = r.headers.get("content-disposition", "")
            assert "attachment" in cd, (url, cd)
            # Filename is slugged from the project name.
            assert "pilot-study-codebook" in cd, (url, cd)


# --------------------------------------------------------------------------- #
# Empty codebook still renders + exports cleanly
# --------------------------------------------------------------------------- #


class TestEmptyCodebookExport:
    """A brand-new project with zero codes should still render the
    Export menu and respond 200 from each URL — the four exporters are
    documented to handle empty input."""

    def test_empty_project_page_still_has_menu(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200
        assert 'id="cb-export-btn"' in r.text

    def test_empty_csv_is_header_only(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=csv")
        assert r.status_code == 200
        assert r.text.startswith("id,name,definition")

    def test_empty_refi_qda_xml_is_minimal(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        assert r.status_code == 200
        # Schema-valid empty CodeBook with a self-closing or
        # explicit-empty <Codes/>.
        assert "<CodeBook" in r.text
        assert "Codes" in r.text
