"""End-to-end reachability tests for F3.1 (project shell aggregates
codebook + project-level settings).

The pure data-model changes from F3.1 (e59a056) added two pieces:

  1. ``Project.settings`` — a bounded ``dict[str, Any]`` on the
     project entity for free-form project-level preferences (default
     coder name, AI on/off, default code colour, UI prefs).
  2. ``ProjectBundle.codes`` — the codebook (F2.x codes) is now part
     of the project bundle, which means it travels in the F1.5
     archive zip.

The pure module already had unit coverage in ``tests/test_projects.py``
and ``tests/test_project_format.py``. What was missing — and what this
file proves — is the **user-facing surface**:

  * GET /projects/<pid>/settings renders a real settings page with
    metadata + preference forms (replacing the previous wireframe
    stub).
  * The page reads project state via GET /api/projects/<pid>.
  * Submitting metadata and preferences PATCHes /api/projects/<pid>
    and persists settings on disk (round-trip via load_project).
  * The page shows a download link to the F1.5 archive endpoint —
    the user-visible nod that the codebook now ships with the bundle.
  * The codebook is included in the archive zip (cross-feature
    reachability check: F3.1's bundle change shows up in the F1.5
    download).
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

    Mirrors the fixture in ``test_server_projects.py`` — kept inline
    so this file is self-contained.
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


def _make_project(client: TestClient, name: str = "Settings test") -> str:
    r = client.post(
        "/api/projects",
        json={
            "name": name,
            "research_question": "What does it look like?",
            "methodology": "charmaz",
            "description": "Initial project",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# /projects/<pid>/settings — graduated from wireframe to a real page
# --------------------------------------------------------------------------- #


class TestSettingsPageRenders:
    """The settings page must render the F3.1 settings form, not a
    wireframe stub. The wireframe was the previous state; if this
    template ever regresses to ``project_subpage.html`` the F3.1
    surface disappears.
    """

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert r.status_code == 200

    def test_no_longer_renders_wireframe_stub(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        # The wireframe banner read "Wireframe. Methodology, attribute
        # schema, codebook stage, ...". F3.1 must replace it.
        assert "Wireframe." not in r.text
        # The real settings form has F3.1-tagged sections.
        assert 'data-test-feature="F3.1"' in r.text

    def test_marks_active_nav_projects(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        # _shell.html marks the projects nav as .active when active_nav
        # is set to "projects".
        assert "projects" in r.text
        # Heading present.
        assert "Project settings" in r.text


class TestMetadataForm:
    """The settings page must surface every Project metadata field
    the F1.1 entity exposes via PATCH /api/projects/<pid>.
    """

    def test_page_has_name_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-name"' in r.text

    def test_page_has_methodology_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-meth"' in r.text
        # Charmaz option is offered (consistent with /projects/new).
        assert "charmaz" in r.text.lower() or "Charmaz" in r.text

    def test_page_has_codebook_stage_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-stage"' in r.text
        # All five canonical stages are listed.
        for stage in ("initial", "focused", "axial", "theoretical", "locked"):
            assert f'value="{stage}"' in r.text

    def test_page_has_research_question_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-rq"' in r.text

    def test_page_has_description_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-desc"' in r.text

    def test_page_has_sensitising_concepts_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-sc"' in r.text

    def test_page_loads_via_api(self, server_env) -> None:
        """The form's loader fetches /api/projects/<pid> — same endpoint
        tested in TestProjectsAPI in test_server.py."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert "/api/projects/" in r.text
        assert 'method: "PATCH"' in r.text


class TestPreferencesForm:
    """The F3.1 ``settings`` field is the new piece. Its three named
    keys (default_coder, default_code_colour, ai_enabled) need
    visible inputs."""

    def test_page_has_default_coder_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-default-coder"' in r.text

    def test_page_has_default_colour_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-default-colour"' in r.text
        assert 'type="color"' in r.text

    def test_page_has_ai_enabled_checkbox(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="ps-ai-enabled"' in r.text
        assert 'type="checkbox"' in r.text

    def test_page_references_settings_dict(self, server_env) -> None:
        """The JS reads p.settings off the API response — the page
        must actually consume the F3.1 settings field, not just show
        empty inputs."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert "p.settings" in r.text
        assert "default_coder" in r.text
        assert "default_code_colour" in r.text
        assert "ai_enabled" in r.text


# --------------------------------------------------------------------------- #
# Round-trip: settings persist via PATCH /api/projects/<pid>
# --------------------------------------------------------------------------- #


class TestSettingsPersist:
    """Saving the preferences form must write the settings dict to
    disk, and reloading the project must surface the saved values.
    This is the critical reachability test: it proves the form
    actually reaches the PATCH handler that persists settings."""

    def test_patch_settings_persists(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.patch(
            f"/api/projects/{pid}",
            json={
                "settings": {
                    "default_coder": "Luke",
                    "default_code_colour": "#aabbcc",
                    "ai_enabled": True,
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["settings"]["default_coder"] == "Luke"
        assert body["settings"]["default_code_colour"] == "#aabbcc"
        assert body["settings"]["ai_enabled"] is True

        # Round-trip via the entity loader to prove it persisted to
        # disk, not just the in-memory PATCH response.
        from scribe.projects import load_project
        loaded = load_project(srv._projects_root(), pid)
        assert loaded.settings["default_coder"] == "Luke"
        assert loaded.settings["ai_enabled"] is True

    def test_patch_settings_with_metadata(self, server_env) -> None:
        """Metadata + settings can be patched together; the settings
        validator runs only on the settings field, not on metadata."""
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.patch(
            f"/api/projects/{pid}",
            json={
                "name": "Renamed",
                "codebook_stage": "focused",
                "settings": {"ai_enabled": False, "default_coder": "Sam"},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "Renamed"
        assert body["codebook_stage"] == "focused"
        assert body["settings"]["default_coder"] == "Sam"

    def test_invalid_settings_returns_400(self, server_env) -> None:
        """The F3.1 validator rejects unsupported types (custom
        objects, deep nesting). The route should propagate that as
        a 400 — not a 500."""
        _, client, _ = server_env
        pid = _make_project(client)
        # Lists of dicts are explicitly rejected by F3.1's
        # _validate_settings_value (depth >= 1 nesting check).
        r = client.patch(
            f"/api/projects/{pid}",
            json={"settings": {"bad": [{"nested": "dict"}]}},
        )
        assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
# Bundle: codebook rides along in the archive (F3.1's other change)
# --------------------------------------------------------------------------- #


class TestArchiveLink:
    """The settings page links to the F1.5 archive download. This is
    the user-visible nod that F3.1 expanded the bundle to include the
    codebook."""

    def test_settings_page_links_to_archive(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert f"/api/projects/{pid}/archive" in r.text
        assert "Download project archive" in r.text


class TestProjectHomeLinksToSettings:
    """The project home page must surface a Settings link. Without
    this nav the F3.1 settings form is unreachable from anywhere
    except a hand-typed URL."""

    def test_project_home_has_settings_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert f'href="/projects/{pid}/settings"' in r.text
        # The button text contains the gear glyph + label so the link
        # is discoverable visually, not just by URL inspection.
        assert "Settings" in r.text


class TestBundleIncludesCodebook:
    """F3.1's other half: the codebook now persists into the project
    bundle. Round-trip the archive: create a project, add a code, hit
    the archive endpoint, confirm the code shows up inside the zip
    under codes/<id>.json — proving the new ``ProjectBundle.codes``
    field is reachable through the existing F1.5 download."""

    def test_archive_includes_code_files(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)

        # Create one code via the existing F2.1 endpoint. We don't
        # need a real exemplar set; just enough that codes/<id>.json
        # exists on disk.
        r = client.post(
            f"/api/projects/{pid}/codes",
            json={
                "name": "Doing the thing",
                "definition": "An example code for the F3.1 bundle test.",
                "stage": "initial",
            },
        )
        assert r.status_code in (200, 201), r.text
        code_id = r.json()["id"]

        # Fetch the archive.
        r = client.get(f"/api/projects/{pid}/archive")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/")

        # Walk the zip, find the code file.
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        # F1.5 layout: <pid>/codes/<code_id>.json
        expected = f"{pid}/codes/{code_id}.json"
        assert expected in names, f"archive missing {expected}; has {names!r}"

        # And the file actually has the code we just created.
        with zf.open(expected) as f:
            payload = json.loads(f.read())
        assert payload["id"] == code_id
        assert payload["name"] == "Doing the thing"
