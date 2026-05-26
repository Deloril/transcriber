"""End-to-end reachability tests for F1.2 (Source entity with persistence).

The pure data model lives in ``scribe/sources.py`` with unit tests in
``tests/test_sources.py``. The HTTP contract for
``/api/projects/<pid>/sources*`` is exercised in
``tests/test_server.py::TestSourcesAPI``. This file proves the
**user-facing surface** is wired together: GET ``/projects/<pid>``
renders an "+ Add source" button → GET
``/projects/<pid>/sources/add`` renders the source picker → submitting
the picker (POST ``/api/projects/<pid>/sources``) creates a Source
that the listing page (``/projects/<pid>/sources``) and coding view
(``/projects/<pid>/sources/<sid>``) can consume.

Why a separate file: per ``scripts/feature-implementer-prompt.md``
every feature gets a single, easy-to-find integration test that
exercises the end-to-end UI path. Keeping these per-feature makes the
audit trail trivial to reconstruct from ``git log``.

This is the F1.2 sibling of ``tests/test_server_projects.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads/outputs/projects."""
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


def _make_project(client: TestClient, name: str = "Holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Project home links to the sources surface
# --------------------------------------------------------------------------- #


class TestProjectHomeLinksToSources:
    """Without an obvious "+ Add source" CTA on the project home page,
    F1.2 isn't reachable in the user-facing sense — users would have
    to type the URL by hand."""

    def test_project_home_renders_add_source_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        # The hero CTA + Snapshot card both link to the picker.
        assert f'href="/projects/{pid}/sources/add"' in r.text
        assert "+ Add source" in r.text

    def test_project_home_renders_view_all_sources_link(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        # Snapshot card "View all" → /projects/<pid>/sources
        assert f'href="/projects/{pid}/sources"' in r.text

    def test_project_home_consumes_sources_json_api(
        self, server_env
    ) -> None:
        """The Sources snapshot card fetches the same endpoint that
        TestSourcesAPI in test_server.py covers."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        # The JS uses a backtick template literal; we just assert the
        # path appears in the rendered HTML.
        assert "/api/projects/${PROJECT_ID}/sources" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/sources — list page
# --------------------------------------------------------------------------- #


class TestSourcesListPage:
    """``/projects/<pid>/sources`` must render the table chrome, an
    "+ Add source" CTA, and the empty-state CTA so first-run users
    have a path forward."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert r.status_code == 200

    def test_has_add_source_action_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert "+ Add source" in r.text
        assert f'href="/projects/{pid}/sources/add"' in r.text

    def test_has_back_to_project_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        # Users must be able to escape the page without the back button.
        assert f'href="/projects/{pid}"' in r.text

    def test_empty_state_offers_add_cta(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        # The empty-state copy + CTA are the second discoverable path.
        assert "No sources attached yet" in r.text

    def test_consumes_the_json_api(self, server_env) -> None:
        """The page's loader fetches /api/projects/<pid>/sources — the
        same endpoint that test_server.py::TestSourcesAPI covers."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert "/api/projects/${PROJECT_ID}/sources" in r.text

    def test_active_nav_marks_projects(self, server_env) -> None:
        """The shell partial must light up the Projects nav so users
        understand where they are in the IA."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert 'class="active"' in r.text

    def test_invalid_project_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/..%2Fevil/sources")
        # Either a 400 or a 404 is acceptable; we just refuse to render
        # the wireframe with a malicious id.
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# /projects/<pid>/sources/add — picker page
# --------------------------------------------------------------------------- #


class TestSourcePickerPage:
    """The picker must render an obvious way to attach a transcription
    from the existing library: the table chrome + an explanation +
    a back link."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        assert r.status_code == 200

    def test_has_back_to_sources_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        assert f'href="/projects/{pid}/sources"' in r.text

    def test_picker_consumes_jobs_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        # The picker's loader hits /api/jobs to populate the table.
        assert '"/api/jobs"' in r.text or "'/api/jobs'" in r.text

    def test_picker_posts_to_sources_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        # The "Attach" button POSTs to the F1.2 sources endpoint.
        assert "/api/projects/${PROJECT_ID}/sources" in r.text
        assert '"POST"' in r.text or "'POST'" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/sources/<sid> — coding view
# --------------------------------------------------------------------------- #


class TestSourceCodingPage:
    """The coding view is the destination after Attach. It must render
    even before any data has loaded (the JS populates the rest)."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Ask for a non-existent source — the page is still rendered;
        # the JS shows a friendly error if the source can't be fetched.
        r = client.get(f"/projects/{pid}/sources/aaaaaaaaaaaa")
        assert r.status_code == 200

    def test_has_back_link_to_sources_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/aaaaaaaaaaaa")
        assert f'href="/projects/{pid}/sources"' in r.text

    def test_coding_view_loads_source_via_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/aaaaaaaaaaaa")
        # The JS calls GET /api/projects/<pid>/sources/<sid>; we just
        # assert the path is present in the rendered output (it lives
        # inside a backtick template literal).
        assert "/api/projects/${PROJECT_ID}/sources/${SOURCE_ID}" in r.text


# --------------------------------------------------------------------------- #
# End-to-end: picker-equivalent POST round-trips through the listing
# --------------------------------------------------------------------------- #


class TestCreateSourceRoundTrip:
    """Simulates what source_picker.html's ``attach()`` JS does on
    click: POST a payload built from a library row, then read it back
    via the listing endpoint that sources_list.html consumes."""

    def test_minimal_post_persists_and_lists(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client, name="Caregivers — pilot")

        # Step 1: equivalent of clicking Attach on a library row.
        payload = {
            "name": "Interview 01 — Saira",
            "source_type": "transcript",
            "transcript_job_id": "abcdef012345",
        }
        r = client.post(f"/api/projects/{pid}/sources", json=payload)
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "Interview 01 — Saira"
        assert created["project_id"] == pid
        # 12-hex-char id matches SOURCE_ID_RE.
        assert re.match(r"^[a-f0-9]{12}$", created["id"])

        # Step 2: the source is on disk where load_source expects it.
        on_disk = json.loads(
            (srv.PROJECTS_DIR / pid / "sources" / f"{created['id']}.json").read_text()
        )
        assert on_disk["name"] == created["name"]
        assert on_disk["transcript_job_id"] == "abcdef012345"

        # Step 3: the listing endpoint that sources_list.html fetches
        # surfaces the new source.
        r = client.get(f"/api/projects/{pid}/sources")
        assert r.status_code == 200
        ids = [s["id"] for s in r.json()["sources"]]
        assert created["id"] in ids

        # Step 4: the project_home Sources snapshot uses the same
        # endpoint — same data should appear there as well.
        r = client.get(f"/api/projects/{pid}/sources")
        assert any(s["id"] == created["id"] for s in r.json()["sources"])

        # Step 5: navigating to /projects/<pid>/sources/<sid> (what the
        # rendered row links to) returns 200 and renders the coding
        # view bound to the new source.
        r = client.get(f"/projects/{pid}/sources/{created['id']}")
        assert r.status_code == 200

    def test_full_post_persists_all_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        payload = {
            "name": "Field notes — Day 3",
            "source_type": "field_notes",
            "language": "en-AU",
            "recording_date": "2026-05-12",
            "notes": "Visited site B. Observed handover ritual.",
            "custom_attributes": {"site": "B", "shift": "PM"},
        }
        r = client.post(f"/api/projects/{pid}/sources", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        # Every field round-trips. Without this the picker's UX promise
        # (that what you typed is what gets saved) breaks silently.
        assert body["name"] == payload["name"]
        assert body["source_type"] == "field_notes"
        assert body["language"] == "en-AU"
        assert body["recording_date"] == "2026-05-12"
        assert body["notes"].startswith("Visited site B")
        assert body["custom_attributes"] == {"site": "B", "shift": "PM"}

    def test_listing_renders_source_after_create(self, server_env) -> None:
        """The list page is JS-populated, so we can't assert the row HTML
        from the static template — but we can assert the API the page
        will fetch returns the newly created row."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/sources",
            json={"name": "Interview 02"},
        )
        assert r.status_code == 201
        sid = r.json()["id"]

        # Page loads (200), and the API the JS consumes returns the
        # source we just made.
        page = client.get(f"/projects/{pid}/sources")
        assert page.status_code == 200

        api = client.get(f"/api/projects/{pid}/sources")
        assert api.status_code == 200
        rows = api.json()["sources"]
        assert any(s["id"] == sid and s["name"] == "Interview 02" for s in rows)


# --------------------------------------------------------------------------- #
# Cascade: deleting a project deletes its sources
# --------------------------------------------------------------------------- #


class TestSourceCascadeOnProjectDelete:
    """The on-disk layout (``projects/<pid>/sources/<sid>.json``)
    promises that deleting a project removes its sources as a side
    effect. The user surface implicitly relies on this — without it,
    "delete project" would orphan source data and the next project
    with the same id would inherit ghost sources."""

    def test_delete_project_removes_source_files(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/sources",
            json={"name": "Doomed source"},
        )
        assert r.status_code == 201
        sid = r.json()["id"]
        path = srv.PROJECTS_DIR / pid / "sources" / f"{sid}.json"
        assert path.exists()

        r = client.delete(f"/api/projects/{pid}")
        # Existing handler returns 200 with a JSON body; either status
        # is acceptable so long as the cascade actually happened.
        assert r.status_code in (200, 204), r.text

        # The cascade is structural: the parent dir is gone.
        assert not path.exists()
