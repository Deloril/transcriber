"""End-to-end reachability tests for F1.1 (Project entity with persistence).

The pure data model lives in ``scribe/projects.py`` with unit tests in
``tests/test_projects.py``. The HTTP contract for ``/api/projects*`` is
exercised in ``tests/test_server.py::TestProjectsAPI``. This file proves
the **user-facing surface** is wired together: GET ``/projects`` renders
a "+ New project" button → GET ``/projects/new`` renders a form whose
fields match the Project entity → submitting the form (POST
``/api/projects``) creates a project that the listing page can consume.

Why a separate file: the Reachable-via gate (see
``scripts/feature-implementer-prompt.md``) requires that every feature
have a single, easy-to-find integration test that exercises the
end-to-end UI path. Keeping these tests grouped per-feature makes the
audit trail trivial to reconstruct from ``git log``.
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


# --------------------------------------------------------------------------- #
# Shell nav: /projects is reachable from every page
# --------------------------------------------------------------------------- #


class TestProjectsLinkInShell:
    """The shell partial (`_shell.html`) must include a Projects nav link.

    Without this, the only way to reach the projects index is to type
    the URL by hand. F1.1 isn't reachable in the user-facing sense if
    no chrome links to it.
    """

    def test_top_nav_links_to_projects_from_home(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert 'href="/projects"' in r.text

    def test_top_nav_links_to_projects_from_library(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        assert r.status_code == 200
        assert 'href="/projects"' in r.text

    def test_top_nav_links_to_projects_from_settings(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert 'href="/projects"' in r.text


# --------------------------------------------------------------------------- #
# /projects — index page renders a "+ New project" affordance
# --------------------------------------------------------------------------- #


class TestProjectsIndexPage:
    """``/projects`` is the entry point: it must render an obvious
    create-project affordance and call the JSON API to populate the table."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        assert r.status_code == 200

    def test_has_new_project_button_pointing_to_new_form(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        # The button label "+ New project" is the user-visible affordance
        # documented in PLANNING.md F1.1.
        assert "+ New project" in r.text
        assert 'href="/projects/new"' in r.text

    def test_has_empty_state_with_create_link(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        # The empty-state CTA also points to /projects/new so first-run
        # users have a second discoverable path.
        assert "Create your first project" in r.text
        assert 'href="/projects/new"' in r.text

    def test_consumes_the_json_api(self, server_env) -> None:
        """The page's loader fetches /api/projects — proves the UI calls
        the same endpoint that test_server.py::TestProjectsAPI covers."""
        _, client, _ = server_env
        r = client.get("/projects")
        assert 'fetch("/api/projects")' in r.text

    def test_active_nav_marks_projects(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects")
        # When viewing /projects the projects nav link should be active.
        # _shell.html toggles a CSS class for the active page.
        assert 'class="active"' in r.text


# --------------------------------------------------------------------------- #
# /projects/new — form fields match the Project entity
# --------------------------------------------------------------------------- #


class TestProjectNewPage:
    """The new-project form must surface every field that the Project
    entity exposes via POST /api/projects, in the order the entity
    documents them. No hidden surface area."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        assert r.status_code == 200

    def test_form_posts_to_api(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        # The form must fetch POST /api/projects, the same endpoint
        # tested in test_server.py::TestProjectsAPI::test_create_minimal.
        assert "fetch(\"/api/projects\"" in r.text or \
               "fetch('/api/projects'" in r.text
        assert "POST" in r.text

    def test_form_has_name_field(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        # `name` is the only required Project field; the input must exist.
        assert 'id="np-name"' in r.text
        assert "required" in r.text

    def test_form_has_research_question_field(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        assert 'id="np-rq"' in r.text

    def test_form_has_methodology_field(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        assert 'id="np-meth"' in r.text
        # At least the Charmaz option is offered (PLANNING.md core
        # principles #6: gerund-form, Charmaz-aligned defaults).
        assert "Charmaz" in r.text or "charmaz" in r.text

    def test_form_has_description_field(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        assert 'id="np-desc"' in r.text

    def test_cancel_link_returns_to_index(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/new")
        # Cancel must be safe — go back to /projects, not lose work
        # silently. PLANNING.md UX polish principle.
        assert 'href="/projects"' in r.text
        assert "Cancel" in r.text


# --------------------------------------------------------------------------- #
# End-to-end: form-equivalent POST round-trips through the listing
# --------------------------------------------------------------------------- #


class TestCreateProjectRoundTrip:
    """Simulates what the project_new.html JS does on submit: POST a
    payload built from the form fields, then read it back via the
    listing endpoint that projects_list.html consumes."""

    def test_minimal_post_persists_and_lists(self, server_env) -> None:
        srv, client, _ = server_env

        # Step 1: equivalent of user filling in just the required field.
        payload = {
            "name": "Caregiver burnout — pilot interviews",
            "research_question": "",
            "methodology": "",
            "description": "",
        }
        r = client.post("/api/projects", json=payload)
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "Caregiver burnout — pilot interviews"
        # 12-hex-char id matches PROJECT_ID_RE.
        assert re.match(r"^[a-f0-9]{12}$", created["id"])

        # Step 2: the project is on disk where load_project expects it.
        on_disk = json.loads(
            (srv.PROJECTS_DIR / created["id"] / "project.json").read_text()
        )
        assert on_disk["name"] == created["name"]

        # Step 3: the listing endpoint that projects_list.html fetches
        # surfaces the new project.
        r = client.get("/api/projects")
        assert r.status_code == 200
        ids = [p["id"] for p in r.json()["projects"]]
        assert created["id"] in ids

        # Step 4: navigating to /projects/{id} (what the rendered row
        # links to) returns 200 and the project home page.
        r = client.get(f"/projects/{created['id']}")
        assert r.status_code == 200

    def test_full_post_persists_all_fields(self, server_env) -> None:
        _, client, _ = server_env
        payload = {
            "name": "Consent in care",
            "research_question": "How do nurses interpret consent?",
            "methodology": "charmaz",
            "sensitising_concepts": ["agency", "structure"],
            "description": "Pilot — five interviews",
            "codebook_stage": "focused",
        }
        r = client.post("/api/projects", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        # Every field round-trips. Without this the form's UX promise
        # (that what you typed is what gets saved) breaks silently.
        assert body["name"] == payload["name"]
        assert body["research_question"] == payload["research_question"]
        assert body["methodology"] == payload["methodology"]
        assert body["sensitising_concepts"] == payload["sensitising_concepts"]
        assert body["description"] == payload["description"]
        assert body["codebook_stage"] == payload["codebook_stage"]

    def test_listing_orders_newest_modified_first_after_patch(
        self, server_env
    ) -> None:
        """The projects_list.html JS sorts by modified_at desc; the API
        returns the data the JS sorts. After an edit, the patched project
        must have a later modified_at than the unedited one — otherwise
        the user's most-recent work doesn't surface at the top."""
        _, client, _ = server_env
        a = client.post("/api/projects", json={"name": "A"}).json()
        b = client.post("/api/projects", json={"name": "B"}).json()
        # Touch A so its modified_at advances past B's.
        r = client.patch(f"/api/projects/{a['id']}", json={"name": "A2"})
        assert r.status_code == 200
        a2 = r.json()
        assert a2["modified_at"] >= b["modified_at"]

    def test_blank_name_is_rejected_with_400(self, server_env) -> None:
        """The form's client-side check for a blank name is mirrored on
        the server — the UI and API agree on "name is required"."""
        _, client, _ = server_env
        r = client.post("/api/projects", json={"name": "   "})
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Project home /projects/{id} — the post-create landing page
# --------------------------------------------------------------------------- #


class TestProjectHomeAfterCreate:
    """After POST /api/projects, the form JS redirects to
    /projects/{id}. That page must render successfully or the user
    sees a broken state immediately after a successful create."""

    def test_home_renders_for_real_project(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post(
            "/api/projects", json={"name": "Landing"}
        ).json()["id"]
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        # The project's name appears somewhere on the page.
        assert "Landing" in r.text

    def test_home_renders_for_unknown_project_id(self, server_env) -> None:
        """Wireframe stays usable for an unknown id — the page shows a
        generic shell, not a 500. (Documented behaviour in
        server.py::project_home_page.)"""
        _, client, _ = server_env
        r = client.get("/projects/aaaaaaaaaaaa")
        assert r.status_code == 200
