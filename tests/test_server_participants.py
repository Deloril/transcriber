"""End-to-end reachability tests for F1.3 (Participant entity).

The pure data model lives in ``scribe/participants.py`` with unit tests
in ``tests/test_participants.py``. The HTTP API
(``/api/projects/<pid>/participants*``) is exercised by
``tests/test_server.py::TestParticipantsAPI``.

This file proves the **user-facing surface** is wired together: the
project home shows a Participants snapshot card → ``/projects/<pid>/
participants`` lists them → ``/projects/<pid>/participants/new``
renders a form that POSTs to ``/api/projects/<pid>/participants`` →
``/projects/<pid>/participants/<part_id>`` renders a detail page that
fetches and edits via the API.

Sibling of ``tests/test_server_sources.py`` (F1.2).
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
# Project home links to the participants surface
# --------------------------------------------------------------------------- #


class TestProjectHomeLinksToParticipants:
    """Without an obvious snapshot card on the project home page, F1.3
    isn't reachable in the user-facing sense — users would have to type
    the URL by hand."""

    def test_project_home_renders_participants_snapshot_card(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        # Snapshot card heading + counter element.
        assert "Participants" in r.text
        assert 'id="partCount"' in r.text
        assert 'id="partList"' in r.text

    def test_project_home_renders_participant_ctas(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        # Both "+ New participant" and "View all" link to the dedicated
        # pages — these are the discoverable paths in.
        assert f'href="/projects/{pid}/participants/new"' in r.text
        assert f'href="/projects/{pid}/participants"' in r.text
        assert "+ New participant" in r.text

    def test_project_home_consumes_participants_json_api(
        self, server_env
    ) -> None:
        """The snapshot card fetches the same endpoint that
        TestParticipantsAPI in test_server.py covers."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        # JS uses a backtick template literal; assert the path appears.
        assert "/api/projects/${PROJECT_ID}/participants" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/participants — list page
# --------------------------------------------------------------------------- #


class TestParticipantsListPage:
    """``/projects/<pid>/participants`` must render the table chrome,
    the "+ New participant" CTA, and a back link to the project home
    so first-run users have a path forward."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        assert r.status_code == 200

    def test_has_new_participant_action_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        assert "+ New participant" in r.text
        assert f'href="/projects/{pid}/participants/new"' in r.text

    def test_has_back_to_project_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        # Users must be able to escape the page without the back button.
        assert f'href="/projects/{pid}"' in r.text

    def test_empty_state_offers_create_cta(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        # The empty-state copy + CTA are the second discoverable path.
        assert "No participants yet" in r.text

    def test_consumes_the_json_api(self, server_env) -> None:
        """The page's loader fetches /api/projects/<pid>/participants —
        the same endpoint TestParticipantsAPI covers."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        assert "/api/projects/${PROJECT_ID}/participants" in r.text

    def test_active_nav_marks_projects(self, server_env) -> None:
        """The shell partial must light up the Projects nav so users
        understand where they are in the IA."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants")
        assert 'class="active"' in r.text

    def test_invalid_project_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/..%2Fevil/participants")
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# /projects/<pid>/participants/new — create form
# --------------------------------------------------------------------------- #


class TestParticipantNewPage:
    """The new-participant form must render every field on the
    Participant entity that a researcher can set, with a Cancel link
    back to the list."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/new")
        assert r.status_code == 200

    def test_has_back_to_participants_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/new")
        assert f'href="/projects/{pid}/participants"' in r.text

    def test_form_has_all_participant_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/new")
        # The Participant dataclass exposes name, pseudonym, demographics,
        # notes. (source_ids is wired via the source coding flow.)
        assert 'id="np-name"' in r.text
        assert 'id="np-pseudonym"' in r.text
        assert 'id="np-notes"' in r.text
        assert 'id="demoRows"' in r.text
        # Add-demographic button so user can add k/v rows.
        assert 'id="addDemoBtn"' in r.text

    def test_form_posts_to_participants_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/new")
        # Submit handler POSTs to the F1.3 endpoint.
        assert "/api/projects/${PROJECT_ID}/participants" in r.text
        assert '"POST"' in r.text or "'POST'" in r.text

    def test_form_has_required_name_marker(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/new")
        # The "*" marker on the name field communicates required-ness.
        assert "required" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/participants/<part_id> — detail page
# --------------------------------------------------------------------------- #


class TestParticipantDetailPage:
    """The detail page is the destination after creation. It must
    render even before the JS fetch resolves (so error states show up
    correctly), with a back link to the list."""

    def test_renders_with_200_for_valid_id_shape(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/aaaaaaaaaaaa")
        assert r.status_code == 200

    def test_has_back_link_to_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/aaaaaaaaaaaa")
        assert f'href="/projects/{pid}/participants"' in r.text

    def test_loads_participant_via_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/aaaaaaaaaaaa")
        assert "/api/projects/${PROJECT_ID}/participants/${PARTICIPANT_ID}" in r.text

    def test_has_save_and_delete_buttons(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/aaaaaaaaaaaa")
        # PATCH + DELETE entry points; the danger-zone is gated behind
        # a confirm() but renders into the DOM unconditionally.
        assert "Save changes" in r.text
        assert "Delete participant" in r.text
        assert '"PATCH"' in r.text or "'PATCH'" in r.text
        assert '"DELETE"' in r.text or "'DELETE'" in r.text

    @pytest.mark.parametrize("bad", ["short", "TOO-LONG-AND-INVALID", "../etc"])
    def test_rejects_malformed_participant_id(
        self, server_env, bad: str
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/participants/{bad}")
        # Either a 400 (handler rejects) or 404 (router can't match) is
        # acceptable, but never 200 with a malformed id.
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# End-to-end: form-equivalent POST round-trips through the listing
# --------------------------------------------------------------------------- #


class TestCreateParticipantRoundTrip:
    """Simulates what participant_new.html's submit handler does:
    POST a payload, then read it back via the listing endpoint that
    participants_list.html consumes."""

    def test_minimal_post_persists_and_lists(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client, name="Caregivers — pilot")

        # Step 1: equivalent of submitting the new-participant form.
        payload = {"name": "P01"}
        r = client.post(
            f"/api/projects/{pid}/participants", json=payload
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "P01"
        assert created["project_id"] == pid
        assert re.match(r"^[a-f0-9]{12}$", created["id"])

        # Step 2: the participant is on disk under projects/<pid>/...
        on_disk = json.loads(
            (
                srv.PROJECTS_DIR
                / pid
                / "participants"
                / f"{created['id']}.json"
            ).read_text()
        )
        assert on_disk["name"] == "P01"

        # Step 3: the listing endpoint that participants_list.html
        # fetches surfaces the new participant.
        r = client.get(f"/api/projects/{pid}/participants")
        assert r.status_code == 200
        ids = [p["id"] for p in r.json()["participants"]]
        assert created["id"] in ids

        # Step 4: the project_home Participants snapshot uses the same
        # endpoint — same data should appear there too.
        r = client.get(f"/api/projects/{pid}/participants")
        assert any(p["id"] == created["id"] for p in r.json()["participants"])

        # Step 5: navigating to /projects/<pid>/participants/<part_id>
        # (what the rendered row links to) returns 200 and renders the
        # detail view bound to the new id.
        r = client.get(f"/projects/{pid}/participants/{created['id']}")
        assert r.status_code == 200

    def test_full_post_persists_all_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        payload = {
            "name": "Saira",
            "pseudonym": "Alex",
            "notes": "Snowballed from P02. Long-haul carer cohort.",
            "demographics": {
                "age_band": "30-44",
                "role": "informal carer",
            },
        }
        r = client.post(f"/api/projects/{pid}/participants", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        # Every field round-trips. Without this, the form's UX promise
        # (what you typed gets saved) breaks silently.
        assert body["name"] == "Saira"
        assert body["pseudonym"] == "Alex"
        assert body["notes"].startswith("Snowballed")
        assert body["demographics"] == {
            "age_band": "30-44",
            "role": "informal carer",
        }

    def test_listing_renders_participant_after_create(
        self, server_env
    ) -> None:
        """The list page is JS-populated, so we can't assert the row
        HTML from the static template — but we can assert the API the
        page will fetch returns the newly created row."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/participants", json={"name": "P02"}
        )
        assert r.status_code == 201
        part_id = r.json()["id"]

        page = client.get(f"/projects/{pid}/participants")
        assert page.status_code == 200

        api = client.get(f"/api/projects/{pid}/participants")
        assert api.status_code == 200
        rows = api.json()["participants"]
        assert any(p["id"] == part_id and p["name"] == "P02" for p in rows)

    def test_patch_round_trip_through_detail_page(self, server_env) -> None:
        """The detail page renders + the PATCH endpoint it calls
        actually updates the on-disk participant."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/participants",
            json={"name": "Initial"},
        )
        part_id = r.json()["id"]

        # Detail page renders for the live id.
        page = client.get(f"/projects/{pid}/participants/{part_id}")
        assert page.status_code == 200

        # PATCH equivalent of the Save button.
        r = client.patch(
            f"/api/projects/{pid}/participants/{part_id}",
            json={"name": "Updated", "pseudonym": "Anon"},
        )
        assert r.status_code == 200
        assert r.json()["name"] == "Updated"
        assert r.json()["pseudonym"] == "Anon"

        # Read-back via the same endpoint the detail page uses on load.
        r = client.get(f"/api/projects/{pid}/participants/{part_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "Updated"


# --------------------------------------------------------------------------- #
# Cascade: deleting a project deletes its participants
# --------------------------------------------------------------------------- #


class TestParticipantCascadeOnProjectDelete:
    """The on-disk layout (``projects/<pid>/participants/<part_id>.json``)
    promises that deleting a project removes its participants as a
    side effect, mirroring the F1.2 source cascade."""

    def test_delete_project_removes_participant_files(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/participants",
            json={"name": "Doomed participant"},
        )
        assert r.status_code == 201
        part_id = r.json()["id"]
        path = (
            srv.PROJECTS_DIR
            / pid
            / "participants"
            / f"{part_id}.json"
        )
        assert path.exists()

        r = client.delete(f"/api/projects/{pid}")
        assert r.status_code in (200, 204), r.text

        # The cascade is structural: the parent dir is gone.
        assert not path.exists()
