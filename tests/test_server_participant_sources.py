"""End-to-end reachability tests for F3.3 (participant ↔ source mapping).

The pure module ``scribe/participant_sources.py`` shipped with passing
unit tests in ``tests/test_participant_sources.py`` (45 cases). What
this file proves is that the **user-facing surface** is wired:

  * REST endpoints expose the inverse navigation, focus-group
    set-style update, single-edge link/unlink, and orphan listing.
  * The source-coding view's side panel renders a "Participants"
    section with an Edit button that hits the new endpoints.
  * The sources-list page renders a "Participants" column whose count
    comes from the new ``/participant_source_map`` endpoint.

Sibling of ``tests/test_server_participants.py`` (F1.3) and
``tests/test_server_sources.py`` (F1.2). The fixture mirrors theirs.
"""

from __future__ import annotations

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


def _make_project(client: TestClient, name: str = "Roster") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "Interview") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_participant(
    client: TestClient, pid: str, name: str = "P01"
) -> str:
    r = client.post(
        f"/api/projects/{pid}/participants",
        json={"name": name},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Coding view exposes the participants panel
# --------------------------------------------------------------------------- #


class TestSourceCodingPageRendersParticipantsPanel:
    """The coding view side panel must surface the participant roster
    + Edit button so a researcher can curate a focus group's roster
    without hand-editing JSON."""

    def test_coding_page_renders_participants_heading(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # The heading on the side panel is the user-facing affordance.
        assert 'data-test-feature="F3.3"' in r.text
        assert "Participants" in r.text
        # And the explicit Edit button.
        assert 'data-test-id="src-edit-participants"' in r.text
        # Save / cancel + checkbox list IDs the JS hangs off.
        assert 'id="partCheckboxList"' in r.text
        assert 'data-test-id="src-save-participants"' in r.text
        # The page hits the new endpoint by URL shape.
        assert (
            "/api/projects/${PROJECT_ID}/sources/${SOURCE_ID}/participants"
            in r.text
        )


class TestSourcesListRendersParticipantColumn:
    """The sources-list page must show a Participants column so a
    focus-group source (multiple participants) is visible at a glance."""

    def test_sources_list_renders_participants_column_header(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert r.status_code == 200
        assert 'data-test-id="src-list-parts-col"' in r.text
        # The page hits the new map endpoint to populate counts.
        assert (
            "/api/projects/${PROJECT_ID}/participant_source_map"
            in r.text
        )


# --------------------------------------------------------------------------- #
# REST: list participants for a source (inverse navigation)
# --------------------------------------------------------------------------- #


class TestListSourceParticipantsAPI:
    def test_empty_when_nothing_linked(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/api/projects/{pid}/sources/{sid}/participants")
        assert r.status_code == 200
        assert r.json() == {"participants": []}

    def test_lists_linked_participants_after_focus_group_save(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid, name="Focus group")
        a = _make_participant(client, pid, name="Alice")
        b = _make_participant(client, pid, name="Bob")
        # Set the focus group's roster in one PUT.
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, b]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["changed"] is True
        assert sorted(body["added"]) == sorted([a, b])
        assert body["removed"] == []
        # Inverse navigation surfaces the same two participants.
        r = client.get(f"/api/projects/{pid}/sources/{sid}/participants")
        assert r.status_code == 200
        ids = sorted(p["id"] for p in r.json()["participants"])
        assert ids == sorted([a, b])

    def test_404_on_missing_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # 12-char hex, valid shape but not on disk.
        r = client.get(
            f"/api/projects/{pid}/sources/0123456789ab/participants"
        )
        assert r.status_code == 404

    def test_400_on_malformed_source_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/sources/NOPE-NOT-HEX/participants"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# REST: PUT focus-group roster (set-style)
# --------------------------------------------------------------------------- #


class TestSetSourceParticipantsAPI:
    def test_idempotent_second_call_no_change(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid, name="A")
        b = _make_participant(client, pid, name="B")
        r1 = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, b]},
        )
        assert r1.status_code == 200
        assert r1.json()["changed"] is True
        # Second call, same desired set.
        r2 = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, b]},
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["changed"] is False
        assert body["added"] == []
        assert body["removed"] == []
        assert sorted(body["unchanged"]) == sorted([a, b])

    def test_diff_reports_added_and_removed(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid, name="A")
        b = _make_participant(client, pid, name="B")
        c = _make_participant(client, pid, name="C")
        # Initial roster: A + B.
        client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, b]},
        )
        # Swap to A + C.
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, c]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == [c]
        assert body["removed"] == [b]
        assert body["unchanged"] == [a]

    def test_empty_list_clears_roster(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid, name="A")
        client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a]},
        )
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": []},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["removed"] == [a]
        # And the inverse listing now empty.
        r2 = client.get(f"/api/projects/{pid}/sources/{sid}/participants")
        assert r2.json() == {"participants": []}

    def test_400_on_unknown_participant_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": ["aaaaaaaaaaaa"]},  # well-formed but unknown
        )
        assert r.status_code == 400

    def test_400_on_non_list_payload(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": "not a list"},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# REST: single-edge link / unlink
# --------------------------------------------------------------------------- #


class TestLinkUnlinkParticipantSourceAPI:
    def test_link_adds_then_already_linked_returns_added_false(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid)
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/participants/{a}"
        )
        assert r.status_code == 200
        assert r.json()["added"] is True
        # Idempotent — second POST is a no-op success.
        r2 = client.post(
            f"/api/projects/{pid}/sources/{sid}/participants/{a}"
        )
        assert r2.status_code == 200
        assert r2.json()["added"] is False
        # Confirmed by the inverse listing.
        listing = client.get(
            f"/api/projects/{pid}/sources/{sid}/participants"
        )
        assert [p["id"] for p in listing.json()["participants"]] == [a]

    def test_unlink_removes_then_idempotent(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid)
        client.post(
            f"/api/projects/{pid}/sources/{sid}/participants/{a}"
        )
        r = client.delete(
            f"/api/projects/{pid}/sources/{sid}/participants/{a}"
        )
        assert r.status_code == 200
        assert r.json()["removed"] is True
        r2 = client.delete(
            f"/api/projects/{pid}/sources/{sid}/participants/{a}"
        )
        assert r2.status_code == 200
        assert r2.json()["removed"] is False


# --------------------------------------------------------------------------- #
# REST: orphan-link audit + project-wide map
# --------------------------------------------------------------------------- #


class TestOrphanLinksAndMap:
    def test_map_includes_every_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        s1 = _make_source(client, pid, name="One")
        s2 = _make_source(client, pid, name="Two")
        a = _make_participant(client, pid)
        client.put(
            f"/api/projects/{pid}/sources/{s1}/participants",
            json={"participant_ids": [a]},
        )
        r = client.get(f"/api/projects/{pid}/participant_source_map")
        assert r.status_code == 200
        m = r.json()["map"]
        assert m[s1] == [a]
        assert m[s2] == []  # empty bucket present

    def test_orphan_after_source_delete(self, server_env) -> None:
        """Deleting a source while a participant references it should
        surface in the orphan-links endpoint. The participant's
        source_ids isn't mutated automatically — that's the auditor's
        decision."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid)
        client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a]},
        )
        # Delete the source out from under the participant.
        r = client.delete(f"/api/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # Orphan endpoint surfaces the dangling reference.
        r2 = client.get(f"/api/projects/{pid}/orphan_participant_links")
        assert r2.status_code == 200
        orphans = r2.json()["orphans"]
        assert orphans == [{"participant_id": a, "source_id": sid}]


# --------------------------------------------------------------------------- #
# Cross-cutting: forward + inverse views agree
# --------------------------------------------------------------------------- #


class TestForwardAndInverseAgree:
    """Setting a focus-group's roster via PUT must show up on each
    participant's ``source_ids`` field and in the inverse listing."""

    def test_forward_source_ids_match_inverse(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        a = _make_participant(client, pid, name="A")
        b = _make_participant(client, pid, name="B")
        client.put(
            f"/api/projects/{pid}/sources/{sid}/participants",
            json={"participant_ids": [a, b]},
        )
        # Forward direction: each participant carries the source.
        for part_id in (a, b):
            p = client.get(
                f"/api/projects/{pid}/participants/{part_id}"
            ).json()
            assert sid in p["source_ids"]
        # Inverse direction: the source's roster lists both.
        listing = client.get(
            f"/api/projects/{pid}/sources/{sid}/participants"
        ).json()
        assert sorted(p["id"] for p in listing["participants"]) == sorted([a, b])
