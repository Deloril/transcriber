"""End-to-end reachability tests for F9.4 (Project checkpoints).

The pure module ``scribe.project_checkpoints`` shipped in 37f47ff with
extensive unit tests covering the dataclass, atomic save, listing,
sha256 verification, and the high-level :func:`create_project_checkpoint`
helper that emits an F9.1 audit event. That commit explicitly
deferred the HTTP / FastAPI surface; until these endpoints + the
checkpoints panel on the audit page landed, the only path to take or
read a project checkpoint was via the Python module directly.

This file covers the F9.4 read + write surface end to end:

  * GET  /api/projects/<pid>/checkpoints                       — list
  * POST /api/projects/<pid>/checkpoints                       — create
  * GET  /api/projects/<pid>/checkpoints/<cid>                 — fetch one
  * GET  /api/projects/<pid>/checkpoints/<cid>/archive         — download body
  * The /projects/<pid>/audit page renders the F9.4 checkpoints panel
    (``data-test-feature="F9.4"``) so the routes are reachable from
    the user-facing surface, not just curl.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.codes import Code, save_code
from scribe.project_checkpoints import (
    CHECKPOINT_ARCHIVE_SUFFIX,
    CHECKPOINT_ID_RE,
    list_checkpoints,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up an isolated TestClient with tmp project / upload / output dirs."""
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


def _make_project(client: TestClient, name: str = "CkProj") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_code(
    projects_root: Path, project_id: str, *, name: str = "Caring"
) -> Code:
    """Persist one Code in the project so checkpoints have something to capture."""
    c = Code.new(project_id=project_id, name=name, definition="seed")
    save_code(projects_root, c)
    return c


# --------------------------------------------------------------------------- #
# Audit page renders the F9.4 panel
# --------------------------------------------------------------------------- #


class TestCheckpointsPanelRenders:
    def test_audit_page_includes_checkpoints_panel(self, server_env) -> None:
        """The /projects/<pid>/audit page must render the F9.4 checkpoints
        panel below the F9.3 snapshots panel. The panel is identified
        by ``data-test-feature="F9.4"`` and ``data-test-id="checkpoints-panel"``.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        body = r.text
        assert 'data-test-feature="F9.4"' in body
        assert 'data-test-id="checkpoints-panel"' in body
        # Form controls render
        assert 'data-test-id="ck-name-input"' in body
        assert 'data-test-id="ck-desc-input"' in body
        assert 'data-test-id="ck-save-btn"' in body
        # List + empty state render
        assert 'data-test-id="ck-list"' in body
        assert 'data-test-id="ck-empty"' in body

    def test_audit_page_links_to_checkpoints_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        # The JS wires /api/projects/{pid}/checkpoints; just confirm
        # the substring appears in the rendered body.
        assert "/checkpoints" in r.text


# --------------------------------------------------------------------------- #
# GET /checkpoints — listing
# --------------------------------------------------------------------------- #


class TestListCheckpoints:
    def test_empty_project_returns_empty_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/checkpoints")
        assert r.status_code == 200
        body = r.json()
        assert body == {"checkpoints": [], "total": 0}

    def test_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        # Properly-formed 12-hex id that doesn't exist
        r = client.get("/api/projects/" + ("0" * 12) + "/checkpoints")
        assert r.status_code == 404

    def test_invalid_project_id_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex/checkpoints")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# POST /checkpoints — creation
# --------------------------------------------------------------------------- #


class TestCreateCheckpoint:
    def test_create_checkpoint_persists_and_returns_summary(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Caring")

        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={
                "name": "Pre-merge",
                "description": "Before merging duplicates",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        cp = body["checkpoint"]
        assert CHECKPOINT_ID_RE.match(cp["id"])
        assert cp["name"] == "Pre-merge"
        assert cp["description"] == "Before merging duplicates"
        # Archive metadata is non-trivial: archive_filename ends with
        # the .scribe.zip suffix and archive_bytes > 0.
        assert cp["archive_filename"].endswith(CHECKPOINT_ARCHIVE_SUFFIX)
        assert cp["archive_bytes"] > 0
        # SHA-256 is 64 hex chars
        assert len(cp["archive_sha256"]) == 64
        # Audit event was emitted and back-written onto the checkpoint.
        assert cp["event_id"]
        assert len(cp["event_id"]) == 12

        # And the persisted file actually exists on disk.
        cps_on_disk = list_checkpoints(tmp_path / "projects", pid)
        assert len(cps_on_disk) == 1
        assert cps_on_disk[0].name == "Pre-merge"

    def test_create_checkpoint_emits_f91_audit_event(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "milestone-1"},
        )
        assert r.status_code == 201
        # The F9.1 events feed should now contain a 'checkpoint' action.
        r2 = client.get(
            f"/api/projects/{pid}/events", params={"action": "checkpoint"}
        )
        assert r2.status_code == 200
        events = r2.json()["events"]
        assert len(events) == 1
        ev = events[0]
        assert ev["action"] == "checkpoint"
        assert ev["entity_type"] == "checkpoint"

    def test_create_checkpoint_blank_name_rejected_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "   "},
        )
        assert r.status_code == 400

    def test_create_checkpoint_missing_name_rejected_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(f"/api/projects/{pid}/checkpoints", json={})
        assert r.status_code == 400

    def test_create_checkpoint_oversized_name_rejected_400(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "x" * 5000},
        )
        assert r.status_code == 400

    def test_create_checkpoint_invalid_actor_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "ok", "actor_coder_id": "not-hex-id-here"},
        )
        assert r.status_code == 400

    def test_create_checkpoint_invalid_parent_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "ok", "parent_checkpoint_id": "not-hex"},
        )
        assert r.status_code == 400

    def test_create_checkpoint_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/" + ("0" * 12) + "/checkpoints",
            json={"name": "ok"},
        )
        assert r.status_code == 404

    def test_non_json_body_rejected(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/checkpoints",
            content="not-json-at-all",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 400

    def test_then_listed_via_get(self, server_env) -> None:
        """End-to-end: POST creates → GET sees it."""
        _, client, _ = server_env
        pid = _make_project(client)
        r1 = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "phase 1 complete"},
        )
        assert r1.status_code == 201
        cid = r1.json()["checkpoint"]["id"]

        r2 = client.get(f"/api/projects/{pid}/checkpoints")
        assert r2.status_code == 200
        body = r2.json()
        assert body["total"] == 1
        assert body["checkpoints"][0]["id"] == cid
        assert body["checkpoints"][0]["name"] == "phase 1 complete"


# --------------------------------------------------------------------------- #
# GET /checkpoints/{cid} — fetch one
# --------------------------------------------------------------------------- #


class TestGetOneCheckpoint:
    def test_get_one_returns_full_checkpoint_metadata(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Caring")

        r1 = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "with codes", "description": "a why"},
        )
        cid = r1.json()["checkpoint"]["id"]

        r2 = client.get(f"/api/projects/{pid}/checkpoints/{cid}")
        assert r2.status_code == 200
        cp = r2.json()["checkpoint"]
        assert cp["id"] == cid
        assert cp["name"] == "with codes"
        assert cp["description"] == "a why"
        # Component counts surface non-trivial data.
        assert isinstance(cp["component_counts"], dict)
        # We have 1 code.
        assert cp["component_counts"].get("codes") == 1

    def test_unknown_checkpoint_id_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/checkpoints/" + ("a" * 12)
        )
        assert r.status_code == 404

    def test_invalid_checkpoint_id_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/checkpoints/not-hex")
        assert r.status_code == 400

    def test_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get(
            "/api/projects/" + ("0" * 12) + "/checkpoints/" + ("a" * 12)
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# GET /checkpoints/{cid}/archive — download body
# --------------------------------------------------------------------------- #


class TestDownloadCheckpointArchive:
    def test_download_returns_zip_with_correct_headers(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client, name="My Project")
        r1 = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "first"},
        )
        cid = r1.json()["checkpoint"]["id"]

        r2 = client.get(f"/api/projects/{pid}/checkpoints/{cid}/archive")
        assert r2.status_code == 200
        # The body is a zip we can open.
        assert r2.headers["content-type"] == "application/zip"
        # filename header includes the project slug and the short id
        cd = r2.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "my-project" in cd.lower()
        assert cid[:8] in cd
        # Body parses as a zip
        zf = zipfile.ZipFile(BytesIO(r2.content))
        names = zf.namelist()
        assert any("project.json" in n for n in names)

    def test_download_archive_sha256_matches_metadata(
        self, server_env
    ) -> None:
        """The bytes returned by the download endpoint hash to the
        SHA-256 the metadata sidecar recorded — the audit-trail
        guarantee that researchers can verify off-line."""
        import hashlib

        _, client, _ = server_env
        pid = _make_project(client)
        r1 = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "verify-me"},
        )
        cp = r1.json()["checkpoint"]
        cid = cp["id"]
        expected_sha = cp["archive_sha256"]

        r2 = client.get(f"/api/projects/{pid}/checkpoints/{cid}/archive")
        assert r2.status_code == 200
        actual_sha = hashlib.sha256(r2.content).hexdigest()
        assert actual_sha == expected_sha

    def test_unknown_checkpoint_id_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/checkpoints/" + ("a" * 12) + "/archive"
        )
        assert r.status_code == 404

    def test_invalid_checkpoint_id_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/checkpoints/not-hex/archive")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Audit timeline integrates with the new feature
# --------------------------------------------------------------------------- #


class TestAuditTimelineIntegration:
    def test_checkpoint_appears_on_events_feed(self, server_env) -> None:
        """A freshly-created checkpoint surfaces immediately on the F9.1
        events feed — the same feed the audit timeline reads — without
        the user needing to reload."""
        _, client, _ = server_env
        pid = _make_project(client)
        r1 = client.post(
            f"/api/projects/{pid}/checkpoints",
            json={"name": "show-up", "description": "evidence"},
        )
        assert r1.status_code == 201
        cid = r1.json()["checkpoint"]["id"]

        r2 = client.get(
            f"/api/projects/{pid}/events",
            params={"entity_type": "checkpoint"},
        )
        assert r2.status_code == 200
        events = r2.json()["events"]
        assert len(events) == 1
        ev = events[0]
        assert ev["entity_id"] == cid
        # The event 'after' payload should at least carry the name
        # (so the timeline can render it without a second fetch).
        after = ev.get("after") or {}
        if isinstance(after, dict):
            assert after.get("name") == "show-up"
