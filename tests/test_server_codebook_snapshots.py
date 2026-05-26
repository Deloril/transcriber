"""End-to-end reachability tests for F9.3 (Named codebook snapshots).

The pure module ``scribe.codebook_snapshots`` shipped in 33d3981 with
70 unit tests covering the dataclass, atomic save, listing, version
pinning, and the high-level :func:`create_codebook_snapshot` helper
that emits an F9.1 audit event. That commit explicitly deferred the
HTTP / FastAPI surface; until these endpoints + the snapshots panel
on the audit page landed, the only path to take or read a snapshot
was via the Python module directly.

This file covers the F9.3 read + write surface end to end:

  * GET  /api/projects/<pid>/snapshots                  — list
  * POST /api/projects/<pid>/snapshots                  — create
  * GET  /api/projects/<pid>/snapshots/<sid>            — fetch one
  * GET  /api/projects/<pid>/snapshots/<sid>/codebook?format=...
  * The /projects/<pid>/audit page renders the F9.3 snapshots panel
    (``data-test-feature="F9.3"``) so the routes are reachable from
    the user-facing surface, not just curl.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.codebook_snapshots import (
    SNAPSHOT_ID_RE,
    list_snapshots,
)
from scribe.codes import Code, save_code


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


def _make_project(client: TestClient, name: str = "SnapP") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_code(
    projects_root: Path, project_id: str, *, name: str = "Suffering"
) -> Code:
    """Persist one Code in the project so snapshots have something to capture."""
    c = Code.new(project_id=project_id, name=name, definition="test code")
    save_code(projects_root, c)
    return c


# --------------------------------------------------------------------------- #
# Audit page renders the F9.3 panel
# --------------------------------------------------------------------------- #


class TestSnapshotsPanelRenders:
    def test_audit_page_includes_snapshots_panel(self, server_env) -> None:
        """The /projects/<pid>/audit page must render the F9.3 snapshots
        panel above the events feed. The panel is identified by
        ``data-test-feature="F9.3"`` and ``data-test-id="snapshots-panel"``.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        body = r.text
        assert 'data-test-feature="F9.3"' in body
        assert 'data-test-id="snapshots-panel"' in body
        # Form controls render
        assert 'data-test-id="snap-name-input"' in body
        assert 'data-test-id="snap-desc-input"' in body
        assert 'data-test-id="snap-save-btn"' in body
        # List + empty state render
        assert 'data-test-id="snap-list"' in body
        assert 'data-test-id="snap-empty"' in body

    def test_audit_page_links_to_snapshots_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        assert "/snapshots" in r.text


# --------------------------------------------------------------------------- #
# GET /snapshots — listing
# --------------------------------------------------------------------------- #


class TestListSnapshots:
    def test_empty_project_returns_empty_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/snapshots")
        assert r.status_code == 200
        body = r.json()
        assert body == {"snapshots": [], "total": 0}

    def test_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        # Properly-formed 12-hex id that doesn't exist
        r = client.get("/api/projects/" + ("0" * 12) + "/snapshots")
        assert r.status_code == 404

    def test_invalid_project_id_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex/snapshots")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# POST /snapshots — creation
# --------------------------------------------------------------------------- #


class TestCreateSnapshot:
    def test_create_snapshot_persists_and_returns_summary(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Suffering")
        _seed_code(tmp_path / "projects", pid, name="Resilience")

        r = client.post(
            f"/api/projects/{pid}/snapshots",
            json={
                "name": "Initial coding done",
                "description": "Before unlocking the codebook",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        snap = body["snapshot"]
        assert SNAPSHOT_ID_RE.match(snap["id"])
        assert snap["name"] == "Initial coding done"
        assert snap["description"] == "Before unlocking the codebook"
        assert snap["code_count"] == 2
        # Audit event was emitted and back-written onto the snapshot.
        assert snap["event_id"]
        assert len(snap["event_id"]) == 12

        # And the persisted file actually exists on disk.
        snaps_on_disk = list_snapshots(tmp_path / "projects", pid)
        assert len(snaps_on_disk) == 1
        assert snaps_on_disk[0].name == "Initial coding done"

    def test_create_snapshot_emits_f91_audit_event(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "milestone-1"},
        )
        assert r.status_code == 201
        # The F9.1 events feed should now contain a 'snapshot' action.
        r2 = client.get(
            f"/api/projects/{pid}/events", params={"action": "snapshot"}
        )
        assert r2.status_code == 200
        events = r2.json()["events"]
        assert len(events) == 1
        ev = events[0]
        assert ev["action"] == "snapshot"
        assert ev["entity_type"] == "snapshot"

    def test_create_snapshot_blank_name_rejected_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "   "},
        )
        assert r.status_code == 400

    def test_create_snapshot_missing_name_rejected_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(f"/api/projects/{pid}/snapshots", json={})
        assert r.status_code == 400

    def test_create_snapshot_oversized_name_rejected_400(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "x" * 5000},
        )
        assert r.status_code == 400

    def test_create_snapshot_invalid_actor_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "ok", "actor_coder_id": "not-hex-id-here"},
        )
        assert r.status_code == 400

    def test_create_snapshot_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/" + ("0" * 12) + "/snapshots",
            json={"name": "ok"},
        )
        assert r.status_code == 404

    def test_non_json_body_rejected(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/snapshots",
            content="not-json-at-all",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 400

    def test_then_listed_via_get(self, server_env) -> None:
        """End-to-end: POST creates → GET sees it."""
        _, client, _ = server_env
        pid = _make_project(client)
        r1 = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "phase 1 complete"},
        )
        assert r1.status_code == 201
        sid = r1.json()["snapshot"]["id"]

        r2 = client.get(f"/api/projects/{pid}/snapshots")
        assert r2.status_code == 200
        body = r2.json()
        assert body["total"] == 1
        assert body["snapshots"][0]["id"] == sid
        assert body["snapshots"][0]["name"] == "phase 1 complete"


# --------------------------------------------------------------------------- #
# GET /snapshots/{sid} — fetch one
# --------------------------------------------------------------------------- #


class TestGetOneSnapshot:
    def test_get_one_returns_full_snapshot_with_codes(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Caring")

        r1 = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "with codes"},
        )
        sid = r1.json()["snapshot"]["id"]

        r2 = client.get(f"/api/projects/{pid}/snapshots/{sid}")
        assert r2.status_code == 200
        snap = r2.json()["snapshot"]
        assert snap["id"] == sid
        assert isinstance(snap["codes"], list)
        assert len(snap["codes"]) == 1
        assert snap["codes"][0]["name"] == "Caring"

    def test_unknown_snapshot_id_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/snapshots/" + ("a" * 12)
        )
        assert r.status_code == 404

    def test_invalid_snapshot_id_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/snapshots/not-hex-id")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GET /snapshots/{sid}/codebook?format=... — historical export
# --------------------------------------------------------------------------- #


class TestSnapshotCodebookExport:
    def test_csv_export_renders_snapshot_codes(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Noticing")

        r1 = client.post(
            f"/api/projects/{pid}/snapshots",
            json={"name": "csv-snap"},
        )
        sid = r1.json()["snapshot"]["id"]

        r2 = client.get(
            f"/api/projects/{pid}/snapshots/{sid}/codebook?format=csv"
        )
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("text/csv")
        assert "attachment" in r2.headers["content-disposition"]
        assert "snapshot-" in r2.headers["content-disposition"]
        # Body contains the snapshotted code name.
        assert "Noticing" in r2.text

    def test_markdown_export(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Witnessing")
        r1 = client.post(
            f"/api/projects/{pid}/snapshots", json={"name": "md-snap"}
        )
        sid = r1.json()["snapshot"]["id"]
        r2 = client.get(
            f"/api/projects/{pid}/snapshots/{sid}/codebook?format=markdown"
        )
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("text/markdown")
        assert "Witnessing" in r2.text

    def test_rtf_export(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        _seed_code(tmp_path / "projects", pid, name="Holding")
        r1 = client.post(
            f"/api/projects/{pid}/snapshots", json={"name": "rtf-snap"}
        )
        sid = r1.json()["snapshot"]["id"]
        r2 = client.get(
            f"/api/projects/{pid}/snapshots/{sid}/codebook?format=rtf"
        )
        assert r2.status_code == 200
        assert "rtf" in r2.headers["content-type"].lower()

    def test_unknown_format_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r1 = client.post(
            f"/api/projects/{pid}/snapshots", json={"name": "fmt-snap"}
        )
        sid = r1.json()["snapshot"]["id"]
        r2 = client.get(
            f"/api/projects/{pid}/snapshots/{sid}/codebook?format=xml"
        )
        assert r2.status_code == 400

    def test_unknown_snapshot_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/snapshots/" + ("a" * 12) + "/codebook?format=csv"
        )
        assert r.status_code == 404

    def test_export_is_historical_not_live_codebook(
        self, server_env
    ) -> None:
        """If a code is added *after* the snapshot, it must not appear
        in the snapshot's codebook export — that's the point of F9.3.
        """
        _, client, tmp_path = server_env
        pid = _make_project(client)
        # Code A exists at snapshot time.
        _seed_code(tmp_path / "projects", pid, name="Code-A-at-snapshot")
        r1 = client.post(
            f"/api/projects/{pid}/snapshots", json={"name": "before-B"}
        )
        sid = r1.json()["snapshot"]["id"]
        # Now add Code B after the snapshot.
        _seed_code(tmp_path / "projects", pid, name="Code-B-after-snapshot")
        # The snapshot's codebook export should still show only Code A.
        r2 = client.get(
            f"/api/projects/{pid}/snapshots/{sid}/codebook?format=csv"
        )
        assert r2.status_code == 200
        assert "Code-A-at-snapshot" in r2.text
        assert "Code-B-after-snapshot" not in r2.text
