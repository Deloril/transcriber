"""End-to-end reachability tests for F9.1 (Append-only event log).

The F9.1 module ``scribe.event_log`` shipped the data plane in
7e4250d: the ``Event`` dataclass, the
``projects/<pid>/events/<eid>.json`` persistence layer, the closed
action / entity-type vocabularies, and the filter helpers. That commit
explicitly deferred the HTTP / FastAPI surface; until these endpoints
landed, the only path to read the F9.1 log was via the Python module
directly. This file covers the F9.1 read surface:

  * GET  /api/projects/<pid>/events                — list (filterable)
  * GET  /api/projects/<pid>/events/<eid>          — single event
  * The /projects/<pid>/audit page renders the F9.1 timeline UI
    (``data-test-feature="F9.1"``) so the route is reachable from the
    user-facing surface, not just curl.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.event_log import (
    EVENT_ACTION_CREATE,
    EVENT_ACTION_UPDATE,
    EVENT_ACTION_DELETE,
    EVENT_ACTION_LOCK,
    EVENT_ACTIONS,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_CODEBOOK,
    EVENT_ENTITY_MEMO,
    EVENT_ENTITY_TYPES,
    Event,
    new_event_id,
    save_event,
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


def _make_project(client: TestClient, name: str = "EvP") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


_HEX_ACTOR = "a" * 12
_HEX_ACTOR_2 = "b" * 12
_HEX_ENTITY = "c" * 12
_HEX_ENTITY_2 = "d" * 12


def _seed_event(
    projects_root: Path,
    project_id: str,
    *,
    action: str = EVENT_ACTION_CREATE,
    entity_type: str = EVENT_ENTITY_CODE,
    entity_id: str = _HEX_ENTITY,
    actor: str = _HEX_ACTOR,
    before: dict | None = None,
    after: dict | None = None,
    notes: str = "",
    now: str | None = None,
) -> Event:
    """Persist one Event against the project store."""
    ev = Event.new(
        project_id=project_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_coder_id=actor,
        before=before,
        after=after,
        notes=notes,
        now=now,
    )
    save_event(projects_root, ev)
    return ev


# --------------------------------------------------------------------------- #
# Audit timeline page (template / UI reachability)
# --------------------------------------------------------------------------- #


class TestAuditPageRenders:
    def test_audit_page_renders_real_template_not_wireframe(
        self, server_env
    ) -> None:
        """The /projects/<pid>/audit page must render the F9.1 timeline UI,
        not the wireframe stub. The template is identified by
        ``data-test-feature="F9.1"`` and ``data-test-id="audit-timeline"``.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        body = r.text
        # Real template markers
        assert 'data-test-feature="F9.1"' in body
        assert 'data-test-id="audit-timeline"' in body
        # Filter controls render
        assert 'data-test-id="audit-filter-action"' in body
        assert 'data-test-id="audit-filter-entity"' in body
        assert 'data-test-id="audit-filter-actor"' in body
        # Events list container renders
        assert 'data-test-id="audit-events-list"' in body
        # Wireframe stub markers must NOT appear
        assert "Wireframe." not in body
        assert "alert(&#39;Stub" not in body and "alert('Stub" not in body

    def test_audit_page_links_to_events_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        # The page wires its fetch() against the F9.1 events route.
        assert "/api/projects/" in r.text
        assert "/events?" in r.text or "/events`" in r.text or "events?" in r.text

    def test_audit_page_invalid_project_id_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/" + ("x" * 200) + "/audit")
        # Validator returns 400; routing layer may return 404. Either is fine.
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# GET /events — listing
# --------------------------------------------------------------------------- #


class TestListEvents:
    def test_empty_project_returns_empty_list_with_vocab(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/events")
        assert r.status_code == 200
        body = r.json()
        assert body["events"] == []
        assert body["total"] == 0
        assert body["returned"] == 0
        assert body["truncated"] is False
        assert body["order"] == "desc"
        # Filter-vocabulary contracts the UI relies on:
        assert set(body["available_actions"]) == set(EVENT_ACTIONS)
        assert set(body["available_entity_types"]) == set(EVENT_ENTITY_TYPES)

    def test_lists_persisted_events_newest_first_by_default(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(
            projects_root, pid,
            action=EVENT_ACTION_CREATE,
            now="2026-04-01T10:00:00Z",
        )
        e2 = _seed_event(
            projects_root, pid,
            action=EVENT_ACTION_UPDATE,
            now="2026-04-02T11:30:00Z",
        )
        r = client.get(f"/api/projects/{pid}/events")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert body["returned"] == 2
        ids = [ev["id"] for ev in body["events"]]
        # Newest-first per default order=desc
        assert ids == [e2.id, e1.id]

    def test_order_asc_returns_chronological(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(projects_root, pid, now="2026-04-01T10:00:00Z")
        e2 = _seed_event(projects_root, pid, now="2026-04-02T10:00:00Z")
        r = client.get(
            f"/api/projects/{pid}/events", params={"order": "asc"}
        )
        assert r.status_code == 200
        ids = [ev["id"] for ev in r.json()["events"]]
        assert ids == [e1.id, e2.id]

    def test_order_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/events", params={"order": "weird"}
        )
        assert r.status_code == 400

    def test_filter_by_action(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e_create = _seed_event(projects_root, pid, action=EVENT_ACTION_CREATE)
        _seed_event(projects_root, pid, action=EVENT_ACTION_UPDATE)
        _seed_event(projects_root, pid, action=EVENT_ACTION_DELETE)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"action": EVENT_ACTION_CREATE},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_create.id
        assert body["events"][0]["action"] == EVENT_ACTION_CREATE

    def test_filter_by_action_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"action": "totally-not-real"},
        )
        assert r.status_code == 400

    def test_filter_by_entity_type(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e_code = _seed_event(projects_root, pid, entity_type=EVENT_ENTITY_CODE)
        _seed_event(projects_root, pid, entity_type=EVENT_ENTITY_MEMO)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"entity_type": EVENT_ENTITY_CODE},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_code.id
        assert body["events"][0]["entity_type"] == EVENT_ENTITY_CODE

    def test_filter_by_entity_type_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"entity_type": "not-a-thing"},
        )
        assert r.status_code == 400

    def test_filter_by_entity_id(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(projects_root, pid, entity_id=_HEX_ENTITY)
        _seed_event(projects_root, pid, entity_id=_HEX_ENTITY_2)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"entity_id": _HEX_ENTITY},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e1.id

    def test_filter_by_actor_coder(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        _seed_event(projects_root, pid, actor=_HEX_ACTOR)
        e_other = _seed_event(projects_root, pid, actor=_HEX_ACTOR_2)
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"actor_coder_id": _HEX_ACTOR_2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_other.id

    def test_filter_by_since_and_until(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        _seed_event(projects_root, pid, now="2026-04-01T10:00:00Z")
        e_mid = _seed_event(projects_root, pid, now="2026-04-15T10:00:00Z")
        _seed_event(projects_root, pid, now="2026-05-01T10:00:00Z")
        r = client.get(
            f"/api/projects/{pid}/events",
            params={
                "since": "2026-04-10T00:00:00Z",
                "until": "2026-04-20T00:00:00Z",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_mid.id

    def test_filter_combines_with_AND(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        target = _seed_event(
            projects_root, pid,
            action=EVENT_ACTION_LOCK,
            entity_type=EVENT_ENTITY_CODEBOOK,
            entity_id="",  # codebook lock has no entity id
        )
        # Other combinations that should be filtered out:
        _seed_event(
            projects_root, pid,
            action=EVENT_ACTION_LOCK,
            entity_type=EVENT_ENTITY_CODE,
        )
        _seed_event(
            projects_root, pid,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODEBOOK,
            entity_id="",
        )
        r = client.get(
            f"/api/projects/{pid}/events",
            params={
                "action": EVENT_ACTION_LOCK,
                "entity_type": EVENT_ENTITY_CODEBOOK,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == target.id

    def test_limit_truncates_keeping_newest_when_desc(
        self, server_env
    ) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(projects_root, pid, now="2026-04-01T10:00:00Z")
        e2 = _seed_event(projects_root, pid, now="2026-04-02T10:00:00Z")
        e3 = _seed_event(projects_root, pid, now="2026-04-03T10:00:00Z")
        r = client.get(
            f"/api/projects/{pid}/events",
            params={"limit": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["returned"] == 2
        assert body["truncated"] is True
        ids = [ev["id"] for ev in body["events"]]
        # desc means newest first; after truncation we keep e3, e2.
        assert ids == [e3.id, e2.id]
        # e1 dropped.
        assert e1.id not in ids

    def test_limit_zero_disables_truncation(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        for i in range(5):
            _seed_event(
                projects_root, pid,
                now=f"2026-04-0{i + 1}T10:00:00Z",
            )
        r = client.get(
            f"/api/projects/{pid}/events", params={"limit": 0}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 5
        assert body["returned"] == 5
        assert body["truncated"] is False

    def test_limit_negative_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/events", params={"limit": -1}
        )
        assert r.status_code == 400

    def test_invalid_project_id_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex/events")
        assert r.status_code == 400

    def test_unknown_project_id_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/000000000000/events")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# GET /events/<eid> — single event
# --------------------------------------------------------------------------- #


class TestGetSingleEvent:
    def test_round_trip(self, server_env) -> None:
        _, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        # Use record_update which auto-computes the diff so we get a
        # realistic round-trip including a populated diff.
        from scribe.event_log import record_update
        ev = record_update(
            projects_root,
            project_id=pid,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_ENTITY,
            before={"name": "old"},
            after={"name": "new"},
            actor_coder_id=_HEX_ACTOR,
        )
        r = client.get(f"/api/projects/{pid}/events/{ev.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["event"]["id"] == ev.id
        assert body["event"]["action"] == EVENT_ACTION_UPDATE
        assert body["event"]["before"] == {"name": "old"}
        assert body["event"]["after"] == {"name": "new"}
        # record_update auto-computes the diff
        assert isinstance(body["event"]["diff"], list)
        assert len(body["event"]["diff"]) >= 1

    def test_missing_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/events/{'1' * 12}")
        assert r.status_code == 404

    def test_invalid_event_id_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/events/not-hex")
        assert r.status_code == 400

    def test_unknown_project_id_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get(
            f"/api/projects/000000000000/events/{'a' * 12}"
        )
        assert r.status_code == 404
