"""End-to-end reachability tests for F8.9 (AI provenance + AIEvent log).

The F8.9 module ``scribe.ai_provenance`` shipped the ``AIProvenance``
schema, the append-only ``AIEvent`` persistence layer, and the
``provenance_from_*`` extractors in 7fc8b24. That commit explicitly
deferred the HTTP / FastAPI surface; meanwhile F8.5 / F8.6 / F8.7 /
F8.8 *write* AIEvents into the project store via the F9.6 invocation
helpers, but no UI could read them.

This file covers the F8.9 read surface:

  * ``GET /api/projects/<pid>/ai/events`` — list (filterable)
  * ``GET /api/projects/<pid>/ai/events/<eid>`` — single event
  * The ``project_ai.html`` template renders the F8.9 panel
    (``data-test-feature="F8.9"``) so the route is reachable from the
    UI, not just curl.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.ai_provenance import (
    AI_EVENT_KIND_DECISION,
    AI_EVENT_KIND_REQUEST,
    AI_EVENT_KINDS,
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_MEMO_DRAFT,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURES,
    AIEvent,
    AIProvenance,
    save_ai_event,
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


def _make_project(client: TestClient, name: str = "EventsP") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


_HEX_CODER = "a" * 12
_HEX_CODER_2 = "b" * 12


def _seed_event(
    projects_root: Path,
    project_id: str,
    *,
    feature: str = AI_FEATURE_CODE_SUGGESTION,
    kind: str = AI_EVENT_KIND_REQUEST,
    actor: str = _HEX_CODER,
    payload: dict | None = None,
    now: str | None = None,
) -> AIEvent:
    """Persist one AIEvent against the project store."""
    prov = AIProvenance.new(feature=feature, generation_model="llama3.2:3b")
    ev = AIEvent.new(
        project_id=project_id,
        feature=feature,
        kind=kind,
        actor_coder_id=actor,
        provenance=prov,
        payload=payload or {},
        now=now,
    )
    save_ai_event(projects_root, ev)
    return ev


# --------------------------------------------------------------------------- #
# GET /ai/events — listing
# --------------------------------------------------------------------------- #


class TestListAIEvents:
    def test_empty_project_returns_empty_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/events")
        assert r.status_code == 200
        body = r.json()
        assert body["events"] == []
        assert body["total"] == 0
        assert body["returned"] == 0
        assert body["truncated"] is False
        # Shape contracts the UI relies on:
        assert body["order"] == "desc"
        assert set(body["available_features"]) == set(AI_FEATURES)
        assert set(body["available_kinds"]) == set(AI_EVENT_KINDS)

    def test_lists_persisted_events_newest_first_by_default(
        self, server_env
    ) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(
            projects_root, pid,
            kind=AI_EVENT_KIND_REQUEST,
            now="2026-04-01T10:00:00Z",
        )
        e2 = _seed_event(
            projects_root, pid,
            kind=AI_EVENT_KIND_DECISION,
            now="2026-04-02T11:30:00Z",
        )
        r = client.get(f"/api/projects/{pid}/ai/events")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert body["returned"] == 2
        ids = [ev["id"] for ev in body["events"]]
        # Newest-first per default order=desc
        assert ids == [e2.id, e1.id]

    def test_order_asc_returns_chronological(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e1 = _seed_event(
            projects_root, pid, now="2026-04-01T10:00:00Z",
        )
        e2 = _seed_event(
            projects_root, pid, now="2026-04-02T10:00:00Z",
        )
        r = client.get(
            f"/api/projects/{pid}/ai/events", params={"order": "asc"}
        )
        assert r.status_code == 200
        ids = [ev["id"] for ev in r.json()["events"]]
        assert ids == [e1.id, e2.id]

    def test_order_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/events", params={"order": "weird"}
        )
        assert r.status_code == 400

    def test_filter_by_feature(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e_code = _seed_event(
            projects_root, pid, feature=AI_FEATURE_CODE_SUGGESTION,
        )
        _seed_event(projects_root, pid, feature=AI_FEATURE_MEMO_DRAFT)
        _seed_event(projects_root, pid, feature=AI_FEATURE_QUOTE_SIMILARITY)
        r = client.get(
            f"/api/projects/{pid}/ai/events",
            params={"feature": AI_FEATURE_CODE_SUGGESTION},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_code.id
        assert body["events"][0]["feature"] == AI_FEATURE_CODE_SUGGESTION

    def test_filter_by_feature_invalid_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/events",
            params={"feature": "totally-not-real"},
        )
        assert r.status_code == 400

    def test_filter_by_kind(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        e_req = _seed_event(projects_root, pid, kind=AI_EVENT_KIND_REQUEST)
        _seed_event(projects_root, pid, kind=AI_EVENT_KIND_DECISION)
        r = client.get(
            f"/api/projects/{pid}/ai/events",
            params={"kind": AI_EVENT_KIND_REQUEST},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_req.id
        assert body["events"][0]["kind"] == AI_EVENT_KIND_REQUEST

    def test_filter_by_actor_coder(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        _seed_event(projects_root, pid, actor=_HEX_CODER)
        e_other = _seed_event(projects_root, pid, actor=_HEX_CODER_2)
        r = client.get(
            f"/api/projects/{pid}/ai/events",
            params={"actor_coder_id": _HEX_CODER_2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == e_other.id

    def test_filter_combines_with_AND(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        target = _seed_event(
            projects_root, pid,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_DECISION,
        )
        # Other combinations that should be filtered out:
        _seed_event(projects_root, pid,
                    feature=AI_FEATURE_CODE_SUGGESTION,
                    kind=AI_EVENT_KIND_REQUEST)
        _seed_event(projects_root, pid,
                    feature=AI_FEATURE_MEMO_DRAFT,
                    kind=AI_EVENT_KIND_DECISION)
        r = client.get(
            f"/api/projects/{pid}/ai/events",
            params={
                "feature": AI_FEATURE_CODE_SUGGESTION,
                "kind": AI_EVENT_KIND_DECISION,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["events"][0]["id"] == target.id

    def test_limit_truncates(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        for i in range(5):
            _seed_event(
                projects_root, pid,
                now=f"2026-04-0{i + 1}T10:00:00Z",
            )
        r = client.get(
            f"/api/projects/{pid}/ai/events", params={"limit": 2}
        )
        assert r.status_code == 200
        body = r.json()
        # Total reflects the un-truncated count; returned reflects the page.
        assert body["total"] == 5
        assert body["returned"] == 2
        assert body["truncated"] is True
        assert len(body["events"]) == 2

    def test_limit_zero_means_no_truncation(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        for i in range(3):
            _seed_event(projects_root, pid, now=f"2026-04-0{i + 1}T10:00:00Z")
        r = client.get(
            f"/api/projects/{pid}/ai/events", params={"limit": 0}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["returned"] == 3
        assert body["truncated"] is False

    def test_negative_limit_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/ai/events", params={"limit": -1}
        )
        assert r.status_code == 400

    def test_404_for_missing_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/" + ("0" * 12) + "/ai/events")
        assert r.status_code == 404

    def test_400_for_malformed_project_id(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/not-hex/ai/events")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GET /ai/events/<eid> — single
# --------------------------------------------------------------------------- #


class TestGetSingleAIEvent:
    def test_returns_persisted_event(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        projects_root = tmp_path / "projects"
        ev = _seed_event(
            projects_root, pid, payload={"source_id": "f" * 12, "n": 7}
        )
        r = client.get(f"/api/projects/{pid}/ai/events/{ev.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["event"]["id"] == ev.id
        assert body["event"]["feature"] == AI_FEATURE_CODE_SUGGESTION
        assert body["event"]["payload"]["n"] == 7
        # Provenance round-trips
        assert body["event"]["provenance"]["feature"] == AI_FEATURE_CODE_SUGGESTION

    def test_404_for_missing_event(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/events/" + ("c" * 12))
        assert r.status_code == 404

    def test_400_for_malformed_event_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/events/not-hex")
        assert r.status_code == 400

    def test_404_for_missing_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get(
            "/api/projects/" + ("0" * 12) + "/ai/events/" + ("a" * 12)
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# project_ai.html template renders the F8.9 panel
# --------------------------------------------------------------------------- #


class TestProjectAIPageF89Panel:
    def test_panel_is_present(self, server_env) -> None:
        """The /projects/<pid>/ai page must surface the F8.9 panel so
        the routes above are reachable from the user-facing surface,
        not just curl. If this regresses, the loop's done-detector
        treats F8.9 as not done."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert 'data-test-feature="F8.9"' in r.text
        assert 'data-test-id="ai-events-card"' in r.text
        # The list/filter/error elements the JS hooks into:
        assert 'data-test-id="ai-events-list"' in r.text
        assert 'data-test-id="ai-events-filter-feature"' in r.text
        assert 'data-test-id="ai-events-filter-kind"' in r.text

    def test_panel_links_to_invocation_log(self, server_env) -> None:
        """The card copy mentions F9.6 so future readers know which
        feature this read surface unblocks."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert "F9.6" in r.text or "invocation log" in r.text.lower()
