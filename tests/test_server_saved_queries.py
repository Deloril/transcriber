"""End-to-end reachability tests for F3.7 (Saved queries).

The pure data + persistence layer in ``scribe/saved_queries.py``
shipped in 6bb1947 with 53 unit tests in ``tests/test_saved_queries.py``.
The F3.7 commit body explicitly deferred the HTTP / UI surface ("the
FastAPI endpoints + UI for managing saved queries are a future
shippable piece"). This file proves the user-facing wiring:

  * GET   /projects/<pid>/queries renders the F3.7 saved-queries panel
    (data-test-feature="F3.7" + a name input + a "Save current query"
    button + a saved-queries list).
  * POST   /api/projects/<pid>/saved-queries creates one.
  * GET    /api/projects/<pid>/saved-queries lists them.
  * GET    /api/projects/<pid>/saved-queries/<sqid> fetches one.
  * PATCH  /api/projects/<pid>/saved-queries/<sqid> updates one.
  * DELETE /api/projects/<pid>/saved-queries/<sqid> removes one.
  * POST   /api/projects/<pid>/saved-queries/<sqid>/run executes
           and bumps run_count + last_run_at.

Mirrors the fixtures in tests/test_server_queries.py and
tests/test_server_matrices.py.
"""

from __future__ import annotations

import hashlib
import json
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


def _make_project(client: TestClient, name: str = "SQ test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str) -> str:
    r = client.post(f"/api/projects/{pid}/codes", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "Src") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _attach_transcript(srv, sid: str, pid: str, segments: list[dict]) -> None:
    """Plant edited.json so the F3.5 runtime can resolve speaker/start/end."""
    job_id = hashlib.sha256(sid.encode()).hexdigest()[:12]
    job_dir = srv.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({"segments": segments}))
    from scribe import sources as _s
    src = _s.load_source(srv._projects_root(), pid, sid)
    src.apply_update({"transcript_job_id": job_id})
    _s.save_source(srv._projects_root(), src)


def _make_application(
    client: TestClient,
    pid: str,
    *,
    code_id: str,
    source_id: str,
    anchor_start: str = "s0w0",
    anchor_end: str = "s0w0",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": code_id,
            "source_id": source_id,
            "anchor_start_word_id": anchor_start,
            "anchor_end_word_id": anchor_end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_saved_query(
    client: TestClient,
    pid: str,
    *,
    name: str = "Power quotes",
    description: str = "",
    code_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> dict:
    inner_query = {
        "project_id": pid,
        "name": name,
        "description": description,
    }
    cids = code_ids or []
    if len(cids) == 1:
        inner_query["codes"] = {"expr": {"op": "code", "code_id": cids[0]}}
    elif len(cids) > 1:
        inner_query["codes"] = {
            "expr": {
                "op": "or",
                "children": [{"op": "code", "code_id": c} for c in cids],
            }
        }
    if source_ids:
        inner_query["sources"] = {"source_ids": list(source_ids)}
    body = {"query": inner_query, "name": name}
    if description:
        body["description"] = description
    r = client.post(f"/api/projects/{pid}/saved-queries", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Page renders the F3.7 panel
# --------------------------------------------------------------------------- #


class TestSavedQueriesPanelRenders:
    """The F3.7 panel must render on the queries page so the saved
    queries surface is reachable without typing a URL by hand."""

    def test_panel_is_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200
        assert 'data-test-feature="F3.7"' in r.text

    def test_panel_has_name_input(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="sq-name-input"' in r.text

    def test_panel_has_save_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="sq-save"' in r.text
        assert "Save current query" in r.text

    def test_panel_has_list_host(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="sq-list"' in r.text

    def test_panel_uses_saved_queries_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        # The page must hit the saved-queries route, not stub it.
        assert "/saved-queries" in r.text


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/saved-queries
# --------------------------------------------------------------------------- #


class TestCreateSavedQuery:
    def test_unknown_project_404s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/saved-queries",
            json={
                "query": {"project_id": "aaaaaaaaaaaa", "name": "X"},
                "name": "X",
            },
        )
        assert r.status_code == 404

    def test_invalid_pid_400s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/not-hex/saved-queries",
            json={"query": {"project_id": "not-hex", "name": "X"}, "name": "X"},
        )
        assert r.status_code == 400

    def test_missing_query_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries", json={"name": "X"},
        )
        assert r.status_code == 400

    def test_invalid_json_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_project_id_mismatch_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries",
            json={
                "query": {"project_id": "ffffffffffff", "name": "X"},
                "name": "X",
            },
        )
        assert r.status_code == 400

    def test_blank_name_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries",
            json={"query": {"project_id": pid, "name": ""}, "name": ""},
        )
        assert r.status_code == 400

    def test_creates_with_top_level_name(self, server_env) -> None:
        """Top-level ``name`` shortcut populates the wrapped Query."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries",
            json={
                "query": {"project_id": pid},
                "name": "Power quotes",
                "description": "Initial coding pass",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["query"]["name"] == "Power quotes"
        assert body["query"]["description"] == "Initial coding pass"
        assert body["run_count"] == 0
        assert body["last_run_at"] == ""
        # 12-hex saved-query id
        assert len(body["id"]) == 12

    def test_create_persists(self, server_env) -> None:
        _, client, tmp = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid, name="Persisted")
        from scribe import saved_queries as _sq
        on_disk = _sq.load_saved_query(tmp / "projects", pid, sq["id"])
        assert on_disk.query.name == "Persisted"


# --------------------------------------------------------------------------- #
# GET /api/projects/<pid>/saved-queries
# --------------------------------------------------------------------------- #


class TestListSavedQueries:
    def test_empty_project(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/saved-queries")
        assert r.status_code == 200
        assert r.json() == {"saved_queries": []}

    def test_lists_saved_queries(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        a = _make_saved_query(client, pid, name="A")
        b = _make_saved_query(client, pid, name="B")
        r = client.get(f"/api/projects/{pid}/saved-queries")
        assert r.status_code == 200
        ids = {sq["id"] for sq in r.json()["saved_queries"]}
        assert ids == {a["id"], b["id"]}

    def test_unknown_project_404s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/aaaaaaaaaaaa/saved-queries")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# GET / PATCH / DELETE one saved query
# --------------------------------------------------------------------------- #


class TestSavedQueryItemRoutes:
    def test_get_unknown_404s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/saved-queries/aaaaaaaaaaaa"
        )
        assert r.status_code == 404

    def test_get_invalid_id_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/saved-queries/not-hex"
        )
        assert r.status_code == 400

    def test_get_returns_one(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid, name="Solo")
        r = client.get(
            f"/api/projects/{pid}/saved-queries/{sq['id']}"
        )
        assert r.status_code == 200
        assert r.json()["id"] == sq["id"]
        assert r.json()["query"]["name"] == "Solo"

    def test_patch_renames(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid, name="Old")
        r = client.patch(
            f"/api/projects/{pid}/saved-queries/{sq['id']}",
            json={"name": "New"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["query"]["name"] == "New"
        # The fetched one persists the change.
        r2 = client.get(
            f"/api/projects/{pid}/saved-queries/{sq['id']}"
        )
        assert r2.json()["query"]["name"] == "New"

    def test_patch_replaces_query(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, "Power")
        sq = _make_saved_query(client, pid, name="Q1")
        r = client.patch(
            f"/api/projects/{pid}/saved-queries/{sq['id']}",
            json={
                "query": {
                    "project_id": pid,
                    "name": "Q1",
                    "codes": {"expr": {"op": "code", "code_id": cid}},
                }
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["query"]["codes"]["expr"]["code_id"] == cid

    def test_patch_invalid_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/saved-queries/{sq['id']}",
            json={"name": ""},  # blank name fails saved-query validation
        )
        assert r.status_code == 400

    def test_patch_unknown_404s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.patch(
            f"/api/projects/{pid}/saved-queries/aaaaaaaaaaaa",
            json={"name": "Whatever"},
        )
        assert r.status_code == 404

    def test_delete_removes(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid)
        r = client.delete(
            f"/api/projects/{pid}/saved-queries/{sq['id']}"
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        r2 = client.get(
            f"/api/projects/{pid}/saved-queries/{sq['id']}"
        )
        assert r2.status_code == 404

    def test_delete_unknown_404s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.delete(
            f"/api/projects/{pid}/saved-queries/aaaaaaaaaaaa"
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/saved-queries/<sqid>/run
# --------------------------------------------------------------------------- #


class TestRunSavedQuery:
    def test_unknown_404s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries/aaaaaaaaaaaa/run"
        )
        assert r.status_code == 404

    def test_run_returns_executor_shape(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, "Power")
        sid = _make_source(client, pid, "S1")
        _attach_transcript(srv, sid, pid, [
            {
                "id": 0,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"id": "s0w0", "text": "hi", "start": 0.0, "end": 0.5},
                ],
            },
        ])
        _make_application(
            client, pid, code_id=cid, source_id=sid,
            anchor_start="s0w0", anchor_end="s0w0",
        )
        sq = _make_saved_query(
            client, pid, name="Q1", code_ids=[cid],
        )
        r = client.post(
            f"/api/projects/{pid}/saved-queries/{sq['id']}/run"
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Shape: same as POST /queries/run, plus saved_query.
        assert "applications" in body
        assert "total_applications" in body
        assert "saved_query" in body
        assert body["saved_query"]["id"] == sq["id"]
        assert body["saved_query"]["run_count"] == 1
        assert body["saved_query"]["last_run_at"] != ""
        # The application matched the code filter.
        assert len(body["applications"]) == 1

    def test_run_bumps_counter_idempotently(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid, name="Q-bump")
        for expected in (1, 2, 3):
            r = client.post(
                f"/api/projects/{pid}/saved-queries/{sq['id']}/run"
            )
            assert r.status_code == 200, r.text
            assert r.json()["saved_query"]["run_count"] == expected

    def test_run_persists_run_count(self, server_env) -> None:
        _, client, tmp = server_env
        pid = _make_project(client)
        sq = _make_saved_query(client, pid, name="Q-persist")
        client.post(f"/api/projects/{pid}/saved-queries/{sq['id']}/run")
        from scribe import saved_queries as _sq
        on_disk = _sq.load_saved_query(tmp / "projects", pid, sq["id"])
        assert on_disk.run_count == 1
        assert on_disk.last_run_at != ""

    def test_run_invalid_id_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/saved-queries/not-hex/run"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_create_list_run_delete(self, server_env) -> None:
        """Cover the full saved-queries lifecycle end-to-end."""
        _, client, _ = server_env
        pid = _make_project(client)

        # Empty list
        r = client.get(f"/api/projects/{pid}/saved-queries")
        assert r.json()["saved_queries"] == []

        # Create
        sq = _make_saved_query(client, pid, name="R1")

        # List shows it
        r = client.get(f"/api/projects/{pid}/saved-queries")
        items = r.json()["saved_queries"]
        assert len(items) == 1 and items[0]["id"] == sq["id"]

        # Run bumps counter
        r = client.post(
            f"/api/projects/{pid}/saved-queries/{sq['id']}/run"
        )
        assert r.json()["saved_query"]["run_count"] == 1

        # Delete
        r = client.delete(
            f"/api/projects/{pid}/saved-queries/{sq['id']}"
        )
        assert r.status_code == 200

        # Empty again
        r = client.get(f"/api/projects/{pid}/saved-queries")
        assert r.json()["saved_queries"] == []
