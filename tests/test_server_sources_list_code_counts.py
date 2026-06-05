"""Tests for the per-row code-count columns on /projects/<pid>/sources.

The sources-list page now renders two extra columns per row:
``Coded spans`` (total applications) and ``Distinct codes`` (number
of unique codes that have at least one application). The data is
derived client-side from a single
``GET /api/projects/<pid>/applications`` call so this file has two
jobs:

* prove the page surfaces the column markers + the JS plumbing
  (``fetchCodeCounts`` helper),
* prove the underlying API returns the shape the helper depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _make_project(client: TestClient, name: str = "Pilot") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "S") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": ""},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _apply(
    client: TestClient, pid: str, cid: str, sid: str,
    start: str, end: str,
) -> str:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": cid, "source_id": sid,
            "anchor_start_word_id": start,
            "anchor_end_word_id": end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestSourcesListExposesCodeCountColumns:
    def test_template_renders_column_markers(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert r.status_code == 200
        body = r.text
        # Column header markers.
        assert 'data-test-id="src-list-apps-col"' in body
        assert 'data-test-id="src-list-codes-col"' in body
        # Per-row cell markers (rendered by the JS row template literal).
        assert 'data-test-id="src-list-apps-count"' in body
        assert 'data-test-id="src-list-codes-count"' in body
        # The fetcher helper that builds the count map.
        assert "fetchCodeCounts" in body

    def test_applications_endpoint_returns_groupable_shape(
        self, server_env
    ) -> None:
        """The JS groups by ``source_id`` and counts distinct
        ``code_id``s. Pin that the listing endpoint surfaces both
        keys per row so a future API tweak doesn't silently break
        the per-row count."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid_a = _make_source(client, pid, "A")
        sid_b = _make_source(client, pid, "B")
        cid_x = _make_code(client, pid, "x")
        cid_y = _make_code(client, pid, "y")

        # Source A: 3 spans with 2 distinct codes.
        _apply(client, pid, cid_x, sid_a, "s0w0", "s0w1")
        _apply(client, pid, cid_x, sid_a, "s0w2", "s0w3")
        _apply(client, pid, cid_y, sid_a, "s0w4", "s0w5")
        # Source B: 1 span, 1 code.
        _apply(client, pid, cid_y, sid_b, "s0w0", "s0w0")

        r = client.get(f"/api/projects/{pid}/applications")
        assert r.status_code == 200, r.text
        apps = r.json()["applications"]
        assert len(apps) == 4
        for a in apps:
            assert "source_id" in a
            assert "code_id" in a

        # Mirror the JS grouping so the test fails clearly if the
        # endpoint shape ever changes.
        by_source: dict[str, dict] = {}
        for a in apps:
            sid = a["source_id"]
            by_source.setdefault(
                sid, {"applications": 0, "codes": set()},
            )
            by_source[sid]["applications"] += 1
            by_source[sid]["codes"].add(a["code_id"])

        assert by_source[sid_a]["applications"] == 3
        assert len(by_source[sid_a]["codes"]) == 2
        assert by_source[sid_b]["applications"] == 1
        assert len(by_source[sid_b]["codes"]) == 1
