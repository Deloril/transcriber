"""End-to-end reachability tests for F3.6 (Matrix views).

The pure builders in ``scribe/matrix.py`` shipped in c206b8d with full
unit coverage. F3.6 had no HTTP route and no UI surface — researchers
could only build matrices via the Python REPL. This file proves the
user-facing wiring:

  * GET  /projects/<pid>/queries renders the F3.6 matrix panel.
  * POST /api/projects/<pid>/matrices/run accepts a {kind, …} payload,
    builds the appropriate matrix, and returns Matrix.to_dict() as
    JSON for the page to render as a table.
  * Three kinds work end-to-end (code-by-source, code-by-code,
    code-by-attribute) and the optional ``query`` field pre-filters
    via the same F3.5 executor the queries route uses.

Mirrors the fixtures in tests/test_server_queries.py — anything new
here that's also useful there should be lifted into a shared module
on the next pass.
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


def _make_project(client: TestClient, name: str = "Mx test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/codes", json={"name": name},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    name: str = "Src",
    *,
    custom_attributes: dict | None = None,
) -> str:
    payload: dict = {
        "name": name, "source_type": "transcript", "language": "en",
    }
    if custom_attributes is not None:
        payload["custom_attributes"] = custom_attributes
    r = client.post(
        f"/api/projects/{pid}/sources", json=payload,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _attach_transcript(srv, sid: str, pid: str, segments: list[dict]) -> None:
    """Plant an edited.json so application_to_query_dict resolves
    speaker / start / end fields. Mirrors the queries-route fixture."""
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


# --------------------------------------------------------------------------- #
# Page renders
# --------------------------------------------------------------------------- #


class TestMatrixPanelRenders:
    """The F3.6 panel must render on the queries page so the matrix
    view is reachable without the user typing the URL by hand."""

    def test_panel_is_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200
        assert 'data-test-feature="F3.6"' in r.text

    def test_panel_has_kind_picker(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="mx-kind-select"' in r.text
        # All three kinds surface as options.
        assert 'value="code-by-source"' in r.text
        assert 'value="code-by-code"' in r.text
        assert 'value="code-by-attribute"' in r.text

    def test_panel_has_run_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="mx-run"' in r.text
        assert "Show matrix" in r.text

    def test_panel_posts_to_matrices_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert "/matrices/run" in r.text
        assert 'method: "POST"' in r.text

    def test_panel_has_use_query_checkbox(self, server_env) -> None:
        """The use-query checkbox is the F3.5 → F3.6 pipeline link."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="mx-use-query"' in r.text


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/matrices/run — basic shape
# --------------------------------------------------------------------------- #


class TestMatrixEndpointShape:
    def test_kind_is_required(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run", json={},
        )
        assert r.status_code == 400
        assert "kind" in r.json()["detail"]

    def test_unknown_kind_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run", json={"kind": "bogus"},
        )
        assert r.status_code == 400

    def test_unknown_project_404s(self, server_env) -> None:
        _, client, _ = server_env
        # 12-hex but no project file on disk.
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/matrices/run",
            json={"kind": "code-by-source"},
        )
        assert r.status_code == 404

    def test_invalid_project_id_400s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/" + ("x" * 200) + "/matrices/run",
            json={"kind": "code-by-source"},
        )
        assert r.status_code == 400

    def test_empty_corpus_returns_empty_matrix(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-source"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "code-by-source"
        assert body["total_applications"] == 0
        assert body["matched_applications"] == 0
        assert body["matrix"]["cells"] == []


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/matrices/run — code-by-source frequency
# --------------------------------------------------------------------------- #


class TestCodeBySourceMatrix:
    def test_two_codes_one_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "code-1")
        c2 = _make_code(client, pid, "code-2")
        s1 = _make_source(client, pid, "S1")
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c2, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-source"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        m = body["matrix"]
        # compact=True drops empty rows by default → just our two codes.
        assert sorted(m["rows"]) == sorted([c1, c2])
        assert m["cols"] == [s1]
        cells = {(r_, c_): v for r_, c_, v in m["cells"]}
        assert cells[(c1, s1)] == 2
        assert cells[(c2, s1)] == 1
        assert body["matched_applications"] == 3

    def test_compact_false_keeps_zero_rows(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "used")
        c2 = _make_code(client, pid, "unused")
        s1 = _make_source(client, pid, "S1")
        _make_application(client, pid, code_id=c1, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-source", "compact": False},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Without compaction, the unused code stays as a zero row.
        assert sorted(body["matrix"]["rows"]) == sorted([c1, c2])

    def test_query_filter_narrows_matrix_to_one_code(self, server_env) -> None:
        """The optional `query` field pre-filters via the F3.5 executor.
        Picking a single code via the query restricts the matrix to
        applications of that code only."""
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "kept")
        c2 = _make_code(client, pid, "filtered-out")
        s1 = _make_source(client, pid, "S1")
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c2, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={
                "kind": "code-by-source",
                "query": {
                    "project_id": pid,
                    "codes": {"expr": {"op": "code", "code_id": c1}},
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["matched_applications"] == 1
        assert body["total_applications"] == 2


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/matrices/run — code-by-code co-occurrence
# --------------------------------------------------------------------------- #


class TestCodeByCodeMatrix:
    def test_source_scope_co_occurrence(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "A")
        c2 = _make_code(client, pid, "B")
        s1 = _make_source(client, pid, "S1")
        # Two A's + one B in the same source ⇒ A·B = 2 (off-diagonal,
        # symmetric); A·A = C(2,2)=1 (diagonal).
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c2, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-code", "scope": "source"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        cells = {(r_, c_): v for r_, c_, v in body["matrix"]["cells"]}
        # Off-diagonal symmetry.
        assert cells[(c1, c2)] == 2
        assert cells[(c2, c1)] == 2
        # Self co-occurrence on A.
        assert cells[(c1, c1)] == 1
        # Self co-occurrence on B is zero (B applied once).
        assert (c2, c2) not in cells

    def test_unknown_scope_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-code", "scope": "weird-scope"},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/matrices/run — code-by-attribute cross-tab
# --------------------------------------------------------------------------- #


class TestCodeByAttributeMatrix:
    def test_attribute_key_required(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={"kind": "code-by-attribute"},
        )
        assert r.status_code == 400
        assert "attribute_key" in r.json()["detail"]

    def test_unknown_attribute_kind_400s(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={
                "kind": "code-by-attribute",
                "attribute_key": "x",
                "attribute_kind": "not-a-thing",
            },
        )
        assert r.status_code == 400

    def test_source_attribute_cross_tab(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "A")
        s1 = _make_source(client, pid, "S1", custom_attributes={"setting": "lab"})
        s2 = _make_source(client, pid, "S2", custom_attributes={"setting": "field"})
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c1, source_id=s2)
        _make_application(client, pid, code_id=c1, source_id=s2)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={
                "kind": "code-by-attribute",
                "attribute_key": "setting",
                "attribute_kind": "source",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        m = body["matrix"]
        # Cols are distinct attribute values, sorted lexicographically
        # ("field" < "lab"). No "missing" col since every source has
        # the attribute.
        assert "field" in m["cols"]
        assert "lab" in m["cols"]
        cells = {(r_, c_): v for r_, c_, v in m["cells"]}
        assert cells[(c1, "lab")] == 1
        assert cells[(c1, "field")] == 2

    def test_missing_values_bucket_into_missing_col(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "A")
        s1 = _make_source(client, pid, "S1", custom_attributes={"setting": "lab"})
        s2 = _make_source(client, pid, "S2")  # no attribute → __missing__
        _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c1, source_id=s2)

        r = client.post(
            f"/api/projects/{pid}/matrices/run",
            json={
                "kind": "code-by-attribute",
                "attribute_key": "setting",
                "attribute_kind": "source",
                "include_missing": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        m = body["matrix"]
        # The missing column key is the matrix module's sentinel.
        assert "__missing__" in m["cols"]


# --------------------------------------------------------------------------- #
# Reachable-via guard: project home links to the queries page that
# now hosts the F3.6 panel.
# --------------------------------------------------------------------------- #


class TestProjectHomeLinksToMatrices:
    def test_home_still_links_to_queries(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert f"/projects/{pid}/queries" in r.text
