"""End-to-end reachability tests for F3.5 (Query builder).

The pure data layer in ``scribe/query.py`` shipped in ae528c2 with
full unit coverage in ``tests/test_query.py``. ``scribe/saved_queries.py``
wraps the persistence side. What was missing — and what this file
proves — is the **user-facing surface**:

  * GET  /projects/<pid>/queries renders a real query-builder page
    (not a wireframe stub).
  * The page reads codes via GET /api/projects/<pid>/codes,
    sources via GET /api/projects/<pid>/sources.
  * POST /api/projects/<pid>/queries/run accepts a Query payload,
    executes the F3.5 filter algebra against the project's
    applications, and returns the matching applications.
  * Project home links to the queries page so the user can find it
    without typing the URL by hand.
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


def _make_project(client: TestClient, name: str = "QB test") -> str:
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
    client: TestClient, pid: str, name: str = "Src",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _attach_transcript(
    srv, sid: str, pid: str, segments: list[dict],
) -> None:
    """Plant an edited.json under outputs/<job_id>/ and update the source
    so its transcript_job_id resolves to that path. The QDPX / speaker-map
    discovery rules read it back via _load_segments_for_source_speaker_map.
    """
    import json
    # Build a 12-hex job id deterministically off the source id (which
    # itself is 12-hex), so we don't accidentally collide across
    # multiple sources in one test.
    import hashlib
    job_id = hashlib.sha256(sid.encode()).hexdigest()[:12]
    job_dir = srv.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({"segments": segments}))
    # Patch the source's transcript_job_id in memory + on disk.
    from scribe import sources as _s
    src = _s.load_source(srv._projects_root(), pid, sid)
    # Source.apply_update keeps the validators happy.
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


class TestQueriesPageRenders:
    """``/projects/<pid>/queries`` must render a real builder, not a
    wireframe stub. Without the data-test-feature marker the F3.5
    surface isn't proven to be the query builder."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200

    def test_marks_F3_5_feature(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-feature="F3.5"' in r.text
        assert "Build a query" in r.text

    def test_no_longer_renders_wireframe_stub(self, server_env) -> None:
        """Regression: the page used to be a project_subpage wireframe
        with an `alert("Stub: F3.5")` button. Make sure that's gone."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert "Stub: F3.5" not in r.text
        assert "Wireframe." not in r.text

    def test_page_has_run_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="qb-run"' in r.text
        assert "Run query" in r.text

    def test_page_has_codes_select(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="qb-codes-select"' in r.text

    def test_page_has_sources_select(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="qb-sources-select"' in r.text

    def test_page_has_speaker_role_picker(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="qb-speaker-role"' in r.text
        # All four canonical roles surface as options.
        for role in ("interviewer", "interviewee", "facilitator", "participant"):
            assert f'value="{role}"' in r.text

    def test_page_posts_to_run_endpoint(self, server_env) -> None:
        """The page's runner POSTs to the F3.5 /queries/run endpoint."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert "/queries/run" in r.text
        assert 'method: "POST"' in r.text

    def test_page_400s_for_malformed_project_id(self, server_env) -> None:
        # The page renders even for unknown 12-hex ids (see
        # _project_id_or_404 in server.py — empty-state UI is intended)
        # so we only assert that obviously-malformed ids are 400.
        _, client, _ = server_env
        long_id = "x" * 200  # exceeds the 64-char shape limit
        r = client.get(f"/projects/{long_id}/queries")
        assert r.status_code == 400

    def test_project_home_links_to_queries(self, server_env) -> None:
        """Project home must surface a link so the queries page is reachable
        without typing the URL by hand."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert 'data-test-id="ph-queries-link"' in r.text
        assert f"/projects/{pid}/queries" in r.text


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/queries/run — basic shape
# --------------------------------------------------------------------------- #


class TestRunQueryEndpointShape:
    def test_run_empty_query_returns_zero_when_no_apps(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {"project_id": pid}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["applications"] == []
        assert body["total_applications"] == 0
        assert body["sources_missing_transcript"] == []
        assert isinstance(body.get("warnings"), list)

    def test_404_for_unknown_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/abcdef012345/queries/run",
            json={"query": {"project_id": "abcdef012345"}},
        )
        assert r.status_code == 404

    def test_400_for_invalid_project_id_in_url(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/!!!/queries/run",
            json={"query": {"project_id": "abcdef012345"}},
        )
        assert r.status_code == 400

    def test_400_for_missing_query_object(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={},
        )
        assert r.status_code == 400

    def test_400_for_invalid_query_payload(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {"project_id": pid, "codes": "not an object"}},
        )
        assert r.status_code == 400

    def test_400_for_project_id_mismatch(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {"project_id": "ffffffffffff"}},
        )
        assert r.status_code == 400

    def test_url_project_id_stamped_when_omitted(self, server_env) -> None:
        """If the body omits project_id, the URL one is stamped in."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {}},
        )
        assert r.status_code == 200, r.text
        # No applications yet — the call still succeeds.
        body = r.json()
        assert body["total_applications"] == 0


# --------------------------------------------------------------------------- #
# POST /api/projects/<pid>/queries/run — actual filter algebra
# --------------------------------------------------------------------------- #


SEGMENTS = [
    {
        "speaker": "INTERVIEWER",
        "start": 0.0, "end": 1.0,
        "words": [
            {"word": "How", "start": 0.0, "end": 0.4},
            {"word": "are", "start": 0.4, "end": 0.8},
        ],
    },
    {
        "speaker": "PARTICIPANT",
        "start": 1.0, "end": 2.5,
        "words": [
            {"word": "Anxious", "start": 1.0, "end": 1.6},
            {"word": "today", "start": 1.6, "end": 2.5},
        ],
    },
]


class TestRunQueryEndpointFilters:
    def test_empty_query_matches_all(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        c2 = _make_code(client, pid, "hope")
        s1 = _make_source(client, pid, "S1")
        _attach_transcript(srv, s1, pid, SEGMENTS)
        a1 = _make_application(client, pid, code_id=c1, source_id=s1)
        a2 = _make_application(
            client, pid, code_id=c2, source_id=s1,
            anchor_start="s1w0", anchor_end="s1w1",
        )

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {"project_id": pid}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_applications"] == 2
        ids = {a["id"] for a in body["applications"]}
        assert ids == {a1, a2}

    def test_filter_by_single_code(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        c2 = _make_code(client, pid, "hope")
        s1 = _make_source(client, pid, "S1")
        _attach_transcript(srv, s1, pid, SEGMENTS)
        a1 = _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c2, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {
                "project_id": pid,
                "codes": {"expr": {"op": "code", "code_id": c1}},
            }},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        ids = {a["id"] for a in body["applications"]}
        assert ids == {a1}
        assert body["total_applications"] == 2

    def test_filter_by_or_of_codes(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        c2 = _make_code(client, pid, "hope")
        c3 = _make_code(client, pid, "irrelevant")
        s1 = _make_source(client, pid, "S1")
        _attach_transcript(srv, s1, pid, SEGMENTS)
        a1 = _make_application(client, pid, code_id=c1, source_id=s1)
        a2 = _make_application(client, pid, code_id=c2, source_id=s1)
        _make_application(client, pid, code_id=c3, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {
                "project_id": pid,
                "codes": {"expr": {"op": "or", "children": [
                    {"op": "code", "code_id": c1},
                    {"op": "code", "code_id": c2},
                ]}},
            }},
        )
        assert r.status_code == 200, r.text
        ids = {a["id"] for a in r.json()["applications"]}
        assert ids == {a1, a2}

    def test_filter_by_source(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        s1 = _make_source(client, pid, "S1")
        s2 = _make_source(client, pid, "S2")
        _attach_transcript(srv, s1, pid, SEGMENTS)
        _attach_transcript(srv, s2, pid, SEGMENTS)
        a_in_s1 = _make_application(client, pid, code_id=c1, source_id=s1)
        _make_application(client, pid, code_id=c1, source_id=s2)

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {
                "project_id": pid,
                "sources": {"source_ids": [s1]},
            }},
        )
        assert r.status_code == 200, r.text
        ids = {a["id"] for a in r.json()["applications"]}
        assert ids == {a_in_s1}

    def test_filter_by_speaker_role(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        s1 = _make_source(client, pid, "S1")
        _attach_transcript(srv, s1, pid, SEGMENTS)
        # PUT a speaker map mapping INTERVIEWER → interviewer,
        # PARTICIPANT → interviewee.
        r = client.put(
            f"/api/projects/{pid}/sources/{s1}/speaker_map",
            json={"entries": [
                {"label": "INTERVIEWER", "role": "interviewer"},
                {"label": "PARTICIPANT", "role": "interviewee"},
            ]},
        )
        assert r.status_code == 200, r.text
        # a_int anchored on segment 0 → speaker INTERVIEWER → interviewer
        a_int = _make_application(
            client, pid, code_id=c1, source_id=s1,
            anchor_start="s0w0", anchor_end="s0w1",
        )
        # a_part anchored on segment 1 → speaker PARTICIPANT → interviewee
        a_part = _make_application(
            client, pid, code_id=c1, source_id=s1,
            anchor_start="s1w0", anchor_end="s1w1",
        )

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {
                "project_id": pid,
                "speakers": {"roles": ["interviewee"]},
            }},
        )
        assert r.status_code == 200, r.text
        ids = {a["id"] for a in r.json()["applications"]}
        assert ids == {a_part}
        assert a_int not in ids

    def test_missing_transcript_recorded_in_response(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        c1 = _make_code(client, pid, "anxiety")
        # No transcript attached → loader returns None for s1.
        s1 = _make_source(client, pid, "S1")
        _make_application(client, pid, code_id=c1, source_id=s1)

        r = client.post(
            f"/api/projects/{pid}/queries/run",
            json={"query": {"project_id": pid}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert s1 in body["sources_missing_transcript"]
        # The application still comes back in the empty-filter case;
        # the missing-transcript notice just explains why speaker /
        # proximity filters won't work.
        assert body["total_applications"] == 1
