"""F9.8 reachability — Time-travel viewer.

The pure module ``scribe.time_travel`` shipped in caccdc3 with 54
unit tests covering the pure helpers + per-entity reconstructors +
the :class:`ProjectStateAtTime` aggregator. The original commit
explicitly deferred both the HTTP / FastAPI route and the audit-page
UI affordance ("Same staged approach as F9.1 / F9.6 / F9.7: ship the
data layer, add the surface incrementally"). Until this iteration
landed, the reconstruction was unreachable through the user-facing
surface — researchers could only call it from Python.

This file is the F9.8 reachability anchor:

  1. The audit timeline page renders a Time-travel panel
     (``data-test-feature="F9.8"``) with an ``as_of`` input + a
     submit button so a researcher can reach the endpoint from the
     UI.
  2. ``GET /api/projects/<pid>/time-travel?as_of=...`` returns 200
     with the :meth:`ProjectStateAtTime.to_dict` payload shape.
  3. The route correctly proxies the include_* query flags through
     to :func:`reconstruct_state_at`, so a researcher who only wants
     the exact-history parts (project + codebook) can opt out of the
     best-effort sections.
  4. Error paths — missing/empty ``as_of`` → 400; missing project →
     404; invalid project id → 400.
  5. Pre-project ``as_of`` returns ``project: null``.
  6. Codebook reconstruction is exact (uses the F2.2 version log) so
     a code's *as-of* definition matches its earlier version, not
     the current one.

Deeper coverage of the reconstruction helpers + edge cases lives in
``tests/test_time_travel.py``; this file is purely the HTTP / UI
reachability contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scribe import server as srv

    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), projects


def _new_project(client: TestClient, name: str = "Pacing study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_versioned_code(
    projects_dir: Path,
    project_id: str,
    *,
    code_id: str,
    name: str = "Pacing the day",
    early_definition: str = "v1 — first cut, before refining",
    late_definition: str = "v2 — sharpened after pilot review",
    early_ts: str = "2026-02-01T10:00:00.000000Z",
    late_ts: str = "2026-04-01T10:00:00.000000Z",
):
    """Seed a Code with two definition versions: v1 at ``early_ts``,
    then v2 at ``late_ts``. The time-travel route should pick v1 for
    an ``as_of`` between the two and v2 for an ``as_of`` after both.
    """
    from scribe.codes import Code
    from scribe.code_versions import save_code_with_version

    c = Code.new(
        project_id=project_id,
        name=name,
        definition=early_definition,
        code_id=code_id,
        now=early_ts,
    )
    save_code_with_version(projects_dir, c, now=early_ts)

    # Mutate the on-disk code: load → edit definition → save with version.
    from scribe.codes import load_code
    c2 = load_code(projects_dir, project_id, code_id)
    c2.definition = late_definition
    c2.modified_at = late_ts
    save_code_with_version(projects_dir, c2, now=late_ts)
    return c2


# --------------------------------------------------------------------------- #
# 1. Template render: the audit page surfaces the F9.8 panel
# --------------------------------------------------------------------------- #


class TestAuditPageRendersF9_8Panel:
    """The audit timeline page must render the time-travel panel
    (input + submit + result placeholder) so the route is reachable
    from the UI. If the panel drops off, the user can't reach the
    reconstruction without curl."""

    def test_panel_card_renders(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert 'data-test-feature="F9.8"' in r.text
        assert 'data-test-id="time-travel-panel"' in r.text

    def test_panel_has_as_of_input(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert 'data-test-id="tt-as-of-input"' in r.text
        assert 'id="tt-as-of"' in r.text

    def test_panel_has_submit_button(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert 'data-test-id="tt-submit-btn"' in r.text
        # Heading text the user will look for.
        assert "Time-travel viewer" in r.text

    def test_panel_has_result_placeholder(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert 'data-test-id="tt-result"' in r.text
        assert 'data-test-id="tt-error"' in r.text


# --------------------------------------------------------------------------- #
# 2. Endpoint contract: returns ProjectStateAtTime.to_dict() shape
# --------------------------------------------------------------------------- #


class TestEndpointShape:

    def test_get_returns_200_with_full_payload(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        assert r.status_code == 200, r.text
        data = r.json()
        # Every documented field of ProjectStateAtTime.to_dict() is
        # present.
        for key in (
            "project_id", "as_of", "project", "codes", "applications",
            "memos", "sources", "participants",
            "codebook_stage", "codebook_locked",
            "best_effort", "warnings",
        ):
            assert key in data, f"missing key: {key}"
        assert data["project_id"] == pid

    def test_payload_lists_are_arrays(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        data = r.json()
        for key in (
            "codes", "applications", "memos",
            "sources", "participants", "warnings",
        ):
            assert isinstance(data[key], list), f"{key} not a list"

    def test_warning_appears_for_pre_lock_log_state(self, env) -> None:
        """No lock events yet → time_travel emits a 'best-effort'
        warning for the codebook stage. Surfacing this through the
        endpoint proves the warnings list reaches the client."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        data = r.json()
        assert data["best_effort"] is True
        assert any(
            "codebook_stage" in w and "lock log" in w
            for w in data["warnings"]
        ), data["warnings"]


# --------------------------------------------------------------------------- #
# 3. Codebook is exact: the route picks the version-at-as_of, not live
# --------------------------------------------------------------------------- #


class TestCodebookReconstructionIsExact:
    """F2.2 version log is the source of truth; an as_of timestamp
    between v1 and v2 should show the v1 definition, even though v2
    is the live state on disk."""

    def test_as_of_between_versions_picks_v1(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        cid = "a" * 12
        _seed_versioned_code(projects_dir, pid, code_id=cid)
        # Between the two version timestamps.
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2026-03-01T00:00:00.000000Z"
        )
        assert r.status_code == 200
        codes = r.json()["codes"]
        assert len(codes) == 1
        assert codes[0]["id"] == cid
        assert codes[0]["definition"].startswith("v1"), codes[0]["definition"]

    def test_as_of_after_both_picks_v2(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        cid = "a" * 12
        _seed_versioned_code(projects_dir, pid, code_id=cid)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2026-12-31T00:00:00.000000Z"
        )
        codes = r.json()["codes"]
        assert codes[0]["definition"].startswith("v2")

    def test_as_of_before_either_returns_no_code(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        cid = "a" * 12
        _seed_versioned_code(projects_dir, pid, code_id=cid)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2025-12-01T00:00:00.000000Z"
        )
        assert r.json()["codes"] == []


# --------------------------------------------------------------------------- #
# 4. include_* query flags drop best-effort sections
# --------------------------------------------------------------------------- #


class TestIncludeFlags:

    def test_include_applications_off_drops_section(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        # Drop in one application via the pure module so the default
        # response would include it.
        from scribe.applications import Application, save_application
        app = Application.new(
            project_id=pid,
            source_id="aaaaaaaaaaaa",
            code_id="cccccccccccc",
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            coder_id="0" * 12,
            definition_version_id_at_apply="dddddddddddd",
            now="2026-02-01T00:00:00.000000Z",
        )
        save_application(projects_dir, app)
        # Default: included.
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        assert len(r.json()["applications"]) == 1
        # Off: empty.
        r2 = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z&include_applications=0"
        )
        assert r2.json()["applications"] == []

    def test_include_memos_off_drops_section(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z&include_memos=false"
        )
        assert r.status_code == 200
        # No memos seeded; shape is preserved regardless.
        assert r.json()["memos"] == []

    def test_include_sources_off_drops_section(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z&include_sources=no"
        )
        assert r.status_code == 200
        assert r.json()["sources"] == []


# --------------------------------------------------------------------------- #
# 5. Pre-project as_of: project is null
# --------------------------------------------------------------------------- #


class TestPreProjectReconstruction:

    def test_pre_project_as_of_returns_null_project(self, env) -> None:
        """An ``as_of`` before the project's ``created_at`` should
        leave ``project`` as null in the response."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/time-travel"
            "?as_of=1999-01-01T00:00:00.000000Z"
        )
        assert r.status_code == 200
        data = r.json()
        assert data["project"] is None
        # Codes / apps lists are also empty because nothing existed.
        assert data["codes"] == []


# --------------------------------------------------------------------------- #
# 6. Error paths
# --------------------------------------------------------------------------- #


class TestErrorPaths:

    def test_missing_as_of_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/time-travel")
        assert r.status_code == 400
        assert "as_of" in r.json()["detail"].lower()

    def test_empty_as_of_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/time-travel?as_of=")
        assert r.status_code == 400

    def test_whitespace_only_as_of_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/time-travel?as_of=%20%20")
        assert r.status_code == 400

    def test_missing_project_returns_404(self, env) -> None:
        client, _ = env
        bogus = "deadbeef0000"
        r = client.get(
            f"/api/projects/{bogus}/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        assert r.status_code == 404
        assert "project" in r.json()["detail"].lower()

    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _ = env
        r = client.get(
            "/api/projects/!!bad!!/time-travel"
            "?as_of=2030-01-01T00:00:00.000000Z"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 7. as_of round-trips through the response (UI reads it back).
# --------------------------------------------------------------------------- #


class TestAsOfRoundTrip:

    def test_as_of_round_trips(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        as_of = "2026-04-12T15:30:45.123456Z"
        r = client.get(
            f"/api/projects/{pid}/time-travel?as_of={as_of}"
        )
        assert r.status_code == 200
        assert r.json()["as_of"] == as_of
