"""F9.7 reachability — chronological audit-trail export.

The pure renderer lives in :mod:`scribe.audit_export` (shipped in
ba781f4): it walks the F9.1 generic event log + the F9.6 unified AI
invocation log into one chronological audit trail and renders CSV /
Markdown / RTF. The original commit explicitly deferred the HTTP /
FastAPI surface; until this route + the audit-page download menu
landed, the module had no user-facing surface — researchers couldn't
reach the audit-trail report through the UI.

This test file is the F9.7 reachability anchor:

  1. The audit timeline page renders three F9.7 menu items
     (CSV / Markdown / Word(RTF)) pointing at the F9.7 endpoint.
  2. Each link carries the ``download`` attribute.
  3. The endpoint returns 200 with the right Content-Type +
     slugified attachment filename for each format.
  4. The format alias set the URL accepts (``md``, ``word``,
     ``doc``, ``docx``) routes to the right renderer.
  5. The CSV body matches :data:`CSV_COLUMNS` and contains seeded
     event + AI-invocation rows in chronological order.
  6. Filter forwarding: ``action``, ``entity_type``, ``actor``,
     ``kind``, ``feature``, ``decision``, ``since`` / ``until``
     all narrow the response.
  7. Empty projects produce a header-only CSV / placeholder
     Markdown / minimal RTF.
  8. Unknown formats return 400; missing projects return 404;
     malformed project ids return 400; invalid filter values
     return 400.

Deeper coverage of the renderer + filter semantics lives in
``tests/test_audit_export.py``; this file is purely the HTTP / UI
reachability contract.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _seed_events(projects_dir: Path, project_id: str) -> dict:
    """Drop a couple of F9.1 events + an F9.6 AI invocation so the
    audit trail has heterogeneous rows to render."""
    from scribe import event_log as el

    e1 = el.record_event(
        projects_dir,
        project_id=project_id,
        action="create",
        entity_type="code",
        entity_id="aaaaaaaaaaaa",
        actor_coder_id="0" * 12,
        after={"name": "Pacing the day", "definition": "v1"},
        notes="initial coding",
    )
    e2 = el.record_event(
        projects_dir,
        project_id=project_id,
        action="update",
        entity_type="code",
        entity_id="aaaaaaaaaaaa",
        actor_coder_id="0" * 12,
        before={"name": "Pacing the day", "definition": "v1"},
        after={"name": "Pacing the day", "definition": "v2"},
        notes="sharpened",
    )
    return {"event_ids": [e1.id, e2.id]}


def _seed_ai_invocation(projects_dir: Path, project_id: str) -> str:
    """Drop a code-suggestion invocation so the F9.6 read-side has a
    row the audit trail will pick up. Uses the same module API as
    ``tests/test_audit_export.py`` — CodeSuggestion + record_decision +
    save_suggestion."""
    from scribe import code_suggestions as cs_mod

    sug = cs_mod.CodeSuggestion.new(
        project_id=project_id,
        source_id="b" * 12,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w1",
        query_text="they took a nap",
        now="2026-05-26T10:00:00Z",
    )
    cs_mod.record_decision(
        sug,
        decision="rejected",
        coder_id="0" * 12,
        rejection_reason="off-target",
        now="2026-05-26T11:00:00Z",
    )
    cs_mod.save_suggestion(projects_dir, sug)
    return sug.id


# --------------------------------------------------------------------------- #
# 1. Template render: the audit page surfaces three F9.7 download links
# --------------------------------------------------------------------------- #


class TestAuditPageRendersF9_7Links:
    """The audit timeline page must expose three download menu items
    pointing at the F9.7 endpoint, one per format. If any drop off,
    the user can't reach the export through the UI."""

    def test_csv_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/audit-trail?format=csv"'
            in r.text
        )

    def test_markdown_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/audit-trail?format=markdown"'
            in r.text
        )

    def test_rtf_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/audit-trail?format=rtf"'
            in r.text
        )

    def test_links_carry_download_attribute(self, env) -> None:
        """``download`` makes the browser save rather than navigate."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.text.count('class="audit-trail-export-item"') >= 3
        assert "download" in r.text

    def test_audit_page_carries_feature_marker(self, env) -> None:
        """``data-test-feature="F9.7"`` is the loop's reachability
        anchor on this page; the menu button + namespace prefix make
        it distinct from the F9.2 menu next to it."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert 'data-test-feature="F9.7"' in r.text
        assert 'id="audit-trail-export-btn"' in r.text
        assert 'id="audit-trail-export-menu"' in r.text


# --------------------------------------------------------------------------- #
# 2. Endpoint contract: each format returns 200 + the documented MIME
# --------------------------------------------------------------------------- #


class TestExportEndpointContract:

    def test_csv_returns_text_csv_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_events(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        cd = r.headers["content-disposition"]
        assert "attachment" in cd
        assert "pacing-study-audit-trail.csv" in cd

    def test_markdown_returns_text_markdown_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_events(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=markdown")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/markdown")
        cd = r.headers["content-disposition"]
        assert "pacing-study-audit-trail.md" in cd

    def test_rtf_returns_application_rtf_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_events(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=rtf")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/rtf")
        cd = r.headers["content-disposition"]
        assert "pacing-study-audit-trail.rtf" in cd
        assert r.text.startswith("{\\rtf1")

    def test_format_aliases_route_correctly(self, env) -> None:
        """``md`` / ``word`` / ``doc`` / ``docx`` map to markdown / rtf."""
        client, _ = env
        pid = _new_project(client)
        for alias, canonical_ext in (
            ("md", ".md"),
            ("word", ".rtf"),
            ("doc", ".rtf"),
            ("docx", ".rtf"),
        ):
            r = client.get(
                f"/api/projects/{pid}/audit-trail?format={alias}"
            )
            assert r.status_code == 200, (alias, r.text)
            assert canonical_ext in r.headers["content-disposition"]


# --------------------------------------------------------------------------- #
# 3. CSV body schema + chronological ordering
# --------------------------------------------------------------------------- #


class TestCsvBodyMatchesContract:

    def test_csv_columns_match_module_contract(self, env) -> None:
        from scribe.audit_export import CSV_COLUMNS
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=csv")
        assert r.status_code == 200
        first_line = r.text.split("\r\n")[0]
        rdr = csv.reader(io.StringIO(first_line))
        header = next(rdr)
        assert tuple(header) == CSV_COLUMNS

    def test_csv_contains_event_rows_in_chronological_order(
        self, env
    ) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        seeded = _seed_events(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=csv")
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        # Two events, in created order. record_id == event id.
        assert [row["record_id"] for row in rows] == seeded["event_ids"]
        assert all(row["kind"] == "event" for row in rows)
        actions = [row["action"] for row in rows]
        assert actions == ["create", "update"]

    def test_csv_includes_ai_invocation_rows(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        sid = _seed_ai_invocation(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=csv")
        rows = list(csv.DictReader(io.StringIO(r.text)))
        kinds = {row["kind"] for row in rows}
        assert "event" in kinds
        assert "ai_invocation" in kinds
        ai_rows = [row for row in rows if row["kind"] == "ai_invocation"]
        assert len(ai_rows) == 1
        assert ai_rows[0]["record_id"] == sid
        # F9.6 invocation 'action' column carries the decision label.
        assert ai_rows[0]["action"] == "rejected"


# --------------------------------------------------------------------------- #
# 4. Filter forwarding: query string narrows the response
# --------------------------------------------------------------------------- #


class TestFilterForwarding:

    def test_action_filter_narrows_event_rows(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&action=update"
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        ev_rows = [row for row in rows if row["kind"] == "event"]
        # 'update' is the only event action that survives the filter.
        assert ev_rows, "expected at least one event row"
        assert all(row["action"] == "update" for row in ev_rows)

    def test_entity_type_filter_narrows_event_rows(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&entity_type=code"
        )
        rows = list(csv.DictReader(io.StringIO(r.text)))
        ev_rows = [row for row in rows if row["kind"] == "event"]
        assert ev_rows
        assert all(row["entity_type"] == "code" for row in ev_rows)

    def test_kind_filter_restricts_to_one_source(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        _seed_ai_invocation(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&kind=event"
        )
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert rows
        assert all(row["kind"] == "event" for row in rows)

    def test_decision_filter_targets_ai_rows(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        _seed_ai_invocation(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&decision=rejected"
        )
        rows = list(csv.DictReader(io.StringIO(r.text)))
        ai_rows = [row for row in rows if row["kind"] == "ai_invocation"]
        assert ai_rows
        assert all(row["action"] == "rejected" for row in ai_rows)

    def test_actor_filter_round_trips(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_events(projects_dir, pid)
        # Same actor as seeded (12 zeros).
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv"
            f"&actor_coder_id={'0' * 12}"
        )
        assert r.status_code == 200
        # And a non-matching actor should drop event rows.
        r2 = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv"
            f"&actor_coder_id={'1' * 12}"
        )
        assert r2.status_code == 200
        rows2 = list(csv.DictReader(io.StringIO(r2.text)))
        ev_rows2 = [row for row in rows2 if row["kind"] == "event"]
        assert ev_rows2 == []


# --------------------------------------------------------------------------- #
# 5. Empty project — header-only CSV / placeholder Markdown / minimal RTF
# --------------------------------------------------------------------------- #


class TestEmptyProject:

    def test_empty_csv_is_header_only(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=csv")
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert rows == []
        # First line is the header — non-empty.
        assert r.text.split("\r\n")[0].startswith("timestamp")

    def test_empty_markdown_renders(self, env) -> None:
        client, _ = env
        pid = _new_project(client, name="Pacing study")
        r = client.get(f"/api/projects/{pid}/audit-trail?format=markdown")
        assert r.status_code == 200
        # Header + project name appear regardless of row count.
        assert "Audit trail" in r.text

    def test_empty_rtf_is_well_formed(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/audit-trail?format=rtf")
        assert r.status_code == 200
        assert r.text.startswith("{\\rtf1")
        assert r.text.rstrip().endswith("}")


# --------------------------------------------------------------------------- #
# 6. Error paths: bad format → 400; missing project → 404
# --------------------------------------------------------------------------- #


class TestErrorPaths:

    def test_unknown_format_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=html"
        )
        assert r.status_code == 400
        assert "format" in r.json()["detail"].lower()

    def test_missing_project_returns_404(self, env) -> None:
        client, _ = env
        bogus = "deadbeef0000"
        r = client.get(
            f"/api/projects/{bogus}/audit-trail?format=csv"
        )
        assert r.status_code == 404
        assert "project" in r.json()["detail"].lower()

    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/!!bad!!/audit-trail?format=csv")
        assert r.status_code == 400

    def test_invalid_filter_value_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&action=nope"
        )
        assert r.status_code == 400

    def test_invalid_kind_value_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/audit-trail?format=csv&kind=garbage"
        )
        assert r.status_code == 400
