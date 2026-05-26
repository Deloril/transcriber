"""F9.2 reachability — definition-at-apply audit report.

The pure renderer lives in :mod:`scribe.definition_at_apply` (shipped
in 2219577). It pairs every Application with the historical Code
definition that was in force at apply time (resolved from the F2.2
per-code version log) plus a drift summary against the current Code
state, and renders CSV / Markdown / RTF. Until this endpoint + the
audit-page download menu landed, the module had no user-facing
surface — researchers couldn't reach the audit-trail report through
the UI.

This test file is the F9.2 reachability anchor:

  1. The audit timeline page renders three F9.2 menu items
     (CSV / Markdown / Word(RTF)) pointing at the F9.2 endpoint.
  2. Each link carries the ``download`` attribute.
  3. The endpoint returns 200 with the right Content-Type +
     slugified attachment filename for each format.
  4. The format alias set the URL accepts (``md``, ``word``,
     ``doc``, ``docx``) routes to the right renderer.
  5. The CSV body matches :data:`CSV_COLUMNS`, contains seeded
     rows, and surfaces drift accurately when the current
     definition has moved on.
  6. Missing version logs are reported in the row's
     ``snapshot_missing`` column rather than failing the report.
  7. Empty projects produce a header-only CSV / placeholder
     Markdown / minimal RTF.
  8. Unknown formats return 400; missing projects return 404.

Deeper coverage of the renderer + drift / row builder lives in
``tests/test_definition_at_apply.py``; this file is purely the
HTTP / UI reachability contract.
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


def _seed_with_drift(projects_dir: Path, project_id: str) -> dict:
    """Drop a code, record a version, edit the code definition, attach
    one application pinned to the original version. The result has
    drift on the ``definition`` field — the report should flag it.
    """
    from scribe import applications as a_mod
    from scribe import code_versions as cv_mod
    from scribe import codes as c_mod
    from scribe import sources as s_mod
    from scribe.applications import Application
    from scribe.codes import Code
    from scribe.sources import Source

    source = Source.new(
        project_id=project_id,
        name="Interview 1",
        source_type="transcript",
    )
    s_mod.save_source(projects_dir, source)

    code = Code.new(
        project_id=project_id,
        name="Pacing the day",
        definition="Adjusting daily activity to manage limited energy.",
    )
    c_mod.save_code(projects_dir, code)
    v1 = cv_mod.record_code_version(
        projects_dir, code, change_note="initial",
    )

    # Move the definition on so the report has drift to surface.
    code.definition = (
        "Distributing tasks across the day to keep symptoms manageable."
    )
    c_mod.save_code(projects_dir, code)
    cv_mod.record_code_version(
        projects_dir, code, change_note="sharpened",
    )

    coder_id = "0" * 12
    app = Application.new(
        project_id=project_id,
        code_id=code.id,
        source_id=source.id,
        coder_id=coder_id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w2",
        definition_version_id_at_apply=v1.id,
    )
    a_mod.save_application(projects_dir, app)
    return {
        "source_id": source.id,
        "code_id": code.id,
        "coder_id": coder_id,
        "app_id": app.id,
        "v1_id": v1.id,
    }


# --------------------------------------------------------------------------- #
# 1. Template render: the audit page surfaces three F9.2 download links
# --------------------------------------------------------------------------- #


class TestAuditPageRendersF9_2Links:
    """The audit timeline page must expose three download menu items
    pointing at the F9.2 endpoint, one per format. If any drop off,
    the user can't reach the export through the UI."""

    def test_csv_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/definition-at-apply?format=csv"'
            in r.text
        )

    def test_markdown_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/definition-at-apply?'
            f'format=markdown"' in r.text
        )

    def test_rtf_link_in_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/definition-at-apply?format=rtf"'
            in r.text
        )

    def test_links_carry_download_attribute(self, env) -> None:
        """``download`` makes the browser save rather than navigate."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.text.count('class="daa-export-item"') >= 3
        assert "download" in r.text

    def test_audit_page_carries_feature_marker(self, env) -> None:
        """``data-test-feature="F9.2"`` is the loop's reachability
        anchor on this page."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert 'data-test-feature="F9.2"' in r.text
        assert 'id="daa-export-btn"' in r.text


# --------------------------------------------------------------------------- #
# 2. Endpoint contract: each format returns 200 + the documented MIME
# --------------------------------------------------------------------------- #


class TestExportEndpointContract:

    def test_csv_returns_text_csv_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pacing-study-definition-at-apply.csv" in cd

    def test_markdown_returns_text_markdown_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/definition-at-apply?format=markdown"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/markdown")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pacing-study-definition-at-apply.md" in cd

    def test_rtf_returns_application_rtf_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=rtf")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/rtf"
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pacing-study-definition-at-apply.rtf" in cd

    def test_default_format_is_csv(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")


# --------------------------------------------------------------------------- #
# 3. Format aliases: ``md`` / ``word`` / ``doc`` / ``docx`` route
# --------------------------------------------------------------------------- #


class TestFormatAliases:

    @pytest.mark.parametrize(
        "alias,expected_ct,expected_ext",
        [
            ("md", "text/markdown", ".md"),
            ("MARKDOWN", "text/markdown", ".md"),
            ("word", "application/rtf", ".rtf"),
            ("doc", "application/rtf", ".rtf"),
            ("docx", "application/rtf", ".rtf"),
            ("RTF", "application/rtf", ".rtf"),
            ("CSV", "text/csv", ".csv"),
        ],
    )
    def test_alias_routes_to_format(
        self, env, alias, expected_ct, expected_ext
    ) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/definition-at-apply?format={alias}"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith(expected_ct)
        assert expected_ext in r.headers.get("content-disposition", "")


# --------------------------------------------------------------------------- #
# 4. End-to-end content: rows + drift accurately surface in the body
# --------------------------------------------------------------------------- #


class TestEndToEndContent:

    def test_csv_columns_match_module_contract(self, env) -> None:
        from scribe.definition_at_apply import CSV_COLUMNS

        client, projects_dir = env
        pid = _new_project(client)
        _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=csv")
        assert r.status_code == 200
        reader = csv.DictReader(io.StringIO(r.text))
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == CSV_COLUMNS

    def test_csv_body_contains_seeded_application(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        ids = _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=csv")
        assert r.status_code == 200
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["application_id"] == ids["app_id"]
        assert row["code_id"] == ids["code_id"]
        assert row["version_id_at_apply"] == ids["v1_id"]
        # The original definition was on v1 ("Adjusting daily activity").
        assert "Adjusting daily activity" in row["definition_at_apply"]
        # The current definition has drifted.
        assert "Distributing tasks" in row["current_definition"]
        assert row["definition_drifted"] == "true"
        assert "definition" in row["drifted_fields"]
        assert row["snapshot_missing"] == "false"
        assert row["code_missing"] == "false"

    def test_markdown_body_includes_project_heading_and_drift(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pacing study")
        _seed_with_drift(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/definition-at-apply?format=markdown"
        )
        assert r.status_code == 200
        body = r.text
        # Top-level heading carries the project name.
        assert "# Definition at apply — Pacing study" in body
        # Drift section lands on the row.
        assert "Definition drift since apply" in body
        # Both "at apply" and "current" definitions are visible.
        assert "Adjusting daily activity" in body
        assert "Distributing tasks" in body

    def test_rtf_body_starts_with_rtf_signature(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_with_drift(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=rtf")
        assert r.status_code == 200
        # Minimal RTF documents always start with ``{\rtf1``.
        assert r.text.startswith("{\\rtf1")
        assert "Adjusting daily activity" in r.text


# --------------------------------------------------------------------------- #
# 5. Missing version log: the report must not fail
# --------------------------------------------------------------------------- #


class TestMissingSnapshotIsSurfaced:
    """An application whose ``definition_version_id_at_apply`` no
    longer resolves on disk (corruption / partial restore) should
    yield a ``snapshot_missing=true`` row, not a 500."""

    def test_missing_version_log_yields_snapshot_missing_row(
        self, env
    ) -> None:
        from scribe import applications as a_mod
        from scribe import codes as c_mod
        from scribe import sources as s_mod
        from scribe.applications import Application
        from scribe.codes import Code
        from scribe.sources import Source

        client, projects_dir = env
        pid = _new_project(client)
        source = Source.new(
            project_id=pid,
            name="Interview 1",
            source_type="transcript",
        )
        s_mod.save_source(projects_dir, source)
        code = Code.new(
            project_id=pid,
            name="Stub code",
            definition="Stub.",
        )
        c_mod.save_code(projects_dir, code)
        # No record_code_version call → log dir doesn't exist for this code.
        app = Application.new(
            project_id=pid,
            code_id=code.id,
            source_id=source.id,
            coder_id="0" * 12,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            # An id that never lived on disk.
            definition_version_id_at_apply="a" * 12,
        )
        a_mod.save_application(projects_dir, app)

        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=csv")
        assert r.status_code == 200, r.text
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 1
        assert rows[0]["snapshot_missing"] == "true"
        assert rows[0]["definition_at_apply"] == ""


# --------------------------------------------------------------------------- #
# 6. Empty project — header-only CSV / placeholder Markdown / minimal RTF
# --------------------------------------------------------------------------- #


class TestEmptyProject:

    def test_empty_csv_is_header_only(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=csv")
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert rows == []
        # First line is the header — non-empty.
        assert r.text.split("\r\n")[0].startswith("application_id")

    def test_empty_markdown_includes_no_applications_placeholder(
        self, env
    ) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/definition-at-apply?format=markdown"
        )
        assert r.status_code == 200
        assert "_(no applications)_" in r.text

    def test_empty_rtf_is_well_formed(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/definition-at-apply?format=rtf")
        assert r.status_code == 200
        assert r.text.startswith("{\\rtf1")
        assert r.text.rstrip().endswith("}")


# --------------------------------------------------------------------------- #
# 7. Error paths: bad format → 400; missing project → 404
# --------------------------------------------------------------------------- #


class TestErrorPaths:

    def test_unknown_format_returns_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/definition-at-apply?format=html"
        )
        assert r.status_code == 400
        assert "format" in r.json()["detail"].lower()

    def test_missing_project_returns_404(self, env) -> None:
        client, _ = env
        # Valid-shape id that doesn't exist on disk.
        bogus = "deadbeef0000"
        r = client.get(
            f"/api/projects/{bogus}/definition-at-apply?format=csv"
        )
        assert r.status_code == 404
        assert "project" in r.json()["detail"].lower()

    def test_invalid_project_id_returns_400(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/!!bad!!/definition-at-apply?format=csv")
        assert r.status_code == 400
