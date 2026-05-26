"""F6.2 reachability verification — coded-segment retrieval report.

The pure renderer lives in ``scribe.retrieval_report`` (shipped in
a24125c) and the CLI in ``scribe.scripts.export_retrieval_report``;
neither was reachable through the user-facing surface. This test
file is the explicit reachability anchor for F6.2:

  1. the queries page renders three F6.2 menu items
     (CSV / Markdown / Word(RTF)),
  2. each menu item points at the F6.2 endpoint with the right
     ``format=`` + ``group_by=`` query string,
  3. the endpoint returns 200 with the right Content-Type and a
     slugified attachment filename for each format,
  4. the alias set the user-facing query string accepts (``md``,
     ``word``, ``doc``, ``docx``) routes to the right renderer,
  5. the filter set (``code`` / ``source`` / ``coder`` /
     ``participant``) is repeatable + AND-combined, drops empty
     repeats, and survives an end-to-end round trip,
  6. unknown formats / group_by values return 400 with an actionable
     message,
  7. missing projects return 404 cleanly,
  8. transcript text is hydrated from ``outputs/<job>/edited.json``
     so a coded span renders as the actual quoted words rather than
     an empty cell.

The deeper pure-renderer coverage lives in
``tests/test_retrieval_report.py``; this file's job is purely the
F6.2 HTTP / UI reachability contract.
"""

from __future__ import annotations

import csv
import io
import json
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
    return TestClient(srv.app), projects, output


def _new_project(client: TestClient, name: str = "Pilot study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_corpus(
    projects_dir: Path,
    output_dir: Path,
    project_id: str,
    *,
    job_id: str = "abcdef012345",
):
    """Drop one source + a few codes / applications + a transcript.

    Returns a dict with the ids the caller needs to assert on.
    """
    from scribe import applications as a_mod
    from scribe import coders as cd_mod
    from scribe import codes as c_mod
    from scribe import sources as s_mod
    from scribe.applications import Application
    from scribe.coders import Coder
    from scribe.codes import Code
    from scribe.sources import Source

    coder = Coder.new(project_id=project_id, name="Field worker")
    cd_mod.save_coder(projects_dir, coder)

    c1 = Code.new(
        project_id=project_id,
        name="Pacing the day",
        definition="Adjusting daily activity to manage limited energy.",
    )
    c2 = Code.new(
        project_id=project_id,
        name="Holding back",
        definition="Withholding information from kin.",
    )
    c_mod.save_code(projects_dir, c1)
    c_mod.save_code(projects_dir, c2)

    source = Source.new(
        project_id=project_id,
        name="Interview 1",
        source_type="transcript",
        transcript_job_id=job_id,
    )
    s_mod.save_source(projects_dir, source)

    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({
        "segments": [
            {
                "speaker": "PARTICIPANT",
                "words": [
                    {"text": "Hello"},
                    {"text": "world"},
                    {"text": "today"},
                ],
            },
            {
                "speaker": "PARTICIPANT",
                "words": [
                    {"text": "Goodbye"},
                    {"text": "everyone"},
                ],
            },
        ],
    }))

    a1 = Application.new(
        project_id=project_id,
        code_id=c1.id,
        source_id=source.id,
        coder_id=coder.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w1",
        definition_version_id_at_apply="aaaabbbbcccc",
    )
    a2 = Application.new(
        project_id=project_id,
        code_id=c2.id,
        source_id=source.id,
        coder_id=coder.id,
        anchor_start_word_id="s1w0",
        anchor_end_word_id="s1w0",
        definition_version_id_at_apply="aaaabbbbcccc",
    )
    a_mod.save_application(projects_dir, a1)
    a_mod.save_application(projects_dir, a2)

    return {
        "source_id": source.id,
        "coder_id": coder.id,
        "code1_id": c1.id,
        "code2_id": c2.id,
        "app1_id": a1.id,
        "app2_id": a2.id,
        "job_id": job_id,
    }


# --------------------------------------------------------------------------- #
# Template render: the three F6.2 download links are present in the
# queries page header
# --------------------------------------------------------------------------- #


class TestQueriesPageRendersF6_2Links:
    """The queries page must render three menu items pointing at the
    F6.2 endpoint, one per format. If any of them ever drops off the
    user can no longer reach the export through the UI."""

    def test_csv_link_in_queries_page(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/retrieval-report?'
            f"format=csv&group_by=code\"" in r.text
        )

    def test_markdown_link_in_queries_page(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/retrieval-report?'
            f"format=markdown&group_by=code\"" in r.text
        )

    def test_rtf_link_in_queries_page(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/retrieval-report?'
            f"format=rtf&group_by=code\"" in r.text
        )

    def test_links_carry_download_attribute(self, env) -> None:
        """``download`` makes the browser save rather than render."""
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        # Coarse: at least three items, each with ``download``.
        assert r.text.count('class="rr-export-item"') >= 3
        assert "download" in r.text

    def test_export_button_carries_feature_marker(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        # The ``data-test-feature="F6.2"`` marker is what the loop's
        # detector + manual review key off when checking that this
        # surface exists.
        assert 'data-test-feature="F6.2"' in r.text
        assert 'id="rr-export-btn"' in r.text


# --------------------------------------------------------------------------- #
# Endpoint contract: each format returns 200 + the right Content-Type
# --------------------------------------------------------------------------- #


class TestExportEndpointContract:
    """Each F6.2 format must respond with the documented MIME type
    and a slugified attachment filename."""

    def test_csv_returns_text_csv_with_attachment(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-coded-segments.csv" in cd

    def test_markdown_returns_text_markdown_with_attachment(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=markdown"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/markdown")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-coded-segments.md" in cd

    def test_rtf_returns_application_rtf_with_attachment(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=rtf")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/rtf"
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-coded-segments.rtf" in cd

    def test_default_format_is_csv(self, env) -> None:
        """No ``format=`` query string falls through to CSV."""
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(f"/api/projects/{pid}/retrieval-report")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")


# --------------------------------------------------------------------------- #
# End-to-end content: code names + quoted text show up in the body
# --------------------------------------------------------------------------- #


class TestEndToEndContent:
    """Once codes + applications + a transcript are seeded, the F6.2
    download must contain them. This is the core 'reachability' claim
    — the user clicks 📥 Coded segments → CSV, gets the rows."""

    def test_csv_body_contains_seeded_applications(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        ids = _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=csv")
        assert r.status_code == 200
        # Parse the CSV; both code names + the quoted text must
        # be present in their respective rows.
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        assert len(rows) == 2
        names = {row["code_name"] for row in rows}
        assert names == {"Pacing the day", "Holding back"}
        # Quoted text from edited.json: "Hello world" for the first
        # application (s0w0..s0w1) and "Goodbye" for the second.
        texts = {row["text"] for row in rows}
        assert "Hello world" in texts
        assert "Goodbye" in texts

    def test_markdown_body_groups_by_code_by_default(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=markdown"
        )
        assert r.status_code == 200
        # Both code names appear as headings somewhere in the body.
        assert "Pacing the day" in r.text
        assert "Holding back" in r.text
        # The Markdown renderer prefaces with a top-level heading.
        assert "# " in r.text

    def test_rtf_body_starts_with_rtf_signature(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=rtf")
        assert r.status_code == 200
        # Minimal RTF documents always start with ``{\rtf1``.
        assert r.text.startswith("{\\rtf1")
        assert "Pacing the day" in r.text


# --------------------------------------------------------------------------- #
# Format aliases: ``md`` / ``MARKDOWN``, ``word`` / ``doc`` / ``docx``
# --------------------------------------------------------------------------- #


class TestFormatAliases:
    """Documented aliases must route to the right renderer; the F6.2
    HTTP surface inherits its alias set from the pure module so a
    user who learns the CLI flag carries it over to the URL."""

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
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format={alias}"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith(expected_ct)
        assert expected_ext in r.headers.get("content-disposition", "")


# --------------------------------------------------------------------------- #
# Filters: code / source / coder / participant are repeatable + AND-combined
# --------------------------------------------------------------------------- #


class TestFilters:
    """The four documented filters must shrink the result set.
    Repeated values stack within a filter (OR); different filters
    AND-combine. Empty repeats (``?code=&code=``) are silently
    dropped — the URL surface intentionally collapses the pure
    module's "match-nothing on empty list" branch."""

    def test_code_filter_restricts_rows(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        ids = _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=csv&code={ids['code1_id']}"
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 1
        assert rows[0]["code_name"] == "Pacing the day"

    def test_source_filter_restricts_rows(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        ids = _seed_corpus(projects_dir, output_dir, pid)
        # Filter to the seeded source — both rows survive.
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=csv&source={ids['source_id']}"
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 2

    def test_filter_combination_is_AND(self, env) -> None:
        """Code1 + Coder = the row coded by Coder under Code1."""
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        ids = _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=csv"
            f"&code={ids['code1_id']}&coder={ids['coder_id']}"
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 1
        assert rows[0]["code_name"] == "Pacing the day"
        assert rows[0]["coder_name"] == "Field worker"

    def test_empty_repeats_are_silently_dropped(self, env) -> None:
        """``?code=&code=`` should behave like "no filter", not "no rows"."""
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=csv&code=&code="
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 2

    def test_unknown_code_id_yields_empty_result(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=csv&code=zzzzzzzzzzzz"
        )
        assert r.status_code == 200
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert rows == []


# --------------------------------------------------------------------------- #
# Group-by: the URL parameter must influence the rendered headings
# --------------------------------------------------------------------------- #


class TestGroupBy:
    """``group_by`` controls the heading axis in the Markdown / RTF
    bodies. CSV is flat by contract — the parameter is honoured but
    the schema doesn't change."""

    def test_group_by_source_produces_source_heading(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=markdown"
            f"&group_by=source"
        )
        assert r.status_code == 200
        # The source's name is "Interview 1"; with group_by=source it
        # appears as a heading.
        assert "Interview 1" in r.text

    def test_group_by_alias_flat_routes_to_none(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        # ``flat`` is the F6.2 alias for ``none``; if the alias
        # routing breaks, this surfaces as a 400.
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=markdown"
            f"&group_by=flat"
        )
        assert r.status_code == 200

    def test_unknown_group_by_returns_400(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(projects_dir, output_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?group_by=banana"
        )
        assert r.status_code == 400
        assert "banana" in r.text.lower() or "group" in r.text.lower()


# --------------------------------------------------------------------------- #
# Failure modes: unknown formats, missing projects
# --------------------------------------------------------------------------- #


class TestFailureModes:
    def test_unknown_format_returns_400(self, env) -> None:
        client, projects_dir, output_dir = env
        pid = _new_project(client, name="Pilot study")
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=xml")
        assert r.status_code == 400
        # Actionable: error message mentions the format name.
        assert "xml" in r.text.lower() or "unsupported" in r.text.lower()

    def test_missing_project_returns_404(self, env) -> None:
        client, _, _ = env
        # Valid project-id shape (12 hex chars) but no project on disk.
        r = client.get(
            "/api/projects/abcdef012345/retrieval-report?format=csv"
        )
        assert r.status_code == 404

    def test_malformed_project_id_returns_400(self, env) -> None:
        client, _, _ = env
        # ``..`` is rejected by the project-id-format guard.
        r = client.get(
            "/api/projects/..bad../retrieval-report?format=csv"
        )
        # FastAPI treats `..` segments specially in URLs; the path
        # is still routed but the project-id-format check fails.
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# Empty-project case: the surface returns a header-only / placeholder
# body rather than 404 when nothing has been coded yet.
# --------------------------------------------------------------------------- #


class TestEmptyProjectF6_2:
    def test_csv_empty_project_is_header_only(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client, name="Empty study")
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=csv")
        assert r.status_code == 200
        # CSV with only the header row + newline; DictReader sees no rows.
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert rows == []

    def test_markdown_empty_project_renders_placeholder(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client, name="Empty study")
        r = client.get(
            f"/api/projects/{pid}/retrieval-report?format=markdown"
        )
        assert r.status_code == 200
        # The renderer always emits a top-level heading even on empty
        # input.
        assert "# " in r.text

    def test_rtf_empty_project_is_minimal_rtf(self, env) -> None:
        client, _, _ = env
        pid = _new_project(client, name="Empty study")
        r = client.get(f"/api/projects/{pid}/retrieval-report?format=rtf")
        assert r.status_code == 200
        assert r.text.startswith("{\\rtf1")
        assert r.text.endswith("}")
