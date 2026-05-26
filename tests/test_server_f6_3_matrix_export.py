"""F6.3 reachability verification — matrix export (CSV / XLSX).

The pure renderer + CLI shipped in c9243ed (``scribe.matrix_export``,
``scribe.scripts.export_matrix``); the F6.3 commit body explicitly
deferred the HTTP route + UI buttons. This file is the explicit
reachability anchor for the user-facing surface:

  1. the queries page renders the F6.3 download menu (button + two
     items, CSV / Excel) inside the matrix panel,
  2. the helper hrefs are rebuilt by JS via ``buildMatrixExportUrl``
     so the URL always reflects the form state,
  3. the new endpoint
     ``GET /api/projects/<pid>/matrices/<kind>/export?format=...``
     returns 200 with the right Content-Type + slugged
     Content-Disposition for both formats,
  4. the alias set the user-facing query string accepts (``xls`` /
     ``excel`` / ``spreadsheet``) routes to the same XLSX renderer,
  5. each of the three matrix kinds (``code-by-source`` /
     ``code-by-code`` / ``code-by-attribute``) round-trips end-to-end
     through the endpoint — including the ``code-by-attribute`` case
     that needs ``attribute_key``,
  6. unknown formats / kinds / attribute_kinds return 400 with an
     actionable message,
  7. missing projects return 404 cleanly,
  8. the empty-project case still responds 200 (header-only CSV /
     a one-cell XLSX shell), and
  9. the JS helpers are imported by the page so the URL math is the
     same on the client and the test path.

The deeper pure-renderer coverage stays in
``tests/test_matrix_export.py`` (65 cases) and the matrix-builder
coverage in ``tests/test_matrix.py``; this file's job is purely the
F6.3 HTTP / UI reachability contract.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated TestClient + tmp uploads/outputs/projects roots."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
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

    client = TestClient(srv.app)
    yield srv, client, projects, output


def _new_project(client: TestClient, name: str = "Pilot study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_code(client: TestClient, pid: str, name: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": ""},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_source(
    client: TestClient,
    pid: str,
    name: str = "Interview 1",
    *,
    custom_attributes: dict | None = None,
) -> str:
    payload: dict = {
        "name": name,
        "source_type": "transcript",
        "language": "en",
    }
    if custom_attributes is not None:
        payload["custom_attributes"] = custom_attributes
    r = client.post(f"/api/projects/{pid}/sources", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_app(
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


def _seed_corpus(client: TestClient, pid: str) -> dict:
    """Create one source with two codes applied so all three matrix
    kinds have data to render. Returns the ids the caller cares about."""
    c1 = _new_code(client, pid, "Pacing the day")
    c2 = _new_code(client, pid, "Holding back")
    s1 = _new_source(
        client, pid, "Interview A",
        custom_attributes={"setting": "home"},
    )
    s2 = _new_source(
        client, pid, "Interview B",
        custom_attributes={"setting": "clinic"},
    )
    a1 = _new_app(
        client, pid, code_id=c1, source_id=s1,
        anchor_start="s0w0", anchor_end="s0w1",
    )
    a2 = _new_app(
        client, pid, code_id=c2, source_id=s1,
        anchor_start="s1w0", anchor_end="s1w0",
    )
    a3 = _new_app(
        client, pid, code_id=c1, source_id=s2,
        anchor_start="s0w0", anchor_end="s0w0",
    )
    return {
        "code1": c1, "code2": c2,
        "source1": s1, "source2": s2,
        "app1": a1, "app2": a2, "app3": a3,
    }


# --------------------------------------------------------------------------- #
# Template render: the F6.3 download menu surfaces in the matrix panel
# --------------------------------------------------------------------------- #


class TestQueriesPageRendersF6_3Menu:
    """The queries page must render the matrix-download dropdown so
    the export is reachable from the matrix panel without the user
    typing the endpoint URL by hand."""

    def test_download_button_present(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert r.status_code == 200, r.text
        assert 'data-test-id="mx-export-btn"' in r.text
        assert 'data-test-feature="F6.3"' in r.text

    def test_csv_and_xlsx_menu_items_present(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert 'data-test-id="mx-export-csv"' in r.text
        assert 'data-test-id="mx-export-xlsx"' in r.text
        assert 'data-fmt="csv"' in r.text
        assert 'data-fmt="xlsx"' in r.text

    def test_menu_items_carry_download_attribute(self, env) -> None:
        """``download`` makes the browser save rather than render."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        # Coarse: at least two items, each with ``download``.
        assert r.text.count('class="mx-export-item"') >= 2
        # The mx-export-item anchors carry ``download``.
        # (every <a class="mx-export-item"> has a download attribute).
        # We cheap-check with a substring: the keyword "download" must
        # appear at least once after the menu starts.
        menu_pos = r.text.find('id="mx-export-menu"')
        assert menu_pos >= 0
        assert "download" in r.text[menu_pos:menu_pos + 2000]

    def test_page_imports_buildMatrixExportUrl_helper(self, env) -> None:
        """The form-state-to-URL math has to come from the same pure
        helper the JS unit tests cover. If this import drops, the URL
        algebra silently drifts between the runtime and the test path."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/queries")
        assert "buildMatrixExportUrl" in r.text


# --------------------------------------------------------------------------- #
# Endpoint contract: each format returns 200 + the right Content-Type
# --------------------------------------------------------------------------- #


class TestExportEndpointContract:
    """Each F6.3 format must respond with the documented MIME type
    and a slugified attachment filename."""

    def test_csv_returns_text_csv_with_attachment(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=csv"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-code-by-source-matrix.csv" in cd

    def test_xlsx_returns_spreadsheetml_with_attachment(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=xlsx"
        )
        assert r.status_code == 200, r.text
        ct = r.headers["content-type"]
        assert "spreadsheetml.sheet" in ct
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-code-by-source-matrix.xlsx" in cd

    def test_default_format_is_csv(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")


# --------------------------------------------------------------------------- #
# End-to-end content: the rendered body actually contains the data
# --------------------------------------------------------------------------- #


class TestEndToEndContent:
    """Round-trip the seeded corpus through each format and assert the
    counts make it through. Same fixture is reused; if the bytes ever
    change shape these tests will tell us."""

    def test_csv_body_contains_code_and_source_names(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=csv"
        )
        assert r.status_code == 200, r.text
        rows = list(csv.reader(io.StringIO(r.text)))
        # Header row carries the source titles.
        assert rows
        flat = "\n".join(",".join(row) for row in rows)
        assert "Pacing the day" in flat
        assert "Holding back" in flat
        assert "Interview A" in flat
        assert "Interview B" in flat
        # Totals row exists by default.
        assert any(row and row[0] == "Total" for row in rows)

    def test_xlsx_body_is_a_valid_zip_with_sheet1(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=xlsx"
        )
        assert r.status_code == 200, r.text
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = set(zf.namelist())
        assert "[Content_Types].xml" in names
        assert "xl/workbook.xml" in names
        assert "xl/worksheets/sheet1.xml" in names
        # The sheet contains the code names rendered as inline strings.
        sheet = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "Pacing the day" in sheet
        assert "Holding back" in sheet

    def test_code_by_code_kind_round_trips(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-code/export?format=csv"
        )
        assert r.status_code == 200, r.text
        flat = r.text
        # Code names appear on both axes.
        assert flat.count("Pacing the day") >= 2
        assert flat.count("Holding back") >= 2

    def test_code_by_attribute_kind_round_trips(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-attribute/export"
            "?format=csv&attribute_key=setting"
        )
        assert r.status_code == 200, r.text
        flat = r.text
        # Attribute values populate the column headers.
        assert "home" in flat
        assert "clinic" in flat
        assert "Pacing the day" in flat


# --------------------------------------------------------------------------- #
# Format aliases — the alias set the user-facing URL accepts
# --------------------------------------------------------------------------- #


class TestFormatAliases:
    """The matrix-export module documents these aliases. The endpoint
    must route them to the same renderer as the canonical key."""

    @pytest.mark.parametrize("alias", ["xls", "excel", "spreadsheet"])
    def test_xlsx_aliases_route_to_xlsx(self, env, alias) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format={alias}"
        )
        assert r.status_code == 200, r.text
        assert "spreadsheetml.sheet" in r.headers["content-type"]

    def test_csv_format_is_case_insensitive(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=CSV"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")

    @pytest.mark.parametrize(
        "alias",
        ["frequency", "freq", "code_by_source", "code-x-source"],
    )
    def test_code_by_source_aliases_route_to_canonical(self, env, alias) -> None:
        _, client, _, _ = env
        pid = _new_project(client, name="Alias study")
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/{alias}/export?format=csv"
        )
        assert r.status_code == 200, r.text
        # Canonical kind shows up in the slugged filename, not the
        # alias the caller used.
        cd = r.headers.get("content-disposition", "")
        assert "code-by-source-matrix.csv" in cd

    @pytest.mark.parametrize(
        "alias", ["cooccurrence", "co-occurrence", "code-x-code"],
    )
    def test_code_by_code_aliases_route_to_canonical(self, env, alias) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/{alias}/export?format=csv"
        )
        assert r.status_code == 200, r.text
        cd = r.headers.get("content-disposition", "")
        assert "code-by-code-matrix.csv" in cd

    @pytest.mark.parametrize("alias", ["cross-tab", "crosstab", "attribute"])
    def test_code_by_attribute_aliases_route_to_canonical(
        self, env, alias
    ) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/{alias}/export"
            "?format=csv&attribute_key=setting"
        )
        assert r.status_code == 200, r.text
        cd = r.headers.get("content-disposition", "")
        assert "code-by-attribute-matrix.csv" in cd


# --------------------------------------------------------------------------- #
# Failure modes: 400 / 404 with actionable messages
# --------------------------------------------------------------------------- #


class TestFailureModes:
    def test_unknown_format_400(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=pdf"
        )
        assert r.status_code == 400
        # The error mentions the accepted formats so a curl user can act.
        body = r.json()
        assert "csv" in str(body).lower()
        assert "xlsx" in str(body).lower()

    def test_unknown_kind_400(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/matrices/bogus/export?format=csv"
        )
        assert r.status_code == 400
        body = r.json()
        assert "code-by-source" in str(body)

    def test_missing_attribute_key_400(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-attribute/export?format=csv"
        )
        assert r.status_code == 400
        assert "attribute_key" in r.json()["detail"]

    def test_unknown_attribute_kind_400(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-attribute/export"
            "?format=csv&attribute_key=setting&attribute_kind=bogus"
        )
        assert r.status_code == 400
        assert "attribute_kind" in r.json()["detail"]

    def test_missing_project_404(self, env) -> None:
        _, client, _, _ = env
        # Valid 12-hex shape, but the project doesn't exist on disk.
        r = client.get(
            "/api/projects/aaaaaaaaaaaa/matrices/code-by-source/export?format=csv"
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Empty-project edge case
# --------------------------------------------------------------------------- #


class TestEmptyProjectF6_3:
    """An empty project still responds 200; the renderer produces a
    header-only CSV / a one-cell XLSX shell. Researchers see "no data"
    rather than a 4xx — which would suggest something is broken."""

    def test_empty_project_csv_200(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=csv"
        )
        assert r.status_code == 200, r.text

    def test_empty_project_xlsx_200(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=xlsx"
        )
        assert r.status_code == 200, r.text
        # Body is a valid zip with at least a sheet1.
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert "xl/worksheets/sheet1.xml" in zf.namelist()


# --------------------------------------------------------------------------- #
# Optional knobs round-trip
# --------------------------------------------------------------------------- #


class TestOptionalKnobs:
    """The query-string knobs (compact / use_titles / include_totals)
    map onto the corresponding pure-module flags."""

    def test_include_totals_off_drops_totals_row(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        r_on = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=csv"
        )
        r_off = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export"
            "?format=csv&include_totals=0"
        )
        assert r_on.status_code == 200 and r_off.status_code == 200
        # Default: a "Total" row footer.
        rows_on = list(csv.reader(io.StringIO(r_on.text)))
        assert any(row and row[0] == "Total" for row in rows_on)
        # include_totals=0: no "Total" row footer.
        rows_off = list(csv.reader(io.StringIO(r_off.text)))
        assert not any(row and row[0] == "Total" for row in rows_off)

    def test_compact_off_keeps_unused_codes(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        _seed_corpus(client, pid)
        # Add a code with no applications. With compact=1 (default)
        # it gets dropped; with compact=0 it stays.
        unused = _new_code(client, pid, "Untouched code")
        r_compact = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export?format=csv"
        )
        r_full = client.get(
            f"/api/projects/{pid}/matrices/code-by-source/export"
            "?format=csv&compact=0"
        )
        assert "Untouched code" not in r_compact.text
        assert "Untouched code" in r_full.text
