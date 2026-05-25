"""Tests for ``scribe.matrix_export`` (F6.3).

The module is pure: it takes a :class:`scribe.matrix.Matrix` (or builds
one upstream via :mod:`scribe.matrix`) and renders it as CSV or XLSX
bytes. Tests cover the format registry, the matrix-kind registry, the
CSV wrapper, the minimal-XLSX writer (including OOXML structural
checks), filename slugging, and the atomic disk-write helper.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scribe.codes import Code
from scribe.matrix import Matrix, code_by_source_matrix
from scribe.matrix_export import (
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_XLSX,
    EXPORT_FORMATS,
    FormatSpec,
    MATRIX_KINDS,
    MATRIX_KIND_CODE_BY_ATTRIBUTE,
    MATRIX_KIND_CODE_BY_CODE,
    MATRIX_KIND_CODE_BY_SOURCE,
    normalise_format,
    normalise_matrix_kind,
    render_matrix,
    slugify_matrix_filename,
    to_csv,
    to_xlsx,
    utc_iso_now,
    write_matrix,
)
from scribe.projects import Project
from scribe.sources import Source


PID = "deadbeef0001"


# --------------------------------------------------------------------------- #
# Tiny builders
# --------------------------------------------------------------------------- #


def _code(name: str) -> Code:
    return Code.new(project_id=PID, name=name)


def _src(name: str) -> Source:
    return Source.new(project_id=PID, name=name)


def _example_matrix() -> Matrix:
    """A small, hand-checked code × source matrix for round-trip tests."""
    c1, c2 = _code("Pacing"), _code("Resting")
    s1, s2 = _src("Interview 1"), _src("Interview 2")
    apps = [
        {"code_id": c1.id, "source_id": s1.id},
        {"code_id": c1.id, "source_id": s1.id},
        {"code_id": c2.id, "source_id": s2.id},
    ]
    return code_by_source_matrix(
        applications=apps, codes=[c1, c2], sources=[s1, s2]
    )


# --------------------------------------------------------------------------- #
# Format registry
# --------------------------------------------------------------------------- #


class TestFormatRegistry:
    def test_canonical_keys_are_csv_and_xlsx(self) -> None:
        assert set(EXPORT_FORMATS.keys()) == {
            EXPORT_FORMAT_CSV,
            EXPORT_FORMAT_XLSX,
        }

    def test_specs_are_format_spec_instances(self) -> None:
        for spec in EXPORT_FORMATS.values():
            assert isinstance(spec, FormatSpec)

    def test_csv_spec_is_text_with_charset(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_CSV]
        assert spec.extension == ".csv"
        assert "charset=utf-8" in spec.media_type
        assert spec.binary is False

    def test_xlsx_spec_is_binary_with_ooxml_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_XLSX]
        assert spec.extension == ".xlsx"
        assert (
            spec.media_type
            == "application/vnd.openxmlformats-officedocument."
               "spreadsheetml.sheet"
        )
        assert spec.binary is True


class TestNormaliseFormat:
    def test_canonical_csv(self) -> None:
        assert normalise_format("csv") == EXPORT_FORMAT_CSV

    def test_canonical_xlsx(self) -> None:
        assert normalise_format("xlsx") == EXPORT_FORMAT_XLSX

    def test_aliases_resolve_to_xlsx(self) -> None:
        for alias in ("xls", "excel", "spreadsheet", "EXCEL", " Xlsx "):
            assert normalise_format(alias) == EXPORT_FORMAT_XLSX

    def test_case_insensitive(self) -> None:
        assert normalise_format("CSV") == EXPORT_FORMAT_CSV
        assert normalise_format(" csv ") == EXPORT_FORMAT_CSV

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="format is required"):
            normalise_format(None)

    def test_unknown_raises_with_options(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            normalise_format("pdf")
        assert "csv" in str(excinfo.value)
        assert "xlsx" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Matrix kinds
# --------------------------------------------------------------------------- #


class TestMatrixKinds:
    def test_three_canonical_kinds(self) -> None:
        assert MATRIX_KINDS == (
            MATRIX_KIND_CODE_BY_SOURCE,
            MATRIX_KIND_CODE_BY_CODE,
            MATRIX_KIND_CODE_BY_ATTRIBUTE,
        )

    def test_canonical_kinds_round_trip(self) -> None:
        for k in MATRIX_KINDS:
            assert normalise_matrix_kind(k) == k

    def test_underscore_aliases(self) -> None:
        assert (
            normalise_matrix_kind("code_by_source")
            == MATRIX_KIND_CODE_BY_SOURCE
        )
        assert (
            normalise_matrix_kind("code_by_code")
            == MATRIX_KIND_CODE_BY_CODE
        )
        assert (
            normalise_matrix_kind("code_by_attribute")
            == MATRIX_KIND_CODE_BY_ATTRIBUTE
        )

    def test_x_aliases(self) -> None:
        assert (
            normalise_matrix_kind("code-x-source")
            == MATRIX_KIND_CODE_BY_SOURCE
        )

    def test_friendly_aliases(self) -> None:
        assert (
            normalise_matrix_kind("frequency")
            == MATRIX_KIND_CODE_BY_SOURCE
        )
        assert (
            normalise_matrix_kind("co-occurrence")
            == MATRIX_KIND_CODE_BY_CODE
        )
        assert (
            normalise_matrix_kind("cooccurrence")
            == MATRIX_KIND_CODE_BY_CODE
        )
        assert (
            normalise_matrix_kind("cross-tab")
            == MATRIX_KIND_CODE_BY_ATTRIBUTE
        )

    def test_case_insensitive_and_trimmed(self) -> None:
        assert (
            normalise_matrix_kind(" CODE-BY-CODE ")
            == MATRIX_KIND_CODE_BY_CODE
        )

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="kind is required"):
            normalise_matrix_kind(None)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported matrix kind"):
            normalise_matrix_kind("frequency-by-quarter")


# --------------------------------------------------------------------------- #
# CSV wrapper
# --------------------------------------------------------------------------- #


class TestToCsv:
    def test_emits_corner_then_columns_then_total(self) -> None:
        m = _example_matrix()
        csv = to_csv(m)
        first = csv.splitlines()[0]
        assert first.startswith("Code × Source,")
        assert first.endswith(",Total")
        assert "Interview 1" in first
        assert "Interview 2" in first

    def test_body_rows_sum_to_grand_total(self) -> None:
        m = _example_matrix()
        csv = to_csv(m)
        # last line is "Total,2,1,3"
        last = csv.splitlines()[-1]
        assert last == "Total,2,1,3"

    def test_use_titles_false_uses_keys(self) -> None:
        m = _example_matrix()
        csv = to_csv(m, use_titles=False)
        # Keys are 12-char hex; names like "Pacing" should not appear.
        assert "Pacing" not in csv
        assert "Resting" not in csv

    def test_include_totals_false_drops_totals(self) -> None:
        m = _example_matrix()
        csv = to_csv(m, include_totals=False)
        lines = csv.splitlines()
        assert "Total" not in lines[0]
        assert lines[-1].split(",")[0] != "Total"

    def test_rejects_non_matrix(self) -> None:
        with pytest.raises(TypeError):
            to_csv({"rows": [], "cols": []})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# XLSX writer
# --------------------------------------------------------------------------- #


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _open_xlsx(payload: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(payload))


def _read_sheet(payload: bytes) -> ET.Element:
    with _open_xlsx(payload) as zf:
        return ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))


class TestToXlsxStructure:
    def test_returns_bytes_starting_with_zip_magic(self) -> None:
        m = _example_matrix()
        out = to_xlsx(m)
        assert isinstance(out, bytes)
        assert out[:2] == b"PK"

    def test_contains_required_parts(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            names = set(zf.namelist())
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "xl/workbook.xml" in names
        assert "xl/_rels/workbook.xml.rels" in names
        assert "xl/worksheets/sheet1.xml" in names
        assert "docProps/core.xml" in names

    def test_content_types_declares_workbook_and_worksheet(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            ct = zf.read("[Content_Types].xml").decode()
        assert "/xl/workbook.xml" in ct
        assert "/xl/worksheets/sheet1.xml" in ct
        assert "spreadsheetml.sheet.main+xml" in ct
        assert "spreadsheetml.worksheet+xml" in ct

    def test_root_rels_points_at_workbook(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            rels = zf.read("_rels/.rels").decode()
        assert 'Target="xl/workbook.xml"' in rels

    def test_workbook_rels_points_at_sheet1(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            rels = zf.read("xl/_rels/workbook.xml.rels").decode()
        assert 'Target="worksheets/sheet1.xml"' in rels

    def test_each_part_parses_as_xml(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            for name in zf.namelist():
                # All package parts in our writer are XML.
                ET.fromstring(zf.read(name))


class TestToXlsxCells:
    def test_header_row_uses_inline_strings(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        first = rows[0]
        cells = first.findall(f"{_XLSX_NS}c")
        for c in cells:
            assert c.attrib.get("t") == "inlineStr"

    def test_numeric_cells_have_no_type_attr(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # Row 2 is the first data row: "Pacing | 2 | 0 | 2".
        cells = rows[1].findall(f"{_XLSX_NS}c")
        # cells[0] is the row label (inlineStr); cells[1:] are numeric.
        assert cells[0].attrib.get("t") == "inlineStr"
        for c in cells[1:]:
            assert c.attrib.get("t") is None
            v = c.find(f"{_XLSX_NS}v")
            assert v is not None
            int(v.text or "")  # parses cleanly

    def test_numeric_cell_values_match_matrix(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # Pacing row → 2, 0, 2 (with totals).
        pacing = rows[1].findall(f"{_XLSX_NS}c")
        values = [int(c.find(f"{_XLSX_NS}v").text) for c in pacing[1:]]  # type: ignore[union-attr,arg-type]
        assert values == [2, 0, 2]
        # Resting row → 0, 1, 1.
        resting = rows[2].findall(f"{_XLSX_NS}c")
        values = [int(c.find(f"{_XLSX_NS}v").text) for c in resting[1:]]  # type: ignore[union-attr,arg-type]
        assert values == [0, 1, 1]

    def test_totals_footer_row_present(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # 1 header + 2 data + 1 totals = 4
        assert len(rows) == 4
        last = rows[-1].findall(f"{_XLSX_NS}c")
        # Label "Total" then 2, 1, grand total 3.
        assert last[0].find(f"{_XLSX_NS}is/{_XLSX_NS}t").text == "Total"  # type: ignore[union-attr]
        nums = [int(c.find(f"{_XLSX_NS}v").text) for c in last[1:]]  # type: ignore[union-attr,arg-type]
        assert nums == [2, 1, 3]

    def test_unicode_cell_label_preserved(self) -> None:
        # The default row label "Code × Source" carries a non-ASCII '×'.
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m))
        first_cell = sheet.find(
            f"{_XLSX_NS}sheetData/{_XLSX_NS}row/{_XLSX_NS}c"
        )
        text_el = first_cell.find(f"{_XLSX_NS}is/{_XLSX_NS}t")  # type: ignore[union-attr]
        assert text_el.text == "Code × Source"  # type: ignore[union-attr]

    def test_xml_special_characters_are_escaped(self) -> None:
        m = Matrix(
            title="Plotter & <Painter>",
            row_label="Code",
            col_label="Source",
            rows=["aa"],
            cols=["bb"],
            row_titles={"aa": "<dangerous>"},
            col_titles={"bb": "A & B"},
            cells={("aa", "bb"): 1},
        )
        out = to_xlsx(m, include_totals=False)
        sheet = _read_sheet(out).find(f"{_XLSX_NS}sheetData")
        # Round-trip parse confirms escaping is correct.
        first_text = sheet.findall(  # type: ignore[union-attr]
            f"{_XLSX_NS}row/{_XLSX_NS}c/{_XLSX_NS}is/{_XLSX_NS}t"
        )[0]
        assert first_text.text == "Plotter & <Painter>"


class TestToXlsxOptions:
    def test_include_totals_false_drops_totals_row(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m, include_totals=False))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # 1 header + 2 data = 3.
        assert len(rows) == 3
        # Header has no "Total" column either.
        header = rows[0].findall(f"{_XLSX_NS}c")
        labels = [
            c.find(f"{_XLSX_NS}is/{_XLSX_NS}t").text  # type: ignore[union-attr]
            for c in header
        ]
        assert "Total" not in labels

    def test_use_titles_false_uses_keys(self) -> None:
        m = _example_matrix()
        sheet = _read_sheet(to_xlsx(m, use_titles=False))
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # First row, first cell is the title; cells [1:-1] are col keys.
        header = rows[0].findall(f"{_XLSX_NS}c")
        labels = [
            c.find(f"{_XLSX_NS}is/{_XLSX_NS}t").text  # type: ignore[union-attr]
            for c in header[1:-1]
        ]
        # All labels should be 12-char hex (the source ids), not names.
        for lab in labels:
            assert lab is not None
            assert re.match(r"^[0-9a-f]{12}$", lab)

    def test_sheet_name_default_uses_matrix_title(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m)) as zf:
            wb = zf.read("xl/workbook.xml").decode()
        # The non-ASCII × in the title is sanitised by xml escaping but
        # otherwise carried through.
        assert "Code × Source" in wb or "Code &#215; Source" in wb

    def test_sheet_name_explicit_overrides(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m, sheet_name="MySheet")) as zf:
            wb = zf.read("xl/workbook.xml").decode()
        assert 'name="MySheet"' in wb

    def test_sheet_name_truncated_to_31_chars(self) -> None:
        m = _example_matrix()
        long_name = "a" * 50
        with _open_xlsx(to_xlsx(m, sheet_name=long_name)) as zf:
            wb = zf.read("xl/workbook.xml").decode()
        # The 31-char cap means we never see "a" * 32 in the name attr.
        assert "a" * 32 not in wb
        assert "a" * 31 in wb

    def test_sheet_name_strips_forbidden_characters(self) -> None:
        m = _example_matrix()
        with _open_xlsx(to_xlsx(m, sheet_name="Bad/Name?:[]*")) as zf:
            wb = zf.read("xl/workbook.xml").decode()
        for ch in ("/", "?", ":", "[", "]", "*", "\\"):
            # The single attribute value must not contain forbidden chars.
            # We grep crudely on the part that holds the sheet name.
            m2 = re.search(r'name="([^"]*)"', wb)
            assert m2 is not None
            assert ch not in m2.group(1)

    def test_empty_matrix_produces_valid_xlsx(self) -> None:
        m = Matrix(title="Empty", row_label="Code", col_label="Source")
        out = to_xlsx(m)
        assert out[:2] == b"PK"
        sheet = _read_sheet(out)
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # Header only (no data rows, no totals because no rows/cols).
        # With include_totals=True and zero rows, we still get header
        # then a "Total" footer row with only the corner label.
        assert len(rows) >= 1

    def test_rejects_non_matrix(self) -> None:
        with pytest.raises(TypeError):
            to_xlsx("not a matrix")  # type: ignore[arg-type]


class TestToXlsxDeterminism:
    def test_same_matrix_same_bytes_with_explicit_timestamp(self) -> None:
        m = _example_matrix()
        a = to_xlsx(m, created_iso="2026-05-01T00:00:00Z")
        b = to_xlsx(m, created_iso="2026-05-01T00:00:00Z")
        assert a == b

    def test_default_timestamp_is_static(self) -> None:
        # Default timestamp is the module-level placeholder; calling
        # twice without an override still produces identical bytes.
        m = _example_matrix()
        assert to_xlsx(m) == to_xlsx(m)


# --------------------------------------------------------------------------- #
# render_matrix dispatch
# --------------------------------------------------------------------------- #


class TestRenderMatrix:
    def test_csv_returns_utf8_bytes(self) -> None:
        m = _example_matrix()
        out = render_matrix("csv", m)
        assert isinstance(out, bytes)
        assert out.decode("utf-8").startswith("Code × Source")

    def test_xlsx_returns_zip_bytes(self) -> None:
        m = _example_matrix()
        out = render_matrix("xlsx", m)
        assert out[:2] == b"PK"

    def test_alias_dispatches_to_xlsx(self) -> None:
        m = _example_matrix()
        out = render_matrix("excel", m)
        assert out[:2] == b"PK"

    def test_unknown_format_raises(self) -> None:
        m = _example_matrix()
        with pytest.raises(ValueError):
            render_matrix("pdf", m)

    def test_kwargs_pass_through_for_csv(self) -> None:
        m = _example_matrix()
        out = render_matrix(
            "csv", m, use_titles=False, include_totals=False
        ).decode()
        # No "Total" column, no human-readable code names.
        assert "Total" not in out.splitlines()[0]
        assert "Pacing" not in out

    def test_kwargs_pass_through_for_xlsx(self) -> None:
        m = _example_matrix()
        out = render_matrix(
            "xlsx", m, include_totals=False
        )
        sheet = _read_sheet(out)
        rows = sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row")
        # No totals → 1 header + 2 data = 3 rows.
        assert len(rows) == 3


# --------------------------------------------------------------------------- #
# slugify_matrix_filename
# --------------------------------------------------------------------------- #


class TestSlugifyFilename:
    def test_with_project_name(self) -> None:
        p = Project.new(name="Pilot Study")
        assert (
            slugify_matrix_filename(p, "csv", "code-by-source")
            == "pilot-study-code-by-source-matrix.csv"
        )

    def test_with_project_name_xlsx(self) -> None:
        p = Project.new(name="Café Doctorate")
        # NFKD normalises 'é' → 'e' so the slug is ASCII-only.
        out = slugify_matrix_filename(p, "xlsx", "code-by-code")
        assert out == "cafe-doctorate-code-by-code-matrix.xlsx"

    def test_no_project(self) -> None:
        out = slugify_matrix_filename(
            None, "csv", MATRIX_KIND_CODE_BY_SOURCE
        )
        assert out == "code-by-source-matrix.csv"

    def test_no_project_kind_alias_resolves(self) -> None:
        # Even when project is None the kind is still normalised so
        # aliases like ``frequency`` reach the canonical filename.
        assert (
            slugify_matrix_filename(None, "csv", "frequency")
            == "code-by-source-matrix.csv"
        )

    def test_kind_is_normalised(self) -> None:
        p = Project.new(name="Pilot")
        # alias 'frequency' → canonical 'code-by-source' in the filename
        assert (
            slugify_matrix_filename(p, "csv", "frequency")
            == "pilot-code-by-source-matrix.csv"
        )
        assert (
            slugify_matrix_filename(p, "csv", "cooccurrence")
            == "pilot-code-by-code-matrix.csv"
        )

    def test_long_project_name_is_capped(self) -> None:
        p = Project.new(name="a" * 200)
        out = slugify_matrix_filename(p, "csv", "code-by-source")
        # Slug capped at 80 chars; full filename longer due to suffix.
        assert out.startswith("a" * 80 + "-code-by-source-matrix.csv")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            slugify_matrix_filename(None, "pdf", "code-by-source")

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            slugify_matrix_filename(None, "csv", "code-by-quarter")


# --------------------------------------------------------------------------- #
# write_matrix
# --------------------------------------------------------------------------- #


class TestWriteMatrix:
    def test_csv_round_trip(self, tmp_path: Path) -> None:
        m = _example_matrix()
        target = tmp_path / "freq.csv"
        out = write_matrix(target, "csv", m)
        assert out == target
        assert target.read_text().startswith("Code × Source")

    def test_xlsx_round_trip(self, tmp_path: Path) -> None:
        m = _example_matrix()
        target = tmp_path / "freq.xlsx"
        out = write_matrix(target, "xlsx", m)
        assert out == target
        body = target.read_bytes()
        assert body[:2] == b"PK"
        # Re-open via zipfile to confirm structure.
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            assert "xl/worksheets/sheet1.xml" in zf.namelist()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        m = _example_matrix()
        deep = tmp_path / "a" / "b" / "c" / "out.csv"
        write_matrix(deep, "csv", m)
        assert deep.exists()

    def test_no_tmp_left_behind(self, tmp_path: Path) -> None:
        m = _example_matrix()
        target = tmp_path / "freq.csv"
        write_matrix(target, "csv", m)
        # The atomic write helper uses ``<name>.tmp``; the suffix file
        # must be gone after success.
        assert not target.with_name(target.name + ".tmp").exists()

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        m = _example_matrix()
        with pytest.raises(ValueError):
            write_matrix(tmp_path / "x", "pdf", m)


# --------------------------------------------------------------------------- #
# utc_iso_now helper
# --------------------------------------------------------------------------- #


class TestUtcIsoNow:
    def test_format_matches_w3cdtf(self) -> None:
        s = utc_iso_now()
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", s)
