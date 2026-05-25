"""Matrix export (F6.3) — CSV / XLSX of frequency + co-occurrence matrices.

Per PLANNING.md F6.3:

  > Frequency and co-occurrence matrix exports (CSV / XLSX).

F3.6 (:mod:`scribe.matrix`) gave us pure builders for the three matrix
views a QDA tool needs (code × source frequency, code × code
co-occurrence, code × attribute cross-tab) plus a CSV serialiser on
:class:`scribe.matrix.Matrix`. F6.3 surfaces those views as a download
artefact: CSV for the spreadsheet pasters, XLSX for the supervisor who
opens the file in Excel and expects formula-friendly numeric cells.

Like the sibling exporters (F2.6 codebook, F6.1 codebook surface, F6.2
retrieval report) this module is **pure**:

  * No FastAPI, no engine imports.
  * Inputs are :class:`scribe.matrix.Matrix` instances (built upstream
    via :mod:`scribe.matrix`) plus an optional :class:`scribe.projects.Project`
    for filename slugging.
  * Outputs are ``str`` (CSV) or ``bytes`` (XLSX).

Two formats, two backends
-------------------------

* :func:`to_csv` — thin wrapper over :meth:`Matrix.to_csv`. Kept here
  so callers see the same module surface for both formats. Returns
  RFC-4180 CSV with a top-left corner cell, optional totals row /
  column, and ``\\r\\n`` line endings (the :mod:`csv` module's default).

* :func:`to_xlsx` — pure-stdlib XLSX writer. Builds a minimal valid
  Office Open XML SpreadsheetML 1.0 package with one worksheet,
  inline-string cells for labels, numeric ``<v>`` cells for counts, and
  nothing else. No styles, no shared-strings table, no calc chain —
  just enough OOXML for Excel / LibreOffice / Numbers / Google Sheets
  to round-trip. Avoids the openpyxl dependency entirely; the trade-off
  is that exports are unstyled (which matches what a researcher
  actually wants for a frequency table — a clean numeric grid they can
  format themselves).

Why inline strings, not a shared-strings table
----------------------------------------------

A typical Scribe matrix is small (≤ a few hundred codes × sources) and
its labels are mostly unique (code names rarely repeat). Inline strings
keep the writer to a single XML stream and are well within Excel's
reading tolerance. Shared-strings would save a few KB on a 10,000-cell
matrix; we'd rather keep the implementation minimal.

Format registry (mirror of F2.6 / F6.1 / F6.2)
---------------------------------------------

:data:`EXPORT_FORMATS` exposes ``{key -> FormatSpec(extension, media_type, label)}``
for the user-facing formats so a UI button or HTTP endpoint can render
the same way the CLI does. :func:`normalise_format` accepts the canonical
keys plus the obvious aliases (``xls`` / ``excel`` / ``spreadsheet`` →
``xlsx``).

Matrix kinds
------------

The CLI / HTTP surface needs a label for the matrix kind so it can
slug the filename and pick the right F3.6 builder.
:data:`MATRIX_KINDS` lists the three:

  * ``code-by-source`` — frequency.
  * ``code-by-code`` — co-occurrence (with optional scope / max_gap).
  * ``code-by-attribute`` — cross-tab.

:func:`normalise_matrix_kind` accepts the canonical keys plus aliases
(``frequency`` → ``code-by-source``, ``cooccurrence`` /
``co-occurrence`` → ``code-by-code``, ``cross-tab`` → ``code-by-attribute``).
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape as _xml_escape

from .matrix import Matrix
from .projects import Project


# --------------------------------------------------------------------------- #
# Format registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FormatSpec:
    """Static description of a user-facing matrix export format.

    Mirrors :class:`scribe.codebook_export.FormatSpec` and
    :class:`scribe.retrieval_report.FormatSpec` so the UI / HTTP surface
    can iterate one shape across all the F6.* exporters.
    """

    key: str
    extension: str
    media_type: str
    label: str
    binary: bool = False


EXPORT_FORMAT_CSV = "csv"
EXPORT_FORMAT_XLSX = "xlsx"


EXPORT_FORMATS: dict[str, FormatSpec] = {
    EXPORT_FORMAT_CSV: FormatSpec(
        key=EXPORT_FORMAT_CSV,
        extension=".csv",
        media_type="text/csv; charset=utf-8",
        label="CSV",
        binary=False,
    ),
    EXPORT_FORMAT_XLSX: FormatSpec(
        key=EXPORT_FORMAT_XLSX,
        extension=".xlsx",
        # Office Open XML SpreadsheetML media type per ECMA-376 +
        # IANA registration. Browsers recognise this and offer "Open
        # in Excel" rather than "save as text".
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        label="Excel (XLSX)",
        binary=True,
    ),
}


_FORMAT_ALIASES: dict[str, str] = {
    "csv": EXPORT_FORMAT_CSV,
    "xlsx": EXPORT_FORMAT_XLSX,
    # Researchers say "Excel" or "spreadsheet"; the CLI accepts both.
    "xls": EXPORT_FORMAT_XLSX,
    "excel": EXPORT_FORMAT_XLSX,
    "spreadsheet": EXPORT_FORMAT_XLSX,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Trims + lower-cases. Recognises ``xls`` / ``excel`` / ``spreadsheet``
    as aliases for ``xlsx`` so the CLI / query string is forgiving.
    Raises :class:`ValueError` for unknown formats with the list of
    accepted keys, so the message is actionable.
    """
    if format is None:
        raise ValueError(
            "Matrix export format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported matrix export format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


# --------------------------------------------------------------------------- #
# Matrix kinds (for filenames, CLI flags, eventual UI buttons)
# --------------------------------------------------------------------------- #


MATRIX_KIND_CODE_BY_SOURCE = "code-by-source"
MATRIX_KIND_CODE_BY_CODE = "code-by-code"
MATRIX_KIND_CODE_BY_ATTRIBUTE = "code-by-attribute"

MATRIX_KINDS: tuple[str, ...] = (
    MATRIX_KIND_CODE_BY_SOURCE,
    MATRIX_KIND_CODE_BY_CODE,
    MATRIX_KIND_CODE_BY_ATTRIBUTE,
)


_MATRIX_KIND_ALIASES: dict[str, str] = {
    "code-by-source": MATRIX_KIND_CODE_BY_SOURCE,
    "code_by_source": MATRIX_KIND_CODE_BY_SOURCE,
    "code-x-source": MATRIX_KIND_CODE_BY_SOURCE,
    "frequency": MATRIX_KIND_CODE_BY_SOURCE,
    "freq": MATRIX_KIND_CODE_BY_SOURCE,
    "code-by-code": MATRIX_KIND_CODE_BY_CODE,
    "code_by_code": MATRIX_KIND_CODE_BY_CODE,
    "code-x-code": MATRIX_KIND_CODE_BY_CODE,
    "co-occurrence": MATRIX_KIND_CODE_BY_CODE,
    "cooccurrence": MATRIX_KIND_CODE_BY_CODE,
    "co-occur": MATRIX_KIND_CODE_BY_CODE,
    "code-by-attribute": MATRIX_KIND_CODE_BY_ATTRIBUTE,
    "code_by_attribute": MATRIX_KIND_CODE_BY_ATTRIBUTE,
    "code-x-attribute": MATRIX_KIND_CODE_BY_ATTRIBUTE,
    "cross-tab": MATRIX_KIND_CODE_BY_ATTRIBUTE,
    "crosstab": MATRIX_KIND_CODE_BY_ATTRIBUTE,
    "attribute": MATRIX_KIND_CODE_BY_ATTRIBUTE,
}


def normalise_matrix_kind(kind: str | None) -> str:
    """Resolve a caller-supplied matrix-kind string to a canonical key.

    Used by the CLI / HTTP surface so they can route to the right F3.6
    builder. Pure: doesn't actually build the matrix.
    """
    if kind is None:
        raise ValueError(
            "Matrix kind is required; expected one of: "
            f"{list(MATRIX_KINDS)}"
        )
    key = str(kind).strip().lower()
    if key in _MATRIX_KIND_ALIASES:
        return _MATRIX_KIND_ALIASES[key]
    raise ValueError(
        f"Unsupported matrix kind: {kind!r}. "
        f"Expected one of: {list(MATRIX_KINDS)}"
    )


# --------------------------------------------------------------------------- #
# CSV — thin wrapper over Matrix.to_csv
# --------------------------------------------------------------------------- #


def to_csv(
    matrix: Matrix,
    *,
    use_titles: bool = True,
    include_totals: bool = True,
) -> str:
    """Serialise a matrix to CSV.

    Thin wrapper over :meth:`Matrix.to_csv`; kept here so the export
    module surface looks the same as F2.6 / F6.1 / F6.2 (each shipped
    a ``to_csv`` of its own).

    The corner cell is the matrix title (or row label fallback);
    column / row labels prefer ``col_titles`` / ``row_titles`` when
    ``use_titles=True``. Totals row + column are included by default
    so a glance tells you the marginal counts.
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"to_csv expected a Matrix; got {type(matrix).__name__}"
        )
    return matrix.to_csv(
        use_titles=use_titles, include_totals=include_totals
    )


# --------------------------------------------------------------------------- #
# XLSX — minimal pure-stdlib writer
# --------------------------------------------------------------------------- #


# Office Open XML namespaces. We lock the strings in module-level
# constants because they appear in dozens of places across the part
# files and a single typo silently produces a "corrupt file" prompt
# in Excel that no test can catch.
_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL_DOC = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_NS_REL_PKG = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
_NS_CT_PKG = (
    "http://schemas.openxmlformats.org/package/2006/content-types"
)
# Content types for the main parts.
_CT_WORKBOOK = (
    "application/vnd.openxmlformats-officedocument."
    "spreadsheetml.sheet.main+xml"
)
_CT_WORKSHEET = (
    "application/vnd.openxmlformats-officedocument."
    "spreadsheetml.worksheet+xml"
)
_CT_CORE_PROPS = (
    "application/vnd.openxmlformats-package.core-properties+xml"
)
_CT_RELS = "application/vnd.openxmlformats-package.relationships+xml"

# Relationship type URIs used by .rels files.
_REL_TYPE_OFFICE_DOC = (
    "http://schemas.openxmlformats.org/officeDocument/"
    "2006/relationships/officeDocument"
)
_REL_TYPE_WORKSHEET = (
    "http://schemas.openxmlformats.org/officeDocument/"
    "2006/relationships/worksheet"
)
_REL_TYPE_CORE_PROPS = (
    "http://schemas.openxmlformats.org/package/"
    "2006/relationships/metadata/core-properties"
)


# Excel sheet-name rules: ≤ 31 chars, no ``: \\ / ? * [ ]`` and not
# empty / not a leading/trailing apostrophe. We slug aggressively to
# stay safe across locales.
_SHEET_NAME_BAD = re.compile(r"[\\/:?*\[\]]")
_SHEET_NAME_MAX = 31


def _safe_sheet_name(name: str) -> str:
    """Sanitise a string into a valid Excel sheet name."""
    cleaned = _SHEET_NAME_BAD.sub(" ", str(name or "")).strip()
    cleaned = cleaned.strip("'")
    if not cleaned:
        cleaned = "Matrix"
    if len(cleaned) > _SHEET_NAME_MAX:
        cleaned = cleaned[:_SHEET_NAME_MAX].rstrip()
    return cleaned


def _column_letter(index_one_based: int) -> str:
    """Convert a 1-based column index to its Excel letter ('A', ...,
    'AA', ...). Raises :class:`ValueError` for non-positive input."""
    if index_one_based <= 0:
        raise ValueError(
            f"Column index must be ≥ 1; got {index_one_based}"
        )
    n = int(index_one_based)
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _is_int_like(v: object) -> bool:
    """True when ``v`` is an integer-shaped scalar that should be
    written as a numeric XLSX cell. Booleans pass through ``int``
    in Python so we filter them out — booleans don't appear in
    Matrix cells today but we guard against future extensions.
    """
    if isinstance(v, bool):
        return False
    return isinstance(v, int)


def _xml_cell(coord: str, value: object) -> str:
    """Render a single ``<c>`` element for the given coordinate."""
    if _is_int_like(value):
        # Numeric cell. Default cell type is "n" so we can omit t=.
        return f'<c r="{coord}"><v>{int(value)}</v></c>'
    if value is None:
        # Empty cells don't need a <c> element; emit one with an
        # explicit inlineStr so the totals/labels stay aligned.
        return f'<c r="{coord}" t="inlineStr"><is><t></t></is></c>'
    s = str(value)
    return (
        f'<c r="{coord}" t="inlineStr">'
        f'<is><t xml:space="preserve">{_xml_escape(s)}</t></is>'
        f'</c>'
    )


def _xml_row(row_index_one_based: int, values: list[object]) -> str:
    """Render a single ``<row>`` element."""
    cells = "".join(
        _xml_cell(_column_letter(j + 1) + str(row_index_one_based), v)
        for j, v in enumerate(values)
    )
    return f'<row r="{row_index_one_based}">{cells}</row>'


def _matrix_table(
    matrix: Matrix,
    *,
    use_titles: bool,
    include_totals: bool,
) -> list[list[object]]:
    """Render the matrix to a list-of-lists with the same shape that
    ``Matrix.to_csv`` produces. Centralises the grid-shape logic so the
    CSV and XLSX writers stay byte-for-byte consistent on totals /
    labels.
    """
    table: list[list[object]] = []
    corner = matrix.title or matrix.row_label
    header: list[object] = [corner]
    for c in matrix.cols:
        if (
            use_titles
            and c in matrix.col_titles
            and matrix.col_titles[c]
        ):
            header.append(matrix.col_titles[c])
        else:
            header.append(c)
    if include_totals:
        header.append("Total")
    table.append(header)

    for r in matrix.rows:
        label = (
            matrix.row_titles[r]
            if (
                use_titles
                and r in matrix.row_titles
                and matrix.row_titles[r]
            )
            else r
        )
        row_out: list[object] = [label]
        for c in matrix.cols:
            row_out.append(matrix.get(r, c))
        if include_totals:
            row_out.append(matrix.row_total(r))
        table.append(row_out)

    if include_totals:
        footer: list[object] = ["Total"]
        for c in matrix.cols:
            footer.append(matrix.col_total(c))
        footer.append(matrix.grand_total())
        table.append(footer)

    return table


def _build_sheet_xml(table: list[list[object]]) -> bytes:
    """Render a ``xl/worksheets/sheet1.xml`` body."""
    rows_xml = "".join(
        _xml_row(i + 1, row) for i, row in enumerate(table)
    )
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<worksheet xmlns="{_NS_MAIN}">'
        f'<sheetData>{rows_xml}</sheetData>'
        f'</worksheet>'
    )
    return body.encode("utf-8")


def _build_workbook_xml(sheet_name: str) -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<workbook xmlns="{_NS_MAIN}" '
        f'xmlns:r="{_NS_REL_DOC}">'
        f'<sheets>'
        f'<sheet name="{_xml_escape(sheet_name)}" '
        f'sheetId="1" r:id="rId1"/>'
        f'</sheets>'
        f'</workbook>'
    )
    return body.encode("utf-8")


def _build_workbook_rels() -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{_NS_REL_PKG}">'
        f'<Relationship Id="rId1" Type="{_REL_TYPE_WORKSHEET}" '
        f'Target="worksheets/sheet1.xml"/>'
        f'</Relationships>'
    )
    return body.encode("utf-8")


def _build_root_rels() -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{_NS_REL_PKG}">'
        f'<Relationship Id="rId1" Type="{_REL_TYPE_OFFICE_DOC}" '
        f'Target="xl/workbook.xml"/>'
        f'<Relationship Id="rId2" Type="{_REL_TYPE_CORE_PROPS}" '
        f'Target="docProps/core.xml"/>'
        f'</Relationships>'
    )
    return body.encode("utf-8")


def _build_content_types() -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Types xmlns="{_NS_CT_PKG}">'
        f'<Default Extension="rels" ContentType="{_CT_RELS}"/>'
        f'<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/xl/workbook.xml" '
        f'ContentType="{_CT_WORKBOOK}"/>'
        f'<Override PartName="/xl/worksheets/sheet1.xml" '
        f'ContentType="{_CT_WORKSHEET}"/>'
        f'<Override PartName="/docProps/core.xml" '
        f'ContentType="{_CT_CORE_PROPS}"/>'
        f'</Types>'
    )
    return body.encode("utf-8")


def _build_core_props(title: str, created_iso: str) -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/'
        'package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:title>{_xml_escape(title)}</dc:title>'
        f'<dc:creator>Scribe</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">'
        f'{_xml_escape(created_iso)}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">'
        f'{_xml_escape(created_iso)}</dcterms:modified>'
        '</cp:coreProperties>'
    )
    return body.encode("utf-8")


# Stable timestamp used inside core.xml when the caller doesn't pass
# one; keeping it deterministic makes byte-for-byte identical XLSX
# output reproducible in tests. Actual creation time isn't relevant
# for a frequency-table export.
_DEFAULT_CREATED_ISO = "2026-01-01T00:00:00Z"


def to_xlsx(
    matrix: Matrix,
    *,
    use_titles: bool = True,
    include_totals: bool = True,
    sheet_name: str | None = None,
    created_iso: str | None = None,
) -> bytes:
    """Serialise a matrix to a minimal XLSX (Office Open XML) byte string.

    The returned bytes are a complete ZIP container with five parts:

      * ``[Content_Types].xml``
      * ``_rels/.rels``
      * ``docProps/core.xml`` (title, creator, timestamps)
      * ``xl/workbook.xml`` + ``xl/_rels/workbook.xml.rels``
      * ``xl/worksheets/sheet1.xml``

    No styles, no shared-strings table; integer cells go out as
    ``<c><v>N</v></c>`` so Excel treats them as numbers (formula-friendly,
    SUM-able). Label cells use inline strings so we don't need a
    separate sharedStrings part.

    ``sheet_name`` defaults to the matrix title (sanitised to ≤ 31
    chars and free of the Excel-forbidden characters). ``created_iso``
    is a deterministic placeholder by default so two byte-equal
    matrices produce two byte-equal XLSX files (the test suite leans
    on this).
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"to_xlsx expected a Matrix; got {type(matrix).__name__}"
        )
    matrix.validate()

    table = _matrix_table(
        matrix,
        use_titles=use_titles,
        include_totals=include_totals,
    )

    name = _safe_sheet_name(
        sheet_name or matrix.title or matrix.row_label or "Matrix"
    )
    created = (created_iso or _DEFAULT_CREATED_ISO).strip()

    sheet_xml = _build_sheet_xml(table)
    workbook_xml = _build_workbook_xml(name)
    workbook_rels = _build_workbook_rels()
    root_rels = _build_root_rels()
    content_types = _build_content_types()
    core_props = _build_core_props(matrix.title or "Matrix", created)

    buf = io.BytesIO()
    # Use ZIP_DEFLATED so the output is reasonably compact (XML
    # compresses to ~20% of source). Office accepts both stored and
    # deflated parts.
    with zipfile.ZipFile(
        buf, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        # Order matches the part dependency tree, but ZIP order is
        # not load-bearing for OOXML parsers — Excel reads parts via
        # the relationship graph, not file ordering.
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/core.xml", core_props)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Render dispatch
# --------------------------------------------------------------------------- #


_RENDERERS: dict[str, Callable[..., bytes]] = {}


def render_matrix(
    format: str,
    matrix: Matrix,
    *,
    use_titles: bool = True,
    include_totals: bool = True,
) -> bytes:
    """Render a matrix as ``bytes`` in ``format``.

    Returns ``bytes`` for both formats so the dispatch layer (HTTP /
    file write) doesn't have to know which formats are textual. The CSV
    bytes are UTF-8-encoded; the XLSX bytes are the raw ZIP container.
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](
        matrix,
        use_titles=use_titles,
        include_totals=include_totals,
    )


def _render_csv(
    matrix: Matrix,
    *,
    use_titles: bool,
    include_totals: bool,
) -> bytes:
    return to_csv(
        matrix,
        use_titles=use_titles,
        include_totals=include_totals,
    ).encode("utf-8")


def _render_xlsx(
    matrix: Matrix,
    *,
    use_titles: bool,
    include_totals: bool,
) -> bytes:
    return to_xlsx(
        matrix,
        use_titles=use_titles,
        include_totals=include_totals,
    )


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_XLSX] = _render_xlsx


# --------------------------------------------------------------------------- #
# Filename + atomic disk write
# --------------------------------------------------------------------------- #


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def _project_slug(project: Project | None) -> str:
    """ASCII-only, dash-separated slug from a project's name."""
    if project is None or not project.name or not project.name.strip():
        return ""
    ascii_name = (
        unicodedata.normalize("NFKD", project.name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_RE.sub("-", ascii_name.lower()).strip("-")
    if len(slug) > _FILENAME_SLUG_MAX:
        slug = slug[:_FILENAME_SLUG_MAX].rstrip("-")
    return slug


def slugify_matrix_filename(
    project: Project | None,
    format: str,
    kind: str = MATRIX_KIND_CODE_BY_SOURCE,
) -> str:
    """Build a download-friendly filename for a matrix export.

    Pattern:

      * ``<project-slug>-<kind>-matrix<ext>`` when the project has a name.
      * ``<kind>-matrix<ext>`` otherwise.

    ``kind`` is normalised through :func:`normalise_matrix_kind` (so
    aliases like ``cooccurrence`` resolve to the canonical
    ``code-by-code``). Raises :class:`ValueError` for unknown formats /
    kinds — the caller wants to know.
    """
    fmt = normalise_format(format)
    spec = EXPORT_FORMATS[fmt]
    canonical_kind = normalise_matrix_kind(kind)
    slug = _project_slug(project)
    base = f"{canonical_kind}-matrix{spec.extension}"
    if slug:
        return f"{slug}-{base}"
    return base


def write_matrix(
    path: Path,
    format: str,
    matrix: Matrix,
    *,
    use_titles: bool = True,
    include_totals: bool = True,
) -> Path:
    """Render the matrix in ``format`` and atomically write to ``path``.

    Atomic via a ``.tmp`` swap so an interrupted write never leaves a
    half-finished export visible to other readers. Creates
    ``path.parent`` if missing. Returns the resolved target path for
    chaining convenience.
    """
    fmt = normalise_format(format)
    payload = render_matrix(
        fmt,
        matrix,
        use_titles=use_titles,
        include_totals=include_totals,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- #
# Convenience: deterministic ISO timestamp helper for callers that want
# the embedded core-properties timestamp to match an explicit clock.
# --------------------------------------------------------------------------- #


def utc_iso_now() -> str:
    """Return ``YYYY-MM-DDTHH:MM:SSZ`` for the current UTC instant.

    Provided so server / CLI callers can pass a real timestamp into
    :func:`to_xlsx` without each having to redo the W3CDTF formatting
    dance.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "EXPORT_FORMATS",
    "EXPORT_FORMAT_CSV",
    "EXPORT_FORMAT_XLSX",
    "FormatSpec",
    "MATRIX_KINDS",
    "MATRIX_KIND_CODE_BY_ATTRIBUTE",
    "MATRIX_KIND_CODE_BY_CODE",
    "MATRIX_KIND_CODE_BY_SOURCE",
    "normalise_format",
    "normalise_matrix_kind",
    "render_matrix",
    "slugify_matrix_filename",
    "to_csv",
    "to_xlsx",
    "utc_iso_now",
    "write_matrix",
]
