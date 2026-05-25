"""Matrix views (F3.6).

Per PLANNING.md F3.6:

  > Matrix views: code × source (frequency); code × code (co-occurrence);
  > code × attribute (cross-tab).

This module provides a small **pure-Python** matrix construction kit that
turns a list of code applications + reference data into the three flagship
matrix views researchers expect from a QDA tool. Like :mod:`scribe.query`
(F3.5) it is deliberately stand-alone — no FastAPI, no engine imports —
so callers can build matrices from saved queries, ad-hoc filters, or the
full corpus without any server in the loop.

What's here
-----------

  * :class:`Matrix` — a tiny grid abstraction with row keys, column keys,
    integer cells, display titles, and helpers for totals / CSV / JSON.
    Two matrices are equal iff their (rows, cols, non-zero cells) match.
  * :func:`code_by_source_matrix` — frequency: how often does each code
    appear in each source?
  * :func:`code_by_code_matrix` — co-occurrence: how often do pairs of
    codes appear together within a chosen scope (source / segment /
    paragraph)?
  * :func:`code_by_attribute_matrix` — cross-tab: how often does each
    code appear against the values of one source attribute (F3.2) or
    participant demographic (F1.3)?

Composes with F3.5: the natural pipeline is "filter applications via
:func:`scribe.query.applications_for_query`, then pass the result here".
F3.6 doesn't re-implement filtering.

Application shape
-----------------

Same generic dict (or attribute-bearing object) shape that F3.5 accepts:

  {
    "code_id":  "<12-hex>",            # required
    "source_id": "<12-hex>",           # required
    "speaker":  "<raw label>",         # optional, drives participant lookup
    "start":    <number>,              # optional, drives proximity scope
    "end":      <number>,              # optional, drives proximity scope
    "participant_id": "<12-hex>",      # optional, fallback to speaker_map lookup
  }

Co-occurrence semantics
-----------------------

The code × code matrix is **undirected** and **symmetric**. For two
distinct codes A and B we count *unordered* pairs ``{a, b}`` of
applications where ``a.code == A``, ``b.code == B``, and the two
applications co-occur within the chosen scope. The diagonal cell
``(A, A)`` is the number of unordered pairs of *distinct* applications
both with code A (so a code applied once in a single source contributes
zero to its own diagonal; a code applied four times to the same source
contributes ``4 choose 2`` = 6 to its diagonal).

The diagonal definition is a deliberate choice: it makes
"self-co-occurrence" comparable in scale to off-diagonal cells (both are
pair counts), and it makes ``cell(A, A) == 0`` if and only if A never
co-occurs with itself in scope. A common alternative — putting "total
applications of A" on the diagonal — mixes counting units and gives
inflated diagonals; we leave that to the simpler frequency matrix where
it belongs.

Conventions match :mod:`scribe.projects` (F1.1), :mod:`scribe.sources`
(F1.2), :mod:`scribe.participants` (F1.3), :mod:`scribe.codes` (F2.1),
:mod:`scribe.source_schema` (F3.2), :mod:`scribe.speaker_map` (F3.4),
and :mod:`scribe.query` (F3.5).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable

from .codes import CODE_ID_RE, Code
from .participants import Participant
from .projects import ProjectValidationError
from .query import (
    PROXIMITY_SCOPES,
    QueryValidationError,
    _app_get,
    _app_required_field,
    _coerce_optional_float,
)
from .sources import SOURCE_ID_RE, Source
from .speaker_map import SpeakerMap


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# How an attribute value should be looked up for the cross-tab matrix.
ATTRIBUTE_KINDS: tuple[str, ...] = ("source", "participant")

# Sentinel column key used when an application has no value for the
# requested attribute (and the caller has asked us to keep those
# applications rather than drop them).
MISSING_ATTRIBUTE_COL_KEY = "__missing__"

# Default display label for the missing-attribute column.
DEFAULT_MISSING_ATTRIBUTE_LABEL = "(missing)"

# How big can a single matrix be before we balk? Generous, but bounded
# so a stray query can't OOM the server.
MAX_ROWS = 4096
MAX_COLS = 4096


# --------------------------------------------------------------------------- #
# Public error type
# --------------------------------------------------------------------------- #


class MatrixError(ProjectValidationError):
    """Raised when matrix inputs are malformed.

    Subclass of :class:`scribe.projects.ProjectValidationError` so the
    same error-handling layer used everywhere else in the F-series
    catches matrix problems uniformly.
    """


# --------------------------------------------------------------------------- #
# Matrix dataclass
# --------------------------------------------------------------------------- #


@dataclass
class Matrix:
    """A two-dimensional grid of integer cells.

    Rows and columns are addressed by string keys (typically code ids,
    source ids, or attribute values). The order of :attr:`rows` and
    :attr:`cols` is meaningful — the UI / CSV export render them in that
    order. ``row_titles`` / ``col_titles`` carry the human-readable
    display strings (code name, source name, attribute value).

    ``cells`` only stores non-zero entries; missing keys are zero by
    construction. This keeps the on-disk JSON and the in-memory shape
    compact for sparse corpora (a typical research project has hundreds
    of codes but each application touches only one).
    """

    title: str = ""
    row_label: str = "row"
    col_label: str = "col"
    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    cells: dict[tuple[str, str], int] = field(default_factory=dict)
    row_titles: dict[str, str] = field(default_factory=dict)
    col_titles: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Cell access
    # ------------------------------------------------------------------ #

    def get(self, row: str, col: str) -> int:
        """Return the integer cell at (row, col); 0 if missing."""
        return int(self.cells.get((row, col), 0))

    def set(self, row: str, col: str, value: int) -> None:
        """Set a cell. Zero values are deleted to keep ``cells`` sparse."""
        v = int(value)
        if v == 0:
            self.cells.pop((row, col), None)
        else:
            self.cells[(row, col)] = v

    def increment(self, row: str, col: str, by: int = 1) -> None:
        """Add ``by`` to the cell at (row, col)."""
        self.set(row, col, self.get(row, col) + int(by))

    # ------------------------------------------------------------------ #
    # Totals
    # ------------------------------------------------------------------ #

    def row_total(self, row: str) -> int:
        return sum(self.get(row, c) for c in self.cols)

    def col_total(self, col: str) -> int:
        return sum(self.get(r, col) for r in self.rows)

    def grand_total(self) -> int:
        return sum(int(v) for v in self.cells.values())

    # ------------------------------------------------------------------ #
    # Compaction
    # ------------------------------------------------------------------ #

    def compact(
        self,
        *,
        drop_empty_rows: bool = True,
        drop_empty_cols: bool = True,
    ) -> "Matrix":
        """Return a copy with all-zero rows/cols removed.

        Display titles are preserved for the rows/cols that survive.
        """
        new_rows = (
            [r for r in self.rows if self.row_total(r) != 0]
            if drop_empty_rows
            else list(self.rows)
        )
        new_cols = (
            [c for c in self.cols if self.col_total(c) != 0]
            if drop_empty_cols
            else list(self.cols)
        )
        new_row_set = set(new_rows)
        new_col_set = set(new_cols)
        new_cells = {
            (r, c): v
            for (r, c), v in self.cells.items()
            if r in new_row_set and c in new_col_set
        }
        return Matrix(
            title=self.title,
            row_label=self.row_label,
            col_label=self.col_label,
            rows=new_rows,
            cols=new_cols,
            cells=new_cells,
            row_titles={
                k: v for k, v in self.row_titles.items() if k in new_row_set
            },
            col_titles={
                k: v for k, v in self.col_titles.items() if k in new_col_set
            },
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "row_label": self.row_label,
            "col_label": self.col_label,
            "rows": list(self.rows),
            "cols": list(self.cols),
            # Cells go out as a list of triples so the JSON is portable
            # (tuple keys are unrepresentable in JSON object keys).
            "cells": [
                [r, c, int(v)] for (r, c), v in self.cells.items() if int(v) != 0
            ],
            "row_titles": dict(self.row_titles),
            "col_titles": dict(self.col_titles),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Matrix":
        if not isinstance(d, Mapping):
            raise MatrixError("Matrix payload must be an object")
        rows = [str(x) for x in (d.get("rows") or [])]
        cols = [str(x) for x in (d.get("cols") or [])]
        raw_cells = d.get("cells") or []
        cells: dict[tuple[str, str], int] = {}
        if not isinstance(raw_cells, list):
            raise MatrixError("Matrix.cells must be a list of [row, col, value]")
        for triple in raw_cells:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise MatrixError(
                    "Matrix.cells entry must be a 3-tuple [row, col, value]"
                )
            r, c, v = triple
            cells[(str(r), str(c))] = int(v)
        return cls(
            title=str(d.get("title", "") or ""),
            row_label=str(d.get("row_label", "row") or "row"),
            col_label=str(d.get("col_label", "col") or "col"),
            rows=rows,
            cols=cols,
            cells=cells,
            row_titles={
                str(k): str(v) for k, v in (d.get("row_titles") or {}).items()
            },
            col_titles={
                str(k): str(v) for k, v in (d.get("col_titles") or {}).items()
            },
        )

    # ------------------------------------------------------------------ #
    # CSV export
    # ------------------------------------------------------------------ #

    def to_csv(
        self,
        *,
        use_titles: bool = True,
        include_totals: bool = True,
    ) -> str:
        """Render the matrix as CSV.

        Top-left corner is the matrix title (or the row_label if no
        title is set). Column headers use ``col_titles`` if available
        and ``use_titles=True``; otherwise the raw column key is shown.
        ``include_totals=True`` adds a "Total" column on the right and
        a "Total" row at the bottom.
        """
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")

        # Header row: corner cell + column labels (+ Total).
        corner = self.title or self.row_label
        header: list[str] = [corner]
        for c in self.cols:
            if use_titles and c in self.col_titles and self.col_titles[c]:
                header.append(self.col_titles[c])
            else:
                header.append(c)
        if include_totals:
            header.append("Total")
        w.writerow(header)

        # Body rows.
        for r in self.rows:
            label = (
                self.row_titles[r]
                if use_titles and r in self.row_titles and self.row_titles[r]
                else r
            )
            row_out: list[Any] = [label]
            for c in self.cols:
                row_out.append(self.get(r, c))
            if include_totals:
                row_out.append(self.row_total(r))
            w.writerow(row_out)

        # Footer total row.
        if include_totals:
            footer: list[Any] = ["Total"]
            for c in self.cols:
                footer.append(self.col_total(c))
            footer.append(self.grand_total())
            w.writerow(footer)

        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if len(self.rows) > MAX_ROWS:
            raise MatrixError(f"Matrix has too many rows (>{MAX_ROWS})")
        if len(self.cols) > MAX_COLS:
            raise MatrixError(f"Matrix has too many cols (>{MAX_COLS})")
        # rows / cols must be unique within their axis (a duplicate would
        # give two columns for the same key and ambiguous lookups).
        if len(set(self.rows)) != len(self.rows):
            raise MatrixError("Matrix.rows contains duplicate keys")
        if len(set(self.cols)) != len(self.cols):
            raise MatrixError("Matrix.cols contains duplicate keys")
        # Every cell must reference a declared row/col.
        row_set = set(self.rows)
        col_set = set(self.cols)
        for r, c in self.cells.keys():
            if r not in row_set:
                raise MatrixError(
                    f"Matrix.cells references unknown row {r!r}"
                )
            if c not in col_set:
                raise MatrixError(
                    f"Matrix.cells references unknown col {c!r}"
                )


# --------------------------------------------------------------------------- #
# Helpers shared across builders
# --------------------------------------------------------------------------- #


def _validate_application_ids(app: Any) -> tuple[str, str]:
    """Read code_id / source_id off an application; raise on bad shape.

    Wraps the shared F3.5 helpers' :class:`QueryValidationError` as
    :class:`MatrixError` so callers see a single error type from this
    module.
    """
    try:
        sid = _app_required_field(app, "source_id")
        cid = _app_required_field(app, "code_id")
    except QueryValidationError as e:
        raise MatrixError(str(e)) from e
    if not SOURCE_ID_RE.match(sid):
        raise MatrixError(
            f"application source_id must be 12-char hex; got {sid!r}"
        )
    if not CODE_ID_RE.match(cid):
        raise MatrixError(
            f"application code_id must be 12-char hex; got {cid!r}"
        )
    return cid, sid


def _ensure_unique(seq: Iterable[str], *, what: str) -> list[str]:
    """Return list(seq) after rejecting duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        sx = str(x)
        if sx in seen:
            raise MatrixError(f"duplicate {what} key: {sx!r}")
        seen.add(sx)
        out.append(sx)
    return out


# --------------------------------------------------------------------------- #
# code × source frequency
# --------------------------------------------------------------------------- #


def code_by_source_matrix(
    *,
    applications: Iterable[Any],
    codes: Sequence[Code],
    sources: Sequence[Source],
    title: str = "Code × Source",
) -> Matrix:
    """Build a code × source frequency matrix.

    Cells count *applications* — one application contributes one to its
    (code_id, source_id) cell. Applications whose code or source is not
    in the supplied lists are silently dropped (this is the typical
    "report on the codebook I currently have, not on stale orphans"
    semantics; callers wanting strict mode should pre-validate).

    Rows preserve the order of ``codes``; cols preserve the order of
    ``sources``. Use :meth:`Matrix.compact` to drop empty rows/cols
    after construction.
    """
    if len(codes) > MAX_ROWS:
        raise MatrixError(f"too many codes (>{MAX_ROWS})")
    if len(sources) > MAX_COLS:
        raise MatrixError(f"too many sources (>{MAX_COLS})")

    code_ids = _ensure_unique((c.id for c in codes), what="code id")
    source_ids = _ensure_unique((s.id for s in sources), what="source id")
    code_id_set = set(code_ids)
    source_id_set = set(source_ids)

    m = Matrix(
        title=title,
        row_label="Code",
        col_label="Source",
        rows=code_ids,
        cols=source_ids,
        row_titles={c.id: c.name for c in codes},
        col_titles={s.id: s.name for s in sources},
    )

    for app in applications:
        cid, sid = _validate_application_ids(app)
        if cid not in code_id_set or sid not in source_id_set:
            continue
        m.increment(cid, sid)

    m.validate()
    return m


# --------------------------------------------------------------------------- #
# code × code co-occurrence
# --------------------------------------------------------------------------- #


def code_by_code_matrix(
    *,
    applications: Iterable[Any],
    codes: Sequence[Code],
    scope: str = "source",
    max_gap: float = 0.0,
    title: str = "Code × Code (co-occurrence)",
) -> Matrix:
    """Build a code × code co-occurrence matrix.

    The matrix is symmetric. ``cell(A, B)`` is the number of unordered
    pairs of distinct applications ``{a, b}`` with ``a.code == A``,
    ``b.code == B``, where ``a`` and ``b`` co-occur in scope:

      * ``scope="source"`` — ``a.source_id == b.source_id``.
      * ``scope="segment"`` — same source AND anchor ranges overlap
        (closed intervals, requires numeric ``start`` / ``end``).
      * ``scope="paragraph"`` — same source AND distance between anchor
        ranges ≤ ``max_gap``. Requires numeric anchors.

    Applications missing the data needed for the chosen scope (no
    numeric anchors for ``segment`` / ``paragraph``) are dropped from
    the count. Applications with code ids outside ``codes`` are dropped
    silently (orphan codes don't pollute the matrix).
    """
    if scope not in PROXIMITY_SCOPES:
        raise MatrixError(
            f"scope must be one of {PROXIMITY_SCOPES}; got {scope!r}"
        )
    if max_gap < 0:
        raise MatrixError("max_gap must be ≥ 0")

    if len(codes) > MAX_ROWS:
        raise MatrixError(f"too many codes (>{MAX_ROWS})")

    code_ids = _ensure_unique((c.id for c in codes), what="code id")
    code_id_set = set(code_ids)

    # Bucket eligible applications by source so the inner loop is O(n²)
    # within a source rather than O(n²) over the whole corpus.
    by_source: dict[str, list[tuple[str, float | None, float | None]]] = {}
    for app in applications:
        cid, sid = _validate_application_ids(app)
        if cid not in code_id_set:
            continue
        start = _coerce_optional_float(_app_get(app, "start"))
        end = _coerce_optional_float(_app_get(app, "end"))
        # For segment / paragraph we need numeric anchors; drop apps
        # that can't be placed.
        if scope in ("segment", "paragraph"):
            if start is None or end is None:
                continue
            # A start > end anchor is malformed; swap rather than reject.
            if start > end:
                start, end = end, start
        by_source.setdefault(sid, []).append((cid, start, end))

    m = Matrix(
        title=title,
        row_label="Code",
        col_label="Code",
        rows=code_ids,
        cols=code_ids,
        row_titles={c.id: c.name for c in codes},
        col_titles={c.id: c.name for c in codes},
    )

    for apps_in_source in by_source.values():
        n = len(apps_in_source)
        # All unordered pairs of distinct applications in this source.
        for i in range(n):
            cid_i, s_i, e_i = apps_in_source[i]
            for j in range(i + 1, n):
                cid_j, s_j, e_j = apps_in_source[j]
                if scope == "source":
                    pass  # same source is enough
                elif scope == "segment":
                    # Closed-interval overlap (s_j ≤ e_i and s_i ≤ e_j).
                    # All four anchors are guaranteed numeric here by the
                    # pre-bucketing pass.
                    assert s_i is not None and e_i is not None
                    assert s_j is not None and e_j is not None
                    if not (s_j <= e_i and s_i <= e_j):
                        continue
                elif scope == "paragraph":
                    assert s_i is not None and e_i is not None
                    assert s_j is not None and e_j is not None
                    if s_j <= e_i and s_i <= e_j:
                        gap = 0.0
                    elif e_j < s_i:
                        gap = s_i - e_j
                    else:
                        gap = s_j - e_i
                    if gap > max_gap:
                        continue
                # Increment both directions for symmetry.
                m.increment(cid_i, cid_j)
                if cid_i != cid_j:
                    m.increment(cid_j, cid_i)

    m.validate()
    return m


# --------------------------------------------------------------------------- #
# code × attribute cross-tab
# --------------------------------------------------------------------------- #


def _resolve_source_attribute(
    app: Any,
    sources_by_id: Mapping[str, Source],
    attribute_key: str,
) -> str | None:
    sid = str(_app_get(app, "source_id", "") or "")
    s = sources_by_id.get(sid)
    if s is None:
        return None
    val = s.custom_attributes.get(attribute_key, "")
    if val == "" or val is None:
        return None
    return str(val)


def _resolve_participant_attribute(
    app: Any,
    participants_by_id: Mapping[str, Participant],
    speaker_maps: Mapping[str, SpeakerMap],
    attribute_key: str,
) -> str | None:
    sid = str(_app_get(app, "source_id", "") or "")
    pid: str | None = None
    label = str(_app_get(app, "speaker", "") or "")
    if label and sid in speaker_maps:
        pid = speaker_maps[sid].participant_for(label)
    if pid is None:
        explicit = _app_get(app, "participant_id")
        if explicit:
            pid = str(explicit)
    if pid is None:
        return None
    p = participants_by_id.get(pid)
    if p is None:
        return None
    val = p.demographics.get(attribute_key, "")
    if val == "" or val is None:
        return None
    return str(val)


def code_by_attribute_matrix(
    *,
    applications: Iterable[Any],
    codes: Sequence[Code],
    attribute_key: str,
    attribute_kind: str = "source",
    sources: Sequence[Source] | None = None,
    participants: Sequence[Participant] | None = None,
    speaker_maps: Mapping[str, SpeakerMap] | None = None,
    include_missing: bool = True,
    missing_label: str = DEFAULT_MISSING_ATTRIBUTE_LABEL,
    title: str | None = None,
) -> Matrix:
    """Build a code × attribute-value cross-tab.

    ``attribute_kind`` selects the lookup path:

      * ``"source"`` — read ``source.custom_attributes[attribute_key]``
        for the application's source. Pass ``sources`` (required).
      * ``"participant"`` — resolve the application's speaker label to
        a participant (via ``speaker_maps[source_id].participant_for``),
        then read ``participant.demographics[attribute_key]``. Falls
        back to an explicit ``participant_id`` field on the
        application. Pass ``participants`` (required) and
        ``speaker_maps`` (optional but typical).

    Columns are the unique non-empty attribute values, in lexicographic
    order. When ``include_missing=True`` (the default), applications
    whose attribute resolves to empty or missing fall into a final
    ``__missing__`` column whose display label is ``missing_label``.

    ``attribute_key`` itself is not validated against any project schema
    here — this module is below F3.2's schema layer. Callers that have a
    schema in hand should validate the key before calling.
    """
    if attribute_kind not in ATTRIBUTE_KINDS:
        raise MatrixError(
            f"attribute_kind must be one of {ATTRIBUTE_KINDS}; "
            f"got {attribute_kind!r}"
        )
    key = str(attribute_key or "").strip()
    if not key:
        raise MatrixError("attribute_key is required")

    if attribute_kind == "source":
        if sources is None:
            raise MatrixError(
                "code_by_attribute_matrix: sources is required for "
                "attribute_kind='source'"
            )
        sources_by_id = {s.id: s for s in sources}
    else:
        if participants is None:
            raise MatrixError(
                "code_by_attribute_matrix: participants is required for "
                "attribute_kind='participant'"
            )
        participants_by_id = {p.id: p for p in participants}
        smap_lookup: dict[str, SpeakerMap] = (
            dict(speaker_maps) if speaker_maps else {}
        )

    if len(codes) > MAX_ROWS:
        raise MatrixError(f"too many codes (>{MAX_ROWS})")

    code_ids = _ensure_unique((c.id for c in codes), what="code id")
    code_id_set = set(code_ids)
    code_titles = {c.id: c.name for c in codes}

    # First pass: count per (code_id, attribute_value); also gather the
    # ordered unique attribute values seen (sorted lexicographically for
    # stable output). Missing values bucket into MISSING_ATTRIBUTE_COL_KEY.
    counts: dict[tuple[str, str], int] = {}
    seen_values: set[str] = set()
    saw_missing = False

    for app in applications:
        cid, sid = _validate_application_ids(app)
        if cid not in code_id_set:
            continue

        if attribute_kind == "source":
            val = _resolve_source_attribute(app, sources_by_id, key)
        else:
            val = _resolve_participant_attribute(
                app, participants_by_id, smap_lookup, key
            )

        if val is None:
            if not include_missing:
                continue
            col_key = MISSING_ATTRIBUTE_COL_KEY
            saw_missing = True
        else:
            col_key = val
            seen_values.add(val)

        counts[(cid, col_key)] = counts.get((cid, col_key), 0) + 1

    sorted_values = sorted(seen_values)
    cols: list[str] = list(sorted_values)
    col_titles: dict[str, str] = {v: v for v in sorted_values}
    if include_missing and saw_missing:
        cols.append(MISSING_ATTRIBUTE_COL_KEY)
        col_titles[MISSING_ATTRIBUTE_COL_KEY] = missing_label

    if len(cols) > MAX_COLS:
        raise MatrixError(f"too many distinct attribute values (>{MAX_COLS})")

    if title is None:
        kind_label = "Source" if attribute_kind == "source" else "Participant"
        title = f"Code × {kind_label} attribute: {key}"

    m = Matrix(
        title=title,
        row_label="Code",
        col_label=key,
        rows=code_ids,
        cols=cols,
        row_titles=code_titles,
        col_titles=col_titles,
        cells=dict(counts),
    )
    m.validate()
    return m
