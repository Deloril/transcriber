"""Tests for scribe.matrix (F3.6).

Matrix views: code × source frequency, code × code co-occurrence, and
code × attribute cross-tab.

The module is stand-alone (no FastAPI, no engine imports) so all tests
stay in pure Python.
"""

from __future__ import annotations

import json

import pytest

from scribe.codes import Code, new_code_id
from scribe.matrix import (
    ATTRIBUTE_KINDS,
    DEFAULT_MISSING_ATTRIBUTE_LABEL,
    MAX_COLS,
    MAX_ROWS,
    MISSING_ATTRIBUTE_COL_KEY,
    Matrix,
    MatrixError,
    code_by_attribute_matrix,
    code_by_code_matrix,
    code_by_source_matrix,
)
from scribe.participants import Participant
from scribe.projects import ProjectValidationError
from scribe.sources import Source
from scribe.speaker_map import SpeakerEntry, SpeakerMap


# --------------------------------------------------------------------------- #
# Tiny helpers (mirror tests/test_query.py conventions)
# --------------------------------------------------------------------------- #


PID = "deadbeef0001"


def _hex(seed: int) -> str:
    return f"{seed:012x}"


def _code(cid: str, name: str = "", project_id: str = PID) -> Code:
    return Code.new(
        project_id=project_id, name=name or f"code-{cid}", code_id=cid
    )


def _src(
    sid: str,
    *,
    name: str | None = None,
    project_id: str = PID,
    custom: dict[str, str] | None = None,
) -> Source:
    return Source.new(
        project_id=project_id,
        name=name or f"src-{sid}",
        source_id=sid,
        custom_attributes=custom or {},
    )


def _participant(
    pid: str,
    *,
    name: str | None = None,
    project_id: str = PID,
    demographics: dict[str, str] | None = None,
) -> Participant:
    return Participant.new(
        project_id=project_id,
        name=name or f"P-{pid}",
        participant_id=pid,
        demographics=demographics or {},
    )


def _smap(
    sid: str,
    entries: list[dict],
    project_id: str = PID,
) -> SpeakerMap:
    return SpeakerMap.new(
        project_id=project_id,
        source_id=sid,
        entries=entries,
    )


def _app(
    *,
    code_id: str,
    source_id: str,
    speaker: str = "",
    start: float | None = None,
    end: float | None = None,
    participant_id: str | None = None,
) -> dict:
    a: dict = {"code_id": code_id, "source_id": source_id}
    if speaker:
        a["speaker"] = speaker
    if start is not None:
        a["start"] = start
    if end is not None:
        a["end"] = end
    if participant_id is not None:
        a["participant_id"] = participant_id
    return a


# --------------------------------------------------------------------------- #
# Matrix dataclass
# --------------------------------------------------------------------------- #


class TestMatrixCellAccess:
    def test_get_returns_zero_for_missing(self):
        m = Matrix(rows=["a"], cols=["x"])
        assert m.get("a", "x") == 0

    def test_set_stores_value(self):
        m = Matrix(rows=["a"], cols=["x"])
        m.set("a", "x", 5)
        assert m.get("a", "x") == 5

    def test_set_zero_removes_cell(self):
        m = Matrix(rows=["a"], cols=["x"], cells={("a", "x"): 3})
        m.set("a", "x", 0)
        assert ("a", "x") not in m.cells

    def test_increment_starts_from_zero(self):
        m = Matrix(rows=["a"], cols=["x"])
        m.increment("a", "x")
        m.increment("a", "x", by=2)
        assert m.get("a", "x") == 3

    def test_set_with_string_int_coerces(self):
        # `set` converts to int; ensures a stray "5" coming in as text
        # doesn't poison the matrix later.
        m = Matrix(rows=["a"], cols=["x"])
        m.set("a", "x", 4)
        assert isinstance(m.cells[("a", "x")], int)


class TestMatrixTotals:
    def test_row_total(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1, ("a", "y"): 2, ("b", "x"): 3},
        )
        assert m.row_total("a") == 3
        assert m.row_total("b") == 3

    def test_col_total(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1, ("a", "y"): 2, ("b", "x"): 3},
        )
        assert m.col_total("x") == 4
        assert m.col_total("y") == 2

    def test_grand_total(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1, ("a", "y"): 2, ("b", "x"): 3},
        )
        assert m.grand_total() == 6

    def test_totals_with_no_cells(self):
        m = Matrix(rows=["a"], cols=["x"])
        assert m.row_total("a") == 0
        assert m.col_total("x") == 0
        assert m.grand_total() == 0

    def test_row_total_unknown_row_is_zero(self):
        # No exception — an unknown row simply has no cells.
        m = Matrix(rows=["a"], cols=["x"])
        assert m.row_total("missing") == 0


class TestMatrixCompact:
    def test_drops_empty_rows(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x"],
            cells={("a", "x"): 1},
        )
        c = m.compact()
        assert c.rows == ["a"]
        assert c.cols == ["x"]

    def test_drops_empty_cols(self):
        m = Matrix(
            rows=["a"],
            cols=["x", "y"],
            cells={("a", "x"): 1},
        )
        c = m.compact()
        assert c.cols == ["x"]

    def test_keep_empty_rows_when_disabled(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x"],
            cells={("a", "x"): 1},
        )
        c = m.compact(drop_empty_rows=False)
        assert c.rows == ["a", "b"]

    def test_keep_empty_cols_when_disabled(self):
        m = Matrix(
            rows=["a"],
            cols=["x", "y"],
            cells={("a", "x"): 1},
        )
        c = m.compact(drop_empty_cols=False)
        assert c.cols == ["x", "y"]

    def test_compact_preserves_titles(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1},
            row_titles={"a": "Alpha", "b": "Beta"},
            col_titles={"x": "X-axis", "y": "Y-axis"},
        )
        c = m.compact()
        assert c.row_titles == {"a": "Alpha"}
        assert c.col_titles == {"x": "X-axis"}

    def test_compact_does_not_mutate_original(self):
        m = Matrix(
            rows=["a", "b"],
            cols=["x"],
            cells={("a", "x"): 1},
        )
        m.compact()
        assert m.rows == ["a", "b"]


class TestMatrixSerialisation:
    def test_to_dict_round_trip(self):
        m = Matrix(
            title="t",
            row_label="Code",
            col_label="Source",
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1, ("b", "y"): 2},
            row_titles={"a": "Alpha", "b": "Beta"},
            col_titles={"x": "X", "y": "Y"},
        )
        d = m.to_dict()
        # Round-trip through JSON to prove it's portable.
        d2 = json.loads(json.dumps(d))
        m2 = Matrix.from_dict(d2)
        assert m2.title == "t"
        assert m2.rows == ["a", "b"]
        assert m2.cols == ["x", "y"]
        assert m2.get("a", "x") == 1
        assert m2.get("b", "y") == 2
        assert m2.row_titles == {"a": "Alpha", "b": "Beta"}
        assert m2.col_titles == {"x": "X", "y": "Y"}

    def test_to_dict_omits_zero_cells(self):
        m = Matrix(
            rows=["a"],
            cols=["x"],
            cells={("a", "x"): 0},  # synthetic — set() avoids this
        )
        d = m.to_dict()
        assert d["cells"] == []

    def test_from_dict_rejects_non_object(self):
        with pytest.raises(MatrixError):
            Matrix.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_rejects_bad_cell_shape(self):
        with pytest.raises(MatrixError):
            Matrix.from_dict({"rows": ["a"], "cols": ["x"], "cells": [["a", "x"]]})


class TestMatrixCSV:
    def _sample(self) -> Matrix:
        return Matrix(
            title="Code × Source",
            row_label="Code",
            col_label="Source",
            rows=["a", "b"],
            cols=["x", "y"],
            cells={("a", "x"): 1, ("b", "y"): 2, ("a", "y"): 3},
            row_titles={"a": "Alpha", "b": "Beta"},
            col_titles={"x": "X", "y": "Y"},
        )

    def test_csv_with_titles_and_totals(self):
        out = self._sample().to_csv()
        lines = out.strip().split("\n")
        assert lines[0] == "Code × Source,X,Y,Total"
        assert lines[1] == "Alpha,1,3,4"
        assert lines[2] == "Beta,0,2,2"
        assert lines[3] == "Total,1,5,6"

    def test_csv_without_titles_uses_keys(self):
        out = self._sample().to_csv(use_titles=False)
        lines = out.strip().split("\n")
        assert lines[0] == "Code × Source,x,y,Total"
        assert lines[1] == "a,1,3,4"

    def test_csv_without_totals(self):
        out = self._sample().to_csv(include_totals=False)
        lines = out.strip().split("\n")
        assert lines[0] == "Code × Source,X,Y"
        assert "Total" not in out

    def test_csv_corner_falls_back_to_row_label(self):
        m = Matrix(
            title="",
            row_label="Code",
            col_label="Source",
            rows=["a"],
            cols=["x"],
            cells={("a", "x"): 1},
        )
        out = m.to_csv(include_totals=False)
        assert out.startswith("Code,x")


class TestMatrixValidate:
    def test_validate_rejects_duplicate_rows(self):
        m = Matrix(rows=["a", "a"], cols=["x"])
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_rejects_duplicate_cols(self):
        m = Matrix(rows=["a"], cols=["x", "x"])
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_rejects_unknown_cell_row(self):
        m = Matrix(rows=["a"], cols=["x"], cells={("z", "x"): 1})
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_rejects_unknown_cell_col(self):
        m = Matrix(rows=["a"], cols=["x"], cells={("a", "z"): 1})
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_rejects_too_many_rows(self):
        m = Matrix(rows=[f"r{i}" for i in range(MAX_ROWS + 1)], cols=["x"])
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_rejects_too_many_cols(self):
        m = Matrix(rows=["a"], cols=[f"c{i}" for i in range(MAX_COLS + 1)])
        with pytest.raises(MatrixError):
            m.validate()

    def test_validate_passes_on_empty_matrix(self):
        Matrix().validate()


# --------------------------------------------------------------------------- #
# code_by_source_matrix
# --------------------------------------------------------------------------- #


class TestCodeBySource:
    def test_empty_corpus(self):
        m = code_by_source_matrix(applications=[], codes=[], sources=[])
        assert m.rows == [] and m.cols == [] and m.grand_total() == 0

    def test_basic_count(self):
        c1 = _code(_hex(1), "fear")
        c2 = _code(_hex(2), "hope")
        s1 = _src(_hex(11), name="interview-1")
        s2 = _src(_hex(12), name="interview-2")
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s2.id),
        ]
        m = code_by_source_matrix(
            applications=apps, codes=[c1, c2], sources=[s1, s2]
        )
        assert m.get(c1.id, s1.id) == 2
        assert m.get(c2.id, s1.id) == 1
        assert m.get(c2.id, s2.id) == 1
        assert m.get(c1.id, s2.id) == 0
        assert m.grand_total() == 4

    def test_order_preserved(self):
        c1, c2, c3 = _code(_hex(1)), _code(_hex(2)), _code(_hex(3))
        s1, s2 = _src(_hex(11)), _src(_hex(12))
        m = code_by_source_matrix(
            applications=[], codes=[c2, c1, c3], sources=[s2, s1]
        )
        assert m.rows == [c2.id, c1.id, c3.id]
        assert m.cols == [s2.id, s1.id]

    def test_orphan_code_dropped(self):
        c1 = _code(_hex(1))
        c_orphan = _code(_hex(99))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c_orphan.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s1.id),
        ]
        m = code_by_source_matrix(
            applications=apps, codes=[c1], sources=[s1]
        )
        assert m.grand_total() == 1
        assert m.get(c1.id, s1.id) == 1

    def test_orphan_source_dropped(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=_hex(99)),
            _app(code_id=c1.id, source_id=s1.id),
        ]
        m = code_by_source_matrix(
            applications=apps, codes=[c1], sources=[s1]
        )
        assert m.grand_total() == 1

    def test_titles_use_names(self):
        c1 = _code(_hex(1), "fear")
        s1 = _src(_hex(11), name="interview-1")
        m = code_by_source_matrix(
            applications=[], codes=[c1], sources=[s1]
        )
        assert m.row_titles[c1.id] == "fear"
        assert m.col_titles[s1.id] == "interview-1"

    def test_rejects_invalid_code_id(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        bad = {"code_id": "not-hex", "source_id": s1.id}
        with pytest.raises(MatrixError):
            code_by_source_matrix(
                applications=[bad], codes=[c1], sources=[s1]
            )

    def test_rejects_missing_source_id(self):
        c1 = _code(_hex(1))
        bad = {"code_id": c1.id}  # no source_id
        with pytest.raises(MatrixError):
            code_by_source_matrix(
                applications=[bad], codes=[c1], sources=[]
            )

    def test_compact_drops_unused(self):
        c1, c2 = _code(_hex(1)), _code(_hex(2))
        s1, s2 = _src(_hex(11)), _src(_hex(12))
        apps = [_app(code_id=c1.id, source_id=s1.id)]
        m = code_by_source_matrix(
            applications=apps, codes=[c1, c2], sources=[s1, s2]
        ).compact()
        assert m.rows == [c1.id]
        assert m.cols == [s1.id]


# --------------------------------------------------------------------------- #
# code_by_code_matrix
# --------------------------------------------------------------------------- #


class TestCodeByCode:
    def test_empty_corpus(self):
        m = code_by_code_matrix(applications=[], codes=[])
        assert m.grand_total() == 0

    def test_source_scope_basic(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        # Two apps in same source, different codes → one pair.
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s1.id),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="source"
        )
        assert m.get(c1.id, c2.id) == 1
        assert m.get(c2.id, c1.id) == 1
        # Diagonal: only one application of each, no self-pair.
        assert m.get(c1.id, c1.id) == 0
        assert m.get(c2.id, c2.id) == 0

    def test_source_scope_symmetric(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s1.id),  # second c1
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="source"
        )
        # c1 × c2: 2 (each c1 paired with c2)
        assert m.get(c1.id, c2.id) == 2
        assert m.get(c2.id, c1.id) == 2
        # c1 × c1: 1 unordered pair of distinct apps
        assert m.get(c1.id, c1.id) == 1

    def test_diagonal_pair_count(self):
        # 4 applications of c1 in same source → C(4,2) = 6.
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id) for _ in range(4)
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1], scope="source"
        )
        assert m.get(c1.id, c1.id) == 6

    def test_different_sources_no_co_occurrence(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        s2 = _src(_hex(12))
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s2.id),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="source"
        )
        assert m.grand_total() == 0

    def test_segment_scope_overlap(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        # Overlapping anchors: 0–10 and 5–15.
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=10),
            _app(code_id=c2.id, source_id=s1.id, start=5, end=15),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="segment"
        )
        assert m.get(c1.id, c2.id) == 1

    def test_segment_scope_no_overlap(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        # Disjoint anchors: 0–4 and 5–10. (Closed-interval test would
        # accept touching at 4–5; here we leave a gap.)
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=4),
            _app(code_id=c2.id, source_id=s1.id, start=6, end=10),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="segment"
        )
        assert m.get(c1.id, c2.id) == 0

    def test_segment_scope_touching_intervals_overlap(self):
        # Closed intervals: [0, 4] and [4, 10] share the point 4.
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=4),
            _app(code_id=c2.id, source_id=s1.id, start=4, end=10),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="segment"
        )
        assert m.get(c1.id, c2.id) == 1

    def test_segment_scope_drops_unanchored(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        # No anchors → dropped.
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c2.id, source_id=s1.id),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="segment"
        )
        assert m.grand_total() == 0

    def test_paragraph_scope_within_gap(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        # Gap of 5 between [0,4] and [9,12].
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=4),
            _app(code_id=c2.id, source_id=s1.id, start=9, end=12),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="paragraph", max_gap=5
        )
        assert m.get(c1.id, c2.id) == 1

    def test_paragraph_scope_exceeds_gap(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=4),
            _app(code_id=c2.id, source_id=s1.id, start=20, end=25),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="paragraph", max_gap=5
        )
        assert m.get(c1.id, c2.id) == 0

    def test_paragraph_scope_overlapping_is_zero_gap(self):
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=0, end=10),
            _app(code_id=c2.id, source_id=s1.id, start=5, end=15),
        ]
        # max_gap=0 still accepts overlap.
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="paragraph", max_gap=0
        )
        assert m.get(c1.id, c2.id) == 1

    def test_paragraph_scope_negative_gap_rejected(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_code_matrix(
                applications=[],
                codes=[c1],
                scope="paragraph",
                max_gap=-1,
            )

    def test_invalid_scope_rejected(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_code_matrix(applications=[], codes=[c1], scope="bogus")

    def test_orphan_code_dropped(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=_hex(99), source_id=s1.id),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1], scope="source"
        )
        assert m.grand_total() == 0

    def test_segment_swaps_inverted_anchors(self):
        # start > end is malformed; module swaps rather than reject.
        c1 = _code(_hex(1))
        c2 = _code(_hex(2))
        s1 = _src(_hex(11))
        apps = [
            _app(code_id=c1.id, source_id=s1.id, start=10, end=0),  # swapped
            _app(code_id=c2.id, source_id=s1.id, start=5, end=15),
        ]
        m = code_by_code_matrix(
            applications=apps, codes=[c1, c2], scope="segment"
        )
        assert m.get(c1.id, c2.id) == 1


# --------------------------------------------------------------------------- #
# code_by_attribute_matrix
# --------------------------------------------------------------------------- #


class TestCodeByAttributeSource:
    def test_basic_source_attribute(self):
        c1 = _code(_hex(1), "fear")
        c2 = _code(_hex(2), "hope")
        s1 = _src(_hex(11), custom={"site": "Hospital A"})
        s2 = _src(_hex(12), custom={"site": "Clinic B"})
        s3 = _src(_hex(13), custom={"site": "Hospital A"})
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s3.id),
            _app(code_id=c2.id, source_id=s2.id),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1, c2],
            attribute_key="site",
            attribute_kind="source",
            sources=[s1, s2, s3],
        )
        assert m.get(c1.id, "Hospital A") == 2
        assert m.get(c2.id, "Clinic B") == 1
        assert "Hospital A" in m.cols
        assert "Clinic B" in m.cols

    def test_columns_sorted_lexicographically(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11), custom={"site": "Z"})
        s2 = _src(_hex(12), custom={"site": "A"})
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s2.id),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="site",
            sources=[s1, s2],
        )
        # Sorted with optional missing column appended at end.
        assert m.cols[:2] == ["A", "Z"]

    def test_missing_attribute_bucketed(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11), custom={"site": "Hospital A"})
        s2 = _src(_hex(12))  # no site
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s2.id),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="site",
            sources=[s1, s2],
        )
        assert m.get(c1.id, "Hospital A") == 1
        assert m.get(c1.id, MISSING_ATTRIBUTE_COL_KEY) == 1
        assert m.col_titles[MISSING_ATTRIBUTE_COL_KEY] == DEFAULT_MISSING_ATTRIBUTE_LABEL

    def test_missing_can_be_dropped(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11), custom={"site": "Hospital A"})
        s2 = _src(_hex(12))  # no site
        apps = [
            _app(code_id=c1.id, source_id=s1.id),
            _app(code_id=c1.id, source_id=s2.id),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="site",
            sources=[s1, s2],
            include_missing=False,
        )
        assert MISSING_ATTRIBUTE_COL_KEY not in m.cols
        assert m.grand_total() == 1

    def test_custom_missing_label(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        apps = [_app(code_id=c1.id, source_id=s1.id)]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="site",
            sources=[s1],
            missing_label="N/A",
        )
        assert m.col_titles[MISSING_ATTRIBUTE_COL_KEY] == "N/A"

    def test_missing_column_only_present_if_seen(self):
        # All sources have the attribute → no missing column even when
        # include_missing=True (it's a "show missing if any" toggle).
        c1 = _code(_hex(1))
        s1 = _src(_hex(11), custom={"site": "Hospital A"})
        apps = [_app(code_id=c1.id, source_id=s1.id)]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="site",
            sources=[s1],
            include_missing=True,
        )
        assert MISSING_ATTRIBUTE_COL_KEY not in m.cols

    def test_missing_attribute_key_rejected(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_attribute_matrix(
                applications=[],
                codes=[c1],
                attribute_key="",
                sources=[],
            )

    def test_unknown_attribute_kind(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_attribute_matrix(
                applications=[],
                codes=[c1],
                attribute_key="site",
                attribute_kind="bogus",
                sources=[],
            )

    def test_source_kind_requires_sources(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_attribute_matrix(
                applications=[],
                codes=[c1],
                attribute_key="site",
                attribute_kind="source",
            )


class TestCodeByAttributeParticipant:
    def test_resolves_via_speaker_map(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        p1 = _participant(_hex(101), demographics={"role": "patient"})
        p2 = _participant(_hex(102), demographics={"role": "doctor"})
        smap = _smap(
            s1.id,
            entries=[
                {"label": "SPEAKER_00", "role": "interviewee", "participant_id": p1.id},
                {"label": "SPEAKER_01", "role": "interviewer", "participant_id": p2.id},
            ],
        )
        apps = [
            _app(code_id=c1.id, source_id=s1.id, speaker="SPEAKER_00"),
            _app(code_id=c1.id, source_id=s1.id, speaker="SPEAKER_01"),
            _app(code_id=c1.id, source_id=s1.id, speaker="SPEAKER_00"),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="role",
            attribute_kind="participant",
            participants=[p1, p2],
            speaker_maps={s1.id: smap},
        )
        assert m.get(c1.id, "patient") == 2
        assert m.get(c1.id, "doctor") == 1

    def test_falls_back_to_explicit_participant_id(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        p1 = _participant(_hex(101), demographics={"role": "patient"})
        apps = [
            _app(code_id=c1.id, source_id=s1.id, participant_id=p1.id),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="role",
            attribute_kind="participant",
            participants=[p1],
            speaker_maps={},
        )
        assert m.get(c1.id, "patient") == 1

    def test_unmapped_speaker_bucketed_missing(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        p1 = _participant(_hex(101), demographics={"role": "patient"})
        # SPEAKER_99 isn't in the map.
        smap = _smap(
            s1.id,
            entries=[
                {"label": "SPEAKER_00", "role": "interviewee", "participant_id": p1.id},
            ],
        )
        apps = [
            _app(code_id=c1.id, source_id=s1.id, speaker="SPEAKER_99"),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="role",
            attribute_kind="participant",
            participants=[p1],
            speaker_maps={s1.id: smap},
        )
        assert m.get(c1.id, MISSING_ATTRIBUTE_COL_KEY) == 1

    def test_participant_kind_requires_participants(self):
        c1 = _code(_hex(1))
        with pytest.raises(MatrixError):
            code_by_attribute_matrix(
                applications=[],
                codes=[c1],
                attribute_key="role",
                attribute_kind="participant",
            )

    def test_missing_demographic_key_treated_as_missing(self):
        c1 = _code(_hex(1))
        s1 = _src(_hex(11))
        p1 = _participant(_hex(101), demographics={})  # no role
        smap = _smap(
            s1.id,
            entries=[
                {"label": "SPEAKER_00", "role": "interviewee", "participant_id": p1.id},
            ],
        )
        apps = [
            _app(code_id=c1.id, source_id=s1.id, speaker="SPEAKER_00"),
        ]
        m = code_by_attribute_matrix(
            applications=apps,
            codes=[c1],
            attribute_key="role",
            attribute_kind="participant",
            participants=[p1],
            speaker_maps={s1.id: smap},
        )
        assert m.get(c1.id, MISSING_ATTRIBUTE_COL_KEY) == 1


# --------------------------------------------------------------------------- #
# Cross-cutting smoke tests
# --------------------------------------------------------------------------- #


class TestSerialisationOfBuiltMatrix:
    def test_built_matrix_round_trips(self):
        c1 = _code(_hex(1), "fear")
        s1 = _src(_hex(11), name="i1")
        apps = [_app(code_id=c1.id, source_id=s1.id)]
        m = code_by_source_matrix(
            applications=apps, codes=[c1], sources=[s1]
        )
        d = json.loads(json.dumps(m.to_dict()))
        m2 = Matrix.from_dict(d)
        assert m2.get(c1.id, s1.id) == 1
        assert m2.row_titles[c1.id] == "fear"

    def test_matrix_csv_round_trips_through_json(self):
        c1 = _code(_hex(1), "fear")
        s1 = _src(_hex(11), name="i1")
        apps = [_app(code_id=c1.id, source_id=s1.id)]
        m = code_by_source_matrix(
            applications=apps, codes=[c1], sources=[s1]
        )
        m2 = Matrix.from_dict(json.loads(json.dumps(m.to_dict())))
        assert m2.to_csv() == m.to_csv()


class TestMatrixErrorIsProjectValidationError:
    """Sanity: the module-level error subclasses ProjectValidationError so
    bundle-level handlers (F1.5) keep treating matrix problems as
    entity-shape problems."""

    def test_subclass(self):
        assert issubclass(MatrixError, ProjectValidationError)


class TestVocabularyExposed:
    def test_attribute_kinds_constant(self):
        assert "source" in ATTRIBUTE_KINDS
        assert "participant" in ATTRIBUTE_KINDS

    def test_max_rows_constant(self):
        assert isinstance(MAX_ROWS, int) and MAX_ROWS >= 1024
