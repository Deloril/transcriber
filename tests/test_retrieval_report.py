"""Tests for ``scribe.retrieval_report`` (F6.2).

Exercise the coded-segment retrieval pipeline in pure Python:

  1. row construction from applications + codes / sources / coders /
     participants + optional segments,
  2. filtering by code / source / coder / participant,
  3. grouping by code / source / participant / none,
  4. CSV / Markdown / RTF rendering,
  5. dispatch + format-alias handling,
  6. filename slug + atomic disk write.

Every helper is pure — these tests don't touch the filesystem
except :func:`write_report`, which uses ``tmp_path``.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import pytest

from scribe.applications import Application
from scribe.coders import Coder
from scribe.codes import Code
from scribe.participants import Participant
from scribe.projects import Project
from scribe.retrieval_report import (
    CSV_COLUMNS,
    CSV_LIST_SEP,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_MARKDOWN,
    EXPORT_FORMAT_RTF,
    EXPORT_FORMATS,
    GROUP_BY_CODE,
    GROUP_BY_KEYS,
    GROUP_BY_NONE,
    GROUP_BY_PARTICIPANT,
    GROUP_BY_SOURCE,
    LABEL_NO_PARTICIPANT,
    LABEL_UNKNOWN,
    RetrievalGroup,
    RetrievalRow,
    build_retrieval_rows,
    filter_rows,
    group_rows,
    normalise_format,
    normalise_group_by,
    render_report,
    slugify_report_filename,
    to_csv,
    to_markdown,
    to_rtf,
    write_report,
)
from scribe.sources import Source


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


PROJ_ID = "abcdef012345"
SRC_A = "111111111111"
SRC_B = "222222222222"
CODE_A = "333333333333"
CODE_B = "444444444444"
CODER_A = "555555555555"
CODER_B = "666666666666"
PART_A = "777777777777"
PART_B = "888888888888"
APP_1 = "999999999991"
APP_2 = "999999999992"
APP_3 = "999999999993"
DEF_VER = "a" * 12


def _project(**overrides: Any) -> Project:
    payload: dict[str, Any] = {
        "name": "Pacing study",
        "methodology": "charmaz",
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Project.new(**payload)


def _code(code_id: str, name: str, project_id: str = PROJ_ID) -> Code:
    return Code.new(
        project_id=project_id,
        name=name,
        code_id=code_id,
        now="2024-01-01T00:00:00.000000Z",
    )


def _source(source_id: str, name: str) -> Source:
    return Source.new(
        project_id=PROJ_ID,
        name=name,
        source_id=source_id,
        now="2024-01-01T00:00:00.000000Z",
    )


def _coder(coder_id: str, name: str) -> Coder:
    return Coder.new(
        project_id=PROJ_ID,
        name=name,
        coder_id=coder_id,
        now="2024-01-01T00:00:00.000000Z",
    )


def _participant(
    pid: str, name: str, source_ids: list[str] | None = None
) -> Participant:
    return Participant.new(
        project_id=PROJ_ID,
        name=name,
        participant_id=pid,
        source_ids=source_ids or [],
        now="2024-01-01T00:00:00.000000Z",
    )


def _application(
    *,
    application_id: str,
    code_id: str = CODE_A,
    source_id: str = SRC_A,
    coder_id: str = CODER_A,
    start_word: str = "s0w0",
    end_word: str = "s0w2",
    start_offset: int | None = None,
    end_offset: int | None = None,
    confidence: float | None = None,
    provenance: dict[str, str] | None = None,
    note: str = "",
    now: str = "2024-01-02T00:00:00.000000Z",
) -> Application:
    return Application.new(
        project_id=PROJ_ID,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start_word,
        anchor_end_word_id=end_word,
        definition_version_id_at_apply=DEF_VER,
        application_id=application_id,
        start_char_offset=start_offset,
        end_char_offset=end_offset,
        confidence=confidence,
        provenance=provenance,
        note=note,
        now=now,
    )


def _segments_3_words() -> list[dict[str, Any]]:
    """A trivial transcript: one segment, three words."""
    return [
        {
            "start": 0.0,
            "end": 1.5,
            "words": [
                {"text": "hello", "start": 0.0, "end": 0.4},
                {"text": "brave", "start": 0.4, "end": 0.9},
                {"text": "world", "start": 0.9, "end": 1.5},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# build_retrieval_rows
# --------------------------------------------------------------------------- #


class TestBuildRetrievalRows:
    def test_minimal_rows_just_applications(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        assert len(rows) == 1
        r = rows[0]
        assert r.application_id == APP_1
        assert r.code_id == CODE_A
        assert r.code_name == ""  # no codes supplied
        assert r.source_id == SRC_A
        assert r.source_name == ""
        assert r.coder_name == ""
        assert r.participant_ids == ()
        assert r.text == ""

    def test_hydrates_code_source_coder_names(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            codes=[_code(CODE_A, "Pacing")],
            sources=[_source(SRC_A, "Interview 01")],
            coders=[_coder(CODER_A, "Luke")],
        )
        r = rows[0]
        assert r.code_name == "Pacing"
        assert r.source_name == "Interview 01"
        assert r.coder_name == "Luke"

    def test_participants_for_source(self) -> None:
        app = _application(application_id=APP_1, source_id=SRC_A)
        # Focus group: two participants both linked to SRC_A.
        ps = [
            _participant(PART_A, "Alice", source_ids=[SRC_A]),
            _participant(PART_B, "Bob", source_ids=[SRC_A]),
        ]
        rows = build_retrieval_rows(
            applications=[app],
            participants=ps,
        )
        r = rows[0]
        assert r.participant_ids == (PART_A, PART_B)
        assert r.participant_names == ("Alice", "Bob")

    def test_participants_only_for_matching_source(self) -> None:
        app = _application(application_id=APP_1, source_id=SRC_A)
        ps = [_participant(PART_A, "Alice", source_ids=[SRC_B])]
        rows = build_retrieval_rows(applications=[app], participants=ps)
        assert rows[0].participant_ids == ()

    def test_text_extracted_from_segments(self) -> None:
        app = _application(
            application_id=APP_1, start_word="s0w0", end_word="s0w2"
        )
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        assert rows[0].text == "hello brave world"

    def test_text_with_subword_offsets(self) -> None:
        app = _application(
            application_id=APP_1,
            start_word="s0w0",
            end_word="s0w0",
            start_offset=1,
            end_offset=4,
        )
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        # ``hello`` -> [1:4] -> "ell"
        assert rows[0].text == "ell"

    def test_text_empty_when_segments_missing_for_source(self) -> None:
        app = _application(application_id=APP_1, source_id=SRC_B)
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        assert rows[0].text == ""

    def test_text_empty_when_no_segments_map(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        assert rows[0].text == ""

    def test_text_empty_for_out_of_range_anchor(self) -> None:
        # Anchor at s9w9 — no such word. anchored_words returns None.
        app = _application(
            application_id=APP_1, start_word="s9w9", end_word="s9w9"
        )
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        assert rows[0].text == ""

    def test_provenance_source_extracted(self) -> None:
        app = _application(
            application_id=APP_1, provenance={"source": "human"}
        )
        rows = build_retrieval_rows(applications=[app])
        assert rows[0].provenance_source == "human"

    def test_confidence_and_note_carried_through(self) -> None:
        app = _application(
            application_id=APP_1,
            confidence=0.875,
            note="Sounds tentative",
        )
        rows = build_retrieval_rows(applications=[app])
        assert rows[0].confidence == 0.875
        assert rows[0].note == "Sounds tentative"

    def test_row_order_matches_application_order(self) -> None:
        a1 = _application(application_id=APP_1)
        a2 = _application(application_id=APP_2, code_id=CODE_B)
        a3 = _application(application_id=APP_3)
        rows = build_retrieval_rows(applications=[a3, a1, a2])
        assert [r.application_id for r in rows] == [APP_3, APP_1, APP_2]


# --------------------------------------------------------------------------- #
# filter_rows
# --------------------------------------------------------------------------- #


def _two_rows() -> list[RetrievalRow]:
    a1 = _application(application_id=APP_1, code_id=CODE_A, source_id=SRC_A,
                      coder_id=CODER_A)
    a2 = _application(application_id=APP_2, code_id=CODE_B, source_id=SRC_B,
                      coder_id=CODER_B)
    ps = [
        _participant(PART_A, "Alice", source_ids=[SRC_A]),
        _participant(PART_B, "Bob", source_ids=[SRC_B]),
    ]
    return build_retrieval_rows(applications=[a1, a2], participants=ps)


class TestFilterRows:
    def test_no_filters_returns_all(self) -> None:
        rows = _two_rows()
        assert filter_rows(rows) == rows

    def test_filter_by_code(self) -> None:
        rows = _two_rows()
        assert [r.application_id for r in filter_rows(rows, code_ids=[CODE_A])] == [APP_1]

    def test_filter_by_source(self) -> None:
        rows = _two_rows()
        assert [r.application_id for r in filter_rows(rows, source_ids=[SRC_B])] == [APP_2]

    def test_filter_by_coder(self) -> None:
        rows = _two_rows()
        assert [r.application_id for r in filter_rows(rows, coder_ids=[CODER_A])] == [APP_1]

    def test_filter_by_participant(self) -> None:
        rows = _two_rows()
        assert [
            r.application_id for r in filter_rows(rows, participant_ids=[PART_B])
        ] == [APP_2]

    def test_filter_combined_and(self) -> None:
        rows = _two_rows()
        # CODE_A on SRC_B doesn't exist → empty.
        assert filter_rows(rows, code_ids=[CODE_A], source_ids=[SRC_B]) == []

    def test_empty_filter_set_matches_nothing(self) -> None:
        rows = _two_rows()
        # An empty list (rather than None) means "match nothing".
        assert filter_rows(rows, code_ids=[]) == []

    def test_filter_by_participant_focus_group_matches(self) -> None:
        # Build a row whose source has TWO participants; filtering on
        # either should match.
        app = _application(application_id=APP_1, source_id=SRC_A)
        ps = [
            _participant(PART_A, "Alice", source_ids=[SRC_A]),
            _participant(PART_B, "Bob", source_ids=[SRC_A]),
        ]
        rows = build_retrieval_rows(applications=[app], participants=ps)
        assert filter_rows(rows, participant_ids=[PART_B]) == rows


# --------------------------------------------------------------------------- #
# group_rows + normalise_group_by
# --------------------------------------------------------------------------- #


class TestNormaliseGroupBy:
    def test_default_is_code(self) -> None:
        assert normalise_group_by(None) == GROUP_BY_CODE

    def test_aliases(self) -> None:
        assert normalise_group_by("Codes") == GROUP_BY_CODE
        assert normalise_group_by("sources") == GROUP_BY_SOURCE
        assert normalise_group_by("participants") == GROUP_BY_PARTICIPANT
        assert normalise_group_by("flat") == GROUP_BY_NONE
        assert normalise_group_by("none") == GROUP_BY_NONE
        assert normalise_group_by("") == GROUP_BY_CODE

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_group_by("year")


class TestGroupRows:
    def test_group_by_code(self) -> None:
        rows = _two_rows()
        groups = group_rows(rows, group_by=GROUP_BY_CODE)
        assert len(groups) == 2
        assert {g.key for g in groups} == {CODE_A, CODE_B}
        # Each group has exactly one row.
        assert all(len(g.rows) == 1 for g in groups)

    def test_group_by_source(self) -> None:
        rows = _two_rows()
        groups = group_rows(rows, group_by=GROUP_BY_SOURCE)
        assert {g.key for g in groups} == {SRC_A, SRC_B}

    def test_group_by_participant_focus_group(self) -> None:
        # Build a single row that has two participants: it should
        # appear in *both* participant groups.
        app = _application(application_id=APP_1, source_id=SRC_A)
        ps = [
            _participant(PART_A, "Alice", source_ids=[SRC_A]),
            _participant(PART_B, "Bob", source_ids=[SRC_A]),
        ]
        rows = build_retrieval_rows(applications=[app], participants=ps)
        groups = group_rows(rows, group_by=GROUP_BY_PARTICIPANT)
        assert {g.key for g in groups} == {PART_A, PART_B}
        assert all(g.rows[0].application_id == APP_1 for g in groups)

    def test_group_by_participant_no_link(self) -> None:
        # No participants linked → goes into the "(no participant)" bucket.
        app = _application(application_id=APP_1, source_id=SRC_A)
        rows = build_retrieval_rows(applications=[app])
        groups = group_rows(rows, group_by=GROUP_BY_PARTICIPANT)
        assert len(groups) == 1
        assert groups[0].key == ""
        assert groups[0].label == LABEL_NO_PARTICIPANT

    def test_group_by_none_returns_single_group(self) -> None:
        rows = _two_rows()
        groups = group_rows(rows, group_by=GROUP_BY_NONE)
        assert len(groups) == 1
        assert groups[0].key == ""
        assert len(groups[0].rows) == 2

    def test_group_label_falls_back_to_unknown(self) -> None:
        # Build a row whose code id has no matching name.
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        groups = group_rows(rows, group_by=GROUP_BY_CODE)
        assert groups[0].label == LABEL_UNKNOWN

    def test_group_first_appearance_order(self) -> None:
        # CODE_B appears first in the row order — it should head the
        # group list.
        a1 = _application(application_id=APP_1, code_id=CODE_B)
        a2 = _application(application_id=APP_2, code_id=CODE_A)
        rows = build_retrieval_rows(applications=[a1, a2])
        groups = group_rows(rows, group_by=GROUP_BY_CODE)
        assert [g.key for g in groups] == [CODE_B, CODE_A]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


class TestToCsv:
    def test_empty_is_header_only(self) -> None:
        out = to_csv([])
        # Header only — single CRLF-terminated row.
        assert out == ",".join(CSV_COLUMNS) + "\r\n"

    def test_round_trip_through_csv_module(self) -> None:
        rows = _two_rows()
        out = to_csv(rows)
        # The csv module's reader should parse the body cleanly.
        reader = csv.DictReader(io.StringIO(out))
        body = list(reader)
        assert len(body) == 2
        assert body[0]["code_id"] == CODE_A
        assert body[1]["code_id"] == CODE_B

    def test_participants_joined_with_separator(self) -> None:
        app = _application(application_id=APP_1, source_id=SRC_A)
        ps = [
            _participant(PART_A, "Alice", source_ids=[SRC_A]),
            _participant(PART_B, "Bob", source_ids=[SRC_A]),
        ]
        rows = build_retrieval_rows(applications=[app], participants=ps)
        out = to_csv(rows)
        reader = csv.DictReader(io.StringIO(out))
        record = next(reader)
        assert record["participant_ids"] == CSV_LIST_SEP.join((PART_A, PART_B))
        assert record["participant_names"] == "Alice" + CSV_LIST_SEP + "Bob"

    def test_quoting_handles_commas_and_quotes(self) -> None:
        app = _application(
            application_id=APP_1, note='Said "yes, definitely"'
        )
        rows = build_retrieval_rows(applications=[app])
        out = to_csv(rows)
        # Round-trips intact.
        body = list(csv.DictReader(io.StringIO(out)))
        assert body[0]["note"] == 'Said "yes, definitely"'

    def test_unicode_passes_through(self) -> None:
        app = _application(application_id=APP_1, note="Réflexion — café")
        rows = build_retrieval_rows(applications=[app])
        out = to_csv(rows)
        body = list(csv.DictReader(io.StringIO(out)))
        assert body[0]["note"] == "Réflexion — café"

    def test_offsets_blank_when_none(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        out = to_csv(rows)
        body = list(csv.DictReader(io.StringIO(out)))
        assert body[0]["start_char_offset"] == ""
        assert body[0]["end_char_offset"] == ""
        assert body[0]["confidence"] == ""

    def test_offsets_serialised_when_set(self) -> None:
        app = _application(
            application_id=APP_1,
            start_word="s0w0",
            end_word="s0w0",
            start_offset=1,
            end_offset=4,
            confidence=0.5,
        )
        rows = build_retrieval_rows(applications=[app])
        out = to_csv(rows)
        body = list(csv.DictReader(io.StringIO(out)))
        assert body[0]["start_char_offset"] == "1"
        assert body[0]["end_char_offset"] == "4"
        assert body[0]["confidence"] == "0.5"

    def test_csv_has_crlf_line_endings(self) -> None:
        rows = _two_rows()
        out = to_csv(rows)
        # RFC 4180: CRLF terminates rows.
        assert "\r\n" in out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestToMarkdown:
    def test_empty_renders_placeholder(self) -> None:
        out = to_markdown([])
        assert out.startswith("# Coded segments")
        assert "_(no coded segments)_" in out

    def test_project_title_used(self) -> None:
        out = to_markdown([], project=_project(name="Pacing"))
        assert "# Coded segments — Pacing" in out

    def test_group_by_code_emits_heading(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            codes=[_code(CODE_A, "Pacing")],
        )
        out = to_markdown(rows, group_by=GROUP_BY_CODE)
        assert "## Pacing" in out
        # Group sub-line includes the id and segment count.
        assert f"`{CODE_A}` · 1 segment(s)" in out

    def test_group_by_source(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            sources=[_source(SRC_A, "Interview 01")],
        )
        out = to_markdown(rows, group_by=GROUP_BY_SOURCE)
        assert "## Interview 01" in out

    def test_group_by_participant_no_link(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        out = to_markdown(rows, group_by=GROUP_BY_PARTICIPANT)
        assert f"## {LABEL_NO_PARTICIPANT}" in out

    def test_group_by_none_omits_headings(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            codes=[_code(CODE_A, "Pacing")],
        )
        out = to_markdown(rows, group_by=GROUP_BY_NONE)
        # No ## heading rendered for the group itself.
        assert "##" not in out
        # Application id still present.
        assert APP_1 in out

    def test_quote_block_with_text(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        out = to_markdown(rows, group_by=GROUP_BY_NONE)
        assert "> hello brave world" in out

    def test_quote_block_placeholder_when_no_text(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        out = to_markdown(rows)
        assert "_(no transcript text available)_" in out

    def test_group_metadata_rendered_when_project_supplied(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(applications=[app])
        out = to_markdown(
            rows, project=_project(), group_by=GROUP_BY_CODE
        )
        assert "**Methodology**" in out
        assert "**Rows**: 1" in out
        assert "**Grouped by**: code" in out

    def test_note_renders_as_italic_line(self) -> None:
        app = _application(application_id=APP_1, note="Tentative")
        rows = build_retrieval_rows(applications=[app])
        out = to_markdown(rows)
        assert "_Note:_ Tentative" in out


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


class TestToRtf:
    def test_starts_with_rtf_preamble(self) -> None:
        out = to_rtf([])
        assert out.startswith(r"{\rtf1")
        assert out.endswith("}")

    def test_empty_renders_placeholder(self) -> None:
        out = to_rtf([])
        assert "(no coded segments)" in out

    def test_project_title_in_heading(self) -> None:
        out = to_rtf([], project=_project(name="Pacing"))
        assert "Coded segments \\u8212?" in out  # em-dash escaped
        assert "Pacing" in out

    def test_group_label_appears(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            codes=[_code(CODE_A, "Pacing")],
        )
        out = to_rtf(rows, group_by=GROUP_BY_CODE)
        assert "Pacing" in out

    def test_text_blockquote_present(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            segments_by_source={SRC_A: _segments_3_words()},
        )
        out = to_rtf(rows)
        # Italic-wrapped paragraph for the quote.
        assert r"\i hello brave world\i0\par" in out

    def test_unicode_escaped(self) -> None:
        app = _application(application_id=APP_1, note="café")
        rows = build_retrieval_rows(applications=[app])
        out = to_rtf(rows)
        # 'é' (U+00E9) → \u233?
        assert "\\u233?" in out

    def test_group_by_none_omits_headings(self) -> None:
        app = _application(application_id=APP_1)
        rows = build_retrieval_rows(
            applications=[app],
            codes=[_code(CODE_A, "Pacing")],
        )
        out = to_rtf(rows, group_by=GROUP_BY_NONE)
        # Bold-large heading paragraph for a group label is fs28; the
        # single occurrence in the output is the document title (fs36).
        assert out.count(r"\b\fs28") == 0


# --------------------------------------------------------------------------- #
# Format dispatch
# --------------------------------------------------------------------------- #


class TestNormaliseFormat:
    def test_csv_alias(self) -> None:
        assert normalise_format("csv") == EXPORT_FORMAT_CSV
        assert normalise_format(" CSV ") == EXPORT_FORMAT_CSV

    def test_md_alias(self) -> None:
        assert normalise_format("md") == EXPORT_FORMAT_MARKDOWN
        assert normalise_format("Markdown") == EXPORT_FORMAT_MARKDOWN

    def test_word_aliases_to_rtf(self) -> None:
        for alias in ("word", "doc", "docx", "rtf"):
            assert normalise_format(alias) == EXPORT_FORMAT_RTF

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format("yaml")

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format(None)


class TestRenderReport:
    def test_csv_dispatch(self) -> None:
        rows = _two_rows()
        out = render_report(EXPORT_FORMAT_CSV, rows)
        assert out.startswith(",".join(CSV_COLUMNS))

    def test_markdown_dispatch_with_project(self) -> None:
        rows = _two_rows()
        out = render_report(
            "md", rows, project=_project(), group_by="source"
        )
        assert "# Coded segments — Pacing study" in out

    def test_rtf_dispatch(self) -> None:
        rows = _two_rows()
        out = render_report("word", rows)
        assert out.startswith(r"{\rtf1")

    def test_csv_ignores_project_and_group_by(self) -> None:
        # CSV renderer must ignore project / group_by rather than
        # blow up — the contract is a flat schema.
        rows = _two_rows()
        a = render_report("csv", rows)
        b = render_report("csv", rows, project=_project(), group_by="source")
        assert a == b


# --------------------------------------------------------------------------- #
# Filename slug + atomic write
# --------------------------------------------------------------------------- #


class TestSlugify:
    def test_uses_project_name(self) -> None:
        out = slugify_report_filename(_project(name="Pacing Study"), "csv")
        assert out == "pacing-study-coded-segments.csv"

    def test_no_project(self) -> None:
        assert slugify_report_filename(None, "md") == "coded-segments.md"

    def test_unicode_downgrades(self) -> None:
        out = slugify_report_filename(_project(name="Café Réflexions"), "rtf")
        assert out == "cafe-reflexions-coded-segments.rtf"

    def test_format_alias_is_resolved(self) -> None:
        out = slugify_report_filename(_project(name="Pacing"), "word")
        assert out.endswith(".rtf")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            slugify_report_filename(_project(), "yaml")

    def test_capped_length(self) -> None:
        long_name = "a" * 200
        out = slugify_report_filename(_project(name=long_name), "csv")
        assert out.endswith("-coded-segments.csv")
        # The slug portion should be ≤ 80 chars.
        slug = out.removesuffix("-coded-segments.csv")
        assert len(slug) <= 80


class TestWriteReport:
    def test_writes_csv(self, tmp_path: Path) -> None:
        rows = _two_rows()
        target = tmp_path / "out.csv"
        result = write_report(target, "csv", rows)
        assert result == target
        body = target.read_bytes().decode("utf-8")
        assert body.startswith(",".join(CSV_COLUMNS))

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        rows = _two_rows()
        target = tmp_path / "nested" / "deep" / "out.csv"
        write_report(target, "csv", rows)
        assert target.exists()

    def test_no_tmp_left_behind(self, tmp_path: Path) -> None:
        rows = _two_rows()
        target = tmp_path / "out.csv"
        write_report(target, "csv", rows)
        # ``.csv.tmp`` swap path must not survive the write.
        assert not (tmp_path / "out.csv.tmp").exists()

    def test_word_alias_writes_rtf(self, tmp_path: Path) -> None:
        rows = _two_rows()
        target = tmp_path / "out.rtf"
        write_report(target, "word", rows)
        body = target.read_text(encoding="utf-8")
        assert body.startswith(r"{\rtf1")


# --------------------------------------------------------------------------- #
# Constants surface
# --------------------------------------------------------------------------- #


class TestConstants:
    def test_export_formats_keys(self) -> None:
        assert set(EXPORT_FORMATS.keys()) == {
            EXPORT_FORMAT_CSV,
            EXPORT_FORMAT_MARKDOWN,
            EXPORT_FORMAT_RTF,
        }

    def test_group_by_keys(self) -> None:
        assert GROUP_BY_KEYS == (
            GROUP_BY_CODE,
            GROUP_BY_SOURCE,
            GROUP_BY_PARTICIPANT,
            GROUP_BY_NONE,
        )
