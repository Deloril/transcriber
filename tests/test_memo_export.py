"""Tests for scribe.memo_export (F5.4).

Exercise the four memo exporters in pure Python: in-memory filter,
CSV, structured Markdown, RTF, and JSONL. Every public function is a
pure ``memos -> str`` (or ``filter_memos`` returning a list), so these
tests don't touch the filesystem.

The tests cover:

  * filter combinators (type / target_type / target_id pair / author /
    tag) + sort order;
  * empty memo lists in every format;
  * embedded commas / quotes / pipes (CSV escaping);
  * multi-line memo bodies;
  * Unicode bodies and titles;
  * link rendering with and without target_names + role;
  * the ``build_target_names`` duck-typed builder;
  * fallback heading derivation.
"""

from __future__ import annotations

import csv
import io
import json
import re

import pytest

from scribe.memo_export import (
    CSV_COLUMNS,
    CSV_LIST_SEP,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_JSONL,
    EXPORT_FORMAT_MARKDOWN,
    EXPORT_FORMAT_RTF,
    EXPORT_FORMATS,
    FormatSpec,
    _format_link_for_csv,
    _heading_for,
    _link_label,
    _markdown_memo_block,
    _provenance_source,
    _rtf_escape,
    build_filter_summary,
    build_target_names,
    filter_memos,
    normalise_format,
    render_memos,
    slugify_memos_filename,
    to_csv,
    to_jsonl,
    to_markdown,
    to_rtf,
)
from scribe.memos import Memo, MemoLink
from scribe.projects import Project, ProjectValidationError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_APPLICATION = "d" * 12
_HEX_PARTICIPANT = "e" * 12
_HEX_MEMO = "f" * 12


def _project(**overrides: object) -> Project:
    payload: dict[str, object] = {
        "name": "Living with chronic illness",
        "methodology": "charmaz",
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Project.new(**payload)  # type: ignore[arg-type]


def _memo(**overrides: object) -> Memo:
    payload: dict[str, object] = {
        "project_id": overrides.pop("project_id", _HEX_PROJECT),
        "type": overrides.pop("type", "theoretical"),
        "title": overrides.pop("title", "On pacing"),
        "body": overrides.pop("body", "Body."),
        "now": overrides.pop("now", "2024-02-01T00:00:00.000000Z"),
    }
    payload.update(overrides)
    return Memo.new(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #


class TestFilterMemos:
    def test_empty_list_is_empty(self) -> None:
        assert filter_memos([]) == []

    def test_no_filter_returns_all_sorted_by_created_then_id(self) -> None:
        early = _memo(now="2024-01-01T00:00:00.000000Z", title="A")
        late = _memo(now="2024-04-01T00:00:00.000000Z", title="B")
        out = filter_memos([late, early])
        assert [m.id for m in out] == [early.id, late.id]

    def test_sort_tiebreaks_on_id(self) -> None:
        # Same created_at; sort should fall back to id.
        a = _memo(now="2024-01-01T00:00:00.000000Z", memo_id="111111111111")
        b = _memo(now="2024-01-01T00:00:00.000000Z", memo_id="222222222222")
        out = filter_memos([b, a])
        assert [m.id for m in out] == [a.id, b.id]

    def test_filter_by_type(self) -> None:
        a = _memo(type="theoretical", title="thy")
        b = _memo(type="reflexive", title="reflex")
        out = filter_memos([a, b], type="reflexive")
        assert [m.id for m in out] == [b.id]

    def test_filter_by_target_type(self) -> None:
        code_link = MemoLink(target_type="code", target_id=_HEX_CODE)
        src_link = MemoLink(target_type="source", target_id=_HEX_SOURCE)
        a = _memo(links=[code_link])
        b = _memo(links=[src_link])
        c = _memo()
        got = {m.id for m in filter_memos([a, b, c], target_type="code")}
        assert got == {a.id}

    def test_filter_by_target_id(self) -> None:
        link_a = MemoLink(target_type="code", target_id=_HEX_CODE)
        link_b = MemoLink(target_type="source", target_id=_HEX_SOURCE)
        # Two memos linking to the *same* target id but different types.
        same = "abcdef012345"
        link_c1 = MemoLink(target_type="code", target_id=same)
        link_c2 = MemoLink(target_type="source", target_id=same)
        m1 = _memo(links=[link_a])
        m2 = _memo(links=[link_b])
        m3 = _memo(links=[link_c1])
        m4 = _memo(links=[link_c2])
        got = {m.id for m in filter_memos([m1, m2, m3, m4], target_id=same)}
        assert got == {m3.id, m4.id}

    def test_filter_target_type_and_id_must_match_same_link(self) -> None:
        # Memo has two links: (code, X) and (source, Y). Filter for
        # (code, Y) must NOT match — the pair has to match on the *same*
        # link.
        x = "111111111111"
        y = "222222222222"
        m = _memo(
            links=[
                MemoLink(target_type="code", target_id=x),
                MemoLink(target_type="source", target_id=y),
            ]
        )
        # (code, y) — type matches first link, id matches second link;
        # not allowed.
        assert filter_memos([m], target_type="code", target_id=y) == []
        # (code, x) is the legitimate hit.
        assert filter_memos([m], target_type="code", target_id=x) == [m]

    def test_filter_by_author_coder_id(self) -> None:
        a = _memo(author_coder_id=_HEX_CODER)
        b = _memo()
        got = filter_memos([a, b], author_coder_id=_HEX_CODER)
        assert got == [a]

    def test_filter_by_tag(self) -> None:
        a = _memo(tags=["category", "theory"])
        b = _memo(tags=["method"])
        got = filter_memos([a, b], tag="theory")
        assert got == [a]

    def test_filters_combine_with_and(self) -> None:
        a = _memo(type="reflexive", tags=["bias"])
        b = _memo(type="reflexive", tags=["other"])
        c = _memo(type="theoretical", tags=["bias"])
        got = filter_memos([a, b, c], type="reflexive", tag="bias")
        assert got == [a]

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_memos([], type="cosmology")

    def test_invalid_target_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_memos([], target_type="bogus")

    def test_invalid_target_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_memos([], target_id="not-hex")

    def test_invalid_author_coder_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_memos([], author_coder_id="not-hex")

    def test_invalid_tag_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_memos([], tag="")
        with pytest.raises(ProjectValidationError):
            filter_memos([], tag="   ")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


class TestToCsv:
    def test_empty_list_is_header_only(self) -> None:
        out = to_csv([])
        rows = list(csv.reader(io.StringIO(out)))
        assert rows == [list(CSV_COLUMNS)]

    def test_columns_match_documented_order(self) -> None:
        m = _memo()
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        assert rows[0] == list(CSV_COLUMNS)

    def test_basic_fields_emitted(self) -> None:
        m = _memo(
            type="theoretical",
            title="On pacing",
            body="Pacing emerges as a recurrent strategy.",
            body_format="markdown",
            author_coder_id=_HEX_CODER,
        )
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        assert len(rows) == 2
        row = dict(zip(rows[0], rows[1]))
        assert row["id"] == m.id
        assert row["type"] == "theoretical"
        assert row["title"] == "On pacing"
        assert row["body"].startswith("Pacing emerges")
        assert row["body_format"] == "markdown"
        assert row["author_coder_id"] == _HEX_CODER

    def test_links_compact_format(self) -> None:
        m = _memo(
            links=[
                MemoLink(
                    target_type="code", target_id=_HEX_CODE
                ),
                MemoLink(
                    target_type="source",
                    target_id=_HEX_SOURCE,
                    role="elaborates",
                ),
            ]
        )
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        cell = dict(zip(rows[0], rows[1]))["links"]
        # Compact: type:id, role appended only when present.
        assert cell == (
            f"code:{_HEX_CODE}{CSV_LIST_SEP}"
            f"source:{_HEX_SOURCE}:elaborates"
        )

    def test_links_named_blank_without_target_names(self) -> None:
        m = _memo(
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)]
        )
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        cell = dict(zip(rows[0], rows[1]))["links_named"]
        assert cell == ""

    def test_links_named_with_target_names(self) -> None:
        m = _memo(
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)]
        )
        names = {("code", _HEX_CODE): "Pacing"}
        rows = list(csv.reader(io.StringIO(to_csv([m], target_names=names))))
        cell = dict(zip(rows[0], rows[1]))["links_named"]
        assert cell == f"Pacing (code:{_HEX_CODE})"

    def test_tags_joined_with_separator(self) -> None:
        m = _memo(tags=["bias", "category-1"])
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        cell = dict(zip(rows[0], rows[1]))["tags"]
        assert cell == f"bias{CSV_LIST_SEP}category-1"

    def test_provenance_source_extracted(self) -> None:
        m = _memo(provenance={"source": "ai_drafted", "model_id": "phi-4"})
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        d = dict(zip(rows[0], rows[1]))
        assert d["provenance_source"] == "ai_drafted"

    def test_csv_quotes_embedded_commas_and_quotes(self) -> None:
        m = _memo(body='Has, "quotes" and, commas, in it.')
        out = to_csv([m])
        rows = list(csv.reader(io.StringIO(out)))
        assert dict(zip(rows[0], rows[1]))["body"] == \
            'Has, "quotes" and, commas, in it.'

    def test_unicode_body_round_trips(self) -> None:
        m = _memo(title="Bocadillos", body="Pacing — comer despacio. Fragüé.")
        out = to_csv([m])
        rows = list(csv.reader(io.StringIO(out)))
        d = dict(zip(rows[0], rows[1]))
        assert d["title"] == "Bocadillos"
        assert "Fragüé" in d["body"]

    def test_csv_uses_crlf_line_endings(self) -> None:
        out = to_csv([_memo()])
        assert "\r\n" in out

    def test_author_coder_id_blank_when_unset(self) -> None:
        m = _memo()
        rows = list(csv.reader(io.StringIO(to_csv([m]))))
        d = dict(zip(rows[0], rows[1]))
        assert d["author_coder_id"] == ""


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestToMarkdown:
    def test_empty_list_renders_no_memos_marker(self) -> None:
        md = to_markdown([])
        assert md.startswith("# Memos")
        assert "(no memos)" in md
        assert md.endswith("\n")

    def test_title_uses_project_name(self) -> None:
        md = to_markdown([], project=_project(name="My Study"))
        assert "# Memos — My Study" in md

    def test_project_metadata_in_header(self) -> None:
        md = to_markdown(
            [_memo(), _memo(memo_id=_HEX_MEMO)],
            project=_project(methodology="charmaz", codebook_stage="focused"),
        )
        assert "**Methodology**: charmaz" in md
        assert "**Stage**: focused" in md
        assert "**Memos**: 2" in md

    def test_memo_count_in_header_without_project(self) -> None:
        md = to_markdown([_memo()])
        assert "**Memos**: 1" in md

    def test_filter_summary_rendered_when_supplied(self) -> None:
        md = to_markdown(
            [_memo()],
            filter_summary="type=theoretical",
        )
        assert "_Filter: type=theoretical_" in md

    def test_per_memo_heading_uses_title(self) -> None:
        m = _memo(title="On pacing")
        md = to_markdown([m])
        assert "## On pacing" in md

    def test_per_memo_heading_falls_back_to_first_body_line(self) -> None:
        m = _memo(title="", body="# Heading\n\nBody continues.")
        md = to_markdown([m])
        # Leading # stripped; body's first line used.
        assert "## Heading" in md

    def test_per_memo_heading_truncates_long_first_line(self) -> None:
        long = "a" * 200
        m = _memo(title="", body=long)
        md = to_markdown([m])
        # Truncated to 80 with ellipsis.
        assert "…" in md

    def test_per_memo_heading_falls_back_to_id_when_blank(self) -> None:
        m = _memo(title="", body="")
        md = to_markdown([m])
        assert f"(untitled memo {m.id})" in md

    def test_inline_metadata_includes_id_and_type(self) -> None:
        m = _memo(type="reflexive")
        md = to_markdown([m])
        assert f"`{m.id}`" in md
        assert "type: reflexive" in md

    def test_inline_metadata_omits_default_format(self) -> None:
        # markdown is the default, so don't render a noisy "format:" bit
        m = _memo(body_format="markdown")
        md = to_markdown([m])
        assert "format: markdown" not in md

    def test_inline_metadata_includes_non_default_format(self) -> None:
        m = _memo(body_format="plain")
        md = to_markdown([m])
        assert "format: plain" in md

    def test_inline_metadata_includes_author_when_set(self) -> None:
        m = _memo(author_coder_id=_HEX_CODER)
        md = to_markdown([m])
        assert f"author: `{_HEX_CODER}`" in md

    def test_body_rendered_below_metadata(self) -> None:
        m = _memo(body="A line.\n\nAnother paragraph.")
        md = to_markdown([m])
        assert "A line." in md
        assert "Another paragraph." in md

    def test_links_rendered_as_bullets(self) -> None:
        m = _memo(
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE),
                MemoLink(
                    target_type="source",
                    target_id=_HEX_SOURCE,
                    role="elaborates",
                ),
            ]
        )
        md = to_markdown([m])
        assert "**Links**" in md
        assert f"- code:{_HEX_CODE}" in md
        assert f"- elaborates: source:{_HEX_SOURCE}" in md

    def test_links_use_target_names_when_supplied(self) -> None:
        m = _memo(
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)]
        )
        names = {("code", _HEX_CODE): "Pacing"}
        md = to_markdown([m], target_names=names)
        assert f"- Pacing (code:{_HEX_CODE})" in md

    def test_tags_section(self) -> None:
        m = _memo(tags=["bias", "method"])
        md = to_markdown([m])
        assert "**Tags**: bias, method" in md

    def test_provenance_section(self) -> None:
        m = _memo(provenance={"source": "ai_drafted", "model_id": "phi-4"})
        md = to_markdown([m])
        assert "**Provenance**: source: ai_drafted; model_id: phi-4" in md

    def test_skips_empty_sections(self) -> None:
        m = _memo(body="Body only.")
        md = to_markdown([m])
        assert "**Links**" not in md
        assert "**Tags**" not in md
        assert "**Provenance**" not in md

    def test_unicode_body_passes_through(self) -> None:
        m = _memo(body="Pacing — comer despacio. Fragüé.")
        md = to_markdown([m])
        assert "Fragüé" in md

    def test_trailing_newline(self) -> None:
        md = to_markdown([_memo()])
        assert md.endswith("\n")

    def test_per_memo_block_drops_trailing_blank(self) -> None:
        block = _markdown_memo_block(_memo(), target_names=None)
        # The helper itself trims trailing blanks; the caller (to_markdown)
        # adds a single inter-memo spacer.
        assert block[-1] != ""


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


class TestToRtf:
    def test_empty_list_is_well_formed(self) -> None:
        out = to_rtf([])
        assert out.startswith("{\\rtf1")
        assert out.endswith("}")
        assert "(no memos)" in out

    def test_title_uses_project_name(self) -> None:
        out = to_rtf([], project=_project(name="My Study"))
        # em-dash escapes via \uNNNN?, but the literal "Memos" and the
        # project name are plain ASCII so they sit unescaped in the body.
        assert "Memos " in out
        assert "My Study" in out

    def test_project_metadata_lines(self) -> None:
        out = to_rtf(
            [_memo()],
            project=_project(methodology="charmaz", codebook_stage="focused"),
        )
        assert "Methodology: charmaz" in out
        assert "Stage: focused" in out
        assert "Memos: 1" in out

    def test_memo_count_emitted_without_project(self) -> None:
        out = to_rtf([_memo()])
        assert "Memos: 1" in out

    def test_filter_summary_rendered_when_supplied(self) -> None:
        out = to_rtf([_memo()], filter_summary="type=reflexive")
        assert "Filter: type=reflexive" in out

    def test_memo_title_bold(self) -> None:
        out = to_rtf([_memo(title="On pacing")])
        # Bold heading begins with \b\fs28 (per _rtf_para_bold).
        assert r"\b\fs28 On pacing" in out

    def test_memo_metadata_line(self) -> None:
        m = _memo(type="reflexive", author_coder_id=_HEX_CODER)
        out = to_rtf([m])
        assert f"id: {m.id}" in out
        assert "type: reflexive" in out
        assert f"author: {_HEX_CODER}" in out

    def test_links_section(self) -> None:
        m = _memo(
            links=[
                MemoLink(
                    target_type="code", target_id=_HEX_CODE, role="exemplifies"
                )
            ]
        )
        out = to_rtf([m])
        assert "Links" in out
        assert f"exemplifies: code:{_HEX_CODE}" in out

    def test_links_use_target_names_when_supplied(self) -> None:
        m = _memo(
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)]
        )
        out = to_rtf([m], target_names={("code", _HEX_CODE): "Pacing"})
        assert "Pacing" in out

    def test_tags_label_block(self) -> None:
        out = to_rtf([_memo(tags=["bias", "method"])])
        assert "Tags" in out
        assert "bias, method" in out

    def test_provenance_label_block(self) -> None:
        out = to_rtf([_memo(provenance={"source": "ai_drafted"})])
        assert "Provenance" in out
        assert "source: ai_drafted" in out

    def test_unicode_chars_escaped_as_uNNNN(self) -> None:
        # em-dash (U+2014 = 8212 decimal, < 0x8000) → unsigned form.
        m = _memo(body="A — B")
        out = to_rtf([m])
        assert "\\u8212?" in out

    def test_astral_plane_char_uses_surrogate_pair(self) -> None:
        # 🦀 is U+1F980; encoded as a surrogate pair in RTF.
        m = _memo(body="🦀")
        out = to_rtf([m])
        # Two \uNNNN? escapes back-to-back from the surrogate pair.
        assert re.search(r"\\u-?\d+\?\\u-?\d+\?", out)


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


class TestToJsonl:
    def test_empty_list_is_empty_string(self) -> None:
        assert to_jsonl([]) == ""

    def test_one_memo_one_line(self) -> None:
        m = _memo(title="Single")
        out = to_jsonl([m])
        assert out.endswith("\n")
        assert out.count("\n") == 1
        parsed = json.loads(out.strip())
        assert parsed["id"] == m.id
        assert parsed["title"] == "Single"

    def test_multiple_memos_one_per_line(self) -> None:
        a = _memo(title="A", memo_id="111111111111")
        b = _memo(title="B", memo_id="222222222222")
        out = to_jsonl([a, b])
        lines = out.rstrip("\n").split("\n")
        assert len(lines) == 2
        ids = [json.loads(line)["id"] for line in lines]
        assert ids == [a.id, b.id]

    def test_unicode_passes_through_without_escape(self) -> None:
        m = _memo(body="café — naïve")
        out = to_jsonl([m])
        # ensure_ascii=False so the literal chars survive.
        assert "café" in out
        assert "naïve" in out

    def test_round_trips_through_memo_from_dict(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE, role="exemplifies")
        m = _memo(links=[link], tags=["bias"], author_coder_id=_HEX_CODER)
        out = to_jsonl([m])
        parsed = json.loads(out.strip())
        # Reconstitute and confirm equality of the public fields.
        rebuilt = Memo.from_dict(parsed)
        assert rebuilt.id == m.id
        assert rebuilt.title == m.title
        assert rebuilt.body == m.body
        assert rebuilt.author_coder_id == _HEX_CODER
        assert [link.to_dict() for link in rebuilt.links] == [
            link.to_dict() for link in m.links
        ]
        assert rebuilt.tags == m.tags


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestFormatLinkForCsv:
    def test_no_role(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE)
        assert _format_link_for_csv(link) == f"code:{_HEX_CODE}"

    def test_with_role(self) -> None:
        link = MemoLink(
            target_type="code", target_id=_HEX_CODE, role="exemplifies"
        )
        assert _format_link_for_csv(link) == f"code:{_HEX_CODE}:exemplifies"


class TestLinkLabel:
    def test_no_name_no_role(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE)
        assert _link_label(link, None) == f"code:{_HEX_CODE}"

    def test_no_name_with_role(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE, role="x")
        assert _link_label(link, None) == f"x: code:{_HEX_CODE}"

    def test_name_no_role(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE)
        names = {("code", _HEX_CODE): "Pacing"}
        assert _link_label(link, names) == f"Pacing (code:{_HEX_CODE})"

    def test_name_with_role(self) -> None:
        link = MemoLink(
            target_type="code", target_id=_HEX_CODE, role="exemplifies"
        )
        names = {("code", _HEX_CODE): "Pacing"}
        assert (
            _link_label(link, names)
            == f"exemplifies: Pacing (code:{_HEX_CODE})"
        )

    def test_falls_back_to_compact_when_target_missing(self) -> None:
        link = MemoLink(target_type="source", target_id=_HEX_SOURCE)
        names = {("code", _HEX_CODE): "Pacing"}
        # Different target_type/target_id; no name match.
        assert _link_label(link, names) == f"source:{_HEX_SOURCE}"


class TestHeadingFor:
    def test_uses_title_when_present(self) -> None:
        assert _heading_for(_memo(title="Hello", body="World")) == "Hello"

    def test_strips_leading_hashes_from_first_body_line(self) -> None:
        m = _memo(title="", body="### Heading\nbody")
        assert _heading_for(m) == "Heading"

    def test_skips_blank_lines(self) -> None:
        m = _memo(title="", body="   \n\n   actual heading\nrest")
        assert _heading_for(m) == "actual heading"

    def test_truncates_to_80_chars(self) -> None:
        long = "a" * 100
        m = _memo(title="", body=long)
        out = _heading_for(m)
        assert len(out) <= 80
        assert out.endswith("…")

    def test_falls_back_to_id_marker(self) -> None:
        m = _memo(title="", body="")
        assert _heading_for(m) == f"(untitled memo {m.id})"


class TestProvenanceSource:
    def test_blank_when_unset(self) -> None:
        assert _provenance_source(_memo()) == ""

    def test_returns_source_value(self) -> None:
        m = _memo(provenance={"source": "ai_modified"})
        assert _provenance_source(m) == "ai_modified"

    def test_blank_when_present_without_source(self) -> None:
        m = _memo(provenance={"model_id": "phi-4", "source": "human"})
        assert _provenance_source(m) == "human"


class TestRtfEscape:
    def test_plain_ascii_passes_through(self) -> None:
        assert _rtf_escape("hello world") == "hello world"

    def test_braces_escaped(self) -> None:
        assert _rtf_escape("{x}") == "\\{x\\}"

    def test_backslash_doubled(self) -> None:
        assert _rtf_escape("\\n") == "\\\\n"

    def test_newline_to_par(self) -> None:
        assert "\\par" in _rtf_escape("a\nb")

    def test_tab_to_tab_command(self) -> None:
        assert "\\tab" in _rtf_escape("a\tb")

    def test_unicode_escaped(self) -> None:
        # — is U+2014 = 8212 decimal; < 0x8000 so emitted unsigned.
        assert "\\u8212?" in _rtf_escape("—")

    def test_high_unicode_signed_form(self) -> None:
        # U+E000 = 57344; ≥ 0x8000 so emitted as signed (-8192).
        assert "\\u-8192?" in _rtf_escape("")


# --------------------------------------------------------------------------- #
# build_target_names
# --------------------------------------------------------------------------- #


class _StubEntity:
    """Duck-typed stand-in for Code/Source/Coder/Participant.

    build_target_names treats anything with ``.id`` and the right
    label attribute as if it were one of those entities, so we don't
    need to import the real classes here.
    """

    def __init__(self, id: str, **kwargs: object) -> None:
        self.id = id
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubApplication:
    def __init__(self, id: str, *, code_id: str, source_id: str) -> None:
        self.id = id
        self.code_id = code_id
        self.source_id = source_id


class TestBuildTargetNames:
    def test_empty_inputs_produce_empty_map(self) -> None:
        assert build_target_names() == {}

    def test_codes_use_name(self) -> None:
        codes = [_StubEntity(_HEX_CODE, name="Pacing")]
        out = build_target_names(codes=codes)
        assert out == {("code", _HEX_CODE): "Pacing"}

    def test_sources_use_name(self) -> None:
        sources = [_StubEntity(_HEX_SOURCE, name="Interview 1")]
        out = build_target_names(sources=sources)
        assert out == {("source", _HEX_SOURCE): "Interview 1"}

    def test_participants_use_name(self) -> None:
        participants = [_StubEntity(_HEX_PARTICIPANT, name="P1")]
        out = build_target_names(participants=participants)
        assert out == {("participant", _HEX_PARTICIPANT): "P1"}

    def test_coders_use_name(self) -> None:
        coders = [_StubEntity(_HEX_CODER, name="Luke")]
        out = build_target_names(coders=coders)
        assert out == {("coder", _HEX_CODER): "Luke"}

    def test_applications_use_code_at_source(self) -> None:
        apps = [
            _StubApplication(
                _HEX_APPLICATION, code_id=_HEX_CODE, source_id=_HEX_SOURCE
            )
        ]
        out = build_target_names(applications=apps)
        assert out == {
            ("application", _HEX_APPLICATION): f"{_HEX_CODE}@{_HEX_SOURCE}"
        }

    def test_memos_use_title(self) -> None:
        m = _memo(title="On pacing")
        out = build_target_names(memos=[m])
        assert out == {("memo", m.id): "On pacing"}

    def test_memos_with_blank_title_skipped(self) -> None:
        # A memo with no title contributes no entry; the link will
        # fall back to the compact form.
        m = _memo(title="", body="body")
        out = build_target_names(memos=[m])
        assert out == {}

    def test_project_uses_name(self) -> None:
        proj = _project(name="MyProj")
        out = build_target_names(project=proj)
        assert out[("project", proj.id)] == "MyProj"

    def test_skips_entities_without_id(self) -> None:
        # Don't crash on partial inputs.
        codes = [_StubEntity("", name="No id"), _StubEntity(_HEX_CODE, name="X")]
        out = build_target_names(codes=codes)
        assert out == {("code", _HEX_CODE): "X"}

    def test_combines_all_entity_types(self) -> None:
        out = build_target_names(
            codes=[_StubEntity(_HEX_CODE, name="C")],
            sources=[_StubEntity(_HEX_SOURCE, name="S")],
            participants=[_StubEntity(_HEX_PARTICIPANT, name="P")],
            coders=[_StubEntity(_HEX_CODER, name="K")],
            applications=[
                _StubApplication(
                    _HEX_APPLICATION, code_id=_HEX_CODE, source_id=_HEX_SOURCE
                )
            ],
            memos=[_memo(title="M")],
            project=_project(name="Proj"),
        )
        assert ("code", _HEX_CODE) in out
        assert ("source", _HEX_SOURCE) in out
        assert ("participant", _HEX_PARTICIPANT) in out
        assert ("coder", _HEX_CODER) in out
        assert ("application", _HEX_APPLICATION) in out
        assert any(k[0] == "memo" for k in out.keys())
        assert any(k[0] == "project" for k in out.keys())


# --------------------------------------------------------------------------- #
# F5.4 user-facing surface — format registry
# --------------------------------------------------------------------------- #


class TestExportFormatsRegistry:
    """The format registry is the public contract for the F5.4
    download endpoint. Adding new formats is fine; renaming or
    removing keys is a breaking change."""

    def test_four_canonical_formats(self) -> None:
        assert set(EXPORT_FORMATS.keys()) == {
            EXPORT_FORMAT_CSV,
            EXPORT_FORMAT_MARKDOWN,
            EXPORT_FORMAT_RTF,
            EXPORT_FORMAT_JSONL,
        }

    def test_format_specs_are_complete(self) -> None:
        for spec in EXPORT_FORMATS.values():
            assert isinstance(spec, FormatSpec)
            assert spec.key
            assert spec.extension.startswith(".")
            assert spec.media_type
            assert spec.label

    def test_csv_extension_and_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_CSV]
        assert spec.extension == ".csv"
        assert spec.media_type.startswith("text/csv")

    def test_markdown_extension_and_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_MARKDOWN]
        assert spec.extension == ".md"
        assert spec.media_type.startswith("text/markdown")

    def test_rtf_extension_and_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_RTF]
        assert spec.extension == ".rtf"
        assert spec.media_type == "application/rtf"

    def test_jsonl_extension_and_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_JSONL]
        assert spec.extension == ".jsonl"
        assert "ndjson" in spec.media_type


class TestNormaliseFormat:
    """``normalise_format`` is the alias-resolver every URL hits."""

    def test_canonical_keys_pass_through(self) -> None:
        for key in EXPORT_FORMATS.keys():
            assert normalise_format(key) == key

    def test_md_alias_routes_to_markdown(self) -> None:
        assert normalise_format("md") == EXPORT_FORMAT_MARKDOWN

    def test_word_alias_routes_to_rtf(self) -> None:
        assert normalise_format("word") == EXPORT_FORMAT_RTF
        assert normalise_format("doc") == EXPORT_FORMAT_RTF
        assert normalise_format("docx") == EXPORT_FORMAT_RTF

    def test_ndjson_alias_routes_to_jsonl(self) -> None:
        assert normalise_format("ndjson") == EXPORT_FORMAT_JSONL
        assert normalise_format("json") == EXPORT_FORMAT_JSONL

    def test_case_insensitive(self) -> None:
        assert normalise_format("CSV") == EXPORT_FORMAT_CSV
        assert normalise_format("Markdown") == EXPORT_FORMAT_MARKDOWN

    def test_strips_whitespace(self) -> None:
        assert normalise_format("  rtf  ") == EXPORT_FORMAT_RTF

    def test_none_raises_with_actionable_message(self) -> None:
        with pytest.raises(ValueError) as exc:
            normalise_format(None)
        assert "required" in str(exc.value).lower()

    def test_unknown_raises_with_format_list(self) -> None:
        with pytest.raises(ValueError) as exc:
            normalise_format("xlsx")
        msg = str(exc.value)
        assert "xlsx" in msg
        assert "csv" in msg
        assert "markdown" in msg


class TestRenderMemos:
    """``render_memos`` dispatches the four pure exporters by format."""

    def test_csv_dispatch_matches_to_csv(self) -> None:
        memos = [_memo()]
        assert render_memos("csv", memos) == to_csv(memos)

    def test_markdown_dispatch_matches_to_markdown(self) -> None:
        memos = [_memo()]
        proj = _project()
        assert render_memos(
            "markdown", memos, project=proj
        ) == to_markdown(memos, project=proj)

    def test_rtf_dispatch_matches_to_rtf(self) -> None:
        memos = [_memo()]
        proj = _project()
        assert render_memos(
            "rtf", memos, project=proj
        ) == to_rtf(memos, project=proj)

    def test_jsonl_dispatch_matches_to_jsonl(self) -> None:
        memos = [_memo()]
        # JSONL ignores project / target_names / filter_summary.
        out = render_memos(
            "jsonl",
            memos,
            project=_project(name="Should not leak"),
            target_names={("code", _HEX_CODE): "Pacing"},
            filter_summary="type=theoretical",
        )
        assert out == to_jsonl(memos)
        # The project name and filter summary do NOT appear in JSONL.
        assert "Should not leak" not in out
        assert "type=theoretical" not in out

    def test_aliases_resolve_through_render(self) -> None:
        memos = [_memo()]
        # ``md`` resolves to markdown.
        assert render_memos("md", memos) == to_markdown(memos)
        # ``word`` resolves to RTF.
        assert render_memos("word", memos) == to_rtf(memos)

    def test_filter_summary_lands_in_markdown(self) -> None:
        memos = [_memo()]
        out = render_memos(
            "markdown",
            memos,
            filter_summary="type=theoretical",
        )
        assert "Filter: type=theoretical" in out

    def test_filter_summary_skipped_for_csv(self) -> None:
        # CSV column-shape is the public contract — no header row.
        memos = [_memo()]
        out = render_memos(
            "csv",
            memos,
            filter_summary="type=theoretical",
        )
        assert "type=theoretical" not in out

    def test_empty_inputs_dont_crash(self) -> None:
        # All four formats accept zero memos.
        assert render_memos("csv", []).startswith("id,")
        assert render_memos("markdown", []) != ""
        assert render_memos("rtf", []).startswith(r"{\rtf")
        assert render_memos("jsonl", []) == ""

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            render_memos("xlsx", [_memo()])


class TestSlugifyMemosFilename:
    """Filename slugs follow the same NFKD-+-ASCII rule as
    ``slugify_codebook_filename``. The ``-memos`` infix distinguishes
    a memos export from a codebook export when both land in the same
    Downloads folder."""

    def test_with_project_name(self) -> None:
        proj = _project(name="Pilot study")
        assert slugify_memos_filename(proj, "csv") == "pilot-study-memos.csv"

    def test_with_unicode_project_name(self) -> None:
        proj = _project(name="Café société")
        # NFKD downgrade strips combining marks.
        assert slugify_memos_filename(proj, "markdown") == "cafe-societe-memos.md"

    def test_no_project_falls_back_to_memos(self) -> None:
        assert slugify_memos_filename(None, "rtf") == "memos.rtf"

    def test_blank_project_name_falls_back(self) -> None:
        # Project.new validates a non-empty name, so to exercise the
        # blank-slug fallback we stub a duck-typed object whose ``name``
        # is whitespace-only.
        class _Stub:
            name = "   "

        assert slugify_memos_filename(_Stub(), "jsonl") == "memos.jsonl"

    def test_jsonl_extension(self) -> None:
        proj = _project(name="X")
        assert slugify_memos_filename(proj, "jsonl") == "x-memos.jsonl"

    def test_resolves_aliases(self) -> None:
        proj = _project(name="X")
        # ``word`` → RTF
        assert slugify_memos_filename(proj, "word") == "x-memos.rtf"
        # ``md`` → Markdown
        assert slugify_memos_filename(proj, "md") == "x-memos.md"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            slugify_memos_filename(_project(), "xlsx")


class TestBuildFilterSummary:
    """The summary line lands in the Markdown / RTF export header so
    the file explains which filters were applied."""

    def test_no_filters_returns_empty(self) -> None:
        assert build_filter_summary() == ""

    def test_single_filter(self) -> None:
        assert build_filter_summary(type="theoretical") == "type=theoretical"

    def test_multiple_filters_join_with_comma(self) -> None:
        out = build_filter_summary(
            type="theoretical",
            target_type="code",
            tag="early",
        )
        assert out == "type=theoretical, target_type=code, tag=early"

    def test_argument_order_is_stable(self) -> None:
        # Stable across calls so snapshot tests stay reliable.
        first = build_filter_summary(type="t", target_type="code")
        second = build_filter_summary(type="t", target_type="code")
        assert first == second

    def test_blanks_treated_as_absent(self) -> None:
        assert build_filter_summary(type="", target_type=None) == ""
