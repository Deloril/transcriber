"""Tests for scribe.codebook_export (F2.6).

Exercise the four codebook exporters in pure Python: CSV, structured
Markdown, RTF, and REFI-QDA Codebook XML. Every function is a pure
``codes -> str`` so these tests don't touch the filesystem.

The exporters round-trip against deliberately tricky inputs:

  * Unicode body text (curly quotes, em-dashes, non-Latin scripts).
  * Embedded commas / quotes / pipes (CSV escaping).
  * Multi-line definitions / memos.
  * Hierarchical codebooks and codes whose parent isn't in the list.
  * Cyclic parent chains that the validator forbids but a hand-edited
    payload could express.
  * Provenance and related-code linkage.
"""

from __future__ import annotations

import csv
import io
import re
from xml.etree import ElementTree as ET

import pytest

from scribe.codebook_export import (
    CSV_COLUMNS,
    CSV_LIST_SEP,
    REFI_QDA_NS,
    REFI_QDA_ORIGIN_DEFAULT,
    _expand_hex_colour,
    _refi_description,
    _resolve_safe_parents,
    _rtf_escape,
    code_id_to_refi_guid,
    refi_guid_to_code_id,
    to_csv,
    to_markdown,
    to_refi_qda_xml,
    to_rtf,
)
from scribe.codes import Code, CodeRelation, new_code_id
from scribe.projects import Project


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _project(**overrides: object) -> Project:
    payload: dict[str, object] = {
        "name": "Living with chronic illness",
        "methodology": "charmaz",
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Project.new(**payload)  # type: ignore[arg-type]


def _code(**overrides: object) -> Code:
    payload: dict[str, object] = {
        "project_id": overrides.pop("project_id", "abcdef012345"),
        "name": "Pacing",
        "definition": "Adjusting daily activity to manage limited energy.",
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Code.new(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


class TestToCsv:
    def test_empty_codebook_is_header_only(self) -> None:
        out = to_csv([])
        rows = list(csv.reader(io.StringIO(out)))
        assert rows == [list(CSV_COLUMNS)]

    def test_columns_match_documented_order(self) -> None:
        # Order is part of the public contract — guard against silent
        # reorderings that would break supervisor scripts.
        c = _code()
        out = to_csv([c])
        rows = list(csv.reader(io.StringIO(out)))
        assert rows[0] == list(CSV_COLUMNS)

    def test_basic_fields_emitted(self) -> None:
        c = _code(
            name="Pacing",
            definition="Reducing activity to fit energy budget.",
            inclusion_criteria="Apply when participant explicitly slows down.",
            exclusion_criteria="Do not apply when energy is unconstrained.",
            theoretical_memo="Connects to the chronic-illness identity work.",
            stage="focused",
            colour="#aabbcc",
            status="active",
        )
        rows = list(csv.reader(io.StringIO(to_csv([c]))))
        assert len(rows) == 2
        row = dict(zip(rows[0], rows[1]))
        assert row["id"] == c.id
        assert row["name"] == "Pacing"
        assert row["definition"] == "Reducing activity to fit energy budget."
        assert row["inclusion_criteria"].startswith("Apply when")
        assert row["exclusion_criteria"].startswith("Do not apply")
        assert row["stage"] == "focused"
        assert row["colour"] == "#aabbcc"
        assert row["status"] == "active"
        assert row["theoretical_memo"].startswith("Connects to")

    def test_exemplars_joined_with_separator(self) -> None:
        c = _code(exemplars=["I rest twice a day", "I sit through phone calls"])
        rows = list(csv.reader(io.StringIO(to_csv([c]))))
        cell = dict(zip(rows[0], rows[1]))["exemplars"]
        assert cell == "I rest twice a day | I sit through phone calls"

    def test_related_codes_formatted_id_colon_relation(self) -> None:
        other_id = "fedcba987654"
        c = _code(
            related_codes=[
                CodeRelation(code_id=other_id, relation_type="associated"),
                CodeRelation(code_id=other_id, relation_type="contrasts_with"),
            ]
        )
        rows = list(csv.reader(io.StringIO(to_csv([c]))))
        cell = dict(zip(rows[0], rows[1]))["related_codes"]
        assert cell == f"{other_id}:associated | {other_id}:contrasts_with"

    def test_parent_name_is_denormalised(self) -> None:
        parent = _code(name="Identity work", code_id="111111111111")
        child = _code(
            name="Pacing", parent_code_id="111111111111", code_id="222222222222"
        )
        rows = list(csv.reader(io.StringIO(to_csv([parent, child]))))
        # Find child row.
        for row in rows[1:]:
            d = dict(zip(rows[0], row))
            if d["id"] == "222222222222":
                assert d["parent_code_id"] == "111111111111"
                assert d["parent_name"] == "Identity work"
                break
        else:  # pragma: no cover
            pytest.fail("child row not found")

    def test_parent_name_blank_when_parent_missing(self) -> None:
        # Parent id refers to a code not in the supplied list — exporter
        # must tolerate it (no exception, blank parent_name cell).
        c = _code(parent_code_id="ffffffffffff")
        rows = list(csv.reader(io.StringIO(to_csv([c]))))
        d = dict(zip(rows[0], rows[1]))
        assert d["parent_code_id"] == "ffffffffffff"
        assert d["parent_name"] == ""

    def test_provenance_source_extracted(self) -> None:
        c = _code(provenance={"source": "ai_suggested", "model_id": "phi-4"})
        rows = list(csv.reader(io.StringIO(to_csv([c]))))
        d = dict(zip(rows[0], rows[1]))
        assert d["provenance_source"] == "ai_suggested"

    def test_csv_quotes_embedded_commas_and_quotes(self) -> None:
        # csv.writer escapes commas and quotes; this guards against a
        # future regression that stops using csv.writer.
        c = _code(definition='Has, "quotes" and, commas, in it.')
        out = to_csv([c])
        rows = list(csv.reader(io.StringIO(out)))
        assert dict(zip(rows[0], rows[1]))["definition"] == \
            'Has, "quotes" and, commas, in it.'

    def test_unicode_definition_round_trips(self) -> None:
        c = _code(
            name="Bocadillos",
            definition="Pacing — comer despacio. — Fragüé fortunately.",
        )
        out = to_csv([c])
        rows = list(csv.reader(io.StringIO(out)))
        d = dict(zip(rows[0], rows[1]))
        assert d["name"] == "Bocadillos"
        assert "Fragüé" in d["definition"]

    def test_csv_uses_crlf_line_endings(self) -> None:
        # The csv module's default. Worth pinning so a future
        # contributor doesn't quietly switch dialects.
        out = to_csv([_code()])
        assert "\r\n" in out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestToMarkdown:
    def test_empty_codebook(self) -> None:
        md = to_markdown([])
        assert md.startswith("# Codebook")
        assert "(empty codebook)" in md

    def test_title_uses_project_name(self) -> None:
        md = to_markdown([], project=_project(name="My Study"))
        assert "# Codebook — My Study" in md

    def test_project_metadata_table(self) -> None:
        md = to_markdown(
            [_code(), _code(name="Other")],
            project=_project(methodology="charmaz", codebook_stage="focused"),
        )
        assert "**Methodology**: charmaz" in md
        assert "**Stage**: focused" in md
        assert "**Codes**: 2" in md

    def test_per_code_heading_and_definition(self) -> None:
        c = _code(name="Pacing", definition="Adjusting activity.")
        md = to_markdown([c])
        assert "## Pacing" in md
        assert "**Definition**" in md
        assert "Adjusting activity." in md

    def test_inline_metadata_line_includes_id_and_stage(self) -> None:
        c = _code(name="Pacing", stage="focused", status="draft")
        md = to_markdown([c])
        # The inline metadata line uses · separators between bits.
        assert f"`{c.id}`" in md
        assert "stage: focused" in md
        assert "status: draft" in md

    def test_exemplars_render_as_bullets(self) -> None:
        c = _code(exemplars=["one", "two"])
        md = to_markdown([c])
        assert "**Exemplars**" in md
        assert "- one" in md
        assert "- two" in md

    def test_related_codes_show_relation_type(self) -> None:
        a_id = "111111111111"
        b_id = "222222222222"
        a = _code(name="Identity work", code_id=a_id)
        b = _code(
            name="Pacing",
            code_id=b_id,
            related_codes=[
                CodeRelation(code_id=a_id, relation_type="broader"),
            ],
        )
        md = to_markdown([a, b])
        # Format: "- _broader_: Identity work (`<id>`)"
        assert "_broader_" in md
        assert "Identity work" in md

    def test_skips_empty_sections(self) -> None:
        c = _code(definition="", inclusion_criteria="", exemplars=[])
        md = to_markdown([c])
        # No empty section labels.
        assert "**Definition**" not in md
        assert "**Inclusion criteria**" not in md
        assert "**Exemplars**" not in md

    def test_parent_label_falls_back_to_id_when_missing(self) -> None:
        c = _code(parent_code_id="ffffffffffff")
        md = to_markdown([c])
        assert "parent: `ffffffffffff`" in md

    def test_parent_label_uses_name_when_present(self) -> None:
        parent = _code(name="Identity work", code_id="111111111111")
        child = _code(
            name="Pacing", parent_code_id="111111111111", code_id="222222222222"
        )
        md = to_markdown([parent, child])
        assert "parent: Identity work (`111111111111`)" in md

    def test_provenance_section(self) -> None:
        c = _code(provenance={"source": "ai_suggested", "model_id": "phi-4"})
        md = to_markdown([c])
        assert "**Provenance**" in md
        assert "source: ai_suggested" in md
        assert "model_id: phi-4" in md

    def test_unicode_passes_through(self) -> None:
        c = _code(name="Días buenos", definition="Días buenos — los aprovecho.")
        md = to_markdown([c])
        assert "## Días buenos" in md
        assert "Días buenos — los aprovecho." in md

    def test_output_ends_with_newline(self) -> None:
        md = to_markdown([_code()])
        assert md.endswith("\n")


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


class TestRtfEscape:
    def test_passes_ascii_through(self) -> None:
        assert _rtf_escape("hello world") == "hello world"

    def test_escapes_braces_and_backslash(self) -> None:
        assert _rtf_escape("a{b}c\\d") == "a\\{b\\}c\\\\d"

    def test_unicode_below_bmp(self) -> None:
        # "é" is U+00E9 = 233 (signed: 233).
        assert _rtf_escape("é") == "\\u233?"

    def test_unicode_signed_wrap(self) -> None:
        # U+8000 = 32768 → signed -32768.
        ch = chr(0x8000)
        out = _rtf_escape(ch)
        assert out == "\\u-32768?"

    def test_astral_plane_emits_surrogate_pair(self) -> None:
        # 🦄 U+1F984 — sanity check the surrogate-pair branch.
        out = _rtf_escape("🦄")
        # Should produce two \\uNNNN? tokens.
        assert out.count("\\u") == 2

    def test_newline_becomes_par(self) -> None:
        assert "\\par" in _rtf_escape("a\nb")

    def test_tab_becomes_tab_token(self) -> None:
        assert "\\tab" in _rtf_escape("a\tb")


class TestToRtf:
    def test_starts_with_rtf_header(self) -> None:
        rtf = to_rtf([_code()])
        assert rtf.startswith("{\\rtf1")
        assert rtf.endswith("}")

    def test_contains_font_table(self) -> None:
        rtf = to_rtf([])
        assert r"{\fonttbl" in rtf

    def test_empty_codebook_renders_placeholder(self) -> None:
        rtf = to_rtf([])
        assert "(empty codebook)" in rtf

    def test_code_name_appears_bolded(self) -> None:
        c = _code(name="Pacing")
        rtf = to_rtf([c])
        # The bolded heading uses \b ... \b0
        assert "Pacing" in rtf
        assert "\\b" in rtf

    def test_definition_label_present(self) -> None:
        c = _code(definition="Test definition body.")
        rtf = to_rtf([c])
        assert "Definition" in rtf
        assert "Test definition body." in rtf

    def test_unicode_codename_escaped(self) -> None:
        c = _code(name="Días buenos")
        rtf = to_rtf([c])
        # Should not contain the raw "í" (RTF needs \u escapes).
        assert "í" not in rtf
        # Should contain a unicode escape for U+00ED = 237.
        assert "\\u237?" in rtf

    def test_braces_in_user_text_are_escaped(self) -> None:
        # Literal braces in user text would otherwise crash an RTF reader.
        c = _code(definition="contains {curly} braces")
        rtf = to_rtf([c])
        # Unescaped { count must equal balanced RTF group count;
        # easier: ensure user text has been escaped.
        assert "\\{curly\\}" in rtf

    def test_exemplars_bullet_lines(self) -> None:
        c = _code(exemplars=["first", "second"])
        rtf = to_rtf([c])
        assert "Exemplars" in rtf
        assert "first" in rtf
        assert "second" in rtf

    def test_project_title(self) -> None:
        rtf = to_rtf([_code()], project=_project(name="My Study"))
        assert "Codebook" in rtf
        # "—" is U+2014 = 8212 → \\u-19764 (signed wrap from 0x2014)
        # OR readable in name string; just check the project name is
        # encoded somewhere.
        assert "My Study" in rtf


# --------------------------------------------------------------------------- #
# REFI-QDA helpers
# --------------------------------------------------------------------------- #


class TestCodeIdToRefiGuid:
    def test_pads_with_zeros(self) -> None:
        assert (
            code_id_to_refi_guid("abcdef012345")
            == "00000000-0000-0000-0000-abcdef012345"
        )

    def test_lower_cases_input(self) -> None:
        assert (
            code_id_to_refi_guid("ABCDEF012345")
            == "00000000-0000-0000-0000-abcdef012345"
        )

    def test_rejects_short_id(self) -> None:
        with pytest.raises(ValueError):
            code_id_to_refi_guid("abc")

    def test_rejects_non_hex(self) -> None:
        with pytest.raises(ValueError):
            code_id_to_refi_guid("zzzzzzzzzzzz")

    def test_round_trip(self) -> None:
        cid = new_code_id()
        guid = code_id_to_refi_guid(cid)
        assert refi_guid_to_code_id(guid) == cid

    def test_recovers_none_on_non_zero_high_bits(self) -> None:
        # A real GUID minted elsewhere shouldn't be readable as one of
        # ours.
        assert (
            refi_guid_to_code_id("11112222-3333-4444-5555-666677778888")
            is None
        )

    def test_recovers_none_on_garbage(self) -> None:
        assert refi_guid_to_code_id("not-a-guid") is None
        assert refi_guid_to_code_id("") is None


class TestExpandHexColour:
    def test_three_to_six(self) -> None:
        assert _expand_hex_colour("#abc") == "#AABBCC"

    def test_six_passes_uppercased(self) -> None:
        assert _expand_hex_colour("#aAbBcC") == "#AABBCC"

    def test_empty_passes_through(self) -> None:
        assert _expand_hex_colour("") == ""


class TestRefiDescription:
    def test_includes_definition(self) -> None:
        body = _refi_description(_code(definition="abc"))
        assert "Definition: abc" in body

    def test_includes_exemplars_block(self) -> None:
        body = _refi_description(_code(exemplars=["one", "two"]))
        assert "Exemplars:" in body
        assert "- one" in body
        assert "- two" in body

    def test_skips_empty(self) -> None:
        body = _refi_description(_code(definition="", exemplars=[]))
        # Only stage/status defaults could land here; default code has
        # both at defaults so body is empty.
        assert body == ""

    def test_omits_default_stage_and_status(self) -> None:
        body = _refi_description(_code(definition="d", stage="initial", status="active"))
        assert "Stage:" not in body
        assert "Status:" not in body

    def test_includes_non_default_stage_and_status(self) -> None:
        body = _refi_description(
            _code(definition="d", stage="focused", status="draft")
        )
        assert "Stage: focused" in body
        assert "Status: draft" in body


# --------------------------------------------------------------------------- #
# REFI-QDA XML — structural tests
# --------------------------------------------------------------------------- #


class TestToRefiQdaXml:
    def test_root_namespace(self) -> None:
        xml = to_refi_qda_xml([])
        root = ET.fromstring(xml)
        # Root tag includes the namespace.
        assert root.tag == f"{{{REFI_QDA_NS}}}CodeBook"

    def test_origin_attribute_set(self) -> None:
        xml = to_refi_qda_xml([])
        root = ET.fromstring(xml)
        assert root.get("origin") == REFI_QDA_ORIGIN_DEFAULT

    def test_custom_origin(self) -> None:
        xml = to_refi_qda_xml([], origin="Scribe 1.2")
        root = ET.fromstring(xml)
        assert root.get("origin") == "Scribe 1.2"

    def test_project_name_attribute_when_supplied(self) -> None:
        xml = to_refi_qda_xml([], project=_project(name="My Study"))
        root = ET.fromstring(xml)
        assert root.get("name") == "My Study"

    def test_empty_codebook_has_codes_container(self) -> None:
        xml = to_refi_qda_xml([])
        root = ET.fromstring(xml)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 0

    def test_single_code_attributes(self) -> None:
        c = _code(name="Pacing", colour="#abc")
        xml = to_refi_qda_xml([c])
        root = ET.fromstring(xml)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        children = list(codes_el)
        assert len(children) == 1
        code_el = children[0]
        assert code_el.get("name") == "Pacing"
        assert code_el.get("guid") == code_id_to_refi_guid(c.id)
        assert code_el.get("isCodable") == "true"
        assert code_el.get("color") == "#AABBCC"

    def test_description_element_present_when_body(self) -> None:
        c = _code(definition="A definition.")
        xml = to_refi_qda_xml([c])
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_NS}}}Codes/{{{REFI_QDA_NS}}}Code"
        )
        desc = code_el.find(f"{{{REFI_QDA_NS}}}Description")
        assert desc is not None
        assert desc.text and "A definition." in desc.text

    def test_description_omitted_when_blank(self) -> None:
        c = _code(definition="", exemplars=[])
        xml = to_refi_qda_xml([c])
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_NS}}}Codes/{{{REFI_QDA_NS}}}Code"
        )
        desc = code_el.find(f"{{{REFI_QDA_NS}}}Description")
        assert desc is None

    def test_hierarchy_is_nested(self) -> None:
        parent_id = "111111111111"
        child_id = "222222222222"
        parent = _code(name="Identity work", code_id=parent_id)
        child = _code(
            name="Pacing", code_id=child_id, parent_code_id=parent_id
        )
        xml = to_refi_qda_xml([parent, child])
        root = ET.fromstring(xml)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        # Only one top-level code; the child is nested under it.
        top_codes = list(codes_el.findall(f"{{{REFI_QDA_NS}}}Code"))
        assert len(top_codes) == 1
        assert top_codes[0].get("name") == "Identity work"
        nested = list(top_codes[0].findall(f"{{{REFI_QDA_NS}}}Code"))
        assert len(nested) == 1
        assert nested[0].get("name") == "Pacing"

    def test_unknown_parent_treated_as_top_level(self) -> None:
        c = _code(parent_code_id="ffffffffffff")
        xml = to_refi_qda_xml([c])
        root = ET.fromstring(xml)
        # Code emits at the top level without exception.
        top = list(root.find(f"{{{REFI_QDA_NS}}}Codes"))
        assert len(top) == 1

    def test_cycle_falls_back_to_flat(self) -> None:
        # Both codes claim each other as parent — _resolve_safe_parents
        # detects the cycle and emits both at top level instead of
        # recursing forever.
        a_id = "111111111111"
        b_id = "222222222222"
        a = _code(code_id=a_id, parent_code_id=b_id)
        b = _code(code_id=b_id, parent_code_id=a_id)
        xml = to_refi_qda_xml([a, b])  # would StackOverflow if naive
        root = ET.fromstring(xml)
        top = list(root.find(f"{{{REFI_QDA_NS}}}Codes"))
        assert len(top) == 2

    def test_xml_is_utf8_with_declaration(self) -> None:
        xml = to_refi_qda_xml([])
        # Python's ET emits a declaration when xml_declaration=True.
        assert xml.startswith("<?xml")
        assert "utf-8" in xml.lower() or "UTF-8" in xml

    def test_unicode_in_name_round_trips(self) -> None:
        c = _code(name="Días buenos")
        xml = to_refi_qda_xml([c])
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_NS}}}Codes/{{{REFI_QDA_NS}}}Code"
        )
        assert code_el.get("name") == "Días buenos"


class TestResolveSafeParents:
    def test_no_parent(self) -> None:
        c = _code()
        result = _resolve_safe_parents([c], {c.id: c})
        assert result == {c.id: None}

    def test_known_parent(self) -> None:
        a_id, b_id = "111111111111", "222222222222"
        a = _code(code_id=a_id)
        b = _code(code_id=b_id, parent_code_id=a_id)
        result = _resolve_safe_parents([a, b], {a_id: a, b_id: b})
        assert result == {a_id: None, b_id: a_id}

    def test_unknown_parent_resolves_to_none(self) -> None:
        c = _code(parent_code_id="ffffffffffff")
        result = _resolve_safe_parents([c], {c.id: c})
        assert result == {c.id: None}

    def test_cycle_resolves_all_to_none(self) -> None:
        a_id, b_id = "111111111111", "222222222222"
        a = _code(code_id=a_id, parent_code_id=b_id)
        b = _code(code_id=b_id, parent_code_id=a_id)
        result = _resolve_safe_parents([a, b], {a_id: a, b_id: b})
        assert result == {a_id: None, b_id: None}


# --------------------------------------------------------------------------- #
# F6.1: format registry + slug + render dispatch + atomic disk write
# --------------------------------------------------------------------------- #


from pathlib import Path

from scribe.codebook_export import (
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_MARKDOWN,
    EXPORT_FORMAT_RTF,
    EXPORT_FORMATS,
    FormatSpec,
    normalise_format,
    render_codebook,
    slugify_codebook_filename,
    write_codebook,
)


class TestExportFormatsRegistry:
    """The ``EXPORT_FORMATS`` table is part of the F6.1 contract.

    UI buttons, the HTTP endpoint, and the CLI all read from it; a
    silent change to keys / extensions / media types breaks all three.
    """

    def test_registry_keys_are_canonical(self) -> None:
        assert set(EXPORT_FORMATS.keys()) == {
            EXPORT_FORMAT_CSV,
            EXPORT_FORMAT_MARKDOWN,
            EXPORT_FORMAT_RTF,
        }

    def test_csv_spec(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_CSV]
        assert isinstance(spec, FormatSpec)
        assert spec.key == "csv"
        assert spec.extension == ".csv"
        assert spec.media_type.startswith("text/csv")
        assert "charset=utf-8" in spec.media_type
        assert spec.label  # human label for UI

    def test_markdown_spec(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_MARKDOWN]
        assert spec.extension == ".md"
        assert spec.media_type.startswith("text/markdown")
        assert "charset=utf-8" in spec.media_type

    def test_rtf_spec(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_RTF]
        assert spec.extension == ".rtf"
        # IETF practice + MS guidance is application/rtf, not text/rtf.
        assert spec.media_type == "application/rtf"

    def test_refi_qda_is_not_in_f6_1_registry(self) -> None:
        # F6.5 owns the REFI-QDA XML button; we keep it off the F6.1
        # surface so adding QDPX-specific metadata (UserCodes etc)
        # later doesn't churn the codebook-only endpoint.
        assert "refi-qda" not in EXPORT_FORMATS
        assert "xml" not in EXPORT_FORMATS

    def test_format_specs_are_immutable(self) -> None:
        # frozen=True dataclasses raise on attribute assignment so the
        # registry stays a real source of truth.
        spec = EXPORT_FORMATS[EXPORT_FORMAT_CSV]
        with pytest.raises(Exception):
            spec.extension = ".tsv"  # type: ignore[misc]


class TestNormaliseFormat:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("csv", "csv"),
            ("CSV", "csv"),
            (" Csv ", "csv"),
            ("markdown", "markdown"),
            ("MD", "markdown"),
            ("md", "markdown"),
            ("rtf", "rtf"),
            ("word", "rtf"),
            ("Word", "rtf"),
            ("doc", "rtf"),
            ("docx", "rtf"),
        ],
    )
    def test_canonical_and_aliases(self, raw: str, expected: str) -> None:
        assert normalise_format(raw) == expected

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format(None)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format("")

    def test_unknown_raises_with_actionable_message(self) -> None:
        with pytest.raises(ValueError) as ei:
            normalise_format("yaml")
        # The error message lists the valid formats so the user can fix it.
        msg = str(ei.value)
        assert "csv" in msg
        assert "markdown" in msg
        assert "rtf" in msg

    def test_xml_alias_not_smuggled_in(self) -> None:
        # We deliberately don't accept ``xml`` here — that's F6.5's
        # surface, and the F6.1 endpoint shouldn't ever return XML.
        with pytest.raises(ValueError):
            normalise_format("xml")
        with pytest.raises(ValueError):
            normalise_format("refi-qda")


class TestRenderCodebook:
    def test_csv_dispatches_to_to_csv(self) -> None:
        c = _code()
        assert render_codebook("csv", [c]) == to_csv([c])

    def test_markdown_dispatches_with_project(self) -> None:
        c = _code()
        p = _project(name="Pilot")
        assert (
            render_codebook("markdown", [c], project=p)
            == to_markdown([c], project=p)
        )

    def test_rtf_dispatches_with_project(self) -> None:
        c = _code()
        p = _project(name="Pilot")
        assert (
            render_codebook("rtf", [c], project=p)
            == to_rtf([c], project=p)
        )

    def test_md_alias(self) -> None:
        c = _code()
        out_md = render_codebook("md", [c])
        out_markdown = render_codebook("markdown", [c])
        assert out_md == out_markdown

    def test_word_alias(self) -> None:
        c = _code()
        out_word = render_codebook("word", [c])
        out_rtf = render_codebook("rtf", [c])
        assert out_word == out_rtf

    def test_csv_ignores_project_header(self) -> None:
        # CSV is a contract; the project must not bleed into the
        # column shape, even if a header sneaks past a refactor.
        c = _code()
        with_proj = render_codebook("csv", [c], project=_project())
        without = render_codebook("csv", [c], project=None)
        assert with_proj == without

    def test_empty_codebook_csv(self) -> None:
        # Header-only CSV is a valid empty-codebook export.
        out = render_codebook("csv", [])
        assert out == to_csv([])
        # First row is the header; sanity check it's non-empty.
        assert "id" in out.splitlines()[0]

    def test_empty_codebook_markdown(self) -> None:
        out = render_codebook("markdown", [])
        assert "_(empty codebook)_" in out

    def test_unknown_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            render_codebook("yaml", [])


class TestSlugifyCodebookFilename:
    def test_no_project_falls_back_to_codebook(self) -> None:
        assert slugify_codebook_filename(None, "csv") == "codebook.csv"
        assert slugify_codebook_filename(None, "markdown") == "codebook.md"
        assert slugify_codebook_filename(None, "rtf") == "codebook.rtf"

    def test_simple_name(self) -> None:
        p = _project(name="Pilot")
        assert (
            slugify_codebook_filename(p, "csv") == "pilot-codebook.csv"
        )

    def test_spaces_become_dashes(self) -> None:
        p = _project(name="Living with chronic illness")
        assert (
            slugify_codebook_filename(p, "csv")
            == "living-with-chronic-illness-codebook.csv"
        )

    def test_punctuation_stripped(self) -> None:
        p = _project(name="Pilot — Phase 1!")
        assert (
            slugify_codebook_filename(p, "md")
            == "pilot-phase-1-codebook.md"
        )

    def test_diacritics_downgraded_to_ascii(self) -> None:
        # NFKD-normalise + drop combining marks → "Café" → "cafe".
        p = _project(name="Café études")
        assert (
            slugify_codebook_filename(p, "csv")
            == "cafe-etudes-codebook.csv"
        )

    def test_non_ascii_only_falls_back_to_codebook(self) -> None:
        # If literally every char drops out, slug is empty and we use
        # the unprefixed default. (Keeps the URL printable.)
        p = _project(name="日本語")
        assert slugify_codebook_filename(p, "csv") == "codebook.csv"

    def test_blank_name_falls_back(self) -> None:
        # Project.new requires a name, so we can't construct one with
        # an empty name. Pass a stub-equivalent via ``name`` patch.
        p = _project(name="Pilot")
        p.name = "   "
        assert slugify_codebook_filename(p, "csv") == "codebook.csv"

    def test_long_name_truncated(self) -> None:
        # Project.new caps name at 200 chars; pick a length above the
        # 80-char slug bound but below the project-name validator.
        p = _project(name="x" * 150)
        out = slugify_codebook_filename(p, "csv")
        # Suffix is preserved; the slug body alone is bounded.
        assert out.endswith("-codebook.csv")
        # The slug portion is at most _FILENAME_SLUG_MAX (80) chars.
        slug = out.removesuffix("-codebook.csv")
        assert len(slug) <= 80
        assert slug == "x" * 80

    def test_extension_matches_format(self) -> None:
        p = _project(name="P")
        assert slugify_codebook_filename(p, "csv").endswith(".csv")
        assert slugify_codebook_filename(p, "markdown").endswith(".md")
        assert slugify_codebook_filename(p, "rtf").endswith(".rtf")

    def test_word_alias_routes_to_rtf_extension(self) -> None:
        p = _project(name="P")
        assert slugify_codebook_filename(p, "word").endswith(".rtf")

    def test_unknown_format_raises(self) -> None:
        p = _project(name="P")
        with pytest.raises(ValueError):
            slugify_codebook_filename(p, "yaml")


class TestWriteCodebook:
    def test_writes_csv_to_disk(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "out.csv"
        result = write_codebook(out, "csv", [c])
        assert result == out
        assert out.exists()
        # Read bytes to assert the exact RFC-4180 ``\r\n`` line endings
        # survive the round-trip — text mode would translate them.
        body = out.read_bytes().decode("utf-8")
        assert body == to_csv([c])
        assert b"\r\n" in out.read_bytes()

    def test_writes_markdown_with_project(self, tmp_path: Path) -> None:
        c = _code()
        p = _project(name="Pilot")
        out = tmp_path / "out.md"
        write_codebook(out, "markdown", [c], project=p)
        text = out.read_text(encoding="utf-8")
        assert text == to_markdown([c], project=p)

    def test_writes_rtf(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "out.rtf"
        write_codebook(out, "rtf", [c])
        # RTF preamble.
        assert out.read_text(encoding="utf-8").startswith(r"{\rtf1")

    def test_unicode_round_trip(self, tmp_path: Path) -> None:
        c = _code(name="Días buenos", definition="Buenos días.")
        out = tmp_path / "out.md"
        write_codebook(out, "markdown", [c])
        text = out.read_text(encoding="utf-8")
        assert "Días buenos" in text

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "nested" / "deep" / "out.csv"
        write_codebook(out, "csv", [c])
        assert out.exists()

    def test_atomic_swap_does_not_leak_tmp(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "out.csv"
        write_codebook(out, "csv", [c])
        # No leftover ``.tmp`` after a successful write.
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_overwrite_is_atomic(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        out.write_text("garbage", encoding="utf-8")
        c = _code()
        write_codebook(out, "csv", [c])
        # New content fully replaced the old.
        assert out.read_bytes().decode("utf-8") == to_csv([c])

    def test_unknown_format_raises_before_touching_disk(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError):
            write_codebook(out, "yaml", [])
        # Nothing written.
        assert not out.exists()

    def test_empty_codebook_writes_header_only_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        write_codebook(out, "csv", [])
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        # First line is the header; no data rows.
        rows = list(csv.reader(io.StringIO(text)))
        assert rows == [list(CSV_COLUMNS)]

    def test_word_alias_writes_rtf(self, tmp_path: Path) -> None:
        # The user types "word" and gets a .rtf-shaped file at the path
        # they specified — extension on the path is whatever they wrote.
        out = tmp_path / "out.docx"
        c = _code()
        write_codebook(out, "word", [c])
        assert out.exists()
        # Body is RTF regardless of what the user named the file.
        assert out.read_text(encoding="utf-8").startswith(r"{\rtf1")


# --------------------------------------------------------------------------- #
# F6.5 — REFI-QDA Codebook XML download surface
#
# F2.6 shipped ``to_refi_qda_xml`` itself; F6.5 wraps it into a download
# surface (project-metadata comment, dedicated filename slug, atomic
# writer). The tests below cover only the F6.5 additions; the underlying
# XML shape is covered exhaustively by ``TestToRefiQdaXml`` above.
# --------------------------------------------------------------------------- #


from scribe.codebook_export import (
    REFI_QDA_XML_EXTENSION,
    REFI_QDA_XML_LABEL,
    REFI_QDA_XML_MEDIA_TYPE,
    _build_project_metadata_comment,
    render_refi_qda_codebook_xml,
    slugify_refi_qda_codebook_xml_filename,
    write_refi_qda_codebook_xml,
)


class TestF65Constants:
    """The F6.5 constants are public — UI / CLI / HTTP all read them."""

    def test_extension_is_dotted_double_extension(self) -> None:
        # ``.refi-qda.xml`` so the file sorts next to a sibling QDPX
        # in the user's downloads and never collides with a plain
        # ``.xml`` artefact from another tool.
        assert REFI_QDA_XML_EXTENSION == ".refi-qda.xml"

    def test_media_type_is_xml_with_utf8(self) -> None:
        assert REFI_QDA_XML_MEDIA_TYPE.startswith("application/xml")
        assert "charset=utf-8" in REFI_QDA_XML_MEDIA_TYPE

    def test_label_is_human_readable(self) -> None:
        assert "REFI-QDA" in REFI_QDA_XML_LABEL


class TestBuildProjectMetadataComment:
    """The comment-block builder is pure and deterministic."""

    def test_only_header_returns_empty_string(self) -> None:
        # A project-shaped object with every metadata field empty
        # produces a single-line "header only" body, which the helper
        # collapses to ``""`` so the caller skips emitting the comment
        # entirely. ``Project.new`` validates ``name`` non-empty, so
        # we use a ``SimpleNamespace`` carrying the same attribute
        # surface to exercise the safety net.
        from types import SimpleNamespace

        stub = SimpleNamespace(
            id="",
            name="",
            research_question="",
            methodology="",
            sensitising_concepts=[],
            codebook_stage="",
            created_at="",
            modified_at="",
        )
        assert _build_project_metadata_comment(stub) == ""  # type: ignore[arg-type]

    def test_includes_project_name(self) -> None:
        p = _project(name="Pilot")
        body = _build_project_metadata_comment(p)
        assert "Pilot" in body
        assert "Project: Pilot" in body

    def test_includes_research_question(self) -> None:
        p = _project(research_question="How do people pace energy?")
        body = _build_project_metadata_comment(p)
        assert "How do people pace energy?" in body
        assert "Research question:" in body

    def test_includes_methodology(self) -> None:
        p = _project(methodology="charmaz")
        body = _build_project_metadata_comment(p)
        assert "Methodology: charmaz" in body

    def test_includes_sensitising_concepts(self) -> None:
        p = _project(
            sensitising_concepts=["identity work", "biographical disruption"],
        )
        body = _build_project_metadata_comment(p)
        assert "identity work" in body
        assert "biographical disruption" in body
        # Joined with comma, not just the first item.
        assert "identity work, biographical disruption" in body

    def test_includes_codebook_stage(self) -> None:
        p = _project(codebook_stage="focused")
        body = _build_project_metadata_comment(p)
        assert "Codebook stage: focused" in body

    def test_includes_timestamps(self) -> None:
        p = _project()
        body = _build_project_metadata_comment(p)
        assert "Created: " in body
        assert "Modified: " in body

    def test_double_dash_collapsed_to_em_dash(self) -> None:
        # XML comments cannot contain ``"--"`` per spec; the helper
        # collapses runs to a single em-dash so user input never
        # produces an unparseable file.
        p = _project(methodology="longitudinal -- 12 months")
        body = _build_project_metadata_comment(p)
        assert "--" not in body

    def test_starts_and_ends_with_newline(self) -> None:
        # The pretty-printer renders the comment as an indented block
        # only when the body has surrounding whitespace.
        p = _project()
        body = _build_project_metadata_comment(p)
        assert body.startswith("\n")
        assert body.endswith("\n")


class TestToRefiQdaXmlWithProjectMetadata:
    """The opt-in flag adds a comment block; default behaviour
    is unchanged so F2.6's existing tests still pass."""

    def test_default_does_not_include_comment(self) -> None:
        p = _project(methodology="charmaz")
        xml = to_refi_qda_xml([], project=p)
        # No <!--…--> in the body when the flag is off.
        assert "<!--" not in xml

    def test_flag_adds_comment(self) -> None:
        p = _project(methodology="charmaz", research_question="RQ")
        xml = to_refi_qda_xml(
            [], project=p, include_project_metadata=True
        )
        assert "<!--" in xml
        assert "Methodology: charmaz" in xml
        assert "Research question: RQ" in xml

    def test_flag_with_no_project_emits_no_comment(self) -> None:
        # Comment is opt-in *and* requires a project. No project → no
        # comment, even with the flag set.
        xml = to_refi_qda_xml([], include_project_metadata=True)
        assert "<!--" not in xml

    def test_xml_remains_parseable(self) -> None:
        p = _project(
            methodology="charmaz",
            research_question="How do people pace energy?",
            sensitising_concepts=["identity work"],
        )
        c = _code()
        xml = to_refi_qda_xml(
            [c], project=p, include_project_metadata=True
        )
        # Round-trip through the parser; the comment doesn't break
        # the XML even though it's the first child of the root.
        root = ET.fromstring(xml)
        assert root.tag == f"{{{REFI_QDA_NS}}}CodeBook"
        # Codes container still present.
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 1

    def test_stub_project_with_empty_metadata_omits_comment(self) -> None:
        # If every project metadata field is empty the comment body
        # collapses to nothing and the writer skips emitting
        # ``<!---->``. ``Project.new`` validates name, so use a
        # ``SimpleNamespace`` to reach the empty-body branch.
        from types import SimpleNamespace

        stub = SimpleNamespace(
            id="",
            name="",
            research_question="",
            methodology="",
            sensitising_concepts=[],
            codebook_stage="",
            created_at="",
            modified_at="",
        )
        xml = to_refi_qda_xml(
            [], project=stub, include_project_metadata=True  # type: ignore[arg-type]
        )
        assert "<!--" not in xml


class TestRenderRefiQdaCodebookXml:
    """The F6.5 entry point always asks for project metadata."""

    def test_includes_metadata_comment_when_project_supplied(self) -> None:
        p = _project(methodology="charmaz")
        xml = render_refi_qda_codebook_xml([], project=p)
        assert "<!--" in xml
        assert "Methodology: charmaz" in xml

    def test_no_project_still_renders(self) -> None:
        # Codebook can be rendered without a project context — used by
        # the CLI when --project's data has been moved.
        xml = render_refi_qda_codebook_xml([])
        # No comment, but valid XML.
        assert "<!--" not in xml
        root = ET.fromstring(xml)
        assert root.tag == f"{{{REFI_QDA_NS}}}CodeBook"

    def test_origin_propagated(self) -> None:
        xml = render_refi_qda_codebook_xml([], origin="Scribe 1.2")
        root = ET.fromstring(xml)
        assert root.get("origin") == "Scribe 1.2"

    def test_empty_codebook_is_valid(self) -> None:
        # Empty <Codes/> is a legal Codebook 1.0 document.
        xml = render_refi_qda_codebook_xml([])
        root = ET.fromstring(xml)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 0

    def test_codes_emit_through(self) -> None:
        c = _code(name="Pacing")
        xml = render_refi_qda_codebook_xml([c], project=_project())
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_NS}}}Codes/{{{REFI_QDA_NS}}}Code"
        )
        assert code_el is not None
        assert code_el.get("name") == "Pacing"


class TestSlugifyRefiQdaCodebookXmlFilename:
    """Same shape as ``slugify_codebook_filename`` but with the
    F6.5-specific extension."""

    def test_no_project_falls_back_to_codebook(self) -> None:
        assert (
            slugify_refi_qda_codebook_xml_filename(None)
            == "codebook.refi-qda.xml"
        )

    def test_simple_name(self) -> None:
        p = _project(name="Pilot")
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "pilot-codebook.refi-qda.xml"
        )

    def test_spaces_become_dashes(self) -> None:
        p = _project(name="Living with chronic illness")
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "living-with-chronic-illness-codebook.refi-qda.xml"
        )

    def test_diacritics_downgraded(self) -> None:
        p = _project(name="Café études")
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "cafe-etudes-codebook.refi-qda.xml"
        )

    def test_non_ascii_only_falls_back(self) -> None:
        p = _project(name="日本語")
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "codebook.refi-qda.xml"
        )

    def test_blank_name_falls_back(self) -> None:
        p = _project(name="Pilot")
        p.name = "   "
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "codebook.refi-qda.xml"
        )

    def test_long_name_truncated(self) -> None:
        p = _project(name="x" * 150)
        out = slugify_refi_qda_codebook_xml_filename(p)
        assert out.endswith("-codebook.refi-qda.xml")
        slug = out.removesuffix("-codebook.refi-qda.xml")
        assert len(slug) <= 80

    def test_punctuation_stripped(self) -> None:
        p = _project(name="Pilot — Phase 1!")
        assert (
            slugify_refi_qda_codebook_xml_filename(p)
            == "pilot-phase-1-codebook.refi-qda.xml"
        )


class TestWriteRefiQdaCodebookXml:
    """Atomic disk-write semantics, mirror of ``write_codebook``."""

    def test_writes_xml_to_disk(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "out.refi-qda.xml"
        result = write_refi_qda_codebook_xml(out, [c])
        assert result == out
        assert out.exists()
        body = out.read_bytes().decode("utf-8")
        # Round-trip: file body matches what the renderer produces.
        assert body == render_refi_qda_codebook_xml([c])

    def test_includes_project_metadata_comment(self, tmp_path: Path) -> None:
        p = _project(methodology="charmaz")
        out = tmp_path / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [], project=p)
        text = out.read_text(encoding="utf-8")
        assert "<!--" in text
        assert "Methodology: charmaz" in text

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        c = _code()
        out = tmp_path / "nested" / "deep" / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [c])
        assert out.exists()

    def test_atomic_swap_does_not_leak_tmp(self, tmp_path: Path) -> None:
        out = tmp_path / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [])
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_overwrite_replaces_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "out.refi-qda.xml"
        out.write_text("garbage", encoding="utf-8")
        c = _code()
        write_refi_qda_codebook_xml(out, [c])
        body = out.read_bytes().decode("utf-8")
        assert body.startswith("<?xml")
        assert "garbage" not in body

    def test_unicode_round_trip(self, tmp_path: Path) -> None:
        c = _code(name="Días buenos", definition="Buenos días.")
        p = _project(name="Estudio piloto")
        out = tmp_path / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [c], project=p)
        text = out.read_text(encoding="utf-8")
        assert "Días buenos" in text
        assert "Estudio piloto" in text

    def test_origin_propagates_to_disk(self, tmp_path: Path) -> None:
        out = tmp_path / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [], origin="Scribe 1.2")
        text = out.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        assert root.get("origin") == "Scribe 1.2"

    def test_empty_codebook_writes_minimal_xml(self, tmp_path: Path) -> None:
        out = tmp_path / "out.refi-qda.xml"
        write_refi_qda_codebook_xml(out, [])
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 0
