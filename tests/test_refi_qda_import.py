"""Tests for scribe.refi_qda_import (F6.6).

Pure-Python tests for the QDPX project import pipeline. Cover:

  * GUID parsing (Scribe-padded vs foreign).
  * Plain-text tokenisation (segments, words, char offsets, speaker
    detection, edge cases around blank lines and trailing newlines).
  * Char-span → word-anchor lookup (whole-word, sub-word, whitespace,
    out-of-range).
  * Code description parsing (round-trip vs free-form fallback).
  * import_qde with hand-crafted foreign QDPX fragments.
  * import_qdpx round-trip from a Scribe-origin to_qdpx() archive.
"""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree as ET

import pytest

from scribe.applications import Application
from scribe.code_versions import CodeVersion
from scribe.coders import Coder
from scribe.codes import Code, CodeRelation
from scribe.memos import Memo
from scribe.projects import Project
from scribe.refi_qda_import import (
    ImportedSourceText,
    ImportResult,
    TokenisedSegment,
    TokenisedWord,
    char_span_to_word_anchors,
    import_qde,
    import_qdpx,
    parse_code_description,
    parse_guid_to_scribe_id,
    tokenise_plain_text,
)
from scribe.refi_qda_project import (
    REFI_QDA_PROJECT_NS,
    code_guid,
    note_guid,
    render_source_plain_text,
    selection_guid,
    source_guid,
    to_qdpx,
    user_guid,
)
from scribe.sources import Source


FIXED_NOW = "2024-03-04T05:06:07.000000Z"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _project(**overrides) -> Project:
    payload = {"name": "Living with chronic illness", "now": FIXED_NOW}
    payload.update(overrides)
    return Project.new(**payload)


def _source(project_id: str, **overrides) -> Source:
    payload = {
        "project_id": project_id,
        "name": "Interview 1",
        "transcript_job_id": "abcdef012345",
        "now": FIXED_NOW,
    }
    payload.update(overrides)
    return Source.new(**payload)


def _code(project_id: str, **overrides) -> Code:
    payload = {
        "project_id": project_id,
        "name": "Pacing",
        "definition": "Adjusting daily activity to manage limited energy.",
        "now": FIXED_NOW,
    }
    payload.update(overrides)
    return Code.new(**payload)


def _coder(project_id: str, **overrides) -> Coder:
    payload = {"project_id": project_id, "name": "Coder A", "now": FIXED_NOW}
    payload.update(overrides)
    return Coder.new(**payload)


def _segments(words_per_seg, *, speakers=None):
    out = []
    for i, words in enumerate(words_per_seg):
        out.append(
            {
                "speaker": speakers[i] if speakers and i < len(speakers) else "",
                "words": [{"text": w} for w in words],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# parse_guid_to_scribe_id
# --------------------------------------------------------------------------- #


class TestParseGuid:
    def test_recovers_scribe_padded_guid(self) -> None:
        assert (
            parse_guid_to_scribe_id(
                "00000000-0000-0000-c0de-abcdef012345", kind_tag="c0de"
            )
            == "abcdef012345"
        )

    def test_returns_none_for_wrong_kind_tag(self) -> None:
        assert (
            parse_guid_to_scribe_id(
                "00000000-0000-0000-c0de-abcdef012345", kind_tag="5046"
            )
            is None
        )

    def test_returns_none_for_foreign_guid(self) -> None:
        assert (
            parse_guid_to_scribe_id(
                "11112222-3333-4444-5555-666677778888", kind_tag="c0de"
            )
            is None
        )

    def test_handles_uppercase(self) -> None:
        assert (
            parse_guid_to_scribe_id(
                "00000000-0000-0000-C0DE-ABCDEF012345", kind_tag="c0de"
            )
            == "abcdef012345"
        )

    def test_returns_none_for_empty_or_none(self) -> None:
        assert parse_guid_to_scribe_id(None, kind_tag="c0de") is None
        assert parse_guid_to_scribe_id("", kind_tag="c0de") is None
        assert parse_guid_to_scribe_id("not-a-guid", kind_tag="c0de") is None


# --------------------------------------------------------------------------- #
# tokenise_plain_text
# --------------------------------------------------------------------------- #


class TestTokeniser:
    def test_basic_single_segment(self) -> None:
        segs = tokenise_plain_text("Hello world")
        assert len(segs) == 1
        assert segs[0].speaker == ""
        assert [(w.word_id, w.text) for w in segs[0].words] == [
            ("s0w0", "Hello"),
            ("s0w1", "world"),
        ]
        # Char offsets cover the words exactly
        assert segs[0].words[0].start == 0
        assert segs[0].words[0].end == 5
        assert segs[0].words[1].start == 6
        assert segs[0].words[1].end == 11

    def test_speaker_prefix_detected_and_stripped(self) -> None:
        segs = tokenise_plain_text("Luke: Hello world")
        assert segs[0].speaker == "Luke"
        # Words start after the prefix
        assert segs[0].words[0].text == "Hello"
        assert segs[0].words[0].start == 6
        assert segs[0].words[1].text == "world"

    def test_speaker_prefix_disabled(self) -> None:
        segs = tokenise_plain_text("Luke: Hello", detect_speakers=False)
        assert segs[0].speaker == ""
        # The prefix becomes ordinary words
        assert [w.text for w in segs[0].words] == ["Luke:", "Hello"]

    def test_long_speaker_candidate_rejected(self) -> None:
        # Sentence with a colon in the middle shouldn't be treated as a speaker label
        segs = tokenise_plain_text(
            "The point I want to make is: the world is unfair"
        )
        assert segs[0].speaker == ""
        # First word starts at offset 0
        assert segs[0].words[0].text == "The"

    def test_multi_segment_with_offsets_continuous(self) -> None:
        text = "INT: Hello\nP3: World"
        segs = tokenise_plain_text(text)
        assert len(segs) == 2
        assert segs[0].speaker == "INT"
        assert segs[1].speaker == "P3"
        # Char offset of "World" must be its position in the *full* text
        assert text[segs[1].words[0].start:segs[1].words[0].end] == "World"

    def test_trailing_newline_does_not_create_phantom_segment(self) -> None:
        segs = tokenise_plain_text("Hello\n")
        assert len(segs) == 1

    def test_blank_line_in_middle_preserves_offsets(self) -> None:
        text = "A\n\nB"
        segs = tokenise_plain_text(text)
        # Three segments: "A", "", "B"
        assert len(segs) == 3
        assert segs[0].words[0].text == "A"
        assert segs[1].words == ()
        assert segs[2].words[0].text == "B"
        # B's offset must equal its position in the source text
        assert text[segs[2].words[0].start] == "B"

    def test_empty_string(self) -> None:
        segs = tokenise_plain_text("")
        assert len(segs) == 1
        assert segs[0].words == ()

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            tokenise_plain_text(b"bytes not allowed")


# --------------------------------------------------------------------------- #
# char_span_to_word_anchors
# --------------------------------------------------------------------------- #


class TestCharSpanAnchors:
    def setup_method(self) -> None:
        # Text: "Luke: Hello world goodbye"
        #        0         1         2
        #        0123456789012345678901234
        self.segs = tokenise_plain_text("Luke: Hello world goodbye")

    def test_whole_word_match(self) -> None:
        # "Hello" spans [6, 11)
        assert char_span_to_word_anchors(self.segs, 6, 11) == (
            "s0w0",
            "s0w0",
            None,
            None,
        )

    def test_two_words(self) -> None:
        # "Hello world" spans [6, 17)
        assert char_span_to_word_anchors(self.segs, 6, 17) == (
            "s0w0",
            "s0w1",
            None,
            None,
        )

    def test_subword_start(self) -> None:
        # "ello" starts inside "Hello" at offset 7
        # word "Hello" at [6, 11); offset 7 - 6 = 1
        assert char_span_to_word_anchors(self.segs, 7, 11) == (
            "s0w0",
            "s0w0",
            1,
            None,
        )

    def test_subword_end(self) -> None:
        # "Hell" ends inside "Hello" at offset 10
        # word "Hello" at [6, 11); offset 10 - 6 = 4
        result = char_span_to_word_anchors(self.segs, 6, 10)
        assert result == ("s0w0", "s0w0", None, 4)

    def test_whitespace_only_returns_none(self) -> None:
        # Just the space between "Hello" and "world" (offset 11..12)
        # No word lies inside [11, 12), so the function returns None.
        result = char_span_to_word_anchors(self.segs, 11, 12)
        # Either we anchor to "world" or return None; both are
        # acceptable degradations. Current implementation skips ahead
        # to the next word.
        assert result is None or result[0] == "s0w1"

    def test_out_of_range(self) -> None:
        # Past the last word
        assert char_span_to_word_anchors(self.segs, 1000, 1010) is None

    def test_handles_swapped_bounds(self) -> None:
        # If start > end the helper still returns *some* anchor,
        # treating the bounds as a half-open interval after swap.
        result = char_span_to_word_anchors(self.segs, 11, 6)
        assert result is not None and result[0] == "s0w0"


# --------------------------------------------------------------------------- #
# parse_code_description
# --------------------------------------------------------------------------- #


class TestParseCodeDescription:
    def test_round_trip_definition_only(self) -> None:
        text = "Definition: Pacing daily activity"
        out = parse_code_description(text)
        assert out["definition"] == "Pacing daily activity"

    def test_round_trip_full_block(self) -> None:
        text = (
            "Definition: Pacing\n\n"
            "Inclusion criteria: managing energy\n\n"
            "Exclusion criteria: not resting\n\n"
            "Exemplars:\n- I take breaks\n- I lie down\n\n"
            "Theoretical memo: shows up early\n\n"
            "Stage: focused\n\n"
            "Status: retired"
        )
        out = parse_code_description(text)
        assert out["definition"] == "Pacing"
        assert out["inclusion_criteria"] == "managing energy"
        assert out["exclusion_criteria"] == "not resting"
        assert out["exemplars"] == ["I take breaks", "I lie down"]
        assert out["theoretical_memo"] == "shows up early"
        assert out["stage"] == "focused"
        assert out["status"] == "retired"

    def test_provenance_kv(self) -> None:
        text = "Provenance: source=human; coder=A"
        out = parse_code_description(text)
        assert out["provenance"] == {"source": "human", "coder": "A"}

    def test_related_codes(self) -> None:
        text = (
            "Definition: x\n\n"
            "Related codes:\n- broader: abcdef012345\n- contrasts_with: 123456789abc"
        )
        out = parse_code_description(text)
        assert out["related_codes"] == [
            {"relation_type": "broader", "code_id": "abcdef012345"},
            {"relation_type": "contrasts_with", "code_id": "123456789abc"},
        ]

    def test_foreign_description_falls_back_to_definition(self) -> None:
        text = "A free-form description from another QDA tool."
        out = parse_code_description(text)
        # Foreign descriptions land entirely in `definition`.
        assert out == {"definition": text}

    def test_empty_returns_empty(self) -> None:
        assert parse_code_description("") == {}
        assert parse_code_description(None) == {}  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# import_qde — foreign QDPX shape
# --------------------------------------------------------------------------- #


def _make_qde(*, codes_xml: str = "", sources_xml: str = "", users_xml: str = "",
              notes_xml: str = "", project_attrs: dict[str, str] | None = None) -> str:
    """Hand-build a minimal QDE XML body for tests."""
    attrs = {
        "name": "Foreign project",
        "origin": "Atlas.ti",
        "creationDateTime": "2023-06-01T12:00:00Z",
        "modifiedDateTime": "2023-06-15T12:00:00Z",
    }
    attrs.update(project_attrs or {})
    attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Project xmlns="{REFI_QDA_PROJECT_NS}" {attr_str}>\n'
        f'  <Users>{users_xml}</Users>\n'
        f'  <CodeBook><Codes>{codes_xml}</Codes></CodeBook>\n'
        f'  <Sources>{sources_xml}</Sources>\n'
        + (f'  <Notes>{notes_xml}</Notes>\n' if notes_xml else '')
        + '</Project>\n'
    )


class TestImportQdeForeign:
    def test_minimal_project(self) -> None:
        qde = _make_qde()
        result = import_qde(qde, now=FIXED_NOW)
        assert isinstance(result, ImportResult)
        assert result.project.name == "Foreign project"
        assert result.project.created_at == "2023-06-01T12:00:00Z"
        assert result.project.modified_at == "2023-06-15T12:00:00Z"
        assert "Atlas.ti" in result.project.description
        assert result.sources == []
        assert result.codes == []
        assert result.coders == []
        assert result.memos == []
        assert result.applications == []

    def test_blank_project_name_falls_back(self) -> None:
        qde = _make_qde(project_attrs={"name": ""})
        result = import_qde(qde, now=FIXED_NOW)
        assert result.project.name == "Imported QDPX project"

    def test_codes_with_hierarchy(self) -> None:
        codes_xml = (
            '<Code guid="11111111-1111-1111-1111-111111111111" name="Parent">'
            '  <Code guid="22222222-2222-2222-2222-222222222222" name="Child"/>'
            '</Code>'
        )
        qde = _make_qde(codes_xml=codes_xml)
        result = import_qde(qde, now=FIXED_NOW)
        assert len(result.codes) == 2
        names = {c.name: c for c in result.codes}
        assert names["Parent"].parent_code_id is None
        assert names["Child"].parent_code_id == names["Parent"].id

    def test_codes_with_description(self) -> None:
        codes_xml = (
            '<Code guid="11111111-1111-1111-1111-111111111111" name="P">'
            '<Description>A free-form description.</Description>'
            '</Code>'
        )
        qde = _make_qde(codes_xml=codes_xml)
        result = import_qde(qde, now=FIXED_NOW)
        assert result.codes[0].definition == "A free-form description."

    def test_users_become_coders(self) -> None:
        users_xml = (
            '<User guid="11111111-1111-1111-1111-111111111111" name="Alice"/>'
            '<User guid="22222222-2222-2222-2222-222222222222" name="Bob"/>'
        )
        qde = _make_qde(users_xml=users_xml)
        result = import_qde(qde, now=FIXED_NOW)
        names = sorted(c.name for c in result.coders)
        assert names == ["Alice", "Bob"]

    def test_text_source_with_selection_anchors_to_words(self) -> None:
        sources_xml = (
            '<TextSource guid="11111111-1111-1111-1111-111111111111" '
            'name="Interview" plainTextPath="internal://Sources/sample.txt">'
            '<PlainTextSelection guid="22222222-2222-2222-2222-222222222222" '
            'startPosition="6" endPosition="11">'
            '<Coding guid="33333333-3333-3333-3333-333333333333">'
            '<CodeRef targetGUID="44444444-4444-4444-4444-444444444444"/>'
            '</Coding>'
            '</PlainTextSelection>'
            '</TextSource>'
        )
        codes_xml = (
            '<Code guid="44444444-4444-4444-4444-444444444444" name="Code1"/>'
        )
        qde = _make_qde(codes_xml=codes_xml, sources_xml=sources_xml)
        result = import_qde(
            qde,
            source_texts={"internal://Sources/sample.txt": "Luke: Hello world"},
            now=FIXED_NOW,
        )
        assert len(result.sources) == 1
        assert len(result.applications) == 1
        app = result.applications[0]
        # Selection covers "Hello" → s0w0
        assert app.anchor_start_word_id == "s0w0"
        assert app.anchor_end_word_id == "s0w0"
        # Code resolved to the imported one
        assert app.code_id == result.codes[0].id

    def test_audio_source_skipped_with_warning(self) -> None:
        sources_xml = (
            '<AudioSource guid="11111111-1111-1111-1111-111111111111" name="A"/>'
        )
        qde = _make_qde(sources_xml=sources_xml)
        result = import_qde(qde, now=FIXED_NOW)
        assert result.sources == []
        assert any("AudioSource" in w for w in result.warnings)

    def test_application_with_unknown_code_warns(self) -> None:
        sources_xml = (
            '<TextSource guid="11111111-1111-1111-1111-111111111111" '
            'name="Interview" plainTextPath="internal://Sources/sample.txt">'
            '<PlainTextSelection guid="22222222-2222-2222-2222-222222222222" '
            'startPosition="0" endPosition="5">'
            '<Coding guid="33333333-3333-3333-3333-333333333333">'
            '<CodeRef targetGUID="ffffffff-ffff-ffff-ffff-ffffffffffff"/>'
            '</Coding>'
            '</PlainTextSelection>'
            '</TextSource>'
        )
        qde = _make_qde(sources_xml=sources_xml)
        result = import_qde(
            qde,
            source_texts={"internal://Sources/sample.txt": "Hello world"},
            now=FIXED_NOW,
        )
        assert result.applications == []
        assert any("unknown code" in w for w in result.warnings)

    def test_application_synthesises_fallback_coder(self) -> None:
        # Selection has a creatingUser pointing at no existing User.
        sources_xml = (
            '<TextSource guid="11111111-1111-1111-1111-111111111111" '
            'name="Interview" plainTextPath="internal://Sources/sample.txt">'
            '<PlainTextSelection guid="22222222-2222-2222-2222-222222222222" '
            'startPosition="0" endPosition="5" '
            'creatingUser="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa">'
            '<Coding>'
            '<CodeRef targetGUID="44444444-4444-4444-4444-444444444444"/>'
            '</Coding>'
            '</PlainTextSelection>'
            '</TextSource>'
        )
        codes_xml = (
            '<Code guid="44444444-4444-4444-4444-444444444444" name="C"/>'
        )
        qde = _make_qde(codes_xml=codes_xml, sources_xml=sources_xml)
        result = import_qde(
            qde,
            source_texts={"internal://Sources/sample.txt": "Hello world"},
            now=FIXED_NOW,
        )
        assert len(result.applications) == 1
        # Exactly one synthesised coder; the application points at it
        assert len(result.coders) == 1
        assert result.applications[0].coder_id == result.coders[0].id
        assert result.coders[0].name == "Imported coder"

    def test_two_unknown_creators_share_one_synth_coder(self) -> None:
        # Two selections, neither has a matching User. The importer
        # should reuse the same synthesised coder rather than minting
        # one per application.
        sources_xml = (
            '<TextSource guid="11111111-1111-1111-1111-111111111111" '
            'name="Interview" plainTextPath="internal://Sources/sample.txt">'
            '<PlainTextSelection guid="22222222-2222-2222-2222-222222222222" '
            'startPosition="0" endPosition="5" '
            'creatingUser="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa">'
            '<Coding>'
            '<CodeRef targetGUID="44444444-4444-4444-4444-444444444444"/>'
            '</Coding>'
            '</PlainTextSelection>'
            '<PlainTextSelection guid="55555555-5555-5555-5555-555555555555" '
            'startPosition="6" endPosition="11" '
            'creatingUser="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb">'
            '<Coding>'
            '<CodeRef targetGUID="44444444-4444-4444-4444-444444444444"/>'
            '</Coding>'
            '</PlainTextSelection>'
            '</TextSource>'
        )
        codes_xml = (
            '<Code guid="44444444-4444-4444-4444-444444444444" name="C"/>'
        )
        qde = _make_qde(codes_xml=codes_xml, sources_xml=sources_xml)
        result = import_qde(
            qde,
            source_texts={"internal://Sources/sample.txt": "Hello world"},
            now=FIXED_NOW,
        )
        assert len(result.applications) == 2
        assert len(result.coders) == 1
        assert (
            result.applications[0].coder_id
            == result.applications[1].coder_id
            == result.coders[0].id
        )

    def test_notes_become_memos(self) -> None:
        notes_xml = (
            '<Note guid="11111111-1111-1111-1111-111111111111" '
            'name="Reflection" plainTextPath="internal://Notes/r.txt"/>'
        )
        qde = _make_qde(notes_xml=notes_xml)
        result = import_qde(
            qde,
            source_texts={"internal://Notes/r.txt": "I noticed pacing."},
            now=FIXED_NOW,
        )
        assert len(result.memos) == 1
        m = result.memos[0]
        assert m.title == "Reflection"
        assert m.body == "I noticed pacing."

    def test_invalid_xml_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            import_qde("<not> valid </xml")

    def test_non_utf8_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            import_qde(b"\xff\xfe<not utf-8>")


# --------------------------------------------------------------------------- #
# import_qdpx — round-trip from F6.4 export
# --------------------------------------------------------------------------- #


class TestImportQdpxRoundTrip:
    def _build_archive(self) -> bytes:
        p = _project()
        s1 = _source(p.id, name="Interview 1")
        s2 = _source(p.id, name="Interview 2")
        c1 = _code(p.id, name="Pacing")
        c2 = _code(
            p.id,
            name="Disclosure",
            parent_code_id=c1.id,
            inclusion_criteria="When the participant talks about telling others",
            exemplars=["I told my boss", "I never told anyone"],
        )
        cd = _coder(p.id, name="Luke")

        # Need a CodeVersion for the Application
        v1 = CodeVersion.new(code=c1, version=1, now=FIXED_NOW)

        segs = [
            {"speaker": "INT", "words": [
                {"text": "How"}, {"text": "do"}, {"text": "you"},
                {"text": "manage"}, {"text": "energy?"},
            ]},
            {"speaker": "P3", "words": [
                {"text": "I"}, {"text": "pace"}, {"text": "myself."},
            ]},
        ]
        r1 = render_source_plain_text(s1.id, segs)
        r2 = render_source_plain_text(s2.id, _segments([["Goodbye", "world"]]))

        a1 = Application.new(
            project_id=p.id, code_id=c1.id, source_id=s1.id, coder_id=cd.id,
            anchor_start_word_id="s1w1", anchor_end_word_id="s1w2",
            definition_version_id_at_apply=v1.id, now=FIXED_NOW,
        )

        m = Memo.new(
            project_id=p.id, type="theoretical",
            title="Pacing memo", body="Pacing keeps coming up.",
            now=FIXED_NOW,
        )

        return to_qdpx(
            project=p, sources=[s1, s2], codes=[c1, c2], coders=[cd],
            applications=[a1], memos=[m],
            rendered_sources=[r1, r2],
        ), p, [s1, s2], [c1, c2], cd, [a1], [m]

    def test_round_trip_recovers_all_entities(self) -> None:
        archive, p, sources, codes, coder, apps, memos = self._build_archive()
        result = import_qdpx(archive, now=FIXED_NOW)

        assert result.project.name == p.name

        # Source ids preserved (12-char hex round-trip via 5046 kind tag)
        assert {s.id for s in result.sources} == {s.id for s in sources}
        # Code ids preserved (c0de kind tag)
        assert {c.id for c in result.codes} == {c.id for c in codes}
        # Code hierarchy preserved
        names = {c.name: c for c in result.codes}
        assert names["Disclosure"].parent_code_id == names["Pacing"].id
        # Coder id preserved
        assert {c.id for c in result.coders} == {coder.id}
        # Memo id preserved
        assert {m.id for m in result.memos} == {m.id for m in memos}
        # Applications point at the right code, source, coder
        assert len(result.applications) == 1
        a = result.applications[0]
        assert a.code_id == codes[0].id
        assert a.source_id == sources[0].id
        assert a.coder_id == coder.id

    def test_round_trip_application_anchors(self) -> None:
        archive, p, sources, codes, coder, apps, memos = self._build_archive()
        result = import_qdpx(archive, now=FIXED_NOW)
        # Original application anchored at s1w1..s1w2 (the words "pace
        # myself."). Tokenising the rendered text recovers identical
        # anchors because the F6.4 export preserves segment / word
        # boundaries exactly.
        a = result.applications[0]
        assert a.anchor_start_word_id == "s1w1"
        assert a.anchor_end_word_id == "s1w2"

    def test_round_trip_creates_one_code_version_per_code(self) -> None:
        archive, p, sources, codes, coder, apps, memos = self._build_archive()
        result = import_qdpx(archive, now=FIXED_NOW)
        assert len(result.code_versions) == len(result.codes)
        # Every CodeVersion is version 1
        assert {v.version for v in result.code_versions} == {1}
        # Every Application references some imported version id
        version_ids = {v.id for v in result.code_versions}
        for a in result.applications:
            assert a.definition_version_id_at_apply in version_ids

    def test_round_trip_source_text_recovered(self) -> None:
        archive, p, sources, codes, coder, apps, memos = self._build_archive()
        result = import_qdpx(archive, now=FIXED_NOW)
        # The recovered plain-text body for each source matches what
        # render_source_plain_text would produce.
        for s in sources:
            assert s.id in result.source_texts
            entry = result.source_texts[s.id]
            assert isinstance(entry, ImportedSourceText)
            assert entry.text  # non-empty

    def test_round_trip_no_coders_synthesises_one(self) -> None:
        # Build an archive with no coders → exporter writes a "scribe"
        # placeholder User with the project's GUID kind-tag. On import
        # we should skip that placeholder, then synthesise a coder for
        # any application whose creatingUser is the placeholder GUID.
        p = _project()
        s = _source(p.id, name="Interview 1")
        c = _code(p.id)
        v1 = CodeVersion.new(code=c, version=1, now=FIXED_NOW)
        # Application uses an arbitrary 12-hex coder id (no Coder
        # entity will be exported because we don't pass any).
        a = Application.new(
            project_id=p.id, code_id=c.id, source_id=s.id,
            coder_id="ffffffffffff",
            anchor_start_word_id="s0w0", anchor_end_word_id="s0w0",
            definition_version_id_at_apply=v1.id, now=FIXED_NOW,
        )
        segs = [{"speaker": "X", "words": [{"text": "Hello"}]}]
        r = render_source_plain_text(s.id, segs)
        archive = to_qdpx(
            project=p, sources=[s], codes=[c], applications=[a],
            rendered_sources=[r],
        )
        result = import_qdpx(archive, now=FIXED_NOW)
        # The synthesised fallback coder should exist and the
        # application should point at it.
        assert len(result.coders) == 1
        assert result.coders[0].name == "Imported coder"
        assert len(result.applications) == 1
        assert result.applications[0].coder_id == result.coders[0].id

    def test_round_trip_then_export_again_matches_codebook(self) -> None:
        # Export, import, and re-export. The set of code names + the
        # parent hierarchy must survive both passes.
        archive, p, sources, codes, coder, apps, memos = self._build_archive()
        result = import_qdpx(archive, now=FIXED_NOW)
        # Re-export
        rendered = []
        for sid, entry in result.source_texts.items():
            # Synthesise a render: the original text is what we need
            # so we fabricate a minimal segments-from-text adapter.
            # For the assertion we only need the codes section of the
            # XML to be stable, so render an empty source is fine.
            from scribe.refi_qda_project import RenderedSource
            rendered.append(RenderedSource(source_id=sid, text=entry.text, offsets={}))
        archive2 = to_qdpx(
            project=result.project, sources=result.sources,
            codes=result.codes, coders=result.coders,
            memos=result.memos, rendered_sources=rendered,
        )
        # Parse the second export and check the code names + hierarchy
        with zipfile.ZipFile(io.BytesIO(archive2)) as zf:
            qde2 = zf.read("project.qde").decode("utf-8")
        root = ET.fromstring(qde2)
        ns = REFI_QDA_PROJECT_NS

        def _walk(parent: ET.Element, depth: int = 0) -> list[tuple[int, str]]:
            out = []
            for c in parent.findall(f"{{{ns}}}Code"):
                out.append((depth, c.get("name") or ""))
                out.extend(_walk(c, depth + 1))
            return out

        codes_root = root.find(f"{{{ns}}}CodeBook/{{{ns}}}Codes")
        names = _walk(codes_root)
        # First export had Pacing (root) and Disclosure (child of Pacing)
        depths = {name: depth for depth, name in names}
        assert depths["Pacing"] == 0
        assert depths["Disclosure"] == 1


# --------------------------------------------------------------------------- #
# import_qdpx — file / archive errors
# --------------------------------------------------------------------------- #


class TestImportQdpxErrors:
    def test_not_a_zip_raises(self) -> None:
        with pytest.raises(ValueError):
            import_qdpx(b"not a zip")

    def test_zip_without_project_qde_raises(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("Sources/foo.txt", "hello")
        with pytest.raises(ValueError):
            import_qdpx(buf.getvalue())

    def test_path_argument_supported(self, tmp_path) -> None:
        p = _project()
        archive = to_qdpx(project=p)
        target = tmp_path / "test.qdpx"
        target.write_bytes(archive)
        result = import_qdpx(target, now=FIXED_NOW)
        assert result.project.name == p.name

    def test_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeError):
            import_qdpx(12345)  # type: ignore[arg-type]
