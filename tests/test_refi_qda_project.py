"""Tests for scribe.refi_qda_project (F6.4).

Pure-Python tests for the QDPX project export pipeline. Cover:

  * GUID padding scheme (round-trip via 4-char kind tag).
  * Plain-text source rendering (segments → flat string + word
    offsets), including speaker labels and skipped whitespace tokens.
  * Application offset computation (word-id anchors → char offsets,
    sub-word offset honouring, orphan handling).
  * project.qde XML shape (namespace, root attributes, Users,
    CodeBook with hierarchy + description, Sources with
    PlainTextSelection/Coding chains, Notes).
  * QDPX zip archive layout (project.qde + Sources/ + Notes/).
  * Filename slugging + Unicode handling.
"""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree as ET

import pytest

from scribe.applications import Application, new_application_id
from scribe.coders import Coder
from scribe.codes import Code
from scribe.memos import Memo
from scribe.projects import Project
from scribe.refi_qda_project import (
    REFI_QDA_PROJECT_NS,
    REFI_QDA_PROJECT_ORIGIN_DEFAULT,
    RenderedSource,
    WordOffset,
    application_plain_text_offsets,
    code_guid,
    note_guid,
    project_guid,
    render_source_plain_text,
    scribe_id_to_guid,
    selection_guid,
    slugify_qdpx_filename,
    source_guid,
    to_qde_xml,
    to_qdpx,
    user_guid,
)
from scribe.sources import Source


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _project(**overrides) -> Project:
    payload = {
        "name": "Living with chronic illness",
        "methodology": "charmaz",
        "now": "2024-03-04T05:06:07.000000Z",
    }
    payload.update(overrides)
    return Project.new(**payload)


def _source(project_id: str, **overrides) -> Source:
    payload = {
        "project_id": project_id,
        "name": "Interview 1",
        "source_type": "transcript",
        "transcript_job_id": "abcdef012345",
        "now": "2024-03-04T05:06:07.000000Z",
    }
    payload.update(overrides)
    return Source.new(**payload)


def _code(project_id: str, **overrides) -> Code:
    payload = {
        "project_id": project_id,
        "name": "Pacing",
        "definition": "Adjusting daily activity to manage limited energy.",
        "now": "2024-03-04T05:06:07.000000Z",
    }
    payload.update(overrides)
    return Code.new(**payload)


def _coder(project_id: str, **overrides) -> Coder:
    payload = {
        "project_id": project_id,
        "name": "Coder A",
        "now": "2024-03-04T05:06:07.000000Z",
    }
    payload.update(overrides)
    return Coder.new(**payload)


def _memo(project_id: str, **overrides) -> Memo:
    payload = {
        "project_id": project_id,
        "type": "theoretical",
        "title": "Why pacing?",
        "body": "Hypothesis: pacing emerges as a mode of self-management.",
        "now": "2024-03-04T05:06:07.000000Z",
    }
    payload.update(overrides)
    return Memo.new(**payload)


def _app(
    project_id: str,
    *,
    code_id: str,
    source_id: str,
    coder_id: str,
    start_word_id: str = "s0w0",
    end_word_id: str = "s0w0",
    start_char_offset: int | None = None,
    end_char_offset: int | None = None,
    application_id: str | None = None,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start_word_id,
        anchor_end_word_id=end_word_id,
        definition_version_id_at_apply="aaaabbbbcccc",
        start_char_offset=start_char_offset,
        end_char_offset=end_char_offset,
        application_id=application_id,
        now="2024-03-04T05:06:07.000000Z",
    )


def _segments(words_per_segment, speakers=None):
    """Build a Scribe-shaped segments list for tests.

    ``words_per_segment`` is a list of lists of word strings.
    ``speakers`` (optional) is a parallel list of speaker labels.
    """
    if speakers is None:
        speakers = [""] * len(words_per_segment)
    out = []
    for i, words in enumerate(words_per_segment):
        out.append({
            "speaker": speakers[i],
            "words": [{"text": w} for w in words],
        })
    return out


# --------------------------------------------------------------------------- #
# GUID mapping
# --------------------------------------------------------------------------- #


class TestScribeIdToGuid:
    def test_basic_shape(self) -> None:
        guid = scribe_id_to_guid("abcdef012345", kind_tag="c0de")
        assert guid == "00000000-0000-0000-c0de-abcdef012345"

    def test_kind_tag_distinguishes_entity_kinds(self) -> None:
        sid = "abcdef012345"
        # Same scribe id, two different kinds → two different guids.
        assert code_guid(sid) != source_guid(sid)
        assert source_guid(sid) != user_guid(sid)
        assert user_guid(sid) != note_guid(sid)
        assert note_guid(sid) != code_guid(sid)

    def test_invalid_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            scribe_id_to_guid("not-hex", kind_tag="c0de")
        with pytest.raises(ValueError):
            scribe_id_to_guid("abc", kind_tag="c0de")  # too short

    def test_invalid_kind_tag_rejected(self) -> None:
        with pytest.raises(ValueError):
            scribe_id_to_guid("abcdef012345", kind_tag="zzzz")  # not hex

    def test_guid_is_lowercase(self) -> None:
        guid = code_guid("ABCDEF012345")
        assert guid == "00000000-0000-0000-c0de-abcdef012345"

    def test_project_guid(self) -> None:
        assert project_guid("abcdef012345").endswith("abcdef012345")

    def test_selection_and_coding_guids_differ_per_application(self) -> None:
        # An application carries two GUIDs in the QDPX layout: one for
        # the PlainTextSelection wrapper, one for the Coding child.
        # Different so importers don't merge them.
        aid = "abcdef012345"
        assert selection_guid(aid) != code_guid(aid)


# --------------------------------------------------------------------------- #
# Plain-text rendering
# --------------------------------------------------------------------------- #


class TestRenderSourcePlainText:
    def test_single_segment_no_speaker(self) -> None:
        segs = _segments([["Hello", "world"]])
        r = render_source_plain_text("abcdef012345", segs)
        assert r.text == "Hello world"
        assert r.offsets["s0w0"] == WordOffset("s0w0", 0, 5)
        assert r.offsets["s0w1"] == WordOffset("s0w1", 6, 11)

    def test_speaker_prefix(self) -> None:
        segs = _segments([["Hi"]], speakers=["LUKE"])
        r = render_source_plain_text("abcdef012345", segs)
        assert r.text == "LUKE: Hi"
        assert r.offsets["s0w0"] == WordOffset("s0w0", 6, 8)

    def test_multiple_segments_separated_by_newline(self) -> None:
        segs = _segments(
            [["Hello"], ["World"]],
            speakers=["A", "B"],
        )
        r = render_source_plain_text("abcdef012345", segs)
        assert r.text == "A: Hello\nB: World"
        assert r.offsets["s0w0"] == WordOffset("s0w0", 3, 8)
        assert r.offsets["s1w0"] == WordOffset("s1w0", 12, 17)

    def test_empty_segments_skipped(self) -> None:
        segs = _segments([["Hello"], [], ["Bye"]])
        r = render_source_plain_text("abcdef012345", segs)
        assert r.text == "Hello\n\nBye"
        # Empty middle segment contributes a newline (segment break)
        # but no words, so no offsets.
        assert "s1w0" not in r.offsets

    def test_whitespace_only_words_skipped(self) -> None:
        segs = _segments([["Hello", "  ", "world"]])
        r = render_source_plain_text("abcdef012345", segs)
        # The whitespace word contributes no characters and no offset.
        assert "s0w1" not in r.offsets
        assert "s0w0" in r.offsets and "s0w2" in r.offsets

    def test_speaker_disabled(self) -> None:
        segs = _segments([["Hi"]], speakers=["LUKE"])
        r = render_source_plain_text(
            "abcdef012345", segs, include_speaker_labels=False
        )
        assert r.text == "Hi"
        assert r.offsets["s0w0"] == WordOffset("s0w0", 0, 2)

    def test_unicode_in_words(self) -> None:
        segs = _segments([["Días", "buenos"]])
        r = render_source_plain_text("abcdef012345", segs)
        # Python str length counts characters, not bytes.
        assert r.text == "Días buenos"
        assert r.offsets["s0w0"].end == 4
        assert r.offsets["s0w1"].start == 5

    def test_handles_non_dict_words(self) -> None:
        # Garbage in segments should not crash; just be skipped.
        segs = [{"speaker": "", "words": [{"text": "hi"}, "garbage", {"text": "there"}]}]
        r = render_source_plain_text("abcdef012345", segs)
        # "garbage" is ignored; "hi there" with one space.
        assert r.text == "hi there"

    def test_no_segments(self) -> None:
        r = render_source_plain_text("abcdef012345", [])
        assert r.text == ""
        assert r.offsets == {}


# --------------------------------------------------------------------------- #
# Application offset conversion
# --------------------------------------------------------------------------- #


class TestApplicationPlainTextOffsets:
    def _setup(self):
        pid = "abcdef012345"
        cid = "111111111111"
        sid = "222222222222"
        cdid = "333333333333"
        segs = _segments([["Hello", "world", "today"]])
        rendered = render_source_plain_text(sid, segs)
        return pid, cid, sid, cdid, rendered

    def test_whole_word_anchor(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        a = _app(pid, code_id=cid, source_id=sid, coder_id=cdid,
                 start_word_id="s0w0", end_word_id="s0w0")
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets == (0, 5)

    def test_multi_word_anchor(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        a = _app(pid, code_id=cid, source_id=sid, coder_id=cdid,
                 start_word_id="s0w0", end_word_id="s0w1")
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets == (0, 11)
        assert rendered.text[offsets[0]:offsets[1]] == "Hello world"

    def test_sub_word_offsets_honoured(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        # "Hel" of "Hello" → start=0, end=3
        a = _app(pid, code_id=cid, source_id=sid, coder_id=cdid,
                 start_word_id="s0w0", end_word_id="s0w0",
                 start_char_offset=0, end_char_offset=3)
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets == (0, 3)
        assert rendered.text[offsets[0]:offsets[1]] == "Hel"

    def test_sub_word_offsets_clamped(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        # Offset wildly past the word's length is clamped, not invalid
        a = _app(pid, code_id=cid, source_id=sid, coder_id=cdid,
                 start_word_id="s0w0", end_word_id="s0w0",
                 end_char_offset=999)
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets is not None
        assert offsets[1] <= len(rendered.text)

    def test_orphan_anchor_returns_none(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        a = _app(pid, code_id=cid, source_id=sid, coder_id=cdid,
                 start_word_id="s9w9", end_word_id="s9w9")
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets is None

    def test_end_before_start_collapses_to_zero_width(self) -> None:
        pid, cid, sid, cdid, rendered = self._setup()
        # Construct an Application directly (bypassing Application.new's
        # validator) so we can test the offset helper's defensive
        # collapse behaviour. The validator forbids start >= end on a
        # single-word anchor at construction time, but if a hand-edited
        # JSON ever gets past us, the helper should not raise.
        a = Application(
            id="100000000001",
            project_id=pid,
            code_id=cid,
            source_id=sid,
            coder_id=cdid,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            definition_version_id_at_apply="aaaabbbbcccc",
            start_char_offset=4,
            end_char_offset=2,
            confidence=None,
            provenance={},
            note="",
            created_at="2024-03-04T05:06:07.000000Z",
            modified_at="2024-03-04T05:06:07.000000Z",
        )
        offsets = application_plain_text_offsets(a, rendered)
        assert offsets is not None
        assert offsets[0] == offsets[1]  # zero-width


# --------------------------------------------------------------------------- #
# Top-level project XML
# --------------------------------------------------------------------------- #


class TestToQdeXml:
    def test_root_namespace_and_attrs(self) -> None:
        p = _project()
        xml = to_qde_xml(project=p)
        root = ET.fromstring(xml)
        assert root.tag == f"{{{REFI_QDA_PROJECT_NS}}}Project"
        assert root.get("name") == p.name
        assert root.get("origin") == REFI_QDA_PROJECT_ORIGIN_DEFAULT
        assert root.get("creatingUserGUID")
        assert root.get("creationDateTime")
        assert root.get("modifiedDateTime")

    def test_creation_datetime_no_microseconds(self) -> None:
        p = _project(now="2024-03-04T05:06:07.123456Z")
        xml = to_qde_xml(project=p)
        root = ET.fromstring(xml)
        # REFI-QDA prefers no fractional seconds.
        assert "." not in root.get("creationDateTime", "")

    def test_xml_declaration_present(self) -> None:
        p = _project()
        xml = to_qde_xml(project=p)
        assert xml.startswith("<?xml")
        assert "utf-8" in xml.lower()

    def test_users_emit_one_per_coder(self) -> None:
        p = _project()
        c1 = _coder(p.id, name="Coder A")
        c2 = _coder(p.id, name="Coder B")
        xml = to_qde_xml(project=p, coders=[c1, c2])
        root = ET.fromstring(xml)
        users = root.find(f"{{{REFI_QDA_PROJECT_NS}}}Users")
        assert users is not None
        users_list = list(users)
        assert len(users_list) == 2
        assert {u.get("name") for u in users_list} == {"Coder A", "Coder B"}

    def test_users_fallback_when_empty(self) -> None:
        p = _project()
        xml = to_qde_xml(project=p)
        root = ET.fromstring(xml)
        users = root.find(f"{{{REFI_QDA_PROJECT_NS}}}Users")
        users_list = list(users)
        # We always emit at least one user so REFI-QDA importers that
        # require a creatingUser don't choke.
        assert len(users_list) == 1
        assert users_list[0].get("name") == "Scribe"

    def test_codebook_hierarchy_nested(self) -> None:
        p = _project()
        parent = _code(p.id, name="Identity work", code_id="111111111111")
        child = _code(p.id, name="Pacing", code_id="222222222222",
                      parent_code_id="111111111111")
        xml = to_qde_xml(project=p, codes=[parent, child])
        root = ET.fromstring(xml)
        codes_el = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}CodeBook/"
            f"{{{REFI_QDA_PROJECT_NS}}}Codes"
        )
        top = list(codes_el.findall(f"{{{REFI_QDA_PROJECT_NS}}}Code"))
        assert len(top) == 1
        assert top[0].get("name") == "Identity work"
        nested = list(top[0].findall(f"{{{REFI_QDA_PROJECT_NS}}}Code"))
        assert len(nested) == 1
        assert nested[0].get("name") == "Pacing"

    def test_code_carries_description(self) -> None:
        p = _project()
        c = _code(p.id, definition="Adjusting activity to manage energy.")
        xml = to_qde_xml(project=p, codes=[c])
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}CodeBook/"
            f"{{{REFI_QDA_PROJECT_NS}}}Codes/"
            f"{{{REFI_QDA_PROJECT_NS}}}Code"
        )
        desc = code_el.find(f"{{{REFI_QDA_PROJECT_NS}}}Description")
        assert desc is not None
        assert "Adjusting activity" in (desc.text or "")

    def test_code_colour_expanded_to_uppercase_six_char(self) -> None:
        p = _project()
        c = _code(p.id, colour="#abc")
        xml = to_qde_xml(project=p, codes=[c])
        root = ET.fromstring(xml)
        code_el = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}CodeBook/"
            f"{{{REFI_QDA_PROJECT_NS}}}Codes/"
            f"{{{REFI_QDA_PROJECT_NS}}}Code"
        )
        assert code_el.get("color") == "#AABBCC"

    def test_text_source_emitted(self) -> None:
        p = _project()
        s = _source(p.id, name="Interview 1")
        xml = to_qde_xml(project=p, sources=[s])
        root = ET.fromstring(xml)
        ts = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}Sources/"
            f"{{{REFI_QDA_PROJECT_NS}}}TextSource"
        )
        assert ts is not None
        assert ts.get("name") == "Interview 1"
        assert ts.get("plainTextPath") == f"internal://Sources/{s.id}.txt"
        assert ts.get("guid") == source_guid(s.id)

    def test_application_emits_selection_and_coding(self) -> None:
        p = _project()
        s = _source(p.id, name="Interview 1")
        cd = _coder(p.id)
        c = _code(p.id, name="Pacing")
        segs = _segments([["Hello", "world"]])
        rendered = render_source_plain_text(s.id, segs)
        a = _app(
            p.id, code_id=c.id, source_id=s.id, coder_id=cd.id,
            start_word_id="s0w0", end_word_id="s0w1",
        )
        xml = to_qde_xml(
            project=p, sources=[s], codes=[c], coders=[cd],
            applications=[a], rendered_sources=[rendered],
        )
        root = ET.fromstring(xml)
        ts = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}Sources/"
            f"{{{REFI_QDA_PROJECT_NS}}}TextSource"
        )
        sel = ts.find(f"{{{REFI_QDA_PROJECT_NS}}}PlainTextSelection")
        assert sel is not None
        assert sel.get("startPosition") == "0"
        assert sel.get("endPosition") == "11"
        assert sel.get("guid") == selection_guid(a.id)
        cod = sel.find(f"{{{REFI_QDA_PROJECT_NS}}}Coding")
        assert cod is not None
        cref = cod.find(f"{{{REFI_QDA_PROJECT_NS}}}CodeRef")
        assert cref is not None
        assert cref.get("targetGUID") == code_guid(c.id)

    def test_orphan_application_skipped(self) -> None:
        # An application whose anchor isn't in the rendering shouldn't
        # produce a selection (and shouldn't raise).
        p = _project()
        s = _source(p.id)
        cd = _coder(p.id)
        c = _code(p.id)
        segs = _segments([["Hello"]])
        rendered = render_source_plain_text(s.id, segs)
        a = _app(p.id, code_id=c.id, source_id=s.id, coder_id=cd.id,
                 start_word_id="s9w9", end_word_id="s9w9")
        xml = to_qde_xml(
            project=p, sources=[s], codes=[c], coders=[cd],
            applications=[a], rendered_sources=[rendered],
        )
        root = ET.fromstring(xml)
        ts = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}Sources/"
            f"{{{REFI_QDA_PROJECT_NS}}}TextSource"
        )
        # No PlainTextSelection because the anchor is orphaned.
        assert ts.find(f"{{{REFI_QDA_PROJECT_NS}}}PlainTextSelection") is None

    def test_application_without_rendering_skipped(self) -> None:
        # If we never rendered the source's transcript, we have no
        # offsets — the source is emitted (so its existence is known
        # to the importer) but no selections are nested.
        p = _project()
        s = _source(p.id)
        cd = _coder(p.id)
        c = _code(p.id)
        a = _app(p.id, code_id=c.id, source_id=s.id, coder_id=cd.id)
        xml = to_qde_xml(
            project=p, sources=[s], codes=[c], coders=[cd], applications=[a],
        )
        root = ET.fromstring(xml)
        ts = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}Sources/"
            f"{{{REFI_QDA_PROJECT_NS}}}TextSource"
        )
        assert ts is not None
        assert ts.find(f"{{{REFI_QDA_PROJECT_NS}}}PlainTextSelection") is None

    def test_notes_emitted_for_memos(self) -> None:
        p = _project()
        m1 = _memo(p.id, title="One")
        m2 = _memo(p.id, title="Two")
        xml = to_qde_xml(project=p, memos=[m1, m2])
        root = ET.fromstring(xml)
        notes = root.find(f"{{{REFI_QDA_PROJECT_NS}}}Notes")
        assert notes is not None
        notes_list = list(notes)
        assert len(notes_list) == 2
        assert all(n.get("plainTextPath", "").startswith("internal://Notes/")
                   for n in notes_list)

    def test_notes_omitted_when_no_memos(self) -> None:
        p = _project()
        xml = to_qde_xml(project=p)
        root = ET.fromstring(xml)
        # No <Notes/> when there's nothing to emit. (Importers tolerate
        # both, but cleaner output prefers omission.)
        assert root.find(f"{{{REFI_QDA_PROJECT_NS}}}Notes") is None

    def test_unicode_round_trips(self) -> None:
        p = _project(name="Días — los tiempos buenos")
        xml = to_qde_xml(project=p)
        root = ET.fromstring(xml)
        assert root.get("name") == "Días — los tiempos buenos"

    def test_code_cycles_do_not_recurse_forever(self) -> None:
        p = _project()
        a = _code(p.id, code_id="aaaaaaaaaaaa", parent_code_id="bbbbbbbbbbbb")
        b = _code(p.id, code_id="bbbbbbbbbbbb", parent_code_id="aaaaaaaaaaaa")
        # If the cycle resolution was naive, this would StackOverflow.
        xml = to_qde_xml(project=p, codes=[a, b])
        root = ET.fromstring(xml)
        codes_el = root.find(
            f"{{{REFI_QDA_PROJECT_NS}}}CodeBook/"
            f"{{{REFI_QDA_PROJECT_NS}}}Codes"
        )
        # Both emitted at top level (cycle → flat).
        assert len(list(codes_el.findall(f"{{{REFI_QDA_PROJECT_NS}}}Code"))) == 2


# --------------------------------------------------------------------------- #
# QDPX archive
# --------------------------------------------------------------------------- #


class TestToQdpx:
    def test_archive_contains_project_qde(self) -> None:
        p = _project()
        archive = to_qdpx(project=p)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            assert "project.qde" in names
            qde = zf.read("project.qde").decode("utf-8")
            # Validates as XML.
            ET.fromstring(qde)

    def test_archive_includes_source_plain_text(self) -> None:
        p = _project()
        s = _source(p.id, name="Interview 1")
        segs = _segments([["Hello", "world"]])
        rendered = render_source_plain_text(s.id, segs)
        archive = to_qdpx(project=p, sources=[s], rendered_sources=[rendered])
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            assert f"Sources/{s.id}.txt" in names
            text = zf.read(f"Sources/{s.id}.txt").decode("utf-8")
            assert text == "Hello world"

    def test_archive_includes_memos_as_notes(self) -> None:
        p = _project()
        m = _memo(p.id, body="Hypothesis: pacing emerges...")
        archive = to_qdpx(project=p, memos=[m])
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            assert f"Notes/{m.id}.txt" in zf.namelist()
            assert zf.read(f"Notes/{m.id}.txt").decode("utf-8") == m.body

    def test_archive_is_zip(self) -> None:
        archive = to_qdpx(project=_project())
        # ZIP files start with "PK".
        assert archive[:2] == b"PK"

    def test_full_project_round_trip(self) -> None:
        # End-to-end: a project with two sources, three codes, two
        # coders, four applications, two memos. We don't compare bytes
        # but assert all the entities are reachable from the archive.
        p = _project()
        s1 = _source(p.id, name="Interview 1", source_id="111111111111")
        s2 = _source(p.id, name="Interview 2", source_id="222222222222")
        cd1 = _coder(p.id, name="Coder A")
        cd2 = _coder(p.id, name="Coder B")
        c1 = _code(p.id, name="Pacing")
        c2 = _code(p.id, name="Disclosure")
        c3 = _code(p.id, name="Identity work")

        segs1 = _segments(
            [["Hello", "world"], ["Goodbye", "now"]],
            speakers=["A", "B"],
        )
        segs2 = _segments([["This", "is", "a", "test"]])
        r1 = render_source_plain_text(s1.id, segs1)
        r2 = render_source_plain_text(s2.id, segs2)

        a1 = _app(p.id, code_id=c1.id, source_id=s1.id, coder_id=cd1.id,
                  start_word_id="s0w0", end_word_id="s0w1",
                  application_id="100000000001")
        a2 = _app(p.id, code_id=c2.id, source_id=s1.id, coder_id=cd2.id,
                  start_word_id="s1w0", end_word_id="s1w1",
                  application_id="100000000002")
        a3 = _app(p.id, code_id=c3.id, source_id=s2.id, coder_id=cd1.id,
                  start_word_id="s0w0", end_word_id="s0w3",
                  application_id="100000000003")

        m1 = _memo(p.id, title="Pacing memo")
        m2 = _memo(p.id, title="Disclosure memo")

        archive = to_qdpx(
            project=p, sources=[s1, s2], codes=[c1, c2, c3],
            coders=[cd1, cd2], applications=[a1, a2, a3],
            memos=[m1, m2],
            rendered_sources=[r1, r2],
        )

        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            assert "project.qde" in names
            assert f"Sources/{s1.id}.txt" in names
            assert f"Sources/{s2.id}.txt" in names
            assert f"Notes/{m1.id}.txt" in names
            assert f"Notes/{m2.id}.txt" in names

            qde = zf.read("project.qde").decode("utf-8")
            root = ET.fromstring(qde)

            # Three codes.
            codes_el = root.find(
                f"{{{REFI_QDA_PROJECT_NS}}}CodeBook/"
                f"{{{REFI_QDA_PROJECT_NS}}}Codes"
            )
            assert len(list(codes_el)) == 3

            # Two text sources.
            sources_el = root.find(f"{{{REFI_QDA_PROJECT_NS}}}Sources")
            assert len(list(sources_el)) == 2

            # Three selections total across both sources.
            sel_count = 0
            for ts in sources_el:
                sel_count += len(list(ts))
            assert sel_count == 3

    def test_xml_uses_default_namespace_no_prefix(self) -> None:
        # Atlas.ti and NVivo are picky about the ``ns0:`` prefix
        # ElementTree emits by default. We register the empty prefix
        # so the XML uses ``xmlns="…"`` instead.
        p = _project()
        archive = to_qdpx(project=p)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            qde = zf.read("project.qde").decode("utf-8")
        # Spot-check: should NOT contain "ns0:".
        assert "ns0:" not in qde


# --------------------------------------------------------------------------- #
# Slug
# --------------------------------------------------------------------------- #


class TestSlugifyQdpxFilename:
    def test_basic(self) -> None:
        p = _project(name="Living with chronic illness")
        assert slugify_qdpx_filename(p) == "living-with-chronic-illness.qdpx"

    def test_unicode_downgraded(self) -> None:
        p = _project(name="Días buenos")
        # NFKD-normalised ASCII downgrade
        assert slugify_qdpx_filename(p) == "dias-buenos.qdpx"

    def test_punctuation_collapsed(self) -> None:
        p = _project(name="A: study (of) work!")
        # All punctuation runs collapse to one dash; trailing dashes stripped.
        out = slugify_qdpx_filename(p)
        assert out.endswith(".qdpx")
        assert "-" in out
        assert ":" not in out

    def test_none_falls_back(self) -> None:
        assert slugify_qdpx_filename(None) == "project.qdpx"

    def test_empty_after_normalisation_falls_back(self) -> None:
        # All-emoji name: NFKD ASCII downgrade yields nothing.
        # We use a name that's all unicode-only chars.
        p = _project(name="日本語")
        assert slugify_qdpx_filename(p) == "project.qdpx"
