"""Unit tests for :mod:`scribe.query_runtime` — the F3.5 runtime
adapter that bridges on-disk Applications + transcripts to the pure
:func:`scribe.query.applications_for_query` executor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe import (
    applications as _applications,
    code_versions as _code_versions,
    codes as _codes,
    coders as _coders,
    participants as _participants,
    projects as _projects,
    sources as _sources,
)
from scribe.projects import ProjectValidationError
from scribe.query import (
    AttributePredicate,
    CodeExpr,
    CodeFilter,
    ParticipantFilter,
    Query,
    SourceFilter,
    SpeakerFilter,
)
from scribe.query_runtime import (
    QueryRunReport,
    application_to_query_dict,
    run_query_against_project,
)


# --------------------------------------------------------------------------- #
# Fixtures — a tiny on-disk project with two sources, two codes,
# two coders, and a handful of applications.
# --------------------------------------------------------------------------- #


@pytest.fixture
def proj(tmp_path: Path):
    """Build an isolated project with two sources, two codes, applications."""
    root = tmp_path / "projects"
    root.mkdir()

    p = _projects.Project.new(name="QR test")
    _projects.save_project(root, p)

    src1 = _sources.Source.new(
        project_id=p.id, name="S1", source_type="transcript",
        language="en",
    )
    src2 = _sources.Source.new(
        project_id=p.id, name="S2", source_type="transcript",
        language="en",
    )
    _sources.save_source(root, src1)
    _sources.save_source(root, src2)

    c1 = _codes.Code.new(project_id=p.id, name="anxiety")
    c2 = _codes.Code.new(project_id=p.id, name="hope")
    _codes.save_code(root, c1)
    _codes.save_code(root, c2)
    v1 = _code_versions.record_code_version(root, c1, change_note="init")
    v2 = _code_versions.record_code_version(root, c2, change_note="init")

    coder = _coders.Coder.new(project_id=p.id, name="Default")
    _coders.save_coder(root, coder)

    apps = []
    # source 1 — anchor segment 0, anchored on word 1
    a1 = _applications.Application.new(
        project_id=p.id,
        code_id=c1.id,
        source_id=src1.id,
        coder_id=coder.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w2",
        definition_version_id_at_apply=v1.id,
    )
    a2 = _applications.Application.new(
        project_id=p.id,
        code_id=c2.id,
        source_id=src1.id,
        coder_id=coder.id,
        anchor_start_word_id="s1w0",
        anchor_end_word_id="s1w1",
        definition_version_id_at_apply=v2.id,
    )
    a3 = _applications.Application.new(
        project_id=p.id,
        code_id=c1.id,
        source_id=src2.id,
        coder_id=coder.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w0",
        definition_version_id_at_apply=v1.id,
    )
    for a in (a1, a2, a3):
        _applications.save_application(root, a)
        apps.append(a)

    return {
        "root": root,
        "project_id": p.id,
        "sources": [src1, src2],
        "codes": [c1, c2],
        "applications": apps,
        "version_ids": {c1.id: v1.id, c2.id: v2.id},
    }


# Two transcript fixtures — "who spoke" + "when" so the adapter has
# data to translate.
SEGMENTS_S1 = [
    {
        "speaker": "INTERVIEWER",
        "start": 0.0,
        "end": 1.5,
        "words": [
            {"word": "How", "start": 0.0, "end": 0.3},
            {"word": "are", "start": 0.3, "end": 0.6},
            {"word": "you", "start": 0.6, "end": 1.0},
        ],
    },
    {
        "speaker": "PARTICIPANT",
        "start": 1.5,
        "end": 3.0,
        "words": [
            {"word": "I'm", "start": 1.5, "end": 1.8},
            {"word": "anxious", "start": 1.8, "end": 2.6},
        ],
    },
]
SEGMENTS_S2 = [
    {
        "speaker": "PARTICIPANT",
        "start": 0.0,
        "end": 0.8,
        "words": [
            {"word": "Hopeful", "start": 0.0, "end": 0.8},
        ],
    },
]


def _segments_loader_factory(proj_data) -> "callable":
    """Factory that mirrors the production segments_loader: maps
    source_id → segments. Returns None for any unknown source."""
    src_ids = [s.id for s in proj_data["sources"]]

    def loader(sid: str):
        if sid == src_ids[0]:
            return SEGMENTS_S1
        if sid == src_ids[1]:
            return SEGMENTS_S2
        return None

    return loader


# --------------------------------------------------------------------------- #
# application_to_query_dict
# --------------------------------------------------------------------------- #


class TestApplicationToQueryDict:
    def test_resolves_speaker_from_segment(self, proj) -> None:
        a1 = proj["applications"][0]  # s0w0..s0w2 on src1
        d = application_to_query_dict(a1, SEGMENTS_S1)
        assert d["speaker"] == "INTERVIEWER"
        assert d["code_id"] == proj["codes"][0].id
        assert d["source_id"] == proj["sources"][0].id

    def test_resolves_second_segment_speaker(self, proj) -> None:
        a2 = proj["applications"][1]  # s1w0..s1w1 on src1
        d = application_to_query_dict(a2, SEGMENTS_S1)
        assert d["speaker"] == "PARTICIPANT"

    def test_includes_timing_when_words_have_stamps(self, proj) -> None:
        a1 = proj["applications"][0]  # s0w0 → s0w2 → 0.0 to 1.0
        d = application_to_query_dict(a1, SEGMENTS_S1)
        assert d["start"] == pytest.approx(0.0)
        assert d["end"] == pytest.approx(1.0)

    def test_no_segments_returns_minimal_dict(self, proj) -> None:
        a1 = proj["applications"][0]
        d = application_to_query_dict(a1, None)
        # Speaker falls back to empty string; no start/end fields.
        assert d["speaker"] == ""
        assert "start" not in d
        assert "end" not in d
        assert d["code_id"] == proj["codes"][0].id

    def test_anchor_segment_out_of_range(self, proj) -> None:
        a1 = proj["applications"][0]
        # Pass a 1-segment transcript — anchor s0 is OK, s1 isn't.
        d = application_to_query_dict(a1, [SEGMENTS_S1[0]])
        # s0 still resolves cleanly here.
        assert d["speaker"] == "INTERVIEWER"

        a2 = proj["applications"][1]  # anchored at s1
        d2 = application_to_query_dict(a2, [SEGMENTS_S1[0]])
        assert d2["speaker"] == ""  # out-of-range silently degrades

    def test_invalid_app_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            application_to_query_dict("not an app", None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# run_query_against_project — integration of the executor
# --------------------------------------------------------------------------- #


class TestRunQueryAgainstProject:
    def test_empty_query_returns_all_applications(self, proj) -> None:
        q = Query(project_id=proj["project_id"])
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
        )
        assert isinstance(report, QueryRunReport)
        assert report.total_applications == 3
        assert len(report.matches) == 3

    def test_filter_by_single_code(self, proj) -> None:
        c1_id = proj["codes"][0].id
        q = Query(
            project_id=proj["project_id"],
            codes=CodeFilter(expr=CodeExpr(op="code", code_id=c1_id)),
        )
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
        )
        assert len(report.matches) == 2  # a1 and a3
        for app in report.matches:
            assert app.code_id == c1_id

    def test_filter_by_source(self, proj) -> None:
        src2 = proj["sources"][1]
        q = Query(
            project_id=proj["project_id"],
            sources=SourceFilter(source_ids=[src2.id]),
        )
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
        )
        assert len(report.matches) == 1
        assert report.matches[0].source_id == src2.id

    def test_filter_by_speaker_role(self, proj) -> None:
        # Save a SpeakerMap on src1 mapping the labels to roles.
        from scribe import speaker_map as sm
        smap = sm.SpeakerMap.new(
            project_id=proj["project_id"],
            source_id=proj["sources"][0].id,
            entries=[
                sm.SpeakerEntry(label="INTERVIEWER", role="interviewer"),
                sm.SpeakerEntry(label="PARTICIPANT", role="interviewee"),
            ],
        )
        sm.save_speaker_map(proj["root"], smap)

        q = Query(
            project_id=proj["project_id"],
            speakers=SpeakerFilter(roles=["interviewee"]),
        )
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
        )
        # Only application a2 has speaker=PARTICIPANT under src1.
        # On src2 there's no speaker_map saved, so the executor falls
        # back to the empty SpeakerMap which has no role mapping —
        # PARTICIPANT label there does not match the role filter.
        # That's fine for this test: we expect a2 only.
        match_ids = {a.id for a in report.matches}
        assert proj["applications"][1].id in match_ids
        assert proj["applications"][0].id not in match_ids
        # a3 (src2, speaker=PARTICIPANT, no speaker_map) — without a
        # role mapping the SpeakerFilter can't see the label as a
        # role match, so it is excluded.
        assert proj["applications"][2].id not in match_ids

    def test_missing_transcript_recorded(self, proj) -> None:
        # Loader that returns None for src2.
        src1 = proj["sources"][0]
        src2 = proj["sources"][1]

        def loader(sid: str):
            if sid == src1.id:
                return SEGMENTS_S1
            return None

        q = Query(project_id=proj["project_id"])
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=loader,
        )
        assert src2.id in report.sources_missing_transcript
        assert src1.id not in report.sources_missing_transcript
        # All three applications still come through (filter is empty).
        assert len(report.matches) == 3

    def test_loader_exception_recorded_as_warning(self, proj) -> None:
        def loader(sid: str):
            raise OSError(f"boom {sid}")

        q = Query(project_id=proj["project_id"])
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=loader,
        )
        assert len(report.warnings) >= 1
        assert any("OSError" in w for w in report.warnings)
        # All sources end up in missing_transcript because the loader
        # always errored.
        assert len(report.sources_missing_transcript) >= 1

    def test_project_id_mismatch_rejected(self, proj) -> None:
        q = Query(project_id="ffffffffffff")  # different
        with pytest.raises(ProjectValidationError):
            run_query_against_project(
                proj["root"], proj["project_id"], q,
                segments_loader=_segments_loader_factory(proj),
            )

    def test_invalid_query_raises(self, proj) -> None:
        with pytest.raises(ProjectValidationError):
            run_query_against_project(
                proj["root"], proj["project_id"], "not a query",  # type: ignore[arg-type]
                segments_loader=_segments_loader_factory(proj),
            )

    def test_caller_supplied_applications_respected(self, proj) -> None:
        # Pass only one of three; result should only consider that one.
        q = Query(project_id=proj["project_id"])
        only_one = [proj["applications"][0]]
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
            applications=only_one,
        )
        assert report.total_applications == 1
        assert len(report.matches) == 1
        assert report.matches[0].id == only_one[0].id

    def test_filter_combining_code_and_source(self, proj) -> None:
        # anxiety code on src1 only — should match a1.
        c1_id = proj["codes"][0].id
        src1 = proj["sources"][0]
        q = Query(
            project_id=proj["project_id"],
            codes=CodeFilter(expr=CodeExpr(op="code", code_id=c1_id)),
            sources=SourceFilter(source_ids=[src1.id]),
        )
        report = run_query_against_project(
            proj["root"], proj["project_id"], q,
            segments_loader=_segments_loader_factory(proj),
        )
        assert len(report.matches) == 1
        assert report.matches[0].id == proj["applications"][0].id


# --------------------------------------------------------------------------- #
# QueryRunReport — defaults
# --------------------------------------------------------------------------- #


class TestQueryRunReport:
    def test_defaults_are_empty_lists(self) -> None:
        rpt = QueryRunReport(matches=[], total_applications=0)
        assert rpt.sources_missing_transcript == []
        assert rpt.warnings == []
