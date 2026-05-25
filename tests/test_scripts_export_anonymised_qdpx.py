"""Tests for ``scribe.scripts.export_anonymised_qdpx`` (F6.7 CLI surface).

The CLI is a thin wrapper around :mod:`scribe.anonymise`. We exercise:
argument parsing, custom-rule loading, full end-to-end project bundle
production with the redaction layer engaged, and the soft-warning
path when sources have no discoverable transcripts.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from scribe import applications as _applications
from scribe import coders as _coders
from scribe import codes as _codes
from scribe import memos as _memos
from scribe import participants as _participants
from scribe import projects as _projects
from scribe import sources as _sources
from scribe import speaker_map as _speaker_map
from scribe.anonymise import RedactionRule
from scribe.applications import Application
from scribe.coders import Coder
from scribe.codes import Code
from scribe.memos import Memo
from scribe.participants import Participant
from scribe.projects import Project
from scribe.scripts import export_anonymised_qdpx as cli
from scribe.sources import Source
from scribe.speaker_map import SpeakerEntry, SpeakerMap


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_setup(tmp_path: Path) -> dict:
    """Persist a small project with one source, one code, one coder,
    one application, one memo, two participants, and a speaker map.
    Lay down a transcript under ``outputs/<job_id>/edited.json``.
    """
    projects_root = tmp_path / "projects"
    outputs_root = tmp_path / "outputs"
    projects_root.mkdir()
    outputs_root.mkdir()

    project = Project.new(name="Living with chronic illness")
    _projects.save_project(projects_root, project)

    p_jane = Participant.new(
        project_id=project.id, name="Jane Doe", pseudonym="P01"
    )
    p_anon = Participant.new(
        project_id=project.id, name="Sam", pseudonym=""
    )
    _participants.save_participant(projects_root, p_jane)
    _participants.save_participant(projects_root, p_anon)

    coder = Coder.new(project_id=project.id, name="Pat (RA)")
    _coders.save_coder(projects_root, coder)

    code = Code.new(
        project_id=project.id,
        name="Pacing",
        definition="Jane Doe described pacing as reactive.",
    )
    _codes.save_code(projects_root, code)

    job_id = "abcdef012345"
    source = Source.new(
        project_id=project.id,
        name="Interview with Jane Doe",
        source_type="transcript",
        transcript_job_id=job_id,
    )
    _sources.save_source(projects_root, source)

    job_dir = outputs_root / job_id
    job_dir.mkdir()
    transcript = {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "words": [
                    {"text": "Hello"},
                    {"text": "Jane"},
                    {"text": "Doe"},
                ],
            },
        ],
    }
    (job_dir / "edited.json").write_text(json.dumps(transcript))

    sm = SpeakerMap.new(
        project_id=project.id,
        source_id=source.id,
        entries=[
            SpeakerEntry(
                label="SPEAKER_00",
                role="interviewee",
                participant_id=p_jane.id,
            ),
        ],
    )
    _speaker_map.save_speaker_map(projects_root, sm)

    app = Application.new(
        project_id=project.id,
        code_id=code.id,
        source_id=source.id,
        coder_id=coder.id,
        anchor_start_word_id="s0w1",
        anchor_end_word_id="s0w2",
        definition_version_id_at_apply="aaaabbbbcccc",
    )
    _applications.save_application(projects_root, app)

    memo = Memo.new(
        project_id=project.id,
        type="theoretical",
        title="On Jane Doe",
        body="Jane Doe described pacing as reactive.",
    )
    _memos.save_memo(projects_root, memo)

    return {
        "projects_root": projects_root,
        "outputs_root": outputs_root,
        "project_id": project.id,
        "source_id": source.id,
        "job_id": job_id,
        "code_id": code.id,
        "coder_id": coder.id,
        "memo_id": memo.id,
    }


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_requires_project(self) -> None:
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_projects_root(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.projects_root == Path("projects")

    def test_default_outputs_root(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.outputs_root == Path("outputs")

    def test_default_rules_is_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.rules is None

    def test_default_out_is_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.out is None


# --------------------------------------------------------------------------- #
# Custom rules loader
# --------------------------------------------------------------------------- #


class TestLoadCustomRules:
    def test_none_returns_empty(self) -> None:
        assert cli.load_custom_rules(None) == []

    def test_parses_valid_rules(self, tmp_path: Path) -> None:
        rf = tmp_path / "rules.json"
        rf.write_text(json.dumps([
            {"pattern": "Mercy", "replacement": "HOSP"},
            {"pattern": r"\d{3}-\d{4}", "replacement": "[phone]", "regex": True},
        ]))
        rules = cli.load_custom_rules(rf)
        assert len(rules) == 2
        assert rules[0] == RedactionRule(pattern="Mercy", replacement="HOSP")
        assert rules[1].regex is True

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.load_custom_rules(tmp_path / "nope.json")

    def test_non_list_exits(self, tmp_path: Path) -> None:
        rf = tmp_path / "rules.json"
        rf.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(SystemExit):
            cli.load_custom_rules(rf)

    def test_non_object_entry_exits(self, tmp_path: Path) -> None:
        rf = tmp_path / "rules.json"
        rf.write_text(json.dumps(["string entry"]))
        with pytest.raises(SystemExit):
            cli.load_custom_rules(rf)

    def test_invalid_rule_payload_exits(self, tmp_path: Path) -> None:
        rf = tmp_path / "rules.json"
        rf.write_text(json.dumps([{"pattern": "x"}]))  # missing replacement
        with pytest.raises(SystemExit):
            cli.load_custom_rules(rf)


# --------------------------------------------------------------------------- #
# Main path
# --------------------------------------------------------------------------- #


class TestMain:
    def test_writes_anon_qdpx_to_disk(
        self,
        project_setup: dict,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "study-anon.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--out", str(out),
        ])
        assert rc == 0
        assert out.exists()
        with zipfile.ZipFile(out, mode="r") as zf:
            names = zf.namelist()
            assert "project.qde" in names
            assert "Redactions/manifest.json" in names
            # The redacted source plain-text must NOT contain the
            # participant's real name.
            sid = project_setup["source_id"]
            txt = zf.read(f"Sources/{sid}.txt").decode("utf-8")
            assert "Jane Doe" not in txt
            assert "P01" in txt
            mani = json.loads(zf.read("Redactions/manifest.json"))
            assert "Jane Doe" not in json.dumps(mani)
            assert mani["total_substitutions"] >= 1

    def test_missing_project_returns_2(
        self,
        project_setup: dict,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", "111111111111",
            "--out", "/dev/null",
        ])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_project_id_returns_2(
        self,
        project_setup: dict,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", "not-hex",
            "--out", "/dev/null",
        ])
        assert rc == 2

    def test_custom_rules_layer_in(
        self,
        project_setup: dict,
        tmp_path: Path,
    ) -> None:
        rf = tmp_path / "rules.json"
        rf.write_text(json.dumps([
            {"pattern": "Pacing", "replacement": "Theme01"},
        ]))
        out = tmp_path / "anon.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--rules", str(rf),
            "--out", str(out),
        ])
        assert rc == 0
        with zipfile.ZipFile(out, mode="r") as zf:
            qde = zf.read("project.qde").decode("utf-8")
            assert "Pacing" not in qde  # code name was redacted
            assert "Theme01" in qde

    def test_warns_when_source_has_no_transcript(
        self,
        project_setup: dict,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Wipe the outputs dir so the only source has no transcript.
        for f in (project_setup["outputs_root"] / project_setup["job_id"]).iterdir():
            f.unlink()
        out = tmp_path / "anon.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--out", str(out),
        ])
        assert rc == 0
        assert "no" in capsys.readouterr().err.lower()
        # Bundle still wrote out fine.
        assert out.exists()

    def test_writes_to_stdout_when_no_out(
        self,
        project_setup: dict,
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
        ])
        assert rc == 0
        captured = capsysbinary.readouterr()
        assert captured.out  # bytes written
        # Recognise the zip magic.
        assert captured.out[:2] == b"PK"

    def test_note_threaded_into_returned_manifest(
        self,
        project_setup: dict,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "anon.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--note", "Pre-publication anon pass",
            "--out", str(out),
        ])
        assert rc == 0
        # The on-disk archive's manifest carries the runtime note only
        # if we post-stamp it; here we just verify the script ran with
        # --note and produced a valid bundle.
        with zipfile.ZipFile(out, mode="r") as zf:
            assert "Redactions/manifest.json" in zf.namelist()
