"""Tests for ``scribe.scripts.export_qdpx`` (F6.4 CLI surface).

The CLI is a thin wrapper around :mod:`scribe.refi_qda_project`. We
exercise: argument parsing, transcript discovery under ``--outputs-root``
(both ``edited.json`` precedence and the *.json fallback), stdout vs
``--out``, missing projects, and the soft-warning path when sources
lack discoverable transcripts.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scribe import applications as _applications
from scribe import coders as _coders
from scribe import codes as _codes
from scribe import memos as _memos
from scribe import projects as _projects
from scribe import refi_qda_project as qdpx
from scribe import sources as _sources
from scribe.applications import Application
from scribe.coders import Coder
from scribe.codes import Code
from scribe.memos import Memo
from scribe.projects import Project
from scribe.scripts import export_qdpx as cli
from scribe.sources import Source


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_setup(tmp_path: Path) -> dict:
    """Build a project on disk with one source, two codes, one
    coder, two applications, and one memo. Also lay down a transcript
    under ``outputs/<job_id>/edited.json`` so the source's
    transcript_job_id resolves.
    """
    projects_root = tmp_path / "projects"
    outputs_root = tmp_path / "outputs"
    projects_root.mkdir()
    outputs_root.mkdir()

    project = Project.new(name="Pilot")
    _projects.save_project(projects_root, project)

    # Coder
    coder = Coder.new(project_id=project.id, name="Coder A")
    _coders.save_coder(projects_root, coder)

    # Codes
    c_pace = Code.new(project_id=project.id, name="Pacing", definition="Activity rationing.")
    c_disc = Code.new(project_id=project.id, name="Disclosure")
    _codes.save_code(projects_root, c_pace)
    _codes.save_code(projects_root, c_disc)

    # Source pointing at a job
    job_id = "abcdef012345"
    source = Source.new(
        project_id=project.id,
        name="Interview 1",
        source_type="transcript",
        transcript_job_id=job_id,
    )
    _sources.save_source(projects_root, source)

    # Lay down a transcript under outputs/<job>/edited.json
    job_dir = outputs_root / job_id
    job_dir.mkdir()
    transcript = {
        "segments": [
            {
                "speaker": "LUKE",
                "words": [{"text": "Hello"}, {"text": "world"}, {"text": "today"}],
            },
            {
                "speaker": "ANA",
                "words": [{"text": "Goodbye"}, {"text": "now"}],
            },
        ],
    }
    (job_dir / "edited.json").write_text(json.dumps(transcript))

    # Applications — anchor "Hello world" and "Goodbye"
    a1 = Application.new(
        project_id=project.id,
        code_id=c_pace.id,
        source_id=source.id,
        coder_id=coder.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w1",
        definition_version_id_at_apply="aaaabbbbcccc",
    )
    a2 = Application.new(
        project_id=project.id,
        code_id=c_disc.id,
        source_id=source.id,
        coder_id=coder.id,
        anchor_start_word_id="s1w0",
        anchor_end_word_id="s1w0",
        definition_version_id_at_apply="aaaabbbbcccc",
    )
    _applications.save_application(projects_root, a1)
    _applications.save_application(projects_root, a2)

    # Memo
    m = Memo.new(
        project_id=project.id,
        type="theoretical",
        title="Why pacing?",
        body="Hypothesis: pacing is a self-management strategy.",
    )
    _memos.save_memo(projects_root, m)

    return {
        "projects_root": projects_root,
        "outputs_root": outputs_root,
        "project_id": project.id,
        "source_id": source.id,
        "code_ids": [c_pace.id, c_disc.id],
        "coder_id": coder.id,
        "application_ids": [a1.id, a2.id],
        "memo_id": m.id,
        "job_id": job_id,
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

    def test_default_origin(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.origin == qdpx.REFI_QDA_PROJECT_ORIGIN_DEFAULT

    def test_out_defaults_to_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.out is None


# --------------------------------------------------------------------------- #
# Transcript discovery
# --------------------------------------------------------------------------- #


class TestLoadSegmentsForSource:
    def test_prefers_edited_json(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        job_dir = outputs / "abcdef012345"
        job_dir.mkdir(parents=True)
        # Two candidates: one is edited.json with the right shape; the
        # other is a sidecar with different content. We assert the
        # script picks edited.json.
        (job_dir / "edited.json").write_text(json.dumps(
            {"segments": [{"speaker": "", "words": [{"text": "edited"}]}]}
        ))
        (job_dir / "input.json").write_text(json.dumps(
            {"segments": [{"speaker": "", "words": [{"text": "original"}]}]}
        ))
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        segs = cli._load_segments_for_source(outputs, source)
        assert segs is not None
        assert segs[0]["words"][0]["text"] == "edited"

    def test_falls_back_to_sidecar_json(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        job_dir = outputs / "abcdef012345"
        job_dir.mkdir(parents=True)
        (job_dir / "input.json").write_text(json.dumps(
            {"segments": [{"speaker": "", "words": [{"text": "original"}]}]}
        ))
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        segs = cli._load_segments_for_source(outputs, source)
        assert segs is not None
        assert segs[0]["words"][0]["text"] == "original"

    def test_no_job_id_returns_none(self, tmp_path: Path) -> None:
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
        )
        assert cli._load_segments_for_source(tmp_path / "outputs", source) is None

    def test_missing_dir_returns_none(self, tmp_path: Path) -> None:
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        assert cli._load_segments_for_source(tmp_path / "outputs", source) is None

    def test_no_parseable_json_returns_none(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        job_dir = outputs / "abcdef012345"
        job_dir.mkdir(parents=True)
        (job_dir / "edited.json").write_text("not json")
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        assert cli._load_segments_for_source(outputs, source) is None

    def test_json_without_segments_returns_none(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"
        job_dir = outputs / "abcdef012345"
        job_dir.mkdir(parents=True)
        (job_dir / "info.json").write_text(json.dumps({"hello": "world"}))
        source = Source.new(
            project_id="000000000001",
            name="t",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        assert cli._load_segments_for_source(outputs, source) is None


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestMain:
    def test_writes_qdpx_to_disk(
        self,
        project_setup: dict,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "study.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--out", str(out),
        ])
        assert rc == 0
        assert out.exists()
        assert out.read_bytes()[:2] == b"PK"  # ZIP magic
        captured = capsys.readouterr()
        assert "Wrote QDPX archive" in captured.err

    def test_archive_contents(
        self,
        project_setup: dict,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "study.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--out", str(out),
        ])
        assert rc == 0
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            assert "project.qde" in names
            sid = project_setup["source_id"]
            mid = project_setup["memo_id"]
            assert f"Sources/{sid}.txt" in names
            assert f"Notes/{mid}.txt" in names

            text = zf.read(f"Sources/{sid}.txt").decode("utf-8")
            assert "LUKE: Hello world today" in text
            assert "ANA: Goodbye now" in text

            qde = zf.read("project.qde").decode("utf-8")
            root = ET.fromstring(qde)
            ns = qdpx.REFI_QDA_PROJECT_NS
            # Two codes in the codebook
            codes_el = root.find(f"{{{ns}}}CodeBook/{{{ns}}}Codes")
            assert len(list(codes_el)) == 2
            # Two selections in the source
            ts = root.find(f"{{{ns}}}Sources/{{{ns}}}TextSource")
            assert len(list(ts)) == 2

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
        # The bytes should be a valid zip.
        assert captured.out[:2] == b"PK"
        with zipfile.ZipFile(io.BytesIO(captured.out)) as zf:
            assert "project.qde" in zf.namelist()

    def test_missing_project_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main([
            "--projects-root", str(tmp_path / "projects"),
            "--project", "ffffffffffff",
        ])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_project_id_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main([
            "--projects-root", str(tmp_path / "projects"),
            "--project", "not-hex",
        ])
        assert rc == 2

    def test_warns_about_missing_transcripts(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Project has a source but no transcript on disk.
        projects_root = tmp_path / "projects"
        outputs_root = tmp_path / "outputs"
        projects_root.mkdir()
        outputs_root.mkdir()
        project = Project.new(name="No Transcripts")
        _projects.save_project(projects_root, project)
        s = Source.new(
            project_id=project.id,
            name="Missing",
            source_type="transcript",
            transcript_job_id="aaaabbbbcccc",
        )
        _sources.save_source(projects_root, s)

        out = tmp_path / "out.qdpx"
        rc = cli.main([
            "--projects-root", str(projects_root),
            "--outputs-root", str(outputs_root),
            "--project", project.id,
            "--out", str(out),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "no discoverable transcript" in captured.err

    def test_out_creates_parent_directories(
        self,
        project_setup: dict,
        tmp_path: Path,
    ) -> None:
        deep = tmp_path / "a" / "b" / "c" / "study.qdpx"
        rc = cli.main([
            "--projects-root", str(project_setup["projects_root"]),
            "--outputs-root", str(project_setup["outputs_root"]),
            "--project", project_setup["project_id"],
            "--out", str(deep),
        ])
        assert rc == 0
        assert deep.exists()
