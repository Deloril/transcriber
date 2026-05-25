"""Tests for ``scribe.scripts.export_retrieval_report`` (F6.2 CLI surface).

The CLI is a thin wrapper over :mod:`scribe.retrieval_report` and the
per-entity persistence helpers. Exercise: argument parsing, format
+ group-by dispatch, filter application, stdout vs ``--out``, and the
failure modes (unknown format, unknown group_by, missing project,
invalid project id).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from scribe import applications as _applications
from scribe import coders as _coders
from scribe import codes as _codes
from scribe import code_versions as _code_versions
from scribe import participants as _participants
from scribe import projects as _projects
from scribe import retrieval_report as rr
from scribe import sources as _sources
from scribe.applications import Application
from scribe.coders import Coder
from scribe.codes import Code
from scribe.participants import Participant
from scribe.scripts import export_retrieval_report as cli
from scribe.sources import Source


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_with_applications(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    """Build a small but full-featured project on disk.

    Returns ``(projects_root, project_id, ids_by_label)`` where
    ``ids_by_label`` maps the human label ("pacing", "interview-1",
    "luke", "alice") to the entity id, so tests can construct
    filter arguments without hard-coding 12-char hexes.
    """
    root = tmp_path / "projects"
    root.mkdir()

    project = _projects.Project.new(name="Pilot study")
    _projects.save_project(root, project)

    pacing = Code.new(project_id=project.id, name="Pacing")
    resting = Code.new(project_id=project.id, name="Resting")
    _codes.save_code(root, pacing)
    _codes.save_code(root, resting)

    src1 = Source.new(project_id=project.id, name="Interview 01")
    src2 = Source.new(project_id=project.id, name="Interview 02")
    _sources.save_source(root, src1)
    _sources.save_source(root, src2)

    luke = Coder.new(project_id=project.id, name="Luke")
    sam = Coder.new(project_id=project.id, name="Sam")
    _coders.save_coder(root, luke)
    _coders.save_coder(root, sam)

    alice = Participant.new(
        project_id=project.id, name="Alice", source_ids=[src1.id]
    )
    bob = Participant.new(
        project_id=project.id, name="Bob", source_ids=[src2.id]
    )
    _participants.save_participant(root, alice)
    _participants.save_participant(root, bob)

    # Code-version ids — per F2.2, applications carry one. We mint
    # fresh ones; their on-disk persistence isn't required for the
    # retrieval-report exporter (it only checks shape).
    v1 = _code_versions.new_code_version_id()
    v2 = _code_versions.new_code_version_id()

    a1 = Application.new(
        project_id=project.id,
        code_id=pacing.id,
        source_id=src1.id,
        coder_id=luke.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w0",
        definition_version_id_at_apply=v1,
    )
    a2 = Application.new(
        project_id=project.id,
        code_id=resting.id,
        source_id=src2.id,
        coder_id=sam.id,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w1",
        definition_version_id_at_apply=v2,
    )
    _applications.save_application(root, a1)
    _applications.save_application(root, a2)

    return (
        root,
        project.id,
        {
            "pacing": pacing.id,
            "resting": resting.id,
            "interview-1": src1.id,
            "interview-2": src2.id,
            "luke": luke.id,
            "sam": sam.id,
            "alice": alice.id,
            "bob": bob.id,
        },
    )


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_required_project(self) -> None:
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_format_is_csv(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.format == rr.EXPORT_FORMAT_CSV

    def test_default_group_by_is_code(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.group_by == rr.GROUP_BY_CODE

    def test_repeatable_filters(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(
            [
                "--project",
                "abcdef012345",
                "--code",
                "111111111111",
                "--code",
                "222222222222",
            ]
        )
        assert ns.code == ["111111111111", "222222222222"]

    def test_filters_default_to_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.code is None
        assert ns.source is None
        assert ns.coder is None
        assert ns.participant is None


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestStdoutDispatch:
    def test_csv_to_stdout(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("application_id,code_id,code_name")
        # Two applications were saved; both should appear in the CSV.
        body = list(csv.DictReader(io.StringIO(out)))
        assert len(body) == 2
        names = {row["code_name"] for row in body}
        assert names == {"Pacing", "Resting"}

    def test_markdown_to_stdout(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "markdown",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("# Coded segments")
        # Project name surfaces in the title.
        assert "Pilot study" in out
        # Default group_by=code → both code names render as headings.
        assert "## Pacing" in out
        assert "## Resting" in out

    def test_md_alias(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "md",
            ]
        )
        assert rc == 0
        assert capsys.readouterr().out.startswith("# Coded segments")

    def test_rtf_to_stdout(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "rtf",
            ]
        )
        assert rc == 0
        assert capsys.readouterr().out.startswith(r"{\rtf1")

    def test_group_by_source(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "markdown",
                "--group-by",
                "source",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Interview 01" in out
        assert "## Interview 02" in out

    def test_group_by_participant(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "markdown",
                "--group-by",
                "participant",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Alice" in out
        assert "## Bob" in out


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


class TestFilters:
    def test_filter_by_code(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, ids = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--code",
                ids["pacing"],
            ]
        )
        assert rc == 0
        body = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        assert len(body) == 1
        assert body[0]["code_name"] == "Pacing"

    def test_filter_by_source_and_coder_combined(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, ids = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--source",
                ids["interview-1"],
                "--coder",
                ids["luke"],
            ]
        )
        assert rc == 0
        body = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        assert len(body) == 1
        assert body[0]["coder_name"] == "Luke"

    def test_filter_by_participant(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, ids = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--participant",
                ids["alice"],
            ]
        )
        assert rc == 0
        body = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        # Alice is linked to interview-1 only → row a1 only.
        assert len(body) == 1
        assert body[0]["source_name"] == "Interview 01"

    def test_unknown_filter_id_returns_zero_rows(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--code",
                "0" * 12,
            ]
        )
        assert rc == 0
        body = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        assert body == []


# --------------------------------------------------------------------------- #
# File output
# --------------------------------------------------------------------------- #


class TestFileDispatch:
    def test_csv_to_file(
        self,
        tmp_path: Path,
        project_with_applications: tuple[Path, str, dict[str, str]],
    ) -> None:
        root, pid, _ = project_with_applications
        out_path = tmp_path / "segments.csv"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--out",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.exists()
        body = out_path.read_bytes().decode("utf-8")
        assert body.startswith("application_id,")
        assert "Pacing" in body

    def test_markdown_to_file_with_group_source(
        self,
        tmp_path: Path,
        project_with_applications: tuple[Path, str, dict[str, str]],
    ) -> None:
        root, pid, _ = project_with_applications
        out_path = tmp_path / "segments.md"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "markdown",
                "--group-by",
                "source",
                "--out",
                str(out_path),
            ]
        )
        assert rc == 0
        body = out_path.read_text(encoding="utf-8")
        assert "## Interview 01" in body

    def test_word_alias_writes_rtf(
        self,
        tmp_path: Path,
        project_with_applications: tuple[Path, str, dict[str, str]],
    ) -> None:
        root, pid, _ = project_with_applications
        out_path = tmp_path / "segments.docx"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "word",
                "--out",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.read_text(encoding="utf-8").startswith(r"{\rtf1")

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        project_with_applications: tuple[Path, str, dict[str, str]],
    ) -> None:
        root, pid, _ = project_with_applications
        out_path = tmp_path / "nested" / "deep" / "segments.csv"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--out",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.exists()

    def test_file_dispatch_logs_count_to_stderr(
        self,
        tmp_path: Path,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        out_path = tmp_path / "segments.csv"
        cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--out",
                str(out_path),
            ]
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        # 2 segments saved → log line includes that count.
        assert "Wrote 2 segment(s)" in captured.err


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailures:
    def test_unknown_format_returns_2(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "yaml",
            ]
        )
        assert rc == 2
        assert "Unsupported retrieval-report format" in capsys.readouterr().err

    def test_unknown_group_by_returns_2(
        self,
        project_with_applications: tuple[Path, str, dict[str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid, _ = project_with_applications
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--group-by",
                "year",
            ]
        )
        assert rc == 2
        assert "Unsupported group_by" in capsys.readouterr().err

    def test_missing_project_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                "0" * 12,
            ]
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err.lower()

    def test_invalid_project_id_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                "not-hex",
            ]
        )
        assert rc == 2
        assert capsys.readouterr().err.strip() != ""


class TestEmptyProject:
    def test_empty_csv_is_header_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        project = _projects.Project.new(name="Empty")
        _projects.save_project(root, project)
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                project.id,
                "--format",
                "csv",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("application_id,")
        # No data rows.
        assert list(csv.DictReader(io.StringIO(out))) == []

    def test_empty_markdown_renders_placeholder(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        project = _projects.Project.new(name="Empty")
        _projects.save_project(root, project)
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                project.id,
                "--format",
                "markdown",
            ]
        )
        assert rc == 0
        assert "_(no coded segments)_" in capsys.readouterr().out
