"""Tests for ``scribe.scripts.export_codebook`` (F6.1 CLI surface).

The CLI is a thin wrapper around :mod:`scribe.codebook_export` and
:mod:`scribe.codes`. Exercise: argument parsing, format dispatch,
stdout vs ``--out``, and the failure modes (unknown format, missing
project, invalid project id).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from scribe import codebook_export as ce
from scribe import codes as _codes
from scribe import projects as _projects
from scribe.codes import Code
from scribe.scripts import export_codebook as cli


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_with_codes(tmp_path: Path) -> tuple[Path, str]:
    """Create a project + a couple of codes on disk, return the
    (projects_root, project_id) tuple."""
    root = tmp_path / "projects"
    root.mkdir()
    project = _projects.Project.new(name="Pilot")
    _projects.save_project(root, project)
    _codes.save_code(root, Code.new(project_id=project.id, name="Pacing"))
    _codes.save_code(root, Code.new(project_id=project.id, name="Resting"))
    return root, project.id


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
        assert ns.format == ce.EXPORT_FORMAT_CSV

    def test_default_projects_root(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.projects_root == Path("projects")

    def test_out_defaults_to_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.out is None


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestStdoutDispatch:
    def test_csv_to_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
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
        assert out.startswith("id,name,definition")
        assert "Pacing" in out
        assert "Resting" in out

    def test_markdown_to_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
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
        assert out.startswith("# Codebook")
        assert "Pilot" in out  # project name renders in the header
        assert "## Pacing" in out

    def test_md_alias_to_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
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
        out = capsys.readouterr().out
        assert out.startswith("# Codebook")

    def test_rtf_to_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
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
        out = capsys.readouterr().out
        assert out.startswith(r"{\rtf1")


class TestFileDispatch:
    def test_csv_to_file(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.csv"
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
        assert body.startswith("id,name,definition")
        assert "Pacing" in body

    def test_markdown_to_file(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.md"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "markdown",
                "--out",
                str(out_path),
            ]
        )
        assert rc == 0
        body = out_path.read_text(encoding="utf-8")
        assert body.startswith("# Codebook")

    def test_word_alias_writes_rtf_body(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.docx"
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
        body = out_path.read_text(encoding="utf-8")
        assert body.startswith(r"{\rtf1")

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "nested" / "deep" / "codebook.csv"
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

    def test_file_dispatch_logs_to_stderr(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.csv"
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
        # Stdout is empty when writing to a file; stderr carries the
        # "wrote N codes" log line.
        assert captured.out == ""
        assert "Wrote" in captured.err
        assert str(out_path) in captured.err


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailures:
    def test_unknown_format_returns_2(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
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
        captured = capsys.readouterr()
        assert "Unsupported codebook export format" in captured.err

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
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

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
        captured = capsys.readouterr()
        # Either "invalid project id" or "not found" — both are valid
        # signals to the user; we just want a non-zero exit + a hint.
        assert captured.err.strip() != ""


class TestEmptyCodebook:
    def test_empty_csv(
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
        assert out.startswith("id,name,definition")
        # Header only — no data rows.
        assert "\nPacing" not in out

    def test_empty_markdown(
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
        out = capsys.readouterr().out
        assert "_(empty codebook)_" in out
