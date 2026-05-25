"""Tests for ``scribe.scripts.export_codebook_refi_qda_xml`` (F6.5 CLI).

The CLI is a thin wrapper around
:func:`scribe.codebook_export.render_refi_qda_codebook_xml` and the
sibling :func:`scribe.codebook_export.write_refi_qda_codebook_xml`.
Exercise: argument parsing, stdout-vs-``--out`` dispatch, the
project-metadata comment, custom ``--origin`` propagation, and the
failure modes (missing project, invalid project id).
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scribe import codebook_export as ce
from scribe import codes as _codes
from scribe import projects as _projects
from scribe.codes import Code
from scribe.codebook_export import REFI_QDA_NS
from scribe.scripts import export_codebook_refi_qda_xml as cli


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_with_codes(tmp_path: Path) -> tuple[Path, str]:
    """Create a project + a couple of codes on disk, return the
    (projects_root, project_id) tuple."""
    root = tmp_path / "projects"
    root.mkdir()
    project = _projects.Project.new(
        name="Pilot",
        methodology="charmaz",
        research_question="How do people pace energy?",
    )
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

    def test_default_projects_root(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.projects_root == Path("projects")

    def test_out_defaults_to_none(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.out is None

    def test_default_origin(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.origin == ce.REFI_QDA_ORIGIN_DEFAULT

    def test_custom_origin(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(
            ["--project", "abcdef012345", "--origin", "Scribe 1.2"]
        )
        assert ns.origin == "Scribe 1.2"


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestStdoutDispatch:
    def test_xml_to_stdout(
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
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("<?xml")
        # Round-trip parses cleanly.
        root_el = ET.fromstring(out)
        assert root_el.tag == f"{{{REFI_QDA_NS}}}CodeBook"

    def test_codes_emit_in_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
        cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
            ]
        )
        out = capsys.readouterr().out
        assert "Pacing" in out
        assert "Resting" in out

    def test_project_metadata_comment_in_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
        cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
            ]
        )
        out = capsys.readouterr().out
        assert "<!--" in out
        assert "Methodology: charmaz" in out
        assert "How do people pace energy?" in out

    def test_custom_origin_in_stdout(
        self,
        project_with_codes: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_codes
        cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--origin",
                "Scribe 1.2",
            ]
        )
        out = capsys.readouterr().out
        root_el = ET.fromstring(out)
        assert root_el.get("origin") == "Scribe 1.2"


class TestFileDispatch:
    def test_writes_xml_to_file(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.refi-qda.xml"
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
        body = out_path.read_text(encoding="utf-8")
        assert body.startswith("<?xml")
        # Round-trip parses cleanly.
        ET.fromstring(body)

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "nested" / "deep" / "codebook.refi-qda.xml"
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
        out_path = tmp_path / "codebook.refi-qda.xml"
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

    def test_file_body_includes_metadata_comment(
        self,
        tmp_path: Path,
        project_with_codes: tuple[Path, str],
    ) -> None:
        root, pid = project_with_codes
        out_path = tmp_path / "codebook.refi-qda.xml"
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
        text = out_path.read_text(encoding="utf-8")
        assert "<!--" in text
        assert "Methodology: charmaz" in text


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailures:
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
    def test_empty_codebook_xml(
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
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # Valid empty codebook — <Codes/> with no children.
        root_el = ET.fromstring(out)
        codes_el = root_el.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 0
