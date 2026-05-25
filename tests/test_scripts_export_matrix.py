"""Tests for ``scribe.scripts.export_matrix`` (F6.3 CLI surface).

The CLI is a thin wrapper around :mod:`scribe.matrix_export`,
:mod:`scribe.matrix`, and the F1.* / F2.* / F3.* persistence layers.
Exercise: argument parsing, kind dispatch, format dispatch, stdout vs
``--out``, totals / titles / compact toggles, and the failure modes
(unknown format, unknown kind, missing project, missing
``--attribute-key`` for the cross-tab kind).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pytest

from scribe import applications as _applications
from scribe import codes as _codes
from scribe import matrix_export as me
from scribe import participants as _participants
from scribe import projects as _projects
from scribe import sources as _sources
from scribe.applications import Application
from scribe.codes import Code
from scribe.participants import Participant
from scribe.scripts import export_matrix as cli
from scribe.sources import Source


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12


def _save_app(
    root: Path,
    *,
    project_id: str,
    code_id: str,
    source_id: str,
    coder_id: str = _HEX_CODER,
    start_word: int = 0,
    end_word: int = 0,
) -> Application:
    """Persist a minimal-but-valid Application; return it.

    The matrix CLI only consults ``code_id`` / ``source_id`` for the
    code-by-source kind (and code-by-attribute via the source / participant
    join). All other fields are filled with shape-valid placeholders so
    the F4.1 persistence-layer validators are happy.
    """
    app = Application.new(
        project_id=project_id,
        source_id=source_id,
        code_id=code_id,
        coder_id=coder_id,
        anchor_start_word_id=f"s0w{start_word}",
        anchor_end_word_id=f"s0w{end_word}",
        definition_version_id_at_apply=_HEX_VERSION,
    )
    _applications.save_application(root, app)
    return app


@pytest.fixture
def project_with_matrix_data(tmp_path: Path) -> tuple[Path, str]:
    """Project with two codes, two sources, three applications.

    Resulting code-by-source matrix:

        Pacing  | 2 | 0
        Resting | 0 | 1
    """
    root = tmp_path / "projects"
    root.mkdir()
    project = _projects.Project.new(name="Pilot Study")
    _projects.save_project(root, project)

    pacing = Code.new(project_id=project.id, name="Pacing")
    resting = Code.new(project_id=project.id, name="Resting")
    _codes.save_code(root, pacing)
    _codes.save_code(root, resting)

    int1 = Source.new(project_id=project.id, name="Interview 1")
    int2 = Source.new(project_id=project.id, name="Interview 2")
    _sources.save_source(root, int1)
    _sources.save_source(root, int2)

    _save_app(
        root,
        project_id=project.id,
        code_id=pacing.id,
        source_id=int1.id,
        start_word=0,
        end_word=2,
    )
    _save_app(
        root,
        project_id=project.id,
        code_id=pacing.id,
        source_id=int1.id,
        start_word=4,
        end_word=6,
    )
    _save_app(
        root,
        project_id=project.id,
        code_id=resting.id,
        source_id=int2.id,
        start_word=0,
        end_word=2,
    )

    return root, project.id


@pytest.fixture
def project_with_attribute_data(tmp_path: Path) -> tuple[Path, str]:
    """Project with source + participant attributes for cross-tab tests."""
    root = tmp_path / "projects"
    root.mkdir()
    project = _projects.Project.new(name="Attr Project")
    _projects.save_project(root, project)

    code = Code.new(project_id=project.id, name="Coping")
    _codes.save_code(root, code)

    s1 = Source.new(
        project_id=project.id,
        name="Site A",
        custom_attributes={"site": "alpha"},
    )
    s2 = Source.new(
        project_id=project.id,
        name="Site B",
        custom_attributes={"site": "beta"},
    )
    _sources.save_source(root, s1)
    _sources.save_source(root, s2)

    p1 = Participant.new(
        project_id=project.id,
        name="Alice",
        demographics={"role": "interviewer"},
    )
    p2 = Participant.new(
        project_id=project.id,
        name="Bob",
        demographics={"role": "participant"},
    )
    _participants.save_participant(root, p1)
    _participants.save_participant(root, p2)

    # No speaker map persisted: even if there were one, F4.1
    # Application doesn't surface a speaker label, so the CLI's
    # participant-attribute path will land everything in the "(missing)"
    # bucket without an upstream hydration pass.

    # F4.1 Application doesn't carry a ``speaker`` field — that
    # comes from the editor's word stream and is hydrated by the
    # server before passing applications into the matrix builder.
    # The CLI test deliberately exercises the un-hydrated path: both
    # apps fall into the "(missing)" bucket on the participant axis.
    _save_app(
        root,
        project_id=project.id,
        code_id=code.id,
        source_id=s1.id,
        start_word=0,
        end_word=2,
    )
    _save_app(
        root,
        project_id=project.id,
        code_id=code.id,
        source_id=s2.id,
        start_word=0,
        end_word=2,
    )
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
        assert ns.format == me.EXPORT_FORMAT_CSV

    def test_default_kind_is_code_by_source(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.kind == me.MATRIX_KIND_CODE_BY_SOURCE

    def test_default_scope_is_source(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(["--project", "abcdef012345"])
        assert ns.scope == "source"

    def test_default_attribute_kind_choices(self) -> None:
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--project",
                    "abcdef012345",
                    "--attribute-kind",
                    "rocket",
                ]
            )

    def test_no_totals_is_a_flag(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(
            ["--project", "abcdef012345", "--no-totals"]
        )
        assert ns.no_totals is True

    def test_compact_is_a_flag(self) -> None:
        parser = cli.build_parser()
        ns = parser.parse_args(
            ["--project", "abcdef012345", "--compact"]
        )
        assert ns.compact is True


# --------------------------------------------------------------------------- #
# Stdout dispatch
# --------------------------------------------------------------------------- #


class TestStdoutDispatch:
    def test_csv_to_stdout(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_matrix_data
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
        out = capsysbinary.readouterr().out.decode()
        assert out.startswith("Code × Source,")
        # Pacing row, two applications in Interview 1.
        # Body must have row labels and at least one nonzero cell.
        assert "Pacing" in out
        assert "Resting" in out

    def test_xlsx_to_stdout_is_zip(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "xlsx",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out
        assert out[:2] == b"PK"
        with zipfile.ZipFile(io.BytesIO(out)) as zf:
            assert "xl/worksheets/sheet1.xml" in zf.namelist()


# --------------------------------------------------------------------------- #
# --out file dispatch
# --------------------------------------------------------------------------- #


class TestFileDispatch:
    def test_csv_out_writes_file(
        self,
        project_with_matrix_data: tuple[Path, str],
        tmp_path: Path,
    ) -> None:
        root, pid = project_with_matrix_data
        target = tmp_path / "freq.csv"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--out",
                str(target),
            ]
        )
        assert rc == 0
        body = target.read_text()
        assert "Pacing" in body
        assert "Resting" in body

    def test_xlsx_out_writes_zip(
        self,
        project_with_matrix_data: tuple[Path, str],
        tmp_path: Path,
    ) -> None:
        root, pid = project_with_matrix_data
        target = tmp_path / "freq.xlsx"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "xlsx",
                "--out",
                str(target),
            ]
        )
        assert rc == 0
        body = target.read_bytes()
        assert body[:2] == b"PK"
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            assert "xl/worksheets/sheet1.xml" in zf.namelist()

    def test_summary_logged_to_stderr(
        self,
        project_with_matrix_data: tuple[Path, str],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--out",
                str(tmp_path / "x.csv"),
            ]
        )
        assert rc == 0
        err = capsys.readouterr().err
        # 2 codes × 2 sources matrix.
        assert "2×2" in err
        assert "Pilot Study" in err


# --------------------------------------------------------------------------- #
# Matrix kinds
# --------------------------------------------------------------------------- #


class TestKindDispatch:
    def test_code_by_code_via_alias(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "cooccurrence",
                "--format",
                "csv",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        # The matrix corner cell carries the F3.6 default title.
        first_line = out.splitlines()[0]
        assert first_line.startswith("Code × Code")

    def test_code_by_attribute_source(
        self,
        project_with_attribute_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_attribute_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "code-by-attribute",
                "--attribute-key",
                "site",
                "--attribute-kind",
                "source",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        # alpha + beta should appear as column labels.
        assert "alpha" in out
        assert "beta" in out

    def test_code_by_attribute_participant_no_speaker_data(
        self,
        project_with_attribute_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        # The F4.1 Application persistence layer doesn't carry a
        # speaker label, so the CLI can't resolve participants without
        # a hydration pass. All applications land in the "(missing)"
        # bucket — assert exactly that.
        root, pid = project_with_attribute_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "code-by-attribute",
                "--attribute-key",
                "role",
                "--attribute-kind",
                "participant",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        assert "(missing)" in out

    def test_code_by_attribute_participant_no_missing(
        self,
        project_with_attribute_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        # ``--no-include-missing`` drops the missing column; with no
        # speaker data on the apps that means the matrix has zero
        # populated columns. Header row is still emitted so the file
        # remains a valid CSV.
        root, pid = project_with_attribute_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "code-by-attribute",
                "--attribute-key",
                "role",
                "--attribute-kind",
                "participant",
                "--no-include-missing",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        assert "(missing)" not in out

    def test_code_by_attribute_requires_key(
        self,
        project_with_attribute_data: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_attribute_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "code-by-attribute",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "--attribute-key is required" in err


# --------------------------------------------------------------------------- #
# Toggles
# --------------------------------------------------------------------------- #


class TestToggles:
    def test_no_titles_uses_keys(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--no-titles",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        assert "Pacing" not in out
        assert "Resting" not in out
        # Code ids should appear instead.
        assert re.search(r"^[0-9a-f]{12},", out, re.M)

    def test_no_totals_drops_total_columns(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--no-totals",
            ]
        )
        assert rc == 0
        out = capsysbinary.readouterr().out.decode()
        # No "Total" footer column.
        assert "Total" not in out.splitlines()[0]

    def test_compact_drops_empty_rows(
        self,
        tmp_path: Path,
        capsysbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        project = _projects.Project.new(name="Compact")
        _projects.save_project(root, project)
        c1 = Code.new(project_id=project.id, name="Used")
        c2 = Code.new(project_id=project.id, name="Unused")
        _codes.save_code(root, c1)
        _codes.save_code(root, c2)
        s1 = Source.new(project_id=project.id, name="Only")
        _sources.save_source(root, s1)
        _save_app(
            root,
            project_id=project.id,
            code_id=c1.id,
            source_id=s1.id,
            start_word=0,
            end_word=2,
        )

        # Without --compact the unused code shows up.
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                project.id,
            ]
        )
        assert rc == 0
        without = capsysbinary.readouterr().out.decode()
        assert "Unused" in without

        # With --compact it doesn't.
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                project.id,
                "--compact",
            ]
        )
        assert rc == 0
        compact = capsysbinary.readouterr().out.decode()
        assert "Unused" not in compact
        assert "Used" in compact


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_unknown_format(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(
            [
                "--projects-root",
                str(tmp_path),
                "--project",
                "abcdef012345",
                "--format",
                "pdf",
            ]
        )
        assert rc == 2
        assert "Unsupported matrix export format" in capsys.readouterr().err

    def test_unknown_kind(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(
            [
                "--projects-root",
                str(tmp_path),
                "--project",
                "abcdef012345",
                "--kind",
                "frequency-by-quarter",
            ]
        )
        assert rc == 2
        assert "Unsupported matrix kind" in capsys.readouterr().err

    def test_missing_project_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(
            [
                "--projects-root",
                str(tmp_path),
                "--project",
                "abcdef012345",
            ]
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_project_id_returns_2(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(
            [
                "--projects-root",
                str(tmp_path),
                "--project",
                "not-hex",
            ]
        )
        assert rc == 2
        # Invalid id error message routes through ProjectValidationError.
        assert "invalid project id" in capsys.readouterr().err.lower()

    def test_bad_scope_returns_2(
        self,
        project_with_matrix_data: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_matrix_data
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--kind",
                "code-by-code",
                "--scope",
                "galaxy",
            ]
        )
        assert rc == 2
        assert "scope" in capsys.readouterr().err.lower()
