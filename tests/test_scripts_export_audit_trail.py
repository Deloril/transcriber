"""Tests for ``scribe.scripts.export_audit_trail`` (F9.7 CLI surface).

The CLI is a thin wrapper around :mod:`scribe.audit_export`. Exercise:
argument parsing, format dispatch, stdout vs ``--out``, filter
forwarding, and the failure modes (unknown format, missing project,
invalid filter values).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from scribe import audit_export as ae
from scribe import projects as _projects
from scribe.event_log import (
    EVENT_ACTION_CREATE,
    EVENT_ENTITY_CODE,
    record_event,
)
from scribe.scripts import export_audit_trail as cli


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def project_with_events(tmp_path: Path) -> tuple[Path, str]:
    """Create a project + an event on disk; return (root, project_id)."""
    root = tmp_path / "projects"
    root.mkdir()
    project = _projects.Project.new(name="Pilot")
    _projects.save_project(root, project)
    record_event(
        root,
        project_id=project.id,
        action=EVENT_ACTION_CREATE,
        entity_type=EVENT_ENTITY_CODE,
        entity_id="111111111111",
        after={"name": "Resting"},
        now="2026-05-26T10:00:00Z",
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

    def test_default_format_is_markdown(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--project", "abcdef012345"])
        assert args.format == ae.EXPORT_FORMAT_MARKDOWN

    def test_kind_can_repeat(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--project",
                "abcdef012345",
                "--kind",
                "event",
                "--kind",
                "ai_invocation",
            ]
        )
        assert args.kind == ["event", "ai_invocation"]

    def test_filters_default_to_none(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--project", "abcdef012345"])
        assert args.since is None
        assert args.until is None
        assert args.actor is None
        assert args.entity_type is None
        assert args.action is None
        assert args.feature is None
        assert args.decision is None
        assert args.kind is None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


class TestMainStdout:
    def test_writes_csv_to_stdout(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
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
        captured = capsys.readouterr()
        assert ",".join(ae.CSV_COLUMNS) in captured.out
        assert "111111111111" in captured.out

    def test_writes_markdown_to_stdout(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
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
        assert "# Audit trail — Pilot" in out
        assert "## 2026-05-26" in out

    def test_format_alias_word_routes_to_rtf(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "word",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith(r"{\rtf1")


class TestMainOutFile:
    def test_writes_to_file_and_logs_count(
        self,
        project_with_events: tuple[Path, str],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
        out = tmp_path / "audit.md"
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "md",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        body = out.read_text()
        assert "# Audit trail" in body
        # Stderr carries the row-count log; stdout is empty.
        captured = capsys.readouterr()
        assert "Wrote 1 row" in captured.err
        assert captured.out == ""


class TestMainFiltersForwarded:
    def test_kind_filter_drops_other_source(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--kind",
                "ai_invocation",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # Header always present; the one event row is now filtered out.
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 1  # just the header

    def test_since_until_drops_out_of_range(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--format",
                "csv",
                "--since",
                "2030-01-01T00:00:00Z",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "111111111111" not in out

    def test_invalid_action_returns_2(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                pid,
                "--action",
                "no-such-action",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "action" in err.lower()


class TestMainErrors:
    def test_unknown_format_returns_2(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, pid = project_with_events
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
        assert "yaml" in capsys.readouterr().err.lower()

    def test_invalid_project_id_returns_2(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, _ = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                "not-hex",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "project id" in err.lower()

    def test_missing_project_returns_2(
        self,
        project_with_events: tuple[Path, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root, _ = project_with_events
        rc = cli.main(
            [
                "--projects-root",
                str(root),
                "--project",
                "abcdef012345",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err.lower()
