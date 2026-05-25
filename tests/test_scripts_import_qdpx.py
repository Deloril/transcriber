"""Tests for scribe.scripts.import_qdpx (F6.6 CLI)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from scribe.applications import Application, list_applications
from scribe.code_versions import (
    CodeVersion,
    code_versions_path,
    read_code_versions,
)
from scribe.coders import Coder, list_coders
from scribe.codes import Code, list_codes
from scribe.memos import Memo, list_memos
from scribe.projects import Project, list_projects, load_project, project_dir
from scribe.refi_qda_project import (
    REFI_QDA_PROJECT_NS,
    render_source_plain_text,
    to_qdpx,
)
from scribe.scripts.import_qdpx import (
    build_parser,
    main,
    persist_import_result,
)
from scribe.sources import Source, list_sources


FIXED_NOW = "2024-03-04T05:06:07.000000Z"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _archive_with_full_project() -> tuple[bytes, dict]:
    """Build a Scribe-origin QDPX archive that includes one of every entity."""
    p = Project.new(name="Test", methodology="charmaz", now=FIXED_NOW)
    s = Source.new(
        project_id=p.id, name="Interview", transcript_job_id="abcdef012345",
        now=FIXED_NOW,
    )
    c = Code.new(
        project_id=p.id, name="Pacing",
        definition="Adjusting daily activity to manage energy.",
        now=FIXED_NOW,
    )
    cd = Coder.new(project_id=p.id, name="Luke", now=FIXED_NOW)
    v1 = CodeVersion.new(code=c, version=1, now=FIXED_NOW)
    segs = [
        {"speaker": "INT", "words": [
            {"text": "How"}, {"text": "do"}, {"text": "you"},
            {"text": "manage?"},
        ]},
        {"speaker": "P3", "words": [{"text": "I"}, {"text": "pace."}]},
    ]
    r = render_source_plain_text(s.id, segs)
    a = Application.new(
        project_id=p.id, code_id=c.id, source_id=s.id, coder_id=cd.id,
        anchor_start_word_id="s1w0", anchor_end_word_id="s1w1",
        definition_version_id_at_apply=v1.id, now=FIXED_NOW,
    )
    m = Memo.new(
        project_id=p.id, type="theoretical",
        title="Pacing memo", body="Pacing keeps coming up.", now=FIXED_NOW,
    )
    archive = to_qdpx(
        project=p, sources=[s], codes=[c], coders=[cd],
        applications=[a], memos=[m], rendered_sources=[r],
    )
    return archive, {
        "project": p, "source": s, "code": c, "coder": cd,
        "application": a, "memo": m,
    }


# --------------------------------------------------------------------------- #
# build_parser
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_parser_requires_archive(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--projects-root", "p"])

    def test_parser_defaults_projects_root(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--archive", "x.qdpx"])
        assert args.projects_root.name == "projects"


# --------------------------------------------------------------------------- #
# persist_import_result
# --------------------------------------------------------------------------- #


class TestPersistImportResult:
    def test_writes_project_and_entities(self, tmp_path) -> None:
        from scribe.refi_qda_import import import_qdpx
        archive, originals = _archive_with_full_project()
        result = import_qdpx(archive, now=FIXED_NOW)
        written = persist_import_result(tmp_path, result)
        # 1 project + 1 source + 1 code + 1 coder + 1 memo + 1 version
        # + 1 application + 1 imported_sources sidecar = 8
        assert written == 8

        # Project on disk
        loaded = load_project(tmp_path, result.project.id)
        assert loaded.name == "Test"

        # Per-entity dirs populated
        assert len(list_sources(tmp_path, result.project.id)) == 1
        assert len(list_codes(tmp_path, result.project.id)) == 1
        assert len(list_coders(tmp_path, result.project.id)) == 1
        assert len(list_memos(tmp_path, result.project.id)) == 1
        assert len(list_applications(tmp_path, result.project.id)) == 1

        # CodeVersion JSONL exists with the right id
        cvp = code_versions_path(
            tmp_path, result.project.id, result.codes[0].id
        )
        assert cvp.is_file()
        versions = read_code_versions(
            tmp_path, result.project.id, result.codes[0].id
        )
        assert len(versions) == 1
        assert versions[0].id == result.code_versions[0].id

    def test_imported_sources_sidecar_layout(self, tmp_path) -> None:
        from scribe.refi_qda_import import import_qdpx
        archive, _orig = _archive_with_full_project()
        result = import_qdpx(archive, now=FIXED_NOW)
        persist_import_result(tmp_path, result)

        # Sidecar at <project>/imported_sources/<sid>.json
        sid = result.sources[0].id
        sidecar = (
            project_dir(tmp_path, result.project.id)
            / "imported_sources" / f"{sid}.json"
        )
        assert sidecar.is_file()
        data = json.loads(sidecar.read_text())
        assert data["source_id"] == sid
        assert "INT:" in data["text"]
        # Tokenised segments have the expected shape
        assert isinstance(data["segments"], list)
        assert data["segments"][0]["speaker"] == "INT"
        assert data["segments"][0]["words"][0]["word_id"] == "s0w0"

    def test_no_warnings_on_round_trip(self, tmp_path) -> None:
        from scribe.refi_qda_import import import_qdpx
        archive, _ = _archive_with_full_project()
        result = import_qdpx(archive, now=FIXED_NOW)
        assert result.warnings == []


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


class TestMain:
    def test_imports_archive_and_prints_id(self, tmp_path, capsys) -> None:
        archive, _ = _archive_with_full_project()
        archive_path = tmp_path / "study.qdpx"
        archive_path.write_bytes(archive)
        projects_root = tmp_path / "projects"

        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(archive_path),
        ])
        captured = capsys.readouterr()
        assert rc == 0

        # Stdout has exactly the new project id
        printed_id = captured.out.strip()
        assert len(printed_id) == 12
        assert all(ch in "0123456789abcdef" for ch in printed_id)

        # Project actually persisted
        projects = list_projects(projects_root)
        assert any(p.id == printed_id for p in projects)
        # Stderr has the human-readable summary
        assert "Imported project" in captured.err
        assert "1 source(s)" in captured.err

    def test_quiet_suppresses_summary(self, tmp_path, capsys) -> None:
        archive, _ = _archive_with_full_project()
        archive_path = tmp_path / "study.qdpx"
        archive_path.write_bytes(archive)
        projects_root = tmp_path / "projects"

        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(archive_path),
            "--quiet",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        # Stderr is silent except possibly a trailing newline
        assert captured.err.strip() == ""

    def test_missing_archive_returns_error(self, tmp_path, capsys) -> None:
        projects_root = tmp_path / "projects"
        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(tmp_path / "nope.qdpx"),
        ])
        captured = capsys.readouterr()
        assert rc == 2
        assert "not found" in captured.err

    def test_invalid_archive_returns_error(self, tmp_path, capsys) -> None:
        projects_root = tmp_path / "projects"
        bad = tmp_path / "bad.qdpx"
        bad.write_bytes(b"not a zip archive")
        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(bad),
        ])
        captured = capsys.readouterr()
        assert rc == 2
        assert "error" in captured.err.lower()

    def test_zip_without_project_qde_returns_error(self, tmp_path, capsys) -> None:
        projects_root = tmp_path / "projects"
        bad = tmp_path / "bad.qdpx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("Sources/foo.txt", "hello")
        bad.write_bytes(buf.getvalue())
        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(bad),
        ])
        captured = capsys.readouterr()
        assert rc == 2
        assert "project.qde" in captured.err

    def test_warnings_surfaced_on_stderr(self, tmp_path, capsys) -> None:
        # Build an archive whose CodeRef points at a non-existent code.
        # The importer logs a warning; the CLI should pass it through.
        qde = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Project xmlns="{REFI_QDA_PROJECT_NS}" name="Foreign">\n'
            f'  <Users/>\n'
            f'  <CodeBook><Codes/></CodeBook>\n'
            f'  <Sources>\n'
            f'    <TextSource guid="11111111-1111-1111-1111-111111111111" '
            f'name="src" plainTextPath="internal://Sources/sample.txt">\n'
            f'      <PlainTextSelection guid="22222222-2222-2222-2222-222222222222" '
            f'startPosition="0" endPosition="5">\n'
            f'        <Coding>\n'
            f'          <CodeRef targetGUID="ffffffff-ffff-ffff-ffff-ffffffffffff"/>\n'
            f'        </Coding>\n'
            f'      </PlainTextSelection>\n'
            f'    </TextSource>\n'
            f'  </Sources>\n'
            f'</Project>\n'
        )
        archive_path = tmp_path / "foreign.qdpx"
        with zipfile.ZipFile(archive_path, mode="w") as zf:
            zf.writestr("project.qde", qde)
            zf.writestr("Sources/sample.txt", "Hello world")

        projects_root = tmp_path / "projects"
        rc = main([
            "--projects-root", str(projects_root),
            "--archive", str(archive_path),
        ])
        captured = capsys.readouterr()
        assert rc == 0
        assert "warning" in captured.err.lower()
        assert "unknown code" in captured.err
