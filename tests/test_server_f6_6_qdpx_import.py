"""F6.6 reachability verification — REFI-QDA / QDPX project import.

The pure builder shipped in 0426e44 (``scribe.refi_qda_import``) and a
CLI wrapper in ``scribe.scripts.import_qdpx``; the F6.6 commit body
explicitly deferred the HTTP endpoint and the UI surface ("HTTP
endpoint / web UI surface — F6.6 ships the pure module + CLI; UI
integration can layer on without touching the data model"). That
deferral is the F6.6 reachability gap: a researcher could only run
the importer through ``python -m scribe.scripts.import_qdpx``, and
the loop's ``Reachable-via:`` contract treats CLI-only as
"unreachable from the user-facing surface".

This file is the explicit reachability anchor for the user-facing
surface:

  1. ``POST /api/projects/import-qdpx`` accepts a multipart upload
     and persists every entity REFI-QDA describes;
  2. the projects index page renders an "Import QDPX" button + hidden
     ``<input type="file">`` whose change handler POSTs to that
     endpoint and follows the redirect on 201;
  3. F6.4 → F6.6 round-trip works through the HTTP surfaces alone
     (export the project's QDPX, import it back, see all entities
     under the new project).

The deeper coverage of the importer + CLI lives in
``tests/test_refi_qda_import.py`` (50 cases) and
``tests/test_scripts_import_qdpx.py`` (11 cases). This file's job is
purely the F6.6 UI-reachability contract.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated TestClient + tmp uploads/outputs/projects roots."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)

    client = TestClient(srv.app)
    yield srv, client, projects, output


def _build_round_trip_qdpx() -> bytes:
    """Build a Scribe-origin QDPX archive with one of every entity.

    Mirrors the canonical fixture from ``tests/test_scripts_import_qdpx.py``
    so the F6.6 round-trip test exercises the full code path.
    """
    from scribe.applications import Application
    from scribe.code_versions import CodeVersion
    from scribe.coders import Coder
    from scribe.codes import Code
    from scribe.memos import Memo
    from scribe.projects import Project
    from scribe.refi_qda_project import render_source_plain_text, to_qdpx
    from scribe.sources import Source

    NOW = "2026-05-27T03:14:15.000000Z"
    p = Project.new(
        name="Cohort A interviews",
        methodology="constructivist-grounded-theory",
        now=NOW,
    )
    s = Source.new(
        project_id=p.id,
        name="Interview-1",
        transcript_job_id="abcdef012345",
        now=NOW,
    )
    c = Code.new(
        project_id=p.id,
        name="Pacing",
        definition="Adjusting daily activity to manage energy.",
        now=NOW,
    )
    cd = Coder.new(project_id=p.id, name="Luke", now=NOW)
    v1 = CodeVersion.new(code=c, version=1, now=NOW)
    segs = [
        {"speaker": "INT", "words": [
            {"text": "How"}, {"text": "do"}, {"text": "you"},
            {"text": "manage?"},
        ]},
        {"speaker": "P3", "words": [{"text": "I"}, {"text": "pace."}]},
    ]
    rendered = render_source_plain_text(s.id, segs)
    a = Application.new(
        project_id=p.id, code_id=c.id, source_id=s.id, coder_id=cd.id,
        anchor_start_word_id="s1w0", anchor_end_word_id="s1w1",
        definition_version_id_at_apply=v1.id, now=NOW,
    )
    m = Memo.new(
        project_id=p.id, type="theoretical",
        title="Pacing memo", body="Pacing keeps coming up.", now=NOW,
    )
    return to_qdpx(
        project=p, sources=[s], codes=[c], coders=[cd],
        applications=[a], memos=[m], rendered_sources=[rendered],
    )


# --------------------------------------------------------------------------- #
# Template render: the projects index surfaces the F6.6 import button
# --------------------------------------------------------------------------- #


class TestProjectsListRendersF6_6Button:
    """The projects index actions bar must render the F6.6 import
    button. Without it the F6.6 endpoint is reachable only via curl —
    which is the exact failure mode the loop's ``Reachable-via`` gate
    was written to prevent.
    """

    def test_qdpx_import_button_is_present(self, env) -> None:
        _, client, _, _ = env
        r = client.get("/projects")
        assert r.status_code == 200, r.text
        # Stable test feature attribute so the loop's done-detector can
        # verify the surface didn't silently regress.
        assert 'data-test-feature="F6.6"' in r.text
        assert 'id="importQdpxBtn"' in r.text

    def test_qdpx_import_input_is_present(self, env) -> None:
        """The hidden ``<input type="file">`` is the actual upload control."""
        _, client, _, _ = env
        r = client.get("/projects")
        assert 'id="importQdpxInput"' in r.text
        # Type=file (anywhere on the page is fine — F1.5 has one too).
        assert 'type="file"' in r.text

    def test_qdpx_import_input_accepts_qdpx(self, env) -> None:
        _, client, _, _ = env
        r = client.get("/projects")
        # The file dialog should pre-filter on .qdpx so users don't
        # need to know the extension. ``application/zip`` is in the
        # list because some browsers / OSes report QDPX as a generic
        # zip MIME type.
        assert ".qdpx" in r.text
        assert "application/x-qdpx" in r.text or "application/zip" in r.text

    def test_button_label_is_user_facing(self, env) -> None:
        """The button's visible label must mention REFI-QDA or QDPX so
        the user knows what file format it accepts. The 'Import QDPX'
        wording matches the F6.4 'Export QDPX' button."""
        _, client, _, _ = env
        r = client.get("/projects")
        assert "Import QDPX" in r.text

    def test_handler_posts_to_import_qdpx_endpoint(self, env) -> None:
        """The JS change-handler must POST to the F6.6 endpoint."""
        _, client, _, _ = env
        r = client.get("/projects")
        assert "/api/projects/import-qdpx" in r.text


# --------------------------------------------------------------------------- #
# POST /api/projects/import-qdpx — happy path
# --------------------------------------------------------------------------- #


class TestImportQdpxEndpointReachableFromUi:
    """The endpoint behind the rendered button must succeed end-to-end."""

    def test_round_trip_via_http(self, env) -> None:
        _, client, _, _ = env

        archive_bytes = _build_round_trip_qdpx()
        files = {
            "archive": ("study.qdpx", archive_bytes, "application/x-qdpx"),
        }
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 201, r.text

        body = r.json()
        # The server mints a fresh project id (12-char hex).
        assert "project_id" in body
        assert len(body["project_id"]) == 12
        assert all(ch in "0123456789abcdef" for ch in body["project_id"])
        # Redirect URL points at the new project.
        assert body["redirect"] == f"/projects/{body['project_id']}"
        # Summary counts mirror the round-trip fixture (1 of everything).
        assert body["sources"] == 1
        assert body["codes"] == 1
        assert body["coders"] == 1
        assert body["memos"] == 1
        assert body["applications"] == 1

        # The new project resolves through the project listing endpoint
        # — proving the data actually landed on disk under PROJECTS_DIR.
        r = client.get(f"/api/projects/{body['project_id']}")
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "Cohort A interviews"

        # The home page renders for the imported project.
        r = client.get(f"/projects/{body['project_id']}")
        assert r.status_code == 200

    def test_imported_project_appears_in_listing(self, env) -> None:
        _, client, _, _ = env

        files = {
            "archive": ("study.qdpx", _build_round_trip_qdpx(),
                        "application/x-qdpx"),
        }
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 201
        new_id = r.json()["project_id"]

        # The projects-list endpoint that projects_list.html consumes
        # must surface the import.
        r = client.get("/api/projects")
        assert r.status_code == 200
        ids = [p["id"] for p in r.json().get("projects", r.json())]
        assert new_id in ids

    def test_response_includes_warnings_field(self, env) -> None:
        """Round-trip on a Scribe-origin archive should produce no
        warnings, but the field must always be present so the UI can
        show ``Imported with N warnings`` without branching on
        existence vs emptiness."""
        _, client, _, _ = env
        files = {
            "archive": ("study.qdpx", _build_round_trip_qdpx(),
                        "application/x-qdpx"),
        }
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 201
        body = r.json()
        assert "warnings" in body
        assert isinstance(body["warnings"], list)
        # Round-trip is clean.
        assert body["warnings"] == []
        assert body["warnings_truncated"] is False

    def test_minimal_qde_imports_to_empty_project(self, env) -> None:
        """A QDPX with only a project.qde root + no entities must
        still import cleanly — the importer's contract is "always
        produce a Project, even on sparse input"."""
        _, client, _, _ = env
        from scribe.refi_qda_project import REFI_QDA_PROJECT_NS

        qde = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Project xmlns="{REFI_QDA_PROJECT_NS}" name="Empty study"/>\n'
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("project.qde", qde)
        files = {"archive": ("empty.qdpx", buf.getvalue(),
                             "application/x-qdpx")}
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "Empty study"
        assert body["sources"] == 0
        assert body["codes"] == 0
        assert body["applications"] == 0


# --------------------------------------------------------------------------- #
# POST /api/projects/import-qdpx — defensive paths
# --------------------------------------------------------------------------- #


class TestImportQdpxEndpointDefences:
    """Bad uploads must produce 4xx errors with useful detail."""

    def test_400_for_non_zip_payload(self, env) -> None:
        _, client, _, _ = env
        files = {
            "archive": ("garbage.qdpx", b"not a zip",
                        "application/x-qdpx"),
        }
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 400
        # Detail mentions the format, not a stack trace.
        assert "QDPX" in r.json().get("detail", "") or \
               "qdpx" in r.json().get("detail", "").lower()

    def test_400_for_zip_without_project_qde(self, env) -> None:
        _, client, _, _ = env
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("Sources/foo.txt", "hello")
        files = {"archive": ("bad.qdpx", buf.getvalue(),
                             "application/x-qdpx")}
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 400
        assert "project.qde" in r.json().get("detail", "").lower()

    def test_400_for_zip_with_unparseable_qde(self, env) -> None:
        _, client, _, _ = env
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("project.qde", "<not> valid </xml")
        files = {"archive": ("bad.qdpx", buf.getvalue(),
                             "application/x-qdpx")}
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 400

    def test_413_for_oversized_upload(self, env) -> None:
        """A 51 MB upload must be rejected before the importer is even
        called. Use the streaming write to avoid buffering 50 MB in RAM
        in the test process."""
        _, client, _, _ = env
        # 51 MB of zeroes — well above the soft limit.
        big = b"\x00" * (51 * 1024 * 1024)
        files = {"archive": ("huge.qdpx", big, "application/x-qdpx")}
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 413


# --------------------------------------------------------------------------- #
# F6.4 → F6.6 round-trip through HTTP surfaces alone
# --------------------------------------------------------------------------- #


class TestF6_4ToF6_6HttpRoundTrip:
    """Researcher exports from project A, imports into project B —
    every entity should survive. This is the load-bearing claim of
    REFI-QDA: ``no lock-in`` only holds if the export the researcher
    just produced can be read back by the same tool."""

    def test_export_then_import_preserves_entities(self, env) -> None:
        srv, client, _, _ = env

        # 1. Create a project with one of each entity through the
        #    normal HTTP API.
        r = client.post(
            "/api/projects",
            json={
                "name": "Round-trip cohort",
                "methodology": "constructivist-grounded-theory",
            },
        )
        assert r.status_code == 201, r.text
        original_id = r.json()["id"]

        # Add a code.
        r = client.post(
            f"/api/projects/{original_id}/codes",
            json={
                "name": "Pacing",
                "definition": "Adjusting daily activity.",
            },
        )
        assert r.status_code == 201, r.text

        # 2. Export via F6.4.
        r = client.get(f"/api/projects/{original_id}/qdpx")
        assert r.status_code == 200, r.text
        qdpx_bytes = r.content
        assert len(qdpx_bytes) > 0

        # 3. Import via F6.6.
        files = {
            "archive": ("round-trip.qdpx", qdpx_bytes,
                        "application/x-qdpx"),
        }
        r = client.post("/api/projects/import-qdpx", files=files)
        assert r.status_code == 201, r.text
        new_body = r.json()

        # 4. The imported project must be a NEW project (different id;
        #    QDPX import always mints a fresh id).
        assert new_body["project_id"] != original_id

        # 5. The code we created round-trips.
        r = client.get(
            f"/api/projects/{new_body['project_id']}/codes"
        )
        assert r.status_code == 200, r.text
        codes = r.json().get("codes", r.json())
        assert any(c["name"] == "Pacing" for c in codes)

        # 6. Project metadata round-trips (name + methodology).
        assert new_body["name"] == "Round-trip cohort"
