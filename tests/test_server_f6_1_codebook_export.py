"""F6.1 reachability verification — codebook export (CSV / Markdown / RTF).

The pure exporters live in ``scribe.codebook_export`` (F2.6) and the
HTTP surface ``GET /api/projects/<pid>/codebook/export?format=<fmt>``
landed in F6.1 (commit 3d1616e). The user-facing dropdown that hits
those URLs landed in F2.6's wiring commit (0e93039), but neither
commit body carried the loop's mandatory ``Reachable-via:`` line.

This test file is the explicit reachability anchor for F6.1: it
proves end-to-end that

  1. the codebook editor template renders three F6.1 menu items
     (CSV / Markdown / Word(RTF)),
  2. each menu item points at the F6.1 endpoint with the right
     ``format=`` query string,
  3. the endpoint returns 200 with the right Content-Type and a
     slugified attachment filename for each format,
  4. the alias set the user-facing query string accepts (``md``,
     ``word``, ``doc``, ``docx``) routes to the right renderer,
  5. unknown formats return 400 with an actionable message,
  6. missing projects return 404 cleanly,
  7. F6.1 does **not** accept ``xml`` / ``refi-qda`` — those are
     F6.5's surface (regression guard so the URL contracts don't
     drift).

The deeper body coverage lives in
``tests/test_server.py::TestExportCodebookAPI`` (per-format body
shape) and ``tests/test_codebook_export.py`` (pure renderer
coverage). This file's job is purely the F6.1 reachability
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), projects


def _new_project(client: TestClient, name: str = "Pilot study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_code(
    projects_dir: Path,
    project_id: str,
    *,
    name: str = "Pacing the day",
    definition: str = "Adjusting daily activity to manage limited energy.",
) -> str:
    from scribe import codes as _codes
    from scribe.codes import Code

    code = Code.new(project_id=project_id, name=name, definition=definition)
    _codes.save_code(projects_dir, code)
    return code.id


# --------------------------------------------------------------------------- #
# Template render: the three F6.1 download links are present in the dropdown
# --------------------------------------------------------------------------- #


class TestCodebookEditorRendersF6_1Links:
    """The codebook editor must render three menu items pointing at
    F6.1's endpoint, one per format. If the dropdown ever loses any
    of them the user can no longer reach the export through the
    UI."""

    def test_csv_link_in_codebook_editor(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/codebook/export?format=csv"' in r.text
        )

    def test_markdown_link_in_codebook_editor(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/codebook/export?format=markdown"'
            in r.text
        )

    def test_rtf_link_in_codebook_editor(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        assert (
            f'href="/api/projects/{pid}/codebook/export?format=rtf"' in r.text
        )

    def test_links_carry_download_attribute(self, env) -> None:
        """``download`` makes the browser save rather than render."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        # Coarse: at least three items, each with ``download``.
        assert r.text.count('class="cb-export-item"') >= 3
        assert "download" in r.text


# --------------------------------------------------------------------------- #
# Endpoint contract: each format returns 200 + the right Content-Type
# --------------------------------------------------------------------------- #


class TestExportEndpointContract:
    """The three F6.1 formats must respond with the documented MIME
    types and a slugified attachment filename."""

    def test_csv_returns_text_csv_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-codebook.csv" in cd

    def test_markdown_returns_text_markdown_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=markdown")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/markdown")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-codebook.md" in cd

    def test_rtf_returns_application_rtf_with_attachment(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=rtf")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/rtf"
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "pilot-study-codebook.rtf" in cd

    def test_default_format_is_csv(self, env) -> None:
        """No ``format=`` query string falls through to CSV."""
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/codebook/export")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")


# --------------------------------------------------------------------------- #
# End-to-end: created code shows up in the downloaded body
# --------------------------------------------------------------------------- #


class TestEndToEndContent:
    """Once a code is created via the API, the F6.1 download must
    contain it. This is the core 'reachability' claim — the user
    creates a code in the editor, hits the dropdown, gets the row."""

    def test_csv_body_contains_seeded_code(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(
            projects_dir,
            pid,
            name="Pacing the day",
            definition="Adjusting daily activity to manage limited energy.",
        )
        r = client.get(f"/api/projects/{pid}/codebook/export?format=csv")
        assert r.status_code == 200
        # Header column shape (the F6.1 public contract).
        first_line = r.text.splitlines()[0]
        assert first_line.startswith("id,name,definition")
        # Seeded code appears in the body.
        assert "Pacing the day" in r.text
        assert "Adjusting daily activity to manage limited energy." in r.text

    def test_markdown_body_contains_seeded_code(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid, name="Pacing the day")
        r = client.get(f"/api/projects/{pid}/codebook/export?format=markdown")
        assert r.status_code == 200
        assert r.text.startswith("# Codebook")
        assert "## Pacing the day" in r.text

    def test_rtf_body_contains_seeded_code(self, env) -> None:
        client, projects_dir = env
        pid = _new_project(client, name="Pilot study")
        _seed_code(projects_dir, pid, name="Pacing the day")
        r = client.get(f"/api/projects/{pid}/codebook/export?format=rtf")
        assert r.status_code == 200
        # RTF preamble + the code name appears (escape sequences may
        # transform some characters but plain ASCII passes through).
        assert r.text.startswith(r"{\rtf1")
        assert "Pacing the day" in r.text


# --------------------------------------------------------------------------- #
# Aliases the user-facing query string accepts
# --------------------------------------------------------------------------- #


class TestFormatAliases:
    """Per F6.1's documentation, ``md`` aliases markdown, and
    ``word`` / ``doc`` / ``docx`` alias RTF. Confirm each alias the
    UI might pass routes to the right renderer."""

    @pytest.mark.parametrize(
        "alias, expected_ct",
        [
            ("md", "text/markdown"),
            ("MARKDOWN", "text/markdown"),
            ("word", "application/rtf"),
            ("doc", "application/rtf"),
            ("docx", "application/rtf"),
            ("RTF", "application/rtf"),
            ("CSV", "text/csv"),
        ],
    )
    def test_alias_dispatches_to_right_renderer(
        self, env, alias: str, expected_ct: str
    ) -> None:
        client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid)
        r = client.get(
            f"/api/projects/{pid}/codebook/export?format={alias}"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith(expected_ct)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailureModes:
    """The endpoint must fail cleanly on bad input — 400 for an
    unrecognised format with an actionable message, 404 for a missing
    project — without leaking a 5xx to the user."""

    def test_unknown_format_is_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=yaml")
        assert r.status_code == 400
        body = r.json()
        # The error message lists the accepted set so the user can fix it.
        assert "yaml" in body["detail"].lower() or "Unsupported" in body[
            "detail"
        ]

    def test_missing_project_is_404(self, env) -> None:
        client, _ = env
        # 12-hex-char pattern matches the project-id schema but the
        # project itself was never created.
        r = client.get(
            "/api/projects/000000000000/codebook/export?format=csv"
        )
        assert r.status_code == 404

    def test_xml_alias_is_rejected_by_f6_1_surface(self, env) -> None:
        """F6.1 owns CSV / Markdown / RTF only. The XML formats live
        on F6.5's separate URL (``/codebook/refi-qda-xml``) so the
        format set on this endpoint stays stable. If someone added
        ``xml`` to ``EXPORT_FORMATS`` here it would shadow F6.5 — guard
        against that with an explicit reject."""
        client, _ = env
        pid = _new_project(client)
        for alias in ("xml", "refi-qda", "refi_qda"):
            r = client.get(
                f"/api/projects/{pid}/codebook/export?format={alias}"
            )
            assert r.status_code == 400, (alias, r.status_code, r.text)


# --------------------------------------------------------------------------- #
# Empty codebook still exports cleanly (no 404, no 5xx)
# --------------------------------------------------------------------------- #


class TestEmptyCodebookF6_1:
    """A brand-new project with no codes should still respond 200 from
    every F6.1 format. Empty input is documented as valid."""

    def test_empty_csv_is_header_only(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=csv")
        assert r.status_code == 200
        # Header row + nothing else.
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("id,name,definition")

    def test_empty_markdown_has_codebook_heading(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/codebook/export?format=markdown"
        )
        assert r.status_code == 200
        assert r.text.startswith("# Codebook")

    def test_empty_rtf_is_minimal_document(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/export?format=rtf")
        assert r.status_code == 200
        assert r.text.startswith(r"{\rtf1")
        assert r.text.rstrip().endswith("}")
