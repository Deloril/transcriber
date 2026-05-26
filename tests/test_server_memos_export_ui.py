"""Tests for the F5.4 memo export UI surface.

PLANNING.md F5.4:

  > "Export all memos" filtered by type / linked-to.

The pure exporters (:mod:`scribe.memo_export`) shipped four formats —
CSV, Markdown, RTF, and JSONL — together with the in-memory
``filter_memos`` companion. Until this surface landed, those four
formats could only be reached from a Python REPL: there was no route,
no button, no test exercising HTTP.

This test file proves three things end-to-end:

  1. The memos page renders an Export dropdown that names all four
     formats and points at the right URLs.
  2. Each href the dropdown advertises returns 200 with the right
     ``Content-Type`` + ``Content-Disposition: attachment`` header.
  3. The endpoint honours the same filter set as
     ``/api/projects/<pid>/memos`` (type / target_type / target_id /
     author_coder_id / tag) so "Export filtered" is one click.

The deeper coverage of the export bodies lives in
``tests/test_memo_export.py``; this file's job is purely reachability
through the FastAPI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads / outputs / projects."""
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


def _create_memo(
    client: TestClient,
    project_id: str,
    *,
    type: str = "free",
    title: str = "Pacing",
    body: str = "Body.",
    tags: list[str] | None = None,
    links: list[dict] | None = None,
) -> str:
    """POST a memo via the public API and return its id."""
    payload: dict = {
        "type": type,
        "title": title,
        "body": body,
        "body_format": "markdown",
        "tags": tags or [],
        "links": links or [],
    }
    r = client.post(
        f"/api/projects/{project_id}/memos",
        json=payload,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Template render: the dropdown is in the page
# --------------------------------------------------------------------------- #


class TestMemosPageRendersExportMenu:
    """The memos page must surface F5.4's four export targets so the
    download routes are reachable without leaving the page."""

    def test_page_renders_with_export_button(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200, r.text
        text = r.text

        assert 'id="mm-export-btn"' in text
        assert 'id="mm-export-menu"' in text
        assert 'data-test-feature="F5.4"' in text
        assert "Export" in text

    def test_menu_lists_all_four_formats(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200
        text = r.text

        assert "CSV" in text
        assert "Markdown" in text
        assert "Word (RTF)" in text or "Word" in text
        assert "JSONL" in text

    def test_menu_links_target_correct_endpoints(self, env) -> None:
        """The four hrefs must point at the live download URL."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        text = r.text

        base = f"/api/projects/{pid}/memos/export"
        assert f"{base}?format=csv" in text
        assert f"{base}?format=markdown" in text
        assert f"{base}?format=rtf" in text
        assert f"{base}?format=jsonl" in text

    def test_menu_items_have_download_attribute(self, env) -> None:
        """``download`` tells the browser to save rather than render."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        text = r.text
        # Four <a class="mm-export-item"> rows in the dropdown.
        assert text.count('class="mm-export-item"') >= 4
        assert "download" in text


# --------------------------------------------------------------------------- #
# Reachability: the URLs from the menu actually resolve
# --------------------------------------------------------------------------- #


class TestMemoExportHrefsAreLive:
    """Every URL the dropdown advertises must respond 200 with the
    expected content-type."""

    def test_csv_returns_csv(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, title="Pacing")
        r = client.get(f"/api/projects/{pid}/memos/export?format=csv")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/csv")
        # The seeded memo's title is in the body.
        assert "Pacing" in r.text

    def test_markdown_returns_markdown(self, env) -> None:
        client, _ = env
        pid = _new_project(client, name="Pilot study")
        _create_memo(client, pid, title="Pacing")
        r = client.get(
            f"/api/projects/{pid}/memos/export?format=markdown"
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/markdown")
        # The Markdown header includes the project name.
        assert "Memos" in r.text and "Pilot study" in r.text
        # Per-memo heading.
        assert "## Pacing" in r.text

    def test_rtf_returns_rtf(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, title="Pacing")
        r = client.get(f"/api/projects/{pid}/memos/export?format=rtf")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/rtf"
        assert r.text.startswith(r"{\rtf1")

    def test_jsonl_returns_ndjson(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, title="Pacing")
        r = client.get(f"/api/projects/{pid}/memos/export?format=jsonl")
        assert r.status_code == 200, r.text
        assert "ndjson" in r.headers["content-type"]
        # One JSON object per line.
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["title"] == "Pacing"

    def test_all_four_have_attachment_disposition(self, env) -> None:
        """Each download URL sets ``Content-Disposition: attachment`` so
        the browser shows a save dialog."""
        client, _ = env
        pid = _new_project(client, name="Pilot study")
        _create_memo(client, pid)
        urls = [
            f"/api/projects/{pid}/memos/export?format=csv",
            f"/api/projects/{pid}/memos/export?format=markdown",
            f"/api/projects/{pid}/memos/export?format=rtf",
            f"/api/projects/{pid}/memos/export?format=jsonl",
        ]
        for url in urls:
            r = client.get(url)
            assert r.status_code == 200, (url, r.text)
            cd = r.headers.get("content-disposition", "")
            assert "attachment" in cd, (url, cd)
            # Filename slug carries the project name + the -memos infix.
            assert "pilot-study-memos" in cd, (url, cd)

    def test_default_format_is_csv(self, env) -> None:
        """Bare ``/memos/export`` (no format) defaults to CSV."""
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid)
        r = client.get(f"/api/projects/{pid}/memos/export")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")

    def test_format_aliases_resolve(self, env) -> None:
        """``md`` / ``word`` / ``ndjson`` aliases route correctly."""
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid)
        # md → markdown
        r = client.get(f"/api/projects/{pid}/memos/export?format=md")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/markdown")
        # word → rtf
        r = client.get(f"/api/projects/{pid}/memos/export?format=word")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/rtf"
        # ndjson → jsonl
        r = client.get(f"/api/projects/{pid}/memos/export?format=ndjson")
        assert r.status_code == 200
        assert "ndjson" in r.headers["content-type"]

    def test_unknown_format_is_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memos/export?format=xlsx")
        assert r.status_code == 400, r.text

    def test_missing_project_is_404(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/abcdef123456/memos/export?format=csv")
        assert r.status_code == 404, r.text


# --------------------------------------------------------------------------- #
# Filter forwarding: the URLs honour the page's filter dropdown
# --------------------------------------------------------------------------- #


class TestMemoExportFilters:
    """F5.4's headline feature is filtered export: "all memos of type
    theoretical" / "all memos linked to this code". The endpoint accepts
    the same filter query params as ``/api/projects/<pid>/memos``."""

    def test_type_filter_restricts_jsonl_output(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, type="theoretical", title="Theoretical one")
        _create_memo(client, pid, type="reflexive", title="Reflexive one")

        r = client.get(
            f"/api/projects/{pid}/memos/export?format=jsonl&type=theoretical"
        )
        assert r.status_code == 200, r.text
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["title"] == "Theoretical one"
        assert record["type"] == "theoretical"

    def test_type_filter_restricts_csv_rows(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, type="theoretical", title="Theoretical one")
        _create_memo(client, pid, type="reflexive", title="Reflexive one")

        r = client.get(
            f"/api/projects/{pid}/memos/export?format=csv&type=theoretical"
        )
        assert r.status_code == 200
        # Header + one row.
        rows = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(rows) == 2
        assert "Theoretical one" in r.text
        assert "Reflexive one" not in r.text

    def test_filter_summary_appears_in_markdown(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        _create_memo(client, pid, type="theoretical")
        r = client.get(
            f"/api/projects/{pid}/memos/export?format=markdown&type=theoretical"
        )
        assert r.status_code == 200
        assert "Filter: type=theoretical" in r.text

    def test_target_type_filter(self, env) -> None:
        """``target_type=code`` returns only memos linked to a code."""
        client, _ = env
        pid = _new_project(client)
        # Memo linked to a code.
        _create_memo(
            client,
            pid,
            type="code",
            title="Linked to code",
            links=[{"target_type": "code", "target_id": "a" * 12}],
        )
        # Memo with no link.
        _create_memo(client, pid, title="Free memo")

        r = client.get(
            f"/api/projects/{pid}/memos/export"
            "?format=jsonl&target_type=code"
        )
        assert r.status_code == 200, r.text
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["title"] == "Linked to code"

    def test_invalid_filter_value_is_400(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/memos/export?format=csv&type=not-a-type"
        )
        assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
# Empty memo list: every format still returns 200
# --------------------------------------------------------------------------- #


class TestEmptyMemoExport:
    """A brand-new project with zero memos still exports cleanly: the
    pure exporters all accept zero memos and produce a sensible empty
    body."""

    def test_empty_csv_is_header_only(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memos/export?format=csv")
        assert r.status_code == 200
        assert r.text.startswith("id,")

    def test_empty_markdown_has_placeholder(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/memos/export?format=markdown"
        )
        assert r.status_code == 200
        assert "no memos" in r.text.lower()

    def test_empty_rtf_is_minimal(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memos/export?format=rtf")
        assert r.status_code == 200
        assert r.text.startswith(r"{\rtf1")

    def test_empty_jsonl_is_empty_string(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/memos/export?format=jsonl")
        assert r.status_code == 200
        # to_jsonl returns "" for empty input — the bytes are zero.
        assert r.text == ""
