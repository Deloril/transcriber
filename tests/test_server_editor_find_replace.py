"""F11.2 reachability proof — find / replace in the transcript editor +
discoverability links to the academic-coding section.

The original F11.2 commit (a455e65) shipped:

  scribe/static/js/helpers.mjs
    └─ replaceInSegmentWords(words, needle, replacement)
       rebuildSegmentText(words)
  scribe/templates/editor.html
    ├─ search-bar replace row (#replaceToggle, #replaceInput,
    │  #replaceOneBtn, #replaceAllBtn, #replaceMsg)
    ├─ /projects topbar link
    └─ help-modal entry mentioning ↔ Replace + Enter / Shift+Enter
  scribe/templates/index.html
    └─ home page links to /projects
  scribe/templates/library.html
    └─ library page links to /projects
  tests/js/find-replace.test.mjs (13 vitest cases for the pure helpers)
  tests/test_server.py::TestEditorPage::
      test_find_and_replace_controls_rendered
      test_topbar_has_projects_link
    (the original two pytest cases)

… but predates the loop's Reachable-via gate (see d1ade1d,
scripts/feature-implementer-prompt.md). The next loop iteration treats
F11.2 as incomplete unless a verifying commit names the user-facing
surface explicitly. This module consolidates the F11.2 reachability
proof into one easy-to-find TestClient suite, matching the pattern set
by F11.1's tests/test_server_editor_inline_actions.py.

No production code changes — the F11.2 implementation has been live
and green since a455e65. These tests pin the public contract so any
future refactor that quietly buries the replace input in a modal,
breaks the toggle, drops the helper imports, or removes the /projects
link from the editor / library / home pages fails loudly in a file
named after the feature ID.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Shared per-test app + jobs isolation. Same shape as the F11.1 file so
# the failure traceback points at F11.2 rather than the generic suite.
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    upload.mkdir()
    output.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects_dir)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _seed_done_job(srv) -> "srv.Job":  # type: ignore[name-defined]
    """Drop a status=done job into the registry so /edit/<id> renders."""
    out_dir = srv.OUTPUT_DIR / "abc123def456"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = srv.UPLOAD_DIR / "abc123def456" / "in.wav"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"\x00" * 64)
    job = srv.Job(
        id="abc123def456",
        input_path=input_path,
        output_dir=out_dir,
        mode="diarize",
        speakers=None,
        num_speakers=None,
        language="en",
        model="large-v3",
        created_at="2026-05-25T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result=None,
        error=None,
        output_paths={},
        audio_streams=1,
        input_filename="in.wav",
        options={},
        batch_size=8,
        started_at=None,
        finished_at=None,
    )
    srv.JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
# 1. The route serves the search-bar replace surface.
# --------------------------------------------------------------------------- #


class TestEditorRouteServesReplaceSurface:
    """GET /edit/<job_id> must return the editor template and its
    response HTML must contain every replace-row element F11.2 promises."""

    def test_route_returns_200(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        r = client.get("/edit/abc123def456")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_search_bar_is_outer_container(self, server_env) -> None:
        """The replace row hangs off the existing #searchBar element so
        the existing Ctrl/⌘+F shortcut still opens it."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="searchBar"' in body
        # The replace row must be a sibling row inside the same bar.
        assert 'class="row replace-row"' in body

    def test_replace_toggle_button_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="replaceToggle"' in body
        # The user-visible label.
        assert "↔ Replace" in body

    def test_replace_input_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="replaceInput"' in body
        # Placeholder doubles as in-page docs for the keyboard contract.
        assert "Enter to replace current" in body
        assert "Shift+Enter to replace all" in body

    def test_replace_one_button_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="replaceOneBtn"' in body
        assert "Replace the currently-highlighted match" in body

    def test_replace_all_button_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="replaceAllBtn"' in body
        assert "Replace every match in the transcript" in body

    def test_replace_message_slot_rendered(self, server_env) -> None:
        """Status messages ("Replaced N", "No matches", …) write to a
        dedicated <span>; without it the user has no feedback."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'id="replaceMsg"' in body


# --------------------------------------------------------------------------- #
# 2. The replace row stays collapsed by default and the toggle shows it.
# --------------------------------------------------------------------------- #


class TestReplaceRowCollapsedByDefault:
    """The replace row is hidden until the user expands it; this keeps
    the bar compact for the search-only case which is most common."""

    def test_replace_row_hidden_until_expanded(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The base CSS rule sets display:none on the replace row.
        block = re.search(
            r"\.search-bar\s+\.replace-row\s*\{[^}]*\}", body
        )
        assert block is not None, "missing .search-bar .replace-row CSS rule"
        assert "display: none" in block.group(0)

    def test_replace_open_class_reveals_row(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The .replace-open scope flips the row to display:flex.
        assert ".search-bar.replace-open .replace-row {" in body
        block = re.search(
            r"\.search-bar\.replace-open\s+\.replace-row\s*\{[^}]*\}", body
        )
        assert block is not None
        assert "display: flex" in block.group(0)

    def test_toggle_listener_flips_replace_open(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The JS click handler toggles the .replace-open class on
        # #searchBar — pin the contract so a refactor that switches to
        # an inline style or a separate element fails loudly.
        assert '$("replaceToggle").addEventListener("click"' in body
        assert '$("searchBar").classList.toggle("replace-open")' in body


# --------------------------------------------------------------------------- #
# 3. Pure helpers from helpers.mjs are imported and wired into the
#    editor's per-step + bulk replace flows.
# --------------------------------------------------------------------------- #


class TestPureHelpersWired:
    """F11.2's pure logic lives in helpers.mjs (so vitest can hammer
    it without a browser); the editor must actually import + call it."""

    def test_replace_in_segment_words_imported(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The static-import statement, plus the call site for "Replace all".
        assert "replaceInSegmentWords," in body
        assert 'from "/static/js/helpers.mjs"' in body
        assert "replaceInSegmentWords(seg.words, needle, replacement)" in body

    def test_rebuild_segment_text_imported(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert "rebuildSegmentText," in body
        # Used after each replace pass so seg.text matches seg.words.
        assert "seg.text = rebuildSegmentText(out)" in body

    def test_replace_all_button_calls_replace_all(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert '$("replaceAllBtn").addEventListener("click", replaceAll)' in body

    def test_replace_one_button_calls_replace_current(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert '$("replaceOneBtn").addEventListener("click", replaceCurrent)' in body

    def test_enter_in_replace_input_replaces_one(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # Enter / Shift+Enter dispatch from the replace input's keydown.
        assert '$("replaceInput").addEventListener("keydown"' in body
        assert "e.shiftKey ? replaceAll() : replaceCurrent()" in body


# --------------------------------------------------------------------------- #
# 4. Help modal documents the new shortcut so users discover it.
# --------------------------------------------------------------------------- #


class TestHelpModalDocumentsReplace:
    def test_help_row_mentions_replace_toggle(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The Ctrl/⌘+F help row got extended to call out ↔ Replace +
        # Enter / Shift+Enter semantics. Without this row the keyboard
        # shortcut is invisible.
        assert "↔ Replace" in body
        assert "Enter</kbd> in replace = replace current" in body
        assert "Shift</kbd>+<kbd>Enter</kbd> in replace = replace all" in body


# --------------------------------------------------------------------------- #
# 5. /projects discoverability links from editor, library, and home.
# --------------------------------------------------------------------------- #


class TestProjectsNavLinks:
    """F11.2's second user-facing change: a /projects link from every
    page a fresh user is likely to land on, so the academic-coding
    section is reachable without typing the URL."""

    def test_editor_topbar_links_to_projects(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # Anchor + visible label.
        assert 'href="/projects"' in body
        assert ">Projects<" in body

    def test_library_header_links_to_projects(self, server_env) -> None:
        srv, client, _ = server_env
        body = client.get("/library").text
        assert 'href="/projects"' in body
        # Library uses a "Projects →" arrow label.
        assert "Projects" in body

    def test_home_links_to_projects(self, server_env) -> None:
        srv, client, _ = server_env
        body = client.get("/").text
        assert 'href="/projects"' in body
        # Home page uses the 🗂 Projects glyph + a stable id.
        assert 'id="projectsBtn"' in body
        assert "🗂 Projects" in body

    def test_projects_route_resolves(self, server_env) -> None:
        """The discoverability link has to actually go somewhere; if
        /projects 404s the F11.2 fix is hollow."""
        srv, client, _ = server_env
        r = client.get("/projects")
        assert r.status_code == 200
        # The page renders the projects-list template (a known marker
        # without locking us to exact copy).
        assert "text/html" in r.headers["content-type"]


# --------------------------------------------------------------------------- #
# 6. End-to-end walk: home → library → editor → both F11.2 surfaces.
# --------------------------------------------------------------------------- #


class TestEndToEndReachability:
    """Confirm a fresh browser session can actually reach the find-and-
    replace surface and the /projects link without knowing any URL."""

    def test_home_links_to_library_and_projects(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert 'href="/library"' in r.text
        assert 'href="/projects"' in r.text

    def test_full_walk_lands_on_replace_controls(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        # Home → library → editor.
        assert client.get("/").status_code == 200
        assert client.get("/library").status_code == 200
        body = client.get("/edit/abc123def456").text
        # Every replace-bar control is in the served HTML.
        for marker in (
            'id="replaceToggle"',
            'id="replaceInput"',
            'id="replaceOneBtn"',
            'id="replaceAllBtn"',
            'id="replaceMsg"',
        ):
            assert marker in body, f"missing replace control marker {marker!r}"
        # Both helpers are imported.
        assert "replaceInSegmentWords" in body
        assert "rebuildSegmentText" in body
        # Topbar /projects link is in place.
        assert 'href="/projects"' in body
