"""F11.1 reachability proof — inline split / annotate buttons in the
transcript editor.

The original F11.1 commit (0a753af) shipped the inline buttons + a
single regression test in tests/test_server.py
(TestEditorPage::test_inline_split_and_annotate_buttons_rendered) but
predates the loop's Reachable-via gate. This module consolidates the
public-contract assertions for F11.1 into one easy-to-find TestClient
suite that exercises the user-facing surface end-to-end:

  GET /edit/<job_id>
    └─ scribe/templates/editor.html
       ├─ .seg-actions container CSS rules
       ├─ inline ✂ split-at-cursor button (.seg-split-btn)
       ├─ inline ＋ add-annotation button (.seg-note-btn)
       ├─ inline ⬆ merge-up button (.seg-mergeup-btn)
       └─ ⋮ dropdown trigger (.seg-menu-btn) — *no longer carries*
          split / add-note items; still carries merge-prev, merge-next,
          insert-after, reassign, delete

The point of having a separate file (instead of trusting the one
existing case) is so that any future refactor that quietly buries the
buttons back in the dropdown — or breaks the visibility-on-hover
affordance, or removes the keyboard-accessibility hooks — fails loudly
in a file named after the feature ID.

No production code changes; the F11.1 implementation has been live and
green since 0a753af.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Shared per-test app + jobs isolation
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same shape as tests/test_server.py::server_env, scoped here so
    failures point at F11.1 rather than the generic suite."""
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
# 1. The route serves the F11.1 surface.
# --------------------------------------------------------------------------- #


class TestEditorRouteServesInlineActions:
    """GET /edit/<job_id> must return the editor template and its
    response HTML must contain every element the F11.1 spec promises."""

    def test_route_returns_200(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        r = client.get("/edit/abc123def456")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_seg_actions_container_css_present(self, server_env) -> None:
        """The flex container rule must be in the served CSS so the
        three+ buttons line up next to each segment text."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The selector and its display rule are both required.
        assert ".seg-actions {" in body
        # Match across whitespace — the rule body is multi-line.
        block = re.search(r"\.seg-actions\s*\{[^}]*\}", body)
        assert block is not None, "missing .seg-actions block in served CSS"
        assert "display: flex" in block.group(0)

    def test_inline_split_button_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The CSS class, the JS that constructs the button, and its
        # icon glyph must all be present.
        assert "seg-split-btn" in body
        assert 'splitBtn.className = "seg-split-btn"' in body
        assert 'splitBtn.textContent = "✂"' in body

    def test_inline_annotate_button_rendered(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert "seg-note-btn" in body
        assert 'noteBtn.className = "seg-note-btn"' in body
        assert 'noteBtn.textContent = "＋"' in body

    def test_inline_merge_up_button_rendered(self, server_env) -> None:
        """F11.1's commit body says merge-up was promoted alongside the
        two planned ones; the test in test_server.py asserts it. Pin
        that here too."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert "seg-mergeup-btn" in body
        assert 'mergeUpBtn.className = "seg-mergeup-btn"' in body

    def test_dropdown_trigger_still_present(self, server_env) -> None:
        """The ⋮ dropdown button stays — F11.1 *promotes* high-frequency
        items but keeps the rest in a dropdown so the row never has
        more than ~4 inline buttons."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert "seg-menu-btn" in body
        assert 'menuBtn.textContent = "⋮"' in body


# --------------------------------------------------------------------------- #
# 2. The dropdown lost the promoted items — and only those.
# --------------------------------------------------------------------------- #


class TestDropdownContentsAfterPromotion:
    """The ⋮ menu must no longer expose split / add-note (they have
    inline buttons) but every other lifecycle action must stay."""

    def test_split_removed_from_dropdown(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The dropdown's HTML literal mustn't include the split item.
        assert 'data-act="split"' not in body

    def test_add_note_removed_from_dropdown(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'data-act="add-note"' not in body

    @pytest.mark.parametrize(
        "action",
        ["merge-prev", "merge-next", "insert-after", "reassign", "delete"],
    )
    def test_remaining_dropdown_items_present(self, server_env, action: str) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert f'data-act="{action}"' in body


# --------------------------------------------------------------------------- #
# 3. Hover / focus / keyboard affordance.
# --------------------------------------------------------------------------- #


class TestVisibilityAffordance:
    """The promoted buttons are hidden by default and visible on row
    hover, segment focus, or any inline button focus — that's the rule
    the editor.html CSS pins. Don't break the affordance."""

    def test_buttons_hidden_by_default(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The base rule sets visibility: hidden on .seg-actions button.
        block = re.search(
            r"\.seg-actions button\s*\{[^}]*\}", body
        )
        assert block is not None, "missing .seg-actions button base rule"
        assert "visibility: hidden" in block.group(0)

    def test_visibility_unhidden_on_segment_hover(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The hover/focus selector list includes :hover, .menu-open,
        # focus-within, and button:focus. The combined rule is one
        # multi-selector block ending with `visibility: visible`.
        assert ".segment:hover .seg-actions button," in body
        assert ".segment.menu-open .seg-actions button," in body
        assert ".seg-actions button:focus { visibility: visible; }" in body

    def test_aria_labels_present(self, server_env) -> None:
        """All three promoted buttons set aria-label so screen readers
        announce them and keyboard navigation reaches them."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'splitBtn.setAttribute("aria-label", "Split at cursor")' in body
        assert 'noteBtn.setAttribute("aria-label", "Add annotation")' in body
        assert (
            'mergeUpBtn.setAttribute("aria-label", "Merge into previous segment")'
            in body
        )

    def test_split_tooltip_carries_keyboard_shortcut_hint(self, server_env) -> None:
        """F11.1 explicitly states no new keyboard shortcuts; the existing
        Shift+Enter still splits. The button's tooltip must surface that
        shortcut so users discover it."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'splitBtn.title = "Split at cursor (Shift+Enter)"' in body


# --------------------------------------------------------------------------- #
# 4. The buttons are wired to the same handlers Shift+Enter and the old
#    dropdown items used.
# --------------------------------------------------------------------------- #


class TestButtonsRoutedToExistingHandlers:
    """F11.1 is purely an *affordance* change: the buttons reuse
    handleSegmentAction("split", ...) and ("add-note", ...) — the same
    code paths already used by Shift+Enter and the dropdown. If those
    routes break the buttons stop working."""

    def test_split_button_routes_to_split_action(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'handleSegmentAction(idx, "split", target)' in body

    def test_note_button_routes_to_add_note_action(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'handleSegmentAction(idx, "add-note", text)' in body

    def test_handler_dispatches_split_to_split_at_cursor(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # The dispatcher itself must still wire "split" to splitAtCursor.
        assert 'if (act === "split") return splitAtCursor(idx, textNode);' in body

    def test_handler_dispatches_add_note_to_open_note_for(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        assert 'if (act === "add-note") return openNoteFor(idx, textNode);' in body

    def test_shift_enter_keyboard_shortcut_still_present(self, server_env) -> None:
        """The existing Shift+Enter shortcut must still call
        splitAtCursor — F11.1 is additive, not a replacement."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        body = client.get("/edit/abc123def456").text
        # Comment marker that flags the shortcut block, plus the call.
        assert "Shift+Enter inside a segment splits at cursor" in body
        assert "splitAtCursor(idx, document.querySelector(" in body


# --------------------------------------------------------------------------- #
# 5. End-to-end walk: home → editor → F11.1 surface.
# --------------------------------------------------------------------------- #


class TestEndToEndReachability:
    """Confirm a fresh browser session can actually reach the inline
    buttons without knowing the magic /edit/<id> URL."""

    def test_home_links_to_library(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        # The home page must link the user to /library so they can
        # discover their done jobs and click into the editor.
        assert 'href="/library"' in r.text

    def test_library_links_to_editor_for_done_job(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_done_job(srv)
        # The library page loads jobs client-side via /api/jobs and
        # renders `<a href="/edit/${id}">Open</a>` per row, so the
        # server-rendered HTML embeds the /edit/ URL pattern in a JS
        # template literal rather than as static markup.
        r = client.get("/library")
        assert r.status_code == 200
        assert "/edit/" in r.text
        # The API that template hits must list the seeded job so the
        # template literal renders a real link at runtime.
        api = client.get("/api/jobs")
        assert api.status_code == 200
        payload = api.json()
        # Response envelope is {"jobs": [...], "total": N}.
        rows = payload["jobs"] if isinstance(payload, dict) else payload
        ids = [row["id"] for row in rows]
        assert "abc123def456" in ids

    def test_full_walk_lands_on_inline_buttons(self, server_env) -> None:
        """One canonical walk that asserts each F11.1 element is in the
        served editor HTML — same shape as F10.4's
        TestEndToEndLayoutSurface."""
        srv, client, _ = server_env
        _seed_done_job(srv)
        # Home → library → editor.
        assert client.get("/").status_code == 200
        assert client.get("/library").status_code == 200
        body = client.get("/edit/abc123def456").text
        # All four inline buttons.
        for cls in (
            "seg-split-btn",
            "seg-mergeup-btn",
            "seg-note-btn",
            "seg-menu-btn",
        ):
            assert cls in body, f"missing inline button class {cls}"
        # The promoted items have left the dropdown.
        assert 'data-act="split"' not in body
        assert 'data-act="add-note"' not in body
        # The remaining dropdown items stayed.
        for kept in ("merge-prev", "merge-next", "insert-after", "reassign", "delete"):
            assert f'data-act="{kept}"' in body
