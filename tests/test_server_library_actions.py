"""End-to-end reachability tests for F10.4 (library row action layout).

The F10.4 fix is a CSS-only patch to ``scribe/templates/library.html``
that drops ``white-space: nowrap`` from ``td.actions`` and wraps the
action elements in a ``<div class="row-actions">`` flex container so
the rightmost button (Delete) can no longer be clipped off the right
edge on a 14" MacBook (any window narrower than ~1180 px).

Pure-text assertions against the template file already live in
``tests/test_library_layout.py``. This file consolidates the
**user-facing surface** proof for F10.4 into one easy-to-find
``TestClient`` integration suite that walks the same path a real
browser walks: ``GET /library`` → the rendered HTML carries the new
``.row-actions`` container + the ``@media (max-width: 1100px)``
breakpoint + the Delete button → none of those elements depend on
horizontal overflow to be visible.

Why a separate file: the Reachable-via gate (see
``scripts/feature-implementer-prompt.md``) requires every feature to
have an integration test that exercises the route via
``fastapi.testclient.TestClient``. ``tests/test_library_layout.py``
asserts against the template *file*; this file asserts against the
*served* HTML response, pinning the surface end-to-end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient for the FastAPI app with isolated tmp dirs."""
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

    return TestClient(srv.app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)


def _strip_css_comments(text: str) -> str:
    """Drop CSS ``/* … */`` comments so explanatory prose can't trigger
    a false positive when we search for ``white-space: nowrap``."""
    return _CSS_COMMENT_RE.sub("", text)


def _strip_media_blocks(text: str) -> str:
    """Drop balanced ``@media (...) { … }`` blocks so a media-query
    override can't accidentally satisfy a base-state assertion."""
    out: list[str] = []
    i = 0
    while i < len(text):
        m = re.search(r"@media\b", text[i:])
        if not m:
            out.append(text[i:])
            break
        start = i + m.start()
        out.append(text[i:start])
        brace_open = text.find("{", start)
        if brace_open == -1:
            out.append(text[start:])
            break
        depth = 0
        j = brace_open
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        i = j
    return "".join(out)


# --------------------------------------------------------------------------- #
# /library is reachable and serves the F10.4 layout fix
# --------------------------------------------------------------------------- #


class TestLibraryRouteServesFix:
    """The route exists and renders the F10.4 layout fix in the response."""

    def test_library_route_returns_200(self, client: TestClient) -> None:
        r = client.get("/library")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_response_html_includes_row_actions_container_class(
        self, client: TestClient
    ) -> None:
        # The CSS rule must exist in the served HTML so the browser
        # can apply flex-wrap to the action cell. If a future edit
        # drops the rule, this assertion fires.
        body = client.get("/library").text
        assert "td.actions .row-actions" in body
        assert "flex-wrap: wrap" in body

    def test_response_html_includes_row_actions_renderer(
        self, client: TestClient
    ) -> None:
        # The JS row renderer must emit `<div class="row-actions">`
        # so the CSS rule has something to attach to. Asserting the
        # exact substring guards against a refactor that quietly
        # drops the wrapper.
        body = client.get("/library").text
        assert '<div class="row-actions">' in body

    def test_response_html_includes_delete_button_inside_row_actions(
        self, client: TestClient
    ) -> None:
        # The Delete button is the one F10.4 was specifically rescuing
        # from off-screen clipping. It must still be the last child
        # inside the row-actions container in the renderer's HTML
        # template.
        body = client.get("/library").text
        m = re.search(
            r'<div class="row-actions">(.+?)</div>',
            body,
            flags=re.DOTALL,
        )
        assert m is not None, "row-actions container missing from rendered HTML"
        inner = m.group(1)
        assert 'data-action="delete"' in inner
        assert 'class="danger"' in inner


class TestLibraryPageNoLongerClipsActions:
    """The base-state CSS must not reintroduce the bug F10.4 fixed."""

    def test_actions_cell_no_nowrap_in_served_css(
        self, client: TestClient
    ) -> None:
        # Strip comments + media blocks before searching so
        # explanatory prose like "the cell used to pin nowrap"
        # and any narrow-viewport-only overrides can't satisfy
        # the assertion. The base-state rule for ``td.actions``
        # must not declare ``white-space: nowrap``.
        body = client.get("/library").text
        clean = _strip_media_blocks(_strip_css_comments(body))
        # Find the td.actions block (not the inner .row-actions one).
        m = re.search(
            r"table\.lib\s+tbody\s+td\.actions\s*\{([^}]*)\}",
            clean,
        )
        assert m is not None, "td.actions CSS rule missing from served HTML"
        rule_body = m.group(1)
        assert "nowrap" not in rule_body, (
            "td.actions still pins white-space: nowrap — the F10.4 fix "
            "would be reverted and the Delete button would clip again"
        )

    def test_actions_cell_no_overflow_hidden(self, client: TestClient) -> None:
        # A would-be alternative to nowrap is `overflow: hidden`, which
        # also clips the Delete button silently. F10.4's fix should
        # not have traded one bug for another.
        body = client.get("/library").text
        clean = _strip_media_blocks(_strip_css_comments(body))
        m = re.search(
            r"table\.lib\s+tbody\s+td\.actions\s*\{([^}]*)\}",
            clean,
        )
        assert m is not None
        rule_body = m.group(1)
        assert "overflow: hidden" not in rule_body
        assert "overflow:hidden" not in rule_body

    def test_narrow_viewport_media_query_present(
        self, client: TestClient
    ) -> None:
        # Below 1100 px the action cell tightens its padding/gap so
        # rows usually still fit on one line at 14" laptop widths.
        # If even that budget is exceeded, the flex container wraps —
        # never clips. The media query is the second half of the fix.
        body = client.get("/library").text
        # Match either spacing variant.
        assert re.search(r"@media\s*\(max-width:\s*1100px\)", body)


# --------------------------------------------------------------------------- #
# /library remains reachable from the home-page surface
# --------------------------------------------------------------------------- #


class TestLibraryReachableFromHome:
    """The home page links to the page F10.4 fixes."""

    def test_home_page_links_to_library(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'href="/library"' in body

    def test_home_page_library_link_label_visible(
        self, client: TestClient
    ) -> None:
        # The label must appear in the rendered HTML so the user can
        # actually see and click it. A bare ``<a href="/library">``
        # with no text would technically reach the page but be
        # invisible — that's the same failure mode F10.4 fixed for
        # the Delete button, applied to the Library link itself.
        body = client.get("/").text
        m = re.search(r'href="/library"[^>]*>([^<]+)</a>', body)
        assert m is not None
        label = m.group(1).strip()
        assert label, "Library link has no visible text"


# --------------------------------------------------------------------------- #
# Helper sanity tests — keep the assertion machinery honest
# --------------------------------------------------------------------------- #


class TestStripCssCommentsHelper:
    def test_drops_block_comment(self) -> None:
        assert _strip_css_comments("a /* x */ b") == "a  b"

    def test_drops_multiline_comment(self) -> None:
        assert _strip_css_comments("a /* x\ny */ b") == "a  b"

    def test_passes_through_when_no_comment(self) -> None:
        assert _strip_css_comments("td.actions { color: red; }") == (
            "td.actions { color: red; }"
        )


class TestStripMediaBlocksHelper:
    def test_drops_simple_media_block(self) -> None:
        text = "a { x: 1; } @media (max-width: 600px) { b { y: 2; } } c { z: 3; }"
        out = _strip_media_blocks(text)
        assert "@media" not in out
        assert "a { x: 1; }" in out
        assert "c { z: 3; }" in out

    def test_handles_nested_braces_inside_media_block(self) -> None:
        text = "@media (max: 1px) { a { x: 1; } b { y: 2; } } d { z: 3; }"
        out = _strip_media_blocks(text)
        assert "@media" not in out
        assert "a {" not in out
        assert "d { z: 3; }" in out

    def test_passes_through_when_no_media_query(self) -> None:
        text = "a { x: 1; } b { y: 2; }"
        assert _strip_media_blocks(text) == text


# --------------------------------------------------------------------------- #
# End-to-end walk: home → library → action layout intact
# --------------------------------------------------------------------------- #


class TestEndToEndLayoutSurface:
    def test_full_walk(self, client: TestClient) -> None:
        # Home page links to library.
        home = client.get("/")
        assert home.status_code == 200
        assert 'href="/library"' in home.text

        # Library page renders.
        lib = client.get("/library")
        assert lib.status_code == 200

        # The F10.4 fix is in the served HTML.
        assert "td.actions .row-actions" in lib.text
        assert "flex-wrap: wrap" in lib.text
        assert '<div class="row-actions">' in lib.text

        # And nothing in the base-state td.actions rule clips overflow.
        clean = _strip_media_blocks(_strip_css_comments(lib.text))
        m = re.search(
            r"table\.lib\s+tbody\s+td\.actions\s*\{([^}]*)\}",
            clean,
        )
        assert m is not None
        rule_body = m.group(1)
        assert "nowrap" not in rule_body
        assert "overflow: hidden" not in rule_body
