"""Tests for F10.4 — library row action button layout fix.

The third action button on each library row (Delete) used to clip off
the right edge of the table on a 14" MacBook (and any window narrower
than ~1180 px) because ``td.actions`` had ``white-space: nowrap`` with
no width budget. F10.4 fixed that by removing the cell-level nowrap
and rendering actions in an inner flex container that wraps cleanly.

These tests pin the fix in place so a future CSS edit can't reintroduce
the overflow without breaking a test:

* The cell-level ``td.actions`` rule must not declare
  ``white-space: nowrap``. (We still allow nowrap on individual buttons
  so a single label like "Discard media" stays on one line — the wrap
  decision happens at the gap between buttons.)
* The renderer must wrap the action elements in a ``.row-actions``
  container so the flex layout has somewhere to attach.
* Buttons must use a flex layout (``display: inline-flex`` or be inside
  a flex parent) so wrapped rows align cleanly.

The tests are pure-text assertions against the template file because
the template ships its CSS inline; nothing else parses or transforms
it before it hits the browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scribe"
    / "templates"
    / "library.html"
)


@pytest.fixture(scope="module")
def template_text() -> str:
    """Read the library template once per test module."""
    return TEMPLATE_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strip_css_comments(css: str) -> str:
    """Strip ``/* ... */`` blocks from CSS so an explanatory comment
    that *mentions* a property (e.g. "we no longer set nowrap") doesn't
    trigger a property-search assertion."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _strip_media_blocks(css: str) -> str:
    """Remove every ``@media (...) { ... }`` block from CSS.

    The viewport-aware overrides under a media query are *intentional*
    and must not be checked by rules that target the base (no-media)
    state. We strip them with a brace-balanced parser so nested rules
    don't fool a naive regex.
    """
    out: list[str] = []
    i = 0
    while i < len(css):
        if css.startswith("@media", i):
            # Find the opening brace of this @media block.
            brace = css.find("{", i)
            if brace == -1:
                # Malformed — stop stripping; let downstream assertions
                # surface the parse error.
                out.append(css[i:])
                break
            depth = 1
            j = brace + 1
            while j < len(css) and depth > 0:
                if css[j] == "{":
                    depth += 1
                elif css[j] == "}":
                    depth -= 1
                j += 1
            i = j
            continue
        out.append(css[i])
        i += 1
    return "".join(out)


def _rule_block(css: str, selector: str) -> str:
    """Return the body of the first ``<selector> { ... }`` block in
    ``css``. Selector match is exact (no regex)."""
    needle = selector + " {"
    idx = css.find(needle)
    if idx == -1:
        idx = css.find(selector + "{")  # tolerate omitted space
        if idx == -1:
            raise AssertionError(f"selector not found: {selector!r}")
        brace = idx + len(selector)
    else:
        brace = idx + len(selector) + 1  # the space char
    # Walk forward to the closing brace, balancing nested braces just
    # in case (rare for plain CSS, but cheap to handle).
    open_idx = css.find("{", brace)
    assert open_idx != -1, f"no opening brace after selector {selector!r}"
    depth = 1
    j = open_idx + 1
    while j < len(css) and depth > 0:
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
        j += 1
    return css[open_idx + 1 : j - 1]


# --------------------------------------------------------------------------- #
# F10.4 — base-state CSS no longer clips the actions cell
# --------------------------------------------------------------------------- #


class TestActionsCellNotNowrap:
    """The ``td.actions`` cell must not declare ``white-space: nowrap``
    outside an ``@media`` block. F10.4's whole point is that wrapping
    is allowed, so the rightmost button stays visible on a 14" MBP."""

    def test_base_actions_rule_is_not_nowrap(self, template_text: str) -> None:
        base = _strip_media_blocks(_strip_css_comments(template_text))
        body = _rule_block(base, "table.lib tbody td.actions")
        assert "nowrap" not in body, (
            "F10.4 regression: td.actions has white-space: nowrap, which "
            "clips the rightmost (Delete) button on a 14\" MacBook. The "
            "cell must allow wrapping; pin nowrap on individual buttons "
            "instead."
        )

    def test_base_actions_rule_does_not_force_overflow(
        self, template_text: str
    ) -> None:
        # Defensive: the cell shouldn't try to hide overflow either,
        # which would just trade one invisibility bug for another.
        base = _strip_media_blocks(_strip_css_comments(template_text))
        body = _rule_block(base, "table.lib tbody td.actions")
        assert "overflow: hidden" not in body
        assert "overflow:hidden" not in body


class TestRowActionsContainer:
    """The renderer wraps actions in ``<div class="row-actions">``
    so the flex layout has a stable parent. Removing the wrapper
    drops back to the old overflow behaviour — fail loudly."""

    def test_template_renders_row_actions_wrapper(
        self, template_text: str
    ) -> None:
        assert '<div class="row-actions">' in template_text, (
            "F10.4 regression: row renderer dropped the .row-actions "
            "flex wrapper. Without it the per-button widths fall back "
            "to inline-block + margin and cells overflow on narrow "
            "viewports."
        )

    def test_row_actions_is_flex(self, template_text: str) -> None:
        css = _strip_css_comments(template_text)
        body = _rule_block(css, "table.lib tbody td.actions .row-actions")
        assert "display: flex" in body or "display:flex" in body
        assert "flex-wrap: wrap" in body or "flex-wrap:wrap" in body


# --------------------------------------------------------------------------- #
# F10.4 — narrow-viewport polish
# --------------------------------------------------------------------------- #


class TestNarrowViewportTighten:
    """Below ~1100 px we trim padding to keep rows visually compact.
    This isn't strictly required for correctness — wrapping alone
    keeps every action visible — but it's the polish that landed
    with the fix and the test pins it in place so a refactor can't
    silently drop the rule."""

    def test_media_query_for_narrow_viewports_exists(
        self, template_text: str
    ) -> None:
        assert (
            "@media (max-width: 1100px)" in template_text
            or "@media (max-width:1100px)" in template_text
        ), (
            "F10.4 fix expects a max-width: 1100px media query that "
            "tightens action button padding for 14\" laptops."
        )

    def test_media_block_targets_actions(self, template_text: str) -> None:
        # Find the media block and assert it scopes its overrides to
        # the actions cell — not to the whole table.
        m = re.search(
            r"@media\s*\(max-width:\s*1100px\)\s*\{(.*?)\n\s*\}\s*</style>",
            template_text,
            re.DOTALL,
        )
        # Fallback: tolerate the closing-brace not directly preceding
        # </style> on different formatting.
        if m is None:
            m = re.search(
                r"@media\s*\(max-width:\s*1100px\)\s*\{",
                template_text,
            )
            assert m is not None
            # Now walk forward balancing braces from m.end() - 1.
            i = m.end() - 1  # at the opening brace
            depth = 1
            j = i + 1
            while j < len(template_text) and depth > 0:
                if template_text[j] == "{":
                    depth += 1
                elif template_text[j] == "}":
                    depth -= 1
                j += 1
            inner = template_text[i + 1 : j - 1]
        else:
            inner = m.group(1)
        assert "td.actions" in inner, (
            "F10.4 narrow-viewport rules should target td.actions, not "
            "leak into the rest of the table layout."
        )


# --------------------------------------------------------------------------- #
# Helper sanity (these test the strip helper itself; cheap to keep)
# --------------------------------------------------------------------------- #


class TestStripCssComments:
    def test_removes_block_comment(self) -> None:
        assert _strip_css_comments("a { /* hi */ x: 1; }") == "a {  x: 1; }"

    def test_removes_multiline_comment(self) -> None:
        css = "a {\n  /* multi\n     line */\n  x: 1;\n}"
        out = _strip_css_comments(css)
        assert "multi" not in out
        assert "x: 1" in out

    def test_handles_no_comments(self) -> None:
        assert _strip_css_comments("a { x: 1; }") == "a { x: 1; }"

    def test_strips_all_of_multiple(self) -> None:
        css = "/* one */ a {} /* two */ b {} /* three */"
        out = _strip_css_comments(css)
        assert "one" not in out and "two" not in out and "three" not in out


class TestStripMediaBlocks:
    def test_strip_removes_full_media_block(self) -> None:
        css = "a { x: 1; } @media (max-width: 100px) { a { x: 2; } } b { y: 3; }"
        out = _strip_media_blocks(css)
        assert "x: 2" not in out
        assert "x: 1" in out
        assert "y: 3" in out

    def test_strip_leaves_no_media_input_unchanged(self) -> None:
        css = "a { x: 1; } b { y: 2; }"
        assert _strip_media_blocks(css) == css

    def test_strip_handles_nested_braces_in_media(self) -> None:
        css = "@media (any) { a { x: 1; } b { y: 2; } } c { z: 3; }"
        out = _strip_media_blocks(css)
        assert "x: 1" not in out
        assert "y: 2" not in out
        assert "z: 3" in out


class TestRuleBlock:
    def test_returns_inner_body(self) -> None:
        css = "a { x: 1; y: 2; }"
        body = _rule_block(css, "a")
        assert "x: 1" in body and "y: 2" in body
        assert "a {" not in body and "}" not in body

    def test_unknown_selector_raises(self) -> None:
        with pytest.raises(AssertionError):
            _rule_block("a { x: 1; }", "b")
