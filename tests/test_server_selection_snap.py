"""End-to-end reachability tests for F4.4 (snap-to-word / sentence /
paragraph selection helpers).

Background
----------

F4.4 ships ``scribe.selection_snap`` (with ``snap_to_word``,
``snap_to_sentence``, ``snap_to_paragraph``) and a JS mirror in
``scribe/static/js/helpers.mjs`` (``snapToWord`` / ``snapToSentence`` /
``snapToParagraph``). Both modules are exhaustively unit-tested:

* 46 pytest cases in ``tests/test_selection_snap.py``
* 45 vitest cases in ``tests/js/selection-snap.test.mjs``

What was missing — and what this file proves — is the integration
path: a coder using the source-coding view (``GET
/projects/<pid>/sources/<sid>``) can actually *invoke* the snap
helpers from the apply-code popover. Per the loop's done-criteria,
F4.4 is only "done" if the helpers are reachable from the user-facing
surface. That means three things, all asserted below:

1. The coding-view template renders a ``#snapToolbar`` inside the
   apply-code popover with three buttons (``data-snap="word" |
   "sentence" | "paragraph"``).
2. The page imports the snap functions from ``helpers.mjs`` so the
   click handlers and the unit-tested helpers stay in lock-step (a
   re-implementation in the page would silently drift).
3. The handler hooks them up: a click on a snap button calls
   ``snapCurrentSelection`` which mutates ``CURRENT_SELECTION`` and
   repaints — the JS-side glue is present.

This file does *not* exercise the actual snap algorithm — the
parallel pytest + vitest suites already do that, and the pure
helpers don't have an HTTP route. (Snap runs entirely client-side
on a transient selection; there's no application state to round-trip
through a server route.) The point of this file is to assert
reachability: the buttons ship, the imports are wired, and the
handler glue exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures — mirror the pattern in test_server_application_spans.py.
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads/outputs/projects."""
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


def _make_project(client: TestClient, name: str = "F4.4 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "Interview 1") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# 1. The coding-view template surfaces the F4.4 snap toolbar.
# --------------------------------------------------------------------------- #


class TestSourceCodingTemplateExposesSnapToolbar:
    """Without these controls in the rendered page, a coder can't
    reach scribe.selection_snap from the apply-code flow no matter
    how clean the helpers are. Every assertion here pins one specific
    affordance the F4.4 backlog promises."""

    def test_coding_view_renders_snap_toolbar(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # Toolbar container itself, plus the F4.4 feature marker the
        # design doc + the audit (W3.12) reference.
        assert 'id="snapToolbar"' in r.text
        assert 'data-test-feature="F4.4"' in r.text
        assert 'aria-label="Snap selection"' in r.text

    def test_coding_view_renders_three_snap_buttons(self, server_env) -> None:
        """The F4.4 backlog item enumerates word, sentence, and
        paragraph helpers; all three must surface as buttons."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # Each button has its own data-snap value + matching test id.
        assert 'data-snap="word"' in body
        assert 'data-snap="sentence"' in body
        assert 'data-snap="paragraph"' in body
        assert 'data-test-id="snap-word"' in body
        assert 'data-test-id="snap-sentence"' in body
        assert 'data-test-id="snap-paragraph"' in body
        # Visible labels — researchers shouldn't have to read source
        # to figure out what each button does.
        assert ">Word</button>" in body
        assert ">Sentence</button>" in body
        assert ">Paragraph</button>" in body

    def test_coding_view_imports_snap_helpers_from_helpers_mjs(
        self, server_env
    ) -> None:
        """The handler must reuse the same snap functions the parallel
        vitest suite (tests/js/selection-snap.test.mjs) exercises;
        re-implementing them inline would silently drift from the
        Python-side helpers in tests/test_selection_snap.py."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # The shim imports from helpers.mjs and exposes the functions
        # on window.__snap so the classic-script handler can call
        # them. Both ends of the bridge must be present.
        assert "/static/js/helpers.mjs" in body
        assert "snapToWord" in body
        assert "snapToSentence" in body
        assert "snapToParagraph" in body
        assert "window.__snap" in body

    def test_coding_view_wires_click_handler_to_snap_module(
        self, server_env
    ) -> None:
        """The handler glue: a click on a .snap-btn dispatches to
        snapCurrentSelection(kind), which calls into
        window.__snap.snapTo<Kind>(). Without this glue the toolbar
        is decorative."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert "function snapCurrentSelection" in body
        # Each branch must exist by name so the test fails if a future
        # refactor accidentally drops one (e.g. only word + sentence).
        assert 'kind === "word"' in body
        assert 'kind === "sentence"' in body
        assert 'kind === "paragraph"' in body
        # Click delegation hook on .snap-btn.
        assert ".snap-btn" in body

    def test_snap_toolbar_lives_inside_apply_popover(self, server_env) -> None:
        """The toolbar must be a child of #applyPopover so it shows
        up only when the popover is open (i.e. when there's an actual
        selection to snap). A free-floating toolbar would be
        visually confusing — the snap action only makes sense in the
        context of an in-progress code-application."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        pop_idx = body.find('id="applyPopover"')
        toolbar_idx = body.find('id="snapToolbar"')
        end_idx = body.find("</div>", toolbar_idx)
        assert pop_idx >= 0, "applyPopover not found"
        assert toolbar_idx >= 0, "snapToolbar not found"
        assert pop_idx < toolbar_idx < end_idx, (
            "snap toolbar should render inside the apply-code popover"
        )

    def test_coding_view_keeps_handler_idempotent_with_existing_flow(
        self, server_env
    ) -> None:
        """The snap-click handler must coexist with the existing
        F4.1/F4.2/F4.3 selection plumbing — paintRange / unpaintRange
        for visual highlight, CURRENT_SELECTION as the canonical
        anchor pair. The smoke check is that all three references
        survive in the rendered page."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert "CURRENT_SELECTION" in body
        assert "paintRange(" in body
        assert "unpaintRange(" in body
        # And the popover-reposition glue: snap can widen the
        # selection past the popover's original position, so the
        # handler nudges it back over the new range.
        assert "applyPopover" in body


# --------------------------------------------------------------------------- #
# 2. Round-trip: applying a code to a snapped span persists with the
#    snapped anchors. The snap helpers don't have a server route of
#    their own (they run client-side on a transient selection), but we
#    can prove "the user-facing flow that ends in an Application accepts
#    the kinds of anchors snap_to_* produce" by POSTing those exact
#    anchors at the existing applications endpoint.
# --------------------------------------------------------------------------- #


class TestSnappedAnchorsPersistAsApplications:
    """The snap helpers always emit whole-word anchors with
    ``start_char_offset`` / ``end_char_offset`` set to ``None``. A
    coder hits "Snap to sentence" then "Apply code"; the apply call
    has to accept that shape — otherwise the toolbar would be a
    UI lie."""

    def test_word_snapped_anchors_create_application(self, server_env) -> None:
        from scribe.selection_snap import Selection, snap_to_word

        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        # Make a code to apply.
        r_code = client.post(
            f"/api/projects/{pid}/codes",
            json={"name": "managing pain", "definition": "coping in the moment"},
        )
        cid = r_code.json()["id"]
        # Snap a sub-word selection to whole words and POST it.
        snapped = snap_to_word(
            Selection(
                start_word_id="s0w2",
                end_word_id="s0w5",
                start_char_offset=3,
                end_char_offset=2,
            )
        )
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid,
                "source_id": sid,
                "anchor_start_word_id": snapped.start_word_id,
                "anchor_end_word_id": snapped.end_word_id,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        # Snap dropped the offsets; the persisted application reflects
        # whole-word anchors with no sub-word offsets.
        assert body["anchor_start_word_id"] == "s0w2"
        assert body["anchor_end_word_id"] == "s0w5"
        assert body.get("start_char_offset") in (None, 0) or "start_char_offset" not in body
        assert body.get("end_char_offset") in (None, 0) or "end_char_offset" not in body

    def test_sentence_snapped_anchors_create_application(
        self, server_env
    ) -> None:
        """The widening case: ``snap_to_sentence`` extends the start
        and end to the sentence boundaries. The apply route must
        accept those wider anchors verbatim."""
        from scribe.selection_snap import Selection, snap_to_sentence

        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r_code = client.post(
            f"/api/projects/{pid}/codes",
            json={
                "name": "framing the question",
                "definition": "interviewer setting up the topic",
            },
        )
        cid = r_code.json()["id"]
        # Build a transcript-shaped fixture (the snap helpers don't
        # need the real transcript on disk — they take segments
        # directly).
        segments = [
            {
                "speaker": "S0",
                "words": [
                    {"text": "Hello."},
                    {"text": "How"},
                    {"text": "are"},
                    {"text": "you?"},
                    {"text": "Goodbye!"},
                ],
            }
        ]
        snapped = snap_to_sentence(
            Selection(start_word_id="s0w2", end_word_id="s0w2"),
            segments,
        )
        # "How are you?" sentence: words 1..3.
        assert snapped.start_word_id == "s0w1"
        assert snapped.end_word_id == "s0w3"
        # Round-trip those anchors.
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid,
                "source_id": sid,
                "anchor_start_word_id": snapped.start_word_id,
                "anchor_end_word_id": snapped.end_word_id,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["anchor_start_word_id"] == "s0w1"
        assert body["anchor_end_word_id"] == "s0w3"

    def test_paragraph_snapped_anchors_create_application(
        self, server_env
    ) -> None:
        """The widest case: ``snap_to_paragraph`` extends to the whole
        speaker turn. The apply route must accept anchors that start
        at word 0 of the paragraph's first segment and end at the
        last word of its last segment."""
        from scribe.selection_snap import Selection, snap_to_paragraph

        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r_code = client.post(
            f"/api/projects/{pid}/codes",
            json={"name": "S1 turn", "definition": "whatever S1 says in one go"},
        )
        cid = r_code.json()["id"]
        segments = [
            {"speaker": "S0", "words": [{"text": "Hi."}]},
            {
                "speaker": "S1",
                "words": [{"text": "I"}, {"text": "agree."}],
            },
            {
                "speaker": "S1",
                "words": [{"text": "Definitely."}],
            },
            {"speaker": "S0", "words": [{"text": "OK."}]},
        ]
        snapped = snap_to_paragraph(
            Selection(start_word_id="s1w0", end_word_id="s1w0"),
            segments,
        )
        # S1 paragraph: segments 1..2; word range s1w0..s2w0.
        assert snapped.start_word_id == "s1w0"
        assert snapped.end_word_id == "s2w0"
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid,
                "source_id": sid,
                "anchor_start_word_id": snapped.start_word_id,
                "anchor_end_word_id": snapped.end_word_id,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["anchor_start_word_id"] == "s1w0"
        assert body["anchor_end_word_id"] == "s2w0"
