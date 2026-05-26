"""End-to-end reachability tests for F5.3 (memo-sorting canvas UI).

The pure module (:mod:`scribe.memo_canvas`), the canvas REST endpoints,
and the JS helpers all shipped in 9e1fd99 — but the original commit
explicitly *deferred* the drag-drop UI:

    > Editor template integration: the helpers are in place; the
    > actual drag-drop UI surface lives in a future pass once F5.4 /
    > F5.5 land and the memo composer pieces are settled.

That deferral is what this file proves is now closed:

  * GET /projects/<pid>/memos/canvas renders a real page (not the
    F5.3 wireframe) with a board, the "+ Place a memo" / "+ Category"
    buttons, the side tray for unplaced memos, and the modal forms.
  * The page consumes the canvas REST surface (cards / categories /
    links endpoints).
  * The /projects/<pid>/memos page now links to the canvas, so the
    page is reachable through the project nav without typing the URL.
  * The whole drag-place-link round-trip works against the API once
    a memo and category exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs (mirrors test_server_memos)."""
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


def _make_project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "Canvas test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_memo(client: TestClient, pid: str, *, title: str = "M") -> str:
    r = client.post(
        f"/api/projects/{pid}/memos",
        json={"type": "theoretical", "title": title, "body": "x"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Page renders the new control + replaces the wireframe
# --------------------------------------------------------------------------- #


class TestCanvasPageRenders:
    """``/projects/<pid>/memos/canvas`` is a real page, not a stub."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos/canvas")
        assert r.status_code == 200

    def test_malformed_path_returns_400(self, server_env) -> None:
        # _project_id_or_404 deliberately accepts well-formed but
        # unknown ids (the empty canvas state is shown so the IA is
        # intact); only obviously-malicious paths get 400.
        _, client, _ = server_env
        # Most other subpages reject overly-long ids the same way.
        r = client.get(f"/projects/{'x' * 80}/memos/canvas")
        assert r.status_code == 400

    def test_no_longer_a_wireframe_stub(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos/canvas")
        body = r.text
        assert r.status_code == 200
        assert "Wireframe." not in body
        assert "Stub: F5.3" not in body
        assert "alert('Stub" not in body
        # Page advertises the F5.3 feature so the UI test can lock
        # the surface to the planning ID.
        assert 'data-test-feature="F5.3"' in body

    def test_action_bar_has_canvas_buttons(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        body = client.get(f"/projects/{pid}/memos/canvas").text
        # The two top-level affordances PLANNING.md describes:
        # group cards + create categories.
        assert 'data-test-id="canvas-add-category"' in body
        assert 'data-test-id="canvas-add-memo"' in body
        assert "+ Category" in body
        assert "+ Place a memo" in body

    def test_renders_board_region(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        body = client.get(f"/projects/{pid}/memos/canvas").text
        assert 'data-test-id="canvas-board"' in body
        assert 'data-test-id="canvas-tray"' in body
        assert 'data-test-id="canvas-categories"' in body

    def test_modals_are_in_dom(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        body = client.get(f"/projects/{pid}/memos/canvas").text
        assert 'data-test-id="canvas-cat-label"' in body
        assert 'data-test-id="canvas-cat-create"' in body
        assert 'data-test-id="canvas-memo-select"' in body
        assert 'data-test-id="canvas-memo-place"' in body
        assert 'data-test-id="canvas-assign-select"' in body
        assert 'data-test-id="canvas-assign-do"' in body

    def test_consumes_canvas_api(self, server_env) -> None:
        """The page's JS calls every canvas endpoint we ship."""
        _, client, _ = server_env
        pid = _make_project(client)
        body = client.get(f"/projects/{pid}/memos/canvas").text
        # The page interpolates PROJECT_ID into the URL via a JS
        # template literal; just confirm each endpoint shape is there.
        assert "/api/projects/${PROJECT_ID}/canvas" in body
        assert "/api/projects/${PROJECT_ID}/canvas/cards/" in body
        assert "/api/projects/${PROJECT_ID}/canvas/categories" in body
        assert "/api/projects/${PROJECT_ID}/canvas/links" in body
        # And the memos listing, used to build the placement picker.
        assert "/api/projects/${PROJECT_ID}/memos" in body


class TestMemosPageLinksToCanvas:
    """The /memos page is the natural entry point, so it must link
    to the canvas — without that link, the canvas page is unreachable
    from the project nav."""

    def test_memos_page_links_to_canvas(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200
        assert f"/projects/{pid}/memos/canvas" in r.text
        assert 'data-test-id="memos-canvas-link"' in r.text


# --------------------------------------------------------------------------- #
# End-to-end: user journey through the canvas
# --------------------------------------------------------------------------- #


class TestCanvasRoundTrip:
    """The page is reachable AND the canvas API surface it consumes
    behaves end-to-end (place a card, create a category, assign,
    link memos)."""

    def test_place_then_remove_card(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        memo_id = _make_memo(client, pid, title="A")
        # Page is reachable
        assert client.get(f"/projects/{pid}/memos/canvas").status_code == 200
        # Initial canvas: empty
        r0 = client.get(f"/api/projects/{pid}/canvas")
        assert r0.status_code == 200
        assert r0.json()["cards"] == []
        # Place
        r1 = client.put(
            f"/api/projects/{pid}/canvas/cards/{memo_id}",
            json={"x": 100, "y": 50},
        )
        assert r1.status_code == 200
        cards = r1.json()["cards"]
        assert len(cards) == 1
        assert cards[0]["memo_id"] == memo_id
        assert cards[0]["x"] == 100.0
        assert cards[0]["y"] == 50.0
        # Remove
        r2 = client.delete(f"/api/projects/{pid}/canvas/cards/{memo_id}")
        assert r2.status_code == 200
        assert r2.json() == {"ok": True}
        # Canvas: empty again
        r3 = client.get(f"/api/projects/{pid}/canvas")
        assert r3.json()["cards"] == []

    def test_category_create_assign_member(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        memo_id = _make_memo(client, pid, title="A")
        # Place card
        client.put(
            f"/api/projects/{pid}/canvas/cards/{memo_id}",
            json={"x": 12, "y": 34},
        )
        # Create category
        rc = client.post(
            f"/api/projects/{pid}/canvas/categories",
            json={"label": "Emerging", "x": 0, "y": 0},
        )
        assert rc.status_code == 201, rc.text
        cat_id = rc.json()["id"]
        # Assign member
        ra = client.put(
            f"/api/projects/{pid}/canvas/categories/{cat_id}/members/{memo_id}"
        )
        assert ra.status_code == 200
        # Verify membership recorded
        rg = client.get(f"/api/projects/{pid}/canvas")
        body = rg.json()
        assert cat_id in body["category_members"]
        assert memo_id in body["category_members"][cat_id]

    def test_link_memos_records_memo_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        a = _make_memo(client, pid, title="A")
        b = _make_memo(client, pid, title="B")
        rl = client.post(
            f"/api/projects/{pid}/canvas/links",
            json={"from_memo_id": a, "to_memo_id": b, "role": "related"},
        )
        assert rl.status_code == 200, rl.text
        # The link is persisted on the source memo
        rm = client.get(f"/api/projects/{pid}/memos/{a}")
        assert rm.status_code == 200
        links = rm.json().get("links", [])
        # A memo→memo link with role="related" should appear pointing at b.
        targets = [
            l for l in links
            if l.get("target_type") == "memo"
            and l.get("target_id") == b
            and l.get("role") == "related"
        ]
        assert targets, f"expected memo→memo link, got {links!r}"
