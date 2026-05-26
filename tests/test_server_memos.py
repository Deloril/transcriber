"""End-to-end reachability tests for F5.1 (Memo entity).

The pure data layer in ``scribe/memos.py`` shipped in 7528995 with
unit coverage in ``tests/test_memos.py``; the right-click POST flow
(F5.2) shipped in c3c3375. What was missing — and what this file
proves — is the **user-facing surface for F5.1 itself**:

  * GET  /projects/<pid>/memos renders a real memos page (not a
    wireframe stub) with a "+ New memo" button, a memo list, a
    create/edit form, and a type filter.
  * GET  /api/projects/<pid>/memos lists persisted memos and accepts
    the same filter set as :func:`scribe.memos.list_memos`.
  * GET  /api/projects/<pid>/memos/<mid> returns a single memo (used
    by the page when a row is clicked to load the edit form).
  * PATCH /api/projects/<pid>/memos/<mid> applies a partial update.
  * DELETE /api/projects/<pid>/memos/<mid> removes the file and
    returns 200/404 appropriately.
  * Project home links to /projects/<pid>/memos so the page is
    reachable without typing the URL by hand.

These tests intentionally exercise the HTTP surface end-to-end via
:class:`fastapi.testclient.TestClient` rather than calling the pure
module directly — that's what makes "F5.1 is reachable" provable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
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


def _make_project(client: TestClient, name: str = "Memos test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_memo(
    client: TestClient,
    pid: str,
    *,
    type: str = "free",
    title: str = "",
    body: str = "Body text.",
    tags: list[str] | None = None,
    links: list[dict] | None = None,
) -> dict:
    payload: dict = {
        "type": type,
        "title": title,
        "body": body,
    }
    if tags is not None:
        payload["tags"] = tags
    if links is not None:
        payload["links"] = links
    r = client.post(f"/api/projects/{pid}/memos", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Page renders the new control + replaces the wireframe
# --------------------------------------------------------------------------- #


class TestMemosPageRenders:
    """``/projects/<pid>/memos`` is a real page, not the F5.1 stub."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200

    def test_no_longer_a_wireframe_stub(self, server_env) -> None:
        """The wireframe used a `<div class="stub">` banner with the
        word 'Wireframe.' — graduating the page kills that affordance."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200
        # The pre-graduation page was a "Wireframe. Backed by:" banner.
        # Real page won't render that.
        assert "Wireframe." not in r.text
        assert "Stub: F5.1" not in r.text
        assert "alert('Stub" not in r.text

    def test_has_new_memo_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200
        # The button label is the user-visible affordance promised
        # by F5.1 in PLANNING.md.
        assert "+ New memo" in r.text
        assert 'data-test-id="memos-new"' in r.text

    def test_has_create_form_with_every_required_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        body = r.text
        # Each form field is keyed by data-test-id so the test can't
        # be fooled by a shared class name elsewhere on the page.
        assert 'data-test-id="memos-form-type"' in body
        assert 'data-test-id="memos-form-title"' in body
        assert 'data-test-id="memos-form-body"' in body
        assert 'data-test-id="memos-form-tags"' in body
        assert 'data-test-id="memos-submit"' in body

    def test_has_type_filter_with_all_eight_memo_types(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        body = r.text
        assert 'data-test-id="memos-type-filter"' in body
        # Every MEMO_TYPES value should appear as an <option>.
        from scribe.memos import MEMO_TYPES
        for t in MEMO_TYPES:
            assert f'value="{t}"' in body, f"missing memo type option {t}"

    def test_consumes_the_json_listing_api(self, server_env) -> None:
        """The page's loader calls /api/projects/<pid>/memos.

        Confirms the page's JS references the listing endpoint via a
        template-string interpolation. If a future refactor moves the
        URL elsewhere, this test will catch the divergence.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert "/api/projects/${PROJECT_ID}/memos" in r.text


class TestProjectHomeLinksToMemos:
    """Project home shell links to /projects/<pid>/memos so the page
    is reachable without hand-typing the URL — same pattern as F3.1
    settings link."""

    def test_project_home_has_memos_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert f'/projects/{pid}/memos' in r.text


# --------------------------------------------------------------------------- #
# JSON API: list / get / patch / delete
# --------------------------------------------------------------------------- #


class TestListMemosAPI:
    def test_empty_project_returns_empty_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/memos")
        assert r.status_code == 200
        assert r.json() == {"memos": []}

    def test_lists_persisted_memos_in_created_order(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        m1 = _make_memo(client, pid, title="First")
        m2 = _make_memo(client, pid, title="Second")
        r = client.get(f"/api/projects/{pid}/memos")
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["memos"]]
        assert ids == [m1["id"], m2["id"]]

    def test_filters_by_type(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        _make_memo(client, pid, type="free", title="Plain")
        thx = _make_memo(client, pid, type="theoretical", title="Theory")
        r = client.get(
            f"/api/projects/{pid}/memos", params={"type": "theoretical"}
        )
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["memos"]]
        assert ids == [thx["id"]]

    def test_invalid_type_filter_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/memos", params={"type": "garbage"}
        )
        assert r.status_code == 400

    def test_unknown_project_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/aaaaaaaaaaaa/memos")
        assert r.status_code == 404


class TestGetMemoAPI:
    def test_round_trips(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        m = _make_memo(
            client, pid, type="theoretical", title="T", body="hello"
        )
        r = client.get(f"/api/projects/{pid}/memos/{m['id']}")
        assert r.status_code == 200
        got = r.json()
        assert got["id"] == m["id"]
        assert got["type"] == "theoretical"
        assert got["title"] == "T"
        assert got["body"] == "hello"

    def test_unknown_memo_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/memos/{'a' * 12}")
        assert r.status_code == 404

    def test_invalid_memo_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/memos/not-a-hex")
        assert r.status_code == 400


class TestPatchMemoAPI:
    def test_updates_title_and_body(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        m = _make_memo(client, pid, title="Old", body="old body")
        r = client.patch(
            f"/api/projects/{pid}/memos/{m['id']}",
            json={"title": "New", "body": "new body"},
        )
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["title"] == "New"
        assert got["body"] == "new body"
        # Verify persistence — re-fetch.
        r2 = client.get(f"/api/projects/{pid}/memos/{m['id']}")
        assert r2.json()["title"] == "New"

    def test_invalid_type_rejected_with_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        m = _make_memo(client, pid, title="T")
        r = client.patch(
            f"/api/projects/{pid}/memos/{m['id']}",
            json={"type": "not-a-real-type"},
        )
        assert r.status_code == 400

    def test_unknown_memo_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.patch(
            f"/api/projects/{pid}/memos/{'a' * 12}", json={"title": "x"}
        )
        assert r.status_code == 404


class TestDeleteMemoAPI:
    def test_deletes_the_file(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        m = _make_memo(client, pid, title="Doomed")
        path = (
            srv.PROJECTS_DIR / pid / "memos" / f"{m['id']}.json"
        )
        assert path.exists()
        r = client.delete(f"/api/projects/{pid}/memos/{m['id']}")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        assert not path.exists()
        # Listing reflects the delete.
        r2 = client.get(f"/api/projects/{pid}/memos")
        assert r2.json()["memos"] == []

    def test_double_delete_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        m = _make_memo(client, pid, title="x")
        r1 = client.delete(f"/api/projects/{pid}/memos/{m['id']}")
        assert r1.status_code == 200
        r2 = client.delete(f"/api/projects/{pid}/memos/{m['id']}")
        assert r2.status_code == 404


# --------------------------------------------------------------------------- #
# End-to-end: create-from-page payload round-trip
# --------------------------------------------------------------------------- #


class TestCreatePageRoundTrip:
    """The page's POST payload — flat shape, type/title/body/tags —
    must round-trip through the listing API the page also consumes."""

    def test_flat_create_appears_in_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "type": "theoretical",
                "title": "Categories of pain",
                "body": "Three sub-types so far: physical, identity, social.",
                "body_format": "markdown",
                "tags": ["early-coding", "pain"],
            },
        )
        assert r.status_code == 201, r.text
        memo_id = r.json()["id"]
        r2 = client.get(f"/api/projects/{pid}/memos")
        assert r2.status_code == 200
        memos = r2.json()["memos"]
        assert len(memos) == 1
        m = memos[0]
        assert m["id"] == memo_id
        assert m["type"] == "theoretical"
        assert m["title"] == "Categories of pain"
        assert m["tags"] == ["early-coding", "pain"]
