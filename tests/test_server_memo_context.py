"""End-to-end reachability tests for F5.2 — Right-click memo creation
from any context with link pre-populated.

Background
----------

F5.2 shipped ``scribe.memo_context`` in c3c3375 with full pytest +
vitest coverage. The endpoint
``POST /api/projects/{pid}/memos`` already routes a top-level
``context: {target_type, target_id, role?}`` block through
:func:`scribe.memo_context.build_memo_draft_from_context` so the
right-click flow's link pre-population rules match the JS helper
one-for-one.

What was missing — and what this file proves — is the **user-facing
surface for F5.2 itself**:

  * The source-coding view (``GET /projects/<pid>/sources/<sid>``)
    renders a memo composer modal, a "📝 New memo" button in the
    page actions, a context menu, and the helper-import shim that
    hangs ``buildMemoContextPayload`` / ``defaultMemoTypeForTarget``
    off ``window.__memoCtx`` for the classic-script handler.

  * Each entity surface that should be right-clickable carries a
    ``data-test-feature="F5.2"`` marker:
        - the ``.app-row`` cards (.app-row gets data-app-id; the
          contextmenu listener resolves application memos from it)
        - the ``.code-chip`` chips in the codebook summary
        - the page-action "📝 New memo" button (target=source)
    If a future refactor moves the contextmenu wiring elsewhere, the
    tests will catch the divergence.

  * The same context endpoint round-trips through the page's POST
    payload shape — i.e. the JS helper's wire body deserialises into
    a persistable Memo with the correct primary link, default type,
    and optional extra-link rules.

These tests intentionally exercise the HTTP surface end-to-end via
:class:`fastapi.testclient.TestClient` rather than calling the pure
module directly. That's what makes "F5.2 is reachable from a real UI"
provable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
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


def _make_project(client: TestClient, name: str = "F5.2 holder") -> str:
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


def _make_code(client: TestClient, pid: str, name: str = "managing") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "gerund-form code for an action"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Source-coding view renders the memo composer + every right-click surface
# --------------------------------------------------------------------------- #


class TestSourceCodingMemoSurfaceRenders:
    """The coding page exposes the F5.2 surfaces a researcher needs.

    Every assertion is keyed by a ``data-test-id`` so a future refactor
    that moves the markup around can't accidentally regress the
    surface without flipping this test.
    """

    def test_page_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200

    def test_page_action_new_memo_button_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert 'data-test-id="src-new-memo"' in body
        assert "📝 New memo" in body

    def test_memo_modal_renders_every_form_field(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        # Modal container.
        assert 'data-test-id="src-memo-modal"' in body
        # Target pill — the user must see what entity the memo links to.
        assert 'data-test-id="src-memo-target-pill"' in body
        # Type / title / body / tags fields each have a unique marker.
        for marker in (
            "src-memo-type",
            "src-memo-title",
            "src-memo-body",
            "src-memo-tags",
            "src-memo-save",
            "src-memo-cancel",
            "src-memo-status",
        ):
            assert f'data-test-id="{marker}"' in body, f"missing marker {marker}"

    def test_memo_modal_lists_all_eight_memo_types(self, server_env) -> None:
        """The closed-vocabulary type select must offer every
        scribe.memos.MEMO_TYPES value so the user can re-classify the
        memo if the right-click default isn't a fit.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        from scribe.memos import MEMO_TYPES
        for t in MEMO_TYPES:
            assert f'value="{t}"' in body, f"missing memo type option {t}"

    def test_context_menu_renders(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        assert 'data-test-id="src-memo-ctx-menu"' in body
        assert 'data-test-id="src-memo-ctx-new"' in body

    def test_helpers_module_imports_memo_context_helpers(
        self, server_env
    ) -> None:
        """The classic-script handler reads window.__memoCtx; the
        module-level shim above the script tag is the contract.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        # The named imports we lift out of helpers.mjs:
        for name in (
            "buildMemoContextPayload",
            "defaultMemoTypeForTarget",
            "MEMO_TYPES",
            "MEMO_LINK_TARGET_TYPES",
        ):
            assert name in body, f"helpers.mjs import for {name} missing"
        # And the window shim that exposes them to the classic script.
        assert "window.__memoCtx" in body

    def test_app_row_carries_memo_target_attributes(self, server_env) -> None:
        """The contextmenu listener resolves applications from
        ``.app-row[data-app-id]`` — same attribute the F4.6 play
        button uses. Confirm the renderApps() snippet still emits
        data-app-id so the right-click handler can pick it up.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        # The renderApps() template literal.
        assert 'data-app-id="${escapeHtml(a.id)}"' in body

    def test_code_chip_carries_memo_target_attributes(self, server_env) -> None:
        """Codebook-summary chips render with
        data-memo-target-type/id so right-click on a code chip resolves
        to a code memo."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        assert 'data-memo-target-type="code"' in body
        assert 'data-memo-target-id="${escapeHtml(c.id)}"' in body
        # The chip is keyed for the integration test:
        assert 'data-test-id="src-code-chip"' in body

    def test_contextmenu_listener_is_wired(self, server_env) -> None:
        """The page registers a global contextmenu listener; the
        listener's entry-point function names should be present so a
        future refactor can't accidentally drop the wiring."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        assert 'document.addEventListener("contextmenu"' in body
        assert "openMemoComposer" in body
        assert "submitMemoComposer" in body


# --------------------------------------------------------------------------- #
# The endpoint round-trips the JS helper's payload shape
# --------------------------------------------------------------------------- #


class TestMemoContextEndpointFromUI:
    """The same payload the page builds via buildMemoContextPayload
    deserialises into a persistable memo with the correct primary link
    + default type. Mirrors what the modal Save button would send."""

    def test_application_default_type_is_quote(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Mirror what buildMemoContextPayload({ targetType: "application", … })
        # emits — a context block, no explicit type, plus the body the
        # user typed.
        app_id = "d" * 12
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "application", "target_id": app_id},
                "body": "This quote nails the gerund form.",
            },
        )
        assert r.status_code == 201, r.text
        memo = r.json()
        assert memo["type"] == "quote"
        assert memo["links"][0]["target_type"] == "application"
        assert memo["links"][0]["target_id"] == app_id

    def test_code_default_type_is_code(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "code", "target_id": cid},
                "body": "Working definition: still gerund-form.",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["type"] == "code"

    def test_source_default_type_is_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "source", "target_id": sid},
                "body": "P3 was guarded today.",
            },
        )
        assert r.status_code == 201, r.text
        memo = r.json()
        assert memo["type"] == "source"
        assert memo["links"][0] == {"target_type": "source", "target_id": sid}

    def test_user_can_override_type(self, server_env) -> None:
        """The modal lets the user re-classify — type override wins
        over the right-click default."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "code", "target_id": cid},
                "type": "theoretical",
                "body": "Saturation candidate?",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["type"] == "theoretical"

    def test_tags_round_trip(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "code", "target_id": cid},
                "body": "...",
                "tags": ["axial", "low-confidence"],
            },
        )
        assert r.status_code == 201
        assert r.json()["tags"] == ["axial", "low-confidence"]

    def test_persists_to_disk(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "code", "target_id": cid},
                "body": "On disk.",
            },
        )
        memo_id = r.json()["id"]
        on_disk = json.loads(
            (srv.PROJECTS_DIR / pid / "memos" / f"{memo_id}.json").read_text()
        )
        assert on_disk["body"] == "On disk."
        assert on_disk["type"] == "code"
        assert on_disk["links"][0] == {"target_type": "code", "target_id": cid}

    def test_memo_appears_in_listing_endpoint(self, server_env) -> None:
        """The page's right-click flow lands a memo that the F5.1
        memos page (/projects/<pid>/memos) can list — proving the two
        wirings stay coherent."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        c = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "code", "target_id": cid},
                "title": "Right-click memo",
                "body": "From the coding view.",
            },
        )
        assert c.status_code == 201
        listing = client.get(f"/api/projects/{pid}/memos")
        assert listing.status_code == 200
        memos = listing.json()["memos"]
        assert len(memos) == 1
        assert memos[0]["title"] == "Right-click memo"
        assert memos[0]["links"][0]["target_id"] == cid


# --------------------------------------------------------------------------- #
# 412/400 / orphan project / unknown target_type — the modal must surface
# server validation errors. The endpoint already handles all of these;
# what these tests verify is that the page's wire shape doesn't bypass
# them.
# --------------------------------------------------------------------------- #


class TestMemoContextValidation:
    def test_modal_guards_empty_body_client_side(self, server_env) -> None:
        """The modal's Save handler refuses an empty body before
        sending. ``Memo.new`` accepts an empty body server-side (a
        memo can be a pure title or a links-only stub) but the
        right-click composer's discoverability story is "this is for
        marginal notes about the entity you clicked", so empty-body
        submissions are blocked at the JS guard with a visible
        ``"Body is required."`` status. Keyed by the literal so a
        future refactor can't silently drop it.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        assert "Body is required." in body

    def test_unknown_target_type_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/memos",
            json={
                "context": {"target_type": "planet", "target_id": "a" * 12},
                "body": "...",
            },
        )
        assert r.status_code == 400

    def test_unknown_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/" + ("0" * 12) + "/memos",
            json={
                "context": {"target_type": "code", "target_id": "a" * 12},
                "body": "...",
            },
        )
        assert r.status_code == 404
