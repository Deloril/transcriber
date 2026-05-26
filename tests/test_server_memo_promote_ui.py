"""Tests for the F5.5 memo → code promote UI surface.

PLANNING.md F5.5:

  > Promote a memo into a code definition (one click).

Pure module + endpoint shipped earlier (commits ee0db0b, 5166e16); the
``/api/projects/<pid>/memos/<mid>/promote-to-code`` route was reachable
only by ``curl`` until this UI surface landed. This test file proves
three things end-to-end:

  1. The memos page renders a "↗ Promote to code" button on the edit
     form (data-test-id="memos-promote-to-code", data-test-feature
     ="F5.5") so the action is reachable from the user-facing surface.
  2. The endpoint behind the button accepts the same minimal payload
     the JS helper produces (``{}`` or ``{"name": "..."}``) and writes
     a real Code into the project.
  3. After a successful POST the new Code shows up on the codebook
     editor page — confirming the redirect target is meaningful.

Deeper coverage of the endpoint's many shapes (overrides, errors,
locked codebook) lives in tests/test_server.py::TestPromoteMemoToCodeAPI;
this file's job is purely reachability through the FastAPI surface.
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
    title: str = "Managing the project",
    body: str = "Notes about pacing and load.",
    type: str = "free",
) -> str:
    r = client.post(
        f"/api/projects/{project_id}/memos",
        json={
            "type": type,
            "title": title,
            "body": body,
            "body_format": "markdown",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Template render: the Promote button is in the page
# --------------------------------------------------------------------------- #


class TestMemosPageRendersPromoteButton:
    """The memos page must surface F5.5's promote action so the
    promote-to-code endpoint is reachable without leaving the page."""

    def test_page_renders_with_promote_button(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        assert r.status_code == 200, r.text
        text = r.text

        assert 'id="mmPromote"' in text
        assert 'data-test-feature="F5.5"' in text
        assert 'data-test-id="memos-promote-to-code"' in text
        assert "Promote to code" in text

    def test_button_has_explanatory_title(self, env) -> None:
        """The hover title carries enough context that the action's
        consequence is visible without a click."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        text = r.text
        # The wording must mention "codebook" so the user understands
        # where the code lands. The F5.5 marker keeps the link to the
        # planning doc explicit.
        assert "codebook" in text.lower()
        assert "F5.5" in text

    def test_helper_imported_for_payload_build(self, env) -> None:
        """The page must import buildPromoteMemoPayload from helpers.mjs
        so the wire body matches scribe/memo_promote.py's shape."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        text = r.text
        assert "buildPromoteMemoPayload" in text
        # And it has to come from the canonical helpers module.
        assert "/static/js/helpers.mjs" in text

    def test_button_posts_to_promote_endpoint(self, env) -> None:
        """The page's JS must reference the live endpoint URL so the
        click handler can't silently target the wrong route."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/memos")
        text = r.text
        # Path template must be present — exact /memos/<id>/promote-to-code.
        assert "/promote-to-code" in text
        assert "/api/projects/${PROJECT_ID}/memos/" in text


# --------------------------------------------------------------------------- #
# Reachability: the endpoint actually accepts what the page sends
# --------------------------------------------------------------------------- #


class TestPromoteFromPageReachable:
    """The Promote button POSTs the same shape ``buildPromoteMemoPayload``
    builds. These tests fire those shapes against the live endpoint to
    prove the round-trip works through the FastAPI surface."""

    def test_empty_body_promotes_with_server_defaults(self, env) -> None:
        """The "promote with no overrides" path: the JS helper produces
        ``{}`` when the user hasn't typed a title. The server fills
        every field from memo_promote.py defaults."""
        client, _ = env
        pid = _new_project(client)
        mid = _create_memo(client, pid, title="Managing the project")

        r = client.post(
            f"/api/projects/{pid}/memos/{mid}/promote-to-code",
            json={},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["code"]["name"] == "Managing the project"
        assert body["code"]["provenance"]["source"] == "promoted_from_memo"
        assert body["code"]["provenance"]["memo_id"] == mid
        assert body["version"]["version"] == 1

    def test_name_override_from_title_field(self, env) -> None:
        """When the user has typed in the title field the JS forwards
        ``{"name": title}``. The server uses that name verbatim."""
        client, _ = env
        pid = _new_project(client)
        mid = _create_memo(client, pid, title="Old title", body="Body.")

        r = client.post(
            f"/api/projects/{pid}/memos/{mid}/promote-to-code",
            json={"name": "Renamed in the form"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["code"]["name"] == "Renamed in the form"

    def test_promoted_code_appears_on_codebook_page(self, env) -> None:
        """After promote, the code is visible on the codebook editor —
        which is also where the page redirects on success. This is the
        end-to-end reachability check the loop's done-criteria
        demands."""
        client, _ = env
        pid = _new_project(client)
        mid = _create_memo(client, pid, title="Pacing")

        r = client.post(
            f"/api/projects/{pid}/memos/{mid}/promote-to-code",
            json={},
        )
        assert r.status_code == 201, r.text
        cid = r.json()["code"]["id"]

        # The codebook list endpoint should now include the new code.
        r2 = client.get(f"/api/projects/{pid}/codes")
        assert r2.status_code == 200, r2.text
        ids = {c["id"] for c in r2.json().get("codes", [])}
        assert cid in ids

        # And the codebook editor page (the redirect target) renders.
        r3 = client.get(f"/projects/{pid}/codebook")
        assert r3.status_code == 200

    def test_back_link_recorded_on_memo(self, env) -> None:
        """The default behaviour of the helper records a memo→code
        back-link with role 'promoted_to'. Hitting the endpoint with
        an empty body must preserve that audit trail."""
        client, _ = env
        pid = _new_project(client)
        mid = _create_memo(client, pid, title="Pacing")

        r = client.post(
            f"/api/projects/{pid}/memos/{mid}/promote-to-code",
            json={},
        )
        assert r.status_code == 201
        cid = r.json()["code"]["id"]

        # Read back the memo and confirm the link landed.
        r2 = client.get(f"/api/projects/{pid}/memos/{mid}")
        assert r2.status_code == 200
        memo = r2.json()
        assert any(
            link["target_type"] == "code"
            and link["target_id"] == cid
            and link.get("role") == "promoted_to"
            for link in memo.get("links", [])
        )
