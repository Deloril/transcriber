"""Tests for the click-to-edit-code surface in the source coding view.

Clicking a highlighted word (.word.is-applied) opens an "edit codes
on this span" popover with two affordances per covering application:
Change… (replace with a different code) and Remove. The replace path
is implemented as DELETE old + POST new because Application.code_id
is intentionally not patchable in place (audit-trail semantics —
see scribe/applications.py::Application.apply_update).

This file pins the user-facing surface:

* the popover element renders into the template,
* the data-test-id markers anchor the affordance for future
  refactors,
* the underlying API actions (POST + DELETE) accept the recode
  shape this client sends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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


def _make_project(client: TestClient, name: str = "Pilot") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    *,
    name: str = "Maria",
    job_id: str | None = None,
) -> str:
    body: dict = {"name": name, "source_type": "transcript"}
    if job_id is not None:
        body["transcript_job_id"] = job_id
    r = client.post(f"/api/projects/{pid}/sources", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": f"def of {name}"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _apply(
    client: TestClient,
    pid: str,
    cid: str,
    sid: str,
    start: str,
    end: str,
) -> dict:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": cid,
            "source_id": sid,
            "anchor_start_word_id": start,
            "anchor_end_word_id": end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


class TestEditPopoverRendersInTemplate:
    """Without these markers a future refactor could silently strip
    the click-to-edit surface; the popover element + button slots
    must keep their data-test-id anchors."""

    def test_edit_popover_element_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # Container.
        assert 'id="editPopover"' in body
        # Header + sub-state container.
        assert 'data-test-id="edit-popover-header"' in body
        assert 'data-test-id="edit-popover-pick-header"' in body
        assert 'data-test-id="edit-popover-pick-list"' in body

    def test_edit_popover_handlers_wired(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        # The two action verbs the popover dispatches on.
        assert 'data-act="change"' in body
        assert 'data-act="remove"' in body
        # The recode helper that the Change → pick flow calls.
        assert "recodeApplication" in body
        assert "removeApplicationById" in body
        # The click delegate that opens the popover when a coded
        # word is clicked.
        assert "applicationsCoveringWord" in body
        assert ".word.is-applied" in body


class TestRecodeRoundTripViaApi:
    """The client's recode flow is DELETE + POST. Pin that those
    endpoints accept the shape the client sends and the resulting
    application lookup reflects the new code."""

    def test_post_then_delete_swaps_the_code_on_a_span(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        old_cid = _make_code(client, pid, "managing pain")
        new_cid = _make_code(client, pid, "asking for help")

        # Initial application
        old = _apply(client, pid, old_cid, sid, "s0w0", "s0w3")

        # Client recode: POST new, then DELETE old.
        new_app = _apply(client, pid, new_cid, sid, "s0w0", "s0w3")
        dr = client.delete(f"/api/projects/{pid}/applications/{old['id']}")
        assert dr.status_code == 200, dr.text

        # Survivor list should carry the new application only.
        r = client.get(
            f"/api/projects/{pid}/applications?source_id={sid}"
        )
        assert r.status_code == 200, r.text
        apps = r.json()["applications"]
        ids = [a["id"] for a in apps]
        codes = [a["code_id"] for a in apps]
        assert old["id"] not in ids
        assert new_app["id"] in ids
        assert new_cid in codes
        assert old_cid not in codes
