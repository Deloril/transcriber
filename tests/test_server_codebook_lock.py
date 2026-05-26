"""Tests for the F2.4 codebook lock / unlock REST surface.

The pure module (`scribe.codebook_lock`) has its own deep test
coverage in `tests/test_codebook_lock.py` (57 tests). This file
exists solely to prove **reachability**: that the lock and unlock
toggles are wired through FastAPI and that the codebook editor
template surfaces controls that hit those endpoints.

Three concerns:

  1. ``GET /api/projects/<pid>/codebook/lock`` reports current state.
  2. ``POST /api/projects/<pid>/codebook/lock`` flips the toggle and
     records a lock event with the supplied reason. ``POST .../unlock``
     requires both a reason and a methodological memo; the spec calls
     this the "breaking the seal" invariant.
  3. The codebook editor template renders a banner + lock toggle
     button + audit log discloser, so the user can reach all three
     endpoints without leaving the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _new_project(client: TestClient, name: str = "Test project") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _lock(client: TestClient, pid: str, reason: str = "freezing for ICR"):
    return client.post(
        f"/api/projects/{pid}/codebook/lock", json={"reason": reason}
    )


def _unlock(
    client: TestClient,
    pid: str,
    *,
    reason: str = "reopened",
    methodological_memo: str = "noticed a missing distinction in the data",
    new_stage: str | None = None,
):
    payload: dict = {"reason": reason, "methodological_memo": methodological_memo}
    if new_stage is not None:
        payload["new_stage"] = new_stage
    return client.post(f"/api/projects/{pid}/codebook/unlock", json=payload)


# --------------------------------------------------------------------------- #
# GET lock state
# --------------------------------------------------------------------------- #


class TestGetLockState:
    def test_initial_state_is_unlocked(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/lock")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["locked"] is False
        assert body["stage"] == "initial"
        assert body["log"] == []

    def test_404_for_unknown_project(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/aaaaaaaaaaaa/codebook/lock")
        assert r.status_code == 404

    def test_400_for_invalid_project_id(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/!!!/codebook/lock")
        # Either 400 (shape) or 404 (not found) is acceptable; the
        # important contract is that we don't 500.
        assert r.status_code in (400, 404)

    def test_state_after_lock_includes_event(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = _lock(client, pid, reason="ready for final coding")
        assert r.status_code == 200, r.text
        r = client.get(f"/api/projects/{pid}/codebook/lock")
        body = r.json()
        assert body["locked"] is True
        assert body["stage"] == "locked"
        assert len(body["log"]) == 1
        ev = body["log"][0]
        assert ev["action"] == "lock"
        assert ev["reason"] == "ready for final coding"
        assert ev["new_stage"] == "locked"
        assert ev["prior_stage"] == "initial"


# --------------------------------------------------------------------------- #
# POST lock
# --------------------------------------------------------------------------- #


class TestLockEndpoint:
    def test_lock_round_trips(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = _lock(client, pid, reason="freezing for ICR")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project"]["codebook_stage"] == "locked"
        assert body["event"]["action"] == "lock"
        assert body["event"]["reason"] == "freezing for ICR"

    def test_lock_requires_non_empty_reason(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codebook/lock", json={"reason": "   "}
        )
        assert r.status_code == 400
        r = client.post(f"/api/projects/{pid}/codebook/lock", json={})
        assert r.status_code == 400

    def test_lock_rejects_invalid_json(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codebook/lock",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_lock_409_if_already_locked(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="first").status_code == 200
        r = _lock(client, pid, reason="second")
        assert r.status_code == 409
        assert "already locked" in r.json()["detail"].lower()

    def test_lock_blocks_create_code(self, env) -> None:
        """The end-to-end purpose of the lock: structural writes are
        refused while the codebook is locked."""
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "pacing"})
        assert r.status_code == 409, r.text
        assert "locked" in r.json()["detail"].lower()

    def test_lock_404_for_unknown_project(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/codebook/lock",
            json={"reason": "ok"},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# POST unlock
# --------------------------------------------------------------------------- #


class TestUnlockEndpoint:
    def test_unlock_round_trips(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = _unlock(
            client, pid,
            reason="reopened after pilot",
            methodological_memo="participants raised a pacing/spoons distinction we hadn't surfaced",
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Default new_stage walks the lock log back to "initial".
        assert body["project"]["codebook_stage"] == "initial"
        assert body["event"]["action"] == "unlock"
        assert body["event"]["reason"] == "reopened after pilot"
        assert "pacing" in body["event"]["methodological_memo"]

    def test_unlock_honours_explicit_new_stage(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = _unlock(
            client, pid,
            new_stage="focused",
        )
        assert r.status_code == 200, r.text
        assert r.json()["project"]["codebook_stage"] == "focused"

    def test_unlock_requires_reason(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = client.post(
            f"/api/projects/{pid}/codebook/unlock",
            json={"reason": "  ", "methodological_memo": "ok"},
        )
        assert r.status_code == 400

    def test_unlock_requires_methodological_memo(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = client.post(
            f"/api/projects/{pid}/codebook/unlock",
            json={"reason": "ok", "methodological_memo": ""},
        )
        assert r.status_code == 400

    def test_unlock_409_if_not_locked(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        # Project is created in 'initial', not 'locked'.
        r = _unlock(client, pid)
        assert r.status_code == 409
        assert "not locked" in r.json()["detail"].lower()

    def test_unlock_rejects_locked_as_new_stage(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = _unlock(client, pid, new_stage="locked")
        assert r.status_code == 400

    def test_unlock_followed_by_create_code_is_allowed(self, env) -> None:
        """Round-trip: lock, refuse code creation, unlock, succeed."""
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "pacing"})
        assert r.status_code == 409
        assert _unlock(client, pid).status_code == 200
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "pacing"})
        assert r.status_code == 201, r.text


# --------------------------------------------------------------------------- #
# Codebook editor template surfaces the controls
# --------------------------------------------------------------------------- #


class TestCodebookEditorRendersLockControls:
    """The codebook editor must surface F2.4's lock toggle and audit log
    so the routes above are reachable through the UI.

    Like the F2.3 lifecycle test, we assert against the rendered HTML
    rather than the JS module so a refactor that breaks the user-facing
    surface fails here, not silently."""

    def test_page_renders_with_lock_banner(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        text = r.text

        # The lock banner container is present and explicitly tagged
        # with the F2.4 feature ref so future audits can find it.
        assert 'id="cb-lock-banner"' in text
        assert 'data-test-feature="F2.4"' in text

        # The toggle button + the audit log discloser button.
        assert 'id="cb-lock-toggle-btn"' in text
        assert 'id="cb-lock-log-btn"' in text

        # The audit log container.
        assert 'id="cb-lock-log"' in text

        # The JS hits both the GET and the POSTs.
        assert "/codebook/lock" in text
        assert "/codebook/unlock" in text

        # The "🔒 Lock codebook" affordance is the default label
        # (rendered into the JS branch, but the literal must be in
        # the page so a server-side rendered snapshot can find it).
        assert "Lock codebook" in text
        assert "Unlock with reason" in text

    def test_locked_codebook_still_renders_page(self, env) -> None:
        """A locked codebook shouldn't break the editor render — the
        banner still loads and the toggle button switches to the
        unlock variant via the JS state machine."""
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="freezing for ICR").status_code == 200
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        # The page itself doesn't pre-render the locked state into
        # HTML (the JS calls /codebook/lock on load), so we only
        # assert the controls are still there.
        assert 'id="cb-lock-banner"' in r.text
        assert 'id="cb-lock-toggle-btn"' in r.text
