"""End-to-end reachability tests for F9.5 (Locked-codebook mode with
reason-to-unlock memo, audit-integrated).

The pure module ``scribe.codebook_lock_audit`` shipped in 0cc18b0
with 42 unit tests covering the audit-integrated lock / unlock
wrappers plus four read-side helpers (``find_unlock_memos``,
``latest_unlock_memo``, ``find_codebook_lock_events``,
``reconcile_unlock_artefacts``). That commit explicitly deferred the
HTTP / FastAPI surface; until this iteration landed, the F9.5 audit
artefacts (F9.1 event on every toggle, F5.1 methodological memo on
every unlock) were unreachable through the user-facing surface.

This file proves the F9.5 wiring end to end:

  * ``POST /api/projects/<pid>/codebook/lock`` now emits an F9.1
    event and surfaces the event id in the response (via the new
    ``audit_event`` field).
  * ``POST /api/projects/<pid>/codebook/unlock`` now creates an
    F5.1 methodological memo with role ``codebook_unlock`` *and* an
    F9.1 event whose ``after`` payload carries the lock-event id +
    memo id, and surfaces both in the response.
  * ``GET /api/projects/<pid>/codebook/lock/audit`` returns the
    cross-referenced rows so the audit page can render a single
    timeline of every lock toggle joined to its event + memo.
  * The ``/projects/<pid>/audit`` page renders an F9.5 panel
    (``data-test-feature="F9.5"``) so the route above is reachable
    from the user-facing surface, not just curl.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up an isolated TestClient with tmp project dirs."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
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

    client = TestClient(srv.app)
    return client, projects


def _new_project(client: TestClient, name: str = "F9.5 reachability") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _lock(
    client: TestClient,
    pid: str,
    *,
    reason: str = "Phase B coding done; freezing for ICR",
    actor_coder_id: str = "",
):
    payload: dict = {"reason": reason}
    if actor_coder_id:
        payload["actor_coder_id"] = actor_coder_id
    return client.post(
        f"/api/projects/{pid}/codebook/lock", json=payload
    )


def _unlock(
    client: TestClient,
    pid: str,
    *,
    reason: str = "Pilot review surfaced a missed dimension",
    methodological_memo: str = (
        "Two interviews after the lock raised a pacing / spoons "
        "distinction we had not surfaced. Reopening focused stage "
        "to add a code; will re-lock after one more pass."
    ),
    new_stage: str | None = None,
    actor_coder_id: str = "",
    author_coder_id: str = "",
):
    payload: dict = {
        "reason": reason,
        "methodological_memo": methodological_memo,
    }
    if new_stage is not None:
        payload["new_stage"] = new_stage
    if actor_coder_id:
        payload["actor_coder_id"] = actor_coder_id
    if author_coder_id:
        payload["author_coder_id"] = author_coder_id
    return client.post(
        f"/api/projects/{pid}/codebook/unlock", json=payload
    )


# --------------------------------------------------------------------------- #
# F9.5 — lock toggle now emits F9.1 event
# --------------------------------------------------------------------------- #


class TestLockEndpointEmitsF91Event:
    def test_lock_response_carries_audit_event(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = _lock(client, pid, reason="freezing for ICR")
        assert r.status_code == 200, r.text
        body = r.json()
        # Backwards-compatible F2.4 fields still present.
        assert body["project"]["codebook_stage"] == "locked"
        assert body["event"]["action"] == "lock"
        # New F9.5 surface: the F9.1 event lands in the response.
        assert "audit_event" in body, body
        ev = body["audit_event"]
        assert ev["action"] == "lock"
        assert ev["entity_type"] == "codebook"
        assert ev["notes"] == "freezing for ICR"
        # The before / after diff captures the stage transition.
        assert ev["before"]["codebook_stage"] == "initial"
        assert ev["after"]["codebook_stage"] == "locked"
        # The audit_event id cross-references the F2.4 lock-log entry.
        assert ev["after"]["lock_event_id"] == body["event"]["id"]

    def test_lock_event_visible_in_events_feed(self, env) -> None:
        """The lock toggle now lands in the F9.1 events feed so the
        audit timeline page picks it up automatically."""
        client, _ = env
        pid = _new_project(client)
        before = client.get(f"/api/projects/{pid}/events").json()
        before_count = before["total"]

        r = _lock(client, pid, reason="locking for second-coder pass")
        assert r.status_code == 200

        after = client.get(f"/api/projects/{pid}/events").json()
        # +1 event recorded.
        assert after["total"] == before_count + 1
        # And the new event is the lock toggle.
        events = after["events"]
        assert any(
            ev["action"] == "lock"
            and ev["entity_type"] == "codebook"
            and ev["notes"] == "locking for second-coder pass"
            for ev in events
        ), events

    def test_actor_coder_id_flows_to_event(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        coder_id = "0123456789ab"
        r = _lock(client, pid, actor_coder_id=coder_id)
        assert r.status_code == 200, r.text
        ev = r.json()["audit_event"]
        assert ev["actor_coder_id"] == coder_id


# --------------------------------------------------------------------------- #
# F9.5 — unlock now creates a methodological memo + F9.1 event
# --------------------------------------------------------------------------- #


class TestUnlockEndpointEmitsMemoAndEvent:
    def test_unlock_response_carries_memo_and_audit_event(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = _unlock(client, pid)
        assert r.status_code == 200, r.text
        body = r.json()
        # Backwards-compatible F2.4 fields still present.
        assert body["event"]["action"] == "unlock"
        # F5.1 memo surfaces in the response.
        assert "memo" in body, body
        memo = body["memo"]
        assert memo["type"] == "methodological"
        assert "pacing" in memo["body"]
        # The memo title prefix is the F9.5 vocabulary.
        assert memo["title"].startswith("Codebook unlock:")
        # The link role anchors it to the project as an unlock memo.
        link = memo["links"][0]
        assert link["target_type"] == "project"
        assert link["target_id"] == pid
        assert link["role"] == "codebook_unlock"
        # The provenance threads back to the F2.4 lock event id.
        prov = memo["provenance"]
        assert prov.get("source") == "other"
        assert prov.get("codebook_lock_event_id") == body["event"]["id"]

        # F9.1 event lands and cross-references both the lock event +
        # the new memo.
        ev = body["audit_event"]
        assert ev["action"] == "unlock"
        assert ev["entity_type"] == "codebook"
        assert ev["after"]["lock_event_id"] == body["event"]["id"]
        assert ev["after"]["memo_id"] == memo["id"]

    def test_unlock_memo_appears_in_memos_list(self, env) -> None:
        """The unlock memo is a real F5.1 memo, not a side-channel
        log entry — so it shows up in the memos list endpoint and
        therefore on the memos page / memo exports / REFI-QDA bundle."""
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = _unlock(client, pid)
        assert r.status_code == 200, r.text
        memo_id = r.json()["memo"]["id"]

        listing = client.get(
            f"/api/projects/{pid}/memos",
            params={"type": "methodological"},
        )
        assert listing.status_code == 200, listing.text
        ids = [m["id"] for m in listing.json()["memos"]]
        assert memo_id in ids, ids

    def test_unlock_event_visible_in_events_feed(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        before = client.get(f"/api/projects/{pid}/events").json()
        before_count = before["total"]

        r = _unlock(client, pid)
        assert r.status_code == 200

        after = client.get(f"/api/projects/{pid}/events").json()
        assert after["total"] == before_count + 1
        events = after["events"]
        assert any(
            ev["action"] == "unlock" and ev["entity_type"] == "codebook"
            for ev in events
        ), events

    def test_actor_and_author_coder_id_flow_through(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        actor = "abcdef012345"
        author = "fedcba543210"
        r = _unlock(
            client, pid,
            actor_coder_id=actor,
            author_coder_id=author,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["audit_event"]["actor_coder_id"] == actor
        assert body["memo"]["author_coder_id"] == author


# --------------------------------------------------------------------------- #
# F9.5 — GET /codebook/lock/audit endpoint
# --------------------------------------------------------------------------- #


class TestLockAuditEndpoint:
    def test_fresh_project_returns_empty_rows(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/lock/audit")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["locked"] is False
        assert body["stage"] == "initial"
        assert body["rows"] == []
        assert body["events"] == []
        assert body["memos"] == []

    def test_after_lock_unlock_round_trip(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        lock_resp = _lock(client, pid, reason="lock1")
        assert lock_resp.status_code == 200
        unlock_resp = _unlock(client, pid)
        assert unlock_resp.status_code == 200

        r = client.get(f"/api/projects/{pid}/codebook/lock/audit")
        assert r.status_code == 200, r.text
        body = r.json()
        # Two rows: the lock + the unlock.
        assert len(body["rows"]) == 2
        actions = [row["action"] for row in body["rows"]]
        assert actions == ["lock", "unlock"]

        # Lock row: no memo (locks don't carry an unlock memo) + an
        # F9.1 event id that matches the lock's audit_event response.
        lock_row = body["rows"][0]
        assert lock_row["lock_event_id"] == lock_resp.json()["event"]["id"]
        assert lock_row["memo_id"] == ""
        assert lock_row["event_id"] == lock_resp.json()["audit_event"]["id"]
        assert lock_row["prior_stage"] == "initial"
        assert lock_row["new_stage"] == "locked"
        assert lock_row["reason"] == "lock1"

        # Unlock row: memo + event ids both populated.
        unlock_row = body["rows"][1]
        assert unlock_row["lock_event_id"] == unlock_resp.json()["event"]["id"]
        assert unlock_row["memo_id"] == unlock_resp.json()["memo"]["id"]
        assert unlock_row["event_id"] == unlock_resp.json()["audit_event"]["id"]
        assert unlock_row["prior_stage"] == "locked"
        # Default new_stage walks back to the most recent prior stage.
        assert unlock_row["new_stage"] == "initial"
        assert "pacing" in unlock_row["methodological_memo"]

        # The events / memos arrays are also returned for the panel
        # to enrich its rendering without a second fetch.
        assert len(body["events"]) == 2
        assert len(body["memos"]) == 1

    def test_404_for_unknown_project(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/aaaaaaaaaaaa/codebook/lock/audit")
        assert r.status_code == 404

    def test_400_for_invalid_project_id(self, env) -> None:
        client, _ = env
        r = client.get("/api/projects/not-a-real-id/codebook/lock/audit")
        assert r.status_code == 400

    def test_state_field_tracks_current_lock(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        # While locked, ``locked: true`` and stage is ``locked``.
        assert _lock(client, pid, reason="locked").status_code == 200
        r = client.get(f"/api/projects/{pid}/codebook/lock/audit")
        body = r.json()
        assert body["locked"] is True
        assert body["stage"] == "locked"
        # After unlock the state flips back.
        assert _unlock(client, pid).status_code == 200
        r = client.get(f"/api/projects/{pid}/codebook/lock/audit")
        body = r.json()
        assert body["locked"] is False
        assert body["stage"] == "initial"


# --------------------------------------------------------------------------- #
# F9.5 — audit page renders the lock-history panel
# --------------------------------------------------------------------------- #


class TestAuditPageRendersLockAuditPanel:
    """The /projects/<pid>/audit page must render the F9.5 panel so
    the route + JS that consume it are reachable from the user-facing
    surface."""

    def test_panel_present_on_audit_page(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200, r.text
        text = r.text
        # The F9.5 panel + its data-test markers.
        assert 'data-test-feature="F9.5"' in text
        assert 'data-test-id="lock-audit-panel"' in text
        assert 'data-test-id="la-list"' in text
        assert 'data-test-id="la-empty"' in text
        # Header copy nudges the user to the codebook editor where
        # the toggle lives.
        assert "Codebook lock history" in text
        assert "/projects/{pid}/codebook".format(pid=pid) in text

    def test_panel_js_targets_audit_endpoint(self, env) -> None:
        """The panel's JS must hit /codebook/lock/audit so the route
        wired above is actually consumed by the rendered page."""
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/audit")
        assert r.status_code == 200
        assert "/codebook/lock/audit" in r.text


# --------------------------------------------------------------------------- #
# F9.5 — old request shape (no actor_coder_id) still works
# --------------------------------------------------------------------------- #


class TestBackwardsCompatibility:
    """The F2.4 contract — request shape, response shape — must still
    hold so existing JS / tests are not broken by the F9.5 wiring."""

    def test_lock_without_actor_works(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codebook/lock",
            json={"reason": "freezing"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["event"]["action"] == "lock"
        assert body["audit_event"]["actor_coder_id"] == ""

    def test_unlock_without_actor_or_author_works(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        assert _lock(client, pid, reason="ready").status_code == 200
        r = client.post(
            f"/api/projects/{pid}/codebook/unlock",
            json={
                "reason": "reopen",
                "methodological_memo": "Reopening: pilot insights",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # ``author_coder_id`` on the Memo dataclass is Optional[str] —
        # serialised as null when unset. The F9.1 Event uses the
        # canonical empty-string sentinel.
        assert body["memo"]["author_coder_id"] in ("", None)
        assert body["audit_event"]["actor_coder_id"] == ""
