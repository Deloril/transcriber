"""End-to-end reachability tests for F4.1 (Application entity with
word-id anchored persistence).

Background
----------

F4.1 ships the on-disk Application entity (``scribe/applications.py``)
with full unit coverage in ``tests/test_applications.py``. The HTTP
contract for ``/api/projects/<pid>/applications*`` is exercised in
``tests/test_codes_applications_rest.py::TestApplicationsREST``.

What was missing — and what this file covers — is the integration
proof that the **user-facing surface** is wired together. Per the
loop's done-criteria, F4.1 is only "done" if a researcher can reach
the data layer through a real route + a real UI control. That means:

1. ``GET /projects/<pid>/sources/<sid>`` must render the coding view
   (``source_coding.html``) with the application UI surface visible:
   ``Coded segments`` heading + ``#appList`` panel + ``#appCount``
   badge + ``#applyPopover`` dialog.
2. The rendered page must reference the F4.1 endpoints via the JS
   that wires the popover ↔ side panel to the data layer.
3. POST → GET-filtered-by-source → DELETE round-trips through the
   exact endpoints the page consumes, with the anchor / coder /
   definition-version invariants the F4.1 docstring promises.

Without this file the F4.1 ID would be in the commit log but with
no proof the user can reach the data model — exactly the failure
mode the loop's done-detector is designed to catch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures — mirror the pattern in test_server_sources.py / test_server_*.py
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


def _make_project(client: TestClient, name: str = "F4.1 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str = "managing pain") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "moments where the participant copes"},
    )
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
# 1. Coding-view template surfaces the F4.1 application UI controls.
# --------------------------------------------------------------------------- #


class TestSourceCodingTemplateExposesApplicationsUI:
    """Without these controls in the rendered page, a user can't
    reach the F4.1 data layer no matter how clean the routes are.
    Each assertion targets one of the three controls the JS binds to:
    - the side-panel list of applications (``#appList`` + ``#appCount``)
    - the floating apply-code popover (``#applyPopover``)
    - the ``Coded segments`` heading anchoring the side panel
    """

    def test_coding_view_renders_coded_segments_panel(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        assert "Coded segments" in r.text
        # The id the side-panel JS binds to.
        assert 'id="appList"' in r.text
        assert 'id="appCount"' in r.text

    def test_coding_view_renders_apply_code_popover(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # The popover is the affordance that calls POST /applications.
        assert 'id="applyPopover"' in r.text
        assert 'aria-label="Apply code"' in r.text

    def test_coding_view_references_applications_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # Both reads (filtered by source_id) and writes route through
        # /api/projects/<pid>/applications. The JS uses backtick template
        # literals, so we look for the path fragment after substitution.
        assert "/api/projects/${PROJECT_ID}/applications" in r.text
        assert "source_id=${encodeURIComponent(SOURCE_ID)}" in r.text

    def test_coding_view_renders_remove_application_action(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        # The DELETE route is wired via the per-row "Remove" button.
        # Match the JS that sets method:"DELETE" on the application id.
        assert "/api/projects/${PROJECT_ID}/applications/${aid}" in r.text


# --------------------------------------------------------------------------- #
# 2. End-to-end: the routes the coding view consumes round-trip an
#    application through the F4.1 data model.
# --------------------------------------------------------------------------- #


class TestApplicationRoundTripThroughCodingViewEndpoints:
    """This is the full circuit: the POST that the popover fires,
    the GET-by-source that the side panel fires on load, and the
    DELETE the per-row Remove button fires. They have to round-trip
    against the F4.1 data layer with the invariants the docstring
    promises (closed [start,end] anchor interval, coder auto-assigned,
    definition-version snapshot recorded)."""

    def test_post_creates_application_with_f41_invariants(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Equivalent to the popover's applyCode() call.
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid,
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w12",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        # The four foreign-key fields F4.1 requires are present.
        assert body["code_id"] == cid
        assert body["source_id"] == sid
        assert body["coder_id"], "coder_id must auto-resolve to the project default"
        # Anchor interval is closed [start,end] over canonical word ids.
        assert body["anchor_start_word_id"] == "s0w0"
        assert body["anchor_end_word_id"] == "s0w12"
        # Definition-version snapshot recorded (F4.1 → F2.2 invariant).
        assert body["definition_version_id_at_apply"]
        assert re.match(r"^[a-f0-9]{12}$", body["definition_version_id_at_apply"])
        # And the application id has the same shape used by the URL.
        assert re.match(r"^[a-f0-9]{12}$", body["id"])

        # On disk where load_application expects it.
        on_disk = json.loads(
            (srv.PROJECTS_DIR / pid / "applications" / f"{body['id']}.json").read_text()
        )
        assert on_disk["code_id"] == cid
        assert on_disk["anchor_start_word_id"] == "s0w0"

    def test_list_filters_by_source_for_side_panel(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid_a = _make_source(client, pid, "Interview A")
        sid_b = _make_source(client, pid, "Interview B")

        # Two apps on A, one on B.
        for w_end in ("s0w3", "s0w7"):
            client.post(
                f"/api/projects/{pid}/applications",
                json={
                    "code_id": cid, "source_id": sid_a,
                    "anchor_start_word_id": "s0w0", "anchor_end_word_id": w_end,
                },
            )
        client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid, "source_id": sid_b,
                "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w2",
            },
        )

        # The side panel calls GET /applications?source_id=<sid> on load.
        r = client.get(f"/api/projects/{pid}/applications?source_id={sid_a}")
        assert r.status_code == 200
        apps = r.json()["applications"]
        assert len(apps) == 2
        assert all(a["source_id"] == sid_a for a in apps)

    def test_delete_round_trips_through_remove_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid, "source_id": sid,
                "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w1",
            },
        )
        aid = r.json()["id"]

        # Equivalent to clicking "Remove" on the side-panel row.
        r = client.delete(f"/api/projects/{pid}/applications/{aid}")
        assert r.status_code == 200

        # Side panel re-fetches: the row is gone.
        r = client.get(f"/api/projects/{pid}/applications?source_id={sid}")
        assert r.status_code == 200
        assert r.json()["applications"] == []

    def test_create_records_human_provenance_by_default(self, server_env) -> None:
        """F4.1's provenance vocabulary is closed: human / ai_accepted /
        ai_modified / imported / other. A bare POST from the popover
        records no provenance source — the human-coder default — which
        F8.9 / F9.6 use to filter AI-mediated work out of human counts.
        """
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid, "source_id": sid,
                "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w1",
            },
        )
        body = r.json()
        # provenance is a dict (possibly empty) — never null.
        assert isinstance(body.get("provenance", {}), dict)
        # No "source" key means human-coder origin per the docstring.
        assert "source" not in (body.get("provenance") or {})
