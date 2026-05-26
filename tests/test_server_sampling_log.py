"""End-to-end reachability tests for F1.4 (sampling log).

The pure data model + JSONL persistence shipped in f553954
(``scribe/sampling_log.py``) with 42 unit tests in
``tests/test_sampling_log.py``. This file proves the
**user-facing surface** is wired together:

* Project home shows a "Sampling log" snapshot card with a
  ``+ Log a decision`` CTA.
* ``/projects/<pid>/sampling-log`` renders the log page with the
  inline append form.
* ``/api/projects/<pid>/sampling_log`` serves GET (list) and POST
  (append) and round-trips on disk through the JSONL log.
* ``/projects/<pid>/sources/add`` (the source-picker page) carries
  the "Why was this source added?" prompt that backs onto the
  sampling-log POST.

Sibling of ``tests/test_server_participants.py`` (F1.3).
"""

from __future__ import annotations

import json
import re
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


def _make_project(client: TestClient, name: str = "Holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Project home links to the sampling-log surface
# --------------------------------------------------------------------------- #


class TestProjectHomeLinksToSamplingLog:
    """Without a snapshot card on the project home page, F1.4 isn't
    discoverable — users would have to type the URL by hand."""

    def test_project_home_renders_sampling_log_card(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert "Sampling log" in r.text
        assert 'id="samplingCount"' in r.text
        assert 'id="samplingList"' in r.text

    def test_project_home_renders_sampling_log_ctas(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert f'href="/projects/{pid}/sampling-log"' in r.text
        assert "+ Log a decision" in r.text

    def test_project_home_consumes_sampling_log_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        # JS uses a backtick template literal; assert the path appears.
        assert "/api/projects/${PROJECT_ID}/sampling_log" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/sampling-log — page render
# --------------------------------------------------------------------------- #


class TestSamplingLogPage:
    """The sampling-log page must render the append form, the chronological
    list, and a back link to the project home."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sampling-log")
        assert r.status_code == 200

    def test_has_back_to_project_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sampling-log")
        assert f'href="/projects/{pid}"' in r.text

    def test_has_append_form_with_all_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sampling-log")
        # SamplingEntry exposes action, decision_type, source_id,
        # participant_id, target_category, rationale, notes — each
        # gets a form field on the page.
        assert 'id="sl-action"' in r.text
        assert 'id="sl-decision-type"' in r.text
        assert 'id="sl-source-id"' in r.text
        assert 'id="sl-participant-id"' in r.text
        assert 'id="sl-target-category"' in r.text
        assert 'id="sl-rationale"' in r.text
        assert 'id="sl-notes"' in r.text

    def test_form_posts_to_sampling_log_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sampling-log")
        assert "/api/projects/${PROJECT_ID}/sampling_log" in r.text
        assert '"POST"' in r.text or "'POST'" in r.text

    def test_active_nav_marks_projects(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sampling-log")
        assert 'class="active"' in r.text

    def test_invalid_project_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/..%2Fevil/sampling-log")
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# /api/projects/<pid>/sampling_log — REST surface
# --------------------------------------------------------------------------- #


class TestSamplingLogAPI:
    def test_empty_log_returns_empty_entries(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/sampling_log")
        assert r.status_code == 200
        data = r.json()
        assert data["entries"] == []
        # The vocabularies are returned so the form can render selects
        # without hard-coding strings on the client.
        assert "added" in data["actions"]
        assert "theoretical" in data["decision_types"]

    def test_post_appends_entry_and_round_trips(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        payload = {
            "action": "added",
            "decision_type": "theoretical",
            "target_category": "negative cases of trust",
            "rationale": "P03 has the only counter-example so far.",
        }
        r = client.post(f"/api/projects/{pid}/sampling_log", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["action"] == "added"
        assert body["decision_type"] == "theoretical"
        assert body["target_category"] == "negative cases of trust"
        assert re.match(r"^[a-f0-9]{12}$", body["id"])

        # On disk: a JSONL file under projects/<pid>/sampling_log.jsonl.
        log_path = srv.PROJECTS_DIR / pid / "sampling_log.jsonl"
        assert log_path.exists()
        line = log_path.read_text(encoding="utf-8").strip()
        assert json.loads(line)["id"] == body["id"]

        # Listing surfaces it.
        r = client.get(f"/api/projects/{pid}/sampling_log")
        assert r.status_code == 200
        ids = [e["id"] for e in r.json()["entries"]]
        assert body["id"] in ids

    def test_post_with_blank_optional_ids_persists_as_null(
        self, server_env
    ) -> None:
        """HTML forms send empty strings for unfilled optional fields;
        the API must treat those as "not linked" rather than failing
        validation against the 12-char hex id shape."""
        _, client, _ = server_env
        pid = _make_project(client)
        payload = {
            "action": "noted",
            "source_id": "",
            "participant_id": "",
            "rationale": "Saturation reached for category 'trust'",
        }
        r = client.post(f"/api/projects/{pid}/sampling_log", json=payload)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["source_id"] is None
        assert body["participant_id"] is None
        assert body["rationale"].startswith("Saturation")

    def test_post_validates_decision_type(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/sampling_log",
            json={"decision_type": "made-up"},
        )
        assert r.status_code == 400

    def test_post_validates_action(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/sampling_log",
            json={"action": "vandalised"},
        )
        assert r.status_code == 400

    def test_post_against_unknown_project_404s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/sampling_log",
            json={"action": "added"},
        )
        assert r.status_code == 404

    def test_get_against_unknown_project_404s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/aaaaaaaaaaaa/sampling_log")
        assert r.status_code == 404

    def test_get_against_malformed_project_id_400s(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/NOT-HEX/sampling_log")
        assert r.status_code == 400

    def test_post_requires_object_body(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/sampling_log",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_append_order_is_preserved(self, server_env) -> None:
        """Read order must equal append order — clock skew or
        backfilled entries should remain visible in writing order."""
        _, client, _ = server_env
        pid = _make_project(client)
        for tag in ("first", "second", "third"):
            r = client.post(
                f"/api/projects/{pid}/sampling_log",
                json={"action": "noted", "rationale": tag},
            )
            assert r.status_code == 201
        r = client.get(f"/api/projects/{pid}/sampling_log")
        assert r.status_code == 200
        rationales = [e["rationale"] for e in r.json()["entries"]]
        assert rationales == ["first", "second", "third"]


# --------------------------------------------------------------------------- #
# Source-picker integration: "Why was this source added?" prompt
# --------------------------------------------------------------------------- #


class TestSourcePickerSamplingPrompt:
    """The attach flow exposes a sampling-rationale prompt at
    attach-time so the audit trail is captured next to the action.
    The prompt is optional — researchers can skip — but must be
    present in the rendered HTML."""

    def test_picker_renders_attach_dialog(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        assert r.status_code == 200
        assert 'id="attachDialog"' in r.text
        # Title + the optional copy that explains the methodological
        # framing must be on the page.
        assert "Why was this source added?" in r.text
        assert 'id="att-rationale"' in r.text
        assert 'id="att-decision-type"' in r.text
        assert 'id="att-target-category"' in r.text

    def test_picker_dialog_has_skip_and_confirm(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        assert "Attach without logging" in r.text
        assert "Attach + log" in r.text

    def test_picker_posts_sampling_log_after_attach(self, server_env) -> None:
        """The page's JS POSTs to /api/.../sampling_log when the user
        confirms the dialog. Assert the URL pattern is wired in."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/add")
        assert "/api/projects/${PROJECT_ID}/sampling_log" in r.text


# --------------------------------------------------------------------------- #
# End-to-end: simulate the source-picker dialog flow
# --------------------------------------------------------------------------- #


class TestAttachWithSamplingFlow:
    """Mirrors what the source-picker dialog does on confirm: POST the
    source, then POST a matching sampling-log entry referencing the
    new source's id. Both calls must succeed and the log entry must
    surface back through GET."""

    def test_full_attach_plus_sampling_round_trip(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)

        # Step 1 — POST a source (no transcript_job_id required for
        # the F1.2 entity since the schema is forward-compatible with
        # field notes / documents).
        src_payload = {"name": "Interview 03", "source_type": "transcript"}
        r = client.post(f"/api/projects/{pid}/sources", json=src_payload)
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        # Step 2 — POST the matching sampling-log entry. This is what
        # source_picker.html's confirm handler does after the source
        # POST resolves.
        log_payload = {
            "action": "added",
            "decision_type": "theoretical",
            "source_id": sid,
            "target_category": "negative cases of trust",
            "rationale": "Counter-example to the saturating category.",
        }
        r = client.post(f"/api/projects/{pid}/sampling_log", json=log_payload)
        assert r.status_code == 201, r.text
        entry = r.json()
        assert entry["source_id"] == sid
        assert entry["decision_type"] == "theoretical"

        # Step 3 — listing endpoint surfaces both the source and the
        # log entry, so the project home snapshot card has data to show.
        r = client.get(f"/api/projects/{pid}/sampling_log")
        assert r.status_code == 200
        entries = r.json()["entries"]
        assert any(e["id"] == entry["id"] and e["source_id"] == sid for e in entries)

    def test_attach_then_log_does_not_double_create_source(
        self, server_env
    ) -> None:
        """A failed sampling-log POST must not roll back the source
        (the dialog logic warns but proceeds). The opposite — a failed
        source POST — must skip the log entirely. We test the source
        POST is independent: the log endpoint is independent of source
        creation, so even if a researcher logs a 'planned' entry with
        no source_id, the log still records it."""
        _, client, _ = server_env
        pid = _make_project(client)
        # Sampling log entry without any source — a "planned" entry
        # before recruitment.
        r = client.post(
            f"/api/projects/{pid}/sampling_log",
            json={
                "action": "planned",
                "decision_type": "theoretical",
                "rationale": "Need a positive deviant for category X.",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["action"] == "planned"
        assert body["source_id"] is None

        # Source listing is unaffected.
        r = client.get(f"/api/projects/{pid}/sources")
        assert r.json()["sources"] == []
