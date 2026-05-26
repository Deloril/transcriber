"""Tests for the codes + applications REST surface.

The data layer (scribe.codes / scribe.applications / scribe.code_versions)
was built per-feature by the loop and is well-covered by its own unit
tests. This file exercises only the HTTP wrapping: the routes are
reachable, validation rejects bad payloads, the lifecycle round-trips,
and the body shapes are what the UI actually consumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Standalone TestClient with storage redirected to tmp."""
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


# --------------------------------------------------------------------------- #
# Codes
# --------------------------------------------------------------------------- #


class TestCodesREST:
    def test_list_empty(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codes")
        assert r.status_code == 200
        assert r.json() == {"codes": []}

    def test_create_and_list(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codes",
            json={"name": "managing pain", "definition": "moments where the participant talks about coping"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["name"] == "managing pain"
        assert body["definition"].startswith("moments")
        assert "id" in body
        # Listing returns the created code.
        r = client.get(f"/api/projects/{pid}/codes")
        assert r.status_code == 200
        codes = r.json()["codes"]
        assert len(codes) == 1
        assert codes[0]["id"] == body["id"]

    def test_create_rejects_empty_name(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/codes", json={"name": ""})
        assert r.status_code == 400

    def test_create_rejects_bad_json(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/codes", content="not json")
        assert r.status_code == 400

    def test_get_one(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "A"})
        cid = r.json()["id"]
        r = client.get(f"/api/projects/{pid}/codes/{cid}")
        assert r.status_code == 200
        assert r.json()["name"] == "A"

    def test_get_404(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codes/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_delete(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "A"})
        cid = r.json()["id"]
        r = client.delete(f"/api/projects/{pid}/codes/{cid}")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        r = client.get(f"/api/projects/{pid}/codes")
        assert r.json()["codes"] == []

    def test_create_records_initial_version(self, env) -> None:
        client, projects_root = env
        pid = _new_project(client)
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "A"})
        cid = r.json()["id"]
        # On-disk: code_versions/<cid>.jsonl should now have one line.
        version_file = projects_root / pid / "code_versions" / f"{cid}.jsonl"
        assert version_file.exists()
        lines = [l for l in version_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["version"] == 1

    def test_create_persists_full_field_set(self, env) -> None:
        """F2.1: every Code-entity field round-trips through POST."""
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codes",
            json={
                "name": "managing pain",
                "definition": "moments of coping",
                "exemplars": ["I just sit with it.", "Take a breath."],
                "theoretical_memo": "links to Charmaz §3.2",
                "stage": "focused",
                "colour": "#a78bfa",
                "status": "active",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["stage"] == "focused"
        assert body["colour"] == "#a78bfa"
        assert body["exemplars"] == ["I just sit with it.", "Take a breath."]
        assert body["theoretical_memo"] == "links to Charmaz §3.2"


# --------------------------------------------------------------------------- #
# F2.1: Code edit (PATCH) — full-field-set round trip + lock + versioning
# --------------------------------------------------------------------------- #


class TestCodePatch:
    """The PATCH endpoint exposes F2.1's full Code-entity field set
    (exemplars / parent / related / theoretical memo / stage / colour /
    status / provenance) and records a new F2.2 version when the
    definition actually changes. F2.4's lock blocks edits with 409.
    """

    def _make_code(self, client: TestClient, pid: str, **extra) -> str:
        r = client.post(
            f"/api/projects/{pid}/codes",
            json={"name": "managing", "definition": "v1", **extra},
        )
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def test_patch_round_trips_simple_fields(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"name": "managing pain", "definition": "v2"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "managing pain"
        assert body["definition"] == "v2"

    def test_patch_persists_advanced_fields(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={
                "exemplars": ["I just sit with it.", "Take a breath."],
                "theoretical_memo": "ties to constructivism",
                "stage": "focused",
                "colour": "#a78bfa",
                "status": "draft",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["exemplars"] == [
            "I just sit with it.", "Take a breath.",
        ]
        assert body["theoretical_memo"] == "ties to constructivism"
        assert body["stage"] == "focused"
        assert body["colour"] == "#a78bfa"
        assert body["status"] == "draft"

    def test_patch_records_new_version_on_definition_change(self, env) -> None:
        client, projects_root = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        version_file = projects_root / pid / "code_versions" / f"{cid}.jsonl"
        assert version_file.exists()
        lines_before = [
            l for l in version_file.read_text().splitlines() if l.strip()
        ]
        assert len(lines_before) == 1
        # Definition change → new version recorded.
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"definition": "v2 — more nuance",
                  "change_note": "broadened scope"},
        )
        assert r.status_code == 200, r.text
        lines_after = [
            l for l in version_file.read_text().splitlines() if l.strip()
        ]
        assert len(lines_after) == 2
        rec = json.loads(lines_after[-1])
        assert rec["version"] == 2
        assert rec["change_note"] == "broadened scope"

    def test_patch_skips_version_when_nothing_changed(self, env) -> None:
        client, projects_root = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        version_file = projects_root / pid / "code_versions" / f"{cid}.jsonl"
        before = len([
            l for l in version_file.read_text().splitlines() if l.strip()
        ])
        # Editing only the colour does not change the F2.2 DEFINITION_FIELDS,
        # so no new version line should appear.
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"colour": "#aabbcc"},
        )
        assert r.status_code == 200, r.text
        after = len([
            l for l in version_file.read_text().splitlines() if l.strip()
        ])
        assert after == before

    def test_patch_404_on_missing_code(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.patch(
            f"/api/projects/{pid}/codes/aaaaaaaaaaaa",
            json={"name": "x"},
        )
        assert r.status_code == 404

    def test_patch_400_on_bad_json(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_patch_400_on_invalid_stage(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"stage": "not-a-real-stage"},
        )
        assert r.status_code == 400

    def test_patch_400_on_invalid_colour(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"colour": "rebeccapurple"},  # named colours rejected
        )
        assert r.status_code == 400

    def test_patch_409_when_codebook_locked(self, env) -> None:
        from scribe import codebook_lock as _lock
        client, projects_root = env
        pid = _new_project(client)
        cid = self._make_code(client, pid)
        # Lock the codebook before attempting an edit.
        _lock.lock_codebook(
            projects_root,
            pid,
            reason="freezing for ICR",
        )
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={"definition": "v2"},
        )
        assert r.status_code == 409


# --------------------------------------------------------------------------- #
# F2.1: Codebook editor template — every field surfaced
# --------------------------------------------------------------------------- #


class TestCodebookEditorTemplate:
    """The codebook editor must expose every Code-entity field — anything
    less and the data layer is unreachable from the user surface (W2.1).
    """

    def test_page_renders_with_full_field_set(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200
        text = r.text
        # Form fields for every editable Code attribute.
        for marker in (
            'id="cb-name"',
            'id="cb-def"',
            'id="cb-incl"',
            'id="cb-excl"',
            'id="cb-exemplars"',
            'id="cb-parent"',
            'id="cb-related"',
            'id="cb-theo"',
            'id="cb-stage"',
            'id="cb-colour"',
            'id="cb-status"',
        ):
            assert marker in text, f"missing {marker} in codebook editor"
        # The save button has both create + edit identities so the
        # JS can flip between modes.
        assert 'id="cb-submit"' in text
        # Stage vocabulary mirrored from CODEBOOK_STAGES.
        for stage in ("initial", "focused", "axial", "theoretical"):
            assert f'value="{stage}"' in text
        # Status vocabulary.
        for status in ("active", "draft", "retired"):
            assert f'value="{status}"' in text


class TestApplicationsREST:
    def _setup(self, client: TestClient) -> tuple[str, str, str]:
        """Create a project + code + source. Returns (pid, cid, sid)."""
        pid = _new_project(client)
        # Code
        r = client.post(f"/api/projects/{pid}/codes", json={"name": "test-code"})
        cid = r.json()["id"]
        # Source — wraps a hypothetical transcript job.
        r = client.post(
            f"/api/projects/{pid}/sources",
            json={"name": "Interview 1", "source_type": "transcript", "language": "en"},
        )
        assert r.status_code == 201, r.text
        sid = r.json()["id"]
        return pid, cid, sid

    def test_list_empty(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/applications")
        assert r.status_code == 200
        assert r.json() == {"applications": []}

    def test_create_application(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={
                "code_id": cid,
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["code_id"] == cid
        assert body["source_id"] == sid
        assert body["anchor_start_word_id"] == "s0w0"
        assert body["anchor_end_word_id"] == "s0w5"
        # A coder is auto-created on first application.
        assert body["coder_id"]
        # Definition version is recorded.
        assert body["definition_version_id_at_apply"]

    def test_list_filters_by_source(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        # Two applications on the same source
        for w_end in ("s0w5", "s0w10"):
            client.post(
                f"/api/projects/{pid}/applications",
                json={"code_id": cid, "source_id": sid,
                      "anchor_start_word_id": "s0w0", "anchor_end_word_id": w_end},
            )
        # ... and one on a different source
        r = client.post(f"/api/projects/{pid}/sources",
                        json={"name": "Other", "source_type": "transcript"})
        other_sid = r.json()["id"]
        client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": cid, "source_id": other_sid,
                  "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w3"},
        )
        # Filter by source
        r = client.get(f"/api/projects/{pid}/applications?source_id={sid}")
        assert r.status_code == 200
        apps = r.json()["applications"]
        assert len(apps) == 2
        assert all(a["source_id"] == sid for a in apps)

    def test_create_requires_code_id(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        r = client.post(f"/api/projects/{pid}/applications",
                        json={"source_id": sid,
                              "anchor_start_word_id": "s0w0",
                              "anchor_end_word_id": "s0w1"})
        assert r.status_code == 400
        assert "code_id" in r.json()["detail"].lower()

    def test_create_requires_source_id(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        r = client.post(f"/api/projects/{pid}/applications",
                        json={"code_id": cid,
                              "anchor_start_word_id": "s0w0",
                              "anchor_end_word_id": "s0w1"})
        assert r.status_code == 400
        assert "source_id" in r.json()["detail"].lower()

    def test_create_404_on_missing_code(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        r = client.post(f"/api/projects/{pid}/applications",
                        json={"code_id": "aaaaaaaaaaaa", "source_id": sid,
                              "anchor_start_word_id": "s0w0",
                              "anchor_end_word_id": "s0w1"})
        assert r.status_code == 404

    def test_delete(self, env) -> None:
        client, _ = env
        pid, cid, sid = self._setup(env[0])
        r = client.post(f"/api/projects/{pid}/applications",
                        json={"code_id": cid, "source_id": sid,
                              "anchor_start_word_id": "s0w0",
                              "anchor_end_word_id": "s0w1"})
        aid = r.json()["id"]
        r = client.delete(f"/api/projects/{pid}/applications/{aid}")
        assert r.status_code == 200
        r = client.get(f"/api/projects/{pid}/applications")
        assert r.json()["applications"] == []
