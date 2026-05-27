"""Tests for ``PATCH /api/job/{id}`` — the rename endpoint that lets
the user override the upload filename with a friendlier ``display_name``.

The original ``input_filename`` is the immutable record of what was
uploaded; ``display_name`` is the editable label that the library row,
the editor topbar, and the library search index all consult instead.
Clearing it (empty string / null) falls back to ``input_filename``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    upload.mkdir()
    output.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), tmp_path


def _seed_job(env, *, display_name: str = "") -> str:
    """Drop a finished Job into srv.JOBS for the rename endpoint to mutate."""
    _, tmp_path = env
    job_id = "abc123def456"
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = srv.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / "raw-audio-2026-05-26.wav"
    input_path.write_bytes(b"\x00" * 64)
    job = srv.Job(
        id=job_id,
        input_path=input_path,
        output_dir=out_dir,
        mode="diarize",
        speakers=None,
        num_speakers=None,
        language="en",
        model="large-v3",
        created_at="2026-05-25T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result={"segments": [], "speakers": [], "language": "en", "mode": "diarize"},
        input_filename="raw-audio-2026-05-26.wav",
        display_name=display_name,
        audio_streams=1,
    )
    srv.JOBS[job_id] = job
    return job_id


class TestPatchJob:
    def test_rename_round_trips(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.patch(
            f"/api/job/{job_id}",
            json={"display_name": "Maria — interview 2"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["display_name"] == "Maria — interview 2"
        # And it's persisted to the in-memory job too.
        assert srv.JOBS[job_id].display_name == "Maria — interview 2"

    def test_get_after_rename_includes_display_name(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env, display_name="hello")
        body = client.get(f"/api/job/{job_id}").json()
        assert body["display_name"] == "hello"
        # Original input_filename is preserved verbatim.
        assert body["input_filename"] == "raw-audio-2026-05-26.wav"

    def test_empty_string_clears_rename(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env, display_name="prior")
        r = client.patch(f"/api/job/{job_id}", json={"display_name": ""})
        assert r.status_code == 200
        assert r.json()["display_name"] == ""
        assert srv.JOBS[job_id].display_name == ""

    def test_null_clears_rename(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env, display_name="prior")
        r = client.patch(f"/api/job/{job_id}", json={"display_name": None})
        assert r.status_code == 200
        assert r.json()["display_name"] == ""

    def test_whitespace_is_trimmed(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.patch(
            f"/api/job/{job_id}",
            json={"display_name": "   spaced out   "},
        )
        assert r.status_code == 200
        assert r.json()["display_name"] == "spaced out"

    def test_rejects_overlong_name(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.patch(
            f"/api/job/{job_id}",
            json={"display_name": "x" * (srv._MAX_DISPLAY_NAME_LEN + 1)},
        )
        assert r.status_code == 400

    def test_rejects_non_string(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        for bad in (42, ["a", "b"], {"x": 1}, True):
            r = client.patch(
                f"/api/job/{job_id}", json={"display_name": bad},
            )
            assert r.status_code == 400, f"expected 400 for {bad!r}"

    def test_400_when_no_field_supplied(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.patch(f"/api/job/{job_id}", json={})
        assert r.status_code == 400

    def test_404_for_unknown_job(self, env) -> None:
        client, _ = env
        r = client.patch("/api/job/aaaaaaaaaaaa", json={"display_name": "x"})
        assert r.status_code == 404

    def test_400_for_invalid_job_id(self, env) -> None:
        client, _ = env
        r = client.patch("/api/job/not-hex", json={"display_name": "x"})
        assert r.status_code == 400

    def test_400_on_invalid_json(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.patch(
            f"/api/job/{job_id}",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400


class TestLibraryRowRendersDisplayName:
    def test_jobs_endpoint_carries_display_name(self, env) -> None:
        client, _ = env
        _seed_job(env, display_name="My nice label")
        body = client.get("/api/jobs").json()
        rows = body["jobs"]
        assert len(rows) == 1
        assert rows[0]["display_name"] == "My nice label"
        assert rows[0]["input_filename"] == "raw-audio-2026-05-26.wav"

    def test_jobs_endpoint_empty_display_name_when_unset(self, env) -> None:
        client, _ = env
        _seed_job(env)
        rows = client.get("/api/jobs").json()["jobs"]
        assert rows[0]["display_name"] == ""


class TestEditorTopbarUsesDisplayName:
    def test_editor_renders_display_name_when_set(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env, display_name="Maria interview 2")
        body = client.get(f"/edit/{job_id}").text
        assert "Maria interview 2" in body
        # The topbar wraps in <strong>...</strong>, so checking the
        # raw filename is *not* in the markup as the visible label.
        # (We can't make a stronger assertion without parsing HTML,
        # but presence of the new label is enough to prove wiring.)

    def test_editor_falls_back_to_input_filename(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        body = client.get(f"/edit/{job_id}").text
        assert "raw-audio-2026-05-26.wav" in body


class TestLibraryUIRenamePencil:
    def test_library_template_renders_rename_button(self, env) -> None:
        client, _ = env
        body = client.get("/library").text
        # The button is rendered per-row in JS, but the static template
        # carries the data-action='rename' string + the PATCH handler.
        assert "data-action='rename'" in body or 'data-action="rename"' in body
        assert "PATCH" in body
