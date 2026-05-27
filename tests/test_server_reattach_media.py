"""Tests for ``POST /api/job/{id}/reattach-media``.

When a transcription's source media has been discarded (or the upload
directory has gone missing for any other reason), the user can point
Scribe at an existing file on disk and we'll symlink it back into
``uploads/<id>/``. Playback works again, ``media_discarded`` flips
back to false, and the job's ``input_path`` follows the symlink.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    extern = tmp_path / "external"   # files outside UPLOAD_DIR live here
    upload.mkdir()
    output.mkdir()
    extern.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), tmp_path, extern


def _seed_discarded_job(env) -> str:
    """Job whose source media has been wiped, just like a real
    post-discard state."""
    _, tmp_path, _ = env
    job_id = "abc123def456"
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Note: NO upload directory — this is a discarded job. The path
    # the Job carries is the path the upload USED to live at; it
    # doesn't exist on disk.
    input_path = srv.UPLOAD_DIR / job_id / "original.mp4"
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
        input_filename="original.mp4",
        media_discarded=True,
        audio_streams=1,
    )
    srv.JOBS[job_id] = job
    return job_id


def _make_external_media(extern: Path, name: str = "reattached.mp4") -> Path:
    """Drop a fake media file outside UPLOAD_DIR. We don't probe it —
    the endpoint accepts on extension + existence — so empty bytes is
    fine for these tests."""
    p = extern / name
    p.write_bytes(b"\x00" * 64)
    return p


class TestReattachHappyPath:
    def test_creates_symlink_and_clears_discard_flag(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        target = _make_external_media(extern)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": str(target)},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["target"] == str(target.resolve())
        # Symlink created at uploads/<id>/<filename>.
        link = Path(body["input_path"])
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
        # Job state updated.
        job = srv.JOBS[job_id]
        assert job.media_discarded is False
        assert job.input_path == link
        assert job.input_filename == "reattached.mp4"

    def test_replaces_existing_symlink(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        first = _make_external_media(extern, "first.mp4")
        second = _make_external_media(extern, "first.mp4")  # same name
        # Reattach twice — second call should silently replace the link.
        client.post(f"/api/job/{job_id}/reattach-media", json={"path": str(first)})
        r = client.post(f"/api/job/{job_id}/reattach-media", json={"path": str(second)})
        assert r.status_code == 200, r.text
        link = Path(r.json()["input_path"])
        assert link.is_symlink()

    def test_media_endpoint_works_after_reattach(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        target = _make_external_media(extern)
        r = client.post(
            f"/api/job/{job_id}/reattach-media", json={"path": str(target)},
        )
        assert r.status_code == 200, r.text
        # The /media endpoint previously returned 410 (discarded); now
        # it should serve the symlink target.
        m = client.get(f"/api/job/{job_id}/media")
        assert m.status_code == 200, m.text


class TestReattachValidation:
    def test_rejects_relative_path(self, env) -> None:
        client, _, _ = env
        job_id = _seed_discarded_job(env)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": "./relative.mp4"},
        )
        assert r.status_code == 400

    def test_rejects_missing_file(self, env) -> None:
        client, _, _ = env
        job_id = _seed_discarded_job(env)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": "/no/such/file.mp4"},
        )
        assert r.status_code == 400

    def test_rejects_unsupported_extension(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        bad = extern / "notes.txt"
        bad.write_text("hello")
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": str(bad)},
        )
        assert r.status_code == 400
        assert "unsupported file type" in r.text.lower()

    def test_rejects_directory(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        d = extern / "a-dir"
        d.mkdir()
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": str(d)},
        )
        assert r.status_code == 400

    def test_rejects_path_already_in_upload_dir(self, env) -> None:
        client, tmp_path, _ = env
        job_id = _seed_discarded_job(env)
        # Pretend a file was somehow placed under UPLOAD_DIR; the
        # endpoint refuses these so the user uses the normal upload
        # flow instead of creating a self-referential link.
        in_uploads = srv.UPLOAD_DIR / "stray.mp4"
        in_uploads.write_bytes(b"\x00" * 8)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": str(in_uploads)},
        )
        assert r.status_code == 400
        assert "uploads directory" in r.text.lower()

    def test_404_for_unknown_job(self, env) -> None:
        client, _, extern = env
        target = _make_external_media(extern)
        r = client.post(
            "/api/job/aaaaaaaaaaaa/reattach-media",
            json={"path": str(target)},
        )
        assert r.status_code == 404

    def test_400_for_invalid_job_id(self, env) -> None:
        client, _, extern = env
        target = _make_external_media(extern)
        r = client.post(
            "/api/job/not-hex/reattach-media",
            json={"path": str(target)},
        )
        assert r.status_code == 400

    def test_400_on_missing_path_field(self, env) -> None:
        client, _, _ = env
        job_id = _seed_discarded_job(env)
        r = client.post(f"/api/job/{job_id}/reattach-media", json={})
        assert r.status_code == 400

    def test_400_on_non_string_path(self, env) -> None:
        client, _, _ = env
        job_id = _seed_discarded_job(env)
        for bad in (42, ["a"], None):
            r = client.post(
                f"/api/job/{job_id}/reattach-media", json={"path": bad},
            )
            assert r.status_code == 400, f"expected 400 for {bad!r}"

    def test_409_when_job_still_running(self, env) -> None:
        client, _, extern = env
        job_id = _seed_discarded_job(env)
        srv.JOBS[job_id].status = "running"
        target = _make_external_media(extern)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": str(target)},
        )
        assert r.status_code == 409


class TestReattachExpandsTilde:
    def test_tilde_expansion(
        self, env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client, _, extern = env
        # Pin HOME so ~ resolves under tmp.
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        media = fake_home / "video.mp4"
        media.write_bytes(b"\x00" * 8)
        job_id = _seed_discarded_job(env)
        r = client.post(
            f"/api/job/{job_id}/reattach-media",
            json={"path": "~/video.mp4"},
        )
        assert r.status_code == 200, r.text


class TestPersistedPathValidator:
    def test_validator_accepts_symlink_outside_upload_dir(
        self, env
    ) -> None:
        """A reattached job's input_path is a symlink under UPLOAD_DIR
        whose target is outside. _validate_persisted_paths must accept
        it on reload — otherwise the server refuses to deserialise the
        job after a restart."""
        _, _, extern = env
        target = _make_external_media(extern)
        link_dir = srv.UPLOAD_DIR / "abcdef012345"
        link_dir.mkdir()
        link = link_dir / target.name
        os.symlink(target, link)
        job = srv.Job(
            id="abcdef012345",
            input_path=link,
            output_dir=srv.OUTPUT_DIR / "abcdef012345",
            mode="diarize",
            speakers=None, num_speakers=None,
            language="en", model="large-v3",
            created_at="2026-05-25T00:00:00Z",
            status="done", progress=1.0, message="Done",
            input_filename=target.name,
        )
        # Should not raise.
        srv._validate_persisted_paths(job)
