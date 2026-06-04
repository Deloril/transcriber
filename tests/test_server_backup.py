"""Tests for the backup / restore HTTP surface.

The pure module (:mod:`scribe.backup`) covers archive shape, restore
semantics, and the safety guards. This file checks that the FastAPI
wrapping plumbs through to those helpers and surfaces errors with
the right HTTP status.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import backup as bk
from scribe import server as srv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    profiles = tmp_path / "profiles.json"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "PROFILES_PATH", profiles)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), tmp_path


def _seed_install(env) -> None:
    """Drop a tiny but realistic install into the env fixture's
    monkeypatched directories."""
    _, tmp_path = env
    out = srv.OUTPUT_DIR / "abc123def456"
    out.mkdir(parents=True)
    (out / "job.json").write_text('{"id": "abc123def456"}')
    (out / "raw-audio.txt").write_text("hello\n")
    (srv.PROJECTS_DIR / "p1").mkdir(parents=True)
    (srv.PROJECTS_DIR / "p1" / "project.json").write_text('{"id": "p1"}')
    up = srv.UPLOAD_DIR / "abc123def456"
    up.mkdir(parents=True)
    (up / "raw-audio.wav").write_bytes(b"WAV\x00" * 8)
    srv.PROFILES_PATH.write_text("[]")


# --------------------------------------------------------------------------- #
# GET /api/backup
# --------------------------------------------------------------------------- #


class TestBackupDownload:
    def test_streams_a_zip_with_manifest(self, env) -> None:
        client, _ = env
        _seed_install(env)
        r = client.get("/api/backup")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/zip"
        assert "scribe-backup-" in r.headers["content-disposition"]
        # The bytes are a real zip with a manifest entry.
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert bk.ARCHIVE_MANIFEST_FILE in zf.namelist()
            manifest = json.loads(zf.read(bk.ARCHIVE_MANIFEST_FILE))
            assert manifest["format"] == "scribe-backup"
            assert manifest["summary"]["outputs_files"] >= 1

    def test_skip_uploads_via_query(self, env) -> None:
        client, _ = env
        _seed_install(env)
        r = client.get("/api/backup", params={"include_uploads": "false"})
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            assert not any(n.startswith("uploads/") for n in names)

    def test_works_on_fresh_install(self, env) -> None:
        # No seeded data; the endpoint still returns a (mostly empty)
        # zip with the manifest, not a 500.
        client, _ = env
        r = client.get("/api/backup")
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert bk.ARCHIVE_MANIFEST_FILE in zf.namelist()


# --------------------------------------------------------------------------- #
# POST /api/backup/inspect
# --------------------------------------------------------------------------- #


class TestBackupInspect:
    def test_returns_manifest_summary(self, env) -> None:
        client, _ = env
        _seed_install(env)
        # Build a backup, then post it back through inspect.
        zip_resp = client.get("/api/backup")
        r = client.post(
            "/api/backup/inspect",
            files={"file": ("backup.zip", zip_resp.content, "application/zip")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["format"] == "scribe-backup"
        assert body["summary"]["outputs_files"] >= 1

    def test_400_on_garbage_upload(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/backup/inspect",
            files={"file": ("not-a-zip.zip", b"hello world", "application/zip")},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# POST /api/restore
# --------------------------------------------------------------------------- #


class TestRestore:
    def test_round_trip(self, env) -> None:
        client, tmp_path = env
        _seed_install(env)
        zip_resp = client.get("/api/backup")
        # Wipe everything we care about so restore re-creates it.
        import shutil
        shutil.rmtree(srv.OUTPUT_DIR / "abc123def456")
        shutil.rmtree(srv.PROJECTS_DIR / "p1")
        srv.PROFILES_PATH.unlink()
        # Restore — needs force=true because an empty outputs dir
        # still exists from the fixture.
        r = client.post(
            "/api/restore",
            files={"file": ("backup.zip", zip_resp.content, "application/zip")},
            data={"force": "true"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["outputs_files"] >= 1
        assert body["projects_files"] >= 1
        # Files are back where we left them.
        assert (srv.OUTPUT_DIR / "abc123def456" / "raw-audio.txt") \
            .read_text() == "hello\n"
        assert srv.PROFILES_PATH.read_text() == "[]"

    def test_400_when_target_non_empty_without_force(self, env) -> None:
        client, _ = env
        _seed_install(env)
        zip_resp = client.get("/api/backup")
        # Default force=false; the seeded install leaves outputs/
        # populated, so the restore must refuse.
        r = client.post(
            "/api/restore",
            files={"file": ("backup.zip", zip_resp.content, "application/zip")},
        )
        assert r.status_code == 400

    def test_skip_uploads_at_restore_time(self, env) -> None:
        client, _ = env
        _seed_install(env)
        zip_resp = client.get("/api/backup")
        # Wipe uploads explicitly; restore with include_uploads=false
        # must leave them empty even though the backup carries them.
        import shutil
        shutil.rmtree(srv.UPLOAD_DIR / "abc123def456")
        r = client.post(
            "/api/restore",
            files={"file": ("backup.zip", zip_resp.content, "application/zip")},
            data={"force": "true", "include_uploads": "false"},
        )
        assert r.status_code == 200
        # Uploads dir is empty — uploads/<id>/ wasn't recreated.
        contents = list(srv.UPLOAD_DIR.iterdir())
        assert contents == []

    def test_400_on_non_zip_upload(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/restore",
            files={"file": ("foo.zip", b"not a zip", "application/zip")},
            data={"force": "true"},
        )
        assert r.status_code == 400


class TestSettingsPageRendersBackupCard:
    def test_card_present(self, env) -> None:
        client, _ = env
        body = client.get("/settings").text
        # The card and its primary controls are in the served HTML.
        assert 'data-test-id="settings-backup-card"' in body
        assert 'id="backupDownloadBtn"' in body
        assert 'id="restoreFile"' in body
        assert 'id="restoreBtn"' in body
        # JS hooks the page exercises.
        assert "/api/backup" in body
        assert "/api/restore" in body
        assert "/api/backup/inspect" in body
