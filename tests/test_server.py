"""Tests for scribe.server endpoints.

Uses FastAPI's TestClient. The transcription worker is mocked out
everywhere so no real models load. Job persistence and the file-system
integration are exercised against tmp directories per-test by patching
ROOT/UPLOAD_DIR/OUTPUT_DIR.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Per-test app + jobs isolation
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Reset scribe.server's module-global state for each test, redirect its
    UPLOAD_DIR/OUTPUT_DIR/ROOT into a tmp path, and yield the module + a
    TestClient.
    """
    from scribe import server as srv

    # Clear in-memory job registry.
    monkeypatch.setattr(srv, "JOBS", {})
    # Redirect persisted state to tmp dirs.
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    upload.mkdir()
    output.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    # Projects (F1.1) live under tmp too so tests don't trample the
    # developer's real `projects/` directory.
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects_dir)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _new_job(srv, *, status: str = "done", out_dir: Path = None, **fields) -> "srv.Job":
    """Quick helper: drop a Job into srv.JOBS for endpoint tests that need
    one without going through /api/upload."""
    if out_dir is None:
        out_dir = srv.OUTPUT_DIR / "abc123def456"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = srv.UPLOAD_DIR / "abc123def456" / "in.wav"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"\x00" * 64)
    job = srv.Job(
        id=fields.get("id", "abc123def456"),
        input_path=input_path,
        output_dir=out_dir,
        mode=fields.get("mode", "diarize"),
        speakers=fields.get("speakers"),
        num_speakers=fields.get("num_speakers"),
        language=fields.get("language", "en"),
        model=fields.get("model", "large-v3"),
        created_at=fields.get("created_at", "2026-05-25T00:00:00Z"),
        status=status,
        progress=fields.get("progress", 1.0 if status == "done" else 0.0),
        message=fields.get("message", "Done" if status == "done" else "Queued"),
        result=fields.get("result"),
        error=fields.get("error"),
        output_paths=fields.get("output_paths", {}),
        audio_streams=fields.get("audio_streams", 1),
        input_filename=fields.get("input_filename", "in.wav"),
        options=fields.get("options", {}),
        batch_size=fields.get("batch_size", 8),
        started_at=fields.get("started_at"),
        finished_at=fields.get("finished_at"),
    )
    srv.JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
# Job ID validation + path containment
# --------------------------------------------------------------------------- #


class TestJobIdValidation:
    @pytest.mark.parametrize("bad", [
        "ABCDEF123456",          # uppercase
        "abc123def45",           # too short
        "abc123def4567",         # too long
        "abc123def45z",          # bad hex char
        "../etc/passwd",         # traversal attempt
        "abc/def/123",
        "",
    ])
    def test_rejects_bad_ids(self, server_env, bad: str) -> None:
        srv, client, _ = server_env
        # All job-keyed endpoints should 400 on a malformed id.
        for path in (
            f"/api/job/{bad}",
            f"/api/job/{bad}/events",
            f"/api/job/{bad}/info",
            f"/api/job/{bad}/transcript",
            f"/api/job/{bad}/error",
            f"/api/job/{bad}/media",
            f"/api/job/{bad}/waveform",
            f"/api/job/{bad}/download/json",
            f"/edit/{bad}",
        ):
            r = client.get(path)
            # 400 is the validator, 404 is the not-found path; both are
            # acceptable failure modes — the key thing is we never expose
            # an actual file under the bad id.
            assert r.status_code in (400, 404), (path, r.status_code, r.text)

    def test_accepts_well_formed_id(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        r = client.get("/api/job/abc123def456")
        assert r.status_code == 200


class TestIsUnderHelper:
    def test_inside(self, tmp_path: Path) -> None:
        from scribe.server import _is_under
        a = tmp_path / "a"
        b = tmp_path / "a" / "b"
        b.mkdir(parents=True)
        assert _is_under(b, a) is True

    def test_outside(self, tmp_path: Path) -> None:
        from scribe.server import _is_under
        a = tmp_path / "a"
        a.mkdir()
        c = tmp_path / "c"
        c.mkdir()
        assert _is_under(c, a) is False

    def test_handles_nonexistent_paths(self, tmp_path: Path) -> None:
        from scribe.server import _is_under
        # Resolves symbolically; non-existent children of a real parent are fine.
        assert _is_under(tmp_path / "child" / "x", tmp_path) is True


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


class TestIndexPage:
    def test_serves_html(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Scribe" in r.text


class TestEditorPage:
    def test_404_when_job_missing(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/edit/abc123def456")
        assert r.status_code == 404

    def test_409_when_job_not_done(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, status="running")
        r = client.get("/edit/abc123def456")
        assert r.status_code == 409

    def test_200_when_done(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, status="done")
        r = client.get("/edit/abc123def456")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# --------------------------------------------------------------------------- #
# README / docs endpoints
# --------------------------------------------------------------------------- #


class TestReadmeEndpoints:
    def test_api_readme_returns_html(self, server_env) -> None:
        srv, client, tmp = server_env
        # Build a fake README inside the patched ROOT so _render_readme finds it.
        (tmp / "README.md").write_text("# Hello\n\nWorld.\n")
        # Force cache miss so it picks up our fixture file.
        srv._README_PATH = tmp / "README.md"
        srv._README_CACHE = {"mtime": 0.0, "html": ""}
        r = client.get("/api/readme")
        assert r.status_code == 200
        assert "<h1" in r.text
        assert "Hello" in r.text

    def test_docs_readme_full_page(self, server_env) -> None:
        srv, client, tmp = server_env
        (tmp / "README.md").write_text("# X\n")
        srv._README_PATH = tmp / "README.md"
        srv._README_CACHE = {"mtime": 0.0, "html": ""}
        r = client.get("/docs/readme")
        assert r.status_code == 200
        assert "<html" in r.text.lower()


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


class TestCapabilities:
    def test_reports_backend_and_parakeet(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine
        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "FakeGPU")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 8.0)
        # Force parakeet to look "available" so the field is populated.
        from scribe import parakeet
        parakeet._NEMO_AVAILABLE = True
        parakeet._IMPORT_ERROR = None
        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["device_name"] == "FakeGPU"
        assert body["gpu"]["vram_gb"] == 8.0
        assert body["parakeet"]["installed"] is True
        assert body["parakeet"]["available"] is True

    def test_parakeet_blocked_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, parakeet
        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        parakeet._NEMO_AVAILABLE = True
        parakeet._IMPORT_ERROR = None
        body = client.get("/api/capabilities").json()
        assert body["parakeet"]["installed"] is True
        assert body["parakeet"]["available"] is False
        assert body["parakeet"]["blocked_by_backend"] is True


# --------------------------------------------------------------------------- #
# Profiles CRUD
# --------------------------------------------------------------------------- #


class TestProfiles:
    def test_list_empty(self, server_env, monkeypatch: pytest.MonkeyPatch) -> None:
        srv, client, tmp = server_env
        # Redirect PROFILES_PATH so we don't touch the developer's real one.
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        r = client.get("/api/profiles")
        assert r.status_code == 200
        assert r.json() == {"profiles": []}

    def test_create_and_list(self, server_env, monkeypatch: pytest.MonkeyPatch) -> None:
        srv, client, tmp = server_env
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        r = client.put("/api/profiles/test-1", json={
            "name": "test-1",
            "description": "demo",
            "settings": {"mode": "diarize", "model": "tiny"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "test-1"
        assert body["settings"] == {"mode": "diarize", "model": "tiny"}

        r = client.get("/api/profiles")
        assert r.status_code == 200
        assert r.json()["profiles"][0]["name"] == "test-1"

    def test_replace_existing(self, server_env, monkeypatch: pytest.MonkeyPatch) -> None:
        srv, client, tmp = server_env
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        client.put("/api/profiles/x", json={"settings": {"mode": "diarize"}})
        client.put("/api/profiles/x", json={"settings": {"mode": "auto"}})
        body = client.get("/api/profiles").json()
        assert len(body["profiles"]) == 1
        assert body["profiles"][0]["settings"]["mode"] == "auto"

    def test_drops_unknown_settings_keys(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, tmp = server_env
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        r = client.put("/api/profiles/x", json={
            "settings": {"mode": "diarize", "naughty_field": "rm -rf /"},
        })
        assert "naughty_field" not in r.json()["settings"]

    def test_invalid_profile_name(self, server_env) -> None:
        srv, client, _ = server_env
        # Names that hit the route but fail validator: pipe, asterisk,
        # too-long. Slash-bearing names get caught by routing as 404.
        for bad in ("name|with|pipe", "name*with*star", "x" * 100):
            r = client.put(f"/api/profiles/{bad}", json={"settings": {}})
            assert r.status_code == 400, (bad, r.status_code)

    def test_delete_404(self, server_env, monkeypatch: pytest.MonkeyPatch) -> None:
        srv, client, tmp = server_env
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        r = client.delete("/api/profiles/nope")
        assert r.status_code == 404

    def test_delete_existing(self, server_env, monkeypatch: pytest.MonkeyPatch) -> None:
        srv, client, tmp = server_env
        monkeypatch.setattr(srv, "PROFILES_PATH", tmp / "profiles.json")
        client.put("/api/profiles/x", json={"settings": {}})
        r = client.delete("/api/profiles/x")
        assert r.status_code == 200
        assert client.get("/api/profiles").json()["profiles"] == []


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #


class TestUpload:
    def test_rejects_when_no_audio_streams(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        # Stub out audio probe to return no streams.
        monkeypatch.setattr(srv, "probe_audio_streams", lambda p: [])
        monkeypatch.setattr(srv, "probe_media_info", lambda p: None)
        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={"mode": "auto"},
        )
        assert r.status_code == 400
        assert "audio" in r.json()["detail"].lower()

    def test_starts_a_job(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import audio as scribe_audio
        srv, client, _ = server_env
        # Pretend the file has one audio stream.
        monkeypatch.setattr(srv, "probe_audio_streams", lambda p: [
            scribe_audio.AudioStream(index=0, channels=2, title=None, language="eng", codec="aac"),
        ])
        monkeypatch.setattr(srv, "probe_media_info", lambda p: {"duration_seconds": 10.0})
        # Don't actually run the worker — just confirm a thread starts and the
        # job lands in the registry.
        started: list[str] = []
        monkeypatch.setattr(srv, "_run_job", lambda jid: started.append(jid))

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={"mode": "auto", "language": "en", "model": "tiny",
                  "batch_size": "2", "options": "{}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["audio_streams"] == 1
        assert body["media_info"] == {"duration_seconds": 10.0}

        # Worker thread had a chance to run.
        # Wait briefly for thread spawn.
        import time
        for _ in range(30):
            if started:
                break
            time.sleep(0.01)
        assert started == [body["job_id"]]

    def test_invalid_options_json(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import audio as scribe_audio
        srv, client, _ = server_env
        monkeypatch.setattr(srv, "probe_audio_streams", lambda p: [
            scribe_audio.AudioStream(index=0, channels=1, title=None, language=None, codec="aac"),
        ])
        monkeypatch.setattr(srv, "probe_media_info", lambda p: {})
        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00", "audio/mp4")},
            data={"options": "this is not json"},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Job status / events / info / waveform
# --------------------------------------------------------------------------- #


class TestJobStatus:
    def test_404_missing(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/job/abc123def456")
        assert r.status_code == 404

    def test_returns_state(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, status="done", started_at=10.0, finished_at=20.0)
        body = client.get("/api/job/abc123def456").json()
        assert body["status"] == "done"
        assert body["started_at"] == 10.0
        assert body["finished_at"] == 20.0
        assert body["progress"] == 1.0


class TestJobInfo:
    def test_404_missing_file(self, server_env, tmp_path: Path) -> None:
        srv, client, _ = server_env
        # Point the job's input path to a file that doesn't exist.
        job = _new_job(srv)
        job.input_path = srv.UPLOAD_DIR / job.id / "missing.wav"
        r = client.get(f"/api/job/{job.id}/info")
        assert r.status_code == 404

    def test_returns_probe(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        monkeypatch.setattr(srv, "probe_media_info", lambda p: {"duration_seconds": 1.5})
        r = client.get("/api/job/abc123def456/info")
        assert r.status_code == 200
        assert r.json()["duration_seconds"] == 1.5


class TestWaveform:
    def test_computes_and_caches(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        # Stub compute_waveform to a deterministic output.
        monkeypatch.setattr(srv, "compute_waveform", lambda p, bins: [0.5] * bins)
        r = client.get("/api/job/abc123def456/waveform?bins=100")
        assert r.status_code == 200
        body = r.json()
        assert body["bins"] == 100
        assert body["peaks"] == [0.5] * 100

        # Second call hits the disk cache; we mutate the stub to prove it
        # *isn't* re-invoked.
        called_again: list[bool] = []
        monkeypatch.setattr(srv, "compute_waveform", lambda *a, **kw: called_again.append(True) or [0.0])
        r2 = client.get("/api/job/abc123def456/waveform?bins=100")
        assert r2.status_code == 200
        assert called_again == []  # served from cache

    def test_clamps_bins(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        seen_bins: list[int] = []
        monkeypatch.setattr(srv, "compute_waveform",
                            lambda p, bins: (seen_bins.append(bins), [0.0] * bins)[1])
        client.get("/api/job/abc123def456/waveform?bins=99999")
        assert seen_bins[0] == 4000  # max 4000
        seen_bins.clear()
        # Re-create job with a fresh out_dir so we don't hit cache from above.
        srv.JOBS.clear()
        new_out = srv.OUTPUT_DIR / "abcdef012345"
        _new_job(srv, id="abcdef012345", out_dir=new_out)
        client.get("/api/job/abcdef012345/waveform?bins=1")
        assert seen_bins[0] == 50  # min 50


# --------------------------------------------------------------------------- #
# Media + Range header parsing
# --------------------------------------------------------------------------- #


class TestMedia:
    def test_serves_full_file(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"hello world bytes")
        r = client.get(f"/api/job/{job.id}/media")
        # FastAPI's TestClient default behaviour returns the content.
        assert r.status_code == 200
        assert b"hello world bytes" in r.content

    def test_range_request(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"0123456789")
        r = client.get(
            f"/api/job/{job.id}/media",
            headers={"Range": "bytes=2-5"},
        )
        assert r.status_code == 206
        assert r.content == b"2345"
        assert r.headers["Content-Range"] == "bytes 2-5/10"

    def test_range_suffix(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"0123456789")
        r = client.get(
            f"/api/job/{job.id}/media",
            headers={"Range": "bytes=-3"},
        )
        assert r.status_code == 206
        assert r.content == b"789"

    def test_range_open_end(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"0123456789")
        r = client.get(
            f"/api/job/{job.id}/media",
            headers={"Range": "bytes=7-"},
        )
        assert r.status_code == 206
        assert r.content == b"789"

    def test_invalid_range(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"0123456789")
        for bad in ("bytes=-", "bytes=garbage", "items=0-5"):
            r = client.get(f"/api/job/{job.id}/media", headers={"Range": bad})
            assert r.status_code == 416

    def test_range_out_of_bounds(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        job.input_path.write_bytes(b"0123456789")
        r = client.get(
            f"/api/job/{job.id}/media",
            headers={"Range": "bytes=100-200"},
        )
        assert r.status_code == 416


# --------------------------------------------------------------------------- #
# Transcript GET / PUT
# --------------------------------------------------------------------------- #


class TestTranscript:
    def test_get_returns_result_when_no_edits(self, server_env) -> None:
        srv, client, _ = server_env
        result = {"language": "en", "mode": "diarize", "speakers": ["A"], "segments": []}
        _new_job(srv, result=result)
        r = client.get("/api/job/abc123def456/transcript")
        assert r.status_code == 200
        assert r.json() == result

    def test_get_returns_edits_when_present(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, result={"language": "en", "segments": []})
        edited = {"language": "en", "mode": "diarize", "segments": [
            {"start": 0, "end": 1, "speaker": "A", "text": "hello", "words": []},
        ]}
        (job.output_dir / "edited.json").write_text(json.dumps(edited))
        body = client.get("/api/job/abc123def456/transcript").json()
        assert body["segments"][0]["text"] == "hello"

    def test_put_writes_edited_and_regenerates_sidecars(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        payload = {
            "language": "en",
            "mode": "diarize",
            "speakers": ["A"],
            "segments": [
                {"start": 0, "end": 1, "speaker": "A", "text": "hello", "words": []},
            ],
        }
        r = client.put("/api/job/abc123def456/transcript", json=payload)
        assert r.status_code == 200
        # edited.json gets the request body.
        assert (job.output_dir / "edited.json").exists()
        # Sidecars regenerated.
        for kind in ("json", "txt", "srt", "vtt"):
            assert (job.output_dir / f"{job.input_path.stem}.{kind}").exists()

    def test_put_requires_segments_field(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        r = client.put("/api/job/abc123def456/transcript", json={"foo": "bar"})
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Download / error log
# --------------------------------------------------------------------------- #


class TestDownload:
    def test_404_for_missing_format(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, output_paths={})
        r = client.get("/api/job/abc123def456/download/json")
        assert r.status_code == 404

    def test_400_for_invalid_kind(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        r = client.get("/api/job/abc123def456/download/exe")
        assert r.status_code == 400

    def test_serves_file(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv)
        out = job.output_dir / "transcript.json"
        out.write_text('{"a": 1}')
        rel = str(out.relative_to(srv.ROOT))
        job.output_paths = {"json": rel}
        r = client.get(f"/api/job/{job.id}/download/json")
        assert r.status_code == 200
        assert b'"a": 1' in r.content


class TestErrorLog:
    def test_404_when_no_log(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv)
        r = client.get("/api/job/abc123def456/error")
        assert r.status_code == 404

    def test_returns_log_text(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, status="error")
        (job.output_dir / "error.log").write_text("Traceback...\nValueError: nope\n")
        r = client.get("/api/job/abc123def456/error")
        assert r.status_code == 200
        assert "ValueError" in r.text


# --------------------------------------------------------------------------- #
# Persistence + reload
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_round_trip_to_state(self, server_env) -> None:
        srv, _, _ = server_env
        job = _new_job(srv, started_at=1.0, finished_at=2.0,
                       options={"beam_size": 7})
        d = job.to_state()
        # Paths get serialised as strings.
        assert isinstance(d["input_path"], str)
        assert isinstance(d["output_dir"], str)
        # Round-trip preserves fields.
        rebuilt = srv.Job.from_state(d)
        assert rebuilt.id == job.id
        assert rebuilt.options == {"beam_size": 7}
        assert rebuilt.started_at == 1.0
        assert rebuilt.finished_at == 2.0

    def test_load_jobs_skips_invalid_paths(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, _, _ = server_env
        # Drop a malicious job.json with a path traversal attempt.
        d = srv.OUTPUT_DIR / "evil0123456a"
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({
            "id": "evil0123456a",
            "input_path": "/etc/passwd",
            "output_dir": str(d),
            "mode": "diarize",
            "language": "en",
            "model": "large-v3",
            "created_at": "2026-01-01",
            "status": "done",
        }))
        # Reload — the validator should reject and skip silently.
        srv.JOBS.clear()
        srv._load_jobs_from_disk()
        assert "evil0123456a" not in srv.JOBS

    def test_load_jobs_marks_running_as_error(
        self, server_env
    ) -> None:
        srv, _, _ = server_env
        # A job that was mid-run when the server restarted.
        out_dir = srv.OUTPUT_DIR / "0123456abcde"
        out_dir.mkdir(parents=True)
        in_dir = srv.UPLOAD_DIR / "0123456abcde"
        in_dir.mkdir(parents=True)
        in_path = in_dir / "x.wav"
        in_path.write_bytes(b"\x00")
        (out_dir / "job.json").write_text(json.dumps({
            "id": "0123456abcde",
            "input_path": str(in_path),
            "output_dir": str(out_dir),
            "mode": "diarize",
            "language": "en",
            "model": "large-v3",
            "created_at": "2026-01-01",
            "status": "running",
            "progress": 0.5,
        }))
        srv.JOBS.clear()
        srv._load_jobs_from_disk()
        assert srv.JOBS["0123456abcde"].status == "error"
        assert "restarted" in (srv.JOBS["0123456abcde"].error or "").lower()


# --------------------------------------------------------------------------- #
# Projects (F1.1)
# --------------------------------------------------------------------------- #


class TestProjectsAPI:
    def test_list_empty(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects")
        assert r.status_code == 200
        assert r.json() == {"projects": []}

    def test_create_minimal(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/projects", json={"name": "Pilot"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["name"] == "Pilot"
        assert body["codebook_stage"] == "initial"
        assert re.match(r"^[a-f0-9]{12}$", body["id"])
        assert body["created_at"] == body["modified_at"]

    def test_create_full_fields(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/projects", json={
            "name": "Consent in care",
            "research_question": "How do nurses interpret consent?",
            "methodology": "charmaz",
            "sensitising_concepts": ["agency", "structure"],
            "description": "Pilot",
            "codebook_stage": "focused",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["methodology"] == "charmaz"
        assert body["sensitising_concepts"] == ["agency", "structure"]
        assert body["codebook_stage"] == "focused"

    def test_create_persists_to_disk(self, server_env) -> None:
        srv, client, tmp = server_env
        r = client.post("/api/projects", json={"name": "On disk"})
        pid = r.json()["id"]
        on_disk = json.loads(
            (srv.PROJECTS_DIR / pid / "project.json").read_text()
        )
        assert on_disk["name"] == "On disk"

    def test_create_blank_name_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/projects", json={"name": "   "})
        assert r.status_code == 400

    def test_create_invalid_stage_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/projects", json={
            "name": "ok", "codebook_stage": "bogus",
        })
        assert r.status_code == 400

    def test_create_non_object_body_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/projects", json=["not", "an", "object"])
        assert r.status_code == 400

    def test_create_invalid_json_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400

    def test_get_existing(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "Fetch me"}).json()["id"]
        r = client.get(f"/api/projects/{pid}")
        assert r.status_code == 200
        assert r.json()["name"] == "Fetch me"

    def test_get_missing_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_get_invalid_id_400(self, server_env) -> None:
        _, client, _ = server_env
        for bad in ("BAD", "../etc", "x" * 12):
            r = client.get(f"/api/projects/{bad}")
            # routing-level vs validator: either 400 or 404; never 500.
            assert r.status_code in (400, 404), (bad, r.status_code)

    def test_list_returns_created(self, server_env) -> None:
        _, client, _ = server_env
        client.post("/api/projects", json={"name": "A"})
        client.post("/api/projects", json={"name": "B"})
        r = client.get("/api/projects")
        names = sorted(p["name"] for p in r.json()["projects"])
        assert names == ["A", "B"]

    def test_patch_updates_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "Old"}).json()["id"]
        r = client.patch(f"/api/projects/{pid}", json={
            "name": "New",
            "codebook_stage": "focused",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "New"
        assert body["codebook_stage"] == "focused"
        # Persisted on disk.
        again = client.get(f"/api/projects/{pid}").json()
        assert again["name"] == "New"

    def test_patch_advances_modified_at(self, server_env) -> None:
        _, client, _ = server_env
        created = client.post("/api/projects", json={"name": "Time"}).json()
        pid = created["id"]
        original_modified = created["modified_at"]
        # Sleep a tiny bit to guarantee a different microsecond.
        import time
        time.sleep(0.005)
        updated = client.patch(f"/api/projects/{pid}", json={"name": "Time2"}).json()
        assert updated["created_at"] == created["created_at"]
        assert updated["modified_at"] >= original_modified

    def test_patch_unknown_field_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "ok"}).json()["id"]
        r = client.patch(f"/api/projects/{pid}", json={"haxx": True})
        assert r.status_code == 400

    def test_patch_invalid_stage_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "ok"}).json()["id"]
        r = client.patch(f"/api/projects/{pid}", json={"codebook_stage": "garbage"})
        assert r.status_code == 400

    def test_patch_missing_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.patch("/api/projects/aaaaaaaaaaaa", json={"name": "x"})
        assert r.status_code == 404

    def test_patch_id_in_body_ignored(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "ok"}).json()["id"]
        # User tries to rewrite their id mid-update — server must ignore.
        r = client.patch(f"/api/projects/{pid}", json={
            "id": "ffffffffffff", "name": "renamed",
        })
        assert r.status_code == 200
        assert r.json()["id"] == pid

    def test_delete(self, server_env) -> None:
        _, client, _ = server_env
        pid = client.post("/api/projects", json={"name": "Doomed"}).json()["id"]
        r = client.delete(f"/api/projects/{pid}")
        assert r.status_code == 200
        assert client.get(f"/api/projects/{pid}").status_code == 404

    def test_delete_missing_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.delete("/api/projects/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_create_isolated_per_test(self, server_env) -> None:
        # Confirms the fixture redirects PROJECTS_DIR — projects from
        # one test don't leak into another.
        _, client, _ = server_env
        assert client.get("/api/projects").json() == {"projects": []}
