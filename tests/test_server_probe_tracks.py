"""Tests for the multi-track preview / picker endpoints.

Three routes:

* POST   /api/probe-tracks                — stage + return per-track metadata
* GET    /api/probe-tracks/<token>/preview/<stream_index>   — 16kHz WAV sample
* DELETE /api/probe-tracks/<token>        — drop the staged upload

Plus the new fields on POST /api/upload (``staged_token``,
``selected_streams``, ``track_speaker_labels``) that connect the picker
back into the existing job pipeline.

We cook a small multi-track file with ffmpeg so the tests don't depend
on external fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_dual_track_mka(out: Path, *, seconds: float = 1.0) -> Path:
    """Build a Matroska file with two distinct audio tracks for picker tests.

    Track 0: 440 Hz sine (left "speaker"). Track 1: 880 Hz sine (right
    "speaker"). The metadata 'title' tag goes on the second track so we
    can assert that probe-tracks surfaces it.
    """
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}",
        "-map", "0:a", "-map", "1:a",
        "-metadata:s:a:1", "title=Guest mic",
        "-c:a", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


# --------------------------------------------------------------------------- #
# POST /api/probe-tracks
# --------------------------------------------------------------------------- #


class TestProbeTracks:
    def test_returns_per_track_metadata(self, env, tmp_path: Path) -> None:
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "two-tracks.mka")
        with media.open("rb") as fh:
            r = client.post(
                "/api/probe-tracks",
                files={"file": ("two-tracks.mka", fh, "audio/x-matroska")},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        # Token + per-track rows.
        assert isinstance(body["token"], str) and len(body["token"]) == 12
        assert body["audio_streams"] == 2
        assert len(body["tracks"]) == 2
        # First track has no title → fallback default label.
        assert body["tracks"][0]["title"] == ""
        assert body["tracks"][0]["default_label"] == "SPEAKER_01"
        # Second track's title comes through verbatim and seeds the
        # default label so the picker pre-fills the speaker name.
        assert body["tracks"][1]["title"] == "Guest mic"
        assert body["tracks"][1]["default_label"] == "Guest mic"
        # And the file is on disk under staged-<token>.
        staged_dir = srv.UPLOAD_DIR / f"staged-{body['token']}"
        assert staged_dir.is_dir()
        assert any(staged_dir.iterdir())

    def test_ordinal_is_position_not_absolute_index(
        self, env, tmp_path: Path,
    ) -> None:
        """ordinal is the 0-based position in the audio-streams list,
        which is what the UI's per-track row uses for display. The
        absolute ffprobe index goes in ``index`` (used by /preview)."""
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        assert [t["ordinal"] for t in body["tracks"]] == [0, 1]
        # Indices must be unique non-negative ints; we don't pin the
        # specific values because containers can renumber tracks.
        assert len({t["index"] for t in body["tracks"]}) == 2

    def test_rejects_file_with_no_audio(self, env, tmp_path: Path) -> None:
        client, _ = env
        bogus = tmp_path / "garbage.bin"
        bogus.write_bytes(b"\x00" * 64)
        with bogus.open("rb") as fh:
            r = client.post(
                "/api/probe-tracks",
                files={"file": ("garbage.bin", fh, "application/octet-stream")},
            )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GET preview
# --------------------------------------------------------------------------- #


class TestPreview:
    def test_returns_wav_bytes_for_a_valid_stream(
        self, env, tmp_path: Path,
    ) -> None:
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "two-tracks.mka", seconds=1.5)
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("two-tracks.mka", fh, "audio/x-matroska")},
            ).json()
        token = body["token"]
        idx = body["tracks"][0]["index"]
        r = client.get(
            f"/api/probe-tracks/{token}/preview/{idx}",
            params={"seconds": 1.0},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/wav"
        # Fixed-rate, mono PCM_S16LE: 16000 samples/sec × 2 bytes ≈ 32KB
        # per second. Allow a wide window for ffmpeg padding.
        assert 20_000 <= len(r.content) <= 80_000

    def test_404_for_unknown_stream(self, env, tmp_path: Path) -> None:
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        r = client.get(
            f"/api/probe-tracks/{body['token']}/preview/9",
        )
        assert r.status_code == 404

    def test_400_for_invalid_token(self, env) -> None:
        client, _ = env
        r = client.get("/api/probe-tracks/not-hex/preview/0")
        assert r.status_code == 400

    def test_404_for_unknown_token(self, env) -> None:
        client, _ = env
        r = client.get("/api/probe-tracks/aaaaaaaaaaaa/preview/0")
        assert r.status_code == 404

    def test_seconds_clamped_to_one_minute(
        self, env, tmp_path: Path,
    ) -> None:
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka", seconds=0.5)
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        idx = body["tracks"][0]["index"]
        # Asking for an hour — server clamps to 60 seconds. Source is
        # only 0.5s, so the resulting WAV is just 0.5s of audio. The
        # important check is "no error" — clamp prevents a hostile
        # client from triggering a giant decode.
        r = client.get(
            f"/api/probe-tracks/{body['token']}/preview/{idx}",
            params={"seconds": 3600},
        )
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# DELETE staged
# --------------------------------------------------------------------------- #


class TestDeleteStaged:
    def test_removes_staging_dir(self, env, tmp_path: Path) -> None:
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        token = body["token"]
        staged = srv.UPLOAD_DIR / f"staged-{token}"
        assert staged.is_dir()
        r = client.delete(f"/api/probe-tracks/{token}")
        assert r.status_code == 200
        assert not staged.exists()

    def test_idempotent_for_unknown_token(self, env) -> None:
        client, _ = env
        # Returns 200 even if the token doesn't exist — the picker
        # may fire a delete on a token that already expired.
        r = client.delete("/api/probe-tracks/aaaaaaaaaaaa")
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# /api/upload — new fields
# --------------------------------------------------------------------------- #


class TestUploadWithStagedToken:
    def test_promotes_staged_file_into_job_dir(
        self, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Don't actually run transcription in tests.
        monkeypatch.setattr(
            srv, "_run_job", lambda *_a, **_kw: None,
        )
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        token = body["token"]
        # Submit to /api/upload using the staged token + selecting only
        # the second track + naming both speakers per-track-index.
        idx0, idx1 = body["tracks"][0]["index"], body["tracks"][1]["index"]
        import json as _json
        r = client.post(
            "/api/upload",
            data={
                "mode": "multi-track",
                "language": "en",
                "model": "large-v3",
                "batch_size": "8",
                "options": "{}",
                "backend": "faster-whisper",
                "staged_token": token,
                "selected_streams": str(idx1),
                "track_speaker_labels": _json.dumps(
                    {str(idx0): "Luke", str(idx1): "Maria"},
                ),
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        job = srv.JOBS[job_id]
        # File moved into the real upload directory.
        assert job.input_path.parent == srv.UPLOAD_DIR / job_id
        assert job.input_path.is_file()
        # Selection survived.
        assert job.selected_stream_indices == [idx1]
        # speakers list is positional per stream (Luke for the first,
        # Maria for the second) — even though we excluded Luke, his
        # label round-trips so the engine can still wire it up if the
        # caller later changes their mind.
        assert job.speakers is not None
        assert "Luke" in job.speakers
        assert "Maria" in job.speakers
        # Staging dir was cleaned up.
        assert not (srv.UPLOAD_DIR / f"staged-{token}").exists()

    def test_400_when_neither_file_nor_token(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(srv, "_run_job", lambda *_a, **_kw: None)
        client, _ = env
        r = client.post(
            "/api/upload",
            data={"mode": "auto", "language": "en", "model": "large-v3"},
        )
        assert r.status_code == 400

    def test_400_on_unknown_selected_stream(
        self, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(srv, "_run_job", lambda *_a, **_kw: None)
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        r = client.post(
            "/api/upload",
            data={
                "mode": "multi-track",
                "staged_token": body["token"],
                "selected_streams": "99",
                "language": "en",
                "model": "large-v3",
                "backend": "faster-whisper",
            },
        )
        assert r.status_code == 400

    def test_400_on_malformed_track_labels(
        self, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(srv, "_run_job", lambda *_a, **_kw: None)
        client, _ = env
        media = _make_dual_track_mka(tmp_path / "x.mka")
        with media.open("rb") as fh:
            body = client.post(
                "/api/probe-tracks",
                files={"file": ("x.mka", fh, "audio/x-matroska")},
            ).json()
        r = client.post(
            "/api/upload",
            data={
                "mode": "multi-track",
                "staged_token": body["token"],
                "track_speaker_labels": "not json",
                "language": "en",
                "model": "large-v3",
                "backend": "faster-whisper",
            },
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GC sweep — staged uploads older than the TTL get cleaned up.
# --------------------------------------------------------------------------- #


class TestUploadPageRendersPicker:
    def test_picker_block_present(self, env) -> None:
        client, _ = env
        body = client.get("/").text
        # The picker UI is hidden by default; rendered + populated by
        # JS once the user picks a multi-track file. We just check the
        # block + JS hooks are in the page.
        assert 'id="trackPicker"' in body
        assert 'id="trackList"' in body
        assert "/api/probe-tracks" in body
        assert "track_speaker_labels" in body
        assert "selected_streams" in body


class TestStagedGc:
    def test_old_staged_uploads_are_removed_on_next_probe(
        self, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        # Make a fake staged dir + backdate it past the TTL.
        old = srv.UPLOAD_DIR / "staged-deadbeef0000"
        old.mkdir()
        (old / "x.bin").write_bytes(b"\x00")
        import os
        ancient = (srv.UPLOAD_DIR / "staged-deadbeef0000").stat().st_mtime - srv._STAGED_TTL_S - 60
        os.utime(old, (ancient, ancient))
        # Trigger GC by probing a fresh file.
        media = _make_dual_track_mka(tmp_path / "fresh.mka")
        with media.open("rb") as fh:
            r = client.post(
                "/api/probe-tracks",
                files={"file": ("fresh.mka", fh, "audio/x-matroska")},
            )
        assert r.status_code == 200
        # Old dir is gone; new one is there.
        assert not old.exists()
        new_token = r.json()["token"]
        assert (srv.UPLOAD_DIR / f"staged-{new_token}").is_dir()
