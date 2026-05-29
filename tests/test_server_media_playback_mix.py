"""Tests for the playback-mix path through ``GET /api/job/<id>/media``.

A multi-track recording (OBS dual-mic, field-recorder N-track, etc.)
plays back in the browser's ``<video>`` element from the *first*
audio stream only — the other tracks are silently invisible. Scribe
detects multi-track jobs at /media time and serves a cached mix
that combines every selected audio stream into one. Single-track
jobs take the original-source fast path unchanged.

The pure mixing logic is covered by ``tests/test_audio.py``; this
file exercises the FastAPI wrapping: cache placement, selection
honouring, fallback on mix failure, single-track bypass.
"""

from __future__ import annotations

import shutil
import subprocess
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


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg"))


def _make_two_track_mka(out: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=0.5",
        "-map", "0:a", "-map", "1:a",
        "-c:a", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _make_single_track_wav(out: Path) -> Path:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
            "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
    )
    return out


def _seed_job(env, *, tracks_path: Path, audio_streams: int,
              selected_indices: list[int] | None = None) -> str:
    """Drop a finished Job into srv.JOBS pointing at ``tracks_path``."""
    _, tmp_path = env
    job_id = "abc123def456"
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = srv.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / tracks_path.name
    shutil.copy2(tracks_path, target)
    job = srv.Job(
        id=job_id,
        input_path=target,
        output_dir=out_dir,
        mode="multi-track",
        speakers=None,
        num_speakers=None,
        language="en",
        model="large-v3",
        created_at="2026-05-25T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result={"segments": [], "speakers": [], "language": "en", "mode": "multi-track"},
        input_filename=tracks_path.name,
        audio_streams=audio_streams,
        selected_stream_indices=selected_indices,
    )
    srv.JOBS[job_id] = job
    return job_id


# --------------------------------------------------------------------------- #


class TestMultiTrackServesMix:
    def test_endpoint_serves_a_mix_for_multi_track_jobs(
        self, env, tmp_path: Path,
    ) -> None:
        client, _ = env
        src = _make_two_track_mka(tmp_path / "two.mka")
        job_id = _seed_job(env, tracks_path=src, audio_streams=2)

        r = client.get(f"/api/job/{job_id}/media")
        assert r.status_code == 200, r.text
        # The mix lands under the job's output_dir as playback.<hash>.<ext>.
        out_dir = srv.OUTPUT_DIR / job_id
        mixes = list(out_dir.glob("playback.*"))
        assert mixes, "expected a cached playback.* file in the output dir"
        # Mix has exactly one audio stream.
        from scribe.audio import probe_audio_streams
        assert len(probe_audio_streams(mixes[0])) == 1

    def test_repeat_call_reuses_cache(
        self, env, tmp_path: Path,
    ) -> None:
        client, _ = env
        src = _make_two_track_mka(tmp_path / "two.mka")
        job_id = _seed_job(env, tracks_path=src, audio_streams=2)
        client.get(f"/api/job/{job_id}/media")
        out_dir = srv.OUTPUT_DIR / job_id
        first = next(out_dir.glob("playback.*"))
        m1 = first.stat().st_mtime
        # Force a perceptible delay; Linux mtime granularity is fine
        # but macOS HFS+ rounds to 1s.
        import time
        time.sleep(1.1)
        client.get(f"/api/job/{job_id}/media")
        assert first.stat().st_mtime == m1

    def test_selection_changes_cache_filename(
        self, env, tmp_path: Path,
    ) -> None:
        """Two different selections must hit different cache files
        so changing the picker doesn't serve stale audio."""
        client, _ = env
        src = _make_two_track_mka(tmp_path / "two.mka")
        from scribe.audio import probe_audio_streams
        streams = probe_audio_streams(src)

        # First job: all tracks (selection = None).
        job_id_a = _seed_job(env, tracks_path=src, audio_streams=2)
        client.get(f"/api/job/{job_id_a}/media")
        cache_a = list((srv.OUTPUT_DIR / job_id_a).glob("playback.*"))[0].name

        # Re-seed the same job id with a one-track selection. The
        # cache filename should differ — the hash is keyed on the
        # selection.
        del srv.JOBS[job_id_a]
        # Wipe the mix dir to start fresh.
        shutil.rmtree(srv.OUTPUT_DIR / job_id_a, ignore_errors=True)
        job_id_b = _seed_job(
            env, tracks_path=src, audio_streams=2,
            selected_indices=[streams[0].index],
        )
        client.get(f"/api/job/{job_id_b}/media")
        cache_b = list((srv.OUTPUT_DIR / job_id_b).glob("playback.*"))[0].name

        assert cache_a != cache_b


class TestSingleTrackBypass:
    def test_single_track_does_not_build_a_mix(
        self, env, tmp_path: Path,
    ) -> None:
        client, _ = env
        src = _make_single_track_wav(tmp_path / "one.wav")
        job_id = _seed_job(env, tracks_path=src, audio_streams=1)
        r = client.get(f"/api/job/{job_id}/media")
        assert r.status_code == 200
        # No playback.* cache file — single-track files take the
        # original-source fast path.
        out_dir = srv.OUTPUT_DIR / job_id
        assert not list(out_dir.glob("playback.*"))


class TestMixFailureFallsBack:
    def test_falls_back_to_source_on_mix_failure(
        self, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the mix can't be built (corrupt source, ffmpeg crash),
        the endpoint serves the source rather than 500ing the player.
        The fallback gives the user *something* — they hear the first
        track only — instead of a broken page."""
        client, _ = env
        src = _make_two_track_mka(tmp_path / "two.mka")
        job_id = _seed_job(env, tracks_path=src, audio_streams=2)

        def boom(*a, **kw):  # noqa: ANN001, ANN201
            raise RuntimeError("simulated ffmpeg failure")

        monkeypatch.setattr(srv, "build_playback_mix", boom)

        r = client.get(f"/api/job/{job_id}/media")
        # Still 200 — fallback served the source file.
        assert r.status_code == 200
