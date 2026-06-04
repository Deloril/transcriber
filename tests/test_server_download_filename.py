"""Tests for ``GET /api/job/{id}/download/{kind}`` — specifically the
filename the response advertises via ``Content-Disposition``.

The endpoint used to hand back ``<input_stem>.<kind>`` regardless of
the user's library rename, so a transcription saved as
``raw-audio-2026-05-26.wav`` and renamed to "Pilot interview" in the
library still downloaded as ``raw-audio-2026-05-26.txt``. This file
pins the post-fix behaviour: prefer the rename, fall back to the
original filename, sanitise unsafe characters, swap the extension.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv
from scribe.server import _download_filename


# --------------------------------------------------------------------------- #
# _download_filename — pure helper
# --------------------------------------------------------------------------- #


class TestDownloadFilenameHelper:
    def test_prefers_display_name(self) -> None:
        out = _download_filename(
            "Pilot interview", "raw-audio-2026-05-26.wav", "raw-audio.txt", "txt",
        )
        assert out == "Pilot interview.txt"

    def test_falls_back_to_input_filename(self) -> None:
        out = _download_filename(
            "", "raw-audio-2026-05-26.wav", "raw-audio.txt", "txt",
        )
        # Strips the upload's ``.wav`` and tags on the kind.
        assert out == "raw-audio-2026-05-26.txt"

    def test_falls_back_to_on_disk_name_last(self) -> None:
        out = _download_filename("", "", "fallback.txt", "srt")
        assert out == "fallback.srt"

    def test_strips_unsafe_chars(self) -> None:
        # Slashes, colons, pipes, and quotes are all banned in
        # filenames on Windows / macOS.
        out = _download_filename(
            "in/q1: Maria | interview <2>", "x.wav", "x.txt", "txt",
        )
        assert "/" not in out
        assert ":" not in out
        assert "|" not in out
        assert "<" not in out and ">" not in out
        assert out.endswith(".txt")

    def test_collapses_runs_of_spaces(self) -> None:
        out = _download_filename(
            "Pilot   interview", "x.wav", "x.txt", "txt",
        )
        assert out == "Pilot interview.txt"

    def test_control_bytes_become_underscores(self) -> None:
        # Tabs / newlines hit the unsafe-chars regex (they're in the
        # ASCII control range) before the whitespace-collapse pass.
        # Result: each one becomes ``_``, not folded into a single space.
        out = _download_filename(
            "Pilot\tinterview\nwith\rMaria", "x.wav", "x.txt", "txt",
        )
        assert out == "Pilot_interview_with_Maria.txt"

    def test_strips_control_bytes(self) -> None:
        out = _download_filename(
            "Pilot\x00interview", "x.wav", "x.txt", "txt",
        )
        assert "\x00" not in out

    def test_caps_long_stems(self) -> None:
        out = _download_filename("a" * 5000, "x.wav", "x.txt", "txt")
        # Stem capped, extension preserved.
        assert len(out) <= 250
        assert out.endswith(".txt")

    def test_swaps_existing_extension(self) -> None:
        # User typed something that looks like a filename already;
        # we still tag on the right kind, not whatever they wrote.
        out = _download_filename("Pilot.txt", "x.wav", "x.txt", "json")
        assert out == "Pilot.json"

    def test_empty_or_dots_only_falls_back(self) -> None:
        # Sanitised to nothing → "transcript" placeholder, not a
        # bare ".txt".
        for raw in ("", "   ", "...", "/// "):
            out = _download_filename(raw, "", "fallback.txt", "txt")
            assert out and not out.startswith(".")
            assert out.endswith(".txt")

    def test_kind_is_used_verbatim(self) -> None:
        # Endpoint already validates the kind; the helper just trusts it.
        for kind in ("json", "txt", "srt", "vtt"):
            out = _download_filename("Pilot", "x", "x", kind)
            assert out == f"Pilot.{kind}"


# --------------------------------------------------------------------------- #
# /api/job/{id}/download/{kind} — end-to-end
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


def _seed_done_job(env, *, display_name: str = "") -> str:
    """Drop a finished job + a real .txt sidecar so /download/txt
    finds something to serve."""
    _, tmp_path = env
    job_id = "abc123def456"
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = srv.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / "raw-audio-2026-05-26.wav"
    input_path.write_bytes(b"\x00" * 64)
    sidecar = out_dir / "raw-audio-2026-05-26.txt"
    sidecar.write_text("hello world\n")
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
        output_paths={
            "txt": str(sidecar.relative_to(srv.ROOT)),
        },
    )
    srv.JOBS[job_id] = job
    return job_id


def _disposition_filename(cd: str) -> str:
    """Pull the (possibly URL-encoded) filename out of a
    ``Content-Disposition`` header.

    Starlette uses RFC 5987's ``filename*=utf-8''<urlencoded>`` form
    when the filename has any characters outside the unencoded
    grammar (e.g. spaces). Decode either spelling so tests can
    compare against the raw string the user would see.
    """
    from urllib.parse import unquote
    parts = [p.strip() for p in cd.split(";")]
    for p in parts:
        if p.lower().startswith("filename*="):
            value = p.split("=", 1)[1]
            # filename*=utf-8''<value> — strip the charset prefix.
            if "''" in value:
                value = value.split("''", 1)[1]
            return unquote(value)
        if p.lower().startswith("filename="):
            value = p.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    raise AssertionError(f"No filename in Content-Disposition: {cd!r}")


class TestDownloadEndpointFilename:
    def test_uses_display_name_when_set(self, env) -> None:
        client, _ = env
        job_id = _seed_done_job(env, display_name="Pilot interview")
        r = client.get(f"/api/job/{job_id}/download/txt")
        assert r.status_code == 200, r.text
        assert _disposition_filename(r.headers["content-disposition"]) \
            == "Pilot interview.txt"

    def test_falls_back_to_input_filename(self, env) -> None:
        client, _ = env
        job_id = _seed_done_job(env, display_name="")
        r = client.get(f"/api/job/{job_id}/download/txt")
        # Original upload was ``raw-audio-2026-05-26.wav``; the .wav
        # is stripped and the kind is appended.
        name = _disposition_filename(r.headers["content-disposition"])
        assert name == "raw-audio-2026-05-26.txt"

    def test_unsafe_chars_in_rename_are_stripped(self, env) -> None:
        client, _ = env
        job_id = _seed_done_job(
            env, display_name="In/Q1: Maria | interview <2>",
        )
        r = client.get(f"/api/job/{job_id}/download/txt")
        name = _disposition_filename(r.headers["content-disposition"])
        for ch in ("/", ":", "|", "<", ">"):
            assert ch not in name, name

    def test_response_body_unchanged(self, env) -> None:
        """Filename is metadata; the bytes are still the sidecar."""
        client, _ = env
        job_id = _seed_done_job(env, display_name="Pilot")
        r = client.get(f"/api/job/{job_id}/download/txt")
        assert r.text == "hello world\n"
