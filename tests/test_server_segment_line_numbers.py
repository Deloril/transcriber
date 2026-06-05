"""Tests for the per-segment line numbers in the two transcript views.

Each row in the editor (``/edit/<id>``) and the coding view
(``/projects/<pid>/sources/<sid>``) carries a 1-based line number
in a right-margin column so the user can say "look at line 42" in
a meeting and the listener can find it instantly.

Numbers are computed at render time from the segment's index in
the current state, so split / reorder / delete all renumber
naturally on the next render.
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
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), tmp_path


def _make_project(client: TestClient, name: str = "P1") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": "S", "source_type": "transcript"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_finished_job(env, *, job_id: str = "abc123def456") -> str:
    """Drop a status=done Job into srv.JOBS so /edit renders."""
    _, tmp_path = env
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = srv.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / "raw.wav"
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
        created_at="2026-06-05T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result={"segments": [], "speakers": [], "language": "en", "mode": "diarize"},
        input_filename="raw.wav",
        audio_streams=1,
    )
    srv.JOBS[job_id] = job
    return job_id


# --------------------------------------------------------------------------- #
# Editor (/edit/<id>) — line number column
# --------------------------------------------------------------------------- #


class TestEditorLineNumbers:
    def test_template_carries_lineno_class(self, env) -> None:
        client, _ = env
        job_id = _seed_finished_job(env)
        body = client.get(f"/edit/{job_id}").text
        # CSS rule + JS that builds the cell + the grid that
        # reserves the column.
        assert ".seg-lineno" in body
        assert 'lineno.className = "seg-lineno"' in body
        assert "lineno.textContent = String(idx + 1)" in body
        assert "200px 1fr 84px 36px" in body


# --------------------------------------------------------------------------- #
# Coding view (/projects/<pid>/sources/<sid>) — line number column
# --------------------------------------------------------------------------- #


class TestCodingViewLineNumbers:
    def test_template_carries_lineno_class(self, env) -> None:
        client, _ = env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        assert ".seg-lineno" in body
        assert 'lineno.className = "seg-lineno"' in body
        assert "lineno.textContent = String(segIdx + 1)" in body

    def test_grid_now_has_three_columns(self, env) -> None:
        client, _ = env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        body = client.get(f"/projects/{pid}/sources/{sid}").text
        # text + gutter + lineno. The gutter column uses a CSS variable.
        assert (
            "minmax(0, 1fr) var(--gutter-width, 0px) 36px"
            in body
        )
