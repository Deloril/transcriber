"""Tests for the source-coding view's handling of editor-side
speaker renames + the new gutter "📝 New memo" floating button.

The editor stores user-typed speaker renames in the transcript
payload's top-level ``speaker_names`` map (saved into edited.json
by ``PUT /api/job/<id>/transcript``). The source coding view used
to render the *raw* ``segment.speaker`` label, so a transcription
labelled SPEAKER_00 → "Maria" in the editor would still show
"SPEAKER_00" in the coding view. This file pins the JS-level
``speakerDisplayName`` helper that consults the speaker_map first
(coding view's authoritative source) and the editor's rename map
second.

The lane-bar memo button is a floating control rendered next to
each coloured gutter bar in renderGutter(); we pin the test-id
marker on the rendered template so a future refactor doesn't drop
it silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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


def _make_project(client: TestClient, name: str = "Pilot") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    *,
    name: str = "Maria interview",
    job_id: str | None = None,
) -> str:
    body: dict = {"name": name, "source_type": "transcript", "language": "en"}
    if job_id is not None:
        body["transcript_job_id"] = job_id
    r = client.post(f"/api/projects/{pid}/sources", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestCodingViewExposesSpeakerRenameAndMemoButton:
    """The coding view template must surface:

    * The ``speakerDisplayName`` helper that resolves an editor-side
      rename or speaker_map override. Without it, a relabel made in
      the editor doesn't appear in the coding view.
    * The gutter ``lane-memo-btn`` floating button so a researcher
      can write a memo about a coded span without the right-click
      detour.
    """

    def test_coding_view_carries_speaker_display_helper(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # The helper that consults TRANSCRIPT.speaker_names + the
        # speaker map.
        assert "speakerDisplayName" in body
        # And the call site — make sure renderTranscript actually
        # uses it for the segment label rather than seg.speaker.
        assert "speakerDisplayName(seg.speaker)" in body

    def test_coding_view_renders_memo_button_marker(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # The CSS class + JS class name + per-button test id.
        assert "lane-memo-btn" in body
        assert 'data-test-id="gutter-lane-memo"' in body or \
            'data-test-id=\\"gutter-lane-memo\\"' in body or \
            '"gutter-lane-memo"' in body
        # Click handler routes to the memo composer with target_type=
        # application — pin the call shape.
        assert 'openMemoComposer("application"' in body
