"""End-to-end reachability tests for F4.6 (One-click playback from any
coded segment).

Background
----------

F4.6 shipped ``scribe.application_playback`` in d476da3 — a pure
helper that takes an :class:`scribe.applications.Application` (with
its ``s<seg>w<word>`` anchor ids) plus the transcript's segments and
returns the wall-clock ``[start, end]`` seconds the editor should
seek through to play just that coded segment back. 34 unit tests
exercise the algorithm in ``tests/test_application_playback.py``.

What was missing — and what this file covers — is the integration
proof that the **user-facing surface** is wired together. Per the
loop's done-criteria, F4.6 is only "done" if a researcher can reach
the data layer through a real route + a real UI control. That means:

1. ``GET /api/projects/<pid>/applications/<aid>/playback`` must
   return the F4.6 ``{application_id, source_id, transcript_job_id,
   start, end}`` envelope by walking the application -> source ->
   transcript chain and calling :func:`playback_range_for_application`.
2. The route must surface 404 + a structured ``reason`` for every
   failure mode (no source, no transcript, untimed words, anchor out
   of range) so the UI can disable the play button rather than
   seeking blindly to zero.
3. The coding view (``GET /projects/<pid>/sources/<sid>``) must
   render the per-row ▶ Play button, the inline ``#playDock``
   ``<audio>`` element, and the JS ``playApplication`` function that
   wires them together.

Without this file the F4.6 ID would be in the commit log but with
no proof a researcher can reach the play button — exactly the
failure mode the loop's done-detector is designed to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures — mirror tests/test_server_application_reanchor.py.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Builders — most tests want a project + code + source + a transcript-
# backed job whose JSON sidecar carries word-level timing so the F4.6
# helper can resolve a non-trivial range.
# --------------------------------------------------------------------------- #


JOB_ID = "abc123def456"


def _make_project(client: TestClient, name: str = "F4.6 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str = "managing pain") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "moments where the participant copes"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    name: str = "Interview 1",
    *,
    job_id: str | None = JOB_ID,
) -> str:
    body: dict = {
        "name": name,
        "source_type": "transcript",
        "language": "en",
    }
    if job_id is not None:
        body["transcript_job_id"] = job_id
    r = client.post(f"/api/projects/{pid}/sources", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _apply(
    client: TestClient,
    pid: str,
    cid: str,
    sid: str,
    start: str,
    end: str,
    *,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> str:
    body: dict = {
        "code_id": cid,
        "source_id": sid,
        "anchor_start_word_id": start,
        "anchor_end_word_id": end,
    }
    if start_offset is not None:
        body["start_char_offset"] = start_offset
    if end_offset is not None:
        body["end_char_offset"] = end_offset
    r = client.post(
        f"/api/projects/{pid}/applications",
        json=body,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _segments_basic() -> list[dict]:
    """Two segments with word-level timing the F4.6 helper can use."""
    return [
        {
            "speaker": "A",
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world how are you",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "A"},
                {"text": "world", "start": 0.5, "end": 1.0, "speaker": "A"},
                {"text": "how", "start": 1.2, "end": 1.4, "speaker": "A"},
                {"text": "are", "start": 1.4, "end": 1.6, "speaker": "A"},
                {"text": "you", "start": 1.6, "end": 2.0, "speaker": "A"},
            ],
        },
        {
            "speaker": "B",
            "start": 3.0,
            "end": 4.0,
            "text": "I am fine thanks",
            "words": [
                {"text": "I", "start": 3.0, "end": 3.1, "speaker": "B"},
                {"text": "am", "start": 3.1, "end": 3.3, "speaker": "B"},
                {"text": "fine", "start": 3.3, "end": 3.6, "speaker": "B"},
                {"text": "thanks", "start": 3.6, "end": 4.0, "speaker": "B"},
            ],
        },
    ]


def _seed_transcript_on_disk(
    output_dir: Path,
    *,
    job_id: str = JOB_ID,
    segments: list[dict] | None = None,
) -> Path:
    """Drop a Scribe-shaped transcript JSON under
    ``outputs/<job_id>/transcript.json`` so
    ``_load_segments_for_source_speaker_map`` can find it.

    Returns the directory the file was written to.
    """
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {"segments": segments if segments is not None else _segments_basic()}
    (job_dir / "transcript.json").write_text(json.dumps(payload))
    return job_dir


# --------------------------------------------------------------------------- #
# 1. Coding-view template surfaces the F4.6 play button + dock + JS.
# --------------------------------------------------------------------------- #


class TestCodingViewExposesPlaybackUI:
    """Without a user-visible button, a researcher with a coded segment
    has no way to play it back even when the data layer can compute
    the range. The dock + button must render in the template; the
    test markers anchor each affordance so future refactors don't drop
    them silently."""

    def test_play_dock_renders_in_template(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # Inline media dock — the host for the <audio> control.
        assert 'id="playDock"' in body
        assert 'data-test-feature="F4.6"' in body
        assert 'data-test-id="play-dock"' in body
        # The actual audio element the JS attaches /api/job/<id>/media to.
        assert 'id="playDockAudio"' in body
        assert 'data-test-id="play-dock-audio"' in body

    def test_play_button_template_string_present(self, server_env) -> None:
        """The per-row ▶ Play button is rendered by the JS template
        literal in renderApps(); we look for the exact attributes the
        click handler keys off."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert 'data-act="play"' in body
        assert 'data-test-id="app-row-play"' in body
        assert "▶ Play" in body

    def test_playback_handler_wired_into_app_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        # The click delegate keys off data-act='play'.
        assert "button[data-act='play']" in body
        # The async handler that calls the F4.6 endpoint.
        assert "function playApplication" in body or "async function playApplication" in body
        # The endpoint path the JS hits.
        assert "/applications/" in body
        assert "/playback" in body

    def test_helpers_module_shim_loads_playback_helpers(self, server_env) -> None:
        """The classic-script handler reaches the helpers.mjs F4.6
        functions through ``window.__playback``. Without this the JS
        side can't fall back to local resolution and the page becomes
        a hard dependency on the server route — that's a live-coding
        regression risk we want to lock down here."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert "playbackRangeForApplication" in body
        assert "window.__playback" in body
        assert 'CustomEvent("scribe:playback-ready")' in body

    def test_gutter_lane_bar_marked_as_playable(self, server_env) -> None:
        """F4.6 also wires the gutter lane bar (F4.3) so a researcher
        can scrub down the gutter to spot-check every code in document
        order. The bar carries an ARIA role + a data-test marker so the
        delegated click handler can find it."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        # The renderGutter() template literal carries the F4.6 markers.
        assert 'data-test-id="gutter-lane-bar"' in body or "gutter-lane-bar" in body
        assert "lane-bar[data-app-id]" in body


# --------------------------------------------------------------------------- #
# 2. The playback endpoint round-trips for the success case.
# --------------------------------------------------------------------------- #


class TestPlaybackEndpointSuccess:
    """The headline F4.6 case: a coded application with a transcript
    that has word-level timing should resolve to a non-empty
    [start, end] interval that matches the anchor word's bounds."""

    def test_returns_range_for_whole_word_anchor(
        self, server_env, tmp_path: Path
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _seed_transcript_on_disk(srv.OUTPUT_DIR)
        # "Hello" .. "world" — words 0..1 of segment 0; bounds 0.0-1.0.
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")

        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["application_id"] == aid
        assert body["source_id"] == sid
        assert body["transcript_job_id"] == JOB_ID
        assert body["start"] == pytest.approx(0.0)
        assert body["end"] == pytest.approx(1.0)

    def test_returns_range_crossing_segments(
        self, server_env, tmp_path: Path
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _seed_transcript_on_disk(srv.OUTPUT_DIR)
        # From "are" (s0w3) to "fine" (s1w2) — should bridge the
        # inter-segment gap and end on s1w2's end-time (3.6).
        aid = _apply(client, pid, cid, sid, "s0w3", "s1w2")

        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["start"] == pytest.approx(1.4)
        assert body["end"] == pytest.approx(3.6)

    def test_returns_range_with_subword_offsets(
        self, server_env, tmp_path: Path
    ) -> None:
        """Sub-word offsets interpolate proportionally across the anchor
        word's text. ``start_char_offset=0`` and ``end_char_offset=5``
        on "world" (chars 0..5 of a 5-char word) should round-trip the
        whole-word bounds."""
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _seed_transcript_on_disk(srv.OUTPUT_DIR)
        aid = _apply(
            client, pid, cid, sid, "s0w1", "s0w1",
            start_offset=0, end_offset=5,
        )
        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["start"] == pytest.approx(0.5)
        assert body["end"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. The playback endpoint surfaces structured failure modes.
# --------------------------------------------------------------------------- #


class TestPlaybackEndpointFailureModes:
    """F4.6 promises the UI a meaningful 404 + ``reason`` over a silent
    seek-to-zero. Every failure mode the pure module surfaces (no
    transcript, no segments, anchor out of range, no usable timing)
    has to map to an HTTP response the UI can read."""

    def test_404_when_application_missing(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        bogus_aid = "f" * 12
        r = client.get(f"/api/projects/{pid}/applications/{bogus_aid}/playback")
        assert r.status_code == 404
        assert r.json()["detail"] == "Application not found"

    def test_400_when_application_id_invalid(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/applications/not-a-real-id/playback")
        assert r.status_code == 400

    def test_404_no_transcript_when_source_unlinked(
        self, server_env
    ) -> None:
        """Source with ``transcript_job_id=None`` -> no playback. The
        ``reason`` field is the UI's hook to disable the play button
        with a helpful tooltip rather than seeking blindly."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid, job_id=None)
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")
        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 404
        body = r.json()
        # ``detail`` carries the structured payload because we passed a
        # dict to HTTPException.
        assert body["detail"]["reason"] == "no-transcript"

    def test_404_no_segments_when_transcript_dir_missing(
        self, server_env
    ) -> None:
        """``transcript_job_id`` is set but the outputs dir is empty —
        the transcript file simply isn't on disk. The endpoint returns
        ``reason=no-segments``."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")
        # No _seed_transcript_on_disk call — the directory doesn't exist.
        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 404
        body = r.json()
        assert body["detail"]["reason"] == "no-segments"

    def test_404_orphan_when_anchor_segment_out_of_range(
        self, server_env, tmp_path: Path
    ) -> None:
        """Anchor ``s5w0`` on a 2-segment transcript is the F4.5 orphan
        condition. F4.6 surfaces it as ``reason=orphan`` so the UI can
        route the user to the orphan queue."""
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _seed_transcript_on_disk(srv.OUTPUT_DIR)
        aid = _apply(client, pid, cid, sid, "s5w0", "s5w1")
        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 404
        body = r.json()
        assert body["detail"]["reason"] == "orphan"

    def test_404_no_timing_when_words_lack_timestamps(
        self, server_env, tmp_path: Path
    ) -> None:
        """A transcript whose words and segments both have no timing
        cannot resolve to a [start, end] interval. The endpoint returns
        ``reason=no-timing``."""
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # No timing on words or segment.
        untimed_segments = [
            {
                "speaker": "A",
                "text": "Hello world",
                "words": [
                    {"text": "Hello", "speaker": "A"},
                    {"text": "world", "speaker": "A"},
                ],
            },
        ]
        _seed_transcript_on_disk(
            srv.OUTPUT_DIR, segments=untimed_segments,
        )
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")
        r = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r.status_code == 404
        body = r.json()
        assert body["detail"]["reason"] == "no-timing"


# --------------------------------------------------------------------------- #
# 4. Endpoint ordering — /playback must not be captured by the
#    parametric ``/applications/{application_id}`` route as
#    application_id="<aid>" + something. (Belt-and-braces; FastAPI
#    matches by registration order, and adding /playback as a sub-path
#    of the parametric capture is the safe ordering.)
# --------------------------------------------------------------------------- #


class TestPlaybackEndpointOrdering:
    def test_playback_subroute_does_not_collide_with_get_application(
        self, server_env, tmp_path: Path
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _seed_transcript_on_disk(srv.OUTPUT_DIR)
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")

        # The bare GET still returns the application document.
        r1 = client.get(f"/api/projects/{pid}/applications/{aid}")
        assert r1.status_code == 200, r1.text
        assert r1.json()["id"] == aid

        # The /playback sub-path returns the F4.6 envelope.
        r2 = client.get(f"/api/projects/{pid}/applications/{aid}/playback")
        assert r2.status_code == 200, r2.text
        assert r2.json()["application_id"] == aid
        # And the two responses are clearly different shapes.
        assert "start" in r2.json()
        assert "start" not in r1.json()
