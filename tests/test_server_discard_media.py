"""End-to-end reachability tests for F10.2 (Discard source media).

The behaviour shipped in commit ``e3ba4b2`` with:

* ``Job.media_discarded`` field on the dataclass + persisted to ``job.json``
  (round-trip + tolerance for legacy/missing keys covered in
  ``tests/test_server.py::TestJobStatePersistence``).
* ``POST /api/job/{id}/discard-media`` endpoint that wipes
  ``uploads/<id>/`` and rewrites ``job.json`` with ``media_discarded: true``
  (happy-path + edge cases covered in
  ``tests/test_server.py::TestDiscardMediaAPI``).
* ``GET /api/job/{id}/media|info|waveform`` degrade to **HTTP 410 Gone**
  when the flag is set (covered in
  ``tests/test_server.py::TestMediaEndpointsAfterDiscard``).
* ``scribe/library.py`` summariser includes ``media_discarded`` in the row
  schema (unit-tested in ``tests/test_library.py``).
* The editor + library templates render the user-facing buttons + the
  📼 "media discarded" indicator (template tests scattered across
  ``tests/test_server.py``).

This file consolidates the **user-facing surface** proof for F10.2 into
one easy-to-find integration suite. It walks the path a real user takes:

    home → /library → click "Discard media" on a row  →  POST hits the
    endpoint  →  next /api/jobs reflects the flag  →  /api/job/<id>/media
    starts returning 410  →  editor for the same job renders the
    "Source media discarded" notice instead of a player.

Why a separate file: the Reachable-via gate (see
``scripts/feature-implementer-prompt.md``) requires every feature to
have an explicit, easy-to-find integration test that exercises the
end-to-end UI path. Keeping the F10.2 reachability proof grouped
per-feature makes the audit trail trivial to reconstruct from
``git log``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads/outputs/projects."""
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


def _new_job(srv, *, status: str = "done", **fields):
    """Drop a Job into srv.JOBS for endpoint tests."""
    job_id = fields.pop("id", "abc123def456")
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = srv.UPLOAD_DIR / job_id / "in.wav"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"\x00" * 64)
    job = srv.Job(
        id=job_id,
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
        media_discarded=fields.get("media_discarded", False),
    )
    srv.JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
# Library page renders the per-row Discard-media affordance
# --------------------------------------------------------------------------- #


class TestLibraryRendersDiscardAffordance:
    """The library page must (a) render a Discard-media button on rows
    that still have media and (b) expose the 📼 indicator on rows where
    the flag is already set. Without these the user has nowhere to
    invoke F10.2 from the library."""

    def test_library_page_renders_discard_button_template(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        assert r.status_code == 200
        # The renderer literal is what hard-wires the button into each
        # eligible row. The data-action attribute is the contract the
        # click delegate dispatches on.
        assert 'data-action="discard-media"' in r.text
        # The user-visible label.
        assert "Discard media" in r.text

    def test_library_page_renders_already_discarded_indicator(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # The 📼 emoji is the single-character visual flag for a row
        # whose media has already been reclaimed.
        assert "📼" in r.text
        # Tooltip / aria-label make the indicator accessible.
        assert "Source media discarded" in r.text

    def test_library_page_posts_to_discard_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # The page must POST to the F10.2 endpoint when the user clicks
        # the row button — the URL is the contract.
        assert "/discard-media" in r.text


# --------------------------------------------------------------------------- #
# Editor renders the Discard button + the discarded-state UI scaffolding
# --------------------------------------------------------------------------- #


class TestEditorRendersDiscardAffordance:
    """The editor's top-bar must render the "🗑 Source" button so a
    user reading a finished transcript can reclaim disk space without
    leaving the page. The discarded-state notice scaffolding must also
    render so the editor can degrade gracefully when the flag flips."""

    def test_editor_page_renders_discard_button(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, id="abc123def456")
        r = client.get("/edit/abc123def456")
        assert r.status_code == 200
        # The id is the contract the page's click handler binds to.
        assert 'id="discardMediaBtn"' in r.text
        # The user-visible label.
        assert "🗑 Source" in r.text

    def test_editor_page_renders_discarded_notice_scaffolding(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, id="abc123def456")
        r = client.get("/edit/abc123def456")
        # The notice card lives in the markup with hidden=True until
        # the JS unhides it after a successful POST or on boot when the
        # flag is already set. Without this scaffolding the editor's
        # graceful-degrade path has nowhere to land.
        assert 'id="mediaDiscardedNotice"' in r.text
        assert "Source media discarded" in r.text

    def test_editor_notice_hidden_attribute_actually_hides(
        self, server_env
    ) -> None:
        """The notice was rendering on *every* transcript regardless
        of the job's media_discarded flag. Cause: the
        ``.media-discarded-notice`` rule sets ``display: flex`` at
        class specificity, which beat the user-agent
        ``[hidden] { display: none }`` rule. Without the attribute-
        selector override here, every editor page renders the banner
        as if media had been discarded.

        Pin the override CSS in the rendered template so a future
        refactor of the styles can't silently drop it again."""
        srv, client, _ = server_env
        _new_job(srv, id="abc123def456")
        body = client.get("/edit/abc123def456").text
        # The banner element ships with the ``hidden`` attribute, so
        # the user-visible behaviour relies on the attribute being
        # honoured. Pin both the markup AND the override CSS rule.
        assert 'id="mediaDiscardedNotice" class="media-discarded-notice" hidden' in body
        assert ".media-discarded-notice[hidden]" in body
        assert "display: none" in body

    def test_editor_page_posts_to_discard_endpoint(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, id="abc123def456")
        r = client.get("/edit/abc123def456")
        # The button's click handler hits this URL — same endpoint the
        # library row uses. Surface contract.
        assert "/discard-media" in r.text


# --------------------------------------------------------------------------- #
# /api/job/{id}/discard-media reachability
# --------------------------------------------------------------------------- #


class TestDiscardMediaEndpointReachable:
    """The endpoint backing both UI surfaces. These tests pin the
    minimum behaviour the JS in editor.html and library.html depends on
    so refactors don't quietly break the click handler.
    """

    def test_post_discards_media_and_persists(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        # Sanity: source recording present.
        assert job.input_path.exists()

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200
        body = r.json()
        # Response shape is the contract — the editor's success branch
        # treats a 2xx as "discard happened" and ignores the body, but
        # the library handler reads ``ok`` to decide whether to flip
        # the row.
        assert body["ok"] is True
        assert body["id"] == job.id
        assert body["already"] is False

        # The flag is set on the live Job …
        assert srv.JOBS[job.id].media_discarded is True
        # … the upload directory is gone …
        assert not job.input_path.parent.exists()
        # … and the persisted job.json reflects the flag.
        persisted = srv._job_state_path(job.output_dir)
        assert persisted.exists()
        data = json.loads(persisted.read_text())
        assert data["media_discarded"] is True

    def test_post_is_idempotent(self, server_env) -> None:
        # The library page POSTs optimistically; a duplicate click
        # shouldn't 5xx. Idempotent behaviour pinned here so the UI
        # handler can rely on it.
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        r1 = client.post(f"/api/job/{job.id}/discard-media")
        r2 = client.post(f"/api/job/{job.id}/discard-media")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["already"] is False
        assert r2.json()["already"] is True

    def test_post_returns_404_for_unknown_job(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post("/api/job/abcdef012345/discard-media")
        assert r.status_code == 404

    def test_post_refuses_in_progress_job(self, server_env) -> None:
        # A worker still needs the source on disk; the endpoint must
        # 409 so the UI surfaces the error rather than silently corrupting
        # an in-flight transcription.
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456", status="running",
                       progress=0.5, message="Transcribing…")
        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 409
        # Flag unchanged; upload dir untouched.
        assert srv.JOBS[job.id].media_discarded is False
        assert job.input_path.exists()

    def test_post_also_deletes_cached_playback_mix(
        self, server_env
    ) -> None:
        """The multi-track player builds a cached
        ``<output>/playback.<hash>.<ext>`` mix on first /media GET.
        That file is *derived* from the source recording, so when the
        user discards media it has to go too — otherwise discarding
        leaves the synthesised mix on disk and the UI's "media gone"
        contract is a lie. Pinned because a regression here silently
        keeps tens of MB of audio on disk per discarded multi-track
        job."""
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        # Drop a fake cached playback file — the mix is opaque to the
        # endpoint, so we don't need ffmpeg.
        mix = job.output_dir / "playback.deadbeef.mp4"
        mix.write_bytes(b"fake mp4 bytes")
        # And a second one with a different selection hash; the
        # cleanup glob has to remove both.
        mix2 = job.output_dir / "playback.cafef00d.mp3"
        mix2.write_bytes(b"fake mp3 bytes")
        # An unrelated sidecar that must NOT be removed — only files
        # matching the playback.* glob are derived from the source.
        sidecar = job.output_dir / "transcript.json"
        sidecar.write_text('{"segments": []}')

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200, r.text
        assert not mix.exists()
        assert not mix2.exists()
        assert sidecar.exists(), \
            "Sidecar transcript JSON must survive discard"

    def test_post_succeeds_when_containment_check_fails(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even when the upload-dir containment check fails (unusual
        mount layout, manually-edited input_path, edge case the
        symlink-tolerant helper can't reason about), the discard
        endpoint still flips the persistent flag. The destructive
        cleanup is best-effort — the user's actual intent is "record
        that this row no longer has media" and that intent must not
        be blocked by a path-canonicalisation surprise.

        Pre-fix: a containment failure returned 403 and left
        media_discarded=False. The user reported hitting this
        repeatedly on a reattached job; even after the symlink-
        tolerant helper landed there were apparently still cases the
        check refused. This test pins that *any* containment failure
        gracefully degrades to 'flip the flag, skip the rmtree'."""
        from scribe import server as srv
        # Force the containment check to always say no, regardless of
        # the helper's underlying logic. That gives us a fixture-free
        # way to exercise the degraded path.
        monkeypatch.setattr(srv, "_link_or_path_is_under", lambda *a, **kw: False)
        srv2, client, _ = server_env
        job = _new_job(srv2, id="abc123def456")

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200, r.text
        body = r.json()
        # Flag flipped; the response carries cleaned_upload_dir=False
        # so a caller that wants to surface "couldn't clean up the
        # orphan dir" can do so. UI today just refreshes the row.
        assert body["already"] is False
        assert body["cleaned_upload_dir"] is False
        assert srv2.JOBS[job.id].media_discarded is True
        # The upload dir is left alone — Scribe didn't trust it
        # enough to rmtree, but the flag persistently records the
        # discard. The user can clean up by hand if they care.
        assert job.input_path.parent.exists()

    def test_post_succeeds_on_reattached_job(
        self, server_env, tmp_path: Path,
    ) -> None:
        """Real-world reproduction of the 403 the user reported.

        Reattach-media (``POST /api/job/<id>/reattach-media``)
        symlinks ``uploads/<id>/<filename>`` at a user-supplied
        absolute path that lives *outside* ``UPLOAD_DIR``. The
        discard endpoint used to ``resolve()`` ``input_path``,
        which followed the symlink to the real file, then refused
        the request with 403 "input_path escapes UPLOAD_DIR" —
        bricking discard for any reattached row.

        The fix: don't follow the symlink (``input_path.parent`` is
        already ``uploads/<id>/``, which IS under UPLOAD_DIR), and
        use the symlink-tolerant ``_link_or_path_is_under`` check.
        """
        srv, client, _ = server_env
        # External media file lives outside UPLOAD_DIR.
        external = tmp_path / "external" / "user-keeps-it-here.mp4"
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_bytes(b"\x00" * 64)

        # Build a job and re-link its input_path as a symlink to the
        # external file — same shape the reattach-media endpoint
        # leaves the FS in.
        job = _new_job(srv, id="abc123def456")
        original = job.input_path
        if original.exists():
            original.unlink()
        original.symlink_to(external)
        # Sanity: input_path is a symlink whose target is outside
        # UPLOAD_DIR — exactly the post-reattach state.
        assert original.is_symlink()
        assert original.resolve() == external.resolve()

        r = client.post(f"/api/job/{job.id}/discard-media")
        # Pre-fix: this returned 403 "input_path escapes UPLOAD_DIR".
        assert r.status_code == 200, r.text
        assert srv.JOBS[job.id].media_discarded is True
        # The symlink (and the uploads/<id>/ dir) are gone; the
        # external file the symlink pointed at is *not* touched —
        # Scribe doesn't own it.
        assert not original.parent.exists()
        assert external.exists(), \
            "Reattached external file must survive discard — Scribe doesn't own it."

    def test_post_succeeds_when_source_already_gone(
        self, server_env
    ) -> None:
        """User-reported scenario: someone deletes the source file
        out from under Scribe (manual rm, OS cleanup, external move).
        Pressing 'Discard media' should still succeed and flip the
        flag so the library row shows 'media discarded' — without
        this, the row reports 'has media' but every /media request
        returns a 404 from disk and the user has no way to clean up
        the persisted state."""
        import shutil as _sh
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        # Pull the upload dir out from under the running server,
        # then call discard. Under the current (working) impl, the
        # rmtree is a no-op and the persistent flag still flips.
        _sh.rmtree(job.input_path.parent, ignore_errors=True)
        assert not job.input_path.exists()

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200, r.text
        body = r.json()
        # ``already`` is False because the persistent flag wasn't set
        # before this call; what matters for the user is that the
        # flag is now True.
        assert body["already"] is False
        assert srv.JOBS[job.id].media_discarded is True
        # And the persisted job.json reflects it so the next library
        # render shows the indicator without a server restart.
        persisted = srv._job_state_path(job.output_dir)
        data = json.loads(persisted.read_text())
        assert data["media_discarded"] is True

    def test_library_reflects_missing_source_after_discard(
        self, server_env
    ) -> None:
        """Tie the discard endpoint to the library row the user
        actually looks at. After a discard on a job whose source was
        already gone, /api/jobs must return ``media_discarded: true``
        for that row — pinning the contract the library's indicator
        keys off of."""
        import shutil as _sh
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        _sh.rmtree(job.input_path.parent, ignore_errors=True)

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200

        rows = client.get("/api/jobs").json()["jobs"]
        match = [row for row in rows if row["id"] == job.id]
        assert match, "job missing from /api/jobs"
        assert match[0]["media_discarded"] is True

    def test_idempotent_call_also_cleans_stragglers(
        self, server_env
    ) -> None:
        """A first discard left the flag set but somehow missed a
        playback file (older Scribe build, or the mix was rebuilt
        between discards). The second idempotent call must still
        sweep ``playback.*`` so we don't leak audio on a re-discard."""
        srv, client, _ = server_env
        job = _new_job(
            srv, id="abc123def456", media_discarded=True,
        )
        # First call already happened in some prior life; the upload
        # dir is already gone and the flag is already set. Drop a
        # leftover playback file directly into the output dir.
        if job.input_path.parent.exists():
            import shutil as _sh
            _sh.rmtree(job.input_path.parent, ignore_errors=True)
        leftover = job.output_dir / "playback.abcd1234.mp4"
        leftover.write_bytes(b"leftover mp4")

        r = client.post(f"/api/job/{job.id}/discard-media")
        assert r.status_code == 200
        body = r.json()
        assert body["already"] is True
        assert not leftover.exists()


# --------------------------------------------------------------------------- #
# Media-bearing endpoints degrade to HTTP 410 once discarded
# --------------------------------------------------------------------------- #


class TestMediaEndpointsGoneAfterDiscard:
    """Three endpoints carry source-media payloads:
    ``/media`` (range-streamed audio/video), ``/info`` (ffprobe metadata),
    and ``/waveform`` (peak-amplitude cache). All three must return
    410 Gone once the flag is set so the editor knows to hide the
    player rather than render a broken <video> element. The transcript
    endpoint is **not** affected — that's the whole point of F10.2.
    """

    def test_media_returns_410_after_discard(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        # Pre-discard: a real GET would 200 + stream the file. We just
        # need to confirm the endpoint behaves before the flag flips,
        # then after.
        r_pre = client.get(f"/api/job/{job.id}/media")
        # Either 200 (file streamed) or 416 (range), but not 410.
        assert r_pre.status_code != 410

        client.post(f"/api/job/{job.id}/discard-media")
        r_post = client.get(f"/api/job/{job.id}/media")
        assert r_post.status_code == 410

    def test_info_returns_410_after_discard(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        client.post(f"/api/job/{job.id}/discard-media")
        r = client.get(f"/api/job/{job.id}/info")
        assert r.status_code == 410

    def test_waveform_returns_410_after_discard(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        client.post(f"/api/job/{job.id}/discard-media")
        r = client.get(f"/api/job/{job.id}/waveform")
        assert r.status_code == 410


# --------------------------------------------------------------------------- #
# /api/jobs reflects the flag for the library page
# --------------------------------------------------------------------------- #


class TestApiJobsExposesDiscardFlag:
    """The library page calls ``/api/jobs`` to populate rows. The flag
    must round-trip into the JSON payload so the row renderer can
    branch on it."""

    def test_api_jobs_carries_media_discarded_field(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, id="abc123def456")
        r = client.get("/api/jobs")
        assert r.status_code == 200
        rows = r.json()["jobs"]
        assert len(rows) == 1
        # Field present, default False.
        assert "media_discarded" in rows[0]
        assert rows[0]["media_discarded"] is False

    def test_api_jobs_flips_field_after_discard(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        client.post(f"/api/job/{job.id}/discard-media")
        r = client.get("/api/jobs")
        rows = r.json()["jobs"]
        assert rows[0]["media_discarded"] is True


# --------------------------------------------------------------------------- #
# End-to-end flow (the whole F10.2 surface in one walk)
# --------------------------------------------------------------------------- #


class TestEndToEndDiscardFlow:
    """A single test that follows the whole user path: arrive at the
    library, see a row, POST the discard, observe the row flag flip,
    confirm the editor still loads the transcript, confirm the media
    endpoint is now 410. If any link in this chain breaks, F10.2's
    surface is broken from the user's point of view.
    """

    def test_full_flow(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456",
                       input_filename="Interview.wav")

        # 1. Library renders the row + the Discard-media button is
        #    in the markup.
        lib = client.get("/library")
        assert lib.status_code == 200
        assert 'data-action="discard-media"' in lib.text

        # 2. /api/jobs (which library.html fetches) lists the job
        #    with media_discarded=False.
        rows = client.get("/api/jobs").json()["jobs"]
        assert rows[0]["id"] == job.id
        assert rows[0]["media_discarded"] is False

        # 3. The user clicks the button → page POSTs to the endpoint.
        post = client.post(f"/api/job/{job.id}/discard-media")
        assert post.status_code == 200
        assert post.json()["ok"] is True

        # 4. /api/jobs now reports media_discarded=True so the row
        #    renderer can switch to the 📼 indicator.
        rows = client.get("/api/jobs").json()["jobs"]
        assert rows[0]["media_discarded"] is True

        # 5. The editor still loads the transcript page (the whole
        #    point: keep the transcript, drop the source).
        edit = client.get(f"/edit/{job.id}")
        assert edit.status_code == 200
        # The discarded-notice scaffolding is in the page so the JS
        # can flip into degraded-playback mode on boot.
        assert 'id="mediaDiscardedNotice"' in edit.text

        # 6. The media endpoint now 410s — the editor's <video>/<audio>
        #    src would 410-out, which is exactly why the JS hides the
        #    player and shows the notice instead.
        media = client.get(f"/api/job/{job.id}/media")
        assert media.status_code == 410
