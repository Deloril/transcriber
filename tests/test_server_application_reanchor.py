"""End-to-end reachability tests for F4.5 (Re-anchor strategy on
transcript edit + orphan review queue).

Background
----------

F4.5 shipped ``scribe.application_reanchor`` in 2ce8928 — a pure
planner that, given the *old* transcript and the *new* transcript,
decides per-application whether the anchor is unchanged, can be
re-anchored to the same content elsewhere, or has been orphaned
(its words are gone). 83 unit tests exercise the algorithm in
``tests/test_application_reanchor.py``.

What was missing — and what this file covers — is the integration
proof that the **user-facing surface** is wired together. Per the
loop's done-criteria, F4.5 is only "done" if a researcher can reach
the data layer through a real route + a real UI control. That means:

1. ``PUT /api/job/<id>/transcript`` must auto-run the F4.5 planner
   for every project whose Source links to that job, applying
   re-anchored outcomes in place and queuing orphans for human
   triage. The PUT response carries a ``reanchor`` summary the
   editor can surface as a toast.
2. ``GET /api/projects/<pid>/orphan_applications`` must return the
   queue contents.
3. ``DELETE /api/projects/<pid>/orphan_applications/<aid>`` must
   remove a queue entry.
4. ``GET /projects/<pid>/orphans`` must render the review page with
   the table-and-buttons surface a coder uses to triage.
5. The coding view (``GET /projects/<pid>/sources/<sid>``) must
   render an "Orphan queue" link that lands on the review page —
   without it, a coder who edited a transcript has no idea any of
   their applications got orphaned.

Without this file the F4.5 ID would be in the commit log but with
no proof the user can reach the orphan queue — exactly the failure
mode the loop's done-detector is designed to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures — mirror tests/test_server_applications.py / _spans.py.
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
# Builders — every test starts from a project + code + source +
# transcript-backed job. The transcript text is short on purpose so
# the F4.5 ``unchanged`` / ``reanchored`` / ``orphaned`` outcomes can
# all be hit deterministically.
# --------------------------------------------------------------------------- #


JOB_ID = "abc123def456"


def _make_project(client: TestClient, name: str = "F4.5 holder") -> str:
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
) -> str:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": cid,
            "source_id": sid,
            "anchor_start_word_id": start,
            "anchor_end_word_id": end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _segments(*sentences: list[str]) -> list[dict]:
    """Tiny helper: each sentence is a list of word-text strings, one
    segment per sentence. ``s<i>w<j>`` ids are the canonical F4.1
    word-id shape (the engine produces them).
    """
    out = []
    for i, words in enumerate(sentences):
        out.append({
            "start": float(i),
            "end": float(i + 1),
            "speaker": "A",
            "text": " ".join(words),
            "words": [
                {"text": w, "start": float(i), "end": float(i) + 0.1 * (j + 1)}
                for j, w in enumerate(words)
            ],
        })
    return out


def _new_job(srv, *, job_id: str = JOB_ID, result: dict | None = None):
    """Drop a Job into srv.JOBS for endpoint tests that need one
    without going through /api/upload. Mirrors tests/test_server.py
    helper of the same name."""
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = srv.UPLOAD_DIR / job_id / "in.wav"
    input_path.parent.mkdir(parents=True, exist_ok=True)
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
        created_at="2026-05-25T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result=result,
        error=None,
        output_paths={},
        audio_streams=1,
        input_filename="in.wav",
        options={},
        batch_size=8,
        started_at=1.0,
        finished_at=2.0,
    )
    srv.JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
# 1. The coding-view template surfaces the F4.5 orphan-queue link.
# --------------------------------------------------------------------------- #


class TestCodingViewExposesOrphanQueueLink:
    """Without a user-visible link, a coder who edits a transcript has
    no way to discover that some of their applications got orphaned.
    The link is rendered unconditionally (the JS hydrates the count);
    queue-empty just means the badge is blank."""

    def test_link_renders_on_source_coding_page(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        # Anchor + test marker so future refactors don't drop it.
        assert 'id="orphanQueueLink"' in body
        assert 'data-test-feature="F4.5"' in body
        assert 'data-test-id="src-orphan-queue-link"' in body
        # Lands on the F4.5 review page.
        assert f'href="/projects/{pid}/orphans"' in body
        # Visible label so a researcher knows what they're clicking.
        assert "Orphan queue" in body

    def test_link_has_count_badge_slot(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert 'id="orphanQueueCount"' in body
        assert 'data-test-id="src-orphan-queue-count"' in body
        # The hydration helper is wired up.
        assert "function refreshOrphanQueueCount" in body
        assert "refreshOrphanQueueCount();" in body
        # And it hits the F4.5 endpoint.
        assert "/orphan_applications" in body


# --------------------------------------------------------------------------- #
# 2. The orphan-review page renders.
# --------------------------------------------------------------------------- #


class TestOrphanReviewPageRenders:
    """``GET /projects/<pid>/orphans`` must render the review page
    with a recognisable container so the link from the coding view
    actually lands somewhere usable."""

    def test_page_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/orphans")
        assert r.status_code == 200
        body = r.text
        # The page header + table marker.
        assert "Orphan applications" in body
        # The two action affordances must be discoverable in the
        # template (the actual rows are added by JS, but the
        # test markers are in the template literal).
        assert 'data-test-id="orphan-table"' in body
        assert "Delete application" in body
        assert "Dismiss" in body
        # JS is wired to the F4.5 endpoint.
        assert "/api/projects/" in body
        assert "/orphan_applications" in body

    def test_page_links_back_to_project(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/orphans")
        assert r.status_code == 200
        assert f'href="/projects/{pid}"' in r.text


# --------------------------------------------------------------------------- #
# 3. The orphan-queue endpoints round-trip.
# --------------------------------------------------------------------------- #


class TestOrphanEndpointsRoundTrip:
    """Empty queue → 200 with [] · queue with entries → returns them ·
    DELETE one → 200 + GET no longer has it."""

    def test_empty_queue(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/orphan_applications")
        assert r.status_code == 200
        body = r.json()
        assert body == {"orphans": [], "count": 0}

    def test_returns_queued_entries(self, server_env, tmp_path: Path) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        # Seed an orphan entry directly via the data layer so this test
        # doesn't depend on the PUT /transcript pipeline.
        from scribe.application_reanchor import (
            OrphanEntry,
            append_orphan_entries,
        )
        entry = OrphanEntry(
            application_id="abcdef012345",
            code_id="abcdef012346",
            source_id="abcdef012347",
            coder_id="abcdef012348",
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w3",
            original_anchored_text=["hello", "there", "world"],
            reason="anchored text not found in new transcript",
            detected_at="2026-05-27T00:00:00Z",
        )
        append_orphan_entries(srv.PROJECTS_DIR, pid, [entry])

        r = client.get(f"/api/projects/{pid}/orphan_applications")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["orphans"][0]["application_id"] == "abcdef012345"
        assert body["orphans"][0]["original_anchored_text"] == [
            "hello", "there", "world",
        ]
        assert body["orphans"][0]["reason"]

    def test_delete_removes_an_entry(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        from scribe.application_reanchor import (
            OrphanEntry,
            append_orphan_entries,
        )
        keep = OrphanEntry(
            application_id="abcdef012345",
            code_id="abcdef012346",
            source_id="abcdef012347",
            coder_id="abcdef012348",
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w1",
            original_anchored_text=["hi"],
            reason="x",
            detected_at="2026-05-27T00:00:00Z",
        )
        gone = OrphanEntry(
            application_id="abcdef012349",
            code_id="abcdef01234a",
            source_id="abcdef01234b",
            coder_id="abcdef01234c",
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w2",
            original_anchored_text=["bye", "now"],
            reason="y",
            detected_at="2026-05-27T01:00:00Z",
        )
        append_orphan_entries(srv.PROJECTS_DIR, pid, [keep, gone])

        # Delete one.
        r = client.delete(
            f"/api/projects/{pid}/orphan_applications/abcdef012349"
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # GET reflects it.
        body = client.get(
            f"/api/projects/{pid}/orphan_applications"
        ).json()
        assert body["count"] == 1
        assert body["orphans"][0]["application_id"] == "abcdef012345"

    def test_delete_404_for_unknown_entry(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.delete(
            f"/api/projects/{pid}/orphan_applications/abcdef012345"
        )
        assert r.status_code == 404

    def test_endpoints_validate_ids(self, server_env) -> None:
        _, client, _ = server_env
        # Bad project id → 400/404 (router rejects long/garbled paths).
        r = client.get("/api/projects/!!notanid/orphan_applications")
        assert r.status_code in (400, 404)
        # Bad application id on DELETE → 400.
        pid = _make_project(client)
        r = client.delete(
            f"/api/projects/{pid}/orphan_applications/NOTHEX"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 4. The transcript PUT auto-reanchors and queues orphans.
#
# This is the integration that proves F4.5 is reachable through the
# editor save flow rather than only as a free-floating module.
# --------------------------------------------------------------------------- #


class TestPutTranscriptTriggersReanchor:

    def test_unchanged_anchor_keeps_application(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Apply the code over "hello there".
        old_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["hello", "there", "world"],
                ["second", "segment", "here"],
            ),
        }
        _new_job(srv, result=old_payload)
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")

        # Edit a *different* segment so the anchor's words are
        # untouched. F4.5 ``unchanged`` should fire.
        new_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["hello", "there", "world"],
                ["different", "wording", "now"],
            ),
        }
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text

        body = r.json()
        # The reanchor summary mentions our project + source.
        summaries = body.get("reanchor", [])
        match = next(
            (s for s in summaries if s.get("source_id") == sid),
            None,
        )
        assert match is not None, f"no summary for our source: {summaries}"
        assert match["unchanged"] == 1
        assert match["reanchored"] == 0
        assert match["orphaned"] == 0

        # The application is still on disk with the original anchors.
        a = client.get(f"/api/projects/{pid}/applications/{aid}").json()
        assert a["anchor_start_word_id"] == "s0w0"
        assert a["anchor_end_word_id"] == "s0w1"

        # And the orphan queue is empty.
        q = client.get(f"/api/projects/{pid}/orphan_applications").json()
        assert q["count"] == 0

    def test_reanchored_application_updates_to_new_word_ids(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Anchor over "hello there" in segment 0.
        old_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["hello", "there", "world"],
                ["unrelated", "filler"],
            ),
        }
        _new_job(srv, result=old_payload)
        aid = _apply(client, pid, cid, sid, "s0w0", "s0w1")

        # Insert a new word at the start of segment 0; "hello there"
        # now lives at s0w1..s0w2 instead of s0w0..s0w1. F4.5
        # ``reanchored`` should fire.
        new_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["um,", "hello", "there", "world"],
                ["unrelated", "filler"],
            ),
        }
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text

        summaries = r.json().get("reanchor", [])
        match = next(
            (s for s in summaries if s.get("source_id") == sid),
            None,
        )
        assert match is not None
        assert match["reanchored"] == 1
        assert match["orphaned"] == 0

        # The application now points at the new word ids.
        a = client.get(f"/api/projects/{pid}/applications/{aid}").json()
        assert a["anchor_start_word_id"] == "s0w1"
        assert a["anchor_end_word_id"] == "s0w2"

        # No orphans queued.
        q = client.get(f"/api/projects/{pid}/orphan_applications").json()
        assert q["count"] == 0

    def test_orphaned_application_lands_in_queue(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Anchor over "managing pain" — wording the new transcript will
        # not contain at all.
        old_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["I", "was", "managing", "pain", "yesterday"],
            ),
        }
        _new_job(srv, result=old_payload)
        aid = _apply(client, pid, cid, sid, "s0w2", "s0w3")

        # Anonymisation pass strips the whole segment's payload —
        # the original words are gone and don't appear elsewhere.
        new_payload = {
            "language": "en",
            "mode": "diarize",
            "segments": _segments(
                ["I", "was", "[redacted]", "yesterday"],
            ),
        }
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text

        summaries = r.json().get("reanchor", [])
        match = next(
            (s for s in summaries if s.get("source_id") == sid),
            None,
        )
        assert match is not None
        assert match["orphaned"] == 1

        # The application is preserved (it kept its old anchors), AND
        # an orphan-queue entry was created for it.
        a = client.get(f"/api/projects/{pid}/applications/{aid}").json()
        assert a["anchor_start_word_id"] == "s0w2"
        assert a["anchor_end_word_id"] == "s0w3"

        q = client.get(f"/api/projects/{pid}/orphan_applications").json()
        assert q["count"] == 1
        ent = q["orphans"][0]
        assert ent["application_id"] == aid
        assert ent["code_id"] == cid
        assert ent["source_id"] == sid
        assert ent["original_anchored_text"] == ["managing", "pain"]

    def test_only_sources_linked_to_this_job_are_touched(
        self, server_env
    ) -> None:
        """A second source with a *different* job id must not have its
        applications inspected (and must not create stray orphans)
        when an unrelated job is saved."""
        srv, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid_match = _make_source(client, pid, "Linked", job_id=JOB_ID)
        sid_other = _make_source(client, pid, "Other", job_id="ffeeddccbbaa")
        _apply(client, pid, cid, sid_other, "s0w0", "s0w0")

        old_payload = {
            "segments": _segments(["hello", "world"]),
        }
        _new_job(srv, result=old_payload)
        # Apply on the matching source — this one will reanchor.
        _apply(client, pid, cid, sid_match, "s0w0", "s0w1")

        new_payload = {
            "segments": _segments(["hello", "world", "again"]),
        }
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text

        summaries = r.json().get("reanchor", [])
        # Exactly one summary — the matching source.
        ours = [s for s in summaries if s.get("project_id") == pid]
        sids = {s.get("source_id") for s in ours}
        assert sids == {sid_match}, sids

    def test_pure_module_features_still_work(self, server_env) -> None:
        """Sanity: even after the F4.5 wiring runs, the F4.1
        round-trip is intact (we haven't broken edit-on-empty-state).
        """
        srv, client, _ = server_env
        pid = _make_project(client)
        _make_source(client, pid)
        old_payload = {
            "segments": _segments(["one", "two"]),
        }
        _new_job(srv, result=old_payload)
        new_payload = {
            "segments": _segments(["one", "two", "three"]),
        }
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text
        # No applications → empty summary, but still 200.
        assert isinstance(r.json().get("reanchor"), list)


# --------------------------------------------------------------------------- #
# 5. Smoke check on the helper itself: skipping projects whose disk
#    state is missing must not crash a transcript save.
# --------------------------------------------------------------------------- #


class TestReanchorHelperResilience:
    """The reanchor helper's failure mode is "skip and continue" —
    a single broken project must not block the transcript edit."""

    def test_handles_zero_projects(self, server_env) -> None:
        srv, client, _ = server_env
        old_payload = {"segments": _segments(["hi"])}
        _new_job(srv, result=old_payload)
        new_payload = {"segments": _segments(["hi", "there"])}
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text
        assert r.json().get("reanchor") == []

    def test_handles_source_with_no_job_link(self, server_env) -> None:
        """Sources that have no transcript_job_id (or one that doesn't
        match the edited job) must be ignored cleanly."""
        srv, client, _ = server_env
        pid = _make_project(client)
        # ``transcript_job_id=None`` skips the constructor's link.
        sid = _make_source(client, pid, "Detached", job_id=None)
        cid = _make_code(client, pid)
        # Not strictly needed, but a no-link application proves the
        # filter actually filters.
        _apply(client, pid, cid, sid, "s0w0", "s0w0")

        old_payload = {"segments": _segments(["x"])}
        _new_job(srv, result=old_payload)
        new_payload = {"segments": _segments(["x", "y"])}
        r = client.put(f"/api/job/{JOB_ID}/transcript", json=new_payload)
        assert r.status_code == 200, r.text
        # No source matches → no summaries.
        summaries = [s for s in r.json().get("reanchor", []) if s.get("project_id") == pid]
        assert summaries == []
