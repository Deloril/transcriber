"""End-to-end reachability tests for F10.3 (Import an existing transcript).

The behaviour shipped in commit ``a4de170`` with:

* ``scribe.transcript_import`` — pure parsers for TXT / SRT / VTT /
  Scribe JSON (60 unit tests in ``tests/test_transcript_import.py``).
* ``POST /api/import`` — endpoint that turns an uploaded transcript
  file into a finished job, optionally with a companion media file
  (13 happy/sad-path tests in
  ``tests/test_server.py::TestImportTranscriptAPI``).
* ``index.html`` — collapsible "Already have a transcript? Import it"
  card under the upload card with two drop zones (transcript +
  optional media) and an "Import transcript" button that POSTs to
  ``/api/import`` and on success redirects to ``/edit/<job_id>``.

The original commit predates the loop's Reachable-via gate (see
``scripts/feature-implementer-prompt.md``) and so its commit body is
missing the line that lets the loop confirm the surface is wired.
This file consolidates the F10.3 reachability proof into a single,
easy-to-find integration suite that walks the user-facing path:

    home (/) → expand the "Already have a transcript?" card → drop a
    transcript file → click Import transcript → POST /api/import →
    job lands as ``status=done`` → ``/api/jobs`` (library backing)
    surfaces it → ``/edit/<id>`` opens with the imported transcript.

If any link in that chain breaks (button id renamed, route URL
changes, `media_discarded` flag dropped, library hides imports) one
of these tests fires.
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


# --------------------------------------------------------------------------- #
# Home page renders the F10.3 import affordance
# --------------------------------------------------------------------------- #


class TestHomePageRendersImportCard:
    """The home page (``GET /``) must render the import card with the
    drop zone + button. Without these the user cannot reach
    ``POST /api/import`` from the UI: that's the whole point of F10.3.
    """

    def test_home_page_renders_html(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_home_page_has_import_card(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The collapsible card is the entry point for the whole flow.
        assert 'id="importCard"' in r.text

    def test_home_page_advertises_supported_formats(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The four formats F10.3 ships parsers for must be advertised
        # on the card so users know what they can drop in.
        assert "TXT" in r.text
        assert "SRT" in r.text
        assert "VTT" in r.text
        # Either spelling is fine — the card calls it "Scribe JSON".
        assert "Scribe JSON" in r.text or "scribe json" in r.text.lower()

    def test_home_page_has_transcript_drop_zone(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The drop zone + hidden file input the page's JS reads from.
        assert 'id="importDrop"' in r.text
        assert 'id="importFile"' in r.text

    def test_home_page_has_companion_media_drop_zone(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The optional media drop zone is what makes
        # ``media_discarded=False`` reachable from the import flow.
        assert 'id="importMediaDrop"' in r.text
        assert 'id="importMediaFile"' in r.text

    def test_home_page_has_import_button(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The button id is the contract: the JS click handler depends
        # on it. Visible label is the user-facing text.
        assert 'id="importGo"' in r.text
        assert "Import transcript" in r.text

    def test_home_page_posts_to_api_import(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # The page must POST to /api/import — that's how the button is
        # wired to the backend route. If this disappears the button is
        # cosmetic.
        assert "/api/import" in r.text

    def test_home_page_redirects_to_editor_after_import(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        r = client.get("/")
        # On success the JS sends the user to /edit/<job_id>. Without
        # this the import succeeds but the user is left on the home
        # page with nothing to click.
        assert "/edit/" in r.text


# --------------------------------------------------------------------------- #
# POST /api/import is reachable + handles the four formats
# --------------------------------------------------------------------------- #


class TestImportEndpointReachable:
    """Hitting ``POST /api/import`` for each supported format must
    create a finished job and return a payload the home-page JS can
    redirect on. Each format gets one smoke test; the deeper parser
    behaviour is covered in ``tests/test_transcript_import.py``."""

    def test_post_imports_txt(self, server_env) -> None:
        srv, client, _ = server_env
        body = b"[00:00] LUKE: Hello.\n\n[00:03] GUEST: Hi back.\n"
        r = client.post(
            "/api/import",
            files={"transcript": ("interview.txt", body, "text/plain")},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["imported"] is True
        assert j["format"] == "txt"
        assert j["job_id"] in srv.JOBS
        assert srv.JOBS[j["job_id"]].status == "done"

    def test_post_imports_srt(self, server_env) -> None:
        _, client, _ = server_env
        body = (
            b"1\n00:00:00,000 --> 00:00:02,000\nLUKE: Hello.\n\n"
            b"2\n00:00:02,500 --> 00:00:04,000\nGUEST: Hi.\n"
        )
        r = client.post(
            "/api/import",
            files={"transcript": ("clip.srt", body, "application/x-subrip")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["format"] == "srt"

    def test_post_imports_vtt(self, server_env) -> None:
        _, client, _ = server_env
        body = (
            b"WEBVTT\n\n"
            b"00:00:00.000 --> 00:00:02.000\n<v LUKE>Hello.\n"
        )
        r = client.post(
            "/api/import",
            files={"transcript": ("clip.vtt", body, "text/vtt")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["format"] == "vtt"

    def test_post_imports_scribe_json(self, server_env) -> None:
        _, client, _ = server_env
        payload = {
            "language": "en",
            "mode": "diarize",
            "speakers": ["LUKE"],
            "segments": [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "LUKE",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 1.0}
                    ],
                }
            ],
        }
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "tx.json",
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["format"] == "scribe-json"

    def test_post_unparseable_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        # An empty file matches no parser; the route surfaces a 400 so
        # the import card's error UI can show it.
        r = client.post(
            "/api/import",
            files={"transcript": ("nope.txt", b"", "text/plain")},
        )
        assert r.status_code == 400

    def test_post_unknown_format_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/import",
            files={"transcript": ("foo.txt", b"[00:00] hi\n", "text/plain")},
            data={"fmt": "csv"},  # not a known format
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Imported jobs flow into the rest of the user-facing surface
# --------------------------------------------------------------------------- #


class TestImportFlowsIntoLibraryAndEditor:
    """An imported job has to look indistinguishable from a transcribed
    one to the rest of the app — otherwise the import card is a dead
    end. This class exercises the chain: import → /api/jobs → /library
    → /edit/<id>.
    """

    def test_imported_job_appears_in_api_jobs(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "interview.txt",
                    b"[00:00] LUKE: Hi.\n",
                    "text/plain",
                )
            },
        )
        job_id = r.json()["job_id"]

        r2 = client.get("/api/jobs")
        assert r2.status_code == 200
        ids = [row["id"] for row in r2.json()["jobs"]]
        assert job_id in ids

    def test_imported_job_marked_media_discarded_when_no_companion(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "interview.txt",
                    b"[00:00] LUKE: Hi.\n",
                    "text/plain",
                )
            },
        )
        j = r.json()
        assert j["media_discarded"] is True
        assert srv.JOBS[j["job_id"]].media_discarded is True

    def test_imported_job_keeps_media_when_companion_attached(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "tx.txt",
                    b"[00:00] LUKE: Hi.\n",
                    "text/plain",
                ),
                # A tiny WAV-shaped blob is enough; the route stores it
                # and the editor can pick it up.
                "media": (
                    "tx.wav",
                    b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 32,
                    "audio/wav",
                ),
            },
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["media_discarded"] is False
        # The companion file was saved under uploads/<id>/.
        assert (srv.UPLOAD_DIR / j["job_id"]).is_dir()
        files = list((srv.UPLOAD_DIR / j["job_id"]).iterdir())
        assert any(p.name == "tx.wav" for p in files)

    def test_editor_opens_for_imported_job(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "interview.txt",
                    b"[00:00] LUKE: Hello world.\n",
                    "text/plain",
                )
            },
        )
        job_id = r.json()["job_id"]

        # The home page redirects to /edit/<id> after a successful
        # import; that page must render or the redirect is a dead end.
        r2 = client.get(f"/edit/{job_id}")
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("text/html")
        # The editor's own filename hint should pick up the imported
        # filename so the user knows which file they're looking at.
        assert "interview.txt" in r2.text

    def test_library_page_renders_after_import(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/import",
            files={
                "transcript": (
                    "interview.txt",
                    b"[00:00] LUKE: Hi.\n",
                    "text/plain",
                )
            },
        )
        assert r.status_code == 200
        r2 = client.get("/library")
        # The library page is a thin client over /api/jobs and would
        # render an empty list even with imports present, so we just
        # assert the page itself is reachable. The imported-row
        # surfacing is exercised in test_imported_job_appears_in_api_jobs.
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("text/html")


# --------------------------------------------------------------------------- #
# End-to-end: home → import → editor (one chain)
# --------------------------------------------------------------------------- #


class TestEndToEndImportFlow:
    """One walk that links every step a real user takes for F10.3.
    If any step regresses, this test fires loudly.
    """

    def test_home_to_import_to_editor(self, server_env) -> None:
        srv, client, _ = server_env

        # 1. Land on the home page; the import card is right there.
        r = client.get("/")
        assert r.status_code == 200
        assert 'id="importCard"' in r.text
        assert 'id="importGo"' in r.text
        assert "/api/import" in r.text

        # 2. Drop a transcript and submit.
        r2 = client.post(
            "/api/import",
            files={
                "transcript": (
                    "session.txt",
                    b"[00:00] LUKE: Imported transcript end-to-end.\n",
                    "text/plain",
                )
            },
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        job_id = body["job_id"]
        assert body["imported"] is True
        assert body["format"] == "txt"

        # 3. The new job is queryable via the library API.
        r3 = client.get("/api/jobs")
        assert r3.status_code == 200
        rows = r3.json()["jobs"]
        assert any(row["id"] == job_id for row in rows)
        # Imported jobs default to media_discarded=True since this
        # caller didn't pass a companion file.
        target = next(row for row in rows if row["id"] == job_id)
        assert target.get("media_discarded") is True

        # 4. The editor page (the JS redirect target) loads.
        r4 = client.get(f"/edit/{job_id}")
        assert r4.status_code == 200

        # 5. The persisted job.json round-trips so a server restart
        # keeps the import alive.
        srv.JOBS.clear()
        srv._load_jobs_from_disk()
        assert job_id in srv.JOBS
        assert srv.JOBS[job_id].status == "done"
