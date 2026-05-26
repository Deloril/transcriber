"""End-to-end reachability tests for F10.1 (Library view).

The pure summariser lives in ``scribe/library.py`` with unit tests in
``tests/test_library.py``. The HTTP endpoints (``GET /api/jobs``,
``DELETE /api/job/<id>``) and the ``/library`` page-render are
exercised in ``tests/test_server.py::TestLibraryAPI``,
``tests/test_server.py::TestLibraryPage``, and
``tests/test_server.py::TestDeleteJobAPI``. The CSS layout fix from
F10.4 is exercised in ``tests/test_library_layout.py``.

This file consolidates the **user-facing surface** proof for F10.1
into a single, easy-to-find integration suite. It walks the same path
a real user follows: arrive at the home page → click "📚 Library" →
the library page loads, fetches ``/api/jobs``, and renders rows whose
action buttons (Open, JSON, SRT, Discard media, Delete) wire up to
real endpoints.

Why a separate file: the Reachable-via gate (see
``scripts/feature-implementer-prompt.md``) requires that every feature
have a single, easy-to-find integration test that exercises the
end-to-end UI path. Keeping the F10.1 reachability proof grouped
per-feature makes the audit trail trivial to reconstruct from
``git log``.
"""

from __future__ import annotations

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
    )
    srv.JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
# /library is discoverable from the home page
# --------------------------------------------------------------------------- #


class TestLibraryDiscoverableFromHome:
    """The home page must surface a clickable link to ``/library`` so a
    fresh user can find the library without typing the URL."""

    def test_home_page_links_to_library(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        # The href is the contract — the icon/label can change but the
        # link target cannot disappear.
        assert 'href="/library"' in r.text

    def test_home_page_library_link_has_visible_label(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        # The 📚 Library affordance is what users actually click.
        assert "📚 Library" in r.text


# --------------------------------------------------------------------------- #
# /library page renders the F10.1 controls
# --------------------------------------------------------------------------- #


class TestLibraryPageRendersControls:
    """The page must render the user-visible affordances F10.1 promises:
    a search box, sortable column headers, an empty-state message, and
    the JS that fetches ``/api/jobs`` to populate rows.
    """

    def test_page_renders_html(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "Scribe Library" in r.text

    def test_page_has_search_box(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # The search input must exist with the id the page's JS reads
        # from. The exact id is the contract: change it and the
        # client-side filter stops working silently.
        assert 'id="search"' in r.text
        assert 'type="search"' in r.text

    def test_page_has_sortable_columns(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # Every column F10.1 promised to sort by must carry data-sort.
        for col in (
            "input_filename",
            "duration_seconds",
            "mode",
            "speaker_count",
            "language",
            "created_at",
            "status",
        ):
            assert f'data-sort="{col}"' in r.text, f"missing sortable column: {col}"

    def test_page_fetches_api_jobs(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # The page is a live thin client over /api/jobs — without that
        # fetch the rows would never appear.
        assert 'fetch("/api/jobs")' in r.text

    def test_page_uses_helpers_for_filter_and_sort(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/library")
        # Search + sort run client-side via the helpers pure module.
        assert "searchLibraryRows" in r.text
        assert "compareLibraryRows" in r.text


# --------------------------------------------------------------------------- #
# /api/jobs surfaces the F10.1 row schema
# --------------------------------------------------------------------------- #


class TestApiJobsContract:
    """``GET /api/jobs`` is the read endpoint the library page calls.
    These tests pin the row schema so future refactors don't quietly
    drop a field the UI depends on.
    """

    def test_empty_registry_returns_empty_list(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert r.json() == {"jobs": [], "total": 0}

    def test_row_schema_matches_renderer(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(
            srv,
            id="abc123def456",
            input_filename="Interview.wav",
            mode="diarize",
            language="en",
            created_at="2026-05-25T10:00:00Z",
            result={
                "language": "en",
                "speakers": ["Luke", "Maria"],
                "segments": [
                    {"start": 0, "end": 60, "speaker": "Luke", "text": "hi", "words": []},
                ],
            },
            output_paths={"json": "outputs/abc123def456/x.json"},
        )
        r = client.get("/api/jobs")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        row = body["jobs"][0]
        # Every field renderRow() in library.html reads from must
        # appear in the row payload. If one disappears, F10.1 breaks
        # silently.
        for key in (
            "id",
            "input_filename",
            "mode",
            "language",
            "status",
            "speakers",
            "speaker_count",
            "duration_seconds",
            "has_outputs",
            "media_discarded",
            "created_at",
        ):
            assert key in row, f"missing row field: {key}"
        assert row["id"] == "abc123def456"
        assert row["speaker_count"] == 2
        assert row["has_outputs"] is True

    def test_query_param_filters_rows(self, server_env) -> None:
        srv, client, _ = server_env
        _new_job(srv, id="aaa111bbb222", input_filename="alpha.wav")
        _new_job(srv, id="ccc333ddd444", input_filename="bravo.wav")
        r = client.get("/api/jobs", params={"q": "bravo"})
        assert r.status_code == 200
        body = r.json()
        # ``total`` counts the rows after the server-side filter,
        # matching what the page would render. The unfiltered listing
        # still works without ``q`` (covered by
        # test_row_schema_matches_renderer).
        assert body["total"] == 1
        ids = [row["id"] for row in body["jobs"]]
        assert ids == ["ccc333ddd444"]


# --------------------------------------------------------------------------- #
# DELETE /api/job/<id> — the row's "Delete" button target
# --------------------------------------------------------------------------- #


class TestDeleteJobReachableFromLibraryRow:
    """The Delete button on each library row POSTs to this endpoint.
    These tests confirm the route exists, removes the registry entry,
    and wipes the on-disk artefacts.
    """

    def test_delete_removes_job_from_registry_and_disk(self, server_env) -> None:
        srv, client, _ = server_env
        job = _new_job(srv, id="abc123def456")
        assert job.id in srv.JOBS
        assert job.input_path.exists()
        assert job.output_dir.exists()

        r = client.delete(f"/api/job/{job.id}")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "id": job.id}

        assert job.id not in srv.JOBS
        assert not job.input_path.parent.exists()
        assert not job.output_dir.exists()

    def test_delete_unknown_id_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        # 12-hex-char id that isn't in the registry.
        r = client.delete("/api/job/abcdef012345")
        assert r.status_code == 404

    def test_after_delete_library_listing_omits_row(self, server_env) -> None:
        srv, client, _ = server_env
        kept = _new_job(srv, id="aaa111bbb222")
        gone = _new_job(srv, id="ccc333ddd444")

        r = client.delete(f"/api/job/{gone.id}")
        assert r.status_code == 200

        r = client.get("/api/jobs")
        assert r.status_code == 200
        ids = [row["id"] for row in r.json()["jobs"]]
        assert ids == [kept.id]


# --------------------------------------------------------------------------- #
# End-to-end: home → library → delete → reload
# --------------------------------------------------------------------------- #


class TestEndToEndLibraryFlow:
    """Walks the same path a real user follows: home page → library
    page → API → delete a row → API again. If any link in this chain
    breaks, the F10.1 surface is no longer reachable.
    """

    def test_full_flow(self, server_env) -> None:
        srv, client, _ = server_env

        # 1. Home page exposes the library link.
        r = client.get("/")
        assert r.status_code == 200
        assert 'href="/library"' in r.text

        # 2. Two finished jobs in the registry.
        _new_job(srv, id="aaa111bbb222", input_filename="alpha.wav")
        _new_job(srv, id="ccc333ddd444", input_filename="bravo.wav")

        # 3. /library page renders.
        r = client.get("/library")
        assert r.status_code == 200
        assert "Scribe Library" in r.text

        # 4. /api/jobs returns both rows.
        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert r.json()["total"] == 2

        # 5. The Delete button's endpoint removes one of them.
        r = client.delete("/api/job/aaa111bbb222")
        assert r.status_code == 200

        # 6. /api/jobs now returns just the survivor.
        r = client.get("/api/jobs")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["jobs"][0]["id"] == "ccc333ddd444"
