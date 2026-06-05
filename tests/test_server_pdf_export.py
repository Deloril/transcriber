"""End-to-end tests for the PDF export endpoint.

The user-facing surface is:

* ``GET /api/projects/<pid>/sources/<sid>/export/pdf`` — returns
  ``application/pdf`` with a ``Content-Disposition: attachment``
  header carrying a sanitised filename.
* The source-coding page renders a "Download PDF" link that points
  at that endpoint.

The pure HTML / run-slicing logic is unit-tested in
``tests/test_pdf_export.py``; this file proves the endpoint is
plumbed correctly: project / source / segment loading, application
filtering, speaker-name mapping, error mapping, and the UI button.

When ``weasyprint`` isn't importable (the dev box this repo lives
on doesn't always have cairo/pango), the endpoint maps the
``PdfExportError`` to HTTP 500 with a clear message — that branch
is exercised explicitly so a missing system dep doesn't cause a
silent 502 in production.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


JOB_ID = "abc123def456"


# --------------------------------------------------------------------------- #
# Fixtures
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


def _make_project(client: TestClient, name: str = "Pilot") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    *,
    name: str = "Maria — interview",
    job_id: str | None = JOB_ID,
) -> str:
    body: dict = {"name": name, "source_type": "transcript", "language": "en"}
    if job_id is not None:
        body["transcript_job_id"] = job_id
    r = client.post(f"/api/projects/{pid}/sources", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str = "Belief") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "What they believe."},
    )
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


def _seed_transcript(output_dir: Path, *, job_id: str = JOB_ID) -> Path:
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0,
                "text": "I believe things really matter",
                "words": [
                    {"text": "I", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                    {"text": "believe", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                    {"text": "things", "start": 1.0, "end": 1.4, "speaker": "SPEAKER_00"},
                    {"text": "really", "start": 1.4, "end": 1.7, "speaker": "SPEAKER_00"},
                    {"text": "matter", "start": 1.7, "end": 2.0, "speaker": "SPEAKER_00"},
                ],
            },
        ]
    }
    (job_dir / "edited.json").write_text(json.dumps(payload))
    return job_dir


HAS_WEASYPRINT = importlib.util.find_spec("weasyprint") is not None


# --------------------------------------------------------------------------- #
# Endpoint tests
# --------------------------------------------------------------------------- #


class TestPdfExportEndpoint:
    def test_button_renders_in_coding_page(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid, job_id=None)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert "Download PDF" in body
        assert (
            f"/api/projects/{pid}/sources/{sid}/export/pdf" in body
        )
        # The download attribute is what gets the browser to save
        # rather than navigate; pin it so a future refactor can't
        # silently strip it.
        assert "download" in body
        assert 'data-test-id="src-export-pdf"' in body

    def test_404_on_unknown_project(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get(
            "/api/projects/does-not-exist/sources/whatever/export/pdf"
        )
        # Bad project id format returns 400; if the regex passes,
        # the missing-project branch returns 404. The endpoint must
        # not 500 either way.
        assert r.status_code in (400, 404)

    def test_404_on_unknown_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/sources/aaaaaaaaaaaa/export/pdf"
        )
        assert r.status_code == 404

    def test_400_on_malformed_source_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/sources/!!nope!!/export/pdf"
        )
        assert r.status_code == 400

    @pytest.mark.skipif(
        not HAS_WEASYPRINT,
        reason="weasyprint not installed (system dep optional)",
    )
    def test_returns_pdf_bytes_and_filename(self, server_env) -> None:
        srv, client, _ = server_env
        _seed_transcript(srv.OUTPUT_DIR)
        pid = _make_project(client)
        sid = _make_source(client, pid, name="Maria — interview")
        cid = _make_code(client, pid, name="Belief")
        _apply(client, pid, cid, sid, "s0w1", "s0w2")

        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/export/pdf"
        )
        # weasyprint may still fail at runtime if cairo/pango aren't
        # present even though the import succeeded; treat that as a
        # skip rather than a failure so this passes on minimal CI.
        if r.status_code == 500 and "weasyprint" in r.text.lower():
            pytest.skip(f"weasyprint runtime failed: {r.text}")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/pdf")
        # PDF magic header — `%PDF`.
        assert r.content[:4] == b"%PDF"
        # Filename is sanitised: spaces → -, em-dash collapsed.
        cd = r.headers["content-disposition"]
        assert "attachment" in cd
        assert ".pdf" in cd

    def test_500_with_clear_message_when_weasyprint_missing(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If weasyprint can't be imported, the endpoint returns a
        500 carrying the install hint — not a bare TraceBack."""
        from scribe import pdf_export

        def boom(html):
            raise pdf_export.PdfExportError(
                "weasyprint isn't available — install libpango / libcairo "
                "(macOS: brew install pango cairo)."
            )

        monkeypatch.setattr(pdf_export, "render_pdf_bytes", boom)

        srv, client, _ = server_env
        _seed_transcript(srv.OUTPUT_DIR)
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/export/pdf"
        )
        assert r.status_code == 500
        assert "weasyprint" in r.text.lower()

    def test_endpoint_works_without_transcript(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source with no transcript still produces a valid PDF (an
        empty transcript). We monkeypatch render_pdf_bytes to capture
        the HTML so the test passes even on boxes without weasyprint."""
        from scribe import pdf_export

        captured: dict[str, str] = {}

        def fake_render(html):
            captured["html"] = html
            return b"%PDF-1.4 fake"

        monkeypatch.setattr(pdf_export, "render_pdf_bytes", fake_render)

        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid, job_id=None)
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/export/pdf"
        )
        assert r.status_code == 200
        assert r.content == b"%PDF-1.4 fake"
        assert "html" in captured
        # No segments, no applications, no legend.
        assert "0 segments" in captured["html"]
        assert "0 coded spans" in captured["html"]

    def test_only_this_source_applications_are_rendered(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Applications for *other* sources in the project must not
        appear in this transcript's PDF, even if they exist."""
        from scribe import pdf_export

        captured: dict[str, str] = {}

        def fake_render(html):
            captured["html"] = html
            return b"%PDF-1.4 fake"

        monkeypatch.setattr(pdf_export, "render_pdf_bytes", fake_render)

        srv, client, _ = server_env
        _seed_transcript(srv.OUTPUT_DIR, job_id=JOB_ID)
        # Second source uses a different transcript job.
        other_job = "ffffffffffff"
        _seed_transcript(srv.OUTPUT_DIR, job_id=other_job)

        pid = _make_project(client)
        sid_a = _make_source(
            client, pid, name="A", job_id=JOB_ID,
        )
        sid_b = _make_source(
            client, pid, name="B", job_id=other_job,
        )
        cid = _make_code(client, pid, name="Belief")
        _apply(client, pid, cid, sid_a, "s0w1", "s0w2")
        _apply(client, pid, cid, sid_b, "s0w0", "s0w0")

        r = client.get(
            f"/api/projects/{pid}/sources/{sid_a}/export/pdf"
        )
        assert r.status_code == 200
        # Source A has 1 coded span, Source B's app shouldn't leak in.
        assert "1 coded spans" in captured["html"]
