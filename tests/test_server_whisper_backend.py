"""End-to-end reachability tests for the Whisper-backend selector (G7.1).

The pluggable inference-engine abstraction lives in
``scribe.whisper_backend`` (covered by ``tests/test_whisper_backend.py``).
This file proves the user-facing surface:

  * ``GET /api/whisper-backends`` returns the registered backends.
  * ``GET /`` (the upload page) renders an Engine ``<select>``
    populated with both backends, with ``faster-whisper`` selected
    by default and ``whisper.cpp`` advertised but disabled.
  * ``POST /api/upload`` accepts a ``backend`` form field, validates
    it against the registry, and stores the choice on the persisted
    ``Job`` so the worker can route the inference call.
  * An unknown ``backend`` is rejected with HTTP 400.
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


def _stub_audio_probes(srv, monkeypatch: pytest.MonkeyPatch) -> None:
    from scribe import audio as scribe_audio

    monkeypatch.setattr(srv, "probe_audio_streams", lambda p: [
        scribe_audio.AudioStream(index=0, channels=2, title=None, language="eng", codec="aac"),
    ])
    monkeypatch.setattr(srv, "probe_media_info", lambda p: {"duration_seconds": 5.0})


# --------------------------------------------------------------------------- #
# A. /api/whisper-backends endpoint
# --------------------------------------------------------------------------- #


class TestWhisperBackendsEndpoint:
    def test_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/whisper-backends")
        assert r.status_code == 200

    def test_advertises_default(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-backends").json()
        assert body["default"] == "faster-whisper"

    def test_advertises_both_backends(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-backends").json()
        ids = [b["id"] for b in body["backends"]]
        assert "faster-whisper" in ids
        assert "whisper.cpp" in ids

    def test_each_backend_has_required_keys(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-backends").json()
        for b in body["backends"]:
            for k in (
                "id", "display_name", "description",
                "supported_devices", "model_format",
                "available", "unavailable_reason",
            ):
                assert k in b

    def test_faster_whisper_lists_no_mps(self, server_env) -> None:
        # The G7.x motivation — call it out here so a future change
        # that adds Metal to CT2's supported set requires a
        # deliberate edit to this test.
        _, client, _ = server_env
        body = client.get("/api/whisper-backends").json()
        fw = next(b for b in body["backends"] if b["id"] == "faster-whisper")
        assert "mps" not in fw["supported_devices"]


# --------------------------------------------------------------------------- #
# B. Upload page renders the Engine selector
# --------------------------------------------------------------------------- #


class TestIndexRendersEngineSelector:
    def test_index_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200

    def test_index_html_contains_engine_select(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # The select carries a stable test-id so the page is
        # scriptable from end-to-end tests.
        assert 'data-test-id="whisper-backend-select"' in html

    def test_index_html_lists_faster_whisper_option(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # Default selected.
        assert 'value="faster-whisper"' in html

    def test_index_html_lists_whisper_cpp_option(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # Even though the backend is unavailable today, the option
        # is still rendered (greyed out via ``disabled``) so the
        # user can see it's coming.
        assert 'value="whisper.cpp"' in html

    def test_whisper_cpp_option_is_disabled(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # The disabled attribute appears on the same option element
        # as the value="whisper.cpp". Check they're co-located.
        snippet_start = html.find('value="whisper.cpp"')
        assert snippet_start != -1
        # Find the enclosing <option ...> tag (search backwards from
        # the value= attribute) and confirm "disabled" is present.
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "disabled" in opt_tag

    def test_default_backend_pre_selected(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # The faster-whisper option must carry ``selected``.
        snippet_start = html.find('value="faster-whisper"')
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "selected" in opt_tag

    def test_form_js_appends_backend_field(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # The submit handler must include the backend in the form
        # data; check the literal append call.
        assert 'fd.append("backend"' in html


# --------------------------------------------------------------------------- #
# C. /api/upload accepts and validates the ``backend`` form field
# --------------------------------------------------------------------------- #


class TestUploadAcceptsBackend:
    def test_default_backend_when_omitted(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={"mode": "auto", "language": "en", "model": "tiny"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        job = srv.JOBS[job_id]
        assert job.whisper_backend == "faster-whisper"

    def test_explicit_faster_whisper_persisted(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={
                "mode": "auto",
                "language": "en",
                "model": "tiny",
                "backend": "faster-whisper",
            },
        )
        assert r.status_code == 200, r.text
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_backend == "faster-whisper"

    def test_whisper_cpp_accepted_even_though_unavailable(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The registry advertises whisper.cpp; the server validates
        # against the registry, not against the runtime availability
        # check. The actual transcribe call would fail when the
        # worker dispatches — that's a separate concern; for the
        # upload endpoint, the id just has to be registered. This
        # mirrors how disabled UI options can still be POSTed by
        # advanced / scripted clients (G7.2 will flip the
        # availability flag once the adapter ships).
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={
                "mode": "auto",
                "language": "en",
                "model": "tiny",
                "backend": "whisper.cpp",
            },
        )
        assert r.status_code == 200, r.text
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_backend == "whisper.cpp"

    def test_unknown_backend_rejected_400(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={
                "mode": "auto",
                "language": "en",
                "model": "tiny",
                "backend": "imaginary-engine",
            },
        )
        assert r.status_code == 400
        assert "imaginary-engine" in r.json()["detail"]

    def test_blank_backend_falls_through_to_default(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={
                "mode": "auto",
                "language": "en",
                "model": "tiny",
                "backend": "",
            },
        )
        assert r.status_code == 200, r.text
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_backend == "faster-whisper"


# --------------------------------------------------------------------------- #
# D. Persistence — round-trip through Job.from_state()
# --------------------------------------------------------------------------- #


class TestJobPersistsBackend:
    def test_job_from_state_default(self) -> None:
        from scribe.server import Job
        job = Job.from_state({
            "id": "abc123def456",
            "input_path": "/tmp/in.wav",
            "output_dir": "/tmp/out",
            "mode": "diarize",
            "speakers": None,
            "num_speakers": None,
            "language": "en",
            "model": "large-v3",
            "created_at": "2026-05-27T00:00:00Z",
        })
        assert job.whisper_backend == "faster-whisper"

    def test_job_from_state_preserves_explicit_backend(self) -> None:
        from scribe.server import Job
        job = Job.from_state({
            "id": "abc123def456",
            "input_path": "/tmp/in.wav",
            "output_dir": "/tmp/out",
            "mode": "diarize",
            "speakers": None,
            "num_speakers": None,
            "language": "en",
            "model": "large-v3",
            "created_at": "2026-05-27T00:00:00Z",
            "whisper_backend": "whisper.cpp",
        })
        assert job.whisper_backend == "whisper.cpp"

    def test_job_to_state_includes_backend(self, tmp_path: Path) -> None:
        from scribe.server import Job
        job = Job(
            id="abc123def456",
            input_path=tmp_path / "in.wav",
            output_dir=tmp_path / "out",
            mode="diarize",
            speakers=None,
            num_speakers=None,
            language="en",
            model="large-v3",
            created_at="2026-05-27T00:00:00Z",
            whisper_backend="whisper.cpp",
        )
        state = job.to_state()
        assert state["whisper_backend"] == "whisper.cpp"
