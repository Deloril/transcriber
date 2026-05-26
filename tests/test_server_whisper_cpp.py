"""End-to-end reachability tests for whisper.cpp / GGUF wiring (G7.2).

The pure-logic tests live in ``tests/test_whisper_cpp.py``. This file
proves the user-facing surface:

  * ``GET /api/whisper-cpp/models`` returns the supported model + quant
    catalogue plus the on-disk cached state.
  * ``GET /`` (the upload page) renders a quant selector
    (``data-test-id="whisper-cpp-quant-select"``) that the JS submits
    on upload alongside the existing ``backend`` field.
  * ``POST /api/upload`` accepts a ``whisper_cpp_quant`` form field,
    validates it (only when the chosen backend is whisper.cpp), and
    persists the choice on ``Job.whisper_cpp_quant``.
  * The persisted field round-trips through ``Job.from_state`` /
    ``to_state``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    # Pin the whisper.cpp cache to a clean tmp dir so the catalogue
    # listing is deterministic regardless of the dev's $HOME.
    monkeypatch.setenv("SCRIBE_WHISPER_CPP_CACHE", str(tmp_path / "wcpp-cache"))

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _stub_audio_probes(srv, monkeypatch: pytest.MonkeyPatch) -> None:
    from scribe import audio as scribe_audio

    monkeypatch.setattr(srv, "probe_audio_streams", lambda p: [
        scribe_audio.AudioStream(index=0, channels=2, title=None, language="eng", codec="aac"),
    ])
    monkeypatch.setattr(srv, "probe_media_info", lambda p: {"duration_seconds": 5.0})


# --------------------------------------------------------------------------- #
# A. /api/whisper-cpp/models — catalogue endpoint
# --------------------------------------------------------------------------- #


class TestWhisperCppModelsEndpoint:
    def test_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/whisper-cpp/models")
        assert r.status_code == 200

    def test_supported_models(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert body["supported_models"] == ["large-v3", "large-v3-turbo", "medium"]

    def test_supported_quants(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert body["supported_quants"] == ["q5_0", "q8_0", "f16"]

    def test_defaults_present(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert body["default_model"] == "large-v3"
        assert body["default_quant"] == "q5_0"

    def test_cache_dir_is_populated(self, server_env) -> None:
        _, client, tmp_path = server_env
        body = client.get("/api/whisper-cpp/models").json()
        # Pinned by the fixture's env var.
        assert body["cache_dir"] == str(tmp_path / "wcpp-cache")

    def test_pywhispercpp_availability_keys(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert "pywhispercpp_available" in body
        assert isinstance(body["pywhispercpp_available"], bool)
        assert "pywhispercpp_unavailable_reason" in body

    def test_models_is_full_grid(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert len(body["models"]) == 3 * 3

    def test_each_row_has_keys(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        for row in body["models"]:
            for k in (
                "model", "quant", "filename", "path",
                "cached", "size_bytes", "download_url",
            ):
                assert k in row, f"missing key {k} in {row}"

    def test_uncached_by_default(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        assert all(row["cached"] is False for row in body["models"])
        assert all(row["size_bytes"] is None for row in body["models"])

    def test_cached_row_after_drop_in(self, server_env) -> None:
        _, client, tmp_path = server_env
        cache = tmp_path / "wcpp-cache"
        cache.mkdir()
        gguf = cache / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake gguf payload")

        body = client.get("/api/whisper-cpp/models").json()
        cached = [r for r in body["models"] if r["cached"]]
        assert len(cached) == 1
        assert cached[0]["model"] == "large-v3"
        assert cached[0]["quant"] == "q5_0"
        assert cached[0]["size_bytes"] == len(b"fake gguf payload")

    def test_download_urls_huggingface(self, server_env) -> None:
        _, client, _ = server_env
        body = client.get("/api/whisper-cpp/models").json()
        for row in body["models"]:
            assert row["download_url"].startswith(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            )


# --------------------------------------------------------------------------- #
# B. Upload page renders the quant selector + JS submits the form field
# --------------------------------------------------------------------------- #


class TestIndexRendersQuantSelector:
    def test_index_returns_200(self, server_env) -> None:
        _, client, _ = server_env
        assert client.get("/").status_code == 200

    def test_quant_select_present(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        assert 'data-test-id="whisper-cpp-quant-select"' in html

    def test_panel_marker_present(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        assert 'data-test-id="whisper-cpp-panel"' in html

    def test_quant_options_listed(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        for q in ("q5_0", "q8_0", "f16"):
            assert f'value="{q}"' in html

    def test_default_quant_selected(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        # Find the quant select and verify q5_0 is selected.
        anchor = html.find('data-test-id="whisper-cpp-quant-select"')
        assert anchor != -1
        end = html.find("</select>", anchor)
        block = html[anchor:end]
        assert 'value="q5_0" selected' in block

    def test_supported_models_listed(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        for m in ("large-v3", "large-v3-turbo", "medium"):
            # The models are listed in the hint text wrapped in <code> tags.
            assert f"<code>{m}</code>" in html

    def test_form_js_appends_quant_field(self, server_env) -> None:
        _, client, _ = server_env
        html = client.get("/").text
        assert 'fd.append("whisper_cpp_quant"' in html

    def test_panel_starts_hidden(self, server_env) -> None:
        # faster-whisper is the default backend, so the panel should
        # render with .hidden so the user sees it appear only when
        # they switch to whisper.cpp.
        _, client, _ = server_env
        html = client.get("/").text
        anchor = html.find('id="whisperCppPanel"')
        assert anchor != -1
        # Look forward for the closing of the opening tag.
        end = html.find(">", anchor)
        opening_tag = html[anchor:end]
        assert "hidden" in opening_tag


# --------------------------------------------------------------------------- #
# C. /api/upload accepts and validates whisper_cpp_quant
# --------------------------------------------------------------------------- #


class TestUploadAcceptsQuant:
    def test_default_quant_when_omitted(
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
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_cpp_quant == "q5_0"

    def test_explicit_quant_persisted_for_whisper_cpp(
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
                "model": "large-v3",
                "backend": "whisper.cpp",
                "whisper_cpp_quant": "q8_0",
            },
        )
        assert r.status_code == 200, r.text
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_backend == "whisper.cpp"
        assert job.whisper_cpp_quant == "q8_0"

    def test_unsupported_quant_rejected_400_when_whisper_cpp(
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
                "model": "large-v3",
                "backend": "whisper.cpp",
                "whisper_cpp_quant": "q4_0",  # not in supported set
            },
        )
        assert r.status_code == 400
        assert "q4_0" in r.json()["detail"]

    def test_unsupported_quant_ignored_when_other_backend(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The validation is backend-aware: when the user picked
        # faster-whisper, we don't enforce the whisper.cpp set
        # (the value is inert anyway). Instead, the form's submitted
        # value is captured but the worker doesn't use it.
        srv, client, _ = server_env
        _stub_audio_probes(srv, monkeypatch)
        monkeypatch.setattr(srv, "_run_job", lambda jid: None)

        r = client.post(
            "/api/upload",
            files={"file": ("x.mp4", b"\x00" * 16, "audio/mp4")},
            data={
                "mode": "auto",
                "language": "en",
                "model": "large-v3",
                "backend": "faster-whisper",
                "whisper_cpp_quant": "q4_0",  # ignored
            },
        )
        assert r.status_code == 200, r.text

    def test_blank_quant_falls_through_to_default(
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
                "model": "large-v3",
                "backend": "whisper.cpp",
                "whisper_cpp_quant": "",
            },
        )
        assert r.status_code == 200, r.text
        job = srv.JOBS[r.json()["job_id"]]
        assert job.whisper_cpp_quant == "q5_0"


# --------------------------------------------------------------------------- #
# D. Persistence — Job.to_state / from_state round-trip
# --------------------------------------------------------------------------- #


class TestJobPersistsQuant:
    def test_from_state_default(self) -> None:
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
        assert job.whisper_cpp_quant == "q5_0"

    def test_from_state_preserves_explicit(self) -> None:
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
            "whisper_cpp_quant": "f16",
        })
        assert job.whisper_cpp_quant == "f16"

    def test_to_state_includes_quant(self, tmp_path: Path) -> None:
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
            whisper_cpp_quant="q8_0",
        )
        state = job.to_state()
        assert state["whisper_cpp_quant"] == "q8_0"


# --------------------------------------------------------------------------- #
# E. Worker dispatch — the engine call carries whisper_cpp_quant
# --------------------------------------------------------------------------- #


class TestWorkerDispatchesQuant:
    """Verify that ``_run_job`` plumbs ``Job.whisper_cpp_quant`` into the
    engine's ``transcribe(...)`` call so the choice survives all the
    way to ``WhisperCppBackend.transcribe``. This is the bridge
    between the upload form and :mod:`scribe.whisper_cpp`."""

    def test_run_job_passes_quant_to_transcribe(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, _client, tmp_path = server_env
        captured: dict[str, Any] = {}

        def _fake_transcribe(
            input_path: Path, **kwargs: Any
        ) -> Any:
            captured.update(kwargs)
            from scribe.engine import TranscriptionResult
            return TranscriptionResult(
                segments=[],
                language="en",
                mode="diarize",
                speaker_labels=[],
                audio_path=input_path,
            )

        def _fake_write_all(result: Any, base: Path) -> dict[str, Path]:
            return {}

        monkeypatch.setattr(srv, "transcribe", _fake_transcribe)
        monkeypatch.setattr(srv, "write_all", _fake_write_all)

        out_dir = tmp_path / "outputs" / "abc123def456"
        out_dir.mkdir(parents=True)
        in_path = tmp_path / "uploads" / "abc123def456" / "in.wav"
        in_path.parent.mkdir(parents=True)
        in_path.write_bytes(b"x")

        job = srv.Job(
            id="abc123def456",
            input_path=in_path,
            output_dir=out_dir,
            mode="diarize",
            speakers=None,
            num_speakers=None,
            language="en",
            model="large-v3",
            created_at="2026-05-27T00:00:00Z",
            whisper_backend="whisper.cpp",
            whisper_cpp_quant="q8_0",
        )
        with srv.JOBS_LOCK:
            srv.JOBS[job.id] = job
        srv._run_job(job.id)

        assert captured.get("whisper_backend") == "whisper.cpp"
        assert captured.get("whisper_cpp_quant") == "q8_0"
