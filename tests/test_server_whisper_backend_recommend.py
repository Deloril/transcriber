"""End-to-end reachability tests for the G7.3 Apple-Silicon recommendation.

G7.3 builds on G7.1 / G7.2 by **flipping the default backend on
Apple Silicon** and surfacing a recommendation banner above the
engine selector. This file proves both the JSON surface
(``GET /api/whisper-backends``) and the UI surface
(``GET /`` rendering the banner) reflect the active GPU backend.

The pure-logic helpers (``default_backend_id`` /
``recommended_backend_for_device``) are covered in
``tests/test_whisper_backend.py``.
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


def _force_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Pin ``engine.gpu_backend()`` to ``backend`` for this test.

    The server reads the active GPU backend via
    ``scribe.engine.gpu_backend()``; monkeypatching the live function
    is the same pattern the G1.1 helper tests use.
    """
    from scribe import engine
    monkeypatch.setattr(engine, "gpu_backend", lambda: backend)


def _force_whisper_cpp_available(
    monkeypatch: pytest.MonkeyPatch, available: bool, reason: str = "",
) -> None:
    """Pin the whisper.cpp backend's ``is_available()`` for this test."""
    from scribe import whisper_backend as wb
    cpp = wb.get_backend("whisper.cpp")
    monkeypatch.setattr(
        cpp, "is_available", lambda: (available, reason),
    )


# --------------------------------------------------------------------------- #
# A. /api/whisper-backends returns device-aware default + recommendation
# --------------------------------------------------------------------------- #


class TestWhisperBackendsApiOnAppleSilicon:
    def test_default_flips_to_whisper_cpp_when_mps_and_available(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)

        body = client.get("/api/whisper-backends").json()
        assert body["default"] == "whisper.cpp", (
            "G7.3 — on mps with whisper.cpp installed, the default "
            "must flip so /api/whisper-backends matches the page render"
        )

    def test_default_stays_faster_whisper_when_mps_but_unavailable(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(
            monkeypatch, False, "pywhispercpp not installed",
        )

        body = client.get("/api/whisper-backends").json()
        # Don't land on a backend that can't run.
        assert body["default"] == "faster-whisper"

    def test_default_stays_faster_whisper_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cuda")
        _force_whisper_cpp_available(monkeypatch, True)
        body = client.get("/api/whisper-backends").json()
        assert body["default"] == "faster-whisper"

    def test_default_stays_faster_whisper_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "rocm")
        _force_whisper_cpp_available(monkeypatch, True)
        body = client.get("/api/whisper-backends").json()
        assert body["default"] == "faster-whisper"

    def test_default_stays_faster_whisper_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cpu")
        _force_whisper_cpp_available(monkeypatch, True)
        body = client.get("/api/whisper-backends").json()
        assert body["default"] == "faster-whisper"

    def test_active_gpu_backend_echoed_in_response(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        body = client.get("/api/whisper-backends").json()
        assert body["active_gpu_backend"] == "mps"

    def test_recommendation_present_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        body = client.get("/api/whisper-backends").json()
        rec = body["recommendation"]
        assert rec is not None
        assert rec["recommended_backend_id"] == "whisper.cpp"
        assert rec["device"] == "mps"
        assert rec["available"] is True

    def test_recommendation_absent_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cuda")
        body = client.get("/api/whisper-backends").json()
        assert body["recommendation"] is None

    def test_recommendation_absent_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "rocm")
        body = client.get("/api/whisper-backends").json()
        assert body["recommendation"] is None

    def test_recommendation_absent_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cpu")
        body = client.get("/api/whisper-backends").json()
        assert body["recommendation"] is None

    def test_recommendation_carries_install_reason_when_unavailable(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(
            monkeypatch, False, "pywhispercpp not installed: No module named 'pywhispercpp'",
        )
        body = client.get("/api/whisper-backends").json()
        rec = body["recommendation"]
        assert rec is not None
        assert rec["available"] is False
        assert "pywhispercpp" in rec["unavailable_reason"]


# --------------------------------------------------------------------------- #
# B. Upload page renders the recommendation banner on mps
# --------------------------------------------------------------------------- #


class TestIndexRendersRecommendationBanner:
    def test_banner_renders_on_mps_when_available(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        html = client.get("/").text
        assert 'data-test-id="whisper-backend-recommendation"' in html
        # The "default flipped" confirmation also lands.
        assert 'data-test-id="whisper-backend-recommendation-flipped"' in html

    def test_banner_renders_on_mps_with_install_prompt_when_unavailable(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(
            monkeypatch, False, "pywhispercpp not installed",
        )
        html = client.get("/").text
        assert 'data-test-id="whisper-backend-recommendation"' in html
        # When unavailable, the install prompt replaces the
        # "default flipped" line.
        assert 'data-test-id="whisper-backend-recommendation-install"' in html
        assert 'data-test-id="whisper-backend-recommendation-flipped"' not in html

    def test_banner_carries_active_gpu_backend_dataset(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        html = client.get("/").text
        assert 'data-active-gpu-backend="mps"' in html
        assert 'data-recommended-backend="whisper.cpp"' in html

    def test_banner_mentions_speedup_number(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The hint is "GPU-accelerated transcription, ~5× faster".
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        html = client.get("/").text
        # Find the banner block and assert the speedup mention.
        i = html.find('data-test-id="whisper-backend-recommendation"')
        assert i != -1
        # Take a generous slice of HTML around the banner element.
        slice_ = html[i: i + 2000]
        assert "5×" in slice_

    def test_banner_absent_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cuda")
        html = client.get("/").text
        assert 'data-test-id="whisper-backend-recommendation"' not in html

    def test_banner_absent_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "rocm")
        html = client.get("/").text
        assert 'data-test-id="whisper-backend-recommendation"' not in html

    def test_banner_absent_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cpu")
        html = client.get("/").text
        assert 'data-test-id="whisper-backend-recommendation"' not in html


# --------------------------------------------------------------------------- #
# C. Engine <select> reflects the device-aware default
# --------------------------------------------------------------------------- #


class TestIndexEngineSelectDefaultFlipsOnMps:
    def test_whisper_cpp_pre_selected_on_mps_when_available(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        html = client.get("/").text

        # Locate the whisper.cpp option tag and confirm "selected".
        snippet_start = html.find('value="whisper.cpp"')
        assert snippet_start != -1
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "selected" in opt_tag, (
            "G7.3 — on mps with whisper.cpp installed, the engine "
            "<option value='whisper.cpp'> must carry 'selected' so "
            "the page lands with the Metal-accelerated path active"
        )

    def test_faster_whisper_not_selected_on_mps_when_available(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(monkeypatch, True)
        html = client.get("/").text

        snippet_start = html.find('value="faster-whisper"')
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "selected" not in opt_tag

    def test_faster_whisper_still_pre_selected_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "cuda")
        html = client.get("/").text

        snippet_start = html.find('value="faster-whisper"')
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "selected" in opt_tag

    def test_faster_whisper_still_pre_selected_on_mps_when_unavailable(
        self, server_env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, client, _ = server_env
        _force_backend(monkeypatch, "mps")
        _force_whisper_cpp_available(
            monkeypatch, False, "pywhispercpp not installed",
        )
        html = client.get("/").text

        snippet_start = html.find('value="faster-whisper"')
        tag_start = html.rfind("<option", 0, snippet_start)
        tag_end = html.find(">", snippet_start)
        opt_tag = html[tag_start:tag_end + 1]
        assert "selected" in opt_tag, (
            "G7.3 — on mps without pywhispercpp installed, the page "
            "must fall back to faster-whisper so the user can still "
            "transcribe (defaulting to a backend that can't run "
            "would be a worse experience than the speedup hint)"
        )
