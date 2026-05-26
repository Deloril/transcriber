"""Tests for Parakeet visibility on AMD ROCm (G5.1).

NeMo (the runtime that loads NVIDIA Parakeet) is CUDA-only — there's no
ROCm support and no community fork. The upload page hides the Parakeet
optgroup entirely when the active backend is ROCm, and the model-hint
strip surfaces a "doesn't run on the active <Backend> backend" warning
when the user has Parakeet pre-selected from a saved profile (e.g.
moved a profile from a CUDA box to an AMD box).

The decision logic lives in pure helpers in
``scribe/static/js/helpers.mjs`` (covered by Vitest) and is wired up by
``scribe/templates/index.html``. These Python-side tests pin the
template wiring + the server-side capabilities contract that the
helpers consume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from scribe import server as srv
from scribe import engine, parakeet


TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scribe"
    / "templates"
    / "index.html"
)


# --------------------------------------------------------------------------- #
# Template wiring
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def index_html() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


class TestIndexTemplateWiring:
    """The upload page must import and call the G5.1 helpers."""

    def test_imports_shouldHideParakeetOptgroup(self, index_html: str) -> None:
        # The helper has to be imported at the top of the module
        # script so it's in scope by the time updateModelHint runs.
        assert "shouldHideParakeetOptgroup" in index_html

    def test_imports_parakeetModelHint(self, index_html: str) -> None:
        assert "parakeetModelHint" in index_html

    def test_optgroup_is_present_for_other_backends(self, index_html: str) -> None:
        # The default rendering still ships the Parakeet options;
        # the JS hides them at runtime on ROCm. Server-side rendering
        # is backend-agnostic.
        assert 'optgroup label="NVIDIA Parakeet' in index_html
        assert "nvidia/parakeet-tdt-0.6b-v2" in index_html

    def test_calls_shouldHideParakeetOptgroup(self, index_html: str) -> None:
        # Pin the wiring: the optgroup display toggle must go
        # through the helper, not duplicate the rocm string match.
        assert "shouldHideParakeetOptgroup(" in index_html

    def test_calls_parakeetModelHint(self, index_html: str) -> None:
        assert "parakeetModelHint(" in index_html

    def test_no_inline_rocm_string_match(self, index_html: str) -> None:
        # The previous implementation hard-coded `=== "rocm"` inline.
        # That logic now lives behind shouldHideParakeetOptgroup so
        # the test suite can reach it. Make sure we didn't leave the
        # duplicate behind.
        assert 'backend === "rocm"' not in index_html


# --------------------------------------------------------------------------- #
# Server-side capabilities contract — what the helpers consume
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Force NeMo to look importable so the parakeet sub-payload is
    # populated; otherwise the gating test is meaningless.
    monkeypatch.setattr(parakeet, "_NEMO_AVAILABLE", True)
    monkeypatch.setattr(parakeet, "_IMPORT_ERROR", None)
    return TestClient(srv.app)


class TestCapabilitiesShape:
    """The /api/capabilities payload is the contract that the JS
    helpers read. Lock the field names + values that G5.1 depends on."""

    def test_rocm_blocks_parakeet(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["parakeet"]["installed"] is True
        assert body["parakeet"]["available"] is False
        assert body["parakeet"]["blocked_by_backend"] is True

    def test_cuda_allows_parakeet(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["parakeet"]["available"] is True
        assert body["parakeet"]["blocked_by_backend"] is False

    def test_cpu_allows_parakeet(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CPU is unusably slow for Parakeet but it's not blocked at
        # the runtime level — the engine code path still loads. Keep
        # the option visible; the model-hint warns about speed.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["parakeet"]["blocked_by_backend"] is False

    def test_mps_blocks_parakeet(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Apple Silicon: NeMo doesn't run on MPS either. Server flags
        # blocked_by_backend=true; the optgroup stays visible (only
        # rocm gets the full hide treatment) but the model-hint will
        # tell the user.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["parakeet"]["blocked_by_backend"] is True
