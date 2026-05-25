"""Tests for scribe.devices — the version + backend report."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from scribe import devices, engine


def _run(capsys: pytest.CaptureFixture) -> str:
    rc = devices.main()
    assert rc == 0
    return capsys.readouterr().out


class TestDevicesMain:
    def test_runs_and_prints_sections(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force a deterministic backend so the output is predictable.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        out = _run(capsys)
        assert "OS:" in out
        assert "Python:" in out
        assert "PyTorch:" in out
        assert "GPU backend:        cpu" in out
        assert "Selected backends:" in out
        assert "Whisper (CTranslate2):" in out
        assert "Alignment (torch):" in out
        assert "Diarization (pyannote):" in out
        assert "Package versions:" in out

    def test_reports_cuda_version_when_present(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "FakeGPU 1000")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        out = _run(capsys)
        assert "GPU backend:        cuda" in out
        assert "CUDA build:" in out
        assert "12.4" in out
        assert "FakeGPU 1000" in out
        assert "24.0 GB" in out
        assert "HIP build:" not in out

    def test_reports_hip_version_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        out = _run(capsys)
        assert "GPU backend:        rocm" in out
        assert "HIP build:" in out
        assert "6.3" in out

    def test_selected_backends_show_honest_rocm_label(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # G1.2: the user-facing report prints the honest backend label
        # ("rocm") in the Selected backends section, not the CUDA-shim
        # device-arg string. PyTorch/CT2 still receive "cuda" at the
        # actual library call — that translation happens at the boundary,
        # not in the helpers we print from here.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        out = _run(capsys)
        # Header line still calls it out.
        assert "GPU backend:        rocm" in out
        # Selected backends now reflect the honest label too.
        assert "Whisper (CTranslate2):  device=rocm" in out
        assert "Alignment (torch):      device=rocm" in out
        assert "Diarization (pyannote): device=rocm" in out

    def test_rdna2_workaround_note(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 6800")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: True)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        out = _run(capsys)
        assert "RDNA 2 detected" in out
        assert "CT2_CUDA_ALLOCATOR=cub_caching" in out

    def test_handles_torch_get_device_name_failure(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)

        def _boom(i: int) -> str:
            raise RuntimeError("hardware on fire")

        monkeypatch.setattr(devices.torch.cuda, "get_device_name", _boom)
        out = _run(capsys)
        # The helper logs the error but doesn't crash main().
        assert "could not query device 0" in out

    def test_package_versions_section(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        # We don't lock the actual versions — just that the names are listed.
        out = _run(capsys)
        for name in ("whisperx", "ctranslate2", "transformers", "huggingface_hub", "pyannote.audio"):
            assert name in out

    def test_parakeet_section(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        out = _run(capsys)
        assert "Parakeet engine:" in out
