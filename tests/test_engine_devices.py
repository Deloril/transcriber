"""Tests for device-selection helpers in scribe.engine."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scribe import engine


class TestGpuBackend:
    def test_cpu_when_no_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine.torch.backends.mps, "is_available", lambda: False)
        assert engine.gpu_backend() == "cpu"

    def test_mps_picked_when_only_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine.torch.backends.mps, "is_available", lambda: True)
        assert engine.gpu_backend() == "mps"

    def test_cuda_when_cuda_no_hip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(engine.torch.version, "hip", None, raising=False)
        monkeypatch.setattr(engine.torch.version, "cuda", "12.4", raising=False)
        assert engine.gpu_backend() == "cuda"

    def test_rocm_when_hip_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(engine.torch.version, "hip", "6.3", raising=False)
        # PyTorch ROCm wheels populate torch.version.cuda too — make sure we
        # don't get confused.
        monkeypatch.setattr(engine.torch.version, "cuda", "12.4-compat", raising=False)
        assert engine.gpu_backend() == "rocm"

    def test_env_override_forces_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(engine.torch.version, "hip", None, raising=False)
        monkeypatch.setenv("SCRIBE_DEVICE", "cpu")
        assert engine.gpu_backend() == "cpu"

    def test_env_override_forces_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SCRIBE_DEVICE", "rocm")
        assert engine.gpu_backend() == "rocm"

    def test_env_override_invalid_value_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(engine.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setenv("SCRIBE_DEVICE", "tpu")  # not in {cuda, rocm, mps, cpu}
        assert engine.gpu_backend() == "cpu"


class TestTorchDevice:
    def test_rocm_translates_to_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # PyTorch ROCm uses the cuda namespace, so the actual torch device is "cuda".
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        assert engine._torch_device() == "cuda"

    def test_other_backends_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for b in ("cuda", "mps", "cpu"):
            monkeypatch.setattr(engine, "gpu_backend", lambda b=b: b)
            assert engine._torch_device() == b


class TestDiarizationDevice:
    def test_default_uses_gpu_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        assert engine._diarization_device() == "cuda"
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        assert engine._diarization_device() == "cuda"  # rocm → cuda namespace
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        assert engine._diarization_device() == "cpu"

    def test_mps_falls_back_to_cpu_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # pyannote on MPS is partial; default avoids it.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        assert engine._diarization_device() == "cpu"

    def test_force_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "mps")
        assert engine._diarization_device() == "mps"

    def test_force_rocm_yields_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "rocm")
        assert engine._diarization_device() == "cuda"


class TestWhisperDeviceAndCompute:
    def test_cpu_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "cpu"
        assert compute == "int8"

    def test_cuda_high_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "cuda"
        assert compute == "float16"

    def test_cuda_low_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 6.0)
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "cuda"
        assert compute == "int8_float16"

    def test_rocm_uses_cuda_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CT2's ROCm wheel takes device="cuda" via its HIP shim.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "cuda"
        assert compute == "float16"

    def test_force_compute_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "float32")
        dev, compute = engine._whisper_device_and_compute()
        assert compute == "float32"

    def test_force_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "cpu")
        dev, _ = engine._whisper_device_and_compute()
        assert dev == "cpu"

    def test_force_device_rocm_translates_to_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "rocm")
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "cuda"


class TestIsRdna2:
    @pytest.mark.parametrize("name", [
        "AMD Radeon RX 6800 XT",
        "AMD Radeon RX 6600",
        "Navi 21",
        "AMD gfx1031 device",
    ])
    def test_detects(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: name)
        assert engine._is_rdna2() is True

    @pytest.mark.parametrize("name", [
        "NVIDIA RTX 1000 Ada Generation Laptop GPU",
        "AMD Radeon RX 7900 XTX",
        "AMD Radeon RX 9070",
        "",
    ])
    def test_not_detected(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: name)
        assert engine._is_rdna2() is False


class TestApplyRocmRuntimeWorkarounds:
    def test_noop_on_non_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine._apply_rocm_runtime_workarounds()
        assert "CT2_CUDA_ALLOCATOR" not in __import__("os").environ

    def test_sets_allocator_on_rdna2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine._apply_rocm_runtime_workarounds()
        import os as _os
        assert _os.environ["CT2_CUDA_ALLOCATOR"] == "cub_caching"

    def test_respects_user_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the user already set the allocator, we don't clobber it.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "MallocAsync")
        engine._apply_rocm_runtime_workarounds()
        import os as _os
        assert _os.environ["CT2_CUDA_ALLOCATOR"] == "MallocAsync"

    def test_skips_non_rdna2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine._apply_rocm_runtime_workarounds()
        import os as _os
        assert "CT2_CUDA_ALLOCATOR" not in _os.environ


class TestPatchPyannoteLstmDropout:
    def test_noop_on_non_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        # Build a fake module with a fake LSTM child that has dropout > 0.
        import torch.nn as nn
        lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=2, dropout=0.5)
        engine._patch_pyannote_lstm_dropout(lstm)
        assert lstm.dropout == 0.5  # untouched

    def test_zeroes_dropout_on_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        # Wrap two LSTMs in a parent module so we exercise the .modules() walk.
        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.LSTM(8, 8, num_layers=2, dropout=0.5)
                self.b = nn.LSTM(8, 8, num_layers=3, dropout=0.3)

        p = Parent()
        engine._patch_pyannote_lstm_dropout(p)
        assert p.a.dropout == 0.0
        assert p.b.dropout == 0.0

    def test_safe_when_pipeline_has_no_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the duck-typing fails, the function should swallow the error.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        engine._patch_pyannote_lstm_dropout(object())
        # No exception = pass.


class TestProgressCapture:
    def test_parses_whisperx_progress_lines(self) -> None:
        seen: list[tuple[str, float]] = []
        cap = engine._ProgressCapture("xx", lambda label, pct: seen.append((label, pct)))
        cap.write("Progress: 12.50%...\n")
        cap.write("Progress: 50.00%...\n")
        cap.write("Progress: 100.00%...\n")
        assert seen == [("xx", 0.125), ("xx", 0.5), ("xx", 1.0)]

    def test_passes_other_lines_to_real_stdout(self, capsys: pytest.CaptureFixture) -> None:
        seen: list[tuple[str, float]] = []
        cap = engine._ProgressCapture("xx", lambda label, pct: seen.append((label, pct)))
        cap.write("Some unrelated log line\n")
        cap.flush()
        out = capsys.readouterr().out
        assert "Some unrelated log line" in out
        assert seen == []

    def test_buffers_partial_lines(self) -> None:
        seen: list[tuple[str, float]] = []
        cap = engine._ProgressCapture("xx", lambda label, pct: seen.append((label, pct)))
        # A "print" tends to deliver content in two writes (text, then "\n").
        cap.write("Progress: 25.00%...")
        assert seen == []
        cap.write("\n")
        assert seen == [("xx", 0.25)]

    def test_clamps_percentage(self) -> None:
        seen: list[tuple[str, float]] = []
        cap = engine._ProgressCapture("xx", lambda label, pct: seen.append((label, pct)))
        cap.write("Progress: 250.00%...\n")
        assert seen == [("xx", 1.0)]

    def test_context_manager_restores_stdout(self) -> None:
        import sys as _sys
        before = _sys.stdout
        with engine._ProgressCapture("xx", lambda *a: None):
            assert _sys.stdout is not before
        assert _sys.stdout is before


class TestSafeLoadModel:
    def test_passes_kwargs_when_supported(self) -> None:
        whisperx = MagicMock()

        def fake_load_model(model_name, *, device, compute_type, language=None, asr_options=None, vad_options=None):
            return ("model", model_name, asr_options, vad_options)

        whisperx.load_model = fake_load_model
        out = engine._safe_load_model(
            whisperx,
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            asr_options={"beam_size": 5},
            vad_options={"chunk_size": 30},
        )
        assert out[1] == "tiny"
        assert out[2] == {"beam_size": 5}
        assert out[3] == {"chunk_size": 30}

    def test_drops_vad_options_if_unsupported(self) -> None:
        whisperx = MagicMock()

        def fake_load_model(model_name, *, device, compute_type, language=None, asr_options=None):
            return ("model", asr_options)

        whisperx.load_model = fake_load_model
        out = engine._safe_load_model(
            whisperx,
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            asr_options={"beam_size": 5},
            vad_options={"chunk_size": 30},
        )
        assert out[1] == {"beam_size": 5}

    def test_retries_without_hotwords_on_typeerror(self) -> None:
        # Some older faster-whisper versions reject hotwords inside asr_options.
        # The shim should retry once with hotwords stripped.
        whisperx = MagicMock()
        calls: list[dict[str, Any]] = []

        def fake_load_model(model_name, *, device, compute_type, language=None, asr_options=None, vad_options=None):
            calls.append(dict(asr_options or {}))
            if "hotwords" in (asr_options or {}):
                raise TypeError("unexpected keyword argument 'hotwords'")
            return ("ok",)

        whisperx.load_model = fake_load_model
        out = engine._safe_load_model(
            whisperx,
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            asr_options={"beam_size": 5, "hotwords": "Salesforce", "initial_prompt": None},
            vad_options={"chunk_size": 30},
        )
        assert out == ("ok",)
        # First call had hotwords, retry stripped it.
        assert "hotwords" in calls[0]
        assert "hotwords" not in calls[1]
        # None-valued keys are also dropped on retry.
        assert "initial_prompt" not in calls[1]
