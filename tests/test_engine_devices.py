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

    def test_empty_hip_string_is_not_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An empty string is falsy and should be treated as "no ROCm".
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(engine.torch.version, "hip", "", raising=False)
        monkeypatch.setattr(engine.torch.version, "cuda", "12.4", raising=False)
        assert engine.gpu_backend() == "cuda"


class TestBackendBooleans:
    """G1.1: convenience boolean helpers around ``gpu_backend()``."""

    @pytest.mark.parametrize(
        "backend,expected",
        [
            ("cuda", {"cuda": True, "rocm": False, "mps": False, "gpu": True}),
            ("rocm", {"cuda": False, "rocm": True, "mps": False, "gpu": True}),
            ("mps",  {"cuda": False, "rocm": False, "mps": True, "gpu": True}),
            ("cpu",  {"cuda": False, "rocm": False, "mps": False, "gpu": False}),
        ],
    )
    def test_booleans_match_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        expected: dict[str, bool],
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        assert engine.is_cuda() is expected["cuda"]
        assert engine.is_rocm() is expected["rocm"]
        assert engine.is_mps() is expected["mps"]
        assert engine.has_gpu() is expected["gpu"]

    def test_is_rocm_honours_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # End-to-end check: SCRIBE_DEVICE flips through gpu_backend() into is_rocm().
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SCRIBE_DEVICE", "rocm")
        assert engine.is_rocm() is True
        assert engine.is_cuda() is False
        assert engine.has_gpu() is True


class TestGpuVendor:
    """G1.1: vendor mapping for vendor-aware UI / engine routing."""

    @pytest.mark.parametrize(
        "backend,vendor",
        [("cuda", "nvidia"), ("rocm", "amd"), ("mps", "apple"), ("cpu", None)],
    )
    def test_maps_backend_to_vendor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        vendor: str | None,
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        assert engine.gpu_vendor() == vendor


class TestGpuRuntimeVersion:
    """G1.1: surface HIP / CUDA runtime version for support tickets."""

    def test_returns_hip_on_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine.torch.version, "hip", "6.3.42131", raising=False)
        monkeypatch.setattr(engine.torch.version, "cuda", "12.4-compat", raising=False)
        assert engine.gpu_runtime_version() == "6.3.42131"

    def test_returns_cuda_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine.torch.version, "hip", None, raising=False)
        monkeypatch.setattr(engine.torch.version, "cuda", "12.4", raising=False)
        assert engine.gpu_runtime_version() == "12.4"

    def test_none_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        assert engine.gpu_runtime_version() is None

    def test_none_on_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # MPS has no equivalent runtime-version string we can surface here.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        assert engine.gpu_runtime_version() is None

    def test_empty_string_normalised_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine.torch.version, "hip", "", raising=False)
        assert engine.gpu_runtime_version() is None


class TestPackageReexports:
    """G1.1: detection helpers should be importable from the top-level
    ``scribe`` package, not just ``scribe.engine``. They're API."""

    def test_top_level_imports(self) -> None:
        import scribe

        assert scribe.gpu_backend is engine.gpu_backend
        assert scribe.gpu_vendor is engine.gpu_vendor
        assert scribe.gpu_runtime_version is engine.gpu_runtime_version
        assert scribe.is_rocm is engine.is_rocm
        assert scribe.is_cuda is engine.is_cuda
        assert scribe.is_mps is engine.is_mps
        assert scribe.has_gpu is engine.has_gpu

    def test_dunder_all_lists_them(self) -> None:
        import scribe

        for name in (
            "gpu_backend",
            "gpu_vendor",
            "gpu_runtime_version",
            "is_rocm",
            "is_cuda",
            "is_mps",
            "has_gpu",
        ):
            assert name in scribe.__all__


class TestTorchDevice:
    """G1.2: helpers return the honest 4-state user-facing label.
    Translation to the actual torch/CT2 device-arg string happens at the
    library boundary via :func:`_to_torch_device_arg`."""

    def test_rocm_returns_rocm_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # G1.2: helper exposes "rocm" honestly so the UI / logs / support
        # output don't claim the user has CUDA when they don't.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        assert engine._torch_device() == "rocm"

    def test_all_backends_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for b in ("cuda", "rocm", "mps", "cpu"):
            monkeypatch.setattr(engine, "gpu_backend", lambda b=b: b)
            assert engine._torch_device() == b


class TestDiarizationDevice:
    def test_default_uses_gpu_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        assert engine._diarization_device() == "cuda"
        # G1.2: the honest "rocm" label is preserved; translation to the
        # CUDA-namespace device arg happens at the call site, not here.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        assert engine._diarization_device() == "rocm"
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

    def test_force_rocm_returns_rocm_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # G1.2: forcing rocm via env var keeps the honest label; the call
        # site applies _to_torch_device_arg before handing to pyannote.
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "rocm")
        assert engine._diarization_device() == "rocm"

    def test_force_cpu_overrides_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "cpu")
        assert engine._diarization_device() == "cpu"


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

    def test_rocm_returns_rocm_label_with_compute_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # G1.2: returns the honest "rocm" label. The compute-type tiering
        # mirrors CUDA (CT2 ROCm wheel uses the same fp16 / int8_float16
        # kernels); translation to device="cuda" happens at the call site.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "rocm"
        assert compute == "float16"

    def test_rocm_low_vram_picks_int8_float16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 6.0)
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "rocm"
        assert compute == "int8_float16"

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

    def test_force_device_rocm_keeps_rocm_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # G1.2: SCRIBE_WHISPER_DEVICE=rocm yields the honest "rocm" label
        # so logs / UI / support output match what the user asked for.
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "rocm")
        dev, compute = engine._whisper_device_and_compute()
        assert dev == "rocm"
        assert compute == "float16"


class TestToTorchDeviceArg:
    """G1.2: translate the user-facing backend label to the literal device
    string the torch / CT2 / pyannote APIs accept."""

    def test_rocm_collapses_to_cuda(self) -> None:
        # PyTorch ROCm and CTranslate2 ROCm both shim onto the CUDA
        # namespace, so the actual device-arg string is "cuda".
        assert engine._to_torch_device_arg("rocm") == "cuda"

    @pytest.mark.parametrize("label", ["cuda", "mps", "cpu"])
    def test_other_labels_passthrough(self, label: str) -> None:
        assert engine._to_torch_device_arg(label) == label

    def test_idempotent_on_cuda(self) -> None:
        # Calling the translator on an already-translated value is safe.
        assert engine._to_torch_device_arg(engine._to_torch_device_arg("rocm")) == "cuda"


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
