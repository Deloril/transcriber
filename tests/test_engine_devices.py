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
        # Disable the gfx-target signal so the test isolates the device-name
        # path. The detector ORs the two signals; on a CPU-only / CUDA test
        # box ``gpu_arch_name()`` already returns None or a non-gfx string,
        # so behaviourally the name-only path is what fires for these inputs.
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: name)
        assert engine._is_rdna2() is True

    @pytest.mark.parametrize("name", [
        "NVIDIA RTX 1000 Ada Generation Laptop GPU",
        "AMD Radeon RX 7900 XTX",
        "AMD Radeon RX 9070",
        "",
    ])
    def test_not_detected(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: name)
        assert engine._is_rdna2() is False


class TestIsRdna2Public:
    """G4.1: public ``is_rdna2`` API + gfx-target-primary detection.

    The detection ORs two signals (gfx target + device name); these tests
    cover each path independently and verify that an authoritative gfx
    target short-circuits before the name string is consulted.
    """

    @pytest.mark.parametrize("gfx", [
        "gfx1030",  # Navi 21 — RX 6800/6900 XT, W6800
        "gfx1031",  # Navi 22 — RX 6700/6750 XT
        "gfx1032",  # Navi 23 — RX 6600/6650 XT
        "gfx1033",  # Van Gogh APU (Steam Deck)
        "gfx1034",  # Navi 24 — RX 6400/6500 XT
        "gfx1035",  # Rembrandt APU (Ryzen 6000-series mobile)
        "gfx1036",  # Rembrandt-R APU (Ryzen 7035-series mobile)
    ])
    def test_detects_via_gfx_target(
        self, monkeypatch: pytest.MonkeyPatch, gfx: str
    ) -> None:
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: gfx)
        # Force a name that would otherwise be a False — proves the gfx
        # target alone is enough to flip the detector to True.
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        assert engine.is_rdna2() is True

    @pytest.mark.parametrize("gfx", [
        "gfx1100",   # RDNA 3 — Navi 31 (RX 7900 XTX/XT)
        "gfx1101",   # RDNA 3 — Navi 32 (RX 7700/7800 XT)
        "gfx1102",   # RDNA 3 — Navi 33 (RX 7600)
        "gfx1200",   # RDNA 4 — Navi 44
        "gfx1201",   # RDNA 4 — Navi 48 (RX 9070 XT)
        "gfx1010",   # RDNA 1 — Navi 10 (Tier 3, not RDNA 2)
        "gfx900",    # Vega — not RDNA at all
        "gfx940",    # CDNA 3 — datacentre
    ])
    def test_does_not_detect_other_gfx(
        self, monkeypatch: pytest.MonkeyPatch, gfx: str
    ) -> None:
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: gfx)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        assert engine.is_rdna2() is False

    def test_authoritative_gfx_overrides_misleading_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the gfx target is unambiguously RDNA 3 (gfx1100), an OEM-
        # rebranded marketing name that *looks* like RDNA 2 must not flip
        # the detector. The gfx-target signal short-circuits.
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6900 OEM")
        assert engine.is_rdna2() is False

    def test_falls_back_to_name_when_gcnarchname_is_marketing_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NVIDIA's PyTorch build populates gcnArchName with the device's
        # marketing name, not a gfx target. We must not trust it; the name
        # path takes over.
        monkeypatch.setattr(
            engine, "gpu_arch_name", lambda: "NVIDIA RTX 1000 Ada Generation Laptop GPU"
        )
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800")
        # Detector should pick up "rx 6" via the name path.
        assert engine.is_rdna2() is True

    def test_returns_false_when_both_signals_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        assert engine.is_rdna2() is False

    def test_private_alias_delegates_to_public(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Backwards-compat: ``_is_rdna2`` must return the same value as
        # ``is_rdna2`` for the same inputs. Existing call sites + tests
        # rely on the underscore alias; new code prefers the public name.
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1031")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        assert engine._is_rdna2() == engine.is_rdna2() is True

        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        assert engine._is_rdna2() == engine.is_rdna2() is False


class TestGpuArchName:
    """G1.3: gfx target reporting for ROCm support tickets."""

    def _props(self, **kw: Any) -> Any:
        m = MagicMock()
        for k, v in kw.items():
            setattr(m, k, v)
        return m

    def test_returns_none_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        assert engine.gpu_arch_name() is None

    def test_returns_gfx_target_on_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            engine.torch.cuda,
            "get_device_properties",
            lambda i: self._props(gcnArchName="gfx1100"),
        )
        assert engine.gpu_arch_name() == "gfx1100"

    def test_strips_feature_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ROCm sometimes returns the gfx target with a feature-flag suffix:
        # "gfx1100:sramecc+:xnack-". Triage only cares about the bare target.
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            engine.torch.cuda,
            "get_device_properties",
            lambda i: self._props(gcnArchName="gfx1100:sramecc+:xnack-"),
        )
        assert engine.gpu_arch_name() == "gfx1100"

    def test_returns_none_when_property_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CUDA cards leave gcnArchName empty / absent.
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        # Build a props object without a gcnArchName attribute.
        class _Props:
            pass
        monkeypatch.setattr(
            engine.torch.cuda, "get_device_properties", lambda i: _Props()
        )
        assert engine.gpu_arch_name() is None

    def test_returns_none_when_property_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            engine.torch.cuda,
            "get_device_properties",
            lambda i: self._props(gcnArchName=""),
        )
        assert engine.gpu_arch_name() is None

    def test_returns_none_on_query_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)

        def _boom(i: int) -> Any:
            raise RuntimeError("driver fell over")

        monkeypatch.setattr(engine.torch.cuda, "get_device_properties", _boom)
        assert engine.gpu_arch_name() is None


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


class TestApplyRocmRuntimeWorkaroundsPublic:
    """G4.1: public ``apply_rocm_runtime_workarounds`` API.

    Mirrors the private-alias suite but exercises the documented entry
    point a worker process / subprocess would call before its CT2 import.
    """

    def test_public_function_exists(self) -> None:
        assert callable(engine.apply_rocm_runtime_workarounds)

    def test_public_function_sets_allocator_on_rdna2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine.apply_rocm_runtime_workarounds()
        import os as _os
        assert _os.environ["CT2_CUDA_ALLOCATOR"] == "cub_caching"

    def test_public_function_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Calling twice on a fresh env should still leave the allocator at
        # "cub_caching"; calling once when the env is already set must not
        # clobber the user's choice on the second invocation either.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine.apply_rocm_runtime_workarounds()
        engine.apply_rocm_runtime_workarounds()
        import os as _os
        assert _os.environ["CT2_CUDA_ALLOCATOR"] == "cub_caching"

    def test_public_function_safe_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine.apply_rocm_runtime_workarounds()
        import os as _os
        assert "CT2_CUDA_ALLOCATOR" not in _os.environ

    def test_public_function_skips_rdna3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RX 7000-series doesn't need the cub_caching allocator workaround,
        # so on a clean ROCm-RDNA-3 machine the env var must stay unset.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        engine.apply_rocm_runtime_workarounds()
        import os as _os
        assert "CT2_CUDA_ALLOCATOR" not in _os.environ


class TestNeedsHsaOverride:
    """G4.2: detector for ``HSA_OVERRIDE_GFX_VERSION=10.3.0``.

    The override has to be in the environment *before* the HIP runtime
    initialises, so we can't auto-apply it from Python (torch is already
    loaded by the time this code runs). The detector drives the
    informational hint surfaced by ``scribe.devices`` and ``setup.sh``.
    """

    def test_false_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False

    def test_false_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False

    def test_false_on_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False

    def test_false_on_rocm_gfx1030(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # gfx1030 (Navi 21, RX 6800/6900-series) is the canonical RDNA 2
        # target ROCm ships kernels for. It doesn't need the override.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False

    @pytest.mark.parametrize("gfx", [
        "gfx1031",  # Navi 22 — RX 6700/6750 XT
        "gfx1032",  # Navi 23 — RX 6600/6650 XT
        "gfx1033",  # Van Gogh APU (Steam Deck)
        "gfx1034",  # Navi 24 — RX 6400/6500 XT
        "gfx1035",  # Rembrandt APU (Ryzen 6000-series mobile)
        "gfx1036",  # Rembrandt-R APU (Ryzen 7035-series mobile)
    ])
    def test_true_on_non_gfx1030_rdna2(
        self, monkeypatch: pytest.MonkeyPatch, gfx: str
    ) -> None:
        # Every other RDNA 2 die needs HSA_OVERRIDE_GFX_VERSION=10.3.0
        # to map onto gfx1030 — that's the whole point of the override.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: gfx)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is True

    @pytest.mark.parametrize("gfx", [
        "gfx1100",  # RDNA 3 — Navi 31
        "gfx1101",  # RDNA 3 — Navi 32
        "gfx1102",  # RDNA 3 — Navi 33
        "gfx1200",  # RDNA 4 — Navi 44
        "gfx1201",  # RDNA 4 — Navi 48
        "gfx1010",  # RDNA 1 — Navi 10 (Tier 3, not RDNA 2)
        "gfx900",   # Vega — not RDNA at all
        "gfx940",   # CDNA 3 — datacentre
    ])
    def test_false_on_non_rdna2_rocm(
        self, monkeypatch: pytest.MonkeyPatch, gfx: str
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: gfx)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False

    def test_false_when_user_already_set_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the user has already exported the variable we must not
        # second-guess it: their value is authoritative even when it's
        # "wrong" by our recommendation.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1032")
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        assert engine.needs_hsa_override() is False

    def test_false_when_user_set_override_to_wrong_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even if the user picked a non-recommended value we treat it
        # as authoritative — they may know something about their setup
        # we don't. Documenting the recommendation is not the same as
        # forcing it.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1032")
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        assert engine.needs_hsa_override() is False

    def test_false_when_arch_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If we couldn't read a gfx target we don't know what we're
        # looking at — recommending an override blindly could break a
        # working setup. Cautious default is False.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.needs_hsa_override() is False


class TestRecommendedHsaOverrideValue:
    """G4.2: thin wrapper that returns ``"10.3.0"`` when needed, else None."""

    def test_returns_none_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.recommended_hsa_override_value() is None

    def test_returns_none_on_gfx1030(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.recommended_hsa_override_value() is None

    def test_returns_10_3_0_on_gfx1032(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1032")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        assert engine.recommended_hsa_override_value() == "10.3.0"

    def test_returns_none_when_user_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1032")
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        assert engine.recommended_hsa_override_value() is None

    def test_constant_matches_documented_value(self) -> None:
        # The constant feeds setup.sh and the devices report; a typo
        # would silently break the recommendation everywhere it's used.
        assert engine.HSA_OVERRIDE_RDNA2_VALUE == "10.3.0"


class TestPackageExports:
    """G4.1: ROCm helpers reachable from the top-level ``scribe`` namespace.

    Lets a worker-process boot script call ``from scribe import
    apply_rocm_runtime_workarounds`` *before* importing CT2, without having
    to know the engine module path.
    """

    def test_apply_rocm_runtime_workarounds_reexported(self) -> None:
        import scribe
        assert scribe.apply_rocm_runtime_workarounds is engine.apply_rocm_runtime_workarounds
        assert "apply_rocm_runtime_workarounds" in scribe.__all__

    def test_is_rdna2_reexported(self) -> None:
        import scribe
        assert scribe.is_rdna2 is engine.is_rdna2
        assert "is_rdna2" in scribe.__all__

    def test_needs_hsa_override_reexported(self) -> None:
        # G4.2: the detector and the recommended-value helper need to be
        # importable from the top level so a downstream installer or
        # diagnostic script can render a hint without poking at
        # ``scribe.engine`` directly.
        import scribe
        assert scribe.needs_hsa_override is engine.needs_hsa_override
        assert "needs_hsa_override" in scribe.__all__

    def test_recommended_hsa_override_value_reexported(self) -> None:
        import scribe
        assert (
            scribe.recommended_hsa_override_value
            is engine.recommended_hsa_override_value
        )
        assert "recommended_hsa_override_value" in scribe.__all__


class TestPatchPyannoteLstmDropout:
    def test_noop_on_non_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        # Build a fake module with a fake LSTM child that has dropout > 0.
        import torch.nn as nn
        lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=2, dropout=0.5)
        n = engine._patch_pyannote_lstm_dropout(lstm)
        assert lstm.dropout == 0.5  # untouched
        assert n == 0  # G3.1: returns count of patched modules

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
        n = engine._patch_pyannote_lstm_dropout(p)
        assert p.a.dropout == 0.0
        assert p.b.dropout == 0.0
        assert n == 2  # G3.1: counts the LSTMs that were actually changed

    def test_safe_when_pipeline_has_no_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the duck-typing fails, the function should swallow the error.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        n = engine._patch_pyannote_lstm_dropout(object())
        assert n == 0
        # No exception = pass.

    def test_walks_pyannote_pipeline_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """G3.1: pyannote.audio Pipelines are *not* nn.Module — they hold
        the segmentation/embedding sub-models in ``_segmentation`` /
        ``_embedding`` instance attributes. The patch must recurse into
        those attributes; otherwise the LSTMs go un-patched and the
        ROCm MIOpen bug still bites.
        """
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        class _SegModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.5)

        class _EmbModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.4)

        class _FakePipeline:
            """Mimics pyannote.audio.core.pipeline.Pipeline: not an nn.Module."""
            def __init__(self) -> None:
                self._segmentation = _SegModel()
                self._embedding = _EmbModel()
                self._scratch = "unrelated"

        fake = _FakePipeline()
        # Sanity: the pipeline is NOT an nn.Module — proves the old
        # ``pipeline.modules()`` walk would have missed both LSTMs.
        assert not isinstance(fake, nn.Module)
        n = engine._patch_pyannote_lstm_dropout(fake)
        assert n == 2
        assert fake._segmentation.lstm.dropout == 0.0
        assert fake._embedding.lstm.dropout == 0.0

    def test_walks_dicts_and_lists_of_sub_pipelines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some pyannote pipelines stash sub-pipelines in a ``_models`` dict
        or a list of inferences. The patch should descend into both."""
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        class _Inner(nn.Module):
            def __init__(self, dr: float) -> None:
                super().__init__()
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=dr)

        class _Outer:
            def __init__(self) -> None:
                self._models = {"seg": _Inner(0.5), "emb": _Inner(0.3)}
                self._inferences = [_Inner(0.2)]

        outer = _Outer()
        n = engine._patch_pyannote_lstm_dropout(outer)
        assert n == 3
        assert outer._models["seg"].lstm.dropout == 0.0
        assert outer._models["emb"].lstm.dropout == 0.0
        assert outer._inferences[0].lstm.dropout == 0.0

    def test_idempotent_on_already_patched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-running the patch on an already-zeroed pipeline should
        report 0 newly-patched modules (otherwise the count is meaningless
        for triage)."""
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.5)

        p = Parent()
        first = engine._patch_pyannote_lstm_dropout(p)
        second = engine._patch_pyannote_lstm_dropout(p)
        assert first == 1
        assert second == 0
        assert p.lstm.dropout == 0.0

    def test_handles_cycles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Don't recurse forever if a pipeline has a back-pointer to itself."""
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        class Cycle:
            def __init__(self) -> None:
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.5)
                self.self_ref: Any = None

        c = Cycle()
        c.self_ref = c  # cycle
        n = engine._patch_pyannote_lstm_dropout(c)
        assert n == 1  # found exactly one LSTM, didn't loop forever
        assert c.lstm.dropout == 0.0

    def test_does_not_invoke_property_descriptors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pyannote.audio.Pipeline defines @property accessors that *load*
        models lazily on access. Walking ``dir(obj)`` would trigger them.
        We must walk only ``__dict__`` (instance attributes) so a triage
        helper never has the side effect of force-loading an embedding net.
        """
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        side_effects: list[str] = []

        class WithProperty:
            def __init__(self) -> None:
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.5)

            @property
            def lazy_embedding(self) -> Any:  # pragma: no cover - test fails if hit
                side_effects.append("LOAD")
                return None

        w = WithProperty()
        engine._patch_pyannote_lstm_dropout(w)
        assert side_effects == []  # property never invoked
        assert w.lstm.dropout == 0.0

    def test_logs_patched_count_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Support tickets need to see whether the patch fired. Log to
        stderr so it appears alongside engine startup output."""
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        import torch.nn as nn

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(8, 8, num_layers=2, dropout=0.5)

        engine._patch_pyannote_lstm_dropout(Parent())
        captured = capsys.readouterr()
        assert "patched 1 pyannote LSTM dropout" in captured.err
        # Stays out of stdout (which is parsed for whisperx progress lines).
        assert "patched" not in captured.out

    def test_no_log_when_nothing_to_patch(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        engine._patch_pyannote_lstm_dropout(object())
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_public_alias_same_function(self) -> None:
        """G3.1: a non-underscore alias is exported so external callers
        (smoke-test, third-party integrations) don't reach into the
        private API."""
        assert engine.patch_pyannote_lstm_dropout is engine._patch_pyannote_lstm_dropout

    def test_public_alias_top_level_import(self) -> None:
        import scribe
        assert scribe.patch_pyannote_lstm_dropout is engine._patch_pyannote_lstm_dropout
        assert "patch_pyannote_lstm_dropout" in scribe.__all__


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
