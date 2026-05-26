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

    # G1.3: ROCm support-ticket details — gfx target + Linux distro.

    def test_reports_gfx_target_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: "gfx1100")
        out = _run(capsys)
        assert "GFX target:" in out
        assert "gfx1100" in out

    def test_omits_gfx_target_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA cards typically leave gcnArchName empty so gpu_arch_name()
        # returns None and the line should not appear.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "FakeGPU 1000")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: None)
        out = _run(capsys)
        assert "GFX target:" not in out

    def test_omits_gfx_target_on_rocm_when_unavailable(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the gcnArchName probe fails, devices.main() must still finish
        # cleanly and just skip the GFX target line.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: None)
        out = _run(capsys)
        assert "GFX target:" not in out
        # And the rest of the report still rendered.
        assert "Selected backends:" in out

    def test_reports_linux_distro_when_available(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        out = _run(capsys)
        assert "Distro:" in out
        assert "Ubuntu 24.04.4 LTS" in out

    def test_omits_distro_on_non_linux(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)
        out = _run(capsys)
        assert "Distro:" not in out


class TestLinuxDistro:
    """G1.3: ``_linux_distro()`` is the helper feeding the ``Distro:`` line."""

    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
        assert devices._linux_distro() is None

    def test_returns_none_when_helper_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Python < 3.10 paths simulated by deleting the platform attribute.
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.delattr(devices.platform, "freedesktop_os_release", raising=False)
        assert devices._linux_distro() is None

    def test_returns_pretty_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {"PRETTY_NAME": "Ubuntu 24.04.4 LTS", "NAME": "Ubuntu"},
            raising=False,
        )
        assert devices._linux_distro() == "Ubuntu 24.04.4 LTS"

    def test_falls_back_to_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No PRETTY_NAME field — fall back to NAME.
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {"NAME": "Fedora Linux"},
            raising=False,
        )
        assert devices._linux_distro() == "Fedora Linux"

    def test_returns_none_on_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # /etc/os-release missing → freedesktop_os_release raises OSError.
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")

        def _missing() -> dict:
            raise OSError("os-release not found")

        monkeypatch.setattr(
            devices.platform, "freedesktop_os_release", _missing, raising=False
        )
        assert devices._linux_distro() is None

    def test_returns_none_when_pretty_name_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both PRETTY_NAME and NAME absent → None.
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {"VERSION_ID": "24.04"},
            raising=False,
        )
        assert devices._linux_distro() is None


class TestCt2RocmPinReport:
    """G2.1: scribe.devices surfaces the CT2 ROCm wheel pin + drift."""

    def _stub_rocm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        installed: str | None,
        pinned: str = "4.7.2",
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX"
        )
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "pinned_ct2_rocm_version", lambda: pinned)
        monkeypatch.setattr(devices, "installed_ct2_version", lambda: installed)
        # ct2_drift_message is allowed to use the real implementation; the
        # tests below pass arguments through it via these stubs.

    def test_reports_pin_when_versions_match(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm(monkeypatch, installed="4.7.2", pinned="4.7.2")
        out = _run(capsys)
        assert "CT2 ROCm pin:" in out
        assert "v4.7.2" in out
        assert "installed: 4.7.2" in out
        # No drift line when versions match.
        assert "drift:" not in out

    def test_warns_on_version_drift(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm(monkeypatch, installed="4.6.0", pinned="4.7.2")
        out = _run(capsys)
        assert "CT2 ROCm pin:" in out
        assert "installed: 4.6.0" in out
        assert "drift:" in out
        assert "setup.sh --rocm" in out

    def test_warns_when_ctranslate2_missing(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm(monkeypatch, installed=None, pinned="4.7.2")
        out = _run(capsys)
        assert "CT2 ROCm pin:" in out
        assert "installed: not installed" in out
        assert "drift:" in out

    def test_pin_omitted_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CT2 build a CUDA user has installed is unrelated to the
        # ROCm pin — don't show the line on NVIDIA hardware.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "FakeGPU")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        out = _run(capsys)
        assert "CT2 ROCm pin:" not in out

    def test_pin_omitted_on_cpu(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        out = _run(capsys)
        assert "CT2 ROCm pin:" not in out


class TestCt2RocmFallbackUrlsReport:
    """G2.2: scribe.devices surfaces user-configured CT2 mirror fallbacks."""

    def _stub_rocm_minimal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same backbone as TestCt2RocmPinReport — keep it minimal here."""
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX"
        )
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(devices, "installed_ct2_version", lambda: "4.7.2")

    def test_no_mirror_line_when_env_unset(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.delenv("SCRIBE_CT2_ROCM_FALLBACK_URLS", raising=False)
        out = _run(capsys)
        assert "CT2 wheel mirrors" not in out

    def test_lists_configured_mirror_urls_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip,https://backup-mirror/b.zip",
        )
        out = _run(capsys)
        assert "CT2 wheel mirrors: 2 configured" in out
        assert "https://internal-mirror/a.zip" in out
        assert "https://backup-mirror/b.zip" in out

    def test_single_mirror_uses_correct_count(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.setenv("SCRIBE_CT2_ROCM_FALLBACK_URLS", "https://only/a.zip")
        out = _run(capsys)
        assert "CT2 wheel mirrors: 1 configured" in out
        assert "https://only/a.zip" in out

    def test_mirror_line_omitted_on_cuda_even_when_env_set(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fallback list is meaningless without the ROCm install path,
        # so we keep it inside the `backend == "rocm"` branch. Setting the
        # env var on a CUDA box shouldn't leak the line.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "FakeGPU")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS", "https://internal-mirror/a.zip"
        )
        out = _run(capsys)
        assert "CT2 wheel mirrors" not in out
        assert "internal-mirror" not in out

    def test_mirror_line_omitted_on_cpu_even_when_env_set(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS", "https://internal-mirror/a.zip"
        )
        out = _run(capsys)
        assert "CT2 wheel mirrors" not in out


class TestRocmDistroSupportTier:
    """G2.3: scribe.devices surfaces the distro support tier on ROCm only."""

    def _stub_rocm_minimal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 7900 XTX"
        )
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(devices, "installed_ct2_version", lambda: "4.7.2")

    def test_first_class_line_when_on_ubuntu_2404(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.setattr(
            devices, "_rocm_distro_tier",
            lambda: ("first-class", "Ubuntu 24.04.4 LTS"),
        )
        out = _run(capsys)
        assert "Distro support:" in out
        assert "first-class" in out
        assert "Ubuntu 24.04.4 LTS" in out

    def test_supported_line_on_rhel(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.setattr(
            devices, "_rocm_distro_tier",
            lambda: ("supported", "Red Hat Enterprise Linux 9.7 (Plow)"),
        )
        out = _run(capsys)
        assert "Distro support:" in out
        assert "supported" in out
        assert "Red Hat Enterprise Linux 9.7" in out

    def test_best_effort_line_on_fedora(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_rocm_minimal(monkeypatch)
        monkeypatch.setattr(
            devices, "_rocm_distro_tier",
            lambda: ("best-effort", "Fedora Linux 41"),
        )
        out = _run(capsys)
        assert "Distro support:" in out
        assert "best-effort" in out
        assert "Fedora Linux 41" in out

    def test_distro_tier_line_omitted_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA users don't care which distro tier they hit; only ROCm
        # users have to think about AMD's support matrix.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "get_device_name", lambda i: "FakeGPU")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        out = _run(capsys)
        assert "Distro support:" not in out

    def test_distro_tier_line_omitted_on_cpu(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        out = _run(capsys)
        assert "Distro support:" not in out


class TestOsReleaseInfo:
    """``_os_release_info()`` underlies both legacy distro pretty-name +
    G2.3 tier classification, so it deserves its own tests."""

    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
        assert devices._os_release_info() is None

    def test_returns_none_when_helper_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.delattr(devices.platform, "freedesktop_os_release", raising=False)
        assert devices._os_release_info() is None

    def test_returns_dict_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu"},
            raising=False,
        )
        info = devices._os_release_info()
        assert info == {"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu"}

    def test_returns_none_on_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")

        def _missing() -> dict:
            raise OSError("os-release missing")

        monkeypatch.setattr(
            devices.platform, "freedesktop_os_release", _missing, raising=False
        )
        assert devices._os_release_info() is None


class TestRocmDistroTier:
    """G2.3: ``_rocm_distro_tier()`` joins os-release detection + classification."""

    def test_returns_first_class_on_ubuntu_2404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {
                "ID": "ubuntu",
                "VERSION_ID": "24.04",
                "PRETTY_NAME": "Ubuntu 24.04.4 LTS",
            },
            raising=False,
        )
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "first-class"
        assert pretty == "Ubuntu 24.04.4 LTS"

    def test_returns_unsupported_on_macos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "unsupported"
        assert pretty is None

    def test_returns_unsupported_on_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Windows")
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "unsupported"
        assert pretty is None

    def test_returns_unsupported_when_os_release_unreadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Linux but os-release missing → tier_for_system handles it via
        # classify_os_release({}) → "unsupported".
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")

        def _missing() -> dict:
            raise OSError("no os-release")

        monkeypatch.setattr(
            devices.platform, "freedesktop_os_release", _missing, raising=False
        )
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "unsupported"
        assert pretty is None

    def test_falls_back_to_name_when_pretty_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            devices.platform,
            "freedesktop_os_release",
            lambda: {"ID": "fedora", "NAME": "Fedora Linux"},
            raising=False,
        )
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "best-effort"
        assert pretty == "Fedora Linux"
