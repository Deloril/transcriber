"""Tests for scribe.rocm_install — pinned CT2 ROCm wheel + drift detection."""

from __future__ import annotations

import re

import pytest

from scribe import rocm_install


class TestPinnedVersion:
    def test_constant_is_a_semver_string(self) -> None:
        # Some semver-shaped string. We don't lock the actual literal here
        # because bumping the pin is a routine maintenance operation.
        v = rocm_install.PINNED_CT2_ROCM_VERSION
        assert isinstance(v, str)
        assert v
        assert re.match(r"^\d+\.\d+(\.\d+)?$", v), f"unexpected pin: {v!r}"

    def test_helper_returns_constant(self) -> None:
        assert rocm_install.pinned_ct2_rocm_version() == rocm_install.PINNED_CT2_ROCM_VERSION


class TestRocmWheelZipUrl:
    def test_default_uses_pinned_version(self) -> None:
        url = rocm_install.rocm_wheel_zip_url()
        assert rocm_install.PINNED_CT2_ROCM_VERSION in url
        assert url.startswith("https://github.com/OpenNMT/CTranslate2/releases/download/")
        assert url.endswith("rocm-python-wheels-Linux.zip")

    def test_explicit_version_overrides_pin(self) -> None:
        url = rocm_install.rocm_wheel_zip_url("4.6.0")
        assert "v4.6.0" in url
        assert "rocm-python-wheels-Linux.zip" in url

    def test_url_uses_v_prefix_on_tag(self) -> None:
        # The release tag convention is "v<version>" — bare "<version>"
        # would 404. Guard against accidental template edits.
        url = rocm_install.rocm_wheel_zip_url("9.9.9")
        assert "/download/v9.9.9/" in url


class TestInstalledCt2Version:
    def test_returns_string_when_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rocm_install, "version", lambda name: "4.7.2")
        assert rocm_install.installed_ct2_version() == "4.7.2"

    def test_returns_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from importlib.metadata import PackageNotFoundError

        def _missing(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr(rocm_install, "version", _missing)
        assert rocm_install.installed_ct2_version() is None

    def test_returns_none_on_unexpected_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Defensive: a corrupted package metadata directory shouldn't crash
        # the device report.
        def _boom(name: str) -> str:
            raise RuntimeError("metadata catastrophe")

        monkeypatch.setattr(rocm_install, "version", _boom)
        assert rocm_install.installed_ct2_version() is None


class TestCt2DriftMessage:
    def test_returns_none_when_versions_match(self) -> None:
        msg = rocm_install.ct2_drift_message(installed="4.7.2", pinned="4.7.2")
        assert msg is None

    def test_warns_when_not_installed(self) -> None:
        msg = rocm_install.ct2_drift_message(installed=None, pinned="4.7.2")
        assert msg is not None
        assert "not found" in msg
        assert "4.7.2" in msg
        assert "setup.sh --rocm" in msg

    def test_warns_on_version_mismatch(self) -> None:
        msg = rocm_install.ct2_drift_message(installed="4.6.0", pinned="4.7.2")
        assert msg is not None
        assert "4.6.0" in msg
        assert "4.7.2" in msg
        assert "setup.sh --rocm" in msg

    def test_uses_pin_constant_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When called with no kwargs, the function should fall back to
        # PINNED_CT2_ROCM_VERSION + installed_ct2_version().
        monkeypatch.setattr(rocm_install, "PINNED_CT2_ROCM_VERSION", "9.9.9")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "1.2.3")
        msg = rocm_install.ct2_drift_message()
        assert msg is not None
        assert "1.2.3" in msg
        assert "9.9.9" in msg

    def test_pinned_kwarg_can_be_explicit(self) -> None:
        # Confirms the second injection point is honoured.
        msg = rocm_install.ct2_drift_message(installed="4.7.2", pinned="4.7.0")
        assert msg is not None
        assert "4.7.0" in msg
        assert "4.7.2" in msg
