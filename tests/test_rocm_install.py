"""Tests for scribe.rocm_install — pinned CT2 ROCm wheel + drift detection."""

from __future__ import annotations

import re

import pytest

from scribe import rocm_install
from scribe.rocm_install import ROCM_FALLBACK_ENV_VAR


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


# G2.2 — fallback-URL plumbing.


class TestSplitFallbackCsv:
    def test_empty_string_is_empty_list(self) -> None:
        assert rocm_install._split_fallback_csv("") == []

    def test_single_url_no_commas(self) -> None:
        assert rocm_install._split_fallback_csv("https://m.example/a.zip") == [
            "https://m.example/a.zip"
        ]

    def test_multiple_urls_split_in_order(self) -> None:
        assert rocm_install._split_fallback_csv(
            "https://m1.example/a.zip,https://m2.example/b.zip"
        ) == [
            "https://m1.example/a.zip",
            "https://m2.example/b.zip",
        ]

    def test_whitespace_around_entries_is_stripped(self) -> None:
        # Real-world env vars often gather whitespace through copy/paste.
        assert rocm_install._split_fallback_csv(
            "  https://m1/a.zip , https://m2/b.zip  "
        ) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_empty_entries_are_dropped(self) -> None:
        # Trailing comma + adjacent commas — both common, both harmless.
        assert rocm_install._split_fallback_csv(
            "https://m1/a.zip,,https://m2/b.zip,"
        ) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_only_whitespace_entries_are_dropped(self) -> None:
        assert rocm_install._split_fallback_csv("  ,  ,") == []


class TestRocmWheelFallbackUrls:
    def test_empty_when_env_unset(self) -> None:
        assert rocm_install.rocm_wheel_fallback_urls(env={}) == []

    def test_empty_when_env_blank(self) -> None:
        assert rocm_install.rocm_wheel_fallback_urls(env={ROCM_FALLBACK_ENV_VAR: ""}) == []

    def test_reads_configured_urls_in_order(self) -> None:
        env = {ROCM_FALLBACK_ENV_VAR: "https://m1/a.zip,https://m2/b.zip"}
        assert rocm_install.rocm_wheel_fallback_urls(env=env) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_strips_whitespace(self) -> None:
        env = {ROCM_FALLBACK_ENV_VAR: " https://m1/a.zip , https://m2/b.zip "}
        assert rocm_install.rocm_wheel_fallback_urls(env=env) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_drops_empty_entries(self) -> None:
        env = {ROCM_FALLBACK_ENV_VAR: "https://m1/a.zip,,https://m2/b.zip,"}
        assert rocm_install.rocm_wheel_fallback_urls(env=env) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_default_env_is_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without env=…, the helper should consult os.environ directly so
        # setup.sh and devices.py both pick up the user's config.
        monkeypatch.setenv(ROCM_FALLBACK_ENV_VAR, "https://from-os/a.zip")
        assert rocm_install.rocm_wheel_fallback_urls() == ["https://from-os/a.zip"]

    def test_default_env_returns_empty_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ROCM_FALLBACK_ENV_VAR, raising=False)
        assert rocm_install.rocm_wheel_fallback_urls() == []

    def test_env_var_constant_name_is_documented(self) -> None:
        # Sanity: the public symbol that setup.sh and the docs reference.
        # If this ever changes we need to bump setup.sh + the docs in lockstep.
        assert ROCM_FALLBACK_ENV_VAR == "SCRIBE_CT2_ROCM_FALLBACK_URLS"


class TestRocmWheelZipUrls:
    def test_returns_primary_only_when_no_fallbacks(self) -> None:
        urls = rocm_install.rocm_wheel_zip_urls(env={})
        assert urls == [rocm_install.rocm_wheel_zip_url()]

    def test_primary_then_fallbacks_in_order(self) -> None:
        env = {ROCM_FALLBACK_ENV_VAR: "https://m1/a.zip,https://m2/b.zip"}
        urls = rocm_install.rocm_wheel_zip_urls(env=env)
        assert urls[0] == rocm_install.rocm_wheel_zip_url()
        assert urls[1:] == ["https://m1/a.zip", "https://m2/b.zip"]

    def test_explicit_version_propagates_to_primary(self) -> None:
        env = {ROCM_FALLBACK_ENV_VAR: "https://m1/a.zip"}
        urls = rocm_install.rocm_wheel_zip_urls("4.6.0", env=env)
        assert "v4.6.0" in urls[0]
        assert urls[1] == "https://m1/a.zip"

    def test_default_version_uses_pin(self) -> None:
        urls = rocm_install.rocm_wheel_zip_urls(env={})
        assert rocm_install.PINNED_CT2_ROCM_VERSION in urls[0]

    def test_primary_is_always_first(self) -> None:
        # Even with fallbacks set, the primary stays at index 0 — setup.sh
        # tries it first and only falls through on failure.
        env = {ROCM_FALLBACK_ENV_VAR: "https://internal-mirror/a.zip"}
        urls = rocm_install.rocm_wheel_zip_urls(env=env)
        assert urls[0].startswith("https://github.com/OpenNMT/CTranslate2/")

    def test_default_env_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ROCM_FALLBACK_ENV_VAR, "https://from-os/a.zip")
        urls = rocm_install.rocm_wheel_zip_urls()
        assert urls[-1] == "https://from-os/a.zip"
