"""Tests for scribe.rocm_distro — distro support tier classification (G2.3)."""

from __future__ import annotations

import pytest

from scribe import rocm_distro
from scribe.rocm_distro import (
    SUPPORT_MATRIX,
    Tier,
    classify,
    classify_os_release,
    detected_tier_line,
    is_rocm_capable,
    normalise_id,
    normalise_version,
    support_matrix_lines,
    tier_explanation,
    tier_for_system,
)


class TestNormaliseId:
    def test_lowercases(self) -> None:
        assert normalise_id("Ubuntu") == "ubuntu"

    def test_strips_whitespace(self) -> None:
        assert normalise_id("  RHEL  ") == "rhel"

    def test_returns_empty_for_none(self) -> None:
        assert normalise_id(None) == ""

    def test_returns_empty_for_blank(self) -> None:
        assert normalise_id("") == ""
        assert normalise_id("   ") == ""


class TestNormaliseVersion:
    def test_lowercases_and_strips(self) -> None:
        assert normalise_version(" 24.04 ") == "24.04"

    def test_handles_none(self) -> None:
        assert normalise_version(None) == ""

    def test_keeps_dots(self) -> None:
        # Don't accidentally split version components.
        assert normalise_version("24.04.4") == "24.04.4"


class TestClassifyUbuntu:
    def test_ubuntu_2204_is_first_class(self) -> None:
        assert classify("ubuntu", "22.04") == "first-class"

    def test_ubuntu_2404_is_first_class(self) -> None:
        assert classify("ubuntu", "24.04") == "first-class"

    def test_ubuntu_24044_point_release_is_first_class(self) -> None:
        # /etc/os-release reports e.g. "24.04.4" — we match major.minor only.
        assert classify("ubuntu", "24.04.4") == "first-class"

    def test_ubuntu_2004_is_best_effort(self) -> None:
        # 20.04 is out of AMD's matrix — we don't refuse to run, but it's
        # not first-class either.
        assert classify("ubuntu", "20.04") == "best-effort"

    def test_ubuntu_2604_unknown_future_version_is_best_effort(self) -> None:
        assert classify("ubuntu", "26.04") == "best-effort"

    def test_ubuntu_no_version_is_best_effort(self) -> None:
        # ID=ubuntu but VERSION_ID missing — can't promote to first-class.
        assert classify("ubuntu", None) == "best-effort"

    def test_uppercase_id_normalised(self) -> None:
        assert classify("Ubuntu", "24.04") == "first-class"


class TestClassifyRhel:
    def test_rhel_9_is_supported(self) -> None:
        assert classify("rhel", "9.7") == "supported"

    def test_rhel_10_is_supported(self) -> None:
        assert classify("rhel", "10.1") == "supported"

    def test_rhel_8_is_best_effort(self) -> None:
        assert classify("rhel", "8.9") == "best-effort"

    def test_rhel_no_version_is_best_effort(self) -> None:
        assert classify("rhel", None) == "best-effort"

    def test_rhel_clones_not_supported_directly(self) -> None:
        # Rocky / Alma are best-effort, not supported, even though they
        # track RHEL — AMD's matrix names RHEL specifically.
        assert classify("rocky", "9.7") == "best-effort"
        assert classify("almalinux", "9.7") == "best-effort"


class TestClassifyBestEffort:
    @pytest.mark.parametrize(
        "distro_id",
        ["fedora", "arch", "debian", "opensuse", "manjaro",
         "endeavouros", "pop", "linuxmint", "elementary", "nixos",
         "alpine", "centos"],
    )
    def test_known_best_effort_distros(self, distro_id: str) -> None:
        assert classify(distro_id, "any") == "best-effort"


class TestClassifyUnknown:
    def test_empty_id_returns_unknown(self) -> None:
        assert classify("", None) == "unknown"

    def test_none_id_returns_unknown(self) -> None:
        assert classify(None, None) == "unknown"

    def test_unfamiliar_distro_returns_unknown(self) -> None:
        # Some niche distro we haven't classified.
        assert classify("voidlinux", "rolling") == "unknown"


class TestClassifyOsRelease:
    def test_empty_mapping_is_unsupported(self) -> None:
        # No os-release at all → not Linux (or at least not classifiable).
        assert classify_os_release({}) == "unsupported"
        assert classify_os_release(None) == "unsupported"

    def test_full_ubuntu_dict(self) -> None:
        info = {
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "PRETTY_NAME": "Ubuntu 24.04.4 LTS",
        }
        assert classify_os_release(info) == "first-class"

    def test_full_rhel_dict(self) -> None:
        info = {"ID": "rhel", "VERSION_ID": "9.7"}
        assert classify_os_release(info) == "supported"

    def test_id_like_falls_back_to_best_effort(self) -> None:
        # A derivative we don't list explicitly, but ID_LIKE covers it.
        info = {"ID": "neon", "ID_LIKE": "ubuntu", "VERSION_ID": "22.04"}
        # Neon isn't in our explicit best-effort set; ID_LIKE → ubuntu →
        # best-effort (not first-class — derivatives don't inherit the
        # cert).
        assert classify_os_release(info) == "best-effort"

    def test_id_like_with_multiple_tokens(self) -> None:
        # Pop!_OS reports "ID_LIKE=ubuntu debian" — first matching token
        # decides.
        info = {"ID": "fictional", "ID_LIKE": "unknownthing debian"}
        assert classify_os_release(info) == "best-effort"

    def test_id_like_unknown_returns_unknown(self) -> None:
        info = {"ID": "fictional", "ID_LIKE": "alsounknown"}
        assert classify_os_release(info) == "unknown"

    def test_explicit_classification_wins_over_id_like(self) -> None:
        # If the ID itself classifies, we shouldn't even consult ID_LIKE.
        info = {"ID": "ubuntu", "VERSION_ID": "24.04", "ID_LIKE": "debian"}
        assert classify_os_release(info) == "first-class"


class TestTierForSystem:
    def test_linux_routes_to_classifier(self) -> None:
        info = {"ID": "ubuntu", "VERSION_ID": "24.04"}
        assert tier_for_system("Linux", info) == "first-class"

    def test_darwin_is_unsupported(self) -> None:
        # macOS — no ROCm wheels.
        assert tier_for_system("Darwin", None) == "unsupported"

    def test_windows_is_unsupported(self) -> None:
        # Windows AMD is the Tier 3 Vulkan path, not ROCm wheels.
        assert tier_for_system("Windows", None) == "unsupported"

    def test_linux_without_os_release_is_unsupported(self) -> None:
        # If we can't read /etc/os-release at all, we treat it like a
        # distro we can't classify. The wheels are still Linux-only,
        # so unsupported is the honest answer.
        assert tier_for_system("Linux", None) == "unsupported"
        assert tier_for_system("Linux", {}) == "unsupported"

    def test_linux_case_insensitive(self) -> None:
        info = {"ID": "ubuntu", "VERSION_ID": "24.04"}
        # Some tools report lowercase 'linux'; tolerate both.
        assert tier_for_system("linux", info) == "first-class"


class TestTierExplanation:
    @pytest.mark.parametrize(
        "tier",
        ["first-class", "supported", "best-effort", "unsupported", "unknown"],
    )
    def test_returns_non_empty_string_for_every_tier(self, tier: Tier) -> None:
        msg = tier_explanation(tier)
        assert isinstance(msg, str)
        assert msg

    def test_first_class_mentions_official(self) -> None:
        assert "official" in tier_explanation("first-class").lower()

    def test_unsupported_mentions_linux_only(self) -> None:
        # Cue the user toward the actual blocker.
        assert "linux" in tier_explanation("unsupported").lower()


class TestIsRocmCapable:
    def test_first_class_is_capable(self) -> None:
        assert is_rocm_capable("first-class") is True

    def test_supported_is_capable(self) -> None:
        assert is_rocm_capable("supported") is True

    def test_best_effort_is_capable(self) -> None:
        # Best-effort still installs; we just can't promise it works.
        assert is_rocm_capable("best-effort") is True

    def test_unknown_is_capable(self) -> None:
        # Unknown distro on Linux is best-effort by another name.
        assert is_rocm_capable("unknown") is True

    def test_unsupported_is_not_capable(self) -> None:
        assert is_rocm_capable("unsupported") is False


class TestSupportMatrix:
    def test_matrix_covers_all_tiers_we_render(self) -> None:
        tiers = {entry[0] for entry in SUPPORT_MATRIX}
        # We render four tiers in the matrix; "unknown" is implied by
        # the absence of a row.
        assert tiers == {"first-class", "supported", "best-effort", "unsupported"}

    def test_first_class_lists_ubuntu_versions(self) -> None:
        first_class = [e for e in SUPPORT_MATRIX if e[0] == "first-class"][0]
        # Both Ubuntu LTS we promote.
        assert "22.04" in first_class[1]
        assert "24.04" in first_class[1]

    def test_supported_lists_rhel_majors(self) -> None:
        supported = [e for e in SUPPORT_MATRIX if e[0] == "supported"][0]
        assert "RHEL 9" in supported[1]
        assert "RHEL 10" in supported[1]

    def test_best_effort_lists_fedora_arch_debian(self) -> None:
        be = [e for e in SUPPORT_MATRIX if e[0] == "best-effort"][0]
        for distro in ("Fedora", "Arch", "Debian"):
            assert distro in be[1]

    def test_unsupported_lists_windows_and_macos(self) -> None:
        unsup = [e for e in SUPPORT_MATRIX if e[0] == "unsupported"][0]
        assert "Windows" in unsup[1]
        assert "macOS" in unsup[1] or "darwin" in unsup[1].lower()


class TestSupportMatrixLines:
    def test_returns_one_line_per_matrix_row(self) -> None:
        lines = support_matrix_lines()
        assert len(lines) == len(SUPPORT_MATRIX)

    def test_each_line_mentions_its_tier(self) -> None:
        lines = support_matrix_lines()
        for tier_label, _, _ in SUPPORT_MATRIX:
            assert any(tier_label in line for line in lines), (
                f"tier {tier_label!r} not surfaced in support_matrix_lines()"
            )


class TestDetectedTierLine:
    def test_renders_with_pretty_name(self) -> None:
        line = detected_tier_line("Ubuntu 24.04.4 LTS", "first-class")
        assert "first-class" in line
        assert "Ubuntu 24.04.4 LTS" in line
        assert "Distro support:" in line

    def test_renders_without_pretty_name(self) -> None:
        # Linux box where /etc/os-release was unreadable.
        line = detected_tier_line(None, "unknown")
        assert "unknown" in line
        assert "(distro not detected)" in line

    def test_includes_tier_explanation(self) -> None:
        line = detected_tier_line("Fedora Linux 41", "best-effort")
        assert "best-effort" in line
        # The explanation phrase should ride along.
        assert "AMD" in line or "matrix" in line


class TestRoundTripRealishOsRelease:
    """Sanity-check on real-ish os-release dicts collected from the wild."""

    def test_ubuntu_2404_pretty(self) -> None:
        info = {
            "PRETTY_NAME": "Ubuntu 24.04.4 LTS",
            "NAME": "Ubuntu",
            "VERSION_ID": "24.04",
            "VERSION": "24.04.4 LTS (Noble Numbat)",
            "VERSION_CODENAME": "noble",
            "ID": "ubuntu",
            "ID_LIKE": "debian",
        }
        assert classify_os_release(info) == "first-class"

    def test_rhel_97(self) -> None:
        info = {
            "NAME": "Red Hat Enterprise Linux",
            "VERSION": "9.7 (Plow)",
            "ID": "rhel",
            "ID_LIKE": "fedora",
            "VERSION_ID": "9.7",
            "PRETTY_NAME": "Red Hat Enterprise Linux 9.7 (Plow)",
        }
        assert classify_os_release(info) == "supported"

    def test_fedora_41(self) -> None:
        info = {
            "NAME": "Fedora Linux",
            "VERSION": "41 (Workstation Edition)",
            "ID": "fedora",
            "VERSION_ID": "41",
            "PRETTY_NAME": "Fedora Linux 41 (Workstation Edition)",
        }
        assert classify_os_release(info) == "best-effort"

    def test_arch_rolling(self) -> None:
        info = {
            "NAME": "Arch Linux",
            "PRETTY_NAME": "Arch Linux",
            "ID": "arch",
            "BUILD_ID": "rolling",
        }
        assert classify_os_release(info) == "best-effort"

    def test_pop_os_via_id_like(self) -> None:
        info = {
            "NAME": "Pop!_OS",
            "ID": "pop",
            "ID_LIKE": "ubuntu debian",
            "VERSION_ID": "22.04",
            "PRETTY_NAME": "Pop!_OS 22.04 LTS",
        }
        # Pop is in our explicit best-effort set, so it's classified there
        # directly without ID_LIKE walking. Either path lands here.
        assert classify_os_release(info) == "best-effort"
