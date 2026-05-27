"""Verification of reachability for G2.3 — Linux distro support tier
(first-class / supported / best-effort / unknown) surfaced through the
user-facing UI.

G2.3 (commit ``d515da3``) shipped the *classifier*: ``scribe.rocm_distro``
with ``classify_os_release`` / ``tier_for_system`` / ``tier_explanation``,
plus :func:`scribe.devices._rocm_distro_tier` that wraps the platform
call. The original commit only surfaced the classification through the
CLI (``python -m scribe.devices`` prints the ``Distro support:`` line)
and through ``setup.sh --rocm`` (the installer warns on best-effort /
unknown). A researcher on Fedora or Arch who never opens a terminal had
no way to see whether their distro was in AMD's official ROCm support
matrix.

This iteration extends the chain so the home page Recording details
card carries the tier classification in the Backend tile sub-line.
Two new fields land on ``GET /api/capabilities``:

  * ``distro_tier``                — one of ``"first-class"`` /
                                     ``"supported"`` / ``"best-effort"`` /
                                     ``"unknown"`` on ROCm; ``None`` on
                                     every non-ROCm backend (a CUDA /
                                     MPS / CPU user doesn't care about
                                     AMD's distro matrix).
  * ``distro_tier_explanation``    — human-readable one-line rationale
                                     (e.g. "AMD officially supports this
                                     distro for ROCm; tested by
                                     Scribe"), populated alongside
                                     ``distro_tier`` and ``None`` on
                                     non-ROCm.

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"<tier> distro"`` to the sub-line when the ROCm payload carries a
truthy ``distro_tier`` — the bare tier label alone reads strangely in
a sub-line ending, so the helper appends " distro" so the segment
copies cleanly into a support thread.

Deeper unit coverage of the classifier itself lives in
``tests/test_rocm_distro.py`` (every tier transition, ID_LIKE walking,
real-ish os-release dicts) and ``tests/test_devices.py`` (CLI render).
The JS-side tile shape is pinned in
``tests/js/backend-stat-tile.test.mjs``. This file is the
integration-level reachability proof.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same fixture shape as the other ``test_server_g*_helpers.py``
    files. Copied so this file is self-contained."""
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


# --------------------------------------------------------------------------- #
# 1. The G2.3 helpers exist and return the documented shapes
# --------------------------------------------------------------------------- #


class TestG2_3HelpersExposeDistroTier:
    """Pin the public surface of the helpers ``/api/capabilities`` and
    ``python -m scribe.devices`` both call into."""

    def test_tier_labels_are_canonical(self) -> None:
        from scribe.rocm_distro import Tier  # noqa: F401  (typing alias)
        from scribe.rocm_distro import tier_explanation

        # Pin the four tier values + a sample of their explanations so
        # the docs / CLI / API / UI can't silently drift apart on
        # spelling. ``"unsupported"`` is also valid but only returned
        # for non-Linux systems — the API gates on backend == "rocm"
        # before classifying so it can't be reported via the UI.
        for label in ("first-class", "supported", "best-effort", "unknown"):
            text = tier_explanation(label)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_classify_recognises_first_class_ubuntu(self) -> None:
        from scribe.rocm_distro import classify

        assert classify("ubuntu", "24.04") == "first-class"
        assert classify("ubuntu", "22.04") == "first-class"

    def test_classify_recognises_supported_rhel(self) -> None:
        from scribe.rocm_distro import classify

        assert classify("rhel", "9") == "supported"
        assert classify("rhel", "10") == "supported"

    def test_classify_recognises_best_effort_distros(self) -> None:
        from scribe.rocm_distro import classify

        assert classify("fedora", "41") == "best-effort"
        assert classify("arch", None) == "best-effort"
        assert classify("debian", "12") == "best-effort"

    def test_devices_helper_returns_tier_and_pretty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``_rocm_distro_tier`` wraps the platform call so the API can
        # classify without re-reading os-release. Make sure the wrapper
        # returns a (tier, pretty) tuple with the documented shape.
        from scribe import devices

        monkeypatch.setattr(
            devices,
            "_os_release_info",
            lambda: {
                "ID": "ubuntu",
                "VERSION_ID": "24.04",
                "PRETTY_NAME": "Ubuntu 24.04.4 LTS",
            },
        )
        monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
        tier, pretty = devices._rocm_distro_tier()
        assert tier == "first-class"
        assert pretty == "Ubuntu 24.04.4 LTS"


# --------------------------------------------------------------------------- #
# 2. The /api/capabilities response carries the new fields
# --------------------------------------------------------------------------- #


def _stub_rocm_branch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pretty: str = "Ubuntu 24.04.4 LTS",
    os_release: dict[str, str] | None = None,
) -> None:
    """Force /api/capabilities into the ROCm branch with deterministic
    distro classification. ``os_release`` defaults to a first-class
    Ubuntu 24.04 mapping so the capabilities body is fully predictable.
    """
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
    monkeypatch.setattr(devices, "_linux_distro", lambda: pretty)
    monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
    monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
    info = os_release if os_release is not None else {
        "ID": "ubuntu",
        "VERSION_ID": "24.04",
        "PRETTY_NAME": pretty,
    }
    monkeypatch.setattr(devices, "_os_release_info", lambda: info)
    monkeypatch.setattr(devices.platform, "system", lambda: "Linux")


class TestG2_3ApiCapabilitiesCarriesDistroTier:
    """Pin the exact JSON shape the helpers/UI consume."""

    def test_first_class_ubuntu_lts_reports_first_class(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["distro_tier"] == "first-class"
        # Explanation is non-empty and references AMD's matrix so a
        # researcher pasting it into a support thread has the context.
        assert body["gpu"]["distro_tier_explanation"]
        assert "AMD" in body["gpu"]["distro_tier_explanation"]

    def test_rhel_9_reports_supported(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="Red Hat Enterprise Linux 9.7 (Plow)",
            os_release={
                "ID": "rhel",
                "VERSION_ID": "9.7",
                "PRETTY_NAME": "Red Hat Enterprise Linux 9.7 (Plow)",
            },
        )
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["distro_tier"] == "supported"

    def test_fedora_reports_best_effort(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="Fedora Linux 41 (Workstation Edition)",
            os_release={
                "ID": "fedora",
                "VERSION_ID": "41",
                "PRETTY_NAME": "Fedora Linux 41 (Workstation Edition)",
            },
        )
        body = client.get("/api/capabilities").json()
        # Fedora isn't in AMD's official matrix; still works in
        # practice via upstream packages → best-effort.
        assert body["gpu"]["distro_tier"] == "best-effort"

    def test_arch_reports_best_effort_without_version(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="Arch Linux",
            os_release={
                "ID": "arch",
                "PRETTY_NAME": "Arch Linux",
            },
        )
        body = client.get("/api/capabilities").json()
        # Rolling-release Arch reports no VERSION_ID; classify() must
        # still land it on best-effort.
        assert body["gpu"]["distro_tier"] == "best-effort"

    def test_unrecognised_distro_reports_unknown(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="Some Custom Linux 1.0",
            os_release={
                "ID": "customlinux",
                "VERSION_ID": "1.0",
                "PRETTY_NAME": "Some Custom Linux 1.0",
            },
        )
        body = client.get("/api/capabilities").json()
        # No ID match, no ID_LIKE → "unknown" (still a Linux box, we
        # just couldn't classify it). The UI renders this as
        # "unknown distro" so the researcher sees the gap.
        assert body["gpu"]["distro_tier"] == "unknown"

    def test_capabilities_omits_tier_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes get None — the field doesn't apply (they're not
        # running an AMD-specific install path). The JS tile collapses
        # the segment when the field is null.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        # Even though the box could be on Ubuntu 24.04 (a first-class
        # ROCm distro), the field collapses to None — a CUDA user
        # doesn't care about the AMD matrix.
        assert body["gpu"]["distro_tier"] is None
        assert body["gpu"]["distro_tier_explanation"] is None

    def test_capabilities_omits_tier_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["gpu"]["distro_tier"] is None
        assert body["gpu"]["distro_tier_explanation"] is None

    def test_capabilities_omits_tier_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Debian GNU/Linux 12")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        # CPU on a Debian box could imagine itself as best-effort if
        # ROCm wheels were ever installed, but the field is gated on
        # backend == "rocm". On CPU it stays None.
        assert body["gpu"]["distro_tier"] is None
        assert body["gpu"]["distro_tier_explanation"] is None

    def test_capabilities_swallows_classifier_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If ``_rocm_distro_tier`` raises (corrupt /etc/os-release,
        # locked-down container, etc.) the API must still respond —
        # fall to None rather than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

        # Patch the symbol the route imports — note ``server`` re-exports
        # ``_rocm_distro_tier`` at the import-site (top of the function),
        # so we patch ``devices._rocm_distro_tier`` which is the source
        # of truth.
        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("os-release probe failed")

        monkeypatch.setattr(devices, "_rocm_distro_tier", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Defensive fallback: None, not a partial value.
        assert body["gpu"]["distro_tier"] is None
        assert body["gpu"]["distro_tier_explanation"] is None
        # The other ROCm-only fields still populate — the new fields
        # are additive and don't break the rest of the surface.
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"

    def test_capabilities_payload_shape_includes_new_fields(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The home page tile reads the gpu payload by key. Pin the full
        # key set so a future response-shape drift fails this test
        # loudly — additive contract extension guard.
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert set(body["gpu"].keys()) == {
            "backend",
            "device_name",
            "vram_gb",
            "gfx_target",
            "distro",
            "ct2_rocm_pin",
            "ct2_installed",
            "ct2_drift_message",
            "ct2_rocm_fallback_urls",
            "distro_tier",
            "distro_tier_explanation",
            "rocm_lstm_patch",
            "rocm_lstm_patch_explanation",
            "rocm_allocator_state",
            "rocm_allocator_value",
            "rocm_allocator_explanation",
            "rocm_hsa_override_state",
            "rocm_hsa_override_value",
            "rocm_hsa_override_explanation",
            "whisper_compute_type",
        }


# --------------------------------------------------------------------------- #
# 3. The CLI surface still prints the same Distro support line
# --------------------------------------------------------------------------- #


class TestG2_3DevicesCliSurface:
    """``python -m scribe.devices`` continues to print the
    ``Distro support:`` line on ROCm. The wire-up to /api/capabilities
    reuses the same ``_rocm_distro_tier`` helper so this is a
    regression guard against the helper getting moved out from under
    the CLI."""

    def test_cli_prints_distro_support_line_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine
        from scribe.devices import main as devices_main

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
        monkeypatch.setattr(
            devices,
            "_rocm_distro_tier",
            lambda: ("first-class", "Ubuntu 24.04.4 LTS"),
        )
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        devices_main()
        out = capsys.readouterr().out
        assert "Distro support:" in out
        assert "first-class" in out
        # The CLI line includes the explanation too.
        assert "AMD officially supports" in out


# --------------------------------------------------------------------------- #
# 4. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG2_3HomePageBackendTileShowsDistroTier:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G2.3 added the
    ``distro_tier`` field which the helper turns into a
    "<tier> distro" sub-line segment when the ROCm payload carries
    a truthy value."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_distro_tier_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.distro_tier so the tile can show "<tier> distro" on
        # ROCm boxes.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "distro_tier" in text, (
            "backendStatTile() must read gpu.distro_tier so the home "
            "page tile sub-line can show the AMD ROCm distro support "
            "tier next to the distro pretty-name."
        )

    def test_index_renders_against_rocm_first_class_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries distro_tier, and the
        # helper turns the value into the sub-line segment.
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the tier verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["distro_tier"] == "first-class"
        assert body["gpu"]["distro_tier_explanation"]


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — distro probe → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG2_3EndToEndChain:
    """Walk the full chain for each realistic scenario. If any link
    drops the tier classification, this fails."""

    def test_first_class_chain_ubuntu_lts(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic happy path: RX 7900 XTX on Ubuntu 24.04, AMD's
        # primary supported config. Tile sub-line should show
        # "first-class distro".
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["distro"] == "Ubuntu 24.04.4 LTS"
        assert gpu["distro_tier"] == "first-class"
        # The four-key tile shape is preserved (additive contract).
        assert "first-class" in gpu["distro_tier"]

    def test_best_effort_chain_arch_user(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic edge case: Arch user, rolling release, no
        # VERSION_ID. Tile sub-line should show "best-effort distro"
        # so the user knows they're on AMD's unofficial path.
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="Arch Linux",
            os_release={"ID": "arch", "PRETTY_NAME": "Arch Linux"},
        )
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["distro"] == "Arch Linux"
        assert gpu["distro_tier"] == "best-effort"

    def test_unknown_chain_carries_explanation_for_support_ticket(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic worst case for triage: fully unrecognised distro.
        # The explanation prompts the user to identify their ID so
        # support has somewhere to start.
        srv, client, _ = server_env
        _stub_rocm_branch(
            monkeypatch,
            pretty="ExoticOS 0.1",
            os_release={
                "ID": "exoticos",
                "VERSION_ID": "0.1",
                "PRETTY_NAME": "ExoticOS 0.1",
            },
        )
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["distro_tier"] == "unknown"
        # Explanation is something a researcher can copy into a bug
        # report verbatim.
        assert isinstance(gpu["distro_tier_explanation"], str)
        assert len(gpu["distro_tier_explanation"]) > 10

    def test_cuda_chain_does_not_carry_tier(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA box on Ubuntu 24.04 (which would be first-class on
        # ROCm): the field collapses to None so the JS knows it
        # doesn't apply on this backend.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["distro_tier"] is None
        assert body["gpu"]["distro_tier_explanation"] is None
