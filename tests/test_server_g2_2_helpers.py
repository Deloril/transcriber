"""Verification of reachability for G2.2 — configurable CT2 ROCm wheel
fallback mirrors surfaced through the user-facing UI.

G2.2 (commit ``02586a3``) shipped the *mechanism* for fallback mirrors:

* ``scribe.rocm_install.rocm_wheel_fallback_urls()`` reads the
  ``SCRIBE_CT2_ROCM_FALLBACK_URLS`` env var (comma-separated) and returns
  the parsed list, dropping empty entries and trimming whitespace.
* ``scribe.rocm_install.rocm_wheel_zip_urls()`` returns the ordered
  ``[primary_github_url, *fallbacks]`` list ``setup.sh --rocm`` walks.
* ``setup.sh --rocm`` walks that ordered list with curl, falling
  through on each failure.
* ``python -m scribe.devices`` prints the configured mirrors under the
  ``CT2 wheel mirrors:`` header on ROCm only.

The original commit only surfaced the mirrors through the CLI (and via
``setup.sh``'s installer log). This iteration extends the chain so a
researcher who never opens a terminal sees the same mirror count on the
home page Recording details card. One new field lands on
``GET /api/capabilities``:

  * ``ct2_rocm_fallback_urls``  — list of configured fallback URLs on
                                  ROCm (empty list when none set);
                                  ``None`` on every non-ROCm backend
                                  so the JS tile cleanly distinguishes
                                  "no mirrors configured" from "the
                                  field doesn't apply on this backend".

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"+N mirror(s)"`` to the sub-line when the ROCm payload carries a
non-empty list — empty list and ``None`` both collapse the segment so
the tile stays clean on the happy path / non-ROCm backends.

Deeper unit coverage of the helpers themselves lives in
``tests/test_rocm_install.py`` (env parser, ordering, primary-first)
and ``tests/test_devices.py`` (CLI render). The JS-side tile shape is
pinned in ``tests/js/backend-stat-tile.test.mjs``. This file is the
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
# 1. The G2.2 helpers exist and return the documented shapes
# --------------------------------------------------------------------------- #


class TestG2_2HelpersExposeFallbackUrls:
    """Pin the public surface of the two helpers ``/api/capabilities``
    and ``python -m scribe.devices`` both call into."""

    def test_env_var_constant_is_canonical(self) -> None:
        from scribe.rocm_install import ROCM_FALLBACK_ENV_VAR

        # Pin the env var name so the docs / setup.sh / tests can't
        # silently drift apart.
        assert ROCM_FALLBACK_ENV_VAR == "SCRIBE_CT2_ROCM_FALLBACK_URLS"

    def test_fallback_urls_unset_returns_empty_list(self) -> None:
        from scribe.rocm_install import rocm_wheel_fallback_urls

        # Empty mapping → empty list (the "no fallbacks configured"
        # state is the common case).
        assert rocm_wheel_fallback_urls(env={}) == []

    def test_fallback_urls_parses_csv(self) -> None:
        from scribe.rocm_install import (
            ROCM_FALLBACK_ENV_VAR,
            rocm_wheel_fallback_urls,
        )

        env = {ROCM_FALLBACK_ENV_VAR: "https://m1/a.zip,https://m2/b.zip"}
        assert rocm_wheel_fallback_urls(env=env) == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_zip_urls_includes_primary_then_fallbacks(self) -> None:
        from scribe.rocm_install import (
            ROCM_FALLBACK_ENV_VAR,
            rocm_wheel_zip_urls,
        )

        env = {ROCM_FALLBACK_ENV_VAR: "https://internal-mirror/a.zip"}
        urls = rocm_wheel_zip_urls(env=env)
        # Primary always first — setup.sh expects this ordering.
        assert "github.com/OpenNMT/CTranslate2" in urls[0]
        assert urls[1] == "https://internal-mirror/a.zip"


# --------------------------------------------------------------------------- #
# 2. The /api/capabilities response carries the new field
# --------------------------------------------------------------------------- #


class TestG2_2ApiCapabilitiesCarriesFallbackUrls:
    """Pin the exact JSON shape the helpers/UI consume."""

    def _stub_rocm(
        self,
        srv,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force /api/capabilities into the ROCm branch with deterministic
        pin / installed pair so the response is fully deterministic."""
        from scribe import engine, devices, rocm_install

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

    def test_capabilities_carries_empty_list_on_rocm_with_no_env_var(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch)
        monkeypatch.delenv("SCRIBE_CT2_ROCM_FALLBACK_URLS", raising=False)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        # ROCm + no env var → empty list (not None — the field applies
        # here, just no values configured).
        assert body["gpu"]["ct2_rocm_fallback_urls"] == []

    def test_capabilities_carries_configured_mirrors_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip,https://backup-mirror/b.zip",
        )
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_fallback_urls"] == [
            "https://internal-mirror/a.zip",
            "https://backup-mirror/b.zip",
        ]

    def test_capabilities_strips_whitespace_and_drops_empty_entries(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch)
        # The parser must tolerate trailing commas + padding so a
        # researcher's slightly-imperfect env var still reaches the UI.
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            " https://m1/a.zip , , https://m2/b.zip ,",
        )
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_fallback_urls"] == [
            "https://m1/a.zip",
            "https://m2/b.zip",
        ]

    def test_capabilities_omits_fallback_field_value_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes get None — the field doesn't apply, the JS tile
        # will check Array.isArray() before reading length so the
        # distinction matters.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip",
        )

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        # Even though the env var is set, CUDA collapses the field to
        # None — the AMD-specific mirror list is unrelated to a CUDA
        # user's CT2 install.
        assert body["gpu"]["ct2_rocm_fallback_urls"] is None

    def test_capabilities_omits_fallback_field_value_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip",
        )

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["gpu"]["ct2_rocm_fallback_urls"] is None

    def test_capabilities_omits_fallback_field_value_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Debian GNU/Linux 12")
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip",
        )

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["gpu"]["ct2_rocm_fallback_urls"] is None

    def test_capabilities_payload_shape_includes_new_field(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The home page tile reads the gpu payload by key. Pin the full
        # key set so a future response-shape drift fails this test
        # loudly — this is the "additive contract extension" guard.
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch)
        monkeypatch.delenv("SCRIBE_CT2_ROCM_FALLBACK_URLS", raising=False)
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
        }

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If rocm_wheel_fallback_urls() raises (corrupt env, who knows),
        # the API must still respond — fall to an empty list rather
        # than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("env probe failed")

        monkeypatch.setattr(rocm_install, "rocm_wheel_fallback_urls", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Defensive fallback: empty list, never None on ROCm even if
        # the helper exploded — a present-but-empty list still tells
        # the UI the field applies on this backend.
        assert body["gpu"]["ct2_rocm_fallback_urls"] == []


# --------------------------------------------------------------------------- #
# 3. The CLI surface still prints the same mirrors (regression guard)
# --------------------------------------------------------------------------- #


class TestG2_2DevicesCliSurface:
    """``python -m scribe.devices`` continues to print the configured
    mirrors on ROCm. The wire-up to /api/capabilities reuses the same
    helper so this is a regression guard against the helper getting
    moved out from under the CLI."""

    def test_cli_lists_configured_mirrors_on_rocm(
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
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip,https://backup-mirror/b.zip",
        )

        devices_main()
        out = capsys.readouterr().out
        assert "CT2 wheel mirrors: 2 configured" in out
        assert "https://internal-mirror/a.zip" in out
        assert "https://backup-mirror/b.zip" in out


# --------------------------------------------------------------------------- #
# 4. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG2_2HomePageBackendTileMirrorsCount:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G2.2 added the
    ``ct2_rocm_fallback_urls`` field which the helper turns into a
    "+N mirror(s)" sub-line segment when the ROCm payload carries a
    non-empty list."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_ct2_rocm_fallback_urls_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.ct2_rocm_fallback_urls so the tile can show
        # "+N mirror(s)" on ROCm with mirrors set.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "ct2_rocm_fallback_urls" in text, (
            "backendStatTile() must read gpu.ct2_rocm_fallback_urls so "
            "the home page tile sub-line can show the mirror count "
            "on ROCm boxes with SCRIBE_CT2_ROCM_FALLBACK_URLS set."
        )
        # The defensive ``Array.isArray`` guard pins the contract:
        # CUDA / MPS / CPU return None and the tile must not blow up
        # treating None as an array.
        assert "Array.isArray" in text, (
            "backendStatTile() must guard the fallback-URL read with "
            "Array.isArray so non-ROCm None values don't blow up."
        )

    def test_index_renders_against_rocm_mirror_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries
        # ct2_rocm_fallback_urls, and the helper turns the list into
        # the sub-line segment.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://internal-mirror/a.zip,https://backup-mirror/b.zip",
        )

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the configured mirror list verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_fallback_urls"] == [
            "https://internal-mirror/a.zip",
            "https://backup-mirror/b.zip",
        ]


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — env var → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG2_2EndToEndChain:
    """Walk the full chain for each realistic scenario. If any link
    drops the mirror list, this fails."""

    def _stub_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scribe import engine, devices, rocm_install

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

    def test_air_gapped_chain_single_mirror(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic deployment: lab admin sets one internal mirror so
        # the fresh box can run setup.sh --rocm without GitHub access.
        # The home page tile must surface "+1 mirror" so the admin
        # can verify the env var landed without dropping to a terminal.
        srv, client, _ = server_env
        self._stub_rocm(monkeypatch)
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://lab-mirror.internal/ct2-rocm.zip",
        )

        body = client.get("/api/capabilities").json()
        urls = body["gpu"]["ct2_rocm_fallback_urls"]
        assert urls == ["https://lab-mirror.internal/ct2-rocm.zip"]
        assert len(urls) == 1

    def test_chain_with_no_mirrors_returns_empty_list_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default ROCm box, no mirror configured. API must distinguish
        # "ROCm + no mirrors" (empty list) from "non-ROCm" (None).
        srv, client, _ = server_env
        self._stub_rocm(monkeypatch)
        monkeypatch.delenv("SCRIBE_CT2_ROCM_FALLBACK_URLS", raising=False)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_fallback_urls"] == []
        # Other ROCm-only fields still populated — the new field is
        # additive, not a replacement.
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"

    def test_cuda_chain_does_not_carry_mirror_list(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA + env var set (e.g. ops set it as a default in the
        # shell, then the user happens to be on an NVIDIA box):
        # the field must be None so the JS knows it doesn't apply.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setenv(
            "SCRIBE_CT2_ROCM_FALLBACK_URLS",
            "https://lab-mirror.internal/ct2-rocm.zip",
        )

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["ct2_rocm_fallback_urls"] is None
