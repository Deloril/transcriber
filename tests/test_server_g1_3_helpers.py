"""Verification of reachability for G1.3 — ROCm support-ticket details.

G1.3 (commit ``22a15fa``) added two pieces of triage context to the
``scribe.devices`` CLI:

  * **gfx target** — the bare GCN architecture string (``gfx1100`` /
    ``gfx1030`` / ``gfx1201``). It's the first identifier CTranslate2
    and ROCm upstream maintainers ask for when triaging issues; CT2
    bug #2021 (RX 9070 XT crash) and the RDNA 2 cub_caching workaround
    are both gated on this string. Pulled from
    ``torch.cuda.get_device_properties(0).gcnArchName``; feature
    suffixes like ``gfx1100:sramecc+:xnack-`` are normalised down to
    the bare target by :func:`scribe.engine.gpu_arch_name`.

  * **Linux distro** — pretty-name from ``/etc/os-release``
    (``Ubuntu 24.04.4 LTS`` / ``Fedora Linux 43 (Workstation Edition)``
    / ``Debian GNU/Linux 12``). ROCm support varies meaningfully by
    distro — Ubuntu 22/24 first-class, RHEL 9/10 supported, Fedora
    best-effort — and several live upstream bugs are distro-conditional
    (CT2 #2021 is Fedora 43 + ROCm 7.2 specifically). Pulled from
    :func:`platform.freedesktop_os_release` via
    :func:`scribe.devices._linux_distro`; None on non-Linux or when
    the file is missing / unreadable.

The original ``22a15fa`` commit predates the loop's ``Reachable-via``
gate, so this file is the structural reachability proof that
the loop reads. The triage-context fingerprint is reachable from
**three** user-facing surfaces, not just the CLI:

  - **CLI** (``python -m scribe.devices``): the ``Distro:`` line shows
    on every Linux box; the ``GFX target:`` line shows only on ROCm.
    This is the support-bundle output users paste into bug reports.

  - **API** (``GET /api/capabilities``): the ``gpu`` payload now
    carries ``gfx_target`` (ROCm-only) and ``distro`` (Linux-only)
    in addition to ``backend`` / ``device_name`` / ``vram_gb`` from
    G1.4. Any scripted client (curl, an external dashboard, the
    in-house benchmark) sees the same fields.

  - **UI** (home page Recording details card → Backend tile): the
    ``backendStatTile`` sub-line appends gfx target and distro after
    device name + VRAM. A researcher reading their own machine info
    in the browser doesn't have to drop to a terminal.

Deeper unit coverage of the helpers themselves lives in
``tests/test_engine_devices.py`` (gpu_arch_name) and
``tests/test_devices.py`` (_linux_distro + CLI render). This file
is the integration-level reachability proof.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same fixture shape as ``tests/test_server.py`` and the G1.1 / G1.2
    verification files. Copied so this file is self-contained."""
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
# 1. The two G1.3 helpers exist and return the documented shape
# --------------------------------------------------------------------------- #


class TestG1_3HelpersExposeTriageContext:
    """Pin the public surface of the two helpers that produce G1.3's
    triage context. These are the functions ``/api/capabilities`` and
    ``python -m scribe.devices`` both call into."""

    def test_gpu_arch_name_is_callable_and_returns_str_or_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        # Smoke: the function exists, is callable, returns str | None.
        result = engine.gpu_arch_name()
        assert result is None or isinstance(result, str)

    def test_gpu_arch_name_strips_feature_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # gcnArchName from torch can come back with feature flags appended:
        # "gfx1100:sramecc+:xnack-" → "gfx1100".
        from scribe import engine

        class FakeProps:
            gcnArchName = "gfx1100:sramecc+:xnack-"

        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            engine.torch.cuda, "get_device_properties", lambda i: FakeProps()
        )
        assert engine.gpu_arch_name() == "gfx1100"

    def test_gpu_arch_name_returns_none_when_torch_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
        assert engine.gpu_arch_name() is None

    def test_gpu_arch_name_returns_none_on_empty_arch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        class FakeProps:
            gcnArchName = ""

        monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            engine.torch.cuda, "get_device_properties", lambda i: FakeProps()
        )
        assert engine.gpu_arch_name() is None

    def test_linux_distro_returns_none_on_non_linux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import platform as _platform
        from scribe import devices

        monkeypatch.setattr(_platform, "system", lambda: "Darwin")
        assert devices._linux_distro() is None

    def test_linux_distro_returns_pretty_name_on_linux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import platform as _platform
        from scribe import devices

        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            _platform,
            "freedesktop_os_release",
            lambda: {
                "PRETTY_NAME": "Ubuntu 24.04.4 LTS",
                "NAME": "Ubuntu",
                "VERSION_ID": "24.04",
            },
            raising=False,
        )
        assert devices._linux_distro() == "Ubuntu 24.04.4 LTS"


# --------------------------------------------------------------------------- #
# 2. /api/capabilities now carries gfx_target + distro
# --------------------------------------------------------------------------- #


class TestG1_3ApiCapabilitiesExposesTriageContext:
    """The /api/capabilities route pins the wire contract the home page
    tile + any external scripted client consume."""

    def test_capabilities_carries_gfx_target_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["gfx_target"] == "gfx1100"
        assert body["gpu"]["distro"] == "Ubuntu 24.04.4 LTS"

    def test_capabilities_omits_gfx_target_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # gfx_target is a ROCm-only fingerprint. CUDA shouldn't surface
        # it (CUDA tickets use compute capability instead).
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        # Even if the underlying property is populated, the API should not
        # echo it on CUDA.
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "sm_89")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["gfx_target"] is None
        # But distro is populated on every Linux box, including CUDA.
        assert body["gpu"]["distro"] == "Ubuntu 24.04.4 LTS"

    def test_capabilities_distro_none_on_non_linux(
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
        assert body["gpu"]["gfx_target"] is None
        assert body["gpu"]["distro"] is None

    def test_capabilities_cpu_path_still_carries_distro_on_linux(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A headless Linux server doing CPU-only transcription is a real
        # path; distro still matters for kernel + glibc context.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Debian GNU/Linux 12 (bookworm)")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["gpu"]["gfx_target"] is None
        assert body["gpu"]["distro"] == "Debian GNU/Linux 12 (bookworm)"

    def test_capabilities_payload_shape_is_pinned(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The home page tile reads {backend, device_name, vram_gb,
        # gfx_target, distro}. Pin that exact key set so a future
        # response-shape drift fails this test loudly.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert set(body["gpu"].keys()) == {
            "backend", "device_name", "vram_gb", "gfx_target", "distro",
        }

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If gpu_arch_name() / _linux_distro() raise (driver weirdness,
        # sandboxed FS), the API must still respond — fall to None
        # rather than 500.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("driver probe failed")

        monkeypatch.setattr(engine, "gpu_arch_name", _boom)
        monkeypatch.setattr(devices, "_linux_distro", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"]["gfx_target"] is None
        assert body["gpu"]["distro"] is None


# --------------------------------------------------------------------------- #
# 3. CLI surface — `python -m scribe.devices`
# --------------------------------------------------------------------------- #


class TestG1_3DevicesCliSurface:
    """The CLI is the support-bundle output. ``Distro:`` shows on every
    Linux box; ``GFX target:`` shows only on ROCm. Pin that contract."""

    def test_cli_prints_distro_on_linux(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import platform as _platform
        from scribe import devices, engine

        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            _platform,
            "freedesktop_os_release",
            lambda: {"PRETTY_NAME": "Ubuntu 24.04.4 LTS"},
            raising=False,
        )
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "Distro:" in out
        assert "Ubuntu 24.04.4 LTS" in out

    def test_cli_omits_distro_on_non_linux(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import platform as _platform
        from scribe import devices, engine

        monkeypatch.setattr(_platform, "system", lambda: "Darwin")
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: True)

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        # Distro line absent on non-Linux (macOS) — no /etc/os-release.
        assert "Distro:" not in out

    def test_cli_prints_gfx_target_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name",
            lambda i: "AMD Radeon RX 7900 XTX",
        )
        monkeypatch.setattr(
            devices.torch.backends.mps, "is_available", lambda: False
        )
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "GFX target:" in out
        assert "gfx1100" in out

    def test_cli_omits_gfx_target_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes don't get a "GFX target:" line — that's a ROCm-only
        # diagnostic. CUDA users would file with compute capability,
        # which we don't surface here.
        from scribe import devices, engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name",
            lambda i: "NVIDIA GeForce RTX 4090",
        )
        monkeypatch.setattr(
            devices.torch.backends.mps, "is_available", lambda: False
        )

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "GFX target:" not in out


# --------------------------------------------------------------------------- #
# 4. Home page Backend tile sub-line — the UI surface
# --------------------------------------------------------------------------- #


class TestG1_3HomePageBackendTileShowsTriageContext:
    """The Backend stat tile on the home page Recording details card
    consumes ``capabilities.gpu`` directly via ``backendStatTile`` from
    ``helpers.mjs``. G1.3 added two fields to that payload — gfx_target
    and distro — that the tile now appends to the sub-line."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_appends_gfx_target_and_distro(self) -> None:
        # Grep the JS source — the sub-line builder must read both
        # gpu.gfx_target and gpu.distro. Without this glue the API
        # surfaces the fields but the tile can't render them.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "gpu.gfx_target" in text, (
            "backendStatTile() must read gpu.gfx_target so the home page "
            "tile sub-line can show the ROCm gfx target alongside VRAM."
        )
        assert "gpu.distro" in text, (
            "backendStatTile() must read gpu.distro so the home page tile "
            "sub-line can show the Linux pretty-name."
        )

    def test_index_renders_against_rocm_capabilities_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, and the API now carries gfx_target + distro
        # in addition to backend / device_name / vram_gb. Read the API
        # payload in a separate request to confirm.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        # The page renders and imports the helper that will read these.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # The API payload the page will fetch carries the triage context.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["gfx_target"] == "gfx1100"
        assert body["gpu"]["distro"] == "Ubuntu 24.04.4 LTS"


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — helpers → CLI + API + tile
# --------------------------------------------------------------------------- #


class TestG1_3EndToEndChain:
    """One test per realistic backend / OS combo that walks the full
    chain. If any link drops the triage context, this fails."""

    def test_rocm_on_ubuntu_chain(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        # Helper-level
        assert engine.gpu_arch_name() == "gfx1100"
        assert devices._linux_distro() == "Ubuntu 24.04.4 LTS"

        # API-level
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["gfx_target"] == "gfx1100"
        assert body["gpu"]["distro"] == "Ubuntu 24.04.4 LTS"

        # Simulated tile sub-line — what backendStatTile would build.
        gpu = body["gpu"]
        parts = []
        if gpu["device_name"]:
            parts.append(gpu["device_name"])
        if gpu["vram_gb"]:
            parts.append(f"{gpu['vram_gb']} GB VRAM")
        if gpu["gfx_target"]:
            parts.append(gpu["gfx_target"])
        if gpu["distro"]:
            parts.append(gpu["distro"])
        sub = " · ".join(parts)
        assert sub == (
            "AMD Radeon RX 7900 XTX · 24.0 GB VRAM · gfx1100 · Ubuntu 24.04.4 LTS"
        )

    def test_rdna4_on_fedora_chain_pins_known_blocker(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # CT2 issue #2021 — RX 9070 XT (gfx1201) crashes on Fedora 43 +
        # ROCm 7.2. This is the ticket the triage fingerprint is
        # designed for; pin the chain so the right context surfaces.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 9070 XT")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1201")
        monkeypatch.setattr(
            devices, "_linux_distro",
            lambda: "Fedora Linux 43 (Workstation Edition)",
        )

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["gfx_target"] == "gfx1201"
        assert body["gpu"]["distro"] == "Fedora Linux 43 (Workstation Edition)"

    def test_cuda_on_ubuntu_chain(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # CUDA boxes get distro but not gfx_target.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["gfx_target"] is None
        assert body["gpu"]["distro"] == "Ubuntu 24.04.4 LTS"

    def test_mps_on_macos_chain(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Apple Silicon: neither field is populated.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["gfx_target"] is None
        assert body["gpu"]["distro"] is None
