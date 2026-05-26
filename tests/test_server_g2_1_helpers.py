"""Verification of reachability for G2.1 — pinned CT2 ROCm wheel + drift
detection surfaced through the user-facing UI.

G2.1 (commit ``bba0bdb``) made ``scribe/rocm_install.py`` the single
source of truth for the CTranslate2 ROCm wheel version Scribe ships
against. The pin is read by:

  * ``setup.sh --rocm`` (the shell installer downloads the
    matching zipped wheel from GitHub Releases), and
  * ``python -m scribe.devices`` (the CLI report prints the pin and
    warns if the actually-installed ``ctranslate2`` package has
    drifted away from it — the most common failure mode after an
    unrelated ``pip install`` upgrades the package transitively).

The original commit only surfaced the pin / drift through the CLI.
This iteration extends the chain so a researcher who never opens a
terminal sees the same information on the home page Recording details
card, and a concrete drift state pops a visible amber banner so the
problem is impossible to miss. Three new fields land on the
``GET /api/capabilities`` ``gpu`` payload:

  * ``ct2_rocm_pin``        — pinned wheel version (e.g. ``"4.7.2"``)
  * ``ct2_installed``       — actually-installed ``ctranslate2`` version
  * ``ct2_drift_message``   — human-readable warning, ``None`` when matched

All three are ``None`` on every non-ROCm backend — a CUDA / MPS / CPU
user's ``ctranslate2`` build is unrelated to the ROCm pin, so showing
it would just be noise. The home page tile (``backendStatTile`` in
``helpers.mjs``) appends ``"CT2 v<pin>"`` to the sub-line on ROCm and
exposes a top-level ``warning`` field that the renderer surfaces as
the ``#backendWarning`` banner under the stats grid.

Deeper unit coverage of the helpers themselves lives in
``tests/test_rocm_install.py`` (pin / drift logic) and
``tests/test_devices.py`` (CLI render). The JS-side tile shape is
pinned in ``tests/js/backend-stat-tile.test.mjs`` (38 cases). This
file is the integration-level reachability proof.
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
# 1. The G2.1 helpers exist and return the documented shapes
# --------------------------------------------------------------------------- #


class TestG2_1HelpersExposePinAndDrift:
    """Pin the public surface of the three helpers ``/api/capabilities``
    and ``python -m scribe.devices`` both call into."""

    def test_pinned_ct2_rocm_version_returns_string(self) -> None:
        from scribe.rocm_install import pinned_ct2_rocm_version

        v = pinned_ct2_rocm_version()
        assert isinstance(v, str)
        # Looks like a semver-ish version, not the empty string.
        assert v
        assert "." in v

    def test_installed_ct2_version_returns_str_or_none(self) -> None:
        from scribe.rocm_install import installed_ct2_version

        result = installed_ct2_version()
        assert result is None or isinstance(result, str)

    def test_ct2_drift_message_matched_returns_none(self) -> None:
        from scribe.rocm_install import ct2_drift_message

        # When installed == pinned, no drift.
        assert ct2_drift_message(installed="4.7.2", pinned="4.7.2") is None

    def test_ct2_drift_message_mismatch_describes_versions(self) -> None:
        from scribe.rocm_install import ct2_drift_message

        msg = ct2_drift_message(installed="4.6.0", pinned="4.7.2")
        assert msg is not None
        assert "4.6.0" in msg
        assert "4.7.2" in msg
        assert "./setup.sh --rocm" in msg

    def test_ct2_drift_message_not_installed_says_so(self) -> None:
        from scribe.rocm_install import ct2_drift_message

        msg = ct2_drift_message(installed=None, pinned="4.7.2")
        assert msg is not None
        assert "not found" in msg
        assert "4.7.2" in msg


# --------------------------------------------------------------------------- #
# 2. /api/capabilities now carries ct2_rocm_pin + ct2_installed + ct2_drift
# --------------------------------------------------------------------------- #


class TestG2_1ApiCapabilitiesCarriesPinAndDrift:
    """The /api/capabilities route is the wire contract the home page
    tile + any external scripted client consume."""

    def _stub_rocm(
        self,
        srv,
        monkeypatch: pytest.MonkeyPatch,
        *,
        installed: str | None = "4.7.2",
        pinned: str = "4.7.2",
    ) -> None:
        """Force /api/capabilities into the ROCm branch with a concrete
        pin / installed pair so the response is fully deterministic."""
        from scribe import engine, devices, rocm_install

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        # The route imports these by name from rocm_install at request
        # time, so monkeypatching the module attributes is sufficient.
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: pinned)
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: installed)

    def test_capabilities_carries_pin_on_rocm(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch, installed="4.7.2", pinned="4.7.2")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"
        assert body["gpu"]["ct2_installed"] == "4.7.2"
        # Matched → no drift message.
        assert body["gpu"]["ct2_drift_message"] is None

    def test_capabilities_carries_drift_when_versions_disagree(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch, installed="4.6.0", pinned="4.7.2")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"
        assert body["gpu"]["ct2_installed"] == "4.6.0"
        msg = body["gpu"]["ct2_drift_message"]
        assert msg is not None
        # The same human-readable drift the CLI prints — pin both
        # version strings so a regression that loses either side fails.
        assert "4.6.0" in msg
        assert "4.7.2" in msg
        assert "./setup.sh --rocm" in msg

    def test_capabilities_carries_drift_when_ct2_not_installed(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A ROCm box with no ctranslate2 installed at all (fresh setup
        # in progress, or a venv where pip uninstalled it) is the most
        # actionable drift case — we want the banner to fire.
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch, installed=None, pinned="4.7.2")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_installed"] is None
        msg = body["gpu"]["ct2_drift_message"]
        assert msg is not None
        assert "not found" in msg
        assert "4.7.2" in msg

    def test_capabilities_omits_pin_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes don't see the ROCm-only pin — a CUDA user's CT2
        # build is unrelated to the AMD wheel pin and showing it would
        # be confusing noise.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["ct2_rocm_pin"] is None
        assert body["gpu"]["ct2_installed"] is None
        assert body["gpu"]["ct2_drift_message"] is None

    def test_capabilities_omits_pin_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Apple Silicon is not a ROCm target.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["gpu"]["ct2_rocm_pin"] is None
        assert body["gpu"]["ct2_installed"] is None
        assert body["gpu"]["ct2_drift_message"] is None

    def test_capabilities_omits_pin_on_cpu(
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
        assert body["gpu"]["ct2_rocm_pin"] is None
        assert body["gpu"]["ct2_installed"] is None
        assert body["gpu"]["ct2_drift_message"] is None

    def test_capabilities_payload_shape_is_pinned(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The home page tile reads the gpu payload by key. Pin the full
        # key set so a future response-shape drift fails this test
        # loudly.
        srv, client, _ = server_env
        self._stub_rocm(srv, monkeypatch, installed="4.7.2", pinned="4.7.2")
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
        # If pinned_ct2_rocm_version() / installed_ct2_version() /
        # ct2_drift_message() raise (corrupt venv, broken metadata),
        # the API must still respond — fall to None rather than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("metadata probe failed")

        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", _boom)
        monkeypatch.setattr(rocm_install, "installed_ct2_version", _boom)
        monkeypatch.setattr(rocm_install, "ct2_drift_message", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"]["ct2_rocm_pin"] is None
        assert body["gpu"]["ct2_installed"] is None
        assert body["gpu"]["ct2_drift_message"] is None


# --------------------------------------------------------------------------- #
# 3. CLI surface — `python -m scribe.devices`
# --------------------------------------------------------------------------- #


class TestG2_1DevicesCliSurface:
    """The CLI is the support-bundle output. Pin both halves of G2.1 —
    the pin line itself, and the drift-warning line that fires when
    the installed ctranslate2 doesn't match."""

    def test_cli_prints_ct2_pin_on_rocm_when_matched(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name",
            lambda i: "AMD Radeon RX 7900 XTX",
        )
        monkeypatch.setattr(
            devices.torch.backends.mps, "is_available", lambda: False
        )
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(devices, "installed_ct2_version", lambda: "4.7.2")

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "CT2 ROCm pin:" in out
        assert "v4.7.2" in out
        # Matched → no drift line.
        assert "drift:" not in out

    def test_cli_warns_on_drift(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name",
            lambda i: "AMD Radeon RX 7900 XTX",
        )
        monkeypatch.setattr(
            devices.torch.backends.mps, "is_available", lambda: False
        )
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)
        monkeypatch.setattr(devices, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(devices, "installed_ct2_version", lambda: "4.6.0")

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "CT2 ROCm pin:" in out
        assert "drift:" in out
        assert "4.6.0" in out
        assert "4.7.2" in out

    def test_cli_omits_ct2_pin_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes don't see the ROCm-only pin.
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
        assert "CT2 ROCm pin:" not in out


# --------------------------------------------------------------------------- #
# 4. Home page Backend tile + drift banner — the UI surface
# --------------------------------------------------------------------------- #


class TestG2_1HomePageBackendTileAndWarning:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G2.1 added the
    ``ct2_rocm_pin`` field to the sub-line and the ``warning`` field
    that the page renders as the ``#backendWarning`` banner under the
    stats grid."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_index_renders_backend_warning_banner_element(
        self, server_env
    ) -> None:
        # The banner element exists in the page HTML — empty + hidden
        # by default; renderStats() populates and shows it when the
        # tile carries a non-null warning. Without this element on the
        # page, a future regression that drops the `warning` field has
        # nowhere to surface.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'id="backendWarning"' in html
        assert "backend-warning" in html  # CSS class

    def test_helpers_mjs_appends_ct2_rocm_pin_to_sub_line(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.ct2_rocm_pin so the tile can show "CT2 v4.7.2" on ROCm.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "ct2_rocm_pin" in text, (
            "backendStatTile() must read gpu.ct2_rocm_pin so the home "
            "page tile sub-line can show the pinned CT2 ROCm wheel "
            "version on ROCm boxes."
        )
        assert "ct2_drift_message" in text, (
            "backendStatTile() must read gpu.ct2_drift_message so the "
            "home page banner can surface the drift warning."
        )

    def test_index_renders_warning_glue_for_backend_warning(
        self, server_env
    ) -> None:
        # The renderStats() body must populate #backendWarning from the
        # tile's `warning` field. Pin the wiring so a future refactor
        # that drops the warning render fails this test.
        srv, client, _ = server_env
        html = client.get("/").text
        # Both pieces must be on the page — the variable read and the
        # element manipulation:
        assert "backendTile.warning" in html or "backendTile && backendTile.warning" in html
        assert "backendWarning" in html
        # And the banner must use the `.hidden` toggle pattern other
        # cards use, not display: none inline (consistency).
        assert "backend-warning hidden" in html

    def test_index_renders_against_rocm_drift_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries ct2_rocm_pin /
        # ct2_installed / ct2_drift_message, and the helper turns
        # them into a tile with a warning string.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.6.0")

        # Page renders + imports the helper.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the drift fields.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"
        assert body["gpu"]["ct2_installed"] == "4.6.0"
        assert body["gpu"]["ct2_drift_message"] is not None
        assert "4.6.0" in body["gpu"]["ct2_drift_message"]
        assert "4.7.2" in body["gpu"]["ct2_drift_message"]


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — helpers → CLI + API + tile
# --------------------------------------------------------------------------- #


class TestG2_1EndToEndChain:
    """One test per realistic scenario that walks the full chain. If
    any link drops the pin / drift, this fails."""

    def test_rocm_matched_chain(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

        # Helper-level
        assert rocm_install.pinned_ct2_rocm_version() == "4.7.2"
        assert rocm_install.installed_ct2_version() == "4.7.2"
        assert rocm_install.ct2_drift_message(installed="4.7.2", pinned="4.7.2") is None

        # API-level
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"
        assert body["gpu"]["ct2_installed"] == "4.7.2"
        assert body["gpu"]["ct2_drift_message"] is None

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
        if gpu["ct2_rocm_pin"]:
            parts.append(f"CT2 v{gpu['ct2_rocm_pin']}")
        sub = " · ".join(parts)
        assert sub == (
            "AMD Radeon RX 7900 XTX · 24.0 GB VRAM · gfx1100 · "
            "Ubuntu 24.04.4 LTS · CT2 v4.7.2"
        )

    def test_rocm_drift_chain_fires_warning_banner_data(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The realistic worst case: a researcher's `pip install` of
        # something else has pulled in an unrelated ctranslate2. The
        # CLI prints "drift:", the API surfaces ct2_drift_message,
        # the tile carries `warning`, and the home page banner shows it.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.6.0")

        body = client.get("/api/capabilities").json()
        msg = body["gpu"]["ct2_drift_message"]
        assert msg is not None
        assert "4.6.0 installed" in msg
        assert "4.7.2" in msg
        # The renderer reads `tile.warning = drift ? String(drift) : null`,
        # so simulate that and pin the rendered banner string.
        warning = msg if msg else None
        assert warning is not None
        assert "ctranslate2" in warning  # the banner names the package

    def test_rocm_no_install_chain(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fresh ROCm box mid-setup: pin set, ctranslate2 not yet
        # installed. The user needs to be told to run setup.sh --rocm.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_installed"] is None
        msg = body["gpu"]["ct2_drift_message"]
        assert msg is not None
        assert "not found" in msg
        assert "./setup.sh --rocm" in msg

    def test_cuda_chain_does_not_carry_pin(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA: the pin / installed / drift fields are all None even if
        # the underlying helpers would happily report a pinned wheel
        # version. Pin the suppression here so a future refactor that
        # leaks the ROCm pin onto a CUDA tile fails this test loudly.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["ct2_rocm_pin"] is None
        assert body["gpu"]["ct2_installed"] is None
        assert body["gpu"]["ct2_drift_message"] is None
