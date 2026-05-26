"""Verification of reachability for G4.2 — RDNA 2
``HSA_OVERRIDE_GFX_VERSION`` workaround surfaced through the user-facing UI.

G4.2 (commit ``a878fdb``) shipped the *detector*: ``needs_hsa_override`` /
``recommended_hsa_override_value`` plus the ``HSA_OVERRIDE_RDNA2_VALUE``
constant. The override has to be in the environment *before* the HIP
runtime initialises, so unlike G4.1's ``CT2_CUDA_ALLOCATOR=cub_caching``
fix we can't auto-apply this from Python (torch is already loaded by
the time engine.py runs). The detector drives the CLI hint in
``python -m scribe.devices`` and the message in ``setup.sh --rocm``,
but the original commit had no Reachable-via line; until this commit a
researcher had no in-app way to confirm the override was active in
their environment.

This iteration extends the chain so the home page Recording details
card carries the live HSA override state in the Backend tile sub-line
(and surfaces a visible warning banner when the state is ``"missing"``,
i.e. HIP is about to refuse to load kernels). Three new fields land on
``GET /api/capabilities``:

  * ``rocm_hsa_override_state``       — one of ``"user-set"`` /
                                        ``"missing"`` on RDNA 2 ROCm
                                        (and gfx1030 ROCm with the env
                                        var set); ``None`` on every
                                        non-ROCm backend AND on RDNA 3 /
                                        RDNA 4 / CDNA cards AND on
                                        gfx1030 with the env var unset
                                        (the workaround is only
                                        actionable for non-gfx1030
                                        RDNA 2 dies).
  * ``rocm_hsa_override_value``       — literal env var value when
                                        set; ``None`` when unset or
                                        N/A.
  * ``rocm_hsa_override_explanation`` — one-line human-readable
                                        rationale referencing the
                                        AMD-only-ships-kernels-for-
                                        gfx1030 limitation and (on the
                                        actionable ``"missing"`` state)
                                        the detected gfx target plus
                                        the export recommendation.

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"HSA <value>"`` / ``"HSA missing"`` to the sub-line and routes the
missing-state explanation through the existing warning banner so the
user sees the problem without scrolling. The CLI
(``python -m scribe.devices``) prints the same state under ``HSA
override:`` so the two surfaces can never disagree.

Deeper unit coverage of the three helpers themselves lives in
``tests/test_engine_devices.py`` (every state across every backend).
The JS-side tile shape is pinned in ``tests/js/backend-stat-tile.test.mjs``.
This file is the integration-level reachability proof.
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
# 1. The G4.2 helpers exist + return the documented shapes
# --------------------------------------------------------------------------- #


class TestG4_2HelpersExposeHsaOverrideStatus:
    """Pin the public surface of the helpers ``/api/capabilities`` and
    ``python -m scribe.devices`` both call into."""

    def test_engine_exposes_hsa_override_state_helper(self) -> None:
        from scribe.engine import rocm_hsa_override_state

        # Always callable; returns a string or None.
        result = rocm_hsa_override_state()
        assert result is None or isinstance(result, str)
        if isinstance(result, str):
            # The helper documents two valid states; a stray return
            # would silently break the JS tile's switch.
            assert result in {"user-set", "missing"}

    def test_engine_exposes_hsa_override_value_helper(self) -> None:
        from scribe.engine import rocm_hsa_override_value

        result = rocm_hsa_override_value()
        assert result is None or isinstance(result, str)

    def test_engine_exposes_hsa_override_explanation_helper(self) -> None:
        from scribe.engine import rocm_hsa_override_explanation

        result = rocm_hsa_override_explanation()
        assert result is None or isinstance(result, str)
        if isinstance(result, str):
            # Must reference the env var name so a researcher
            # pasting the line into a ROCm support thread has the
            # searchable keyword chain.
            assert "HSA_OVERRIDE_GFX_VERSION" in result

    def test_helpers_re_exported_from_top_level_scribe(self) -> None:
        # Mirrors the G4.1 / G3.1 precedent.
        import scribe

        assert hasattr(scribe, "rocm_hsa_override_state")
        assert hasattr(scribe, "rocm_hsa_override_value")
        assert hasattr(scribe, "rocm_hsa_override_explanation")
        assert "rocm_hsa_override_state" in scribe.__all__
        assert "rocm_hsa_override_value" in scribe.__all__
        assert "rocm_hsa_override_explanation" in scribe.__all__


# --------------------------------------------------------------------------- #
# 2. The /api/capabilities response carries the new fields
# --------------------------------------------------------------------------- #


def _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force /api/capabilities into the ROCm + RDNA 2 + non-gfx1030
    branch (RX 6700 XT — gfx1031) with deterministic distro
    classification so the HSA override fields can be inspected in
    isolation. Mirrors ``_stub_rocm_rdna2_branch`` in
    ``test_server_g4_1_helpers.py`` but on a die that NEEDS the
    override (gfx1031, not gfx1030) so the G4.2 fields populate."""
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6700 XT")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 12.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1031")
    monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
    monkeypatch.setattr(engine, "is_rdna2", lambda: True)
    monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
    monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
    monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
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


class TestG4_2ApiCapabilitiesCarriesHsaOverrideState:
    """Pin the exact JSON shape the helpers/UI consume."""

    def test_rocm_non_gfx1030_user_set_reports_value(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["rocm_hsa_override_state"] == "user-set"
        assert body["gpu"]["rocm_hsa_override_value"] == "10.3.0"
        explain = body["gpu"]["rocm_hsa_override_explanation"]
        assert isinstance(explain, str)
        assert "HSA_OVERRIDE_GFX_VERSION" in explain
        assert "10.3.0" in explain

    def test_rocm_non_gfx1030_user_set_non_recommended_value(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_hsa_override_state"] == "user-set"
        assert body["gpu"]["rocm_hsa_override_value"] == "11.0.0"
        # Explanation must quote the user value so a support bundle
        # is self-explanatory.
        assert "11.0.0" in body["gpu"]["rocm_hsa_override_explanation"]

    def test_rocm_non_gfx1030_missing_reports_warning_actionable(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_hsa_override_state"] == "missing"
        assert body["gpu"]["rocm_hsa_override_value"] is None
        # The ``missing`` state is the actionable one — the message
        # must tell the user how to fix it (gfx target + export hint).
        explain = body["gpu"]["rocm_hsa_override_explanation"]
        assert "gfx1031" in explain
        assert "10.3.0" in explain
        assert "HSA_OVERRIDE_GFX_VERSION" in explain

    def test_rocm_gfx1030_unset_reports_none(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # gfx1030 doesn't need the override (AMD ships kernels for it).
        # State collapses to None even on ROCm.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800 XT")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.setattr(engine, "is_rdna2", lambda: True)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None

    def test_rocm_gfx1030_user_set_still_reports_user_set(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even on gfx1030, if the user has set the variable we surface
        # it (researchers may have legitimate reasons — mixed-GPU box,
        # forced compatibility).
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800 XT")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.setattr(engine, "is_rdna2", lambda: True)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_hsa_override_state"] == "user-set"
        assert body["gpu"]["rocm_hsa_override_value"] == "10.3.0"

    def test_capabilities_omits_hsa_override_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        # Even if the env var is set on a CUDA box (e.g. a researcher
        # who's tinkered with their environment), the field must
        # collapse to None — the variable is meaningless without HIP.
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None

    def test_capabilities_omits_hsa_override_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None

    def test_capabilities_omits_hsa_override_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None

    def test_capabilities_omits_hsa_override_on_rocm_rdna3(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RX 7900 XTX is RDNA 3 — the override doesn't apply (ROCm
        # ships kernels for gfx1100 directly).
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        # ``None`` (not "missing") on RDNA 3 — the override is
        # irrelevant.
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If ``rocm_hsa_override_state`` raises (deeply unexpected —
        # the helper just does an env var lookup and an arch probe),
        # the API must still respond — fall to None rather than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("env probe failed")

        monkeypatch.setattr(engine, "rocm_hsa_override_state", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Defensive fallback: None, not a partial value.
        assert body["gpu"]["rocm_hsa_override_state"] is None
        assert body["gpu"]["rocm_hsa_override_value"] is None
        assert body["gpu"]["rocm_hsa_override_explanation"] is None
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
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
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
        }


# --------------------------------------------------------------------------- #
# 3. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG4_2HomePageBackendTileShowsHsaOverride:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G4.2 added the
    ``rocm_hsa_override_state`` field which the helper turns into a
    ``"HSA <state>"`` sub-line segment. The missing state additionally
    surfaces the explanation through the existing warning banner."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_rocm_hsa_override_state_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.rocm_hsa_override_state so the tile can show the HSA
        # segment on RDNA 2 non-gfx1030 ROCm boxes.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "rocm_hsa_override_state" in text, (
            "backendStatTile() must read gpu.rocm_hsa_override_state so "
            "the home page tile sub-line can show the HSA override "
            "state on RDNA 2 ROCm boxes."
        )
        # The terse segment label is what users will see + grep on; pin
        # both variants so a future helpers refactor can't quietly drop
        # one.
        assert "HSA missing" in text
        # The "HSA <value>" template is interpolated at runtime; the
        # literal token "HSA " or "HSA user-set" must appear.
        assert "HSA " in text or "HSA user-set" in text

    def test_helpers_mjs_routes_missing_to_warning_banner(self) -> None:
        # The missing state is the actionable one; the helper must
        # thread ``rocm_hsa_override_explanation`` into the warning
        # field so the index template's ``backendWarning`` banner
        # picks it up.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "rocm_hsa_override_explanation" in text, (
            "backendStatTile() must read gpu.rocm_hsa_override_explanation "
            "so the missing-state warning can surface through the "
            "existing backendWarning banner."
        )

    def test_index_renders_backend_warning_banner(self, server_env) -> None:
        # The ``backendWarning`` element pre-dates G4.2 — pin that it's
        # still present so a future template refactor can't quietly
        # drop the surface the missing-state warning relies on.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'id="backendWarning"' in html
        assert "backend-warning" in html

    def test_index_renders_against_rocm_rdna2_non_gfx1030_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries
        # rocm_hsa_override_state, and the helper turns ``"user-set"``
        # into the sub-line segment.
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the override state verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_hsa_override_state"] == "user-set"
        assert body["gpu"]["rocm_hsa_override_value"] == "10.3.0"
        assert body["gpu"]["rocm_hsa_override_explanation"]


# --------------------------------------------------------------------------- #
# 4. End-to-end chain — workaround detector → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG4_2EndToEndChain:
    """Walk the full chain for each realistic state. If any link drops
    the override state, this fails."""

    def test_rocm_non_gfx1030_user_set_state_chain(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic happy path: RX 6700 XT on Ubuntu 24.04 with the
        # user having exported HSA_OVERRIDE_GFX_VERSION=10.3.0 in
        # their shell.
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["rocm_hsa_override_state"] == "user-set"
        assert gpu["rocm_hsa_override_value"] == "10.3.0"
        # Explanation isn't empty + names the env var.
        assert "HSA_OVERRIDE_GFX_VERSION" in gpu["rocm_hsa_override_explanation"]

    def test_rocm_non_gfx1030_missing_state_is_actionable(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The danger state: RX 6700 XT box where HIP won't load
        # kernels until the user exports the override. The chain must
        # thread the warning all the way through to the JSON payload
        # — the home page surfaces it in the warning banner.
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["rocm_hsa_override_state"] == "missing"
        assert gpu["rocm_hsa_override_value"] is None
        # Actionable explanation must mention both the gfx target and
        # the recommended export.
        assert "gfx1031" in gpu["rocm_hsa_override_explanation"]
        assert "10.3.0" in gpu["rocm_hsa_override_explanation"]

    def test_gfx1030_chain_collapses_cleanly(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # gfx1030 doesn't need the override — all three fields must be
        # None when the env var is unset (the user-set branch is
        # tested separately).
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800 XT")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
        monkeypatch.setattr(engine, "is_rdna2", lambda: True)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["rocm_hsa_override_state"] is None
        assert gpu["rocm_hsa_override_value"] is None
        assert gpu["rocm_hsa_override_explanation"] is None
        # Other ROCm fields still populate — additive guarantee.
        assert gpu["ct2_rocm_pin"] == "4.7.2"

    def test_non_rdna2_rocm_chain_collapses_cleanly(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RX 7900 XTX on Ubuntu — RDNA 3, override doesn't apply.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["rocm_hsa_override_state"] is None
        assert gpu["rocm_hsa_override_value"] is None
        assert gpu["rocm_hsa_override_explanation"] is None

    def test_g4_1_and_g4_2_independent_on_rdna2_non_gfx1030(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both workarounds apply on a non-gfx1030 RDNA 2 die. They're
        # independent: G4.1 (cub_caching) is auto-applicable from
        # Python; G4.2 (HSA override) has to be set in the shell. The
        # API must report both states independently so the UI can
        # render two separate warnings if needed.
        srv, client, _ = server_env
        _stub_rocm_rdna2_non_gfx1030_branch(monkeypatch)
        # G4.1 auto-fix is applied; G4.2 user hasn't set the env var.
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["rocm_allocator_state"] == "auto"
        assert gpu["rocm_hsa_override_state"] == "missing"
        # The two states are reported independently — a researcher
        # who fixed one but not the other sees the right message.
