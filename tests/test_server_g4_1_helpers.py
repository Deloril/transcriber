"""Verification of reachability for G4.1 — RDNA 2 cub_caching allocator
workaround surfaced through the user-facing UI.

G4.1 (commit ``55d6459``) shipped the *workaround helper*:
``apply_rocm_runtime_workarounds`` (and its private alias
``_apply_rocm_runtime_workarounds``) which sets
``CT2_CUDA_ALLOCATOR=cub_caching`` on RDNA 2 hardware. Without this
env var, CTranslate2's default ``MallocAsync`` allocator crashes with
"illegal memory access" shortly after a model load on RX 6000-series
cards (CT2 #2012). The fix runs at engine import time, but a worker
subprocess that imports ``ctranslate2`` *before* ``scribe.engine``
can race past the helper and load CT2 with the bad allocator. Until
G4.1 was wired, a researcher had no way to confirm the workaround was
actually in their environment without dropping to
``python -m scribe.devices`` — and even then, the CLI line is hidden
behind a terminal hop.

This iteration extends the chain so the home page Recording details
card carries the live allocator state in the Backend tile sub-line
(and surfaces a visible warning banner when the state is ``"unset"``,
i.e. CT2 is about to crash). Three new fields land on
``GET /api/capabilities``:

  * ``rocm_allocator_state``        — one of ``"auto"`` /
                                      ``"user-overridden"`` /
                                      ``"unset"`` on RDNA 2 ROCm;
                                      ``None`` on every non-ROCm
                                      backend AND on RDNA 3 / RDNA 4 /
                                      CDNA cards (the workaround
                                      doesn't apply there).
  * ``rocm_allocator_value``        — literal env var value when set;
                                      ``None`` when unset or N/A.
  * ``rocm_allocator_explanation``  — one-line human-readable
                                      rationale referencing CT2 #2012
                                      so a researcher pasting the
                                      line into a support thread has
                                      the searchable keyword chain.

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"alloc cub_caching"`` / ``"alloc <user>"`` / ``"alloc unset"`` to
the sub-line and routes the unset-state explanation through the
existing warning banner (G2.1's drift warning) so the user sees the
problem without scrolling. The CLI (``python -m scribe.devices``)
prints the same state under ``Allocator:`` so the two surfaces can
never disagree.

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
# 1. The G4.1 helpers exist + return the documented shapes
# --------------------------------------------------------------------------- #


class TestG4_1HelpersExposeAllocatorStatus:
    """Pin the public surface of the helpers ``/api/capabilities`` and
    ``python -m scribe.devices`` both call into. ``apply_rocm_runtime_workarounds``
    has been part of the public API since G4.1 shipped; the three new
    helpers add the *status* projection used by the UI surface."""

    def test_engine_exposes_allocator_state_helper(self) -> None:
        from scribe.engine import rocm_allocator_state

        # Always callable; returns a string or None.
        result = rocm_allocator_state()
        assert result is None or isinstance(result, str)
        if isinstance(result, str):
            # The helper documents three valid states; a stray return
            # would silently break the JS tile's switch.
            assert result in {"auto", "user-overridden", "unset"}

    def test_engine_exposes_allocator_value_helper(self) -> None:
        from scribe.engine import rocm_allocator_value

        result = rocm_allocator_value()
        assert result is None or isinstance(result, str)

    def test_engine_exposes_allocator_explanation_helper(self) -> None:
        from scribe.engine import rocm_allocator_explanation

        result = rocm_allocator_explanation()
        assert result is None or isinstance(result, str)
        if isinstance(result, str):
            # Must reference the upstream issue id so a researcher
            # pasting the line into a CT2 issue tracker has the
            # searchable keyword.
            assert "2012" in result

    def test_helpers_re_exported_from_top_level_scribe(self) -> None:
        # Mirrors the G3.1 precedent: helpers are useful to lab admins,
        # smoke-test scripts, and third-party integrations, so they're
        # reachable from the top-level ``scribe`` namespace.
        import scribe

        assert hasattr(scribe, "rocm_allocator_state")
        assert hasattr(scribe, "rocm_allocator_value")
        assert hasattr(scribe, "rocm_allocator_explanation")
        assert "rocm_allocator_state" in scribe.__all__
        assert "rocm_allocator_value" in scribe.__all__
        assert "rocm_allocator_explanation" in scribe.__all__


# --------------------------------------------------------------------------- #
# 2. The /api/capabilities response carries the new fields
# --------------------------------------------------------------------------- #


def _stub_rocm_rdna2_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force /api/capabilities into the ROCm + RDNA 2 branch with
    deterministic distro classification so the allocator-status fields
    can be inspected in isolation. Mirrors the ``_stub_rocm_branch``
    helper in ``test_server_g3_1_helpers.py`` but on RDNA 2 hardware
    so the G4.1 fields populate."""
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800 XT")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
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


class TestG4_1ApiCapabilitiesCarriesAllocatorState:
    """Pin the exact JSON shape the helpers/UI consume."""

    def test_rocm_rdna2_auto_reports_cub_caching(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["rocm_allocator_state"] == "auto"
        assert body["gpu"]["rocm_allocator_value"] == "cub_caching"
        explain = body["gpu"]["rocm_allocator_explanation"]
        assert isinstance(explain, str)
        assert "2012" in explain
        assert "cub_caching" in explain

    def test_rocm_rdna2_user_override_reports_value(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "MallocAsync")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_allocator_state"] == "user-overridden"
        assert body["gpu"]["rocm_allocator_value"] == "MallocAsync"
        # Explanation must quote the user value so a support bundle
        # is self-explanatory.
        assert "MallocAsync" in body["gpu"]["rocm_allocator_explanation"]

    def test_rocm_rdna2_unset_reports_warning_actionable(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_allocator_state"] == "unset"
        assert body["gpu"]["rocm_allocator_value"] is None
        # The ``unset`` state is the actionable one — the message must
        # tell the user how to fix it.
        explain = body["gpu"]["rocm_allocator_explanation"]
        assert (
            "apply_rocm_runtime_workarounds" in explain
            or "cub_caching" in explain
        )

    def test_capabilities_omits_allocator_on_cuda(
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
        # who's tinkered with their environment), the allocator field
        # must collapse to None — the workaround doesn't apply.
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["rocm_allocator_state"] is None
        assert body["gpu"]["rocm_allocator_value"] is None
        assert body["gpu"]["rocm_allocator_explanation"] is None

    def test_capabilities_omits_allocator_on_mps(
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
        assert body["gpu"]["rocm_allocator_state"] is None
        assert body["gpu"]["rocm_allocator_value"] is None
        assert body["gpu"]["rocm_allocator_explanation"] is None

    def test_capabilities_omits_allocator_on_cpu(
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
        assert body["gpu"]["rocm_allocator_state"] is None
        assert body["gpu"]["rocm_allocator_value"] is None
        assert body["gpu"]["rocm_allocator_explanation"] is None

    def test_capabilities_omits_allocator_on_rocm_rdna3(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RX 7900 XTX is RDNA 3, not RDNA 2 — the cub_caching
        # workaround doesn't apply, so the field must be None even on
        # ROCm.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        monkeypatch.setattr(engine, "is_rdna2", lambda: False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        # ``None`` (not "auto" / "unset") on non-RDNA-2 ROCm — the
        # workaround is irrelevant on RDNA 3 cards.
        assert body["gpu"]["rocm_allocator_state"] is None
        assert body["gpu"]["rocm_allocator_value"] is None
        assert body["gpu"]["rocm_allocator_explanation"] is None

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If ``rocm_allocator_state`` raises (deeply unexpected — the
        # helper just does an env var lookup and an is_rdna2() probe),
        # the API must still respond — fall to None rather than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD RX 6800 XT")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("env probe failed")

        monkeypatch.setattr(engine, "rocm_allocator_state", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Defensive fallback: None, not a partial value.
        assert body["gpu"]["rocm_allocator_state"] is None
        assert body["gpu"]["rocm_allocator_value"] is None
        assert body["gpu"]["rocm_allocator_explanation"] is None
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
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")
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
# 3. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG4_1HomePageBackendTileShowsAllocator:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G4.1 added the
    ``rocm_allocator_state`` field which the helper turns into a
    ``"alloc <state>"`` sub-line segment. The unset state additionally
    surfaces the explanation through the existing warning banner."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_rocm_allocator_state_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.rocm_allocator_state so the tile can show the alloc
        # segment on RDNA 2 ROCm boxes.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "rocm_allocator_state" in text, (
            "backendStatTile() must read gpu.rocm_allocator_state so the "
            "home page tile sub-line can show the cub_caching workaround "
            "state on RDNA 2 ROCm boxes."
        )
        # The terse segment label is what users will see + grep on; pin
        # all three variants so a future helpers refactor can't quietly
        # drop one.
        assert "alloc cub_caching" in text
        assert "alloc unset" in text

    def test_helpers_mjs_routes_unset_to_warning_banner(self) -> None:
        # The unset state is the actionable one; the helper must thread
        # ``rocm_allocator_explanation`` into the warning field so the
        # index template's ``backendWarning`` banner picks it up.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "rocm_allocator_explanation" in text, (
            "backendStatTile() must read gpu.rocm_allocator_explanation "
            "so the unset-state warning can surface through the "
            "existing backendWarning banner."
        )

    def test_index_renders_backend_warning_banner(self, server_env) -> None:
        # The ``backendWarning`` element pre-dates G4.1 (G2.1 wired the
        # CT2 wheel drift through it) — pin that it's still present so a
        # future template refactor can't quietly drop the surface the
        # G4.1 unset-state warning relies on.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'id="backendWarning"' in html
        assert "backend-warning" in html

    def test_index_renders_against_rocm_rdna2_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries rocm_allocator_state,
        # and the helper turns ``"auto"`` into the sub-line segment.
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the allocator state verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_allocator_state"] == "auto"
        assert body["gpu"]["rocm_allocator_value"] == "cub_caching"
        assert body["gpu"]["rocm_allocator_explanation"]


# --------------------------------------------------------------------------- #
# 4. End-to-end chain — workaround helper → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG4_1EndToEndChain:
    """Walk the full chain for each realistic state. If any link drops
    the allocator state, this fails."""

    def test_rocm_rdna2_auto_state_chain(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic happy path: RX 6800 XT on Ubuntu 24.04 with the
        # cub_caching env var set by apply_rocm_runtime_workarounds().
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.setenv("CT2_CUDA_ALLOCATOR", "cub_caching")
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["rocm_allocator_state"] == "auto"
        assert gpu["rocm_allocator_value"] == "cub_caching"
        # Explanation isn't empty + contains the searchable issue id.
        assert "2012" in gpu["rocm_allocator_explanation"]

    def test_rocm_rdna2_unset_state_is_actionable(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The danger state: RDNA 2 box where CT2 will crash on the
        # next model load. The chain must thread the warning all the
        # way through to the JSON payload — the home page surfaces it
        # in the warning banner.
        srv, client, _ = server_env
        _stub_rocm_rdna2_branch(monkeypatch)
        monkeypatch.delenv("CT2_CUDA_ALLOCATOR", raising=False)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["rocm_allocator_state"] == "unset"
        assert gpu["rocm_allocator_value"] is None
        assert "2012" in gpu["rocm_allocator_explanation"]

    def test_non_rdna2_chain_collapses_cleanly(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RX 7900 XTX on Ubuntu — the workaround is irrelevant, so all
        # three fields must be None.
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
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["rocm_allocator_state"] is None
        assert gpu["rocm_allocator_value"] is None
        assert gpu["rocm_allocator_explanation"] is None
        # Other ROCm fields still populate — additive guarantee.
        assert gpu["ct2_rocm_pin"] == "4.7.2"
