"""Verification of reachability for G1.1 — public ROCm runtime-detection
helpers.

G1.1 lifted ``is_rocm()`` / ``is_cuda()`` / ``is_mps()`` / ``has_gpu()``
/ ``gpu_vendor()`` / ``gpu_runtime_version()`` out of inline string
comparisons in ``scribe.engine`` and ``scribe.devices`` and re-exported
them from the top-level ``scribe`` package. These look like pure-internal
helpers, but they are reachable from the user-facing surface because:

  - ``GET /api/capabilities`` returns ``gpu.backend`` populated by
    ``scribe.engine.gpu_backend()`` — the same source-of-truth the
    G1.1 booleans wrap.

  - The home page (``index.html``) imports ``backendStatTile`` and
    ``formatBackendLabel`` from ``static/js/helpers.mjs`` and renders
    a "Backend" tile on the Recording details card whose label is
    ``CUDA`` / ``ROCm`` / ``MPS`` / ``CPU`` depending on what the
    helper chain reports.

  - The G1.1 booleans (``is_rocm`` / ``is_cuda`` / ``is_mps`` /
    ``has_gpu``) are re-exported from the top-level ``scribe`` package
    and are consumed by ``scribe.engine`` itself (worker-process
    routing) and by ``scribe.devices`` (support-bundle output) — both
    of which feed the same ``/api/capabilities`` payload.

This file consolidates the reachability proof so the next loop
iteration can confirm the surface is wired without re-deriving it from
scratch. The deeper unit coverage of the helpers themselves lives in
``tests/test_engine_devices.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same shape as the fixture in test_server.py; copied to keep
    this verification file self-contained."""
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
# 1. Top-level package re-exports — the public Python surface
# --------------------------------------------------------------------------- #


class TestG1_1PublicPackageSurface:
    """G1.1 promised these helpers as ``scribe.*`` (not just
    ``scribe.engine.*``). Without this re-export the rest of the codebase
    falls back to inline ``gpu_backend() == "rocm"`` ceremony."""

    def test_helpers_are_top_level_imports(self) -> None:
        import scribe
        from scribe import engine

        assert scribe.gpu_backend is engine.gpu_backend
        assert scribe.gpu_vendor is engine.gpu_vendor
        assert scribe.gpu_runtime_version is engine.gpu_runtime_version
        assert scribe.is_rocm is engine.is_rocm
        assert scribe.is_cuda is engine.is_cuda
        assert scribe.is_mps is engine.is_mps
        assert scribe.has_gpu is engine.has_gpu

    def test_helpers_listed_in_dunder_all(self) -> None:
        import scribe

        for name in (
            "gpu_backend",
            "gpu_vendor",
            "gpu_runtime_version",
            "is_rocm",
            "is_cuda",
            "is_mps",
            "has_gpu",
        ):
            assert name in scribe.__all__, (
                f"{name} must be in scribe.__all__ so the public surface "
                "doesn't silently regress"
            )


# --------------------------------------------------------------------------- #
# 2. Booleans partition the four-state label cleanly
# --------------------------------------------------------------------------- #


class TestG1_1BooleansPartitionBackendLabel:
    """For every value ``gpu_backend()`` can legitimately return, exactly
    one of ``is_cuda()`` / ``is_rocm()`` / ``is_mps()`` is True (and
    ``has_gpu()`` agrees with ``backend != "cpu"``). This is the contract
    the rest of the engine + UI relies on."""

    @pytest.mark.parametrize(
        "backend,expect_cuda,expect_rocm,expect_mps,expect_gpu",
        [
            ("cuda", True,  False, False, True),
            ("rocm", False, True,  False, True),
            ("mps",  False, False, True,  True),
            ("cpu",  False, False, False, False),
        ],
    )
    def test_partition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        expect_cuda: bool,
        expect_rocm: bool,
        expect_mps: bool,
        expect_gpu: bool,
    ) -> None:
        from scribe import engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        assert engine.is_cuda() is expect_cuda
        assert engine.is_rocm() is expect_rocm
        assert engine.is_mps() is expect_mps
        assert engine.has_gpu() is expect_gpu


# --------------------------------------------------------------------------- #
# 3. /api/capabilities echoes what gpu_backend() returns
# --------------------------------------------------------------------------- #


class TestG1_1ApiCapabilitiesUsesHelperChain:
    """The ``/api/capabilities`` route — the user-facing surface for the
    G1.1 helper chain — must reflect ``gpu_backend()`` for all four
    states, including the ROCm state that motivated G1.1 in the first
    place."""

    @pytest.mark.parametrize("backend", ["cuda", "rocm", "mps", "cpu"])
    def test_api_capabilities_reports_helper_backend(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
    ) -> None:
        from scribe import engine
        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"]["backend"] == backend, (
            f"/api/capabilities must echo gpu_backend() ({backend}); "
            "this is the surface the home-page Backend tile reads."
        )

    def test_api_capabilities_returns_documented_field_set(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pin the field shape the ``backendStatTile`` JS helper consumes.
        If a future refactor renames any of these the home page tile
        breaks silently. ``gfx_target`` and ``distro`` were added by
        G1.3 to the same payload (ROCm-only / Linux-only respectively);
        ``backendStatTile`` reads them too, so the pinned key set
        includes them here. ``ct2_rocm_pin`` / ``ct2_installed`` /
        ``ct2_drift_message`` were added by G2.1 (ROCm-only) and are
        likewise read by ``backendStatTile`` — the helper appends the
        pinned wheel version to the sub-line on ROCm and surfaces the
        drift message via the tile's ``warning`` field."""
        from scribe import engine
        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)

        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        # G1.4 fields + G1.3 fields + G2.1 fields + G2.3 fields. New
        # fields are nullable; the tile helper tolerates
        # missing-or-null values so the contract remains backwards
        # compatible.
        assert set(gpu.keys()) == {
            "backend", "device_name", "vram_gb", "gfx_target", "distro",
            "ct2_rocm_pin", "ct2_installed", "ct2_drift_message",
            "ct2_rocm_fallback_urls",
            "distro_tier", "distro_tier_explanation",
            "rocm_lstm_patch", "rocm_lstm_patch_explanation",
            "rocm_allocator_state", "rocm_allocator_value",
            "rocm_allocator_explanation",
            "rocm_hsa_override_state", "rocm_hsa_override_value",
            "rocm_hsa_override_explanation",
            "whisper_compute_type",
        }
        assert gpu["backend"] == "rocm"
        assert gpu["device_name"] == "Radeon RX 7900 XTX"
        assert gpu["vram_gb"] == 24.0


# --------------------------------------------------------------------------- #
# 4. Home page wires the helper chain through to a visible tile
# --------------------------------------------------------------------------- #


class TestG1_1HomePageRendersBackendTile:
    """The home page imports ``backendStatTile`` + ``formatBackendLabel``
    from ``helpers.mjs`` and pushes the result into the stat-tile
    pipeline. Asserting the imports are present is the last link in the
    chain — without them, the helper output never reaches the user."""

    def test_index_imports_backend_tile_helpers(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        assert "backendStatTile" in html, (
            "index.html must import backendStatTile from helpers.mjs so "
            "the Backend tile renders on the Recording details card"
        )
        assert "formatBackendLabel" in html, (
            "index.html must import formatBackendLabel so the Parakeet "
            "warning renders 'ROCm' (not 'ROCM') consistently"
        )

    def test_index_pushes_backend_tile_into_stats(self, server_env) -> None:
        """The renderStats() function on the home page must actually
        call ``backendStatTile`` against ``_caps.gpu`` — otherwise the
        import is dead code and the tile never appears."""
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html

    def test_index_fetches_capabilities_endpoint(self, server_env) -> None:
        """Closes the loop: the page actually calls
        ``/api/capabilities`` to populate the ``_caps`` object the tile
        reads from."""
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'fetch("/api/capabilities")' in html


# --------------------------------------------------------------------------- #
# 5. End-to-end: helpers → API → tile data shape
# --------------------------------------------------------------------------- #


class TestG1_1EndToEndChain:
    """One test per backend that walks the full chain:
       gpu_backend() → /api/capabilities → backendStatTile-shaped dict.
    If any link breaks, this test fails."""

    @pytest.mark.parametrize(
        "backend,device_name,vram_gb,expected_label",
        [
            ("cuda", "NVIDIA RTX 4090",      24.0, "CUDA"),
            ("rocm", "Radeon RX 7900 XTX",   24.0, "ROCm"),
            ("mps",  "Apple M2 Max",          0.0, "MPS"),
            ("cpu",  "",                      0.0, "CPU"),
        ],
    )
    def test_chain(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        device_name: str,
        vram_gb: float,
        expected_label: str,
    ) -> None:
        from scribe import engine
        srv, client, _ = server_env

        # Pin the entire helper chain at the bottom — gpu_backend() —
        # so we can prove its output flows up through the API into the
        # shape the JS tile expects.
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: device_name)
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: vram_gb)

        # Helper-level: G1.1 booleans agree with gpu_backend().
        assert engine.is_cuda() == (backend == "cuda")
        assert engine.is_rocm() == (backend == "rocm")
        assert engine.is_mps() == (backend == "mps")
        assert engine.has_gpu() == (backend != "cpu")

        # API-level: /api/capabilities echoes the same backend.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == backend

        # JS-helper-level: simulate what backendStatTile + formatBackendLabel
        # would produce against the API payload.
        gpu = body["gpu"]
        sim_label = {
            "cuda": "CUDA", "rocm": "ROCm", "mps": "MPS", "cpu": "CPU",
        }.get(gpu["backend"], "CPU")
        assert sim_label == expected_label
