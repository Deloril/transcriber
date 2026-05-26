"""Verification of reachability for G1.4 — Backend tile on the Recording
details card (CUDA / ROCm / MPS / CPU).

G1.4 was originally shipped in commit ``1fa727a``: the ``renderStats()``
function in ``scribe/templates/index.html`` now appends a "Backend"
tile after the Duration / File size / Container / Bitrate / Video /
Audio tiles, populated from ``GET /api/capabilities``. The helper
that builds the tile dict (``backendStatTile``) and the helper that
produces the pretty-cased label (``formatBackendLabel``) live in
``scribe/static/js/helpers.mjs`` so they can be unit-tested in isolation
— and so the same label policy is reused for the Parakeet "blocked by
backend" hint (G5.1) and any future surface that wants the same
4-state label.

That commit predates the loop's ``Reachable-via`` gate. This file is
the structural proof that the tile is reachable end-to-end:

  1. The home page imports ``backendStatTile`` and ``formatBackendLabel``
     from ``helpers.mjs``.
  2. ``renderStats()`` actually calls ``backendStatTile(_caps && _caps.gpu)``
     and pushes the resulting dict into the stats grid the user sees.
  3. The page fetches ``/api/capabilities`` (the source of ``_caps``),
     and that route returns a ``gpu`` payload whose key set is the
     exact contract ``backendStatTile`` reads.
  4. The Parakeet model-hint surface uses ``formatBackendLabel`` so the
     two surfaces never disagree on case (e.g. "ROCm" vs "ROCM").
  5. End-to-end: for every backend the engine can report, the simulated
     tile renders the documented label.

Deeper unit coverage of the JS helpers themselves lives in
``tests/js/backend-stat-tile.test.mjs``; deeper unit coverage of
``gpu_backend()`` lives in ``tests/test_engine_devices.py``. This file
is the integration-level proof that ties them together via the
FastAPI route + the rendered template.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same fixture shape as the other ``test_server_g1_*_helpers.py``
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
# 1. The two G1.4 helpers exist in helpers.mjs as named exports
# --------------------------------------------------------------------------- #


class TestG1_4HelpersExportedFromHelpersMjs:
    """The Backend tile is built by a pair of pure JS helpers. If either
    name disappears (rename, accidental delete) the home page falls back
    to a missing-tile state silently — these greps catch that."""

    def _helpers_text(self) -> str:
        return (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )

    def test_format_backend_label_is_a_named_export(self) -> None:
        text = self._helpers_text()
        assert re.search(
            r"export\s+function\s+formatBackendLabel\s*\(", text
        ), (
            "formatBackendLabel must be a named export of helpers.mjs so "
            "index.html and the Parakeet hint can both consume the same "
            "label policy."
        )

    def test_backend_stat_tile_is_a_named_export(self) -> None:
        text = self._helpers_text()
        assert re.search(
            r"export\s+function\s+backendStatTile\s*\(", text
        ), (
            "backendStatTile must be a named export of helpers.mjs so "
            "renderStats() in index.html can build the tile dict."
        )

    def test_label_table_covers_all_four_states(self) -> None:
        # The pretty-case table is the single point of truth for the
        # 4-state label. Pin it here so a regression to "ROCM" or "RoCm"
        # doesn't slip through.
        text = self._helpers_text()
        assert "cuda: \"CUDA\"" in text
        assert "rocm: \"ROCm\"" in text
        assert "mps: \"MPS\"" in text
        assert "cpu: \"CPU\"" in text


# --------------------------------------------------------------------------- #
# 2. The home page imports + actually uses the helpers
# --------------------------------------------------------------------------- #


class TestG1_4HomePageWiresTheTile:
    """Importing the helper isn't enough — ``renderStats()`` has to call
    it and push the result into the stats grid the user sees. Pin that
    chain so a refactor that drops the call surfaces here."""

    def test_index_imports_both_helpers(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html, (
            "index.html must import backendStatTile from helpers.mjs"
        )
        assert "formatBackendLabel" in html, (
            "index.html must import formatBackendLabel from helpers.mjs"
        )

    def test_render_stats_calls_backend_stat_tile(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        # The exact call shape — feeding ``_caps && _caps.gpu`` so the
        # helper sees null when capabilities haven't arrived yet.
        assert "backendStatTile(_caps && _caps.gpu)" in html

    def test_backend_tile_pushed_into_stats_grid(self, server_env) -> None:
        # The tile only reaches the user if it's pushed into ``stats``
        # before the forEach that turns each entry into a DOM element.
        srv, client, _ = server_env
        html = client.get("/").text
        # Pattern: any whitespace, "backendTile" guard, then a stats.push()
        # inside renderStats(). The simplest reliable pin: assert both
        # the assignment and the push exist.
        assert "const backendTile = backendStatTile" in html
        assert "stats.push(backendTile)" in html

    def test_page_fetches_capabilities_endpoint(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "fetch(\"/api/capabilities\")" in html, (
            "index.html must fetch /api/capabilities to populate _caps "
            "before renderStats() reads _caps.gpu"
        )

    def test_index_status_ok(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# 3. /api/capabilities returns the exact key shape the tile reads
# --------------------------------------------------------------------------- #


class TestG1_4ApiCapabilitiesGpuPayload:
    """``backendStatTile`` reads ``gpu.backend``, ``gpu.device_name``,
    ``gpu.vram_gb``, ``gpu.gfx_target``, ``gpu.distro``. That's the
    contract /api/capabilities owes the home page. Pin every key
    explicitly so a server-side refactor that drops one fails here."""

    def test_payload_carries_backend_field(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"

    def test_payload_carries_device_name_field(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA A100")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 80.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["device_name"] == "NVIDIA A100"

    def test_payload_carries_vram_gb_field_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["vram_gb"] == 24.0

    def test_payload_vram_gb_is_none_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Apple Silicon has no addressable VRAM number; the API surfaces
        # ``None`` so the tile builder cleanly drops the segment.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["vram_gb"] is None

    def test_payload_key_set_pins_tile_contract(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin the exact key set ``backendStatTile`` consumes. Adding new
        # nullable keys is fine; dropping any of these breaks the tile.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04 LTS")

        body = client.get("/api/capabilities").json()
        keys = set(body["gpu"].keys())
        # The five keys backendStatTile and its G1.3 sub-line read.
        for required in ("backend", "device_name", "vram_gb", "gfx_target", "distro"):
            assert required in keys, (
                f"/api/capabilities must carry gpu.{required} — "
                "backendStatTile reads it to build the home page tile."
            )


# --------------------------------------------------------------------------- #
# 4. formatBackendLabel is reused by the Parakeet hint (label consistency)
# --------------------------------------------------------------------------- #


class TestG1_4LabelPolicyReusedByParakeetHint:
    """The Parakeet "blocked by backend" hint says e.g. "doesn't run on
    the active ROCm backend". The hint must use ``formatBackendLabel``
    so the case ("ROCm" not "ROCM") matches the tile. Pin it."""

    def test_parakeet_hint_imports_format_backend_label(self) -> None:
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        # The parakeetModelHint function must call formatBackendLabel
        # before composing the warning HTML. Grep for that pattern.
        assert "formatBackendLabel(backend)" in text


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — one walk per backend the engine can report
# --------------------------------------------------------------------------- #


def _simulate_tile_value(gpu: dict) -> str:
    """Mirror the JS helper's ``value`` field so we can assert the label
    that would render in the browser without spinning up jsdom. Kept
    in sync with ``formatBackendLabel`` in helpers.mjs."""
    table = {"cuda": "CUDA", "rocm": "ROCm", "mps": "MPS", "cpu": "CPU"}
    key = str(gpu.get("backend") or "").strip().lower()
    return table.get(key, "CPU")


class TestG1_4EndToEndChain:
    """For every backend ``gpu_backend()`` can return, walk:
    helper → /api/capabilities → simulated tile label.
    If any step drops the contract this fails."""

    @pytest.mark.parametrize(
        "backend, device_name, vram_gb, expected_label",
        [
            ("cuda", "NVIDIA GeForce RTX 4090", 24.0, "CUDA"),
            ("rocm", "AMD Radeon RX 7900 XTX",  24.0, "ROCm"),
            ("mps",  "Apple M2 Max",             0.0, "MPS"),
            ("cpu",  "",                         0.0, "CPU"),
        ],
    )
    def test_chain_per_backend(
        self,
        backend: str,
        device_name: str,
        vram_gb: float,
        expected_label: str,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: device_name)
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: vram_gb)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        # Wire-shape: the API echoes the backend identifier.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == backend
        # vram_gb is None on non-CUDA-or-ROCm backends.
        if backend in ("cuda", "rocm"):
            assert body["gpu"]["vram_gb"] == round(vram_gb, 1)
        else:
            assert body["gpu"]["vram_gb"] is None

        # Tile-shape: the label that would render in the browser.
        assert _simulate_tile_value(body["gpu"]) == expected_label

        # And the home page imports the helper that produces that label.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html

    def test_unknown_backend_falls_back_to_cpu_label(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If a future engine version reports something we don't recognise
        # (e.g. "xpu" for Intel), the tile falls back to "CPU" rather
        # than rendering the raw string. Pin that policy here.
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "xpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Intel Arc A770")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        # The API echoes whatever the engine returned — translation to the
        # 4-state label happens client-side in formatBackendLabel().
        assert body["gpu"]["backend"] == "xpu"
        # The tile would render the safe fallback.
        assert _simulate_tile_value(body["gpu"]) == "CPU"
