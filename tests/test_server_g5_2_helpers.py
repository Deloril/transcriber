"""Verification of reachability for G5.2 — CTranslate2 compute-type
tiering surfaced through the user-facing UI.

G5.2 (commit ``a497895``) shipped the *engine*-side tiering helpers:

  * ``_pick_compute_type(backend, vram_gb, *, rdna2)`` — pure helper that
    resolves a backend × VRAM × architecture matrix into a CT2
    compute-type string.
  * ``_whisper_device_and_compute()`` — engine consumer that calls the
    helper from the model-load path and honours ``SCRIBE_COMPUTE_TYPE``.

The original commit had no Reachable-via line; until this commit a
researcher had no in-app way to confirm the compute-type tier their
next transcription would run at. A 16 GB RX 6800 user couldn't tell
whether Scribe stayed conservative on the Tier-2 RDNA 2 path
(int8_float16) or accidentally leaked fp16; a CUDA RTX-4090 user
couldn't confirm fp16 was active without dropping to a terminal.

This iteration extends the chain so the home page Recording details
card carries the live compute-type tier in the Backend tile sub-line.
One new field lands on ``GET /api/capabilities``:

  * ``whisper_compute_type`` — one of ``"float16"`` (≥8 GB CUDA / RDNA 3 /
    RDNA 4 happy path), ``"int8_float16"`` (<8 GB GPU *or* any RDNA 2
    ROCm box), ``"int8"`` (CPU / MPS, where CT2 has no GPU backend),
    or any ``SCRIBE_COMPUTE_TYPE`` user-override string echoed
    verbatim. Unlike the ROCm-specific G3.x / G4.x fields this one is
    populated on every backend — the question "what precision will
    this run at?" is meaningful even on CPU / MPS.

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"compute <type>"`` to the sub-line. The CLI surface
(``python -m scribe.devices``) already prints
``Whisper (CTranslate2):  device=… compute=<type>`` so the two
surfaces can never disagree (both pull from
``_whisper_device_and_compute``).

Deeper unit coverage of the tiering helper itself lives in
``tests/test_engine_devices.py::TestPickComputeType``. The JS-side
sub-line shape is pinned in ``tests/js/backend-stat-tile.test.mjs``
under ``describe("backendStatTile (G5.2 whisper_compute_type)")``.
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


def _stub_cuda_high_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CUDA + ≥8 GB happy path so the helper picks fp16."""
    from scribe import engine, devices

    monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
    monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
    monkeypatch.setattr(engine, "is_rdna2", lambda: False)
    monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
    # Strip the env-var override so the autodetect path runs.
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


def _stub_cuda_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CUDA + <8 GB laptop path so the helper picks int8_float16."""
    from scribe import engine, devices

    monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA GeForce GTX 1660")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 6.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
    monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
    monkeypatch.setattr(engine, "is_rdna2", lambda: False)
    monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


def _stub_rocm_rdna3(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm + RDNA 3 (RX 7900 XTX 24 GB) — fp16 happy path on AMD."""
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
    monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
    monkeypatch.setattr(engine, "is_rdna2", lambda: False)
    monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
    monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
    monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


def _stub_rocm_rdna2_high_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm + RDNA 2 + 16 GB — Tier-2 conservative int8_float16."""
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 6800")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1030")
    monkeypatch.setattr(engine, "_is_rdna2", lambda: True)
    monkeypatch.setattr(engine, "is_rdna2", lambda: True)
    monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
    monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
    monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


def _stub_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apple Silicon — no CT2 GPU backend; int8 fallback."""
    from scribe import engine, devices

    monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
    monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
    monkeypatch.setattr(engine, "is_rdna2", lambda: False)
    monkeypatch.setattr(devices, "_linux_distro", lambda: None)
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


def _stub_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GPU at all — int8 fallback."""
    from scribe import engine, devices

    monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
    monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
    monkeypatch.setattr(engine, "is_rdna2", lambda: False)
    monkeypatch.setattr(devices, "_linux_distro", lambda: None)
    monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)


# --------------------------------------------------------------------------- #
# 1. The /api/capabilities response carries the new field on every backend
# --------------------------------------------------------------------------- #


class TestG5_2ApiCapabilitiesCarriesComputeType:
    """Pin the exact JSON shape the helpers / UI consume."""

    def test_cuda_high_vram_picks_float16(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["whisper_compute_type"] == "float16"

    def test_cuda_low_vram_picks_int8_float16(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_cuda_low_vram(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        assert body["gpu"]["whisper_compute_type"] == "int8_float16"

    def test_rocm_rdna3_picks_float16(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_rdna3(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["whisper_compute_type"] == "float16"

    def test_rocm_rdna2_stays_conservative_at_high_vram(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # G5.2 Tier-2 conservative path: RX 6800 16 GB still gets the
        # int8_float16 tier because the cub_caching workaround (G4.1)
        # was only validated on int8 quants. This is the single most
        # important cell in the matrix to pin — without it a refactor
        # could silently regress AMD users to a path that crashes on
        # CT2 #2012.
        srv, client, _ = server_env
        _stub_rocm_rdna2_high_vram(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        assert body["gpu"]["whisper_compute_type"] == "int8_float16"

    def test_mps_picks_int8(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_mps(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "mps"
        assert body["gpu"]["whisper_compute_type"] == "int8"

    def test_cpu_picks_int8(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_cpu(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["gpu"]["whisper_compute_type"] == "int8"

    def test_scribe_compute_type_env_override_wins(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The engine honours ``SCRIBE_COMPUTE_TYPE`` unconditionally —
        # the API must echo whatever the user pinned, not the autodetect
        # tier. Researchers running a forced quant for benchmarking need
        # the pin reflected in the UI.
        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)
        monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "int8_float32")
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == "int8_float32"

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If ``_whisper_device_and_compute`` raises (deeply unexpected —
        # the helper just does env var lookups + an arch probe), the API
        # must still respond — fall to None rather than 500. Mirrors
        # the defensive pattern used for every other ROCm field.
        from scribe import engine

        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("compute probe failed")

        monkeypatch.setattr(engine, "_whisper_device_and_compute", _boom)
        # The route caches the helper at import time via ``from .engine
        # import _whisper_device_and_compute`` inside the function body;
        # patching ``engine._whisper_device_and_compute`` directly is
        # what gets imported on the next call.

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"]["whisper_compute_type"] is None
        # The other fields still populate — additive guarantee.
        assert body["gpu"]["backend"] == "cuda"

    def test_capabilities_payload_shape_includes_compute_type(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin the full key set so a future response-shape drift fails
        # this test loudly. Mirrors the precedent in
        # test_server_g4_2_helpers.py.
        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert "whisper_compute_type" in body["gpu"]
        # The field is populated on every backend (no nullability
        # collapse for non-ROCm).
        assert body["gpu"]["whisper_compute_type"] is not None


# --------------------------------------------------------------------------- #
# 2. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG5_2HomePageBackendTileShowsComputeType:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G5.2 added the
    ``whisper_compute_type`` field which the helper turns into a
    ``"compute <type>"`` sub-line segment, populated on every backend
    (CUDA / ROCm / MPS / CPU)."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_whisper_compute_type_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.whisper_compute_type so the tile can show the compute
        # tier on every backend.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "whisper_compute_type" in text, (
            "backendStatTile() must read gpu.whisper_compute_type so "
            "the home page tile sub-line can show the CT2 compute-type "
            "tier the next transcription will run at."
        )
        # The terse segment label is what users will see + grep on; pin
        # the literal so a future helpers refactor can't quietly change
        # it to "ct2 " or "fp16" or similar.
        assert "compute " in text, (
            "backendStatTile() must use the literal 'compute ' prefix "
            "for the G5.2 sub-line segment so a researcher pasting the "
            "tile into a CT2 / faster-whisper bug report can grep it."
        )

    def test_index_renders_against_cuda_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries
        # whisper_compute_type, and the helper turns ``"float16"``
        # into the sub-line segment.
        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the compute type verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == "float16"

    def test_index_renders_against_rocm_rdna2_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The most important AMD case — confirm the conservative
        # int8_float16 tier surfaces all the way to the JSON the home
        # page consumes.
        srv, client, _ = server_env
        _stub_rocm_rdna2_high_vram(monkeypatch)

        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == "int8_float16"


# --------------------------------------------------------------------------- #
# 3. End-to-end chain — engine helper → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG5_2EndToEndChain:
    """Walk the full chain for each realistic state. If any link drops
    the compute type, this fails."""

    def test_engine_helper_and_api_agree_on_cuda_high_vram(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        _stub_cuda_high_vram(monkeypatch)
        # Call the engine helper directly — it's the source of truth.
        _dev, compute_engine = engine._whisper_device_and_compute()
        # Now read the API — the field must match the engine helper.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == compute_engine

    def test_engine_helper_and_api_agree_on_rocm_rdna2(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The Tier-2 conservative path is the single most important
        # cell — pin that the API and the engine helper return the
        # same string for the same hardware.
        from scribe import engine

        srv, client, _ = server_env
        _stub_rocm_rdna2_high_vram(monkeypatch)
        _dev, compute_engine = engine._whisper_device_and_compute()
        assert compute_engine == "int8_float16"
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == compute_engine

    def test_engine_helper_and_api_agree_on_mps(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        _stub_mps(monkeypatch)
        _dev, compute_engine = engine._whisper_device_and_compute()
        assert compute_engine == "int8"
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["whisper_compute_type"] == compute_engine

    def test_compute_type_field_present_on_every_backend(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Unlike the ROCm-specific G3.x / G4.x fields, the compute-type
        # answer is meaningful on every backend (CPU users want to
        # know they're running int8). Walk all four and confirm the
        # field populates everywhere.
        srv, client, _ = server_env

        stubs = [
            ("cuda", _stub_cuda_high_vram, "float16"),
            ("rocm", _stub_rocm_rdna3, "float16"),
            ("mps", _stub_mps, "int8"),
            ("cpu", _stub_cpu, "int8"),
        ]
        for backend_label, stub_fn, expected in stubs:
            stub_fn(monkeypatch)
            body = client.get("/api/capabilities").json()
            assert body["gpu"]["backend"] == backend_label
            assert body["gpu"]["whisper_compute_type"] == expected, (
                f"backend={backend_label!r}: expected compute={expected!r} "
                f"but got {body['gpu']['whisper_compute_type']!r}"
            )
