"""Verification of reachability for G3.1 — pyannote LSTM dropout
MIOpen workaround surfaced through the user-facing UI.

G3.1 (commit ``6371fe8``) shipped the *patch helper*:
``_patch_pyannote_lstm_dropout`` (and its public alias
``patch_pyannote_lstm_dropout``) which walks a pyannote.audio Pipeline's
sub-modules and forces every ``nn.LSTM.dropout`` to ``0.0`` to avoid
the open MIOpen header bug (pyannote-audio #1995). It fires
automatically from ``run_diarize`` and ``parakeet`` when the active
backend is ROCm; deep unit coverage of the walker / cycle-safety /
container shapes lives in ``tests/test_engine_devices.py``. The
original commit had no user-facing surface — a researcher had no way
to confirm the workaround was in their install before kicking off a
diarization run.

This iteration extends the chain so the home page Recording details
card carries the patch state in the Backend tile sub-line. Two new
fields land on ``GET /api/capabilities``:

  * ``rocm_lstm_patch``              — ``True`` on ROCm (the patch
                                       helper is registered + will
                                       fire on diarization load);
                                       ``None`` on every non-ROCm
                                       backend (the helper short-
                                       circuits there so reporting
                                       "applies" elsewhere would be
                                       misleading).
  * ``rocm_lstm_patch_explanation``  — one-line rationale carrying the
                                       upstream issue id (pyannote-audio
                                       #1995) so a researcher pasting
                                       the line into a support thread
                                       has the searchable keywords.

The home page tile (``backendStatTile`` in ``helpers.mjs``) appends
``"LSTM patched"`` to the sub-line on ROCm when ``rocm_lstm_patch``
is truthy. The CLI (``python -m scribe.devices``) prints the same
state under ``LSTM dropout patch:`` so the two surfaces can never
disagree.

Deeper unit coverage of the patch helper itself lives in
``tests/test_engine_devices.py`` (every walker shape — Pipeline, dict,
list, idempotence, cycle safety, @property safety). The JS-side tile
shape is pinned in ``tests/js/backend-stat-tile.test.mjs``. This file
is the integration-level reachability proof.
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
# 1. The G3.1 helpers exist + return the documented shapes
# --------------------------------------------------------------------------- #


class TestG3_1HelpersExposePatchStatus:
    """Pin the public surface of the helpers ``/api/capabilities`` and
    ``python -m scribe.devices`` both call into. ``patch_pyannote_lstm_dropout``
    has been part of the public API since G3.1 shipped; the two new
    helpers add the *status* projection used by the UI surface."""

    def test_engine_exposes_patch_status_helper(self) -> None:
        from scribe.engine import rocm_lstm_dropout_patch_active

        # Always callable, returns a bool (True on ROCm, False elsewhere).
        result = rocm_lstm_dropout_patch_active()
        assert isinstance(result, bool)

    def test_engine_exposes_patch_explanation_helper(self) -> None:
        from scribe.engine import rocm_lstm_dropout_patch_explanation

        text = rocm_lstm_dropout_patch_explanation()
        assert isinstance(text, str)
        # The explanation must reference the upstream issue id so a
        # researcher pasting it into a support thread has searchable
        # keywords (the loop's whole reason for surfacing this is so
        # users have something concrete to attach to a #1995 ticket).
        assert "1995" in text
        assert "MIOpen" in text or "pyannote" in text

    def test_active_helper_returns_true_on_rocm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        assert engine.rocm_lstm_dropout_patch_active() is True

    @pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
    def test_active_helper_returns_false_off_rocm(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        from scribe import engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        assert engine.rocm_lstm_dropout_patch_active() is False

    def test_helpers_re_exported_from_top_level_scribe(self) -> None:
        # G2.1 / G2.3 / G2.2 didn't re-export their helpers at the
        # top level (they're internal to scribe.devices / scribe.server),
        # but G3.1's two helpers SHOULD be reachable from ``scribe``
        # because external callers (smoke-test scripts, third-party
        # integrations, lab admins reading device probes) will want to
        # call them without reaching into a sub-module — same rationale
        # as ``patch_pyannote_lstm_dropout`` itself, which got a public
        # alias in the original G3.1 commit for exactly this reason.
        import scribe

        assert hasattr(scribe, "rocm_lstm_dropout_patch_active")
        assert hasattr(scribe, "rocm_lstm_dropout_patch_explanation")
        assert "rocm_lstm_dropout_patch_active" in scribe.__all__
        assert "rocm_lstm_dropout_patch_explanation" in scribe.__all__


# --------------------------------------------------------------------------- #
# 2. The /api/capabilities response carries the new fields
# --------------------------------------------------------------------------- #


def _stub_rocm_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force /api/capabilities into the ROCm branch with deterministic
    distro classification. The patch-status fields piggy-back on the
    same branch so we can reuse the same set-up the G2.x tests use."""
    from scribe import engine, devices, rocm_install

    monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
    monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
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


class TestG3_1ApiCapabilitiesCarriesPatchStatus:
    """Pin the exact JSON shape the helpers/UI consume."""

    def test_rocm_reports_patch_active_and_explanation(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"
        # Boolean flag — JS reads via ``=== true`` so the field must be
        # an actual bool, not a truthy string.
        assert body["gpu"]["rocm_lstm_patch"] is True
        # Explanation references the upstream issue id so a researcher
        # pasting the JSON into a support thread has the searchable
        # keyword chain (pyannote-audio #1995 → MIOpen → hiprand_xorwow).
        text = body["gpu"]["rocm_lstm_patch_explanation"]
        assert isinstance(text, str)
        assert "1995" in text

    def test_capabilities_omits_patch_on_cuda(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cuda"
        # ``None`` (not False) on non-ROCm — distinguishes "doesn't
        # apply on this backend" from "applies but disabled".
        assert body["gpu"]["rocm_lstm_patch"] is None
        assert body["gpu"]["rocm_lstm_patch_explanation"] is None

    def test_capabilities_omits_patch_on_mps(
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
        assert body["gpu"]["rocm_lstm_patch"] is None
        assert body["gpu"]["rocm_lstm_patch_explanation"] is None

    def test_capabilities_omits_patch_on_cpu(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "cpu"
        assert body["gpu"]["rocm_lstm_patch"] is None
        assert body["gpu"]["rocm_lstm_patch_explanation"] is None

    def test_capabilities_swallows_helper_exception(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If ``rocm_lstm_dropout_patch_active`` raises (deeply
        # unexpected — the helper just calls is_rocm() which is
        # bulletproof, but defensive symmetry with the other
        # ROCm-only fields), the API must still respond — fall to
        # None rather than 500.
        from scribe import engine, devices, rocm_install

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: "gfx1100")
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")
        monkeypatch.setattr(rocm_install, "pinned_ct2_rocm_version", lambda: "4.7.2")
        monkeypatch.setattr(rocm_install, "installed_ct2_version", lambda: "4.7.2")

        def _boom(*a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("torch.version.hip probe failed")

        monkeypatch.setattr(engine, "rocm_lstm_dropout_patch_active", _boom)

        r = client.get("/api/capabilities")
        assert r.status_code == 200
        body = r.json()
        # Defensive fallback: None, not a partial value.
        assert body["gpu"]["rocm_lstm_patch"] is None
        assert body["gpu"]["rocm_lstm_patch_explanation"] is None
        # The other ROCm-only fields still populate — the new fields
        # are additive and don't break the rest of the surface.
        assert body["gpu"]["ct2_rocm_pin"] == "4.7.2"

    def test_capabilities_payload_shape_includes_new_fields(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The home page tile reads the gpu payload by key. Pin the full
        # key set so a future response-shape drift fails this test
        # loudly — additive contract extension guard.
        #
        # The list grows as G-features wire new fields through to the
        # tile (G4.1 added the three ``rocm_allocator_*`` fields).
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
        }


# --------------------------------------------------------------------------- #
# 3. The CLI surface prints the LSTM dropout patch line on ROCm
# --------------------------------------------------------------------------- #


class TestG3_1DevicesCliSurface:
    """``python -m scribe.devices`` prints the ``LSTM dropout patch:``
    line on ROCm. The wire-up to /api/capabilities reuses the same
    helpers so this is a regression guard against the CLI losing the
    line if a future refactor moves the helper somewhere new."""

    def test_cli_prints_lstm_patch_line_on_rocm(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine
        from scribe.devices import main as devices_main

        # The CLI imports gpu_backend / rocm_lstm_dropout_patch_active /
        # rocm_lstm_dropout_patch_explanation directly at module load
        # time, so we patch the symbols on ``devices`` (the bound names)
        # rather than on ``engine`` (the source).
        monkeypatch.setattr(devices, "gpu_backend", lambda: "rocm")
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
        monkeypatch.setattr(devices, "rocm_lstm_dropout_patch_active", lambda: True)

        devices_main()
        out = capsys.readouterr().out
        assert "LSTM dropout patch:" in out
        assert "active" in out
        # The line includes the explanation in parentheses so a
        # researcher reading the CLI output once knows what the patch
        # does without spelunking through the source.
        assert "1995" in out

    def test_cli_omits_lstm_patch_line_on_cuda(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CUDA boxes don't carry the patch line — it would just be
        # noise. The route gates on backend == "rocm", so does the
        # CLI line.
        from scribe import devices, engine
        from scribe.devices import main as devices_main

        monkeypatch.setattr(devices, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name", lambda i: "NVIDIA RTX 4090"
        )
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.version, "cuda", "12.4", raising=False)
        monkeypatch.setattr(devices.torch.version, "hip", None, raising=False)
        monkeypatch.setattr(devices, "_linux_distro", lambda: "Ubuntu 24.04.4 LTS")

        devices_main()
        out = capsys.readouterr().out
        assert "LSTM dropout patch:" not in out


# --------------------------------------------------------------------------- #
# 4. The home page imports the helper + renders the tile / sub-line glue
# --------------------------------------------------------------------------- #


class TestG3_1HomePageBackendTileShowsLstmPatch:
    """The home page consumes ``capabilities.gpu`` directly via
    ``backendStatTile`` from ``helpers.mjs``. G3.1 added the
    ``rocm_lstm_patch`` field which the helper turns into a
    ``"LSTM patched"`` sub-line segment when the ROCm payload carries
    a truthy value."""

    def test_index_imports_backend_tile_helper(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html

    def test_helpers_mjs_reads_rocm_lstm_patch_field(self) -> None:
        # Grep the JS source — the sub-line builder must read
        # gpu.rocm_lstm_patch so the tile can show "LSTM patched"
        # on ROCm boxes.
        text = (Path("scribe") / "static" / "js" / "helpers.mjs").read_text(
            encoding="utf-8"
        )
        assert "rocm_lstm_patch" in text, (
            "backendStatTile() must read gpu.rocm_lstm_patch so the home "
            "page tile sub-line can show the pyannote-audio #1995 "
            "MIOpen workaround state on ROCm boxes."
        )
        assert "LSTM patched" in text, (
            "backendStatTile() must render the literal 'LSTM patched' "
            "segment for the G3.1 sub-line."
        )

    def test_index_renders_against_rocm_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: the page imports backendStatTile, fetches
        # /api/capabilities, the API now carries rocm_lstm_patch, and
        # the helper turns ``true`` into the sub-line segment.
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)

        # Page renders + imports the helper + fetches capabilities.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html
        assert "fetch(\"/api/capabilities\")" in html

        # API payload carries the patch state verbatim.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["rocm_lstm_patch"] is True
        assert body["gpu"]["rocm_lstm_patch_explanation"]


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — patch helper → API → JS helper input shape
# --------------------------------------------------------------------------- #


class TestG3_1EndToEndChain:
    """Walk the full chain for each realistic backend. If any link
    drops the patch state, this fails."""

    def test_rocm_chain_carries_patch_active(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Realistic happy path: RX 7900 XTX on Ubuntu 24.04. The tile
        # sub-line should carry "LSTM patched" alongside everything
        # else.
        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        gpu = body["gpu"]
        assert gpu["backend"] == "rocm"
        assert gpu["rocm_lstm_patch"] is True
        # Explanation isn't empty + contains the searchable issue id.
        assert "1995" in gpu["rocm_lstm_patch_explanation"]

    @pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
    def test_non_rocm_chain_drops_patch_state(
        self, server_env, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        # CUDA / MPS / CPU: ``rocm_lstm_patch`` collapses to None so the
        # JS knows the field doesn't apply on this backend. This is the
        # contract that lets ``backendStatTile`` distinguish "patch
        # absent" from "patch doesn't apply here".
        from scribe import engine, devices

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: f"Device-{backend}")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        monkeypatch.setattr(engine, "gpu_arch_name", lambda: None)
        monkeypatch.setattr(devices, "_linux_distro", lambda: None)

        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == backend
        assert body["gpu"]["rocm_lstm_patch"] is None
        assert body["gpu"]["rocm_lstm_patch_explanation"] is None

    def test_explanation_is_consistent_across_surfaces(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CLI prints the same explanation the API returns. Pin the
        # source-of-truth: ``rocm_lstm_dropout_patch_explanation`` is
        # called by both surfaces, so the strings must match.
        from scribe.engine import rocm_lstm_dropout_patch_explanation

        srv, client, _ = server_env
        _stub_rocm_branch(monkeypatch)
        body = client.get("/api/capabilities").json()
        api_text = body["gpu"]["rocm_lstm_patch_explanation"]
        helper_text = rocm_lstm_dropout_patch_explanation()
        assert api_text == helper_text
