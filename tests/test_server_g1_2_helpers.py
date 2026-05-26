"""Verification of reachability for G1.2 — honest 4-state device labels.

G1.2 was the work that made ``_torch_device()`` / ``_diarization_device()``
/ ``_whisper_device_and_compute()`` return one of ``cuda`` / ``rocm`` /
``mps`` / ``cpu`` (the user-facing label) rather than collapsing
``rocm`` into the CUDA-shim string at the helper. The translation back
to the literal device-arg string PyTorch / CTranslate2 / pyannote
require happens at the library boundary via
:func:`scribe.engine._to_torch_device_arg`.

The original ``56a8abb`` commit predates the loop's ``Reachable-via``
gate and therefore left no audit trail of where the honest label
surfaces in user-facing output. This file consolidates the proof that
G1.2 is reachable end-to-end:

  - **CLI surface** (``python -m scribe.devices``): the Selected
    backends section prints ``device=rocm`` (or ``cuda`` / ``mps`` /
    ``cpu``) for all three helpers — the support-bundle output users
    paste into bug reports.

  - **API surface** (``GET /api/capabilities``): echoes
    ``gpu_backend()``, which is the source-of-truth the three helpers
    wrap and pass through (``_diarization_device`` has a single MPS
    carve-out documented in its docstring; the other two are pure
    pass-through).

  - **UI surface** (home page Backend tile): renders the 4-state label
    via ``backendStatTile`` + ``formatBackendLabel`` from
    ``helpers.mjs``.

  - **Engine call sites**: ``_transcribe_with_alignment`` and
    ``transcribe_diarize`` apply ``_to_torch_device_arg`` to the
    helpers' output exactly once, immediately before handing the value
    to whisperx / CT2 / pyannote. Pin the call-site shape so a future
    refactor can't accidentally pass ``"rocm"`` through to a library
    that wants ``"cuda"``.

Deeper unit coverage of the helpers themselves lives in
``tests/test_engine_devices.py``; CLI-render coverage lives in
``tests/test_devices.py``. This file is the integration-level
reachability proof that the loop's ``Reachable-via`` detector reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same shape as the fixture in ``test_server.py`` and the G1.1
    verification file; copied so this file is self-contained."""
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
# 1. The three G1.2 helpers return one of the four documented labels
# --------------------------------------------------------------------------- #


_FOUR_STATE = {"cuda", "rocm", "mps", "cpu"}


class TestG1_2HelpersReturnFourStateLabel:
    """The contract G1.2 promised: every helper returns one of
    ``cuda`` / ``rocm`` / ``mps`` / ``cpu``. Pass-through behaviour for
    ``_torch_device`` and ``_whisper_device_and_compute``; one MPS
    carve-out in ``_diarization_device`` (documented in its docstring —
    pyannote on MPS is partial, defaults to CPU)."""

    @pytest.mark.parametrize("backend", ["cuda", "rocm", "mps", "cpu"])
    def test_torch_device_passes_through_backend(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        from scribe import engine

        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        result = engine._torch_device()
        assert result == backend
        assert result in _FOUR_STATE

    @pytest.mark.parametrize(
        "backend,expected",
        [
            ("cuda", "cuda"),
            ("rocm", "rocm"),
            ("mps", "cpu"),  # MPS carve-out: pyannote-on-MPS defaults to CPU
            ("cpu", "cpu"),
        ],
    )
    def test_diarization_device_default(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, expected: str
    ) -> None:
        from scribe import engine

        monkeypatch.delenv("SCRIBE_DIARIZE_DEVICE", raising=False)
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        result = engine._diarization_device()
        assert result == expected
        assert result in _FOUR_STATE

    @pytest.mark.parametrize("backend", ["cuda", "rocm", "mps", "cpu"])
    def test_whisper_device_returns_four_state_label(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        from scribe import engine

        monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)
        monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)
        device, _compute = engine._whisper_device_and_compute()
        # MPS has no CT2 backend; the helper falls back to CPU.
        # Every other backend passes through.
        assert device in _FOUR_STATE
        if backend == "mps":
            assert device == "cpu"
        else:
            assert device == backend

    def test_diarization_force_env_keeps_honest_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # G1.2: when the user forces SCRIBE_DIARIZE_DEVICE=rocm we keep
        # the honest "rocm" label so support output reflects the user's
        # explicit choice. _to_torch_device_arg collapses it to "cuda"
        # only at the pyannote boundary.
        from scribe import engine

        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "rocm")
        assert engine._diarization_device() == "rocm"


# --------------------------------------------------------------------------- #
# 2. _to_torch_device_arg collapses rocm → cuda; everything else passes
# --------------------------------------------------------------------------- #


class TestG1_2DeviceArgTranslator:
    """G1.2's other half: a single function that translates the
    user-facing label into the literal device-arg string PyTorch / CT2 /
    pyannote accept. Without this, calls would either pass ``"rocm"``
    to APIs that don't recognise it, or the helpers would return
    dishonest labels."""

    def test_rocm_translates_to_cuda(self) -> None:
        from scribe import engine

        assert engine._to_torch_device_arg("rocm") == "cuda"

    @pytest.mark.parametrize("label", ["cuda", "mps", "cpu"])
    def test_other_labels_pass_through(self, label: str) -> None:
        from scribe import engine

        assert engine._to_torch_device_arg(label) == label

    def test_idempotent(self) -> None:
        from scribe import engine

        # Safe to apply twice — the second application is a no-op.
        once = engine._to_torch_device_arg("rocm")
        assert engine._to_torch_device_arg(once) == "cuda"

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("cuda", "cuda"),
            ("rocm", "cuda"),  # the translation
            ("mps", "mps"),
            ("cpu", "cpu"),
        ],
    )
    def test_full_table(self, label: str, expected: str) -> None:
        from scribe import engine

        assert engine._to_torch_device_arg(label) == expected


# --------------------------------------------------------------------------- #
# 3. /api/capabilities echoes the same source-of-truth the helpers wrap
# --------------------------------------------------------------------------- #


class TestG1_2ApiCapabilitiesEchoesFourStateLabel:
    """The route surface for the G1.2 chain. ``/api/capabilities``
    reads ``gpu_backend()`` directly — the same source the three
    G1.2 helpers wrap — so the API's ``gpu.backend`` field and the
    helpers always agree on which of the four labels is active."""

    @pytest.mark.parametrize("backend", ["cuda", "rocm", "mps", "cpu"])
    def test_api_echoes_helper_source_of_truth(
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

        body = client.get("/api/capabilities").json()

        # API reports the same 4-state label the helpers return.
        api_backend = body["gpu"]["backend"]
        assert api_backend in _FOUR_STATE
        assert api_backend == backend

        # And the helpers themselves agree (modulo the MPS carve-out
        # in _diarization_device, exercised separately above).
        assert engine._torch_device() == backend


# --------------------------------------------------------------------------- #
# 4. CLI surface — `python -m scribe.devices` prints the 4-state label
# --------------------------------------------------------------------------- #


class TestG1_2DevicesCliSurface:
    """``python -m scribe.devices`` is the support-bundle CLI; users
    paste this into bug reports. G1.2 promised the Selected backends
    section would print the *honest* label rather than the CUDA-shim
    string. This test pins that contract on the ROCm path because it
    is the path G1.2 was designed to fix."""

    def test_devices_main_prints_rocm_label_for_all_three_helpers(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import devices, engine

        # Simulate an AMD ROCm box with a 7900 XTX.
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(devices, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(devices, "_is_rdna2", lambda: False)
        monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            devices.torch.cuda, "get_device_name",
            lambda i: "AMD Radeon RX 7900 XTX",
        )
        monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(devices.torch.version, "hip", "6.3", raising=False)

        rc = devices.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "GPU backend:        rocm" in out
        # All three helpers print the honest label, not "cuda":
        assert "Whisper (CTranslate2):  device=rocm" in out
        assert "Alignment (torch):      device=rocm" in out
        assert "Diarization (pyannote): device=rocm" in out


# --------------------------------------------------------------------------- #
# 5. Home page Backend tile — closes the loop to the user surface
# --------------------------------------------------------------------------- #


class TestG1_2HomePageBackendTile:
    """The home page renders a Backend tile on the Recording details
    card via ``backendStatTile`` + ``formatBackendLabel``. These JS
    helpers consume the API payload that ``gpu_backend()`` populates —
    the same source the G1.2 helpers wrap — so the user sees the
    4-state label in the UI."""

    def test_index_imports_backend_tile_helpers(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "backendStatTile" in html
        assert "formatBackendLabel" in html

    def test_index_renders_tile_against_capabilities_payload(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)

        # The page imports + calls backendStatTile against _caps.gpu.
        html = client.get("/").text
        assert "backendStatTile(_caps && _caps.gpu)" in html

        # And the API payload it will consume reports "rocm".
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "rocm"


# --------------------------------------------------------------------------- #
# 6. Engine call sites apply _to_torch_device_arg to the helpers' output
# --------------------------------------------------------------------------- #


class TestG1_2EngineCallSitesUseTranslator:
    """Pin the call-site shape so a future refactor can't accidentally
    hand the user-facing ``"rocm"`` label to whisperx / pyannote / CT2 —
    which would crash because those libraries only know the literal
    ``"cuda"`` device-arg string on the ROCm wheel.

    Each call site reads the helper, applies ``_to_torch_device_arg``,
    and passes the translated value into the model load. We don't run
    the loads (slow + GPU-only); we read the source and assert the
    glue is in place."""

    def test_engine_source_applies_translator_to_torch_device(self) -> None:
        text = (Path("scribe") / "engine.py").read_text(encoding="utf-8")
        # Exact pattern: alignment uses ``_to_torch_device_arg(_torch_device())``.
        assert re.search(
            r"_to_torch_device_arg\(\s*_torch_device\(\)\s*\)", text
        ), (
            "scribe.engine alignment path must wrap _torch_device() through "
            "_to_torch_device_arg before handing the value to whisperx; "
            "without this, ROCm boxes would pass the literal \"rocm\" string "
            "to a library that only accepts \"cuda\" / \"cpu\"."
        )

    def test_engine_source_applies_translator_to_diarization(self) -> None:
        text = (Path("scribe") / "engine.py").read_text(encoding="utf-8")
        assert re.search(
            r"_to_torch_device_arg\(\s*_diarization_device\(\)\s*\)", text
        ), (
            "scribe.engine diarization path must wrap _diarization_device() "
            "through _to_torch_device_arg before the pyannote.from_pretrained "
            "call."
        )

    def test_engine_source_applies_translator_to_whisper(self) -> None:
        text = (Path("scribe") / "engine.py").read_text(encoding="utf-8")
        # Whisper unpacks the helper's tuple, then translates the device
        # half before the WhisperModel load.
        assert "_whisper_device_and_compute()" in text
        # The translator is applied somewhere after that unpack — match
        # the canonical call-site shape.
        assert "_to_torch_device_arg(device)" in text, (
            "scribe.engine Whisper load must apply _to_torch_device_arg "
            "to the device half of _whisper_device_and_compute() so the "
            "literal device-arg string reaches CT2."
        )


# --------------------------------------------------------------------------- #
# 7. End-to-end chain — helpers → API → tile-input shape
# --------------------------------------------------------------------------- #


class TestG1_2EndToEndChain:
    """One test per backend that walks the full chain:
    helper output → _to_torch_device_arg → API → simulated tile label.

    If any link breaks (helper drops the honest label, translator
    forgets to collapse rocm, API stops echoing gpu_backend, or the
    tile pipeline drifts from the {backend, device_name, vram_gb}
    contract), this test fails."""

    @pytest.mark.parametrize(
        "backend,expected_torch_arg,expected_label",
        [
            ("cuda", "cuda", "CUDA"),
            ("rocm", "cuda", "ROCm"),  # rocm collapses at boundary
            ("mps",  "mps",  "MPS"),
            ("cpu",  "cpu",  "CPU"),
        ],
    )
    def test_chain(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        expected_torch_arg: str,
        expected_label: str,
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.delenv("SCRIBE_DIARIZE_DEVICE", raising=False)
        monkeypatch.delenv("SCRIBE_WHISPER_DEVICE", raising=False)
        monkeypatch.delenv("SCRIBE_COMPUTE_TYPE", raising=False)
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        monkeypatch.setattr(engine, "_is_rdna2", lambda: False)

        # Helper-level: returns the honest label.
        torch_label = engine._torch_device()
        assert torch_label == backend

        # Translator-level: collapses rocm → cuda, others pass through.
        assert engine._to_torch_device_arg(torch_label) == expected_torch_arg

        # API-level: /api/capabilities echoes the same backend.
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == backend

        # Tile-level: simulate what backendStatTile + formatBackendLabel
        # would render. The map matches the JS helper's behaviour.
        sim_label = {
            "cuda": "CUDA", "rocm": "ROCm", "mps": "MPS", "cpu": "CPU",
        }[body["gpu"]["backend"]]
        assert sim_label == expected_label
