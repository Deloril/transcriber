"""Verification of reachability for G5.1 — Parakeet visibility on AMD
ROCm.

NVIDIA NeMo (the runtime that loads NVIDIA Parakeet) is CUDA-only — it
has no AMD/ROCm port and no community fork. G5.1 (commit ``5e817a0``)
ships two pure JavaScript helpers in ``scribe/static/js/helpers.mjs``
that decide what the upload page does on each backend:

  * ``shouldHideParakeetOptgroup(backend)`` — returns ``true`` only for
    ROCm. The Parakeet optgroup is hidden entirely there because the
    model literally cannot load.
  * ``parakeetModelHint({ model, backend, parakeet })`` — returns a
    structured ``{ kind, tone, html }`` decision the page renders as
    the model-hint strip. ``"blocked"`` / ``"missing"`` /
    ``"info"`` / ``"none"`` cover the four states the user can land in.

The original commit shipped with template wiring + JS unit coverage in
``tests/js/parakeet-visibility.test.mjs`` (17 cases) and a slim Python
suite in ``tests/test_parakeet_visibility.py`` (10 cases) but predates
the loop's ``Reachable-via`` gate, so it left no audit trail of where
the visibility decision surfaces in user-facing output. This file
consolidates the structural reachability proof so the next iteration
can confirm the surface is wired without re-deriving the chain from
scratch.

Coverage matrix:

  1. Both helpers are named exports of ``helpers.mjs`` (rename guard).
  2. The upload page imports + actually calls each helper.
  3. ``GET /api/capabilities`` returns the exact contract the helpers
     consume — ``parakeet.{available, installed, blocked_by_backend,
     error}`` and ``gpu.backend``.
  4. The blocked-by-backend flag flips with the active backend the way
     the helpers expect (rocm → blocked, cuda/cpu → ok, mps → blocked
     but optgroup stays visible).
  5. End-to-end: for every backend the engine can report, walk
     /api/capabilities → simulated ``shouldHideParakeetOptgroup`` →
     simulated ``parakeetModelHint`` and assert the documented kind
     surfaces.

Deeper unit coverage of the JS helpers themselves lives in
``tests/js/parakeet-visibility.test.mjs``. Deeper unit coverage of
``gpu_backend()`` lives in ``tests/test_engine_devices.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


HELPERS_MJS = Path("scribe") / "static" / "js" / "helpers.mjs"
INDEX_HTML = Path("scribe") / "templates" / "index.html"


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same fixture shape as the other ``test_server_g*_helpers.py``
    files. Copied so this file is self-contained."""
    from scribe import server as srv
    from scribe import parakeet

    # Force NeMo to look importable so the parakeet sub-payload is
    # populated. Without this the gating tests would only reflect
    # "NeMo isn't installed", not the backend gate G5.1 surfaces.
    monkeypatch.setattr(parakeet, "_NEMO_AVAILABLE", True)
    monkeypatch.setattr(parakeet, "_IMPORT_ERROR", None)

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
# 1. The two G5.1 helpers exist in helpers.mjs as named exports
# --------------------------------------------------------------------------- #


class TestG5_1HelpersExportedFromHelpersMjs:
    """If either helper name disappears (rename, accidental delete) the
    upload page falls back to a tantalising-but-broken Parakeet option
    on AMD. These greps catch that."""

    def _helpers_text(self) -> str:
        return HELPERS_MJS.read_text(encoding="utf-8")

    def test_shouldHideParakeetOptgroup_is_a_named_export(self) -> None:
        text = self._helpers_text()
        assert re.search(
            r"export\s+function\s+shouldHideParakeetOptgroup\s*\(", text
        ), (
            "shouldHideParakeetOptgroup must be a named export of "
            "helpers.mjs so index.html can hide the optgroup on ROCm."
        )

    def test_parakeetModelHint_is_a_named_export(self) -> None:
        text = self._helpers_text()
        assert re.search(
            r"export\s+function\s+parakeetModelHint\s*\(", text
        ), (
            "parakeetModelHint must be a named export of helpers.mjs so "
            "index.html can render the model-hint strip from its decision."
        )

    def test_blocked_branch_uses_format_backend_label(self) -> None:
        # The 'blocked' branch must run the backend through
        # formatBackendLabel so the hint reads "ROCm" not "ROCM" — and
        # so the hint and the home page Backend tile (G1.4) agree.
        text = self._helpers_text()
        assert "formatBackendLabel(backend)" in text, (
            "parakeetModelHint must call formatBackendLabel(backend) so "
            "the hint label matches the home page tile (G1.4)."
        )

    def test_blocked_branch_escapes_backend_label(self) -> None:
        # Defence in depth: the backend label is server-controlled so
        # it should still be HTML-escaped before going into innerHTML.
        text = self._helpers_text()
        # The blocked-branch HTML literal includes ``escapeHtml(label)``.
        assert "escapeHtml(label)" in text, (
            "The blocked branch must escapeHtml(label) before composing "
            "the HTML — the hint goes into innerHTML."
        )


# --------------------------------------------------------------------------- #
# 2. The upload page imports + actually calls the helpers
# --------------------------------------------------------------------------- #


class TestG5_1UploadPageWiresTheHelpers:
    """Importing isn't enough — both helpers have to actually be called
    so the optgroup hide and the hint strip are reachable from the
    user-facing page. Pin both call sites."""

    def test_index_imports_shouldHideParakeetOptgroup(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "shouldHideParakeetOptgroup" in html, (
            "index.html must import shouldHideParakeetOptgroup from "
            "helpers.mjs"
        )

    def test_index_imports_parakeetModelHint(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "parakeetModelHint" in html, (
            "index.html must import parakeetModelHint from helpers.mjs"
        )

    def test_index_calls_shouldHideParakeetOptgroup(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        # The decision call has to go through the helper, not duplicate
        # the rocm string match inline.
        assert "shouldHideParakeetOptgroup(" in html

    def test_index_calls_parakeetModelHint(self, server_env) -> None:
        srv, client, _ = server_env
        html = client.get("/").text
        assert "parakeetModelHint(" in html

    def test_no_inline_rocm_string_match(self, server_env) -> None:
        # The previous implementation had `=== "rocm"` inline. Pin that
        # the duplicate doesn't sneak back in.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'backend === "rocm"' not in html

    def test_optgroup_present_in_default_render(self, server_env) -> None:
        # Server-side rendering is backend-agnostic; the JS hides the
        # optgroup at runtime on ROCm. Without the optgroup in the HTML
        # there'd be nothing for the helper to hide.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'optgroup label="NVIDIA Parakeet' in html
        assert "nvidia/parakeet-tdt-0.6b-v2" in html

    def test_optgroup_query_targets_label(self, server_env) -> None:
        # Pin the runtime querySelector that locates the optgroup. If
        # either side of this contract drifts (HTML label or JS
        # selector), the hide-on-ROCm behaviour silently breaks.
        srv, client, _ = server_env
        html = client.get("/").text
        assert 'querySelector(\'optgroup[label*="Parakeet"]\')' in html

    def test_index_status_ok(self, server_env) -> None:
        srv, client, _ = server_env
        r = client.get("/")
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# 3. /api/capabilities carries the full parakeet sub-payload contract
# --------------------------------------------------------------------------- #


class TestG5_1CapabilitiesParakeetPayload:
    """``parakeetModelHint`` reads ``parakeet.blocked_by_backend``,
    ``parakeet.available``, and ``parakeet.installed``. The /api/capabilities
    endpoint is the only source of those fields, so the contract is
    pinned here."""

    def test_payload_carries_blocked_by_backend(self, server_env) -> None:
        srv, client, _ = server_env
        body = client.get("/api/capabilities").json()
        assert "parakeet" in body
        assert "blocked_by_backend" in body["parakeet"]

    def test_payload_carries_available(self, server_env) -> None:
        srv, client, _ = server_env
        body = client.get("/api/capabilities").json()
        assert "available" in body["parakeet"]

    def test_payload_carries_installed(self, server_env) -> None:
        srv, client, _ = server_env
        body = client.get("/api/capabilities").json()
        assert "installed" in body["parakeet"]

    def test_payload_carries_error_field(self, server_env) -> None:
        # parakeetModelHint reads `p.error` indirectly (the renderer uses
        # it for diagnostics on the missing branch). Pin its presence.
        srv, client, _ = server_env
        body = client.get("/api/capabilities").json()
        assert "error" in body["parakeet"]

    def test_payload_key_set_pins_helper_contract(self, server_env) -> None:
        # The four keys the JS helpers read. Adding new nullable keys is
        # fine; dropping any of these breaks the hint.
        srv, client, _ = server_env
        body = client.get("/api/capabilities").json()
        keys = set(body["parakeet"].keys())
        for required in ("available", "installed", "error", "blocked_by_backend"):
            assert required in keys, (
                f"/api/capabilities must carry parakeet.{required} — "
                "parakeetModelHint reads it to decide the hint kind."
            )


# --------------------------------------------------------------------------- #
# 4. blocked_by_backend flips with backend the way the helpers expect
# --------------------------------------------------------------------------- #


class TestG5_1BlockedByBackendMatchesHelperContract:
    """The pure helpers in helpers.mjs assume the server flips
    ``blocked_by_backend=True`` exactly when NeMo can't run on the
    active backend (rocm + mps). CUDA / CPU keep it False. Pin the
    backend-by-backend behaviour so the JS helpers and the server
    can never disagree."""

    def test_rocm_sets_blocked_by_backend_true(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "rocm")
        monkeypatch.setattr(
            engine, "_gpu_device_name", lambda: "AMD Radeon RX 7900 XTX"
        )
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        body = client.get("/api/capabilities").json()
        assert body["parakeet"]["blocked_by_backend"] is True
        assert body["parakeet"]["available"] is False
        # NeMo is installed; only the backend is wrong.
        assert body["parakeet"]["installed"] is True

    def test_mps_sets_blocked_by_backend_true(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # MPS also can't run NeMo. Server flags it but the optgroup
        # stays visible (only ROCm gets the full hide); the model-hint
        # tells the user instead.
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "mps")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Apple M2 Max")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        body = client.get("/api/capabilities").json()
        assert body["parakeet"]["blocked_by_backend"] is True

    def test_cuda_sets_blocked_by_backend_false(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "NVIDIA RTX 4090")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 24.0)
        body = client.get("/api/capabilities").json()
        assert body["parakeet"]["blocked_by_backend"] is False
        assert body["parakeet"]["available"] is True

    def test_cpu_sets_blocked_by_backend_false(
        self, server_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CPU is unusably slow but NeMo *can* load — keep blocked False
        # so the hint reads "GPU recommended" rather than "blocked".
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "cpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)
        body = client.get("/api/capabilities").json()
        assert body["parakeet"]["blocked_by_backend"] is False


# --------------------------------------------------------------------------- #
# 5. End-to-end chain — simulate the JS decisions per backend
# --------------------------------------------------------------------------- #


def _simulate_should_hide(backend: str | None) -> bool:
    """Mirror ``shouldHideParakeetOptgroup`` from helpers.mjs without
    spinning up jsdom. Kept in sync intentionally — if either side
    changes, the matching JS test in
    ``tests/js/parakeet-visibility.test.mjs`` also has to change, so
    the divergence surfaces immediately."""
    return str(backend or "").strip().lower() == "rocm"


def _simulate_hint_kind(model: str | None, parakeet_payload: dict) -> str:
    """Mirror ``parakeetModelHint`` enough to assert which ``kind``
    branch fires for a given (model, backend, parakeet payload). Same
    in-sync caveat as ``_simulate_should_hide``."""
    is_parakeet = "parakeet" in str(model or "").lower()
    if not is_parakeet:
        return "none"
    p = parakeet_payload or {}
    if p.get("blocked_by_backend"):
        return "blocked"
    if not p.get("available") and not p.get("installed"):
        return "missing"
    return "info"


class TestG5_1EndToEndChainPerBackend:
    """For every backend the engine can report, walk the chain:
    /api/capabilities → simulated shouldHideParakeetOptgroup →
    simulated parakeetModelHint(model='parakeet'). If any link in
    the chain breaks the contract this fails."""

    @pytest.mark.parametrize(
        "backend, expect_optgroup_hidden, expect_hint_kind_for_parakeet",
        [
            ("cuda", False, "info"),
            ("rocm", True,  "blocked"),
            ("mps",  False, "blocked"),  # optgroup visible, hint blocks
            ("cpu",  False, "info"),
        ],
    )
    def test_chain_per_backend(
        self,
        backend: str,
        expect_optgroup_hidden: bool,
        expect_hint_kind_for_parakeet: str,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: backend)
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 0.0)

        body = client.get("/api/capabilities").json()
        gpu_backend = body["gpu"]["backend"]
        parakeet = body["parakeet"]

        assert gpu_backend == backend
        assert _simulate_should_hide(gpu_backend) is expect_optgroup_hidden
        kind = _simulate_hint_kind(
            "nvidia/parakeet-tdt-0.6b-v2", parakeet
        )
        assert kind == expect_hint_kind_for_parakeet

        # Whisper-on-anything always gets ``kind="none"`` (hint hidden).
        assert _simulate_hint_kind("large-v3", parakeet) == "none"

    def test_unknown_backend_does_not_hide_optgroup(
        self,
        server_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If a future engine version reports something we don't
        # recognise (e.g. "xpu" for Intel), the hide rule is
        # narrowly ROCm-only — don't accidentally hide on
        # unknown-but-not-ROCm backends.
        from scribe import engine

        srv, client, _ = server_env
        monkeypatch.setattr(engine, "gpu_backend", lambda: "xpu")
        monkeypatch.setattr(engine, "_gpu_device_name", lambda: "Intel Arc A770")
        monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: 16.0)
        body = client.get("/api/capabilities").json()
        assert body["gpu"]["backend"] == "xpu"
        assert _simulate_should_hide(body["gpu"]["backend"]) is False
