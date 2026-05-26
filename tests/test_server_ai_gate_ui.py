"""End-to-end reachability tests for F8.10 (AI gate UI on the project AI page).

The data plane shipped in 267e77a (``scribe.ai_gate``) with HTTP
routes ``GET / PUT /api/projects/<pid>/ai/gate``. The original commit
explicitly deferred the UI surface; the only path to *edit* the gate
config from the browser was to navigate to the project settings page,
which exposed a single ``ai_enabled`` checkbox that wrote a different
key.

This file proves the UI surface is wired:

  * GET /projects/<pid>/ai renders a real F8.10 card with editable
    threshold inputs, an override <select>, and the enabled checkbox.
  * The card exposes a Save button, a Force-open button, and a Reset
    button — every one of them carries a stable ``data-test-id`` so
    the page is scriptable.
  * The page-side submit flow (PUT /api/projects/<pid>/ai/gate)
    round-trips the new config — pumped through the existing JSON
    endpoint, which we exercise via TestClient since we don't run a
    headless browser.

Pure-helper coverage for the JS lives in
``tests/js/ai-gate-form.test.mjs``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe.ai_gate import (
    DEFAULT_MIN_CODES,
    DEFAULT_MIN_HAND_CODED_SOURCES,
    GATE_OVERRIDE_AUTO,
    GATE_OVERRIDE_FORCE_OFF,
    GATE_OVERRIDE_FORCE_ON,
    SETTING_KEY_ENABLED,
    SETTING_KEY_MIN_CODES,
    SETTING_KEY_MIN_HAND_CODED_SOURCES,
    SETTING_KEY_OVERRIDE,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _make_project(client: TestClient, name: str = "Gate-UI test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# A. Page renders the form / controls
# --------------------------------------------------------------------------- #


class TestAIGateUIRenders:
    def test_card_present_with_test_feature(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        # The card is identified for both the F-feature audit and as a
        # stable selector for in-page scripts.
        assert 'data-test-id="ai-gate-card"' in r.text
        assert 'data-test-feature="F8.10"' in r.text

    def test_status_row_still_present(self, server_env) -> None:
        # The status row was the original (read-only) F8.10 surface;
        # the editable form is *additive*, not a replacement, so the
        # existing test in test_server_project_ai.py keeps working.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-gate-status"' in r.text
        assert 'data-test-id="ai-gate-progress"' in r.text

    def test_form_inputs_present(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        # Every editable field carries a stable test id so a future
        # Playwright suite (or any browser script) can reach it.
        for tid in (
            "ai-gate-form",
            "ai-gate-min-codes",
            "ai-gate-min-hand",
            "ai-gate-override",
            "ai-gate-enabled",
            "ai-gate-save",
            "ai-gate-force-on",
            "ai-gate-reset",
            "ai-gate-msg",
        ):
            assert f'data-test-id="{tid}"' in r.text, (
                f"missing data-test-id={tid} on /projects/<id>/ai"
            )

    def test_override_options_include_all_three_states(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        for value in (
            GATE_OVERRIDE_AUTO,
            GATE_OVERRIDE_FORCE_ON,
            GATE_OVERRIDE_FORCE_OFF,
        ):
            assert f'value="{value}"' in r.text, (
                f"missing override option {value!r}"
            )

    def test_min_threshold_inputs_have_numeric_bounds(self, server_env) -> None:
        # The page exposes type=number with min/max so a UI bug can't
        # write 1e9 or negative values.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        # Loosely scrape: confirm both numeric inputs exist with the
        # correct attributes.
        for tid in ("ai-gate-min-codes", "ai-gate-min-hand"):
            chunk = r.text.split(f'data-test-id="{tid}"')[0].rsplit("<input", 1)[-1]
            assert 'type="number"' in chunk
            assert 'min="0"' in chunk
            assert 'max="10000"' in chunk


# --------------------------------------------------------------------------- #
# B. The form's submit flow round-trips through the existing endpoint.
# --------------------------------------------------------------------------- #


class TestAIGateUIRoundTrip:
    def test_default_config_load_via_get(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/ai/gate")
        assert r.status_code == 200
        body = r.json()
        assert body["config"][SETTING_KEY_MIN_CODES] == DEFAULT_MIN_CODES
        assert (
            body["config"][SETTING_KEY_MIN_HAND_CODED_SOURCES]
            == DEFAULT_MIN_HAND_CODED_SOURCES
        )
        assert body["config"][SETTING_KEY_OVERRIDE] == GATE_OVERRIDE_AUTO
        assert body["config"][SETTING_KEY_ENABLED] is True

    def test_form_submit_updates_thresholds(self, server_env) -> None:
        # Mirror what the page-side ``submitGateForm`` JS does: PUT the
        # canonical body, then GET to confirm persistence.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 5,
                "min_hand_coded_sources": 1,
                "override": GATE_OVERRIDE_AUTO,
                "enabled": True,
            },
        )
        assert r.status_code == 200, r.text
        roundtrip = client.get(f"/api/projects/{pid}/ai/gate").json()
        assert roundtrip["config"][SETTING_KEY_MIN_CODES] == 5
        assert roundtrip["config"][SETTING_KEY_MIN_HAND_CODED_SOURCES] == 1
        assert roundtrip["config"][SETTING_KEY_OVERRIDE] == GATE_OVERRIDE_AUTO

    def test_force_on_button_payload_opens_the_gate(self, server_env) -> None:
        # Mirror ``forceOnGate``: PUT with override=force_on; the
        # status now reports allowed=True.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 8,
                "min_hand_coded_sources": 2,
                "override": GATE_OVERRIDE_FORCE_ON,
                "enabled": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"]["allowed"] is True
        assert body["status"]["override"] == GATE_OVERRIDE_FORCE_ON

    def test_reset_button_payload_restores_spec_defaults(self, server_env) -> None:
        # Mirror ``resetGate``: PUT the spec defaults and confirm.
        _, client, _ = server_env
        pid = _make_project(client)
        # First push a non-default config so the reset is visible.
        client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"min_codes": 99, "override": GATE_OVERRIDE_FORCE_OFF},
        )
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 8,
                "min_hand_coded_sources": 2,
                "override": GATE_OVERRIDE_AUTO,
                "enabled": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["config"][SETTING_KEY_MIN_CODES] == DEFAULT_MIN_CODES
        assert (
            body["config"][SETTING_KEY_MIN_HAND_CODED_SOURCES]
            == DEFAULT_MIN_HAND_CODED_SOURCES
        )
        assert body["config"][SETTING_KEY_OVERRIDE] == GATE_OVERRIDE_AUTO

    def test_form_submit_validation_returns_400(self, server_env) -> None:
        # The page-side submit catches HTTP errors and surfaces them
        # via setMsg(...,'err'); the server rejects negative thresholds.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={"min_codes": -1, "override": GATE_OVERRIDE_AUTO},
        )
        assert r.status_code == 400

    def test_disabling_policy_round_trips(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 8,
                "min_hand_coded_sources": 2,
                "override": GATE_OVERRIDE_AUTO,
                "enabled": False,
            },
        )
        assert r.status_code == 200, r.text
        roundtrip = client.get(f"/api/projects/{pid}/ai/gate").json()
        assert roundtrip["config"][SETTING_KEY_ENABLED] is False
        # When the policy is disabled, the gate is open by definition.
        assert roundtrip["status"]["allowed"] is True


# --------------------------------------------------------------------------- #
# C. Page exposes the helper functions on window for the inline script.
# --------------------------------------------------------------------------- #


class TestAIGateUIScriptShape:
    def test_page_exposes_helper_functions_on_window(self, server_env) -> None:
        # The page-side wiring keeps backwards-compatible names on
        # ``window`` so a future console-driven script (or in-DOM test)
        # can reach the same helpers the Vitest suite covers.
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert "window._formatGateProgress" in r.text
        assert "window._populateGateForm" in r.text

    def test_page_loads_helpers_from_helpers_mjs(self, server_env) -> None:
        # The canonical home for the pure helpers is helpers.mjs;
        # confirm the static module is reachable so the inline form
        # behaviour stays test-coverable from Vitest.
        _, client, _ = server_env
        r = client.get("/static/js/helpers.mjs")
        assert r.status_code == 200
        body = r.text
        assert "export function formatGateProgress" in body
        assert "export function gateFormPayload" in body
        assert "export function gateForceOnPayload" in body
