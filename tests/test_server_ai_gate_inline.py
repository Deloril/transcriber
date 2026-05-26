"""F8.13 — inline AI-gate override on the source-coding view.

The F8.10 ``ai_gate`` data plane (GET / PUT /api/projects/<pid>/ai/gate)
has been wired since 267e77a; the F8.10 wiring commit (eef6779) put a
"Force open" button on /projects/<pid>/ai. F8.13 closes the loop on the
*source coding* surface: when ✨ Suggest with AI / 🔎 Find similar
quotes / Review pass triggers a 412, the popover / similar modal /
review pane now shows the structured gate status with a one-click
"Force open AI for this project" button instead of plain text.

This file proves four things end-to-end against the live FastAPI app:

  1. The source_coding.html template carries the new inline-block test
     ids and imports the F8.13 helpers from helpers.mjs (so the
     module-script wiring at the top of the page can find them).
  2. POST /api/projects/<pid>/ai/suggestions on a fresh project
     returns 412 with a structured ``{detail, gate}`` body the inline
     block can render (the gate keys the JS reads — message,
     code_count, min_codes, hand_coded_source_count,
     min_hand_coded_sources, override, enabled, reason — are all
     present).
  3. POST /api/projects/<pid>/ai/quote-searches on a fresh project
     also returns the same structured 412.
  4. PUT /api/projects/<pid>/ai/gate with override=force_on (the body
     the inline button sends) opens the gate; a follow-up POST to the
     suggestions endpoint is no longer blocked.

Pure-helper coverage for the JS side lives in
tests/js/ai-gate-inline.test.mjs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scribe import server as srv

    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), projects


def _new_project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "Gate-inline test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_source(client: TestClient, pid: str) -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": "S", "source_type": "transcript"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# A. Template surface
# --------------------------------------------------------------------------- #


class TestSourceCodingTemplateSurfacesInlineGate:
    def test_template_imports_inline_gate_helpers(self, env) -> None:
        # The module-script block at the top of the page imports
        # extractGateStatus / renderInlineGateBlockHtml /
        # gateForceOnPayload from helpers.mjs — F8.13 won't work
        # without them, so pin them.
        client, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200, r.text
        assert "extractGateStatus" in r.text
        assert "renderInlineGateBlockHtml" in r.text
        assert "gateForceOnPayload" in r.text
        # The window.__aiGate bridge is the contract the classic-script
        # block at the bottom relies on.
        assert "window.__aiGate" in r.text
        # The shared retry slot + GET /ai/gate cache are present so a
        # successful PUT preserves the user's saved thresholds.
        assert "_AI_GATE_RETRY" in r.text
        assert "_AI_GATE_CFG_CACHE" in r.text

    def test_template_has_force_on_click_handler(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        # The page binds a delegated click handler on the document so
        # any inline-gate block (popover / modal / review pane) can
        # share the same Force-open path. The contract: a button with
        # data-gate-action="force-on" triggers the PUT.
        assert 'data-gate-action="force-on"' in r.text
        # The PUT it issues is /api/projects/<pid>/ai/gate.
        assert "/ai/gate" in r.text

    def test_template_has_inline_gate_styles(self, env) -> None:
        # Rendering the block without CSS would leave it as a wall of
        # unstyled text; pin the styles too.
        client, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert ".ai-gate-inline" in r.text
        assert ".ai-gate-inline-force-on" in r.text
        assert ".ai-gate-inline-settings" in r.text


# --------------------------------------------------------------------------- #
# B. The 412 envelope is structured (the JS helper can extract it).
# --------------------------------------------------------------------------- #


class TestGateClosedReturnsActionablePayload:
    def test_suggestions_412_carries_full_gate_status(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "hello",
                "mode": "existing",
            },
        )
        assert r.status_code == 412, r.text
        body = r.json()
        # FastAPI wraps custom HTTPException(detail=...) bodies in
        # {"detail": <our body>}. The JS helper expects that envelope.
        assert "detail" in body
        inner = body["detail"]
        assert isinstance(inner, dict)
        assert inner["detail"] == "AI gate not satisfied"
        gate = inner["gate"]
        # Every field the inline block reads must be present.
        for field in (
            "allowed", "reason", "message",
            "code_count", "hand_coded_source_count",
            "min_codes", "min_hand_coded_sources",
            "override", "enabled",
        ):
            assert field in gate, f"missing gate.{field}"
        assert gate["allowed"] is False
        assert gate["override"] == "auto"
        assert gate["enabled"] is True
        # reason is the stable code the block puts on data-gate-reason.
        assert gate["reason"]
        # message is the human string the block puts in
        # .ai-gate-inline-msg.
        assert isinstance(gate["message"], str) and gate["message"]

    def test_quote_searches_412_carries_full_gate_status(self, env) -> None:
        # text-mode query; the gate is checked early in the handler so
        # we don't need a seeded application to confirm the structured
        # 412 envelope.
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/ai/quote-searches",
            json={
                "query_text": "managing pain at home",
                "top_k": 5,
            },
        )
        assert r.status_code == 412, r.text
        inner = r.json()["detail"]
        assert inner["detail"] == "AI gate not satisfied"
        gate = inner["gate"]
        assert gate["allowed"] is False
        assert isinstance(gate.get("message"), str) and gate["message"]


# --------------------------------------------------------------------------- #
# C. The Force-open button's PUT body opens the gate.
# --------------------------------------------------------------------------- #


class TestForceOpenButtonRoundTrip:
    def test_put_force_on_opens_the_gate(self, env) -> None:
        # gateForceOnPayload(null) returns
        # {min_codes: 8, min_hand_coded_sources: 2, override: "force_on",
        #  enabled: true} — same body the inline button PUTs.
        client, _ = env
        pid = _new_project(client)
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 8,
                "min_hand_coded_sources": 2,
                "override": "force_on",
                "enabled": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"]["allowed"] is True
        assert body["status"]["reason"] == "force_on"

    def test_force_open_then_suggestions_no_longer_412(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        sid = _new_source(client, pid)
        # First call: gate closed → 412.
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "hello",
                "mode": "existing",
            },
        )
        assert r.status_code == 412, r.text
        # User clicks Force-open: PUT override=force_on with the spec
        # defaults (the JS helper falls back to these when no config
        # has been GET'd yet).
        r = client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 8,
                "min_hand_coded_sources": 2,
                "override": "force_on",
                "enabled": True,
            },
        )
        assert r.status_code == 200, r.text
        # Retry: gate is open. The endpoint now fails further along
        # (no AI backend configured) — but specifically *not* with 412.
        r = client.post(
            f"/api/projects/{pid}/ai/suggestions",
            json={
                "source_id": sid,
                "anchor_start_word_id": "s0w0",
                "anchor_end_word_id": "s0w5",
                "query_text": "hello",
                "mode": "existing",
            },
        )
        assert r.status_code != 412, (
            "Force-open didn't unblock the gate: still 412"
        )

    def test_force_open_preserves_user_thresholds(self, env) -> None:
        # If a user has previously customised thresholds (e.g.
        # min_codes = 12), the inline button should pass those through
        # so flipping override doesn't silently reset them. The JS
        # helper's contract: read the cached config first, apply
        # override=force_on. The server-side proof: a PUT carrying the
        # custom thresholds with override=force_on persists them both.
        client, _ = env
        pid = _new_project(client)
        # Customise.
        client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 12,
                "min_hand_coded_sources": 3,
                "override": "auto",
                "enabled": True,
            },
        )
        # Inline force-on with the cached config.
        client.put(
            f"/api/projects/{pid}/ai/gate",
            json={
                "min_codes": 12,
                "min_hand_coded_sources": 3,
                "override": "force_on",
                "enabled": True,
            },
        )
        body = client.get(f"/api/projects/{pid}/ai/gate").json()
        assert body["config"]["min_codes"] == 12
        assert body["config"]["min_hand_coded_sources"] == 3
        assert body["config"]["override"] == "force_on"
