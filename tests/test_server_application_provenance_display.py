"""End-to-end reachability tests for F9.9 (per-application provenance
display on hover).

Background
----------

F9.9 ships :mod:`scribe.application_provenance_display` — a pure
builder + three formatters that turn an Application + the related
Code / CodeVersion / Coder into a structured display surface for
the editor's hover tooltip. That module ships with 55 unit tests
in ``tests/test_application_provenance_display.py`` and a full JS
mirror exercised by ``tests/js/application-provenance-display.test.mjs``.

What was missing — and what this file covers — is the integration
proof that the **user-facing surface** is wired together. Per the
loop's done-criteria, F9.9 is only "done" if a researcher can
reach the data layer through a real route + a real UI control.
That means:

1. ``GET /api/projects/<pid>/applications/<aid>/provenance`` must
   return a JSON envelope with the structured display dict, the
   one-line summary, and the pre-rendered text + HTML (so the
   coding view can drop the HTML directly into the popover).

2. The route must walk the entity graph defensively: a deleted
   code, a missing version snapshot, a coder that was retired —
   none of these should 500. The display fields carry the
   ``code_missing`` / ``snapshot_missing`` / ``(unknown)`` hints
   per the F9.9 module's contract.

3. The coding view (``GET /projects/<pid>/sources/<sid>``) must
   render the popover container, the JS module shim that imports
   the helpers.mjs mirror, and the per-row / gutter ``data-prov-
   hover`` markers so the document-level hover delegate can find
   them.

4. End-to-end: build an application, hit the route, and confirm
   the fields the popover wants to display (code name, anchor
   label, coder name, ai feature label, drift hint) all round-
   trip from the project state through the route into a JSON
   payload the JS can render without further lookups.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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


def _make_project(client: TestClient, name: str = "F9.9 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(
    client: TestClient,
    pid: str,
    name: str = "Negotiating identity",
    *,
    definition: str = "moments where the participant negotiates membership",
    colour: str | None = None,
    stage: str | None = None,
) -> str:
    body: dict = {"name": name, "definition": definition}
    if colour is not None:
        body["colour"] = colour
    if stage is not None:
        body["stage"] = stage
    r = client.post(f"/api/projects/{pid}/codes", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient, pid: str, name: str = "Interview 1",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_coder(
    client: TestClient, pid: str, name: str = "Alex",
    *, role: str = "researcher",
) -> str:
    r = client.post(
        f"/api/projects/{pid}/coders",
        json={"name": name, "role": role},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _apply(
    client: TestClient,
    pid: str,
    cid: str,
    sid: str,
    start: str = "s0w0",
    end: str = "s0w12",
    *,
    coder_id: str | None = None,
    note: str = "",
) -> str:
    body: dict = {
        "code_id": cid,
        "source_id": sid,
        "anchor_start_word_id": start,
        "anchor_end_word_id": end,
    }
    if coder_id is not None:
        body["coder_id"] = coder_id
    if note:
        body["note"] = note
    r = client.post(
        f"/api/projects/{pid}/applications", json=body,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# 1. Coding-view template surfaces the F9.9 popover affordances.
# --------------------------------------------------------------------------- #


class TestSourceCodingTemplateExposesProvenanceUI:
    """Without these markers in the rendered page, a researcher
    hovering over a coded segment gets nothing — even if the route
    is perfect."""

    def test_popover_container_renders(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert 'id="provenancePopover"' in body
        assert 'data-test-feature="F9.9"' in body
        assert 'data-test-id="provenance-popover"' in body
        assert 'class="provenance-popover-body"' in body
        # The popover starts hidden — it's a hover target, not a
        # always-on panel.
        assert "hidden" in body[body.find('id="provenancePopover"'):
                                 body.find('id="provenancePopover"') + 400]

    def test_app_row_template_carries_prov_hover_marker(self, server_env) -> None:
        """The renderApps() template literal stamps every .app-row with
        data-prov-hover='F9.9'; the document-level mouseover delegate
        keys off this attribute."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        body = r.text
        assert 'data-prov-hover="F9.9"' in body
        # Hover delegate selector — both .app-row and gutter .lane-bar.
        assert ".app-row[data-app-id][data-prov-hover='F9.9']" in body
        assert ".lane-bar[data-app-id][data-prov-hover='F9.9']" in body

    def test_helpers_module_shim_imports_provenance_helpers(self, server_env) -> None:
        """The classic-script handler reaches the helpers.mjs F9.9
        functions through ``window.__provenance``. Without this the JS
        side has no offline fallback if the route 404s mid-edit."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert "buildProvenanceDisplay" in body
        assert "formatProvenanceHtml" in body
        assert "formatProvenanceText" in body
        assert "provenanceSummaryLabel" in body
        assert "window.__provenance" in body
        assert 'CustomEvent("scribe:provenance-ready")' in body

    def test_route_url_template_present_in_javascript(self, server_env) -> None:
        """The fetch URL is built per-application; we look for the
        path fragment the JS uses to call the F9.9 endpoint so a
        future refactor can't silently change the contract."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        assert "/applications/" in body
        assert "/provenance" in body

    def test_popover_css_classes_match_module_html(self, server_env) -> None:
        """The popover body is filled with the F9.9 module's
        ``format_provenance_html`` output; the CSS in the template
        targets the same class names so server-side and client-side
        renders look identical."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        body = r.text
        # Every class produced by format_provenance_html must have a
        # matching style rule somewhere in the page.
        for cls in (
            "provenance-display",
            "provenance-title",
            "provenance-version",
            "provenance-swatch",
            "provenance-source",
            "provenance-meta",
            "provenance-warn",
            "provenance-drift",
            "provenance-ai",
            "provenance-ai-head",
            "provenance-ai-notes",
            "provenance-extra",
            "provenance-note",
            "provenance-note-head",
            "provenance-note-body",
            "provenance-role",
        ):
            assert cls in body, f"missing CSS hook for .{cls}"


# --------------------------------------------------------------------------- #
# 2. Endpoint round-trips for the headline case.
# --------------------------------------------------------------------------- #


class TestProvenanceEndpointSuccess:
    """Building a project + code + coder + application should resolve
    to a fully-populated display payload."""

    def test_returns_envelope_with_html_and_summary(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, name="Negotiating identity")
        sid = _make_source(client, pid)
        coder_id = _make_coder(client, pid, name="Alex")
        aid = _apply(client, pid, cid, sid, coder_id=coder_id, note="early-stage")

        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Envelope keys.
        assert body["application_id"] == aid
        assert "display" in body
        assert "summary" in body
        assert "text" in body
        assert "html" in body
        # The display dict has every field the JS mirror expects;
        # spot-check the most important ones.
        d = body["display"]
        assert d["application_id"] == aid
        assert d["code_id"] == cid
        assert d["code_name"] == "Negotiating identity"
        assert d["coder_id"] == coder_id
        assert d["coder_name"] == "Alex"
        assert d["coder_role"] == "researcher"
        assert d["source_id"] == sid
        assert d["source_name"] == "Interview 1"
        assert d["anchor_label"] == "s0w0–s0w12"
        # AI provenance unset for a vanilla human-coded application.
        assert d["ai_present"] is False
        # The provenance source label defaults to "Human-coded".
        assert d["provenance_source_label"] == "Human-coded"
        # Note round-trips through.
        assert d["note"] == "early-stage"

    def test_html_renders_safe_markup(self, server_env) -> None:
        """The pre-rendered HTML must wrap everything in the
        provenance-display root and contain the code name; this is
        what the popover writes into innerHTML."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, name="Negotiating identity")
        sid = _make_source(client, pid)
        aid = _apply(client, pid, cid, sid)

        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        html = r.json()["html"]
        assert html.startswith('<div class="provenance-display">')
        assert html.endswith("</div>")
        assert "Negotiating identity" in html
        # Anchor label is part of the <dl class="provenance-meta">.
        assert "s0w0–s0w12" in html

    def test_summary_label_includes_coder_and_provenance(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        coder_id = _make_coder(client, pid, name="Alex")
        aid = _apply(client, pid, cid, sid, coder_id=coder_id)

        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        summary = r.json()["summary"]
        assert "Alex" in summary
        assert "Human-coded" in summary

    def test_text_format_is_multiline_and_includes_code_name(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, name="Negotiating identity")
        sid = _make_source(client, pid)
        aid = _apply(client, pid, cid, sid)

        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        text = r.json()["text"]
        assert "\n" in text
        assert "Negotiating identity" in text
        # Plain-text format is suitable for a title= attribute.
        assert "<" not in text


# --------------------------------------------------------------------------- #
# 3. Defensive paths — missing related entities.
# --------------------------------------------------------------------------- #


class TestProvenanceEndpointDefensivePaths:
    """When an application's code or coder has been deleted, the
    display surface still renders with the F9.9 contract's
    "missing" hints. The route must not 500 in any of these
    cases."""

    def test_missing_application_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # 12-hex-char id that doesn't exist on disk.
        r = client.get(
            f"/api/projects/{pid}/applications/{'a' * 12}/provenance"
        )
        assert r.status_code == 404

    def test_missing_project_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get(
            f"/api/projects/{'0' * 12}/applications/{'a' * 12}/provenance"
        )
        assert r.status_code == 404

    def test_invalid_application_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/applications/not-a-valid-id/provenance"
        )
        assert r.status_code == 400

    def test_deleted_code_yields_code_missing_payload(self, server_env) -> None:
        """If the application's code has been deleted out from under
        it, the display still renders with code_name='(unknown)' and
        code_missing=True — the audit story is "show what's left"."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        aid = _apply(client, pid, cid, sid)
        # Delete the code from disk to simulate a destructive
        # lifecycle op that orphaned the application.
        from scribe import codes as _codes
        from scribe import server as srv
        _codes.code_state_path(srv.PROJECTS_DIR, pid, cid).unlink()
        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        d = r.json()["display"]
        assert d["code_missing"] is True
        assert d["code_name"] == "(unknown)"
        # Code id is still echoed so the renderer can deep-link.
        assert d["code_id"] == cid

    def test_default_coder_falls_back_gracefully(self, server_env) -> None:
        """An application made without an explicit coder_id falls back
        to the project's auto-created default coder ("You" in the
        single-user F2.5 fallback). The display must still surface a
        real coder name — never blank, never "(unknown)" when the on-
        disk Coder resolves."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # No coder id: the server resolves to the default coder.
        aid = _apply(client, pid, cid, sid)
        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        d = r.json()["display"]
        assert d["coder_id"] != ""
        # Default coder name from _ensure_default_coder. The popover
        # surfaces it as a real name rather than an empty / unknown
        # placeholder.
        assert d["coder_name"] not in {"", "(unknown)"}

    def test_deleted_coder_yields_unknown_coder_label(self, server_env) -> None:
        """If the application's coder has been deleted out from under
        it, the display falls back to '(unknown)' rather than blank
        — the popover always tells the user *something*."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        coder_id = _make_coder(client, pid, name="Alex")
        aid = _apply(client, pid, cid, sid, coder_id=coder_id)
        # Remove the coder file from disk to simulate destructive
        # cleanup that orphaned the application.
        from scribe import coders as _coders
        from scribe import server as srv
        _coders.coder_state_path(srv.PROJECTS_DIR, pid, coder_id).unlink()
        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200
        d = r.json()["display"]
        assert d["coder_name"] == "(unknown)"


# --------------------------------------------------------------------------- #
# 4. Drift surface — F9.2's drifted_definition_fields surfaces inline.
# --------------------------------------------------------------------------- #


class TestProvenanceEndpointDriftHint:
    """If the code's current definition has drifted from the version
    snapshot recorded at apply, the hover should surface the drift
    inline rather than hiding it behind the F9.2 audit report."""

    def test_drift_after_definition_edit_surfaces_in_payload(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid, name="Negotiating identity",
                         definition="initial definition")
        sid = _make_source(client, pid)
        aid = _apply(client, pid, cid, sid)

        # Edit the code's definition so the application's recorded
        # version snapshot drifts from the current version. The PATCH
        # endpoint records a new version; the application's
        # ``definition_version_id_at_apply`` still points at the old
        # snapshot, which is what F9.2's drift detection compares.
        r = client.patch(
            f"/api/projects/{pid}/codes/{cid}",
            json={
                "definition": "edited definition",
                "change_note": "tightened scope",
            },
        )
        assert r.status_code == 200, r.text

        r = client.get(
            f"/api/projects/{pid}/applications/{aid}/provenance"
        )
        assert r.status_code == 200, r.text
        d = r.json()["display"]
        assert d["definition_drifted"] is True
        assert "definition" in list(d["drifted_fields"])
        # The HTML body must surface the drift as a visible hint.
        html = r.json()["html"]
        assert "Definition has changed since apply" in html


# --------------------------------------------------------------------------- #
# 5. Module reachability — confirm the module is imported into server.
# --------------------------------------------------------------------------- #


class TestModuleWiredIntoServer:
    def test_application_provenance_display_module_is_imported(self) -> None:
        """The route reaches into scribe.application_provenance_display.
        Confirm the import resolves so a future refactor can't drop
        the module without breaking the test suite."""
        from scribe import server as srv

        assert hasattr(srv, "_application_provenance_display")
        from scribe.application_provenance_display import (
            build_provenance_display,
            format_provenance_html,
        )
        assert callable(build_provenance_display)
        assert callable(format_provenance_html)
