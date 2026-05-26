"""End-to-end reachability tests for F4.2 (multiple non-contiguous
applications per (code, source)).

Background
----------

F4.2 ships ``scribe.application_spans`` — pure helpers that surface
*how* a (code, source) pair is structured once it carries more than
one application: anchor-sorted ordering, overlap / disjoint /
adjacency, exact-duplicate detection, and the headline
``non_contiguous_components`` operation. All of that is tested as
pure logic in ``tests/test_application_spans.py`` (72 cases).

What was missing — and what this file covers — is the integration
proof that the **user-facing surface** is wired together. Per the
loop's done-criteria, F4.2 is only "done" if a researcher can reach
the data layer through a real route + a real UI control. That means:

1. ``GET /api/projects/<pid>/applications/spans?source_id=<sid>``
   must return the F4.2 structure: per-code component count,
   per-component anchor bounds, and a duplicate-anchor diagnostic.
2. The static ``spans`` route must beat the parametric
   ``{application_id}`` route (registration-order bug we'd hit if
   the spans route was added at the bottom).
3. The coding view (``GET /projects/<pid>/sources/<sid>``) must
   render the spans-summary panel + the per-row place-badge JS so
   a researcher actually *sees* "code X applies in 2 places".
4. The endpoint must round-trip multi-component, single-component,
   and duplicate-anchor cases against the F4.2 module's contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures — mirror the pattern in test_server_applications.py.
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


def _make_project(client: TestClient, name: str = "F4.2 holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str = "managing pain") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "moments where the participant copes"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "Interview 1") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "source_type": "transcript", "language": "en"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _apply(
    client: TestClient,
    pid: str,
    cid: str,
    sid: str,
    start: str,
    end: str,
) -> str:
    r = client.post(
        f"/api/projects/{pid}/applications",
        json={
            "code_id": cid,
            "source_id": sid,
            "anchor_start_word_id": start,
            "anchor_end_word_id": end,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# 1. Coding view template surfaces the F4.2 spans UI.
# --------------------------------------------------------------------------- #


class TestSourceCodingTemplateExposesSpansUI:
    """Without these controls in the rendered page, a user can't see
    F4.2's "this code applies in 3 places" insight no matter how
    correct the endpoint is."""

    def test_coding_view_renders_spans_summary_panel(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # The diagnostic panel anchors above #appList and is referenced
        # by the JS that paints non-contiguous components.
        assert 'id="spansSummary"' in r.text
        assert 'aria-label="Code span summary"' in r.text

    def test_coding_view_references_spans_endpoint(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        # JS uses backtick template literals; we look for the path
        # fragment + the source_id query-param the panel paints from.
        assert "/api/projects/${PROJECT_ID}/applications/spans" in r.text
        assert "source_id=${encodeURIComponent(SOURCE_ID)}" in r.text

    def test_coding_view_renders_place_badge_helpers(self, server_env) -> None:
        """The per-row "place i/N" badge is the F4.2 row-level affordance:
        without ``placeBadgeFor`` the JS can't decorate rows with their
        component index, and the diagnostic panel alone can't tell a
        researcher which row belongs to which place."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert "function placeBadgeFor" in r.text
        assert "place-badge" in r.text
        # The lifecycle hooks: the panel must refresh on apply / delete.
        assert "refreshSpans" in r.text


# --------------------------------------------------------------------------- #
# 2. Endpoint contract: empty / single-component / multi-component /
#    duplicate-anchor / multi-code cases all round-trip through the
#    routes the page consumes.
# --------------------------------------------------------------------------- #


class TestSpansEndpointRoundTrip:
    """The headline F4.2 case is "I've coded this idea in 3 different
    non-contiguous places". The endpoint has to count those places
    correctly, surface duplicate anchors as a diagnostic, and not
    auto-merge anything (researchers explicitly want the gap)."""

    def test_spans_static_route_beats_parametric_application_id(self, server_env) -> None:
        """F4.2 lives at ``/applications/spans``. ``/applications/{aid}``
        comes after in the file because FastAPI dispatches by
        registration order — without that order, ``spans`` would be
        captured as ``application_id="spans"`` and rejected by
        ``_check_application_id``."""
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        # No applications: but the route must still return a 200 + the
        # F4.2-shaped envelope rather than a 400-bad-application-id.
        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {
            "source_id": sid,
            "by_code": [],
            "duplicate_anchor_groups": [],
        }

    def test_spans_requires_source_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Missing required query param -> FastAPI 422 (validation).
        r = client.get(f"/api/projects/{pid}/applications/spans")
        assert r.status_code == 422

    def test_spans_404_for_unknown_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        bogus = "f" * 12
        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={bogus}"
        )
        assert r.status_code == 404

    def test_single_application_reports_one_component(self, server_env) -> None:
        """One application = one place. ``component_count == 1`` is the
        baseline; the UI hides the "(N places)" badge in this case."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _apply(client, pid, cid, sid, "s0w0", "s0w3")
        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["by_code"]) == 1
        entry = body["by_code"][0]
        assert entry["code_id"] == cid
        assert entry["application_count"] == 1
        assert entry["component_count"] == 1
        assert len(entry["components"]) == 1
        assert entry["components"][0] == {
            "start_word_id": "s0w0",
            "end_word_id": "s0w3",
            "size": 1,
        }
        assert entry["duplicate_anchor_count"] == 0
        assert body["duplicate_anchor_groups"] == []

    def test_three_non_contiguous_applications_report_three_places(
        self, server_env
    ) -> None:
        """The F4.2 headline: the same code applied to three clearly
        separated spans should report ``component_count == 3``."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Three non-overlapping, non-adjacent ranges.
        _apply(client, pid, cid, sid, "s0w0", "s0w2")
        _apply(client, pid, cid, sid, "s0w10", "s0w12")
        _apply(client, pid, cid, sid, "s0w20", "s0w22")

        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        assert r.status_code == 200
        entry = r.json()["by_code"][0]
        assert entry["application_count"] == 3
        assert entry["component_count"] == 3
        # Components are anchor-sorted earliest first.
        starts = [c["start_word_id"] for c in entry["components"]]
        assert starts == ["s0w0", "s0w10", "s0w20"]

    def test_overlapping_applications_collapse_into_one_component(
        self, server_env
    ) -> None:
        """Two applications that overlap on the transcript belong to
        the same non-contiguous component — they're one "place" with
        two coders / two anchors / two views of the same idea."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _apply(client, pid, cid, sid, "s0w0", "s0w5")
        _apply(client, pid, cid, sid, "s0w3", "s0w8")  # overlaps the first

        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        entry = r.json()["by_code"][0]
        assert entry["application_count"] == 2
        assert entry["component_count"] == 1
        assert entry["components"][0]["size"] == 2

    def test_duplicate_anchors_surface_as_diagnostic(self, server_env) -> None:
        """Identical anchors on the same (code, source) almost always
        mean a UX or import bug; F4.2 surfaces them so the user can
        decide what to do — never auto-merges."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        _apply(client, pid, cid, sid, "s0w0", "s0w3")
        _apply(client, pid, cid, sid, "s0w0", "s0w3")  # identical anchor

        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        body = r.json()
        entry = body["by_code"][0]
        assert entry["duplicate_anchor_count"] == 2
        assert len(body["duplicate_anchor_groups"]) == 1
        group = body["duplicate_anchor_groups"][0]
        assert group["code_id"] == cid
        assert len(group["application_ids"]) == 2

    def test_multi_code_buckets_independently(self, server_env) -> None:
        """Two different codes on the same source should produce two
        ``by_code`` entries — they don't share components."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid_a = _make_code(client, pid, "managing pain")
        cid_b = _make_code(client, pid, "seeking help")
        sid = _make_source(client, pid)
        _apply(client, pid, cid_a, sid, "s0w0", "s0w3")
        _apply(client, pid, cid_a, sid, "s0w20", "s0w22")
        _apply(client, pid, cid_b, sid, "s0w5", "s0w8")

        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid}"
        )
        body = r.json()
        by_code = {e["code_id"]: e for e in body["by_code"]}
        assert by_code[cid_a]["component_count"] == 2
        assert by_code[cid_b]["component_count"] == 1

    def test_apps_on_other_sources_dont_leak_in(self, server_env) -> None:
        """The endpoint is scoped by ``source_id``; applications on a
        sibling source must not influence either the count or the
        components."""
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid_a = _make_source(client, pid, "Interview A")
        sid_b = _make_source(client, pid, "Interview B")
        _apply(client, pid, cid, sid_a, "s0w0", "s0w3")
        _apply(client, pid, cid, sid_b, "s0w0", "s0w3")
        _apply(client, pid, cid, sid_b, "s0w50", "s0w55")

        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid_a}"
        )
        entry = r.json()["by_code"][0]
        assert entry["application_count"] == 1
        assert entry["component_count"] == 1
        # Sibling source has 2 components, scoped independently.
        r = client.get(
            f"/api/projects/{pid}/applications/spans?source_id={sid_b}"
        )
        entry = r.json()["by_code"][0]
        assert entry["application_count"] == 2
        assert entry["component_count"] == 2
