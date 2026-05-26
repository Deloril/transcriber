"""End-to-end reachability tests for F2.5 (multi-coder mode + ICR).

The pure data layer lives in ``scribe/coders.py`` and ``scribe/icr.py``
with ~135 unit tests across ``tests/test_coders.py`` and
``tests/test_icr.py``. cae5570 shipped both as pure modules with no
HTTP / template surface — researchers couldn't add a second coder,
attribute applications to a chosen coder, or compute Cohen's kappa
from the UI.

This file proves the **user-facing surface** is now wired together:

  * Project home shows a Coders snapshot card → ``/projects/<pid>/coders``
    lists them → ``/projects/<pid>/coders/new`` POSTs to the new
    ``/api/projects/<pid>/coders`` endpoint → the detail page PATCHes
    + DELETEs through that surface.
  * The ICR page (``/projects/<pid>/icr``) consumes
    ``GET /api/projects/<pid>/icr?coder_a=...&coder_b=...`` to compute
    overall Cohen's kappa + per-code kappa.
  * ``POST /api/projects/<pid>/applications`` accepts an optional
    ``coder_id`` so the coding view can attribute the application to
    whichever coder is "active" in the user's session.

Sibling of ``tests/test_server_participants.py`` (F1.3).
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
    """TestClient with isolated tmp dirs for uploads/outputs/projects."""
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


def _make_project(client: TestClient, name: str = "Holder") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_code(client: TestClient, pid: str, name: str = "Empathy") -> str:
    r = client.post(
        f"/api/projects/{pid}/codes",
        json={"name": name, "definition": "Caring for someone else."},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(client: TestClient, pid: str, name: str = "Interview 01") -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={"name": name, "type": "interview"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Project home links to the coders surface (snapshot card discoverability)
# --------------------------------------------------------------------------- #


class TestProjectHomeLinksToCoders:
    def test_project_home_renders_coders_snapshot_card(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert "Coders" in r.text
        assert 'id="coderCount"' in r.text
        assert 'id="coderList"' in r.text

    def test_project_home_renders_coder_ctas(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert f'href="/projects/{pid}/coders/new"' in r.text
        assert f'href="/projects/{pid}/coders"' in r.text
        assert f'href="/projects/{pid}/icr"' in r.text
        assert "+ New coder" in r.text

    def test_project_home_consumes_coders_json_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}")
        assert "/api/projects/${PROJECT_ID}/coders" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/coders — list page
# --------------------------------------------------------------------------- #


class TestCodersListPage:
    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        assert r.status_code == 200

    def test_has_new_coder_action_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        assert "+ New coder" in r.text
        assert f'href="/projects/{pid}/coders/new"' in r.text

    def test_has_icr_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        # The ICR page is the second discoverable path off the coders list.
        assert f'href="/projects/{pid}/icr"' in r.text
        assert "Inter-coder reliability" in r.text

    def test_has_back_to_project_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        assert f'href="/projects/{pid}"' in r.text

    def test_empty_state_offers_create_cta(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        assert "No coders yet" in r.text

    def test_invalid_project_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/..%2Fevil/coders")
        assert r.status_code in (400, 404)

    def test_consumes_the_json_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders")
        assert "/api/projects/${PROJECT_ID}/coders" in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/coders/new — create form
# --------------------------------------------------------------------------- #


class TestCoderNewPage:
    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/new")
        assert r.status_code == 200

    def test_has_back_to_coders_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/new")
        assert f'href="/projects/{pid}/coders"' in r.text

    def test_form_has_all_coder_fields(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/new")
        # Coder dataclass fields: name, role, status, email, colour, notes.
        assert 'id="nc-name"' in r.text
        assert 'id="nc-role"' in r.text
        assert 'id="nc-status"' in r.text
        assert 'id="nc-email"' in r.text
        assert 'id="nc-colour"' in r.text
        assert 'id="nc-notes"' in r.text

    def test_form_posts_to_coders_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/new")
        assert "/api/projects/${PROJECT_ID}/coders" in r.text
        assert '"POST"' in r.text or "'POST'" in r.text

    def test_role_select_includes_documented_options(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/new")
        for role in ("researcher", "second_coder", "reviewer", "trainee", "other"):
            assert f'value="{role}"' in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/coders/<cid> — detail page
# --------------------------------------------------------------------------- #


class TestCoderDetailPage:
    def test_renders_with_200_for_valid_id_shape(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/aaaaaaaaaaaa")
        assert r.status_code == 200

    def test_invalid_coder_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/nope")
        assert r.status_code == 400

    def test_has_back_link_to_list(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/aaaaaaaaaaaa")
        assert f'href="/projects/{pid}/coders"' in r.text

    def test_loads_coder_via_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/aaaaaaaaaaaa")
        assert "/api/projects/${PROJECT_ID}/coders/${CODER_ID}" in r.text

    def test_has_save_set_active_and_delete_buttons(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/coders/aaaaaaaaaaaa")
        assert "Save changes" in r.text
        assert "Use as active coder" in r.text
        assert 'id="deleteBtn"' in r.text


# --------------------------------------------------------------------------- #
# /projects/<pid>/icr — comparison page
# --------------------------------------------------------------------------- #


class TestICRPage:
    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/icr")
        assert r.status_code == 200

    def test_has_back_to_project_link(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/icr")
        assert f'href="/projects/{pid}"' in r.text
        assert f'href="/projects/{pid}/coders"' in r.text

    def test_has_coder_selectors_and_submit(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/icr")
        # The form fields the ICR computation needs.
        assert 'id="ic-coder-a"' in r.text
        assert 'id="ic-coder-b"' in r.text
        assert 'id="ic-source"' in r.text
        assert 'id="ic-run"' in r.text

    def test_consumes_icr_json_api(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/icr")
        assert "/api/projects/${PROJECT_ID}/icr" in r.text

    def test_invalid_project_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/projects/..%2Fevil/icr")
        assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# REST API: coders CRUD round-trip
# --------------------------------------------------------------------------- #


class TestCodersAPI:
    def test_list_initially_empty(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/coders")
        assert r.status_code == 200
        assert r.json() == {"coders": []}

    def test_create_round_trip(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Coder B", "role": "second_coder",
                  "email": "b@example.org", "colour": "#aaccee",
                  "notes": "trained on initial codebook"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        cid = body["id"]
        assert body["name"] == "Coder B"
        assert body["role"] == "second_coder"
        assert body["email"] == "b@example.org"
        assert body["colour"] == "#aaccee"

        listed = client.get(f"/api/projects/{pid}/coders").json()["coders"]
        assert any(c["id"] == cid for c in listed)

    def test_create_validation_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Missing name
        r = client.post(f"/api/projects/{pid}/coders", json={"role": "researcher"})
        assert r.status_code == 400
        # Bad role
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "x", "role": "boss"},
        )
        assert r.status_code == 400
        # Bad email
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "x", "email": "not-an-email"},
        )
        assert r.status_code == 400

    def test_create_in_missing_project_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.post(
            "/api/projects/aaaaaaaaaaaa/coders",
            json={"name": "Ghost"},
        )
        assert r.status_code == 404

    def test_get_404_when_missing(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/coders/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_get_400_on_bad_shape(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/coders/not-hex")
        assert r.status_code == 400

    def test_patch_round_trip(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Coder B", "role": "researcher"},
        )
        cid = r.json()["id"]
        r = client.patch(
            f"/api/projects/{pid}/coders/{cid}",
            json={"role": "second_coder", "status": "inactive"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "second_coder"
        assert body["status"] == "inactive"
        # Round-trip
        fetched = client.get(f"/api/projects/{pid}/coders/{cid}").json()
        assert fetched["role"] == "second_coder"
        assert fetched["status"] == "inactive"

    def test_delete_round_trip(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Coder B"},
        )
        cid = r.json()["id"]
        r = client.delete(f"/api/projects/{pid}/coders/{cid}")
        assert r.status_code == 200
        assert client.get(f"/api/projects/{pid}/coders/{cid}").status_code == 404

    def test_delete_404_when_missing(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.delete(f"/api/projects/{pid}/coders/aaaaaaaaaaaa")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Application creation honours the optional coder_id field
# --------------------------------------------------------------------------- #


class TestApplicationCoderAttribution:
    def _make_app(
        self, client: TestClient, pid: str, code_id: str, source_id: str,
        coder_id: str | None = None,
    ):
        body = {
            "code_id": code_id,
            "source_id": source_id,
            "anchor_start_word_id": "s0w1",
            "anchor_end_word_id": "s0w3",
        }
        if coder_id is not None:
            body["coder_id"] = coder_id
        return client.post(f"/api/projects/{pid}/applications", json=body)

    def test_default_coder_used_when_unspecified(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        r = self._make_app(client, pid, cid, sid)
        assert r.status_code == 201, r.text
        # Default coder is auto-created
        coders = client.get(f"/api/projects/{pid}/coders").json()["coders"]
        assert len(coders) == 1
        default_coder_id = coders[0]["id"]
        assert r.json()["coder_id"] == default_coder_id

    def test_explicit_coder_id_is_honoured(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        # Make a second coder
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Second", "role": "second_coder"},
        )
        second_id = r.json()["id"]
        r = self._make_app(client, pid, cid, sid, coder_id=second_id)
        assert r.status_code == 201, r.text
        assert r.json()["coder_id"] == second_id

    def test_invalid_coder_id_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        r = self._make_app(client, pid, cid, sid, coder_id="not-hex")
        assert r.status_code == 400

    def test_unknown_coder_id_returns_404(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        cid = _make_code(client, pid)
        sid = _make_source(client, pid)
        r = self._make_app(client, pid, cid, sid, coder_id="bbbbbbbbbbbb")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# REST API: ICR computation
# --------------------------------------------------------------------------- #


class TestICRAPI:
    def _seed_two_coders_and_apps(self, client: TestClient, pid: str):
        """Set up two coders + a code + a source + applications with overlap."""
        sid = _make_source(client, pid)
        code_a = _make_code(client, pid, "Empathy")
        code_b = _make_code(client, pid, "Resistance")
        # First application to bootstrap the default coder.
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": code_a, "source_id": sid,
                  "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w2"},
        )
        assert r.status_code == 201
        coder_default = client.get(
            f"/api/projects/{pid}/coders"
        ).json()["coders"][0]["id"]
        # Second coder
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Coder B", "role": "second_coder"},
        )
        coder_b_id = r.json()["id"]
        # Second coder applies the same code to the same span — perfect agreement on item w-0
        client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": code_a, "source_id": sid,
                  "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w2",
                  "coder_id": coder_b_id},
        )
        # And applies a different code on a different item — disagreement on w-5
        client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": code_b, "source_id": sid,
                  "anchor_start_word_id": "s0w5", "anchor_end_word_id": "s0w7",
                  "coder_id": coder_b_id},
        )
        # Default coder applies code_a to that same w-5 span — disagreement code-wise
        client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": code_a, "source_id": sid,
                  "anchor_start_word_id": "s0w5", "anchor_end_word_id": "s0w7"},
        )
        return coder_default, coder_b_id, sid, code_a, code_b

    def test_icr_400_on_bad_coder_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/icr?coder_a=not-hex&coder_b=aaaaaaaaaaaa"
        )
        assert r.status_code == 400

    def test_icr_404_on_unknown_coder(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/icr?coder_a=aaaaaaaaaaaa&coder_b=bbbbbbbbbbbb"
        )
        assert r.status_code == 404

    def test_icr_returns_perfect_agreement_when_no_apps(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Make two coders
        r1 = client.post(
            f"/api/projects/{pid}/coders", json={"name": "A"}
        )
        r2 = client.post(
            f"/api/projects/{pid}/coders", json={"name": "B"}
        )
        a, b = r1.json()["id"], r2.json()["id"]
        r = client.get(f"/api/projects/{pid}/icr?coder_a={a}&coder_b={b}")
        assert r.status_code == 200
        body = r.json()
        assert body["n_items"] == 0
        assert body["overall_kappa"] == 1.0  # vacuous perfect agreement

    def test_icr_full_round_trip_with_real_overlap(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        a_id, b_id, sid, code_a, code_b = self._seed_two_coders_and_apps(
            client, pid
        )
        r = client.get(
            f"/api/projects/{pid}/icr?coder_a={a_id}&coder_b={b_id}"
        )
        assert r.status_code == 200
        body = r.json()
        # 2 distinct (source, anchor_start) items: w-0 + w-5.
        assert body["n_items"] == 2
        # Both applied something on both items.
        assert body["n_both_applied_any"] == 2
        # per_code reports kappa for both code_a and code_b.
        per_code_ids = {row["code_id"] for row in body["per_code"]}
        assert code_a in per_code_ids
        assert code_b in per_code_ids
        # Each per_code row carries name + label + counts.
        for row in body["per_code"]:
            assert "code_name" in row
            assert "label" in row
            assert "n_a_applied" in row
            assert "n_b_applied" in row

    def test_icr_source_filter_narrows(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        a_id, b_id, sid, *_ = self._seed_two_coders_and_apps(client, pid)
        # Different source — no apps under it, so n_items=0
        other_sid = _make_source(client, pid, "Other")
        r = client.get(
            f"/api/projects/{pid}/icr?coder_a={a_id}&coder_b={b_id}"
            f"&source_id={other_sid}"
        )
        assert r.status_code == 200
        assert r.json()["n_items"] == 0

    def test_icr_overall_kappa_is_a_finite_number(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        a_id, b_id, *_ = self._seed_two_coders_and_apps(client, pid)
        r = client.get(
            f"/api/projects/{pid}/icr?coder_a={a_id}&coder_b={b_id}"
        )
        body = r.json()
        # kappa is a real number, not NaN. Range is [-1, 1].
        kappa = body["overall_kappa"]
        assert isinstance(kappa, (int, float))
        assert -1.0 <= kappa <= 1.0
        assert body["overall_label"] in (
            "poor", "slight", "fair", "moderate",
            "substantial", "almost perfect",
        )


# --------------------------------------------------------------------------- #
# Application list filter by coder_id (helper for the ICR view)
# --------------------------------------------------------------------------- #


class TestSourceCodingPagePicksUpActiveCoder:
    """The source-coding template's applyCode() reads the active coder
    id from localStorage and includes it in the POST. This test only
    asserts the JS surface: that the template *contains* the
    integration with the F2.5 active-coder key. The end-to-end
    behaviour is covered by TestApplicationCoderAttribution above
    (the server-side path)."""

    def test_source_coding_template_reads_active_coder_id(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # JS reads scribe.active_coder_id.<pid> from localStorage and
        # forwards it as coder_id on /api/projects/<pid>/applications.
        assert "scribe.active_coder_id" in r.text
        assert "coder_id" in r.text


class TestApplicationsListCoderIdFilter:
    def test_coder_id_filter_returns_only_that_coders_apps(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        cid = _make_code(client, pid)
        # Bootstrap default coder
        r = client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": cid, "source_id": sid,
                  "anchor_start_word_id": "s0w0", "anchor_end_word_id": "s0w1"},
        )
        default_coder = client.get(
            f"/api/projects/{pid}/coders"
        ).json()["coders"][0]["id"]
        # Second coder + their application
        r = client.post(
            f"/api/projects/{pid}/coders",
            json={"name": "Second"},
        )
        b = r.json()["id"]
        client.post(
            f"/api/projects/{pid}/applications",
            json={"code_id": cid, "source_id": sid,
                  "anchor_start_word_id": "s0w2", "anchor_end_word_id": "s0w3",
                  "coder_id": b},
        )
        # Filter
        a_apps = client.get(
            f"/api/projects/{pid}/applications?coder_id={default_coder}"
        ).json()["applications"]
        b_apps = client.get(
            f"/api/projects/{pid}/applications?coder_id={b}"
        ).json()["applications"]
        assert len(a_apps) == 1
        assert len(b_apps) == 1
        assert a_apps[0]["coder_id"] == default_coder
        assert b_apps[0]["coder_id"] == b

    def test_invalid_coder_id_filter_returns_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/applications?coder_id=not-hex")
        assert r.status_code == 400
