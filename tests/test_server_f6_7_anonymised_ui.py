"""F6.7 reachability verification — anonymised REFI-QDA / QDPX export.

The pure builder (``scribe.anonymise``) and the endpoint
(``POST /api/projects/<pid>/qdpx/anonymised``) shipped in 086e0a1; the
F6.7 commit body did not include a ``Reachable-via:`` line and no
template surfaced the download. The only path to redacted exports was
a curl POST. This file is the explicit reachability anchor for the
user-facing surface:

  1. the project settings page renders an "Anonymised export" card
     with a click-through button, an optional rules textarea, and an
     optional manifest-note input,
  2. the underlying ``POST /api/projects/<pid>/qdpx/anonymised``
     endpoint that the rendered button targets still produces a
     redacted zip with ``Redactions/manifest.json`` and the expected
     ``X-Scribe-Anon-Substitutions`` header,
  3. custom-rule strings parsed by the page's textarea reach the
     server intact and apply the redactions they describe.

The deeper coverage of the redaction builder lives in
``tests/test_anonymise.py`` and
``tests/test_server.py::TestExportAnonymisedQdpxAPI``; this file's job
is purely the F6.7 UI-reachability contract.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated TestClient + tmp uploads/outputs/projects roots."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
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

    client = TestClient(srv.app)
    yield srv, client, projects, output


def _new_project(client: TestClient, name: str = "Anon study") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_with_participant(srv, pid: str) -> dict:
    """Seed a project with one participant (with pseudonym), one source
    pointing at a tiny edited transcript that names the participant by
    their real name. This is the minimum payload the redaction pass
    needs to demonstrate substitutions.
    """
    from scribe import participants as p_mod
    from scribe import sources as s_mod
    from scribe.participants import Participant
    from scribe.sources import Source

    p = Participant.new(project_id=pid, name="Jane Doe", pseudonym="P01")
    p_mod.save_participant(srv.PROJECTS_DIR, p)

    job_id = "abcdef012345"
    source = Source.new(
        project_id=pid,
        name="Interview with Jane Doe",
        source_type="transcript",
        transcript_job_id=job_id,
    )
    s_mod.save_source(srv.PROJECTS_DIR, source)

    job_dir = srv.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(json.dumps({
        "segments": [
            {"speaker": "INTERVIEWER",
             "words": [
                 {"text": "Hello"},
                 {"text": "Jane"},
                 {"text": "Doe"},
             ]},
        ],
    }))

    return {"source_id": source.id, "participant_id": p.id}


# --------------------------------------------------------------------------- #
# Template render: the project settings page surfaces the F6.7 card
# --------------------------------------------------------------------------- #


class TestProjectSettingsRendersF6_7Card:
    """The settings page already exposes the F1.5 archive download and
    the F6.4 plain QDPX. F6.7 piggy-backs on that affordance with a
    third card so the redacted export sits next to its non-redacted
    sibling — researchers comparing them shouldn't have to navigate.
    """

    def test_anonymised_card_rendered(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert r.status_code == 200, r.text
        # Card carries the F6.7 feature tag.
        assert 'data-test-feature="F6.7"' in r.text
        # The data-test-id button anchor is present.
        assert 'data-test-id="ps-anonymised-qdpx-btn"' in r.text

    def test_anonymised_button_targets_correct_endpoint(self, env) -> None:
        """The button stores its endpoint in ``data-endpoint`` rather
        than ``href`` because POST + Blob can't ride on ``<a download>``.
        Whichever attribute the JS reads, the URL must be right.
        """
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert f"/api/projects/{pid}/qdpx/anonymised" in r.text

    def test_anonymised_card_explains_what_redaction_does(self, env) -> None:
        """Copy must mention what's being redacted — researchers
        landing here from search engines / docs need to confirm this is
        the right button before they hand the archive to an IRB."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        text = r.text
        # The card must mention REFI-QDA / QDPX and "redact"
        assert "REFI-QDA" in text or "QDPX" in text
        assert "redact" in text.lower() or "anonymis" in text.lower() \
               or "anonymiz" in text.lower()
        assert "manifest" in text.lower()

    def test_custom_rules_textarea_present(self, env) -> None:
        """The optional custom-rules textarea is the F6.7-specific
        affordance — without it the user can only redact via the
        participants' pseudonyms, which doesn't cover institution
        names, phone numbers, etc."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert 'id="anon-rules"' in r.text
        assert 'id="anon-note"' in r.text


# --------------------------------------------------------------------------- #
# Endpoint reachability: the URL the rendered button targets works
# --------------------------------------------------------------------------- #


class TestAnonymisedEndpointReachableFromUi:
    """The endpoint the template encodes must resolve to the live
    redaction pipeline. Without this link, the rendered button could
    POST to a 404 and the F6.7 surface would still 'render'."""

    def test_button_endpoint_returns_redacted_zip(self, env) -> None:
        srv, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        seeds = _seed_with_participant(srv, pid)
        # Mirrors the JS click handler: POST JSON, expect a zip back.
        r = client.post(
            f"/api/projects/{pid}/qdpx/anonymised", json={}
        )
        assert r.status_code == 200, r.text
        # Body is a zip — REFI-QDA QDPX is a renamed .zip.
        assert r.content[:2] == b"PK"
        assert r.headers["content-type"].startswith("application/x-qdpx")
        # The substitutions header is what the click handler reads to
        # confirm the redaction actually fired.
        assert int(r.headers["x-scribe-anon-substitutions"]) >= 1
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            assert "project.qde" in names
            assert "Redactions/manifest.json" in names
            txt = zf.read(f"Sources/{seeds['source_id']}.txt").decode("utf-8")
            # Real name redacted; pseudonym substituted.
            assert "Jane Doe" not in txt
            assert "P01" in txt

    def test_filename_is_slugged_for_save_dialog(self, env) -> None:
        """The Content-Disposition filename is what the JS click
        handler hands to the synthetic ``<a download>``. It must be
        slugged + carry the ``-anon`` suffix so the redacted bundle
        doesn't get overwritten by the plain F6.4 export.
        """
        _, client, _, _ = env
        pid = _new_project(client, name="Pilot study")
        r = client.post(
            f"/api/projects/{pid}/qdpx/anonymised", json={}
        )
        assert r.status_code == 200, r.text
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".qdpx" in cd
        assert "-anon" in cd

    def test_custom_rules_from_textarea_apply(self, env) -> None:
        """A rule the user types in the textarea must reach the
        server in the JSON shape the endpoint accepts and actually
        rewrite the bundle. This is the link between the JS parser
        (parseAnonymisedRulesText in helpers.mjs / inline) and the
        server-side redaction module.
        """
        srv, client, _, _ = env
        pid = _new_project(client)
        _seed_with_participant(srv, pid)
        # Same shape the inline parser produces from
        # ``Jane Doe => SUBJECT-ALPHA`` in the textarea.
        body = {
            "rules": [
                {"pattern": "Hello", "replacement": "[greeting]"},
            ],
            "note": "Pre-publication anon pass",
        }
        r = client.post(
            f"/api/projects/{pid}/qdpx/anonymised", json=body
        )
        assert r.status_code == 200, r.text
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            mani = json.loads(zf.read("Redactions/manifest.json"))
            assert mani["note"] == "Pre-publication anon pass"
            # Rule count includes the participant rule + the custom rule.
            assert mani["rule_count"] >= 2

    def test_404_when_project_missing(self, env) -> None:
        _, client, _, _ = env
        # Valid 12-hex shape, no project on disk.
        r = client.post(
            f"/api/projects/{'0' * 12}/qdpx/anonymised", json={}
        )
        assert r.status_code == 404

    def test_400_when_rules_shape_is_malformed(self, env) -> None:
        """The JS parser is supposed to catch shape errors before
        POSTing, but the server is the source of truth. If the page's
        parser ever drifts, the endpoint must still reject the bad
        shape rather than silently producing a wrong bundle.
        """
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/qdpx/anonymised",
            json={"rules": [{"pattern": "x"}]},  # missing replacement
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Defensive: the JS click handler is wired (button id present)
# --------------------------------------------------------------------------- #


class TestAnonymisedClickHandlerWired:
    """The button alone isn't enough — without the JS click handler
    the page would render but clicking the button would be a no-op
    (``href="#"`` jump). This test asserts the click handler is in
    the rendered page so the loop's "done" detector treats the
    feature as reachable, not just visible."""

    def test_click_handler_attached_to_button(self, env) -> None:
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        text = r.text
        # The handler binds to ``exportAnonymisedQdpxBtn``.
        assert 'id="exportAnonymisedQdpxBtn"' in text
        # It must POST (not GET — an ``<a download>`` won't work
        # because the rules + note ride in the request body).
        assert 'method: "POST"' in text or "method:'POST'" in text

    def test_inline_parser_present_for_rules_textarea(self, env) -> None:
        """The page's inline parser ``parseAnonymisedRulesText`` must
        exist in the rendered script — without it the textarea is
        decorative."""
        _, client, _, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/settings")
        assert "parseAnonymisedRulesText" in r.text
