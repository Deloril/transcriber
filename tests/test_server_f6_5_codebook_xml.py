"""F6.5 reachability verification — REFI-QDA Codebook XML download surface.

The pure renderer (``scribe.codebook_export.to_refi_qda_xml`` /
``render_refi_qda_codebook_xml``) shipped in 4aef4fb (F2.6) and
542bc18 (F6.5); the FastAPI route
``GET /api/projects/<pid>/codebook/refi-qda-xml`` shipped in 542bc18.
The codebook editor's '📥 Export ▾' dropdown that surfaces the URL
shipped in 0e93039 (F2.6 wiring). The 542bc18 commit body did not
include a ``Reachable-via:`` line, so the loop's done-detector treats
F6.5 as incomplete. This file is the explicit reachability anchor —
it asserts the codebook editor renders the REFI-QDA XML download
link, that the link targets the right URL, and that hitting the URL
returns the expected XML body + headers.

Deeper coverage of the renderer + endpoint lives in:

  * ``tests/test_codebook_export.py``
    (TestRefiQdaXmlRender / TestSlugifyRefiQdaCodebookXmlFilename / …)
  * ``tests/test_server.py::TestExportCodebookRefiQdaXmlAPI``
  * ``tests/test_server_codebook_export_ui.py``
    (the F2.6 dropdown that exposes the link)

This file is intentionally narrow: it is the F6.5 "you can click this
button and get the file" contract. If the dropdown stops rendering
the REFI-QDA XML link, or the endpoint stops returning an XML body,
the loop's audit grep should land here.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

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
    yield srv, client, projects


def _new_project(client: TestClient, name: str = "Pilot study", **fields) -> str:
    payload = {"name": name}
    payload.update(fields)
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_code(projects_dir: Path, project_id: str, name: str = "Pacing") -> str:
    """Persist one code so the export endpoint has something to render."""
    from scribe import codes as _codes
    from scribe.codes import Code

    code = Code.new(
        project_id=project_id,
        name=name,
        definition="Adjusting daily activity to manage limited energy.",
    )
    _codes.save_code(projects_dir, code)
    return code.id


# --------------------------------------------------------------------------- #
# Template render: the codebook editor surfaces the F6.5 download link
# --------------------------------------------------------------------------- #


class TestCodebookEditorRendersF6_5Link:
    """The codebook editor's Export dropdown must include the
    REFI-QDA Codebook XML target. The F6.5 endpoint deliberately lives
    on its own URL (separate from the F6.1 ``format=`` switch); the
    dropdown is the only first-party path to it."""

    def test_export_button_present(self, env) -> None:
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        # The Export dropdown is the umbrella surface; F6.5 lives in it.
        assert 'id="cb-export-btn"' in r.text
        assert 'id="cb-export-menu"' in r.text

    def test_refi_qda_label_in_menu(self, env) -> None:
        """The menu must name the format with a string a researcher
        will recognise (REFI-QDA), not just a URL fragment."""
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert "REFI-QDA" in r.text

    def test_refi_qda_link_targets_correct_endpoint(self, env) -> None:
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        # F6.5's URL is project-scoped and stable.
        assert f"/api/projects/{pid}/codebook/refi-qda-xml" in r.text

    def test_refi_qda_link_carries_download_attribute(self, env) -> None:
        """``download`` on the anchor tells the browser to save the body
        rather than try to render the XML in a viewer (some browsers
        will pretty-print XML inline by default)."""
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        text = r.text
        idx = text.find(f"/api/projects/{pid}/codebook/refi-qda-xml")
        assert idx > 0, "REFI-QDA XML href not in codebook editor"
        # Anchor tag wrapping the href lives in a small radius.
        snippet = text[max(0, idx - 400): idx + 400]
        assert "download" in snippet
        assert 'data-fmt="refi-qda-xml"' in snippet


# --------------------------------------------------------------------------- #
# Endpoint reachability: clicking the rendered href lands on a real XML body
# --------------------------------------------------------------------------- #


class TestRefiQdaXmlEndpointReachableFromUi:
    """The href the codebook editor renders must resolve to the F6.5
    endpoint and return a parseable REFI-QDA Codebook XML body."""

    def test_endpoint_returns_200_with_codebook_xml(self, env) -> None:
        _, client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid, name="Pacing")
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        assert r.status_code == 200, r.text
        # Body parses as XML and the root is in the REFI-QDA Codebook
        # 1.0 namespace.
        root = ET.fromstring(r.text)
        assert root.tag.endswith("}CodeBook")
        assert "urn:QDA-XML:codebook:1.0" in r.text

    def test_content_type_is_application_xml_utf8(self, env) -> None:
        _, client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid)
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        ct = r.headers["content-type"]
        assert ct.startswith("application/xml")
        assert "charset=utf-8" in ct

    def test_attachment_filename_matches_f6_5_extension(self, env) -> None:
        """F6.5 uses the dotted ``.refi-qda.xml`` extension so the file
        survives a downloads-folder mix with arbitrary ``.xml`` artefacts.
        """
        _, client, _ = env
        pid = _new_project(client, name="Pilot")
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        cd = r.headers["content-disposition"]
        assert "attachment" in cd
        assert ".refi-qda.xml" in cd
        assert 'filename="pilot-codebook.refi-qda.xml"' in cd

    def test_codes_round_trip_into_xml_body(self, env) -> None:
        _, client, projects_dir = env
        pid = _new_project(client)
        _seed_code(projects_dir, pid, name="Pacing")
        _seed_code(projects_dir, pid, name="Resting")
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        from scribe.codebook_export import REFI_QDA_NS

        root = ET.fromstring(r.text)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        names = sorted(
            (c.get("name") or "")
            for c in codes_el.findall(f"{{{REFI_QDA_NS}}}Code")
        )
        assert names == ["Pacing", "Resting"]

    def test_project_metadata_comment_emitted(self, env) -> None:
        """F6.5's wrapper always passes ``include_project_metadata=True``
        so methodology / RQ context lands in an XML comment alongside
        the codebook (the schema has no native slot)."""
        _, client, _ = env
        pid = _new_project(
            client,
            name="Pilot",
            methodology="charmaz",
            research_question="How do people pace energy?",
        )
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        text = r.text
        assert "<!--" in text
        assert "Methodology: charmaz" in text
        assert "How do people pace energy?" in text


# --------------------------------------------------------------------------- #
# F6.1 isolation guard: the format=xml path stays rejected on the F6.1 endpoint
# --------------------------------------------------------------------------- #


class TestF6_5DoesNotLeakIntoF6_1Surface:
    """F6.5 lives on its own URL specifically so the F6.1
    ``format=`` switch can stay tight (only csv / markdown / rtf).
    Future refactors must not collapse the two surfaces."""

    def test_format_xml_rejected_on_export_endpoint(self, env) -> None:
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/codebook/export?format=xml"
        )
        assert r.status_code == 400

    def test_format_refi_qda_rejected_on_export_endpoint(self, env) -> None:
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(
            f"/api/projects/{pid}/codebook/export?format=refi-qda"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Empty codebook: F6.5 must not 5xx when a researcher exports before coding
# --------------------------------------------------------------------------- #


class TestEmptyCodebookF6_5:
    """A brand-new project's codebook export must still produce a valid
    REFI-QDA XML body. This is the failure mode the F2.6 / F6.5
    renderers were both written to handle — guarding it here keeps the
    UI's claim ('Export ▾' on every project) honest."""

    def test_empty_returns_200_minimal_xml(self, env) -> None:
        _, client, _ = env
        pid = _new_project(client)
        r = client.get(f"/api/projects/{pid}/codebook/refi-qda-xml")
        assert r.status_code == 200
        from scribe.codebook_export import REFI_QDA_NS

        root = ET.fromstring(r.text)
        codes_el = root.find(f"{{{REFI_QDA_NS}}}Codes")
        assert codes_el is not None
        assert len(codes_el) == 0


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestRefiQdaXmlFailureModes:
    """The endpoint must fail gracefully on missing / malformed ids."""

    def test_404_when_project_missing(self, env) -> None:
        _, client, _ = env
        r = client.get(
            f"/api/projects/{'0' * 12}/codebook/refi-qda-xml"
        )
        assert r.status_code == 404

    def test_400_on_malformed_project_id(self, env) -> None:
        _, client, _ = env
        r = client.get(
            "/api/projects/not-hex/codebook/refi-qda-xml"
        )
        assert r.status_code == 400
