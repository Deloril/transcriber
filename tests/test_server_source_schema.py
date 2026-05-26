"""End-to-end reachability tests for F3.2 (Source attribute schema —
user-defined columns per source).

The pure data layer in ``scribe/source_schema.py`` shipped in 9468fae
with full unit coverage in ``tests/test_source_schema.py``. What was
missing — and what this file proves — is the **user-facing surface**:

  * GET /projects/<pid>/sources/schema renders a real schema editor
    page with a per-attribute row form.
  * The page reads schema state via GET /api/projects/<pid>/source_schema.
  * Submitting the form (PUT /api/projects/<pid>/source_schema)
    persists an ``AttributeDefinition`` list to disk and reloading
    surfaces the saved values via the same loader.
  * The sources list page (/projects/<pid>/sources) renders a column
    per attribute defined in the schema, so the user-defined columns
    F3.2 promises actually show up where the user looks for them.
  * The sources list links to the schema editor so the schema is
    reachable from the listing it controls.

The schema editor is the F3.2 sibling of project_settings.html for
F3.1: same structural pattern (page route renders a real form,
data-test-feature marker, loader/saver via the JSON API).
"""

from __future__ import annotations

import json
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


def _make_project(client: TestClient, name: str = "Schema test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    *,
    name: str = "S1",
    custom_attributes: dict | None = None,
) -> str:
    r = client.post(
        f"/api/projects/{pid}/sources",
        json={
            "name": name,
            "source_type": "transcript",
            "custom_attributes": custom_attributes or {},
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------------- #
# Schema editor page renders
# --------------------------------------------------------------------------- #


class TestSchemaPageRenders:
    """``/projects/<pid>/sources/schema`` must render a real form, not
    a wireframe stub. Without the data-test-feature marker the F3.2
    surface isn't proven to be the schema editor."""

    def test_renders_with_200(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        assert r.status_code == 200

    def test_marks_F3_2_feature(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        assert 'data-test-feature="F3.2"' in r.text
        assert "Source attribute schema" in r.text

    def test_page_has_add_row_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        # The "+ Add attribute" button is the primary action.
        assert 'data-test-id="ss-add-row"' in r.text
        assert "+ Add attribute" in r.text

    def test_page_has_save_button(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        assert 'data-test-id="ss-submit"' in r.text
        assert "Save schema" in r.text

    def test_page_offers_all_attribute_types(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        # The five canonical types from scribe.source_schema.ATTRIBUTE_TYPES
        # must each be a select option in the row template.
        for t in ("text", "number", "date", "boolean", "select"):
            assert f'value="{t}"' in r.text

    def test_page_loads_via_api(self, server_env) -> None:
        """The form's loader fetches the F3.2 GET endpoint."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources/schema")
        assert f"/api/projects/${{PROJECT_ID}}/source_schema" in r.text \
            or "/source_schema" in r.text
        # Saver uses PUT.
        assert 'method: "PUT"' in r.text


# --------------------------------------------------------------------------- #
# GET /api/projects/<pid>/source_schema — empty schema fallback
# --------------------------------------------------------------------------- #


class TestSchemaGetEndpoint:
    def test_returns_empty_schema_when_unset(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/api/projects/{pid}/source_schema")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project_id"] == pid
        assert body["attributes"] == []

    def test_returns_404_for_unknown_project(self, server_env) -> None:
        _, client, _ = server_env
        # 12-hex project id that doesn't exist on disk (matches the
        # PROJECT_ID_RE regex but no project.json is present).
        r = client.get("/api/projects/abcdef012345/source_schema")
        assert r.status_code == 404

    def test_returns_400_for_invalid_project_id(self, server_env) -> None:
        _, client, _ = server_env
        r = client.get("/api/projects/!!!/source_schema")
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# PUT /api/projects/<pid>/source_schema — round-trip
# --------------------------------------------------------------------------- #


class TestSchemaPutEndpoint:
    def test_put_round_trips_via_get(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        payload = {
            "attributes": [
                {"key": "site", "label": "Site",
                 "type": "select", "required": True,
                 "options": ["Hospital A", "Hospital B"]},
                {"key": "wave", "label": "Interview wave",
                 "type": "number", "required": False, "options": []},
            ]
        }
        r = client.put(
            f"/api/projects/{pid}/source_schema", json=payload
        )
        assert r.status_code == 200, r.text
        body = r.json()
        keys = [a["key"] for a in body["attributes"]]
        assert keys == ["site", "wave"]
        # GET sees the same shape.
        r2 = client.get(f"/api/projects/{pid}/source_schema")
        assert r2.status_code == 200
        keys2 = [a["key"] for a in r2.json()["attributes"]]
        assert keys2 == ["site", "wave"]

    def test_put_persists_on_disk(self, server_env) -> None:
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [
                {"key": "consent", "type": "boolean", "required": True},
            ]},
        )
        assert r.status_code == 200, r.text
        # File on disk:
        path = tmp_path / "projects" / pid / "source_schema.json"
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert on_disk["project_id"] == pid
        assert on_disk["attributes"][0]["key"] == "consent"
        assert on_disk["attributes"][0]["type"] == "boolean"
        assert on_disk["attributes"][0]["required"] is True

    def test_put_replaces_existing_schema(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [{"key": "alpha", "type": "text"}]},
        )
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [{"key": "beta", "type": "text"}]},
        )
        assert r.status_code == 200
        keys = [a["key"] for a in r.json()["attributes"]]
        assert keys == ["beta"]

    def test_put_accepts_empty_attributes(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        # Saving an empty schema is a legitimate operation (clear the
        # column set). It must not 400.
        r = client.put(
            f"/api/projects/{pid}/source_schema", json={"attributes": []}
        )
        assert r.status_code == 200, r.text
        assert r.json()["attributes"] == []

    def test_put_rejects_invalid_type_with_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [
                {"key": "x", "type": "blob"},
            ]},
        )
        assert r.status_code == 400, r.text

    def test_put_rejects_select_with_no_options_with_400(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [
                {"key": "site", "type": "select", "options": []},
            ]},
        )
        assert r.status_code == 400, r.text

    def test_put_rejects_duplicate_keys_with_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [
                {"key": "site", "type": "text"},
                {"key": "site", "type": "text"},
            ]},
        )
        assert r.status_code == 400, r.text

    def test_put_rejects_non_dict_body_with_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema", json=["not", "an", "object"]
        )
        assert r.status_code == 400

    def test_put_rejects_attributes_non_list_with_400(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": "not a list"},
        )
        assert r.status_code == 400

    def test_put_rejects_unknown_project_with_404(self, server_env) -> None:
        _, client, _ = server_env
        r = client.put(
            "/api/projects/abcdef012345/source_schema",
            json={"attributes": []},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Sources list page surfaces schema-driven columns
# --------------------------------------------------------------------------- #


class TestSourcesListUsesSchema:
    """``/projects/<pid>/sources`` must render the schema editor link
    (so the F3.2 page is reachable from the table it controls), and
    its JS must fetch the schema and render a column per attribute.
    Without these the F3.2 columns are unreachable from the listing."""

    def test_sources_list_links_to_schema_editor(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert r.status_code == 200
        assert f'href="/projects/{pid}/sources/schema"' in r.text
        assert 'data-test-id="src-schema-link"' in r.text

    def test_sources_list_marks_F3_2_feature(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert 'data-test-feature="F3.2"' in r.text

    def test_sources_list_fetches_schema_endpoint(self, server_env) -> None:
        """The page's JS must call the F3.2 GET endpoint — otherwise
        the columns aren't reachable from the table at all."""
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/sources")
        assert "/source_schema" in r.text

    def test_sources_list_renders_attribute_keys(self, server_env) -> None:
        """When the schema has attributes, the page must render row
        cells keyed by attribute key. We can't easily run the JS in a
        TestClient, so we assert the render path via the JS source."""
        _, client, _ = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [
                {"key": "site", "type": "text"},
            ]},
        )
        r = client.get(f"/projects/{pid}/sources")
        # Server-side: the schema is reachable, and the page hands the
        # JS what it needs to render columns.
        assert r.status_code == 200
        # The render path uses data-attr-key cells and an attr-col header.
        assert "data-attr-key" in r.text
        assert "attr-col" in r.text


# --------------------------------------------------------------------------- #
# End-to-end: sources list + schema columns, post-population
# --------------------------------------------------------------------------- #


class TestSchemaShapesSourceList:
    """Cross-feature reachability: F3.2's schema and F1.2's sources
    align. PATCHing a source's custom_attributes via the existing
    route surfaces values that the schema describes."""

    def test_source_custom_attributes_via_patch(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [{"key": "site", "type": "text"}]},
        )
        sid = _make_source(client, pid, name="Interview 1")
        # PATCH the source with attribute values.
        r = client.patch(
            f"/api/projects/{pid}/sources/{sid}",
            json={"custom_attributes": {"site": "Hospital A"}},
        )
        assert r.status_code == 200, r.text
        # The list endpoint surfaces the attribute alongside the source.
        listing = client.get(f"/api/projects/{pid}/sources").json()
        rows = listing["sources"]
        assert len(rows) == 1
        assert rows[0]["custom_attributes"] == {"site": "Hospital A"}

    def test_schema_persists_independent_of_sources(self, server_env) -> None:
        """Saving the schema first, then adding sources, must work —
        and the schema must still be retrievable after sources have
        been added (no accidental coupling between the two)."""
        _, client, _ = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [{"key": "lang", "type": "text"}]},
        )
        _make_source(client, pid, name="A")
        _make_source(client, pid, name="B")
        r = client.get(f"/api/projects/{pid}/source_schema")
        assert r.status_code == 200
        keys = [a["key"] for a in r.json()["attributes"]]
        assert keys == ["lang"]

    def test_delete_project_wipes_schema(self, server_env) -> None:
        """A project's schema lives inside its directory; deleting the
        project must wipe the schema along with everything else."""
        srv, client, tmp_path = server_env
        pid = _make_project(client)
        client.put(
            f"/api/projects/{pid}/source_schema",
            json={"attributes": [{"key": "k", "type": "text"}]},
        )
        path = tmp_path / "projects" / pid / "source_schema.json"
        assert path.exists()
        r = client.delete(f"/api/projects/{pid}")
        assert r.status_code == 200, r.text
        assert not path.exists()
