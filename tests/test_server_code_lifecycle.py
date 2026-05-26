"""Tests for the F2.3 code-lifecycle REST surface.

The pure module (`scribe.code_lifecycle`) has its own deep test
coverage in `tests/test_code_lifecycle.py`. This file exists solely to
prove **reachability**: that the rename / retire / parent / promote /
merge / split ops are wired through FastAPI and that the codebook
editor template surfaces controls that hit those endpoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


# --------------------------------------------------------------------------- #
# Test client + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _new_project(client: TestClient, name: str = "Test project") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _new_code(
    client: TestClient,
    pid: str,
    *,
    name: str = "managing pain",
    definition: str = "v1 — first pass",
    parent_code_id: str | None = None,
) -> str:
    payload: dict = {"name": name, "definition": definition}
    if parent_code_id is not None:
        payload["parent_code_id"] = parent_code_id
    r = client.post(f"/api/projects/{pid}/codes", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _lock_codebook(client: TestClient, pid: str) -> None:
    """Lock the codebook via the existing F2.4 endpoint, if exposed."""
    # The lock toggle isn't a public route yet (W2.4); flip the project's
    # codebook_stage directly via the projects PATCH if available, else
    # via the lower-level helper.
    from scribe.projects import load_project, save_project
    p = load_project(srv._projects_root(), pid)
    p.apply_update({"codebook_stage": "locked"})
    save_project(srv._projects_root(), p)


# --------------------------------------------------------------------------- #
# Rename
# --------------------------------------------------------------------------- #


class TestRenameEndpoint:
    def test_rename_round_trips(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid, name="pacing")
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/rename",
            json={"name": "managing pacing", "change_note": "broadened"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code"]["id"] == cid
        assert body["code"]["name"] == "managing pacing"
        assert body["version"] is not None
        # The version is recorded in F2.2's log — verify via the GET.
        r2 = client.get(f"/api/projects/{pid}/codes/{cid}/versions")
        assert r2.status_code == 200
        names = [v["snapshot"]["name"] for v in r2.json()["versions"]]
        assert "pacing" in names
        assert "managing pacing" in names

    def test_rename_rejects_empty(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/rename",
            json={"name": "   "},
        )
        assert r.status_code == 400

    def test_rename_404_for_unknown_code(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.post(
            f"/api/projects/{pid}/codes/aaaaaaaaaaaa/rename",
            json={"name": "x"},
        )
        assert r.status_code == 404

    def test_rename_blocked_on_locked_codebook(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        _lock_codebook(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/rename",
            json={"name": "anything"},
        )
        assert r.status_code == 409


# --------------------------------------------------------------------------- #
# Retire
# --------------------------------------------------------------------------- #


class TestRetireEndpoint:
    def test_retire_marks_status(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/retire",
            json={"change_note": "no longer used"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code"]["status"] == "retired"

    def test_retire_idempotent(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        r1 = client.post(f"/api/projects/{pid}/codes/{cid}/retire", json={})
        r2 = client.post(f"/api/projects/{pid}/codes/{cid}/retire", json={})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["code"]["status"] == "retired"


# --------------------------------------------------------------------------- #
# Parent / promote
# --------------------------------------------------------------------------- #


class TestParentEndpoint:
    def test_set_parent_attaches_child(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        parent = _new_code(client, pid, name="pain")
        child = _new_code(client, pid, name="pacing")
        r = client.post(
            f"/api/projects/{pid}/codes/{child}/parent",
            json={"parent_code_id": parent},
        )
        assert r.status_code == 200, r.text
        assert r.json()["code"]["parent_code_id"] == parent

    def test_set_parent_clears_with_null(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        parent = _new_code(client, pid, name="pain")
        child = _new_code(client, pid, name="pacing", parent_code_id=parent)
        r = client.post(
            f"/api/projects/{pid}/codes/{child}/parent",
            json={"parent_code_id": None},
        )
        assert r.status_code == 200, r.text
        assert r.json()["code"]["parent_code_id"] is None

    def test_set_parent_rejects_cycle(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        a = _new_code(client, pid, name="a")
        b = _new_code(client, pid, name="b", parent_code_id=a)
        # Setting a's parent to b would close a cycle.
        r = client.post(
            f"/api/projects/{pid}/codes/{a}/parent",
            json={"parent_code_id": b},
        )
        assert r.status_code == 400

    def test_promote_lifts_one_level(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        gp = _new_code(client, pid, name="gp")
        p = _new_code(client, pid, name="p", parent_code_id=gp)
        c = _new_code(client, pid, name="c", parent_code_id=p)
        r = client.post(f"/api/projects/{pid}/codes/{c}/promote", json={})
        assert r.status_code == 200, r.text
        assert r.json()["code"]["parent_code_id"] == gp


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


class TestMergeEndpoint:
    def test_merge_collapses_sources_into_target(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        target = _new_code(client, pid, name="managing pain")
        a = _new_code(client, pid, name="pacing")
        b = _new_code(client, pid, name="resting")
        r = client.post(
            f"/api/projects/{pid}/codes/merge",
            json={
                "target_code_id": target,
                "source_code_ids": [a, b],
                "change_note": "consolidating during focused pass",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["target"]["id"] == target
        retired_ids = [c["id"] for c in body["retired"]]
        assert set(retired_ids) == {a, b}
        for c in body["retired"]:
            assert c["status"] == "retired"
            assert c["provenance"].get("merged_into") == target

    def test_merge_rejects_target_in_sources(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        target = _new_code(client, pid, name="t")
        r = client.post(
            f"/api/projects/{pid}/codes/merge",
            json={"target_code_id": target, "source_code_ids": [target]},
        )
        assert r.status_code == 400

    def test_merge_requires_target(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        a = _new_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/merge",
            json={"source_code_ids": [a]},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


class TestSplitEndpoint:
    def test_split_creates_new_codes(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid, name="managing pain", definition="broad")
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/split",
            json={
                "new_codes": [
                    {"name": "pacing", "definition": "physical pacing"},
                    {"name": "asking for help"},
                ],
                "change_note": "more granular categories",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"]["id"] == cid
        assert body["source"]["status"] == "retired"
        new_names = sorted(c["name"] for c in body["new_codes"])
        assert new_names == ["asking for help", "pacing"]
        for c in body["new_codes"]:
            assert c["provenance"].get("split_from") == cid
        # The retired source's provenance lists the new ids.
        split_into = body["source"]["provenance"]["split_into"].split(",")
        assert sorted(split_into) == sorted(c["id"] for c in body["new_codes"])

    def test_split_requires_two_new_codes(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/split",
            json={"new_codes": [{"name": "only one"}]},
        )
        assert r.status_code == 400

    def test_split_rejects_missing_name(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        cid = _new_code(client, pid)
        r = client.post(
            f"/api/projects/{pid}/codes/{cid}/split",
            json={"new_codes": [{"name": "ok"}, {"definition": "no name"}]},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Reachability through the codebook editor template
# --------------------------------------------------------------------------- #


class TestCodebookEditorRendersLifecycleControls:
    """The codebook editor must surface the F2.3 lifecycle ops as a
    per-row ⋮ menu so the routes above are reachable through the UI.

    These assertions are deliberately phrased against the rendered HTML
    rather than against the JS module, so a refactor that breaks the
    user-facing surface shows up here."""

    def test_page_renders_with_lifecycle_menu(self, env) -> None:
        client, _ = env
        pid = _new_project(client)
        r = client.get(f"/projects/{pid}/codebook")
        assert r.status_code == 200, r.text
        text = r.text
        # Per-row menu button + menu container.
        assert "cb-row-menu-btn" in text
        assert "cb-row-menu" in text
        # Each lifecycle action surfaced in the menu.
        for action in ("rename", "retire", "promote", "merge", "split"):
            assert f'data-act="{action}"' in text, (
                f"Missing data-act='{action}' in codebook editor template"
            )
        # The JS must call each lifecycle endpoint we just added.
        for ep in ("/rename", "/retire", "/parent", "/promote", "/codes/merge", "/split"):
            assert ep in text, f"Codebook editor doesn't reference {ep}"
