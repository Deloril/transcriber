"""F8.12 reachability tests — baked-in model recommendations.

Per ``PLANNING.md`` F8.12:

  > Model recommendations baked in: laptop default Llama 3.2 3B or
  > Phi-3.5 3.8B; mid-tier Phi-4 14B or Mistral Nemo 12B; large-tier
  > Qwen 2.5 32B or Llama 3.3 70B. Embedding default ``bge-m3``
  > (multilingual) or ``nomic-embed-text-v1.5``.

The pure data plane shipped in ``scribe/model_recommendations.py``
(commit 967e663) and the wire format in
``GET /api/system/model-recommendations``. The user-facing surface was
graduated alongside F8.11 in commit 3c880fa (project AI page +
settings page tier picker card). The original F8.12 commit body did
not carry a ``Reachable-via:`` line, so this file pins the F8.12-
specific assertions independently:

* All six spec generative tags + both embedding tags appear in the
  recommendations endpoint payload.
* Each tier has exactly one default-flagged generative model + ≥1
  alternative.
* The endpoint payload is the same data the project AI page consumes
  (the page references the endpoint URL).
* The Pull button (``POST /api/projects/<pid>/ai/backend/pull``)
  accepts every F8.12 spec tag — proving the recommendations are not
  just labels but actually wired into the download manager.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from scribe.ai_backend import HTTPResponse


# --------------------------------------------------------------------------- #
# Spec tags from PLANNING.md F8.12. If F8.12's spec changes, update here.
# --------------------------------------------------------------------------- #

SPEC_GENERATIVE_TAGS: dict[str, set[str]] = {
    "small": {"llama3.2:3b", "phi3.5:3.8b"},
    "mid": {"phi4:14b", "mistral-nemo:12b"},
    "large": {"qwen2.5:32b", "llama3.3:70b"},
}

SPEC_EMBEDDING_TAGS: set[str] = {"bge-m3", "nomic-embed-text:v1.5"}


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/test_server_model_tiers_ui.py)
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
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    monkeypatch.setattr(srv, "_model_tiers_snapshot_override", None)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _make_project(client: TestClient, name: str = "F8.12 reachability") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


class _StubTransport:
    """Minimal Ollama transport stub — same shape as
    tests/test_server_ai_backend.py::StubTransport."""

    def __init__(self, routes: dict[tuple[str, str], HTTPResponse]) -> None:
        self.routes = dict(routes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HTTPResponse:
        # Match path-suffix; the project AI backend builds the full URL
        # from the saved base_url + endpoint path.
        for (m, path), resp in self.routes.items():
            if m == method and url.endswith(path):
                self.calls.append(
                    {"method": method, "url": url, "body": body}
                )
                return resp
        raise AssertionError(f"unexpected {method} {url}")


# --------------------------------------------------------------------------- #
# Endpoint shape: every F8.12 spec tag is present
# --------------------------------------------------------------------------- #


class TestF8_12RecommendationsEndpoint:
    def test_all_spec_generative_tags_present_per_tier(
        self, server_env
    ) -> None:
        _srv, client, _ = server_env
        r = client.get("/api/system/model-recommendations")
        assert r.status_code == 200, r.text
        body = r.json()
        tier_models: dict[str, set[str]] = {
            tier["id"]: {m["tag"] for m in tier["recommended_models"]}
            for tier in body["tiers"]
        }
        for tier_id, expected in SPEC_GENERATIVE_TAGS.items():
            present = tier_models.get(tier_id, set())
            missing = expected - present
            assert not missing, (
                f"F8.12 tier {tier_id!r} is missing spec tags "
                f"{sorted(missing)!r}; got {sorted(present)!r}"
            )

    def test_all_spec_embedding_tags_present(self, server_env) -> None:
        _srv, client, _ = server_env
        r = client.get("/api/system/model-recommendations")
        body = r.json()
        embed_tags = {m["tag"] for m in body["embedding_models"]}
        missing = SPEC_EMBEDDING_TAGS - embed_tags
        assert not missing, (
            f"F8.12 embedding pool is missing spec tags "
            f"{sorted(missing)!r}; got {sorted(embed_tags)!r}"
        )

    def test_each_tier_has_exactly_one_default_generative(
        self, server_env
    ) -> None:
        _srv, client, _ = server_env
        body = client.get("/api/system/model-recommendations").json()
        for tier in body["tiers"]:
            defaults = [
                m for m in tier["recommended_models"] if m.get("is_default")
            ]
            assert len(defaults) == 1, (
                f"tier {tier['id']!r} should have exactly one default "
                f"recommended model; got {len(defaults)}"
            )

    def test_embedding_pool_has_exactly_one_default(self, server_env) -> None:
        _srv, client, _ = server_env
        body = client.get("/api/system/model-recommendations").json()
        defaults = [
            m for m in body["embedding_models"] if m.get("is_default")
        ]
        assert len(defaults) == 1
        # Per the spec, the multilingual default is bge-m3.
        assert defaults[0]["tag"] == "bge-m3"

    def test_every_recommended_model_carries_required_metadata(
        self, server_env
    ) -> None:
        """Pin the wire shape — the UI consumes ``tag``, ``family``,
        ``display_name``, ``parameter_size_b``, ``kind``,
        ``is_default``, ``notes``."""
        _srv, client, _ = server_env
        body = client.get("/api/system/model-recommendations").json()
        required_keys = {
            "tag",
            "family",
            "display_name",
            "parameter_size_b",
            "kind",
            "is_default",
            "notes",
        }
        for tier in body["tiers"]:
            for m in tier["recommended_models"]:
                assert required_keys <= set(m.keys()), (
                    f"tier {tier['id']!r} model {m.get('tag')!r} "
                    f"missing keys {required_keys - set(m.keys())!r}"
                )
                assert m["kind"] == "generative"
        for m in body["embedding_models"]:
            assert required_keys <= set(m.keys())
            assert m["kind"] == "embedding"


# --------------------------------------------------------------------------- #
# UI surface: project AI page advertises F8.12 + consumes the endpoint
# --------------------------------------------------------------------------- #


class TestF8_12ProjectAIPageReachability:
    def test_project_ai_page_advertises_F8_12(self, server_env) -> None:
        """The tier picker card carries an explicit ``F8.12`` pill so a
        reader of the page (and grep of the page source) can find the
        feature without spelunking through the JS."""
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        assert "F8.12" in r.text

    def test_project_ai_page_calls_recommendations_endpoint(
        self, server_env
    ) -> None:
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert r.status_code == 200
        # The page's JS must hit /api/system/model-recommendations to
        # populate per-tier model picks (F8.12).
        assert "/api/system/model-recommendations" in r.text

    def test_project_ai_page_has_tier_picker_card(self, server_env) -> None:
        """The actual surface element — F8.11's data-test-feature
        wraps the F8.12 model rows."""
        _srv, client, _ = server_env
        pid = _make_project(client)
        r = client.get(f"/projects/{pid}/ai")
        assert 'data-test-id="ai-tier-picker-card"' in r.text
        assert 'data-test-id="ai-tier-list"' in r.text
        assert 'data-test-id="ai-embed-list"' in r.text

    def test_settings_page_advertises_F8_12(self, server_env) -> None:
        """The top-level Settings page also surfaces the F8.12 picks
        in a read-only verdict + per-tier model rows."""
        _srv, client, _ = server_env
        r = client.get("/settings")
        assert r.status_code == 200
        assert "F8.12" in r.text
        assert 'data-test-id="settings-tier-picker"' in r.text


# --------------------------------------------------------------------------- #
# Download manager: every F8.12 spec tag is pullable through the UI
# --------------------------------------------------------------------------- #


class TestF8_12PullEveryRecommendedTag:
    """Each F8.12 spec tag has to round-trip through
    ``POST /api/projects/<pid>/ai/backend/pull`` so the "Pull"
    button on the tier picker isn't theatre."""

    @pytest.mark.parametrize(
        "tag",
        sorted(
            {*SPEC_GENERATIVE_TAGS["small"],
             *SPEC_GENERATIVE_TAGS["mid"],
             *SPEC_GENERATIVE_TAGS["large"],
             *SPEC_EMBEDDING_TAGS}
        ),
    )
    def test_pull_route_accepts_recommended_tag(
        self, server_env, tag: str
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        ndjson = (
            json.dumps({"status": "pulling manifest"}) + "\n"
            + json.dumps({"status": "success"}) + "\n"
        ).encode("utf-8")
        srv._ai_backend_transport_override = _StubTransport(
            {
                ("POST", "/api/pull"): HTTPResponse(
                    status=200,
                    headers={"Content-Type": "application/x-ndjson"},
                    body=ndjson,
                ),
            }
        )
        try:
            r = client.post(
                f"/api/projects/{pid}/ai/backend/pull",
                json={"model": tag},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["success"] is True
            assert body["model"] == tag
        finally:
            srv._ai_backend_transport_override = None
