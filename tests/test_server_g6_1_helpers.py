"""Verification of reachability for G6.1 — smoke-test plan surface.

The G6.1 CLI smoke test (``scribe.scripts.check_rocm``) shipped in
commit ``ef375e1`` with deep unit coverage in
``tests/test_scripts_check_rocm.py`` (35 cases) but predates the
loop's ``Reachable-via:`` gate, so it left no audit trail of where
the smoke-test plan surfaces in user-facing output. The CLI is
itself a user-facing surface, but until this commit there was no
in-app way to discover it without dropping to the README.

This commit graduates the smoke-test plan into the home page and
into a JSON route the rest of the UI can consume:

  * ``GET /api/diagnostics/smoke-test-plan`` returns the plan
    (CLI invocation, ordered stages, defaults, exit codes,
    fail-fast policy) without importing torch / whisperx and
    without spawning the smoke test itself.

  * ``GET /`` renders a ``data-test-feature="G6.1"`` panel with
    the same data so a researcher who hasn't memorised the CLI
    can copy the invocation straight from the page.

Coverage matrix:

  1. The pure-helper ``smoke_test_plan()`` returns the expected
     contract (keys, ordered stages, defaults, exit codes).

  2. The route returns 200 and the same payload with the
     documented JSON shape.

  3. The home page renders the panel with the CLI command, the
     stage list, and the exit-code list — when the helper is
     present.

  4. The home page collapses cleanly when the helper raises
     (defensive — strip-down deploys without ``scribe.scripts``).

  5. End-to-end: route + template + helper agree on the CLI
     invocation strings, the stage order, and the exit codes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


INDEX_HTML = Path("scribe") / "templates" / "index.html"


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same fixture shape as the other ``test_server_g*_helpers.py``
    files. Copied so this file is self-contained."""
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


# --------------------------------------------------------------------------- #
# 1. The pure helper returns the documented contract
# --------------------------------------------------------------------------- #


class TestG6_1SmokeTestPlanContract:
    """The pure helper is consumed by both the route and the template,
    so its shape is part of the public contract — drift here breaks
    the JSON consumers and the home page render simultaneously."""

    def test_returns_a_dict(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        assert isinstance(plan, dict)

    def test_top_level_keys_are_pinned(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        # Adding new keys is fine; dropping any of these breaks the UI
        # or scripted callers.
        for key in (
            "feature_id",
            "cli",
            "cli_venv",
            "stages",
            "defaults",
            "exit_codes",
            "fail_fast",
            "docs_anchor",
        ):
            assert key in plan, f"smoke_test_plan() missing key: {key!r}"

    def test_feature_id_is_g6_1(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        assert smoke_test_plan()["feature_id"] == "G6.1"

    def test_cli_is_python_module_invocation(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        assert plan["cli"] == "python -m scribe.scripts.check_rocm"
        # The README's "Verify it took:" snippet uses the venv form.
        assert plan["cli_venv"] == ".venv/bin/python -m scribe.scripts.check_rocm"

    def test_stages_are_ordered_and_named(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        names = [s["name"] for s in plan["stages"]]
        # Order is part of the contract — run_smoke_test stops at the
        # first failure, so a researcher reading the plan should see
        # the same order they'd see in the report.
        assert names == [
            "load_whisper",
            "load_audio",
            "transcribe_silence",
            "load_align_model",
            "load_diarize",
            "run_diarize",
        ]

    def test_each_stage_has_a_summary(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        for stage in plan["stages"]:
            assert "summary" in stage and isinstance(stage["summary"], str)
            assert len(stage["summary"]) > 5

    def test_defaults_match_argparse(self) -> None:
        from scribe.scripts.check_rocm import build_parser, smoke_test_plan
        plan = smoke_test_plan()
        ns = build_parser().parse_args([])
        # Defaults the CLI uses must match what the panel advertises.
        assert plan["defaults"]["seconds"] == ns.seconds
        assert plan["defaults"]["model"] == ns.model
        assert plan["defaults"]["language"] == ns.language
        assert plan["defaults"]["include_diarize"] == ns.include_diarize

    def test_exit_codes_cover_zero_one_two(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        codes = sorted(ec["code"] for ec in plan["exit_codes"])
        # ``main()`` returns 0 / 1 / 2 — those three integers must be
        # part of the surfaced contract so a researcher writing a CI
        # wrapper can rely on them.
        assert codes == [0, 1, 2]
        for ec in plan["exit_codes"]:
            assert "meaning" in ec
            assert isinstance(ec["meaning"], str)

    def test_fail_fast_is_true(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        # ``run_smoke_test`` does in fact stop at the first failure;
        # surfacing this on the panel sets the right user expectation.
        assert plan["fail_fast"] is True

    def test_docs_anchor_targets_readme_section(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        plan = smoke_test_plan()
        # Anchor the home page can link to in /docs/readme. Markdown's
        # toc extension lower-cases + slugifies headings; "Linux (AMD GPU
        # / ROCm)" → "linux-amd-gpu--rocm".
        assert plan["docs_anchor"] == "linux-amd-gpu--rocm"


# --------------------------------------------------------------------------- #
# 2. The plan is JSON-serialisable as-is
# --------------------------------------------------------------------------- #


class TestG6_1SmokeTestPlanIsJsonSerialisable:
    """The route returns the plan dict directly via ``JSONResponse`` so
    every value must round-trip through ``json.dumps`` cleanly."""

    def test_round_trips_through_json(self) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        text = json.dumps(smoke_test_plan())
        decoded = json.loads(text)
        assert decoded == smoke_test_plan()


# --------------------------------------------------------------------------- #
# 3. ``GET /api/diagnostics/smoke-test-plan`` returns the plan
# --------------------------------------------------------------------------- #


class TestG6_1ApiSmokeTestPlanRoute:
    """Route plumbing — proves the FastAPI endpoint is registered and
    returns the helper's payload unmodified."""

    def test_route_returns_200(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/smoke-test-plan")
        assert r.status_code == 200

    def test_route_returns_json(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/smoke-test-plan")
        # FastAPI sets ``application/json`` for ``JSONResponse``.
        assert r.headers["content-type"].startswith("application/json")

    def test_route_payload_matches_helper(self, server_env) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/smoke-test-plan")
        assert r.json() == smoke_test_plan()

    def test_route_payload_is_pinned(self, server_env) -> None:
        # Surface the contract a JS / curl client gets so adding new
        # keys is non-breaking but dropping any of these breaks here.
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/smoke-test-plan")
        body = r.json()
        for key in (
            "feature_id",
            "cli",
            "cli_venv",
            "stages",
            "defaults",
            "exit_codes",
            "fail_fast",
            "docs_anchor",
        ):
            assert key in body, f"route payload missing key: {key!r}"

    def test_route_advertises_six_stages(self, server_env) -> None:
        # Drift between the route count and the home-page panel's
        # ``data-stage-count`` attribute breaks the structural pin
        # downstream.
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/smoke-test-plan")
        assert len(r.json()["stages"]) == 6


# --------------------------------------------------------------------------- #
# 4. The home page renders the smoke-test panel
# --------------------------------------------------------------------------- #


class TestG6_1HomePageRendersSmokeTestPanel:
    """Template render check — the home page must include the panel
    so a user can discover the CLI invocation without leaving the
    page."""

    def test_home_page_includes_panel(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert 'data-test-feature="G6.1"' in r.text
        assert 'data-test-id="smoke-test-plan-card"' in r.text

    def test_panel_shows_venv_cli_command(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # The README's verbatim invocation must appear inside the
        # ``smoke-test-plan-cli-venv`` element so a copy-paste from
        # the page matches the documented command exactly.
        assert ".venv/bin/python -m scribe.scripts.check_rocm" in r.text

    def test_panel_shows_module_cli_command(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # The activated-venv form is rendered in the secondary slot;
        # both forms are documented so a researcher who sourced the
        # venv manually still has a one-liner.
        assert "python -m scribe.scripts.check_rocm" in r.text

    def test_panel_lists_every_stage(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # Each stage gets its own ``data-test-id`` so a future test
        # can pin a specific stage without grepping prose.
        for name in (
            "load_whisper",
            "load_audio",
            "transcribe_silence",
            "load_align_model",
            "load_diarize",
            "run_diarize",
        ):
            assert f'data-test-id="smoke-test-plan-stage-{name}"' in r.text

    def test_panel_lists_every_exit_code(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        for code in (0, 1, 2):
            assert f'data-test-id="smoke-test-plan-exit-code-{code}"' in r.text

    def test_panel_data_stage_count_matches_helper(self, server_env) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        _srv, client, _root = server_env
        r = client.get("/")
        n = len(smoke_test_plan()["stages"])
        # The attribute is the structural pin between the panel and
        # the helper — no accidental count drift.
        assert f'data-stage-count="{n}"' in r.text

    def test_panel_links_to_in_app_readme(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # The panel must link to /docs/readme so a user can read the
        # full ROCm install context without leaving the app.
        assert 'href="/docs/readme"' in r.text


# --------------------------------------------------------------------------- #
# 5. Defensive: the home page still renders if the helper raises
# --------------------------------------------------------------------------- #


class TestG6_1HomePagePanelIsDefensive:
    """A stripped-down deploy that lacks ``scribe.scripts.check_rocm``
    must not break the upload page. The index route catches the
    import failure and skips the panel — proven here by stubbing
    the helper to raise."""

    def test_panel_is_skipped_when_helper_raises(
        self, server_env, monkeypatch
    ) -> None:
        from scribe.scripts import check_rocm

        def _boom() -> dict:
            raise RuntimeError("simulated import / eval failure")

        monkeypatch.setattr(check_rocm, "smoke_test_plan", _boom)
        _srv, client, _root = server_env
        r = client.get("/")
        # 200 — the page still renders.
        assert r.status_code == 200
        # No panel marker — the {% if smoke_test_plan %} branch is
        # skipped when the index route caught the exception.
        assert 'data-test-id="smoke-test-plan-card"' not in r.text


# --------------------------------------------------------------------------- #
# 6. End-to-end: route + template + helper agree
# --------------------------------------------------------------------------- #


class TestG6_1EndToEndAgreement:
    """Ties the three surfaces together so any drift between the
    helper, the JSON route, and the rendered HTML fails loudly."""

    def test_cli_string_matches_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        _srv, client, _root = server_env
        plan = smoke_test_plan()
        api = client.get("/api/diagnostics/smoke-test-plan").json()
        html = client.get("/").text
        assert plan["cli_venv"] == api["cli_venv"]
        assert plan["cli_venv"] in html

    def test_stage_names_match_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        _srv, client, _root = server_env
        plan = smoke_test_plan()
        api = client.get("/api/diagnostics/smoke-test-plan").json()
        html = client.get("/").text
        helper_names = [s["name"] for s in plan["stages"]]
        api_names = [s["name"] for s in api["stages"]]
        assert helper_names == api_names
        for name in helper_names:
            assert f'data-test-id="smoke-test-plan-stage-{name}"' in html

    def test_exit_codes_match_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.check_rocm import smoke_test_plan
        _srv, client, _root = server_env
        plan = smoke_test_plan()
        api = client.get("/api/diagnostics/smoke-test-plan").json()
        html = client.get("/").text
        helper_codes = sorted(ec["code"] for ec in plan["exit_codes"])
        api_codes = sorted(ec["code"] for ec in api["exit_codes"])
        assert helper_codes == api_codes == [0, 1, 2]
        for code in helper_codes:
            assert f'data-test-id="smoke-test-plan-exit-code-{code}"' in html


# --------------------------------------------------------------------------- #
# 7. Template structure pin — guards against accidental refactor
# --------------------------------------------------------------------------- #


class TestG6_1TemplatePinning:
    """Pure-source greps so renaming a marker without updating the JS
    or this test file (the two coupling points) fails here."""

    def _index_text(self) -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    def test_template_has_feature_attribute(self) -> None:
        text = self._index_text()
        assert 'data-test-feature="G6.1"' in text

    def test_template_iterates_stages(self) -> None:
        text = self._index_text()
        # Loop variable name is part of the contract because the
        # data-test-id pattern uses ``stage.name`` directly.
        assert re.search(
            r"\{% for stage in smoke_test_plan\.stages %\}", text
        ), "template must loop over smoke_test_plan.stages"

    def test_template_iterates_exit_codes(self) -> None:
        text = self._index_text()
        assert re.search(
            r"\{% for ec in smoke_test_plan\.exit_codes %\}", text
        ), "template must loop over smoke_test_plan.exit_codes"

    def test_template_renders_only_when_payload_present(self) -> None:
        text = self._index_text()
        # Defensive ``{% if smoke_test_plan %}`` guard so the panel
        # disappears cleanly on the failure path.
        assert "{% if smoke_test_plan %}" in text
