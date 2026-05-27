"""Verification of reachability for G7.4 — whisper-backend benchmark plan.

The G7.4 script (``scribe.scripts.bench_whisper``) is the user-facing
CLI for measuring the faster-whisper / whisper.cpp speedup on a single
machine, but a researcher who hasn't yet memorised the invocation
should be able to discover it from the home page. This file pins:

  * ``GET /api/diagnostics/whisper-benchmark-plan`` returns the plan
    (run/accuracy/markdown CLI invocations, ordered backend list,
    defaults, exit codes, fail-isolated policy, headline metric)
    without importing torch / whisperx and without spawning the
    benchmark itself.

  * ``GET /`` renders a ``data-test-feature="G7.4"`` panel with the
    same data so a researcher who hasn't memorised the CLI can copy
    the invocation straight from the page.

Coverage matrix mirrors ``test_server_g6_2_helpers.py``:

  1. The pure-helper ``whisper_benchmark_plan()`` returns the
     expected contract (keys, ordered backends, defaults, exit codes,
     modes, headline metric).

  2. The route returns 200 and the same payload with the documented
     JSON shape.

  3. The home page renders the panel with the CLI commands (three
     modes), the backend list, the exit-code list, and the optional
     flag for whisper.cpp.

  4. The home page collapses cleanly when the helper raises
     (defensive — strip-down deploys without ``scribe.scripts``).

  5. End-to-end: route + template + helper agree on the CLI
     invocation strings, the backend order, and the exit codes.
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


class TestG7_4WhisperBenchmarkPlanContract:
    """The pure helper is consumed by both the route and the template,
    so its shape is part of the public contract."""

    def test_returns_a_dict(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        assert isinstance(whisper_benchmark_plan(), dict)

    def test_top_level_keys_are_pinned(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        plan = whisper_benchmark_plan()
        for key in (
            "feature_id",
            "cli",
            "cli_venv",
            "backends",
            "defaults",
            "exit_codes",
            "modes",
            "fail_isolated",
            "metric",
            "docs_anchor",
        ):
            assert key in plan, f"whisper_benchmark_plan() missing key: {key!r}"

    def test_feature_id_is_g7_4(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        assert whisper_benchmark_plan()["feature_id"] == "G7.4"

    def test_cli_is_python_module_invocation(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        plan = whisper_benchmark_plan()
        assert plan["cli"] == "python -m scribe.scripts.bench_whisper"
        assert plan["cli_venv"] == ".venv/bin/python -m scribe.scripts.bench_whisper"

    def test_backends_are_ordered_baseline_first(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        plan = whisper_benchmark_plan()
        ids = [b["id"] for b in plan["backends"]]
        # Baseline first, candidate second — any future renderer must
        # respect this ordering for the speedup column to make sense.
        assert ids == ["faster-whisper", "whisper.cpp"]

    def test_each_backend_has_summary_and_optional_flag(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        for backend in whisper_benchmark_plan()["backends"]:
            assert "summary" in backend and isinstance(backend["summary"], str)
            assert len(backend["summary"]) > 5
            assert "optional" in backend and isinstance(backend["optional"], bool)

    def test_optional_flag_is_correct(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        by_id = {b["id"]: b for b in whisper_benchmark_plan()["backends"]}
        # faster-whisper is always required; whisper.cpp may not be
        # installed on every box.
        assert by_id["faster-whisper"]["optional"] is False
        assert by_id["whisper.cpp"]["optional"] is True

    def test_modes_are_speed_accuracy_markdown(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        names = [m["name"] for m in whisper_benchmark_plan()["modes"]]
        # Order is part of the contract: cheapest first, then accuracy,
        # then "ship the README table".
        assert names == ["speed", "accuracy", "markdown"]

    def test_each_mode_advertises_both_cli_forms(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        for mode in whisper_benchmark_plan()["modes"]:
            assert "summary" in mode and isinstance(mode["summary"], str)
            assert "cli" in mode and isinstance(mode["cli"], str)
            assert "cli_venv" in mode and isinstance(mode["cli_venv"], str)

    def test_accuracy_and_markdown_modes_mention_their_flags(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        modes = {m["name"]: m for m in whisper_benchmark_plan()["modes"]}
        assert "--reference" in modes["accuracy"]["cli"]
        assert "--reference" in modes["accuracy"]["cli_venv"]
        assert "--markdown" in modes["markdown"]["cli"]
        assert "--markdown" in modes["markdown"]["cli_venv"]

    def test_defaults_match_argparse(self) -> None:
        from scribe.scripts.bench_whisper import (
            build_parser,
            whisper_benchmark_plan,
            DEFAULT_BACKENDS,
        )
        plan = whisper_benchmark_plan()
        ns = build_parser().parse_args(["fake.wav"])
        assert plan["defaults"]["model"] == ns.model
        assert plan["defaults"]["language"] == ns.language
        assert plan["defaults"]["quant"] == ns.quant
        assert plan["defaults"]["reference"] == ns.reference
        assert plan["defaults"]["output"] == ns.output
        assert plan["defaults"]["markdown"] == ns.markdown
        assert plan["defaults"]["label"] == ns.label
        # ``backends`` default is None in argparse → DEFAULT_BACKENDS at
        # runtime; the plan repeats that explicitly for the panel.
        assert plan["defaults"]["backends"] == list(DEFAULT_BACKENDS)

    def test_exit_codes_cover_zero_one_two(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        codes = sorted(ec["code"] for ec in whisper_benchmark_plan()["exit_codes"])
        assert codes == [0, 1, 2]
        for ec in whisper_benchmark_plan()["exit_codes"]:
            assert "meaning" in ec and isinstance(ec["meaning"], str)

    def test_fail_isolated_is_true(self) -> None:
        # The driver continues after a backend failure — this is part
        # of the "two backends side-by-side" contract.
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        assert whisper_benchmark_plan()["fail_isolated"] is True

    def test_metric_is_wer(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        # The headline column in the Markdown table is WER. A scripted
        # consumer (CI bot, support script) should be able to learn
        # this from the plan.
        assert whisper_benchmark_plan()["metric"] == "WER"

    def test_docs_anchor_targets_apple_silicon_section(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        # Distinct anchor from the G6.x AMD/ROCm pair — Apple Silicon
        # is the audience this benchmark is for.
        assert whisper_benchmark_plan()["docs_anchor"] == "apple-silicon-gpu-whisper-cpp"


# --------------------------------------------------------------------------- #
# 2. The plan is JSON-serialisable as-is
# --------------------------------------------------------------------------- #


class TestG7_4WhisperBenchmarkPlanIsJsonSerialisable:
    def test_round_trips_through_json(self) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        text = json.dumps(whisper_benchmark_plan())
        decoded = json.loads(text)
        assert decoded == whisper_benchmark_plan()


# --------------------------------------------------------------------------- #
# 3. ``GET /api/diagnostics/whisper-benchmark-plan`` returns the plan
# --------------------------------------------------------------------------- #


class TestG7_4ApiWhisperBenchmarkPlanRoute:
    """Route plumbing — proves the FastAPI endpoint is registered and
    returns the helper's payload unmodified."""

    def test_route_returns_200(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        assert r.status_code == 200

    def test_route_returns_json(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        assert r.headers["content-type"].startswith("application/json")

    def test_route_payload_matches_helper(self, server_env) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        assert r.json() == whisper_benchmark_plan()

    def test_route_payload_is_pinned(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        body = r.json()
        for key in (
            "feature_id",
            "cli",
            "cli_venv",
            "backends",
            "defaults",
            "exit_codes",
            "modes",
            "fail_isolated",
            "metric",
            "docs_anchor",
        ):
            assert key in body

    def test_route_advertises_two_backends(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        assert len(r.json()["backends"]) == 2

    def test_route_advertises_three_modes(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/api/diagnostics/whisper-benchmark-plan")
        modes = r.json()["modes"]
        assert len(modes) == 3
        assert sorted(m["name"] for m in modes) == [
            "accuracy",
            "markdown",
            "speed",
        ]


# --------------------------------------------------------------------------- #
# 4. The home page renders the whisper-benchmark panel
# --------------------------------------------------------------------------- #


class TestG7_4HomePageRendersWhisperBenchmarkPanel:
    """Template render check — the home page must include the panel
    so a user can discover the CLI invocation without leaving the
    page."""

    def test_home_page_includes_panel(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert r.status_code == 200
        assert 'data-test-feature="G7.4"' in r.text
        assert 'data-test-id="whisper-benchmark-plan-card"' in r.text

    def test_panel_shows_speed_mode_cli_command(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert 'data-test-id="whisper-benchmark-plan-mode-speed-cli-venv"' in r.text
        # Speed-mode CLI is the cheapest run; pin the venv form.
        assert ".venv/bin/python -m scribe.scripts.bench_whisper" in r.text

    def test_panel_shows_accuracy_mode_cli_command(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert 'data-test-id="whisper-benchmark-plan-mode-accuracy-cli-venv"' in r.text
        assert "--reference" in r.text

    def test_panel_shows_markdown_mode_cli_command(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert 'data-test-id="whisper-benchmark-plan-mode-markdown-cli-venv"' in r.text
        assert "--markdown" in r.text

    def test_panel_lists_every_backend(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        for bid in ("faster-whisper", "whisper.cpp"):
            assert f'data-test-id="whisper-benchmark-plan-backend-{bid}"' in r.text

    def test_panel_lists_every_exit_code(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        for code in (0, 1, 2):
            assert (
                f'data-test-id="whisper-benchmark-plan-exit-code-{code}"' in r.text
            )

    def test_panel_lists_every_mode(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        for mode in ("speed", "accuracy", "markdown"):
            assert f'data-test-id="whisper-benchmark-plan-mode-{mode}"' in r.text

    def test_panel_data_backend_count_matches_helper(self, server_env) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        r = client.get("/")
        n = len(whisper_benchmark_plan()["backends"])
        assert f'data-backend-count="{n}"' in r.text

    def test_panel_data_mode_count_matches_helper(self, server_env) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        r = client.get("/")
        n = len(whisper_benchmark_plan()["modes"])
        assert f'data-mode-count="{n}"' in r.text

    def test_panel_advertises_metric(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # The metric label is part of the contract — a user reading
        # the panel learns the headline column without leaving the
        # page.
        assert "WER" in r.text

    def test_panel_links_to_in_app_readme(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        assert 'href="/docs/readme"' in r.text

    def test_panel_marks_whisper_cpp_as_optional(self, server_env) -> None:
        _srv, client, _root = server_env
        r = client.get("/")
        # Inside the whisper.cpp <li> we render "(optional)". Use a
        # span class hint as anchor so the test matches even if other
        # panels have unrelated optional rows.
        match = re.search(
            r'data-test-id="whisper-benchmark-plan-backend-whisper\.cpp">'
            r'.*?\(optional\)',
            r.text,
            re.DOTALL,
        )
        assert match is not None


# --------------------------------------------------------------------------- #
# 5. Defensive: the home page still renders if the helper raises
# --------------------------------------------------------------------------- #


class TestG7_4HomePagePanelIsDefensive:
    """A stripped-down deploy that lacks ``scribe.scripts.bench_whisper``
    must not break the upload page. The index route catches the
    import failure and skips the panel."""

    def test_panel_is_skipped_when_helper_raises(
        self, server_env, monkeypatch
    ) -> None:
        from scribe.scripts import bench_whisper

        def _boom() -> dict:
            raise RuntimeError("simulated import / eval failure")

        monkeypatch.setattr(bench_whisper, "whisper_benchmark_plan", _boom)
        _srv, client, _root = server_env
        r = client.get("/")
        # 200 — the page still renders.
        assert r.status_code == 200
        # No panel marker — the {% if whisper_benchmark_plan %} branch
        # is skipped when the index route caught the exception.
        assert 'data-test-id="whisper-benchmark-plan-card"' not in r.text


# --------------------------------------------------------------------------- #
# 6. End-to-end: route + template + helper agree
# --------------------------------------------------------------------------- #


class TestG7_4EndToEndAgreement:
    def test_cli_string_matches_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        plan = whisper_benchmark_plan()
        api = client.get("/api/diagnostics/whisper-benchmark-plan").json()
        html = client.get("/").text
        assert plan["cli_venv"] == api["cli_venv"]
        speed = next(m for m in plan["modes"] if m["name"] == "speed")
        assert ".venv/bin/python -m scribe.scripts.bench_whisper" in speed["cli_venv"]
        assert ".venv/bin/python -m scribe.scripts.bench_whisper" in html
        # ``<audio>`` placeholder is autoescaped:
        assert "&lt;audio&gt;" in html

    def test_backend_ids_match_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        plan = whisper_benchmark_plan()
        api = client.get("/api/diagnostics/whisper-benchmark-plan").json()
        html = client.get("/").text
        helper_ids = [b["id"] for b in plan["backends"]]
        api_ids = [b["id"] for b in api["backends"]]
        assert helper_ids == api_ids
        for bid in helper_ids:
            assert f'data-test-id="whisper-benchmark-plan-backend-{bid}"' in html

    def test_exit_codes_match_across_helper_route_and_html(
        self, server_env
    ) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        plan = whisper_benchmark_plan()
        api = client.get("/api/diagnostics/whisper-benchmark-plan").json()
        html = client.get("/").text
        helper_codes = sorted(ec["code"] for ec in plan["exit_codes"])
        api_codes = sorted(ec["code"] for ec in api["exit_codes"])
        assert helper_codes == api_codes == [0, 1, 2]
        for code in helper_codes:
            assert (
                f'data-test-id="whisper-benchmark-plan-exit-code-{code}"' in html
            )

    def test_modes_match_across_helper_route_and_html(self, server_env) -> None:
        from scribe.scripts.bench_whisper import whisper_benchmark_plan
        _srv, client, _root = server_env
        plan = whisper_benchmark_plan()
        api = client.get("/api/diagnostics/whisper-benchmark-plan").json()
        html = client.get("/").text
        helper_modes = [m["name"] for m in plan["modes"]]
        api_modes = [m["name"] for m in api["modes"]]
        assert helper_modes == api_modes == ["speed", "accuracy", "markdown"]
        for name in helper_modes:
            assert f'data-test-id="whisper-benchmark-plan-mode-{name}"' in html


# --------------------------------------------------------------------------- #
# 7. Template structure pin — guards against accidental refactor
# --------------------------------------------------------------------------- #


class TestG7_4TemplatePinning:
    """Pure-source greps so renaming a marker without updating this
    test fails here."""

    def _index_text(self) -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    def test_template_has_feature_attribute(self) -> None:
        text = self._index_text()
        assert 'data-test-feature="G7.4"' in text

    def test_template_iterates_backends(self) -> None:
        text = self._index_text()
        # Loop variable name is part of the contract — the
        # data-test-id pattern uses ``backend.id`` directly.
        assert re.search(
            r"\{% for backend in whisper_benchmark_plan\.backends %\}", text
        )

    def test_template_iterates_exit_codes(self) -> None:
        text = self._index_text()
        assert re.search(
            r"\{% for ec in whisper_benchmark_plan\.exit_codes %\}", text
        )

    def test_template_iterates_modes(self) -> None:
        text = self._index_text()
        assert re.search(
            r"\{% for mode in whisper_benchmark_plan\.modes %\}", text
        )

    def test_template_renders_only_when_payload_present(self) -> None:
        text = self._index_text()
        assert "{% if whisper_benchmark_plan %}" in text
