"""Tests for ``scribe.scripts.bench_whisper`` (G7.4 whisper-backend bench).

The script's runtime path loads real ML weights, so the tests exercise
the parts that don't:

* the **pure helpers** — :func:`wav_duration_seconds`, :func:`time_call`,
  :func:`normalise_text`, :func:`levenshtein`, :func:`word_error_rate`,
  :func:`hypothesis_text_from_segments`, the :class:`BackendTiming` and
  :class:`WhisperBenchmarkReport` dataclasses;
* the **driver** :func:`run_whisper_benchmark` with the
  :class:`BenchHooks` injection point so a fake ``run_backend`` stand-in
  exercises the full report-shape contract without a single MB of weights;
* the **argparse + main** layer, including ``--reference``, ``--output``,
  ``--markdown``, and the per-backend exit-code semantics.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from scribe.scripts import bench_whisper as bw


# --------------------------------------------------------------------------- #
# tiny WAV factory — same shape as the bench_rocm tests
# --------------------------------------------------------------------------- #


def _silent_wav(path: Path, *, seconds: float = 1.0, sr: int = 16000) -> Path:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return path


# --------------------------------------------------------------------------- #
# wav_duration_seconds
# --------------------------------------------------------------------------- #


class TestWavDurationSeconds:
    def test_returns_correct_duration_for_one_second(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=1.0, sr=16000)
        assert bw.wav_duration_seconds(wav) == pytest.approx(1.0, abs=1e-3)

    def test_handles_fractional_seconds(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.25, sr=16000)
        assert bw.wav_duration_seconds(wav) == pytest.approx(0.25, abs=1e-3)


# --------------------------------------------------------------------------- #
# _rtf
# --------------------------------------------------------------------------- #


class TestRtf:
    def test_returns_audio_over_wall(self) -> None:
        assert bw._rtf(10.0, 2.0) == pytest.approx(5.0)

    def test_below_one_means_slower_than_real_time(self) -> None:
        assert bw._rtf(1.0, 2.0) == pytest.approx(0.5)

    def test_zero_wall_returns_none(self) -> None:
        assert bw._rtf(10.0, 0.0) is None

    def test_zero_audio_returns_none(self) -> None:
        assert bw._rtf(0.0, 1.0) is None


# --------------------------------------------------------------------------- #
# time_call
# --------------------------------------------------------------------------- #


class TestTimeCall:
    def test_returns_result_when_callable_succeeds(self) -> None:
        secs, result, exc = bw.time_call(lambda: 42)
        assert exc is None
        assert result == 42
        assert secs >= 0.0

    def test_captures_exception_without_raising(self) -> None:
        boom = ValueError("nope")

        def raises() -> None:
            raise boom

        secs, result, exc = bw.time_call(raises)
        assert exc is boom
        assert result is None
        assert secs >= 0.0

    def test_lets_keyboard_interrupt_propagate(self) -> None:
        def raises() -> None:
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            bw.time_call(raises)


# --------------------------------------------------------------------------- #
# normalise_text
# --------------------------------------------------------------------------- #


class TestNormaliseText:
    def test_lowercases(self) -> None:
        assert bw.normalise_text("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert bw.normalise_text("Hello, world!") == ["hello", "world"]

    def test_keeps_apostrophes(self) -> None:
        # "don't" stays one token — that's the convention every WER tool
        # follows so our numbers compare cleanly.
        assert bw.normalise_text("don't stop") == ["don't", "stop"]

    def test_empty_returns_empty_list(self) -> None:
        assert bw.normalise_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert bw.normalise_text("   ") == []


# --------------------------------------------------------------------------- #
# levenshtein
# --------------------------------------------------------------------------- #


class TestLevenshtein:
    def test_zero_for_equal_lists(self) -> None:
        assert bw.levenshtein(["a", "b"], ["a", "b"]) == 0

    def test_substitution_costs_one(self) -> None:
        assert bw.levenshtein(["a"], ["b"]) == 1

    def test_insertion_costs_one(self) -> None:
        assert bw.levenshtein(["a"], ["a", "b"]) == 1

    def test_deletion_costs_one(self) -> None:
        assert bw.levenshtein(["a", "b"], ["a"]) == 1

    def test_empty_a_is_length_of_b(self) -> None:
        assert bw.levenshtein([], ["a", "b", "c"]) == 3

    def test_empty_b_is_length_of_a(self) -> None:
        assert bw.levenshtein(["a", "b"], []) == 2

    def test_both_empty_is_zero(self) -> None:
        assert bw.levenshtein([], []) == 0

    def test_classic_word_example(self) -> None:
        # "the quick brown fox" vs "the slow brown fox" = 1 sub.
        assert bw.levenshtein(
            ["the", "quick", "brown", "fox"],
            ["the", "slow", "brown", "fox"],
        ) == 1


# --------------------------------------------------------------------------- #
# word_error_rate
# --------------------------------------------------------------------------- #


class TestWordErrorRate:
    def test_zero_for_perfect_match(self) -> None:
        assert bw.word_error_rate("hello world", "hello world") == 0.0

    def test_normalises_case_and_punctuation(self) -> None:
        # WER must be 0 even when punctuation / case differ.
        assert bw.word_error_rate("Hello, world!", "hello world") == 0.0

    def test_one_substitution(self) -> None:
        # 1 sub / 2 ref words = 0.5
        assert bw.word_error_rate("hello world", "hello earth") == pytest.approx(0.5)

    def test_one_deletion(self) -> None:
        # 1 deletion / 2 ref words = 0.5
        assert bw.word_error_rate("hello world", "hello") == pytest.approx(0.5)

    def test_one_insertion(self) -> None:
        # 1 insertion / 2 ref words = 0.5
        assert bw.word_error_rate("hello world", "hello big world") == pytest.approx(0.5)

    def test_returns_none_for_empty_reference(self) -> None:
        # Undefined; refuse to publish a fake number.
        assert bw.word_error_rate("", "hi") is None

    def test_can_exceed_one(self) -> None:
        # 2 ref words, 5 hypothesis words → 5 inserts → 5/2 = 2.5
        # WER >1 is allowed; we don't clamp it.
        assert bw.word_error_rate("a b", "x y z w v") > 1.0


# --------------------------------------------------------------------------- #
# hypothesis_text_from_segments
# --------------------------------------------------------------------------- #


class TestHypothesisTextFromSegments:
    def test_glues_segment_text_with_single_spaces(self) -> None:
        segs = [{"text": " hello"}, {"text": " world "}]
        assert bw.hypothesis_text_from_segments(segs) == "hello world"

    def test_skips_empty_text(self) -> None:
        segs = [{"text": "hi"}, {"text": "   "}, {"text": "there"}]
        assert bw.hypothesis_text_from_segments(segs) == "hi there"

    def test_skips_non_dict_entries(self) -> None:
        segs = [{"text": "hi"}, "stray-string", None, {"text": "there"}]
        assert bw.hypothesis_text_from_segments(segs) == "hi there"

    def test_empty_segments_returns_empty_string(self) -> None:
        assert bw.hypothesis_text_from_segments([]) == ""


# --------------------------------------------------------------------------- #
# BackendTiming.render
# --------------------------------------------------------------------------- #


class TestBackendTimingRender:
    def test_renders_ok_with_rtf_and_wer(self) -> None:
        t = bw.BackendTiming(
            backend_id="faster-whisper",
            seconds=2.0,
            ok=True,
            wer=0.1,
        )
        line = t.render(audio_seconds=10.0)
        assert line.startswith("[OK")
        assert "faster-whisper" in line
        assert "5.00× RT" in line
        assert "WER= 10.0%" in line

    def test_renders_fail_with_detail(self) -> None:
        t = bw.BackendTiming(
            backend_id="whisper.cpp",
            seconds=0.5,
            ok=False,
            detail="ImportError: pywhispercpp",
        )
        line = t.render(audio_seconds=10.0)
        assert "[FAIL]" in line
        assert "whisper.cpp" in line
        assert "pywhispercpp" in line

    def test_renders_no_wer_when_unset(self) -> None:
        t = bw.BackendTiming(backend_id="x", seconds=1.0, ok=True, wer=None)
        line = t.render(audio_seconds=10.0)
        assert "WER=  --" in line


# --------------------------------------------------------------------------- #
# WhisperBenchmarkReport
# --------------------------------------------------------------------------- #


def _empty_report(**overrides: Any) -> bw.WhisperBenchmarkReport:
    base = dict(
        audio_path="/tmp/sample.wav",
        audio_seconds=10.0,
        model_name="tiny",
        language="en",
    )
    base.update(overrides)
    return bw.WhisperBenchmarkReport(**base)


class TestWhisperBenchmarkReport:
    def test_ok_is_false_when_no_rows(self) -> None:
        assert _empty_report().ok is False

    def test_ok_is_true_when_all_rows_pass(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="a", seconds=1.0, ok=True))
        r.add(bw.BackendTiming(backend_id="b", seconds=2.0, ok=True))
        assert r.ok is True

    def test_ok_is_false_with_any_failure(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="a", seconds=1.0, ok=True))
        r.add(bw.BackendTiming(backend_id="b", seconds=0.1, ok=False, detail="x"))
        assert r.ok is False

    def test_row_lookup_by_id(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="faster-whisper", seconds=1.0, ok=True))
        assert r.row("faster-whisper") is not None
        assert r.row("missing") is None

    def test_fastest_backend_id_picks_smallest_seconds(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=1.0, ok=True))
        assert r.fastest_backend_id == "cpp"

    def test_fastest_backend_id_ignores_failures(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=0.1, ok=False))
        assert r.fastest_backend_id == "fw"

    def test_fastest_backend_id_none_when_all_failed(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=0.1, ok=False))
        assert r.fastest_backend_id is None

    def test_speedup_over_returns_baseline_over_candidate(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=1.0, ok=True))
        # Candidate (whisper.cpp) is 4× faster than baseline (faster-whisper).
        assert r.speedup_over("fw", "cpp") == pytest.approx(4.0)

    def test_speedup_over_none_for_failed_row(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=0.1, ok=False))
        assert r.speedup_over("fw", "cpp") is None

    def test_speedup_over_none_for_missing_row(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        assert r.speedup_over("fw", "missing") is None

    def test_to_dict_round_trips_through_from_dict(self) -> None:
        r = _empty_report(hardware="M4 Max", gpu_backend="mps")
        r.add(
            bw.BackendTiming(
                backend_id="fw",
                seconds=2.0,
                ok=True,
                hypothesis="hello world",
                wer=0.0,
            )
        )
        r.add(
            bw.BackendTiming(
                backend_id="cpp",
                seconds=0.5,
                ok=False,
                detail="ImportError: pywhispercpp",
            )
        )
        r.skip("vulkan", "not available on this box")
        rehydrated = bw.WhisperBenchmarkReport.from_dict(
            json.loads(json.dumps(r.to_dict()))
        )
        assert rehydrated.hardware == "M4 Max"
        assert rehydrated.gpu_backend == "mps"
        assert len(rehydrated.rows) == 2
        assert rehydrated.rows[0].hypothesis == "hello world"
        assert rehydrated.rows[1].ok is False
        assert rehydrated.skipped == r.skipped

    def test_render_contains_header_and_rows(self) -> None:
        r = _empty_report(hardware="M4 Max", gpu_backend="mps")
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=1.0, ok=True))
        out = r.render()
        assert "Scribe whisper-backend benchmark" in out
        assert "M4 Max" in out
        assert "mps" in out
        assert "fw" in out
        assert "cpp" in out

    def test_render_handles_empty_run(self) -> None:
        out = _empty_report().render()
        assert "(no backends ran)" in out

    def test_render_lists_skipped(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True))
        r.skip("whisper.cpp", "pywhispercpp not installed")
        out = r.render()
        assert "Skipped:" in out
        assert "pywhispercpp" in out

    def test_render_truncates_long_reference_preview(self) -> None:
        long = "this is a very long reference text " * 10
        r = _empty_report(reference=long)
        r.add(bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True))
        out = r.render()
        assert "Reference:" in out
        # Truncation marker appears so the header doesn't overflow.
        assert "..." in out


# --------------------------------------------------------------------------- #
# render_markdown
# --------------------------------------------------------------------------- #


class TestRenderMarkdown:
    def test_renders_table_header(self) -> None:
        r = _empty_report(hardware="M4 Max", gpu_backend="mps")
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True, wer=0.05))
        out = r.render_markdown()
        assert "### Whisper backend benchmark — tiny on M4 Max" in out
        assert "| Backend | Wall-clock | RTF | WER |" in out
        assert "| --- | --- | --- | --- |" in out

    def test_renders_rows_with_metrics(self) -> None:
        r = _empty_report(audio_seconds=10.0, hardware="M4 Max")
        r.add(bw.BackendTiming(backend_id="fw", seconds=4.0, ok=True, wer=0.1))
        r.add(bw.BackendTiming(backend_id="cpp", seconds=1.0, ok=True, wer=0.12))
        out = r.render_markdown()
        # Wall-clock seconds in the cell:
        assert "| `fw` | 4.00s | 2.50× | 10.0% |" in out
        # RTF is the candidate divided by wall, so 1s wall = 10x RT.
        assert "| `cpp` | 1.00s | 10.00× | 12.0% |" in out

    def test_renders_failed_row_with_footnote(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=2.0, ok=True))
        r.add(
            bw.BackendTiming(
                backend_id="cpp",
                seconds=0.1,
                ok=False,
                detail="ImportError: pywhispercpp not installed",
            )
        )
        out = r.render_markdown()
        # Fail row shows FAIL in the wall-clock cell …
        assert "| `cpp` | FAIL |" in out
        # … and the detail lands in a footnote (Markdown blockquote).
        assert "> `cpp`: ImportError: pywhispercpp not installed" in out

    def test_renders_em_dash_for_missing_metrics(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=2.0, ok=True, wer=None))
        out = r.render_markdown()
        # WER cell uses an em-dash placeholder when no reference was set.
        assert "| `fw` | 2.00s | 5.00× | — |" in out

    def test_renders_skipped_block(self) -> None:
        r = _empty_report()
        r.add(bw.BackendTiming(backend_id="fw", seconds=2.0, ok=True))
        r.skip("vulkan", "not available")
        out = r.render_markdown()
        assert "> Skipped — vulkan: not available" in out


# --------------------------------------------------------------------------- #
# whisper_benchmark_plan — pure metadata
# --------------------------------------------------------------------------- #


class TestWhisperBenchmarkPlan:
    def test_returns_a_dict(self) -> None:
        assert isinstance(bw.whisper_benchmark_plan(), dict)

    def test_top_level_keys(self) -> None:
        plan = bw.whisper_benchmark_plan()
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
            assert key in plan

    def test_feature_id_is_g7_4(self) -> None:
        assert bw.whisper_benchmark_plan()["feature_id"] == "G7.4"

    def test_metric_is_wer(self) -> None:
        assert bw.whisper_benchmark_plan()["metric"] == "WER"

    def test_default_backends_match_registry_ids(self) -> None:
        from scribe.whisper_backend import (
            BACKEND_FASTER_WHISPER,
            BACKEND_WHISPER_CPP,
        )
        plan = bw.whisper_benchmark_plan()
        ids = [b["id"] for b in plan["backends"]]
        # Order is part of the contract: faster-whisper is the
        # baseline, whisper.cpp is the candidate.
        assert ids == [BACKEND_FASTER_WHISPER, BACKEND_WHISPER_CPP]

    def test_whisper_cpp_is_marked_optional(self) -> None:
        plan = bw.whisper_benchmark_plan()
        by_id = {b["id"]: b for b in plan["backends"]}
        assert by_id["whisper.cpp"]["optional"] is True
        assert by_id["faster-whisper"]["optional"] is False

    def test_exit_codes_cover_zero_one_two(self) -> None:
        codes = sorted(
            ec["code"] for ec in bw.whisper_benchmark_plan()["exit_codes"]
        )
        assert codes == [0, 1, 2]

    def test_modes_speed_accuracy_markdown(self) -> None:
        names = [m["name"] for m in bw.whisper_benchmark_plan()["modes"]]
        assert names == ["speed", "accuracy", "markdown"]

    def test_modes_advertise_both_cli_forms(self) -> None:
        for mode in bw.whisper_benchmark_plan()["modes"]:
            assert "cli" in mode and isinstance(mode["cli"], str)
            assert "cli_venv" in mode and isinstance(mode["cli_venv"], str)

    def test_accuracy_mode_mentions_reference(self) -> None:
        accuracy = next(
            m for m in bw.whisper_benchmark_plan()["modes"] if m["name"] == "accuracy"
        )
        assert "--reference" in accuracy["cli"]

    def test_markdown_mode_mentions_markdown_flag(self) -> None:
        md = next(
            m for m in bw.whisper_benchmark_plan()["modes"] if m["name"] == "markdown"
        )
        assert "--markdown" in md["cli"]

    def test_fail_isolated_is_true(self) -> None:
        assert bw.whisper_benchmark_plan()["fail_isolated"] is True

    def test_cli_strings_match_module_invocation(self) -> None:
        plan = bw.whisper_benchmark_plan()
        assert plan["cli"] == "python -m scribe.scripts.bench_whisper"
        assert plan["cli_venv"] == ".venv/bin/python -m scribe.scripts.bench_whisper"

    def test_defaults_match_argparse(self) -> None:
        plan = bw.whisper_benchmark_plan()
        ns = bw.build_parser().parse_args(["fake.wav"])
        assert plan["defaults"]["model"] == ns.model
        assert plan["defaults"]["language"] == ns.language
        assert plan["defaults"]["quant"] == ns.quant
        # backend default in argparse is None ⇒ DEFAULT_BACKENDS at runtime;
        # the plan repeats the runtime default explicitly so the panel
        # shows the user what will run.
        assert plan["defaults"]["backends"] == list(bw.DEFAULT_BACKENDS)
        assert plan["defaults"]["reference"] == ns.reference
        assert plan["defaults"]["output"] == ns.output
        assert plan["defaults"]["markdown"] == ns.markdown
        assert plan["defaults"]["label"] == ns.label

    def test_round_trips_through_json(self) -> None:
        text = json.dumps(bw.whisper_benchmark_plan())
        decoded = json.loads(text)
        assert decoded == bw.whisper_benchmark_plan()


# --------------------------------------------------------------------------- #
# FakeHooks for run_whisper_benchmark
# --------------------------------------------------------------------------- #


class FakeRunBackend:
    """Recording stand-in for ``BenchHooks.run_backend``.

    Each call appends to ``self.calls`` so tests can pin which backends
    were exercised and in what order. ``responses`` maps backend_id →
    either a ``segments``-shaped dict (success) or an ``Exception`` to
    raise (failure)."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self,
        backend_id: str,
        audio_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((backend_id, dict(kwargs, audio_path=audio_path)))
        resp = self.responses.get(backend_id)
        if isinstance(resp, Exception):
            raise resp
        if resp is None:
            return {"segments": []}
        return resp


# --------------------------------------------------------------------------- #
# run_whisper_benchmark
# --------------------------------------------------------------------------- #


class TestRunWhisperBenchmark:
    def test_runs_every_default_backend(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=1.0)
        fake = FakeRunBackend({
            "faster-whisper": {"segments": [{"text": "hello world"}]},
            "whisper.cpp": {"segments": [{"text": "hello world"}]},
        })
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="test-host",
            gpu_backend_label="cpu",
        )
        ids = [r.backend_id for r in report.rows]
        assert ids == ["faster-whisper", "whisper.cpp"]
        assert report.ok is True

    def test_passes_model_and_language_through(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        fake = FakeRunBackend({})
        bw.run_whisper_benchmark(
            audio_path=wav,
            model_name="large-v3-turbo",
            language="fr",
            backends=("faster-whisper",),
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="cpu",
        )
        kwargs = fake.calls[0][1]
        assert kwargs["model_name"] == "large-v3-turbo"
        assert kwargs["language"] == "fr"

    def test_passes_quant_only_through_asr_options(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        fake = FakeRunBackend({})
        bw.run_whisper_benchmark(
            audio_path=wav,
            backends=("whisper.cpp",),
            quant="q8_0",
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="mps",
        )
        kwargs = fake.calls[0][1]
        assert kwargs["asr_options"] == {"whisper_cpp_quant": "q8_0"}

    def test_failed_backend_does_not_stop_other_backends(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.5)
        fake = FakeRunBackend({
            "faster-whisper": {"segments": [{"text": "hi"}]},
            "whisper.cpp": ImportError("pywhispercpp not installed"),
        })
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="cpu",
        )
        # Both backends ran, one failed:
        assert [r.backend_id for r in report.rows] == [
            "faster-whisper",
            "whisper.cpp",
        ]
        assert report.row("faster-whisper").ok is True
        assert report.row("whisper.cpp").ok is False
        assert "ImportError" in report.row("whisper.cpp").detail
        # Overall report is not "ok" because one row failed; the user
        # must opt in to publishing partial results.
        assert report.ok is False

    def test_computes_wer_when_reference_provided(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.5)
        fake = FakeRunBackend({
            "faster-whisper": {"segments": [{"text": "hello earth"}]},
        })
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            backends=("faster-whisper",),
            reference="hello world",
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="cpu",
        )
        row = report.row("faster-whisper")
        assert row is not None
        # 1 sub / 2 words = 0.5
        assert row.wer == pytest.approx(0.5)
        assert row.hypothesis == "hello earth"

    def test_skips_wer_when_no_reference(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.5)
        fake = FakeRunBackend({
            "faster-whisper": {"segments": [{"text": "hi"}]},
        })
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            backends=("faster-whisper",),
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="cpu",
        )
        assert report.row("faster-whisper").wer is None

    def test_records_hardware_and_gpu_backend_label(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.5)
        fake = FakeRunBackend({
            "faster-whisper": {"segments": []},
        })
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            backends=("faster-whisper",),
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="M4 Max 24GB",
            gpu_backend_label="mps",
        )
        assert report.hardware == "M4 Max 24GB"
        assert report.gpu_backend == "mps"

    def test_audio_seconds_override_is_used_when_provided(
        self, tmp_path: Path
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        fake = FakeRunBackend({"faster-whisper": {"segments": []}})
        report = bw.run_whisper_benchmark(
            audio_path=wav,
            backends=("faster-whisper",),
            audio_seconds=42.0,
            hooks=bw.BenchHooks(run_backend=fake),
            hardware="x",
            gpu_backend_label="cpu",
        )
        assert report.audio_seconds == 42.0

    def test_missing_audio_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            bw.run_whisper_benchmark(audio_path=tmp_path / "no.wav")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_defaults(self) -> None:
        ns = bw.build_parser().parse_args(["sample.wav"])
        assert ns.audio == Path("sample.wav")
        assert ns.model == "tiny"
        assert ns.language == "en"
        assert ns.quant is None
        assert ns.backend is None
        assert ns.reference is None
        assert ns.output is None
        assert ns.markdown is None
        assert ns.label is None

    def test_repeating_backend(self) -> None:
        ns = bw.build_parser().parse_args(
            ["sample.wav", "--backend", "faster-whisper", "--backend", "whisper.cpp"]
        )
        assert ns.backend == ["faster-whisper", "whisper.cpp"]

    def test_reference_and_output(self) -> None:
        ns = bw.build_parser().parse_args(
            ["sample.wav", "--reference", "ref.txt", "--output", "out.json"]
        )
        assert ns.reference == Path("ref.txt")
        assert ns.output == Path("out.json")

    def test_markdown_and_label(self) -> None:
        ns = bw.build_parser().parse_args(
            ["sample.wav", "--markdown", "out.md", "--label", "M4 Max"]
        )
        assert ns.markdown == Path("out.md")
        assert ns.label == "M4 Max"


def _make_fake_run_for_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[bw.BackendTiming],
    skipped: list[tuple[str, str]] | None = None,
    record: dict[str, Any] | None = None,
) -> None:
    def fake_run(**kw: Any) -> bw.WhisperBenchmarkReport:
        if record is not None:
            record.update(kw)
        r = bw.WhisperBenchmarkReport(
            audio_path=str(kw["audio_path"]),
            audio_seconds=float(kw.get("audio_seconds") or 0.0),
            model_name=kw.get("model_name", "tiny"),
            language=kw.get("language", "en"),
            hardware=kw.get("hardware") or "test-host",
            gpu_backend=kw.get("gpu_backend_label") or "cpu",
            reference=kw.get("reference") or "",
        )
        for row in rows:
            r.add(row)
        for backend_id, reason in (skipped or []):
            r.skip(backend_id, reason)
        return r

    monkeypatch.setattr(bw, "run_whisper_benchmark", fake_run)


class TestMain:
    def test_returns_two_when_no_audio(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        rc = bw.main([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "audio path is required" in err

    def test_returns_two_when_audio_missing(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        rc = bw.main([str(tmp_path / "no.wav")])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_returns_two_when_reference_missing(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        rc = bw.main(
            [str(wav), "--reference", str(tmp_path / "no-ref.txt")]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "reference file not found" in err

    def test_returns_zero_on_successful_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        _make_fake_run_for_main(
            monkeypatch,
            rows=[
                bw.BackendTiming(backend_id="faster-whisper", seconds=1.0, ok=True),
                bw.BackendTiming(backend_id="whisper.cpp", seconds=0.5, ok=True),
            ],
        )
        rc = bw.main([str(wav)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scribe whisper-backend benchmark" in out

    def test_returns_one_when_a_backend_failed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        _make_fake_run_for_main(
            monkeypatch,
            rows=[
                bw.BackendTiming(backend_id="faster-whisper", seconds=1.0, ok=True),
                bw.BackendTiming(
                    backend_id="whisper.cpp",
                    seconds=0.1,
                    ok=False,
                    detail="ImportError: pywhispercpp",
                ),
            ],
        )
        rc = bw.main([str(wav)])
        assert rc == 1

    def test_writes_json_when_output_specified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        out_path = tmp_path / "report.json"
        _make_fake_run_for_main(
            monkeypatch,
            rows=[bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True)],
        )
        rc = bw.main([str(wav), "--output", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["rows"][0]["backend_id"] == "fw"

    def test_writes_markdown_when_markdown_specified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        md_path = tmp_path / "bench.md"
        _make_fake_run_for_main(
            monkeypatch,
            rows=[
                bw.BackendTiming(
                    backend_id="fw", seconds=2.0, ok=True, wer=0.05
                ),
                bw.BackendTiming(
                    backend_id="cpp", seconds=0.5, ok=True, wer=0.07
                ),
            ],
        )
        rc = bw.main([str(wav), "--markdown", str(md_path)])
        assert rc == 0
        text = md_path.read_text()
        assert "| Backend | Wall-clock | RTF | WER |" in text
        assert "| `fw` |" in text
        assert "| `cpp` |" in text

    def test_forwards_label_override_as_hardware(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        record: dict[str, Any] = {}
        _make_fake_run_for_main(
            monkeypatch,
            rows=[bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True)],
            record=record,
        )
        bw.main([str(wav), "--label", "M4 Max"])
        assert record["hardware"] == "M4 Max"

    def test_forwards_reference_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        ref_path = tmp_path / "ref.txt"
        ref_path.write_text("hello world\n")
        record: dict[str, Any] = {}
        _make_fake_run_for_main(
            monkeypatch,
            rows=[bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True)],
            record=record,
        )
        bw.main([str(wav), "--reference", str(ref_path)])
        assert "hello world" in record["reference"]

    def test_forwards_repeated_backend_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.2)
        record: dict[str, Any] = {}
        _make_fake_run_for_main(
            monkeypatch,
            rows=[bw.BackendTiming(backend_id="fw", seconds=1.0, ok=True)],
            record=record,
        )
        bw.main([str(wav), "--backend", "faster-whisper"])
        assert record["backends"] == ("faster-whisper",)
