"""Tests for ``scribe.scripts.bench_rocm`` (G6.2 in-house benchmark).

The script's runtime path loads real ML weights, so the tests exercise
the parts that don't:

* the **pure helpers** — :func:`wav_duration_seconds`, :func:`time_call`,
  :func:`format_speedup`, the :class:`Timing`, :class:`BenchmarkReport`,
  and :class:`BenchmarkComparison` dataclasses, plus
  :func:`compare_reports`;
* the **driver** :func:`run_benchmark` with the :class:`BenchHooks`
  injection point that lets a test stand in fake loaders for whisperx /
  pyannote without a single MB of weights;
* the **argparse + main** layer, including ``--compare`` round-tripping
  saved JSON reports.

The real model loaders themselves are not exercised — that's what the
``slow`` / ``gpu`` markers are for, and G6.2 is meant to be runnable on
any developer machine.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from scribe.scripts import bench_rocm as br


# --------------------------------------------------------------------------- #
# tiny WAV factory shared with check_rocm tests; copied locally so the two
# test modules don't grow a coupling.
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
        assert br.wav_duration_seconds(wav) == pytest.approx(1.0, abs=1e-3)

    def test_handles_fractional_seconds(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.25, sr=16000)
        assert br.wav_duration_seconds(wav) == pytest.approx(0.25, abs=1e-3)

    def test_uses_actual_sample_rate(self, tmp_path: Path) -> None:
        wav = _silent_wav(tmp_path / "a.wav", seconds=0.5, sr=8000)
        # 4000 frames / 8000 sr == 0.5s; the function must not assume 16k.
        assert br.wav_duration_seconds(wav) == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# _rtf
# --------------------------------------------------------------------------- #


class TestRtf:
    def test_returns_audio_over_wall(self) -> None:
        assert br._rtf(10.0, 2.0) == pytest.approx(5.0)

    def test_below_one_means_slower_than_real_time(self) -> None:
        assert br._rtf(1.0, 2.0) == pytest.approx(0.5)

    def test_zero_wall_returns_none(self) -> None:
        assert br._rtf(10.0, 0.0) is None

    def test_negative_wall_returns_none(self) -> None:
        assert br._rtf(10.0, -1.0) is None

    def test_zero_audio_returns_none(self) -> None:
        # A zero-length audio file is meaningless to report RTF on.
        assert br._rtf(0.0, 1.0) is None


# --------------------------------------------------------------------------- #
# format_speedup
# --------------------------------------------------------------------------- #


class TestFormatSpeedup:
    def test_none_returns_n_a(self) -> None:
        assert br.format_speedup(None) == "n/a"

    def test_zero_returns_n_a(self) -> None:
        # 0 means we can't divide — the report can't make a claim.
        assert br.format_speedup(0.0) == "n/a"

    def test_factor_above_one_renders_faster(self) -> None:
        assert br.format_speedup(2.0) == "2.00× faster"

    def test_factor_one_renders_faster_at_parity(self) -> None:
        # Exact parity is rendered as "1.00× faster" rather than a special
        # case: it's honest, and the comparison column shows both numbers.
        assert br.format_speedup(1.0) == "1.00× faster"

    def test_factor_below_one_inverts_to_slower(self) -> None:
        # 0.5x faster reads wrong; we want "2.00x slower".
        assert br.format_speedup(0.5) == "2.00× slower"

    def test_negative_returns_n_a(self) -> None:
        assert br.format_speedup(-0.4) == "n/a"


# --------------------------------------------------------------------------- #
# time_call
# --------------------------------------------------------------------------- #


class TestTimeCall:
    def test_returns_result_when_callable_succeeds(self) -> None:
        secs, result, exc = br.time_call(lambda: 99)
        assert exc is None
        assert result == 99
        assert secs >= 0.0

    def test_captures_exception_without_raising(self) -> None:
        boom = ValueError("nope")

        def raises() -> None:
            raise boom

        secs, result, exc = br.time_call(raises)
        assert exc is boom
        assert result is None
        assert secs >= 0.0

    def test_lets_keyboard_interrupt_propagate(self) -> None:
        def raises() -> None:
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            br.time_call(raises)


# --------------------------------------------------------------------------- #
# Timing.render
# --------------------------------------------------------------------------- #


class TestTimingRender:
    def test_renders_ok_with_rtf(self) -> None:
        t = br.Timing("transcribe", 2.0, ok=True, detail="model=tiny")
        line = t.render(audio_seconds=10.0)
        assert line.startswith("[OK")
        assert "transcribe" in line
        assert "5.00× RT" in line  # 10s of audio in 2s wall = 5x
        assert "model=tiny" in line

    def test_renders_fail_with_detail(self) -> None:
        t = br.Timing("align", 1.5, ok=False, detail="RuntimeError: HSA")
        line = t.render(audio_seconds=10.0)
        assert "[FAIL]" in line
        assert "align" in line
        assert "HSA" in line

    def test_renders_no_rtf_when_wall_is_zero(self) -> None:
        # Defensive: a zero-time stage shouldn't print "infx RT".
        t = br.Timing("load_audio", 0.0)
        line = t.render(audio_seconds=10.0)
        assert "--" in line


# --------------------------------------------------------------------------- #
# BenchmarkReport
# --------------------------------------------------------------------------- #


def _empty_report(**overrides: Any) -> br.BenchmarkReport:
    base = dict(
        backend="cpu",
        whisper_device="cpu",
        whisper_compute="int8",
        torch_device="cpu",
        model_name="tiny",
        language="en",
        audio_path="/tmp/sample.wav",
        audio_seconds=10.0,
    )
    base.update(overrides)
    return br.BenchmarkReport(**base)


class TestBenchmarkReport:
    def test_ok_is_false_when_no_stages_ran(self) -> None:
        assert _empty_report().ok is False

    def test_ok_is_true_when_all_stages_pass(self) -> None:
        r = _empty_report()
        r.add(br.Timing("a", 1.0))
        r.add(br.Timing("b", 2.0))
        assert r.ok is True

    def test_ok_is_false_with_any_failure(self) -> None:
        r = _empty_report()
        r.add(br.Timing("a", 1.0))
        r.add(br.Timing("b", 2.0, ok=False, detail="boom"))
        assert r.ok is False

    def test_total_seconds_sums_stages(self) -> None:
        r = _empty_report()
        r.add(br.Timing("a", 0.5))
        r.add(br.Timing("b", 1.5))
        assert r.total_seconds == pytest.approx(2.0)

    def test_overall_rtf_uses_total(self) -> None:
        r = _empty_report(audio_seconds=10.0)
        r.add(br.Timing("a", 1.0))
        r.add(br.Timing("b", 1.0))
        assert r.overall_rtf == pytest.approx(5.0)

    def test_overall_rtf_none_when_no_stages(self) -> None:
        # No stages → total_seconds is 0; RTF is undefined.
        assert _empty_report().overall_rtf is None

    def test_stage_lookup_by_name(self) -> None:
        r = _empty_report()
        r.add(br.Timing("transcribe", 1.0))
        assert r.stage("transcribe") is not None
        assert r.stage("missing") is None

    def test_to_dict_round_trips_through_from_dict(self) -> None:
        r = _empty_report(hardware="RX 7900 XTX (ROCm/HIP 6.3)")
        r.add(br.Timing("a", 1.0, detail="model=tiny"))
        r.add(br.Timing("b", 2.0, ok=False, detail="boom"))
        r.skip("load_diarize", "HF_TOKEN not set")
        d = r.to_dict()
        # Round-trip via JSON to be sure no non-serialisable fields snuck in.
        rehydrated = br.BenchmarkReport.from_dict(json.loads(json.dumps(d)))
        assert rehydrated.backend == r.backend
        assert rehydrated.audio_seconds == r.audio_seconds
        assert rehydrated.hardware == r.hardware
        assert len(rehydrated.timings) == 2
        assert rehydrated.timings[0].detail == "model=tiny"
        assert rehydrated.timings[1].ok is False
        assert rehydrated.skipped == r.skipped

    def test_render_contains_header_and_lines(self) -> None:
        r = _empty_report(
            backend="rocm",
            whisper_device="rocm",
            whisper_compute="float16",
            torch_device="rocm",
            audio_seconds=4.0,
            hardware="RX 7900 XTX",
        )
        r.add(br.Timing("transcribe", 1.0))
        r.add(br.Timing("align", 1.0))
        out = r.render()
        assert "Backend:" in out
        assert "rocm" in out
        assert "compute=float16" in out
        assert "RX 7900 XTX" in out
        assert "transcribe" in out
        assert "align" in out
        assert "total" in out
        # 4s audio in 2s wall = 2x RT shown on the total row.
        assert "2.00× RT" in out

    def test_render_handles_empty_run(self) -> None:
        out = _empty_report().render()
        assert "(no stages ran)" in out

    def test_render_lists_skipped(self) -> None:
        r = _empty_report()
        r.add(br.Timing("a", 0.1))
        r.skip("load_diarize", "HF_TOKEN not set")
        out = r.render()
        assert "Skipped:" in out
        assert "HF_TOKEN" in out


# --------------------------------------------------------------------------- #
# compare_reports / BenchmarkComparison
# --------------------------------------------------------------------------- #


def _filled_report(*, backend: str, stages: list[tuple[str, float]]) -> br.BenchmarkReport:
    r = br.BenchmarkReport(
        backend=backend,
        whisper_device=backend,
        whisper_compute="float16",
        torch_device=backend,
        model_name="tiny",
        language="en",
        audio_path="/tmp/sample.wav",
        audio_seconds=10.0,
    )
    for name, secs in stages:
        r.add(br.Timing(name, secs))
    return r


class TestCompareReports:
    def test_pairs_stages_by_name(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0), ("align", 1.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0), ("align", 2.0)])
        cmp = br.compare_reports(a, b)
        names = [r.name for r in cmp.stages]
        assert names == ["transcribe", "align"]
        # cuda baseline 2s vs rocm 4s ⇒ candidate 0.5x ⇒ slower.
        transcribe = cmp.stages[0]
        assert transcribe.baseline_seconds == 2.0
        assert transcribe.candidate_seconds == 4.0
        assert transcribe.speedup == pytest.approx(0.5)

    def test_total_sums_only_common_stages(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0), ("align", 1.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0), ("align", 2.0)])
        cmp = br.compare_reports(a, b)
        assert cmp.total is not None
        assert cmp.total.baseline_seconds == pytest.approx(3.0)
        assert cmp.total.candidate_seconds == pytest.approx(6.0)
        assert cmp.total.speedup == pytest.approx(0.5)

    def test_unmatched_stage_keeps_other_side_none(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0), ("align", 1.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0)])
        cmp = br.compare_reports(a, b)
        align_row = next(r for r in cmp.stages if r.name == "align")
        assert align_row.baseline_seconds == 1.0
        assert align_row.candidate_seconds is None
        assert align_row.speedup is None

    def test_total_excludes_unmatched_stages(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0), ("align", 1.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0)])
        cmp = br.compare_reports(a, b)
        assert cmp.total is not None
        # Only the common 'transcribe' row counts.
        assert cmp.total.baseline_seconds == pytest.approx(2.0)
        assert cmp.total.candidate_seconds == pytest.approx(4.0)

    def test_candidate_only_stage_is_appended(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        b = _filled_report(
            backend="rocm", stages=[("transcribe", 4.0), ("run_diarize", 0.5)]
        )
        cmp = br.compare_reports(a, b)
        names = [r.name for r in cmp.stages]
        assert names == ["transcribe", "run_diarize"]
        diarize_row = cmp.stages[1]
        assert diarize_row.baseline_seconds is None
        assert diarize_row.candidate_seconds == 0.5

    def test_failed_stage_drops_out_of_speedup(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        b = br.BenchmarkReport(
            backend="rocm",
            whisper_device="rocm",
            whisper_compute="float16",
            torch_device="rocm",
            model_name="tiny",
            language="en",
            audio_path="/tmp/sample.wav",
            audio_seconds=10.0,
        )
        b.add(br.Timing("transcribe", 0.1, ok=False, detail="kaboom"))
        cmp = br.compare_reports(a, b)
        # Failed candidate timing is treated as missing — we don't want a
        # 'rocm 20× faster' row for a stage that crashed before doing work.
        assert cmp.stages[0].candidate_seconds is None
        assert cmp.stages[0].speedup is None

    def test_total_is_none_when_no_common_stages(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        b = _filled_report(backend="rocm", stages=[("align", 1.0)])
        cmp = br.compare_reports(a, b)
        assert cmp.total is None

    def test_render_includes_warning_for_different_audio_paths(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0)])
        b.audio_path = "/tmp/different.wav"
        cmp = br.compare_reports(a, b)
        out = cmp.render()
        assert "different files" in out

    def test_render_shows_per_stage_verdict(self) -> None:
        a = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        b = _filled_report(backend="rocm", stages=[("transcribe", 4.0)])
        out = br.compare_reports(a, b).render()
        assert "transcribe" in out
        # Slower verdict surfaces; baseline is faster than candidate here.
        assert "slower" in out
        assert "total" in out


# --------------------------------------------------------------------------- #
# FakeHooks for run_benchmark
# --------------------------------------------------------------------------- #


class FakeHooks:
    """Recording stand-in for :class:`BenchHooks`. Each method counts its
    calls and stores its kwargs so tests can assert on the wiring."""

    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _maybe_fail(self, name: str) -> None:
        if self.fail_at == name:
            raise RuntimeError(f"forced failure at {name}")

    def load_whisper(self, **kw: Any) -> Any:
        self.calls.append(("load_whisper", kw))
        self._maybe_fail("load_whisper")
        return f"asr<{kw['model_name']}@{kw['device_arg']}/{kw['compute']}>"

    def load_audio(self, wav_path: Path) -> Any:
        self.calls.append(("load_audio", {"wav_path": wav_path}))
        self._maybe_fail("load_audio")
        return f"audio<{wav_path.name}>"

    def transcribe(self, asr: Any, audio: Any) -> Any:
        self.calls.append(("transcribe", {"asr": asr, "audio": audio}))
        self._maybe_fail("transcribe")
        return {"segments": [{"start": 0, "end": 1, "text": "hi"}]}

    def load_align(self, **kw: Any) -> Any:
        self.calls.append(("load_align", kw))
        self._maybe_fail("load_align")
        return ("align_model", "metadata")

    def align(self, **kw: Any) -> Any:
        self.calls.append(("align", kw))
        self._maybe_fail("align")
        return {"segments": kw.get("segments", [])}

    def load_diarize(self, **kw: Any) -> Any:
        self.calls.append(("load_diarize", kw))
        self._maybe_fail("load_diarize")
        return "pyannote_pipeline"

    def run_diarize(self, pipeline: Any, wav_path: Path) -> Any:
        self.calls.append(
            ("run_diarize", {"pipeline": pipeline, "wav_path": wav_path})
        )
        self._maybe_fail("run_diarize")
        return [("SPEAKER_00", 0.0, 1.0)]

    def as_bench_hooks(self) -> br.BenchHooks:
        return br.BenchHooks(
            load_whisper=self.load_whisper,
            load_audio=self.load_audio,
            transcribe=self.transcribe,
            load_align=self.load_align,
            align=self.align,
            load_diarize=self.load_diarize,
            run_diarize=self.run_diarize,
        )


def _force_cpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every engine helper return cpu / cpu / int8 / cpu so tests
    don't depend on whatever GPU is installed on the runner."""
    monkeypatch.setenv("SCRIBE_DEVICE", "cpu")
    monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "cpu")


# --------------------------------------------------------------------------- #
# run_benchmark
# --------------------------------------------------------------------------- #


class TestRunBenchmark:
    def test_happy_path_runs_first_five_stages_without_diarize(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=1.0)
        fake = FakeHooks()
        report = br.run_benchmark(
            audio_path=wav, hooks=fake.as_bench_hooks(), hardware="test-host"
        )
        names = [t.name for t in report.timings]
        assert names == [
            "load_whisper",
            "load_audio",
            "transcribe",
            "load_align_model",
            "align",
        ]
        assert report.ok
        assert report.backend == "cpu"
        assert report.audio_seconds == pytest.approx(1.0, abs=1e-3)
        # Skipped diarize is recorded so the JSON tells the truth.
        assert any("load_diarize" in s for s in report.skipped)

    def test_translates_rocm_label_to_cuda_at_library_boundary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Same plumbing as engine: rocm label collapses to "cuda" device-arg.
        monkeypatch.setenv("SCRIBE_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "float16")
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        report = br.run_benchmark(
            audio_path=wav, hooks=fake.as_bench_hooks(), hardware="rocm-test"
        )
        load_kw = dict(fake.calls)["load_whisper"]
        assert report.whisper_device == "rocm"
        assert load_kw["device_arg"] == "cuda"
        align_kw = dict(fake.calls)["load_align"]
        assert align_kw["device_arg"] == "cuda"

    def test_passes_model_name_and_language_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        br.run_benchmark(
            audio_path=wav,
            model_name="small",
            language="fr",
            hooks=fake.as_bench_hooks(),
        )
        load_kw = dict(fake.calls)["load_whisper"]
        assert load_kw["model_name"] == "small"
        assert load_kw["language"] == "fr"
        align_kw = dict(fake.calls)["load_align"]
        assert align_kw["language"] == "fr"

    def test_align_receives_segments_from_transcribe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        br.run_benchmark(audio_path=wav, hooks=fake.as_bench_hooks())
        align_kw = dict(fake.calls)["align"]
        # The transcribe stub returns a single 'hi' segment; align must
        # receive that segment list, not an empty one.
        assert align_kw["segments"] == [{"start": 0, "end": 1, "text": "hi"}]
        assert align_kw["align_model"] == "align_model"
        assert align_kw["metadata"] == "metadata"

    def test_skips_diarize_when_no_token_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        report = br.run_benchmark(
            audio_path=wav, include_diarize=True, hooks=fake.as_bench_hooks()
        )
        assert "load_diarize" not in [t.name for t in report.timings]
        assert any("HF_TOKEN" in s for s in report.skipped)
        assert report.ok  # skipped is not the same as failed

    def test_runs_diarize_when_token_provided_directly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        report = br.run_benchmark(
            audio_path=wav,
            include_diarize=True,
            hf_token="hf_fake",
            hooks=fake.as_bench_hooks(),
        )
        names = [t.name for t in report.timings]
        assert "load_diarize" in names
        assert "run_diarize" in names
        assert report.ok
        diar_kw = dict(fake.calls)["load_diarize"]
        assert diar_kw["hf_token"] == "hf_fake"

    def test_runs_diarize_when_token_in_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        br.run_benchmark(
            audio_path=wav,
            include_diarize=True,
            hooks=fake.as_bench_hooks(),
        )
        diar_kw = dict(fake.calls)["load_diarize"]
        assert diar_kw["hf_token"] == "hf_env"

    def test_stops_at_first_failure_and_reports_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks(fail_at="transcribe")
        report = br.run_benchmark(
            audio_path=wav,
            include_diarize=True,
            hf_token="x",
            hooks=fake.as_bench_hooks(),
        )
        names = [t.name for t in report.timings]
        # Stages run in order until the failure …
        assert names == ["load_whisper", "load_audio", "transcribe"]
        # … and align / diarize never start.
        assert "align" not in names
        assert "load_diarize" not in names
        assert not report.ok
        failure = next(t for t in report.timings if not t.ok)
        assert failure.name == "transcribe"
        assert "RuntimeError" in failure.detail

    def test_stops_at_align_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks(fail_at="align")
        report = br.run_benchmark(audio_path=wav, hooks=fake.as_bench_hooks())
        names = [t.name for t in report.timings]
        assert names[-1] == "align"
        assert not report.ok

    def test_missing_audio_file_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        with pytest.raises(FileNotFoundError):
            br.run_benchmark(audio_path=tmp_path / "no.wav")

    def test_audio_seconds_override_is_used_when_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # When the CLI has already probed the file (e.g. for an MP4), the
        # driver shouldn't try to read it as a WAV again.
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        fake = FakeHooks()
        report = br.run_benchmark(
            audio_path=wav,
            audio_seconds=42.0,
            hooks=fake.as_bench_hooks(),
        )
        assert report.audio_seconds == 42.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_defaults(self) -> None:
        ns = br.build_parser().parse_args(["sample.wav"])
        assert ns.audio == Path("sample.wav")
        assert ns.model == "tiny"
        assert ns.language == "en"
        assert ns.include_diarize is False
        assert ns.output is None
        assert ns.label is None
        assert ns.compare is None

    def test_compare_takes_two_paths(self) -> None:
        ns = br.build_parser().parse_args(["--compare", "a.json", "b.json"])
        assert ns.compare == [Path("a.json"), Path("b.json")]
        assert ns.audio is None  # audio not required for --compare

    def test_output_and_label_overrides(self) -> None:
        ns = br.build_parser().parse_args(
            ["sample.wav", "--output", "out.json", "--label", "RX 7900 XTX"]
        )
        assert ns.output == Path("out.json")
        assert ns.label == "RX 7900 XTX"


class TestMain:
    def test_returns_two_when_no_audio_and_no_compare(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        rc = br.main([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "audio path is required" in err

    def test_returns_two_when_audio_missing(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        rc = br.main([str(tmp_path / "missing.wav")])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_returns_zero_on_successful_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        # Replace run_benchmark with a fake to keep the test honest about
        # the CLI plumbing without bringing whisperx into scope.
        recorded: dict[str, Any] = {}

        def fake_run(**kw: Any) -> br.BenchmarkReport:
            recorded.update(kw)
            r = _empty_report()
            r.add(br.Timing("transcribe", 1.0))
            return r

        monkeypatch.setattr(br, "run_benchmark", fake_run)
        rc = br.main([str(wav)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scribe ROCm/CUDA benchmark" in out
        # WAV header read picked up the 0.2s duration the test wrote.
        assert recorded["audio_seconds"] == pytest.approx(0.2, abs=1e-3)
        assert recorded["model_name"] == "tiny"

    def test_returns_one_on_stage_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)

        def fake_run(**kw: Any) -> br.BenchmarkReport:
            r = _empty_report()
            r.add(br.Timing("load_whisper", 0.1, ok=False, detail="boom"))
            return r

        monkeypatch.setattr(br, "run_benchmark", fake_run)
        rc = br.main([str(wav)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_writes_json_output_when_requested(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        out_path = tmp_path / "out.json"

        def fake_run(**kw: Any) -> br.BenchmarkReport:
            r = _empty_report(audio_seconds=kw["audio_seconds"])
            r.add(br.Timing("transcribe", 1.0))
            return r

        monkeypatch.setattr(br, "run_benchmark", fake_run)
        rc = br.main([str(wav), "--output", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["backend"] == "cpu"
        assert data["timings"][0]["name"] == "transcribe"

    def test_forwards_label_override_as_hardware(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _force_cpu_backend(monkeypatch)
        wav = _silent_wav(tmp_path / "x.wav", seconds=0.2)
        recorded: dict[str, Any] = {}

        def fake_run(**kw: Any) -> br.BenchmarkReport:
            recorded.update(kw)
            r = _empty_report()
            r.add(br.Timing("a", 1.0))
            return r

        monkeypatch.setattr(br, "run_benchmark", fake_run)
        br.main([str(wav), "--label", "RX 7900 XTX"])
        assert recorded["hardware"] == "RX 7900 XTX"

    def test_compare_subcommand_round_trips_through_json(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        baseline = _filled_report(backend="cuda", stages=[("transcribe", 2.0)])
        candidate = _filled_report(backend="rocm", stages=[("transcribe", 4.0)])
        bp = tmp_path / "cuda.json"
        cp = tmp_path / "rocm.json"
        bp.write_text(json.dumps(baseline.to_dict()))
        cp.write_text(json.dumps(candidate.to_dict()))
        rc = br.main(["--compare", str(bp), str(cp)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Comparison: baseline=cuda" in out
        assert "transcribe" in out
        assert "slower" in out

    def test_compare_with_missing_baseline_returns_two(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        rc = br.main(
            ["--compare", str(tmp_path / "no.json"), str(tmp_path / "still-no.json")]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "baseline" in err and "not found" in err

    def test_compare_with_missing_candidate_returns_two(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        bp = tmp_path / "cuda.json"
        bp.write_text(
            json.dumps(_filled_report(backend="cuda", stages=[("a", 1.0)]).to_dict())
        )
        rc = br.main(["--compare", str(bp), str(tmp_path / "missing.json")])
        assert rc == 2
        err = capsys.readouterr().err
        assert "candidate" in err and "not found" in err
