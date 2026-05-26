"""Tests for ``scribe.scripts.check_rocm`` (G6.1 smoke-test script).

The script is a CLI that loads real ML weights at runtime, so the tests
exercise:

* the **pure helpers** — :func:`make_silent_wav`, :func:`time_call`,
  :class:`Stage`, :class:`SmokeReport` — directly;
* the **stage driver** :func:`run_smoke_test` with the
  :class:`StageHooks` injection point that lets a test stand in fake
  loaders for whisperx / pyannote without a single MB of weights;
* the **argparse + main** layer with the same hooks and a forced backend
  via ``SCRIBE_DEVICE``.

The real model loaders themselves are not exercised — that's what the
``slow`` / ``gpu`` markers are for, and G6.1 is meant to be runnable on
any developer machine.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from scribe.scripts import check_rocm as cr


# --------------------------------------------------------------------------- #
# make_silent_wav
# --------------------------------------------------------------------------- #


class TestMakeSilentWav:
    def test_writes_a_valid_pcm_wav(self, tmp_path: Path) -> None:
        out = cr.make_silent_wav(tmp_path / "s.wav", seconds=1.0, sr=16000)
        assert out.exists()
        with wave.open(str(out), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            assert w.getnframes() == 16000

    def test_returns_the_path_for_chainability(self, tmp_path: Path) -> None:
        target = tmp_path / "x.wav"
        assert cr.make_silent_wav(target, seconds=0.1) is target

    def test_writes_pure_silence(self, tmp_path: Path) -> None:
        out = cr.make_silent_wav(tmp_path / "s.wav", seconds=0.25, sr=8000)
        with wave.open(str(out), "rb") as w:
            frames = w.readframes(w.getnframes())
        # Every byte is zero — pure silence.
        assert frames == b"\x00\x00" * 2000
        assert set(frames) == {0}

    def test_rejects_non_positive_seconds(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="seconds"):
            cr.make_silent_wav(tmp_path / "x.wav", seconds=0.0)
        with pytest.raises(ValueError, match="seconds"):
            cr.make_silent_wav(tmp_path / "x.wav", seconds=-1.0)

    def test_rejects_non_positive_sample_rate(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sr"):
            cr.make_silent_wav(tmp_path / "x.wav", seconds=1.0, sr=0)


# --------------------------------------------------------------------------- #
# time_call
# --------------------------------------------------------------------------- #


class TestTimeCall:
    def test_returns_result_when_callable_succeeds(self) -> None:
        secs, result, exc = cr.time_call(lambda: 42)
        assert exc is None
        assert result == 42
        assert secs >= 0.0

    def test_captures_exception_without_raising(self) -> None:
        boom = RuntimeError("nope")

        def raises() -> None:
            raise boom

        secs, result, exc = cr.time_call(raises)
        assert exc is boom
        assert result is None
        assert secs >= 0.0

    def test_lets_keyboard_interrupt_propagate(self) -> None:
        # KeyboardInterrupt is BaseException, not Exception — must NOT be
        # swallowed by the timing wrapper, otherwise Ctrl-C during a real
        # smoke test would silently look like a stage success.
        def raises() -> None:
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            cr.time_call(raises)


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #


class TestStage:
    def test_render_for_ok_stage(self) -> None:
        line = cr.Stage("load_whisper", True, 1.234, "model=tiny").render()
        assert line.startswith("[OK")
        assert "load_whisper" in line
        assert "1.23" in line
        assert "model=tiny" in line

    def test_render_for_failed_stage(self) -> None:
        line = cr.Stage("load_align_model", False, 0.5, "RuntimeError: kaboom").render()
        assert line.startswith("[FAIL]")
        assert "load_align_model" in line
        assert "kaboom" in line

    def test_render_omits_detail_when_blank(self) -> None:
        line = cr.Stage("load_audio", True, 0.1).render()
        assert "  " in line  # padding still present
        assert not line.endswith(" ")  # no trailing detail spaces


# --------------------------------------------------------------------------- #
# SmokeReport
# --------------------------------------------------------------------------- #


def _empty_report(backend: str = "cpu") -> cr.SmokeReport:
    return cr.SmokeReport(
        backend=backend,
        whisper_device="cpu",
        whisper_compute="int8",
        torch_device="cpu",
    )


class TestSmokeReport:
    def test_ok_is_false_when_no_stages_ran(self) -> None:
        r = _empty_report()
        assert r.ok is False

    def test_ok_is_true_when_all_stages_pass(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("a", True, 0.1))
        r.add(cr.Stage("b", True, 0.2))
        assert r.ok is True

    def test_ok_is_false_with_any_failure(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("a", True, 0.1))
        r.add(cr.Stage("b", False, 0.2, "boom"))
        r.add(cr.Stage("c", True, 0.3))
        assert r.ok is False

    def test_first_failure_picks_earliest(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("a", True, 0.0))
        r.add(cr.Stage("b", False, 0.0, "first"))
        r.add(cr.Stage("c", False, 0.0, "second"))
        f = r.first_failure
        assert f is not None and f.detail == "first"

    def test_first_failure_none_when_all_ok(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("a", True, 0.0))
        assert r.first_failure is None

    def test_render_includes_header_and_stage_lines(self) -> None:
        r = _empty_report(backend="rocm")
        r.whisper_device = "rocm"
        r.whisper_compute = "float16"
        r.torch_device = "rocm"
        r.add(cr.Stage("load_whisper", True, 0.5, "model=tiny"))
        r.add(cr.Stage("transcribe_silence", True, 0.2))
        out = r.render()
        assert "Backend:" in out
        assert "rocm" in out
        assert "compute=float16" in out
        assert "load_whisper" in out
        assert "transcribe_silence" in out
        assert "Backend looks healthy" in out

    def test_render_includes_skipped_section_when_set(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("load_whisper", True, 0.1))
        r.skip("load_diarize", "HF_TOKEN not set")
        out = r.render()
        assert "Skipped:" in out
        assert "HF_TOKEN not set" in out

    def test_render_calls_out_first_failure(self) -> None:
        r = _empty_report()
        r.add(cr.Stage("load_whisper", True, 0.1))
        r.add(cr.Stage("transcribe_silence", False, 0.5, "RuntimeError: HSA fault"))
        out = r.render()
        assert "FAILED at: transcribe_silence" in out
        assert "HSA fault" in out
        assert "Backend looks healthy" not in out


# --------------------------------------------------------------------------- #
# Stage hooks: a fake StageHooks impl that records every call
# --------------------------------------------------------------------------- #


class FakeHooks:
    """Recording stand-in for :class:`StageHooks`. Each method counts its
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
        return {"segments": []}

    def load_align(self, **kw: Any) -> Any:
        self.calls.append(("load_align", kw))
        self._maybe_fail("load_align")
        return ("align_model", "metadata")

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

    def as_stage_hooks(self) -> cr.StageHooks:
        return cr.StageHooks(
            load_whisper=self.load_whisper,
            load_audio=self.load_audio,
            transcribe=self.transcribe,
            load_align=self.load_align,
            load_diarize=self.load_diarize,
            run_diarize=self.run_diarize,
        )


def _force_cpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every engine helper return the cpu / cpu / int8 / cpu shape so
    the tests don't depend on whatever GPU is actually installed."""
    monkeypatch.setenv("SCRIBE_DEVICE", "cpu")
    monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "cpu")


# --------------------------------------------------------------------------- #
# run_smoke_test
# --------------------------------------------------------------------------- #


class TestRunSmokeTest:
    def test_happy_path_runs_first_four_stages_without_diarize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks()
        report = cr.run_smoke_test(
            seconds=0.2,
            include_diarize=False,
            hooks=fake.as_stage_hooks(),
        )
        names = [s.name for s in report.stages]
        assert names == [
            "load_whisper",
            "load_audio",
            "transcribe_silence",
            "load_align_model",
        ]
        assert report.ok
        assert report.backend == "cpu"
        # Skipped diarize is recorded so the report tells the truth.
        assert any("load_diarize" in s for s in report.skipped)

    def test_translates_rocm_label_to_cuda_at_library_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same plumbing as engine: rocm label collapses to "cuda" device-arg.
        monkeypatch.setenv("SCRIBE_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_WHISPER_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_DIARIZE_DEVICE", "rocm")
        monkeypatch.setenv("SCRIBE_COMPUTE_TYPE", "float16")
        fake = FakeHooks()
        report = cr.run_smoke_test(
            seconds=0.2,
            include_diarize=False,
            hooks=fake.as_stage_hooks(),
        )
        load_kw = dict(fake.calls)["load_whisper"]
        # The honest backend label still says "rocm" in the report …
        assert report.whisper_device == "rocm"
        # … but the actual device argument handed to whisperx is "cuda".
        assert load_kw["device_arg"] == "cuda"
        align_kw = dict(fake.calls)["load_align"]
        assert align_kw["device_arg"] == "cuda"

    def test_passes_model_name_and_language_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks()
        cr.run_smoke_test(
            seconds=0.1,
            model_name="small",
            language="fr",
            include_diarize=False,
            hooks=fake.as_stage_hooks(),
        )
        load_kw = dict(fake.calls)["load_whisper"]
        assert load_kw["model_name"] == "small"
        assert load_kw["language"] == "fr"
        align_kw = dict(fake.calls)["load_align"]
        assert align_kw["language"] == "fr"

    def test_skips_diarize_when_no_token_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        # _isolate_env autouse fixture already deletes HF_TOKEN, but be
        # explicit here so the test reads top-to-bottom.
        monkeypatch.delenv("HF_TOKEN", raising=False)
        fake = FakeHooks()
        report = cr.run_smoke_test(
            seconds=0.1,
            include_diarize=True,
            hooks=fake.as_stage_hooks(),
        )
        assert "load_diarize" not in [s.name for s in report.stages]
        assert any("HF_TOKEN" in s for s in report.skipped)
        # Still healthy — skipped is not the same as failed.
        assert report.ok

    def test_runs_diarize_when_token_provided_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks()
        report = cr.run_smoke_test(
            seconds=0.1,
            include_diarize=True,
            hf_token="hf_fake_test_token",
            hooks=fake.as_stage_hooks(),
        )
        names = [s.name for s in report.stages]
        assert "load_diarize" in names
        assert "run_diarize" in names
        assert report.ok
        diar_kw = dict(fake.calls)["load_diarize"]
        assert diar_kw["hf_token"] == "hf_fake_test_token"

    def test_runs_diarize_when_token_in_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        fake = FakeHooks()
        cr.run_smoke_test(
            seconds=0.1,
            include_diarize=True,
            hooks=fake.as_stage_hooks(),
        )
        diar_kw = dict(fake.calls)["load_diarize"]
        assert diar_kw["hf_token"] == "hf_env_token"

    def test_stops_at_first_failure_and_reports_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks(fail_at="transcribe")
        report = cr.run_smoke_test(
            seconds=0.1,
            include_diarize=True,
            hf_token="x",
            hooks=fake.as_stage_hooks(),
        )
        names = [s.name for s in report.stages]
        # Stages run in order until the failure …
        assert names == ["load_whisper", "load_audio", "transcribe_silence"]
        # … and the alignment / diarize stages never run.
        assert "load_align_model" not in names
        assert "load_diarize" not in names
        assert not report.ok
        failure = report.first_failure
        assert failure is not None
        assert failure.name == "transcribe_silence"
        assert "RuntimeError" in failure.detail

    def test_stops_at_load_whisper_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks(fail_at="load_whisper")
        report = cr.run_smoke_test(
            seconds=0.1, include_diarize=False, hooks=fake.as_stage_hooks()
        )
        assert [s.name for s in report.stages] == ["load_whisper"]
        assert not report.ok

    def test_workdir_argument_is_used_when_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _force_cpu_backend(monkeypatch)
        fake = FakeHooks()
        cr.run_smoke_test(
            seconds=0.1,
            include_diarize=False,
            hooks=fake.as_stage_hooks(),
            workdir=tmp_path,
        )
        # The fake load_audio recorded the wav path; it must live under
        # the supplied tmp_path, not a system tempdir.
        audio_kw = dict(fake.calls)["load_audio"]
        assert audio_kw["wav_path"].parent == tmp_path
        assert audio_kw["wav_path"].exists()


# --------------------------------------------------------------------------- #
# build_parser / main
# --------------------------------------------------------------------------- #


class TestBuildParser:
    def test_defaults(self) -> None:
        ns = cr.build_parser().parse_args([])
        assert ns.seconds == 5.0
        assert ns.model == "tiny"
        assert ns.language == "en"
        assert ns.include_diarize is False

    def test_include_diarize_flag_flips(self) -> None:
        ns = cr.build_parser().parse_args(["--include-diarize"])
        assert ns.include_diarize is True

    def test_seconds_and_language_overrides(self) -> None:
        ns = cr.build_parser().parse_args(
            ["--seconds", "2.5", "--language", "de", "--model", "small"]
        )
        assert ns.seconds == 2.5
        assert ns.language == "de"
        assert ns.model == "small"


class TestMain:
    def _patch_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        report: cr.SmokeReport,
    ) -> list[dict[str, Any]]:
        """Replace run_smoke_test with a recorder; return the captured
        kwargs list so the test can assert on the call shape."""
        captured: list[dict[str, Any]] = []

        def fake_run(**kw: Any) -> cr.SmokeReport:
            captured.append(kw)
            return report

        monkeypatch.setattr(cr, "run_smoke_test", fake_run)
        return captured

    def test_returns_zero_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        report = _empty_report()
        report.add(cr.Stage("load_whisper", True, 0.1))
        self._patch_run(monkeypatch, report)
        rc = cr.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scribe ROCm smoke test" in out
        assert "Backend looks healthy" in out

    def test_returns_one_on_stage_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        report = _empty_report()
        report.add(cr.Stage("load_whisper", False, 0.1, "boom"))
        self._patch_run(monkeypatch, report)
        rc = cr.main([])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED at: load_whisper" in out

    def test_returns_two_on_invalid_seconds(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        rc = cr.main(["--seconds", "0"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "--seconds" in err

    def test_forwards_cli_flags_to_run_smoke_test(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        report = _empty_report()
        report.add(cr.Stage("load_whisper", True, 0.1))
        captured = self._patch_run(monkeypatch, report)
        cr.main(
            [
                "--seconds", "1.5",
                "--model", "small",
                "--language", "fr",
                "--include-diarize",
            ]
        )
        assert len(captured) == 1
        kw = captured[0]
        assert kw["seconds"] == 1.5
        assert kw["model_name"] == "small"
        assert kw["language"] == "fr"
        assert kw["include_diarize"] is True
