"""
G6.1 — Smoke test for the AMD/ROCm path.

Run after installing the ROCm wheel via ``./setup.sh --rocm`` (or as a
generic device-config sanity check on any backend) with::

    .venv/bin/python -m scribe.scripts.check_rocm

The script does the smallest end-to-end exercise of the Scribe stack
that still touches every layer:

1. Generate a 5-second silent WAV (no model downloads beyond what's
   listed below; the audio itself is in-process).
2. Load the ``tiny`` Whisper model via WhisperX on the active GPU
   backend (CUDA, ROCm, MPS-falls-back-to-CPU, or CPU).
3. Run a one-shot transcription on the silence to prove the inference
   loop reaches the kernels.
4. Load the wav2vec2 alignment model on the same backend.
5. (Optional, requires ``HF_TOKEN``) Load the pyannote diarization
   pipeline and run it once on the silent clip. This is the layer that
   trips the MIOpen LSTM-dropout bug (G3.1) on RDNA cards, so it's the
   highest-value smoke step on AMD.

Each stage is wall-clock timed and reported with a status line. The
script exits 0 iff every requested stage succeeded; non-zero on the
first failure with a descriptive message that's safe to paste into a
GitHub issue.

Library notes:

* CTranslate2's ROCm wheel still takes ``device="cuda"`` (HIP shim).
  We translate the honest ``"rocm"`` label at the library boundary via
  :func:`scribe.engine._to_torch_device_arg`.
* On MPS (Apple Silicon) CTranslate2 has no Metal backend; the script
  drops to CPU there as designed.
* Diarization is off by default because it requires an HF account and
  the gated ``pyannote/speaker-diarization-3.1`` model. Pass
  ``--include-diarize`` (or set ``HF_TOKEN`` and pass
  ``--include-diarize``) to exercise it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Pure helpers — no model loads, fully testable
# --------------------------------------------------------------------------- #


def make_silent_wav(path: Path, *, seconds: float = 5.0, sr: int = 16000) -> Path:
    """Write a mono 16-bit PCM WAV of pure silence to ``path``.

    The audio Scribe expects everywhere is mono 16 kHz 16-bit PCM, so the
    fixture matches that exactly. Returns ``path`` for chainability with
    :func:`tempfile.TemporaryDirectory`. Pure I/O — no torch / no CT2.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds!r}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}")
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return path


@dataclass
class Stage:
    """Result of a single smoke-test stage.

    Captured as a dataclass so the report can be rendered, asserted on
    in tests, and (later) serialised to JSON for the support bundle.
    """

    name: str
    ok: bool
    seconds: float
    detail: str = ""

    def render(self) -> str:
        """One-line human-readable status. ``[OK]`` / ``[FAIL]`` lead so
        the line greps cleanly out of CI logs."""
        status = "OK  " if self.ok else "FAIL"
        line = f"[{status}] {self.name:<28s} {self.seconds:6.2f}s"
        if self.detail:
            line = f"{line}  {self.detail}"
        return line


@dataclass
class SmokeReport:
    """Aggregate result of a check_rocm run.

    ``backend`` / ``whisper_device`` / ``whisper_compute`` / ``torch_device``
    record what the engine helpers picked at the time of the run, so the
    output is self-describing without re-querying the device state.
    """

    backend: str
    whisper_device: str
    whisper_compute: str
    torch_device: str
    stages: list[Stage] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, stage: Stage) -> None:
        self.stages.append(stage)

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append(f"{name}: {reason}")

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.ok for s in self.stages)

    @property
    def first_failure(self) -> Optional[Stage]:
        for s in self.stages:
            if not s.ok:
                return s
        return None

    def render(self) -> str:
        """Multi-line report suitable for ``print()`` or a support ticket."""
        lines: list[str] = []
        lines.append("Scribe ROCm smoke test")
        lines.append("=" * 40)
        lines.append(f"Backend:         {self.backend}")
        lines.append(f"Whisper device:  {self.whisper_device}  compute={self.whisper_compute}")
        lines.append(f"Torch device:    {self.torch_device}")
        lines.append("")
        if not self.stages:
            lines.append("(no stages ran)")
        else:
            for s in self.stages:
                lines.append(s.render())
        if self.skipped:
            lines.append("")
            lines.append("Skipped:")
            for s in self.skipped:
                lines.append(f"  - {s}")
        lines.append("")
        if self.ok:
            lines.append(
                "All stages reached without crashing. Backend looks healthy."
            )
        elif not self.stages:
            lines.append("No stages were exercised — nothing to assert about.")
        else:
            failure = self.first_failure
            assert failure is not None  # narrow for type-checkers
            lines.append(
                f"FAILED at: {failure.name} — {failure.detail or 'see traceback above'}"
            )
        return "\n".join(lines) + "\n"


def time_call(fn: Callable[[], Any]) -> tuple[float, Any, Optional[BaseException]]:
    """Run ``fn`` and return ``(seconds, result_or_None, exception_or_None)``.

    Catches ``Exception`` (not ``BaseException``) so KeyboardInterrupt and
    SystemExit still abort the whole run cleanly. Pure timing wrapper —
    no logging, no printing, just a measurement primitive.
    """
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — we want the broad sweep here
        return time.perf_counter() - t0, None, exc
    return time.perf_counter() - t0, result, None


def _format_exc(exc: BaseException) -> str:
    """Short, paste-friendly exception summary for the report line."""
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Stage drivers — these touch real models when called for real, but each
# has its loader callable injected so tests can replace them with mocks.
# --------------------------------------------------------------------------- #


def _real_load_whisper(*, model_name: str, device_arg: str, compute: str, language: str) -> Any:
    """Default WhisperX loader. Imported lazily so ``check_rocm`` is
    importable on machines without WhisperX installed yet."""
    import whisperx  # type: ignore

    return whisperx.load_model(
        model_name, device=device_arg, compute_type=compute, language=language
    )


def _real_load_audio(wav_path: Path) -> Any:
    import whisperx  # type: ignore

    return whisperx.load_audio(str(wav_path))


def _real_transcribe(asr: Any, audio: Any) -> Any:
    return asr.transcribe(audio, batch_size=1)


def _real_load_align(*, language: str, device_arg: str) -> Any:
    import whisperx  # type: ignore

    return whisperx.load_align_model(language_code=language, device=device_arg)


def _real_load_diarize(*, hf_token: str, device_arg: str) -> Any:
    import whisperx  # type: ignore

    if hasattr(whisperx, "DiarizationPipeline"):
        return whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device_arg)
    from whisperx.diarize import DiarizationPipeline  # type: ignore

    return DiarizationPipeline(use_auth_token=hf_token, device=device_arg)


def _real_run_diarize(pipeline: Any, wav_path: Path) -> Any:
    return pipeline(str(wav_path))


# Bundle of injectable callables so tests can swap them out without
# monkeypatching the whisperx module itself.
@dataclass
class StageHooks:
    load_whisper: Callable[..., Any] = _real_load_whisper
    load_audio: Callable[[Path], Any] = _real_load_audio
    transcribe: Callable[[Any, Any], Any] = _real_transcribe
    load_align: Callable[..., Any] = _real_load_align
    load_diarize: Callable[..., Any] = _real_load_diarize
    run_diarize: Callable[[Any, Path], Any] = _real_run_diarize


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_smoke_test(
    *,
    seconds: float = 5.0,
    model_name: str = "tiny",
    language: str = "en",
    include_diarize: bool = False,
    hf_token: Optional[str] = None,
    workdir: Optional[Path] = None,
    hooks: Optional[StageHooks] = None,
) -> SmokeReport:
    """Run the smoke test end-to-end and return a structured report.

    The ``hooks`` argument exists so tests can inject mock loaders that
    return sentinel objects instead of pulling real model weights — the
    real production path uses the default :class:`StageHooks`, which
    forwards to whisperx.

    Stages (in order):

    1. ``load_whisper``    — instantiate the WhisperX wrapper.
    2. ``load_audio``      — decode the silent WAV to a NumPy array.
    3. ``transcribe``      — run a one-shot inference (silence → empty).
    4. ``load_align``      — load the language's wav2vec2 alignment model.
    5. ``load_diarize``    — (optional) load the pyannote pipeline.
    6. ``run_diarize``     — (optional) run pyannote on the silent clip.

    Stages stop at the first failure: a wedged backend won't hand back
    useful timings for later stages, so we bail fast. The report records
    every stage that ran plus an explanation for any that were skipped.
    """
    # Late import so the module is importable on systems without engine
    # deps installed yet (e.g. just `pip install -e .[dev]` for tests).
    from scribe.engine import (
        _to_torch_device_arg,
        _torch_device,
        _whisper_device_and_compute,
        gpu_backend,
    )

    hooks = hooks or StageHooks()

    backend = gpu_backend()
    w_dev, w_compute = _whisper_device_and_compute()
    t_dev = _torch_device()

    report = SmokeReport(
        backend=backend,
        whisper_device=w_dev,
        whisper_compute=w_compute,
        torch_device=t_dev,
    )

    # All-in-one tempdir so the silent WAV cleans up no matter which
    # exit branch the function takes.
    cm = tempfile.TemporaryDirectory() if workdir is None else None
    try:
        if cm is not None:
            base = Path(cm.name)
        else:
            assert workdir is not None
            base = workdir
        wav = base / "silent.wav"
        make_silent_wav(wav, seconds=seconds)

        # --- 1. Load Whisper -------------------------------------------------
        whisper_dev_arg = _to_torch_device_arg(w_dev)
        secs, asr, exc = time_call(
            lambda: hooks.load_whisper(
                model_name=model_name,
                device_arg=whisper_dev_arg,
                compute=w_compute,
                language=language,
            )
        )
        if exc is not None:
            report.add(Stage("load_whisper", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("load_whisper", True, secs, f"model={model_name}"))

        # --- 2. Load audio ---------------------------------------------------
        secs, audio, exc = time_call(lambda: hooks.load_audio(wav))
        if exc is not None:
            report.add(Stage("load_audio", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("load_audio", True, secs))

        # --- 3. Transcribe ---------------------------------------------------
        secs, _result, exc = time_call(lambda: hooks.transcribe(asr, audio))
        if exc is not None:
            report.add(Stage("transcribe_silence", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("transcribe_silence", True, secs))

        # --- 4. Alignment ----------------------------------------------------
        align_dev_arg = _to_torch_device_arg(t_dev)
        secs, _align, exc = time_call(
            lambda: hooks.load_align(language=language, device_arg=align_dev_arg)
        )
        if exc is not None:
            report.add(Stage("load_align_model", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("load_align_model", True, secs, f"lang={language}"))

        # --- 5/6. Diarization (optional) -------------------------------------
        if not include_diarize:
            report.skip(
                "load_diarize",
                "not requested (pass --include-diarize to exercise pyannote)",
            )
            return report

        token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        if not token:
            report.skip(
                "load_diarize",
                "HF_TOKEN not set — pyannote model is gated; cannot load",
            )
            return report

        secs, pipeline, exc = time_call(
            lambda: hooks.load_diarize(hf_token=token, device_arg=align_dev_arg)
        )
        if exc is not None:
            report.add(Stage("load_diarize", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("load_diarize", True, secs))

        secs, _diar, exc = time_call(lambda: hooks.run_diarize(pipeline, wav))
        if exc is not None:
            report.add(Stage("run_diarize", False, secs, _format_exc(exc)))
            return report
        report.add(Stage("run_diarize", True, secs))

        return report
    finally:
        if cm is not None:
            cm.cleanup()


# --------------------------------------------------------------------------- #
# Reachability surface — the smoke-test *plan* (no model loads)
#
# G6.1 reaches the user as a CLI script; the home-page UI surface is a
# read-only panel that shows the same CLI invocation + a JSON-shaped
# description of what each stage does. This helper returns that plan
# without touching whisperx / pyannote, so the FastAPI route and the
# Jinja template can both render it deterministically.
# --------------------------------------------------------------------------- #


# Stable list of stages in the order ``run_smoke_test`` exercises them. The
# CLI driver, the JSON route, and the template-rendered checklist on the
# home page all read this list so they can never disagree about the
# stage names or the count.
SMOKE_TEST_STAGES: tuple[dict[str, str], ...] = (
    {
        "name": "load_whisper",
        "summary": "Instantiate the WhisperX wrapper for the active backend.",
    },
    {
        "name": "load_audio",
        "summary": "Decode the silent fixture WAV to a NumPy array.",
    },
    {
        "name": "transcribe_silence",
        "summary": "Run a one-shot inference on silence (proves the kernels reach).",
    },
    {
        "name": "load_align_model",
        "summary": "Load the language's wav2vec2 alignment model.",
    },
    {
        "name": "load_diarize",
        "summary": "Optional. Load pyannote/speaker-diarization-3.1 (needs HF_TOKEN).",
    },
    {
        "name": "run_diarize",
        "summary": "Optional. Run pyannote on the silent clip (probes MIOpen LSTM bug).",
    },
)


# Process exit codes the CLI returns. The plan surface advertises these so
# scripted callers (CI bots, support-scripts) can rely on them being part
# of the contract rather than reading them out of source.
SMOKE_TEST_EXIT_CODES: tuple[dict[str, str], ...] = (
    {"code": 0, "meaning": "healthy — every requested stage reached"},
    {"code": 1, "meaning": "stage_failure — first failing stage's exception is in the report"},
    {"code": 2, "meaning": "bad_cli_args — argparse rejected the invocation"},
)


# Default values match the CLI defaults baked into ``build_parser``. We
# repeat them here as data so the home-page panel can render them
# without re-importing argparse.
SMOKE_TEST_DEFAULTS: dict[str, Any] = {
    "seconds": 5.0,
    "model": "tiny",
    "language": "en",
    "include_diarize": False,
}


# CLI invocation strings the home page surfaces verbatim. The
# ``cli_venv`` form matches the README's "Verify it took:" snippet so a
# user copy-pasting from the panel runs the exact command we document.
SMOKE_TEST_CLI = "python -m scribe.scripts.check_rocm"
SMOKE_TEST_CLI_VENV = ".venv/bin/python -m scribe.scripts.check_rocm"


def smoke_test_plan() -> dict[str, Any]:
    """Return a structured description of what the smoke test does.

    No model loads, no I/O, no torch import — pure metadata. The FastAPI
    route ``GET /api/diagnostics/smoke-test-plan`` returns this dict as
    JSON; the home-page template renders the same fields into a panel
    so a user knows what the CLI invocation will run before running it.

    Keys:

    * ``cli`` / ``cli_venv`` — copy-paste invocation strings.
    * ``stages`` — ordered list of ``{name, summary}`` dicts mirroring
      the order in which ``run_smoke_test`` executes them.
    * ``defaults`` — argparse defaults (seconds / model / language /
      include_diarize).
    * ``exit_codes`` — ``[{code, meaning}, ...]`` triples.
    * ``fail_fast`` — ``True`` (the driver stops at the first failure
      so a wedged backend doesn't produce nonsense timings later).
    * ``feature_id`` — ``"G6.1"`` so JS test-id selectors can pin the
      panel.
    * ``docs_anchor`` — README section heading the user can jump to
      from the in-app docs viewer (``/docs/readme``) for context.
    """
    return {
        "feature_id": "G6.1",
        "cli": SMOKE_TEST_CLI,
        "cli_venv": SMOKE_TEST_CLI_VENV,
        "stages": [dict(stage) for stage in SMOKE_TEST_STAGES],
        "defaults": dict(SMOKE_TEST_DEFAULTS),
        "exit_codes": [dict(ec) for ec in SMOKE_TEST_EXIT_CODES],
        "fail_fast": True,
        "docs_anchor": "linux-amd-gpu--rocm",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.check_rocm",
        description=(
            "Smoke test for the Scribe ROCm/CUDA/MPS/CPU stack (G6.1). "
            "Loads a tiny Whisper, runs one inference, loads the alignment "
            "model, optionally exercises pyannote diarization. Reports "
            "wall-clock timings."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Length of the silent test clip in seconds (default: 5).",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name to load (default: tiny). "
        "Override only if you specifically want to test a larger model.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for the alignment model (default: en).",
    )
    parser.add_argument(
        "--include-diarize",
        action="store_true",
        help=(
            "Also load + run pyannote speaker-diarization-3.1. Requires "
            "HF_TOKEN; skipped with a notice when unset. This is the layer "
            "that trips the MIOpen LSTM-dropout bug (G3.1) on AMD."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = healthy, 1 = stage
    failure, 2 = bad CLI args)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seconds <= 0:
        print(f"error: --seconds must be > 0 (got {args.seconds!r})", file=sys.stderr)
        return 2

    report = run_smoke_test(
        seconds=args.seconds,
        model_name=args.model,
        language=args.language,
        include_diarize=args.include_diarize,
    )
    sys.stdout.write(report.render())
    sys.stdout.flush()
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
