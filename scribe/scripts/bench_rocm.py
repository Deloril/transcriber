"""
G6.2 — In-house benchmark for the AMD/ROCm path.

Where G6.1's ``check_rocm`` is the *does-it-crash* gate, this script is
the *is-it-actually-fast* gate. Run it once on a CUDA box and once on a
ROCm box, save the JSON reports, and ``--compare`` them before publishing
any performance numbers.

Usage::

    # Single-box run on a representative file:
    .venv/bin/python -m scribe.scripts.bench_rocm path/to/sample.wav \
        --model tiny --output cuda.json

    # Same file, second box:
    .venv/bin/python -m scribe.scripts.bench_rocm path/to/sample.wav \
        --model tiny --output rocm.json

    # Compare:
    .venv/bin/python -m scribe.scripts.bench_rocm \
        --compare cuda.json rocm.json

Each stage is wall-clock timed and converted to a real-time-factor
(``audio_seconds / wall_seconds``) so the numbers stay meaningful across
clip lengths. The comparison view computes per-stage speedup ratios with
clear "faster"/"slower" wording.

Library notes mirror G6.1:

* CTranslate2's ROCm wheel still takes ``device="cuda"`` (HIP shim);
  we translate the honest ``"rocm"`` label at the library boundary via
  :func:`scribe.engine._to_torch_device_arg`.
* Diarization is gated by ``HF_TOKEN``; when missing, the diarize stages
  are skipped (recorded as such in the report) rather than failing.
* The driver is hook-injectable so tests can run the full report-shape
  contract without a single MB of model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Pure helpers — no model loads, fully testable
# --------------------------------------------------------------------------- #


def wav_duration_seconds(path: Path) -> float:
    """Return the duration of a PCM WAV in seconds.

    Used only for benchmark reports; we want a quick header-only read
    that doesn't need ffprobe on the host. For non-WAV input the CLI
    falls back to ``probe_media_info`` from :mod:`scribe.audio`. Raises
    :class:`ValueError` for empty or malformed WAVs (sample rate 0).
    """
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    if rate <= 0:
        raise ValueError(f"{path}: invalid sample rate {rate}")
    return frames / float(rate)


def _rtf(audio_seconds: float, wall_seconds: float) -> Optional[float]:
    """Real-time factor: how many seconds of audio per second of wall clock.

    A value > 1 means faster than real time. Returns ``None`` for the
    degenerate ``wall_seconds <= 0`` case so the report can render
    ``--`` instead of crashing on division.
    """
    if wall_seconds <= 0 or audio_seconds <= 0:
        return None
    return audio_seconds / wall_seconds


def format_speedup(factor: Optional[float]) -> str:
    """Render a speedup factor as a human-readable phrase.

    ``None`` → ``"n/a"``. ``factor >= 1`` → ``"<x>× faster"``;
    ``factor < 1`` → ``"<1/x>× slower"``. The candidate is always the
    point of view: "candidate is X× faster than baseline".
    """
    if factor is None:
        return "n/a"
    if factor <= 0:
        return "n/a"
    if factor >= 1.0:
        return f"{factor:.2f}× faster"
    return f"{1.0 / factor:.2f}× slower"


def time_call(fn: Callable[[], Any]) -> tuple[float, Any, Optional[BaseException]]:
    """Run ``fn`` and return ``(seconds, result_or_None, exception_or_None)``.

    Catches ``Exception`` (not ``BaseException``) so KeyboardInterrupt
    still aborts a real run cleanly. Same semantics as
    :func:`scribe.scripts.check_rocm.time_call` — kept local so the two
    scripts don't develop accidental coupling.
    """
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — broad sweep is the point here
        return time.perf_counter() - t0, None, exc
    return time.perf_counter() - t0, result, None


def _format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


@dataclass
class Timing:
    """Wall-clock measurement for a single stage of the pipeline.

    ``ok=False`` is recorded with the exception summary in ``detail`` so
    the report can show partial runs (e.g. transcribe worked, alignment
    crashed) instead of dropping everything on the floor.
    """

    name: str
    seconds: float
    ok: bool = True
    detail: str = ""

    def render(self, audio_seconds: float) -> str:
        status = "OK  " if self.ok else "FAIL"
        rtf = _rtf(audio_seconds, self.seconds)
        rtf_str = f"{rtf:6.2f}× RT" if rtf is not None else "  --   "
        line = f"[{status}] {self.name:<22s} {self.seconds:7.2f}s  {rtf_str}"
        if self.detail:
            line = f"{line}  {self.detail}"
        return line


@dataclass
class BenchmarkReport:
    """Full result of a benchmark run.

    Designed to be JSON-serialised (``to_dict`` / ``from_dict``) so two
    machines' reports can be compared offline. Every field that goes
    into the comparison is plain-data — no objects, no callables.
    """

    backend: str
    whisper_device: str
    whisper_compute: str
    torch_device: str
    model_name: str
    language: str
    audio_path: str
    audio_seconds: float
    hardware: str = ""
    started_at: float = 0.0
    timings: list[Timing] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, timing: Timing) -> None:
        self.timings.append(timing)

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append(f"{name}: {reason}")

    @property
    def ok(self) -> bool:
        """A report is ``ok`` only when at least one stage ran *and*
        every stage that ran succeeded. Empty/all-skipped runs are not
        a green light — there's nothing to make a claim about."""
        return bool(self.timings) and all(t.ok for t in self.timings)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)

    @property
    def overall_rtf(self) -> Optional[float]:
        """Real-time factor across the whole pipeline. Useful as a
        single headline number, but per-stage RTF is what you actually
        publish — different stages have wildly different RTFs."""
        return _rtf(self.audio_seconds, self.total_seconds)

    def stage(self, name: str) -> Optional[Timing]:
        for t in self.timings:
            if t.name == name:
                return t
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        timings = [Timing(**t) for t in data.get("timings", [])]
        return cls(
            backend=data["backend"],
            whisper_device=data["whisper_device"],
            whisper_compute=data["whisper_compute"],
            torch_device=data["torch_device"],
            model_name=data["model_name"],
            language=data["language"],
            audio_path=data["audio_path"],
            audio_seconds=float(data["audio_seconds"]),
            hardware=data.get("hardware", ""),
            started_at=float(data.get("started_at", 0.0)),
            timings=timings,
            skipped=list(data.get("skipped", [])),
        )

    def render(self) -> str:
        lines: list[str] = []
        lines.append("Scribe ROCm/CUDA benchmark")
        lines.append("=" * 48)
        lines.append(f"Backend:         {self.backend}")
        lines.append(
            f"Whisper device:  {self.whisper_device}  compute={self.whisper_compute}"
        )
        lines.append(f"Torch device:    {self.torch_device}")
        if self.hardware:
            lines.append(f"Hardware:        {self.hardware}")
        lines.append(f"Model:           {self.model_name}  lang={self.language}")
        lines.append(
            f"Audio:           {self.audio_path}  ({self.audio_seconds:.2f}s)"
        )
        lines.append("")
        if not self.timings:
            lines.append("(no stages ran)")
        else:
            for t in self.timings:
                lines.append(t.render(self.audio_seconds))
            total = self.total_seconds
            rtf = self.overall_rtf
            rtf_str = f"{rtf:6.2f}× RT" if rtf is not None else "  --   "
            lines.append("-" * 48)
            lines.append(f"  total                  {total:7.2f}s  {rtf_str}")
        if self.skipped:
            lines.append("")
            lines.append("Skipped:")
            for s in self.skipped:
                lines.append(f"  - {s}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class StageComparison:
    """One row of a side-by-side comparison.

    ``speedup`` is candidate-relative: ``baseline_seconds /
    candidate_seconds``. Values > 1 mean candidate is faster; < 1 means
    slower; ``None`` when one side didn't run that stage."""

    name: str
    baseline_seconds: Optional[float]
    candidate_seconds: Optional[float]
    speedup: Optional[float]


@dataclass
class BenchmarkComparison:
    """Two-report comparison. Baseline first, candidate second.

    Holds per-stage rows plus a ``total`` row so reports can call out
    one headline number without forcing the reader to do arithmetic."""

    baseline: BenchmarkReport
    candidate: BenchmarkReport
    stages: list[StageComparison] = field(default_factory=list)
    total: Optional[StageComparison] = None

    def render(self) -> str:
        b = self.baseline
        c = self.candidate
        lines: list[str] = []
        lines.append(
            f"Comparison: baseline={b.backend} ({b.whisper_compute})  "
            f"candidate={c.backend} ({c.whisper_compute})"
        )
        lines.append("=" * 72)
        lines.append(f"Model:           {b.model_name}  lang={b.language}")
        lines.append(f"Audio:           {b.audio_seconds:.2f}s")
        if b.audio_path != c.audio_path:
            lines.append(
                f"  warning: baseline & candidate ran on different files "
                f"({b.audio_path} vs {c.audio_path})"
            )
        lines.append("")
        header = f"{'stage':<22s}  {'baseline':>10s}  {'candidate':>10s}  {'verdict':>16s}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.stages:
            base = (
                f"{row.baseline_seconds:7.2f}s"
                if row.baseline_seconds is not None
                else "    --  "
            )
            cand = (
                f"{row.candidate_seconds:7.2f}s"
                if row.candidate_seconds is not None
                else "    --  "
            )
            verdict = format_speedup(row.speedup)
            lines.append(f"{row.name:<22s}  {base:>10s}  {cand:>10s}  {verdict:>16s}")
        if self.total is not None:
            lines.append("-" * len(header))
            base = f"{self.total.baseline_seconds:7.2f}s"
            cand = f"{self.total.candidate_seconds:7.2f}s"
            verdict = format_speedup(self.total.speedup)
            lines.append(
                f"{'total':<22s}  {base:>10s}  {cand:>10s}  {verdict:>16s}"
            )
        return "\n".join(lines) + "\n"


def compare_reports(
    baseline: BenchmarkReport, candidate: BenchmarkReport
) -> BenchmarkComparison:
    """Build a per-stage comparison between two benchmark reports.

    Pairs stages by name (so reordering between runs is fine), and
    keeps a stage that ran on only one side with the missing side as
    ``None`` and ``speedup`` as ``None``. The total row uses the sum
    of stages that *both* sides ran, so a missing diarize on one side
    doesn't poison the headline ratio."""
    by_name_b = {t.name: t for t in baseline.timings}
    by_name_c = {t.name: t for t in candidate.timings}

    # Preserve baseline order, then append candidate-only stages.
    seen: set[str] = set()
    rows: list[StageComparison] = []
    for t in baseline.timings:
        if t.name in seen:
            continue
        seen.add(t.name)
        rows.append(_pair_row(t.name, by_name_b, by_name_c))
    for t in candidate.timings:
        if t.name in seen:
            continue
        seen.add(t.name)
        rows.append(_pair_row(t.name, by_name_b, by_name_c))

    # Total: sum stages that both ran. If neither side ran any common
    # stage, the total is None — there's nothing honest to report.
    common = [r for r in rows if r.baseline_seconds is not None and r.candidate_seconds is not None]
    total: Optional[StageComparison] = None
    if common:
        bsum = sum(r.baseline_seconds or 0.0 for r in common)
        csum = sum(r.candidate_seconds or 0.0 for r in common)
        total = StageComparison(
            name="total",
            baseline_seconds=bsum,
            candidate_seconds=csum,
            speedup=(bsum / csum) if csum > 0 else None,
        )

    return BenchmarkComparison(
        baseline=baseline, candidate=candidate, stages=rows, total=total
    )


def _pair_row(
    name: str,
    by_name_b: dict[str, Timing],
    by_name_c: dict[str, Timing],
) -> StageComparison:
    b = by_name_b.get(name)
    c = by_name_c.get(name)
    bs = b.seconds if (b is not None and b.ok) else None
    cs = c.seconds if (c is not None and c.ok) else None
    speedup: Optional[float] = None
    if bs is not None and cs is not None and cs > 0:
        speedup = bs / cs
    return StageComparison(
        name=name, baseline_seconds=bs, candidate_seconds=cs, speedup=speedup
    )


# --------------------------------------------------------------------------- #
# Stage hooks — real implementations, swappable in tests
# --------------------------------------------------------------------------- #


def _real_load_whisper(*, model_name: str, device_arg: str, compute: str, language: str) -> Any:
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


def _real_align(*, segments: Any, align_model: Any, metadata: Any, audio: Any, device_arg: str) -> Any:
    import whisperx  # type: ignore

    return whisperx.align(
        segments, align_model, metadata, audio, device_arg, return_char_alignments=False
    )


def _real_load_diarize(*, hf_token: str, device_arg: str) -> Any:
    import whisperx  # type: ignore

    if hasattr(whisperx, "DiarizationPipeline"):
        return whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device_arg)
    from whisperx.diarize import DiarizationPipeline  # type: ignore

    return DiarizationPipeline(use_auth_token=hf_token, device=device_arg)


def _real_run_diarize(pipeline: Any, wav_path: Path) -> Any:
    return pipeline(str(wav_path))


@dataclass
class BenchHooks:
    """Injectable bundle of stage drivers.

    Identical pattern to :class:`scribe.scripts.check_rocm.StageHooks`
    but with the additional ``align`` step the benchmark exercises (the
    smoke test only loads the alignment model; the benchmark runs it)."""

    load_whisper: Callable[..., Any] = _real_load_whisper
    load_audio: Callable[[Path], Any] = _real_load_audio
    transcribe: Callable[[Any, Any], Any] = _real_transcribe
    load_align: Callable[..., Any] = _real_load_align
    align: Callable[..., Any] = _real_align
    load_diarize: Callable[..., Any] = _real_load_diarize
    run_diarize: Callable[[Any, Path], Any] = _real_run_diarize


# --------------------------------------------------------------------------- #
# Hardware label
# --------------------------------------------------------------------------- #


def _default_hardware_label() -> str:
    """Best-effort one-line description of the current GPU.

    Imported lazily so the module is still importable on a stripped-down
    test box. Falls back to platform info on CPU; never raises."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                hip = getattr(torch.version, "hip", None)
                if hip:
                    return f"{name} (ROCm/HIP {hip})"
                cuda = getattr(torch.version, "cuda", None)
                if cuda:
                    return f"{name} (CUDA {cuda})"
                return name
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return f"{platform.system()} {platform.machine()} (cpu)"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_benchmark(
    *,
    audio_path: Path,
    model_name: str = "tiny",
    language: str = "en",
    include_diarize: bool = False,
    hf_token: Optional[str] = None,
    hardware: Optional[str] = None,
    audio_seconds: Optional[float] = None,
    hooks: Optional[BenchHooks] = None,
    started_at: Optional[float] = None,
) -> BenchmarkReport:
    """Run the benchmark end-to-end and return a :class:`BenchmarkReport`.

    Stages (in order):

    1. ``load_whisper``       — instantiate the WhisperX wrapper.
    2. ``load_audio``         — decode the WAV to a NumPy array.
    3. ``transcribe``         — run inference on the full file.
    4. ``load_align_model``   — load the language's wav2vec2 model.
    5. ``align``              — run forced alignment on the segments.
    6. ``load_diarize``       — (optional) load the pyannote pipeline.
    7. ``run_diarize``        — (optional) run pyannote on the file.

    Stages stop at the first failure: a wedged backend doesn't produce
    useful timings for later stages. Skips (no HF token, ``include_diarize``
    False) are recorded structurally so the JSON is self-describing.

    The ``hooks`` argument exists so tests can inject stand-ins that
    return sentinel objects instead of pulling real model weights — the
    real production path uses the default :class:`BenchHooks`."""
    from scribe.engine import (
        _to_torch_device_arg,
        _torch_device,
        _whisper_device_and_compute,
        gpu_backend,
    )

    hooks = hooks or BenchHooks()

    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    if audio_seconds is None:
        audio_seconds = wav_duration_seconds(audio_path)

    backend = gpu_backend()
    w_dev, w_compute = _whisper_device_and_compute()
    t_dev = _torch_device()

    report = BenchmarkReport(
        backend=backend,
        whisper_device=w_dev,
        whisper_compute=w_compute,
        torch_device=t_dev,
        model_name=model_name,
        language=language,
        audio_path=str(audio_path),
        audio_seconds=audio_seconds,
        hardware=hardware if hardware is not None else _default_hardware_label(),
        started_at=started_at if started_at is not None else time.time(),
    )

    whisper_dev_arg = _to_torch_device_arg(w_dev)
    align_dev_arg = _to_torch_device_arg(t_dev)

    # 1. load_whisper
    secs, asr, exc = time_call(
        lambda: hooks.load_whisper(
            model_name=model_name,
            device_arg=whisper_dev_arg,
            compute=w_compute,
            language=language,
        )
    )
    if exc is not None:
        report.add(Timing("load_whisper", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("load_whisper", secs, detail=f"model={model_name}"))

    # 2. load_audio
    secs, audio, exc = time_call(lambda: hooks.load_audio(audio_path))
    if exc is not None:
        report.add(Timing("load_audio", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("load_audio", secs))

    # 3. transcribe
    secs, asr_result, exc = time_call(lambda: hooks.transcribe(asr, audio))
    if exc is not None:
        report.add(Timing("transcribe", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("transcribe", secs))

    # 4. load_align_model
    secs, align_pair, exc = time_call(
        lambda: hooks.load_align(language=language, device_arg=align_dev_arg)
    )
    if exc is not None:
        report.add(Timing("load_align_model", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("load_align_model", secs, detail=f"lang={language}"))

    # 5. align — runs the actual forced alignment
    align_model, metadata = align_pair if isinstance(align_pair, tuple) else (align_pair, None)
    segments = (asr_result or {}).get("segments", []) if isinstance(asr_result, dict) else []
    secs, _aligned, exc = time_call(
        lambda: hooks.align(
            segments=segments,
            align_model=align_model,
            metadata=metadata,
            audio=audio,
            device_arg=align_dev_arg,
        )
    )
    if exc is not None:
        report.add(Timing("align", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("align", secs))

    # 6/7 diarize (optional)
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
        report.add(Timing("load_diarize", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("load_diarize", secs))

    secs, _diar, exc = time_call(lambda: hooks.run_diarize(pipeline, audio_path))
    if exc is not None:
        report.add(Timing("run_diarize", secs, ok=False, detail=_format_exc(exc)))
        return report
    report.add(Timing("run_diarize", secs))

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.bench_rocm",
        description=(
            "In-house benchmark for the Scribe ROCm/CUDA/MPS/CPU stack (G6.2). "
            "Runs the full Whisper → align → (optional) diarize pipeline on a "
            "representative audio file, recording wall-clock and RTF for each "
            "stage. Use --compare to diff two saved reports before publishing "
            "performance numbers."
        ),
    )
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a representative audio file (WAV; other formats need ffprobe).",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model name to load (default: tiny). Use 'large-v3' for the "
        "real production-shape benchmark.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for the alignment model (default: en).",
    )
    parser.add_argument(
        "--include-diarize",
        action="store_true",
        help="Also load + run pyannote diarization. Requires HF_TOKEN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report to this path (in addition to the human-readable "
        "summary on stdout). Pair with --compare on the other box.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Override the auto-detected hardware label. Useful when two boxes "
        "have the same GPU name but different drivers.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        type=Path,
        default=None,
        help="Compare two saved JSON reports instead of running a benchmark. "
        "Prints a per-stage speedup table.",
    )
    return parser


def _audio_seconds_from_any(path: Path) -> float:
    """Pick the cheapest accurate way to measure audio duration.

    WAV → ``wave`` module header read (no ffprobe dep). Anything else
    → ``scribe.audio.probe_media_info`` (which shells out to ffprobe).
    Raises ``ValueError`` if neither path produces a usable duration."""
    if path.suffix.lower() == ".wav":
        return wav_duration_seconds(path)
    from scribe.audio import probe_media_info

    info = probe_media_info(path)
    dur = info.get("duration_seconds")
    if not isinstance(dur, (int, float)) or dur <= 0:
        raise ValueError(f"could not determine duration of {path}")
    return float(dur)


def _do_compare(baseline_path: Path, candidate_path: Path) -> int:
    """Load two saved JSON reports and print the comparison."""
    baseline = BenchmarkReport.from_dict(json.loads(baseline_path.read_text()))
    candidate = BenchmarkReport.from_dict(json.loads(candidate_path.read_text()))
    cmp = compare_reports(baseline, candidate)
    sys.stdout.write(cmp.render())
    sys.stdout.flush()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code:

    * 0 — benchmark or comparison ran without stage failure
    * 1 — at least one stage failed
    * 2 — bad CLI args (no audio + no --compare; nonexistent file; …)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.compare is not None:
        baseline, candidate = args.compare
        if not baseline.exists():
            print(f"error: baseline report not found: {baseline}", file=sys.stderr)
            return 2
        if not candidate.exists():
            print(f"error: candidate report not found: {candidate}", file=sys.stderr)
            return 2
        return _do_compare(baseline, candidate)

    if args.audio is None:
        print(
            "error: an audio path is required (or pass --compare a.json b.json)",
            file=sys.stderr,
        )
        return 2
    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    try:
        audio_seconds = _audio_seconds_from_any(args.audio)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = run_benchmark(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        include_diarize=args.include_diarize,
        hardware=args.label,
        audio_seconds=audio_seconds,
    )
    sys.stdout.write(report.render())
    sys.stdout.flush()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
