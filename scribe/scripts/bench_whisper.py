"""
G7.4 — Apple Silicon Whisper backend benchmark.

Where G6.2 (``bench_rocm``) compares the *same* Whisper backend across
*different* hardware (CUDA vs ROCm), this script compares *different*
backends on the *same* machine: faster-whisper (the historical default)
vs whisper.cpp (the G7.2 GGUF/Metal adapter). Run it on a fresh Apple
Silicon box and the script writes a small Markdown table the README can
embed:

* wall-clock per backend (``transcribe`` stage only — that's the
  number the user actually feels);
* real-time-factor (``audio_seconds / wall_seconds``);
* WER (word error rate) vs an optional reference transcript, computed
  with a self-contained Levenshtein implementation so the script has no
  external dependencies beyond what the backends themselves pull in;
* the active GPU backend label (``mps`` / ``cuda`` / ``rocm`` / ``cpu``)
  so the published baseline reads honestly.

Usage::

    # Just speed (no reference):
    .venv/bin/python -m scribe.scripts.bench_whisper sample.wav --model tiny

    # Speed + accuracy:
    .venv/bin/python -m scribe.scripts.bench_whisper sample.wav \
        --reference reference.txt --model large-v3-turbo

    # Save Markdown table for the README:
    .venv/bin/python -m scribe.scripts.bench_whisper sample.wav \
        --reference reference.txt --markdown bench.md

The pipeline stages are intentionally narrower than G6.2's: this script
only measures the inference call, because that's what the backend
choice changes. Alignment + diarization are the same across backends
and out of scope for the comparison.

Library notes:

* Backend choice is keyed by the ``scribe.whisper_backend`` registry
  ids (``faster-whisper``, ``whisper.cpp``).
* whisper.cpp's quant comes from ``--quant`` (default mirrors
  :data:`scribe.whisper_cpp.DEFAULT_QUANT`).
* The driver is hook-injectable via :class:`BenchHooks` so tests run
  the full report-shape contract without pulling a single MB of
  weights.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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

    Mirror of :func:`scribe.scripts.bench_rocm.wav_duration_seconds`.
    Header-only read so we never need ffprobe on a stripped-down host.
    Raises :class:`ValueError` for empty / malformed WAVs."""
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    if rate <= 0:
        raise ValueError(f"{path}: invalid sample rate {rate}")
    return frames / float(rate)


def _rtf(audio_seconds: float, wall_seconds: float) -> Optional[float]:
    """Real-time factor: ``audio / wall`` seconds.

    Returns ``None`` for the degenerate ``wall_seconds <= 0`` case so the
    rendered table prints ``--`` instead of crashing on division.
    """
    if wall_seconds <= 0 or audio_seconds <= 0:
        return None
    return audio_seconds / wall_seconds


def time_call(fn: Callable[[], Any]) -> tuple[float, Any, Optional[BaseException]]:
    """Run ``fn`` and return ``(seconds, result_or_None, exception_or_None)``.

    Catches ``Exception`` so KeyboardInterrupt still aborts cleanly.
    Same semantics as :func:`scribe.scripts.bench_rocm.time_call`."""
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, None, exc
    return time.perf_counter() - t0, result, None


def _format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Word error rate
# --------------------------------------------------------------------------- #


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalise_text(text: str) -> list[str]:
    """Lowercase + word-tokenise ``text`` for WER comparison.

    The default jiwer / sclite pipeline strips punctuation, lowercases,
    and collapses whitespace. We mirror that here so the WER number is
    comparable to the published numbers in upstream benchmarks. Pure —
    no external dep."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def levenshtein(a: list[str], b: list[str]) -> int:
    """Standard token-level edit distance.

    Self-contained so the script doesn't pull jiwer / nltk just to run
    one comparison. ``O(|a| * |b|)`` time, ``O(|b|)`` space using a
    rolling row. Empty inputs are handled at the boundaries."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """WER = edit distance / reference length, in [0, 1+] (no upper cap).

    Returns ``None`` for an empty reference (the metric is undefined).
    Higher = worse. Matches the conventional formula used in WhisperX /
    OpenAI-Whisper papers."""
    ref = normalise_text(reference)
    hyp = normalise_text(hypothesis)
    if not ref:
        return None
    return levenshtein(ref, hyp) / float(len(ref))


def hypothesis_text_from_segments(segments: list[dict[str, Any]] | tuple) -> str:
    """Glue Whisper segment ``text`` fields into a single string.

    Both faster-whisper and whisper.cpp produce ``segments`` with a
    ``text`` field (possibly with leading whitespace); we strip and
    join with single spaces so WER is computed on the same shape
    regardless of which backend produced the transcript."""
    if not segments:
        return ""
    parts = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        t = seg.get("text")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Per-backend row + full report dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class BackendTiming:
    """One backend's measurement (faster-whisper or whisper.cpp).

    ``ok=False`` is recorded with the exception summary in ``detail``
    so the report can show partial runs (e.g. faster-whisper succeeded
    but whisper.cpp failed because pywhispercpp isn't installed)
    instead of dropping the whole comparison."""

    backend_id: str
    seconds: float = 0.0
    ok: bool = False
    detail: str = ""
    hypothesis: str = ""
    wer: Optional[float] = None

    def render(self, audio_seconds: float) -> str:
        status = "OK  " if self.ok else "FAIL"
        rtf = _rtf(audio_seconds, self.seconds)
        rtf_str = f"{rtf:6.2f}× RT" if rtf is not None else "  --   "
        wer_str = (
            f"WER={self.wer * 100:5.1f}%"
            if self.wer is not None
            else "WER=  --  "
        )
        line = (
            f"[{status}] {self.backend_id:<16s} {self.seconds:7.2f}s  "
            f"{rtf_str}  {wer_str}"
        )
        if self.detail:
            line = f"{line}  {self.detail}"
        return line


@dataclass
class WhisperBenchmarkReport:
    """Full result of a whisper-backend comparison run.

    JSON-serialisable so two runs can be diffed offline. The Markdown
    renderer is what the README embeds."""

    audio_path: str
    audio_seconds: float
    model_name: str
    language: str
    gpu_backend: str = ""
    hardware: str = ""
    reference: str = ""
    started_at: float = 0.0
    rows: list[BackendTiming] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, row: BackendTiming) -> None:
        self.rows.append(row)

    def skip(self, backend_id: str, reason: str) -> None:
        self.skipped.append(f"{backend_id}: {reason}")

    @property
    def ok(self) -> bool:
        """A report is ``ok`` only when at least one backend ran *and*
        every backend that ran succeeded. An all-failed run is not a
        green light — there's nothing comparable to publish."""
        return bool(self.rows) and all(r.ok for r in self.rows)

    def row(self, backend_id: str) -> Optional[BackendTiming]:
        for r in self.rows:
            if r.backend_id == backend_id:
                return r
        return None

    @property
    def fastest_backend_id(self) -> Optional[str]:
        """Backend id of the fastest *successful* row, or ``None``."""
        ok_rows = [r for r in self.rows if r.ok and r.seconds > 0]
        if not ok_rows:
            return None
        return min(ok_rows, key=lambda r: r.seconds).backend_id

    def speedup_over(self, baseline_id: str, candidate_id: str) -> Optional[float]:
        """``baseline_seconds / candidate_seconds`` — > 1 ⇒ candidate
        faster. Returns ``None`` if either row is missing or failed."""
        b = self.row(baseline_id)
        c = self.row(candidate_id)
        if b is None or c is None:
            return None
        if not b.ok or not c.ok:
            return None
        if c.seconds <= 0:
            return None
        return b.seconds / c.seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhisperBenchmarkReport":
        rows = [BackendTiming(**r) for r in data.get("rows", [])]
        return cls(
            audio_path=data["audio_path"],
            audio_seconds=float(data["audio_seconds"]),
            model_name=data["model_name"],
            language=data["language"],
            gpu_backend=data.get("gpu_backend", ""),
            hardware=data.get("hardware", ""),
            reference=data.get("reference", ""),
            started_at=float(data.get("started_at", 0.0)),
            rows=rows,
            skipped=list(data.get("skipped", [])),
        )

    def render(self) -> str:
        lines: list[str] = []
        lines.append("Scribe whisper-backend benchmark (G7.4)")
        lines.append("=" * 56)
        if self.hardware:
            lines.append(f"Hardware:        {self.hardware}")
        if self.gpu_backend:
            lines.append(f"GPU backend:     {self.gpu_backend}")
        lines.append(f"Model:           {self.model_name}  lang={self.language}")
        lines.append(
            f"Audio:           {self.audio_path}  ({self.audio_seconds:.2f}s)"
        )
        if self.reference:
            preview = self.reference.strip().replace("\n", " ")
            if len(preview) > 50:
                preview = preview[:47] + "..."
            lines.append(f"Reference:       {preview}")
        lines.append("")
        if not self.rows:
            lines.append("(no backends ran)")
        else:
            for r in self.rows:
                lines.append(r.render(self.audio_seconds))
        if self.skipped:
            lines.append("")
            lines.append("Skipped:")
            for s in self.skipped:
                lines.append(f"  - {s}")
        return "\n".join(lines) + "\n"

    def render_markdown(self) -> str:
        """Markdown table the README embeds.

        Columns: backend / wall-clock / RTF / WER. The header carries
        the hardware + model so a copied snippet stands alone. Failed
        rows are still rendered (with ``FAIL`` in the wall-clock cell
        and the exception summary in a footnote) — burying a failure
        looks like cherry-picking; surfacing it is what we want.
        """
        lines: list[str] = []
        title = f"Whisper backend benchmark — {self.model_name}"
        if self.hardware:
            title = f"{title} on {self.hardware}"
        lines.append(f"### {title}")
        lines.append("")
        meta_bits: list[str] = []
        if self.gpu_backend:
            meta_bits.append(f"GPU backend: `{self.gpu_backend}`")
        meta_bits.append(f"Audio: `{self.audio_path}` ({self.audio_seconds:.1f}s)")
        meta_bits.append(f"Language: `{self.language}`")
        lines.append(" · ".join(meta_bits))
        lines.append("")
        lines.append("| Backend | Wall-clock | RTF | WER |")
        lines.append("| --- | --- | --- | --- |")
        notes: list[str] = []
        for r in self.rows:
            wall = "FAIL" if not r.ok else f"{r.seconds:.2f}s"
            rtf = _rtf(self.audio_seconds, r.seconds)
            rtf_str = f"{rtf:.2f}×" if rtf is not None and r.ok else "—"
            wer_str = (
                f"{r.wer * 100:.1f}%" if r.wer is not None else "—"
            )
            lines.append(
                f"| `{r.backend_id}` | {wall} | {rtf_str} | {wer_str} |"
            )
            if not r.ok and r.detail:
                notes.append(f"`{r.backend_id}`: {r.detail}")
        if notes:
            lines.append("")
            for note in notes:
                lines.append(f"> {note}")
        if self.skipped:
            lines.append("")
            for s in self.skipped:
                lines.append(f"> Skipped — {s}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Stage hooks — real implementations, swappable in tests
# --------------------------------------------------------------------------- #


def _real_run_backend(
    backend_id: str,
    audio_path: Path,
    *,
    model_name: str,
    language: str,
    asr_options: dict[str, Any],
) -> dict[str, Any]:
    """Default driver — dispatch to the registered backend.

    Pure adapter: no torch import here, every model load happens
    inside the backend's ``transcribe`` call. Tests inject a fake
    callable via :class:`BenchHooks` to skip the real inference."""
    from scribe.whisper_backend import get_backend, ProgressFn

    backend = get_backend(backend_id)

    def _noop(stage: str, value: float) -> None:
        return

    progress: ProgressFn = _noop
    return backend.transcribe(
        audio_path,
        model_name=model_name,
        language=language,
        asr_options=dict(asr_options),
        vad_options={},
        progress=progress,
    )


@dataclass
class BenchHooks:
    """Injectable bundle of stage drivers.

    The single hook ``run_backend(backend_id, audio_path, ...)`` is the
    only thing tests need to swap out — every other stage is pure
    bookkeeping. Keeping the interface this small means the test
    surface stays small too."""

    run_backend: Callable[..., dict[str, Any]] = _real_run_backend


# --------------------------------------------------------------------------- #
# Hardware label
# --------------------------------------------------------------------------- #


def _default_hardware_label() -> str:
    """Best-effort one-line description of the current GPU.

    Mirror of :func:`scribe.scripts.bench_rocm._default_hardware_label`
    so the report shape stays consistent. Imported lazily so the module
    is still importable on a stripped-down test box."""
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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return f"{platform.system()} {platform.machine()} (Apple Silicon / MPS)"
    except Exception:  # noqa: BLE001
        pass
    return f"{platform.system()} {platform.machine()} (cpu)"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


DEFAULT_BACKENDS: tuple[str, ...] = ("faster-whisper", "whisper.cpp")


def run_whisper_benchmark(
    *,
    audio_path: Path,
    model_name: str = "tiny",
    language: str = "en",
    backends: tuple[str, ...] | list[str] = DEFAULT_BACKENDS,
    quant: Optional[str] = None,
    reference: Optional[str] = None,
    hardware: Optional[str] = None,
    audio_seconds: Optional[float] = None,
    hooks: Optional[BenchHooks] = None,
    gpu_backend_label: Optional[str] = None,
    started_at: Optional[float] = None,
) -> WhisperBenchmarkReport:
    """Run every requested backend on the same audio and return a
    :class:`WhisperBenchmarkReport`.

    For each backend id in ``backends`` (default: faster-whisper +
    whisper.cpp), the driver:

    1. Calls ``hooks.run_backend(backend_id, audio_path, ...)`` and
       wall-clocks the call.
    2. Captures the returned ``segments`` and computes a
       hypothesis-text glue.
    3. Computes WER vs ``reference`` if one was provided.
    4. Records a :class:`BackendTiming` row.

    Failures in one backend don't stop the others — that's the whole
    point of the comparison. ``run_whisper_benchmark`` never raises for
    backend-side errors; those land as ``ok=False`` rows."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    hooks = hooks or BenchHooks()

    if audio_seconds is None:
        audio_seconds = wav_duration_seconds(audio_path)

    if gpu_backend_label is None:
        try:
            from scribe.engine import gpu_backend
            gpu_backend_label = gpu_backend()
        except Exception:  # noqa: BLE001
            gpu_backend_label = ""

    report = WhisperBenchmarkReport(
        audio_path=str(audio_path),
        audio_seconds=audio_seconds,
        model_name=model_name,
        language=language,
        gpu_backend=gpu_backend_label or "",
        hardware=hardware if hardware is not None else _default_hardware_label(),
        reference=reference or "",
        started_at=started_at if started_at is not None else time.time(),
    )

    asr_options: dict[str, Any] = {}
    if quant:
        # whisper.cpp takes the quant via asr_options; faster-whisper
        # ignores it. Pass on both rather than maintain a per-backend
        # asr_options dict — the backends themselves are the right place
        # to filter.
        asr_options["whisper_cpp_quant"] = quant

    for backend_id in backends:
        secs, result, exc = time_call(
            lambda bid=backend_id: hooks.run_backend(
                bid,
                audio_path,
                model_name=model_name,
                language=language,
                asr_options=dict(asr_options),
            )
        )
        if exc is not None:
            report.add(
                BackendTiming(
                    backend_id=backend_id,
                    seconds=secs,
                    ok=False,
                    detail=_format_exc(exc),
                )
            )
            continue
        segments = (result or {}).get("segments", []) if isinstance(result, dict) else []
        hyp = hypothesis_text_from_segments(segments)
        wer = word_error_rate(reference, hyp) if reference else None
        report.add(
            BackendTiming(
                backend_id=backend_id,
                seconds=secs,
                ok=True,
                hypothesis=hyp,
                wer=wer,
            )
        )

    return report


# --------------------------------------------------------------------------- #
# G7.4 — Read-only "what does this benchmark do?" plan surface.
#
# Mirrors the G6.1 / G6.2 ``*_plan()`` pattern. The CLI itself is the
# user-facing surface for actually *running* the benchmark, but a user
# wanting to discover the invocation needs an in-app surface.
# --------------------------------------------------------------------------- #


# Ordered list of backends the benchmark exercises — matches
# ``DEFAULT_BACKENDS`` and the registry ids in
# :mod:`scribe.whisper_backend`. The optional flag is part of the
# contract so the panel can render whisper.cpp with an "install
# pywhispercpp first" hint when it isn't available.
WHISPER_BENCHMARK_BACKENDS: tuple[dict[str, Any], ...] = (
    {
        "id": "faster-whisper",
        "summary": (
            "Default CTranslate2 backend. CPU + int8 fallback on Apple "
            "Silicon (no Metal). Acts as the benchmark baseline."
        ),
        "optional": False,
    },
    {
        "id": "whisper.cpp",
        "summary": (
            "GGUF weights with CPU + Metal + Vulkan. ~5× faster than "
            "faster-whisper on Apple Silicon (the reason G7.x exists)."
        ),
        "optional": True,
    },
)


# Process exit codes returned by ``main()``. Surfaced so a CI wrapper
# can rely on them being part of the contract.
WHISPER_BENCHMARK_EXIT_CODES: tuple[dict[str, Any], ...] = (
    {"code": 0, "meaning": "healthy — every requested backend produced a transcript"},
    {"code": 1, "meaning": "backend_failure — at least one backend failed (other rows still recorded)"},
    {"code": 2, "meaning": "bad_cli_args — missing audio path or unreadable reference file"},
)


# CLI mode descriptions. Three modes: a single-box benchmark run, a
# "speed only" mode that omits the reference (fastest path), and a
# "speed + WER" mode that requires --reference and writes a Markdown
# table the README can embed.
WHISPER_BENCHMARK_MODES: tuple[dict[str, Any], ...] = (
    {
        "name": "speed",
        "summary": "Wall-clock + RTF only (no reference transcript needed).",
        "cli": "python -m scribe.scripts.bench_whisper <audio> [--model …]",
        "cli_venv": ".venv/bin/python -m scribe.scripts.bench_whisper <audio> [--model …]",
    },
    {
        "name": "accuracy",
        "summary": "Speed + WER vs --reference (publishable headline number).",
        "cli": "python -m scribe.scripts.bench_whisper <audio> --reference ref.txt",
        "cli_venv": ".venv/bin/python -m scribe.scripts.bench_whisper <audio> --reference ref.txt",
    },
    {
        "name": "markdown",
        "summary": "Write a Markdown table for the README (combine with --reference).",
        "cli": "python -m scribe.scripts.bench_whisper <audio> --reference ref.txt --markdown bench.md",
        "cli_venv": ".venv/bin/python -m scribe.scripts.bench_whisper <audio> --reference ref.txt --markdown bench.md",
    },
)


# Defaults baked into ``build_parser``. Restated here as data so the
# home-page panel doesn't have to re-import argparse. The
# ``test_defaults_match_argparse`` test pins these against the parser.
WHISPER_BENCHMARK_DEFAULTS: dict[str, Any] = {
    "model": "tiny",
    "language": "en",
    "quant": None,
    "backends": list(DEFAULT_BACKENDS),
    "reference": None,
    "output": None,
    "markdown": None,
    "label": None,
}


WHISPER_BENCHMARK_CLI = "python -m scribe.scripts.bench_whisper"
WHISPER_BENCHMARK_CLI_VENV = ".venv/bin/python -m scribe.scripts.bench_whisper"


def whisper_benchmark_plan() -> dict[str, Any]:
    """Return a structured description of what the benchmark does.

    No model loads, no I/O, no torch import — pure metadata. The
    FastAPI route ``GET /api/diagnostics/whisper-benchmark-plan``
    returns this dict as JSON; the home-page template renders the
    same fields into a panel so a user knows what the CLI invocation
    will run before running it.

    Keys:

    * ``feature_id`` — ``"G7.4"`` so JS test-id selectors can pin the
      panel.
    * ``cli`` / ``cli_venv`` — copy-paste invocation strings.
    * ``backends`` — ordered list of ``{id, summary, optional}`` dicts
      mirroring :data:`DEFAULT_BACKENDS`.
    * ``defaults`` — argparse defaults (model / language / quant /
      backends / reference / output / markdown / label).
    * ``exit_codes`` — ``[{code, meaning}, ...]`` triples.
    * ``modes`` — ``speed`` / ``accuracy`` / ``markdown`` invocation
      shapes.
    * ``fail_isolated`` — ``True``: a failure in one backend doesn't
      stop the others; this is the whole point of the comparison.
    * ``docs_anchor`` — README section heading for the in-app docs
      viewer.
    * ``metric`` — ``"WER"`` so a JS / curl client knows the headline
      column without re-reading the spec.
    """
    return {
        "feature_id": "G7.4",
        "cli": WHISPER_BENCHMARK_CLI,
        "cli_venv": WHISPER_BENCHMARK_CLI_VENV,
        "backends": [dict(b) for b in WHISPER_BENCHMARK_BACKENDS],
        "defaults": dict(WHISPER_BENCHMARK_DEFAULTS),
        "exit_codes": [dict(ec) for ec in WHISPER_BENCHMARK_EXIT_CODES],
        "modes": [dict(mode) for mode in WHISPER_BENCHMARK_MODES],
        "fail_isolated": True,
        "metric": "WER",
        "docs_anchor": "apple-silicon-gpu-whisper-cpp",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.bench_whisper",
        description=(
            "Benchmark Scribe's Whisper inference backends against each "
            "other on the same audio (G7.4). Reports wall-clock, real-"
            "time-factor, and (optionally) word error rate vs a "
            "reference transcript. Writes a Markdown table the README "
            "can embed."
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
        help="Whisper model name (default: tiny). Use 'large-v3-turbo' for the "
        "real production-shape benchmark on Apple Silicon.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en).",
    )
    parser.add_argument(
        "--quant",
        default=None,
        help="whisper.cpp quant override (e.g. q5_0 / q8_0 / f16). "
        "Ignored by faster-whisper.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend id to include. Repeatable. Defaults to "
        "faster-whisper + whisper.cpp.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to a plain-text reference transcript. When set, WER "
        "is computed and surfaced in the report and the Markdown table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report to this path (in addition to the "
        "human-readable summary on stdout).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Write a Markdown table to this path (the README embeds it).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Override the auto-detected hardware label.",
    )
    return parser


def _audio_seconds_from_any(path: Path) -> float:
    """Pick the cheapest accurate way to measure audio duration."""
    if path.suffix.lower() == ".wav":
        return wav_duration_seconds(path)
    from scribe.audio import probe_media_info

    info = probe_media_info(path)
    dur = info.get("duration_seconds")
    if not isinstance(dur, (int, float)) or dur <= 0:
        raise ValueError(f"could not determine duration of {path}")
    return float(dur)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code:

    * 0 — every requested backend produced a transcript
    * 1 — at least one backend failed (other rows still recorded)
    * 2 — bad CLI args (no audio; nonexistent audio / reference file)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.audio is None:
        print("error: an audio path is required", file=sys.stderr)
        return 2
    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    reference_text: Optional[str] = None
    if args.reference is not None:
        if not args.reference.exists():
            print(
                f"error: reference file not found: {args.reference}",
                file=sys.stderr,
            )
            return 2
        reference_text = args.reference.read_text(encoding="utf-8")

    try:
        audio_seconds = _audio_seconds_from_any(args.audio)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = tuple(args.backend) if args.backend else DEFAULT_BACKENDS

    report = run_whisper_benchmark(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        backends=backends,
        quant=args.quant,
        reference=reference_text,
        hardware=args.label,
        audio_seconds=audio_seconds,
    )
    sys.stdout.write(report.render())
    sys.stdout.flush()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report.render_markdown())

    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
