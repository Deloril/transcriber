"""
Transcription engine.

Strategy
--------
- Whisper (faster-whisper / CTranslate2) device:
    - CUDA → "cuda" + float16 (or int8_float16 on small-VRAM cards)
    - Apple Silicon / CPU → "cpu" + int8 (CTranslate2 has no MPS backend;
      on M-series CPUs with Accelerate it's fast enough for offline batch
      use, and accuracy is identical to GPU).
- Forced alignment (wav2vec2) uses torch — CUDA > MPS > CPU.
- Diarization (pyannote) uses torch — CUDA > MPS > CPU. (pyannote sometimes
  has rough edges on MPS; falls back to CPU automatically if needed.)

Two modes:
  - "multi-track": each audio track is one speaker. Transcribe each track
    independently, label segments by track, merge on the timeline. No AI
    diarization required — labels are perfect.
  - "diarize":     single mixed track. Run pyannote diarization, run WhisperX
    with alignment, then assign speakers to words.

Environment overrides (advanced):
  SCRIBE_DEVICE        cuda | mps | cpu       (force torch device)
  SCRIBE_WHISPER_DEVICE cuda | cpu            (force CTranslate2 device)
  SCRIBE_COMPUTE_TYPE  float16 | int8_float16 | int8 | float32
"""

from __future__ import annotations

import gc
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from .audio import AudioStream, extract_track_to_wav, probe_audio_streams

ProgressFn = Callable[[str, float], None]
"""Callback: (message, fraction_in_0_to_1)."""


def _noop_progress(_msg: str, _f: float) -> None:
    return None


@dataclass
class AdvancedOptions:
    """User-tunable knobs that map onto faster-whisper / WhisperX VAD parameters."""

    # Decoding
    beam_size: int = 5                  # 1 disables beam search; higher = slower, more accurate
    best_of: int = 5                    # candidates considered when temperature > 0
    temperature: float = 0.0            # 0 = deterministic; tuple-like fallback handled in caller

    # Hallucination guards
    no_speech_threshold: float = 0.45   # higher = more aggressive about skipping silence
    compression_ratio_threshold: float = 2.4  # gzip ratio above this → segment dropped (loop guard)
    condition_on_previous_text: bool = False  # leave off; main source of repeat-loop hallucinations

    # VAD / chunking — these are the long-monologue knobs
    chunk_size: int = 30                # max seconds per chunk before hard cut
    vad_onset: float = 0.500            # speech start threshold
    vad_offset: float = 0.363           # speech end threshold

    # Domain biasing
    initial_prompt: str = ""            # free text prepended to model context
    hotwords: str = ""                  # comma- or space-separated bias words

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AdvancedOptions":
        if not d:
            return cls()
        kwargs: dict[str, Any] = {}
        for f in (
            "beam_size", "best_of", "temperature",
            "no_speech_threshold", "compression_ratio_threshold", "condition_on_previous_text",
            "chunk_size", "vad_onset", "vad_offset",
            "initial_prompt", "hotwords",
        ):
            if f in d and d[f] is not None and d[f] != "":
                kwargs[f] = d[f]
        return cls(**kwargs)

    def asr_options(self) -> dict[str, Any]:
        """Subset to pass to faster-whisper's load_model asr_options."""
        # faster-whisper expects "temperatures" as a tuple. The default fallback
        # ladder it ships with is (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) — we replace it
        # with a single value so the user sees the exact temperature they asked
        # for, no surprise fallback decoding.
        return {
            "beam_size": int(self.beam_size),
            "best_of": int(self.best_of),
            "temperatures": (float(self.temperature),),
            "no_speech_threshold": float(self.no_speech_threshold),
            "compression_ratio_threshold": float(self.compression_ratio_threshold),
            "condition_on_previous_text": bool(self.condition_on_previous_text),
            "initial_prompt": (self.initial_prompt or None),
            "hotwords": (self.hotwords or None),
        }

    def vad_options(self) -> dict[str, Any]:
        return {
            "chunk_size": int(self.chunk_size),
            "vad_onset": float(self.vad_onset),
            "vad_offset": float(self.vad_offset),
        }


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "score": self.score,
        }


@dataclass
class Segment:
    text: str
    start: float
    end: float
    speaker: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str
    mode: Literal["multi-track", "diarize"]
    speaker_labels: list[str]
    audio_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "mode": self.mode,
            "speakers": self.speaker_labels,
            "segments": [s.to_dict() for s in self.segments],
        }


# --------------------------------------------------------------------------- #
# PyTorch 2.6+ checkpoint loading
# --------------------------------------------------------------------------- #
# In PyTorch 2.6, torch.load() flipped its default to weights_only=True. The
# pyannote and whisperx checkpoints pickle a handful of config containers
# (omegaconf ListConfig/DictConfig, numpy reconstructors, etc) that the strict
# unpickler rejects. We allowlist a closed set of known-safe globals using
# torch.serialization.add_safe_globals — that keeps the strict mode for
# everything else but lets the model files we explicitly fetch from
# HuggingFace deserialize.

_SAFE_GLOBALS_REGISTERED = False


def _register_safe_globals() -> None:
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    _SAFE_GLOBALS_REGISTERED = True
    if not hasattr(torch.serialization, "add_safe_globals"):
        return  # PyTorch < 2.6, nothing to do
    safe: list[Any] = []
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Metadata
        safe.extend([ListConfig, DictConfig, ContainerMetadata, Metadata])
    except Exception:
        pass
    try:
        from omegaconf.nodes import AnyNode
        safe.append(AnyNode)
    except Exception:
        pass
    try:
        import collections
        safe.extend([collections.OrderedDict, collections.defaultdict])
    except Exception:
        pass
    try:
        import numpy as np
        safe.extend([
            np.ndarray, np.dtype,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.bool_, np.bytes_, np.str_,
        ])
        from numpy.core.multiarray import _reconstruct, scalar  # type: ignore
        safe.extend([_reconstruct, scalar])
    except Exception:
        pass
    try:
        from pyannote.audio.core.task import Specifications  # type: ignore
        safe.append(Specifications)
    except Exception:
        pass
    try:
        # typing.Any and a few stdlib types pyannote/lightning checkpoints reach for.
        import typing
        safe.append(typing.Any)
    except Exception:
        pass
    try:
        import builtins
        safe.extend([slice, range, complex, bytes, set, frozenset, dict, list, tuple])
    except Exception:
        pass
    try:
        torch.serialization.add_safe_globals(safe)
    except Exception as e:  # noqa: BLE001
        print(f"[scribe] add_safe_globals failed (continuing): {e}")


# Register at import so anything that calls torch.load through whisperx or
# pyannote downstream gets the allowlist applied before the first checkpoint
# load.
_register_safe_globals()


def _shim_hf_hub_download() -> None:
    """
    huggingface_hub 1.0 removed the `use_auth_token` kwarg in favour of `token`,
    but pyannote-audio 3.4 still passes the old name. requirements.txt pins
    huggingface_hub<1.0 to avoid this, but if a user's environment ends up on
    1.x anyway (e.g. another package upgraded it transitively), we translate
    the old kwarg into the new one transparently. Belt-and-braces.
    """
    try:
        import huggingface_hub as _hf
    except Exception:
        return
    if not hasattr(_hf, "hf_hub_download"):
        return
    if getattr(_hf.hf_hub_download, "_scribe_shimmed", False):
        return
    _orig = _hf.hf_hub_download

    def _shimmed(*args: Any, **kwargs: Any):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return _orig(*args, **kwargs)

    _shimmed._scribe_shimmed = True  # type: ignore[attr-defined]
    _hf.hf_hub_download = _shimmed  # type: ignore[assignment]
    # Some callers do `from huggingface_hub import hf_hub_download` before our
    # shim runs; we can't fix those captured references retroactively, but
    # patching the module attribute covers the common late-import path.


_shim_hf_hub_download()


# Belt-and-braces: pyannote's checkpoint loader trips the strict loader on
# globals we can't fully enumerate (lightning hyperparameters with arbitrary
# user types). Pyannote also calls torch.load(..., weights_only=True)
# *explicitly* in some paths, so a polite "respect the caller" wrapper isn't
# enough — we have to force-override. Opt out via SCRIBE_STRICT_TORCH_LOAD=1.
if (
    os.environ.get("SCRIBE_STRICT_TORCH_LOAD", "").strip() not in {"1", "true", "True"}
    and not getattr(torch.load, "_scribe_patched", False)
):
    _orig_torch_load = torch.load

    def _scribe_torch_load(*args: Any, **kwargs: Any):
        # Force the legacy load path regardless of what the caller passed.
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _scribe_torch_load._scribe_patched = True  # type: ignore[attr-defined]
    torch.load = _scribe_torch_load  # type: ignore[assignment]
    try:
        # Modules that imported `from torch.serialization import load` before
        # our patch landed will still hold the original reference. Patching
        # the attribute here covers any later `import torch.serialization` usage.
        import torch.serialization as _ts
        _ts.load = _scribe_torch_load  # type: ignore[assignment]
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
#
# Backend taxonomy:
#   "cuda" — NVIDIA CUDA via the official PyTorch wheel
#   "rocm" — AMD ROCm via the pytorch-rocm wheel. PyTorch's ROCm build aliases
#            torch.cuda.* to HIP, so torch.cuda.is_available() returns True on
#            both — torch.version.hip is the discriminator. CTranslate2's
#            ROCm wheel (v4.7.0+) takes device="cuda" too via its HIP shim.
#   "mps"  — Apple Silicon Metal Performance Shaders
#   "cpu"  — fallback


def gpu_backend() -> str:
    """
    Canonical four-state backend label. Cheap; safe to call repeatedly.
    Honours SCRIBE_DEVICE for testing/forcing.
    """
    forced = os.environ.get("SCRIBE_DEVICE", "").strip().lower()
    if forced in {"cuda", "rocm", "mps", "cpu"}:
        return forced
    if torch.cuda.is_available():
        # ROCm wheels populate both torch.version.cuda (compat string) and
        # torch.version.hip; CUDA wheels leave torch.version.hip as None.
        if getattr(torch.version, "hip", None):
            return "rocm"
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gpu_device_name() -> str:
    """Best-effort GPU model name; safe to call when no GPU."""
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(0) or ""
    except Exception:
        return ""


def _is_rdna2() -> bool:
    """Detect AMD RDNA 2 (RX 6000-series) via device name. Used to apply the
    CT2_CUDA_ALLOCATOR=cub_caching workaround (CT2 issue #2012)."""
    name = _gpu_device_name().lower()
    return any(tag in name for tag in (
        "rx 6", "radeon rx 6", "navi 21", "navi 22", "navi 23", "navi 24",
        "gfx1030", "gfx1031", "gfx1032", "gfx1034",
    ))


def _torch_device() -> str:
    """
    PyTorch device string for alignment / pyannote. Note that under ROCm,
    PyTorch *takes* "cuda" as the device label (its HIP backend aliases the
    CUDA namespace), so we translate "rocm" → "cuda" at this boundary.
    """
    backend = gpu_backend()
    if backend == "rocm":
        return "cuda"   # PyTorch ROCm uses the cuda namespace
    return backend


def _cuda_vram_gb() -> float:
    """VRAM of device 0 in GiB. Works for both CUDA and ROCm because
    torch.cuda.* is HIP-aliased on the ROCm wheel."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _diarization_device() -> str:
    """
    pyannote on CUDA/ROCm/CPU. Default to GPU when present, CPU otherwise.
    On MPS some ops still fall back to CPU and the partial-MPS path is
    slower than plain CPU, so we skip MPS unless explicitly forced.
    """
    forced = os.environ.get("SCRIBE_DIARIZE_DEVICE", "").strip().lower()
    if forced in {"cuda", "rocm", "mps", "cpu"}:
        # Translate rocm → cuda for the actual torch device string.
        return "cuda" if forced == "rocm" else forced
    backend = gpu_backend()
    if backend in ("cuda", "rocm"):
        return "cuda"
    return "cpu"


def _whisper_device_and_compute() -> tuple[str, str]:
    """
    faster-whisper device + compute type.

    CTranslate2 has no MPS backend, but on its ROCm wheel (v4.7.0+) the
    device flag is still "cuda" — same code path as NVIDIA. We auto-pick
    int8_float16 on smaller VRAM cards to keep large-v3 comfortably under
    the budget.
    """
    forced_device = os.environ.get("SCRIBE_WHISPER_DEVICE", "").strip().lower()
    forced_compute = os.environ.get("SCRIBE_COMPUTE_TYPE", "").strip().lower()

    backend = gpu_backend()
    if forced_device in {"cuda", "rocm", "cpu"}:
        device = "cuda" if forced_device in {"cuda", "rocm"} else "cpu"
    elif backend in ("cuda", "rocm"):
        device = "cuda"
    else:
        device = "cpu"

    if forced_compute:
        return device, forced_compute

    if device == "cuda":
        # large-v3 fp16 wants ~6 GB; int8_float16 brings it under ~4 GB.
        return device, "int8_float16" if _cuda_vram_gb() < 8 else "float16"
    return device, "int8"


def _apply_rocm_runtime_workarounds() -> None:
    """
    AMD-specific environment fixes that have to be set before CT2 / pyannote
    load any models. Idempotent and safe to call on non-AMD machines.

    - RDNA 2 (gfx103x): CT2's default MallocAsync allocator crashes on these
      cards with "illegal memory access." Switch to cub_caching as documented
      in CT2 issue #2012.
    """
    if gpu_backend() != "rocm":
        return
    if _is_rdna2() and not os.environ.get("CT2_CUDA_ALLOCATOR"):
        os.environ["CT2_CUDA_ALLOCATOR"] = "cub_caching"


_apply_rocm_runtime_workarounds()


def _patch_pyannote_lstm_dropout(pipeline: Any) -> None:
    """
    pyannote-audio 3.4's segmentation model uses nn.LSTM(dropout=0.5,...).
    On ROCm ≥ 6.1.1, MIOpen can't compile the dropout kernel because the
    hiprand_xorwow.h header was removed (pyannote-audio issue #1995).
    Workaround: force dropout=0.0 after loading. Inference behaviour is
    unchanged (dropout is a no-op outside training).

    Idempotent. No-op on non-ROCm machines.
    """
    if gpu_backend() != "rocm":
        return
    try:
        import torch.nn as nn
        for module in pipeline.modules() if hasattr(pipeline, "modules") else []:
            if isinstance(module, nn.LSTM) and module.dropout:
                module.dropout = 0.0
    except Exception as e:  # noqa: BLE001
        print(f"[scribe] could not patch pyannote LSTM dropout: {e}")


# --------------------------------------------------------------------------- #
# WhisperX wrappers
# --------------------------------------------------------------------------- #


def _load_whisperx():
    """Lazy import — whisperx pulls in heavy deps."""
    import whisperx  # type: ignore
    return whisperx


class _ProgressCapture:
    """
    File-like object that pretends to be stdout. WhisperX prints progress as
    'Progress: 42.31%...' when its `print_progress=True` flag is set; we
    intercept that, parse the percent, and forward it to our own callback —
    while still letting any other prints through to the real stdout so we
    don't swallow useful diagnostics.
    """

    _re = re.compile(r"Progress:\s*([\d.]+)\s*%")

    def __init__(self, label: str, on_pct: Callable[[str, float], None]) -> None:
        self.label = label
        self.on_pct = on_pct
        self._buf = ""
        self._real = sys.stdout

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        # process complete lines
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = self._re.search(line)
            if m:
                try:
                    pct = float(m.group(1)) / 100.0
                except ValueError:
                    pct = None
                if pct is not None:
                    self.on_pct(self.label, max(0.0, min(1.0, pct)))
                continue
            # not a progress line — let it through to the real stdout
            self._real.write(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._real.write(self._buf)
            self._buf = ""
        self._real.flush()

    def __enter__(self) -> "_ProgressCapture":
        sys.stdout = self
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.flush()
        finally:
            sys.stdout = self._real


def _safe_load_model(whisperx, *, model_name, device, compute_type, language, asr_options, vad_options):
    """
    Call whisperx.load_model defensively — older/newer versions accept different
    keyword arguments and silently complaining beats crashing the whole job.
    """
    import inspect
    sig = inspect.signature(whisperx.load_model)
    kw: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "language": language if language != "auto" else None,
    }
    if "asr_options" in sig.parameters:
        kw["asr_options"] = asr_options
    if "vad_options" in sig.parameters:
        kw["vad_options"] = vad_options
    elif "vad_method" in sig.parameters:
        # newer signature; pass-through anyway, vad_options separately if accepted
        pass
    try:
        return whisperx.load_model(model_name, **kw)
    except TypeError:
        # Strip any keys CTranslate2 / faster-whisper rejected and retry once.
        clean_asr = {k: v for k, v in asr_options.items() if v is not None and k != "hotwords"}
        kw["asr_options"] = clean_asr
        return whisperx.load_model(model_name, **kw)


def _transcribe_with_alignment(
    audio_path: Path,
    *,
    model_name: str,
    language: str,
    batch_size: int,
    options: "AdvancedOptions",
    progress: ProgressFn,
    progress_base: float,
    progress_span: float,
    hf_token: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Run transcription (Whisper or Parakeet) + word-level alignment on a single
    audio file. Returns (aligned_segments, detected_language).
    """
    whisperx = _load_whisperx()

    # Parakeet path: NeMo TDT model, English only, native transcription.
    # Returns the same {"start", "end", "text"} shape Whisper does, so the
    # alignment+diarization tail of this function works unchanged.
    from .parakeet import is_parakeet_model
    if is_parakeet_model(model_name):
        from .parakeet import transcribe_with_parakeet
        # Reserve [0.00..0.55] for transcription, leave [0.55..1.00] for
        # the shared alignment phase below.
        seg_dicts, detected_lang = transcribe_with_parakeet(
            audio_path,
            model_name=model_name,
            hf_token=hf_token,
            options=options,
            progress=progress,
            progress_base=progress_base,
            progress_span=progress_span * 0.55,
        )
        asr_result = {"segments": seg_dicts, "language": detected_lang}
        audio = whisperx.load_audio(str(audio_path))
    else:
        device, compute_type = _whisper_device_and_compute()

        progress("Loading Whisper model", progress_base + 0.0 * progress_span)
        asr = _safe_load_model(
            whisperx,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            asr_options=options.asr_options(),
            vad_options=options.vad_options(),
        )

        audio = whisperx.load_audio(str(audio_path))

        # Streaming progress for the long transcribe loop. WhisperX's per-chunk
        # ratio gets remapped into our [0.10..0.55] slice of progress_span.
        transcribe_lo = 0.10
        transcribe_hi = 0.55

        def _on_transcribe_pct(label: str, pct: float) -> None:
            progress(label, progress_base + (transcribe_lo + pct * (transcribe_hi - transcribe_lo)) * progress_span)

        progress("Transcribing audio", progress_base + transcribe_lo * progress_span)
        with _ProgressCapture("Transcribing audio", _on_transcribe_pct):
            asr_result = asr.transcribe(
                audio,
                batch_size=batch_size,
                language=None if language == "auto" else language,
                chunk_size=int(options.chunk_size),
                print_progress=True,
            )
        detected_lang = asr_result.get("language") or language or "en"

        # Free Whisper before loading alignment model — keeps memory low.
        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress("Loading alignment model", progress_base + 0.55 * progress_span)
    align_device = _torch_device()
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang,
        device=align_device,
    )

    # WhisperX's align() only emits progress during a fast preprocessing pass,
    # not the slower GPU forced-alignment pass — instrumenting it would mislead
    # users into thinking we're nearly done when we aren't. Just announce the
    # phase and let the bar move on completion.
    progress("Aligning words", progress_base + 0.65 * progress_span)
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio,
        align_device,
        return_char_alignments=False,
    )

    del align_model
    gc.collect()

    progress("Alignment done", progress_base + progress_span)
    return aligned["segments"], detected_lang


# --------------------------------------------------------------------------- #
# Mode: multi-track
# --------------------------------------------------------------------------- #


def _label_for_track(stream: AudioStream, idx: int, override: str | None) -> str:
    if override:
        return override.upper()
    if stream.title:
        return stream.title.strip().upper().replace(" ", "_")
    return f"SPEAKER_{idx + 1:02d}"


def transcribe_multi_track(
    input_path: Path,
    *,
    work_dir: Path,
    speaker_labels: list[str] | None = None,
    model_name: str = "large-v3",
    language: str = "en",
    batch_size: int = 8,
    options: AdvancedOptions | None = None,
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """
    Transcribe a recording where each audio stream is one speaker.

    `speaker_labels`, if given, must have one entry per audio stream and is
    used as the spoken name for that track (e.g. ["Luke", "Guest"]).
    """
    opts = options or AdvancedOptions()
    streams = probe_audio_streams(input_path)
    if len(streams) < 2:
        raise ValueError(
            f"Expected ≥2 audio streams for multi-track mode, found {len(streams)}. "
            "Use mode='diarize' for single-track recordings."
        )

    if speaker_labels and len(speaker_labels) != len(streams):
        raise ValueError(
            f"speaker_labels has {len(speaker_labels)} entries but recording has "
            f"{len(streams)} audio streams"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[Segment] = []
    used_language = language
    labels: list[str] = []

    n = len(streams)
    for i, stream in enumerate(streams):
        label = _label_for_track(stream, i, speaker_labels[i] if speaker_labels else None)
        labels.append(label)
        track_wav = work_dir / f"track_{i:02d}_{label}.wav"

        progress(f"Extracting track {i + 1}/{n} ({label})", i / n)
        extract_track_to_wav(input_path, track_wav, stream_index=stream.index)

        progress(f"Transcribing track {i + 1}/{n} ({label})", i / n)
        seg_dicts, detected = _transcribe_with_alignment(
            track_wav,
            model_name=model_name,
            language=language,
            batch_size=batch_size,
            options=opts,
            progress=progress,
            progress_base=i / n,
            progress_span=1.0 / n,
            hf_token=os.environ.get("HF_TOKEN"),
        )
        used_language = detected

        for seg in seg_dicts:
            words = [
                Word(
                    text=w.get("word", "").strip(),
                    start=float(w.get("start", seg["start"])),
                    end=float(w.get("end", seg["end"])),
                    speaker=label,
                    score=float(w["score"]) if w.get("score") is not None else None,
                )
                for w in seg.get("words", [])
                if w.get("word")
            ]
            text = (seg.get("text") or "").strip()
            if not text and not words:
                continue
            all_segments.append(
                Segment(
                    text=text,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    speaker=label,
                    words=words,
                )
            )

    all_segments.sort(key=lambda s: s.start)

    progress("Done", 1.0)
    return TranscriptionResult(
        segments=all_segments,
        language=used_language,
        mode="multi-track",
        speaker_labels=labels,
        audio_path=input_path,
    )


# --------------------------------------------------------------------------- #
# Mode: diarize
# --------------------------------------------------------------------------- #


def transcribe_diarize(
    input_path: Path,
    *,
    work_dir: Path,
    hf_token: str | None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_name: str = "large-v3",
    language: str = "en",
    batch_size: int = 8,
    options: AdvancedOptions | None = None,
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """Single-track transcription with pyannote AI diarization."""
    if not hf_token:
        raise RuntimeError(
            "Diarization mode requires a Hugging Face token. Set HF_TOKEN in .env "
            "after accepting the licenses on the pyannote model pages. See README."
        )

    opts = options or AdvancedOptions()
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "input.wav"

    progress("Extracting audio", 0.02)
    extract_track_to_wav(input_path, wav_path, stream_index=None)

    seg_dicts, detected_lang = _transcribe_with_alignment(
        wav_path,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
        options=opts,
        progress=progress,
        progress_base=0.05,
        progress_span=0.7,
        hf_token=hf_token,
    )

    whisperx = _load_whisperx()
    diarize_device = _diarization_device()

    progress("Loading diarization model", 0.78)
    # whisperx renamed this between versions; handle both.
    if hasattr(whisperx, "DiarizationPipeline"):
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=diarize_device
        )
    else:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
        diarize_pipeline = DiarizationPipeline(
            use_auth_token=hf_token, device=diarize_device
        )
    # AMD ROCm: patch out LSTM dropout to dodge MIOpen header bug (#1995).
    # No-op on every other backend.
    _patch_pyannote_lstm_dropout(getattr(diarize_pipeline, "model", diarize_pipeline))

    progress("Running speaker diarization", 0.82)
    diar_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diar_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diar_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diar_kwargs["max_speakers"] = max_speakers

    diar_segments = diarize_pipeline(str(wav_path), **diar_kwargs)

    progress("Assigning speakers to words", 0.92)
    assigned = whisperx.assign_word_speakers(
        diar_segments, {"segments": seg_dicts, "language": detected_lang}
    )

    segments_out: list[Segment] = []
    speakers_seen: list[str] = []

    for seg in assigned["segments"]:
        seg_speaker = seg.get("speaker") or "SPEAKER_??"
        words: list[Word] = []
        for w in seg.get("words", []):
            wtext = (w.get("word") or "").strip()
            if not wtext:
                continue
            wsp = (w.get("speaker") or seg_speaker)
            words.append(
                Word(
                    text=wtext,
                    start=float(w.get("start", seg["start"])),
                    end=float(w.get("end", seg["end"])),
                    speaker=wsp,
                    score=float(w["score"]) if w.get("score") is not None else None,
                )
            )
            if wsp not in speakers_seen:
                speakers_seen.append(wsp)

        text = (seg.get("text") or "").strip()
        if not text and not words:
            continue

        if seg_speaker not in speakers_seen:
            speakers_seen.append(seg_speaker)

        segments_out.append(
            Segment(
                text=text,
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=seg_speaker,
                words=words,
            )
        )

    segments_out.sort(key=lambda s: s.start)
    progress("Done", 1.0)

    return TranscriptionResult(
        segments=segments_out,
        language=detected_lang,
        mode="diarize",
        speaker_labels=speakers_seen,
        audio_path=input_path,
    )


# --------------------------------------------------------------------------- #
# Auto entry
# --------------------------------------------------------------------------- #


def transcribe(
    input_path: Path,
    *,
    work_dir: Path,
    mode: Literal["auto", "multi-track", "diarize"] = "auto",
    speaker_labels: list[str] | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_name: str = "large-v3",
    language: str = "en",
    batch_size: int = 8,
    hf_token: str | None = None,
    options: AdvancedOptions | None = None,
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """
    Entry point.

    mode='auto' picks multi-track if the input has ≥2 audio streams,
    otherwise diarize.
    """
    opts = options or AdvancedOptions()
    if mode == "auto":
        streams = probe_audio_streams(input_path)
        mode = "multi-track" if len(streams) >= 2 else "diarize"

    if mode == "multi-track":
        return transcribe_multi_track(
            input_path,
            work_dir=work_dir,
            speaker_labels=speaker_labels,
            model_name=model_name,
            language=language,
            batch_size=batch_size,
            options=opts,
            progress=progress,
        )
    return transcribe_diarize(
        input_path,
        work_dir=work_dir,
        hf_token=hf_token or os.environ.get("HF_TOKEN"),
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
        options=opts,
        progress=progress,
    )
