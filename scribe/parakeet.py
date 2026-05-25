"""
Parakeet (NVIDIA NeMo) transcription path.

Lazy-loaded — NeMo is a large optional dependency. The public function here
returns the same (aligned_segments, language) shape that the WhisperX path
returns, so the rest of the engine (forced alignment, diarization, output
writers) is unchanged.

We deliberately reuse whisperx.vads.Pyannote for VAD chunking. Parakeet TDT
*can* transcribe arbitrary-length audio in one shot, but for hour-long
interviews the VAD-on-silences strategy guarantees clean chunk boundaries
and keeps memory bounded — same long-monologue fix that motivated this
project.
"""

from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ProgressFn = Callable[[str, float], None]


def _noop_progress(_msg: str, _f: float) -> None:
    return None


def is_parakeet_model(name: str) -> bool:
    """True if the model name should be routed through this engine."""
    n = name.lower()
    return n.startswith("nvidia/parakeet") or n.startswith("parakeet")


_NEMO_AVAILABLE: bool | None = None
_IMPORT_ERROR: str | None = None


def nemo_available() -> tuple[bool, str | None]:
    """Cheap probe: is `nemo_toolkit[asr]` importable?"""
    global _NEMO_AVAILABLE, _IMPORT_ERROR
    if _NEMO_AVAILABLE is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import nemo.collections.asr  # type: ignore  # noqa: F401
            _NEMO_AVAILABLE = True
            _IMPORT_ERROR = None
        except Exception as e:  # noqa: BLE001
            _NEMO_AVAILABLE = False
            _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    return _NEMO_AVAILABLE, _IMPORT_ERROR


def _normalise_model_id(name: str) -> str:
    """Accept short names (parakeet-tdt-0.6b-v2) or HF IDs (nvidia/...)."""
    if "/" in name:
        return name
    return f"nvidia/{name}"


_MODEL_CACHE: dict[str, Any] = {}


def _load_nemo_model(model_id: str, device: str):
    """Load (and cache) a NeMo ASR model. Returns the loaded model on `device`."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import nemo.collections.asr as nemo_asr  # type: ignore

    # Parakeet TDT models are EncDecRNNTBPEModel; some Parakeet CTC variants
    # are EncDecCTCModelBPE. Try ASRModel.from_pretrained which is the
    # base-class loader and handles both.
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model.eval()
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception:
            # Fallback to CPU if the requested device is unavailable
            model = model.cpu()
    _MODEL_CACHE[model_id] = model
    return model


def _vad_chunks(audio: np.ndarray, hf_token: str | None, *, vad_onset: float, vad_offset: float, chunk_size: int) -> list[dict[str, float]]:
    """
    Re-run the same VAD strategy WhisperX uses, but yield (start, end) pairs.

    The whisperx.vads.Pyannote class loads pyannote VAD weights and exposes
    a static merge_chunks. We reuse that here so chunk boundaries are
    consistent across the two engine paths.
    """
    import whisperx  # type: ignore
    from whisperx.vads import Pyannote  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad = Pyannote(
        device,
        use_auth_token=hf_token,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
    )
    waveform = Pyannote.preprocess_audio(audio)
    raw = vad({"waveform": waveform, "sample_rate": 16000})
    segments = Pyannote.merge_chunks(raw, chunk_size, onset=vad_onset, offset=vad_offset)
    if not segments:
        # Whole-file fallback if VAD found nothing — better than silently
        # returning an empty transcript.
        duration = float(len(audio) / 16000.0)
        return [{"start": 0.0, "end": duration}]
    out: list[dict[str, float]] = []
    for s in segments:
        out.append({"start": float(s["start"]), "end": float(s["end"])})
    return out


def transcribe_with_parakeet(
    audio_path: Path,
    *,
    model_name: str,
    hf_token: str | None,
    options,  # AdvancedOptions — type hinted lazily to avoid circular import
    progress: ProgressFn = _noop_progress,
    progress_base: float = 0.0,
    progress_span: float = 1.0,
) -> tuple[list[dict[str, Any]], str]:
    """
    Transcribe `audio_path` with the requested Parakeet model.

    Returns a list of segment dicts shaped like WhisperX's pre-alignment
    output (`{"start", "end", "text"}`) plus the language. The rest of the
    pipeline (forced alignment via wav2vec2, then diarization) is identical
    to the Whisper path.
    """
    ok, err = nemo_available()
    if not ok:
        raise RuntimeError(
            "Parakeet requires NVIDIA NeMo. Install with:\n"
            "  pip install -r requirements-parakeet.txt\n"
            f"Import error: {err}"
        )

    import whisperx  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress("Loading Parakeet model", progress_base + 0.0 * progress_span)
    model_id = _normalise_model_id(model_name)
    model = _load_nemo_model(model_id, device)

    progress("Loading audio", progress_base + 0.05 * progress_span)
    audio = whisperx.load_audio(str(audio_path))  # 16 kHz mono float32

    progress("Detecting speech", progress_base + 0.10 * progress_span)
    chunks = _vad_chunks(
        audio,
        hf_token,
        vad_onset=float(options.vad_onset),
        vad_offset=float(options.vad_offset),
        chunk_size=int(options.chunk_size),
    )

    progress(f"Transcribing audio ({len(chunks)} chunks)", progress_base + 0.15 * progress_span)

    # Slice audio per chunk and feed Parakeet. NeMo's transcribe() accepts a
    # list of arrays directly (numpy, 16 kHz, mono, float32 in [-1, 1]).
    segments_out: list[dict[str, Any]] = []
    SR = 16000
    transcribe_lo = 0.15
    transcribe_hi = 0.95
    n = max(1, len(chunks))

    for i, ch in enumerate(chunks):
        start_t = float(ch["start"])
        end_t = float(ch["end"])
        f1 = max(0, int(start_t * SR))
        f2 = min(len(audio), int(end_t * SR))
        if f2 <= f1:
            continue
        slice_ = audio[f1:f2].astype(np.float32, copy=False)
        if slice_.size == 0:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.inference_mode():
                    out = model.transcribe([slice_], batch_size=1, verbose=False)
        except Exception as e:  # noqa: BLE001
            # If a single chunk fails (e.g. a non-speech artefact), record an
            # empty segment and keep going — losing one chunk is far better
            # than aborting the whole transcription.
            segments_out.append({"start": start_t, "end": end_t, "text": ""})
            print(f"[scribe] Parakeet chunk {i + 1}/{n} failed: {e}")
        else:
            text = _extract_nemo_text(out)
            if text:
                segments_out.append({"start": start_t, "end": end_t, "text": text})

        progress(
            f"Transcribing audio ({i + 1}/{n})",
            progress_base + (transcribe_lo + ((i + 1) / n) * (transcribe_hi - transcribe_lo)) * progress_span,
        )

    # Free the model afterwards if we want to keep peak VRAM low. We leave
    # it in the cache so subsequent jobs don't re-download/re-load — but on
    # tight VRAM machines, callers can clear _MODEL_CACHE between jobs.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress("Parakeet done", progress_base + progress_span)
    return segments_out, "en"


def _extract_nemo_text(result: Any) -> str:
    """
    NeMo's transcribe() return shape varies by model and version. This handles
    the four shapes Parakeet TDT/CTC have shipped over the last year.
    """
    if not result:
        return ""
    # Tuple of (hypotheses, scores) or (hypotheses, all_hypotheses)
    if isinstance(result, tuple):
        result = result[0]
    if not result:
        return ""
    item = result[0]
    # Hypothesis object with .text
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text.strip()
    # Plain string
    if isinstance(item, str):
        return item.strip()
    # Dict-shaped {"text": "..."}
    if isinstance(item, dict) and "text" in item:
        return str(item["text"]).strip()
    return str(item).strip()


def free_models() -> None:
    """Free any cached NeMo models. Call between jobs on tight-VRAM systems."""
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
