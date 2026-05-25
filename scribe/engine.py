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
# Device selection
# --------------------------------------------------------------------------- #


def _torch_device() -> str:
    forced = os.environ.get("SCRIBE_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cuda_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _diarization_device() -> str:
    """
    pyannote runs cleanly on CUDA and CPU; on MPS some ops still fall back
    and the result can be slower than plain CPU. Default to CUDA if present,
    else CPU. Override with SCRIBE_DIARIZE_DEVICE.
    """
    forced = os.environ.get("SCRIBE_DIARIZE_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def _whisper_device_and_compute() -> tuple[str, str]:
    """faster-whisper device + compute type. CTranslate2 has no MPS backend."""
    forced_device = os.environ.get("SCRIBE_WHISPER_DEVICE", "").strip().lower()
    forced_compute = os.environ.get("SCRIBE_COMPUTE_TYPE", "").strip().lower()

    if forced_device in {"cuda", "cpu"}:
        device = forced_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if forced_compute:
        return device, forced_compute

    if device == "cuda":
        # large-v3 fp16 wants ~6 GB; int8_float16 brings it under ~4 GB.
        return device, "int8_float16" if _cuda_vram_gb() < 8 else "float16"
    return device, "int8"


# --------------------------------------------------------------------------- #
# WhisperX wrappers
# --------------------------------------------------------------------------- #


def _load_whisperx():
    """Lazy import — whisperx pulls in heavy deps."""
    import whisperx  # type: ignore
    return whisperx


def _transcribe_with_alignment(
    audio_path: Path,
    *,
    model_name: str,
    language: str,
    batch_size: int,
    progress: ProgressFn,
    progress_base: float,
    progress_span: float,
) -> tuple[list[dict[str, Any]], str]:
    """
    Run WhisperX transcription + word-level alignment on a single audio file.

    Returns (aligned_segments, detected_language).
    """
    whisperx = _load_whisperx()
    device, compute_type = _whisper_device_and_compute()

    progress("Loading Whisper model", progress_base + 0.0 * progress_span)
    asr = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language if language != "auto" else None,
        asr_options={
            # These are what unblocks long monologues: VAD chunks on silences
            # rather than fixed 30s windows, and we don't condition on prior
            # text (which is the main source of repeat-loop hallucinations).
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.45,
            "compression_ratio_threshold": 2.4,
        },
    )

    audio = whisperx.load_audio(str(audio_path))

    progress("Transcribing audio", progress_base + 0.1 * progress_span)
    asr_result = asr.transcribe(audio, batch_size=batch_size, language=None if language == "auto" else language)
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
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """
    Transcribe a recording where each audio stream is one speaker.

    `speaker_labels`, if given, must have one entry per audio stream and is
    used as the spoken name for that track (e.g. ["Luke", "Guest"]).
    """
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
            progress=progress,
            progress_base=i / n,
            progress_span=1.0 / n,
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
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """Single-track transcription with pyannote AI diarization."""
    if not hf_token:
        raise RuntimeError(
            "Diarization mode requires a Hugging Face token. Set HF_TOKEN in .env "
            "after accepting the licenses on the pyannote model pages. See README."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "input.wav"

    progress("Extracting audio", 0.02)
    extract_track_to_wav(input_path, wav_path, stream_index=None)

    seg_dicts, detected_lang = _transcribe_with_alignment(
        wav_path,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
        progress=progress,
        progress_base=0.05,
        progress_span=0.7,
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
    progress: ProgressFn = _noop_progress,
) -> TranscriptionResult:
    """
    Entry point.

    mode='auto' picks multi-track if the input has ≥2 audio streams,
    otherwise diarize.
    """
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
        progress=progress,
    )
