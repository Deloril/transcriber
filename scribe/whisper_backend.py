"""Pluggable transcription-engine backend abstraction (G7.1).

Today :mod:`scribe.engine` is hard-wired to faster-whisper / CTranslate2.
That's the right default on CUDA / ROCm / CPU, but CT2 has no Metal
backend, so on Apple Silicon Whisper falls back to ``cpu`` + ``int8``
and a 60-minute video takes ~60-70 minutes — real user pain
(see ``PLANNING.md`` for the full motivation).

This module introduces a thin abstraction so the user can pick which
inference engine handles the Whisper pass. The pure logic in
:mod:`scribe.engine` (VAD chunking, alignment, diarization handoff)
stays put; only the **inference call** routes through a
:class:`WhisperBackend` implementation.

The G7.1 deliverable is the abstraction itself plus the
``faster-whisper`` concrete backend that wraps the existing engine
helper. The ``whisper.cpp`` backend is registered as a stub so the UI
can surface it; the actual adapter (GGUF model loading, Metal /
Vulkan inference, word-timestamp extraction) lands in G7.2. See
``WhisperCppBackend`` for the placeholder.

Backend objects are stateless and registered at import time; the
registry is keyed by a stable id (``faster-whisper``, ``whisper.cpp``)
that the upload form, ``Job.whisper_backend`` field, and persisted
``job.json`` all share. New backends register a subclass + an entry
in :data:`BACKEND_REGISTRY` and the rest of the system picks them up
automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ProgressFn = Callable[[str, float], None]


# --------------------------------------------------------------------------- #
# Backend ids — stable strings used in the UI, ``Job.whisper_backend``,
# and persisted ``job.json``. Don't rename without a migration.
# --------------------------------------------------------------------------- #

BACKEND_FASTER_WHISPER = "faster-whisper"
BACKEND_WHISPER_CPP = "whisper.cpp"


# --------------------------------------------------------------------------- #
# Public dataclass returned by ``describe_backends()`` to the UI.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackendInfo:
    """Serialisable snapshot of a registered backend.

    Returned by :func:`describe_backends` and consumed by the upload
    page's engine selector. ``available`` is a runtime check
    (``is_available()``) — the registry always advertises every
    backend so the UI can grey out ones that need extra setup
    instead of hiding them silently.
    """

    id: str
    display_name: str
    description: str
    supported_devices: tuple[str, ...]
    model_format: str
    available: bool
    unavailable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "supported_devices": list(self.supported_devices),
            "model_format": self.model_format,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #


class WhisperBackend(ABC):
    """Abstract Whisper inference backend.

    Subclasses set the class-level identity / metadata fields and
    implement :meth:`is_available` + :meth:`transcribe`. Everything
    else (alignment, diarization, write-out) stays in
    :mod:`scribe.engine`; backends only own the
    "audio in → segment dicts out" inference call.
    """

    id: str = ""
    display_name: str = ""
    description: str = ""
    supported_devices: tuple[str, ...] = ()
    model_format: str = ""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return ``(True, "")`` if the backend can run on this machine.

        Otherwise ``(False, reason)`` — short human-readable string the
        UI can render in a hint strip ("pywhispercpp not installed —
        see G7.2"). Should never raise; check imports + binary
        presence + return cleanly.
        """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        model_name: str,
        language: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        progress: ProgressFn,
        progress_base: float = 0.0,
        progress_span: float = 1.0,
    ) -> dict[str, Any]:
        """Run inference on ``audio_path``.

        Returns ``{"segments": [...], "language": "<iso>"}`` in the
        whisperx-compatible shape consumed by the rest of
        :mod:`scribe.engine` (alignment, diarization). Subclasses are
        free to use whatever inference path they want as long as the
        return shape matches.

        ``progress_base`` + ``progress_span`` map the backend's
        internal 0–1 progress into the engine's wider progress
        contract — same convention the existing engine helpers use.
        """

    def info(self) -> BackendInfo:
        avail, reason = self.is_available()
        return BackendInfo(
            id=self.id,
            display_name=self.display_name,
            description=self.description,
            supported_devices=tuple(self.supported_devices),
            model_format=self.model_format,
            available=avail,
            unavailable_reason=reason,
        )


# --------------------------------------------------------------------------- #
# faster-whisper (default)
# --------------------------------------------------------------------------- #


class FasterWhisperBackend(WhisperBackend):
    """Wraps the existing whisperx / CTranslate2 inference path.

    This is the historical default — every backend that has shipped to
    date has been faster-whisper. The implementation delegates to
    :func:`scribe.engine._run_faster_whisper_inference` so the
    abstraction stays a thin shim and the existing
    ``_safe_load_model`` + ``asr.transcribe`` machinery (with all its
    fallback handling and progress-capture plumbing) remains the
    single source of truth for the inference call.
    """

    id = BACKEND_FASTER_WHISPER
    display_name = "faster-whisper"
    description = (
        "Default. CTranslate2 INT8 / FP16 on CUDA / ROCm / CPU. "
        "No Metal — Apple Silicon falls back to CPU."
    )
    supported_devices = ("cuda", "rocm", "cpu")
    model_format = "ct2"

    def is_available(self) -> tuple[bool, str]:
        try:
            import whisperx  # noqa: F401
        except Exception as e:  # pragma: no cover - env-dependent
            return (False, f"whisperx not installed: {e}")
        return (True, "")

    def transcribe(
        self,
        audio_path: Path,
        *,
        model_name: str,
        language: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        progress: ProgressFn,
        progress_base: float = 0.0,
        progress_span: float = 1.0,
    ) -> dict[str, Any]:
        # Lazy import so test imports of this module don't pull in
        # whisperx (which yanks torch + ctranslate2 + nemo etc).
        from . import engine as _engine

        # ``asr_options`` carries ``batch_size`` + ``chunk_size`` (the
        # engine bakes them in before dispatch); the helper strips
        # both before forwarding the rest to ``whisperx.load_model``.
        return _engine._run_faster_whisper_inference(
            audio_path,
            model_name=model_name,
            language=language,
            asr_options=dict(asr_options),
            vad_options=dict(vad_options),
            progress=progress,
            progress_base=progress_base,
            progress_span=progress_span,
        )


# --------------------------------------------------------------------------- #
# whisper.cpp (placeholder for G7.2)
# --------------------------------------------------------------------------- #


class WhisperCppBackend(WhisperBackend):
    """GGUF / whisper.cpp inference backend (G7.2).

    Wraps :mod:`scribe.whisper_cpp` (which in turn wraps
    ``pywhispercpp``) so Apple Silicon and Vulkan-capable boxes get
    GPU-accelerated Whisper. Routes ``model_name`` straight through to
    the GGUF catalogue; the quant comes off ``asr_options`` (the engine
    bakes ``whisper_cpp_quant`` in before dispatch). Word timestamps are
    emitted natively via pywhispercpp's ``token_timestamps=True``, so
    the engine can skip the whisperx alignment pass for whisper.cpp
    transcripts (the converter populates ``segments[i]["words"]`` with
    per-word start/end seconds).

    :meth:`is_available` reports True iff ``pywhispercpp`` is
    importable. The runtime catalogue (which GGUFs the user has on
    disk) is *not* part of availability — a missing GGUF is a clear
    runtime ``FileNotFoundError`` with a download URL, not a silent
    "backend disabled" surprise.
    """

    id = BACKEND_WHISPER_CPP
    display_name = "whisper.cpp"
    description = (
        "GGUF weights with CPU + Metal + Vulkan acceleration. "
        "Recommended on Apple Silicon (G7.2)."
    )
    supported_devices = ("cpu", "mps", "cuda", "vulkan")
    model_format = "gguf"

    def is_available(self) -> tuple[bool, str]:
        from . import whisper_cpp
        return whisper_cpp.is_pywhispercpp_available()

    def transcribe(
        self,
        audio_path: Path,
        *,
        model_name: str,
        language: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        progress: ProgressFn,
        progress_base: float = 0.0,
        progress_span: float = 1.0,
    ) -> dict[str, Any]:
        from . import whisper_cpp

        quant = str(
            asr_options.get("whisper_cpp_quant")
            or whisper_cpp.DEFAULT_QUANT
        )
        # vad_options is unused by whisper.cpp (the GGUF model has its
        # own VAD); accept and ignore so the backend signature stays
        # uniform across implementations.
        _ = vad_options
        inference_options: dict[str, Any] = {}
        if "n_threads" in asr_options:
            inference_options["n_threads"] = asr_options["n_threads"]
        return whisper_cpp.transcribe(
            audio_path,
            model=model_name,
            quant=quant,
            language=language,
            progress=progress,
            progress_base=progress_base,
            progress_span=progress_span,
            inference_options=inference_options,
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


# The registry is mutable so tests / future backends can register at
# runtime. ``register_backend`` enforces uniqueness.
BACKEND_REGISTRY: dict[str, WhisperBackend] = {}


def register_backend(backend: WhisperBackend) -> None:
    """Register a backend by its ``id``.

    Idempotent if the same instance is registered twice; raises
    ``ValueError`` if the id already maps to a *different* instance.
    """
    if not backend.id:
        raise ValueError("WhisperBackend.id must be a non-empty string")
    existing = BACKEND_REGISTRY.get(backend.id)
    if existing is not None and existing is not backend:
        raise ValueError(
            f"Backend id {backend.id!r} is already registered "
            f"(existing={type(existing).__name__})"
        )
    BACKEND_REGISTRY[backend.id] = backend


def unregister_backend(backend_id: str) -> None:
    """Remove a backend by id (test / extension support)."""
    BACKEND_REGISTRY.pop(backend_id, None)


def available_backend_ids() -> list[str]:
    """All registered backend ids, in registration order."""
    return list(BACKEND_REGISTRY.keys())


def get_backend(backend_id: str) -> WhisperBackend:
    """Look up a backend by id, raising ``KeyError`` if unknown."""
    if backend_id not in BACKEND_REGISTRY:
        raise KeyError(
            f"Unknown whisper backend {backend_id!r} — "
            f"registered: {sorted(BACKEND_REGISTRY)}"
        )
    return BACKEND_REGISTRY[backend_id]


def describe_backends() -> list[dict[str, Any]]:
    """List every registered backend as a JSON-serialisable dict.

    Used by the ``GET /api/whisper-backends`` endpoint and by the
    upload page's server-rendered ``<select>``.
    """
    return [b.info().to_dict() for b in BACKEND_REGISTRY.values()]


def default_backend_id(device_label: str | None = None) -> str:
    """Return the default backend id for a given device label.

    G7.3 — when ``device_label == "mps"`` (Apple Silicon) AND the
    :class:`WhisperCppBackend` reports available
    (``pywhispercpp`` importable), prefer ``whisper.cpp``. Metal
    acceleration via GGUF weights is roughly 5× faster than CT2's
    CPU fallback on Apple Silicon, so the default flips for that
    audience. faster-whisper stays in the registry — users can
    A/B for accuracy — but the page lands with whisper.cpp pre-
    selected.

    On every other backend (``cuda`` / ``rocm`` / ``cpu``) the
    historical ``faster-whisper`` default holds. If whisper.cpp's
    runtime check fails (e.g. ``pywhispercpp`` not installed), the
    function falls back to ``faster-whisper`` so the page never
    lands on a backend that can't actually run.

    Accepts ``None`` (no device known) and unknown labels — both
    fall through to the historical default.
    """
    label = (device_label or "").strip().lower()
    if label == "mps":
        cpp = BACKEND_REGISTRY.get(BACKEND_WHISPER_CPP)
        if cpp is not None:
            try:
                available, _reason = cpp.is_available()
            except Exception:  # pragma: no cover - defensive
                available = False
            if available:
                return BACKEND_WHISPER_CPP
    return BACKEND_FASTER_WHISPER


# --------------------------------------------------------------------------- #
# G7.3 — UI recommendation hint
# --------------------------------------------------------------------------- #


def recommended_backend_for_device(
    device_label: str | None,
) -> dict[str, Any] | None:
    """Return a serialisable hint when the active device has a non-default
    backend recommendation, else ``None``.

    G7.3 — Apple Silicon (``mps``) gets a "whisper.cpp recommended"
    banner above the engine selector with ~5× speedup messaging.
    The hint is always rendered when the device matches, regardless
    of whether ``pywhispercpp`` is installed; the
    ``available`` / ``unavailable_reason`` fields tell the UI
    whether to render an "install pywhispercpp" prompt or a "default
    flipped" confirmation.

    The dict shape is:

    .. code-block:: json

        {
          "device": "mps",
          "recommended_backend_id": "whisper.cpp",
          "available": false,
          "unavailable_reason": "...",
          "headline": "Apple Silicon detected — whisper.cpp recommended",
          "detail": "GPU-accelerated transcription, ~5× faster ..."
        }

    All other devices (``cuda`` / ``rocm`` / ``cpu`` / unknown / None)
    return ``None`` because faster-whisper is already the right
    default there and showing a banner would be noise.
    """
    label = (device_label or "").strip().lower()
    if label != "mps":
        return None
    cpp = BACKEND_REGISTRY.get(BACKEND_WHISPER_CPP)
    if cpp is None:
        return None
    try:
        available, reason = cpp.is_available()
    except Exception:  # pragma: no cover - defensive
        available = False
        reason = "whisper.cpp backend probe raised"
    return {
        "device": "mps",
        "recommended_backend_id": BACKEND_WHISPER_CPP,
        "available": bool(available),
        "unavailable_reason": "" if available else (reason or ""),
        "headline": "Apple Silicon detected — whisper.cpp recommended",
        "detail": (
            "GPU-accelerated transcription, ~5× faster than the "
            "faster-whisper CPU fallback. whisper.cpp routes Whisper "
            "through Metal via GGUF weights; faster-whisper stays "
            "available for A/B comparisons."
        ),
    }


def is_valid_backend_id(backend_id: str | None) -> bool:
    """True iff ``backend_id`` is a non-empty string in the registry."""
    if not isinstance(backend_id, str) or not backend_id:
        return False
    return backend_id in BACKEND_REGISTRY


# --------------------------------------------------------------------------- #
# Initial registration. Order matters — first-registered shows first
# in the UI dropdown.
# --------------------------------------------------------------------------- #

register_backend(FasterWhisperBackend())
register_backend(WhisperCppBackend())
