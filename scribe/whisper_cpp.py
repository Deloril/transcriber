"""whisper.cpp / GGUF inference adapter for the WhisperBackend abstraction (G7.2).

This is the inference adapter that backs
:class:`scribe.whisper_backend.WhisperCppBackend`. The G7.1 commit landed
the abstraction + a placeholder; G7.2 wraps `pywhispercpp
<https://pypi.org/project/pywhispercpp/>`_ so Apple Silicon users
(and Vulkan-capable AMD / Intel iGPU boxes) get GPU-accelerated Whisper
without paying the ~5–7× CPU tax that CTranslate2 currently demands.

The module is split into three layers so the unit-test suite can
exercise the full shape without ``pywhispercpp`` installed:

* **Pure data + path layer** — the GGUF model/quant catalogue
  (:data:`SUPPORTED_MODELS`, :data:`SUPPORTED_QUANTS`), filename
  conventions (:func:`gguf_filename`), cache-dir resolution
  (:func:`default_cache_dir`, :func:`gguf_path`), and the
  :func:`hf_download_url` helper that composes the Hugging Face
  ``ggerganov/whisper.cpp`` URL where the GGUF lives.

* **Conversion layer** — :func:`convert_segments` reshapes
  pywhispercpp's per-segment / per-token output into the whisperx
  shape (``{"segments": [...], "language": "..."}``) that the rest of
  :mod:`scribe.engine` consumes. The caller is free to feed a fake
  list to exercise this layer.

* **Inference shim** — :func:`transcribe` validates inputs, resolves
  the GGUF path, invokes ``pywhispercpp`` with ``token_timestamps``
  enabled (the ``--max-len 1`` style word-timestamp mode), and
  converts the result. Tests inject a fake ``inference_fn``; production
  uses the lazy ``_default_inference`` shim that imports pywhispercpp
  on demand.

Stateless, side-effect-free except for the lazy import path. No
network IO at import time; downloading GGUF weights is outside this
module's scope (the user is expected to drop the file into
``~/.scribe/models/whisper.cpp/`` themselves; we surface clear hints
when it's missing).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Catalogue — supported GGUF model + quant combinations.
#
# Pinned to the set the PLANNING.md G7.2 entry calls out: large-v3,
# large-v3-turbo, medium; q5_0 / q8_0 / f16. ggerganov publishes more
# (large-v2, base, tiny, etc.) but Scribe's UI deliberately only
# advertises the accuracy-grade models — letting users pick "tiny.en"
# would silently regress quality compared to faster-whisper's default
# large-v3.
# --------------------------------------------------------------------------- #

SUPPORTED_MODELS: tuple[str, ...] = ("large-v3", "large-v3-turbo", "medium")
SUPPORTED_QUANTS: tuple[str, ...] = ("q5_0", "q8_0", "f16")

DEFAULT_MODEL: str = "large-v3"
DEFAULT_QUANT: str = "q5_0"

#: Hugging Face repository where the GGUF weights live. Constant, not
#: a runtime config — switching repos is a deliberate code change.
HF_REPO: str = "ggerganov/whisper.cpp"

#: Override the cache directory via this env var (CI / dev machines
#: where ``$HOME`` isn't the right home for model files).
ENV_CACHE_DIR: str = "SCRIBE_WHISPER_CPP_CACHE"


@dataclass(frozen=True)
class ModelEntry:
    """One row of the ``GET /api/whisper-cpp/models`` response.

    ``cached`` is True iff the GGUF file exists on disk under the
    resolved cache dir; ``size_bytes`` is the file size if cached
    (None otherwise). ``download_url`` is the Hugging Face URL the
    user can use to populate the cache; we don't fetch it from this
    module to keep test runs offline-safe.
    """

    model: str
    quant: str
    filename: str
    path: str
    cached: bool
    size_bytes: int | None
    download_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "quant": self.quant,
            "filename": self.filename,
            "path": self.path,
            "cached": self.cached,
            "size_bytes": self.size_bytes,
            "download_url": self.download_url,
        }


# --------------------------------------------------------------------------- #
# Pure helpers — no I/O.
# --------------------------------------------------------------------------- #


def is_supported_model(model: str) -> bool:
    """True iff ``model`` is in :data:`SUPPORTED_MODELS`."""
    return isinstance(model, str) and model in SUPPORTED_MODELS


def is_supported_quant(quant: str) -> bool:
    """True iff ``quant`` is in :data:`SUPPORTED_QUANTS`."""
    return isinstance(quant, str) and quant in SUPPORTED_QUANTS


def validate(model: str, quant: str) -> None:
    """Raise :class:`ValueError` if ``model`` / ``quant`` aren't supported."""
    if not is_supported_model(model):
        raise ValueError(
            f"Unsupported whisper.cpp model {model!r}. "
            f"Supported: {list(SUPPORTED_MODELS)}"
        )
    if not is_supported_quant(quant):
        raise ValueError(
            f"Unsupported whisper.cpp quant {quant!r}. "
            f"Supported: {list(SUPPORTED_QUANTS)}"
        )


def gguf_filename(model: str, quant: str) -> str:
    """Canonical GGUF filename for ``(model, quant)``.

    ggerganov's convention is ``ggml-<model>[-<quant>].bin`` — the
    f16 builds drop the quant suffix entirely. Quantised builds carry
    it. Examples:

    * ``("large-v3", "f16")`` → ``"ggml-large-v3.bin"``
    * ``("large-v3", "q5_0")`` → ``"ggml-large-v3-q5_0.bin"``
    * ``("medium", "q8_0")`` → ``"ggml-medium-q8_0.bin"``
    """
    validate(model, quant)
    if quant == "f16":
        return f"ggml-{model}.bin"
    return f"ggml-{model}-{quant}.bin"


def hf_download_url(model: str, quant: str) -> str:
    """Hugging Face ``resolve/main`` URL for the GGUF weight.

    The user (or a future ``download_manager`` feature) fetches this;
    we don't shell out to wget at import time. URL is the canonical
    one ggerganov publishes — switching mirrors is a deliberate code
    change.
    """
    fname = gguf_filename(model, quant)
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{fname}"


def default_cache_dir() -> Path:
    """Return the default cache directory.

    Honoured precedence:

    1. ``$SCRIBE_WHISPER_CPP_CACHE`` env var if set + non-empty.
    2. ``~/.scribe/models/whisper.cpp/`` per PLANNING.md G7.2.

    The directory is **not** created — callers that need to write into
    it call ``mkdir(parents=True, exist_ok=True)`` themselves. Returning
    a non-existent path is fine (and expected on a fresh install).
    """
    override = os.environ.get(ENV_CACHE_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".scribe" / "models" / "whisper.cpp"


def gguf_path(
    model: str,
    quant: str,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve the GGUF file path under ``cache_dir`` (or the default)."""
    cache_dir = cache_dir or default_cache_dir()
    return cache_dir / gguf_filename(model, quant)


def is_cached(
    model: str,
    quant: str,
    cache_dir: Path | None = None,
) -> bool:
    """True iff the GGUF for ``(model, quant)`` exists on disk."""
    return gguf_path(model, quant, cache_dir).is_file()


def download_gguf(
    model: str,
    quant: str,
    *,
    cache_dir: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    chunk_size: int = 1024 * 1024,
    timeout_s: float = 30.0,
    url_opener: Callable[[str, float], Any] | None = None,
) -> Path:
    """Stream a GGUF weight from Hugging Face into the cache directory.

    The file lands at :func:`gguf_path` once the byte stream completes.
    Atomic-ish: the bytes go to a sibling ``.partial`` file first and
    only get renamed after a clean close, so an interrupted download
    leaves a recognisable stub instead of a half-baked GGUF that
    pywhispercpp would gladly load and crash on.

    ``progress`` (optional) is called with ``(bytes_downloaded,
    total_bytes_or_None)`` after each chunk — used by the HTTP layer
    to surface percent-complete to the UI poller. Total may be
    ``None`` if the server didn't send Content-Length, which the
    Hugging Face CDN does in practice but we shouldn't rely on.

    ``url_opener`` is the test seam — production passes ``None`` and
    we use :mod:`urllib.request.urlopen`. Tests pass a stub that
    yields a known byte stream so we don't hit the live CDN.
    """
    validate(model, quant)
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = gguf_path(model, quant, cache_dir)
    if target.is_file():
        # Idempotent: a second click does no work and surfaces the
        # cached size for the UI to report.
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    url = hf_download_url(model, quant)

    if url_opener is None:
        # Local import keeps the module's import surface narrow; the
        # urllib subimport is otherwise unused here.
        from urllib.request import urlopen as _urlopen

        def _open(u: str, t: float) -> Any:  # noqa: ANN401
            return _urlopen(u, timeout=t)

        opener = _open
    else:
        opener = url_opener

    bytes_done = 0
    total: int | None = None
    try:
        with opener(url, timeout_s) as resp:
            total_hdr = None
            try:
                total_hdr = resp.headers.get("Content-Length")
            except Exception:
                total_hdr = None
            if total_hdr:
                try:
                    total = int(total_hdr)
                except (TypeError, ValueError):
                    total = None
            with partial.open("wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_done += len(chunk)
                    if progress is not None:
                        try:
                            progress(bytes_done, total)
                        except Exception:
                            # A misbehaving progress callback shouldn't
                            # nuke the download.
                            pass
        partial.replace(target)
    except Exception:
        # Leave the partial file behind for a future "resume" feature
        # to inspect, but never hand back a half-written target.
        if target.is_file():
            try:
                target.unlink()
            except OSError:
                pass
        raise
    return target


def list_catalogue(cache_dir: Path | None = None) -> list[ModelEntry]:
    """Return a :class:`ModelEntry` for every supported (model, quant) pair.

    Used by ``GET /api/whisper-cpp/models`` to render the Settings page
    table: which combinations are present, which need a download, what
    URL to fetch.
    """
    cache_dir = cache_dir or default_cache_dir()
    rows: list[ModelEntry] = []
    for model in SUPPORTED_MODELS:
        for quant in SUPPORTED_QUANTS:
            p = gguf_path(model, quant, cache_dir)
            cached = p.is_file()
            size = p.stat().st_size if cached else None
            rows.append(
                ModelEntry(
                    model=model,
                    quant=quant,
                    filename=gguf_filename(model, quant),
                    path=str(p),
                    cached=cached,
                    size_bytes=size,
                    download_url=hf_download_url(model, quant),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Availability — pywhispercpp import probe.
# --------------------------------------------------------------------------- #


def is_pywhispercpp_available() -> tuple[bool, str]:
    """Probe whether ``pywhispercpp`` is importable on this machine.

    Returns ``(True, "")`` if the import succeeds; ``(False, reason)``
    with a short human-readable reason otherwise. Never raises — the
    UI calls this on every page render to grey out the option.
    """
    try:
        import pywhispercpp  # noqa: F401
    except Exception as e:  # pragma: no cover - env-dependent
        return (
            False,
            "pywhispercpp not installed — `pip install pywhispercpp` "
            f"to enable Metal/Vulkan transcription ({type(e).__name__})",
        )
    return (True, "")


# --------------------------------------------------------------------------- #
# Segment conversion — pywhispercpp → whisperx shape.
#
# pywhispercpp exposes ``Model.transcribe(file, ...)`` which returns a
# list of ``Segment`` objects with ``.t0``, ``.t1`` (centiseconds), and
# ``.text``. With ``token_timestamps=True`` (or shell-out's ``--max-len
# 1``) each "segment" is one token, which gives us the word-level
# resolution Scribe's editor needs. The conversion treats each token
# as a word and groups them into whisperx-shaped segments by
# punctuation-driven sentence breaks.
#
# We accept dict input (not pywhispercpp.Segment objects) so unit tests
# can exercise the conversion without the real library installed. The
# caller-facing inference shim does the object→dict normalisation.
# --------------------------------------------------------------------------- #


def _centiseconds_to_seconds(cs: float | int) -> float:
    """pywhispercpp returns t0/t1 in centiseconds; convert to seconds."""
    return float(cs) / 100.0


def _is_sentence_end(text: str) -> bool:
    """Heuristic: a token whose stripped text ends in .!? closes a segment."""
    s = (text or "").strip()
    if not s:
        return False
    return s[-1] in ".!?"


def convert_segments(
    raw_tokens: list[dict[str, Any]],
    *,
    language: str,
    max_words_per_segment: int = 50,
) -> dict[str, Any]:
    """Reshape pywhispercpp token output into the whisperx-shaped dict.

    Each input dict needs ``t0`` (centiseconds, int/float), ``t1``
    (centiseconds), and ``text`` (the token). Tokens are grouped into
    segments at sentence boundaries (.!?) or every
    ``max_words_per_segment`` tokens, whichever comes first. The
    output mirrors what whisperx's ``align`` step would emit so the
    rest of :mod:`scribe.engine` (alignment, diarization handoff)
    works unchanged — we *replace* the alignment pass for whisper.cpp
    because the GGUF model already gave us word timestamps.

    Empty input → ``{"segments": [], "language": language}``.
    """
    if max_words_per_segment < 1:
        raise ValueError("max_words_per_segment must be >= 1")

    segments: list[dict[str, Any]] = []
    cur_words: list[dict[str, Any]] = []

    def _flush() -> None:
        if not cur_words:
            return
        text = "".join(w["word"] for w in cur_words).strip()
        seg = {
            "start": cur_words[0]["start"],
            "end": cur_words[-1]["end"],
            "text": text,
            "words": list(cur_words),
        }
        segments.append(seg)
        cur_words.clear()

    for tok in raw_tokens or []:
        raw_text = tok.get("text", "")
        if raw_text is None:
            continue
        # Skip whisper.cpp's special tokens (they're wrapped in
        # square brackets in pywhispercpp's text output).
        stripped = raw_text.strip()
        if stripped.startswith("[_") or stripped.startswith("[BLANK"):
            continue
        # Empty / pure-whitespace tokens carry no info; drop them.
        if not stripped:
            continue
        word = {
            "word": raw_text,
            "start": _centiseconds_to_seconds(tok.get("t0", 0)),
            "end": _centiseconds_to_seconds(tok.get("t1", tok.get("t0", 0))),
            "score": float(tok["score"]) if tok.get("score") is not None else None,
        }
        cur_words.append(word)
        if _is_sentence_end(raw_text) or len(cur_words) >= max_words_per_segment:
            _flush()

    _flush()
    return {"segments": segments, "language": language}


# --------------------------------------------------------------------------- #
# Inference shim — production path imports pywhispercpp lazily.
#
# The ``inference_fn`` parameter on :func:`transcribe` is the test
# seam: pass a callable to bypass the import. The default
# implementation lives in :func:`_default_inference` and is only
# reached when the caller doesn't override.
# --------------------------------------------------------------------------- #


InferenceFn = Callable[[Path, Path, str, dict[str, Any]], list[dict[str, Any]]]


def decode_audio_for_whisper_cpp(audio_path: Path) -> "Any":  # numpy.ndarray at runtime
    """Decode any audio/video file to 16 kHz mono float32 PCM via ffmpeg.

    ``pywhispercpp.Model.transcribe`` accepts either a path (its own
    loader) or a pre-decoded numpy array. The path code path is
    fragile on macOS, and the array path itself is fussy: the
    pybind11 binding behind ``whisper_full`` constructs a
    ``std::vector<float>`` from the numpy buffer and raises a
    cryptic ``ValueError: vector`` if the array is read-only,
    non-C-contiguous, non-writable, mis-aligned, empty, or the
    wrong dtype. We decode with ffmpeg ourselves and hand back an
    array that satisfies all of those constraints.

    The bytes are written to a temp file rather than read from
    ffmpeg's stdout pipe because for long recordings (multi-hour
    interviews → hundreds of MB) the OS pipe buffer can stall, and
    the resulting array gets silently truncated on some macOS
    builds. ``np.fromfile`` reads the whole thing in one syscall.

    Returns a writable, C-contiguous, 1-D ``numpy.ndarray`` of
    dtype ``float32`` in roughly [-1.0, 1.0]. Raises
    :class:`RuntimeError` if ffmpeg fails or the source has no
    audio.
    """
    import subprocess
    import tempfile

    import numpy as np  # lazy import

    ffmpeg = _ffmpeg_or_die()
    with tempfile.NamedTemporaryFile(
        prefix="scribe-whispercpp-", suffix=".f32",
        delete=False,
    ) as fh:
        pcm_path = Path(fh.name)
    try:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            str(pcm_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed to decode {audio_path}: {stderr or e}"
            ) from e
        size = pcm_path.stat().st_size
        if size == 0:
            raise RuntimeError(
                f"ffmpeg produced no audio bytes for {audio_path}. "
                "The file may have no audio track."
            )
        # ``np.fromfile`` returns a writable, C-contiguous array;
        # we still defensively copy + ascontiguousarray so the
        # array we hand pywhispercpp satisfies every flag the
        # pybind11 binding inspects.
        audio = np.fromfile(str(pcm_path), dtype=np.float32)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not audio.flags["WRITEABLE"]:  # pragma: no cover - belt + braces
            audio = audio.copy()
        if audio.size == 0:
            raise RuntimeError(
                f"Decoded zero samples from {audio_path}. The file's "
                "audio track may be empty or unreadable."
            )
        return audio
    finally:
        try:
            pcm_path.unlink()
        except OSError:
            pass


def _ffmpeg_or_die() -> str:
    """Return the absolute path to ffmpeg or raise a helpful error.

    Kept tiny + stand-alone so tests don't need to monkeypatch
    :mod:`scribe.audio` to exercise the decode helper.
    """
    import shutil
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with: brew install ffmpeg"
        )
    return p


def _default_inference(
    audio_path: Path,
    gguf_path: Path,
    language: str,
    options: dict[str, Any],
) -> list[dict[str, Any]]:  # pragma: no cover - requires pywhispercpp
    """Real inference path. Imports pywhispercpp lazily; never called by tests.

    Two pywhispercpp footguns the kwargs below dodge:

    1. **n_threads=0 doesn't mean "auto".** The docstring claims it
       falls through to ``os.cpu_count()`` but in 1.4.x it just passes
       ``0`` to whisper.cpp's thread-pool allocator, which raises
       ``ValueError: cannot create std::vector larger than max_size()``
       — surfaced through Scribe's job pipeline as a truncated
       ``ValueError: vector``. Default to a real positive thread
       count and clamp anything <= 0 to that.

    2. **Audio comes in as a numpy array we decode ourselves.** Passing
       a path lets pywhispercpp's bundled loader try to decode the
       file, which is fragile on macOS. ``decode_audio_for_whisper_cpp``
       routes through ffmpeg → 16 kHz mono float32, which is what the
       binding wants natively.
    """
    from pywhispercpp.model import Model  # type: ignore[import-not-found]

    # Pick a sane default thread count so n_threads=0 doesn't flow
    # through to whisper.cpp. ``os.cpu_count()`` returns None on some
    # exotic platforms; fall through to 4 in that case.
    cpu_default = os.cpu_count() or 4
    n_threads = int(options.get("n_threads") or cpu_default)
    if n_threads < 1:
        n_threads = cpu_default

    model = Model(
        str(gguf_path),
        language=None if language in (None, "", "auto") else language,
        token_timestamps=True,
        n_threads=n_threads,
    )
    # Decode ourselves rather than letting pywhispercpp's loader do
    # it; see decode_audio_for_whisper_cpp() for why.
    audio = decode_audio_for_whisper_cpp(audio_path)
    raw = model.transcribe(audio)
    out: list[dict[str, Any]] = []
    for seg in raw:
        # pywhispercpp segments expose .t0 / .t1 / .text
        out.append(
            {
                "t0": getattr(seg, "t0", 0),
                "t1": getattr(seg, "t1", 0),
                "text": getattr(seg, "text", ""),
            }
        )
    return out


def transcribe(
    audio_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    quant: str = DEFAULT_QUANT,
    language: str = "en",
    cache_dir: Path | None = None,
    progress: Callable[[str, float], None] | None = None,
    progress_base: float = 0.0,
    progress_span: float = 1.0,
    inference_fn: InferenceFn | None = None,
    inference_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run whisper.cpp inference and return a whisperx-shaped dict.

    Steps:

    1. Validate ``model`` / ``quant`` against the supported catalogue.
    2. Resolve the GGUF path; raise :class:`FileNotFoundError` (with
       the download URL in the message) if the file isn't cached. The
       UI catches this and prompts the user to fetch the weight.
    3. Drive ``progress(...)`` through the standard
       ``progress_base``/``progress_span`` budgeting convention.
    4. Call ``inference_fn`` (lazily defaulting to pywhispercpp).
    5. Hand off the raw token list to :func:`convert_segments`.

    Returns ``{"segments": [...], "language": "<iso>"}``. Word
    timestamps are populated on every segment so the engine can skip
    the whisperx alignment pass for whisper.cpp transcripts.
    """
    validate(model, quant)
    path = gguf_path(model, quant, cache_dir)
    if not path.is_file():
        url = hf_download_url(model, quant)
        raise FileNotFoundError(
            f"GGUF model not cached: {path}. Download from {url} "
            f"or run the model download manager."
        )

    fn = inference_fn or _default_inference
    opts = dict(inference_options or {})

    if progress is not None:
        progress(
            f"Loading whisper.cpp model ({model}, {quant})",
            progress_base + 0.0 * progress_span,
        )

    raw_tokens = fn(audio_path, path, language, opts)

    if progress is not None:
        progress("Converting whisper.cpp segments", progress_base + 0.95 * progress_span)

    result = convert_segments(raw_tokens, language=language)

    if progress is not None:
        progress("whisper.cpp inference complete", progress_base + 1.0 * progress_span)

    return result
