"""Pluggable model backend abstraction (F8.1).

Per PLANNING.md F8.1:

  > Pluggable model backend abstraction (Ollama HTTP API first;
  > llama.cpp / transformers later). User selects model from a list.

This module is the **plumbing** the rest of the academic-coding
engine (F8.2 embedding index, F8.3/F8.4 code suggestion, F8.5 find
similar quotes, F8.6 whole-transcript review, F8.8 memo drafts) will
bolt onto. Nothing in here loads a model, embeds anything, or talks
to a real network — it defines the *shape* of a backend and ships
one concrete implementation for Ollama, with the HTTP transport
deliberately injectable so tests can run offline.

Scope deliberately limited to the F8.1 core. Deferred to follow-up
features:

  * Streaming generation (Ollama supports SSE; we expose the simple
    non-streaming surface first because callers don't need streaming
    for "suggest one code on this span" — that's a 1–4 s call.).
  * llama.cpp / transformers concrete backends (the ABC is here so
    those can plug in without changes to call-sites).
  * Auth tokens for remote-hosted backends (``extra_headers`` is on
    ``BackendConfig`` but there is no UI flow yet).
  * Hardware-tier autodetection + model-tier picker (F8.11 / F8.12).

Design notes
------------

* **No global state.** Every operation takes a ``BackendConfig`` and a
  ``transport`` (callable). This keeps the module testable without
  mocking and lets multiple projects target different Ollama hosts in
  the same Scribe install.

* **Errors are typed.** ``BackendUnavailable`` means "the daemon isn't
  reachable / unhealthy"; ``BackendValidationError`` means "you asked
  for something the backend can't do (model not found, empty prompt,
  invalid config)"; ``BackendError`` is the catch-all base class. The
  server layer maps these to 502 / 400 / 500 respectively.

* **Settings live on the Project (F1.1 / F3.1).** A project-scoped
  config matters because different research corpora may want
  different models — multilingual ``bge-m3`` for non-English data,
  ``llama3.2:3b`` for laptop, etc. ``load_backend_config(project)``
  reads from ``project.settings`` and falls back to sensible defaults
  (localhost Ollama, no models pinned).

* **The Ollama API surface used:**
    - ``GET  /api/version``  — health check, returns ``{version}``
    - ``GET  /api/tags``     — list installed models
    - ``POST /api/generate`` — non-stream completion
    - ``POST /api/embed``    — batch embeddings (Ollama 0.2+)
  We intentionally keep the API surface narrow; this is a
  building-block module, not a full Ollama client.
"""

from __future__ import annotations

import abc
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .projects import (
    MAX_SETTINGS_STRING_LEN,
    Project,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Backend providers we know about. The string identifiers are stable and
# are written to disk in ``project.settings``; renaming any of them would
# be an on-disk migration. ``ollama`` ships in F8.1; the others are
# placeholders so the registry shape is forward-compatible.
PROVIDER_OLLAMA = "ollama"
PROVIDER_LLAMA_CPP = "llama_cpp"      # deferred (F8.x)
PROVIDER_TRANSFORMERS = "transformers"  # deferred (F8.x)

KNOWN_PROVIDERS: tuple[str, ...] = (
    PROVIDER_OLLAMA,
    PROVIDER_LLAMA_CPP,
    PROVIDER_TRANSFORMERS,
)

# The default endpoint Ollama listens on out-of-the-box. Callers that
# point at a different host (LAN box, custom port) override this.
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Reasonable timeouts. Generation is slow on CPU (whole-transcript review
# is minutes; one suggestion is seconds). Default to 120 s for generate
# and 30 s for everything else; both are overridable.
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_GENERATE_TIMEOUT_S = 120.0

# Settings keys (extend ``scribe.projects.SETTING_AI_*``). Stored as a
# nested dict under ``project.settings["ai_backend"]`` so we can grow
# the schema without colliding with other AI-prefixed settings.
#
# Note: ``Project.settings`` only allows one level of nested dicts (see
# ``scribe.projects._validate_settings_value``). The ``ai_backend`` dict
# therefore holds **scalars only**; headers — which are themselves a
# {key: value} map — live in a *sibling* setting ``ai_backend_headers``
# at the top level, where they remain a depth-1 dict-of-scalars.
SETTING_AI_BACKEND = "ai_backend"
SETTING_AI_BACKEND_HEADERS = "ai_backend_headers"
SETTING_KEY_PROVIDER = "provider"
SETTING_KEY_BASE_URL = "base_url"
SETTING_KEY_DEFAULT_MODEL = "default_model"
SETTING_KEY_DEFAULT_EMBEDDING_MODEL = "default_embedding_model"
SETTING_KEY_REQUEST_TIMEOUT = "request_timeout_s"
SETTING_KEY_GENERATE_TIMEOUT = "generate_timeout_s"

# Field length / cardinality limits. These mirror the Project settings
# limits so a backend config can't bloat ``project.json``.
MAX_BASE_URL_LEN = 1024
MAX_MODEL_NAME_LEN = 256
MAX_PROMPT_LEN = 256 * 1024  # 256 KiB — generous for whole-transcript pass
MAX_EMBED_BATCH = 256        # one ``embed`` call ≤ 256 inputs
MAX_EMBED_INPUT_LEN = 64 * 1024


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BackendError(Exception):
    """Base class for all backend-related failures."""


class BackendUnavailable(BackendError):
    """The backend daemon is unreachable, slow, or returned 5xx.

    Maps to HTTP 502 at the server layer. Distinct from
    ``BackendValidationError`` because the caller's input is fine — the
    daemon is the problem.
    """


class BackendValidationError(BackendError):
    """The caller asked for something the backend rejected.

    Examples: requested model isn't installed, prompt is empty, config
    has an unknown provider. Maps to HTTP 400.
    """


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackendConfig:
    """How to reach a model backend, with sensible defaults.

    All fields are validated by ``validate``; construct via
    ``BackendConfig.new`` to apply defaults + validation in one step.
    """

    provider: str = PROVIDER_OLLAMA
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_model: str = ""
    default_embedding_model: str = ""
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    generate_timeout_s: float = DEFAULT_GENERATE_TIMEOUT_S
    # Free-form headers attached to every request. Stored as a tuple of
    # (key, value) pairs so the dataclass stays hashable / frozen-safe.
    extra_headers: tuple[tuple[str, str], ...] = ()

    # ------------------------------------------------------------------ #
    # Construction / validation
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        provider: str = PROVIDER_OLLAMA,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        default_model: str = "",
        default_embedding_model: str = "",
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        generate_timeout_s: float = DEFAULT_GENERATE_TIMEOUT_S,
        extra_headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
    ) -> "BackendConfig":
        if extra_headers is None:
            headers: tuple[tuple[str, str], ...] = ()
        elif isinstance(extra_headers, Mapping):
            headers = tuple((str(k), str(v)) for k, v in extra_headers.items())
        else:
            headers = tuple((str(k), str(v)) for k, v in extra_headers)
        c = cls(
            provider=provider,
            base_url=base_url,
            default_model=default_model,
            default_embedding_model=default_embedding_model,
            request_timeout_s=float(request_timeout_s),
            generate_timeout_s=float(generate_timeout_s),
            extra_headers=headers,
        )
        c.validate()
        return c

    def validate(self) -> None:
        if self.provider not in KNOWN_PROVIDERS:
            raise BackendValidationError(
                f"Unknown provider {self.provider!r}; "
                f"must be one of {KNOWN_PROVIDERS}"
            )
        url = (self.base_url or "").strip()
        if not url:
            raise BackendValidationError("base_url is required")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise BackendValidationError(
                f"base_url must start with http:// or https://; got {url!r}"
            )
        if len(url) > MAX_BASE_URL_LEN:
            raise BackendValidationError(
                f"base_url exceeds {MAX_BASE_URL_LEN} chars"
            )
        if len(self.default_model) > MAX_MODEL_NAME_LEN:
            raise BackendValidationError(
                f"default_model exceeds {MAX_MODEL_NAME_LEN} chars"
            )
        if len(self.default_embedding_model) > MAX_MODEL_NAME_LEN:
            raise BackendValidationError(
                f"default_embedding_model exceeds {MAX_MODEL_NAME_LEN} chars"
            )
        if not (self.request_timeout_s == self.request_timeout_s):  # NaN check
            raise BackendValidationError("request_timeout_s must be finite")
        if self.request_timeout_s <= 0 or self.request_timeout_s > 24 * 3600:
            raise BackendValidationError(
                "request_timeout_s must be in (0, 86400]"
            )
        if self.generate_timeout_s <= 0 or self.generate_timeout_s > 24 * 3600:
            raise BackendValidationError(
                "generate_timeout_s must be in (0, 86400]"
            )
        for k, v in self.extra_headers:
            if not k or not isinstance(k, str):
                raise BackendValidationError(
                    "extra_headers keys must be non-empty strings"
                )
            if len(v) > MAX_SETTINGS_STRING_LEN:
                raise BackendValidationError(
                    f"extra_headers[{k!r}] value exceeds "
                    f"{MAX_SETTINGS_STRING_LEN} chars"
                )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return only the **scalar** fields, suitable for storing as
        ``project.settings["ai_backend"]``. Headers are persisted via
        a separate sibling key (see ``store_backend_config``).
        """
        return {
            SETTING_KEY_PROVIDER: self.provider,
            SETTING_KEY_BASE_URL: self.base_url,
            SETTING_KEY_DEFAULT_MODEL: self.default_model,
            SETTING_KEY_DEFAULT_EMBEDDING_MODEL: self.default_embedding_model,
            SETTING_KEY_REQUEST_TIMEOUT: self.request_timeout_s,
            SETTING_KEY_GENERATE_TIMEOUT: self.generate_timeout_s,
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any] | None,
        *,
        extra_headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
    ) -> "BackendConfig":
        """Inverse of ``to_dict``.

        Headers are passed in separately because they are stored under a
        sibling key in ``project.settings`` (see notes at module top).
        """
        d = dict(d or {})
        return cls.new(
            provider=str(d.get(SETTING_KEY_PROVIDER, PROVIDER_OLLAMA) or PROVIDER_OLLAMA),
            base_url=str(d.get(SETTING_KEY_BASE_URL, DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL),
            default_model=str(d.get(SETTING_KEY_DEFAULT_MODEL, "") or ""),
            default_embedding_model=str(
                d.get(SETTING_KEY_DEFAULT_EMBEDDING_MODEL, "") or ""
            ),
            request_timeout_s=float(
                d.get(SETTING_KEY_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT_S)
                or DEFAULT_REQUEST_TIMEOUT_S
            ),
            generate_timeout_s=float(
                d.get(SETTING_KEY_GENERATE_TIMEOUT, DEFAULT_GENERATE_TIMEOUT_S)
                or DEFAULT_GENERATE_TIMEOUT_S
            ),
            extra_headers=extra_headers,
        )


@dataclass(frozen=True)
class ModelInfo:
    """One model installed on / known to a backend.

    Fields are deliberately small + provider-agnostic; ``raw`` carries
    the underlying provider's full payload so callers that need
    quantisation / family details can inspect it without needing the
    abstract layer to grow per-provider fields.
    """

    name: str
    provider: str
    kind: str = "generative"        # "generative" | "embedding" | "unknown"
    parameter_size: str = ""        # human label e.g. "3B", "8B"
    quantisation: str = ""          # e.g. "Q4_K_M"
    family: str = ""                # e.g. "llama", "qwen"
    size_bytes: int = 0             # on-disk size; 0 if unknown
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    system: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    # Provider-specific options pass through as-is. Use sparingly.
    options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRequest:
    model: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True)
class PullProgressEvent:
    """One progress event from Ollama's ``/api/pull`` NDJSON stream.

    Ollama emits events whose shape varies through the pull lifecycle:

      * ``{"status": "pulling manifest"}``
      * ``{"status": "pulling <digest>", "digest": "...", "total": N,
         "completed": M}``  (repeated as bytes arrive)
      * ``{"status": "verifying sha256 digest"}``
      * ``{"status": "writing manifest"}``
      * ``{"status": "removing any unused layers"}``
      * ``{"status": "success"}``

    Or, on failure, ``{"error": "..."}``. We canonicalise both into
    one type.
    """

    status: str
    digest: str = ""
    total: int = 0
    completed: int = 0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        """0.0 – 100.0 for layer downloads; 0.0 for non-byte phases."""
        if self.total <= 0:
            return 0.0
        return min(100.0, max(0.0, 100.0 * self.completed / self.total))

    @property
    def is_terminal_success(self) -> bool:
        return self.status == "success" and not self.error

    @property
    def is_error(self) -> bool:
        return bool(self.error) or self.status == "error"


@dataclass(frozen=True)
class PullSummary:
    """Outcome of a complete ``pull_model`` call.

    ``events`` is the raw ordered stream so callers (and the audit
    log) can replay exactly what the daemon reported. ``success``
    and ``error`` are the convenience flags for the common
    "did it finish?" check.
    """

    model: str
    provider: str
    success: bool
    error: str = ""
    events: tuple[PullProgressEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "success": self.success,
            "error": self.error,
            "events": [
                {
                    "status": e.status,
                    "digest": e.digest,
                    "total": e.total,
                    "completed": e.completed,
                    "error": e.error,
                    "percent": e.percent,
                }
                for e in self.events
            ],
        }


@dataclass(frozen=True)
class BackendHealth:
    ok: bool
    provider: str
    base_url: str
    detail: str = ""        # e.g. Ollama version string on success, or error
    error: str = ""         # populated when ok is False


# --------------------------------------------------------------------------- #
# HTTP transport
#
# All network I/O goes through a ``Transport`` callable so tests can
# substitute a stub without monkeypatching ``urllib``. The real default
# transport is small enough to read at a glance.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BackendError(f"Backend returned non-JSON body: {e}") from e


# Transport signature: (method, url, headers, body, timeout_s) → HTTPResponse.
# ``body`` is None for GET/HEAD; bytes for POST/PUT.
Transport = Callable[
    [str, str, Mapping[str, str], bytes | None, float],
    HTTPResponse,
]


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_s: float,
) -> HTTPResponse:
    """Default transport using stdlib ``urllib.request``.

    Translates connection / timeout / 5xx errors into
    ``BackendUnavailable``; surfaces 4xx with the body intact so the
    caller can decide whether to map to a validation error.
    """
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read()
            return HTTPResponse(
                status=resp.status,
                body=payload,
                headers=dict(resp.headers),
            )
    except urllib.error.HTTPError as e:
        # 4xx still returns; let the caller decide. Wrap 5xx so callers
        # can branch on the typed exception.
        try:
            payload = e.read() or b""
        except Exception:  # noqa: BLE001
            payload = b""
        if e.code >= 500:
            raise BackendUnavailable(
                f"{method} {url} → HTTP {e.code}: {payload[:500]!r}"
            ) from e
        return HTTPResponse(
            status=e.code,
            body=payload,
            headers=dict(e.headers or {}),
        )
    except urllib.error.URLError as e:  # connection refused / DNS / timeout
        raise BackendUnavailable(f"{method} {url}: {e.reason}") from e
    except TimeoutError as e:
        raise BackendUnavailable(f"{method} {url}: timed out") from e
    except OSError as e:
        raise BackendUnavailable(f"{method} {url}: {e}") from e


# --------------------------------------------------------------------------- #
# Backend ABC
# --------------------------------------------------------------------------- #


class ModelBackend(abc.ABC):
    """Abstract interface every backend implements.

    Methods take the ``BackendConfig`` explicitly rather than reading
    from a global, so a single process can fan out to multiple backends
    (e.g. local Ollama for embeddings + a LAN box for the heavy LLM).
    """

    name: str  # canonical provider id

    @abc.abstractmethod
    def health_check(
        self, config: BackendConfig, *, transport: Transport = urllib_transport
    ) -> BackendHealth: ...

    @abc.abstractmethod
    def list_models(
        self, config: BackendConfig, *, transport: Transport = urllib_transport
    ) -> list[ModelInfo]: ...

    @abc.abstractmethod
    def generate(
        self,
        config: BackendConfig,
        request: GenerationRequest,
        *,
        transport: Transport = urllib_transport,
    ) -> GenerationResponse: ...

    @abc.abstractmethod
    def embed(
        self,
        config: BackendConfig,
        request: EmbeddingRequest,
        *,
        transport: Transport = urllib_transport,
    ) -> EmbeddingResponse: ...

    # ------------------------------------------------------------------ #
    # Download manager (F8.11)
    #
    # Not every provider has a "pull this model into local storage"
    # concept (llama.cpp wants a path to a GGUF file the user got
    # themselves; transformers pulls from HF Hub via its own machinery).
    # We give the ABC a default that raises ``BackendValidationError``
    # so the abstract surface stays narrow and individual backends opt
    # in. Ollama overrides; the others can grow their own when added.
    # ------------------------------------------------------------------ #

    def pull_model(  # pragma: no cover - default raises; tested via Ollama
        self,
        config: BackendConfig,
        model: str,
        *,
        transport: Transport = urllib_transport,
        progress_callback: "Callable[[PullProgressEvent], None] | None" = None,
    ) -> "PullSummary":
        raise BackendValidationError(
            f"Backend {self.name!r} does not support pull_model"
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_BACKENDS: dict[str, ModelBackend] = {}


def register_backend(backend: ModelBackend) -> None:
    """Register a backend by its ``name``. Idempotent — registering the
    same name twice replaces the prior instance, which is what tests
    want when they swap in a fake.
    """
    if not getattr(backend, "name", ""):
        raise BackendValidationError(
            "Backend must have a non-empty .name attribute"
        )
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> ModelBackend:
    """Look up a registered backend or raise ``BackendValidationError``."""
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise BackendValidationError(
            f"No backend registered for provider {name!r}; "
            f"known: {sorted(_BACKENDS.keys())}"
        ) from e


def list_backends() -> list[str]:
    """Return the registered provider names, sorted."""
    return sorted(_BACKENDS.keys())


def backend_for_config(config: BackendConfig) -> ModelBackend:
    """Convenience: validate ``config`` and resolve its backend."""
    config.validate()
    return get_backend(config.provider)


# --------------------------------------------------------------------------- #
# Ollama implementation
# --------------------------------------------------------------------------- #


def _join_url(base: str, path: str) -> str:
    """Join base + path with exactly one slash. Avoids urllib.parse.urljoin's
    "drop the path component" surprise when ``base`` lacks a trailing /.
    """
    return base.rstrip("/") + "/" + path.lstrip("/")


def _request_json(
    config: BackendConfig,
    method: str,
    path: str,
    *,
    body: Any = None,
    timeout_s: float | None = None,
    transport: Transport = urllib_transport,
) -> Any:
    """Tiny helper: serialise body, attach headers, parse response JSON.

    On non-2xx, raises ``BackendValidationError`` for 4xx (the caller
    asked for something the backend rejected — usually "model not
    installed") and ``BackendUnavailable`` for 5xx (which the transport
    has *already* translated, but we keep the branch for robustness).
    """
    url = _join_url(config.base_url, path)
    headers: dict[str, str] = {"Accept": "application/json"}
    for k, v in config.extra_headers:
        headers[k] = v
    payload: bytes | None = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    timeout = float(timeout_s if timeout_s is not None else config.request_timeout_s)
    resp = transport(method, url, headers, payload, timeout)
    if resp.status >= 500:
        # Defensive: a custom transport might not raise.
        raise BackendUnavailable(
            f"{method} {url} → HTTP {resp.status}: {resp.body[:500]!r}"
        )
    if resp.status >= 400:
        raise BackendValidationError(
            f"{method} {url} → HTTP {resp.status}: {resp.body[:500]!r}"
        )
    return resp.json()


def _classify_ollama_kind(details: Mapping[str, Any]) -> str:
    """Crude generative-vs-embedding split.

    Ollama doesn't expose this directly; we infer from the model family
    and the ``embed`` modality hint when present. Wrong-but-safe: a
    misclassified model still works because callers always pass an
    explicit model name to ``generate`` / ``embed``.
    """
    fam = str(details.get("family", "")).lower()
    families = [str(f).lower() for f in (details.get("families") or [])]
    name_hint = " ".join(families + [fam])
    if any(t in name_hint for t in ("bge", "embed", "nomic", "mxbai", "arctic", "gte", "e5")):
        return "embedding"
    return "generative"


class OllamaBackend(ModelBackend):
    """Ollama HTTP API backend.

    Implements the four ABC methods using the small Ollama surface:

      - ``GET  /api/version``  for ``health_check``.
      - ``GET  /api/tags``     for ``list_models``.
      - ``POST /api/generate`` for ``generate`` (non-stream).
      - ``POST /api/embed``    for ``embed`` (batch).

    The non-streaming generate keeps the call site predictable; once
    F8.6 (whole-transcript pass) needs progressive output we'll add a
    ``generate_stream`` companion method.
    """

    name = PROVIDER_OLLAMA

    def health_check(
        self, config: BackendConfig, *, transport: Transport = urllib_transport
    ) -> BackendHealth:
        try:
            data = _request_json(
                config, "GET", "/api/version", transport=transport
            )
        except BackendUnavailable as e:
            return BackendHealth(
                ok=False,
                provider=self.name,
                base_url=config.base_url,
                error=str(e),
            )
        except BackendError as e:
            return BackendHealth(
                ok=False,
                provider=self.name,
                base_url=config.base_url,
                error=str(e),
            )
        version = ""
        if isinstance(data, dict):
            version = str(data.get("version", "") or "")
        return BackendHealth(
            ok=True,
            provider=self.name,
            base_url=config.base_url,
            detail=version,
        )

    def list_models(
        self, config: BackendConfig, *, transport: Transport = urllib_transport
    ) -> list[ModelInfo]:
        data = _request_json(
            config, "GET", "/api/tags", transport=transport
        )
        if not isinstance(data, dict):
            raise BackendError(
                f"Unexpected /api/tags payload: {type(data).__name__}"
            )
        models = data.get("models")
        if not isinstance(models, list):
            return []
        out: list[ModelInfo] = []
        for entry in models:
            if not isinstance(entry, dict):
                continue
            details = entry.get("details") or {}
            if not isinstance(details, dict):
                details = {}
            out.append(
                ModelInfo(
                    name=str(entry.get("name", "") or ""),
                    provider=self.name,
                    kind=_classify_ollama_kind(details),
                    parameter_size=str(details.get("parameter_size", "") or ""),
                    quantisation=str(details.get("quantization_level", "") or ""),
                    family=str(details.get("family", "") or ""),
                    size_bytes=int(entry.get("size", 0) or 0),
                    raw=dict(entry),
                )
            )
        # Sort for stable UI ordering. Names look like "llama3.2:3b".
        out.sort(key=lambda m: m.name.lower())
        return out

    def generate(
        self,
        config: BackendConfig,
        request: GenerationRequest,
        *,
        transport: Transport = urllib_transport,
    ) -> GenerationResponse:
        _validate_generation_request(request)
        body: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.system:
            body["system"] = request.system
        options = dict(request.options) if request.options else {}
        if request.temperature is not None:
            options["temperature"] = float(request.temperature)
        if request.max_tokens is not None:
            options["num_predict"] = int(request.max_tokens)
        if request.stop:
            options["stop"] = list(request.stop)
        if options:
            body["options"] = options
        data = _request_json(
            config,
            "POST",
            "/api/generate",
            body=body,
            timeout_s=config.generate_timeout_s,
            transport=transport,
        )
        if not isinstance(data, dict):
            raise BackendError(
                f"Unexpected /api/generate payload: {type(data).__name__}"
            )
        return GenerationResponse(
            text=str(data.get("response", "") or ""),
            model=str(data.get("model", request.model) or request.model),
            provider=self.name,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            total_duration_ns=int(data.get("total_duration", 0) or 0),
            raw=dict(data),
        )

    def pull_model(
        self,
        config: BackendConfig,
        model: str,
        *,
        transport: Transport = urllib_transport,
        progress_callback: Callable[[PullProgressEvent], None] | None = None,
    ) -> PullSummary:
        """Download a model via Ollama's ``/api/pull`` endpoint (F8.11).

        Sends the request with ``stream: true`` so the daemon emits an
        NDJSON event stream; the buffered transport returns the whole
        body once the pull completes. ``progress_callback`` (if given)
        is invoked for each parsed event in order.

        Errors map cleanly:

          * Daemon unreachable / 5xx → ``BackendUnavailable``.
          * 4xx (e.g. unknown model name) → ``BackendValidationError``.
          * Daemon emitted an ``{"error": "..."}`` event → returned in
            ``PullSummary.error`` with ``success=False``. We *don't*
            raise on a daemon-reported error because the caller often
            wants to surface the partial event list to the UI.
        """
        if not model.strip():
            raise BackendValidationError("pull_model: model is required")
        if len(model) > MAX_MODEL_NAME_LEN:
            raise BackendValidationError(
                f"pull_model: model exceeds {MAX_MODEL_NAME_LEN} chars"
            )
        url = _join_url(config.base_url, "/api/pull")
        payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        }
        for k, v in config.extra_headers:
            headers[k] = v
        # Pulls take *minutes* on slow links; reuse the long generate
        # timeout rather than the short request timeout.
        resp = transport(
            "POST", url, headers, payload, config.generate_timeout_s
        )
        if resp.status >= 500:
            raise BackendUnavailable(
                f"POST {url} → HTTP {resp.status}: {resp.body[:500]!r}"
            )
        if resp.status >= 400:
            raise BackendValidationError(
                f"POST {url} → HTTP {resp.status}: {resp.body[:500]!r}"
            )
        events = parse_pull_stream(resp.body)
        if progress_callback is not None:
            for ev in events:
                try:
                    progress_callback(ev)
                except Exception:  # noqa: BLE001
                    # The callback is observability-only; never let a
                    # buggy UI hook break a successful pull.
                    pass
        if not events:
            return PullSummary(
                model=model,
                provider=self.name,
                success=False,
                error="pull returned no events",
            )
        last = events[-1]
        # Find any error event in the stream (Ollama puts the error
        # on the last line, but be defensive).
        err = ""
        for ev in events:
            if ev.is_error:
                err = ev.error or "pull failed"
                break
        success = (not err) and last.is_terminal_success
        return PullSummary(
            model=model,
            provider=self.name,
            success=success,
            error=err if not success else "",
            events=tuple(events),
        )

    def embed(
        self,
        config: BackendConfig,
        request: EmbeddingRequest,
        *,
        transport: Transport = urllib_transport,
    ) -> EmbeddingResponse:
        _validate_embedding_request(request)
        body = {
            "model": request.model,
            # Ollama's /api/embed accepts ``input`` as either a string or
            # a list of strings. We always send a list for consistency.
            "input": list(request.inputs),
        }
        data = _request_json(
            config,
            "POST",
            "/api/embed",
            body=body,
            transport=transport,
        )
        if not isinstance(data, dict):
            raise BackendError(
                f"Unexpected /api/embed payload: {type(data).__name__}"
            )
        raw_vecs = data.get("embeddings")
        if not isinstance(raw_vecs, list):
            raise BackendError("/api/embed payload missing 'embeddings' list")
        vectors: list[tuple[float, ...]] = []
        for v in raw_vecs:
            if not isinstance(v, list):
                raise BackendError("/api/embed embedding entries must be lists")
            try:
                vectors.append(tuple(float(x) for x in v))
            except (TypeError, ValueError) as e:
                raise BackendError(
                    "/api/embed embedding contains non-numeric values"
                ) from e
        if len(vectors) != len(request.inputs):
            raise BackendError(
                f"/api/embed returned {len(vectors)} vectors for "
                f"{len(request.inputs)} inputs"
            )
        # Sanity: all vectors should have the same dim.
        if vectors:
            dim = len(vectors[0])
            for i, v in enumerate(vectors):
                if len(v) != dim:
                    raise BackendError(
                        f"/api/embed vector {i} has dim {len(v)} != {dim}"
                    )
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=str(data.get("model", request.model) or request.model),
            provider=self.name,
            raw=dict(data),
        )


# --------------------------------------------------------------------------- #
# Request validation (shared by Ollama; future backends reuse)
# --------------------------------------------------------------------------- #


def _validate_generation_request(req: GenerationRequest) -> None:
    if not req.model.strip():
        raise BackendValidationError("GenerationRequest.model is required")
    if len(req.model) > MAX_MODEL_NAME_LEN:
        raise BackendValidationError(
            f"GenerationRequest.model exceeds {MAX_MODEL_NAME_LEN} chars"
        )
    if not req.prompt:
        raise BackendValidationError("GenerationRequest.prompt is required")
    if len(req.prompt) > MAX_PROMPT_LEN:
        raise BackendValidationError(
            f"GenerationRequest.prompt exceeds {MAX_PROMPT_LEN} chars"
        )
    if req.temperature is not None:
        t = float(req.temperature)
        if t < 0 or t > 5 or t != t:
            raise BackendValidationError(
                "temperature must be a finite number in [0, 5]"
            )
    if req.max_tokens is not None and req.max_tokens < 0:
        raise BackendValidationError("max_tokens must be ≥ 0 if set")


def _validate_embedding_request(req: EmbeddingRequest) -> None:
    if not req.model.strip():
        raise BackendValidationError("EmbeddingRequest.model is required")
    if len(req.model) > MAX_MODEL_NAME_LEN:
        raise BackendValidationError(
            f"EmbeddingRequest.model exceeds {MAX_MODEL_NAME_LEN} chars"
        )
    if not req.inputs:
        raise BackendValidationError("EmbeddingRequest.inputs is empty")
    if len(req.inputs) > MAX_EMBED_BATCH:
        raise BackendValidationError(
            f"EmbeddingRequest.inputs has {len(req.inputs)} items "
            f"(>{MAX_EMBED_BATCH})"
        )
    for i, item in enumerate(req.inputs):
        if not isinstance(item, str):
            raise BackendValidationError(
                f"EmbeddingRequest.inputs[{i}] must be a string"
            )
        if not item:
            raise BackendValidationError(
                f"EmbeddingRequest.inputs[{i}] is empty"
            )
        if len(item) > MAX_EMBED_INPUT_LEN:
            raise BackendValidationError(
                f"EmbeddingRequest.inputs[{i}] exceeds "
                f"{MAX_EMBED_INPUT_LEN} chars"
            )


# --------------------------------------------------------------------------- #
# Pull stream parsing (F8.11)
#
# Kept as standalone functions (not methods) so any future backend
# wrapping a similar NDJSON pull endpoint can reuse them, and so tests
# exercise the parser without spinning up a backend.
# --------------------------------------------------------------------------- #


def parse_pull_event(line: str) -> PullProgressEvent:
    """Parse one NDJSON line from Ollama's ``/api/pull`` stream.

    Empty lines are a programmer error — the streaming caller is
    expected to filter them. Returns a ``PullProgressEvent`` with the
    ``raw`` payload preserved.
    """
    if not line or not line.strip():
        raise BackendError("parse_pull_event: empty line")
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise BackendError(f"pull event is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise BackendError(
            f"pull event must be an object, got {type(data).__name__}"
        )
    err = str(data.get("error", "") or "")
    if err:
        return PullProgressEvent(
            status=str(data.get("status", "error") or "error"),
            error=err,
            raw=dict(data),
        )
    total = int(data.get("total", 0) or 0)
    completed = int(data.get("completed", 0) or 0)
    return PullProgressEvent(
        status=str(data.get("status", "") or ""),
        digest=str(data.get("digest", "") or ""),
        total=total,
        completed=completed,
        raw=dict(data),
    )


def parse_pull_stream(body: bytes | str) -> list[PullProgressEvent]:
    """Parse the full NDJSON body from ``/api/pull`` into events.

    Skips blank lines (the daemon sometimes emits them between
    layers). Tolerant of a trailing newline. Order is preserved so
    callers can find the terminal ``success`` / ``error`` event by
    checking the last entry.
    """
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", errors="replace")
    else:
        text = body
    events: list[PullProgressEvent] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        events.append(parse_pull_event(line))
    return events


# --------------------------------------------------------------------------- #
# Project-settings integration
# --------------------------------------------------------------------------- #


def load_backend_config(project: Project) -> BackendConfig:
    """Read the AI-backend config from a Project's settings, applying
    defaults for any missing fields.

    Old projects (no ``ai_backend`` key) get the localhost-Ollama
    default; this is *connection* config only, not "we will use AI
    automatically" — F8.10 still gates AI invocations until enough
    coding has happened by hand.

    Headers (``ai_backend_headers``) live in a sibling top-level
    setting because ``Project.settings`` only allows one level of
    nested dicts; we splice them back in here.
    """
    settings = project.settings or {}
    raw = settings.get(SETTING_AI_BACKEND)
    if raw is None:
        raw_dict: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        raw_dict = dict(raw)
    else:
        raise BackendValidationError(
            f"project.settings[{SETTING_AI_BACKEND!r}] must be an object"
        )
    headers_raw = settings.get(SETTING_AI_BACKEND_HEADERS)
    if headers_raw is None:
        headers: Mapping[str, str] = {}
    elif isinstance(headers_raw, Mapping):
        headers = {str(k): str(v) for k, v in headers_raw.items()}
    else:
        raise BackendValidationError(
            f"project.settings[{SETTING_AI_BACKEND_HEADERS!r}] "
            "must be an object"
        )
    return BackendConfig.from_dict(raw_dict, extra_headers=headers)


def store_backend_config(project: Project, config: BackendConfig) -> None:
    """Persist a backend config into ``project.settings``.

    Mutates the project in place but does not save it; callers
    sandwich this between ``load_project`` / ``save_project`` so the
    write is atomic at their granularity. Validates first so a bad
    config never lands on disk.

    Headers go into the sibling top-level setting
    ``ai_backend_headers`` (see notes at module top); they're removed
    when the config has none, so settings stay tidy.
    """
    config.validate()
    settings = dict(project.settings or {})
    settings[SETTING_AI_BACKEND] = config.to_dict()
    if config.extra_headers:
        settings[SETTING_AI_BACKEND_HEADERS] = {
            k: v for k, v in config.extra_headers
        }
    else:
        settings.pop(SETTING_AI_BACKEND_HEADERS, None)
    # ``apply_update`` re-runs the project's own settings validation,
    # which catches over-large blobs etc. before they hit the disk.
    project.apply_update({"settings": settings})


# --------------------------------------------------------------------------- #
# Auto-register the Ollama backend
# --------------------------------------------------------------------------- #

register_backend(OllamaBackend())
