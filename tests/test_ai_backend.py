"""Tests for scribe.ai_backend (F8.1).

Covers:

  * BackendConfig validation, defaults, dict round-trip.
  * Request validation (GenerationRequest, EmbeddingRequest).
  * Registry: register / get / list, error on unknown.
  * OllamaBackend: health_check, list_models, generate, embed — all
    with a stub transport so no real HTTP fires.
  * load_backend_config / store_backend_config round-trip via
    Project.settings.
  * urllib_transport's error mapping (4xx surfaces; 5xx → BackendUnavailable).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from scribe.ai_backend import (
    DEFAULT_GENERATE_TIMEOUT_S,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT_S,
    KNOWN_PROVIDERS,
    MAX_BASE_URL_LEN,
    MAX_EMBED_BATCH,
    MAX_EMBED_INPUT_LEN,
    MAX_MODEL_NAME_LEN,
    MAX_PROMPT_LEN,
    PROVIDER_LLAMA_CPP,
    PROVIDER_OLLAMA,
    PROVIDER_TRANSFORMERS,
    SETTING_AI_BACKEND,
    SETTING_AI_BACKEND_HEADERS,
    SETTING_KEY_BASE_URL,
    SETTING_KEY_DEFAULT_EMBEDDING_MODEL,
    SETTING_KEY_DEFAULT_MODEL,
    SETTING_KEY_GENERATE_TIMEOUT,
    SETTING_KEY_PROVIDER,
    SETTING_KEY_REQUEST_TIMEOUT,
    BackendConfig,
    BackendError,
    BackendHealth,
    BackendUnavailable,
    BackendValidationError,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    HTTPResponse,
    ModelBackend,
    OllamaBackend,
    backend_for_config,
    get_backend,
    list_backends,
    load_backend_config,
    register_backend,
    store_backend_config,
    urllib_transport,
)
from scribe.projects import Project


# --------------------------------------------------------------------------- #
# Stub transport
# --------------------------------------------------------------------------- #


class StubTransport:
    """Records calls and returns canned responses keyed by (method, path)."""

    def __init__(self, routes: dict[tuple[str, str], HTTPResponse]) -> None:
        self.routes = dict(routes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HTTPResponse:
        # Strip the base for routing so tests don't have to repeat it.
        path = url
        for base in ("http://127.0.0.1:11434", "http://example.test"):
            if path.startswith(base):
                path = path[len(base):]
                break
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        try:
            return self.routes[(method, path)]
        except KeyError as e:
            raise AssertionError(
                f"Unexpected transport call: {method} {url} (path={path}); "
                f"known routes: {sorted(self.routes.keys())}"
            ) from e


def _ok(body: dict[str, Any]) -> HTTPResponse:
    return HTTPResponse(status=200, body=json.dumps(body).encode("utf-8"))


# --------------------------------------------------------------------------- #
# BackendConfig
# --------------------------------------------------------------------------- #


class TestBackendConfigDefaults:
    def test_new_with_no_args_uses_localhost_ollama(self) -> None:
        c = BackendConfig.new()
        assert c.provider == PROVIDER_OLLAMA
        assert c.base_url == DEFAULT_OLLAMA_BASE_URL
        assert c.default_model == ""
        assert c.default_embedding_model == ""
        assert c.request_timeout_s == DEFAULT_REQUEST_TIMEOUT_S
        assert c.generate_timeout_s == DEFAULT_GENERATE_TIMEOUT_S
        assert c.extra_headers == ()

    def test_known_providers_listed(self) -> None:
        for p in (PROVIDER_OLLAMA, PROVIDER_LLAMA_CPP, PROVIDER_TRANSFORMERS):
            assert p in KNOWN_PROVIDERS


class TestBackendConfigValidation:
    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(provider="acme-magic")

    def test_empty_base_url_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(base_url="")

    def test_non_http_base_url_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(base_url="ftp://example.com")

    def test_overlong_base_url_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(
                base_url="http://" + "a" * (MAX_BASE_URL_LEN + 1)
            )

    def test_overlong_default_model_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(default_model="x" * (MAX_MODEL_NAME_LEN + 1))

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(request_timeout_s=-1)

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(request_timeout_s=0)

    def test_huge_timeout_rejected(self) -> None:
        with pytest.raises(BackendValidationError):
            BackendConfig.new(request_timeout_s=10 ** 9)

    def test_extra_headers_dict_round_trip(self) -> None:
        c = BackendConfig.new(extra_headers={"Authorization": "Bearer x"})
        assert c.extra_headers == (("Authorization", "Bearer x"),)


class TestBackendConfigSerialisation:
    def test_round_trip_scalars(self) -> None:
        c = BackendConfig.new(
            provider=PROVIDER_OLLAMA,
            base_url="http://example.test:11434",
            default_model="llama3.2:3b",
            default_embedding_model="bge-m3",
            request_timeout_s=15,
            generate_timeout_s=300,
        )
        d = c.to_dict()
        # Stable, documented keys land in the dict.
        assert d[SETTING_KEY_PROVIDER] == PROVIDER_OLLAMA
        assert d[SETTING_KEY_BASE_URL] == "http://example.test:11434"
        assert d[SETTING_KEY_DEFAULT_MODEL] == "llama3.2:3b"
        assert d[SETTING_KEY_DEFAULT_EMBEDDING_MODEL] == "bge-m3"
        assert d[SETTING_KEY_REQUEST_TIMEOUT] == 15.0
        assert d[SETTING_KEY_GENERATE_TIMEOUT] == 300.0
        # ``to_dict`` deliberately excludes headers (they live in a
        # sibling settings key).
        assert "extra_headers" not in d
        # Round-trip back.
        c2 = BackendConfig.from_dict(d)
        assert c2 == c

    def test_from_dict_with_none_uses_defaults(self) -> None:
        c = BackendConfig.from_dict(None)
        assert c == BackendConfig.new()

    def test_from_dict_ignores_unknown_keys(self) -> None:
        c = BackendConfig.from_dict(
            {SETTING_KEY_PROVIDER: PROVIDER_OLLAMA, "future_field": 42}
        )
        assert c.provider == PROVIDER_OLLAMA

    def test_from_dict_accepts_extra_headers_kwarg(self) -> None:
        c = BackendConfig.from_dict({}, extra_headers={"X-Token": "abc"})
        assert c.extra_headers == (("X-Token", "abc"),)


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #


class TestGenerationRequestValidation:
    def test_empty_model_rejected(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        with pytest.raises(BackendValidationError):
            _validate_generation_request(
                GenerationRequest(model="", prompt="hi")
            )

    def test_empty_prompt_rejected(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        with pytest.raises(BackendValidationError):
            _validate_generation_request(
                GenerationRequest(model="m", prompt="")
            )

    def test_overlong_prompt_rejected(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        with pytest.raises(BackendValidationError):
            _validate_generation_request(
                GenerationRequest(
                    model="m", prompt="x" * (MAX_PROMPT_LEN + 1)
                )
            )

    def test_temperature_out_of_range_rejected(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        for bad in (-0.1, 5.1, float("nan")):
            with pytest.raises(BackendValidationError):
                _validate_generation_request(
                    GenerationRequest(model="m", prompt="p", temperature=bad)
                )

    def test_negative_max_tokens_rejected(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        with pytest.raises(BackendValidationError):
            _validate_generation_request(
                GenerationRequest(model="m", prompt="p", max_tokens=-1)
            )

    def test_valid_request_passes(self) -> None:
        from scribe.ai_backend import _validate_generation_request

        _validate_generation_request(
            GenerationRequest(
                model="llama3.2:3b",
                prompt="Hello",
                temperature=0.2,
                max_tokens=64,
            )
        )


class TestEmbeddingRequestValidation:
    def test_empty_inputs_rejected(self) -> None:
        from scribe.ai_backend import _validate_embedding_request

        with pytest.raises(BackendValidationError):
            _validate_embedding_request(
                EmbeddingRequest(model="bge-m3", inputs=())
            )

    def test_too_many_inputs_rejected(self) -> None:
        from scribe.ai_backend import _validate_embedding_request

        with pytest.raises(BackendValidationError):
            _validate_embedding_request(
                EmbeddingRequest(
                    model="bge-m3",
                    inputs=tuple("x" for _ in range(MAX_EMBED_BATCH + 1)),
                )
            )

    def test_empty_string_input_rejected(self) -> None:
        from scribe.ai_backend import _validate_embedding_request

        with pytest.raises(BackendValidationError):
            _validate_embedding_request(
                EmbeddingRequest(model="bge-m3", inputs=("ok", ""))
            )

    def test_overlong_input_rejected(self) -> None:
        from scribe.ai_backend import _validate_embedding_request

        with pytest.raises(BackendValidationError):
            _validate_embedding_request(
                EmbeddingRequest(
                    model="bge-m3",
                    inputs=("x" * (MAX_EMBED_INPUT_LEN + 1),),
                )
            )

    def test_non_string_input_rejected(self) -> None:
        from scribe.ai_backend import _validate_embedding_request

        with pytest.raises(BackendValidationError):
            _validate_embedding_request(
                EmbeddingRequest(model="bge-m3", inputs=(42,))  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_ollama_is_registered_by_default(self) -> None:
        b = get_backend(PROVIDER_OLLAMA)
        assert isinstance(b, OllamaBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(BackendValidationError):
            get_backend("does-not-exist")

    def test_list_backends_includes_ollama(self) -> None:
        assert PROVIDER_OLLAMA in list_backends()

    def test_register_backend_requires_name(self) -> None:
        class Anon(ModelBackend):
            name = ""

            def health_check(self, config, *, transport=urllib_transport):
                return BackendHealth(ok=True, provider="x", base_url=config.base_url)

            def list_models(self, config, *, transport=urllib_transport):
                return []

            def generate(self, config, request, *, transport=urllib_transport):
                raise NotImplementedError

            def embed(self, config, request, *, transport=urllib_transport):
                raise NotImplementedError

        with pytest.raises(BackendValidationError):
            register_backend(Anon())

    def test_backend_for_config_validates_first(self) -> None:
        with pytest.raises(BackendValidationError):
            backend_for_config(BackendConfig(provider="acme-magic"))

    def test_backend_for_config_returns_registered_backend(self) -> None:
        c = BackendConfig.new()
        b = backend_for_config(c)
        assert b.name == PROVIDER_OLLAMA


# --------------------------------------------------------------------------- #
# OllamaBackend.health_check
# --------------------------------------------------------------------------- #


class TestOllamaHealthCheck:
    def test_returns_ok_with_version(self) -> None:
        transport = StubTransport(
            {("GET", "/api/version"): _ok({"version": "0.5.7"})}
        )
        c = BackendConfig.new()
        h = OllamaBackend().health_check(c, transport=transport)
        assert h.ok is True
        assert h.detail == "0.5.7"
        assert h.provider == PROVIDER_OLLAMA
        # And the transport actually got called once on /api/version.
        assert len(transport.calls) == 1
        assert transport.calls[0]["method"] == "GET"
        assert transport.calls[0]["path"] == "/api/version"

    def test_returns_not_ok_when_unavailable(self) -> None:
        def boom(*args, **kwargs):
            raise BackendUnavailable("connection refused")

        c = BackendConfig.new()
        h = OllamaBackend().health_check(c, transport=boom)
        assert h.ok is False
        assert "connection refused" in h.error
        assert h.provider == PROVIDER_OLLAMA
        assert h.base_url == c.base_url

    def test_returns_not_ok_on_4xx(self) -> None:
        transport = StubTransport(
            {("GET", "/api/version"): HTTPResponse(status=404, body=b"not found")}
        )
        c = BackendConfig.new()
        h = OllamaBackend().health_check(c, transport=transport)
        assert h.ok is False
        assert "404" in h.error or "not found" in h.error


# --------------------------------------------------------------------------- #
# OllamaBackend.list_models
# --------------------------------------------------------------------------- #


class TestOllamaListModels:
    def test_parses_tag_payload(self) -> None:
        payload = {
            "models": [
                {
                    "name": "llama3.2:3b",
                    "size": 2_000_000_000,
                    "details": {
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M",
                    },
                },
                {
                    "name": "bge-m3:latest",
                    "size": 569_000_000,
                    "details": {
                        "family": "bert",
                        "families": ["bge"],
                        "parameter_size": "568M",
                        "quantization_level": "F16",
                    },
                },
            ]
        }
        transport = StubTransport({("GET", "/api/tags"): _ok(payload)})
        c = BackendConfig.new()
        models = OllamaBackend().list_models(c, transport=transport)
        # Sorted by name (case-insensitive), so bge comes first.
        assert [m.name for m in models] == ["bge-m3:latest", "llama3.2:3b"]
        bge = models[0]
        assert bge.kind == "embedding"
        assert bge.parameter_size == "568M"
        assert bge.quantisation == "F16"
        assert bge.size_bytes == 569_000_000
        llama = models[1]
        assert llama.kind == "generative"
        assert llama.family == "llama"

    def test_handles_empty_or_missing_models(self) -> None:
        transport = StubTransport({("GET", "/api/tags"): _ok({"models": []})})
        c = BackendConfig.new()
        assert OllamaBackend().list_models(c, transport=transport) == []

        transport2 = StubTransport({("GET", "/api/tags"): _ok({})})
        assert OllamaBackend().list_models(c, transport=transport2) == []

    def test_skips_non_dict_entries(self) -> None:
        transport = StubTransport(
            {
                ("GET", "/api/tags"): _ok(
                    {
                        "models": [
                            "junk",
                            {"name": "good:1", "details": {"family": "llama"}},
                        ]
                    }
                )
            }
        )
        c = BackendConfig.new()
        models = OllamaBackend().list_models(c, transport=transport)
        assert [m.name for m in models] == ["good:1"]

    def test_4xx_raises_validation_error(self) -> None:
        transport = StubTransport(
            {("GET", "/api/tags"): HTTPResponse(status=404, body=b"nope")}
        )
        c = BackendConfig.new()
        with pytest.raises(BackendValidationError):
            OllamaBackend().list_models(c, transport=transport)

    def test_5xx_raises_unavailable(self) -> None:
        transport = StubTransport(
            {("GET", "/api/tags"): HTTPResponse(status=502, body=b"bad gateway")}
        )
        c = BackendConfig.new()
        with pytest.raises(BackendUnavailable):
            OllamaBackend().list_models(c, transport=transport)


# --------------------------------------------------------------------------- #
# OllamaBackend.generate
# --------------------------------------------------------------------------- #


class TestOllamaGenerate:
    def test_happy_path(self) -> None:
        payload = {
            "model": "llama3.2:3b",
            "response": "Hello, world.",
            "prompt_eval_count": 4,
            "eval_count": 8,
            "total_duration": 1_234_567,
        }
        transport = StubTransport({("POST", "/api/generate"): _ok(payload)})
        c = BackendConfig.new()
        resp = OllamaBackend().generate(
            c,
            GenerationRequest(
                model="llama3.2:3b",
                prompt="Hi",
                temperature=0.2,
                max_tokens=32,
                stop=("</s>",),
            ),
            transport=transport,
        )
        assert resp.text == "Hello, world."
        assert resp.model == "llama3.2:3b"
        assert resp.provider == PROVIDER_OLLAMA
        assert resp.prompt_tokens == 4
        assert resp.completion_tokens == 8
        # The body sent to /api/generate carries the right shape.
        body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        assert body["model"] == "llama3.2:3b"
        assert body["prompt"] == "Hi"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.2
        assert body["options"]["num_predict"] == 32
        assert body["options"]["stop"] == ["</s>"]

    def test_uses_generate_timeout_not_request_timeout(self) -> None:
        transport = StubTransport(
            {("POST", "/api/generate"): _ok({"response": "ok"})}
        )
        c = BackendConfig.new(request_timeout_s=10, generate_timeout_s=300)
        OllamaBackend().generate(
            c,
            GenerationRequest(model="m", prompt="hi"),
            transport=transport,
        )
        assert transport.calls[0]["timeout_s"] == 300.0

    def test_includes_system_prompt_when_set(self) -> None:
        transport = StubTransport(
            {("POST", "/api/generate"): _ok({"response": "ok"})}
        )
        c = BackendConfig.new()
        OllamaBackend().generate(
            c,
            GenerationRequest(model="m", prompt="hi", system="Be terse."),
            transport=transport,
        )
        body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        assert body["system"] == "Be terse."

    def test_4xx_from_ollama_raises_validation(self) -> None:
        transport = StubTransport(
            {("POST", "/api/generate"): HTTPResponse(status=404, body=b"model not found")}
        )
        c = BackendConfig.new()
        with pytest.raises(BackendValidationError):
            OllamaBackend().generate(
                c,
                GenerationRequest(model="missing", prompt="hi"),
                transport=transport,
            )

    def test_invalid_request_rejected_before_network(self) -> None:
        # Should never call the transport.
        called: list[Any] = []

        def transport(*args, **kwargs):
            called.append(args)
            return _ok({})

        c = BackendConfig.new()
        with pytest.raises(BackendValidationError):
            OllamaBackend().generate(
                c, GenerationRequest(model="", prompt="x"), transport=transport
            )
        assert called == []


# --------------------------------------------------------------------------- #
# OllamaBackend.embed
# --------------------------------------------------------------------------- #


class TestOllamaEmbed:
    def test_happy_path(self) -> None:
        payload = {
            "model": "bge-m3:latest",
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
        }
        transport = StubTransport({("POST", "/api/embed"): _ok(payload)})
        c = BackendConfig.new()
        resp = OllamaBackend().embed(
            c,
            EmbeddingRequest(model="bge-m3:latest", inputs=("hello", "world")),
            transport=transport,
        )
        assert resp.dim == 3
        assert resp.vectors == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        assert resp.provider == PROVIDER_OLLAMA
        body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        assert body["model"] == "bge-m3:latest"
        assert body["input"] == ["hello", "world"]

    def test_dim_zero_when_no_inputs_returned(self) -> None:
        # Defensive: Ollama may return [] if it considered nothing
        # embeddable; we require inputs upfront so this should never
        # happen, but the property handles it.
        r = EmbeddingResponse(
            vectors=(), model="m", provider=PROVIDER_OLLAMA, raw={}
        )
        assert r.dim == 0

    def test_mismatched_count_raises(self) -> None:
        transport = StubTransport(
            {
                ("POST", "/api/embed"): _ok(
                    {"embeddings": [[0.1, 0.2]], "model": "bge-m3"}
                )
            }
        )
        c = BackendConfig.new()
        with pytest.raises(BackendError):
            OllamaBackend().embed(
                c,
                EmbeddingRequest(model="bge-m3", inputs=("a", "b")),
                transport=transport,
            )

    def test_inconsistent_dim_raises(self) -> None:
        transport = StubTransport(
            {
                ("POST", "/api/embed"): _ok(
                    {
                        "embeddings": [[0.1, 0.2], [0.3]],
                        "model": "bge-m3",
                    }
                )
            }
        )
        c = BackendConfig.new()
        with pytest.raises(BackendError):
            OllamaBackend().embed(
                c,
                EmbeddingRequest(model="bge-m3", inputs=("a", "b")),
                transport=transport,
            )

    def test_non_numeric_vector_raises(self) -> None:
        transport = StubTransport(
            {
                ("POST", "/api/embed"): _ok(
                    {"embeddings": [["nan-as-string"]]}
                )
            }
        )
        c = BackendConfig.new()
        with pytest.raises(BackendError):
            OllamaBackend().embed(
                c,
                EmbeddingRequest(model="bge-m3", inputs=("a",)),
                transport=transport,
            )

    def test_missing_embeddings_key_raises(self) -> None:
        transport = StubTransport({("POST", "/api/embed"): _ok({})})
        c = BackendConfig.new()
        with pytest.raises(BackendError):
            OllamaBackend().embed(
                c,
                EmbeddingRequest(model="bge-m3", inputs=("a",)),
                transport=transport,
            )


# --------------------------------------------------------------------------- #
# Project settings round-trip
# --------------------------------------------------------------------------- #


class TestProjectSettingsRoundTrip:
    def test_load_default_when_missing(self) -> None:
        p = Project.new(name="p")
        c = load_backend_config(p)
        assert c == BackendConfig.new()

    def test_store_then_load(self) -> None:
        p = Project.new(name="p")
        c = BackendConfig.new(
            base_url="http://lan-box:11434",
            default_model="llama3.2:3b",
            default_embedding_model="bge-m3",
            request_timeout_s=20,
            generate_timeout_s=240,
        )
        store_backend_config(p, c)
        # The settings dict carries the canonical key.
        assert SETTING_AI_BACKEND in p.settings
        assert p.settings[SETTING_AI_BACKEND][SETTING_KEY_BASE_URL] == "http://lan-box:11434"
        # No headers were set, so no headers key should exist.
        assert SETTING_AI_BACKEND_HEADERS not in p.settings
        # Round-trip through JSON to mimic the on-disk path.
        clone = Project.from_dict(json.loads(json.dumps(p.to_dict())))
        c2 = load_backend_config(clone)
        assert c2 == c

    def test_store_then_load_with_headers(self) -> None:
        p = Project.new(name="p")
        c = BackendConfig.new(
            base_url="http://lan-box:11434",
            extra_headers={"X-Token": "abc", "X-Trace": "42"},
        )
        store_backend_config(p, c)
        assert p.settings[SETTING_AI_BACKEND_HEADERS] == {
            "X-Token": "abc",
            "X-Trace": "42",
        }
        clone = Project.from_dict(json.loads(json.dumps(p.to_dict())))
        c2 = load_backend_config(clone)
        assert dict(c2.extra_headers) == {"X-Token": "abc", "X-Trace": "42"}

    def test_store_clears_stale_headers(self) -> None:
        p = Project.new(name="p")
        c1 = BackendConfig.new(extra_headers={"X-Token": "abc"})
        store_backend_config(p, c1)
        assert SETTING_AI_BACKEND_HEADERS in p.settings
        # Storing a header-less config should drop the stale key.
        c2 = BackendConfig.new()
        store_backend_config(p, c2)
        assert SETTING_AI_BACKEND_HEADERS not in p.settings

    def test_load_rejects_non_object_headers(self) -> None:
        p = Project.new(
            name="p",
            settings={SETTING_AI_BACKEND_HEADERS: "string-not-dict"},
        )
        with pytest.raises(BackendValidationError):
            load_backend_config(p)

    def test_store_rejects_invalid_config(self) -> None:
        p = Project.new(name="p")
        bad = BackendConfig(provider="acme-magic")  # bypasses .new validation
        with pytest.raises(BackendValidationError):
            store_backend_config(p, bad)

    def test_load_rejects_non_object(self) -> None:
        p = Project.new(name="p", settings={SETTING_AI_BACKEND: "string"})
        with pytest.raises(BackendValidationError):
            load_backend_config(p)


# --------------------------------------------------------------------------- #
# urllib_transport (real default)
# --------------------------------------------------------------------------- #


class TestUrllibTransport:
    """Exercises the default transport without hitting a real network.

    Uses ``monkeypatch`` to replace ``urllib.request.urlopen`` with stubs
    that simulate success, 4xx, 5xx, and connection errors.
    """

    def test_2xx_returns_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import ai_backend as mod

        class _Resp:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok":true}'

        def fake_urlopen(req, timeout):
            assert req.method == "GET"
            assert req.full_url == "http://x/y"
            return _Resp()

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        r = urllib_transport("GET", "http://x/y", {}, None, 5.0)
        assert r.status == 200
        assert r.json() == {"ok": True}

    def test_4xx_returns_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import ai_backend as mod
        import urllib.error

        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None
            )

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        r = urllib_transport("GET", "http://x/y", {}, None, 5.0)
        assert r.status == 404

    def test_5xx_raises_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import ai_backend as mod
        import urllib.error

        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(
                req.full_url, 502, "Bad Gateway", {}, None
            )

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(BackendUnavailable):
            urllib_transport("GET", "http://x/y", {}, None, 5.0)

    def test_connection_refused_raises_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import ai_backend as mod
        import urllib.error

        def fake_urlopen(req, timeout):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(BackendUnavailable):
            urllib_transport("GET", "http://x/y", {}, None, 5.0)

    def test_timeout_raises_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scribe import ai_backend as mod

        def fake_urlopen(req, timeout):
            raise TimeoutError("timed out")

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(BackendUnavailable):
            urllib_transport("GET", "http://x/y", {}, None, 0.1)


# --------------------------------------------------------------------------- #
# HTTPResponse helpers
# --------------------------------------------------------------------------- #


class TestHTTPResponseJson:
    def test_returns_none_on_empty_body(self) -> None:
        assert HTTPResponse(200, b"").json() is None

    def test_raises_on_non_json(self) -> None:
        with pytest.raises(BackendError):
            HTTPResponse(200, b"\xffnot json").json()

    def test_parses_json_object(self) -> None:
        assert HTTPResponse(200, b'{"a":1}').json() == {"a": 1}


# --------------------------------------------------------------------------- #
# BackendConfig.extra_headers attaches to outbound requests
# --------------------------------------------------------------------------- #


class TestExtraHeaders:
    def test_extra_headers_attached(self) -> None:
        transport = StubTransport(
            {("GET", "/api/version"): _ok({"version": "1.0"})}
        )
        c = BackendConfig.new(extra_headers={"X-Token": "abc"})
        OllamaBackend().health_check(c, transport=transport)
        assert transport.calls[0]["headers"].get("X-Token") == "abc"


# --------------------------------------------------------------------------- #
# Pull stream parsing (F8.11)
# --------------------------------------------------------------------------- #


from scribe.ai_backend import (  # noqa: E402  -- re-imported for grouping
    PullProgressEvent,
    PullSummary,
    parse_pull_event,
    parse_pull_stream,
)


class TestParsePullEvent:
    def test_pulling_manifest_event(self) -> None:
        ev = parse_pull_event('{"status": "pulling manifest"}')
        assert ev.status == "pulling manifest"
        assert ev.total == 0
        assert ev.completed == 0
        assert ev.error == ""
        assert ev.is_terminal_success is False
        assert ev.is_error is False

    def test_layer_progress_event(self) -> None:
        ev = parse_pull_event(
            '{"status":"pulling abc","digest":"sha256:abc",'
            '"total":2048,"completed":1024}'
        )
        assert ev.status == "pulling abc"
        assert ev.digest == "sha256:abc"
        assert ev.total == 2048
        assert ev.completed == 1024
        assert ev.percent == 50.0

    def test_terminal_success_event(self) -> None:
        ev = parse_pull_event('{"status": "success"}')
        assert ev.is_terminal_success is True
        assert ev.is_error is False

    def test_error_event(self) -> None:
        ev = parse_pull_event('{"error": "manifest not found"}')
        assert ev.is_error is True
        assert ev.error == "manifest not found"
        # Falls back to "error" status when none is supplied.
        assert ev.status == "error"

    def test_percent_zero_when_total_unknown(self) -> None:
        ev = parse_pull_event('{"status": "verifying"}')
        assert ev.percent == 0.0

    def test_percent_clamped_to_100(self) -> None:
        # Defensive: if Ollama ever reports completed > total we clamp.
        ev = parse_pull_event(
            '{"status":"x","total":100,"completed":150}'
        )
        assert ev.percent == 100.0

    def test_empty_line_rejected(self) -> None:
        with pytest.raises(BackendError):
            parse_pull_event("")
        with pytest.raises(BackendError):
            parse_pull_event("   \n")

    def test_non_json_rejected(self) -> None:
        with pytest.raises(BackendError):
            parse_pull_event("not json")

    def test_non_object_rejected(self) -> None:
        with pytest.raises(BackendError):
            parse_pull_event("[1, 2]")


class TestParsePullStream:
    def test_full_stream_round_trip(self) -> None:
        body = (
            b'{"status": "pulling manifest"}\n'
            b'{"status": "pulling l1", "digest": "sha256:1", "total": 100, "completed": 50}\n'
            b'{"status": "pulling l1", "digest": "sha256:1", "total": 100, "completed": 100}\n'
            b'{"status": "verifying sha256 digest"}\n'
            b'{"status": "writing manifest"}\n'
            b'{"status": "success"}\n'
        )
        events = parse_pull_stream(body)
        assert len(events) == 6
        assert events[0].status == "pulling manifest"
        assert events[-1].is_terminal_success
        # Bytes vs str both accepted.
        events2 = parse_pull_stream(body.decode("utf-8"))
        assert [e.status for e in events] == [e.status for e in events2]

    def test_blank_lines_skipped(self) -> None:
        body = b'\n{"status":"a"}\n\n\n{"status":"b"}\n'
        events = parse_pull_stream(body)
        assert [e.status for e in events] == ["a", "b"]

    def test_empty_body_returns_empty_list(self) -> None:
        assert parse_pull_stream(b"") == []
        assert parse_pull_stream("\n\n  \n") == []

    def test_error_inside_stream_preserved(self) -> None:
        body = (
            b'{"status": "pulling manifest"}\n'
            b'{"error": "model not found"}\n'
        )
        events = parse_pull_stream(body)
        assert len(events) == 2
        assert events[1].is_error
        assert events[1].error == "model not found"


# --------------------------------------------------------------------------- #
# OllamaBackend.pull_model (F8.11)
# --------------------------------------------------------------------------- #


class TestOllamaPullModel:
    def _ok_stream(self) -> bytes:
        return (
            b'{"status": "pulling manifest"}\n'
            b'{"status": "pulling l1", "digest": "sha256:1", "total": 10, "completed": 10}\n'
            b'{"status": "writing manifest"}\n'
            b'{"status": "success"}\n'
        )

    def test_happy_path_returns_success_summary(self) -> None:
        body = self._ok_stream()
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, body)})
        c = BackendConfig.new()
        summary = OllamaBackend().pull_model(c, "llama3.2:3b", transport=transport)
        assert summary.success is True
        assert summary.error == ""
        assert summary.model == "llama3.2:3b"
        assert summary.provider == PROVIDER_OLLAMA
        assert len(summary.events) == 4
        # Body contains the model name + stream flag.
        sent = json.loads(transport.calls[0]["body"])
        assert sent == {"model": "llama3.2:3b", "stream": True}
        # Long timeout used (generate_timeout, not request_timeout).
        assert transport.calls[0]["timeout_s"] == c.generate_timeout_s

    def test_progress_callback_invoked_per_event(self) -> None:
        body = self._ok_stream()
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, body)})
        c = BackendConfig.new()
        seen: list[str] = []
        OllamaBackend().pull_model(
            c,
            "llama3.2:3b",
            transport=transport,
            progress_callback=lambda ev: seen.append(ev.status),
        )
        assert seen == [
            "pulling manifest",
            "pulling l1",
            "writing manifest",
            "success",
        ]

    def test_progress_callback_exception_is_swallowed(self) -> None:
        body = self._ok_stream()
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, body)})
        c = BackendConfig.new()

        def boom(_ev: PullProgressEvent) -> None:
            raise RuntimeError("UI bug")

        # Doesn't propagate.
        summary = OllamaBackend().pull_model(
            c, "x:1b", transport=transport, progress_callback=boom
        )
        assert summary.success is True

    def test_daemon_error_event_yields_failed_summary(self) -> None:
        body = (
            b'{"status": "pulling manifest"}\n'
            b'{"error": "model not found"}\n'
        )
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, body)})
        summary = OllamaBackend().pull_model(
            BackendConfig.new(), "nope:99", transport=transport
        )
        assert summary.success is False
        assert "model not found" in summary.error
        assert len(summary.events) == 2

    def test_empty_stream_yields_failed_summary(self) -> None:
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, b"")})
        summary = OllamaBackend().pull_model(
            BackendConfig.new(), "x:1b", transport=transport
        )
        assert summary.success is False
        assert "no events" in summary.error

    def test_4xx_raises_validation_error(self) -> None:
        transport = StubTransport(
            {("POST", "/api/pull"): HTTPResponse(404, b"not found")}
        )
        with pytest.raises(BackendValidationError):
            OllamaBackend().pull_model(
                BackendConfig.new(), "missing:1b", transport=transport
            )

    def test_5xx_raises_unavailable(self) -> None:
        transport = StubTransport(
            {("POST", "/api/pull"): HTTPResponse(500, b"boom")}
        )
        with pytest.raises(BackendUnavailable):
            OllamaBackend().pull_model(
                BackendConfig.new(), "x:1b", transport=transport
            )

    def test_empty_model_name_rejected(self) -> None:
        transport = StubTransport({})
        with pytest.raises(BackendValidationError):
            OllamaBackend().pull_model(
                BackendConfig.new(), "", transport=transport
            )

    def test_overlong_model_name_rejected(self) -> None:
        transport = StubTransport({})
        with pytest.raises(BackendValidationError):
            OllamaBackend().pull_model(
                BackendConfig.new(),
                "x" * (MAX_MODEL_NAME_LEN + 1),
                transport=transport,
            )

    def test_extra_headers_attached_to_pull(self) -> None:
        body = self._ok_stream()
        transport = StubTransport({("POST", "/api/pull"): HTTPResponse(200, body)})
        c = BackendConfig.new(extra_headers={"X-Token": "abc"})
        OllamaBackend().pull_model(c, "x:1b", transport=transport)
        assert transport.calls[0]["headers"].get("X-Token") == "abc"


# --------------------------------------------------------------------------- #
# ModelBackend ABC default
# --------------------------------------------------------------------------- #


class TestPullModelDefault:
    def test_abc_default_raises_validation_error(self) -> None:
        # Subclass that doesn't override pull_model should report a
        # clean BackendValidationError, not NotImplementedError.
        class FakeBackend(ModelBackend):
            name = "fake"

            def health_check(self, config, *, transport=urllib_transport):
                return BackendHealth(
                    ok=True, provider="fake", base_url=config.base_url
                )

            def list_models(self, config, *, transport=urllib_transport):
                return []

            def generate(self, config, request, *, transport=urllib_transport):
                raise NotImplementedError

            def embed(self, config, request, *, transport=urllib_transport):
                raise NotImplementedError

        with pytest.raises(BackendValidationError):
            FakeBackend().pull_model(
                BackendConfig.new(), "x:1b", transport=lambda *a, **k: None
            )


# --------------------------------------------------------------------------- #
# PullSummary serialisation
# --------------------------------------------------------------------------- #


class TestPullSummaryToDict:
    def test_to_dict_includes_events_with_percent(self) -> None:
        events = (
            PullProgressEvent(status="pulling manifest"),
            PullProgressEvent(
                status="pulling l1",
                digest="sha256:1",
                total=100,
                completed=50,
            ),
            PullProgressEvent(status="success"),
        )
        s = PullSummary(
            model="m",
            provider=PROVIDER_OLLAMA,
            success=True,
            events=events,
        )
        d = s.to_dict()
        assert d["model"] == "m"
        assert d["success"] is True
        assert len(d["events"]) == 3
        assert d["events"][1]["percent"] == 50.0
        assert d["events"][1]["digest"] == "sha256:1"
