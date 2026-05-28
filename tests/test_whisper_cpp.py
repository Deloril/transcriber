"""Unit tests for the whisper.cpp / GGUF inference adapter (G7.2).

Pure-Python coverage. The real ``pywhispercpp`` import is exercised by
the ``slow``-marked engine tests; here we use ``inference_fn`` injection
so we can test every branch of :func:`scribe.whisper_cpp.transcribe`
without the binary on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scribe import whisper_cpp as wcpp


# --------------------------------------------------------------------------- #
# Module-level catalogue + constants
# --------------------------------------------------------------------------- #


class TestCatalogue:
    def test_supported_models_match_planning(self) -> None:
        # PLANNING.md G7.2 calls out exactly these three.
        assert wcpp.SUPPORTED_MODELS == ("large-v3", "large-v3-turbo", "medium")

    def test_supported_quants_match_planning(self) -> None:
        # PLANNING.md G7.2 calls out exactly these three.
        assert wcpp.SUPPORTED_QUANTS == ("q5_0", "q8_0", "f16")

    def test_default_model_is_supported(self) -> None:
        assert wcpp.DEFAULT_MODEL in wcpp.SUPPORTED_MODELS

    def test_default_quant_is_supported(self) -> None:
        assert wcpp.DEFAULT_QUANT in wcpp.SUPPORTED_QUANTS

    def test_hf_repo_is_ggerganov(self) -> None:
        assert wcpp.HF_REPO == "ggerganov/whisper.cpp"


class TestSupportPredicates:
    def test_is_supported_model_true(self) -> None:
        assert wcpp.is_supported_model("large-v3") is True

    def test_is_supported_model_false(self) -> None:
        assert wcpp.is_supported_model("tiny") is False

    def test_is_supported_model_handles_non_string(self) -> None:
        assert wcpp.is_supported_model(None) is False  # type: ignore[arg-type]
        assert wcpp.is_supported_model(123) is False  # type: ignore[arg-type]

    def test_is_supported_quant_true(self) -> None:
        for q in ("q5_0", "q8_0", "f16"):
            assert wcpp.is_supported_quant(q) is True

    def test_is_supported_quant_false(self) -> None:
        assert wcpp.is_supported_quant("q4_0") is False
        assert wcpp.is_supported_quant("") is False

    def test_is_supported_quant_handles_non_string(self) -> None:
        assert wcpp.is_supported_quant(None) is False  # type: ignore[arg-type]


class TestValidate:
    def test_valid_pair_returns_none(self) -> None:
        assert wcpp.validate("large-v3", "q5_0") is None

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported whisper.cpp model"):
            wcpp.validate("tiny.en", "q5_0")

    def test_invalid_quant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported whisper.cpp quant"):
            wcpp.validate("large-v3", "q4_0")

    def test_error_includes_supported_list_for_model(self) -> None:
        with pytest.raises(ValueError) as exc:
            wcpp.validate("xxx", "q5_0")
        assert "large-v3" in str(exc.value)


# --------------------------------------------------------------------------- #
# Filename + URL composition
# --------------------------------------------------------------------------- #


class TestGgufFilename:
    def test_q5_0_includes_quant_suffix(self) -> None:
        assert wcpp.gguf_filename("large-v3", "q5_0") == "ggml-large-v3-q5_0.bin"

    def test_q8_0_includes_quant_suffix(self) -> None:
        assert wcpp.gguf_filename("medium", "q8_0") == "ggml-medium-q8_0.bin"

    def test_f16_drops_quant_suffix(self) -> None:
        # ggerganov's convention: f16 builds have no quant suffix.
        assert wcpp.gguf_filename("large-v3", "f16") == "ggml-large-v3.bin"
        assert wcpp.gguf_filename("medium", "f16") == "ggml-medium.bin"
        assert wcpp.gguf_filename("large-v3-turbo", "f16") == "ggml-large-v3-turbo.bin"

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError):
            wcpp.gguf_filename("tiny", "q5_0")

    def test_invalid_quant_raises(self) -> None:
        with pytest.raises(ValueError):
            wcpp.gguf_filename("large-v3", "q4_0")


class TestHfDownloadUrl:
    def test_url_uses_resolve_main(self) -> None:
        url = wcpp.hf_download_url("large-v3", "q5_0")
        assert url == (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            "ggml-large-v3-q5_0.bin"
        )

    def test_f16_url_drops_quant(self) -> None:
        url = wcpp.hf_download_url("medium", "f16")
        assert url.endswith("/ggml-medium.bin")

    def test_validates_inputs(self) -> None:
        with pytest.raises(ValueError):
            wcpp.hf_download_url("nope", "q5_0")


# --------------------------------------------------------------------------- #
# Cache directory + path resolution
# --------------------------------------------------------------------------- #


class TestDefaultCacheDir:
    def test_default_under_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(wcpp.ENV_CACHE_DIR, raising=False)
        # Pin Path.home() so the test doesn't depend on the dev's $HOME.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        d = wcpp.default_cache_dir()
        assert d == tmp_path / ".scribe" / "models" / "whisper.cpp"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        custom = tmp_path / "custom-cache"
        monkeypatch.setenv(wcpp.ENV_CACHE_DIR, str(custom))
        assert wcpp.default_cache_dir() == custom

    def test_env_override_blank_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(wcpp.ENV_CACHE_DIR, "   ")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert wcpp.default_cache_dir() == (
            tmp_path / ".scribe" / "models" / "whisper.cpp"
        )

    def test_env_override_expands_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(wcpp.ENV_CACHE_DIR, "~/explicit-cache")
        d = wcpp.default_cache_dir()
        assert "~" not in str(d)


class TestGgufPath:
    def test_uses_provided_cache_dir(self, tmp_path: Path) -> None:
        p = wcpp.gguf_path("large-v3", "q5_0", cache_dir=tmp_path)
        assert p == tmp_path / "ggml-large-v3-q5_0.bin"

    def test_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(wcpp.ENV_CACHE_DIR, str(tmp_path))
        p = wcpp.gguf_path("large-v3", "f16")
        assert p == tmp_path / "ggml-large-v3.bin"


class TestIsCached:
    def test_returns_false_when_missing(self, tmp_path: Path) -> None:
        assert wcpp.is_cached("large-v3", "q5_0", cache_dir=tmp_path) is False

    def test_returns_true_when_present(self, tmp_path: Path) -> None:
        p = tmp_path / "ggml-large-v3-q5_0.bin"
        p.write_bytes(b"fake gguf")
        assert wcpp.is_cached("large-v3", "q5_0", cache_dir=tmp_path) is True

    def test_directory_does_not_count_as_cached(self, tmp_path: Path) -> None:
        # Defensive: a stray directory with the GGUF name shouldn't be
        # treated as a cached model.
        (tmp_path / "ggml-large-v3-q5_0.bin").mkdir()
        assert wcpp.is_cached("large-v3", "q5_0", cache_dir=tmp_path) is False


# --------------------------------------------------------------------------- #
# Catalogue listing
# --------------------------------------------------------------------------- #


class TestListCatalogue:
    def test_returns_full_grid(self, tmp_path: Path) -> None:
        rows = wcpp.list_catalogue(cache_dir=tmp_path)
        # 3 models × 3 quants = 9 rows.
        assert len(rows) == 3 * 3
        # Every row should be a ModelEntry.
        assert all(isinstance(r, wcpp.ModelEntry) for r in rows)

    def test_cached_flag_reflects_disk(self, tmp_path: Path) -> None:
        target = tmp_path / "ggml-large-v3-q5_0.bin"
        target.write_bytes(b"x" * 42)
        rows = wcpp.list_catalogue(cache_dir=tmp_path)
        cached = [r for r in rows if r.cached]
        assert len(cached) == 1
        only = cached[0]
        assert only.model == "large-v3"
        assert only.quant == "q5_0"
        assert only.size_bytes == 42

    def test_uncached_rows_have_none_size(self, tmp_path: Path) -> None:
        rows = wcpp.list_catalogue(cache_dir=tmp_path)
        for r in rows:
            assert r.cached is False
            assert r.size_bytes is None

    def test_to_dict_keys(self, tmp_path: Path) -> None:
        rows = wcpp.list_catalogue(cache_dir=tmp_path)
        d = rows[0].to_dict()
        for k in (
            "model", "quant", "filename", "path", "cached",
            "size_bytes", "download_url",
        ):
            assert k in d


# --------------------------------------------------------------------------- #
# Availability probe
# --------------------------------------------------------------------------- #


class TestPywhispercppAvailability:
    def test_returns_tuple(self) -> None:
        avail, reason = wcpp.is_pywhispercpp_available()
        assert isinstance(avail, bool)
        assert isinstance(reason, str)

    def test_unavailable_branch_carries_reason(self) -> None:
        avail, reason = wcpp.is_pywhispercpp_available()
        if not avail:
            # Hint must point the user at a fix.
            assert "pywhispercpp" in reason.lower()


# --------------------------------------------------------------------------- #
# Segment conversion — pywhispercpp tokens → whisperx-shaped dict
# --------------------------------------------------------------------------- #


class TestConvertSegments:
    def test_empty_input(self) -> None:
        out = wcpp.convert_segments([], language="en")
        assert out == {"segments": [], "language": "en"}

    def test_single_sentence_one_segment(self) -> None:
        toks = [
            {"t0": 0, "t1": 50, "text": "Hello"},
            {"t0": 60, "t1": 130, "text": " world."},
        ]
        out = wcpp.convert_segments(toks, language="en")
        assert len(out["segments"]) == 1
        seg = out["segments"][0]
        assert seg["text"] == "Hello world."
        assert seg["start"] == pytest.approx(0.0)
        assert seg["end"] == pytest.approx(1.3)
        # Every word has the expected start/end in seconds.
        assert seg["words"][0]["word"] == "Hello"
        assert seg["words"][0]["start"] == pytest.approx(0.0)
        assert seg["words"][0]["end"] == pytest.approx(0.5)
        assert seg["words"][1]["word"] == " world."
        assert seg["words"][1]["start"] == pytest.approx(0.6)
        assert seg["words"][1]["end"] == pytest.approx(1.3)

    def test_two_sentences_split_on_punctuation(self) -> None:
        toks = [
            {"t0": 0, "t1": 30, "text": "Hi"},
            {"t0": 30, "t1": 60, "text": " there."},
            {"t0": 80, "t1": 120, "text": " How"},
            {"t0": 120, "t1": 200, "text": " are you?"},
        ]
        out = wcpp.convert_segments(toks, language="en")
        assert len(out["segments"]) == 2
        assert out["segments"][0]["text"] == "Hi there."
        assert out["segments"][1]["text"] == "How are you?"

    def test_max_words_per_segment_forces_flush(self) -> None:
        toks = [
            {"t0": 10 * i, "t1": 10 * (i + 1), "text": f" w{i}"}
            for i in range(5)
        ]  # five tokens, no punctuation
        out = wcpp.convert_segments(toks, language="en", max_words_per_segment=2)
        # 5 tokens / 2 per seg → 3 segments (2, 2, 1).
        assert len(out["segments"]) == 3

    def test_max_words_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            wcpp.convert_segments([], language="en", max_words_per_segment=0)

    def test_skips_special_tokens(self) -> None:
        toks = [
            {"t0": 0, "t1": 10, "text": "[_BEG_]"},
            {"t0": 10, "t1": 20, "text": "Hello."},
            {"t0": 25, "t1": 30, "text": "[BLANK_AUDIO]"},
        ]
        out = wcpp.convert_segments(toks, language="en")
        assert len(out["segments"]) == 1
        # Special tokens dropped.
        assert "[" not in out["segments"][0]["text"]

    def test_skips_empty_tokens(self) -> None:
        toks = [
            {"t0": 0, "t1": 5, "text": "   "},
            {"t0": 10, "t1": 20, "text": "Hi."},
        ]
        out = wcpp.convert_segments(toks, language="en")
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "Hi."

    def test_passes_language_through(self) -> None:
        out = wcpp.convert_segments([], language="fr")
        assert out["language"] == "fr"

    def test_score_propagated_when_present(self) -> None:
        toks = [
            {"t0": 0, "t1": 10, "text": "Hi.", "score": 0.97},
        ]
        out = wcpp.convert_segments(toks, language="en")
        assert out["segments"][0]["words"][0]["score"] == pytest.approx(0.97)

    def test_score_none_when_absent(self) -> None:
        toks = [{"t0": 0, "t1": 10, "text": "Hi."}]
        out = wcpp.convert_segments(toks, language="en")
        assert out["segments"][0]["words"][0]["score"] is None

    def test_centiseconds_to_seconds(self) -> None:
        # The whisper.cpp internal clock is centiseconds; sanity check.
        assert wcpp._centiseconds_to_seconds(150) == pytest.approx(1.5)
        assert wcpp._centiseconds_to_seconds(0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# transcribe() — the public entry point
# --------------------------------------------------------------------------- #


class TestTranscribe:
    def test_validates_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            wcpp.transcribe(
                tmp_path / "x.wav",
                model="tiny",
                quant="q5_0",
                cache_dir=tmp_path,
                inference_fn=lambda *a, **kw: [],
            )

    def test_validates_quant(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            wcpp.transcribe(
                tmp_path / "x.wav",
                model="large-v3",
                quant="q4_0",
                cache_dir=tmp_path,
                inference_fn=lambda *a, **kw: [],
            )

    def test_missing_gguf_raises_with_url(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError) as exc:
            wcpp.transcribe(
                tmp_path / "x.wav",
                model="large-v3",
                quant="q5_0",
                cache_dir=tmp_path,
                inference_fn=lambda *a, **kw: [],
            )
        assert "ggml-large-v3-q5_0.bin" in str(exc.value)
        assert "huggingface.co" in str(exc.value)

    def test_calls_inference_fn_with_paths(self, tmp_path: Path) -> None:
        gguf = tmp_path / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake")
        audio = tmp_path / "in.wav"
        audio.write_bytes(b"riff")

        captured: dict[str, Any] = {}

        def _fake(audio_path: Path, gp: Path, lang: str, opts: dict[str, Any]) -> list[dict[str, Any]]:
            captured["audio_path"] = audio_path
            captured["gguf_path"] = gp
            captured["lang"] = lang
            captured["opts"] = opts
            return [{"t0": 0, "t1": 50, "text": "Hi."}]

        out = wcpp.transcribe(
            audio,
            model="large-v3",
            quant="q5_0",
            language="en",
            cache_dir=tmp_path,
            inference_fn=_fake,
            inference_options={"n_threads": 6},
        )
        assert captured["audio_path"] == audio
        assert captured["gguf_path"] == gguf
        assert captured["lang"] == "en"
        assert captured["opts"] == {"n_threads": 6}
        assert out["language"] == "en"
        assert len(out["segments"]) == 1
        assert out["segments"][0]["text"] == "Hi."

    def test_progress_called_at_phase_boundaries(self, tmp_path: Path) -> None:
        gguf = tmp_path / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake")
        events: list[tuple[str, float]] = []

        def _track(label: str, frac: float) -> None:
            events.append((label, frac))

        wcpp.transcribe(
            tmp_path / "x.wav",
            model="large-v3",
            quant="q5_0",
            cache_dir=tmp_path,
            progress=_track,
            progress_base=0.0,
            progress_span=1.0,
            inference_fn=lambda *a, **kw: [],
        )
        # We expect a Loading event at 0.0 and a complete event at 1.0.
        assert any(f == 0.0 for _, f in events)
        assert any(f == 1.0 for _, f in events)

    def test_progress_respects_base_and_span(self, tmp_path: Path) -> None:
        gguf = tmp_path / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake")
        events: list[float] = []
        wcpp.transcribe(
            tmp_path / "x.wav",
            model="large-v3",
            quant="q5_0",
            cache_dir=tmp_path,
            progress=lambda m, f: events.append(f),
            progress_base=0.5,
            progress_span=0.4,
            inference_fn=lambda *a, **kw: [],
        )
        # Every emitted fraction should fall in [0.5, 0.9].
        for f in events:
            assert 0.5 <= f <= 0.9 + 1e-9

    def test_progress_optional(self, tmp_path: Path) -> None:
        # progress=None must not raise.
        gguf = tmp_path / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake")
        out = wcpp.transcribe(
            tmp_path / "x.wav",
            model="large-v3",
            quant="q5_0",
            cache_dir=tmp_path,
            progress=None,
            inference_fn=lambda *a, **kw: [],
        )
        assert out == {"segments": [], "language": "en"}

    def test_default_inference_options_empty_dict(self, tmp_path: Path) -> None:
        gguf = tmp_path / "ggml-large-v3-q5_0.bin"
        gguf.write_bytes(b"fake")
        captured: dict[str, Any] = {}

        def _fake(audio_path: Path, gp: Path, lang: str, opts: dict[str, Any]) -> list[dict[str, Any]]:
            captured["opts"] = opts
            return []

        wcpp.transcribe(
            tmp_path / "x.wav",
            model="large-v3",
            quant="q5_0",
            cache_dir=tmp_path,
            inference_fn=_fake,
        )
        assert captured["opts"] == {}

    def test_returns_whisperx_shape(self, tmp_path: Path) -> None:
        gguf = tmp_path / "ggml-medium-q8_0.bin"
        gguf.write_bytes(b"fake")
        out = wcpp.transcribe(
            tmp_path / "x.wav",
            model="medium",
            quant="q8_0",
            cache_dir=tmp_path,
            language="fr",
            inference_fn=lambda *a, **kw: [
                {"t0": 0, "t1": 50, "text": "Bonjour."},
            ],
        )
        assert "segments" in out and "language" in out
        assert out["language"] == "fr"
        assert len(out["segments"]) == 1
        seg = out["segments"][0]
        assert "start" in seg and "end" in seg and "text" in seg and "words" in seg


# --------------------------------------------------------------------------- #
# Sentence-end heuristic
# --------------------------------------------------------------------------- #


class TestSentenceEnd:
    def test_period_is_end(self) -> None:
        assert wcpp._is_sentence_end("Hello.") is True

    def test_exclamation_is_end(self) -> None:
        assert wcpp._is_sentence_end("Hi!") is True

    def test_question_is_end(self) -> None:
        assert wcpp._is_sentence_end("What?") is True

    def test_no_punctuation_not_end(self) -> None:
        assert wcpp._is_sentence_end("Hello") is False

    def test_empty_not_end(self) -> None:
        assert wcpp._is_sentence_end("") is False

    def test_whitespace_only_not_end(self) -> None:
        assert wcpp._is_sentence_end("   ") is False

    def test_strips_trailing_whitespace(self) -> None:
        assert wcpp._is_sentence_end("Hello. ") is True


# --------------------------------------------------------------------------- #
# download_gguf — streamed fetch with progress callback
# --------------------------------------------------------------------------- #


class _FakeStream:
    """In-memory stand-in for the urlopen() context manager.

    Yields ``payload`` in fixed-size chunks. Optional ``content_length``
    populates the Content-Length header so the real progress code path
    that reads ``resp.headers.get("Content-Length")`` is exercised.
    """

    def __init__(
        self, payload: bytes, *, content_length: int | None,
        chunk_size: int = 16,
    ) -> None:
        self._payload = payload
        self._pos = 0
        self._chunk_size = chunk_size

        class _Headers:
            def __init__(self, total: int | None) -> None:
                self._total = total

            def get(self, key: str) -> str | None:
                if key.lower() != "content-length":
                    return None
                return None if self._total is None else str(self._total)

        self.headers = _Headers(content_length)

    def read(self, n: int) -> bytes:
        chunk = self._payload[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class TestDownloadGguf:
    def test_writes_file_atomically(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        payload = b"GGUF-bytes" * 100
        opener_calls: list[str] = []

        def fake_opener(url: str, timeout: float):
            opener_calls.append(url)
            return _FakeStream(payload, content_length=len(payload))

        out = wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=tmp_path,
            url_opener=fake_opener,
        )
        assert out == tmp_path / "ggml-large-v3-q5_0.bin"
        assert out.read_bytes() == payload
        # No leftover .partial after a clean download.
        assert not (tmp_path / "ggml-large-v3-q5_0.bin.partial").exists()
        # URL is the canonical Hugging Face one.
        assert opener_calls == [wcpp.hf_download_url("large-v3", "q5_0")]

    def test_invokes_progress_with_running_totals(
        self, tmp_path: Path,
    ) -> None:
        payload = b"x" * 100
        events: list[tuple[int, int | None]] = []

        def fake_opener(url: str, timeout: float):
            return _FakeStream(payload, content_length=100, chunk_size=20)

        wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=tmp_path,
            progress=lambda done, total: events.append((done, total)),
            url_opener=fake_opener,
            chunk_size=20,
        )
        # Progress called once per chunk, last value equals payload size.
        assert events[-1] == (100, 100)
        # Monotonic non-decreasing.
        for prev, curr in zip(events, events[1:]):
            assert curr[0] >= prev[0]

    def test_progress_total_none_when_no_content_length(
        self, tmp_path: Path,
    ) -> None:
        payload = b"abc" * 10

        def fake_opener(url: str, timeout: float):
            return _FakeStream(payload, content_length=None, chunk_size=8)

        seen: list[tuple[int, int | None]] = []
        wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=tmp_path,
            progress=lambda done, total: seen.append((done, total)),
            url_opener=fake_opener,
            chunk_size=8,
        )
        assert seen
        assert all(total is None for _, total in seen)

    def test_idempotent_when_cached(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "ggml-large-v3-q5_0.bin"
        target.write_bytes(b"already-here")
        opener_called = False

        def fake_opener(url: str, timeout: float):
            nonlocal opener_called
            opener_called = True
            return _FakeStream(b"new-bytes", content_length=9)

        out = wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=tmp_path,
            url_opener=fake_opener,
        )
        # Cached path returned without re-fetching.
        assert out == target
        assert opener_called is False
        assert target.read_bytes() == b"already-here"

    def test_failure_does_not_leave_target_file(
        self, tmp_path: Path,
    ) -> None:
        def boom(url: str, timeout: float):
            raise OSError("simulated network failure")

        with pytest.raises(OSError):
            wcpp.download_gguf(
                "large-v3", "q5_0",
                cache_dir=tmp_path,
                url_opener=boom,
            )
        # No half-written target survives.
        assert not (tmp_path / "ggml-large-v3-q5_0.bin").exists()

    def test_progress_callback_failure_does_not_break_download(
        self, tmp_path: Path,
    ) -> None:
        payload = b"y" * 30

        def fake_opener(url: str, timeout: float):
            return _FakeStream(payload, content_length=30, chunk_size=10)

        def bad_progress(done, total):
            raise RuntimeError("ignored")

        out = wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=tmp_path,
            progress=bad_progress,
            url_opener=fake_opener,
            chunk_size=10,
        )
        assert out.read_bytes() == payload

    def test_creates_cache_dir(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "fresh"
        # Doesn't exist yet.
        assert not cache.exists()

        def fake_opener(url: str, timeout: float):
            return _FakeStream(b"x", content_length=1)

        wcpp.download_gguf(
            "large-v3", "q5_0",
            cache_dir=cache,
            url_opener=fake_opener,
        )
        assert cache.is_dir()

    def test_validates_model_and_quant(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError):
            wcpp.download_gguf(
                "not-a-model", "q5_0",
                cache_dir=tmp_path,
                url_opener=lambda u, t: _FakeStream(b"", content_length=0),
            )


# --------------------------------------------------------------------------- #
# decode_audio_for_whisper_cpp — ffmpeg-driven decode that sidesteps the
# pywhispercpp ``ValueError: vector`` bug on macOS where its bundled
# loader hands the C++ binding an empty audio buffer.
# --------------------------------------------------------------------------- #


class TestDecodeAudioForWhisperCpp:
    def _silent_wav(self, tmp_path: Path, *, ms: int = 200) -> Path:
        """Make a tiny silent WAV file via ffmpeg so we don't depend on
        any test fixture being shipped in the repo. Returns the path."""
        import subprocess
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not on PATH")
        out = tmp_path / "silent.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"anullsrc=r=16000:cl=mono",
                "-t", f"{ms / 1000:.3f}",
                "-c:a", "pcm_s16le",
                str(out),
            ],
            check=True,
        )
        return out

    def test_returns_float32_numpy_array(self, tmp_path: Path) -> None:
        np = pytest.importorskip("numpy")
        wav = self._silent_wav(tmp_path)
        out = wcpp.decode_audio_for_whisper_cpp(wav)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        # 16kHz × 0.2s = 3200 samples; ffmpeg may pad slightly so
        # check shape is in the right ballpark.
        assert 2_500 <= out.size <= 4_000

    def test_decodes_to_16khz_mono(self, tmp_path: Path) -> None:
        # Make a non-mono, non-16kHz source and confirm we still get
        # back a flat mono 16kHz buffer.
        import subprocess
        import shutil
        np = pytest.importorskip("numpy")
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not on PATH")
        src = tmp_path / "stereo-44k.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", "0.1",
                "-c:a", "pcm_s16le",
                str(src),
            ],
            check=True,
        )
        out = wcpp.decode_audio_for_whisper_cpp(src)
        assert out.ndim == 1
        # 16kHz × 0.1s ≈ 1600 samples (allow padding).
        assert 1_400 <= out.size <= 2_000

    def test_silent_audio_returns_zeros(self, tmp_path: Path) -> None:
        np = pytest.importorskip("numpy")
        wav = self._silent_wav(tmp_path)
        out = wcpp.decode_audio_for_whisper_cpp(wav)
        # All samples are silent, so peak amplitude is ~0. Use a
        # generous bound — even ffmpeg's null source has occasional
        # rounding artefacts.
        assert float(np.abs(out).max()) < 1e-3

    def test_missing_file_raises_runtime_error(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            wcpp.decode_audio_for_whisper_cpp(tmp_path / "no-such-file.wav")

    def test_no_audio_track_raises(self, tmp_path: Path) -> None:
        # A bare text file isn't audio; ffmpeg will fail at decode.
        bogus = tmp_path / "not-audio.txt"
        bogus.write_text("this is not audio")
        with pytest.raises(RuntimeError):
            wcpp.decode_audio_for_whisper_cpp(bogus)

    def test_default_inference_calls_decode_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Surface-level proof that _default_inference now passes a
        numpy array (not a path string) into pywhispercpp — that's the
        whole reason this helper exists. We stub both the decode and
        the Model so the test runs without ffmpeg or pywhispercpp."""
        np = pytest.importorskip("numpy")
        decoded = np.zeros(3200, dtype=np.float32)
        called_with: list = []

        monkeypatch.setattr(
            wcpp, "decode_audio_for_whisper_cpp", lambda p: decoded,
        )

        # Fake Model class so we can prove .transcribe got the array.
        class _FakeSeg:
            def __init__(self, t0, t1, text):
                self.t0, self.t1, self.text = t0, t1, text

        class _FakeModel:
            def __init__(self, *a, **kw):
                pass
            def transcribe(self, audio):
                called_with.append(audio)
                return [_FakeSeg(0, 100, "hi")]

        # _default_inference imports lazily; inject a fake module.
        import sys
        import types
        fake = types.ModuleType("pywhispercpp.model")
        fake.Model = _FakeModel  # type: ignore[attr-defined]
        sys.modules["pywhispercpp"] = types.ModuleType("pywhispercpp")
        sys.modules["pywhispercpp.model"] = fake

        try:
            out = wcpp._default_inference(
                tmp_path / "audio.wav",
                tmp_path / "ggml.bin",
                "en",
                {},
            )
        finally:
            del sys.modules["pywhispercpp.model"]
            del sys.modules["pywhispercpp"]

        # The single transcribe call got the decoded array, not the path.
        assert len(called_with) == 1
        assert isinstance(called_with[0], np.ndarray)
        assert called_with[0].dtype == np.float32
        # And the result was reshaped into the t0/t1/text dict shape.
        assert out == [{"t0": 0, "t1": 100, "text": "hi"}]
