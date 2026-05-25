"""Tests for scribe.parakeet — model-id helpers, NeMo probe, output extractor."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from scribe import parakeet


class TestIsParakeetModel:
    @pytest.mark.parametrize("name", [
        "parakeet-tdt-0.6b-v2",
        "Parakeet-TDT-0.6B-v3",
        "nvidia/parakeet-tdt-0.6b-v2",
        "NVIDIA/Parakeet-Tdt-0.6B-V3",
    ])
    def test_positive(self, name: str) -> None:
        assert parakeet.is_parakeet_model(name) is True

    @pytest.mark.parametrize("name", [
        "large-v3",
        "large-v3-turbo",
        "medium.en",
        "distil-large-v3",
        "openai/whisper-large-v3",
        "",
    ])
    def test_negative(self, name: str) -> None:
        assert parakeet.is_parakeet_model(name) is False


class TestNormaliseModelId:
    def test_keeps_huggingface_id(self) -> None:
        assert parakeet._normalise_model_id("nvidia/parakeet-tdt-0.6b-v2") == \
            "nvidia/parakeet-tdt-0.6b-v2"

    def test_prepends_nvidia(self) -> None:
        assert parakeet._normalise_model_id("parakeet-tdt-0.6b-v2") == \
            "nvidia/parakeet-tdt-0.6b-v2"


class TestNemoAvailable:
    def setup_method(self) -> None:
        # Reset cache so each test gets a fresh probe.
        parakeet._NEMO_AVAILABLE = None
        parakeet._IMPORT_ERROR = None

    def test_returns_true_when_importable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a fake nemo.collections.asr module so the import succeeds
        # without actually loading NeMo.
        fake_collections = ModuleType("nemo.collections")
        fake_asr = ModuleType("nemo.collections.asr")
        fake_nemo = ModuleType("nemo")
        fake_nemo.collections = fake_collections  # type: ignore[attr-defined]
        fake_collections.asr = fake_asr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
        monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
        monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)
        ok, err = parakeet.nemo_available()
        assert ok is True
        assert err is None

    def test_returns_false_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the import to fail by stubbing sys.modules entries to None,
        # which forces ImportError.
        for k in list(sys.modules):
            if k == "nemo" or k.startswith("nemo."):
                monkeypatch.delitem(sys.modules, k, raising=False)
        monkeypatch.setitem(sys.modules, "nemo", None)
        ok, err = parakeet.nemo_available()
        assert ok is False
        assert err is not None
        assert "ImportError" in err or "Module" in err

    def test_caches_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parakeet._NEMO_AVAILABLE = True
        parakeet._IMPORT_ERROR = None
        # Subsequent calls don't re-probe; even monkeying with sys.modules
        # shouldn't change the cached answer.
        monkeypatch.setitem(sys.modules, "nemo", None)
        ok, err = parakeet.nemo_available()
        assert ok is True
        assert err is None


class TestExtractNemoText:
    def test_empty_input(self) -> None:
        assert parakeet._extract_nemo_text(None) == ""
        assert parakeet._extract_nemo_text([]) == ""
        assert parakeet._extract_nemo_text(()) == ""

    def test_string_list(self) -> None:
        assert parakeet._extract_nemo_text(["hello world"]) == "hello world"

    def test_strings_in_tuple(self) -> None:
        # NeMo sometimes returns (hypotheses, scores) — we look at hypotheses[0].
        assert parakeet._extract_nemo_text((["hello"], [0.99])) == "hello"

    def test_dict_shape(self) -> None:
        assert parakeet._extract_nemo_text([{"text": "  hello  "}]) == "hello"

    def test_hypothesis_object(self) -> None:
        class FakeHypothesis:
            text = "  whisper "

        assert parakeet._extract_nemo_text([FakeHypothesis()]) == "whisper"

    def test_strips_whitespace(self) -> None:
        assert parakeet._extract_nemo_text(["   hi   "]) == "hi"

    def test_unknown_object_falls_back_to_str(self) -> None:
        # An object without .text or dict shape — we coerce to str().
        out = parakeet._extract_nemo_text([12345])
        assert out == "12345"


class TestFreeModels:
    def test_clears_cache(self) -> None:
        parakeet._MODEL_CACHE["fake/model"] = MagicMock()
        parakeet.free_models()
        assert parakeet._MODEL_CACHE == {}


class TestNormalisedModelIdRoundTrip:
    """Sanity: is_parakeet_model accepts both the short and HF forms."""

    def test_normalise_then_is_parakeet(self) -> None:
        for short in ("parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3"):
            full = parakeet._normalise_model_id(short)
            assert parakeet.is_parakeet_model(full)
            assert parakeet.is_parakeet_model(short)


# --------------------------------------------------------------------------- #
# Real-model integration test — opt-in via -m slow + -m gpu.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.gpu
def test_parakeet_transcribes_silent_clip(silent_wav, tmp_path) -> None:
    """Real Parakeet end-to-end on a 1s silence clip. Requires NeMo + a GPU.

    This is *not* exercised by the default suite; run it with:
        pytest -m slow -m gpu tests/test_parakeet.py
    """
    pytest.importorskip("nemo")
    from scribe.engine import AdvancedOptions
    segments, lang = parakeet.transcribe_with_parakeet(
        silent_wav,
        model_name="nvidia/parakeet-tdt-0.6b-v2",
        hf_token=None,
        options=AdvancedOptions(),
    )
    assert lang == "en"
    # Silence may produce no segments or one empty segment — we accept both.
    assert isinstance(segments, list)
