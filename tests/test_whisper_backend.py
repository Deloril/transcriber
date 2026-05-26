"""Unit tests for the WhisperBackend abstraction (G7.1).

Pure-Python coverage of the registry, factory, and metadata. Server
endpoint + UI tests live in ``tests/test_server_whisper_backend.py``.
The actual inference paths (``FasterWhisperBackend.transcribe``) need
whisperx + torch and are exercised by the ``slow``-marked engine
tests; here we only verify the abstraction itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scribe import whisper_backend as wb


# --------------------------------------------------------------------------- #
# Module-level constants + registry shape
# --------------------------------------------------------------------------- #


class TestModuleConstants:
    def test_default_backend_id_is_faster_whisper(self) -> None:
        assert wb.BACKEND_FASTER_WHISPER == "faster-whisper"

    def test_whisper_cpp_id_stable(self) -> None:
        # The form posts this string verbatim; renaming silently
        # would orphan in-flight jobs.
        assert wb.BACKEND_WHISPER_CPP == "whisper.cpp"

    def test_default_backend_returns_faster_whisper(self) -> None:
        assert wb.default_backend_id() == "faster-whisper"

    def test_default_backend_cuda_is_faster_whisper(self) -> None:
        # G7.3 — the Apple-Silicon flip is mps-only; CUDA boxes stay
        # on faster-whisper because CT2 has the better int8/fp16 tier
        # there.
        assert wb.default_backend_id("cuda") == "faster-whisper"

    def test_default_backend_rocm_is_faster_whisper(self) -> None:
        # G7.3 — AMD ROCm boxes stay on faster-whisper too; the CT2
        # ROCm wheel is the right path there.
        assert wb.default_backend_id("rocm") == "faster-whisper"

    def test_default_backend_cpu_is_faster_whisper(self) -> None:
        assert wb.default_backend_id("cpu") == "faster-whisper"

    def test_default_backend_with_unknown_device_still_works(self) -> None:
        assert wb.default_backend_id("unknown-arch") == "faster-whisper"

    def test_default_backend_mps_flips_when_whisper_cpp_available(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # G7.3 — Apple Silicon: when pywhispercpp is importable
        # (is_available() True), the default flips to whisper.cpp so
        # the upload page lands with the Metal-accelerated path
        # pre-selected.
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        assert wb.default_backend_id("mps") == "whisper.cpp"

    def test_default_backend_mps_falls_back_when_whisper_cpp_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # G7.3 — if the user is on Apple Silicon but hasn't installed
        # pywhispercpp, the default must fall back to faster-whisper;
        # otherwise the page would land on a backend that can't run.
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(
            cpp, "is_available", lambda: (False, "pywhispercpp not installed"),
        )
        assert wb.default_backend_id("mps") == "faster-whisper"

    def test_default_backend_mps_uppercase_normalised(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Defensive — accept "MPS" / " mps " from callers that pass
        # the label through unfiltered.
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        assert wb.default_backend_id("MPS") == "whisper.cpp"
        assert wb.default_backend_id(" mps ") == "whisper.cpp"


# --------------------------------------------------------------------------- #
# BackendInfo dataclass
# --------------------------------------------------------------------------- #


class TestBackendInfo:
    def test_to_dict_roundtrip(self) -> None:
        info = wb.BackendInfo(
            id="x",
            display_name="X",
            description="d",
            supported_devices=("cpu",),
            model_format="ct2",
            available=True,
        )
        d = info.to_dict()
        assert d["id"] == "x"
        assert d["display_name"] == "X"
        assert d["description"] == "d"
        assert d["supported_devices"] == ["cpu"]
        assert d["model_format"] == "ct2"
        assert d["available"] is True
        assert d["unavailable_reason"] == ""

    def test_to_dict_includes_reason_when_unavailable(self) -> None:
        info = wb.BackendInfo(
            id="x",
            display_name="X",
            description="d",
            supported_devices=(),
            model_format="gguf",
            available=False,
            unavailable_reason="missing dep",
        )
        d = info.to_dict()
        assert d["available"] is False
        assert d["unavailable_reason"] == "missing dep"


# --------------------------------------------------------------------------- #
# G7.3 — recommended_backend_for_device hint
# --------------------------------------------------------------------------- #


class TestRecommendedBackendForDevice:
    def test_no_hint_for_cuda(self) -> None:
        assert wb.recommended_backend_for_device("cuda") is None

    def test_no_hint_for_rocm(self) -> None:
        assert wb.recommended_backend_for_device("rocm") is None

    def test_no_hint_for_cpu(self) -> None:
        assert wb.recommended_backend_for_device("cpu") is None

    def test_no_hint_for_none(self) -> None:
        assert wb.recommended_backend_for_device(None) is None

    def test_no_hint_for_unknown_label(self) -> None:
        assert wb.recommended_backend_for_device("unknown-arch") is None

    def test_mps_returns_dict_with_required_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        hint = wb.recommended_backend_for_device("mps")
        assert hint is not None
        for key in (
            "device", "recommended_backend_id", "available",
            "unavailable_reason", "headline", "detail",
        ):
            assert key in hint, f"missing key {key!r} in {hint!r}"

    def test_mps_recommends_whisper_cpp(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        hint = wb.recommended_backend_for_device("mps")
        assert hint["recommended_backend_id"] == "whisper.cpp"
        assert hint["device"] == "mps"
        assert hint["available"] is True
        assert hint["unavailable_reason"] == ""

    def test_mps_when_unavailable_carries_reason(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(
            cpp,
            "is_available",
            lambda: (False, "pywhispercpp not installed"),
        )
        hint = wb.recommended_backend_for_device("mps")
        assert hint is not None
        assert hint["available"] is False
        assert "pywhispercpp" in hint["unavailable_reason"]

    def test_mps_detail_mentions_speedup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The whole point of the banner is to surface the speedup
        # number so the user understands *why* the default flipped.
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        hint = wb.recommended_backend_for_device("mps")
        assert "5×" in hint["detail"] or "5x" in hint["detail"].lower()

    def test_mps_uppercase_label_normalised(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cpp = wb.get_backend("whisper.cpp")
        monkeypatch.setattr(cpp, "is_available", lambda: (True, ""))
        assert wb.recommended_backend_for_device("MPS") is not None
        assert wb.recommended_backend_for_device(" mps ") is not None


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_registry_has_default_backends(self) -> None:
        ids = wb.available_backend_ids()
        assert "faster-whisper" in ids
        assert "whisper.cpp" in ids

    def test_registry_order_faster_whisper_first(self) -> None:
        # The UI dropdown should show the recommended default first.
        ids = wb.available_backend_ids()
        assert ids.index("faster-whisper") < ids.index("whisper.cpp")

    def test_get_backend_returns_instance(self) -> None:
        be = wb.get_backend("faster-whisper")
        assert isinstance(be, wb.FasterWhisperBackend)
        assert be.id == "faster-whisper"

    def test_get_backend_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            wb.get_backend("does-not-exist")

    def test_is_valid_backend_id_known(self) -> None:
        assert wb.is_valid_backend_id("faster-whisper") is True
        assert wb.is_valid_backend_id("whisper.cpp") is True

    def test_is_valid_backend_id_unknown(self) -> None:
        assert wb.is_valid_backend_id("imaginary") is False

    def test_is_valid_backend_id_handles_none_and_blank(self) -> None:
        assert wb.is_valid_backend_id(None) is False  # type: ignore[arg-type]
        assert wb.is_valid_backend_id("") is False
        assert wb.is_valid_backend_id(123) is False  # type: ignore[arg-type]

    def test_register_idempotent_on_same_instance(self) -> None:
        be = wb.get_backend("faster-whisper")
        # Re-registering the same instance is a no-op, not an error.
        wb.register_backend(be)
        assert wb.get_backend("faster-whisper") is be

    def test_register_blank_id_rejected(self) -> None:
        class _NoId(wb.WhisperBackend):
            id = ""

            def is_available(self) -> tuple[bool, str]:
                return (True, "")

            def transcribe(self, *a: Any, **kw: Any) -> dict[str, Any]:
                return {"segments": [], "language": "en"}

        with pytest.raises(ValueError):
            wb.register_backend(_NoId())

    def test_register_collision_rejected(self) -> None:
        class _Collide(wb.WhisperBackend):
            id = "faster-whisper"

            def is_available(self) -> tuple[bool, str]:
                return (True, "")

            def transcribe(self, *a: Any, **kw: Any) -> dict[str, Any]:
                return {"segments": [], "language": "en"}

        with pytest.raises(ValueError):
            wb.register_backend(_Collide())

    def test_unregister_then_reregister(self) -> None:
        # Use a custom id to avoid disturbing the default registry.
        class _Tmp(wb.WhisperBackend):
            id = "test-only-tmp-backend"
            display_name = "tmp"
            description = "for tests"
            supported_devices = ("cpu",)
            model_format = "test"

            def is_available(self) -> tuple[bool, str]:
                return (True, "")

            def transcribe(self, *a: Any, **kw: Any) -> dict[str, Any]:
                return {"segments": [], "language": "en"}

        be = _Tmp()
        wb.register_backend(be)
        try:
            assert "test-only-tmp-backend" in wb.available_backend_ids()
        finally:
            wb.unregister_backend("test-only-tmp-backend")
        assert "test-only-tmp-backend" not in wb.available_backend_ids()

    def test_unregister_unknown_is_silent(self) -> None:
        # No-op, no exception.
        wb.unregister_backend("never-existed")


# --------------------------------------------------------------------------- #
# describe_backends() — the API surface consumed by the UI / route
# --------------------------------------------------------------------------- #


class TestDescribeBackends:
    def test_returns_list_of_dicts(self) -> None:
        described = wb.describe_backends()
        assert isinstance(described, list)
        assert all(isinstance(b, dict) for b in described)

    def test_each_entry_has_required_keys(self) -> None:
        for b in wb.describe_backends():
            for k in (
                "id", "display_name", "description",
                "supported_devices", "model_format",
                "available", "unavailable_reason",
            ):
                assert k in b, f"missing key {k} in {b}"

    def test_faster_whisper_advertised(self) -> None:
        ids = [b["id"] for b in wb.describe_backends()]
        assert "faster-whisper" in ids

    def test_whisper_cpp_advertised(self) -> None:
        # Post-G7.2, whisper.cpp's availability tracks ``pywhispercpp``
        # being importable. The UI greys out the option when the
        # dependency isn't installed; the registry always lists it so
        # the user can see it's there. We don't assert availability
        # here because the test environment may or may not have
        # pywhispercpp installed — what we *can* assert is that
        # describe_backends() returns the row at all.
        cpp = next(
            b for b in wb.describe_backends() if b["id"] == "whisper.cpp"
        )
        assert "available" in cpp
        assert isinstance(cpp["available"], bool)
        # When unavailable, the reason must be non-empty so the UI
        # can render it; when available, the reason can be the empty
        # string (the BackendInfo default).
        if not cpp["available"]:
            assert cpp["unavailable_reason"]


# --------------------------------------------------------------------------- #
# FasterWhisperBackend metadata
# --------------------------------------------------------------------------- #


class TestFasterWhisperBackend:
    def test_id_and_format(self) -> None:
        be = wb.FasterWhisperBackend()
        assert be.id == "faster-whisper"
        assert be.model_format == "ct2"

    def test_supported_devices(self) -> None:
        be = wb.FasterWhisperBackend()
        # The CT2 path runs on CUDA, ROCm, and CPU. MPS is not
        # supported (which is the whole motivation for G7.x).
        assert "cuda" in be.supported_devices
        assert "rocm" in be.supported_devices
        assert "cpu" in be.supported_devices
        assert "mps" not in be.supported_devices

    def test_info_shape(self) -> None:
        info = wb.FasterWhisperBackend().info()
        assert info.id == "faster-whisper"
        assert info.model_format == "ct2"


# --------------------------------------------------------------------------- #
# WhisperCppBackend placeholder
# --------------------------------------------------------------------------- #


class TestWhisperCppBackend:
    def test_id_and_format(self) -> None:
        be = wb.WhisperCppBackend()
        assert be.id == "whisper.cpp"
        assert be.model_format == "gguf"

    def test_advertises_apple_silicon_target(self) -> None:
        be = wb.WhisperCppBackend()
        # Whatever set we land on, mps must be in supported_devices —
        # Apple Silicon GPU acceleration is the whole point of this
        # backend.
        assert "mps" in be.supported_devices

    def test_availability_tracks_pywhispercpp_import(self) -> None:
        # G7.2 — the placeholder is gone; availability now reflects
        # whether ``pywhispercpp`` is installed. We can't assume
        # either way in CI, but we can assert that the unavailable
        # branch carries a non-empty reason (so the UI renders
        # something useful).
        avail, reason = wb.WhisperCppBackend().is_available()
        assert isinstance(avail, bool)
        if not avail:
            assert reason
            # A user-actionable hint: mention pywhispercpp so the
            # reader knows what to install.
            assert "pywhispercpp" in reason.lower()

    def test_transcribe_routes_through_whisper_cpp_module(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # G7.2 — backend.transcribe() delegates to
        # scribe.whisper_cpp.transcribe(), reading the GGUF quant
        # from asr_options and forwarding model_name as the model.
        from scribe import whisper_cpp

        called: dict[str, Any] = {}

        def _fake_transcribe(
            audio_path: Path,
            *,
            model: str,
            quant: str,
            language: str,
            progress: Any,
            progress_base: float,
            progress_span: float,
            inference_options: dict[str, Any],
        ) -> dict[str, Any]:
            called["audio_path"] = audio_path
            called["model"] = model
            called["quant"] = quant
            called["language"] = language
            called["progress_base"] = progress_base
            called["progress_span"] = progress_span
            called["inference_options"] = inference_options
            return {"segments": [], "language": language}

        monkeypatch.setattr(whisper_cpp, "transcribe", _fake_transcribe)

        be = wb.WhisperCppBackend()
        out = be.transcribe(
            tmp_path / "in.wav",
            model_name="large-v3-turbo",
            language="en",
            asr_options={"whisper_cpp_quant": "q8_0", "n_threads": 4},
            vad_options={"vad_onset": 0.5},
            progress=lambda m, f: None,
            progress_base=0.1,
            progress_span=0.5,
        )
        assert out == {"segments": [], "language": "en"}
        assert called["model"] == "large-v3-turbo"
        assert called["quant"] == "q8_0"
        assert called["language"] == "en"
        assert called["progress_base"] == 0.1
        assert called["progress_span"] == 0.5
        assert called["inference_options"] == {"n_threads": 4}

    def test_transcribe_defaults_quant_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scribe import whisper_cpp

        captured: dict[str, Any] = {}

        def _fake(audio_path: Path, **kw: Any) -> dict[str, Any]:
            captured.update(kw)
            return {"segments": [], "language": "en"}

        monkeypatch.setattr(whisper_cpp, "transcribe", _fake)

        be = wb.WhisperCppBackend()
        be.transcribe(
            tmp_path / "x.wav",
            model_name="large-v3",
            language="en",
            asr_options={},  # no whisper_cpp_quant
            vad_options={},
            progress=lambda m, f: None,
        )
        # Falls through to the module-level default.
        assert captured["quant"] == whisper_cpp.DEFAULT_QUANT


# --------------------------------------------------------------------------- #
# WhisperBackend ABC — ensures abstract methods are enforced
# --------------------------------------------------------------------------- #


class TestWhisperBackendABC:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            wb.WhisperBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_transcribe(self) -> None:
        class _OnlyAvail(wb.WhisperBackend):
            id = "x"

            def is_available(self) -> tuple[bool, str]:
                return (True, "")

        with pytest.raises(TypeError):
            _OnlyAvail()  # type: ignore[abstract]

    def test_subclass_must_implement_is_available(self) -> None:
        class _OnlyTrans(wb.WhisperBackend):
            id = "x"

            def transcribe(self, *a: Any, **kw: Any) -> dict[str, Any]:
                return {"segments": [], "language": "en"}

        with pytest.raises(TypeError):
            _OnlyTrans()  # type: ignore[abstract]
