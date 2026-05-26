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

    def test_default_backend_with_mps_still_faster_whisper_today(self) -> None:
        # G7.3 will flip this for MPS once the cpp adapter ships;
        # for G7.1 we keep the historical default deterministic
        # regardless of the live device label.
        assert wb.default_backend_id("mps") == "faster-whisper"

    def test_default_backend_with_unknown_device_still_works(self) -> None:
        assert wb.default_backend_id("unknown-arch") == "faster-whisper"


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

    def test_whisper_cpp_advertised_but_unavailable(self) -> None:
        # The placeholder reports unavailable until G7.2; the UI
        # greys out the option but still lists it so the user
        # knows what's coming.
        cpp = next(
            b for b in wb.describe_backends() if b["id"] == "whisper.cpp"
        )
        assert cpp["available"] is False
        assert "G7.2" in cpp["unavailable_reason"] or cpp["unavailable_reason"]


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

    def test_is_unavailable_until_adapter_ships(self) -> None:
        # G7.1 ships the registration; G7.2 ships the inference
        # path. Until then the placeholder reports unavailable.
        avail, reason = wb.WhisperCppBackend().is_available()
        assert avail is False
        assert reason  # non-empty — UI renders this

    def test_transcribe_raises_not_implemented(self) -> None:
        be = wb.WhisperCppBackend()
        with pytest.raises(NotImplementedError):
            be.transcribe(
                Path("/tmp/x.wav"),
                model_name="large-v3",
                language="en",
                asr_options={},
                vad_options={},
                progress=lambda m, f: None,
            )


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
