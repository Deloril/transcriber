"""Tests for the compatibility shims in scribe.engine — torch.load
weights_only, hf_hub use_auth_token, and the safe-globals registry."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestSafeGlobalsRegistry:
    def test_register_safe_globals_idempotent(self) -> None:
        from scribe import engine

        # Already registered at import. Calling again must not raise.
        engine._SAFE_GLOBALS_REGISTERED = False  # force re-run
        engine._register_safe_globals()
        assert engine._SAFE_GLOBALS_REGISTERED is True
        engine._register_safe_globals()
        assert engine._SAFE_GLOBALS_REGISTERED is True

    def test_register_handles_missing_optional_imports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If add_safe_globals doesn't exist (PyTorch < 2.6), the function
        # should bail out cleanly.
        from scribe import engine

        monkeypatch.delattr(engine.torch.serialization, "add_safe_globals", raising=False)
        engine._SAFE_GLOBALS_REGISTERED = False
        # Should not raise.
        engine._register_safe_globals()


class TestHfHubShim:
    def test_translates_use_auth_token(self) -> None:
        from scribe import engine

        # Build a fake huggingface_hub module with the new-style API
        # (token kwarg, no use_auth_token).
        fake_hf = SimpleNamespace()
        captured_kwargs: dict = {}

        def fake_download(*args: object, **kwargs: object):
            captured_kwargs.update(kwargs)
            return "ok"

        fake_hf.hf_hub_download = fake_download

        # Inject the fake module then run the shim.
        sys.modules["huggingface_hub"] = fake_hf  # type: ignore[assignment]
        try:
            engine._shim_hf_hub_download()
            # Calling with the legacy kwarg should now work.
            result = fake_hf.hf_hub_download("repo", use_auth_token="abc")
            assert result == "ok"
            assert captured_kwargs == {"token": "abc"}
        finally:
            sys.modules.pop("huggingface_hub", None)

    def test_idempotent(self) -> None:
        from scribe import engine

        fake_hf = SimpleNamespace()

        def fake_download(*args: object, **kwargs: object):
            return "ok"

        fake_hf.hf_hub_download = fake_download
        sys.modules["huggingface_hub"] = fake_hf  # type: ignore[assignment]
        try:
            engine._shim_hf_hub_download()
            first = fake_hf.hf_hub_download
            engine._shim_hf_hub_download()
            second = fake_hf.hf_hub_download
            # Should not be re-wrapped.
            assert first is second
        finally:
            sys.modules.pop("huggingface_hub", None)

    def test_no_huggingface_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scribe import engine

        # Simulate huggingface_hub not being importable.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        # Should not raise.
        engine._shim_hf_hub_download()

    def test_does_not_overwrite_explicit_token(self) -> None:
        # If the caller passes both `use_auth_token` and `token`, prefer the
        # caller-supplied `token` and drop the legacy kwarg silently.
        from scribe import engine

        fake_hf = SimpleNamespace()
        captured: dict = {}

        def fake_download(*args: object, **kwargs: object):
            captured.update(kwargs)
            return "ok"

        fake_hf.hf_hub_download = fake_download
        sys.modules["huggingface_hub"] = fake_hf  # type: ignore[assignment]
        try:
            engine._shim_hf_hub_download()
            fake_hf.hf_hub_download("repo", use_auth_token="legacy", token="explicit")
            assert captured == {"token": "explicit"}
        finally:
            sys.modules.pop("huggingface_hub", None)
