"""Tests for ``scribe.global_ai_backend`` — the project-less variant
of the AI backend config used by features that aren't bound to a
project (the transcript editor's "Tidy speech with AI" button is the
canonical caller).

We exercise:

* defaults when no file exists,
* round-trip via :func:`save_global_config` / :func:`load_global_config`,
* atomic write (no half-file under any error),
* defensive parsing of malformed / oversized files,
* SCRIBE_HOME override (so tests don't touch a real ``~/.scribe``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.ai_backend import (
    BackendConfig,
    BackendValidationError,
    DEFAULT_OLLAMA_BASE_URL,
    PROVIDER_OLLAMA,
)
from scribe import global_ai_backend as gab


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin SCRIBE_HOME to a tmp dir so we never touch the real home."""
    monkeypatch.setenv("SCRIBE_HOME", str(tmp_path))
    return tmp_path


class TestLoadDefaults:
    def test_no_file_returns_defaults(self, home: Path) -> None:
        cfg = gab.load_global_config()
        assert cfg.provider == PROVIDER_OLLAMA
        assert cfg.base_url == DEFAULT_OLLAMA_BASE_URL
        assert cfg.default_model == ""
        assert cfg.default_embedding_model == ""

    def test_blank_file_returns_defaults(self, home: Path) -> None:
        gab.config_path().parent.mkdir(parents=True, exist_ok=True)
        gab.config_path().write_text("   \n")
        cfg = gab.load_global_config()
        assert cfg.provider == PROVIDER_OLLAMA


class TestRoundTrip:
    def test_save_then_load(self, home: Path) -> None:
        original = BackendConfig.new(
            provider=PROVIDER_OLLAMA,
            base_url="http://lan-box:11434",
            default_model="llama3.2:3b",
            default_embedding_model="bge-m3",
            request_timeout_s=30.0,
            generate_timeout_s=300.0,
            extra_headers={"Authorization": "Bearer x"},
        )
        path = gab.save_global_config(original)
        assert path.exists()
        loaded = gab.load_global_config()
        assert loaded.provider == original.provider
        assert loaded.base_url == original.base_url
        assert loaded.default_model == original.default_model
        assert loaded.default_embedding_model == original.default_embedding_model
        assert loaded.extra_headers == (("Authorization", "Bearer x"),)

    def test_save_omits_headers_when_empty(self, home: Path) -> None:
        cfg = BackendConfig.new(
            provider=PROVIDER_OLLAMA,
            base_url="http://localhost:11434",
            default_model="llama3.2:3b",
        )
        gab.save_global_config(cfg)
        body = json.loads(gab.config_path().read_text())
        assert "extra_headers" not in body

    def test_save_creates_parent_dir(self, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        # SCRIBE_HOME points at a path that doesn't yet exist.
        nested = tmp_path / "deep" / "scribe-home"
        monkeypatch.setenv("SCRIBE_HOME", str(nested))
        gab.save_global_config(BackendConfig.from_dict({}))
        assert nested.is_dir()


class TestMalformed:
    def test_invalid_json_raises(self, home: Path) -> None:
        gab.config_path().parent.mkdir(parents=True, exist_ok=True)
        gab.config_path().write_text("{not valid json")
        with pytest.raises(BackendValidationError):
            gab.load_global_config()

    def test_top_level_must_be_object(self, home: Path) -> None:
        gab.config_path().parent.mkdir(parents=True, exist_ok=True)
        gab.config_path().write_text("[1, 2, 3]")
        with pytest.raises(BackendValidationError):
            gab.load_global_config()

    def test_extra_headers_must_be_object(self, home: Path) -> None:
        gab.config_path().parent.mkdir(parents=True, exist_ok=True)
        gab.config_path().write_text(json.dumps({"extra_headers": "nope"}))
        with pytest.raises(BackendValidationError):
            gab.load_global_config()

    def test_oversized_file_refuses(self, home: Path) -> None:
        gab.config_path().parent.mkdir(parents=True, exist_ok=True)
        gab.config_path().write_text("x" * (gab._MAX_FILE_BYTES + 1))
        with pytest.raises(BackendValidationError):
            gab.load_global_config()


class TestAtomicWrite:
    def test_temp_file_cleaned_up_on_failure(
        self, home: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force os.replace to fail; the temp file must not leak.
        import os
        cfg = BackendConfig.from_dict({})

        original_replace = os.replace
        def boom(*args, **kwargs):  # noqa: ARG001
            raise OSError("simulated replace failure")
        monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(OSError):
            gab.save_global_config(cfg)
        # Restore so we can inspect the directory.
        monkeypatch.setattr(os, "replace", original_replace)
        siblings = list(gab.config_path().parent.glob(".ai_backend.*.tmp"))
        assert siblings == []

    def test_validate_runs_before_write(self, home: Path) -> None:
        # An invalid base_url must never land on disk even partially.
        bad = BackendConfig(provider=PROVIDER_OLLAMA, base_url="not-a-url")
        with pytest.raises(BackendValidationError):
            gab.save_global_config(bad)
        assert not gab.config_path().exists()
