"""Tests for the API-keys CLI (``scribe.scripts.api_keys``).

Cover the three subcommands + the usage / unknown-command paths.
We pin SCRIBE_HOME to a tmp dir so this never touches the
developer's real ~/.scribe.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scribe import api_auth
from scribe.scripts import api_keys


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(api_auth.ENV_HOME, str(tmp_path))
    return tmp_path


class TestMint:
    def test_prints_plaintext_once(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["mint", "claude-mcp"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Minted key" in out
        assert "sk_scribe_" in out
        # The plaintext appears in the curl example too.
        assert "Authorization: Bearer sk_scribe_" in out

    def test_persists_to_disk(self, home: Path) -> None:
        api_keys.main(["mint", "persistent"])
        assert (home / "api_keys.json").is_file()
        keys = api_auth.load_keys()
        assert len(keys) == 1
        assert keys[0].label == "persistent"

    def test_blank_label_rejected(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["mint", ""])
        assert rc == 2


class TestList:
    def test_empty_state(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["list"])
        assert rc == 0
        assert "no API keys" in capsys.readouterr().out

    def test_lists_minted(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        api_auth.mint_api_key("alpha")
        api_auth.mint_api_key("beta")
        rc = api_keys.main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        # Plaintext is never re-shown.
        assert "sk_scribe_" not in out


class TestRevoke:
    def test_drops_key(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, _ = api_auth.mint_api_key("doomed")
        rc = api_keys.main(["revoke", record.id])
        assert rc == 0
        assert "Revoked" in capsys.readouterr().out
        assert api_auth.load_keys() == []

    def test_unknown_id_returns_1(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["revoke", "key-deadbeef00"])
        assert rc == 1


class TestUsage:
    def test_no_args_prints_usage(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main([])
        assert rc == 2
        # Captures the help message.
        cap = capsys.readouterr()
        assert "Usage" in cap.err

    def test_unknown_command(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["yeet"])
        assert rc == 2

    def test_help_flag(
        self, home: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = api_keys.main(["--help"])
        assert rc == 0
