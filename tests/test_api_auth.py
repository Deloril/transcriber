"""Tests for ``scribe.api_auth`` — API-key minting + verification.

Pin:
  - mint() returns the plaintext exactly once + persists only the
    hash (the plaintext does NOT round-trip through load_keys),
  - verify_api_key() returns the matching record on success and
    None on a near-miss,
  - revoke() drops the record by id,
  - constant-time compare semantics aren't directly testable, but
    we at least exercise the wrong-key path,
  - load_keys() handles missing / corrupt files gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe import api_auth


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin SCRIBE_HOME to a tmp dir so mint + load round-trip
    against a clean filesystem each test."""
    monkeypatch.setenv(api_auth.ENV_HOME, str(tmp_path))
    return tmp_path


class TestMint:
    def test_returns_record_and_plaintext(self, home: Path) -> None:
        record, plaintext = api_auth.mint_api_key("claude-mcp")
        assert record.label == "claude-mcp"
        assert record.id.startswith("key-")
        assert plaintext.startswith(api_auth.TOKEN_PREFIX)
        # Plaintext has roughly the right entropy length.
        assert len(plaintext) > len(api_auth.TOKEN_PREFIX) + 16

    def test_plaintext_does_not_persist(self, home: Path) -> None:
        _, plaintext = api_auth.mint_api_key("ephemeral")
        on_disk = (home / "api_keys.json").read_text()
        assert plaintext not in on_disk

    def test_persists_hash(self, home: Path) -> None:
        _, plaintext = api_auth.mint_api_key("hashed")
        on_disk = json.loads((home / "api_keys.json").read_text())
        assert "keys" in on_disk
        assert len(on_disk["keys"]) == 1
        assert on_disk["keys"][0]["hash"].startswith("sha256:")

    def test_blank_label_falls_back(self, home: Path) -> None:
        record, _ = api_auth.mint_api_key("")
        assert record.label == "unnamed"

    def test_caps_label_length(self, home: Path) -> None:
        record, _ = api_auth.mint_api_key("x" * 5000)
        assert len(record.label) <= 200


class TestLoad:
    def test_no_file_yields_empty(self, home: Path) -> None:
        assert api_auth.load_keys() == []

    def test_corrupt_file_yields_empty(self, home: Path) -> None:
        (home / "api_keys.json").write_text("{not valid json")
        assert api_auth.load_keys() == []

    def test_returns_records_round_trip(self, home: Path) -> None:
        record, _ = api_auth.mint_api_key("first")
        loaded = api_auth.load_keys()
        assert len(loaded) == 1
        assert loaded[0].id == record.id
        assert loaded[0].label == record.label
        assert loaded[0].hash == record.hash


class TestVerify:
    def test_correct_token_matches(self, home: Path) -> None:
        record, plaintext = api_auth.mint_api_key("matchme")
        out = api_auth.verify_api_key(plaintext)
        assert out is not None
        assert out.id == record.id

    def test_wrong_token_returns_none(self, home: Path) -> None:
        api_auth.mint_api_key("matchme")
        assert api_auth.verify_api_key("sk_scribe_NOPE") is None

    def test_empty_token_returns_none(self, home: Path) -> None:
        api_auth.mint_api_key("matchme")
        assert api_auth.verify_api_key("") is None
        assert api_auth.verify_api_key(None) is None  # type: ignore[arg-type]

    def test_verify_updates_last_used_at(self, home: Path) -> None:
        record, plaintext = api_auth.mint_api_key(
            "tracker", now="2026-06-05T00:00:00Z",
        )
        # Pre-verify: never used.
        loaded = api_auth.load_keys()
        assert loaded[0].last_used_at is None
        api_auth.verify_api_key(plaintext, now="2026-06-05T10:30:00Z")
        loaded = api_auth.load_keys()
        assert loaded[0].last_used_at == "2026-06-05T10:30:00Z"


class TestRevoke:
    def test_revoke_removes_record(self, home: Path) -> None:
        record, plaintext = api_auth.mint_api_key("doomed")
        assert api_auth.revoke_api_key(record.id) is True
        assert api_auth.load_keys() == []
        # And it can't authenticate any more.
        assert api_auth.verify_api_key(plaintext) is None

    def test_unknown_id_returns_false(self, home: Path) -> None:
        api_auth.mint_api_key("survivor")
        assert api_auth.revoke_api_key("key-deadbeef00") is False
        # Original key still there.
        assert len(api_auth.load_keys()) == 1


class TestMultipleKeys:
    def test_each_key_verifies_independently(self, home: Path) -> None:
        _, plaintext_a = api_auth.mint_api_key("a")
        _, plaintext_b = api_auth.mint_api_key("b")
        assert api_auth.verify_api_key(plaintext_a) is not None
        assert api_auth.verify_api_key(plaintext_b) is not None
        assert api_auth.verify_api_key("sk_scribe_other") is None
