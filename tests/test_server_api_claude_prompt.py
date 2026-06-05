"""Tests for ``GET /docs/api-claude-prompt`` — the markdown helper
that serves docs/scribe-api-claude-prompt.md so a consumer-side
Claude Code session can ``curl`` it down as ``CLAUDE.md``.

Pin:

* the route exists and returns text/markdown,
* the body contains the headings the prompt is supposed to carry
  (so a future doc-restructure can't silently strip the bits the
  consumer-side session needs),
* missing file → 404 (clean error rather than a 500 from
  ``read_text`` blowing up).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(srv, "JOBS", {})
    return TestClient(srv.app), tmp_path


class TestApiClaudePromptDoc:
    def test_returns_raw_markdown(self, env) -> None:
        client, _ = env
        r = client.get("/docs/api-claude-prompt")
        assert r.status_code == 200, r.text
        assert "text/markdown" in r.headers["content-type"]

    def test_carries_expected_sections(self, env) -> None:
        client, _ = env
        body = client.get("/docs/api-claude-prompt").text
        # Anchors the consumer-side session relies on. If a refactor
        # drops one of these, the test surfaces it before someone's
        # CLAUDE.md silently loses the section.
        assert "Connection" in body
        assert "/api/v1/" in body
        assert "SCRIBE_API_KEY" in body
        assert "/api/v1/transcripts" in body
        assert "/api/v1/search" in body
        assert "/ask" in body

    def test_404_when_file_missing(
        self, env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        # Point at a non-existent file to exercise the missing-file
        # branch without deleting the real one.
        monkeypatch.setattr(srv, "_API_PROMPT_PATH", tmp_path / "nope.md")
        client, _ = env
        r = client.get("/docs/api-claude-prompt")
        assert r.status_code == 404
