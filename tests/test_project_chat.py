"""Tests for ``scribe.project_chat`` — the pure conversation layer.

The HTTP wrapper, AI backend, and embedding-index plumbing all live
elsewhere. Here we cover validation, serialisation, prompt building,
and on-disk persistence — the things that have to stay rock-solid
regardless of which model is wired up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe import project_chat as pc
from scribe.project_chat import (
    ChatValidationError,
    Citation,
    Conversation,
    Message,
    ROLE_ASSISTANT,
    ROLE_USER,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


PID = "abcdef012345"
SID1 = "111111111111"
SID2 = "222222222222"


def _msg(role: str = ROLE_USER, content: str = "hello",
         created_at: str = "2026-05-26T10:00:00Z",
         message_id: str = "aaaaaaaaaaaa",
         citations=None) -> Message:
    return Message(
        id=message_id, role=role, content=content,
        created_at=created_at, citations=list(citations or []),
    )


# --------------------------------------------------------------------------- #
# Citation
# --------------------------------------------------------------------------- #


class TestCitation:
    def test_round_trip(self) -> None:
        c = Citation(
            source_id=SID1,
            text_preview="hello there",
            score=0.82,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w5",
            transcript_job_id="job123abc456",
        )
        d = c.to_dict()
        assert Citation.from_dict(d) == c

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ChatValidationError):
            Citation.from_dict("not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Message
# --------------------------------------------------------------------------- #


class TestMessage:
    def test_validate_rejects_bad_id(self) -> None:
        m = _msg(message_id="not-hex")
        with pytest.raises(ChatValidationError):
            m.validate()

    def test_validate_rejects_unknown_role(self) -> None:
        m = _msg(role="system")
        with pytest.raises(ChatValidationError):
            m.validate()

    def test_validate_rejects_overlong_content(self) -> None:
        m = _msg(content="x" * (pc.MAX_MESSAGE_LEN + 1))
        with pytest.raises(ChatValidationError):
            m.validate()

    def test_round_trip(self) -> None:
        m = _msg(citations=[Citation(SID1, "snippet", 0.5)])
        d = m.to_dict()
        m2 = Message.from_dict(d)
        assert m2 == m


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #


class TestConversationNew:
    def test_minimal_conversation(self) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1],
            title="Pilot interview",
            now="2026-05-26T10:00:00Z",
        )
        assert c.project_id == PID
        assert c.source_ids == [SID1]
        assert c.title == "Pilot interview"
        assert c.messages == []

    def test_validates_source_ids(self) -> None:
        with pytest.raises(ChatValidationError):
            Conversation.new(
                project_id=PID, source_ids=["not-hex"],
                title="x", now="2026-05-26T10:00:00Z",
            )

    def test_caps_title_length(self) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1],
            title="x" * (pc.MAX_TITLE_LEN + 50),
            now="2026-05-26T10:00:00Z",
        )
        assert len(c.title) == pc.MAX_TITLE_LEN

    def test_explicit_id_used_when_supplied(self) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1], title="t",
            now="2026-05-26T10:00:00Z",
            conversation_id="cccccccccccc",
        )
        assert c.id == "cccccccccccc"


class TestConversationAppend:
    def _conv(self) -> Conversation:
        return Conversation.new(
            project_id=PID, source_ids=[SID1], title="t",
            now="2026-05-26T10:00:00Z",
        )

    def test_append_updates_updated_at(self) -> None:
        c = self._conv()
        m = _msg(created_at="2026-05-26T11:00:00Z")
        c.append_message(m)
        assert c.updated_at == "2026-05-26T11:00:00Z"
        assert len(c.messages) == 1

    def test_append_validates_message(self) -> None:
        c = self._conv()
        with pytest.raises(ChatValidationError):
            c.append_message(_msg(role="weird"))

    def test_append_caps_at_max_messages(self) -> None:
        c = self._conv()
        for i in range(pc.MAX_MESSAGES_PER_CONVERSATION):
            c.append_message(_msg(message_id=f"{i:012x}"))
        with pytest.raises(ChatValidationError):
            c.append_message(_msg(message_id="ffffffffffff"))


class TestConversationRoundTrip:
    def test_full_round_trip(self) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1, SID2], title="Two sources",
            now="2026-05-26T10:00:00Z",
        )
        c.append_message(_msg(
            role=ROLE_USER, content="What surprised you?",
            created_at="2026-05-26T10:01:00Z",
            message_id="aaaaaaaaaaaa",
        ))
        c.append_message(_msg(
            role=ROLE_ASSISTANT, content="The participant pushes back at [S1].",
            created_at="2026-05-26T10:01:30Z",
            message_id="bbbbbbbbbbbb",
            citations=[Citation(SID1, "I pushed back here", 0.91, "s5w0", "s5w8")],
        ))
        d = c.to_dict()
        c2 = Conversation.from_dict(d)
        assert c2 == c


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #


class TestPromptBuilding:
    def test_format_snippets_numbers_them(self) -> None:
        out = pc.format_snippets_for_prompt([
            Citation(SID1, "first", 0.9),
            Citation(SID2, "second", 0.7),
        ])
        assert "[S1] first" in out
        assert "[S2] second" in out

    def test_format_snippets_drops_empty(self) -> None:
        out = pc.format_snippets_for_prompt([
            Citation(SID1, "real", 0.9),
            Citation(SID2, "   ", 0.4),  # whitespace only
        ])
        # Only one snippet in the output.
        assert out.count("[S") == 1

    def test_format_snippets_empty_returns_placeholder(self) -> None:
        assert "no relevant snippets" in pc.format_snippets_for_prompt([])

    def test_format_history_renders_roles(self) -> None:
        history = [
            _msg(role=ROLE_USER, content="hi", message_id="aaaaaaaaaaaa"),
            _msg(role=ROLE_ASSISTANT, content="hello", message_id="bbbbbbbbbbbb"),
        ]
        out = pc.format_history_for_prompt(history)
        assert "User: hi" in out
        assert "Assistant: hello" in out

    def test_format_history_truncates_to_recent(self) -> None:
        # 30 turns; only the last 12 should appear.
        msgs = [
            _msg(content=f"turn {i}", message_id=f"{i:012x}",
                 role=ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT)
            for i in range(30)
        ]
        out = pc.format_history_for_prompt(msgs)
        assert "turn 0" not in out
        assert "turn 29" in out

    def test_build_chat_prompt_assembles_blocks(self) -> None:
        prompt = pc.build_chat_prompt(
            user_question="What themes recur?",
            snippets=[Citation(SID1, "I find it hard to ask for help.", 0.9)],
            history=[],
        )
        # System prompt rules are present.
        assert "qualitative researcher" in prompt
        # Snippet got numbered.
        assert "[S1]" in prompt
        # User question made it through.
        assert "What themes recur?" in prompt
        # Tail instructions present.
        assert "Reply directly" in prompt

    def test_build_chat_prompt_includes_history_when_present(self) -> None:
        history = [
            _msg(role=ROLE_USER, content="prior question",
                 message_id="aaaaaaaaaaaa"),
        ]
        prompt = pc.build_chat_prompt(
            user_question="next question",
            snippets=[],
            history=history,
        )
        assert "prior question" in prompt
        assert "Conversation so far" in prompt

    def test_build_chat_prompt_no_history_omits_history_block(self) -> None:
        prompt = pc.build_chat_prompt(
            user_question="q",
            snippets=[],
            history=[],
        )
        assert "Conversation so far" not in prompt


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    @pytest.fixture
    def project_root(self, tmp_path: Path) -> Path:
        # Mirror the on-disk shape: a project must already exist.
        (tmp_path / PID).mkdir()
        return tmp_path

    def test_save_creates_chats_dir(self, project_root: Path) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1],
            title="t", now="2026-05-26T10:00:00Z",
        )
        path = pc.save_conversation(project_root, c)
        assert path.exists()
        assert path.parent.name == "chats"
        assert json.loads(path.read_text())["id"] == c.id

    def test_save_refuses_when_project_missing(self, tmp_path: Path) -> None:
        # No project dir created.
        c = Conversation.new(
            project_id=PID, source_ids=[SID1],
            title="t", now="2026-05-26T10:00:00Z",
        )
        with pytest.raises(FileNotFoundError):
            pc.save_conversation(tmp_path, c)

    def test_load_round_trip(self, project_root: Path) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1],
            title="t", now="2026-05-26T10:00:00Z",
        )
        c.append_message(_msg(message_id="aaaaaaaaaaaa"))
        pc.save_conversation(project_root, c)
        loaded = pc.load_conversation(project_root, PID, c.id)
        assert loaded == c

    def test_load_raises_when_missing(self, project_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            pc.load_conversation(project_root, PID, "ffffffffffff")

    def test_list_returns_newest_first(self, project_root: Path) -> None:
        c1 = Conversation.new(
            project_id=PID, source_ids=[SID1], title="old",
            now="2026-05-25T10:00:00Z",
            conversation_id="111111111111",
        )
        c2 = Conversation.new(
            project_id=PID, source_ids=[SID1], title="new",
            now="2026-05-26T10:00:00Z",
            conversation_id="222222222222",
        )
        pc.save_conversation(project_root, c1)
        pc.save_conversation(project_root, c2)
        out = pc.list_conversations(project_root, PID)
        assert [c.id for c in out] == ["222222222222", "111111111111"]

    def test_list_skips_malformed_files(self, project_root: Path) -> None:
        cd = pc.chats_dir(project_root, PID)
        cd.mkdir()
        # Valid file
        good = Conversation.new(
            project_id=PID, source_ids=[SID1], title="ok",
            now="2026-05-26T10:00:00Z",
        )
        pc.save_conversation(project_root, good)
        # Garbage file
        (cd / "ffffffffffff.json").write_text("{not json")
        out = pc.list_conversations(project_root, PID)
        assert len(out) == 1
        assert out[0].id == good.id

    def test_list_empty_when_no_chats_dir(self, project_root: Path) -> None:
        assert pc.list_conversations(project_root, PID) == []

    def test_delete_returns_true_when_present(
        self, project_root: Path,
    ) -> None:
        c = Conversation.new(
            project_id=PID, source_ids=[SID1], title="t",
            now="2026-05-26T10:00:00Z",
        )
        pc.save_conversation(project_root, c)
        assert pc.delete_conversation(project_root, PID, c.id) is True
        assert pc.delete_conversation(project_root, PID, c.id) is False

    def test_invalid_id_raises(self, project_root: Path) -> None:
        with pytest.raises(ChatValidationError):
            pc.conversation_path(project_root, PID, "not-hex")


# --------------------------------------------------------------------------- #
# Title heuristic
# --------------------------------------------------------------------------- #


class TestTitleHeuristic:
    def test_short_question_used_verbatim(self) -> None:
        assert pc.derive_title_from_first_question("What surprised you?") \
            == "What surprised you?"

    def test_long_question_clipped(self) -> None:
        long = "x " * 200
        assert len(pc.derive_title_from_first_question(long)) == 80

    def test_whitespace_collapsed(self) -> None:
        assert pc.derive_title_from_first_question("a\n\nb   c") == "a b c"

    def test_empty_falls_back(self) -> None:
        assert pc.derive_title_from_first_question("") == "New conversation"
        assert pc.derive_title_from_first_question("   ") == "New conversation"
