"""Conversational exploration of a project's transcripts.

This module is the pure data-and-prompt layer behind a "talk to your
project" feature: the user picks one or more sources, asks open-ended
questions, and a local LLM answers using snippets pulled from the
chosen transcripts via the embedding index. Each assistant turn
carries citations back to specific spans so the user can click
through to the editor and see the supporting text in context.

It is **not** a coding tool. The F8.10 gate exists to protect the
inductive opening of grounded theory — to stop the AI from suggesting
codes before the researcher has built any. Asking "what themes come
up around childcare?" or "where does the participant push back?"
isn't coding; it's reading-with-help. So this module skips the gate.

Design choices:

* **No engine state lives here.** The HTTP wrapper (in
  :mod:`scribe.server`) owns the AI backend, the embedding index, and
  the source-transcript loader; this module takes those in as
  arguments / callbacks. That keeps the unit tests trivial — no
  network, no filesystem of segments — and matches the pattern set
  by :mod:`scribe.transcript_tidy`.
* **Conversations are append-only.** A turn once committed never
  mutates. This mirrors the audit-log philosophy in §F9: we want a
  reproducible record of what the AI said so a researcher can cite
  it (or refute it) later.
* **Citations are anchor-shaped**, not vector-shaped. The assistant
  records the source ids + word-id ranges of the snippets it was
  given. The UI links those back to ``/edit/<job_id>?word=<wid>`` so
  citations are clickable.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# --------------------------------------------------------------------------- #
# IDs, caps, validation
# --------------------------------------------------------------------------- #


CONVERSATION_ID_RE = re.compile(r"^[a-f0-9]{12}$")
MESSAGE_ID_RE = re.compile(r"^[a-f0-9]{12}$")
SOURCE_ID_RE = re.compile(r"^[a-f0-9]{12}$")
PROJECT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

MAX_TITLE_LEN = 200
MAX_MESSAGE_LEN = 32 * 1024            # 32 KiB per turn
MAX_MESSAGES_PER_CONVERSATION = 200    # caps prompt growth + disk usage
MAX_CONVERSATIONS_PER_PROJECT = 1000

# How many embedded snippets to feed the model per turn. The trade-off
# is signal vs context-window pressure: too few snippets and the model
# answers from priors, too many and the answer drifts on tangentially
# similar text. Six was the sweet spot in the F8.3 ranker on real data;
# we use the same default here.
DEFAULT_RETRIEVAL_TOP_K = 6

# Snippets shorter than this are usually navigation noise (headers,
# stage directions in interview transcripts) — drop them so the
# model's context isn't burned on text that won't help it answer.
MIN_SNIPPET_CHARS = 40

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLES: tuple[str, ...] = (ROLE_USER, ROLE_ASSISTANT)


class ChatValidationError(ValueError):
    """Raised when conversation / message data fails validation.

    Distinct from :class:`scribe.projects.ProjectValidationError` only
    so callers can branch HTTP statuses if they want to (we don't
    today — both map to 400).
    """


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def new_message_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Citation:
    """One snippet the assistant was given when generating its turn.

    Carries enough info for the UI to render a link back to the
    editor at the right segment. We persist text *previews*, not full
    text, so a long conversation doesn't bloat — the link can
    re-fetch the live span if the user clicks through.
    """

    source_id: str
    text_preview: str
    score: float
    anchor_start_word_id: str = ""
    anchor_end_word_id: str = ""
    transcript_job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Citation":
        if not isinstance(d, Mapping):
            raise ChatValidationError("Citation payload must be an object")
        return cls(
            source_id=str(d.get("source_id", "")),
            text_preview=str(d.get("text_preview", "")),
            score=float(d.get("score", 0.0)),
            anchor_start_word_id=str(d.get("anchor_start_word_id", "")),
            anchor_end_word_id=str(d.get("anchor_end_word_id", "")),
            transcript_job_id=str(d.get("transcript_job_id", "")),
        )


@dataclass
class Message:
    """One turn in a conversation. Immutable once persisted."""

    id: str
    role: str
    content: str
    created_at: str
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Message":
        if not isinstance(d, Mapping):
            raise ChatValidationError("Message payload must be an object")
        m = cls(
            id=str(d.get("id", "")),
            role=str(d.get("role", "")),
            content=str(d.get("content", "")),
            created_at=str(d.get("created_at", "")),
            citations=[
                Citation.from_dict(c) for c in (d.get("citations") or [])
                if isinstance(c, Mapping)
            ],
        )
        m.validate()
        return m

    def validate(self) -> None:
        if not MESSAGE_ID_RE.match(self.id):
            raise ChatValidationError(
                f"Message.id must be 12-char hex; got {self.id!r}"
            )
        if self.role not in ROLES:
            raise ChatValidationError(
                f"Message.role must be one of {ROLES}; got {self.role!r}"
            )
        if not isinstance(self.content, str):
            raise ChatValidationError("Message.content must be a string")
        if len(self.content) > MAX_MESSAGE_LEN:
            raise ChatValidationError(
                f"Message.content exceeds {MAX_MESSAGE_LEN} chars"
            )


@dataclass
class Conversation:
    """A conversation scoped to one project + a chosen set of sources.

    ``source_ids`` is the conversation's *initial* corpus. We persist
    it so a user reopening an old conversation gets the same context
    even if their project's source list has grown. New conversations
    can be opened against different sets.
    """

    id: str
    project_id: str
    source_ids: list[str]
    title: str
    messages: list[Message]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["messages"] = [m.to_dict() for m in self.messages]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Conversation":
        if not isinstance(d, Mapping):
            raise ChatValidationError("Conversation payload must be an object")
        c = cls(
            id=str(d.get("id", "")),
            project_id=str(d.get("project_id", "")),
            source_ids=[str(s) for s in (d.get("source_ids") or [])],
            title=str(d.get("title", "")),
            messages=[
                Message.from_dict(m) for m in (d.get("messages") or [])
                if isinstance(m, Mapping)
            ],
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )
        c.validate()
        return c

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        source_ids: Iterable[str],
        title: str,
        now: str,
        conversation_id: str | None = None,
    ) -> "Conversation":
        sids = []
        for s in source_ids:
            s = str(s)
            if not SOURCE_ID_RE.match(s):
                raise ChatValidationError(
                    f"Invalid source_id: {s!r} (expected 12-char hex)"
                )
            sids.append(s)
        c = cls(
            id=conversation_id or new_conversation_id(),
            project_id=project_id,
            source_ids=sids,
            title=(title or "").strip()[:MAX_TITLE_LEN],
            messages=[],
            created_at=now,
            updated_at=now,
        )
        c.validate()
        return c

    def validate(self) -> None:
        if not CONVERSATION_ID_RE.match(self.id):
            raise ChatValidationError(
                f"Conversation.id must be 12-char hex; got {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ChatValidationError(
                f"Conversation.project_id must be 12-char hex; "
                f"got {self.project_id!r}"
            )
        if len(self.title) > MAX_TITLE_LEN:
            raise ChatValidationError(
                f"Conversation.title exceeds {MAX_TITLE_LEN} chars"
            )
        for s in self.source_ids:
            if not SOURCE_ID_RE.match(s):
                raise ChatValidationError(
                    f"Conversation.source_ids[*] must be 12-char hex; "
                    f"got {s!r}"
                )
        if len(self.messages) > MAX_MESSAGES_PER_CONVERSATION:
            raise ChatValidationError(
                f"Conversation has {len(self.messages)} messages "
                f"(max {MAX_MESSAGES_PER_CONVERSATION})"
            )
        for m in self.messages:
            m.validate()

    def append_message(self, message: Message) -> None:
        if len(self.messages) >= MAX_MESSAGES_PER_CONVERSATION:
            raise ChatValidationError(
                f"Conversation already has the maximum "
                f"{MAX_MESSAGES_PER_CONVERSATION} messages"
            )
        message.validate()
        self.messages.append(message)
        self.updated_at = message.created_at


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are an assistant helping a qualitative researcher explore "
    "their interview transcripts. The user has selected one or more "
    "sources from their project. You are given short snippets from "
    "those transcripts that were retrieved as most relevant to the "
    "current question.\n\n"
    "Rules:\n"
    "  1. Ground every claim in the supplied snippets. Quote when "
    "useful (use double quotes). If the snippets don't answer the "
    "question, say so plainly — do not invent.\n"
    "  2. After each substantive claim, add an inline citation in "
    "square brackets like [S1] or [S3] referring to the snippet "
    "number you used. Multiple snippets per claim are fine: [S1][S4].\n"
    "  3. Be concise. Researchers reading transcripts want signal, "
    "not summary-of-summary.\n"
    "  4. Suggest follow-up directions sparingly — one or two at the "
    "end, only when there's a clear next question worth chasing.\n"
    "  5. You are not coding the transcript and you are not "
    "diagnosing the participants. Stay descriptive."
)


def format_snippets_for_prompt(citations: Sequence[Citation]) -> str:
    """Render retrieved snippets as a numbered block for the prompt.

    The numbering ([S1], [S2], ...) is what the assistant cites with;
    keeping the output stable means the UI can map citation tokens
    back to specific :class:`Citation` records to render links.
    """
    if not citations:
        return "(no relevant snippets retrieved)"
    parts: list[str] = []
    for i, c in enumerate(citations, start=1):
        snippet = c.text_preview.strip()
        if not snippet:
            continue
        parts.append(f"[S{i}] {snippet}")
    if not parts:
        return "(no relevant snippets retrieved)"
    return "\n\n".join(parts)


def format_history_for_prompt(messages: Sequence[Message]) -> str:
    """Format prior turns as a transcript the LLM can continue.

    We keep the last N messages because most local models have
    sub-32k context windows; growing the history forever wastes
    tokens on snippets that no longer matter for the current turn.
    """
    if not messages:
        return ""
    parts: list[str] = []
    for m in messages[-12:]:  # last 6 user/assistant pairs
        prefix = "User" if m.role == ROLE_USER else "Assistant"
        parts.append(f"{prefix}: {m.content.strip()}")
    return "\n\n".join(parts)


def build_chat_prompt(
    *,
    user_question: str,
    snippets: Sequence[Citation],
    history: Sequence[Message],
) -> str:
    """Assemble the full prompt the LLM sees for one turn.

    Kept as a single function so the UI can preview the exact bytes
    being sent to the model when debugging an unhelpful answer.
    """
    blocks: list[str] = [SYSTEM_PROMPT.strip()]
    if history:
        blocks.append("--- Conversation so far ---")
        blocks.append(format_history_for_prompt(history))
    blocks.append("--- Snippets retrieved for this question ---")
    blocks.append(format_snippets_for_prompt(snippets))
    blocks.append("--- Current question ---")
    blocks.append(user_question.strip())
    blocks.append(
        "--- Your answer ---\n"
        "Reply directly. Cite snippets inline as [S<n>]."
    )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def chats_dir(projects_root: Path, project_id: str) -> Path:
    """``<projects_root>/<pid>/chats/`` — does not create the dir."""
    if not PROJECT_ID_RE.match(project_id):
        raise ChatValidationError(f"Invalid project id: {project_id!r}")
    return projects_root / project_id / "chats"


def conversation_path(
    projects_root: Path, project_id: str, conversation_id: str,
) -> Path:
    if not CONVERSATION_ID_RE.match(conversation_id):
        raise ChatValidationError(
            f"Invalid conversation id: {conversation_id!r}"
        )
    return chats_dir(projects_root, project_id) / f"{conversation_id}.json"


def save_conversation(
    projects_root: Path, conversation: Conversation,
) -> Path:
    """Persist atomically (tmp file + rename). Validates first."""
    conversation.validate()
    parent = projects_root / conversation.project_id
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its conversations."
        )
    cd = chats_dir(projects_root, conversation.project_id)
    cd.mkdir(parents=True, exist_ok=True)
    target = conversation_path(
        projects_root, conversation.project_id, conversation.id,
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_conversation(
    projects_root: Path, project_id: str, conversation_id: str,
) -> Conversation:
    p = conversation_path(projects_root, project_id, conversation_id)
    if not p.exists():
        raise FileNotFoundError(f"No conversation at {p}")
    return Conversation.from_dict(json.loads(p.read_text()))


def list_conversations(
    projects_root: Path, project_id: str,
) -> list[Conversation]:
    """Return all conversations for a project, newest first.

    Sort key is ``updated_at`` so a long-running thread stays at the
    top once it's been touched. Files that fail to parse are skipped
    silently — a hand-edited conversation shouldn't break the list.
    """
    cd = chats_dir(projects_root, project_id)
    if not cd.is_dir():
        return []
    out: list[Conversation] = []
    for p in cd.glob("*.json"):
        if p.name.endswith(".tmp"):
            continue
        try:
            out.append(Conversation.from_dict(json.loads(p.read_text())))
        except (json.JSONDecodeError, ChatValidationError, OSError):
            continue
    out.sort(key=lambda c: (c.updated_at or "", c.id), reverse=True)
    return out


def delete_conversation(
    projects_root: Path, project_id: str, conversation_id: str,
) -> bool:
    p = conversation_path(projects_root, project_id, conversation_id)
    if not p.exists():
        return False
    p.unlink()
    return True


# --------------------------------------------------------------------------- #
# Title heuristic
# --------------------------------------------------------------------------- #


def derive_title_from_first_question(question: str) -> str:
    """Use the first ~80 chars of the user's first question as the
    conversation title, so the list view shows something more useful
    than "Conversation abc123def456"."""
    cleaned = re.sub(r"\s+", " ", (question or "").strip())
    if not cleaned:
        return "New conversation"
    return cleaned[:80]
