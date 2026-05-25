"""Audit-trail export — chronological Markdown / RTF / CSV (F9.7).

Per PLANNING.md F9.7:

  > Audit trail export (chronological Markdown / Word; filterable).

The audit trail is the read-side composition of every event Scribe
has recorded against a project: F9.1 :class:`scribe.event_log.Event`
rows for *general* operations (codes, applications, memos, lock
toggles, snapshots, checkpoints, …) **plus** F9.6
:class:`scribe.ai_invocation_log.InvocationLogEntry` rows for *AI*
operations (one per per-engine record). Together they constitute the
"what happened in this project, in what order, by whom" timeline a
methods chapter or a thesis appendix needs.

This module is *pure*: every function takes already-loaded data
structures and returns text. The :func:`build_audit_trail` reader
walks the on-disk stores once and returns a list of :class:`AuditRow`
objects; the :func:`to_csv` / :func:`to_markdown` / :func:`to_rtf`
renderers operate on that list. The CLI script
(:mod:`scribe.scripts.export_audit_trail`) and the eventual HTTP
endpoint compose them.

Why one module over two stores
------------------------------

F9.1 is the canonical event log for everything except AI: when a
user creates a code, edits a memo, locks the codebook, takes a
snapshot, or creates a checkpoint, an :class:`Event` is recorded in
``projects/<pid>/events/<eid>.json``. F9.6 unifies the per-engine AI
suggestion / pass / search records (F8.3 / F8.4 / F8.5 / F8.6 / F8.7
/ F8.8) with the AI event log (F8.9). Neither is sufficient on its
own:

* F9.1 doesn't see AI invocations — they go through F8.9.
* F9.6 doesn't see codebook lock toggles, snapshots, checkpoints —
  those are pure F9.1 events.

The audit-trail export is therefore "F9.1 ∪ F9.6", de-duplicated by
construction (the two stores never reference the same row), sorted by
``timestamp`` ascending. Filters on top of that are AND-combined.

What's emitted per row
----------------------

A :class:`AuditRow` flattens both source kinds to a uniform schema:

* ``timestamp`` — ISO-8601 UTC; primary sort key.
* ``kind`` — :data:`AUDIT_KIND_EVENT` or :data:`AUDIT_KIND_AI_INVOCATION`.
* ``record_id`` — 12-char hex id of the underlying record (event id
  for F9.1, suggestion id for F9.6).
* ``actor_coder_id`` — 12-char hex coder id of the human responsible
  (or ``""`` for system / anonymous).
* ``action`` — for F9.1 events, :data:`scribe.event_log.EVENT_ACTIONS`
  (``create`` / ``update`` / ``lock`` / etc.); for F9.6 invocations,
  :data:`scribe.ai_invocation_log.INVOCATION_DECISIONS` (``pending``
  / ``accepted`` / ``modified`` / ``rejected`` / ``request_only``).
* ``entity_type`` — for F9.1 events, the affected entity type
  (``code`` / ``application`` / ``codebook`` / etc.); for F9.6
  invocations, the AI feature (``code_suggestion`` / ``memo_draft``
  / etc.).
* ``entity_id`` — 12-char hex id of the affected entity (``""`` if
  none).
* ``summary`` — one-line human description (used as the first column
  in CSV, the line item in Markdown / RTF).
* ``notes`` — free-form supplementary text. For F9.1 events, the
  event ``notes`` (e.g. an unlock reason); for F9.6 invocations, the
  ``rejection_reason`` if one was supplied.

Three formats, one shape
------------------------

* :func:`to_csv` — flat CSV; one row per :class:`AuditRow`. The format
  the supervisor opens in Excel.
* :func:`to_markdown` — chronological Markdown report grouped by
  ``YYYY-MM-DD`` headings. The format that gets pasted into a thesis
  appendix.
* :func:`to_rtf` — minimal RTF 1.x document. Same writer style as
  :mod:`scribe.codebook_export.to_rtf` and
  :mod:`scribe.retrieval_report.to_rtf`.

Filters
-------

:func:`build_audit_trail` accepts the union of useful per-store
filters:

* ``since`` / ``until`` — inclusive ISO-8601 bounds on ``timestamp``.
* ``kinds`` — restrict to one of (or both of)
  :data:`AUDIT_KINDS`.
* ``actor_coder_id`` — restrict to one human's activity.
* ``entity_type`` — F9.1-side filter (matches event ``entity_type``).
* ``action`` — F9.1-side filter (matches event ``action``).
* ``feature`` — F9.6-side filter (matches AI feature).
* ``decision`` — F9.6-side filter (matches AI decision).

Filters that don't apply to a row's source-kind are *ignored* for
that row (e.g. a ``feature`` filter doesn't drop F9.1 events). To
restrict to *only* F9.1 events or *only* F9.6 invocations, pass
``kinds=[AUDIT_KIND_EVENT]`` or ``kinds=[AUDIT_KIND_AI_INVOCATION]``.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from xml.sax.saxutils import escape as _xml_escape  # noqa: F401  (future-proofing)

from .ai_invocation_log import (
    DECISION_REQUEST_ONLY,
    INVOCATION_DECISIONS,
    InvocationLogEntry,
    build_invocation_log,
)
from .ai_provenance import AI_DECISIONS, AI_FEATURES
from .codebook_export import _rtf_escape, _rtf_para, _rtf_para_bold
from .coders import CODER_ID_RE
from .event_log import (
    EVENT_ACTIONS,
    EVENT_ENTITY_TYPES,
    Event,
    list_events,
)
from .projects import (
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Two source kinds for an audit row. Closed set; if a future Scribe
# feature adds a third kind it lands here, not as a free string.
AUDIT_KIND_EVENT = "event"
AUDIT_KIND_AI_INVOCATION = "ai_invocation"
AUDIT_KINDS: tuple[str, ...] = (
    AUDIT_KIND_EVENT,
    AUDIT_KIND_AI_INVOCATION,
)


# Maximum text the renderers emit without truncation. The summary cap
# matches F9.6's :data:`scribe.ai_invocation_log.MAX_SUMMARY_LEN` so a
# round-tripped row stays the same length. Notes can be longer (up to
# F9.1's ``MAX_NOTES_LEN``) — the renderers respect them as-is.
MAX_SUMMARY_LEN = 240


# --------------------------------------------------------------------------- #
# AuditRow dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuditRow:
    """One chronological row in the audit trail (F9.7).

    Frozen so a caller that holds onto the list returned by
    :func:`build_audit_trail` can't accidentally mutate fields and
    confuse a downstream renderer.

    See module docstring for field semantics.
    """

    timestamp: str
    kind: str
    record_id: str
    actor_coder_id: str = ""
    action: str = ""
    entity_type: str = ""
    entity_id: str = ""
    summary: str = ""
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for JSON / CSV / template rendering."""
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "record_id": self.record_id,
            "actor_coder_id": self.actor_coder_id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "summary": self.summary,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #


def _truncate(text: str, *, limit: int = MAX_SUMMARY_LEN) -> str:
    """Trim a string to ``limit`` characters with a trailing ellipsis."""
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _payload_label(payload: dict[str, Any] | None) -> str:
    """Pick a human-friendly label out of a payload dict.

    Looks for ``name`` then ``title`` then ``label``; returns the first
    non-empty string found, stripped. Empty if no label is present.
    """
    if not payload:
        return ""
    for key in ("name", "title", "label"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def summary_for_event(ev: Event) -> str:
    """Return a one-line human summary for a F9.1 event.

    Format: ``"<Action> <entity_type>[ — <label>][ (<entity_id>)]"``.
    The label is pulled from ``after`` (preferred) or ``before``. The
    full string is bounded by :data:`MAX_SUMMARY_LEN`.
    """
    action = (ev.action or "").strip()
    entity = (ev.entity_type or "").strip()
    parts: list[str] = []
    if action:
        parts.append(action.capitalize())
    if entity:
        parts.append(entity)
    head = " ".join(parts) if parts else "Event"
    label = _payload_label(ev.after) or _payload_label(ev.before)
    if label:
        head = f"{head} — {label}"
    if ev.entity_id:
        head = f"{head} ({ev.entity_id})"
    return _truncate(head)


def summary_for_invocation(entry: InvocationLogEntry) -> str:
    """Return a one-line human summary for an F9.6 invocation entry.

    Format: ``"AI <feature>: <decision>[ — <inner-summary>]"``. The
    inner summary is the per-engine summary already truncated to
    :data:`scribe.ai_invocation_log.MAX_SUMMARY_LEN`.
    """
    feature = (entry.feature or "").replace("_", " ")
    decision = entry.decision or DECISION_REQUEST_ONLY
    inner = (entry.summary or "").strip()
    head = f"AI {feature}: {decision}".strip()
    if inner:
        head = f"{head} — {inner}"
    return _truncate(head)


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def _row_from_event(ev: Event) -> AuditRow:
    extra: dict[str, str] = {}
    if ev.before is not None:
        bl = _payload_label(ev.before)
        if bl:
            extra["before_label"] = bl
    if ev.after is not None:
        al = _payload_label(ev.after)
        if al:
            extra["after_label"] = al
    if ev.diff:
        extra["diff_count"] = str(len(ev.diff))
    return AuditRow(
        timestamp=ev.created_at,
        kind=AUDIT_KIND_EVENT,
        record_id=ev.id,
        actor_coder_id=ev.actor_coder_id,
        action=ev.action,
        entity_type=ev.entity_type,
        entity_id=ev.entity_id,
        summary=summary_for_event(ev),
        notes=(ev.notes or "").strip(),
        extra=extra,
    )


def _row_from_invocation(entry: InvocationLogEntry) -> AuditRow:
    actor = entry.decided_by_coder_id or entry.requested_by_coder_id
    extra: dict[str, str] = {}
    if entry.generation_model:
        extra["generation_model"] = entry.generation_model
    if entry.embedding_model:
        extra["embedding_model"] = entry.embedding_model
    if entry.related_entity_ids:
        extra["related_entity_ids"] = ",".join(entry.related_entity_ids)
    if entry.ai_event_ids:
        extra["ai_event_ids"] = ",".join(entry.ai_event_ids)
    if entry.decided_at:
        extra["decided_at"] = entry.decided_at
    primary_entity = (
        entry.related_entity_ids[0] if entry.related_entity_ids else ""
    )
    return AuditRow(
        timestamp=entry.created_at,
        kind=AUDIT_KIND_AI_INVOCATION,
        record_id=entry.suggestion_id,
        actor_coder_id=actor,
        action=entry.decision or DECISION_REQUEST_ONLY,
        entity_type=entry.feature,
        entity_id=primary_entity,
        summary=summary_for_invocation(entry),
        notes=(entry.rejection_reason or "").strip(),
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


def _validate_filter_in(
    label: str, value: str | None, allowed: Sequence[str]
) -> None:
    if value is None:
        return
    if value not in allowed:
        raise ProjectValidationError(
            f"Invalid {label} filter: {value!r}; expected one of {allowed!r}"
        )


def build_audit_trail(
    projects_root: Path,
    project_id: str,
    *,
    since: str | None = None,
    until: str | None = None,
    kinds: Sequence[str] | None = None,
    actor_coder_id: str | None = None,
    entity_type: str | None = None,
    action: str | None = None,
    feature: str | None = None,
    decision: str | None = None,
) -> list[AuditRow]:
    """Walk F9.1 events + F9.6 invocations into a chronological audit trail.

    Returned rows are sorted by ``(timestamp, kind, record_id)`` —
    timestamp first (oldest → newest), then a stable tie-break on
    ``kind`` and ``record_id``. Two events with the same timestamp
    appear in deterministic order across runs.

    Filters
    -------

    All filters AND-combine. Filters that don't apply to a row's
    source kind are *not* applied to that row — see module docstring.
    The exception is ``kinds``: it acts as the "include this row at
    all?" gate.

    Raises
    ------
    ProjectValidationError
        If any filter argument doesn't match its closed-set domain.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    if kinds is not None:
        bad = [k for k in kinds if k not in AUDIT_KINDS]
        if bad:
            raise ProjectValidationError(
                f"Invalid kinds filter: {bad!r}; expected subset of "
                f"{AUDIT_KINDS!r}"
            )
        kinds_set: set[str] | None = set(kinds)
        if not kinds_set:
            raise ProjectValidationError(
                "kinds filter must be None or non-empty"
            )
    else:
        kinds_set = None
    if (
        actor_coder_id is not None
        and actor_coder_id
        and not CODER_ID_RE.match(actor_coder_id)
    ):
        raise ProjectValidationError(
            f"Invalid actor_coder_id filter: {actor_coder_id!r}"
        )
    _validate_filter_in("entity_type", entity_type, EVENT_ENTITY_TYPES)
    _validate_filter_in("action", action, EVENT_ACTIONS)
    _validate_filter_in("feature", feature, AI_FEATURES)
    _validate_filter_in("decision", decision, INVOCATION_DECISIONS)

    rows: list[AuditRow] = []

    # F9.1 — pass per-store filters down so the file walker doesn't
    # parse rows we'd just throw away.
    if kinds_set is None or AUDIT_KIND_EVENT in kinds_set:
        events = list_events(
            projects_root,
            project_id,
            action=action,
            entity_type=entity_type,
            actor_coder_id=actor_coder_id if actor_coder_id else None,
            since=since,
            until=until,
        )
        for ev in events:
            rows.append(_row_from_event(ev))

    # F9.6 — same idea, with the AI-side filters.
    if kinds_set is None or AUDIT_KIND_AI_INVOCATION in kinds_set:
        invocations = build_invocation_log(
            projects_root,
            project_id,
            feature=feature,
            decision=decision,
            actor_coder_id=actor_coder_id if actor_coder_id else None,
            since=since,
            until=until,
        )
        for entry in invocations:
            rows.append(_row_from_invocation(entry))

    rows.sort(key=lambda r: (r.timestamp, r.kind, r.record_id))
    return rows


# --------------------------------------------------------------------------- #
# CSV renderer
# --------------------------------------------------------------------------- #


# Public column contract. Order is part of the schema — downstream
# scripts can index by position. New fields go at the *end* so old
# consumers keep working.
CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "kind",
    "record_id",
    "actor_coder_id",
    "action",
    "entity_type",
    "entity_id",
    "summary",
    "notes",
)


def to_csv(rows: Iterable[AuditRow]) -> str:
    """Serialise rows to RFC-4180 CSV text.

    Columns: see :data:`CSV_COLUMNS`. Empty input yields header-only
    output (still valid). The renderer never raises on missing fields
    — defaults from the dataclass apply.

    Line endings are CRLF per RFC 4180; callers writing the bytes to
    disk should use ``write_bytes`` to avoid the host platform
    rewriting them.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.timestamp,
                row.kind,
                row.record_id,
                row.actor_coder_id,
                row.action,
                row.entity_type,
                row.entity_id,
                row.summary,
                row.notes,
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Markdown renderer
# --------------------------------------------------------------------------- #


_DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _day_for(timestamp: str) -> str:
    """Return the ``YYYY-MM-DD`` prefix of a timestamp, or ``""``.

    We don't try to parse weird shapes — Scribe writes ISO-8601 UTC
    everywhere, so the regex is fine. Falls back to an empty day for a
    malformed timestamp; the renderer groups those under "Undated".
    """
    if not timestamp:
        return ""
    m = _DAY_RE.match(timestamp)
    return m.group(1) if m else ""


def _md_escape(text: str) -> str:
    """Minimal Markdown escape for inline use.

    We don't want a stray backtick or pipe to break the rendered
    table. Newlines collapse to spaces so a multi-line ``notes`` field
    stays on one line in Markdown — full text remains in CSV / RTF.
    """
    s = (text or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = s.replace("`", "\\`").replace("|", "\\|")
    return s.strip()


def to_markdown(
    rows: Sequence[AuditRow], *, project: Project | None = None
) -> str:
    """Render the audit trail as a chronological Markdown document.

    Layout:

    1. ``# Audit trail`` heading (with project name when supplied).
    2. Project metadata (methodology, stage, row count) if a
       ``project`` is provided.
    3. ``## YYYY-MM-DD`` heading per day, in chronological order.
    4. Per-row line: ``- HH:MM:SS · <kind> · <action> · <summary>``,
       with the actor / record id appended in parentheses, and a
       blockquoted ``notes`` line beneath when non-empty.

    Empty input produces a heading-only document; the renderer never
    raises.
    """
    lines: list[str] = []
    title = "Audit trail"
    if project is not None and project.name and project.name.strip():
        title = f"Audit trail — {project.name.strip()}"
    lines.append(f"# {title}")
    lines.append("")

    if project is not None:
        meta_rows: list[tuple[str, str]] = []
        if project.methodology:
            meta_rows.append(("Methodology", project.methodology))
        if project.codebook_stage:
            meta_rows.append(("Stage", project.codebook_stage))
        meta_rows.append(("Rows", str(len(rows))))
        for label, value in meta_rows:
            lines.append(f"- **{label}**: {value}")
        lines.append("")

    if not rows:
        lines.append("_(no audit-trail entries)_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # Group by ``YYYY-MM-DD``; preserve original chronological order
    # of the rows (already sorted by timestamp).
    current_day: str | None = None
    for row in rows:
        day = _day_for(row.timestamp) or "Undated"
        if day != current_day:
            current_day = day
            lines.append(f"## {day}")
            lines.append("")
        time_part = ""
        if row.timestamp and len(row.timestamp) >= 19:
            # ISO-8601: the time-of-day starts at index 11 (``T``) and
            # runs ``HH:MM:SS``. Cheaper than a strptime; we control
            # the format.
            time_part = row.timestamp[11:19]
        bits: list[str] = []
        if time_part:
            bits.append(time_part)
        bits.append(row.kind)
        if row.action:
            bits.append(row.action)
        if row.summary:
            bits.append(_md_escape(row.summary))
        else:
            bits.append("(no summary)")
        suffix_bits: list[str] = []
        if row.actor_coder_id:
            suffix_bits.append(f"actor={row.actor_coder_id}")
        suffix_bits.append(f"id={row.record_id}")
        head = " · ".join(bits)
        lines.append(f"- {head} ({', '.join(suffix_bits)})")
        if row.notes:
            note_line = _md_escape(row.notes)
            if note_line:
                lines.append(f"  > {note_line}")
        # No blank line per row — the day headings break it up.
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# RTF renderer
# --------------------------------------------------------------------------- #


def to_rtf(
    rows: Sequence[AuditRow], *, project: Project | None = None
) -> str:
    """Serialise the audit trail as a minimal RTF 1.x document.

    Word, LibreOffice, and Pages all open RTF natively. The output
    style mirrors :func:`scribe.codebook_export.to_rtf` and
    :func:`scribe.retrieval_report.to_rtf` — bold day headings,
    plain-paragraph rows.

    Empty input produces a heading-only document with a "(no
    entries)" placeholder; the renderer never raises.
    """
    parts: list[str] = []
    parts.append(r"{\rtf1\ansi\ansicpg1252\deff0")
    parts.append(r"{\fonttbl{\f0\fnil Calibri;}}")
    parts.append(r"\fs22")  # 11pt body

    title = "Audit trail"
    if project is not None and project.name and project.name.strip():
        title = f"Audit trail — {project.name.strip()}"
    parts.append(_rtf_para_bold(title, fs=36))

    if project is not None:
        meta_lines: list[str] = []
        if project.methodology:
            meta_lines.append(f"Methodology: {project.methodology}")
        if project.codebook_stage:
            meta_lines.append(f"Stage: {project.codebook_stage}")
        meta_lines.append(f"Rows: {len(rows)}")
        for ml in meta_lines:
            parts.append(_rtf_para(ml))
        parts.append(r"\par")

    if not rows:
        parts.append(_rtf_para("(no audit-trail entries)"))
        parts.append("}")
        return "".join(parts)

    current_day: str | None = None
    for row in rows:
        day = _day_for(row.timestamp) or "Undated"
        if day != current_day:
            current_day = day
            parts.append(_rtf_para_bold(day, fs=28))
        time_part = ""
        if row.timestamp and len(row.timestamp) >= 19:
            time_part = row.timestamp[11:19]
        bits: list[str] = []
        if time_part:
            bits.append(time_part)
        bits.append(row.kind)
        if row.action:
            bits.append(row.action)
        if row.summary:
            bits.append(row.summary)
        else:
            bits.append("(no summary)")
        suffix_bits: list[str] = []
        if row.actor_coder_id:
            suffix_bits.append(f"actor={row.actor_coder_id}")
        suffix_bits.append(f"id={row.record_id}")
        line = " · ".join(bits) + f" ({', '.join(suffix_bits)})"
        parts.append(_rtf_para(line))
        if row.notes:
            # Indent notes a touch with a leading bullet so they read
            # as supplementary in Word.
            parts.append(_rtf_para(f"    note: {row.notes}"))

    parts.append("}")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Format dispatch + filename helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FormatSpec:
    """Static description of an audit-trail export format."""

    key: str
    extension: str
    media_type: str
    label: str


EXPORT_FORMAT_CSV = "csv"
EXPORT_FORMAT_MARKDOWN = "markdown"
EXPORT_FORMAT_RTF = "rtf"


EXPORT_FORMATS: dict[str, FormatSpec] = {
    EXPORT_FORMAT_CSV: FormatSpec(
        key=EXPORT_FORMAT_CSV,
        extension=".csv",
        media_type="text/csv; charset=utf-8",
        label="CSV",
    ),
    EXPORT_FORMAT_MARKDOWN: FormatSpec(
        key=EXPORT_FORMAT_MARKDOWN,
        extension=".md",
        media_type="text/markdown; charset=utf-8",
        label="Markdown",
    ),
    EXPORT_FORMAT_RTF: FormatSpec(
        key=EXPORT_FORMAT_RTF,
        extension=".rtf",
        media_type="application/rtf",
        label="RTF (Word)",
    ),
}


_FORMAT_ALIASES: dict[str, str] = {
    "csv": EXPORT_FORMAT_CSV,
    "md": EXPORT_FORMAT_MARKDOWN,
    "markdown": EXPORT_FORMAT_MARKDOWN,
    "rtf": EXPORT_FORMAT_RTF,
    "word": EXPORT_FORMAT_RTF,
    "doc": EXPORT_FORMAT_RTF,
    "docx": EXPORT_FORMAT_RTF,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Case-insensitive; trims whitespace; recognises the ``md`` /
    ``word`` / ``doc`` / ``docx`` aliases. Raises :class:`ValueError`
    on unknown inputs with the list of supported keys.
    """
    if format is None:
        raise ValueError(
            "Audit-trail export format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported audit-trail export format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


_RENDERERS: dict[str, Callable[..., str]] = {}


def render_audit_trail(
    format: str,
    rows: Sequence[AuditRow],
    *,
    project: Project | None = None,
) -> str:
    """Render the audit trail in ``format`` and return the body string.

    Empty rows are valid input. ``project`` is forwarded only to
    Markdown / RTF — CSV intentionally has no header row beyond the
    column names so the schema stays stable.
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](rows, project=project)


def _render_csv(rows: Sequence[AuditRow], *, project: Project | None) -> str:
    del project
    return to_csv(rows)


def _render_markdown(
    rows: Sequence[AuditRow], *, project: Project | None
) -> str:
    return to_markdown(rows, project=project)


def _render_rtf(rows: Sequence[AuditRow], *, project: Project | None) -> str:
    return to_rtf(rows, project=project)


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_MARKDOWN] = _render_markdown
_RENDERERS[EXPORT_FORMAT_RTF] = _render_rtf


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def slugify_audit_trail_filename(
    project: Project | None, format: str
) -> str:
    """Build a download-friendly filename for an audit-trail export.

    Pattern: ``<slug>-audit-trail<ext>`` with a project name,
    ``audit-trail<ext>`` otherwise. Same ASCII-only / dash-separated
    / NFKD-normalised slug as the codebook exporter.
    """
    fmt = normalise_format(format)
    spec = EXPORT_FORMATS[fmt]
    slug = ""
    if project is not None and project.name and project.name.strip():
        ascii_name = (
            unicodedata.normalize("NFKD", project.name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        slug = _SLUG_RE.sub("-", ascii_name.lower()).strip("-")
        if len(slug) > _FILENAME_SLUG_MAX:
            slug = slug[:_FILENAME_SLUG_MAX].rstrip("-")
    if slug:
        return f"{slug}-audit-trail{spec.extension}"
    return f"audit-trail{spec.extension}"


def write_audit_trail(
    path: Path,
    format: str,
    rows: Sequence[AuditRow],
    *,
    project: Project | None = None,
) -> Path:
    """Render the audit trail and atomically write it to ``path``.

    Writes are atomic via a ``<path>.tmp`` swap; an interrupted write
    leaves no half-finished output visible. Creates ``path.parent`` if
    missing. Returns the final ``path``.
    """
    fmt = normalise_format(format)
    text = render_audit_trail(fmt, rows, project=project)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(target)
    return target


__all__ = [
    "AUDIT_KIND_AI_INVOCATION",
    "AUDIT_KIND_EVENT",
    "AUDIT_KINDS",
    "AuditRow",
    "CSV_COLUMNS",
    "EXPORT_FORMATS",
    "EXPORT_FORMAT_CSV",
    "EXPORT_FORMAT_MARKDOWN",
    "EXPORT_FORMAT_RTF",
    "FormatSpec",
    "MAX_SUMMARY_LEN",
    "build_audit_trail",
    "normalise_format",
    "render_audit_trail",
    "slugify_audit_trail_filename",
    "summary_for_event",
    "summary_for_invocation",
    "to_csv",
    "to_markdown",
    "to_rtf",
    "write_audit_trail",
]
