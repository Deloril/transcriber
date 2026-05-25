"""Coded-segment retrieval report (F6.2).

Per PLANNING.md F6.2:

  > Coded-segment retrieval report (per code, filterable, grouped by
  > source / participant).

A *retrieval report* is the bread-and-butter analysis output of any
QDA tool: "show me every place I applied this code, ordered by source,
with the actual quoted text and who coded it." A researcher pulls one
of these to write a results section, to share a corpus of supporting
quotes with a co-author, or to spot-check their own coding.

F2.6 / F6.1 already shipped *codebook* exporters (definitions, not
applications). F6.2 is the application-level companion: every output
row is a single :class:`scribe.applications.Application`, hydrated
with its code, source, coder, and participants, optionally with the
quoted text pulled from a transcript.

This module is **pure**: every function takes already-loaded entities
and returns text or row tuples. No filesystem I/O, no FastAPI, no
engine imports — same shape as :mod:`scribe.codebook_export`. The CLI
(:mod:`scribe.scripts.export_retrieval_report`) and the eventual HTTP
endpoint do the disk reads and call in here for the rendering.

What a row carries
------------------

A :class:`RetrievalRow` flattens the cross-entity references into
display-ready fields:

* ``code_name`` from the matching :class:`scribe.codes.Code` (looked
  up by ``code_id``).
* ``source_name`` from the matching :class:`scribe.sources.Source`.
* ``coder_name`` from the matching :class:`scribe.coders.Coder` —
  empty when no coder list is supplied.
* ``participant_ids`` / ``participant_names`` — the participants
  associated with this row's source. Many sources have one
  participant, but focus groups (F3.3) have several; both are
  carried so per-participant grouping never hides anyone.
* ``text`` — the quoted text, extracted from
  ``segments_by_source[source_id]`` when supplied. Optional: a row
  remains valid (and useful for index reports) when the transcript
  isn't available.
* timestamps + confidence + provenance + note — flat-table-friendly
  views of the application's own fields.

Three formats, one shape
------------------------

* :func:`to_csv` — flat CSV, one row per application. The format the
  supervisor opens in Excel or pipes into another tool.
* :func:`to_markdown` — grouped Markdown report with code / source /
  participant headings. The format that gets pasted into a results
  section.
* :func:`to_rtf` — minimal RTF 1.x document. Same writer style as
  :mod:`scribe.codebook_export.to_rtf` so the output renders
  consistently in Word / LibreOffice / Pages.

Grouping
--------

The Markdown and RTF renderers accept ``group_by="code" | "source" |
"participant" | "none"``. The CSV renderer is intentionally flat —
CSV consumers want one schema, not an indented document — but it
*does* accept an ``order_by`` argument so the row order matches a
chosen grouping when piped through another tool.

The default is ``group_by="code"`` (the F6.2 phrasing is "per code").
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .application_reanchor import anchored_words
from .applications import Application
from .codebook_export import _rtf_escape, _rtf_para, _rtf_para_bold
from .coders import Coder
from .codes import Code
from .participants import Participant
from .projects import Project
from .sources import Source


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


GROUP_BY_CODE = "code"
GROUP_BY_SOURCE = "source"
GROUP_BY_PARTICIPANT = "participant"
GROUP_BY_NONE = "none"

# Public ordering of group-by keys: drives the CLI ``--group-by``
# choices and the error message when an unknown value is passed.
GROUP_BY_KEYS: tuple[str, ...] = (
    GROUP_BY_CODE,
    GROUP_BY_SOURCE,
    GROUP_BY_PARTICIPANT,
    GROUP_BY_NONE,
)


# Sentinels used as placeholder labels when a row's grouping target is
# missing. "(unknown)" reads better in a Word document than an empty
# heading; "(no participant)" makes participant grouping legible when
# a source has none linked to it.
LABEL_UNKNOWN = "(unknown)"
LABEL_NO_PARTICIPANT = "(no participant)"


# CSV columns. Order is part of the public contract — new fields go
# at the end so old consumer scripts don't drift. Mirrors
# :mod:`scribe.codebook_export`'s convention.
CSV_COLUMNS: tuple[str, ...] = (
    "application_id",
    "code_id",
    "code_name",
    "source_id",
    "source_name",
    "coder_id",
    "coder_name",
    "participant_ids",
    "participant_names",
    "anchor_start_word_id",
    "anchor_end_word_id",
    "start_char_offset",
    "end_char_offset",
    "text",
    "confidence",
    "provenance_source",
    "note",
    "created_at",
)


# Multi-valued cell separator: same character used by codebook_export.
CSV_LIST_SEP = " | "


# --------------------------------------------------------------------------- #
# Row data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievalRow:
    """One coded-segment in a retrieval report.

    Fields are intentionally flat strings (with a couple of
    optional-numeric exceptions for ``confidence`` and the char
    offsets) so the same row can drop straight into a CSV cell or a
    Markdown bullet point. ``participant_ids`` and ``participant_names``
    are tuples to preserve the focus-group case (one source, many
    participants, F3.3) while still being hashable.
    """

    application_id: str
    code_id: str
    code_name: str
    source_id: str
    source_name: str
    coder_id: str
    coder_name: str
    participant_ids: tuple[str, ...]
    participant_names: tuple[str, ...]
    anchor_start_word_id: str
    anchor_end_word_id: str
    start_char_offset: int | None
    end_char_offset: int | None
    text: str
    confidence: float | None
    provenance_source: str
    note: str
    created_at: str


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def _index(items: Iterable[object], attr: str = "id") -> dict[str, object]:
    """Index a sequence by attribute (default ``id``)."""
    out: dict[str, object] = {}
    for item in items:
        key = getattr(item, attr, None)
        if isinstance(key, str):
            out[key] = item
    return out


def _participants_by_source(
    participants: Sequence[Participant],
) -> dict[str, list[Participant]]:
    """Map ``source_id → list[Participant]`` from the participants list.

    A participant appears once per source it's linked to. Order within
    a source mirrors the order participants are listed in (matches
    ``list_participants`` — created_at ascending then id). Used by
    :func:`build_retrieval_rows` so the resulting rows expose the
    same focus-group-friendly structure the rest of the F3.3 stack
    uses.
    """
    out: dict[str, list[Participant]] = {}
    for p in participants:
        for sid in p.source_ids:
            out.setdefault(sid, []).append(p)
    return out


def _extract_text(
    application: Application,
    segments_by_source: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> str:
    """Pull the quoted text for an application from the supplied segments.

    Returns ``""`` if no segments map is supplied or the application's
    ``source_id`` is missing from it. Returns ``""`` (rather than
    raising) when the anchor falls outside the supplied transcript —
    F4.5's orphan queue is the right place to surface those, not a
    rendering helper.

    Sub-word ``start_char_offset`` / ``end_char_offset`` on the
    application *are* honoured so a quote that starts mid-word
    (``"...crimi***nalisation***..."``) renders as the actual coded
    fragment, not the whole word. Whole-word anchors (offsets None)
    fall through with the words joined by single spaces — the same
    join the editor uses for plain-text export.
    """
    if segments_by_source is None:
        return ""
    segs = segments_by_source.get(application.source_id)
    if segs is None:
        return ""
    words = anchored_words(application, segs)
    if not words:
        return ""

    # Apply sub-word offsets on the boundary words. Special-case
    # single-word anchors (start == end) so both offsets slice the
    # *same* original token: word[start:end], not (word[start:])[:end],
    # which would lose characters.
    so = application.start_char_offset
    eo = application.end_char_offset
    if len(words) == 1 and (so is not None or eo is not None):
        only = words[0]
        lo = 0 if so is None else max(0, min(so, len(only)))
        hi = len(only) if eo is None else max(0, min(eo, len(only)))
        if hi < lo:
            hi = lo
        words = [only[lo:hi]]
    else:
        if so is not None and words:
            first = words[0]
            lo = max(0, min(so, len(first)))
            words = [first[lo:], *words[1:]]
        if eo is not None and words:
            last = words[-1]
            hi = max(0, min(eo, len(last)))
            words = [*words[:-1], last[:hi]]

    # Drop any word that became empty after offset slicing — rare, but
    # keeps the joined output clean (no leading / trailing spaces).
    words = [w for w in words if w]
    return " ".join(words)


def build_retrieval_rows(
    *,
    applications: Sequence[Application],
    codes: Sequence[Code] = (),
    sources: Sequence[Source] = (),
    coders: Sequence[Coder] = (),
    participants: Sequence[Participant] = (),
    segments_by_source: (
        Mapping[str, Sequence[Mapping[str, object]]] | None
    ) = None,
) -> list[RetrievalRow]:
    """Hydrate applications into a list of :class:`RetrievalRow`.

    The lookup tables are all optional so callers can build "id-only"
    reports during early development (no codebook yet) and incrementally
    plumb in richer data. Missing-name lookups surface as empty strings
    rather than placeholders so a downstream tool can detect them.

    Row order matches application order. F4.2's :func:`sort_by_anchor`
    is the right helper to apply *before* this function if you want
    document order; CSV-by-creation-time (the
    :func:`scribe.applications.list_applications` default) is what most
    callers will hand in.
    """
    code_index = _index(codes)
    source_index = _index(sources)
    coder_index = _index(coders)
    participants_for = _participants_by_source(participants)

    rows: list[RetrievalRow] = []
    for app in applications:
        code = code_index.get(app.code_id)
        source = source_index.get(app.source_id)
        coder = coder_index.get(app.coder_id)
        ps = participants_for.get(app.source_id, [])
        text = _extract_text(app, segments_by_source)
        rows.append(
            RetrievalRow(
                application_id=app.id,
                code_id=app.code_id,
                code_name=getattr(code, "name", "") if code is not None else "",
                source_id=app.source_id,
                source_name=getattr(source, "name", "") if source is not None else "",
                coder_id=app.coder_id,
                coder_name=getattr(coder, "name", "") if coder is not None else "",
                participant_ids=tuple(p.id for p in ps),
                participant_names=tuple(p.name for p in ps),
                anchor_start_word_id=app.anchor_start_word_id,
                anchor_end_word_id=app.anchor_end_word_id,
                start_char_offset=app.start_char_offset,
                end_char_offset=app.end_char_offset,
                text=text,
                confidence=app.confidence,
                provenance_source=str(app.provenance.get("source", "")),
                note=app.note,
                created_at=app.created_at,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def filter_rows(
    rows: Sequence[RetrievalRow],
    *,
    code_ids: Iterable[str] | None = None,
    source_ids: Iterable[str] | None = None,
    coder_ids: Iterable[str] | None = None,
    participant_ids: Iterable[str] | None = None,
) -> list[RetrievalRow]:
    """Return rows matching every supplied filter (AND-combined).

    Any filter set to ``None`` is treated as "don't filter on this
    field". An empty iterable means "match nothing for this field" —
    which is occasionally what a UI control sends when the user has
    selected zero entries; surfacing that as "no rows" is more
    honest than silently returning everything.

    Matching is by exact id. Participant filtering checks whether
    *any* of a row's participant_ids appears in the supplied set
    (focus-group rows match if any of their participants do).
    """
    code_set = frozenset(code_ids) if code_ids is not None else None
    source_set = frozenset(source_ids) if source_ids is not None else None
    coder_set = frozenset(coder_ids) if coder_ids is not None else None
    part_set = (
        frozenset(participant_ids) if participant_ids is not None else None
    )

    out: list[RetrievalRow] = []
    for r in rows:
        if code_set is not None and r.code_id not in code_set:
            continue
        if source_set is not None and r.source_id not in source_set:
            continue
        if coder_set is not None and r.coder_id not in coder_set:
            continue
        if part_set is not None:
            if not any(pid in part_set for pid in r.participant_ids):
                continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievalGroup:
    """One group of rows under a heading.

    ``key`` is a stable identifier (the code/source/participant id, or
    ``""`` for the "ungrouped" placeholder). ``label`` is the human-
    readable heading for that group. ``rows`` is the rows that fell
    into the group, in insertion order.
    """

    key: str
    label: str
    rows: tuple[RetrievalRow, ...]


def normalise_group_by(group_by: str | None) -> str:
    """Resolve a caller-supplied group-by string to a canonical key.

    Defaults: ``None`` → ``"code"``. Trims + lower-cases and accepts
    the obvious aliases ("codes" / "sources" / "participants" /
    "flat"). Raises :class:`ValueError` for anything else.
    """
    if group_by is None:
        return GROUP_BY_CODE
    g = str(group_by).strip().lower()
    aliases = {
        "code": GROUP_BY_CODE,
        "codes": GROUP_BY_CODE,
        "source": GROUP_BY_SOURCE,
        "sources": GROUP_BY_SOURCE,
        "participant": GROUP_BY_PARTICIPANT,
        "participants": GROUP_BY_PARTICIPANT,
        "none": GROUP_BY_NONE,
        "flat": GROUP_BY_NONE,
        "": GROUP_BY_CODE,
    }
    if g in aliases:
        return aliases[g]
    raise ValueError(
        f"Unsupported group_by: {group_by!r}. Expected one of: "
        f"{list(GROUP_BY_KEYS)}"
    )


def group_rows(
    rows: Sequence[RetrievalRow], *, group_by: str = GROUP_BY_CODE
) -> list[RetrievalGroup]:
    """Bucket rows into groups, returning a deterministic ordered list.

    Group ordering inside the result is by *first appearance of the
    group key in the input*, so a caller who has pre-sorted rows
    (e.g. by ``sort_by_anchor`` for document order) gets matching
    group order without a second sort pass. Within a group, row
    order is preserved.

    For ``group_by == "participant"``, rows whose source has multiple
    linked participants appear in **every** matching group (focus-
    group support per F3.3). Rows with no participants land in a
    single ``"(no participant)"`` bucket so they're still surfaced.
    """
    g = normalise_group_by(group_by)

    if g == GROUP_BY_NONE:
        return [RetrievalGroup(key="", label="", rows=tuple(rows))]

    # Use a list-of-keys + dict pattern so we keep first-appearance
    # ordering deterministically.
    keys_in_order: list[str] = []
    bucket: dict[str, tuple[str, list[RetrievalRow]]] = {}

    def push(key: str, label: str, row: RetrievalRow) -> None:
        if key not in bucket:
            keys_in_order.append(key)
            bucket[key] = (label, [])
        bucket[key][1].append(row)

    for r in rows:
        if g == GROUP_BY_CODE:
            push(r.code_id, r.code_name or LABEL_UNKNOWN, r)
        elif g == GROUP_BY_SOURCE:
            push(r.source_id, r.source_name or LABEL_UNKNOWN, r)
        else:  # participant
            if not r.participant_ids:
                push("", LABEL_NO_PARTICIPANT, r)
            else:
                for pid, pname in zip(r.participant_ids, r.participant_names):
                    push(pid, pname or LABEL_UNKNOWN, r)

    return [
        RetrievalGroup(key=k, label=bucket[k][0], rows=tuple(bucket[k][1]))
        for k in keys_in_order
    ]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _format_offset(v: int | None) -> str:
    """Render a sub-word char offset for CSV: ``""`` for None, str(int) else."""
    return "" if v is None else str(v)


def _format_confidence(v: float | None) -> str:
    """Render confidence for CSV: ``""`` for None, repr that doesn't lose precision else.

    We avoid scientific notation (``"1e-05"``) so the cell pastes into
    Excel cleanly; the format used is ``g`` with 6 significant digits,
    which round-trips for any float we'd realistically see in a
    confidence cell.
    """
    if v is None:
        return ""
    return format(float(v), ".6g")


def to_csv(rows: Sequence[RetrievalRow]) -> str:
    """Serialise retrieval rows to CSV (RFC 4180, ``\\r\\n`` line endings).

    Empty input produces a header-only document — that's a valid
    "no matches" report, not an error. List-valued cells
    (``participant_ids``, ``participant_names``) are joined with
    :data:`CSV_LIST_SEP`, matching :mod:`scribe.codebook_export`'s
    convention.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for r in rows:
        writer.writerow(
            [
                r.application_id,
                r.code_id,
                r.code_name,
                r.source_id,
                r.source_name,
                r.coder_id,
                r.coder_name,
                CSV_LIST_SEP.join(r.participant_ids),
                CSV_LIST_SEP.join(r.participant_names),
                r.anchor_start_word_id,
                r.anchor_end_word_id,
                _format_offset(r.start_char_offset),
                _format_offset(r.end_char_offset),
                r.text,
                _format_confidence(r.confidence),
                r.provenance_source,
                r.note,
                r.created_at,
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _md_quote_block(text: str) -> list[str]:
    """Render text as a Markdown blockquote, line by line."""
    if not text:
        return ["> _(no transcript text available)_"]
    return [f"> {ln}" if ln else ">" for ln in text.splitlines() or [""]]


def _md_row_block(r: RetrievalRow, *, group_by: str) -> list[str]:
    """Render one row as a Markdown sub-block.

    Inline metadata line uses bullet separators (``·``) matching the
    codebook export's heading style. Fields that are obvious from the
    surrounding group heading (e.g. code name when group_by="code")
    are *not* repeated, so the document doesn't shout the same
    heading twice.
    """
    out: list[str] = []
    bits: list[str] = [f"`{r.application_id}`"]
    if group_by != GROUP_BY_CODE and r.code_name:
        bits.append(f"code: {r.code_name}")
    if group_by != GROUP_BY_SOURCE and r.source_name:
        bits.append(f"source: {r.source_name}")
    if r.coder_name:
        bits.append(f"coder: {r.coder_name}")
    if (
        group_by != GROUP_BY_PARTICIPANT
        and r.participant_names
    ):
        bits.append(
            "participant: " + ", ".join(r.participant_names)
        )
    if r.confidence is not None:
        bits.append(f"confidence: {_format_confidence(r.confidence)}")
    if r.provenance_source:
        bits.append(f"provenance: {r.provenance_source}")
    if r.created_at:
        bits.append(f"at {r.created_at}")
    out.append(" · ".join(bits))
    out.append("")
    out.extend(_md_quote_block(r.text))
    out.append("")
    if r.note:
        out.append(f"_Note:_ {r.note}")
        out.append("")
    return out


def to_markdown(
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None = None,
    group_by: str = GROUP_BY_CODE,
) -> str:
    """Serialise retrieval rows to a structured Markdown document.

    Layout:

    1. ``# Coded segments`` heading (optionally including project name).
    2. Project metadata bullets if a ``project`` is supplied.
    3. ``## <group label>`` per group (per ``group_by``), each with a
       block per row containing inline metadata, a Markdown
       blockquote of the quoted text, and an optional ``_Note:_`` line.

    Empty input renders the heading + a placeholder
    ``_(no coded segments)_`` line — a valid empty report.
    """
    g = normalise_group_by(group_by)
    lines: list[str] = []
    title = "Coded segments"
    if project is not None and project.name.strip():
        title = f"Coded segments — {project.name}"
    lines.append(f"# {title}")
    lines.append("")

    if project is not None:
        meta_rows: list[tuple[str, str]] = []
        if project.methodology:
            meta_rows.append(("Methodology", project.methodology))
        if project.codebook_stage:
            meta_rows.append(("Stage", project.codebook_stage))
        meta_rows.append(("Rows", str(len(rows))))
        if g != GROUP_BY_NONE:
            meta_rows.append(("Grouped by", g))
        for label, value in meta_rows:
            lines.append(f"- **{label}**: {value}")
        lines.append("")

    if not rows:
        lines.append("_(no coded segments)_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    groups = group_rows(rows, group_by=g)
    for grp in groups:
        if g != GROUP_BY_NONE:
            lines.append(f"## {grp.label}")
            if grp.key:
                lines.append("")
                lines.append(f"`{grp.key}` · {len(grp.rows)} segment(s)")
            else:
                lines.append("")
                lines.append(f"{len(grp.rows)} segment(s)")
            lines.append("")
        for r in grp.rows:
            lines.extend(_md_row_block(r, group_by=g))

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


def _rtf_blockquote(text: str) -> list[str]:
    """Render text as a single italic paragraph (RTF stand-in for blockquote).

    True blockquotes in RTF require ``\\li`` / ``\\ri`` indents and a
    fair bit of state-machine bookkeeping. The italic-paragraph fallback
    looks the part in Word / LibreOffice and keeps the writer pure.
    """
    if not text:
        return [r"\i (no transcript text available)\i0\par "]
    out: list[str] = []
    for ln in text.splitlines() or [""]:
        out.append(r"\i " + _rtf_escape(ln) + r"\i0\par ")
    return out


def _rtf_row_block(r: RetrievalRow, *, group_by: str) -> list[str]:
    out: list[str] = []
    bits: list[str] = [f"id: {r.application_id}"]
    if group_by != GROUP_BY_CODE and r.code_name:
        bits.append(f"code: {r.code_name}")
    if group_by != GROUP_BY_SOURCE and r.source_name:
        bits.append(f"source: {r.source_name}")
    if r.coder_name:
        bits.append(f"coder: {r.coder_name}")
    if (
        group_by != GROUP_BY_PARTICIPANT
        and r.participant_names
    ):
        bits.append(
            "participant: " + ", ".join(r.participant_names)
        )
    if r.confidence is not None:
        bits.append(f"confidence: {_format_confidence(r.confidence)}")
    if r.provenance_source:
        bits.append(f"provenance: {r.provenance_source}")
    if r.created_at:
        bits.append(f"at {r.created_at}")
    out.append(_rtf_para(" · ".join(bits)))
    out.extend(_rtf_blockquote(r.text))
    if r.note:
        out.append(_rtf_para_bold("Note"))
        out.append(_rtf_para(r.note))
    out.append(r"\par ")  # spacer between rows
    return out


def to_rtf(
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None = None,
    group_by: str = GROUP_BY_CODE,
) -> str:
    """Serialise retrieval rows to a minimal RTF 1.x document.

    Word / LibreOffice / Pages all open RTF natively. Output is
    ASCII-encoded with Unicode characters escaped per the RTF
    ``\\uNNNN?`` rule, matching :mod:`scribe.codebook_export.to_rtf`.
    """
    g = normalise_group_by(group_by)
    parts: list[str] = []
    parts.append(r"{\rtf1\ansi\ansicpg1252\deff0")
    parts.append(r"{\fonttbl{\f0\fnil Calibri;}}")
    parts.append(r"\fs22")  # 11pt body

    title = "Coded segments"
    if project is not None and project.name.strip():
        title = f"Coded segments — {project.name}"
    parts.append(_rtf_para_bold(title, fs=36))

    if project is not None:
        meta: list[str] = []
        if project.methodology:
            meta.append(f"Methodology: {project.methodology}")
        if project.codebook_stage:
            meta.append(f"Stage: {project.codebook_stage}")
        meta.append(f"Rows: {len(rows)}")
        if g != GROUP_BY_NONE:
            meta.append(f"Grouped by: {g}")
        for ml in meta:
            parts.append(_rtf_para(ml))
        parts.append(r"\par ")

    if not rows:
        parts.append(_rtf_para("(no coded segments)"))
        parts.append("}")
        return "".join(parts)

    groups = group_rows(rows, group_by=g)
    for grp in groups:
        if g != GROUP_BY_NONE:
            parts.append(_rtf_para_bold(grp.label, fs=28))
            sub = (
                f"{grp.key} · {len(grp.rows)} segment(s)"
                if grp.key
                else f"{len(grp.rows)} segment(s)"
            )
            parts.append(_rtf_para(sub))
        for r in grp.rows:
            parts.extend(_rtf_row_block(r, group_by=g))

    parts.append("}")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Format registry + dispatch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FormatSpec:
    """Static description of a user-facing retrieval-report format.

    Mirrors :class:`scribe.codebook_export.FormatSpec` so the HTTP and
    CLI surface look the same regardless of which body they're
    fetching.
    """

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
    "md": EXPORT_FORMAT_MARKDOWN,
    "markdown": EXPORT_FORMAT_MARKDOWN,
    "csv": EXPORT_FORMAT_CSV,
    "rtf": EXPORT_FORMAT_RTF,
    "word": EXPORT_FORMAT_RTF,
    "doc": EXPORT_FORMAT_RTF,
    "docx": EXPORT_FORMAT_RTF,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Trims + lower-cases. Accepts the same alias set as
    :func:`scribe.codebook_export.normalise_format` so a user who
    learns one CLI flag set carries it across.
    """
    if format is None:
        raise ValueError(
            "Retrieval-report format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported retrieval-report format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


_RENDERERS: dict[str, Callable[..., str]] = {}


def render_report(
    format: str,
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None = None,
    group_by: str = GROUP_BY_CODE,
) -> str:
    """Render rows in ``format``; dispatches to the right renderer.

    ``project`` is forwarded to Markdown / RTF only — CSV's column
    contract is the public schema and intentionally excludes a
    project header.
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](rows, project=project, group_by=group_by)


def _render_csv(
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None,
    group_by: str,
) -> str:
    # CSV ignores the project header and is always flat.
    del project, group_by
    return to_csv(rows)


def _render_markdown(
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None,
    group_by: str,
) -> str:
    return to_markdown(rows, project=project, group_by=group_by)


def _render_rtf(
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None,
    group_by: str,
) -> str:
    return to_rtf(rows, project=project, group_by=group_by)


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_MARKDOWN] = _render_markdown
_RENDERERS[EXPORT_FORMAT_RTF] = _render_rtf


# --------------------------------------------------------------------------- #
# Filename / disk-write helpers
# --------------------------------------------------------------------------- #


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def slugify_report_filename(
    project: Project | None, format: str
) -> str:
    """Build a download-friendly filename for a retrieval-report export.

    Pattern: ``<project-slug>-coded-segments<ext>`` if a project name
    is available; ``coded-segments<ext>`` otherwise. ASCII-only,
    lowercased, dash-separated, capped at :data:`_FILENAME_SLUG_MAX`
    characters before the suffix.
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
        return f"{slug}-coded-segments{spec.extension}"
    return f"coded-segments{spec.extension}"


def write_report(
    path: Path,
    format: str,
    rows: Sequence[RetrievalRow],
    *,
    project: Project | None = None,
    group_by: str = GROUP_BY_CODE,
) -> Path:
    """Render the retrieval report to ``format`` and atomically write to ``path``.

    Atomic via a ``.tmp`` swap so an interrupted write never leaves a
    half-finished export visible. Creates ``path.parent`` if missing.
    Returns the resolved target path.
    """
    fmt = normalise_format(format)
    text = render_report(fmt, rows, project=project, group_by=group_by)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(target)
    return target


__all__ = [
    "CSV_COLUMNS",
    "CSV_LIST_SEP",
    "EXPORT_FORMATS",
    "EXPORT_FORMAT_CSV",
    "EXPORT_FORMAT_MARKDOWN",
    "EXPORT_FORMAT_RTF",
    "FormatSpec",
    "GROUP_BY_CODE",
    "GROUP_BY_KEYS",
    "GROUP_BY_NONE",
    "GROUP_BY_PARTICIPANT",
    "GROUP_BY_SOURCE",
    "LABEL_NO_PARTICIPANT",
    "LABEL_UNKNOWN",
    "RetrievalGroup",
    "RetrievalRow",
    "build_retrieval_rows",
    "filter_rows",
    "group_rows",
    "normalise_format",
    "normalise_group_by",
    "render_report",
    "slugify_report_filename",
    "to_csv",
    "to_markdown",
    "to_rtf",
    "write_report",
]
