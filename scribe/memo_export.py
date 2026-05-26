"""Memo export (F5.4).

Per PLANNING.md F5.4:

  > "Export all memos" filtered by type / linked-to.

Memos are the connective tissue of grounded-theory analysis (see the
F5.1 module for the full rationale). When a researcher writes a methods
chapter, hands a project to a supervisor, or moves to another QDA
tool, the memo corpus needs to leave Scribe in a portable shape. F5.4
is the bundle of pure exporters that produce those handoff artefacts,
together with an in-memory ``filter_memos`` that mirrors the
:func:`scribe.memos.list_memos` filter semantics so callers can apply
filters once and emit several formats from the same set.

This module is **pure**: every function takes a sequence of
:class:`scribe.memos.Memo` (plus optional metadata for label
resolution) and returns a string. No filesystem I/O, no FastAPI, no
engine imports — same shape as :mod:`scribe.codebook_export`. Callers
(CLI, server, tests) decide where the bytes go.

Four formats, four functions
----------------------------

* :func:`to_csv` — one row per memo. Columns cover the canonical F5.1
  field set with multi-valued cells (links, tags) joined by " | ".
  RFC-4180 CRLF line endings; ``QUOTE_MINIMAL`` escaping.

* :func:`to_markdown` — heading-per-memo Markdown document with
  optional project header, per-memo metadata line, body block, link
  list, tag list, and provenance. The format that pastes into a
  thesis appendix or methods chapter.

* :func:`to_rtf` — minimal RTF 1.x document. Word, LibreOffice, and
  Pages all open RTF natively; zero new dependencies. Same Unicode
  escaping rules as :func:`scribe.codebook_export.to_rtf`.

* :func:`to_jsonl` — one JSON object per line (newline-delimited).
  Loss-less round-trip of :meth:`Memo.to_dict`; the format you reach
  for when a downstream script wants to ingest the memos
  programmatically without parsing CSV.

Filter
------

:func:`filter_memos` is the in-memory companion to
:func:`scribe.memos.list_memos`. It applies the same rules — type,
target_type / target_id, author_coder_id, tag — but on an existing
list rather than the disk. The server can list once, filter several
times, and emit several formats from the same memo set. Filter rules
are documented inline; in particular ``target_type`` alone matches any
memo with a link of that type, ``target_id`` alone matches any memo
linking to that id, and the two together require both to match on the
*same* link.

Target name resolution
----------------------

Memo links carry a ``(target_type, target_id)`` pair. The id alone is
machine-readable but not researcher-readable, so the Markdown / RTF
exporters accept an optional ``target_names`` dict
``(target_type, target_id) → human label``. The CSV exporter accepts
the same dict and emits a denormalised ``links_named`` column. Builds
of that dict typically come from a server endpoint that has the
project's :class:`Code` / :class:`Source` / :class:`Participant` /
:class:`Coder` / :class:`Application` / sibling-Memo lists at hand;
the helper :func:`build_target_names` in this module is the canonical
one-stop builder.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .memos import (
    MEMO_LINK_TARGET_TYPES,
    MEMO_TYPES,
    Memo,
    MemoLink,
    TARGET_ID_RE,
)
from .projects import Project, ProjectValidationError


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #

# CSV column order — part of the public contract. Adding new columns
# goes at the end so existing consumer scripts keep working.
CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "type",
    "title",
    "body",
    "body_format",
    "author_coder_id",
    "links",
    "links_named",
    "tags",
    "provenance_source",
    "created_at",
    "modified_at",
)

# Multi-valued cell separator. Matches the codebook export's ``" | "``
# convention so a researcher who's used to one format isn't surprised
# by the other.
CSV_LIST_SEP = " | "

# Per-link CSV cell format. ``role`` is folded in only when present so
# the common case stays compact.
def _format_link_for_csv(link: MemoLink) -> str:
    if link.role:
        return f"{link.target_type}:{link.target_id}:{link.role}"
    return f"{link.target_type}:{link.target_id}"


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #


def filter_memos(
    memos: Iterable[Memo],
    *,
    type: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    author_coder_id: str | None = None,
    tag: str | None = None,
) -> list[Memo]:
    """Filter an in-memory memo list, mirroring :func:`scribe.memos.list_memos`.

    Filter combinators are AND. The ``target_type`` / ``target_id``
    pair is special-cased: passed together, both must match on the
    *same* link; passed alone, the memo must carry at least one link
    matching the supplied half. Result is sorted by
    ``(created_at, id)`` ascending — same order as
    :func:`list_memos`, important because grounded-theory memo
    development is timeline-driven (early memos *should* read first).

    Validation matches :func:`list_memos`: a bad filter raises
    :class:`scribe.projects.ProjectValidationError` rather than
    silently returning [].
    """
    if type is not None and type not in MEMO_TYPES:
        raise ProjectValidationError(
            f"Invalid type filter: {type!r}; must be one of {MEMO_TYPES}"
        )
    if target_type is not None and target_type not in MEMO_LINK_TARGET_TYPES:
        raise ProjectValidationError(
            f"Invalid target_type filter: {target_type!r}; "
            f"must be one of {MEMO_LINK_TARGET_TYPES}"
        )
    if target_id is not None and not TARGET_ID_RE.match(target_id):
        raise ProjectValidationError(
            f"Invalid target_id filter: {target_id!r}"
        )
    if (
        author_coder_id is not None
        and not TARGET_ID_RE.match(author_coder_id)
    ):
        raise ProjectValidationError(
            f"Invalid author_coder_id filter: {author_coder_id!r}"
        )
    if tag is not None:
        if not isinstance(tag, str) or not tag.strip():
            raise ProjectValidationError(
                "tag filter must be a non-empty string"
            )

    out: list[Memo] = []
    for m in memos:
        if type is not None and m.type != type:
            continue
        if target_type is not None or target_id is not None:
            matches = False
            for link in m.links:
                if (
                    target_type is not None
                    and link.target_type != target_type
                ):
                    continue
                if (
                    target_id is not None
                    and link.target_id != target_id
                ):
                    continue
                matches = True
                break
            if not matches:
                continue
        if (
            author_coder_id is not None
            and m.author_coder_id != author_coder_id
        ):
            continue
        if tag is not None and tag not in m.tags:
            continue
        out.append(m)
    out.sort(key=lambda x: (x.created_at, x.id))
    return out


# --------------------------------------------------------------------------- #
# Helpers shared across formats
# --------------------------------------------------------------------------- #


TargetNameMap = Mapping[tuple[str, str], str]


def _link_label(
    link: MemoLink,
    target_names: TargetNameMap | None,
) -> str:
    """Render a single link as a human-friendly label.

    Format:

      * with name + role:  ``"<role>: <name> (<target_type>:<target_id>)"``
      * with name no role: ``"<name> (<target_type>:<target_id>)"``
      * no name with role: ``"<role>: <target_type>:<target_id>"``
      * no name no role:   ``"<target_type>:<target_id>"``

    Used by Markdown and RTF; CSV uses the compact form.
    """
    name = ""
    if target_names is not None:
        name = target_names.get((link.target_type, link.target_id), "") or ""
    base = f"{link.target_type}:{link.target_id}"
    if name and link.role:
        return f"{link.role}: {name} ({base})"
    if name:
        return f"{name} ({base})"
    if link.role:
        return f"{link.role}: {base}"
    return base


def build_target_names(
    *,
    codes: Iterable[object] | None = None,
    sources: Iterable[object] | None = None,
    participants: Iterable[object] | None = None,
    coders: Iterable[object] | None = None,
    applications: Iterable[object] | None = None,
    memos: Iterable[Memo] | None = None,
    project: Project | None = None,
) -> dict[tuple[str, str], str]:
    """Build a ``(target_type, target_id) → name`` map.

    Every argument is a duck-typed iterable: each item is expected to
    have an ``id`` attribute and a sensible ``name`` / ``title`` /
    ``code_id`` attribute. We tolerate missing attributes so the
    helper survives partial inputs (a project that has only some of
    its sibling lists hydrated).

    The point of this is to keep the exporters dependency-free of the
    other entity modules — :mod:`scribe.memo_export` should not
    import :mod:`scribe.codes`, :mod:`scribe.sources`, etc., because
    that pulls a lot of unrelated code into a 'pure exporter' module.
    The duck-typing convention matches how
    :class:`scribe.codebook_export` keeps itself decoupled.
    """
    out: dict[tuple[str, str], str] = {}

    def _record(
        target_type: str,
        items: Iterable[object] | None,
        attr_label: str,
    ) -> None:
        if not items:
            return
        for item in items:
            tid = getattr(item, "id", None)
            if not tid or not isinstance(tid, str):
                continue
            label = getattr(item, attr_label, "") or ""
            label = str(label)
            if label:
                out[(target_type, tid)] = label

    _record("code", codes, "name")
    _record("source", sources, "name")
    _record("participant", participants, "name")
    _record("coder", coders, "name")
    # Applications don't carry a human name — they're spans inside a
    # source. Use a fallback "<code_id>@<source_id>" so the link list
    # is at least disambiguable on the page.
    if applications is not None:
        for a in applications:
            aid = getattr(a, "id", None)
            if not aid or not isinstance(aid, str):
                continue
            cid = getattr(a, "code_id", "") or ""
            sid = getattr(a, "source_id", "") or ""
            if cid or sid:
                out[("application", aid)] = f"{cid}@{sid}"
    if memos is not None:
        for m in memos:
            mid = getattr(m, "id", None)
            if not mid or not isinstance(mid, str):
                continue
            title = getattr(m, "title", "") or ""
            if title:
                out[("memo", mid)] = str(title)
    if project is not None and getattr(project, "id", None):
        name = getattr(project, "name", "") or ""
        if name:
            out[("project", project.id)] = str(name)
    return out


def _provenance_source(memo: Memo) -> str:
    """Convenience for the CSV ``provenance_source`` column."""
    return memo.provenance.get("source", "")


def _heading_for(memo: Memo) -> str:
    """Pick the human-readable heading for a memo.

    Title wins; otherwise the first non-empty line of the body, with
    leading Markdown ``#`` chars stripped so a memo whose body starts
    ``# A theme`` doesn't render as a sub-heading. Long lines are
    truncated to 80 chars with an ellipsis. Last-resort fallback is
    the memo's id, which guarantees a non-empty heading.
    """
    if memo.title.strip():
        return memo.title.strip()
    for line in memo.body.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            if len(s) > 80:
                s = s[:77].rstrip() + "…"
            return s
    return f"(untitled memo {memo.id})"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def to_csv(
    memos: Sequence[Memo],
    *,
    target_names: TargetNameMap | None = None,
) -> str:
    """Serialise a memo list to CSV.

    Columns are :data:`CSV_COLUMNS`. Multi-valued cells (``links``,
    ``links_named``, ``tags``) are joined with :data:`CSV_LIST_SEP`.

    ``links`` carries the compact ``<target_type>:<target_id>[:<role>]``
    form; ``links_named`` carries the human-readable label produced
    via ``target_names`` (when supplied) — same content, denormalised
    for spreadsheet use. ``links_named`` is left blank when no name
    map is supplied.

    Empty input yields a header-only CSV — that's a valid empty memo
    export, not an error. Output uses ``\\r\\n`` line endings per
    RFC 4180 (the :mod:`csv` module's default).
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for m in memos:
        compact_links = CSV_LIST_SEP.join(
            _format_link_for_csv(link) for link in m.links
        )
        named_links = ""
        if target_names is not None:
            named_links = CSV_LIST_SEP.join(
                _link_label(link, target_names) for link in m.links
            )
        writer.writerow(
            [
                m.id,
                m.type,
                m.title,
                m.body,
                m.body_format,
                m.author_coder_id or "",
                compact_links,
                named_links,
                CSV_LIST_SEP.join(m.tags),
                _provenance_source(m),
                m.created_at,
                m.modified_at,
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def to_markdown(
    memos: Sequence[Memo],
    *,
    project: Project | None = None,
    target_names: TargetNameMap | None = None,
    title: str = "Memos",
    filter_summary: str = "",
) -> str:
    """Serialise a memo list to a structured Markdown document.

    Layout:

    1. ``# <title>`` heading. With a ``project``, the project name is
       appended (``# Memos — My Study``).
    2. Optional project metadata line (methodology, stage).
    3. Optional ``filter_summary`` line — the caller can pass the
       human description of which filters were applied so the export
       header explains itself.
    4. ``## <heading>`` per memo, each with:

       * inline metadata line (id, type, body_format, author, created),
       * body (the memo's prose; rendered as-is, since markdown bodies
         already are markdown — non-markdown bodies render acceptably
         too because the reader treats them as plain paragraphs),
       * **Links** as a bullet list,
       * **Tags** as an inline comma-joined line,
       * **Provenance** as an inline ``key: value;`` line.

    Sections that would be empty are omitted entirely.

    Output ends with a trailing newline so concatenation in shell
    pipelines composes cleanly.
    """
    lines: list[str] = []
    head = title
    if project is not None and project.name.strip():
        head = f"{title} — {project.name}"
    lines.append(f"# {head}")
    lines.append("")

    if project is not None:
        meta_rows: list[tuple[str, str]] = []
        if project.methodology:
            meta_rows.append(("Methodology", project.methodology))
        if project.codebook_stage:
            meta_rows.append(("Stage", project.codebook_stage))
        meta_rows.append(("Memos", str(len(memos))))
        for label, value in meta_rows:
            lines.append(f"- **{label}**: {value}")
        lines.append("")
    else:
        lines.append(f"- **Memos**: {len(memos)}")
        lines.append("")

    if filter_summary.strip():
        lines.append(f"_Filter: {filter_summary.strip()}_")
        lines.append("")

    if not memos:
        lines.append("_(no memos)_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    for m in memos:
        lines.extend(_markdown_memo_block(m, target_names))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _markdown_memo_block(
    m: Memo, target_names: TargetNameMap | None
) -> list[str]:
    """Render one memo as a Markdown section. Pure helper for tests."""
    out: list[str] = []
    out.append(f"## {_heading_for(m)}")
    out.append("")

    inline_bits: list[str] = [f"`{m.id}`", f"type: {m.type}"]
    if m.body_format and m.body_format != "markdown":
        inline_bits.append(f"format: {m.body_format}")
    if m.author_coder_id:
        inline_bits.append(f"author: `{m.author_coder_id}`")
    if m.created_at:
        inline_bits.append(f"created: {m.created_at}")
    out.append(" · ".join(inline_bits))
    out.append("")

    if m.body.strip():
        out.append(m.body.rstrip())
        out.append("")

    if m.links:
        out.append("**Links**")
        out.append("")
        for link in m.links:
            out.append(f"- {_link_label(link, target_names)}")
        out.append("")

    if m.tags:
        out.append("**Tags**: " + ", ".join(m.tags))
        out.append("")

    if m.provenance:
        prov = "; ".join(f"{k}: {v}" for k, v in m.provenance.items())
        out.append(f"**Provenance**: {prov}")
        out.append("")

    while out and out[-1] == "":
        out.pop()
    return out


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


# RTF escape, lifted verbatim from scribe.codebook_export so the two
# exporters render identical Unicode handling. We don't import the
# helper across modules because both are private and codebook_export
# is its own contract; copying keeps each module self-contained.
_RTF_SPECIAL_RE = re.compile(r"[\\{}]")


def _rtf_escape(text: str) -> str:
    """Escape a Python string for RTF body text.

    * ``\\`` ``{`` ``}`` → escaped with a leading backslash.
    * Any non-ASCII char → ``\\uNNNN?`` where NNNN is signed 16-bit;
      astral-plane code points are emitted as a surrogate pair.
    * Newline → ``\\par``; CR is dropped; tab → ``\\tab``.
    """
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "\n":
            out.append("\\par\n")
        elif ch == "\r":
            continue
        elif ch == "\t":
            out.append("\\tab ")
        elif 0x20 <= ord(ch) <= 0x7E:
            out.append(ch)
        else:
            cp = ord(ch)
            if cp <= 0xFFFF:
                signed = cp if cp < 0x8000 else cp - 0x10000
                out.append(f"\\u{signed}?")
            else:
                cp -= 0x10000
                hi = 0xD800 + (cp >> 10)
                lo = 0xDC00 + (cp & 0x3FF)
                hi_signed = hi if hi < 0x8000 else hi - 0x10000
                lo_signed = lo if lo < 0x8000 else lo - 0x10000
                out.append(f"\\u{hi_signed}?\\u{lo_signed}?")
    return "".join(out)


def _rtf_para(text: str, *, fs: int | None = None) -> str:
    prefix = ""
    if fs is not None:
        prefix = rf"\fs{fs} "
    return prefix + _rtf_escape(text) + r"\par "


def _rtf_para_bold(text: str, *, fs: int | None = None) -> str:
    prefix = r"\b "
    if fs is not None:
        prefix = rf"\b\fs{fs} "
    return prefix + _rtf_escape(text) + r"\b0\par "


def to_rtf(
    memos: Sequence[Memo],
    *,
    project: Project | None = None,
    target_names: TargetNameMap | None = None,
    title: str = "Memos",
    filter_summary: str = "",
) -> str:
    """Serialise a memo list to a minimal RTF 1.x document.

    Word, LibreOffice, and Pages all open RTF natively so this covers
    the "open in Word" handoff path without a python-docx dependency.
    Output is ASCII-encoded RTF with Unicode escaped per RTF 1.x
    ``\\uNNNN?``.
    """
    parts: list[str] = []
    parts.append(r"{\rtf1\ansi\ansicpg1252\deff0")
    parts.append(r"{\fonttbl{\f0\fnil Calibri;}}")
    parts.append(r"\fs22")  # base body font 11 pt

    head = title
    if project is not None and project.name.strip():
        head = f"{title} — {project.name}"
    parts.append(_rtf_para_bold(head, fs=36))

    if project is not None:
        meta_lines: list[str] = []
        if project.methodology:
            meta_lines.append(f"Methodology: {project.methodology}")
        if project.codebook_stage:
            meta_lines.append(f"Stage: {project.codebook_stage}")
        meta_lines.append(f"Memos: {len(memos)}")
        for ml in meta_lines:
            parts.append(_rtf_para(ml))
    else:
        parts.append(_rtf_para(f"Memos: {len(memos)}"))

    if filter_summary.strip():
        parts.append(_rtf_para(f"Filter: {filter_summary.strip()}"))

    parts.append(r"\par ")

    if not memos:
        parts.append(_rtf_para("(no memos)"))
        parts.append("}")
        return "".join(parts)

    for m in memos:
        parts.extend(_rtf_memo_block(m, target_names))

    parts.append("}")
    return "".join(parts)


def _rtf_memo_block(
    m: Memo, target_names: TargetNameMap | None
) -> list[str]:
    out: list[str] = []
    out.append(_rtf_para_bold(_heading_for(m), fs=28))

    meta_bits: list[str] = [f"id: {m.id}", f"type: {m.type}"]
    if m.body_format and m.body_format != "markdown":
        meta_bits.append(f"format: {m.body_format}")
    if m.author_coder_id:
        meta_bits.append(f"author: {m.author_coder_id}")
    if m.created_at:
        meta_bits.append(f"created: {m.created_at}")
    out.append(_rtf_para(" · ".join(meta_bits)))

    if m.body.strip():
        out.append(_rtf_para(m.body.rstrip()))

    if m.links:
        out.append(_rtf_para_bold("Links"))
        for link in m.links:
            out.append(_rtf_para(f"•\t{_link_label(link, target_names)}"))

    if m.tags:
        out.append(_rtf_label_block("Tags", ", ".join(m.tags)))

    if m.provenance:
        prov = "; ".join(f"{k}: {v}" for k, v in m.provenance.items())
        out.append(_rtf_label_block("Provenance", prov))

    out.append(r"\par ")
    return out


def _rtf_label_block(label: str, body: str) -> str:
    return _rtf_para_bold(label) + _rtf_para(body)


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


def to_jsonl(memos: Sequence[Memo]) -> str:
    """Serialise to newline-delimited JSON.

    One memo per line; each line is :meth:`Memo.to_dict`'s output run
    through ``json.dumps`` with ``ensure_ascii=False`` and stable key
    order. Empty input → empty string (not ``"\\n"``), so concatenation
    of two empty sets stays empty.

    The format is loss-less: a downstream script can reconstitute the
    exact memo dicts. Useful when the export is fed back into another
    Scribe project, into a notebook, or into ``jq`` for analysis.
    """
    if not memos:
        return ""
    return (
        "\n".join(
            json.dumps(m.to_dict(), ensure_ascii=False, sort_keys=True)
            for m in memos
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# F5.4 user-facing surface — format registry + filename slug
#
# The four pure exporters above each take ``Sequence[Memo]`` and return
# a string. F5.4 closes the loop by giving the HTTP endpoint, the CLI,
# and any future button in the UI a single common entry point — same
# pattern :mod:`scribe.codebook_export` uses for F6.1.
#
# What gets added here:
#
#  * :data:`EXPORT_FORMATS` — registry of {key -> FormatSpec(extension,
#    media_type, label)} for the user-facing formats. F5.4 covers all
#    four: CSV, Markdown, RTF (= "Word"), and JSONL (the format we
#    keep for round-trip / scripting use cases).
#
#  * :func:`normalise_format` — accepts the canonical keys plus
#    ergonomic aliases (``md`` for Markdown; ``word`` / ``doc`` /
#    ``docx`` for RTF; ``ndjson`` for JSONL). Raises :class:`ValueError`
#    on anything else, with the list of accepted keys, so the message
#    is actionable.
#
#  * :func:`render_memos` — single-dispatch helper that picks the
#    right exporter for ``format`` and returns the rendered string.
#    Forwards ``project`` / ``target_names`` / ``filter_summary`` only
#    to renderers that consume them (CSV ignores ``filter_summary``;
#    JSONL ignores everything except the memos themselves so the
#    line-shape stays the public contract).
#
#  * :func:`slugify_memos_filename` — derives an attachment filename
#    like ``my-project-memos.csv`` from a project. Same NFKD-normalise
#    + ASCII-downgrade rule as ``slugify_codebook_filename``.
#
#  * :func:`build_filter_summary` — compose the human description of
#    which filters were applied so the Markdown / RTF exporter header
#    can explain itself ("Filter: type=theoretical, target_type=code").
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FormatSpec:
    """Static description of a user-facing memo export format.

    * ``key`` — canonical lookup key (``csv`` / ``markdown`` / ``rtf``
      / ``jsonl``).
    * ``extension`` — file extension *with* leading dot (``.csv``).
    * ``media_type`` — IANA media type for HTTP ``Content-Type``;
      includes ``charset=utf-8`` for the text formats so browsers
      don't guess Latin-1.
    * ``label`` — human-readable name for UI buttons / log lines.
    """

    key: str
    extension: str
    media_type: str
    label: str


EXPORT_FORMAT_CSV = "csv"
EXPORT_FORMAT_MARKDOWN = "markdown"
EXPORT_FORMAT_RTF = "rtf"
EXPORT_FORMAT_JSONL = "jsonl"


# Registry of formats F5.4 surfaces. All four pure exporters above are
# represented; the dropdown in the memos page renders a row per entry.
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
        # Word and LibreOffice both open ``.rtf`` natively. We deliver
        # an ``application/rtf`` body — the historical ``text/rtf`` is
        # accepted but ``application/rtf`` is what Microsoft + the IETF
        # converged on (RFC 1521 plus subsequent practice).
        extension=".rtf",
        media_type="application/rtf",
        label="RTF (Word)",
    ),
    EXPORT_FORMAT_JSONL: FormatSpec(
        key=EXPORT_FORMAT_JSONL,
        extension=".jsonl",
        # ``application/x-ndjson`` is the de-facto type for newline-
        # delimited JSON. ``charset=utf-8`` because to_jsonl emits
        # ``ensure_ascii=False`` so the bytes can be non-ASCII.
        media_type="application/x-ndjson; charset=utf-8",
        label="JSONL",
    ),
}


# Aliases the user might type. Resolved before lookup in EXPORT_FORMATS.
_FORMAT_ALIASES: dict[str, str] = {
    "md": EXPORT_FORMAT_MARKDOWN,
    "markdown": EXPORT_FORMAT_MARKDOWN,
    "csv": EXPORT_FORMAT_CSV,
    "rtf": EXPORT_FORMAT_RTF,
    # "Word" routes to RTF — RTF is the format Word opens natively
    # without a ``.docx`` ZIP ceremony, and F5.4 ships an RTF exporter.
    "word": EXPORT_FORMAT_RTF,
    "doc": EXPORT_FORMAT_RTF,
    "docx": EXPORT_FORMAT_RTF,
    "jsonl": EXPORT_FORMAT_JSONL,
    "ndjson": EXPORT_FORMAT_JSONL,
    "json": EXPORT_FORMAT_JSONL,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Case-insensitive; trims whitespace; recognises a small handful of
    aliases (``md`` → ``markdown``; ``word`` / ``doc`` / ``docx`` →
    ``rtf``; ``ndjson`` / ``json`` → ``jsonl``). Raises :class:`ValueError`
    for unknown formats with the list of accepted keys, so the message
    is actionable.
    """
    if format is None:
        raise ValueError(
            "Memo export format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported memo export format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


# Map normalised format key → renderer. Populated below to avoid
# forward references.
_RENDERERS: dict[str, Callable[..., str]] = {}


def render_memos(
    format: str,
    memos: Sequence[Memo],
    *,
    project: Project | None = None,
    target_names: TargetNameMap | None = None,
    filter_summary: str = "",
) -> str:
    """Render a memo list in ``format`` and return the string body.

    Dispatches via :func:`normalise_format` so callers can pass the
    same alias set the HTTP query string accepts. ``project`` /
    ``target_names`` / ``filter_summary`` are forwarded only to the
    renderers that consume them; the JSONL renderer ignores everything
    except the memo list so the line-shape stays the public contract.

    Empty inputs are valid and produce a header-only CSV / a placeholder
    Markdown / a minimal RTF / an empty JSONL string. Never raises on
    empty input.
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](
        memos,
        project=project,
        target_names=target_names,
        filter_summary=filter_summary,
    )


def _render_csv(
    memos: Sequence[Memo],
    *,
    project: Project | None,
    target_names: TargetNameMap | None,
    filter_summary: str,
) -> str:
    # CSV ignores project + filter_summary — column shape is the public
    # contract and a row-1 header would break downstream importers.
    del project, filter_summary
    return to_csv(memos, target_names=target_names)


def _render_markdown(
    memos: Sequence[Memo],
    *,
    project: Project | None,
    target_names: TargetNameMap | None,
    filter_summary: str,
) -> str:
    return to_markdown(
        memos,
        project=project,
        target_names=target_names,
        filter_summary=filter_summary,
    )


def _render_rtf(
    memos: Sequence[Memo],
    *,
    project: Project | None,
    target_names: TargetNameMap | None,
    filter_summary: str,
) -> str:
    return to_rtf(
        memos,
        project=project,
        target_names=target_names,
        filter_summary=filter_summary,
    )


def _render_jsonl(
    memos: Sequence[Memo],
    *,
    project: Project | None,
    target_names: TargetNameMap | None,
    filter_summary: str,
) -> str:
    # JSONL is loss-less round-trip of Memo.to_dict; project header /
    # filter summary / target_names would all break the line shape.
    del project, target_names, filter_summary
    return to_jsonl(memos)


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_MARKDOWN] = _render_markdown
_RENDERERS[EXPORT_FORMAT_RTF] = _render_rtf
_RENDERERS[EXPORT_FORMAT_JSONL] = _render_jsonl


# Slug regex: collapse runs of non-alphanumeric ASCII into a single
# dash. We NFKD-normalise + strip combining marks first so "Élise"
# becomes "elise", not the empty string. Same approach as
# :func:`scribe.codebook_export.slugify_codebook_filename`.
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def slugify_memos_filename(
    project: Project | None, format: str
) -> str:
    """Build a download-friendly filename for a memos export.

    Pattern: ``<project-slug>-memos<ext>`` if a project name is
    available; ``memos<ext>`` otherwise. The slug is ASCII-only,
    lowercased, dash-separated, and capped at
    :data:`_FILENAME_SLUG_MAX` characters before the suffix.

    Raises :class:`ValueError` for unknown formats (delegates to
    :func:`normalise_format`).
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
        return f"{slug}-memos{spec.extension}"
    return f"memos{spec.extension}"


def build_filter_summary(
    *,
    type: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    author_coder_id: str | None = None,
    tag: str | None = None,
) -> str:
    """Build a human-readable description of the active filter.

    Returns the empty string when no filter is active, so the Markdown
    / RTF exporters can omit the filter line entirely. The format is
    ``key=value`` pairs joined by ``", "``, in the same order the
    function arguments are declared; this makes the summary stable for
    snapshot tests and the export header self-documenting.
    """
    parts: list[str] = []
    if type:
        parts.append(f"type={type}")
    if target_type:
        parts.append(f"target_type={target_type}")
    if target_id:
        parts.append(f"target_id={target_id}")
    if author_coder_id:
        parts.append(f"author_coder_id={author_coder_id}")
    if tag:
        parts.append(f"tag={tag}")
    return ", ".join(parts)
