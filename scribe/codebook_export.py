"""Codebook export (F2.6).

Per PLANNING.md F2.6:

  > Codebook export: CSV, structured Markdown, RTF/Word (with
  > definitions and exemplars), REFI-QDA Codebook XML.

A codebook is a project's full set of :class:`scribe.codes.Code`
entries (F2.1). At various points in a research project the codebook
needs to leave Scribe — pasted into a thesis appendix, shared with a
supervisor, mailed to a second coder, or imported into another QDA
tool. F2.6 is the bundle of exporters that produce those handoff
artefacts.

This module is **pure**: every function takes a list of Codes plus an
optional :class:`scribe.projects.Project` (for header metadata) and
returns text. No filesystem I/O, no FastAPI, no engine imports — same
shape as :mod:`scribe.codes`, :mod:`scribe.code_versions`, and the
other F1.* / F2.* modules. Callers (CLI, server, tests) decide where
the bytes go.

Four formats, four functions
----------------------------

* :func:`to_csv` — one row per code; columns cover the canonical F2.1
  field set. Tab-and-comma-safe (uses :mod:`csv` with QUOTE_MINIMAL),
  CRLF-terminated lines per RFC 4180. The format the supervisor opens
  in Excel.

* :func:`to_markdown` — heading-per-code Markdown document with
  definition, criteria, exemplars, related codes, and provenance. The
  format that gets pasted into a methods chapter or shared as a
  hand-off README.

* :func:`to_rtf` — minimal RTF 1.x document. RTF is a stable
  plain-text format Word, LibreOffice, and Pages all open natively;
  it side-steps the python-docx dependency and is what F2.6 means by
  "RTF/Word." Includes bolded headings, indented exemplar bullets,
  and inline coloured swatches for codes that carry a colour.

* :func:`to_refi_qda_xml` — REFI-QDA Codebook 1.0 XML
  (``urn:QDA-XML:codebook:1.0``). The "no lock-in" exchange format:
  any tool that imports REFI-QDA Codebook XML (Atlas.ti, MAXQDA,
  NVivo, QDA Miner, Quirkos…) will accept the output. Code hierarchy
  is encoded by nesting; non-hierarchy fields (inclusion / exclusion
  criteria, exemplars, theoretical memo) are folded into the
  ``<Description>`` body with labelled sections so no information is
  lost in the round-trip.

The internal-id → REFI-QDA-GUID mapping pads our 12-char hex code IDs
to the 8-4-4-4-12 GUID layout deterministically (zeros for the high
bits). That keeps the mapping bijective so a future REFI-QDA *import*
(F6.6) can recover the original Scribe id.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.etree import ElementTree as ET

from .codes import Code, CodeRelation
from .projects import Project


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #

# REFI-QDA Codebook 1.0 schema namespace + root.
REFI_QDA_NS = "urn:QDA-XML:codebook:1.0"
REFI_QDA_ORIGIN_DEFAULT = "Scribe"

# CSV columns. Order is part of the public contract — the supervisor
# expects "id, name, definition" not the alphabet. New fields go at the
# end so old consumer scripts don't drift.
CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "definition",
    "inclusion_criteria",
    "exclusion_criteria",
    "exemplars",
    "parent_code_id",
    "parent_name",
    "related_codes",
    "theoretical_memo",
    "stage",
    "colour",
    "status",
    "provenance_source",
    "created_at",
    "modified_at",
)

# When folding multi-valued fields into a single CSV cell we use a
# narrow separator that's vanishingly rare in research prose. " | " is
# also what the existing transcript exporter uses for speaker labels.
CSV_LIST_SEP = " | "

# Same for related-code formatting in the CSV cell:
#   ``<code_id>:<relation_type>``
# Compact, machine-parseable, and a stable round-trip with
# ``CodeRelation``.
CSV_RELATION_FMT = "{code_id}:{relation_type}"


# --------------------------------------------------------------------------- #
# Helpers shared across formats
# --------------------------------------------------------------------------- #


def _index_by_id(codes: Iterable[Code]) -> dict[str, Code]:
    """Index codes by ``id`` so lookups (parent name, related-code
    name) don't go quadratic on a 500-code codebook.
    """
    return {c.id: c for c in codes}


def _parent_name(code: Code, by_id: dict[str, Code]) -> str:
    """Return the parent code's name, or empty string if no/unknown parent.

    Unknown parents (id pointing at a code that's not in the supplied
    list) are tolerated — exporters operate on whatever they're given.
    """
    if not code.parent_code_id:
        return ""
    parent = by_id.get(code.parent_code_id)
    return parent.name if parent else ""


def _format_related_codes_csv(rels: Sequence[CodeRelation]) -> str:
    """Format a code's related-code list as a single CSV cell.

    Empty list → empty string (so CSV consumers see a blank cell, not
    the literal four characters ``"[]"`` or similar).
    """
    if not rels:
        return ""
    return CSV_LIST_SEP.join(
        CSV_RELATION_FMT.format(
            code_id=r.code_id, relation_type=r.relation_type
        )
        for r in rels
    )


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def to_csv(codes: Sequence[Code]) -> str:
    """Serialise a codebook to CSV.

    The exact column order is :data:`CSV_COLUMNS`. Lists (``exemplars``,
    ``related_codes``) are joined with :data:`CSV_LIST_SEP`. The
    ``parent_name`` column is a denormalised convenience (parent ids
    alone aren't human-readable in Excel); it's blank for top-level
    codes.

    Empty input yields a header-only CSV — that's a valid empty
    codebook export, not an error.

    Output uses ``\\r\\n`` line endings per RFC 4180 (the :mod:`csv`
    module's default), so the file can be sent to a Windows recipient
    without re-coding.
    """
    by_id = _index_by_id(codes)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for c in codes:
        writer.writerow(
            [
                c.id,
                c.name,
                c.definition,
                c.inclusion_criteria,
                c.exclusion_criteria,
                CSV_LIST_SEP.join(c.exemplars),
                c.parent_code_id or "",
                _parent_name(c, by_id),
                _format_related_codes_csv(c.related_codes),
                c.theoretical_memo,
                c.stage,
                c.colour,
                c.status,
                c.provenance.get("source", ""),
                c.created_at,
                c.modified_at,
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def to_markdown(
    codes: Sequence[Code], *, project: Project | None = None
) -> str:
    """Serialise a codebook to a structured Markdown document.

    Layout:

    1. ``# Codebook`` heading (optionally including the project name).
    2. Project metadata table (methodology, stage, code count) if a
       ``project`` is supplied.
    3. ``## <name>`` per code, each with:
       * inline metadata line (id, stage, status, colour, parent),
       * **Definition** block,
       * **Inclusion criteria** / **Exclusion criteria** blocks if
         present,
       * **Exemplars** as a bullet list,
       * **Related codes** as a bullet list with relation type,
       * **Theoretical memo** as a final block.

    Sections that would be empty are omitted entirely — a codebook
    where every code has only a name produces a clean document, not a
    forest of "_(none)_".

    The output is plain CommonMark, no extensions; pastes cleanly into
    a thesis appendix or a Notion page.
    """
    by_id = _index_by_id(codes)

    lines: list[str] = []
    title = "Codebook"
    if project is not None and project.name.strip():
        title = f"Codebook — {project.name}"
    lines.append(f"# {title}")
    lines.append("")

    if project is not None:
        meta_rows = []
        if project.methodology:
            meta_rows.append(("Methodology", project.methodology))
        if project.codebook_stage:
            meta_rows.append(("Stage", project.codebook_stage))
        meta_rows.append(("Codes", str(len(codes))))
        if meta_rows:
            for label, value in meta_rows:
                lines.append(f"- **{label}**: {value}")
            lines.append("")

    if not codes:
        lines.append("_(empty codebook)_")
        lines.append("")
        return "\n".join(lines)

    for c in codes:
        lines.extend(_markdown_code_block(c, by_id))
        lines.append("")  # blank line between codes

    return "\n".join(lines).rstrip() + "\n"


def _markdown_code_block(c: Code, by_id: dict[str, Code]) -> list[str]:
    """Render one code as a Markdown section. Pure helper for tests."""
    out: list[str] = []
    out.append(f"## {c.name}")
    out.append("")

    inline_bits: list[str] = [f"`{c.id}`"]
    if c.stage:
        inline_bits.append(f"stage: {c.stage}")
    if c.status:
        inline_bits.append(f"status: {c.status}")
    if c.colour:
        inline_bits.append(f"colour: {c.colour}")
    if c.parent_code_id:
        pname = _parent_name(c, by_id)
        parent_label = (
            f"{pname} (`{c.parent_code_id}`)"
            if pname
            else f"`{c.parent_code_id}`"
        )
        inline_bits.append(f"parent: {parent_label}")
    out.append(" · ".join(inline_bits))
    out.append("")

    if c.definition:
        out.append("**Definition**")
        out.append("")
        out.append(c.definition)
        out.append("")

    if c.inclusion_criteria:
        out.append("**Inclusion criteria**")
        out.append("")
        out.append(c.inclusion_criteria)
        out.append("")

    if c.exclusion_criteria:
        out.append("**Exclusion criteria**")
        out.append("")
        out.append(c.exclusion_criteria)
        out.append("")

    if c.exemplars:
        out.append("**Exemplars**")
        out.append("")
        for ex in c.exemplars:
            out.append(f"- {ex}")
        out.append("")

    if c.related_codes:
        out.append("**Related codes**")
        out.append("")
        for r in c.related_codes:
            target = by_id.get(r.code_id)
            label = (
                f"{target.name} (`{r.code_id}`)"
                if target
                else f"`{r.code_id}`"
            )
            out.append(f"- _{r.relation_type}_: {label}")
        out.append("")

    if c.theoretical_memo:
        out.append("**Theoretical memo**")
        out.append("")
        out.append(c.theoretical_memo)
        out.append("")

    if c.provenance:
        prov_bits = [f"{k}: {v}" for k, v in c.provenance.items()]
        out.append("**Provenance**")
        out.append("")
        out.append("; ".join(prov_bits))
        out.append("")

    # Strip the trailing blank line — caller adds the inter-code spacer.
    while out and out[-1] == "":
        out.pop()
    return out


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


# RTF escape: backslash, braces, and any non-ASCII char must be
# encoded as \\u<dec>?. Anything 0x20..0x7E (printable ASCII) other
# than the three special chars passes through unchanged.
_RTF_SPECIAL_RE = re.compile(r"[\\{}]")


def _rtf_escape(text: str) -> str:
    """Escape a Python string for RTF body text.

    * ``\\`` ``{`` ``}`` → escaped with a leading backslash.
    * Any non-ASCII character → ``\\uNNNN?`` where NNNN is the signed
      16-bit code point and ``?`` is a placeholder for readers that
      can't render it. Code points above 0xFFFF are encoded as a
      surrogate pair (RTF readers pick the closer encoding).
    * Newlines → RTF paragraph break ``\\par``.
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
            # Drop CR; RTF readers prefer \par paragraph breaks.
            continue
        elif ch == "\t":
            out.append("\\tab ")
        elif 0x20 <= ord(ch) <= 0x7E:
            out.append(ch)
        else:
            cp = ord(ch)
            if cp <= 0xFFFF:
                # Signed 16-bit per RTF spec.
                signed = cp if cp < 0x8000 else cp - 0x10000
                out.append(f"\\u{signed}?")
            else:
                # Astral plane → encode as surrogate pair.
                cp -= 0x10000
                hi = 0xD800 + (cp >> 10)
                lo = 0xDC00 + (cp & 0x3FF)
                hi_signed = hi if hi < 0x8000 else hi - 0x10000
                lo_signed = lo if lo < 0x8000 else lo - 0x10000
                out.append(f"\\u{hi_signed}?\\u{lo_signed}?")
    return "".join(out)


def to_rtf(
    codes: Sequence[Code], *, project: Project | None = None
) -> str:
    """Serialise a codebook to a minimal RTF 1.x document.

    Word, LibreOffice, and Pages all open RTF natively, so this
    satisfies F2.6's "RTF/Word" requirement without a heavy
    docx-generating dependency. The output is ASCII-encoded RTF with
    Unicode characters escaped per the RTF 1.x ``\\uNNNN?`` rule, so
    it transports cleanly through email and version control.

    Headings use bold + larger font sizes (RTF ``\\fs`` is half-points,
    so ``\\fs36`` = 18 pt). The single font in the table is "Calibri"
    — Word and LibreOffice both substitute a sensible default if the
    user doesn't have it.
    """
    by_id = _index_by_id(codes)

    parts: list[str] = []
    # RTF preamble: version, charset, font table.
    parts.append(r"{\rtf1\ansi\ansicpg1252\deff0")
    parts.append(r"{\fonttbl{\f0\fnil Calibri;}}")
    parts.append(r"\fs22")  # base body font 11pt

    title = "Codebook"
    if project is not None and project.name.strip():
        title = f"Codebook — {project.name}"
    parts.append(_rtf_para_bold(title, fs=36))

    if project is not None:
        meta_lines: list[str] = []
        if project.methodology:
            meta_lines.append(f"Methodology: {project.methodology}")
        if project.codebook_stage:
            meta_lines.append(f"Stage: {project.codebook_stage}")
        meta_lines.append(f"Codes: {len(codes)}")
        for ml in meta_lines:
            parts.append(_rtf_para(ml))
        parts.append(r"\par")

    if not codes:
        parts.append(_rtf_para("(empty codebook)"))
        parts.append("}")
        return "".join(parts)

    for c in codes:
        parts.extend(_rtf_code_block(c, by_id))

    parts.append("}")
    return "".join(parts)


def _rtf_para(text: str, *, fs: int | None = None) -> str:
    """Wrap ``text`` in a single RTF paragraph."""
    prefix = ""
    if fs is not None:
        prefix = rf"\fs{fs} "
    return prefix + _rtf_escape(text) + r"\par "


def _rtf_para_bold(text: str, *, fs: int | None = None) -> str:
    """Bold paragraph helper used for headings."""
    prefix = r"\b "
    if fs is not None:
        prefix = rf"\b\fs{fs} "
    return prefix + _rtf_escape(text) + r"\b0\par "


def _rtf_label_block(label: str, body: str) -> str:
    """Bold label paragraph followed by body paragraph(s)."""
    return _rtf_para_bold(label) + _rtf_para(body)


def _rtf_code_block(c: Code, by_id: dict[str, Code]) -> list[str]:
    """Render one code as RTF. Pure helper for tests."""
    out: list[str] = []
    out.append(_rtf_para_bold(c.name, fs=28))

    meta_bits: list[str] = [f"id: {c.id}"]
    if c.stage:
        meta_bits.append(f"stage: {c.stage}")
    if c.status:
        meta_bits.append(f"status: {c.status}")
    if c.colour:
        meta_bits.append(f"colour: {c.colour}")
    if c.parent_code_id:
        parent_name = _parent_name(c, by_id)
        plabel = (
            f"{parent_name} ({c.parent_code_id})"
            if parent_name
            else c.parent_code_id
        )
        meta_bits.append(f"parent: {plabel}")
    out.append(_rtf_para(" · ".join(meta_bits)))

    if c.definition:
        out.append(_rtf_label_block("Definition", c.definition))
    if c.inclusion_criteria:
        out.append(_rtf_label_block("Inclusion criteria", c.inclusion_criteria))
    if c.exclusion_criteria:
        out.append(_rtf_label_block("Exclusion criteria", c.exclusion_criteria))

    if c.exemplars:
        out.append(_rtf_para_bold("Exemplars"))
        for ex in c.exemplars:
            # RTF "bullet" via U+2022 followed by tab.
            out.append(_rtf_para(f"•\t{ex}"))

    if c.related_codes:
        out.append(_rtf_para_bold("Related codes"))
        for r in c.related_codes:
            target = by_id.get(r.code_id)
            label = (
                f"{target.name} ({r.code_id})"
                if target
                else r.code_id
            )
            out.append(_rtf_para(f"•\t{r.relation_type}: {label}"))

    if c.theoretical_memo:
        out.append(_rtf_label_block("Theoretical memo", c.theoretical_memo))

    if c.provenance:
        prov_bits = "; ".join(f"{k}: {v}" for k, v in c.provenance.items())
        out.append(_rtf_label_block("Provenance", prov_bits))

    # Spacer between codes.
    out.append(r"\par ")
    return out


# --------------------------------------------------------------------------- #
# REFI-QDA Codebook XML
# --------------------------------------------------------------------------- #


def code_id_to_refi_guid(code_id: str) -> str:
    """Map a Scribe 12-char hex code id to a REFI-QDA GUID string.

    REFI-QDA wants ``8-4-4-4-12`` hex GUIDs. Our internal IDs are
    12-char hex (matching every other Scribe id). We pad the high bits
    with zeros, producing a stable bijection so a future REFI-QDA
    importer (F6.6) can recover the original Scribe id by stripping
    the leading zeros.

    Lower-cased per RFC 4122 convention; REFI-QDA is case-insensitive.
    """
    if not isinstance(code_id, str) or len(code_id) != 12:
        raise ValueError(
            f"code_id must be 12-char hex; got {code_id!r}"
        )
    if not all(ch in "0123456789abcdef" for ch in code_id.lower()):
        raise ValueError(
            f"code_id must be 12-char hex; got {code_id!r}"
        )
    cid = code_id.lower()
    return f"00000000-0000-0000-0000-{cid}"


def refi_guid_to_code_id(guid: str) -> str | None:
    """Inverse of :func:`code_id_to_refi_guid`.

    Returns the 12-char Scribe code id if ``guid`` is a zero-padded
    GUID we minted; ``None`` otherwise (a real GUID minted elsewhere
    won't have all-zero high bits).
    """
    if not isinstance(guid, str):
        return None
    g = guid.lower().strip()
    m = re.match(
        r"^00000000-0000-0000-0000-([0-9a-f]{12})$", g
    )
    return m.group(1) if m else None


def _refi_description(c: Code) -> str:
    """Build the ``<Description>`` body for a Code.

    REFI-QDA Codebook 1.0 has *one* free-text body per code. Scribe
    has six (definition, inclusion, exclusion, exemplars, theoretical
    memo, provenance). We fold them into a single labelled block so
    no information is lost. The ordering matches the on-screen
    codebook entry layout for readability.

    Returns an empty string only when every Scribe field is empty —
    in which case the writer omits the ``<Description>`` element
    entirely (it's optional in the schema).
    """
    sections: list[str] = []
    if c.definition:
        sections.append(f"Definition: {c.definition}")
    if c.inclusion_criteria:
        sections.append(f"Inclusion criteria: {c.inclusion_criteria}")
    if c.exclusion_criteria:
        sections.append(f"Exclusion criteria: {c.exclusion_criteria}")
    if c.exemplars:
        examples = "\n".join(f"- {e}" for e in c.exemplars)
        sections.append(f"Exemplars:\n{examples}")
    if c.related_codes:
        rels = "\n".join(
            f"- {r.relation_type}: {r.code_id}"
            for r in c.related_codes
        )
        sections.append(f"Related codes:\n{rels}")
    if c.theoretical_memo:
        sections.append(f"Theoretical memo: {c.theoretical_memo}")
    if c.provenance:
        prov_bits = "; ".join(f"{k}={v}" for k, v in c.provenance.items())
        sections.append(f"Provenance: {prov_bits}")
    if c.stage and c.stage != "initial":
        # Stage default ("initial") is dropped to keep the diff small.
        sections.append(f"Stage: {c.stage}")
    if c.status and c.status != "active":
        sections.append(f"Status: {c.status}")
    return "\n\n".join(sections)


def to_refi_qda_xml(
    codes: Sequence[Code],
    *,
    project: Project | None = None,
    origin: str = REFI_QDA_ORIGIN_DEFAULT,
) -> str:
    """Serialise a codebook to REFI-QDA Codebook 1.0 XML.

    The output's root is ``<CodeBook xmlns="urn:QDA-XML:codebook:1.0"
    origin="...">`` per the schema. Codes are emitted as a forest:
    top-level codes (those without a known parent_code_id in the
    supplied list) sit directly under ``<Codes>``; child codes are
    nested. Cycles in the parent chain — which the F2.1 validator
    forbids but a hand-edited tree could theoretically introduce —
    fall back to flat emission so the export never deadlocks.

    ``isCodable="true"`` is set on every code: Scribe codes are all
    applicable (the REFI-QDA "category-only" mode for un-codable
    grouping nodes isn't part of our model).

    Output is UTF-8 with an XML declaration and pretty-printed indent,
    matching what NVivo / Atlas.ti emit. Empty codebooks produce a
    valid ``<CodeBook><Codes/></CodeBook>``.
    """
    by_id = _index_by_id(codes)

    # Detect cycles up-front; in a cyclic state we emit flat instead
    # of recursing forever. This also flags codes whose parent isn't
    # in `by_id` — they're emitted at the top level (an unknown
    # parent is treated the same as no parent, matching the hierarchy
    # walk in the markdown / RTF helpers).
    safe_parents = _resolve_safe_parents(codes, by_id)

    # Build the XML tree by hand so we control attribute order and
    # the namespace declaration form (NVivo doesn't like the ``ns0:``
    # prefix that ElementTree emits by default).
    ET.register_namespace("", REFI_QDA_NS)
    root = ET.Element(f"{{{REFI_QDA_NS}}}CodeBook")
    if origin:
        root.set("origin", origin)
    if project is not None and project.name.strip():
        root.set("name", project.name)

    codes_el = ET.SubElement(root, f"{{{REFI_QDA_NS}}}Codes")

    # Emit top-level codes (those whose effective parent is None /
    # unknown), then recurse through children.
    children_of: dict[str, list[Code]] = {}
    roots: list[Code] = []
    for c in codes:
        eff_parent = safe_parents.get(c.id)
        if eff_parent is None:
            roots.append(c)
        else:
            children_of.setdefault(eff_parent, []).append(c)

    for c in roots:
        _refi_emit_code(codes_el, c, children_of)

    _indent(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def _resolve_safe_parents(
    codes: Sequence[Code], by_id: dict[str, Code]
) -> dict[str, str | None]:
    """For each code, return the parent id we'll use during XML emit.

    A ``None`` value means "emit at top level". A non-None value means
    "nest inside the named parent." Cycles → flat emission for every
    code in the cycle (each maps to ``None``) so we never recurse
    forever; unknown parents → flat emission for the orphan only.
    """
    result: dict[str, str | None] = {}
    for c in codes:
        p = c.parent_code_id
        if not p or p not in by_id:
            result[c.id] = None
            continue
        # Walk up the chain; if we revisit ourselves it's a cycle.
        seen: set[str] = {c.id}
        cursor = p
        cycle = False
        while cursor:
            if cursor in seen:
                cycle = True
                break
            seen.add(cursor)
            parent_code = by_id.get(cursor)
            if parent_code is None:
                break
            cursor = parent_code.parent_code_id or ""
        result[c.id] = None if cycle else p
    return result


def _refi_emit_code(
    parent_el: ET.Element,
    code: Code,
    children_of: dict[str, list[Code]],
) -> None:
    el = ET.SubElement(parent_el, f"{{{REFI_QDA_NS}}}Code")
    el.set("guid", code_id_to_refi_guid(code.id))
    el.set("name", code.name)
    el.set("isCodable", "true")
    if code.colour:
        # REFI-QDA wants #RRGGBB exactly; expand 3-char form.
        el.set("color", _expand_hex_colour(code.colour))

    desc_text = _refi_description(code)
    if desc_text:
        desc_el = ET.SubElement(el, f"{{{REFI_QDA_NS}}}Description")
        desc_el.text = desc_text

    for child in children_of.get(code.id, []):
        _refi_emit_code(el, child, children_of)


def _expand_hex_colour(colour: str) -> str:
    """Expand ``#RGB`` → ``#RRGGBB``; pass-through for already-6-char."""
    if not colour or not colour.startswith("#"):
        return colour
    body = colour[1:]
    if len(body) == 3:
        return "#" + "".join(ch * 2 for ch in body).upper()
    if len(body) == 6:
        return "#" + body.upper()
    return colour


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-print an ElementTree (Python <3.9 compatible).

    Mirrors the algorithm in the stdlib docs. Works for any namespace.
    """
    pad = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


# --------------------------------------------------------------------------- #
# Format registry + slug + disk-write helpers (F6.1)
#
# F2.6 shipped four pure ``codes -> str`` exporters. F6.1 surfaces them
# behind one common interface so the HTTP endpoint, the CLI, and any
# future UI button all dispatch through the same code path.
#
# What gets added here:
#
#  * :data:`EXPORT_FORMATS` — registry of {key -> FormatSpec(extension,
#    media_type, label)} for the user-facing formats. F6.1 covers CSV,
#    Markdown, and RTF (= "Word"); REFI-QDA XML is intentionally
#    omitted so it can grow its own button at F6.5.
#
#  * :func:`normalise_format` — accepts the canonical keys plus
#    ergonomic aliases (``md`` for Markdown; ``word`` / ``doc`` /
#    ``docx`` for RTF). Raises :class:`ValueError` on anything else.
#
#  * :func:`render_codebook` — single-dispatch helper that picks the
#    right exporter for ``format`` and returns the rendered string.
#
#  * :func:`slugify_codebook_filename` — derives an attachment
#    filename like ``my-project-codebook.csv`` from a project. ASCII-
#    only, lowercased, dash-separated, NFKD-normalised so accented
#    project names downgrade gracefully. Falls back to ``codebook.<ext>``
#    when no project name is available.
#
#  * :func:`write_codebook` — atomic disk write through a ``.tmp``
#    swap so a failed export never leaves a half-written file.
#
# All four are pure: they do no validation beyond format-key
# normalisation. Errors raised: :class:`ValueError` for unknown
# formats, :class:`OSError` from the filesystem in :func:`write_codebook`.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FormatSpec:
    """Static description of a user-facing codebook export format.

    * ``key`` — canonical lookup key (``csv`` / ``markdown`` / ``rtf``).
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


# Registry of formats F6.1 surfaces. REFI-QDA XML is deliberately
# absent — F6.5 owns that button so it can grow its own
# project-archive metadata, ``<UserCodes>`` per-coder layer, etc.
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
}


# Aliases the user might type. Resolved before lookup in EXPORT_FORMATS.
_FORMAT_ALIASES: dict[str, str] = {
    "md": EXPORT_FORMAT_MARKDOWN,
    "markdown": EXPORT_FORMAT_MARKDOWN,
    "csv": EXPORT_FORMAT_CSV,
    "rtf": EXPORT_FORMAT_RTF,
    # "Word" routes to RTF — RTF is the format Word opens natively
    # without a ``.docx`` ZIP ceremony, and F2.6 ships an RTF exporter.
    # We accept the obvious nicknames so the UX is forgiving.
    "word": EXPORT_FORMAT_RTF,
    "doc": EXPORT_FORMAT_RTF,
    "docx": EXPORT_FORMAT_RTF,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Case-insensitive; trims whitespace; recognises a small handful of
    aliases (``md`` → ``markdown``; ``word`` / ``doc`` / ``docx`` →
    ``rtf``). Raises :class:`ValueError` for unknown formats with the
    list of accepted keys, so the message is actionable.
    """
    if format is None:
        raise ValueError(
            "Codebook export format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported codebook export format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


# Map normalised format key → renderer. Populated below the
# function definitions to avoid forward references.
_RENDERERS: dict[str, Callable[..., str]] = {}


def render_codebook(
    format: str,
    codes: Sequence[Code],
    *,
    project: Project | None = None,
) -> str:
    """Render the codebook in ``format`` and return the string body.

    Dispatches via :func:`normalise_format` so callers can pass the
    same alias set the HTTP query string accepts. ``project`` is
    forwarded only to the renderers that use it (Markdown + RTF) — CSV
    intentionally has no project header so the column shape stays the
    public contract.

    Empty codebooks are valid input and produce a header-only CSV / a
    placeholder Markdown / a minimal RTF. Never raises on empty input.
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](codes, project=project)


def _render_csv(codes: Sequence[Code], *, project: Project | None) -> str:
    # CSV ignores the project header — schema is the public contract.
    del project
    return to_csv(codes)


def _render_markdown(codes: Sequence[Code], *, project: Project | None) -> str:
    return to_markdown(codes, project=project)


def _render_rtf(codes: Sequence[Code], *, project: Project | None) -> str:
    return to_rtf(codes, project=project)


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_MARKDOWN] = _render_markdown
_RENDERERS[EXPORT_FORMAT_RTF] = _render_rtf


# Slug regex: collapse runs of non-alphanumeric ASCII into a single
# dash. We NFKD-normalise + strip combining marks first so "Élise"
# becomes "elise", not the empty string. The result is bounded in
# length below so runaway project names don't produce 5 KB filenames.
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def slugify_codebook_filename(
    project: Project | None, format: str
) -> str:
    """Build a download-friendly filename for a codebook export.

    Pattern: ``<project-slug>-codebook<ext>`` if a project name is
    available; ``codebook<ext>`` otherwise. The slug is ASCII-only,
    lowercased, dash-separated, and capped at
    :data:`_FILENAME_SLUG_MAX` characters before the suffix.

    Raises :class:`ValueError` for unknown formats (delegates to
    :func:`normalise_format`).
    """
    fmt = normalise_format(format)
    spec = EXPORT_FORMATS[fmt]
    slug = ""
    if project is not None and project.name and project.name.strip():
        # NFKD-normalise so combining marks separate from base
        # characters, then drop anything non-ASCII. This downgrades
        # "Café" to "cafe" rather than dropping the whole word.
        ascii_name = (
            unicodedata.normalize("NFKD", project.name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        slug = _SLUG_RE.sub("-", ascii_name.lower()).strip("-")
        if len(slug) > _FILENAME_SLUG_MAX:
            slug = slug[:_FILENAME_SLUG_MAX].rstrip("-")
    if slug:
        return f"{slug}-codebook{spec.extension}"
    return f"codebook{spec.extension}"


def write_codebook(
    path: Path,
    format: str,
    codes: Sequence[Code],
    *,
    project: Project | None = None,
) -> Path:
    """Render the codebook in ``format`` and write it to ``path``.

    Writes are atomic: the body is written to ``<path>.tmp`` first and
    only ``replace()``-d into place on success, so an interrupted write
    never leaves a half-finished export visible to other readers.
    Creates ``path.parent`` if missing.

    Returns ``path`` (as :class:`pathlib.Path`) for chaining convenience.
    """
    fmt = normalise_format(format)
    text = render_codebook(fmt, codes, project=project)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    # Write bytes rather than text so the platform's newline policy
    # never rewrites the CSV exporter's RFC-4180 ``\r\n`` line endings
    # into ``\n`` (which would break strict CSV re-importers like
    # Microsoft Power Query in legacy mode).
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(target)
    return target
