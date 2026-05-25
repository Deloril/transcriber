"""REFI-QDA / QDPX project export (F6.4).

Per PLANNING.md F6.4 — the **"no lock-in" feature**:

  > REFI-QDA / QDPX project export (priority — this is the "no
  > lock-in" feature).

A QDPX file is a zip archive containing a single ``project.qde`` XML
manifest plus a ``Sources/`` folder with each source's plain-text
representation. Any QDA tool that imports REFI-QDA (Atlas.ti, MAXQDA,
NVivo, QDA Miner, Quirkos, Dedoose) will accept the output.

This module is **pure** in the same sense as :mod:`scribe.codebook_export`,
:mod:`scribe.retrieval_report`, and :mod:`scribe.matrix_export`: it
takes already-loaded entities + transcript segments and returns
``str`` / ``bytes``. The CLI (:mod:`scribe.scripts.export_qdpx`) and
the eventual HTTP endpoint do disk I/O and call in here for rendering.

What we emit
------------

A ``project.qde`` shaped like::

    <?xml version="1.0" encoding="UTF-8"?>
    <Project xmlns="urn:QDA-XML:project:1.0"
             name="..." origin="Scribe"
             creatingUserGUID="..." creationDateTime="...">
      <Users>
        <User guid="..." name="..." id="..."/>
      </Users>
      <CodeBook>
        <Codes>
          <Code guid="..." name="..." color="..." isCodable="true">
            <Description>...</Description>
            <Code .../>  <!-- nested children -->
          </Code>
        </Codes>
      </CodeBook>
      <Sources>
        <TextSource guid="..." name="..."
                    plainTextPath="internal://Sources/<sid>.txt"
                    creatingUser="..." creationDateTime="...">
          <PlainTextSelection guid="..." name="..."
                              startPosition="..." endPosition="..."
                              creatingUser="..." creationDateTime="...">
            <Coding guid="..." creatingUser="..." creationDateTime="...">
              <CodeRef targetGUID="..."/>
            </Coding>
          </PlainTextSelection>
        </TextSource>
      </Sources>
      <Notes>
        <Note guid="..." name="..."
              plainTextPath="internal://Notes/<nid>.txt"
              creatingUser="..." creationDateTime="..."/>
      </Notes>
    </Project>

Plus a zip layout::

    project.qde
    Sources/<source_id>.txt    (one per text source)
    Notes/<memo_id>.txt        (one per memo)

The plain-text source files are derived from the Scribe transcript
segments: each segment becomes a single line, optionally prefixed with
``"<speaker>: "``. Word-id anchored applications are converted to
plain-text ``[startPosition, endPosition]`` offsets (REFI-QDA's native
unit) by walking the same rendering pipeline that produced the file —
that's what makes the round-trip safe.

Scope and what's deferred
-------------------------

This iteration ships the **core** of F6.4: a pure builder for QDPX
archives covering project metadata, codebook (with hierarchy +
descriptions + colours), text sources with plain-text representations,
code applications with computed character offsets, memos as Notes, and
coders as Users.

Deferred (a future iteration can extend this module without breaking
on-disk format):

  * Cases / Sets — Atlas.ti / MAXQDA expose ``<Cases>`` for
    participant-level grouping. Doable from the F3.3 participant↔source
    mapping but not yet emitted; QDA tools degrade gracefully when
    Cases are absent.
  * Variables / classification fields — F3.2 source-attribute schema
    could be mapped to ``<Variable>`` / ``<VariableValue>`` triples;
    deferred to keep this PR scoped.
  * Audio / video media — only plain-text sources are emitted today.
    A future pass can copy the source media into ``Sources/`` and
    upgrade the element to ``<AudioSource>`` / ``<VideoSource>``.
  * REFI-QDA project import — that's F6.6, a separate feature.

The id mapping
--------------

REFI-QDA wants 8-4-4-4-12 hex GUIDs everywhere. Scribe uses 12-char
hex everywhere. We pad with zeros (``00000000-0000-0000-tttt-XXXXXXXXXXXX``
where ``tttt`` is a 4-char tag identifying the entity kind), giving a
bijective mapping a future importer can reverse. The kind tag means
the same Scribe id used as both a code and a coder won't produce the
same GUID — which would otherwise be a collision risk on round-trip.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

from .applications import Application, parse_word_id
from .application_reanchor import collect_word_texts
from .codes import Code
from .coders import Coder
from .memos import Memo
from .projects import Project, utcnow_iso
from .sources import Source


# --------------------------------------------------------------------------- #
# REFI-QDA Project schema
# --------------------------------------------------------------------------- #

# Namespace URI for the REFI-QDA Project XML schema. Distinct from the
# Codebook XML namespace handled in :mod:`scribe.codebook_export`.
REFI_QDA_PROJECT_NS = "urn:QDA-XML:project:1.0"

# What we set as the ``origin`` attribute. Identifies Scribe to a
# downstream importer.
REFI_QDA_PROJECT_ORIGIN_DEFAULT = "Scribe"

# REFI-QDA datetimes use ISO-8601 with no fractional seconds and a
# ``Z`` suffix per the schema. Helper below normalises the Scribe
# stored timestamps (which carry microseconds) to that.
_DATETIME_FRACTION_RE = re.compile(r"\.\d+")

# Kind tags for the GUID padding scheme. 4 hex chars each, distinct
# per entity kind so the mapping {kind, scribe_id} → guid is bijective.
_KIND_TAG_CODE = "c0de"
_KIND_TAG_SOURCE = "5046"      # "5046" ≈ "SoFi" reversed, just stable hex
_KIND_TAG_USER = "0c0d"
_KIND_TAG_NOTE = "10e7"
_KIND_TAG_SELECTION = "5e1c"
_KIND_TAG_CODING = "c0d1"
_KIND_TAG_PROJECT = "9301"

_SCRIBE_ID_RE = re.compile(r"^[0-9a-fA-F]{12}$")

# Default speaker label suffix used when an originating segment has no
# speaker tag (single-track / mono transcripts).
_NO_SPEAKER = ""


# --------------------------------------------------------------------------- #
# GUID mapping
# --------------------------------------------------------------------------- #


def scribe_id_to_guid(scribe_id: str, *, kind_tag: str) -> str:
    """Map a 12-char hex Scribe id to a REFI-QDA 8-4-4-4-12 GUID.

    ``kind_tag`` must be 4 hex chars and identifies the entity kind
    (code, source, user, note, …). Combined with the 12-char id it
    forms the trailing 4-12 segment of the GUID; the leading
    8-4 segment is zeros so the mapping is reversible. Lower-cased
    per RFC 4122 convention.

    A future REFI-QDA *importer* (F6.6) can recognise our GUIDs and
    recover the original Scribe ids without renumbering every entity.
    """
    if not isinstance(scribe_id, str) or not _SCRIBE_ID_RE.match(scribe_id):
        raise ValueError(f"scribe_id must be 12-char hex; got {scribe_id!r}")
    if not isinstance(kind_tag, str) or not re.match(r"^[0-9a-f]{4}$", kind_tag):
        raise ValueError(f"kind_tag must be 4-char hex; got {kind_tag!r}")
    return f"00000000-0000-0000-{kind_tag}-{scribe_id.lower()}"


def code_guid(code_id: str) -> str:
    """REFI-QDA GUID for a Scribe :class:`Code`."""
    return scribe_id_to_guid(code_id, kind_tag=_KIND_TAG_CODE)


def source_guid(source_id: str) -> str:
    """REFI-QDA GUID for a Scribe :class:`Source`."""
    return scribe_id_to_guid(source_id, kind_tag=_KIND_TAG_SOURCE)


def user_guid(coder_id: str) -> str:
    """REFI-QDA GUID for a Scribe :class:`Coder` (REFI-QDA "User")."""
    return scribe_id_to_guid(coder_id, kind_tag=_KIND_TAG_USER)


def note_guid(memo_id: str) -> str:
    """REFI-QDA GUID for a Scribe :class:`Memo` (REFI-QDA "Note")."""
    return scribe_id_to_guid(memo_id, kind_tag=_KIND_TAG_NOTE)


def selection_guid(application_id: str) -> str:
    """REFI-QDA GUID for the ``<PlainTextSelection>`` of an application."""
    return scribe_id_to_guid(application_id, kind_tag=_KIND_TAG_SELECTION)


def coding_guid(application_id: str) -> str:
    """REFI-QDA GUID for the ``<Coding>`` of an application.

    Distinct from :func:`selection_guid` because a single selection
    can in theory carry several Coding children (different codes on
    the same span). Scribe models that as separate Applications, but
    the REFI-QDA shape still wants distinct GUIDs.
    """
    return scribe_id_to_guid(application_id, kind_tag=_KIND_TAG_CODING)


def project_guid(project_id: str) -> str:
    """REFI-QDA GUID for the project itself."""
    return scribe_id_to_guid(project_id, kind_tag=_KIND_TAG_PROJECT)


# --------------------------------------------------------------------------- #
# Plain-text source rendering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WordOffset:
    """Plain-text [start, end) char offsets for one word in a source.

    ``start`` is inclusive, ``end`` is exclusive — matches Python
    string slicing conventions and the REFI-QDA schema's
    ``[startPosition, endPosition)`` interpretation (NVivo and
    Atlas.ti both treat ``endPosition`` as exclusive).
    """

    word_id: str
    start: int
    end: int


@dataclass(frozen=True)
class RenderedSource:
    """The plain-text rendering of one source plus its word-offset map."""

    source_id: str
    text: str
    # word_id → WordOffset. Order-preserving (insertion order).
    offsets: dict[str, WordOffset]


def render_source_plain_text(
    source_id: str,
    segments: Sequence[Mapping[str, Any]],
    *,
    include_speaker_labels: bool = True,
) -> RenderedSource:
    """Render a Scribe transcript to flat plain text + word offsets.

    Each segment becomes a single newline-separated line. With
    ``include_speaker_labels=True`` (the default), a non-empty
    ``segment["speaker"]`` is emitted as ``"<speaker>: "`` at the
    start of the line — the same shape ``writers.write_txt`` uses,
    minus the timestamp prefix (timestamps would interfere with QDA
    tools that scan for participant turns by line-leading regex).

    Word offsets are computed against the rendered text so a caller
    can convert a word-id-anchored application into a
    ``[startPosition, endPosition]`` pair in REFI-QDA's units. Words
    are joined within a segment by a single space; words that are
    pure whitespace contribute nothing to the offset map.
    """
    parts: list[str] = []
    offsets: dict[str, WordOffset] = {}
    cursor = 0  # current write position in ``"".join(parts)``
    last_speaker: str | None = None
    seg_count = 0

    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            continue
        words = seg.get("words")
        if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
            words = []

        # Newline between segments. We add one *before* each segment
        # except the first so a trailing blank line never appears.
        if seg_count > 0:
            parts.append("\n")
            cursor += 1
        seg_count += 1

        # Speaker prefix. ``writers.write_txt`` groups consecutive
        # same-speaker segments under one heading; we keep it simpler
        # here (one prefix per segment) so the offset bookkeeping is
        # straightforward, and so the REFI-QDA emitter doesn't have
        # to know about speaker grouping.
        speaker = ""
        if include_speaker_labels:
            raw_speaker = seg.get("speaker") or ""
            if isinstance(raw_speaker, str) and raw_speaker.strip():
                speaker = raw_speaker.strip()
        if speaker:
            prefix = f"{speaker}: "
            parts.append(prefix)
            cursor += len(prefix)
            last_speaker = speaker
        else:
            last_speaker = None

        for w_idx, w in enumerate(words):
            if not isinstance(w, Mapping):
                continue
            text = str(w.get("text", "") or "")
            if not text.strip():
                # A pure-whitespace word still costs us a space if a
                # neighbour follows; but we'd rather skip than carry
                # a zero-length offset.
                continue

            # Space separator between words within a segment. Adds
            # exactly one space before the second-and-later non-empty
            # words.
            if w_idx > 0 and parts and not parts[-1].endswith(" ") and not parts[-1].endswith("\n"):
                # Only add if the previous chunk wasn't already a
                # speaker prefix (which ends in ": " — has a trailing
                # space). The endswith-space check handles that case.
                parts.append(" ")
                cursor += 1

            start = cursor
            parts.append(text)
            cursor += len(text)
            wid = f"s{seg_idx}w{w_idx}"
            offsets[wid] = WordOffset(word_id=wid, start=start, end=cursor)

    return RenderedSource(
        source_id=source_id,
        text="".join(parts),
        offsets=offsets,
    )


def application_plain_text_offsets(
    application: Application,
    rendered: RenderedSource,
) -> tuple[int, int] | None:
    """Convert an application's word-id anchor → [start, end) char offsets.

    Returns ``None`` when either anchor word id is missing from
    ``rendered.offsets`` (i.e. the anchor is orphaned in this
    rendering — F4.5 territory; we don't try to recover it here).

    Sub-word ``start_char_offset`` / ``end_char_offset`` on the
    application are honoured: they shift the start/end inward,
    bounded by the word's own length. The ``end`` returned is
    exclusive, matching :class:`WordOffset` and REFI-QDA's positional
    semantics.
    """
    start_w = rendered.offsets.get(application.anchor_start_word_id)
    end_w = rendered.offsets.get(application.anchor_end_word_id)
    if start_w is None or end_w is None:
        return None

    start = start_w.start
    end = end_w.end

    if application.start_char_offset is not None:
        word_len = start_w.end - start_w.start
        off = max(0, min(application.start_char_offset, word_len))
        start = start_w.start + off

    if application.end_char_offset is not None:
        word_len = end_w.end - end_w.start
        off = max(0, min(application.end_char_offset, word_len))
        end = end_w.start + off

    if end < start:
        end = start
    return (start, end)


# --------------------------------------------------------------------------- #
# Code-tree helpers
# --------------------------------------------------------------------------- #


def _index_codes(codes: Sequence[Code]) -> dict[str, Code]:
    return {c.id: c for c in codes}


def _resolve_safe_parents(
    codes: Sequence[Code], by_id: dict[str, Code]
) -> dict[str, str | None]:
    """Mirror of :func:`scribe.codebook_export._resolve_safe_parents`.

    For each code, return the parent id we'll nest under in the XML
    tree (``None`` → top level). Cyclic chains fall back to flat
    emission for everyone in the cycle so we never recurse forever;
    unknown parents → flat emission for the orphan only.
    """
    result: dict[str, str | None] = {}
    for c in codes:
        p = c.parent_code_id
        if not p or p not in by_id:
            result[c.id] = None
            continue
        seen: set[str] = {c.id}
        cursor: str | None = p
        cycle = False
        while cursor:
            if cursor in seen:
                cycle = True
                break
            seen.add(cursor)
            parent_code = by_id.get(cursor)
            if parent_code is None:
                break
            cursor = parent_code.parent_code_id or None
        result[c.id] = None if cycle else p
    return result


def _expand_hex_colour(colour: str) -> str:
    """Mirror of :func:`scribe.codebook_export._expand_hex_colour`."""
    if not colour or not colour.startswith("#"):
        return colour
    body = colour[1:]
    if len(body) == 3:
        return "#" + "".join(ch * 2 for ch in body).upper()
    if len(body) == 6:
        return "#" + body.upper()
    return colour


def _code_description(c: Code) -> str:
    """Build the ``<Description>`` body for a Code.

    Mirrors :func:`scribe.codebook_export._refi_description` but
    re-implemented locally so this module doesn't depend on the
    Codebook-XML internals (which may diverge over time). Folds the
    six Scribe code fields into a single labelled block.
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
        sections.append(f"Stage: {c.stage}")
    if c.status and c.status != "active":
        sections.append(f"Status: {c.status}")
    return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Datetime helpers
# --------------------------------------------------------------------------- #


def _normalise_iso_datetime(ts: str) -> str:
    """Strip fractional seconds from a Scribe ISO-8601 timestamp.

    REFI-QDA's schema accepts ``YYYY-MM-DDTHH:MM:SSZ`` (no fractional
    seconds); Scribe's :func:`scribe.projects.utcnow_iso` writes
    microseconds. We trim them off here so the output validates
    against strict importers (NVivo's parser is famously picky).

    Empty / invalid inputs fall back to the current UTC time so the
    output always has a creationDateTime.
    """
    if isinstance(ts, str) and ts:
        cleaned = _DATETIME_FRACTION_RE.sub("", ts)
        if cleaned.endswith("Z"):
            return cleaned
        # Add Z if it's missing but the rest looks ISO-shaped
        if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", cleaned):
            return cleaned + "Z"
    # Fallback — never want to emit empty datetimes
    return _normalise_iso_datetime(utcnow_iso())


# --------------------------------------------------------------------------- #
# XML emit
# --------------------------------------------------------------------------- #


def _q(tag: str) -> str:
    """Namespaced tag for the project XML."""
    return f"{{{REFI_QDA_PROJECT_NS}}}{tag}"


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-print, namespace-safe."""
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


def _emit_code(parent_el: ET.Element, code: Code, children_of: dict[str, list[Code]]) -> None:
    el = ET.SubElement(parent_el, _q("Code"))
    el.set("guid", code_guid(code.id))
    el.set("name", code.name)
    el.set("isCodable", "true")
    if code.colour:
        el.set("color", _expand_hex_colour(code.colour))

    desc_text = _code_description(code)
    if desc_text:
        desc_el = ET.SubElement(el, _q("Description"))
        desc_el.text = desc_text

    for child in children_of.get(code.id, []):
        _emit_code(el, child, children_of)


def _emit_users(root: ET.Element, coders: Sequence[Coder]) -> None:
    if not coders:
        # An empty <Users/> is valid but importers prefer at least one
        # user. Don't fabricate one — emit empty and let the writer
        # supply a fallback.
        users_el = ET.SubElement(root, _q("Users"))
        return
    users_el = ET.SubElement(root, _q("Users"))
    for c in coders:
        u = ET.SubElement(users_el, _q("User"))
        u.set("guid", user_guid(c.id))
        u.set("name", c.name or c.id)
        # ``id`` is the human-readable handle in REFI-QDA; safe to
        # reuse the Scribe coder id.
        u.set("id", c.id)


def _emit_codebook(root: ET.Element, codes: Sequence[Code]) -> None:
    cb_el = ET.SubElement(root, _q("CodeBook"))
    codes_el = ET.SubElement(cb_el, _q("Codes"))

    by_id = _index_codes(codes)
    safe_parents = _resolve_safe_parents(codes, by_id)
    children_of: dict[str, list[Code]] = {}
    roots: list[Code] = []
    for c in codes:
        p = safe_parents.get(c.id)
        if p is None:
            roots.append(c)
        else:
            children_of.setdefault(p, []).append(c)
    for c in roots:
        _emit_code(codes_el, c, children_of)


def _emit_sources(
    root: ET.Element,
    sources: Sequence[Source],
    rendered_by_source: dict[str, RenderedSource],
    applications_by_source: dict[str, list[Application]],
    creating_user: str,
    fallback_dt: str,
) -> None:
    sources_el = ET.SubElement(root, _q("Sources"))
    for s in sources:
        rendered = rendered_by_source.get(s.id)
        ts = ET.SubElement(sources_el, _q("TextSource"))
        ts.set("guid", source_guid(s.id))
        ts.set("name", s.name or s.id)
        ts.set("plainTextPath", f"internal://Sources/{s.id}.txt")
        ts.set("creatingUser", creating_user)
        ts.set("creationDateTime", _normalise_iso_datetime(s.created_at or fallback_dt))

        for app in applications_by_source.get(s.id, []):
            if rendered is None:
                continue  # no plain text → cannot place this app
            offsets = application_plain_text_offsets(app, rendered)
            if offsets is None:
                continue  # orphan in this rendering; skip
            start, end = offsets
            sel = ET.SubElement(ts, _q("PlainTextSelection"))
            sel.set("guid", selection_guid(app.id))
            sel.set("name", "")
            sel.set("startPosition", str(start))
            sel.set("endPosition", str(end))
            sel.set("creatingUser", user_guid(app.coder_id) if _SCRIBE_ID_RE.match(app.coder_id or "") else creating_user)
            sel.set("creationDateTime", _normalise_iso_datetime(app.created_at or fallback_dt))

            cod = ET.SubElement(sel, _q("Coding"))
            cod.set("guid", coding_guid(app.id))
            cod.set("creatingUser", sel.get("creatingUser") or creating_user)
            cod.set("creationDateTime", sel.get("creationDateTime") or fallback_dt)
            cref = ET.SubElement(cod, _q("CodeRef"))
            cref.set("targetGUID", code_guid(app.code_id))


def _emit_notes(root: ET.Element, memos: Sequence[Memo], creating_user: str, fallback_dt: str) -> None:
    if not memos:
        # <Notes/> is optional; omit if empty so consumers don't see
        # an empty container.
        return
    notes_el = ET.SubElement(root, _q("Notes"))
    for m in memos:
        n = ET.SubElement(notes_el, _q("Note"))
        n.set("guid", note_guid(m.id))
        # REFI-QDA Notes have a ``name``; we use the title or fall
        # back to the type+id so a Note never has an empty name.
        name = m.title.strip() if m.title else f"[{m.type}] {m.id}"
        n.set("name", name)
        n.set("plainTextPath", f"internal://Notes/{m.id}.txt")
        n.set("creatingUser", creating_user)
        n.set("creationDateTime", _normalise_iso_datetime(m.created_at or fallback_dt))


# --------------------------------------------------------------------------- #
# Top-level builders
# --------------------------------------------------------------------------- #


def to_qde_xml(
    *,
    project: Project,
    sources: Sequence[Source] = (),
    codes: Sequence[Code] = (),
    applications: Sequence[Application] = (),
    memos: Sequence[Memo] = (),
    coders: Sequence[Coder] = (),
    rendered_sources: Sequence[RenderedSource] = (),
    origin: str = REFI_QDA_PROJECT_ORIGIN_DEFAULT,
    now: str | None = None,
) -> str:
    """Build the ``project.qde`` XML body.

    All inputs are optional except ``project``; a project with no
    sources / codes / memos still produces a valid (if sparse)
    document. ``rendered_sources`` is an iterable of
    :class:`RenderedSource` produced upstream — usually one per
    source via :func:`render_source_plain_text`. Sources without a
    matching :class:`RenderedSource` will appear in the XML but
    without any nested ``<PlainTextSelection>`` elements (their
    applications can't be placed without offsets).

    ``coders`` populates ``<Users>``. If empty, a single fallback
    "Scribe" user is emitted so REFI-QDA importers that require a
    creatingUser don't choke.

    Output is UTF-8 with an XML declaration and pretty-printed
    indent.
    """
    fallback_dt = _normalise_iso_datetime(now or utcnow_iso())

    rendered_by_source = {r.source_id: r for r in rendered_sources}
    apps_by_source: dict[str, list[Application]] = {}
    for a in applications:
        apps_by_source.setdefault(a.source_id, []).append(a)
    # Stable order within each source: by created_at then id.
    for sid, lst in apps_by_source.items():
        lst.sort(key=lambda a: (a.created_at or "", a.id))

    ET.register_namespace("", REFI_QDA_PROJECT_NS)
    root = ET.Element(_q("Project"))
    root.set("name", project.name)
    root.set("origin", origin or REFI_QDA_PROJECT_ORIGIN_DEFAULT)
    root.set("creationDateTime", _normalise_iso_datetime(project.created_at or fallback_dt))
    root.set("modifiedDateTime", _normalise_iso_datetime(project.modified_at or fallback_dt))

    # If we have at least one coder, prefer the first as the
    # creatingUserGUID; otherwise mint a deterministic project-level
    # fallback. Either way the Users element below carries it.
    if coders:
        creating_user = user_guid(coders[0].id)
    else:
        creating_user = project_guid(project.id)
    root.set("creatingUserGUID", creating_user)

    # Users
    if coders:
        _emit_users(root, coders)
    else:
        users_el = ET.SubElement(root, _q("Users"))
        u = ET.SubElement(users_el, _q("User"))
        u.set("guid", creating_user)
        u.set("name", "Scribe")
        u.set("id", "scribe")

    # CodeBook
    _emit_codebook(root, codes)

    # Sources + applications
    _emit_sources(
        root,
        sources,
        rendered_by_source,
        apps_by_source,
        creating_user=creating_user,
        fallback_dt=fallback_dt,
    )

    # Notes (memos)
    _emit_notes(root, memos, creating_user=creating_user, fallback_dt=fallback_dt)

    _indent(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def to_qdpx(
    *,
    project: Project,
    sources: Sequence[Source] = (),
    codes: Sequence[Code] = (),
    applications: Sequence[Application] = (),
    memos: Sequence[Memo] = (),
    coders: Sequence[Coder] = (),
    rendered_sources: Sequence[RenderedSource] = (),
    origin: str = REFI_QDA_PROJECT_ORIGIN_DEFAULT,
    now: str | None = None,
) -> bytes:
    """Bundle a project into a QDPX zip archive.

    Layout::

        project.qde
        Sources/<source_id>.txt
        Notes/<memo_id>.txt

    Returns the archive as ``bytes`` so the caller (CLI / HTTP
    handler) can stream it to disk or to the wire without an
    intermediate temp file. Deterministic content order: ``project.qde``
    first, then sources alphabetised by id, then notes alphabetised
    by id.
    """
    qde = to_qde_xml(
        project=project,
        sources=sources,
        codes=codes,
        applications=applications,
        memos=memos,
        coders=coders,
        rendered_sources=rendered_sources,
        origin=origin,
        now=now,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.qde", qde.encode("utf-8"))

        # Sources/ — one .txt per rendered source. Sorted for
        # determinism. Sources without a rendering (no transcript
        # supplied) are skipped silently; the XML already omitted
        # any selections for them.
        for r in sorted(rendered_sources, key=lambda r: r.source_id):
            zf.writestr(f"Sources/{r.source_id}.txt", r.text.encode("utf-8"))

        # Notes/ — one .txt per memo. Body only; the title (if any)
        # is on the XML <Note name="..."> attribute. Sorted by id.
        for m in sorted(memos, key=lambda m: m.id):
            body = m.body if m.body is not None else ""
            zf.writestr(f"Notes/{m.id}.txt", body.encode("utf-8"))

    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Filename slug
# --------------------------------------------------------------------------- #


_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def slugify_qdpx_filename(project: Project | None) -> str:
    """Derive a ``<slug>.qdpx`` attachment filename from a project.

    ASCII-only, lowercase, dash-separated. Falls back to
    ``project.qdpx`` when no usable name is available.
    """
    if project is None or not project.name:
        return "project.qdpx"
    name = unicodedata.normalize("NFKD", project.name)
    name = name.encode("ascii", "ignore").decode("ascii").lower()
    slug = _PUNCT_RE.sub("-", name).strip("-")
    if not slug:
        return "project.qdpx"
    return f"{slug}.qdpx"


# --------------------------------------------------------------------------- #
# Re-export wordlist helper for tests + downstream callers
# --------------------------------------------------------------------------- #

__all__ = [
    "REFI_QDA_PROJECT_NS",
    "REFI_QDA_PROJECT_ORIGIN_DEFAULT",
    "RenderedSource",
    "WordOffset",
    "application_plain_text_offsets",
    "code_guid",
    "coding_guid",
    "note_guid",
    "project_guid",
    "render_source_plain_text",
    "scribe_id_to_guid",
    "selection_guid",
    "slugify_qdpx_filename",
    "source_guid",
    "to_qde_xml",
    "to_qdpx",
    "user_guid",
]
