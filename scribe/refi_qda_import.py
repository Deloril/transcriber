"""REFI-QDA / QDPX project import (F6.6).

Per PLANNING.md F6.6 — the **inverse** of F6.4:

  > REFI-QDA / QDPX project import (later milestone).

A QDPX file is a zip archive containing a single ``project.qde`` XML
manifest plus a ``Sources/`` folder with each source's plain-text
representation (and optionally a ``Notes/`` folder for memos). This
module accepts such an archive, parses it, and produces a Scribe
:class:`Project` plus the entities it owns (sources, codes, coders,
memos, applications, and one synthetic :class:`CodeVersion` per
imported code so applications have a valid
``definition_version_id_at_apply``).

This module is **pure** in the same sense as
:mod:`scribe.refi_qda_project`: it takes already-loaded bytes / XML /
text and returns dataclasses. The CLI / HTTP layer that wraps it does
disk I/O and project persistence.

What we accept
--------------

* Scribe-origin QDPX files (the inverse of F6.4). GUIDs use Scribe's
  padding scheme (``00000000-0000-0000-tttt-XXXXXXXXXXXX``) so the
  importer recovers the original 12-char Scribe ids on round-trip.
  Sources (``5046``), Codes (``c0de``), Users (``0c0d``), Notes
  (``10e7``), and Selections (``5e1c``) all decode back to their
  original ids; entity ``project_id`` fields are rewritten to the new
  Project's id (Scribe doesn't try to preserve the project id across
  installs — that would make collisions on a shared workstation
  silent).

* Foreign QDPX files (Atlas.ti, MAXQDA, NVivo, Quirkos, Dedoose, QDA
  Miner). Their GUIDs don't match Scribe's padding, so we mint fresh
  Scribe ids for every entity. The mapping ``QDPX guid → Scribe id``
  is built up as the XML walks so ``CodeRef``/``creatingUser``
  cross-references resolve consistently.

What we don't yet handle
------------------------

* Cases / Sets — REFI-QDA's participant-grouping construct. F1.3 has
  Participants but the QDPX schema's Cases element is a separate
  feature (F6.6 will be revisited when F3.3 + F6.6 round-trip lands).
* Variables / VariableValues — F3.2's source-attribute schema could
  consume these on import. Deferred to keep this PR scoped.
* Audio / video media in ``Sources/`` — only ``<TextSource>`` is
  consumed; ``<AudioSource>``/``<VideoSource>`` are skipped with a
  warning. F10.3 (transcript import) is the right place to wire
  media playback in.
* Code revision history. The exporter doesn't emit it; on import each
  imported code gets a single synthetic ``version=1`` snapshot whose
  id every imported application of that code references.

Design principles
-----------------

* **Pure data**. No filesystem writes from this module; the caller
  decides where things land. Returned :class:`ImportResult` carries
  every entity needed to persist the project plus the rendered
  source texts so callers can write transcript sidecars without
  re-tokenising.
* **Best-effort, never throw on quirks**. A missing ``creatingUser``,
  an unparseable ``startPosition``, a ``CodeRef`` pointing at an
  unknown code — all surface as a string in
  :attr:`ImportResult.warnings` and the affected element is
  skipped. The whole archive imports as far as it can rather than
  failing closed.
* **Round-trippable on Scribe-origin data**. Export → import →
  export should produce the same set of entities (modulo new
  Project / CodeVersion ids that the import mints).

Tokenisation of plain-text sources
----------------------------------

REFI-QDA anchors selections to ``[startPosition, endPosition)`` char
offsets in the source's plain-text body. Scribe anchors applications
to word ids of the shape ``s<seg>w<word>`` (see
:mod:`scribe.applications`). To bridge the two we tokenise each
imported source into segments (one per newline-separated line) and
words (one per non-whitespace run within a line), recording the
char-offset span for every word. A speaker prefix on the line of the
form ``SPEAKER: ...`` is detected and stripped if it looks like one;
the remaining words get sequential ``s{seg}w{word}`` ids matching
the same scheme used by the export pipeline. Looking up a selection
becomes a binary-style search over the words for the ``startPosition``
and ``endPosition``.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

from .applications import (
    Application,
    new_application_id,
)
from .codes import (
    CODE_COLOUR_RE,
    Code,
    CodeRelation,
    new_code_id,
)
from .code_versions import CodeVersion
from .coders import Coder, new_coder_id
from .memos import Memo, new_memo_id
from .projects import Project, new_project_id, utcnow_iso
from .refi_qda_project import REFI_QDA_PROJECT_NS
from .sources import Source, new_source_id


# --------------------------------------------------------------------------- #
# Kind-tag constants — mirrored from refi_qda_project so this module
# doesn't reach into a sibling's private API. If those tags ever change
# both sides break together.
# --------------------------------------------------------------------------- #

_KIND_TAG_CODE = "c0de"
_KIND_TAG_SOURCE = "5046"
_KIND_TAG_USER = "0c0d"
_KIND_TAG_NOTE = "10e7"
_KIND_TAG_SELECTION = "5e1c"
_KIND_TAG_CODING = "c0d1"
_KIND_TAG_PROJECT = "9301"


# Regex for the Scribe GUID padding scheme. Group 1 captures the kind
# tag, group 2 captures the 12-char Scribe id. Both lower-case (per
# RFC 4122 / refi_qda_project.scribe_id_to_guid).
_SCRIBE_GUID_RE = re.compile(
    r"^00000000-0000-0000-([0-9a-f]{4})-([0-9a-f]{12})$"
)


def parse_guid_to_scribe_id(guid: str | None, *, kind_tag: str) -> str | None:
    """If ``guid`` matches Scribe's padding for ``kind_tag``, return the id.

    Returns ``None`` for foreign GUIDs (Atlas.ti UUIDs etc.) and for
    GUIDs whose kind tag doesn't match. Callers that don't care about
    the kind (only "is this a Scribe GUID at all?") can pass any of
    the recognised kind tags and get a yes/no answer in one call.
    """
    if not isinstance(guid, str) or not guid:
        return None
    m = _SCRIBE_GUID_RE.match(guid.lower().strip())
    if not m:
        return None
    if m.group(1) != kind_tag.lower():
        return None
    return m.group(2)


def _scribe_guid_kind(guid: str | None) -> str | None:
    """Return the kind tag of a Scribe-padded GUID, or ``None``."""
    if not isinstance(guid, str) or not guid:
        return None
    m = _SCRIBE_GUID_RE.match(guid.lower().strip())
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Plain-text tokenisation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenisedWord:
    """One non-whitespace word in a tokenised source.

    ``start`` is inclusive, ``end`` exclusive — same convention as
    :class:`scribe.refi_qda_project.WordOffset` and Python slicing.
    """

    word_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenisedSegment:
    """One segment (line) of a tokenised source.

    ``speaker`` is the detected speaker prefix (empty string when
    absent). ``words`` is the list of non-whitespace words in
    document-order.
    """

    speaker: str
    words: tuple[TokenisedWord, ...]


# Heuristic for the "SPEAKER: " prefix the F6.4 exporter emits. We
# accept up to four short whitespace-separated tokens before the
# colon (covers "INT:", "Luke Pearson:", "P3:", "INTERVIEWER 2:",
# "Coder A:"); anything longer is treated as ordinary text so we
# don't strip the first chunk of a long sentence containing a colon.
_SPEAKER_PREFIX_RE = re.compile(r"^([^:\n]{1,80}):\s+")
_NON_WS_RE = re.compile(r"\S+")


def _looks_like_speaker(candidate: str) -> bool:
    """Cheap filter: a speaker label is short and word-shaped."""
    s = candidate.strip()
    if not s or len(s) > 80:
        return False
    # No more than four whitespace-separated tokens. "Luke" / "Coder A"
    # / "Speaker 1" pass; "I went to the shop" rejects.
    if len(s.split()) > 4:
        return False
    # Must contain at least one letter so "::" / "1234" don't
    # qualify. (Numbers-only tags are unusual; if they exist, the user
    # can clean up post-import.)
    if not re.search(r"[A-Za-z]", s):
        return False
    return True


def tokenise_plain_text(
    text: str,
    *,
    detect_speakers: bool = True,
) -> list[TokenisedSegment]:
    """Tokenise a plain-text source into segments + words with offsets.

    Each ``\\n``-separated line becomes one segment. Within a line, a
    leading ``"SPEAKER: "`` prefix is detected (when
    ``detect_speakers``) and recorded on the segment; the remaining
    text is split on whitespace runs into words, each with its
    ``[start, end)`` char offset relative to the *whole* ``text``.

    The word ids follow the same ``s{seg_idx}w{word_idx}`` shape the
    F6.4 exporter (and the live ASR pipeline) produce, so an
    application built from these words round-trips through the rest
    of the codebase.

    A trailing newline at the end of ``text`` does not produce an
    empty trailing segment. An entirely-blank line in the middle
    *does* — it preserves alignment with the original char offsets.
    """
    if not isinstance(text, str):
        raise TypeError("tokenise_plain_text: text must be a string")

    segments: list[TokenisedSegment] = []
    cursor = 0
    lines = text.split("\n")
    # If the text ends in \n, ``split`` produces a trailing empty
    # string — drop it so we don't emit a phantom blank segment past
    # the final newline.
    if lines and lines[-1] == "" and text.endswith("\n"):
        lines = lines[:-1]

    for seg_idx, line in enumerate(lines):
        offset_in_line = 0
        speaker = ""

        if detect_speakers:
            m = _SPEAKER_PREFIX_RE.match(line)
            if m and _looks_like_speaker(m.group(1)):
                speaker = m.group(1).strip()
                offset_in_line = m.end()

        words: list[TokenisedWord] = []
        # Find each non-whitespace run after the (optional) speaker
        # prefix. Char offsets are computed against ``text``, so
        # ``cursor + offset_in_line + match_start`` is the absolute
        # start position.
        for w_idx, wm in enumerate(_NON_WS_RE.finditer(line, offset_in_line)):
            start = cursor + wm.start()
            end = cursor + wm.end()
            words.append(
                TokenisedWord(
                    word_id=f"s{seg_idx}w{w_idx}",
                    text=wm.group(0),
                    start=start,
                    end=end,
                )
            )

        segments.append(TokenisedSegment(speaker=speaker, words=tuple(words)))
        # +1 for the consumed ``\n`` (except after the last line where
        # there's no \n in the original text — but split() already
        # accounted for that, so adding 1 here is harmless when we're
        # past the end).
        cursor += len(line) + 1

    return segments


def char_span_to_word_anchors(
    segments: Sequence[TokenisedSegment],
    start: int,
    end: int,
) -> tuple[str, str, int | None, int | None] | None:
    """Locate the word ids that bracket the char span ``[start, end)``.

    Returns ``(start_word_id, end_word_id, start_offset, end_offset)``
    where the offsets are sub-word char offsets into the start/end
    word respectively (``None`` when the span begins/ends exactly on
    a word boundary, which is the common case after the F6.4
    exporter's whitespace-joined rendering).

    Returns ``None`` when no word lies inside the span — e.g. when
    the selection covers only whitespace, or when the span lies
    outside every word's char range. Callers should treat that as
    "selection couldn't be re-anchored; warn and skip".

    Boundary semantics:
      * ``start`` is interpreted as the first character of the
        selection; we pick the word that *contains* ``start`` in its
        ``[w.start, w.end)`` half-open interval, falling back to the
        first word that starts at or after ``start`` when ``start``
        lands in inter-word whitespace.
      * ``end`` is interpreted as one-past the last character; we
        pick the word that contains ``end - 1``, falling back to the
        last word that ends at or before ``end``.
    """
    if start > end:
        start, end = end, start

    # Flatten words across all segments. Cheap, the whole tree is
    # small; doing this once per call is fine.
    flat: list[TokenisedWord] = []
    for seg in segments:
        flat.extend(seg.words)
    if not flat:
        return None

    # ----- Find start word -------------------------------------------------
    start_word: TokenisedWord | None = None
    start_offset: int | None = None
    for w in flat:
        if w.start <= start < w.end:
            # Inside this word
            start_word = w
            so = start - w.start
            start_offset = so if so > 0 else None
            break
        if w.start >= start:
            # First word at or past start (start lands in whitespace)
            start_word = w
            start_offset = None
            break
    if start_word is None:
        # ``start`` is past every word; nothing to anchor
        return None

    # ----- Find end word ---------------------------------------------------
    end_target = max(start, end - 1)
    end_word: TokenisedWord | None = None
    end_offset: int | None = None
    for w in flat:
        if w.start <= end_target < w.end:
            end_word = w
            eo = end - w.start
            # Offset only matters if it's strictly inside the word;
            # at-or-past-the-end means "to the end of this word".
            if eo < (w.end - w.start):
                end_offset = eo
            else:
                end_offset = None
            break
        if w.end <= end:
            # Track the last word ending within / at the boundary;
            # we'll keep updating until we find one strictly inside
            # or run out.
            end_word = w
            end_offset = None
            continue
        if w.start >= end:
            # We've walked past; stop.
            break
    if end_word is None:
        # Span ends before the first word; treat as empty — no anchor.
        return None

    # Maintain start ≤ end on word ids. If the search produced a swap
    # (rare; only on degenerate inputs), nudge end up to start.
    from .applications import compare_word_ids
    if compare_word_ids(start_word.word_id, end_word.word_id) > 0:
        end_word = start_word
        end_offset = None

    return start_word.word_id, end_word.word_id, start_offset, end_offset


# --------------------------------------------------------------------------- #
# Code description structured-parse
# --------------------------------------------------------------------------- #

# Section markers emitted by ``refi_qda_project._code_description``.
# Order matters because we walk the description top-to-bottom and
# split on the first occurrence of each marker. Anything before the
# first known marker becomes ``definition`` (free text).
_CODE_DESCRIPTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("definition", "Definition: "),
    ("inclusion_criteria", "Inclusion criteria: "),
    ("exclusion_criteria", "Exclusion criteria: "),
    ("exemplars", "Exemplars:"),
    ("related_codes", "Related codes:"),
    ("theoretical_memo", "Theoretical memo: "),
    ("provenance", "Provenance: "),
    ("stage", "Stage: "),
    ("status", "Status: "),
)


def parse_code_description(text: str) -> dict[str, Any]:
    """Best-effort structured parse of a Code's ``<Description>`` body.

    The F6.4 exporter writes a labelled, double-newline-separated
    block ("Definition: …", "Inclusion criteria: …", "Exemplars:\\n-
    …", …). We invert that here so a Scribe-origin round-trip
    preserves all six free-text fields plus exemplars, related codes,
    provenance, stage, and status.

    For foreign QDPX files whose Description doesn't follow the
    Scribe shape, the whole text is dumped into ``definition`` and
    other fields stay empty — the safe degradation path.

    Returns a dict whose keys are a subset of the Scribe :class:`Code`
    field names. Unknown / missing fields are simply absent (not
    populated with empty values) so the caller can selectively
    update the Code constructor.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    # Split on double newline; each chunk *should* start with a known
    # marker. Anything that doesn't is appended to the previous
    # chunk's free-text tail (so a multi-paragraph Definition still
    # parses cleanly).
    chunks = re.split(r"\n\s*\n", text.strip())
    # If no marker is recognised in any chunk, treat the whole thing
    # as the definition body. That's the foreign-QDPX path.
    found_any = any(
        any(c.startswith(prefix) for _, prefix in _CODE_DESCRIPTION_MARKERS)
        for c in chunks
    )
    if not found_any:
        return {"definition": text.strip()}

    out: dict[str, Any] = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        matched = False
        for field_name, prefix in _CODE_DESCRIPTION_MARKERS:
            if chunk.startswith(prefix):
                body = chunk[len(prefix):].strip()
                if field_name == "exemplars":
                    out["exemplars"] = _parse_bulleted_lines(body)
                elif field_name == "related_codes":
                    out["related_codes"] = _parse_related_codes(body)
                elif field_name == "provenance":
                    out["provenance"] = _parse_provenance_kv(body)
                else:
                    out[field_name] = body
                matched = True
                break
        if not matched:
            # Free text not under any marker — append to definition.
            existing = out.get("definition", "")
            out["definition"] = (existing + "\n\n" + chunk).strip() if existing else chunk
    return out


def _parse_bulleted_lines(body: str) -> list[str]:
    """Parse "- foo\\n- bar" into ["foo", "bar"]."""
    lines = []
    for raw in body.splitlines():
        s = raw.strip()
        if s.startswith("- "):
            lines.append(s[2:].strip())
        elif s.startswith("-"):
            lines.append(s[1:].strip())
        elif s:
            lines.append(s)
    return [ln for ln in lines if ln]


def _parse_related_codes(body: str) -> list[dict[str, str]]:
    """Parse "- relation_type: code_id" lines into CodeRelation dicts.

    Lines that don't match are dropped silently — the export emits a
    closed-vocabulary form so anything else is foreign and can't be
    mapped to a :class:`CodeRelation` reliably.
    """
    out: list[dict[str, str]] = []
    for raw in body.splitlines():
        s = raw.strip()
        if s.startswith("- "):
            s = s[2:].strip()
        elif s.startswith("-"):
            s = s[1:].strip()
        if not s:
            continue
        if ":" not in s:
            continue
        rel_type, _, code_id = s.partition(":")
        rel_type = rel_type.strip()
        code_id = code_id.strip()
        if rel_type and code_id:
            out.append({"relation_type": rel_type, "code_id": code_id})
    return out


def _parse_provenance_kv(body: str) -> dict[str, str]:
    """Parse "k1=v1; k2=v2" into a {k: v} dict.

    Drops malformed pairs (no '=') silently. Foreign-origin
    descriptions whose provenance line happens to have a different
    shape will yield an empty dict, which is the right safe default.
    """
    out: dict[str, str] = {}
    for chunk in body.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, _, v = chunk.partition("=")
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ImportedSourceText:
    """One source's plain text + tokenisation, returned to callers.

    Persistence layers can use this to write a transcript sidecar
    (e.g. ``imported/<source_id>.txt``) and a synthetic segments
    JSON that the editor can later open. F6.6 itself doesn't write
    to disk; callers compose with :mod:`scribe.sources` to do so.
    """

    source_id: str
    text: str
    segments: list[TokenisedSegment]


@dataclass
class ImportResult:
    """All entities recovered from a QDPX archive.

    The :class:`Project` is always present (even if the input was
    sparse — the import always produces a project with a fresh id and
    the name parsed from the QDE root). All other fields default to
    empty lists for the easy-checking code path.

    ``code_versions`` carries one synthetic snapshot per imported
    code so the matching applications have a valid
    ``definition_version_id_at_apply`` to point at. Callers persist
    the version log alongside the codes.

    ``source_texts`` is keyed by the new (post-import) source id so a
    persistence layer can write the plain-text body somewhere it can
    later be re-rendered for editor playback.

    ``warnings`` is a list of human-readable strings describing
    elements that were skipped or downgraded (foreign GUIDs without
    a matching kind, references to unknown codes, malformed
    selections). Surface these in the import-summary UI.
    """

    project: Project
    sources: list[Source] = field(default_factory=list)
    codes: list[Code] = field(default_factory=list)
    coders: list[Coder] = field(default_factory=list)
    memos: list[Memo] = field(default_factory=list)
    applications: list[Application] = field(default_factory=list)
    code_versions: list[CodeVersion] = field(default_factory=list)
    source_texts: dict[str, ImportedSourceText] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# XML helpers
# --------------------------------------------------------------------------- #


def _q(tag: str) -> str:
    """Namespaced tag for the QDE schema."""
    return f"{{{REFI_QDA_PROJECT_NS}}}{tag}"


def _findall_local(parent: ET.Element, name: str) -> list[ET.Element]:
    """Find direct children by local name, namespace-agnostic.

    QDE files in the wild sometimes drop the namespace (Atlas.ti's
    older exports) or prefix it differently. Walking by local-name
    keeps us flexible without preprocessing the XML.
    """
    out: list[ET.Element] = []
    for child in parent:
        # Strip ``{ns}`` wrapper if present
        local = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
        if local == name:
            out.append(child)
    return out


# --------------------------------------------------------------------------- #
# Top-level import builders
# --------------------------------------------------------------------------- #


def import_qde(
    qde_xml: str | bytes,
    source_texts: Mapping[str, str] | None = None,
    *,
    project_id: str | None = None,
    now: str | None = None,
) -> ImportResult:
    """Parse a ``project.qde`` body and return the recovered entities.

    ``source_texts`` maps a TextSource's ``plainTextPath`` (e.g.
    ``"internal://Sources/abcdef012345.txt"``) **or** the bare
    filename (``"abcdef012345.txt"``) to the file's plain-text body.
    Provide whichever shape your unzip layer produces; the importer
    accepts both. If a TextSource's plainTextPath isn't in the map,
    that source is still produced but with no tokenisation /
    selection placement (its applications are skipped + warned).

    ``project_id`` lets the caller pin the new project's id — handy
    when the calling layer already minted an id (so the projects
    directory is created up front). When omitted we mint a fresh id.

    ``now`` overrides the timestamp stamped on every freshly-built
    entity. Tests pass a fixed value for determinism; production
    callers leave it at ``None`` to use ``utcnow_iso()``.
    """
    if isinstance(qde_xml, bytes):
        try:
            qde_xml = qde_xml.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"QDE body is not UTF-8: {e}") from e
    if not isinstance(qde_xml, str):
        raise TypeError("qde_xml must be str or bytes")

    try:
        root = ET.fromstring(qde_xml)
    except ET.ParseError as e:
        raise ValueError(f"QDE body is not well-formed XML: {e}") from e

    src_text_map = _normalise_source_text_map(source_texts or {})
    ts = now or utcnow_iso()
    proj_id = project_id or new_project_id()

    warnings: list[str] = []

    # --- Project ----------------------------------------------------------
    project = _build_project(root, project_id=proj_id, now=ts)

    # --- Codes ------------------------------------------------------------
    codes, code_versions, code_id_by_guid = _build_codes(
        root, project_id=proj_id, now=ts, warnings=warnings
    )

    # --- Coders -----------------------------------------------------------
    coders, coder_id_by_guid = _build_coders(
        root, project_id=proj_id, now=ts, warnings=warnings
    )

    # --- Sources + applications ------------------------------------------
    # ``coders`` and ``coder_id_by_guid`` are passed mutably so the apps
    # builder can synthesise a fallback coder when an application's
    # creatingUser doesn't resolve to any imported User.
    sources, applications, source_texts_out = _build_sources_and_apps(
        root,
        project_id=proj_id,
        src_text_map=src_text_map,
        code_id_by_guid=code_id_by_guid,
        coder_id_by_guid=coder_id_by_guid,
        coders=coders,
        code_versions=code_versions,
        now=ts,
        warnings=warnings,
    )

    # --- Memos (notes) ----------------------------------------------------
    memos = _build_memos(root, project_id=proj_id, src_text_map=src_text_map, now=ts, warnings=warnings)

    return ImportResult(
        project=project,
        sources=sources,
        codes=codes,
        coders=coders,
        memos=memos,
        applications=applications,
        code_versions=code_versions,
        source_texts=source_texts_out,
        warnings=warnings,
    )


def import_qdpx(
    archive: bytes | Path,
    *,
    project_id: str | None = None,
    now: str | None = None,
) -> ImportResult:
    """Open a QDPX zip archive and import everything inside it.

    ``archive`` is either the raw zip bytes or a :class:`Path` to the
    file on disk. The archive is read into memory; QDPX files in
    practice are single-megabyte zips, well within budget.

    Sources / Notes plain-text bodies are pulled from the
    ``Sources/`` and ``Notes/`` folders inside the zip and passed to
    :func:`import_qde` keyed by their archive entry name (so an
    archive with ``Sources/abcdef.txt`` resolves a TextSource's
    ``plainTextPath="internal://Sources/abcdef.txt"`` cleanly).

    Raises :class:`ValueError` if the archive lacks a ``project.qde``
    member or that member is not parseable XML.
    """
    if isinstance(archive, Path):
        data = archive.read_bytes()
    elif isinstance(archive, (bytes, bytearray)):
        data = bytes(archive)
    else:
        raise TypeError("archive must be bytes or pathlib.Path")

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError(f"Not a valid zip archive: {e}") from e

    qde_member: str | None = None
    for name in zf.namelist():
        if name.lower().endswith("project.qde") and "/" not in name.strip("/"):
            qde_member = name
            break
        if name == "project.qde":
            qde_member = name
            break
    if qde_member is None:
        # Some tools nest under a folder; fall back to any member
        # named ``project.qde``.
        for name in zf.namelist():
            if name.split("/")[-1] == "project.qde":
                qde_member = name
                break
    if qde_member is None:
        raise ValueError("QDPX archive has no project.qde member")

    qde_bytes = zf.read(qde_member)

    src_texts: dict[str, str] = {}
    for name in zf.namelist():
        if "/" not in name:
            continue
        # Pick up Sources/... and Notes/... text bodies. Other
        # folders (Audio/, Video/) are ignored.
        parts = name.split("/")
        if parts[0] in ("Sources", "Notes") and len(parts) == 2:
            try:
                src_texts[name] = zf.read(name).decode("utf-8")
            except UnicodeDecodeError:
                # Skip non-text members silently; the warnings list
                # in the result will report it via the source-resolve
                # step.
                continue

    return import_qde(
        qde_bytes,
        source_texts=src_texts,
        project_id=project_id,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Per-entity builders
# --------------------------------------------------------------------------- #


def _normalise_source_text_map(raw: Mapping[str, str]) -> dict[str, str]:
    """Accept either ``"Sources/<id>.txt"`` or ``"internal://Sources/<id>.txt"``.

    The QDE schema uses the ``internal://`` prefix on
    ``plainTextPath``; callers passing zip member names like
    ``"Sources/<id>.txt"`` shouldn't have to know that. We index
    every entry under both shapes so the resolver doesn't have to
    branch.
    """
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out[k] = v
        if k.startswith("internal://"):
            stripped = k[len("internal://"):]
            out[stripped] = v
            # Bare filename too
            if "/" in stripped:
                out[stripped.split("/")[-1]] = v
        else:
            out[f"internal://{k}"] = v
            if "/" in k:
                out[k.split("/")[-1]] = v
    return out


def _build_project(root: ET.Element, *, project_id: str, now: str) -> Project:
    """Build a Scribe :class:`Project` from the QDE root element."""
    name = (root.get("name") or "Imported QDPX project").strip()
    if not name:
        name = "Imported QDPX project"
    created_at = root.get("creationDateTime") or now
    modified_at = root.get("modifiedDateTime") or created_at
    origin = (root.get("origin") or "").strip()
    description = f"Imported from QDPX (origin={origin or 'unknown'})"

    p = Project.new(
        name=name,
        description=description,
        project_id=project_id,
        now=now,
    )
    # Preserve the original timestamps when they validate; Project
    # validates that the values aren't required to be parseable, just
    # bounded strings.
    p.created_at = created_at
    p.modified_at = modified_at
    return p


def _build_codes(
    root: ET.Element,
    *,
    project_id: str,
    now: str,
    warnings: list[str],
) -> tuple[list[Code], list[CodeVersion], dict[str, str]]:
    """Build the codebook plus a synthetic version per code.

    Returns ``(codes, versions, code_id_by_guid)``. The map enables
    later ``CodeRef`` resolution from selection elements.
    """
    codes: list[Code] = []
    versions: list[CodeVersion] = []
    by_guid: dict[str, str] = {}

    cb_els = _findall_local(root, "CodeBook")
    if not cb_els:
        return codes, versions, by_guid

    # We only care about the (single) CodeBook element. If multiple are
    # present (malformed) we walk all of them.
    code_elements: list[tuple[ET.Element, str | None]] = []
    for cb in cb_els:
        codes_el_list = _findall_local(cb, "Codes")
        for codes_el in codes_el_list:
            for child in _findall_local(codes_el, "Code"):
                code_elements.append((child, None))

    # Walk the tree depth-first so children resolve their parent's id.
    def walk(el: ET.Element, parent_scribe_id: str | None) -> None:
        guid = (el.get("guid") or "").strip()
        recovered = parse_guid_to_scribe_id(guid, kind_tag=_KIND_TAG_CODE)
        scribe_id = recovered or new_code_id()
        if guid:
            by_guid[guid] = scribe_id

        name = (el.get("name") or "").strip() or "(unnamed code)"
        colour = (el.get("color") or "").strip()
        if colour and not CODE_COLOUR_RE.match(colour):
            colour = ""  # drop malformed; safer than failing the import

        # <Description> body
        desc_text = ""
        for d in _findall_local(el, "Description"):
            if d.text:
                desc_text = d.text
                break

        parsed = parse_code_description(desc_text)
        # Pull related_codes / exemplars / provenance / stage / status
        # out of the structured parse; everything else falls through
        # as a free-text definition / inclusion / exclusion / memo.
        related_dicts = parsed.pop("related_codes", []) or []
        exemplars = parsed.pop("exemplars", []) or []
        provenance = parsed.pop("provenance", {}) or {}

        try:
            c = Code.new(
                project_id=project_id,
                code_id=scribe_id,
                name=name,
                definition=parsed.get("definition", "") or "",
                inclusion_criteria=parsed.get("inclusion_criteria", "") or "",
                exclusion_criteria=parsed.get("exclusion_criteria", "") or "",
                exemplars=exemplars,
                parent_code_id=parent_scribe_id,
                related_codes=[],  # resolved in a second pass
                theoretical_memo=parsed.get("theoretical_memo", "") or "",
                stage=parsed.get("stage") or "initial",
                colour=colour,
                status=parsed.get("status") or "active",
                provenance={
                    **provenance,
                    "imported_from": "refi-qda",
                },
                now=now,
            )
        except Exception as e:
            warnings.append(f"code {guid!r} skipped: {e}")
            return

        codes.append(c)

        # Stash the raw related-code list on the side so a second pass
        # can resolve cross-references after the GUID map is populated.
        c.__dict__["_pending_relations"] = related_dicts

        # Synthetic version=1 snapshot so applications have a valid
        # definition_version_id_at_apply. ``CodeVersion.new`` builds
        # the snapshot from the live code, which is exactly what we
        # want.
        v = CodeVersion.new(code=c, version=1, change_note="imported from QDPX", now=now)
        versions.append(v)

        for child in _findall_local(el, "Code"):
            walk(child, scribe_id)

    for el, _parent in code_elements:
        walk(el, None)

    # Second pass: resolve related_codes references against the
    # GUID-or-id map. Foreign relations whose target can't be resolved
    # — or whose relation_type lies outside Scribe's closed vocabulary —
    # are dropped with a warning.
    code_ids = {c.id for c in codes}
    for c in codes:
        pending = c.__dict__.pop("_pending_relations", []) or []
        valid: list[CodeRelation] = []
        for entry in pending:
            rel = entry.get("relation_type", "").strip()
            target_raw = entry.get("code_id", "").strip()
            if not rel or not target_raw:
                continue
            # The exporter stores the target as the *Scribe* code id
            # (12-char hex), not a GUID. For round-trip files that
            # matches our recovered code ids directly. For foreign-
            # origin descriptions we still try the GUID map and the
            # Scribe-padded GUID as fallbacks.
            if re.match(r"^[0-9a-f]{12}$", target_raw):
                target_id: str | None = target_raw
            else:
                target_id = by_guid.get(target_raw)
                if target_id is None:
                    target_id = parse_guid_to_scribe_id(
                        target_raw, kind_tag=_KIND_TAG_CODE
                    )
            if not target_id or target_id not in code_ids:
                warnings.append(
                    f"code {c.id} relation {rel!r}: target {target_raw!r} not found"
                )
                continue
            r = CodeRelation(code_id=target_id, relation_type=rel)
            try:
                r.validate()
            except Exception as e:
                warnings.append(f"code {c.id} relation {rel!r} dropped: {e}")
                continue
            valid.append(r)
        if valid:
            c.related_codes = valid
            try:
                c.validate()
            except Exception as e:  # pragma: no cover — defensive
                warnings.append(f"code {c.id} relations rolled back: {e}")
                c.related_codes = []

    return codes, versions, by_guid


def _build_coders(
    root: ET.Element,
    *,
    project_id: str,
    now: str,
    warnings: list[str],
) -> tuple[list[Coder], dict[str, str]]:
    """Build :class:`Coder` entities from ``<Users>``."""
    coders: list[Coder] = []
    by_guid: dict[str, str] = {}

    for users_el in _findall_local(root, "Users"):
        for u in _findall_local(users_el, "User"):
            guid = (u.get("guid") or "").strip()
            kind = _scribe_guid_kind(guid)
            # The exporter creates a placeholder User with the
            # project's GUID kind tag (9301) when there were no real
            # coders. Skip importing that one — it's not a real coder.
            if kind == _KIND_TAG_PROJECT:
                continue

            recovered = parse_guid_to_scribe_id(guid, kind_tag=_KIND_TAG_USER)
            scribe_id = recovered or new_coder_id()
            if guid:
                by_guid[guid] = scribe_id

            name = (u.get("name") or u.get("id") or scribe_id).strip()
            if not name:
                name = scribe_id

            try:
                c = Coder.new(
                    project_id=project_id,
                    coder_id=scribe_id,
                    name=name,
                    now=now,
                )
            except Exception as e:
                warnings.append(f"user {guid!r} skipped: {e}")
                continue
            coders.append(c)

    return coders, by_guid


def _build_sources_and_apps(
    root: ET.Element,
    *,
    project_id: str,
    src_text_map: Mapping[str, str],
    code_id_by_guid: Mapping[str, str],
    coder_id_by_guid: dict[str, str],
    coders: list[Coder],
    code_versions: Sequence[CodeVersion],
    now: str,
    warnings: list[str],
) -> tuple[list[Source], list[Application], dict[str, ImportedSourceText]]:
    """Build sources + their applications. Audio/video sources are skipped."""
    sources: list[Source] = []
    applications: list[Application] = []
    source_texts: dict[str, ImportedSourceText] = {}

    # code_id → its v1 CodeVersion id, for definition_version_id_at_apply
    version_by_code: dict[str, str] = {v.code_id: v.id for v in code_versions}

    # Synthesised fallback coder. Created lazily, exactly once per
    # import call, and added to ``coders`` so applications referencing
    # it have a real entity to point at. Local closure ⇒ no module
    # state to leak across imports.
    fallback_coder_id: str | None = None

    def _ensure_fallback_coder() -> str:
        nonlocal fallback_coder_id
        if fallback_coder_id is not None:
            return fallback_coder_id
        cid = new_coder_id()
        try:
            coders.append(
                Coder.new(
                    project_id=project_id,
                    coder_id=cid,
                    name="Imported coder",
                    role="other",
                    notes="Synthesised by F6.6 importer for applications "
                          "whose creatingUser GUID couldn't be resolved.",
                    now=now,
                )
            )
        except Exception as e:
            # Should never happen — the inputs we pass always validate
            # — but if it did, surface a warning so the import doesn't
            # silently produce dangling refs.
            warnings.append(f"could not synthesise fallback coder: {e}")
            raise
        fallback_coder_id = cid
        return cid

    sources_els = _findall_local(root, "Sources")
    if not sources_els:
        return sources, applications, source_texts

    # We accept the QDE-defined element names plus a couple of
    # case-insensitive variants in case a foreign exporter capitalises
    # differently.
    text_source_local_names = {"TextSource"}
    other_source_local_names = {"AudioSource", "VideoSource", "PictureSource", "PDFSource"}

    for sources_el in sources_els:
        for child in list(sources_el):
            local = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
            if local in other_source_local_names:
                warnings.append(
                    f"non-text source {child.get('guid', '?')!r} ({local}) skipped"
                )
                continue
            if local not in text_source_local_names:
                continue

            guid = (child.get("guid") or "").strip()
            recovered = parse_guid_to_scribe_id(guid, kind_tag=_KIND_TAG_SOURCE)
            scribe_id = recovered or new_source_id()

            name = (child.get("name") or "").strip() or "(unnamed source)"
            plain_path = (child.get("plainTextPath") or "").strip()
            text = src_text_map.get(plain_path, "")
            if not text:
                # Try bare filename as a fallback (zip member name w/o
                # the internal:// prefix)
                if plain_path.startswith("internal://"):
                    text = src_text_map.get(plain_path[len("internal://"):], "")

            try:
                s = Source.new(
                    project_id=project_id,
                    name=name,
                    source_id=scribe_id,
                    source_type="transcript",
                    transcript_job_id=None,
                    now=now,
                )
            except Exception as e:
                warnings.append(f"source {guid!r} skipped: {e}")
                continue
            sources.append(s)

            # Tokenise and stash for the caller's persistence layer.
            tok_segments = tokenise_plain_text(text) if text else []
            source_texts[scribe_id] = ImportedSourceText(
                source_id=scribe_id,
                text=text,
                segments=tok_segments,
            )

            # Walk PlainTextSelection / Coding pairs.
            for sel in _findall_local(child, "PlainTextSelection"):
                if not text:
                    warnings.append(
                        f"selection on source {scribe_id} skipped: no plain text"
                    )
                    continue
                try:
                    start = int(sel.get("startPosition") or "0")
                    end = int(sel.get("endPosition") or "0")
                except (TypeError, ValueError):
                    warnings.append(
                        f"selection {sel.get('guid', '?')} on source {scribe_id} "
                        "has malformed startPosition/endPosition"
                    )
                    continue
                anchors = char_span_to_word_anchors(tok_segments, start, end)
                if anchors is None:
                    warnings.append(
                        f"selection {sel.get('guid', '?')} on source {scribe_id} "
                        f"covers no words ({start}-{end})"
                    )
                    continue
                start_wid, end_wid, start_off, end_off = anchors

                creating_user = (sel.get("creatingUser") or "").strip()
                created_at = (sel.get("creationDateTime") or now).strip() or now

                # Each Coding child is one Application
                for cod in _findall_local(sel, "Coding"):
                    cref_els = _findall_local(cod, "CodeRef")
                    for cref in cref_els:
                        target_guid = (cref.get("targetGUID") or "").strip()
                        code_id = code_id_by_guid.get(target_guid)
                        if code_id is None:
                            # Try to recover Scribe-padded GUID directly
                            code_id = parse_guid_to_scribe_id(
                                target_guid, kind_tag=_KIND_TAG_CODE
                            )
                        if code_id is None:
                            warnings.append(
                                f"application skipped: unknown code GUID {target_guid!r}"
                            )
                            continue

                        version_id = version_by_code.get(code_id)
                        if version_id is None:
                            warnings.append(
                                f"application skipped: no version for code {code_id}"
                            )
                            continue

                        coder_id = coder_id_by_guid.get(creating_user)
                        if coder_id is None:
                            # Best-effort recovery: if creatingUser is
                            # Scribe-padded with a User kind tag, recover.
                            recovered = parse_guid_to_scribe_id(
                                creating_user, kind_tag=_KIND_TAG_USER
                            )
                            if recovered is not None and recovered in {c.id for c in coders}:
                                coder_id = recovered
                        if coder_id is None:
                            # No coder anywhere — synthesise one and
                            # add it to the coders list so the
                            # application's coder_id points at a real
                            # entity.
                            coder_id = _ensure_fallback_coder()
                            if creating_user:
                                # Cache the mapping so the next app
                                # with the same unknown creator
                                # resolves without another lookup.
                                coder_id_by_guid[creating_user] = coder_id

                        # Application id from Selection guid (5e1c).
                        app_recovered = parse_guid_to_scribe_id(
                            (sel.get("guid") or "").strip(),
                            kind_tag=_KIND_TAG_SELECTION,
                        )
                        app_id = app_recovered or new_application_id()
                        # If multiple Codings per Selection, second
                        # and later need fresh ids to avoid collision.
                        if any(a.id == app_id for a in applications):
                            app_id = new_application_id()

                        try:
                            a = Application.new(
                                project_id=project_id,
                                application_id=app_id,
                                code_id=code_id,
                                source_id=scribe_id,
                                coder_id=coder_id,
                                anchor_start_word_id=start_wid,
                                anchor_end_word_id=end_wid,
                                start_char_offset=start_off,
                                end_char_offset=end_off,
                                definition_version_id_at_apply=version_id,
                                provenance={"source": "imported"},
                                now=created_at,
                            )
                        except Exception as e:
                            warnings.append(
                                f"application on selection {sel.get('guid', '?')} "
                                f"failed: {e}"
                            )
                            continue
                        applications.append(a)

    return sources, applications, source_texts


def _build_memos(
    root: ET.Element,
    *,
    project_id: str,
    src_text_map: Mapping[str, str],
    now: str,
    warnings: list[str],
) -> list[Memo]:
    """Build :class:`Memo` entities from ``<Notes>``."""
    memos: list[Memo] = []
    for notes_el in _findall_local(root, "Notes"):
        for n in _findall_local(notes_el, "Note"):
            guid = (n.get("guid") or "").strip()
            recovered = parse_guid_to_scribe_id(guid, kind_tag=_KIND_TAG_NOTE)
            scribe_id = recovered or new_memo_id()
            title = (n.get("name") or "").strip()
            plain_path = (n.get("plainTextPath") or "").strip()
            body = src_text_map.get(plain_path, "")
            if not body and plain_path.startswith("internal://"):
                body = src_text_map.get(plain_path[len("internal://"):], "")

            try:
                m = Memo.new(
                    project_id=project_id,
                    memo_id=scribe_id,
                    type="free",
                    title=title,
                    body=body,
                    body_format="plain",
                    provenance={"source": "imported"},
                    now=now,
                )
            except Exception as e:
                warnings.append(f"note {guid!r} skipped: {e}")
                continue
            memos.append(m)
    return memos


# --------------------------------------------------------------------------- #
# Public symbols
# --------------------------------------------------------------------------- #

__all__ = [
    "ImportResult",
    "ImportedSourceText",
    "TokenisedSegment",
    "TokenisedWord",
    "char_span_to_word_anchors",
    "import_qde",
    "import_qdpx",
    "parse_code_description",
    "parse_guid_to_scribe_id",
    "tokenise_plain_text",
]
