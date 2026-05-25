"""Embedding index over coded segments and uncoded paragraphs (F8.2).

Per PLANNING.md F8.2:

  > Embedding index of every coded segment + every uncoded paragraph.
  > Built on import; refreshed on edit.

Why this exists
---------------

The Phase-C AI features all sit on top of an *embedding index*: F8.3
("suggest codes from existing codebook"), F8.5 ("find similar quotes"),
F8.6 (whole-transcript review pass), and F8.8 (memo drafts) all need a
nearest-neighbour search over the project's text. Building that index
incrementally, in a way that survives transcript edits and code
re-applications, is its own little design exercise — F8.2 is the
plumbing that exercise produces. Nothing in here ships an actual model
load; embedding values come from a caller-supplied callable so the
F8.1 backend can drive the bus and tests can stub it.

What gets indexed
-----------------

Two kinds of entries, matching the literal spec:

* ``coded_segment`` — one entry per :class:`scribe.applications.Application`.
  The text is the words covered by the application's anchor (sub-word
  character offsets honoured; matches the retrieval-report extraction).
* ``uncoded_paragraph`` — one entry per paragraph (speaker turn) that
  has **no application** touching any of its words. Paragraphs follow
  the same definition F4.4's selection-snap helpers use: maximal runs
  of consecutive segments sharing a non-None ``speaker``; a missing /
  None speaker breaks the run into singletons.

A paragraph that has even one applied code drops out of the
``uncoded_paragraph`` set entirely. Its coded sub-spans are still
present as ``coded_segment`` entries; the uncoded-vs-coded split is
analytic, not exhaustive coverage. (This is the literal F8.2 reading;
later features can refine it.)

Identity & deduplication
------------------------

Each desired entry has a **natural key** uniquely identifying its
slot in the index:

* coded_segment: ``(kind, source_id, application_id)``
* uncoded_paragraph: ``(kind, source_id, paragraph_start_seg,
  paragraph_end_seg)``

The on-disk filename is a deterministic 12-char hex hash of that
natural key — so refresh can locate, update, or drop an entry without
loading every file first, and rebuilding the index from scratch
produces byte-identical filenames given the same inputs.

Refresh diff
------------

:func:`refresh_embedding_index` is the workhorse. Given the current
applications + segments, it computes the desired set of spans, diffs
against existing on-disk entries, and:

* embeds new spans;
* re-embeds spans whose text changed (different ``text_hash``) or
  whose model changed (different ``model_name``);
* leaves untouched spans that already match;
* deletes existing entries whose natural key is no longer desired.

The embed callable is invoked once per chunk of ``batch_size`` items
(default 32) so backends with HTTP or RPC overhead aren't
death-by-a-thousand-calls.

Stand-alone module: no FastAPI, no engine imports. Conventions
match the rest of the F-feature stack
(:mod:`scribe.applications`, :mod:`scribe.application_spans`,
:mod:`scribe.application_reanchor`, :mod:`scribe.retrieval_report`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .application_reanchor import anchored_words, collect_word_texts
from .applications import (
    APPLICATION_ID_RE,
    Application,
    parse_word_id,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .selection_snap import paragraph_ranges
from .sources import SOURCE_ID_RE


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Two kinds of indexed entries: a coded segment (the anchored text of
# one Application) and an uncoded paragraph (a paragraph with no
# applications touching it). Closed set — anything else surfaces as a
# validation error rather than a silent drop.
EMBEDDING_KIND_CODED_SEGMENT = "coded_segment"
EMBEDDING_KIND_UNCODED_PARAGRAPH = "uncoded_paragraph"
EMBEDDING_KINDS: tuple[str, ...] = (
    EMBEDDING_KIND_CODED_SEGMENT,
    EMBEDDING_KIND_UNCODED_PARAGRAPH,
)

# Embedding entry IDs follow the same 12-char hex shape as every other
# id in Scribe; in practice they are derived deterministically from the
# entry's natural key.
EMBEDDING_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding the index.
# Mirrors how applications / sources / participants are organised.
EMBEDDINGS_DIRNAME = "embeddings"

# Field length / cardinality limits. Generous, but bounded so a stray
# upstream bug can't write a 50 MB embedding entry.
MAX_TEXT_PREVIEW_LEN = 500
MAX_MODEL_NAME_LEN = 256
MAX_VECTOR_DIM = 8192          # bge-m3 is 1024; LLM-style is 4096; 8192 is generous
MIN_VECTOR_DIM = 1
DEFAULT_BATCH_SIZE = 32

# A SHA-256 hex digest is 64 chars; we use the canonical full-length
# string so future bumps in algorithm don't collide with old entries.
TEXT_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


# --------------------------------------------------------------------------- #
# Indexable span (target shape)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexableSpan:
    """A single slot in the desired index, with its text resolved.

    ``text`` is the canonicalised span text (whitespace collapsed,
    trimmed). The natural :meth:`key` is used both for diffing against
    existing entries and for deriving the deterministic on-disk
    filename via :func:`entry_id_for_key`.
    """

    kind: str
    source_id: str
    application_id: str | None
    paragraph_start_segment: int | None
    paragraph_end_segment: int | None
    anchor_start_word_id: str
    anchor_end_word_id: str
    text: str

    def key(self) -> tuple[str, ...]:
        if self.kind == EMBEDDING_KIND_CODED_SEGMENT:
            return (self.kind, self.source_id, self.application_id or "")
        return (
            self.kind,
            self.source_id,
            str(self.paragraph_start_segment),
            str(self.paragraph_end_segment),
        )


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` (UTF-8).

    Used to detect "needs re-embedding" cheaply: two refreshes that see
    the same input text on the same model leave the entry untouched.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entry_id_for_key(key: Sequence[str]) -> str:
    """Return the deterministic 12-char hex id for a natural key.

    The entry id is the first 12 hex chars of SHA-256 over a tab-joined
    representation of the key. Tab is chosen as the separator because
    it cannot appear inside source / application ids (12-char hex) or
    inside the canonicalised kind strings.
    """
    joined = "\t".join(str(part) for part in key)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Text canonicalisation + extraction
# --------------------------------------------------------------------------- #


_WS_RUN_RE = re.compile(r"\s+", flags=re.UNICODE)


def canonical_text(text: str) -> str:
    """Collapse runs of whitespace to single spaces and trim ends.

    Embedding models are insensitive to leading/trailing space and to
    runs of whitespace, but we still want byte-stable hashes — two
    refreshes from a re-saved transcript file (which may swap
    ``\\n`` for ``\\n\\n``) should not trigger an unnecessary re-embed.
    """
    if not isinstance(text, str):
        return ""
    s = _WS_RUN_RE.sub(" ", text)
    return s.strip()


def extract_application_text(
    application: Application,
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """Return the canonicalised quoted text for an application.

    Mirrors :func:`scribe.retrieval_report._extract_text` so the
    embedding's input is exactly what a researcher sees in the
    retrieval report. Returns ``""`` if the anchor falls outside the
    transcript — the caller will skip it (the orphan queue is the
    right place to surface those, not an embedder).
    """
    words = anchored_words(application, segments)
    if not words:
        return ""
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
    words = [w for w in words if w]
    return canonical_text(" ".join(words))


def extract_paragraph_text(
    segments: Sequence[Mapping[str, Any]],
    start_seg: int,
    end_seg: int,
) -> str:
    """Return the canonicalised text covering segments[start..end] (inclusive).

    Used for ``uncoded_paragraph`` entries. We join every word in
    those segments with single spaces and canonicalise.
    """
    if start_seg < 0 or end_seg < start_seg or end_seg >= len(segments):
        return ""
    words_2d = collect_word_texts(segments)
    pieces: list[str] = []
    for si in range(start_seg, end_seg + 1):
        for w in words_2d[si]:
            if w:
                pieces.append(w)
    return canonical_text(" ".join(pieces))


# --------------------------------------------------------------------------- #
# Desired-spans enumeration
# --------------------------------------------------------------------------- #


def _segments_touched_by_application(
    application: Application,
    segments: Sequence[Mapping[str, Any]],
) -> set[int]:
    """Return the set of segment indices the application's anchor spans.

    Word ids beyond the transcript silently shrink the set rather than
    raise — same forgiving stance as :func:`anchored_words`. Different
    source ids than the segments belong to are caller-checked outside
    this helper.
    """
    sa_seg, _ = parse_word_id(application.anchor_start_word_id)
    ea_seg, _ = parse_word_id(application.anchor_end_word_id)
    if sa_seg < 0:
        sa_seg = 0
    if ea_seg >= len(segments):
        ea_seg = len(segments) - 1
    if ea_seg < sa_seg:
        return set()
    return set(range(sa_seg, ea_seg + 1))


def desired_index_spans(
    *,
    applications: Sequence[Application],
    segments_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[IndexableSpan]:
    """Compute the full set of :class:`IndexableSpan` for the project.

    Order of the returned list:

    1. ``coded_segment`` entries, in application creation order
       (mirrors :func:`scribe.applications.list_applications` default).
    2. ``uncoded_paragraph`` entries, by ``(source_id,
       paragraph_start_segment)`` so the natural traversal is
       deterministic.

    Spans whose text resolves to empty (anchor out of range, paragraph
    has no words) are dropped — embedding the empty string is wasted
    work and tends to surface as backend validation errors.
    """
    spans: list[IndexableSpan] = []

    # ---- coded_segment entries ------------------------------------------
    apps_by_source: dict[str, list[Application]] = {}
    for a in applications:
        text = extract_application_text(
            a, segments_by_source.get(a.source_id, ())
        )
        if not text:
            continue
        spans.append(
            IndexableSpan(
                kind=EMBEDDING_KIND_CODED_SEGMENT,
                source_id=a.source_id,
                application_id=a.id,
                paragraph_start_segment=None,
                paragraph_end_segment=None,
                anchor_start_word_id=a.anchor_start_word_id,
                anchor_end_word_id=a.anchor_end_word_id,
                text=text,
            )
        )
        apps_by_source.setdefault(a.source_id, []).append(a)

    # ---- uncoded_paragraph entries --------------------------------------
    # For each source we know about, partition into paragraphs and emit
    # entries for paragraphs with zero applications touching them.
    paragraph_entries: list[IndexableSpan] = []
    for source_id, segments in segments_by_source.items():
        if not segments:
            continue
        if not isinstance(source_id, str) or not SOURCE_ID_RE.match(source_id):
            # Forgiving: tolerate odd keys but skip them rather than
            # blowing up the whole refresh.
            continue
        # Compute the union of segment indices touched by any
        # application on this source.
        touched: set[int] = set()
        for a in apps_by_source.get(source_id, []):
            touched.update(
                _segments_touched_by_application(a, segments)
            )
        for p_start, p_end in paragraph_ranges(segments):
            # Skip paragraphs that any application touches.
            if any(i in touched for i in range(p_start, p_end + 1)):
                continue
            text = extract_paragraph_text(segments, p_start, p_end)
            if not text:
                continue
            # Anchor the paragraph at first/last word of the range.
            words_2d = collect_word_texts(segments)
            # Find first segment in [p_start, p_end] with at least one
            # word, and the last one similarly. Empty ranges already
            # rejected above (text would be empty).
            first_seg_with_words = None
            last_seg_with_words = None
            for si in range(p_start, p_end + 1):
                if words_2d[si]:
                    if first_seg_with_words is None:
                        first_seg_with_words = si
                    last_seg_with_words = si
            if first_seg_with_words is None or last_seg_with_words is None:
                continue
            anchor_start = f"s{first_seg_with_words}w0"
            last_word_idx = len(words_2d[last_seg_with_words]) - 1
            anchor_end = f"s{last_seg_with_words}w{last_word_idx}"
            paragraph_entries.append(
                IndexableSpan(
                    kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
                    source_id=source_id,
                    application_id=None,
                    paragraph_start_segment=p_start,
                    paragraph_end_segment=p_end,
                    anchor_start_word_id=anchor_start,
                    anchor_end_word_id=anchor_end,
                    text=text,
                )
            )
    paragraph_entries.sort(
        key=lambda s: (s.source_id, s.paragraph_start_segment or 0)
    )
    spans.extend(paragraph_entries)
    return spans


# --------------------------------------------------------------------------- #
# Entry data model
# --------------------------------------------------------------------------- #


@dataclass
class EmbeddingEntry:
    """One persisted entry in the embedding index.

    All cross-entity references (``project_id``, ``source_id``,
    ``application_id``) are validated for *shape* only — F8.2 is the
    storage layer; verifying that the referenced application still
    exists is the refresh layer's job (which simply drops orphans).
    """

    id: str
    project_id: str
    source_id: str
    kind: str
    application_id: str | None
    paragraph_start_segment: int | None
    paragraph_end_segment: int | None
    anchor_start_word_id: str
    anchor_end_word_id: str
    text_preview: str
    text_hash: str
    vector: tuple[float, ...]
    model_name: str
    dim: int
    created_at: str
    modified_at: str

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        span: IndexableSpan,
        vector: Sequence[float],
        model_name: str,
        text_for_hash: str | None = None,
        now: str | None = None,
    ) -> "EmbeddingEntry":
        """Build a fresh EmbeddingEntry from a span + a freshly-computed vector."""
        ts = now or utcnow_iso()
        text = text_for_hash if text_for_hash is not None else span.text
        e = cls(
            id=entry_id_for_key(span.key()),
            project_id=project_id,
            source_id=span.source_id,
            kind=span.kind,
            application_id=span.application_id,
            paragraph_start_segment=span.paragraph_start_segment,
            paragraph_end_segment=span.paragraph_end_segment,
            anchor_start_word_id=span.anchor_start_word_id,
            anchor_end_word_id=span.anchor_end_word_id,
            text_preview=text[:MAX_TEXT_PREVIEW_LEN],
            text_hash=text_hash(text),
            vector=tuple(float(x) for x in vector),
            model_name=str(model_name),
            dim=len(tuple(vector)),
            created_at=ts,
            modified_at=ts,
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Tuples become lists in JSON; do that explicitly so on-disk
        # files are easy to inspect.
        d["vector"] = list(self.vector)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EmbeddingEntry":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "EmbeddingEntry payload must be an object"
            )
        for required in (
            "id",
            "project_id",
            "source_id",
            "kind",
            "anchor_start_word_id",
            "anchor_end_word_id",
            "text_hash",
            "vector",
            "model_name",
            "dim",
        ):
            if required not in d:
                raise ProjectValidationError(
                    f"EmbeddingEntry payload missing required key: {required}"
                )
        vec = d.get("vector") or []
        if not isinstance(vec, (list, tuple)):
            raise ProjectValidationError(
                "EmbeddingEntry.vector must be a list of numbers"
            )
        try:
            vector = tuple(float(x) for x in vec)
        except (TypeError, ValueError) as e:
            raise ProjectValidationError(
                "EmbeddingEntry.vector contains non-numeric values"
            ) from e
        ps = d.get("paragraph_start_segment")
        pe = d.get("paragraph_end_segment")
        e = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            source_id=str(d["source_id"]),
            kind=str(d["kind"]),
            application_id=(
                str(d["application_id"]) if d.get("application_id") else None
            ),
            paragraph_start_segment=int(ps) if ps is not None else None,
            paragraph_end_segment=int(pe) if pe is not None else None,
            anchor_start_word_id=str(d["anchor_start_word_id"]),
            anchor_end_word_id=str(d["anchor_end_word_id"]),
            text_preview=str(d.get("text_preview", "") or ""),
            text_hash=str(d["text_hash"]),
            vector=vector,
            model_name=str(d["model_name"]),
            dim=int(d["dim"]),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not EMBEDDING_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid embedding id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not SOURCE_ID_RE.match(self.source_id):
            raise ProjectValidationError(
                f"Invalid source id: {self.source_id!r}"
            )
        if self.kind not in EMBEDDING_KINDS:
            raise ProjectValidationError(
                f"kind must be one of {EMBEDDING_KINDS}; got {self.kind!r}"
            )
        if self.kind == EMBEDDING_KIND_CODED_SEGMENT:
            if not self.application_id:
                raise ProjectValidationError(
                    "coded_segment entries require application_id"
                )
            if not APPLICATION_ID_RE.match(self.application_id):
                raise ProjectValidationError(
                    f"Invalid application id: {self.application_id!r}"
                )
            if (
                self.paragraph_start_segment is not None
                or self.paragraph_end_segment is not None
            ):
                raise ProjectValidationError(
                    "coded_segment entries must not set paragraph indices"
                )
        else:
            if self.application_id is not None:
                raise ProjectValidationError(
                    "uncoded_paragraph entries must not set application_id"
                )
            if self.paragraph_start_segment is None or self.paragraph_end_segment is None:
                raise ProjectValidationError(
                    "uncoded_paragraph entries require paragraph indices"
                )
            if self.paragraph_start_segment < 0:
                raise ProjectValidationError(
                    "paragraph_start_segment must be ≥ 0"
                )
            if self.paragraph_end_segment < self.paragraph_start_segment:
                raise ProjectValidationError(
                    "paragraph_end_segment must be ≥ paragraph_start_segment"
                )
        # Anchors: shape and ordering, parse_word_id raises on garbage.
        sa_seg, sa_word = parse_word_id(self.anchor_start_word_id)
        ea_seg, ea_word = parse_word_id(self.anchor_end_word_id)
        if (sa_seg, sa_word) > (ea_seg, ea_word):
            raise ProjectValidationError(
                f"anchor_start_word_id must be ≤ anchor_end_word_id; "
                f"got {self.anchor_start_word_id!r} > "
                f"{self.anchor_end_word_id!r}"
            )
        # Hash shape.
        if not TEXT_HASH_RE.match(self.text_hash):
            raise ProjectValidationError(
                f"text_hash must be 64-char hex; got {self.text_hash!r}"
            )
        # Vector shape.
        if not isinstance(self.vector, tuple):
            raise ProjectValidationError(
                "vector must be a tuple of floats"
            )
        if not (MIN_VECTOR_DIM <= len(self.vector) <= MAX_VECTOR_DIM):
            raise ProjectValidationError(
                f"vector length must be in [{MIN_VECTOR_DIM}, {MAX_VECTOR_DIM}]; "
                f"got {len(self.vector)}"
            )
        if self.dim != len(self.vector):
            raise ProjectValidationError(
                f"dim {self.dim} != len(vector) {len(self.vector)}"
            )
        for i, x in enumerate(self.vector):
            if not isinstance(x, float):
                raise ProjectValidationError(
                    f"vector[{i}] must be a float"
                )
            if math.isnan(x) or math.isinf(x):
                raise ProjectValidationError(
                    f"vector[{i}] must be finite; got {x}"
                )
        # Model name.
        if not self.model_name:
            raise ProjectValidationError("model_name is required")
        if len(self.model_name) > MAX_MODEL_NAME_LEN:
            raise ProjectValidationError(
                f"model_name exceeds {MAX_MODEL_NAME_LEN} chars"
            )
        # Text preview length.
        if len(self.text_preview) > MAX_TEXT_PREVIEW_LEN:
            raise ProjectValidationError(
                f"text_preview exceeds {MAX_TEXT_PREVIEW_LEN} chars"
            )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def key(self) -> tuple[str, ...]:
        if self.kind == EMBEDDING_KIND_CODED_SEGMENT:
            return (self.kind, self.source_id, self.application_id or "")
        return (
            self.kind,
            self.source_id,
            str(self.paragraph_start_segment),
            str(self.paragraph_end_segment),
        )


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def embeddings_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's embeddings index."""
    return project_dir(projects_root, project_id) / EMBEDDINGS_DIRNAME


def embedding_state_path(
    projects_root: Path, project_id: str, embedding_id: str
) -> Path:
    """Return the path of a single embedding entry's JSON file."""
    if not EMBEDDING_ID_RE.match(embedding_id):
        raise ProjectValidationError(
            f"Invalid embedding id: {embedding_id!r}"
        )
    return embeddings_dir(projects_root, project_id) / f"{embedding_id}.json"


def save_embedding_entry(
    projects_root: Path, entry: EmbeddingEntry
) -> Path:
    """Persist an entry. Mirrors ``save_application``."""
    entry.validate()
    parent = project_dir(projects_root, entry.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its embedding entries."
        )
    ed = embeddings_dir(projects_root, entry.project_id)
    ed.mkdir(parents=True, exist_ok=True)
    target = embedding_state_path(projects_root, entry.project_id, entry.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_embedding_entry(
    projects_root: Path, project_id: str, embedding_id: str
) -> EmbeddingEntry:
    """Load an embedding entry by id. Raises ``FileNotFoundError`` if missing."""
    p = embedding_state_path(projects_root, project_id, embedding_id)
    if not p.exists():
        raise FileNotFoundError(f"No embedding entry at {p}")
    return EmbeddingEntry.from_dict(json.loads(p.read_text()))


def list_embedding_entries(
    projects_root: Path,
    project_id: str,
    *,
    kind: str | None = None,
    source_id: str | None = None,
) -> list[EmbeddingEntry]:
    """List all embedding entries in a project, optionally filtered.

    Filters AND-combine when both are passed. Skips files that don't
    parse so a single corrupt file doesn't break a search. Sorted by
    (source_id, kind, id) for deterministic ordering.
    """
    if kind is not None and kind not in EMBEDDING_KINDS:
        raise ProjectValidationError(
            f"kind filter must be one of {EMBEDDING_KINDS}; got {kind!r}"
        )
    if source_id is not None and not SOURCE_ID_RE.match(source_id):
        raise ProjectValidationError(
            f"Invalid source id filter: {source_id!r}"
        )
    ed = embeddings_dir(projects_root, project_id)
    if not ed.exists():
        return []
    out: list[EmbeddingEntry] = []
    for f in sorted(ed.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        eid = f.stem
        if not EMBEDDING_ID_RE.match(eid):
            continue
        try:
            e = EmbeddingEntry.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if kind is not None and e.kind != kind:
            continue
        if source_id is not None and e.source_id != source_id:
            continue
        out.append(e)
    out.sort(key=lambda e: (e.source_id, e.kind, e.id))
    return out


def delete_embedding_entry(
    projects_root: Path, project_id: str, embedding_id: str
) -> bool:
    """Remove an entry's file. Returns False if it didn't exist."""
    p = embedding_state_path(projects_root, project_id, embedding_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(
            f"Refusing to delete outside root: {p}"
        )
    p.unlink()
    return True


def clear_embedding_index(projects_root: Path, project_id: str) -> int:
    """Delete every entry in the index. Returns the number of entries removed.

    Convenience for "the model changed, blow it all away" flows. Doesn't
    remove the directory itself; the next refresh will populate it.
    """
    ed = embeddings_dir(projects_root, project_id)
    if not ed.exists():
        return 0
    real_root = projects_root.resolve()
    n = 0
    for f in list(ed.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        real_p = f.resolve()
        if not str(real_p).startswith(str(real_root)):
            continue
        f.unlink()
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #


# Public type alias for the embed callable. Given a list of texts,
# return a list of vectors (same length, same order). Exact return type
# is sequences-of-floats so callers can use lists, tuples, numpy arrays
# under the hood; we coerce to tuple-of-float at storage time.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class RefreshResult:
    """Summary of one :func:`refresh_embedding_index` call.

    All four fields are tuples of natural-key tuples, so a caller can
    compose them with the desired-spans output without re-loading
    files. Counts (``len(...)``) are the typical UI metric.
    """

    added: tuple[tuple[str, ...], ...] = ()
    updated: tuple[tuple[str, ...], ...] = ()
    removed: tuple[tuple[str, ...], ...] = ()
    unchanged: tuple[tuple[str, ...], ...] = ()

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def updated_count(self) -> int:
        return len(self.updated)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged)


def refresh_embedding_index(
    *,
    projects_root: Path,
    project_id: str,
    applications: Sequence[Application],
    segments_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    embed_fn: EmbedFn,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    now: str | None = None,
) -> RefreshResult:
    """Bring the on-disk index in sync with the desired set.

    Steps:

    1. Compute desired spans from ``applications`` + ``segments_by_source``.
    2. Compare against existing entries (deterministic ids do the
       lookup work).
    3. For each desired span: if no existing entry, embed (added). If
       an existing entry exists but its ``text_hash`` or ``model_name``
       differ, embed (updated). Otherwise leave alone (unchanged).
    4. For each existing entry whose key isn't desired: delete (removed).

    Embeddings are batched to ``batch_size`` at a time (default
    :data:`DEFAULT_BATCH_SIZE`) so HTTP overhead amortises. ``embed_fn``
    is invoked exactly ⌈n_to_embed / batch_size⌉ times.

    Returns a :class:`RefreshResult` summarising the four categories
    by natural key so the caller can log / UI-report which spans
    changed.

    Notes:

    * No embed call is made if every desired span is already up-to-date
      and no orphans exist; this is the "fast path" the editor invokes
      after non-content edits.
    * Project directory must already exist (i.e. the project has been
      saved at least once); otherwise FileNotFoundError surfaces.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    if not project_dir(projects_root, project_id).exists():
        raise FileNotFoundError(
            f"Project directory does not exist: "
            f"{project_dir(projects_root, project_id)}"
        )
    if batch_size < 1:
        raise ProjectValidationError(
            f"batch_size must be ≥ 1; got {batch_size}"
        )
    if not model_name or not isinstance(model_name, str):
        raise ProjectValidationError("model_name is required (non-empty string)")
    if len(model_name) > MAX_MODEL_NAME_LEN:
        raise ProjectValidationError(
            f"model_name exceeds {MAX_MODEL_NAME_LEN} chars"
        )

    desired = desired_index_spans(
        applications=applications,
        segments_by_source=segments_by_source,
    )
    desired_by_key: dict[tuple[str, ...], IndexableSpan] = {
        s.key(): s for s in desired
    }
    desired_keys = set(desired_by_key.keys())

    existing = list_embedding_entries(projects_root, project_id)
    existing_by_key: dict[tuple[str, ...], EmbeddingEntry] = {
        e.key(): e for e in existing
    }

    added: list[tuple[str, ...]] = []
    updated: list[tuple[str, ...]] = []
    unchanged: list[tuple[str, ...]] = []
    to_embed: list[tuple[IndexableSpan, str]] = []  # (span, status)

    for key, span in desired_by_key.items():
        prior = existing_by_key.get(key)
        h = text_hash(span.text)
        if (
            prior is not None
            and prior.text_hash == h
            and prior.model_name == model_name
        ):
            unchanged.append(key)
            continue
        status = "updated" if prior is not None else "added"
        to_embed.append((span, status))

    # Batch the embed calls. Order of returned vectors matches input.
    if to_embed:
        for i in range(0, len(to_embed), batch_size):
            chunk = to_embed[i : i + batch_size]
            texts = [s.text for s, _ in chunk]
            vectors = embed_fn(texts)
            if len(vectors) != len(chunk):
                raise ProjectValidationError(
                    f"embed_fn returned {len(vectors)} vectors for "
                    f"{len(chunk)} inputs"
                )
            for (span, status), vec in zip(chunk, vectors):
                entry = EmbeddingEntry.new(
                    project_id=project_id,
                    span=span,
                    vector=vec,
                    model_name=model_name,
                    now=now,
                )
                save_embedding_entry(projects_root, entry)
                if status == "added":
                    added.append(span.key())
                else:
                    updated.append(span.key())

    removed: list[tuple[str, ...]] = []
    for key, e in existing_by_key.items():
        if key in desired_keys:
            continue
        delete_embedding_entry(projects_root, project_id, e.id)
        removed.append(key)

    return RefreshResult(
        added=tuple(added),
        updated=tuple(updated),
        removed=tuple(removed),
        unchanged=tuple(unchanged),
    )


# --------------------------------------------------------------------------- #
# Similarity search (consumed by F8.3 / F8.5)
# --------------------------------------------------------------------------- #


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity in [-1, 1] for two equal-length vectors.

    Two zero vectors return 0.0 (rather than NaN); this is the
    forgiving, search-friendly answer.
    """
    if len(a) != len(b):
        raise ProjectValidationError(
            f"vector length mismatch: {len(a)} vs {len(b)}"
        )
    if not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        xf = float(x)
        yf = float(y)
        dot += xf * yf
        na += xf * xf
        nb += yf * yf
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search_similar(
    *,
    projects_root: Path,
    project_id: str,
    query_vector: Sequence[float],
    kind: str | None = None,
    source_id: str | None = None,
    top_k: int = 10,
    min_score: float = -1.0,
) -> list[tuple[float, EmbeddingEntry]]:
    """Linear-scan nearest-neighbour search over the on-disk index.

    Returns the ``top_k`` (score, entry) pairs whose cosine similarity
    is ≥ ``min_score``, sorted by score descending. Mismatched
    dimensionalities are silently skipped — different model families
    produce different-shape vectors and the right answer is "those are
    not comparable", not raising.

    The implementation is deliberately the simplest thing that works:
    load every entry, score, sort. For projects with thousands of
    entries this is fine (a 1024-dim cosine in pure Python is
    microseconds; 10_000 of them is tens of milliseconds). When that
    stops being true, this is the function to swap for a vector-store
    backend without touching callers.
    """
    if top_k < 1:
        raise ProjectValidationError(f"top_k must be ≥ 1; got {top_k}")
    qv = tuple(float(x) for x in query_vector)
    entries = list_embedding_entries(
        projects_root, project_id, kind=kind, source_id=source_id
    )
    scored: list[tuple[float, EmbeddingEntry]] = []
    for e in entries:
        if e.dim != len(qv):
            continue
        s = cosine_similarity(qv, e.vector)
        if s < min_score:
            continue
        scored.append((s, e))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return scored[:top_k]


# --------------------------------------------------------------------------- #
# Convenience adapter for the F8.1 backend
# --------------------------------------------------------------------------- #


def make_embed_fn_from_backend(
    config: "BackendConfig",  # noqa: F821 — runtime import below
    model: str,
    *,
    backend: "ModelBackend | None" = None,  # noqa: F821
    transport: "Transport | None" = None,  # noqa: F821
) -> EmbedFn:
    """Wrap a F8.1 backend's ``embed`` method as an :data:`EmbedFn`.

    Imports the F8.1 module locally to keep this F8.2 module's import
    surface narrow (and avoid a top-level circular dependency once
    F8.x grows).

    The returned callable, when invoked, calls
    :meth:`scribe.ai_backend.ModelBackend.embed` once per call (no
    further batching — this F8.2 module already chunks, so doubling up
    would just complicate request shapes).
    """
    from .ai_backend import (
        BackendConfig,
        EmbeddingRequest,
        backend_for_config,
        urllib_transport,
    )

    if not isinstance(config, BackendConfig):
        raise ProjectValidationError("config must be a BackendConfig")
    if not model or not isinstance(model, str):
        raise ProjectValidationError("model is required (non-empty string)")
    chosen_backend = backend if backend is not None else backend_for_config(config)
    chosen_transport = transport if transport is not None else urllib_transport

    def _fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        request = EmbeddingRequest(model=model, inputs=tuple(str(t) for t in texts))
        response = chosen_backend.embed(
            config, request, transport=chosen_transport
        )
        return tuple(tuple(v) for v in response.vectors)

    return _fn
