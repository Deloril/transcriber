"""Re-anchoring on transcript edit + orphan review queue (F4.5).

Per PLANNING.md F4.5:

  > Re-anchoring strategy on transcript edit; "orphaned application"
  > review queue when anchors are deleted.

F4.1 anchored each :class:`scribe.applications.Application` to a
``s<segment>w<word>`` word-id pair. Word ids are stable *within a
transcript*, but a researcher who edits the transcript — fixing a
typo, splitting a run-on sentence, anonymising a name — can shift,
insert, delete, or rewrite the words those ids name. Without help,
the audit trail breaks: code applications keep pointing at offsets
that now mean different text.

This module is the F4.5 "help". Given the *old* transcript and the
*new* transcript (both in the canonical Scribe ``segments`` shape),
it computes for each application one of three outcomes:

* **unchanged** — the same word ids still hold the same word texts
  in the new transcript. The application survives the edit
  bit-identical (sub-word offsets included).
* **reanchored** — the original ids no longer match, but the *text*
  the application originally covered was found elsewhere in the new
  transcript. The application's anchor word ids are updated; sub-
  word offsets are dropped (we know the words but not the precise
  characters anymore).
* **orphaned** — the original anchored text was not found at all.
  The application keeps its old anchors and goes on the project's
  *orphan review queue* for a human to triage.

The matching is normalised case-insensitively and with punctuation
stripped, so common ASR fixups (``"hello,"`` → ``"hello"``) don't
cause spurious orphans. Pure-punctuation tokens in the original
anchor are skipped during matching but still bracketed by the word
boundaries we anchor on. When several matches exist in the new
transcript, the one whose start position is closest to the original
``(segment, word)`` location wins — so a typo fix close to the
anchor doesn't suddenly point the anchor at a textual collision
elsewhere in the file.

What this module is **not**:

* It does **not** auto-apply outcomes to persisted applications.
  Producing a :class:`ReanchorPlan` and applying / queuing each
  outcome are separate steps, so a caller can preview, log, or
  surface a UI prompt before mutating data. F9.1's event log will
  later record both halves.
* It does **not** parse the ``edited.json`` editor sidecar or know
  about ``segments[].words[]`` keys beyond ``text`` and ``speaker``.
  Anything else (timing, score, token IDs) is left untouched.
* It does **not** modify the orphan queue file from the planner —
  there are explicit ``append_orphan_entries`` /
  ``remove_from_orphan_queue`` helpers so persistence is opt-in and
  testable.

Stand-alone (no FastAPI, no engine), matching the conventions of
:mod:`scribe.application_spans`, :mod:`scribe.selection_snap`,
:mod:`scribe.application_gutter`, and :mod:`scribe.matrix`.

On-disk persistence
-------------------

The orphan queue lives at::

    projects/<project_id>/orphan_queue.json

A single JSON object with an ``entries`` array; entries are
:class:`OrphanEntry` records carrying enough context for a reviewer
to relocate (or delete, or re-apply manually) the orphaned
application even after a long time has passed. Append-add semantics
deduplicate by ``application_id`` (a fresh outcome on the same
application replaces the older entry rather than stacking copies).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .applications import (
    APPLICATION_ID_RE,
    Application,
    make_word_id,
    parse_word_id,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


REANCHOR_STATUS_UNCHANGED = "unchanged"
REANCHOR_STATUS_REANCHORED = "reanchored"
REANCHOR_STATUS_ORPHANED = "orphaned"
REANCHOR_STATUSES: tuple[str, ...] = (
    REANCHOR_STATUS_UNCHANGED,
    REANCHOR_STATUS_REANCHORED,
    REANCHOR_STATUS_ORPHANED,
)


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #


# Strip everything that isn't a "word character" (Unicode letters /
# digits / underscore). The ``\w`` shorthand under ``re.UNICODE`` (the
# default in Python 3) does the right thing for non-ASCII transcripts.
# This is **not** a tokeniser — it's a fuzzy comparator. ``"hello,"``
# normalises to ``"hello"``; ``"--"`` to ``""`` (empty, treated as
# pure punctuation).
_PUNCT_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_word(text: str) -> str:
    """Lowercase and strip non-word characters for matching.

    Used as the equivalence relation when re-anchoring: two word texts
    are "the same" if their normalised forms are equal. Reasonable for
    transcripts, where editor fixups tend to be capitalisation,
    trailing punctuation, or stray whitespace.

    Empty / non-string input returns an empty string. Pure-punctuation
    tokens also normalise to empty and are treated as "skip this
    token" by :func:`find_text_run`.
    """
    if not isinstance(text, str):
        return ""
    return _PUNCT_RE.sub("", text).lower()


# --------------------------------------------------------------------------- #
# Transcript helpers
# --------------------------------------------------------------------------- #


def collect_word_texts(
    segments: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    """Return ``segments[].words[].text`` as a 2-D list of strings.

    Missing or malformed segments contribute an empty list at their
    index so segment-index alignment with the input is preserved. The
    function never raises on missing keys; that's by design — the F4.5
    re-anchorer wants to be able to compute on partially-corrupted
    transcripts (and surface that as orphans) rather than abort.
    """
    out: list[list[str]] = []
    for seg in segments:
        if isinstance(seg, Mapping):
            ws = seg.get("words")
            if (
                isinstance(ws, Sequence)
                and not isinstance(ws, (str, bytes))
            ):
                out.append([
                    str(w.get("text", "")) if isinstance(w, Mapping) else ""
                    for w in ws
                ])
                continue
        out.append([])
    return out


def anchored_words(
    application: Application,
    segments: Sequence[Mapping[str, Any]],
) -> list[str] | None:
    """Return the list of word texts an application's anchor covers.

    Returns ``None`` if either anchor word id falls outside the
    transcript. The caller can use the ``None`` signal to short-circuit
    the fast path in :func:`reanchor_application`.

    The list is in document order, inclusive on both ends, with one
    string per word. Sub-word character offsets are intentionally **not**
    applied — re-anchoring works at word resolution; sub-word offsets
    are dropped on a successful re-anchor since the new word's
    characters may not match the old one's exactly.
    """
    sa_seg, sa_word = parse_word_id(application.anchor_start_word_id)
    ea_seg, ea_word = parse_word_id(application.anchor_end_word_id)
    words_2d = collect_word_texts(segments)
    if sa_seg < 0 or sa_seg >= len(words_2d):
        return None
    if ea_seg < 0 or ea_seg >= len(words_2d):
        return None
    if sa_word < 0 or sa_word >= len(words_2d[sa_seg]):
        return None
    if ea_word < 0 or ea_word >= len(words_2d[ea_seg]):
        return None

    out: list[str] = []
    for si in range(sa_seg, ea_seg + 1):
        seg = words_2d[si]
        start = sa_word if si == sa_seg else 0
        end = ea_word if si == ea_seg else len(seg) - 1
        if end < start:
            continue
        out.extend(seg[start:end + 1])
    return out


# --------------------------------------------------------------------------- #
# Text-run search
# --------------------------------------------------------------------------- #


def find_text_run(
    target: Sequence[str],
    segments: Sequence[Mapping[str, Any]],
    *,
    near: tuple[int, int] | None = None,
) -> tuple[str, str] | None:
    """Locate a contiguous word-run in ``segments`` matching ``target``.

    Matching is **normalised**: case-insensitive, punctuation-stripped
    (see :func:`normalize_word`). Pure-punctuation tokens in either
    ``target`` or ``segments`` are skipped — they don't contribute to
    the match and don't break it.

    Returns the (start_word_id, end_word_id) of the matched span, or
    ``None`` when no match exists. The returned boundary tokens are
    the first and last *non-empty-after-normalisation* tokens that
    matched — surrounding pure-punctuation tokens are not included in
    the bracket.

    When several matches exist and ``near=(seg_idx, word_idx)`` is
    given, prefer the match whose start position is closest to that
    hint (segment-distance dominates, word-distance breaks ties). With
    no hint, the earliest-starting match wins.

    Empty ``target`` returns ``None``: an empty selection is
    meaningless, and a caller that hits this case has likely already
    detected an out-of-range anchor in the *old* transcript and
    should orphan rather than search.
    """
    norm_target = [normalize_word(t) for t in target]
    norm_target = [t for t in norm_target if t]
    if not norm_target:
        return None

    # Flatten the new transcript into [(seg, word, normalised)] tuples,
    # then strip the empty-normalised entries — those are punctuation
    # tokens and shouldn't drive the match.
    nonempty: list[tuple[int, int, str]] = []
    for si, seg_words in enumerate(collect_word_texts(segments)):
        for wi, text in enumerate(seg_words):
            n = normalize_word(text)
            if n:
                nonempty.append((si, wi, n))

    m = len(norm_target)
    if m > len(nonempty):
        return None

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for s in range(0, len(nonempty) - m + 1):
        match = True
        for k in range(m):
            if nonempty[s + k][2] != norm_target[k]:
                match = False
                break
        if match:
            start_seg_word = (nonempty[s][0], nonempty[s][1])
            end_seg_word = (nonempty[s + m - 1][0], nonempty[s + m - 1][1])
            candidates.append((start_seg_word, end_seg_word))

    if not candidates:
        return None

    if near is not None:
        ns, nw = near
        # Big multiplier on segment delta keeps "same segment, far
        # word" closer than "different segment, same word".
        def distance(c: tuple[tuple[int, int], tuple[int, int]]) -> int:
            (ss, sw), _ = c
            return abs(ss - ns) * 1_000_000 + abs(sw - nw)
        candidates.sort(key=distance)

    start, end = candidates[0]
    return (make_word_id(*start), make_word_id(*end))


# --------------------------------------------------------------------------- #
# Reanchor outcome + plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReanchorOutcome:
    """The computed result of re-anchoring one application against a new transcript.

    All fields below are populated for every status; the meaning shifts
    a little:

    * ``status == "unchanged"``: ``new_*`` mirror the original anchors
      and offsets exactly.
    * ``status == "reanchored"``: ``new_anchor_*`` carry the located
      span; ``new_*_char_offset`` are always ``None`` (sub-word
      offsets are dropped on re-anchor).
    * ``status == "orphaned"``: ``new_anchor_*`` are ``None``; the
      caller should keep the application's old anchors and persist the
      outcome via :func:`append_orphan_entries`.

    ``original_anchored_text`` is a snapshot of the word texts the
    application covered in the *old* transcript. It survives on the
    orphan-queue entry so a reviewer can search for it later, even
    after further edits.
    """

    application_id: str
    status: str
    new_anchor_start_word_id: str | None
    new_anchor_end_word_id: str | None
    new_start_char_offset: int | None
    new_end_char_offset: int | None
    original_anchored_text: tuple[str, ...]
    reason: str

    def as_patch(self) -> dict[str, Any]:
        """Return a dict suitable for :meth:`Application.apply_update`.

        Only valid for ``unchanged`` / ``reanchored`` outcomes — orphans
        have no patch (the application keeps its old anchors). Raises
        :class:`ProjectValidationError` for orphan outcomes.
        """
        if self.status == REANCHOR_STATUS_ORPHANED:
            raise ProjectValidationError(
                f"orphaned outcome for {self.application_id!r} has no patch"
            )
        return {
            "anchor_start_word_id": self.new_anchor_start_word_id,
            "anchor_end_word_id": self.new_anchor_end_word_id,
            "start_char_offset": self.new_start_char_offset,
            "end_char_offset": self.new_end_char_offset,
        }


@dataclass
class ReanchorPlan:
    """A batch of :class:`ReanchorOutcome` keyed by application id.

    Iteration order matches the order of the input applications. The
    ``unchanged`` / ``reanchored`` / ``orphaned`` properties are
    convenience filters; they don't reorder.
    """

    outcomes: list[ReanchorOutcome] = field(default_factory=list)

    @property
    def unchanged(self) -> list[ReanchorOutcome]:
        return [o for o in self.outcomes if o.status == REANCHOR_STATUS_UNCHANGED]

    @property
    def reanchored(self) -> list[ReanchorOutcome]:
        return [o for o in self.outcomes if o.status == REANCHOR_STATUS_REANCHORED]

    @property
    def orphaned(self) -> list[ReanchorOutcome]:
        return [o for o in self.outcomes if o.status == REANCHOR_STATUS_ORPHANED]

    def for_application(self, application_id: str) -> ReanchorOutcome | None:
        """Return the outcome whose ``application_id`` matches, or None."""
        for o in self.outcomes:
            if o.application_id == application_id:
                return o
        return None


# --------------------------------------------------------------------------- #
# Reanchor algorithm
# --------------------------------------------------------------------------- #


def _normalised_equal(a: Sequence[str], b: Sequence[str]) -> bool:
    """Two word-text sequences are equivalent under :func:`normalize_word`?

    Skips empty-normalised tokens on both sides — punctuation churn
    doesn't count as a real change. Two empty sequences are equal.
    """
    na = [normalize_word(t) for t in a]
    na = [t for t in na if t]
    nb = [normalize_word(t) for t in b]
    nb = [t for t in nb if t]
    return na == nb


def reanchor_application(
    application: Application,
    old_segments: Sequence[Mapping[str, Any]],
    new_segments: Sequence[Mapping[str, Any]],
) -> ReanchorOutcome:
    """Compute a re-anchor outcome for one application.

    See module docstring for the three outcome statuses. The function
    is pure (no I/O); it takes both transcripts as inputs and returns
    a value.
    """
    old_text = anchored_words(application, old_segments) or []
    sa_seg, sa_word = parse_word_id(application.anchor_start_word_id)

    # Fast path: same word ids still hold the same text.
    new_text_at_old_anchor = anchored_words(application, new_segments)
    if (
        new_text_at_old_anchor is not None
        and _normalised_equal(new_text_at_old_anchor, old_text)
        and old_text
    ):
        return ReanchorOutcome(
            application_id=application.id,
            status=REANCHOR_STATUS_UNCHANGED,
            new_anchor_start_word_id=application.anchor_start_word_id,
            new_anchor_end_word_id=application.anchor_end_word_id,
            new_start_char_offset=application.start_char_offset,
            new_end_char_offset=application.end_char_offset,
            original_anchored_text=tuple(old_text),
            reason="anchor word ids still hold the same text",
        )

    if not old_text:
        # The original anchors are out of range in the *old* transcript
        # — we have no reference text to search for. Orphan and let a
        # human resolve it.
        return ReanchorOutcome(
            application_id=application.id,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=(),
            reason="anchor out of range in old transcript",
        )

    found = find_text_run(old_text, new_segments, near=(sa_seg, sa_word))
    if found is None:
        return ReanchorOutcome(
            application_id=application.id,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=tuple(old_text),
            reason="anchored text not found in new transcript",
        )
    new_start, new_end = found
    return ReanchorOutcome(
        application_id=application.id,
        status=REANCHOR_STATUS_REANCHORED,
        new_anchor_start_word_id=new_start,
        new_anchor_end_word_id=new_end,
        new_start_char_offset=None,
        new_end_char_offset=None,
        original_anchored_text=tuple(old_text),
        reason="anchored text relocated by content match",
    )


def reanchor_applications(
    applications: Iterable[Application],
    old_segments: Sequence[Mapping[str, Any]],
    new_segments: Sequence[Mapping[str, Any]],
) -> ReanchorPlan:
    """Compute a :class:`ReanchorPlan` for many applications at once."""
    outcomes = [
        reanchor_application(a, old_segments, new_segments)
        for a in applications
    ]
    return ReanchorPlan(outcomes=outcomes)


def apply_reanchor_outcome(
    application: Application,
    outcome: ReanchorOutcome,
    *,
    now: str | None = None,
) -> Application:
    """Return a (possibly new) Application reflecting the outcome.

    * ``unchanged`` returns the input as-is, ``modified_at`` not
      advanced (we don't want to dirty an unchanged record).
    * ``reanchored`` returns a deep copy with new anchor ids and the
      sub-word offsets cleared. ``modified_at`` is bumped via
      :meth:`Application.apply_update`.
    * ``orphaned`` raises :class:`ProjectValidationError` — orphans
      keep their old anchors and go to the orphan queue, not into the
      application file.

    Mismatched ``outcome.application_id`` raises.
    """
    if outcome.application_id != application.id:
        raise ProjectValidationError(
            f"outcome is for {outcome.application_id!r}; "
            f"got application {application.id!r}"
        )
    if outcome.status == REANCHOR_STATUS_UNCHANGED:
        return application
    if outcome.status == REANCHOR_STATUS_ORPHANED:
        raise ProjectValidationError(
            f"cannot apply orphaned outcome to application "
            f"{application.id!r}; queue it via append_orphan_entries"
        )
    if outcome.status != REANCHOR_STATUS_REANCHORED:
        raise ProjectValidationError(
            f"unknown reanchor status: {outcome.status!r}"
        )
    new = Application.from_dict(application.to_dict())
    new.apply_update(outcome.as_patch(), now=now)
    return new


# --------------------------------------------------------------------------- #
# Orphan review queue
# --------------------------------------------------------------------------- #


@dataclass
class OrphanEntry:
    """One entry in a project's orphan-application review queue.

    Carries enough context for a reviewer to relocate (or delete, or
    re-apply manually) an orphaned application even after a long time.
    The ``application_id`` doubles as the queue's primary key.
    """

    application_id: str
    code_id: str
    source_id: str
    coder_id: str
    old_anchor_start_word_id: str
    old_anchor_end_word_id: str
    original_anchored_text: list[str] = field(default_factory=list)
    reason: str = ""
    detected_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OrphanEntry":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "OrphanEntry payload must be an object"
            )
        for required in (
            "application_id",
            "code_id",
            "source_id",
            "coder_id",
            "old_anchor_start_word_id",
            "old_anchor_end_word_id",
        ):
            if required not in d:
                raise ProjectValidationError(
                    f"OrphanEntry payload missing required key: {required}"
                )
        text = d.get("original_anchored_text") or []
        if not isinstance(text, Sequence) or isinstance(text, (str, bytes)):
            raise ProjectValidationError(
                "original_anchored_text must be a list of strings"
            )
        return cls(
            application_id=str(d["application_id"]),
            code_id=str(d["code_id"]),
            source_id=str(d["source_id"]),
            coder_id=str(d["coder_id"]),
            old_anchor_start_word_id=str(d["old_anchor_start_word_id"]),
            old_anchor_end_word_id=str(d["old_anchor_end_word_id"]),
            original_anchored_text=[str(t) for t in text],
            reason=str(d.get("reason", "") or ""),
            detected_at=str(d.get("detected_at", "") or ""),
        )

    def validate(self) -> None:
        if not APPLICATION_ID_RE.match(self.application_id):
            raise ProjectValidationError(
                f"Invalid orphan application_id: {self.application_id!r}"
            )
        # Word-id shapes are validated cheaply via parse_word_id; if
        # malformed it raises ProjectValidationError.
        parse_word_id(self.old_anchor_start_word_id)
        parse_word_id(self.old_anchor_end_word_id)


def make_orphan_entry(
    application: Application,
    outcome: ReanchorOutcome,
    *,
    now: str | None = None,
) -> OrphanEntry:
    """Construct an OrphanEntry from an orphaned outcome + its application.

    Raises if ``outcome.status != "orphaned"`` or if the outcome's
    ``application_id`` doesn't match.
    """
    if outcome.status != REANCHOR_STATUS_ORPHANED:
        raise ProjectValidationError(
            f"cannot make orphan entry from {outcome.status!r} outcome"
        )
    if outcome.application_id != application.id:
        raise ProjectValidationError(
            f"outcome is for {outcome.application_id!r}; "
            f"got application {application.id!r}"
        )
    entry = OrphanEntry(
        application_id=application.id,
        code_id=application.code_id,
        source_id=application.source_id,
        coder_id=application.coder_id,
        old_anchor_start_word_id=application.anchor_start_word_id,
        old_anchor_end_word_id=application.anchor_end_word_id,
        original_anchored_text=list(outcome.original_anchored_text),
        reason=outcome.reason,
        detected_at=now or utcnow_iso(),
    )
    entry.validate()
    return entry


def orphan_queue_path(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk path for a project's orphan queue file."""
    return project_dir(projects_root, project_id) / "orphan_queue.json"


def load_orphan_queue(
    projects_root: Path, project_id: str
) -> list[OrphanEntry]:
    """Load the orphan queue for a project. Empty list if absent.

    Skips malformed entries rather than refusing the whole file — a
    single broken row shouldn't block reviewing the rest of the queue.
    Sorted by ``detected_at`` ascending, ties broken by
    ``application_id`` for stable output.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(
            f"Invalid project id: {project_id!r}"
        )
    p = orphan_queue_path(projects_root, project_id)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ProjectValidationError(
            f"orphan_queue.json is not valid JSON: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise ProjectValidationError(
            "orphan_queue.json must be a JSON object"
        )
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ProjectValidationError(
            "orphan_queue.json 'entries' must be a list"
        )
    out: list[OrphanEntry] = []
    for raw in raw_entries:
        try:
            e = OrphanEntry.from_dict(raw)
            e.validate()
        except ProjectValidationError:
            continue
        out.append(e)
    out.sort(key=lambda e: (e.detected_at, e.application_id))
    return out


def save_orphan_queue(
    projects_root: Path,
    project_id: str,
    entries: Sequence[OrphanEntry],
) -> Path:
    """Persist a complete orphan queue file (overwriting any existing file).

    Validates every entry before writing; an invalid entry aborts the
    whole save (no partial write). Atomic via temp-file + rename.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(
            f"Invalid project id: {project_id!r}"
        )
    parent = project_dir(projects_root, project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its orphan queue."
        )
    # Validate up-front; partial writes are not allowed.
    for e in entries:
        e.validate()
    target = orphan_queue_path(projects_root, project_id)
    payload = {"entries": [e.to_dict() for e in entries]}
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def append_orphan_entries(
    projects_root: Path,
    project_id: str,
    entries: Iterable[OrphanEntry],
) -> list[OrphanEntry]:
    """Add entries to the queue, deduplicating by application_id.

    A fresh entry for an application_id already in the queue *replaces*
    the older entry (it carries newer ``detected_at`` and ``reason``
    context). Returns the merged, sorted list of entries on disk after
    the operation.
    """
    existing = load_orphan_queue(projects_root, project_id)
    by_id: dict[str, OrphanEntry] = {e.application_id: e for e in existing}
    for new_e in entries:
        new_e.validate()
        by_id[new_e.application_id] = new_e
    merged = sorted(
        by_id.values(),
        key=lambda e: (e.detected_at, e.application_id),
    )
    save_orphan_queue(projects_root, project_id, merged)
    return merged


def remove_from_orphan_queue(
    projects_root: Path,
    project_id: str,
    application_id: str,
) -> bool:
    """Remove the entry for ``application_id``. Returns False if absent."""
    if not APPLICATION_ID_RE.match(application_id):
        raise ProjectValidationError(
            f"Invalid application id: {application_id!r}"
        )
    existing = load_orphan_queue(projects_root, project_id)
    filtered = [e for e in existing if e.application_id != application_id]
    if len(filtered) == len(existing):
        return False
    save_orphan_queue(projects_root, project_id, filtered)
    return True


def record_orphans_from_plan(
    projects_root: Path,
    project_id: str,
    plan: ReanchorPlan,
    applications_by_id: Mapping[str, Application],
    *,
    now: str | None = None,
) -> list[OrphanEntry]:
    """Persist every orphaned outcome in ``plan`` to the project's queue.

    Convenience wrapper around :func:`make_orphan_entry` +
    :func:`append_orphan_entries`. Skips outcomes whose application id
    is missing from ``applications_by_id`` (defensive — the caller
    should typically pass the live application set, but a stale plan
    shouldn't crash the save). Returns the merged queue contents.
    """
    timestamp = now or utcnow_iso()
    new_entries: list[OrphanEntry] = []
    for outcome in plan.orphaned:
        app = applications_by_id.get(outcome.application_id)
        if app is None:
            continue
        new_entries.append(make_orphan_entry(app, outcome, now=timestamp))
    if not new_entries:
        return load_orphan_queue(projects_root, project_id)
    return append_orphan_entries(projects_root, project_id, new_entries)
