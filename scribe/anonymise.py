"""Anonymised export — redaction pass before bundling (F6.7).

Per PLANNING.md F6.7:

    Anonymised export — regenerate transcripts with a redaction pass
    before bundling.

The threat model is mundane but real: a research project carries
participant names, place names, employer names, and stray identifiers
(phone numbers, email addresses, NHS numbers, …) that must not survive
into the bundle a researcher hands to a journal, a co-author, or a
data-archive submission. Today's QDPX export (:mod:`scribe.refi_qda_project`)
ships everything as-is. F6.7 adds a transformation layer that:

  * Takes a project's entities (sources, codes, applications, memos,
    coders, participants, speaker maps, transcript segments) plus an
    optional list of free-form **redaction rules**;
  * Builds a :class:`RedactionPlan` from participants' ``name`` →
    ``pseudonym`` mappings, speaker labels' display-name → role
    pseudonyms, and the user-supplied custom rules;
  * Produces a *new* set of entities and segments with the rules
    applied — no in-place mutation of the project's on-disk state;
  * Hands those redacted entities to :func:`scribe.refi_qda_project.to_qdpx`
    for bundling.

This module is **pure** in the same sense as
:mod:`scribe.refi_qda_project` and :mod:`scribe.codebook_export`: it
takes already-loaded entities and returns redacted copies + bundled
``bytes``. The CLI (:mod:`scribe.scripts.export_anonymised_qdpx`) and
any HTTP endpoint do disk I/O and call in here for the transformation.

What gets redacted
------------------

* **Transcript segments** — every word's ``text`` and every
  segment's ``speaker`` field. Speaker labels are first looked up in
  the per-source :class:`SpeakerMap` to pick up role-based pseudonyms
  ("Interviewer", "Participant 03"), then run through the same
  rule-based pass as the rest of the text.
* **Source names + notes + custom-attribute values** — the source's
  display name often carries an interviewee's name; notes and
  attribute values almost always do.
* **Code definitions, inclusion/exclusion criteria, exemplars,
  theoretical memos, names** — exemplars are quoted text from the
  transcript; a rule that redacts the transcript must redact the
  exemplar too or the leak is reintroduced.
* **Memo titles + bodies + tags** — analytic memos quote participants
  freely.
* **Coder names** — the ICR matrix in a published paper shouldn't
  reveal which research-assistant did the second pass.
* **Project name + research question + sensitising concepts +
  description** — these sometimes name the field site.

What we **do not** redact:

* GUIDs, timestamps, version ids, code ids, application ids — the
  audit trail is a feature, not a leak.
* Code colours, code stages, methodology field — non-textual or
  vocabulary-bounded.
* Word-level timestamps (``start`` / ``end`` floats) — useful for
  resync and don't identify anyone.

How rules work
--------------

Each :class:`RedactionRule` has a ``pattern``, a ``replacement``, and
two flags:

* ``case_insensitive`` (default ``True``) — match without regard to
  letter case. When False, exact-case match.
* ``whole_word`` (default ``True``) — match only at word boundaries.
  Without this, a rule for ``"Pat"`` would also blank out ``"patio"``.
* ``regex`` (default ``False``) — when True, ``pattern`` is a Python
  regex; ``whole_word`` is ignored (the pattern fully controls
  matching) and ``case_insensitive`` becomes ``re.IGNORECASE``.

Rules apply in the order they're listed in :class:`RedactionPlan`.
Ordering matters when one replacement can become a substring of
another — e.g. you'd put rules for full names before rules for
surnames so ``"Pat Smith"`` doesn't end up as ``"P03 Smith"``.

Manifest
--------

The bundle includes a ``Redactions/manifest.json`` file listing, for
each rule, the **replacement** value (only) and a count of how many
substitutions it made across all entities. The original strings are
never written to the manifest — that would defeat the purpose. The
manifest is for transparency: reviewers can see the project was
redacted, and how aggressively, without being handed the key.

Scope and what's deferred
-------------------------

This iteration ships:

* The pure :class:`RedactionRule` / :class:`RedactionPlan` data model.
* :func:`build_redaction_plan` to seed a plan from participants +
  speaker maps + custom rules.
* :func:`redact_text` / :func:`redact_segments` /
  :func:`redact_source` / :func:`redact_code` / :func:`redact_memo` /
  :func:`redact_coder` / :func:`redact_project` — pure transformations
  returning new entity copies with redactions applied.
* :func:`build_anonymised_qdpx` — orchestrator that calls into
  :mod:`scribe.refi_qda_project` for the bundle layout.

Deferred (a later iteration can extend this module without breaking
on-disk format):

* AI-driven NER-based redaction (auto-detect names / orgs / locations
  via a local LLM). F8 will revisit; rule-based is the audit-friendly
  baseline.
* Audio / video media redaction (re-bleeping the source file). Out of
  scope for a *transcript* anonymiser.
* Selective redaction of individual applications or memos. Today the
  pass is whole-project; the rule list is the dial.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .applications import Application
from .coders import Coder
from .codes import Code, CodeRelation
from .memos import Memo, MemoLink
from .participants import Participant
from .projects import Project
from .refi_qda_project import (
    REFI_QDA_PROJECT_ORIGIN_DEFAULT,
    RenderedSource,
    render_source_plain_text,
    to_qdpx,
)
from .sources import Source
from .speaker_map import SpeakerMap


# --------------------------------------------------------------------------- #
# Rule data model
# --------------------------------------------------------------------------- #

# Bound the rule list so a rogue payload can't compile 100k regexes
# inside a hot loop.
MAX_RULES = 1024
# Bound each pattern / replacement so a single huge string can't OOM
# the process.
MAX_RULE_PATTERN_LEN = 1024
MAX_RULE_REPLACEMENT_LEN = 1024


@dataclass(frozen=True)
class RedactionRule:
    """One literal-or-regex substitution rule.

    The defaults match the common case: literal text, case-insensitive,
    word-boundaried. ``regex=True`` switches to Python regex semantics
    — useful for things like ``r"\\b\\d{3}-\\d{4}\\b"`` for phone
    numbers.

    Validation happens in :meth:`compile`. Construction itself never
    raises; that keeps dataclass introspection clean and makes
    serialisation round-trips safe.
    """

    pattern: str
    replacement: str
    case_insensitive: bool = True
    whole_word: bool = True
    regex: bool = False

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "replacement": self.replacement,
            "case_insensitive": self.case_insensitive,
            "whole_word": self.whole_word,
            "regex": self.regex,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RedactionRule":
        if not isinstance(d, Mapping):
            raise ValueError("RedactionRule payload must be an object")
        if "pattern" not in d or "replacement" not in d:
            raise ValueError(
                "RedactionRule payload missing required keys "
                "'pattern' and 'replacement'"
            )
        return cls(
            pattern=str(d["pattern"]),
            replacement=str(d["replacement"]),
            case_insensitive=bool(d.get("case_insensitive", True)),
            whole_word=bool(d.get("whole_word", True)),
            regex=bool(d.get("regex", False)),
        )

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #

    def compile(self) -> re.Pattern[str]:
        """Compile the rule into a ready-to-use :class:`re.Pattern`.

        Raises ``ValueError`` for empty patterns, oversized strings, or
        invalid regex syntax (when ``regex=True``).
        """
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("RedactionRule pattern must be a non-empty string")
        if len(self.pattern) > MAX_RULE_PATTERN_LEN:
            raise ValueError(
                f"RedactionRule pattern must be ≤ {MAX_RULE_PATTERN_LEN} chars"
            )
        if not isinstance(self.replacement, str):
            raise ValueError("RedactionRule replacement must be a string")
        if len(self.replacement) > MAX_RULE_REPLACEMENT_LEN:
            raise ValueError(
                f"RedactionRule replacement must be ≤ "
                f"{MAX_RULE_REPLACEMENT_LEN} chars"
            )

        flags = re.IGNORECASE if self.case_insensitive else 0

        if self.regex:
            try:
                return re.compile(self.pattern, flags)
            except re.error as exc:
                raise ValueError(
                    f"RedactionRule regex pattern is invalid: {exc}"
                ) from exc

        body = re.escape(self.pattern)
        if self.whole_word:
            # \b is fine for ASCII; for accented characters we extend
            # with lookarounds so "Łukasz" or "François" still match.
            # Using a non-word-boundary lookaround so the rule still
            # matches at start/end of string.
            body = rf"(?<!\w){body}(?!\w)"
        return re.compile(body, flags)


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass
class _CompiledRule:
    """Internal: a rule paired with its compiled pattern."""

    rule: RedactionRule
    pattern: re.Pattern[str]


@dataclass
class RedactionPlan:
    """An ordered list of :class:`RedactionRule` plus optional metadata.

    Construct via :func:`build_redaction_plan` or directly. Use
    :meth:`apply` to redact a single string and :meth:`counts` to get
    per-rule substitution counts after a round of substitution.

    The plan **mutates its own counter** on each :meth:`apply`. This
    is deliberate: it lets us thread a single plan through every
    entity in the project and emit one combined manifest at the end.
    Callers that need a fresh count can call :meth:`reset_counts`.
    """

    rules: list[RedactionRule] = field(default_factory=list)
    # Optional human-readable label shown in the manifest header.
    note: str = ""

    # Internals — populated lazily on first apply().
    _compiled: list[_CompiledRule] | None = field(default=None, init=False, repr=False)
    _counts: list[int] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Compilation + counter management
    # ------------------------------------------------------------------ #

    def _ensure_compiled(self) -> None:
        if self._compiled is not None:
            return
        if len(self.rules) > MAX_RULES:
            raise ValueError(
                f"Plan must have ≤ {MAX_RULES} rules; got {len(self.rules)}"
            )
        self._compiled = [_CompiledRule(rule=r, pattern=r.compile()) for r in self.rules]
        self._counts = [0] * len(self._compiled)

    def reset_counts(self) -> None:
        """Zero the per-rule substitution counter."""
        if self._counts is not None:
            self._counts = [0] * len(self._counts)

    def counts(self) -> list[int]:
        """Per-rule substitution counts since construction or last reset.

        Same length as :attr:`rules`; zero for rules that haven't matched
        anything yet (or for an empty plan).
        """
        self._ensure_compiled()
        return list(self._counts or [])

    # ------------------------------------------------------------------ #
    # Apply
    # ------------------------------------------------------------------ #

    def apply(self, text: str) -> str:
        """Apply every rule, in order, to ``text``.

        Non-string inputs (None, int) are returned untouched after a
        ``str()`` coerce — the redactor never crashes mid-bundle on a
        weirdly-typed field. Empty strings are returned as-is without
        running any patterns.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return text
        self._ensure_compiled()
        assert self._compiled is not None and self._counts is not None
        out = text
        for i, cr in enumerate(self._compiled):
            new_out, n = cr.pattern.subn(cr.rule.replacement, out)
            if n:
                self._counts[i] += n
                out = new_out
        return out

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #

    def manifest(self) -> dict[str, Any]:
        """A safe-to-publish summary of the plan + match counts.

        Crucially, this **does not** include rule patterns (which are
        the original identifiers); only the replacement values + match
        counts.
        """
        self._ensure_compiled()
        assert self._counts is not None
        rows: list[dict[str, Any]] = []
        for i, r in enumerate(self.rules):
            rows.append(
                {
                    "replacement": r.replacement,
                    "case_insensitive": r.case_insensitive,
                    "whole_word": r.whole_word,
                    "regex": r.regex,
                    "match_count": self._counts[i],
                }
            )
        return {
            "note": self.note,
            "rule_count": len(self.rules),
            "total_substitutions": sum(self._counts),
            "rules": rows,
        }


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #


def _participant_rules(participants: Sequence[Participant]) -> list[RedactionRule]:
    """Generate ``name → pseudonym`` rules for every participant with a pseudonym.

    Participants without a non-empty ``pseudonym`` contribute no rule —
    we don't have a substitution to make and silently emitting their
    real name would be a footgun. Rules are emitted in **descending
    name length** so multi-word names ("Pat Smith") win over surnames
    ("Smith") when both are configured.
    """
    out: list[RedactionRule] = []
    seen_patterns: set[tuple[str, str]] = set()
    for p in participants:
        name = (p.name or "").strip()
        pseudonym = (p.pseudonym or "").strip()
        if not name or not pseudonym:
            continue
        if name == pseudonym:
            # Already anonymised; nothing to do.
            continue
        key = (name.lower(), pseudonym)
        if key in seen_patterns:
            continue
        seen_patterns.add(key)
        out.append(RedactionRule(pattern=name, replacement=pseudonym))
    out.sort(key=lambda r: len(r.pattern), reverse=True)
    return out


def _speaker_rules(
    speaker_maps: Sequence[SpeakerMap],
    participants_by_id: Mapping[str, Participant],
) -> tuple[list[RedactionRule], dict[str, str]]:
    """Generate rules for transcript speaker labels + display names.

    Returns ``(rules, label_to_pseudonym)``:

    * ``rules`` — one :class:`RedactionRule` per (label or display name)
      that maps to a known pseudonym.  Emitted longest-first so a
      display name like "Dr Pat Smith" wins over "Pat".
    * ``label_to_pseudonym`` — a flat ``label → pseudonym`` lookup the
      segment redactor uses to rewrite the ``speaker`` field directly
      (skipping the regex pass for that field). Keyed by the *raw*
      transcript label.

    Labels that resolve to a participant without a pseudonym
    contribute no rule — same reasoning as :func:`_participant_rules`.
    """
    rules: list[RedactionRule] = []
    seen_patterns: set[tuple[str, str]] = set()
    label_map: dict[str, str] = {}

    for sm in speaker_maps:
        for entry in sm.entries:
            label = (entry.label or "").strip()
            display = (entry.display_name or "").strip()
            pseudonym = ""
            if entry.participant_id:
                p = participants_by_id.get(entry.participant_id)
                if p:
                    pseudonym = (p.pseudonym or "").strip()
            if not pseudonym:
                continue
            if label:
                label_map[label] = pseudonym
            for original in (display, label):
                if not original:
                    continue
                if original == pseudonym:
                    continue
                key = (original.lower(), pseudonym)
                if key in seen_patterns:
                    continue
                seen_patterns.add(key)
                rules.append(
                    RedactionRule(pattern=original, replacement=pseudonym)
                )

    rules.sort(key=lambda r: len(r.pattern), reverse=True)
    return rules, label_map


def build_redaction_plan(
    *,
    participants: Sequence[Participant] = (),
    speaker_maps: Sequence[SpeakerMap] = (),
    custom_rules: Sequence[RedactionRule | Mapping[str, Any]] = (),
    note: str = "",
) -> tuple[RedactionPlan, dict[str, str]]:
    """Compose a :class:`RedactionPlan` from project entities + extras.

    Returns ``(plan, label_map)`` where ``label_map`` is the speaker
    label → pseudonym lookup used by :func:`redact_segments` to rewrite
    the per-segment ``speaker`` field directly. ``label_map`` is empty
    when no speaker maps were provided.

    Rule order:

      1. **Custom rules first** — researchers know things the data
         model can't see (the field site's name, an unlisted
         co-participant, an institution). They get the highest
         priority so a generic participant rule can't accidentally
         pre-empt them.
      2. **Speaker rules** — display names like "Dr Smith" want to
         win over the participant rule for "Smith".
      3. **Participant rules** — the project's recorded name list.
    """
    custom: list[RedactionRule] = []
    for r in custom_rules:
        if isinstance(r, RedactionRule):
            custom.append(r)
        elif isinstance(r, Mapping):
            custom.append(RedactionRule.from_dict(r))
        else:
            raise ValueError(
                "custom_rules entries must be RedactionRule or dict objects"
            )

    by_pid = {p.id: p for p in participants}
    speaker_rules, label_map = _speaker_rules(speaker_maps, by_pid)
    participant_rules = _participant_rules(participants)

    rules = custom + speaker_rules + participant_rules
    # Preserve order while de-duping identical (pattern, replacement,
    # flags) triples — a participant rule and a custom rule for the
    # same person would otherwise count their substitutions twice.
    seen: set[tuple] = set()
    deduped: list[RedactionRule] = []
    for r in rules:
        key = (r.pattern, r.replacement, r.case_insensitive, r.whole_word, r.regex)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    plan = RedactionPlan(rules=deduped, note=note)
    return plan, label_map


# --------------------------------------------------------------------------- #
# Per-entity transformations
# --------------------------------------------------------------------------- #


def redact_text(text: str, plan: RedactionPlan) -> str:
    """Run the plan over a free-text string. Convenience wrapper."""
    return plan.apply(text)


def _redact_words_in_segment(
    words: Sequence[Mapping[str, Any]],
    plan: RedactionPlan,
) -> list[dict[str, Any]]:
    """Redact the word list of a single segment.

    Multi-word patterns are why this can't just be a per-word map:
    ``"Jane Doe"`` won't match the word ``"Jane"`` alone with
    ``whole_word=True``. So we:

      1. Join the words with single spaces (the same join the QDPX
         renderer uses) to build a flat segment text.
      2. Apply the plan to the joined text.
      3. Split the redacted text back into words on whitespace.

    When the word count is preserved (the common case: single-token
    pseudonyms, literal-length swaps), each output word inherits its
    original word's timestamps and any extra fields. When the count
    changes (e.g. ``"Jane Doe"`` → ``"P01"``, or a regex that splits
    one word into many), we spread timestamps proportionally across
    the segment's time range so the SRT/VTT writers still produce
    monotonic output.
    """
    if not words:
        return []
    raw_words: list[Mapping[str, Any]] = [
        w for w in words if isinstance(w, Mapping)
    ]
    if not raw_words:
        return []

    texts = [str(w.get("text", "") or "") for w in raw_words]
    joined = " ".join(texts)
    redacted = plan.apply(joined)

    new_texts = [t for t in redacted.split(" ") if t]
    if not new_texts:
        return []

    if len(new_texts) == len(raw_words):
        # Common path: preserve word-level structure 1-1.
        out: list[dict[str, Any]] = []
        for w, new_text in zip(raw_words, new_texts):
            new_w = dict(w)
            new_w["text"] = new_text
            out.append(new_w)
        return out

    # Word count changed. Best-effort timestamp redistribution across
    # the segment's full time range. Anything missing falls back to
    # plain text-only words (downstream writers tolerate that).
    first_start = raw_words[0].get("start")
    last_end = raw_words[-1].get("end")
    if (
        not isinstance(first_start, (int, float))
        or not isinstance(last_end, (int, float))
        or last_end <= first_start
    ):
        return [{"text": t} for t in new_texts]

    span = float(last_end) - float(first_start)
    step = span / len(new_texts)
    out_resized: list[dict[str, Any]] = []
    for j, t in enumerate(new_texts):
        ws = float(first_start) + step * j
        we = float(first_start) + step * (j + 1)
        out_resized.append({"text": t, "start": ws, "end": we})
    return out_resized


def redact_segments(
    segments: Sequence[Mapping[str, Any]],
    plan: RedactionPlan,
    *,
    speaker_label_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return a deep-redacted copy of a Scribe-shaped segments list.

    Each output segment is a fresh dict with:

      * ``speaker`` — first replaced via ``speaker_label_map`` (raw
        label → pseudonym), then run through ``plan`` for any rules
        not captured by the map (e.g. a custom rule for an unmapped
        speaker label).
      * ``words`` — a fresh list of fresh dicts. The plan is applied
        at the *segment* level (joining word texts with single
        spaces) so multi-word patterns like ``"Jane Doe"`` redact
        correctly; words are then split back. See
        :func:`_redact_words_in_segment` for timestamp handling when
        the word count changes.
      * Any other top-level keys (``start``, ``end``, ``id``) are
        copied through unchanged so the writer can still emit SRT/VTT
        from a redacted transcript.

    Inputs that aren't ``Mapping`` are skipped silently so a stray
    None in the list can't crash a bundle export.
    """
    label_map = dict(speaker_label_map or {})
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue
        new_seg: dict[str, Any] = dict(seg)

        raw_speaker = seg.get("speaker") or ""
        if isinstance(raw_speaker, str) and raw_speaker:
            mapped = label_map.get(raw_speaker.strip(), raw_speaker)
            new_seg["speaker"] = plan.apply(mapped)
        else:
            new_seg["speaker"] = ""

        words = seg.get("words")
        if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
            new_seg["words"] = _redact_words_in_segment(words, plan)
        else:
            new_seg["words"] = []

        # Some segments carry a ``text`` summary; redact it too.
        if "text" in seg and isinstance(seg.get("text"), str):
            new_seg["text"] = plan.apply(seg["text"])

        out.append(new_seg)
    return out


def redact_source(source: Source, plan: RedactionPlan) -> Source:
    """Return a redacted copy of a :class:`Source`.

    Redacts ``name``, ``notes``, and every value in ``custom_attributes``.
    Keys are left untouched (they're the schema, not the data; F3.2
    bounds them with a strict regex anyway). Required-field validation
    is preserved: an empty redacted name falls back to a stable
    ``"<source-redacted>"`` label so the entity still passes
    :meth:`Source.validate`.
    """
    new_name = plan.apply(source.name) or "<source-redacted>"
    new_notes = plan.apply(source.notes) if source.notes else ""
    new_attrs = {
        k: plan.apply(v) for k, v in source.custom_attributes.items()
    }
    return replace(
        source,
        name=new_name,
        notes=new_notes,
        custom_attributes=new_attrs,
    )


def redact_code(code: Code, plan: RedactionPlan) -> Code:
    """Return a redacted copy of a :class:`Code`.

    The code's ``name`` is intentionally redacted too: researchers
    sometimes name a code after a participant ("Pat's pacing
    strategy"). Required-field validation is preserved with a
    fallback label.

    ``related_codes`` (typed cross-references) and ``provenance`` are
    structural and pass through untouched.
    """
    new_name = plan.apply(code.name) or "<code-redacted>"
    new_exemplars = [plan.apply(e) for e in code.exemplars]
    return replace(
        code,
        name=new_name,
        definition=plan.apply(code.definition),
        inclusion_criteria=plan.apply(code.inclusion_criteria),
        exclusion_criteria=plan.apply(code.exclusion_criteria),
        exemplars=new_exemplars,
        theoretical_memo=plan.apply(code.theoretical_memo),
    )


def redact_memo(memo: Memo, plan: RedactionPlan) -> Memo:
    """Return a redacted copy of a :class:`Memo`.

    Title, body, and tag values are run through ``plan``. Links
    (target ids, target types, link types) are structural and pass
    through; provenance (AI invocation refs) ditto.
    """
    new_links = [
        MemoLink(
            target_type=ln.target_type,
            target_id=ln.target_id,
            role=ln.role,
        )
        for ln in memo.links
    ]
    new_tags = [plan.apply(t) for t in memo.tags]
    return replace(
        memo,
        title=plan.apply(memo.title),
        body=plan.apply(memo.body),
        tags=new_tags,
        links=new_links,
    )


def redact_coder(coder: Coder, plan: RedactionPlan) -> Coder:
    """Return a redacted copy of a :class:`Coder`.

    Coder names sometimes carry initials of real researchers. We
    redact them so an ICR table doesn't out a research assistant.
    """
    new_name = plan.apply(coder.name) or "<coder-redacted>"
    return replace(coder, name=new_name)


def redact_project(project: Project, plan: RedactionPlan) -> Project:
    """Return a redacted copy of a :class:`Project`.

    The ``name``, ``research_question``, ``description``, and every
    sensitising concept run through ``plan``. ``methodology`` and
    ``codebook_stage`` are vocabulary-bounded and pass through.
    """
    new_name = plan.apply(project.name) or "<project-redacted>"
    new_concepts = [plan.apply(c) for c in project.sensitising_concepts]
    return replace(
        project,
        name=new_name,
        research_question=plan.apply(project.research_question),
        description=plan.apply(project.description),
        sensitising_concepts=new_concepts,
    )


# --------------------------------------------------------------------------- #
# Top-level orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class AnonymisedBundle:
    """Result of :func:`build_anonymised_qdpx`.

    ``archive`` is the bytes of the QDPX zip; ``manifest`` is the
    structured per-rule match-count summary already embedded in the
    archive at ``Redactions/manifest.json`` (returned here so a CLI
    caller can print it without re-opening the zip).
    """

    archive: bytes
    manifest: dict[str, Any]


def build_anonymised_qdpx(
    *,
    project: Project,
    sources: Sequence[Source] = (),
    codes: Sequence[Code] = (),
    applications: Sequence[Application] = (),
    memos: Sequence[Memo] = (),
    coders: Sequence[Coder] = (),
    participants: Sequence[Participant] = (),
    speaker_maps: Sequence[SpeakerMap] = (),
    segments_by_source_id: Mapping[str, Sequence[Mapping[str, Any]]] = {},
    custom_rules: Sequence[RedactionRule | Mapping[str, Any]] = (),
    plan: RedactionPlan | None = None,
    speaker_label_map: Mapping[str, str] | None = None,
    note: str = "",
    origin: str = REFI_QDA_PROJECT_ORIGIN_DEFAULT,
    now: str | None = None,
) -> AnonymisedBundle:
    """Build a redacted QDPX bundle.

    Either pass an already-built ``plan`` + ``speaker_label_map`` (e.g.
    when the caller wants to inspect / mutate them first), or let this
    function compose one from ``participants`` + ``speaker_maps`` +
    ``custom_rules``.

    ``segments_by_source_id`` is the raw transcript segments looked up
    by the caller (typically from ``outputs/<job>/edited.json``). A
    source whose id is missing from this map is included in the
    bundle but without selections — same fall-back behaviour as
    :func:`scribe.refi_qda_project.to_qdpx`.

    ``note`` is a free-form annotation stamped onto the bundled
    manifest's header. Useful for "Pre-publication anon pass" or
    similar provenance flags.

    The returned bundle includes a top-level ``Redactions/manifest.json``
    file alongside the standard QDPX layout. The manifest is a safe
    artifact (replacement values + match counts only — never the
    original identifiers).
    """
    if plan is None:
        plan, default_label_map = build_redaction_plan(
            participants=participants,
            speaker_maps=speaker_maps,
            custom_rules=custom_rules,
            note=note,
        )
    else:
        default_label_map = {}
        if note and not plan.note:
            plan.note = note
    if speaker_label_map is None:
        speaker_label_map = default_label_map

    # ------------------------------------------------------------------ #
    # Apply the plan to every entity. We deliberately redact in this
    # order: project last (so the plan's counts include every
    # contribution); transcript segments before sources (so the
    # speaker rewriter has a chance to populate label_map).
    # ------------------------------------------------------------------ #

    redacted_segments: dict[str, list[dict[str, Any]]] = {}
    rendered_sources: list[RenderedSource] = []
    for s in sources:
        segs = segments_by_source_id.get(s.id)
        if segs is None:
            continue
        new_segs = redact_segments(
            segs, plan, speaker_label_map=speaker_label_map
        )
        redacted_segments[s.id] = new_segs
        rendered_sources.append(render_source_plain_text(s.id, new_segs))

    new_sources = [redact_source(s, plan) for s in sources]
    new_codes = [redact_code(c, plan) for c in codes]
    new_memos = [redact_memo(m, plan) for m in memos]
    new_coders = [redact_coder(c, plan) for c in coders]
    new_project = redact_project(project, plan)

    # ------------------------------------------------------------------ #
    # Bundle. We call the existing QDPX writer for the standard layout
    # and then re-zip with our manifest spliced in. Cheaper than
    # forking the writer, and keeps the write path single-source.
    # ------------------------------------------------------------------ #

    base = to_qdpx(
        project=new_project,
        sources=new_sources,
        codes=new_codes,
        applications=list(applications),  # Applications carry no free text
        memos=new_memos,
        coders=new_coders,
        rendered_sources=rendered_sources,
        origin=origin,
        now=now,
    )

    manifest = plan.manifest()
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False).encode(
        "utf-8"
    )

    out_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(base), mode="r") as src_zip:
        with zipfile.ZipFile(
            out_buf, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as dst_zip:
            for info in src_zip.infolist():
                dst_zip.writestr(info, src_zip.read(info.filename))
            dst_zip.writestr("Redactions/manifest.json", manifest_json)

    return AnonymisedBundle(archive=out_buf.getvalue(), manifest=manifest)


__all__ = [
    "MAX_RULES",
    "MAX_RULE_PATTERN_LEN",
    "MAX_RULE_REPLACEMENT_LEN",
    "AnonymisedBundle",
    "RedactionPlan",
    "RedactionRule",
    "build_anonymised_qdpx",
    "build_redaction_plan",
    "redact_code",
    "redact_coder",
    "redact_memo",
    "redact_project",
    "redact_segments",
    "redact_source",
    "redact_text",
]
