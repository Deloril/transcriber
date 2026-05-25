"""AI memo-draft engine for the academic-coding workflow (F8.8).

Per PLANNING.md F8.8:

  > Memo-draft action on a code (LLM seeds with the code's exemplars;
  > researcher rewrites).

A *memo* (F5.1) is the analytic-thinking layer of grounded theory —
running notes a researcher writes about what a category *means*. F8.8
gives the researcher a one-click jump-start: pick a code, click
"Draft a memo", and the LLM produces a first-draft body seeded by the
code's own exemplars (and optionally by the actual quoted text of the
code's existing applications). The researcher then rewrites it. The
draft is **never auto-saved as a memo** — there's an explicit accept /
modify / reject lifecycle so the audit trail (F9.6) carries every
invocation, including rejected ones.

What this module does
---------------------

1. **Collect seed material.** Given a :class:`scribe.codes.Code`, the
   engine assembles a ranked list of seed snippets:

   * Every non-empty entry from ``code.exemplars`` (the code's own
     curated examples; highest priority).
   * Optionally, the quoted text of each existing
     :class:`scribe.applications.Application` of the code, pulled
     from a caller-supplied segments map. This lets the LLM ground
     its draft in *real* coded segments, not just the exemplar list
     the researcher pre-curated. Best-of-both: an active code with
     dozens of applications gets a richer seed; a brand-new code with
     just two exemplars still gets a useful draft.

   The combined list is canonicalised, deduped, and capped at
   ``max_seed_snippets`` so the prompt stays cheap.

2. **Build a structured prompt.** :func:`make_memo_draft_prompt`
   renders a Charmaz-friendly prompt that asks for a first-draft memo
   body with a short title and a one-sentence rationale. The prompt
   makes it clear the output is a *starting point* for the researcher
   to rewrite — not a finished artefact.

3. **Parse the response.** :func:`parse_memo_draft_response` is
   tolerant: strict JSON, fenced JSON, or "Here you go:"-prefixed
   JSON all parse cleanly. On hard failure the response degrades
   into a single-field draft with the raw text as the body, so the
   researcher still gets something to edit instead of a blank page.

4. **Persist a :class:`MemoDraft`.** Each invocation produces a
   record at ``projects/<pid>/memo_drafts/<did>.json``. The decision
   lifecycle ``pending → accepted | modified | rejected`` mirrors
   F8.3 / F8.4. Even rejected drafts are kept (F9.6 wants them).

5. **Promote on accept.** :func:`promote_memo_draft_to_memo` takes
   an accepted (or modified) :class:`MemoDraft` and creates an
   actual :class:`scribe.memos.Memo` via the F5.1 path, with
   ``provenance['source'] = 'ai_drafted'`` and a back-link to the
   source code. The decision recorder stamps the new memo's id on
   the draft so the audit trail closes.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.8 is the engine; the
  ``/api/projects/<id>/memo-drafts`` routes are deferred and will
  be a thin shell over this module, mirroring the F8.3 / F8.5 / F8.7
  split.
* **No automatic memo creation on draft.** Drafts start as records;
  only an explicit ``promote_memo_draft_to_memo`` call creates a
  :class:`Memo`. The decision recorder just stamps the audit trail.
* **Pure callable.** ``generate_fn`` matches the F8.1 / F8.3 shape so
  the same backend adapter drives every engine. F8.8 doesn't need an
  embedding model — the seed material is selected by code identity,
  not by embedding similarity.

This module is stand-alone — no FastAPI, no engine imports — so the
data model can be tested in pure Python and reused by the CLI later.
Conventions match the rest of the F-feature stack
(:mod:`scribe.code_suggestions`, :mod:`scribe.new_code_suggestions`,
:mod:`scribe.memos`, :mod:`scribe.memo_promote`).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .applications import APPLICATION_ID_RE, Application
from .application_reanchor import anchored_words
from .codes import CODE_ID_RE, Code
from .coders import CODER_ID_RE
from .embedding_index import canonical_text
from .memos import (
    MAX_BODY_LEN as MEMO_MAX_BODY_LEN,
    MAX_TITLE_LEN as MEMO_MAX_TITLE_LEN,
    MEMO_ID_RE,
    MEMO_TYPES,
    Memo,
    MemoLink,
    save_memo,
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


# Memo-draft IDs follow the same 12-char hex shape as every other id.
MEMO_DRAFT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory under ``projects/<id>/`` holding draft records.
MEMO_DRAFTS_DIRNAME = "memo_drafts"

# Decision lifecycle states. Same shape as F8.3 / F8.4 so a UI can
# render every "AI suggested X" list with the same row component.
MEMO_DRAFT_DECISION_PENDING = "pending"
MEMO_DRAFT_DECISION_ACCEPTED = "accepted"
MEMO_DRAFT_DECISION_MODIFIED = "modified"
MEMO_DRAFT_DECISION_REJECTED = "rejected"
MEMO_DRAFT_DECISIONS: tuple[str, ...] = (
    MEMO_DRAFT_DECISION_PENDING,
    MEMO_DRAFT_DECISION_ACCEPTED,
    MEMO_DRAFT_DECISION_MODIFIED,
    MEMO_DRAFT_DECISION_REJECTED,
)
TERMINAL_MEMO_DRAFT_DECISIONS: frozenset[str] = frozenset(
    {
        MEMO_DRAFT_DECISION_ACCEPTED,
        MEMO_DRAFT_DECISION_MODIFIED,
        MEMO_DRAFT_DECISION_REJECTED,
    }
)

# Provenance source recorded on the promoted Memo. Matches
# :data:`scribe.memos.MEMO_PROVENANCE_SOURCES`.
MEMO_DRAFT_PROVENANCE_SOURCE = "ai_drafted"

# Closed vocabulary for the "kind" of seed snippet used. Lets a UI
# render the source of each snippet ("from exemplar 2", "from coded
# segment ab12...") instead of just an opaque blob of text.
SEED_KIND_EXEMPLAR = "exemplar"
SEED_KIND_APPLICATION = "application"
SEED_KINDS: tuple[str, ...] = (SEED_KIND_EXEMPLAR, SEED_KIND_APPLICATION)

# Defaults. Tuned for "click a code, get a one-page draft".
DEFAULT_MAX_SEED_SNIPPETS = 12       # cap on snippets passed to the prompt
DEFAULT_INCLUDE_APPLICATIONS = True  # mix in real coded text by default
DEFAULT_MEMO_TYPE = "theoretical"    # the typical use-case is a theoretical memo

# Field-length / cardinality bounds. Generous, but bounded so a buggy
# model can't write a 50 MB draft record.
MAX_TITLE_LEN = MEMO_MAX_TITLE_LEN          # 200
MAX_BODY_LEN = MEMO_MAX_BODY_LEN            # 64 KiB
MAX_RATIONALE_LEN = 2000
MAX_SEED_SNIPPET_LEN = 2000
MAX_SEED_SNIPPETS_PERSISTED = 64
MAX_RAW_LLM_RESPONSE_LEN = 16 * 1024
MAX_PROMPT_LEN = 32 * 1024
MAX_REJECTION_REASON_LEN = 2000
MAX_NOTES_LEN = 4000

# Allowed callable signature. Matches F8.3 / F8.4 / F8.6.
GenerateFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Helper data classes
# --------------------------------------------------------------------------- #


@dataclass
class SeedSnippet:
    """One piece of evidence fed to the LLM as draft seed material.

    ``ref`` is interpreted relative to ``kind``:

      * ``exemplar`` — the exemplar's index in ``code.exemplars`` as a
        string (e.g. ``"3"``).
      * ``application`` — the 12-char hex application id whose anchored
        text became the snippet.

    The text is **canonicalised** (single-spaced, no leading/trailing
    whitespace) so two calls on the same code produce identical
    snippets — handy for caching and for the audit trail.
    """

    kind: str
    ref: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "text": self.text}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SeedSnippet":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "SeedSnippet payload must be an object"
            )
        s = cls(
            kind=str(d.get("kind", "") or ""),
            ref=str(d.get("ref", "") or ""),
            text=str(d.get("text", "") or ""),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.kind not in SEED_KINDS:
            raise ProjectValidationError(
                f"SeedSnippet.kind must be one of {SEED_KINDS}; "
                f"got {self.kind!r}"
            )
        if not isinstance(self.ref, str) or not self.ref:
            raise ProjectValidationError(
                "SeedSnippet.ref must be a non-empty string"
            )
        if self.kind == SEED_KIND_APPLICATION and not APPLICATION_ID_RE.match(
            self.ref
        ):
            raise ProjectValidationError(
                f"SeedSnippet.ref must be a 12-char hex application id "
                f"when kind=application; got {self.ref!r}"
            )
        if self.kind == SEED_KIND_EXEMPLAR:
            try:
                idx = int(self.ref)
            except ValueError as e:
                raise ProjectValidationError(
                    f"SeedSnippet.ref must be a non-negative integer "
                    f"when kind=exemplar; got {self.ref!r}"
                ) from e
            if idx < 0:
                raise ProjectValidationError(
                    f"SeedSnippet.ref must be ≥ 0 when kind=exemplar; "
                    f"got {self.ref!r}"
                )
        if not isinstance(self.text, str):
            raise ProjectValidationError("SeedSnippet.text must be a string")
        if not self.text:
            raise ProjectValidationError("SeedSnippet.text must be non-empty")
        if len(self.text) > MAX_SEED_SNIPPET_LEN:
            raise ProjectValidationError(
                f"SeedSnippet.text exceeds {MAX_SEED_SNIPPET_LEN} chars"
            )


# --------------------------------------------------------------------------- #
# Draft record
# --------------------------------------------------------------------------- #


@dataclass
class MemoDraft:
    """One AI memo-draft invocation, persisted as the audit record.

    The draft is *the LLM's first try*; the researcher's eventual memo
    is a separate :class:`scribe.memos.Memo` minted via
    :func:`promote_memo_draft_to_memo`. Decision lifecycle::

        pending → accepted | modified | rejected

    * ``accepted`` — the researcher accepted the draft as-is. The
      memo created from it gets ``accepted_memo_id``.
    * ``modified`` — the researcher edited the draft before saving.
      The memo's body diverges from ``body``; both records are kept
      so the audit trail can show "AI drafted X, human saved Y".
    * ``rejected`` — neither the body nor the title was useful.
      Forbids ``accepted_memo_id``; accepts a free-text reason.

    Anchor fields *deliberately* don't exist here: an F8.8 draft is
    grounded in a *code*, not a transcript span. The seed-material
    snippets carry their own per-source references already.
    """

    id: str
    project_id: str
    code_id: str
    memo_type: str
    title: str
    body: str
    rationale: str
    seed_snippets: list[SeedSnippet] = field(default_factory=list)
    generation_model: str = ""
    decision: str = MEMO_DRAFT_DECISION_PENDING
    decided_at: str = ""
    decided_by_coder_id: str = ""
    accepted_memo_id: str | None = None
    rejection_reason: str = ""
    notes: str = ""
    prompt: str = ""
    raw_llm_response: str = ""
    created_at: str = ""
    modified_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        code_id: str,
        memo_type: str = DEFAULT_MEMO_TYPE,
        title: str = "",
        body: str = "",
        rationale: str = "",
        seed_snippets: Iterable[SeedSnippet | Mapping[str, Any]] | None = None,
        generation_model: str = "",
        prompt: str = "",
        raw_llm_response: str = "",
        notes: str = "",
        draft_id: str | None = None,
        now: str | None = None,
    ) -> "MemoDraft":
        ts = now or utcnow_iso()
        coerced: list[SeedSnippet] = []
        for s in seed_snippets or ():
            if isinstance(s, SeedSnippet):
                coerced.append(s)
            else:
                coerced.append(SeedSnippet.from_dict(s))
        d = cls(
            id=draft_id or new_memo_draft_id(),
            project_id=project_id,
            code_id=code_id,
            memo_type=memo_type,
            title=str(title or ""),
            body=str(body or ""),
            rationale=str(rationale or ""),
            seed_snippets=coerced,
            generation_model=str(generation_model or ""),
            decision=MEMO_DRAFT_DECISION_PENDING,
            decided_at="",
            decided_by_coder_id="",
            accepted_memo_id=None,
            rejection_reason="",
            notes=str(notes or ""),
            prompt=str(prompt or ""),
            raw_llm_response=str(raw_llm_response or ""),
            created_at=ts,
            modified_at=ts,
        )
        d.validate()
        return d

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "code_id": self.code_id,
            "memo_type": self.memo_type,
            "title": self.title,
            "body": self.body,
            "rationale": self.rationale,
            "seed_snippets": [s.to_dict() for s in self.seed_snippets],
            "generation_model": self.generation_model,
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by_coder_id": self.decided_by_coder_id,
            "accepted_memo_id": self.accepted_memo_id,
            "rejection_reason": self.rejection_reason,
            "notes": self.notes,
            "prompt": self.prompt,
            "raw_llm_response": self.raw_llm_response,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoDraft":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "MemoDraft payload must be an object"
            )
        for required in ("id", "project_id", "code_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"MemoDraft payload missing required key: {required}"
                )
        snippets_raw = d.get("seed_snippets") or []
        if not isinstance(snippets_raw, list):
            raise ProjectValidationError(
                "seed_snippets must be a list of objects"
            )
        m = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            code_id=str(d["code_id"]),
            memo_type=str(d.get("memo_type", DEFAULT_MEMO_TYPE)
                          or DEFAULT_MEMO_TYPE),
            title=str(d.get("title", "") or ""),
            body=str(d.get("body", "") or ""),
            rationale=str(d.get("rationale", "") or ""),
            seed_snippets=[SeedSnippet.from_dict(s) for s in snippets_raw],
            generation_model=str(d.get("generation_model", "") or ""),
            decision=str(d.get("decision", MEMO_DRAFT_DECISION_PENDING)
                         or MEMO_DRAFT_DECISION_PENDING),
            decided_at=str(d.get("decided_at", "") or ""),
            decided_by_coder_id=str(d.get("decided_by_coder_id", "") or ""),
            accepted_memo_id=(
                str(d["accepted_memo_id"])
                if d.get("accepted_memo_id")
                else None
            ),
            rejection_reason=str(d.get("rejection_reason", "") or ""),
            notes=str(d.get("notes", "") or ""),
            prompt=str(d.get("prompt", "") or ""),
            raw_llm_response=str(d.get("raw_llm_response", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not MEMO_DRAFT_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid memo-draft id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"Invalid code id: {self.code_id!r}"
            )
        if self.memo_type not in MEMO_TYPES:
            raise ProjectValidationError(
                f"memo_type must be one of {MEMO_TYPES}; "
                f"got {self.memo_type!r}"
            )

        if not isinstance(self.title, str):
            raise ProjectValidationError("title must be a string")
        title = self.title.strip()
        if len(title) > MAX_TITLE_LEN:
            raise ProjectValidationError(
                f"title must be ≤ {MAX_TITLE_LEN} chars"
            )
        # Persist trimmed.
        self.title = title

        if not isinstance(self.body, str):
            raise ProjectValidationError("body must be a string")
        if len(self.body) > MAX_BODY_LEN:
            raise ProjectValidationError(
                f"body must be ≤ {MAX_BODY_LEN} chars"
            )

        if not isinstance(self.rationale, str):
            raise ProjectValidationError("rationale must be a string")
        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise ProjectValidationError(
                f"rationale must be ≤ {MAX_RATIONALE_LEN} chars"
            )

        if not isinstance(self.seed_snippets, list):
            raise ProjectValidationError("seed_snippets must be a list")
        if len(self.seed_snippets) > MAX_SEED_SNIPPETS_PERSISTED:
            raise ProjectValidationError(
                f"seed_snippets exceeds {MAX_SEED_SNIPPETS_PERSISTED} entries"
            )
        for s in self.seed_snippets:
            s.validate()

        if (
            not isinstance(self.generation_model, str)
            or len(self.generation_model) > 256
        ):
            raise ProjectValidationError(
                "generation_model must be a string ≤ 256 chars"
            )

        if self.decision not in MEMO_DRAFT_DECISIONS:
            raise ProjectValidationError(
                f"decision must be one of {MEMO_DRAFT_DECISIONS}; "
                f"got {self.decision!r}"
            )
        if self.decision in TERMINAL_MEMO_DRAFT_DECISIONS:
            if not self.decided_at:
                raise ProjectValidationError(
                    f"decided_at must be set when decision is {self.decision!r}"
                )
            if not self.decided_by_coder_id or not CODER_ID_RE.match(
                self.decided_by_coder_id
            ):
                raise ProjectValidationError(
                    f"decided_by_coder_id must be a 12-char hex coder id "
                    f"when decision is {self.decision!r}"
                )
        if self.accepted_memo_id is not None and not MEMO_ID_RE.match(
            self.accepted_memo_id
        ):
            raise ProjectValidationError(
                f"accepted_memo_id must be 12-char hex or null; "
                f"got {self.accepted_memo_id!r}"
            )
        if self.decision == MEMO_DRAFT_DECISION_REJECTED:
            if self.accepted_memo_id:
                raise ProjectValidationError(
                    "rejected drafts must not record an accepted memo id"
                )
        if len(self.rejection_reason) > MAX_REJECTION_REASON_LEN:
            raise ProjectValidationError(
                f"rejection_reason exceeds {MAX_REJECTION_REASON_LEN} chars"
            )
        if len(self.notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )
        if len(self.prompt) > MAX_PROMPT_LEN:
            raise ProjectValidationError(
                f"prompt exceeds {MAX_PROMPT_LEN} chars"
            )
        if len(self.raw_llm_response) > MAX_RAW_LLM_RESPONSE_LEN:
            raise ProjectValidationError(
                f"raw_llm_response exceeds {MAX_RAW_LLM_RESPONSE_LEN} chars"
            )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: Mapping[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update in place. Mirrors F8.3's apply_update.

        Only ``notes``, ``rejection_reason``, ``accepted_memo_id``,
        ``title``, and ``body`` can be patched freely. The last two
        let a UI surface a "before promoting, edit the draft" affordance
        — handy on the ``modified`` path. Decision transitions go
        through :func:`record_memo_draft_decision`.
        """
        if not isinstance(patch, Mapping):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "notes" in patch:
            self.notes = str(patch["notes"] or "")
        if "rejection_reason" in patch:
            self.rejection_reason = str(patch["rejection_reason"] or "")
        if "accepted_memo_id" in patch:
            v = patch["accepted_memo_id"]
            self.accepted_memo_id = str(v) if v else None
        if "title" in patch:
            self.title = str(patch["title"] or "")
        if "body" in patch:
            self.body = str(patch["body"] or "")
        self.validate()
        self.modified_at = now or utcnow_iso()


_ALLOWED_PATCH_KEYS = {
    "notes",
    "rejection_reason",
    "accepted_memo_id",
    "title",
    "body",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_memo_draft_id() -> str:
    """Mint a new 12-char hex memo-draft id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Seed-material collection
# --------------------------------------------------------------------------- #


def collect_seed_snippets(
    *,
    code: Code,
    applications: Sequence[Application] = (),
    segments_by_source: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    include_applications: bool = DEFAULT_INCLUDE_APPLICATIONS,
    max_snippets: int = DEFAULT_MAX_SEED_SNIPPETS,
) -> list[SeedSnippet]:
    """Build the LLM-input seed list for a draft.

    Order:

      1. Every non-empty entry from ``code.exemplars`` (in original
         order; the ``ref`` records the post-validation index, since
         ``Code.validate`` strips empty exemplars at the entity layer
         before they reach this helper).
      2. Every application of this code (matched by ``code.id``) whose
         anchored text resolves cleanly. Applications are walked in the
         order they appear in ``applications``; the caller decides the
         sort (the F8.x stack typically uses created-at ascending).

    Empty / whitespace-only snippets are dropped. Duplicates (same
    canonicalised text) are deduped — researchers often enter the same
    quote twice as both an exemplar and the first application.

    Truncated to ``max_snippets`` (capped at
    :data:`MAX_SEED_SNIPPETS_PERSISTED` regardless of caller choice).
    """
    if not isinstance(code, Code):
        raise TypeError("collect_seed_snippets expects a Code")
    cap = max(1, min(int(max_snippets), MAX_SEED_SNIPPETS_PERSISTED))

    out: list[SeedSnippet] = []
    seen_text: set[str] = set()

    for i, raw in enumerate(code.exemplars):
        t = canonical_text(raw)
        if not t:
            continue
        if t in seen_text:
            continue
        if len(t) > MAX_SEED_SNIPPET_LEN:
            t = t[:MAX_SEED_SNIPPET_LEN]
            # Re-canonicalise after slicing to avoid stranded whitespace.
            t = canonical_text(t)
            if not t or t in seen_text:
                continue
        seen_text.add(t)
        out.append(SeedSnippet(
            kind=SEED_KIND_EXEMPLAR,
            ref=str(i),
            text=t,
        ))
        if len(out) >= cap:
            return out

    if include_applications and segments_by_source is not None:
        for app in applications:
            if app.code_id != code.id:
                continue
            segs = segments_by_source.get(app.source_id)
            if not segs:
                continue
            words = anchored_words(app, segs)
            if not words:
                continue
            t = canonical_text(" ".join(w for w in words if w))
            if not t:
                continue
            if t in seen_text:
                continue
            if len(t) > MAX_SEED_SNIPPET_LEN:
                t = t[:MAX_SEED_SNIPPET_LEN]
                t = canonical_text(t)
                if not t or t in seen_text:
                    continue
            seen_text.add(t)
            out.append(SeedSnippet(
                kind=SEED_KIND_APPLICATION,
                ref=app.id,
                text=t,
            ))
            if len(out) >= cap:
                break
    return out


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


# Strict JSON; explicit "this is a starting point, not a finished
# memo" instruction so the model doesn't over-claim authority.
MEMO_DRAFT_PROMPT_TEMPLATE = (
    "You are assisting a qualitative researcher with thematic coding "
    "of an interview corpus. Their methodology favours constructivist "
    "grounded theory (Charmaz). They have a code in their codebook and "
    "they want a FIRST-DRAFT analytic memo about it — a starting point "
    "for them to rewrite, not a finished artefact.\n"
    "\n"
    "Code name: {code_name}\n"
    "Code definition: {code_definition}\n"
    "{inclusion_block}{exclusion_block}{existing_memo_block}"
    "Seed quotes / exemplars (the basis for the draft):\n"
    "{snippets_block}\n"
    "\n"
    "Guidance for the memo body:\n"
    "- Write in the researcher's analytic voice, in the first person "
    "singular (\"I notice…\"). Memos are private analytic notes.\n"
    "- Surface tensions, edges, and questions the seed quotes raise — "
    "not just a summary.\n"
    "- Cite the seed quotes by their text (short snippets in quotes), "
    "not by index.\n"
    "- 200–500 words; concise rather than exhaustive.\n"
    "- Markdown is fine; no headings deeper than ##.\n"
    "\n"
    "Respond with strict JSON only — a single object with keys:\n"
    '  "title":     short string (≤ 12 words) summarising the memo\n'
    '  "body":      the draft memo, Markdown\n'
    '  "rationale": one sentence describing why these seed quotes '
    "support the draft\n"
    "\n"
    "If the seed material is too thin to ground a memo, respond with an "
    'object whose "body" is the empty string. Do not invent quotes.'
)


def make_memo_draft_prompt(
    *,
    code: Code,
    seed_snippets: Sequence[SeedSnippet],
    include_existing_memo: bool = True,
) -> str:
    """Render the prompt sent to the generation backend.

    ``include_existing_memo`` controls whether the code's
    ``theoretical_memo`` (F2.1) is included as background context.
    Default ``True`` — if the researcher wrote a stub memo by hand,
    the LLM should anchor on it; if they didn't, the block is omitted.
    """
    if not isinstance(code, Code):
        raise TypeError("make_memo_draft_prompt expects a Code")
    name = canonical_text(code.name) or "(unnamed)"
    defn = canonical_text(code.definition) or "(no definition)"
    incl = canonical_text(code.inclusion_criteria)
    excl = canonical_text(code.exclusion_criteria)
    existing = canonical_text(code.theoretical_memo) if include_existing_memo else ""

    inclusion_block = (
        f"Inclusion criteria: {incl}\n" if incl else ""
    )
    exclusion_block = (
        f"Exclusion criteria: {excl}\n" if excl else ""
    )
    if existing:
        existing_memo_block = (
            f"Existing memo / notes (anchor on this if useful):\n"
            f"\"\"\"\n{existing}\n\"\"\"\n\n"
        )
    else:
        existing_memo_block = ""

    if seed_snippets:
        lines: list[str] = []
        for i, s in enumerate(seed_snippets, start=1):
            lines.append(f"{i}. \"{s.text}\"")
        snippets_block = "\n".join(lines)
    else:
        snippets_block = "(no seed material; the code has no exemplars or applications)"

    return MEMO_DRAFT_PROMPT_TEMPLATE.format(
        code_name=name,
        code_definition=defn,
        inclusion_block=inclusion_block,
        exclusion_block=exclusion_block,
        existing_memo_block=existing_memo_block,
        snippets_block=snippets_block,
    )


# --------------------------------------------------------------------------- #
# Response parser
# --------------------------------------------------------------------------- #


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParsedDraft:
    """Result of parsing the LLM response into structured fields."""

    title: str
    body: str
    rationale: str


def parse_memo_draft_response(response_text: str) -> ParsedDraft:
    """Parse the model's JSON response into a structured draft.

    Tolerant of:

    * Plain JSON objects.
    * JSON wrapped in a ``​```json ... ``​``` fence.
    * Models that prefix the JSON with a "Sure! Here you go:" line —
      the helper extracts the largest ``{ ... }`` slice and parses
      that.

    Hard-failure fallback: if no JSON parses, the entire response text
    is returned as ``body`` (truncated to :data:`MAX_BODY_LEN`) so the
    researcher gets *something* to edit instead of a blank page. Title
    and rationale fall back to empty in that case.

    All fields are truncated to module-level limits.
    """
    text = (response_text or "").strip()
    if not text:
        return ParsedDraft(title="", body="", rationale="")

    candidates: list[str] = []
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    lb = text.find("{")
    rb = text.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        candidates.append(text[lb : rb + 1])
    candidates.append(text)

    parsed: Mapping[str, Any] | None = None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            parsed = obj
            break

    if parsed is None:
        # Fallback: hand the raw response back as the body so the user
        # has somewhere to start editing.
        body = text[:MAX_BODY_LEN]
        return ParsedDraft(title="", body=body, rationale="")

    title = _trim_field(parsed.get("title"), MAX_TITLE_LEN)
    body = _trim_field(parsed.get("body"), MAX_BODY_LEN)
    rationale = _trim_field(parsed.get("rationale"), MAX_RATIONALE_LEN)
    return ParsedDraft(title=title, body=body, rationale=rationale)


def _trim_field(v: Any, cap: int) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    if len(v) > cap:
        v = v[:cap]
    return v


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def draft_memo_for_code(
    *,
    project_id: str,
    code: Code,
    generate_fn: GenerateFn,
    applications: Sequence[Application] = (),
    segments_by_source: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    memo_type: str = DEFAULT_MEMO_TYPE,
    include_applications: bool = DEFAULT_INCLUDE_APPLICATIONS,
    include_existing_memo: bool = True,
    max_seed_snippets: int = DEFAULT_MAX_SEED_SNIPPETS,
    generation_model: str = "",
    now: str | None = None,
) -> MemoDraft:
    """End-to-end: build a :class:`MemoDraft` for a code.

    Workflow:

      1. Collect seed snippets (:func:`collect_seed_snippets`).
      2. Build the prompt (:func:`make_memo_draft_prompt`).
      3. Call ``generate_fn``; parse the response
         (:func:`parse_memo_draft_response`).
      4. Wrap in a :class:`MemoDraft` with ``decision="pending"``.

    Caller persists the result via :func:`save_memo_draft`. The
    :class:`MemoDraft` is returned even if the model produced an
    empty body — the audit trail (F9.6) cares about the call having
    happened, not just successful drafts.
    """
    if not isinstance(code, Code):
        raise TypeError("draft_memo_for_code expects a Code")
    if memo_type not in MEMO_TYPES:
        raise ProjectValidationError(
            f"memo_type must be one of {MEMO_TYPES}; got {memo_type!r}"
        )
    if code.project_id != project_id:
        raise ProjectValidationError(
            f"code.project_id ({code.project_id!r}) != "
            f"project_id ({project_id!r})"
        )

    snippets = collect_seed_snippets(
        code=code,
        applications=applications,
        segments_by_source=segments_by_source,
        include_applications=include_applications,
        max_snippets=max_seed_snippets,
    )
    prompt = make_memo_draft_prompt(
        code=code,
        seed_snippets=snippets,
        include_existing_memo=include_existing_memo,
    )
    if len(prompt) > MAX_PROMPT_LEN:
        # Defensive: a code with an enormous theoretical_memo could push
        # past the cap. Truncate end-first; the seed snippets at the top
        # are the most important context.
        prompt = prompt[:MAX_PROMPT_LEN]

    raw_response = str(generate_fn(prompt) or "")
    if len(raw_response) > MAX_RAW_LLM_RESPONSE_LEN:
        raw_response = raw_response[:MAX_RAW_LLM_RESPONSE_LEN]
    parsed = parse_memo_draft_response(raw_response)

    return MemoDraft.new(
        project_id=project_id,
        code_id=code.id,
        memo_type=memo_type,
        title=parsed.title,
        body=parsed.body,
        rationale=parsed.rationale,
        seed_snippets=snippets,
        generation_model=generation_model,
        prompt=prompt,
        raw_llm_response=raw_response,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Decision lifecycle
# --------------------------------------------------------------------------- #


def record_memo_draft_decision(
    draft: MemoDraft,
    *,
    decision: str,
    coder_id: str,
    accepted_memo_id: str | None = None,
    rejection_reason: str = "",
    notes: str | None = None,
    now: str | None = None,
) -> None:
    """Move a draft from ``pending`` into a terminal state.

    Mutates ``draft`` in place and re-runs ``validate``. Decision
    transitions out of a terminal state are not allowed — that would
    rewrite the audit trail; create a new draft instead.

    * ``accepted`` — researcher accepted the draft as-is.
      ``accepted_memo_id`` is the resulting :class:`Memo` id; it can be
      passed here directly, or set later via
      :func:`MemoDraft.apply_update`. Most callers will create the
      Memo first via :func:`promote_memo_draft_to_memo`, which records
      the id automatically.
    * ``modified`` — researcher edited the draft before saving.
      Same fields as ``accepted``.
    * ``rejected`` — none of the draft was useful. Forbids
      ``accepted_memo_id`` and accepts a free-text reason.
    """
    if decision not in TERMINAL_MEMO_DRAFT_DECISIONS:
        raise ProjectValidationError(
            f"decision must be one of {sorted(TERMINAL_MEMO_DRAFT_DECISIONS)}; "
            f"got {decision!r}"
        )
    if draft.decision in TERMINAL_MEMO_DRAFT_DECISIONS:
        raise ProjectValidationError(
            f"Draft {draft.id} already has decision {draft.decision!r}; "
            "create a new draft instead of overwriting the audit trail."
        )
    if not isinstance(coder_id, str) or not CODER_ID_RE.match(coder_id):
        raise ProjectValidationError(
            f"coder_id must be a 12-char hex coder id; got {coder_id!r}"
        )

    if decision in (MEMO_DRAFT_DECISION_ACCEPTED, MEMO_DRAFT_DECISION_MODIFIED):
        if accepted_memo_id is not None:
            if not MEMO_ID_RE.match(str(accepted_memo_id)):
                raise ProjectValidationError(
                    "accepted_memo_id must be 12-char hex if set"
                )
            draft.accepted_memo_id = str(accepted_memo_id)
        # accepted_memo_id is *optional* on the decision call — the
        # caller can attach it later via apply_update once the Memo
        # has been minted.
    else:  # rejected
        if accepted_memo_id is not None:
            raise ProjectValidationError(
                "rejected drafts must not record an accepted memo id"
            )
        draft.accepted_memo_id = None

    draft.decision = decision
    draft.decided_at = now or utcnow_iso()
    draft.decided_by_coder_id = coder_id
    if rejection_reason:
        if len(rejection_reason) > MAX_REJECTION_REASON_LEN:
            raise ProjectValidationError(
                f"rejection_reason exceeds {MAX_REJECTION_REASON_LEN} chars"
            )
        draft.rejection_reason = rejection_reason
    elif decision != MEMO_DRAFT_DECISION_REJECTED:
        draft.rejection_reason = ""
    if notes is not None:
        if len(notes) > MAX_NOTES_LEN:
            raise ProjectValidationError(
                f"notes exceeds {MAX_NOTES_LEN} chars"
            )
        draft.notes = notes
    draft.modified_at = now or utcnow_iso()
    draft.validate()


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #


# Default link role on the back-link from the new memo to its source code.
DEFAULT_BACK_LINK_ROLE = "drafted_for"


def promote_memo_draft_to_memo(
    projects_root: Path,
    draft: MemoDraft,
    *,
    coder_id: str,
    decision: str = MEMO_DRAFT_DECISION_ACCEPTED,
    title: str | None = None,
    body: str | None = None,
    author_coder_id: str | None = None,
    extra_links: Iterable[MemoLink | Mapping[str, Any]] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> Memo:
    """Promote a draft into a saved :class:`Memo`.

    * Creates a :class:`Memo` of type ``draft.memo_type`` whose body
      defaults to ``draft.body`` (override via ``body=``) and title
      defaults to ``draft.title`` (override via ``title=``).
    * Adds a :class:`MemoLink` to the source code with role
      ``"drafted_for"``. Caller can append more links via
      ``extra_links``.
    * Sets ``provenance['source'] = 'ai_drafted'`` and
      ``provenance['draft_id'] = draft.id``. Reserved keys cannot be
      overridden via ``extra_provenance`` (avoids audit-trail
      forgery — same pattern as F5.5).
    * Calls :func:`record_memo_draft_decision` with ``decision``
      (default ``"accepted"``) and stamps the new memo's id on the
      draft.
    * Persists the memo and the updated draft.

    Returns the saved :class:`Memo`. Raises
    :class:`ProjectValidationError` on any invariant violation; the
    function is best-effort atomic at the per-file level (each save
    uses the existing ``.json.tmp`` rename pattern).
    """
    if decision not in (MEMO_DRAFT_DECISION_ACCEPTED,
                        MEMO_DRAFT_DECISION_MODIFIED):
        raise ProjectValidationError(
            "promote_memo_draft_to_memo only supports decision "
            "'accepted' or 'modified'; for 'rejected' use "
            "record_memo_draft_decision directly"
        )
    if draft.decision in TERMINAL_MEMO_DRAFT_DECISIONS:
        raise ProjectValidationError(
            f"Draft {draft.id} already has decision {draft.decision!r}; "
            "create a new draft instead of promoting twice."
        )

    chosen_title = (
        str(title).strip()
        if title is not None
        else (draft.title or "").strip()
    )
    chosen_body = (
        str(body) if body is not None else draft.body
    )

    # Build provenance with reserved keys non-overridable (matches
    # F5.5's _normalise_extra_provenance pattern).
    provenance: dict[str, str] = {}
    reserved = {"source", "draft_id"}
    if extra_provenance is not None:
        if not isinstance(extra_provenance, Mapping):
            raise ProjectValidationError(
                "extra_provenance must be a mapping of string→string"
            )
        for raw_k, raw_v in extra_provenance.items():
            k = str(raw_k).strip()
            if not k:
                continue
            if k in reserved:
                raise ProjectValidationError(
                    f"extra_provenance cannot override reserved key {k!r}"
                )
            provenance[k] = str(raw_v)
    provenance["source"] = MEMO_DRAFT_PROVENANCE_SOURCE
    provenance["draft_id"] = draft.id

    # Build the back-link to the source code, plus any extras.
    links: list[MemoLink] = [
        MemoLink(
            target_type="code",
            target_id=draft.code_id,
            role=DEFAULT_BACK_LINK_ROLE,
        )
    ]
    if extra_links is not None:
        for raw in extra_links:
            if isinstance(raw, MemoLink):
                links.append(raw)
            elif isinstance(raw, Mapping):
                links.append(MemoLink.from_dict(raw))
            else:
                raise ProjectValidationError(
                    "extra_links entries must be MemoLink or dict"
                )

    memo = Memo.new(
        project_id=draft.project_id,
        type=draft.memo_type,
        title=chosen_title,
        body=chosen_body,
        author_coder_id=author_coder_id if author_coder_id else None,
        links=links,
        provenance=provenance,
        now=now,
    )
    save_memo(projects_root, memo)

    record_memo_draft_decision(
        draft,
        decision=decision,
        coder_id=coder_id,
        accepted_memo_id=memo.id,
        now=now,
    )
    save_memo_draft(projects_root, draft)
    return memo


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def memo_drafts_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's memo drafts.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / MEMO_DRAFTS_DIRNAME


def memo_draft_state_path(
    projects_root: Path, project_id: str, draft_id: str
) -> Path:
    if not MEMO_DRAFT_ID_RE.match(draft_id):
        raise ProjectValidationError(
            f"Invalid memo-draft id: {draft_id!r}"
        )
    return memo_drafts_dir(projects_root, project_id) / f"{draft_id}.json"


def save_memo_draft(projects_root: Path, draft: MemoDraft) -> Path:
    """Persist a memo-draft atomically.

    Writes to a ``.json.tmp`` sibling and renames into place — same
    convention as the rest of the F-feature stack.
    """
    draft.validate()
    parent = project_dir(projects_root, draft.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving memo drafts."
        )
    md = memo_drafts_dir(projects_root, draft.project_id)
    md.mkdir(parents=True, exist_ok=True)
    target = memo_draft_state_path(projects_root, draft.project_id, draft.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(draft.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_memo_draft(
    projects_root: Path, project_id: str, draft_id: str
) -> MemoDraft:
    """Load a memo-draft by id. Raises ``FileNotFoundError`` if missing."""
    p = memo_draft_state_path(projects_root, project_id, draft_id)
    if not p.exists():
        raise FileNotFoundError(f"No memo draft at {p}")
    return MemoDraft.from_dict(json.loads(p.read_text()))


def list_memo_drafts(
    projects_root: Path,
    project_id: str,
    *,
    code_id: str | None = None,
    decision: str | None = None,
) -> list[MemoDraft]:
    """List all memo-drafts in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse. Sorted by
    ``created_at`` ascending (the audit-trail ordering).
    """
    if code_id is not None and not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(
            f"Invalid code id filter: {code_id!r}"
        )
    if decision is not None and decision not in MEMO_DRAFT_DECISIONS:
        raise ProjectValidationError(
            f"Invalid decision filter: {decision!r}"
        )
    md = memo_drafts_dir(projects_root, project_id)
    if not md.exists():
        return []
    out: list[MemoDraft] = []
    for f in sorted(md.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        did = f.stem
        if not MEMO_DRAFT_ID_RE.match(did):
            continue
        try:
            d = MemoDraft.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if code_id is not None and d.code_id != code_id:
            continue
        if decision is not None and d.decision != decision:
            continue
        out.append(d)
    out.sort(key=lambda d: (d.created_at, d.id))
    return out


def delete_memo_draft(
    projects_root: Path, project_id: str, draft_id: str
) -> bool:
    """Remove a memo-draft file. Returns False if it didn't exist.

    Production code should prefer keeping drafts for the audit trail;
    deletion is exposed for tests and the REFI-QDA import path.
    """
    p = memo_draft_state_path(projects_root, project_id, draft_id)
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
