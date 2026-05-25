"""Memo entity for the academic-coding workflow (F5.1).

Per PLANNING.md F5.1:

  > Memo entity with type (code / theoretical / methodological /
  > reflexive / quote / source / project), rich text body, multi-
  > target links.

A *memo* is a free-form analytic note. Memos are the connective tissue
of grounded-theory analysis: they record category insights, theoretical
moves, methodological reasoning, reflexive positionality, and quote-
specific reactions. Charmaz, Strauss/Corbin, and Glaser all share the
same root — write memos early, write them often, refactor them upward
into theory.

This module provides the data model + persistence; later features
extend it:

  * F5.2 — right-click memo creation from any context (UI; this module
    already supports the multi-target ``links`` shape it needs)
  * F5.3 — memo-sorting canvas (drag/group cards, link memo↔memo)
  * F5.4 — export-all-memos filtered by type / linked-to (uses
    :func:`list_memos`)
  * F5.5 — promote-a-memo-into-a-code-definition (one-click)
  * F8.8 — AI memo-draft action (will populate ``provenance``)

Memos live alongside their project on disk::

    projects/<project_id>/memos/<memo_id>.json

so ``delete_project`` cleans them up for free, mirroring how Sources
(F1.2), Participants (F1.3), Codes (F2.1), Coders (F2.5), and
Applications (F4.1) are stored.

Why a single ``Memo`` table with a ``type`` field instead of seven
separate entities?

  * Researchers reclassify memos all the time — a "reflexive" memo
    becomes a "methodological" one; an early "quote" memo gets
    promoted to "theoretical". A type field is a one-line edit; a
    table change is a refactor.
  * The on-disk format and validation rules are identical across
    types. Splitting would duplicate code without adding any
    type-specific behaviour at this layer.
  * The export feature (F5.4) wants a uniform iterator. Same for the
    sorting canvas (F5.3): all memos, all on one surface.

Multi-target links
------------------

PLANNING calls out memos that are "code-attached, source-attached,
project-attached, or free-floating". F5.1 generalises to **multi-
target** because real memos cross-cut: a methodological memo can be
linked to two sources *and* the project; a theoretical memo can link
to a code *and* a quote (application) that exemplifies it. Each
``MemoLink`` carries:

  * ``target_type`` — closed vocabulary: ``code`` / ``source`` /
    ``application`` / ``participant`` / ``coder`` / ``project`` /
    ``memo``. ``memo`` is included so F5.3 can wire memo→memo edges
    without a schema change.
  * ``target_id`` — the 12-char hex id of the target. We validate the
    shape but not the existence of the file: the audit trail tolerates
    a target that's been deleted (you can still see "this memo was
    written about code X" even after X is hard-deleted).
  * ``role`` — optional free-form short label for the link's role
    ("exemplifies", "contradicts", "elaborates"). Ungoverned
    vocabulary because researchers will want their own terms; the
    sorting canvas will display whatever's there.

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused by
the CLI later. Conventions match ``scribe.projects`` (F1.1),
``scribe.codes`` (F2.1), ``scribe.applications`` (F4.1).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Memo IDs follow the same 12-char hex shape as project / source /
# code / coder / application / job IDs. Keeps URL routing rules and
# path-traversal guards uniform across the app.
MEMO_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# The seven memo types from PLANNING.md F5.1, plus ``free`` as the
# unclassified default. Free memos exist before the researcher decides
# what kind of memo they're writing — and that "I'll classify it
# later" affordance matters in early-phase coding when ideas come
# faster than categories.
#
#   * ``code``           — a memo about a particular code's meaning,
#                          edges, exemplars (separable from the code's
#                          own ``theoretical_memo`` field, F2.1).
#   * ``theoretical``    — analytic memo about a category or concept;
#                          the workhorse of grounded theory.
#   * ``methodological`` — decisions about *how* the analysis is being
#                          done ("why I split code X into X-a/X-b on
#                          2026-04-12").
#   * ``reflexive``      — researcher's positionality, emotional
#                          reactions, biases noticed in the moment.
#   * ``quote``          — memo attached to a specific quote / coded
#                          segment; the "marginal note" of QDA.
#   * ``source``         — interview-level reflection ("P3 was
#                          guarded; revisit consent question").
#   * ``project``        — corpus-wide observation that doesn't fit a
#                          single code or source.
#   * ``free``           — uncategorised; classify later.
MEMO_TYPES: tuple[str, ...] = (
    "code",
    "theoretical",
    "methodological",
    "reflexive",
    "quote",
    "source",
    "project",
    "free",
)

# Body format. Markdown is the editor-friendly default — round-trips
# cleanly through CSV / Word export (F5.4, F6.1). ``plain`` is for
# pasted-in text that the researcher hasn't formatted; ``html`` is
# reserved for future rich-text editor backing (NVivo-style RTF
# export, when we get there).
MEMO_BODY_FORMATS: tuple[str, ...] = (
    "markdown",
    "plain",
    "html",
)

# Closed vocabulary for ``MemoLink.target_type``. We include every
# entity that exists today and ``memo`` (for F5.3 memo↔memo edges).
# Adding a new type later is a one-line change here + matching id-
# regex import; no on-disk migration needed.
MEMO_LINK_TARGET_TYPES: tuple[str, ...] = (
    "code",
    "source",
    "application",
    "participant",
    "coder",
    "project",
    "memo",
)

# All these entities share the same 12-char hex id shape, so a single
# regex covers any target_id. We keep this as a separate constant so
# if a future entity changes shape the validation error message can
# point at the right rule.
TARGET_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Provenance keys (origin metadata). Same shape as the other entities'
# ``provenance`` field — small free-form string→string with a closed
# ``source`` vocabulary.
MEMO_PROVENANCE_SOURCES: tuple[str, ...] = (
    "human",
    "ai_drafted",
    "ai_modified",
    "imported",
    "promoted_from_quote",
    "other",
)

# Field length / cardinality limits. Generous, but bounded so a typo
# in the UI can't write a 50 MB memo.json. Memos are first-class
# analytic artefacts; ``body`` is more generous than other entities'
# notes fields because a single memo can be a multi-page essay.
MAX_TITLE_LEN = 200
MAX_BODY_LEN = 64 * 1024  # 64 KiB — long memo, well bounded
MAX_TAG_LEN = 64
MAX_TAGS = 32
MAX_LINK_ROLE_LEN = 64
MAX_LINKS = 64
MAX_PROVENANCE_KEYS = 16
MAX_PROVENANCE_VALUE_LEN = 1000

# Tag shape: a short slug-ish token. Allow letters, digits, hyphen,
# underscore, space; must start with a letter. Same flavour as
# ``Code`` provenance keys but applied to the user-facing tag itself.
TAG_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")

# Role on a MemoLink. Same regex as Tag — short, slug-ish, no
# punctuation that would break the export formats.
LINK_ROLE_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")

# Provenance key shape (matches Code / Application).
PROVENANCE_KEY_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")


# --------------------------------------------------------------------------- #
# Helper data classes
# --------------------------------------------------------------------------- #


@dataclass
class MemoLink:
    """One typed link from a Memo to a target entity.

    ``target_type`` ∈ :data:`MEMO_LINK_TARGET_TYPES`; ``target_id`` is
    a 12-char hex id whose meaning is fixed by ``target_type``. The
    optional ``role`` is a free-form short label ("exemplifies",
    "contradicts", "elaborates") — ungoverned vocabulary because each
    researcher prefers their own.

    Validation enforces shape only. Existence of the target file is
    *not* checked: deleting an entity does not retroactively erase the
    memos that referenced it. The audit trail wins over referential
    integrity here, because methodological transparency is the whole
    point of memos.
    """

    target_type: str
    target_id: str
    role: str = ""

    def to_dict(self) -> dict[str, str]:
        out = {"target_type": self.target_type, "target_id": self.target_id}
        if self.role:
            out["role"] = self.role
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoLink":
        if not isinstance(d, dict):
            raise ProjectValidationError("MemoLink payload must be an object")
        for required in ("target_type", "target_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"MemoLink payload missing required key: {required}"
                )
        return cls(
            target_type=str(d["target_type"]),
            target_id=str(d["target_id"]),
            role=str(d.get("role", "") or ""),
        )

    def validate(self) -> None:
        if self.target_type not in MEMO_LINK_TARGET_TYPES:
            raise ProjectValidationError(
                f"target_type must be one of {MEMO_LINK_TARGET_TYPES}; "
                f"got {self.target_type!r}"
            )
        if not TARGET_ID_RE.match(self.target_id):
            raise ProjectValidationError(
                f"target_id must be 12-char hex; got {self.target_id!r}"
            )
        role = self.role.strip()
        if role:
            if len(role) > MAX_LINK_ROLE_LEN:
                raise ProjectValidationError(
                    f"link role must be ≤ {MAX_LINK_ROLE_LEN} chars"
                )
            if not LINK_ROLE_RE.match(role):
                raise ProjectValidationError(
                    f"link role {role!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
        self.role = role


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Memo:
    """One analytic memo within a project.

    Memos are mutable: titles change, bodies grow, links are added and
    removed as analysis progresses. ``modified_at`` tracks the most
    recent edit. F9.1's append-only event log will record each edit
    with the diff, but the memo file itself holds the latest state.

    Required fields:
      * ``id``, ``project_id`` — the usual 12-char hex pair.
      * ``type`` — one of :data:`MEMO_TYPES`. Defaults to ``"free"``
        so a "new memo" button can hand back a valid Memo before the
        researcher has chosen.
      * ``body`` — the analytic content. Empty body is allowed (a
        title-only memo is a placeholder for "I want to write this
        later").

    Optional fields:
      * ``title`` — short headline. Memos without titles still work;
        the export uses the first line of the body as the heading.
      * ``body_format`` — ``markdown`` (default) / ``plain`` / ``html``.
      * ``author_coder_id`` — who wrote it. Mirrors how Applications
        carry ``coder_id`` for human authorship; reflexive memos
        especially want this so the audit trail records *whose*
        positionality is on file.
      * ``links`` — a list of :class:`MemoLink`. Multi-target.
      * ``tags`` — short string labels for free-form clustering.
      * ``provenance`` — small string→string dict for AI / import
        origin; F8.8 / F8.9 will populate this.
    """

    id: str
    project_id: str
    type: str = "free"
    title: str = ""
    body: str = ""
    body_format: str = "markdown"
    author_coder_id: str | None = None
    links: list[MemoLink] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
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
        type: str = "free",
        title: str = "",
        body: str = "",
        body_format: str = "markdown",
        author_coder_id: str | None = None,
        links: Iterable[MemoLink | dict[str, Any]] | None = None,
        tags: Iterable[str] | None = None,
        provenance: dict[str, str] | None = None,
        memo_id: str | None = None,
        now: str | None = None,
    ) -> "Memo":
        """Build a fresh Memo, validate, and stamp ``created_at`` /
        ``modified_at``.
        """
        coerced_links: list[MemoLink] = []
        for link in links or ():
            if isinstance(link, MemoLink):
                coerced_links.append(link)
            elif isinstance(link, dict):
                coerced_links.append(MemoLink.from_dict(link))
            else:
                raise ProjectValidationError(
                    "links entries must be MemoLink or dict"
                )
        ts = now or utcnow_iso()
        m = cls(
            id=memo_id or new_memo_id(),
            project_id=project_id,
            type=type,
            title=title,
            body=body,
            body_format=body_format,
            author_coder_id=author_coder_id if author_coder_id else None,
            links=coerced_links,
            tags=list(tags or ()),
            provenance=dict(provenance or {}),
            created_at=ts,
            modified_at=ts,
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop ``role`` from each link if empty so the on-disk shape
        # matches MemoLink.to_dict (which omits empty role for compactness).
        d["links"] = [link.to_dict() for link in self.links]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Memo":
        if not isinstance(d, dict):
            raise ProjectValidationError("Memo payload must be an object")
        for required in ("id", "project_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"Memo payload missing required key: {required}"
                )
        raw_links = d.get("links")
        if raw_links is None:
            raw_links = []
        if not isinstance(raw_links, list):
            raise ProjectValidationError("links must be a list")
        links = [MemoLink.from_dict(link) for link in raw_links]

        raw_tags = d.get("tags")
        if raw_tags is None:
            raw_tags = []
        if not isinstance(raw_tags, list):
            raise ProjectValidationError("tags must be a list")

        m = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            type=str(d.get("type", "free") or "free"),
            title=str(d.get("title", "") or ""),
            body=str(d.get("body", "") or ""),
            body_format=str(d.get("body_format", "markdown") or "markdown"),
            author_coder_id=(
                str(d["author_coder_id"]) if d.get("author_coder_id") else None
            ),
            links=links,
            tags=[str(t) for t in raw_tags],
            provenance={
                str(k): str(v)
                for k, v in (d.get("provenance") or {}).items()
            },
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: dict[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update in place.

        Mirrors ``Application.apply_update`` / ``Code.apply_update``.
        Cross-entity ids and timestamps are entity-managed; passing
        ``id`` / ``project_id`` / ``created_at`` / ``modified_at`` is
        allowed (and ignored) so a client can round-trip a fetched
        object.

        Updating ``links`` / ``tags`` replaces the whole list; the UI
        layer is expected to read-modify-write. We don't expose
        per-link mutation here because every memo edit is meant to
        produce one event in the audit log (F9.1) and a "set the link
        list to X" event is more legible than "remove link 3, add link
        with role Y at index 2".
        """
        if not isinstance(patch, dict):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS - _IGNORED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "type" in patch:
            self.type = str(patch["type"] or "free")
        if "title" in patch:
            self.title = str(patch["title"] or "")
        if "body" in patch:
            self.body = str(patch["body"] or "")
        if "body_format" in patch:
            self.body_format = str(patch["body_format"] or "markdown")
        if "author_coder_id" in patch:
            v = patch["author_coder_id"]
            self.author_coder_id = str(v) if v else None
        if "links" in patch:
            raw_links = patch["links"] or []
            if not isinstance(raw_links, list):
                raise ProjectValidationError("links must be a list")
            new_links: list[MemoLink] = []
            for link in raw_links:
                if isinstance(link, MemoLink):
                    new_links.append(link)
                elif isinstance(link, dict):
                    new_links.append(MemoLink.from_dict(link))
                else:
                    raise ProjectValidationError(
                        "links entries must be MemoLink or dict"
                    )
            self.links = new_links
        if "tags" in patch:
            raw_tags = patch["tags"] or []
            if not isinstance(raw_tags, list):
                raise ProjectValidationError("tags must be a list")
            self.tags = [str(t) for t in raw_tags]
        if "provenance" in patch:
            prov = patch["provenance"] or {}
            if not isinstance(prov, dict):
                raise ProjectValidationError(
                    "provenance must be an object of string→string"
                )
            self.provenance = {str(k): str(v) for k, v in prov.items()}

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Multi-target convenience
    # ------------------------------------------------------------------ #

    def has_link_to(
        self, target_type: str, target_id: str
    ) -> bool:
        """Return True iff this memo has a link with the given (type, id)."""
        return any(
            link.target_type == target_type and link.target_id == target_id
            for link in self.links
        )

    def link_target_ids(self, target_type: str) -> list[str]:
        """Return all ``target_id``s linked from this memo for a given type.

        Order preserved (= insertion order). Useful for UI ("which
        codes does this memo touch?") and for the F5.4 export filters.
        """
        return [
            link.target_id
            for link in self.links
            if link.target_type == target_type
        ]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not MEMO_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid memo id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        if self.type not in MEMO_TYPES:
            raise ProjectValidationError(
                f"type must be one of {MEMO_TYPES}; got {self.type!r}"
            )
        if self.body_format not in MEMO_BODY_FORMATS:
            raise ProjectValidationError(
                f"body_format must be one of {MEMO_BODY_FORMATS}; "
                f"got {self.body_format!r}"
            )

        title = self.title.strip()
        if len(title) > MAX_TITLE_LEN:
            raise ProjectValidationError(
                f"title must be ≤ {MAX_TITLE_LEN} chars"
            )
        # Persist trimmed so on-disk state is canonical (matches the
        # ``Project.name`` / ``Source.name`` pattern).
        self.title = title

        if len(self.body) > MAX_BODY_LEN:
            raise ProjectValidationError(
                f"body must be ≤ {MAX_BODY_LEN} chars"
            )

        if self.author_coder_id is not None:
            if not TARGET_ID_RE.match(self.author_coder_id):
                raise ProjectValidationError(
                    f"author_coder_id must be 12-char hex; "
                    f"got {self.author_coder_id!r}"
                )

        # Tags: dedupe (preserving order), validate each.
        if not isinstance(self.tags, list):
            raise ProjectValidationError("tags must be a list")
        if len(self.tags) > MAX_TAGS:
            raise ProjectValidationError(
                f"At most {MAX_TAGS} tags allowed"
            )
        seen: set[str] = set()
        cleaned_tags: list[str] = []
        for raw_t in self.tags:
            t = str(raw_t).strip()
            if not t:
                continue
            if len(t) > MAX_TAG_LEN:
                raise ProjectValidationError(
                    f"tag must be ≤ {MAX_TAG_LEN} chars"
                )
            if not TAG_RE.match(t):
                raise ProjectValidationError(
                    f"tag {t!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            if t in seen:
                continue
            seen.add(t)
            cleaned_tags.append(t)
        self.tags = cleaned_tags

        # Links: validate each, dedupe by (target_type, target_id, role).
        if not isinstance(self.links, list):
            raise ProjectValidationError("links must be a list")
        if len(self.links) > MAX_LINKS:
            raise ProjectValidationError(
                f"At most {MAX_LINKS} links allowed"
            )
        seen_links: set[tuple[str, str, str]] = set()
        cleaned_links: list[MemoLink] = []
        for link in self.links:
            if not isinstance(link, MemoLink):
                raise ProjectValidationError(
                    "links entries must be MemoLink"
                )
            link.validate()
            key = (link.target_type, link.target_id, link.role)
            if key in seen_links:
                continue
            seen_links.add(key)
            cleaned_links.append(link)
        self.links = cleaned_links

        # Provenance: free-form dict bounded by shape limits.
        if not isinstance(self.provenance, dict):
            raise ProjectValidationError(
                "provenance must be an object of string→string"
            )
        if len(self.provenance) > MAX_PROVENANCE_KEYS:
            raise ProjectValidationError(
                f"At most {MAX_PROVENANCE_KEYS} provenance keys allowed"
            )
        cleaned_prov: dict[str, str] = {}
        for raw_k, raw_v in self.provenance.items():
            k = str(raw_k).strip()
            if not k:
                continue
            if not PROVENANCE_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"provenance key {k!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            v = str(raw_v)
            if len(v) > MAX_PROVENANCE_VALUE_LEN:
                raise ProjectValidationError(
                    f"provenance[{k!r}] value too long "
                    f"(>{MAX_PROVENANCE_VALUE_LEN})"
                )
            cleaned_prov[k] = v
        if "source" in cleaned_prov:
            if cleaned_prov["source"] not in MEMO_PROVENANCE_SOURCES:
                raise ProjectValidationError(
                    f"provenance.source must be one of "
                    f"{MEMO_PROVENANCE_SOURCES}; "
                    f"got {cleaned_prov['source']!r}"
                )
        self.provenance = cleaned_prov


# Fields a PATCH may set. ``project_id`` and the timestamps are
# entity-managed; passing them is allowed (and ignored) so a client
# can round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "type",
    "title",
    "body",
    "body_format",
    "author_coder_id",
    "links",
    "tags",
    "provenance",
}
_IGNORED_PATCH_KEYS = {"id", "project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_memo_id() -> str:
    """Mint a new 12-char hex memo id (matches every other entity id shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def memos_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's memos.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "memos"


def memo_state_path(
    projects_root: Path, project_id: str, memo_id: str
) -> Path:
    if not MEMO_ID_RE.match(memo_id):
        raise ProjectValidationError(f"Invalid memo id: {memo_id!r}")
    return memos_dir(projects_root, project_id) / f"{memo_id}.json"


def save_memo(projects_root: Path, memo: Memo) -> Path:
    """Persist a memo to ``<projects_root>/<pid>/memos/<mid>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). Mirrors the convention
    of ``save_source`` / ``save_code`` / ``save_application`` — a memo
    without a project is meaningless and we surface that early.
    """
    memo.validate()
    parent = project_dir(projects_root, memo.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its memos."
        )
    md = memos_dir(projects_root, memo.project_id)
    md.mkdir(parents=True, exist_ok=True)
    target = memo_state_path(projects_root, memo.project_id, memo.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memo.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_memo(
    projects_root: Path, project_id: str, memo_id: str
) -> Memo:
    """Load a memo by id. Raises ``FileNotFoundError`` if missing."""
    p = memo_state_path(projects_root, project_id, memo_id)
    if not p.exists():
        raise FileNotFoundError(f"No memo at {p}")
    return Memo.from_dict(json.loads(p.read_text()))


def list_memos(
    projects_root: Path,
    project_id: str,
    *,
    type: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    author_coder_id: str | None = None,
    tag: str | None = None,
) -> list[Memo]:
    """List all memos in a project, optionally filtered.

    Filters are AND-combined when multiple are supplied.

      * ``type`` — restrict to a single :data:`MEMO_TYPES` value.
      * ``target_type`` / ``target_id`` — restrict to memos that link
        to a specific entity. Pass both together to filter by an exact
        target; passing ``target_type`` alone returns memos that link
        to *any* entity of that type.
      * ``author_coder_id`` — restrict to a single author.
      * ``tag`` — restrict to memos carrying the given tag (exact
        match, case-sensitive — tag vocabulary is researcher-driven).

    Skips files that don't parse as a valid Memo so a single corrupt
    file doesn't break the view (audit log will eventually surface
    this — F9.7). Sorted by ``created_at`` ascending, then by ``id``
    as a stable tiebreaker — the timeline matters for grounded-theory
    memo development (early memos *should* read first).
    """
    if type is not None and type not in MEMO_TYPES:
        raise ProjectValidationError(
            f"Invalid type filter: {type!r}; "
            f"must be one of {MEMO_TYPES}"
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
            raise ProjectValidationError("tag filter must be a non-empty string")

    md = memos_dir(projects_root, project_id)
    if not md.exists():
        return []
    out: list[Memo] = []
    for f in sorted(md.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        mid = f.stem
        if not MEMO_ID_RE.match(mid):
            continue
        try:
            m = Memo.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
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


def delete_memo(
    projects_root: Path, project_id: str, memo_id: str
) -> bool:
    """Remove a memo file. Returns False if it didn't exist.

    Hard delete is exposed because memos, unlike Codes (F2.3), don't
    have a separate retire-vs-delete distinction at the F5.1 layer.
    F9.1's event log will record the deletion for audit. Future
    features may add a soft-delete; F5.1 keeps it simple.
    """
    p = memo_state_path(projects_root, project_id, memo_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True


def count_memos(
    projects_root: Path,
    project_id: str,
    *,
    type: str | None = None,
) -> int:
    """Convenience for UI badges — number of memos in a project (or by type).

    Reads the directory once. F5.4 export will reuse the same filter
    semantics.
    """
    return len(list_memos(projects_root, project_id, type=type))
