"""Memo-sorting canvas (F5.3).

Per PLANNING.md F5.3:

  > Memo-sorting canvas (drag/group cards, link memo→memo, link
  > memo→category).

Grounded theory's "memo-sorting" move comes from Glaser via Charmaz:
once the researcher has a stack of memos, lay them out, move them
around, and let the categories *emerge* from spatial clustering.
NVivo and ATLAS.ti both ship a "concept map" or "network view" for
exactly this; F5.3 is Scribe's pure-data answer.

What's here
-----------

* :class:`MemoCanvasCard` — one memo's position on the canvas
  (``memo_id`` + ``x`` + ``y``). Not every memo has to be on the
  canvas; the canvas tracks only memos the researcher chose to lay
  out.
* :class:`MemoCanvasCategory` — a named cluster ("emerging concept",
  "structural conditions", "in-vivo themes"). Categories carry their
  own anchor coordinates so the UI can render them as labelled
  rectangles or pills, plus an optional ``color`` for visual
  grouping.
* :class:`MemoCanvas` — the project-level singleton: list of cards,
  list of categories, mapping from category → ordered list of
  memo-ids in that category.
* On-disk persistence: one JSON file per project at
  ``projects/<pid>/memo_canvas.json``. Lazy: a project that has never
  used the canvas returns an empty :class:`MemoCanvas` from
  :func:`load_canvas`.
* Mutation helpers: ``move_card``, ``remove_card``, ``add_category``,
  ``update_category``, ``remove_category``, ``assign_card_to_category``,
  ``unassign_card_from_category``. Each mutates in place and bumps
  ``modified_at``.

Memo→memo links are *already* supported via :class:`scribe.memos.MemoLink`
with ``target_type = "memo"``. F5.3 deliberately does **not** duplicate
that here — :func:`link_memos_on_canvas` is a tiny wrapper that
records the link on the source memo via the existing memos API, so
the audit trail and persistence path stay unified.

Memo→category links are canvas-only: a category is a canvas concept,
not a top-level Scribe entity. Storing the membership inside the
canvas keeps :class:`scribe.memos.MemoLink` focused on inter-entity
links and avoids inventing a "category" target_type that only the
canvas would understand.

Why a singleton (one canvas per project) and not many?

  * The PLANNING entry says "the canvas", singular. NVivo's "concept
    map" surface is similarly singular per project; analysts who want
    multiple boards split by stage typically use the codebook stage
    field plus filtering, not multiple maps.
  * One file per project keeps F1.5's bundle round-trip simple —
    every project has exactly one ``memo_canvas.json``, present or
    absent.
  * If the multi-canvas use case lands later, the on-disk format can
    grow a ``canvas_id`` field without disturbing existing files.

Conventions match :mod:`scribe.projects` (F1.1), :mod:`scribe.memos`
(F5.1), :mod:`scribe.memo_context` (F5.2). Stand-alone — no FastAPI,
no engine imports.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .memos import (
    MEMO_ID_RE,
    Memo,
    MemoLink,
    load_memo,
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

# Category ids share the 12-char hex shape used by every other Scribe
# entity. Keeps URL routing and traversal guards uniform if/when a
# future feature exposes per-category endpoints.
CATEGORY_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Hex colour shape for category swatches: ``#rrggbb``. We accept only
# the long form so every consumer (CSS, Word export, REFI-QDA) gets
# the same 7-char string. Empty string means "no colour assigned".
CATEGORY_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Filename for the project-level singleton canvas. One canvas per
# project; absent file = empty canvas.
CANVAS_FILENAME = "memo_canvas.json"

# Bounded coordinate range. Real UIs render in pixel space; we cap at
# a generous ±1e6 so an off-by-NaN doesn't write a garbage file but
# don't constrain the researcher's logical layout space. Also
# rejects ``inf`` and ``NaN``.
MAX_COORD = 1_000_000.0

# Field length / cardinality limits. Generous, but bounded so a typo
# in the UI can't write a 50 MB ``memo_canvas.json``.
MAX_LABEL_LEN = 120
MAX_CARDS = 5000
MAX_CATEGORIES = 500
MAX_MEMBERS_PER_CATEGORY = 5000

# Category labels are user-facing strings — researchers will use
# spaces and apostrophes (e.g. "P3's coping strategies"). We keep the
# rule loose: non-empty after strip, no control characters, ≤ 120
# chars. A separate regex would over-constrain.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_category_id() -> str:
    """Mint a new 12-char hex category id."""
    return uuid.uuid4().hex[:12]


def _coerce_coord(value: Any, *, name: str) -> float:
    """Validate and coerce a coordinate.

    Accepts int / float; rejects NaN, ±inf, and values outside
    ``±MAX_COORD``. Returns a Python float.
    """
    if isinstance(value, bool):
        # bool is a subclass of int — researchers should never write
        # True/False as a coordinate, and silently coercing hides bugs.
        raise ProjectValidationError(f"{name} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise ProjectValidationError(f"{name} must be a number")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ProjectValidationError(f"{name} must be finite (got {value!r})")
    if abs(f) > MAX_COORD:
        raise ProjectValidationError(
            f"{name} out of range (|{name}| ≤ {MAX_COORD:g})"
        )
    return f


def _validate_label(label: str) -> str:
    """Validate a category label; return the trimmed canonical form."""
    if not isinstance(label, str):
        raise ProjectValidationError("label must be a string")
    cleaned = label.strip()
    if not cleaned:
        raise ProjectValidationError("label must be non-empty")
    if len(cleaned) > MAX_LABEL_LEN:
        raise ProjectValidationError(
            f"label must be ≤ {MAX_LABEL_LEN} chars"
        )
    if _CONTROL_CHAR_RE.search(cleaned):
        raise ProjectValidationError("label must not contain control characters")
    return cleaned


def _validate_color(color: str) -> str:
    """Validate a category color; return the canonical lowercase form.

    Empty string is allowed and means "no colour assigned".
    """
    if not isinstance(color, str):
        raise ProjectValidationError("color must be a string")
    if color == "":
        return ""
    if not CATEGORY_COLOR_RE.match(color):
        raise ProjectValidationError(
            f"color must be #rrggbb (got {color!r})"
        )
    return color.lower()


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class MemoCanvasCard:
    """One memo's position on the sorting canvas.

    The presence of a card means "this memo is on the canvas". A memo
    can exist in the project without appearing on the canvas (the
    canvas is a *working* surface, not a master list).

    ``memo_id`` references :class:`scribe.memos.Memo.id`. We don't
    fail loudly if the memo has been deleted — the audit trail
    tolerates dangling references the same way ``MemoLink`` does
    (F5.1) — but :func:`prune_orphans` will sweep them on demand.
    """

    memo_id: str
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return {"memo_id": self.memo_id, "x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoCanvasCard":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("MemoCanvasCard payload must be an object")
        for required in ("memo_id", "x", "y"):
            if required not in d:
                raise ProjectValidationError(
                    f"MemoCanvasCard payload missing required key: {required}"
                )
        return cls(
            memo_id=str(d["memo_id"]),
            x=_coerce_coord(d["x"], name="x"),
            y=_coerce_coord(d["y"], name="y"),
        )

    def validate(self) -> None:
        if not MEMO_ID_RE.match(self.memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {self.memo_id!r}"
            )
        # Coerce coordinates through the same path so on-disk shapes
        # are canonical (e.g. ``int`` becomes ``float``).
        self.x = _coerce_coord(self.x, name="x")
        self.y = _coerce_coord(self.y, name="y")


@dataclass
class MemoCanvasCategory:
    """One named cluster ("category") on the canvas.

    Categories are canvas-only objects: they don't appear in the
    Memo entity, the codebook, or the project file. They exist so the
    researcher can cluster memos visually with a label and a colour.

    ``label`` is required (a category without a name is just an
    arbitrary group). ``color`` is optional ("" = unset). ``x`` / ``y``
    is the anchor where the UI draws the label; cluster geometry
    (which memos *appear* clustered) is the UI's call, not the data
    model's.
    """

    id: str
    label: str
    color: str = ""
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "x": float(self.x),
            "y": float(self.y),
        }
        if self.color:
            out["color"] = self.color
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoCanvasCategory":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "MemoCanvasCategory payload must be an object"
            )
        for required in ("id", "label"):
            if required not in d:
                raise ProjectValidationError(
                    f"MemoCanvasCategory payload missing required key: {required}"
                )
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", "") or ""),
            color=str(d.get("color", "") or ""),
            x=_coerce_coord(d.get("x", 0.0), name="x"),
            y=_coerce_coord(d.get("y", 0.0), name="y"),
        )

    def validate(self) -> None:
        if not CATEGORY_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"category id must be 12-char hex; got {self.id!r}"
            )
        self.label = _validate_label(self.label)
        self.color = _validate_color(self.color)
        self.x = _coerce_coord(self.x, name="x")
        self.y = _coerce_coord(self.y, name="y")


@dataclass
class MemoCanvas:
    """The project-level singleton sorting canvas.

    Stores the cards (memo positions), the categories (named groups),
    and the per-category membership lists. One canvas per project;
    absent canvas file means empty canvas.
    """

    project_id: str
    cards: list[MemoCanvasCard] = field(default_factory=list)
    categories: list[MemoCanvasCategory] = field(default_factory=list)
    category_members: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = ""
    modified_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def empty(cls, project_id: str, *, now: str | None = None) -> "MemoCanvas":
        """Build a fresh empty canvas for a project."""
        ts = now or utcnow_iso()
        c = cls(
            project_id=project_id,
            cards=[],
            categories=[],
            category_members={},
            created_at=ts,
            modified_at=ts,
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "cards": [card.to_dict() for card in self.cards],
            "categories": [cat.to_dict() for cat in self.categories],
            "category_members": {
                cat_id: list(members)
                for cat_id, members in self.category_members.items()
            },
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoCanvas":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("MemoCanvas payload must be an object")
        if "project_id" not in d:
            raise ProjectValidationError(
                "MemoCanvas payload missing required key: project_id"
            )
        raw_cards = d.get("cards") or []
        if not isinstance(raw_cards, list):
            raise ProjectValidationError("cards must be a list")
        raw_cats = d.get("categories") or []
        if not isinstance(raw_cats, list):
            raise ProjectValidationError("categories must be a list")
        raw_members = d.get("category_members") or {}
        if not isinstance(raw_members, dict):
            raise ProjectValidationError("category_members must be an object")
        members: dict[str, list[str]] = {}
        for k, v in raw_members.items():
            if not isinstance(v, list):
                raise ProjectValidationError(
                    f"category_members[{k!r}] must be a list"
                )
            members[str(k)] = [str(x) for x in v]
        c = cls(
            project_id=str(d["project_id"]),
            cards=[MemoCanvasCard.from_dict(x) for x in raw_cards],
            categories=[MemoCanvasCategory.from_dict(x) for x in raw_cats],
            category_members=members,
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        # Cards: validate each, dedupe by memo_id (last-write-wins,
        # mirroring how move_card overwrites in place).
        if not isinstance(self.cards, list):
            raise ProjectValidationError("cards must be a list")
        if len(self.cards) > MAX_CARDS:
            raise ProjectValidationError(
                f"At most {MAX_CARDS} cards allowed"
            )
        seen_cards: dict[str, MemoCanvasCard] = {}
        for card in self.cards:
            if not isinstance(card, MemoCanvasCard):
                raise ProjectValidationError(
                    "cards entries must be MemoCanvasCard"
                )
            card.validate()
            seen_cards[card.memo_id] = card
        # Preserve insertion order (= the order they were first added);
        # keep a single entry per memo_id with the latest position.
        ordered: list[MemoCanvasCard] = []
        seen_ids: set[str] = set()
        for card in self.cards:
            if card.memo_id in seen_ids:
                continue
            seen_ids.add(card.memo_id)
            ordered.append(seen_cards[card.memo_id])
        self.cards = ordered

        # Categories: validate each, no duplicate ids, no duplicate
        # labels (case-sensitive — researchers do use "Care" and "care"
        # to mean different things, but if they want exact dupes they
        # almost certainly meant to update an existing one).
        if not isinstance(self.categories, list):
            raise ProjectValidationError("categories must be a list")
        if len(self.categories) > MAX_CATEGORIES:
            raise ProjectValidationError(
                f"At most {MAX_CATEGORIES} categories allowed"
            )
        seen_cat_ids: set[str] = set()
        seen_labels: set[str] = set()
        cleaned_cats: list[MemoCanvasCategory] = []
        for cat in self.categories:
            if not isinstance(cat, MemoCanvasCategory):
                raise ProjectValidationError(
                    "categories entries must be MemoCanvasCategory"
                )
            cat.validate()
            if cat.id in seen_cat_ids:
                raise ProjectValidationError(
                    f"Duplicate category id: {cat.id!r}"
                )
            if cat.label in seen_labels:
                raise ProjectValidationError(
                    f"Duplicate category label: {cat.label!r}"
                )
            seen_cat_ids.add(cat.id)
            seen_labels.add(cat.label)
            cleaned_cats.append(cat)
        self.categories = cleaned_cats

        # category_members keys must reference real categories; values
        # must be lists of valid memo_ids without duplicates.
        if not isinstance(self.category_members, dict):
            raise ProjectValidationError("category_members must be an object")
        cleaned_members: dict[str, list[str]] = {}
        for cat_id, members in self.category_members.items():
            if cat_id not in seen_cat_ids:
                raise ProjectValidationError(
                    f"category_members references unknown category: "
                    f"{cat_id!r}"
                )
            if not isinstance(members, list):
                raise ProjectValidationError(
                    f"category_members[{cat_id!r}] must be a list"
                )
            if len(members) > MAX_MEMBERS_PER_CATEGORY:
                raise ProjectValidationError(
                    f"category {cat_id!r} has too many members "
                    f"(>{MAX_MEMBERS_PER_CATEGORY})"
                )
            seen_in_cat: set[str] = set()
            cleaned_one: list[str] = []
            for raw_m in members:
                m = str(raw_m)
                if not MEMO_ID_RE.match(m):
                    raise ProjectValidationError(
                        f"category_members[{cat_id!r}] contains invalid "
                        f"memo id: {m!r}"
                    )
                if m in seen_in_cat:
                    continue
                seen_in_cat.add(m)
                cleaned_one.append(m)
            cleaned_members[cat_id] = cleaned_one
        # Drop entries for categories that exist but have no members
        # only if the caller never set them; an empty list is an
        # explicit "this category exists with zero members" — keep it.
        self.category_members = cleaned_members

    # ------------------------------------------------------------------ #
    # Card mutations
    # ------------------------------------------------------------------ #

    def card_for_memo(self, memo_id: str) -> MemoCanvasCard | None:
        for card in self.cards:
            if card.memo_id == memo_id:
                return card
        return None

    def move_card(
        self, memo_id: str, x: float, y: float, *, now: str | None = None
    ) -> MemoCanvasCard:
        """Set or insert a card for ``memo_id`` at ``(x, y)``.

        Idempotent: calling twice with the same coordinates leaves the
        canvas in the same state (still bumps ``modified_at`` so the
        audit trail records the touch).
        """
        if not MEMO_ID_RE.match(memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {memo_id!r}"
            )
        cx = _coerce_coord(x, name="x")
        cy = _coerce_coord(y, name="y")
        existing = self.card_for_memo(memo_id)
        if existing is not None:
            existing.x = cx
            existing.y = cy
            self.modified_at = now or utcnow_iso()
            return existing
        if len(self.cards) >= MAX_CARDS:
            raise ProjectValidationError(
                f"At most {MAX_CARDS} cards allowed"
            )
        card = MemoCanvasCard(memo_id=memo_id, x=cx, y=cy)
        card.validate()
        self.cards.append(card)
        self.modified_at = now or utcnow_iso()
        return card

    def remove_card(self, memo_id: str, *, now: str | None = None) -> bool:
        """Remove the card for ``memo_id`` and any category memberships.

        Returns False if the memo wasn't on the canvas. The Memo
        entity itself is not touched — removing a card just takes the
        memo off the sorting board.
        """
        if not MEMO_ID_RE.match(memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {memo_id!r}"
            )
        before = len(self.cards)
        self.cards = [c for c in self.cards if c.memo_id != memo_id]
        if len(self.cards) == before:
            return False
        # Also remove from every category's membership list.
        for cat_id, members in self.category_members.items():
            self.category_members[cat_id] = [
                m for m in members if m != memo_id
            ]
        self.modified_at = now or utcnow_iso()
        return True

    # ------------------------------------------------------------------ #
    # Category mutations
    # ------------------------------------------------------------------ #

    def category_for_id(self, category_id: str) -> MemoCanvasCategory | None:
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return None

    def category_for_label(self, label: str) -> MemoCanvasCategory | None:
        for cat in self.categories:
            if cat.label == label:
                return cat
        return None

    def add_category(
        self,
        *,
        label: str,
        color: str = "",
        x: float = 0.0,
        y: float = 0.0,
        category_id: str | None = None,
        now: str | None = None,
    ) -> MemoCanvasCategory:
        """Create a new category. Returns it.

        Labels must be unique within the canvas (case-sensitive); a
        duplicate raises :class:`ProjectValidationError`. Pass
        ``category_id`` to mint with a known id (used by tests and
        REFI-QDA round-tripping later).
        """
        clean_label = _validate_label(label)
        clean_color = _validate_color(color)
        cx = _coerce_coord(x, name="x")
        cy = _coerce_coord(y, name="y")

        if self.category_for_label(clean_label) is not None:
            raise ProjectValidationError(
                f"Duplicate category label: {clean_label!r}"
            )
        if len(self.categories) >= MAX_CATEGORIES:
            raise ProjectValidationError(
                f"At most {MAX_CATEGORIES} categories allowed"
            )

        cid = category_id or new_category_id()
        if not CATEGORY_ID_RE.match(cid):
            raise ProjectValidationError(
                f"category id must be 12-char hex; got {cid!r}"
            )
        if self.category_for_id(cid) is not None:
            raise ProjectValidationError(
                f"Duplicate category id: {cid!r}"
            )

        cat = MemoCanvasCategory(
            id=cid,
            label=clean_label,
            color=clean_color,
            x=cx,
            y=cy,
        )
        cat.validate()
        self.categories.append(cat)
        self.category_members[cid] = []
        self.modified_at = now or utcnow_iso()
        return cat

    def update_category(
        self,
        category_id: str,
        *,
        label: str | None = None,
        color: str | None = None,
        x: float | None = None,
        y: float | None = None,
        now: str | None = None,
    ) -> MemoCanvasCategory:
        """Patch one or more fields on an existing category.

        Renaming a category to an existing label raises (case-sensitive
        uniqueness, same rule as :meth:`add_category`). Returns the
        updated category.
        """
        cat = self.category_for_id(category_id)
        if cat is None:
            raise ProjectValidationError(
                f"Unknown category id: {category_id!r}"
            )
        if label is not None:
            clean_label = _validate_label(label)
            if clean_label != cat.label:
                clash = self.category_for_label(clean_label)
                if clash is not None and clash.id != category_id:
                    raise ProjectValidationError(
                        f"Duplicate category label: {clean_label!r}"
                    )
                cat.label = clean_label
        if color is not None:
            cat.color = _validate_color(color)
        if x is not None:
            cat.x = _coerce_coord(x, name="x")
        if y is not None:
            cat.y = _coerce_coord(y, name="y")
        cat.validate()
        self.modified_at = now or utcnow_iso()
        return cat

    def remove_category(
        self, category_id: str, *, now: str | None = None
    ) -> bool:
        """Drop a category and its membership list.

        Cards are NOT removed from the canvas; they just lose the
        association. Returns False if the category was not present.
        """
        if not CATEGORY_ID_RE.match(category_id):
            raise ProjectValidationError(
                f"category id must be 12-char hex; got {category_id!r}"
            )
        before = len(self.categories)
        self.categories = [c for c in self.categories if c.id != category_id]
        if len(self.categories) == before:
            return False
        self.category_members.pop(category_id, None)
        self.modified_at = now or utcnow_iso()
        return True

    # ------------------------------------------------------------------ #
    # Membership mutations
    # ------------------------------------------------------------------ #

    def assign_card_to_category(
        self,
        memo_id: str,
        category_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        """Make ``memo_id`` a member of ``category_id``.

        The memo must already have a card on the canvas (use
        :meth:`move_card` first). The category must exist. Returns
        True if the assignment was added, False if it was already
        present (idempotent).
        """
        if not MEMO_ID_RE.match(memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {memo_id!r}"
            )
        if self.card_for_memo(memo_id) is None:
            raise ProjectValidationError(
                f"memo {memo_id!r} is not on the canvas; "
                "place it with move_card first"
            )
        if self.category_for_id(category_id) is None:
            raise ProjectValidationError(
                f"Unknown category id: {category_id!r}"
            )
        members = self.category_members.setdefault(category_id, [])
        if memo_id in members:
            return False
        if len(members) >= MAX_MEMBERS_PER_CATEGORY:
            raise ProjectValidationError(
                f"category {category_id!r} already has the maximum "
                f"{MAX_MEMBERS_PER_CATEGORY} members"
            )
        members.append(memo_id)
        self.modified_at = now or utcnow_iso()
        return True

    def unassign_card_from_category(
        self,
        memo_id: str,
        category_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        """Remove ``memo_id`` from ``category_id``.

        Returns False if the memo wasn't a member; never raises for
        that case (idempotent removal). Raises if the category id is
        invalid or unknown.
        """
        if not MEMO_ID_RE.match(memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {memo_id!r}"
            )
        if self.category_for_id(category_id) is None:
            raise ProjectValidationError(
                f"Unknown category id: {category_id!r}"
            )
        members = self.category_members.get(category_id, [])
        if memo_id not in members:
            return False
        self.category_members[category_id] = [
            m for m in members if m != memo_id
        ]
        self.modified_at = now or utcnow_iso()
        return True

    def categories_for_memo(self, memo_id: str) -> list[str]:
        """Return the list of category ids that ``memo_id`` belongs to.

        Order follows the canvas's category order (= insertion order).
        """
        if not MEMO_ID_RE.match(memo_id):
            raise ProjectValidationError(
                f"memo_id must be 12-char hex; got {memo_id!r}"
            )
        out: list[str] = []
        for cat in self.categories:
            if memo_id in self.category_members.get(cat.id, []):
                out.append(cat.id)
        return out


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def canvas_state_path(projects_root: Path, project_id: str) -> Path:
    """Return the path to a project's canvas file.

    Validates ``project_id`` to prevent traversal. Does not create the
    parent directory.
    """
    return project_dir(projects_root, project_id) / CANVAS_FILENAME


def save_canvas(projects_root: Path, canvas: MemoCanvas) -> Path:
    """Persist a canvas to ``<projects_root>/<pid>/memo_canvas.json``.

    The parent ``projects/<pid>`` directory must already exist. Same
    convention as ``save_memo`` / ``save_source``.
    """
    canvas.validate()
    parent = project_dir(projects_root, canvas.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its memo canvas."
        )
    parent.mkdir(parents=True, exist_ok=True)
    target = canvas_state_path(projects_root, canvas.project_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canvas.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_canvas(projects_root: Path, project_id: str) -> MemoCanvas:
    """Load the project's canvas, returning an empty one if no file
    exists yet.

    The empty-default behaviour is deliberate: most projects will not
    have used the canvas, and the UI wants a stable shape regardless.
    The returned empty canvas's ``created_at`` and ``modified_at``
    reflect the *moment of first read*, not the creation of the
    project — that's the common convention for lazy singletons.
    """
    p = canvas_state_path(projects_root, project_id)
    if not p.exists():
        return MemoCanvas.empty(project_id)
    return MemoCanvas.from_dict(json.loads(p.read_text()))


def delete_canvas(projects_root: Path, project_id: str) -> bool:
    """Remove a project's canvas file. Returns False if absent."""
    p = canvas_state_path(projects_root, project_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True


# --------------------------------------------------------------------------- #
# Memo→memo links — wraps the existing memos API
# --------------------------------------------------------------------------- #


def link_memos_on_canvas(
    projects_root: Path,
    project_id: str,
    *,
    from_memo_id: str,
    to_memo_id: str,
    role: str = "",
    now: str | None = None,
) -> Memo:
    """Add a memo→memo link from ``from_memo_id`` to ``to_memo_id``.

    Persists by reading the source memo, appending a :class:`MemoLink`
    with ``target_type = "memo"``, and saving. Idempotent: if the
    triple ``(target_type="memo", target_id=to_memo_id, role)`` is
    already present, returns the unchanged memo (no save, no
    timestamp bump). Self-links (a memo linking to itself) raise — F5.3
    is about *between-memo* synthesis.

    Returns the (possibly mutated) source memo. The canvas itself is
    untouched: this helper exists so callers can wire memo→memo edges
    from the canvas surface without hand-rolling a Memo round-trip.
    """
    if not MEMO_ID_RE.match(from_memo_id):
        raise ProjectValidationError(
            f"from_memo_id must be 12-char hex; got {from_memo_id!r}"
        )
    if not MEMO_ID_RE.match(to_memo_id):
        raise ProjectValidationError(
            f"to_memo_id must be 12-char hex; got {to_memo_id!r}"
        )
    if from_memo_id == to_memo_id:
        raise ProjectValidationError(
            "Cannot link a memo to itself"
        )

    memo = load_memo(projects_root, project_id, from_memo_id)
    cleaned_role = (role or "").strip()
    for existing in memo.links:
        if (
            existing.target_type == "memo"
            and existing.target_id == to_memo_id
            and existing.role == cleaned_role
        ):
            return memo

    new_links = list(memo.links) + [
        MemoLink(target_type="memo", target_id=to_memo_id, role=cleaned_role)
    ]
    memo.apply_update({"links": [link.to_dict() for link in new_links]}, now=now)
    save_memo(projects_root, memo)
    return memo


# --------------------------------------------------------------------------- #
# Pure layout helpers (used by tests + UI)
# --------------------------------------------------------------------------- #


def clamp_to_bounds(
    x: float,
    y: float,
    *,
    min_x: float = -MAX_COORD,
    min_y: float = -MAX_COORD,
    max_x: float = MAX_COORD,
    max_y: float = MAX_COORD,
) -> tuple[float, float]:
    """Clamp a coordinate pair to the given inclusive bounds.

    Validates that the input is finite (rejects NaN / inf) so a
    misbehaving drag handler can't poison the canvas with garbage.
    Returns ``(x, y)`` as floats.
    """
    cx = _coerce_coord(x, name="x")
    cy = _coerce_coord(y, name="y")
    if min_x > max_x or min_y > max_y:
        raise ProjectValidationError(
            "min bounds must be ≤ max bounds"
        )
    cx = max(min_x, min(max_x, cx))
    cy = max(min_y, min(max_y, cy))
    return cx, cy


def snap_to_grid(x: float, y: float, *, grid: float = 1.0) -> tuple[float, float]:
    """Snap a coordinate to the nearest multiple of ``grid``.

    A grid of ``1.0`` (default) is a no-op rounding; the typical
    on-screen grid is something like 16 px. Raises if ``grid <= 0``
    (a zero-spacing grid is undefined).
    """
    if grid <= 0:
        raise ProjectValidationError(f"grid must be positive; got {grid!r}")
    cx = _coerce_coord(x, name="x")
    cy = _coerce_coord(y, name="y")
    return (round(cx / grid) * grid, round(cy / grid) * grid)


def hit_test_card(
    cards: Iterable[MemoCanvasCard],
    x: float,
    y: float,
    *,
    half_width: float = 80.0,
    half_height: float = 50.0,
) -> str | None:
    """Return the ``memo_id`` of the topmost card under ``(x, y)``.

    Uses an axis-aligned bounding box around each card centred on
    ``(card.x, card.y)`` with the given half-extents. "Topmost" =
    last in the cards list (latest add wins for layered hit-testing,
    same as DOM z-order semantics).

    Returns ``None`` if no card is hit. Pure; the canvas itself is
    not mutated.
    """
    cx = _coerce_coord(x, name="x")
    cy = _coerce_coord(y, name="y")
    if half_width <= 0 or half_height <= 0:
        raise ProjectValidationError(
            "half_width and half_height must be positive"
        )
    hit: str | None = None
    for card in cards:
        if (
            abs(card.x - cx) <= half_width
            and abs(card.y - cy) <= half_height
        ):
            hit = card.memo_id  # later = on top
    return hit


def prune_orphans(
    canvas: MemoCanvas,
    valid_memo_ids: Iterable[str],
    *,
    now: str | None = None,
) -> tuple[int, int]:
    """Remove cards / memberships whose memo no longer exists.

    Returns ``(cards_removed, members_removed)``. ``valid_memo_ids``
    is the set of memo ids the caller treats as "real"; anything else
    on the canvas gets swept. Categories themselves are preserved
    (a category with zero members is still a meaningful canvas
    object).
    """
    valid = {str(m) for m in valid_memo_ids}
    cards_before = len(canvas.cards)
    canvas.cards = [c for c in canvas.cards if c.memo_id in valid]
    cards_removed = cards_before - len(canvas.cards)

    members_removed = 0
    for cat_id, members in list(canvas.category_members.items()):
        kept = [m for m in members if m in valid]
        members_removed += len(members) - len(kept)
        canvas.category_members[cat_id] = kept

    if cards_removed or members_removed:
        canvas.modified_at = now or utcnow_iso()
    return cards_removed, members_removed
