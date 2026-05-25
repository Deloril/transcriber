"""Tests for scribe.memo_canvas (F5.3 — memo-sorting canvas).

The canvas is a project-level singleton holding card positions
(memo→x,y), categories (named clusters), and category memberships
(category→list of memo ids). Memo→memo links continue to live on the
Memo entity itself (F5.1's MemoLink); this module wraps the round-trip
with :func:`link_memos_on_canvas`.

Tests cover:

* Pure layout helpers — ``clamp_to_bounds`` / ``snap_to_grid`` /
  ``hit_test_card``.
* Data model invariants — id shapes, label / colour / coordinate
  validation, dedupe rules.
* Mutation semantics — move / remove cards, add / update / remove
  categories, assign / unassign memberships.
* Persistence — save / load round-trip, lazy "no file = empty canvas".
* Cross-entity linkage — link_memos_on_canvas wraps the Memo entity.
* :func:`prune_orphans` — sweeps cards and memberships when memos are
  deleted from the project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scribe.memos import (
    Memo,
    MemoLink,
    load_memo,
    save_memo,
)
from scribe.memo_canvas import (
    CANVAS_FILENAME,
    CATEGORY_COLOR_RE,
    CATEGORY_ID_RE,
    MAX_CARDS,
    MAX_CATEGORIES,
    MAX_COORD,
    MAX_LABEL_LEN,
    MAX_MEMBERS_PER_CATEGORY,
    MemoCanvas,
    MemoCanvasCard,
    MemoCanvasCategory,
    canvas_state_path,
    clamp_to_bounds,
    delete_canvas,
    hit_test_card,
    link_memos_on_canvas,
    load_canvas,
    new_category_id,
    prune_orphans,
    save_canvas,
    snap_to_grid,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


PROJECT_ID = "0" * 12
MEMO_A = "a" * 12
MEMO_B = "b" * 12
MEMO_C = "c" * 12
CAT_X = "1" * 12
CAT_Y = "2" * 12


def _saved_project(tmp_path: Path, *, name: str = "Canvas Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _empty_canvas(project_id: str = PROJECT_ID) -> MemoCanvas:
    return MemoCanvas.empty(project_id)


# --------------------------------------------------------------------------- #
# new_category_id
# --------------------------------------------------------------------------- #


class TestNewCategoryId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert CATEGORY_ID_RE.match(new_category_id())

    def test_unique(self) -> None:
        ids = {new_category_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Pure layout helpers
# --------------------------------------------------------------------------- #


class TestClampToBounds:
    def test_within_bounds_pass_through(self) -> None:
        assert clamp_to_bounds(10.0, 20.0, min_x=0, min_y=0, max_x=100, max_y=100) == (10.0, 20.0)

    def test_clamped_low(self) -> None:
        assert clamp_to_bounds(-5, -5, min_x=0, min_y=0, max_x=100, max_y=100) == (0.0, 0.0)

    def test_clamped_high(self) -> None:
        assert clamp_to_bounds(500, 500, min_x=0, min_y=0, max_x=100, max_y=100) == (100.0, 100.0)

    def test_returns_floats(self) -> None:
        x, y = clamp_to_bounds(3, 4, min_x=0, min_y=0, max_x=100, max_y=100)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_default_bounds_use_max_coord(self) -> None:
        # Default bounds are ±MAX_COORD; values inside pass through.
        x, y = clamp_to_bounds(123456.0, -789.0)
        assert x == 123456.0
        assert y == -789.0

    def test_rejects_nan(self) -> None:
        with pytest.raises(ProjectValidationError):
            clamp_to_bounds(float("nan"), 0.0)

    def test_rejects_inf(self) -> None:
        with pytest.raises(ProjectValidationError):
            clamp_to_bounds(0.0, float("inf"))

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ProjectValidationError):
            clamp_to_bounds(0.0, 0.0, min_x=10, max_x=0, min_y=0, max_y=10)

    def test_rejects_bool(self) -> None:
        with pytest.raises(ProjectValidationError):
            clamp_to_bounds(True, 0.0)


class TestSnapToGrid:
    def test_grid_one_is_round_to_int(self) -> None:
        assert snap_to_grid(3.4, 7.6) == (3.0, 8.0)

    def test_grid_sixteen(self) -> None:
        # 22 → nearest multiple of 16 is 16; 25 → 32.
        assert snap_to_grid(22, 25, grid=16) == (16.0, 32.0)

    def test_zero_grid_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            snap_to_grid(1.0, 1.0, grid=0)

    def test_negative_grid_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            snap_to_grid(1.0, 1.0, grid=-4)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ProjectValidationError):
            snap_to_grid(float("nan"), 0)


class TestHitTestCard:
    def test_no_cards(self) -> None:
        assert hit_test_card([], 0, 0) is None

    def test_inside_box(self) -> None:
        cards = [MemoCanvasCard(memo_id=MEMO_A, x=100.0, y=100.0)]
        assert hit_test_card(cards, 110, 95, half_width=80, half_height=50) == MEMO_A

    def test_outside_box(self) -> None:
        cards = [MemoCanvasCard(memo_id=MEMO_A, x=100.0, y=100.0)]
        assert hit_test_card(cards, 300, 300, half_width=80, half_height=50) is None

    def test_topmost_wins_when_overlapping(self) -> None:
        # Both cards' boxes contain (100, 100); the one later in the
        # list is "on top" — same as DOM stacking order.
        cards = [
            MemoCanvasCard(memo_id=MEMO_A, x=100.0, y=100.0),
            MemoCanvasCard(memo_id=MEMO_B, x=110.0, y=110.0),
        ]
        assert hit_test_card(cards, 105, 105, half_width=80, half_height=50) == MEMO_B

    def test_zero_half_extent_raises(self) -> None:
        cards = [MemoCanvasCard(memo_id=MEMO_A, x=0, y=0)]
        with pytest.raises(ProjectValidationError):
            hit_test_card(cards, 0, 0, half_width=0, half_height=10)
        with pytest.raises(ProjectValidationError):
            hit_test_card(cards, 0, 0, half_width=10, half_height=0)


# --------------------------------------------------------------------------- #
# MemoCanvasCard
# --------------------------------------------------------------------------- #


class TestMemoCanvasCardValidate:
    def test_round_trip(self) -> None:
        c = MemoCanvasCard(memo_id=MEMO_A, x=12.5, y=-3.5)
        c.validate()
        d = c.to_dict()
        c2 = MemoCanvasCard.from_dict(d)
        c2.validate()
        assert c2.memo_id == MEMO_A
        assert c2.x == 12.5
        assert c2.y == -3.5

    def test_invalid_memo_id_raises(self) -> None:
        c = MemoCanvasCard(memo_id="not-hex", x=0, y=0)
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_nan_coord_raises(self) -> None:
        c = MemoCanvasCard(memo_id=MEMO_A, x=float("nan"), y=0)
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_inf_coord_raises(self) -> None:
        c = MemoCanvasCard(memo_id=MEMO_A, x=0, y=float("inf"))
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_too_large_coord_raises(self) -> None:
        c = MemoCanvasCard(memo_id=MEMO_A, x=MAX_COORD * 2, y=0)
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_from_dict_missing_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvasCard.from_dict({"memo_id": MEMO_A, "x": 0})  # missing y

    def test_from_dict_not_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvasCard.from_dict("not-a-dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# MemoCanvasCategory
# --------------------------------------------------------------------------- #


class TestMemoCanvasCategoryValidate:
    def test_round_trip(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="Care", color="#FF8800", x=10, y=20)
        cat.validate()
        # Color is canonicalised to lowercase.
        assert cat.color == "#ff8800"
        d = cat.to_dict()
        cat2 = MemoCanvasCategory.from_dict(d)
        cat2.validate()
        assert cat2.id == CAT_X
        assert cat2.label == "Care"
        assert cat2.color == "#ff8800"

    def test_empty_label_raises(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="   ", color="")
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_invalid_id_raises(self) -> None:
        cat = MemoCanvasCategory(id="not-hex", label="Care")
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_no_color_is_empty_string(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="Care")
        cat.validate()
        assert cat.color == ""
        d = cat.to_dict()
        # Empty colour is omitted from on-disk shape (compact).
        assert "color" not in d

    def test_bad_color_format(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="Care", color="red")
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_short_color_format_rejected(self) -> None:
        # We accept only #rrggbb, not #rgb.
        cat = MemoCanvasCategory(id=CAT_X, label="Care", color="#f80")
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_label_with_control_chars_rejected(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="bad\x01label")
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_label_too_long(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="x" * (MAX_LABEL_LEN + 1))
        with pytest.raises(ProjectValidationError):
            cat.validate()

    def test_label_canonicalised_strip(self) -> None:
        cat = MemoCanvasCategory(id=CAT_X, label="  Care  ")
        cat.validate()
        assert cat.label == "Care"


# --------------------------------------------------------------------------- #
# MemoCanvas — empty / new / serialisation
# --------------------------------------------------------------------------- #


class TestMemoCanvasEmpty:
    def test_empty_has_project_id(self) -> None:
        c = MemoCanvas.empty(PROJECT_ID)
        assert c.project_id == PROJECT_ID
        assert c.cards == []
        assert c.categories == []
        assert c.category_members == {}

    def test_empty_has_timestamps(self) -> None:
        c = MemoCanvas.empty(PROJECT_ID)
        assert c.created_at != ""
        assert c.modified_at != ""

    def test_empty_invalid_project_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.empty("not-hex")


class TestMemoCanvasFromDict:
    def test_minimal(self) -> None:
        c = MemoCanvas.from_dict({"project_id": PROJECT_ID})
        assert c.project_id == PROJECT_ID
        assert c.cards == []
        assert c.categories == []

    def test_round_trip(self) -> None:
        c = MemoCanvas.empty(PROJECT_ID)
        c.move_card(MEMO_A, 10, 20)
        cat = c.add_category(label="Care")
        c.assign_card_to_category(MEMO_A, cat.id)
        d = c.to_dict()
        c2 = MemoCanvas.from_dict(d)
        assert c2.cards[0].memo_id == MEMO_A
        assert c2.cards[0].x == 10.0
        assert c2.cards[0].y == 20.0
        assert c2.categories[0].id == cat.id
        assert c2.category_members[cat.id] == [MEMO_A]

    def test_missing_project_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.from_dict({"cards": []})

    def test_cards_not_list_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.from_dict({"project_id": PROJECT_ID, "cards": "not-list"})

    def test_categories_not_list_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.from_dict(
                {"project_id": PROJECT_ID, "categories": "not-list"}
            )

    def test_category_members_not_dict_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.from_dict(
                {"project_id": PROJECT_ID, "category_members": "not-dict"}
            )

    def test_category_members_value_not_list_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoCanvas.from_dict(
                {
                    "project_id": PROJECT_ID,
                    "categories": [{"id": CAT_X, "label": "Care"}],
                    "category_members": {CAT_X: "not-list"},
                }
            )


class TestMemoCanvasValidate:
    def test_dupe_card_memo_id_dedupes_to_latest(self) -> None:
        # The on-disk file shouldn't normally contain duplicates, but
        # if it does we keep the last position (matching move_card's
        # last-write-wins semantics).
        c = MemoCanvas(
            project_id=PROJECT_ID,
            cards=[
                MemoCanvasCard(memo_id=MEMO_A, x=10, y=10),
                MemoCanvasCard(memo_id=MEMO_A, x=99, y=99),
            ],
        )
        c.validate()
        assert len(c.cards) == 1
        assert c.cards[0].x == 99
        assert c.cards[0].y == 99

    def test_dupe_category_id_raises(self) -> None:
        c = MemoCanvas(
            project_id=PROJECT_ID,
            categories=[
                MemoCanvasCategory(id=CAT_X, label="A"),
                MemoCanvasCategory(id=CAT_X, label="B"),
            ],
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_dupe_category_label_raises(self) -> None:
        c = MemoCanvas(
            project_id=PROJECT_ID,
            categories=[
                MemoCanvasCategory(id=CAT_X, label="Care"),
                MemoCanvasCategory(id=CAT_Y, label="Care"),
            ],
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_unknown_category_in_members_raises(self) -> None:
        c = MemoCanvas(
            project_id=PROJECT_ID,
            categories=[],
            category_members={CAT_X: [MEMO_A]},
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_invalid_memo_id_in_members_raises(self) -> None:
        c = MemoCanvas(
            project_id=PROJECT_ID,
            categories=[MemoCanvasCategory(id=CAT_X, label="Care")],
            category_members={CAT_X: ["not-hex"]},
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_dupe_member_dedupes(self) -> None:
        c = MemoCanvas(
            project_id=PROJECT_ID,
            categories=[MemoCanvasCategory(id=CAT_X, label="Care")],
            category_members={CAT_X: [MEMO_A, MEMO_A, MEMO_B]},
        )
        c.validate()
        assert c.category_members[CAT_X] == [MEMO_A, MEMO_B]

    def test_too_many_cards_raises(self) -> None:
        # Quick check: build a list near the cap (don't actually
        # build MAX_CARDS cards — keep the fixture cheap).
        c = MemoCanvas(project_id=PROJECT_ID)
        # All same memo_id collapses on validate; instead, fake the
        # post-dedupe count by patching the list length check.
        cards = []
        for i in range(MAX_CARDS + 1):
            # Use a unique-ish hex id per card to avoid the dedupe
            # path; we want the cap to fire.
            mid = f"{i:012x}"
            cards.append(MemoCanvasCard(memo_id=mid, x=0, y=0))
        c.cards = cards
        with pytest.raises(ProjectValidationError):
            c.validate()


# --------------------------------------------------------------------------- #
# Card mutations
# --------------------------------------------------------------------------- #


class TestMoveCard:
    def test_inserts_new_card(self) -> None:
        c = _empty_canvas()
        card = c.move_card(MEMO_A, 12, 34)
        assert card.memo_id == MEMO_A
        assert card.x == 12.0
        assert card.y == 34.0
        assert len(c.cards) == 1

    def test_updates_existing(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 10, 20)
        c.move_card(MEMO_A, 30, 40)
        assert len(c.cards) == 1
        assert c.cards[0].x == 30.0
        assert c.cards[0].y == 40.0

    def test_invalid_memo_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.move_card("not-hex", 0, 0)

    def test_nan_coord_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.move_card(MEMO_A, float("nan"), 0)

    def test_too_large_coord_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.move_card(MEMO_A, MAX_COORD * 2, 0)

    def test_bumps_modified_at(self) -> None:
        c = _empty_canvas()
        before = c.modified_at
        c.move_card(MEMO_A, 1, 1, now="2026-05-26T00:00:01Z")
        assert c.modified_at == "2026-05-26T00:00:01Z"
        assert c.modified_at != before

    def test_card_for_memo(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 10, 20)
        card = c.card_for_memo(MEMO_A)
        assert card is not None and card.x == 10.0
        assert c.card_for_memo(MEMO_B) is None


class TestRemoveCard:
    def test_removes_existing(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 10, 20)
        c.move_card(MEMO_B, 30, 40)
        assert c.remove_card(MEMO_A) is True
        assert [card.memo_id for card in c.cards] == [MEMO_B]

    def test_missing_returns_false(self) -> None:
        c = _empty_canvas()
        assert c.remove_card(MEMO_A) is False

    def test_invalid_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.remove_card("not-hex")

    def test_also_removes_from_categories(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 10, 20)
        c.move_card(MEMO_B, 30, 40)
        cat = c.add_category(label="Care")
        c.assign_card_to_category(MEMO_A, cat.id)
        c.assign_card_to_category(MEMO_B, cat.id)
        c.remove_card(MEMO_A)
        assert c.category_members[cat.id] == [MEMO_B]


# --------------------------------------------------------------------------- #
# Category mutations
# --------------------------------------------------------------------------- #


class TestAddCategory:
    def test_creates_with_id(self) -> None:
        c = _empty_canvas()
        cat = c.add_category(label="Care", color="#aabbcc", x=5, y=6)
        assert CATEGORY_ID_RE.match(cat.id)
        assert cat.label == "Care"
        assert cat.color == "#aabbcc"
        assert c.category_members[cat.id] == []

    def test_explicit_id(self) -> None:
        c = _empty_canvas()
        cat = c.add_category(label="Care", category_id=CAT_X)
        assert cat.id == CAT_X

    def test_dupe_label_raises(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care")
        with pytest.raises(ProjectValidationError):
            c.add_category(label="Care")

    def test_dupe_id_raises(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        with pytest.raises(ProjectValidationError):
            c.add_category(label="Pain", category_id=CAT_X)

    def test_invalid_category_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.add_category(label="Care", category_id="not-hex")

    def test_strip_label(self) -> None:
        c = _empty_canvas()
        cat = c.add_category(label="   Care   ")
        assert cat.label == "Care"


class TestUpdateCategory:
    def test_rename(self) -> None:
        c = _empty_canvas()
        cat = c.add_category(label="Care", category_id=CAT_X)
        c.update_category(CAT_X, label="Caring")
        assert cat.label == "Caring"

    def test_rename_to_existing_label_raises(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        c.add_category(label="Pain", category_id=CAT_Y)
        with pytest.raises(ProjectValidationError):
            c.update_category(CAT_X, label="Pain")

    def test_rename_to_same_label_is_noop(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        c.update_category(CAT_X, label="Care")
        assert c.category_for_id(CAT_X).label == "Care"

    def test_color_update(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X, color="#aaaaaa")
        c.update_category(CAT_X, color="#bbbbbb")
        assert c.category_for_id(CAT_X).color == "#bbbbbb"

    def test_clear_color(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X, color="#aaaaaa")
        c.update_category(CAT_X, color="")
        assert c.category_for_id(CAT_X).color == ""

    def test_move_anchor(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X, x=10, y=20)
        c.update_category(CAT_X, x=100, y=200)
        cat = c.category_for_id(CAT_X)
        assert cat.x == 100.0
        assert cat.y == 200.0

    def test_unknown_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.update_category(CAT_X, label="Care")


class TestRemoveCategory:
    def test_removes(self) -> None:
        c = _empty_canvas()
        cat = c.add_category(label="Care", category_id=CAT_X)
        assert c.remove_category(CAT_X) is True
        assert c.categories == []
        assert CAT_X not in c.category_members

    def test_missing_returns_false(self) -> None:
        c = _empty_canvas()
        assert c.remove_category(CAT_X) is False

    def test_invalid_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.remove_category("not-hex")

    def test_cards_not_removed(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 10, 20)
        cat = c.add_category(label="Care", category_id=CAT_X)
        c.assign_card_to_category(MEMO_A, CAT_X)
        c.remove_category(CAT_X)
        # Card stays on the canvas, just no longer associated.
        assert c.card_for_memo(MEMO_A) is not None


# --------------------------------------------------------------------------- #
# Membership mutations
# --------------------------------------------------------------------------- #


class TestAssignCardToCategory:
    def test_basic(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        cat = c.add_category(label="Care", category_id=CAT_X)
        assert c.assign_card_to_category(MEMO_A, CAT_X) is True
        assert c.category_members[CAT_X] == [MEMO_A]

    def test_idempotent(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.add_category(label="Care", category_id=CAT_X)
        assert c.assign_card_to_category(MEMO_A, CAT_X) is True
        assert c.assign_card_to_category(MEMO_A, CAT_X) is False
        assert c.category_members[CAT_X] == [MEMO_A]

    def test_card_must_be_on_canvas(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        with pytest.raises(ProjectValidationError):
            c.assign_card_to_category(MEMO_A, CAT_X)

    def test_unknown_category_raises(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        with pytest.raises(ProjectValidationError):
            c.assign_card_to_category(MEMO_A, CAT_X)

    def test_invalid_memo_id_raises(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        with pytest.raises(ProjectValidationError):
            c.assign_card_to_category("not-hex", CAT_X)


class TestUnassignCardFromCategory:
    def test_basic(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.add_category(label="Care", category_id=CAT_X)
        c.assign_card_to_category(MEMO_A, CAT_X)
        assert c.unassign_card_from_category(MEMO_A, CAT_X) is True
        assert c.category_members[CAT_X] == []

    def test_missing_member_returns_false(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        assert c.unassign_card_from_category(MEMO_A, CAT_X) is False

    def test_unknown_category_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.unassign_card_from_category(MEMO_A, CAT_X)

    def test_invalid_memo_id_raises(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        with pytest.raises(ProjectValidationError):
            c.unassign_card_from_category("not-hex", CAT_X)


class TestCategoriesForMemo:
    def test_basic(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.add_category(label="Care", category_id=CAT_X)
        c.add_category(label="Pain", category_id=CAT_Y)
        c.assign_card_to_category(MEMO_A, CAT_X)
        c.assign_card_to_category(MEMO_A, CAT_Y)
        assert c.categories_for_memo(MEMO_A) == [CAT_X, CAT_Y]

    def test_empty_when_unassigned(self) -> None:
        c = _empty_canvas()
        c.add_category(label="Care", category_id=CAT_X)
        assert c.categories_for_memo(MEMO_A) == []

    def test_invalid_memo_id_raises(self) -> None:
        c = _empty_canvas()
        with pytest.raises(ProjectValidationError):
            c.categories_for_memo("not-hex")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_load_returns_empty_when_absent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = load_canvas(tmp_path, proj.id)
        assert c.project_id == proj.id
        assert c.cards == []
        assert c.categories == []
        # No file on disk yet.
        assert not canvas_state_path(tmp_path, proj.id).exists()

    def test_save_then_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = MemoCanvas.empty(proj.id)
        c.move_card(MEMO_A, 12, 34)
        cat = c.add_category(label="Care")
        c.assign_card_to_category(MEMO_A, cat.id)
        save_canvas(tmp_path, c)

        loaded = load_canvas(tmp_path, proj.id)
        assert loaded.project_id == proj.id
        assert loaded.cards[0].memo_id == MEMO_A
        assert loaded.cards[0].x == 12.0
        assert loaded.categories[0].label == "Care"
        assert loaded.category_members[cat.id] == [MEMO_A]

    def test_save_rejects_when_project_dir_missing(self, tmp_path: Path) -> None:
        c = MemoCanvas.empty(PROJECT_ID)
        with pytest.raises(FileNotFoundError):
            save_canvas(tmp_path, c)

    def test_save_writes_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = MemoCanvas.empty(proj.id)
        c.move_card(MEMO_A, 1, 2)
        path = save_canvas(tmp_path, c)
        # Temp file is cleaned up.
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists()
        assert path.exists()
        # File parses as JSON.
        data = json.loads(path.read_text())
        assert data["project_id"] == proj.id

    def test_filename_constant(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = MemoCanvas.empty(proj.id)
        save_canvas(tmp_path, c)
        assert (project_dir(tmp_path, proj.id) / CANVAS_FILENAME).exists()

    def test_delete_canvas(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = MemoCanvas.empty(proj.id)
        c.move_card(MEMO_A, 1, 2)
        save_canvas(tmp_path, c)
        assert delete_canvas(tmp_path, proj.id) is True
        assert delete_canvas(tmp_path, proj.id) is False

    def test_delete_canvas_returns_false_when_absent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert delete_canvas(tmp_path, proj.id) is False


# --------------------------------------------------------------------------- #
# link_memos_on_canvas
# --------------------------------------------------------------------------- #


class TestLinkMemosOnCanvas:
    def _saved_memo(
        self, tmp_path: Path, project_id: str, *, body: str = "x"
    ) -> Memo:
        m = Memo.new(project_id=project_id, type="theoretical", body=body)
        save_memo(tmp_path, m)
        return m

    def test_adds_memo_link(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._saved_memo(tmp_path, proj.id, body="A")
        b = self._saved_memo(tmp_path, proj.id, body="B")
        link_memos_on_canvas(
            tmp_path,
            proj.id,
            from_memo_id=a.id,
            to_memo_id=b.id,
            role="elaborates",
        )
        loaded = load_memo(tmp_path, proj.id, a.id)
        assert any(
            link.target_type == "memo"
            and link.target_id == b.id
            and link.role == "elaborates"
            for link in loaded.links
        )

    def test_idempotent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._saved_memo(tmp_path, proj.id, body="A")
        b = self._saved_memo(tmp_path, proj.id, body="B")
        link_memos_on_canvas(
            tmp_path, proj.id, from_memo_id=a.id, to_memo_id=b.id, role=""
        )
        link_memos_on_canvas(
            tmp_path, proj.id, from_memo_id=a.id, to_memo_id=b.id, role=""
        )
        loaded = load_memo(tmp_path, proj.id, a.id)
        memo_links = [
            link for link in loaded.links if link.target_type == "memo"
        ]
        assert len(memo_links) == 1

    def test_distinct_role_creates_separate_link(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._saved_memo(tmp_path, proj.id, body="A")
        b = self._saved_memo(tmp_path, proj.id, body="B")
        link_memos_on_canvas(
            tmp_path, proj.id, from_memo_id=a.id, to_memo_id=b.id, role="elaborates"
        )
        link_memos_on_canvas(
            tmp_path, proj.id, from_memo_id=a.id, to_memo_id=b.id, role="contradicts"
        )
        loaded = load_memo(tmp_path, proj.id, a.id)
        roles = sorted(
            link.role
            for link in loaded.links
            if link.target_type == "memo" and link.target_id == b.id
        )
        assert roles == ["contradicts", "elaborates"]

    def test_self_link_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._saved_memo(tmp_path, proj.id, body="A")
        with pytest.raises(ProjectValidationError):
            link_memos_on_canvas(
                tmp_path,
                proj.id,
                from_memo_id=a.id,
                to_memo_id=a.id,
            )

    def test_invalid_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            link_memos_on_canvas(
                tmp_path,
                PROJECT_ID,
                from_memo_id="not-hex",
                to_memo_id=MEMO_B,
            )

    def test_missing_source_memo_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            link_memos_on_canvas(
                tmp_path,
                proj.id,
                from_memo_id=MEMO_A,
                to_memo_id=MEMO_B,
            )


# --------------------------------------------------------------------------- #
# prune_orphans
# --------------------------------------------------------------------------- #


class TestPruneOrphans:
    def test_removes_card_for_missing_memo(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.move_card(MEMO_B, 1, 1)
        cards_removed, members_removed = prune_orphans(c, [MEMO_A])
        assert cards_removed == 1
        assert members_removed == 0
        assert [card.memo_id for card in c.cards] == [MEMO_A]

    def test_removes_membership_for_missing_memo(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.move_card(MEMO_B, 1, 1)
        c.add_category(label="Care", category_id=CAT_X)
        c.assign_card_to_category(MEMO_A, CAT_X)
        c.assign_card_to_category(MEMO_B, CAT_X)
        cards_removed, members_removed = prune_orphans(c, [MEMO_A])
        assert cards_removed == 1
        assert members_removed == 1
        assert c.category_members[CAT_X] == [MEMO_A]

    def test_no_change_returns_zero(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        cards_removed, members_removed = prune_orphans(c, [MEMO_A])
        assert (cards_removed, members_removed) == (0, 0)

    def test_categories_themselves_preserved(self) -> None:
        c = _empty_canvas()
        c.move_card(MEMO_A, 0, 0)
        c.add_category(label="Care", category_id=CAT_X)
        c.assign_card_to_category(MEMO_A, CAT_X)
        prune_orphans(c, [])  # all memos gone
        # Category survives even with zero members.
        assert c.category_for_id(CAT_X) is not None
        assert c.category_members[CAT_X] == []
