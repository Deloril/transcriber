"""Tests for scribe.application_gutter (F4.3).

Per PLANNING.md F4.3:

  > Unlimited overlapping codes on a span; gutter/margin renderer.

This suite covers the lane-assignment algorithm, per-source
bucketing, stack-depth computation, ``applications_at_word``,
``lane_envelope``, and JSON serialisation. All tests are pure
Python; no persistence, no FastAPI.
"""

from __future__ import annotations

import pytest

from scribe.applications import Application, ProjectValidationError
from scribe.application_gutter import (
    GutterLayout,
    LanePlacement,
    applications_at_word,
    assign_lanes,
    assign_lanes_per_source,
    lane_envelope,
    serialise_layout,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE_A = "a" * 12
_HEX_CODE_B = "b" * 12
_HEX_CODE_C = "c" * 12
_HEX_SOURCE_1 = "1" * 12
_HEX_SOURCE_2 = "2" * 12
_HEX_CODER = "d" * 12
_HEX_VERSION = "e" * 12


def _hex_id(seed: int) -> str:
    """Twelve-char lowercase hex id from a non-negative integer seed."""
    return f"{seed:012x}"


def _app(
    *,
    code_id: str = _HEX_CODE_A,
    source_id: str = _HEX_SOURCE_1,
    coder_id: str = _HEX_CODER,
    start: str = "s0w0",
    end: str = "s0w5",
    start_offset: int | None = None,
    end_offset: int | None = None,
    application_id: str | None = None,
    now: str = "2026-01-01T00:00:00.000000Z",
) -> Application:
    """Build a valid Application for use in F4.3 layout tests."""
    return Application.new(
        project_id=_HEX_PROJECT,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start,
        anchor_end_word_id=end,
        definition_version_id_at_apply=_HEX_VERSION,
        start_char_offset=start_offset,
        end_char_offset=end_offset,
        application_id=application_id,
        now=now,
    )


# --------------------------------------------------------------------------- #
# assign_lanes — empty / trivial
# --------------------------------------------------------------------------- #


class TestAssignLanesEmpty:
    def test_empty_input(self) -> None:
        layout = assign_lanes([])
        assert layout.source_id == ""
        assert layout.placements == ()
        assert layout.lane_count == 0
        assert layout.max_stack_depth == 0

    def test_single_application(self) -> None:
        a = _app(application_id=_hex_id(1))
        layout = assign_lanes([a])
        assert layout.source_id == _HEX_SOURCE_1
        assert layout.lane_count == 1
        assert layout.max_stack_depth == 0
        assert layout.placements == (
            LanePlacement(application_id=a.id, lane=0, stack_depth=0),
        )

    def test_disjoint_pair_shares_one_lane(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w3")
        b = _app(application_id=_hex_id(2), start="s0w5", end="s0w9")
        layout = assign_lanes([a, b])
        assert layout.lane_count == 1
        assert layout.placements[0].lane == 0
        assert layout.placements[1].lane == 0
        assert layout.max_stack_depth == 0


# --------------------------------------------------------------------------- #
# assign_lanes — overlap behaviour
# --------------------------------------------------------------------------- #


class TestAssignLanesOverlap:
    def test_overlapping_pair_uses_two_lanes(self) -> None:
        # F4.3: overlapping codes are the *norm*, not an edge case.
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        layout = assign_lanes([a, b])
        assert layout.lane_count == 2
        assert layout.placements[0].lane == 0
        assert layout.placements[1].lane == 1
        # Both overlap each other: stack depth 1 each.
        assert all(p.stack_depth == 1 for p in layout.placements)
        assert layout.max_stack_depth == 1

    def test_six_codes_on_one_utterance(self) -> None:
        # The motivating example from coding-engine-research.md §4:
        # "A 30-word participant utterance routinely picks up 4–6
        # codes (topic, emotion, action, in-vivo phrase, reflexive
        # note)." All six must each get their own lane.
        apps = [
            _app(application_id=_hex_id(i), start="s0w0", end="s0w29")
            for i in range(1, 7)
        ]
        layout = assign_lanes(apps)
        assert layout.lane_count == 6
        # Every placement is on a distinct lane.
        assert {p.lane for p in layout.placements} == {0, 1, 2, 3, 4, 5}
        # Each one is overlapped by the other 5.
        assert all(p.stack_depth == 5 for p in layout.placements)
        assert layout.max_stack_depth == 5

    def test_lane_zero_reused_after_overlap_ends(self) -> None:
        # A in lane 0 (s0w0..s0w3); B in lane 1 because it overlaps A
        # (s0w2..s0w7). C starts at s0w10 — past A and B both — so
        # the algorithm must re-use lane 0, not open a third.
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w3")
        b = _app(application_id=_hex_id(2), start="s0w2", end="s0w7")
        c = _app(application_id=_hex_id(3), start="s0w10", end="s0w15")
        layout = assign_lanes([a, b, c])
        assert layout.lane_count == 2
        assert layout.placements[0].lane == 0
        assert layout.placements[1].lane == 1
        assert layout.placements[2].lane == 0  # reused
        # Stack depth: A↔B overlap; C is solo.
        depths = {p.application_id: p.stack_depth for p in layout.placements}
        assert depths[a.id] == 1
        assert depths[b.id] == 1
        assert depths[c.id] == 0

    def test_lowest_free_lane_preferred(self) -> None:
        # With 3 mutually overlapping then a fourth that only overlaps
        # the latest, the algorithm should re-use lane 0 for the
        # fourth (as soon as lane 0 frees up), not open lane 3.
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w2")
        b = _app(application_id=_hex_id(2), start="s0w1", end="s0w5")
        c = _app(application_id=_hex_id(3), start="s0w4", end="s0w9")
        d = _app(application_id=_hex_id(4), start="s0w6", end="s0w12")
        layout = assign_lanes([a, b, c, d])
        # A→0, B→1 (overlaps A), C→0 (A finished by w2 < w4),
        # D→1 (B finished by w5 < w6).
        assert layout.lane_count == 2
        lane_by_id = {p.application_id: p.lane for p in layout.placements}
        assert lane_by_id[a.id] == 0
        assert lane_by_id[b.id] == 1
        assert lane_by_id[c.id] == 0
        assert lane_by_id[d.id] == 1

    def test_touching_at_a_point_can_share_lane(self) -> None:
        # F4.2 spec: applications "touch" at a point (one ends where
        # the other starts) is *not* overlap. They can therefore
        # share a lane in the gutter.
        a = _app(
            application_id=_hex_id(1),
            start="s0w0",
            end="s0w5",
            end_offset=3,  # ends within s0w5 at offset 3
        )
        b = _app(
            application_id=_hex_id(2),
            start="s0w5",
            end="s0w9",
            start_offset=3,  # starts at s0w5 offset 3 — exactly where a ends
        )
        layout = assign_lanes([a, b])
        # Strictly touching: should fit in a single lane.
        assert layout.lane_count == 1
        assert all(p.stack_depth == 0 for p in layout.placements)


# --------------------------------------------------------------------------- #
# assign_lanes — ordering / determinism
# --------------------------------------------------------------------------- #


class TestAssignLanesOrdering:
    def test_input_order_does_not_matter(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        c = _app(application_id=_hex_id(3), start="s0w10", end="s0w15")
        forwards = assign_lanes([a, b, c])
        reversed_ = assign_lanes([c, b, a])
        assert forwards == reversed_

    def test_placements_in_document_order(self) -> None:
        c = _app(application_id=_hex_id(3), start="s0w10", end="s0w15")
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        layout = assign_lanes([c, a, b])
        # Expect a, b, c in that order.
        assert [p.application_id for p in layout.placements] == [a.id, b.id, c.id]

    def test_tie_break_by_application_id(self) -> None:
        # Two applications at the same anchor — different ids —
        # must come out in id-sorted order in the placements tuple.
        a = _app(application_id=_hex_id(0xFFFF), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(0x0001), start="s0w0", end="s0w5")
        layout = assign_lanes([a, b])
        # b's id sorts first lexically.
        assert layout.placements[0].application_id == b.id
        assert layout.placements[1].application_id == a.id

    def test_same_input_produces_equal_layout(self) -> None:
        # Layouts should be value-equal for caching purposes.
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        l1 = assign_lanes([a, b])
        l2 = assign_lanes([a, b])
        assert l1 == l2
        assert hash(l1) == hash(l2)


# --------------------------------------------------------------------------- #
# assign_lanes — cross-source guard
# --------------------------------------------------------------------------- #


class TestAssignLanesCrossSourceGuard:
    def test_mixed_sources_raise(self) -> None:
        a = _app(application_id=_hex_id(1), source_id=_HEX_SOURCE_1)
        b = _app(application_id=_hex_id(2), source_id=_HEX_SOURCE_2)
        with pytest.raises(ValueError, match="single-source"):
            assign_lanes([a, b])

    def test_single_source_does_not_raise(self) -> None:
        a = _app(application_id=_hex_id(1), source_id=_HEX_SOURCE_2)
        b = _app(application_id=_hex_id(2), source_id=_HEX_SOURCE_2)
        layout = assign_lanes([a, b])
        assert layout.source_id == _HEX_SOURCE_2


# --------------------------------------------------------------------------- #
# Stack depth nuances
# --------------------------------------------------------------------------- #


class TestStackDepth:
    def test_solo_app_zero_depth(self) -> None:
        a = _app(application_id=_hex_id(1))
        layout = assign_lanes([a])
        assert layout.placements[0].stack_depth == 0

    def test_pairwise_overlap_depth_one_each(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        layout = assign_lanes([a, b])
        assert all(p.stack_depth == 1 for p in layout.placements)

    def test_chain_overlap_asymmetric_depths(self) -> None:
        # A overlaps B; B overlaps C; A and C don't touch directly.
        # B sees both A and C (depth 2); A and C each see only B (depth 1).
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w3")
        b = _app(application_id=_hex_id(2), start="s0w2", end="s0w7")
        c = _app(application_id=_hex_id(3), start="s0w6", end="s0w10")
        layout = assign_lanes([a, b, c])
        depths = {p.application_id: p.stack_depth for p in layout.placements}
        assert depths[a.id] == 1
        assert depths[b.id] == 2
        assert depths[c.id] == 1
        assert layout.max_stack_depth == 2


# --------------------------------------------------------------------------- #
# Sub-word offsets in the layout
# --------------------------------------------------------------------------- #


class TestSubWordOffsets:
    def test_subword_overlap_uses_two_lanes(self) -> None:
        # Both anchored to s0w5 but overlapping ranges of characters.
        a = _app(
            application_id=_hex_id(1),
            start="s0w5",
            end="s0w5",
            start_offset=0,
            end_offset=4,
        )
        b = _app(
            application_id=_hex_id(2),
            start="s0w5",
            end="s0w5",
            start_offset=2,
            end_offset=8,
        )
        layout = assign_lanes([a, b])
        # They overlap on chars 2..4, so two lanes.
        assert layout.lane_count == 2

    def test_subword_disjoint_shares_lane(self) -> None:
        a = _app(
            application_id=_hex_id(1),
            start="s0w5",
            end="s0w5",
            start_offset=0,
            end_offset=3,
        )
        b = _app(
            application_id=_hex_id(2),
            start="s0w5",
            end="s0w5",
            start_offset=4,
            end_offset=7,
        )
        layout = assign_lanes([a, b])
        # Char 3 < char 4: strictly disjoint, single lane.
        assert layout.lane_count == 1


# --------------------------------------------------------------------------- #
# assign_lanes_per_source
# --------------------------------------------------------------------------- #


class TestAssignLanesPerSource:
    def test_empty_input(self) -> None:
        assert assign_lanes_per_source([]) == {}

    def test_buckets_by_source(self) -> None:
        a = _app(application_id=_hex_id(1), source_id=_HEX_SOURCE_1)
        b = _app(application_id=_hex_id(2), source_id=_HEX_SOURCE_2)
        out = assign_lanes_per_source([a, b])
        assert set(out.keys()) == {_HEX_SOURCE_1, _HEX_SOURCE_2}
        assert out[_HEX_SOURCE_1].lane_count == 1
        assert out[_HEX_SOURCE_2].lane_count == 1

    def test_independent_lane_numbering(self) -> None:
        # Lane 0 in source A is unrelated to lane 0 in source B.
        a1 = _app(
            application_id=_hex_id(1),
            source_id=_HEX_SOURCE_1,
            start="s0w0",
            end="s0w3",
        )
        a2 = _app(
            application_id=_hex_id(2),
            source_id=_HEX_SOURCE_1,
            start="s0w1",
            end="s0w5",
        )
        b1 = _app(
            application_id=_hex_id(3),
            source_id=_HEX_SOURCE_2,
            start="s0w0",
            end="s0w3",
        )
        out = assign_lanes_per_source([a1, a2, b1])
        # Source 1: two overlapping → 2 lanes.
        assert out[_HEX_SOURCE_1].lane_count == 2
        # Source 2: solo → 1 lane.
        assert out[_HEX_SOURCE_2].lane_count == 1


# --------------------------------------------------------------------------- #
# applications_at_word
# --------------------------------------------------------------------------- #


class TestApplicationsAtWord:
    def test_word_inside_span_matches(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        out = applications_at_word([a], _HEX_SOURCE_1, "s0w3")
        assert [x.id for x in out] == [a.id]

    def test_word_at_start_boundary_matches(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        out = applications_at_word([a], _HEX_SOURCE_1, "s0w0")
        assert [x.id for x in out] == [a.id]

    def test_word_at_end_boundary_matches(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        out = applications_at_word([a], _HEX_SOURCE_1, "s0w5")
        assert [x.id for x in out] == [a.id]

    def test_word_outside_span_does_not_match(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        assert applications_at_word([a], _HEX_SOURCE_1, "s0w6") == []
        assert applications_at_word([a], _HEX_SOURCE_1, "s1w0") == []

    def test_filters_out_other_sources(self) -> None:
        a = _app(application_id=_hex_id(1), source_id=_HEX_SOURCE_2)
        assert applications_at_word([a], _HEX_SOURCE_1, "s0w0") == []

    def test_returns_in_document_order(self) -> None:
        # Same anchor — tie broken by application id (lexical).
        a = _app(application_id=_hex_id(2), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        out = applications_at_word([a, b], _HEX_SOURCE_1, "s0w3")
        # b's id (000...001) < a's id (000...002), so b first.
        assert [x.id for x in out] == [b.id, a.id]

    def test_invalid_word_id_raises(self) -> None:
        a = _app(application_id=_hex_id(1))
        with pytest.raises(ProjectValidationError):
            applications_at_word([a], _HEX_SOURCE_1, "not-a-word-id")


# --------------------------------------------------------------------------- #
# lane_envelope
# --------------------------------------------------------------------------- #


class TestLaneEnvelope:
    def test_empty_layout(self) -> None:
        assert lane_envelope(assign_lanes([])) == []

    def test_each_lane_counted(self) -> None:
        # 3 disjoint apps → all in lane 0 → envelope [3].
        apps = [
            _app(application_id=_hex_id(i + 1), start=f"s0w{i*4}", end=f"s0w{i*4+2}")
            for i in range(3)
        ]
        layout = assign_lanes(apps)
        assert layout.lane_count == 1
        assert lane_envelope(layout) == [3]

    def test_two_lane_layout(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        c = _app(application_id=_hex_id(3), start="s0w10", end="s0w12")
        # a→0, b→1, c→0 (a finished). Envelope: [2, 1].
        layout = assign_lanes([a, b, c])
        assert lane_envelope(layout) == [2, 1]


# --------------------------------------------------------------------------- #
# GutterLayout.lane_for
# --------------------------------------------------------------------------- #


class TestLaneFor:
    def test_returns_none_when_unknown(self) -> None:
        a = _app(application_id=_hex_id(1))
        layout = assign_lanes([a])
        assert layout.lane_for("doesnotexist1") is None

    def test_returns_lane_when_found(self) -> None:
        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        layout = assign_lanes([a, b])
        assert layout.lane_for(a.id) == 0
        assert layout.lane_for(b.id) == 1

    def test_empty_layout(self) -> None:
        layout = assign_lanes([])
        assert layout.lane_for("anything0001") is None


# --------------------------------------------------------------------------- #
# serialise_layout
# --------------------------------------------------------------------------- #


class TestSerialiseLayout:
    def test_empty_layout(self) -> None:
        out = serialise_layout(assign_lanes([]))
        assert out == {
            "source_id": "",
            "lane_count": 0,
            "max_stack_depth": 0,
            "placements": [],
        }

    def test_layout_round_trips_through_json(self) -> None:
        import json

        a = _app(application_id=_hex_id(1), start="s0w0", end="s0w5")
        b = _app(application_id=_hex_id(2), start="s0w3", end="s0w8")
        layout = assign_lanes([a, b])
        text = json.dumps(serialise_layout(layout))
        loaded = json.loads(text)
        assert loaded["source_id"] == _HEX_SOURCE_1
        assert loaded["lane_count"] == 2
        assert loaded["max_stack_depth"] == 1
        assert {p["application_id"] for p in loaded["placements"]} == {a.id, b.id}

    def test_serialised_keys_stable(self) -> None:
        # Tighten on the keys so an accidental rename gets caught.
        a = _app(application_id=_hex_id(1))
        out = serialise_layout(assign_lanes([a]))
        assert set(out.keys()) == {
            "source_id",
            "lane_count",
            "max_stack_depth",
            "placements",
        }
        assert set(out["placements"][0].keys()) == {
            "application_id",
            "lane",
            "stack_depth",
        }


# --------------------------------------------------------------------------- #
# End-to-end: a small but realistic gutter
# --------------------------------------------------------------------------- #


class TestRealisticGutter:
    def test_mixed_codes_and_overlaps(self) -> None:
        # Three codes, six applications. Layout should:
        #   - place independent spans on lane 0;
        #   - stack overlapping spans to higher lanes;
        #   - reuse lower lanes once their previous occupants end.
        apps = [
            # Long topic code spanning the whole excerpt.
            _app(
                application_id=_hex_id(1),
                code_id=_HEX_CODE_A,
                start="s0w0",
                end="s2w10",
            ),
            # An emotion code on the first sentence.
            _app(
                application_id=_hex_id(2),
                code_id=_HEX_CODE_B,
                start="s0w2",
                end="s0w9",
            ),
            # An in-vivo phrase on a few words inside the emotion.
            _app(
                application_id=_hex_id(3),
                code_id=_HEX_CODE_C,
                start="s0w4",
                end="s0w6",
            ),
            # Second emotion later in the excerpt.
            _app(
                application_id=_hex_id(4),
                code_id=_HEX_CODE_B,
                start="s1w2",
                end="s1w7",
            ),
            # Reflexive note nested inside second emotion.
            _app(
                application_id=_hex_id(5),
                code_id=_HEX_CODE_C,
                start="s1w3",
                end="s1w5",
            ),
            # Disjoint span in the next sentence.
            _app(
                application_id=_hex_id(6),
                code_id=_HEX_CODE_B,
                start="s2w5",
                end="s2w9",
            ),
        ]
        layout = assign_lanes(apps)

        # Three lanes: the long topic occupies lane 0 the whole way;
        # the emotion codes stack on lane 1; the nested in-vivo /
        # reflexive notes stack on lane 2.
        assert layout.lane_count == 3
        lane_by_id = {p.application_id: p.lane for p in layout.placements}
        assert lane_by_id[apps[0].id] == 0  # long topic — lane 0
        assert lane_by_id[apps[1].id] == 1  # first emotion
        assert lane_by_id[apps[2].id] == 2  # nested in-vivo
        assert lane_by_id[apps[3].id] == 1  # second emotion
        assert lane_by_id[apps[4].id] == 2  # nested reflexive
        assert lane_by_id[apps[5].id] == 1  # disjoint span — re-uses lane 1

        # The long topic overlaps all five other applications, so its
        # stack depth is 5 — and that's the maximum across the layout.
        depths = {p.application_id: p.stack_depth for p in layout.placements}
        assert depths[apps[0].id] == 5
        assert layout.max_stack_depth == 5

        # Lane envelope: lane 0 holds 1 (topic); lanes 1 and 2 hold
        # 3 and 2 respectively.
        assert lane_envelope(layout) == [1, 3, 2]
