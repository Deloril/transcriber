"""Tests for scribe.application_spans (F4.2).

F4.1 gave us :class:`scribe.applications.Application`. F4.2 layers
the multi-application operations researchers do once a code carries
more than one application on the same source. These tests cover:

* anchor_key + sort_by_anchor (document ordering)
* applications_overlap / applications_disjoint / applications_adjacent
  (pairwise relations, including the cross-segment-with-counts case)
* applications_for_code_source / group_by_code_source (filtering)
* find_duplicate_anchors (exact-duplicate detection)
* overlap_clusters (transitive overlap groups)
* non_contiguous_components / count_non_contiguous_components
  (the F4.2 headline: "how many places in this source did I code this?")

All tests are pure Python; no persistence, no FastAPI.
"""

from __future__ import annotations

import math

import pytest

from scribe.applications import Application, ProjectValidationError
from scribe.application_spans import (
    anchor_key,
    applications_adjacent,
    applications_disjoint,
    applications_for_code_source,
    applications_overlap,
    count_non_contiguous_components,
    find_duplicate_anchors,
    group_by_code_source,
    non_contiguous_components,
    overlap_clusters,
    sort_by_anchor,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE_A = "a" * 12
_HEX_CODE_B = "b" * 12
_HEX_SOURCE_1 = "1" * 12
_HEX_SOURCE_2 = "2" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12


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
    """Build a valid Application for use in span-helper tests."""
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
# anchor_key
# --------------------------------------------------------------------------- #


class TestAnchorKey:
    def test_whole_word_anchor(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        # No offsets: start_offset normalised to 0, end_offset to +inf.
        key = anchor_key(a)
        assert key[0] == (0, 0, 0)
        assert key[1] == (0, 5, math.inf)

    def test_explicit_offsets(self) -> None:
        a = _app(start="s0w0", end="s0w0", start_offset=2, end_offset=7)
        assert anchor_key(a) == ((0, 0, 2), (0, 0, 7))

    def test_natural_segment_ordering(self) -> None:
        # s10 must sort *after* s2 — that's the whole point of the
        # tuple-of-ints key.
        a2 = _app(start="s2w0", end="s2w0", end_offset=1)
        a10 = _app(start="s10w0", end="s10w0", end_offset=1)
        assert anchor_key(a2) < anchor_key(a10)

    def test_end_of_word_sorts_after_explicit_offset(self) -> None:
        # Two apps end on the same word; the one with end=None ("to end
        # of word") must sort after the one with an explicit offset.
        a_explicit = _app(end="s0w5", end_offset=3)
        a_whole = _app(end="s0w5", end_offset=None)
        assert anchor_key(a_explicit) < anchor_key(a_whole)


# --------------------------------------------------------------------------- #
# sort_by_anchor
# --------------------------------------------------------------------------- #


class TestSortByAnchor:
    def test_returns_in_document_order(self) -> None:
        a1 = _app(start="s0w0", end="s0w5")
        a2 = _app(start="s1w0", end="s1w3")
        a3 = _app(start="s10w0", end="s10w2")
        # Pass in shuffled order; expect document order back.
        out = sort_by_anchor([a3, a1, a2])
        assert [a.id for a in out] == [a1.id, a2.id, a3.id]

    def test_stable_on_id_when_anchor_equal(self) -> None:
        # Two apps with identical anchors but different ids — stable
        # ordering across runs requires id-as-tiebreaker.
        a_high = _app(application_id="f" * 12)
        a_low = _app(application_id="0" * 12)
        out = sort_by_anchor([a_high, a_low])
        assert out[0].id == "0" * 12
        assert out[1].id == "f" * 12

    def test_empty_input(self) -> None:
        assert sort_by_anchor([]) == []

    def test_iterable_input(self) -> None:
        # Generators must work, not just lists.
        a1 = _app(start="s1w0", end="s1w0", end_offset=1)
        a2 = _app(start="s0w0", end="s0w0", end_offset=1)
        out = sort_by_anchor(x for x in [a1, a2])
        assert [a.anchor_start_word_id for a in out] == ["s0w0", "s1w0"]


# --------------------------------------------------------------------------- #
# applications_overlap
# --------------------------------------------------------------------------- #


class TestApplicationsOverlap:
    def test_different_sources_never_overlap(self) -> None:
        a = _app(source_id=_HEX_SOURCE_1, start="s0w0", end="s0w5")
        b = _app(source_id=_HEX_SOURCE_2, start="s0w0", end="s0w5")
        assert applications_overlap(a, b) is False

    def test_identical_anchors_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w0", end="s0w5")
        assert applications_overlap(a, b) is True

    def test_partial_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w3", end="s0w7")
        assert applications_overlap(a, b) is True

    def test_one_contains_the_other(self) -> None:
        outer = _app(start="s0w0", end="s0w10")
        inner = _app(start="s0w3", end="s0w7")
        assert applications_overlap(outer, inner) is True
        assert applications_overlap(inner, outer) is True

    def test_disjoint_within_segment(self) -> None:
        a = _app(start="s0w0", end="s0w2")
        b = _app(start="s0w5", end="s0w8")
        assert applications_overlap(a, b) is False

    def test_disjoint_across_segments(self) -> None:
        a = _app(start="s0w0", end="s0w99")
        b = _app(start="s2w0", end="s2w5")
        assert applications_overlap(a, b) is False

    def test_touching_at_word_boundary_is_not_overlap(self) -> None:
        # End-word of `a` is one less than start-word of `b` —
        # they're adjacent, not overlapping.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w6", end="s0w9")
        assert applications_overlap(a, b) is False

    def test_sub_word_offsets_distinguish_touching_from_overlap(self) -> None:
        # Both anchored on s0w0 but offsets carve disjoint halves.
        left = _app(start="s0w0", end="s0w0", start_offset=0, end_offset=5)
        right = _app(start="s0w0", end="s0w0", start_offset=5, end_offset=10)
        assert applications_overlap(left, right) is False
        # …but ranges that genuinely cross the offset boundary do overlap.
        crossing = _app(start="s0w0", end="s0w0", start_offset=4, end_offset=6)
        assert applications_overlap(left, crossing) is True
        assert applications_overlap(right, crossing) is True

    def test_whole_word_envelopes_sub_word_anchor(self) -> None:
        # A whole-word anchor on s0w0 should overlap any sub-word
        # anchor on s0w0 — that's the +inf end-offset semantics.
        whole = _app(start="s0w0", end="s0w0", end_offset=None)
        sub = _app(start="s0w0", end="s0w0", start_offset=2, end_offset=4)
        assert applications_overlap(whole, sub) is True

    def test_different_codes_can_overlap(self) -> None:
        # Overlap is a span/source notion, code-agnostic.
        a = _app(code_id=_HEX_CODE_A, start="s0w0", end="s0w5")
        b = _app(code_id=_HEX_CODE_B, start="s0w2", end="s0w7")
        assert applications_overlap(a, b) is True


# --------------------------------------------------------------------------- #
# applications_disjoint
# --------------------------------------------------------------------------- #


class TestApplicationsDisjoint:
    def test_disjoint_when_same_source_no_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w2")
        b = _app(start="s0w5", end="s0w8")
        assert applications_disjoint(a, b) is True

    def test_not_disjoint_when_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w3", end="s0w8")
        assert applications_disjoint(a, b) is False

    def test_different_source_is_not_disjoint(self) -> None:
        # Different sources are not comparable in this F4.2 sense —
        # disjoint() returns False, matching the docstring.
        a = _app(source_id=_HEX_SOURCE_1)
        b = _app(source_id=_HEX_SOURCE_2)
        assert applications_disjoint(a, b) is False


# --------------------------------------------------------------------------- #
# applications_adjacent
# --------------------------------------------------------------------------- #


class TestApplicationsAdjacent:
    def test_adjacent_within_segment_consecutive_words(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w6", end="s0w9")
        assert applications_adjacent(a, b) is True
        # Symmetric.
        assert applications_adjacent(b, a) is True

    def test_not_adjacent_when_gap_present(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w7", end="s0w9")
        assert applications_adjacent(a, b) is False

    def test_not_adjacent_when_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w5", end="s0w9")
        # They overlap on s0w5 → adjacency is false by definition.
        assert applications_adjacent(a, b) is False

    def test_not_adjacent_across_segment_without_counts(self) -> None:
        # End of segment 0 vs start of segment 1: cannot decide
        # adjacency without segment_word_counts.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        assert applications_adjacent(a, b) is False

    def test_adjacent_across_segment_with_counts(self) -> None:
        # Tell the helper segment 0 has 6 words: s0w5 is the last word.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        counts = {0: 6, 1: 4}
        assert applications_adjacent(a, b, segment_word_counts=counts) is True

    def test_not_adjacent_across_segment_when_end_is_not_last_word(self) -> None:
        # Segment 0 has 10 words but the application ends at w5 — so
        # there's a gap before s1w0.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        counts = {0: 10}
        assert applications_adjacent(a, b, segment_word_counts=counts) is False

    def test_adjacency_skipping_segments_is_never_true(self) -> None:
        # s0 end of segment to s2 start: not adjacent even if seg 0
        # ends at the right word, because seg 1 lies between.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s2w0", end="s2w3")
        counts = {0: 6}
        assert applications_adjacent(a, b, segment_word_counts=counts) is False

    def test_different_sources_are_not_adjacent(self) -> None:
        a = _app(source_id=_HEX_SOURCE_1, start="s0w0", end="s0w5")
        b = _app(source_id=_HEX_SOURCE_2, start="s0w6", end="s0w9")
        assert applications_adjacent(a, b) is False

    def test_sub_word_offset_at_touch_breaks_adjacency_first_end(self) -> None:
        # First app has a sub-word end_offset → not a clean boundary.
        a = _app(start="s0w0", end="s0w5", end_offset=3)
        b = _app(start="s0w6", end="s0w9")
        assert applications_adjacent(a, b) is False

    def test_sub_word_offset_at_touch_breaks_adjacency_second_start(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w6", end="s0w9", start_offset=2)
        assert applications_adjacent(a, b) is False

    def test_unrelated_segments_not_adjacent(self) -> None:
        # End of one application is in seg 0; start of the other is in
        # seg 5 — clearly not adjacent regardless of counts.
        a = _app(start="s0w0", end="s0w2")
        b = _app(start="s5w0", end="s5w2")
        assert applications_adjacent(a, b, segment_word_counts={0: 3}) is False

    def test_segment_word_counts_with_zero_or_missing(self) -> None:
        # Missing entry → conservatively not-adjacent.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        assert applications_adjacent(a, b, segment_word_counts={}) is False
        # Zero word count is nonsensical → also not-adjacent (won't crash).
        assert applications_adjacent(a, b, segment_word_counts={0: 0}) is False


# --------------------------------------------------------------------------- #
# applications_for_code_source
# --------------------------------------------------------------------------- #


class TestApplicationsForCodeSource:
    def test_filters_by_code_and_source(self) -> None:
        match = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_1)
        wrong_code = _app(code_id=_HEX_CODE_B, source_id=_HEX_SOURCE_1)
        wrong_source = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_2)
        out = applications_for_code_source(
            [match, wrong_code, wrong_source],
            _HEX_CODE_A,
            _HEX_SOURCE_1,
        )
        assert [a.id for a in out] == [match.id]

    def test_returns_in_document_order(self) -> None:
        first = _app(start="s0w0", end="s0w2")
        second = _app(start="s0w5", end="s0w7")
        third = _app(start="s3w0", end="s3w2")
        out = applications_for_code_source(
            [third, first, second], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert [a.id for a in out] == [first.id, second.id, third.id]

    def test_returns_empty_on_no_match(self) -> None:
        a = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_1)
        out = applications_for_code_source([a], _HEX_CODE_B, _HEX_SOURCE_1)
        assert out == []

    def test_empty_input(self) -> None:
        assert applications_for_code_source([], _HEX_CODE_A, _HEX_SOURCE_1) == []


# --------------------------------------------------------------------------- #
# group_by_code_source
# --------------------------------------------------------------------------- #


class TestGroupByCodeSource:
    def test_buckets_by_pair(self) -> None:
        a = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_1)
        b = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_2)
        c = _app(code_id=_HEX_CODE_B, source_id=_HEX_SOURCE_1)
        out = group_by_code_source([a, b, c])
        assert set(out.keys()) == {
            (_HEX_CODE_A, _HEX_SOURCE_1),
            (_HEX_CODE_A, _HEX_SOURCE_2),
            (_HEX_CODE_B, _HEX_SOURCE_1),
        }
        assert [x.id for x in out[(_HEX_CODE_A, _HEX_SOURCE_1)]] == [a.id]

    def test_each_bucket_sorted(self) -> None:
        first = _app(start="s0w0", end="s0w2")
        second = _app(start="s1w0", end="s1w2")
        third = _app(start="s2w0", end="s2w2")
        out = group_by_code_source([third, first, second])
        bucket = out[(_HEX_CODE_A, _HEX_SOURCE_1)]
        assert [x.id for x in bucket] == [first.id, second.id, third.id]

    def test_empty_input(self) -> None:
        assert group_by_code_source([]) == {}

    def test_pair_with_no_apps_does_not_appear(self) -> None:
        a = _app(code_id=_HEX_CODE_A)
        out = group_by_code_source([a])
        # The function shouldn't fabricate empty buckets.
        assert (_HEX_CODE_B, _HEX_SOURCE_1) not in out


# --------------------------------------------------------------------------- #
# find_duplicate_anchors
# --------------------------------------------------------------------------- #


class TestFindDuplicateAnchors:
    def test_returns_empty_when_all_unique(self) -> None:
        a = _app(start="s0w0", end="s0w2")
        b = _app(start="s0w5", end="s0w7")
        assert find_duplicate_anchors([a, b]) == []

    def test_finds_exact_duplicates_same_coder(self) -> None:
        a = _app(application_id="0" * 12)
        b = _app(application_id="f" * 12)  # identical anchor, different id
        groups = find_duplicate_anchors([a, b])
        assert len(groups) == 1
        assert {x.id for x in groups[0]} == {"0" * 12, "f" * 12}

    def test_finds_duplicates_across_coders(self) -> None:
        # Multi-coder case (F2.5): same span, different coders → still a
        # duplicate worth surfacing for reconciliation.
        a = _app(coder_id="c" * 12, application_id="0" * 12)
        b = _app(coder_id="e" * 12, application_id="1" * 12)
        groups = find_duplicate_anchors([a, b])
        assert len(groups) == 1

    def test_different_codes_not_duplicates(self) -> None:
        # Same span on the same source with two different codes is the
        # whole point of overlapping codes (F4.3) — not a duplicate.
        a = _app(code_id=_HEX_CODE_A)
        b = _app(code_id=_HEX_CODE_B)
        assert find_duplicate_anchors([a, b]) == []

    def test_different_sources_not_duplicates(self) -> None:
        a = _app(source_id=_HEX_SOURCE_1)
        b = _app(source_id=_HEX_SOURCE_2)
        assert find_duplicate_anchors([a, b]) == []

    def test_different_offsets_not_duplicates(self) -> None:
        a = _app(start="s0w0", end="s0w0", start_offset=0, end_offset=5)
        b = _app(start="s0w0", end="s0w0", start_offset=0, end_offset=6)
        assert find_duplicate_anchors([a, b]) == []

    def test_none_offset_not_collapsed_with_zero(self) -> None:
        # `None` start_offset and `0` start_offset are *not* the same
        # record-shape, even though they have the same semantics —
        # find_duplicate_anchors compares offsets verbatim. (Future:
        # canonicalise on save if this turns out to be noisy.)
        a = _app(start="s0w0", end="s0w5", start_offset=None)
        b = _app(start="s0w0", end="s0w5", start_offset=0)
        assert find_duplicate_anchors([a, b]) == []

    def test_groups_sorted_by_id(self) -> None:
        a_late = _app(application_id="f" * 12)
        a_early = _app(application_id="0" * 12)
        groups = find_duplicate_anchors([a_late, a_early])
        assert groups[0][0].id == "0" * 12
        assert groups[0][1].id == "f" * 12

    def test_three_way_duplicate(self) -> None:
        a = _app(application_id="1" * 12)
        b = _app(application_id="2" * 12)
        c = _app(application_id="3" * 12)
        groups = find_duplicate_anchors([a, b, c])
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_empty_input(self) -> None:
        assert find_duplicate_anchors([]) == []


# --------------------------------------------------------------------------- #
# overlap_clusters
# --------------------------------------------------------------------------- #


class TestOverlapClusters:
    def test_empty_when_all_disjoint(self) -> None:
        a = _app(start="s0w0", end="s0w2")
        b = _app(start="s0w5", end="s0w7")
        c = _app(start="s2w0", end="s2w2")
        assert overlap_clusters([a, b, c]) == []

    def test_pair_overlap(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w3", end="s0w8")
        clusters = overlap_clusters([a, b])
        assert len(clusters) == 1
        assert {x.id for x in clusters[0]} == {a.id, b.id}

    def test_transitive_overlap(self) -> None:
        # A overlaps B; B overlaps C; A and C don't overlap directly.
        # All three must end up in one cluster.
        a = _app(start="s0w0", end="s0w3")
        b = _app(start="s0w2", end="s0w7")
        c = _app(start="s0w6", end="s0w10")
        clusters = overlap_clusters([a, b, c])
        assert len(clusters) == 1
        assert {x.id for x in clusters[0]} == {a.id, b.id, c.id}

    def test_overlap_across_codes(self) -> None:
        # F4.3 will use this: a single span carrying many codes shows
        # up as one overlap cluster regardless of code id.
        a = _app(code_id=_HEX_CODE_A, start="s0w0", end="s0w5")
        b = _app(code_id=_HEX_CODE_B, start="s0w3", end="s0w8")
        clusters = overlap_clusters([a, b])
        assert len(clusters) == 1

    def test_no_cross_source_overlap(self) -> None:
        a = _app(source_id=_HEX_SOURCE_1, start="s0w0", end="s0w5")
        b = _app(source_id=_HEX_SOURCE_2, start="s0w0", end="s0w5")
        # Same span text but different source — no overlap, no cluster.
        assert overlap_clusters([a, b]) == []

    def test_cluster_apps_sorted_by_anchor(self) -> None:
        a_first = _app(start="s0w0", end="s0w5")
        b_second = _app(start="s0w3", end="s0w8")
        clusters = overlap_clusters([b_second, a_first])
        assert [x.id for x in clusters[0]] == [a_first.id, b_second.id]

    def test_multiple_clusters_sorted(self) -> None:
        # Two independent overlap clusters; outer order is by first
        # member's anchor.
        a1 = _app(start="s0w0", end="s0w3", application_id="a" * 12)
        a2 = _app(start="s0w2", end="s0w5", application_id="b" * 12)
        b1 = _app(start="s5w0", end="s5w3", application_id="c" * 12)
        b2 = _app(start="s5w2", end="s5w5", application_id="d" * 12)
        clusters = overlap_clusters([b1, a1, b2, a2])
        assert len(clusters) == 2
        first_ids = {x.id for x in clusters[0]}
        second_ids = {x.id for x in clusters[1]}
        assert first_ids == {a1.id, a2.id}
        assert second_ids == {b1.id, b2.id}

    def test_singleton_not_returned(self) -> None:
        a = _app(start="s0w0", end="s0w3")
        assert overlap_clusters([a]) == []

    def test_empty_input(self) -> None:
        assert overlap_clusters([]) == []


# --------------------------------------------------------------------------- #
# non_contiguous_components / count
# --------------------------------------------------------------------------- #


class TestNonContiguousComponents:
    def test_three_separate_applications_yield_three_components(self) -> None:
        # The headline F4.2 case: same code, three places in a source.
        a1 = _app(start="s0w0", end="s0w3")
        a2 = _app(start="s2w0", end="s2w3")
        a3 = _app(start="s5w0", end="s5w3")
        comps = non_contiguous_components(
            [a1, a2, a3], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert len(comps) == 3
        assert [len(c) for c in comps] == [1, 1, 1]
        assert [c[0].id for c in comps] == [a1.id, a2.id, a3.id]

    def test_overlapping_apps_collapse_to_one_component(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w3", end="s0w8")  # overlaps a
        c = _app(start="s5w0", end="s5w3")  # disjoint
        comps = non_contiguous_components(
            [a, b, c], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert len(comps) == 2
        # First component holds the overlapping pair.
        assert {x.id for x in comps[0]} == {a.id, b.id}
        assert [x.id for x in comps[1]] == [c.id]

    def test_within_segment_adjacent_collapse(self) -> None:
        # End-of-a is one less than start-of-b within the same segment
        # → adjacent → same component.
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w6", end="s0w9")
        comps = non_contiguous_components(
            [a, b], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert len(comps) == 1
        assert {x.id for x in comps[0]} == {a.id, b.id}

    def test_cross_segment_adjacency_requires_counts(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        # No counts: treated as non-contiguous.
        comps = non_contiguous_components(
            [a, b], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert len(comps) == 2
        # With counts saying segment 0 has 6 words: adjacent → one component.
        comps = non_contiguous_components(
            [a, b],
            _HEX_CODE_A,
            _HEX_SOURCE_1,
            segment_word_counts={0: 6},
        )
        assert len(comps) == 1

    def test_filters_by_code_and_source(self) -> None:
        # Different code or different source must not contribute.
        target = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_1)
        wrong_code = _app(code_id=_HEX_CODE_B, source_id=_HEX_SOURCE_1)
        wrong_source = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_2)
        comps = non_contiguous_components(
            [target, wrong_code, wrong_source], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert len(comps) == 1
        assert comps[0][0].id == target.id

    def test_no_apps_returns_empty(self) -> None:
        comps = non_contiguous_components([], _HEX_CODE_A, _HEX_SOURCE_1)
        assert comps == []

    def test_components_sorted_by_anchor(self) -> None:
        # Pass in reverse order; expect document order back.
        a_first = _app(start="s0w0", end="s0w2")
        a_second = _app(start="s2w0", end="s2w2")
        a_third = _app(start="s5w0", end="s5w2")
        comps = non_contiguous_components(
            [a_third, a_first, a_second], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert [c[0].id for c in comps] == [a_first.id, a_second.id, a_third.id]


class TestCountNonContiguousComponents:
    def test_count_matches_components_len(self) -> None:
        a1 = _app(start="s0w0", end="s0w2")
        a2 = _app(start="s5w0", end="s5w2")
        n = count_non_contiguous_components(
            [a1, a2], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert n == 2

    def test_count_zero_when_no_match(self) -> None:
        a = _app(code_id=_HEX_CODE_A, source_id=_HEX_SOURCE_1)
        n = count_non_contiguous_components([a], _HEX_CODE_B, _HEX_SOURCE_1)
        assert n == 0

    def test_count_one_when_overlap_collapses(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s0w3", end="s0w8")
        n = count_non_contiguous_components(
            [a, b], _HEX_CODE_A, _HEX_SOURCE_1
        )
        assert n == 1

    def test_count_with_segment_counts_collapses_cross_segment(self) -> None:
        a = _app(start="s0w0", end="s0w5")
        b = _app(start="s1w0", end="s1w3")
        n = count_non_contiguous_components(
            [a, b], _HEX_CODE_A, _HEX_SOURCE_1, segment_word_counts={0: 6}
        )
        assert n == 1

    def test_empty_input(self) -> None:
        assert count_non_contiguous_components([], _HEX_CODE_A, _HEX_SOURCE_1) == 0
