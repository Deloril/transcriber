"""Tests for scribe.application_reanchor (F4.5).

F4.1 anchored each Application to a (segment, word) word id pair. F4.5
gives us the *re-anchoring strategy* that runs when a transcript is
edited: same word ids → unchanged; new word ids found by content match
→ reanchored; not found → orphaned. These tests cover:

* normalize_word (the equivalence relation used for matching)
* collect_word_texts (transcript helper)
* anchored_words (extract a span's word texts)
* find_text_run (locate a word run, near-hint preference)
* reanchor_application + reanchor_applications (the three outcomes)
* apply_reanchor_outcome (mutating an application from an outcome)
* OrphanEntry + the orphan-queue persistence helpers
* record_orphans_from_plan (the save-orphans-from-plan convenience)

Pure Python; no FastAPI. Filesystem tests use ``tmp_path`` and the
existing project save helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.applications import (
    Application,
    make_word_id,
)
from scribe.application_reanchor import (
    OrphanEntry,
    REANCHOR_STATUS_ORPHANED,
    REANCHOR_STATUS_REANCHORED,
    REANCHOR_STATUS_UNCHANGED,
    REANCHOR_STATUSES,
    ReanchorOutcome,
    ReanchorPlan,
    anchored_words,
    append_orphan_entries,
    apply_reanchor_outcome,
    collect_word_texts,
    find_text_run,
    load_orphan_queue,
    make_orphan_entry,
    normalize_word,
    orphan_queue_path,
    reanchor_application,
    reanchor_applications,
    record_orphans_from_plan,
    remove_from_orphan_queue,
    save_orphan_queue,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12


def _seg(words, speaker="S1"):
    """Build a Scribe-shape segment from a list of word strings."""
    return {"speaker": speaker, "words": [{"text": w} for w in words]}


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_app(
    *,
    project_id: str = _HEX_PROJECT,
    application_id: str | None = None,
    anchor_start_word_id: str = "s0w0",
    anchor_end_word_id: str = "s0w2",
    start_char_offset: int | None = None,
    end_char_offset: int | None = None,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=_HEX_CODE,
        source_id=_HEX_SOURCE,
        coder_id=_HEX_CODER,
        anchor_start_word_id=anchor_start_word_id,
        anchor_end_word_id=anchor_end_word_id,
        definition_version_id_at_apply=_HEX_VERSION,
        start_char_offset=start_char_offset,
        end_char_offset=end_char_offset,
        application_id=application_id,
    )


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


class TestStatuses:
    def test_three_distinct_values(self) -> None:
        assert len(set(REANCHOR_STATUSES)) == 3
        assert REANCHOR_STATUS_UNCHANGED in REANCHOR_STATUSES
        assert REANCHOR_STATUS_REANCHORED in REANCHOR_STATUSES
        assert REANCHOR_STATUS_ORPHANED in REANCHOR_STATUSES


# --------------------------------------------------------------------------- #
# normalize_word
# --------------------------------------------------------------------------- #


class TestNormalizeWord:
    def test_lowercases(self) -> None:
        assert normalize_word("Hello") == "hello"
        assert normalize_word("WORLD") == "world"

    def test_strips_trailing_punctuation(self) -> None:
        assert normalize_word("hello,") == "hello"
        assert normalize_word("hello.") == "hello"
        assert normalize_word("hello!") == "hello"
        assert normalize_word("hello?") == "hello"

    def test_strips_leading_punctuation(self) -> None:
        assert normalize_word('"hello') == "hello"
        assert normalize_word("(hello") == "hello"

    def test_pure_punctuation_is_empty(self) -> None:
        assert normalize_word("--") == ""
        assert normalize_word("...") == ""
        assert normalize_word(",") == ""

    def test_empty_input(self) -> None:
        assert normalize_word("") == ""

    def test_non_string_safe(self) -> None:
        assert normalize_word(None) == ""  # type: ignore[arg-type]
        assert normalize_word(42) == ""  # type: ignore[arg-type]

    def test_unicode_letters_preserved(self) -> None:
        # Unicode letters should survive normalisation.
        assert normalize_word("Naïve") == "naïve"
        assert normalize_word("café.") == "café"

    def test_idempotent_on_normalised(self) -> None:
        for raw in ["hello,", "World!", "test"]:
            once = normalize_word(raw)
            twice = normalize_word(once)
            assert once == twice


# --------------------------------------------------------------------------- #
# collect_word_texts
# --------------------------------------------------------------------------- #


class TestCollectWordTexts:
    def test_simple(self) -> None:
        segs = [_seg(["hello", "world"]), _seg(["foo"])]
        assert collect_word_texts(segs) == [["hello", "world"], ["foo"]]

    def test_missing_words_key(self) -> None:
        segs = [{"speaker": "S1"}, _seg(["foo"])]
        assert collect_word_texts(segs) == [[], ["foo"]]

    def test_non_dict_segment(self) -> None:
        segs = [None, _seg(["foo"])]
        assert collect_word_texts(segs) == [[], ["foo"]]

    def test_word_without_text(self) -> None:
        segs = [{"speaker": "S1", "words": [{"text": "a"}, {}]}]
        assert collect_word_texts(segs) == [["a", ""]]

    def test_empty_input(self) -> None:
        assert collect_word_texts([]) == []


# --------------------------------------------------------------------------- #
# anchored_words
# --------------------------------------------------------------------------- #


class TestAnchoredWords:
    def test_within_segment(self) -> None:
        segs = [_seg(["a", "b", "c", "d"])]
        a = _make_app(
            anchor_start_word_id="s0w1",
            anchor_end_word_id="s0w2",
        )
        assert anchored_words(a, segs) == ["b", "c"]

    def test_full_segment(self) -> None:
        segs = [_seg(["a", "b", "c"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
        )
        assert anchored_words(a, segs) == ["a", "b", "c"]

    def test_cross_segment(self) -> None:
        segs = [_seg(["a", "b"]), _seg(["c", "d"])]
        a = _make_app(
            anchor_start_word_id="s0w1",
            anchor_end_word_id="s1w0",
        )
        assert anchored_words(a, segs) == ["b", "c"]

    def test_returns_none_segment_out_of_range(self) -> None:
        segs = [_seg(["a", "b"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s5w0",
        )
        assert anchored_words(a, segs) is None

    def test_returns_none_word_out_of_range(self) -> None:
        segs = [_seg(["a", "b"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w9",
        )
        assert anchored_words(a, segs) is None

    def test_returns_none_for_empty_segment(self) -> None:
        segs = [{"speaker": "S1", "words": []}]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
        )
        assert anchored_words(a, segs) is None


# --------------------------------------------------------------------------- #
# find_text_run
# --------------------------------------------------------------------------- #


class TestFindTextRun:
    def test_basic_match(self) -> None:
        segs = [_seg(["hello", "world", "today"])]
        assert find_text_run(["hello", "world"], segs) == ("s0w0", "s0w1")

    def test_match_with_punctuation_in_transcript(self) -> None:
        segs = [_seg(["Hello,", "world!"])]
        assert find_text_run(["hello", "world"], segs) == ("s0w0", "s0w1")

    def test_match_with_punctuation_in_target(self) -> None:
        segs = [_seg(["hello", "world"])]
        assert find_text_run(["Hello,", "world!"], segs) == ("s0w0", "s0w1")

    def test_match_case_insensitive(self) -> None:
        segs = [_seg(["HELLO", "WORLD"])]
        assert find_text_run(["hello", "world"], segs) == ("s0w0", "s0w1")

    def test_match_across_segments(self) -> None:
        segs = [_seg(["a", "b"]), _seg(["c", "d"])]
        assert find_text_run(["b", "c"], segs) == ("s0w1", "s1w0")

    def test_no_match(self) -> None:
        segs = [_seg(["hello", "world"])]
        assert find_text_run(["foo", "bar"], segs) is None

    def test_empty_target_returns_none(self) -> None:
        segs = [_seg(["a", "b"])]
        assert find_text_run([], segs) is None

    def test_pure_punctuation_target_returns_none(self) -> None:
        segs = [_seg(["a", "b"])]
        assert find_text_run(["...", ",", "--"], segs) is None

    def test_target_longer_than_transcript(self) -> None:
        segs = [_seg(["a"])]
        assert find_text_run(["a", "b", "c"], segs) is None

    def test_punctuation_token_in_transcript_skipped(self) -> None:
        # A punctuation-only token in the transcript should not break a
        # contiguous content match.
        segs = [{"speaker": "S1", "words": [
            {"text": "hello"},
            {"text": "--"},
            {"text": "world"},
        ]}]
        # Match "hello world" — the "--" token is skipped.
        result = find_text_run(["hello", "world"], segs)
        assert result == ("s0w0", "s0w2")

    def test_near_hint_picks_closer_match(self) -> None:
        # Two occurrences of "the cat"; near-hint should pick the
        # second one.
        segs = [
            _seg(["the", "cat"]),
            _seg(["something", "else"]),
            _seg(["the", "cat"]),
        ]
        # Without near, picks the first.
        assert find_text_run(["the", "cat"], segs) == ("s0w0", "s0w1")
        # With near pointing at segment 2, picks the second.
        assert find_text_run(["the", "cat"], segs, near=(2, 0)) == (
            "s2w0",
            "s2w1",
        )

    def test_near_hint_word_distance_breaks_segment_tie(self) -> None:
        segs = [_seg(["the", "cat", "the", "cat", "the", "cat"])]
        # Two matches at w0/w1 and w2/w3 and w4/w5. Near (0, 4) is
        # closest to w4.
        assert find_text_run(["the", "cat"], segs, near=(0, 4)) == (
            "s0w4",
            "s0w5",
        )


# --------------------------------------------------------------------------- #
# ReanchorOutcome
# --------------------------------------------------------------------------- #


class TestReanchorOutcome:
    def test_as_patch_for_unchanged(self) -> None:
        o = ReanchorOutcome(
            application_id="x" * 12,
            status=REANCHOR_STATUS_UNCHANGED,
            new_anchor_start_word_id="s0w0",
            new_anchor_end_word_id="s0w1",
            new_start_char_offset=2,
            new_end_char_offset=5,
            original_anchored_text=("hi",),
            reason="noop",
        )
        patch = o.as_patch()
        assert patch == {
            "anchor_start_word_id": "s0w0",
            "anchor_end_word_id": "s0w1",
            "start_char_offset": 2,
            "end_char_offset": 5,
        }

    def test_as_patch_for_reanchored(self) -> None:
        o = ReanchorOutcome(
            application_id="x" * 12,
            status=REANCHOR_STATUS_REANCHORED,
            new_anchor_start_word_id="s1w3",
            new_anchor_end_word_id="s1w5",
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hello", "world"),
            reason="match",
        )
        patch = o.as_patch()
        assert patch["anchor_start_word_id"] == "s1w3"
        assert patch["start_char_offset"] is None

    def test_as_patch_orphaned_raises(self) -> None:
        o = ReanchorOutcome(
            application_id="x" * 12,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hi",),
            reason="lost",
        )
        with pytest.raises(ProjectValidationError):
            o.as_patch()


# --------------------------------------------------------------------------- #
# ReanchorPlan
# --------------------------------------------------------------------------- #


class TestReanchorPlan:
    def _outcome(self, app_id: str, status: str) -> ReanchorOutcome:
        return ReanchorOutcome(
            application_id=app_id,
            status=status,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=(),
            reason="",
        )

    def test_filters(self) -> None:
        plan = ReanchorPlan(
            outcomes=[
                self._outcome("a" * 12, REANCHOR_STATUS_UNCHANGED),
                self._outcome("b" * 12, REANCHOR_STATUS_REANCHORED),
                self._outcome("c" * 12, REANCHOR_STATUS_ORPHANED),
                self._outcome("d" * 12, REANCHOR_STATUS_UNCHANGED),
            ]
        )
        assert [o.application_id for o in plan.unchanged] == ["a" * 12, "d" * 12]
        assert [o.application_id for o in plan.reanchored] == ["b" * 12]
        assert [o.application_id for o in plan.orphaned] == ["c" * 12]

    def test_for_application_found(self) -> None:
        plan = ReanchorPlan(
            outcomes=[self._outcome("a" * 12, REANCHOR_STATUS_UNCHANGED)]
        )
        assert plan.for_application("a" * 12).application_id == "a" * 12

    def test_for_application_missing(self) -> None:
        plan = ReanchorPlan(outcomes=[])
        assert plan.for_application("a" * 12) is None


# --------------------------------------------------------------------------- #
# reanchor_application
# --------------------------------------------------------------------------- #


class TestReanchorApplication:
    def test_unchanged_when_text_identical(self) -> None:
        old = [_seg(["hello", "world", "today"])]
        new = [_seg(["hello", "world", "today"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_UNCHANGED
        assert outcome.new_anchor_start_word_id == "s0w0"
        assert outcome.new_anchor_end_word_id == "s0w1"
        assert outcome.original_anchored_text == ("hello", "world")

    def test_unchanged_when_only_punctuation_changes(self) -> None:
        # Same words, just punctuation tweaks. Considered unchanged
        # under the normalised equivalence.
        old = [_seg(["Hello,", "world."])]
        new = [_seg(["Hello", "world!"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_UNCHANGED

    def test_unchanged_preserves_offsets(self) -> None:
        old = [_seg(["hello", "world"])]
        new = [_seg(["hello", "world"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            start_char_offset=1,
            end_char_offset=4,
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_UNCHANGED
        assert outcome.new_start_char_offset == 1
        assert outcome.new_end_char_offset == 4

    def test_reanchored_after_word_inserted_before(self) -> None:
        # Insert "Um," at the start; the original "hello world" shifts.
        old = [_seg(["hello", "world", "today"])]
        new = [_seg(["Um", "hello", "world", "today"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_REANCHORED
        assert outcome.new_anchor_start_word_id == "s0w1"
        assert outcome.new_anchor_end_word_id == "s0w2"
        # Sub-word offsets dropped on reanchor.
        assert outcome.new_start_char_offset is None
        assert outcome.new_end_char_offset is None

    def test_reanchored_drops_offsets_even_when_present(self) -> None:
        old = [_seg(["hello", "world"])]
        new = [_seg(["um", "hello", "world"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            start_char_offset=1,
            end_char_offset=4,
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_REANCHORED
        assert outcome.new_start_char_offset is None
        assert outcome.new_end_char_offset is None

    def test_orphaned_when_text_not_found(self) -> None:
        old = [_seg(["hello", "world"])]
        new = [_seg(["goodbye", "moon"])]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_ORPHANED
        assert outcome.new_anchor_start_word_id is None
        assert outcome.original_anchored_text == ("hello", "world")

    def test_orphaned_when_old_anchor_out_of_range(self) -> None:
        # The old transcript itself doesn't have those words; treat as
        # orphaned (we have no reference text to search for).
        old = [_seg(["hi"])]
        new = [_seg(["hi"])]
        a = _make_app(
            anchor_start_word_id="s5w0",
            anchor_end_word_id="s5w0",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_ORPHANED
        assert outcome.original_anchored_text == ()

    def test_reanchor_picks_match_near_original(self) -> None:
        # Two copies of the original text; near-hint at the original
        # location.
        old = [_seg(["the", "cat", "ran"])]
        new = [
            _seg(["the", "cat", "ran"]),
            _seg(["the", "cat", "appears"]),
        ]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        outcome = reanchor_application(a, old, new)
        # Closest to the original (0, 0) is the segment-0 match.
        assert outcome.new_anchor_start_word_id == "s0w0"

    def test_reanchor_picks_distant_match_when_only_one(self) -> None:
        # Original was at (0, 0). New transcript has the text only at
        # segment 2.
        old = [_seg(["the", "cat", "ran"])]
        new = [
            _seg(["a", "different", "thing"]),
            _seg(["another", "line"]),
            _seg(["the", "cat", "ran"]),
        ]
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
        )
        outcome = reanchor_application(a, old, new)
        assert outcome.status == REANCHOR_STATUS_REANCHORED
        assert outcome.new_anchor_start_word_id == "s2w0"
        assert outcome.new_anchor_end_word_id == "s2w2"


# --------------------------------------------------------------------------- #
# reanchor_applications
# --------------------------------------------------------------------------- #


class TestReanchorApplications:
    def test_batches_outcomes_in_input_order(self) -> None:
        old = [_seg(["a", "b", "c"])]
        new = [_seg(["a", "b", "c"])]
        ids = ["1" * 12, "2" * 12, "3" * 12]
        apps = [
            _make_app(
                application_id=i,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
            )
            for i in ids
        ]
        plan = reanchor_applications(apps, old, new)
        assert [o.application_id for o in plan.outcomes] == ids
        assert all(
            o.status == REANCHOR_STATUS_UNCHANGED for o in plan.outcomes
        )

    def test_mixed_outcomes(self) -> None:
        old = [_seg(["alpha", "beta", "gamma", "delta"])]
        new = [_seg(["alpha", "beta", "delta"])]  # gamma deleted
        # App 1 covers alpha/beta — unchanged.
        a1 = _make_app(
            application_id="1" * 12,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
        )
        # App 2 covers gamma — the word is gone; orphan.
        a2 = _make_app(
            application_id="2" * 12,
            anchor_start_word_id="s0w2",
            anchor_end_word_id="s0w2",
        )
        # App 3 covers delta — index moved from w3 to w2; reanchor.
        a3 = _make_app(
            application_id="3" * 12,
            anchor_start_word_id="s0w3",
            anchor_end_word_id="s0w3",
        )
        plan = reanchor_applications([a1, a2, a3], old, new)
        assert plan.for_application("1" * 12).status == REANCHOR_STATUS_UNCHANGED
        assert plan.for_application("2" * 12).status == REANCHOR_STATUS_ORPHANED
        re3 = plan.for_application("3" * 12)
        assert re3.status == REANCHOR_STATUS_REANCHORED
        assert re3.new_anchor_start_word_id == "s0w2"


# --------------------------------------------------------------------------- #
# apply_reanchor_outcome
# --------------------------------------------------------------------------- #


class TestApplyReanchorOutcome:
    def test_unchanged_returns_input(self) -> None:
        a = _make_app()
        outcome = ReanchorOutcome(
            application_id=a.id,
            status=REANCHOR_STATUS_UNCHANGED,
            new_anchor_start_word_id=a.anchor_start_word_id,
            new_anchor_end_word_id=a.anchor_end_word_id,
            new_start_char_offset=a.start_char_offset,
            new_end_char_offset=a.end_char_offset,
            original_anchored_text=("hello",),
            reason="",
        )
        result = apply_reanchor_outcome(a, outcome)
        assert result is a  # identity preserved
        assert result.modified_at == a.modified_at  # not bumped

    def test_reanchored_returns_new_application(self) -> None:
        a = _make_app(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            start_char_offset=2,
            end_char_offset=5,
        )
        outcome = ReanchorOutcome(
            application_id=a.id,
            status=REANCHOR_STATUS_REANCHORED,
            new_anchor_start_word_id="s0w3",
            new_anchor_end_word_id="s0w5",
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hello", "world"),
            reason="match",
        )
        result = apply_reanchor_outcome(a, outcome, now="2026-05-26T00:00:00Z")
        assert result is not a
        assert result.id == a.id
        assert result.anchor_start_word_id == "s0w3"
        assert result.anchor_end_word_id == "s0w5"
        # Sub-word offsets dropped.
        assert result.start_char_offset is None
        assert result.end_char_offset is None
        # Modified-at advanced.
        assert result.modified_at == "2026-05-26T00:00:00Z"
        # Original is untouched.
        assert a.anchor_start_word_id == "s0w0"
        assert a.start_char_offset == 2

    def test_orphaned_raises(self) -> None:
        a = _make_app()
        outcome = ReanchorOutcome(
            application_id=a.id,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hello",),
            reason="lost",
        )
        with pytest.raises(ProjectValidationError):
            apply_reanchor_outcome(a, outcome)

    def test_mismatched_id_raises(self) -> None:
        a = _make_app(application_id="a" * 12)
        outcome = ReanchorOutcome(
            application_id="b" * 12,  # mismatch
            status=REANCHOR_STATUS_UNCHANGED,
            new_anchor_start_word_id=a.anchor_start_word_id,
            new_anchor_end_word_id=a.anchor_end_word_id,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hi",),
            reason="",
        )
        with pytest.raises(ProjectValidationError):
            apply_reanchor_outcome(a, outcome)

    def test_unknown_status_raises(self) -> None:
        a = _make_app()
        outcome = ReanchorOutcome(
            application_id=a.id,
            status="bogus",
            new_anchor_start_word_id=a.anchor_start_word_id,
            new_anchor_end_word_id=a.anchor_end_word_id,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=(),
            reason="",
        )
        with pytest.raises(ProjectValidationError):
            apply_reanchor_outcome(a, outcome)


# --------------------------------------------------------------------------- #
# OrphanEntry
# --------------------------------------------------------------------------- #


class TestOrphanEntry:
    def test_round_trip(self) -> None:
        e = OrphanEntry(
            application_id="a" * 12,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w1",
            old_anchor_end_word_id="s0w3",
            original_anchored_text=["hello", "world"],
            reason="text gone",
            detected_at="2026-05-26T00:00:00Z",
        )
        e.validate()
        d = e.to_dict()
        e2 = OrphanEntry.from_dict(d)
        assert e2 == e

    def test_validate_rejects_bad_application_id(self) -> None:
        e = OrphanEntry(
            application_id="not-hex",
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
        )
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_validate_rejects_bad_word_id(self) -> None:
        e = OrphanEntry(
            application_id="a" * 12,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="bogus",
            old_anchor_end_word_id="s0w0",
        )
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_from_dict_missing_required(self) -> None:
        with pytest.raises(ProjectValidationError):
            OrphanEntry.from_dict({"application_id": "a" * 12})

    def test_from_dict_text_must_be_list(self) -> None:
        d = {
            "application_id": "a" * 12,
            "code_id": _HEX_CODE,
            "source_id": _HEX_SOURCE,
            "coder_id": _HEX_CODER,
            "old_anchor_start_word_id": "s0w0",
            "old_anchor_end_word_id": "s0w0",
            "original_anchored_text": "not-a-list",
        }
        with pytest.raises(ProjectValidationError):
            OrphanEntry.from_dict(d)

    def test_from_dict_non_object_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            OrphanEntry.from_dict("not-a-dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# make_orphan_entry
# --------------------------------------------------------------------------- #


class TestMakeOrphanEntry:
    def test_builds_from_application_and_orphan_outcome(self) -> None:
        a = _make_app(
            anchor_start_word_id="s0w1",
            anchor_end_word_id="s0w2",
        )
        outcome = ReanchorOutcome(
            application_id=a.id,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("hello", "world"),
            reason="text gone",
        )
        e = make_orphan_entry(a, outcome, now="2026-05-26T01:02:03Z")
        assert e.application_id == a.id
        assert e.code_id == a.code_id
        assert e.source_id == a.source_id
        assert e.coder_id == a.coder_id
        assert e.old_anchor_start_word_id == "s0w1"
        assert e.old_anchor_end_word_id == "s0w2"
        assert e.original_anchored_text == ["hello", "world"]
        assert e.detected_at == "2026-05-26T01:02:03Z"

    def test_rejects_non_orphan_outcome(self) -> None:
        a = _make_app()
        outcome = ReanchorOutcome(
            application_id=a.id,
            status=REANCHOR_STATUS_UNCHANGED,
            new_anchor_start_word_id=a.anchor_start_word_id,
            new_anchor_end_word_id=a.anchor_end_word_id,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=("a",),
            reason="",
        )
        with pytest.raises(ProjectValidationError):
            make_orphan_entry(a, outcome)

    def test_rejects_id_mismatch(self) -> None:
        a = _make_app(application_id="a" * 12)
        outcome = ReanchorOutcome(
            application_id="b" * 12,
            status=REANCHOR_STATUS_ORPHANED,
            new_anchor_start_word_id=None,
            new_anchor_end_word_id=None,
            new_start_char_offset=None,
            new_end_char_offset=None,
            original_anchored_text=(),
            reason="",
        )
        with pytest.raises(ProjectValidationError):
            make_orphan_entry(a, outcome)


# --------------------------------------------------------------------------- #
# Orphan queue persistence
# --------------------------------------------------------------------------- #


class TestOrphanQueuePersistence:
    def test_load_returns_empty_when_missing(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert load_orphan_queue(tmp_path, p.id) == []

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = OrphanEntry(
            application_id="a" * 12,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
            original_anchored_text=["hello"],
            reason="text gone",
            detected_at="2026-05-26T00:00:00Z",
        )
        save_orphan_queue(tmp_path, p.id, [e])
        loaded = load_orphan_queue(tmp_path, p.id)
        assert loaded == [e]

    def test_save_validates_entries(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        bad = OrphanEntry(
            application_id="not-hex",
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
        )
        with pytest.raises(ProjectValidationError):
            save_orphan_queue(tmp_path, p.id, [bad])

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            save_orphan_queue(tmp_path, "1" * 12, [])

    def test_save_rejects_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            save_orphan_queue(tmp_path, "not-hex", [])

    def test_load_rejects_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            load_orphan_queue(tmp_path, "not-hex")

    def test_load_skips_malformed_entries(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        # Hand-craft a queue file with one good and one bad entry.
        good = {
            "application_id": "a" * 12,
            "code_id": _HEX_CODE,
            "source_id": _HEX_SOURCE,
            "coder_id": _HEX_CODER,
            "old_anchor_start_word_id": "s0w0",
            "old_anchor_end_word_id": "s0w0",
            "original_anchored_text": ["hi"],
            "reason": "",
            "detected_at": "2026-05-26T00:00:00Z",
        }
        bad = {
            "application_id": "not-hex",
            "code_id": _HEX_CODE,
            "source_id": _HEX_SOURCE,
            "coder_id": _HEX_CODER,
            "old_anchor_start_word_id": "s0w0",
            "old_anchor_end_word_id": "s0w0",
        }
        path = orphan_queue_path(tmp_path, p.id)
        path.write_text(json.dumps({"entries": [good, bad]}))
        loaded = load_orphan_queue(tmp_path, p.id)
        assert len(loaded) == 1
        assert loaded[0].application_id == "a" * 12

    def test_load_rejects_non_object(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        path = orphan_queue_path(tmp_path, p.id)
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ProjectValidationError):
            load_orphan_queue(tmp_path, p.id)

    def test_load_rejects_bad_json(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        path = orphan_queue_path(tmp_path, p.id)
        path.write_text("{not valid json")
        with pytest.raises(ProjectValidationError):
            load_orphan_queue(tmp_path, p.id)


# --------------------------------------------------------------------------- #
# append_orphan_entries / remove_from_orphan_queue
# --------------------------------------------------------------------------- #


class TestAppendOrphanEntries:
    def _entry(self, app_id: str, *, detected_at: str = "2026-05-26T00:00:00Z") -> OrphanEntry:
        return OrphanEntry(
            application_id=app_id,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
            original_anchored_text=["x"],
            reason="r",
            detected_at=detected_at,
        )

    def test_append_creates_queue_when_absent(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = self._entry("a" * 12)
        merged = append_orphan_entries(tmp_path, p.id, [e])
        assert merged == [e]

    def test_append_dedupes_by_application_id(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e1 = self._entry("a" * 12, detected_at="2026-05-26T00:00:00Z")
        e2 = self._entry("a" * 12, detected_at="2026-05-26T01:00:00Z")
        append_orphan_entries(tmp_path, p.id, [e1])
        merged = append_orphan_entries(tmp_path, p.id, [e2])
        assert len(merged) == 1
        # Newer entry replaces older.
        assert merged[0].detected_at == "2026-05-26T01:00:00Z"

    def test_append_preserves_other_entries(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e1 = self._entry("a" * 12, detected_at="2026-05-26T00:00:00Z")
        e2 = self._entry("b" * 12, detected_at="2026-05-26T01:00:00Z")
        append_orphan_entries(tmp_path, p.id, [e1])
        merged = append_orphan_entries(tmp_path, p.id, [e2])
        assert len(merged) == 2
        ids = {e.application_id for e in merged}
        assert ids == {"a" * 12, "b" * 12}

    def test_append_sorts_by_detected_at(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e_late = self._entry("a" * 12, detected_at="2026-12-01T00:00:00Z")
        e_early = self._entry("b" * 12, detected_at="2026-01-01T00:00:00Z")
        merged = append_orphan_entries(tmp_path, p.id, [e_late, e_early])
        assert merged[0].application_id == "b" * 12
        assert merged[1].application_id == "a" * 12

    def test_append_validates_new_entries(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        bad = OrphanEntry(
            application_id="not-hex",
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
        )
        with pytest.raises(ProjectValidationError):
            append_orphan_entries(tmp_path, p.id, [bad])


class TestRemoveFromOrphanQueue:
    def test_remove_existing(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = OrphanEntry(
            application_id="a" * 12,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
            detected_at="2026-05-26T00:00:00Z",
        )
        append_orphan_entries(tmp_path, p.id, [e])
        assert remove_from_orphan_queue(tmp_path, p.id, "a" * 12) is True
        assert load_orphan_queue(tmp_path, p.id) == []

    def test_remove_missing_returns_false(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert remove_from_orphan_queue(tmp_path, p.id, "a" * 12) is False

    def test_remove_invalid_id_raises(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            remove_from_orphan_queue(tmp_path, p.id, "not-hex")


# --------------------------------------------------------------------------- #
# record_orphans_from_plan
# --------------------------------------------------------------------------- #


class TestRecordOrphansFromPlan:
    def test_records_all_orphans(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        a1 = _make_app(
            project_id=p.id,
            application_id="1" * 12,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
        )
        a2 = _make_app(
            project_id=p.id,
            application_id="2" * 12,
            anchor_start_word_id="s0w1",
            anchor_end_word_id="s0w1",
        )
        old = [_seg(["hello", "world"])]
        new = [_seg(["foo", "bar"])]  # both orphan
        plan = reanchor_applications([a1, a2], old, new)
        assert len(plan.orphaned) == 2
        merged = record_orphans_from_plan(
            tmp_path, p.id, plan, {a1.id: a1, a2.id: a2},
            now="2026-05-26T00:00:00Z",
        )
        assert {e.application_id for e in merged} == {a1.id, a2.id}
        assert all(e.detected_at == "2026-05-26T00:00:00Z" for e in merged)

    def test_skips_outcomes_without_application(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        # Plan with an outcome that has no matching application in the
        # mapping — should be silently skipped.
        plan = ReanchorPlan(outcomes=[
            ReanchorOutcome(
                application_id="z" * 12,
                status=REANCHOR_STATUS_ORPHANED,
                new_anchor_start_word_id=None,
                new_anchor_end_word_id=None,
                new_start_char_offset=None,
                new_end_char_offset=None,
                original_anchored_text=("hi",),
                reason="lost",
            ),
        ])
        merged = record_orphans_from_plan(tmp_path, p.id, plan, {})
        assert merged == []

    def test_no_orphans_returns_existing_queue(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        # Prior orphan in queue
        prior = OrphanEntry(
            application_id="a" * 12,
            code_id=_HEX_CODE,
            source_id=_HEX_SOURCE,
            coder_id=_HEX_CODER,
            old_anchor_start_word_id="s0w0",
            old_anchor_end_word_id="s0w0",
            detected_at="2026-05-26T00:00:00Z",
        )
        append_orphan_entries(tmp_path, p.id, [prior])
        # Plan with only unchanged outcomes
        plan = ReanchorPlan(outcomes=[])
        merged = record_orphans_from_plan(tmp_path, p.id, plan, {})
        assert len(merged) == 1
        assert merged[0].application_id == "a" * 12
