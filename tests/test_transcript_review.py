"""Tests for scribe.transcript_review (F8.6).

Covers:

  * ReviewItem dataclass: validate, round-trip, mutual exclusion of
    suggestion_id / error.
  * ReviewPass entity: validate, round-trip, status timestamp
    invariants, progress getters.
  * enumerate_review_items:
      - paragraph mode + sentence mode;
      - skip_already_coded honoured at paragraph + word granularity;
      - skip_already_coded=False produces a full sweep;
      - empty paragraphs/sentences dropped;
      - source-mismatched applications ignored.
  * start_review_pass: persists, items frozen.
  * process_next_review_item:
      - flips pending → running, stamps started_at;
      - persists CodeSuggestion + pass record;
      - records per-item error and continues;
      - flips to completed when nothing pending;
      - rejects calls on terminal passes.
  * run_review_pass: drives to completion + on_step + max_steps.
  * cancel_review_pass / mark_review_pass_failed: terminal-state guard.
  * Persistence: save / load / list / delete, with filters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.applications import Application
from scribe.codes import Code
from scribe.code_suggestions import (
    SUGGESTION_DECISION_PENDING,
    list_suggestions,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.transcript_review import (
    MAX_ERROR_MESSAGE_LEN,
    MAX_ITEMS,
    MAX_ITEM_ERROR_LEN,
    MAX_NOTES_LEN,
    MAX_TEXT_PREVIEW_LEN,
    REVIEW_GRANULARITIES,
    REVIEW_GRANULARITY_PARAGRAPH,
    REVIEW_GRANULARITY_SENTENCE,
    REVIEW_PASSES_DIRNAME,
    REVIEW_PASS_ID_RE,
    REVIEW_STATUS_CANCELLED,
    REVIEW_STATUS_COMPLETED,
    REVIEW_STATUS_FAILED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_RUNNING,
    REVIEW_STATUSES,
    REVIEW_TERMINAL_STATUSES,
    ReviewItem,
    ReviewPass,
    cancel_review_pass,
    delete_review_pass,
    enumerate_review_items,
    is_terminal_status,
    list_review_passes,
    load_review_pass,
    mark_review_pass_failed,
    new_review_pass_id,
    process_next_review_item,
    review_pass_state_path,
    review_passes_dir,
    run_review_pass,
    save_review_pass,
    start_review_pass,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "aaaaaaaaaaaa"
_HEX_SOURCE = "bbbbbbbbbbbb"
_HEX_SOURCE_2 = "cccccccccccc"
_HEX_CODER = "0123456789ab"
_HEX_VERSION = "fedcba987654"
_HEX_CODE_A = "1111aaaa1111"
_HEX_CODE_B = "2222bbbb2222"
_HEX_APP_A = "aa00aa00aa00"
_HEX_APP_B = "bb11bb11bb11"


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_app(
    *,
    project_id: str,
    code_id: str,
    application_id: str,
    source_id: str = _HEX_SOURCE,
    coder_id: str = _HEX_CODER,
    start: str = "s0w0",
    end: str = "s0w0",
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start,
        anchor_end_word_id=end,
        definition_version_id_at_apply=_HEX_VERSION,
        application_id=application_id,
    )


def _make_code(
    project_id: str,
    *,
    code_id: str | None = None,
    name: str = "code-name",
    definition: str = "",
    exemplars: list[str] | None = None,
    status: str = "active",
) -> Code:
    return Code.new(
        project_id=project_id,
        code_id=code_id,
        name=name,
        definition=definition,
        exemplars=list(exemplars or []),
        status=status,
    )


def _segments(turns: Sequence[tuple[str | None, str]]) -> list[dict[str, Any]]:
    """Build segments where each turn = (speaker, full text).

    Words are space-split; texts must be ASCII for the helper. The
    last word of every segment carries a sentence-final ``.`` so
    sentence boundaries are unambiguous in tests that don't care.
    """
    out: list[dict[str, Any]] = []
    for speaker, text in turns:
        words = [{"text": w} for w in text.split(" ") if w]
        out.append({"speaker": speaker, "words": words})
    return out


def _const_embed(vec: Sequence[float]):
    """Stub embed_fn that returns ``vec`` for every input."""
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [tuple(float(x) for x in vec)] * len(texts)
    return fn


def _const_generate(text: str = "[]"):
    def fn(prompt: str) -> str:
        return text
    return fn


# --------------------------------------------------------------------------- #
# new_review_pass_id / is_terminal_status
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_new_id_is_12_hex(self) -> None:
        rid = new_review_pass_id()
        assert REVIEW_PASS_ID_RE.match(rid)

    def test_new_id_unique(self) -> None:
        ids = {new_review_pass_id() for _ in range(50)}
        assert len(ids) == 50

    @pytest.mark.parametrize(
        "status",
        [REVIEW_STATUS_COMPLETED, REVIEW_STATUS_CANCELLED, REVIEW_STATUS_FAILED],
    )
    def test_is_terminal_true(self, status: str) -> None:
        assert is_terminal_status(status) is True

    @pytest.mark.parametrize(
        "status", [REVIEW_STATUS_PENDING, REVIEW_STATUS_RUNNING]
    )
    def test_is_terminal_false(self, status: str) -> None:
        assert is_terminal_status(status) is False


# --------------------------------------------------------------------------- #
# ReviewItem
# --------------------------------------------------------------------------- #


class TestReviewItem:
    def _make(self, **kw: Any) -> ReviewItem:
        defaults: dict[str, Any] = dict(
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
            paragraph_start_segment=0,
            paragraph_end_segment=0,
            text_preview="hi there",
        )
        defaults.update(kw)
        return ReviewItem(**defaults)

    def test_pending_by_default(self) -> None:
        it = self._make()
        assert it.is_pending is True
        assert it.is_processed is False

    def test_processed_with_suggestion_id(self) -> None:
        it = self._make(suggestion_id="0123456789ab")
        assert it.is_processed is True
        assert it.is_pending is False

    def test_processed_with_error(self) -> None:
        it = self._make(error="something broke")
        assert it.is_processed is True
        assert it.is_pending is False

    def test_validate_anchor_ordering(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(
                anchor_start_word_id="s2w0", anchor_end_word_id="s1w0"
            ).validate()

    def test_validate_negative_paragraph_start(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(paragraph_start_segment=-1).validate()

    def test_validate_paragraph_end_before_start(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(
                paragraph_start_segment=3, paragraph_end_segment=2
            ).validate()

    def test_validate_text_preview_length(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(text_preview="x" * (MAX_TEXT_PREVIEW_LEN + 1)).validate()

    def test_validate_suggestion_id_shape(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(suggestion_id="not-hex").validate()

    def test_validate_mutual_exclusion(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(
                suggestion_id="0123456789ab", error="oops"
            ).validate()

    def test_validate_error_length(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(error="x" * (MAX_ITEM_ERROR_LEN + 1)).validate()

    def test_round_trip(self) -> None:
        it = self._make(suggestion_id="0123456789ab")
        round_trip = ReviewItem.from_dict(it.to_dict())
        assert round_trip == it

    def test_round_trip_error(self) -> None:
        it = self._make(error="bad")
        round_trip = ReviewItem.from_dict(it.to_dict())
        assert round_trip == it

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            ReviewItem.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_invalid_int(self) -> None:
        with pytest.raises(ProjectValidationError):
            ReviewItem.from_dict(
                {
                    "anchor_start_word_id": "s0w0",
                    "anchor_end_word_id": "s0w0",
                    "paragraph_start_segment": "abc",
                    "paragraph_end_segment": 0,
                    "text_preview": "x",
                }
            )


# --------------------------------------------------------------------------- #
# enumerate_review_items
# --------------------------------------------------------------------------- #


class TestEnumerate:
    def test_paragraph_mode_basic(self) -> None:
        # Three turns; two distinct speakers — 3 paragraphs (because
        # the middle one's speaker differs, and the third's same as
        # second only if speakers match).
        segs = _segments(
            [
                ("ALICE", "hello there friend."),
                ("BOB", "general kenobi."),
                ("BOB", "you are a bold one."),
            ]
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE, segments=segs
        )
        # Bob's two consecutive segments fold into one paragraph.
        assert len(items) == 2
        assert items[0].paragraph_start_segment == 0
        assert items[0].paragraph_end_segment == 0
        assert items[0].text_preview == "hello there friend."
        assert items[1].paragraph_start_segment == 1
        assert items[1].paragraph_end_segment == 2
        assert "general kenobi" in items[1].text_preview

    def test_paragraph_mode_skips_coded(self) -> None:
        segs = _segments(
            [
                ("A", "first one."),
                ("B", "second one."),
            ]
        )
        # Application covers first paragraph entirely.
        app = _make_app(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE_A,
            application_id=_HEX_APP_A,
            start="s0w0",
            end="s0w1",
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            applications=[app],
        )
        # Only the second paragraph remains.
        assert [it.paragraph_start_segment for it in items] == [1]

    def test_paragraph_mode_skip_disabled(self) -> None:
        segs = _segments(
            [("A", "first one."), ("B", "second one.")]
        )
        app = _make_app(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE_A,
            application_id=_HEX_APP_A,
            start="s0w0",
            end="s0w1",
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            applications=[app],
            skip_already_coded=False,
        )
        assert len(items) == 2

    def test_paragraph_mode_ignores_other_source_apps(self) -> None:
        segs = _segments([("A", "first one.")])
        # Application is on a different source; should not affect.
        app = _make_app(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE_A,
            application_id=_HEX_APP_A,
            source_id=_HEX_SOURCE_2,
            start="s0w0",
            end="s0w1",
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            applications=[app],
        )
        assert len(items) == 1

    def test_paragraph_anchor_uses_first_and_last_word(self) -> None:
        segs = _segments([("A", "hello there friend.")])
        items = enumerate_review_items(
            source_id=_HEX_SOURCE, segments=segs
        )
        assert items[0].anchor_start_word_id == "s0w0"
        # 3 words → last index = 2.
        assert items[0].anchor_end_word_id == "s0w2"

    def test_sentence_mode_splits_on_period(self) -> None:
        segs = _segments(
            [("A", "hello there. friend of mine. final words")]
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            granularity=REVIEW_GRANULARITY_SENTENCE,
        )
        assert len(items) == 3
        # First sentence: words 0-1
        assert items[0].anchor_start_word_id == "s0w0"
        assert items[0].anchor_end_word_id == "s0w1"
        # Second sentence: words 2-4
        assert items[1].anchor_start_word_id == "s0w2"
        assert items[1].anchor_end_word_id == "s0w4"
        # Third (no terminator): words 5-6 ("final", "words")
        assert items[2].anchor_start_word_id == "s0w5"
        assert items[2].anchor_end_word_id == "s0w6"

    def test_sentence_mode_skip_already_coded(self) -> None:
        # Two sentences in segment 0; first one is partly coded.
        segs = _segments([("A", "hello there. world of bees.")])
        app = _make_app(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE_A,
            application_id=_HEX_APP_A,
            start="s0w0",
            end="s0w0",
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            applications=[app],
            granularity=REVIEW_GRANULARITY_SENTENCE,
        )
        # First sentence dropped; second remains.
        assert len(items) == 1
        assert items[0].anchor_start_word_id == "s0w2"

    def test_empty_segments(self) -> None:
        items = enumerate_review_items(
            source_id=_HEX_SOURCE, segments=[]
        )
        assert items == []

    def test_drops_empty_paragraph(self) -> None:
        # A segment with no words should be skipped.
        segs: list[dict[str, Any]] = [
            {"speaker": None, "words": []},
            {"speaker": "A", "words": [{"text": "hi"}]},
        ]
        items = enumerate_review_items(
            source_id=_HEX_SOURCE, segments=segs
        )
        # The empty segment is its own paragraph (None speaker → singleton),
        # but extract_paragraph_text returns empty so it's dropped.
        # The "A" segment becomes one item.
        assert len(items) == 1
        assert items[0].paragraph_start_segment == 1

    def test_invalid_source_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            enumerate_review_items(source_id="not-hex", segments=[])

    def test_invalid_granularity(self) -> None:
        with pytest.raises(ProjectValidationError):
            enumerate_review_items(
                source_id=_HEX_SOURCE,
                segments=[],
                granularity="bogus",
            )

    def test_sentence_mode_records_segment_indices(self) -> None:
        segs = _segments(
            [
                ("A", "first turn."),
                ("B", "second turn."),
            ]
        )
        items = enumerate_review_items(
            source_id=_HEX_SOURCE,
            segments=segs,
            granularity=REVIEW_GRANULARITY_SENTENCE,
        )
        # Each segment has 1 sentence; both have the same segment
        # index on both ends.
        assert all(
            it.paragraph_start_segment == it.paragraph_end_segment
            for it in items
        )
        assert [it.paragraph_start_segment for it in items] == [0, 1]


# --------------------------------------------------------------------------- #
# ReviewPass dataclass
# --------------------------------------------------------------------------- #


class TestReviewPass:
    def _basic(self, **kw: Any) -> ReviewPass:
        defaults: dict[str, Any] = dict(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
        )
        defaults.update(kw)
        return ReviewPass.new(**defaults)

    def test_minimal_pending(self) -> None:
        p = self._basic()
        assert p.status == REVIEW_STATUS_PENDING
        assert p.granularity == REVIEW_GRANULARITY_PARAGRAPH
        assert p.skip_already_coded is True
        assert p.total_spans == 0
        assert p.completed_spans == 0
        assert p.started_at == ""
        assert p.completed_at == ""
        assert REVIEW_PASS_ID_RE.match(p.id)

    def test_validate_invalid_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(project_id="not-hex")

    def test_validate_invalid_source_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(source_id="not-hex")

    def test_validate_invalid_granularity(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(granularity="bogus")

    def test_validate_top_k_min(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(top_k=0)

    def test_validate_max_candidates_min(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(max_candidates=0)

    def test_validate_embedding_weight_bounds(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(embedding_weight=1.5)
        with pytest.raises(ProjectValidationError):
            self._basic(embedding_weight=-0.1)

    def test_validate_min_score_bounds(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(min_score=2.0)

    def test_validate_notes_length(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(notes="x" * (MAX_NOTES_LEN + 1))

    def test_round_trip(self) -> None:
        items = [
            ReviewItem(
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w2",
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                text_preview="hi",
            )
        ]
        p = self._basic(
            items=items,
            embedding_model="bge-m3",
            generation_model="phi-4",
            top_k=3,
            max_candidates=8,
            embedding_weight=0.7,
            min_score=0.1,
            notes="seed pass",
        )
        round_trip = ReviewPass.from_dict(p.to_dict())
        assert round_trip == p

    def test_status_pending_no_timestamps(self) -> None:
        p = self._basic()
        p.started_at = "2026-01-01T00:00:00Z"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_status_running_requires_started_at(self) -> None:
        p = self._basic()
        p.status = REVIEW_STATUS_RUNNING
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_status_completed_requires_both_timestamps(self) -> None:
        p = self._basic()
        p.status = REVIEW_STATUS_COMPLETED
        p.started_at = "2026-01-01T00:00:00Z"
        with pytest.raises(ProjectValidationError):
            p.validate()  # missing completed_at

    def test_error_message_only_on_failed(self) -> None:
        p = self._basic()
        p.error_message = "something"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_progress_getters(self) -> None:
        items = [
            ReviewItem(
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                text_preview="a",
                suggestion_id="0123456789ab",
            ),
            ReviewItem(
                anchor_start_word_id="s1w0",
                anchor_end_word_id="s1w0",
                paragraph_start_segment=1,
                paragraph_end_segment=1,
                text_preview="b",
                error="failed",
            ),
            ReviewItem(
                anchor_start_word_id="s2w0",
                anchor_end_word_id="s2w0",
                paragraph_start_segment=2,
                paragraph_end_segment=2,
                text_preview="c",
            ),
        ]
        p = self._basic(items=items)
        assert p.total_spans == 3
        assert p.completed_spans == 2
        assert p.succeeded_spans == 1
        assert p.failed_spans == 1
        assert p.pending_indices() == [2]

    def test_apply_update_notes(self) -> None:
        p = self._basic()
        p.apply_update({"notes": "new"})
        assert p.notes == "new"

    def test_apply_update_unknown_key(self) -> None:
        p = self._basic()
        with pytest.raises(ProjectValidationError):
            p.apply_update({"status": REVIEW_STATUS_RUNNING})


# --------------------------------------------------------------------------- #
# start_review_pass
# --------------------------------------------------------------------------- #


class TestStartReviewPass:
    def test_persists_pass_with_items(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first."), ("B", "second.")])
        rp = start_review_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            segments=segs,
        )
        assert rp.status == REVIEW_STATUS_PENDING
        assert rp.total_spans == 2
        loaded = load_review_pass(tmp_path, proj.id, rp.id)
        assert loaded.total_spans == 2
        # Items don't include a suggestion_id yet.
        assert all(it.is_pending for it in loaded.items)

    def test_no_items_when_all_coded(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first."), ("B", "second.")])
        apps = [
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_A,
                application_id=_HEX_APP_A,
                start="s0w0",
                end="s0w1",
            ),
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_A,
                application_id=_HEX_APP_B,
                start="s1w0",
                end="s1w1",
            ),
        ]
        rp = start_review_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            segments=segs,
            applications=apps,
        )
        assert rp.total_spans == 0


# --------------------------------------------------------------------------- #
# process_next_review_item
# --------------------------------------------------------------------------- #


class TestProcessNext:
    def _setup(self, tmp_path: Path) -> tuple[Project, ReviewPass, list[Code]]:
        proj = _saved_project(tmp_path)
        segs = _segments(
            [
                ("A", "first turn."),
                ("B", "second turn."),
            ]
        )
        rp = start_review_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            segments=segs,
            embedding_model="test-embed",
        )
        codes = [
            _make_code(
                proj.id,
                code_id=_HEX_CODE_A,
                name="example",
                definition="catch-all definition",
            )
        ]
        return proj, rp, codes

    def test_first_step_flips_to_running(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)
        idx, suggestion = process_next_review_item(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
        )
        assert idx == 0
        assert suggestion is not None
        assert suggestion.decision == SUGGESTION_DECISION_PENDING
        assert rp.status == REVIEW_STATUS_RUNNING
        assert rp.started_at != ""
        assert rp.items[0].suggestion_id == suggestion.id

    def test_step_persists_suggestion(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)
        idx, suggestion = process_next_review_item(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
        )
        assert suggestion is not None
        persisted = list_suggestions(tmp_path, proj.id)
        assert any(s.id == suggestion.id for s in persisted)

    def test_completes_when_no_more_pending(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)
        # Two items in the pass.
        for _ in range(rp.total_spans):
            process_next_review_item(
                rp,
                projects_root=tmp_path,
                codes=codes,
                applications=[],
                embed_fn=_const_embed([1.0, 0.0, 0.0]),
            )
        assert rp.status == REVIEW_STATUS_COMPLETED
        assert rp.completed_at != ""
        # Round-trip from disk.
        loaded = load_review_pass(tmp_path, proj.id, rp.id)
        assert loaded.status == REVIEW_STATUS_COMPLETED

    def test_per_item_error_recorded(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)

        def bad_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise RuntimeError("embedder offline")

        idx, suggestion = process_next_review_item(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=bad_embed,
        )
        assert suggestion is None
        assert rp.items[idx].error.startswith("RuntimeError")
        # Pass keeps running, not failed.
        assert rp.status == REVIEW_STATUS_RUNNING

    def test_validation_error_recorded_as_item_error(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)

        def empty_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return []  # triggers ProjectValidationError inside suggest_codes_for_span

        idx, suggestion = process_next_review_item(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=empty_embed,
        )
        assert suggestion is None
        assert rp.items[idx].error  # non-empty

    def test_terminal_pass_rejects_step(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)
        cancel_review_pass(rp, projects_root=tmp_path)
        with pytest.raises(ProjectValidationError):
            process_next_review_item(
                rp,
                projects_root=tmp_path,
                codes=codes,
                applications=[],
                embed_fn=_const_embed([1.0, 0.0, 0.0]),
            )

    def test_step_with_generate_fn(self, tmp_path: Path) -> None:
        proj, rp, codes = self._setup(tmp_path)
        idx, suggestion = process_next_review_item(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
            generate_fn=_const_generate("[]"),
        )
        assert suggestion is not None


# --------------------------------------------------------------------------- #
# run_review_pass
# --------------------------------------------------------------------------- #


class TestRunReviewPass:
    def test_drives_to_completion(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments(
            [("A", "first."), ("B", "second."), ("A", "third.")]
        )
        rp = start_review_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            segments=segs,
        )
        codes = [
            _make_code(
                proj.id,
                code_id=_HEX_CODE_A,
                name="x",
                definition="d",
            )
        ]
        seen: list[int] = []
        out = run_review_pass(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
            on_step=lambda i, s: seen.append(i),
        )
        assert out.status == REVIEW_STATUS_COMPLETED
        assert seen == list(range(rp.total_spans))

    def test_max_steps_truncates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments(
            [("A", "first."), ("B", "second."), ("A", "third.")]
        )
        rp = start_review_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            segments=segs,
        )
        codes = [
            _make_code(
                proj.id, code_id=_HEX_CODE_A, name="x", definition="d"
            )
        ]
        out = run_review_pass(
            rp,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
            max_steps=1,
        )
        assert out.status == REVIEW_STATUS_RUNNING
        assert out.completed_spans == 1
        # Resume from disk.
        loaded = load_review_pass(tmp_path, proj.id, rp.id)
        assert loaded.completed_spans == 1
        # Continue.
        run_review_pass(
            loaded,
            projects_root=tmp_path,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
        )
        assert loaded.status == REVIEW_STATUS_COMPLETED

    def test_run_on_empty_pass_completes_immediately(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(
            project_id=proj.id, source_id=_HEX_SOURCE
        )
        save_review_pass(tmp_path, rp)
        out = run_review_pass(
            rp,
            projects_root=tmp_path,
            codes=[],
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
        )
        # Empty pass loop simply exits — no items to drive a transition.
        assert out.status == REVIEW_STATUS_PENDING


# --------------------------------------------------------------------------- #
# cancel_review_pass / mark_review_pass_failed
# --------------------------------------------------------------------------- #


class TestLifecycleTransitions:
    def test_cancel_pending(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        cancel_review_pass(rp, projects_root=tmp_path)
        assert rp.status == REVIEW_STATUS_CANCELLED
        assert rp.started_at != ""
        assert rp.completed_at != ""

    def test_cancel_idempotent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        cancel_review_pass(rp, projects_root=tmp_path)
        first = rp.completed_at
        cancel_review_pass(rp, projects_root=tmp_path)
        assert rp.completed_at == first

    def test_cancel_after_completed_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        rp.status = REVIEW_STATUS_COMPLETED
        rp.started_at = "2026-01-01T00:00:00Z"
        rp.completed_at = "2026-01-01T01:00:00Z"
        save_review_pass(tmp_path, rp)
        with pytest.raises(ProjectValidationError):
            cancel_review_pass(rp, projects_root=tmp_path)

    def test_fail_pending(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        mark_review_pass_failed(
            rp,
            projects_root=tmp_path,
            error_message="model unreachable",
        )
        assert rp.status == REVIEW_STATUS_FAILED
        assert rp.error_message == "model unreachable"
        assert rp.completed_at != ""

    def test_fail_truncates_long_message(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        mark_review_pass_failed(
            rp,
            projects_root=tmp_path,
            error_message="x" * (MAX_ERROR_MESSAGE_LEN + 100),
        )
        assert len(rp.error_message) == MAX_ERROR_MESSAGE_LEN

    def test_fail_requires_message(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        with pytest.raises(ProjectValidationError):
            mark_review_pass_failed(
                rp, projects_root=tmp_path, error_message=""
            )

    def test_fail_after_terminal_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        rp.status = REVIEW_STATUS_CANCELLED
        rp.started_at = "2026-01-01T00:00:00Z"
        rp.completed_at = "2026-01-01T01:00:00Z"
        save_review_pass(tmp_path, rp)
        with pytest.raises(ProjectValidationError):
            mark_review_pass_failed(
                rp, projects_root=tmp_path, error_message="late"
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        path = save_review_pass(tmp_path, rp)
        assert path.exists()
        loaded = load_review_pass(tmp_path, proj.id, rp.id)
        assert loaded.id == rp.id
        assert loaded.status == REVIEW_STATUS_PENDING

    def test_save_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        rd = review_passes_dir(tmp_path, proj.id)
        leftovers = [f.name for f in rd.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []

    def test_save_without_project_raises(self, tmp_path: Path) -> None:
        rp = ReviewPass.new(project_id=_HEX_PROJECT, source_id=_HEX_SOURCE)
        with pytest.raises(FileNotFoundError):
            save_review_pass(tmp_path, rp)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_review_pass(tmp_path, proj.id, "0" * 12)

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            review_pass_state_path(tmp_path, _HEX_PROJECT, "not-hex")

    def test_list_filters_by_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        b = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE_2)
        save_review_pass(tmp_path, a)
        save_review_pass(tmp_path, b)
        out = list_review_passes(tmp_path, proj.id, source_id=_HEX_SOURCE)
        assert [r.id for r in out] == [a.id]

    def test_list_filters_by_status(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        b = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, a)
        save_review_pass(tmp_path, b)
        cancel_review_pass(b, projects_root=tmp_path)
        out = list_review_passes(
            tmp_path, proj.id, status=REVIEW_STATUS_CANCELLED
        )
        assert [r.id for r in out] == [b.id]

    def test_list_invalid_status_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_review_passes(tmp_path, proj.id, status="bogus")

    def test_list_invalid_source_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_review_passes(tmp_path, proj.id, source_id="not-hex")

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_review_passes(tmp_path, proj.id) == []

    def test_list_skips_invalid_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rd = review_passes_dir(tmp_path, proj.id)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "stray.txt").write_text("nope")
        (rd / "not-hex.json").write_text("{}")
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        out = list_review_passes(tmp_path, proj.id)
        assert [r.id for r in out] == [rp.id]

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        rp = ReviewPass.new(project_id=proj.id, source_id=_HEX_SOURCE)
        save_review_pass(tmp_path, rp)
        assert delete_review_pass(tmp_path, proj.id, rp.id) is True
        assert delete_review_pass(tmp_path, proj.id, rp.id) is False

    def test_directories(self, tmp_path: Path) -> None:
        path = review_passes_dir(tmp_path, _HEX_PROJECT)
        assert path.name == REVIEW_PASSES_DIRNAME
        assert not path.exists()
