"""Tests for scribe.ai_second_coder (F8.7).

Covers:

  * SecondCoderItemDiff / SecondCoderDiff: validate, round-trip,
    set-algebra accessors.
  * CodeICR / SecondCoderICR: round-trip.
  * SecondCoderPass entity: validate, round-trip, status timestamp
    invariants.
  * start_second_coder_pass:
      - refuses on unlocked codebook (CodebookNotLockedError);
      - rejects bad ids / granularity;
      - persists pass with linked review_pass_id.
  * compute_second_coder_diff: matches AI top-N + human applications;
    respects min_score; ignores other-source / other-coder applications.
  * compute_second_coder_icr:
      - empty / no-codes shortcut;
      - per-code kappa values;
      - flattened overall kappa over multi-label encoding;
      - items_with_full_agreement / disagreement counts.
  * process_next_second_coder_item / run_second_coder_pass:
      - delegates to the inner review pass;
      - flips status running → completed and computes ICR;
      - rejects on terminal pass.
  * cancel_second_coder_pass / mark_second_coder_pass_failed:
      - propagates to inner review pass when supplied;
      - terminal-state guard.
  * Persistence: save / load / list / delete with filters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.applications import Application
from scribe.codebook_lock import lock_codebook
from scribe.code_suggestions import (
    CodeCandidate,
    CodeSuggestion,
    save_suggestion,
)
from scribe.codes import Code
from scribe.icr import interpret_kappa
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.transcript_review import (
    REVIEW_GRANULARITY_PARAGRAPH,
    REVIEW_GRANULARITY_SENTENCE,
    REVIEW_STATUS_CANCELLED,
    REVIEW_STATUS_COMPLETED,
    REVIEW_STATUS_FAILED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_RUNNING,
    ReviewItem,
    ReviewPass,
    load_review_pass,
    save_review_pass,
)
from scribe.ai_second_coder import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_N,
    MAX_NOTES_LEN,
    SECOND_CODER_PASS_ID_RE,
    SECOND_CODER_PASSES_DIRNAME,
    SECOND_CODER_STATUS_CANCELLED,
    SECOND_CODER_STATUS_COMPLETED,
    SECOND_CODER_STATUS_FAILED,
    SECOND_CODER_STATUS_PENDING,
    SECOND_CODER_STATUS_RUNNING,
    SECOND_CODER_STATUSES,
    SECOND_CODER_TERMINAL_STATUSES,
    CodebookNotLockedError,
    CodeICR,
    SecondCoderDiff,
    SecondCoderICR,
    SecondCoderItemDiff,
    SecondCoderPass,
    cancel_second_coder_pass,
    compute_and_store_icr,
    compute_second_coder_diff,
    compute_second_coder_icr,
    delete_second_coder_pass,
    is_terminal_second_coder_status,
    list_second_coder_passes,
    load_second_coder_pass,
    mark_second_coder_pass_failed,
    new_second_coder_pass_id,
    process_next_second_coder_item,
    run_second_coder_pass,
    save_second_coder_pass,
    second_coder_pass_state_path,
    second_coder_passes_dir,
    start_second_coder_pass,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "aaaaaaaaaaaa"
_HEX_SOURCE = "bbbbbbbbbbbb"
_HEX_SOURCE_2 = "cccccccccccc"
_HEX_HUMAN = "0123456789ab"
_HEX_HUMAN_2 = "1111aaaa1111"
_HEX_VERSION = "fedcba987654"
_HEX_CODE_A = "1111cccc1111"
_HEX_CODE_B = "2222dddd2222"
_HEX_CODE_C = "3333eeee3333"
_HEX_PASS = "0102030405ab"


def _saved_project(tmp_path: Path, *, name: str = "P", locked: bool = True) -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    if locked:
        lock_codebook(tmp_path, p.id, reason="ICR pass")
    return p


def _make_app(
    *,
    project_id: str,
    code_id: str,
    application_id: str,
    coder_id: str = _HEX_HUMAN,
    source_id: str = _HEX_SOURCE,
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
    code_id: str,
    name: str = "code",
    definition: str = "",
) -> Code:
    return Code.new(
        project_id=project_id,
        code_id=code_id,
        name=name,
        definition=definition,
    )


def _segments(turns: Sequence[tuple[str | None, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for speaker, text in turns:
        words = [{"text": w} for w in text.split(" ") if w]
        out.append({"speaker": speaker, "words": words})
    return out


def _const_embed(vec: Sequence[float]):
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [tuple(float(x) for x in vec)] * len(texts)
    return fn


def _make_candidate(
    code_id: str,
    *,
    name: str = "c",
    score: float = 0.7,
) -> CodeCandidate:
    return CodeCandidate(
        code_id=code_id,
        code_name=name,
        embedding_score=float(score),
        llm_score=0.0,
        combined_score=float(score),
        rationale="",
        matches=[],
    )


def _make_suggestion(
    *,
    project_id: str,
    source_id: str = _HEX_SOURCE,
    anchor_start: str = "s0w0",
    anchor_end: str = "s0w2",
    suggestion_id: str | None = None,
    candidates: Sequence[CodeCandidate] = (),
) -> CodeSuggestion:
    return CodeSuggestion.new(
        project_id=project_id,
        source_id=source_id,
        anchor_start_word_id=anchor_start,
        anchor_end_word_id=anchor_end,
        query_text="hello world",
        suggestion_id=suggestion_id,
        candidates=list(candidates),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_new_id_is_12_hex(self) -> None:
        rid = new_second_coder_pass_id()
        assert SECOND_CODER_PASS_ID_RE.match(rid)

    def test_new_id_unique(self) -> None:
        ids = {new_second_coder_pass_id() for _ in range(50)}
        assert len(ids) == 50

    @pytest.mark.parametrize(
        "status",
        [
            SECOND_CODER_STATUS_COMPLETED,
            SECOND_CODER_STATUS_CANCELLED,
            SECOND_CODER_STATUS_FAILED,
        ],
    )
    def test_terminal_true(self, status: str) -> None:
        assert is_terminal_second_coder_status(status) is True

    @pytest.mark.parametrize(
        "status",
        [SECOND_CODER_STATUS_PENDING, SECOND_CODER_STATUS_RUNNING],
    )
    def test_terminal_false(self, status: str) -> None:
        assert is_terminal_second_coder_status(status) is False


# --------------------------------------------------------------------------- #
# SecondCoderItemDiff
# --------------------------------------------------------------------------- #


class TestSecondCoderItemDiff:
    def _make(self, **kw: Any) -> SecondCoderItemDiff:
        defaults: dict[str, Any] = dict(
            item_index=0,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
            paragraph_start_segment=0,
            paragraph_end_segment=0,
            suggestion_id="abcdef012345",
            ai_code_ids=[_HEX_CODE_A],
            human_code_ids=[_HEX_CODE_A, _HEX_CODE_B],
        )
        defaults.update(kw)
        return SecondCoderItemDiff(**defaults)

    def test_validate_negative_index(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(item_index=-1).validate()

    def test_validate_paragraph_end_before_start(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(
                paragraph_start_segment=2, paragraph_end_segment=1
            ).validate()

    def test_validate_bad_anchor(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(anchor_start_word_id="not-a-word-id").validate()

    def test_validate_bad_suggestion_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(suggestion_id="zzz").validate()

    def test_validate_empty_code_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make(ai_code_ids=[""]).validate()
        with pytest.raises(ProjectValidationError):
            self._make(human_code_ids=[""]).validate()

    def test_round_trip(self) -> None:
        it = self._make()
        rt = SecondCoderItemDiff.from_dict(it.to_dict())
        assert rt == it

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            SecondCoderItemDiff.from_dict([])  # type: ignore[arg-type]

    def test_set_algebra(self) -> None:
        it = self._make(
            ai_code_ids=[_HEX_CODE_A, _HEX_CODE_B],
            human_code_ids=[_HEX_CODE_B, _HEX_CODE_C],
        )
        assert it.agreement_codes == [_HEX_CODE_B]
        assert it.ai_only_codes == [_HEX_CODE_A]
        assert it.human_only_codes == [_HEX_CODE_C]


# --------------------------------------------------------------------------- #
# SecondCoderDiff round-trip
# --------------------------------------------------------------------------- #


class TestSecondCoderDiff:
    def test_round_trip(self) -> None:
        items = [
            SecondCoderItemDiff(
                item_index=0,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                suggestion_id="abcdef012345",
                ai_code_ids=[_HEX_CODE_A],
                human_code_ids=[_HEX_CODE_A],
            ),
        ]
        d = SecondCoderDiff(items=items)
        rt = SecondCoderDiff.from_dict(d.to_dict())
        assert rt == d

    def test_from_dict_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            SecondCoderDiff.from_dict([])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CodeICR / SecondCoderICR round-trip
# --------------------------------------------------------------------------- #


class TestICRDataclasses:
    def test_code_icr_round_trip(self) -> None:
        c = CodeICR(
            code_id=_HEX_CODE_A,
            ai_count=2,
            human_count=3,
            both_count=1,
            kappa=0.42,
            interpretation="fair",
        )
        rt = CodeICR.from_dict(c.to_dict())
        assert rt == c

    def test_code_icr_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeICR.from_dict([])  # type: ignore[arg-type]

    def test_second_coder_icr_round_trip(self) -> None:
        icr = SecondCoderICR(
            n_items=4,
            n_codes=2,
            overall_observed_agreement=0.75,
            overall_expected_agreement=0.5,
            overall_kappa=0.5,
            overall_interpretation="moderate",
            items_with_full_agreement=2,
            items_with_any_disagreement=2,
            per_code=[
                CodeICR(
                    code_id=_HEX_CODE_A,
                    ai_count=3,
                    human_count=3,
                    both_count=2,
                    kappa=0.5,
                    interpretation="moderate",
                ),
            ],
        )
        rt = SecondCoderICR.from_dict(icr.to_dict())
        assert rt == icr

    def test_second_coder_icr_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            SecondCoderICR.from_dict([])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SecondCoderPass entity
# --------------------------------------------------------------------------- #


class TestSecondCoderPassEntity:
    def _basic(self, **kw: Any) -> SecondCoderPass:
        defaults: dict[str, Any] = dict(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
        )
        defaults.update(kw)
        return SecondCoderPass.new(**defaults)

    def test_pending_default(self) -> None:
        p = self._basic()
        assert p.status == SECOND_CODER_STATUS_PENDING
        assert p.review_pass_id == ""
        assert p.icr_results == {}

    def test_validate_bad_human_coder(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(human_coder_id="not-hex")

    def test_validate_bad_review_pass_id(self) -> None:
        p = self._basic()
        p.review_pass_id = "not-hex"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_validate_bad_top_n(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(top_n=0)

    def test_validate_bad_min_score(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(min_score=2.0)

    def test_validate_bad_granularity(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(granularity="line")

    def test_validate_notes_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(notes="x" * (MAX_NOTES_LEN + 1))

    def test_pending_must_not_have_completed_at(self) -> None:
        p = self._basic()
        p.completed_at = "2026-01-01T00:00:00Z"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_completed_must_have_timestamps(self) -> None:
        p = self._basic()
        p.status = SECOND_CODER_STATUS_COMPLETED
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_failed_requires_started_at(self) -> None:
        p = self._basic()
        p.status = SECOND_CODER_STATUS_FAILED
        p.error_message = "broken"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_error_message_only_when_failed(self) -> None:
        p = self._basic()
        p.error_message = "should not be set"
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_round_trip(self) -> None:
        p = self._basic(top_n=3, min_score=0.5)
        rt = SecondCoderPass.from_dict(p.to_dict())
        assert rt == p

    def test_from_dict_missing_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            SecondCoderPass.from_dict({"id": "x"})

    def test_from_dict_bad_icr_results_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            SecondCoderPass.from_dict(
                {
                    "id": "0102030405ab",
                    "project_id": _HEX_PROJECT,
                    "source_id": _HEX_SOURCE,
                    "human_coder_id": _HEX_HUMAN,
                    "icr_results": [],
                }
            )

    def test_apply_update_notes(self) -> None:
        p = self._basic()
        p.apply_update({"notes": "new"})
        assert p.notes == "new"

    def test_apply_update_unknown_key(self) -> None:
        p = self._basic()
        with pytest.raises(ProjectValidationError):
            p.apply_update({"status": SECOND_CODER_STATUS_RUNNING})


# --------------------------------------------------------------------------- #
# start_second_coder_pass
# --------------------------------------------------------------------------- #


class TestStart:
    def test_refuses_unlocked_codebook(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path, locked=False)
        segs = _segments([("A", "first.")])
        with pytest.raises(CodebookNotLockedError):
            start_second_coder_pass(
                projects_root=tmp_path,
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                human_coder_id=_HEX_HUMAN,
                segments=segs,
            )

    def test_starts_on_locked_codebook(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first."), ("B", "second.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
            embedding_model="test-embed",
        )
        assert sp.status == SECOND_CODER_STATUS_PENDING
        assert sp.review_pass_id != ""
        # Inner review pass exists on disk.
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        assert rp.total_spans == 2
        # skip_already_coded=False is the F8.7 contract.
        assert rp.skip_already_coded is False
        # Outer pass loadable from disk.
        loaded = load_second_coder_pass(tmp_path, proj.id, sp.id)
        assert loaded.id == sp.id
        assert loaded.review_pass_id == sp.review_pass_id

    def test_rejects_bad_human_coder(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        with pytest.raises(ProjectValidationError):
            start_second_coder_pass(
                projects_root=tmp_path,
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                human_coder_id="not-hex",
                segments=segs,
            )

    def test_rejects_bad_granularity(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        with pytest.raises(ProjectValidationError):
            start_second_coder_pass(
                projects_root=tmp_path,
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                human_coder_id=_HEX_HUMAN,
                segments=segs,
                granularity="line",
            )

    def test_rejects_bad_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            start_second_coder_pass(
                projects_root=tmp_path,
                project_id="not-hex",
                source_id=_HEX_SOURCE,
                human_coder_id=_HEX_HUMAN,
                segments=_segments([("A", "x.")]),
            )

    def test_rejects_bad_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            start_second_coder_pass(
                projects_root=tmp_path,
                project_id=proj.id,
                source_id="not-hex",
                human_coder_id=_HEX_HUMAN,
                segments=_segments([("A", "x.")]),
            )

    def test_persists_pass_with_top_n_and_min_score(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
            top_n=3,
            min_score=0.4,
        )
        loaded = load_second_coder_pass(tmp_path, proj.id, sp.id)
        assert loaded.top_n == 3
        assert loaded.min_score == 0.4


# --------------------------------------------------------------------------- #
# compute_second_coder_diff
# --------------------------------------------------------------------------- #


class TestComputeDiff:
    def _setup_with_review_pass(
        self,
        tmp_path: Path,
        *,
        ai_top_codes_per_item: Sequence[Sequence[str]],
        human_apps: Sequence[Application],
        top_n: int = 1,
        min_score: float = 0.0,
    ) -> tuple[SecondCoderPass, ReviewPass]:
        proj = _saved_project(tmp_path)
        segs = _segments(
            [("A", "first turn."), ("B", "second turn.")]
        )
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
            top_n=top_n,
            min_score=min_score,
        )
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        # For each item, save a hand-crafted suggestion with the
        # supplied AI top candidates and link via item.suggestion_id.
        for idx, code_ids in enumerate(ai_top_codes_per_item):
            it = rp.items[idx]
            cands = [
                _make_candidate(cid, score=0.9 - 0.1 * j)
                for j, cid in enumerate(code_ids)
            ]
            sug = _make_suggestion(
                project_id=proj.id,
                anchor_start=it.anchor_start_word_id,
                anchor_end=it.anchor_end_word_id,
                candidates=cands,
            )
            save_suggestion(tmp_path, sug)
            it.suggestion_id = sug.id
        # Persist the modified review pass.
        save_review_pass(tmp_path, rp)
        return sp, rp

    def test_diff_picks_top_n_above_min_score(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A, _HEX_CODE_B], [_HEX_CODE_B]],
            human_apps=[],
            top_n=1,
        )
        # Human applied code A on first paragraph, code B on second.
        apps = [
            _make_app(
                project_id=sp.project_id,
                code_id=_HEX_CODE_A,
                application_id="aaa1aaa1aaa1",
                start="s0w0",
                end="s0w1",
            ),
            _make_app(
                project_id=sp.project_id,
                code_id=_HEX_CODE_B,
                application_id="bbb1bbb1bbb1",
                start="s1w0",
                end="s1w1",
            ),
        ]
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=apps,
        )
        assert len(diff.items) == 2
        # Item 0: AI's top-1 was code A; human applied A → agreement.
        assert diff.items[0].ai_code_ids == [_HEX_CODE_A]
        assert diff.items[0].human_code_ids == [_HEX_CODE_A]
        assert diff.items[0].agreement_codes == [_HEX_CODE_A]
        # Item 1: AI top-1 was code B; human applied B → agreement.
        assert diff.items[1].ai_code_ids == [_HEX_CODE_B]
        assert diff.items[1].human_code_ids == [_HEX_CODE_B]

    def test_diff_top_n_two(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[
                [_HEX_CODE_A, _HEX_CODE_B],
                [_HEX_CODE_C, _HEX_CODE_A],
            ],
            human_apps=[],
            top_n=2,
        )
        apps = [
            _make_app(
                project_id=sp.project_id,
                code_id=_HEX_CODE_A,
                application_id="aaa1aaa1aaa1",
                start="s0w0",
                end="s0w1",
            ),
        ]
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=apps,
        )
        assert sorted(diff.items[0].ai_code_ids) == sorted(
            [_HEX_CODE_A, _HEX_CODE_B]
        )
        assert sorted(diff.items[1].ai_code_ids) == sorted(
            [_HEX_CODE_C, _HEX_CODE_A]
        )

    def test_diff_min_score_filters(self, tmp_path: Path) -> None:
        # Top candidate score is 0.9; second is 0.8. With min_score=0.85,
        # only the top one survives even with top_n=2.
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A, _HEX_CODE_B]],
            human_apps=[],
            top_n=2,
            min_score=0.85,
        )
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=[],
        )
        assert diff.items[0].ai_code_ids == [_HEX_CODE_A]

    def test_diff_ignores_other_coder_apps(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A], [_HEX_CODE_B]],
            human_apps=[],
        )
        # An application by a *different* coder should not count.
        apps = [
            _make_app(
                project_id=sp.project_id,
                code_id=_HEX_CODE_A,
                application_id="aaa1aaa1aaa1",
                coder_id=_HEX_HUMAN_2,
                start="s0w0",
                end="s0w1",
            ),
        ]
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=apps,
        )
        assert diff.items[0].human_code_ids == []

    def test_diff_ignores_other_source_apps(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A], [_HEX_CODE_B]],
            human_apps=[],
        )
        apps = [
            _make_app(
                project_id=sp.project_id,
                code_id=_HEX_CODE_A,
                application_id="aaa1aaa1aaa1",
                source_id=_HEX_SOURCE_2,
                start="s0w0",
                end="s0w1",
            ),
        ]
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=apps,
        )
        assert diff.items[0].human_code_ids == []

    def test_diff_skips_pending_items(self, tmp_path: Path) -> None:
        # Make a pass with no suggestions; all items pending.
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[],   # no items with suggestions
            human_apps=[],
        )
        # Everything still pending.
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=[],
        )
        assert diff.items == []

    def test_diff_records_per_item_error(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A], [_HEX_CODE_B]],
            human_apps=[],
        )
        # Mark the first item as errored; clear its suggestion id.
        rp.items[0].suggestion_id = None
        rp.items[0].error = "embedder offline"
        save_review_pass(tmp_path, rp)
        diff = compute_second_coder_diff(
            projects_root=tmp_path,
            pass_record=sp,
            review_pass=rp,
            applications=[],
        )
        # Errored item still appears in the diff but has empty AI codes
        # and the error string.
        assert diff.items[0].error == "embedder offline"
        assert diff.items[0].ai_code_ids == []

    def test_diff_rejects_mismatched_project(self, tmp_path: Path) -> None:
        sp, rp = self._setup_with_review_pass(
            tmp_path,
            ai_top_codes_per_item=[[_HEX_CODE_A]],
            human_apps=[],
        )
        # Hand a review pass with a different project_id.
        rp.project_id = "ffffffffffff"
        with pytest.raises(ProjectValidationError):
            compute_second_coder_diff(
                projects_root=tmp_path,
                pass_record=sp,
                review_pass=rp,
                applications=[],
            )


# --------------------------------------------------------------------------- #
# compute_second_coder_icr
# --------------------------------------------------------------------------- #


def _diff_from_pairs(
    pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> SecondCoderDiff:
    """Helper: build a SecondCoderDiff from (ai_codes, human_codes) pairs."""
    items: list[SecondCoderItemDiff] = []
    for idx, (ai, human) in enumerate(pairs):
        items.append(
            SecondCoderItemDiff(
                item_index=idx,
                anchor_start_word_id=f"s{idx}w0",
                anchor_end_word_id=f"s{idx}w0",
                paragraph_start_segment=idx,
                paragraph_end_segment=idx,
                suggestion_id="abcdef012345",
                ai_code_ids=list(ai),
                human_code_ids=list(human),
            )
        )
    return SecondCoderDiff(items=items)


class TestComputeICR:
    def test_empty_diff(self) -> None:
        icr = compute_second_coder_icr(SecondCoderDiff(items=[]))
        assert icr.n_items == 0
        assert icr.n_codes == 0
        assert icr.overall_kappa == 1.0
        assert icr.per_code == []

    def test_only_errored_items(self) -> None:
        items = [
            SecondCoderItemDiff(
                item_index=0,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                suggestion_id="",
                error="oops",
            )
        ]
        icr = compute_second_coder_icr(SecondCoderDiff(items=items))
        assert icr.n_items == 0
        assert icr.overall_kappa == 1.0

    def test_perfect_agreement(self) -> None:
        diff = _diff_from_pairs(
            [
                ([_HEX_CODE_A], [_HEX_CODE_A]),
                ([_HEX_CODE_B], [_HEX_CODE_B]),
                ([_HEX_CODE_A], [_HEX_CODE_A]),
            ]
        )
        icr = compute_second_coder_icr(diff)
        assert icr.n_items == 3
        assert icr.n_codes == 2
        assert icr.items_with_full_agreement == 3
        assert icr.items_with_any_disagreement == 0
        # All per-code kappas should be 1.0 (perfect agreement on each).
        for c in icr.per_code:
            assert c.kappa == pytest.approx(1.0)
            assert c.interpretation == interpret_kappa(c.kappa)
        assert icr.overall_kappa == pytest.approx(1.0)
        assert icr.overall_interpretation == "almost perfect"

    def test_total_disagreement_picks_chance_level(self) -> None:
        # AI says A on every item; human says B on every item. Both
        # lists are constant single-label, but disjoint marginals →
        # observed = 0, expected = 0 (each side never picks the
        # other's label), kappa formula resolves cleanly to 0.
        diff = _diff_from_pairs(
            [
                ([_HEX_CODE_A], [_HEX_CODE_B]),
                ([_HEX_CODE_A], [_HEX_CODE_B]),
            ]
        )
        icr = compute_second_coder_icr(diff)
        # With our binary multi-label encoding, each item contributes
        # 2 (item, code) decisions — both disagree. observed = 0;
        # expected ≠ 0 (both labels appear). Kappa is negative.
        assert icr.overall_kappa <= 0
        # Items with any disagreement should be all of them.
        assert icr.items_with_any_disagreement == 2

    def test_per_code_kappa_breakdown(self) -> None:
        # Code A: AI applies on items 0, 1; human applies on items 0, 1.
        # → perfect agreement on A.
        # Code B: AI applies on item 0; human applies on item 1.
        # → perfect disagreement on B.
        diff = _diff_from_pairs(
            [
                ([_HEX_CODE_A, _HEX_CODE_B], [_HEX_CODE_A]),
                ([_HEX_CODE_A], [_HEX_CODE_A, _HEX_CODE_B]),
            ]
        )
        icr = compute_second_coder_icr(diff)
        per_code_map = {c.code_id: c for c in icr.per_code}
        # Code A perfectly agrees → kappa 1.0.
        assert per_code_map[_HEX_CODE_A].kappa == pytest.approx(1.0)
        assert per_code_map[_HEX_CODE_A].both_count == 2
        # Code B disagrees on both items.
        assert per_code_map[_HEX_CODE_B].both_count == 0
        # AI/human counts.
        assert per_code_map[_HEX_CODE_A].ai_count == 2
        assert per_code_map[_HEX_CODE_A].human_count == 2
        assert per_code_map[_HEX_CODE_B].ai_count == 1
        assert per_code_map[_HEX_CODE_B].human_count == 1

    def test_partial_agreement_counts(self) -> None:
        # Item 0: full agreement; Item 1: partial; Item 2: full disagreement.
        diff = _diff_from_pairs(
            [
                ([_HEX_CODE_A], [_HEX_CODE_A]),
                ([_HEX_CODE_A, _HEX_CODE_B], [_HEX_CODE_A]),
                ([_HEX_CODE_C], [_HEX_CODE_B]),
            ]
        )
        icr = compute_second_coder_icr(diff)
        assert icr.items_with_full_agreement == 1
        assert icr.items_with_any_disagreement == 2

    def test_no_codes_in_universe(self) -> None:
        # Both sides applied nothing; every code-decision is "absent".
        diff = _diff_from_pairs([([], []), ([], [])])
        icr = compute_second_coder_icr(diff)
        assert icr.n_items == 2
        assert icr.n_codes == 0
        assert icr.overall_kappa == pytest.approx(1.0)
        assert icr.items_with_full_agreement == 2


# --------------------------------------------------------------------------- #
# process_next_second_coder_item / run_second_coder_pass
# --------------------------------------------------------------------------- #


class TestProcessAndRun:
    def _setup(self, tmp_path: Path) -> tuple[Project, SecondCoderPass, ReviewPass, list[Code]]:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first."), ("B", "second.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
            embedding_model="test-embed",
        )
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        codes = [
            _make_code(
                proj.id, code_id=_HEX_CODE_A, name="x", definition="d"
            )
        ]
        return proj, sp, rp, codes

    def test_first_step_flips_running(self, tmp_path: Path) -> None:
        _, sp, rp, codes = self._setup(tmp_path)
        idx, suggestion = process_next_second_coder_item(
            sp,
            projects_root=tmp_path,
            review_pass=rp,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
        )
        assert idx == 0
        assert suggestion is not None
        assert sp.status == SECOND_CODER_STATUS_RUNNING
        assert sp.started_at != ""

    def test_completes_when_review_pass_does(self, tmp_path: Path) -> None:
        _, sp, rp, codes = self._setup(tmp_path)
        for _ in range(rp.total_spans):
            process_next_second_coder_item(
                sp,
                projects_root=tmp_path,
                review_pass=rp,
                codes=codes,
                applications=[],
                embed_fn=_const_embed([1.0, 0.0, 0.0]),
            )
        assert sp.status == SECOND_CODER_STATUS_COMPLETED
        assert sp.completed_at != ""
        # ICR results stored.
        assert "n_items" in sp.icr_results
        # Round-trip from disk.
        loaded = load_second_coder_pass(tmp_path, sp.project_id, sp.id)
        assert loaded.status == SECOND_CODER_STATUS_COMPLETED
        assert loaded.icr_results["n_items"] == sp.icr_results["n_items"]

    def test_terminal_pass_rejects_step(self, tmp_path: Path) -> None:
        _, sp, rp, codes = self._setup(tmp_path)
        cancel_second_coder_pass(
            sp, projects_root=tmp_path, review_pass=rp
        )
        with pytest.raises(ProjectValidationError):
            process_next_second_coder_item(
                sp,
                projects_root=tmp_path,
                review_pass=rp,
                codes=codes,
                applications=[],
                embed_fn=_const_embed([1.0, 0.0, 0.0]),
            )

    def test_run_drives_to_completion(self, tmp_path: Path) -> None:
        _, sp, rp, codes = self._setup(tmp_path)
        seen: list[int] = []
        out = run_second_coder_pass(
            sp,
            projects_root=tmp_path,
            review_pass=rp,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
            on_step=lambda i, s: seen.append(i),
        )
        assert out.status == SECOND_CODER_STATUS_COMPLETED
        assert seen == list(range(rp.total_spans))

    def test_run_max_steps(self, tmp_path: Path) -> None:
        _, sp, rp, codes = self._setup(tmp_path)
        out = run_second_coder_pass(
            sp,
            projects_root=tmp_path,
            review_pass=rp,
            codes=codes,
            applications=[],
            embed_fn=_const_embed([1.0, 0.0, 0.0]),
            max_steps=1,
        )
        assert out.status == SECOND_CODER_STATUS_RUNNING
        assert rp.completed_spans == 1


# --------------------------------------------------------------------------- #
# cancel / fail
# --------------------------------------------------------------------------- #


class TestCancelFail:
    def test_cancel_propagates_to_review_pass(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        cancel_second_coder_pass(
            sp, projects_root=tmp_path, review_pass=rp
        )
        assert sp.status == SECOND_CODER_STATUS_CANCELLED
        rp_loaded = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        assert rp_loaded.status == "cancelled"

    def test_cancel_idempotent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        cancel_second_coder_pass(sp, projects_root=tmp_path)
        # Second call is a no-op (no exception).
        cancel_second_coder_pass(sp, projects_root=tmp_path)
        assert sp.status == SECOND_CODER_STATUS_CANCELLED

    def test_cancel_completed_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        # Force terminal state directly.
        sp.status = SECOND_CODER_STATUS_COMPLETED
        sp.started_at = "2026-01-01T00:00:00Z"
        sp.completed_at = "2026-01-01T00:00:01Z"
        with pytest.raises(ProjectValidationError):
            cancel_second_coder_pass(sp, projects_root=tmp_path)

    def test_mark_failed(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        mark_second_coder_pass_failed(
            sp,
            projects_root=tmp_path,
            error_message="model unreachable",
        )
        assert sp.status == SECOND_CODER_STATUS_FAILED
        assert sp.error_message == "model unreachable"

    def test_mark_failed_propagates_to_review_pass(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        mark_second_coder_pass_failed(
            sp,
            projects_root=tmp_path,
            review_pass=rp,
            error_message="model unreachable",
        )
        rp_loaded = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        assert rp_loaded.status == "failed"

    def test_mark_failed_requires_message(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        with pytest.raises(ProjectValidationError):
            mark_second_coder_pass_failed(
                sp, projects_root=tmp_path, error_message=""
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sp = SecondCoderPass.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
        )
        save_second_coder_pass(tmp_path, sp)
        loaded = load_second_coder_pass(tmp_path, proj.id, sp.id)
        assert loaded == sp

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        sp = SecondCoderPass.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
        )
        with pytest.raises(FileNotFoundError):
            save_second_coder_pass(tmp_path, sp)

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            second_coder_pass_state_path(
                tmp_path, _HEX_PROJECT, "not-hex"
            )

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_second_coder_pass(tmp_path, proj.id, _HEX_PASS)

    def test_list_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sp1 = SecondCoderPass.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            now="2026-01-01T00:00:00Z",
        )
        sp2 = SecondCoderPass.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE_2,
            human_coder_id=_HEX_HUMAN_2,
            now="2026-01-02T00:00:00Z",
        )
        save_second_coder_pass(tmp_path, sp1)
        save_second_coder_pass(tmp_path, sp2)
        # No filter
        all_passes = list_second_coder_passes(tmp_path, proj.id)
        assert {p.id for p in all_passes} == {sp1.id, sp2.id}
        # By source.
        only_s1 = list_second_coder_passes(
            tmp_path, proj.id, source_id=_HEX_SOURCE
        )
        assert [p.id for p in only_s1] == [sp1.id]
        # By coder.
        only_h2 = list_second_coder_passes(
            tmp_path, proj.id, human_coder_id=_HEX_HUMAN_2
        )
        assert [p.id for p in only_h2] == [sp2.id]
        # By status.
        only_pending = list_second_coder_passes(
            tmp_path, proj.id, status=SECOND_CODER_STATUS_PENDING
        )
        assert {p.id for p in only_pending} == {sp1.id, sp2.id}

    def test_list_invalid_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_second_coder_passes(tmp_path, proj.id, source_id="not-hex")
        with pytest.raises(ProjectValidationError):
            list_second_coder_passes(
                tmp_path, proj.id, human_coder_id="not-hex"
            )
        with pytest.raises(ProjectValidationError):
            list_second_coder_passes(tmp_path, proj.id, status="bogus")

    def test_list_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # No second_coder_passes/ subdirectory exists yet.
        assert list_second_coder_passes(tmp_path, proj.id) == []

    def test_list_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        d = second_coder_passes_dir(tmp_path, proj.id)
        d.mkdir(parents=True, exist_ok=True)
        # Bad JSON.
        (d / f"{_HEX_PASS}.json").write_text("not-json")
        # Non-hex stem (skipped by filter).
        (d / "not-hex.json").write_text("{}")
        assert list_second_coder_passes(tmp_path, proj.id) == []

    def test_delete_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sp = SecondCoderPass.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
        )
        save_second_coder_pass(tmp_path, sp)
        assert delete_second_coder_pass(tmp_path, proj.id, sp.id) is True
        assert delete_second_coder_pass(tmp_path, proj.id, sp.id) is False


# --------------------------------------------------------------------------- #
# compute_and_store_icr
# --------------------------------------------------------------------------- #


class TestComputeAndStoreICR:
    def test_stamps_results_on_pass(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", "first.")])
        sp = start_second_coder_pass(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            human_coder_id=_HEX_HUMAN,
            segments=segs,
        )
        rp = load_review_pass(tmp_path, proj.id, sp.review_pass_id)
        icr = compute_and_store_icr(
            sp,
            projects_root=tmp_path,
            review_pass=rp,
            applications=[],
        )
        assert isinstance(icr, SecondCoderICR)
        assert sp.icr_results == icr.to_dict()
