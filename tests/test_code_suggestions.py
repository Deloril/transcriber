"""Tests for scribe.code_suggestions (F8.3).

Covers:

  * CandidateMatch + CodeCandidate dataclasses (validate, round-trip).
  * CodeSuggestion entity (validate, round-trip, apply_update,
    decision invariants).
  * score_candidates: index-only, embed-only, combined, model-name
    filtering, retired-code exclusion, dim mismatch handling,
    exemplar refs preserved.
  * Prompt builder + LLM response parser (plain JSON, fenced, with
    prose, malformed).
  * apply_llm_rerank: ranking, missing codes, weight bounds.
  * suggest_codes_for_span: end-to-end with stub embed/generate.
  * record_decision: each branch + double-decision guard.
  * Persistence: save/load/list/delete + filter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.applications import Application, save_application
from scribe.codes import Code, save_code
from scribe.embedding_index import (
    EMBEDDING_KIND_CODED_SEGMENT,
    EMBEDDING_KIND_UNCODED_PARAGRAPH,
    EmbeddingEntry,
    IndexableSpan,
    save_embedding_entry,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.code_suggestions import (
    CANDIDATE_MATCH_DEFINITION,
    CANDIDATE_MATCH_EXEMPLAR,
    CANDIDATE_MATCH_KINDS,
    CANDIDATE_MATCH_SEGMENT,
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TOP_K,
    MAX_CANDIDATES_PERSISTED,
    MAX_QUERY_TEXT_LEN,
    MAX_RAW_LLM_RESPONSE_LEN,
    SUGGESTION_DECISION_ACCEPTED,
    SUGGESTION_DECISION_MODIFIED,
    SUGGESTION_DECISION_PENDING,
    SUGGESTION_DECISION_REJECTED,
    SUGGESTION_DECISIONS,
    SUGGESTION_ID_RE,
    SUGGESTIONS_DIRNAME,
    CandidateMatch,
    CodeCandidate,
    CodeSuggestion,
    apply_llm_rerank,
    delete_suggestion,
    list_suggestions,
    load_suggestion,
    make_suggestion_prompt,
    new_suggestion_id,
    parse_llm_ranking,
    record_decision,
    save_suggestion,
    score_candidates,
    suggest_codes_for_span,
    suggestion_state_path,
    suggestions_dir,
)


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "aaaaaaaaaaaa"
_HEX_SOURCE = "bbbbbbbbbbbb"
_HEX_SOURCE_2 = "cccccccccccc"
_HEX_CODER = "0123456789ab"
_HEX_VERSION = "fedcba987654"


def _project_id(p: Project) -> str:
    return p.id


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


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


def _make_entry(
    *,
    project_id: str,
    application_id: str,
    source_id: str = _HEX_SOURCE,
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
    model_name: str = "test-embed",
) -> EmbeddingEntry:
    span = IndexableSpan(
        kind=EMBEDDING_KIND_CODED_SEGMENT,
        source_id=source_id,
        application_id=application_id,
        paragraph_start_segment=None,
        paragraph_end_segment=None,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w0",
        text="example text",
    )
    return EmbeddingEntry.new(
        project_id=project_id,
        span=span,
        vector=vector,
        model_name=model_name,
    )


def _const_embed(vec: Sequence[float]):
    """Stub embed_fn that returns the same vector for every input."""
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [tuple(float(x) for x in vec)] * len(texts)
    return fn


def _per_text_embed(routes: Mapping[str, Sequence[float]], default: Sequence[float]):
    """Stub embed_fn that returns a different vector per input."""
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        out: list[tuple[float, ...]] = []
        for t in texts:
            v = routes.get(t, default)
            out.append(tuple(float(x) for x in v))
        return out
    return fn


# --------------------------------------------------------------------------- #
# CandidateMatch
# --------------------------------------------------------------------------- #


class TestCandidateMatch:
    def test_validate_segment_requires_app_id(self) -> None:
        m = CandidateMatch(kind=CANDIDATE_MATCH_SEGMENT, ref="not-hex", score=0.5)
        with pytest.raises(ProjectValidationError):
            m.validate()

    def test_validate_segment_accepts_hex(self) -> None:
        m = CandidateMatch(
            kind=CANDIDATE_MATCH_SEGMENT, ref="0123456789ab", score=0.5
        )
        m.validate()  # no raise

    def test_validate_definition_ref(self) -> None:
        m = CandidateMatch(
            kind=CANDIDATE_MATCH_DEFINITION, ref="definition", score=0.7
        )
        m.validate()

    def test_validate_exemplar_ref(self) -> None:
        m = CandidateMatch(
            kind=CANDIDATE_MATCH_EXEMPLAR, ref="exemplar:3", score=0.4
        )
        m.validate()

    def test_validate_unknown_kind(self) -> None:
        m = CandidateMatch(kind="bogus", ref="x", score=0.0)
        with pytest.raises(ProjectValidationError):
            m.validate()

    def test_validate_score_out_of_range(self) -> None:
        with pytest.raises(ProjectValidationError):
            CandidateMatch(
                kind=CANDIDATE_MATCH_DEFINITION, ref="definition", score=1.5
            ).validate()

    def test_validate_score_nan(self) -> None:
        with pytest.raises(ProjectValidationError):
            CandidateMatch(
                kind=CANDIDATE_MATCH_DEFINITION,
                ref="definition",
                score=float("nan"),
            ).validate()

    def test_round_trip(self) -> None:
        m = CandidateMatch(
            kind=CANDIDATE_MATCH_EXEMPLAR, ref="exemplar:0", score=0.92
        )
        round_trip = CandidateMatch.from_dict(m.to_dict())
        assert round_trip == m

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            CandidateMatch.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_rejects_non_numeric_score(self) -> None:
        with pytest.raises(ProjectValidationError):
            CandidateMatch.from_dict(
                {"kind": CANDIDATE_MATCH_DEFINITION, "ref": "definition", "score": "high"}
            )


# --------------------------------------------------------------------------- #
# CodeCandidate
# --------------------------------------------------------------------------- #


class TestCodeCandidate:
    def test_validate_score_bounds(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeCandidate(
                code_id="0" * 12,
                code_name="x",
                embedding_score=1.5,
                llm_score=0.0,
                combined_score=0.0,
            ).validate()

    def test_validate_bad_code_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeCandidate(
                code_id="not-hex",
                code_name="x",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            ).validate()

    def test_round_trip_includes_matches(self) -> None:
        cand = CodeCandidate(
            code_id="0" * 12,
            code_name="x",
            embedding_score=0.7,
            llm_score=0.5,
            combined_score=0.6,
            rationale="why",
            matches=[
                CandidateMatch(
                    kind=CANDIDATE_MATCH_DEFINITION,
                    ref="definition",
                    score=0.7,
                )
            ],
        )
        d = cand.to_dict()
        out = CodeCandidate.from_dict(d)
        assert out.code_name == "x"
        assert out.matches[0].kind == CANDIDATE_MATCH_DEFINITION
        assert out.matches[0].score == pytest.approx(0.7)

    def test_from_dict_missing_code_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeCandidate.from_dict({"code_name": "x"})

    def test_validate_rationale_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeCandidate(
                code_id="0" * 12,
                code_name="x",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
                rationale="x" * 5000,
            ).validate()


# --------------------------------------------------------------------------- #
# CodeSuggestion entity
# --------------------------------------------------------------------------- #


class TestCodeSuggestionNew:
    def test_default_pending(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            query_text="hello world",
        )
        assert s.decision == SUGGESTION_DECISION_PENDING
        assert s.decided_at == ""
        assert s.decided_by_coder_id == ""
        assert s.created_at == s.modified_at
        assert SUGGESTION_ID_RE.match(s.id)

    def test_explicit_id(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            query_text="hello",
            suggestion_id="abcdef012345",
        )
        assert s.id == "abcdef012345"

    def test_anchor_inversion_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w5",
                anchor_end_word_id="s0w0",
                query_text="hello",
            )

    def test_query_text_size_limit(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="x" * (MAX_QUERY_TEXT_LEN + 1),
            )

    def test_candidates_dict_coerced(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hello",
            candidates=[
                {
                    "code_id": "0" * 12,
                    "code_name": "x",
                    "embedding_score": 0.5,
                    "llm_score": 0.0,
                    "combined_score": 0.5,
                }
            ],
        )
        assert len(s.candidates) == 1
        assert isinstance(s.candidates[0], CodeCandidate)

    def test_too_many_candidates(self) -> None:
        many = [
            CodeCandidate(
                code_id=f"{i:012x}",
                code_name=str(i),
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
            for i in range(MAX_CANDIDATES_PERSISTED + 1)
        ]
        with pytest.raises(ProjectValidationError):
            CodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="hello",
                candidates=many,
            )

    def test_round_trip(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
            query_text="hello world",
            embedding_model="bge-m3",
            generation_model="llama3.2:3b",
            candidates=[
                CodeCandidate(
                    code_id="0" * 12,
                    code_name="x",
                    embedding_score=0.7,
                    llm_score=0.4,
                    combined_score=0.6,
                    rationale="hi",
                )
            ],
            start_char_offset=0,
            end_char_offset=5,
            raw_llm_response="[]",
        )
        out = CodeSuggestion.from_dict(s.to_dict())
        assert out.embedding_model == "bge-m3"
        assert out.candidates[0].code_name == "x"
        assert out.start_char_offset == 0
        assert out.end_char_offset == 5
        assert out.raw_llm_response == "[]"


class TestCodeSuggestionDecisionInvariants:
    def _base(self, **kw: Any) -> CodeSuggestion:
        return CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
            **kw,
        )

    def test_accepted_requires_code_id(self) -> None:
        s = self._base()
        s.decision = SUGGESTION_DECISION_ACCEPTED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_rejected_forbids_accepted_code_id(self) -> None:
        s = self._base()
        s.decision = SUGGESTION_DECISION_REJECTED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        s.accepted_code_id = "0" * 12
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_terminal_requires_coder(self) -> None:
        s = self._base()
        s.decision = SUGGESTION_DECISION_REJECTED
        s.decided_at = "2026-05-26T00:00:00Z"
        # decided_by_coder_id intentionally left blank
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_terminal_requires_decided_at(self) -> None:
        s = self._base()
        s.decision = SUGGESTION_DECISION_REJECTED
        s.decided_by_coder_id = _HEX_CODER
        # decided_at intentionally blank
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_pending_allows_blank_decided_fields(self) -> None:
        s = self._base()
        # default state passes validate
        s.validate()


class TestCodeSuggestionApplyUpdate:
    def test_apply_notes_and_reason(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        s.apply_update({"notes": "interesting", "rejection_reason": "weak fit"})
        assert s.notes == "interesting"
        assert s.rejection_reason == "weak fit"

    def test_apply_unknown_key_rejected(self) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        with pytest.raises(ProjectValidationError):
            s.apply_update({"decision": "accepted"})


# --------------------------------------------------------------------------- #
# score_candidates
# --------------------------------------------------------------------------- #


class TestScoreCandidatesIndexOnly:
    def test_groups_segments_by_code_max(self) -> None:
        # Two applications of code A, one of code B.
        codes = [
            _make_code(_HEX_PROJECT, code_id="aa" * 6, name="A"),
            _make_code(_HEX_PROJECT, code_id="bb" * 6, name="B"),
        ]
        a1 = "a" + "1" * 11   # application id
        a2 = "a" + "2" * 11
        b1 = "b" + "1" * 11
        # Index entries: a1 strong match (1,0,0), a2 weak (0.5,0.5,0.7), b1 strong (0.9,0.1,0.4)
        e_a1 = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0, 0.0),
        )
        e_a2 = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a2,
            vector=(0.0, 1.0, 0.0),
        )
        e_b1 = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=b1,
            vector=(0.7071, 0.7071, 0.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id, a2: codes[0].id, b1: codes[1].id},
            index_entries=[e_a1, e_a2, e_b1],
            embed_fn=None,
            embedding_model="test-embed",
        )
        # A should win because a1 is identical to query.
        assert [c.code_name for c in cand] == ["A", "B"]
        assert cand[0].embedding_score == pytest.approx(1.0, abs=1e-6)
        # A's matches: a1 (1.0) and a2 (0.0); both kept; max is 1.0.
        assert any(m.ref == a1 and m.score == pytest.approx(1.0) for m in cand[0].matches)

    def test_skips_uncoded_paragraph_entries(self) -> None:
        codes = [_make_code(_HEX_PROJECT, code_id="aa" * 6, name="A")]
        # An uncoded_paragraph entry shouldn't contribute even if it
        # somehow ended up in the entries list.
        para_span = IndexableSpan(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=None,
            paragraph_start_segment=0,
            paragraph_end_segment=0,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text="x",
        )
        para_entry = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=para_span,
            vector=(1.0, 0.0),
            model_name="m",
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={},
            index_entries=[para_entry],
        )
        assert cand == []

    def test_filter_by_model_name(self) -> None:
        codes = [_make_code(_HEX_PROJECT, code_id="aa" * 6, name="A")]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0),
            model_name="other-model",
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
            embedding_model="bge-m3",   # mismatch
        )
        assert cand == []

    def test_empty_model_name_does_not_filter(self) -> None:
        codes = [_make_code(_HEX_PROJECT, code_id="aa" * 6, name="A")]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0),
            model_name="anything",
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
            embedding_model="",        # disabled
        )
        assert len(cand) == 1

    def test_dim_mismatch_skipped(self) -> None:
        codes = [_make_code(_HEX_PROJECT, code_id="aa" * 6, name="A")]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0, 0.0, 0.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),     # dim 2 vs dim 4
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
        )
        assert cand == []

    def test_retired_codes_excluded(self) -> None:
        codes = [
            _make_code(_HEX_PROJECT, code_id="aa" * 6, name="A", status="retired"),
        ]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0),
            model_name="bge-m3",
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
            embedding_model="bge-m3",
        )
        assert cand == []

    def test_min_score_drops_low_matches(self) -> None:
        codes = [_make_code(_HEX_PROJECT, code_id="aa" * 6, name="A")]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(0.0, 1.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
            min_score=0.5,
        )
        assert cand == []

    def test_max_candidates_truncates(self) -> None:
        codes = []
        entries = []
        mapping = {}
        for i in range(10):
            cid = f"{i:012x}"
            codes.append(_make_code(_HEX_PROJECT, code_id=cid, name=f"C{i}"))
            aid = f"a{i:011x}"
            mapping[aid] = cid
            entries.append(
                _make_entry(
                    project_id=_HEX_PROJECT,
                    application_id=aid,
                    vector=(1.0, float(i)),
                )
            )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application=mapping,
            index_entries=entries,
            max_candidates=3,
        )
        assert len(cand) == 3


class TestScoreCandidatesEmbedFn:
    def test_definition_only_match(self) -> None:
        # Code A has only a definition. Embed_fn returns identity for
        # the definition text, so it should match the query exactly.
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                definition="resilience under pressure",
            ),
        ]
        embed_fn = _per_text_embed(
            {"resilience under pressure": (1.0, 0.0, 0.0)},
            default=(0.0, 0.0, 0.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0, 0.0),
            codes=codes,
            code_id_by_application={},
            index_entries=[],
            embed_fn=embed_fn,
        )
        assert len(cand) == 1
        assert cand[0].embedding_score == pytest.approx(1.0, abs=1e-6)
        assert cand[0].matches[0].kind == CANDIDATE_MATCH_DEFINITION
        assert cand[0].matches[0].ref == CANDIDATE_MATCH_DEFINITION

    def test_exemplar_ref_uses_code_list_index(self) -> None:
        # Code's own validate() silently drops empty exemplars, so the
        # surviving list is `["alpha", "beta"]`. The exemplar ref tracks
        # this post-validation list — the index the UI would link back to.
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                exemplars=["alpha", "beta"],
            ),
        ]
        embed_fn = _per_text_embed(
            {"alpha": (0.0, 1.0), "beta": (1.0, 0.0)},
            default=(0.0, 0.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),     # matches `beta` (index 1)
            codes=codes,
            code_id_by_application={},
            index_entries=[],
            embed_fn=embed_fn,
        )
        assert cand[0].embedding_score == pytest.approx(1.0, abs=1e-6)
        top = cand[0].matches[0]
        assert top.kind == CANDIDATE_MATCH_EXEMPLAR
        assert top.ref == "exemplar:1"

    def test_combines_index_and_definition(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                definition="weak match text",
            ),
        ]
        a1 = "a" + "1" * 11
        e = _make_entry(
            project_id=_HEX_PROJECT,
            application_id=a1,
            vector=(1.0, 0.0),    # exact match
            model_name="m",
        )
        embed_fn = _per_text_embed(
            {"weak match text": (0.0, 1.0)},     # orthogonal → 0
            default=(0.0, 0.0),
        )
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={a1: codes[0].id},
            index_entries=[e],
            embed_fn=embed_fn,
            embedding_model="m",
        )
        assert len(cand) == 1
        # Top match is the segment, not the definition.
        assert cand[0].matches[0].kind == CANDIDATE_MATCH_SEGMENT
        assert cand[0].embedding_score == pytest.approx(1.0, abs=1e-6)

    def test_dim_mismatch_in_embed_fn_silently_skipped(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                definition="bad-dim",
            )
        ]
        # embed_fn returns the wrong dim — should be ignored, not crash.
        embed_fn = _per_text_embed({"bad-dim": (1.0, 0.0, 0.0, 0.0)}, default=(0.0,))
        cand = score_candidates(
            query_vector=(1.0, 0.0),
            codes=codes,
            code_id_by_application={},
            index_entries=[],
            embed_fn=embed_fn,
        )
        assert cand == []   # no usable matches

    def test_embed_fn_wrong_count_raises(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                definition="x",
            ),
        ]
        # Buggy embed_fn returns zero vectors regardless of input.
        def bad_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return []
        with pytest.raises(ProjectValidationError):
            score_candidates(
                query_vector=(1.0, 0.0),
                codes=codes,
                code_id_by_application={},
                index_entries=[],
                embed_fn=bad_embed,
            )


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


class TestMakeSuggestionPrompt:
    def test_includes_query_and_definitions(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="resilience",
                definition="bouncing back",
            ),
        ]
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="resilience",
                embedding_score=0.9,
                llm_score=0.0,
                combined_score=0.9,
            )
        ]
        prompt = make_suggestion_prompt(
            query_text="I just kept going.",
            candidates=cands,
            codes=codes,
        )
        assert "I just kept going." in prompt
        assert "resilience" in prompt
        assert "bouncing back" in prompt
        assert "aa" * 6 in prompt    # the code id should be visible
        assert "JSON" in prompt

    def test_skips_unknown_candidates(self) -> None:
        # Candidate references a code that's not in the `codes` list.
        cands = [
            CodeCandidate(
                code_id="ff" * 6,
                code_name="ghost",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
        ]
        prompt = make_suggestion_prompt(
            query_text="hi",
            candidates=cands,
            codes=[],
        )
        assert "(none)" in prompt

    def test_includes_inclusion_when_set(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="A",
                definition="def",
            ),
        ]
        codes[0].inclusion_criteria = "applies when X"
        codes[0].exclusion_criteria = "not when Y"
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
        ]
        prompt = make_suggestion_prompt(
            query_text="hi", candidates=cands, codes=codes
        )
        assert "applies when X" in prompt
        assert "not when Y" in prompt


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #


class TestParseLlmRanking:
    def test_plain_json(self) -> None:
        text = '[{"code_id": "aaaaaaaaaaaa", "score": 0.9, "rationale": "fits"}]'
        out = parse_llm_ranking(text)
        assert out == [("aaaaaaaaaaaa", 0.9, "fits")]

    def test_fenced_json(self) -> None:
        text = "```json\n[{\"code_id\": \"aaaaaaaaaaaa\", \"score\": 0.5, \"rationale\": \"x\"}]\n```"
        out = parse_llm_ranking(text)
        assert out == [("aaaaaaaaaaaa", 0.5, "x")]

    def test_with_prose_prefix(self) -> None:
        text = (
            "Sure! Here is your ranking:\n"
            '[{"code_id": "aaaaaaaaaaaa", "score": 0.7, "rationale": "ok"}]\n'
            "Hope that helps!"
        )
        out = parse_llm_ranking(text)
        assert out == [("aaaaaaaaaaaa", 0.7, "ok")]

    def test_invalid_code_id_dropped(self) -> None:
        text = '[{"code_id": "not-hex", "score": 0.5, "rationale": "x"}]'
        out = parse_llm_ranking(text)
        assert out == []

    def test_score_clamped(self) -> None:
        text = '[{"code_id": "aaaaaaaaaaaa", "score": 5.0, "rationale": "x"}]'
        out = parse_llm_ranking(text)
        assert out[0][1] == 1.0
        text = '[{"code_id": "aaaaaaaaaaaa", "score": -0.5, "rationale": "x"}]'
        out = parse_llm_ranking(text)
        assert out[0][1] == 0.0

    def test_non_numeric_score_dropped(self) -> None:
        text = '[{"code_id": "aaaaaaaaaaaa", "score": "high", "rationale": "x"}]'
        assert parse_llm_ranking(text) == []

    def test_non_finite_score_dropped(self) -> None:
        text = '[{"code_id": "aaaaaaaaaaaa", "score": NaN, "rationale": "x"}]'
        # JSON's NaN handling depends on parser; on stdlib it becomes
        # float('nan'); we drop it.
        out = parse_llm_ranking(text)
        assert out == []

    def test_missing_rationale_default_empty(self) -> None:
        text = '[{"code_id": "aaaaaaaaaaaa", "score": 0.5}]'
        out = parse_llm_ranking(text)
        assert out == [("aaaaaaaaaaaa", 0.5, "")]

    def test_empty_response_returns_empty(self) -> None:
        assert parse_llm_ranking("") == []
        assert parse_llm_ranking("   ") == []

    def test_garbage_returns_empty(self) -> None:
        assert parse_llm_ranking("definitely not json") == []

    def test_object_at_top_level_returns_empty(self) -> None:
        # An object — not an array — falls through.
        text = '{"code_id": "aaaaaaaaaaaa"}'
        assert parse_llm_ranking(text) == []


# --------------------------------------------------------------------------- #
# apply_llm_rerank
# --------------------------------------------------------------------------- #


class TestApplyLlmRerank:
    def test_combines_with_default_weight(self) -> None:
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
        ]
        out = apply_llm_rerank(cands, [("aa" * 6, 1.0, "perfect")])
        assert out[0].llm_score == pytest.approx(1.0)
        assert out[0].rationale == "perfect"
        # combined = 0.6 * 0.5 + 0.4 * 1.0 = 0.7
        assert out[0].combined_score == pytest.approx(0.7, abs=1e-6)

    def test_unmentioned_codes_zero_llm_score(self) -> None:
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.9,
                llm_score=0.0,
                combined_score=0.9,
            ),
            CodeCandidate(
                code_id="bb" * 6,
                code_name="B",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            ),
        ]
        # LLM mentions only B.
        out = apply_llm_rerank(
            cands, [("bb" * 6, 1.0, "actually a perfect fit")]
        )
        # B's combined: 0.6*0.5 + 0.4*1.0 = 0.7
        # A's combined: 0.6*0.9 + 0.4*0   = 0.54
        # → B should now outrank A.
        assert out[0].code_id == "bb" * 6
        assert out[0].llm_score == 1.0
        assert out[1].llm_score == 0.0
        assert out[1].combined_score == pytest.approx(0.54, abs=1e-6)

    def test_duplicate_code_in_response_uses_max(self) -> None:
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
        ]
        out = apply_llm_rerank(
            cands,
            [("aa" * 6, 0.4, "first"), ("aa" * 6, 0.9, "second")],
        )
        assert out[0].llm_score == 0.9
        # Rationale is the one that came with the max.
        assert out[0].rationale == "second"

    def test_weight_zero_pure_llm(self) -> None:
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.9,
                llm_score=0.0,
                combined_score=0.9,
            )
        ]
        out = apply_llm_rerank(
            cands,
            [("aa" * 6, 0.2, "weak")],
            embedding_weight=0.0,
        )
        assert out[0].combined_score == pytest.approx(0.2)

    def test_weight_one_pure_embedding(self) -> None:
        cands = [
            CodeCandidate(
                code_id="aa" * 6,
                code_name="A",
                embedding_score=0.9,
                llm_score=0.0,
                combined_score=0.9,
            )
        ]
        out = apply_llm_rerank(
            cands,
            [("aa" * 6, 0.2, "weak")],
            embedding_weight=1.0,
        )
        assert out[0].combined_score == pytest.approx(0.9)
        # llm_score and rationale still recorded for transparency.
        assert out[0].llm_score == 0.2

    def test_weight_out_of_range_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            apply_llm_rerank([], [], embedding_weight=2.0)
        with pytest.raises(ProjectValidationError):
            apply_llm_rerank([], [], embedding_weight=-0.1)


# --------------------------------------------------------------------------- #
# suggest_codes_for_span (orchestration)
# --------------------------------------------------------------------------- #


class TestSuggestCodesForSpan:
    def test_end_to_end_with_index(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        code = _make_code(proj.id, code_id="aa" * 6, name="A")
        save_code(tmp_path, code)
        a1 = "a" + "1" * 11
        app = _make_app(
            project_id=proj.id, code_id=code.id, application_id=a1
        )
        save_application(tmp_path, app)
        # Save an entry for that application.
        e = _make_entry(
            project_id=proj.id,
            application_id=a1,
            vector=(1.0, 0.0, 0.0),
            model_name="bge-m3",
        )
        save_embedding_entry(tmp_path, e)
        embed_fn = _const_embed((1.0, 0.0, 0.0))   # query == segment
        s = suggest_codes_for_span(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hello world",
            codes=[code],
            applications=[app],
            embed_fn=embed_fn,
            generate_fn=None,
            embedding_model="bge-m3",
        )
        assert s.decision == SUGGESTION_DECISION_PENDING
        assert len(s.candidates) == 1
        assert s.candidates[0].code_id == code.id
        assert s.candidates[0].embedding_score == pytest.approx(1.0, abs=1e-6)
        assert s.embedding_model == "bge-m3"
        assert s.generation_model == ""
        assert s.raw_llm_response == ""

    def test_with_generate_fn_records_response(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        code = _make_code(
            proj.id,
            code_id="aa" * 6,
            name="A",
            definition="resilience",
        )
        save_code(tmp_path, code)
        embed_fn = _const_embed((1.0, 0.0))   # constant means everything matches
        rendered_prompt: list[str] = []
        def gen_fn(prompt: str) -> str:
            rendered_prompt.append(prompt)
            return f'[{{"code_id": "{code.id}", "score": 0.85, "rationale": "fits"}}]'
        s = suggest_codes_for_span(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="bouncing back",
            codes=[code],
            applications=[],
            embed_fn=embed_fn,
            generate_fn=gen_fn,
            generation_model="llama3.2:3b",
        )
        assert s.generation_model == "llama3.2:3b"
        # Prompt was built with the candidate.
        assert "resilience" in rendered_prompt[0]
        # Raw response stored.
        assert "0.85" in s.raw_llm_response
        # LLM rerank ran: llm_score reflects parsed value.
        assert s.candidates[0].llm_score == pytest.approx(0.85)

    def test_empty_query_text_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            suggest_codes_for_span(
                projects_root=tmp_path,
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="    ",
                codes=[],
                applications=[],
                embed_fn=_const_embed((1.0, 0.0)),
            )

    def test_truncated_to_top_k(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        codes = []
        apps = []
        entries = []
        # 8 codes, all with strong index match.
        for i in range(8):
            cid = f"{i:012x}"
            c = _make_code(proj.id, code_id=cid, name=f"C{i}")
            save_code(tmp_path, c)
            codes.append(c)
            aid = f"a{i:011x}"
            a = _make_app(
                project_id=proj.id,
                code_id=cid,
                application_id=aid,
            )
            save_application(tmp_path, a)
            apps.append(a)
            e = _make_entry(
                project_id=proj.id,
                application_id=aid,
                vector=(1.0 - 0.01 * i, 0.01 * i),
                model_name="m",
            )
            save_embedding_entry(tmp_path, e)
            entries.append(e)
        s = suggest_codes_for_span(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hello",
            codes=codes,
            applications=apps,
            embed_fn=_const_embed((1.0, 0.0)),
            embedding_model="m",
            top_k=3,
            max_candidates=DEFAULT_MAX_CANDIDATES,
        )
        assert len(s.candidates) == 3

    def test_huge_llm_response_truncated(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        code = _make_code(
            proj.id,
            code_id="aa" * 6,
            name="A",
            definition="x",
        )
        save_code(tmp_path, code)
        big_response = "x" * (MAX_RAW_LLM_RESPONSE_LEN + 100)
        s = suggest_codes_for_span(
            projects_root=tmp_path,
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hello",
            codes=[code],
            applications=[],
            embed_fn=_const_embed((1.0, 0.0)),
            generate_fn=lambda _: big_response,
        )
        assert len(s.raw_llm_response) == MAX_RAW_LLM_RESPONSE_LEN


# --------------------------------------------------------------------------- #
# record_decision
# --------------------------------------------------------------------------- #


def _pending_suggestion() -> CodeSuggestion:
    return CodeSuggestion.new(
        project_id=_HEX_PROJECT,
        source_id=_HEX_SOURCE,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w0",
        query_text="x",
        candidates=[
            CodeCandidate(
                code_id="0" * 12,
                code_name="A",
                embedding_score=0.5,
                llm_score=0.0,
                combined_score=0.5,
            )
        ],
    )


class TestRecordDecision:
    def test_accept_records_code_and_coder(self) -> None:
        s = _pending_suggestion()
        record_decision(
            s,
            decision=SUGGESTION_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
            accepted_code_id="0" * 12,
            accepted_application_id="ab" * 6,
        )
        assert s.decision == SUGGESTION_DECISION_ACCEPTED
        assert s.accepted_code_id == "0" * 12
        assert s.accepted_application_id == "ab" * 6
        assert s.decided_by_coder_id == _HEX_CODER
        assert s.decided_at != ""

    def test_modify_records_different_code(self) -> None:
        s = _pending_suggestion()
        # Researcher picked a code that wasn't even in the candidate list.
        record_decision(
            s,
            decision=SUGGESTION_DECISION_MODIFIED,
            coder_id=_HEX_CODER,
            accepted_code_id="ff" * 6,
        )
        assert s.decision == SUGGESTION_DECISION_MODIFIED
        assert s.accepted_code_id == "ff" * 6
        assert s.accepted_application_id is None

    def test_reject_with_reason(self) -> None:
        s = _pending_suggestion()
        record_decision(
            s,
            decision=SUGGESTION_DECISION_REJECTED,
            coder_id=_HEX_CODER,
            rejection_reason="Off-topic; just small talk.",
        )
        assert s.decision == SUGGESTION_DECISION_REJECTED
        assert s.accepted_code_id is None
        assert s.rejection_reason == "Off-topic; just small talk."

    def test_reject_with_accepted_code_rejected(self) -> None:
        s = _pending_suggestion()
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_REJECTED,
                coder_id=_HEX_CODER,
                accepted_code_id="0" * 12,
            )

    def test_accept_without_code_id_rejected(self) -> None:
        s = _pending_suggestion()
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
            )

    def test_double_decision_rejected(self) -> None:
        s = _pending_suggestion()
        record_decision(
            s,
            decision=SUGGESTION_DECISION_REJECTED,
            coder_id=_HEX_CODER,
        )
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_code_id="0" * 12,
            )

    def test_pending_decision_not_terminal(self) -> None:
        s = _pending_suggestion()
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_PENDING,
                coder_id=_HEX_CODER,
            )

    def test_bad_coder_id_rejected(self) -> None:
        s = _pending_suggestion()
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_REJECTED,
                coder_id="not-hex",
            )

    def test_bad_application_id_rejected(self) -> None:
        s = _pending_suggestion()
        with pytest.raises(ProjectValidationError):
            record_decision(
                s,
                decision=SUGGESTION_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_code_id="0" * 12,
                accepted_application_id="not-hex",
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        path = save_suggestion(tmp_path, s)
        assert path.exists()
        loaded = load_suggestion(tmp_path, proj.id, s.id)
        assert loaded.id == s.id
        assert loaded.decision == SUGGESTION_DECISION_PENDING

    def test_save_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        save_suggestion(tmp_path, s)
        # No leftover .json.tmp
        sd = suggestions_dir(tmp_path, proj.id)
        leftovers = [f.name for f in sd.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []

    def test_save_without_project_raises(self, tmp_path: Path) -> None:
        s = CodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        with pytest.raises(FileNotFoundError):
            save_suggestion(tmp_path, s)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_suggestion(tmp_path, proj.id, "0" * 12)

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            suggestion_state_path(tmp_path, _HEX_PROJECT, "not-hex")

    def test_list_filters_by_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="a",
        )
        s2 = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE_2,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="b",
        )
        save_suggestion(tmp_path, s1)
        save_suggestion(tmp_path, s2)
        only_first = list_suggestions(tmp_path, proj.id, source_id=_HEX_SOURCE)
        assert [s.id for s in only_first] == [s1.id]

    def test_list_filters_by_decision(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="a",
        )
        s2 = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="b",
        )
        record_decision(
            s2,
            decision=SUGGESTION_DECISION_REJECTED,
            coder_id=_HEX_CODER,
        )
        save_suggestion(tmp_path, s1)
        save_suggestion(tmp_path, s2)
        rejected = list_suggestions(
            tmp_path, proj.id, decision=SUGGESTION_DECISION_REJECTED
        )
        assert [s.id for s in rejected] == [s2.id]
        pending = list_suggestions(
            tmp_path, proj.id, decision=SUGGESTION_DECISION_PENDING
        )
        assert [s.id for s in pending] == [s1.id]

    def test_list_invalid_decision_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_suggestions(tmp_path, proj.id, decision="bogus")

    def test_list_skips_non_suggestion_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sd = suggestions_dir(tmp_path, proj.id)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "stray.txt").write_text("nothing")
        (sd / "not-hex.json").write_text("{}")
        s = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        save_suggestion(tmp_path, s)
        out = list_suggestions(tmp_path, proj.id)
        assert [r.id for r in out] == [s.id]

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_suggestions(tmp_path, proj.id) == []

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = CodeSuggestion.new(
            project_id=proj.id,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
        )
        save_suggestion(tmp_path, s)
        assert delete_suggestion(tmp_path, proj.id, s.id) is True
        assert delete_suggestion(tmp_path, proj.id, s.id) is False

    def test_directories(self, tmp_path: Path) -> None:
        # suggestions_dir doesn't create.
        path = suggestions_dir(tmp_path, _HEX_PROJECT)
        assert path.name == SUGGESTIONS_DIRNAME
        assert not path.exists()

    def test_new_suggestion_id_is_unique_hex(self) -> None:
        ids = {new_suggestion_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(SUGGESTION_ID_RE.match(i) for i in ids)
