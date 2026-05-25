"""Tests for scribe.new_code_suggestions (F8.4).

Covers:

  * ``looks_like_gerund`` heuristic.
  * NewCodeProposal dataclass: validate, round-trip, name trimming,
    duplicate-field bounds, exemplar truncation/dropping.
  * NewCodeSuggestion entity: validate, round-trip, decision
    invariants, apply_update.
  * ``find_near_duplicates``: returns top-K, drops below threshold,
    handles empty corpus, handles dim mismatch.
  * ``annotate_proposal_duplicates``: per-proposal nearest match,
    one batched embed call, no codes → unchanged.
  * Prompt builder: gerund nudge present, existing-codes block present,
    omitted when no shortlist.
  * ``parse_proposals_response``: plain JSON, fenced JSON, prose
    prefix, malformed → empty, drops missing names, clamps confidence.
  * ``suggest_new_codes_for_span``: end-to-end with stub
    embed/generate; canonicalisation; raw response truncation.
  * ``record_new_code_decision``: each branch + double-decision guard.
  * Persistence: save/load/list/delete + filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.codes import Code
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.new_code_suggestions import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DEFAULT_NEAR_DUPLICATE_TOP_K,
    DEFAULT_NUM_PROPOSALS,
    MAX_DEFINITION_LEN,
    MAX_NAME_LEN,
    MAX_PROPOSALS_PERSISTED,
    MAX_QUERY_TEXT_LEN,
    MAX_QUOTE_EXCERPTS,
    MAX_QUOTE_EXCERPT_LEN,
    MAX_RAW_LLM_RESPONSE_LEN,
    NEW_CODE_DECISION_ACCEPTED,
    NEW_CODE_DECISION_MODIFIED,
    NEW_CODE_DECISION_PENDING,
    NEW_CODE_DECISION_REJECTED,
    NEW_CODE_DECISIONS,
    NEW_CODE_SUGGESTION_ID_RE,
    NEW_CODE_SUGGESTIONS_DIRNAME,
    NewCodeProposal,
    NewCodeSuggestion,
    annotate_proposal_duplicates,
    delete_new_code_suggestion,
    find_near_duplicates,
    list_new_code_suggestions,
    load_new_code_suggestion,
    looks_like_gerund,
    make_new_code_prompt,
    new_code_suggestion_state_path,
    new_code_suggestions_dir,
    new_new_code_suggestion_id,
    parse_proposals_response,
    record_new_code_decision,
    save_new_code_suggestion,
    suggest_new_codes_for_span,
)


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "aaaaaaaaaaaa"
_HEX_SOURCE = "bbbbbbbbbbbb"
_HEX_SOURCE_2 = "cccccccccccc"
_HEX_CODER = "0123456789ab"


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


def _per_text_embed(
    routes: Mapping[str, Sequence[float]],
    default: Sequence[float],
):
    """Deterministic embed_fn that maps text → vector."""

    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        out: list[tuple[float, ...]] = []
        for t in texts:
            v = routes.get(t, default)
            out.append(tuple(float(x) for x in v))
        return out

    return fn


def _const_embed(vec: Sequence[float]):
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [tuple(float(x) for x in vec)] * len(texts)

    return fn


# --------------------------------------------------------------------------- #
# looks_like_gerund
# --------------------------------------------------------------------------- #


class TestLooksLikeGerund:
    def test_simple_gerund(self) -> None:
        assert looks_like_gerund("negotiating identity") is True

    def test_capitalised_gerund(self) -> None:
        assert looks_like_gerund("Managing uncertainty") is True

    def test_short_word_rejected(self) -> None:
        # "ring" ends in -ing but is too short to be a gerund
        # (we require ≥ 4 chars before the "ing", so 7-letter total).
        assert looks_like_gerund("ring loud") is False
        assert looks_like_gerund("sing along") is False

    def test_noun_phrase_rejected(self) -> None:
        assert looks_like_gerund("identity work") is False
        assert looks_like_gerund("the way forward") is False

    def test_empty(self) -> None:
        assert looks_like_gerund("") is False
        assert looks_like_gerund("   ") is False

    def test_non_string(self) -> None:
        assert looks_like_gerund(123) is False  # type: ignore[arg-type]
        assert looks_like_gerund(None) is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# NewCodeProposal
# --------------------------------------------------------------------------- #


class TestNewCodeProposalValidate:
    def test_name_required(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(name="").validate()

    def test_name_whitespace_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(name="   ").validate()

    def test_name_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(name="x" * (MAX_NAME_LEN + 1)).validate()

    def test_definition_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(
                name="x", definition="x" * (MAX_DEFINITION_LEN + 1)
            ).validate()

    def test_too_many_excerpts(self) -> None:
        p = NewCodeProposal(
            name="x", quote_excerpts=["e"] * (MAX_QUOTE_EXCERPTS + 1)
        )
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_excerpt_too_long(self) -> None:
        p = NewCodeProposal(
            name="x",
            quote_excerpts=["x" * (MAX_QUOTE_EXCERPT_LEN + 1)],
        )
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_excerpts_drop_empty(self) -> None:
        p = NewCodeProposal(
            name="x", quote_excerpts=["", "  ", "real"]
        )
        p.validate()
        assert p.quote_excerpts == ["real"]

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(name="x", confidence=2.0).validate()

    def test_confidence_clamped_within_tolerance(self) -> None:
        # Within the validator's tolerance band (≤ 1.0001) → clamped to 1.0.
        p = NewCodeProposal(name="x", confidence=1.00005)
        p.validate()
        assert p.confidence == 1.0

    def test_nearest_existing_code_id_invalid(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(
                name="x", nearest_existing_code_id="not-hex"
            ).validate()

    def test_nearest_similarity_out_of_range(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal(
                name="x", nearest_existing_similarity=1.5
            ).validate()

    def test_name_trimmed_in_place(self) -> None:
        p = NewCodeProposal(name="  managing change  ")
        p.validate()
        assert p.name == "managing change"

    def test_gerund_flag_recomputed(self) -> None:
        # Caller passes is_gerund=True for a non-gerund name; the
        # validator overrides because the flag must always reflect the
        # current name.
        p = NewCodeProposal(name="identity work", is_gerund=True)
        p.validate()
        assert p.is_gerund is False

        p2 = NewCodeProposal(name="negotiating change", is_gerund=False)
        p2.validate()
        assert p2.is_gerund is True


class TestNewCodeProposalSerialisation:
    def test_round_trip(self) -> None:
        p = NewCodeProposal(
            name="managing uncertainty",
            definition="Acting under incomplete information",
            rationale="Speaker explicitly says they don't know",
            quote_excerpts=["I had no idea what to do"],
            confidence=0.7,
            nearest_existing_code_id="0" * 12,
            nearest_existing_similarity=0.6,
        )
        out = NewCodeProposal.from_dict(p.to_dict())
        assert out.name == "managing uncertainty"
        assert out.is_gerund is True
        assert out.confidence == pytest.approx(0.7)
        assert out.nearest_existing_code_id == "0" * 12
        assert out.nearest_existing_similarity == pytest.approx(0.6)

    def test_from_dict_missing_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal.from_dict({"definition": "x"})

    def test_from_dict_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_excerpts_must_be_list(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeProposal.from_dict(
                {"name": "x", "quote_excerpts": "not a list"}
            )

    def test_from_dict_empty_nearest_id_treated_as_none(self) -> None:
        p = NewCodeProposal.from_dict(
            {"name": "x", "nearest_existing_code_id": ""}
        )
        assert p.nearest_existing_code_id is None


# --------------------------------------------------------------------------- #
# NewCodeSuggestion
# --------------------------------------------------------------------------- #


class TestNewCodeSuggestionNew:
    def test_default_pending(self) -> None:
        s = NewCodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            query_text="hello world",
        )
        assert s.decision == NEW_CODE_DECISION_PENDING
        assert s.decided_at == ""
        assert s.decided_by_coder_id == ""
        assert s.created_at == s.modified_at
        assert NEW_CODE_SUGGESTION_ID_RE.match(s.id)
        assert s.accepted_proposal_index is None
        assert s.created_code_id is None

    def test_explicit_id(self) -> None:
        s = NewCodeSuggestion.new(
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
            NewCodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w5",
                anchor_end_word_id="s0w0",
                query_text="hello",
            )

    def test_query_text_size_limit(self) -> None:
        with pytest.raises(ProjectValidationError):
            NewCodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="x" * (MAX_QUERY_TEXT_LEN + 1),
            )

    def test_proposals_dict_coerced(self) -> None:
        s = NewCodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hello",
            proposals=[{"name": "doing things"}],
        )
        assert len(s.proposals) == 1
        assert isinstance(s.proposals[0], NewCodeProposal)
        assert s.proposals[0].name == "doing things"

    def test_too_many_proposals(self) -> None:
        many = [
            NewCodeProposal(name=f"name{i}")
            for i in range(MAX_PROPOSALS_PERSISTED + 1)
        ]
        with pytest.raises(ProjectValidationError):
            NewCodeSuggestion.new(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="hi",
                proposals=many,
            )

    def test_round_trip(self) -> None:
        s = NewCodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w2",
            query_text="hello world",
            embedding_model="bge-m3",
            generation_model="llama3.2:3b",
            proposals=[
                NewCodeProposal(
                    name="navigating change",
                    definition="def",
                    rationale="r",
                    confidence=0.8,
                )
            ],
            start_char_offset=0,
            end_char_offset=5,
            raw_llm_response="[]",
        )
        out = NewCodeSuggestion.from_dict(s.to_dict())
        assert out.embedding_model == "bge-m3"
        assert out.proposals[0].name == "navigating change"
        assert out.start_char_offset == 0
        assert out.end_char_offset == 5
        assert out.raw_llm_response == "[]"


class TestDecisionInvariants:
    def _base(self, **kw: Any) -> NewCodeSuggestion:
        defaults: dict[str, Any] = dict(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
            proposals=[NewCodeProposal(name="navigating change")],
        )
        defaults.update(kw)
        return NewCodeSuggestion.new(**defaults)

    def test_accepted_requires_proposal_index(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_ACCEPTED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_modified_requires_proposal_index(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_MODIFIED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_rejected_forbids_proposal_index(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_REJECTED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        s.accepted_proposal_index = 0
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_rejected_forbids_created_code_id(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_REJECTED
        s.decided_at = "2026-05-26T00:00:00Z"
        s.decided_by_coder_id = _HEX_CODER
        s.created_code_id = "0" * 12
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_terminal_requires_coder(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_REJECTED
        s.decided_at = "2026-05-26T00:00:00Z"
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_terminal_requires_decided_at(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_REJECTED
        s.decided_by_coder_id = _HEX_CODER
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_invalid_proposal_index_high(self) -> None:
        s = self._base()
        s.decision = NEW_CODE_DECISION_ACCEPTED
        s.decided_at = "t"
        s.decided_by_coder_id = _HEX_CODER
        s.accepted_proposal_index = 99
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_invalid_proposal_index_negative(self) -> None:
        s = self._base()
        s.accepted_proposal_index = -1
        with pytest.raises(ProjectValidationError):
            s.validate()

    def test_invalid_created_code_id(self) -> None:
        s = self._base()
        s.created_code_id = "not-hex"
        with pytest.raises(ProjectValidationError):
            s.validate()


class TestApplyUpdate:
    def _base(self) -> NewCodeSuggestion:
        return NewCodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
            proposals=[NewCodeProposal(name="doing X")],
        )

    def test_unknown_field_rejected(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            s.apply_update({"bogus": 1})

    def test_notes_patched(self) -> None:
        s = self._base()
        s.apply_update({"notes": "saw this twice"})
        assert s.notes == "saw this twice"

    def test_created_code_id_patched(self) -> None:
        s = self._base()
        s.apply_update({"created_code_id": "0" * 12})
        assert s.created_code_id == "0" * 12

    def test_created_code_id_cleared(self) -> None:
        s = self._base()
        s.created_code_id = "0" * 12
        s.apply_update({"created_code_id": None})
        assert s.created_code_id is None

    def test_invalid_payload_type(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            s.apply_update("not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# find_near_duplicates
# --------------------------------------------------------------------------- #


class TestFindNearDuplicates:
    def test_returns_top_k(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="resilience",
                definition="bouncing back",
            ),
            _make_code(
                _HEX_PROJECT,
                code_id="bb" * 6,
                name="grit",
                definition="determination",
            ),
        ]
        embed_fn = _per_text_embed(
            {
                "resilience. bouncing back": (1.0, 0.0),
                "grit. determination": (0.0, 1.0),
            },
            default=(0.0, 0.0),
        )
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),
            codes=codes,
            embed_fn=embed_fn,
            top_k=5,
            min_score=0.0,
        )
        assert out[0][0] == "aa" * 6
        assert out[0][1] == "resilience"
        assert out[0][2] == pytest.approx(1.0, abs=1e-6)
        # grit has zero similarity
        assert any(r[0] == "bb" * 6 for r in out)

    def test_min_score_filter(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="resilience",
                definition="bouncing back",
            ),
            _make_code(
                _HEX_PROJECT,
                code_id="bb" * 6,
                name="grit",
                definition="determination",
            ),
        ]
        embed_fn = _per_text_embed(
            {
                "resilience. bouncing back": (1.0, 0.0),
                "grit. determination": (0.0, 1.0),
            },
            default=(0.0, 0.0),
        )
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),
            codes=codes,
            embed_fn=embed_fn,
            top_k=5,
            min_score=0.5,
        )
        assert len(out) == 1
        assert out[0][0] == "aa" * 6

    def test_empty_codes(self) -> None:
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),
            codes=[],
            embed_fn=_const_embed((1.0, 0.0)),
        )
        assert out == []

    def test_retired_codes_skipped(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="retired-code",
                definition="old",
                status="retired",
            ),
        ]
        embed_fn = _per_text_embed(
            {"retired-code. old": (1.0, 0.0)},
            default=(0.0, 0.0),
        )
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),
            codes=codes,
            embed_fn=embed_fn,
            min_score=0.0,
        )
        assert out == []

    def test_dim_mismatch_skipped(self) -> None:
        codes = [
            _make_code(_HEX_PROJECT, code_id="aa" * 6, name="x", definition="d"),
        ]
        embed_fn = _per_text_embed(
            {"x. d": (1.0, 0.0, 0.0)},  # 3D
            default=(0.0, 0.0, 0.0),
        )
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),  # 2D
            codes=codes,
            embed_fn=embed_fn,
        )
        assert out == []

    def test_exemplar_max_score(self) -> None:
        # Definition is orthogonal to the query but an exemplar matches.
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="x",
                definition="off-topic",
                exemplars=["matching exemplar"],
            ),
        ]
        embed_fn = _per_text_embed(
            {
                "x. off-topic": (0.0, 1.0),
                "matching exemplar": (1.0, 0.0),
            },
            default=(0.0, 0.0),
        )
        out = find_near_duplicates(
            query_vector=(1.0, 0.0),
            codes=codes,
            embed_fn=embed_fn,
            min_score=0.5,
        )
        assert len(out) == 1
        assert out[0][2] == pytest.approx(1.0, abs=1e-6)

    def test_embed_fn_wrong_count_raises(self) -> None:
        codes = [
            _make_code(_HEX_PROJECT, code_id="aa" * 6, name="x", definition="d"),
        ]

        def bad_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return []

        with pytest.raises(ProjectValidationError):
            find_near_duplicates(
                query_vector=(1.0, 0.0),
                codes=codes,
                embed_fn=bad_embed,
            )


# --------------------------------------------------------------------------- #
# annotate_proposal_duplicates
# --------------------------------------------------------------------------- #


class TestAnnotateProposalDuplicates:
    def test_per_proposal_nearest(self) -> None:
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="resilience",
                definition="bouncing back",
            ),
            _make_code(
                _HEX_PROJECT,
                code_id="bb" * 6,
                name="grit",
                definition="determination",
            ),
        ]
        proposals = [
            NewCodeProposal(
                name="bouncing back", definition="similar to resilience"
            ),
            NewCodeProposal(
                name="grinding it out", definition="similar to grit"
            ),
        ]
        # Map proposal "name. definition" → orientations
        embed_fn = _per_text_embed(
            {
                "bouncing back. similar to resilience": (1.0, 0.0),
                "grinding it out. similar to grit": (0.0, 1.0),
                "resilience. bouncing back": (1.0, 0.0),
                "grit. determination": (0.0, 1.0),
            },
            default=(0.0, 0.0),
        )
        out = annotate_proposal_duplicates(
            proposals,
            codes=codes,
            embed_fn=embed_fn,
        )
        assert out[0].nearest_existing_code_id == "aa" * 6
        assert out[0].nearest_existing_similarity == pytest.approx(1.0, abs=1e-6)
        assert out[1].nearest_existing_code_id == "bb" * 6
        assert out[1].nearest_existing_similarity == pytest.approx(1.0, abs=1e-6)

    def test_no_codes_returns_clones(self) -> None:
        proposals = [
            NewCodeProposal(name="doing things"),
        ]
        embed_fn = _const_embed((1.0, 0.0))
        out = annotate_proposal_duplicates(
            proposals, codes=[], embed_fn=embed_fn
        )
        assert len(out) == 1
        # Returned object is a clone, not the same instance — caller's
        # input shouldn't be mutated.
        assert out[0] is not proposals[0]
        assert out[0].nearest_existing_code_id is None

    def test_empty_proposals(self) -> None:
        out = annotate_proposal_duplicates(
            [],
            codes=[_make_code(_HEX_PROJECT, code_id="aa" * 6, name="x")],
            embed_fn=_const_embed((1.0,)),
        )
        assert out == []

    def test_dim_mismatch_silent(self) -> None:
        codes = [
            _make_code(_HEX_PROJECT, code_id="aa" * 6, name="x", definition="d"),
        ]
        proposals = [NewCodeProposal(name="probe")]
        # Proposal embeds at 2D, corpus at 3D → no match recorded.
        embed_fn = _per_text_embed(
            {
                "probe": (1.0, 0.0),
                "x. d": (1.0, 0.0, 0.0),
            },
            default=(0.0, 0.0),
        )
        out = annotate_proposal_duplicates(
            proposals, codes=codes, embed_fn=embed_fn
        )
        assert out[0].nearest_existing_code_id is None


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


class TestMakeNewCodePrompt:
    def test_includes_query(self) -> None:
        prompt = make_new_code_prompt(query_text="I just kept going.")
        assert "I just kept going." in prompt
        assert "JSON" in prompt
        assert "GERUND" in prompt or "gerund" in prompt.lower()

    def test_existing_codes_block_included(self) -> None:
        prompt = make_new_code_prompt(
            query_text="hi",
            existing_codes_to_avoid=[
                ("aa" * 6, "resilience", 0.9),
                ("bb" * 6, "grit", 0.8),
            ],
        )
        assert "resilience" in prompt
        assert "grit" in prompt
        # The "do NOT duplicate" framing must be present.
        assert "do not duplicate" in prompt.lower() or "do NOT duplicate" in prompt

    def test_existing_codes_block_omitted_when_empty(self) -> None:
        prompt = make_new_code_prompt(query_text="hi")
        # No "Existing codes" header in the absence of a shortlist.
        assert "Existing codes already in the codebook" not in prompt

    def test_existing_codes_skip_blank_names(self) -> None:
        prompt = make_new_code_prompt(
            query_text="hi",
            existing_codes_to_avoid=[
                ("aa" * 6, "", 0.9),
                ("bb" * 6, "real-name", 0.8),
            ],
        )
        assert "real-name" in prompt
        # Should still render a usable block.
        assert "Existing codes already in the codebook" in prompt

    def test_num_proposals_propagates(self) -> None:
        prompt = make_new_code_prompt(query_text="hi", num_proposals=3)
        assert "at most 3" in prompt


# --------------------------------------------------------------------------- #
# parse_proposals_response
# --------------------------------------------------------------------------- #


class TestParseProposalsResponse:
    def test_plain_json(self) -> None:
        text = json.dumps(
            [
                {
                    "name": "navigating change",
                    "definition": "d",
                    "rationale": "r",
                    "quote_excerpts": ["q"],
                    "confidence": 0.7,
                }
            ]
        )
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert out[0].name == "navigating change"
        assert out[0].confidence == pytest.approx(0.7)
        assert out[0].quote_excerpts == ["q"]

    def test_fenced_json(self) -> None:
        text = (
            "```json\n"
            '[{"name": "doing X", "definition": "d", "confidence": 0.5}]\n'
            "```"
        )
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert out[0].name == "doing X"

    def test_prose_prefix(self) -> None:
        text = (
            "Sure — here are some ideas:\n"
            '[{"name": "managing X"}]\n'
            "Hope that helps!"
        )
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert out[0].name == "managing X"

    def test_empty_string(self) -> None:
        assert parse_proposals_response("") == []

    def test_malformed_json(self) -> None:
        # Not parseable as JSON, no [...] either.
        assert parse_proposals_response("not json at all") == []

    def test_drops_missing_name(self) -> None:
        text = json.dumps(
            [
                {"definition": "no name"},
                {"name": "valid name"},
            ]
        )
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert out[0].name == "valid name"

    def test_drops_empty_name(self) -> None:
        text = json.dumps([{"name": "  "}, {"name": "valid"}])
        out = parse_proposals_response(text)
        assert [p.name for p in out] == ["valid"]

    def test_clamps_confidence(self) -> None:
        text = json.dumps([{"name": "n", "confidence": 5.0}])
        out = parse_proposals_response(text)
        assert out[0].confidence == 1.0
        text2 = json.dumps([{"name": "n", "confidence": -2}])
        out2 = parse_proposals_response(text2)
        assert out2[0].confidence == 0.0

    def test_non_numeric_confidence_zero(self) -> None:
        text = json.dumps([{"name": "n", "confidence": "high"}])
        out = parse_proposals_response(text)
        assert out[0].confidence == 0.0

    def test_long_name_truncated(self) -> None:
        text = json.dumps([{"name": "x" * (MAX_NAME_LEN + 100)}])
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert len(out[0].name) <= MAX_NAME_LEN

    def test_long_excerpt_truncated(self) -> None:
        text = json.dumps(
            [
                {
                    "name": "n",
                    "quote_excerpts": ["x" * (MAX_QUOTE_EXCERPT_LEN + 50)],
                }
            ]
        )
        out = parse_proposals_response(text)
        assert len(out) == 1
        assert len(out[0].quote_excerpts[0]) <= MAX_QUOTE_EXCERPT_LEN

    def test_excerpt_count_capped(self) -> None:
        text = json.dumps(
            [
                {
                    "name": "n",
                    "quote_excerpts": [f"e{i}" for i in range(MAX_QUOTE_EXCERPTS + 5)],
                }
            ]
        )
        out = parse_proposals_response(text)
        assert len(out[0].quote_excerpts) == MAX_QUOTE_EXCERPTS

    def test_excerpts_not_a_list(self) -> None:
        text = json.dumps([{"name": "n", "quote_excerpts": "not a list"}])
        out = parse_proposals_response(text)
        assert out[0].quote_excerpts == []

    def test_non_array_top_level(self) -> None:
        # Object instead of array → empty.
        text = json.dumps({"name": "n"})
        assert parse_proposals_response(text) == []

    def test_gerund_flag_set(self) -> None:
        text = json.dumps([{"name": "navigating change"}])
        out = parse_proposals_response(text)
        assert out[0].is_gerund is True


# --------------------------------------------------------------------------- #
# suggest_new_codes_for_span
# --------------------------------------------------------------------------- #


class TestSuggestNewCodesForSpan:
    def test_end_to_end(self) -> None:
        # No existing codes — generate_fn returns one proposal.
        embed_fn = _const_embed((1.0, 0.0))

        captured: dict[str, str] = {}

        def gen(prompt: str) -> str:
            captured["prompt"] = prompt
            return json.dumps(
                [
                    {
                        "name": "navigating change",
                        "definition": "Adapting to new circumstances",
                        "rationale": "Speaker reflects on a transition",
                        "confidence": 0.8,
                    }
                ]
            )

        s = suggest_new_codes_for_span(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w3",
            query_text="It changed everything for me.",
            codes=[],
            embed_fn=embed_fn,
            generate_fn=gen,
            embedding_model="bge-m3",
            generation_model="llama3.2:3b",
        )
        assert s.decision == NEW_CODE_DECISION_PENDING
        assert len(s.proposals) == 1
        assert s.proposals[0].name == "navigating change"
        assert s.proposals[0].is_gerund is True
        assert s.embedding_model == "bge-m3"
        assert s.generation_model == "llama3.2:3b"
        assert "It changed everything" in captured["prompt"]
        assert "GERUND" in captured["prompt"] or "gerund" in captured["prompt"].lower()
        # No existing codes → block is omitted.
        assert "Existing codes already in the codebook" not in captured["prompt"]

    def test_existing_codes_appear_in_prompt(self) -> None:
        # An existing code that's similar to the span: should appear in the
        # "do NOT duplicate" block.
        codes = [
            _make_code(
                _HEX_PROJECT,
                code_id="aa" * 6,
                name="resilience",
                definition="bouncing back",
            ),
        ]
        # Query → (1, 0); existing code corpus → (1, 0); proposal → (0, 1)
        embed_fn = _per_text_embed(
            {
                "It just kept going": (1.0, 0.0),
                "resilience. bouncing back": (1.0, 0.0),
                # The annotate pass will also embed the proposal text:
                "managing through": (0.0, 1.0),
                "managing through. d": (0.0, 1.0),
            },
            default=(0.0, 0.0),
        )

        captured: dict[str, str] = {}

        def gen(prompt: str) -> str:
            captured["prompt"] = prompt
            return json.dumps(
                [{"name": "managing through", "definition": "d"}]
            )

        s = suggest_new_codes_for_span(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="It just kept going",
            codes=codes,
            embed_fn=embed_fn,
            generate_fn=gen,
            near_duplicate_threshold=0.5,
        )
        assert "resilience" in captured["prompt"]
        assert "do NOT duplicate" in captured["prompt"] or "do not duplicate" in captured["prompt"].lower()
        # The proposal got an annotated nearest-existing match recorded.
        assert s.proposals[0].nearest_existing_code_id == "aa" * 6

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            suggest_new_codes_for_span(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="   ",
                codes=[],
                embed_fn=_const_embed((1.0,)),
                generate_fn=lambda _: "[]",
            )

    def test_huge_query_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            suggest_new_codes_for_span(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="x" * (MAX_QUERY_TEXT_LEN + 1),
                codes=[],
                embed_fn=_const_embed((1.0,)),
                generate_fn=lambda _: "[]",
            )

    def test_embed_fn_returns_nothing(self) -> None:
        with pytest.raises(ProjectValidationError):

            def empty(texts: Sequence[str]) -> Sequence[Sequence[float]]:
                return []

            suggest_new_codes_for_span(
                project_id=_HEX_PROJECT,
                source_id=_HEX_SOURCE,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                query_text="hi",
                codes=[],
                embed_fn=empty,
                generate_fn=lambda _: "[]",
            )

    def test_raw_response_truncated(self) -> None:
        embed_fn = _const_embed((1.0,))
        big = "x" * (MAX_RAW_LLM_RESPONSE_LEN + 100)
        s = suggest_new_codes_for_span(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hi",
            codes=[],
            embed_fn=embed_fn,
            generate_fn=lambda _: big,
        )
        # The raw response is truncated to MAX_RAW_LLM_RESPONSE_LEN; the
        # parse will fail (it's all 'x') and proposals will be empty.
        assert len(s.raw_llm_response) == MAX_RAW_LLM_RESPONSE_LEN
        assert s.proposals == []

    def test_canonicalises_query(self) -> None:
        # Whitespace-collapsed before storing.
        embed_fn = _const_embed((1.0,))
        s = suggest_new_codes_for_span(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="  hello   world  ",
            codes=[],
            embed_fn=embed_fn,
            generate_fn=lambda _: "[]",
        )
        assert s.query_text == "hello world"

    def test_caps_to_num_proposals(self) -> None:
        embed_fn = _const_embed((1.0,))
        # Model returns 5 proposals but we asked for 2.
        text = json.dumps(
            [{"name": f"name {i}"} for i in range(5)]
        )
        s = suggest_new_codes_for_span(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hi",
            codes=[],
            embed_fn=embed_fn,
            generate_fn=lambda _: text,
            num_proposals=2,
        )
        assert len(s.proposals) == 2


# --------------------------------------------------------------------------- #
# record_new_code_decision
# --------------------------------------------------------------------------- #


class TestRecordDecision:
    def _base(self) -> NewCodeSuggestion:
        return NewCodeSuggestion.new(
            project_id=_HEX_PROJECT,
            source_id=_HEX_SOURCE,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="x",
            proposals=[
                NewCodeProposal(name="navigating change"),
                NewCodeProposal(name="managing uncertainty"),
            ],
        )

    def test_accept(self) -> None:
        s = self._base()
        record_new_code_decision(
            s,
            decision=NEW_CODE_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
            accepted_proposal_index=0,
            created_code_id="0" * 12,
        )
        assert s.decision == NEW_CODE_DECISION_ACCEPTED
        assert s.accepted_proposal_index == 0
        assert s.created_code_id == "0" * 12
        assert s.decided_at != ""
        assert s.decided_by_coder_id == _HEX_CODER

    def test_modify(self) -> None:
        s = self._base()
        record_new_code_decision(
            s,
            decision=NEW_CODE_DECISION_MODIFIED,
            coder_id=_HEX_CODER,
            accepted_proposal_index=1,
            created_code_id="ff" * 6,
        )
        assert s.decision == NEW_CODE_DECISION_MODIFIED
        assert s.accepted_proposal_index == 1
        assert s.created_code_id == "ff" * 6

    def test_reject(self) -> None:
        s = self._base()
        record_new_code_decision(
            s,
            decision=NEW_CODE_DECISION_REJECTED,
            coder_id=_HEX_CODER,
            rejection_reason="too generic",
        )
        assert s.decision == NEW_CODE_DECISION_REJECTED
        assert s.accepted_proposal_index is None
        assert s.created_code_id is None
        assert s.rejection_reason == "too generic"

    def test_accept_without_created_code_allowed(self) -> None:
        # Caller may save the Code in a follow-up step; created_code_id
        # is optional even on accept.
        s = self._base()
        record_new_code_decision(
            s,
            decision=NEW_CODE_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
            accepted_proposal_index=0,
        )
        assert s.decision == NEW_CODE_DECISION_ACCEPTED
        assert s.created_code_id is None

    def test_accept_requires_proposal_index(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
            )

    def test_accept_invalid_index(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_proposal_index=99,
            )

    def test_accept_negative_index(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_proposal_index=-1,
            )

    def test_reject_forbids_proposal_index(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_REJECTED,
                coder_id=_HEX_CODER,
                accepted_proposal_index=0,
            )

    def test_invalid_decision(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision="bogus",
                coder_id=_HEX_CODER,
            )

    def test_invalid_coder_id(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_REJECTED,
                coder_id="not-hex",
            )

    def test_double_decision_rejected(self) -> None:
        s = self._base()
        record_new_code_decision(
            s,
            decision=NEW_CODE_DECISION_REJECTED,
            coder_id=_HEX_CODER,
        )
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_REJECTED,
                coder_id=_HEX_CODER,
            )

    def test_invalid_created_code_id(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_proposal_index=0,
                created_code_id="not-hex",
            )

    def test_pending_initial_decision(self) -> None:
        s = self._base()
        with pytest.raises(ProjectValidationError):
            record_new_code_decision(
                s,
                decision=NEW_CODE_DECISION_PENDING,
                coder_id=_HEX_CODER,
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def _suggestion(
        self,
        project_id: str,
        *,
        source_id: str = _HEX_SOURCE,
        suggestion_id: str | None = None,
    ) -> NewCodeSuggestion:
        return NewCodeSuggestion.new(
            project_id=project_id,
            source_id=source_id,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            query_text="hi",
            proposals=[NewCodeProposal(name="navigating change")],
            suggestion_id=suggestion_id,
        )

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        loaded = load_new_code_suggestion(tmp_path, proj.id, s.id)
        assert loaded.id == s.id
        assert loaded.proposals[0].name == "navigating change"

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        assert (
            project_dir(tmp_path, proj.id) / NEW_CODE_SUGGESTIONS_DIRNAME
        ).is_dir()

    def test_save_atomic_no_tmp_left(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        sd = new_code_suggestions_dir(tmp_path, proj.id)
        # No leftover .tmp file.
        leftovers = [f for f in sd.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        s = self._suggestion(_HEX_PROJECT)  # never saved this project
        with pytest.raises(FileNotFoundError):
            save_new_code_suggestion(tmp_path, s)

    def test_load_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_new_code_suggestion(tmp_path, proj.id, "0" * 12)

    def test_invalid_suggestion_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            new_code_suggestion_state_path(tmp_path, proj.id, "not-hex")

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_new_code_suggestions(tmp_path, proj.id) == []

    def test_list_filters_by_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = self._suggestion(proj.id, source_id=_HEX_SOURCE)
        s2 = self._suggestion(proj.id, source_id=_HEX_SOURCE_2)
        save_new_code_suggestion(tmp_path, s1)
        save_new_code_suggestion(tmp_path, s2)
        out = list_new_code_suggestions(
            tmp_path, proj.id, source_id=_HEX_SOURCE
        )
        assert [s.id for s in out] == [s1.id]

    def test_list_filters_by_decision(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = self._suggestion(proj.id)
        s2 = self._suggestion(proj.id)
        record_new_code_decision(
            s2,
            decision=NEW_CODE_DECISION_REJECTED,
            coder_id=_HEX_CODER,
        )
        save_new_code_suggestion(tmp_path, s1)
        save_new_code_suggestion(tmp_path, s2)
        out = list_new_code_suggestions(
            tmp_path, proj.id, decision=NEW_CODE_DECISION_REJECTED
        )
        assert [s.id for s in out] == [s2.id]

    def test_list_skips_corrupt(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sd = new_code_suggestions_dir(tmp_path, proj.id)
        sd.mkdir(parents=True, exist_ok=True)
        # Garbage file with a valid-looking name.
        (sd / ("0" * 12 + ".json")).write_text("not json")
        # And a real one.
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        out = list_new_code_suggestions(tmp_path, proj.id)
        assert [x.id for x in out] == [s.id]

    def test_list_skips_non_hex_filenames(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sd = new_code_suggestions_dir(tmp_path, proj.id)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "junk.json").write_text("{}")
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        out = list_new_code_suggestions(tmp_path, proj.id)
        assert [x.id for x in out] == [s.id]

    def test_invalid_decision_filter_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_new_code_suggestions(tmp_path, proj.id, decision="bogus")

    def test_invalid_source_filter_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_new_code_suggestions(tmp_path, proj.id, source_id="not-hex")

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = self._suggestion(proj.id)
        save_new_code_suggestion(tmp_path, s)
        assert delete_new_code_suggestion(tmp_path, proj.id, s.id) is True
        assert delete_new_code_suggestion(tmp_path, proj.id, s.id) is False

    def test_id_factory(self) -> None:
        id1 = new_new_code_suggestion_id()
        id2 = new_new_code_suggestion_id()
        assert NEW_CODE_SUGGESTION_ID_RE.match(id1)
        assert NEW_CODE_SUGGESTION_ID_RE.match(id2)
        assert id1 != id2
