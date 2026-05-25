"""Tests for scribe.memo_drafts (F8.8 — AI memo-draft action on a code).

Per PLANNING.md F8.8:

  > Memo-draft action on a code (LLM seeds with the code's exemplars;
  > researcher rewrites).

Covers the engine + persistence + decision lifecycle + promotion
helpers. Mirrors the F8.3 / F8.4 test layout so the AI suggester
trinity (existing-codes / new-codes / memo-drafts) reads the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scribe.applications import Application, make_word_id
from scribe.codes import Code
from scribe.memo_drafts import (
    DEFAULT_BACK_LINK_ROLE,
    DEFAULT_INCLUDE_APPLICATIONS,
    DEFAULT_MAX_SEED_SNIPPETS,
    DEFAULT_MEMO_TYPE,
    MAX_BODY_LEN,
    MAX_NOTES_LEN,
    MAX_RATIONALE_LEN,
    MAX_RAW_LLM_RESPONSE_LEN,
    MAX_REJECTION_REASON_LEN,
    MAX_SEED_SNIPPETS_PERSISTED,
    MAX_SEED_SNIPPET_LEN,
    MAX_TITLE_LEN,
    MEMO_DRAFT_DECISION_ACCEPTED,
    MEMO_DRAFT_DECISION_MODIFIED,
    MEMO_DRAFT_DECISION_PENDING,
    MEMO_DRAFT_DECISION_REJECTED,
    MEMO_DRAFT_ID_RE,
    MEMO_DRAFT_PROVENANCE_SOURCE,
    MEMO_DRAFTS_DIRNAME,
    MemoDraft,
    ParsedDraft,
    SEED_KIND_APPLICATION,
    SEED_KIND_EXEMPLAR,
    SeedSnippet,
    collect_seed_snippets,
    delete_memo_draft,
    draft_memo_for_code,
    list_memo_drafts,
    load_memo_draft,
    make_memo_draft_prompt,
    memo_draft_state_path,
    memo_drafts_dir,
    new_memo_draft_id,
    parse_memo_draft_response,
    promote_memo_draft_to_memo,
    record_memo_draft_decision,
    save_memo_draft,
)
from scribe.memos import MemoLink, list_memos, load_memo
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)


# Sentinel hex ids — 12-char hex matches every entity-id regex.
_HEX_PROJECT = "a" * 12
_HEX_PROJECT_2 = "b" * 12
_HEX_CODE = "c" * 12
_HEX_CODE_2 = "d" * 12
_HEX_SOURCE = "e" * 12
_HEX_SOURCE_2 = "f" * 12
_HEX_CODER = "0" * 11 + "a"
_HEX_APP = "9" * 12
_HEX_APP_2 = "8" * 12
_HEX_MEMO = "1" * 12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_code(
    project_id: str,
    *,
    code_id: str = _HEX_CODE,
    name: str = "managing identity",
    definition: str = "How participants navigate identity tensions.",
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    exemplars: list[str] | None = None,
    theoretical_memo: str = "",
    status: str = "active",
) -> Code:
    return Code.new(
        project_id=project_id,
        code_id=code_id,
        name=name,
        definition=definition,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        exemplars=list(exemplars or []),
        theoretical_memo=theoretical_memo,
        status=status,
    )


_HEX_CODE_VERSION = "2" * 12


def _make_application(
    project_id: str,
    *,
    application_id: str = _HEX_APP,
    code_id: str = _HEX_CODE,
    source_id: str = _HEX_SOURCE,
    start_word: str = "s0w0",
    end_word: str = "s0w1",
    coder_id: str = _HEX_CODER,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        anchor_start_word_id=start_word,
        anchor_end_word_id=end_word,
        coder_id=coder_id,
        application_id=application_id,
        definition_version_id_at_apply=_HEX_CODE_VERSION,
    )


def _make_segments(words: list[list[str]]) -> list[dict[str, Any]]:
    """Build a minimal segments fixture with the given word grid."""
    out: list[dict[str, Any]] = []
    for ws in words:
        out.append({
            "start": 0.0,
            "end": 1.0,
            "text": " ".join(ws),
            "words": [
                {"text": w, "start": float(i), "end": float(i) + 0.5}
                for i, w in enumerate(ws)
            ],
        })
    return out


def _stub_generate(response: str):
    """Return a generate_fn that always replies with ``response``."""
    captured: dict[str, Any] = {"prompts": []}

    def fn(prompt: str) -> str:
        captured["prompts"].append(prompt)
        return response

    fn.captured = captured  # type: ignore[attr-defined]
    return fn


# --------------------------------------------------------------------------- #
# new_memo_draft_id
# --------------------------------------------------------------------------- #


class TestNewMemoDraftId:
    def test_shape(self) -> None:
        assert MEMO_DRAFT_ID_RE.match(new_memo_draft_id())

    def test_uniqueness(self) -> None:
        ids = {new_memo_draft_id() for _ in range(200)}
        assert len(ids) == 200


# --------------------------------------------------------------------------- #
# SeedSnippet
# --------------------------------------------------------------------------- #


class TestSeedSnippet:
    def test_round_trip_exemplar(self) -> None:
        s = SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="0", text="hello")
        s.validate()
        assert SeedSnippet.from_dict(s.to_dict()) == s

    def test_round_trip_application(self) -> None:
        s = SeedSnippet(kind=SEED_KIND_APPLICATION, ref=_HEX_APP, text="hello")
        s.validate()
        assert SeedSnippet.from_dict(s.to_dict()) == s

    def test_invalid_kind(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(kind="other", ref="0", text="hello").validate()

    def test_empty_text(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="0", text="").validate()

    def test_too_long_text(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(
                kind=SEED_KIND_EXEMPLAR,
                ref="0",
                text="x" * (MAX_SEED_SNIPPET_LEN + 1),
            ).validate()

    def test_application_ref_must_be_hex(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(
                kind=SEED_KIND_APPLICATION, ref="not-hex", text="hello"
            ).validate()

    def test_exemplar_ref_must_be_int(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(
                kind=SEED_KIND_EXEMPLAR, ref="abc", text="hello"
            ).validate()

    def test_exemplar_ref_negative_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(
                kind=SEED_KIND_EXEMPLAR, ref="-1", text="hello"
            ).validate()

    def test_empty_ref(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet(
                kind=SEED_KIND_EXEMPLAR, ref="", text="hello"
            ).validate()

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            SeedSnippet.from_dict("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# MemoDraft entity
# --------------------------------------------------------------------------- #


class TestMemoDraftConstruction:
    def test_minimal_new(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        assert MEMO_DRAFT_ID_RE.match(d.id)
        assert d.project_id == _HEX_PROJECT
        assert d.code_id == _HEX_CODE
        assert d.memo_type == DEFAULT_MEMO_TYPE
        assert d.decision == MEMO_DRAFT_DECISION_PENDING
        assert d.created_at == d.modified_at
        assert d.created_at  # non-empty
        assert d.seed_snippets == []
        assert d.accepted_memo_id is None

    def test_new_passes_explicit_id(self) -> None:
        explicit = "1" * 12
        d = MemoDraft.new(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE,
            draft_id=explicit,
        )
        assert d.id == explicit

    def test_new_validates_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(project_id="bad", code_id=_HEX_CODE)

    def test_new_validates_code_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(project_id=_HEX_PROJECT, code_id="bad")

    def test_new_validates_memo_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(
                project_id=_HEX_PROJECT,
                code_id=_HEX_CODE,
                memo_type="not-a-type",
            )

    def test_new_seeds_dict_inputs(self) -> None:
        d = MemoDraft.new(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE,
            seed_snippets=[
                {"kind": SEED_KIND_EXEMPLAR, "ref": "0", "text": "hello"},
                SeedSnippet(
                    kind=SEED_KIND_APPLICATION, ref=_HEX_APP, text="world"
                ),
            ],
        )
        assert len(d.seed_snippets) == 2
        assert isinstance(d.seed_snippets[0], SeedSnippet)


class TestMemoDraftValidate:
    def test_terminal_decision_requires_decided_at(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.decision = MEMO_DRAFT_DECISION_ACCEPTED
        d.decided_by_coder_id = _HEX_CODER
        d.accepted_memo_id = _HEX_MEMO
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_terminal_decision_requires_coder(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.decision = MEMO_DRAFT_DECISION_ACCEPTED
        d.decided_at = "2026-01-01T00:00:00Z"
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_terminal_decision_rejects_bad_coder(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.decision = MEMO_DRAFT_DECISION_ACCEPTED
        d.decided_at = "2026-01-01T00:00:00Z"
        d.decided_by_coder_id = "bad"
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_rejected_forbids_accepted_memo_id(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.decision = MEMO_DRAFT_DECISION_REJECTED
        d.decided_at = "2026-01-01T00:00:00Z"
        d.decided_by_coder_id = _HEX_CODER
        d.accepted_memo_id = _HEX_MEMO
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_accepted_memo_id_must_be_hex(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.accepted_memo_id = "bad-not-hex"
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_title_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(
                project_id=_HEX_PROJECT,
                code_id=_HEX_CODE,
                title="x" * (MAX_TITLE_LEN + 1),
            )

    def test_body_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(
                project_id=_HEX_PROJECT,
                code_id=_HEX_CODE,
                body="x" * (MAX_BODY_LEN + 1),
            )

    def test_rationale_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(
                project_id=_HEX_PROJECT,
                code_id=_HEX_CODE,
                rationale="x" * (MAX_RATIONALE_LEN + 1),
            )

    def test_too_many_seed_snippets(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        # bypass collect_seed_snippets cap by stuffing the list directly
        d.seed_snippets = [
            SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref=str(i), text="t")
            for i in range(MAX_SEED_SNIPPETS_PERSISTED + 1)
        ]
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_title_trimmed_in_place(self) -> None:
        d = MemoDraft.new(
            project_id=_HEX_PROJECT, code_id=_HEX_CODE, title="  hi  "
        )
        assert d.title == "hi"

    def test_generation_model_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.new(
                project_id=_HEX_PROJECT,
                code_id=_HEX_CODE,
                generation_model="x" * 257,
            )

    def test_decision_must_be_known(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.decision = "not-a-decision"
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_rejection_reason_too_long(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.rejection_reason = "x" * (MAX_REJECTION_REASON_LEN + 1)
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_notes_too_long(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.notes = "x" * (MAX_NOTES_LEN + 1)
        with pytest.raises(ProjectValidationError):
            d.validate()

    def test_raw_response_too_long(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.raw_llm_response = "x" * (MAX_RAW_LLM_RESPONSE_LEN + 1)
        with pytest.raises(ProjectValidationError):
            d.validate()


class TestMemoDraftRoundTrip:
    def test_full_round_trip(self) -> None:
        d = MemoDraft.new(
            project_id=_HEX_PROJECT,
            code_id=_HEX_CODE,
            memo_type="theoretical",
            title="A draft",
            body="Lorem ipsum.",
            rationale="Because of the seeds.",
            seed_snippets=[
                SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="0", text="hello"),
            ],
            generation_model="qwen-32b",
            prompt="prompt text",
            raw_llm_response="raw text",
            notes="notes",
        )
        clone = MemoDraft.from_dict(d.to_dict())
        assert clone == d

    def test_from_dict_missing_required(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.from_dict({"project_id": _HEX_PROJECT, "code_id": _HEX_CODE})

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_rejects_bad_seed_snippets_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoDraft.from_dict({
                "id": "a" * 12,
                "project_id": _HEX_PROJECT,
                "code_id": _HEX_CODE,
                "seed_snippets": "not-a-list",
            })


class TestMemoDraftApplyUpdate:
    def test_known_keys(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        ts0 = d.modified_at
        d.apply_update({
            "notes": "updated",
            "title": "  trimmed  ",
            "body": "new body",
        })
        assert d.notes == "updated"
        assert d.title == "trimmed"
        assert d.body == "new body"
        assert d.modified_at >= ts0

    def test_accepted_memo_id_set_and_clear(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        d.apply_update({"accepted_memo_id": _HEX_MEMO})
        assert d.accepted_memo_id == _HEX_MEMO
        d.apply_update({"accepted_memo_id": None})
        assert d.accepted_memo_id is None

    def test_rejects_unknown_key(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            d.apply_update({"id": "0" * 12})

    def test_rejects_non_mapping(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            d.apply_update("nope")  # type: ignore[arg-type]

    def test_apply_update_revalidates(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            d.apply_update({"title": "x" * (MAX_TITLE_LEN + 1)})


# --------------------------------------------------------------------------- #
# collect_seed_snippets
# --------------------------------------------------------------------------- #


class TestCollectSeedSnippets:
    def test_exemplars_only_no_apps(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=["alpha", "beta"])
        out = collect_seed_snippets(code=c)
        assert len(out) == 2
        assert out[0].kind == SEED_KIND_EXEMPLAR
        assert out[0].ref == "0"
        assert out[0].text == "alpha"
        assert out[1].ref == "1"
        assert out[1].text == "beta"

    def test_drops_empty_exemplars(self) -> None:
        # Code.validate strips empties at the entity layer, so by the
        # time we collect snippets there's only one exemplar to start
        # from. We index against the post-validation list — same shape
        # the rest of the F-feature stack consumes.
        c = _make_code(_HEX_PROJECT, exemplars=["", "  ", "real"])
        assert c.exemplars == ["real"]
        out = collect_seed_snippets(code=c)
        assert len(out) == 1
        assert out[0].ref == "0"
        assert out[0].text == "real"

    def test_dedup_by_text(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=["alpha", " alpha ", "ALPHA"])
        out = collect_seed_snippets(code=c)
        # canonical_text lowercases? No — let's check what canonical_text does.
        # canonical_text only canonicalises whitespace; case is preserved.
        # So "alpha" and "ALPHA" are distinct. Just check the dedup of " alpha " vs "alpha".
        assert any(s.text == "alpha" for s in out)
        # "ALPHA" survives because canonical_text doesn't lowercase.
        # Confirm we have 2 entries (alpha, ALPHA).
        assert len(out) == 2
        assert {s.text for s in out} == {"alpha", "ALPHA"}

    def test_includes_application_text(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=["the exemplar"])
        app = _make_application(
            _HEX_PROJECT,
            start_word="s0w0",
            end_word="s0w1",
        )
        segs = _make_segments([["the", "applied", "text"]])
        out = collect_seed_snippets(
            code=c,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert len(out) == 2
        assert out[0].kind == SEED_KIND_EXEMPLAR
        assert out[1].kind == SEED_KIND_APPLICATION
        assert out[1].ref == app.id
        assert out[1].text == "the applied"

    def test_skips_other_codes_applications(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=[])
        # An application for a different code:
        app_other = _make_application(
            _HEX_PROJECT,
            application_id=_HEX_APP,
            code_id=_HEX_CODE_2,
        )
        segs = _make_segments([["irrelevant", "words"]])
        out = collect_seed_snippets(
            code=c,
            applications=[app_other],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert out == []

    def test_skips_when_segments_missing(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=[])
        app = _make_application(_HEX_PROJECT)
        out = collect_seed_snippets(
            code=c,
            applications=[app],
            segments_by_source={},
        )
        assert out == []

    def test_skips_apps_when_include_applications_false(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=["e"])
        app = _make_application(_HEX_PROJECT)
        segs = _make_segments([["the", "applied"]])
        out = collect_seed_snippets(
            code=c,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
            include_applications=False,
        )
        assert len(out) == 1
        assert out[0].kind == SEED_KIND_EXEMPLAR

    def test_dedups_exemplar_vs_application_text(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=["the applied"])
        app = _make_application(
            _HEX_PROJECT, start_word="s0w0", end_word="s0w1"
        )
        segs = _make_segments([["the", "applied"]])
        out = collect_seed_snippets(
            code=c,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        # Exemplar and application both produce "the applied"; second is deduped.
        assert len(out) == 1
        assert out[0].kind == SEED_KIND_EXEMPLAR

    def test_caps_at_max_snippets(self) -> None:
        c = _make_code(
            _HEX_PROJECT, exemplars=[f"exemplar{i}" for i in range(20)]
        )
        out = collect_seed_snippets(code=c, max_snippets=5)
        assert len(out) == 5

    def test_caps_at_persisted_max(self) -> None:
        # Code's MAX_EXEMPLARS happens to equal MAX_SEED_SNIPPETS_PERSISTED,
        # so the cap test asserts the helper clamps even when the caller
        # supplies a big max_snippets. We exercise it at the hard cap.
        from scribe.codes import MAX_EXEMPLARS as CODE_MAX_EXEMPLARS

        c = _make_code(
            _HEX_PROJECT,
            exemplars=[f"exemplar{i}" for i in range(CODE_MAX_EXEMPLARS)],
        )
        out = collect_seed_snippets(
            code=c, max_snippets=MAX_SEED_SNIPPETS_PERSISTED + 100,
        )
        # The helper must not exceed MAX_SEED_SNIPPETS_PERSISTED regardless
        # of caller request.
        assert len(out) <= MAX_SEED_SNIPPETS_PERSISTED
        assert len(out) == min(CODE_MAX_EXEMPLARS, MAX_SEED_SNIPPETS_PERSISTED)

    def test_rejects_non_code(self) -> None:
        with pytest.raises(TypeError):
            collect_seed_snippets(code="not-a-code")  # type: ignore[arg-type]

    def test_orphan_application_skipped(self) -> None:
        c = _make_code(_HEX_PROJECT, exemplars=[])
        # Application that anchors past the end of the segment.
        app = _make_application(
            _HEX_PROJECT,
            start_word=make_word_id(0, 99),
            end_word=make_word_id(0, 99),
        )
        segs = _make_segments([["only", "two"]])
        out = collect_seed_snippets(
            code=c,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert out == []


# --------------------------------------------------------------------------- #
# make_memo_draft_prompt
# --------------------------------------------------------------------------- #


class TestMakeMemoDraftPrompt:
    def test_renders_code_name_and_definition(self) -> None:
        c = _make_code(_HEX_PROJECT)
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "managing identity" in prompt
        assert "How participants navigate identity tensions." in prompt

    def test_lists_seed_snippets(self) -> None:
        c = _make_code(_HEX_PROJECT)
        snippets = [
            SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="0", text="quote one"),
            SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="1", text="quote two"),
        ]
        prompt = make_memo_draft_prompt(code=c, seed_snippets=snippets)
        assert "1." in prompt
        assert "quote one" in prompt
        assert "2." in prompt
        assert "quote two" in prompt

    def test_omits_inclusion_when_empty(self) -> None:
        c = _make_code(_HEX_PROJECT, inclusion_criteria="")
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "Inclusion criteria" not in prompt

    def test_includes_inclusion_when_set(self) -> None:
        c = _make_code(_HEX_PROJECT, inclusion_criteria="When P talks about X.")
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "Inclusion criteria" in prompt
        assert "When P talks about X." in prompt

    def test_includes_existing_memo(self) -> None:
        c = _make_code(_HEX_PROJECT, theoretical_memo="prior thoughts")
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "Existing memo" in prompt
        assert "prior thoughts" in prompt

    def test_excludes_existing_memo_when_flag_off(self) -> None:
        c = _make_code(_HEX_PROJECT, theoretical_memo="prior thoughts")
        prompt = make_memo_draft_prompt(
            code=c, seed_snippets=[], include_existing_memo=False
        )
        assert "Existing memo" not in prompt

    def test_no_seed_message(self) -> None:
        c = _make_code(_HEX_PROJECT)
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "no seed material" in prompt

    def test_unnamed_code(self) -> None:
        # Codes require a name; we still want the prompt to render
        # something sensible. Force-build one via direct dataclass.
        c = _make_code(_HEX_PROJECT, name="x", definition="")
        c.name = "   "  # post-validate; canonical_text will strip
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "(unnamed)" in prompt
        assert "(no definition)" in prompt

    def test_rejects_non_code(self) -> None:
        with pytest.raises(TypeError):
            make_memo_draft_prompt(
                code="not-a-code",  # type: ignore[arg-type]
                seed_snippets=[],
            )

    def test_strict_json_directive_present(self) -> None:
        c = _make_code(_HEX_PROJECT)
        prompt = make_memo_draft_prompt(code=c, seed_snippets=[])
        assert "strict JSON" in prompt
        assert '"title"' in prompt
        assert '"body"' in prompt
        assert '"rationale"' in prompt


# --------------------------------------------------------------------------- #
# parse_memo_draft_response
# --------------------------------------------------------------------------- #


class TestParseMemoDraftResponse:
    def test_plain_json_object(self) -> None:
        raw = json.dumps({
            "title": "T",
            "body": "B",
            "rationale": "R",
        })
        p = parse_memo_draft_response(raw)
        assert p.title == "T"
        assert p.body == "B"
        assert p.rationale == "R"

    def test_fenced_json(self) -> None:
        raw = "```json\n" + json.dumps({"title": "T", "body": "B"}) + "\n```"
        p = parse_memo_draft_response(raw)
        assert p.title == "T"
        assert p.body == "B"
        assert p.rationale == ""

    def test_prose_prefixed_json(self) -> None:
        raw = (
            "Sure, here is the draft:\n"
            + json.dumps({"title": "Hello", "body": "World"})
            + "\nLet me know if you need changes."
        )
        p = parse_memo_draft_response(raw)
        assert p.title == "Hello"
        assert p.body == "World"

    def test_empty_response(self) -> None:
        p = parse_memo_draft_response("")
        assert p == ParsedDraft(title="", body="", rationale="")

    def test_whitespace_only(self) -> None:
        p = parse_memo_draft_response("   \n  ")
        assert p == ParsedDraft(title="", body="", rationale="")

    def test_garbage_falls_back_to_body(self) -> None:
        raw = "Not JSON at all, just rambling prose."
        p = parse_memo_draft_response(raw)
        assert p.body == raw
        assert p.title == ""
        assert p.rationale == ""

    def test_truncates_long_body(self) -> None:
        long_body = "x" * (MAX_BODY_LEN + 100)
        raw = json.dumps({"title": "t", "body": long_body})
        p = parse_memo_draft_response(raw)
        assert len(p.body) == MAX_BODY_LEN

    def test_truncates_long_title(self) -> None:
        long_title = "x" * (MAX_TITLE_LEN + 100)
        raw = json.dumps({"title": long_title, "body": "b"})
        p = parse_memo_draft_response(raw)
        assert len(p.title) == MAX_TITLE_LEN

    def test_truncates_long_rationale(self) -> None:
        long_r = "x" * (MAX_RATIONALE_LEN + 100)
        raw = json.dumps({"title": "t", "body": "b", "rationale": long_r})
        p = parse_memo_draft_response(raw)
        assert len(p.rationale) == MAX_RATIONALE_LEN

    def test_truncates_long_garbage_fallback(self) -> None:
        # Hard-fallback path should also truncate.
        raw = "Not JSON: " + ("y" * (MAX_BODY_LEN + 50))
        p = parse_memo_draft_response(raw)
        assert len(p.body) == MAX_BODY_LEN

    def test_array_response_extracts_first_object(self) -> None:
        # An array isn't a Mapping at the top, but the {} extraction
        # finds the inner object and parses that (best-effort recovery).
        raw = json.dumps([{"title": "T", "body": "B"}])
        p = parse_memo_draft_response(raw)
        assert p.title == "T"
        assert p.body == "B"

    def test_array_only_response_falls_back(self) -> None:
        # An array literal with no inner object falls all the way back.
        raw = json.dumps([1, 2, 3])
        p = parse_memo_draft_response(raw)
        assert p.body == raw  # body holds the raw text
        assert p.title == ""

    def test_null_fields(self) -> None:
        raw = json.dumps({"title": None, "body": None, "rationale": None})
        p = parse_memo_draft_response(raw)
        assert p.title == ""
        assert p.body == ""
        assert p.rationale == ""

    def test_non_string_fields_coerced(self) -> None:
        raw = json.dumps({"title": 123, "body": 456})
        p = parse_memo_draft_response(raw)
        assert p.title == "123"
        assert p.body == "456"


# --------------------------------------------------------------------------- #
# draft_memo_for_code
# --------------------------------------------------------------------------- #


class TestDraftMemoForCode:
    def test_end_to_end_minimal(self) -> None:
        code = _make_code(_HEX_PROJECT, exemplars=["alpha"])
        gen = _stub_generate(json.dumps({
            "title": "On managing identity",
            "body": "I notice...",
            "rationale": "alpha is illustrative",
        }))
        draft = draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
        )
        assert draft.decision == MEMO_DRAFT_DECISION_PENDING
        assert draft.title == "On managing identity"
        assert draft.body == "I notice..."
        assert draft.rationale == "alpha is illustrative"
        assert len(draft.seed_snippets) == 1
        assert draft.seed_snippets[0].text == "alpha"
        assert draft.code_id == code.id
        assert draft.prompt  # prompt was recorded
        assert draft.raw_llm_response  # raw response recorded

    def test_passes_generation_model_through(self) -> None:
        code = _make_code(_HEX_PROJECT)
        gen = _stub_generate("{}")
        draft = draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
            generation_model="qwen-32b",
        )
        assert draft.generation_model == "qwen-32b"

    def test_validates_memo_type(self) -> None:
        code = _make_code(_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            draft_memo_for_code(
                project_id=_HEX_PROJECT,
                code=code,
                generate_fn=_stub_generate("{}"),
                memo_type="not-a-type",
            )

    def test_rejects_mismatched_project(self) -> None:
        code = _make_code(_HEX_PROJECT_2)
        with pytest.raises(ProjectValidationError):
            draft_memo_for_code(
                project_id=_HEX_PROJECT,
                code=code,
                generate_fn=_stub_generate("{}"),
            )

    def test_rejects_non_code(self) -> None:
        with pytest.raises(TypeError):
            draft_memo_for_code(
                project_id=_HEX_PROJECT,
                code="not-a-code",  # type: ignore[arg-type]
                generate_fn=_stub_generate("{}"),
            )

    def test_truncates_long_raw_response(self) -> None:
        code = _make_code(_HEX_PROJECT)
        big = "x" * (MAX_RAW_LLM_RESPONSE_LEN + 100)
        gen = _stub_generate(big)
        draft = draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
        )
        assert len(draft.raw_llm_response) == MAX_RAW_LLM_RESPONSE_LEN

    def test_sends_seed_in_prompt(self) -> None:
        code = _make_code(_HEX_PROJECT, exemplars=["the seed quote"])
        gen = _stub_generate("{}")
        draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
        )
        assert any("the seed quote" in p for p in gen.captured["prompts"])

    def test_handles_empty_response_body(self) -> None:
        code = _make_code(_HEX_PROJECT)
        gen = _stub_generate(json.dumps({"title": "", "body": "", "rationale": ""}))
        draft = draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
        )
        # Empty body permitted — audit trail records the call.
        assert draft.body == ""
        assert draft.title == ""

    def test_includes_application_seeds(self) -> None:
        code = _make_code(_HEX_PROJECT, exemplars=[])
        app = _make_application(
            _HEX_PROJECT, start_word="s0w0", end_word="s0w1"
        )
        segs = _make_segments([["the", "applied"]])
        gen = _stub_generate("{}")
        draft = draft_memo_for_code(
            project_id=_HEX_PROJECT,
            code=code,
            generate_fn=gen,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert len(draft.seed_snippets) == 1
        assert draft.seed_snippets[0].kind == SEED_KIND_APPLICATION
        # And the prompt actually quotes the application text.
        assert any("the applied" in p for p in gen.captured["prompts"])


# --------------------------------------------------------------------------- #
# record_memo_draft_decision
# --------------------------------------------------------------------------- #


class TestRecordDecision:
    def test_accept_with_memo_id(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
            accepted_memo_id=_HEX_MEMO,
        )
        assert d.decision == MEMO_DRAFT_DECISION_ACCEPTED
        assert d.accepted_memo_id == _HEX_MEMO
        assert d.decided_at
        assert d.decided_by_coder_id == _HEX_CODER

    def test_accept_without_memo_id(self) -> None:
        # Caller may attach the memo id via apply_update later.
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
        )
        assert d.decision == MEMO_DRAFT_DECISION_ACCEPTED
        assert d.accepted_memo_id is None

    def test_modify(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_MODIFIED,
            coder_id=_HEX_CODER,
            accepted_memo_id=_HEX_MEMO,
        )
        assert d.decision == MEMO_DRAFT_DECISION_MODIFIED
        assert d.accepted_memo_id == _HEX_MEMO

    def test_reject_with_reason(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_REJECTED,
            coder_id=_HEX_CODER,
            rejection_reason="off-topic",
        )
        assert d.decision == MEMO_DRAFT_DECISION_REJECTED
        assert d.rejection_reason == "off-topic"
        assert d.accepted_memo_id is None

    def test_reject_forbids_accepted_memo_id(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_REJECTED,
                coder_id=_HEX_CODER,
                accepted_memo_id=_HEX_MEMO,
            )

    def test_double_decision_blocked(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_ACCEPTED,
            coder_id=_HEX_CODER,
            accepted_memo_id=_HEX_MEMO,
        )
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_REJECTED,
                coder_id=_HEX_CODER,
            )

    def test_pending_decision_invalid(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_PENDING,
                coder_id=_HEX_CODER,
            )

    def test_bad_coder_id(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_ACCEPTED,
                coder_id="bad",
            )

    def test_bad_accepted_memo_id_shape(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_ACCEPTED,
                coder_id=_HEX_CODER,
                accepted_memo_id="bad",
            )

    def test_long_rejection_reason_rejected(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_REJECTED,
                coder_id=_HEX_CODER,
                rejection_reason="x" * (MAX_REJECTION_REASON_LEN + 1),
            )

    def test_notes_set_via_decision(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d,
            decision=MEMO_DRAFT_DECISION_REJECTED,
            coder_id=_HEX_CODER,
            notes="see related code",
        )
        assert d.notes == "see related code"

    def test_long_notes_rejected(self) -> None:
        d = MemoDraft.new(project_id=_HEX_PROJECT, code_id=_HEX_CODE)
        with pytest.raises(ProjectValidationError):
            record_memo_draft_decision(
                d,
                decision=MEMO_DRAFT_DECISION_REJECTED,
                coder_id=_HEX_CODER,
                notes="x" * (MAX_NOTES_LEN + 1),
            )


# --------------------------------------------------------------------------- #
# promote_memo_draft_to_memo
# --------------------------------------------------------------------------- #


class TestPromoteMemoDraftToMemo:
    def test_basic_promotion(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(
            project_id=project.id,
            code_id=code.id,
            title="A title",
            body="A body",
            rationale="why",
        )
        save_memo_draft(tmp_path, draft)
        memo = promote_memo_draft_to_memo(
            tmp_path,
            draft,
            coder_id=_HEX_CODER,
        )
        assert memo.project_id == project.id
        assert memo.title == "A title"
        assert memo.body == "A body"
        assert memo.type == DEFAULT_MEMO_TYPE
        assert memo.provenance["source"] == MEMO_DRAFT_PROVENANCE_SOURCE
        assert memo.provenance["draft_id"] == draft.id
        # Back-link to the source code is present.
        assert any(
            link.target_type == "code"
            and link.target_id == code.id
            and link.role == DEFAULT_BACK_LINK_ROLE
            for link in memo.links
        )
        # Draft was decision-stamped and saved.
        assert draft.decision == MEMO_DRAFT_DECISION_ACCEPTED
        assert draft.accepted_memo_id == memo.id

        # Memo persisted to disk.
        loaded = load_memo(tmp_path, project.id, memo.id)
        assert loaded.id == memo.id

    def test_promotion_with_modified_decision(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(
            project_id=project.id,
            code_id=code.id,
            body="original body",
        )
        save_memo_draft(tmp_path, draft)
        memo = promote_memo_draft_to_memo(
            tmp_path,
            draft,
            coder_id=_HEX_CODER,
            decision=MEMO_DRAFT_DECISION_MODIFIED,
            body="my own body",
            title="my own title",
        )
        assert memo.body == "my own body"
        assert memo.title == "my own title"
        assert draft.decision == MEMO_DRAFT_DECISION_MODIFIED

    def test_promotion_rejects_non_promote_decisions(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id)
        save_memo_draft(tmp_path, draft)
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
                decision=MEMO_DRAFT_DECISION_REJECTED,
            )
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
                decision=MEMO_DRAFT_DECISION_PENDING,
            )

    def test_promotion_blocked_after_terminal_decision(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id)
        save_memo_draft(tmp_path, draft)
        record_memo_draft_decision(
            draft,
            decision=MEMO_DRAFT_DECISION_REJECTED,
            coder_id=_HEX_CODER,
        )
        save_memo_draft(tmp_path, draft)
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
            )

    def test_extra_provenance_blocked_for_reserved_keys(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id, body="b")
        save_memo_draft(tmp_path, draft)
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
                extra_provenance={"source": "human"},
            )
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
                extra_provenance={"draft_id": "x" * 12},
            )

    def test_extra_provenance_attached(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id, body="b")
        save_memo_draft(tmp_path, draft)
        memo = promote_memo_draft_to_memo(
            tmp_path,
            draft,
            coder_id=_HEX_CODER,
            extra_provenance={"model id": "qwen-32b"},
        )
        assert memo.provenance["model id"] == "qwen-32b"
        # Reserved keys are still present.
        assert memo.provenance["source"] == MEMO_DRAFT_PROVENANCE_SOURCE
        assert memo.provenance["draft_id"] == draft.id

    def test_extra_links_appended(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id, body="b")
        save_memo_draft(tmp_path, draft)
        memo = promote_memo_draft_to_memo(
            tmp_path,
            draft,
            coder_id=_HEX_CODER,
            extra_links=[
                MemoLink(target_type="source", target_id=_HEX_SOURCE),
                {"target_type": "code", "target_id": _HEX_CODE_2},
            ],
        )
        assert any(
            link.target_type == "source" and link.target_id == _HEX_SOURCE
            for link in memo.links
        )
        assert any(
            link.target_type == "code" and link.target_id == _HEX_CODE_2
            for link in memo.links
        )

    def test_extra_links_invalid_type(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id, body="b")
        save_memo_draft(tmp_path, draft)
        with pytest.raises(ProjectValidationError):
            promote_memo_draft_to_memo(
                tmp_path,
                draft,
                coder_id=_HEX_CODER,
                extra_links=["not-a-link"],  # type: ignore[list-item]
            )

    def test_promoted_memo_appears_in_list(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        code = _make_code(project.id)
        draft = MemoDraft.new(project_id=project.id, code_id=code.id, body="hi")
        save_memo_draft(tmp_path, draft)
        promote_memo_draft_to_memo(tmp_path, draft, coder_id=_HEX_CODER)
        memos = list_memos(tmp_path, project.id)
        assert len(memos) == 1
        assert memos[0].body == "hi"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = MemoDraft.new(
            project_id=project.id,
            code_id=_HEX_CODE,
            body="hello",
            seed_snippets=[
                SeedSnippet(kind=SEED_KIND_EXEMPLAR, ref="0", text="t"),
            ],
        )
        save_memo_draft(tmp_path, d)
        loaded = load_memo_draft(tmp_path, project.id, d.id)
        assert loaded == d

    def test_save_atomic_no_tmp(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        save_memo_draft(tmp_path, d)
        files = list(memo_drafts_dir(tmp_path, project.id).iterdir())
        names = [f.name for f in files]
        assert any(n.endswith(".json") and not n.endswith(".tmp") for n in names)
        assert not any(n.endswith(".json.tmp") for n in names)

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            memo_draft_state_path(tmp_path, "a" * 12, "bad-id")

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        # No project saved → directory missing → save raises.
        d = MemoDraft.new(project_id="b" * 12, code_id=_HEX_CODE)
        with pytest.raises(FileNotFoundError):
            save_memo_draft(tmp_path, d)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_memo_draft(tmp_path, project.id, "1" * 12)

    def test_list_filters_by_code_id(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d1 = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        d2 = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE_2)
        save_memo_draft(tmp_path, d1)
        save_memo_draft(tmp_path, d2)
        out = list_memo_drafts(tmp_path, project.id, code_id=_HEX_CODE)
        assert [d.id for d in out] == [d1.id]

    def test_list_filters_by_decision(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d1 = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        save_memo_draft(tmp_path, d1)
        d2 = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        record_memo_draft_decision(
            d2, decision=MEMO_DRAFT_DECISION_REJECTED, coder_id=_HEX_CODER,
        )
        save_memo_draft(tmp_path, d2)

        pending = list_memo_drafts(
            tmp_path, project.id, decision=MEMO_DRAFT_DECISION_PENDING
        )
        rejected = list_memo_drafts(
            tmp_path, project.id, decision=MEMO_DRAFT_DECISION_REJECTED
        )
        assert [d.id for d in pending] == [d1.id]
        assert [d.id for d in rejected] == [d2.id]

    def test_list_invalid_filter_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_memo_drafts(tmp_path, project.id, code_id="bad")
        with pytest.raises(ProjectValidationError):
            list_memo_drafts(tmp_path, project.id, decision="bogus")

    def test_list_skips_corrupt(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        save_memo_draft(tmp_path, d)
        # Drop a corrupt file alongside.
        bad = memo_drafts_dir(tmp_path, project.id) / ("0" * 12 + ".json")
        bad.write_text("not json")
        out = list_memo_drafts(tmp_path, project.id)
        assert [x.id for x in out] == [d.id]

    def test_list_skips_non_id_files(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        save_memo_draft(tmp_path, d)
        # File whose stem doesn't match the regex.
        weird = memo_drafts_dir(tmp_path, project.id) / "not-a-hex-stem.json"
        weird.write_text(json.dumps(d.to_dict()))
        out = list_memo_drafts(tmp_path, project.id)
        assert [x.id for x in out] == [d.id]

    def test_list_empty_when_dir_missing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        out = list_memo_drafts(tmp_path, project.id)
        assert out == []

    def test_delete(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = MemoDraft.new(project_id=project.id, code_id=_HEX_CODE)
        save_memo_draft(tmp_path, d)
        assert delete_memo_draft(tmp_path, project.id, d.id) is True
        assert delete_memo_draft(tmp_path, project.id, d.id) is False
        with pytest.raises(FileNotFoundError):
            load_memo_draft(tmp_path, project.id, d.id)

    def test_delete_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            delete_memo_draft(tmp_path, "a" * 12, "bad-id")

    def test_drafts_dirname_constant(self) -> None:
        assert MEMO_DRAFTS_DIRNAME == "memo_drafts"

    def test_list_sorted_by_created_at(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d1 = MemoDraft.new(
            project_id=project.id, code_id=_HEX_CODE, now="2026-05-25T00:00:00Z"
        )
        d2 = MemoDraft.new(
            project_id=project.id, code_id=_HEX_CODE, now="2026-05-26T00:00:00Z"
        )
        save_memo_draft(tmp_path, d2)
        save_memo_draft(tmp_path, d1)
        out = list_memo_drafts(tmp_path, project.id)
        assert [x.id for x in out] == [d1.id, d2.id]
