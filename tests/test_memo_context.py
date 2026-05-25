"""Tests for scribe.memo_context (F5.2 — right-click memo creation).

F5.1 shipped the Memo entity + persistence. F5.2 is the right-click
flow that pre-populates a draft memo with a link to whatever the user
clicked. These tests cover the pure-Python core:

* default_memo_type_for_target — closed-vocabulary mapping with a
  ``free`` fallback for unknown targets.
* build_memo_draft — primary-link prepopulation, smart type defaults,
  ``extra_links`` merging with dedupe against the primary, full
  validation.
* MemoContext — dict round-trip, validation, the helper that the
  server endpoint forwards.
* build_memo_draft_from_context — convenience wrapper used by the
  POST endpoint.

Endpoint-level tests (POST /api/projects/{pid}/memos accepting either
``{"context": ...}`` or a flat memo body) live in test_server.py.
"""

from __future__ import annotations

import pytest

from scribe.memos import (
    MEMO_LINK_TARGET_TYPES,
    MEMO_TYPES,
    Memo,
    MemoLink,
)
from scribe.memo_context import (
    DEFAULT_MEMO_TYPE_BY_TARGET,
    MemoContext,
    build_memo_draft,
    build_memo_draft_from_context,
    default_memo_type_for_target,
)
from scribe.projects import ProjectValidationError


# Hex sentinels matching every entity's 12-char id shape.
PROJECT_ID = "0" * 12
CODE_ID = "a" * 12
SOURCE_ID = "b" * 12
APP_ID = "d" * 12
PARTICIPANT_ID = "e" * 12
CODER_ID = "c" * 12
MEMO_ID = "f" * 12


# --------------------------------------------------------------------------- #
# default_memo_type_for_target
# --------------------------------------------------------------------------- #


class TestDefaultMemoTypeForTarget:
    def test_code_target_defaults_to_code_type(self) -> None:
        assert default_memo_type_for_target("code") == "code"

    def test_application_target_defaults_to_quote_type(self) -> None:
        # PLANNING.md F5.1 explicitly calls out the "quote / margin
        # annotation" affordance; a right-click on an application is
        # exactly that.
        assert default_memo_type_for_target("application") == "quote"

    def test_source_target_defaults_to_source_type(self) -> None:
        assert default_memo_type_for_target("source") == "source"

    def test_project_target_defaults_to_project_type(self) -> None:
        assert default_memo_type_for_target("project") == "project"

    def test_coder_target_defaults_to_methodological_type(self) -> None:
        # Notes about a coder are typically methodological.
        assert default_memo_type_for_target("coder") == "methodological"

    def test_memo_target_defaults_to_theoretical_type(self) -> None:
        # Memo->memo edges (F5.3) almost always carry theoretical
        # synthesis.
        assert default_memo_type_for_target("memo") == "theoretical"

    def test_participant_target_defaults_to_free(self) -> None:
        # Notes about a participant span too many flavours to presume.
        assert default_memo_type_for_target("participant") == "free"

    def test_unknown_target_defaults_to_free(self) -> None:
        # Forward-compat: a future entity in the UI shouldn't crash
        # the right-click flow before this mapping is updated.
        assert default_memo_type_for_target("not-a-real-target") == "free"

    def test_empty_string_defaults_to_free(self) -> None:
        assert default_memo_type_for_target("") == "free"

    def test_non_string_target_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            default_memo_type_for_target(123)  # type: ignore[arg-type]

    def test_every_default_is_a_valid_memo_type(self) -> None:
        # Defending against typos in the constant: every value the
        # mapping returns has to be in MEMO_TYPES, otherwise the
        # composer hands back an invalid Memo.
        for v in DEFAULT_MEMO_TYPE_BY_TARGET.values():
            assert v in MEMO_TYPES

    def test_every_link_target_type_has_a_default(self) -> None:
        # Every entity the user can right-click on must have an
        # explicit default — silent fall-throughs to ``"free"`` for
        # known target types would be a bug surfacing as "the wrong
        # type was preselected".
        for t in MEMO_LINK_TARGET_TYPES:
            assert t in DEFAULT_MEMO_TYPE_BY_TARGET


# --------------------------------------------------------------------------- #
# build_memo_draft — primary link prepopulation
# --------------------------------------------------------------------------- #


class TestBuildMemoDraftPrimary:
    def test_returns_a_memo_instance(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
        )
        assert isinstance(m, Memo)
        assert m.project_id == PROJECT_ID

    def test_primary_link_is_first_in_list(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
        )
        assert len(m.links) == 1
        assert m.links[0].target_type == "code"
        assert m.links[0].target_id == CODE_ID
        assert m.links[0].role == ""

    def test_role_is_carried_onto_primary_link(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="application",
            target_id=APP_ID,
            role="exemplifies",
        )
        assert m.links[0].role == "exemplifies"

    def test_default_type_is_inferred_from_target(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
        )
        assert m.type == "code"

        m2 = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="application",
            target_id=APP_ID,
        )
        assert m2.type == "quote"

    def test_explicit_type_overrides_default(self) -> None:
        # Researcher right-clicks a code, but wants this to be a
        # *theoretical* memo about a category that includes the code.
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
            type="theoretical",
        )
        assert m.type == "theoretical"

    def test_explicit_invalid_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type="code",
                target_id=CODE_ID,
                type="not-a-type",
            )

    def test_invalid_target_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type="planet",
                target_id=CODE_ID,
            )

    def test_non_string_target_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type=123,  # type: ignore[arg-type]
                target_id=CODE_ID,
            )

    def test_invalid_target_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type="code",
                target_id="not-hex",
            )

    def test_non_string_target_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type="code",
                target_id=42,  # type: ignore[arg-type]
            )

    def test_invalid_project_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id="not-hex",
                target_type="code",
                target_id=CODE_ID,
            )


# --------------------------------------------------------------------------- #
# build_memo_draft — extra links + composer fields
# --------------------------------------------------------------------------- #


class TestBuildMemoDraftExtras:
    def test_extra_link_dict_is_appended_after_primary(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="application",
            target_id=APP_ID,
            extra_links=[
                {"target_type": "code", "target_id": CODE_ID, "role": "applies"},
            ],
        )
        assert len(m.links) == 2
        assert (m.links[0].target_type, m.links[0].target_id) == (
            "application",
            APP_ID,
        )
        assert (m.links[1].target_type, m.links[1].target_id) == (
            "code",
            CODE_ID,
        )
        assert m.links[1].role == "applies"

    def test_extra_link_memo_link_instance_accepted(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="application",
            target_id=APP_ID,
            extra_links=[MemoLink(target_type="code", target_id=CODE_ID)],
        )
        assert len(m.links) == 2
        assert m.links[1].target_type == "code"

    def test_extra_link_dedupes_against_primary(self) -> None:
        # Caller hands the same primary triple twice; we collapse so
        # the on-disk link order stays [primary, ...real extras].
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
            role="exemplifies",
            extra_links=[
                {
                    "target_type": "code",
                    "target_id": CODE_ID,
                    "role": "exemplifies",
                },
            ],
        )
        assert len(m.links) == 1

    def test_extra_link_with_different_role_is_kept(self) -> None:
        # Same target, *different* role → distinct link (the on-disk
        # dedupe key is (target_type, target_id, role)).
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
            role="exemplifies",
            extra_links=[
                {
                    "target_type": "code",
                    "target_id": CODE_ID,
                    "role": "contradicts",
                },
            ],
        )
        assert len(m.links) == 2

    def test_invalid_extra_link_payload_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft(
                project_id=PROJECT_ID,
                target_type="code",
                target_id=CODE_ID,
                extra_links=["not-a-link"],  # type: ignore[list-item]
            )

    def test_composer_fields_round_trip(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="application",
            target_id=APP_ID,
            title="Note her hesitation",
            body="P3 pauses before answering. Compare to P07.",
            body_format="markdown",
            author_coder_id=CODER_ID,
            tags=["hesitation", "P3"],
            provenance={"source": "human"},
        )
        assert m.title == "Note her hesitation"
        assert m.body.startswith("P3 pauses")
        assert m.body_format == "markdown"
        assert m.author_coder_id == CODER_ID
        assert "hesitation" in m.tags and "P3" in m.tags
        assert m.provenance.get("source") == "human"

    def test_default_body_is_empty(self) -> None:
        # A fresh right-click memo has an empty body — the user is
        # *about* to type. Title-only memos are valid (placeholders).
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
        )
        assert m.body == ""
        assert m.title == ""

    def test_passes_now_through_to_memo_new(self) -> None:
        # Deterministic timestamps for the audit trail.
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
            now="2026-05-26T00:00:00Z",
        )
        assert m.created_at == "2026-05-26T00:00:00Z"
        assert m.modified_at == "2026-05-26T00:00:00Z"

    def test_passes_memo_id_through(self) -> None:
        m = build_memo_draft(
            project_id=PROJECT_ID,
            target_type="code",
            target_id=CODE_ID,
            memo_id=MEMO_ID,
        )
        assert m.id == MEMO_ID


# --------------------------------------------------------------------------- #
# MemoContext — dict round-trip + validation
# --------------------------------------------------------------------------- #


class TestMemoContext:
    def test_from_dict_basic(self) -> None:
        ctx = MemoContext.from_dict(
            {"target_type": "code", "target_id": CODE_ID}
        )
        assert ctx.target_type == "code"
        assert ctx.target_id == CODE_ID
        assert ctx.role == ""

    def test_from_dict_with_role(self) -> None:
        ctx = MemoContext.from_dict(
            {
                "target_type": "application",
                "target_id": APP_ID,
                "role": "exemplifies",
            }
        )
        assert ctx.role == "exemplifies"

    def test_from_dict_missing_required_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoContext.from_dict({"target_type": "code"})
        with pytest.raises(ProjectValidationError):
            MemoContext.from_dict({"target_id": CODE_ID})

    def test_from_dict_non_dict_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoContext.from_dict("nope")  # type: ignore[arg-type]

    def test_to_dict_omits_empty_role(self) -> None:
        ctx = MemoContext(target_type="code", target_id=CODE_ID)
        assert ctx.to_dict() == {
            "target_type": "code",
            "target_id": CODE_ID,
        }

    def test_to_dict_includes_role_when_present(self) -> None:
        ctx = MemoContext(
            target_type="code", target_id=CODE_ID, role="exemplifies"
        )
        assert ctx.to_dict() == {
            "target_type": "code",
            "target_id": CODE_ID,
            "role": "exemplifies",
        }

    def test_validate_rejects_unknown_target_type(self) -> None:
        ctx = MemoContext(target_type="planet", target_id=CODE_ID)
        with pytest.raises(ProjectValidationError):
            ctx.validate()

    def test_validate_rejects_bad_target_id(self) -> None:
        ctx = MemoContext(target_type="code", target_id="not-hex")
        with pytest.raises(ProjectValidationError):
            ctx.validate()

    def test_validate_rejects_bad_role_punctuation(self) -> None:
        # Role shape mirrors MemoLink's: letters/digits/underscore/
        # hyphen/space, must start with a letter. Punctuation is out.
        ctx = MemoContext(
            target_type="code", target_id=CODE_ID, role="!!nope"
        )
        with pytest.raises(ProjectValidationError):
            ctx.validate()

    def test_validate_accepts_well_formed_role(self) -> None:
        ctx = MemoContext(
            target_type="code", target_id=CODE_ID, role="exemplifies"
        )
        ctx.validate()  # no raise


# --------------------------------------------------------------------------- #
# build_memo_draft_from_context — server-endpoint convenience
# --------------------------------------------------------------------------- #


class TestBuildMemoDraftFromContext:
    def test_dict_context_round_trip(self) -> None:
        m = build_memo_draft_from_context(
            project_id=PROJECT_ID,
            context={"target_type": "code", "target_id": CODE_ID},
            title="Codebook entry edge case",
            body="When does this code *not* apply?",
        )
        assert m.type == "code"  # default for ``code`` target
        assert m.title == "Codebook entry edge case"
        assert len(m.links) == 1
        assert (m.links[0].target_type, m.links[0].target_id) == (
            "code",
            CODE_ID,
        )

    def test_memo_context_instance_accepted(self) -> None:
        ctx = MemoContext(
            target_type="application", target_id=APP_ID, role="exemplifies"
        )
        m = build_memo_draft_from_context(
            project_id=PROJECT_ID, context=ctx
        )
        assert m.type == "quote"
        assert m.links[0].role == "exemplifies"

    def test_invalid_context_dict_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft_from_context(
                project_id=PROJECT_ID,
                context={"target_type": "code"},  # missing target_id
            )

    def test_invalid_context_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            build_memo_draft_from_context(
                project_id=PROJECT_ID,
                context="not-a-context",  # type: ignore[arg-type]
            )

    def test_extra_kwargs_forwarded(self) -> None:
        m = build_memo_draft_from_context(
            project_id=PROJECT_ID,
            context={"target_type": "code", "target_id": CODE_ID},
            type="theoretical",
            tags=["initial-coding"],
            extra_links=[{"target_type": "source", "target_id": SOURCE_ID}],
        )
        assert m.type == "theoretical"
        assert "initial-coding" in m.tags
        assert len(m.links) == 2
        assert m.links[1].target_type == "source"

    def test_validate_runs_before_build(self) -> None:
        # Bad role on the context should surface as a validation error
        # before the Memo is constructed (otherwise the error message
        # points at MemoLink, which is true but less actionable).
        with pytest.raises(ProjectValidationError):
            build_memo_draft_from_context(
                project_id=PROJECT_ID,
                context={
                    "target_type": "code",
                    "target_id": CODE_ID,
                    "role": "!!bad!!",
                },
            )


# --------------------------------------------------------------------------- #
# Cross-feature: built memos round-trip through save_memo
# --------------------------------------------------------------------------- #


class TestRoundTripWithPersistence:
    """build_memo_draft must produce something that ``save_memo`` accepts.

    The whole point is that the right-click flow ends with one
    ``save_memo`` call. If the helper hands back something that
    ``save_memo`` rejects, the feature is broken. We exercise the full
    project → save_memo → load_memo cycle here rather than mocking.
    """

    def test_save_load_round_trip(self, tmp_path) -> None:
        from scribe.memos import load_memo, save_memo
        from scribe.projects import Project, save_project

        proj = Project.new(name="P")
        save_project(tmp_path, proj)

        m = build_memo_draft(
            project_id=proj.id,
            target_type="application",
            target_id=APP_ID,
            role="exemplifies",
            title="Why this quote sticks",
            body="P3 says the thing.",
            extra_links=[
                {"target_type": "code", "target_id": CODE_ID},
            ],
        )
        save_memo(tmp_path, m)
        loaded = load_memo(tmp_path, proj.id, m.id)
        assert loaded.title == "Why this quote sticks"
        assert loaded.type == "quote"  # default for application target
        assert len(loaded.links) == 2
        # Order is preserved on disk: primary then extras.
        assert loaded.links[0].target_type == "application"
        assert loaded.links[1].target_type == "code"
