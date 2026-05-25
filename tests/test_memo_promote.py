"""Tests for scribe.memo_promote (F5.5 — promote a memo into a code).

F5.1 shipped the Memo entity; F5.5 closes the analytic loop by giving
researchers the one-click path that turns a matured memo into a real
:class:`scribe.codes.Code`. These tests cover the three layers:

* :func:`derive_code_name_from_memo` — title preferred, body fallback,
  Markdown sigils stripped, raises on empty.
* :func:`build_code_from_memo` — pure builder. Defaults documented in
  the module docstring; we lock them in here so a future tweak that
  changes (say) the default ``status`` shows up as a test failure.
* :func:`promote_memo_to_code` — high-level persistence: code saved,
  v1 recorded in the version log, back-link added to the memo.

Endpoint-level tests (POST /api/projects/{pid}/memos/{mid}/promote-to-code)
live in ``tests/test_server.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.code_versions import (
    latest_code_version,
    read_code_versions,
)
from scribe.codes import (
    CODE_PROVENANCE_SOURCES,
    Code,
    CodeRelation,
    list_codes,
    load_code,
)
from scribe.memo_promote import (
    DEFAULT_BACK_LINK_ROLE,
    CodePromotionResult,
    build_code_from_memo,
    derive_code_name_from_memo,
    promote_memo_to_code,
)
from scribe.memos import (
    Memo,
    MemoLink,
    load_memo,
    save_memo,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)


# Hex-only sentinels that fit the 12-char id regex everywhere.
_HEX_PROJECT = "0" * 12
_HEX_MEMO = "f" * 12
_HEX_OTHER_MEMO = "1" * 12
_HEX_CODE = "a" * 12
_HEX_CODER = "c" * 12


# --------------------------------------------------------------------------- #
# Test fixtures / helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _saved_memo(
    tmp_path: Path,
    project: Project,
    *,
    title: str = "Managing the project",
    body: str = "Notes on managing.",
    type: str = "free",
    links=None,
) -> Memo:
    m = Memo.new(
        project_id=project.id,
        title=title,
        body=body,
        type=type,
        links=links or [],
    )
    save_memo(tmp_path, m)
    return m


# --------------------------------------------------------------------------- #
# derive_code_name_from_memo
# --------------------------------------------------------------------------- #


class TestDeriveCodeName:
    def test_title_preferred(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="  Managing  ", body="ignored")
        assert derive_code_name_from_memo(m) == "Managing"

    def test_falls_back_to_first_body_line(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT, title="", body="Caring for kin\nMore detail"
        )
        assert derive_code_name_from_memo(m) == "Caring for kin"

    def test_skips_blank_lines(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="", body="\n\n  \nFinally")
        assert derive_code_name_from_memo(m) == "Finally"

    def test_strips_markdown_heading(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="", body="## Resisting authority")
        assert derive_code_name_from_memo(m) == "Resisting authority"

    def test_strips_markdown_bullets(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="", body="- pacing the day")
        assert derive_code_name_from_memo(m) == "pacing the day"

    def test_strips_markdown_blockquote(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="", body="> reflecting")
        assert derive_code_name_from_memo(m) == "reflecting"

    def test_truncates_to_max_name_len(self) -> None:
        # Build a body whose first line is >200 chars.
        long_line = "x" * 500
        m = Memo.new(project_id=_HEX_PROJECT, title="", body=long_line)
        from scribe.codes import MAX_NAME_LEN

        out = derive_code_name_from_memo(m)
        assert len(out) == MAX_NAME_LEN
        assert out == "x" * MAX_NAME_LEN

    def test_title_at_memo_max_fits_code_max(self) -> None:
        # Memo.title cap (MAX_TITLE_LEN=200) currently matches Code.name
        # cap (MAX_NAME_LEN=200), so any valid memo title fits as a
        # code name unchanged. Pin the invariant — if either constant
        # ever drifts, this test surfaces it and the truncation guard
        # in ``derive_code_name_from_memo`` kicks in to keep the build
        # safe.
        from scribe.codes import MAX_NAME_LEN
        from scribe.memos import MAX_TITLE_LEN

        assert MAX_TITLE_LEN <= MAX_NAME_LEN
        m = Memo.new(project_id=_HEX_PROJECT, title="y" * MAX_TITLE_LEN)
        out = derive_code_name_from_memo(m)
        assert len(out) <= MAX_NAME_LEN
        assert out == "y" * MAX_TITLE_LEN

    def test_raises_when_title_and_body_both_empty(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            derive_code_name_from_memo(m)

    def test_raises_when_only_whitespace(self) -> None:
        # Title is whitespace-stripped to "" by validate; an empty body
        # leaves nothing to derive a name from.
        m = Memo.new(project_id=_HEX_PROJECT, title="   ", body="\n  \n")
        with pytest.raises(ProjectValidationError):
            derive_code_name_from_memo(m)

    def test_rejects_non_memo_input(self) -> None:
        with pytest.raises(TypeError):
            derive_code_name_from_memo({"title": "fake"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# build_code_from_memo — defaults
# --------------------------------------------------------------------------- #


class TestBuildCodeFromMemoDefaults:
    def test_returns_a_code_instance(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Caring", body="")
        c = build_code_from_memo(memo=m)
        assert isinstance(c, Code)
        assert c.project_id == _HEX_PROJECT

    def test_default_name_from_title(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing", body="ignored body")
        c = build_code_from_memo(memo=m)
        assert c.name == "Pacing"

    def test_default_name_falls_back_to_body(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="", body="# Drifting")
        c = build_code_from_memo(memo=m)
        assert c.name == "Drifting"

    def test_default_definition_is_memo_body(self) -> None:
        body = "A long-running pattern of avoiding eye contact in interviews."
        m = Memo.new(project_id=_HEX_PROJECT, title="Avoiding", body=body)
        c = build_code_from_memo(memo=m)
        assert c.definition == body

    def test_theoretical_memo_seeded_only_for_theoretical_type(self) -> None:
        body = "Body of analytic notes."
        m_th = Memo.new(
            project_id=_HEX_PROJECT,
            title="Concept X",
            body=body,
            type="theoretical",
        )
        c_th = build_code_from_memo(memo=m_th)
        assert c_th.theoretical_memo == body

        m_other = Memo.new(
            project_id=_HEX_PROJECT,
            title="Note",
            body=body,
            type="reflexive",
        )
        c_other = build_code_from_memo(memo=m_other)
        assert c_other.theoretical_memo == ""

    def test_provenance_records_source_and_memo_id(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m)
        assert c.provenance.get("source") == "promoted_from_memo"
        assert c.provenance.get("memo_id") == m.id

    def test_provenance_value_is_in_closed_vocabulary(self) -> None:
        # Belt-and-braces — F2.1 keeps a closed set, our default must
        # satisfy it. Otherwise Code.validate would reject our build.
        assert "promoted_from_memo" in CODE_PROVENANCE_SOURCES

    def test_default_status_is_active(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m)
        assert c.status == "active"

    def test_default_stage_is_initial(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m)
        assert c.stage == "initial"

    def test_default_status_is_overridable(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m, status="draft")
        assert c.status == "draft"

    def test_default_stage_is_overridable(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m, stage="focused")
        assert c.stage == "focused"

    def test_explicit_name_wins(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m, name="Explicit name")
        assert c.name == "Explicit name"

    def test_explicit_definition_wins(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing", body="long body")
        c = build_code_from_memo(memo=m, definition="Short.")
        assert c.definition == "Short."

    def test_explicit_theoretical_memo_wins(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            title="Pacing",
            body="long body",
            type="theoretical",
        )
        c = build_code_from_memo(memo=m, theoretical_memo="Custom.")
        assert c.theoretical_memo == "Custom."

    def test_blank_explicit_name_falls_back_to_derivation(self) -> None:
        # Passing an empty name is the same as not passing one — we
        # don't want a button-click that omits the field to land a
        # code with name="" (Code.validate would refuse anyway).
        m = Memo.new(project_id=_HEX_PROJECT, title="Caring")
        c = build_code_from_memo(memo=m, name="   ")
        assert c.name == "Caring"

    def test_extra_provenance_merges(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(
            memo=m, extra_provenance={"promoted_by": _HEX_CODER}
        )
        assert c.provenance["source"] == "promoted_from_memo"
        assert c.provenance["memo_id"] == m.id
        assert c.provenance["promoted_by"] == _HEX_CODER

    def test_extra_provenance_cannot_override_source(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(
                memo=m, extra_provenance={"source": "human"}
            )

    def test_extra_provenance_cannot_override_memo_id(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(
                memo=m, extra_provenance={"memo_id": _HEX_OTHER_MEMO}
            )

    def test_extra_provenance_must_be_a_mapping(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(
                memo=m, extra_provenance=[("foo", "bar")]  # type: ignore[arg-type]
            )

    def test_unknown_status_raises(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(memo=m, status="not-a-status")

    def test_unknown_stage_raises(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(memo=m, stage="not-a-stage")

    def test_too_long_body_for_definition_raises(self) -> None:
        # Memo.body cap is 64 KiB; Code.definition cap is 4 000.
        long_body = "x" * (4001)
        m = Memo.new(project_id=_HEX_PROJECT, title="Long", body=long_body)
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(memo=m)

    def test_too_long_body_with_explicit_short_definition_succeeds(self) -> None:
        long_body = "x" * (4001)
        m = Memo.new(project_id=_HEX_PROJECT, title="Long", body=long_body)
        c = build_code_from_memo(memo=m, definition="Short.")
        assert c.definition == "Short."

    def test_explicit_code_id_is_used(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(memo=m, code_id=_HEX_CODE)
        assert c.id == _HEX_CODE

    def test_related_codes_accept_dicts(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        c = build_code_from_memo(
            memo=m,
            related_codes=[
                {"code_id": _HEX_CODE, "relation_type": "associated"},
            ],
        )
        assert len(c.related_codes) == 1
        assert c.related_codes[0].code_id == _HEX_CODE
        assert c.related_codes[0].relation_type == "associated"

    def test_related_codes_accept_instances(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        rel = CodeRelation(code_id=_HEX_CODE, relation_type="broader")
        c = build_code_from_memo(memo=m, related_codes=[rel])
        assert len(c.related_codes) == 1

    def test_related_codes_reject_garbage(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="Pacing")
        with pytest.raises(ProjectValidationError):
            build_code_from_memo(memo=m, related_codes=[42])  # type: ignore[list-item]

    def test_rejects_non_memo_input(self) -> None:
        with pytest.raises(TypeError):
            build_code_from_memo(memo={"title": "x"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# promote_memo_to_code — persistence
# --------------------------------------------------------------------------- #


class TestPromoteMemoToCode:
    def test_returns_a_promotion_result(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        assert isinstance(result, CodePromotionResult)
        assert isinstance(result.code, Code)

    def test_persists_the_code_file(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing", body="Body")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        # The code is round-trippable from disk.
        loaded = load_code(tmp_path, proj.id, result.code.id)
        assert loaded.name == "Managing"
        assert loaded.definition == "Body"

    def test_records_v1_in_version_log(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        versions = read_code_versions(tmp_path, proj.id, result.code.id)
        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].id == result.version.id

    def test_v1_change_note_records_promotion_lineage(
        self, tmp_path: Path
    ) -> None:
        # The default change_note ought to identify the source memo
        # so the version log stands on its own as an audit record.
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        assert memo.id in result.version.change_note

    def test_explicit_change_note_wins(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(
            tmp_path, proj.id, memo.id, change_note="Custom note."
        )
        assert result.version.change_note == "Custom note."

    def test_back_link_is_added_by_default(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        # The memo should now carry a link to the new code.
        reloaded = load_memo(tmp_path, proj.id, memo.id)
        assert reloaded.has_link_to("code", result.code.id)
        match = [
            link
            for link in reloaded.links
            if link.target_type == "code" and link.target_id == result.code.id
        ]
        assert match
        assert match[0].role == DEFAULT_BACK_LINK_ROLE

    def test_back_link_role_overridable(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(
            tmp_path, proj.id, memo.id, back_link_role="became"
        )
        reloaded = load_memo(tmp_path, proj.id, memo.id)
        match = [
            link
            for link in reloaded.links
            if link.target_id == result.code.id
        ]
        assert match[0].role == "became"

    def test_back_link_can_be_disabled(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        original_modified = memo.modified_at
        result = promote_memo_to_code(
            tmp_path, proj.id, memo.id, record_back_link=False
        )
        reloaded = load_memo(tmp_path, proj.id, memo.id)
        assert not reloaded.has_link_to("code", result.code.id)
        # And modified_at on disk shouldn't have advanced.
        assert reloaded.modified_at == original_modified

    def test_back_link_idempotent_when_already_present(
        self, tmp_path: Path
    ) -> None:
        # Promoting twice (e.g. user clicked button twice) should not
        # duplicate the back-link nor leave the memo in a state that
        # fails Memo.validate.
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        first = promote_memo_to_code(tmp_path, proj.id, memo.id)
        # Use the same generated code id on the second pass to
        # exercise the de-dup path explicitly.
        second = promote_memo_to_code(
            tmp_path, proj.id, memo.id, code_id=first.code.id
        )
        reloaded = load_memo(tmp_path, proj.id, memo.id)
        matches = [
            link for link in reloaded.links
            if link.target_type == "code" and link.target_id == first.code.id
        ]
        assert len(matches) == 1
        # And the second call returns the same code id.
        assert second.code.id == first.code.id

    def test_existing_memo_links_preserved(self, tmp_path: Path) -> None:
        # Memos can carry pre-existing links (e.g. to the source they
        # were written about). The back-link append must not clobber them.
        proj = _saved_project(tmp_path)
        prior = MemoLink(target_type="source", target_id="b" * 12)
        memo = _saved_memo(
            tmp_path, proj, title="Managing", links=[prior]
        )
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        reloaded = load_memo(tmp_path, proj.id, memo.id)
        # Both links present.
        assert reloaded.has_link_to("source", "b" * 12)
        assert reloaded.has_link_to("code", result.code.id)

    def test_two_promotions_yield_distinct_codes(
        self, tmp_path: Path
    ) -> None:
        # Promoting two different memos should mint two different
        # codes — ids come from new_code_id().
        proj = _saved_project(tmp_path)
        m1 = _saved_memo(tmp_path, proj, title="Managing")
        m2 = Memo.new(project_id=proj.id, title="Pacing")
        save_memo(tmp_path, m2)
        r1 = promote_memo_to_code(tmp_path, proj.id, m1.id)
        r2 = promote_memo_to_code(tmp_path, proj.id, m2.id)
        assert r1.code.id != r2.code.id
        all_codes = list_codes(tmp_path, proj.id)
        ids = {c.id for c in all_codes}
        assert r1.code.id in ids
        assert r2.code.id in ids

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        # No project on disk — load_memo can't resolve the path.
        with pytest.raises(FileNotFoundError):
            promote_memo_to_code(tmp_path, "0" * 12, _HEX_MEMO)

    def test_missing_memo_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            promote_memo_to_code(tmp_path, proj.id, _HEX_MEMO)

    def test_invalid_project_id_shape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            promote_memo_to_code(tmp_path, "not-hex", _HEX_MEMO)

    def test_invalid_memo_id_shape_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            promote_memo_to_code(tmp_path, proj.id, "ZZZ")

    def test_overrides_forwarded(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(
            tmp_path, proj, title="Managing", body="Original body."
        )
        result = promote_memo_to_code(
            tmp_path,
            proj.id,
            memo.id,
            name="Renamed",
            definition="Crisp definition.",
            inclusion_criteria="if X",
            exclusion_criteria="not Y",
            exemplars=["A worked example"],
            stage="focused",
            colour="#0a0a0a",
            status="draft",
        )
        loaded = load_code(tmp_path, proj.id, result.code.id)
        assert loaded.name == "Renamed"
        assert loaded.definition == "Crisp definition."
        assert loaded.inclusion_criteria == "if X"
        assert loaded.exclusion_criteria == "not Y"
        assert loaded.exemplars == ["A worked example"]
        assert loaded.stage == "focused"
        assert loaded.colour == "#0a0a0a"
        assert loaded.status == "draft"

    def test_extra_provenance_persists(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(
            tmp_path,
            proj.id,
            memo.id,
            extra_provenance={"promoted_by": _HEX_CODER},
        )
        loaded = load_code(tmp_path, proj.id, result.code.id)
        assert loaded.provenance["source"] == "promoted_from_memo"
        assert loaded.provenance["memo_id"] == memo.id
        assert loaded.provenance["promoted_by"] == _HEX_CODER

    def test_explicit_code_id_used_on_disk(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(
            tmp_path, proj.id, memo.id, code_id=_HEX_CODE
        )
        assert result.code.id == _HEX_CODE
        # latest_code_version finds it under that id.
        latest = latest_code_version(tmp_path, proj.id, _HEX_CODE)
        assert latest is not None
        assert latest.version == 1

    def test_pinned_now_flows_to_code_and_version(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        ts = "2026-05-01T12:00:00Z"
        result = promote_memo_to_code(
            tmp_path, proj.id, memo.id, now=ts
        )
        assert result.code.created_at == ts
        assert result.code.modified_at == ts
        assert result.version.created_at == ts

    def test_does_not_record_a_v2_on_first_promotion(
        self, tmp_path: Path
    ) -> None:
        # No prior versions, save_code_with_version always writes v1
        # — but we explicitly check we did not somehow write twice.
        proj = _saved_project(tmp_path)
        memo = _saved_memo(tmp_path, proj, title="Managing")
        result = promote_memo_to_code(tmp_path, proj.id, memo.id)
        versions = read_code_versions(tmp_path, proj.id, result.code.id)
        assert [v.version for v in versions] == [1]
