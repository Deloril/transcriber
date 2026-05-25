"""Tests for scribe.codes (F2.1).

These exercise the Code entity in pure Python: the field set called for
in PLANNING.md F2.1, validation rules, serialisation round-trips, the
CodeRelation typed-link helper, partial updates, and the file-system
persistence helpers. Endpoint-level tests will live in test_server.py
once F2.1 grows an HTTP surface; today the model + persistence are the
public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.projects import (
    CODEBOOK_STAGES,
    Project,
    ProjectValidationError,
    delete_project,
    project_dir,
    save_project,
)
from scribe.codes import (
    CODE_COLOUR_RE,
    CODE_ID_RE,
    CODE_PROVENANCE_SOURCES,
    CODE_RELATION_TYPES,
    CODE_STATUSES,
    MAX_DEFINITION_LEN,
    MAX_EXEMPLAR_LEN,
    MAX_EXEMPLARS,
    MAX_EXCLUSION_CRITERIA_LEN,
    MAX_INCLUSION_CRITERIA_LEN,
    MAX_NAME_LEN,
    MAX_PROVENANCE_KEYS,
    MAX_PROVENANCE_VALUE_LEN,
    MAX_RELATED_CODES,
    MAX_THEORETICAL_MEMO_LEN,
    PROVENANCE_KEY_RE,
    Code,
    CodeRelation,
    code_state_path,
    codes_dir,
    delete_code,
    list_codes,
    load_code,
    new_code_id,
    save_code,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewCodeId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert CODE_ID_RE.match(new_code_id())

    def test_unique(self) -> None:
        ids = {new_code_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_relation_types_include_charmaz_skos_basics(self) -> None:
        # Hard-coded sanity: these are the values older on-disk codes
        # may carry; renaming any silently would break loading.
        for rel in (
            "broader",
            "narrower",
            "associated",
            "contrasts_with",
            "causes",
            "follows",
        ):
            assert rel in CODE_RELATION_TYPES

    def test_statuses_include_active_draft_retired(self) -> None:
        for s in ("active", "draft", "retired"):
            assert s in CODE_STATUSES

    def test_provenance_sources_include_human_and_ai(self) -> None:
        for s in ("human", "ai_suggested", "ai_modified", "promoted_from_memo"):
            assert s in CODE_PROVENANCE_SOURCES

    def test_code_stage_vocab_matches_project(self) -> None:
        # F2.4 will rely on the same vocabulary across project + code,
        # so guard against accidental partial drift in either module.
        # The full equality is exercised by the parametrised
        # ``test_each_stage_accepted`` below.
        assert "initial" in CODEBOOK_STAGES
        assert "focused" in CODEBOOK_STAGES
        assert "axial" in CODEBOOK_STAGES
        assert "theoretical" in CODEBOOK_STAGES
        assert "locked" in CODEBOOK_STAGES


# --------------------------------------------------------------------------- #
# Colour regex
# --------------------------------------------------------------------------- #


class TestColourRegex:
    @pytest.mark.parametrize("good", [
        "#abc", "#ABC", "#abcdef", "#A1B2C3", "#000", "#000000", "#FFFFFF",
    ])
    def test_accepts_good(self, good: str) -> None:
        assert CODE_COLOUR_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "abc",          # missing #
        "#ab",          # too short
        "#abcd",        # 4 chars (rgba shorthand we don't support)
        "#abcde",       # 5 chars
        "#abcdefg",     # 7 chars
        "rgba(0,0,0,0)",
        "red",
    ])
    def test_rejects_bad(self, bad: str) -> None:
        assert not CODE_COLOUR_RE.match(bad)


# --------------------------------------------------------------------------- #
# Provenance key regex
# --------------------------------------------------------------------------- #


class TestProvenanceKeyRegex:
    @pytest.mark.parametrize("good", [
        "source", "model_id", "model id", "Run-1", "AbC_123-x y",
    ])
    def test_accepts_good(self, good: str) -> None:
        assert PROVENANCE_KEY_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "", " starts with space", "1leading_digit", "has.dot",
        "has/slash", "has\nnewline",
    ])
    def test_rejects_bad(self, bad: str) -> None:
        assert not PROVENANCE_KEY_RE.match(bad)


# --------------------------------------------------------------------------- #
# CodeRelation
# --------------------------------------------------------------------------- #


class TestCodeRelation:
    def test_to_from_dict_round_trip(self) -> None:
        r = CodeRelation(code_id="0123456789ab", relation_type="broader")
        r.validate()
        d = r.to_dict()
        assert d == {"code_id": "0123456789ab", "relation_type": "broader"}
        r2 = CodeRelation.from_dict(d)
        assert r2.to_dict() == d

    def test_validate_rejects_bad_id(self) -> None:
        r = CodeRelation(code_id="UPPERCASE123", relation_type="broader")
        with pytest.raises(ProjectValidationError):
            r.validate()

    def test_validate_rejects_unknown_type(self) -> None:
        r = CodeRelation(code_id="0123456789ab", relation_type="invented")
        with pytest.raises(ProjectValidationError):
            r.validate()

    def test_from_dict_requires_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeRelation.from_dict({"code_id": "0123456789ab"})
        with pytest.raises(ProjectValidationError):
            CodeRelation.from_dict({"relation_type": "broader"})

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeRelation.from_dict("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Code.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestCodeNew:
    def test_minimal(self) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="Pacing")
        assert c.name == "Pacing"
        assert c.project_id == "aaaaaaaaaaaa"
        assert c.id and CODE_ID_RE.match(c.id)
        assert c.definition == ""
        assert c.inclusion_criteria == ""
        assert c.exclusion_criteria == ""
        assert c.exemplars == []
        assert c.parent_code_id is None
        assert c.related_codes == []
        assert c.theoretical_memo == ""
        assert c.stage == "initial"
        assert c.colour == ""
        assert c.status == "active"
        assert c.provenance == {}
        assert c.created_at == c.modified_at
        assert c.created_at  # non-empty

    def test_strips_name_whitespace(self) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="  Trim me  ")
        assert c.name == "Trim me"

    def test_blank_name_rejected(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(ProjectValidationError):
                Code.new(project_id="aaaaaaaaaaaa", name=bad)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa", name="x" * (MAX_NAME_LEN + 1)
            )

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="UPPERCASE123", name="ok")
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="../escape", name="ok")
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="short", name="ok")

    def test_invalid_code_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa", name="ok", code_id="UPPERCASE123"
            )

    def test_definition_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                definition="x" * (MAX_DEFINITION_LEN + 1),
            )

    def test_inclusion_criteria_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                inclusion_criteria="x" * (MAX_INCLUSION_CRITERIA_LEN + 1),
            )

    def test_exclusion_criteria_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                exclusion_criteria="x" * (MAX_EXCLUSION_CRITERIA_LEN + 1),
            )

    def test_theoretical_memo_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                theoretical_memo="x" * (MAX_THEORETICAL_MEMO_LEN + 1),
            )

    @pytest.mark.parametrize("stage", CODEBOOK_STAGES)
    def test_each_stage_accepted(self, stage: str) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="ok", stage=stage)
        assert c.stage == stage

    def test_invalid_stage_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="aaaaaaaaaaaa", name="ok", stage="late")

    @pytest.mark.parametrize("status", CODE_STATUSES)
    def test_each_status_accepted(self, status: str) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="ok", status=status)
        assert c.status == status

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="aaaaaaaaaaaa", name="ok", status="archived")

    def test_colour_empty_ok(self) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="ok", colour="")
        assert c.colour == ""

    @pytest.mark.parametrize("good", ["#abc", "#abcdef", "#A1B2C3"])
    def test_colour_hex_accepted(self, good: str) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="ok", colour=good)
        assert c.colour == good

    @pytest.mark.parametrize("bad", ["red", "abcdef", "#abcd", "#abcdefg"])
    def test_colour_invalid_rejected(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(project_id="aaaaaaaaaaaa", name="ok", colour=bad)

    def test_parent_code_id_validated(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            parent_code_id="0123456789ab",
        )
        assert c.parent_code_id == "0123456789ab"

    def test_parent_code_id_bad_shape_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                parent_code_id="../etc",
            )
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                parent_code_id="UPPERCASE123",
            )

    def test_parent_code_id_empty_becomes_none(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa", name="ok", parent_code_id=""
        )
        assert c.parent_code_id is None

    def test_parent_code_id_self_reference_rejected(self) -> None:
        # Catching a self-parent at validate-time is cheap and prevents
        # the obvious foot-gun. F2.3 will add full cycle detection.
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                code_id="0123456789ab",
                parent_code_id="0123456789ab",
            )

    def test_exemplars_accepted(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            exemplars=["I felt stuck.", "Just couldn't move."],
        )
        assert c.exemplars == ["I felt stuck.", "Just couldn't move."]

    def test_exemplars_strip_and_drop_empty(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            exemplars=["  trimmed  ", "", "   ", "kept"],
        )
        assert c.exemplars == ["trimmed", "kept"]

    def test_exemplars_too_many_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                exemplars=["q"] * (MAX_EXEMPLARS + 1),
            )

    def test_exemplar_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                exemplars=["x" * (MAX_EXEMPLAR_LEN + 1)],
            )

    def test_related_codes_accepts_dicts_and_dataclasses(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            related_codes=[
                {"code_id": "0123456789ab", "relation_type": "broader"},
                CodeRelation(
                    code_id="cafebabecafe", relation_type="associated"
                ),
            ],
        )
        assert [r.relation_type for r in c.related_codes] == [
            "broader",
            "associated",
        ]
        assert [r.code_id for r in c.related_codes] == [
            "0123456789ab",
            "cafebabecafe",
        ]

    def test_related_codes_dedup_on_pair(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            related_codes=[
                {"code_id": "0123456789ab", "relation_type": "broader"},
                {"code_id": "0123456789ab", "relation_type": "broader"},
                # Same target, different relation: kept (it's a
                # different *typed* link).
                {"code_id": "0123456789ab", "relation_type": "associated"},
            ],
        )
        assert len(c.related_codes) == 2

    def test_related_codes_too_many_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                related_codes=[
                    # Distinct ids so dedup doesn't shrink the list.
                    {"code_id": f"{i:012x}", "relation_type": "associated"}
                    for i in range(MAX_RELATED_CODES + 1)
                ],
            )

    def test_related_codes_self_reference_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                code_id="0123456789ab",
                related_codes=[
                    {"code_id": "0123456789ab", "relation_type": "broader"},
                ],
            )

    def test_related_codes_unknown_type_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                related_codes=[
                    {"code_id": "0123456789ab", "relation_type": "imagined"},
                ],
            )

    def test_provenance_accepted(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            provenance={"source": "ai_suggested", "model_id": "phi-4"},
        )
        assert c.provenance == {
            "source": "ai_suggested",
            "model_id": "phi-4",
        }

    def test_provenance_drops_blank_keys(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            provenance={"  ": "ignored", "source": "human"},
        )
        assert c.provenance == {"source": "human"}

    def test_provenance_coerces_values_to_str(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            provenance={"source": "human", "round": 3},
        )
        assert c.provenance == {"source": "human", "round": "3"}

    def test_provenance_rejects_bad_keys(self) -> None:
        for bad in ("1leading", "has/slash", "has.dot", "x" * 100):
            with pytest.raises(ProjectValidationError):
                Code.new(
                    project_id="aaaaaaaaaaaa",
                    name="ok",
                    provenance={bad: "v"},
                )

    def test_provenance_value_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                provenance={"source": "human", "extra": "x" * (MAX_PROVENANCE_VALUE_LEN + 1)},
            )

    def test_provenance_too_many_keys_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                provenance={f"k{i}": "v" for i in range(MAX_PROVENANCE_KEYS + 1)},
            )

    def test_provenance_unknown_source_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                provenance={"source": "telepathy"},
            )

    @pytest.mark.parametrize("source", CODE_PROVENANCE_SOURCES)
    def test_provenance_each_known_source_accepted(self, source: str) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            provenance={"source": source},
        )
        assert c.provenance["source"] == source

    def test_explicit_code_id(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            code_id="bbbbbbbbbbbb",
        )
        assert c.id == "bbbbbbbbbbbb"

    def test_explicit_now_used(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            now="2024-01-01T00:00:00.000000Z",
        )
        assert c.created_at == "2024-01-01T00:00:00.000000Z"
        assert c.modified_at == "2024-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="Managing exhaustion",
            definition="How participants describe being depleted.",
            inclusion_criteria="Statements about energy/sleep.",
            exclusion_criteria="Statements about boredom.",
            exemplars=["I just couldn't get out of bed."],
            parent_code_id="cafebabecafe",
            related_codes=[
                {"code_id": "0123456789ab", "relation_type": "associated"},
            ],
            theoretical_memo="Connects to capacity-bounding category.",
            stage="focused",
            colour="#A1B2C3",
            status="active",
            provenance={"source": "human"},
        )
        d = c.to_dict()
        assert json.dumps(d)  # JSON-serialisable
        c2 = Code.from_dict(d)
        assert c2.to_dict() == d

    def test_from_dict_requires_required_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.from_dict({"name": "x", "project_id": "aaaaaaaaaaaa"})  # no id
        with pytest.raises(ProjectValidationError):
            Code.from_dict({"id": "bbbbbbbbbbbb", "name": "x"})  # no project_id
        with pytest.raises(ProjectValidationError):
            Code.from_dict(
                {"id": "bbbbbbbbbbbb", "project_id": "aaaaaaaaaaaa"}
            )  # no name

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Code.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_defaults_missing_fields(self) -> None:
        c = Code.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "name": "ok",
        })
        assert c.definition == ""
        assert c.parent_code_id is None
        assert c.related_codes == []
        assert c.exemplars == []
        assert c.stage == "initial"
        assert c.status == "active"
        assert c.colour == ""
        assert c.provenance == {}

    def test_from_dict_treats_falsy_parent_as_none(self) -> None:
        c = Code.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "name": "ok",
            "parent_code_id": "",
        })
        assert c.parent_code_id is None


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def _fresh(self) -> Code:
        return Code.new(
            project_id="aaaaaaaaaaaa",
            name="Old name",
            now="2024-01-01T00:00:00.000000Z",
        )

    def test_updates_name_and_advances_modified_at(self) -> None:
        c = self._fresh()
        c.apply_update({"name": "New name"}, now="2024-06-01T00:00:00.000000Z")
        assert c.name == "New name"
        assert c.created_at == "2024-01-01T00:00:00.000000Z"
        assert c.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_updates_definition_and_criteria(self) -> None:
        c = self._fresh()
        c.apply_update({
            "definition": "d",
            "inclusion_criteria": "in",
            "exclusion_criteria": "out",
        })
        assert c.definition == "d"
        assert c.inclusion_criteria == "in"
        assert c.exclusion_criteria == "out"

    def test_updates_exemplars(self) -> None:
        c = self._fresh()
        c.apply_update({"exemplars": ["a", "b"]})
        assert c.exemplars == ["a", "b"]
        # Replacing with an empty list clears.
        c.apply_update({"exemplars": []})
        assert c.exemplars == []

    def test_updates_parent_code_id(self) -> None:
        c = self._fresh()
        c.apply_update({"parent_code_id": "0123456789ab"})
        assert c.parent_code_id == "0123456789ab"

    def test_clears_parent_via_empty(self) -> None:
        c = self._fresh()
        c.parent_code_id = "0123456789ab"
        c.apply_update({"parent_code_id": ""})
        assert c.parent_code_id is None

    def test_clears_parent_via_null(self) -> None:
        c = self._fresh()
        c.parent_code_id = "0123456789ab"
        c.apply_update({"parent_code_id": None})
        assert c.parent_code_id is None

    def test_updates_related_codes_replaces_list(self) -> None:
        c = self._fresh()
        c.apply_update({
            "related_codes": [
                {"code_id": "0123456789ab", "relation_type": "broader"},
            ]
        })
        assert len(c.related_codes) == 1
        assert c.related_codes[0].relation_type == "broader"
        # Subsequent update fully replaces — it's not a merge.
        c.apply_update({"related_codes": []})
        assert c.related_codes == []

    def test_updates_stage_and_status_and_colour(self) -> None:
        c = self._fresh()
        c.apply_update({
            "stage": "focused",
            "status": "draft",
            "colour": "#abcdef",
        })
        assert c.stage == "focused"
        assert c.status == "draft"
        assert c.colour == "#abcdef"

    def test_updates_provenance(self) -> None:
        c = self._fresh()
        c.apply_update({"provenance": {"source": "ai_suggested"}})
        assert c.provenance == {"source": "ai_suggested"}

    def test_unknown_fields_rejected(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update({"random_thing": 1})

    def test_id_in_patch_ignored(self) -> None:
        c = self._fresh()
        original = c.id
        c.apply_update({"id": "ffffffffffff", "name": "renamed"})
        assert c.id == original

    def test_project_id_in_patch_ignored(self) -> None:
        c = self._fresh()
        original = c.project_id
        c.apply_update({"project_id": "ffffffffffff", "name": "renamed"})
        assert c.project_id == original

    def test_failed_validation_does_not_advance_clock(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update(
                {"stage": "bogus"},
                now="2099-01-01T00:00:00.000000Z",
            )
        assert c.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_non_dict_patch_rejected(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update("not a dict")  # type: ignore[arg-type]

    def test_related_codes_must_be_list(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update({"related_codes": {"oops": True}})

    def test_exemplars_must_be_list(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update({"exemplars": "oops"})

    def test_provenance_must_be_dict(self) -> None:
        c = self._fresh()
        with pytest.raises(ProjectValidationError):
            c.apply_update({"provenance": ["oops"]})


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="Pacing", definition="d")
        path = save_code(tmp_path, c)
        assert path.exists()
        assert path == code_state_path(tmp_path, proj.id, c.id)

        loaded = load_code(tmp_path, proj.id, c.id)
        assert loaded.to_dict() == c.to_dict()

    def test_save_creates_codes_subdir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="ok")
        save_code(tmp_path, c)
        assert codes_dir(tmp_path, proj.id).is_dir()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="ok")
        save_code(tmp_path, c)
        cd = codes_dir(tmp_path, proj.id)
        assert not (cd / f"{c.id}.json.tmp").exists()
        assert (cd / f"{c.id}.json").exists()

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        # Don't save_project — directory does not exist.
        c = Code.new(project_id="aaaaaaaaaaaa", name="orphan")
        with pytest.raises(FileNotFoundError):
            save_code(tmp_path, c)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_code(tmp_path, proj.id, "bbbbbbbbbbbb")

    def test_load_validates_code_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            load_code(tmp_path, proj.id, "../etc/passwd")

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_codes(tmp_path, proj.id) == []

    def test_list_no_project_dir(self, tmp_path: Path) -> None:
        # No save_project; codes_dir won't exist.
        assert list_codes(tmp_path, "aaaaaaaaaaaa") == []

    def test_list_skips_stray_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        cd = codes_dir(tmp_path, proj.id)
        cd.mkdir()
        # Wrong id shape: dropped.
        (cd / "not-a-code.json").write_text("{}")
        # Valid id but corrupt JSON: dropped.
        (cd / "aaaaaaaaaaaa.json").write_text("not json")
        # Tmp file: dropped.
        (cd / "bbbbbbbbbbbb.json.tmp").write_text("{}")
        # Wrong shape, not .json: dropped.
        (cd / "bbbbbbbbbbbb.txt").write_text("nope")
        assert list_codes(tmp_path, proj.id) == []

    def test_list_sorted_by_created_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Code.new(
            project_id=proj.id, name="A", now="2024-01-01T00:00:00.000000Z"
        )
        b = Code.new(
            project_id=proj.id, name="B", now="2024-02-01T00:00:00.000000Z"
        )
        c = Code.new(
            project_id=proj.id, name="C", now="2024-03-01T00:00:00.000000Z"
        )
        # Save in a deliberately scrambled order.
        save_code(tmp_path, b)
        save_code(tmp_path, a)
        save_code(tmp_path, c)
        names = [x.name for x in list_codes(tmp_path, proj.id)]
        assert names == ["A", "B", "C"]

    def test_save_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="ok")
        c.name = ""  # corrupt directly to bypass apply_update
        with pytest.raises(ProjectValidationError):
            save_code(tmp_path, c)
        # Nothing got written.
        assert not code_state_path(tmp_path, proj.id, c.id).exists()

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="Doomed")
        save_code(tmp_path, c)
        assert code_state_path(tmp_path, proj.id, c.id).exists()
        assert delete_code(tmp_path, proj.id, c.id) is True
        assert not code_state_path(tmp_path, proj.id, c.id).exists()
        # Idempotent.
        assert delete_code(tmp_path, proj.id, c.id) is False

    def test_delete_invalid_code_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            delete_code(tmp_path, proj.id, "../escape")

    def test_project_deletion_cascades(self, tmp_path: Path) -> None:
        # When a project is deleted (via delete_project), its codes go
        # with it because they live inside the project dir. Mirrors the
        # test_sources cascade assertion.
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="x")
        save_code(tmp_path, c)
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()

    def test_save_persists_full_field_set_in_file(self, tmp_path: Path) -> None:
        # Belt-and-braces: verify the on-disk JSON contains every PLANNING
        # F2.1 field. Catches accidental field drops in to_dict.
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id,
            name="Pacing",
            definition="d",
            inclusion_criteria="i",
            exclusion_criteria="e",
            exemplars=["q"],
            parent_code_id="0123456789ab",
            related_codes=[
                {"code_id": "cafebabecafe", "relation_type": "broader"},
            ],
            theoretical_memo="m",
            stage="focused",
            colour="#abcdef",
            status="draft",
            provenance={"source": "human"},
        )
        save_code(tmp_path, c)
        on_disk = json.loads(code_state_path(tmp_path, proj.id, c.id).read_text())
        for field_name in (
            "id",
            "project_id",
            "name",
            "definition",
            "inclusion_criteria",
            "exclusion_criteria",
            "exemplars",
            "parent_code_id",
            "related_codes",
            "theoretical_memo",
            "stage",
            "colour",
            "status",
            "provenance",
            "created_at",
            "modified_at",
        ):
            assert field_name in on_disk, f"missing field on disk: {field_name}"
        # Spot-check the related_codes shape.
        assert on_disk["related_codes"] == [
            {"code_id": "cafebabecafe", "relation_type": "broader"},
        ]
