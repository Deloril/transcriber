"""Tests for scribe.source_schema (F3.2).

These exercise the AttributeDefinition / SourceAttributeSchema entities
in pure Python: validation, serialisation round-trips, partial updates,
value coercion + cross-validation, and the file-system persistence
helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.source_schema import (
    ATTRIBUTE_TYPES,
    MAX_ATTRIBUTES,
    MAX_DESCRIPTION_LEN,
    MAX_LABEL_LEN,
    MAX_OPTIONS,
    SCHEMA_FILENAME,
    AttributeDefinition,
    SourceAttributeSchema,
    coerce_attributes,
    coerce_value,
    delete_source_schema,
    load_or_empty_source_schema,
    load_source_schema,
    save_source_schema,
    source_schema_path,
    validate_attributes,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _basic_schema(project_id: str) -> SourceAttributeSchema:
    return SourceAttributeSchema.new(
        project_id=project_id,
        attributes=[
            AttributeDefinition(key="site", label="Site", type="text"),
            AttributeDefinition(key="round", label="Round", type="number"),
        ],
    )


# --------------------------------------------------------------------------- #
# AttributeDefinition validation
# --------------------------------------------------------------------------- #


class TestAttributeDefinition:
    def test_minimal(self) -> None:
        a = AttributeDefinition(key="site")
        a.validate()
        assert a.key == "site"
        assert a.type == "text"
        assert a.required is False
        assert a.options == []

    def test_strips_label_whitespace(self) -> None:
        a = AttributeDefinition(key="site", label="  Site Name  ")
        a.validate()
        assert a.label == "Site Name"

    def test_blank_key_rejected(self) -> None:
        for bad in ("", "   ", "\t"):
            with pytest.raises(ProjectValidationError):
                AttributeDefinition(key=bad).validate()

    def test_invalid_key_chars_rejected(self) -> None:
        for bad in ("1leading_digit", "has.dot", "has/slash", "has\nnewline"):
            with pytest.raises(ProjectValidationError):
                AttributeDefinition(key=bad).validate()

    def test_label_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(
                key="site", label="x" * (MAX_LABEL_LEN + 1)
            ).validate()

    @pytest.mark.parametrize("good", ATTRIBUTE_TYPES)
    def test_each_type_accepted(self, good: str) -> None:
        opts = ["A"] if good == "select" else []
        a = AttributeDefinition(key="x", type=good, options=opts)
        a.validate()
        assert a.type == good

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(key="x", type="ooga-booga").validate()

    def test_select_requires_options(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(key="site", type="select").validate()

    def test_options_only_for_select(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(
                key="site", type="text", options=["a", "b"]
            ).validate()

    def test_options_dedupe_and_strip(self) -> None:
        a = AttributeDefinition(
            key="site",
            type="select",
            options=["A", "  B  ", "A", "", "B"],
        )
        a.validate()
        assert a.options == ["A", "B"]

    def test_too_many_options_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(
                key="site",
                type="select",
                options=[f"opt{i}" for i in range(MAX_OPTIONS + 1)],
            ).validate()

    def test_description_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition(
                key="x", description="x" * (MAX_DESCRIPTION_LEN + 1)
            ).validate()

    def test_round_trip(self) -> None:
        a = AttributeDefinition(
            key="round",
            label="Round",
            type="select",
            required=True,
            options=["1", "2", "3"],
            description="Interview round",
        )
        a.validate()
        d = a.to_dict()
        assert json.dumps(d)
        a2 = AttributeDefinition.from_dict(d)
        assert a2.to_dict() == d

    def test_from_dict_requires_key(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition.from_dict({"label": "x"})

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            AttributeDefinition.from_dict("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SourceAttributeSchema construction + validation
# --------------------------------------------------------------------------- #


class TestSchemaNew:
    def test_minimal(self) -> None:
        s = SourceAttributeSchema.new(project_id="aaaaaaaaaaaa")
        assert s.project_id == "aaaaaaaaaaaa"
        assert s.attributes == []
        assert s.created_at == s.modified_at
        assert s.created_at  # non-empty

    def test_accepts_attribute_dicts_and_objects(self) -> None:
        s = SourceAttributeSchema.new(
            project_id="aaaaaaaaaaaa",
            attributes=[
                {"key": "site", "type": "text"},
                AttributeDefinition(key="round", type="number"),
            ],
        )
        assert [a.key for a in s.attributes] == ["site", "round"]

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.new(project_id="UPPERCASE123")
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.new(project_id="../escape")

    def test_too_many_attributes_rejected(self) -> None:
        attrs = [
            AttributeDefinition(key=f"col{i}")
            for i in range(MAX_ATTRIBUTES + 1)
        ]
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.new(
                project_id="aaaaaaaaaaaa", attributes=attrs
            )

    def test_duplicate_keys_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.new(
                project_id="aaaaaaaaaaaa",
                attributes=[
                    AttributeDefinition(key="site"),
                    AttributeDefinition(key="site"),
                ],
            )

    def test_explicit_now_used(self) -> None:
        s = SourceAttributeSchema.new(
            project_id="aaaaaaaaaaaa", now="2024-01-01T00:00:00.000000Z"
        )
        assert s.created_at == "2024-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves(self) -> None:
        s = _basic_schema("aaaaaaaaaaaa")
        d = s.to_dict()
        assert json.dumps(d)
        s2 = SourceAttributeSchema.from_dict(d)
        assert s2.to_dict() == d

    def test_from_dict_requires_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.from_dict({"attributes": []})

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_attributes_must_be_list(self) -> None:
        with pytest.raises(ProjectValidationError):
            SourceAttributeSchema.from_dict(
                {"project_id": "aaaaaaaaaaaa", "attributes": {}}
            )


# --------------------------------------------------------------------------- #
# apply_update / mutators
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def _fresh(self) -> SourceAttributeSchema:
        return SourceAttributeSchema.new(
            project_id="aaaaaaaaaaaa",
            now="2024-01-01T00:00:00.000000Z",
        )

    def test_replace_attributes(self) -> None:
        s = self._fresh()
        s.apply_update(
            {"attributes": [{"key": "site"}, {"key": "round", "type": "number"}]},
            now="2024-06-01T00:00:00.000000Z",
        )
        assert [a.key for a in s.attributes] == ["site", "round"]
        assert s.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_unknown_fields_rejected(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update({"random_thing": 1})

    def test_attributes_must_be_list(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update({"attributes": {"site": 1}})

    def test_failed_update_does_not_advance_clock(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update(
                {"attributes": [{"key": "1bad"}]},
                now="2099-01-01T00:00:00.000000Z",
            )
        assert s.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_project_id_in_patch_ignored(self) -> None:
        s = self._fresh()
        s.apply_update({"project_id": "ffffffffffff"})
        assert s.project_id == "aaaaaaaaaaaa"

    def test_non_dict_patch_rejected(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update("not a dict")  # type: ignore[arg-type]

    def test_add_attribute(self) -> None:
        s = self._fresh()
        a = s.add_attribute({"key": "site"})
        assert a.key == "site"
        assert s.attributes[0].key == "site"

    def test_add_attribute_duplicate_rejected(self) -> None:
        s = self._fresh()
        s.add_attribute({"key": "site"})
        with pytest.raises(ProjectValidationError):
            s.add_attribute({"key": "site"})

    def test_add_attribute_rejects_garbage(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.add_attribute("nope")  # type: ignore[arg-type]

    def test_remove_attribute(self) -> None:
        s = self._fresh()
        s.add_attribute({"key": "site"})
        assert s.remove_attribute("site") is True
        assert s.attributes == []
        # Idempotent.
        assert s.remove_attribute("site") is False

    def test_by_key_and_keys(self) -> None:
        s = _basic_schema("aaaaaaaaaaaa")
        assert s.keys() == ["site", "round"]
        assert s.by_key("site").label == "Site"
        assert s.by_key("missing") is None


# --------------------------------------------------------------------------- #
# coerce_value
# --------------------------------------------------------------------------- #


class TestCoerceValue:
    def test_text(self) -> None:
        assert coerce_value("  hello  ", "text") == "hello"
        assert coerce_value(None, "text") == ""
        assert coerce_value("", "text") == ""

    def test_number_int(self) -> None:
        assert coerce_value("5", "number") == "5"
        assert coerce_value("-3", "number") == "-3"
        assert coerce_value("+7", "number") == "7"

    def test_number_float(self) -> None:
        assert coerce_value("3.14", "number") == "3.14"
        assert coerce_value("-2.5", "number") == "-2.5"

    def test_number_invalid(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_value("five", "number")

    def test_number_nan_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_value("nan", "number")
        with pytest.raises(ProjectValidationError):
            coerce_value("inf", "number")
        with pytest.raises(ProjectValidationError):
            coerce_value("-inf", "number")

    def test_date_valid(self) -> None:
        assert coerce_value("2024-03-15", "date") == "2024-03-15"

    @pytest.mark.parametrize(
        "bad",
        [
            "2024/03/15",
            "15-03-2024",
            "2024-13-01",
            "2024-03-32",
            "not-a-date",
            "2024-00-15",
        ],
    )
    def test_date_invalid(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_value(bad, "date")

    @pytest.mark.parametrize("v", ["true", "True", "yes", "1", "Y", "t"])
    def test_boolean_true(self, v: str) -> None:
        assert coerce_value(v, "boolean") == "true"

    @pytest.mark.parametrize("v", ["false", "False", "no", "0", "N", "f"])
    def test_boolean_false(self, v: str) -> None:
        assert coerce_value(v, "boolean") == "false"

    def test_boolean_invalid(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_value("maybe", "boolean")

    def test_select_passthrough(self) -> None:
        # coerce_value just normalises whitespace; option-membership is
        # checked at the schema layer.
        assert coerce_value("  Hospital A  ", "select") == "Hospital A"

    def test_unknown_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_value("x", "ooga")

    def test_non_string_input_coerced(self) -> None:
        assert coerce_value(7, "number") == "7"
        assert coerce_value(True, "boolean") == "true"


# --------------------------------------------------------------------------- #
# validate_attributes
# --------------------------------------------------------------------------- #


class TestValidateAttributes:
    def _schema(self) -> SourceAttributeSchema:
        return SourceAttributeSchema.new(
            project_id="aaaaaaaaaaaa",
            attributes=[
                AttributeDefinition(key="site", type="select",
                                    options=["A", "B"], required=True),
                AttributeDefinition(key="round", type="number"),
                AttributeDefinition(key="active", type="boolean"),
                AttributeDefinition(key="when", type="date"),
            ],
        )

    def test_passes_valid(self) -> None:
        errors = validate_attributes(
            {
                "site": "A",
                "round": "2",
                "active": "yes",
                "when": "2024-04-01",
            },
            self._schema(),
        )
        assert errors == []

    def test_required_missing(self) -> None:
        errors = validate_attributes({"round": "1"}, self._schema())
        assert any("'site' is required" in e for e in errors)

    def test_required_empty(self) -> None:
        errors = validate_attributes({"site": ""}, self._schema())
        assert any("'site' is required" in e for e in errors)

    def test_select_value_not_in_options(self) -> None:
        errors = validate_attributes({"site": "C"}, self._schema())
        assert any("not in options" in e for e in errors)

    def test_number_not_a_number(self) -> None:
        errors = validate_attributes(
            {"site": "A", "round": "lots"}, self._schema()
        )
        assert any("'round'" in e and "number" in e for e in errors)

    def test_unknown_key_lenient(self) -> None:
        errors = validate_attributes(
            {"site": "A", "ad_hoc": "extra"}, self._schema()
        )
        assert errors == []

    def test_unknown_key_strict(self) -> None:
        errors = validate_attributes(
            {"site": "A", "ad_hoc": "extra"}, self._schema(), strict=True
        )
        assert any("Unknown attribute key" in e for e in errors)

    def test_non_dict_input(self) -> None:
        errors = validate_attributes(["not", "a", "dict"], self._schema())  # type: ignore[arg-type]
        assert errors == ["custom_attributes must be a dict"]


# --------------------------------------------------------------------------- #
# coerce_attributes
# --------------------------------------------------------------------------- #


class TestCoerceAttributes:
    def _schema(self) -> SourceAttributeSchema:
        return SourceAttributeSchema.new(
            project_id="aaaaaaaaaaaa",
            attributes=[
                AttributeDefinition(key="site", type="select", options=["A"]),
                AttributeDefinition(key="round", type="number"),
                AttributeDefinition(key="active", type="boolean"),
            ],
        )

    def test_canonicalises(self) -> None:
        out = coerce_attributes(
            {"site": "A", "round": "2", "active": "yes"}, self._schema()
        )
        assert out == {"site": "A", "round": "2", "active": "true"}

    def test_passes_through_unknown_keys_lenient(self) -> None:
        out = coerce_attributes(
            {"site": "A", "ad_hoc": "extra"}, self._schema()
        )
        assert out == {"site": "A", "ad_hoc": "extra"}

    def test_strict_rejects_unknown(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_attributes(
                {"site": "A", "ad_hoc": "extra"},
                self._schema(),
                strict=True,
            )

    def test_validation_failure_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            coerce_attributes(
                {"site": "C"},  # not in options
                self._schema(),
            )


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _basic_schema(proj.id)
        path = save_source_schema(tmp_path, s)
        assert path.exists()
        assert path == source_schema_path(tmp_path, proj.id)
        assert path.name == SCHEMA_FILENAME

        loaded = load_source_schema(tmp_path, proj.id)
        assert loaded.to_dict() == s.to_dict()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _basic_schema(proj.id)
        save_source_schema(tmp_path, s)
        target = source_schema_path(tmp_path, proj.id)
        assert target.exists()
        assert not target.with_suffix(".json.tmp").exists()

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        s = _basic_schema("aaaaaaaaaaaa")
        with pytest.raises(FileNotFoundError):
            save_source_schema(tmp_path, s)

    def test_save_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _basic_schema(proj.id)
        # Corrupt the schema directly to bypass apply_update.
        s.attributes.append(AttributeDefinition(key="site"))  # duplicate
        with pytest.raises(ProjectValidationError):
            save_source_schema(tmp_path, s)
        # Nothing was written.
        assert not source_schema_path(tmp_path, proj.id).exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_source_schema(tmp_path, proj.id)

    def test_load_or_empty_returns_empty_when_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = load_or_empty_source_schema(tmp_path, proj.id)
        assert s.project_id == proj.id
        assert s.attributes == []

    def test_load_or_empty_returns_loaded_when_present(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        original = _basic_schema(proj.id)
        save_source_schema(tmp_path, original)
        loaded = load_or_empty_source_schema(tmp_path, proj.id)
        assert loaded.to_dict() == original.to_dict()

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _basic_schema(proj.id)
        save_source_schema(tmp_path, s)
        assert delete_source_schema(tmp_path, proj.id) is True
        # Idempotent.
        assert delete_source_schema(tmp_path, proj.id) is False

    def test_project_deletion_cascades(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_source_schema(tmp_path, _basic_schema(proj.id))
        from scribe.projects import delete_project, project_dir
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()
