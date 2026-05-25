"""Tests for scribe.projects (F1.1).

These exercise the Project entity in pure Python: validation,
serialisation round-trips, partial updates, and the file-system
persistence helpers. Endpoint-level tests live in test_server.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.projects import (
    CODEBOOK_STAGES,
    MAX_NAME_LEN,
    MAX_RESEARCH_QUESTION_LEN,
    MAX_SENSITISING_CONCEPT_LEN,
    MAX_SENSITISING_CONCEPTS,
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
    delete_project,
    list_projects,
    load_project,
    new_project_id,
    project_dir,
    project_state_path,
    save_project,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewProjectId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert PROJECT_ID_RE.match(new_project_id())

    def test_unique(self) -> None:
        ids = {new_project_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# utcnow_iso
# --------------------------------------------------------------------------- #


class TestUtcNowIso:
    def test_ends_with_z(self) -> None:
        s = utcnow_iso()
        assert s.endswith("Z")

    def test_parseable(self) -> None:
        from datetime import datetime
        s = utcnow_iso()
        # Strip Z and parse to ensure it's a real ISO-8601 stamp.
        datetime.fromisoformat(s[:-1])


# --------------------------------------------------------------------------- #
# Project.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestProjectNew:
    def test_minimal(self) -> None:
        p = Project.new(name="My research")
        assert p.name == "My research"
        assert p.id and PROJECT_ID_RE.match(p.id)
        assert p.codebook_stage == "initial"
        assert p.research_question == ""
        assert p.methodology == ""
        assert p.sensitising_concepts == []
        assert p.created_at == p.modified_at
        assert p.created_at  # non-empty

    def test_strips_name_whitespace(self) -> None:
        p = Project.new(name="   spaced out   ")
        assert p.name == "spaced out"

    def test_blank_name_rejected(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(ProjectValidationError):
                Project.new(name=bad)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(name="x" * (MAX_NAME_LEN + 1))

    def test_research_question_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(name="ok", research_question="x" * (MAX_RESEARCH_QUESTION_LEN + 1))

    def test_invalid_codebook_stage_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(name="ok", codebook_stage="bogus")

    @pytest.mark.parametrize("stage", CODEBOOK_STAGES)
    def test_each_codebook_stage_accepted(self, stage: str) -> None:
        p = Project.new(name="ok", codebook_stage=stage)
        assert p.codebook_stage == stage

    def test_sensitising_concepts_strips_blanks(self) -> None:
        p = Project.new(
            name="ok",
            sensitising_concepts=["agency", "  ", "structure", "", " power  "],
        )
        assert p.sensitising_concepts == ["agency", "structure", "power"]

    def test_sensitising_concepts_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(
                name="ok",
                sensitising_concepts=["x" * (MAX_SENSITISING_CONCEPT_LEN + 1)],
            )

    def test_sensitising_concepts_too_many_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(
                name="ok",
                sensitising_concepts=[f"c{i}" for i in range(MAX_SENSITISING_CONCEPTS + 1)],
            )

    def test_explicit_project_id(self) -> None:
        p = Project.new(name="ok", project_id="aaaaaaaaaaaa")
        assert p.id == "aaaaaaaaaaaa"

    def test_explicit_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.new(name="ok", project_id="UPPERCASE123")

    def test_explicit_now_is_used(self) -> None:
        p = Project.new(name="ok", now="2024-01-01T00:00:00.000000Z")
        assert p.created_at == "2024-01-01T00:00:00.000000Z"
        assert p.modified_at == "2024-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        p = Project.new(
            name="Study A",
            research_question="How do nurses interpret consent?",
            methodology="charmaz",
            sensitising_concepts=["consent", "agency"],
            description="Pilot study.",
            codebook_stage="focused",
        )
        d = p.to_dict()
        # JSON-serialisable
        assert json.dumps(d)
        p2 = Project.from_dict(d)
        assert p2.to_dict() == d

    def test_from_dict_requires_id_and_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.from_dict({"name": "x"})  # no id
        with pytest.raises(ProjectValidationError):
            Project.from_dict({"id": "aaaaaaaaaaaa"})  # no name

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Project.from_dict("hello")  # type: ignore[arg-type]

    def test_from_dict_defaults_missing_fields(self) -> None:
        p = Project.from_dict({"id": "aaaaaaaaaaaa", "name": "ok"})
        assert p.codebook_stage == "initial"
        assert p.sensitising_concepts == []


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def test_updates_name_and_advances_modified_at(self) -> None:
        p = Project.new(name="Old", now="2024-01-01T00:00:00.000000Z")
        p.apply_update({"name": "New"}, now="2024-06-01T00:00:00.000000Z")
        assert p.name == "New"
        assert p.created_at == "2024-01-01T00:00:00.000000Z"
        assert p.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_updates_codebook_stage(self) -> None:
        p = Project.new(name="ok")
        p.apply_update({"codebook_stage": "focused"})
        assert p.codebook_stage == "focused"

    def test_unknown_fields_rejected(self) -> None:
        p = Project.new(name="ok")
        with pytest.raises(ProjectValidationError):
            p.apply_update({"random_thing": 1})

    def test_id_in_patch_is_ignored(self) -> None:
        p = Project.new(name="ok")
        original = p.id
        p.apply_update({"id": "ffffffffffff", "name": "renamed"})
        assert p.id == original

    def test_failed_validation_does_not_mutate(self) -> None:
        p = Project.new(name="ok", now="2024-01-01T00:00:00.000000Z")
        with pytest.raises(ProjectValidationError):
            p.apply_update(
                {"codebook_stage": "bogus"}, now="2099-01-01T00:00:00.000000Z"
            )
        # codebook_stage was set to the bad value, but validate() raised
        # — the question is whether modified_at advanced. It must NOT.
        assert p.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_non_dict_patch_rejected(self) -> None:
        p = Project.new(name="ok")
        with pytest.raises(ProjectValidationError):
            p.apply_update("not a dict")  # type: ignore[arg-type]

    def test_sensitising_concepts_must_be_list(self) -> None:
        p = Project.new(name="ok")
        with pytest.raises(ProjectValidationError):
            p.apply_update({"sensitising_concepts": "agency,structure"})

    def test_clearing_research_question(self) -> None:
        p = Project.new(name="ok", research_question="something")
        p.apply_update({"research_question": ""})
        assert p.research_question == ""


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        p = Project.new(name="Saved", methodology="charmaz")
        path = save_project(tmp_path, p)
        assert path.exists()
        assert path == project_state_path(tmp_path, p.id)

        loaded = load_project(tmp_path, p.id)
        assert loaded.to_dict() == p.to_dict()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        # The temp file must not linger after a successful save.
        p = Project.new(name="ok")
        save_project(tmp_path, p)
        d = project_dir(tmp_path, p.id)
        assert not (d / "project.json.tmp").exists()
        assert (d / "project.json").exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_project(tmp_path, "aaaaaaaaaaaa")

    def test_list_empty(self, tmp_path: Path) -> None:
        assert list_projects(tmp_path) == []

    def test_list_root_does_not_exist(self, tmp_path: Path) -> None:
        assert list_projects(tmp_path / "nope") == []

    def test_list_skips_non_project_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-project").mkdir()
        (tmp_path / "ZZZUPPERCASE").mkdir()
        assert list_projects(tmp_path) == []

    def test_list_skips_dir_without_project_json(self, tmp_path: Path) -> None:
        (tmp_path / "aaaaaaaaaaaa").mkdir()
        assert list_projects(tmp_path) == []

    def test_list_skips_corrupt_project_json(self, tmp_path: Path) -> None:
        d = tmp_path / "aaaaaaaaaaaa"
        d.mkdir()
        (d / "project.json").write_text("not valid json {")
        # Plus a healthy one.
        good = Project.new(name="Healthy")
        save_project(tmp_path, good)
        out = list_projects(tmp_path)
        assert [p.id for p in out] == [good.id]

    def test_list_orders_by_modified_at_desc(self, tmp_path: Path) -> None:
        a = Project.new(name="A", now="2024-01-01T00:00:00.000000Z")
        b = Project.new(name="B", now="2025-01-01T00:00:00.000000Z")
        c = Project.new(name="C", now="2026-01-01T00:00:00.000000Z")
        save_project(tmp_path, a)
        save_project(tmp_path, b)
        save_project(tmp_path, c)
        ordered = [p.name for p in list_projects(tmp_path)]
        assert ordered == ["C", "B", "A"]

    def test_save_validates(self, tmp_path: Path) -> None:
        p = Project.new(name="ok")
        p.name = ""  # corrupt directly to bypass apply_update
        with pytest.raises(ProjectValidationError):
            save_project(tmp_path, p)
        # And nothing got written.
        assert not project_state_path(tmp_path, p.id).exists()

    def test_delete(self, tmp_path: Path) -> None:
        p = Project.new(name="Doomed")
        save_project(tmp_path, p)
        assert project_dir(tmp_path, p.id).exists()
        assert delete_project(tmp_path, p.id) is True
        assert not project_dir(tmp_path, p.id).exists()
        assert delete_project(tmp_path, p.id) is False  # second call

    def test_project_dir_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            project_dir(tmp_path, "../escape")

    def test_save_persists_canonical_form(self, tmp_path: Path) -> None:
        # Whitespace in name gets stripped before write.
        p = Project.new(name="  trim me  ")
        save_project(tmp_path, p)
        on_disk = json.loads(project_state_path(tmp_path, p.id).read_text())
        assert on_disk["name"] == "trim me"
