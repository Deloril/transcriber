"""Tests for scribe.sources (F1.2).

These exercise the Source entity in pure Python: validation,
serialisation round-trips, partial updates, and the file-system
persistence helpers. Endpoint-level tests live in test_server.py.
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
from scribe.sources import (
    CUSTOM_ATTR_KEY_RE,
    MAX_CUSTOM_ATTR_VALUE_LEN,
    MAX_CUSTOM_ATTRS,
    MAX_LANGUAGE_LEN,
    MAX_NAME_LEN,
    MAX_NOTES_LEN,
    SOURCE_ID_RE,
    SOURCE_TYPES,
    Source,
    delete_source,
    list_sources,
    load_source,
    new_source_id,
    save_source,
    source_state_path,
    sources_dir,
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


class TestNewSourceId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert SOURCE_ID_RE.match(new_source_id())

    def test_unique(self) -> None:
        ids = {new_source_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Source.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestSourceNew:
    def test_minimal(self) -> None:
        s = Source.new(project_id="aaaaaaaaaaaa", name="Interview 1")
        assert s.name == "Interview 1"
        assert s.project_id == "aaaaaaaaaaaa"
        assert s.id and SOURCE_ID_RE.match(s.id)
        assert s.source_type == "transcript"
        assert s.transcript_job_id is None
        assert s.language == ""
        assert s.recording_date == ""
        assert s.notes == ""
        assert s.custom_attributes == {}
        assert s.created_at == s.modified_at
        assert s.created_at  # non-empty

    def test_strips_name_whitespace(self) -> None:
        s = Source.new(project_id="aaaaaaaaaaaa", name="  trimmed  ")
        assert s.name == "trimmed"

    def test_blank_name_rejected(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(ProjectValidationError):
                Source.new(project_id="aaaaaaaaaaaa", name=bad)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(project_id="aaaaaaaaaaaa", name="x" * (MAX_NAME_LEN + 1))

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(project_id="UPPERCASE123", name="ok")
        with pytest.raises(ProjectValidationError):
            Source.new(project_id="../escape", name="ok")
        with pytest.raises(ProjectValidationError):
            Source.new(project_id="short", name="ok")

    def test_invalid_source_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                source_id="UPPERCASE123",
            )

    @pytest.mark.parametrize("kind", SOURCE_TYPES)
    def test_each_source_type_accepted(self, kind: str) -> None:
        s = Source.new(project_id="aaaaaaaaaaaa", name="ok", source_type=kind)
        assert s.source_type == kind

    def test_invalid_source_type_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                source_type="podcast",
            )

    def test_transcript_job_id_validated(self) -> None:
        # Good: 12-char hex.
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            transcript_job_id="0123456789ab",
        )
        assert s.transcript_job_id == "0123456789ab"

    def test_transcript_job_id_bad_shape_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                transcript_job_id="../etc",
            )
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                transcript_job_id="UPPERCASE123",
            )

    def test_transcript_job_id_empty_string_becomes_none(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa", name="ok", transcript_job_id=""
        )
        # "" is falsy → cleared to None.
        assert s.transcript_job_id is None

    def test_language_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                language="x" * (MAX_LANGUAGE_LEN + 1),
            )

    def test_language_bcp47_accepted(self) -> None:
        # We don't lock to ISO-639-1 alone; "en-US" / "zh-Hant" are common.
        for lang in ("en", "en-US", "zh-Hant", ""):
            s = Source.new(
                project_id="aaaaaaaaaaaa", name="ok", language=lang
            )
            assert s.language == lang

    def test_recording_date_valid(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa", name="ok", recording_date="2024-03-15"
        )
        assert s.recording_date == "2024-03-15"

    def test_recording_date_empty_ok(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa", name="ok", recording_date=""
        )
        assert s.recording_date == ""

    @pytest.mark.parametrize("bad", [
        "2024/03/15",     # wrong separator
        "15-03-2024",     # wrong order
        "2024-3-15",      # not zero-padded
        "2024-13-15",     # bad month
        "2024-03-32",     # bad day
        "not-a-date",
        "2024-00-15",     # zero month
        "2024-03-00",     # zero day
    ])
    def test_recording_date_invalid_rejected(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa", name="ok", recording_date=bad
            )

    def test_notes_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                notes="x" * (MAX_NOTES_LEN + 1),
            )

    def test_custom_attributes_accepted(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            custom_attributes={
                "site": "Hospital A",
                "round": "2",
                "interviewer initials": "LP",
            },
        )
        assert s.custom_attributes == {
            "site": "Hospital A",
            "round": "2",
            "interviewer initials": "LP",
        }

    def test_custom_attributes_drops_blank_keys(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            custom_attributes={"  ": "ignored", "site": "A"},
        )
        assert s.custom_attributes == {"site": "A"}

    def test_custom_attributes_coerces_values_to_str(self) -> None:
        # JSON only carries strings, but the dataclass shouldn't blow up
        # if a Python caller passes ints (numeric attrs are common).
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            custom_attributes={"round": 2, "active": True},
        )
        assert s.custom_attributes == {"round": "2", "active": "True"}

    def test_custom_attributes_rejects_bad_keys(self) -> None:
        for bad_key in (
            "1leading_digit",
            "has/slash",
            "has.dot",
            "has\\backslash",
            "has\nnewline",
            "x" * 100,  # too long
        ):
            with pytest.raises(ProjectValidationError):
                Source.new(
                    project_id="aaaaaaaaaaaa",
                    name="ok",
                    custom_attributes={bad_key: "v"},
                )

    def test_custom_attributes_value_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                custom_attributes={"k": "x" * (MAX_CUSTOM_ATTR_VALUE_LEN + 1)},
            )

    def test_custom_attributes_too_many_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                custom_attributes={f"k{i}": "v" for i in range(MAX_CUSTOM_ATTRS + 1)},
            )

    def test_explicit_source_id(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            source_id="bbbbbbbbbbbb",
        )
        assert s.id == "bbbbbbbbbbbb"

    def test_explicit_now_used(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            now="2024-01-01T00:00:00.000000Z",
        )
        assert s.created_at == "2024-01-01T00:00:00.000000Z"
        assert s.modified_at == "2024-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Custom attribute key regex
# --------------------------------------------------------------------------- #


class TestCustomAttrKeyRegex:
    @pytest.mark.parametrize("good", [
        "site", "Site", "interview_round", "round-2", "interview round",
        "AbC_123-x y",
    ])
    def test_accepts_good(self, good: str) -> None:
        assert CUSTOM_ATTR_KEY_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "", " starts with space", "1starts_with_digit", "has.dot",
        "has/slash", "has\nnewline",
    ])
    def test_rejects_bad(self, bad: str) -> None:
        assert not CUSTOM_ATTR_KEY_RE.match(bad)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="Interview 7",
            source_type="transcript",
            transcript_job_id="0123456789ab",
            language="en-US",
            recording_date="2024-04-01",
            notes="Recorded over Zoom; some background noise.",
            custom_attributes={"site": "B", "round": "1"},
        )
        d = s.to_dict()
        assert json.dumps(d)  # JSON-serialisable
        s2 = Source.from_dict(d)
        assert s2.to_dict() == d

    def test_from_dict_requires_required_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.from_dict({"name": "x", "project_id": "aaaaaaaaaaaa"})  # no id
        with pytest.raises(ProjectValidationError):
            Source.from_dict({"id": "bbbbbbbbbbbb", "name": "x"})  # no project_id
        with pytest.raises(ProjectValidationError):
            Source.from_dict(
                {"id": "bbbbbbbbbbbb", "project_id": "aaaaaaaaaaaa"}
            )  # no name

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Source.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_defaults_missing_fields(self) -> None:
        s = Source.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "name": "ok",
        })
        assert s.source_type == "transcript"
        assert s.transcript_job_id is None
        assert s.language == ""
        assert s.recording_date == ""
        assert s.custom_attributes == {}

    def test_from_dict_treats_falsy_transcript_job_id_as_none(self) -> None:
        s = Source.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "name": "ok",
            "transcript_job_id": "",  # null-equivalent on the wire
        })
        assert s.transcript_job_id is None


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def _fresh(self) -> Source:
        return Source.new(
            project_id="aaaaaaaaaaaa",
            name="Old",
            now="2024-01-01T00:00:00.000000Z",
        )

    def test_updates_name_and_advances_modified_at(self) -> None:
        s = self._fresh()
        s.apply_update({"name": "New"}, now="2024-06-01T00:00:00.000000Z")
        assert s.name == "New"
        assert s.created_at == "2024-01-01T00:00:00.000000Z"
        assert s.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_updates_source_type(self) -> None:
        s = self._fresh()
        s.apply_update({"source_type": "field_notes"})
        assert s.source_type == "field_notes"

    def test_updates_transcript_job_id(self) -> None:
        s = self._fresh()
        s.apply_update({"transcript_job_id": "0123456789ab"})
        assert s.transcript_job_id == "0123456789ab"

    def test_clears_transcript_job_id_via_empty(self) -> None:
        s = self._fresh()
        s.transcript_job_id = "0123456789ab"
        s.apply_update({"transcript_job_id": ""})
        assert s.transcript_job_id is None

    def test_clears_transcript_job_id_via_null(self) -> None:
        s = self._fresh()
        s.transcript_job_id = "0123456789ab"
        s.apply_update({"transcript_job_id": None})
        assert s.transcript_job_id is None

    def test_updates_custom_attributes_replaces_dict(self) -> None:
        s = self._fresh()
        s.apply_update({"custom_attributes": {"a": "1"}})
        assert s.custom_attributes == {"a": "1"}
        # Subsequent update fully replaces — it's not a merge.
        s.apply_update({"custom_attributes": {"b": "2"}})
        assert s.custom_attributes == {"b": "2"}

    def test_unknown_fields_rejected(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update({"random_thing": 1})

    def test_id_in_patch_ignored(self) -> None:
        s = self._fresh()
        original = s.id
        s.apply_update({"id": "ffffffffffff", "name": "renamed"})
        assert s.id == original

    def test_project_id_in_patch_ignored(self) -> None:
        s = self._fresh()
        original = s.project_id
        s.apply_update({"project_id": "ffffffffffff", "name": "renamed"})
        assert s.project_id == original

    def test_failed_validation_does_not_advance_clock(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update(
                {"source_type": "bogus"},
                now="2099-01-01T00:00:00.000000Z",
            )
        assert s.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_non_dict_patch_rejected(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update("not a dict")  # type: ignore[arg-type]

    def test_custom_attributes_must_be_dict(self) -> None:
        s = self._fresh()
        with pytest.raises(ProjectValidationError):
            s.apply_update({"custom_attributes": ["nope"]})


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="One", language="en")
        path = save_source(tmp_path, s)
        assert path.exists()
        assert path == source_state_path(tmp_path, proj.id, s.id)

        loaded = load_source(tmp_path, proj.id, s.id)
        assert loaded.to_dict() == s.to_dict()

    def test_save_creates_sources_subdir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="One")
        save_source(tmp_path, s)
        assert sources_dir(tmp_path, proj.id).is_dir()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="ok")
        save_source(tmp_path, s)
        sd = sources_dir(tmp_path, proj.id)
        assert not (sd / f"{s.id}.json.tmp").exists()
        assert (sd / f"{s.id}.json").exists()

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        # Don't save_project — directory does not exist.
        s = Source.new(project_id="aaaaaaaaaaaa", name="orphan")
        with pytest.raises(FileNotFoundError):
            save_source(tmp_path, s)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_source(tmp_path, proj.id, "bbbbbbbbbbbb")

    def test_load_validates_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            load_source(tmp_path, proj.id, "../etc/passwd")

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_sources(tmp_path, proj.id) == []

    def test_list_no_project_dir(self, tmp_path: Path) -> None:
        # No save_project; sources_dir won't exist.
        assert list_sources(tmp_path, "aaaaaaaaaaaa") == []

    def test_list_skips_stray_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sd = sources_dir(tmp_path, proj.id)
        sd.mkdir()
        # Wrong id shape: dropped.
        (sd / "not-a-source.json").write_text("{}")
        # Valid id but corrupt JSON: dropped.
        (sd / "aaaaaaaaaaaa.json").write_text("not json")
        # Tmp file: dropped.
        (sd / "bbbbbbbbbbbb.json.tmp").write_text("{}")
        assert list_sources(tmp_path, proj.id) == []

    def test_list_sorted_by_created_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Source.new(
            project_id=proj.id, name="A", now="2024-01-01T00:00:00.000000Z"
        )
        b = Source.new(
            project_id=proj.id, name="B", now="2024-02-01T00:00:00.000000Z"
        )
        c = Source.new(
            project_id=proj.id, name="C", now="2024-03-01T00:00:00.000000Z"
        )
        # Save in a deliberately scrambled order.
        save_source(tmp_path, b)
        save_source(tmp_path, a)
        save_source(tmp_path, c)
        names = [s.name for s in list_sources(tmp_path, proj.id)]
        assert names == ["A", "B", "C"]

    def test_save_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="ok")
        s.name = ""  # corrupt directly to bypass apply_update
        with pytest.raises(ProjectValidationError):
            save_source(tmp_path, s)
        # Nothing got written.
        assert not source_state_path(tmp_path, proj.id, s.id).exists()

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="Doomed")
        save_source(tmp_path, s)
        assert source_state_path(tmp_path, proj.id, s.id).exists()
        assert delete_source(tmp_path, proj.id, s.id) is True
        assert not source_state_path(tmp_path, proj.id, s.id).exists()
        # Idempotent.
        assert delete_source(tmp_path, proj.id, s.id) is False

    def test_delete_invalid_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            delete_source(tmp_path, proj.id, "../escape")

    def test_project_deletion_cascades(self, tmp_path: Path) -> None:
        # When a project is deleted (via delete_project), its sources go
        # with it because they live inside the project dir. We don't
        # unit-test delete_project here but assert the file layout the
        # cascade depends on.
        proj = _saved_project(tmp_path)
        s = Source.new(project_id=proj.id, name="x")
        save_source(tmp_path, s)
        from scribe.projects import delete_project, project_dir
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()
