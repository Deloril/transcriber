"""Tests for scribe.participants (F1.3).

These exercise the Participant entity in pure Python: validation,
serialisation round-trips, partial updates, source-link helpers, and
the file-system persistence helpers. Endpoint-level tests live in
test_server.py.
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
from scribe.participants import (
    DEMOGRAPHIC_KEY_RE,
    MAX_DEMOGRAPHIC_VALUE_LEN,
    MAX_DEMOGRAPHICS,
    MAX_NAME_LEN,
    MAX_NOTES_LEN,
    MAX_PSEUDONYM_LEN,
    MAX_SOURCE_LINKS,
    PARTICIPANT_ID_RE,
    Participant,
    delete_participant,
    list_participants,
    load_participant,
    new_participant_id,
    participant_state_path,
    participants_dir,
    save_participant,
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


class TestNewParticipantId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert PARTICIPANT_ID_RE.match(new_participant_id())

    def test_unique(self) -> None:
        ids = {new_participant_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Participant.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestParticipantNew:
    def test_minimal(self) -> None:
        p = Participant.new(project_id="aaaaaaaaaaaa", name="P01")
        assert p.name == "P01"
        assert p.project_id == "aaaaaaaaaaaa"
        assert p.id and PARTICIPANT_ID_RE.match(p.id)
        assert p.pseudonym == ""
        assert p.demographics == {}
        assert p.notes == ""
        assert p.source_ids == []
        assert p.created_at == p.modified_at
        assert p.created_at  # non-empty

    def test_strips_name_whitespace(self) -> None:
        p = Participant.new(project_id="aaaaaaaaaaaa", name="  trimmed  ")
        assert p.name == "trimmed"

    def test_strips_pseudonym_whitespace(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa", name="P01", pseudonym="  Anon  "
        )
        assert p.pseudonym == "Anon"

    def test_blank_name_rejected(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(ProjectValidationError):
                Participant.new(project_id="aaaaaaaaaaaa", name=bad)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa", name="x" * (MAX_NAME_LEN + 1)
            )

    def test_pseudonym_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="P01",
                pseudonym="x" * (MAX_PSEUDONYM_LEN + 1),
            )

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(project_id="UPPERCASE123", name="ok")
        with pytest.raises(ProjectValidationError):
            Participant.new(project_id="../escape", name="ok")
        with pytest.raises(ProjectValidationError):
            Participant.new(project_id="short", name="ok")

    def test_invalid_participant_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                participant_id="UPPERCASE123",
            )

    def test_explicit_participant_id(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            participant_id="bbbbbbbbbbbb",
        )
        assert p.id == "bbbbbbbbbbbb"

    def test_explicit_now_used(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            now="2024-01-01T00:00:00.000000Z",
        )
        assert p.created_at == "2024-01-01T00:00:00.000000Z"
        assert p.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_notes_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                notes="x" * (MAX_NOTES_LEN + 1),
            )

    def test_demographics_accepted(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            demographics={
                "age band": "30-39",
                "role": "nurse",
                "site": "Hospital A",
            },
        )
        assert p.demographics == {
            "age band": "30-39",
            "role": "nurse",
            "site": "Hospital A",
        }

    def test_demographics_drops_blank_keys(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            demographics={"  ": "ignored", "role": "nurse"},
        )
        assert p.demographics == {"role": "nurse"}

    def test_demographics_coerces_values_to_str(self) -> None:
        # JSON only carries strings, but the dataclass shouldn't blow up
        # if a Python caller passes ints (numeric attrs are common).
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            demographics={"age": 42, "active": True},
        )
        assert p.demographics == {"age": "42", "active": "True"}

    def test_demographics_rejects_bad_keys(self) -> None:
        for bad_key in (
            "1leading_digit",
            "has/slash",
            "has.dot",
            "has\\backslash",
            "has\nnewline",
            "x" * 100,  # too long
        ):
            with pytest.raises(ProjectValidationError):
                Participant.new(
                    project_id="aaaaaaaaaaaa",
                    name="ok",
                    demographics={bad_key: "v"},
                )

    def test_demographics_value_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                demographics={"k": "x" * (MAX_DEMOGRAPHIC_VALUE_LEN + 1)},
            )

    def test_demographics_too_many_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                demographics={
                    f"k{i}": "v" for i in range(MAX_DEMOGRAPHICS + 1)
                },
            )

    def test_source_ids_accepted(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            source_ids=["0123456789ab", "fedcba987654"],
        )
        assert p.source_ids == ["0123456789ab", "fedcba987654"]

    def test_source_ids_dedup_preserves_order(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            source_ids=[
                "0123456789ab",
                "fedcba987654",
                "0123456789ab",  # dup
            ],
        )
        assert p.source_ids == ["0123456789ab", "fedcba987654"]

    def test_source_ids_drops_blank(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            source_ids=["", "0123456789ab", "  "],
        )
        assert p.source_ids == ["0123456789ab"]

    def test_source_ids_rejects_bad_shape(self) -> None:
        for bad in ("UPPERCASE123", "../escape", "short"):
            with pytest.raises(ProjectValidationError):
                Participant.new(
                    project_id="aaaaaaaaaaaa",
                    name="P01",
                    source_ids=[bad],
                )

    def test_source_ids_must_be_list(self) -> None:
        # The dataclass takes Iterable[str] so a string would zip into
        # individual chars and fail; document the explicit-list expectation.
        # (Calling with a non-iterable directly bypasses .new but still
        # gets caught at validate().)
        p = Participant(
            id="aaaaaaaaaaaa",
            project_id="aaaaaaaaaaaa",
            name="ok",
        )
        p.source_ids = "not-a-list"  # type: ignore[assignment]
        with pytest.raises(ProjectValidationError):
            p.validate()

    def test_source_ids_too_many_rejected(self) -> None:
        # Pre-build a list of distinct 12-char hex ids over the limit.
        ids = [f"{i:012x}" for i in range(MAX_SOURCE_LINKS + 1)]
        with pytest.raises(ProjectValidationError):
            Participant.new(
                project_id="aaaaaaaaaaaa",
                name="P01",
                source_ids=ids,
            )


# --------------------------------------------------------------------------- #
# Demographic key regex
# --------------------------------------------------------------------------- #


class TestDemographicKeyRegex:
    @pytest.mark.parametrize("good", [
        "age", "Age", "age_band", "age-band", "age band",
        "AbC_123-x y",
    ])
    def test_accepts_good(self, good: str) -> None:
        assert DEMOGRAPHIC_KEY_RE.match(good)

    @pytest.mark.parametrize("bad", [
        "", " starts with space", "1starts_with_digit", "has.dot",
        "has/slash", "has\nnewline",
    ])
    def test_rejects_bad(self, bad: str) -> None:
        assert not DEMOGRAPHIC_KEY_RE.match(bad)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        p = Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P03",
            pseudonym="Anon C",
            demographics={"age band": "40-49", "role": "consultant"},
            notes="Senior clinician.",
            source_ids=["0123456789ab", "fedcba987654"],
        )
        d = p.to_dict()
        assert json.dumps(d)  # JSON-serialisable
        p2 = Participant.from_dict(d)
        assert p2.to_dict() == d

    def test_from_dict_requires_required_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.from_dict(
                {"name": "x", "project_id": "aaaaaaaaaaaa"}
            )  # no id
        with pytest.raises(ProjectValidationError):
            Participant.from_dict(
                {"id": "bbbbbbbbbbbb", "name": "x"}
            )  # no project_id
        with pytest.raises(ProjectValidationError):
            Participant.from_dict(
                {"id": "bbbbbbbbbbbb", "project_id": "aaaaaaaaaaaa"}
            )  # no name

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Participant.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_defaults_missing_fields(self) -> None:
        p = Participant.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "name": "ok",
        })
        assert p.pseudonym == ""
        assert p.demographics == {}
        assert p.notes == ""
        assert p.source_ids == []


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def _fresh(self) -> Participant:
        return Participant.new(
            project_id="aaaaaaaaaaaa",
            name="Old",
            now="2024-01-01T00:00:00.000000Z",
        )

    def test_updates_name_and_advances_modified_at(self) -> None:
        p = self._fresh()
        p.apply_update({"name": "New"}, now="2024-06-01T00:00:00.000000Z")
        assert p.name == "New"
        assert p.created_at == "2024-01-01T00:00:00.000000Z"
        assert p.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_updates_pseudonym(self) -> None:
        p = self._fresh()
        p.apply_update({"pseudonym": "Anon"})
        assert p.pseudonym == "Anon"

    def test_updates_demographics_replaces_dict(self) -> None:
        p = self._fresh()
        p.apply_update({"demographics": {"role": "nurse"}})
        assert p.demographics == {"role": "nurse"}
        # Subsequent update fully replaces — it's not a merge.
        p.apply_update({"demographics": {"site": "B"}})
        assert p.demographics == {"site": "B"}

    def test_updates_source_ids(self) -> None:
        p = self._fresh()
        p.apply_update({"source_ids": ["0123456789ab"]})
        assert p.source_ids == ["0123456789ab"]

    def test_updates_notes(self) -> None:
        p = self._fresh()
        p.apply_update({"notes": "extra context"})
        assert p.notes == "extra context"

    def test_unknown_fields_rejected(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.apply_update({"random_thing": 1})

    def test_id_in_patch_ignored(self) -> None:
        p = self._fresh()
        original = p.id
        p.apply_update({"id": "ffffffffffff", "name": "renamed"})
        assert p.id == original

    def test_project_id_in_patch_ignored(self) -> None:
        p = self._fresh()
        original = p.project_id
        p.apply_update({"project_id": "ffffffffffff", "name": "renamed"})
        assert p.project_id == original

    def test_failed_validation_does_not_advance_clock(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.apply_update(
                {"source_ids": ["BADBADBADBAD"]},
                now="2099-01-01T00:00:00.000000Z",
            )
        assert p.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_non_dict_patch_rejected(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.apply_update("not a dict")  # type: ignore[arg-type]

    def test_demographics_must_be_dict(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.apply_update({"demographics": ["nope"]})

    def test_source_ids_must_be_list(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.apply_update({"source_ids": "0123456789ab"})


# --------------------------------------------------------------------------- #
# Source-link helpers
# --------------------------------------------------------------------------- #


class TestSourceLinks:
    def _fresh(self) -> Participant:
        return Participant.new(
            project_id="aaaaaaaaaaaa",
            name="P01",
            now="2024-01-01T00:00:00.000000Z",
        )

    def test_add_source_returns_true_on_new(self) -> None:
        p = self._fresh()
        added = p.add_source("0123456789ab", now="2024-02-01T00:00:00.000000Z")
        assert added is True
        assert p.source_ids == ["0123456789ab"]
        assert p.modified_at == "2024-02-01T00:00:00.000000Z"

    def test_add_source_idempotent(self) -> None:
        p = self._fresh()
        p.add_source("0123456789ab", now="2024-02-01T00:00:00.000000Z")
        again = p.add_source(
            "0123456789ab", now="2024-03-01T00:00:00.000000Z"
        )
        assert again is False
        assert p.source_ids == ["0123456789ab"]
        # No advance on a no-op.
        assert p.modified_at == "2024-02-01T00:00:00.000000Z"

    def test_add_source_validates_shape(self) -> None:
        p = self._fresh()
        with pytest.raises(ProjectValidationError):
            p.add_source("BADBAD")

    def test_remove_source(self) -> None:
        p = self._fresh()
        p.add_source("0123456789ab", now="2024-02-01T00:00:00.000000Z")
        removed = p.remove_source(
            "0123456789ab", now="2024-03-01T00:00:00.000000Z"
        )
        assert removed is True
        assert p.source_ids == []
        assert p.modified_at == "2024-03-01T00:00:00.000000Z"

    def test_remove_source_unknown_returns_false(self) -> None:
        p = self._fresh()
        out = p.remove_source(
            "0123456789ab", now="2024-03-01T00:00:00.000000Z"
        )
        assert out is False
        # No advance on a no-op.
        assert p.modified_at == "2024-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="P01", pseudonym="Anon")
        path = save_participant(tmp_path, p)
        assert path.exists()
        assert path == participant_state_path(tmp_path, proj.id, p.id)

        loaded = load_participant(tmp_path, proj.id, p.id)
        assert loaded.to_dict() == p.to_dict()

    def test_save_creates_participants_subdir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="P01")
        save_participant(tmp_path, p)
        assert participants_dir(tmp_path, proj.id).is_dir()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="ok")
        save_participant(tmp_path, p)
        pd = participants_dir(tmp_path, proj.id)
        assert not (pd / f"{p.id}.json.tmp").exists()
        assert (pd / f"{p.id}.json").exists()

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        # Don't save_project — directory does not exist.
        p = Participant.new(project_id="aaaaaaaaaaaa", name="orphan")
        with pytest.raises(FileNotFoundError):
            save_participant(tmp_path, p)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_participant(tmp_path, proj.id, "bbbbbbbbbbbb")

    def test_load_validates_participant_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            load_participant(tmp_path, proj.id, "../etc/passwd")

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_participants(tmp_path, proj.id) == []

    def test_list_no_project_dir(self, tmp_path: Path) -> None:
        # No save_project; participants_dir won't exist.
        assert list_participants(tmp_path, "aaaaaaaaaaaa") == []

    def test_list_skips_stray_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        pd = participants_dir(tmp_path, proj.id)
        pd.mkdir()
        # Wrong id shape: dropped.
        (pd / "not-a-participant.json").write_text("{}")
        # Valid id but corrupt JSON: dropped.
        (pd / "aaaaaaaaaaaa.json").write_text("not json")
        # Tmp file: dropped.
        (pd / "bbbbbbbbbbbb.json.tmp").write_text("{}")
        assert list_participants(tmp_path, proj.id) == []

    def test_list_sorted_by_created_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Participant.new(
            project_id=proj.id,
            name="A",
            now="2024-01-01T00:00:00.000000Z",
        )
        b = Participant.new(
            project_id=proj.id,
            name="B",
            now="2024-02-01T00:00:00.000000Z",
        )
        c = Participant.new(
            project_id=proj.id,
            name="C",
            now="2024-03-01T00:00:00.000000Z",
        )
        # Save in a deliberately scrambled order.
        save_participant(tmp_path, b)
        save_participant(tmp_path, a)
        save_participant(tmp_path, c)
        names = [p.name for p in list_participants(tmp_path, proj.id)]
        assert names == ["A", "B", "C"]

    def test_save_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="ok")
        p.name = ""  # corrupt directly to bypass apply_update
        with pytest.raises(ProjectValidationError):
            save_participant(tmp_path, p)
        # Nothing got written.
        assert not participant_state_path(tmp_path, proj.id, p.id).exists()

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="Doomed")
        save_participant(tmp_path, p)
        assert participant_state_path(tmp_path, proj.id, p.id).exists()
        assert delete_participant(tmp_path, proj.id, p.id) is True
        assert not participant_state_path(tmp_path, proj.id, p.id).exists()
        # Idempotent.
        assert delete_participant(tmp_path, proj.id, p.id) is False

    def test_delete_invalid_participant_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            delete_participant(tmp_path, proj.id, "../escape")

    def test_project_deletion_cascades(self, tmp_path: Path) -> None:
        # When a project is deleted (via delete_project), its
        # participants go with it because they live inside the project
        # dir. We don't unit-test delete_project here but assert the
        # file layout the cascade depends on.
        proj = _saved_project(tmp_path)
        p = Participant.new(project_id=proj.id, name="x")
        save_participant(tmp_path, p)
        from scribe.projects import delete_project, project_dir
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()
