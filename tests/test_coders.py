"""Tests for scribe.coders (F2.5).

Pure-Python tests for the Coder entity: validation, serialisation
round-trips, partial updates, and file-system persistence helpers.
ICR statistics live in test_icr.py. Endpoint-level tests will be
added once F4.1 wires applications into a multi-coder workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.coders import (
    CODER_COLOUR_RE,
    CODER_EMAIL_RE,
    CODER_ID_RE,
    CODER_ROLES,
    CODER_STATUSES,
    Coder,
    MAX_EMAIL_LEN,
    MAX_NAME_LEN,
    MAX_NOTES_LEN,
    coder_state_path,
    coders_dir,
    delete_coder,
    list_coders,
    load_coder,
    new_coder_id,
    save_coder,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
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


class TestNewCoderId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert CODER_ID_RE.match(new_coder_id())

    def test_unique(self) -> None:
        ids = {new_coder_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Coder.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestCoderNew:
    def test_minimal(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        assert c.name == "Luke"
        assert c.project_id == "aaaaaaaaaaaa"
        assert c.id and CODER_ID_RE.match(c.id)
        assert c.role == "researcher"
        assert c.email == ""
        assert c.colour == ""
        assert c.status == "active"
        assert c.notes == ""
        assert c.created_at == c.modified_at
        assert c.created_at  # non-empty

    def test_strips_name_whitespace(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="  trimmed  ")
        assert c.name == "trimmed"

    def test_strips_email_whitespace(self) -> None:
        c = Coder.new(
            project_id="aaaaaaaaaaaa", name="Luke", email="  a@b.co  "
        )
        assert c.email == "a@b.co"

    def test_blank_name_rejected(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with pytest.raises(ProjectValidationError):
                Coder.new(project_id="aaaaaaaaaaaa", name=bad)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa", name="x" * (MAX_NAME_LEN + 1)
            )

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(project_id="UPPERCASE123", name="ok")
        with pytest.raises(ProjectValidationError):
            Coder.new(project_id="../escape", name="ok")
        with pytest.raises(ProjectValidationError):
            Coder.new(project_id="short", name="ok")

    def test_invalid_coder_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                coder_id="UPPERCASE123",
            )

    def test_explicit_coder_id(self) -> None:
        c = Coder.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            coder_id="bbbbbbbbbbbb",
        )
        assert c.id == "bbbbbbbbbbbb"

    def test_explicit_now_used(self) -> None:
        c = Coder.new(
            project_id="aaaaaaaaaaaa",
            name="ok",
            now="2024-01-01T00:00:00.000000Z",
        )
        assert c.created_at == "2024-01-01T00:00:00.000000Z"
        assert c.modified_at == "2024-01-01T00:00:00.000000Z"

    @pytest.mark.parametrize("role", list(CODER_ROLES))
    def test_all_known_roles_accepted(self, role: str) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="ok", role=role)
        assert c.role == role

    def test_unknown_role_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa", name="ok", role="overlord"
            )

    @pytest.mark.parametrize("status", list(CODER_STATUSES))
    def test_all_known_statuses_accepted(self, status: str) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="ok", status=status)
        assert c.status == status

    def test_unknown_status_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa", name="ok", status="zombie"
            )

    def test_notes_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                notes="x" * (MAX_NOTES_LEN + 1),
            )

    @pytest.mark.parametrize("good", [
        "#abc", "#ABC", "#aabbcc", "#AABBCC", "#123",
    ])
    def test_colour_accepted(self, good: str) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="ok", colour=good)
        assert c.colour == good

    @pytest.mark.parametrize("bad", [
        "abc", "#a", "#ab", "#abcd", "#abcdefg", "rgba(0,0,0,1)",
        "red", "#xyz",
    ])
    def test_bad_colour_rejected(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(project_id="aaaaaaaaaaaa", name="ok", colour=bad)

    @pytest.mark.parametrize("good", [
        "a@b.co", "luke.pearson@example.org",
        "with+plus@example.co.uk", "x@y.z",
    ])
    def test_email_accepted(self, good: str) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="ok", email=good)
        assert c.email == good

    @pytest.mark.parametrize("bad", [
        "no-at-sign", "@nouser.com", "user@", "user@domain",
        "user @domain.co", "user@dom ain.co", "two@@signs.co",
    ])
    def test_bad_email_rejected(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(project_id="aaaaaaaaaaaa", name="ok", email=bad)

    def test_email_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.new(
                project_id="aaaaaaaaaaaa",
                name="ok",
                email="a@" + ("x" * MAX_EMAIL_LEN) + ".co",
            )


# --------------------------------------------------------------------------- #
# Email regex (sanity)
# --------------------------------------------------------------------------- #


class TestEmailRegex:
    def test_minimum_acceptable(self) -> None:
        assert CODER_EMAIL_RE.match("a@b.c")

    def test_rejects_no_dot_in_domain(self) -> None:
        assert not CODER_EMAIL_RE.match("a@b")


# --------------------------------------------------------------------------- #
# Colour regex (sanity)
# --------------------------------------------------------------------------- #


class TestColourRegex:
    @pytest.mark.parametrize("good", ["#000", "#fff", "#000000", "#FFFFFF"])
    def test_accepts(self, good: str) -> None:
        assert CODER_COLOUR_RE.match(good)

    @pytest.mark.parametrize(
        "bad", ["", "000", "#", "#0", "#00", "#0000", "#00000", "#0000000"]
    )
    def test_rejects(self, bad: str) -> None:
        assert not CODER_COLOUR_RE.match(bad)


# --------------------------------------------------------------------------- #
# Serialisation round-trip
# --------------------------------------------------------------------------- #


class TestSerialisation:
    def test_round_trip_minimal(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        d = c.to_dict()
        c2 = Coder.from_dict(d)
        assert c2.to_dict() == d

    def test_round_trip_full(self) -> None:
        c = Coder.new(
            project_id="aaaaaaaaaaaa",
            name="Coder B",
            role="second_coder",
            email="b@example.org",
            colour="#aabbcc",
            status="active",
            notes="trained on initial codebook 2026-04-12",
        )
        d = c.to_dict()
        c2 = Coder.from_dict(d)
        assert c2.to_dict() == d

    def test_from_dict_missing_required_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            Coder.from_dict({"name": "no id"})
        with pytest.raises(ProjectValidationError):
            Coder.from_dict({"id": "aaaaaaaaaaaa"})  # missing project_id, name

    def test_from_dict_not_an_object(self) -> None:
        for bad in ([], "str", 42, None):
            with pytest.raises(ProjectValidationError):
                Coder.from_dict(bad)  # type: ignore[arg-type]

    def test_from_dict_defaults_role_and_status(self) -> None:
        # A persisted file written by an earlier schema (before role /
        # status existed) shouldn't blow up on load.
        c = Coder.from_dict({
            "id": "aaaaaaaaaaaa",
            "project_id": "bbbbbbbbbbbb",
            "name": "Legacy",
        })
        assert c.role == "researcher"
        assert c.status == "active"


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def test_update_changes_name(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        before = c.modified_at
        c.apply_update({"name": "Lucas"}, now="2025-01-01T00:00:00.000000Z")
        assert c.name == "Lucas"
        assert c.modified_at == "2025-01-01T00:00:00.000000Z"
        assert c.modified_at != before

    def test_update_unknown_field_rejected(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        with pytest.raises(ProjectValidationError):
            c.apply_update({"role_in_chief": "boss"})

    def test_update_ignores_managed_fields(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        original_id = c.id
        original_created = c.created_at
        c.apply_update({
            "id": "ffffffffffff",
            "project_id": "ffffffffffff",
            "created_at": "2099-01-01T00:00:00.000000Z",
            "modified_at": "2099-01-01T00:00:00.000000Z",
            "name": "Touched",
        })
        assert c.id == original_id
        assert c.project_id == "aaaaaaaaaaaa"
        assert c.created_at == original_created
        assert c.name == "Touched"

    def test_update_must_be_an_object(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        with pytest.raises(ProjectValidationError):
            c.apply_update("string-not-a-dict")  # type: ignore[arg-type]

    def test_update_invalid_role_does_not_advance_modified_at(self) -> None:
        # Mirrors the contract on Source / Participant / Code: a failed
        # update doesn't bump modified_at. In-memory state can have
        # been touched (the entity is single-shot mutable; the caller
        # is expected to discard it on error and reload from disk),
        # but the timestamp doesn't lie about a non-event.
        c = Coder.new(
            project_id="aaaaaaaaaaaa",
            name="Luke",
            now="2025-01-01T00:00:00.000000Z",
        )
        before = c.modified_at
        with pytest.raises(ProjectValidationError):
            c.apply_update(
                {"role": "overlord"}, now="2099-01-01T00:00:00.000000Z"
            )
        assert c.modified_at == before

    def test_update_status_to_inactive(self) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        c.apply_update({"status": "inactive"})
        assert c.status == "inactive"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_creates_file_in_project_subdir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Coder.new(project_id=proj.id, name="Luke")
        path = save_coder(tmp_path, c)
        assert path.exists()
        assert path.parent == coders_dir(tmp_path, proj.id)
        assert path.name == f"{c.id}.json"

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Coder.new(project_id=proj.id, name="Luke", role="reviewer")
        save_coder(tmp_path, c)
        loaded = json.loads(coder_state_path(tmp_path, proj.id, c.id).read_text())
        assert loaded["name"] == "Luke"
        assert loaded["role"] == "reviewer"

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        c = Coder.new(project_id="aaaaaaaaaaaa", name="Luke")
        with pytest.raises(FileNotFoundError):
            save_coder(tmp_path, c)

    def test_load_round_trips(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Coder.new(
            project_id=proj.id,
            name="Coder B",
            role="second_coder",
            email="b@example.org",
            colour="#aabbcc",
            notes="onboarded 2026-04-12",
        )
        save_coder(tmp_path, c)
        c2 = load_coder(tmp_path, proj.id, c.id)
        assert c2.to_dict() == c.to_dict()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_coder(tmp_path, proj.id, "0123456789ab")

    def test_invalid_coder_id_in_state_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            coder_state_path(tmp_path, "aaaaaaaaaaaa", "UPPERCASE123")

    def test_list_coders_sorted_by_created_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c1 = Coder.new(
            project_id=proj.id,
            name="A",
            now="2025-01-01T00:00:00.000000Z",
        )
        c2 = Coder.new(
            project_id=proj.id,
            name="B",
            now="2025-01-02T00:00:00.000000Z",
        )
        c3 = Coder.new(
            project_id=proj.id,
            name="C",
            now="2025-01-03T00:00:00.000000Z",
        )
        # save out of order
        save_coder(tmp_path, c3)
        save_coder(tmp_path, c1)
        save_coder(tmp_path, c2)
        names = [c.name for c in list_coders(tmp_path, proj.id)]
        assert names == ["A", "B", "C"]

    def test_list_coders_empty_for_no_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_coders(tmp_path, proj.id) == []

    def test_list_coders_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        good = Coder.new(project_id=proj.id, name="Good")
        save_coder(tmp_path, good)
        cd = coders_dir(tmp_path, proj.id)
        (cd / "ffffffffffff.json").write_text("{not json")
        # one with valid JSON but wrong shape
        (cd / "eeeeeeeeeeee.json").write_text(json.dumps({"oops": True}))
        # tmp file should be skipped
        (cd / "dddddddddddd.json.tmp").write_text("{}")
        # bad-id filename (not 12-hex) should be skipped
        (cd / "not-a-real-id.json").write_text(
            json.dumps(good.to_dict())
        )
        out = list_coders(tmp_path, proj.id)
        assert [c.id for c in out] == [good.id]

    def test_delete_coder(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Coder.new(project_id=proj.id, name="Luke")
        save_coder(tmp_path, c)
        assert delete_coder(tmp_path, proj.id, c.id) is True
        assert delete_coder(tmp_path, proj.id, c.id) is False

    def test_delete_coder_invalid_id_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            delete_coder(tmp_path, proj.id, "UPPERCASE123")


# --------------------------------------------------------------------------- #
# Integration with delete_project
# --------------------------------------------------------------------------- #


class TestProjectDeletionCascade:
    def test_deleting_project_removes_coders_dir(self, tmp_path: Path) -> None:
        from scribe.projects import delete_project

        proj = _saved_project(tmp_path)
        c = Coder.new(project_id=proj.id, name="Luke")
        save_coder(tmp_path, c)
        cd = coders_dir(tmp_path, proj.id)
        assert cd.exists()
        delete_project(tmp_path, proj.id)
        assert not cd.exists()
