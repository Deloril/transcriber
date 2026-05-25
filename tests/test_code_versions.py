"""Tests for scribe.code_versions (F2.2).

Exercise the code-revision-history primitives in pure Python: the
``CodeVersion`` dataclass, definition-change detection, the append-only
JSONL log on disk, and the ``save_code_with_version`` orchestration
helper that ties this all together for the F4.1 application workflow.

Endpoint-level tests will live in test_server.py once F2.2 grows an
HTTP surface; today the model + persistence are the public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.codes import (
    Code,
    save_code,
)
from scribe.code_versions import (
    CODE_VERSION_ID_RE,
    DEFINITION_FIELDS,
    MAX_CHANGE_NOTE_LEN,
    CodeVersion,
    code_versions_dir,
    code_versions_path,
    count_code_versions,
    definition_changed,
    definition_signature,
    find_code_version,
    latest_code_version,
    new_code_version_id,
    read_code_versions,
    record_code_version,
    save_code_with_version,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    delete_project,
    project_dir,
    save_project,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _fresh_code(
    project_id: str, *, name: str = "Pacing", **overrides: object
) -> Code:
    """Build a baseline Code for tests; overrides land on Code.new()."""
    payload: dict[str, object] = {
        "project_id": project_id,
        "name": name,
        "definition": "How participants describe being depleted.",
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Code.new(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewCodeVersionId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert CODE_VERSION_ID_RE.match(new_code_version_id())

    def test_unique(self) -> None:
        ids = {new_code_version_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# DEFINITION_FIELDS coverage
# --------------------------------------------------------------------------- #


class TestDefinitionFields:
    def test_includes_core_definition_fields(self) -> None:
        # These are methodologically definitional in Charmaz; renaming
        # one silently would change what counts as "a new revision".
        for f in (
            "name",
            "definition",
            "inclusion_criteria",
            "exclusion_criteria",
            "exemplars",
            "theoretical_memo",
            "parent_code_id",
            "related_codes",
        ):
            assert f in DEFINITION_FIELDS

    def test_excludes_metadata_fields(self) -> None:
        # Metadata: must NOT trigger a new version on change.
        for f in (
            "stage",
            "colour",
            "status",
            "provenance",
            "id",
            "project_id",
            "created_at",
            "modified_at",
        ):
            assert f not in DEFINITION_FIELDS


# --------------------------------------------------------------------------- #
# definition_signature / definition_changed
# --------------------------------------------------------------------------- #


class TestDefinitionSignature:
    def test_from_code(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        sig = definition_signature(c)
        assert set(sig.keys()) == set(DEFINITION_FIELDS)
        assert sig["name"] == "Pacing"
        assert sig["definition"] == "How participants describe being depleted."

    def test_from_dict(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        sig_a = definition_signature(c)
        sig_b = definition_signature(c.to_dict())
        assert sig_a == sig_b

    def test_missing_list_keys_default_to_empty(self) -> None:
        # Older snapshots may omit a list field; signature should still
        # compare equal to a current code with an empty list.
        sig = definition_signature(
            {"id": "bbbbbbbbbbbb", "project_id": "aaaaaaaaaaaa", "name": "x"}
        )
        assert sig["exemplars"] == []
        assert sig["related_codes"] == []

    def test_rejects_non_code_non_dict(self) -> None:
        with pytest.raises(TypeError):
            definition_signature("nope")  # type: ignore[arg-type]


class TestDefinitionChanged:
    def test_no_previous_means_changed(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        assert definition_changed(None, c) is True

    def test_identical_definition_unchanged(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        assert definition_changed(v, c) is False

    def test_metadata_only_change_unchanged(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        # Mutate non-definitional fields directly.
        c.stage = "focused"
        c.colour = "#abcdef"
        c.status = "draft"
        c.provenance = {"source": "human"}
        c.modified_at = "2099-01-01T00:00:00.000000Z"
        assert definition_changed(v, c) is False

    def test_definition_change_detected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        c.definition = "Different text"
        assert definition_changed(v, c) is True

    def test_name_change_detected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        c.name = "Renamed"
        assert definition_changed(v, c) is True

    def test_exemplars_change_detected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        c.exemplars = ["new exemplar"]
        assert definition_changed(v, c) is True

    def test_parent_change_detected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        c.parent_code_id = "0123456789ab"
        assert definition_changed(v, c) is True

    def test_accepts_raw_dict_previous(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        prev_dict = v.snapshot
        assert definition_changed(prev_dict, c) is False
        c.definition = "x"
        assert definition_changed(prev_dict, c) is True

    def test_rejects_bad_previous_type(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        with pytest.raises(TypeError):
            definition_changed(42, c)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CodeVersion construction + validation
# --------------------------------------------------------------------------- #


class TestCodeVersionNew:
    def test_basic(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        assert v.code_id == c.id
        assert v.project_id == c.project_id
        assert v.version == 1
        assert v.created_at  # non-empty
        assert CODE_VERSION_ID_RE.match(v.id)
        assert v.snapshot["name"] == "Pacing"
        assert v.snapshot["definition"] == c.definition
        assert v.change_note == ""

    def test_explicit_now_used(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(
            code=c, version=1, now="2024-06-01T00:00:00.000000Z"
        )
        assert v.created_at == "2024-06-01T00:00:00.000000Z"

    def test_explicit_version_id_used(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1, version_id="cafebabecafe")
        assert v.id == "cafebabecafe"

    def test_change_note_kept(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(
            code=c, version=1, change_note="Sharpened exclusion."
        )
        assert v.change_note == "Sharpened exclusion."

    def test_change_note_too_long_rejected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        with pytest.raises(ProjectValidationError):
            CodeVersion.new(
                code=c,
                version=1,
                change_note="x" * (MAX_CHANGE_NOTE_LEN + 1),
            )

    def test_zero_or_negative_version_rejected(self) -> None:
        c = _fresh_code("aaaaaaaaaaaa")
        with pytest.raises(ProjectValidationError):
            CodeVersion.new(code=c, version=0)
        with pytest.raises(ProjectValidationError):
            CodeVersion.new(code=c, version=-1)

    def test_snapshot_decoupled_from_code(self) -> None:
        # Mutating the code after capture must not leak into the
        # snapshot — that's the whole point of an immutable version.
        c = _fresh_code("aaaaaaaaaaaa")
        v = CodeVersion.new(code=c, version=1)
        c.name = "Mutated"
        c.exemplars = ["after the fact"]
        assert v.snapshot["name"] == "Pacing"
        assert v.snapshot["exemplars"] == []


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        c = _fresh_code(
            "aaaaaaaaaaaa",
            inclusion_criteria="Statements about energy.",
            exemplars=["ex1"],
        )
        v = CodeVersion.new(
            code=c,
            version=2,
            change_note="Sharpened.",
            now="2024-06-01T00:00:00.000000Z",
        )
        d = v.to_dict()
        assert json.dumps(d)  # JSON-serialisable
        v2 = CodeVersion.from_dict(d)
        assert v2.to_dict() == d

    def test_from_dict_requires_required_keys(self) -> None:
        for missing in ("id", "code_id", "project_id", "version", "created_at"):
            full = {
                "id": "ffffffffffff",
                "code_id": "0123456789ab",
                "project_id": "aaaaaaaaaaaa",
                "version": 1,
                "created_at": "2024-01-01T00:00:00.000000Z",
                "snapshot": {},
            }
            full.pop(missing)
            with pytest.raises(ProjectValidationError):
                CodeVersion.from_dict(full)

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_coerces_numeric_version(self) -> None:
        v = CodeVersion.from_dict({
            "id": "ffffffffffff",
            "code_id": "0123456789ab",
            "project_id": "aaaaaaaaaaaa",
            "version": "3",  # string survives coercion
            "created_at": "2024-01-01T00:00:00.000000Z",
            "snapshot": {},
        })
        assert v.version == 3

    def test_from_dict_rejects_garbage_version(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion.from_dict({
                "id": "ffffffffffff",
                "code_id": "0123456789ab",
                "project_id": "aaaaaaaaaaaa",
                "version": "not a number",
                "created_at": "2024-01-01T00:00:00.000000Z",
            })

    def test_from_dict_rejects_non_dict_snapshot(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion.from_dict({
                "id": "ffffffffffff",
                "code_id": "0123456789ab",
                "project_id": "aaaaaaaaaaaa",
                "version": 1,
                "created_at": "2024-01-01T00:00:00.000000Z",
                "snapshot": "not an object",
            })

    def test_validate_id_shapes(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="UPPERCASE123",
                code_id="0123456789ab",
                project_id="aaaaaaaaaaaa",
                version=1,
                created_at="2024-01-01T00:00:00.000000Z",
                snapshot={},
            ).validate()
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="ffffffffffff",
                code_id="UPPERCASE123",
                project_id="aaaaaaaaaaaa",
                version=1,
                created_at="2024-01-01T00:00:00.000000Z",
                snapshot={},
            ).validate()
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="ffffffffffff",
                code_id="0123456789ab",
                project_id="UPPERCASE123",
                version=1,
                created_at="2024-01-01T00:00:00.000000Z",
                snapshot={},
            ).validate()

    def test_validate_rejects_mismatched_snapshot_ids(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="ffffffffffff",
                code_id="0123456789ab",
                project_id="aaaaaaaaaaaa",
                version=1,
                created_at="2024-01-01T00:00:00.000000Z",
                snapshot={"id": "deadbeefdead", "project_id": "aaaaaaaaaaaa"},
            ).validate()
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="ffffffffffff",
                code_id="0123456789ab",
                project_id="aaaaaaaaaaaa",
                version=1,
                created_at="2024-01-01T00:00:00.000000Z",
                snapshot={"id": "0123456789ab", "project_id": "deadbeefdead"},
            ).validate()

    def test_validate_requires_created_at(self) -> None:
        with pytest.raises(ProjectValidationError):
            CodeVersion(
                id="ffffffffffff",
                code_id="0123456789ab",
                project_id="aaaaaaaaaaaa",
                version=1,
                created_at="",
                snapshot={},
            ).validate()


# --------------------------------------------------------------------------- #
# Persistence — record_code_version
# --------------------------------------------------------------------------- #


class TestRecordCodeVersion:
    def test_first_record_creates_v1(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        v = record_code_version(tmp_path, c)
        assert v.version == 1
        assert v.code_id == c.id
        assert v.project_id == c.project_id
        # File should exist with one JSON line.
        path = code_versions_path(tmp_path, proj.id, c.id)
        assert path.exists()
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["version"] == 1
        assert parsed["code_id"] == c.id

    def test_subsequent_records_increment(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        v1 = record_code_version(tmp_path, c)
        v2 = record_code_version(tmp_path, c)
        v3 = record_code_version(tmp_path, c)
        assert (v1.version, v2.version, v3.version) == (1, 2, 3)

    def test_change_note_persisted(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        v = record_code_version(
            tmp_path, c, change_note="Initial line-by-line code"
        )
        loaded = read_code_versions(tmp_path, proj.id, c.id)
        assert loaded[0].change_note == "Initial line-by-line code"
        assert v.change_note == "Initial line-by-line code"

    def test_record_validates_change_note_length(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        with pytest.raises(ProjectValidationError):
            record_code_version(
                tmp_path, c, change_note="x" * (MAX_CHANGE_NOTE_LEN + 1)
            )
        # Nothing got written.
        assert not code_versions_path(tmp_path, proj.id, c.id).exists()

    def test_requires_existing_project(self, tmp_path: Path) -> None:
        # Don't save_project — directory does not exist.
        c = _fresh_code("aaaaaaaaaaaa")
        with pytest.raises(FileNotFoundError):
            record_code_version(tmp_path, c)

    def test_creates_versions_subdir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        record_code_version(tmp_path, c)
        assert code_versions_dir(tmp_path, proj.id).is_dir()


# --------------------------------------------------------------------------- #
# Persistence — read_code_versions / latest / find / count
# --------------------------------------------------------------------------- #


class TestReadCodeVersions:
    def test_empty_when_no_log(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        assert read_code_versions(tmp_path, proj.id, c.id) == []

    def test_empty_when_no_project(self, tmp_path: Path) -> None:
        # No save_project; reading is a no-op rather than an error.
        assert read_code_versions(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb") == []

    def test_invalid_project_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            read_code_versions(tmp_path, "../escape", "bbbbbbbbbbbb")

    def test_invalid_code_id_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            read_code_versions(tmp_path, proj.id, "../escape")

    def test_skips_corrupt_lines(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        record_code_version(tmp_path, c)
        path = code_versions_path(tmp_path, proj.id, c.id)
        # Append a not-JSON line and a JSON-but-invalid line.
        with path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
            f.write('{"id": "bad shape"}\n')
        # Append a valid second version manually.
        record_code_version(tmp_path, c)
        versions = read_code_versions(tmp_path, proj.id, c.id)
        # Two valid (the corrupt lines were skipped).
        assert [v.version for v in versions] == [1, 2]

    def test_preserves_chronological_order(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        record_code_version(tmp_path, c, now="2024-03-01T00:00:00.000000Z")
        record_code_version(tmp_path, c, now="2024-01-01T00:00:00.000000Z")
        record_code_version(tmp_path, c, now="2024-02-01T00:00:00.000000Z")
        # Order is *append order*, not sorted by created_at.
        versions = read_code_versions(tmp_path, proj.id, c.id)
        assert [v.version for v in versions] == [1, 2, 3]
        assert [v.created_at for v in versions] == [
            "2024-03-01T00:00:00.000000Z",
            "2024-01-01T00:00:00.000000Z",
            "2024-02-01T00:00:00.000000Z",
        ]


class TestLatestCodeVersion:
    def test_none_when_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        assert latest_code_version(tmp_path, proj.id, c.id) is None

    def test_returns_last_appended(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        record_code_version(tmp_path, c)
        c.definition = "v2"
        record_code_version(tmp_path, c)
        c.definition = "v3"
        v3 = record_code_version(tmp_path, c)
        latest = latest_code_version(tmp_path, proj.id, c.id)
        assert latest is not None
        assert latest.version == 3
        assert latest.id == v3.id
        assert latest.snapshot["definition"] == "v3"


class TestFindCodeVersion:
    def test_returns_match(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        v1 = record_code_version(tmp_path, c)
        c.definition = "v2"
        v2 = record_code_version(tmp_path, c)
        found = find_code_version(tmp_path, proj.id, c.id, v1.id)
        assert found is not None
        assert found.version == 1
        # Confirms the returned version pins to the *historical* definition.
        assert found.snapshot["definition"] == "How participants describe being depleted."
        # Sanity: the v2 lookup still works.
        found2 = find_code_version(tmp_path, proj.id, c.id, v2.id)
        assert found2 is not None
        assert found2.snapshot["definition"] == "v2"

    def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        record_code_version(tmp_path, c)
        assert find_code_version(tmp_path, proj.id, c.id, "deadbeefdead") is None

    def test_invalid_version_id_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        with pytest.raises(ProjectValidationError):
            find_code_version(tmp_path, proj.id, c.id, "../escape")


class TestCountCodeVersions:
    def test_zero_when_no_log(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        assert count_code_versions(tmp_path, proj.id, c.id) == 0

    def test_counts_valid_lines(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code(tmp_path, c)
        for _ in range(5):
            record_code_version(tmp_path, c)
        assert count_code_versions(tmp_path, proj.id, c.id) == 5


# --------------------------------------------------------------------------- #
# save_code_with_version — orchestration
# --------------------------------------------------------------------------- #


class TestSaveCodeWithVersion:
    def test_first_save_records_v1(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        path, recorded = save_code_with_version(tmp_path, c)
        assert path.exists()
        assert recorded is not None
        assert recorded.version == 1
        # File on disk has exactly one line.
        log = code_versions_path(tmp_path, proj.id, c.id)
        assert log.exists()
        assert len([ln for ln in log.read_text().splitlines() if ln.strip()]) == 1

    def test_no_change_does_not_record_new_version(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        _, v1 = save_code_with_version(tmp_path, c)
        # Save again with no changes.
        _, v_after = save_code_with_version(tmp_path, c)
        assert v_after is not None
        assert v_after.id == v1.id  # same version returned
        assert count_code_versions(tmp_path, proj.id, c.id) == 1

    def test_metadata_only_change_does_not_record(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code_with_version(tmp_path, c)
        # Toggle non-definitional fields.
        c.apply_update({"stage": "focused", "colour": "#abcdef", "status": "draft"})
        _, recorded = save_code_with_version(tmp_path, c)
        assert recorded is not None
        assert recorded.version == 1  # unchanged
        assert count_code_versions(tmp_path, proj.id, c.id) == 1

    def test_definition_change_records_v2(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code_with_version(tmp_path, c)
        c.apply_update({"definition": "Tighter."})
        _, recorded = save_code_with_version(tmp_path, c)
        assert recorded is not None
        assert recorded.version == 2
        # Both versions readable.
        versions = read_code_versions(tmp_path, proj.id, c.id)
        assert [v.version for v in versions] == [1, 2]
        assert versions[0].snapshot["definition"] == "How participants describe being depleted."
        assert versions[1].snapshot["definition"] == "Tighter."

    def test_change_note_recorded_only_with_new_version(
        self, tmp_path: Path
    ) -> None:
        # If no new version is recorded, the change_note has nothing to
        # attach to and is silently ignored. Documenting the contract.
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code_with_version(tmp_path, c, change_note="Initial.")
        _, recorded = save_code_with_version(
            tmp_path, c, change_note="Should not be recorded."
        )
        assert recorded is not None
        assert recorded.version == 1
        assert recorded.change_note == "Initial."

        c.apply_update({"definition": "Now changed."})
        _, recorded2 = save_code_with_version(
            tmp_path, c, change_note="Sharpened."
        )
        assert recorded2 is not None
        assert recorded2.version == 2
        assert recorded2.change_note == "Sharpened."

    def test_renumbering_runs_through_many_edits(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        for i in range(5):
            c.apply_update({"definition": f"Version {i}"})
            save_code_with_version(tmp_path, c)
        versions = read_code_versions(tmp_path, proj.id, c.id)
        assert [v.version for v in versions] == [1, 2, 3, 4, 5]
        # Each snapshot holds the right text.
        for i, v in enumerate(versions):
            assert v.snapshot["definition"] == f"Version {i}"

    def test_save_persists_full_field_set_in_snapshot(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        c = _fresh_code(
            proj.id,
            inclusion_criteria="i",
            exclusion_criteria="e",
            exemplars=["q"],
            theoretical_memo="m",
        )
        save_code_with_version(tmp_path, c)
        versions = read_code_versions(tmp_path, proj.id, c.id)
        assert len(versions) == 1
        snap = versions[0].snapshot
        # Every persisted field on the Code should be present in the snapshot.
        for fld in (
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
            assert fld in snap, f"snapshot missing {fld}"

    def test_returns_existing_version_when_no_change(
        self, tmp_path: Path
    ) -> None:
        # Documenting the contract used by F4.1's application creator:
        # ``save_code_with_version`` always returns *some* version the
        # caller can pin a new application to.
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        _, v1 = save_code_with_version(tmp_path, c)
        # Re-save without touching anything.
        _, v_again = save_code_with_version(tmp_path, c)
        assert v_again is not None
        assert v_again.id == v1.id


# --------------------------------------------------------------------------- #
# Cascade behaviour
# --------------------------------------------------------------------------- #


class TestCascade:
    def test_project_deletion_removes_versions(self, tmp_path: Path) -> None:
        # Like sources / participants / sampling-log, the version log
        # lives inside the project dir, so delete_project sweeps it.
        proj = _saved_project(tmp_path)
        c = _fresh_code(proj.id)
        save_code_with_version(tmp_path, c)
        c.apply_update({"definition": "v2"})
        save_code_with_version(tmp_path, c)
        assert code_versions_path(tmp_path, proj.id, c.id).exists()
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()
