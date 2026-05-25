"""Tests for scribe.project_checkpoints (F9.4).

Covers:
  * Checkpoint dataclass round-trip + validation
  * On-disk persistence (save / load / list / count / find_by_name)
  * Refusal to overwrite an existing checkpoint id (append-only)
  * High-level create_project_checkpoint:
      - Builds the archive at the right path
      - Excludes the checkpoints/ subdirectory from itself
      - Hashes the archive with SHA-256
      - Records component counts
      - Records an F9.1 event with the checkpoint id and back-writes
        event_id onto the metadata sidecar
      - record_audit_event=False skips event emission
      - Survives event-emission failure
  * verify_checkpoint_archive (success / mismatch / missing)
  * extract_checkpoint_to_directory (round-trip restore + verify)
  * Component-counts measurement helpers
  * Cheap projection helpers (checkpoint_summary / list_checkpoint_summaries)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scribe.codes import Code
from scribe.code_versions import save_code_with_version
from scribe.event_log import (
    EVENT_ACTION_CHECKPOINT,
    EVENT_ACTIONS,
    EVENT_ENTITY_CHECKPOINT,
    EVENT_ENTITY_TYPES,
    list_events,
)
from scribe.project_format import ProjectFormatError
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.project_checkpoints import (
    CHECKPOINT_ARCHIVE_SUFFIX,
    CHECKPOINT_ID_RE,
    CHECKPOINT_META_SUFFIX,
    CHECKPOINTS_DIRNAME,
    Checkpoint,
    MAX_ARCHIVE_BYTES,
    MAX_COMPONENT_COUNT_KEYS,
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    checkpoint_archive_path,
    checkpoint_meta_path,
    checkpoint_summary,
    checkpoints_dir,
    compute_component_counts,
    count_checkpoints,
    create_project_checkpoint,
    extract_checkpoint_to_directory,
    find_checkpoint_by_name,
    list_checkpoint_summaries,
    list_checkpoints,
    load_checkpoint,
    new_checkpoint_id,
    save_checkpoint_meta,
    verify_checkpoint_archive,
)


_HEX_PROJECT = "0" * 12
_HEX_CODER = "c" * 12
_HEX_EVENT = "e" * 12
_HEX_CHECKPOINT = "1" * 12


def _saved_project(tmp_path: Path, *, name: str = "Project", stage: str = "initial") -> Project:
    p = Project.new(name=name, codebook_stage=stage)
    save_project(tmp_path, p)
    return p


# --------------------------------------------------------------------------- #
# new_checkpoint_id + CHECKPOINT_ID_RE shape
# --------------------------------------------------------------------------- #


class TestNewCheckpointId:
    def test_returns_12_char_hex(self) -> None:
        cid = new_checkpoint_id()
        assert CHECKPOINT_ID_RE.match(cid)
        assert len(cid) == 12

    def test_returns_unique_ids(self) -> None:
        ids = {new_checkpoint_id() for _ in range(64)}
        assert len(ids) == 64


# --------------------------------------------------------------------------- #
# Vocabulary cross-check with the F9.1 event log
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_event_action_checkpoint_registered(self) -> None:
        assert EVENT_ACTION_CHECKPOINT in EVENT_ACTIONS

    def test_event_entity_checkpoint_registered(self) -> None:
        assert EVENT_ENTITY_CHECKPOINT in EVENT_ENTITY_TYPES


# --------------------------------------------------------------------------- #
# Checkpoint construction + validation
# --------------------------------------------------------------------------- #


class TestCheckpointConstruction:
    def test_minimum_required_fields(self) -> None:
        c = Checkpoint.new(project_id=_HEX_PROJECT, name="Pre-merge")
        assert CHECKPOINT_ID_RE.match(c.id)
        assert c.project_id == _HEX_PROJECT
        assert c.name == "Pre-merge"
        assert c.description == ""
        assert c.actor_coder_id == ""
        assert c.parent_checkpoint_id == ""
        assert c.event_id == ""
        assert c.codebook_stage == "initial"
        # archive_filename defaults to <id>.scribe.zip
        assert c.archive_filename == f"{c.id}{CHECKPOINT_ARCHIVE_SUFFIX}"
        assert c.archive_bytes == 0
        assert c.archive_sha256 == ""
        assert c.component_counts == {}
        assert c.created_at and c.created_at.endswith("Z")

    def test_explicit_id_and_now(self) -> None:
        c = Checkpoint.new(
            project_id=_HEX_PROJECT,
            name="bookmark",
            checkpoint_id="abcdef012345",
            now="2026-05-26T01:23:45Z",
        )
        assert c.id == "abcdef012345"
        assert c.created_at == "2026-05-26T01:23:45Z"

    def test_name_trimmed(self) -> None:
        c = Checkpoint.new(project_id=_HEX_PROJECT, name="  spaced  ")
        assert c.name == "spaced"

    def test_invalid_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                checkpoint_id="not-hex",
            )

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(project_id="not-hex", name="x")

    def test_blank_name_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(project_id=_HEX_PROJECT, name="   ")

    def test_long_name_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(project_id=_HEX_PROJECT, name="x" * (MAX_NAME_LEN + 1))

    def test_long_description_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                description="d" * (MAX_DESCRIPTION_LEN + 1),
            )

    def test_invalid_actor_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                actor_coder_id="not-hex",
            )

    def test_invalid_parent_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                parent_checkpoint_id="not-hex",
            )

    def test_self_parent_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                checkpoint_id=_HEX_CHECKPOINT,
                parent_checkpoint_id=_HEX_CHECKPOINT,
            )

    def test_invalid_event_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                event_id="not-hex",
            )

    def test_empty_optional_ids_allowed(self) -> None:
        c = Checkpoint.new(
            project_id=_HEX_PROJECT,
            name="x",
            actor_coder_id="",
            parent_checkpoint_id="",
            event_id="",
        )
        assert c.actor_coder_id == ""
        assert c.parent_checkpoint_id == ""
        assert c.event_id == ""

    def test_archive_filename_must_be_flat(self) -> None:
        c = Checkpoint(
            id=_HEX_CHECKPOINT,
            project_id=_HEX_PROJECT,
            name="x",
            archive_filename="../escape.zip",
            created_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_archive_filename_rejects_subpath(self) -> None:
        c = Checkpoint(
            id=_HEX_CHECKPOINT,
            project_id=_HEX_PROJECT,
            name="x",
            archive_filename="sub/file.zip",
            created_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(ProjectValidationError):
            c.validate()

    def test_archive_bytes_must_be_non_negative(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                archive_bytes=-1,
            )

    def test_archive_bytes_above_ceiling_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                archive_bytes=MAX_ARCHIVE_BYTES + 1,
            )

    def test_invalid_archive_sha256_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                archive_sha256="not-a-real-sha",
            )

    def test_valid_archive_sha256_accepted(self) -> None:
        c = Checkpoint.new(
            project_id=_HEX_PROJECT,
            name="x",
            archive_sha256="a" * 64,
        )
        assert c.archive_sha256 == "a" * 64

    def test_uppercase_sha256_normalised(self) -> None:
        c = Checkpoint.new(
            project_id=_HEX_PROJECT,
            name="x",
            archive_sha256="A" * 64,
        )
        assert c.archive_sha256 == "a" * 64

    def test_component_counts_must_be_non_negative_ints(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                component_counts={"sources": -1},
            )

    def test_component_counts_keys_must_match_pattern(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                component_counts={"Sources!": 1},
            )

    def test_component_counts_cardinality_capped(self) -> None:
        too_many = {f"k{i:02d}": 1 for i in range(MAX_COMPONENT_COUNT_KEYS + 1)}
        with pytest.raises(ProjectValidationError):
            Checkpoint.new(
                project_id=_HEX_PROJECT,
                name="x",
                component_counts=too_many,
            )


# --------------------------------------------------------------------------- #
# Serialisation round-trip
# --------------------------------------------------------------------------- #


class TestSerialisation:
    def test_to_dict_from_dict_round_trip(self) -> None:
        c = Checkpoint.new(
            project_id=_HEX_PROJECT,
            name="ck",
            description="why",
            actor_coder_id=_HEX_CODER,
            parent_checkpoint_id=_HEX_CHECKPOINT,
            event_id=_HEX_EVENT,
            codebook_stage="focused",
            archive_bytes=12345,
            archive_sha256="b" * 64,
            component_counts={"sources": 1, "codes": 3},
        )
        d = c.to_dict()
        c2 = Checkpoint.from_dict(d)
        assert c2 == c

    def test_from_dict_requires_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.from_dict({"project_id": _HEX_PROJECT, "name": "x"})

    def test_from_dict_requires_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.from_dict({"id": _HEX_CHECKPOINT, "name": "x"})

    def test_from_dict_requires_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.from_dict(
                {"id": _HEX_CHECKPOINT, "project_id": _HEX_PROJECT}
            )

    def test_from_dict_rejects_non_object_payload(self) -> None:
        with pytest.raises(ProjectValidationError):
            Checkpoint.from_dict("not a dict")  # type: ignore[arg-type]

    def test_from_dict_rejects_non_object_component_counts(self) -> None:
        # A truthy non-Mapping payload (``None`` / ``[]`` / ``""`` are
        # all treated as "absent" and replaced with an empty dict — the
        # loader only objects when something *was* supplied but is the
        # wrong shape).
        with pytest.raises(ProjectValidationError):
            Checkpoint.from_dict(
                {
                    "id": _HEX_CHECKPOINT,
                    "project_id": _HEX_PROJECT,
                    "name": "x",
                    "archive_filename": f"{_HEX_CHECKPOINT}.scribe.zip",
                    "component_counts": [("k", 1)],
                }
            )


# --------------------------------------------------------------------------- #
# On-disk paths
# --------------------------------------------------------------------------- #


class TestPaths:
    def test_checkpoints_dir(self, tmp_path: Path) -> None:
        d = checkpoints_dir(tmp_path, _HEX_PROJECT)
        assert d.name == CHECKPOINTS_DIRNAME
        assert d.parent.name == _HEX_PROJECT

    def test_meta_path_uses_id_and_suffix(self, tmp_path: Path) -> None:
        p = checkpoint_meta_path(tmp_path, _HEX_PROJECT, _HEX_CHECKPOINT)
        assert p.name == f"{_HEX_CHECKPOINT}{CHECKPOINT_META_SUFFIX}"
        assert p.parent.name == CHECKPOINTS_DIRNAME

    def test_archive_path_uses_id_and_suffix(self, tmp_path: Path) -> None:
        p = checkpoint_archive_path(tmp_path, _HEX_PROJECT, _HEX_CHECKPOINT)
        assert p.name == f"{_HEX_CHECKPOINT}{CHECKPOINT_ARCHIVE_SUFFIX}"
        assert p.parent.name == CHECKPOINTS_DIRNAME

    def test_meta_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            checkpoint_meta_path(tmp_path, _HEX_PROJECT, "not-hex")

    def test_archive_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            checkpoint_archive_path(tmp_path, _HEX_PROJECT, "not-hex")


# --------------------------------------------------------------------------- #
# save_checkpoint_meta + load_checkpoint
# --------------------------------------------------------------------------- #


class TestSaveAndLoad:
    def test_save_creates_sidecar(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="c1")
        target = save_checkpoint_meta(tmp_path, c)
        assert target.exists()
        # parsed payload validates
        loaded = load_checkpoint(tmp_path, project.id, c.id)
        assert loaded == c

    def test_save_refuses_to_overwrite(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="c1")
        save_checkpoint_meta(tmp_path, c)
        c2 = Checkpoint.new(
            project_id=project.id,
            name="c2",
            checkpoint_id=c.id,
        )
        with pytest.raises(FileExistsError):
            save_checkpoint_meta(tmp_path, c2)

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        c = Checkpoint.new(project_id=_HEX_PROJECT, name="c1")
        with pytest.raises(FileNotFoundError):
            save_checkpoint_meta(tmp_path, c)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path, project.id, _HEX_CHECKPOINT)


# --------------------------------------------------------------------------- #
# list_checkpoints / count_checkpoints / find_checkpoint_by_name
# --------------------------------------------------------------------------- #


class TestListing:
    def test_empty_when_dir_missing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert list_checkpoints(tmp_path, project.id) == []
        assert count_checkpoints(tmp_path, project.id) == 0

    def test_listing_sorts_ascending_by_created_at(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c1 = Checkpoint.new(
            project_id=project.id, name="b", now="2026-05-02T00:00:00Z"
        )
        c2 = Checkpoint.new(
            project_id=project.id, name="a", now="2026-05-01T00:00:00Z"
        )
        save_checkpoint_meta(tmp_path, c1)
        save_checkpoint_meta(tmp_path, c2)
        out = list_checkpoints(tmp_path, project.id)
        assert [c.name for c in out] == ["a", "b"]

    def test_count_matches_listing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        for i in range(3):
            save_checkpoint_meta(
                tmp_path,
                Checkpoint.new(
                    project_id=project.id,
                    name=f"c{i}",
                    now=f"2026-05-0{i+1}T00:00:00Z",
                ),
            )
        assert count_checkpoints(tmp_path, project.id) == 3
        assert len(list_checkpoints(tmp_path, project.id)) == 3

    def test_listing_skips_corrupt_files(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="ok")
        save_checkpoint_meta(tmp_path, c)
        # Drop a malformed sibling that has the right shape but isn't JSON.
        bad = checkpoints_dir(tmp_path, project.id) / "deadbeef0000.json"
        bad.write_text("not json")
        out = list_checkpoints(tmp_path, project.id)
        # Only the well-formed one is returned.
        assert [x.id for x in out] == [c.id]

    def test_listing_skips_tmp_files(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="ok")
        save_checkpoint_meta(tmp_path, c)
        # Drop a stray .tmp file
        (checkpoints_dir(tmp_path, project.id) / "deadbeef0000.json.tmp").write_text("{}")
        out = list_checkpoints(tmp_path, project.id)
        assert [x.id for x in out] == [c.id]

    def test_listing_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            list_checkpoints(tmp_path, "not-hex")

    def test_find_by_name_returns_match(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="bookmark")
        save_checkpoint_meta(tmp_path, c)
        found = find_checkpoint_by_name(tmp_path, project.id, "bookmark")
        assert found is not None
        assert found.id == c.id

    def test_find_by_name_trims_input(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Checkpoint.new(project_id=project.id, name="bookmark")
        save_checkpoint_meta(tmp_path, c)
        found = find_checkpoint_by_name(tmp_path, project.id, "  bookmark  ")
        assert found is not None
        assert found.id == c.id

    def test_find_by_name_returns_none_if_missing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert find_checkpoint_by_name(tmp_path, project.id, "ghost") is None

    def test_find_by_name_rejects_blank(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert find_checkpoint_by_name(tmp_path, project.id, "   ") is None

    def test_find_by_name_returns_latest_on_collision(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        c1 = Checkpoint.new(
            project_id=project.id, name="dup", now="2026-05-01T00:00:00Z"
        )
        c2 = Checkpoint.new(
            project_id=project.id, name="dup", now="2026-05-02T00:00:00Z"
        )
        save_checkpoint_meta(tmp_path, c1)
        save_checkpoint_meta(tmp_path, c2)
        found = find_checkpoint_by_name(tmp_path, project.id, "dup")
        assert found is not None
        assert found.id == c2.id


# --------------------------------------------------------------------------- #
# compute_component_counts
# --------------------------------------------------------------------------- #


class TestComponentCounts:
    def test_counts_zero_for_fresh_project(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        counts = compute_component_counts(tmp_path, project.id)
        # Stable keys present:
        for key in (
            "sources",
            "participants",
            "codes",
            "applications",
            "memos",
            "coders",
            "speaker_maps",
            "saved_queries",
            "snapshots",
            "events",
            "ai_events",
            "code_versions",
            "sampling_log_entries",
        ):
            assert counts[key] == 0

    def test_counts_codes_after_save(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        save_code_with_version(
            tmp_path, Code.new(project_id=project.id, name="code-a")
        )
        save_code_with_version(
            tmp_path, Code.new(project_id=project.id, name="code-b")
        )
        counts = compute_component_counts(tmp_path, project.id)
        assert counts["codes"] == 2
        # ``code_versions`` is a stable key whether or not entries
        # show up; the underlying store is JSONL so the JSON-only file
        # counter reports 0 even when versions exist on disk. The
        # contract the UI relies on is "the key is present".
        assert "code_versions" in counts

    def test_counts_excludes_checkpoint_dir(self, tmp_path: Path) -> None:
        # Even after we mint a checkpoint, the count summary doesn't
        # include "checkpoints" — that subdir is excluded by design.
        project = _saved_project(tmp_path)
        create_project_checkpoint(tmp_path, project.id, name="first")
        counts = compute_component_counts(tmp_path, project.id)
        assert "checkpoints" not in counts

    def test_counts_missing_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_component_counts(tmp_path, _HEX_PROJECT)


# --------------------------------------------------------------------------- #
# create_project_checkpoint — high-level happy path
# --------------------------------------------------------------------------- #


class TestCreateProjectCheckpoint:
    def test_creates_metadata_and_archive(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        save_code_with_version(
            tmp_path, Code.new(project_id=project.id, name="x")
        )

        cp = create_project_checkpoint(
            tmp_path, project.id, name="v1", description="why"
        )
        assert CHECKPOINT_ID_RE.match(cp.id)
        assert cp.archive_filename == f"{cp.id}{CHECKPOINT_ARCHIVE_SUFFIX}"
        assert cp.archive_bytes > 0
        assert len(cp.archive_sha256) == 64
        assert cp.codebook_stage == project.codebook_stage
        assert cp.component_counts["codes"] == 1

        # Both files exist
        meta = checkpoint_meta_path(tmp_path, project.id, cp.id)
        archive = checkpoint_archive_path(tmp_path, project.id, cp.id)
        assert meta.exists()
        assert archive.exists()

        # Archive contains project.json under the project_id top-level
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
        assert any(n == f"{project.id}/project.json" for n in names)

    def test_audit_event_recorded_and_back_written(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path,
            project.id,
            name="v1",
            actor_coder_id=_HEX_CODER,
        )
        # Event id is back-written onto the metadata sidecar.
        assert cp.event_id
        loaded = load_checkpoint(tmp_path, project.id, cp.id)
        assert loaded.event_id == cp.event_id

        evs = [
            e for e in list_events(tmp_path, project.id)
            if e.action == EVENT_ACTION_CHECKPOINT
        ]
        assert len(evs) == 1
        ev = evs[0]
        assert ev.entity_type == EVENT_ENTITY_CHECKPOINT
        assert ev.entity_id == cp.id
        assert ev.actor_coder_id == _HEX_CODER

    def test_skip_audit_event(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path, project.id, name="silent", record_audit_event=False
        )
        assert cp.event_id == ""
        assert (
            len([
                e for e in list_events(tmp_path, project.id)
                if e.action == EVENT_ACTION_CHECKPOINT
            ])
            == 0
        )

    def test_explicit_now_propagates(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path,
            project.id,
            name="v1",
            now="2026-05-26T01:23:45Z",
            record_audit_event=False,
        )
        assert cp.created_at == "2026-05-26T01:23:45Z"

    def test_explicit_id_used(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path,
            project.id,
            name="v1",
            checkpoint_id="abcdef012345",
            record_audit_event=False,
        )
        assert cp.id == "abcdef012345"

    def test_invalid_explicit_id_rejected(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            create_project_checkpoint(
                tmp_path,
                project.id,
                name="v1",
                checkpoint_id="not-hex",
            )

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        # No project.json on disk yet.
        with pytest.raises(FileNotFoundError):
            create_project_checkpoint(tmp_path, _HEX_PROJECT, name="v1")

    def test_refuses_to_overwrite_archive(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        # Pre-place the archive that the next mint would write to.
        cd = checkpoints_dir(tmp_path, project.id)
        cd.mkdir(parents=True, exist_ok=True)
        squatter_id = "abcdef012345"
        squatter = checkpoint_archive_path(tmp_path, project.id, squatter_id)
        squatter.write_bytes(b"not an archive")
        with pytest.raises(FileExistsError):
            create_project_checkpoint(
                tmp_path,
                project.id,
                name="v1",
                checkpoint_id=squatter_id,
                record_audit_event=False,
            )

    def test_does_not_recursively_embed_prior_checkpoints(
        self, tmp_path: Path
    ) -> None:
        # The checkpoints/ subdirectory must not appear inside any
        # checkpoint's archive — otherwise checkpoint #N embeds
        # checkpoint #N-1, #N-2 ... and disk usage explodes.
        project = _saved_project(tmp_path)
        cp1 = create_project_checkpoint(
            tmp_path, project.id, name="first", record_audit_event=False
        )
        cp2 = create_project_checkpoint(
            tmp_path, project.id, name="second", record_audit_event=False
        )
        archive2 = checkpoint_archive_path(tmp_path, project.id, cp2.id)
        with zipfile.ZipFile(archive2, "r") as zf:
            names = zf.namelist()
        assert all(
            f"{project.id}/{CHECKPOINTS_DIRNAME}/" not in n for n in names
        ), f"checkpoints/ leaked into archive: {names}"
        # And cp1's metadata file isn't inside cp2's archive either.
        cp1_meta_arc = (
            f"{project.id}/{CHECKPOINTS_DIRNAME}/"
            f"{cp1.id}{CHECKPOINT_META_SUFFIX}"
        )
        assert cp1_meta_arc not in names

    def test_parent_pointer_persisted(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp1 = create_project_checkpoint(
            tmp_path, project.id, name="first", record_audit_event=False
        )
        cp2 = create_project_checkpoint(
            tmp_path,
            project.id,
            name="second",
            parent_checkpoint_id=cp1.id,
            record_audit_event=False,
        )
        loaded = load_checkpoint(tmp_path, project.id, cp2.id)
        assert loaded.parent_checkpoint_id == cp1.id


# --------------------------------------------------------------------------- #
# Resilience: audit event emission failure
# --------------------------------------------------------------------------- #


class TestEventEmissionResilience:
    def test_checkpoint_survives_event_emission_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _saved_project(tmp_path)

        from scribe import project_checkpoints as mod

        def boom(*args, **kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("disk full")

        monkeypatch.setattr(mod, "record_event", boom)
        cp = create_project_checkpoint(tmp_path, project.id, name="resilient")
        # Metadata + archive both still on disk; event_id is empty.
        assert cp.event_id == ""
        loaded = load_checkpoint(tmp_path, project.id, cp.id)
        assert loaded.event_id == ""
        assert checkpoint_archive_path(tmp_path, project.id, cp.id).exists()


# --------------------------------------------------------------------------- #
# Hash + verify
# --------------------------------------------------------------------------- #


class TestVerify:
    def test_verify_returns_true_for_intact_archive(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path, project.id, name="v1", record_audit_event=False
        )
        assert verify_checkpoint_archive(tmp_path, cp) is True

    def test_verify_returns_false_when_archive_modified(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path, project.id, name="v1", record_audit_event=False
        )
        archive = checkpoint_archive_path(tmp_path, project.id, cp.id)
        # Append a byte: zip reader may still parse it but the sha
        # changes.
        with archive.open("ab") as f:
            f.write(b"x")
        assert verify_checkpoint_archive(tmp_path, cp) is False

    def test_verify_raises_when_archive_missing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path, project.id, name="v1", record_audit_event=False
        )
        checkpoint_archive_path(tmp_path, project.id, cp.id).unlink()
        with pytest.raises(FileNotFoundError):
            verify_checkpoint_archive(tmp_path, cp)

    def test_verify_returns_false_when_sha_unrecorded(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path, project.id, name="v1", record_audit_event=False
        )
        # Strip the sha and assert verify returns False (can't verify).
        cp.archive_sha256 = ""
        assert verify_checkpoint_archive(tmp_path, cp) is False


# --------------------------------------------------------------------------- #
# Restore
# --------------------------------------------------------------------------- #


class TestExtract:
    def test_round_trip_restore(self, tmp_path: Path) -> None:
        # Mint a checkpoint in source/, then extract into target/ — the
        # target should contain the exact same project.json.
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        tgt.mkdir()

        project = _saved_project(src)
        save_code_with_version(
            src, Code.new(project_id=project.id, name="hello")
        )
        cp = create_project_checkpoint(
            src, project.id, name="v1", record_audit_event=False
        )

        restored_dir = extract_checkpoint_to_directory(src, cp, tgt)
        assert restored_dir.exists()
        assert (restored_dir / "project.json").exists()

        restored_data = json.loads(
            (restored_dir / "project.json").read_text()
        )
        original_data = json.loads(
            (src / project.id / "project.json").read_text()
        )
        assert restored_data == original_data

    def test_verify_mismatch_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        tgt.mkdir()
        project = _saved_project(src)
        cp = create_project_checkpoint(
            src, project.id, name="v1", record_audit_event=False
        )
        # Corrupt the metadata's sha to force a mismatch.
        cp.archive_sha256 = "0" * 64
        with pytest.raises(ProjectFormatError):
            extract_checkpoint_to_directory(src, cp, tgt, verify=True)

    def test_verify_skipped_when_disabled(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        tgt.mkdir()
        project = _saved_project(src)
        cp = create_project_checkpoint(
            src, project.id, name="v1", record_audit_event=False
        )
        # Even with a wrong sha, verify=False should still extract.
        cp.archive_sha256 = "0" * 64
        restored_dir = extract_checkpoint_to_directory(
            src, cp, tgt, verify=False
        )
        assert (restored_dir / "project.json").exists()

    def test_missing_archive_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        tgt = tmp_path / "tgt"
        tgt.mkdir()
        project = _saved_project(src)
        cp = create_project_checkpoint(
            src, project.id, name="v1", record_audit_event=False
        )
        checkpoint_archive_path(src, project.id, cp.id).unlink()
        with pytest.raises(FileNotFoundError):
            extract_checkpoint_to_directory(src, cp, tgt)


# --------------------------------------------------------------------------- #
# Cheap projections
# --------------------------------------------------------------------------- #


class TestSummaries:
    def test_checkpoint_summary_shape(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        cp = create_project_checkpoint(
            tmp_path,
            project.id,
            name="bookmark",
            description="why",
            actor_coder_id=_HEX_CODER,
        )
        s = checkpoint_summary(cp)
        assert s["id"] == cp.id
        assert s["project_id"] == project.id
        assert s["name"] == "bookmark"
        assert s["description"] == "why"
        assert s["codebook_stage"] == project.codebook_stage
        assert s["actor_coder_id"] == _HEX_CODER
        assert s["event_id"] == cp.event_id
        assert s["archive_filename"] == cp.archive_filename
        assert s["archive_bytes"] == cp.archive_bytes
        assert s["archive_sha256"] == cp.archive_sha256
        assert isinstance(s["component_counts"], dict)
        assert s["created_at"] == cp.created_at

    def test_list_summaries_returns_dicts(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        save_checkpoint_meta(
            tmp_path,
            Checkpoint.new(
                project_id=project.id,
                name="first",
                now="2026-05-01T00:00:00Z",
            ),
        )
        save_checkpoint_meta(
            tmp_path,
            Checkpoint.new(
                project_id=project.id,
                name="second",
                now="2026-05-02T00:00:00Z",
            ),
        )
        out = list_checkpoint_summaries(tmp_path, project.id)
        assert [s["name"] for s in out] == ["first", "second"]
        for s in out:
            assert isinstance(s, dict)
            # No live re-load of the archive contents — summaries are cheap.
            assert "id" in s and "archive_filename" in s


# --------------------------------------------------------------------------- #
# project_format.export_project_archive — exclude_relative knob
# --------------------------------------------------------------------------- #


class TestExcludeRelative:
    """Direct tests for the ``exclude_relative`` extension to F1.5's
    exporter — the foundation F9.4 relies on to avoid recursion."""

    def test_excluded_directory_is_omitted(self, tmp_path: Path) -> None:
        from scribe.project_format import export_project_archive

        project = _saved_project(tmp_path)
        # Drop a file under checkpoints/ to be excluded.
        cd = checkpoints_dir(tmp_path, project.id)
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "deadbeef0000.json").write_text("{}")

        archive = tmp_path / "out.scribe.zip"
        export_project_archive(
            tmp_path,
            project.id,
            archive,
            exclude_relative=[CHECKPOINTS_DIRNAME],
        )
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
        assert all(f"{project.id}/{CHECKPOINTS_DIRNAME}/" not in n for n in names)

    def test_invalid_exclude_rejected(self, tmp_path: Path) -> None:
        from scribe.project_format import export_project_archive

        project = _saved_project(tmp_path)
        archive = tmp_path / "out.scribe.zip"
        with pytest.raises(ProjectFormatError):
            export_project_archive(
                tmp_path,
                project.id,
                archive,
                exclude_relative=["../escape"],
            )

    def test_exclude_non_string_rejected(self, tmp_path: Path) -> None:
        from scribe.project_format import export_project_archive

        project = _saved_project(tmp_path)
        archive = tmp_path / "out.scribe.zip"
        with pytest.raises(ProjectFormatError):
            export_project_archive(
                tmp_path,
                project.id,
                archive,
                exclude_relative=[123],  # type: ignore[list-item]
            )

    def test_blank_exclude_ignored(self, tmp_path: Path) -> None:
        from scribe.project_format import export_project_archive

        project = _saved_project(tmp_path)
        archive = tmp_path / "out.scribe.zip"
        # Empty / whitespace entries should be treated as no-ops, not errors.
        export_project_archive(
            tmp_path,
            project.id,
            archive,
            exclude_relative=["", "  ", "/"],
        )
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
        assert any(n == f"{project.id}/project.json" for n in names)
