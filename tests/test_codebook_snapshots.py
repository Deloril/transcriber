"""Tests for scribe.codebook_snapshots (F9.3).

Covers:
  * Snapshot dataclass round-trip + validation
  * Code-id / version-id cross checks
  * On-disk persistence (save / load / list / count)
  * Refusal to overwrite an existing snapshot id
  * find_snapshot_by_name semantics
  * High-level create_codebook_snapshot:
      - Reads live codes + versions
      - Records an F9.1 event with snapshot id
      - Back-writes event_id onto the snapshot
      - record_audit_event=False skips event emission
  * Reconstruction helpers (reconstruct_codes_from_snapshot,
    render_codebook_at_snapshot, code_at_snapshot,
    code_version_id_at_snapshot, snapshot_summary)
  * Size + cardinality caps
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.code_versions import (
    record_code_version,
    save_code_with_version,
)
from scribe.codes import Code, save_code
from scribe.event_log import (
    EVENT_ACTION_SNAPSHOT,
    EVENT_ENTITY_SNAPSHOT,
    list_events,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.codebook_snapshots import (
    MAX_CODES_PER_SNAPSHOT,
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    MAX_SNAPSHOT_BYTES,
    SNAPSHOT_ID_RE,
    SNAPSHOTS_DIRNAME,
    Snapshot,
    code_at_snapshot,
    code_version_id_at_snapshot,
    count_snapshots,
    create_codebook_snapshot,
    find_snapshot_by_name,
    list_snapshot_summaries,
    list_snapshots,
    load_snapshot,
    new_snapshot_id,
    reconstruct_codes_from_snapshot,
    render_codebook_at_snapshot,
    save_snapshot,
    snapshot_state_path,
    snapshot_summary,
    snapshots_dir,
)


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12
_HEX_EVENT = "e" * 12


def _saved_project(tmp_path: Path, *, name: str = "Project", stage: str = "initial") -> Project:
    p = Project.new(name=name, codebook_stage=stage)
    save_project(tmp_path, p)
    return p


def _make_code(project: Project, *, name: str = "code", definition: str = "") -> Code:
    return Code.new(
        project_id=project.id,
        name=name,
        definition=definition,
    )


# --------------------------------------------------------------------------- #
# new_snapshot_id + SNAPSHOT_ID_RE shape
# --------------------------------------------------------------------------- #


class TestNewSnapshotId:
    def test_returns_12_char_hex(self) -> None:
        sid = new_snapshot_id()
        assert SNAPSHOT_ID_RE.match(sid)
        assert len(sid) == 12

    def test_returns_unique_ids(self) -> None:
        ids = {new_snapshot_id() for _ in range(64)}
        assert len(ids) == 64


# --------------------------------------------------------------------------- #
# Snapshot construction + validation
# --------------------------------------------------------------------------- #


class TestSnapshotConstruction:
    def test_minimum_required_fields(self) -> None:
        s = Snapshot.new(project_id=_HEX_PROJECT, name="Initial coding done")
        assert SNAPSHOT_ID_RE.match(s.id)
        assert s.project_id == _HEX_PROJECT
        assert s.name == "Initial coding done"
        assert s.description == ""
        assert s.codebook_stage == "initial"
        assert s.actor_coder_id == ""
        assert s.event_id == ""
        assert s.codes == []
        assert s.code_versions == {}
        assert s.created_at and s.created_at.endswith("Z")

    def test_explicit_snapshot_id_used(self) -> None:
        s = Snapshot.new(
            project_id=_HEX_PROJECT,
            name="Bookmark",
            snapshot_id="abcdef012345",
        )
        assert s.id == "abcdef012345"

    def test_explicit_now_used(self) -> None:
        s = Snapshot.new(
            project_id=_HEX_PROJECT,
            name="x",
            now="2026-05-26T12:00:00Z",
        )
        assert s.created_at == "2026-05-26T12:00:00Z"

    def test_name_trimmed(self) -> None:
        s = Snapshot.new(project_id=_HEX_PROJECT, name="   spaced name  ")
        assert s.name == "spaced name"

    def test_invalid_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id=_HEX_PROJECT, name="x", snapshot_id="not-hex")

    def test_invalid_project_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id="not-hex", name="x")

    def test_blank_name_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id=_HEX_PROJECT, name="   ")

    def test_long_name_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id=_HEX_PROJECT, name="x" * (MAX_NAME_LEN + 1))

    def test_long_description_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="x",
                description="x" * (MAX_DESCRIPTION_LEN + 1),
            )

    def test_unknown_stage_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="x",
                codebook_stage="banana",
            )

    def test_invalid_actor_coder_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="x",
                actor_coder_id="not-hex",
            )

    def test_empty_actor_coder_id_allowed(self) -> None:
        s = Snapshot.new(project_id=_HEX_PROJECT, name="x", actor_coder_id="")
        assert s.actor_coder_id == ""

    def test_invalid_event_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="x",
                event_id="not-hex",
            )

    def test_codes_can_be_code_instances(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        s = Snapshot.new(project_id=_HEX_PROJECT, name="b", codes=[c])
        assert len(s.codes) == 1
        assert s.codes[0]["id"] == c.id
        assert s.codes[0]["name"] == "hello"

    def test_codes_can_be_dicts(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        s = Snapshot.new(project_id=_HEX_PROJECT, name="b", codes=[c.to_dict()])
        assert len(s.codes) == 1
        assert s.codes[0]["id"] == c.id

    def test_codes_unsupported_type_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id=_HEX_PROJECT, name="b", codes=["not a code"])  # type: ignore[list-item]

    def test_duplicate_code_ids_rejected(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="dup")
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id=_HEX_PROJECT, name="b", codes=[c, c])

    def test_code_project_id_must_match(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="x")
        # Snap's project_id differs from code's.
        with pytest.raises(ProjectValidationError):
            Snapshot.new(project_id="1" * 12, name="b", codes=[c])

    def test_code_versions_must_reference_present_codes(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        # Pinning a version for a code id NOT in `codes` is rejected.
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="b",
                codes=[c],
                code_versions={_HEX_CODE: _HEX_VERSION},
            )

    def test_code_versions_keys_must_be_hex(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="b",
                codes=[c],
                code_versions={"bad-id": _HEX_VERSION},
            )

    def test_code_versions_values_must_be_hex(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="b",
                codes=[c],
                code_versions={c.id: "not-hex"},
            )

    def test_too_many_codes_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.new(
                project_id=_HEX_PROJECT,
                name="b",
                codes=[
                    Code.new(project_id=_HEX_PROJECT, name=f"c{i}")
                    for i in range(MAX_CODES_PER_SNAPSHOT + 1)
                ],
            )


class TestSnapshotRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="hello")
        s = Snapshot.new(
            project_id=_HEX_PROJECT,
            name="bookmark",
            description="desc",
            codebook_stage="focused",
            actor_coder_id=_HEX_CODER,
            codes=[c],
            code_versions={c.id: _HEX_VERSION},
        )
        d = s.to_dict()
        s2 = Snapshot.from_dict(d)
        assert s2.id == s.id
        assert s2.name == s.name
        assert s2.description == s.description
        assert s2.codebook_stage == "focused"
        assert s2.actor_coder_id == _HEX_CODER
        assert s2.code_versions == {c.id: _HEX_VERSION}
        assert s2.codes[0]["id"] == c.id

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_missing_required_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({"project_id": _HEX_PROJECT, "name": "x"})
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({"id": "0" * 12, "name": "x"})
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({"id": "0" * 12, "project_id": _HEX_PROJECT})

    def test_from_dict_codes_not_list_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({
                "id": "0" * 12,
                "project_id": _HEX_PROJECT,
                "name": "x",
                "codes": "nope",
            })

    def test_from_dict_codes_entry_not_object_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({
                "id": "0" * 12,
                "project_id": _HEX_PROJECT,
                "name": "x",
                "codes": ["not-an-object"],
            })

    def test_from_dict_code_versions_not_object_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Snapshot.from_dict({
                "id": "0" * 12,
                "project_id": _HEX_PROJECT,
                "name": "x",
                "codes": [],
                "code_versions": [],
            })


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestSnapshotPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="kind-of-pacing")
        s = Snapshot.new(
            project_id=project.id,
            name="initial done",
            codes=[c],
        )
        save_snapshot(tmp_path, s)
        loaded = load_snapshot(tmp_path, project.id, s.id)
        assert loaded.id == s.id
        assert loaded.codes[0]["name"] == "kind-of-pacing"

    def test_save_creates_snapshots_dir(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="x")
        save_snapshot(tmp_path, s)
        d = snapshots_dir(tmp_path, project.id)
        assert d.exists()
        assert d.name == SNAPSHOTS_DIRNAME

    def test_save_refuses_to_overwrite_existing_id(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="x")
        save_snapshot(tmp_path, s)
        with pytest.raises(FileExistsError):
            save_snapshot(tmp_path, s)

    def test_save_requires_existing_project_dir(self, tmp_path: Path) -> None:
        s = Snapshot.new(project_id=_HEX_PROJECT, name="x")
        with pytest.raises(FileNotFoundError):
            save_snapshot(tmp_path, s)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path, project.id, "f" * 12)

    def test_snapshot_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            snapshot_state_path(tmp_path, _HEX_PROJECT, "bad-id")

    def test_save_persists_atomic_temp_file(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="x")
        save_snapshot(tmp_path, s)
        # The temp file should not linger after a successful save.
        d = snapshots_dir(tmp_path, project.id)
        assert any(f.name == f"{s.id}.json" for f in d.iterdir())
        assert not any(f.name.endswith(".tmp") for f in d.iterdir())


# --------------------------------------------------------------------------- #
# list_snapshots / count_snapshots / find_snapshot_by_name
# --------------------------------------------------------------------------- #


class TestListAndFind:
    def test_empty_list_when_no_snapshots(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert list_snapshots(tmp_path, project.id) == []
        assert count_snapshots(tmp_path, project.id) == 0

    def test_list_sorted_by_created_at_ascending(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s1 = Snapshot.new(project_id=project.id, name="first", now="2026-05-01T00:00:00Z")
        s2 = Snapshot.new(project_id=project.id, name="second", now="2026-05-02T00:00:00Z")
        s3 = Snapshot.new(project_id=project.id, name="third", now="2026-05-03T00:00:00Z")
        save_snapshot(tmp_path, s2)
        save_snapshot(tmp_path, s3)
        save_snapshot(tmp_path, s1)
        out = list_snapshots(tmp_path, project.id)
        assert [s.name for s in out] == ["first", "second", "third"]

    def test_list_skips_corrupt_files(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="ok")
        save_snapshot(tmp_path, s)
        # Drop a corrupt file with a valid-shaped id.
        d = snapshots_dir(tmp_path, project.id)
        bad = d / f"{'b' * 12}.json"
        bad.write_text("not valid json")
        out = list_snapshots(tmp_path, project.id)
        assert len(out) == 1 and out[0].name == "ok"

    def test_list_skips_non_hex_filenames(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        d = snapshots_dir(tmp_path, project.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "not-a-snapshot-id.json").write_text("{}")
        # also a tmp file should be ignored
        (d / "abcdefabcdef.json.tmp").write_text("{}")
        assert list_snapshots(tmp_path, project.id) == []
        assert count_snapshots(tmp_path, project.id) == 0

    def test_count_is_cheaper_than_parse(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        for i in range(3):
            s = Snapshot.new(
                project_id=project.id,
                name=f"snap-{i}",
                now=f"2026-05-0{i+1}T00:00:00Z",
            )
            save_snapshot(tmp_path, s)
        assert count_snapshots(tmp_path, project.id) == 3

    def test_count_zero_when_dir_missing(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert count_snapshots(tmp_path, project.id) == 0

    def test_list_invalid_project_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            list_snapshots(tmp_path, "not-hex")

    def test_find_by_name_returns_match(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="initial done")
        save_snapshot(tmp_path, s)
        found = find_snapshot_by_name(tmp_path, project.id, "initial done")
        assert found is not None and found.id == s.id

    def test_find_by_name_trims_input(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = Snapshot.new(project_id=project.id, name="initial done")
        save_snapshot(tmp_path, s)
        found = find_snapshot_by_name(tmp_path, project.id, "  initial done  ")
        assert found is not None and found.id == s.id

    def test_find_by_name_returns_none_when_no_match(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert find_snapshot_by_name(tmp_path, project.id, "missing") is None

    def test_find_by_name_returns_none_for_blank(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        assert find_snapshot_by_name(tmp_path, project.id, "  ") is None
        assert find_snapshot_by_name(tmp_path, project.id, "") is None

    def test_find_by_name_picks_latest_on_collision(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        older = Snapshot.new(
            project_id=project.id,
            name="initial done",
            now="2026-04-01T00:00:00Z",
        )
        newer = Snapshot.new(
            project_id=project.id,
            name="initial done",
            now="2026-05-01T00:00:00Z",
        )
        save_snapshot(tmp_path, older)
        save_snapshot(tmp_path, newer)
        found = find_snapshot_by_name(tmp_path, project.id, "initial done")
        assert found is not None and found.id == newer.id


# --------------------------------------------------------------------------- #
# create_codebook_snapshot — high-level helper
# --------------------------------------------------------------------------- #


class TestCreateCodebookSnapshot:
    def test_captures_current_codes_and_versions(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path, stage="focused")
        c1 = Code.new(project_id=project.id, name="pacing")
        save_code_with_version(tmp_path, c1)
        c2 = Code.new(project_id=project.id, name="agency")
        save_code_with_version(tmp_path, c2)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="initial done",
            description="line by line complete",
        )
        assert snap.codebook_stage == "focused"
        names = sorted(c["name"] for c in snap.codes)
        assert names == ["agency", "pacing"]
        # both codes should have a version pin from save_code_with_version
        assert set(snap.code_versions.keys()) == {c1.id, c2.id}

    def test_no_codes_yields_empty_snapshot(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="empty bookmark",
        )
        assert snap.codes == []
        assert snap.code_versions == {}

    def test_version_pin_omitted_for_codes_without_versions(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        c1 = Code.new(project_id=project.id, name="versioned")
        save_code_with_version(tmp_path, c1)
        c2 = Code.new(project_id=project.id, name="unversioned")
        save_code(tmp_path, c2)  # plain save — no version log
        snap = create_codebook_snapshot(tmp_path, project.id, name="x")
        assert c1.id in snap.code_versions
        assert c2.id not in snap.code_versions
        # but both codes are still embedded in the snapshot
        assert len(snap.codes) == 2

    def test_emits_audit_event_by_default(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(tmp_path, project.id, name="bookmark-1")
        events = list_events(tmp_path, project.id)
        snap_events = [
            e for e in events
            if e.action == EVENT_ACTION_SNAPSHOT
            and e.entity_type == EVENT_ENTITY_SNAPSHOT
        ]
        assert len(snap_events) == 1
        ev = snap_events[0]
        assert ev.entity_id == snap.id
        # Summary payload — small projection, not the whole codebook.
        assert ev.after is not None
        assert ev.after["snapshot_id"] == snap.id
        assert ev.after["name"] == "bookmark-1"
        assert ev.after["code_count"] == 0

    def test_event_id_back_written_on_snapshot(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(tmp_path, project.id, name="bookmark-2")
        # In-memory snapshot has the event id
        assert snap.event_id != ""
        # And the persisted file does too
        loaded = load_snapshot(tmp_path, project.id, snap.id)
        assert loaded.event_id == snap.event_id

    def test_event_emission_can_be_skipped(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="silent",
            record_audit_event=False,
        )
        assert snap.event_id == ""
        # Confirm no snapshot event was logged.
        evs = [
            e for e in list_events(tmp_path, project.id)
            if e.action == EVENT_ACTION_SNAPSHOT
        ]
        assert evs == []

    def test_actor_coder_id_propagated(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="bookmark-3",
            actor_coder_id=_HEX_CODER,
        )
        assert snap.actor_coder_id == _HEX_CODER
        # And the event records the same actor.
        evs = [
            e for e in list_events(tmp_path, project.id)
            if e.action == EVENT_ACTION_SNAPSHOT
        ]
        assert len(evs) == 1
        assert evs[0].actor_coder_id == _HEX_CODER

    def test_explicit_now_propagates(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="bookmark-4",
            now="2026-05-26T01:23:45Z",
        )
        assert snap.created_at == "2026-05-26T01:23:45Z"

    def test_explicit_snapshot_id_used(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="bookmark-5",
            snapshot_id="abcdef012345",
        )
        assert snap.id == "abcdef012345"

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            create_codebook_snapshot(
                tmp_path, _HEX_PROJECT, name="will fail"
            )


# --------------------------------------------------------------------------- #
# Reconstruction helpers
# --------------------------------------------------------------------------- #


class TestReconstruction:
    def test_reconstruct_codes_returns_code_instances(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        c1 = Code.new(project_id=project.id, name="a")
        c2 = Code.new(project_id=project.id, name="b")
        snap = Snapshot.new(
            project_id=project.id,
            name="r",
            codes=[c1, c2],
        )
        rebuilt = reconstruct_codes_from_snapshot(snap)
        assert all(isinstance(c, Code) for c in rebuilt)
        assert {c.id for c in rebuilt} == {c1.id, c2.id}

    def test_render_at_snapshot_uses_embedded_codes_not_live(
        self, tmp_path: Path
    ) -> None:
        # The whole point of F9.3: even after the live codebook
        # changes, the snapshot keeps the original wording.
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="early-name", definition="def-1")
        save_code_with_version(tmp_path, c)
        snap = create_codebook_snapshot(tmp_path, project.id, name="t1")

        # Now mutate the live code
        c.apply_update({"name": "later-name", "definition": "def-2"})
        save_code_with_version(tmp_path, c)

        # Snapshot still reports the early state.
        out = render_codebook_at_snapshot(snap, format="csv")
        assert "early-name" in out
        assert "def-1" in out
        assert "later-name" not in out

    def test_render_supports_markdown_and_rtf(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="hi", definition="why")
        save_code_with_version(tmp_path, c)
        snap = create_codebook_snapshot(tmp_path, project.id, name="t2")
        md = render_codebook_at_snapshot(snap, format="markdown", project=project)
        assert "hi" in md
        rtf = render_codebook_at_snapshot(snap, format="rtf", project=project)
        # RTF starts with the {\rtf1 header
        assert rtf.startswith(r"{\rtf1") or rtf.startswith("{\\rtf1")

    def test_code_at_snapshot_returns_code_or_none(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="x", definition="d1")
        snap = Snapshot.new(project_id=project.id, name="r", codes=[c])
        found = code_at_snapshot(snap, c.id)
        assert found is not None
        assert isinstance(found, Code)
        assert found.definition == "d1"
        assert code_at_snapshot(snap, "f" * 12) is None

    def test_code_at_snapshot_validates_id(self) -> None:
        snap = Snapshot.new(project_id=_HEX_PROJECT, name="r")
        with pytest.raises(ProjectValidationError):
            code_at_snapshot(snap, "bad")

    def test_code_version_id_at_snapshot(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="x")
        save_code_with_version(tmp_path, c)
        snap = create_codebook_snapshot(tmp_path, project.id, name="r")
        vid = code_version_id_at_snapshot(snap, c.id)
        assert vid is not None
        assert len(vid) == 12

    def test_code_version_id_returns_none_for_unpinned(self) -> None:
        c = Code.new(project_id=_HEX_PROJECT, name="x")
        snap = Snapshot.new(project_id=_HEX_PROJECT, name="r", codes=[c])
        assert code_version_id_at_snapshot(snap, c.id) is None

    def test_code_version_id_validates_id(self) -> None:
        snap = Snapshot.new(project_id=_HEX_PROJECT, name="r")
        with pytest.raises(ProjectValidationError):
            code_version_id_at_snapshot(snap, "bad")


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


class TestSummaries:
    def test_snapshot_summary_shape(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c = Code.new(project_id=project.id, name="x")
        save_code_with_version(tmp_path, c)
        snap = create_codebook_snapshot(
            tmp_path,
            project.id,
            name="bookmark",
            description="why",
            actor_coder_id=_HEX_CODER,
        )
        summary = snapshot_summary(snap)
        assert summary["id"] == snap.id
        assert summary["project_id"] == project.id
        assert summary["name"] == "bookmark"
        assert summary["description"] == "why"
        assert summary["codebook_stage"] == "initial"
        assert summary["actor_coder_id"] == _HEX_CODER
        assert summary["event_id"] == snap.event_id
        assert summary["code_count"] == 1
        assert summary["version_pin_count"] == 1
        assert summary["created_at"] == snap.created_at

    def test_list_snapshot_summaries_returns_list_of_dicts(
        self, tmp_path: Path
    ) -> None:
        project = _saved_project(tmp_path)
        s1 = Snapshot.new(
            project_id=project.id,
            name="first",
            now="2026-05-01T00:00:00Z",
        )
        s2 = Snapshot.new(
            project_id=project.id,
            name="second",
            now="2026-05-02T00:00:00Z",
        )
        save_snapshot(tmp_path, s1)
        save_snapshot(tmp_path, s2)
        out = list_snapshot_summaries(tmp_path, project.id)
        assert [s["name"] for s in out] == ["first", "second"]
        # Summaries should not include the embedded codes list.
        for s in out:
            assert "codes" not in s


# --------------------------------------------------------------------------- #
# Audit event back-write resilience
# --------------------------------------------------------------------------- #


class TestEventEmissionResilience:
    def test_snapshot_survives_event_emission_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _saved_project(tmp_path)

        # Make record_event blow up after the snapshot is saved. The
        # snapshot file should still be on disk; event_id should be
        # left empty; the function should return without raising.
        from scribe import codebook_snapshots as mod

        def boom(*args, **kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("disk full")

        monkeypatch.setattr(mod, "record_event", boom)
        snap = create_codebook_snapshot(tmp_path, project.id, name="resilient")
        assert snap.event_id == ""
        # File still exists and parses fine.
        loaded = load_snapshot(tmp_path, project.id, snap.id)
        assert loaded.id == snap.id
        assert loaded.event_id == ""
