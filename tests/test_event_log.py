"""Tests for scribe.event_log (F9.1).

Covers:
  * Event dataclass round-trip + validation
  * Append-only persistence (save / load / list / count)
  * Filter combinations on list_events
  * compute_diff semantics (added / removed / changed; equal skipped)
  * Convenience emitters (record_event / record_create / record_update / record_delete)
  * Payload size + depth caps
  * Refusal to overwrite an existing event id
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.event_log import (
    DIFF_OP_ADDED,
    DIFF_OP_CHANGED,
    DIFF_OP_REMOVED,
    DIFF_OPS,
    EVENT_ACTION_CREATE,
    EVENT_ACTION_DELETE,
    EVENT_ACTION_LOCK,
    EVENT_ACTION_OTHER,
    EVENT_ACTION_RENAME,
    EVENT_ACTION_SNAPSHOT,
    EVENT_ACTION_UPDATE,
    EVENT_ACTIONS,
    EVENT_ENTITY_APPLICATION,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_CODEBOOK,
    EVENT_ENTITY_MEMO,
    EVENT_ENTITY_PROJECT,
    EVENT_ENTITY_SOURCE,
    EVENT_ENTITY_TYPES,
    EVENT_ID_RE,
    EVENTS_DIRNAME,
    Event,
    MAX_DIFF_ENTRIES,
    MAX_NOTES_LEN,
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_DEPTH,
    MAX_PAYLOAD_KEYS,
    MAX_PAYLOAD_LIST_LEN,
    MAX_PAYLOAD_STRING_LEN,
    compute_diff,
    count_events,
    event_state_path,
    events_dir,
    list_events,
    load_event,
    new_event_id,
    record_create,
    record_delete,
    record_event,
    record_update,
    save_event,
)


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


# --------------------------------------------------------------------------- #
# new_event_id + EVENT_ID_RE shape
# --------------------------------------------------------------------------- #


class TestNewEventId:
    def test_returns_12_char_hex(self) -> None:
        eid = new_event_id()
        assert EVENT_ID_RE.match(eid)
        assert len(eid) == 12

    def test_returns_unique_ids(self) -> None:
        ids = {new_event_id() for _ in range(64)}
        assert len(ids) == 64


# --------------------------------------------------------------------------- #
# Vocabulary sanity
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_actions_includes_core_ops(self) -> None:
        for a in (
            EVENT_ACTION_CREATE,
            EVENT_ACTION_UPDATE,
            EVENT_ACTION_DELETE,
            EVENT_ACTION_RENAME,
            EVENT_ACTION_LOCK,
            EVENT_ACTION_SNAPSHOT,
            EVENT_ACTION_OTHER,
        ):
            assert a in EVENT_ACTIONS

    def test_entity_types_cover_f1_to_f6_modules(self) -> None:
        for e in (
            EVENT_ENTITY_PROJECT,
            EVENT_ENTITY_SOURCE,
            EVENT_ENTITY_CODE,
            EVENT_ENTITY_APPLICATION,
            EVENT_ENTITY_MEMO,
            EVENT_ENTITY_CODEBOOK,
        ):
            assert e in EVENT_ENTITY_TYPES

    def test_diff_ops_are_three_canonical(self) -> None:
        assert set(DIFF_OPS) == {DIFF_OP_ADDED, DIFF_OP_REMOVED, DIFF_OP_CHANGED}


# --------------------------------------------------------------------------- #
# compute_diff
# --------------------------------------------------------------------------- #


class TestComputeDiff:
    def test_no_change_returns_empty_list(self) -> None:
        assert compute_diff({"a": 1}, {"a": 1}) == []

    def test_changed_value_emits_changed_row(self) -> None:
        diff = compute_diff({"a": 1}, {"a": 2})
        assert diff == [{"path": "a", "op": DIFF_OP_CHANGED, "before": 1, "after": 2}]

    def test_added_key_emits_added_row(self) -> None:
        diff = compute_diff({}, {"a": "hello"})
        assert diff == [{"path": "a", "op": DIFF_OP_ADDED, "after": "hello"}]

    def test_removed_key_emits_removed_row(self) -> None:
        diff = compute_diff({"a": "hello"}, {})
        assert diff == [{"path": "a", "op": DIFF_OP_REMOVED, "before": "hello"}]

    def test_none_before_acts_as_empty(self) -> None:
        diff = compute_diff(None, {"a": 1})
        assert diff == [{"path": "a", "op": DIFF_OP_ADDED, "after": 1}]

    def test_none_after_acts_as_empty(self) -> None:
        diff = compute_diff({"a": 1}, None)
        assert diff == [{"path": "a", "op": DIFF_OP_REMOVED, "before": 1}]

    def test_both_none_returns_empty(self) -> None:
        assert compute_diff(None, None) == []

    def test_multiple_keys_sorted(self) -> None:
        diff = compute_diff(
            {"name": "old", "stage": "initial", "deleted_at": "2026-01-01"},
            {"name": "new", "stage": "initial", "added_at": "2026-02-01"},
        )
        # sorted by key name
        assert [r["path"] for r in diff] == ["added_at", "deleted_at", "name"]
        assert diff[0]["op"] == DIFF_OP_ADDED
        assert diff[1]["op"] == DIFF_OP_REMOVED
        assert diff[2]["op"] == DIFF_OP_CHANGED

    def test_deep_equality_skips_unchanged_lists(self) -> None:
        diff = compute_diff({"x": [1, 2, 3]}, {"x": [1, 2, 3]})
        assert diff == []

    def test_deep_equality_picks_up_changed_lists(self) -> None:
        diff = compute_diff({"x": [1, 2]}, {"x": [1, 2, 3]})
        assert len(diff) == 1
        assert diff[0]["op"] == DIFF_OP_CHANGED
        assert diff[0]["after"] == [1, 2, 3]

    def test_rejects_non_mapping_inputs(self) -> None:
        with pytest.raises(ProjectValidationError):
            compute_diff([], {"a": 1})  # type: ignore[arg-type]
        with pytest.raises(ProjectValidationError):
            compute_diff({"a": 1}, "not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Event construction + validation
# --------------------------------------------------------------------------- #


class TestEventConstruction:
    def test_minimum_required_fields(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
        )
        assert EVENT_ID_RE.match(ev.id)
        assert ev.project_id == _HEX_PROJECT
        assert ev.action == EVENT_ACTION_CREATE
        assert ev.entity_type == EVENT_ENTITY_CODE
        assert ev.entity_id == ""
        assert ev.actor_coder_id == ""
        assert ev.before is None
        assert ev.after is None
        assert ev.diff == []
        assert ev.notes == ""
        # created_at filled at construction
        assert ev.created_at and ev.created_at.endswith("Z")

    def test_explicit_event_id_used(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_PROJECT,
            event_id="abcdef012345",
        )
        assert ev.id == "abcdef012345"

    def test_explicit_now_used(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            now="2026-05-26T12:00:00Z",
        )
        assert ev.created_at == "2026-05-26T12:00:00Z"

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action="frobnicate",
                entity_type=EVENT_ENTITY_CODE,
            )

    def test_invalid_entity_type_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type="banana",
            )

    def test_invalid_project_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id="not-hex",
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
            )

    def test_invalid_entity_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                entity_id="not-hex",
            )

    def test_empty_entity_id_allowed(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_SNAPSHOT,
            entity_type=EVENT_ENTITY_CODEBOOK,
            entity_id="",
        )
        assert ev.entity_id == ""

    def test_invalid_actor_coder_id_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                actor_coder_id="not-hex",
            )

    def test_notes_overflow_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                notes="x" * (MAX_NOTES_LEN + 1),
            )


class TestEventPayloadValidation:
    def test_before_must_be_dict_or_none(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                before=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_too_many_keys_rejected(self) -> None:
        big = {f"k{i}": i for i in range(MAX_PAYLOAD_KEYS + 1)}
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after=big,
            )

    def test_oversize_string_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after={"name": "x" * (MAX_PAYLOAD_STRING_LEN + 1)},
            )

    def test_oversize_list_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after={"items": list(range(MAX_PAYLOAD_LIST_LEN + 1))},
            )

    def test_too_deep_nesting_rejected(self) -> None:
        # Build a payload deeper than MAX_PAYLOAD_DEPTH
        node: dict = {"leaf": 1}
        for _ in range(MAX_PAYLOAD_DEPTH + 2):
            node = {"child": node}
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after=node,
            )

    def test_combined_payload_too_big_rejected(self) -> None:
        # A 200 KiB string fits under MAX_PAYLOAD_STRING_LEN (16k)? No:
        # build many keys with allowed-sized strings but combined > cap.
        chunk = "x" * MAX_PAYLOAD_STRING_LEN
        # 16 KiB * 32 = 512 KiB > 256 KiB cap
        big = {f"k{i:02d}": chunk for i in range(32)}
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after=big,
            )

    def test_supports_nested_lists_and_dicts(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={
                "name": "Tea drinking",
                "exemplars": ["one", "two", "three"],
                "metadata": {"weight": 1.5, "tags": ["a", "b"]},
            },
        )
        assert ev.after is not None
        assert ev.after["metadata"]["tags"] == ["a", "b"]


class TestEventDiffValidation:
    def test_diff_must_be_list_of_objects(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                diff=[{"path": "x"}],  # missing op
            )

    def test_diff_unknown_op_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                diff=[{"path": "x", "op": "nuked"}],
            )

    def test_diff_path_required(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                diff=[{"path": "", "op": DIFF_OP_CHANGED, "before": 1, "after": 2}],
            )

    def test_diff_too_many_entries_rejected(self) -> None:
        diff = [
            {"path": f"k{i}", "op": DIFF_OP_ADDED, "after": i}
            for i in range(MAX_DIFF_ENTRIES + 1)
        ]
        with pytest.raises(ProjectValidationError):
            Event.new(
                project_id=_HEX_PROJECT,
                action=EVENT_ACTION_UPDATE,
                entity_type=EVENT_ENTITY_CODE,
                diff=diff,
            )

    def test_with_computed_diff_fills_diff_from_before_after(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "old"},
            after={"name": "new"},
        )
        assert ev.diff == []  # not auto-computed in Event.new
        ev2 = ev.with_computed_diff()
        assert ev2.diff == [
            {"path": "name", "op": DIFF_OP_CHANGED, "before": "old", "after": "new"}
        ]

    def test_with_computed_diff_preserves_existing_diff(self) -> None:
        existing = [{"path": "name", "op": DIFF_OP_CHANGED, "before": "x", "after": "y"}]
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            before={"name": "x"},
            after={"name": "y"},
            diff=existing,
        )
        ev2 = ev.with_computed_diff()
        assert ev2.diff == existing


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestEventRoundTrip:
    def test_to_and_from_dict_round_trip(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            actor_coder_id=_HEX_CODER,
            before={"name": "old"},
            after={"name": "new", "stage": "focused"},
            diff=[
                {"path": "name", "op": DIFF_OP_CHANGED, "before": "old", "after": "new"},
                {"path": "stage", "op": DIFF_OP_ADDED, "after": "focused"},
            ],
            notes="renamed via UI",
        )
        d = ev.to_dict()
        rt = Event.from_dict(d)
        assert rt == ev

    def test_to_dict_clones_payloads(self) -> None:
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"items": [1, 2, 3]},
        )
        d = ev.to_dict()
        assert ev.after is not None
        d["after"]["items"].append(99)
        # Mutating the returned dict must not bleed back into the event.
        assert ev.after["items"] == [1, 2, 3]

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_from_dict_requires_action_and_entity_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            Event.from_dict({"id": "0" * 12, "project_id": _HEX_PROJECT})


# --------------------------------------------------------------------------- #
# Persistence — append-only on disk
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_round_trips_via_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = Event.new(
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            actor_coder_id=_HEX_CODER,
            after={"name": "Tea drinking"},
        )
        path = save_event(tmp_path, ev)
        assert path.exists()
        loaded = load_event(tmp_path, proj.id, ev.id)
        assert loaded == ev

    def test_save_creates_events_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = Event.new(
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"name": "x"},
        )
        save_event(tmp_path, ev)
        assert events_dir(tmp_path, proj.id).is_dir()
        assert events_dir(tmp_path, proj.id).name == EVENTS_DIRNAME

    def test_event_state_path_uses_event_id(self, tmp_path: Path) -> None:
        eid = "abcdef012345"
        p = event_state_path(tmp_path, _HEX_PROJECT, eid)
        assert p.name == f"{eid}.json"

    def test_event_state_path_rejects_bad_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            event_state_path(tmp_path, _HEX_PROJECT, "not-hex")

    def test_save_refuses_when_project_missing(self, tmp_path: Path) -> None:
        # No project dir yet
        ev = Event.new(
            project_id=_HEX_PROJECT,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"x": 1},
        )
        with pytest.raises(FileNotFoundError):
            save_event(tmp_path, ev)

    def test_save_refuses_to_overwrite_existing_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = Event.new(
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"x": 1},
        )
        save_event(tmp_path, ev)
        with pytest.raises(FileExistsError):
            save_event(tmp_path, ev)

    def test_save_writes_via_temp_file(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = Event.new(
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"x": 1},
        )
        save_event(tmp_path, ev)
        ed = events_dir(tmp_path, proj.id)
        # No leftover .tmp file
        assert not list(ed.glob("*.tmp"))

    def test_load_missing_raises_filenotfound(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_event(tmp_path, proj.id, "ffffffffffff")

    def test_no_delete_helper_exposed(self) -> None:
        # F9.1 explicitly disallows event deletion. There is no
        # ``delete_event`` symbol; this test pins that.
        import scribe.event_log as mod
        assert not hasattr(mod, "delete_event")


# --------------------------------------------------------------------------- #
# list_events
# --------------------------------------------------------------------------- #


class TestListEvents:
    def _three_events(self, tmp_path: Path) -> tuple[Project, list[Event]]:
        proj = _saved_project(tmp_path)
        ev1 = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            actor_coder_id=_HEX_CODER,
            after={"name": "One"},
            now="2026-05-26T10:00:00Z",
        )
        ev2 = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            actor_coder_id=_HEX_CODER,
            before={"name": "One"},
            after={"name": "OnePrime"},
            now="2026-05-26T11:00:00Z",
        )
        ev3 = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_LOCK,
            entity_type=EVENT_ENTITY_CODEBOOK,
            actor_coder_id=_HEX_CODER,
            now="2026-05-26T12:00:00Z",
        )
        return proj, [ev1, ev2, ev3]

    def test_returns_all_chronologically(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        out = list_events(tmp_path, proj.id)
        assert [e.id for e in out] == [evs[0].id, evs[1].id, evs[2].id]

    def test_filter_by_action(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        creates = list_events(tmp_path, proj.id, action=EVENT_ACTION_CREATE)
        assert [e.id for e in creates] == [evs[0].id]

    def test_filter_by_entity_type(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        cb = list_events(tmp_path, proj.id, entity_type=EVENT_ENTITY_CODEBOOK)
        assert [e.id for e in cb] == [evs[2].id]

    def test_filter_by_entity_id(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        out = list_events(tmp_path, proj.id, entity_id=_HEX_CODE)
        assert [e.id for e in out] == [evs[0].id, evs[1].id]

    def test_filter_by_actor(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        out = list_events(tmp_path, proj.id, actor_coder_id=_HEX_CODER)
        assert len(out) == 3
        out2 = list_events(tmp_path, proj.id, actor_coder_id="d" * 12)
        assert out2 == []

    def test_filter_since_until(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        out = list_events(
            tmp_path, proj.id,
            since="2026-05-26T10:30:00Z",
            until="2026-05-26T11:30:00Z",
        )
        assert [e.id for e in out] == [evs[1].id]

    def test_filter_invalid_action_raises(self, tmp_path: Path) -> None:
        proj, _ = self._three_events(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_events(tmp_path, proj.id, action="frobnicate")

    def test_filter_invalid_entity_type_raises(self, tmp_path: Path) -> None:
        proj, _ = self._three_events(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_events(tmp_path, proj.id, entity_type="banana")

    def test_filter_invalid_actor_raises(self, tmp_path: Path) -> None:
        proj, _ = self._three_events(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_events(tmp_path, proj.id, actor_coder_id="not-hex")

    def test_missing_project_returns_empty(self, tmp_path: Path) -> None:
        # Project dir doesn't exist at all
        assert list_events(tmp_path, _HEX_PROJECT) == []

    def test_corrupt_file_skipped(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        ed = events_dir(tmp_path, proj.id)
        (ed / "deadbeefcafe.json").write_text("{ not valid json")
        out = list_events(tmp_path, proj.id)
        # Should still get our three back without raising
        assert {e.id for e in out} == {evs[0].id, evs[1].id, evs[2].id}

    def test_non_hex_filename_skipped(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        ed = events_dir(tmp_path, proj.id)
        (ed / "README.json").write_text(json.dumps({"hi": 1}))
        out = list_events(tmp_path, proj.id)
        assert len(out) == 3

    def test_temp_files_skipped(self, tmp_path: Path) -> None:
        proj, evs = self._three_events(tmp_path)
        ed = events_dir(tmp_path, proj.id)
        (ed / "abcdef012345.json.tmp").write_text("{}")
        out = list_events(tmp_path, proj.id)
        assert len(out) == 3


# --------------------------------------------------------------------------- #
# count_events
# --------------------------------------------------------------------------- #


class TestCountEvents:
    def test_no_dir_returns_zero(self, tmp_path: Path) -> None:
        assert count_events(tmp_path, _HEX_PROJECT) == 0

    def test_counts_only_valid_filenames(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        for _ in range(3):
            record_event(
                tmp_path,
                project_id=proj.id,
                action=EVENT_ACTION_CREATE,
                entity_type=EVENT_ENTITY_CODE,
                after={"x": 1},
            )
        ed = events_dir(tmp_path, proj.id)
        # Stray non-conforming files are ignored
        (ed / "README.json").write_text("{}")
        (ed / "abcdef012345.json.tmp").write_text("{}")
        assert count_events(tmp_path, proj.id) == 3


# --------------------------------------------------------------------------- #
# Convenience emitters
# --------------------------------------------------------------------------- #


class TestRecordEvent:
    def test_record_event_persists_and_returns_event(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            after={"name": "Tea"},
        )
        # Round-trips from disk
        loaded = load_event(tmp_path, proj.id, ev.id)
        assert loaded == ev

    def test_record_event_auto_diffs_by_default(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "old"},
            after={"name": "new"},
        )
        assert ev.diff == [
            {"path": "name", "op": DIFF_OP_CHANGED, "before": "old", "after": "new"}
        ]

    def test_record_event_auto_diff_can_be_disabled(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "old"},
            after={"name": "new"},
            auto_diff=False,
        )
        assert ev.diff == []

    def test_record_event_explicit_diff_used_as_is(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        custom_diff = [
            {"path": "name", "op": DIFF_OP_CHANGED, "before": "x", "after": "y"}
        ]
        ev = record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "x"},
            after={"name": "y"},
            diff=custom_diff,
        )
        assert ev.diff == custom_diff


class TestRecordCreateUpdateDelete:
    def test_record_create_sets_action(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_create(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            after={"name": "New code"},
            actor_coder_id=_HEX_CODER,
        )
        assert ev.action == EVENT_ACTION_CREATE
        assert ev.before is None
        assert ev.after == {"name": "New code"}
        assert ev.diff == [
            {"path": "name", "op": DIFF_OP_ADDED, "after": "New code"}
        ]

    def test_record_update_sets_action_and_diff(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_update(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "Tea"},
            after={"name": "Tea drinking"},
            actor_coder_id=_HEX_CODER,
            notes="Charmaz gerund-form rename",
        )
        assert ev.action == EVENT_ACTION_UPDATE
        assert ev.before == {"name": "Tea"}
        assert ev.after == {"name": "Tea drinking"}
        assert ev.diff == [
            {
                "path": "name",
                "op": DIFF_OP_CHANGED,
                "before": "Tea",
                "after": "Tea drinking",
            }
        ]
        assert ev.notes == "Charmaz gerund-form rename"

    def test_record_delete_sets_action(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_delete(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "Retired code"},
            actor_coder_id=_HEX_CODER,
        )
        assert ev.action == EVENT_ACTION_DELETE
        assert ev.before == {"name": "Retired code"}
        assert ev.after is None
        # diff has one ``removed`` row for the surviving key
        assert ev.diff == [
            {"path": "name", "op": DIFF_OP_REMOVED, "before": "Retired code"}
        ]

    def test_record_delete_persists_to_disk(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_delete(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_MEMO,
            entity_id="d" * 12,
            before={"body": "..."},
        )
        loaded = load_event(tmp_path, proj.id, ev.id)
        assert loaded.action == EVENT_ACTION_DELETE
        assert loaded.before == {"body": "..."}


# --------------------------------------------------------------------------- #
# Realistic workflow — three operations end-to-end via the audit trail
# --------------------------------------------------------------------------- #


class TestAuditTrailWorkflow:
    def test_create_then_update_then_lock_appears_in_log(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)

        # 1. Create a code
        record_create(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            after={"name": "Resisting", "stage": "initial"},
            actor_coder_id=_HEX_CODER,
            now="2026-05-26T08:00:00Z",
        )
        # 2. Update its stage as the analysis matures
        record_update(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            before={"name": "Resisting", "stage": "initial"},
            after={"name": "Resisting", "stage": "focused"},
            actor_coder_id=_HEX_CODER,
            now="2026-05-26T09:00:00Z",
        )
        # 3. Lock the codebook
        record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_LOCK,
            entity_type=EVENT_ENTITY_CODEBOOK,
            actor_coder_id=_HEX_CODER,
            notes="Initial coding done; locking before second-coder pass",
            now="2026-05-26T10:00:00Z",
        )

        log = list_events(tmp_path, proj.id)
        assert [e.action for e in log] == [
            EVENT_ACTION_CREATE,
            EVENT_ACTION_UPDATE,
            EVENT_ACTION_LOCK,
        ]
        assert log[1].diff == [
            {
                "path": "stage",
                "op": DIFF_OP_CHANGED,
                "before": "initial",
                "after": "focused",
            }
        ]
        assert log[2].notes.startswith("Initial coding")
        assert count_events(tmp_path, proj.id) == 3

    def test_audit_trail_is_immutable_after_write(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = record_create(
            tmp_path,
            project_id=proj.id,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE,
            after={"name": "x"},
        )
        # Re-saving the same event id MUST refuse — no overwriting history.
        with pytest.raises(FileExistsError):
            save_event(tmp_path, ev)
