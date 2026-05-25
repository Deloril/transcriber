"""Tests for scribe.time_travel (F9.8).

Covers:
  * Pure helpers: pick_version_at, replay_entity_states,
    lock_state_from_log, filter_by_created_at.
  * Per-entity reconstructors: codes / project / applications / memos
    / sources / participants / codebook lock state.
  * The aggregator reconstruct_state_at + ProjectStateAtTime.to_dict.
  * Edge cases: empty inputs, invalid as_of, deleted codes, codes with
    no version log, modified-after-as_of warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.applications import Application, save_application
from scribe.codebook_lock import LockEvent, append_lock_event
from scribe.code_versions import (
    CodeVersion,
    record_code_version,
    save_code_with_version,
)
from scribe.codes import Code, code_state_path, save_code
from scribe.event_log import (
    EVENT_ACTION_CREATE,
    EVENT_ACTION_DELETE,
    EVENT_ACTION_UPDATE,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_PROJECT,
    Event,
    record_event,
)
from scribe.memos import Memo, save_memo
from scribe.participants import Participant, save_participant
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.sources import Source, save_source
from scribe.time_travel import (
    ENTITY_KIND_CODE,
    ENTITY_KIND_PROJECT,
    ENTITY_KINDS,
    ProjectStateAtTime,
    filter_by_created_at,
    lock_state_from_log,
    pick_version_at,
    reconstruct_applications_at,
    reconstruct_codebook_lock_state_at,
    reconstruct_codes_at,
    reconstruct_memos_at,
    reconstruct_participants_at,
    reconstruct_project_at,
    reconstruct_sources_at,
    reconstruct_state_at,
    replay_entity_states,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE_A = "a" * 12
_HEX_CODE_B = "b" * 12
_HEX_CODER = "c" * 12


def _saved_project(
    tmp_path: Path,
    *,
    stage: str = "initial",
    now: str = "2026-01-01T00:00:00.000000Z",
    project_id: str = _HEX_PROJECT,
) -> Project:
    p = Project.new(
        name="P", codebook_stage=stage, now=now, project_id=project_id
    )
    save_project(tmp_path, p)
    return p


def _saved_code(
    tmp_path: Path,
    project: Project,
    *,
    name: str = "code",
    definition: str = "",
    code_id: str | None = None,
    now: str = "2026-01-01T00:00:00.000000Z",
) -> Code:
    c = Code.new(
        project_id=project.id,
        name=name,
        definition=definition,
        code_id=code_id,
        now=now,
    )
    save_code(tmp_path, c)
    return c


# --------------------------------------------------------------------------- #
# pick_version_at
# --------------------------------------------------------------------------- #


class TestPickVersionAt:
    def _make_v(self, *, version: int, ts: str) -> CodeVersion:
        c = Code.new(project_id=_HEX_PROJECT, name=f"c{version}", code_id=_HEX_CODE_A)
        return CodeVersion.new(code=c, version=version, now=ts)

    def test_returns_none_when_empty(self) -> None:
        assert pick_version_at([], "2026-01-01T00:00:00.000000Z") is None

    def test_returns_none_when_all_after(self) -> None:
        v1 = self._make_v(version=1, ts="2026-06-01T00:00:00.000000Z")
        v2 = self._make_v(version=2, ts="2026-06-02T00:00:00.000000Z")
        assert pick_version_at([v1, v2], "2026-01-01T00:00:00.000000Z") is None

    def test_returns_latest_eligible(self) -> None:
        v1 = self._make_v(version=1, ts="2026-01-01T00:00:00.000000Z")
        v2 = self._make_v(version=2, ts="2026-02-01T00:00:00.000000Z")
        v3 = self._make_v(version=3, ts="2026-06-01T00:00:00.000000Z")
        chosen = pick_version_at([v1, v2, v3], "2026-03-01T00:00:00.000000Z")
        assert chosen is not None
        assert chosen.version == 2

    def test_inclusive_at_exact_boundary(self) -> None:
        v1 = self._make_v(version=1, ts="2026-03-01T00:00:00.000000Z")
        chosen = pick_version_at([v1], "2026-03-01T00:00:00.000000Z")
        assert chosen is not None
        assert chosen.version == 1

    def test_unsorted_input_handled(self) -> None:
        v1 = self._make_v(version=1, ts="2026-01-01T00:00:00.000000Z")
        v2 = self._make_v(version=2, ts="2026-02-01T00:00:00.000000Z")
        v3 = self._make_v(version=3, ts="2026-06-01T00:00:00.000000Z")
        chosen = pick_version_at([v3, v1, v2], "2026-03-01T00:00:00.000000Z")
        assert chosen is not None
        assert chosen.version == 2

    def test_skips_versions_with_empty_timestamp(self) -> None:
        v_bad = CodeVersion(
            id="d" * 12,
            code_id=_HEX_CODE_A,
            project_id=_HEX_PROJECT,
            version=1,
            created_at="",  # malformed
            snapshot={"id": _HEX_CODE_A, "project_id": _HEX_PROJECT, "name": "x"},
        )
        v_good = self._make_v(version=2, ts="2026-02-01T00:00:00.000000Z")
        chosen = pick_version_at([v_bad, v_good], "2026-03-01T00:00:00.000000Z")
        assert chosen is not None
        assert chosen.version == 2

    def test_invalid_as_of_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            pick_version_at([], "")

    def test_tiebreaker_prefers_higher_version(self) -> None:
        # Same timestamp, different version numbers: the higher version
        # wins (matches the canonical "latest revision" intuition).
        v1 = self._make_v(version=1, ts="2026-01-01T00:00:00.000000Z")
        v2 = self._make_v(version=2, ts="2026-01-01T00:00:00.000000Z")
        chosen = pick_version_at([v1, v2], "2026-06-01T00:00:00.000000Z")
        assert chosen is not None
        assert chosen.version == 2


# --------------------------------------------------------------------------- #
# replay_entity_states
# --------------------------------------------------------------------------- #


class TestReplayEntityStates:
    def _ev(self, **kw: object) -> Event:
        defaults = {
            "project_id": _HEX_PROJECT,
            "action": EVENT_ACTION_CREATE,
            "entity_type": EVENT_ENTITY_CODE,
            "entity_id": _HEX_CODE_A,
        }
        defaults.update(kw)
        return Event.new(**defaults)  # type: ignore[arg-type]

    def test_empty_returns_empty(self) -> None:
        assert replay_entity_states([]) == {}

    def test_single_create_records_after(self) -> None:
        ev = self._ev(after={"id": _HEX_CODE_A, "name": "a"})
        out = replay_entity_states([ev])
        assert out == {_HEX_CODE_A: {"id": _HEX_CODE_A, "name": "a"}}

    def test_update_overrides_create(self) -> None:
        ev1 = self._ev(
            action=EVENT_ACTION_CREATE,
            after={"id": _HEX_CODE_A, "name": "a"},
            now="2026-01-01T00:00:00.000000Z",
        )
        ev2 = self._ev(
            action=EVENT_ACTION_UPDATE,
            before={"id": _HEX_CODE_A, "name": "a"},
            after={"id": _HEX_CODE_A, "name": "b"},
            now="2026-01-02T00:00:00.000000Z",
        )
        out = replay_entity_states([ev2, ev1])  # purposely unsorted
        assert out[_HEX_CODE_A] == {"id": _HEX_CODE_A, "name": "b"}

    def test_delete_sets_none(self) -> None:
        ev1 = self._ev(
            action=EVENT_ACTION_CREATE,
            after={"id": _HEX_CODE_A, "name": "a"},
            now="2026-01-01T00:00:00.000000Z",
        )
        ev2 = self._ev(
            action=EVENT_ACTION_DELETE,
            before={"id": _HEX_CODE_A, "name": "a"},
            after=None,
            now="2026-01-02T00:00:00.000000Z",
        )
        out = replay_entity_states([ev1, ev2])
        assert out == {_HEX_CODE_A: None}

    def test_filter_by_as_of(self) -> None:
        ev1 = self._ev(
            after={"id": _HEX_CODE_A, "name": "a"},
            now="2026-01-01T00:00:00.000000Z",
        )
        ev2 = self._ev(
            action=EVENT_ACTION_UPDATE,
            before={"id": _HEX_CODE_A, "name": "a"},
            after={"id": _HEX_CODE_A, "name": "b"},
            now="2026-06-01T00:00:00.000000Z",
        )
        out = replay_entity_states(
            [ev1, ev2], as_of="2026-03-01T00:00:00.000000Z"
        )
        assert out[_HEX_CODE_A] == {"id": _HEX_CODE_A, "name": "a"}

    def test_filter_by_entity_type(self) -> None:
        ev_code = self._ev(after={"id": _HEX_CODE_A, "name": "c"})
        ev_proj = self._ev(
            entity_type=EVENT_ENTITY_PROJECT,
            entity_id=_HEX_PROJECT,
            after={"id": _HEX_PROJECT, "name": "p"},
        )
        out = replay_entity_states(
            [ev_code, ev_proj], entity_type=EVENT_ENTITY_CODE
        )
        assert _HEX_CODE_A in out
        assert _HEX_PROJECT not in out

    def test_skips_events_with_empty_entity_id(self) -> None:
        # A snapshot or similar project-wide event has entity_id ""
        # *unless* the snapshot id is provided. Either way, empty
        # entity_id should not crash; it just isn't assigned a slot.
        ev = self._ev(
            entity_type=EVENT_ENTITY_PROJECT,
            entity_id="",
            after={"id": _HEX_PROJECT, "name": "p"},
        )
        assert replay_entity_states([ev]) == {}


# --------------------------------------------------------------------------- #
# lock_state_from_log
# --------------------------------------------------------------------------- #


class TestLockStateFromLog:
    def _lock_event(
        self,
        *,
        action: str,
        ts: str,
        new_stage: str = "",
        prior_stage: str = "",
    ) -> LockEvent:
        return LockEvent.new(
            project_id=_HEX_PROJECT,
            action=action,
            reason="x" * 5,
            prior_stage=prior_stage,
            new_stage=new_stage,
            methodological_memo=("y" * 12) if action == "unlock" else "",
            now=ts,
        )

    def test_no_events_returns_fallback_unlocked(self) -> None:
        locked, stage = lock_state_from_log(
            [], "2026-06-01T00:00:00.000000Z", fallback_stage="focused"
        )
        assert locked is False
        assert stage == "focused"

    def test_garbage_fallback_coerced_to_initial(self) -> None:
        locked, stage = lock_state_from_log(
            [], "2026-06-01T00:00:00.000000Z", fallback_stage="???"
        )
        assert locked is False
        assert stage == "initial"

    def test_lock_event_in_window(self) -> None:
        ev = self._lock_event(
            action="lock",
            ts="2026-03-01T00:00:00.000000Z",
            prior_stage="focused",
            new_stage="locked",
        )
        locked, stage = lock_state_from_log([ev], "2026-06-01T00:00:00.000000Z")
        assert locked is True
        assert stage == "locked"

    def test_unlock_after_lock(self) -> None:
        e1 = self._lock_event(
            action="lock",
            ts="2026-03-01T00:00:00.000000Z",
            prior_stage="focused",
            new_stage="locked",
        )
        e2 = self._lock_event(
            action="unlock",
            ts="2026-04-01T00:00:00.000000Z",
            prior_stage="locked",
            new_stage="theoretical",
        )
        locked, stage = lock_state_from_log(
            [e1, e2], "2026-06-01T00:00:00.000000Z"
        )
        assert locked is False
        assert stage == "theoretical"

    def test_as_of_before_unlock_returns_locked(self) -> None:
        e1 = self._lock_event(
            action="lock",
            ts="2026-03-01T00:00:00.000000Z",
            prior_stage="focused",
            new_stage="locked",
        )
        e2 = self._lock_event(
            action="unlock",
            ts="2026-04-01T00:00:00.000000Z",
            prior_stage="locked",
            new_stage="theoretical",
        )
        # Pick a moment between the lock and the unlock.
        locked, stage = lock_state_from_log(
            [e1, e2], "2026-03-15T00:00:00.000000Z"
        )
        assert locked is True
        assert stage == "locked"

    def test_invalid_as_of_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            lock_state_from_log([], "")

    def test_unsorted_input_handled(self) -> None:
        e1 = self._lock_event(
            action="lock",
            ts="2026-03-01T00:00:00.000000Z",
            prior_stage="focused",
            new_stage="locked",
        )
        e2 = self._lock_event(
            action="unlock",
            ts="2026-04-01T00:00:00.000000Z",
            prior_stage="locked",
            new_stage="theoretical",
        )
        locked, stage = lock_state_from_log(
            [e2, e1], "2026-06-01T00:00:00.000000Z"
        )
        assert locked is False
        assert stage == "theoretical"


# --------------------------------------------------------------------------- #
# filter_by_created_at
# --------------------------------------------------------------------------- #


class TestFilterByCreatedAt:
    def test_filters_inclusive(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c1 = _saved_code(
            tmp_path, proj, name="a", code_id="a" * 12,
            now="2026-01-01T00:00:00.000000Z",
        )
        c2 = _saved_code(
            tmp_path, proj, name="b", code_id="b" * 12,
            now="2026-06-01T00:00:00.000000Z",
        )
        out = filter_by_created_at([c1, c2], "2026-03-01T00:00:00.000000Z")
        assert [c.id for c in out] == [c1.id]

    def test_skips_empty_timestamps(self) -> None:
        class F:
            def __init__(self, ts: str) -> None:
                self.created_at = ts

        out = filter_by_created_at(
            [F(""), F("2026-01-01T00:00:00.000000Z")],
            "2026-06-01T00:00:00.000000Z",
        )
        assert len(out) == 1

    def test_non_string_timestamp_raises(self) -> None:
        class F:
            def __init__(self) -> None:
                self.created_at = 42  # type: ignore[assignment]

        with pytest.raises(ProjectValidationError):
            filter_by_created_at([F()], "2026-06-01T00:00:00.000000Z")

    def test_invalid_as_of_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            filter_by_created_at([], "")


# --------------------------------------------------------------------------- #
# reconstruct_codes_at
# --------------------------------------------------------------------------- #


class TestReconstructCodesAt:
    def test_returns_definition_at_time(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="alpha", code_id=_HEX_CODE_A,
            definition="v1",
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-01-01T00:00:00.000000Z")
        # Update definition.
        c2 = Code.from_dict(c.to_dict())
        c2.apply_update({"definition": "v2"}, now="2026-06-01T00:00:00.000000Z")
        save_code_with_version(
            tmp_path, c2, now="2026-06-01T00:00:00.000000Z"
        )

        # Time-travel back to before the second edit.
        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-03-01T00:00:00.000000Z"
        )
        assert len(out) == 1
        assert out[0].id == _HEX_CODE_A
        assert out[0].definition == "v1"

        # And forward to after.
        out2 = reconstruct_codes_at(
            tmp_path, proj.id, "2026-09-01T00:00:00.000000Z"
        )
        assert out2[0].definition == "v2"

    def test_skips_codes_created_after(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="alpha", code_id=_HEX_CODE_A,
            now="2026-06-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-06-01T00:00:00.000000Z")
        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-01-01T00:00:00.000000Z"
        )
        assert out == []

    def test_returns_multiple_codes_sorted_by_name(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c_a = Code.new(
            project_id=proj.id, name="zeta", code_id=_HEX_CODE_A,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c_a, now="2026-01-01T00:00:00.000000Z")
        c_b = Code.new(
            project_id=proj.id, name="alpha", code_id=_HEX_CODE_B,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c_b, now="2026-01-01T00:00:00.000000Z")

        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert [c.name for c in out] == ["alpha", "zeta"]

    def test_excludes_codes_deleted_via_event(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="x", code_id=_HEX_CODE_A,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-01-01T00:00:00.000000Z")
        record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_DELETE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE_A,
            before=c.to_dict(),
            now="2026-02-01T00:00:00.000000Z",
        )
        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert out == []

    def test_recreate_after_delete_brings_code_back(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="x", code_id=_HEX_CODE_A,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-01-01T00:00:00.000000Z")
        record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_DELETE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE_A,
            before=c.to_dict(),
            now="2026-02-01T00:00:00.000000Z",
        )
        record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id=_HEX_CODE_A,
            after=c.to_dict(),
            now="2026-03-01T00:00:00.000000Z",
        )
        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert len(out) == 1
        assert out[0].id == _HEX_CODE_A

    def test_falls_back_to_live_code_with_no_versions(
        self, tmp_path: Path
    ) -> None:
        # Code saved without version log — should still appear when
        # created_at <= as_of.
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="legacy", code_id=_HEX_CODE_A,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code(tmp_path, c)  # no save_code_with_version
        out = reconstruct_codes_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert len(out) == 1
        assert out[0].id == _HEX_CODE_A

    def test_invalid_project_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            reconstruct_codes_at(
                tmp_path, "not-hex", "2026-06-01T00:00:00.000000Z"
            )

    def test_nonexistent_project_returns_empty(self, tmp_path: Path) -> None:
        # No project on disk — every reader returns []; we should
        # cleanly yield no codes rather than crash.
        out = reconstruct_codes_at(
            tmp_path, _HEX_PROJECT, "2026-06-01T00:00:00.000000Z"
        )
        assert out == []


# --------------------------------------------------------------------------- #
# reconstruct_codebook_lock_state_at
# --------------------------------------------------------------------------- #


class TestReconstructCodebookLockStateAt:
    def test_no_log_falls_back_to_project_stage(self, tmp_path: Path) -> None:
        _saved_project(tmp_path, stage="focused")
        locked, stage = reconstruct_codebook_lock_state_at(
            tmp_path, _HEX_PROJECT, "2026-06-01T00:00:00.000000Z"
        )
        assert locked is False
        assert stage == "focused"

    def test_lock_event_recovered(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        ev = LockEvent.new(
            project_id=_HEX_PROJECT,
            action="lock",
            reason="finalised",
            prior_stage="focused",
            new_stage="locked",
            now="2026-03-01T00:00:00.000000Z",
        )
        append_lock_event(tmp_path, ev)
        locked, stage = reconstruct_codebook_lock_state_at(
            tmp_path, _HEX_PROJECT, "2026-06-01T00:00:00.000000Z"
        )
        assert locked is True
        assert stage == "locked"

    def test_unlock_after_lock(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        e1 = LockEvent.new(
            project_id=_HEX_PROJECT,
            action="lock",
            reason="finalised",
            prior_stage="focused",
            new_stage="locked",
            now="2026-03-01T00:00:00.000000Z",
        )
        append_lock_event(tmp_path, e1)
        e2 = LockEvent.new(
            project_id=_HEX_PROJECT,
            action="unlock",
            reason="reopen",
            prior_stage="locked",
            new_stage="theoretical",
            methodological_memo="needed for axial pass",
            now="2026-04-01T00:00:00.000000Z",
        )
        append_lock_event(tmp_path, e2)
        locked, stage = reconstruct_codebook_lock_state_at(
            tmp_path, _HEX_PROJECT, "2026-03-15T00:00:00.000000Z"
        )
        assert locked is True
        locked2, stage2 = reconstruct_codebook_lock_state_at(
            tmp_path, _HEX_PROJECT, "2026-06-01T00:00:00.000000Z"
        )
        assert locked2 is False
        assert stage2 == "theoretical"


# --------------------------------------------------------------------------- #
# reconstruct_project_at
# --------------------------------------------------------------------------- #


class TestReconstructProjectAt:
    def test_returns_none_when_project_not_yet_created(
        self, tmp_path: Path
    ) -> None:
        _saved_project(tmp_path, now="2026-06-01T00:00:00.000000Z")
        # Project was created in June; ask for January.
        p = reconstruct_project_at(
            tmp_path, _HEX_PROJECT, "2026-01-01T00:00:00.000000Z"
        )
        assert p is None

    def test_falls_back_to_live_when_no_events(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path, now="2026-01-01T00:00:00.000000Z")
        p = reconstruct_project_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert p is not None
        assert p.id == proj.id

    def test_uses_event_after_when_present(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path, now="2026-01-01T00:00:00.000000Z")
        # Modify the project state on disk so we can prove the event
        # payload won (not the live).
        proj.apply_update({"name": "live-name"}, now="2026-09-01T00:00:00.000000Z")
        save_project(tmp_path, proj)
        # Record an F9.1 event that captures the project as of June.
        snap_payload = {
            "id": proj.id,
            "name": "event-name",
            "research_question": "",
            "methodology": "",
            "sensitising_concepts": [],
            "codebook_stage": "initial",
            "description": "",
            "settings": {},
            "created_at": "2026-01-01T00:00:00.000000Z",
            "modified_at": "2026-06-01T00:00:00.000000Z",
        }
        record_event(
            tmp_path,
            project_id=proj.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_PROJECT,
            entity_id=proj.id,
            before=None,
            after=snap_payload,
            now="2026-06-01T00:00:00.000000Z",
        )
        p = reconstruct_project_at(
            tmp_path, proj.id, "2026-07-01T00:00:00.000000Z"
        )
        assert p is not None
        assert p.name == "event-name"

    def test_invalid_project_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            reconstruct_project_at(
                tmp_path, "x", "2026-01-01T00:00:00.000000Z"
            )

    def test_invalid_as_of_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            reconstruct_project_at(tmp_path, _HEX_PROJECT, "")


# --------------------------------------------------------------------------- #
# reconstruct_applications_at + memos / sources / participants
# --------------------------------------------------------------------------- #


class TestBestEffortFilters:
    def _saved_app(
        self, tmp_path: Path, project: Project, *, now: str
    ) -> Application:
        a = Application.new(
            project_id=project.id,
            code_id=_HEX_CODE_A,
            source_id="d" * 12,
            coder_id=_HEX_CODER,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            definition_version_id_at_apply="e" * 12,
            now=now,
        )
        save_application(tmp_path, a)
        return a

    def _saved_memo(
        self, tmp_path: Path, project: Project, *, now: str
    ) -> Memo:
        m = Memo.new(
            project_id=project.id,
            type="free",
            body="memo body",
            now=now,
        )
        save_memo(tmp_path, m)
        return m

    def _saved_source(
        self, tmp_path: Path, project: Project, *, now: str
    ) -> Source:
        s = Source.new(project_id=project.id, name="src", now=now)
        save_source(tmp_path, s)
        return s

    def _saved_participant(
        self, tmp_path: Path, project: Project, *, now: str
    ) -> Participant:
        p = Participant.new(project_id=project.id, name="P01", now=now)
        save_participant(tmp_path, p)
        return p

    def test_applications_filtered_by_created_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a1 = self._saved_app(tmp_path, proj, now="2026-01-01T00:00:00.000000Z")
        self._saved_app(tmp_path, proj, now="2026-09-01T00:00:00.000000Z")
        out = reconstruct_applications_at(
            tmp_path, proj.id, "2026-03-01T00:00:00.000000Z"
        )
        assert [a.id for a in out] == [a1.id]

    def test_memos_filtered(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m1 = self._saved_memo(tmp_path, proj, now="2026-01-01T00:00:00.000000Z")
        self._saved_memo(tmp_path, proj, now="2026-09-01T00:00:00.000000Z")
        out = reconstruct_memos_at(
            tmp_path, proj.id, "2026-03-01T00:00:00.000000Z"
        )
        assert [m.id for m in out] == [m1.id]

    def test_sources_filtered(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = self._saved_source(tmp_path, proj, now="2026-01-01T00:00:00.000000Z")
        self._saved_source(tmp_path, proj, now="2026-09-01T00:00:00.000000Z")
        out = reconstruct_sources_at(
            tmp_path, proj.id, "2026-03-01T00:00:00.000000Z"
        )
        assert [s.id for s in out] == [s1.id]

    def test_participants_filtered(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p1 = self._saved_participant(
            tmp_path, proj, now="2026-01-01T00:00:00.000000Z"
        )
        self._saved_participant(
            tmp_path, proj, now="2026-09-01T00:00:00.000000Z"
        )
        out = reconstruct_participants_at(
            tmp_path, proj.id, "2026-03-01T00:00:00.000000Z"
        )
        assert [p.id for p in out] == [p1.id]


# --------------------------------------------------------------------------- #
# reconstruct_state_at + ProjectStateAtTime
# --------------------------------------------------------------------------- #


class TestReconstructStateAt:
    def test_full_state_assembled(self, tmp_path: Path) -> None:
        proj = _saved_project(
            tmp_path, stage="focused", now="2026-01-01T00:00:00.000000Z"
        )
        c = Code.new(
            project_id=proj.id, name="alpha", code_id=_HEX_CODE_A,
            definition="v1",
            now="2026-01-15T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-01-15T00:00:00.000000Z")

        state = reconstruct_state_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        assert isinstance(state, ProjectStateAtTime)
        assert state.project_id == proj.id
        assert state.as_of == "2026-06-01T00:00:00.000000Z"
        assert state.project is not None
        assert state.project.id == proj.id
        assert len(state.codes) == 1
        assert state.codes[0].definition == "v1"
        # No lock log → fallback path → best_effort flagged.
        assert state.best_effort is True
        assert any("codebook_stage" in w for w in state.warnings)
        assert state.codebook_locked is False
        assert state.codebook_stage == "focused"

    def test_include_flags_skip_sections(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(
            project_id=proj.id, type="free", body="x",
            now="2026-01-01T00:00:00.000000Z",
        )
        save_memo(tmp_path, m)
        state = reconstruct_state_at(
            tmp_path,
            proj.id,
            "2026-06-01T00:00:00.000000Z",
            include_memos=False,
        )
        assert state.memos == []

    def test_modified_after_warning(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Create a memo, then "edit" it later by stamping modified_at.
        m = Memo.new(
            project_id=proj.id, type="free", body="x",
            now="2026-01-01T00:00:00.000000Z",
        )
        save_memo(tmp_path, m)
        # Patch on-disk to simulate later edit.
        from scribe.memos import memo_state_path

        path = memo_state_path(tmp_path, proj.id, m.id)
        d = json.loads(path.read_text())
        d["modified_at"] = "2026-09-01T00:00:00.000000Z"
        path.write_text(json.dumps(d))

        state = reconstruct_state_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        # Memo created in window, modified after — should be present
        # AND warned about.
        assert len(state.memos) == 1
        assert any(
            "modified after as_of" in w for w in state.warnings
        )

    def test_to_dict_round_trips_to_json(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = Code.new(
            project_id=proj.id, name="alpha", code_id=_HEX_CODE_A,
            now="2026-01-01T00:00:00.000000Z",
        )
        save_code_with_version(tmp_path, c, now="2026-01-01T00:00:00.000000Z")
        state = reconstruct_state_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        encoded = json.dumps(state.to_dict())
        round_tripped = json.loads(encoded)
        assert round_tripped["project_id"] == proj.id
        assert round_tripped["as_of"] == "2026-06-01T00:00:00.000000Z"
        assert len(round_tripped["codes"]) == 1
        assert round_tripped["codes"][0]["id"] == _HEX_CODE_A
        assert "best_effort" in round_tripped
        assert isinstance(round_tripped["warnings"], list)

    def test_invalid_project_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            reconstruct_state_at(
                tmp_path, "bogus", "2026-06-01T00:00:00.000000Z"
            )

    def test_invalid_as_of_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            reconstruct_state_at(tmp_path, _HEX_PROJECT, "")

    def test_no_best_effort_when_no_optional_data(
        self, tmp_path: Path
    ) -> None:
        # Project with a lock event in the window and no apps/memos/etc.
        proj = _saved_project(tmp_path)
        ev = LockEvent.new(
            project_id=proj.id,
            action="lock",
            reason="finalised",
            prior_stage="focused",
            new_stage="locked",
            now="2026-03-01T00:00:00.000000Z",
        )
        append_lock_event(tmp_path, ev)
        state = reconstruct_state_at(
            tmp_path, proj.id, "2026-06-01T00:00:00.000000Z"
        )
        # Lock came from log → no fallback warning.
        assert all(
            "codebook_stage" not in w for w in state.warnings
        )
        assert state.codebook_stage == "locked"
        assert state.codebook_locked is True


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


class TestEntityKinds:
    def test_kinds_is_closed_set(self) -> None:
        assert ENTITY_KIND_CODE in ENTITY_KINDS
        assert ENTITY_KIND_PROJECT in ENTITY_KINDS
        # Vocabulary stable; spec stays fixed for downstream consumers.
        assert len(ENTITY_KINDS) == len(set(ENTITY_KINDS))
