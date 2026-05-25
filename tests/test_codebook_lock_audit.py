"""Tests for scribe.codebook_lock_audit (F9.5).

Covers the audit-integrated wrappers around F2.4's lock primitives:

* ``lock_codebook_with_audit`` — emits an F9.1 event alongside the
  F2.4 lock-log entry, with stage transitions and lock-event id.
* ``unlock_codebook_with_memo`` — flips the stage, creates a first-
  class methodological :class:`Memo` linked to the project with role
  ``codebook_unlock``, and records an F9.1 ``unlock`` event whose
  ``after`` payload cross-references both sidecars.
* Read helpers — ``find_unlock_memos``, ``latest_unlock_memo``,
  ``find_codebook_lock_events``, ``reconcile_unlock_artefacts``.

Pure-Python; no FastAPI surface yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.codebook_lock import (
    LockedCodebookError,
    is_codebook_locked,
    read_lock_log,
)
from scribe.codebook_lock_audit import (
    UNLOCK_LINK_ROLE,
    LockResult,
    UnlockResult,
    find_codebook_lock_events,
    find_unlock_memos,
    latest_unlock_memo,
    lock_codebook_with_audit,
    reconcile_unlock_artefacts,
    unlock_codebook_with_memo,
)
from scribe.coders import Coder, save_coder
from scribe.event_log import (
    EVENT_ACTION_LOCK,
    EVENT_ACTION_UNLOCK,
    EVENT_ENTITY_CODEBOOK,
    list_events,
    load_event,
)
from scribe.memos import (
    MAX_TITLE_LEN,
    Memo,
    list_memos,
    load_memo,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    load_project,
    save_project,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(
    tmp_path: Path,
    *,
    name: str = "Project",
    stage: str = "focused",
) -> Project:
    p = Project.new(name=name, codebook_stage=stage)
    save_project(tmp_path, p)
    return p


def _saved_coder(tmp_path: Path, project_id: str, *, name: str = "Researcher") -> Coder:
    c = Coder.new(project_id=project_id, name=name)
    save_coder(tmp_path, c)
    return c


# --------------------------------------------------------------------------- #
# lock_codebook_with_audit
# --------------------------------------------------------------------------- #


class TestLockCodebookWithAudit:
    def test_returns_lockresult(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        result = lock_codebook_with_audit(
            tmp_path, p.id, reason="Final coding pass"
        )
        assert isinstance(result, LockResult)
        # Project state flipped to locked.
        assert result.project.codebook_stage == "locked"
        assert is_codebook_locked(tmp_path, p.id)
        # Lock event recorded the prior stage.
        assert result.lock_event.action == "lock"
        assert result.lock_event.prior_stage == "focused"
        assert result.lock_event.new_stage == "locked"
        assert result.lock_event.reason == "Final coding pass"

    def test_appends_f24_lock_log_entry(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        lock_codebook_with_audit(tmp_path, p.id, reason="r")
        log = read_lock_log(tmp_path, p.id)
        assert len(log) == 1
        assert log[0].action == "lock"

    def test_records_f91_event(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="initial")
        result = lock_codebook_with_audit(
            tmp_path, p.id, reason="Done iterating"
        )
        events = list_events(tmp_path, p.id)
        assert len(events) == 1
        ev = events[0]
        assert ev.id == result.event.id
        assert ev.action == EVENT_ACTION_LOCK
        assert ev.entity_type == EVENT_ENTITY_CODEBOOK
        assert ev.before == {"codebook_stage": "initial"}
        assert ev.after == {
            "codebook_stage": "locked",
            "lock_event_id": result.lock_event.id,
        }
        assert ev.notes == "Done iterating"

    def test_event_carries_actor(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        coder = _saved_coder(tmp_path, p.id)
        result = lock_codebook_with_audit(
            tmp_path, p.id, reason="r", actor_coder_id=coder.id
        )
        assert result.event.actor_coder_id == coder.id

    def test_actor_optional(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        result = lock_codebook_with_audit(tmp_path, p.id, reason="r")
        assert result.event.actor_coder_id == ""

    def test_invalid_actor_rejected(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="actor_coder_id"):
            lock_codebook_with_audit(
                tmp_path,
                p.id,
                reason="r",
                actor_coder_id="not-hex",
            )

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            lock_codebook_with_audit(
                tmp_path, "not-hex", reason="r"
            )

    def test_relock_rejected(self, tmp_path: Path) -> None:
        # F2.4 rejects re-lock; the wrapper must surface that.
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(LockedCodebookError, match="already locked"):
            lock_codebook_with_audit(tmp_path, p.id, reason="r")
        # No F9.1 event written on the failed call.
        assert list_events(tmp_path, p.id) == []

    def test_now_propagates(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        ts = "2026-05-26T10:00:00.000000Z"
        result = lock_codebook_with_audit(
            tmp_path, p.id, reason="r", now=ts
        )
        assert result.lock_event.created_at == ts
        assert result.event.created_at == ts
        assert result.project.modified_at == ts


# --------------------------------------------------------------------------- #
# unlock_codebook_with_memo
# --------------------------------------------------------------------------- #


class TestUnlockCodebookWithMemo:
    def test_returns_unlockresult(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="Discovered new boundary case",
            methodological_memo=(
                "Source 12 surfaced a participant describing 'pacing' "
                "in a way our current definition excludes."
            ),
        )
        assert isinstance(result, UnlockResult)
        # Project unlocked, default stage is the prior 'focused'.
        assert result.project.codebook_stage == "focused"
        assert not is_codebook_locked(tmp_path, p.id)
        # Lock event recorded the unlock.
        assert result.lock_event.action == "unlock"
        assert result.lock_event.prior_stage == "locked"
        assert result.lock_event.new_stage == "focused"
        assert "pacing" in result.lock_event.methodological_memo

    def test_creates_methodological_memo(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="Memo body explaining the boundary case.",
        )
        assert isinstance(result.memo, Memo)
        assert result.memo.type == "methodological"
        assert result.memo.body == "Memo body explaining the boundary case."
        # Linked to the project with the codebook_unlock role.
        assert len(result.memo.links) == 1
        link = result.memo.links[0]
        assert link.target_type == "project"
        assert link.target_id == p.id
        assert link.role == UNLOCK_LINK_ROLE
        # Provenance points back at the lock log entry.
        assert (
            result.memo.provenance["codebook_lock_event_id"]
            == result.lock_event.id
        )
        assert result.memo.provenance["codebook_unlock_reason"] == "r"
        assert (
            result.memo.provenance["codebook_unlock_prior_stage"] == "locked"
        )
        assert (
            result.memo.provenance["codebook_unlock_new_stage"] == "focused"
        )
        # Memo is on disk.
        loaded = load_memo(tmp_path, p.id, result.memo.id)
        assert loaded == result.memo

    def test_memo_title_includes_reason(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="Add inclusion criteria for delays",
            methodological_memo="m",
        )
        assert result.memo.title.startswith("Codebook unlock:")
        assert "Add inclusion criteria for delays" in result.memo.title

    def test_long_reason_truncates_provenance(self, tmp_path: Path) -> None:
        # F2.4 allows 2000-char reasons but F5.1 caps provenance values
        # at 1000. The wrapper must truncate to keep the memo valid.
        from scribe.memos import MAX_PROVENANCE_VALUE_LEN
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        long_reason = "x" * 1500
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason=long_reason,
            methodological_memo="m",
        )
        prov_reason = result.memo.provenance["codebook_unlock_reason"]
        assert len(prov_reason) <= MAX_PROVENANCE_VALUE_LEN
        assert prov_reason.endswith("…")
        # The full reason still lives on the lock-log entry.
        assert result.lock_event.reason == long_reason

    def test_long_reason_truncates_title(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        long_reason = "x" * 1500  # well above MAX_TITLE_LEN
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason=long_reason,
            methodological_memo="m",
        )
        # Title respects the global cap.
        assert len(result.memo.title) <= MAX_TITLE_LEN
        # Ends with an ellipsis to signal truncation.
        assert result.memo.title.endswith("…")

    def test_records_f91_unlock_event(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        # Two events expected at the end: lock, then unlock.
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="memo body",
        )
        events = list_events(tmp_path, p.id)
        assert len(events) == 2
        unlock_event = events[-1]
        assert unlock_event.id == result.event.id
        assert unlock_event.action == EVENT_ACTION_UNLOCK
        assert unlock_event.entity_type == EVENT_ENTITY_CODEBOOK
        assert unlock_event.before == {"codebook_stage": "locked"}
        assert unlock_event.after == {
            "codebook_stage": "focused",
            "lock_event_id": result.lock_event.id,
            "memo_id": result.memo.id,
        }
        assert unlock_event.notes == "r"

    def test_author_coder_id_on_memo(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        coder = _saved_coder(tmp_path, p.id)
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
            author_coder_id=coder.id,
        )
        assert result.memo.author_coder_id == coder.id

    def test_actor_separate_from_author(self, tmp_path: Path) -> None:
        # actor_coder_id (event) and author_coder_id (memo) can differ.
        p = _saved_project(tmp_path, stage="focused")
        actor = _saved_coder(tmp_path, p.id, name="Lead")
        author = _saved_coder(tmp_path, p.id, name="Junior")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
            actor_coder_id=actor.id,
            author_coder_id=author.id,
        )
        assert result.event.actor_coder_id == actor.id
        assert result.memo.author_coder_id == author.id

    def test_invalid_actor_rejected(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="actor_coder_id"):
            unlock_codebook_with_memo(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="m",
                actor_coder_id="not-hex",
            )

    def test_invalid_author_rejected(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="author_coder_id"):
            unlock_codebook_with_memo(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="m",
                author_coder_id="not-hex",
            )

    def test_explicit_new_stage(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="initial")
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
            new_stage="axial",
        )
        assert result.project.codebook_stage == "axial"
        assert result.event.after["codebook_stage"] == "axial"

    def test_rejects_when_not_locked(self, tmp_path: Path) -> None:
        # F2.4 rejects an unlock when the codebook isn't locked. The
        # wrapper must propagate the error and write nothing.
        p = _saved_project(tmp_path, stage="focused")
        with pytest.raises(ProjectValidationError, match="not locked"):
            unlock_codebook_with_memo(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="m",
            )
        # No memo, no event.
        assert list_memos(tmp_path, p.id) == []
        assert list_events(tmp_path, p.id) == []

    def test_rejects_empty_methodological_memo(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(
            ProjectValidationError, match="methodological_memo"
        ):
            unlock_codebook_with_memo(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="",
            )

    def test_rejects_empty_reason(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="reason"):
            unlock_codebook_with_memo(
                tmp_path,
                p.id,
                reason="",
                methodological_memo="m",
            )

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            unlock_codebook_with_memo(
                tmp_path,
                "nope",
                reason="r",
                methodological_memo="m",
            )

    def test_now_propagates_to_all_three(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(
            tmp_path, p.id, reason="lock", now="2026-05-26T10:00:00.000000Z"
        )
        ts = "2026-05-26T11:00:00.000000Z"
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
            now=ts,
        )
        assert result.lock_event.created_at == ts
        assert result.memo.created_at == ts
        assert result.event.created_at == ts


# --------------------------------------------------------------------------- #
# Read-side helpers
# --------------------------------------------------------------------------- #


class TestFindUnlockMemos:
    def test_empty_when_no_unlocks(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert find_unlock_memos(tmp_path, p.id) == []

    def test_returns_unlock_memos_only(self, tmp_path: Path) -> None:
        # Two unrelated methodological memos + one unlock memo. Only
        # the unlock-tagged one should come back.
        p = _saved_project(tmp_path, stage="focused")
        # Unrelated methodological memo (manual).
        from scribe.memos import Memo, MemoLink, save_memo
        unrelated = Memo.new(
            project_id=p.id,
            type="methodological",
            body="general note",
            links=[
                MemoLink(target_type="project", target_id=p.id, role="note")
            ],
        )
        save_memo(tmp_path, unrelated)
        # Unlock memo.
        lock_codebook_with_audit(tmp_path, p.id, reason="lock")
        result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="unlock body",
        )
        memos = find_unlock_memos(tmp_path, p.id)
        assert [m.id for m in memos] == [result.memo.id]

    def test_orders_oldest_first(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(
            tmp_path, p.id, reason="l1", now="2026-01-01T00:00:00.000000Z"
        )
        u1 = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r1",
            methodological_memo="m1",
            now="2026-01-02T00:00:00.000000Z",
        )
        lock_codebook_with_audit(
            tmp_path, p.id, reason="l2", now="2026-02-01T00:00:00.000000Z"
        )
        u2 = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r2",
            methodological_memo="m2",
            now="2026-02-02T00:00:00.000000Z",
        )
        memos = find_unlock_memos(tmp_path, p.id)
        assert [m.id for m in memos] == [u1.memo.id, u2.memo.id]

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            find_unlock_memos(tmp_path, "nope")


class TestLatestUnlockMemo:
    def test_none_when_empty(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert latest_unlock_memo(tmp_path, p.id) is None

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook_with_audit(
            tmp_path, p.id, reason="l1", now="2026-01-01T00:00:00.000000Z"
        )
        unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r1",
            methodological_memo="m1",
            now="2026-01-02T00:00:00.000000Z",
        )
        lock_codebook_with_audit(
            tmp_path, p.id, reason="l2", now="2026-02-01T00:00:00.000000Z"
        )
        u2 = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r2",
            methodological_memo="m2",
            now="2026-02-02T00:00:00.000000Z",
        )
        latest = latest_unlock_memo(tmp_path, p.id)
        assert latest is not None
        assert latest.id == u2.memo.id


class TestFindCodebookLockEvents:
    def test_empty_when_no_events(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert find_codebook_lock_events(tmp_path, p.id) == []

    def test_returns_lock_and_unlock_only(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        # Unrelated event (a project create) shouldn't show up.
        from scribe.event_log import (
            EVENT_ACTION_CREATE,
            EVENT_ENTITY_PROJECT,
            record_event,
        )
        record_event(
            tmp_path,
            project_id=p.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_PROJECT,
            entity_id=p.id,
            after={"name": "x"},
        )
        # Lock + unlock pair via F9.5 wrappers.
        lock_codebook_with_audit(tmp_path, p.id, reason="l")
        unlock_codebook_with_memo(
            tmp_path, p.id, reason="r", methodological_memo="m"
        )
        events = find_codebook_lock_events(tmp_path, p.id)
        assert [ev.action for ev in events] == [
            EVENT_ACTION_LOCK,
            EVENT_ACTION_UNLOCK,
        ]
        # All entity_type=codebook.
        for ev in events:
            assert ev.entity_type == EVENT_ENTITY_CODEBOOK

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            find_codebook_lock_events(tmp_path, "nope")


class TestReconcileUnlockArtefacts:
    def test_empty_when_no_lock_log(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert reconcile_unlock_artefacts(tmp_path, p.id) == []

    def test_lock_row_has_event_only(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_result = lock_codebook_with_audit(tmp_path, p.id, reason="r")
        rows = reconcile_unlock_artefacts(tmp_path, p.id)
        assert rows == [
            {
                "lock_event_id": lock_result.lock_event.id,
                "action": "lock",
                "memo_id": "",  # no memo for a lock action
                "event_id": lock_result.event.id,
            }
        ]

    def test_unlock_row_has_memo_and_event(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_result = lock_codebook_with_audit(tmp_path, p.id, reason="l")
        unlock_result = unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
        )
        rows = reconcile_unlock_artefacts(tmp_path, p.id)
        assert len(rows) == 2
        assert rows[0]["lock_event_id"] == lock_result.lock_event.id
        assert rows[0]["action"] == "lock"
        assert rows[0]["memo_id"] == ""
        assert rows[0]["event_id"] == lock_result.event.id
        assert rows[1]["lock_event_id"] == unlock_result.lock_event.id
        assert rows[1]["action"] == "unlock"
        assert rows[1]["memo_id"] == unlock_result.memo.id
        assert rows[1]["event_id"] == unlock_result.event.id

    def test_legacy_lock_log_without_sidecars(
        self, tmp_path: Path
    ) -> None:
        # Simulate an F2.4-only project: lock toggles via the raw
        # primitives (no F9.1 events, no F5.1 memos). reconcile must
        # tolerate this and surface empty memo/event ids.
        from scribe.codebook_lock import lock_codebook, unlock_codebook
        p = _saved_project(tmp_path, stage="focused")
        _, raw_lock = lock_codebook(tmp_path, p.id, reason="l")
        _, raw_unlock = unlock_codebook(
            tmp_path, p.id, reason="r", methodological_memo="m"
        )
        rows = reconcile_unlock_artefacts(tmp_path, p.id)
        assert len(rows) == 2
        assert rows[0]["lock_event_id"] == raw_lock.id
        assert rows[0]["event_id"] == ""
        assert rows[0]["memo_id"] == ""
        assert rows[1]["lock_event_id"] == raw_unlock.id
        assert rows[1]["event_id"] == ""
        assert rows[1]["memo_id"] == ""

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            reconcile_unlock_artefacts(tmp_path, "nope")


# --------------------------------------------------------------------------- #
# Integration — full lock/unlock round-trip surfaces correctly everywhere
# --------------------------------------------------------------------------- #


class TestEndToEndRoundTrip:
    def test_full_lock_unlock_lock_cycle(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        # Lock 1.
        lock_codebook_with_audit(
            tmp_path, p.id, reason="initial lock"
        )
        # Unlock 1.
        unlock_codebook_with_memo(
            tmp_path,
            p.id,
            reason="reopen for boundary case",
            methodological_memo="memo body 1",
        )
        # Lock 2.
        lock_codebook_with_audit(
            tmp_path, p.id, reason="re-freeze"
        )
        # Three F2.4 lock-log lines.
        lock_log = read_lock_log(tmp_path, p.id)
        assert [e.action for e in lock_log] == ["lock", "unlock", "lock"]
        # Three F9.1 events.
        events = find_codebook_lock_events(tmp_path, p.id)
        assert [e.action for e in events] == ["lock", "unlock", "lock"]
        # One unlock memo.
        memos = find_unlock_memos(tmp_path, p.id)
        assert len(memos) == 1
        assert memos[0].body == "memo body 1"
        # The memo is also surfaced by the generic memo list, filtered
        # by methodological type.
        all_methodological = list_memos(
            tmp_path, p.id, type="methodological"
        )
        assert len(all_methodological) == 1
        assert all_methodological[0].id == memos[0].id

    def test_event_can_be_loaded_by_id(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        result = lock_codebook_with_audit(tmp_path, p.id, reason="r")
        loaded = load_event(tmp_path, p.id, result.event.id)
        assert loaded == result.event

    def test_project_modified_at_advances(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        before = load_project(tmp_path, p.id).modified_at
        lock_codebook_with_audit(
            tmp_path, p.id, reason="r", now="2026-05-26T10:00:00.000000Z"
        )
        assert (
            load_project(tmp_path, p.id).modified_at
            == "2026-05-26T10:00:00.000000Z"
        )
        assert (
            load_project(tmp_path, p.id).modified_at != before
        )
