"""Tests for scribe.codebook_lock (F2.4).

Pure-Python coverage of the locked-codebook stage marker:

* the LockEvent data model: validation, round-trip, vocabularies,
  the unlock-requires-memo invariant;
* state accessors (``is_codebook_locked`` / ``assert_codebook_unlocked``);
* lock / unlock toggles (``lock_codebook`` / ``unlock_codebook``):
  prior-stage capture, default new_stage resolution, idempotency
  rejection, methodological-memo requirement on unlock;
* guarded helpers: ``guarded_save_code`` and
  ``guarded_save_code_with_version`` raise when locked, succeed when not.

Endpoint-level tests will live in test_server.py once F2.4 grows an
HTTP surface; for now the model + persistence are the public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.codebook_lock import (
    LOCK_ACTIONS,
    LOCK_EVENT_ID_RE,
    LOCKED_STAGE,
    MAX_METHODOLOGICAL_MEMO_LEN,
    MAX_REASON_LEN,
    LockedCodebookError,
    LockEvent,
    append_lock_event,
    assert_codebook_unlocked,
    count_lock_events,
    guarded_save_code,
    guarded_save_code_with_version,
    is_codebook_locked,
    latest_lock_event,
    lock_codebook,
    lock_log_path,
    new_lock_event_id,
    read_lock_log,
    unlock_codebook,
)
from scribe.codes import (
    Code,
    load_code,
    save_code,
)
from scribe.code_versions import (
    count_code_versions,
    read_code_versions,
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
    stage: str = "initial",
) -> Project:
    p = Project.new(name=name, codebook_stage=stage)
    save_project(tmp_path, p)
    return p


def _make_code(
    tmp_path: Path,
    project_id: str,
    *,
    name: str = "Pacing",
    code_id: str | None = None,
) -> Code:
    c = Code.new(project_id=project_id, name=name, code_id=code_id)
    save_code(tmp_path, c)
    return c


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewLockEventId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert LOCK_EVENT_ID_RE.match(new_lock_event_id())

    def test_unique(self) -> None:
        ids = {new_lock_event_id() for _ in range(20)}
        assert len(ids) == 20


# --------------------------------------------------------------------------- #
# LockEvent — construction + validation
# --------------------------------------------------------------------------- #


class TestLockEventNew:
    def test_lock_action_with_reason(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id,
            action="lock",
            reason="Final coding pass",
            prior_stage="focused",
            new_stage="locked",
        )
        assert e.action == "lock"
        assert e.reason == "Final coding pass"
        assert e.prior_stage == "focused"
        assert e.new_stage == "locked"
        assert e.methodological_memo == ""
        assert LOCK_EVENT_ID_RE.match(e.id)
        assert e.created_at  # stamped automatically

    def test_unlock_requires_methodological_memo(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="methodological_memo"):
            LockEvent.new(
                project_id=p.id,
                action="unlock",
                reason="Discovered new boundary case",
                methodological_memo="",  # blank: rejected
                prior_stage="locked",
                new_stage="theoretical",
            )

    def test_unlock_with_memo(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id,
            action="unlock",
            reason="Discovered new boundary case in source 12",
            methodological_memo=(
                "Source 12 surfaced a participant describing 'pacing' "
                "in a way our current definition excludes. Reopening "
                "to refine the inclusion criteria."
            ),
            prior_stage="locked",
            new_stage="theoretical",
        )
        assert e.action == "unlock"
        assert "pacing" in e.methodological_memo

    def test_lock_does_not_require_memo(self, tmp_path: Path) -> None:
        # Locking is the safe direction — no memo needed.
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id,
            action="lock",
            reason="Done iterating",
        )
        assert e.methodological_memo == ""

    def test_reason_required_for_lock_too(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="reason"):
            LockEvent.new(
                project_id=p.id,
                action="lock",
                reason="   ",
            )

    def test_reason_length_capped(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match=str(MAX_REASON_LEN)):
            LockEvent.new(
                project_id=p.id,
                action="lock",
                reason="x" * (MAX_REASON_LEN + 1),
            )

    def test_memo_length_capped(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(
            ProjectValidationError, match=str(MAX_METHODOLOGICAL_MEMO_LEN)
        ):
            LockEvent.new(
                project_id=p.id,
                action="unlock",
                reason="ok",
                methodological_memo="x" * (MAX_METHODOLOGICAL_MEMO_LEN + 1),
                prior_stage="locked",
                new_stage="theoretical",
            )

    def test_action_must_be_known(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="action"):
            LockEvent.new(
                project_id=p.id,
                action="freeze",  # not in LOCK_ACTIONS
                reason="nope",
            )

    def test_stage_values_validated(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="prior_stage"):
            LockEvent.new(
                project_id=p.id,
                action="lock",
                reason="ok",
                prior_stage="banana",
            )
        with pytest.raises(ProjectValidationError, match="new_stage"):
            LockEvent.new(
                project_id=p.id,
                action="lock",
                reason="ok",
                new_stage="cucumber",
            )

    def test_invalid_project_id(self) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            LockEvent.new(
                project_id="not-hex",
                action="lock",
                reason="ok",
            )

    def test_reason_is_trimmed(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id,
            action="lock",
            reason="   Final pass.  ",
        )
        assert e.reason == "Final pass."


class TestLockEventRoundTrip:
    def test_to_dict_from_dict(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id,
            action="unlock",
            reason="boundary case",
            methodological_memo="memo body",
            prior_stage="locked",
            new_stage="axial",
        )
        d = e.to_dict()
        assert d["action"] == "unlock"
        e2 = LockEvent.from_dict(d)
        assert e2 == e

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError, match="must be an object"):
            LockEvent.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_requires_keys(self) -> None:
        with pytest.raises(ProjectValidationError, match="missing required key"):
            LockEvent.from_dict({"id": "abc"})

    def test_lock_actions_const_only_two(self) -> None:
        assert LOCK_ACTIONS == ("lock", "unlock")


# --------------------------------------------------------------------------- #
# Lock log on disk
# --------------------------------------------------------------------------- #


class TestLockLogPath:
    def test_lives_under_project(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        path = lock_log_path(tmp_path, p.id)
        assert path.name == "codebook_lock_log.jsonl"
        assert path.parent.name == p.id

    def test_invalid_project_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            lock_log_path(tmp_path, "not-hex")


class TestAppendLockEvent:
    def test_appends_one_line(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(
            project_id=p.id, action="lock", reason="done"
        )
        target = append_lock_event(tmp_path, e)
        assert target.exists()
        text = target.read_text(encoding="utf-8")
        # one line, ending in newline
        assert text.endswith("\n")
        assert text.count("\n") == 1
        payload = json.loads(text.rstrip())
        assert payload["id"] == e.id
        assert payload["action"] == "lock"

    def test_appends_multiple_lines(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e1 = LockEvent.new(project_id=p.id, action="lock", reason="r1")
        e2 = LockEvent.new(
            project_id=p.id,
            action="unlock",
            reason="r2",
            methodological_memo="memo",
            prior_stage="locked",
            new_stage="focused",
        )
        append_lock_event(tmp_path, e1)
        append_lock_event(tmp_path, e2)
        text = lock_log_path(tmp_path, p.id).read_text(encoding="utf-8")
        assert text.count("\n") == 2

    def test_requires_project_to_exist(self, tmp_path: Path) -> None:
        # No project.json saved.
        e = LockEvent.new(
            project_id="abcdef012345", action="lock", reason="r"
        )
        with pytest.raises(FileNotFoundError):
            append_lock_event(tmp_path, e)


class TestReadLockLog:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert read_lock_log(tmp_path, p.id) == []

    def test_round_trips(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e1 = LockEvent.new(project_id=p.id, action="lock", reason="r1")
        e2 = LockEvent.new(
            project_id=p.id,
            action="unlock",
            reason="r2",
            methodological_memo="memo",
            prior_stage="locked",
            new_stage="theoretical",
        )
        append_lock_event(tmp_path, e1)
        append_lock_event(tmp_path, e2)
        events = read_lock_log(tmp_path, p.id)
        assert [ev.action for ev in events] == ["lock", "unlock"]
        assert events[0] == e1
        assert events[1] == e2

    def test_skips_corrupt_lines(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        e = LockEvent.new(project_id=p.id, action="lock", reason="r")
        append_lock_event(tmp_path, e)
        # Corrupt a line: append garbage and a malformed JSON.
        with lock_log_path(tmp_path, p.id).open("a", encoding="utf-8") as f:
            f.write("not-json\n")
            f.write("{}\n")  # JSON but missing fields → from_dict fails
        events = read_lock_log(tmp_path, p.id)
        assert len(events) == 1
        assert events[0].id == e.id

    def test_invalid_project_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            read_lock_log(tmp_path, "not-hex")


class TestLatestLockEvent:
    def test_none_when_empty(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert latest_lock_event(tmp_path, p.id) is None

    def test_returns_last(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        a = LockEvent.new(project_id=p.id, action="lock", reason="a")
        b = LockEvent.new(
            project_id=p.id,
            action="unlock",
            reason="b",
            methodological_memo="memo",
            prior_stage="locked",
            new_stage="theoretical",
        )
        append_lock_event(tmp_path, a)
        append_lock_event(tmp_path, b)
        latest = latest_lock_event(tmp_path, p.id)
        assert latest is not None
        assert latest.id == b.id


class TestCountLockEvents:
    def test_zero_when_empty(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert count_lock_events(tmp_path, p.id) == 0

    def test_counts_valid_only(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        for i in range(3):
            append_lock_event(
                tmp_path,
                LockEvent.new(
                    project_id=p.id, action="lock", reason=f"r{i}"
                ),
            )
        assert count_lock_events(tmp_path, p.id) == 3


# --------------------------------------------------------------------------- #
# State accessors
# --------------------------------------------------------------------------- #


class TestIsCodebookLocked:
    def test_false_initially(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert is_codebook_locked(tmp_path, p.id) is False

    def test_true_when_stage_is_locked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        assert is_codebook_locked(tmp_path, p.id) is True

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            is_codebook_locked(tmp_path, "abcdef012345")


class TestAssertCodebookUnlocked:
    def test_no_raise_when_unlocked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        # Should be a no-op.
        assert_codebook_unlocked(tmp_path, p.id)

    def test_raises_when_locked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(LockedCodebookError, match="locked"):
            assert_codebook_unlocked(tmp_path, p.id)

    def test_locked_codebook_error_is_validation_error(
        self, tmp_path: Path
    ) -> None:
        # LockedCodebookError must inherit from ProjectValidationError so
        # the existing HTTP error mapping handles it without changes.
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError):
            assert_codebook_unlocked(tmp_path, p.id)


# --------------------------------------------------------------------------- #
# Lock / unlock toggles
# --------------------------------------------------------------------------- #


class TestLockCodebook:
    def test_sets_stage_and_appends_event(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        project, event = lock_codebook(
            tmp_path, p.id, reason="Ready for ICR"
        )
        # Project now reads as locked.
        assert project.codebook_stage == "locked"
        # On-disk project mirrors the change.
        reloaded = load_project(tmp_path, p.id)
        assert reloaded.codebook_stage == "locked"
        # Event recorded with prior stage.
        assert event.action == "lock"
        assert event.prior_stage == "focused"
        assert event.new_stage == "locked"
        assert event.reason == "Ready for ICR"
        log = read_lock_log(tmp_path, p.id)
        assert len(log) == 1
        assert log[0] == event

    def test_modified_at_advances(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        before = load_project(tmp_path, p.id).modified_at
        project, _ = lock_codebook(
            tmp_path, p.id, reason="done", now="2026-05-26T12:00:00.000000Z"
        )
        assert project.modified_at == "2026-05-26T12:00:00.000000Z"
        assert project.modified_at != before

    def test_rejects_empty_reason(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError, match="reason"):
            lock_codebook(tmp_path, p.id, reason="")
        with pytest.raises(ProjectValidationError, match="reason"):
            lock_codebook(tmp_path, p.id, reason="   ")

    def test_rejects_relock(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(LockedCodebookError, match="already locked"):
            lock_codebook(tmp_path, p.id, reason="?")

    def test_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            lock_codebook(tmp_path, "nope", reason="r")

    def test_missing_project(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            lock_codebook(tmp_path, "abcdef012345", reason="r")


class TestUnlockCodebook:
    def test_unlocks_to_default_stage_after_lock(
        self, tmp_path: Path
    ) -> None:
        # Lock from 'focused' → unlock should default back to 'focused'.
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook(tmp_path, p.id, reason="initial lock")
        project, event = unlock_codebook(
            tmp_path,
            p.id,
            reason="Discovered new code",
            methodological_memo="Memo body explaining the boundary case",
        )
        assert project.codebook_stage == "focused"
        assert event.action == "unlock"
        assert event.prior_stage == "locked"
        assert event.new_stage == "focused"
        assert event.methodological_memo.startswith("Memo body")

    def test_default_stage_falls_back_to_theoretical(
        self, tmp_path: Path
    ) -> None:
        # No prior lock event: default to 'theoretical'. We seed the
        # project as locked directly (e.g. an imported project) and try
        # to unlock without any history.
        p = _saved_project(tmp_path, stage="locked")
        project, event = unlock_codebook(
            tmp_path,
            p.id,
            reason="opening for revision",
            methodological_memo="memo",
        )
        assert project.codebook_stage == "theoretical"
        assert event.new_stage == "theoretical"

    def test_explicit_new_stage(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="initial")
        lock_codebook(tmp_path, p.id, reason="lock")
        project, _ = unlock_codebook(
            tmp_path,
            p.id,
            reason="r",
            methodological_memo="m",
            new_stage="axial",
        )
        assert project.codebook_stage == "axial"

    def test_rejects_locked_as_new_stage(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="locked"):
            unlock_codebook(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="m",
                new_stage="locked",
            )

    def test_rejects_unknown_new_stage(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="must be one of"):
            unlock_codebook(
                tmp_path,
                p.id,
                reason="r",
                methodological_memo="m",
                new_stage="banana",
            )

    def test_rejects_when_not_locked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="initial")
        with pytest.raises(ProjectValidationError, match="not locked"):
            unlock_codebook(
                tmp_path, p.id, reason="r", methodological_memo="m"
            )

    def test_rejects_empty_reason(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(ProjectValidationError, match="reason"):
            unlock_codebook(
                tmp_path, p.id, reason="", methodological_memo="m"
            )

    def test_rejects_empty_methodological_memo(
        self, tmp_path: Path
    ) -> None:
        p = _saved_project(tmp_path, stage="locked")
        with pytest.raises(
            ProjectValidationError, match="methodological_memo"
        ):
            unlock_codebook(
                tmp_path, p.id, reason="r", methodological_memo=""
            )
        with pytest.raises(
            ProjectValidationError, match="methodological_memo"
        ):
            unlock_codebook(
                tmp_path, p.id, reason="r", methodological_memo="   "
            )

    def test_appends_to_lock_log(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook(tmp_path, p.id, reason="lock-r")
        unlock_codebook(
            tmp_path, p.id, reason="unlock-r", methodological_memo="m"
        )
        log = read_lock_log(tmp_path, p.id)
        assert [e.action for e in log] == ["lock", "unlock"]


class TestLockUnlockRoundTrip:
    def test_lock_then_unlock_then_relock(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path, stage="focused")
        lock_codebook(tmp_path, p.id, reason="r1")
        assert is_codebook_locked(tmp_path, p.id)
        unlock_codebook(
            tmp_path, p.id, reason="r2", methodological_memo="m1"
        )
        assert not is_codebook_locked(tmp_path, p.id)
        # Re-lock works.
        lock_codebook(tmp_path, p.id, reason="r3")
        assert is_codebook_locked(tmp_path, p.id)
        log = read_lock_log(tmp_path, p.id)
        assert [e.action for e in log] == ["lock", "unlock", "lock"]
        # The third event should record the prior stage we unlocked
        # back to ('focused', the default).
        assert log[-1].prior_stage == "focused"


# --------------------------------------------------------------------------- #
# Guarded helpers
# --------------------------------------------------------------------------- #


class TestGuardedSaveCode:
    def test_passes_when_unlocked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        c = Code.new(project_id=p.id, name="Pacing")
        path = guarded_save_code(tmp_path, c)
        assert path.exists()
        assert load_code(tmp_path, p.id, c.id).name == "Pacing"

    def test_blocks_when_locked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        # Lock the codebook.
        lock_codebook(tmp_path, p.id, reason="ICR start")
        # Now any guarded save is rejected.
        c = Code.new(project_id=p.id, name="Pacing")
        with pytest.raises(LockedCodebookError, match="locked"):
            guarded_save_code(tmp_path, c)
        # The unguarded save_code still works (importers / migrations
        # bypass the lock by design).
        save_code(tmp_path, c)
        assert load_code(tmp_path, p.id, c.id).name == "Pacing"


class TestGuardedSaveCodeWithVersion:
    def test_passes_when_unlocked(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        c = Code.new(project_id=p.id, name="Pacing")
        target, version = guarded_save_code_with_version(
            tmp_path, c, change_note="initial"
        )
        assert target.exists()
        assert version is not None
        assert version.version == 1
        assert count_code_versions(tmp_path, p.id, c.id) == 1

    def test_blocks_when_locked(self, tmp_path: Path) -> None:
        # Seed: save a code with v1, *then* lock, *then* try to edit it.
        p = _saved_project(tmp_path)
        c = _make_code(tmp_path, p.id)
        # First version isn't recorded yet because we used save_code; do
        # it explicitly through the un-guarded helper to seed v1.
        from scribe.code_versions import record_code_version
        record_code_version(tmp_path, c, change_note="v1")
        lock_codebook(tmp_path, p.id, reason="freeze")
        c.apply_update({"definition": "an updated def"})
        with pytest.raises(LockedCodebookError):
            guarded_save_code_with_version(
                tmp_path, c, change_note="should not land"
            )
        # Version log unchanged.
        assert count_code_versions(tmp_path, p.id, c.id) == 1
        # On-disk code body unchanged too.
        assert load_code(tmp_path, p.id, c.id).definition == ""


# --------------------------------------------------------------------------- #
# Project file format alignment
# --------------------------------------------------------------------------- #


class TestProjectStageVocabularyAlignment:
    def test_locked_is_a_valid_stage(self) -> None:
        # F2.4 leans on the existing F1.1 vocabulary; ensure the constant
        # we depend on still includes 'locked'.
        from scribe.projects import CODEBOOK_STAGES
        assert LOCKED_STAGE in CODEBOOK_STAGES
