"""Tests for scribe.participant_sources (F3.3).

Exercises the inverse navigation, set-style focus-group helpers,
single-edge mutation helpers, and orphan detection in pure Python.
The module itself doesn't introduce a new on-disk format — it just
walks participant + source files — so the tests live close to the
F1.3 / F1.2 patterns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.participants import (
    Participant,
    delete_participant,
    list_participants,
    load_participant,
    save_participant,
)
from scribe.sources import (
    Source,
    delete_source,
    save_source,
)
from scribe.participant_sources import (
    OrphanLink,
    ParticipantSourceChange,
    find_orphan_links,
    link_participant_to_source,
    list_participants_for_source,
    list_sources_for_participant,
    participant_source_map,
    set_participants_for_source,
    unlink_participant_from_source,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _save_source(
    tmp_path: Path,
    project_id: str,
    *,
    name: str,
    sid: str | None = None,
    now: str | None = None,
) -> Source:
    s = Source.new(
        project_id=project_id, name=name, source_id=sid, now=now
    )
    save_source(tmp_path, s)
    return s


def _save_participant(
    tmp_path: Path,
    project_id: str,
    *,
    name: str,
    pid: str | None = None,
    source_ids: list[str] | None = None,
    now: str | None = None,
) -> Participant:
    p = Participant.new(
        project_id=project_id,
        name=name,
        participant_id=pid,
        source_ids=source_ids,
        now=now,
    )
    save_participant(tmp_path, p)
    return p


# --------------------------------------------------------------------------- #
# list_participants_for_source
# --------------------------------------------------------------------------- #


class TestListParticipantsForSource:
    def test_empty_when_nothing_linked(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Interview")
        # A participant with no link should not appear.
        _save_participant(tmp_path, proj.id, name="P01")
        assert list_participants_for_source(tmp_path, proj.id, s.id) == []

    def test_returns_only_linked(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Focus group")
        a = _save_participant(
            tmp_path,
            proj.id,
            name="A",
            source_ids=[s.id],
            now="2024-01-01T00:00:00.000000Z",
        )
        b = _save_participant(
            tmp_path,
            proj.id,
            name="B",
            source_ids=[s.id],
            now="2024-02-01T00:00:00.000000Z",
        )
        # Unrelated participant, not linked to s.
        _save_participant(
            tmp_path,
            proj.id,
            name="C",
            now="2024-03-01T00:00:00.000000Z",
        )
        ids = [p.id for p in list_participants_for_source(tmp_path, proj.id, s.id)]
        assert ids == [a.id, b.id]  # ordered by created_at

    def test_validates_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_participants_for_source(tmp_path, proj.id, "BAD")
        with pytest.raises(ProjectValidationError):
            list_participants_for_source(tmp_path, proj.id, "../escape")

    def test_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            list_participants_for_source(tmp_path, "UPPERCASE123", "0123456789ab")

    def test_no_project_dir(self, tmp_path: Path) -> None:
        # Fresh tmp_path with no project on disk: empty result, not an error.
        assert (
            list_participants_for_source(tmp_path, "aaaaaaaaaaaa", "0123456789ab")
            == []
        )

    def test_finds_participant_in_focus_group(self, tmp_path: Path) -> None:
        # One source, three participants: classic focus-group pattern.
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        names = []
        for i, t in enumerate(["P1", "P2", "P3"]):
            _save_participant(
                tmp_path,
                proj.id,
                name=t,
                source_ids=[fg.id],
                now=f"2024-0{i+1}-01T00:00:00.000000Z",
            )
            names.append(t)
        got = list_participants_for_source(tmp_path, proj.id, fg.id)
        assert [p.name for p in got] == names


# --------------------------------------------------------------------------- #
# list_sources_for_participant
# --------------------------------------------------------------------------- #


class TestListSourcesForParticipant:
    def test_empty_when_no_links(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = _save_participant(tmp_path, proj.id, name="P01")
        assert list_sources_for_participant(tmp_path, proj.id, p.id) == []

    def test_resolves_ids_to_source_objects(self, tmp_path: Path) -> None:
        # One participant with two interviews — the longitudinal pattern.
        proj = _saved_project(tmp_path)
        s1 = _save_source(
            tmp_path, proj.id, name="Wave 1", now="2024-01-01T00:00:00.000000Z"
        )
        s2 = _save_source(
            tmp_path, proj.id, name="Wave 2", now="2024-02-01T00:00:00.000000Z"
        )
        p = _save_participant(
            tmp_path, proj.id, name="P01", source_ids=[s1.id, s2.id]
        )
        out = list_sources_for_participant(tmp_path, proj.id, p.id)
        assert [s.id for s in out] == [s1.id, s2.id]
        assert all(isinstance(s, Source) for s in out)

    def test_preserves_participant_link_order(self, tmp_path: Path) -> None:
        # Even when the on-disk source order (created_at) differs from the
        # participant's preferred order, the result follows the participant's
        # source_ids order — researchers may want a specific reading order.
        proj = _saved_project(tmp_path)
        first = _save_source(
            tmp_path, proj.id, name="A", now="2024-01-01T00:00:00.000000Z"
        )
        second = _save_source(
            tmp_path, proj.id, name="B", now="2024-02-01T00:00:00.000000Z"
        )
        p = _save_participant(
            tmp_path, proj.id, name="P01", source_ids=[second.id, first.id]
        )
        out = list_sources_for_participant(tmp_path, proj.id, p.id)
        assert [s.id for s in out] == [second.id, first.id]

    def test_skips_dangling_references(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Real")
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P01",
            source_ids=[s.id, "deadbeefcafe"],  # second one doesn't exist
        )
        out = list_sources_for_participant(tmp_path, proj.id, p.id)
        assert [x.id for x in out] == [s.id]

    def test_validates_participant_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_sources_for_participant(tmp_path, proj.id, "BAD")

    def test_missing_participant_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            list_sources_for_participant(tmp_path, proj.id, "deadbeefcafe")


# --------------------------------------------------------------------------- #
# participant_source_map
# --------------------------------------------------------------------------- #


class TestParticipantSourceMap:
    def test_empty_project(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert participant_source_map(tmp_path, proj.id) == {}

    def test_sources_with_no_participants_appear_as_empty_lists(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Lonely")
        m = participant_source_map(tmp_path, proj.id)
        assert m == {s.id: []}

    def test_full_mapping(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = _save_source(
            tmp_path, proj.id, name="A", now="2024-01-01T00:00:00.000000Z"
        )
        s2 = _save_source(
            tmp_path, proj.id, name="B", now="2024-02-01T00:00:00.000000Z"
        )
        a = _save_participant(
            tmp_path,
            proj.id,
            name="P-a",
            source_ids=[s1.id, s2.id],
            now="2024-03-01T00:00:00.000000Z",
        )
        b = _save_participant(
            tmp_path,
            proj.id,
            name="P-b",
            source_ids=[s2.id],
            now="2024-04-01T00:00:00.000000Z",
        )
        m = participant_source_map(tmp_path, proj.id)
        assert m == {
            s1.id: [a.id],
            s2.id: [a.id, b.id],
        }

    def test_orphan_source_id_visible(self, tmp_path: Path) -> None:
        # A reference to a non-existent source still shows up so the
        # caller can investigate — but only because some participant
        # actually references it.
        proj = _saved_project(tmp_path)
        a = _save_participant(
            tmp_path,
            proj.id,
            name="P-a",
            source_ids=["deadbeefcafe"],
        )
        m = participant_source_map(tmp_path, proj.id)
        assert m == {"deadbeefcafe": [a.id]}

    def test_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            participant_source_map(tmp_path, "UPPERCASE123")


# --------------------------------------------------------------------------- #
# find_orphan_links
# --------------------------------------------------------------------------- #


class TestFindOrphanLinks:
    def test_no_orphans(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        _save_participant(tmp_path, proj.id, name="P", source_ids=[s.id])
        assert find_orphan_links(tmp_path, proj.id) == []

    def test_reports_missing_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P",
            source_ids=[s.id, "deadbeefcafe"],
        )
        out = find_orphan_links(tmp_path, proj.id)
        assert out == [OrphanLink(participant_id=p.id, source_id="deadbeefcafe")]

    def test_after_source_deletion(self, tmp_path: Path) -> None:
        # Real-world: a researcher excludes an interview after consent
        # was withdrawn. The participant's source_ids still references it.
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Withdrawn")
        p = _save_participant(
            tmp_path, proj.id, name="P", source_ids=[s.id]
        )
        delete_source(tmp_path, proj.id, s.id)
        out = find_orphan_links(tmp_path, proj.id)
        assert out == [OrphanLink(participant_id=p.id, source_id=s.id)]

    def test_stable_ordering(self, tmp_path: Path) -> None:
        # Outer order = participant created_at; inner order = participant's
        # own list order.
        proj = _saved_project(tmp_path)
        a = _save_participant(
            tmp_path,
            proj.id,
            name="A",
            source_ids=["aaaaaaaaaaaa", "bbbbbbbbbbbb"],
            now="2024-01-01T00:00:00.000000Z",
        )
        b = _save_participant(
            tmp_path,
            proj.id,
            name="B",
            source_ids=["cccccccccccc"],
            now="2024-02-01T00:00:00.000000Z",
        )
        out = find_orphan_links(tmp_path, proj.id)
        assert out == [
            OrphanLink(a.id, "aaaaaaaaaaaa"),
            OrphanLink(a.id, "bbbbbbbbbbbb"),
            OrphanLink(b.id, "cccccccccccc"),
        ]


# --------------------------------------------------------------------------- #
# link_participant_to_source / unlink_participant_from_source
# --------------------------------------------------------------------------- #


class TestSingleEdgeMutation:
    def test_link_adds_and_persists(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P",
            now="2024-01-01T00:00:00.000000Z",
        )
        added = link_participant_to_source(
            tmp_path,
            proj.id,
            p.id,
            s.id,
            now="2024-06-01T00:00:00.000000Z",
        )
        assert added is True
        loaded = load_participant(tmp_path, proj.id, p.id)
        assert loaded.source_ids == [s.id]
        assert loaded.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_link_idempotent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P",
            source_ids=[s.id],
            now="2024-01-01T00:00:00.000000Z",
        )
        added = link_participant_to_source(
            tmp_path,
            proj.id,
            p.id,
            s.id,
            now="2024-09-09T00:00:00.000000Z",
        )
        assert added is False
        loaded = load_participant(tmp_path, proj.id, p.id)
        assert loaded.modified_at == "2024-01-01T00:00:00.000000Z"  # not bumped

    def test_link_rejects_unknown_source_by_default(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        p = _save_participant(tmp_path, proj.id, name="P")
        with pytest.raises(ProjectValidationError):
            link_participant_to_source(
                tmp_path, proj.id, p.id, "deadbeefcafe"
            )

    def test_link_allows_forward_reference_when_opted_in(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        p = _save_participant(tmp_path, proj.id, name="P")
        added = link_participant_to_source(
            tmp_path,
            proj.id,
            p.id,
            "deadbeefcafe",
            require_source_exists=False,
        )
        assert added is True
        loaded = load_participant(tmp_path, proj.id, p.id)
        assert loaded.source_ids == ["deadbeefcafe"]

    def test_link_validates_ids(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        with pytest.raises(ProjectValidationError):
            link_participant_to_source(tmp_path, proj.id, "BAD", s.id)
        with pytest.raises(ProjectValidationError):
            link_participant_to_source(
                tmp_path, proj.id, "deadbeefcafe", "BAD"
            )

    def test_link_missing_participant_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        with pytest.raises(FileNotFoundError):
            link_participant_to_source(
                tmp_path, proj.id, "deadbeefcafe", s.id
            )

    def test_unlink_removes_and_persists(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="A")
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P",
            source_ids=[s.id],
            now="2024-01-01T00:00:00.000000Z",
        )
        removed = unlink_participant_from_source(
            tmp_path,
            proj.id,
            p.id,
            s.id,
            now="2024-06-01T00:00:00.000000Z",
        )
        assert removed is True
        loaded = load_participant(tmp_path, proj.id, p.id)
        assert loaded.source_ids == []
        assert loaded.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_unlink_no_op(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = _save_participant(
            tmp_path,
            proj.id,
            name="P",
            now="2024-01-01T00:00:00.000000Z",
        )
        removed = unlink_participant_from_source(
            tmp_path,
            proj.id,
            p.id,
            "deadbeefcafe",
            now="2099-01-01T00:00:00.000000Z",
        )
        assert removed is False
        loaded = load_participant(tmp_path, proj.id, p.id)
        # No clock advance on a no-op.
        assert loaded.modified_at == "2024-01-01T00:00:00.000000Z"

    def test_unlink_does_not_require_source_to_exist(self, tmp_path: Path) -> None:
        # The cleanup-after-deletion path: the source has been removed,
        # but participants still reference it. We must be able to clean
        # those references without recreating the source.
        proj = _saved_project(tmp_path)
        s = _save_source(tmp_path, proj.id, name="Doomed")
        p = _save_participant(
            tmp_path, proj.id, name="P", source_ids=[s.id]
        )
        delete_source(tmp_path, proj.id, s.id)
        removed = unlink_participant_from_source(
            tmp_path, proj.id, p.id, s.id
        )
        assert removed is True


# --------------------------------------------------------------------------- #
# set_participants_for_source — focus-group editor pattern
# --------------------------------------------------------------------------- #


class TestSetParticipantsForSource:
    def test_initial_assignment(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(tmp_path, proj.id, name="A")
        b = _save_participant(tmp_path, proj.id, name="B")
        c = _save_participant(tmp_path, proj.id, name="C")

        change = set_participants_for_source(
            tmp_path,
            proj.id,
            fg.id,
            [a.id, b.id, c.id],
            now="2024-06-01T00:00:00.000000Z",
        )
        assert isinstance(change, ParticipantSourceChange)
        assert change.source_id == fg.id
        assert sorted(change.added) == sorted([a.id, b.id, c.id])
        assert change.removed == []
        assert change.unchanged == []
        assert change.changed is True

        # Inverse query reflects the new state.
        got = [p.id for p in list_participants_for_source(tmp_path, proj.id, fg.id)]
        assert sorted(got) == sorted([a.id, b.id, c.id])

    def test_diff_only_writes_changed_participants(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        # Existing roster: A, B linked.
        a = _save_participant(
            tmp_path,
            proj.id,
            name="A",
            source_ids=[fg.id],
            now="2024-01-01T00:00:00.000000Z",
        )
        b = _save_participant(
            tmp_path,
            proj.id,
            name="B",
            source_ids=[fg.id],
            now="2024-01-01T00:00:00.000000Z",
        )
        c = _save_participant(
            tmp_path,
            proj.id,
            name="C",
            now="2024-01-01T00:00:00.000000Z",
        )

        # New roster: B stays, A leaves, C joins.
        change = set_participants_for_source(
            tmp_path,
            proj.id,
            fg.id,
            [b.id, c.id],
            now="2024-06-01T00:00:00.000000Z",
        )
        assert change.added == [c.id]
        assert change.removed == [a.id]
        assert change.unchanged == [b.id]

        # A's link gone, A.modified_at advanced.
        loaded_a = load_participant(tmp_path, proj.id, a.id)
        assert loaded_a.source_ids == []
        assert loaded_a.modified_at == "2024-06-01T00:00:00.000000Z"

        # B unchanged on disk (no clock advance).
        loaded_b = load_participant(tmp_path, proj.id, b.id)
        assert loaded_b.source_ids == [fg.id]
        assert loaded_b.modified_at == "2024-01-01T00:00:00.000000Z"

        # C linked, C.modified_at advanced.
        loaded_c = load_participant(tmp_path, proj.id, c.id)
        assert loaded_c.source_ids == [fg.id]
        assert loaded_c.modified_at == "2024-06-01T00:00:00.000000Z"

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(tmp_path, proj.id, name="A")
        b = _save_participant(tmp_path, proj.id, name="B")

        first = set_participants_for_source(
            tmp_path,
            proj.id,
            fg.id,
            [a.id, b.id],
            now="2024-06-01T00:00:00.000000Z",
        )
        assert first.changed is True

        # Mid-test snapshot of modified_at after first run.
        ts_a = load_participant(tmp_path, proj.id, a.id).modified_at
        ts_b = load_participant(tmp_path, proj.id, b.id).modified_at

        second = set_participants_for_source(
            tmp_path,
            proj.id,
            fg.id,
            [a.id, b.id],
            now="2099-12-31T00:00:00.000000Z",
        )
        assert second.added == []
        assert second.removed == []
        assert sorted(second.unchanged) == sorted([a.id, b.id])
        assert second.changed is False
        # No clock bump on no-op.
        assert load_participant(tmp_path, proj.id, a.id).modified_at == ts_a
        assert load_participant(tmp_path, proj.id, b.id).modified_at == ts_b

    def test_empty_clears_source_from_all(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(
            tmp_path, proj.id, name="A", source_ids=[fg.id]
        )
        b = _save_participant(
            tmp_path, proj.id, name="B", source_ids=[fg.id]
        )
        change = set_participants_for_source(tmp_path, proj.id, fg.id, [])
        assert sorted(change.removed) == sorted([a.id, b.id])
        assert change.added == []
        assert (
            list_participants_for_source(tmp_path, proj.id, fg.id) == []
        )

    def test_dedupes_input_ids(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(tmp_path, proj.id, name="A")
        change = set_participants_for_source(
            tmp_path, proj.id, fg.id, [a.id, a.id]
        )
        assert change.added == [a.id]
        assert load_participant(tmp_path, proj.id, a.id).source_ids == [fg.id]

    def test_unknown_participant_id_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(tmp_path, proj.id, name="A")
        with pytest.raises(ProjectValidationError) as exc:
            set_participants_for_source(
                tmp_path, proj.id, fg.id, [a.id, "deadbeefcafe"]
            )
        assert "deadbeefcafe" in str(exc.value)
        # Failure must not have written anything.
        assert (
            load_participant(tmp_path, proj.id, a.id).source_ids == []
        )

    def test_invalid_id_shape_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(
                tmp_path, proj.id, fg.id, ["UPPERCASE123"]
            )
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(
                tmp_path, proj.id, fg.id, ["../escape"]
            )

    def test_unknown_source_rejected_by_default(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _save_participant(tmp_path, proj.id, name="A")
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(
                tmp_path, proj.id, "deadbeefcafe", [a.id]
            )

    def test_unknown_source_allowed_when_opted_in(self, tmp_path: Path) -> None:
        # Importer use-case: stage participant->source links before
        # the source files have been written.
        proj = _saved_project(tmp_path)
        a = _save_participant(tmp_path, proj.id, name="A")
        change = set_participants_for_source(
            tmp_path,
            proj.id,
            "deadbeefcafe",
            [a.id],
            require_source_exists=False,
        )
        assert change.added == [a.id]
        assert (
            load_participant(tmp_path, proj.id, a.id).source_ids
            == ["deadbeefcafe"]
        )

    def test_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(
                tmp_path, "UPPERCASE123", "0123456789ab", []
            )

    def test_validates_source_id_shape(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(tmp_path, proj.id, "BAD", [])

    def test_failure_does_not_partially_apply(self, tmp_path: Path) -> None:
        # Validation happens before any write — confirm by passing a bad id
        # in the middle of the desired list.
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG-1")
        a = _save_participant(
            tmp_path,
            proj.id,
            name="A",
            now="2024-01-01T00:00:00.000000Z",
        )
        b = _save_participant(
            tmp_path,
            proj.id,
            name="B",
            now="2024-01-01T00:00:00.000000Z",
        )
        with pytest.raises(ProjectValidationError):
            set_participants_for_source(
                tmp_path,
                proj.id,
                fg.id,
                [a.id, "deadbeefcafe", b.id],
                now="2024-06-01T00:00:00.000000Z",
            )
        # Neither A nor B was touched.
        assert load_participant(tmp_path, proj.id, a.id).source_ids == []
        assert load_participant(tmp_path, proj.id, b.id).source_ids == []
        assert (
            load_participant(tmp_path, proj.id, a.id).modified_at
            == "2024-01-01T00:00:00.000000Z"
        )

    def test_focus_group_round_trip(self, tmp_path: Path) -> None:
        # End-to-end: build a focus group, edit it, dissolve it, and
        # confirm each Participant.source_ids tracks the desired state.
        proj = _saved_project(tmp_path)
        fg = _save_source(tmp_path, proj.id, name="FG")
        a = _save_participant(tmp_path, proj.id, name="A")
        b = _save_participant(tmp_path, proj.id, name="B")
        c = _save_participant(tmp_path, proj.id, name="C")

        # Round 1: A + B.
        set_participants_for_source(tmp_path, proj.id, fg.id, [a.id, b.id])
        assert sorted(
            p.id for p in list_participants_for_source(tmp_path, proj.id, fg.id)
        ) == sorted([a.id, b.id])

        # Round 2: B leaves, C joins.
        set_participants_for_source(tmp_path, proj.id, fg.id, [a.id, c.id])
        assert sorted(
            p.id for p in list_participants_for_source(tmp_path, proj.id, fg.id)
        ) == sorted([a.id, c.id])

        # Round 3: dissolved.
        set_participants_for_source(tmp_path, proj.id, fg.id, [])
        assert (
            list_participants_for_source(tmp_path, proj.id, fg.id) == []
        )
        # Each participant's source_ids has fg.id removed.
        assert load_participant(tmp_path, proj.id, a.id).source_ids == []
        assert load_participant(tmp_path, proj.id, b.id).source_ids == []
        assert load_participant(tmp_path, proj.id, c.id).source_ids == []


# --------------------------------------------------------------------------- #
# ParticipantSourceChange data class
# --------------------------------------------------------------------------- #


class TestParticipantSourceChange:
    def test_changed_property(self) -> None:
        c = ParticipantSourceChange(source_id="0123456789ab")
        assert c.changed is False
        c.unchanged.append("aaaaaaaaaaaa")
        assert c.changed is False
        c.added.append("bbbbbbbbbbbb")
        assert c.changed is True
        c.added.clear()
        c.removed.append("cccccccccccc")
        assert c.changed is True


# --------------------------------------------------------------------------- #
# Integration: orphan link surfaces after a delete
# --------------------------------------------------------------------------- #


class TestIntegration:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s1 = _save_source(tmp_path, proj.id, name="A")
        s2 = _save_source(tmp_path, proj.id, name="B")
        p = _save_participant(
            tmp_path, proj.id, name="P", source_ids=[s1.id, s2.id]
        )
        # All clean.
        assert find_orphan_links(tmp_path, proj.id) == []
        assert [s.id for s in list_sources_for_participant(tmp_path, proj.id, p.id)] == [
            s1.id,
            s2.id,
        ]
        # Delete one source — orphan surfaces.
        delete_source(tmp_path, proj.id, s2.id)
        orphans = find_orphan_links(tmp_path, proj.id)
        assert orphans == [OrphanLink(participant_id=p.id, source_id=s2.id)]
        # Cleanup with unlink.
        unlink_participant_from_source(tmp_path, proj.id, p.id, s2.id)
        assert find_orphan_links(tmp_path, proj.id) == []
        # Then the participant disappears too — no orphans either.
        delete_participant(tmp_path, proj.id, p.id)
        assert list_participants(tmp_path, proj.id) == []
