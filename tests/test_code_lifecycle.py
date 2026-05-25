"""Tests for scribe.code_lifecycle (F2.3).

Exercise the rename / retire / promote / demote / set_code_parent /
merge / split lifecycle ops in pure Python:

* invariants (history preserved, no dangling back-pointers, no cycles);
* idempotency on no-op shapes (retiring an already-retired code,
  promoting a root code);
* every op interacts correctly with F2.2's version log.

Endpoint-level tests will live in test_server.py once F2.3 grows an
HTTP surface; for now the model is the public API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.codes import (
    Code,
    CodeRelation,
    list_codes,
    load_code,
    save_code,
)
from scribe.code_lifecycle import (
    demote_code,
    merge_codes,
    promote_code,
    rename_code,
    retire_code,
    set_code_parent,
    split_code,
)
from scribe.code_versions import (
    count_code_versions,
    latest_code_version,
    read_code_versions,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_code(
    tmp_path: Path,
    project_id: str,
    *,
    name: str = "Pacing",
    code_id: str | None = None,
    parent_code_id: str | None = None,
    exemplars: list[str] | None = None,
    related_codes: list[dict] | None = None,
    now: str | None = None,
    **rest,
) -> Code:
    c = Code.new(
        project_id=project_id,
        name=name,
        code_id=code_id,
        parent_code_id=parent_code_id,
        exemplars=exemplars,
        related_codes=related_codes,
        now=now,
        **rest,
    )
    save_code(tmp_path, c)
    return c


# --------------------------------------------------------------------------- #
# rename_code
# --------------------------------------------------------------------------- #


class TestRenameCode:
    def test_renames_and_records_version(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id, name="Old name")
        # Baseline: no versions yet.
        assert count_code_versions(tmp_path, proj.id, c.id) == 0

        renamed, version = rename_code(
            tmp_path, proj.id, c.id, "New name"
        )
        assert renamed.name == "New name"
        # Version log now has at least one entry (first save records v1).
        assert count_code_versions(tmp_path, proj.id, c.id) >= 1
        assert version.snapshot["name"] == "New name"

        # On disk reflects the rename.
        on_disk = load_code(tmp_path, proj.id, c.id)
        assert on_disk.name == "New name"

    def test_rename_advances_modified_at(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(
            tmp_path, proj.id, name="Old", now="2024-01-01T00:00:00.000000Z"
        )
        rename_code(
            tmp_path,
            proj.id,
            c.id,
            "New",
            now="2024-06-01T00:00:00.000000Z",
        )
        loaded = load_code(tmp_path, proj.id, c.id)
        assert loaded.modified_at == "2024-06-01T00:00:00.000000Z"
        assert loaded.created_at == "2024-01-01T00:00:00.000000Z"

    def test_rename_after_initial_save_records_v2(self, tmp_path: Path) -> None:
        # If a baseline v1 has already been recorded, a rename pushes v2.
        from scribe.code_versions import save_code_with_version

        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="A")
        save_code_with_version(tmp_path, c)  # records v1
        assert count_code_versions(tmp_path, proj.id, c.id) == 1

        _, version = rename_code(tmp_path, proj.id, c.id, "B")
        assert version.version == 2
        assert count_code_versions(tmp_path, proj.id, c.id) == 2

    def test_rename_validates_name(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id, name="Ok")
        for bad in ("", "   ", "x" * 1000):
            with pytest.raises(ProjectValidationError):
                rename_code(tmp_path, proj.id, c.id, bad)
        # Disk state untouched.
        assert load_code(tmp_path, proj.id, c.id).name == "Ok"

    def test_rename_change_note_recorded(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id, name="A")
        _, version = rename_code(
            tmp_path, proj.id, c.id, "B", change_note="Sharper label"
        )
        assert version.change_note == "Sharper label"

    def test_rename_invalid_ids_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            rename_code(tmp_path, "BAD", "0123456789ab", "x")
        with pytest.raises(ProjectValidationError):
            rename_code(tmp_path, proj.id, "BAD", "x")


# --------------------------------------------------------------------------- #
# retire_code
# --------------------------------------------------------------------------- #


class TestRetireCode:
    def test_sets_status_retired(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id)
        retired, version = retire_code(tmp_path, proj.id, c.id)
        assert retired.status == "retired"
        assert load_code(tmp_path, proj.id, c.id).status == "retired"
        # Audit version recorded even though only metadata changed.
        assert version.snapshot["status"] == "retired"

    def test_records_audit_version_for_metadata_only(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id)
        retire_code(tmp_path, proj.id, c.id)
        # Without F2.3's audit hook, save_code_with_version on a
        # status-only change wouldn't record. We force a snapshot.
        assert count_code_versions(tmp_path, proj.id, c.id) >= 1

    def test_idempotent_on_already_retired(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id)
        retire_code(tmp_path, proj.id, c.id)
        v_count = count_code_versions(tmp_path, proj.id, c.id)
        # Second retire: same disk state, no extra version row.
        retire_code(tmp_path, proj.id, c.id)
        assert load_code(tmp_path, proj.id, c.id).status == "retired"
        assert count_code_versions(tmp_path, proj.id, c.id) == v_count

    def test_change_note_recorded(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id)
        _, version = retire_code(
            tmp_path, proj.id, c.id, change_note="Subsumed by category"
        )
        assert version.change_note == "Subsumed by category"

    def test_retire_invalid_ids_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            retire_code(tmp_path, "BAD", "0123456789ab")
        with pytest.raises(ProjectValidationError):
            retire_code(tmp_path, proj.id, "BAD")

    def test_idempotent_seeds_baseline_if_missing(
        self, tmp_path: Path
    ) -> None:
        # Pre-existing code that's already retired but has no version log
        # (e.g. data predating F2.2). Calling retire should seed a
        # baseline snapshot so callers always have something to point at.
        proj = _saved_project(tmp_path)
        c = Code.new(project_id=proj.id, name="legacy", status="retired")
        save_code(tmp_path, c)
        assert count_code_versions(tmp_path, proj.id, c.id) == 0
        _, version = retire_code(tmp_path, proj.id, c.id)
        assert version.snapshot["status"] == "retired"
        assert count_code_versions(tmp_path, proj.id, c.id) == 1


# --------------------------------------------------------------------------- #
# set_code_parent / promote_code / demote_code
# --------------------------------------------------------------------------- #


class TestSetCodeParent:
    def test_set_parent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B")
        moved, version = set_code_parent(tmp_path, proj.id, b.id, a.id)
        assert moved.parent_code_id == a.id
        assert version.snapshot["parent_code_id"] == a.id

    def test_detach_via_none(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        moved, version = set_code_parent(tmp_path, proj.id, b.id, None)
        assert moved.parent_code_id is None
        assert version.snapshot["parent_code_id"] is None

    def test_detach_via_empty_string(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        moved, _ = set_code_parent(tmp_path, proj.id, b.id, "")
        assert moved.parent_code_id is None

    def test_self_parent_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, a.id, a.id)

    def test_nonexistent_parent_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, a.id, "ffffffffffff")

    def test_cycle_detection_simple(self, tmp_path: Path) -> None:
        # B is child of A. Setting A.parent = B closes a cycle.
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, a.id, b.id)

    def test_cycle_detection_indirect(self, tmp_path: Path) -> None:
        # C → B → A. Setting A.parent = C closes a cycle through B.
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        c = _make_code(tmp_path, proj.id, name="C", parent_code_id=b.id)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, a.id, c.id)

    def test_invalid_ids_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, "BAD", "0123456789ab", None)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, "BAD", None)
        # Parent id is also validated when non-empty.
        a = _make_code(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            set_code_parent(tmp_path, proj.id, a.id, "BAD")


class TestPromoteCode:
    def test_promote_lifts_one_level(self, tmp_path: Path) -> None:
        # C → B → A; promote C → C.parent_code_id == A.
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        c = _make_code(tmp_path, proj.id, name="C", parent_code_id=b.id)
        promoted, version = promote_code(tmp_path, proj.id, c.id)
        assert promoted.parent_code_id == a.id
        assert version.snapshot["parent_code_id"] == a.id

    def test_promote_to_root(self, tmp_path: Path) -> None:
        # B → A; promote B → root.
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        promoted, _ = promote_code(tmp_path, proj.id, b.id)
        assert promoted.parent_code_id is None

    def test_promote_root_is_noop(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        # No prior versions.
        v_before = count_code_versions(tmp_path, proj.id, a.id)
        promoted, version = promote_code(tmp_path, proj.id, a.id)
        assert promoted.parent_code_id is None
        # Even on no-op we ensure a baseline snapshot exists, so the
        # version count goes 0→1 the *first* time. A subsequent promote
        # finds the latest and returns it without writing again.
        v_after = count_code_versions(tmp_path, proj.id, a.id)
        assert v_after >= max(1, v_before)
        promoted2, version2 = promote_code(tmp_path, proj.id, a.id)
        # No new write the second time around.
        assert count_code_versions(tmp_path, proj.id, a.id) == v_after
        assert version2.id == version.id


class TestDemoteCode:
    def test_demote_sets_parent(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B")
        demoted, version = demote_code(tmp_path, proj.id, b.id, a.id)
        assert demoted.parent_code_id == a.id
        assert version.snapshot["parent_code_id"] == a.id

    def test_demote_empty_parent_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        b = _make_code(tmp_path, proj.id, name="B")
        with pytest.raises(ProjectValidationError):
            demote_code(tmp_path, proj.id, b.id, "")
        with pytest.raises(ProjectValidationError):
            demote_code(tmp_path, proj.id, b.id, None)  # type: ignore[arg-type]

    def test_demote_cycle_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = _make_code(tmp_path, proj.id, name="A")
        b = _make_code(tmp_path, proj.id, name="B", parent_code_id=a.id)
        # demote A under B would create A→B→A.
        with pytest.raises(ProjectValidationError):
            demote_code(tmp_path, proj.id, a.id, b.id)


# --------------------------------------------------------------------------- #
# merge_codes
# --------------------------------------------------------------------------- #


class TestMergeCodes:
    def test_target_absorbs_exemplars_deduped(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(
            tmp_path, proj.id, name="Target", exemplars=["a", "b"]
        )
        src = _make_code(
            tmp_path,
            proj.id,
            name="Src",
            exemplars=["b", "c"],
        )
        merged_target, _ = merge_codes(
            tmp_path, proj.id, [src.id], target.id
        )
        # Order: target's first, then source's new ones.
        assert merged_target.exemplars == ["a", "b", "c"]

    def test_target_absorbs_related_codes_filtered(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        # Three peripheral codes the relations point at.
        x = _make_code(tmp_path, proj.id, name="X")
        y = _make_code(tmp_path, proj.id, name="Y")
        z = _make_code(tmp_path, proj.id, name="Z")
        target = _make_code(
            tmp_path,
            proj.id,
            name="Target",
            related_codes=[
                {"code_id": x.id, "relation_type": "associated"},
            ],
        )
        src = _make_code(
            tmp_path,
            proj.id,
            name="Src",
            related_codes=[
                {"code_id": y.id, "relation_type": "associated"},
                # Edge points back at target — should be dropped.
                {"code_id": target.id, "relation_type": "broader"},
                # Duplicate of target's existing edge — should be dropped.
                {"code_id": x.id, "relation_type": "associated"},
                # Distinct — should be kept.
                {"code_id": z.id, "relation_type": "narrower"},
            ],
        )
        merged_target, _ = merge_codes(
            tmp_path, proj.id, [src.id], target.id
        )
        rel_pairs = [
            (r.code_id, r.relation_type) for r in merged_target.related_codes
        ]
        assert (x.id, "associated") in rel_pairs
        assert (y.id, "associated") in rel_pairs
        assert (z.id, "narrower") in rel_pairs
        # No edges to target itself or to the merged source.
        assert (target.id, "broader") not in rel_pairs
        for r in merged_target.related_codes:
            assert r.code_id != src.id

    def test_drops_relations_to_other_sources(self, tmp_path: Path) -> None:
        # Multi-source merge: src2 has an edge to src1; that edge must
        # not survive on the target (would point at a retired code).
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src1 = _make_code(tmp_path, proj.id, name="S1")
        src2 = _make_code(
            tmp_path,
            proj.id,
            name="S2",
            related_codes=[
                {"code_id": src1.id, "relation_type": "associated"},
            ],
        )
        merged_target, _ = merge_codes(
            tmp_path, proj.id, [src1.id, src2.id], target.id
        )
        for r in merged_target.related_codes:
            assert r.code_id != src1.id
            assert r.code_id != src2.id

    def test_other_codes_parent_pointer_rerouted(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(tmp_path, proj.id, name="S")
        child = _make_code(
            tmp_path, proj.id, name="C", parent_code_id=src.id
        )
        merge_codes(tmp_path, proj.id, [src.id], target.id)
        loaded_child = load_code(tmp_path, proj.id, child.id)
        assert loaded_child.parent_code_id == target.id
        # Re-route records a version on the child.
        assert count_code_versions(tmp_path, proj.id, child.id) >= 1

    def test_other_codes_related_pointers_rerouted(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(tmp_path, proj.id, name="S")
        # Other code points at the source via associated.
        other = _make_code(
            tmp_path,
            proj.id,
            name="O",
            related_codes=[
                {"code_id": src.id, "relation_type": "associated"},
            ],
        )
        merge_codes(tmp_path, proj.id, [src.id], target.id)
        loaded_other = load_code(tmp_path, proj.id, other.id)
        rel = [
            (r.code_id, r.relation_type)
            for r in loaded_other.related_codes
        ]
        assert rel == [(target.id, "associated")]

    def test_reroute_drops_self_edges(self, tmp_path: Path) -> None:
        # If the target itself has an edge to the source, the rewrite
        # would create a self-edge on the target. We absorbed those at
        # the target step; for a *third* code that already had an edge
        # to the target AND an edge to the source, the duplicate after
        # rewrite must collapse.
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(tmp_path, proj.id, name="S")
        # The "other" code points at both target and src via the same
        # relation. After rewrite, both edges would become (target,
        # associated) — the dup must collapse to one.
        other = _make_code(
            tmp_path,
            proj.id,
            name="O",
            related_codes=[
                {"code_id": target.id, "relation_type": "associated"},
                {"code_id": src.id, "relation_type": "associated"},
            ],
        )
        merge_codes(tmp_path, proj.id, [src.id], target.id)
        loaded_other = load_code(tmp_path, proj.id, other.id)
        rel = [
            (r.code_id, r.relation_type)
            for r in loaded_other.related_codes
        ]
        assert rel == [(target.id, "associated")]

    def test_self_edge_dropped_when_target_points_at_source(
        self, tmp_path: Path
    ) -> None:
        # Conceptually the target itself might point at the source via
        # a related-code edge before the merge. After rewrite that edge
        # would land on (target, target) — must be dropped.
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="S")
        target = _make_code(
            tmp_path,
            proj.id,
            name="T",
            related_codes=[
                {"code_id": src.id, "relation_type": "associated"},
            ],
        )
        merged_target, _ = merge_codes(
            tmp_path, proj.id, [src.id], target.id
        )
        # The (target, target) edge must not land on the target.
        for r in merged_target.related_codes:
            assert r.code_id != target.id
            assert r.code_id != src.id

    def test_sources_retired_with_provenance(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src1 = _make_code(tmp_path, proj.id, name="S1")
        src2 = _make_code(tmp_path, proj.id, name="S2")
        _, retired = merge_codes(
            tmp_path, proj.id, [src1.id, src2.id], target.id
        )
        assert {s.id for s in retired} == {src1.id, src2.id}
        for sid in (src1.id, src2.id):
            loaded = load_code(tmp_path, proj.id, sid)
            assert loaded.status == "retired"
            assert loaded.provenance.get("merged_into") == target.id
            # Audit version recorded on the source.
            assert count_code_versions(tmp_path, proj.id, sid) >= 1

    def test_target_records_version(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(
            tmp_path, proj.id, name="S", exemplars=["new"]
        )
        v_before = count_code_versions(tmp_path, proj.id, target.id)
        merge_codes(tmp_path, proj.id, [src.id], target.id)
        v_after = count_code_versions(tmp_path, proj.id, target.id)
        # Definitional change → a new version row.
        assert v_after > v_before

    def test_dedupes_source_list(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(
            tmp_path, proj.id, name="S", exemplars=["q"]
        )
        merged_target, retired = merge_codes(
            tmp_path, proj.id, [src.id, src.id], target.id
        )
        assert len(retired) == 1
        # No double-add of exemplars.
        assert merged_target.exemplars.count("q") == 1

    def test_empty_sources_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        with pytest.raises(ProjectValidationError):
            merge_codes(tmp_path, proj.id, [], target.id)

    def test_target_in_sources_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        with pytest.raises(ProjectValidationError):
            merge_codes(tmp_path, proj.id, [target.id], target.id)

    def test_invalid_ids_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            merge_codes(tmp_path, "BAD", ["0123456789ab"], "0123456789ab")
        target = _make_code(tmp_path, proj.id, name="T")
        with pytest.raises(ProjectValidationError):
            merge_codes(tmp_path, proj.id, ["BAD"], target.id)
        with pytest.raises(ProjectValidationError):
            merge_codes(tmp_path, proj.id, [target.id], "BAD")

    def test_history_preserved_on_target(self, tmp_path: Path) -> None:
        # The target's existing version log must not be rewritten.
        from scribe.code_versions import save_code_with_version

        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        # Seed a v1 manually so we have a known history before the merge.
        save_code_with_version(tmp_path, target)
        prior_versions = read_code_versions(tmp_path, proj.id, target.id)
        assert len(prior_versions) == 1

        src = _make_code(
            tmp_path, proj.id, name="S", exemplars=["q"]
        )
        merge_codes(tmp_path, proj.id, [src.id], target.id)

        post_versions = read_code_versions(tmp_path, proj.id, target.id)
        # Old v1 still there, byte-identical at the IDs we care about.
        assert post_versions[0].id == prior_versions[0].id
        assert post_versions[0].version == 1
        # New version for the merge.
        assert post_versions[-1].version == 2


# --------------------------------------------------------------------------- #
# split_code
# --------------------------------------------------------------------------- #


class TestSplitCode:
    def test_creates_new_codes(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(
            tmp_path,
            proj.id,
            name="Pacing",
            definition="Source def",
            exemplars=["e1", "e2", "e3"],
        )
        retired_src, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [
                {"name": "Pacing self", "exemplars": ["e1"]},
                {"name": "Pacing other", "exemplars": ["e2", "e3"]},
            ],
        )
        assert retired_src.status == "retired"
        assert len(new_codes) == 2
        assert {c.name for c in new_codes} == {"Pacing self", "Pacing other"}

    def test_new_codes_default_to_source_fields(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(
            tmp_path,
            proj.id,
            name="X",
            definition="src def",
            inclusion_criteria="src in",
            exclusion_criteria="src out",
            theoretical_memo="src memo",
            stage="focused",
            colour="#abcdef",
        )
        _, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [
                {"name": "Y"},
                {"name": "Z"},
            ],
        )
        for c in new_codes:
            assert c.definition == "src def"
            assert c.inclusion_criteria == "src in"
            assert c.exclusion_criteria == "src out"
            assert c.theoretical_memo == "src memo"
            assert c.stage == "focused"
            assert c.colour == "#abcdef"

    def test_new_code_provenance_marks_split_from(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        _, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [{"name": "A"}, {"name": "B"}],
        )
        for c in new_codes:
            assert c.provenance.get("split_from") == src.id
            assert c.provenance.get("source") == "human"

    def test_source_provenance_marks_split_into(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        retired_src, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [{"name": "A"}, {"name": "B"}],
        )
        loaded_src = load_code(tmp_path, proj.id, src.id)
        expected = ",".join(c.id for c in new_codes)
        assert loaded_src.provenance.get("split_into") == expected
        assert loaded_src.status == "retired"

    def test_audit_version_on_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        v_before = count_code_versions(tmp_path, proj.id, src.id)
        split_code(
            tmp_path,
            proj.id,
            src.id,
            [{"name": "A"}, {"name": "B"}],
        )
        v_after = count_code_versions(tmp_path, proj.id, src.id)
        assert v_after > v_before

    def test_each_new_code_has_initial_version(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        _, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [{"name": "A"}, {"name": "B"}],
        )
        for c in new_codes:
            versions = read_code_versions(tmp_path, proj.id, c.id)
            assert len(versions) == 1
            assert versions[0].version == 1

    def test_minimum_two_specs_required(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        with pytest.raises(ProjectValidationError):
            split_code(tmp_path, proj.id, src.id, [])
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path, proj.id, src.id, [{"name": "only one"}]
            )

    def test_each_spec_must_have_name(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                proj.id,
                src.id,
                [{"name": "A"}, {"definition": "no name"}],
            )
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                proj.id,
                src.id,
                [{"name": "A"}, {"name": "   "}],
            )

    def test_spec_must_be_dict(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                proj.id,
                src.id,
                [{"name": "A"}, "not a dict"],  # type: ignore[list-item]
            )

    def test_spec_provenance_must_be_dict(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(tmp_path, proj.id, name="X")
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                proj.id,
                src.id,
                [
                    {"name": "A"},
                    {"name": "B", "provenance": ["bad"]},
                ],
            )

    def test_invalid_ids_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                "BAD",
                "0123456789ab",
                [{"name": "A"}, {"name": "B"}],
            )
        with pytest.raises(ProjectValidationError):
            split_code(
                tmp_path,
                proj.id,
                "BAD",
                [{"name": "A"}, {"name": "B"}],
            )

    def test_overrides_apply(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _make_code(
            tmp_path,
            proj.id,
            name="X",
            stage="initial",
            colour="#aaaaaa",
        )
        _, new_codes = split_code(
            tmp_path,
            proj.id,
            src.id,
            [
                {"name": "A", "stage": "focused", "colour": "#bbbbbb"},
                {"name": "B"},
            ],
        )
        # A: explicit overrides took effect.
        a = next(c for c in new_codes if c.name == "A")
        assert a.stage == "focused"
        assert a.colour == "#bbbbbb"
        # B: defaults from source.
        b = next(c for c in new_codes if c.name == "B")
        assert b.stage == "initial"
        assert b.colour == "#aaaaaa"


# --------------------------------------------------------------------------- #
# Cross-op invariant: history is never destroyed
# --------------------------------------------------------------------------- #


class TestHistoryPreserved:
    def test_rename_then_retire_keeps_full_log(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        c = _make_code(tmp_path, proj.id, name="A")
        rename_code(tmp_path, proj.id, c.id, "B")
        rename_code(tmp_path, proj.id, c.id, "C")
        retire_code(tmp_path, proj.id, c.id)
        versions = read_code_versions(tmp_path, proj.id, c.id)
        names = [v.snapshot["name"] for v in versions]
        # Names appear in order: B (v1, first save_with_version), C (v2),
        # then a final audit row at status=retired (still name C).
        assert names == ["B", "C", "C"]
        statuses = [v.snapshot["status"] for v in versions]
        assert statuses == ["active", "active", "retired"]

    def test_merge_does_not_delete_source_files(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        target = _make_code(tmp_path, proj.id, name="T")
        src = _make_code(tmp_path, proj.id, name="S")
        merge_codes(tmp_path, proj.id, [src.id], target.id)
        # Source file is still on disk; only its status changed.
        assert load_code(tmp_path, proj.id, src.id).status == "retired"
        # And shows up in list_codes.
        ids = {c.id for c in list_codes(tmp_path, proj.id)}
        assert src.id in ids
        assert target.id in ids
