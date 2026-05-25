"""Tests for scribe.saved_queries (F3.7).

Saved queries (named, re-runnable). The module is a thin wrapper
around :class:`scribe.query.Query`; tests focus on the persistence
layer, the run-tracking semantics, and the cross-entity invariants
(project-id mismatch, name requirement, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.codes import new_code_id
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.query import (
    CodeExpr,
    CodeFilter,
    Query,
    QueryValidationError,
    SourceFilter,
)
from scribe.saved_queries import (
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    SAVED_QUERIES_DIRNAME,
    SAVED_QUERY_ID_RE,
    SavedQuery,
    delete_saved_query,
    list_saved_queries,
    load_saved_query,
    new_saved_query_id,
    record_run,
    run_saved_query,
    save_saved_query,
    saved_queries_dir,
    saved_query_state_path,
)
from scribe.sources import Source


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _query(
    project_id: str,
    *,
    name: str = "Quotes about power",
    description: str = "",
    code_id: str | None = None,
) -> Query:
    cid = code_id or new_code_id()
    return Query(
        project_id=project_id,
        name=name,
        description=description,
        codes=CodeFilter(expr=CodeExpr.code(cid)),
    )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewSavedQueryId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert SAVED_QUERY_ID_RE.match(new_saved_query_id())

    def test_unique(self) -> None:
        ids = {new_saved_query_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# SavedQuery.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestSavedQueryNew:
    def test_minimal(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        assert SAVED_QUERY_ID_RE.match(sq.id)
        assert sq.project_id == p.id
        assert sq.created_at == sq.modified_at
        assert sq.last_run_at == ""
        assert sq.run_count == 0
        assert sq.query.name == "Quotes about power"

    def test_explicit_id_and_now(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        ts = "2026-01-15T12:00:00.000000Z"
        sq = SavedQuery.new(
            project_id=p.id,
            query=q,
            saved_query_id="aaaaaaaaaaaa",
            now=ts,
        )
        assert sq.id == "aaaaaaaaaaaa"
        assert sq.created_at == ts
        assert sq.modified_at == ts

    def test_rejects_project_mismatch(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        with pytest.raises(QueryValidationError):
            SavedQuery.new(project_id="aaaaaaaaaaaa", query=q)

    def test_rejects_blank_query_name(self) -> None:
        p = Project.new(name="P")
        # A SavedQuery requires a non-empty display name even though
        # the underlying Query allows empty.
        q = Query(project_id=p.id, name="   ")
        with pytest.raises(QueryValidationError):
            SavedQuery.new(project_id=p.id, query=q)

    def test_rejects_invalid_id(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        with pytest.raises(QueryValidationError):
            SavedQuery.new(
                project_id=p.id,
                query=q,
                saved_query_id="not-hex",
            )

    def test_rejects_invalid_project_id(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        # Manually break the project id at the SavedQuery level — the
        # Query itself stays valid, but the wrapper fails.
        sq = SavedQuery(
            id=new_saved_query_id(),
            project_id="zz!!nothex!!",
            query=q,
        )
        with pytest.raises(QueryValidationError):
            sq.validate()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidate:
    def test_validate_strips_name(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id, name="  My query  ")
        sq = SavedQuery.new(project_id=p.id, query=q)
        assert sq.query.name == "My query"
        # Property mirrors the underlying query.
        assert sq.name == "My query"

    def test_rejects_name_too_long(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id, name="x" * (MAX_NAME_LEN + 1))
        with pytest.raises(QueryValidationError):
            SavedQuery.new(project_id=p.id, query=q)

    def test_rejects_description_too_long(self) -> None:
        p = Project.new(name="P")
        # The wrapped Query.validate() catches this too — the cap is
        # shared. Test it here so a future refactor of the sharing
        # doesn't silently weaken the bound.
        q = Query(
            project_id=p.id,
            name="Q",
            description="x" * (MAX_DESCRIPTION_LEN + 1),
        )
        with pytest.raises(QueryValidationError):
            SavedQuery.new(project_id=p.id, query=q)

    def test_rejects_non_query_object(self) -> None:
        sq = SavedQuery(
            id=new_saved_query_id(),
            project_id="aaaaaaaaaaaa",
            query="not a Query",  # type: ignore[arg-type]
        )
        with pytest.raises(QueryValidationError):
            sq.validate()

    def test_rejects_negative_run_count(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        sq.run_count = -1
        with pytest.raises(QueryValidationError):
            sq.validate()


# --------------------------------------------------------------------------- #
# Round-trip through dict
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_round_trip_preserves_fields(self) -> None:
        p = Project.new(name="P")
        cid = new_code_id()
        q = Query(
            project_id=p.id,
            name="Power quotes",
            description="An audit-time keepsake.",
            sources=SourceFilter(languages=["en"]),
            codes=CodeFilter(expr=CodeExpr.code(cid)),
        )
        sq = SavedQuery.new(project_id=p.id, query=q)
        sq.run_count = 5
        sq.last_run_at = "2026-02-01T10:00:00.000000Z"
        round_tripped = SavedQuery.from_dict(sq.to_dict())
        assert round_tripped.id == sq.id
        assert round_tripped.project_id == sq.project_id
        assert round_tripped.query.to_dict() == sq.query.to_dict()
        assert round_tripped.created_at == sq.created_at
        assert round_tripped.modified_at == sq.modified_at
        assert round_tripped.last_run_at == sq.last_run_at
        assert round_tripped.run_count == 5

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_requires_id(self) -> None:
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict({"project_id": "a" * 12, "query": {}})

    def test_from_dict_requires_query(self) -> None:
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict(
                {"id": "a" * 12, "project_id": "a" * 12}
            )

    def test_from_dict_rejects_query_non_object(self) -> None:
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict(
                {
                    "id": "a" * 12,
                    "project_id": "a" * 12,
                    "query": "scalar",
                }
            )

    def test_from_dict_accepts_integer_float_run_count(self) -> None:
        # JSON round-trips can flatten ints to floats; we accept that
        # but reject non-integer floats.
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        d = sq.to_dict()
        d["run_count"] = 3.0
        round_tripped = SavedQuery.from_dict(d)
        assert round_tripped.run_count == 3

    def test_from_dict_rejects_fractional_run_count(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        d = sq.to_dict()
        d["run_count"] = 3.5
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict(d)

    def test_from_dict_rejects_bool_run_count(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        d = sq.to_dict()
        d["run_count"] = True
        with pytest.raises(QueryValidationError):
            SavedQuery.from_dict(d)


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def test_renames_via_name_shortcut(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id, name="Old")
        sq = SavedQuery.new(
            project_id=p.id,
            query=q,
            now="2026-01-01T00:00:00.000000Z",
        )
        sq.apply_update({"name": "New"}, now="2026-01-02T00:00:00.000000Z")
        assert sq.query.name == "New"
        assert sq.modified_at == "2026-01-02T00:00:00.000000Z"
        # created_at not changed.
        assert sq.created_at == "2026-01-01T00:00:00.000000Z"

    def test_updates_description(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        sq.apply_update({"description": "  Better explanation  "})
        # The wrapped Query trims at validate-time? Description has no
        # trim — but length cap is enforced. Empty description allowed.
        assert sq.query.description == "  Better explanation  "

    def test_replaces_full_query(self) -> None:
        p = Project.new(name="P")
        cid = new_code_id()
        q = _query(p.id, code_id=cid)
        sq = SavedQuery.new(project_id=p.id, query=q)
        new_cid = new_code_id()
        new_q_dict = Query(
            project_id=p.id,
            name="Replaced",
            codes=CodeFilter(expr=CodeExpr.code(new_cid)),
        ).to_dict()
        sq.apply_update({"query": new_q_dict})
        assert sq.query.name == "Replaced"
        assert sq.query.referenced_code_ids() == {new_cid}

    def test_rejects_query_with_wrong_project_id(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        wrong = Query(project_id="bbbbbbbbbbbb", name="x").to_dict()
        with pytest.raises(QueryValidationError):
            sq.apply_update({"query": wrong})

    def test_failed_update_does_not_advance_modified_at(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(
            project_id=p.id,
            query=q,
            now="2026-01-01T00:00:00.000000Z",
        )
        with pytest.raises(QueryValidationError):
            sq.apply_update({"name": ""}, now="2026-02-02T00:00:00.000000Z")
        assert sq.modified_at == "2026-01-01T00:00:00.000000Z"

    def test_rejects_unknown_field(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        with pytest.raises(QueryValidationError):
            sq.apply_update({"colour": "#fff"})

    def test_ignores_managed_fields(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        original_id = sq.id
        original_run_count = sq.run_count
        sq.apply_update(
            {
                "id": "ffffffffffff",
                "project_id": "bbbbbbbbbbbb",
                "created_at": "2099-01-01T00:00:00.000000Z",
                "last_run_at": "2099-02-02T00:00:00.000000Z",
                "run_count": 999,
                "name": "Just renaming",
            }
        )
        assert sq.id == original_id
        assert sq.project_id == p.id
        assert sq.run_count == original_run_count
        assert sq.last_run_at == ""
        assert sq.query.name == "Just renaming"

    def test_rejects_non_object_update(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        with pytest.raises(QueryValidationError):
            sq.apply_update("not a dict")  # type: ignore[arg-type]

    def test_rejects_query_payload_non_object(self) -> None:
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        with pytest.raises(QueryValidationError):
            sq.apply_update({"query": "scalar"})


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_then_load(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        q = _query(p.id, name="Persisted")
        sq = SavedQuery.new(project_id=p.id, query=q)
        save_saved_query(tmp_path, sq)
        loaded = load_saved_query(tmp_path, p.id, sq.id)
        assert loaded.id == sq.id
        assert loaded.name == "Persisted"

    def test_save_atomic_via_tmp(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        save_saved_query(tmp_path, sq)
        # No leftover .json.tmp after a successful save.
        sd = saved_queries_dir(tmp_path, p.id)
        assert not any(f.name.endswith(".json.tmp") for f in sd.iterdir())

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        # No project saved → directory doesn't exist.
        p = Project.new(name="P")
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        with pytest.raises(FileNotFoundError):
            save_saved_query(tmp_path, sq)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_saved_query(tmp_path, p.id, "aaaaaaaaaaaa")

    def test_save_path_is_queries_dir(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        q = _query(p.id)
        sq = SavedQuery.new(project_id=p.id, query=q)
        target = save_saved_query(tmp_path, sq)
        assert target.parent.name == SAVED_QUERIES_DIRNAME
        assert target.name == f"{sq.id}.json"

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(QueryValidationError):
            saved_query_state_path(tmp_path, p.id, "../escape")

    def test_state_path_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            saved_query_state_path(tmp_path, "../escape", "a" * 12)


# --------------------------------------------------------------------------- #
# list_saved_queries
# --------------------------------------------------------------------------- #


class TestListSavedQueries:
    def test_empty_when_no_dir(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert list_saved_queries(tmp_path, p.id) == []

    def test_lists_in_modified_desc_order(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq_old = SavedQuery.new(
            project_id=p.id,
            query=_query(p.id, name="Old"),
            now="2026-01-01T00:00:00.000000Z",
        )
        sq_new = SavedQuery.new(
            project_id=p.id,
            query=_query(p.id, name="New"),
            now="2026-02-01T00:00:00.000000Z",
        )
        # Save in the *opposite* order to prove sort isn't FS-order.
        save_saved_query(tmp_path, sq_old)
        save_saved_query(tmp_path, sq_new)
        names = [s.name for s in list_saved_queries(tmp_path, p.id)]
        assert names == ["New", "Old"]

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(project_id=p.id, query=_query(p.id))
        save_saved_query(tmp_path, sq)
        # Drop a malformed JSON file in the dir.
        sd = saved_queries_dir(tmp_path, p.id)
        (sd / "bbbbbbbbbbbb.json").write_text("{ not json")
        # And a wrong-shape file.
        (sd / "cccccccccccc.json").write_text(
            json.dumps({"id": "cccccccccccc", "project_id": "x"})
        )
        # And a non-12-hex filename.
        (sd / "not-an-id.json").write_text("{}")
        # And a leftover tmp file.
        (sd / "dddddddddddd.json.tmp").write_text("{}")
        listed = list_saved_queries(tmp_path, p.id)
        assert [s.id for s in listed] == [sq.id]

    def test_rejects_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(QueryValidationError):
            list_saved_queries(tmp_path, "not-hex")


# --------------------------------------------------------------------------- #
# delete_saved_query
# --------------------------------------------------------------------------- #


class TestDelete:
    def test_delete_removes_file(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(project_id=p.id, query=_query(p.id))
        save_saved_query(tmp_path, sq)
        assert delete_saved_query(tmp_path, p.id, sq.id) is True
        with pytest.raises(FileNotFoundError):
            load_saved_query(tmp_path, p.id, sq.id)

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert (
            delete_saved_query(tmp_path, p.id, "aaaaaaaaaaaa") is False
        )

    def test_delete_validates_id(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(QueryValidationError):
            delete_saved_query(tmp_path, p.id, "../escape")


# --------------------------------------------------------------------------- #
# record_run + run_saved_query
# --------------------------------------------------------------------------- #


class TestRecordRun:
    def test_increments_run_count(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(project_id=p.id, query=_query(p.id))
        save_saved_query(tmp_path, sq)
        updated = record_run(tmp_path, p.id, sq.id)
        assert updated.run_count == 1
        assert updated.last_run_at != ""

    def test_persists_across_loads(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(project_id=p.id, query=_query(p.id))
        save_saved_query(tmp_path, sq)
        record_run(tmp_path, p.id, sq.id)
        record_run(tmp_path, p.id, sq.id)
        record_run(tmp_path, p.id, sq.id)
        loaded = load_saved_query(tmp_path, p.id, sq.id)
        assert loaded.run_count == 3
        assert loaded.last_run_at != ""

    def test_does_not_bump_modified_at(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(
            project_id=p.id,
            query=_query(p.id),
            now="2026-01-01T00:00:00.000000Z",
        )
        save_saved_query(tmp_path, sq)
        updated = record_run(
            tmp_path,
            p.id,
            sq.id,
            now="2026-06-15T12:00:00.000000Z",
        )
        # Recording a run is not a methodological edit — modified_at
        # is the codebook-state timestamp, last_run_at is the audit
        # timestamp.
        assert updated.modified_at == "2026-01-01T00:00:00.000000Z"
        assert updated.last_run_at == "2026-06-15T12:00:00.000000Z"

    def test_uses_provided_now(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        sq = SavedQuery.new(project_id=p.id, query=_query(p.id))
        save_saved_query(tmp_path, sq)
        ts = "2026-09-01T10:00:00.000000Z"
        updated = record_run(tmp_path, p.id, sq.id, now=ts)
        assert updated.last_run_at == ts

    def test_missing_query_raises(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            record_run(tmp_path, p.id, "aaaaaaaaaaaa")


class TestRunSavedQuery:
    def test_executes_and_records(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        cid = new_code_id()
        cid_other = new_code_id()
        q = Query(
            project_id=p.id,
            name="By code",
            codes=CodeFilter(expr=CodeExpr.code(cid)),
        )
        sq = SavedQuery.new(project_id=p.id, query=q)
        save_saved_query(tmp_path, sq)

        sid = "11" * 6
        apps = [
            {"code_id": cid, "source_id": sid},
            {"code_id": cid_other, "source_id": sid},
            {"code_id": cid, "source_id": sid},
        ]
        updated, matches = run_saved_query(
            tmp_path, p.id, sq.id, apps
        )
        assert len(matches) == 2
        assert all(a["code_id"] == cid for a in matches)
        assert updated.run_count == 1

    def test_record_false_does_not_increment(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        cid = new_code_id()
        q = Query(
            project_id=p.id,
            name="By code",
            codes=CodeFilter(expr=CodeExpr.code(cid)),
        )
        sq = SavedQuery.new(project_id=p.id, query=q)
        save_saved_query(tmp_path, sq)

        sid = "22" * 6
        apps = [{"code_id": cid, "source_id": sid}]
        updated, matches = run_saved_query(
            tmp_path, p.id, sq.id, apps, record=False
        )
        assert len(matches) == 1
        assert updated.run_count == 0
        assert updated.last_run_at == ""
        # Verify on disk too.
        loaded = load_saved_query(tmp_path, p.id, sq.id)
        assert loaded.run_count == 0

    def test_passes_sources_for_source_filter(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        cid = new_code_id()
        sid_kept = "10" * 6
        sid_dropped = "20" * 6
        q = Query(
            project_id=p.id,
            name="EN sources only",
            sources=SourceFilter(languages=["en"]),
            codes=CodeFilter(expr=CodeExpr.code(cid)),
        )
        sq = SavedQuery.new(project_id=p.id, query=q)
        save_saved_query(tmp_path, sq)

        sources = [
            Source.new(
                project_id=p.id,
                name="kept",
                source_id=sid_kept,
                language="en",
            ),
            Source.new(
                project_id=p.id,
                name="dropped",
                source_id=sid_dropped,
                language="fr",
            ),
        ]
        apps = [
            {"code_id": cid, "source_id": sid_kept},
            {"code_id": cid, "source_id": sid_dropped},
        ]
        _, matches = run_saved_query(
            tmp_path, p.id, sq.id, apps, sources=sources, record=False
        )
        assert [a["source_id"] for a in matches] == [sid_kept]


# --------------------------------------------------------------------------- #
# Multiple projects don't see each other's queries
# --------------------------------------------------------------------------- #


class TestProjectIsolation:
    def test_queries_scoped_to_project(self, tmp_path: Path) -> None:
        a = _saved_project(tmp_path, name="A")
        b = _saved_project(tmp_path, name="B")
        sq_a = SavedQuery.new(project_id=a.id, query=_query(a.id, name="A1"))
        sq_b = SavedQuery.new(project_id=b.id, query=_query(b.id, name="B1"))
        save_saved_query(tmp_path, sq_a)
        save_saved_query(tmp_path, sq_b)
        a_list = list_saved_queries(tmp_path, a.id)
        b_list = list_saved_queries(tmp_path, b.id)
        assert [s.name for s in a_list] == ["A1"]
        assert [s.name for s in b_list] == ["B1"]
