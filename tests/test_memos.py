"""Tests for scribe.memos (F5.1).

These exercise the Memo entity + MemoLink helper in pure Python:
validation rules, multi-target link semantics, serialisation round-
trips, partial updates, and the file-system persistence helpers.
Endpoint-level tests will live in test_server.py once F5.x grows an
HTTP surface; today the model + persistence are the public API.
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
    delete_project,
)
from scribe.memos import (
    LINK_ROLE_RE,
    MAX_BODY_LEN,
    MAX_LINK_ROLE_LEN,
    MAX_LINKS,
    MAX_PROVENANCE_KEYS,
    MAX_PROVENANCE_VALUE_LEN,
    MAX_TAGS,
    MAX_TAG_LEN,
    MAX_TITLE_LEN,
    MEMO_BODY_FORMATS,
    MEMO_ID_RE,
    MEMO_LINK_TARGET_TYPES,
    MEMO_PROVENANCE_SOURCES,
    MEMO_TYPES,
    PROVENANCE_KEY_RE,
    TAG_RE,
    TARGET_ID_RE,
    Memo,
    MemoLink,
    count_memos,
    delete_memo,
    list_memos,
    load_memo,
    memo_state_path,
    memos_dir,
    new_memo_id,
    save_memo,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


# Hex-only sentinels so they slot into the 12-char hex regex used by
# every id field in scribe (project / source / code / coder / version /
# memo / application).
_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_APPLICATION = "d" * 12
_HEX_PARTICIPANT = "e" * 12
_HEX_MEMO = "f" * 12


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewMemoId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert MEMO_ID_RE.match(new_memo_id())

    def test_unique(self) -> None:
        ids = {new_memo_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# MemoLink — validation + round-trip
# --------------------------------------------------------------------------- #


class TestMemoLinkValidate:
    def test_valid_link(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE, role="exemplifies")
        link.validate()
        assert link.target_type == "code"
        assert link.target_id == _HEX_CODE
        assert link.role == "exemplifies"

    def test_role_is_optional(self) -> None:
        link = MemoLink(target_type="source", target_id=_HEX_SOURCE)
        link.validate()
        assert link.role == ""

    def test_role_is_trimmed(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE, role="  notes  ")
        link.validate()
        assert link.role == "notes"

    def test_rejects_unknown_target_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink(target_type="cosmology", target_id=_HEX_CODE).validate()

    def test_rejects_bad_target_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink(target_type="code", target_id="not-hex").validate()

    def test_rejects_role_too_long(self) -> None:
        bad = "a" * (MAX_LINK_ROLE_LEN + 1)
        with pytest.raises(ProjectValidationError):
            MemoLink(target_type="code", target_id=_HEX_CODE, role=bad).validate()

    def test_rejects_role_with_punctuation(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink(
                target_type="code", target_id=_HEX_CODE, role="not/allowed"
            ).validate()

    def test_rejects_role_starting_with_digit(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink(
                target_type="code", target_id=_HEX_CODE, role="1bad"
            ).validate()

    def test_link_role_re_constants_align(self) -> None:
        # Sanity-check that the exposed regex matches what MemoLink validates.
        assert LINK_ROLE_RE.match("ok-role")
        assert not LINK_ROLE_RE.match("1starts-with-digit")


class TestMemoLinkSerialisation:
    def test_to_dict_omits_empty_role(self) -> None:
        link = MemoLink(target_type="code", target_id=_HEX_CODE)
        d = link.to_dict()
        assert d == {"target_type": "code", "target_id": _HEX_CODE}

    def test_to_dict_includes_role(self) -> None:
        link = MemoLink(target_type="source", target_id=_HEX_SOURCE, role="raises")
        d = link.to_dict()
        assert d == {
            "target_type": "source",
            "target_id": _HEX_SOURCE,
            "role": "raises",
        }

    def test_from_dict_round_trip(self) -> None:
        original = MemoLink(
            target_type="application", target_id=_HEX_APPLICATION, role="counters"
        )
        round_tripped = MemoLink.from_dict(original.to_dict())
        assert round_tripped == original

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_requires_target_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink.from_dict({"target_id": _HEX_CODE})

    def test_from_dict_requires_target_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            MemoLink.from_dict({"target_type": "code"})


# --------------------------------------------------------------------------- #
# Memo.new — defaults + validation entry point
# --------------------------------------------------------------------------- #


class TestMemoNew:
    def test_defaults_to_free_type(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        assert m.type == "free"
        assert m.body_format == "markdown"
        assert m.title == ""
        assert m.body == ""
        assert m.author_coder_id is None
        assert m.links == []
        assert m.tags == []
        assert m.provenance == {}

    def test_stamps_timestamps(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        assert m.created_at  # ISO string non-empty
        assert m.modified_at == m.created_at

    def test_now_override(self) -> None:
        ts = "2026-01-02T03:04:05Z"
        m = Memo.new(project_id=_HEX_PROJECT, now=ts)
        assert m.created_at == ts
        assert m.modified_at == ts

    def test_id_override(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, memo_id=_HEX_MEMO)
        assert m.id == _HEX_MEMO

    def test_links_accepts_dicts_and_memo_links(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE),
                {"target_type": "source", "target_id": _HEX_SOURCE},
            ],
        )
        assert len(m.links) == 2
        assert m.links[0].target_type == "code"
        assert m.links[1].target_type == "source"

    def test_links_rejects_other(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, links=["not-a-link"])  # type: ignore[list-item]

    def test_full_payload(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            type="theoretical",
            title="Becoming visible",
            body="The category 'becoming visible' looks like…",
            body_format="markdown",
            author_coder_id=_HEX_CODER,
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
            tags=["category", "visibility"],
            provenance={"source": "human"},
        )
        assert m.type == "theoretical"
        assert m.author_coder_id == _HEX_CODER
        assert m.tags == ["category", "visibility"]
        assert m.provenance == {"source": "human"}


# --------------------------------------------------------------------------- #
# Memo.validate — types, fields, limits
# --------------------------------------------------------------------------- #


class TestMemoValidateIds:
    def test_rejects_bad_memo_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, memo_id="not-hex")

    def test_rejects_bad_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id="not-hex")

    def test_rejects_bad_author_coder_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, author_coder_id="not-hex")


class TestMemoValidateType:
    def test_each_known_type_accepted(self) -> None:
        for t in MEMO_TYPES:
            m = Memo.new(project_id=_HEX_PROJECT, type=t)
            assert m.type == t

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, type="cosmology")


class TestMemoValidateBodyFormat:
    def test_each_known_format_accepted(self) -> None:
        for f in MEMO_BODY_FORMATS:
            m = Memo.new(project_id=_HEX_PROJECT, body_format=f)
            assert m.body_format == f

    def test_rejects_unknown_format(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, body_format="org")


class TestMemoValidateTitleAndBody:
    def test_title_is_trimmed(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, title="  hello  ")
        assert m.title == "hello"

    def test_title_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, title="x" * (MAX_TITLE_LEN + 1))

    def test_body_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, body="x" * (MAX_BODY_LEN + 1))

    def test_body_at_limit(self) -> None:
        # Exactly the boundary should still be accepted.
        m = Memo.new(project_id=_HEX_PROJECT, body="x" * MAX_BODY_LEN)
        assert len(m.body) == MAX_BODY_LEN


class TestMemoValidateTags:
    def test_dedupes_preserving_order(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT, tags=["alpha", "beta", "alpha", "gamma"]
        )
        assert m.tags == ["alpha", "beta", "gamma"]

    def test_drops_empty_tags(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT, tags=["alpha", "", "  ", "beta"]
        )
        assert m.tags == ["alpha", "beta"]

    def test_rejects_too_many_tags(self) -> None:
        many = [f"t{i}" for i in range(MAX_TAGS + 1)]
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, tags=many)

    def test_rejects_tag_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, tags=["x" * (MAX_TAG_LEN + 1)])

    def test_rejects_tag_with_punctuation(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, tags=["bad/tag"])

    def test_rejects_tag_starting_with_digit(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, tags=["1tag"])

    def test_tag_re_sanity(self) -> None:
        assert TAG_RE.match("ok-tag")
        assert not TAG_RE.match("1bad")


class TestMemoValidateLinks:
    def test_dedupes_links_by_full_key(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE, role="primary"),
                MemoLink(target_type="code", target_id=_HEX_CODE, role="primary"),
                MemoLink(target_type="code", target_id=_HEX_CODE, role="contrast"),
            ],
        )
        # Same (type, id, role) is a duplicate; differing role is kept.
        assert len(m.links) == 2

    def test_rejects_too_many_links(self) -> None:
        many = [
            MemoLink(target_type="code", target_id=f"{i:012x}")
            for i in range(MAX_LINKS + 1)
        ]
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, links=many)

    def test_rejects_invalid_link_in_list(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(
                project_id=_HEX_PROJECT,
                links=[MemoLink(target_type="bogus", target_id=_HEX_CODE)],
            )

    def test_rejects_non_link_in_list(self) -> None:
        # If something other than a MemoLink survives into validate(),
        # it should raise (defence in depth — Memo.new already checks).
        m = Memo(
            id=_HEX_MEMO,
            project_id=_HEX_PROJECT,
            links=["not-a-link"],  # type: ignore[list-item]
        )
        with pytest.raises(ProjectValidationError):
            m.validate()


class TestMemoValidateProvenance:
    def test_accepts_known_source(self) -> None:
        for src in MEMO_PROVENANCE_SOURCES:
            m = Memo.new(project_id=_HEX_PROJECT, provenance={"source": src})
            assert m.provenance["source"] == src

    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(
                project_id=_HEX_PROJECT, provenance={"source": "telepathy"}
            )

    def test_drops_empty_keys(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT, provenance={"   ": "v", "model": "phi-4"}
        )
        assert m.provenance == {"model": "phi-4"}

    def test_rejects_bad_provenance_key(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(
                project_id=_HEX_PROJECT, provenance={"1bad": "value"}
            )

    def test_rejects_provenance_value_too_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.new(
                project_id=_HEX_PROJECT,
                provenance={"model": "x" * (MAX_PROVENANCE_VALUE_LEN + 1)},
            )

    def test_rejects_too_many_provenance_keys(self) -> None:
        prov = {f"k{i}": "v" for i in range(MAX_PROVENANCE_KEYS + 1)}
        with pytest.raises(ProjectValidationError):
            Memo.new(project_id=_HEX_PROJECT, provenance=prov)

    def test_rejects_non_dict_provenance(self) -> None:
        m = Memo(
            id=_HEX_MEMO, project_id=_HEX_PROJECT, provenance=[]  # type: ignore[arg-type]
        )
        with pytest.raises(ProjectValidationError):
            m.validate()

    def test_provenance_key_re_sanity(self) -> None:
        assert PROVENANCE_KEY_RE.match("model-id")
        assert not PROVENANCE_KEY_RE.match("1bad")


# --------------------------------------------------------------------------- #
# Multi-target convenience helpers
# --------------------------------------------------------------------------- #


class TestMemoLinkConvenience:
    def test_has_link_to(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE),
                MemoLink(target_type="source", target_id=_HEX_SOURCE),
            ],
        )
        assert m.has_link_to("code", _HEX_CODE)
        assert m.has_link_to("source", _HEX_SOURCE)
        assert not m.has_link_to("code", _HEX_SOURCE)
        assert not m.has_link_to("application", _HEX_CODE)

    def test_link_target_ids_filtered(self) -> None:
        other_code = "1" * 12
        m = Memo.new(
            project_id=_HEX_PROJECT,
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE),
                MemoLink(target_type="code", target_id=other_code),
                MemoLink(target_type="source", target_id=_HEX_SOURCE),
            ],
        )
        assert m.link_target_ids("code") == [_HEX_CODE, other_code]
        assert m.link_target_ids("source") == [_HEX_SOURCE]
        assert m.link_target_ids("application") == []


# --------------------------------------------------------------------------- #
# Serialisation round-trip
# --------------------------------------------------------------------------- #


class TestMemoSerialisation:
    def test_round_trip(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            type="reflexive",
            title="Positionality on consent",
            body="I noticed I felt protective of P3 when…",
            body_format="markdown",
            author_coder_id=_HEX_CODER,
            links=[
                MemoLink(target_type="participant", target_id=_HEX_PARTICIPANT),
                MemoLink(target_type="source", target_id=_HEX_SOURCE),
            ],
            tags=["reflexivity", "consent"],
            provenance={"source": "human"},
        )
        d = m.to_dict()
        # to_dict uses MemoLink.to_dict (omits empty role) so links are dicts.
        assert isinstance(d["links"], list)
        assert d["links"][0] == {
            "target_type": "participant",
            "target_id": _HEX_PARTICIPANT,
        }
        # JSON-encodable.
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        m2 = Memo.from_dict(decoded)
        assert m2 == m

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_requires_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.from_dict({"project_id": _HEX_PROJECT})

    def test_from_dict_requires_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.from_dict({"id": _HEX_MEMO})

    def test_from_dict_rejects_non_list_links(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.from_dict(
                {"id": _HEX_MEMO, "project_id": _HEX_PROJECT, "links": {}}
            )

    def test_from_dict_rejects_non_list_tags(self) -> None:
        with pytest.raises(ProjectValidationError):
            Memo.from_dict(
                {"id": _HEX_MEMO, "project_id": _HEX_PROJECT, "tags": {}}
            )

    def test_from_dict_defaults(self) -> None:
        m = Memo.from_dict({"id": _HEX_MEMO, "project_id": _HEX_PROJECT})
        assert m.type == "free"
        assert m.body_format == "markdown"
        assert m.title == ""
        assert m.body == ""
        assert m.links == []
        assert m.tags == []


# --------------------------------------------------------------------------- #
# Memo.apply_update — partial mutations
# --------------------------------------------------------------------------- #


class TestMemoApplyUpdate:
    def test_updates_basic_fields(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, now="2026-01-01T00:00:00Z")
        m.apply_update(
            {
                "type": "theoretical",
                "title": "  trimmed  ",
                "body": "new body",
                "body_format": "plain",
            },
            now="2026-01-02T00:00:00Z",
        )
        assert m.type == "theoretical"
        assert m.title == "trimmed"
        assert m.body == "new body"
        assert m.body_format == "plain"
        assert m.modified_at == "2026-01-02T00:00:00Z"
        assert m.created_at == "2026-01-01T00:00:00Z"

    def test_updates_links_replaces_list(self) -> None:
        m = Memo.new(
            project_id=_HEX_PROJECT,
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        m.apply_update(
            {
                "links": [
                    {"target_type": "source", "target_id": _HEX_SOURCE},
                    MemoLink(target_type="application", target_id=_HEX_APPLICATION),
                ]
            }
        )
        assert len(m.links) == 2
        assert m.links[0].target_type == "source"
        assert m.links[1].target_type == "application"

    def test_updates_tags_replaces_list(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, tags=["old"])
        m.apply_update({"tags": ["new", "newer"]})
        assert m.tags == ["new", "newer"]

    def test_updates_author(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        m.apply_update({"author_coder_id": _HEX_CODER})
        assert m.author_coder_id == _HEX_CODER
        m.apply_update({"author_coder_id": ""})
        assert m.author_coder_id is None

    def test_updates_provenance(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        m.apply_update({"provenance": {"source": "ai_drafted", "model": "phi-4"}})
        assert m.provenance == {"source": "ai_drafted", "model": "phi-4"}

    def test_rejects_non_dict_patch(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update("not a dict")  # type: ignore[arg-type]

    def test_rejects_unknown_field(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update({"weight": 2})

    def test_ignores_managed_keys(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        original_id = m.id
        m.apply_update(
            {
                "id": "ffffffffffff",
                "project_id": "ffffffffffff",
                "created_at": "2050-01-01T00:00:00Z",
                "modified_at": "2050-01-01T00:00:00Z",
                "title": "ok",
            }
        )
        assert m.id == original_id
        assert m.title == "ok"

    def test_failed_validation_does_not_advance_clock(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, now="2026-01-01T00:00:00Z")
        with pytest.raises(ProjectValidationError):
            m.apply_update({"type": "cosmology"})
        # modified_at unchanged because validate() raised before stamping.
        assert m.modified_at == "2026-01-01T00:00:00Z"

    def test_links_must_be_list(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update({"links": {"not": "a list"}})

    def test_links_entries_must_be_link_or_dict(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update({"links": ["string"]})

    def test_tags_must_be_list(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update({"tags": "tag-string"})

    def test_provenance_must_be_dict(self) -> None:
        m = Memo.new(project_id=_HEX_PROJECT)
        with pytest.raises(ProjectValidationError):
            m.apply_update({"provenance": ["a", "b"]})


# --------------------------------------------------------------------------- #
# Persistence — save / load / list / delete
# --------------------------------------------------------------------------- #


class TestSaveLoadMemo:
    def test_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(
            project_id=proj.id,
            type="theoretical",
            title="On categories",
            body="Categories emerge by…",
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        path = save_memo(tmp_path, m)
        assert path.exists()
        loaded = load_memo(tmp_path, proj.id, m.id)
        assert loaded == m

    def test_save_creates_memos_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(project_id=proj.id, body="hi")
        save_memo(tmp_path, m)
        assert memos_dir(tmp_path, proj.id).is_dir()

    def test_save_writes_pretty_json(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(project_id=proj.id, body="é one café")
        path = save_memo(tmp_path, m)
        text = path.read_text(encoding="utf-8")
        # ensure_ascii=False keeps the non-ASCII visible.
        assert "café" in text

    def test_save_atomic_no_lingering_tmp(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(project_id=proj.id, body="hi")
        save_memo(tmp_path, m)
        tmps = list(memos_dir(tmp_path, proj.id).glob("*.tmp"))
        assert tmps == []

    def test_save_requires_existing_project(self, tmp_path: Path) -> None:
        m = Memo.new(project_id=_HEX_PROJECT, body="hi")
        with pytest.raises(FileNotFoundError):
            save_memo(tmp_path, m)

    def test_save_invokes_validate(self, tmp_path: Path) -> None:
        # Construct a Memo bypassing .new() so validate runs at save time.
        proj = _saved_project(tmp_path)
        m = Memo(id=_HEX_MEMO, project_id=proj.id, type="cosmology")
        with pytest.raises(ProjectValidationError):
            save_memo(tmp_path, m)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_memo(tmp_path, proj.id, _HEX_MEMO)

    def test_memo_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            memo_state_path(tmp_path, _HEX_PROJECT, "not-hex")


class TestListMemos:
    def test_empty_when_dir_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_memos(tmp_path, proj.id) == []

    def test_lists_in_created_at_order(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Memo.new(
            project_id=proj.id, body="first", now="2026-01-01T00:00:00Z"
        )
        b = Memo.new(
            project_id=proj.id, body="second", now="2026-01-02T00:00:00Z"
        )
        c = Memo.new(
            project_id=proj.id, body="third", now="2026-01-03T00:00:00Z"
        )
        # Save out-of-order to be sure list_memos sorts, not the FS.
        save_memo(tmp_path, c)
        save_memo(tmp_path, a)
        save_memo(tmp_path, b)
        ids = [m.id for m in list_memos(tmp_path, proj.id)]
        assert ids == [a.id, b.id, c.id]

    def test_filter_by_type(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ref = Memo.new(project_id=proj.id, type="reflexive")
        the = Memo.new(project_id=proj.id, type="theoretical")
        free = Memo.new(project_id=proj.id, type="free")
        for m in (ref, the, free):
            save_memo(tmp_path, m)
        got = list_memos(tmp_path, proj.id, type="reflexive")
        assert [m.id for m in got] == [ref.id]

    def test_filter_by_target_type(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m_code = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        m_source = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="source", target_id=_HEX_SOURCE)],
        )
        m_both = Memo.new(
            project_id=proj.id,
            links=[
                MemoLink(target_type="code", target_id=_HEX_CODE),
                MemoLink(target_type="source", target_id=_HEX_SOURCE),
            ],
        )
        for m in (m_code, m_source, m_both):
            save_memo(tmp_path, m)
        got_codes = {m.id for m in list_memos(tmp_path, proj.id, target_type="code")}
        assert got_codes == {m_code.id, m_both.id}
        got_srcs = {m.id for m in list_memos(tmp_path, proj.id, target_type="source")}
        assert got_srcs == {m_source.id, m_both.id}

    def test_filter_by_target_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        other_code = "1" * 12
        m_a = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        m_b = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="code", target_id=other_code)],
        )
        save_memo(tmp_path, m_a)
        save_memo(tmp_path, m_b)
        got = list_memos(
            tmp_path, proj.id, target_type="code", target_id=_HEX_CODE
        )
        assert [m.id for m in got] == [m_a.id]

    def test_filter_by_target_id_alone(self, tmp_path: Path) -> None:
        # Without target_type, filter on target_id matches across types.
        proj = _saved_project(tmp_path)
        same_id = _HEX_CODE  # coincidental id used for two entity types
        m_c = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="code", target_id=same_id)],
        )
        m_p = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="participant", target_id=same_id)],
        )
        m_other = Memo.new(
            project_id=proj.id,
            links=[MemoLink(target_type="source", target_id=_HEX_SOURCE)],
        )
        for m in (m_c, m_p, m_other):
            save_memo(tmp_path, m)
        got = {m.id for m in list_memos(tmp_path, proj.id, target_id=same_id)}
        assert got == {m_c.id, m_p.id}

    def test_filter_by_author(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        other_coder = "9" * 12
        a = Memo.new(project_id=proj.id, author_coder_id=_HEX_CODER)
        b = Memo.new(project_id=proj.id, author_coder_id=other_coder)
        save_memo(tmp_path, a)
        save_memo(tmp_path, b)
        got = list_memos(tmp_path, proj.id, author_coder_id=_HEX_CODER)
        assert [m.id for m in got] == [a.id]

    def test_filter_by_tag(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Memo.new(project_id=proj.id, tags=["category", "visibility"])
        b = Memo.new(project_id=proj.id, tags=["methodology"])
        save_memo(tmp_path, a)
        save_memo(tmp_path, b)
        got = list_memos(tmp_path, proj.id, tag="category")
        assert [m.id for m in got] == [a.id]

    def test_filter_combines_with_and(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        good = Memo.new(
            project_id=proj.id,
            type="theoretical",
            tags=["category"],
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        wrong_type = Memo.new(
            project_id=proj.id,
            type="reflexive",
            tags=["category"],
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        wrong_tag = Memo.new(
            project_id=proj.id,
            type="theoretical",
            tags=["methodology"],
            links=[MemoLink(target_type="code", target_id=_HEX_CODE)],
        )
        for m in (good, wrong_type, wrong_tag):
            save_memo(tmp_path, m)
        got = list_memos(
            tmp_path,
            proj.id,
            type="theoretical",
            tag="category",
            target_type="code",
        )
        assert [m.id for m in got] == [good.id]

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ok = Memo.new(project_id=proj.id, body="ok")
        save_memo(tmp_path, ok)
        # Invalid id stem.
        bad_path = memos_dir(tmp_path, proj.id) / "not-hex.json"
        bad_path.write_text("{}")
        # Bad JSON.
        another_bad = memos_dir(tmp_path, proj.id) / ("9" * 12 + ".json")
        another_bad.write_text("not json")
        # Valid id, invalid payload.
        invalid_payload = memos_dir(tmp_path, proj.id) / ("8" * 12 + ".json")
        invalid_payload.write_text(json.dumps({"id": "8" * 12}))
        got = list_memos(tmp_path, proj.id)
        assert [m.id for m in got] == [ok.id]

    def test_skips_tmp_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ok = Memo.new(project_id=proj.id, body="ok")
        save_memo(tmp_path, ok)
        # A leftover tmp shouldn't be parsed as a real memo.
        leftover = memos_dir(tmp_path, proj.id) / (_HEX_MEMO + ".json.tmp")
        leftover.write_text("garbage")
        got = list_memos(tmp_path, proj.id)
        assert [m.id for m in got] == [ok.id]

    def test_rejects_bad_filter_values(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, type="cosmology")
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, target_type="bogus")
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, target_id="not-hex")
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, author_coder_id="not-hex")
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, tag="")
        with pytest.raises(ProjectValidationError):
            list_memos(tmp_path, proj.id, tag="   ")


class TestDeleteMemo:
    def test_returns_true_when_deleted(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        m = Memo.new(project_id=proj.id, body="bye")
        save_memo(tmp_path, m)
        assert delete_memo(tmp_path, proj.id, m.id) is True
        assert not memo_state_path(tmp_path, proj.id, m.id).exists()

    def test_returns_false_when_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert delete_memo(tmp_path, proj.id, _HEX_MEMO) is False

    def test_delete_project_removes_memos_dir(self, tmp_path: Path) -> None:
        # Sanity-check the "delete_project cleans up for free" promise.
        proj = _saved_project(tmp_path)
        m = Memo.new(project_id=proj.id, body="bye")
        save_memo(tmp_path, m)
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()


class TestCountMemos:
    def test_count_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert count_memos(tmp_path, proj.id) == 0

    def test_count_total_and_by_type(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        for t in ("theoretical", "theoretical", "reflexive", "free"):
            save_memo(tmp_path, Memo.new(project_id=proj.id, type=t))
        assert count_memos(tmp_path, proj.id) == 4
        assert count_memos(tmp_path, proj.id, type="theoretical") == 2
        assert count_memos(tmp_path, proj.id, type="reflexive") == 1
        assert count_memos(tmp_path, proj.id, type="quote") == 0


# --------------------------------------------------------------------------- #
# Constants — sanity-check exports
# --------------------------------------------------------------------------- #


class TestConstants:
    def test_memo_types_complete(self) -> None:
        # PLANNING.md F5.1 lists exactly these types (we add ``free``
        # as the unclassified default).
        assert set(MEMO_TYPES) == {
            "code",
            "theoretical",
            "methodological",
            "reflexive",
            "quote",
            "source",
            "project",
            "free",
        }

    def test_memo_link_target_types_cover_all_entities(self) -> None:
        # We can link to every entity that exists today.
        assert "code" in MEMO_LINK_TARGET_TYPES
        assert "source" in MEMO_LINK_TARGET_TYPES
        assert "application" in MEMO_LINK_TARGET_TYPES
        assert "participant" in MEMO_LINK_TARGET_TYPES
        assert "coder" in MEMO_LINK_TARGET_TYPES
        assert "project" in MEMO_LINK_TARGET_TYPES
        # And memo-to-memo edges (for F5.3).
        assert "memo" in MEMO_LINK_TARGET_TYPES

    def test_target_id_re_matches_other_id_shapes(self) -> None:
        assert TARGET_ID_RE.match(_HEX_CODE)
        assert TARGET_ID_RE.match(_HEX_PARTICIPANT)
        assert not TARGET_ID_RE.match("not-hex")
        assert not TARGET_ID_RE.match("FFFFFFFFFFFF")  # uppercase is not allowed

    def test_provenance_sources_includes_human(self) -> None:
        assert "human" in MEMO_PROVENANCE_SOURCES
        assert "ai_drafted" in MEMO_PROVENANCE_SOURCES
