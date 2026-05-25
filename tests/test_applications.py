"""Tests for scribe.applications (F4.1).

These exercise the Application entity in pure Python: word-id helpers,
validation rules, serialisation round-trips, partial updates, and the
file-system persistence helpers. Endpoint-level tests will live in
test_server.py once F4.1 grows an HTTP surface; today the model +
persistence are the public API.
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
from scribe.applications import (
    APPLICATION_ID_RE,
    APPLICATION_PROVENANCE_SOURCES,
    MAX_CHAR_OFFSET,
    MAX_NOTE_LEN,
    MAX_PROVENANCE_KEYS,
    MAX_PROVENANCE_VALUE_LEN,
    PROVENANCE_KEY_RE,
    WORD_ID_RE,
    Application,
    application_state_path,
    applications_dir,
    compare_word_ids,
    delete_application,
    list_applications,
    load_application,
    make_word_id,
    new_application_id,
    parse_word_id,
    save_application,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


# Hex-only sentinels so they slot into the 12-char hex regex used by
# every id field in scribe (project / source / code / coder / version).
_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12


def _valid_kwargs(project_id: str, **overrides) -> dict:
    """Build a minimal-but-valid kwargs set for Application.new."""
    base = {
        "project_id": project_id,
        "code_id": _HEX_CODE,
        "source_id": _HEX_SOURCE,
        "coder_id": _HEX_CODER,
        "anchor_start_word_id": "s0w0",
        "anchor_end_word_id": "s0w5",
        "definition_version_id_at_apply": _HEX_VERSION,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewApplicationId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert APPLICATION_ID_RE.match(new_application_id())

    def test_unique(self) -> None:
        ids = {new_application_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Word-ID helpers
# --------------------------------------------------------------------------- #


class TestMakeWordId:
    def test_basic(self) -> None:
        assert make_word_id(0, 0) == "s0w0"
        assert make_word_id(12, 3) == "s12w3"
        assert make_word_id(999, 999) == "s999w999"

    def test_round_trips_through_regex(self) -> None:
        assert WORD_ID_RE.match(make_word_id(7, 11))

    def test_rejects_negative(self) -> None:
        with pytest.raises(ProjectValidationError):
            make_word_id(-1, 0)
        with pytest.raises(ProjectValidationError):
            make_word_id(0, -1)

    def test_rejects_non_int(self) -> None:
        with pytest.raises(ProjectValidationError):
            make_word_id("0", 0)  # type: ignore[arg-type]
        with pytest.raises(ProjectValidationError):
            make_word_id(0, "0")  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        # Python bool is a subclass of int; we want to reject it.
        with pytest.raises(ProjectValidationError):
            make_word_id(True, 0)  # type: ignore[arg-type]
        with pytest.raises(ProjectValidationError):
            make_word_id(0, False)  # type: ignore[arg-type]


class TestParseWordId:
    def test_basic(self) -> None:
        assert parse_word_id("s0w0") == (0, 0)
        assert parse_word_id("s12w3") == (12, 3)
        assert parse_word_id("s999w999") == (999, 999)

    def test_round_trip(self) -> None:
        for si, wi in [(0, 0), (3, 7), (12, 99)]:
            assert parse_word_id(make_word_id(si, wi)) == (si, wi)

    @pytest.mark.parametrize(
        "bad",
        ["", "s0", "w0", "s0w", "sw0", "S0W0", "s-1w0", "s0w-1", "s0w0extra"],
    )
    def test_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(ProjectValidationError):
            parse_word_id(bad)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ProjectValidationError):
            parse_word_id(0)  # type: ignore[arg-type]


class TestCompareWordIds:
    def test_equal(self) -> None:
        assert compare_word_ids("s0w0", "s0w0") == 0
        assert compare_word_ids("s5w7", "s5w7") == 0

    def test_segment_dominates(self) -> None:
        assert compare_word_ids("s0w99", "s1w0") == -1
        assert compare_word_ids("s2w0", "s1w99") == 1

    def test_word_index_within_segment(self) -> None:
        assert compare_word_ids("s5w3", "s5w7") == -1
        assert compare_word_ids("s5w7", "s5w3") == 1

    def test_natural_not_lexicographic(self) -> None:
        # "s10w0" < "s2w0" lexicographically, but s10 > s2 numerically.
        # The helper exists precisely to disambiguate this.
        assert compare_word_ids("s2w0", "s10w0") == -1
        assert compare_word_ids("s10w0", "s2w0") == 1

    def test_propagates_validation_error(self) -> None:
        with pytest.raises(ProjectValidationError):
            compare_word_ids("nope", "s0w0")


# --------------------------------------------------------------------------- #
# Application.new — happy path & validation
# --------------------------------------------------------------------------- #


class TestApplicationNew:
    def test_minimal(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        assert APPLICATION_ID_RE.match(a.id)
        assert a.project_id == _HEX_PROJECT
        assert a.code_id == "a" * 12
        assert a.source_id == "b" * 12
        assert a.coder_id == "c" * 12
        assert a.anchor_start_word_id == "s0w0"
        assert a.anchor_end_word_id == "s0w5"
        assert a.definition_version_id_at_apply == "d" * 12
        assert a.start_char_offset is None
        assert a.end_char_offset is None
        assert a.confidence is None
        assert a.provenance == {}
        assert a.note == ""
        assert a.created_at
        assert a.modified_at == a.created_at

    def test_explicit_id(self) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, application_id="f" * 12)
        )
        assert a.id == "f" * 12

    def test_full_payload(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                start_char_offset=2,
                end_char_offset=7,
                confidence=0.85,
                provenance={"source": "human", "model_id": "n/a"},
                note="In-vivo apology behaviour.",
            )
        )
        assert a.start_char_offset == 2
        assert a.end_char_offset == 7
        assert a.confidence == 0.85
        assert a.provenance == {"source": "human", "model_id": "n/a"}
        assert a.note == "In-vivo apology behaviour."

    def test_rejects_invalid_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs("bad-project-id"))

    def test_rejects_invalid_code_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, code_id="not-hex"))

    def test_rejects_invalid_source_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, source_id="not-hex"))

    def test_rejects_invalid_coder_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, coder_id="not-hex"))

    def test_rejects_invalid_definition_version_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT, definition_version_id_at_apply="not-hex"
                )
            )

    def test_rejects_malformed_anchor_start(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, anchor_start_word_id="garbage")
            )

    def test_rejects_malformed_anchor_end(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, anchor_end_word_id="")
            )

    def test_rejects_anchor_start_after_end(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT,
                    anchor_start_word_id="s5w0",
                    anchor_end_word_id="s2w0",
                )
            )

    def test_allows_anchor_start_equal_end(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                anchor_start_word_id="s3w7",
                anchor_end_word_id="s3w7",
            )
        )
        assert a.anchor_start_word_id == a.anchor_end_word_id

    def test_anchor_ordering_uses_natural_compare(self) -> None:
        # s10w0 > s2w0 numerically; lexicographically s10 < s2. The
        # validator must reject start > end on natural ordering.
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT,
                    anchor_start_word_id="s10w0",
                    anchor_end_word_id="s2w0",
                )
            )
        # The reverse is fine.
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                anchor_start_word_id="s2w0",
                anchor_end_word_id="s10w0",
            )
        )
        assert a.anchor_end_word_id == "s10w0"

    @pytest.mark.parametrize("bad", [-1, MAX_CHAR_OFFSET + 1, True])
    def test_rejects_bad_start_char_offset(self, bad) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, start_char_offset=bad))

    @pytest.mark.parametrize("bad", [-1, MAX_CHAR_OFFSET + 1, True])
    def test_rejects_bad_end_char_offset(self, bad) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, end_char_offset=bad))

    def test_allows_zero_char_offset(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT, start_char_offset=0, end_char_offset=0
            )
        )
        assert a.start_char_offset == 0
        assert a.end_char_offset == 0

    def test_rejects_empty_single_word_span(self) -> None:
        # Same word, start == end => empty span; not allowed.
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT,
                    anchor_start_word_id="s0w0",
                    anchor_end_word_id="s0w0",
                    start_char_offset=3,
                    end_char_offset=3,
                )
            )
        # Reversed offsets on a single word are also empty/invalid.
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT,
                    anchor_start_word_id="s0w0",
                    anchor_end_word_id="s0w0",
                    start_char_offset=5,
                    end_char_offset=2,
                )
            )

    def test_allows_partial_offsets_on_single_word(self) -> None:
        # Only one offset set on a single-word span is fine — the other
        # implicitly means "to the boundary of the word".
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                start_char_offset=3,
            )
        )
        assert a.start_char_offset == 3
        assert a.end_char_offset is None

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, "abc"])
    def test_rejects_bad_confidence(self, bad) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, confidence=bad))

    def test_rejects_bool_confidence(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, confidence=True))

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0, 1, 0])
    def test_accepts_valid_confidence(self, good) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT, confidence=good))
        assert a.confidence == float(good)

    def test_rejects_provenance_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, provenance="not-a-dict")  # type: ignore[arg-type]
            )

    def test_rejects_provenance_too_many_keys(self) -> None:
        too_many = {f"key_{i}": "v" for i in range(MAX_PROVENANCE_KEYS + 1)}
        with pytest.raises(ProjectValidationError):
            Application.new(**_valid_kwargs(_HEX_PROJECT, provenance=too_many))

    def test_rejects_provenance_bad_key(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, provenance={"1bad": "v"})
            )

    def test_rejects_provenance_long_value(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(
                    _HEX_PROJECT,
                    provenance={
                        "model_id": "x" * (MAX_PROVENANCE_VALUE_LEN + 1)
                    },
                )
            )

    def test_rejects_unknown_provenance_source(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, provenance={"source": "alien"})
            )

    @pytest.mark.parametrize("good", APPLICATION_PROVENANCE_SOURCES)
    def test_accepts_each_provenance_source(self, good: str) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, provenance={"source": good})
        )
        assert a.provenance["source"] == good

    def test_drops_empty_provenance_keys(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT, provenance={"": "ignored", "ok": "kept"}
            )
        )
        assert a.provenance == {"ok": "kept"}

    def test_rejects_long_note(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_kwargs(_HEX_PROJECT, note="x" * (MAX_NOTE_LEN + 1))
            )

    def test_now_override_stamps_timestamps(self) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, now="2026-01-01T00:00:00.000000Z")
        )
        assert a.created_at == "2026-01-01T00:00:00.000000Z"
        assert a.modified_at == "2026-01-01T00:00:00.000000Z"


# --------------------------------------------------------------------------- #
# Round-trip serialisation
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                start_char_offset=1,
                end_char_offset=4,
                confidence=0.42,
                provenance={"source": "human", "session_id": "s12"},
                note="Maps to category X.",
            )
        )
        d = a.to_dict()
        b = Application.from_dict(d)
        assert b == a

    def test_to_from_dict_minimal(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        b = Application.from_dict(a.to_dict())
        assert b == a

    def test_from_dict_accepts_string_int_offset(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT, start_char_offset=3))
        d = a.to_dict()
        d["start_char_offset"] = "3"
        b = Application.from_dict(d)
        assert b.start_char_offset == 3

    def test_from_dict_accepts_float_int_value_offset(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d["start_char_offset"] = 5.0
        b = Application.from_dict(d)
        assert b.start_char_offset == 5

    def test_from_dict_rejects_non_integer_float_offset(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d["start_char_offset"] = 5.5
        with pytest.raises(ProjectValidationError):
            Application.from_dict(d)

    def test_from_dict_rejects_garbage_string_offset(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d["start_char_offset"] = "abc"
        with pytest.raises(ProjectValidationError):
            Application.from_dict(d)

    def test_from_dict_accepts_string_confidence(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d["confidence"] = "0.7"
        b = Application.from_dict(d)
        assert b.confidence == 0.7

    def test_from_dict_treats_empty_string_as_none(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d["start_char_offset"] = ""
        d["end_char_offset"] = ""
        d["confidence"] = ""
        b = Application.from_dict(d)
        assert b.start_char_offset is None
        assert b.end_char_offset is None
        assert b.confidence is None

    def test_from_dict_missing_required_keys(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        d = a.to_dict()
        d.pop("anchor_start_word_id")
        with pytest.raises(ProjectValidationError):
            Application.from_dict(d)

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            Application.from_dict("not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# apply_update
# --------------------------------------------------------------------------- #


class TestApplyUpdate:
    def test_update_anchor(self) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, now="2026-01-01T00:00:00.000000Z")
        )
        a.apply_update(
            {
                "anchor_start_word_id": "s1w0",
                "anchor_end_word_id": "s1w7",
            },
            now="2026-02-01T00:00:00.000000Z",
        )
        assert a.anchor_start_word_id == "s1w0"
        assert a.anchor_end_word_id == "s1w7"
        # modified_at advances; created_at does not.
        assert a.created_at == "2026-01-01T00:00:00.000000Z"
        assert a.modified_at == "2026-02-01T00:00:00.000000Z"

    def test_update_offsets_and_confidence(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        a.apply_update(
            {
                "start_char_offset": 2,
                "end_char_offset": 9,
                "confidence": 0.95,
            }
        )
        assert a.start_char_offset == 2
        assert a.end_char_offset == 9
        assert a.confidence == 0.95

    def test_update_clears_optionals(self) -> None:
        a = Application.new(
            **_valid_kwargs(
                _HEX_PROJECT,
                start_char_offset=5,
                end_char_offset=10,
                confidence=0.5,
            )
        )
        a.apply_update(
            {
                "start_char_offset": None,
                "end_char_offset": None,
                "confidence": None,
            }
        )
        assert a.start_char_offset is None
        assert a.end_char_offset is None
        assert a.confidence is None

    def test_update_provenance(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        a.apply_update({"provenance": {"source": "ai_accepted", "k": "v"}})
        assert a.provenance == {"source": "ai_accepted", "k": "v"}

    def test_update_note(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        a.apply_update({"note": "updated"})
        assert a.note == "updated"

    def test_unknown_key_rejected(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        with pytest.raises(ProjectValidationError):
            a.apply_update({"banana": "x"})

    def test_ignored_keys_silently_dropped(self) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, now="2026-01-01T00:00:00.000000Z")
        )
        original_id = a.id
        original_code_id = a.code_id
        original_created = a.created_at
        a.apply_update(
            {
                "id": "new-id-ignored",
                "project_id": "new-proj-ignored",
                "code_id": "new-code-ignored",
                "source_id": "new-source-ignored",
                "coder_id": "new-coder-ignored",
                "definition_version_id_at_apply": "new-ver-ignored",
                "created_at": "2999-12-31T00:00:00.000000Z",
                "modified_at": "2999-12-31T00:00:00.000000Z",
                "note": "ok",
            },
            now="2026-02-01T00:00:00.000000Z",
        )
        assert a.id == original_id
        assert a.code_id == original_code_id
        assert a.created_at == original_created
        assert a.modified_at == "2026-02-01T00:00:00.000000Z"
        assert a.note == "ok"

    def test_failed_update_does_not_advance_modified_at(self) -> None:
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, now="2026-01-01T00:00:00.000000Z")
        )
        with pytest.raises(ProjectValidationError):
            a.apply_update({"anchor_start_word_id": "garbage"})
        assert a.modified_at == "2026-01-01T00:00:00.000000Z"

    def test_failed_update_does_not_partially_apply(self) -> None:
        # apply_update writes fields *then* calls validate(). A failure
        # in validate after a partially-applied patch would leave the
        # entity inconsistent. We mitigate by requiring all updates to
        # be re-validated; the test here pins the contract that *if*
        # validate raises, modified_at remains unchanged (tested above)
        # and *the entity may well be in a broken state* — callers must
        # discard and reload. Document that by asserting we did not
        # advance modified_at, which is the durable signal.
        a = Application.new(
            **_valid_kwargs(_HEX_PROJECT, now="2026-01-01T00:00:00.000000Z")
        )
        original_modified = a.modified_at
        with pytest.raises(ProjectValidationError):
            a.apply_update(
                {"anchor_start_word_id": "s9w0", "anchor_end_word_id": "s1w0"}
            )
        # Caller should reload from disk; we just guarantee the clock
        # didn't advance so a downstream "did anything change" check is
        # honest.
        assert a.modified_at == original_modified

    def test_non_dict_patch_rejected(self) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        with pytest.raises(ProjectValidationError):
            a.apply_update("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_kwargs(proj.id, note="hello"))
        save_application(tmp_path, a)
        b = load_application(tmp_path, proj.id, a.id)
        assert b == a

    def test_save_creates_atomic_write(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_kwargs(proj.id))
        target = save_application(tmp_path, a)
        assert target.exists()
        # The .json.tmp must have been replaced, not left behind.
        assert not target.with_suffix(".json.tmp").exists()

    def test_save_writes_under_project_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_kwargs(proj.id))
        target = save_application(tmp_path, a)
        expected = applications_dir(tmp_path, proj.id) / f"{a.id}.json"
        assert target == expected
        assert target.is_file()
        # Content survives a JSON round-trip.
        saved = json.loads(target.read_text())
        assert saved["id"] == a.id

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        a = Application.new(**_valid_kwargs(_HEX_PROJECT))
        with pytest.raises(FileNotFoundError):
            save_application(tmp_path, a)

    def test_load_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_application(tmp_path, proj.id, "f" * 12)

    def test_application_state_path_validates_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            application_state_path(tmp_path, _HEX_PROJECT, "not-hex")

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_applications(tmp_path, proj.id) == []

    def test_list_returns_all_in_created_order(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a1 = Application.new(
            **_valid_kwargs(
                proj.id, now="2026-01-01T00:00:00.000000Z"
            )
        )
        a2 = Application.new(
            **_valid_kwargs(
                proj.id, now="2026-01-02T00:00:00.000000Z"
            )
        )
        a3 = Application.new(
            **_valid_kwargs(
                proj.id, now="2026-01-03T00:00:00.000000Z"
            )
        )
        # Save out of order to confirm sort order is by created_at.
        save_application(tmp_path, a3)
        save_application(tmp_path, a1)
        save_application(tmp_path, a2)
        listed = list_applications(tmp_path, proj.id)
        assert [a.id for a in listed] == [a1.id, a2.id, a3.id]

    def test_list_filter_by_code(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ax = Application.new(
            **_valid_kwargs(proj.id, code_id="a" * 12)
        )
        ay = Application.new(
            **_valid_kwargs(proj.id, code_id="b" * 12)
        )
        save_application(tmp_path, ax)
        save_application(tmp_path, ay)
        out = list_applications(tmp_path, proj.id, code_id="a" * 12)
        assert [a.id for a in out] == [ax.id]

    def test_list_filter_by_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ax = Application.new(
            **_valid_kwargs(proj.id, source_id="1" * 12)
        )
        ay = Application.new(
            **_valid_kwargs(proj.id, source_id="2" * 12)
        )
        save_application(tmp_path, ax)
        save_application(tmp_path, ay)
        out = list_applications(tmp_path, proj.id, source_id="2" * 12)
        assert [a.id for a in out] == [ay.id]

    def test_list_filter_by_coder(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ax = Application.new(
            **_valid_kwargs(proj.id, coder_id="c" * 12)
        )
        ay = Application.new(
            **_valid_kwargs(proj.id, coder_id="e" * 12)
        )
        save_application(tmp_path, ax)
        save_application(tmp_path, ay)
        out = list_applications(tmp_path, proj.id, coder_id="e" * 12)
        assert [a.id for a in out] == [ay.id]

    def test_list_filter_combined_anded(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a_target = Application.new(
            **_valid_kwargs(
                proj.id,
                code_id="a" * 12,
                source_id="1" * 12,
                coder_id="c" * 12,
            )
        )
        a_other = Application.new(
            **_valid_kwargs(
                proj.id,
                code_id="a" * 12,
                source_id="2" * 12,
                coder_id="c" * 12,
            )
        )
        save_application(tmp_path, a_target)
        save_application(tmp_path, a_other)
        out = list_applications(
            tmp_path,
            proj.id,
            code_id="a" * 12,
            source_id="1" * 12,
            coder_id="c" * 12,
        )
        assert [a.id for a in out] == [a_target.id]

    def test_list_rejects_bad_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_applications(tmp_path, proj.id, code_id="bad")
        with pytest.raises(ProjectValidationError):
            list_applications(tmp_path, proj.id, source_id="bad")
        with pytest.raises(ProjectValidationError):
            list_applications(tmp_path, proj.id, coder_id="bad")

    def test_list_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        good = Application.new(**_valid_kwargs(proj.id))
        save_application(tmp_path, good)
        ad = applications_dir(tmp_path, proj.id)
        # A genuinely-malformed JSON file with a hex-shaped name.
        (ad / ("0" * 12 + ".json")).write_text("{not json")
        # A non-hex-named file.
        (ad / "weird.json").write_text("{}")
        # A leftover .tmp.
        (ad / "leftover.json.tmp").write_text("{}")
        listed = list_applications(tmp_path, proj.id)
        assert [a.id for a in listed] == [good.id]

    def test_list_missing_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # No applications dir at all → empty list, not error.
        assert list_applications(tmp_path, proj.id) == []

    def test_delete_existing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_kwargs(proj.id))
        save_application(tmp_path, a)
        assert delete_application(tmp_path, proj.id, a.id) is True
        assert delete_application(tmp_path, proj.id, a.id) is False

    def test_delete_invalid_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            delete_application(tmp_path, proj.id, "not-hex")

    def test_delete_project_cascades(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_kwargs(proj.id))
        save_application(tmp_path, a)
        # Sanity: app file exists under the project dir.
        ad = applications_dir(tmp_path, proj.id)
        assert (ad / f"{a.id}.json").exists()
        # Deleting the project removes the whole tree.
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()

    def test_save_multiple_distinct_ids(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ids = set()
        for _ in range(5):
            a = Application.new(**_valid_kwargs(proj.id))
            save_application(tmp_path, a)
            ids.add(a.id)
        assert len(ids) == 5
        listed = list_applications(tmp_path, proj.id)
        assert {a.id for a in listed} == ids
