"""Tests for scribe.sampling_log (F1.4).

Pure-Python coverage of the SamplingEntry data model and the append-
only JSONL log: validation, round-trips, the action / decision-type
vocabularies, optional source/participant linkage, and the on-disk
helpers (``append_sampling_entry``, ``read_sampling_log``,
``count_sampling_entries``).

Endpoint-level tests would live in test_server.py once F1.4 grows an
HTTP surface; today the model + persistence are the public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.sampling_log import (
    MAX_NOTES_LEN,
    MAX_RATIONALE_LEN,
    MAX_TARGET_CATEGORY_LEN,
    SAMPLING_ACTIONS,
    SAMPLING_DECISION_TYPES,
    SAMPLING_ENTRY_ID_RE,
    SamplingEntry,
    append_sampling_entry,
    count_sampling_entries,
    new_sampling_entry_id,
    read_sampling_log,
    sampling_log_path,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


class TestNewSamplingEntryId:
    def test_shape_matches_regex(self) -> None:
        for _ in range(10):
            assert SAMPLING_ENTRY_ID_RE.match(new_sampling_entry_id())

    def test_unique(self) -> None:
        ids = {new_sampling_entry_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_actions_include_added_planned_removed_noted(self) -> None:
        # The four documented actions must all be present; this guards
        # against accidental rename of a value the on-disk format depends on.
        for action in ("added", "planned", "removed", "noted"):
            assert action in SAMPLING_ACTIONS

    def test_decision_types_include_empty_and_theoretical(self) -> None:
        assert "" in SAMPLING_DECISION_TYPES
        assert "theoretical" in SAMPLING_DECISION_TYPES

    def test_decision_types_cover_common_strategies(self) -> None:
        # Spot-check that the standard methodological vocabulary is
        # represented; not exhaustive, but enough to catch a regression.
        for dt in (
            "purposive",
            "convenience",
            "snowball",
            "criterion",
            "negative_case",
            "maximum_variation",
            "other",
        ):
            assert dt in SAMPLING_DECISION_TYPES


# --------------------------------------------------------------------------- #
# SamplingEntry.new — defaults + validation
# --------------------------------------------------------------------------- #


class TestSamplingEntryNew:
    def test_minimal(self) -> None:
        e = SamplingEntry.new(project_id="aaaaaaaaaaaa")
        assert e.project_id == "aaaaaaaaaaaa"
        assert e.id and SAMPLING_ENTRY_ID_RE.match(e.id)
        assert e.action == "added"
        assert e.decision_type == ""
        assert e.source_id is None
        assert e.participant_id is None
        assert e.target_category == ""
        assert e.rationale == ""
        assert e.notes == ""
        assert e.created_at  # non-empty timestamp

    def test_full_payload(self) -> None:
        e = SamplingEntry.new(
            project_id="aaaaaaaaaaaa",
            action="added",
            decision_type="theoretical",
            source_id="0123456789ab",
            participant_id="fedcba987654",
            target_category="managing disclosure",
            rationale=(
                "P03 hinted at variation by age; recruited an older "
                "participant to test category boundary."
            ),
            notes="Followed up via clinic referral.",
        )
        assert e.action == "added"
        assert e.decision_type == "theoretical"
        assert e.source_id == "0123456789ab"
        assert e.participant_id == "fedcba987654"
        assert e.target_category == "managing disclosure"

    def test_strips_target_category_whitespace(self) -> None:
        e = SamplingEntry.new(
            project_id="aaaaaaaaaaaa",
            target_category="  managing disclosure  ",
        )
        assert e.target_category == "managing disclosure"

    def test_blank_source_id_normalised_to_none(self) -> None:
        # Both "" and None should land as None on disk so the JSON
        # form is unambiguous.
        e1 = SamplingEntry.new(project_id="aaaaaaaaaaaa", source_id="")
        e2 = SamplingEntry.new(project_id="aaaaaaaaaaaa", source_id=None)
        assert e1.source_id is None
        assert e2.source_id is None

    def test_blank_participant_id_normalised_to_none(self) -> None:
        e1 = SamplingEntry.new(
            project_id="aaaaaaaaaaaa", participant_id=""
        )
        e2 = SamplingEntry.new(
            project_id="aaaaaaaaaaaa", participant_id=None
        )
        assert e1.participant_id is None
        assert e2.participant_id is None

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(project_id="aaaaaaaaaaaa", action="banana")

    def test_invalid_decision_type_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa", decision_type="vibe-based"
            )

    def test_invalid_project_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(project_id="UPPERCASE123")
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(project_id="../escape")
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(project_id="short")

    def test_invalid_source_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa", source_id="UPPERCASE123"
            )
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa", source_id="../escape"
            )

    def test_invalid_participant_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa", participant_id="UPPERCASE123"
            )

    def test_invalid_entry_id_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa",
                entry_id="UPPERCASE123",
            )

    def test_explicit_entry_id(self) -> None:
        e = SamplingEntry.new(
            project_id="aaaaaaaaaaaa",
            entry_id="bbbbbbbbbbbb",
        )
        assert e.id == "bbbbbbbbbbbb"

    def test_explicit_now(self) -> None:
        e = SamplingEntry.new(
            project_id="aaaaaaaaaaaa",
            now="2024-04-01T00:00:00.000000Z",
        )
        assert e.created_at == "2024-04-01T00:00:00.000000Z"

    def test_target_category_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa",
                target_category="x" * (MAX_TARGET_CATEGORY_LEN + 1),
            )

    def test_rationale_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa",
                rationale="x" * (MAX_RATIONALE_LEN + 1),
            )

    def test_notes_too_long_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.new(
                project_id="aaaaaaaaaaaa",
                notes="x" * (MAX_NOTES_LEN + 1),
            )


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_preserves_fields(self) -> None:
        e = SamplingEntry.new(
            project_id="aaaaaaaaaaaa",
            action="added",
            decision_type="theoretical",
            source_id="0123456789ab",
            participant_id="fedcba987654",
            target_category="managing disclosure",
            rationale="Test",
            notes="N",
        )
        d = e.to_dict()
        assert json.dumps(d)  # JSON-serialisable
        e2 = SamplingEntry.from_dict(d)
        assert e2.to_dict() == d

    def test_from_dict_requires_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.from_dict({
                "project_id": "aaaaaaaaaaaa",
                "created_at": "2024-04-01T00:00:00.000000Z",
            })

    def test_from_dict_requires_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.from_dict({
                "id": "bbbbbbbbbbbb",
                "created_at": "2024-04-01T00:00:00.000000Z",
            })

    def test_from_dict_requires_created_at(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.from_dict({
                "id": "bbbbbbbbbbbb",
                "project_id": "aaaaaaaaaaaa",
            })

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectValidationError):
            SamplingEntry.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_defaults_optional_fields(self) -> None:
        e = SamplingEntry.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "created_at": "2024-04-01T00:00:00.000000Z",
        })
        assert e.action == "added"
        assert e.decision_type == ""
        assert e.source_id is None
        assert e.participant_id is None
        assert e.target_category == ""
        assert e.rationale == ""
        assert e.notes == ""

    def test_from_dict_treats_empty_source_id_as_none(self) -> None:
        e = SamplingEntry.from_dict({
            "id": "bbbbbbbbbbbb",
            "project_id": "aaaaaaaaaaaa",
            "created_at": "2024-04-01T00:00:00.000000Z",
            "source_id": "",
            "participant_id": None,
        })
        assert e.source_id is None
        assert e.participant_id is None


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_append_and_read(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = SamplingEntry.new(
            project_id=proj.id,
            action="added",
            decision_type="theoretical",
            source_id="0123456789ab",
            rationale="First entry",
            now="2024-04-01T00:00:00.000000Z",
        )
        path = append_sampling_entry(tmp_path, e)
        assert path == sampling_log_path(tmp_path, proj.id)
        assert path.exists()

        entries = read_sampling_log(tmp_path, proj.id)
        assert len(entries) == 1
        assert entries[0].to_dict() == e.to_dict()

    def test_append_is_jsonl_one_line_per_entry(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        for i in range(3):
            e = SamplingEntry.new(
                project_id=proj.id,
                rationale=f"entry {i}",
                now=f"2024-04-0{i + 1}T00:00:00.000000Z",
            )
            append_sampling_entry(tmp_path, e)
        text = sampling_log_path(tmp_path, proj.id).read_text()
        # Exactly 3 newline-terminated lines.
        assert text.count("\n") == 3
        # Each line parses as JSON.
        for line in text.splitlines():
            assert json.loads(line)

    def test_append_preserves_order(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Append in a deliberate order — and importantly *out* of
        # created_at order — to assert read order matches append order,
        # not sorted-by-time.
        a = SamplingEntry.new(
            project_id=proj.id,
            rationale="first appended",
            now="2024-06-01T00:00:00.000000Z",
        )
        b = SamplingEntry.new(
            project_id=proj.id,
            rationale="second appended (earlier ts)",
            now="2024-04-01T00:00:00.000000Z",
        )
        c = SamplingEntry.new(
            project_id=proj.id,
            rationale="third appended",
            now="2024-05-01T00:00:00.000000Z",
        )
        append_sampling_entry(tmp_path, a)
        append_sampling_entry(tmp_path, b)
        append_sampling_entry(tmp_path, c)
        rationales = [
            x.rationale for x in read_sampling_log(tmp_path, proj.id)
        ]
        assert rationales == [
            "first appended",
            "second appended (earlier ts)",
            "third appended",
        ]

    def test_append_requires_existing_project(self, tmp_path: Path) -> None:
        # No save_project — directory does not exist.
        e = SamplingEntry.new(project_id="aaaaaaaaaaaa")
        with pytest.raises(FileNotFoundError):
            append_sampling_entry(tmp_path, e)

    def test_append_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = SamplingEntry.new(project_id=proj.id)
        e.action = "totally-bogus"  # corrupt directly
        with pytest.raises(ProjectValidationError):
            append_sampling_entry(tmp_path, e)
        # No file was written.
        assert not sampling_log_path(tmp_path, proj.id).exists()

    def test_read_missing_log_returns_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Project saved but no entries appended.
        assert read_sampling_log(tmp_path, proj.id) == []

    def test_read_no_project_returns_empty(self, tmp_path: Path) -> None:
        # Project never created — no errors, just empty.
        assert read_sampling_log(tmp_path, "aaaaaaaaaaaa") == []

    def test_read_validates_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            read_sampling_log(tmp_path, "../etc/passwd")

    def test_read_skips_corrupt_lines(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        good = SamplingEntry.new(
            project_id=proj.id,
            rationale="kept",
            now="2024-04-01T00:00:00.000000Z",
        )
        append_sampling_entry(tmp_path, good)
        # Hand-write some garbage lines to the JSONL.
        path = sampling_log_path(tmp_path, proj.id)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")  # blank — skipped silently
            f.write("not valid json\n")
            f.write(json.dumps({"id": "tooshort"}) + "\n")  # invalid id
            f.write(json.dumps({
                "id": "cccccccccccc",
                # missing project_id and created_at — invalid
            }) + "\n")
        entries = read_sampling_log(tmp_path, proj.id)
        # Only the good one survives.
        assert len(entries) == 1
        assert entries[0].rationale == "kept"

    def test_read_skips_blank_trailing_lines(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = SamplingEntry.new(project_id=proj.id, rationale="x")
        append_sampling_entry(tmp_path, e)
        path = sampling_log_path(tmp_path, proj.id)
        # Add trailing whitespace; reader must not crash.
        with path.open("a", encoding="utf-8") as f:
            f.write("\n\n   \n")
        entries = read_sampling_log(tmp_path, proj.id)
        assert len(entries) == 1

    def test_count_sampling_entries(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert count_sampling_entries(tmp_path, proj.id) == 0
        for i in range(4):
            append_sampling_entry(
                tmp_path,
                SamplingEntry.new(
                    project_id=proj.id, rationale=f"r{i}"
                ),
            )
        assert count_sampling_entries(tmp_path, proj.id) == 4

    def test_sampling_log_path_validates_project_id(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ProjectValidationError):
            sampling_log_path(tmp_path, "../escape")

    def test_project_deletion_cascades(self, tmp_path: Path) -> None:
        # The sampling log lives inside the project directory, so
        # delete_project should clean it up. Mirrors the same assertion
        # in test_participants.
        proj = _saved_project(tmp_path)
        append_sampling_entry(
            tmp_path,
            SamplingEntry.new(project_id=proj.id, rationale="x"),
        )
        from scribe.projects import delete_project, project_dir
        assert project_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not project_dir(tmp_path, proj.id).exists()

    def test_unicode_rationale_round_trips(self, tmp_path: Path) -> None:
        # Researchers often work with non-English data; the JSONL must
        # not re-encode UTF-8 as escaped \\uXXXX (we set ensure_ascii=False).
        proj = _saved_project(tmp_path)
        e = SamplingEntry.new(
            project_id=proj.id,
            rationale="サンプリングの理由",
            target_category="éèçà",
        )
        append_sampling_entry(tmp_path, e)
        # Bytes-level: the original glyphs are present, not escaped.
        raw = sampling_log_path(tmp_path, proj.id).read_text(
            encoding="utf-8"
        )
        assert "サンプリングの理由" in raw
        assert "éèçà" in raw
        loaded = read_sampling_log(tmp_path, proj.id)
        assert loaded[0].rationale == "サンプリングの理由"
        assert loaded[0].target_category == "éèçà"
