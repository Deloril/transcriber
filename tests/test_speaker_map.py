"""Tests for scribe.speaker_map (F3.4).

Speaker awareness for queries: per-source maps of raw transcript
speaker labels → role + (optional) participant link, plus pure-Python
helpers that filter / count transcript segments by role / participant
/ label.

The module is stand-alone (no FastAPI, no engine imports), so all
tests stay in pure Python.
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
from scribe.participants import (
    Participant,
    save_participant,
)
from scribe.sources import (
    Source,
    save_source,
)
from scribe.speaker_map import (
    MAX_DISPLAY_NAME_LEN,
    MAX_ENTRIES,
    MAX_LABEL_LEN,
    PARTICIPANT_VOICE_ROLES,
    SPEAKER_MAPS_DIRNAME,
    SPEAKER_ROLES,
    SpeakerEntry,
    SpeakerMap,
    delete_speaker_map,
    filter_segments_by_label,
    filter_segments_by_participant,
    filter_segments_by_role,
    list_speaker_maps,
    load_or_empty_speaker_map,
    load_speaker_map,
    merge_segments_into_map,
    participant_distribution,
    participant_voice_segments,
    role_distribution,
    save_speaker_map,
    speaker_labels_in_segments,
    speaker_map_from_segments,
    speaker_map_state_path,
    speaker_maps_dir,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _saved_source(
    tmp_path: Path, project_id: str, *, name: str = "S", sid: str | None = None
) -> Source:
    s = Source.new(project_id=project_id, name=name, source_id=sid)
    save_source(tmp_path, s)
    return s


def _saved_participant(
    tmp_path: Path, project_id: str, *, name: str = "P03", pid: str | None = None
) -> Participant:
    p = Participant.new(project_id=project_id, name=name, participant_id=pid)
    save_participant(tmp_path, p)
    return p


def _seg(speaker: str, text: str = "") -> dict:
    """Tiny segment-shaped dict, like Scribe writes to JSON."""
    return {"speaker": speaker, "text": text, "start": 0.0, "end": 1.0}


# --------------------------------------------------------------------------- #
# SpeakerEntry
# --------------------------------------------------------------------------- #


class TestSpeakerEntry:
    def test_defaults(self) -> None:
        e = SpeakerEntry(label="SPEAKER_00")
        e.validate()
        assert e.role == "unknown"
        assert e.participant_id is None
        assert e.display_name == ""

    def test_round_trip(self) -> None:
        e = SpeakerEntry(
            label="SPEAKER_00",
            role="interviewer",
            participant_id="abcdef012345",
            display_name="Researcher",
            notes="opening prompt only",
        )
        e.validate()
        assert SpeakerEntry.from_dict(e.to_dict()) == e

    def test_label_strip_and_required(self) -> None:
        e = SpeakerEntry(label="  Luke  ")
        e.validate()
        assert e.label == "Luke"

        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="").validate()
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="    ").validate()

    def test_label_length_bound(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="x" * (MAX_LABEL_LEN + 1)).validate()

    def test_role_must_be_known(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="x", role="moderator").validate()

    def test_participant_id_shape(self) -> None:
        # Valid 12-hex passes.
        SpeakerEntry(
            label="x", participant_id="0123456789ab"
        ).validate()
        # Anything else rejects.
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="x", participant_id="not-hex").validate()
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="x", participant_id="ABCDEF012345").validate()

    def test_display_name_length(self) -> None:
        long = "n" * (MAX_DISPLAY_NAME_LEN + 1)
        with pytest.raises(ProjectValidationError):
            SpeakerEntry(label="x", display_name=long).validate()

    def test_from_dict_requires_label(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerEntry.from_dict({"role": "interviewer"})

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerEntry.from_dict("not-a-dict")  # type: ignore[arg-type]

    def test_from_dict_handles_empty_participant(self) -> None:
        # Empty string / None should normalise to None on disk.
        e = SpeakerEntry.from_dict({"label": "X", "participant_id": ""})
        assert e.participant_id is None
        e = SpeakerEntry.from_dict({"label": "X", "participant_id": None})
        assert e.participant_id is None


# --------------------------------------------------------------------------- #
# SpeakerMap construction + serialisation
# --------------------------------------------------------------------------- #


class TestSpeakerMapConstruction:
    def test_new_with_no_entries(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        assert m.entries == []
        assert m.created_at == m.modified_at

    def test_new_accepts_dicts_and_entries(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(label="SPEAKER_00", role="interviewer"),
                {"label": "SPEAKER_01", "role": "interviewee"},
            ],
        )
        assert [e.label for e in m.entries] == ["SPEAKER_00", "SPEAKER_01"]
        assert m.entries[1].role == "interviewee"

    def test_new_rejects_non_dict_entry(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            SpeakerMap.new(
                project_id=proj.id,
                source_id=src.id,
                entries=["SPEAKER_00"],  # type: ignore[list-item]
            )

    def test_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[SpeakerEntry(label="A"), SpeakerEntry(label="B")],
        )
        again = SpeakerMap.from_dict(m.to_dict())
        assert again.project_id == m.project_id
        assert again.source_id == m.source_id
        assert [e.label for e in again.entries] == ["A", "B"]

    def test_from_dict_requires_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerMap.from_dict({"source_id": "abcdef012345"})
        with pytest.raises(ProjectValidationError):
            SpeakerMap.from_dict({"project_id": "abcdef012345"})

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerMap.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_rejects_non_list_entries(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            SpeakerMap.from_dict({
                "project_id": proj.id,
                "source_id": src.id,
                "entries": "nope",
            })

    def test_validate_rejects_invalid_project_id(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ProjectValidationError):
            SpeakerMap.new(project_id="not-hex", source_id="abcdef012345")

    def test_validate_rejects_invalid_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            SpeakerMap.new(project_id=proj.id, source_id="not-hex")

    def test_validate_rejects_duplicate_labels(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            SpeakerMap.new(
                project_id=proj.id,
                source_id=src.id,
                entries=[
                    SpeakerEntry(label="A"),
                    SpeakerEntry(label="A"),
                ],
            )


# --------------------------------------------------------------------------- #
# SpeakerMap lookups
# --------------------------------------------------------------------------- #


class TestSpeakerMapLookups:
    def _map(self, tmp_path: Path) -> SpeakerMap:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        return SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewer",
                    display_name="Researcher",
                ),
                SpeakerEntry(
                    label="SPEAKER_01",
                    role="interviewee",
                    participant_id="abcdef012345",
                ),
                SpeakerEntry(
                    label="SPEAKER_02",
                    role="interviewee",
                    participant_id="bbbbbbbbbbbb",
                ),
            ],
        )

    def test_get_and_has(self, tmp_path: Path) -> None:
        m = self._map(tmp_path)
        assert m.get("SPEAKER_00") is not None
        assert m.has("SPEAKER_00") is True
        assert m.get("missing") is None
        assert m.has("missing") is False

    def test_role_for(self, tmp_path: Path) -> None:
        m = self._map(tmp_path)
        assert m.role_for("SPEAKER_00") == "interviewer"
        assert m.role_for("SPEAKER_01") == "interviewee"
        # Absent labels report unknown rather than raising.
        assert m.role_for("missing") == "unknown"

    def test_participant_for(self, tmp_path: Path) -> None:
        m = self._map(tmp_path)
        assert m.participant_for("SPEAKER_01") == "abcdef012345"
        assert m.participant_for("SPEAKER_00") is None
        assert m.participant_for("missing") is None

    def test_display_name_for_falls_back_to_label(
        self, tmp_path: Path
    ) -> None:
        m = self._map(tmp_path)
        assert m.display_name_for("SPEAKER_00") == "Researcher"
        # No display name set → label.
        assert m.display_name_for("SPEAKER_01") == "SPEAKER_01"
        # Absent label echoes itself.
        assert m.display_name_for("Ghost") == "Ghost"

    def test_labels_for_role_accepts_str_or_iterable(
        self, tmp_path: Path
    ) -> None:
        m = self._map(tmp_path)
        assert m.labels_for_role("interviewee") == [
            "SPEAKER_01",
            "SPEAKER_02",
        ]
        assert m.labels_for_role(["interviewer", "interviewee"]) == [
            "SPEAKER_00",
            "SPEAKER_01",
            "SPEAKER_02",
        ]
        assert m.labels_for_role("facilitator") == []

    def test_labels_for_role_rejects_unknown_role(
        self, tmp_path: Path
    ) -> None:
        m = self._map(tmp_path)
        with pytest.raises(ProjectValidationError):
            m.labels_for_role("moderator")

    def test_labels_for_participant(self, tmp_path: Path) -> None:
        m = self._map(tmp_path)
        assert m.labels_for_participant("abcdef012345") == ["SPEAKER_01"]
        assert m.labels_for_participant("bbbbbbbbbbbb") == ["SPEAKER_02"]
        assert m.labels_for_participant("ffffffffffff") == []

    def test_labels_for_participant_rejects_bad_id(
        self, tmp_path: Path
    ) -> None:
        m = self._map(tmp_path)
        with pytest.raises(ProjectValidationError):
            m.labels_for_participant("not-hex")

    def test_participants_distinct_in_insertion_order(
        self, tmp_path: Path
    ) -> None:
        m = self._map(tmp_path)
        assert m.participants() == ["abcdef012345", "bbbbbbbbbbbb"]

    def test_labels(self, tmp_path: Path) -> None:
        m = self._map(tmp_path)
        assert m.labels() == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


# --------------------------------------------------------------------------- #
# SpeakerMap mutation
# --------------------------------------------------------------------------- #


class TestSpeakerMapMutation:
    def test_upsert_inserts_new(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id, now="t0")
        e = m.upsert_entry(
            "SPEAKER_00",
            role="interviewer",
            display_name="Researcher",
            now="t1",
        )
        assert e.label == "SPEAKER_00"
        assert e.role == "interviewer"
        assert m.entries == [e]
        assert m.modified_at == "t1"

    def test_upsert_patches_existing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewee",
                    display_name="Sam",
                )
            ],
            now="t0",
        )
        e = m.upsert_entry("SPEAKER_00", role="interviewer", now="t1")
        # role overwritten, display_name kept.
        assert e.role == "interviewer"
        assert e.display_name == "Sam"
        assert m.modified_at == "t1"
        # Same entry mutated in place; still only one row.
        assert len(m.entries) == 1

    def test_upsert_explicit_empty_string_clears(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(
                    label="A",
                    participant_id="abcdef012345",
                    display_name="Sam",
                )
            ],
        )
        # Pass empty string for participant_id → clears link.
        m.upsert_entry("A", participant_id="")
        assert m.entries[0].participant_id is None
        # Pass empty string for display_name → clears.
        m.upsert_entry("A", display_name="")
        assert m.entries[0].display_name == ""

    def test_upsert_rejects_unknown_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        with pytest.raises(ProjectValidationError):
            m.upsert_entry("A", role="moderator")

    def test_upsert_enforces_max_entries(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        # Build a map at the limit.
        entries = [SpeakerEntry(label=f"L{i:04d}") for i in range(MAX_ENTRIES)]
        m = SpeakerMap.new(
            project_id=proj.id, source_id=src.id, entries=entries
        )
        with pytest.raises(ProjectValidationError):
            m.upsert_entry("OVER")
        # Updating an existing entry at the limit is fine.
        m.upsert_entry("L0000", role="interviewer")
        assert m.role_for("L0000") == "interviewer"

    def test_remove_entry(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[SpeakerEntry(label="A"), SpeakerEntry(label="B")],
            now="t0",
        )
        assert m.remove_entry("A", now="t1") is True
        assert m.labels() == ["B"]
        assert m.modified_at == "t1"
        # Idempotent on missing label.
        assert m.remove_entry("Ghost", now="t2") is False
        assert m.modified_at == "t1"

    def test_set_role_and_link_unlink(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        m.set_role("A", "interviewer")
        assert m.role_for("A") == "interviewer"
        m.link_participant("A", "abcdef012345")
        assert m.participant_for("A") == "abcdef012345"
        m.unlink_participant("A")
        assert m.participant_for("A") is None

    def test_unlink_unknown_label_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        with pytest.raises(ProjectValidationError):
            m.unlink_participant("ghost")


# --------------------------------------------------------------------------- #
# Pure helpers — segment introspection
# --------------------------------------------------------------------------- #


class TestSpeakerLabelsInSegments:
    def test_empty_input(self) -> None:
        assert speaker_labels_in_segments([]) == []

    def test_first_occurrence_order(self) -> None:
        segs = [
            _seg("SPEAKER_01"),
            _seg("SPEAKER_00"),
            _seg("SPEAKER_01"),  # repeat → skipped
            _seg("SPEAKER_02"),
            _seg("SPEAKER_00"),  # repeat → skipped
        ]
        assert speaker_labels_in_segments(segs) == [
            "SPEAKER_01",
            "SPEAKER_00",
            "SPEAKER_02",
        ]

    def test_skips_empty_and_missing(self) -> None:
        segs = [
            {"speaker": "", "text": ""},
            {"speaker": None, "text": ""},
            {"text": "no speaker key"},
            _seg("Luke"),
        ]
        assert speaker_labels_in_segments(segs) == ["Luke"]

    def test_supports_attribute_objects(self) -> None:
        class S:
            def __init__(self, sp: str) -> None:
                self.speaker = sp

        assert speaker_labels_in_segments([S("Luke"), S("Sam"), S("Luke")]) == [
            "Luke",
            "Sam",
        ]


class TestSpeakerMapFromSegments:
    def test_seeds_with_default_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        segs = [_seg("SPEAKER_00"), _seg("SPEAKER_01"), _seg("SPEAKER_00")]
        m = speaker_map_from_segments(
            project_id=proj.id, source_id=src.id, segments=segs
        )
        assert m.labels() == ["SPEAKER_00", "SPEAKER_01"]
        assert all(e.role == "unknown" for e in m.entries)

    def test_seeds_with_custom_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        segs = [_seg("Luke"), _seg("Guest")]
        m = speaker_map_from_segments(
            project_id=proj.id,
            source_id=src.id,
            segments=segs,
            default_role="interviewee",
        )
        assert all(e.role == "interviewee" for e in m.entries)

    def test_rejects_unknown_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        with pytest.raises(ProjectValidationError):
            speaker_map_from_segments(
                project_id=proj.id,
                source_id=src.id,
                segments=[_seg("X")],
                default_role="moderator",
            )


class TestMergeSegmentsIntoMap:
    def test_adds_only_new_labels(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(label="SPEAKER_00", role="interviewer"),
            ],
            now="t0",
        )
        added = merge_segments_into_map(
            m,
            [_seg("SPEAKER_00"), _seg("SPEAKER_01"), _seg("SPEAKER_02")],
            now="t1",
        )
        assert added == ["SPEAKER_01", "SPEAKER_02"]
        # Existing role preserved.
        assert m.role_for("SPEAKER_00") == "interviewer"
        assert m.role_for("SPEAKER_01") == "unknown"
        assert m.modified_at == "t1"

    def test_no_op_no_clock_advance(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[SpeakerEntry(label="A", role="interviewer")],
            now="t0",
        )
        added = merge_segments_into_map(m, [_seg("A")], now="t1")
        assert added == []
        assert m.modified_at == "t0"

    def test_custom_new_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        merge_segments_into_map(
            m, [_seg("A")], new_role="interviewee"
        )
        assert m.role_for("A") == "interviewee"

    def test_rejects_unknown_role(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        with pytest.raises(ProjectValidationError):
            merge_segments_into_map(m, [_seg("A")], new_role="moderator")


# --------------------------------------------------------------------------- #
# Pure helpers — filtering
# --------------------------------------------------------------------------- #


class TestFilterSegments:
    def _setup(self, tmp_path: Path) -> tuple[SpeakerMap, list[dict]]:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(label="Luke", role="interviewer"),
                SpeakerEntry(
                    label="Sam",
                    role="interviewee",
                    participant_id="abcdef012345",
                ),
                SpeakerEntry(
                    label="Pat",
                    role="interviewee",
                    participant_id="bbbbbbbbbbbb",
                ),
            ],
        )
        segs = [
            _seg("Luke", "Q1"),
            _seg("Sam", "A1"),
            _seg("Pat", "A2"),
            _seg("Luke", "Q2"),
            _seg("Sam", "A3"),
            _seg("Stranger", "?"),
            _seg("", "void"),
        ]
        return m, segs

    def test_filter_by_label_str(self, tmp_path: Path) -> None:
        _, segs = self._setup(tmp_path)
        out = filter_segments_by_label(segs, "Luke")
        assert [s["text"] for s in out] == ["Q1", "Q2"]

    def test_filter_by_label_list(self, tmp_path: Path) -> None:
        _, segs = self._setup(tmp_path)
        out = filter_segments_by_label(segs, ["Sam", "Pat"])
        assert [s["text"] for s in out] == ["A1", "A2", "A3"]

    def test_filter_by_label_excludes_empty(self, tmp_path: Path) -> None:
        _, segs = self._setup(tmp_path)
        # Empty label can't be a wanted label (empty/missing is dropped).
        assert filter_segments_by_label(segs, "") == []

    def test_filter_by_role_interviewer(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_role(segs, m, "interviewer")
        assert [s["text"] for s in out] == ["Q1", "Q2"]

    def test_filter_by_role_interviewee(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_role(segs, m, "interviewee")
        assert [s["text"] for s in out] == ["A1", "A2", "A3"]

    def test_filter_by_role_treats_unmapped_as_unknown(
        self, tmp_path: Path
    ) -> None:
        m, segs = self._setup(tmp_path)
        # "Stranger" is unmapped → role == unknown.
        out = filter_segments_by_role(segs, m, "unknown")
        assert [s["text"] for s in out] == ["?"]

    def test_filter_by_role_include_unmapped(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_role(
            segs, m, "interviewer", include_unmapped=True
        )
        # Interviewer + unmapped Stranger.
        assert [s["text"] for s in out] == ["Q1", "Q2", "?"]

    def test_filter_by_role_iterable_roles(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_role(
            segs, m, ["interviewer", "interviewee"]
        )
        assert [s["text"] for s in out] == ["Q1", "A1", "A2", "Q2", "A3"]

    def test_filter_by_role_rejects_unknown_role(
        self, tmp_path: Path
    ) -> None:
        m, _ = self._setup(tmp_path)
        with pytest.raises(ProjectValidationError):
            filter_segments_by_role([], m, "moderator")

    def test_filter_by_participant_one(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_participant(segs, m, "abcdef012345")
        assert [s["text"] for s in out] == ["A1", "A3"]

    def test_filter_by_participant_many(self, tmp_path: Path) -> None:
        m, segs = self._setup(tmp_path)
        out = filter_segments_by_participant(
            segs, m, ["abcdef012345", "bbbbbbbbbbbb"]
        )
        assert [s["text"] for s in out] == ["A1", "A2", "A3"]

    def test_filter_by_participant_drops_unmapped(
        self, tmp_path: Path
    ) -> None:
        m, segs = self._setup(tmp_path)
        # Stranger is unmapped → no participant link → never matches.
        out = filter_segments_by_participant(segs, m, "ffffffffffff")
        assert out == []

    def test_filter_by_participant_validates_id(
        self, tmp_path: Path
    ) -> None:
        m, segs = self._setup(tmp_path)
        with pytest.raises(ProjectValidationError):
            filter_segments_by_participant(segs, m, "not-hex")

    def test_participant_voice_segments_default(
        self, tmp_path: Path
    ) -> None:
        m, segs = self._setup(tmp_path)
        out = participant_voice_segments(segs, m)
        # Default = interviewee + facilitator → just the interviewees.
        assert [s["text"] for s in out] == ["A1", "A2", "A3"]

    def test_participant_voice_segments_override(
        self, tmp_path: Path
    ) -> None:
        m, segs = self._setup(tmp_path)
        out = participant_voice_segments(
            segs, m, voice_roles=["interviewer"]
        )
        assert [s["text"] for s in out] == ["Q1", "Q2"]


# --------------------------------------------------------------------------- #
# Pure helpers — counting
# --------------------------------------------------------------------------- #


class TestRoleAndParticipantDistribution:
    def test_role_distribution_has_all_roles(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(label="A", role="interviewer"),
                SpeakerEntry(label="B", role="interviewee"),
            ],
        )
        segs = [
            _seg("A"),
            _seg("B"),
            _seg("B"),
            _seg("Ghost"),  # unmapped → unknown
            _seg(""),
        ]
        d = role_distribution(segs, m)
        # Every canonical role is a key, even at zero.
        for r in SPEAKER_ROLES:
            assert r in d
        assert d["interviewer"] == 1
        assert d["interviewee"] == 2
        # Unmapped → unknown bucket.
        assert d["unknown"] == 1
        # Empty / missing speaker → "" bucket.
        assert d[""] == 1
        # No facilitator segments.
        assert d["facilitator"] == 0

    def test_participant_distribution(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[
                SpeakerEntry(
                    label="Luke", role="interviewer"
                ),  # no participant link
                SpeakerEntry(
                    label="Sam",
                    role="interviewee",
                    participant_id="abcdef012345",
                ),
                SpeakerEntry(
                    label="Pat",
                    role="interviewee",
                    participant_id="bbbbbbbbbbbb",
                ),
            ],
        )
        segs = [
            _seg("Luke"),
            _seg("Sam"),
            _seg("Sam"),
            _seg("Sam"),
            _seg("Pat"),
            _seg("Ghost"),
            _seg(""),
        ]
        d = participant_distribution(segs, m)
        # Linked participants.
        assert d["abcdef012345"] == 3
        assert d["bbbbbbbbbbbb"] == 1
        # Empty bucket = no link (interviewer + ghost + empty speaker).
        assert d[""] == 3


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[SpeakerEntry(label="X", role="interviewer")],
        )
        path = save_speaker_map(tmp_path, m)
        assert path.exists()
        again = load_speaker_map(tmp_path, proj.id, src.id)
        assert again.entries[0].label == "X"
        assert again.entries[0].role == "interviewer"

    def test_save_creates_speaker_maps_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        save_speaker_map(tmp_path, m)
        d = speaker_maps_dir(tmp_path, proj.id)
        assert d.exists()
        assert d.name == SPEAKER_MAPS_DIRNAME

    def test_save_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = SpeakerMap.new(project_id=proj.id, source_id=src.id)
        save_speaker_map(tmp_path, m)
        sd = speaker_maps_dir(tmp_path, proj.id)
        # No leftover .tmp files.
        assert not list(sd.glob("*.tmp"))

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        # No project saved → directory missing.
        m = SpeakerMap.new(
            project_id="ffffffffffff", source_id="ffffffffffff"
        )
        with pytest.raises(FileNotFoundError):
            save_speaker_map(tmp_path, m)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        with pytest.raises(FileNotFoundError):
            load_speaker_map(tmp_path, proj.id, src.id)

    def test_load_or_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        m = load_or_empty_speaker_map(tmp_path, proj.id, src.id)
        assert m.entries == []
        assert m.project_id == proj.id
        assert m.source_id == src.id

    def test_state_path_validates_source_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            speaker_map_state_path(tmp_path, proj.id, "../../etc")

    def test_list_speaker_maps_skips_garbage(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src1 = _saved_source(tmp_path, proj.id)
        src2 = _saved_source(tmp_path, proj.id, name="S2")
        save_speaker_map(
            tmp_path,
            SpeakerMap.new(project_id=proj.id, source_id=src1.id),
        )
        save_speaker_map(
            tmp_path,
            SpeakerMap.new(project_id=proj.id, source_id=src2.id),
        )
        # Drop a corrupt JSON in the dir.
        sd = speaker_maps_dir(tmp_path, proj.id)
        (sd / "deadbeef0000.json").write_text("{ not-json")
        # And a wrong-shape filename.
        (sd / "junk.json").write_text("{}")
        out = list_speaker_maps(tmp_path, proj.id)
        sids = sorted(m.source_id for m in out)
        assert sids == sorted([src1.id, src2.id])

    def test_list_speaker_maps_empty_when_dir_missing(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        # No save → no dir → empty list.
        assert list_speaker_maps(tmp_path, proj.id) == []

    def test_list_speaker_maps_validates_project_id(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ProjectValidationError):
            list_speaker_maps(tmp_path, "not-hex")

    def test_delete_speaker_map(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        src = _saved_source(tmp_path, proj.id)
        save_speaker_map(
            tmp_path,
            SpeakerMap.new(project_id=proj.id, source_id=src.id),
        )
        assert delete_speaker_map(tmp_path, proj.id, src.id) is True
        assert delete_speaker_map(tmp_path, proj.id, src.id) is False


# --------------------------------------------------------------------------- #
# Integration: ProjectBundle round-trip with speaker maps
# --------------------------------------------------------------------------- #


class TestBundleIntegration:
    def test_bundle_round_trip_persists_speaker_maps(
        self, tmp_path: Path
    ) -> None:
        from scribe.project_format import (
            COMPONENT_SPEAKER_MAPS_DIR,
            DEFAULT_COMPONENT_PATHS,
            ProjectBundle,
            load_project_bundle,
            save_project_bundle,
        )

        # Component name registered.
        assert COMPONENT_SPEAKER_MAPS_DIR in DEFAULT_COMPONENT_PATHS

        proj = Project.new(name="Bundle test")
        src = Source.new(project_id=proj.id, name="S1")
        smap = SpeakerMap.new(
            project_id=proj.id,
            source_id=src.id,
            entries=[SpeakerEntry(label="X", role="interviewer")],
        )
        bundle = ProjectBundle(
            project=proj, sources=[src], speaker_maps=[smap]
        )
        save_project_bundle(tmp_path, bundle)

        loaded = load_project_bundle(tmp_path, proj.id)
        assert len(loaded.speaker_maps) == 1
        assert loaded.speaker_maps[0].entries[0].label == "X"

    def test_bundle_validate_rejects_wrong_project(
        self, tmp_path: Path
    ) -> None:
        from scribe.project_format import ProjectBundle, ProjectFormatError

        proj = Project.new(name="A")
        # SpeakerMap with a different project id.
        smap = SpeakerMap.new(
            project_id="ffffffffffff",
            source_id="aaaaaaaaaaaa",
        )
        with pytest.raises(ProjectFormatError):
            ProjectBundle(
                project=proj, speaker_maps=[smap]
            ).validate()

    def test_bundle_validate_rejects_duplicate_source(
        self, tmp_path: Path
    ) -> None:
        from scribe.project_format import ProjectBundle, ProjectFormatError

        proj = Project.new(name="A")
        sid = "aaaaaaaaaaaa"
        m1 = SpeakerMap.new(project_id=proj.id, source_id=sid)
        m2 = SpeakerMap.new(project_id=proj.id, source_id=sid)
        with pytest.raises(ProjectFormatError):
            ProjectBundle(
                project=proj, speaker_maps=[m1, m2]
            ).validate()


# --------------------------------------------------------------------------- #
# Module surface checks
# --------------------------------------------------------------------------- #


class TestModuleSurface:
    def test_speaker_roles_includes_unknown(self) -> None:
        # "unknown" is the default role so it must always be in the
        # vocabulary.
        assert "unknown" in SPEAKER_ROLES
        # And the obvious researcher-relevant ones.
        assert "interviewer" in SPEAKER_ROLES
        assert "interviewee" in SPEAKER_ROLES

    def test_participant_voice_roles_subset(self) -> None:
        for r in PARTICIPANT_VOICE_ROLES:
            assert r in SPEAKER_ROLES
