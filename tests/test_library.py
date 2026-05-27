"""Tests for ``scribe.library`` — the pure helpers that summarise
persisted transcription jobs into the row-shaped dicts the home page's
library list renders (F10.1).

The helpers here are deliberately framework-free: they take Job-shaped
dicts (or live :class:`scribe.server.Job` objects via ``to_state()``)
and reduce them to a stable row schema. Everything we test in this
module is a *pure function* — no FastAPI, no filesystem.
"""

from __future__ import annotations

from typing import Any

import pytest

from scribe import library


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _job_state(**fields: Any) -> dict[str, Any]:
    """Build a Job.to_state()-shaped dict with the bare minimum
    required by the summariser, overriding fields per-test."""
    base: dict[str, Any] = {
        "id": "abc123def456",
        "input_path": "/uploads/abc123def456/in.wav",
        "output_dir": "/outputs/abc123def456",
        "mode": "diarize",
        "speakers": None,
        "num_speakers": None,
        "language": "en",
        "model": "large-v3",
        "created_at": "2026-05-25T10:00:00Z",
        "status": "done",
        "progress": 1.0,
        "message": "Done",
        "result": None,
        "error": None,
        "output_paths": {},
        "audio_streams": 1,
        "input_filename": "in.wav",
        "options": {},
        "batch_size": 8,
        "started_at": None,
        "finished_at": None,
        "media_discarded": False,
    }
    base.update(fields)
    return base


def _result(
    *,
    language: str = "en",
    mode: str = "diarize",
    speakers: list[str] | None = None,
    end: float = 0.0,
) -> dict[str, Any]:
    """Build a minimal TranscriptionResult.to_dict()-shaped dict."""
    segments = []
    if end > 0:
        segments.append({"start": 0.0, "end": end, "speaker": "A", "text": "hi", "words": []})
    return {
        "language": language,
        "mode": mode,
        "speakers": speakers if speakers is not None else ["A", "B"],
        "segments": segments,
    }


# --------------------------------------------------------------------------- #
# summarise_job
# --------------------------------------------------------------------------- #


class TestSummariseJobShape:
    """Every row has the same keys, even on a sparse job."""

    EXPECTED_KEYS = {
        "id", "input_filename", "display_name",
        "mode", "language", "model",
        "status", "progress", "message",
        "created_at", "started_at", "finished_at",
        "audio_streams", "speakers", "speaker_count",
        "duration_seconds", "has_outputs", "media_discarded", "error",
    }

    def test_returns_all_expected_keys(self) -> None:
        row = library.summarise_job(_job_state())
        assert set(row.keys()) == self.EXPECTED_KEYS

    def test_returns_all_keys_even_on_empty_dict(self) -> None:
        row = library.summarise_job({})
        assert set(row.keys()) == self.EXPECTED_KEYS

    def test_status_defaults_to_done_when_missing(self) -> None:
        # Legacy job.json files written before the status field existed
        # are treated as ``done`` rather than crashing the listing.
        row = library.summarise_job({"id": "abc123def456"})
        assert row["status"] == "done"

    def test_status_passes_through(self) -> None:
        for status in ("queued", "running", "done", "error"):
            row = library.summarise_job(_job_state(status=status))
            assert row["status"] == status


class TestSummariseJobSpeakers:
    def test_none_result_yields_empty_speakers(self) -> None:
        row = library.summarise_job(_job_state(result=None))
        assert row["speakers"] == []
        assert row["speaker_count"] == 0

    def test_extracts_speakers_from_result(self) -> None:
        row = library.summarise_job(_job_state(
            result=_result(speakers=["Luke", "Maria"]),
        ))
        assert row["speakers"] == ["Luke", "Maria"]
        assert row["speaker_count"] == 2

    def test_strips_whitespace_and_empties(self) -> None:
        row = library.summarise_job(_job_state(
            result=_result(speakers=["  Luke  ", "", "  ", "Maria"]),
        ))
        assert row["speakers"] == ["Luke", "Maria"]
        assert row["speaker_count"] == 2

    def test_ignores_non_string_speaker_entries(self) -> None:
        row = library.summarise_job(_job_state(
            result=_result(speakers=["Luke", 42, None, {"x": 1}, "Maria"]),
        ))
        assert row["speakers"] == ["Luke", "Maria"]

    def test_malformed_result_doesnt_crash(self) -> None:
        # ``result`` could be anything if the file was hand-edited.
        for bad in ("string", 42, [], {"speakers": "not a list"}):
            row = library.summarise_job(_job_state(result=bad))
            assert row["speakers"] == []
            assert row["speaker_count"] == 0

    def test_speaker_names_override_canonical_ids(self) -> None:
        # Editor stores user renames in result["speaker_names"] without
        # mutating the canonical "speakers" list. The library should
        # show the renames so the row reflects what the user has
        # labelled the speakers.
        result = {
            "language": "en", "mode": "diarize",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_names": {"SPEAKER_00": "Luke", "SPEAKER_01": "Maria"},
            "segments": [],
        }
        row = library.summarise_job(_job_state(result=result))
        assert row["speakers"] == ["Luke", "Maria"]

    def test_partial_speaker_names_falls_through(self) -> None:
        # Only one speaker has been renamed; the other still appears as
        # its canonical id.
        result = {
            "language": "en", "mode": "diarize",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_names": {"SPEAKER_00": "Luke"},
            "segments": [],
        }
        row = library.summarise_job(_job_state(result=result))
        assert row["speakers"] == ["Luke", "SPEAKER_01"]

    def test_empty_speaker_name_does_not_override(self) -> None:
        # An empty string in speaker_names shouldn't blank out the row.
        result = {
            "language": "en", "mode": "diarize",
            "speakers": ["SPEAKER_00"],
            "speaker_names": {"SPEAKER_00": "   "},
            "segments": [],
        }
        row = library.summarise_job(_job_state(result=result))
        assert row["speakers"] == ["SPEAKER_00"]

    def test_speaker_names_not_a_dict_is_ignored(self) -> None:
        # Defensive: hand-edited file with the wrong shape doesn't crash.
        result = {
            "language": "en", "mode": "diarize",
            "speakers": ["SPEAKER_00"],
            "speaker_names": "not-a-dict",
            "segments": [],
        }
        row = library.summarise_job(_job_state(result=result))
        assert row["speakers"] == ["SPEAKER_00"]


class TestSummariseJobDuration:
    def test_returns_max_segment_end(self) -> None:
        row = library.summarise_job(_job_state(
            result={"language": "en", "mode": "diarize", "speakers": [],
                    "segments": [
                        {"start": 0.0, "end": 12.5, "speaker": "A", "text": "", "words": []},
                        {"start": 12.5, "end": 30.0, "speaker": "A", "text": "", "words": []},
                        {"start": 30.0, "end": 28.7, "speaker": "A", "text": "", "words": []},
                    ]},
        ))
        assert row["duration_seconds"] == 30.0

    def test_returns_none_for_empty_segments(self) -> None:
        row = library.summarise_job(_job_state(
            result={"language": "en", "mode": "diarize", "speakers": [], "segments": []},
        ))
        assert row["duration_seconds"] is None

    def test_returns_none_for_no_result(self) -> None:
        row = library.summarise_job(_job_state(result=None))
        assert row["duration_seconds"] is None

    def test_handles_non_numeric_end(self) -> None:
        row = library.summarise_job(_job_state(
            result={"segments": [
                {"end": "garbage"},
                {"end": 5.0},
                {"end": None},
            ]},
        ))
        assert row["duration_seconds"] == 5.0


class TestSummariseJobLanguage:
    def test_prefers_detected_language_from_result(self) -> None:
        # User uploaded with language=auto, engine detected `de` —
        # we surface the detected one so the row reflects reality.
        row = library.summarise_job(_job_state(
            language="auto",
            result=_result(language="de"),
        ))
        assert row["language"] == "de"

    def test_falls_back_to_requested_language(self) -> None:
        row = library.summarise_job(_job_state(language="en", result=None))
        assert row["language"] == "en"

    def test_handles_missing_language_in_result(self) -> None:
        row = library.summarise_job(_job_state(
            language="en",
            result={"segments": [], "speakers": []},
        ))
        assert row["language"] == "en"


class TestSummariseJobOutputs:
    def test_has_outputs_true_when_paths_present(self) -> None:
        row = library.summarise_job(_job_state(
            output_paths={"json": "outputs/x/x.json", "srt": "outputs/x/x.srt"},
        ))
        assert row["has_outputs"] is True

    def test_has_outputs_false_when_empty(self) -> None:
        row = library.summarise_job(_job_state(output_paths={}))
        assert row["has_outputs"] is False


class TestSummariseJobDisplayName:
    """User-set rename surfaced separately from the immutable upload
    filename so the library row + editor topbar can show the friendly
    label without losing the original."""

    def test_defaults_to_empty_string(self) -> None:
        row = library.summarise_job(_job_state())
        assert row["display_name"] == ""
        # Original filename still carried through verbatim.
        assert row["input_filename"] == "in.wav"

    def test_passes_through_when_set(self) -> None:
        row = library.summarise_job(_job_state(display_name="Maria — interview 2"))
        assert row["display_name"] == "Maria — interview 2"
        assert row["input_filename"] == "in.wav"

    def test_search_matches_display_name(self) -> None:
        row = library.summarise_job(_job_state(
            input_filename="raw-audio-2026-05-26.wav",
            display_name="Pilot interview with Maria",
        ))
        assert library.matches_query(row, "pilot") is True
        assert library.matches_query(row, "maria") is True
        # And the original filename is still searchable.
        assert library.matches_query(row, "raw-audio") is True


class TestSummariseJobMediaDiscarded:
    """F10.2 — the row carries a ``media_discarded`` boolean so the
    library page can render the small "📼 discarded" icon and hide
    the per-row "Discard media" action."""

    def test_defaults_to_false(self) -> None:
        row = library.summarise_job(_job_state())
        assert row["media_discarded"] is False

    def test_passes_true_through(self) -> None:
        row = library.summarise_job(_job_state(media_discarded=True))
        assert row["media_discarded"] is True

    def test_truthy_non_bool_coerces_to_bool(self) -> None:
        # Hand-edited job.json with an int / string truthy value still
        # resolves to True; we never trust the raw value.
        for raw in (1, "yes", [1, 2]):
            row = library.summarise_job(_job_state(media_discarded=raw))
            assert row["media_discarded"] is True

    def test_falsy_non_bool_coerces_to_false(self) -> None:
        for raw in (0, "", [], None):
            row = library.summarise_job(_job_state(media_discarded=raw))
            assert row["media_discarded"] is False

    def test_missing_key_is_false(self) -> None:
        d = _job_state()
        d.pop("media_discarded", None)
        row = library.summarise_job(d)
        assert row["media_discarded"] is False

    def test_string_false_resolves_to_false(self) -> None:
        # Regression: ``bool("false")`` is True in Python because the
        # string is non-empty. An older serialiser or hand-edit could
        # leave a row that read as True even though the user clearly
        # intended False. The _to_bool helper handles the canonical
        # falsy strings explicitly.
        for raw in ("false", "FALSE", "False", " false ", "no", "0", "off", "null"):
            row = library.summarise_job(_job_state(media_discarded=raw))
            assert row["media_discarded"] is False, (
                f"expected {raw!r} to coerce to False"
            )


class TestSummariseJobAcceptsLiveJob:
    """Pure-Python check that summarise_job works on a live Job instance
    via ``to_state()``. We import the dataclass lazily to avoid pulling
    the FastAPI app stack into the unit tests by accident."""

    def test_accepts_job_object_with_to_state(self) -> None:
        from pathlib import Path
        from scribe.server import Job

        job = Job(
            id="abc123def456",
            input_path=Path("/uploads/abc123def456/in.wav"),
            output_dir=Path("/outputs/abc123def456"),
            mode="diarize",
            speakers=None,
            num_speakers=None,
            language="en",
            model="large-v3",
            created_at="2026-05-25T10:00:00Z",
            status="done",
            progress=1.0,
            message="Done",
            input_filename="in.wav",
            audio_streams=1,
        )
        row = library.summarise_job(job)
        assert row["id"] == "abc123def456"
        assert row["status"] == "done"
        assert row["mode"] == "diarize"
        assert row["input_filename"] == "in.wav"

    def test_rejects_non_job_non_dict(self) -> None:
        with pytest.raises(TypeError):
            library.summarise_job(42)
        with pytest.raises(TypeError):
            library.summarise_job("hello")
        with pytest.raises(TypeError):
            library.summarise_job(None)


class TestSummariseJobFieldCap:
    def test_long_filename_is_clipped(self) -> None:
        long = "x" * 10_000
        row = library.summarise_job(_job_state(input_filename=long))
        # Cap is 4000 in the implementation; the exact value isn't part
        # of the contract but "way smaller than 10 000" is.
        assert len(row["input_filename"]) < len(long)
        assert len(row["input_filename"]) <= 4000


# --------------------------------------------------------------------------- #
# summarise_jobs (sort behaviour)
# --------------------------------------------------------------------------- #


class TestSummariseJobsSort:
    def test_newest_first(self) -> None:
        rows = library.summarise_jobs([
            _job_state(id="aaaaaaaaaaaa", created_at="2026-01-01T00:00:00Z"),
            _job_state(id="bbbbbbbbbbbb", created_at="2026-05-25T00:00:00Z"),
            _job_state(id="cccccccccccc", created_at="2026-03-15T00:00:00Z"),
        ])
        assert [r["id"] for r in rows] == [
            "bbbbbbbbbbbb", "cccccccccccc", "aaaaaaaaaaaa",
        ]

    def test_missing_created_at_sinks_to_bottom(self) -> None:
        rows = library.summarise_jobs([
            _job_state(id="aaaaaaaaaaaa", created_at="2026-01-01T00:00:00Z"),
            _job_state(id="bbbbbbbbbbbb", created_at=""),
            _job_state(id="cccccccccccc", created_at="2026-03-15T00:00:00Z"),
        ])
        # First two are the timestamped jobs (newest first); bare row last.
        ids = [r["id"] for r in rows]
        assert ids[0] == "cccccccccccc"
        assert ids[1] == "aaaaaaaaaaaa"
        assert ids[2] == "bbbbbbbbbbbb"

    def test_deterministic_on_tie(self) -> None:
        rows = library.summarise_jobs([
            _job_state(id="aaaaaaaaaaaa", created_at="2026-05-25T00:00:00Z"),
            _job_state(id="bbbbbbbbbbbb", created_at="2026-05-25T00:00:00Z"),
            _job_state(id="cccccccccccc", created_at="2026-05-25T00:00:00Z"),
        ])
        # Tie-break is id descending so the order is always the same
        # regardless of input ordering.
        assert [r["id"] for r in rows] == [
            "cccccccccccc", "bbbbbbbbbbbb", "aaaaaaaaaaaa",
        ]

    def test_empty_input(self) -> None:
        assert library.summarise_jobs([]) == []


# --------------------------------------------------------------------------- #
# matches_query / filter_rows
# --------------------------------------------------------------------------- #


class TestMatchesQuery:
    @pytest.fixture
    def row(self) -> dict[str, Any]:
        return library.summarise_job(_job_state(
            input_filename="Interview-Maria-2025-04-12.wav",
            mode="diarize",
            language="en",
            model="large-v3",
            status="done",
            result=_result(speakers=["Luke", "Maria Gonzalez"]),
        ))

    def test_empty_query_matches(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "") is True
        assert library.matches_query(row, "   ") is True

    def test_matches_filename(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "interview") is True
        assert library.matches_query(row, "MARIA") is True       # case-insensitive
        assert library.matches_query(row, "2025") is True

    def test_matches_speaker(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "luke") is True
        assert library.matches_query(row, "gonzalez") is True

    def test_matches_status(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "done") is True

    def test_matches_mode(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "diarize") is True

    def test_matches_model(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "large-v3") is True

    def test_no_match(self, row: dict[str, Any]) -> None:
        assert library.matches_query(row, "nonexistent") is False
        assert library.matches_query(row, "queued") is False


class TestFilterRows:
    def test_preserves_input_order(self) -> None:
        # filter_rows mustn't sort; the caller controls order.
        rows = [
            library.summarise_job(_job_state(
                id="aaaaaaaaaaaa",
                input_filename="zzz.wav",
                created_at="2026-05-25T00:00:00Z",
            )),
            library.summarise_job(_job_state(
                id="bbbbbbbbbbbb",
                input_filename="aaa-zzz.wav",
                created_at="2026-01-01T00:00:00Z",
            )),
        ]
        out = library.filter_rows(rows, "zzz")
        assert [r["id"] for r in out] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]

    def test_empty_query_returns_all(self) -> None:
        rows = [library.summarise_job(_job_state(id="aaaaaaaaaaaa"))]
        assert library.filter_rows(rows, "") == rows

    def test_no_matches_returns_empty(self) -> None:
        rows = [library.summarise_job(_job_state(input_filename="a.wav"))]
        assert library.filter_rows(rows, "zzz") == []
