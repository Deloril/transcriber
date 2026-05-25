"""Tests for ``scribe.audit_export`` (F9.7).

Covers:

* :class:`AuditRow` shape + ``to_dict``.
* Summary helpers (``summary_for_event``, ``summary_for_invocation``).
* ``build_audit_trail`` aggregation, filtering, ordering.
* CSV / Markdown / RTF renderers — empty cases + populated cases.
* Format dispatch (``normalise_format``, ``render_audit_trail``,
  ``slugify_audit_trail_filename``, ``write_audit_trail``).

Pure unit tests. No FastAPI surface here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe import audit_export as ae
from scribe.ai_invocation_log import (
    DECISION_REQUEST_ONLY,
    InvocationLogEntry,
)
from scribe.ai_provenance import (
    AI_DECISION_ACCEPTED,
    AI_DECISION_REJECTED,
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
)
from scribe.code_suggestions import (
    CodeSuggestion,
    save_suggestion,
)
from scribe.coders import Coder, save_coder
from scribe.event_log import (
    EVENT_ACTION_CREATE,
    EVENT_ACTION_UPDATE,
    EVENT_ENTITY_CODE,
    EVENT_ENTITY_PROJECT,
    Event,
    record_create,
    record_event,
    record_update,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.quote_similarity import (
    QuoteSearch,
    save_quote_search,
)


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def saved_project(tmp_path: Path) -> tuple[Path, Project]:
    root = tmp_path / "projects"
    root.mkdir()
    p = Project.new(name="Pilot study", methodology="grounded theory")
    save_project(root, p)
    return root, p


@pytest.fixture
def saved_coder(saved_project: tuple[Path, Project]) -> Coder:
    root, project = saved_project
    coder = Coder.new(project_id=project.id, name="Researcher A")
    save_coder(root, coder)
    return coder


# --------------------------------------------------------------------------- #
# AuditRow
# --------------------------------------------------------------------------- #


class TestAuditRow:
    def test_default_field_values(self) -> None:
        r = ae.AuditRow(
            timestamp="2026-05-26T10:00:00Z",
            kind=ae.AUDIT_KIND_EVENT,
            record_id="abcdef012345",
        )
        assert r.actor_coder_id == ""
        assert r.action == ""
        assert r.entity_type == ""
        assert r.entity_id == ""
        assert r.summary == ""
        assert r.notes == ""
        assert r.extra == {}

    def test_to_dict_round_trip_preserves_fields(self) -> None:
        r = ae.AuditRow(
            timestamp="2026-05-26T10:00:00Z",
            kind=ae.AUDIT_KIND_EVENT,
            record_id="abcdef012345",
            actor_coder_id="0123456789ab",
            action="create",
            entity_type="code",
            entity_id="111111111111",
            summary="Created code — Suffering",
            notes="initial pass",
            extra={"after_label": "Suffering"},
        )
        d = r.to_dict()
        assert d["timestamp"] == "2026-05-26T10:00:00Z"
        assert d["kind"] == "event"
        assert d["record_id"] == "abcdef012345"
        assert d["actor_coder_id"] == "0123456789ab"
        assert d["action"] == "create"
        assert d["entity_type"] == "code"
        assert d["entity_id"] == "111111111111"
        assert d["summary"] == "Created code — Suffering"
        assert d["notes"] == "initial pass"
        assert d["extra"] == {"after_label": "Suffering"}

    def test_extra_dict_is_independent_per_row(self) -> None:
        r1 = ae.AuditRow(
            timestamp="2026-05-26T10:00:00Z",
            kind=ae.AUDIT_KIND_EVENT,
            record_id="abcdef012345",
        )
        r2 = ae.AuditRow(
            timestamp="2026-05-26T10:01:00Z",
            kind=ae.AUDIT_KIND_EVENT,
            record_id="abcdef012346",
        )
        # default_factory means each row gets its own dict — confirm
        # mutating one doesn't bleed into the other.
        r1.extra["k"] = "v"
        assert r2.extra == {}


# --------------------------------------------------------------------------- #
# AUDIT_KINDS vocabulary
# --------------------------------------------------------------------------- #


class TestAuditKinds:
    def test_kinds_tuple_contents(self) -> None:
        assert ae.AUDIT_KINDS == ("event", "ai_invocation")
        assert ae.AUDIT_KIND_EVENT == "event"
        assert ae.AUDIT_KIND_AI_INVOCATION == "ai_invocation"


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #


class TestSummaryForEvent:
    def test_basic_create_with_label(self) -> None:
        ev = Event.new(
            project_id="aaaaaaaaaaaa",
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            after={"name": "Suffering"},
        )
        s = ae.summary_for_event(ev)
        assert s.startswith("Create code")
        assert "Suffering" in s
        assert "111111111111" in s

    def test_update_uses_after_label_first(self) -> None:
        ev = Event.new(
            project_id="aaaaaaaaaaaa",
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            before={"name": "Old name"},
            after={"name": "New name"},
        )
        s = ae.summary_for_event(ev)
        assert "New name" in s
        assert "Old name" not in s

    def test_update_falls_back_to_before_label(self) -> None:
        ev = Event.new(
            project_id="aaaaaaaaaaaa",
            action="delete",
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            before={"name": "About to be deleted"},
        )
        s = ae.summary_for_event(ev)
        assert "About to be deleted" in s

    def test_no_label_no_dash_separator(self) -> None:
        ev = Event.new(
            project_id="aaaaaaaaaaaa",
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_PROJECT,
        )
        s = ae.summary_for_event(ev)
        assert " — " not in s

    def test_summary_truncates_overlong(self) -> None:
        long_name = "x" * (ae.MAX_SUMMARY_LEN + 50)
        ev = Event.new(
            project_id="aaaaaaaaaaaa",
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            after={"name": long_name},
        )
        s = ae.summary_for_event(ev)
        assert len(s) <= ae.MAX_SUMMARY_LEN
        assert s.endswith("…")


class TestSummaryForInvocation:
    def _entry(self, **overrides) -> InvocationLogEntry:
        defaults = dict(
            feature=AI_FEATURE_CODE_SUGGESTION,
            suggestion_id="abcdef012345",
            project_id="aaaaaaaaaaaa",
            created_at="2026-05-26T10:00:00Z",
            decision=AI_DECISION_ACCEPTED,
            summary="this is the user's quoted span",
        )
        defaults.update(overrides)
        return InvocationLogEntry(**defaults)

    def test_includes_feature_decision_and_inner_summary(self) -> None:
        entry = self._entry()
        s = ae.summary_for_invocation(entry)
        assert "AI code suggestion" in s
        assert AI_DECISION_ACCEPTED in s
        assert "this is the user's quoted span" in s

    def test_request_only_renders_decision_label(self) -> None:
        entry = self._entry(
            decision=DECISION_REQUEST_ONLY,
            summary="quote search by application",
        )
        s = ae.summary_for_invocation(entry)
        assert "request_only" in s
        assert "quote search by application" in s

    def test_empty_summary_drops_trailing_dash(self) -> None:
        entry = self._entry(summary="")
        s = ae.summary_for_invocation(entry)
        assert "—" not in s

    def test_truncates_overlong(self) -> None:
        entry = self._entry(summary="x" * (ae.MAX_SUMMARY_LEN + 100))
        s = ae.summary_for_invocation(entry)
        assert len(s) <= ae.MAX_SUMMARY_LEN
        assert s.endswith("…")


# --------------------------------------------------------------------------- #
# build_audit_trail
# --------------------------------------------------------------------------- #


class TestBuildAuditTrailEmpty:
    def test_no_events_no_invocations_returns_empty(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        rows = ae.build_audit_trail(root, project.id)
        assert rows == []

    def test_unknown_project_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError, match="project id"):
            ae.build_audit_trail(tmp_path, "not-hex")


class TestBuildAuditTrailOrdering:
    def test_events_emerge_chronologically(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        record_event(
            root,
            project_id=project.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            after={"name": "Resting"},
            now="2026-05-26T10:00:00Z",
        )
        record_event(
            root,
            project_id=project.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="222222222222",
            after={"name": "Pacing"},
            now="2026-05-26T09:00:00Z",
        )
        rows = ae.build_audit_trail(root, project.id)
        assert [r.entity_id for r in rows] == ["222222222222", "111111111111"]
        for r in rows:
            assert r.kind == ae.AUDIT_KIND_EVENT

    def test_mix_of_kinds_sorted_by_timestamp(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        record_event(
            root,
            project_id=project.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            after={"name": "Resting"},
            now="2026-05-26T10:30:00Z",
        )
        # F9.6 invocation via a CodeSuggestion record.
        s = CodeSuggestion.new(
            project_id=project.id,
            source_id="000000000001",
            anchor_start_word_id="s1w1",
            anchor_end_word_id="s1w2",
            query_text="some quoted text",
            now="2026-05-26T10:00:00Z",
        )
        save_suggestion(root, s)
        rows = ae.build_audit_trail(root, project.id)
        assert len(rows) == 2
        assert rows[0].kind == ae.AUDIT_KIND_AI_INVOCATION
        assert rows[1].kind == ae.AUDIT_KIND_EVENT
        assert rows[0].timestamp < rows[1].timestamp


class TestBuildAuditTrailFilters:
    def _populate(
        self, root: Path, project: Project, coder: Coder
    ) -> None:
        record_event(
            root,
            project_id=project.id,
            action=EVENT_ACTION_CREATE,
            entity_type=EVENT_ENTITY_CODE,
            entity_id="111111111111",
            actor_coder_id=coder.id,
            after={"name": "Resting"},
            now="2026-05-26T09:00:00Z",
        )
        record_event(
            root,
            project_id=project.id,
            action=EVENT_ACTION_UPDATE,
            entity_type=EVENT_ENTITY_PROJECT,
            entity_id=project.id,
            before={"name": "Pilot study"},
            after={"name": "Pilot v2"},
            now="2026-05-26T11:00:00Z",
        )
        s = CodeSuggestion.new(
            project_id=project.id,
            source_id="000000000001",
            anchor_start_word_id="s1w1",
            anchor_end_word_id="s1w2",
            query_text="quoted span A",
            now="2026-05-26T10:00:00Z",
        )
        save_suggestion(root, s)

    def test_kinds_event_only(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root, project.id, kinds=[ae.AUDIT_KIND_EVENT]
        )
        assert all(r.kind == ae.AUDIT_KIND_EVENT for r in rows)
        assert len(rows) == 2

    def test_kinds_ai_only(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root, project.id, kinds=[ae.AUDIT_KIND_AI_INVOCATION]
        )
        assert len(rows) == 1
        assert rows[0].kind == ae.AUDIT_KIND_AI_INVOCATION

    def test_kinds_invalid_raises(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        with pytest.raises(ProjectValidationError, match="kinds"):
            ae.build_audit_trail(root, project.id, kinds=["bogus"])

    def test_kinds_empty_raises(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        with pytest.raises(ProjectValidationError, match="kinds"):
            ae.build_audit_trail(root, project.id, kinds=[])

    def test_action_filter_applies_to_events_only(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root, project.id, action=EVENT_ACTION_CREATE
        )
        # The AI invocation row is *not* dropped — its 'action' field
        # is request_only / pending / accepted, and the action filter
        # is documented as F9.1-only (kinds gate is the source filter).
        kinds = [r.kind for r in rows]
        assert kinds.count(ae.AUDIT_KIND_EVENT) == 1
        assert kinds.count(ae.AUDIT_KIND_AI_INVOCATION) == 1

    def test_entity_type_filter_applies_to_events(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root, project.id, entity_type=EVENT_ENTITY_PROJECT
        )
        event_rows = [r for r in rows if r.kind == ae.AUDIT_KIND_EVENT]
        assert len(event_rows) == 1
        assert event_rows[0].entity_type == EVENT_ENTITY_PROJECT

    def test_actor_filter_keeps_only_that_humans_rows(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root, project.id, actor_coder_id=saved_coder.id
        )
        # Only the create-code event has actor_coder_id == saved_coder.id.
        assert len(rows) == 1
        assert rows[0].actor_coder_id == saved_coder.id
        assert rows[0].action == EVENT_ACTION_CREATE

    def test_actor_filter_invalid_raises(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        with pytest.raises(ProjectValidationError, match="actor_coder_id"):
            ae.build_audit_trail(
                root, project.id, actor_coder_id="not-hex"
            )

    def test_since_until_bounds(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        rows = ae.build_audit_trail(
            root,
            project.id,
            since="2026-05-26T10:00:00Z",
            until="2026-05-26T10:30:00Z",
        )
        assert len(rows) == 1
        assert rows[0].kind == ae.AUDIT_KIND_AI_INVOCATION
        assert rows[0].timestamp == "2026-05-26T10:00:00Z"

    def test_feature_filter_applies_to_invocations(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        self._populate(root, project, saved_coder)
        # Restrict to a feature that no row matches.
        rows = ae.build_audit_trail(
            root, project.id, feature=AI_FEATURE_QUOTE_SIMILARITY
        )
        invocation_rows = [
            r for r in rows if r.kind == ae.AUDIT_KIND_AI_INVOCATION
        ]
        assert invocation_rows == []
        # Events still come through.
        assert any(r.kind == ae.AUDIT_KIND_EVENT for r in rows)

    def test_decision_filter_keeps_only_matching_invocations(
        self, saved_project: tuple[Path, Project], saved_coder: Coder
    ) -> None:
        root, project = saved_project
        s = CodeSuggestion.new(
            project_id=project.id,
            source_id="000000000001",
            anchor_start_word_id="s1w1",
            anchor_end_word_id="s1w2",
            query_text="quoted A",
            now="2026-05-26T10:00:00Z",
        )
        from scribe.code_suggestions import record_decision
        record_decision(
            s,
            decision=AI_DECISION_REJECTED,
            coder_id=saved_coder.id,
            rejection_reason="not coded that way",
            now="2026-05-26T11:00:00Z",
        )
        save_suggestion(root, s)
        s2 = CodeSuggestion.new(
            project_id=project.id,
            source_id="000000000001",
            anchor_start_word_id="s1w3",
            anchor_end_word_id="s1w4",
            query_text="quoted B",
            now="2026-05-26T10:30:00Z",
        )
        save_suggestion(root, s2)
        rows = ae.build_audit_trail(
            root,
            project.id,
            kinds=[ae.AUDIT_KIND_AI_INVOCATION],
            decision=AI_DECISION_REJECTED,
        )
        assert len(rows) == 1
        assert rows[0].action == AI_DECISION_REJECTED
        assert "not coded that way" in rows[0].notes


class TestRowFromInvocationFields:
    def test_quote_search_uses_request_only_action(
        self, saved_project: tuple[Path, Project]
    ) -> None:
        root, project = saved_project
        q = QuoteSearch.new(
            project_id=project.id,
            query_kind="text",
            query_text="resting on the couch",
            top_k=5,
            now="2026-05-26T10:00:00Z",
        )
        save_quote_search(root, q)
        rows = ae.build_audit_trail(root, project.id)
        assert len(rows) == 1
        assert rows[0].action == DECISION_REQUEST_ONLY
        assert rows[0].entity_type == AI_FEATURE_QUOTE_SIMILARITY


# --------------------------------------------------------------------------- #
# Renderers — to_csv
# --------------------------------------------------------------------------- #


class TestToCsv:
    def test_empty_rows_yield_header_only(self) -> None:
        text = ae.to_csv([])
        lines = text.splitlines()
        assert lines == [",".join(ae.CSV_COLUMNS)]

    def test_csv_uses_crlf_line_terminator(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="111111111111",
                action="create",
                entity_type="code",
                summary="Create code — Resting",
            )
        ]
        text = ae.to_csv(rows)
        assert "\r\n" in text
        assert text.count("\r\n") >= 2  # header + one row

    def test_csv_columns_in_documented_order(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="r" * 12,
                actor_coder_id="a" * 12,
                action="create",
                entity_type="code",
                entity_id="e" * 12,
                summary="Created something",
                notes="nothing to see",
            )
        ]
        text = ae.to_csv(rows)
        lines = text.split("\r\n")
        assert lines[0].split(",") == list(ae.CSV_COLUMNS)
        # Each cell is in the right column.
        body = lines[1].split(",")
        assert body[0] == "2026-05-26T10:00:00Z"
        assert body[1] == "event"
        assert body[2] == "r" * 12

    def test_csv_quotes_special_characters(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="111111111111",
                summary='line with, comma "and" quote',
            )
        ]
        text = ae.to_csv(rows)
        assert '"line with, comma ""and"" quote"' in text


# --------------------------------------------------------------------------- #
# Renderers — to_markdown
# --------------------------------------------------------------------------- #


class TestToMarkdown:
    def test_empty_rows_with_no_project(self) -> None:
        out = ae.to_markdown([])
        assert out.startswith("# Audit trail")
        assert "(no audit-trail entries)" in out

    def test_empty_rows_with_project_meta(self) -> None:
        p = Project.new(name="Alpha", methodology="grounded theory")
        out = ae.to_markdown([], project=p)
        assert "# Audit trail — Alpha" in out
        assert "**Methodology**: grounded theory" in out
        assert "**Rows**: 0" in out

    def test_rows_grouped_by_day(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T09:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="r" * 12,
                action="create",
                entity_type="code",
                summary="Created code A",
            ),
            ae.AuditRow(
                timestamp="2026-05-26T11:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="s" * 12,
                action="create",
                entity_type="code",
                summary="Created code B",
            ),
            ae.AuditRow(
                timestamp="2026-05-27T09:00:00Z",
                kind=ae.AUDIT_KIND_AI_INVOCATION,
                record_id="t" * 12,
                action="accepted",
                entity_type="code_suggestion",
                summary="AI code suggestion: accepted",
            ),
        ]
        out = ae.to_markdown(rows)
        assert "## 2026-05-26" in out
        assert "## 2026-05-27" in out
        # Day header appears before its rows.
        idx_day1 = out.index("## 2026-05-26")
        idx_day2 = out.index("## 2026-05-27")
        assert idx_day1 < idx_day2
        assert out.index("Created code A") > idx_day1
        assert out.index("Created code A") < idx_day2

    def test_row_line_includes_actor_and_record_id(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                actor_coder_id="0123456789ab",
                action="create",
                entity_type="code",
                summary="Created code A",
            )
        ]
        out = ae.to_markdown(rows)
        assert "actor=0123456789ab" in out
        assert "id=abcdef012345" in out
        # Time-of-day appears.
        assert "10:00:00" in out

    def test_notes_render_as_blockquote(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                action="unlock",
                entity_type="codebook",
                summary="Unlock codebook",
                notes="Reopening for axial pass — see methods §3.4.",
            )
        ]
        out = ae.to_markdown(rows)
        assert "> Reopening for axial pass" in out

    def test_undated_rows_grouped_under_undated_heading(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                action="create",
                entity_type="code",
                summary="No timestamp",
            )
        ]
        out = ae.to_markdown(rows)
        assert "## Undated" in out


# --------------------------------------------------------------------------- #
# Renderers — to_rtf
# --------------------------------------------------------------------------- #


class TestToRtf:
    def test_empty_rows_returns_minimal_rtf(self) -> None:
        out = ae.to_rtf([])
        assert out.startswith(r"{\rtf1\ansi\ansicpg1252\deff0")
        assert out.endswith("}")
        assert "(no audit-trail entries)" in out
        assert "Audit trail" in out

    def test_project_metadata_appears(self) -> None:
        p = Project.new(name="Alpha", methodology="grounded theory")
        out = ae.to_rtf([], project=p)
        assert "Audit trail \\u8212?" in out  # em-dash escape
        assert "Alpha" in out
        assert "Methodology: grounded theory" in out

    def test_rows_emit_day_headings_and_lines(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T09:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                action="create",
                entity_type="code",
                summary="Created code A",
            ),
            ae.AuditRow(
                timestamp="2026-05-27T09:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="bcdef0123456",
                action="create",
                entity_type="code",
                summary="Created code B",
            ),
        ]
        out = ae.to_rtf(rows)
        assert "2026-05-26" in out
        assert "2026-05-27" in out
        assert "Created code A" in out
        assert "Created code B" in out

    def test_notes_render_as_indented_line(self) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                action="unlock",
                entity_type="codebook",
                summary="Unlock codebook",
                notes="Reopening for axial pass.",
            )
        ]
        out = ae.to_rtf(rows)
        assert "note: Reopening for axial pass." in out


# --------------------------------------------------------------------------- #
# Format dispatch
# --------------------------------------------------------------------------- #


class TestNormaliseFormat:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("csv", ae.EXPORT_FORMAT_CSV),
            ("CSV", ae.EXPORT_FORMAT_CSV),
            (" csv ", ae.EXPORT_FORMAT_CSV),
            ("md", ae.EXPORT_FORMAT_MARKDOWN),
            ("markdown", ae.EXPORT_FORMAT_MARKDOWN),
            ("rtf", ae.EXPORT_FORMAT_RTF),
            ("Word", ae.EXPORT_FORMAT_RTF),
            ("doc", ae.EXPORT_FORMAT_RTF),
            ("docx", ae.EXPORT_FORMAT_RTF),
        ],
    )
    def test_aliases_resolve(self, raw: str, expected: str) -> None:
        assert ae.normalise_format(raw) == expected

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="format is required"):
            ae.normalise_format(None)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            ae.normalise_format("yaml")


class TestRenderAuditTrail:
    def test_csv_dispatch(self) -> None:
        text = ae.render_audit_trail(ae.EXPORT_FORMAT_CSV, [])
        assert text.startswith(",".join(ae.CSV_COLUMNS))

    def test_markdown_dispatch(self) -> None:
        text = ae.render_audit_trail(ae.EXPORT_FORMAT_MARKDOWN, [])
        assert text.startswith("# Audit trail")

    def test_rtf_dispatch(self) -> None:
        text = ae.render_audit_trail(ae.EXPORT_FORMAT_RTF, [])
        assert text.startswith(r"{\rtf1")


class TestSlugifyAuditTrailFilename:
    def test_no_project(self) -> None:
        for fmt, ext in (
            (ae.EXPORT_FORMAT_CSV, ".csv"),
            (ae.EXPORT_FORMAT_MARKDOWN, ".md"),
            (ae.EXPORT_FORMAT_RTF, ".rtf"),
        ):
            assert ae.slugify_audit_trail_filename(None, fmt) == (
                f"audit-trail{ext}"
            )

    def test_project_name_becomes_slug(self) -> None:
        p = Project.new(name="My Pilot Study")
        out = ae.slugify_audit_trail_filename(p, ae.EXPORT_FORMAT_MARKDOWN)
        assert out == "my-pilot-study-audit-trail.md"

    def test_diacritics_downgraded(self) -> None:
        p = Project.new(name="Élise — projet")
        out = ae.slugify_audit_trail_filename(p, ae.EXPORT_FORMAT_CSV)
        assert out == "elise-projet-audit-trail.csv"

    def test_unknown_format_raises(self) -> None:
        p = Project.new(name="X")
        with pytest.raises(ValueError):
            ae.slugify_audit_trail_filename(p, "bogus")


class TestWriteAuditTrail:
    def test_writes_to_disk_atomically(self, tmp_path: Path) -> None:
        rows = [
            ae.AuditRow(
                timestamp="2026-05-26T10:00:00Z",
                kind=ae.AUDIT_KIND_EVENT,
                record_id="abcdef012345",
                action="create",
                entity_type="code",
                summary="Created code A",
            )
        ]
        target = tmp_path / "out" / "audit.csv"
        result = ae.write_audit_trail(target, ae.EXPORT_FORMAT_CSV, rows)
        assert result == target
        assert target.exists()
        body = target.read_bytes()
        assert b"abcdef012345" in body
        # Tmp swap left no half-written sibling.
        assert not (tmp_path / "out" / "audit.csv.tmp").exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "tree" / "audit.md"
        ae.write_audit_trail(target, ae.EXPORT_FORMAT_MARKDOWN, [])
        assert target.exists()

    def test_unknown_format_raises_before_writing(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "audit.bogus"
        with pytest.raises(ValueError):
            ae.write_audit_trail(target, "bogus", [])
        assert not target.exists()
