"""Tests for ``scribe.definition_at_apply`` (F9.2).

Exercise the audit-trail surface of code-definition versioning:

  1. Code reconstruction from a CodeVersion snapshot.
  2. Drift detection against the current Code state.
  3. On-disk lookup of the version-at-apply, including the missing
     case (no version log, missing version id).
  4. Row builders that hydrate every application with its at-apply
     snapshot + drift information.
  5. CSV / Markdown / RTF rendering, format dispatch + alias handling.
  6. Filename slug + atomic disk write.

Pure helpers don't touch the filesystem; the disk-touching helpers
(``lookup_definition_at_apply``, ``build_definition_at_apply_rows``,
``write_report``) use ``tmp_path`` so the suite stays hermetic.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest

from scribe.applications import Application
from scribe.codes import Code, save_code
from scribe.code_versions import (
    CodeVersion,
    record_code_version,
    save_code_with_version,
)
from scribe.definition_at_apply import (
    CSV_COLUMNS,
    CSV_LIST_SEP,
    DefinitionAtApply,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_MARKDOWN,
    EXPORT_FORMAT_RTF,
    EXPORT_FORMATS,
    PLACEHOLDER_NO_CURRENT,
    PLACEHOLDER_NO_SNAPSHOT,
    build_definition_at_apply_rows,
    code_from_version_snapshot,
    drifted_definition_fields,
    lookup_definition_at_apply,
    normalise_format,
    render_report,
    slugify_report_filename,
    to_csv,
    to_markdown,
    to_rtf,
    write_report,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


PROJ_ID = "abcdef012345"
SRC_A = "111111111111"
CODE_A = "333333333333"
CODE_B = "444444444444"
CODER_A = "555555555555"
APP_1 = "999999999991"
APP_2 = "999999999992"
APP_3 = "999999999993"
DEF_VER_A = "a" * 12
DEF_VER_B = "b" * 12
DEF_VER_MISSING = "c" * 12


def _project(**overrides: Any) -> Project:
    payload: dict[str, Any] = {
        "name": "Pacing study",
        "methodology": "charmaz",
        "project_id": PROJ_ID,
        "now": "2024-01-01T00:00:00.000000Z",
    }
    payload.update(overrides)
    return Project.new(**payload)


def _saved_project(tmp_path: Path, **overrides: Any) -> Project:
    p = _project(**overrides)
    save_project(tmp_path, p)
    return p


def _code(
    code_id: str = CODE_A,
    *,
    name: str = "Pacing",
    project_id: str = PROJ_ID,
    definition: str = "How participants describe being depleted.",
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    exemplars: list[str] | None = None,
    theoretical_memo: str = "",
) -> Code:
    return Code.new(
        project_id=project_id,
        name=name,
        code_id=code_id,
        definition=definition,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        exemplars=exemplars or [],
        theoretical_memo=theoretical_memo,
        now="2024-01-01T00:00:00.000000Z",
    )


def _application(
    *,
    application_id: str = APP_1,
    code_id: str = CODE_A,
    source_id: str = SRC_A,
    coder_id: str = CODER_A,
    start_word: str = "s0w0",
    end_word: str = "s0w2",
    version_id: str = DEF_VER_A,
    project_id: str = PROJ_ID,
    now: str = "2024-01-02T00:00:00.000000Z",
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start_word,
        anchor_end_word_id=end_word,
        definition_version_id_at_apply=version_id,
        application_id=application_id,
        now=now,
    )


# --------------------------------------------------------------------------- #
# code_from_version_snapshot
# --------------------------------------------------------------------------- #


class TestCodeFromVersionSnapshot:
    def test_round_trip(self) -> None:
        c = _code(definition="def v1")
        v = CodeVersion.new(
            code=c, version=1, version_id=DEF_VER_A,
            now="2024-01-02T00:00:00.000000Z",
        )
        rebuilt = code_from_version_snapshot(v)
        assert rebuilt.id == c.id
        assert rebuilt.project_id == c.project_id
        assert rebuilt.name == c.name
        assert rebuilt.definition == "def v1"

    def test_rejects_non_codeversion(self) -> None:
        with pytest.raises(TypeError):
            code_from_version_snapshot("not a CodeVersion")  # type: ignore[arg-type]

    def test_rejects_non_dict_snapshot(self) -> None:
        c = _code()
        v = CodeVersion.new(
            code=c, version=1, version_id=DEF_VER_A,
            now="2024-01-02T00:00:00.000000Z",
        )
        # Hack the snapshot to a non-dict to simulate a malformed log.
        object.__setattr__(v, "snapshot", "not a dict")
        with pytest.raises(ProjectValidationError):
            code_from_version_snapshot(v)


# --------------------------------------------------------------------------- #
# drifted_definition_fields
# --------------------------------------------------------------------------- #


class TestDriftedDefinitionFields:
    def test_no_drift_when_identical(self) -> None:
        c = _code(definition="d", inclusion_criteria="i")
        snapshot = c.to_dict()
        assert drifted_definition_fields(snapshot, c) == ()

    def test_definition_change_detected(self) -> None:
        old = _code(definition="old")
        snapshot = old.to_dict()
        new = _code(definition="new")
        drifted = drifted_definition_fields(snapshot, new)
        assert "definition" in drifted

    def test_metadata_only_change_does_not_drift(self) -> None:
        # ``status`` and ``colour`` are not in DEFINITION_FIELDS.
        c1 = Code.new(
            project_id=PROJ_ID, name="X", code_id=CODE_A,
            now="2024-01-01T00:00:00.000000Z", status="active",
            colour="#abcdef",
        )
        snapshot = c1.to_dict()
        c2 = Code.new(
            project_id=PROJ_ID, name="X", code_id=CODE_A,
            now="2024-01-01T00:00:00.000000Z", status="retired",
            colour="#123456",
        )
        assert drifted_definition_fields(snapshot, c2) == ()

    def test_exemplars_added_drifts(self) -> None:
        old = _code()
        snapshot = old.to_dict()
        new = _code(exemplars=["I felt drained."])
        assert "exemplars" in drifted_definition_fields(snapshot, new)

    def test_missing_snapshot_treated_as_all_drifted(self) -> None:
        c = _code()
        result = drifted_definition_fields(None, c)
        assert "name" in result and "definition" in result

    def test_missing_current_treated_as_all_drifted(self) -> None:
        c = _code()
        snapshot = c.to_dict()
        result = drifted_definition_fields(snapshot, None)
        assert "name" in result and "definition" in result

    def test_returns_tuple_in_definition_fields_order(self) -> None:
        old = _code(definition="x")
        snapshot = old.to_dict()
        new = _code(definition="y", exemplars=["a"])
        drifted = drifted_definition_fields(snapshot, new)
        # Order matches DEFINITION_FIELDS ("definition" before "exemplars").
        assert drifted == ("definition", "exemplars")

    def test_empty_exemplars_compares_equal_to_missing_exemplars(self) -> None:
        c = _code()
        snapshot = c.to_dict()
        snapshot.pop("exemplars", None)  # simulate older snapshot
        # Current has empty exemplars list.
        assert "exemplars" not in drifted_definition_fields(snapshot, c)


# --------------------------------------------------------------------------- #
# lookup_definition_at_apply
# --------------------------------------------------------------------------- #


class TestLookupDefinitionAtApply:
    def test_returns_recorded_version(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        c = _code()
        save_code(tmp_path, c)
        v = record_code_version(
            tmp_path, c, now="2024-01-02T00:00:00.000000Z"
        )
        app = _application(version_id=v.id)
        looked_up = lookup_definition_at_apply(tmp_path, app)
        assert looked_up is not None
        assert looked_up.id == v.id
        assert looked_up.version == 1

    def test_returns_none_when_log_missing(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        # No version recorded — log dir doesn't exist for this code.
        app = _application(version_id=DEF_VER_A)
        assert lookup_definition_at_apply(tmp_path, app) is None

    def test_returns_none_when_version_id_missing_from_log(
        self, tmp_path: Path
    ) -> None:
        _saved_project(tmp_path)
        c = _code()
        save_code(tmp_path, c)
        record_code_version(tmp_path, c, now="2024-01-02T00:00:00.000000Z")
        # App points at a different (unknown) version id.
        app = _application(version_id=DEF_VER_MISSING)
        assert lookup_definition_at_apply(tmp_path, app) is None

    def test_rejects_non_application(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            lookup_definition_at_apply(tmp_path, "not an app")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# build_definition_at_apply_rows
# --------------------------------------------------------------------------- #


def _record_v1_v2(
    tmp_path: Path,
    code_id: str = CODE_A,
    *,
    v1_def: str = "early definition",
    v2_def: str = "sharper definition",
) -> tuple[CodeVersion, CodeVersion]:
    """Save a code, then record two versions with different definitions.

    Returns ``(v1, v2)``. Each call returns fresh ids — mirrors how the
    real workflow accumulates revisions during focused coding.
    """
    c = _code(code_id=code_id, definition=v1_def)
    save_code(tmp_path, c)
    v1 = record_code_version(
        tmp_path, c, now="2024-01-02T00:00:00.000000Z"
    )

    # Bump the definition and record again.
    c.definition = v2_def
    save_code(tmp_path, c)
    v2 = record_code_version(
        tmp_path, c, now="2024-01-03T00:00:00.000000Z"
    )
    return v1, v2


class TestBuildRows:
    def test_minimal_row_no_codes_supplied(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        v1, _ = _record_v1_v2(tmp_path)
        app = _application(version_id=v1.id)

        rows = build_definition_at_apply_rows(tmp_path, [app])
        assert len(rows) == 1
        r = rows[0]
        assert r.application_id == APP_1
        assert r.code_id == CODE_A
        assert r.version_id_at_apply == v1.id
        assert r.version_number_at_apply == 1
        assert r.definition_at_apply == "early definition"
        assert r.snapshot_missing is False
        assert r.code_missing is True  # no current code supplied
        # Drift defaults to all fields when current code is missing.
        assert "definition" in r.drifted_fields
        assert r.definition_drifted is True

    def test_drift_detected_against_current_code(
        self, tmp_path: Path
    ) -> None:
        _saved_project(tmp_path)
        v1, v2 = _record_v1_v2(tmp_path)
        # Application made under v1, current code is at v2 state.
        app = _application(version_id=v1.id)

        # Reconstruct "current" from v2 snapshot for the drift check.
        current = code_from_version_snapshot(v2)
        rows = build_definition_at_apply_rows(
            tmp_path, [app], codes=[current]
        )
        r = rows[0]
        assert r.snapshot_missing is False
        assert r.code_missing is False
        assert r.definition_drifted is True
        assert r.drifted_fields == ("definition",)
        assert r.definition_at_apply == "early definition"
        assert r.current_definition == "sharper definition"

    def test_no_drift_when_application_at_current_version(
        self, tmp_path: Path
    ) -> None:
        _saved_project(tmp_path)
        v1, v2 = _record_v1_v2(tmp_path)
        # Application made under v2, current code matches v2.
        app = _application(version_id=v2.id)
        current = code_from_version_snapshot(v2)
        rows = build_definition_at_apply_rows(
            tmp_path, [app], codes=[current]
        )
        r = rows[0]
        assert r.snapshot_missing is False
        assert r.code_missing is False
        assert r.definition_drifted is False
        assert r.drifted_fields == ()

    def test_missing_snapshot_marked(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        # No version recorded → log missing → row is snapshot_missing.
        app = _application(version_id=DEF_VER_A)
        rows = build_definition_at_apply_rows(tmp_path, [app])
        r = rows[0]
        assert r.snapshot_missing is True
        assert r.version_number_at_apply == 0
        assert r.definition_at_apply == ""
        # When the snapshot is missing we don't advertise drift.
        assert r.definition_drifted is False

    def test_caches_version_lookup_per_code_version_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_project(tmp_path)
        v1, _ = _record_v1_v2(tmp_path)
        app1 = _application(application_id=APP_1, version_id=v1.id)
        app2 = _application(application_id=APP_2, version_id=v1.id)

        calls: list[tuple[str, str]] = []

        from scribe import definition_at_apply as mod

        original = mod.lookup_definition_at_apply

        def counting(projects_root: Path, application: Application) -> Any:
            calls.append(
                (application.code_id, application.definition_version_id_at_apply)
            )
            return original(projects_root, application)

        monkeypatch.setattr(mod, "lookup_definition_at_apply", counting)

        rows = build_definition_at_apply_rows(tmp_path, [app1, app2])
        # Two applications, same (code, version) → one disk read.
        assert len(rows) == 2
        assert len(calls) == 1

    def test_rejects_non_application_input(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            build_definition_at_apply_rows(tmp_path, ["not an app"])  # type: ignore[list-item]

    def test_rejects_non_code_in_codes(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        v1, _ = _record_v1_v2(tmp_path)
        app = _application(version_id=v1.id)
        with pytest.raises(TypeError):
            build_definition_at_apply_rows(
                tmp_path, [app], codes=["not a Code"]  # type: ignore[list-item]
            )

    def test_preserves_input_order(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        v1, _ = _record_v1_v2(tmp_path)
        a = _application(application_id=APP_1, version_id=v1.id)
        b = _application(application_id=APP_2, version_id=v1.id)
        c = _application(application_id=APP_3, version_id=v1.id)
        rows = build_definition_at_apply_rows(tmp_path, [b, c, a])
        assert [r.application_id for r in rows] == [APP_2, APP_3, APP_1]

    def test_exemplars_carried_through_snapshot(
        self, tmp_path: Path
    ) -> None:
        _saved_project(tmp_path)
        c = _code(exemplars=["I felt drained.", "Just so tired."])
        save_code(tmp_path, c)
        v = record_code_version(
            tmp_path, c, now="2024-01-02T00:00:00.000000Z"
        )
        app = _application(version_id=v.id)
        rows = build_definition_at_apply_rows(tmp_path, [app])
        assert rows[0].exemplars_at_apply == (
            "I felt drained.",
            "Just so tired.",
        )


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _row(
    *,
    application_id: str = APP_1,
    code_id: str = CODE_A,
    version_id: str = DEF_VER_A,
    version_number: int = 1,
    name_at_apply: str = "Pacing",
    definition_at_apply: str = "early",
    exemplars_at_apply: tuple[str, ...] = (),
    current_name: str = "Pacing",
    current_definition: str = "early",
    current_exemplars: tuple[str, ...] = (),
    snapshot_missing: bool = False,
    code_missing: bool = False,
    drifted_fields: tuple[str, ...] = (),
    **overrides: Any,
) -> DefinitionAtApply:
    payload: dict[str, Any] = dict(
        application_id=application_id,
        code_id=code_id,
        source_id=SRC_A,
        coder_id=CODER_A,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w2",
        application_created_at="2024-01-02T00:00:00.000000Z",
        version_id_at_apply=version_id,
        version_number_at_apply=version_number,
        version_recorded_at="2024-01-02T00:00:00.000000Z" if not snapshot_missing else "",
        version_change_note="",
        name_at_apply=name_at_apply,
        definition_at_apply=definition_at_apply,
        inclusion_criteria_at_apply="",
        exclusion_criteria_at_apply="",
        exemplars_at_apply=exemplars_at_apply,
        theoretical_memo_at_apply="",
        current_name=current_name,
        current_definition=current_definition,
        current_inclusion_criteria="",
        current_exclusion_criteria="",
        current_exemplars=current_exemplars,
        current_theoretical_memo="",
        snapshot_missing=snapshot_missing,
        code_missing=code_missing,
        definition_drifted=bool(drifted_fields) and not snapshot_missing,
        drifted_fields=drifted_fields,
    )
    payload.update(overrides)
    return DefinitionAtApply(**payload)


class TestToCsv:
    def test_header_only_for_empty(self) -> None:
        out = to_csv([])
        reader = csv.reader(io.StringIO(out))
        rows = list(reader)
        assert len(rows) == 1
        assert tuple(rows[0]) == CSV_COLUMNS

    def test_round_trip(self) -> None:
        r = _row()
        out = to_csv([r])
        reader = csv.DictReader(io.StringIO(out))
        records = list(reader)
        assert len(records) == 1
        rec = records[0]
        assert rec["application_id"] == APP_1
        assert rec["version_id_at_apply"] == DEF_VER_A
        assert rec["version_number_at_apply"] == "1"
        assert rec["name_at_apply"] == "Pacing"
        assert rec["snapshot_missing"] == "false"
        assert rec["code_missing"] == "false"
        assert rec["definition_drifted"] == "false"

    def test_drifted_fields_joined(self) -> None:
        r = _row(
            drifted_fields=("name", "definition"),
            current_name="Resting",
            current_definition="late",
        )
        out = to_csv([r])
        reader = csv.DictReader(io.StringIO(out))
        rec = next(reader)
        assert rec["drifted_fields"] == "name" + CSV_LIST_SEP + "definition"
        assert rec["definition_drifted"] == "true"

    def test_exemplars_joined(self) -> None:
        r = _row(
            exemplars_at_apply=("a", "b"),
            current_exemplars=("a", "b", "c"),
        )
        out = to_csv([r])
        reader = csv.DictReader(io.StringIO(out))
        rec = next(reader)
        assert rec["exemplars_at_apply"] == "a" + CSV_LIST_SEP + "b"
        assert rec["current_exemplars"] == "a" + CSV_LIST_SEP + "b" + CSV_LIST_SEP + "c"

    def test_missing_snapshot_renders_empties_and_zero(self) -> None:
        r = _row(
            snapshot_missing=True,
            version_number=0,
            name_at_apply="",
            definition_at_apply="",
        )
        out = to_csv([r])
        reader = csv.DictReader(io.StringIO(out))
        rec = next(reader)
        assert rec["snapshot_missing"] == "true"
        assert rec["version_number_at_apply"] == "0"
        assert rec["name_at_apply"] == ""

    def test_uses_crlf_line_endings(self) -> None:
        out = to_csv([_row()])
        # csv.writer with default dialect emits \r\n line terminators.
        assert "\r\n" in out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestToMarkdown:
    def test_empty_renders_placeholder(self) -> None:
        md = to_markdown([])
        assert "# Definition at apply" in md
        assert "_(no applications)_" in md

    def test_includes_project_name_and_metadata(self) -> None:
        proj = _project(name="Pacing study")
        md = to_markdown([_row()], project=proj)
        assert "# Definition at apply — Pacing study" in md
        assert "**Methodology**: charmaz" in md
        assert "**Applications**: 1" in md
        assert "**Drifted**: 0" in md

    def test_drift_block_shows_current_fields(self) -> None:
        r = _row(
            drifted_fields=("definition",),
            definition_at_apply="early",
            current_definition="late",
        )
        md = to_markdown([r])
        assert "Definition drift since apply" in md
        assert "early" in md
        assert "late" in md
        assert "`definition`" in md

    def test_unchanged_block_when_no_drift(self) -> None:
        r = _row(drifted_fields=())
        md = to_markdown([r])
        assert "Definition unchanged since apply." in md

    def test_missing_snapshot_renders_placeholder(self) -> None:
        r = _row(snapshot_missing=True, version_number=0)
        md = to_markdown([r])
        assert PLACEHOLDER_NO_SNAPSHOT in md

    def test_missing_current_renders_placeholder(self) -> None:
        r = _row(code_missing=True, drifted_fields=("name",))
        md = to_markdown([r])
        assert PLACEHOLDER_NO_CURRENT in md

    def test_application_heading_present(self) -> None:
        md = to_markdown([_row(application_id=APP_1)])
        assert f"### Application `{APP_1}`" in md

    def test_change_note_rendered(self) -> None:
        r = _row(version_change_note="Sharpened after focused pass")
        md = to_markdown([r])
        assert "Sharpened after focused pass" in md

    def test_exemplars_section_when_present(self) -> None:
        r = _row(exemplars_at_apply=("first exemplar",))
        md = to_markdown([r])
        assert "Exemplars at apply" in md
        assert "first exemplar" in md


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


class TestToRtf:
    def test_envelope_present(self) -> None:
        rtf = to_rtf([])
        assert rtf.startswith(r"{\rtf1")
        assert rtf.endswith("}")

    def test_application_heading(self) -> None:
        rtf = to_rtf([_row()])
        assert f"Application {APP_1}" in rtf

    def test_drift_section(self) -> None:
        r = _row(drifted_fields=("definition",), current_definition="late")
        rtf = to_rtf([r])
        assert "Definition drift since apply" in rtf
        assert "late" in rtf

    def test_unchanged_message(self) -> None:
        r = _row(drifted_fields=())
        rtf = to_rtf([r])
        assert "Definition unchanged since apply." in rtf

    def test_missing_snapshot_renders_explanation(self) -> None:
        r = _row(snapshot_missing=True, version_number=0)
        rtf = to_rtf([r])
        assert "snapshot not found" in rtf

    def test_project_metadata_rendered(self) -> None:
        proj = _project(name="P")
        rtf = to_rtf([_row()], project=proj)
        assert "Methodology: charmaz" in rtf
        assert "Applications: 1" in rtf

    def test_unicode_escaping_round_trip(self) -> None:
        # Curly quote should land as \uNNNN? not as raw bytes.
        r = _row(definition_at_apply="ladies’ night")
        rtf = to_rtf([r])
        # 0x2019 is positive in 16-bit signed -> 8217.
        assert "\\u8217?" in rtf
        # And the raw curly quote must NOT appear unescaped.
        assert "’" not in rtf


# --------------------------------------------------------------------------- #
# Format dispatch + filename + write
# --------------------------------------------------------------------------- #


class TestExportFormats:
    def test_canonical_keys_present(self) -> None:
        # The registry advertises exactly the three rendering targets.
        assert set(EXPORT_FORMATS.keys()) == {
            EXPORT_FORMAT_CSV,
            EXPORT_FORMAT_MARKDOWN,
            EXPORT_FORMAT_RTF,
        }

    def test_format_specs_carry_extension_and_media_type(self) -> None:
        spec = EXPORT_FORMATS[EXPORT_FORMAT_CSV]
        assert spec.extension == ".csv"
        assert spec.media_type.startswith("text/csv")
        assert spec.label == "CSV"


class TestNormaliseFormat:
    def test_canonical_keys(self) -> None:
        assert normalise_format("csv") == EXPORT_FORMAT_CSV
        assert normalise_format("markdown") == EXPORT_FORMAT_MARKDOWN
        assert normalise_format("rtf") == EXPORT_FORMAT_RTF

    def test_aliases(self) -> None:
        assert normalise_format("md") == EXPORT_FORMAT_MARKDOWN
        assert normalise_format("Word") == EXPORT_FORMAT_RTF
        assert normalise_format("docx") == EXPORT_FORMAT_RTF
        assert normalise_format("DOC") == EXPORT_FORMAT_RTF

    def test_strips_whitespace_case_insensitive(self) -> None:
        assert normalise_format("  CSV ") == EXPORT_FORMAT_CSV

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format(None)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_format("yaml")


class TestRenderReport:
    def test_csv_dispatch(self) -> None:
        out = render_report("csv", [_row()])
        # CSV starts with the header.
        assert out.split("\r\n", 1)[0].split(",")[0] == "application_id"

    def test_markdown_dispatch(self) -> None:
        out = render_report("md", [])
        assert out.startswith("# Definition at apply")

    def test_rtf_dispatch(self) -> None:
        out = render_report("rtf", [])
        assert out.startswith(r"{\rtf1")

    def test_csv_ignores_project(self) -> None:
        proj = _project(name="X")
        out = render_report("csv", [_row()], project=proj)
        # Project name must not appear in the CSV body.
        assert "X" not in out.splitlines()[0]


class TestSlugifyFilename:
    def test_with_project(self) -> None:
        proj = _project(name="My Pacing Study")
        assert (
            slugify_report_filename(proj, "csv")
            == "my-pacing-study-definition-at-apply.csv"
        )

    def test_without_project(self) -> None:
        assert (
            slugify_report_filename(None, "rtf")
            == "definition-at-apply.rtf"
        )

    def test_strips_non_ascii(self) -> None:
        proj = _project(name="Café")
        slug = slugify_report_filename(proj, "md")
        # NFKD strips the accent, "ascii" ignore drops anything that
        # didn't decompose.
        assert slug == "cafe-definition-at-apply.md"

    def test_none_project_falls_back(self) -> None:
        # Project must always have a non-blank name (validated by
        # Project.validate); the falls-back-to-bare-slug branch only
        # triggers when no project is supplied at all.
        assert (
            slugify_report_filename(None, "csv")
            == "definition-at-apply.csv"
        )

    def test_long_name_truncated(self) -> None:
        proj = _project(name="x" * 200)
        slug = slugify_report_filename(proj, "csv")
        # 80 char slug + suffix.
        assert slug.endswith("-definition-at-apply.csv")
        prefix = slug[: -len("-definition-at-apply.csv")]
        assert len(prefix) == 80


class TestWriteReport:
    def test_atomic_write_creates_parent(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "out.csv"
        write_report(target, "csv", [_row()])
        assert target.is_file()
        assert target.read_text().startswith("application_id")

    def test_writes_markdown(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        write_report(target, "md", [_row()], project=_project())
        body = target.read_text()
        assert body.startswith("# Definition at apply")

    def test_replaces_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        target.write_text("STALE")
        write_report(target, "csv", [_row()])
        assert "STALE" not in target.read_text()

    def test_no_temp_file_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        write_report(target, "csv", [_row()])
        assert not (tmp_path / "out.csv.tmp").exists()


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


class TestEndToEndScenario:
    def test_definition_drift_round_trip(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # v1: initial coding pass; v2: focused-coding sharpening.
        c = _code(definition="early definition")
        save_code(tmp_path, c)
        v1 = record_code_version(
            tmp_path, c, now="2024-01-02T00:00:00.000000Z",
            change_note="initial draft",
        )
        c.definition = "sharper definition"
        save_code(tmp_path, c)
        v2 = record_code_version(
            tmp_path, c, now="2024-01-03T00:00:00.000000Z",
            change_note="post-focused-pass refinement",
        )

        # Two applications: one made under v1, one under v2.
        app_v1 = _application(application_id=APP_1, version_id=v1.id)
        app_v2 = _application(application_id=APP_2, version_id=v2.id)

        rows = build_definition_at_apply_rows(
            tmp_path, [app_v1, app_v2], codes=[c],
        )
        assert len(rows) == 2

        # v1 row: definition drifted.
        r1 = rows[0]
        assert r1.application_id == APP_1
        assert r1.version_number_at_apply == 1
        assert r1.definition_at_apply == "early definition"
        assert r1.current_definition == "sharper definition"
        assert r1.definition_drifted is True

        # v2 row: no drift.
        r2 = rows[1]
        assert r2.application_id == APP_2
        assert r2.version_number_at_apply == 2
        assert r2.definition_at_apply == "sharper definition"
        assert r2.definition_drifted is False

        # The Markdown report surfaces the drift visibly.
        md = to_markdown(rows, project=proj)
        assert "early definition" in md
        assert "sharper definition" in md
        assert "Definition drift since apply" in md
        # And the CSV round-trips deterministically.
        csv_text = to_csv(rows)
        rd = csv.DictReader(io.StringIO(csv_text))
        recs = list(rd)
        assert recs[0]["definition_drifted"] == "true"
        assert recs[1]["definition_drifted"] == "false"

    def test_save_code_with_version_to_application_round_trip(
        self, tmp_path: Path
    ) -> None:
        # Mirrors the canonical workflow: save_code_with_version mints
        # v1; an application points at it; the report finds it.
        _saved_project(tmp_path)
        c = _code()
        path, recorded = save_code_with_version(tmp_path, c)
        assert recorded is not None
        app = _application(version_id=recorded.id)
        rows = build_definition_at_apply_rows(
            tmp_path, [app], codes=[c]
        )
        assert rows[0].snapshot_missing is False
        assert rows[0].version_number_at_apply == 1
        assert rows[0].definition_drifted is False
