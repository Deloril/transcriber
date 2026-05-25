"""Tests for scribe.application_provenance_display (F9.9).

Per PLANNING.md F9.9:

  > Per-application provenance display on hover.

This suite covers the pure-Python builder + formatters that turn an
:class:`Application` (and the related Code / CodeVersion / Coder) into
a structured display surface for the editor's hover tooltip:

* :func:`build_provenance_display` — happy paths plus all the
  "missing related entity" branches.
* :func:`provenance_summary_label` — compact one-line label.
* :func:`format_provenance_text` — plain-text title= rendering.
* :func:`format_provenance_html` — escaped HTML for innerHTML.

All tests are pure Python; no FastAPI, no engine, no disk I/O.
"""

from __future__ import annotations

import pytest

from scribe.applications import Application
from scribe.application_provenance_display import (
    AI_DECISION_LABELS,
    AI_FEATURE_LABELS,
    DEFAULT_PROVENANCE_SOURCE_LABEL,
    PROVENANCE_SOURCE_LABELS,
    ProvenanceDisplay,
    build_provenance_display,
    format_provenance_html,
    format_provenance_text,
    provenance_summary_label,
)
from scribe.ai_provenance import (
    AI_DECISION_ACCEPTED,
    AI_FEATURE_CODE_SUGGESTION,
    AIProvenance,
)
from scribe.coders import Coder
from scribe.codes import Code
from scribe.code_versions import CodeVersion


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


PROJECT_ID = "0" * 12
CODE_ID = "a" * 12
SOURCE_ID = "1" * 12
CODER_ID = "d" * 12
VERSION_ID = "e" * 12


def _hex(n: int) -> str:
    return f"{n:012x}"


def _make_application(
    *,
    coder_id: str = CODER_ID,
    code_id: str = CODE_ID,
    source_id: str = SOURCE_ID,
    project_id: str = PROJECT_ID,
    version_id: str = VERSION_ID,
    start: str = "s0w0",
    end: str = "s0w12",
    start_offset: int | None = None,
    end_offset: int | None = None,
    confidence: float | None = None,
    note: str = "",
    provenance: dict | None = None,
    ai_provenance: AIProvenance | None = None,
    application_id: str | None = None,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start,
        anchor_end_word_id=end,
        definition_version_id_at_apply=version_id,
        start_char_offset=start_offset,
        end_char_offset=end_offset,
        confidence=confidence,
        provenance=provenance,
        ai_provenance=ai_provenance,
        note=note,
        application_id=application_id,
    )


def _make_code(
    *,
    code_id: str = CODE_ID,
    name: str = "Negotiating identity",
    definition: str = "Initial def",
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    exemplars: list[str] | None = None,
    theoretical_memo: str = "",
    colour: str = "#aabbcc",
    stage: str = "initial",
) -> Code:
    return Code.new(
        project_id=PROJECT_ID,
        name=name,
        definition=definition,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        exemplars=list(exemplars or []),
        theoretical_memo=theoretical_memo,
        colour=colour,
        stage=stage,
        code_id=code_id,
    )


def _make_version(
    code: Code,
    *,
    version: int = 1,
    version_id: str = VERSION_ID,
    change_note: str = "",
    now: str = "2026-04-15T10:00:00Z",
) -> CodeVersion:
    return CodeVersion.new(
        code=code,
        version=version,
        version_id=version_id,
        change_note=change_note,
        now=now,
    )


def _make_coder(
    *,
    coder_id: str = CODER_ID,
    name: str = "Alex",
    role: str = "researcher",
) -> Coder:
    return Coder.new(
        project_id=PROJECT_ID,
        name=name,
        role=role,
        coder_id=coder_id,
    )


# --------------------------------------------------------------------------- #
# Builder happy paths
# --------------------------------------------------------------------------- #


class TestBuildBasic:
    def test_minimal_application_no_relations(self) -> None:
        """No related entities supplied → safe placeholders, no crash."""
        app = _make_application()
        d = build_provenance_display(app)
        assert isinstance(d, ProvenanceDisplay)
        assert d.application_id == app.id
        assert d.code_id == app.code_id
        assert d.code_name == "(unknown)"
        assert d.code_missing is True
        assert d.coder_name == "(unknown)"
        assert d.snapshot_missing is True
        assert d.version_number_at_apply == ""
        assert d.provenance_source == ""
        assert d.provenance_source_label == DEFAULT_PROVENANCE_SOURCE_LABEL
        assert d.ai_present is False
        assert d.drifted_fields == ()
        assert d.definition_drifted is False

    def test_with_code_version_and_coder(self) -> None:
        """All relations supplied → fully hydrated display."""
        app = _make_application()
        code = _make_code()
        version = _make_version(code, version=3)
        coder = _make_coder()
        d = build_provenance_display(
            app, code=code, code_version=version, coder=coder
        )
        assert d.code_name == "Negotiating identity"
        assert d.code_colour == "#aabbcc"
        assert d.code_stage == "initial"
        assert d.coder_name == "Alex"
        assert d.coder_role == "researcher"
        assert d.version_number_at_apply == "v3"
        assert d.version_recorded_at == "2026-04-15T10:00:00Z"
        assert d.snapshot_missing is False
        assert d.name_at_apply == "Negotiating identity"
        assert d.code_missing is False
        assert d.definition_drifted is False
        assert d.drifted_fields == ()

    def test_anchor_label_single_word_collapses(self) -> None:
        app = _make_application(start="s0w4", end="s0w4")
        d = build_provenance_display(app)
        assert d.anchor_label == "s0w4"

    def test_anchor_label_subword_offsets_force_range(self) -> None:
        app = _make_application(
            start="s0w4", end="s0w4", start_offset=0, end_offset=3
        )
        d = build_provenance_display(app)
        # Even though start == end word, the sub-word offsets mean
        # we still want to render the range form.
        assert d.anchor_label == "s0w4–s0w4"

    def test_anchor_label_multi_word(self) -> None:
        app = _make_application(start="s0w0", end="s1w7")
        d = build_provenance_display(app)
        assert d.anchor_label == "s0w0–s1w7"

    def test_source_name_supplied(self) -> None:
        app = _make_application()
        d = build_provenance_display(app, source_name="Interview 03")
        assert d.source_name == "Interview 03"

    def test_source_name_default_empty(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        assert d.source_name == ""


class TestProvenanceSourceLabels:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("human", "Human-coded"),
            ("ai_accepted", "AI-suggested · accepted"),
            ("ai_modified", "AI-suggested · accepted with edits"),
            ("imported", "Imported"),
            ("other", "Other"),
        ],
    )
    def test_known_source(self, src: str, expected: str) -> None:
        app = _make_application(provenance={"source": src})
        d = build_provenance_display(app)
        assert d.provenance_source == src
        assert d.provenance_source_label == expected

    def test_missing_source_defaults_to_human(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        assert d.provenance_source == ""
        assert d.provenance_source_label == "Human-coded"

    def test_provenance_source_labels_cover_vocabulary(self) -> None:
        # Sanity check: every closed source has a label.
        from scribe.applications import APPLICATION_PROVENANCE_SOURCES

        for src in APPLICATION_PROVENANCE_SOURCES:
            assert src in PROVENANCE_SOURCE_LABELS

    def test_extra_provenance_keys_pass_through(self) -> None:
        app = _make_application(
            provenance={
                "source": "imported",
                "import_format": "qdpx",
                "import_run": "run-2026-04-12",
            }
        )
        d = build_provenance_display(app)
        # Reserved-ish keys (source / model_id / etc.) are extracted;
        # everything else lands in the extra block, sorted alphabetically.
        assert d.extra_provenance == (
            "import_format: qdpx",
            "import_run: run-2026-04-12",
        )

    def test_reserved_keys_filtered_from_extras(self) -> None:
        app = _make_application(
            provenance={
                "source": "ai_accepted",
                "model_id": "llama3.2:3b",
                "embedding_model": "bge-m3",
                "suggestion_id": "1" * 12,
                "accepted_at": "2026-04-15T10:00:00Z",
                "feature": "code_suggestion",
                "backend": "ollama",
                "custom": "kept",
            }
        )
        d = build_provenance_display(app)
        assert d.extra_provenance == ("custom: kept",)


# --------------------------------------------------------------------------- #
# Drift detection (relative to current Code)
# --------------------------------------------------------------------------- #


class TestDriftDetection:
    def test_no_drift_when_code_unchanged(self) -> None:
        app = _make_application()
        code = _make_code()
        version = _make_version(code)
        d = build_provenance_display(app, code=code, code_version=version)
        assert d.definition_drifted is False
        assert d.drifted_fields == ()

    def test_drift_detected_on_definition_change(self) -> None:
        app = _make_application()
        original = _make_code(definition="Initial def")
        version = _make_version(original)
        # Modify the in-memory Code to simulate a current state that
        # has drifted from the snapshot at apply.
        current = _make_code(definition="A revised, fuller definition.")
        d = build_provenance_display(app, code=current, code_version=version)
        assert d.definition_drifted is True
        assert "definition" in d.drifted_fields

    def test_drift_skipped_when_no_current_code(self) -> None:
        app = _make_application()
        code = _make_code()
        version = _make_version(code)
        d = build_provenance_display(app, code_version=version)
        assert d.code_missing is True
        assert d.definition_drifted is False
        assert d.drifted_fields == ()

    def test_drift_skipped_when_no_version(self) -> None:
        app = _make_application()
        code = _make_code()
        d = build_provenance_display(app, code=code)
        assert d.snapshot_missing is True
        assert d.definition_drifted is False
        assert d.drifted_fields == ()


# --------------------------------------------------------------------------- #
# AI provenance (F8.9)
# --------------------------------------------------------------------------- #


class TestAIProvenance:
    def test_no_ai_section_when_absent(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        assert d.ai_present is False
        assert d.ai_feature == ""
        assert d.ai_decision_label == ""
        assert d.ai_decided_by_coder_name == ""

    def test_ai_section_populated_when_present(self) -> None:
        aip = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            embedding_model="bge-m3",
            suggestion_id="1" * 12,
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=CODER_ID,
            decided_at="2026-04-15T10:00:00Z",
            confidence=0.82,
            prompt_hash="cafebabe1234",
            notes="Span tightened by reviewer.",
        )
        app = _make_application(ai_provenance=aip)
        d = build_provenance_display(app)
        assert d.ai_present is True
        assert d.ai_feature == AI_FEATURE_CODE_SUGGESTION
        assert d.ai_feature_label == AI_FEATURE_LABELS[AI_FEATURE_CODE_SUGGESTION]
        assert d.ai_backend == "ollama"
        assert d.ai_generation_model == "llama3.2:3b"
        assert d.ai_embedding_model == "bge-m3"
        assert d.ai_suggestion_id == "1" * 12
        assert d.ai_decision == AI_DECISION_ACCEPTED
        assert d.ai_decision_label == AI_DECISION_LABELS[AI_DECISION_ACCEPTED]
        assert d.ai_decided_by_coder_id == CODER_ID
        assert d.ai_decided_at == "2026-04-15T10:00:00Z"
        assert d.ai_confidence == "0.82"
        assert d.ai_prompt_hash == "cafebabe1234"
        assert d.ai_notes == "Span tightened by reviewer."
        # No coder supplied → "(unknown)" placeholder
        assert d.ai_decided_by_coder_name == "(unknown)"

    def test_ai_decided_by_coder_name_when_supplied(self) -> None:
        aip = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=CODER_ID,
        )
        app = _make_application(ai_provenance=aip)
        coder = _make_coder(name="Sam")
        d = build_provenance_display(app, decided_by_coder=coder)
        assert d.ai_decided_by_coder_name == "Sam"

    def test_ai_decided_by_blank_when_no_coder_id_recorded(self) -> None:
        # Pending decisions don't have a decider yet → name should be ""
        aip = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        app = _make_application(ai_provenance=aip)
        d = build_provenance_display(app)
        assert d.ai_present is True
        assert d.ai_decided_by_coder_id == ""
        assert d.ai_decided_by_coder_name == ""


# --------------------------------------------------------------------------- #
# Confidence formatting
# --------------------------------------------------------------------------- #


class TestConfidenceFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, ""),
            (0.0, "0.00"),
            (1.0, "1.00"),
            (0.823, "0.82"),
            (0.8265, "0.83"),
        ],
    )
    def test_confidence_render(self, value, expected) -> None:
        app = _make_application(confidence=value)
        d = build_provenance_display(app)
        assert d.confidence == expected


# --------------------------------------------------------------------------- #
# Builder validation
# --------------------------------------------------------------------------- #


class TestBuilderValidation:
    def test_rejects_non_application(self) -> None:
        with pytest.raises(TypeError):
            build_provenance_display({"id": "abc"})  # type: ignore[arg-type]

    def test_rejects_non_code(self) -> None:
        app = _make_application()
        with pytest.raises(TypeError):
            build_provenance_display(app, code="oops")  # type: ignore[arg-type]

    def test_rejects_non_codeversion(self) -> None:
        app = _make_application()
        with pytest.raises(TypeError):
            build_provenance_display(app, code_version="oops")  # type: ignore[arg-type]

    def test_rejects_non_coder(self) -> None:
        app = _make_application()
        with pytest.raises(TypeError):
            build_provenance_display(app, coder="oops")  # type: ignore[arg-type]

    def test_rejects_non_decided_by_coder(self) -> None:
        app = _make_application()
        with pytest.raises(TypeError):
            build_provenance_display(app, decided_by_coder=42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# provenance_summary_label
# --------------------------------------------------------------------------- #


class TestSummaryLabel:
    def test_full_summary(self) -> None:
        app = _make_application(provenance={"source": "human"})
        coder = _make_coder()
        d = build_provenance_display(app, coder=coder)
        s = provenance_summary_label(d)
        # Coder · provenance source label · ISO date prefix
        assert "Alex" in s
        assert "Human-coded" in s
        # created_at is an ISO timestamp; first 10 chars are the date.
        assert d.created_at[:10] in s

    def test_no_coder_skipped(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)  # no coder
        s = provenance_summary_label(d)
        assert "(unknown)" not in s
        assert "Human-coded" in s

    def test_blank_created_at_omits_date(self) -> None:
        app = _make_application()
        # Synthesise a display with empty created_at.
        d = ProvenanceDisplay(
            application_id=app.id,
            anchor_label=app.anchor_start_word_id,
            created_at="",
            modified_at="",
            confidence="",
            note="",
            code_id=app.code_id,
            code_name="X",
            code_colour="",
            code_stage="",
            version_id_at_apply=app.definition_version_id_at_apply,
            version_number_at_apply="",
            version_recorded_at="",
            version_change_note="",
            snapshot_missing=True,
            name_at_apply="",
            coder_id=app.coder_id,
            coder_name="Alex",
            coder_role="",
            source_id=app.source_id,
            source_name="",
            provenance_source="",
            provenance_source_label="Human-coded",
            ai_present=False,
            ai_feature="",
            ai_feature_label="",
            ai_backend="",
            ai_generation_model="",
            ai_embedding_model="",
            ai_suggestion_id="",
            ai_decision="",
            ai_decision_label="",
            ai_decided_by_coder_id="",
            ai_decided_by_coder_name="",
            ai_decided_at="",
            ai_confidence="",
            ai_prompt_hash="",
            ai_notes="",
            code_missing=True,
            definition_drifted=False,
            drifted_fields=(),
            extra_provenance=(),
        )
        s = provenance_summary_label(d)
        assert s == "Alex · Human-coded"


# --------------------------------------------------------------------------- #
# format_provenance_text
# --------------------------------------------------------------------------- #


class TestFormatText:
    def test_minimal_text(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        text = format_provenance_text(d)
        # Title row begins with "(unknown)" code name + (id)
        assert text.splitlines()[0].startswith("(unknown) (")
        assert "Human-coded" in text
        assert f"anchor {d.anchor_label}" in text

    def test_text_includes_drift_hint(self) -> None:
        app = _make_application()
        original = _make_code(definition="A")
        version = _make_version(original)
        current = _make_code(definition="B")
        d = build_provenance_display(app, code=current, code_version=version)
        text = format_provenance_text(d)
        assert "Definition has changed since apply" in text
        assert "definition" in text

    def test_text_includes_snapshot_missing_warning(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        text = format_provenance_text(d)
        assert "Definition snapshot at apply not found." in text

    def test_text_includes_ai_section(self) -> None:
        aip = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            decision=AI_DECISION_ACCEPTED,
            decided_at="2026-04-15T10:00:00Z",
            confidence=0.91,
        )
        app = _make_application(ai_provenance=aip)
        text = format_provenance_text(build_provenance_display(app))
        assert "AI: Code suggestion · ollama · llama3.2:3b · Accepted" in text
        assert "AI confidence 0.91" in text

    def test_text_includes_note(self) -> None:
        app = _make_application(note="Multi\nline\nnote")
        d = build_provenance_display(app)
        text = format_provenance_text(d)
        assert "Note:" in text
        assert "Multi" in text
        assert "line" in text

    def test_text_extra_provenance_block(self) -> None:
        app = _make_application(provenance={"source": "human", "x": "y"})
        d = build_provenance_display(app)
        text = format_provenance_text(d)
        assert "x: y" in text


# --------------------------------------------------------------------------- #
# format_provenance_html
# --------------------------------------------------------------------------- #


class TestFormatHtml:
    def test_minimal_html_well_formed(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)
        html = format_provenance_html(d)
        assert html.startswith('<div class="provenance-display">')
        assert html.endswith("</div>")
        # No raw user content — but provenance source label leaks through OK
        assert "Human-coded" in html

    def test_html_escapes_user_content(self) -> None:
        app = _make_application(note="<script>alert(1)</script>")
        d = build_provenance_display(app)
        html = format_provenance_html(d)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_html_escapes_code_name(self) -> None:
        app = _make_application()
        # Bypass Code.validate's name regex — manually craft the
        # display to verify escaping is unconditional.
        d = ProvenanceDisplay(
            application_id=app.id,
            anchor_label="s0w0",
            created_at="",
            modified_at="",
            confidence="",
            note="",
            code_id=app.code_id,
            code_name='<img onerror="x">',
            code_colour="",
            code_stage="",
            version_id_at_apply="",
            version_number_at_apply="",
            version_recorded_at="",
            version_change_note="",
            snapshot_missing=False,
            name_at_apply="",
            coder_id=app.coder_id,
            coder_name="",
            coder_role="",
            source_id=app.source_id,
            source_name="",
            provenance_source="",
            provenance_source_label="Human-coded",
            ai_present=False,
            ai_feature="",
            ai_feature_label="",
            ai_backend="",
            ai_generation_model="",
            ai_embedding_model="",
            ai_suggestion_id="",
            ai_decision="",
            ai_decision_label="",
            ai_decided_by_coder_id="",
            ai_decided_by_coder_name="",
            ai_decided_at="",
            ai_confidence="",
            ai_prompt_hash="",
            ai_notes="",
            code_missing=False,
            definition_drifted=False,
            drifted_fields=(),
            extra_provenance=(),
        )
        html = format_provenance_html(d)
        assert "<img onerror" not in html
        assert "&lt;img onerror=&quot;x&quot;&gt;" in html

    def test_html_includes_drift_section(self) -> None:
        app = _make_application()
        original = _make_code(definition="A")
        version = _make_version(original)
        current = _make_code(definition="B")
        d = build_provenance_display(app, code=current, code_version=version)
        html = format_provenance_html(d)
        assert 'class="provenance-drift"' in html
        assert "definition" in html

    def test_html_includes_ai_section(self) -> None:
        aip = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            decision=AI_DECISION_ACCEPTED,
            decided_at="2026-04-15T10:00:00Z",
            confidence=0.91,
        )
        app = _make_application(ai_provenance=aip)
        d = build_provenance_display(app)
        html = format_provenance_html(d)
        assert 'class="provenance-ai"' in html
        assert "ollama" in html
        assert "llama3.2:3b" in html
        assert "Accepted" in html

    def test_html_includes_swatch_when_colour_set(self) -> None:
        app = _make_application()
        code = _make_code(colour="#abc123")
        version = _make_version(code)
        d = build_provenance_display(app, code=code, code_version=version)
        html = format_provenance_html(d)
        assert "provenance-swatch" in html
        assert "#abc123" in html

    def test_html_no_swatch_when_colour_blank(self) -> None:
        app = _make_application()
        code = _make_code(colour="")
        version = _make_version(code)
        d = build_provenance_display(app, code=code, code_version=version)
        html = format_provenance_html(d)
        assert "provenance-swatch" not in html

    def test_html_extra_provenance_renders_as_dl(self) -> None:
        app = _make_application(provenance={"source": "human", "import_run": "r1"})
        d = build_provenance_display(app)
        html = format_provenance_html(d)
        assert 'class="provenance-extra"' in html
        assert "<dt>import_run</dt><dd>r1</dd>" in html

    def test_html_snapshot_missing_warning_shows(self) -> None:
        app = _make_application()
        d = build_provenance_display(app)  # no version → snapshot_missing
        html = format_provenance_html(d)
        assert 'class="provenance-warn"' in html


# --------------------------------------------------------------------------- #
# Vocabulary completeness — guard against drift between modules
# --------------------------------------------------------------------------- #


class TestVocabularyCompleteness:
    def test_all_ai_features_have_a_label(self) -> None:
        from scribe.ai_provenance import AI_FEATURES

        for f in AI_FEATURES:
            assert f in AI_FEATURE_LABELS

    def test_all_ai_decisions_have_a_label(self) -> None:
        from scribe.ai_provenance import AI_DECISIONS

        for dec in AI_DECISIONS:
            assert dec in AI_DECISION_LABELS

    def test_provenance_source_labels_match_application_vocabulary(self) -> None:
        from scribe.applications import APPLICATION_PROVENANCE_SOURCES

        for src in APPLICATION_PROVENANCE_SOURCES:
            assert src in PROVENANCE_SOURCE_LABELS
