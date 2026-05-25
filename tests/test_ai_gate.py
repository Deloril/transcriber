"""Tests for scribe.ai_gate (F8.10).

Covers:

  * AIGateConfig defaults / validation / dict round-trip.
  * is_application_human / count_hand_coded_sources.
  * evaluate_ai_gate: every reason branch.
  * evaluate_project_ai_gate: end-to-end through the disk.
  * load_ai_gate_config / store_ai_gate_config round-trip via Project.settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.ai_provenance import (
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURE_TRANSCRIPT_REVIEW,
    AIProvenance,
)
from scribe.applications import (
    Application,
    save_application,
)
from scribe.codes import (
    Code,
    save_code,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    save_project,
)
from scribe.ai_gate import (
    DEFAULT_MIN_CODES,
    DEFAULT_MIN_HAND_CODED_SOURCES,
    GATE_OVERRIDE_AUTO,
    GATE_OVERRIDE_FORCE_OFF,
    GATE_OVERRIDE_FORCE_ON,
    GATE_OVERRIDES,
    MAX_THRESHOLD,
    REASON_DISABLED,
    REASON_FEATURE_EXEMPT,
    REASON_FORCE_OFF,
    REASON_FORCE_ON,
    REASON_INSUFFICIENT_BOTH,
    REASON_INSUFFICIENT_CODES,
    REASON_INSUFFICIENT_HAND_CODED_SOURCES,
    REASON_THRESHOLD_MET,
    SETTING_AI_GATE,
    SETTING_AI_GATE_EXEMPT_FEATURES,
    SETTING_KEY_ENABLED,
    SETTING_KEY_MIN_CODES,
    SETTING_KEY_MIN_HAND_CODED_SOURCES,
    SETTING_KEY_OVERRIDE,
    AIGateConfig,
    AIGateStatus,
    count_hand_coded_sources,
    default_ai_gate_config,
    evaluate_ai_gate,
    evaluate_project_ai_gate,
    is_application_human,
    load_ai_gate_config,
    store_ai_gate_config,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12


def _saved_project(
    tmp_path: Path,
    *,
    name: str = "Project",
    settings: dict | None = None,
) -> Project:
    p = Project.new(
        name=name,
        project_id=_HEX_PROJECT,
        settings=settings or {},
    )
    save_project(tmp_path, p)
    return p


def _make_application(
    *,
    source_id: str,
    code_id: str = _HEX_CODE,
    project_id: str = _HEX_PROJECT,
    application_id: str | None = None,
    ai_provenance: AIProvenance | None = None,
    provenance: dict | None = None,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=_HEX_CODER,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w5",
        definition_version_id_at_apply=_HEX_VERSION,
        ai_provenance=ai_provenance,
        provenance=provenance,
        application_id=application_id,
    )


# --------------------------------------------------------------------------- #
# AIGateConfig defaults + validation
# --------------------------------------------------------------------------- #


class TestAIGateConfigDefaults:
    def test_default_matches_spec(self) -> None:
        c = default_ai_gate_config()
        # PLANNING.md F8.10 spec: ≥ 8 codes AND ≥ 2 transcripts.
        assert c.min_codes == 8
        assert c.min_hand_coded_sources == 2
        assert c.override == GATE_OVERRIDE_AUTO
        assert c.enabled is True
        assert c.exempt_features == ()

    def test_default_constants_match(self) -> None:
        assert DEFAULT_MIN_CODES == 8
        assert DEFAULT_MIN_HAND_CODED_SOURCES == 2

    def test_explicit_construction(self) -> None:
        c = AIGateConfig.new(
            min_codes=4,
            min_hand_coded_sources=1,
            override=GATE_OVERRIDE_FORCE_ON,
            enabled=False,
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY],
        )
        assert c.min_codes == 4
        assert c.min_hand_coded_sources == 1
        assert c.override == GATE_OVERRIDE_FORCE_ON
        assert c.enabled is False
        assert c.exempt_features == (AI_FEATURE_QUOTE_SIMILARITY,)


class TestAIGateConfigValidation:
    def test_negative_min_codes_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(min_codes=-1)

    def test_huge_min_codes_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(min_codes=MAX_THRESHOLD + 1)

    def test_negative_min_sources_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(min_hand_coded_sources=-1)

    def test_unknown_override_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(override="maybe")

    def test_known_overrides_accepted(self) -> None:
        for o in GATE_OVERRIDES:
            AIGateConfig.new(override=o)

    def test_unknown_feature_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(exempt_features=["nope"])

    def test_duplicate_exempt_features_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.new(
                exempt_features=[
                    AI_FEATURE_QUOTE_SIMILARITY,
                    AI_FEATURE_QUOTE_SIMILARITY,
                ]
            )

    def test_min_threshold_zero_allowed(self) -> None:
        # A user who wants no gate but keeps the override "auto" can
        # set both thresholds to 0 — every count satisfies.
        c = AIGateConfig.new(min_codes=0, min_hand_coded_sources=0)
        s = evaluate_ai_gate(
            config=c, code_count=0, hand_coded_source_count=0
        )
        assert s.allowed is True
        assert s.reason == REASON_THRESHOLD_MET


# --------------------------------------------------------------------------- #
# Dict round-trip
# --------------------------------------------------------------------------- #


class TestAIGateConfigDict:
    def test_to_dict_has_scalar_fields_only(self) -> None:
        c = default_ai_gate_config()
        d = c.to_dict()
        assert d == {
            SETTING_KEY_MIN_CODES: 8,
            SETTING_KEY_MIN_HAND_CODED_SOURCES: 2,
            SETTING_KEY_OVERRIDE: GATE_OVERRIDE_AUTO,
            SETTING_KEY_ENABLED: True,
        }
        # exempt_features lives in a *sibling* settings key, so it is
        # NOT part of to_dict (which goes into ``settings["ai_gate"]``).
        assert "exempt_features" not in d

    def test_from_dict_round_trip(self) -> None:
        original = AIGateConfig.new(
            min_codes=5,
            min_hand_coded_sources=3,
            override=GATE_OVERRIDE_FORCE_OFF,
            enabled=False,
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY],
        )
        d = original.to_dict()
        restored = AIGateConfig.from_dict(
            d, exempt_features=list(original.exempt_features)
        )
        assert restored == original

    def test_from_dict_empty_uses_defaults(self) -> None:
        c = AIGateConfig.from_dict({})
        assert c == default_ai_gate_config()

    def test_from_dict_none_uses_defaults(self) -> None:
        c = AIGateConfig.from_dict(None)
        assert c == default_ai_gate_config()

    def test_int_coerced_from_string(self) -> None:
        # JSON / form posts can deliver "8" as a string. Be lenient.
        c = AIGateConfig.from_dict({SETTING_KEY_MIN_CODES: "12"})
        assert c.min_codes == 12

    def test_bool_coerced_from_string(self) -> None:
        c = AIGateConfig.from_dict({SETTING_KEY_ENABLED: "false"})
        assert c.enabled is False

    def test_int_rejects_non_integer_float(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIGateConfig.from_dict({SETTING_KEY_MIN_CODES: 1.5})

    def test_int_rejects_bool(self) -> None:
        # bool is a subclass of int but ``True`` as a threshold is a
        # type confusion — reject it explicitly.
        with pytest.raises(ProjectValidationError):
            AIGateConfig.from_dict({SETTING_KEY_MIN_CODES: True})


# --------------------------------------------------------------------------- #
# Hand-coded application detection
# --------------------------------------------------------------------------- #


class TestIsApplicationHuman:
    def test_application_with_no_provenance_is_human(self) -> None:
        a = _make_application(source_id="b" * 12)
        assert is_application_human(a) is True

    def test_application_with_ai_provenance_is_not_human(self) -> None:
        a = _make_application(
            source_id="b" * 12,
            ai_provenance=AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION
            ),
        )
        assert is_application_human(a) is False

    def test_application_with_ai_accepted_marker_is_not_human(self) -> None:
        # The closed ``APPLICATION_PROVENANCE_SOURCES`` vocabulary uses
        # ``ai_accepted`` for "AI suggested, human pressed accept".
        a = _make_application(
            source_id="b" * 12,
            provenance={"source": "ai_accepted"},
        )
        assert is_application_human(a) is False

    def test_application_with_ai_modified_marker_is_not_human(self) -> None:
        a = _make_application(
            source_id="b" * 12,
            provenance={"source": "ai_modified"},
        )
        assert is_application_human(a) is False

    def test_provenance_human_marker_is_human(self) -> None:
        a = _make_application(
            source_id="b" * 12,
            provenance={"source": "human"},
        )
        assert is_application_human(a) is True

    def test_imported_provenance_counts_as_human(self) -> None:
        # An imported QDPX project carries human coding from another
        # tool; the gate treats it as hand-coded.
        a = _make_application(
            source_id="b" * 12,
            provenance={"source": "imported"},
        )
        assert is_application_human(a) is True


class TestCountHandCodedSources:
    def test_empty_list(self) -> None:
        assert count_hand_coded_sources([]) == 0

    def test_distinct_sources(self) -> None:
        apps = [
            _make_application(source_id="b" * 12),
            _make_application(source_id="b" * 11 + "c"),
            _make_application(source_id="b" * 11 + "d"),
        ]
        assert count_hand_coded_sources(apps) == 3

    def test_dedupes_same_source(self) -> None:
        apps = [
            _make_application(source_id="b" * 12),
            _make_application(source_id="b" * 12),
            _make_application(source_id="b" * 12),
        ]
        assert count_hand_coded_sources(apps) == 1

    def test_ignores_ai_applications(self) -> None:
        apps = [
            _make_application(source_id="b" * 12),  # human
            _make_application(
                source_id="b" * 11 + "c",
                ai_provenance=AIProvenance.new(
                    feature=AI_FEATURE_CODE_SUGGESTION
                ),
            ),  # AI
            _make_application(
                source_id="b" * 11 + "d",
                provenance={"source": "ai_modified"},
            ),  # AI (legacy provenance dict)
        ]
        assert count_hand_coded_sources(apps) == 1

    def test_source_with_both_human_and_ai_counts_once(self) -> None:
        # If a transcript has *any* human application, it counts as
        # hand-coded — even if it also has AI applications.
        sid = "b" * 12
        apps = [
            _make_application(source_id=sid),  # human
            _make_application(
                source_id=sid,
                ai_provenance=AIProvenance.new(
                    feature=AI_FEATURE_CODE_SUGGESTION
                ),
            ),
        ]
        assert count_hand_coded_sources(apps) == 1


# --------------------------------------------------------------------------- #
# evaluate_ai_gate — branches
# --------------------------------------------------------------------------- #


class TestEvaluateAIGateThresholds:
    def test_blocked_when_both_below(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=3,
            hand_coded_source_count=0,
        )
        assert s.allowed is False
        assert s.reason == REASON_INSUFFICIENT_BOTH
        assert s.code_count == 3
        assert s.min_codes == 8
        assert "8" in s.message
        assert "2" in s.message

    def test_blocked_when_only_codes_short(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=3,
            hand_coded_source_count=5,
        )
        assert s.allowed is False
        assert s.reason == REASON_INSUFFICIENT_CODES

    def test_blocked_when_only_sources_short(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=20,
            hand_coded_source_count=1,
        )
        assert s.allowed is False
        assert s.reason == REASON_INSUFFICIENT_HAND_CODED_SOURCES

    def test_allowed_when_both_meet(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=8,
            hand_coded_source_count=2,
        )
        assert s.allowed is True
        assert s.reason == REASON_THRESHOLD_MET

    def test_allowed_when_both_exceed(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=20,
            hand_coded_source_count=10,
        )
        assert s.allowed is True
        assert s.reason == REASON_THRESHOLD_MET

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            evaluate_ai_gate(
                config=default_ai_gate_config(),
                code_count=-1,
                hand_coded_source_count=0,
            )


class TestEvaluateAIGateOverride:
    def test_force_off_blocks_even_when_threshold_met(self) -> None:
        c = AIGateConfig.new(override=GATE_OVERRIDE_FORCE_OFF)
        s = evaluate_ai_gate(
            config=c, code_count=100, hand_coded_source_count=100
        )
        assert s.allowed is False
        assert s.reason == REASON_FORCE_OFF

    def test_force_on_allows_even_with_zero_codes(self) -> None:
        c = AIGateConfig.new(override=GATE_OVERRIDE_FORCE_ON)
        s = evaluate_ai_gate(
            config=c, code_count=0, hand_coded_source_count=0
        )
        assert s.allowed is True
        assert s.reason == REASON_FORCE_ON


class TestEvaluateAIGateDisabled:
    def test_disabled_gate_always_allows(self) -> None:
        c = AIGateConfig.new(enabled=False)
        s = evaluate_ai_gate(
            config=c, code_count=0, hand_coded_source_count=0
        )
        assert s.allowed is True
        assert s.reason == REASON_DISABLED


class TestEvaluateAIGateFeatureExemption:
    def test_exempt_feature_is_allowed(self) -> None:
        c = AIGateConfig.new(
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY]
        )
        s = evaluate_ai_gate(
            config=c,
            code_count=0,
            hand_coded_source_count=0,
            feature=AI_FEATURE_QUOTE_SIMILARITY,
        )
        assert s.allowed is True
        assert s.reason == REASON_FEATURE_EXEMPT
        assert s.feature_exempt is True
        assert s.feature == AI_FEATURE_QUOTE_SIMILARITY

    def test_non_exempt_feature_still_gated(self) -> None:
        c = AIGateConfig.new(
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY]
        )
        s = evaluate_ai_gate(
            config=c,
            code_count=0,
            hand_coded_source_count=0,
            feature=AI_FEATURE_CODE_SUGGESTION,
        )
        assert s.allowed is False
        assert s.feature_exempt is False
        assert s.feature == AI_FEATURE_CODE_SUGGESTION

    def test_unknown_feature_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            evaluate_ai_gate(
                config=default_ai_gate_config(),
                code_count=0,
                hand_coded_source_count=0,
                feature="bogus",
            )

    def test_force_off_beats_feature_exemption(self) -> None:
        # Override is stronger than the per-feature carve-out.
        c = AIGateConfig.new(
            override=GATE_OVERRIDE_FORCE_OFF,
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY],
        )
        s = evaluate_ai_gate(
            config=c,
            code_count=100,
            hand_coded_source_count=100,
            feature=AI_FEATURE_QUOTE_SIMILARITY,
        )
        assert s.allowed is False
        assert s.reason == REASON_FORCE_OFF


# --------------------------------------------------------------------------- #
# AIGateStatus serialisation
# --------------------------------------------------------------------------- #


class TestAIGateStatusToDict:
    def test_dict_has_all_fields(self) -> None:
        s = evaluate_ai_gate(
            config=default_ai_gate_config(),
            code_count=3,
            hand_coded_source_count=1,
        )
        d = s.to_dict()
        assert set(d.keys()) == {
            "allowed",
            "reason",
            "message",
            "code_count",
            "hand_coded_source_count",
            "min_codes",
            "min_hand_coded_sources",
            "override",
            "enabled",
            "feature",
            "feature_exempt",
        }
        assert d["allowed"] is False
        assert d["reason"] == REASON_INSUFFICIENT_BOTH


# --------------------------------------------------------------------------- #
# Project settings round-trip
# --------------------------------------------------------------------------- #


class TestSettingsIntegration:
    def test_load_returns_defaults_when_missing(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        assert load_ai_gate_config(p) == default_ai_gate_config()

    def test_store_then_load(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        c = AIGateConfig.new(
            min_codes=5,
            min_hand_coded_sources=1,
            override=GATE_OVERRIDE_FORCE_ON,
            enabled=True,
            exempt_features=[AI_FEATURE_QUOTE_SIMILARITY],
        )
        store_ai_gate_config(p, c)
        # The settings dict has the expected shape.
        assert p.settings[SETTING_AI_GATE] == c.to_dict()
        assert p.settings[SETTING_AI_GATE_EXEMPT_FEATURES] == [
            AI_FEATURE_QUOTE_SIMILARITY
        ]
        # And it loads back identically.
        assert load_ai_gate_config(p) == c

    def test_store_with_no_exempt_features_omits_sibling_key(
        self, tmp_path: Path
    ) -> None:
        p = _saved_project(
            tmp_path,
            settings={
                SETTING_AI_GATE_EXEMPT_FEATURES: [
                    AI_FEATURE_QUOTE_SIMILARITY
                ]
            },
        )
        # Now write a config with no exempt features; the sibling key
        # should be removed, not left dangling.
        c = AIGateConfig.new()
        store_ai_gate_config(p, c)
        assert SETTING_AI_GATE_EXEMPT_FEATURES not in p.settings

    def test_store_preserves_other_settings(self, tmp_path: Path) -> None:
        p = _saved_project(
            tmp_path,
            settings={"default_coder": "alice"},
        )
        store_ai_gate_config(p, AIGateConfig.new(min_codes=1))
        assert p.settings["default_coder"] == "alice"
        assert p.settings[SETTING_AI_GATE][SETTING_KEY_MIN_CODES] == 1

    def test_load_rejects_non_object_setting(self, tmp_path: Path) -> None:
        # A user who hand-edits the file to set ai_gate=42 is forced to
        # see the validation error, not silently get defaults.
        p = _saved_project(tmp_path)
        # Bypass validate() so we can write a deliberately malformed
        # value (the validation we want to test is at *load* time).
        p.settings = {SETTING_AI_GATE: 42}
        with pytest.raises(ProjectValidationError):
            load_ai_gate_config(p)

    def test_load_rejects_non_list_exempt_features(
        self, tmp_path: Path
    ) -> None:
        p = _saved_project(tmp_path)
        p.settings = {SETTING_AI_GATE_EXEMPT_FEATURES: "not-a-list"}
        with pytest.raises(ProjectValidationError):
            load_ai_gate_config(p)


# --------------------------------------------------------------------------- #
# evaluate_project_ai_gate — end-to-end through disk
# --------------------------------------------------------------------------- #


class TestEvaluateProjectAIGate:
    def test_empty_project_is_blocked(self, tmp_path: Path) -> None:
        _saved_project(tmp_path)
        s = evaluate_project_ai_gate(tmp_path, _HEX_PROJECT)
        assert s.allowed is False
        assert s.reason == REASON_INSUFFICIENT_BOTH
        assert s.code_count == 0
        assert s.hand_coded_source_count == 0

    def test_threshold_met_unblocks(self, tmp_path: Path) -> None:
        # A trivially-met threshold (1 code, 1 hand-coded source) so
        # we don't have to mint 8 codes in the test.
        p = _saved_project(tmp_path)
        store_ai_gate_config(
            p,
            AIGateConfig.new(min_codes=1, min_hand_coded_sources=1),
        )
        save_project(tmp_path, p)
        # Mint one code.
        c = Code.new(project_id=_HEX_PROJECT, name="Pacing")
        save_code(tmp_path, c)
        # Mint one human application against a source.
        a = _make_application(source_id="b" * 12, code_id=c.id)
        save_application(tmp_path, a)
        s = evaluate_project_ai_gate(tmp_path, _HEX_PROJECT)
        assert s.allowed is True
        assert s.reason == REASON_THRESHOLD_MET
        assert s.code_count == 1
        assert s.hand_coded_source_count == 1

    def test_ai_application_does_not_count(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        store_ai_gate_config(
            p,
            AIGateConfig.new(min_codes=1, min_hand_coded_sources=1),
        )
        save_project(tmp_path, p)
        c = Code.new(project_id=_HEX_PROJECT, name="Pacing")
        save_code(tmp_path, c)
        # Only AI applications — should NOT unlock the gate.
        a = _make_application(
            source_id="b" * 12,
            code_id=c.id,
            ai_provenance=AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION
            ),
        )
        save_application(tmp_path, a)
        s = evaluate_project_ai_gate(tmp_path, _HEX_PROJECT)
        assert s.allowed is False
        assert s.reason == REASON_INSUFFICIENT_HAND_CODED_SOURCES
        assert s.code_count == 1
        assert s.hand_coded_source_count == 0

    def test_feature_passes_through(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        store_ai_gate_config(
            p,
            AIGateConfig.new(
                exempt_features=[AI_FEATURE_QUOTE_SIMILARITY]
            ),
        )
        save_project(tmp_path, p)
        s = evaluate_project_ai_gate(
            tmp_path, _HEX_PROJECT, feature=AI_FEATURE_QUOTE_SIMILARITY
        )
        assert s.allowed is True
        assert s.reason == REASON_FEATURE_EXEMPT
        assert s.feature == AI_FEATURE_QUOTE_SIMILARITY

    def test_force_off_propagates(self, tmp_path: Path) -> None:
        p = _saved_project(tmp_path)
        store_ai_gate_config(
            p,
            AIGateConfig.new(override=GATE_OVERRIDE_FORCE_OFF),
        )
        save_project(tmp_path, p)
        s = evaluate_project_ai_gate(tmp_path, _HEX_PROJECT)
        assert s.allowed is False
        assert s.reason == REASON_FORCE_OFF
