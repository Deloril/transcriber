"""Tests for ``scribe.anonymise`` (F6.7).

Pure-Python tests for the redaction-pass module. Cover:

  * RedactionRule shape, serialisation round-trip, compile() validation.
  * RedactionPlan apply, counts, manifest (no leaks of original
    identifiers).
  * build_redaction_plan from participants + speaker maps + custom rules,
    including ordering (custom > speaker > participant) and de-duping.
  * redact_segments — speaker rewrite via label_map, multi-word
    redaction with timestamp redistribution, robustness to malformed
    inputs.
  * redact_source / redact_code / redact_memo / redact_coder /
    redact_project — each entity preserves required-field validation
    when the redaction empties a field.
  * build_anonymised_qdpx — full end-to-end bundle with a
    Redactions/manifest.json file alongside the standard QDPX layout.
"""

from __future__ import annotations

import io
import json
import zipfile
from xml.etree import ElementTree as ET

import pytest

from scribe.anonymise import (
    AnonymisedBundle,
    MAX_RULES,
    MAX_RULE_PATTERN_LEN,
    RedactionPlan,
    RedactionRule,
    build_anonymised_qdpx,
    build_redaction_plan,
    redact_code,
    redact_coder,
    redact_memo,
    redact_project,
    redact_segments,
    redact_source,
    redact_text,
)
from scribe.applications import Application
from scribe.coders import Coder
from scribe.codes import Code
from scribe.memos import Memo, MemoLink
from scribe.participants import Participant
from scribe.projects import Project
from scribe.refi_qda_project import REFI_QDA_PROJECT_NS
from scribe.sources import Source
from scribe.speaker_map import SpeakerEntry, SpeakerMap


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_project() -> Project:
    return Project.new(
        name="Living with chronic illness",
        research_question="How do informants narrate Jane Doe's experience?",
        sensitising_concepts=["pacing", "Jane Doe disclosure"],
        description="Field site: Mercy General Hospital with Jane Doe.",
        now="2024-03-04T05:06:07.000000Z",
    )


@pytest.fixture
def sample_participants(sample_project: Project) -> list[Participant]:
    return [
        Participant.new(
            project_id=sample_project.id,
            name="Jane Doe",
            pseudonym="P01",
            now="2024-03-04T05:06:07.000000Z",
        ),
        Participant.new(
            project_id=sample_project.id,
            name="Pat",
            pseudonym="P02",
            now="2024-03-04T05:06:07.000000Z",
        ),
        Participant.new(
            project_id=sample_project.id,
            # Participant with no pseudonym → contributes no rule.
            name="Sam Anonymous",
            pseudonym="",
            now="2024-03-04T05:06:07.000000Z",
        ),
    ]


# --------------------------------------------------------------------------- #
# RedactionRule
# --------------------------------------------------------------------------- #


class TestRedactionRule:
    def test_defaults(self) -> None:
        r = RedactionRule(pattern="Pat", replacement="P02")
        assert r.case_insensitive is True
        assert r.whole_word is True
        assert r.regex is False

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        r = RedactionRule(
            pattern="\\d{3}-\\d{4}",
            replacement="<phone>",
            case_insensitive=False,
            whole_word=False,
            regex=True,
        )
        d = r.to_dict()
        r2 = RedactionRule.from_dict(d)
        assert r2 == r

    def test_from_dict_requires_pattern_and_replacement(self) -> None:
        with pytest.raises(ValueError):
            RedactionRule.from_dict({"pattern": "x"})
        with pytest.raises(ValueError):
            RedactionRule.from_dict({"replacement": "x"})

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError):
            RedactionRule.from_dict("not a dict")  # type: ignore[arg-type]

    def test_compile_rejects_empty_pattern(self) -> None:
        with pytest.raises(ValueError):
            RedactionRule(pattern="", replacement="X").compile()

    def test_compile_rejects_oversized_pattern(self) -> None:
        big = "a" * (MAX_RULE_PATTERN_LEN + 1)
        with pytest.raises(ValueError):
            RedactionRule(pattern=big, replacement="X").compile()

    def test_compile_rejects_invalid_regex(self) -> None:
        with pytest.raises(ValueError):
            RedactionRule(pattern="(unclosed", replacement="X", regex=True).compile()

    def test_compile_literal_match(self) -> None:
        p = RedactionRule(pattern="Pat", replacement="P02").compile()
        assert p.search("Hello Pat!") is not None

    def test_whole_word_does_not_match_substring(self) -> None:
        p = RedactionRule(pattern="Pat", replacement="P02").compile()
        # "patio" must not match — "Pat" should be a whole word.
        assert p.search("the patio") is None

    def test_whole_word_false_matches_substring(self) -> None:
        p = RedactionRule(
            pattern="Pat", replacement="P02", whole_word=False
        ).compile()
        assert p.search("the patio") is not None

    def test_case_insensitive_default(self) -> None:
        p = RedactionRule(pattern="Pat", replacement="P02").compile()
        assert p.search("PAT said") is not None

    def test_case_sensitive(self) -> None:
        p = RedactionRule(
            pattern="Pat", replacement="P02", case_insensitive=False
        ).compile()
        assert p.search("PAT") is None
        assert p.search("Pat") is not None

    def test_regex_mode(self) -> None:
        p = RedactionRule(
            pattern=r"\d{3}-\d{4}",
            replacement="<phone>",
            regex=True,
        ).compile()
        assert p.search("Call 555-1234.") is not None


# --------------------------------------------------------------------------- #
# RedactionPlan
# --------------------------------------------------------------------------- #


class TestRedactionPlan:
    def test_empty_plan_returns_text_unchanged(self) -> None:
        plan = RedactionPlan(rules=[])
        assert plan.apply("hello") == "hello"
        assert plan.counts() == []

    def test_apply_runs_rules_in_order(self) -> None:
        # Two rules where order matters: redact "Jane Doe" → P01 first,
        # then "Doe" → X. After the first match, "Doe" is gone.
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
            RedactionRule(pattern="Doe", replacement="X"),
        ])
        assert plan.apply("Jane Doe was here") == "P01 was here"

    def test_counts_track_substitutions(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        plan.apply("Pat met Pat at noon.")
        assert plan.counts() == [2]

    def test_counts_accumulate_across_calls(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        plan.apply("Pat A")
        plan.apply("Pat B Pat C")
        assert plan.counts() == [3]

    def test_reset_counts(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        plan.apply("Pat said")
        plan.reset_counts()
        assert plan.counts() == [0]

    def test_apply_handles_none(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="x", replacement="y"),
        ])
        assert plan.apply(None) == ""  # type: ignore[arg-type]

    def test_apply_handles_empty(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="x", replacement="y"),
        ])
        assert plan.apply("") == ""

    def test_apply_coerces_non_string(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="42", replacement="REDACTED"),
        ])
        assert plan.apply(42) == "REDACTED"  # type: ignore[arg-type]

    def test_too_many_rules_rejected(self) -> None:
        rules = [
            RedactionRule(pattern=f"x{i}", replacement="y")
            for i in range(MAX_RULES + 1)
        ]
        plan = RedactionPlan(rules=rules)
        with pytest.raises(ValueError):
            plan.apply("hello")

    def test_manifest_does_not_leak_patterns(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        plan.apply("Jane Doe and Pat met")
        m = plan.manifest()
        # Only replacements + counts must appear; no patterns.
        as_json = json.dumps(m)
        assert "Jane Doe" not in as_json
        assert "Pat" not in as_json
        assert "P01" in as_json
        assert "P02" in as_json
        assert m["rule_count"] == 2
        assert m["total_substitutions"] == 2
        assert m["rules"][0]["replacement"] == "P01"
        assert m["rules"][0]["match_count"] == 1
        assert m["rules"][1]["match_count"] == 1


# --------------------------------------------------------------------------- #
# build_redaction_plan
# --------------------------------------------------------------------------- #


class TestBuildRedactionPlan:
    def test_participants_with_pseudonyms_become_rules(
        self, sample_participants: list[Participant]
    ) -> None:
        plan, _ = build_redaction_plan(participants=sample_participants)
        rules = plan.rules
        # Two rules for the two participants with non-empty pseudonyms;
        # Sam Anonymous (no pseudonym) contributes nothing.
        replacements = {r.replacement for r in rules}
        assert "P01" in replacements
        assert "P02" in replacements

    def test_participant_without_pseudonym_skipped(
        self, sample_participants: list[Participant]
    ) -> None:
        plan, _ = build_redaction_plan(participants=sample_participants)
        # No rule for "Sam Anonymous" since its pseudonym is empty.
        for r in plan.rules:
            assert r.pattern != "Sam Anonymous"

    def test_participant_rules_sorted_longest_first(self) -> None:
        # Multi-word pattern should win over surname pattern.
        ps = [
            Participant.new(project_id="aaaaaaaaaaaa", name="Pat Smith", pseudonym="P03"),
            Participant.new(project_id="aaaaaaaaaaaa", name="Smith", pseudonym="X"),
        ]
        plan, _ = build_redaction_plan(participants=ps)
        # The multi-word pattern is first.
        patterns = [r.pattern for r in plan.rules]
        assert patterns.index("Pat Smith") < patterns.index("Smith")

    def test_custom_rules_come_first(self) -> None:
        ps = [
            Participant.new(project_id="aaaaaaaaaaaa", name="Pat", pseudonym="P02"),
        ]
        plan, _ = build_redaction_plan(
            participants=ps,
            custom_rules=[RedactionRule(pattern="MercyHospital", replacement="HOSP")],
        )
        assert plan.rules[0].pattern == "MercyHospital"

    def test_speaker_map_yields_label_map_and_rules(self) -> None:
        p = Participant.new(project_id="aaaaaaaaaaaa", name="Jane", pseudonym="P01")
        sm = SpeakerMap.new(
            project_id="aaaaaaaaaaaa",
            source_id="bbbbbbbbbbbb",
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewee",
                    participant_id=p.id,
                    display_name="Dr Jane Smith",
                ),
            ],
        )
        plan, label_map = build_redaction_plan(
            participants=[p], speaker_maps=[sm]
        )
        # Speaker label maps directly to pseudonym.
        assert label_map["SPEAKER_00"] == "P01"
        # Display name + label both contribute rules.
        replacements = {r.replacement for r in plan.rules}
        assert replacements == {"P01"}
        patterns = {r.pattern for r in plan.rules}
        assert "Dr Jane Smith" in patterns
        assert "SPEAKER_00" in patterns
        assert "Jane" in patterns

    def test_speaker_entry_without_pseudonym_emits_no_rule(self) -> None:
        # Participant has no pseudonym; speaker map references it.
        p = Participant.new(project_id="aaaaaaaaaaaa", name="Jane", pseudonym="")
        sm = SpeakerMap.new(
            project_id="aaaaaaaaaaaa",
            source_id="bbbbbbbbbbbb",
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewee",
                    participant_id=p.id,
                ),
            ],
        )
        plan, label_map = build_redaction_plan(
            participants=[p], speaker_maps=[sm]
        )
        assert label_map == {}
        assert plan.rules == []

    def test_dedupes_identical_rules(self) -> None:
        # Custom rule duplicates the participant rule; dedupe keeps one.
        ps = [
            Participant.new(project_id="aaaaaaaaaaaa", name="Pat", pseudonym="P02"),
        ]
        plan, _ = build_redaction_plan(
            participants=ps,
            custom_rules=[RedactionRule(pattern="Pat", replacement="P02")],
        )
        seen = sum(1 for r in plan.rules if r.pattern == "Pat")
        assert seen == 1

    def test_custom_rules_may_be_dicts(self) -> None:
        plan, _ = build_redaction_plan(
            custom_rules=[
                {"pattern": "MercyHospital", "replacement": "HOSP"},
            ],
        )
        assert plan.rules[0].pattern == "MercyHospital"
        assert plan.rules[0].replacement == "HOSP"

    def test_custom_rules_reject_garbage(self) -> None:
        with pytest.raises(ValueError):
            build_redaction_plan(custom_rules=["not a rule"])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# redact_text
# --------------------------------------------------------------------------- #


class TestRedactText:
    def test_basic(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        assert redact_text("Pat said hi.", plan) == "P02 said hi."


# --------------------------------------------------------------------------- #
# redact_segments
# --------------------------------------------------------------------------- #


class TestRedactSegments:
    def test_speaker_label_map_rewrites_speaker_field(self) -> None:
        plan = RedactionPlan(rules=[])
        segs = [{"speaker": "SPEAKER_00", "words": []}]
        out = redact_segments(segs, plan, speaker_label_map={"SPEAKER_00": "P01"})
        assert out[0]["speaker"] == "P01"

    def test_speaker_runs_through_plan_when_no_label_map(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="LUKE", replacement="Interviewer"),
        ])
        segs = [{"speaker": "LUKE", "words": []}]
        out = redact_segments(segs, plan)
        assert out[0]["speaker"] == "Interviewer"

    def test_per_word_single_token_redaction(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane", replacement="P01"),
        ])
        segs = [
            {"speaker": "", "words": [
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "Jane", "start": 1.0, "end": 2.0},
            ]}
        ]
        out = redact_segments(segs, plan)
        words = out[0]["words"]
        assert [w["text"] for w in words] == ["Hello", "P01"]
        # Timestamps preserved when count is unchanged.
        assert words[1]["start"] == 1.0 and words[1]["end"] == 2.0

    def test_multiword_redaction_collapses_words(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
        ])
        segs = [
            {"speaker": "", "words": [
                {"text": "Hello", "start": 0.0, "end": 1.0},
                {"text": "Jane", "start": 1.0, "end": 2.0},
                {"text": "Doe", "start": 2.0, "end": 3.0},
            ]}
        ]
        out = redact_segments(segs, plan)
        texts = [w["text"] for w in out[0]["words"]]
        assert texts == ["Hello", "P01"]
        # Timestamps redistributed proportionally across the segment span.
        assert out[0]["words"][0]["start"] == 0.0
        assert out[0]["words"][-1]["end"] == 3.0

    def test_empty_segment_words_remains_empty(self) -> None:
        plan = RedactionPlan(rules=[])
        out = redact_segments([{"speaker": "X", "words": []}], plan)
        assert out[0]["words"] == []

    def test_skips_non_mapping_segments(self) -> None:
        plan = RedactionPlan(rules=[])
        out = redact_segments([None, {"speaker": "X", "words": []}], plan)  # type: ignore[list-item]
        assert len(out) == 1

    def test_skips_non_mapping_words(self) -> None:
        plan = RedactionPlan(rules=[])
        segs = [{"speaker": "", "words": [None, {"text": "ok"}]}]
        out = redact_segments(segs, plan)  # type: ignore[list-item]
        assert [w["text"] for w in out[0]["words"]] == ["ok"]

    def test_segment_text_field_redacted_when_present(self) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane", replacement="P01"),
        ])
        segs = [{"speaker": "", "text": "Jane spoke", "words": []}]
        out = redact_segments(segs, plan)
        assert out[0]["text"] == "P01 spoke"

    def test_redaction_without_timestamps_falls_back_to_text_only(self) -> None:
        # Multi-word redaction on words missing timestamps → text-only out.
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
        ])
        segs = [{"speaker": "", "words": [
            {"text": "Hi"}, {"text": "Jane"}, {"text": "Doe"},
        ]}]
        out = redact_segments(segs, plan)
        assert [w["text"] for w in out[0]["words"]] == ["Hi", "P01"]
        assert "start" not in out[0]["words"][0]


# --------------------------------------------------------------------------- #
# Per-entity redaction
# --------------------------------------------------------------------------- #


class TestRedactSource:
    def test_redacts_name_notes_attributes(self) -> None:
        s = Source.new(
            project_id="aaaaaaaaaaaa",
            name="Interview with Jane Doe",
            notes="Jane Doe was tired.",
            custom_attributes={"site": "Mercy General"},
        )
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
            RedactionRule(pattern="Mercy General", replacement="HOSP"),
        ])
        s2 = redact_source(s, plan)
        assert s2.name == "Interview with P01"
        assert s2.notes == "P01 was tired."
        assert s2.custom_attributes == {"site": "HOSP"}
        # Validation invariants survive (not re-validated, but shape OK).
        assert s2.id == s.id
        assert s2.project_id == s.project_id

    def test_fallback_when_redaction_empties_name(self) -> None:
        s = Source.new(project_id="aaaaaaaaaaaa", name="Jane Doe")
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement=""),
        ])
        s2 = redact_source(s, plan)
        assert s2.name == "<source-redacted>"


class TestRedactCode:
    def test_redacts_definition_and_exemplars(self) -> None:
        c = Code.new(
            project_id="aaaaaaaaaaaa",
            name="Pacing",
            definition="Activity rationing as Pat described it.",
            inclusion_criteria="Pat-style examples",
            exclusion_criteria="not Pat",
            exemplars=["Pat does ten minutes a day", "Pat rests"],
            theoretical_memo="Pat's pacing is reactive.",
        )
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        c2 = redact_code(c, plan)
        assert "Pat" not in c2.definition
        assert "P02" in c2.definition
        assert all("Pat" not in e for e in c2.exemplars)
        assert "P02" in c2.theoretical_memo

    def test_fallback_when_redaction_empties_name(self) -> None:
        c = Code.new(project_id="aaaaaaaaaaaa", name="Pat")
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement=""),
        ])
        c2 = redact_code(c, plan)
        assert c2.name == "<code-redacted>"


class TestRedactMemo:
    def test_redacts_title_body_tags_preserves_links(self) -> None:
        link = MemoLink(target_type="code", target_id="aaaaaaaaaaa1")
        m = Memo.new(
            project_id="aaaaaaaaaaaa",
            type="theoretical",
            title="On Pat",
            body="Pat's strategy is interesting.",
            tags=["Pat", "strategy"],
            links=[link],
        )
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement="P02"),
        ])
        m2 = redact_memo(m, plan)
        assert "Pat" not in m2.title
        assert "Pat" not in m2.body
        assert "P02" in m2.tags
        # Links preserved structurally.
        assert len(m2.links) == 1
        assert m2.links[0].target_id == "aaaaaaaaaaa1"


class TestRedactCoder:
    def test_redacts_name(self) -> None:
        coder = Coder.new(project_id="aaaaaaaaaaaa", name="Pat Smith")
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat Smith", replacement="Coder A"),
        ])
        c2 = redact_coder(coder, plan)
        assert c2.name == "Coder A"

    def test_fallback_when_redaction_empties_name(self) -> None:
        coder = Coder.new(project_id="aaaaaaaaaaaa", name="Pat")
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Pat", replacement=""),
        ])
        c2 = redact_coder(coder, plan)
        assert c2.name == "<coder-redacted>"


class TestRedactProject:
    def test_redacts_text_fields(self, sample_project: Project) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="Jane Doe", replacement="P01"),
            RedactionRule(pattern="Mercy General Hospital", replacement="HOSP"),
        ])
        p2 = redact_project(sample_project, plan)
        assert "Jane Doe" not in p2.name
        assert "Jane Doe" not in p2.research_question
        assert "Jane Doe" not in p2.description
        assert "Mercy General Hospital" not in p2.description
        assert all("Jane Doe" not in c for c in p2.sensitising_concepts)


# --------------------------------------------------------------------------- #
# build_anonymised_qdpx
# --------------------------------------------------------------------------- #


class TestBuildAnonymisedQdpx:
    def test_emits_qdpx_zip_with_manifest(
        self,
        sample_project: Project,
        sample_participants: list[Participant],
    ) -> None:
        s = Source.new(
            project_id=sample_project.id,
            name="Interview with Jane Doe",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        c = Code.new(
            project_id=sample_project.id,
            name="Pacing",
            definition="Pat-style activity rationing.",
        )
        coder = Coder.new(project_id=sample_project.id, name="RA Pat")
        memo = Memo.new(
            project_id=sample_project.id,
            type="theoretical",
            title="Why pacing?",
            body="Jane Doe described pacing as reactive.",
        )
        app = Application.new(
            project_id=sample_project.id,
            code_id=c.id,
            source_id=s.id,
            coder_id=coder.id,
            anchor_start_word_id="s0w1",
            anchor_end_word_id="s0w2",
            definition_version_id_at_apply="aaaabbbbcccc",
        )
        sm = SpeakerMap.new(
            project_id=sample_project.id,
            source_id=s.id,
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewee",
                    participant_id=sample_participants[0].id,
                ),
            ],
        )
        segments = [
            {"speaker": "SPEAKER_00", "words": [
                {"text": "Hello"}, {"text": "Jane"}, {"text": "Doe"},
            ]},
        ]

        bundle = build_anonymised_qdpx(
            project=sample_project,
            sources=[s],
            codes=[c],
            applications=[app],
            memos=[memo],
            coders=[coder],
            participants=sample_participants,
            speaker_maps=[sm],
            segments_by_source_id={s.id: segments},
        )

        assert isinstance(bundle, AnonymisedBundle)
        assert bundle.archive  # non-empty bytes
        # Open and inspect the zip layout.
        with zipfile.ZipFile(io.BytesIO(bundle.archive), mode="r") as zf:
            names = sorted(zf.namelist())
            assert "project.qde" in names
            assert "Redactions/manifest.json" in names
            assert any(n.startswith("Sources/") for n in names)

            # The plain-text source must contain the pseudonym, not the
            # original participant name.
            txt = zf.read(f"Sources/{s.id}.txt").decode("utf-8")
            assert "Jane Doe" not in txt
            assert "P01" in txt

            # The manifest must contain pseudonyms but NOT real names.
            mani = json.loads(zf.read("Redactions/manifest.json"))
            mani_json = json.dumps(mani)
            assert "Jane Doe" not in mani_json
            assert "Pat" not in mani_json
            assert mani["total_substitutions"] >= 1

    def test_qde_xml_redacts_project_name(
        self,
        sample_project: Project,
        sample_participants: list[Participant],
    ) -> None:
        bundle = build_anonymised_qdpx(
            project=sample_project,
            participants=sample_participants,
            custom_rules=[RedactionRule(pattern="chronic illness", replacement="REDACT")],
        )
        with zipfile.ZipFile(io.BytesIO(bundle.archive), mode="r") as zf:
            qde = zf.read("project.qde").decode("utf-8")
            root = ET.fromstring(qde)
            # Root project name reflects redaction.
            name = root.get("name") or ""
            assert "chronic illness" not in name
            assert "REDACT" in name
            # Namespace-correct.
            assert root.tag == f"{{{REFI_QDA_PROJECT_NS}}}Project"

    def test_segments_without_source_omit_selections(
        self,
        sample_project: Project,
        sample_participants: list[Participant],
    ) -> None:
        s = Source.new(
            project_id=sample_project.id,
            name="Interview",
            source_type="transcript",
            transcript_job_id="abcdef012345",
        )
        # No segments_by_source_id entry → source bundles without selections.
        bundle = build_anonymised_qdpx(
            project=sample_project,
            sources=[s],
            participants=sample_participants,
        )
        with zipfile.ZipFile(io.BytesIO(bundle.archive), mode="r") as zf:
            qde = zf.read("project.qde").decode("utf-8")
            # No Sources/<id>.txt was written.
            assert f"Sources/{s.id}.txt" not in zf.namelist()
            # But the Source element is still in the XML.
            assert s.id in qde

    def test_caller_can_pass_explicit_plan(
        self, sample_project: Project
    ) -> None:
        plan = RedactionPlan(rules=[
            RedactionRule(pattern="chronic", replacement="REDACT"),
        ])
        bundle = build_anonymised_qdpx(
            project=sample_project,
            plan=plan,
        )
        # Apply was invoked at least once via the project redaction.
        assert plan.counts()[0] >= 1
        # Bundle reflects the plan we passed in.
        with zipfile.ZipFile(io.BytesIO(bundle.archive), mode="r") as zf:
            mani = json.loads(zf.read("Redactions/manifest.json"))
            assert mani["rule_count"] == 1
            assert mani["rules"][0]["replacement"] == "REDACT"

    def test_manifest_returned_separately_matches_archive(
        self,
        sample_project: Project,
        sample_participants: list[Participant],
    ) -> None:
        bundle = build_anonymised_qdpx(
            project=sample_project,
            participants=sample_participants,
        )
        with zipfile.ZipFile(io.BytesIO(bundle.archive), mode="r") as zf:
            from_archive = json.loads(zf.read("Redactions/manifest.json"))
        assert from_archive == bundle.manifest
