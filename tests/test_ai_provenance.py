"""Tests for scribe.ai_provenance (F8.9).

Covers:
  * AIProvenance dataclass round-trip + validation
  * AIEvent dataclass round-trip + validation
  * Append-only persistence (save/load/list)
  * Per-engine extractors that lift suggestion records into AIProvenance
  * Application.ai_provenance round-trip
  * Projection helper to legacy free-form dict
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribe.applications import Application, save_application, load_application
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
    delete_project,
)
from scribe.ai_provenance import (
    AI_DECISION_ACCEPTED,
    AI_DECISION_MODIFIED,
    AI_DECISION_PENDING,
    AI_DECISION_REJECTED,
    AI_DECISIONS,
    AI_EVENT_ID_RE,
    AI_EVENT_KIND_APPLICATION,
    AI_EVENT_KIND_DECISION,
    AI_EVENT_KIND_ERROR,
    AI_EVENT_KIND_REQUEST,
    AI_EVENT_KINDS,
    AI_EVENTS_DIRNAME,
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_MEMO_DRAFT,
    AI_FEATURE_NEW_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURE_SECOND_CODER,
    AI_FEATURE_TRANSCRIPT_REVIEW,
    AI_FEATURES,
    AIEvent,
    AIProvenance,
    ai_event_state_path,
    ai_events_dir,
    hash_prompt,
    list_ai_events,
    load_ai_event,
    new_ai_event_id,
    provenance_from_code_suggestion,
    provenance_from_memo_draft,
    provenance_from_new_code_suggestion,
    provenance_from_second_coder_pass,
    provenance_from_transcript_review_pass,
    save_ai_event,
)


_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12
_HEX_SUGGESTION = "e" * 12


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _valid_application_kwargs(project_id: str, **overrides) -> dict:
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
# AIProvenance — basic shape + round-trip
# --------------------------------------------------------------------------- #


class TestAIProvenanceConstruction:
    def test_minimum_required_field_is_feature(self) -> None:
        p = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        assert p.feature == AI_FEATURE_CODE_SUGGESTION
        assert p.decision == AI_DECISION_PENDING
        assert p.confidence is None
        assert p.notes == ""

    def test_all_fields_populate(self) -> None:
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            embedding_model="bge-m3",
            suggestion_id=_HEX_SUGGESTION,
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=_HEX_CODER,
            decided_at="2026-01-01T00:00:00Z",
            confidence=0.75,
            prompt_hash="abcdef0123",
            notes="seeded by exemplar #2",
        )
        assert p.confidence == 0.75
        assert p.suggestion_id == _HEX_SUGGESTION
        assert p.decision == AI_DECISION_ACCEPTED


class TestAIProvenanceValidation:
    def test_rejects_unknown_feature(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(feature="alien")

    def test_rejects_unknown_decision(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION, decision="approved"
            )

    def test_rejects_bad_suggestion_id_shape(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION,
                suggestion_id="not-12-hex",
            )

    def test_rejects_bad_coder_id_shape(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION,
                decided_by_coder_id="not-12-hex",
            )

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION, confidence=1.5
            )
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION, confidence=-0.1
            )

    def test_rejects_confidence_bool(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION, confidence=True  # type: ignore[arg-type]
            )

    def test_rejects_long_model_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION,
                generation_model="m" * 257,
            )

    def test_rejects_control_chars_in_model_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.new(
                feature=AI_FEATURE_CODE_SUGGESTION,
                generation_model="bad\x01name",
            )

    def test_notes_allows_unicode_newlines(self) -> None:
        # Non-ASCII unicode + newlines welcome in notes.
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            notes="Codé:\nfollow-up needed — ñ",
        )
        assert "\n" in p.notes


class TestAIProvenanceRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        p = AIProvenance.new(
            feature=AI_FEATURE_MEMO_DRAFT,
            backend="ollama",
            generation_model="phi-4",
            decision=AI_DECISION_MODIFIED,
            decided_by_coder_id=_HEX_CODER,
            decided_at="2026-02-02T01:01:01Z",
            confidence=0.9,
            suggestion_id=_HEX_SUGGESTION,
        )
        round_tripped = AIProvenance.from_dict(p.to_dict())
        assert round_tripped == p

    def test_from_dict_missing_feature_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.from_dict({})

    def test_from_dict_non_object_rejected(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIProvenance.from_dict("not an object")  # type: ignore[arg-type]

    def test_from_dict_coerces_int_confidence(self) -> None:
        d = {"feature": AI_FEATURE_CODE_SUGGESTION, "confidence": 1}
        p = AIProvenance.from_dict(d)
        assert p.confidence == 1.0


class TestProjectionToApplicationDict:
    def test_accepted_emits_source(self) -> None:
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            embedding_model="bge-m3",
            suggestion_id=_HEX_SUGGESTION,
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=_HEX_CODER,
            decided_at="2026-03-03T00:00:00Z",
        )
        d = p.to_application_provenance_dict()
        assert d["source"] == "ai_accepted"
        assert d["model_id"] == "llama3.2:3b"
        assert d["embedding_model"] == "bge-m3"
        assert d["suggestion_id"] == _HEX_SUGGESTION
        assert d["accepted_at"] == "2026-03-03T00:00:00Z"
        assert d["feature"] == AI_FEATURE_CODE_SUGGESTION
        assert d["backend"] == "ollama"

    def test_modified_emits_modified_source(self) -> None:
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            decision=AI_DECISION_MODIFIED,
        )
        assert p.to_application_provenance_dict()["source"] == "ai_modified"

    def test_pending_omits_source(self) -> None:
        p = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        d = p.to_application_provenance_dict()
        assert "source" not in d
        assert d["feature"] == AI_FEATURE_CODE_SUGGESTION

    def test_rejected_omits_source(self) -> None:
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION, decision=AI_DECISION_REJECTED
        )
        d = p.to_application_provenance_dict()
        assert "source" not in d

    def test_projection_round_trips_into_application(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        p = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            decision=AI_DECISION_ACCEPTED,
            decided_at="2026-04-04T00:00:00Z",
        )
        # The legacy dict the helper produces must validate as an
        # Application.provenance value.
        a = Application.new(
            **_valid_application_kwargs(
                proj.id, provenance=p.to_application_provenance_dict()
            )
        )
        assert a.provenance["source"] == "ai_accepted"
        assert a.provenance["model_id"] == "llama3.2:3b"


# --------------------------------------------------------------------------- #
# AIEvent + persistence
# --------------------------------------------------------------------------- #


class TestNewAIEventId:
    def test_shape(self) -> None:
        for _ in range(10):
            assert AI_EVENT_ID_RE.match(new_ai_event_id())

    def test_unique(self) -> None:
        ids = {new_ai_event_id() for _ in range(50)}
        assert len(ids) == 50


class TestAIEventConstruction:
    def test_basic_request_event(self) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        ev = AIEvent.new(
            project_id=_HEX_PROJECT,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
            payload={"span_words": 5, "source_id": _HEX_SOURCE},
        )
        assert ev.feature == AI_FEATURE_CODE_SUGGESTION
        assert ev.kind == AI_EVENT_KIND_REQUEST
        assert ev.payload["span_words"] == 5

    def test_feature_must_match_provenance_feature(self) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        with pytest.raises(ProjectValidationError):
            AIEvent.new(
                project_id=_HEX_PROJECT,
                feature=AI_FEATURE_MEMO_DRAFT,
                kind=AI_EVENT_KIND_REQUEST,
                actor_coder_id=_HEX_CODER,
                provenance=prov,
            )

    def test_unknown_kind_rejected(self) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        with pytest.raises(ProjectValidationError):
            AIEvent.new(
                project_id=_HEX_PROJECT,
                feature=AI_FEATURE_CODE_SUGGESTION,
                kind="confused",
                actor_coder_id=_HEX_CODER,
                provenance=prov,
            )

    def test_bad_project_id(self) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        with pytest.raises(ProjectValidationError):
            AIEvent.new(
                project_id="bad",
                feature=AI_FEATURE_CODE_SUGGESTION,
                kind=AI_EVENT_KIND_REQUEST,
                actor_coder_id=_HEX_CODER,
                provenance=prov,
            )

    def test_actor_coder_id_optional(self) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        # Background-job invocations may not have a per-user actor —
        # empty is allowed.
        ev = AIEvent.new(
            project_id=_HEX_PROJECT,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id="",
            provenance=prov,
        )
        assert ev.actor_coder_id == ""


class TestAIEventPayloadValidation:
    def _make_event(self, payload):
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        return AIEvent.new(
            project_id=_HEX_PROJECT,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
            payload=payload,
        )

    def test_payload_accepts_scalars(self) -> None:
        ev = self._make_event(
            {"a": 1, "b": "ok", "c": 1.5, "d": True, "e": None}
        )
        assert ev.payload["d"] is True
        assert ev.payload["e"] is None

    def test_payload_accepts_one_level_lists(self) -> None:
        ev = self._make_event({"codes": ["x", "y", "z"]})
        assert ev.payload["codes"] == ["x", "y", "z"]

    def test_payload_rejects_too_many_keys(self) -> None:
        many = {f"k{i}": i for i in range(33)}
        with pytest.raises(ProjectValidationError):
            self._make_event(many)

    def test_payload_rejects_bad_keys(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make_event({"1nope": 1})

    def test_payload_rejects_two_level_nesting(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make_event({"x": {"y": {"z": 1}}})

    def test_payload_rejects_nonstring_long(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make_event({"x": "y" * 5000})

    def test_payload_rejects_unsupported_type(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._make_event({"x": object()})

    def test_payload_empty_default(self) -> None:
        ev = self._make_event(None)
        assert ev.payload == {}


class TestAIEventRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        prov = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            generation_model="llama3.2:3b",
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=_HEX_CODER,
            decided_at="2026-01-01T00:00:00Z",
            suggestion_id=_HEX_SUGGESTION,
        )
        ev = AIEvent.new(
            project_id=_HEX_PROJECT,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_DECISION,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
            payload={"accepted_code_id": _HEX_CODE},
        )
        round_tripped = AIEvent.from_dict(ev.to_dict())
        assert round_tripped == ev

    def test_from_dict_missing_required_fields(self) -> None:
        with pytest.raises(ProjectValidationError):
            AIEvent.from_dict({})


class TestAIEventPersistence:
    def test_save_creates_under_project(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        ev = AIEvent.new(
            project_id=proj.id,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
        )
        target = save_ai_event(tmp_path, ev)
        assert target.exists()
        assert target.parent.name == AI_EVENTS_DIRNAME

    def test_save_round_trips(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(
            feature=AI_FEATURE_MEMO_DRAFT,
            generation_model="phi-4",
            decision=AI_DECISION_ACCEPTED,
        )
        ev = AIEvent.new(
            project_id=proj.id,
            feature=AI_FEATURE_MEMO_DRAFT,
            kind=AI_EVENT_KIND_DECISION,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
            payload={"memo_id": "f" * 12},
        )
        save_ai_event(tmp_path, ev)
        loaded = load_ai_event(tmp_path, proj.id, ev.id)
        assert loaded == ev

    def test_save_refuses_to_overwrite_append_only(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        ev = AIEvent.new(
            project_id=proj.id,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
        )
        save_ai_event(tmp_path, ev)
        with pytest.raises(FileExistsError):
            save_ai_event(tmp_path, ev)

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        ev = AIEvent.new(
            project_id=_HEX_PROJECT,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
        )
        with pytest.raises(FileNotFoundError):
            save_ai_event(tmp_path, ev)

    def test_load_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_ai_event(tmp_path, proj.id, "0" * 12)

    def test_state_path_validates_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            ai_event_state_path(tmp_path, proj.id, "not-hex")


class TestAIEventList:
    def _emit(
        self, root: Path, proj: Project, *, feature: str, kind: str, ts: str = ""
    ) -> AIEvent:
        prov = AIProvenance.new(feature=feature)
        ev = AIEvent.new(
            project_id=proj.id,
            feature=feature,
            kind=kind,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
            now=ts or None,
        )
        save_ai_event(root, ev)
        return ev

    def test_list_empty(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_ai_events(tmp_path, proj.id) == []

    def test_list_returns_events_in_created_order(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST, ts="2026-01-01T00:00:00Z",
        )
        b = self._emit(
            tmp_path, proj, feature=AI_FEATURE_MEMO_DRAFT,
            kind=AI_EVENT_KIND_REQUEST, ts="2026-01-02T00:00:00Z",
        )
        c = self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_DECISION, ts="2026-01-03T00:00:00Z",
        )
        ids = [e.id for e in list_ai_events(tmp_path, proj.id)]
        assert ids == [a.id, b.id, c.id]

    def test_list_filter_by_feature(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST, ts="2026-01-01T00:00:00Z",
        )
        b = self._emit(
            tmp_path, proj, feature=AI_FEATURE_MEMO_DRAFT,
            kind=AI_EVENT_KIND_REQUEST, ts="2026-01-02T00:00:00Z",
        )
        out = list_ai_events(tmp_path, proj.id, feature=AI_FEATURE_MEMO_DRAFT)
        assert [e.id for e in out] == [b.id]

    def test_list_filter_by_kind(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_DECISION, ts="2026-01-01T00:00:00Z",
        )
        self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST, ts="2026-01-02T00:00:00Z",
        )
        out = list_ai_events(tmp_path, proj.id, kind=AI_EVENT_KIND_DECISION)
        assert [e.id for e in out] == [a.id]

    def test_list_filter_by_actor(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        ev_self = AIEvent.new(
            project_id=proj.id,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id="9" * 12,
            provenance=prov,
        )
        ev_other = AIEvent.new(
            project_id=proj.id,
            feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
            actor_coder_id=_HEX_CODER,
            provenance=prov,
        )
        save_ai_event(tmp_path, ev_self)
        save_ai_event(tmp_path, ev_other)
        out = list_ai_events(tmp_path, proj.id, actor_coder_id=_HEX_CODER)
        assert [e.id for e in out] == [ev_other.id]

    def test_list_rejects_bad_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_ai_events(tmp_path, proj.id, feature="alien")
        with pytest.raises(ProjectValidationError):
            list_ai_events(tmp_path, proj.id, kind="confused")
        with pytest.raises(ProjectValidationError):
            list_ai_events(tmp_path, proj.id, actor_coder_id="bad")

    def test_list_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        ev = self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
        )
        ed = ai_events_dir(tmp_path, proj.id)
        (ed / "0123456789ab.json").write_text("{not json}")
        out = list_ai_events(tmp_path, proj.id)
        assert [e.id for e in out] == [ev.id]

    def test_list_missing_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # No events directory was ever created.
        assert list_ai_events(tmp_path, proj.id) == []

    def test_delete_project_cascades(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        self._emit(
            tmp_path, proj, feature=AI_FEATURE_CODE_SUGGESTION,
            kind=AI_EVENT_KIND_REQUEST,
        )
        assert ai_events_dir(tmp_path, proj.id).exists()
        delete_project(tmp_path, proj.id)
        assert not ai_events_dir(tmp_path, proj.id).exists()


# --------------------------------------------------------------------------- #
# Application.ai_provenance integration
# --------------------------------------------------------------------------- #


class TestApplicationCarriesAIProvenance:
    def test_optional_default_is_none(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_application_kwargs(proj.id))
        assert a.ai_provenance is None
        # Round-trips on disk as null.
        save_application(tmp_path, a)
        loaded = load_application(tmp_path, proj.id, a.id)
        assert loaded.ai_provenance is None

    def test_can_attach_ai_provenance(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(
            feature=AI_FEATURE_CODE_SUGGESTION,
            backend="ollama",
            generation_model="llama3.2:3b",
            decision=AI_DECISION_ACCEPTED,
            decided_by_coder_id=_HEX_CODER,
            decided_at="2026-01-01T00:00:00Z",
            suggestion_id=_HEX_SUGGESTION,
            confidence=0.8,
        )
        a = Application.new(
            **_valid_application_kwargs(proj.id, ai_provenance=prov)
        )
        assert a.ai_provenance is prov
        save_application(tmp_path, a)
        loaded = load_application(tmp_path, proj.id, a.id)
        assert loaded.ai_provenance == prov

    def test_dict_form_is_accepted(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(
            **_valid_application_kwargs(
                proj.id,
                ai_provenance={
                    "feature": AI_FEATURE_CODE_SUGGESTION,
                    "generation_model": "llama3.2:3b",
                    "decision": AI_DECISION_ACCEPTED,
                },
            )
        )
        assert isinstance(a.ai_provenance, AIProvenance)
        assert a.ai_provenance.generation_model == "llama3.2:3b"

    def test_invalid_ai_provenance_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_application_kwargs(
                    proj.id,
                    ai_provenance="not an object",  # type: ignore[arg-type]
                )
            )

    def test_invalid_ai_provenance_dict_rejected(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            Application.new(
                **_valid_application_kwargs(
                    proj.id, ai_provenance={"feature": "alien"}
                )
            )

    def test_apply_update_replaces_ai_provenance(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_application_kwargs(proj.id))
        a.apply_update(
            {
                "ai_provenance": {
                    "feature": AI_FEATURE_CODE_SUGGESTION,
                    "decision": AI_DECISION_ACCEPTED,
                }
            }
        )
        assert a.ai_provenance is not None
        assert a.ai_provenance.decision == AI_DECISION_ACCEPTED

    def test_apply_update_clears_ai_provenance(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        prov = AIProvenance.new(feature=AI_FEATURE_CODE_SUGGESTION)
        a = Application.new(
            **_valid_application_kwargs(proj.id, ai_provenance=prov)
        )
        a.apply_update({"ai_provenance": None})
        assert a.ai_provenance is None

    def test_old_application_without_ai_provenance_loads(
        self, tmp_path: Path
    ) -> None:
        # Pre-F8.9 applications on disk have no ``ai_provenance`` field.
        # ``from_dict`` must default to ``None`` without raising.
        proj = _saved_project(tmp_path)
        a = Application.new(**_valid_application_kwargs(proj.id))
        # Persist, then strip the ai_provenance key to simulate an old file.
        save_application(tmp_path, a)
        path = (
            tmp_path / proj.id / "applications" / f"{a.id}.json"
        )
        d = json.loads(path.read_text())
        d.pop("ai_provenance", None)
        path.write_text(json.dumps(d))
        loaded = load_application(tmp_path, proj.id, a.id)
        assert loaded.ai_provenance is None


# --------------------------------------------------------------------------- #
# Per-engine extractors
# --------------------------------------------------------------------------- #


class _FakeCodeSuggestion:
    """Stand-in matching the F8.3 :class:`CodeSuggestion` field set."""

    def __init__(
        self,
        *,
        id="abcdef012345",
        generation_model="llama3.2:3b",
        embedding_model="bge-m3",
        decision=AI_DECISION_ACCEPTED,
        decided_by_coder_id=_HEX_CODER,
        decided_at="2026-04-01T00:00:00Z",
    ):
        self.id = id
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.decision = decision
        self.decided_by_coder_id = decided_by_coder_id
        self.decided_at = decided_at


class TestProvenanceExtractors:
    def test_from_code_suggestion(self) -> None:
        s = _FakeCodeSuggestion()
        p = provenance_from_code_suggestion(s, backend="ollama")
        assert p.feature == AI_FEATURE_CODE_SUGGESTION
        assert p.backend == "ollama"
        assert p.generation_model == "llama3.2:3b"
        assert p.embedding_model == "bge-m3"
        assert p.suggestion_id == "abcdef012345"
        assert p.decision == AI_DECISION_ACCEPTED
        assert p.decided_by_coder_id == _HEX_CODER

    def test_from_new_code_suggestion(self) -> None:
        s = _FakeCodeSuggestion()
        p = provenance_from_new_code_suggestion(s)
        assert p.feature == AI_FEATURE_NEW_CODE_SUGGESTION

    def test_from_memo_draft(self) -> None:
        s = _FakeCodeSuggestion(decision=AI_DECISION_MODIFIED)
        p = provenance_from_memo_draft(s)
        assert p.feature == AI_FEATURE_MEMO_DRAFT
        assert p.decision == AI_DECISION_MODIFIED

    def test_from_transcript_review_pass(self) -> None:
        class _Pass:
            id = "abcdef012345"
            generation_model = "llama3.2:3b"
            embedding_model = "bge-m3"

        p = provenance_from_transcript_review_pass(_Pass())
        assert p.feature == AI_FEATURE_TRANSCRIPT_REVIEW
        assert p.suggestion_id == "abcdef012345"
        # A pass record itself has no decision; it's the per-item suggestion
        # that gets adjudicated. We should default to pending.
        assert p.decision == AI_DECISION_PENDING

    def test_from_second_coder_pass(self) -> None:
        class _Pass:
            id = "abcdef012345"
            generation_model = "llama3.2:3b"

        p = provenance_from_second_coder_pass(_Pass())
        assert p.feature == AI_FEATURE_SECOND_CODER
        assert p.suggestion_id == "abcdef012345"

    def test_unknown_decision_falls_back_to_pending(self) -> None:
        s = _FakeCodeSuggestion(decision="weird")
        p = provenance_from_code_suggestion(s)
        assert p.decision == AI_DECISION_PENDING

    def test_missing_attrs_default_safely(self) -> None:
        class _Bare:
            pass

        p = provenance_from_code_suggestion(_Bare())
        assert p.feature == AI_FEATURE_CODE_SUGGESTION
        assert p.generation_model == ""
        assert p.suggestion_id == ""


# --------------------------------------------------------------------------- #
# hash_prompt
# --------------------------------------------------------------------------- #


class TestHashPrompt:
    def test_deterministic(self) -> None:
        assert hash_prompt("hello") == hash_prompt("hello")

    def test_distinct_inputs_distinct_outputs(self) -> None:
        assert hash_prompt("hello") != hash_prompt("hello!")

    def test_default_length(self) -> None:
        assert len(hash_prompt("x")) == 16

    def test_length_clamped_to_min(self) -> None:
        assert len(hash_prompt("x", length=2)) == 4

    def test_length_clamped_to_max(self) -> None:
        # 256 → clamped to 64 (full sha256 hex).
        assert len(hash_prompt("x", length=256)) == 64

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ProjectValidationError):
            hash_prompt(b"bytes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Vocabulary contract
# --------------------------------------------------------------------------- #


class TestVocabularyClosed:
    def test_features_set_includes_quote_similarity(self) -> None:
        assert AI_FEATURE_QUOTE_SIMILARITY in AI_FEATURES

    def test_event_kinds_match_constants(self) -> None:
        assert set(AI_EVENT_KINDS) == {
            AI_EVENT_KIND_REQUEST,
            AI_EVENT_KIND_DECISION,
            AI_EVENT_KIND_APPLICATION,
            AI_EVENT_KIND_ERROR,
        }

    def test_decisions_match_constants(self) -> None:
        assert set(AI_DECISIONS) == {
            AI_DECISION_PENDING,
            AI_DECISION_ACCEPTED,
            AI_DECISION_MODIFIED,
            AI_DECISION_REJECTED,
        }
