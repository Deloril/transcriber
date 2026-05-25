"""AI provenance + AI event log for the academic-coding workflow (F8.9).

Per PLANNING.md F8.9:

  > AI provenance fields on every application + every event.

The F8.x AI engines (code suggestions F8.3, new-code suggestions F8.4,
quote similarity F8.5, transcript review F8.6, second-coder F8.7, memo
drafts F8.8) all record their own per-feature suggestion records with
``generation_model`` / ``embedding_model`` / decision lifecycle. F8.9
adds two missing pieces:

1. **A canonical** :class:`AIProvenance` **schema** that stamps "an AI
   touched this" onto a downstream artefact (an Application, a Memo, an
   audit-log entry). Today every engine stuffs ``model_id``,
   ``suggestion_id``, etc. into the free-form ``provenance: dict[str, str]``
   on Application; F8.9 formalises the field set so callers can stop
   reinventing keys.

2. **An append-only AI event log** at
   ``projects/<pid>/ai_events/<eid>.json``. Every AI invocation
   (request, decision, application-from-suggestion) gets one
   :class:`AIEvent`. Rejections, errors, and accept-with-modification
   all land here so PLANNING §"AI invocation log including *rejected*
   suggestions" (F9.6) has a real backing store. F9.1's generic event
   log can later wrap or subsume this; until then, AI events live
   on their own so the AI features can already write into the log.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.9 is the data model + writer.
  The ``/api/projects/<id>/ai-events`` routes can be added later; for
  now F9.6 / F9.7 (export) and the AI engines (writers) are the
  consumers.
* **No automatic events from existing engines.** F8.9 ships the
  schema and the persistence helpers; wiring each engine's ``record_decision``
  to also emit an ``AIEvent`` is a follow-on. The structured
  ``AIProvenance`` field on Application *is* wired up here, since
  Application is the single place every AI feature funnels into.
* **Stand-alone, pure Python.** Mirrors :mod:`scribe.applications`,
  :mod:`scribe.coders`, :mod:`scribe.code_suggestions`. Tests stay
  pure-Python.

Integrating with existing entities
----------------------------------

* :class:`scribe.applications.Application` gains an optional
  ``ai_provenance: AIProvenance | None`` field. This is the structured
  twin of the free-form ``provenance: dict[str, str]`` it has carried
  since F4.1; both round-trip on disk for backwards compatibility.
* Helpers convert from the existing per-feature suggestion records to
  an :class:`AIProvenance` instance so callers don't reinvent the
  field set. See :func:`provenance_from_code_suggestion`,
  :func:`provenance_from_new_code_suggestion`,
  :func:`provenance_from_memo_draft`,
  :func:`provenance_from_transcript_review`,
  :func:`provenance_from_second_coder`.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .coders import CODER_ID_RE
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# AI event ids share the same 12-char hex shape as every other id in
# Scribe; keeps URL routing and traversal-guards uniform.
AI_EVENT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory for AI events. F9.1's generic event log may
# later supersede this — for now AI events live on their own so F8.9
# can ship without F9.1.
AI_EVENTS_DIRNAME = "ai_events"

# Closed feature vocabulary. Mirrors the F8.x roster. ``other`` covers
# future engines so we can ship a feature without a schema migration.
AI_FEATURE_CODE_SUGGESTION = "code_suggestion"
AI_FEATURE_NEW_CODE_SUGGESTION = "new_code_suggestion"
AI_FEATURE_QUOTE_SIMILARITY = "quote_similarity"
AI_FEATURE_TRANSCRIPT_REVIEW = "transcript_review"
AI_FEATURE_SECOND_CODER = "second_coder"
AI_FEATURE_MEMO_DRAFT = "memo_draft"
AI_FEATURE_OTHER = "other"
AI_FEATURES: tuple[str, ...] = (
    AI_FEATURE_CODE_SUGGESTION,
    AI_FEATURE_NEW_CODE_SUGGESTION,
    AI_FEATURE_QUOTE_SIMILARITY,
    AI_FEATURE_TRANSCRIPT_REVIEW,
    AI_FEATURE_SECOND_CODER,
    AI_FEATURE_MEMO_DRAFT,
    AI_FEATURE_OTHER,
)

# AI event "kind" — what stage of the AI lifecycle this entry records.
#   - request:   the engine invoked a backend (with prompt + models).
#   - decision:  a human resolved the suggestion (accepted / modified /
#                rejected). Carries the suggestion id and decision.
#   - application: an Application was minted from an accepted suggestion.
#                Carries application_id; useful when the F4.1 layer
#                builds the Application later than the decision was
#                recorded.
#   - error:     the backend failed (timeout, parse error, refusal).
#                Per-item errors are preserved so F9.6 / F9.7 can
#                surface them.
AI_EVENT_KIND_REQUEST = "request"
AI_EVENT_KIND_DECISION = "decision"
AI_EVENT_KIND_APPLICATION = "application"
AI_EVENT_KIND_ERROR = "error"
AI_EVENT_KINDS: tuple[str, ...] = (
    AI_EVENT_KIND_REQUEST,
    AI_EVENT_KIND_DECISION,
    AI_EVENT_KIND_APPLICATION,
    AI_EVENT_KIND_ERROR,
)

# Decision vocabulary mirrors the per-engine SUGGESTION_DECISIONS so
# the AI event log doesn't drift from the suggestion stores.
AI_DECISION_PENDING = "pending"
AI_DECISION_ACCEPTED = "accepted"
AI_DECISION_MODIFIED = "modified"
AI_DECISION_REJECTED = "rejected"
AI_DECISIONS: tuple[str, ...] = (
    AI_DECISION_PENDING,
    AI_DECISION_ACCEPTED,
    AI_DECISION_MODIFIED,
    AI_DECISION_REJECTED,
)

# Field-length and cardinality caps. Generous, but bounded so a stray
# upstream bug can't write a 50 MB event record.
MAX_MODEL_NAME_LEN = 256
MAX_BACKEND_NAME_LEN = 64
MAX_NOTES_LEN = 4000
MAX_PROMPT_HASH_LEN = 64       # sha256 hex is 64 chars; we accept short forms too
MAX_PAYLOAD_KEYS = 32
MAX_PAYLOAD_KEY_LEN = 64
MAX_PAYLOAD_STRING_LEN = 4000
MAX_PAYLOAD_LIST_LEN = 64
MAX_PAYLOAD_BYTES = 16 * 1024  # 16 KiB serialised — bigger than per-key cap suggests

# Backend / model name shape: free-form printable, no control chars.
# Allows ``ollama``, ``llama.cpp``, ``llama3.2:3b``, ``bge-m3``, etc.
PRINTABLE_RE = re.compile(r"^[\x20-\x7e]*$")

# Suggestion ids referenced by an AIProvenance must match the same
# 12-char hex shape used by every per-engine suggestion record
# (CodeSuggestion, NewCodeSuggestion, MemoDraft, ReviewPass /
# SecondCoderPass, etc.).
SUGGESTION_ID_RE = re.compile(r"^[a-f0-9]{12}$")


# --------------------------------------------------------------------------- #
# AIProvenance — the structured "an AI touched this" stamp
# --------------------------------------------------------------------------- #


@dataclass
class AIProvenance:
    """Structured AI provenance attached to an artefact (F8.9).

    The fields are deliberately a *superset* of what any one AI engine
    needs, so the same dataclass can stamp an Application coming from
    F8.3, a Memo coming from F8.8, or an audit-log entry coming from
    F9.6. Each field is optional — the only required entry is
    ``feature``, which says *which* AI engine produced this.

    Fields
    ------
    feature
        One of :data:`AI_FEATURES`. Required.
    backend
        The :mod:`scribe.ai_backend` adapter id, e.g. ``"ollama"``.
        Optional but strongly recommended.
    generation_model
        The chat / completion model id, e.g. ``"llama3.2:3b"``.
    embedding_model
        The embedding model id, e.g. ``"bge-m3"``.
    suggestion_id
        12-char hex id of the originating per-engine suggestion record
        (CodeSuggestion / NewCodeSuggestion / MemoDraft / etc.) so the
        full audit chain reconstructs.
    decision
        One of :data:`AI_DECISIONS`. ``pending`` for "not yet
        adjudicated"; ``accepted`` / ``modified`` / ``rejected``
        otherwise. Mirrors the per-engine vocabulary.
    decided_by_coder_id
        12-char hex coder id; the human who pressed accept / modify /
        reject.
    decided_at
        ISO-8601 UTC timestamp; when the decision was recorded.
    confidence
        Optional float in [0, 1]; the suggestion's confidence at apply
        time. Maps to ``Application.confidence`` for downstream codes.
    prompt_hash
        Optional short sha256 (or other digest) of the prompt body, so
        F9.6 / F9.7 reports can dedupe identical invocations without
        keeping the full prompt text on every record.
    notes
        Free-form short text. Bounded by :data:`MAX_NOTES_LEN`.
    """

    feature: str
    backend: str = ""
    generation_model: str = ""
    embedding_model: str = ""
    suggestion_id: str = ""
    decision: str = AI_DECISION_PENDING
    decided_by_coder_id: str = ""
    decided_at: str = ""
    confidence: float | None = None
    prompt_hash: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        feature: str,
        backend: str = "",
        generation_model: str = "",
        embedding_model: str = "",
        suggestion_id: str = "",
        decision: str = AI_DECISION_PENDING,
        decided_by_coder_id: str = "",
        decided_at: str = "",
        confidence: float | None = None,
        prompt_hash: str = "",
        notes: str = "",
    ) -> "AIProvenance":
        """Build a validated :class:`AIProvenance`."""
        p = cls(
            feature=feature,
            backend=backend,
            generation_model=generation_model,
            embedding_model=embedding_model,
            suggestion_id=suggestion_id,
            decision=decision,
            decided_by_coder_id=decided_by_coder_id,
            decided_at=decided_at,
            confidence=confidence,
            prompt_hash=prompt_hash,
            notes=notes,
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "backend": self.backend,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "suggestion_id": self.suggestion_id,
            "decision": self.decision,
            "decided_by_coder_id": self.decided_by_coder_id,
            "decided_at": self.decided_at,
            "confidence": self.confidence,
            "prompt_hash": self.prompt_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AIProvenance":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "AIProvenance payload must be an object"
            )
        if "feature" not in d:
            raise ProjectValidationError(
                "AIProvenance payload missing required key: feature"
            )
        p = cls(
            feature=str(d.get("feature", "") or ""),
            backend=str(d.get("backend", "") or ""),
            generation_model=str(d.get("generation_model", "") or ""),
            embedding_model=str(d.get("embedding_model", "") or ""),
            suggestion_id=str(d.get("suggestion_id", "") or ""),
            decision=str(d.get("decision", AI_DECISION_PENDING) or AI_DECISION_PENDING),
            decided_by_coder_id=str(d.get("decided_by_coder_id", "") or ""),
            decided_at=str(d.get("decided_at", "") or ""),
            confidence=_optional_float(d.get("confidence"), "confidence"),
            prompt_hash=str(d.get("prompt_hash", "") or ""),
            notes=str(d.get("notes", "") or ""),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if self.feature not in AI_FEATURES:
            raise ProjectValidationError(
                f"AIProvenance.feature must be one of {AI_FEATURES}; "
                f"got {self.feature!r}"
            )
        for label, val, cap in (
            ("backend", self.backend, MAX_BACKEND_NAME_LEN),
            ("generation_model", self.generation_model, MAX_MODEL_NAME_LEN),
            ("embedding_model", self.embedding_model, MAX_MODEL_NAME_LEN),
            ("prompt_hash", self.prompt_hash, MAX_PROMPT_HASH_LEN),
            ("notes", self.notes, MAX_NOTES_LEN),
        ):
            if not isinstance(val, str):
                raise ProjectValidationError(
                    f"AIProvenance.{label} must be a string"
                )
            if len(val) > cap:
                raise ProjectValidationError(
                    f"AIProvenance.{label} exceeds {cap} chars"
                )
            if not PRINTABLE_RE.match(val):
                # ``notes`` allows newlines; everything else is single-
                # line. We tighten ``notes`` only on length, not on
                # printable-ASCII, so non-Latin text is welcome.
                if label != "notes":
                    raise ProjectValidationError(
                        f"AIProvenance.{label} contains control or "
                        "non-printable characters"
                    )
        if self.suggestion_id:
            if not SUGGESTION_ID_RE.match(self.suggestion_id):
                raise ProjectValidationError(
                    "AIProvenance.suggestion_id must be 12-char hex or "
                    f"empty; got {self.suggestion_id!r}"
                )
        if self.decision not in AI_DECISIONS:
            raise ProjectValidationError(
                f"AIProvenance.decision must be one of {AI_DECISIONS}; "
                f"got {self.decision!r}"
            )
        if self.decided_by_coder_id:
            if not CODER_ID_RE.match(self.decided_by_coder_id):
                raise ProjectValidationError(
                    "AIProvenance.decided_by_coder_id must be 12-char hex "
                    f"or empty; got {self.decided_by_coder_id!r}"
                )
        if not isinstance(self.decided_at, str):
            raise ProjectValidationError(
                "AIProvenance.decided_at must be a string"
            )
        if self.confidence is not None:
            if (
                not isinstance(self.confidence, (int, float))
                or isinstance(self.confidence, bool)
            ):
                raise ProjectValidationError(
                    "AIProvenance.confidence must be a number in [0, 1] "
                    "or null"
                )
            self.confidence = float(self.confidence)
            if not (0.0 <= self.confidence <= 1.0):
                raise ProjectValidationError(
                    "AIProvenance.confidence must be in [0, 1]; got "
                    f"{self.confidence}"
                )
        # Cross-field consistency: a terminal decision should carry a
        # decided_by_coder_id (so the audit trail isn't anonymous). We
        # treat this as a *warning*, not an error, by allowing empty
        # decided_by_coder_id for backward-compat with rows written
        # before F8.9. Forward-going writers should always populate it.

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def to_application_provenance_dict(self) -> dict[str, str]:
        """Project the structured fields into the free-form dict shape
        expected by :class:`scribe.applications.Application.provenance`.

        The legacy dict used keys ``source``, ``model_id``,
        ``suggestion_id``, ``accepted_at`` (per the F4.1 docstring).
        We map our richer fields onto that same key set so existing
        consumers keep working without changes:

        * ``source`` is set to ``ai_accepted`` or ``ai_modified`` to
          match :data:`scribe.applications.APPLICATION_PROVENANCE_SOURCES`.
          For ``pending`` / ``rejected`` we *omit* ``source`` because
          those decisions don't produce an Application — emitting a
          value not in the closed Application vocabulary would fail
          validation downstream.
        * ``model_id`` is ``generation_model`` (or empty if unset).
        * ``embedding_model`` / ``suggestion_id`` carry through verbatim.
        * ``accepted_at`` is ``decided_at`` for accepted/modified.
        * ``feature`` and ``backend`` are added so reports can group by
          AI engine. Both are camelCase-safe (kebab/letters/digits).

        Empty values are omitted so the resulting dict stays compact
        and so it round-trips through Application's ``provenance`` cap
        (16 keys).
        """
        out: dict[str, str] = {}
        # Map decision → legacy "source" vocabulary used on Application.
        # The legacy Application vocabulary only allows ``ai_accepted``
        # and ``ai_modified``; for ``pending`` / ``rejected`` we omit
        # ``source`` entirely (those decisions don't produce an
        # Application, so the field would just fail downstream
        # validation).
        source_map = {
            AI_DECISION_ACCEPTED: "ai_accepted",
            AI_DECISION_MODIFIED: "ai_modified",
        }
        src = source_map.get(self.decision)
        if src:
            out["source"] = src
        if self.generation_model:
            out["model_id"] = self.generation_model
        if self.embedding_model:
            out["embedding_model"] = self.embedding_model
        if self.suggestion_id:
            out["suggestion_id"] = self.suggestion_id
        if self.decided_at and src:
            out["accepted_at"] = self.decided_at
        if self.feature:
            out["feature"] = self.feature
        if self.backend:
            out["backend"] = self.backend
        return out


# --------------------------------------------------------------------------- #
# AIEvent — append-only AI invocation log entry
# --------------------------------------------------------------------------- #


@dataclass
class AIEvent:
    """One entry in a project's AI event log (F8.9 / F9.6).

    An event is **append-only** — once written, the file is the audit
    record. The :func:`save_ai_event` helper guards against overwriting
    an existing event id; callers mint a fresh id with :func:`new_ai_event_id`.

    ``payload`` is a free-form dict of small scalars (and one level of
    nested dict / list of scalars), bounded by :data:`MAX_PAYLOAD_*`
    constants. It carries event-specific context — a span anchor for a
    request, a code id for a decision, an application id for an
    application event. The :class:`AIProvenance` captures *who* and
    *what model*; the payload captures *which span / code / record*.
    """

    id: str
    project_id: str
    feature: str
    kind: str
    actor_coder_id: str
    provenance: AIProvenance
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        feature: str,
        kind: str,
        actor_coder_id: str,
        provenance: AIProvenance,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        now: str | None = None,
    ) -> "AIEvent":
        ts = now or utcnow_iso()
        ev = cls(
            id=event_id or new_ai_event_id(),
            project_id=project_id,
            feature=feature,
            kind=kind,
            actor_coder_id=actor_coder_id,
            provenance=provenance,
            payload=dict(payload or {}),
            created_at=ts,
        )
        ev.validate()
        return ev

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "feature": self.feature,
            "kind": self.kind,
            "actor_coder_id": self.actor_coder_id,
            "provenance": self.provenance.to_dict(),
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AIEvent":
        if not isinstance(d, Mapping):
            raise ProjectValidationError(
                "AIEvent payload must be an object"
            )
        for required in ("id", "project_id", "feature", "kind", "provenance"):
            if required not in d:
                raise ProjectValidationError(
                    f"AIEvent payload missing required key: {required}"
                )
        ev = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            feature=str(d.get("feature", "") or ""),
            kind=str(d.get("kind", "") or ""),
            actor_coder_id=str(d.get("actor_coder_id", "") or ""),
            provenance=AIProvenance.from_dict(d["provenance"]),
            payload=_validate_payload_dict(d.get("payload") or {}),
            created_at=str(d.get("created_at", "") or ""),
        )
        ev.validate()
        return ev

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not AI_EVENT_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid AI event id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if self.feature not in AI_FEATURES:
            raise ProjectValidationError(
                f"AIEvent.feature must be one of {AI_FEATURES}; "
                f"got {self.feature!r}"
            )
        if self.kind not in AI_EVENT_KINDS:
            raise ProjectValidationError(
                f"AIEvent.kind must be one of {AI_EVENT_KINDS}; "
                f"got {self.kind!r}"
            )
        if self.actor_coder_id:
            if not CODER_ID_RE.match(self.actor_coder_id):
                raise ProjectValidationError(
                    "AIEvent.actor_coder_id must be 12-char hex or empty; "
                    f"got {self.actor_coder_id!r}"
                )
        if not isinstance(self.provenance, AIProvenance):
            raise ProjectValidationError(
                "AIEvent.provenance must be an AIProvenance instance"
            )
        self.provenance.validate()
        # A consistency check: provenance.feature should match the
        # event's feature. We enforce this so reports don't have to
        # reconcile two diverging fields.
        if self.provenance.feature != self.feature:
            raise ProjectValidationError(
                "AIEvent.feature must match provenance.feature; "
                f"got event {self.feature!r} vs provenance "
                f"{self.provenance.feature!r}"
            )
        # Re-validate payload shape (covers the in-memory mutation case
        # where a caller poked at .payload after construction).
        self.payload = _validate_payload_dict(self.payload)


# --------------------------------------------------------------------------- #
# Payload validation
# --------------------------------------------------------------------------- #


_PAYLOAD_KEY_RE = re.compile(r"^[A-Za-z][\w\-]{0,63}$")


def _validate_payload_dict(raw: Any) -> dict[str, Any]:
    """Validate (and normalise) an event payload dict.

    Mirrors :func:`scribe.projects._validate_settings_dict` but tuned
    for AI event payloads — flat dict of scalars plus one level of
    nested dict or list of scalars, bounded by :data:`MAX_PAYLOAD_*`.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ProjectValidationError(
            "AIEvent.payload must be an object of string→scalar"
        )
    out: dict[str, Any] = {}
    if len(raw) > MAX_PAYLOAD_KEYS:
        raise ProjectValidationError(
            f"AIEvent.payload may have at most {MAX_PAYLOAD_KEYS} keys"
        )
    for raw_k, raw_v in raw.items():
        k = str(raw_k)
        if not _PAYLOAD_KEY_RE.match(k):
            raise ProjectValidationError(
                f"AIEvent.payload key {k!r} must match "
                "letters/digits/underscore/hyphen, ≤64 chars, "
                "starting with a letter"
            )
        out[k] = _validate_payload_value(raw_v, depth=0, path=k)
    # Enforce overall serialised-size ceiling to keep audit-trail
    # payloads small.
    encoded = json.dumps(out, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ProjectValidationError(
            f"AIEvent.payload exceeds {MAX_PAYLOAD_BYTES} bytes when serialised"
        )
    return out


def _validate_payload_value(v: Any, *, depth: int, path: str) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        if len(v) > MAX_PAYLOAD_STRING_LEN:
            raise ProjectValidationError(
                f"AIEvent.payload[{path}] string exceeds "
                f"{MAX_PAYLOAD_STRING_LEN} chars"
            )
        return v
    if isinstance(v, list):
        if depth > 0:
            raise ProjectValidationError(
                f"AIEvent.payload[{path}] nested list not allowed beyond "
                "one level"
            )
        if len(v) > MAX_PAYLOAD_LIST_LEN:
            raise ProjectValidationError(
                f"AIEvent.payload[{path}] list exceeds "
                f"{MAX_PAYLOAD_LIST_LEN} entries"
            )
        return [
            _validate_payload_value(item, depth=depth + 1, path=f"{path}[{i}]")
            for i, item in enumerate(v)
        ]
    if isinstance(v, dict):
        if depth > 0:
            raise ProjectValidationError(
                f"AIEvent.payload[{path}] nested object not allowed beyond "
                "one level"
            )
        if len(v) > MAX_PAYLOAD_KEYS:
            raise ProjectValidationError(
                f"AIEvent.payload[{path}] object has too many keys"
            )
        return {
            str(k): _validate_payload_value(
                vv, depth=depth + 1, path=f"{path}.{k}"
            )
            for k, vv in v.items()
        }
    raise ProjectValidationError(
        f"AIEvent.payload[{path}] unsupported type {type(v).__name__}"
    )


# --------------------------------------------------------------------------- #
# Optional-numeric coercion
# --------------------------------------------------------------------------- #


def _optional_float(v: Any, field_name: str) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise ProjectValidationError(
            f"{field_name} must be a number or null; got bool"
        )
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError as e:
            raise ProjectValidationError(
                f"{field_name} must be a number or null; got {v!r}"
            ) from e
    raise ProjectValidationError(
        f"{field_name} must be a number or null; got {type(v).__name__}"
    )


# --------------------------------------------------------------------------- #
# Hashing helper — short prompt digest
# --------------------------------------------------------------------------- #


def hash_prompt(prompt: str, *, length: int = 16) -> str:
    """Return a short hex digest of the prompt for prompt_hash.

    sha256 truncated to ``length`` hex chars. ``length`` clamps to
    [4, 64]. Pure helper; deterministic; no side effects.
    """
    if not isinstance(prompt, str):
        raise ProjectValidationError("hash_prompt requires a string")
    n = max(4, min(64, int(length)))
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:n]


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_ai_event_id() -> str:
    """Mint a new 12-char hex AI event id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def ai_events_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's AI events."""
    return project_dir(projects_root, project_id) / AI_EVENTS_DIRNAME


def ai_event_state_path(
    projects_root: Path, project_id: str, event_id: str
) -> Path:
    """Return the path for a single AI event JSON file."""
    if not AI_EVENT_ID_RE.match(event_id):
        raise ProjectValidationError(f"Invalid AI event id: {event_id!r}")
    return ai_events_dir(projects_root, project_id) / f"{event_id}.json"


def save_ai_event(projects_root: Path, event: AIEvent) -> Path:
    """Persist an AI event atomically; refuses to overwrite.

    AI events are append-only — once written, the on-disk record is the
    audit trail. Re-saving an existing id raises :class:`FileExistsError`.
    Use a fresh id (``new_ai_event_id``) for every invocation.
    """
    event.validate()
    parent = project_dir(projects_root, event.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving AI events."
        )
    ed = ai_events_dir(projects_root, event.project_id)
    ed.mkdir(parents=True, exist_ok=True)
    target = ai_event_state_path(projects_root, event.project_id, event.id)
    if target.exists():
        raise FileExistsError(
            f"AI event {event.id} already exists; events are append-only"
        )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(event.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_ai_event(
    projects_root: Path, project_id: str, event_id: str
) -> AIEvent:
    """Load an AI event by id. Raises ``FileNotFoundError`` if missing."""
    p = ai_event_state_path(projects_root, project_id, event_id)
    if not p.exists():
        raise FileNotFoundError(f"No AI event at {p}")
    return AIEvent.from_dict(json.loads(p.read_text()))


def list_ai_events(
    projects_root: Path,
    project_id: str,
    *,
    feature: str | None = None,
    kind: str | None = None,
    actor_coder_id: str | None = None,
) -> list[AIEvent]:
    """List AI events in a project, optionally filtered.

    Filters AND-combine. Skips files that don't parse so a single
    corrupt event doesn't break the view (matches the rest of the
    F-feature stack). Sorted by ``created_at`` ascending so the natural
    reading order is the order events were emitted.
    """
    if feature is not None and feature not in AI_FEATURES:
        raise ProjectValidationError(f"Invalid feature filter: {feature!r}")
    if kind is not None and kind not in AI_EVENT_KINDS:
        raise ProjectValidationError(f"Invalid kind filter: {kind!r}")
    if actor_coder_id is not None and not CODER_ID_RE.match(actor_coder_id):
        raise ProjectValidationError(
            f"Invalid actor_coder_id filter: {actor_coder_id!r}"
        )
    ed = ai_events_dir(projects_root, project_id)
    if not ed.exists():
        return []
    out: list[AIEvent] = []
    for f in sorted(ed.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        eid = f.stem
        if not AI_EVENT_ID_RE.match(eid):
            continue
        try:
            ev = AIEvent.from_dict(json.loads(f.read_text()))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
        if feature is not None and ev.feature != feature:
            continue
        if kind is not None and ev.kind != kind:
            continue
        if actor_coder_id is not None and ev.actor_coder_id != actor_coder_id:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.created_at, e.id))
    return out


# --------------------------------------------------------------------------- #
# Per-engine extractors
# --------------------------------------------------------------------------- #


def provenance_from_code_suggestion(
    suggestion: Any, *, backend: str = ""
) -> AIProvenance:
    """Build an :class:`AIProvenance` from a F8.3 :class:`CodeSuggestion`.

    Reads only the fields the per-engine record exposes; we keep the
    parameter type loose (``Any``) so this module doesn't have to import
    code_suggestions and risk a circular import.
    """
    return AIProvenance.new(
        feature=AI_FEATURE_CODE_SUGGESTION,
        backend=backend,
        generation_model=getattr(suggestion, "generation_model", "") or "",
        embedding_model=getattr(suggestion, "embedding_model", "") or "",
        suggestion_id=getattr(suggestion, "id", "") or "",
        decision=_normalise_decision(
            getattr(suggestion, "decision", AI_DECISION_PENDING)
        ),
        decided_by_coder_id=getattr(suggestion, "decided_by_coder_id", "") or "",
        decided_at=getattr(suggestion, "decided_at", "") or "",
    )


def provenance_from_new_code_suggestion(
    suggestion: Any, *, backend: str = ""
) -> AIProvenance:
    """Build an :class:`AIProvenance` from a F8.4 NewCodeSuggestion."""
    return AIProvenance.new(
        feature=AI_FEATURE_NEW_CODE_SUGGESTION,
        backend=backend,
        generation_model=getattr(suggestion, "generation_model", "") or "",
        embedding_model=getattr(suggestion, "embedding_model", "") or "",
        suggestion_id=getattr(suggestion, "id", "") or "",
        decision=_normalise_decision(
            getattr(suggestion, "decision", AI_DECISION_PENDING)
        ),
        decided_by_coder_id=getattr(suggestion, "decided_by_coder_id", "") or "",
        decided_at=getattr(suggestion, "decided_at", "") or "",
    )


def provenance_from_memo_draft(
    draft: Any, *, backend: str = ""
) -> AIProvenance:
    """Build an :class:`AIProvenance` from a F8.8 MemoDraft."""
    return AIProvenance.new(
        feature=AI_FEATURE_MEMO_DRAFT,
        backend=backend,
        generation_model=getattr(draft, "generation_model", "") or "",
        suggestion_id=getattr(draft, "id", "") or "",
        decision=_normalise_decision(
            getattr(draft, "decision", AI_DECISION_PENDING)
        ),
        decided_by_coder_id=getattr(draft, "decided_by_coder_id", "") or "",
        decided_at=getattr(draft, "decided_at", "") or "",
    )


def provenance_from_transcript_review_pass(
    pass_record: Any, *, backend: str = ""
) -> AIProvenance:
    """Build an :class:`AIProvenance` from a F8.6 transcript-review pass.

    A review pass is a *background job* — it doesn't itself have a
    decision; each per-item suggestion does. The provenance returned
    here represents the *invocation*, with ``decision=pending``.
    """
    return AIProvenance.new(
        feature=AI_FEATURE_TRANSCRIPT_REVIEW,
        backend=backend,
        generation_model=getattr(pass_record, "generation_model", "") or "",
        embedding_model=getattr(pass_record, "embedding_model", "") or "",
        suggestion_id=getattr(pass_record, "id", "") or "",
        decision=AI_DECISION_PENDING,
    )


def provenance_from_second_coder_pass(
    pass_record: Any, *, backend: str = ""
) -> AIProvenance:
    """Build an :class:`AIProvenance` from a F8.7 second-coder pass."""
    return AIProvenance.new(
        feature=AI_FEATURE_SECOND_CODER,
        backend=backend,
        generation_model=getattr(pass_record, "generation_model", "") or "",
        embedding_model=getattr(pass_record, "embedding_model", "") or "",
        suggestion_id=getattr(pass_record, "id", "") or "",
        decision=AI_DECISION_PENDING,
    )


def _normalise_decision(raw: Any) -> str:
    """Coerce an arbitrary decision string onto :data:`AI_DECISIONS`.

    Returns ``pending`` for anything we don't recognise so the
    downstream ``validate`` doesn't blow up on a per-engine vocabulary
    that drifts. Per-engine vocabularies all happen to match today, but
    this guard keeps F8.9 robust against future churn.
    """
    s = str(raw or AI_DECISION_PENDING).strip().lower()
    if s in AI_DECISIONS:
        return s
    return AI_DECISION_PENDING
