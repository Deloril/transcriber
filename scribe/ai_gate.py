"""First-N-transcripts AI-off mode (F8.10).

Per PLANNING.md F8.10:

  > Configurable threshold (default: AI suggestions disabled until
  > codebook has ≥ 8 codes AND ≥ 2 transcripts hand-coded). Rationale:
  > protects the inductive opening of grounded theory.

Constructivist grounded theory (Charmaz) and most other inductive
methodologies depend on the researcher *opening up* their data before
imposing categories. If an LLM starts suggesting code names on the
first paragraph of the first transcript, the analyst is no longer in
the inductive frame — they are evaluating someone else's categories.
The methodological literature is unambiguous about why this matters
(see ``docs/research/coding-engine-research.md`` for citations) and
this module is the technical guard rail.

Design
------

1. **A query, not a hard wall.** ``evaluate_project_ai_gate`` returns a
   :class:`AIGateStatus` describing whether AI is allowed *right now*,
   and *why or why not*. Callers (HTTP endpoints in front of each AI
   feature, the UI status badge, ``ai/gate`` REST endpoint) consult the
   gate before invoking AI; if it's blocked they refuse the call and
   surface the reason.

2. **Project-scoped configuration.** ``AIGateConfig`` lives in
   ``project.settings["ai_gate"]``. Every field is optional and
   defaults are baked into :func:`default_ai_gate_config`. The
   defaults are the spec (8 codes, 2 hand-coded sources).

3. **Override switch.** Some research frames (deductive, framework
   analysis) don't need the inductive-opening protection at all. The
   ``override`` field has three values:

     * ``auto`` (default) — the threshold is checked.
     * ``force_off`` — the gate is permanently closed regardless of
       counts. Useful for teaching / pre-registration corpora.
     * ``force_on`` — the gate is permanently open. The audit trail
       still records that this project bypassed the gate, so reviewers
       see it.

4. **Per-feature exemption list.** Some AI features (F8.5 quote
   similarity — "show me semantically similar passages") arguably
   don't impose categories at all and can usefully run on transcript
   1. ``exempt_features`` (a list of :data:`AI_FEATURES` ids) lets the
   user carve out exceptions; default is empty (every feature gated)
   to preserve the inductive default.

5. **What counts as "hand-coded".** A source is "hand-coded" if it has
   at least one ``Application`` whose provenance is *not* AI:

     * ``ai_provenance`` is None (the structured F8.9 stamp), AND
     * ``provenance.get("source")`` is not one of the ``ai_*`` markers.

   Counting *applications* (not sources) would over-reward a single
   transcript with a hundred codings; counting *sources* matches the
   spec's "≥ 2 transcripts hand-coded".

Boundaries
----------

* **No HTTP / FastAPI surface here.** F8.10's REST surface is added
  in :mod:`scribe.server` separately; this module is pure Python so
  tests stay fast.
* **No automatic blocking of AI engine call sites.** The existing
  AI engines (:mod:`scribe.code_suggestions` etc.) aren't modified.
  Callers (HTTP layer, future CLI) must consult the gate explicitly.
  This keeps the gate's *policy* decoupled from the engines'
  *mechanism*; future iterations can add the wiring without re-
  designing F8.10 itself.
* **No wall-clock dependencies.** Every helper takes counts /
  applications as inputs so tests don't need a live disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ai_provenance import AI_FEATURES
from .applications import Application, list_applications
from .codes import list_codes
from .projects import (
    Project,
    ProjectValidationError,
    load_project,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Settings key. Stored as a one-level-nested dict under
# ``project.settings["ai_gate"]``; all values are scalars so they
# satisfy ``_validate_settings_value``.
SETTING_AI_GATE = "ai_gate"

# Keys inside the ``ai_gate`` settings dict.
SETTING_KEY_MIN_CODES = "min_codes"
SETTING_KEY_MIN_HAND_CODED_SOURCES = "min_hand_coded_sources"
SETTING_KEY_OVERRIDE = "override"
SETTING_KEY_ENABLED = "enabled"

# ``exempt_features`` is a *list*, which can't live inside the nested
# ``ai_gate`` dict (Project settings allows lists at depth 0 but not
# inside another dict — see ``_validate_settings_value``). It lives at
# the top level of ``project.settings`` as a sibling key, mirroring the
# ``ai_backend`` / ``ai_backend_headers`` split.
SETTING_AI_GATE_EXEMPT_FEATURES = "ai_gate_exempt_features"

# Override values. ``auto`` honours the thresholds; ``force_off`` /
# ``force_on`` short-circuit them in opposite directions.
GATE_OVERRIDE_AUTO = "auto"
GATE_OVERRIDE_FORCE_OFF = "force_off"
GATE_OVERRIDE_FORCE_ON = "force_on"
GATE_OVERRIDES: tuple[str, ...] = (
    GATE_OVERRIDE_AUTO,
    GATE_OVERRIDE_FORCE_OFF,
    GATE_OVERRIDE_FORCE_ON,
)

# Defaults from PLANNING.md F8.10.
DEFAULT_MIN_CODES = 8
DEFAULT_MIN_HAND_CODED_SOURCES = 2

# Bounds. The thresholds are positive integers, capped so a UI bug
# can't write "minimum 1e9 codes". 10 000 is well above any
# realistically large codebook (Charmaz herself worked with ~150).
MIN_THRESHOLD = 0
MAX_THRESHOLD = 10_000

# Reason codes for ``AIGateStatus``. Stable strings so the UI can
# branch on them; the human ``message`` is for display only.
REASON_FORCE_OFF = "force_off"
REASON_FORCE_ON = "force_on"
REASON_DISABLED = "disabled"
REASON_FEATURE_EXEMPT = "feature_exempt"
REASON_INSUFFICIENT_CODES = "insufficient_codes"
REASON_INSUFFICIENT_HAND_CODED_SOURCES = "insufficient_hand_coded_sources"
REASON_INSUFFICIENT_BOTH = "insufficient_both"
REASON_THRESHOLD_MET = "threshold_met"
REASONS: tuple[str, ...] = (
    REASON_FORCE_OFF,
    REASON_FORCE_ON,
    REASON_DISABLED,
    REASON_FEATURE_EXEMPT,
    REASON_INSUFFICIENT_CODES,
    REASON_INSUFFICIENT_HAND_CODED_SOURCES,
    REASON_INSUFFICIENT_BOTH,
    REASON_THRESHOLD_MET,
)

# Provenance markers we treat as "the AI did this". Mirrors the
# closed ``provenance.source`` vocabulary in
# :data:`scribe.applications.APPLICATION_PROVENANCE_SOURCES` —
# specifically the ``ai_*`` members. ``imported`` and ``other`` are
# treated as human (an imported transcript still represents a human
# coding decision somewhere; ``other`` escape-hatch is rare and
# defaulting to "human" is safe — the worst case is the gate opens
# slightly earlier than the spec wants, never later).
_AI_PROVENANCE_MARKERS: tuple[str, ...] = (
    "ai_accepted",
    "ai_modified",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AIGateConfig:
    """How the AI-off gate behaves for a given project.

    All fields have sensible defaults from PLANNING.md F8.10. Construct
    via :meth:`new` to apply defaults + validation in one step.
    """

    min_codes: int = DEFAULT_MIN_CODES
    min_hand_coded_sources: int = DEFAULT_MIN_HAND_CODED_SOURCES
    override: str = GATE_OVERRIDE_AUTO
    enabled: bool = True
    # Features (from :data:`scribe.ai_provenance.AI_FEATURES`) that
    # bypass the gate entirely. Default empty: every AI feature is
    # gated until thresholds are met.
    exempt_features: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # Construction / validation
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        min_codes: int = DEFAULT_MIN_CODES,
        min_hand_coded_sources: int = DEFAULT_MIN_HAND_CODED_SOURCES,
        override: str = GATE_OVERRIDE_AUTO,
        enabled: bool = True,
        exempt_features: Iterable[str] | None = None,
    ) -> "AIGateConfig":
        feats = tuple(str(f) for f in (exempt_features or ()))
        c = cls(
            min_codes=int(min_codes),
            min_hand_coded_sources=int(min_hand_coded_sources),
            override=str(override),
            enabled=bool(enabled),
            exempt_features=feats,
        )
        c.validate()
        return c

    def validate(self) -> None:
        if not isinstance(self.min_codes, int) or isinstance(self.min_codes, bool):
            raise ProjectValidationError("min_codes must be an int")
        if not (MIN_THRESHOLD <= self.min_codes <= MAX_THRESHOLD):
            raise ProjectValidationError(
                f"min_codes must be in [{MIN_THRESHOLD}, {MAX_THRESHOLD}]; "
                f"got {self.min_codes}"
            )
        if (
            not isinstance(self.min_hand_coded_sources, int)
            or isinstance(self.min_hand_coded_sources, bool)
        ):
            raise ProjectValidationError("min_hand_coded_sources must be an int")
        if not (
            MIN_THRESHOLD <= self.min_hand_coded_sources <= MAX_THRESHOLD
        ):
            raise ProjectValidationError(
                f"min_hand_coded_sources must be in "
                f"[{MIN_THRESHOLD}, {MAX_THRESHOLD}]; got "
                f"{self.min_hand_coded_sources}"
            )
        if self.override not in GATE_OVERRIDES:
            raise ProjectValidationError(
                f"override must be one of {GATE_OVERRIDES}; "
                f"got {self.override!r}"
            )
        if not isinstance(self.enabled, bool):
            raise ProjectValidationError("enabled must be a bool")
        seen: set[str] = set()
        for f in self.exempt_features:
            if not isinstance(f, str):
                raise ProjectValidationError(
                    "exempt_features entries must be strings"
                )
            if f not in AI_FEATURES:
                raise ProjectValidationError(
                    f"exempt_features contains unknown feature {f!r}; "
                    f"must be one of {AI_FEATURES}"
                )
            if f in seen:
                raise ProjectValidationError(
                    f"exempt_features has duplicate {f!r}"
                )
            seen.add(f)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return only the **scalar** fields, suitable for storing as
        ``project.settings["ai_gate"]``. ``exempt_features`` is a list
        and lives in a sibling top-level setting (see
        :func:`store_ai_gate_config`).
        """
        return {
            SETTING_KEY_MIN_CODES: self.min_codes,
            SETTING_KEY_MIN_HAND_CODED_SOURCES: self.min_hand_coded_sources,
            SETTING_KEY_OVERRIDE: self.override,
            SETTING_KEY_ENABLED: self.enabled,
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any] | None,
        *,
        exempt_features: Iterable[str] | None = None,
    ) -> "AIGateConfig":
        """Inverse of :meth:`to_dict`.

        ``exempt_features`` is passed in separately because it lives
        under a sibling settings key (see notes at module top).
        """
        d = dict(d or {})
        return cls.new(
            min_codes=_coerce_int(
                d.get(SETTING_KEY_MIN_CODES, DEFAULT_MIN_CODES),
                SETTING_KEY_MIN_CODES,
            ),
            min_hand_coded_sources=_coerce_int(
                d.get(
                    SETTING_KEY_MIN_HAND_CODED_SOURCES,
                    DEFAULT_MIN_HAND_CODED_SOURCES,
                ),
                SETTING_KEY_MIN_HAND_CODED_SOURCES,
            ),
            override=str(
                d.get(SETTING_KEY_OVERRIDE, GATE_OVERRIDE_AUTO)
                or GATE_OVERRIDE_AUTO
            ),
            enabled=_coerce_bool(
                d.get(SETTING_KEY_ENABLED, True),
                SETTING_KEY_ENABLED,
            ),
            exempt_features=exempt_features,
        )


def _coerce_int(v: Any, label: str) -> int:
    """Settings come back from JSON as int OR str; allow either."""
    if isinstance(v, bool):
        # bool is an int subclass — refuse it explicitly so we don't
        # silently treat ``True`` as ``min_codes=1``.
        raise ProjectValidationError(f"{label} must be an int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ProjectValidationError(f"{label} must be a whole number")
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as e:
            raise ProjectValidationError(f"{label} must be an int") from e
    raise ProjectValidationError(f"{label} must be an int")


def _coerce_bool(v: Any, label: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "on"):
            return True
        if s in ("false", "no", "0", "off"):
            return False
    raise ProjectValidationError(f"{label} must be a bool")


def default_ai_gate_config() -> AIGateConfig:
    """Return the spec defaults: 8 codes AND 2 hand-coded transcripts."""
    return AIGateConfig.new()


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AIGateStatus:
    """The result of asking "may we run AI right now?".

    ``allowed`` is the binary answer the caller acts on; ``reason`` is
    a stable code (one of :data:`REASONS`); ``message`` is the human
    string for display. Numeric counts let the UI show progress
    ("3/8 codes, 1/2 transcripts hand-coded").
    """

    allowed: bool
    reason: str
    message: str
    code_count: int
    hand_coded_source_count: int
    min_codes: int
    min_hand_coded_sources: int
    override: str
    enabled: bool
    feature: str = ""
    feature_exempt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Hand-coded source counting
# --------------------------------------------------------------------------- #


def is_application_human(application: Application) -> bool:
    """True when an Application has no AI fingerprint.

    "Human" means:

      * No structured F8.9 :class:`AIProvenance` attached, AND
      * No legacy free-form ``provenance.source`` with an ``ai_*``
        marker.

    Anything else (including missing-or-empty provenance) is human.
    Defensive: a future AI engine that forgets to stamp provenance
    will silently *under*-gate (i.e. count its applications as
    human-coded). That's the wrong direction for the threshold but
    still safe — the worst case is the gate opens slightly earlier
    than the spec wants, never later.
    """
    if application.ai_provenance is not None:
        return False
    src = ""
    prov = application.provenance or {}
    if isinstance(prov, dict):
        src = str(prov.get("source", "") or "").lower()
    if src in _AI_PROVENANCE_MARKERS:
        return False
    return True


def count_hand_coded_sources(applications: Sequence[Application]) -> int:
    """Count distinct ``source_id``s that have at least one human
    application. Order-independent; deterministic.
    """
    seen: set[str] = set()
    for a in applications:
        if not is_application_human(a):
            continue
        if a.source_id:
            seen.add(a.source_id)
    return len(seen)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_ai_gate(
    *,
    config: AIGateConfig,
    code_count: int,
    hand_coded_source_count: int,
    feature: str = "",
) -> AIGateStatus:
    """Pure function: given a config + counts, return the gate status.

    ``feature`` is one of :data:`scribe.ai_provenance.AI_FEATURES`; if
    set and present in ``config.exempt_features``, the gate opens
    regardless of thresholds. An empty string means "the caller hasn't
    declared a feature yet" and uses the strictest path.
    """
    config.validate()
    if code_count < 0 or hand_coded_source_count < 0:
        raise ProjectValidationError(
            "code_count and hand_coded_source_count must be ≥ 0"
        )
    if feature and feature not in AI_FEATURES:
        raise ProjectValidationError(
            f"feature must be one of {AI_FEATURES} (or empty); "
            f"got {feature!r}"
        )

    # 1. Gate disabled (the user doesn't want guard rails at all).
    if not config.enabled:
        return AIGateStatus(
            allowed=True,
            reason=REASON_DISABLED,
            message=(
                "AI gate is disabled for this project — AI features are "
                "available without thresholds."
            ),
            code_count=code_count,
            hand_coded_source_count=hand_coded_source_count,
            min_codes=config.min_codes,
            min_hand_coded_sources=config.min_hand_coded_sources,
            override=config.override,
            enabled=False,
            feature=feature,
            feature_exempt=False,
        )

    # 2. Override forces a hard answer.
    if config.override == GATE_OVERRIDE_FORCE_OFF:
        return AIGateStatus(
            allowed=False,
            reason=REASON_FORCE_OFF,
            message=(
                "AI features are forced off for this project. Change "
                "the override in project settings to re-enable."
            ),
            code_count=code_count,
            hand_coded_source_count=hand_coded_source_count,
            min_codes=config.min_codes,
            min_hand_coded_sources=config.min_hand_coded_sources,
            override=config.override,
            enabled=True,
            feature=feature,
            feature_exempt=False,
        )
    if config.override == GATE_OVERRIDE_FORCE_ON:
        return AIGateStatus(
            allowed=True,
            reason=REASON_FORCE_ON,
            message=(
                "AI features are forced on for this project (gate "
                "thresholds bypassed)."
            ),
            code_count=code_count,
            hand_coded_source_count=hand_coded_source_count,
            min_codes=config.min_codes,
            min_hand_coded_sources=config.min_hand_coded_sources,
            override=config.override,
            enabled=True,
            feature=feature,
            feature_exempt=False,
        )

    # 3. Per-feature exemption (e.g. F8.5 quote similarity).
    if feature and feature in config.exempt_features:
        return AIGateStatus(
            allowed=True,
            reason=REASON_FEATURE_EXEMPT,
            message=(
                f"Feature {feature!r} is exempt from the AI gate for "
                "this project."
            ),
            code_count=code_count,
            hand_coded_source_count=hand_coded_source_count,
            min_codes=config.min_codes,
            min_hand_coded_sources=config.min_hand_coded_sources,
            override=config.override,
            enabled=True,
            feature=feature,
            feature_exempt=True,
        )

    # 4. Threshold check.
    codes_ok = code_count >= config.min_codes
    sources_ok = hand_coded_source_count >= config.min_hand_coded_sources
    if codes_ok and sources_ok:
        return AIGateStatus(
            allowed=True,
            reason=REASON_THRESHOLD_MET,
            message=(
                "Threshold met: codebook has "
                f"{code_count}/{config.min_codes} codes and "
                f"{hand_coded_source_count}/{config.min_hand_coded_sources} "
                "transcripts hand-coded."
            ),
            code_count=code_count,
            hand_coded_source_count=hand_coded_source_count,
            min_codes=config.min_codes,
            min_hand_coded_sources=config.min_hand_coded_sources,
            override=config.override,
            enabled=True,
            feature=feature,
            feature_exempt=False,
        )

    if not codes_ok and not sources_ok:
        reason = REASON_INSUFFICIENT_BOTH
        message = (
            "AI suggestions are disabled until the codebook has at "
            f"least {config.min_codes} codes "
            f"(currently {code_count}) AND at least "
            f"{config.min_hand_coded_sources} transcripts have been "
            f"hand-coded (currently {hand_coded_source_count}). "
            "This protects the inductive opening of grounded theory."
        )
    elif not codes_ok:
        reason = REASON_INSUFFICIENT_CODES
        message = (
            "AI suggestions are disabled until the codebook has at "
            f"least {config.min_codes} codes "
            f"(currently {code_count}). This protects the inductive "
            "opening of grounded theory."
        )
    else:
        reason = REASON_INSUFFICIENT_HAND_CODED_SOURCES
        message = (
            "AI suggestions are disabled until at least "
            f"{config.min_hand_coded_sources} transcripts have been "
            f"hand-coded (currently {hand_coded_source_count}). "
            "This protects the inductive opening of grounded theory."
        )
    return AIGateStatus(
        allowed=False,
        reason=reason,
        message=message,
        code_count=code_count,
        hand_coded_source_count=hand_coded_source_count,
        min_codes=config.min_codes,
        min_hand_coded_sources=config.min_hand_coded_sources,
        override=config.override,
        enabled=True,
        feature=feature,
        feature_exempt=False,
    )


def evaluate_project_ai_gate(
    projects_root: Path,
    project_id: str,
    *,
    feature: str = "",
) -> AIGateStatus:
    """Convenience: load the project + codes + applications, evaluate.

    Reads the project from disk, counts codes, counts distinct
    hand-coded source ids, and calls :func:`evaluate_ai_gate`. Use
    this from HTTP endpoints; tests can call the pure function with
    explicit counts and skip the disk hop.
    """
    project = load_project(projects_root, project_id)
    config = load_ai_gate_config(project)
    codes = list_codes(projects_root, project_id)
    code_count = len(codes)
    applications = list_applications(projects_root, project_id)
    hand_coded = count_hand_coded_sources(applications)
    return evaluate_ai_gate(
        config=config,
        code_count=code_count,
        hand_coded_source_count=hand_coded,
        feature=feature,
    )


# --------------------------------------------------------------------------- #
# Project-settings integration
# --------------------------------------------------------------------------- #


def load_ai_gate_config(project: Project) -> AIGateConfig:
    """Read the AI-gate config from a Project's settings.

    Old projects (no ``ai_gate`` key) get the spec defaults; this is
    deliberate — F8.10 wants the gate ON by default. The ``enabled``
    knob is for users who explicitly want to opt out of guard rails;
    "no settings recorded" means "use the spec".
    """
    settings = project.settings or {}
    raw = settings.get(SETTING_AI_GATE)
    if raw is None:
        raw_dict: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        raw_dict = dict(raw)
    else:
        raise ProjectValidationError(
            f"project.settings[{SETTING_AI_GATE!r}] must be an object"
        )
    feats_raw = settings.get(SETTING_AI_GATE_EXEMPT_FEATURES)
    if feats_raw is None:
        feats: tuple[str, ...] = ()
    elif isinstance(feats_raw, list):
        feats = tuple(str(f) for f in feats_raw)
    else:
        raise ProjectValidationError(
            f"project.settings[{SETTING_AI_GATE_EXEMPT_FEATURES!r}] "
            "must be a list"
        )
    return AIGateConfig.from_dict(raw_dict, exempt_features=feats)


def store_ai_gate_config(project: Project, config: AIGateConfig) -> None:
    """Persist a gate config into ``project.settings``.

    Mutates the project in place. Validates first so a bad config
    never lands on disk. Empty ``exempt_features`` removes the
    sibling key so settings stays tidy.
    """
    config.validate()
    settings = dict(project.settings or {})
    settings[SETTING_AI_GATE] = config.to_dict()
    if config.exempt_features:
        settings[SETTING_AI_GATE_EXEMPT_FEATURES] = list(config.exempt_features)
    else:
        settings.pop(SETTING_AI_GATE_EXEMPT_FEATURES, None)
    project.apply_update({"settings": settings})
