"""Baked-in model recommendations per tier (F8.12).

Per PLANNING.md F8.12:

  > Model recommendations baked in: laptop default Llama 3.2 3B or
  > Phi-3.5 3.8B; mid-tier Phi-4 14B or Mistral Nemo 12B; large-tier
  > Qwen 2.5 32B or Llama 3.3 70B. Embedding default ``bge-m3``
  > (multilingual) or ``nomic-embed-text-v1.5``.

This module pairs concrete model picks with the tiers F8.11 produced.
The tier picker (``scribe.model_tiers``) decides which tier the user's
hardware can run; this module decides *which models* to suggest within
that tier. The two are kept separate on purpose:

* ``model_tiers`` is hardware-only and stable. It says "you can run
  small/mid/large" and nothing about specific weights.
* ``model_recommendations`` is opinionated and likely to change as the
  open-weights ecosystem moves. Updating model picks here doesn't
  require touching tier shapes or the autodetection logic.

Design notes
------------

* **Pure data + lookup.** No HTTP, no model loads, no network. The
  *download* of a recommended model is the existing F8.11 pull-manager
  (``OllamaBackend.pull_model``) — this module just hands the UI the
  Ollama-compatible tag.
* **One default per tier.** Each tier has ≥1 recommended model and
  exactly one ``is_default=True`` pick. The default is what the UI
  pre-selects; the alternatives let users choose for licence /
  language / family reasons.
* **Embeddings are not tier-bound.** They're tiny relative to LLMs and
  fit on every tier. The picker exposes them as a flat list with one
  multilingual default (``bge-m3``) and one English-leaning alternative
  (``nomic-embed-text:v1.5``).
* **Wire format colocated with the data.** ``summarise_recommendations``
  returns the same ``{hardware, tiers, recommended}`` shape as
  ``model_tiers.summarise`` plus per-tier ``recommended_models`` and a
  top-level ``embedding_models`` array. Server tests assert that shape;
  the JS UI consumes it directly without re-deriving anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from . import model_tiers as _mt


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

KIND_GENERATIVE = "generative"
KIND_EMBEDDING = "embedding"

KNOWN_KINDS: tuple[str, ...] = (KIND_GENERATIVE, KIND_EMBEDDING)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecommendedModel:
    """One concrete model pick suggested for a tier (or embeddings).

    The ``tag`` is the Ollama-compatible model tag — what the user
    types into ``ollama pull`` or what we send to ``POST /api/pull``
    (see :func:`scribe.ai_backend.OllamaBackend.pull_model`).

    Tiering (``tier_id``) is only meaningful for ``KIND_GENERATIVE``.
    Embedding models live outside the tier system — they fit on any
    machine — so they carry ``tier_id=""``.
    """

    tag: str
    """Ollama tag, e.g. ``"llama3.2:3b"``. Empty string is invalid."""

    display_name: str
    """Human-friendly label, e.g. ``"Llama 3.2 3B"``."""

    family: str
    """Model family slug: ``"llama"``, ``"phi"``, ``"qwen"``, etc."""

    parameter_size_b: float
    """Parameter count in billions. ``0.0`` if not applicable (e.g.
    embedding models that quote dimensions instead)."""

    kind: str
    """One of :data:`KNOWN_KINDS`."""

    tier_id: str
    """One of :data:`scribe.model_tiers.KNOWN_TIER_IDS`, or ``""`` for
    embedding models."""

    is_default: bool
    """The pre-selected pick within its tier (or within the embedding
    pool). Exactly one default per tier; exactly one default
    embedding."""

    licence: str
    """Short licence label — surface-only, not legal advice."""

    multilingual: bool
    """Whether the model is documented to handle non-English data well.
    The grounded-theory researcher-with-Spanish-interviews case is
    exactly the one that breaks if this lies, so be conservative."""

    context_window: int
    """Documented max context window in tokens. ``0`` if unknown."""

    notes: str
    """One-sentence rationale shown in the UI tooltip."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The recommendations themselves
# --------------------------------------------------------------------------- #
#
# These match the F8.12 spec verbatim. Order within a tier is "default
# first, then alternatives in author-judgement order"; the UI is free
# to re-sort but should treat ``is_default=True`` as the pre-selected
# pick.
#
# Every Ollama tag here was checked against the upstream library at
# the time of this commit. If a tag goes stale, fix it with a one-line
# patch — don't restructure this list.


_GENERATIVE_RECOMMENDATIONS: tuple[RecommendedModel, ...] = (
    # --- small (3–4B) -------------------------------------------------- #
    RecommendedModel(
        tag="llama3.2:3b",
        display_name="Llama 3.2 3B",
        family="llama",
        parameter_size_b=3.0,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_SMALL,
        is_default=True,
        licence="Llama 3.2 Community License",
        multilingual=True,
        context_window=128_000,
        notes=(
            "Strong instruction-following at 3B; safe laptop default. "
            "128k context fits an interview transcript whole."
        ),
    ),
    RecommendedModel(
        tag="phi3.5:3.8b",
        display_name="Phi-3.5 Mini 3.8B",
        family="phi",
        parameter_size_b=3.8,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_SMALL,
        is_default=False,
        licence="MIT",
        multilingual=True,
        context_window=128_000,
        notes=(
            "MIT-licensed alternative to Llama 3.2; preferred when "
            "downstream redistribution rules matter."
        ),
    ),
    # --- mid (8–14B) --------------------------------------------------- #
    RecommendedModel(
        tag="phi4:14b",
        display_name="Phi-4 14B",
        family="phi",
        parameter_size_b=14.0,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_MID,
        is_default=True,
        licence="MIT",
        multilingual=False,
        context_window=16_000,
        notes=(
            "Microsoft's reasoning-tuned 14B; strong on the code-vs-"
            "no-code judgement we ask for in suggestion mode."
        ),
    ),
    RecommendedModel(
        tag="mistral-nemo:12b",
        display_name="Mistral Nemo 12B",
        family="mistral",
        parameter_size_b=12.0,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_MID,
        is_default=False,
        licence="Apache 2.0",
        multilingual=True,
        context_window=128_000,
        notes=(
            "Apache-2.0, multilingual, 128k context. Pick this when "
            "the corpus has non-English data."
        ),
    ),
    # --- large (32–70B) ------------------------------------------------ #
    RecommendedModel(
        tag="qwen2.5:32b",
        display_name="Qwen 2.5 32B",
        family="qwen",
        parameter_size_b=32.0,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_LARGE,
        is_default=True,
        licence="Apache 2.0",
        multilingual=True,
        context_window=128_000,
        notes=(
            "Best quality-per-VRAM in the 24 GB class. Multilingual, "
            "Apache-licensed, 128k context."
        ),
    ),
    RecommendedModel(
        tag="llama3.3:70b",
        display_name="Llama 3.3 70B",
        family="llama",
        parameter_size_b=70.0,
        kind=KIND_GENERATIVE,
        tier_id=_mt.TIER_LARGE,
        is_default=False,
        licence="Llama 3.3 Community License",
        multilingual=True,
        context_window=128_000,
        notes=(
            "Top-tier reasoning when you have a 48 GB+ GPU or a "
            "second GPU to shard over."
        ),
    ),
)


_EMBEDDING_RECOMMENDATIONS: tuple[RecommendedModel, ...] = (
    RecommendedModel(
        tag="bge-m3",
        display_name="BGE-M3",
        family="bge",
        parameter_size_b=0.567,
        kind=KIND_EMBEDDING,
        tier_id="",
        is_default=True,
        licence="MIT",
        multilingual=True,
        context_window=8_192,
        notes=(
            "Multilingual default. Handles 100+ languages and long "
            "passages; the right baseline for a research corpus."
        ),
    ),
    RecommendedModel(
        tag="nomic-embed-text:v1.5",
        display_name="Nomic Embed Text v1.5",
        family="nomic",
        parameter_size_b=0.137,
        kind=KIND_EMBEDDING,
        tier_id="",
        is_default=False,
        licence="Apache 2.0",
        multilingual=False,
        context_window=8_192,
        notes=(
            "Compact English-leaning alternative; faster to index, "
            "smaller on disk. Pick this for English-only corpora."
        ),
    ),
)


ALL_RECOMMENDATIONS: tuple[RecommendedModel, ...] = (
    _GENERATIVE_RECOMMENDATIONS + _EMBEDDING_RECOMMENDATIONS
)


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def generative_recommendations() -> tuple[RecommendedModel, ...]:
    """Every generative recommendation across every tier, in fixed order."""
    return _GENERATIVE_RECOMMENDATIONS


def embedding_recommendations() -> tuple[RecommendedModel, ...]:
    """Every embedding recommendation, in fixed order."""
    return _EMBEDDING_RECOMMENDATIONS


def recommended_models_for_tier(tier_id: str) -> tuple[RecommendedModel, ...]:
    """All generative recommendations for one tier, default first.

    Raises ``ValueError`` if ``tier_id`` isn't a known tier id; this
    catches typos in caller code rather than silently returning an
    empty list.
    """
    if tier_id not in _mt.KNOWN_TIER_IDS:
        raise ValueError(
            f"Unknown tier {tier_id!r}; known: {_mt.KNOWN_TIER_IDS}"
        )
    matches = tuple(m for m in _GENERATIVE_RECOMMENDATIONS if m.tier_id == tier_id)
    # Stable order: default first, then by declaration order.
    return tuple(sorted(matches, key=lambda m: (not m.is_default,)))


def default_generative_for_tier(tier_id: str) -> RecommendedModel:
    """The pre-selected generative model for a tier.

    Raises ``ValueError`` for an unknown tier or — defensively — if the
    tier ended up with no default (which would be a data-table bug).
    """
    for m in recommended_models_for_tier(tier_id):
        if m.is_default:
            return m
    raise ValueError(
        f"No default generative model registered for tier {tier_id!r}"
    )


def default_embedding(*, multilingual: bool = True) -> RecommendedModel:
    """The pre-selected embedding model.

    With ``multilingual=True`` (the default) returns whichever embedding
    model has ``multilingual=True`` *and* ``is_default=True``. With
    ``multilingual=False`` returns whichever monolingual / English-
    leaning model is registered. Raises ``ValueError`` if no match
    exists — meaning the data table itself is misconfigured.
    """
    if multilingual:
        for m in _EMBEDDING_RECOMMENDATIONS:
            if m.is_default and m.multilingual:
                return m
        raise ValueError("No multilingual default embedding registered")
    for m in _EMBEDDING_RECOMMENDATIONS:
        if not m.multilingual:
            return m
    raise ValueError("No monolingual embedding registered")


def recommended_model_by_tag(tag: str) -> RecommendedModel | None:
    """Find a recommendation by its Ollama tag, or ``None``.

    Empty / whitespace-only tags always return ``None`` so callers can
    pass a user-input string straight through without pre-validation.
    """
    if not tag or not tag.strip():
        return None
    for m in ALL_RECOMMENDATIONS:
        if m.tag == tag:
            return m
    return None


def is_recommended_tag(tag: str) -> bool:
    """``True`` if ``tag`` is a tag this module recommends."""
    return recommended_model_by_tag(tag) is not None


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


def summarise_recommendations(snapshot: _mt.HardwareSnapshot) -> dict[str, Any]:
    """Hardware summary + tier recommendations + embedding picks.

    Returns the JSON-serialisable shape consumed by the
    ``/api/system/model-recommendations`` endpoint:

    .. code-block:: text

        {
          "hardware":   {...},                     # from model_tiers
          "tiers":      [
            {
              ...tier fields...,
              "fit": "comfortable",
              "reason": "...",
              "effective_vram_gb": ...,
              "recommended_models": [
                {"tag": "...", "display_name": "...", ...},
                ...
              ],
            },
            ...
          ],
          "recommended": "<tier_id>",              # from model_tiers
          "embedding_models": [
            {"tag": "...", "display_name": "...", ...},
            ...
          ]
        }

    Note we deliberately *don't* mutate the existing
    ``model_tiers.summarise`` shape; downstream consumers (notably the
    F8.11 server endpoint) keep returning exactly what they did
    before. F8.12 lives behind its own endpoint.
    """
    base = _mt.summarise(snapshot)
    out_tiers: list[dict[str, Any]] = []
    for tier_entry in base["tiers"]:
        tier_id = tier_entry["id"]
        models = recommended_models_for_tier(tier_id)
        out_tiers.append({
            **tier_entry,
            "recommended_models": [m.to_dict() for m in models],
        })
    return {
        "hardware": base["hardware"],
        "tiers": out_tiers,
        "recommended": base["recommended"],
        "embedding_models": [m.to_dict() for m in _EMBEDDING_RECOMMENDATIONS],
    }
