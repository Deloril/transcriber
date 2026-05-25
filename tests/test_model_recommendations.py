"""Tests for scribe.model_recommendations (F8.12).

The data table itself is the most important thing here: the F8.12 spec
names six generative models and two embedding models; any one of them
silently disappearing is a regression. The lookup helpers and the
``summarise_recommendations`` wire format get full coverage too — the
JS UI will consume that shape verbatim.
"""

from __future__ import annotations

import pytest

from scribe.model_recommendations import (
    ALL_RECOMMENDATIONS,
    KIND_EMBEDDING,
    KIND_GENERATIVE,
    KNOWN_KINDS,
    RecommendedModel,
    default_embedding,
    default_generative_for_tier,
    embedding_recommendations,
    generative_recommendations,
    is_recommended_tag,
    recommended_model_by_tag,
    recommended_models_for_tier,
    summarise_recommendations,
)
from scribe.model_tiers import (
    HardwareSnapshot,
    KNOWN_TIER_IDS,
    TIER_LARGE,
    TIER_MID,
    TIER_SMALL,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _snap(*, backend: str = "cuda", vram: float = 24.0, ram: float = 64.0,
          name: str = "X", cpus: int = 8) -> HardwareSnapshot:
    return HardwareSnapshot(
        gpu_backend=backend,
        gpu_name=name,
        vram_gb=vram,
        system_ram_gb=ram,
        cpu_count=cpus,
    )


# --------------------------------------------------------------------------- #
# Data table — the spec names these models verbatim
# --------------------------------------------------------------------------- #


class TestSpecModels:
    """The F8.12 spec lists exactly these models. Any removal is a
    regression; renames must update both the spec and these tests."""

    def test_small_tier_includes_llama_3_2_3b(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_SMALL)}
        assert "llama3.2:3b" in tags

    def test_small_tier_includes_phi_3_5(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_SMALL)}
        assert "phi3.5:3.8b" in tags

    def test_mid_tier_includes_phi_4_14b(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_MID)}
        assert "phi4:14b" in tags

    def test_mid_tier_includes_mistral_nemo(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_MID)}
        assert "mistral-nemo:12b" in tags

    def test_large_tier_includes_qwen_2_5_32b(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_LARGE)}
        assert "qwen2.5:32b" in tags

    def test_large_tier_includes_llama_3_3_70b(self) -> None:
        tags = {m.tag for m in recommended_models_for_tier(TIER_LARGE)}
        assert "llama3.3:70b" in tags

    def test_embedding_includes_bge_m3(self) -> None:
        tags = {m.tag for m in embedding_recommendations()}
        assert "bge-m3" in tags

    def test_embedding_includes_nomic_v1_5(self) -> None:
        tags = {m.tag for m in embedding_recommendations()}
        assert "nomic-embed-text:v1.5" in tags


class TestDefaults:
    """Per-tier defaults are what the UI pre-selects."""

    def test_small_default_is_llama_3_2(self) -> None:
        d = default_generative_for_tier(TIER_SMALL)
        assert d.tag == "llama3.2:3b"
        assert d.is_default is True

    def test_mid_default_is_phi_4(self) -> None:
        d = default_generative_for_tier(TIER_MID)
        assert d.tag == "phi4:14b"

    def test_large_default_is_qwen_2_5_32b(self) -> None:
        d = default_generative_for_tier(TIER_LARGE)
        assert d.tag == "qwen2.5:32b"

    def test_default_embedding_is_bge_m3(self) -> None:
        d = default_embedding()
        assert d.tag == "bge-m3"
        assert d.multilingual is True

    def test_monolingual_default_is_nomic(self) -> None:
        d = default_embedding(multilingual=False)
        assert d.tag == "nomic-embed-text:v1.5"
        assert d.multilingual is False

    def test_each_tier_has_exactly_one_default(self) -> None:
        for tier in KNOWN_TIER_IDS:
            defaults = [
                m for m in recommended_models_for_tier(tier) if m.is_default
            ]
            assert len(defaults) == 1, f"{tier!r} should have one default"

    def test_exactly_one_embedding_default(self) -> None:
        defaults = [m for m in embedding_recommendations() if m.is_default]
        assert len(defaults) == 1


# --------------------------------------------------------------------------- #
# Field shape
# --------------------------------------------------------------------------- #


class TestRecommendedModelDataclass:
    def test_all_fields_present(self) -> None:
        m = recommended_model_by_tag("llama3.2:3b")
        assert m is not None
        d = m.to_dict()
        assert set(d.keys()) == {
            "tag",
            "display_name",
            "family",
            "parameter_size_b",
            "kind",
            "tier_id",
            "is_default",
            "licence",
            "multilingual",
            "context_window",
            "notes",
        }

    def test_kinds_are_known(self) -> None:
        for m in ALL_RECOMMENDATIONS:
            assert m.kind in KNOWN_KINDS

    def test_known_kinds_constant(self) -> None:
        assert set(KNOWN_KINDS) == {KIND_GENERATIVE, KIND_EMBEDDING}

    def test_generative_models_carry_tier_id(self) -> None:
        for m in generative_recommendations():
            assert m.tier_id in KNOWN_TIER_IDS, m.tag

    def test_embedding_models_have_blank_tier_id(self) -> None:
        for m in embedding_recommendations():
            # Embedding models aren't tier-bound; carry empty tier_id.
            assert m.tier_id == "", m.tag

    def test_parameter_size_in_tier_range(self) -> None:
        # Each generative model's parameter_size_b should fall within
        # its tier's [min, max] window so the recommendations stay
        # consistent with the picker's tier shapes.
        from scribe.model_tiers import tier_by_id

        for m in generative_recommendations():
            tier = tier_by_id(m.tier_id)
            assert tier.parameter_min_b <= m.parameter_size_b <= tier.parameter_max_b, (
                f"{m.tag} ({m.parameter_size_b}B) outside "
                f"{m.tier_id} [{tier.parameter_min_b}, {tier.parameter_max_b}]"
            )

    def test_no_duplicate_tags(self) -> None:
        tags = [m.tag for m in ALL_RECOMMENDATIONS]
        assert len(tags) == len(set(tags))

    def test_every_tag_nonempty(self) -> None:
        for m in ALL_RECOMMENDATIONS:
            assert m.tag and m.tag.strip()
            assert m.display_name and m.display_name.strip()
            assert m.family and m.family.strip()

    def test_at_least_one_multilingual_per_tier(self) -> None:
        # F8.12 multilingual story: every tier has at least one
        # multilingual option for non-English corpora.
        for tier in KNOWN_TIER_IDS:
            ml = [m for m in recommended_models_for_tier(tier) if m.multilingual]
            assert ml, f"tier {tier!r} has no multilingual option"


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


class TestRecommendedModelsForTier:
    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError):
            recommended_models_for_tier("xl")

    def test_default_first(self) -> None:
        for tier in KNOWN_TIER_IDS:
            models = recommended_models_for_tier(tier)
            assert models, f"tier {tier!r} returned no models"
            assert models[0].is_default, (
                f"first entry for {tier!r} should be the default"
            )

    def test_returns_only_generative(self) -> None:
        for tier in KNOWN_TIER_IDS:
            for m in recommended_models_for_tier(tier):
                assert m.kind == KIND_GENERATIVE

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            recommended_models_for_tier("")


class TestRecommendedModelByTag:
    def test_known_tag(self) -> None:
        m = recommended_model_by_tag("phi4:14b")
        assert m is not None
        assert m.tier_id == TIER_MID

    def test_unknown_tag(self) -> None:
        assert recommended_model_by_tag("does-not-exist") is None

    def test_empty_returns_none(self) -> None:
        assert recommended_model_by_tag("") is None
        assert recommended_model_by_tag("   ") is None

    def test_is_recommended_tag_true(self) -> None:
        assert is_recommended_tag("bge-m3") is True

    def test_is_recommended_tag_false(self) -> None:
        assert is_recommended_tag("not-a-real-model:42b") is False


class TestDefaultEmbeddingErrors:
    def test_default_embedding_returns_recommended_model(self) -> None:
        d = default_embedding()
        assert isinstance(d, RecommendedModel)
        assert d.kind == KIND_EMBEDDING


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


class TestSummariseRecommendations:
    def test_top_level_shape(self) -> None:
        out = summarise_recommendations(_snap())
        assert set(out.keys()) == {
            "hardware",
            "tiers",
            "recommended",
            "embedding_models",
        }

    def test_tiers_carry_recommended_models(self) -> None:
        out = summarise_recommendations(_snap())
        assert len(out["tiers"]) == 3
        for entry in out["tiers"]:
            assert "recommended_models" in entry
            assert isinstance(entry["recommended_models"], list)
            assert len(entry["recommended_models"]) >= 1
            # Each tier still carries fit/reason from F8.11.
            assert "fit" in entry
            assert "reason" in entry

    def test_per_tier_default_flagged(self) -> None:
        out = summarise_recommendations(_snap())
        for entry in out["tiers"]:
            defaults = [
                m for m in entry["recommended_models"] if m["is_default"]
            ]
            assert len(defaults) == 1, entry["id"]

    def test_embedding_models_top_level(self) -> None:
        out = summarise_recommendations(_snap())
        assert isinstance(out["embedding_models"], list)
        assert len(out["embedding_models"]) >= 2
        tags = {m["tag"] for m in out["embedding_models"]}
        assert "bge-m3" in tags
        assert "nomic-embed-text:v1.5" in tags

    def test_embedding_models_jsonable(self) -> None:
        # No tuples / dataclass instances leaking into the wire format.
        import json

        out = summarise_recommendations(_snap())
        encoded = json.dumps(out)
        assert "bge-m3" in encoded

    def test_recommendation_matches_hardware_picker(self) -> None:
        out_24gb = summarise_recommendations(_snap(vram=24.0, ram=64.0))
        out_8gb = summarise_recommendations(
            _snap(backend="cuda", vram=8.0, ram=16.0)
        )
        out_cpu = summarise_recommendations(
            _snap(backend="cpu", vram=0.0, ram=4.0)
        )
        assert out_24gb["recommended"] == TIER_LARGE
        assert out_8gb["recommended"] == TIER_SMALL
        assert out_cpu["recommended"] == TIER_SMALL

    def test_recommended_tier_has_recommended_models(self) -> None:
        out = summarise_recommendations(_snap())
        recommended_tier = next(
            t for t in out["tiers"] if t["id"] == out["recommended"]
        )
        assert len(recommended_tier["recommended_models"]) >= 1
