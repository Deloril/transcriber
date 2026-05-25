"""Tests for scribe.model_tiers (F8.11).

Hardware probing is wrapped behind ``HardwareSnapshot`` so every test
constructs the exact VRAM / RAM combination it cares about. The only
test that exercises the *real* probe is ``TestDetectHardware``, which
just asserts shape, not values.
"""

from __future__ import annotations

import pytest

from scribe.model_tiers import (
    FIT_COMFORTABLE,
    FIT_INFEASIBLE,
    FIT_MARGINAL,
    KNOWN_FITS,
    KNOWN_TIER_IDS,
    MODEL_TIERS,
    MPS_USABLE_RAM_FRACTION,
    HardwareSnapshot,
    ModelTier,
    TIER_LARGE,
    TIER_MID,
    TIER_SMALL,
    TierFit,
    detect_hardware,
    evaluate_all_tiers,
    evaluate_tier,
    recommend_tier,
    summarise,
    system_ram_gb,
    tier_by_id,
)


# --------------------------------------------------------------------------- #
# Tier shape
# --------------------------------------------------------------------------- #


class TestTierDefinitions:
    def test_three_tiers_exist_in_size_order(self) -> None:
        assert len(MODEL_TIERS) == 3
        assert [t.id for t in MODEL_TIERS] == [TIER_SMALL, TIER_MID, TIER_LARGE]

    def test_known_tier_ids_constant_matches(self) -> None:
        assert KNOWN_TIER_IDS == tuple(t.id for t in MODEL_TIERS)

    def test_tier_parameter_ranges_are_sensible(self) -> None:
        # Per the F8.11 spec: small ~3B, mid 8–14B, large 32–70B.
        small = tier_by_id(TIER_SMALL)
        mid = tier_by_id(TIER_MID)
        large = tier_by_id(TIER_LARGE)
        assert small.parameter_max_b < mid.parameter_min_b + 3  # overlap allowed
        assert mid.parameter_max_b <= large.parameter_min_b + 5
        assert small.parameter_min_b >= 1
        assert large.parameter_max_b <= 100

    def test_recommended_vram_strictly_increases(self) -> None:
        prev = 0.0
        for t in MODEL_TIERS:
            assert t.recommended_vram_gb > prev
            prev = t.recommended_vram_gb

    def test_minimum_vram_below_recommended(self) -> None:
        for t in MODEL_TIERS:
            assert t.minimum_vram_gb < t.recommended_vram_gb

    def test_to_dict_round_trip(self) -> None:
        d = MODEL_TIERS[0].to_dict()
        assert d["id"] == TIER_SMALL
        assert "recommended_vram_gb" in d


class TestTierLookup:
    def test_tier_by_id_known(self) -> None:
        assert tier_by_id(TIER_LARGE).id == TIER_LARGE

    def test_tier_by_id_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            tier_by_id("xl")

    def test_known_fits_constant(self) -> None:
        assert set(KNOWN_FITS) == {FIT_COMFORTABLE, FIT_MARGINAL, FIT_INFEASIBLE}


# --------------------------------------------------------------------------- #
# Hardware detection
# --------------------------------------------------------------------------- #


class TestSystemRamGb:
    def test_returns_float(self) -> None:
        ram = system_ram_gb()
        assert isinstance(ram, float)
        # On a real CI machine this is positive; on a sandbox where
        # sysconf fails it's 0.0. Both are valid.
        assert ram >= 0.0


class TestDetectHardware:
    def test_returns_snapshot_with_known_backend(self) -> None:
        snap = detect_hardware()
        assert snap.gpu_backend in ("cuda", "rocm", "mps", "cpu")
        assert snap.system_ram_gb >= 0.0
        assert snap.cpu_count >= 0
        # vram is non-negative; will be > 0 only on a real GPU
        assert snap.vram_gb >= 0.0

    def test_to_dict_exposes_canonical_fields(self) -> None:
        snap = detect_hardware()
        d = snap.to_dict()
        assert set(d.keys()) == {
            "gpu_backend",
            "gpu_name",
            "vram_gb",
            "system_ram_gb",
            "cpu_count",
        }


# --------------------------------------------------------------------------- #
# Tier fit verdict — the core decision table
# --------------------------------------------------------------------------- #


def _snap(*, backend: str = "cpu", vram: float = 0.0, ram: float = 0.0,
          name: str = "", cpus: int = 4) -> HardwareSnapshot:
    return HardwareSnapshot(
        gpu_backend=backend,
        gpu_name=name,
        vram_gb=vram,
        system_ram_gb=ram,
        cpu_count=cpus,
    )


class TestEvaluateTierGpu:
    def test_24gb_gpu_fits_all_tiers_comfortably(self) -> None:
        snap = _snap(backend="cuda", vram=24.0, ram=64.0)
        for tier in MODEL_TIERS:
            fit = evaluate_tier(tier, snap)
            assert fit.fit == FIT_COMFORTABLE
            assert fit.tier_id == tier.id

    def test_8gb_gpu_comfortable_for_small_only(self) -> None:
        snap = _snap(backend="cuda", vram=8.0, ram=16.0)
        fits = {f.tier_id: f.fit for f in evaluate_all_tiers(snap)}
        assert fits[TIER_SMALL] == FIT_COMFORTABLE
        # 8 GB is below mid's minimum (10 GB) → mid falls through to
        # the CPU/RAM check; 16 GB RAM < 32 GB recommended_ram_gb_cpu
        # so mid is infeasible at this snapshot.
        assert fits[TIER_MID] == FIT_INFEASIBLE
        assert fits[TIER_LARGE] == FIT_INFEASIBLE

    def test_marginal_gpu_below_recommended_above_minimum(self) -> None:
        # Small tier: recommended 8, minimum 4 — 5 GB GPU = marginal.
        snap = _snap(backend="cuda", vram=5.0, ram=8.0)
        fit = evaluate_tier(tier_by_id(TIER_SMALL), snap)
        assert fit.fit == FIT_MARGINAL
        assert "minimum" in fit.reason

    def test_rocm_gpu_uses_vram_path(self) -> None:
        # ROCm should be treated identically to CUDA for tier fit.
        snap = _snap(backend="rocm", vram=24.0, ram=32.0)
        for tier in MODEL_TIERS:
            assert evaluate_tier(tier, snap).fit == FIT_COMFORTABLE


class TestEvaluateTierMps:
    def test_apple_unified_memory_uses_ram_with_headroom(self) -> None:
        # 32 GB MPS box → effective VRAM = 32 * 0.75 = 24 GB.
        snap = _snap(backend="mps", vram=0.0, ram=32.0)
        fit = evaluate_tier(tier_by_id(TIER_LARGE), snap)
        # 24 GB effective ≥ 24 GB recommended → comfortable.
        assert fit.fit == FIT_COMFORTABLE
        assert abs(fit.effective_vram_gb - 32.0 * MPS_USABLE_RAM_FRACTION) < 1e-9

    def test_8gb_mac_marginal_for_small(self) -> None:
        # 8 GB → 6 GB effective; small recommends 8, minimum 4 → marginal.
        snap = _snap(backend="mps", vram=0.0, ram=8.0)
        fit = evaluate_tier(tier_by_id(TIER_SMALL), snap)
        assert fit.fit == FIT_MARGINAL


class TestEvaluateTierCpu:
    def test_cpu_only_with_8gb_ram_marginal_for_small(self) -> None:
        snap = _snap(backend="cpu", vram=0.0, ram=8.0)
        fit = evaluate_tier(tier_by_id(TIER_SMALL), snap)
        assert fit.fit == FIT_MARGINAL
        assert "CPU" in fit.reason

    def test_cpu_only_with_4gb_ram_infeasible_for_small(self) -> None:
        snap = _snap(backend="cpu", vram=0.0, ram=4.0)
        fit = evaluate_tier(tier_by_id(TIER_SMALL), snap)
        assert fit.fit == FIT_INFEASIBLE

    def test_cpu_only_64gb_ram_marginal_for_large(self) -> None:
        snap = _snap(backend="cpu", vram=0.0, ram=64.0)
        fit = evaluate_tier(tier_by_id(TIER_LARGE), snap)
        # CPU is never "comfortable" — only marginal.
        assert fit.fit == FIT_MARGINAL


# --------------------------------------------------------------------------- #
# Recommendation rule
# --------------------------------------------------------------------------- #


class TestRecommendTier:
    def test_24gb_gpu_recommends_large(self) -> None:
        assert recommend_tier(_snap(backend="cuda", vram=24.0, ram=64.0)) == TIER_LARGE

    def test_16gb_gpu_recommends_mid(self) -> None:
        assert recommend_tier(_snap(backend="cuda", vram=16.0, ram=32.0)) == TIER_MID

    def test_8gb_gpu_recommends_small(self) -> None:
        assert recommend_tier(_snap(backend="cuda", vram=8.0, ram=16.0)) == TIER_SMALL

    def test_5gb_gpu_marginal_small(self) -> None:
        # No tier is comfortable; small tier is marginal → recommend small.
        assert recommend_tier(_snap(backend="cuda", vram=5.0, ram=16.0)) == TIER_SMALL

    def test_cpu_only_64gb_ram_picks_largest_marginal(self) -> None:
        # Every tier is marginal on a 64 GB CPU box → recommend the
        # largest marginal one (large).
        assert recommend_tier(_snap(backend="cpu", vram=0.0, ram=64.0)) == TIER_LARGE

    def test_cpu_only_4gb_ram_floor_to_small(self) -> None:
        # Nothing fits at all → floor to small (the user always has
        # *some* option to try).
        assert recommend_tier(_snap(backend="cpu", vram=0.0, ram=4.0)) == TIER_SMALL

    def test_mps_24gb_box_recommends_large(self) -> None:
        # 32 GB MPS → 24 GB effective → comfortable for large.
        assert recommend_tier(_snap(backend="mps", vram=0.0, ram=32.0)) == TIER_LARGE


# --------------------------------------------------------------------------- #
# evaluate_all_tiers ordering
# --------------------------------------------------------------------------- #


class TestEvaluateAllTiers:
    def test_returns_one_fit_per_tier_in_size_order(self) -> None:
        snap = _snap(backend="cuda", vram=12.0, ram=32.0)
        fits = evaluate_all_tiers(snap)
        assert [f.tier_id for f in fits] == list(KNOWN_TIER_IDS)

    def test_mid_marginal_at_12gb(self) -> None:
        snap = _snap(backend="cuda", vram=12.0, ram=32.0)
        fits = {f.tier_id: f.fit for f in evaluate_all_tiers(snap)}
        # 12 GB ≥ 16 recommended? no. 12 GB ≥ 10 minimum? yes → marginal.
        assert fits[TIER_MID] == FIT_MARGINAL


# --------------------------------------------------------------------------- #
# summarise — wire format
# --------------------------------------------------------------------------- #


class TestSummarise:
    def test_summary_shape(self) -> None:
        snap = _snap(backend="cuda", vram=24.0, ram=64.0, name="X", cpus=8)
        out = summarise(snap)
        assert set(out.keys()) == {"hardware", "tiers", "recommended"}
        assert out["hardware"]["gpu_backend"] == "cuda"
        assert out["recommended"] == TIER_LARGE
        assert len(out["tiers"]) == 3
        # Each tier carries its fit + reason inline.
        for entry in out["tiers"]:
            assert "fit" in entry
            assert "reason" in entry
            assert "effective_vram_gb" in entry
            assert entry["fit"] in KNOWN_FITS

    def test_recommended_id_appears_in_tiers_list(self) -> None:
        snap = _snap(backend="cuda", vram=8.0, ram=16.0)
        out = summarise(snap)
        ids = {t["id"] for t in out["tiers"]}
        assert out["recommended"] in ids


# --------------------------------------------------------------------------- #
# TierFit dataclass
# --------------------------------------------------------------------------- #


class TestTierFit:
    def test_fields(self) -> None:
        f = TierFit(
            tier_id=TIER_SMALL,
            fit=FIT_COMFORTABLE,
            reason="x",
            effective_vram_gb=8.0,
        )
        assert f.tier_id == TIER_SMALL
        assert f.fit == FIT_COMFORTABLE
        assert f.effective_vram_gb == 8.0
