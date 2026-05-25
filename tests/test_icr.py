"""Tests for scribe.icr (F2.5).

Pure-math tests for inter-coder reliability statistics: observed
agreement, expected agreement, Cohen's kappa, confusion matrix,
multi-label per-code kappa, and the Landis & Koch interpretation
labels.

The reference values for kappa come from worked examples in Cohen
(1960), Landis & Koch (1977), and the SPSS / scikit-learn agreement
literature, hand-checked against the formula
``kappa = (p_o - p_e) / (1 - p_e)``.
"""

from __future__ import annotations

import math

import pytest

from scribe.icr import (
    ICRError,
    cohens_kappa,
    confusion_matrix,
    expected_agreement,
    interpret_kappa,
    observed_agreement,
    per_code_kappa,
)


# --------------------------------------------------------------------------- #
# observed_agreement
# --------------------------------------------------------------------------- #


class TestObservedAgreement:
    def test_perfect_agreement(self) -> None:
        assert observed_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_agreement(self) -> None:
        assert observed_agreement(["a", "a", "a"], ["b", "b", "b"]) == 0.0

    def test_half_agreement(self) -> None:
        assert observed_agreement(
            ["a", "b", "a", "b"], ["a", "a", "a", "a"]
        ) == 0.5

    def test_empty_inputs_is_one(self) -> None:
        # Vacuous case: nothing to disagree about.
        assert observed_agreement([], []) == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ICRError):
            observed_agreement(["a"], ["a", "b"])

    def test_works_with_ints_and_bools(self) -> None:
        assert observed_agreement([1, 2, 3], [1, 2, 3]) == 1.0
        assert observed_agreement(
            [True, False, True], [True, False, False]
        ) == pytest.approx(2 / 3)

    def test_iterators_consumed_correctly(self) -> None:
        # The function should materialise iterators; passing generators
        # twice through internal helpers shouldn't lose data.
        a = iter(["x", "y", "z"])
        b = iter(["x", "y", "z"])
        assert observed_agreement(a, b) == 1.0


# --------------------------------------------------------------------------- #
# expected_agreement
# --------------------------------------------------------------------------- #


class TestExpectedAgreement:
    def test_two_label_balanced(self) -> None:
        # Each rater splits 50/50 over two labels independently:
        # p_e = 0.5*0.5 + 0.5*0.5 = 0.5
        assert expected_agreement(
            ["a", "a", "b", "b"], ["a", "b", "a", "b"]
        ) == pytest.approx(0.5)

    def test_three_label_uniform(self) -> None:
        # Each rater puts 1/3 on each of three labels independently:
        # p_e = 3 * (1/3)^2 = 1/3
        a = ["x", "y", "z", "x", "y", "z"]
        b = ["z", "x", "y", "y", "z", "x"]  # different order, same marginals
        assert expected_agreement(a, b) == pytest.approx(1 / 3)

    def test_skewed_marginals(self) -> None:
        # Worked example: rater A always 'yes'; rater B 1/4 'yes', 3/4 'no'.
        # p_e = (1)(0.25) + (0)(0.75) = 0.25
        a = ["yes"] * 4
        b = ["yes", "no", "no", "no"]
        assert expected_agreement(a, b) == pytest.approx(0.25)

    def test_empty_is_one(self) -> None:
        assert expected_agreement([], []) == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ICRError):
            expected_agreement([], ["a"])


# --------------------------------------------------------------------------- #
# cohens_kappa — the headline statistic
# --------------------------------------------------------------------------- #


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        assert cohens_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_chance_level(self) -> None:
        # Construct a case where p_o == p_e: kappa should be 0.
        # Both raters split 50/50 independently; observed agreement is
        # also 50% so the formula returns 0.
        a = ["a", "a", "b", "b"]
        b = ["a", "b", "a", "b"]
        assert cohens_kappa(a, b) == pytest.approx(0.0)

    def test_worked_textbook_example(self) -> None:
        # Cohen (1960) Table 1 reproduced in many texts:
        #   90/100 agreements; rater A's marginal 50/50, B's 50/50 →
        #   p_o = 0.90, p_e = 0.50, kappa = 0.80.
        a = (["a"] * 50) + (["b"] * 50)
        b = (["a"] * 45) + (["b"] * 5) + (["a"] * 5) + (["b"] * 45)
        # observed: 45 a-a + 45 b-b = 90; chance: 0.5*0.5 + 0.5*0.5 = 0.5
        assert observed_agreement(a, b) == pytest.approx(0.90)
        assert expected_agreement(a, b) == pytest.approx(0.50)
        assert cohens_kappa(a, b) == pytest.approx(0.80)

    def test_better_than_chance_positive(self) -> None:
        a = ["yes", "no", "yes", "no", "yes"]
        b = ["yes", "no", "yes", "yes", "yes"]
        # observed: 4/5 = 0.8; marginals: a={yes:3,no:2}, b={yes:4,no:1}
        # p_e = (3/5)*(4/5) + (2/5)*(1/5) = 12/25 + 2/25 = 14/25 = 0.56
        # kappa = (0.8 - 0.56)/(1 - 0.56) = 0.24/0.44 ≈ 0.5454...
        assert cohens_kappa(a, b) == pytest.approx(
            (0.8 - 0.56) / (1 - 0.56)
        )

    def test_negative_kappa_when_systematic_disagreement(self) -> None:
        # Two raters who systematically swap labels: produces
        # negative kappa.
        a = ["x", "y", "x", "y", "x", "y"]
        b = ["y", "x", "y", "x", "y", "x"]
        assert cohens_kappa(a, b) < 0

    def test_empty_is_one(self) -> None:
        assert cohens_kappa([], []) == 1.0

    def test_all_same_single_label_resolves_to_one(self) -> None:
        # p_e == 1.0 here (only one label exists). p_o == 1.0 too;
        # we resolve 0/0 to 1.0 because every comparison agreed.
        a = ["a"] * 5
        b = ["a"] * 5
        assert cohens_kappa(a, b) == 1.0

    def test_disjoint_single_labels_resolves_to_zero(self) -> None:
        # Each rater consistently picks a different single label. The
        # marginals are concentrated on different labels, so chance
        # agreement on either label is zero (one rater never picks
        # that label). The formula is well-defined here — kappa is
        # exactly 0.0, not -1.0:
        #
        #   marginals: a: {a:5}; b: {b:5}; labels = {a, b}
        #   p_e = (5/5)(0/5) + (0/5)(5/5) = 0
        #   p_o = 0
        #   kappa = (0 - 0) / (1 - 0) = 0.0
        #
        # The "p_e == 1" undefined case requires both raters to pick
        # exactly the same single label, which forces p_o == 1.0 too
        # (covered by test_all_same_single_label_resolves_to_one).
        a = ["a"] * 5
        b = ["b"] * 5
        assert cohens_kappa(a, b) == pytest.approx(0.0)

    def test_pe_equals_one_with_disagreement(self) -> None:
        # Construct a case where p_e is forced to 1.0 but p_o < 1.0.
        # That requires marginals that perfectly mirror each other on
        # one label: e.g. both raters use only label "x" — but then
        # observed must also be 1.0.
        # The undefined p_e==1 with p_o!=1 case is mathematically
        # impossible to construct (it would require both raters to
        # always pick the same label, which forces observed=1.0). We
        # document this and only verify the symmetric case here.
        a = ["x"] * 4
        b = ["x"] * 4
        assert cohens_kappa(a, b) == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ICRError):
            cohens_kappa(["a"], ["a", "b"])

    def test_symmetric_in_inputs(self) -> None:
        a = ["yes", "no", "yes", "no", "yes"]
        b = ["yes", "no", "yes", "yes", "yes"]
        assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))

    def test_kappa_in_valid_range(self) -> None:
        # Random-ish small examples: kappa should always be in [-1, 1].
        cases = [
            (["a", "b", "c"], ["a", "b", "c"]),
            (["a", "a", "b"], ["b", "b", "a"]),
            (["x", "y", "x", "y"], ["y", "x", "y", "x"]),
            (["1", "2", "3", "1", "2", "3"], ["1", "2", "3", "1", "2", "1"]),
        ]
        for a, b in cases:
            k = cohens_kappa(a, b)
            assert -1.0 - 1e-9 <= k <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# confusion_matrix
# --------------------------------------------------------------------------- #


class TestConfusionMatrix:
    def test_simple(self) -> None:
        cm = confusion_matrix(
            ["a", "a", "b", "b"], ["a", "b", "a", "b"]
        )
        assert cm == {("a", "a"): 1, ("a", "b"): 1, ("b", "a"): 1, ("b", "b"): 1}

    def test_all_diagonal_for_perfect_agreement(self) -> None:
        cm = confusion_matrix(["a", "b", "c"], ["a", "b", "c"])
        assert cm == {("a", "a"): 1, ("b", "b"): 1, ("c", "c"): 1}

    def test_empty(self) -> None:
        assert confusion_matrix([], []) == {}

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ICRError):
            confusion_matrix(["a"], ["a", "b"])


# --------------------------------------------------------------------------- #
# per_code_kappa — the multi-label coding case
# --------------------------------------------------------------------------- #


class TestPerCodeKappa:
    def test_perfect_agreement_all_codes(self) -> None:
        a = {"item1": {"c1", "c2"}, "item2": {"c1"}, "item3": set()}
        b = {"item1": {"c1", "c2"}, "item2": {"c1"}, "item3": set()}
        result = per_code_kappa(a, b)
        # Every code should have kappa 1.0.
        for k in ("c1", "c2"):
            assert result[k] == 1.0

    def test_disagreement_one_code(self) -> None:
        # 4 items. Coder A applies c1 to items 1,2. Coder B applies c1
        # to items 1,3. Per item: agree on (1: yes/yes, 2: yes/no,
        # 3: no/yes, 4: no/no). 2 agreements out of 4.
        a = {"i1": {"c1"}, "i2": {"c1"}, "i3": set(), "i4": set()}
        b = {"i1": {"c1"}, "i2": set(), "i3": {"c1"}, "i4": set()}
        items = ["i1", "i2", "i3", "i4"]
        result = per_code_kappa(a, b, items=items)
        # Both raters: 2 yes / 2 no. p_o = 0.5; p_e = 0.5; kappa = 0.
        assert result["c1"] == pytest.approx(0.0)

    def test_missing_item_treated_as_no_codes(self) -> None:
        # Coder B has no entry for "i2"; treated as empty set.
        a = {"i1": {"c1"}, "i2": {"c1"}}
        b = {"i1": {"c1"}}  # i2 missing
        items = ["i1", "i2"]
        result = per_code_kappa(a, b, items=items)
        # A: yes,yes; B: yes,no. p_o = 0.5.
        # marginals: A {yes:2,no:0}; B {yes:1,no:1}.
        # p_e = (1)(0.5) + (0)(0.5) = 0.5; kappa = 0.
        assert result["c1"] == pytest.approx(0.0)

    def test_explicit_items_and_codes_drive_iteration(self) -> None:
        # If we pass both items and codes explicitly, the result
        # contains exactly those codes.
        a = {"i1": {"c1", "c2"}}
        b = {"i1": {"c1"}}
        result = per_code_kappa(
            a, b, items=["i1"], codes=["c1", "c2", "c3"]
        )
        assert set(result.keys()) == {"c1", "c2", "c3"}
        # c3 was never applied → vacuous agreement on absence → kappa 1.0
        assert result["c3"] == 1.0

    def test_default_items_is_union_of_keys(self) -> None:
        a = {"i1": {"c1"}}
        b = {"i2": {"c1"}}
        # Items default to {i1, i2}; per item: A=(c1, no), B=(no, c1).
        # observed: 0/2; expected: 0.5; kappa < 0.
        result = per_code_kappa(a, b)
        assert result["c1"] < 0

    def test_default_codes_is_union_of_applied_codes(self) -> None:
        a = {"i1": {"c1", "c2"}}
        b = {"i1": {"c2", "c3"}}
        result = per_code_kappa(a, b, items=["i1"])
        assert set(result.keys()) == {"c1", "c2", "c3"}

    def test_empty_inputs_returns_empty(self) -> None:
        assert per_code_kappa({}, {}) == {}

    def test_works_with_int_ids(self) -> None:
        # Item ids and code ids are often ints (or hex strings); the
        # helper must not assume a particular type.
        a = {1: {100, 101}, 2: {100}}
        b = {1: {100, 101}, 2: {100}}
        result = per_code_kappa(a, b)
        assert result[100] == 1.0
        assert result[101] == 1.0


# --------------------------------------------------------------------------- #
# interpret_kappa — Landis & Koch labels
# --------------------------------------------------------------------------- #


class TestInterpretKappa:
    @pytest.mark.parametrize("k,label", [
        (-0.5, "poor"),
        (-0.1, "poor"),
        (0.00, "poor"),
        (0.10, "slight"),
        (0.20, "slight"),
        (0.21, "fair"),
        (0.40, "fair"),
        (0.41, "moderate"),
        (0.60, "moderate"),
        (0.61, "substantial"),
        (0.80, "substantial"),
        (0.81, "almost perfect"),
        (1.00, "almost perfect"),
    ])
    def test_landis_koch_buckets(self, k: float, label: str) -> None:
        assert interpret_kappa(k) == label

    def test_above_one_clamped(self) -> None:
        # Shouldn't happen in practice but should not crash.
        assert interpret_kappa(1.5) == "almost perfect"

    def test_nan_is_undefined(self) -> None:
        assert interpret_kappa(float("nan")) == "undefined"

    def test_isnan_does_not_match_negative_zero(self) -> None:
        # -0.0 is a valid number, not nan; should be "poor".
        assert not math.isnan(-0.0)
        assert interpret_kappa(-0.0) == "poor"
