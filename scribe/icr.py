"""Inter-Coder Reliability statistics (F2.5, part 2).

Per PLANNING.md F2.5:

  > Multi-coder mode. Per-coder application; ICR computation
  > (Cohen's kappa first, Krippendorff's alpha later); reconciliation
  > UI.

This module ships the **Cohen's kappa first** half: a small set of
pure functions that compute observed / expected agreement, the
confusion matrix, and Cohen's kappa for two raters' codings of the
same items. A multi-label helper (``per_code_kappa``) extends the
two-rater case to the realistic scenario where each segment can
carry multiple codes.

What this module is **not**: Krippendorff's alpha (deferred — F2.5
spec, "later"), three-or-more-rater statistics, weighted kappa, or
Fleiss' kappa. Those can be added later without changing the public
surface here.

What this module is **not** wired to: per-application coder linkage
or the reconciliation UI. Both wait on F4.1 (the Application entity
itself). The functions below take label lists / dicts directly so
they can be exercised today against synthetic fixtures.

Methodological note
-------------------

Cohen's kappa adjusts observed agreement for the agreement that would
be expected by chance:

    kappa = (p_o - p_e) / (1 - p_e)

where ``p_o`` is the observed proportion of agreement and ``p_e`` is
the chance-expected agreement given each rater's marginal frequencies.
Kappa runs in ``[-1, 1]``; ``1`` is perfect agreement; ``0`` is
chance-level; negative is worse-than-chance (rare in practice). The
Landis & Koch (1977) interpretation labels are conventional:

    < 0.00  poor
    0.01–0.20  slight
    0.21–0.40  fair
    0.41–0.60  moderate
    0.61–0.80  substantial
    0.81–1.00  almost perfect

These are heuristics, not hard cutoffs — methodologically conservative
practice expects ≥ 0.80 for publishable codebooks (Krippendorff
2004). We expose the labels through :func:`interpret_kappa` so reports
can render a human-readable summary.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Iterable


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ICRError(ValueError):
    """Raised when an ICR computation receives invalid input.

    Pure ``ValueError`` subclass so callers can either catch it
    specifically or rely on the broader exception. The HTTP layer (when
    F2.5 grows endpoints) can map it to a 400.
    """


# --------------------------------------------------------------------------- #
# Two-rater single-label kappa (the canonical Cohen's kappa)
# --------------------------------------------------------------------------- #


def _validate_paired_labels(
    rater_a: Iterable[Hashable], rater_b: Iterable[Hashable]
) -> tuple[list[Hashable], list[Hashable]]:
    """Materialise both inputs and check shape.

    Both must be the same length. Empty lists are allowed and return
    ``([], [])`` — the caller's downstream logic decides what to do
    (typically: kappa is conventionally ``1.0`` for the empty case
    since there's nothing to disagree about).
    """
    a = list(rater_a)
    b = list(rater_b)
    if len(a) != len(b):
        raise ICRError(
            f"rater_a and rater_b must be the same length; "
            f"got {len(a)} vs {len(b)}"
        )
    return a, b


def observed_agreement(
    rater_a: Iterable[Hashable], rater_b: Iterable[Hashable]
) -> float:
    """Proportion of items on which the two raters assign the same label.

    Returns 1.0 for two empty inputs (vacuously perfect agreement). The
    computation is symmetric in the inputs.
    """
    a, b = _validate_paired_labels(rater_a, rater_b)
    if not a:
        return 1.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def expected_agreement(
    rater_a: Iterable[Hashable], rater_b: Iterable[Hashable]
) -> float:
    """Chance-level agreement implied by each rater's marginal frequencies.

    For each label ``L`` that appeared in either rater's column, take
    the product ``p_a(L) * p_b(L)`` (the chance both raters say ``L``
    independently) and sum. Returns 1.0 for two empty inputs (matches
    the convention used by ``observed_agreement`` so kappa is
    well-defined as 0/0 → 1.0 in the empty case).
    """
    a, b = _validate_paired_labels(rater_a, rater_b)
    n = len(a)
    if n == 0:
        return 1.0
    counts_a = Counter(a)
    counts_b = Counter(b)
    labels = set(counts_a) | set(counts_b)
    total = 0.0
    for lab in labels:
        p_a = counts_a.get(lab, 0) / n
        p_b = counts_b.get(lab, 0) / n
        total += p_a * p_b
    return total


def cohens_kappa(
    rater_a: Iterable[Hashable], rater_b: Iterable[Hashable]
) -> float:
    """Cohen's kappa for two raters' single-label codings of N items.

    Formal definition: ``kappa = (p_o - p_e) / (1 - p_e)`` where
    ``p_o`` is observed agreement and ``p_e`` is chance-expected
    agreement.

    Edge cases:

    * Both inputs empty → returns ``1.0`` (vacuously perfect; the
      convention used by ``observed_agreement`` / ``expected_agreement``
      above keeps kappa well-defined here).
    * Both raters always agree → returns ``1.0``.
    * Both raters always picked the same single label →
      ``p_o == p_e == 1.0``, so kappa is undefined (0/0). We return
      ``1.0`` because every comparison agreed; the only way to land in
      this branch is full agreement on a single category.
    * Raters always pick different single labels (all "A" vs all
      "B") → ``p_o = 0`` and ``p_e = 0`` (each label is used by only
      one rater, so chance agreement on any label is zero), so the
      formula resolves cleanly to ``0.0``: the raters agree at exactly
      chance level given their marginals. We return ``-1.0`` from the
      ``p_e == 1`` symmetry branch only when the data forces it
      (which requires identical single-label distributions plus zero
      observed agreement — mathematically impossible to construct, but
      the branch is documented as a sentinel).

    These boundary conventions avoid spamming ``nan`` into reports.
    """
    a, b = _validate_paired_labels(rater_a, rater_b)
    if not a:
        return 1.0
    p_o = observed_agreement(a, b)
    p_e = expected_agreement(a, b)
    # When p_e == 1.0 the formula is 0/0; resolve via observed agreement.
    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_o, 1.0) else -1.0
    return (p_o - p_e) / (1.0 - p_e)


def confusion_matrix(
    rater_a: Iterable[Hashable], rater_b: Iterable[Hashable]
) -> dict[tuple[Hashable, Hashable], int]:
    """Build a 2-rater confusion matrix as a dict.

    Keys are ``(label_from_a, label_from_b)`` pairs; values are counts.
    Useful as the underlying matrix for reconciliation views (later
    F2.5 work) and for diagnostics.
    """
    a, b = _validate_paired_labels(rater_a, rater_b)
    cm: Counter[tuple[Hashable, Hashable]] = Counter()
    for x, y in zip(a, b):
        cm[(x, y)] += 1
    return dict(cm)


# --------------------------------------------------------------------------- #
# Multi-label per-code kappa (for the realistic coding scenario)
# --------------------------------------------------------------------------- #


def per_code_kappa(
    coder_a_applications: dict[Hashable, set[Hashable]],
    coder_b_applications: dict[Hashable, set[Hashable]],
    *,
    items: Iterable[Hashable] | None = None,
    codes: Iterable[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Cohen's kappa per code, treating each item as a binary decision.

    Real qualitative coding is multi-label: any item (line, segment,
    paragraph) can carry zero, one, or many codes. The cleanest way to
    extract a Cohen's-kappa-style number is to fix a code, then ask
    "did rater A apply this code to item I?" / "did rater B apply this
    code to item I?" — that's a binary decision, and we have a list of
    items, so we have two equal-length binary lists per code and can
    compute kappa directly.

    Inputs:

    * ``coder_a_applications`` / ``coder_b_applications`` —
      ``{item_id: {code_id, ...}}``. Items absent from a coder's dict
      are treated as "this coder applied no codes here". Items with
      empty sets are equivalent.
    * ``items`` — optional explicit list of items to score over.
      Defaults to the union of the keys of both coders' dicts.
    * ``codes`` — optional explicit list of codes to score over.
      Defaults to the union of every code applied by either coder
      across all items.

    Output: ``{code_id: kappa}``. Codes neither coder ever applied
    receive a kappa of ``1.0`` (vacuously perfect agreement on absence
    — every item gets the same "absent" label from both sides).
    """
    # Materialise lists / sets so we walk them deterministically.
    if items is None:
        merged_items: set[Hashable] = (
            set(coder_a_applications.keys()) | set(coder_b_applications.keys())
        )
        items_list: list[Hashable] = sorted(merged_items, key=_sort_key)
    else:
        items_list = list(items)

    if codes is None:
        merged_codes: set[Hashable] = set()
        for codes_for_item in coder_a_applications.values():
            merged_codes |= set(codes_for_item)
        for codes_for_item in coder_b_applications.values():
            merged_codes |= set(codes_for_item)
        codes_list: list[Hashable] = sorted(merged_codes, key=_sort_key)
    else:
        codes_list = list(codes)

    out: dict[Hashable, float] = {}
    for code_id in codes_list:
        a_labels: list[bool] = []
        b_labels: list[bool] = []
        for item_id in items_list:
            a_set = coder_a_applications.get(item_id, set())
            b_set = coder_b_applications.get(item_id, set())
            a_labels.append(code_id in a_set)
            b_labels.append(code_id in b_set)
        out[code_id] = cohens_kappa(a_labels, b_labels)
    return out


def _sort_key(value: Hashable) -> tuple[int, str]:
    """Stable sort key that's safe for mixed-type hashable inputs.

    ``per_code_kappa`` may receive int, str, or tuple item / code ids
    depending on the caller. ``sorted`` blows up on heterogeneous
    types in Python 3, so we convert everything to a (type-rank,
    repr) pair. Pure aesthetic — the kappa values don't depend on
    the order, only the determinism of iteration does.
    """
    return (0, repr(value))


# --------------------------------------------------------------------------- #
# Interpretation (Landis & Koch 1977 conventional labels)
# --------------------------------------------------------------------------- #


# Conventional cutoffs; see module docstring for citation. Stored as
# ``(upper_bound_inclusive, label)`` so we can walk the list in order
# and stop at the first match.
_LANDIS_KOCH: tuple[tuple[float, str], ...] = (
    (0.00, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.00, "almost perfect"),
)


def interpret_kappa(kappa: float) -> str:
    """Return the Landis & Koch interpretation label for a kappa value.

    ``kappa <= 0`` returns ``"poor"``; ``kappa > 1`` (shouldn't happen
    with valid data, but be safe) clamps to ``"almost perfect"``.
    NaN inputs return ``"undefined"`` so reports can flag them
    explicitly.
    """
    if math.isnan(kappa):
        return "undefined"
    if kappa <= 0.00:
        return "poor"
    for upper, label in _LANDIS_KOCH[1:]:
        if kappa <= upper:
            return label
    return "almost perfect"
