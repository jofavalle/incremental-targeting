"""The balance and propensity evaluators, against values computed another way."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from uplift.evaluate.balance import (
    auc,
    bootstrap_auc_interval,
    implied_propensity_sd,
    propensity_is_constant,
    standardized_mean_differences,
)


def _auc_by_pairs(score, t):
    # The definition itself: every treated-control pair, ties worth one half.
    treated = [s for s, y in zip(score, t) if y == 1]
    control = [s for s, y in zip(score, t) if y == 0]
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(treated, control)
    )
    return wins / (len(treated) * len(control))


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_the_pair_count_with_ties(seed):
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 6, 60).astype(float)  # few values: many ties
    t = (rng.random(60) < 0.7).astype(np.int8)
    assert auc(score, t) == pytest.approx(_auc_by_pairs(score, t))


def test_auc_extremes():
    t = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), t) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), t) == 0.0
    assert auc(np.full(4, 0.85), t) == 0.5


def test_auc_needs_both_arms():
    with pytest.raises(ValueError, match="Both arms"):
        auc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_bootstrap_interval_brackets_the_auc():
    rng = np.random.default_rng(0)
    t = (rng.random(5_000) < 0.85).astype(np.int8)
    score = rng.random(5_000) + 0.2 * t
    low, high = bootstrap_auc_interval(score, t, n_resamples=100, seed=1)
    assert low < auc(score, t) < high
    assert (low, high) == bootstrap_auc_interval(score, t, n_resamples=100, seed=1)


def test_smd_of_a_known_shift():
    # Treated [0, 2] and controls [-1, 1]: means 1 and 0, sample variances 2 and 2.
    x = np.array([[0.0], [2.0], [-1.0], [1.0]], dtype=np.float32)
    t = np.array([1, 1, 0, 0])
    np.testing.assert_allclose(standardized_mean_differences(x, t), [1 / np.sqrt(2)])


def test_smd_is_zero_for_identical_arms_and_for_a_constant_column():
    x = np.array([[1.0, 5.0], [2.0, 5.0], [1.0, 5.0], [2.0, 5.0]], dtype=np.float32)
    t = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(standardized_mean_differences(x, t), [0.0, 0.0])


@pytest.mark.parametrize("sigma", [0.01, 0.03])
def test_implied_sd_recovers_a_simulated_spread(sigma):
    rng = np.random.default_rng(7)
    e = np.clip(0.85 + sigma * rng.standard_normal(1_000_000), 0.01, 0.99)
    t = (rng.random(e.size) < e).astype(np.int8)
    implied = implied_propensity_sd(auc(e, t), treated_share=t.mean())
    assert implied == pytest.approx(sigma, rel=0.1)


def test_implied_sd_is_zero_without_signal():
    assert implied_propensity_sd(0.5, 0.85) == 0.0
    assert implied_propensity_sd(0.49, 0.85) == 0.0


def test_the_rule_is_on_the_lower_bound_and_exceeding_means_strictly_above():
    assert propensity_is_constant(0.505, 0.51)
    assert propensity_is_constant(0.51, 0.51)
    assert not propensity_is_constant(0.5101, 0.51)
