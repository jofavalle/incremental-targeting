"""The balance and propensity evaluators, against values computed another way."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from uplift.evaluate.balance import (
    auc,
    bootstrap_auc_interval,
    linear_propensity_sd_bound,
    propensity_is_constant,
    propensity_sd_if_normal,
    smd_noise_se,
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


def test_bootstrap_interval_has_its_nominal_width():
    # A score with no signal: the AUC's standard error is Hanley and McNeil's
    # sqrt((n1 + n0 + 1) / (12 n1 n0)), and a 95 % interval spans about 2 * 1.96 of it.
    rng = np.random.default_rng(3)
    t = (rng.random(20_000) < 0.85).astype(np.int8)
    score = rng.random(t.size)
    n1, n0 = int(t.sum()), int(t.size - t.sum())
    se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    low, high = bootstrap_auc_interval(score, t, n_resamples=200, seed=4)
    assert 0.8 < (high - low) / (2 * 1.96 * se) < 1.2


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
def test_the_normal_translation_recovers_a_normal_spread(sigma):
    rng = np.random.default_rng(7)
    e = np.clip(0.85 + sigma * rng.standard_normal(1_000_000), 0.01, 0.99)
    t = (rng.random(e.size) < e).astype(np.int8)
    implied = propensity_sd_if_normal(auc(e, t), treated_share=t.mean())
    assert implied == pytest.approx(sigma, rel=0.1)


def test_the_normal_translation_is_zero_without_signal():
    assert propensity_sd_if_normal(0.5, 0.85) == 0.0
    assert propensity_sd_if_normal(0.49, 0.85) == 0.0


def test_the_rule_is_on_the_lower_bound_and_exceeding_means_strictly_above():
    assert propensity_is_constant(0.505, 0.51)
    assert propensity_is_constant(0.51, 0.51)
    assert not propensity_is_constant(0.5101, 0.51)


def _features_and_treatment(propensity, n_rows=400_000, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, 12)).astype(np.float32)
    # Not the identity covariance: one feature on another scale, and one
    # correlated with it, so the bound must really use Sigma^-1.
    x[:, 0] *= 10
    x[:, 1] += 0.05 * x[:, 0]
    e = propensity(x)
    return x, (rng.random(n_rows) < e).astype(np.int8), e


def test_the_linear_bound_recovers_a_linear_spread():
    # e linear in x: its best linear predictor is itself, so the bound is the spread.
    x, t, e = _features_and_treatment(lambda x: 0.85 + 0.002 * x[:, 0])
    bound, noise = linear_propensity_sd_bound(x, t, chunk_rows=50_000)
    assert bound == pytest.approx(e.std(), rel=0.1)
    assert bound > 5 * noise


def test_the_linear_bound_stays_below_a_skewed_spread():
    # A tail, the shape the AUC translation gets wrong: the bound must not exceed it.
    x, t, e = _features_and_treatment(lambda x: np.where(x[:, 0] > 20.0, 0.96, 0.845))
    bound, _ = linear_propensity_sd_bound(x, t)
    assert 0 < bound <= e.std() * 1.05


def test_the_linear_bound_sits_at_its_noise_floor_under_a_constant_propensity():
    x, t, _ = _features_and_treatment(lambda x: np.full(len(x), 0.85))
    bound, noise = linear_propensity_sd_bound(x, t)
    assert bound < 2 * noise


def test_the_smd_noise_level_matches_simulated_smds():
    rng = np.random.default_rng(5)
    n_rows = 20_000
    smds = []
    for _ in range(300):
        x = rng.standard_normal((n_rows, 1)).astype(np.float32)
        t = (rng.random(n_rows) < 0.85).astype(np.int8)
        smds.append(standardized_mean_differences(x, t)[0])
    n1 = int(0.85 * n_rows)
    assert np.std(smds) == pytest.approx(smd_noise_se(n1, n_rows - n1), rel=0.15)
