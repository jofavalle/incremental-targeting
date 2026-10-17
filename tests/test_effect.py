"""The effect estimators: exact identities, known answers, and the rules of §6.5."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import norm

from uplift.config import CONFIG, REPORTS_DIR
from uplift.evaluate.effect import (
    difference_of_means,
    hajek,
    paired_bootstrap,
    percentile_interval,
    propensity_strata,
    source_blocks,
    stratified_difference,
)

QUANTILES = CONFIG.effect.stratum_quantiles


def _two_tests(n_rows: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Two pooled tests with treatment ratios 0.96 and 0.55 and the same true effect.

    The second test has the higher base rate and the lower ratio, so pooling them
    biases the raw difference of means; within each test it is unbiased.
    """
    rng = np.random.default_rng(seed)
    test = (rng.random(n_rows) < 0.5).astype(np.intp)
    e = np.where(test == 0, 0.96, 0.55)
    t = (rng.random(n_rows) < e).astype(np.int8)
    base = np.where(test == 0, 0.02, 0.10)
    y = (rng.random(n_rows) < base + 0.01 * t).astype(np.int8)
    return {"test": test, "e": e, "t": t, "y": y}


def test_difference_of_means_by_hand():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    result = difference_of_means(y, t, level=0.95)
    assert result["p1"] == 0.75 and result["p0"] == 0.25
    assert result["estimate"] == pytest.approx(0.5)
    se = np.sqrt(0.75 * 0.25 / 4 + 0.25 * 0.75 / 4)
    assert result["se"] == pytest.approx(se)
    assert result["ci"] == pytest.approx([0.5 - norm.ppf(0.975) * se, 0.5 + norm.ppf(0.975) * se])


def test_one_group_is_the_difference_of_means():
    data = _two_tests(20_000)
    plain = difference_of_means(data["y"], data["t"], level=0.95)["estimate"]
    assert stratified_difference(data["y"], data["t"]) == pytest.approx(plain, abs=1e-15)


def test_hajek_with_a_constant_propensity_is_the_difference_of_means():
    data = _two_tests(20_000)
    plain = stratified_difference(data["y"], data["t"])
    constant = np.full(data["t"].size, 0.85)
    assert hajek(data["y"], data["t"], constant, clip=0.01) == pytest.approx(plain, abs=1e-12)


def test_pooled_tests_bias_the_difference_and_the_versions_recover_the_effect():
    # The world §2.3 worries about: the true effect is 0.01 in both tests, the raw
    # difference is far from it, and weighting by e or stratifying by test is not.
    data = _two_tests(400_000, seed=1)
    y, t = data["y"], data["t"]
    raw = stratified_difference(y, t)
    weighted = hajek(y, t, data["e"], clip=0.01)
    by_test = stratified_difference(y, t, groups=data["test"])
    assert abs(raw - 0.01) > 0.01
    assert weighted == pytest.approx(0.01, abs=0.003)
    assert by_test == pytest.approx(0.01, abs=0.003)


def test_integer_weights_are_repeated_rows():
    # The bootstrap passes counts instead of copying rows: both must agree.
    data = _two_tests(5_000, seed=2)
    counts = np.random.default_rng(3).integers(0, 4, data["t"].size)
    rows = np.repeat(np.arange(counts.size), counts)
    y, t, e, test = data["y"], data["t"], data["e"], data["test"]
    assert stratified_difference(y, t, test, weights=counts) == pytest.approx(
        stratified_difference(y[rows], t[rows], test[rows])
    )
    assert hajek(y, t, e, 0.01, weights=counts) == pytest.approx(
        hajek(y[rows], t[rows], e[rows], 0.01)
    )


def test_groups_weigh_by_their_share_of_units_not_of_treated():
    # Group 0 has 4 units, 3 treated, and a difference of 2/3; group 1 has 6
    # units, 1 treated, and a difference of 0. The ATE weighs them 0.4 and 0.6,
    # 4/15; weighing by treated units, 0.75 and 0.25, would give the ATT, 1/2.
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    t = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert stratified_difference(y, t, groups) == pytest.approx(4 / 15)


def test_a_group_with_one_arm_is_refused():
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 1, 1])
    with pytest.raises(ValueError, match="lacks one of the arms"):
        stratified_difference(y, t, groups=np.array([0, 0, 1, 1]))


def test_the_clip_bounds_the_weights():
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 1, 0, 0])
    tiny = np.array([1e-9, 0.5, 0.5, 0.5])
    assert hajek(y, t, tiny, clip=0.01) == hajek(y, t, np.array([0.01, 0.5, 0.5, 0.5]), 0.01)
    with pytest.raises(ValueError, match="clip"):
        hajek(y, t, tiny, clip=0.5)


def test_strata_follow_the_quantiles_of_a_continuous_propensity():
    rng = np.random.default_rng(4)
    e = rng.uniform(0.80, 0.90, 100_000)
    t = (rng.random(e.size) < e).astype(np.int8)
    labels = propensity_strata(e, t, QUANTILES)
    sizes = np.bincount(labels) / e.size
    assert sizes.size == len(QUANTILES) + 1 == 11
    np.testing.assert_allclose(sizes, [0.1] * 9 + [0.09, 0.01], atol=1e-4)
    order = np.argsort(e)
    assert (np.diff(labels[order]) >= 0).all()  # strata are ordered by e_hat


def test_ties_are_never_split():
    # A boosted model gives a step function: here half the units share one value,
    # which spans five deciles. They must land in one stratum.
    rng = np.random.default_rng(5)
    e = np.concatenate([np.full(50_000, 0.85), rng.uniform(0.86, 0.95, 50_000)])
    t = (rng.random(e.size) < e).astype(np.int8)
    labels = propensity_strata(e, t, QUANTILES)
    assert np.unique(labels[e == 0.85]).size == 1
    assert np.unique(labels).size < 11  # the strata the tie swallowed are gone
    assert np.array_equal(np.unique(labels), np.arange(np.unique(labels).size))


def test_a_unit_on_a_cut_goes_to_the_stratum_below():
    # The median of these values is 2, a value units actually take. "Cuts strictly
    # below" puts them with the 1s; counting cuts at or below would put them with
    # the 3s.
    e = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    t = np.array([1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(propensity_strata(e, t, (0.5,)), [0, 0, 0, 0, 1, 1])


def test_a_stratum_without_controls_is_merged_with_the_one_below():
    e = np.linspace(0.0, 1.0, 1000)
    t = np.ones(1000, dtype=np.int8)
    t[::7] = 0
    t[e > 0.99] = 1  # the top 1 % has no controls
    labels = propensity_strata(e, t, QUANTILES)
    assert labels.max() == 9
    assert (labels[e > 0.99] == 9).all() and (labels[(e > 0.9) & (e <= 0.99)] == 9).all()


def test_an_inner_stratum_without_controls_is_merged_with_the_one_below_too():
    e = np.linspace(0.0, 1.0, 1000)
    t = np.ones(1000, dtype=np.int8)
    t[::7] = 0
    t[(e > 0.5) & (e <= 0.6)] = 1  # the sixth decile has no controls
    labels = propensity_strata(e, t, QUANTILES)

    def stratum(low, high):
        return np.unique(labels[(e > low) & (e < high)]).item()

    assert labels.max() == 9
    assert stratum(0.52, 0.58) == stratum(0.42, 0.48)  # the sixth joins the fifth
    assert stratum(0.62, 0.68) == stratum(0.42, 0.48) + 1  # the seventh stays apart


def test_a_lowest_stratum_without_controls_is_merged_with_the_one_above():
    e = np.linspace(0.0, 1.0, 1000)
    t = np.ones(1000, dtype=np.int8)
    t[::7] = 0
    t[e <= 0.1] = 1
    labels = propensity_strata(e, t, QUANTILES)
    assert labels.max() == 9
    assert (labels[e <= 0.2] == 0).all()


@pytest.mark.parametrize("quantiles", [(0.5, 0.4), (0.0, 0.5), (0.5, 1.0)])
def test_strata_refuse_quantiles_that_are_not_increasing_inside_the_unit_interval(quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        propensity_strata(np.linspace(0, 1, 10), np.array([0, 1] * 5), quantiles)


RUNS = [[1, 5], [0, 2], [1, 3], [0, 1], [1, 4]]  # five runs: two pairs and a last treated


def _rows_of(runs):
    t = np.concatenate([np.full(length, value) for value, length in runs])
    return np.arange(t.size), t


def test_blocks_pair_each_treated_run_with_the_next_control_run():
    row_id, t = _rows_of(RUNS)
    blocks = source_blocks(row_id, t, RUNS)
    expected = [0] * 5 + [0] * 2 + [1] * 3 + [1] * 1 + [1] * 4  # the last run joins pair 1
    np.testing.assert_array_equal(blocks, expected)


def test_blocks_of_the_real_file_are_six():
    runs = json.loads((REPORTS_DIR / "data_summary.json").read_text())["source_order"]
    runs = runs["treatment_run_lengths"]
    starts = np.cumsum([0] + [length for _, length in runs[:-1]])
    ends = starts + np.array([length for _, length in runs]) - 1
    row_id = np.concatenate([starts, ends])  # the first and last row of each run
    t = np.array([value for value, _ in runs] * 2)
    blocks = source_blocks(row_id, t, runs)
    np.testing.assert_array_equal(blocks[:13], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])
    np.testing.assert_array_equal(blocks[13:], blocks[:13])


def test_blocks_refuse_runs_that_do_not_describe_the_rows():
    row_id, t = _rows_of(RUNS)
    t[0] = 0
    with pytest.raises(ValueError, match="wrong runs"):
        source_blocks(row_id, t, RUNS)
    with pytest.raises(ValueError, match="outside"):
        source_blocks(np.array([15]), np.array([1]), RUNS)
    with pytest.raises(ValueError, match="alternate"):
        source_blocks(row_id, t, [[0, 5], [1, 10]])


def test_the_paired_bootstrap_shares_its_resamples():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, 2_000)

    def statistic(counts):
        mean = counts @ x / counts.sum()
        return [mean, mean + 1.0, counts.sum()]

    draws = paired_bootstrap(statistic, x.size, 300, np.random.default_rng(7))
    assert draws.shape == (300, 3)
    assert (draws[:, 2] == x.size).all()  # every resample draws n units
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 1.0)  # paired: the gap has no noise
    low, high = percentile_interval(draws[:, :1], level=0.95)[:, 0]
    se = x.std() / np.sqrt(x.size)
    assert low < x.mean() < high
    assert high - low == pytest.approx(2 * norm.ppf(0.975) * se, rel=0.2)
    again = paired_bootstrap(statistic, x.size, 300, np.random.default_rng(7))
    np.testing.assert_array_equal(draws, again)
