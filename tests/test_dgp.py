"""The data generating process: the truth it returns is the truth it simulates."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from uplift.config import CONFIG
from uplift.simulate.dgp import (
    baseline_logit,
    calibrate_intercept,
    calibration_sample,
    effect,
    generate,
    replication_rng,
)

SIMULATION = CONFIG.simulation
SCENARIOS = {s.name: s for s in SIMULATION.scenarios}
LIFT = SIMULATION.relative_effect

# A smaller calibration sample than config.py's keeps the suite fast; the
# calibration itself is tested against a fresh sample.
X_CAL = calibration_sample(200_000, seed=SIMULATION.calibration_seed)


@lru_cache
def _calibrated(baseline, base_rate):
    return calibrate_intercept(baseline_logit(X_CAL, baseline), base_rate)


def _intercept(scenario):
    return _calibrated(scenario.baseline, scenario.base_rate)


def _draw(name, n, seed=0, **changes):
    scenario = replace(SCENARIOS[name], **changes)
    rng = np.random.default_rng(seed)
    return scenario, generate(scenario, n, rng, _intercept(scenario), LIFT)


def test_dgp_recovers_known_ate():
    # The difference of means of a randomized replication estimates the mean of
    # the true effect: the generator is a randomized experiment with known truth.
    _, data = _draw("reference", 400_000, seed=1)
    treated = data.t == 1
    difference = data.y[treated].mean() - data.y[~treated].mean()
    p1, p0 = data.y[treated].mean(), data.y[~treated].mean()
    se = np.sqrt(p1 * (1 - p1) / treated.sum() + p0 * (1 - p0) / (~treated).sum())
    assert difference == pytest.approx(data.tau_true.mean(), abs=4 * se)


def test_the_true_effect_is_the_clipped_difference():
    for name in SCENARIOS:
        scenario, data = _draw(name, 50_000, seed=2)
        tau = effect(data.x, scenario.effect, LIFT * scenario.base_rate)
        np.testing.assert_allclose(data.mu1, np.clip(data.mu0 + tau, 0, 1))
        np.testing.assert_allclose(data.tau_true, data.mu1 - data.mu0)


def test_clipping_binds_where_the_design_says_it_can():
    # A negative effect at a low base rate pushes mu0 + tau below 0: there the
    # true effect is smaller in size than the input one.
    scenario, data = _draw("sleeping_dogs", 50_000, seed=3, base_rate=0.003)
    tau = effect(data.x, scenario.effect, LIFT * scenario.base_rate)
    clipped = data.mu0 + tau < 0
    assert clipped.any()
    assert (data.tau_true[clipped] > tau[clipped]).all()


@pytest.mark.parametrize(
    "name", ["reference", "rare_outcome", "common_outcome", "complex_baseline"]
)
def test_the_base_rate_is_calibrated_on_a_fresh_sample(name):
    scenario, data = _draw(name, 400_000, seed=4)
    se = np.sqrt(scenario.base_rate * (1 - scenario.base_rate) / data.mu0.size)
    assert data.mu0.mean() == pytest.approx(scenario.base_rate, abs=max(4 * se, 1e-4))


def test_the_effects_have_the_declared_means_and_shapes():
    x = np.random.default_rng(5).standard_normal((1_000_000, 12))
    for kind in ("heterogeneous", "step", "sleeping_dogs"):
        assert effect(x, kind, 0.01).mean() == pytest.approx(0.01, rel=0.01)
    assert (effect(x, "null", 0.01) == 0).all()
    negative = (effect(x, "sleeping_dogs", 0.01) < 0).mean()
    assert negative == pytest.approx(0.2914, abs=0.002)  # about 29 %, as §6.2 says
    assert effect(x, "sleeping_dogs", 0.01).min() > -0.01  # down to minus the mean


def test_the_propensity_is_constant_unless_confounded():
    _, randomized = _draw("reference", 100_000, seed=6)
    assert (randomized.e == 0.85).all()
    assert randomized.t.mean() == pytest.approx(0.85, abs=0.005)
    _, confounded = _draw("confounded", 100_000, seed=6)
    assert np.corrcoef(confounded.e, confounded.x[:, 0])[0, 1] > 0.9
    assert confounded.e.min() < 0.3 and confounded.e.max() > 0.99  # thin overlap


def test_the_complex_baseline_differs_from_the_simple_one():
    x = X_CAL[:10_000]
    assert np.corrcoef(baseline_logit(x, "simple"), baseline_logit(x, "complex"))[0, 1] < 0.6


def test_the_same_generator_gives_the_same_data_and_replications_differ():
    scenario = SCENARIOS["reference"]
    a = generate(scenario, 1000, replication_rng(7, 0, 0), _intercept(scenario), LIFT)
    b = generate(scenario, 1000, replication_rng(7, 0, 0), _intercept(scenario), LIFT)
    c = generate(scenario, 1000, replication_rng(7, 0, 1), _intercept(scenario), LIFT)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)


def test_the_learners_see_float32_covariates_and_flags():
    _, data = _draw("reference", 100)
    assert data.x.dtype == np.float32 and data.x.shape == (100, 12)
    assert data.t.dtype == np.int8 and data.y.dtype == np.int8


def test_unknown_kinds_are_refused():
    x = np.zeros((2, 12))
    with pytest.raises(ValueError, match="Unknown baseline"):
        baseline_logit(x, "cubic")
    with pytest.raises(ValueError, match="Unknown effect"):
        effect(x, "huge", 0.01)
