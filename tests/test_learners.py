"""The four learners on simulated data with a known effect, and their mechanics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from uplift.config import CONFIG
from uplift.data.split import assign_folds
from uplift.learners import LEARNERS, DRLearner, SLearner, XLearner
from uplift.simulate.dgp import baseline_logit, calibrate_intercept, calibration_sample, generate

# Fewer rounds and smaller leaves than config.py: the suite must stay fast, and
# what is tested is the mechanics, not the tuning.
SMALL = replace(CONFIG.learners, n_estimators=60, min_child_samples=100)
PROPENSITY = replace(CONFIG.propensity, n_estimators=60, min_child_samples=100)
SCENARIOS = {s.name: s for s in CONFIG.simulation.scenarios}
LIFT = CONFIG.simulation.relative_effect


def _simulate(name, n, seed, **changes):
    scenario = replace(SCENARIOS[name], **changes)
    x_cal = calibration_sample(100_000, seed=0)
    intercept = calibrate_intercept(baseline_logit(x_cal, scenario.baseline), scenario.base_rate)
    return generate(scenario, n, np.random.default_rng(seed), intercept, LIFT)


def _learner(name, constant_propensity=True):
    if name in ("X", "DR"):
        return LEARNERS[name](SMALL, PROPENSITY, constant_propensity, threads=1)
    return LEARNERS[name](SMALL, threads=1)


@pytest.fixture(scope="module")
def strong():
    # A common outcome and a large effect, so that 40 000 rows carry the signal.
    data = _simulate("reference", 40_000, seed=1, base_rate=0.2)
    fitted = {name: _learner(name).fit(data.x, data.t, data.y) for name in LEARNERS}
    return data, {name: learner.predict(data.x) for name, learner in fitted.items()}


def test_every_learner_averages_to_the_ate(strong):
    # §6.4, check 1: mean(tau_hat) against the difference of means on the same rows.
    data, tau_hat = strong
    treated = data.t == 1
    p1, p0 = data.y[treated].mean(), data.y[~treated].mean()
    se = np.sqrt(p1 * (1 - p1) / treated.sum() + p0 * (1 - p0) / (~treated).sum())
    for name, values in tau_hat.items():
        assert values.mean() == pytest.approx(p1 - p0, abs=4 * se), name


def test_the_learners_find_the_heterogeneity(strong):
    # Mechanics, not a ranking: each one finds some of the signal. Which one finds
    # most is the question of the simulation study (§6.2), not of a unit test.
    data, tau_hat = strong
    for name, values in tau_hat.items():
        assert np.corrcoef(values, data.tau_true)[0, 1] > 0.2, name


def test_the_dr_pseudo_outcome_is_cross_fitted():
    # Changing one unit's outcome may move the models of the other folds, never
    # the models of its own fold: the other units of that fold keep their phi.
    data = _simulate("reference", 6_000, seed=2, base_rate=0.2)
    learner = _learner("DR")
    phi = learner.pseudo_outcome(data.x, data.t, data.y)
    folds = assign_folds([data.t], SMALL.n_folds, SMALL.fold_seed)
    unit = int(np.flatnonzero(folds == 0)[0])
    y = data.y.copy()
    y[unit] = 1 - y[unit]
    changed = learner.pseudo_outcome(data.x, data.t, y)
    same_fold = (folds == 0) & (np.arange(y.size) != unit)
    np.testing.assert_array_equal(changed[same_fold], phi[same_fold])
    assert not np.array_equal(changed[folds != 0], phi[folds != 0])


def test_the_dr_pseudo_outcome_averages_to_the_ate():
    data = _simulate("reference", 20_000, seed=3, base_rate=0.2)
    phi = _learner("DR").pseudo_outcome(data.x, data.t, data.y)
    treated = data.t == 1
    difference = data.y[treated].mean() - data.y[~treated].mean()
    assert phi.mean() == pytest.approx(difference, abs=0.01)


class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(x.shape[0], self.value)


def test_the_dr_pseudo_outcome_survives_wrong_outcome_models(monkeypatch):
    # Double robustness with a known propensity: outcome models stuck at 0.5, far
    # from a rate of 0.2, still give a phi whose mean is exactly the difference of
    # means, because the residual terms correct them. A sign error in either
    # residual term would not survive this.
    data = _simulate("reference", 6_000, seed=8, base_rate=0.2)
    monkeypatch.setattr("uplift.learners.dr.base.fit", lambda *args, **kwargs: _Constant(0.5))
    phi = _learner("DR").pseudo_outcome(data.x, data.t, data.y)
    treated = data.t == 1
    difference = data.y[treated].mean() - data.y[~treated].mean()
    assert phi.mean() == pytest.approx(difference, abs=1e-12)


def test_the_x_learner_weighs_the_control_model_by_the_propensity():
    data = _simulate("reference", 10_000, seed=4)
    learner = XLearner(SMALL, PROPENSITY, constant_propensity=True, threads=1)
    learner.fit(data.x, data.t, data.y)
    e = learner.propensity(data.x)
    assert np.unique(e).size == 1 and e[0] == pytest.approx(data.t.mean())
    expected = e * learner.tau0.predict(data.x) + (1 - e) * learner.tau1.predict(data.x)
    np.testing.assert_allclose(learner.predict(data.x), expected)


def test_an_estimated_propensity_follows_x1_and_is_clipped():
    data = _simulate("confounded", 20_000, seed=5)
    learner = XLearner(SMALL, PROPENSITY, constant_propensity=False, threads=1)
    e = learner.fit(data.x, data.t, data.y).propensity(data.x)
    assert np.corrcoef(e, data.x[:, 0])[0, 1] > 0.8
    assert e.min() >= PROPENSITY.clip and e.max() <= 1 - PROPENSITY.clip


def test_the_s_learner_appends_the_treatment_as_a_thirteenth_column():
    data = _simulate("reference", 5_000, seed=6)
    learner = SLearner(SMALL, threads=1).fit(data.x, data.t, data.y)
    assert learner.model.num_feature() == 13


def test_a_fit_is_reproducible():
    data = _simulate("reference", 5_000, seed=7)
    first = DRLearner(SMALL, PROPENSITY, threads=1).fit(data.x, data.t, data.y).predict(data.x)
    again = DRLearner(SMALL, PROPENSITY, threads=1).fit(data.x, data.t, data.y).predict(data.x)
    np.testing.assert_array_equal(first, again)
