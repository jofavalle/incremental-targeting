"""The effect settings of config.py: what design.md §6.5 fixes, and usable as they stand."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.config import CONFIG
from uplift.data.split import STRATA, assign_folds

EFFECT = CONFIG.effect


def test_the_strata_cuts_are_the_nine_deciles_and_the_top_percentile():
    cuts = np.asarray(EFFECT.stratum_quantiles)
    np.testing.assert_allclose(cuts[:-1], np.arange(1, 10) / 10)
    assert cuts[-1] == 0.99
    assert (np.diff(cuts) > 0).all() and 0 < cuts[0] and cuts[-1] < 1


def test_the_folds_are_stratified_by_the_treatment_alone():
    assert EFFECT.fold_strata == ("treatment",)
    assert set(EFFECT.fold_strata) <= set(STRATA)


def test_the_fold_settings_are_accepted_by_assign_folds():
    treatment = (np.random.default_rng(0).random(1000) < 0.85).astype(np.int8)
    labels = assign_folds([treatment], EFFECT.n_folds, EFFECT.fold_seed)
    assert set(np.unique(labels)) == set(range(EFFECT.n_folds))


def test_the_numpy_draws_have_distinct_seeds():
    # One seed on the same columns couples two draws (uplift.data.split), so
    # every draw that must be independent of the others has its own.
    seeds = [CONFIG.split.seed, EFFECT.fold_seed, EFFECT.bootstrap_seed]
    assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test_each_tail_of_a_bootstrap_interval_rests_on_enough_resamples():
    # A percentile interval with too few resamples in a tail sits on its extreme
    # order statistics, which move with every seed.
    assert 0 < EFFECT.level < 1
    assert isinstance(EFFECT.n_bootstrap, int)
    per_tail = (1 - EFFECT.level) / 2 * EFFECT.n_bootstrap
    assert per_tail >= 10
    assert per_tail == pytest.approx(25)  # the figure the comment in config.py quotes
