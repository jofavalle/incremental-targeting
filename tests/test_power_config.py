"""The power settings of config.py: what design.md §8.3 fixes, and usable as they stand."""

from __future__ import annotations

import numpy as np

from uplift.config import CONFIG

POWER = CONFIG.power


def test_level_and_power_are_the_declared_ones():
    assert POWER.alpha == 0.05 and POWER.power == 0.80


def test_the_reported_shares_are_the_sample_a_decile_and_a_percentile():
    assert POWER.subgroup_shares == (1.0, 0.1, 0.01)
    assert POWER.decision_share in POWER.subgroup_shares


def test_the_curve_shares_decrease_inside_the_unit_interval():
    shares = np.asarray(POWER.curve_shares)
    assert (np.diff(shares) < 0).all() and 0 < shares[-1] and shares[0] < 1


def test_every_share_below_one_splits_the_sample_into_whole_subgroups():
    # Bonferroni divides the level by 1 / share: it must be a whole number.
    for share in (*POWER.subgroup_shares[1:], *POWER.curve_shares):
        assert abs(1 / share - round(1 / share)) < 1e-9
