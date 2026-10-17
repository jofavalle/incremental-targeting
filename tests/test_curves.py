"""The curves of §7.1: by hand, ties, the random line and the oracle."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.evaluate.curves import RankedScore, area_over_random, break_even

PHI = np.linspace(0.0, 1.0, 101)


def _world(n, seed=0):
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.0, 0.2, n)
    t = (rng.random(n) < 0.85).astype(np.int8)
    y = (rng.random(n) < 0.1 + tau * t).astype(np.int8)
    return tau, t, y


def test_the_curves_by_hand():
    # Scores 4, 3, 2, 1. Top 2: one treated with y = 1, one control with y = 0.
    score = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 1])
    u, q = RankedScore(score, t, y).curves(np.array([0.5, 1.0]))
    assert u[0] == pytest.approx((1 / 1 - 0 / 1) * 2)
    assert q[0] == pytest.approx(1 - 0 * 1 / 1)
    assert u[1] == pytest.approx((1 / 2 - 1 / 2) * 4)


def test_the_curves_are_related_as_section_7_1_says():
    tau, t, y = _world(50_000)
    ranked = RankedScore(tau, t, y)
    u, q = ranked.curves(PHI)
    # U(n) = Q(n) n / N_T(n) at every block boundary, so at the end too.
    assert u[-1] == pytest.approx(q[-1] * t.size / t.sum())


def test_qini_random_is_flat():
    tau, t, y = _world(200_000, seed=1)
    noise = np.random.default_rng(2).random(t.size)
    u, q = RankedScore(noise, t, y).curves(PHI)
    assert abs(area_over_random(u, PHI, t.size)) < 0.002
    assert abs(area_over_random(q, PHI, t.size)) < 0.002


def test_a_constant_score_is_exactly_the_random_line():
    tau, t, y = _world(10_000, seed=3)
    u, _ = RankedScore(np.zeros(t.size), t, y).curves(PHI)
    np.testing.assert_allclose(u, PHI * u[-1], atol=1e-9)
    assert area_over_random(u, PHI, t.size) == pytest.approx(0.0, abs=1e-12)


def test_the_oracle_is_above_the_random_line_and_above_a_noisy_score():
    tau, t, y = _world(200_000, seed=4)
    noisy = tau + np.random.default_rng(5).normal(0, 0.1, t.size)
    oracle = area_over_random(RankedScore(tau, t, y).curves(PHI)[0], PHI, t.size)
    worse = area_over_random(RankedScore(noisy, t, y).curves(PHI)[0], PHI, t.size)
    assert oracle > 0.005 and oracle > worse


def test_weights_are_repeated_units():
    tau, t, y = _world(5_000, seed=6)
    counts = np.random.default_rng(7).integers(0, 3, t.size)
    rows = np.repeat(np.arange(t.size), counts)
    weighted = RankedScore(tau, t, y).curves(PHI, weights=counts)
    repeated = RankedScore(tau[rows], t[rows], y[rows]).curves(PHI)
    np.testing.assert_allclose(weighted[0], repeated[0])
    np.testing.assert_allclose(weighted[1], repeated[1])


def test_break_even_is_where_the_slope_falls_below_cost_over_value():
    # A concave curve, slope 0.02 for the first half and 0.005 after.
    n = 1000.0
    curve = np.where(PHI <= 0.5, 0.02 * PHI * n, 0.01 * n + 0.005 * (PHI - 0.5) * n)
    assert break_even(curve, PHI, n, cost_over_value=0.01) == pytest.approx(0.5)
    assert break_even(curve, PHI, n, cost_over_value=0.001) == 1.0
    assert break_even(curve, PHI, n, cost_over_value=0.05) == 0.0


def test_the_area_needs_the_whole_axis():
    with pytest.raises(ValueError, match="from 0 to 1"):
        area_over_random(np.zeros(3), np.array([0.0, 0.5, 0.9]), 10)
