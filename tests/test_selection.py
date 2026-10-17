"""PEHE, the transformed outcome and the criterion: identities and the §6.3 argument."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.evaluate.selection import criterion, pehe, transformed_outcome


def test_pehe_by_hand():
    assert pehe(np.array([0.1, 0.3]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(0.05))
    assert pehe(np.zeros(3), np.zeros(3)) == 0.0


def test_the_transformed_outcome_takes_the_values_section_6_3_quotes():
    z = transformed_outcome(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), 0.85)
    np.testing.assert_allclose(z, [1 / 0.85, -1 / 0.15, 0, 0])


def test_z_is_unbiased_for_the_effect():
    rng = np.random.default_rng(0)
    n, e = 2_000_000, 0.85
    tau = np.where(rng.random(n) < 0.5, 0.02, 0.08)
    t = (rng.random(n) < e).astype(np.int8)
    y = (rng.random(n) < 0.1 + tau * t).astype(np.int8)
    z = transformed_outcome(y, t, e)
    for value in (0.02, 0.08):
        group = tau == value
        se = z[group].std() / np.sqrt(group.sum())
        assert z[group].mean() == pytest.approx(value, abs=4 * se)


def test_the_criterion_ranks_as_the_pehe_out_of_sample():
    # E[(tau_hat - Z)^2] = PEHE^2 + E[Var(Z | X)] for a tau_hat independent of Z:
    # the better estimate has the smaller criterion, by about the PEHE gap.
    rng = np.random.default_rng(1)
    n, e = 2_000_000, 0.85
    tau = rng.uniform(0.0, 0.1, n)
    t = (rng.random(n) < e).astype(np.int8)
    y = (rng.random(n) < 0.1 + tau * t).astype(np.int8)
    z = transformed_outcome(y, t, e)
    good, bad = tau + rng.normal(0, 0.01, n), tau + rng.normal(0, 0.05, n)
    gap = criterion(bad, z) - criterion(good, z)
    assert gap == pytest.approx(pehe(bad, tau) ** 2 - pehe(good, tau) ** 2, rel=0.1)
    assert criterion(good, z) < criterion(bad, z)


def test_a_propensity_on_the_boundary_is_refused():
    with pytest.raises(ValueError, match="strictly between"):
        transformed_outcome(np.array([1]), np.array([1]), 1.0)
