"""How a learner is scored: against the truth in simulation, against Z on real data.

- `pehe`: the precision in estimation of heterogeneous effects (§6.1), computable
  only where tau is known, that is, in simulation.
- `transformed_outcome`: Z = Y (T - e) / (e (1 - e)), whose expectation given X
  is tau(X) (§6.3).
- `criterion`: the mean squared error of tau_hat against Z. Out of sample it
  ranks learners as the PEHE does, because the variance of Z given X does not
  depend on tau_hat; in sample it rewards overfitting, so it is computed on rows
  the learner never saw.
"""

from __future__ import annotations

import numpy as np


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """sqrt(mean((tau_hat - tau)^2))."""
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


def transformed_outcome(y: np.ndarray, t: np.ndarray, e) -> np.ndarray:
    """Z for each unit; `e` is the propensity, a constant or one value per unit."""
    e = np.asarray(e, dtype=np.float64)
    if (e <= 0).any() or (e >= 1).any():
        raise ValueError("The propensity must lie strictly between 0 and 1.")
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(y, dtype=np.float64) * (t - e) / (e * (1 - e))


def criterion(tau_hat: np.ndarray, z: np.ndarray) -> float:
    """mean((tau_hat - Z)^2), to be compared between learners on the same rows."""
    return float(np.mean((np.asarray(tau_hat) - np.asarray(z)) ** 2))
