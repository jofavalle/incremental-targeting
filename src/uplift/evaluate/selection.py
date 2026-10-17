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

from uplift.evaluate.effect import difference_of_means, propensity_strata


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


def criterion(tau_hat: np.ndarray, z: np.ndarray, weights: np.ndarray | None = None) -> float:
    """mean((tau_hat - Z)^2), to be compared between learners on the same rows.

    `weights` are frequency weights, as the paired bootstrap passes them.
    """
    squared = (np.asarray(tau_hat) - np.asarray(z)) ** 2
    if weights is None:
        return float(squared.mean())
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights @ squared / weights.sum())


def calibration_by_decile(
    tau_hat: np.ndarray, y: np.ndarray, t: np.ndarray, level: float
) -> list[dict]:
    """Observed uplift against mean prediction, within each decile of tau_hat (§6.4).

    The deciles follow the rule of the strata of e_hat (§6.5): ties are never
    split, and a decile without one of the arms is merged with its neighbour. The
    observed uplift is a difference of means, valid inside any subset defined by
    X alone (§2.3), with its normal interval.
    """
    deciles = propensity_strata(tau_hat, t, [i / 10 for i in range(1, 10)])
    rows = []
    for d in range(int(deciles.max()) + 1):
        inside = deciles == d
        observed = difference_of_means(y[inside], t[inside], level)
        rows.append(
            {
                "rows": int(inside.sum()),
                "mean_prediction": float(tau_hat[inside].mean()),
                "observed": observed["estimate"],
                "ci": observed["ci"],
            }
        )
    return rows
