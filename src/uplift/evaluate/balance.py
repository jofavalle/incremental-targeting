"""Balance between the arms and the propensity check (docs/design.md §6.4).

Pure functions on arrays. The driver that fits the classifier, applies the rule
and writes the record is `uplift.propensity_check`.

## What each one answers

- `standardized_mean_differences`: does any feature differ between treated and
  controls, on average? Descriptive; it decides nothing.
- `auc` and `bootstrap_auc_interval`: can a classifier tell the arms apart from
  the features? Under a constant propensity its out-of-sample AUC is 0.5.
- `implied_propensity_sd`: what spread of the propensity an AUC corresponds to,
  so that the number reads as a size and not only as a test.
- `propensity_is_constant`: the rule of `PropensityConfig`, in one place.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, rankdata


def _arms(t: np.ndarray) -> np.ndarray:
    treated = np.asarray(t).astype(bool)
    if treated.all() or not treated.any():
        raise ValueError("Both arms must be present.")
    return treated


def standardized_mean_differences(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SMD of each column of `x` between treated (`t == 1`) and controls.

    ``(mean_T - mean_C) / sqrt((var_T + var_C) / 2)``, with sample variances: the
    pooled form of Austin (2009). Accumulated in float64, one column at a time,
    so a float32 matrix of millions of rows is never copied whole. A column with
    no variance in either arm gets 0.
    """
    treated = _arms(t)
    smd = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        column = x[:, j]
        in_treated, in_control = column[treated], column[~treated]
        diff = in_treated.mean(dtype=np.float64) - in_control.mean(dtype=np.float64)
        pooled = (
            in_treated.var(dtype=np.float64, ddof=1) + in_control.var(dtype=np.float64, ddof=1)
        ) / 2
        smd[j] = diff / math.sqrt(pooled) if pooled > 0 else 0.0
    return smd


def auc(score: np.ndarray, t: np.ndarray) -> float:
    """Area under the ROC curve of `score` for the labels `t`, by ranks.

    The Mann-Whitney form: the probability that a random treated unit scores
    above a random control, ties counted as one half (average ranks).
    """
    treated = _arms(t)
    n1 = int(treated.sum())
    n0 = treated.size - n1
    ranks = rankdata(score)
    return float((ranks[treated].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc_interval(
    score: np.ndarray, t: np.ndarray, n_resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the AUC over `n_resamples` bootstrap resamples of the rows."""
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        rows = rng.integers(0, score.size, score.size)
        values[b] = auc(score[rows], t[rows])
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def implied_propensity_sd(auc_value: float, treated_share: float) -> float:
    """The standard deviation of the propensity that would produce `auc_value`.

    Inverts ``AUC = Phi(sigma_e / (sqrt(2) e_bar (1 - e_bar)))``, the small-spread
    approximation derived in `PropensityConfig`. An AUC at or below 0.5 means no
    detectable spread, and gives 0.
    """
    if auc_value <= 0.5:
        return 0.0
    spread = math.sqrt(2) * treated_share * (1 - treated_share)
    return float(spread * norm.ppf(auc_value))


def propensity_is_constant(auc_low: float, threshold: float) -> bool:
    """The rule: constant unless the lower bound of the AUC interval exceeds `threshold`."""
    return auc_low <= threshold
