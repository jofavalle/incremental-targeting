"""Balance between the arms and the propensity check (docs/design.md §6.4).

Pure functions on arrays. The driver that fits the classifier, applies the rule
and writes the record is `uplift.propensity_check`.

## What each one answers

- `standardized_mean_differences`: does any feature differ between treated and
  controls, on average? Descriptive; it decides nothing.
- `auc` and `bootstrap_auc_interval`: can a classifier tell the arms apart from
  the features? Under a constant propensity its out-of-sample AUC is 0.5.
- `linear_propensity_sd_bound`: a lower bound on the spread of the true
  propensity, from the mean differences alone and whatever its shape. This is
  the size the AUC cannot give.
- `smd_noise_se`: the noise level of an SMD under one ratio for everybody, so
  that a small SMD can be read against it.
- `propensity_sd_if_normal`: the spread a normal propensity would need to
  produce a given AUC. Only a translation: a skewed propensity gives the same
  AUC with a larger spread.
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


def propensity_sd_if_normal(auc_value: float, treated_share: float) -> float:
    """The standard deviation a NORMAL propensity would need to produce `auc_value`.

    Inverts ``AUC = Phi(sigma_e / (sqrt(2) e_bar (1 - e_bar)))``, the small-spread
    approximation derived in `PropensityConfig`, which holds for a normal
    propensity only. A heavier tail gives the same AUC with a larger spread, so
    this is neither an estimate nor a bound of the real spread; for a bound, see
    `linear_propensity_sd_bound`. An AUC at or below 0.5 gives 0.
    """
    if auc_value <= 0.5:
        return 0.0
    spread = math.sqrt(2) * treated_share * (1 - treated_share)
    return float(spread * norm.ppf(auc_value))


def smd_noise_se(n_treated: int, n_control: int) -> float:
    """Standard error of an SMD under one treatment ratio for everybody, about sqrt(1/n1 + 1/n0)."""
    return math.sqrt(1 / n_treated + 1 / n_control)


def linear_propensity_sd_bound(
    x: np.ndarray, t: np.ndarray, chunk_rows: int = 1 << 20
) -> tuple[float, float]:
    """A lower bound on the standard deviation of the true propensity e(X), and its noise floor.

    ``E[X T] = E[X e(X)]``, so ``Cov(X, e(X)) = Cov(X, T) = e_bar (1 - e_bar) delta``,
    with ``delta`` the difference of the feature means between the arms: the mean
    differences ARE covariances with the propensity. The variance of e(X) is at
    least that of its best linear predictor from X,
    ``(e_bar (1 - e_bar))**2 delta' Sigma^-1 delta``, whatever the shape of e.

    Under a constant propensity, sampling noise alone gives a bound of about
    ``e_bar (1 - e_bar) sqrt(p (1/n1 + 1/n0))``, returned second, so that the bound
    can be read against it. Two passes in float64 over blocks of rows: the means,
    then the centred cross-products, so a float32 matrix is never copied whole
    and the covariance does not lose digits to large means.
    """
    treated = _arms(t)
    n_rows, n_features = x.shape
    n1 = int(treated.sum())
    n0 = n_rows - n1
    blocks = range(0, n_rows, chunk_rows)

    total = np.zeros(n_features)
    total_treated = np.zeros(n_features)
    for start in blocks:
        block = x[start : start + chunk_rows].astype(np.float64)
        total += block.sum(axis=0)
        total_treated += block[treated[start : start + chunk_rows]].sum(axis=0)
    mean = total / n_rows
    delta = total_treated / n1 - (total - total_treated) / n0

    cross = np.zeros((n_features, n_features))
    for start in blocks:
        centred = x[start : start + chunk_rows].astype(np.float64) - mean
        cross += centred.T @ centred
    sigma = cross / n_rows

    scale = (n1 / n_rows) * (n0 / n_rows)
    bound = scale * math.sqrt(float(delta @ np.linalg.solve(sigma, delta)))
    noise = scale * math.sqrt(n_features * (1 / n1 + 1 / n0))
    return bound, noise


def propensity_is_constant(auc_low: float, threshold: float) -> bool:
    """The rule: constant unless the lower bound of the AUC interval exceeds `threshold`."""
    return auc_low <= threshold
