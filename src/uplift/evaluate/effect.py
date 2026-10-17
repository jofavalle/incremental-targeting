"""The average effect and its three sensitivity versions (docs/design.md §2.3, §6.5).

Pure functions on arrays. The driver that cross-fits the propensity, runs the
bootstrap and writes the record is `uplift.ate`.

## What each one computes

- `difference_of_means`: the primary estimate, with the normal interval of a
  difference of two proportions (§2.3).
- `stratified_difference`: the difference of means within groups, averaged with
  the group shares. With one group it is the primary estimate; with the strata of
  e_hat, the subclassified version; with the blocks of the source file, the
  block version.
- `hajek`: the weighted version, the normalized inverse-propensity form of §2.3.
- `propensity_strata`: the strata of e_hat, by the rule of §6.5.
- `source_blocks`: the six blocks of the source file, from its run lengths.
- `paired_bootstrap` and `percentile_interval`: one set of resamples for every
  version, so that each difference from the primary estimate has an interval.

The estimators take optional frequency `weights`: the bootstrap passes the number
of times each unit was drawn, instead of copying the rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import norm


def _arms(t: np.ndarray) -> np.ndarray:
    treated = np.asarray(t).astype(bool)
    if treated.all() or not treated.any():
        raise ValueError("Both arms must be present.")
    return treated


def _weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"The level must be in (0, 1), got {level}.")
    return float(level)


def difference_of_means(y: np.ndarray, t: np.ndarray, level: float) -> dict:
    """The primary estimate, with the normal interval of a difference of proportions.

    `y` is a binary outcome, so each arm's variance is p (1 - p). At millions of
    rows the normal approximation is adequate (§2.3).
    """
    treated = _arms(t)
    y = np.asarray(y, dtype=np.float64)
    n1, n0 = int(treated.sum()), int((~treated).sum())
    p1, p0 = float(y[treated].mean()), float(y[~treated].mean())
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = float(norm.ppf(0.5 + _check_level(level) / 2))
    estimate = p1 - p0
    return {
        "estimate": estimate,
        "ci": [estimate - z * se, estimate + z * se],
        "se": se,
        "p1": p1,
        "p0": p0,
        "n1": n1,
        "n0": n0,
    }


def stratified_difference(
    y: np.ndarray,
    t: np.ndarray,
    groups: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """The difference of means within each group, averaged with the group shares.

    `groups` labels each unit with a non-negative integer; None is one group, the
    plain difference of means. A group with no units carries no weight; a group
    with units in one arm only has no difference, and is an error.
    """
    treated = _arms(t)
    arm = treated.astype(np.intp)
    groups = np.zeros(arm.size, dtype=np.intp) if groups is None else np.asarray(groups, np.intp)
    w = _weights(weights, arm.size)
    n_cells = 2 * (int(groups.max()) + 1)
    cell = groups * 2 + arm
    size = np.bincount(cell, weights=w, minlength=n_cells).reshape(-1, 2)
    total = np.bincount(cell, weights=w * y, minlength=n_cells).reshape(-1, 2)
    present = size.sum(axis=1) > 0
    if (size[present] == 0).any():
        raise ValueError("A group lacks one of the arms, so its difference of means is undefined.")
    means = total[present] / size[present]
    shares = size[present].sum(axis=1) / size.sum()
    return float(shares @ (means[:, 1] - means[:, 0]))


def hajek(
    y: np.ndarray, t: np.ndarray, e: np.ndarray, clip: float, weights: np.ndarray | None = None
) -> float:
    """The weighted version: the inverse-propensity mean of each arm, normalized (§2.3).

    Treated units weigh 1 / e and controls 1 / (1 - e), with e clipped to
    [clip, 1 - clip]. Normalizing by the sum of the weights in each arm (the Hájek
    form) makes a constant e cancel: the result is then the difference of means.
    """
    if not 0.0 < clip < 0.5:
        raise ValueError(f"The clip must be in (0, 0.5), got {clip}.")
    treated = _arms(t)
    e = np.clip(np.asarray(e, dtype=np.float64), clip, 1.0 - clip)
    inverse = np.where(treated, 1.0 / e, 1.0 / (1.0 - e)) * _weights(weights, treated.size)
    arm = treated.astype(np.intp)
    size = np.bincount(arm, weights=inverse, minlength=2)
    total = np.bincount(arm, weights=inverse * y, minlength=2)
    means = total / size
    return float(means[1] - means[0])


def propensity_strata(e: np.ndarray, t: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """The strata of e_hat by the rule of §6.5, labelled 0, 1, ... from the lowest.

    The cuts are the `quantiles` of e_hat over all units, both arms pooled. A unit's
    stratum is the number of cuts strictly below its e_hat, so equal values always
    share a stratum. Empty strata are dropped; a stratum that lacks an arm is merged
    with the one below it, the lowest with the one above.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if not ((np.diff(quantiles) > 0).all() and 0 < quantiles[0] and quantiles[-1] < 1):
        raise ValueError(f"The quantiles must increase strictly inside (0, 1), got {quantiles}.")
    treated = _arms(t)
    e = np.asarray(e, dtype=np.float64)
    raw = np.searchsorted(np.quantile(e, quantiles), e, side="left")
    labels = np.unique(raw, return_inverse=True)[1]
    while True:
        n_strata = int(labels.max()) + 1
        with_treated = np.bincount(labels[treated], minlength=n_strata) > 0
        with_controls = np.bincount(labels[~treated], minlength=n_strata) > 0
        lacking = np.flatnonzero(~(with_treated & with_controls))
        if lacking.size == 0:
            return labels.astype(np.int8)
        stratum = int(lacking[0])
        labels[labels == stratum] = stratum - 1 if stratum > 0 else stratum + 1
        labels = np.unique(labels, return_inverse=True)[1]


def source_blocks(row_id: np.ndarray, t: np.ndarray, runs: Sequence[Sequence[int]]) -> np.ndarray:
    """The block of the source file each unit comes from (§6.5), labelled 0, 1, ...

    `runs` is the list of [treatment, length] pairs that `reports/data_summary.json`
    records, in file order, and `row_id` the position of each unit in that file.
    Each treated run is paired with the control run that follows it, and a last
    treated run, with no control after it, joins the last pair: thirteen runs make
    six blocks. Every unit's treatment must be that of its run, or the runs do not
    describe these rows.
    """
    values = np.array([value for value, _ in runs], dtype=np.int64)
    lengths = np.array([length for _, length in runs], dtype=np.int64)
    if values.size < 2 or not np.array_equal(values, (np.arange(values.size) + 1) % 2):
        raise ValueError("The runs must alternate treated and control, starting with treated.")
    ends = np.cumsum(lengths)
    row_id = np.asarray(row_id, dtype=np.int64)
    if row_id.size and (row_id.min() < 0 or row_id.max() >= ends[-1]):
        raise ValueError(f"A row_id falls outside the {int(ends[-1])} rows the runs describe.")
    run = np.searchsorted(ends, row_id, side="right")
    if not np.array_equal(values[run], np.asarray(t, dtype=np.int64)):
        raise ValueError("Some units have a treatment other than their run's: wrong runs.")
    return np.minimum(run // 2, values.size // 2 - 1).astype(np.int8)


def paired_bootstrap(
    statistic: Callable[[np.ndarray], Sequence[float]],
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws of `statistic` over resamples of the n units, shape (n_resamples, k).

    Each resample draws n units with replacement and passes how many times each
    one was drawn as frequency weights. Every column comes from the same
    resamples, so the difference of two columns is a paired bootstrap of the
    difference of the two statistics.
    """
    if n_resamples < 1:
        raise ValueError(f"Need at least one resample, got {n_resamples}.")
    draws: np.ndarray | None = None
    for b in range(n_resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        value = np.asarray(statistic(counts), dtype=np.float64)
        if draws is None:
            draws = np.empty((n_resamples, value.size))
        draws[b] = value
    return draws


def percentile_interval(draws: np.ndarray, level: float) -> np.ndarray:
    """The percentile interval of each column of `draws`, shape (2, k)."""
    level = _check_level(level)
    return np.quantile(draws, [0.5 - level / 2, 0.5 + level / 2], axis=0)
