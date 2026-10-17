"""Minimum detectable effects for the ATE and for subgroup contrasts (docs/design.md §8).

Pure functions. The driver that reads the measured rates and the test sizes,
applies the rule of §8.3 and writes the record is `uplift.power_analysis`.

For a difference of two proportions with n1 treated and n0 controls, the
variance of the full-sample estimate is

    V = p1 (1 - p1) / n1 + p0 (1 - p0) / n0,

and the smallest effect detected at two-sided level alpha with the given power is
(z_{1 - alpha/2} + z_power) sqrt(V). A subgroup with a share s of the sample has
variance V / s and the rest V / (1 - s), so their contrast has V / (s (1 - s)).
The level of a contrast is divided by the 1 / s subgroups of that size
(Bonferroni, §8.3). Every subgroup is assumed to have the rates of the whole.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def full_sample_variance(p1: float, p0: float, n1: int, n0: int) -> float:
    """V, the variance of the difference of means over the whole sample."""
    for name, p in (("p1", p1), ("p0", p0)):
        if not 0.0 < p < 1.0:
            raise ValueError(f"The rate {name} must be in (0, 1), got {p}.")
    if n1 <= 0 or n0 <= 0:
        raise ValueError(f"Both arms need units, got n1={n1} and n0={n0}.")
    return p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0


def subgroups_of_share(share: float) -> int:
    """How many subgroups of this share make the sample: the Bonferroni divisor."""
    if not 0.0 < share <= 1.0:
        raise ValueError(f"The share must be in (0, 1], got {share}.")
    m = round(1 / share)
    if abs(1 / share - m) > 1e-9:
        raise ValueError(f"The share {share} does not split the sample into whole subgroups.")
    return m


def mde(variance: float, share: float, alpha: float, power: float) -> float:
    """The MDE of a subgroup of `share` against the rest, or of the ATE if share is 1."""
    m = subgroups_of_share(share)
    z = norm.ppf(1 - alpha / (2 * m)) + norm.ppf(power)
    if m == 1:
        return float(z * math.sqrt(variance))
    return float(z * math.sqrt(variance / (share * (1 - share))))
