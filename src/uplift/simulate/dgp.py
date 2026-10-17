"""The data generating process of the simulation study (docs/design.md §6.2).

`generate` draws one replication of one configuration of the grid. It returns
the covariates, the treatment and the observed outcome, which are all a learner
may see, and next to them the truth, which is for evaluation only: the true
effect after clipping, both outcome probabilities and the true propensity.

## The true effect is the clipped one

mu1 = clip(mu0 + tau, 0, 1), so where mu0 + tau leaves [0, 1] the effect a unit
can show is not the input tau. `tau_true = mu1 - mu0` is what the learners are
scored against; scoring against the input would charge them for the clipping.

## Reproducibility

Every quantity is computed from the float32 covariates the learners see, and
the draws follow a fixed order (covariates, treatment, then the two potential
outcomes), so a configuration, a sample size and a generator give the same data.
`replication_rng` derives the generator of replication r of configuration s
from the root seed in config.py, so any single replication can be rerun alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

N_FEATURES = 12
BASELINES = ("simple", "complex")
EFFECTS = ("heterogeneous", "step", "null", "sleeping_dogs")

#: The confounded propensity: logit(0.85) + 1.5 x1 (§6.2).
CONFOUNDED_RATIO = 0.85
CONFOUNDING_SLOPE = 1.5


@dataclass(frozen=True)
class Simulated:
    """One replication. Only `x`, `t` and `y` may reach a learner."""

    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    tau_true: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    e: np.ndarray


def baseline_logit(x: np.ndarray, kind: str) -> np.ndarray:
    """eta(x) of §6.2, without the intercept; columns 0..11 are x1..x12."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "simple":
        return 0.6 * x[:, 0] + 0.4 * x[:, 1] - 0.3 * x[:, 2]
    if kind == "complex":
        return (
            0.8 * np.sin(2 * x[:, 0])
            + 0.6 * x[:, 1] * x[:, 2]
            + 0.5 * np.abs(x[:, 3])
            - 0.4 * x[:, 4] ** 2
            + 0.3 * x[:, 5]
        )
    raise ValueError(f"Unknown baseline {kind!r}; the baselines are {BASELINES}.")


def effect(x: np.ndarray, kind: str, mean: float) -> np.ndarray:
    """tau(x) of §6.2 on the probability scale, with mean `mean` (or 0) over N(0, I)."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "heterogeneous":
        return 2 * mean * expit(x[:, 0] + x[:, 6])
    if kind == "step":
        return 2 * mean * (x[:, 6] > 0)
    if kind == "null":
        return np.zeros(x.shape[0])
    if kind == "sleeping_dogs":
        return mean * (1 + 2 * np.tanh(x[:, 6]))
    raise ValueError(f"Unknown effect {kind!r}; the effects are {EFFECTS}.")


def propensity(x: np.ndarray, treat_ratio: float, confounded: bool) -> np.ndarray:
    """e(x): the treatment ratio, or logit(0.85) + 1.5 x1 through the logistic."""
    if confounded:
        x1 = np.asarray(x, dtype=np.float64)[:, 0]
        return expit(logit(CONFOUNDED_RATIO) + CONFOUNDING_SLOPE * x1)
    return np.full(np.asarray(x).shape[0], float(treat_ratio))


def calibration_sample(rows: int, seed: int) -> np.ndarray:
    """The seeded draws of X on which every intercept is calibrated."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, N_FEATURES)).astype(np.float32)


def calibrate_intercept(eta: np.ndarray, base_rate: float, iterations: int = 200) -> float:
    """The a for which the mean of sigmoid(a + eta) is `base_rate`, by bisection.

    The mean is increasing in a, so bisection on a bracket that surely contains
    the root converges; 200 halvings of [-30, 30] reach the float64 resolution.
    """
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"The base rate must be in (0, 1), got {base_rate}.")
    low, high = -30.0, 30.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if expit(middle + eta).mean() < base_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def replication_rng(root_seed: int, scenario: int, replication: int) -> np.random.Generator:
    """The generator of replication `replication` of configuration `scenario`."""
    return np.random.default_rng(np.random.SeedSequence([root_seed, scenario, replication]))


def generate(
    scenario, n: int, rng: np.random.Generator, intercept: float, relative_effect: float
) -> Simulated:
    """One replication of `scenario` (anything shaped like config.Scenario), n units.

    `intercept` is the calibrated a of its baseline and base rate.
    """
    if n < 2:
        raise ValueError(f"Need at least two units, got {n}.")
    x = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    e = propensity(x, scenario.treat_ratio, scenario.confounded)
    t = (rng.random(n) < e).astype(np.int8)
    mu0 = expit(intercept + baseline_logit(x, scenario.baseline))
    tau = effect(x, scenario.effect, relative_effect * scenario.base_rate)
    mu1 = np.clip(mu0 + tau, 0.0, 1.0)
    y0 = rng.random(n) < mu0
    y1 = rng.random(n) < mu1
    y = np.where(t == 1, y1, y0).astype(np.int8)
    return Simulated(x=x, t=t, y=y, tau_true=mu1 - mu0, mu0=mu0, mu1=mu1, e=e)
