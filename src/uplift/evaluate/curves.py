"""Uplift and Qini curves, the AUUC, and the value of contacting the top K (§7.1, §9).

Sort the units by a score, descending. For the top n, with N_T(n) and N_C(n)
treated and control units and Y_T(n) and Y_C(n) their summed outcomes:

    U(n) = (Y_T / N_T - Y_C / N_C) n        the incremental outcomes of treating the top n
    Q(n) = Y_T - Y_C N_T / N_C              the Qini curve

Units with the same score form one block. The curves are evaluated at block
boundaries and interpolated linearly in between, which is what a random order
within the block gives in expectation; a constant score is one block, so it gets
the random line, from (0, 0) to (N, U(N)). Where one arm is still missing at the
top of the ranking the curve is undefined, and the interpolation runs from the
origin to the first point where both arms are present.

`RankedScore` sorts once and evaluates the curves for any frequency weights, so
the bootstrap resamples units without sorting again.
"""

from __future__ import annotations

import numpy as np


class RankedScore:
    """A score sorted once, with its tie blocks, ready to give curves for any weights."""

    def __init__(self, score: np.ndarray, t: np.ndarray, y: np.ndarray):
        score = np.asarray(score, dtype=np.float64)
        self.order = np.argsort(-score, kind="stable")
        ranked = score[self.order]
        self.starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
        self.t = np.asarray(t, dtype=np.float64)[self.order]
        self.y = np.asarray(y, dtype=np.float64)[self.order]

    def curves(self, phi: np.ndarray, weights: np.ndarray | None = None):
        """U and Q at the fractions `phi` of the (weighted) number of units."""
        w = np.ones(self.t.size) if weights is None else np.asarray(weights, np.float64)[self.order]
        treated = w * self.t
        control = w - treated
        nt = np.add.reduceat(treated, self.starts).cumsum()
        nc = np.add.reduceat(control, self.starts).cumsum()
        yt = np.add.reduceat(treated * self.y, self.starts).cumsum()
        yc = np.add.reduceat(control * self.y, self.starts).cumsum()
        n = nt + nc
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (yt / nt - yc / nc) * n
            q = yt - yc * nt / nc
        defined = (nt > 0) & (nc > 0)
        if not defined[-1]:
            raise ValueError("Both arms must be present.")
        x = np.r_[0.0, n[defined]]
        at = np.asarray(phi, dtype=np.float64) * n[-1]
        return np.interp(at, x, np.r_[0.0, u[defined]]), np.interp(at, x, np.r_[0.0, q[defined]])


def area_over_random(curve: np.ndarray, phi: np.ndarray, n: float) -> float:
    """(1/N) * integral over phi of [curve(phi N) - phi curve(N)], by trapezoids (§7.1).

    `phi` must run from 0 to 1, so that curve[-1] is the value at N.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi[0] != 0.0 or phi[-1] != 1.0:
        raise ValueError("The fraction axis must run from 0 to 1.")
    gap = np.asarray(curve) - phi * curve[-1]
    return float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(phi)) / n)


def break_even(curve: np.ndarray, phi: np.ndarray, n: float, cost_over_value: float) -> float:
    """The first fraction at which one more contact returns less than it costs.

    The marginal incremental outcome per contact is the slope of U between grid
    points; contacting pays while slope * value > cost. Returns 0 if the first
    contacts already do not pay, and 1 if every contact does.
    """
    slopes = np.diff(curve) / (np.diff(phi) * n)
    below = np.flatnonzero(slopes <= cost_over_value)
    return float(phi[below[0]]) if below.size else 1.0
