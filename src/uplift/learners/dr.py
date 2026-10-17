"""DR-learner: a cross-fitted AIPW pseudo-outcome regressed on X (§5.4).

    phi = mu_hat_1(X) - mu_hat_0(X)
          + T (Y - mu_hat_1(X)) / e(X) - (1 - T) (Y - mu_hat_0(X)) / (1 - e(X))

Its expectation given X is tau(X) if either the outcome models or the propensity
are right. The nuisance models of each unit's phi are fitted on the other folds
(`assign_folds`, stratified by the treatment), so no pseudo-outcome uses a model
that saw its unit. The propensity is the treated share when it is a known
constant, or the propensity classifier, cross-fitted and clipped, when it is
not (§5.5). Written by hand, without econml or causalml.
"""

from __future__ import annotations

import numpy as np

from uplift.data.split import assign_folds
from uplift.learners import base


class DRLearner:
    def __init__(self, cfg, propensity_cfg, constant_propensity: bool = True, threads: int = 2):
        self.cfg, self.propensity_cfg, self.threads = cfg, propensity_cfg, threads
        self.constant_propensity = constant_propensity

    def pseudo_outcome(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """phi for every unit, from nuisance models fitted on the other folds."""
        t = np.asarray(t).astype(np.int8)
        treated = t.astype(bool)
        y = np.asarray(y, dtype=np.float64)
        folds = assign_folds([t], self.cfg.n_folds, self.cfg.fold_seed)
        phi = np.empty(t.size)
        share = float(treated.mean())
        for k in range(self.cfg.n_folds):
            held = folds == k
            fit_on = ~held
            mu0 = base.fit(
                x[fit_on & ~treated], y[fit_on & ~treated], self.cfg, "binary", self.threads
            )
            mu1 = base.fit(
                x[fit_on & treated], y[fit_on & treated], self.cfg, "binary", self.threads
            )
            if self.constant_propensity:
                e = np.full(int(held.sum()), share)
            else:
                model = base.propensity_model(
                    x[fit_on], t[fit_on], self.propensity_cfg, self.threads
                )
                e = base.clip(model.predict(x[held]), self.propensity_cfg.clip)
            m0, m1 = mu0.predict(x[held]), mu1.predict(x[held])
            th, yh = t[held], y[held]
            phi[held] = m1 - m0 + th * (yh - m1) / e - (1 - th) * (yh - m0) / (1 - e)
        return phi

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> DRLearner:
        self.model = base.fit(x, self.pseudo_outcome(x, t, y), self.cfg, "regression", self.threads)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
