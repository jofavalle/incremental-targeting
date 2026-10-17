"""X-learner, built for unbalanced treatment (§5.3).

1. mu_hat_0 on the controls and mu_hat_1 on the treated, as the T-learner does.
2. Imputed effects: D1 = Y - mu_hat_0(X) for the treated, D0 = mu_hat_1(X) - Y
   for the controls. Each model is evaluated on the arm it was not fitted on.
3. tau_hat_1 regresses D1 on X over the treated, tau_hat_0 regresses D0 over the
   controls.
4. tau_hat(x) = e(x) tau_hat_0(x) + (1 - e(x)) tau_hat_1(x).

With e near 0.85 the weight goes to tau_hat_0: it is fitted on the small control
arm, but its imputation D0 rests on mu_hat_1, learnt on the large treated arm.
The propensity is the treated share when it is a known constant, or the
propensity classifier, clipped, when it is not (§5.5).
"""

from __future__ import annotations

import numpy as np

from uplift.learners import base


class XLearner:
    def __init__(self, cfg, propensity_cfg, constant_propensity: bool = True, threads: int = 2):
        self.cfg, self.propensity_cfg, self.threads = cfg, propensity_cfg, threads
        self.constant_propensity = constant_propensity

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> XLearner:
        treated = np.asarray(t).astype(bool)
        y = np.asarray(y, dtype=np.float64)
        mu0 = base.fit(x[~treated], y[~treated], self.cfg, "binary", self.threads)
        mu1 = base.fit(x[treated], y[treated], self.cfg, "binary", self.threads)
        d1 = y[treated] - mu0.predict(x[treated])
        d0 = mu1.predict(x[~treated]) - y[~treated]
        self.tau1 = base.fit(x[treated], d1, self.cfg, "regression", self.threads)
        self.tau0 = base.fit(x[~treated], d0, self.cfg, "regression", self.threads)
        if self.constant_propensity:
            self.share, self.e_model = float(treated.mean()), None
        else:
            self.e_model = base.propensity_model(x, t, self.propensity_cfg, self.threads)
        return self

    def propensity(self, x: np.ndarray) -> np.ndarray:
        if self.e_model is None:
            return np.full(x.shape[0], self.share)
        return base.clip(self.e_model.predict(x), self.propensity_cfg.clip)

    def predict(self, x: np.ndarray) -> np.ndarray:
        e = self.propensity(x)
        return e * self.tau0.predict(x) + (1 - e) * self.tau1.predict(x)
