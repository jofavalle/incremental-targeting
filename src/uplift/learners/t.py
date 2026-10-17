"""T-learner: two models, tau_hat(x) = mu_hat_1(x) - mu_hat_0(x) (§5.2)."""

from __future__ import annotations

import numpy as np

from uplift.learners import base


class TLearner:
    def __init__(self, cfg, threads: int = 2):
        self.cfg, self.threads = cfg, threads

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> TLearner:
        treated = np.asarray(t).astype(bool)
        self.mu1 = base.fit(x[treated], y[treated], self.cfg, "binary", self.threads)
        self.mu0 = base.fit(x[~treated], y[~treated], self.cfg, "binary", self.threads)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.mu1.predict(x) - self.mu0.predict(x)
