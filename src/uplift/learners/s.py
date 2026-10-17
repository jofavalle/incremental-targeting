"""S-learner: one model on (X, T), tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0) (§5.1).

The treatment is appended as a thirteenth column, on purpose: it is not a feature
of `feature_matrix`, and nothing else in the project appends it.
"""

from __future__ import annotations

import numpy as np

from uplift.learners import base


def _with_treatment(x: np.ndarray, t) -> np.ndarray:
    column = np.broadcast_to(np.asarray(t, dtype=np.float32), (x.shape[0],))
    return np.column_stack([x, column]).astype(np.float32)


class SLearner:
    def __init__(self, cfg, threads: int = 2):
        self.cfg, self.threads = cfg, threads

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> SLearner:
        self.model = base.fit(_with_treatment(x, t), y, self.cfg, "binary", self.threads)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        treated = self.model.predict(_with_treatment(x, 1))
        control = self.model.predict(_with_treatment(x, 0))
        return treated - control
