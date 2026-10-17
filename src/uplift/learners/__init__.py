"""The four meta-learners of docs/design.md §5, written by hand on LightGBM.

Each one has `fit(x, t, y)` and `predict(x)`, which returns tau_hat. They take
`cfg` shaped like `LearnerConfig` (and, for the X- and DR-learners, `PropensityConfig`
for the propensity classifier and its clip) and are told whether the propensity
is a known constant or must be estimated (§5.5).
"""

from uplift.learners.dr import DRLearner
from uplift.learners.s import SLearner
from uplift.learners.t import TLearner
from uplift.learners.x import XLearner

LEARNERS = {"S": SLearner, "T": TLearner, "X": XLearner, "DR": DRLearner}

__all__ = ["LEARNERS", "DRLearner", "SLearner", "TLearner", "XLearner"]
