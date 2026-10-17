"""The LightGBM fits the four learners share, and the propensity they may use."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from uplift.propensity_check import lightgbm_params


def params(cfg, objective: str, threads: int) -> dict:
    """The shared setting of `LearnerConfig`, for a classifier or a regressor."""
    if objective not in ("binary", "regression"):
        raise ValueError(f"Unknown objective {objective!r}; use 'binary' or 'regression'.")
    return {
        "objective": objective,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_child_samples,
        "seed": cfg.random_state,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
        "verbose": -1,
    }


def fit(x: np.ndarray, target: np.ndarray, cfg, objective: str, threads: int) -> lgb.Booster:
    """One LightGBM model with the shared setting and fixed rounds."""
    if x.shape[0] == 0:
        raise ValueError("Cannot fit a model on zero rows.")
    dataset = lgb.Dataset(x, label=target, free_raw_data=True)
    return lgb.train(params(cfg, objective, threads), dataset, cfg.n_estimators)


def propensity_model(x: np.ndarray, t: np.ndarray, propensity_cfg, threads: int) -> lgb.Booster:
    """The classifier of the propensity check, for a propensity that is not constant."""
    dataset = lgb.Dataset(x, label=t, free_raw_data=True)
    return lgb.train(lightgbm_params(propensity_cfg, threads), dataset, propensity_cfg.n_estimators)


def clip(e: np.ndarray, bound: float) -> np.ndarray:
    """e kept inside [bound, 1 - bound], so that 1 / e and 1 / (1 - e) stay finite."""
    return np.clip(e, bound, 1.0 - bound)
