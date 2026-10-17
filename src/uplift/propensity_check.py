"""Propensity check driver: can the features tell the two arms apart?

    python -m uplift.propensity_check               # 2 threads, as on the smallest machine
    python -m uplift.propensity_check --threads 6

It runs before any ATE (docs/design.md §6.4) and decides which branch the rest of
the analysis follows: a constant propensity, or the weighted forms of §2.3. The
rule (`PropensityConfig` in config.py) was committed before this driver existed.
It writes:

    reports/propensity_check.json                  the record, versioned
    reports/figures/fig1_propensity_overlap.png    the estimated propensity by arm

## The single-look rule

Only train and validation are read, and only the features and the treatment,
through `feature_matrix`: no outcome is looked at, and the test partition is not
opened. The classifier is fitted on train and scored on validation, so its AUC is
out of sample: a classifier that only memorized noise scores 0.5 there.

## What a pass means

A constant propensity is never confirmed, only not refuted: tests with different
treatment ratios that the features cannot tell apart are invisible to any
classifier on the features (§2.3).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import FEATURES
from uplift.evaluate.balance import (
    auc,
    bootstrap_auc_interval,
    linear_propensity_sd_bound,
    propensity_is_constant,
    propensity_sd_if_normal,
    smd_noise_se,
    standardized_mean_differences,
)
from uplift.features import feature_matrix
from uplift.runinfo import run_metadata, write_json

DEFAULT_OUT = REPORTS_DIR / "propensity_check.json"
DEFAULT_FIGURE = REPORTS_DIR / "figures" / "fig1_propensity_overlap.png"

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The feature matrix and the treatment of one partition, and nothing else."""
    table = pq.read_table(path, columns=[*FEATURES, "treatment"])
    return feature_matrix(table), table["treatment"].to_numpy()


def _lightgbm_params(cfg, threads: int) -> dict:
    return {
        "objective": "binary",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_child_samples,
        "seed": cfg.random_state,
        # Same inputs, parameters and thread count give the same model.
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
        "verbose": -1,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{q:g}": float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))}


def run_check(
    train_path: Path, val_path: Path, cfg, threads: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit on train, score validation and apply the rule.

    Returns the record, and e_hat and the treatment on validation for the figure.
    `cfg` is anything with the fields of `PropensityConfig`.
    """
    started = time.perf_counter()
    pa.set_cpu_count(threads)

    x_train, t_train = _load(train_path)
    smd = standardized_mean_differences(x_train, t_train)
    sd_bound, sd_noise = linear_propensity_sd_bound(x_train, t_train)
    n1_train = int(t_train.sum())
    params = _lightgbm_params(cfg, threads)
    booster = lgb.train(params, lgb.Dataset(x_train, label=t_train), cfg.n_estimators)
    train_rows, train_share = t_train.size, float(t_train.mean())
    del x_train

    x_val, t_val = _load(val_path)
    e_hat = booster.predict(x_val)
    del x_val
    auc_value = auc(e_hat, t_val)
    low, high = bootstrap_auc_interval(e_hat, t_val, cfg.n_bootstrap, cfg.random_state)
    constant = propensity_is_constant(low, cfg.auc_threshold)
    treated = t_val.astype(bool)

    record = {
        "partitions": {
            "train": {"rows": train_rows, "treatment_share": train_share},
            "val": {"rows": int(t_val.size), "treatment_share": float(t_val.mean())},
        },
        "smd_train": dict(zip(FEATURES, smd.tolist())),
        "smd_flagged": [name for name, v in zip(FEATURES, smd) if abs(v) > cfg.smd_flag],
        "smd_flag": cfg.smd_flag,
        "smd_noise_se": smd_noise_se(n1_train, train_rows - n1_train),
        "auc_val": auc_value,
        "auc_val_ci95": [low, high],
        "auc_threshold": cfg.auc_threshold,
        # The size of the spread: a bound from the mean differences on train, with
        # its noise floor, and the normal-only translation of the AUC (which is
        # neither an estimate nor a bound for a skewed propensity).
        "propensity_sd_linear_bound": sd_bound,
        "propensity_sd_linear_noise": sd_noise,
        "propensity_sd_if_normal": propensity_sd_if_normal(auc_value, float(t_val.mean())),
        "decision": "constant" if constant else "weighted",
        "e_hat_val": {
            "treated": _quantiles(e_hat[treated]),
            "control": _quantiles(e_hat[~treated]),
        },
        "classifier": {"n_estimators": cfg.n_estimators, **params},
        "n_bootstrap": cfg.n_bootstrap,
        "seed": cfg.random_state,
        "run": {
            **run_metadata(packages=("numpy", "pyarrow", "lightgbm", "scipy")),
            "threads": threads,
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
    return record, e_hat, t_val


def plot_overlap(e_hat: np.ndarray, t: np.ndarray, path: Path, note: str) -> None:
    """Histogram of the estimated propensity by arm, as densities on common bins.

    The bins span the central 99.9 % of e_hat: a handful of extreme values would
    otherwise stretch the axis and squeeze the mass into one bar. The density axis
    is logarithmic, because the arms differ, if at all, in the tails.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    treated = t.astype(bool)
    low, high = np.quantile(e_hat, [0.0005, 0.9995])
    bins = np.linspace(low, high, 81)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(e_hat[treated], bins=bins, density=True, alpha=0.6, label="treated")
    ax.hist(e_hat[~treated], bins=bins, density=True, alpha=0.6, label="control")
    ax.axvline(t.mean(), color="black", linewidth=1, label=f"treated share {t.mean():.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("estimated propensity on validation, e_hat(x) (central 99.9 %)")
    ax.set_ylabel("density (log scale)")
    ax.set_title(f"Can the features tell the arms apart?\n{note}", fontsize=9)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.parquet")
    parser.add_argument("--val", type=Path, default=DATA_PROCESSED / "val.parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    record, e_hat, t_val = run_check(args.train, args.val, CONFIG.propensity, args.threads)
    write_json(args.out, record)
    low, high = record["auc_val_ci95"]
    note = f"AUC {record['auc_val']:.4f} [{low:.4f}, {high:.4f}], rule: weighted if low > "
    plot_overlap(e_hat, t_val, args.figure, note + f"{record['auc_threshold']}")
    print(f"{note}{record['auc_threshold']} -> {record['decision']}; record -> {args.out}")


if __name__ == "__main__":
    main()
