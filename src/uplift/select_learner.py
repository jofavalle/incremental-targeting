"""The learner chosen on validation by the transformed-outcome criterion (docs/design.md §6.3).

    python -m uplift.select_learner               # 2 threads, as on the smallest machine
    python -m uplift.select_learner --threads 6

The four learners are fitted on train for the primary outcome, which the power
analysis decided (`reports/power.json`), and scored on validation. The rule was
fixed in `SelectionConfig` and `design.md` §6.3, and pushed, before the ATE was
seen. It writes:

    reports/selection.json                   the record, versioned
    reports/figures/fig5_calibration.png     calibration by decile of tau_hat, per learner
    data/processed/learners.pkl              the fitted learners, for the test (not versioned)

## What it reads

Train and validation, never test: the learners are pickled so that the test is
opened once, later, by `uplift.evaluate_test`. The features reach the learners
only through `feature_matrix`.

## What the record holds

The criterion of each learner, the paired bootstrap interval of each difference
from the chosen one (smallest criterion), and the two cheap checks of §6.4 on
validation: the mean of tau_hat against the ATE, and the calibration by decile.
A choice that no interval supports is reported as such, not replaced.
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import FEATURES
from uplift.evaluate.effect import difference_of_means, paired_bootstrap, percentile_interval
from uplift.evaluate.selection import calibration_by_decile, criterion, transformed_outcome
from uplift.features import feature_matrix
from uplift.learners import LEARNERS
from uplift.runinfo import run_metadata, write_json

DEFAULT_OUT = REPORTS_DIR / "selection.json"
DEFAULT_FIGURE = REPORTS_DIR / "figures" / "fig5_calibration.png"
DEFAULT_POWER = REPORTS_DIR / "power.json"
DEFAULT_MODELS = DATA_PROCESSED / "learners.pkl"


def _load(path: Path, outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=[*FEATURES, "treatment", outcome])
    return feature_matrix(table), table["treatment"].to_numpy(), table[outcome].to_numpy()


def fit_learners(x, t, y, cfg, threads: int) -> dict:
    """The four learners on one sample; the propensity is the known constant (§5.5)."""
    fitted = {}
    for name, cls in LEARNERS.items():
        started = time.perf_counter()
        if name in ("X", "DR"):
            learner = cls(cfg.learners, cfg.propensity, constant_propensity=True, threads=threads)
        else:
            learner = cls(cfg.learners, threads=threads)
        fitted[name] = learner.fit(x, t, y)
        print(f"  {name}: fitted in {time.perf_counter() - started:.0f} s", flush=True)
    return fitted


def select(tau_hat: dict, y, t, cfg, rng) -> dict:
    """Criteria, the chosen learner and the paired intervals of the differences."""
    z = transformed_outcome(y, t, float(t.mean()))
    names = list(tau_hat)
    values = {name: criterion(tau_hat[name], z) for name in names}
    chosen = min(values, key=values.get)

    def statistic(counts):
        return [criterion(tau_hat[name], z, weights=counts) for name in names]

    draws = paired_bootstrap(statistic, t.size, cfg.selection.n_bootstrap, rng)
    j = names.index(chosen)
    separation = {}
    for i, name in enumerate(names):
        if name != chosen:
            low, high = percentile_interval(
                draws[:, i : i + 1] - draws[:, j : j + 1], cfg.effect.level
            )[:, 0]
            separation[name] = {
                "difference": values[name] - values[chosen],
                "ci": [float(low), float(high)],
                "separated": bool(low > 0),
            }
    return {
        "criterion": values,
        "chosen": chosen,
        "difference_from_chosen": separation,
        "supported_by_every_interval": all(s["separated"] for s in separation.values()),
        "e": float(t.mean()),
    }


def run_selection(train_path: Path, val_path: Path, power_path: Path, cfg, threads: int):
    """Fit on train, score on validation; returns the record and the fitted learners."""
    started = time.perf_counter()
    pa.set_cpu_count(threads)
    outcome = json.loads(power_path.read_text())["decision"]["primary"]
    x, t, y = _load(train_path, outcome)
    print(f"fitting the four learners on train for {outcome}", flush=True)
    fitted = fit_learners(x, t, y, cfg, threads)
    del x
    x_val, t_val, y_val = _load(val_path, outcome)
    tau_hat = {name: learner.predict(x_val) for name, learner in fitted.items()}
    del x_val

    rng = np.random.default_rng(cfg.selection.bootstrap_seed)
    ate = difference_of_means(y_val, t_val, cfg.effect.level)
    checks = {
        name: {
            "mean_tau_hat": float(values.mean()),
            "within_ate_interval": bool(ate["ci"][0] <= values.mean() <= ate["ci"][1]),
            "calibration": calibration_by_decile(values, y_val, t_val, cfg.effect.level),
        }
        for name, values in tau_hat.items()
    }
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "outcome": outcome,
        "partition": "val",
        **select(tau_hat, y_val, t_val, cfg, rng),
        "ate_val": {"estimate": ate["estimate"], "ci": ate["ci"]},
        "checks": checks,
        "n_bootstrap": cfg.selection.n_bootstrap,
        "seed": cfg.selection.bootstrap_seed,
        "learner_setting": {
            k: getattr(cfg.learners, k)
            for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")
        },
        "run": {
            **run_metadata(packages=("numpy", "pyarrow", "lightgbm", "scipy")),
            "threads": threads,
            "seconds": round(time.perf_counter() - started, 1),
            "peak_rss_mib": round(peak_kib / 1024),
        },
    }
    return record, fitted


def plot_calibration(record: dict, path: Path) -> None:
    """Observed uplift against mean prediction by decile, one panel per learner."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(record["checks"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    for ax, name in zip(axes, names):
        rows = record["checks"][name]["calibration"]
        predicted = [100 * r["mean_prediction"] for r in rows]
        observed = [100 * r["observed"] for r in rows]
        low = [100 * (r["observed"] - r["ci"][0]) for r in rows]
        high = [100 * (r["ci"][1] - r["observed"]) for r in rows]
        ax.errorbar(predicted, observed, yerr=[low, high], fmt="o", capsize=2)
        edge = [min(predicted + observed), max(predicted + observed)]
        ax.plot(edge, edge, color="black", linewidth=0.8)
        chosen = " (chosen)" if name == record["chosen"] else ""
        ax.set_title(f"{name}-learner{chosen}", fontsize=9)
        ax.set_xlabel("mean predicted uplift, points")
    axes[0].set_ylabel("observed uplift, points")
    fig.suptitle(
        f"Calibration by decile of tau_hat on validation, {record['outcome']}", fontsize=10
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.parquet")
    parser.add_argument("--val", type=Path, default=DATA_PROCESSED / "val.parquet")
    parser.add_argument("--power", type=Path, default=DEFAULT_POWER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    record, fitted = run_selection(args.train, args.val, args.power, CONFIG, args.threads)
    write_json(args.out, record)
    plot_calibration(record, args.figure)
    args.models.parent.mkdir(parents=True, exist_ok=True)
    with args.models.open("wb") as handle:
        pickle.dump({"outcome": record["outcome"], "learners": fitted}, handle)
    print(f"chosen: {record['chosen']}; record -> {args.out}; learners -> {args.models}")


if __name__ == "__main__":
    main()
