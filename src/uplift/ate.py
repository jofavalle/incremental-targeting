"""The average effect on visit and conversion, with its three sensitivity versions.

    python -m uplift.ate               # 2 threads, as on the smallest machine
    python -m uplift.ate --threads 6

The primary estimate is the difference of means with its normal interval
(docs/design.md §2.3). The weighted, subclassified and block versions of §6.5 go
next to it, each with its bootstrap interval and the interval of its difference
from the primary; they never replace it. Every setting was fixed in
`EffectConfig` and pushed before this driver existed. It writes:

    reports/ate.json                       the record, versioned
    reports/figures/fig2_ate_versions.png  the four estimates, by outcome and partition

## What it reads

Train and validation, never test (§4.2). The features reach the classifier only
through `feature_matrix`. `row_id` is read for one purpose, to place each unit in
its block of the source file, and never enters a model (§3.3).

## The cross-fitted propensity

Train is cut into folds by `assign_folds`. Each train unit gets the e_hat of the
model fitted on the other folds, and each validation unit the mean of the fold
models, all of them out of sample for it (§6.5). The classifier and the clip are
those of the propensity check, unchanged.

## What the intervals measure

One paired bootstrap per partition resamples the units with e_hat, strata and
blocks held fixed. It measures the sampling noise given e_hat, not the noise of
estimating e_hat. The block version rests on a reading the source does not
confirm (that each pair of runs is one of the assembled tests), and the record
says so next to it.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import FEATURES, ROW_ID
from uplift.data.split import assign_folds
from uplift.evaluate.effect import (
    difference_of_means,
    hajek,
    paired_bootstrap,
    percentile_interval,
    propensity_strata,
    source_blocks,
    stratified_difference,
)
from uplift.features import feature_matrix
from uplift.propensity_check import lightgbm_params
from uplift.runinfo import run_metadata, write_json

DEFAULT_OUT = REPORTS_DIR / "ate.json"
DEFAULT_FIGURE = REPORTS_DIR / "figures" / "fig2_ate_versions.png"
DEFAULT_SUMMARY = REPORTS_DIR / "data_summary.json"

OUTCOMES = ("visit", "conversion")
VERSIONS = ("primary", "weighted", "subclassified", "blocks")
BLOCKS_CONDITION = (
    "conditional on the unconfirmed reading that each pair of runs of the source "
    "file is one of the assembled tests (docs/design.md §6.5)"
)
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _load(path: Path) -> dict:
    """The features, the treatment, both outcomes and `row_id` of one partition."""
    table = pq.read_table(path, columns=[*FEATURES, "treatment", *OUTCOMES, ROW_ID])
    data = {name: table[name].to_numpy() for name in ("treatment", *OUTCOMES, ROW_ID)}
    data["x"] = feature_matrix(table)
    return data


def cross_fit(x: np.ndarray, t: np.ndarray, folds: np.ndarray, params: dict, rounds: int):
    """Out-of-fold e_hat on the fitted rows, and the fold models for new rows."""
    e_hat = np.empty(t.size)
    boosters = []
    for k in range(int(folds.max()) + 1):
        held = folds == k
        dataset = lgb.Dataset(x[~held], label=t[~held], free_raw_data=True)
        booster = lgb.train(params, dataset, rounds)
        del dataset
        e_hat[held] = booster.predict(x[held])
        boosters.append(booster)
        print(f"  fold {k}: fitted on {int((~held).sum())} rows", flush=True)
    return e_hat, boosters


def _versions(data: dict, e_hat, strata, blocks, clip, weights=None) -> list[float]:
    """The four estimates for each outcome, in the order OUTCOMES x VERSIONS."""
    t = data["treatment"]
    values = []
    for outcome in OUTCOMES:
        y = data[outcome]
        values += [
            stratified_difference(y, t, weights=weights),
            hajek(y, t, e_hat, clip, weights=weights),
            stratified_difference(y, t, strata, weights=weights),
            stratified_difference(y, t, blocks, weights=weights),
        ]
    return values


def _groups(labels: np.ndarray, t: np.ndarray) -> list[dict]:
    return [
        {"rows": int((labels == g).sum()), "treated": int(t[labels == g].sum())}
        for g in range(int(labels.max()) + 1)
    ]


def estimate_partition(data: dict, e_hat, runs, cfg, clip, rng, label: str) -> dict:
    """Point estimates, intervals and the strata and blocks behind them, for one partition."""
    t = data["treatment"]
    strata = propensity_strata(e_hat, t, cfg.stratum_quantiles)
    blocks = source_blocks(data[ROW_ID], t, runs)
    point = _versions(data, e_hat, strata, blocks, clip)

    started = time.perf_counter()

    def statistic(counts):
        return _versions(data, e_hat, strata, blocks, clip, weights=counts)

    draws = paired_bootstrap(statistic, t.size, cfg.n_bootstrap, rng)
    print(f"  {label}: {cfg.n_bootstrap} resamples in {time.perf_counter() - started:.0f} s")
    k = len(VERSIONS)
    outcomes = {}
    for i, outcome in enumerate(OUTCOMES):
        block = draws[:, i * k : (i + 1) * k]
        ci = percentile_interval(block, cfg.level)
        diff_ci = percentile_interval(block[:, 1:] - block[:, :1], cfg.level)
        primary = difference_of_means(data[outcome], t, cfg.level)
        entry = {
            "primary": {
                "estimate": primary["estimate"],
                "ci": primary["ci"],
                "ci_bootstrap": ci[:, 0].tolist(),
                "p1": primary["p1"],
                "p0": primary["p0"],
            }
        }
        for j, version in enumerate(VERSIONS[1:], start=1):
            entry[version] = {
                "estimate": point[i * k + j],
                "ci": ci[:, j].tolist(),
                "difference": point[i * k + j] - point[i * k],
                "difference_ci": diff_ci[:, j - 1].tolist(),
            }
        entry["blocks"]["condition"] = BLOCKS_CONDITION
        outcomes[outcome] = entry

    clipped = (e_hat < clip) | (e_hat > 1 - clip)
    return {
        "rows": int(t.size),
        "treated": int(t.sum()),
        "e_hat": {f"q{q:g}": float(v) for q, v in zip(QUANTILES, np.quantile(e_hat, QUANTILES))},
        "e_hat_clipped": int(clipped.sum()),
        "strata": _groups(strata, t),
        "blocks": _groups(blocks, t),
        "outcomes": outcomes,
    }


def run_ate(train_path: Path, val_path: Path, summary_path: Path, cfg, threads: int) -> dict:
    """Cross-fit e_hat, estimate on train and validation, and return the record.

    `cfg` is anything with `effect` and `propensity` attributes shaped like
    `EffectConfig` and `PropensityConfig`.
    """
    started = time.perf_counter()
    pa.set_cpu_count(threads)
    effect, propensity = cfg.effect, cfg.propensity
    runs = json.loads(summary_path.read_text())["source_order"]["treatment_run_lengths"]
    params = lightgbm_params(propensity, threads)

    train = _load(train_path)
    strata_columns = [train[name] for name in effect.fold_strata]
    folds = assign_folds(strata_columns, effect.n_folds, effect.fold_seed)
    print(f"cross-fitting e_hat on train, {effect.n_folds} folds", flush=True)
    e_train, boosters = cross_fit(
        train.pop("x"), train["treatment"], folds, params, propensity.n_estimators
    )

    val = _load(val_path)
    e_val = np.mean([booster.predict(val["x"]) for booster in boosters], axis=0)
    del val["x"]

    rng = np.random.default_rng(effect.bootstrap_seed)
    partitions = {}
    for label, data, e_hat in (("train", train, e_train), ("val", val, e_val)):
        partitions[label] = estimate_partition(
            data, e_hat, runs, effect, propensity.clip, rng, label
        )
    partitions["train"]["e_hat_source"] = "out of fold"
    partitions["val"]["e_hat_source"] = "mean of the fold models"

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "partitions": partitions,
        "level": effect.level,
        "n_bootstrap": effect.n_bootstrap,
        "n_folds": effect.n_folds,
        "fold_strata": list(effect.fold_strata),
        "stratum_quantiles": list(effect.stratum_quantiles),
        "clip": propensity.clip,
        "seeds": {"folds": effect.fold_seed, "bootstrap": effect.bootstrap_seed},
        "classifier": {"n_estimators": propensity.n_estimators, **params},
        "run": {
            **run_metadata(packages=("numpy", "pyarrow", "lightgbm", "scipy")),
            "threads": threads,
            "seconds": round(time.perf_counter() - started, 1),
            "peak_rss_mib": round(peak_kib / 1024),
        },
    }


def plot_versions(record: dict, path: Path) -> None:
    """The four estimates with their intervals, one panel per outcome, in points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["primary", "weighted", "subclassified", "blocks\n(if each pair\nof runs is a test)"]
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(10, 4), sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        for offset, (partition, marker) in zip((-0.12, 0.12), (("train", "o"), ("val", "s"))):
            entry = record["partitions"][partition]["outcomes"][outcome]
            for row, version in enumerate(VERSIONS):
                estimate = 100 * entry[version]["estimate"]
                low, high = (100 * v for v in entry[version]["ci"])
                ax.errorbar(
                    estimate,
                    row + offset,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=marker,
                    color="C0" if partition == "train" else "C1",
                    capsize=3,
                    label=partition if row == 0 else None,
                )
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"ATE on {outcome}")
        ax.set_xlabel("effect, percentage points")
        ax.legend(fontsize=8)
    axes[0].set_yticks(range(len(VERSIONS)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    level = round(100 * record["level"])
    fig.suptitle(f"The primary estimate and its sensitivity versions, {level} % intervals")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.parquet")
    parser.add_argument("--val", type=Path, default=DATA_PROCESSED / "val.parquet")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    record = run_ate(args.train, args.val, args.summary, CONFIG, args.threads)
    write_json(args.out, record)
    plot_versions(record, args.figure)
    for outcome in OUTCOMES:
        entry = record["partitions"]["val"]["outcomes"][outcome]
        low, high = entry["primary"]["ci"]
        print(f"{outcome}: ATE {entry['primary']['estimate']:.5f} [{low:.5f}, {high:.5f}] (val)")
    print(f"record -> {args.out}; figure -> {args.figure}")


if __name__ == "__main__":
    main()
