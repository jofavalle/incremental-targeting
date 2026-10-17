"""The simulation study: four learners against the truth, on the eight configurations.

    python -m uplift.simulate.study               # 2 threads, as on the smallest machine
    python -m uplift.simulate.study --threads 6

Each replication draws a training sample and an independent validation sample
of the same configuration. The learners are fitted on the first and scored on the
second, by the PEHE and by the transformed-outcome criterion (docs/design.md
§6.2, §6.3). The values were fixed in `SimulationConfig` and pushed before this
driver existed. It writes:

    reports/simulation.json                    the record, versioned
    reports/figures/fig4_pehe_by_scenario.png  the PEHE by configuration and learner

## Resuming

The study takes hours. Every finished replication is appended to a work file
under data/processed/, tagged with a fingerprint of the settings it ran with; a
restarted run skips the replications already there with the same fingerprint,
so an interruption costs at most one replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.evaluate.selection import criterion, pehe, transformed_outcome
from uplift.learners import LEARNERS
from uplift.runinfo import run_metadata, write_json
from uplift.simulate.dgp import (
    baseline_logit,
    calibrate_intercept,
    calibration_sample,
    generate,
    replication_rng,
)

DEFAULT_OUT = REPORTS_DIR / "simulation.json"
DEFAULT_FIGURE = REPORTS_DIR / "figures" / "fig4_pehe_by_scenario.png"
DEFAULT_WORK = DATA_PROCESSED / "simulation_replications.jsonl"

LEARNER_NAMES = tuple(LEARNERS)


def fingerprint(cfg, threads: int) -> str:
    """A short hash of everything a replication's numbers depend on.

    The settings, the thread count (a LightGBM fit is reproducible only at a
    fixed count) and the versions of the packages that compute it: a run resumed
    with any of them changed starts afresh instead of mixing replications.
    """
    versions = run_metadata(packages=("numpy", "scipy", "lightgbm"))["packages"]
    settings = {k: asdict(getattr(cfg, k)) for k in ("simulation", "learners", "propensity")}
    payload = {"settings": settings, "threads": threads, "packages": versions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def intercepts(cfg) -> dict[tuple[str, float], float]:
    """The calibrated intercept of every baseline and base rate of the grid."""
    sim = cfg.simulation
    x = calibration_sample(sim.calibration_rows, sim.calibration_seed)
    pairs = {(s.baseline, s.base_rate) for s in sim.scenarios}
    return {
        (baseline, rate): calibrate_intercept(baseline_logit(x, baseline), rate)
        for baseline, rate in sorted(pairs)
    }


def _learner(name: str, cfg, constant_propensity: bool, threads: int):
    if name in ("X", "DR"):
        return LEARNERS[name](cfg.learners, cfg.propensity, constant_propensity, threads)
    return LEARNERS[name](cfg.learners, threads)


def run_replication(scenario, index: int, replication: int, intercept: float, cfg, threads: int):
    """Fit the four learners on one training draw and score them on a validation draw."""
    started = time.perf_counter()
    sim = cfg.simulation
    rng = replication_rng(sim.replication_seed, index, replication)
    train = generate(scenario, sim.n, rng, intercept, sim.relative_effect)
    val = generate(scenario, sim.n_validation, rng, intercept, sim.relative_effect)
    # Z uses the known ratio, or the true e(x) where it varies: the check is the
    # criterion's ranking, not the estimation of e (§6.2).
    z = transformed_outcome(val.y, val.t, val.e if scenario.confounded else scenario.treat_ratio)
    learners = {}
    for name in LEARNER_NAMES:
        model = _learner(name, cfg, not scenario.confounded, threads)
        tau_hat = model.fit(train.x, train.t, train.y).predict(val.x)
        learners[name] = {
            "pehe": pehe(tau_hat, val.tau_true),
            "criterion": criterion(tau_hat, z),
            "mean_tau_hat": float(tau_hat.mean()),
        }
    return {
        "scenario": scenario.name,
        "replication": replication,
        "sd_tau_true": float(val.tau_true.std()),
        "mean_tau_true": float(val.tau_true.mean()),
        "learners": learners,
        "seconds": round(time.perf_counter() - started, 1),
    }


def summarize(runs: list[dict]) -> dict:
    """Mean PEHE with its Monte Carlo error, pairwise differences and criterion agreement."""
    r = len(runs)
    pehes = np.array([[run["learners"][name]["pehe"] for name in LEARNER_NAMES] for run in runs])
    crits = np.array(
        [[run["learners"][name]["criterion"] for name in LEARNER_NAMES] for run in runs]
    )
    bias = np.array(
        [
            [run["learners"][name]["mean_tau_hat"] - run["mean_tau_true"] for name in LEARNER_NAMES]
            for run in runs
        ]
    )
    sd_tau = np.array([run["sd_tau_true"] for run in runs])

    def mean_se(values):
        return {"mean": float(values.mean()), "mc_se": float(values.std(ddof=1) / np.sqrt(r))}

    summary = {
        "replications": r,
        "pehe": {name: mean_se(pehes[:, j]) for j, name in enumerate(LEARNER_NAMES)},
        "pehe_normalized": None,
        "criterion": {name: float(crits[:, j].mean()) for j, name in enumerate(LEARNER_NAMES)},
        "mean_tau_hat_minus_true": {
            name: float(bias[:, j].mean()) for j, name in enumerate(LEARNER_NAMES)
        },
        "pairwise_pehe_difference": {
            f"{LEARNER_NAMES[a]}-{LEARNER_NAMES[b]}": mean_se(pehes[:, a] - pehes[:, b])
            for a, b in combinations(range(len(LEARNER_NAMES)), 2)
        },
        "winner_by_pehe": LEARNER_NAMES[int(np.argmin(pehes.mean(axis=0)))],
        "criterion_picks_the_pehe_winner": float(
            (np.argmin(crits, axis=1) == np.argmin(pehes, axis=1)).mean()
        ),
    }
    if (sd_tau > 0).all():  # the null effect has no spread to normalize by
        normalized = pehes / sd_tau[:, None]
        summary["pehe_normalized"] = {
            name: mean_se(normalized[:, j]) for j, name in enumerate(LEARNER_NAMES)
        }
    return summary


def _load(work: Path, tag: str) -> dict[tuple[str, int], dict]:
    """The finished replications with this fingerprint.

    A last line cut short by a crash mid-write is skipped with a warning: that
    replication runs again.
    """
    done = {}
    if work.exists():
        lines = work.read_text().splitlines()
        for number, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines) - 1:
                    print(f"  skipping a truncated last line of {work.name}", flush=True)
                    continue
                raise
            if entry.pop("fingerprint") == tag:
                done[(entry["scenario"], entry["replication"])] = entry
    return done


def run_study(cfg, threads: int, work: Path) -> dict:
    """Every replication of every configuration, resuming from `work`; returns the record."""
    started = time.perf_counter()
    tag = fingerprint(cfg, threads)
    done = _load(work, tag)
    calibrated = intercepts(cfg)
    work.parent.mkdir(parents=True, exist_ok=True)
    sim = cfg.simulation
    for index, scenario in enumerate(sim.scenarios):
        intercept = calibrated[(scenario.baseline, scenario.base_rate)]
        for replication in range(sim.replications):
            if (scenario.name, replication) in done:
                continue
            run = run_replication(scenario, index, replication, intercept, cfg, threads)
            with work.open("a") as handle:
                handle.write(json.dumps({**run, "fingerprint": tag}) + "\n")
            done[(scenario.name, replication)] = run
            print(
                f"  {scenario.name} {replication + 1}/{sim.replications}: {run['seconds']} s",
                flush=True,
            )

    scenarios = {}
    for scenario in sim.scenarios:
        runs = [done[(scenario.name, r)] for r in range(sim.replications)]
        scenarios[scenario.name] = {
            "settings": asdict(scenario),
            "intercept": calibrated[(scenario.baseline, scenario.base_rate)],
            **summarize(runs),
            "runs": runs,
        }
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total = sum(run["seconds"] for s in scenarios.values() for run in s["runs"])
    return {
        "n": sim.n,
        "n_validation": sim.n_validation,
        "replications": sim.replications,
        "relative_effect": sim.relative_effect,
        "learners": LEARNER_NAMES,
        "learner_setting": asdict(cfg.learners),
        "fingerprint": tag,
        "scenarios": scenarios,
        "run": {
            **run_metadata(packages=("numpy", "lightgbm", "scipy")),
            "threads": threads,
            "replication_seconds": round(total, 1),
            "seconds_this_session": round(time.perf_counter() - started, 1),
            "peak_rss_mib": round(peak_kib / 1024),
        },
    }


def plot_pehe(record: dict, path: Path) -> None:
    """PEHE over the spread of the true effect, by configuration and learner; the null apart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, s in record["scenarios"].items() if s["pehe_normalized"] is not None]
    nulls = [n for n, s in record["scenarios"].items() if s["pehe_normalized"] is None]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [len(names), max(len(nulls), 1)]}
    )
    width = 0.8 / len(record["learners"])
    for panel, (ax, group, key, label) in enumerate(
        (
            (axes[0], names, "pehe_normalized", "PEHE / sd of the true effect"),
            (axes[1], nulls, "pehe", "PEHE (absolute)"),
        )
    ):
        for j, learner in enumerate(record["learners"]):
            means = [record["scenarios"][n][key][learner]["mean"] for n in group]
            errors = [2 * record["scenarios"][n][key][learner]["mc_se"] for n in group]
            positions = np.arange(len(group)) + (j - 1.5) * width
            ax.bar(
                positions,
                means,
                width,
                yerr=errors,
                capsize=2,
                label=learner if panel == 0 else None,
            )
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([n.replace("_", "\n") for n in group], fontsize=8)
        ax.set_ylabel(label)
        if key == "pehe_normalized":
            # Predicting the mean effect for everyone has a PEHE of exactly the sd
            # of the true effect: a learner above this line does worse than that.
            ax.axhline(1.0, color="black", linewidth=1, label="the mean effect for everyone")
    axes[0].legend(title="learner", fontsize=8)
    fig.suptitle(
        f"Simulation: n = {record['n']:,}, {record['replications']} replications, "
        "bars of 2 Monte Carlo standard errors",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)

    record = run_study(CONFIG, args.threads, args.work)
    write_json(args.out, record)
    plot_pehe(record, args.figure)
    for name, scenario in record["scenarios"].items():
        print(f"{name}: winner by PEHE {scenario['winner_by_pehe']}")
    print(f"record -> {args.out}; figure -> {args.figure}")


if __name__ == "__main__":
    main()
