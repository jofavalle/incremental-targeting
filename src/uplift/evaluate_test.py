"""The test partition, opened once: the curves of the learners and the four policies.

    python -m uplift.evaluate_test               # 2 threads, as on the smallest machine

The learners were fitted on train and one was chosen on validation
(`uplift.select_learner`, which pickled them); nothing here is chosen on test.
Every setting was fixed in `EvaluationConfig` and `DecisionConfig`, and pushed,
before the ATE was seen (docs/design.md §7.3, §9.5). It writes:

    reports/test_evaluation.json             the record, versioned
    reports/figures/fig6_uplift_curves.png   the uplift curves of the four learners
    reports/figures/fig7_budget_return.png   incremental conversions against capacity
    reports/figures/fig8_tornado.png         the money assumptions, swept

## The curves (§7)

For the primary outcome: the uplift and Qini curves of each learner, their areas
over the random line, bootstrap intervals with tau_hat held fixed, and the paired
difference of each learner's AUUC from the chosen one's.

## The policies (§9)

Scored on conversions. Policy 1 contacts nobody, policy 2 at random (K times the
ATE, in expectation), policy 3 the top K by a classifier of conversion fitted on
train, policy 4 the top K by the chosen learner's tau_hat. The difference between
policies 4 and 3 carries a paired bootstrap interval at every capacity, and the
money assumptions are swept over their declared ranges.
"""

from __future__ import annotations

import argparse
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import FEATURES
from uplift.evaluate.curves import RankedScore, area_over_random, break_even
from uplift.evaluate.effect import paired_bootstrap, percentile_interval
from uplift.features import feature_matrix
from uplift.learners import base
from uplift.runinfo import run_metadata, write_json

DEFAULT_OUT = REPORTS_DIR / "test_evaluation.json"
DEFAULT_MODELS = DATA_PROCESSED / "learners.pkl"
FIGURES = REPORTS_DIR / "figures"
POLICIES = ("nobody", "random", "propensity", "uplift")


def _load(path: Path, outcomes: list[str]):
    table = pq.read_table(path, columns=[*FEATURES, "treatment", *outcomes])
    return (
        feature_matrix(table),
        table["treatment"].to_numpy(),
        {o: table[o].to_numpy() for o in outcomes},
    )


def _interval(draws: np.ndarray, level: float) -> list[float]:
    return percentile_interval(draws.reshape(-1, 1), level)[:, 0].tolist()


def run_test(train_path: Path, test_path: Path, models_path: Path, cfg, threads: int):
    """Score everything on test once; returns the record and what the figures need."""
    started = time.perf_counter()
    pa.set_cpu_count(threads)
    with models_path.open("rb") as handle:
        saved = pickle.load(handle)
    primary, learners = saved["outcome"], saved["learners"]
    chosen = saved["chosen"]
    counted = cfg.decision.counted_outcome

    x_train, _, y_train = _load(train_path, [counted])
    print(f"fitting the classifier of {counted} for policy 3", flush=True)
    propensity_model = base.fit(x_train, y_train[counted], cfg.learners, "binary", threads)
    del x_train, y_train

    outcomes = sorted({primary, counted})
    x, t, y = _load(test_path, outcomes)
    tau_hat = {name: learner.predict(x) for name, learner in learners.items()}
    score3 = propensity_model.predict(x)
    del x
    n = t.size
    phi = np.linspace(0.0, 1.0, cfg.evaluation.curve_points)
    capacities = np.asarray(cfg.decision.capacities)
    grid = np.union1d(phi, capacities)  # the money sweeps need both
    at_capacity = np.searchsorted(grid, capacities)

    ranked = {name: RankedScore(values, t, y[primary]) for name, values in tau_hat.items()}
    decision = {
        "propensity": RankedScore(score3, t, y[counted]),
        "uplift": RankedScore(tau_hat[chosen], t, y[counted]),
    }
    names = list(ranked)

    def statistic(counts):
        areas = []
        for name in names:
            u, q = ranked[name].curves(phi, counts)
            areas += [
                area_over_random(u, phi, counts.sum()),
                area_over_random(q, phi, counts.sum()),
            ]
        values = []
        for policy in ("propensity", "uplift"):
            u, _ = decision[policy].curves(grid, counts)
            values += list(u[at_capacity])
        return areas + values + [u[-1]]  # U(N) on conversions: the random policy's slope

    rng = np.random.default_rng(cfg.evaluation.bootstrap_seed)
    draws = paired_bootstrap(statistic, n, cfg.evaluation.n_bootstrap, rng)
    level = cfg.effect.level
    k = len(capacities)

    curves = {}
    point = {}
    for i, name in enumerate(names):
        u, q = ranked[name].curves(phi)
        point[name] = u
        curves[name] = {
            "auuc": area_over_random(u, phi, n),
            "auuc_ci": _interval(draws[:, 2 * i], level),
            "qini": area_over_random(q, phi, n),
            "qini_ci": _interval(draws[:, 2 * i + 1], level),
            "uplift_curve": u.tolist(),
        }
    for i, name in enumerate(names):
        if name != chosen:
            j = names.index(chosen)
            curves[name]["auuc_minus_chosen"] = curves[name]["auuc"] - curves[chosen]["auuc"]
            curves[name]["auuc_minus_chosen_ci"] = _interval(
                draws[:, 2 * i] - draws[:, 2 * j], level
            )

    offset = 2 * len(names)
    u3, _ = decision["propensity"].curves(grid)
    u4, _ = decision["uplift"].curves(grid)
    ate_total = u4[-1]  # U(N): the same for any ranking
    policy_values = {
        "nobody": np.zeros(k),
        "random": capacities * ate_total,
        "propensity": u3[at_capacity],
        "uplift": u4[at_capacity],
    }
    per_contact = 1000 / (capacities * n * cfg.decision.contact_cost)
    difference = draws[:, offset + k : offset + 2 * k] - draws[:, offset : offset + k]
    policies = {
        policy: {
            "incremental": values.tolist(),
            "per_1000_contacts": (values * per_contact).tolist(),
        }
        for policy, values in policy_values.items()
    }
    for j, policy in enumerate(("propensity", "uplift")):
        block = draws[:, offset + j * k : offset + (j + 1) * k]
        policies[policy]["incremental_ci"] = percentile_interval(block, level).T.tolist()
    policies["uplift_minus_propensity"] = {
        "incremental": (policy_values["uplift"] - policy_values["propensity"]).tolist(),
        "incremental_ci": percentile_interval(difference, level).T.tolist(),
    }

    money = []
    for value in cfg.decision.conversion_values:
        for cost in cfg.decision.contact_costs:
            row = {"conversion_value": value, "contact_cost": cost}
            for policy, u in (("propensity", u3), ("uplift", u4)):
                net = value * u - cost * grid * n
                best = int(np.argmax(net))
                row[policy] = {
                    "best_capacity": float(grid[best]),
                    "best_net_value_per_1000_units": float(1000 * net[best] / n),
                    "break_even_capacity": break_even(u, grid, n, cost / value),
                }
            row["uplift_minus_propensity_per_1000_units"] = (
                row["uplift"]["best_net_value_per_1000_units"]
                - row["propensity"]["best_net_value_per_1000_units"]
            )
            money.append(row)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "partition": "test",
        "rows": int(n),
        "treated": int(t.sum()),
        "primary_outcome": primary,
        "counted_outcome": counted,
        "chosen_learner": chosen,
        "policy_4_ranks_by": f"uplift on {primary}",
        "phi": phi.tolist(),
        "curves": curves,
        "capacities": capacities.tolist(),
        "policies": policies,
        "money": money,
        "n_bootstrap": cfg.evaluation.n_bootstrap,
        "seed": cfg.evaluation.bootstrap_seed,
        "level": level,
        "run": {
            **run_metadata(packages=("numpy", "pyarrow", "lightgbm", "scipy")),
            "threads": threads,
            "seconds": round(time.perf_counter() - started, 1),
            "peak_rss_mib": round(peak_kib / 1024),
        },
    }
    return record, point


def plot_figures(record: dict, figures: Path) -> None:
    """The uplift curves, the budget-return curve and the tornado of the money sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    phi = np.asarray(record["phi"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in record["curves"].items():
        tag = " (chosen)" if name == record["chosen_learner"] else ""
        ax.plot(phi, curve["uplift_curve"], label=f"{name}-learner{tag}, AUUC {curve['auuc']:.5f}")
    end = record["curves"][record["chosen_learner"]]["uplift_curve"][-1]
    ax.plot([0, 1], [0, end], color="black", linestyle="--", linewidth=1, label="random")
    ax.set_xlabel("fraction of the test partition contacted, by tau_hat")
    ax.set_ylabel(f"incremental {record['primary_outcome']}s, U(n)")
    ax.set_title("Uplift curves on the test partition, opened once", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig6_uplift_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    capacities = np.asarray(record["capacities"])
    for policy in POLICIES:
        values = np.asarray(record["policies"][policy]["incremental"])
        ax.plot(capacities, values, marker="o", label=policy)
        if "incremental_ci" in record["policies"][policy]:
            ci = np.asarray(record["policies"][policy]["incremental_ci"])
            ax.fill_between(capacities, ci[:, 0], ci[:, 1], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("capacity K, share of the test partition (log scale)")
    ax.set_ylabel(f"incremental {record['counted_outcome']}s")
    ax.set_title("Budget-return curve: the whole grid of K", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig7_budget_return.png", dpi=150)
    plt.close(fig)

    centre = {(r["conversion_value"], r["contact_cost"]): r for r in record["money"]}
    values = sorted({r["conversion_value"] for r in record["money"]})
    costs = sorted({r["contact_cost"] for r in record["money"]})
    key = "uplift_minus_propensity_per_1000_units"
    middle_value, middle_cost = values[len(values) // 2], costs[len(costs) // 2]
    rows = [
        ("value of a conversion", [centre[(v, middle_cost)][key] for v in values]),
        ("cost of a contact", [centre[(middle_value, c)][key] for c in costs]),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    base_value = centre[(middle_value, middle_cost)][key]
    for i, (label, sweep) in enumerate(rows):
        ax.barh(i, max(sweep) - min(sweep), left=min(sweep), height=0.5)
        ax.text(max(sweep), i, f"  {min(sweep):.2f} to {max(sweep):.2f}", va="center", fontsize=8)
    ax.axvline(base_value, color="black", linewidth=1)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.set_xlabel("best net value of policy 4 minus policy 3, per 1,000 units")
    ax.set_title("Sensitivity to the money assumptions", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "fig8_tornado.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.parquet")
    parser.add_argument("--test", type=Path, default=DATA_PROCESSED / "test.parquet")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    record, _ = run_test(args.train, args.test, args.models, CONFIG, args.threads)
    write_json(args.out, record)
    plot_figures(record, args.figures)
    print(f"record -> {args.out}; figures -> {args.figures}")


if __name__ == "__main__":
    main()
