"""The power analysis, which decides the primary outcome (docs/design.md §8).

    python -m uplift.power_analysis

The rule was fixed in `PowerConfig` and `design.md` §8.3, and pushed, while the
ATE it compares against was still running. It writes:

    reports/power.json                       the record, versioned
    reports/figures/fig3_mde_by_subgroup.png the MDE of a contrast against subgroup size

## What it reads

The arm sizes of the test partition, from `reports/split_summary.json`: counts,
never an outcome, since the test outcomes are looked at once, at the end. The
rates and the ATE measured on train, from `reports/ate.json`; the partitions
share them by construction (§4.2).

## The decision

An outcome supports heterogeneity at a share s if the MDE of a contrast between
a subgroup of that share and the rest is at most its measured ATE. `visit` stays
primary unless `conversion` supports it at the decile level (§8.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uplift.config import CONFIG, REPORTS_DIR
from uplift.evaluate.power import full_sample_variance, mde, subgroups_of_share
from uplift.runinfo import run_metadata, write_json

DEFAULT_ATE = REPORTS_DIR / "ate.json"
DEFAULT_SPLIT = REPORTS_DIR / "split_summary.json"
DEFAULT_OUT = REPORTS_DIR / "power.json"
DEFAULT_FIGURE = REPORTS_DIR / "figures" / "fig3_mde_by_subgroup.png"

OUTCOMES = ("visit", "conversion")


def _test_arms(split_summary: dict) -> tuple[int, int]:
    test = split_summary["partitions"]["test"]
    n1 = round(test["rows"] * test["treatment_share"])
    return n1, test["rows"] - n1


def _row(variance: float, share: float, ate: float, n0: int, cfg) -> dict:
    m = subgroups_of_share(share)
    value = mde(variance, share, cfg.alpha, cfg.power)
    return {
        "share": share,
        "subgroups": m,
        "alpha": cfg.alpha / m,
        "controls_in_subgroup": round(share * n0),
        "mde": value,
        "mde_over_ate": value / abs(ate) if ate else None,  # no effect: no benchmark
        "supports": value <= abs(ate),
    }


def run_power(ate_record: dict, split_summary: dict, cfg) -> dict:
    """Apply §8.3 to the measured rates and the test sizes; `cfg` is like `PowerConfig`."""
    n1, n0 = _test_arms(split_summary)
    outcomes = {}
    for outcome in OUTCOMES:
        primary = ate_record["partitions"]["train"]["outcomes"][outcome]["primary"]
        variance = full_sample_variance(primary["p1"], primary["p0"], n1, n0)
        ate = primary["estimate"]
        outcomes[outcome] = {
            "p1": primary["p1"],
            "p0": primary["p0"],
            "ate": ate,
            "subgroups": [_row(variance, s, ate, n0, cfg) for s in cfg.subgroup_shares],
            "curve": [_row(variance, s, ate, n0, cfg) for s in cfg.curve_shares],
        }

    decisive = next(
        row for row in outcomes["conversion"]["subgroups"] if row["share"] == cfg.decision_share
    )
    return {
        "sample": {"partition": "test", "treated": n1, "controls": n0},
        "rates_from": "train, reports/ate.json",
        "alpha": cfg.alpha,
        "power": cfg.power,
        "correction": "Bonferroni over the 1 / share subgroups of a size",
        "outcomes": outcomes,
        "decision": {
            "primary": "conversion" if decisive["supports"] else "visit",
            "share": cfg.decision_share,
            "rule": (
                "visit stays primary unless conversion supports heterogeneity at share "
                f"{cfg.decision_share}: the MDE of the contrast at most its measured ATE"
            ),
            "conversion_mde_over_ate_at_decision_share": decisive["mde_over_ate"],
        },
        "run": run_metadata(packages=("numpy", "scipy")),
    }


def plot_mde(record: dict, path: Path) -> None:
    """MDE of a contrast over the measured ATE, against subgroup share, both log scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for outcome in OUTCOMES:
        curve = record["outcomes"][outcome]["curve"]
        ax.plot(
            [row["share"] for row in curve],
            [row["mde_over_ate"] for row in curve],
            marker="o",
            label=f"{outcome} (ATE {100 * record['outcomes'][outcome]['ate']:.3f} points)",
        )
    ax.axhline(1.0, color="black", linewidth=1, label="MDE equal to the measured ATE")
    share = record["decision"]["share"]
    ax.axvline(share, color="grey", linestyle=":", linewidth=1, label=f"decision share {share}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("subgroup share of the test partition (Bonferroni over 1 / share subgroups)")
    ax.set_ylabel("MDE of subgroup against rest / measured ATE")
    level = record["alpha"]
    ax.set_title(
        f"Which heterogeneity is detectable? alpha {level}, power {record['power']}", fontsize=10
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ate", type=Path, default=DEFAULT_ATE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)

    record = run_power(
        json.loads(args.ate.read_text()), json.loads(args.split.read_text()), CONFIG.power
    )
    write_json(args.out, record)
    plot_mde(record, args.figure)
    print(f"primary outcome: {record['decision']['primary']}; record -> {args.out}")


if __name__ == "__main__":
    main()
