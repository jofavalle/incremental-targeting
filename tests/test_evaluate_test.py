"""The test evaluation end to end, in a simulated world where uplift targeting should win."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uplift.config import CONFIG
from uplift.data.ingest import FEATURES
from uplift.evaluate_test import POLICIES, main, run_test
from uplift.select_learner import fit_learners
from uplift.simulate.dgp import baseline_logit, calibrate_intercept, calibration_sample, generate

SMALL = replace(
    CONFIG,
    learners=replace(CONFIG.learners, n_estimators=30, min_child_samples=100),
    evaluation=replace(CONFIG.evaluation, n_bootstrap=30),
)


def _world(tmp_path):
    # A strong heterogeneous effect on a common outcome, the same flag as visit
    # and conversion, so that policy 4 has something to find.
    scenario = replace(CONFIG.simulation.scenarios[0], base_rate=0.2)
    intercept = calibrate_intercept(baseline_logit(calibration_sample(50_000, 0), "simple"), 0.2)
    paths = {}
    for seed, name, n in ((1, "train", 30_000), (2, "test", 20_000)):
        data = generate(scenario, n, np.random.default_rng(seed), intercept, 0.5)
        columns = {f: data.x[:, j] for j, f in enumerate(FEATURES)}
        columns.update(treatment=data.t, visit=data.y, conversion=data.y, exposure=data.t)
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table(columns), paths[name])
        if name == "train":
            learners = fit_learners(data.x, data.t, data.y, SMALL, threads=1)
    paths["models"] = tmp_path / "learners.pkl"
    with paths["models"].open("wb") as handle:
        pickle.dump({"outcome": "visit", "chosen": "T", "learners": learners}, handle)
    return paths


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    paths = _world(tmp_path_factory.mktemp("test_eval"))
    record, _ = run_test(paths["train"], paths["test"], paths["models"], SMALL, threads=1)
    return record, paths


def test_the_curves_rank_above_random(evaluated):
    record, _ = evaluated
    assert set(record["curves"]) == {"S", "T", "X", "DR"}
    for curve in record["curves"].values():
        low, high = curve["auuc_ci"]
        assert low <= curve["auuc"] <= high
        assert curve["auuc"] > 0
    assert "auuc_minus_chosen" not in record["curves"]["T"]
    assert "auuc_minus_chosen_ci" in record["curves"]["S"]


def test_the_policies_follow_section_9(evaluated):
    record, _ = evaluated
    policies = record["policies"]
    capacities = np.asarray(record["capacities"])
    assert set(POLICIES) <= set(policies)
    assert policies["nobody"]["incremental"] == [0.0] * capacities.size
    random = np.asarray(policies["random"]["incremental"])
    np.testing.assert_allclose(random / capacities, random[-1])  # K times the ATE
    uplift = np.asarray(policies["uplift"]["incremental"])
    assert uplift[-1] == pytest.approx(random[-1])  # contacting everybody: the same
    # Where the effect is heterogeneous, ranking by uplift beats random at small K.
    assert uplift[3] > random[3]


def test_the_money_sweep_covers_the_declared_grid(evaluated):
    record, _ = evaluated
    combos = {(r["conversion_value"], r["contact_cost"]) for r in record["money"]}
    expected = {
        (v, c) for v in CONFIG.decision.conversion_values for c in CONFIG.decision.contact_costs
    }
    assert combos == expected
    for row in record["money"]:
        for policy in ("propensity", "uplift"):
            assert 0.0 <= row[policy]["break_even_capacity"] <= 1.0


def test_the_driver_writes_the_record_and_three_figures(evaluated, tmp_path, monkeypatch):
    _, paths = evaluated
    monkeypatch.setattr("uplift.evaluate_test.CONFIG", SMALL)
    out = tmp_path / "test_evaluation.json"
    main(
        ["--train", str(paths["train"]), "--test", str(paths["test"]), "--models"]
        + [str(paths["models"]), "--out", str(out), "--figures", str(tmp_path), "--threads", "1"]
    )
    record = json.loads(out.read_text())
    assert record["partition"] == "test" and record["chosen_learner"] == "T"
    assert str(tmp_path) not in out.read_text()
    for name in ("fig6_uplift_curves.png", "fig7_budget_return.png", "fig8_tornado.png"):
        assert (tmp_path / name).stat().st_size > 0
