"""Selection on validation: the criterion picks the better estimate, end to end."""

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
from uplift.evaluate.selection import criterion
from uplift.select_learner import main, select
from uplift.simulate.dgp import baseline_logit, calibrate_intercept, calibration_sample, generate

SMALL = replace(
    CONFIG,
    learners=replace(CONFIG.learners, n_estimators=30, min_child_samples=100),
    selection=replace(CONFIG.selection, n_bootstrap=50),
)


def test_weighted_criterion_is_the_criterion_of_repeated_rows():
    rng = np.random.default_rng(0)
    tau_hat, z, counts = rng.normal(size=50), rng.normal(size=50), rng.integers(0, 4, 50)
    rows = np.repeat(np.arange(50), counts)
    assert criterion(tau_hat, z, weights=counts) == pytest.approx(criterion(tau_hat[rows], z[rows]))


def test_select_prefers_the_estimate_closer_to_the_truth():
    rng = np.random.default_rng(1)
    n = 400_000
    tau = rng.uniform(0.0, 0.2, n)
    t = (rng.random(n) < 0.85).astype(np.int8)
    y = (rng.random(n) < 0.2 + tau * t).astype(np.int8)
    tau_hat = {"good": tau + rng.normal(0, 0.01, n), "bad": tau + rng.normal(0, 0.2, n)}
    result = select(tau_hat, y, t, SMALL, np.random.default_rng(2))
    assert result["chosen"] == "good"
    assert result["difference_from_chosen"]["bad"]["separated"]
    assert result["supported_by_every_interval"]


def _partitions(tmp_path):
    scenario = replace(CONFIG.simulation.scenarios[0], base_rate=0.2)
    x_cal = calibration_sample(50_000, seed=0)
    intercept = calibrate_intercept(baseline_logit(x_cal, "simple"), 0.2)
    paths = {}
    for seed, name, n in ((3, "train", 30_000), (4, "val", 15_000)):
        data = generate(scenario, n, np.random.default_rng(seed), intercept, 0.25)
        columns = {f: data.x[:, j] for j, f in enumerate(FEATURES)}
        columns.update(treatment=data.t, visit=data.y, conversion=data.y, exposure=data.t)
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table(columns), paths[name])
    paths["power"] = tmp_path / "power.json"
    paths["power"].write_text(json.dumps({"decision": {"primary": "visit"}}))
    return paths


def test_the_driver_selects_checks_and_keeps_the_learners(tmp_path, monkeypatch):
    paths = _partitions(tmp_path)
    out, figure, models = tmp_path / "selection.json", tmp_path / "fig5.png", tmp_path / "l.pkl"
    monkeypatch.setattr("uplift.select_learner.CONFIG", SMALL)
    main(
        ["--train", str(paths["train"]), "--val", str(paths["val"])]
        + ["--power", str(paths["power"]), "--out", str(out), "--figure", str(figure)]
        + ["--models", str(models), "--threads", "1"]
    )
    record = json.loads(out.read_text())
    assert record["outcome"] == "visit" and record["partition"] == "val"
    assert record["chosen"] in {"S", "T", "X", "DR"}
    assert set(record["difference_from_chosen"]) == {"S", "T", "X", "DR"} - {record["chosen"]}
    for check in record["checks"].values():
        assert len(check["calibration"]) <= 10
        assert sum(row["rows"] for row in check["calibration"]) == 15_000
    with models.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["outcome"] == "visit" and set(saved["learners"]) == {"S", "T", "X", "DR"}
    assert saved["chosen"] == record["chosen"]
    assert str(tmp_path) not in out.read_text()
    assert figure.stat().st_size > 0
