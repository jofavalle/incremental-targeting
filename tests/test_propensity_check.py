"""The propensity check end to end, in two simulated worlds with a known answer.

In one, the treatment is drawn with one ratio for everybody: the check must say
"constant". In the other, the treatment ratio depends on a feature, as tests with
different ratios that the features tell apart would: it must say "weighted". Both
worlds carry an `exposure` column that gives the arm away; if the driver read it,
the first world would come out "weighted".
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uplift.config import CONFIG
from uplift.data.ingest import FEATURES
from uplift.propensity_check import main, run_check

# Small and fast, with the real threshold: the rule under test is the real one.
SMALL = replace(CONFIG.propensity, n_estimators=30, min_child_samples=50, n_bootstrap=30)


def _partition(path, n_rows, seed, propensity):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n_rows, len(FEATURES))).astype(np.float32)
    t = (rng.random(n_rows) < propensity(x)).astype(np.int8)
    exposure = (t & (rng.random(n_rows) < 0.5)).astype(np.int8)
    columns = {name: x[:, j] for j, name in enumerate(FEATURES)}
    columns.update(treatment=t, exposure=exposure)
    pq.write_table(pa.table(columns), path)
    return path


def _world(tmp_path, propensity):
    train = _partition(tmp_path / "train.parquet", 20_000, 1, propensity)
    val = _partition(tmp_path / "val.parquet", 10_000, 2, propensity)
    return train, val


def constant(x):
    return np.full(len(x), 0.85)


def by_test(x):
    # Two "tests" told apart by f0, with treatment ratios 0.96 and 0.55.
    return np.where(x[:, 0] > 0, 0.96, 0.55)


@pytest.mark.parametrize("propensity, decision", [(constant, "constant"), (by_test, "weighted")])
def test_the_check_tells_the_two_worlds_apart(tmp_path, propensity, decision):
    train, val = _world(tmp_path, propensity)
    record, e_hat, t_val = run_check(train, val, SMALL, threads=1)
    low, high = record["auc_val_ci95"]
    assert low <= record["auc_val"] <= high
    assert record["decision"] == decision
    assert e_hat.shape == t_val.shape == (10_000,)
    for arm in ("treated", "control"):
        q = list(record["e_hat_val"][arm].values())
        assert 0 <= min(q) and max(q) <= 1  # probabilities, not logits
    if decision == "weighted":
        treated, control = record["e_hat_val"]["treated"], record["e_hat_val"]["control"]
        assert treated["q0.5"] > control["q0.5"]  # the arms are not swapped
        assert record["propensity_sd_linear_bound"] > 5 * record["propensity_sd_linear_noise"]


@pytest.mark.parametrize(
    "interval, decision", [((0.505, 0.52), "constant"), ((0.5105, 0.52), "weighted")]
)
def test_the_rule_reads_the_lower_bound_not_the_point(tmp_path, monkeypatch, interval, decision):
    # The real result hinged on this: point 0.5102 above the threshold, lower bound below.
    train, val = _world(tmp_path, constant)
    monkeypatch.setattr("uplift.propensity_check.auc", lambda score, t: 0.515)
    monkeypatch.setattr("uplift.propensity_check.bootstrap_auc_interval", lambda *a, **k: interval)
    record, _, _ = run_check(train, val, SMALL, threads=1)
    assert record["decision"] == decision


def test_the_driver_writes_the_record_and_the_figure(tmp_path, monkeypatch):
    train, val = _world(tmp_path, constant)
    out, figure = tmp_path / "propensity_check.json", tmp_path / "figures" / "fig1.png"
    monkeypatch.setattr("uplift.propensity_check.CONFIG", replace(CONFIG, propensity=SMALL))
    main(["--train", str(train), "--val", str(val), "--out", str(out), "--figure", str(figure)])

    record = json.loads(out.read_text())
    assert set(record) == {
        "partitions",
        "smd_train",
        "smd_flagged",
        "smd_flag",
        "auc_val",
        "auc_val_ci95",
        "auc_threshold",
        "smd_noise_se",
        "propensity_sd_linear_bound",
        "propensity_sd_linear_noise",
        "propensity_sd_if_normal",
        "decision",
        "e_hat_val",
        "classifier",
        "n_bootstrap",
        "seed",
        "run",
    }
    assert set(record["smd_train"]) == set(FEATURES)  # features only, never exposure
    assert record["auc_threshold"] == CONFIG.propensity.auc_threshold
    assert str(tmp_path) not in out.read_text()
    assert figure.stat().st_size > 0
