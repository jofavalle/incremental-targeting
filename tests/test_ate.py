"""The ATE driver end to end, in two simulated source files with a known effect.

Both files are sorted by treatment in runs, as the real one is. In the first the
base rate is the same everywhere, so every version must find the effect. In the
second the two blocks of runs are tests with different treatment ratios and base
rates, which the features cannot tell apart: the difference of means is biased,
and only the block version, which reads the file order, recovers the effect.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uplift.ate import OUTCOMES, VERSIONS, cross_fit, main, run_ate
from uplift.config import CONFIG
from uplift.data.ingest import FEATURES, ROW_ID
from uplift.data.split import assign_partitions
from uplift.evaluate.balance import auc

EFFECT = 0.01

# Two pairs of runs and a last treated run, which joins the second pair: blocks
# with treatment ratios 0.77 and 0.88.
RUNS = [[1, 60_000], [0, 18_000], [1, 60_000], [0, 12_000], [1, 30_000]]

# Small and fast, with the real rules: strata, folds stratification, level and seeds.
SMALL = replace(
    CONFIG,
    propensity=replace(CONFIG.propensity, n_estimators=20, min_child_samples=50),
    effect=replace(CONFIG.effect, n_folds=3, n_bootstrap=40),
)


def _source(tmp_path, base_by_block):
    """Train and validation of a sorted file, and its data summary; returns the paths."""
    rng = np.random.default_rng(0)
    t = np.concatenate([np.full(length, value, dtype=np.int8) for value, length in RUNS])
    n_rows = t.size
    run = np.repeat(np.arange(len(RUNS)), [length for _, length in RUNS])
    block = np.minimum(run // 2, 1)
    base = np.asarray(base_by_block)[block]
    visit = (rng.random(n_rows) < base + EFFECT * t).astype(np.int8)
    conversion = (rng.random(n_rows) < base / 10 + EFFECT / 10 * t).astype(np.int8)
    columns = {name: rng.normal(0.0, 1.0, n_rows).astype(np.float32) for name in FEATURES}
    columns.update(
        treatment=t,
        visit=visit,
        conversion=conversion,
        exposure=(t & (rng.random(n_rows) < 0.5)).astype(np.int8),
        **{ROW_ID: np.arange(n_rows, dtype=np.uint32)},
    )
    table = pa.table(columns)
    labels = assign_partitions([t, visit, conversion], (0.6, 0.2, 0.2), seed=1)
    paths = {}
    for k, name in enumerate(("train", "val")):
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(table.filter(pa.array(labels == k)), paths[name])
    paths["summary"] = tmp_path / "data_summary.json"
    paths["summary"].write_text(json.dumps({"source_order": {"treatment_run_lengths": RUNS}}))
    return paths


@pytest.fixture(scope="module")
def homogeneous(tmp_path_factory):
    paths = _source(tmp_path_factory.mktemp("homogeneous"), base_by_block=(0.05, 0.05))
    return run_ate(paths["train"], paths["val"], paths["summary"], SMALL, threads=1)


@pytest.fixture(scope="module")
def pooled_tests(tmp_path_factory):
    paths = _source(tmp_path_factory.mktemp("pooled"), base_by_block=(0.02, 0.10))
    return run_ate(paths["train"], paths["val"], paths["summary"], SMALL, threads=1)


def test_every_version_finds_a_homogeneous_effect(homogeneous):
    for partition in ("train", "val"):
        entry = homogeneous["partitions"][partition]["outcomes"]["visit"]
        for version in VERSIONS:
            se = _se(entry[version]["ci"])
            assert entry[version]["estimate"] == pytest.approx(EFFECT, abs=4 * se)


def test_the_cross_fitted_propensity_is_out_of_fold():
    # A model that memorizes noise scores the rows it saw almost perfectly; out of
    # fold it cannot, so its AUC stays near 0.5. In-sample e_hat would push the
    # weighted version towards the observed treatment.
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, (4_000, len(FEATURES))).astype(np.float32)
    t = (rng.random(4_000) < 0.85).astype(np.int8)
    folds = np.arange(4_000) % 4
    params = {"objective": "binary", "num_leaves": 63, "min_data_in_leaf": 1, "verbose": -1}
    params.update(num_threads=1, seed=0, deterministic=True, force_row_wise=True)
    e_hat, boosters = cross_fit(x, t, folds, params, rounds=200)
    assert len(boosters) == 4
    assert auc(e_hat, t) < 0.56
    assert auc(boosters[0].predict(x[folds != 0]), t[folds != 0]) > 0.9  # it did memorize


def test_the_normal_and_the_bootstrap_intervals_agree(homogeneous):
    entry = homogeneous["partitions"]["train"]["outcomes"]["visit"]["primary"]
    normal = entry["ci"][1] - entry["ci"][0]
    bootstrap = entry["ci_bootstrap"][1] - entry["ci_bootstrap"][0]
    assert bootstrap == pytest.approx(normal, rel=0.35)  # 40 resamples: a rough check


def _se(interval):
    return (interval[1] - interval[0]) / (2 * 1.96)


def test_only_the_blocks_see_tests_the_features_cannot_tell_apart(pooled_tests):
    entry = pooled_tests["partitions"]["train"]["outcomes"]["visit"]
    primary, blocks = entry["primary"]["estimate"], entry["blocks"]["estimate"]
    assert primary - EFFECT > 6 * _se(entry["primary"]["ci"])  # pooling biases it
    assert blocks == pytest.approx(EFFECT, abs=4 * _se(entry["blocks"]["ci"]))
    low, high = entry["blocks"]["difference_ci"]
    assert high < 0  # and the paired interval says the gap is not noise
    # The features carry no trace of the tests, so weighting cannot correct it.
    assert abs(entry["weighted"]["difference"]) < abs(entry["blocks"]["difference"]) / 4


def test_the_record_describes_its_strata_blocks_and_settings(homogeneous):
    assert homogeneous["n_folds"] == 3 and homogeneous["n_bootstrap"] == 40
    assert homogeneous["seeds"] == {
        "folds": CONFIG.effect.fold_seed,
        "bootstrap": CONFIG.effect.bootstrap_seed,
    }
    for partition in ("train", "val"):
        record = homogeneous["partitions"][partition]
        assert sum(s["rows"] for s in record["strata"]) == record["rows"]
        assert len(record["strata"]) <= len(CONFIG.effect.stratum_quantiles) + 1
        assert [b["rows"] for b in record["blocks"]] and len(record["blocks"]) == 2
        assert sum(b["treated"] for b in record["blocks"]) == record["treated"]
        for outcome in OUTCOMES:
            entry = record["outcomes"][outcome]
            assert set(entry) == set(VERSIONS)
            assert "unconfirmed" in entry["blocks"]["condition"]
            for version in VERSIONS[1:]:
                low, high = entry[version]["ci"]
                assert low <= entry[version]["estimate"] <= high
    assert homogeneous["partitions"]["train"]["e_hat_source"] == "out of fold"
    assert homogeneous["partitions"]["val"]["e_hat_source"] == "mean of the fold models"


def test_the_driver_writes_the_record_and_the_figure(tmp_path, monkeypatch):
    paths = _source(tmp_path, base_by_block=(0.05, 0.05))
    out, figure = tmp_path / "ate.json", tmp_path / "figures" / "fig2.png"
    monkeypatch.setattr("uplift.ate.CONFIG", SMALL)
    main(
        ["--train", str(paths["train"]), "--val", str(paths["val"]), "--summary"]
        + [str(paths["summary"]), "--out", str(out), "--figure", str(figure), "--threads", "1"]
    )
    text = out.read_text()
    assert str(tmp_path) not in text  # a versioned record never carries a local path
    assert set(json.loads(text)["partitions"]) == {"train", "val"}  # never test
    assert figure.stat().st_size > 0
