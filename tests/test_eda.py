"""The minimal EDA: distinct values, the float32 collapse and duplicated vectors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import format_rows, make_columns, write_csv_gz

from uplift.data import eda
from uplift.data.eda import (
    distinct_with_counts,
    duplicate_summary,
    float32_collapse,
    row_hashes,
    run_eda,
)
from uplift.data.ingest import FEATURES, SOURCE_COLUMNS, ingest
from uplift.data.split import split


@pytest.mark.parametrize("merge_every", [5_000_000, 7])
def test_distinct_with_counts_is_unique_over_the_concatenation(monkeypatch, merge_every):
    # With a tiny merge threshold the running merge is exercised on every chunk.
    monkeypatch.setattr(eda, "MERGE_EVERY", merge_every)
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 30, size).astype(np.float64) for size in (5, 40, 1, 100)]
    values, counts = distinct_with_counts(chunks)
    expected_values, expected_counts = np.unique(np.concatenate(chunks), return_counts=True)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(counts, expected_counts)


def test_the_float32_collapse_by_hand():
    # 1 and 1 + 1e-9 are distinct in float64 and the same float32.
    values = np.array([1.0, 1.0 + 1e-9, 2.0])
    counts = np.array([3, 2, 5])
    assert float32_collapse(values, counts) == {
        "distinct_float64": 3,
        "distinct_float32": 2,
        "float64_values_merged": 2,
        "rows_with_a_merged_value": 5,
    }


def test_equal_rows_hash_equal_and_different_rows_differ():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, (100_000, 12)).astype(np.float32)
    h = row_hashes(x)
    assert np.unique(h).size == x.shape[0]  # no collision among 100 000 rows
    np.testing.assert_array_equal(row_hashes(x[[3, 3, 7]]), h[[3, 3, 7]])
    swapped = x[:5].copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    assert (row_hashes(swapped) != h[:5]).all()  # the column order counts
    zero, minus_zero = np.zeros((1, 12), np.float32), -np.zeros((1, 12), np.float32)
    assert row_hashes(zero) == row_hashes(minus_zero)  # equal values, different bits


def test_duplicate_summary_by_hand():
    hashes = {
        "train": np.array([1, 2, 2, 3], dtype=np.uint64),
        "val": np.array([2, 4], dtype=np.uint64),
        "test": np.array([5, 5, 1, 6], dtype=np.uint64),
    }
    assert duplicate_summary(hashes) == {
        "rows": 10,
        "distinct_vectors": 6,
        "rows_in_a_repeated_vector": 7,  # 1 twice, 2 three times, 5 twice
        "largest_repetition": 3,
        "val_rows_whose_vector_is_in_train": 0.5,
        "test_rows_whose_vector_is_in_train": 0.25,
    }


def test_the_eda_end_to_end_reads_no_outcome(tmp_path):
    columns = make_columns(3_000, seed=4)
    columns["f0"][:10] = 1.0
    columns["f0"][10:20] = 1.0 + 1e-9  # ten rows collapse onto the first ten
    columns["f1"][100:200] = columns["f1"][0]  # and f1 repeats, so vectors do not
    src = write_csv_gz(tmp_path / "src.csv.gz", ",".join(SOURCE_COLUMNS), format_rows(columns))
    ingest(src, tmp_path / "all.parquet")
    split(tmp_path / "all.parquet", tmp_path, SimpleNamespace(seed=0, fractions=(0.6, 0.2, 0.2)))

    summary = run_eda(src, tmp_path / "all.parquet", tmp_path, block_size=16 << 10)
    for name in FEATURES:
        values = summary["features"][name]
        # %.17g writes a float64 exactly, so the stream must match the arrays.
        expected = float32_collapse(*np.unique(columns[name], return_counts=True))
        assert {k: values[k] for k in expected} == expected
        assert values["distinct_float32"] == values["distinct_float32_in_parquet"]
    # Random normals collapse too, now and then: the injected pair is a floor.
    assert summary["features"]["f0"]["rows_with_a_merged_value"] >= 20
    duplicates = summary["duplicates"]
    assert duplicates["rows"] == 3_000 and duplicates["distinct_vectors"] == 3_000
    assert set(summary) == {"source_file", "features", "duplicates", "run"}
