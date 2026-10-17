"""The stratified split: ratios per partition, exact cuts, determinism and the streamed files."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import format_rows, make_columns, write_csv_gz

from uplift.config import CONFIG
from uplift.data.ingest import ROW_ID, SOURCE_COLUMNS, ingest
from uplift.data.split import PARTITIONS, STRATA, assign_partitions, split

FRACTIONS = (0.6, 0.2, 0.2)


def _cfg(seed: int) -> SimpleNamespace:
    # Duck-typed like the real SplitConfig: only `seed` and `fractions` matter.
    return SimpleNamespace(seed=seed, fractions=FRACTIONS)


def _strata(n_rows: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n_rows) < 0.85).astype(np.int8)
    visit = (rng.random(n_rows) < 0.05).astype(np.int8)
    conversion = (visit & (rng.random(n_rows) < 0.06)).astype(np.int8)
    return [treatment, visit, conversion]


def test_split_preserves_treatment_ratio():
    columns = _strata(100_000)
    treatment = columns[0]
    labels = assign_partitions(columns, FRACTIONS, seed=1)
    for k, fraction in enumerate(FRACTIONS):
        in_partition = labels == k
        # Exact cuts per stratum: the treated count is off by at most one row
        # per stratum, never by sampling noise.
        assert abs(treatment[in_partition].sum() - fraction * treatment.sum()) <= 2 ** len(STRATA)
        assert in_partition.sum() == pytest.approx(fraction * labels.size, abs=2 ** len(STRATA))


def test_every_stratum_is_cut_at_the_fractions():
    columns = _strata(50_000, seed=2)
    labels = assign_partitions(columns, FRACTIONS, seed=3)
    code = columns[0] * 4 + columns[1] * 2 + columns[2]
    for stratum in np.unique(code):
        in_stratum = code == stratum
        counts = np.bincount(labels[in_stratum], minlength=3)
        for k, fraction in enumerate(FRACTIONS):
            assert abs(counts[k] - fraction * in_stratum.sum()) <= 1


def test_labels_cover_every_row_once():
    labels = assign_partitions(_strata(10_000), FRACTIONS, seed=0)
    assert labels.dtype == np.int8
    assert (labels >= 0).all()
    assert set(np.unique(labels)) == {0, 1, 2}


def test_same_seed_same_partition_other_seed_other_partition():
    columns = _strata(10_000)
    first = assign_partitions(columns, FRACTIONS, seed=7)
    np.testing.assert_array_equal(first, assign_partitions(columns, FRACTIONS, seed=7))
    assert (first != assign_partitions(columns, FRACTIONS, seed=8)).mean() > 0.3


@pytest.mark.parametrize("fractions", [(0.6, 0.4), (0.6, 0.3, 0.2), (0.8, 0.2, 0.0)])
def test_rejects_invalid_fractions(fractions):
    with pytest.raises(ValueError, match="fraction"):
        assign_partitions(_strata(100), fractions, seed=0)


@pytest.mark.parametrize("seed", [None, 1.5, True, "7"])
def test_rejects_a_seed_that_is_not_an_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        assign_partitions(_strata(100), FRACTIONS, seed=seed)


def test_rejects_non_binary_strata():
    columns = _strata(100)
    columns[1] = columns[1] * 2
    with pytest.raises(ValueError, match="binary"):
        assign_partitions(columns, FRACTIONS, seed=0)


@pytest.fixture(scope="module")
def ingested_parquet(tmp_path_factory, synthetic_csv_gz):
    dst = tmp_path_factory.mktemp("processed") / "criteo_uplift.parquet"
    ingest(synthetic_csv_gz, dst, rows_per_group=3000, block_size=64 << 10)
    return dst


def _row_ids(path):
    return pq.read_table(path, columns=[ROW_ID])[ROW_ID].to_numpy()


def test_streamed_files_match_the_assignment(tmp_path, ingested_parquet):
    summary = split(ingested_parquet, tmp_path, _cfg(11))
    source = pq.read_table(ingested_parquet)
    labels = assign_partitions([source[name].to_numpy() for name in STRATA], FRACTIONS, seed=11)

    ids = {name: _row_ids(tmp_path / f"{name}.parquet") for name in PARTITIONS}
    for k, name in enumerate(PARTITIONS):
        np.testing.assert_array_equal(ids[name], source[ROW_ID].to_numpy()[labels == k])
        assert pq.ParquetFile(tmp_path / f"{name}.parquet").schema_arrow == source.schema
        assert summary["partitions"][name]["rows"] == int((labels == k).sum())
    assert set(summary) == {"source_file", "seed", "fractions", "strata", "partitions", "run"}
    for name in PARTITIONS:  # sizes and treatment share only: no outcome by arm
        assert set(summary["partitions"][name]) == {"rows", "treatment_share"}
    every_id = np.concatenate(list(ids.values()))
    assert np.unique(every_id).size == every_id.size == source.num_rows

    treatment = source["treatment"].to_numpy()
    for name in PARTITIONS:
        share = summary["partitions"][name]["treatment_share"]
        assert share == pytest.approx(treatment.mean(), abs=0.01)


def test_partition_does_not_depend_on_row_group_size(tmp_path, synthetic_csv_gz):
    ids = []
    for rows_per_group in (1000, 7000):
        work = tmp_path / str(rows_per_group)
        ingest(synthetic_csv_gz, work / "all.parquet", rows_per_group=rows_per_group)
        split(work / "all.parquet", work, _cfg(5))
        ids.append(_row_ids(work / "test.parquet"))
    np.testing.assert_array_equal(ids[0], ids[1])


def test_leaves_nothing_behind_on_a_failure_mid_stream(tmp_path, ingested_parquet, monkeypatch):
    original = pq.ParquetFile.read_row_group

    def fail_on_the_second_group(self, group, *args, **kwargs):
        if group == 1:
            raise OSError("disk went away")
        return original(self, group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", fail_on_the_second_group)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="disk went away"):
        split(ingested_parquet, out, _cfg(0))
    assert list(out.iterdir()) == []


def test_refuses_to_write_an_empty_partition(tmp_path):
    src = write_csv_gz(
        tmp_path / "one.csv.gz", ",".join(SOURCE_COLUMNS), format_rows(make_columns(1))
    )
    ingest(src, tmp_path / "one.parquet")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="empty"):
        split(tmp_path / "one.parquet", out, _cfg(0))
    assert not out.exists()


def test_the_real_config_is_a_valid_split():
    labels = assign_partitions(_strata(10_000), CONFIG.split.fractions, CONFIG.split.seed)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_a_config_without_a_seed_is_refused(tmp_path, ingested_parquet):
    with pytest.raises(AttributeError, match="seed"):
        split(ingested_parquet, tmp_path, SimpleNamespace(fractions=FRACTIONS))
