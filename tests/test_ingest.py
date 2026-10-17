"""The streaming ingest: types, order, refusals, and what it writes and does not write."""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import numpy as np
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest
from conftest import N_ROWS, format_rows, make_columns, write_csv_gz

from uplift.data.ingest import (
    BINARY,
    FEATURES,
    MAX_RECORDED_RUNS,
    ROW_ID,
    SCHEMA,
    SOURCE_COLUMNS,
    ingest,
)
from uplift.runinfo import write_json

HEADER = ",".join(SOURCE_COLUMNS)
ROWS_PER_GROUP = 4096
BLOCK_SIZE = 64 << 10  # small, so the suite goes through many batches and groups


@pytest.fixture(scope="module")
def ingested(tmp_path_factory, synthetic_csv_gz):
    dst = tmp_path_factory.mktemp("processed") / "criteo_uplift.parquet"
    summary = ingest(synthetic_csv_gz, dst, rows_per_group=ROWS_PER_GROUP, block_size=BLOCK_SIZE)
    return dst, summary


def test_schema_types_and_row_groups(ingested):
    dst, summary = ingested
    parquet = pq.ParquetFile(dst)
    assert parquet.schema_arrow == SCHEMA
    assert parquet.metadata.num_rows == N_ROWS
    expected_groups = math.ceil(N_ROWS / ROWS_PER_GROUP)
    assert parquet.metadata.num_row_groups == expected_groups == summary["parquet"]["row_groups"]
    sizes = [parquet.metadata.row_group(g).num_rows for g in range(expected_groups)]
    assert sizes[:-1] == [ROWS_PER_GROUP] * (expected_groups - 1)


def test_row_id_follows_the_source_order(ingested):
    dst, _ = ingested
    row_id = pq.read_table(dst, columns=[ROW_ID])[ROW_ID].to_numpy()
    np.testing.assert_array_equal(row_id, np.arange(N_ROWS, dtype=np.uint32))


def test_values_round_trip(ingested, synthetic_columns):
    dst, _ = ingested
    table = pq.read_table(dst)
    for name in FEATURES:
        expected = synthetic_columns[name].astype(np.float32)
        np.testing.assert_array_equal(table[name].to_numpy(), expected)
    for name in BINARY:
        np.testing.assert_array_equal(table[name].to_numpy(), synthetic_columns[name])


def test_summary_counts(ingested, synthetic_columns):
    _, summary = ingested
    cols = synthetic_columns
    assert summary["rows"] == N_ROWS
    for name in BINARY:
        assert summary["counts"][name] == int(cols[name].sum())
        assert summary["rates"][name] == pytest.approx(cols[name].mean())
    consistency = summary["consistency"]
    exposed_controls = int(((cols["exposure"] == 1) & (cols["treatment"] == 0)).sum())
    converted_without_visit = int(((cols["conversion"] == 1) & (cols["visit"] == 0)).sum())
    assert consistency["exposure_without_treatment"] == exposed_controls
    assert consistency["conversion_without_visit"] == converted_without_visit
    # The synthetic treatment is i.i.d., so it has thousands of runs: counted, not listed.
    expected_runs = 1 + int(np.count_nonzero(np.diff(cols["treatment"])))
    assert expected_runs > MAX_RECORDED_RUNS
    assert summary["source_order"] == {
        "treatment_runs": expected_runs,
        "treatment_run_lengths": None,
    }


def test_summary_keys_hold_no_outcome_by_arm(ingested):
    _, summary = ingested
    assert set(summary) == {
        "source_file",
        "rows",
        "counts",
        "rates",
        "consistency",
        "parquet",
        "source_order",
        "run",
    }
    assert set(summary["counts"]) == set(summary["rates"]) == set(BINARY)


def test_treatment_runs_are_merged_across_batches(tmp_path):
    columns = make_columns(3000, seed=4)
    pattern = np.repeat([1, 0, 1, 0], [1700, 400, 800, 100]).astype(np.int8)
    columns["treatment"] = pattern
    columns["exposure"] = columns["exposure"] & pattern
    src = write_csv_gz(tmp_path / "runs.csv.gz", HEADER, format_rows(columns))
    summary = ingest(src, tmp_path / "out.parquet", rows_per_group=512, block_size=4 << 10)
    assert summary["source_order"] == {
        "treatment_runs": 4,
        "treatment_run_lengths": [[1, 1700], [0, 400], [1, 800], [0, 100]],
    }


def test_summary_records_no_path(ingested, synthetic_csv_gz, tmp_path):
    dst, summary = ingested
    out = tmp_path / "data_summary.json"
    write_json(out, summary)
    text = out.read_text()
    assert summary["source_file"] == synthetic_csv_gz.name
    for directory in (synthetic_csv_gz.parent, dst.parent, tmp_path, Path.cwd(), Path.home()):
        assert str(directory) not in text
    assert socket.gethostname() not in text
    assert json.loads(text) == summary


def test_no_csv_is_written(ingested, synthetic_csv_gz):
    dst, _ = ingested
    assert sorted(p.name for p in dst.parent.iterdir()) == [dst.name]
    # Nor next to the source, which is where a decompress-then-read would leave it.
    assert sorted(p.name for p in synthetic_csv_gz.parent.iterdir()) == [synthetic_csv_gz.name]


def test_never_reads_the_whole_file_at_once(tmp_path, synthetic_csv_gz, monkeypatch):
    def whole_file_read(*args, **kwargs):
        raise AssertionError("the ingest must stream, not read_csv the whole file")

    monkeypatch.setattr(pacsv, "read_csv", whole_file_read)
    summary = ingest(synthetic_csv_gz, tmp_path / "out.parquet", block_size=BLOCK_SIZE)
    assert summary["rows"] == N_ROWS


def _bad_file(tmp_path, header=HEADER, edit=None):
    lines = format_rows(make_columns(50, seed=1))
    if edit is not None:
        lines[20] = edit(lines[20])
    return write_csv_gz(tmp_path / "bad.csv.gz", header, lines)


def _set_field(name, value):
    position = SOURCE_COLUMNS.index(name)

    def edit(line):
        fields = line.split(",")
        fields[position] = value
        return ",".join(fields)

    return edit


@pytest.mark.parametrize(
    "header, edit, message",
    [
        (HEADER.replace("conversion,visit", "visit,conversion"), None, "Unexpected header"),
        (HEADER, _set_field("f3", ""), "null"),
        (HEADER, _set_field("f0", "inf"), "non-finite"),
        (HEADER, _set_field("treatment", "2"), "treatment"),
        (HEADER, _set_field("exposure", "-1"), "exposure"),
    ],
)
def test_refuses_a_malformed_file_and_leaves_nothing(tmp_path, header, edit, message):
    src = _bad_file(tmp_path, header, edit)
    dst = tmp_path / "out" / "criteo_uplift.parquet"
    with pytest.raises(ValueError, match=message):
        ingest(src, dst, rows_per_group=16, block_size=1 << 10)
    assert list(dst.parent.iterdir()) == []


def test_refuses_a_header_without_rows(tmp_path):
    src = write_csv_gz(tmp_path / "empty.csv.gz", HEADER, [])
    dst = tmp_path / "out" / "criteo_uplift.parquet"
    with pytest.raises(ValueError, match="no rows"):
        ingest(src, dst)
    assert list(dst.parent.iterdir()) == []
