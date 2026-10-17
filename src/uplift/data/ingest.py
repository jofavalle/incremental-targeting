"""Criteo-UPLIFT CSV.gz to Parquet, as a stream.

The CSV (about 3 GB uncompressed) is never written to disk. The .gz is
decompressed in memory block by block, parsed with pyarrow's streaming CSV
reader and written out in row groups, so peak memory is a couple of row groups
and not the dataset. That is what lets the ingest run on the smallest machine of
the project (2 cores, 3.7 GiB).

Types are declared, never inferred: the twelve features as float32 and the four
binary columns as int8. A `row_id` column records the position of each row in
the source file. The file is not in random order (its first rows are sorted by
feature value), so nothing downstream may rely on position; the row_id makes the
split reproducible and joinable back to the source.

The ingest refuses a file that is not what it expects: a header other than the
sixteen columns of v2.1, a null, a non-finite feature, or a binary column
outside {0, 1}. It returns a summary with overall counts and rates, which is
where the dataset figures quoted in docs/design.md come from. Nothing is broken
down by treatment arm: that would be the ATE, and it belongs to the analysis.

`exposure` is kept. It is a post-treatment variable and must never be a feature
(docs/design.md, section 3.3), but it is part of the record of the experiment.

Usage:
    python -m uplift.data.ingest
    python -m uplift.data.ingest --threads 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from uplift.runinfo import run_metadata, write_json

FEATURES = tuple(f"f{i}" for i in range(12))
BINARY = ("treatment", "conversion", "visit", "exposure")
SOURCE_COLUMNS = FEATURES + BINARY  # the header of v2.1, in order
ROW_ID = "row_id"

SOURCE_TYPES = {**{name: pa.float32() for name in FEATURES}, **{name: pa.int8() for name in BINARY}}
SCHEMA = pa.schema(
    [(ROW_ID, pa.uint32())] + [(name, SOURCE_TYPES[name]) for name in SOURCE_COLUMNS]
)

DEFAULT_SRC = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
DEFAULT_DST = Path("data/processed/criteo_uplift.parquet")
DEFAULT_SUMMARY = Path("reports/data_summary.json")

ROWS_PER_GROUP = 1 << 20
BLOCK_SIZE = 16 << 20  # bytes of CSV text per parsed batch


def _check_batch(batch: pa.RecordBatch, first_row: int) -> None:
    """Raise ValueError if the batch holds a null, a non-finite feature or a non-binary flag."""
    span = f"rows {first_row} to {first_row + batch.num_rows - 1}"
    for name in SOURCE_COLUMNS:
        nulls = batch.column(name).null_count
        if nulls:
            raise ValueError(f"{name}: {nulls} null values in {span}")
    for name in FEATURES:
        if not np.isfinite(batch.column(name).to_numpy()).all():
            raise ValueError(f"{name}: non-finite values in {span}")
    for name in BINARY:
        values = batch.column(name).to_numpy()
        if values.min() < 0 or values.max() > 1:
            raise ValueError(f"{name}: values outside {{0, 1}} in {span}")


def ingest(
    src: Path,
    dst: Path,
    *,
    rows_per_group: int = ROWS_PER_GROUP,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Stream `src` (CSV.gz) into `dst` (Parquet, zstd) and return the summary.

    The Parquet is written to a temporary name and renamed at the end, so a
    failed run never leaves a file that looks complete.
    """
    started = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")

    counts = dict.fromkeys(BINARY, 0)
    exposure_without_treatment = 0
    conversion_without_visit = 0
    rows = 0
    row_groups = 0

    read_options = pacsv.ReadOptions(block_size=block_size)
    convert_options = pacsv.ConvertOptions(column_types=SOURCE_TYPES)
    try:
        with pa.input_stream(src, compression="gzip") as stream:
            reader = pacsv.open_csv(
                stream, read_options=read_options, convert_options=convert_options
            )
            header = tuple(reader.schema.names)
            if header != SOURCE_COLUMNS:
                raise ValueError(f"unexpected header {header}, expected {SOURCE_COLUMNS}")

            with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
                pending: list[pa.RecordBatch] = []
                pending_rows = 0
                for batch in reader:
                    n = batch.num_rows
                    if n == 0:
                        continue
                    _check_batch(batch, rows)
                    if rows + n > np.iinfo(np.uint32).max:
                        raise ValueError("more rows than a uint32 row_id can number")

                    flags = {name: batch.column(name).to_numpy() for name in BINARY}
                    for name in BINARY:
                        counts[name] += int(flags[name].sum(dtype=np.int64))
                    exposure_without_treatment += int(
                        np.count_nonzero((flags["exposure"] == 1) & (flags["treatment"] == 0))
                    )
                    conversion_without_visit += int(
                        np.count_nonzero((flags["conversion"] == 1) & (flags["visit"] == 0))
                    )

                    row_id = pa.array(np.arange(rows, rows + n, dtype=np.uint32))
                    pending.append(
                        pa.RecordBatch.from_arrays([row_id, *batch.columns], schema=SCHEMA)
                    )
                    pending_rows += n
                    rows += n

                    # Full row groups of exactly `rows_per_group`; the remainder waits.
                    while pending_rows >= rows_per_group:
                        table = pa.Table.from_batches(pending, schema=SCHEMA)
                        writer.write_table(
                            table.slice(0, rows_per_group), row_group_size=rows_per_group
                        )
                        row_groups += 1
                        rest = table.slice(rows_per_group)
                        pending = rest.to_batches()
                        pending_rows = rest.num_rows

                if pending_rows:
                    writer.write_table(
                        pa.Table.from_batches(pending, schema=SCHEMA),
                        row_group_size=rows_per_group,
                    )
                    row_groups += 1
        if rows == 0:
            raise ValueError(f"{src.name} holds a header and no rows")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)

    return {
        "source_file": src.name,
        "rows": rows,
        "counts": counts,
        "rates": {name: counts[name] / rows for name in BINARY},
        "consistency": {
            "exposure_without_treatment": exposure_without_treatment,
            "conversion_without_visit": conversion_without_visit,
        },
        "parquet": {
            "file": dst.name,
            "row_groups": row_groups,
            "rows_per_group": rows_per_group,
            "compression": "zstd",
        },
        "run": {**run_metadata(), "seconds": round(time.perf_counter() - started, 1)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Criteo-UPLIFT CSV.gz to Parquet, as a stream.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--threads", type=int, default=None, help="Arrow CPU threads (default: all cores)"
    )
    args = parser.parse_args(argv)

    if args.threads:
        pa.set_cpu_count(args.threads)
        pa.set_io_thread_count(args.threads)
    summary = ingest(args.src, args.dst)
    write_json(args.summary, summary)
    print(f"{summary['rows']} rows -> {args.dst}; summary -> {args.summary}")


if __name__ == "__main__":
    main()
