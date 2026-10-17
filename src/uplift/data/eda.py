"""The minimal EDA: ties created by float32, and duplicated feature vectors.

    python -m uplift.data.eda

It writes `reports/eda_summary.json`. Descriptive: it decides nothing, and it
reads no outcome.

## Ties from float32 (docs/design.md §3.4)

The source writes each feature with 17 significant digits; the Parquet keeps it
as float32, about seven. Distinct source values can then collapse into one.
For each feature, the source is streamed from the CSV.gz in float64, one feature
per pass so that memory holds one feature's distinct values, and compared with
its float32 rounding: how many distinct values there were, how many remain, and
how many rows hold a value that now shares its float32 with another. The float32
count of the Parquet is reported next to it, as a check of the ingest: pyarrow
parses the text straight to float32, numpy rounds the float64.

## Duplicated feature vectors

Each row of the twelve float32 features is reduced to a 64-bit hash (six 64-bit
words, mixed). Among 14 million distinct vectors two hashes coincide with
probability about n^2 / 2^65, a few in a million, so the counts are exact for
any practical purpose, and the docstring says so instead of the code paying for
an exact 48-byte sort. Reported: distinct vectors, rows in a repeated vector,
the largest repetition, and the share of validation and test rows whose vector
also appears in train. The partitions are read for their features only.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from uplift.config import DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import BLOCK_SIZE, DEFAULT_DST, DEFAULT_SRC, FEATURES
from uplift.data.split import PARTITIONS
from uplift.runinfo import run_metadata, write_json

DEFAULT_OUT = REPORTS_DIR / "eda_summary.json"

#: Merge the per-batch distinct values once this many have piled up.
MERGE_EVERY = 5_000_000

_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def distinct_with_counts(chunks: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The distinct values of the concatenated `chunks` and how often each occurs."""
    values, counts = np.empty(0), np.empty(0, dtype=np.int64)
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    pending_size = 0

    def merge() -> tuple[np.ndarray, np.ndarray]:
        all_values = np.concatenate([values, *(v for v, _ in pending)])
        all_counts = np.concatenate([counts, *(c for _, c in pending)])
        merged, inverse = np.unique(all_values, return_inverse=True)
        return merged, np.bincount(inverse, weights=all_counts).astype(np.int64)

    for chunk in chunks:
        pending.append(np.unique(np.asarray(chunk), return_counts=True))
        pending_size += pending[-1][0].size
        if pending_size >= MERGE_EVERY:
            values, counts = merge()
            pending, pending_size = [], 0
    if pending:
        values, counts = merge()
    return values, counts


def float32_collapse(values: np.ndarray, counts: np.ndarray) -> dict:
    """What rounding distinct float64 `values` (with their row `counts`) to float32 does."""
    rounded = values.astype(np.float32)
    _, inverse, sizes = np.unique(rounded, return_inverse=True, return_counts=True)
    shared = sizes[inverse] > 1  # float64 values whose float32 is not theirs alone
    return {
        "distinct_float64": int(values.size),
        "distinct_float32": int(sizes.size),
        "float64_values_merged": int(shared.sum()),
        "rows_with_a_merged_value": int(counts[shared].sum()),
    }


def row_hashes(x: np.ndarray) -> np.ndarray:
    """A 64-bit hash of each row of a float32 matrix with an even number of columns."""
    x = np.ascontiguousarray(x, dtype=np.float32) + np.float32(0.0)  # -0.0 becomes 0.0
    words = x.view(np.uint64)
    h = np.zeros(x.shape[0], dtype=np.uint64)
    for j in range(words.shape[1]):
        h = (h ^ words[:, j]) * _MULTIPLIER
        h ^= h >> np.uint64(31)
    return h


def duplicate_summary(hashes: dict[str, np.ndarray]) -> dict:
    """Repeated vectors over all partitions, and the overlap of each one with train."""
    pooled = np.concatenate(list(hashes.values()))
    _, counts = np.unique(pooled, return_counts=True)
    summary = {
        "rows": int(pooled.size),
        "distinct_vectors": int(counts.size),
        "rows_in_a_repeated_vector": int(counts[counts > 1].sum()),
        "largest_repetition": int(counts.max()),
    }
    train = np.unique(hashes["train"])
    for name, h in hashes.items():
        if name != "train":
            summary[f"{name}_rows_whose_vector_is_in_train"] = float(np.isin(h, train).mean())
    return summary


def _source_feature(src: Path, name: str, block_size: int) -> Iterable[np.ndarray]:
    """One feature of the source CSV.gz, in float64, batch by batch."""
    read_options = pacsv.ReadOptions(block_size=block_size)
    convert_options = pacsv.ConvertOptions(
        column_types={name: pa.float64()}, include_columns=[name]
    )
    with pa.input_stream(src, compression="gzip") as stream:
        for batch in pacsv.open_csv(
            stream, read_options=read_options, convert_options=convert_options
        ):
            yield batch.column(0).to_numpy()


def run_eda(src: Path, parquet: Path, processed: Path, block_size: int = BLOCK_SIZE) -> dict:
    started = time.perf_counter()
    features = {}
    for name in FEATURES:
        values, counts = distinct_with_counts(_source_feature(src, name, block_size))
        column = pq.read_table(parquet, columns=[name])[name].to_numpy()
        features[name] = {
            **float32_collapse(values, counts),
            "distinct_float32_in_parquet": int(np.unique(column).size),
        }
        del column
        print(f"  {name}: {features[name]}", flush=True)

    hashes = {}
    for partition in PARTITIONS:
        parquet_file = pq.ParquetFile(processed / f"{partition}.parquet")
        parts = []
        for group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group, columns=list(FEATURES))
            parts.append(row_hashes(np.column_stack([table[n].to_numpy() for n in FEATURES])))
        hashes[partition] = np.concatenate(parts)

    return {
        "source_file": src.name,
        "features": features,
        "duplicates": duplicate_summary(hashes),
        "run": {**run_metadata(), "seconds": round(time.perf_counter() - started, 1)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_DST)
    parser.add_argument("--processed", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    summary = run_eda(args.src, args.parquet, args.processed)
    write_json(args.out, summary)
    print(f"duplicates: {summary['duplicates']}; summary -> {args.out}")


if __name__ == "__main__":
    main()
