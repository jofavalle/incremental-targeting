"""Random train, validation and test split of the ingested Parquet, stratified.

Criteo-UPLIFT is a randomized experiment with no time axis: units are
exchangeable, so the split is random, not temporal (docs/design.md §4).
It is stratified by treatment and by both outcomes, so each partition keeps the
treatment ratio of the experiment and its share of the rare positives. Both
outcomes, because which one is primary is decided later, by the power analysis.

Within each stratum (treatment x visit x conversion, up to eight cells) the rows
are shuffled with a seeded generator and cut at exact counts, so the fractions
hold per stratum up to one row of rounding, not only in expectation.

Memory stays bounded: the assignment needs only the three stratification
columns (int8), and the rows are then streamed row group by row group into the
three output files.

Reproducibility: the same input, seed and fractions give the same partition
under the locked NumPy. NumPy does not promise that Generator streams are stable
across releases, which is one more reason uv.lock is part of the contract.

## Coupling with config.py

`config.py`, the single source for the seed and the fractions, defines
`SplitConfig` with `seed` and `fractions`. `split` is **duck-typed**: it accepts
any object with those two attributes, so the tests pass a plain namespace and
the module imposes nothing the config does not already declare. Neither value
has a default here: both are fixed in config.py before any evaluation.

Usage:
    python -m uplift.data.split
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.config import CONFIG, DATA_PROCESSED, REPORTS_DIR
from uplift.data.ingest import DEFAULT_DST
from uplift.runinfo import run_metadata, write_json

STRATA = ("treatment", "visit", "conversion")
PARTITIONS = ("train", "val", "test")

DEFAULT_OUT_DIR = DATA_PROCESSED
DEFAULT_SUMMARY = REPORTS_DIR / "split_summary.json"


def _check_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(f) for f in fractions)
    if len(values) != len(PARTITIONS):
        raise ValueError(f"Need one fraction per partition {PARTITIONS}, got {values}.")
    if any(f <= 0 for f in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError(f"The fractions must be positive and add up to 1, got {values}.")
    return values  # type: ignore[return-value]


def _check_seed(seed: object) -> int:
    # None would draw a fresh partition on every run, and record "seed": null.
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"The seed must be an integer, got {seed!r}.")
    return int(seed)


def assign_partitions(
    columns: Sequence[np.ndarray], fractions: Sequence[float], seed: int
) -> np.ndarray:
    """Label each row 0 (train), 1 (val) or 2 (test), stratified by the binary `columns`.

    Rows sharing the same combination of values form a stratum. Each stratum is
    shuffled and cut so that partition k gets round(fractions[k] * n) of its n
    rows (test takes the remainder). Strata are visited in a fixed order, so the
    labels depend only on the column values, the fractions and the seed.
    """
    fractions = _check_fractions(fractions)
    seed = _check_seed(seed)
    n_rows = len(columns[0])
    code = np.zeros(n_rows, dtype=np.uint8)
    for column in columns:
        if len(column) != n_rows:
            raise ValueError("The stratification columns differ in length.")
        if column.size and (column.min() < 0 or column.max() > 1):
            raise ValueError("The stratification columns must be binary.")
        code = code * 2 + column.astype(np.uint8)

    labels = np.full(n_rows, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for stratum in range(2 ** len(columns)):
        rows = np.flatnonzero(code == stratum)
        if rows.size == 0:
            continue
        rng.shuffle(rows)
        n_train = round(fractions[0] * rows.size)
        n_val = round(fractions[1] * rows.size)
        labels[rows[:n_train]] = 0
        labels[rows[n_train : n_train + n_val]] = 1
        labels[rows[n_train + n_val :]] = 2
    if (labels < 0).any():
        raise RuntimeError("Rows were left without a partition.")
    return labels


def split(src: Path, out_dir: Path, cfg) -> dict:
    """Write train.parquet, val.parquet and test.parquet under `out_dir`; return the summary.

    `cfg` is anything with `seed` and `fractions` (see the module docstring).
    The summary holds sizes and the treatment share of each partition. It holds
    no outcome by arm: the outcomes of the test partition are looked at once,
    at the end, and not in a split report.
    """
    started = time.perf_counter()
    for attribute in ("seed", "fractions"):
        if not hasattr(cfg, attribute):
            raise AttributeError(f"SplitConfig has no attribute '{attribute}'; see config.py.")
    seed = _check_seed(cfg.seed)
    fractions = _check_fractions(cfg.fractions)
    parquet = pq.ParquetFile(src)
    strata = parquet.read(columns=list(STRATA))
    treatment = strata.column("treatment").to_numpy()
    labels = assign_partitions([strata.column(name).to_numpy() for name in STRATA], fractions, seed)
    del strata
    sizes = np.bincount(labels, minlength=len(PARTITIONS))
    if (sizes == 0).any():
        raise ValueError(f"A partition would be empty: sizes {dict(zip(PARTITIONS, sizes))}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    finals = [out_dir / f"{name}.parquet" for name in PARTITIONS]
    tmps = [path.with_name(path.name + ".tmp") for path in finals]
    try:
        with contextlib.ExitStack() as stack:
            writers = [
                stack.enter_context(pq.ParquetWriter(tmp, parquet.schema_arrow, compression="zstd"))
                for tmp in tmps
            ]
            offset = 0
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group)
                group_labels = labels[offset : offset + table.num_rows]
                offset += table.num_rows
                for k, writer in enumerate(writers):
                    writer.write_table(table.filter(pa.array(group_labels == k)))
            if offset != labels.size:
                raise RuntimeError(f"Streamed {offset} rows but assigned {labels.size}.")
    except BaseException:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, final in zip(tmps, finals):
        tmp.replace(final)

    treated = np.bincount(labels, weights=treatment, minlength=len(PARTITIONS))
    return {
        "source_file": src.name,
        "seed": seed,
        "fractions": dict(zip(PARTITIONS, fractions)),
        "strata": list(STRATA),
        "partitions": {
            name: {"rows": int(sizes[k]), "treatment_share": float(treated[k] / sizes[k])}
            for k, name in enumerate(PARTITIONS)
        },
        "run": {**run_metadata(), "seconds": round(time.perf_counter() - started, 1)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stratified random train/val/test split.")
    parser.add_argument("--src", type=Path, default=DEFAULT_DST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)

    summary = split(args.src, args.out_dir, CONFIG.split)
    write_json(args.summary, summary)
    sizes = ", ".join(f"{k} {v['rows']}" for k, v in summary["partitions"].items())
    print(f"{sizes} -> {args.out_dir}; summary -> {args.summary}")


if __name__ == "__main__":
    main()
