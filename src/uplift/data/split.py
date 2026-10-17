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

## Folds and subsamples

The source file is sorted by treatment in thirteen runs, and the partitions keep
that order, so any contiguous slice of them comes from one arm (§3.4). Folds for
cross-fitting (§5.4, §6.5), subsamples and early-stopping sets are drawn here,
with the same stratified shuffle as the partition:

- `assign_folds` deals the shuffled rows of each stratum in turn, like cards, so
  the fold sizes differ by at most one row, overall and within every stratum.
- `draw_subsample` takes the same fraction of every stratum and returns the rows
  in random order, so that even a prefix of it is a random draw.

This closes the gap by offering a tested way, not by forbidding a slice: nothing
stops `x[:n]` elsewhere, and a static guard against it would be brittle. The
number of folds, the fraction and the seed have no defaults here; they belong in
config.py, fixed before the result that uses them.

Draws that must be independent need distinct seeds. All three functions shuffle
the strata the same way, so on the same columns one seed couples them:
`draw_subsample` with fraction 0.6 returns the train rows of `assign_partitions`,
and two subsamples of different fractions come out nested.

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
from collections.abc import Iterator, Sequence
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


def _stratum_codes(columns: Sequence[np.ndarray]) -> np.ndarray:
    """The stratum of each row: the binary `columns` read as the bits of one number."""
    if len(columns) == 0:
        raise ValueError("Need at least one stratification column.")
    n_rows = len(columns[0])
    code = np.zeros(n_rows, dtype=np.uint8)
    for column in columns:
        if len(column) != n_rows:
            raise ValueError("The stratification columns differ in length.")
        if column.size and (column.min() < 0 or column.max() > 1):
            raise ValueError("The stratification columns must be binary.")
        code = code * 2 + column.astype(np.uint8)
    return code


def _shuffled_strata(
    code: np.ndarray, n_strata: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """The rows of each non-empty stratum, shuffled, strata in a fixed order.

    The order of the strata and of the draws is the contract that makes a
    partition reproducible from its seed: every caller consumes `rng` the same way.
    """
    for stratum in range(n_strata):
        rows = np.flatnonzero(code == stratum)
        if rows.size == 0:
            continue
        rng.shuffle(rows)
        yield rows


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
    code = _stratum_codes(columns)

    labels = np.full(code.size, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for rows in _shuffled_strata(code, 2 ** len(columns), rng):
        n_train = round(fractions[0] * rows.size)
        n_val = round(fractions[1] * rows.size)
        labels[rows[:n_train]] = 0
        labels[rows[n_train : n_train + n_val]] = 1
        labels[rows[n_train + n_val :]] = 2
    if (labels < 0).any():
        raise RuntimeError("Rows were left without a partition.")
    return labels


def assign_folds(columns: Sequence[np.ndarray], n_folds: int, seed: int) -> np.ndarray:
    """Label each row with a fold in 0..n_folds-1, stratified by the binary `columns`.

    Each stratum is shuffled and its rows are dealt to the folds in turn, the turn
    carrying over from one stratum to the next. The balance is exact, not in
    expectation: fold sizes differ by at most one row, overall and within every
    stratum. Fold k is held out with `labels == k` and fitted on `labels != k`.
    """
    seed = _check_seed(seed)
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError(f"The number of folds must be an integer, got {n_folds!r}.")
    if not 2 <= n_folds <= np.iinfo(np.int8).max:
        raise ValueError(f"The number of folds must be between 2 and 127, got {n_folds}.")
    code = _stratum_codes(columns)
    if code.size < n_folds:
        raise ValueError(f"A fold would be empty: {code.size} rows for {n_folds} folds.")

    labels = np.full(code.size, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    start = 0
    for rows in _shuffled_strata(code, 2 ** len(columns), rng):
        labels[rows] = (start + np.arange(rows.size)) % n_folds
        start += rows.size
    if (labels < 0).any():
        raise RuntimeError("Rows were left without a fold.")
    return labels


def draw_subsample(columns: Sequence[np.ndarray], fraction: float, seed: int) -> np.ndarray:
    """Indices of a random subsample, stratified by the binary `columns`, in random order.

    Each stratum of n rows contributes round(fraction * n) of them, so the share of
    every stratum holds up to one row. The indices are shuffled across strata, not
    sorted: sorted, they would follow the source order, and a prefix of them would
    again be a block of one arm.
    """
    seed = _check_seed(seed)
    if isinstance(fraction, bool) or not isinstance(
        fraction, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"The fraction must be a number, got {fraction!r}.")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"The fraction must be in (0, 1], got {fraction}.")
    code = _stratum_codes(columns)

    rng = np.random.default_rng(seed)
    strata = _shuffled_strata(code, 2 ** len(columns), rng)
    picked = [rows[: round(fraction * rows.size)] for rows in strata]
    subsample = np.concatenate(picked) if picked else np.empty(0, dtype=np.intp)
    if subsample.size == 0:
        raise ValueError(f"The subsample would be empty: fraction {fraction} of {code.size} rows.")
    rng.shuffle(subsample)
    return subsample


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
