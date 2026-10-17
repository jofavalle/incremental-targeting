"""Peak memory of the three heavy stages, each measured in its own process.

It answers one question: which of the three machines of the project can run
the full pipeline. The smallest has 2 cores and 3.7 GiB, so every stage runs
with 2 threads unless told otherwise.

Stages:
  ingest  the real CSV.gz to Parquet, as `make data` runs it.
  split   the stratified 60/20/20 split, with a throwaway seed: the peak does not
          depend on which rows land where.
  fit     a LightGBM binary fit on the whole train partition, f0..f11 plus
          treatment against visit. That is the largest single fit the project
          plans (an S-learner on train).

Each stage runs in a fresh child process that reports its own peak resident set
size (getrusage, RUSAGE_SELF), so no stage inherits the memory of the one
before. The work files go to a temporary directory under data/processed/ that
is removed at the end. Nothing here is a result of the analysis: the seed and
the LightGBM settings are those of a load test, not of a model.

Usage:
    python scripts/peak_memory.py                       # reports/peak_memory.json
    python scripts/peak_memory.py --threads 2 --rounds 100
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.data.ingest import DEFAULT_DST, DEFAULT_SRC, FEATURES, ingest
from uplift.data.split import split
from uplift.runinfo import run_metadata, write_json

STAGES = ("ingest", "split", "fit")
PROBE_SEED = 0  # throwaway: the split files are deleted and the peak does not depend on it
PROBE_FRACTIONS = (0.6, 0.2, 0.2)  # the 60/20/20 of docs/design.md; the fit needs train's size
DEFAULT_OUT = Path("reports/peak_memory.json")


def _fit(workdir: Path, threads: int, rounds: int) -> dict:
    import lightgbm as lgb

    columns = [*FEATURES, "treatment"]
    table = pq.read_table(workdir / "train.parquet", columns=[*columns, "visit"])
    n_rows = table.num_rows
    # One C-contiguous float32 matrix, filled column by column: LightGBM takes it
    # without a further copy.
    x = np.empty((n_rows, len(columns)), dtype=np.float32)
    for j, name in enumerate(columns):
        x[:, j] = table[name].to_numpy()
    y = table["visit"].to_numpy().astype(np.float32)
    del table
    dataset = lgb.Dataset(x, label=y)
    params = {"objective": "binary", "num_threads": threads, "verbose": -1}
    lgb.train(params, dataset, num_boost_round=rounds)
    return {"train_rows": n_rows, "rounds": rounds, "lightgbm_params": params}


def run_stage(stage: str, src: Path, workdir: Path, threads: int, rounds: int) -> dict:
    """Run one stage in this process and return its peak RSS and wall time."""
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    started = time.perf_counter()
    extra: dict = {}
    if stage == "ingest":
        extra = {"rows": ingest(src, workdir / DEFAULT_DST.name)["rows"]}
    elif stage == "split":
        summary = split(
            workdir / DEFAULT_DST.name, workdir, fractions=PROBE_FRACTIONS, seed=PROBE_SEED
        )
        extra = {"rows": {k: v["rows"] for k, v in summary["partitions"].items()}}
    elif stage == "fit":
        extra = _fit(workdir, threads, rounds)
    else:
        raise ValueError(f"unknown stage {stage}")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    return {
        "peak_rss_mib": round(peak_kib / 1024, 1),
        "seconds": round(time.perf_counter() - started, 1),
        **extra,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Peak memory of ingest, split and fit.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-parent", type=Path, default=DEFAULT_DST.parent)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.stage:  # child process: one stage, result on stdout
        result = run_stage(args.stage, args.src, args.workdir, args.threads, args.rounds)
        print(json.dumps(result))
        return

    # The parent runs no stage, but the run record it writes must show the
    # thread count the children ran with.
    pa.set_cpu_count(args.threads)
    args.work_parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="peak_memory_", dir=args.work_parent))
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads)}
    stages = {}
    try:
        for stage in STAGES:
            child = subprocess.run(
                [sys.executable, __file__, "--stage", stage, "--src", str(args.src)]
                + ["--workdir", str(workdir), "--threads", str(args.threads)]
                + ["--rounds", str(args.rounds)],
                capture_output=True,
                text=True,
                env=env,
            )
            if child.returncode != 0:
                sys.exit(f"stage {stage} failed:\n{child.stderr}")
            stages[stage] = json.loads(child.stdout.strip().splitlines()[-1])
            print(f"{stage}: {stages[stage]['peak_rss_mib']} MiB, {stages[stage]['seconds']} s")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report = {
        "source_file": args.src.name,
        "threads": args.threads,
        "stages": stages,
        "run": run_metadata(packages=("numpy", "pyarrow", "lightgbm")),
    }
    write_json(args.out, report)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
