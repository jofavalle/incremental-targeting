"""Synthetic Criteo-UPLIFT files, so the suite runs without the dataset.

The generator writes the real v2.1 header and the same text format (floats with
17 significant digits, flags as 0 or 1), gzip compressed, at a size that keeps
the whole suite in seconds. Rates are chosen to look like the experiment
(treatment near 85 %, rare outcomes), not to match it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from uplift.data.ingest import BINARY, FEATURES, SOURCE_COLUMNS

N_ROWS = 20_000


def make_columns(n_rows: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Arrays for the sixteen source columns: float64 features and int8 flags."""
    rng = np.random.default_rng(seed)
    columns = {name: rng.normal(5.0, 3.0, n_rows) for name in FEATURES}
    treatment = (rng.random(n_rows) < 0.85).astype(np.int8)
    visit = (rng.random(n_rows) < 0.04 + 0.01 * treatment).astype(np.int8)
    conversion = (visit & (rng.random(n_rows) < 0.08)).astype(np.int8)
    exposure = (treatment & (rng.random(n_rows) < 0.04)).astype(np.int8)
    columns.update(treatment=treatment, conversion=conversion, visit=visit, exposure=exposure)
    return columns


def write_csv_gz(path: Path, header: str, lines: list[str]) -> Path:
    """Write a gzip compressed CSV from raw text, for the malformed cases."""
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(header + "\n")
        handle.writelines(line + "\n" for line in lines)
    return path


def format_rows(columns: dict[str, np.ndarray]) -> list[str]:
    """The source text format: features with %.17g, flags as integers."""
    n_rows = len(columns[FEATURES[0]])
    return [
        ",".join(
            [f"{columns[name][i]:.17g}" for name in FEATURES]
            + [str(int(columns[name][i])) for name in BINARY]
        )
        for i in range(n_rows)
    ]


@pytest.fixture(scope="session")
def synthetic_columns() -> dict[str, np.ndarray]:
    return make_columns(N_ROWS)


@pytest.fixture(scope="session")
def synthetic_csv_gz(tmp_path_factory, synthetic_columns) -> Path:
    path = tmp_path_factory.mktemp("raw") / "criteo-research-uplift-v2.1.csv.gz"
    return write_csv_gz(path, ",".join(SOURCE_COLUMNS), format_rows(synthetic_columns))
