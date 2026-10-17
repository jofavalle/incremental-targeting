"""The feature matrix: the only way a model sees the data.

Twelve columns, `f0` to `f11`, selected by name. Never by excluding columns:
the Parquet also holds `treatment`, both outcomes, `exposure` and `row_id`, and
a matrix built as "everything but the targets" would carry two of them straight
into a model:

- `exposure` is a post-treatment variable. Conditioning on it, as a feature or
  as a filter, destroys the randomization (docs/design.md §3.3).
- `row_id` is the position in the source file, which is sorted by treatment
  within blocks, so it predicts the arm almost perfectly (§3.4).

The treatment itself is not a feature either: a learner that needs it (the
S-learner) appends it on purpose, next to this matrix.

`tests/test_no_exposure_leakage.py` guards all of this.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from uplift.data.ingest import FEATURES, ROW_ID

#: Columns that must never reach a feature matrix, and why.
FORBIDDEN: dict[str, str] = {
    "exposure": "post-treatment: conditioning on it destroys the randomization (§3.3)",
    ROW_ID: "the position in the source file, which predicts the treatment (§3.4)",
    "treatment": "the treatment, which a learner appends on purpose, never as a feature",
    "visit": "an outcome",
    "conversion": "an outcome",
}


def feature_matrix(table: pa.Table, columns: Sequence[str] = FEATURES) -> np.ndarray:
    """The `columns` of `table` as one C-contiguous float32 array, in that order.

    Only names in FEATURES are accepted, so a column cannot slip in by being
    present in the table: inclusion, not exclusion.
    """
    forbidden = [name for name in columns if name in FORBIDDEN]
    if forbidden:
        reasons = "; ".join(f"{name} is {FORBIDDEN[name]}" for name in forbidden)
        raise ValueError(f"Forbidden columns in a feature matrix: {reasons}.")
    unknown = [name for name in columns if name not in FEATURES]
    if unknown:
        raise ValueError(f"Columns {unknown} are not features; the features are {FEATURES}.")

    matrix = np.empty((table.num_rows, len(columns)), dtype=np.float32)
    for j, name in enumerate(columns):
        matrix[:, j] = table[name].to_numpy()
    return matrix
