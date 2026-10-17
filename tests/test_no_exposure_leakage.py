"""The guard against the central trap of the dataset: `exposure` inside a model.

Whether a treated user is exposed depends on that user's own browsing, after
the treatment was assigned. Conditioning on it compares exposed treated users
with all the controls, and the randomization is gone (docs/design.md §3.3). The
analysis is intention to treat.

`row_id` is guarded too: the source file is sorted by treatment within blocks,
so the position of a row predicts its arm (§3.4).

What this test protects is the feature matrix, the way a model sees the data:
neither column can enter it. It cannot see a filter on `exposure`, nor a
contiguous slice of a partition (the files keep the source order, so a slice
carries the position without the column). design.md forbids both in words, and
any code that fits a model must draw its subsets by permutation (§3.4).

It runs on the ingested synthetic file, which carries `exposure` and `row_id`
exactly as the real Parquet does, so the guard is never vacuous.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uplift.data.ingest import FEATURES, ROW_ID, ingest
from uplift.features import FORBIDDEN, feature_matrix


@pytest.fixture(scope="module")
def table(tmp_path_factory, synthetic_csv_gz):
    dst = tmp_path_factory.mktemp("processed") / "criteo_uplift.parquet"
    ingest(synthetic_csv_gz, dst)
    return pq.read_table(dst)


def test_the_risk_is_real(table):
    # Without these columns in the data the guard below would be testing nothing.
    assert {"exposure", ROW_ID, "treatment", "visit", "conversion"} <= set(table.column_names)


def test_the_features_are_exactly_f0_to_f11(table):
    assert FEATURES == tuple(f"f{i}" for i in range(12))
    matrix = feature_matrix(table)
    assert matrix.shape == (table.num_rows, 12)
    assert matrix.dtype == np.float32 and matrix.flags["C_CONTIGUOUS"]
    for j, name in enumerate(FEATURES):
        np.testing.assert_array_equal(matrix[:, j], table[name].to_numpy())


@pytest.mark.parametrize(
    "column, reason",
    [
        ("exposure", "post-treatment"),
        (ROW_ID, "position in the source file"),
        ("treatment", "appends on purpose"),
        ("visit", "an outcome"),
        ("conversion", "an outcome"),
    ],
)
def test_a_forbidden_column_is_refused_with_its_reason(table, column, reason):
    # A literal list, not FORBIDDEN itself: dropping a column from FORBIDDEN must
    # fail here, not silently remove its case.
    assert column in FORBIDDEN
    with pytest.raises(ValueError, match=f"{column} is .*{reason}"):
        feature_matrix(table, columns=[*FEATURES, column])


def test_the_matrix_does_not_move_when_exposure_or_row_id_do(table):
    # The adversarial version: scramble everything that is not a feature. A
    # matrix that let exposure, row_id, the treatment or an outcome in changes.
    rng = np.random.default_rng(0)
    scrambled = table
    for name in ("exposure", ROW_ID, "treatment", "visit", "conversion"):
        column = table[name].to_numpy()
        index = scrambled.column_names.index(name)
        scrambled = scrambled.set_column(index, name, pa.array(rng.permutation(column)))
    np.testing.assert_array_equal(feature_matrix(scrambled), feature_matrix(table))


def test_a_new_column_in_the_table_cannot_slip_in(table):
    widened = table.append_column("exposure_lagged", table["exposure"])
    np.testing.assert_array_equal(feature_matrix(widened), feature_matrix(table))
    with pytest.raises(ValueError, match="not features"):
        feature_matrix(widened, columns=[*FEATURES, "exposure_lagged"])
