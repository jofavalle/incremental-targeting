"""Folds and subsamples: exact balance, and never a block of the file sorted by treatment."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.data.split import assign_folds, assign_partitions, draw_subsample

N_ROWS = 60_003  # not a multiple of the number of folds
N_FOLDS = 5

#: Treated share of each pair of runs, inside the range the source shows (0.55 to 0.96).
PAIR_SHARES = (0.96, 0.85, 0.70, 0.90, 0.55, 0.88)

DRAWS = {
    "folds": lambda columns, seed: assign_folds(columns, N_FOLDS, seed),
    "subsample": lambda columns, seed: draw_subsample(columns, 0.5, seed),
}


def _sorted_like_the_source(n_rows: int, seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Treatment and visit, sorted like the source file, and the run each row belongs to.

    Six pairs of a treated run followed by a control run, then a last treated run:
    thirteen runs, about 82 % treated. The visit flag is random, a second stratum
    that has nothing to do with position.
    """
    pair = (n_rows - n_rows // 16) // len(PAIR_SHARES)
    runs = []
    for share in PAIR_SHARES:
        treated = round(share * pair)
        runs += [np.ones(treated, dtype=np.int8), np.zeros(pair - treated, dtype=np.int8)]
    runs.append(np.ones(n_rows - pair * len(PAIR_SHARES), dtype=np.int8))
    treatment = np.concatenate(runs)
    run = np.repeat(np.arange(len(runs)), [r.size for r in runs])
    visit = (np.random.default_rng(seed).random(n_rows) < 0.05).astype(np.int8)
    return [treatment, visit], run


def _runs_spread_over_folds(labels: np.ndarray, run: np.ndarray, n_folds: int) -> bool:
    """True if every run puts 1/n_folds of its rows in each fold, within 5 standard errors."""
    p = 1 / n_folds
    for r in np.unique(run):
        in_run = labels[run == r]
        shares = np.bincount(in_run, minlength=n_folds) / in_run.size
        if np.abs(shares - p).max() > 5 * np.sqrt(p * (1 - p) / in_run.size):
            return False
    return True


def test_the_synthetic_file_has_the_thirteen_runs_of_the_source():
    (treatment, _), run = _sorted_like_the_source(N_ROWS)
    assert np.unique(run).size == 13
    assert (np.diff(treatment) != 0).sum() == 12


def test_folds_cover_every_row_once():
    columns, _ = _sorted_like_the_source(N_ROWS)
    labels = assign_folds(columns, N_FOLDS, seed=0)
    assert labels.dtype == np.int8
    assert labels.size == N_ROWS
    assert set(np.unique(labels)) == set(range(N_FOLDS))


def test_fold_sizes_differ_by_at_most_one_row():
    columns, _ = _sorted_like_the_source(N_ROWS)
    sizes = np.bincount(assign_folds(columns, N_FOLDS, seed=1))
    assert sizes.max() - sizes.min() <= 1


def test_every_stratum_is_dealt_evenly():
    # Exact, not in expectation: within each stratum the fold counts differ by
    # at most one row, so every fold has the treated share of the whole.
    columns, _ = _sorted_like_the_source(N_ROWS)
    labels = assign_folds(columns, N_FOLDS, seed=2)
    code = columns[0] * 2 + columns[1]
    for stratum in np.unique(code):
        counts = np.bincount(labels[code == stratum], minlength=N_FOLDS)
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("strata", ["treatment and visit", "visit only"])
def test_every_run_of_the_sorted_file_is_spread_over_the_folds(strata):
    # The permutation, not the stratification, is what breaks the runs: with the
    # treatment left out of the strata the runs are spread just the same.
    columns, run = _sorted_like_the_source(N_ROWS)
    if strata == "visit only":
        columns = columns[1:]
    labels = assign_folds(columns, N_FOLDS, seed=3)
    assert _runs_spread_over_folds(labels, run, N_FOLDS)


def test_the_run_check_catches_contiguous_folds():
    # Without this, the test above could pass by comparing nothing.
    _, run = _sorted_like_the_source(N_ROWS)
    blocks = np.array_split(np.arange(N_ROWS), N_FOLDS)
    contiguous = np.concatenate([np.full(block.size, k) for k, block in enumerate(blocks)])
    assert not _runs_spread_over_folds(contiguous, run, N_FOLDS)


def test_same_seed_same_folds_other_seed_other_folds():
    columns, _ = _sorted_like_the_source(N_ROWS)
    first = assign_folds(columns, N_FOLDS, seed=7)
    np.testing.assert_array_equal(first, assign_folds(columns, N_FOLDS, seed=7))
    assert (first != assign_folds(columns, N_FOLDS, seed=8)).mean() > 0.5


@pytest.mark.parametrize("n_folds", [1, 0, -3, 128, 2.0, True, None])
def test_rejects_an_invalid_number_of_folds(n_folds):
    columns, _ = _sorted_like_the_source(1000)
    with pytest.raises(ValueError, match="folds"):
        assign_folds(columns, n_folds, seed=0)


def test_rejects_more_folds_than_rows():
    with pytest.raises(ValueError, match="empty"):
        assign_folds([np.array([0, 1, 1], dtype=np.int8)], N_FOLDS, seed=0)


@pytest.mark.parametrize("draw", DRAWS.values(), ids=list(DRAWS))
@pytest.mark.parametrize("seed", [None, 1.5, True, "7", -1])
def test_rejects_a_seed_that_is_not_a_non_negative_integer(draw, seed):
    columns, _ = _sorted_like_the_source(1000)
    with pytest.raises(ValueError, match="seed must be"):
        draw(columns, seed)


@pytest.mark.parametrize("draw", DRAWS.values(), ids=list(DRAWS))
@pytest.mark.parametrize("bad", ["two", "a half", "nan"])
def test_rejects_non_binary_strata(draw, bad):
    # Inside [0, 1] is not enough: 0.5 and NaN would pass a range check.
    columns, _ = _sorted_like_the_source(1000)
    column = columns[1].astype(np.float64)
    column[0] = {"two": 2.0, "a half": 0.5, "nan": np.nan}[bad]
    with pytest.raises(ValueError, match="binary"):
        draw([columns[0], column], 0)


@pytest.mark.parametrize("draw", DRAWS.values(), ids=list(DRAWS))
def test_rejects_more_strata_than_the_code_holds(draw):
    columns, _ = _sorted_like_the_source(1000)
    with pytest.raises(ValueError, match="At most 8"):
        draw([columns[0]] * 9, 0)


@pytest.mark.parametrize("draw", DRAWS.values(), ids=list(DRAWS))
def test_rejects_no_stratification_column(draw):
    with pytest.raises(ValueError, match="at least one"):
        draw([], 0)


def test_subsample_rows_are_unique_and_in_range():
    columns, _ = _sorted_like_the_source(N_ROWS)
    rows = draw_subsample(columns, 0.3, seed=0)
    assert np.unique(rows).size == rows.size
    assert rows.min() >= 0 and rows.max() < N_ROWS


def test_every_stratum_keeps_its_share_in_the_subsample():
    columns, _ = _sorted_like_the_source(N_ROWS)
    rows = draw_subsample(columns, 0.3, seed=1)
    code = columns[0] * 2 + columns[1]
    picked = np.bincount(code[rows], minlength=4)
    for stratum, size in enumerate(np.bincount(code, minlength=4)):
        assert picked[stratum] == round(0.3 * size)


def test_every_run_of_the_sorted_file_keeps_its_share_in_the_subsample():
    # Taking the first rows of each stratum would keep the strata exact and still
    # be a block of the file: the early runs in full, the late ones not at all.
    columns, run = _sorted_like_the_source(N_ROWS)
    picked = np.zeros(N_ROWS, dtype=bool)
    picked[draw_subsample(columns, 0.3, seed=4)] = True
    for r in np.unique(run):
        in_run = picked[run == r]
        assert abs(in_run.mean() - 0.3) < 5 * np.sqrt(0.3 * 0.7 / in_run.size)


def test_subsample_is_in_random_order_so_a_prefix_has_both_arms():
    (treatment, visit), _ = _sorted_like_the_source(N_ROWS)
    rows = draw_subsample([treatment, visit], 0.3, seed=2)
    prefix = treatment[rows[:1000]]
    se = np.sqrt(treatment.mean() * (1 - treatment.mean()) / prefix.size)
    assert abs(prefix.mean() - treatment.mean()) < 5 * se
    # The same rows sorted would follow the source order: the first thousand
    # all come from the first run, which is treated.
    assert treatment[np.sort(rows)[:1000]].all()


def test_a_full_subsample_is_a_permutation_of_every_row():
    columns, _ = _sorted_like_the_source(N_ROWS)
    rows = draw_subsample(columns, 1.0, seed=3)
    np.testing.assert_array_equal(np.sort(rows), np.arange(N_ROWS))
    assert (np.diff(rows) < 0).any()


def test_same_seed_same_subsample_other_seed_other_subsample():
    columns, _ = _sorted_like_the_source(N_ROWS)
    first = draw_subsample(columns, 0.3, seed=7)
    np.testing.assert_array_equal(first, draw_subsample(columns, 0.3, seed=7))
    assert not np.array_equal(np.sort(first), np.sort(draw_subsample(columns, 0.3, seed=8)))


def test_one_seed_couples_the_draws():
    # What the module docstring warns about: on the same columns, the same seed
    # gives the train rows of the partition and nested subsamples.
    columns, _ = _sorted_like_the_source(N_ROWS)
    train = np.flatnonzero(assign_partitions(columns, (0.6, 0.2, 0.2), seed=9) == 0)
    np.testing.assert_array_equal(np.sort(draw_subsample(columns, 0.6, seed=9)), train)
    small, large = draw_subsample(columns, 0.1, seed=9), draw_subsample(columns, 0.2, seed=9)
    assert np.isin(small, large).all()


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5, float("nan"), True, None, "0.5"])
def test_rejects_a_fraction_outside_the_unit_interval(fraction):
    columns, _ = _sorted_like_the_source(1000)
    with pytest.raises(ValueError, match="fraction"):
        draw_subsample(columns, fraction, seed=0)


def test_rejects_an_empty_subsample():
    with pytest.raises(ValueError, match="empty"):
        draw_subsample([np.array([0, 1, 1], dtype=np.int8)], 0.1, seed=0)
