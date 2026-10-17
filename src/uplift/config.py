"""Central configuration for incremental-targeting.

Every value the analysis depends on lives here. No magic numbers elsewhere in
the codebase.

Each value is committed before any result that depends on it, so the history
shows that it was fixed first (docs/design.md §1): an assumption is never bent
after seeing the results. This version fixes the paths, the split, the
propensity check and the settings of the effect estimates. The simulation grid
with `n` and `R`, the costs and the grid of budgets `K` arrive with their
milestones, each one before its evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# This file is at src/uplift/config.py, so the project root is three parents up.
# Resolving it this way means the code works regardless of the working directory
# it is invoked from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    # A non-editable install puts this file inside the virtual environment, and
    # every path below would then point there, silently.
    raise RuntimeError(
        f"config.py is not inside the repository (resolved {PROJECT_ROOT}); "
        "install the project in editable mode, as `uv sync` does."
    )

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Random split, stratified by treatment and both outcomes (docs/design.md §4).

    Random and not temporal: the units of a randomized experiment are
    exchangeable. Stratified by treatment, visit and conversion together, so
    every partition keeps the treatment ratio and its share of rare positives.
    """

    #: Seed of the generator that shuffles each stratum. It is the date the split
    #: was fixed, chosen before any outcome was looked at, and it is never tuned:
    #: a seed picked for the result it gives is a hidden degree of freedom.
    seed: int = 20261016

    #: Train, validation and test. Train fits the learners, validation selects
    #: among them (§6.3), and the test is looked at once (§4.2).
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)


# --------------------------------------------------------------------------
# Propensity check
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PropensityConfig:
    """The balance and propensity check, which runs before any ATE (docs/design.md §6.4).

    A classifier of the treatment on the twelve features is fitted on train and
    scored on validation. If the propensity were constant, as one treatment ratio
    for everybody would make it, its AUC would be 0.5; with millions of rows any
    interval excludes 0.5, so the rule needs a number, and it is fixed here before
    the check runs.

    Why 0.51. Let ``e(x)`` vary with standard deviation ``sigma_e`` around its
    mean ``e_bar``. Tilting by ``e`` (treated) and by ``1 - e`` (controls) shifts
    the mean of ``e`` between the arms by ``sigma_e**2 / (e_bar (1 - e_bar))``,
    so the AUC of the true propensity is about
    ``Phi(sigma_e / (sqrt(2) e_bar (1 - e_bar)))``, that is
    ``AUC - 0.5 = 2.21 sigma_e`` at ``e_bar = 0.85`` (checked by simulation up to
    ``sigma_e = 0.05``). An AUC of 0.51 is a propensity that moves by about 0.45
    points around 0.85; below it, weighting would change the control weights
    ``1 / (1 - e)`` by about 3 % and the treated weights by about 0.5 %.

    The check can refute a constant propensity, never confirm it: tests with
    different ratios that the features cannot tell apart stay invisible to it
    (§2.3).
    """

    #: The weighted forms of §2.3 replace every difference of means if the lower
    #: bound of the 95 % bootstrap interval of the validation AUC exceeds this.
    auc_threshold: float = 0.51

    #: Bootstrap resamples of the validation partition for the AUC interval.
    n_bootstrap: int = 200

    #: Standardized mean differences above this, in absolute value, are flagged
    #: (the usual convention, Austin 2009). Descriptive: they decide nothing.
    smd_flag: float = 0.1

    #: Clip of an estimated propensity, used by the weighted branch if it applies
    #: and by the weighted sensitivity version of §6.5 (`EffectConfig`): it keeps
    #: 1 / e and 1 / (1 - e) finite.
    clip: float = 0.01

    #: The classifier, LightGBM. Fixed rounds and no early stopping, so no
    #: contiguous evaluation slice is ever needed (§3.4). A large leaf minimum
    #: keeps it from fitting noise: what it should find, if anything, is a
    #: propensity that varies between whole tests, not between a few users.
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 1_000

    #: The date this rule was fixed, as for the split; never tuned.
    random_state: int = 20261016


# --------------------------------------------------------------------------
# Effect estimates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectConfig:
    """The ATE and its three sensitivity versions (docs/design.md §2.3, §6.5).

    Fixed on 2026-10-16, after the propensity check and before any effect was
    estimated. The primary estimate is the difference of means; the weighted,
    subclassified and block versions go next to it and never replace it. The
    classifier of the cross-fitted propensity and its clip are those of
    `PropensityConfig`, unchanged.
    """

    #: Level of every interval: the normal one of the primary estimate, and the
    #: bootstrap ones of each version and of its difference from the primary.
    level: float = 0.95

    #: Folds of the cross-fitted propensity on train. Each model sees 4/5 of
    #: train, less than the full-train fit `make memory` measured, so it fits
    #: wherever the propensity check did. Two folds would halve the data of each
    #: model; ten would double the time for models that see 90 % instead of 80 %.
    n_folds: int = 5

    #: The folds are stratified by the treatment alone: the model predicts it
    #: from the features, and no other share per fold matters to it.
    fold_strata: tuple[str, ...] = ("treatment",)

    #: Cuts of the strata of e_hat, as quantiles of the evaluated partition with
    #: both arms pooled: the nine deciles and 0.99, so the top 1 % is a stratum of
    #: its own. Eleven strata, and ties are never split (§6.5).
    stratum_quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)

    #: Resamples of the paired bootstrap, one set shared by the three versions.
    #: At a level of 0.95 each tail of a percentile interval rests on 25 of them.
    n_bootstrap: int = 1_000

    #: Seeds of the two NumPy draws: the date these values were fixed, with one
    #: digit per draw. Distinct from each other and from the split seed, because
    #: one seed on the same columns couples the draws (`uplift.data.split`).
    fold_seed: int = 202610161
    bootstrap_seed: int = 202610162


# --------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The whole configuration, passed explicitly to the code that uses it."""

    split: SplitConfig = field(default_factory=SplitConfig)
    propensity: PropensityConfig = field(default_factory=PropensityConfig)
    effect: EffectConfig = field(default_factory=EffectConfig)


CONFIG = Config()
