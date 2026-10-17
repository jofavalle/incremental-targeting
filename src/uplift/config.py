"""Central configuration for incremental-targeting.

Every value the analysis depends on lives here. No magic numbers elsewhere in
the codebase.

Each value is committed before any result that depends on it, so the history
shows that it was fixed first (docs/design.md §1): an assumption is never bent
after seeing the results. This version fixes the paths, the split, the
propensity check, the settings of the effect estimates, the power analysis, the
simulation study, the learners' setting, the selection on validation, the
evaluation on test and the decision layer with its costs and budgets.
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
# Power analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerConfig:
    """The power analysis, which decides the primary outcome (docs/design.md §8).

    Fixed on 2026-10-17, while the ATE was running and before it was seen: the
    benchmark is the measured ATE, so the rule had to come first. The sample is
    the test partition, by its arm sizes only; the rates are those of train.
    """

    #: Two-sided level and power of every minimum detectable effect.
    alpha: float = 0.05
    power: float = 0.80

    #: Shares of the sample reported in the record: the whole sample (the ATE
    #: against zero), a decile and a percentile, each contrasted with the rest.
    #: The level is Bonferroni-corrected over the 1 / share subgroups of a size.
    subgroup_shares: tuple[float, ...] = (1.0, 0.1, 0.01)

    #: Shares for the curve of the MDE of a contrast against subgroup size.
    curve_shares: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)

    #: `conversion` replaces `visit` as primary only if it supports heterogeneity
    #: at this share: the MDE of the contrast at most its measured ATE (§8.3).
    decision_share: float = 0.1


# --------------------------------------------------------------------------
# Simulation study
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One configuration of the simulation grid (docs/design.md §6.2).

    The defaults are the reference configuration; each scenario changes one thing.
    """

    #: A short name, used in the records and the tables.
    name: str

    #: The numbered questions of §6.2 this configuration answers.
    questions: tuple[int, ...]

    base_rate: float = 0.05
    treat_ratio: float = 0.85

    #: "simple" (linear logit) or "complex" (nonlinear, with an interaction).
    baseline: str = "simple"

    #: "heterogeneous", "step", "null" or "sleeping_dogs".
    effect: str = "heterogeneous"

    #: If true, the propensity depends on x1, which also moves the baseline and
    #: the effect (scenario 6).
    confounded: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    """The generator and the grid of the simulation study (docs/design.md §6.2).

    Fixed on 2026-10-17, before any learner existed, and never extended. The
    sample sizes and the number of replications were added the same day, once
    the learners existed and the cost of one replication had been measured
    (about 22 s with 2 threads at n = 100,000), and before the study ran.
    """

    #: Units of the training sample of each replication.
    n: int = 100_000

    #: Units of the independent validation sample, on which the PEHE and the
    #: transformed-outcome criterion are computed (§6.2, §6.3).
    n_validation: int = 100_000

    #: Replications per configuration. The Monte Carlo standard error of a PEHE
    #: is its spread over replications divided by sqrt(40), about a sixth of it;
    #: pairwise differences between learners remove the shared part.
    replications: int = 40

    #: The mean effect is this lift times the base rate: of the order of the lift
    #: measured on `visit`.
    relative_effect: float = 0.25

    #: The intercept of the baseline is calibrated to the base rate on this many
    #: seeded draws of X, shared by every configuration.
    calibration_rows: int = 1_000_000
    calibration_seed: int = 202610171

    #: Root seed of the replications: replication r of configuration s draws from
    #: SeedSequence([replication_seed, s, r]), so each one can be rerun alone.
    replication_seed: int = 202610172

    scenarios: tuple[Scenario, ...] = (
        Scenario("reference", questions=(1, 2)),
        Scenario("rare_outcome", questions=(1,), base_rate=0.003),
        Scenario("common_outcome", questions=(1,), base_rate=0.20),
        Scenario("balanced", questions=(2,), treat_ratio=0.5),
        Scenario("complex_baseline", questions=(3,), baseline="complex", effect="step"),
        Scenario("null_effect", questions=(4,), effect="null"),
        Scenario("sleeping_dogs", questions=(5,), effect="sleeping_dogs"),
        Scenario("confounded", questions=(6,), confounded=True),
    )


# --------------------------------------------------------------------------
# Learners
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerConfig:
    """The base learner shared by the S-, T-, X- and DR-learners (docs/design.md §5).

    Fixed on 2026-10-17, before any learner was evaluated, and the same for the
    four, so the comparison is between meta-learners and not between base models.
    Fixed rounds and no early stopping (§5.5).
    """

    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31

    #: Rows a leaf needs. At a rate of 0.3 %, a leaf of a few hundred rows holds
    #: about one positive; smaller leaves would fit noise.
    min_child_samples: int = 500

    #: Cross-fitting of the DR-learner's nuisance models, folds by `assign_folds`
    #: stratified by the treatment, with a seed of their own.
    n_folds: int = 5
    fold_seed: int = 202610173

    #: The seed of every LightGBM fit, which with `deterministic` and a fixed
    #: thread count makes a fit reproducible.
    random_state: int = 202610174


# --------------------------------------------------------------------------
# Selection, evaluation and the decision layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionConfig:
    """The learner chosen on validation by the transformed-outcome criterion (§6.3).

    Fixed on 2026-10-17, before the ATE was seen. The smallest criterion wins;
    the paired bootstrap says whether anything separates it from the others.
    """

    n_bootstrap: int = 1_000
    bootstrap_seed: int = 202610175


@dataclass(frozen=True)
class EvaluationConfig:
    """The uplift and Qini curves on the test partition, opened once (§7).

    Fixed on 2026-10-17, before the ATE was seen.
    """

    #: The fraction axis of the AUUC: 0, 0.01, ..., 1 (§7.1).
    curve_points: int = 101

    #: Resamples of the test units for the bands and the paired differences.
    n_bootstrap: int = 1_000
    bootstrap_seed: int = 202610176


@dataclass(frozen=True)
class DecisionConfig:
    """The four policies and the budget-return curve (§9).

    Fixed on 2026-10-17, before the ATE was seen. The cost of a contact is the
    unit of money, so the value of a conversion reads in contact costs.
    """

    #: The outcome every policy is scored on: the business metric (§9.5).
    counted_outcome: str = "conversion"

    #: Capacities K as shares of the test partition: the whole grid is reported.
    capacities: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

    #: The centre of the money assumptions, and their sweeps for the tornado.
    contact_cost: float = 1.0
    conversion_value: float = 1_000.0
    contact_costs: tuple[float, ...] = (0.5, 1.0, 2.0)
    conversion_values: tuple[float, ...] = (250.0, 500.0, 1_000.0, 2_000.0, 4_000.0)


# --------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The whole configuration, passed explicitly to the code that uses it."""

    split: SplitConfig = field(default_factory=SplitConfig)
    propensity: PropensityConfig = field(default_factory=PropensityConfig)
    effect: EffectConfig = field(default_factory=EffectConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    learners: LearnerConfig = field(default_factory=LearnerConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


CONFIG = Config()
