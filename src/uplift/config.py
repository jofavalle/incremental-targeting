"""Central configuration for incremental-targeting.

Every value the analysis depends on lives here. No magic numbers elsewhere in
the codebase.

Each value is committed before any result that depends on it, so the history
shows that it was fixed first (docs/design.md §1): an assumption is never bent
after seeing the results. This version fixes the paths and the split. The
threshold of the propensity check, the simulation grid with `n` and `R`, the
costs and the grid of budgets `K` arrive with their milestones, each one before
its evaluation.
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
# Everything together
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The whole configuration, passed explicitly to the code that uses it."""

    split: SplitConfig = field(default_factory=SplitConfig)


CONFIG = Config()
