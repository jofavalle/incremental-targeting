"""Selection, evaluation and decision settings: what design.md §6.3, §7.3 and §9.5 fix."""

from __future__ import annotations

import numpy as np

from uplift.config import CONFIG

DECISION = CONFIG.decision


def test_the_counted_outcome_is_the_business_metric():
    assert DECISION.counted_outcome == "conversion"


def test_the_capacities_are_the_declared_grid():
    capacities = np.asarray(DECISION.capacities)
    assert DECISION.capacities == (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    assert (np.diff(capacities) > 0).all() and capacities[0] > 0 and capacities[-1] == 1


def test_the_money_sweeps_contain_their_centres():
    assert DECISION.contact_cost == 1.0 and DECISION.conversion_value == 1_000.0
    assert DECISION.contact_cost in DECISION.contact_costs
    assert DECISION.conversion_value in DECISION.conversion_values
    assert min(DECISION.conversion_values) == 250 and max(DECISION.conversion_values) == 4_000


def test_the_auuc_axis_is_hundredths():
    assert np.allclose(np.linspace(0, 1, CONFIG.evaluation.curve_points)[:3], [0, 0.01, 0.02])


def test_every_numpy_draw_has_its_own_seed():
    seeds = [
        CONFIG.split.seed,
        CONFIG.effect.fold_seed,
        CONFIG.effect.bootstrap_seed,
        CONFIG.simulation.calibration_seed,
        CONFIG.simulation.replication_seed,
        CONFIG.learners.fold_seed,
        CONFIG.selection.bootstrap_seed,
        CONFIG.evaluation.bootstrap_seed,
    ]
    assert len(set(seeds)) == len(seeds)
