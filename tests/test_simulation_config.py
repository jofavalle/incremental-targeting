"""The simulation grid of config.py: what design.md §6.2 fixes, and nothing more."""

from __future__ import annotations

from dataclasses import asdict

from uplift.config import CONFIG, Scenario

SIMULATION = CONFIG.simulation
REFERENCE = Scenario("reference", questions=(1, 2))


def test_the_grid_is_eight_configurations_answering_the_six_questions():
    assert len(SIMULATION.scenarios) == 8
    names = [s.name for s in SIMULATION.scenarios]
    assert len(set(names)) == len(names)
    answered = {q for s in SIMULATION.scenarios for q in s.questions}
    assert answered == {1, 2, 3, 4, 5, 6}


def test_each_scenario_changes_what_its_question_needs_from_the_reference():
    reference = asdict(REFERENCE)
    changed = {
        "rare_outcome": {"base_rate"},
        "common_outcome": {"base_rate"},
        "balanced": {"treat_ratio"},
        "complex_baseline": {"baseline", "effect"},
        "null_effect": {"effect"},
        "sleeping_dogs": {"effect"},
        "confounded": {"confounded"},
        "reference": set(),
    }
    for scenario in SIMULATION.scenarios:
        fields = asdict(scenario)
        differ = {k for k in fields if k not in ("name", "questions") and fields[k] != reference[k]}
        assert differ == changed[scenario.name], scenario.name


def test_the_values_are_the_declared_ones():
    by_name = {s.name: s for s in SIMULATION.scenarios}
    assert by_name["rare_outcome"].base_rate == 0.003  # about the rate of conversion
    assert by_name["common_outcome"].base_rate == 0.20
    assert by_name["balanced"].treat_ratio == 0.5
    assert SIMULATION.relative_effect == 0.25
    assert {s.effect for s in SIMULATION.scenarios} == {
        "heterogeneous",
        "step",
        "null",
        "sleeping_dogs",
    }


def test_the_simulation_seeds_differ_from_every_other_draw():
    seeds = [
        CONFIG.split.seed,
        CONFIG.effect.fold_seed,
        CONFIG.effect.bootstrap_seed,
        SIMULATION.calibration_seed,
        SIMULATION.replication_seed,
    ]
    assert len(set(seeds)) == len(seeds)
