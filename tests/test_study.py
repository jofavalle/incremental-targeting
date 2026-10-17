"""The simulation study driver: a tiny grid end to end, resuming, and the summary by hand."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from uplift.config import CONFIG
from uplift.simulate import study
from uplift.simulate.study import fingerprint, plot_pehe, run_study, summarize

SIM = CONFIG.simulation
KEEP = ("reference", "null_effect", "confounded")
TINY = replace(
    CONFIG,
    simulation=replace(
        SIM,
        n=3_000,
        n_validation=3_000,
        replications=2,
        calibration_rows=20_000,
        scenarios=tuple(s for s in SIM.scenarios if s.name in KEEP),
    ),
    learners=replace(CONFIG.learners, n_estimators=15, min_child_samples=50),
    propensity=replace(CONFIG.propensity, n_estimators=15, min_child_samples=50),
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    work = tmp_path_factory.mktemp("study") / "work.jsonl"
    return run_study(TINY, threads=1, work=work), work


def test_the_record_holds_every_configuration_and_learner(tiny):
    record, _ = tiny
    assert list(record["scenarios"]) == list(KEEP)
    for name, scenario in record["scenarios"].items():
        assert len(scenario["runs"]) == 2
        assert set(scenario["pehe"]) == {"S", "T", "X", "DR"}
        assert 0.0 <= scenario["criterion_picks_the_pehe_winner"] <= 1.0
        assert (scenario["pehe_normalized"] is None) == (name == "null_effect")
    assert record["n"] == 3_000 and record["replications"] == 2


def test_a_finished_study_resumes_without_refitting(tiny, monkeypatch):
    record, work = tiny

    def refuse(*args, **kwargs):
        raise AssertionError("a finished replication was fitted again")

    monkeypatch.setattr(study, "run_replication", refuse)
    again = run_study(TINY, threads=1, work=work)
    assert again["scenarios"] == record["scenarios"]


def test_replications_of_other_settings_are_not_reused(tiny):
    _, work = tiny
    other = replace(TINY, learners=replace(TINY.learners, n_estimators=16))
    assert fingerprint(other, 1) != fingerprint(TINY, 1)
    assert fingerprint(TINY, 2) != fingerprint(TINY, 1)  # the thread count matters too
    assert study._load(work, fingerprint(other, 1)) == {}
    assert len(study._load(work, fingerprint(TINY, 1))) == 6


def test_a_truncated_last_line_is_skipped(tiny, tmp_path):
    _, work = tiny
    copy = tmp_path / "work.jsonl"
    copy.write_text(work.read_text() + '{"scenario": "refer')
    assert len(study._load(copy, fingerprint(TINY, 1))) == 6


def test_the_summary_by_hand():
    def run(pehes, crits):
        learners = {
            name: {"pehe": p, "criterion": c, "mean_tau_hat": 0.01}
            for name, p, c in zip(("S", "T", "X", "DR"), pehes, crits)
        }
        return {"sd_tau_true": 0.02, "mean_tau_true": 0.01, "learners": learners}

    runs = [
        run([0.03, 0.02, 0.01, 0.04], [5, 4, 3, 6]),
        run([0.03, 0.04, 0.01, 0.02], [5, 3, 4, 6]),
    ]
    summary = summarize(runs)
    assert summary["pehe"]["X"]["mean"] == pytest.approx(0.01)
    assert summary["pehe"]["T"]["mc_se"] == pytest.approx(np.std([0.02, 0.04], ddof=1) / np.sqrt(2))
    assert summary["winner_by_pehe"] == "X"
    assert summary["criterion_picks_the_pehe_winner"] == 0.5  # right once, wrong once
    assert summary["pairwise_pehe_difference"]["T-X"]["mean"] == pytest.approx(0.02)
    assert summary["pehe_normalized"]["X"]["mean"] == pytest.approx(0.5)


def test_the_figure_and_no_local_path(tiny, tmp_path):
    record, work = tiny
    plot_pehe(record, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert str(work.parent) not in str(record)
