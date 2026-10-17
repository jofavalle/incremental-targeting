"""The power analysis: the formulas of §8 by hand, and the decision rule of §8.3."""

from __future__ import annotations

import json
import math

import pytest
from scipy.stats import norm

from uplift.config import CONFIG
from uplift.evaluate.power import full_sample_variance, mde, subgroups_of_share
from uplift.power_analysis import main, run_power

POWER = CONFIG.power


def test_the_ate_mde_by_hand():
    variance = full_sample_variance(0.05, 0.04, 850_000, 150_000)
    assert variance == pytest.approx(0.05 * 0.95 / 850_000 + 0.04 * 0.96 / 150_000)
    z = norm.ppf(0.975) + norm.ppf(0.8)
    assert mde(variance, 1.0, 0.05, 0.8) == pytest.approx(z * math.sqrt(variance))


def test_a_decile_against_the_rest_as_section_8_2_says():
    # Against the decile alone, the contrast with the rest multiplies the variance
    # by 1 + 1/9, so the MDE by about 1.054; Bonferroni then tests at 0.005.
    variance = full_sample_variance(0.05, 0.04, 850_000, 150_000)
    decile = mde(variance, 0.1, 0.05, 0.8)
    alone = (norm.ppf(1 - 0.005 / 2) + norm.ppf(0.8)) * math.sqrt(variance / 0.1)
    assert decile / alone == pytest.approx(math.sqrt(1 + 1 / 9))
    assert math.sqrt(1 + 1 / 9) == pytest.approx(1.054, abs=1e-3)


def test_smaller_subgroups_need_larger_effects():
    variance = full_sample_variance(0.05, 0.04, 850_000, 150_000)
    values = [mde(variance, s, 0.05, 0.8) for s in POWER.curve_shares]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("share", [0.0, 1.5, 0.3])
def test_shares_that_do_not_split_the_sample_are_refused(share):
    with pytest.raises(ValueError, match="share"):
        subgroups_of_share(share)


def _records(conversion_ate: float):
    def primary(p0, ate):
        return {"primary": {"estimate": ate, "p0": p0, "p1": p0 + ate}}

    ate = {
        "partitions": {
            "train": {
                "outcomes": {
                    "visit": primary(0.0380, 0.0100),
                    "conversion": primary(0.0020, conversion_ate),
                }
            }
        }
    }
    split = {"partitions": {"test": {"rows": 2_795_918, "treatment_share": 0.85}}}
    return ate, split


@pytest.mark.parametrize(
    "conversion_ate, primary",
    [(0.0003, "visit"), (0.0015, "conversion")],
)
def test_conversion_becomes_primary_only_if_it_supports_deciles(conversion_ate, primary):
    record = run_power(*_records(conversion_ate), POWER)
    assert record["decision"]["primary"] == primary
    decile = next(r for r in record["outcomes"]["conversion"]["subgroups"] if r["share"] == 0.1)
    assert decile["supports"] == (primary == "conversion")
    assert decile["alpha"] == pytest.approx(0.005)


def test_the_sample_is_the_test_partition_by_its_counts(tmp_path):
    record = run_power(*_records(0.0003), POWER)
    assert record["sample"] == {"partition": "test", "treated": 2_376_530, "controls": 419_388}
    visit = record["outcomes"]["visit"]
    assert [row["share"] for row in visit["subgroups"]] == list(POWER.subgroup_shares)
    assert visit["subgroups"][1]["controls_in_subgroup"] == round(0.1 * 419_388)


def test_a_null_effect_has_no_benchmark_and_supports_nothing():
    record = run_power(*_records(0.0), POWER)
    for row in record["outcomes"]["conversion"]["subgroups"]:
        assert row["mde_over_ate"] is None and row["supports"] is False


def test_the_driver_writes_the_record_and_the_figure(tmp_path):
    ate, split = _records(0.0003)
    (tmp_path / "ate.json").write_text(json.dumps(ate))
    (tmp_path / "split.json").write_text(json.dumps(split))
    out, figure = tmp_path / "power.json", tmp_path / "fig3.png"
    main(
        ["--ate", str(tmp_path / "ate.json"), "--split", str(tmp_path / "split.json")]
        + ["--out", str(out), "--figure", str(figure)]
    )
    record = json.loads(out.read_text())
    assert record["decision"]["primary"] == "visit"
    assert str(tmp_path) not in out.read_text()
    assert figure.stat().st_size > 0
