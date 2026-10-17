# Incremental Targeting

**Who should a campaign contact when the budget is fixed? The users it changes,
not the users most likely to convert.**

> **Status: data and design in place (October 2026). No effect has been
> estimated yet.** The data are downloaded, verified against the published
> figures and split, behind guards against the dataset's traps; the design is
> fixed in [`docs/design.md`](docs/design.md). Every result below is blank on
> purpose, and each will be filled in from a measured run, never from an
> expectation.

**Headline:** incremental conversions per 1,000 of budget, ranking by predicted
uplift against ranking by propensity to convert, on the held-out test
partition: **not measured yet**.

---

## The problem

A company can afford to contact `K` customers. The usual answer is to train a
classifier of conversion and contact the top `K` by score. That ranks the wrong
quantity.

| Segment | Converts if contacted | Converts if not | Worth contacting? |
|---|---|---|---|
| Persuadables | yes | no | **Yes. The only ones that matter.** |
| Sure things | yes | yes | No: they convert anyway |
| Lost causes | no | no | No |
| Sleeping dogs | no | yes | **No: contacting them loses the conversion** |

A propensity model puts the sure things at the top, because they are by
construction the most likely to convert, and the budget goes to people who would
have converted anyway. What should be ranked is the incremental effect of the
contact, the uplift:

$$\tau(x) = E[\,Y(1) - Y(0) \mid X = x\,].$$

**The thesis:** under a fixed budget, ranking by conversion probability instead
of incremental effect leaves conversions on the table. This project measures how
many, or shows with a calculation why they cannot be measured here.

## The data

[Criteo-UPLIFT v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
(Diemert et al., 2018): advertising incrementality tests in which users were
randomly assigned to be shown ads or not. As measured by the ingest
([`reports/data_summary.json`](reports/data_summary.json)), every figure matching
the published ones:

| | |
|---|---|
| Users | 13,979,592 |
| Features | 12, dense and anonymized (`f0` to `f11`) |
| Treated | 85.0 % |
| Visit rate | 4.70 % |
| Conversion rate | 0.29 % |

The file is downloaded from the `criteo` organization on Hugging Face, pinned to
a revision and checked against the SHA256 that its LFS pointer publishes; the
download link on the Criteo page no longer resolves. The data are licensed
CC BY-NC-SA 4.0 and are never committed here.

## Key decisions

Each one is argued in `docs/design.md`.

1. **Intention to treat, and `exposure` is never a feature** (§3.3). `exposure`
   records whether the ad was actually displayed, which depends on the user's
   own browsing after assignment: only 3.6 % of the treated were exposed.
   Conditioning on it destroys the randomization. Models see the data only
   through one feature matrix, selected by name, and
   [`tests/test_no_exposure_leakage.py`](tests/test_no_exposure_leakage.py)
   checks that `exposure` cannot enter it.
2. **A random split, not a temporal one, and never by position** (§3.4, §4). A
   randomized experiment has no time axis. But the source file is sorted by
   treatment, in thirteen runs (the first is 5,571,799 treated users in a row),
   so a positional cut gives one-arm partitions and the row position predicts
   the treatment: `row_id` is kept out of the feature matrix by the same test,
   and any subset must be drawn by permutation, never as a contiguous slice.
   The split is random, stratified by treatment and both outcomes, 60/20/20,
   and the test partition is looked at once.
3. **Estimators validated against simulated truth before they touch real data**
   (§6). The error of an uplift model cannot be computed on real data, because a
   user's counterfactual is never observed. A Monte Carlo study over six regimes,
   where the truth is known, will measure each learner's error (PEHE) and predict
   which one should win on Criteo. On the real data the learner is chosen with
   the transformed outcome, on the validation partition only.
4. **Four meta-learners, each to be written by hand** (§5): S, T, X and a doubly
   robust learner with cross-fitting, all on LightGBM, so that the comparison is
   between meta-learners and not between base models.
5. **Every result with an error bar, every assumption fixed first** (§7 to §9).
   Each value in [`src/uplift/config.py`](src/uplift/config.py) is committed
   before any result that depends on it: today the split; the costs, budgets
   and thresholds will follow the same rule. The budget-return curve will be
   published whole, including where uplift targeting does not win.

## Results

Blank until measured.

| Result | Value |
|---|---|
| Incremental conversions per 1,000 of budget, uplift against propensity ranking | not measured |
| PEHE of the four learners across six simulated regimes | not measured |
| Learner predicted by the simulation, and whether it won on Criteo | not measured |
| Power analysis: smallest detectable heterogeneity, per outcome | not measured |
| AUUC of the chosen learner, with its bootstrap interval | not measured |

**Written down before the results, on purpose.** If the effect is not
heterogeneous enough to rank by, that is the result: when uplift modeling is not
worth the budget, and how to know beforehand, with the power calculation that
shows it. If ranking by propensity turns out to be nearly as good, the result is
the map, from the simulation, of where that approximation holds and where it
breaks. In neither case are the budget, the costs or the thresholds moved until
the conclusion looks better.

## Limitations, known in advance

- The effect measured is that of being assigned to the campaign (intention to
  treat), diluted by the 96 % of treated users who never saw an ad. The effect on
  the exposed needs instrumental variables, and is out of scope.
- Fine heterogeneity on conversion, at a rate of 0.29 %, may not be identifiable
  at this sample size. The power analysis decides, before any model is fitted.
- The costs and the value of a conversion are assumptions, swept over ranges
  fixed in advance.
- The constant propensity of a randomized design will be checked first, but a
  check on the features can only refute it, never confirm it (§2.3).

## Run it

```bash
uv sync --locked

# The suite, on synthetic data: no download, seconds.
make test
```

With the real data:

```bash
make data      # 311 MB, checksum verified, streamed into Parquet: the CSV never hits the disk
make split     # stratified train/validation/test, seed from config.py
make memory    # peak memory of each stage, with 2 threads
```

`make data` writes `reports/data_summary.json` and `make split` writes
`reports/split_summary.json`. Every figure quoted in this repository comes from
one of those records, is derived from them by arithmetic, or is quoted from the
source, which is named.

## Repository structure

```
src/uplift/
├── config.py         # every value the analysis depends on, fixed before evaluation
├── features.py       # the feature matrix: f0 to f11, by name, never by exclusion
├── runinfo.py        # machine, versions and threads, recorded with every number
└── data/             # streaming ingest (CSV.gz to Parquet), stratified split

tests/                # including the exposure leakage guard; all on synthetic data
scripts/              # checksum-verified download, peak memory probe
docs/design.md        # the specification: causal framework, estimators, validation
reports/              # the run summaries behind every quoted figure
```

## License

Code: MIT. Data: Criteo-UPLIFT, CC BY-NC-SA 4.0, not redistributed. Cite it as
Diemert, E., Betlei, A., Renaudin, C. and Amini, M.-R. (2018). *A Large Scale
Benchmark for Uplift Modeling.* AdKDD and TargetAd Workshop, KDD.
