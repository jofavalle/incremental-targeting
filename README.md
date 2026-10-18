# Incremental Targeting

**Who should a campaign contact when the budget is fixed? The users it changes,
not the users most likely to convert.**

> **Status (October 2026): every result below is measured.** Every setting
> was fixed in [`docs/design.md`](docs/design.md) and `config.py` before the
> result it governs, and the history shows it.

**Headline: on Criteo, ranking by predicted uplift does not beat ranking by
propensity to convert.** On the held-out test partition, opened once, contacting
the top 10 % by predicted uplift yields 7.5 incremental conversions per 1,000
contacts and the top 10 % by propensity 8.2, against 1.2 at random. Across the
whole grid of budgets the uplift ranking is never ahead, and where the paired
interval separates the two, it favours propensity. Both rankings beat random
targeting ten times or more at budgets of 5 % and below.

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

**Balance, checked before any effect**
([`reports/propensity_check.json`](reports/propensity_check.json)). The rule,
committed before the check ran, treats the propensity as constant unless the
lower bound of the 95 % interval of an out-of-sample AUC exceeds 0.51. A
classifier of the treatment on the features, fitted on train, scores 0.5102 on
validation (0.5093 to 0.5111): the lower bound is under the threshold and the
point estimate over it, so by the rule the primary analysis proceeds with a
constant propensity. The arms are not identical, though, and that is reported
with the same weight. The standardized mean differences are all below 0.05, but
5 to 50 times their noise level (about 0.001); and they imply a propensity that
varies by at least 0.68 points around 0.85 (a linear-projection bound whose noise
floor is 0.04 points), more than the 0.45 points the threshold was translated
into when it was fixed, a translation that assumed a normal spread. The figure
shows the skew: a tail of users assigned to the treatment more often. Every
effect is therefore reported with sensitivity versions that use an
estimated propensity, declared before any effect was computed (`docs/design.md`
§6.5).

![Estimated propensity by arm, on validation](reports/figures/fig1_propensity_overlap.png)

**Ties and repeated vectors** ([`reports/eda_summary.json`](reports/eda_summary.json)).
Storing the features as `float32` merges distinct source values in two
features. In `f2`, 2,051,900 source values become 808,663, and 4,836,685 rows
(34.6 %) hold a value that now shares its `float32` with another; in `f10`,
517,372 become 483,634, and 80,469 rows (0.6 %). The other ten keep every
value; four of them take fewer than 300. A quarter of the rows (3,560,505) share
their whole twelve-feature vector with another row, in groups of up to 24, and
18.5 % of the validation rows and 18.6 % of the test rows have a vector that
also appears in train. The vectors are compared in `float32`, so part of that
repetition may be the merging above and not the source. The data cannot say
whether the rest are repeated users or coarse features; if they are repeated
users, the evaluation on test is partly on users the models have seen, and its
figures lean optimistic.

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
3. **Estimators checked against simulated truth** (§6). The error of an uplift
   model cannot be computed on real data, because a user's counterfactual is
   never observed. A Monte Carlo study over six scenarios, where the truth is
   known, measures each learner's error (PEHE) and whether the criterion used on
   real data picks the right learner. Its generator, grid and size were
   committed before any learner was fitted on real data, but its results came in
   after the test partition was opened: it is a check on settings fixed in
   advance, not a prediction made before the test. On the real data the learner
   is chosen with the transformed outcome, on the validation partition only.
4. **Four meta-learners, each written by hand** (§5): S, T, X and a doubly
   robust learner with cross-fitting, all on LightGBM, so that the comparison is
   between meta-learners and not between base models.
5. **Every result with an error bar, every assumption fixed first** (§7 to §9).
   Each value in [`src/uplift/config.py`](src/uplift/config.py) is committed
   before any result that depends on it: the split, the propensity check, the
   effect estimates, the power analysis and its rule for the primary outcome,
   the simulation grid with its size, the learners' setting, the selection, the
   test evaluation, and the costs and budgets of the decision layer. The rule for
   the primary outcome, for one, was pushed while the ATE it compares against
   was still running. The budget-return curve is published whole, including
   where uplift targeting does not win.

## Results

| Result | Value |
|---|---|
| Incremental conversions per 1,000 contacts at a 10 % budget, uplift against propensity ranking | 7.5 against 8.2 (random: 1.2); uplift never ahead on the grid |
| PEHE of the four learners across six simulated scenarios | S smallest in all eight configurations; only in three does any learner match or beat predicting the mean effect for everyone |
| Learner the simulation favours, and whether it won on Criteo | S-learner; chosen on validation, largest AUUC on test with the X-learner's not separated from it. The simulation's results came after the test |
| Power analysis: smallest detectable heterogeneity, per outcome | conversion: a decile against the rest, not a percentile; visit: down to 2 % subgroups |
| AUUC of the chosen learner (S), with its bootstrap interval | 0.00045 (0.00038 to 0.00052), the highest of the four |

### The average effect, and how much it depends on the propensity

([`reports/ate.json`](reports/ate.json)). Assignment to the campaign raises
visits by **1.03 points** (95 % interval 0.97 to 1.10 on validation; 3.82 % to
4.85 %) and conversions by **0.115 points** (0.100 to 0.130; 0.194 % to
0.309 %), a relative lift of 27 % and 59 %. The partitions agree by
construction (`docs/design.md` §4.2), and the bootstrap interval matches the
normal one.

The three sensitivity versions declared before this was computed (§6.5) do not
agree with it, and the gap is far larger than their noise. On visit, weighting
by a cross-fitted propensity gives 0.75 points and subclassifying by it 0.74:
about a quarter less, each with a paired interval of the difference that
excludes zero. Reading the file order as tests, the block version gives 1.31
points: about a quarter more. Conversion shows the same pattern, smaller below
and larger above (0.101, 0.099 and 0.152 points against 0.115). So the
propensity's small spread is not harmless. Correcting for it moves the effect on
visit by a quarter, which is what happens if the users the features mark as more
likely to be treated also visit more. The two corrections
that read the features agree with each other and disagree with the one that
reads the file order, which says that the features do not carry the tests the
way the blocks would, or that the reading of the runs as tests is wrong; the
source settles neither. By the rule fixed before the check, the primary
estimate stays the difference of means, and every effect below is read with
this range in mind.

![The primary estimate and its sensitivity versions](reports/figures/fig2_ate_versions.png)

### The primary outcome, decided by the power analysis

([`reports/power.json`](reports/power.json)). The rule, pushed while the ATE was
running (§8.3): an outcome supports heterogeneity at a subgroup size if a
subgroup whose effect differs from the rest's by as much as the average effect
would be detected with power 0.8, at a level divided by the number of such
subgroups, on the sample sizes of the test partition; `visit` stays primary
unless `conversion` supports deciles. It does: against the rest, a decile's
minimum detectable effect on conversion is 0.81 times the ATE, so **conversion,
the business metric, becomes the primary outcome**. The margin is not wide: a
subgroup of 5 % needs 1.18 times the ATE and a percentile 2.9 times, and with
the weighted or subclassified ATE the decile ratio would be 0.92 or 0.95.
Visit supports subgroups down to 2 % (0.93) and not percentiles (1.37).

![Minimum detectable effect of a subgroup contrast, over the ATE](reports/figures/fig3_mde_by_subgroup.png)

### The simulation study: which learner, and how good any of them is

([`reports/simulation.json`](reports/simulation.json)). Eight configurations, each
answering one question of §6.2, with 40 replications of 100,000 units each,
fitted on one draw and scored on an independent one. The PEHE is divided by the
spread of the true effect, so **1 is exactly the error of predicting the mean
effect for everyone**.

- **The S-learner has the smallest PEHE in all eight configurations, and the
  X-learner the second smallest in all eight,** with paired differences of tens
  of Monte Carlo standard errors. In the configuration closest to conversion (a
  rate of 0.3 %, 85/15) the S-learner wins too. On Criteo it was chosen on
  validation and had the largest area on test, the X-learner's not separated
  from it, so **the simulation and Criteo agree**. It is an agreement, not a
  prediction: the study's generator, grid and size were committed before any
  learner was fitted on real data, so it could not adapt to the test, but its
  results came in after the test partition was opened.
- **The transformed-outcome criterion picks the PEHE's winner** in every
  replication of seven configurations and in 95 % of the confounded one, so the
  selection on Criteo rests on a criterion that works where it can be checked.
- **No learner is good in absolute terms.** Only in three configurations does the
  best one do as well as the mean effect for everyone or better (common outcome
  0.95, complex baseline 1.00, sleeping dogs 0.71). In the reference
  configuration (a rate of 5 %, 85/15) the S-learner's error is 1.96 times the
  constant's, and at the rate of conversion it is 5.1 times; the others are at
  14 to 29.
- **The S-learner shrinks, as feared, and wins anyway.** Its mean prediction is
  about 13 % under the mean effect, in the reference configuration and with a
  complex baseline alike. The X- and DR-learners are nearly unbiased on average
  and the T-learner overshoots by about a tenth, but all three are far noisier. One model whose two predictions share every split cancels much of
  the baseline's noise; two or four separate fits add theirs.
- **With a null effect, the T-, X- and DR-learners invent heterogeneity**, with a
  PEHE of 0.026 to 0.040 on a base rate of 0.05; the S-learner's is 0.0027.
- **Imbalance hurts the T-learner a little and the X-learner does not rescue it
  enough**: from 50/50 to 85/15 the T-learner goes from 4.97 to 5.80 and the
  X-learner from 3.17 to 4.15.
- **The DR-learner does not correct the confounding; it is the worst there**
  (14.7, against 1.93 for the S-learner, which barely moves from its reference
  1.96). There the true propensity nears 1 where $x_1$ is large (§6.2), and
  the pseudo-outcome weighs each control's residual by $1/(1 - \hat e)$, which
  only the clip of $\hat e$ at 0.99 keeps at 100 or below. The record does not keep
  the estimated propensities, so how close they came to the clip is not
  measured.

These are statements about this generator at this sample size, with one
LightGBM setting for all four learners. Criteo's train partition is 84 times
larger.

![PEHE by configuration and learner](reports/figures/fig4_pehe_by_scenario.png)

### The learner, chosen on validation

([`reports/selection.json`](reports/selection.json)). The four learners were
fitted on train for conversion and scored on validation by the transformed
outcome (§6.3). **The S-learner has the lowest criterion**, and the paired
bootstrap interval of its difference from each of the other three excludes
zero; the gaps are small against the criterion itself, as they must be when $Z$
is mostly noise. Two cheap checks on validation (§6.4). The first fails for all
four learners: the mean of $\hat\tau$ is 0.094 to 0.099 points, under the
interval of the ATE (0.100 to 0.130), so by §2.3 each of them predicts too small
an effect on average. Policy 4 uses only the order of $\hat\tau$, and its
incremental conversions are counted from the outcomes, so the levels of
$\hat\tau$ are not to be read as effects. By decile of $\hat\tau$, the second
check, most of the uplift sits in the top decile: it holds 81 % of the
incremental conversions summed over the S-learner's deciles, and there the
S-learner predicts 0.87 points and 0.73 are observed (0.58 to 0.88), more than
six times the average effect.
Several learners give fewer than ten deciles, because their $\hat\tau$ takes
few distinct values and ties are never split.

![Calibration by decile of the predicted uplift, on validation](reports/figures/fig5_calibration.png)

### The test partition, opened once

([`reports/test_evaluation.json`](reports/test_evaluation.json)). Every setting
was pushed before the ATE was seen (§7.3, §9.5); the break-even estimator was
amended after an adversarial review and before the test was opened.

**The learners' curves.** On conversion, the S-learner has the largest area over
the random line, 0.00045 (0.00038 to 0.00052), as it had the best criterion on
validation. The X-learner's area is not separated from it; the DR- and
T-learners' are smaller, with paired intervals that exclude zero.

![Uplift curves of the four learners on test](reports/figures/fig6_uplift_curves.png)

**The four policies.** Contacting everybody yields 3,224 incremental
conversions on the test partition. Policy 4 (top $K$ by the S-learner's
$\hat\tau$) never yields more than policy 3 (top $K$ by a classifier of
conversion): it is behind at every capacity below 100 %, where the two
coincide by construction, by up to 177 conversions, and the paired interval of the difference excludes zero at 2 %,
10 % and 20 %, always in favour of propensity. Both beat random targeting ten
times or more at budgets of 5 % and below: at 5 %, 12.4 incremental conversions
per 1,000 contacts by propensity and 12.2 by uplift, against 1.2; at 0.5 %, 55.9
and 46.4 in the same order, where the paired interval of their difference
includes zero.

![Incremental conversions against capacity, four policies](reports/figures/fig7_budget_return.png)

**The money assumptions.** With a conversion worth 1,000 contact costs, policy 3
does best at 20 % of the users (15 % to 27 %) and policy 4 at 25 % (16 % to 28 %),
intervals that overlap almost entirely, and policy 3's
best net value is 19.8 higher per 1,000 users (paired interval -44.2 to 4.7 for
policy 4 minus policy 3). Over the whole sweep, values from 250 to 4,000 and
costs from 0.5 to 2, policy 4 is behind by 8 to 57 per 1,000 users. The net
value is $c\,[(v/c)\,U(K) - K]$, so the best budget and the sign of the
difference depend on the ratio $v/c$ alone: the fifteen points are seven
ratios, and the cost only scales each. The interval excludes zero at three
points, never in policy 4's favour, and the three are one ratio, a conversion
worth 500 contacts. Each policy is taken at the best budget of its own test
curve, which flatters both.

![Sensitivity of the difference to the money assumptions](reports/figures/fig8_tornado.png)

**Why, as a hypothesis.** The effect on conversion is large in relative terms
(59 %) and concentrated in the users the models rank first, most of the uplift
sitting in the top decile (above). If the uplift grows with the baseline rate,
as a proportional lift would make it, the users most likely to convert are also
the ones the campaign moves most, and a classifier of conversion, fitted on all
of train with about 25,000 positives, ranks them with less noise than an uplift
model, which is a difference of two predictions. The data here do not test this;
the simulation study does not include a propensity policy, by its grid fixed in
advance.

**Written down before the results, on purpose.** If the effect is not
heterogeneous enough to rank by, that is the result: when uplift modeling is not
worth the budget, and how to know beforehand, with the power calculation that
shows it. If ranking by propensity turns out to be nearly as good, the result is
the map, from the simulation, of where that approximation holds and where it
breaks. In neither case are the budget, the costs or the thresholds moved until
the conclusion looks better.

**What happened.** The second case: ranking by propensity is as good or better.
The simulation's grid, fixed in advance and not extended, has no propensity
policy, so the map of where that approximation holds is not drawn here; the
hypothesis in the test section is where it would start.

## Limitations, known in advance

- The effect measured is that of being assigned to the campaign (intention to
  treat), diluted by the 96 % of treated users who never saw an ad. The effect on
  the exposed needs instrumental variables, and is out of scope.
- Fine heterogeneity on conversion, at a rate of 0.29 %, is not detectable at
  this sample size below subgroups of about 10 % (the power analysis above).
- The costs and the value of a conversion are assumptions, swept over ranges
  fixed in advance.
- 18.6 % of the test rows repeat a feature vector seen in train. If those are
  the same users, the test figures are partly in sample.
- The propensity is treated as constant by a rule fixed in advance, which it
  passed at the margin, while the features show a small but real spread (§6.4,
  §6.5). A check on the features can only refute a constant propensity, never
  confirm it (§2.3).

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
make eda       # ties that float32 creates, and duplicated feature vectors
make propensity  # balance and propensity check, before any effect: about 3 minutes
make ate       # the ATE and its three sensitivity versions, with a cross-fitted propensity
make power     # the power analysis, which decides the primary outcome
make simulate  # the simulation study against known truth: hours, and it resumes if stopped
make select    # the four learners on train, one chosen on validation by the transformed outcome
make evaluate  # the test partition, opened once: uplift curves and the four targeting policies
make memory    # peak memory of each stage, with 2 threads
```

`make data` writes `reports/data_summary.json`, `make split` writes
`reports/split_summary.json`, `make eda` writes `reports/eda_summary.json`,
`make propensity` writes
`reports/propensity_check.json`, `make ate` writes `reports/ate.json` and
`make power` writes `reports/power.json`, each of the last three with its
figure. Every figure quoted in this
repository comes from one of those records, is derived from them by arithmetic,
or is quoted from the source, which is named.

## Repository structure

```
src/uplift/
├── config.py         # every value the analysis depends on, fixed before evaluation
├── features.py       # the feature matrix: f0 to f11, by name, never by exclusion
├── runinfo.py        # machine, versions and threads, recorded with every number
├── propensity_check.py  # the balance and propensity check, run before any effect
├── ate.py            # the ATE with its three sensitivity versions
├── power_analysis.py # the power analysis, which decides the primary outcome
├── select_learner.py # the learner chosen on validation
├── evaluate_test.py  # the test partition, opened once: curves and policies
├── learners/         # the S-, T-, X- and DR-learners, written by hand
├── simulate/         # the generator with known truth, and the simulation study
├── data/             # streaming ingest (CSV.gz to Parquet), stratified split, folds and subsamples
└── evaluate/         # evaluators: balance, effects, power, selection and curves

tests/                # including the exposure leakage guard; all on synthetic data
scripts/              # checksum-verified download, peak memory probe
docs/design.md        # the specification: causal framework, estimators, validation
reports/              # the run records behind every quoted figure, and figures/
```

## License

Code: MIT. Data: Criteo-UPLIFT, CC BY-NC-SA 4.0, not redistributed. Cite it as
Diemert, E., Betlei, A., Renaudin, C. and Amini, M.-R. (2018). *A Large Scale
Benchmark for Uplift Modeling.* AdKDD and TargetAd Workshop, KDD.
