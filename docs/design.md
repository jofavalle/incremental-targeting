# Design

## 1. Purpose and status

This document is the specification of the project. When the code, the tests or
the README disagree with it, this document is settled first and the code follows.
Section numbers are stable: docstrings and tests refer to them.

Some choices are **fixed before evaluation**: they are decided before any result
on the validation or test partitions is seen, and they are not changed after.
They are the split seed and fractions, the simulation grid with its sample size
`n` and number of replications `R`, the costs and the grid of budgets `K`, and the
definitions of the evaluation curves in section 7, the AUC threshold of the
propensity check (section 6.4) and the multiplicity correction of the power
analysis (section 8.2). Their values live in `src/uplift/config.py`, and each
one is committed there before any result that depends on it, so the history
shows that it was fixed first.

Every figure quoted here is of one of three kinds, and says which: measured by
a run that wrote a JSON file under `reports/` with the machine, the package
versions, the seed and the threads; derived from those by arithmetic, not by a
new run; or quoted from the source, which is named.

## 2. Causal framework

### 2.1 Potential outcomes

Each user $i$ has two potential outcomes: $Y_i(1)$ if shown the campaign
(treated) and $Y_i(0)$ if not. The quantity to estimate and to rank by is the
conditional average treatment effect, or uplift:

$$\tau(x) = E[\,Y(1) - Y(0) \mid X = x\,].$$

### 2.2 The fundamental problem

Only one potential outcome is ever observed:
$Y_i = T_i\,Y_i(1) + (1 - T_i)\,Y_i(0)$. There is no ground truth for
$\tau(x)$ on real data, so the error of an estimator (the PEHE, section 6.1)
cannot be computed on Criteo. Every validation choice in section 6 follows from
this.

### 2.3 What is identified

If treatment is randomized with one ratio for everybody, $T$ is independent of
$(Y(0), Y(1), X)$ and the average effect is a difference of means,

$$\mathrm{ATE} = E[Y \mid T = 1] - E[Y \mid T = 0],$$

reported with its confidence interval (normal approximation for a difference
of two proportions, appropriate at millions of rows).

**That premise is checked, not assumed, and the check can only refute it.**
Criteo-UPLIFT assembles several incrementality tests, each randomized, and the
file is sorted by treatment within blocks (section 3.4). If the tests had
different treatment ratios, the pooled data are randomized only within a test.
When the features tell the tests apart, randomization holds given $X$: the
propensity $e(x) = P(T = 1 \mid X = x)$ is not constant, the raw difference of
means is not the ATE, and every formula below that uses a constant $e$ must use
$e(x)$. Section 6.4 tests this before any effect is estimated. A pass does not
prove the premise: a check on $X$ cannot see tests that differ in ratio but not
in $X$. That residual case is an **assumption** of the project, with its source:
the publisher describes the treatment as randomized and gives one treatment
ratio for the dataset.

If section 6.4 refutes a constant propensity, $\hat e(x)$ is estimated with
cross-fitting (no unit's $\hat e$ comes from a model fitted on that unit, which
would push $\hat e$ towards $T$) and clipped away from 0 and 1. Every difference
of means in this design then takes its weighted form: over a set $S$ of units,

$$\hat\tau_S = \frac{1}{|S|}\sum_{i \in S}\Big[\frac{T_i\,Y_i}{\hat e(X_i)} - \frac{(1 - T_i)\,Y_i}{1 - \hat e(X_i)}\Big],$$

or its normalized (Hájek) version. That covers the ATE, $Z$ (section 6.3), the
X- and DR-learners (sections 5.3 and 5.4), the calibration by decile (section
6.4), the curves of section 7 and the policy values of section 9.

Two consequences carry the rest of the design, under constant $e$ or given $X$:

- **Any CATE estimator must average to the ATE.** The mean of $\hat\tau(X)$ over
  a sample is compared with the ATE measured on the same sample, within its
  interval. A disagreement means a biased estimator.
- **Randomization holds inside any subset defined by $X$ alone.** The difference
  of means within such a subset is unbiased for the average effect there when
  $e$ is constant, and after weighting by $e(x)$ when it is not. Sections 6.4,
  7 and 9 rely on it. It does not hold for subsets defined by anything that
  happens after treatment (section 3.3).

## 3. Data

### 3.1 Source

Criteo-UPLIFT v2.1 (Diemert, Betlei, Renaudin and Amini, *A Large Scale
Benchmark for Uplift Modeling*, AdKDD and TargetAd Workshop, KDD 2018). It
assembles several incrementality tests in which users were randomly assigned to
be shown advertising or not. v2.1 is the version Criteo calls unbiased; the
original release (25,309,483 rows) has a leak according to its publisher and is
not used.

Measured by the ingest (`reports/data_summary.json`) against the figures the
source publishes:

| Property | Measured | Published |
|---|---|---|
| Rows | 13,979,592 | 13,979,592 |
| Features | `f0` to `f11`, dense, anonymized | 12 |
| Treatment share | 0.8500001 | 0.85 |
| Visit rate | 0.0469920 | 0.046992 |
| Conversion rate | 0.0029167 | 0.00292 |
| Exposure rate | 0.0306312 | not stated |

There are 11,882,655 treated and 2,096,937 control users (the second derived:
rows minus treated). Two consistency checks hold exactly: no control user is
exposed, and no conversion happens without a visit.

The link on the Criteo AI Lab page redirects to a storage host that no longer
resolves (checked 2026-10-16). The file is taken from the `criteo` organization
on Hugging Face, pinned to a dataset revision, and verified against the SHA256
that its LFS pointer publishes (`scripts/download_data.sh`).

**Licence: CC BY-NC-SA 4.0** (non-commercial, share alike). The data are never
committed to this repository; only aggregate figures are published.

### 3.2 Which outcome

- `visit` (rate 0.047) is the **provisional** primary outcome: it carries enough
  signal to estimate heterogeneity.
- `conversion` (rate 0.0029) is the secondary, reported outcome. It is the
  business metric, and its effect is a fraction of a rate of three per thousand.

The power analysis (section 8) confirms or revises this choice, before any CATE
model is fitted. If `conversion` turns out not to support fine heterogeneity,
that is a result to report, with the calculation that predicts it.

### 3.3 The trap: `exposure` is a post-treatment variable

`treatment` is the randomized arm. `exposure` records whether the ad was
actually displayed, which only happens to treated users and depends on their
own browsing. In this dataset **only 3.6 % of the treated were exposed**
(428,212 of 11,882,655; derived from the counts, since no control is exposed).

Conditioning on `exposure`, by filtering to exposed users or by using it as a
feature, conditions on a consequence of treatment and of the user's behaviour.
Exposed users differ systematically from unexposed ones within the treated arm,
and the comparison with controls is no longer randomized: it is selection bias.

The analysis is therefore **intention to treat**: `treatment = 1` against
`treatment = 0`, whatever the exposure. The effect it measures is the effect of
being assigned to the campaign, diluted by the 96 % of treated users who never
saw an ad. The effect on the exposed (the complier average causal effect) is
identified by instrumental variables, with `treatment` as the instrument for
`exposure`; it is an extension, out of scope here.

A test, `test_no_exposure_leakage`, asserts that `exposure` never reaches a
feature matrix.

**`row_id` has the same status.** It records the position of a row in the
source file, and position predicts treatment almost perfectly (section 3.4).
Feature matrices are built by selecting `f0` to `f11` by name, never by
excluding columns, so that no other column can slip in.

### 3.4 Infrastructure

- The download keeps the file compressed (311,422,618 bytes, the size its LFS
  pointer states and the download checks). The ingest streams the gzip through
  pyarrow's CSV reader into Parquet (zstd, row groups of $2^{20}$ rows) and
  never writes the uncompressed CSV to disk.
- Types are declared, not inferred: features as `float32`, the four flags as
  `int8`. `float32` keeps about seven significant digits, so distinct source
  values can collapse into one; the effect on ties is checked in the EDA.
- **The file is not in random order: it is sorted by treatment within
  blocks.** `reports/data_summary.json` records thirteen runs of constant
  treatment, the first of 5,571,799 treated users in a row. The treated share of
  consecutive pairs of runs ranges from 0.55 to 0.96 (derived); if each pair is
  one of the assembled tests, which the source does not say, the tests had
  different ratios (section 2.3).
- Nothing may rely on position. A head, a tail or a contiguous slice is a biased
  sample, and the train, validation and test files keep the source order, so
  this holds inside them too. Folds, subsamples and early-stopping sets are
  drawn by a seeded random permutation, never as contiguous blocks.
- A `row_id` column keeps the position of each row, so that any partition can
  be traced back to the source. It is never a feature (section 3.3).
- The smallest machine of the project has 2 cores and 3.7 GiB. `make memory`
  measures the peak resident memory of the ingest, the split and a LightGBM fit
  on the whole train partition, each in its own process, with 2 threads.

## 4. Splits

### 4.1 Random, stratified

There is no time axis: the units of a randomized experiment are exchangeable,
and the correct split is random. A temporal split, right for a stream of
transactions, would here be a rule applied without its reason.

| Partition | Fraction | Use |
|---|---|---|
| Train | 60 % | Fitting the learners |
| Validation | 20 % | Model selection (section 6.3), tuning |
| Test | 20 % | Final evaluation. **Looked at once.** |

The split is stratified by `treatment`, `visit` and `conversion` together (up
to eight strata). Both outcomes, because the primary one is decided after the
split, by the power analysis. Each stratum is shuffled with a seeded generator
and cut at exact counts, so the treatment ratio and the share of positives hold
in every partition up to one row of rounding per stratum, not only in
expectation. With these fractions the test partition holds about 2.8 million
users, of whom about 419 thousand are controls (derived: 20 % of 2,096,937).

The seed and the fractions are fixed in `config.py`. The partition is
reproducible from the same input, seed and fractions under the locked NumPy
(`uv.lock`): NumPy does not promise stable random streams across releases.

### 4.2 The test partition is looked at once

The learner is chosen on validation. Choosing it by looking at the test would
bias the reported curves upwards. The split report records the size and the
treatment share of each partition, and no outcome broken down by arm.

Stratifying by the outcomes has a consequence that must be said. With exact
cuts in every cell of treatment and outcome, the two-by-two table of each
partition is that of the whole dataset up to rounding, so **the ATE is the same
in every partition by construction**: the ATE measured on train, before any
model, is numerically the ATE of the test. The single look protects the evaluation of
the CATE (the ranking, the curves, the policies), not the ATE. Stratifying by
treatment alone would leave the outcome counts of the test to sampling noise
(about 1 % on the count of conversions, derived) and keep its ATE unknown until
the end. The design keeps the stratification by outcome, for the share of rare
positives in every partition, and accepts this consequence.

## 5. Estimators

Four meta-learners, compared. Each fails in a different way, and the simulation
study (section 6.2) predicts which one should win on these data and why.

The base learner is LightGBM for all four, with the same settings, so the
comparison is between meta-learners and not between base models. Notation:
$\mu_t(x) = E[Y \mid X = x, T = t]$ and $e = P(T = 1)$, the propensity.

### 5.1 S-learner

One model on $(X, T)$ jointly, $\hat\tau(x) = \hat\mu(x, 1) - \hat\mu(x, 0)$.

Failure mode: $T$ is one feature among thirteen, and a regularized model can
shrink its effect towards zero when the baseline $\mu_0$ is complex and the
effect small. The S-learner tends to underestimate heterogeneity, at the limit
predicting the same uplift everywhere.

### 5.2 T-learner

Two independent models, $\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$.

Failure mode: high variance when one arm is small. With an 85/15 split the
control model learns from 15 % of the data (about 1.26 million rows of train,
derived), and its errors pass straight into $\hat\tau$. Both models also fit the
complex baseline separately, and the difference of two noisy estimates is noisier
than either.

### 5.3 X-learner

Built for unbalanced treatment, which is the case here.

1. Fit $\hat\mu_0$ and $\hat\mu_1$, as the T-learner does.
2. Impute individual effects: for the treated, $D^1_i = Y_i - \hat\mu_0(X_i)$;
   for the controls, $D^0_i = \hat\mu_1(X_i) - Y_i$.
3. Regress $D^1$ on $X$ over the treated, giving $\hat\tau_1$, and $D^0$ on $X$
   over the controls, giving $\hat\tau_0$.
4. Combine: $\hat\tau(x) = e\,\hat\tau_0(x) + (1 - e)\,\hat\tau_1(x)$.

**Why the weight goes where it goes.** With $e \approx 0.85$, the weight 0.85
falls on $\hat\tau_0$, which is fitted on the controls, the *small* arm. The
point is not the size of the arm it is fitted on but the quality of its
imputation: $D^0$ is built from $\hat\mu_1$, learnt on the large treated arm and
therefore accurate, while $D^1$ is built from $\hat\mu_0$, learnt on the small
control arm and therefore noisy. The X-learner trusts the imputation that rests
on the well-estimated model.

### 5.4 DR-learner

A doubly robust pseudo-outcome (AIPW) regressed on $X$:

$$\varphi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i)
+ \frac{T_i\,(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)}
- \frac{(1 - T_i)\,(Y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)},$$

then $\hat\tau(x)$ is a regression of $\varphi$ on $X$. Its expectation given
$X$ is $\tau(X)$ if either the outcome models or the propensity are correct.

It is written by hand, without `econml` or `causalml`, with **cross-fitting**:
the data are cut into folds by a seeded random permutation (never contiguous
blocks, section 3.4), the nuisance models ($\hat\mu_0$, $\hat\mu_1$,
$\hat e$) are fitted on all folds but one and $\varphi$ is computed on the
held-out fold, so that no unit's pseudo-outcome uses a model fitted on that
unit. An estimated propensity is clipped away from 0 and 1, at a bound fixed in
`config.py`.

If section 6.4 does not refute a constant propensity, $e$ is known and double
robustness buys little. The DR-learner earns its place in simulation scenario 6,
with confounding. There the S- and T-learners remain consistent, since there is
no unmeasured confounding, but where overlap is poor they extrapolate and their
regularization bias is no longer averaged out, so they are expected to degrade;
the DR-learner corrects with the propensity.

### 5.5 Implementation

All four learners are written by hand in `src/uplift/learners/`. Writing the
DR-learner too, rather than taking it from a library, departs from the original
plan: it is the one whose correctness depends on details (cross-fitting,
clipping) that a library call hides.

## 6. Validation

### 6.1 The problem

The metric one would want is the precision in estimation of heterogeneous
effects,

$$\mathrm{PEHE} = \sqrt{E\big[(\hat\tau(X) - \tau(X))^2\big]},$$

and it cannot be computed on real data, because $\tau(X)$ is not observed. Three
lines of validation answer how to know whether an estimator works.

### 6.2 Simulation study

Synthetic data where $\tau(x)$ is known exactly, to check that the estimators
recover it before they are trusted on real data: a Monte Carlo validation of a
method against simulated truth.

**The data generating process.** Covariates in 12 dimensions; a propensity
that is constant (randomized) or depends on $X$ (scenario 6); a baseline
$\mu_0(x)$ at a chosen base rate; an effect function; $\mu_1 = \mathrm{clip}(\mu_0 + \tau, 0, 1)$;
and Bernoulli draws of both potential outcomes, of which only the one matching
$T$ is observed.

**The true effect is the effect after clipping:**
$\tau_{\text{true}}(x) = \mu_1(x) - \mu_0(x)$, not the input $\tau(x)$. With a
low base rate or a negative effect the two differ, and scoring against the
input would charge the learners for the clipping. The generator returns
$\tau_{\text{true}}$, for evaluation only, never for fitting. Two tests guard
it: that the returned effect equals the clipped difference, and
(`test_dgp_recovers_known_ate`) that the difference of means on generated data
recovers the mean of $\tau_{\text{true}}$ within Monte Carlo error.

**The scenarios.** Each answers one question. The grid is fixed in writing here
and in `config.py`, and not extended.

| # | Scenario | Question |
|---|---|---|
| 1 | Rare outcome (several base rates, down to that of `conversion`) | Why is `conversion` hard? |
| 2 | Unbalanced treatment (50/50 against 85/15) | Does the T-learner degrade? Does the X-learner recover? |
| 3 | Complex baseline, simple effect | Does the S-learner shrink the effect towards zero? |
| 4 | Null effect, $\tau \equiv 0$ | Do the learners invent heterogeneity? |
| 5 | Sleeping dogs, $\tau(x) < 0$ in a region | Do the learners find them? |
| 6 | Confounding, $e(x)$ depends on $X$ | Do S and T degrade where overlap is poor? Does DR correct it? |

Scenario 2 replicates the 85/15 of Criteo: the simulation predicts which learner
should win on the real data, and that prediction is checked.

**The metric.** For each scenario and learner, over $R$ replications with sample
size $n$ (both fixed in `config.py` before running):

- the PEHE, absolute;
- the PEHE normalized by the standard deviation of $\tau_{\text{true}}$, so that
  scenarios with different effect sizes can be compared. In scenario 4 that
  standard deviation is zero and only the absolute PEHE is reported;
- each with its Monte Carlo standard error, the standard deviation over
  replications divided by $\sqrt{R}$. All learners are fitted on the same
  replications, so differences between learners are compared pairwise, which
  removes the variation shared by the replication.

### 6.3 The transformed outcome: selection on real data

With a known, constant propensity $e$, define

$$Z_i = Y_i\,\frac{T_i - e}{e\,(1 - e)},$$

with $e(X_i)$ in place of $e$ if section 6.4 finds the propensity not constant.

Then $E[Z \mid X = x] = \tau(x)$: $Z$ is an unbiased and very noisy estimate of
the individual effect, computable on Criteo. With $e = 0.85$, $Z$ is $1/e
\approx 1.18$ for a treated positive and $-1/(1 - e) \approx -6.67$ for a
control positive. For an estimate $\hat\tau$ that is **independent of $Z$ given
$X$**,

$$E[(\hat\tau(X) - Z)^2] = E[(\hat\tau(X) - \tau(X))^2] + E[\mathrm{Var}(Z \mid X)],$$

and the second term does not depend on $\hat\tau$, so ranking learners by the
mean squared error against $Z$ ranks them by PEHE.

The independence condition is the whole point. It holds only **out of sample**:
on the rows a learner was fitted on, the cross term does not vanish and the
criterion rewards overfitting. The criterion is therefore computed **on the
validation partition only**, never on train. Because $Z$ is so noisy, the
learners are compared by a **paired bootstrap** of the difference of their
criteria: validation rows are resampled, the difference is recomputed on each
resample with the same rows for both learners, and its percentile interval is
reported. Two learners whose interval covers zero are not ranked.

The simulation checks that this criterion picks the same winner as the true
PEHE. If it does, the criterion is trusted on real data; if it does not, that is
a finding.

### 6.4 Cheap checks on real data

1. **$\mathrm{mean}(\hat\tau) \approx \mathrm{ATE}$** on the same partition,
   within the interval of the ATE.
2. **Balance and constant propensity.** Standardized mean differences of every
   feature between the arms; and a classifier of $T$ on $X$, fitted on train
   and evaluated on validation. The AUC above which the propensity is declared
   not constant is fixed in `config.py` before the check runs: with millions of
   rows any interval excludes 0.5, so "close to 0.5" needs a number. The check
   runs **first**, before any ATE, because of the order of the file (section
   3.4). It can refute a constant propensity, not confirm it (section 2.3); if
   it refutes it, the weighted forms of section 2.3 replace every difference of
   means.
3. **Calibration by decile of $\hat\tau$.** Within each decile of the predicted
   uplift, the observed uplift (difference of means within the decile, valid by
   section 2.3) against the mean prediction, each with its interval. The
   diagonal is the ideal.

### 6.5 Sensitivity to the propensity, declared after the check

Declared on 2026-10-16, after the check of section 6.4 ran and before any effect
was estimated.

The check passed at the margin of its rule. The validation AUC was 0.5102 and
its lower bound 0.5093, under the threshold of 0.51, so the primary analysis
keeps a constant propensity, as the rule decided. But the arms differ, by a small
and measurable amount (`reports/propensity_check.json`): the mean differences
bound the spread of the true propensity from below at 0.0068, sixteen times the
noise floor of that bound, and more than the 0.0045 the threshold was translated
into when it was fixed. That translation assumed a normal spread, and the
estimated propensity is skewed, with a tail of users assigned to the treatment
more often.

So every effect estimate is reported together with three sensitivity versions,
which never replace it:

1. **Weighted**: the Hájek form of section 2.3, with $\hat e$ cross-fitted on
   train (folds drawn by a seeded permutation, section 3.4) and clipped at the
   bound fixed in `config.py`, and the paired bootstrap interval of its
   difference from the primary estimate.
2. **Subclassified**: the difference of means within strata of $\hat e$ (its
   deciles, with the top 1 % as a stratum of its own), averaged with the weights
   of the strata.
3. **By blocks of the source file**, conditional on the unconfirmed reading that
   each pair of runs is one of the assembled tests: six blocks, each treated run
   with the control run that follows it, and the last treated run, which has no
   control after it, joined to the sixth. The difference of means within each
   block, averaged with the block weights. It is the only version that reaches
   the case no check on the features can see (section 2.3), and it is labelled
   with its condition wherever it appears. `row_id` defines the blocks here as a
   stratifier of an estimator, never as a feature of a model (section 3.3).

A discrepancy between the primary estimate and these versions is a finding,
reported with its size; it does not switch the branch after the fact.

**Fixed on 2026-10-16, before any effect.** The details the three versions
leave open are settled here; their values live in `EffectConfig`, in
`config.py`.

- **Cross-fitting.** Train is cut into `n_folds` folds by a seeded permutation
  (section 3.4), stratified by the treatment, the only variable whose share per
  fold matters to a model of $T$. The classifier and the clip are those of the
  propensity check (`PropensityConfig`). A train unit gets the $\hat e$ of the
  model that did not see it; a validation unit gets the mean of the `n_folds`
  models, all of them out of sample for it.
- **Strata of $\hat e$.** The cuts are the quantiles `stratum_quantiles` (the
  nine deciles and 0.99) of $\hat e$ in the evaluated partition, both arms
  pooled: eleven strata. A unit's stratum is the number of cuts strictly below
  its $\hat e$, so ties are never split. The $\hat e$ of a boosted model is a
  step function, so the strata come out unequal, and their sizes are recorded.
  An empty stratum is dropped; a stratum missing an arm is merged with the one
  below it (the lowest, with the one above). The weights are the shares of
  units, so the estimand stays the ATE.
- **Blocks.** The run boundaries are the run lengths recorded in
  `reports/data_summary.json`, accumulated along `row_id`.
- **Intervals.** The primary estimate keeps the normal interval of section 2.3.
  One paired bootstrap serves the three versions: `n_bootstrap` resamples of
  the units of the partition, with $\hat e$, strata and blocks held fixed, give
  percentile intervals at level `level` for each version and for its difference
  from the primary estimate. It measures the sampling noise given $\hat e$, not
  the noise of estimating $\hat e$. The difference is what tells a discrepancy
  from noise, so the subclassified and block versions get it too, not only the
  weighted one.

## 7. Evaluation

### 7.1 Uplift and Qini curves: definitions, fixed before evaluation

Sort the $N$ units of the evaluated partition by $\hat\tau$, descending. For the
top $n$ units let $N_T(n)$ and $N_C(n)$ be the number of treated and control
units, and $Y_T(n)$ and $Y_C(n)$ the sums of the outcome in each arm.

- **Uplift curve:**
  $U(n) = \Big(\frac{Y_T(n)}{N_T(n)} - \frac{Y_C(n)}{N_C(n)}\Big)\,n$,
  defined where both arms are present. It estimates the incremental outcomes
  obtained by treating the top $n$.
- **Qini curve:** $Q(n) = Y_T(n) - Y_C(n)\,\frac{N_T(n)}{N_C(n)}$. It scales the
  control outcomes to the size of the treated group, which matters with an
  85/15 split; the two curves are related by $U(n) = Q(n)\,n / N_T(n)$.
- **Ties.** Units with the same $\hat\tau$ form one block; the curves are
  evaluated at block boundaries and interpolated linearly in between, which is
  what a random order within the block gives in expectation. A learner that
  predicts one value everywhere gets the random line.
- **Random baseline:** the straight line from $(0, 0)$ to $(N, U(N))$, or to
  $(N, Q(N))$ for the Qini curve.
- **AUUC:** the area between the uplift curve and its random baseline, on the
  fraction axis $\phi = n / N$ and divided by $N$:
  $\mathrm{AUUC} = \frac{1}{N}\int_0^1 \big[U(\phi N) - \phi\,U(N)\big]\,d\phi$,
  by the trapezoid rule on $\phi \in \{0, 0.01, \ldots, 1\}$ with
  $U(0) = 0$. The Qini
  coefficient is the same area for $Q$.

Two tests pin the implementation: a random $\hat\tau$ gives a flat curve
(`test_qini_random_is_flat`, AUUC zero within Monte Carlo error), and on
simulated data the oracle, ranking by $\tau_{\text{true}}$ itself, gives an AUUC
clearly above zero and not below that of any learner.

### 7.2 Their limitation

The curves are themselves noisy estimates: they are differences of means in
ever smaller subsets, and their variance grows towards the top of the ranking.
They are reported with **bootstrap bands** (test units resampled, $\hat\tau$
kept fixed), and differences between learners with a paired bootstrap. Two
AUUCs with overlapping intervals do not rank the learners, and the text says
so.

## 8. Power analysis

The power analysis decides the primary outcome (section 3.2) and is a result in
itself. This section fixes its method; the numbers come from its run, before
any CATE model is fitted.

### 8.1 Power for the ATE

Two-proportion test with unequal arms. With $n_1$ treated and $n_0$ control
units, the smallest effect detected at level $\alpha$ with power $1 - \beta$ is

$$\mathrm{MDE} = (z_{1-\alpha/2} + z_{1-\beta})\,\sqrt{\frac{p_1(1 - p_1)}{n_1} + \frac{p_0(1 - p_0)}{n_0}}.$$

With $n_1 \approx 5.7\,n_0$ (85/15), the control term dominates: the control arm
is the binding one. Two choices make the calculation honest:

- **The sample is the one evaluated**, the test partition (about 419 thousand
  controls), not the full dataset.
- **The rates are the measured ones**: $p_0$ is the measured control rate, and
  the MDE is compared with the measured ATE, not with an assumed relative
  lift.

If the propensity is not constant (section 2.3), the variance is that of the
weighted estimator, larger than the formula above, and the calculation uses it.

### 8.2 Power for heterogeneity

The uplift is estimated over subgroups, and power falls with subgroup size:
about 42 thousand controls in a decile of the test partition and about 4.2
thousand in a percentile (derived). Heterogeneity is a **difference between
subgroups**, not a subgroup against zero, so the relevant minimum detectable
effect is that of a contrast. For a decile against the rest of the sample the
variance grows by a factor $1 + 1/9$ and the MDE by about 1.054; for two
deciles, by $\sqrt{2}$. With ten or a hundred subgroups the level is corrected
for multiplicity, with a correction fixed before evaluation. For each outcome
the result is the MDE of such a contrast as a function of subgroup size. If fine heterogeneity on `conversion` is
not detectable at these sizes, that is the conclusion, reported with the
calculation; on `visit` the same calculation is expected to be far more
favourable, and is checked.

## 9. Decision layer

### 9.1 Allocation

A budget $B$ and a cost per contact $c$ give a capacity $K = B / c$. A policy
contacts the top $K$ users by its score.

### 9.2 Four policies

Evaluated on the test partition, once.

| # | Policy | Score |
|---|---|---|
| 1 | Contact nobody | none; the baseline |
| 2 | Contact at random | random order |
| 3 | Top $K$ by propensity to convert | a classifier of the outcome on $X$, fitted on train with both arms pooled and $T$ not a feature |
| 4 | Top $K$ by predicted uplift | $\hat\tau$ of the learner chosen on validation |

For a policy, the incremental conversions of contacting its top $K$ are the
uplift curve of its score at $K$, $U(K)$ of section 7.1: valid because the top
$K$ is defined by $X$ alone, and randomization holds within it (in its weighted
form if the propensity is not constant, section 2.3). Policy 1 gives
zero and policy 2 gives $K \cdot \mathrm{ATE}$ in expectation. The headline
metric is incremental conversions per 1,000 of budget, $1000\,U(K) / (K c)$.

The figure the project is about is **the difference between policies 3 and 4**,
and it is reported with a **paired bootstrap interval**: test units are
resampled and both policies are evaluated on the same resample. The grid of $K$
and the costs are fixed in `config.py` before evaluation.

### 9.3 The budget-return curve

Incremental conversions against $K$, for the four policies, over the **whole**
grid of $K$: the range where policy 4 wins and the range where it does not.
Choosing the budget at which policy 4 looks best after seeing the results, and
reporting only that, would be p-hacking under another name. The budget at which
the marginal uplift of one more contact equals its cost is an output of the
analysis, not an input.

### 9.4 Sensitivity

The cost per contact and the value of a conversion are assumptions. Their
ranges are fixed in `config.py` before evaluation, swept, and summarized in a
tornado plot. The conclusion either survives the plausible range or the text
says it does not.
