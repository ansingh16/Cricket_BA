# Cricket_BA: does a hierarchical Bayesian model help with IPL wickets?

How often does a batter get out, and how often does a bowler take a wicket? Most IPL
players have faced or bowled only a few dozen balls, so their raw rates are mostly noise.
This project fits a hierarchical (partially pooled) Bayesian model in PyMC to 18 seasons
of IPL ball-by-ball data and compares it, on later seasons it has not seen, with a tuned
L2-regularised logistic regression.

**Short answer.** On point prediction the two models are close to a tie: the
hierarchical model's test log loss is lower by 0.00013, a 0.07% improvement whose 95%
interval only just excludes zero. What the Bayesian model adds is honest uncertainty:
every player gets an interval that is wide when the data is thin and narrow when it is
not, and those intervals cover roughly the share of outcomes they claim to. The largest
gain in prediction is for batters the model has never seen.

## Question and set-up

* **Target.** On a legal ball, does the bowler take the striker's wicket (bowled, caught,
  caught and bowled, lbw, stumped, hit wicket)? Run outs, retirements and super overs
  are left out. 267,801 legal balls, 12,586 such wickets (4.7%).
* **Split by time, never at random.** Train on 2008 to 2023 (234,995 balls), test on
  2024 and 2025 (32,806 balls). In the test seasons 12.7% of balls involve a batter, and
  11.2% a bowler, who never appears in training.
* **Baselines.** The overall training rate, the rate per phase, and an L2 logistic
  regression on one-hot batter, bowler and phase (powerplay, middle, death). Its penalty
  and feature set were chosen by expanding-window cross-validation over 2019 to 2023:
  C = 0.03, and venue was left out because it did not lower validation log loss.
* **Hierarchical model.**
  `logit p = alpha[phase] + sigma_bat * z_bat[batter] + sigma_bowl * z_bowl[bowler]`,
  with `z ~ N(0, 1)` (non-centred) and half-normal priors on the two pooling scales,
  which are learnt from the data. A player first seen in the test seasons gets a fresh
  effect drawn from `N(0, sigma)`.
* **Making NUTS fast without approximating.** The linear predictor depends only on
  (batter, bowler, phase), so a Bernoulli likelihood on the 234,995 balls equals a
  binomial likelihood on the 37,970 cells up to a constant. Same posterior, about three
  minutes to sample. `tests/test_hierarchical.py` checks the equality numerically.

## Results

All numbers below are from `make results` (see *How to reproduce*), on the 2024-25 test
seasons.

| Model | Log loss | Brier score | Mean predicted | Observed rate |
|---|---|---|---|---|
| Overall training rate | 0.19973 | 0.04784 | 0.0465 | 0.0504 |
| Rate per phase | 0.19805 | 0.04766 | 0.0463 | 0.0504 |
| Logistic L2 (batter, bowler, phase) | 0.19726 | 0.04759 | 0.0485 | 0.0504 |
| Hierarchical (PyMC) | **0.19713** | **0.04758** | 0.0485 | 0.0504 |

Hierarchical minus logistic, with 95% intervals from 1,000 bootstrap resamples of whole
matches: log loss **-0.00013** (-0.00026 to -0.00001), Brier score -0.000011 (-0.000025
to +0.000002). The log-loss gain is real but tiny; the Brier difference cannot be told
apart from zero.

The phase-rate row comes from `results/baseline_test_metrics.csv`, written by
`notebooks/02_baseline.ipynb`; the other rows from `results/test_metrics.csv`.

**Sampler.** 4 chains of 1,000 draws after 1,000 tuning steps: no divergences, R-hat at
most 1.01, bulk effective sample size at least 779 for every parameter.

### Where the hierarchical model helps

Test log loss split by how many training balls the player had (difference is
hierarchical minus logistic; negative favours the hierarchical model):

| Player | Training balls | Test balls | Difference | 95% interval |
|---|---|---|---|---|
| batter | unseen | 4,160 | -0.00031 | -0.00067 to -0.00001 |
| batter | 1-99 | 2,741 | -0.00011 | -0.00078 to +0.00051 |
| batter | 100-499 | 8,639 | -0.00014 | -0.00040 to +0.00012 |
| batter | 500+ | 17,266 | -0.00010 | -0.00027 to +0.00008 |
| bowler | unseen | 3,685 | +0.00006 | -0.00018 to +0.00030 |
| bowler | 1-99 | 1,646 | +0.00006 | -0.00039 to +0.00050 |
| bowler | 100-499 | 8,033 | -0.00022 | -0.00046 to -0.00001 |
| bowler | 500+ | 19,442 | -0.00015 | -0.00034 to +0.00003 |

The clearest gain is for batters new in the test seasons. The logistic model treats a
new batter as exactly average; the hierarchical model knows how far from average a new
batter tends to be, and averages over that. For thin but seen players the intervals are
wide and include zero. I expected the gain to be largest for thin players, and the data
does not show that clearly.

### Estimates for thin and well-known players

![Raw rates against both models' estimates](figures/shrinkage_batters.png)

Each point is a batter's wicket probability against an average bowler in the middle
overs. Raw rates for batters with under 100 training balls have a standard deviation of
0.150 (some are 0, some are 1). Both models pull them hard towards the average; the L2
logistic regression pulls even harder (standard deviation 0.0013 against 0.0027), with
one penalty for every player and no measure of how unsure it is. The hierarchical model
says how unsure it is:

| Batter | Training balls | Wickets | Raw rate | Logistic | Hierarchical mean | 80% interval |
|---|---|---|---|---|---|---|
| V Kohli | 5,568 | 186 | 0.0334 | 0.0339 | 0.0320 | 0.0292 to 0.0349 |
| S Dhawan | 5,181 | 170 | 0.0328 | 0.0343 | 0.0332 | 0.0302 to 0.0362 |
| RG Sharma | 4,759 | 198 | 0.0416 | 0.0380 | 0.0379 | 0.0348 to 0.0411 |
| LPC Silva | 26 | 2 | 0.0769 | 0.0436 | 0.0456 | 0.0340 to 0.0580 |
| T Thushara | 18 | 1 | 0.0556 | 0.0428 | 0.0439 | 0.0329 to 0.0560 |
| Basil Thampi | 35 | 1 | 0.0286 | 0.0418 | 0.0415 | 0.0317 to 0.0526 |

The three thin batters were drawn at random (fixed seed) from those with 15 to 40
training balls. The median width of an 80% interval is 0.0236 for batters with under 100
training balls and 0.0115 for those with 500 or more.

### Are the intervals calibrated?

For each player with at least 60 legal balls in 2024-25, the model gives a predictive
distribution for their total wickets over those balls (with their actual opponents and
phases). How often does the real total land inside the central interval?

| Players | Model | n | 50% | 80% | 95% |
|---|---|---|---|---|---|
| batters | hierarchical | 109 | 0.58 (PIT 0.44) | 0.85 (PIT 0.81) | 0.96 (PIT 0.95) |
| batters | logistic plug-in | 109 | 0.51 (PIT 0.39) | 0.81 (PIT 0.78) | 0.96 (PIT 0.94) |
| bowlers | hierarchical | 112 | 0.54 (PIT 0.46) | 0.82 (PIT 0.76) | 0.96 (PIT 0.96) |
| bowlers | logistic plug-in | 112 | 0.53 (PIT 0.44) | 0.79 (PIT 0.74) | 0.96 (PIT 0.95) |

"Logistic plug-in" is binomial noise around the logistic point predictions, with no
parameter uncertainty. Wicket totals are whole numbers, so an interval with inclusive
ends covers a little more than its nominal level; the PIT figure is a randomised version
that corrects for this. With about 110 players per row, each figure has a standard error
of about 4 percentage points, and the simulation adds about 1 to 2 points of Monte Carlo
noise (the notebook and the pipeline differ by that much). So the 80% and 95% intervals
are about right for both models, the hierarchical ones sit slightly closer to nominal,
and this test is too small to say that one is better calibrated than the other. The 50%
PIT figures are below 0.5 in every row, which points to the shared under-prediction
described under *Limitations*.

![Calibration on the test seasons](figures/calibration_test.png)

## Limitations

* **Both models under-predict 2024-25.** Wicket rates rose after 2021 and the test
  seasons (5.0% of legal balls) sit above every training season. Both models predict
  4.85%. Neither has a season or trend term, so the offset belongs to the split, not to
  either model, but it does affect the calibration figures.
* **The model expects batters to differ slightly more than they do.** The posterior
  predictive check puts the observed spread of batter rates (batters with 100 or more
  balls) at the 5th percentile of the replicated spread. A likely cause is batting role:
  openers and tail-enders face very different balls and the model has no term for it.
* **Player skill is held fixed over 16 training seasons.** A batter's 2010 form counts
  the same as his 2023 form. A time-varying effect would be the natural next step.
* **Bowler effects are small and hard to pin down** (pooling scale 0.068 against 0.219
  for batters, on the log-odds scale), so the bowler side of the model adds little.
* **Venue was dropped** because cross-validation did not support it, and the package
  model has no venue term.
* **One competition.** Results are for the IPL only; whether they hold for other T20
  leagues is untested.
* **The point-prediction gain is tiny.** Anyone choosing between these two models for
  ball-by-ball probabilities alone could reasonably pick the cheaper logistic model.

## Data and licence

Ball-by-ball data for every IPL match from 2008 to 2025 (1,169 matches), from
[Cricsheet](https://cricsheet.org/). Cricsheet's JSON match data is licensed under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
The tables and results in this repository are works produced from that data. Thanks to
Cricsheet for collecting and publishing it.

The raw files are not stored here. `make data` downloads
`https://cricsheet.org/downloads/ipl_json.zip` once (Cricsheet's robots.txt allows
`/downloads/`). The zip grows as seasons finish, so the pipeline keeps the 2008 to 2025
seasons explicitly. Four unchanged match files are kept in `tests/fixtures/` for the
tests, under the same licence.

**Data checks.** The JSON has no scorecard totals, but every chase carries a target. In
all 1,142 chases not affected by Duckworth-Lewis, the parsed first-innings total equals
the target minus one. Players are keyed by Cricsheet registry id, not by name: one name
("Harmeet Singh") belongs to two different players, and two players appear under two
names. 59 raw venue names reduce to 38 grounds.

An earlier parser (October 2025, not in this repository) produced a ball-level CSV in
which every row had match id "unknown", innings 0 and ball 0, because it read keys that
Cricsheet does not provide. `src/cricket_ba/parse.py` replaces it, and
`tests/test_parse.py` includes a regression test.

## How to reproduce

Needs [uv](https://docs.astral.sh/uv/) and a C compiler (PyTensor compiles the model).

```
uv sync --locked     # .venv with the exact versions in uv.lock
make data            # download the Cricsheet JSON
make results         # parse, fit both models, write results/ (about 4 minutes on 8 cores)
make test lint       # what CI runs: ruff and pytest, no MCMC
```

The notebooks were run in order: `01_eda`, `02_baseline`, `03_hierarchical_pymc`.
`make results` reruns the analysis from the package code in `src/cricket_ba/` and is the
source of the numbers above. It reproduces the notebooks' sampler diagnostics and
baseline numbers exactly; the hierarchical numbers agree to the precision shown here,
and differ further down because two random steps are drawn in a different order. The
seed (20261112), sampler settings and package versions are in
`results/hierarchical_run.json`.

```
src/cricket_ba/   parse.py (Cricsheet JSON to tables), features.py, baseline.py,
                  hierarchical.py (PyMC model), evaluate.py, pipeline.py
notebooks/        01_eda, 02_baseline, 03_hierarchical_pymc
tests/            pytest, with four Cricsheet fixture matches
results/          tables written by `make results` and the notebooks
figures/          figures written by the notebooks
scripts/          download_data.py
```

## How I built this with an AI coding agent

I built this project with Claude Code (Anthropic's coding agent) working in my
terminal, from a written brief that set the question, the build plan and the rules:
split by time, report whatever the data shows, no fabricated numbers, real commit
timestamps.

**What the agent did.** It read the existing folder and confirmed the old parser's bug
against the data, surveyed all 1,169 match files for edge cases (super overs, miscounted
overs, penalty runs, no-results), wrote the new parser and its tests, drafted and ran
the three notebooks, ported the code into `src/cricket_ba/`, and wrote the CI workflow
and a first draft of this README.

**What I decided.** The question and the framing (that the case for the Bayesian model is
uncertainty and thin-data players, not accuracy), the time split, the rule that every
number must come from code run in this repository, and the move from conda to uv when
the conda-forge solve was still running after many minutes.

**Where the agent was wrong and had to be caught.**

* The posterior predictive check first reported that the observed spread of batter rates
  sat at the 0th percentile of the replicated spread. The cause was an indexing bug: the
  replicated wicket counts were grouped in order of first appearance (`pd.factorize`)
  but divided by ball counts in sorted order. The implausible number in a short trial run
  was the clue; the fix is in `03_hierarchical_pymc`.
* A parser test assumed a missing winner is stored as `None`. Under pandas 3 it is `NaN`,
  and the test failed after the environment changed.
* A model test looked for the prior draws in the wrong part of PyMC's `InferenceData`,
  and a dataclass used a mutable default. Both failed when the tests first ran and were
  fixed before committing.
