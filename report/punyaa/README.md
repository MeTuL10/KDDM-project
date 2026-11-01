# Rental-Rate Prediction with Gradient-Boosted Ensembles: Methods and Experiments

This folder documents a leakage-safe pipeline for predicting HDB monthly rents,
the sequence of experiments run on it, and what the evidence does and does not
support. The implementation is in [`punyaa/`](../../punyaa/README.md) and is covered
by 76 unit tests.

## Summary

The test period lies entirely beyond the labelled data, so every model is validated
chronologically and every target-derived feature uses only earlier months. A
blend of three LightGBM and three CatBoost models (v2) reached a validation RMSE of
**463.155** and a public leaderboard score of **434.81**. Adding a LightGBM model of
rent per square metre (v3) lowered the validation RMSE to **462.906** and the
leaderboard score to **434.70**. Twenty-five experiments and analyses tested which further
changes help. Most did not, and an irreducible-noise analysis indicates that on rows with an
exact twin the model is already consistent with the noise level. Backtests in
rising markets show that the pipeline lags a market that moves, mostly through a
level error.

Leaderboard scores were supplied by the author and are public-leaderboard only.
Validation scores are model-selection scores, as explained in
[`problem_and_protocol.md`](problem_and_protocol.md).

## Reading order

| # | Document | Question it answers |
| ---: | --- | --- |
| 1 | [`problem_and_protocol.md`](problem_and_protocol.md) | What is being predicted, how is it validated, and how much error is irreducible? |
| 2 | [`data_processing.md`](data_processing.md) | How are the inputs cleaned, joined, and made leakage-safe? |
| 3 | [`model_experiments.md`](model_experiments.md) | Which models were built, how were they blended, and how did they score? |
| 4 | [`improvement_experiments.md`](improvement_experiments.md) | Which changes to the v2 ensemble helped? |
| 5 | [`extrapolation_experiments.md`](extrapolation_experiments.md) | Can a different leaf model or a calibration step improve the v2 ensemble? |
| 6 | [`results_summary.md`](results_summary.md) | What does the full set of experiments support, and what does it not show? |

The selected pipeline and the moving-market backtests are summarized below in this
file.

## Headline results

| Version | Validation RMSE (W1) | Validation MAE | Public leaderboard |
| --- | ---: | ---: | ---: |
| v1, direct models only | 463.716 | 335.000 | not reported |
| v2, six-model blend | 463.155 | 334.492 | 434.81 |
| v3, v2 plus rent-per-square-metre LightGBM | **462.906** | 334.961 | **434.70** |

W1 fits on 117,988 rows before May 2024 and validates on the 32,012 rows from May
2024 onward. Early stopping and blend weights use these labels.

## Selected pipeline

The task is tabular, mixes high-cardinality location categories with a few numeric
attributes, and has 150,000 labelled rows, a regime that gradient-boosted trees
handle natively. The ensemble is therefore built from LightGBM and CatBoost.
XGBoost (validation RMSE 477.348) and scikit-learn's standard GBM (470.333) were
tested as challengers under the same protocol and did not displace the detailed
LightGBM (468.262) or CatBoost (463.549); see
[`improvement_experiments.md`](improvement_experiments.md).

**Inputs.** Twenty-one features, eight of them categorical: structural attributes,
native location categories, MRT and mall distance, nearest-place names, an exact
24-month global rent anchor, and smoothed town-level and property-level rent
histories. Every target-derived value uses strictly earlier calendar months
([`data_processing.md`](data_processing.md)).

**Ensemble.** Three LightGBM models (31, 63, and 127 leaves) and three CatBoost
models (seeds 42, 2026, and 5228) are blended with non-negative weights bounded to
[0, 0.85] and selected on W1. The five components with positive weight are refitted
on all 150,000 labelled rows with 5% more iterations than their validation runs
selected. An optional residual model against an official median-rent file was not
used because the file was absent, and an optional set of 6, 12, and 24-month recent
histories stayed opt-in because its quick run was worse (468.494 against 465.295 for
the core profile).

| Component | Validation RMSE | Blend weight |
| --- | ---: | ---: |
| LightGBM, regularized | 470.282 | 0.1469 |
| LightGBM, balanced | 469.112 | 0.0000 |
| LightGBM, detailed | 468.262 | 0.0378 |
| CatBoost, seed 42 | 464.154 | 0.1791 |
| CatBoost, seed 2026 | 463.883 | 0.2509 |
| CatBoost, seed 5228 | 463.549 | 0.3854 |
| **Selected blend (v2)** | **463.155** | 1.0000 |

The blend has MAE 334.492 and sits 0.561 RMSE below the v1 blend. CatBoost carries
about 81.5% of the weight. The balanced LightGBM gets zero weight because it adds
no signal beyond the other components. Across three six-month rolling windows the
detailed LightGBM was best every time (mean RMSE 482.931, against 483.639 for
balanced and 484.743 for regularized). Those windows overlap W1 and use their own
labels for early stopping, so they are a stability check and not independent
estimates.

The v2 file scored 434.81 on the public leaderboard. v3 blends 16.0657%
rent-per-square-metre LightGBM with 83.9343% of v2 and scored 434.70. Its local gain
of 0.249 RMSE has a month-resampled 95% interval of about [-0.002, 0.486] and comes
with 0.470 worse MAE, so it is a small and uncertain improvement
([`improvement_experiments.md`](improvement_experiments.md)).

## Moving-market backtests

The test months run up to 16 months past the last label (2025-03 to 2026-07), and
trees are piecewise constant, so a tree model that reaches past its data can only
repeat a level it has already seen. To test whether the pipeline copes when rents
move, the evaluation was repeated at three earlier forecast origins. Each fits on
every row before the origin and validates on the next twelve months, with the same
features and the detailed LightGBM. The code is
[`punyaa/market_regimes.py`](../../punyaa/market_regimes.py).

| Origin | Fit rows | Mean-rent change, first to last month | Control RMSE | Control bias | Damped-index RMSE | Oracle-index RMSE | Twin-row noise SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-01 | 37,619 | +30.0% | 629.9 | -312.2 | 602.0 | 500.2 | 471.3 |
| 2023-01 | 70,183 | +5.5% | 629.1 | -152.0 | 613.0 | 604.6 | 588.5 |
| 2024-01 | 105,837 | +2.0% | 493.9 | +14.2 | 493.2 | 490.6 | 479.3 |

Bias is the mean prediction minus actual. The control is the pipeline as it stands.
The oracle column is explained under finding 2 and the damped column under finding
4. The noise column is the
irreducible-noise estimate of [`problem_and_protocol.md`](problem_and_protocol.md)
Section E, recomputed on each window and valid for the quarter of rows that have an
exact twin.

1. **The pipeline lags a rising market.** In the 2022 window the control
   under-predicts by 312 on average and by 630 in the last three months, and the
   error grows with the forecast horizon, as it would if the model were repeating
   the level of the last month it saw. The flat 2024 window shows none of this.
2. **Most of that error is a level error.** The oracle deflates the target by a
   monthly market index that includes the validation months, which the real problem
   cannot know, so it is an unattainable bound. With it, the 2022 RMSE falls from
   629.9 to 500.2, within 29 of the noise estimate, and removes 130 of the control's
   159 excess over the noise estimate. The cross-sectional part of the model is not
   the main source of the error.
3. **The earlier remedies do not repair it.** Dropping the 24-month lag changes RMSE
   by at most 2.6 at any origin, so the lag does not carry the level. Piecewise-linear
   leaves change RMSE by -2.3, -1.6, and +0.8. Monotone constraints helped in 2022
   only (-27.8, with 2,599 boosting iterations against 548 for the control) and cost
   2.6 and 3.3 at the other origins. Removing the history features together with the
   three time features costs 51 to 324, so the time features carry most of the level.
   The reading that a tree places its last date split at the latest date it saw, and
   so repeats that level, fits this pattern but is an inference, not something these
   experiments isolate.
4. **A projected index recovers part of the gap, with a risk.** The index is a
   hedonic month effect (sparse least squares of log rent on month, property, flat
   type, flat model, and log floor area, with a small ridge term). Three projections
   past its last month were run with settings fixed in advance: flat, linear (the
   slope of the last six months), and damped (the same slope with each step shrunk
   by 0.9). Linear scored 576.8, 698.9, and 493.4 across the three origins, the best
   in 2022 and the worst in 2023, where the trend faded. Flat scored 642.8, 650.9,
   and 492.9, worse than the control in both moving markets. Damped scored 602.0,
   613.0, and 493.2, better than the control at every origin but with a remaining
   bias of -274 in 2022 and +70 in 2023. CatBoost shows the same pattern (control
   629.3, 676.6, 490.4; damped 603.0, 613.2, 494.0), and damped costs it 3.5 in the
   flat year. The damped mode was chosen after seeing the three outcomes, so they
   cannot also validate it, and it is not used in any submission.
5. **The index at the end of the labelled data was close to flat.** It rose 2.8%
   over the last 12 months and its six-month slope is +0.04% per month, so a damped
   projection would add only about 9 SGD at 16 months. The same adjustment ranges
   from +9 to +151 SGD across slope windows of 6, 12, and 3 months (+51 SGD for 12),
   because the monthly index moves by more than 2 percentage points between
   neighbouring months. The
   data cannot say what happened after 2025-03.

Limits that apply to all of this: the early origins fit on fewer rows than the final
model (37,619 for 2022, with the 24-month lag missing for 8.3% of those rows), three
origins are three observations, early stopping uses each window's own labels, and
the CatBoost fits for the damped and oracle variants in 2023 and 2024 stopped at or
near the iteration cap. The wider consequences for the test period are stated once, in
[`results_summary.md`](results_summary.md) Section C.

### Leaderboard shift probe

The level of the test period can be measured with the leaderboard. Adding a constant
*c* to every prediction changes the squared error to RMSE₀² + 2*b*c + *c*², where *b*
is the mean error (prediction minus actual) and RMSE₀ is the unshifted score.
Submitting the v3 predictions shifted by +50 therefore gives
*b* = (RMSE₅₀² - RMSE₀² - 2500) / 100, and removing a bias of that size would lower
the score by about *b*² / (2 RMSE₀). At a score of 434.70, a bias of 100 would cost
11.66 RMSE (423.04 without it). The files `punyaa_probe_plus50.csv` and
`punyaa_probe_minus50.csv` in `outputs/punyaa_probe/` hold the shifted predictions,
and `solve_bias.py` in the same folder computes *b* from the returned score. The
score covers the public part of the test set only. No probe score has been recorded,
so the test-period bias remains unmeasured.

## Provenance

The feature design draws on ideas documented in the shared project repository
(a compact static feature set, named geographic context, a 24-month rent anchor, and
group-level rent history). The work reported here is the causal and smoothed
implementation of those ideas, the ensemble, the evaluation protocol, and every
experiment in documents 3 to 5 and the moving-market backtests.

## Reproducing

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v
```

The full command list, with the output location for each result, including the
backtests above, is in [`results_summary.md`](results_summary.md) Section D.
