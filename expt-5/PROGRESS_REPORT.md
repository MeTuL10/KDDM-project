# expt-5: improving on expt-2 V2 (public leaderboard 433.607)

## The base
The 433.607 submission is `expt-2/outputs/submissions/submission_ensemble_v2.csv` (V2):
official median rent per (town, flat type, quarter) as an anchor, models predict rent minus
anchor, 70 features including hierarchical Bayesian target encodings (town, block, block x
flat type, street, subzone), LightGBM + XGBoost + CatBoost bagged over 5 random folds,
blended 0.545 / 0.075 / 0.380. The step from V1 (443.000) to V2 came from the block-level
encodings. V2's own local OOT scores (e.g. LightGBM 450.31) were slightly leaky: the encodings
were computed once over all rows.

## Method
`src/expt5_lib.py` rebuilds V2's recipe with the encodings recomputed inside each split, and
scores on two holdouts: H1 (train <= 2024-06, validate 2024-07..2025-03, expt-2's OOT split) and
H2 (train <= 2023-11, validate 2023-12..2025-03, 16 months). Differences use paired bootstrap
intervals (`archived_modules/evaluation.py`). Scripts: `v2_clean_experiments.py`, `v2_improvements.py`,
`final_candidate.py`; logs in `outputs/`.

## Results (RMSE, lower is better)
| variant | H1 | H2 |
|---|---|---|
| V2 recipe, clean, LightGBM | 452.48 | 485.96 |
| + fallback anchor (town ratio) | 452.51 | 485.69 |
| + time-aware address premium | 452.23 | 485.12 (CI -1.28..-0.34) |
| + both, regularised LightGBM | 452.09 | 484.15 (CI -2.51..-1.21) |
| smoothed anchor as feature / as base | 452.93 / 453.93 | 484.95 / 486.01 (worse, dropped) |
| LightGBM 5-seed bag | 451.78 | 484.40 |
| CatBoost alone | 451.24 | 482.62 |
| **final: LightGBM bag + CatBoost, 50/50** | **450.90** | **482.68** |

XGBoost adds nothing to the blend. Hyperparameter spread across the regularisation sweep was
only 0.4 to 0.8, so tuning is not where the gain is.

## Final candidate
`outputs/submission_v2_improved.csv`: the final recipe refit on all 150,000 training rows
(LightGBM 200 trees x 5 seeds, CatBoost 350 iterations), 50,000 rows, no nulls, correlation
0.9961 with V2's predictions (mean abs difference 21.5). V2's file and the root `submission.csv`
were not touched.

## Caveats
- Local gain over clean V2 is 1.6 (H1) to 3.3 (H2) points, about 0.35% to 0.7%. The leaderboard
  may move less, the same, or not at all; the leaderboard score is the real test.
- Trees, regularisation and blend weights were chosen on the same holdouts they are reported on
  (H1 and H2 share the 2024-07..2025-03 window), so the gains are mildly optimistic.
- 2026-Q3 test rows use the 2026-Q2 anchor (V2 extrapolated a small growth term).
- Using the median-rent file's test-quarter values remains a course-rules question, as for V2.

## Leaderboard result and what it changed
`submission_v2_improved.csv` scored **434.029** against V2's 433.607. The local gains (1.6 to 3.3
points) did not transfer. Two earlier changes (V3 deeper trees, this candidate) each landed about
0.4 worse on the public score, which covers half the test rows; differences that small are
probably noise, but the gains clearly did not show up either. Lesson: change one thing per
submission, and treat sub-point local gains as unproven.

## Experiments 2 and 1 (literature-guided; all on V2's own benchmark and features)
Papers: Sigrist 2022 (GPBoost, JMLR 23(232)), Pargent et al. 2022 (regularised target encoding),
Prokhorenkova et al. 2018 (CatBoost ordered statistics). Scripts `exp2_gpboost.py` and
`exp1_catboost_tuned_te.py`; logs and JSON in `outputs/`. Learning rate 0.1 for LightGBM and
GPBoost (GPBoost re-estimates variances every round, ~10 s/round); rounds and iterations are the
best on the held-out window for every variant alike (optimistic but equal).

| variant | H1 RMSE | H2 RMSE | vs LightGBM on V2 encodings |
|---|---|---|---|
| LightGBM on V2 encodings (reference) | 453.40 | 486.22 | - |
| **Exp 2: GPBoost random effects replacing the encodings** | 458.66 | 494.20 | +5.26 / +7.98, CIs exclude 0: worse |
| Exp 1a: V2 encodings with shrinkage from GPBoost variances (conditional) | 454.06 | 486.31 | not better |
| Exp 1a: same, from an intercept-only fit (marginal) | 453.37 | 485.53 | -0.03 (n.s.) / -0.69 (CI -1.30..-0.09) |
| CatBoost on V2 encodings (no raw categoricals) | **449.92** | **482.62** | -3.48 / -3.60, CIs exclude 0 |
| Exp 1b: CatBoost with raw block / block x type / street | 454.52 | 485.67 | worse than CatBoost on V2 encodings |
| Exp 1b: raw categoricals + tuned encodings | 452.55 | 484.46 | worse than CatBoost on V2 encodings |
| Exp 1c: blend GPBoost + CatBoost on V2 encodings | 451.35 | 485.13 | worse than CatBoost alone |

Findings. (1) GPBoost does not replace V2's encodings: it is clearly worse on both holdouts.
(2) The random-effects likelihood independently recovers V2's hand-set shrinkage constants:
street 16 vs 15, subzone 21 vs 20, block x type 10 vs 5; only block and town want much stronger
shrinkage (67-90 and 90-190 vs 10 and 30), and using them gives no reliable gain. This explains why
V2 worked and says there is little left in tuning its encodings. (3) Raw high-cardinality
categoricals hurt CatBoost, with or without the encodings. (4) CatBoost on V2's own encodings is
the strongest single model (about 3.5 points better than LightGBM on both holdouts), which matches
V2's blend giving it the largest weight. Neither experiment improves on V2.

## V2 reproduced exactly; one-change candidate (blend weights only)
`src/v2_reproduce.py` re-runs expt-2 V2's 5-fold training (same features parquet, folds, seeds,
hyperparameters) and saves each model's out-of-fold and test predictions to `outputs/`
(`v2_oof_preds.csv`, `v2_test_preds_by_model.csv`). It matches V2's own submission file exactly
(correlation 1.000000, max difference 0.00) and V2's reported OOF RMSE (470.99), so the 433.607
base is recovered and any future blend can be tried without retraining. V2's file and the root
`submission.csv` were not touched.

`src/blend_weights_evidence.py` fits LightGBM / XGBoost / CatBoost with V2's exact settings and
encodings recomputed inside each split, on both time holdouts. Single models (H1 / H2):
LightGBM 452.48 / 485.89, XGBoost 452.91 / 484.69, CatBoost 450.43 / 482.57. V2's own weights
(0.545 / 0.075 / 0.380) give 450.87 / 483.42. The best weights over the simplex are
(0.0, 0.2, 0.8): 450.36 / 482.33, i.e. -0.51 (CI -0.86..-0.18) and -1.09 (CI -1.38..-0.78) against
V2's weights. The surface is flat: any set with 0.75 to 0.85 on CatBoost is within 0.03 of the best.

`src/make_cbweights_submission.py` applies (0.0, 0.2, 0.8) to V2's saved test predictions:
`outputs/submission_v2_cbweights.csv` (50,000 rows, no nulls, correlation 0.99901 with V2, mean abs
difference 16.2, max 331). This changes only the weights, so a leaderboard difference isolates
them. Expected effect is small (0.5 to 1.1 locally) and may not show on the public score.

## External unit-level data (floor, unit details)
Checked, not pursued. The official rental registry (data.gov.sg "Renting Out of Flats", about 210K
rows, Jan 2021 to Aug 2026) has the same fields as ours and no floor or unit field; its only extra
value is the answer key (the expt-3 lookup). HDB Property Information is block-level, and the
resale data's storey ranges describe a different market (expt-4's resale features made out-of-time
RMSE worse: 464 vs 451). HDB describes the rent data as owner-declared and unverified, so part of the
within-address variation is not a floor effect.

## Would synthetic rows in the test timeframe help? (learning curve and recency, `src/learning_curve.py`)
V2's recipe, encodings recomputed inside each training set, LightGBM with V2's settings and 106
trees fixed, validation window fixed at 2024-10..2025-03 (17,198 rows).

Learning curve (random subsamples of the 132,802 earlier rows): 10% 465.77, 25% 457.34, 50% 451.67,
75% 446.99, 100% 443.64. Not flat: roughly 6 to 8 RMSE per doubling of rows and no sign of
saturation, about 3.4 for the last 33k rows. My prior that more data would not help was wrong.
The likely reason is that the block and block-by-flat-type encodings (14.8k block-by-type levels,
about 8 rows each) are still estimated from few transactions.

Recency (training ends N months before the window, against a same-size random control drawn from
all earlier rows): cost of older data -0.17 at 3 months, -0.08 at 6, +0.92 at 9, +1.84 at 12,
+4.17 at 15. Recent rows barely matter within about 6 months and matter increasingly after 9. Over
the test set's 0 to 17 month horizon that averages about 1 RMSE, rising to 2 to 4 for the latest
months, which bounds what real in-timeframe rows could add through recency.

Synthetic rows built from existing data carry neither benefit: the volume gain comes from real
transactions at specific blocks, and recency from real recent rents. A model-labelled row only
repeats the model's own prediction, and re-dated old rents are what the median-rent anchor already
supplies. Legitimate ways to use the volume finding: better priors for the sparse block-level
encodings (nearest-neighbour blocks instead of town or street), and recency weighting (small).

## What could a 411 team have done? (data checks, `train.csv`/`test.csv` only)
- A unit fingerprint (block, flat type, flat model, floor area, lease start) has exactly as many
  groups as block x flat type (14,828 each; mean 10.1 rows): within a block every flat of a given type
  shares the same model, area and lease date. 99.2% of test rows have a block-type seen in train.
  The columns cannot separate units within a block-type, so finer encodings add nothing.
- Among flats in the same block, flat type and month (cells with 2+ rows, 26.8% of rows), rent has a
  pooled SD of 464 (457 over the last 12 months of train). Our local RMSE (about 450) is already of that
  order, so a legitimate model without unit-level information has little room below it. A score about
  5% under ours would need information that separates flats within a block (not in these columns),
  label-adjacent data (as in expt-3's lookup), or public-leaderboard probing. Caveat: the SD is
  measured on multi-row cells only, and the test set seems kinder than our holdouts (leaderboard
  about 15 below local), so this is a strong hint, not a proof.

## Experiment 3: external resale signals and neighbour-block priors (`src/exp3_external_neighbours.py`)
Data: expt-4's 201K data.gov.sg resale transactions (2017-01..2025-02), joined to rental blocks after
street-name normalisation (98% of rental rows have block resale history). Resale rows before each
holdout's cutoff only. New features: block and block-by-type resale price premium over the town /
flat type / quarter norm (shrunk), mean storey, transaction count; and the rental residuals of the 5
and 15 nearest other blocks of the same flat type (neighbour priors). V2's recipe is the base;
LightGBM and CatBoost with early stopping on the held-out window for every variant alike.

| variant (blend 0.2 LightGBM / 0.8 CatBoost) | H1 | H2 |
|---|---|---|
| base | 450.30 | 482.53 |
| + resale signals | 449.74 (-0.56, CI -1.01..-0.16) | 482.38 (-0.15, n.s.) |
| + neighbour priors | 450.05 (-0.25, n.s.) | 483.34 (+0.82, worse) |
| + both | 449.88 (-0.42, CI -0.87..-0.01) | 482.75 (+0.22, n.s.) |

LightGBM alone: resale -0.71 / -1.27, both -0.56 / -0.98 (intervals exclude 0), but CatBoost, the
stronger model, gains at most 0.5 on H1 and nothing on H2. The mean-storey feature (the nearest
thing to a floor proxy) took 0.5% of LightGBM's gain; the block-by-type resale premium took 6 to 10%.

Verdict: the "real modelling" ideas I estimated at 5 to 10 points deliver under 1 point, so that
estimate was too optimistic. Seed bagging (0.2 to 0.5) and CatBoost-heavy weights (0.5 to 1.1) add
about 1 to 2 more. Together that is nowhere near the pre-set bar of about 3 points on both holdouts,
so no new candidate was built; `submission_v2_cbweights.csv` remains the only pending submission.
Block-level resale information confirms the earlier conclusion: the remaining error is mostly
unit-level noise that block-level data cannot resolve.

## Leaderboard log (public score, lower is better)
| submission | public score | vs V2 | local evidence at the time |
|---|---|---|---|
| root pipeline, tuned round 2 (before the address feature) | 451.42 | +17.8 | validation 456.70 |
| expt-2 V1 (tri-model, no block encodings) | 443.000 | +9.4 | OOT 457.61 (blend) |
| **expt-2 V2** | **433.60708** | - | OOT 449.26 (leaky), clean about 452 |
| expt-2 V3 (deeper trees, Optuna) | 434.053 | +0.445 | OOT 447.80 (leaky) |
| expt-5 `v2_improved` (address premium, regularised, LightGBM bag + CatBoost) | 434.02878 | +0.42 | -1.6 / -3.3 on two clean holdouts |
| expt-5 `v2_cbweights` (V2 with CatBoost-heavy weights only) | 433.63052 | +0.023 | -0.5 / -1.1, CIs excluding 0 |
| expt-5 `v2_honest` (V2 with an honest forecast-time test anchor: frozen at 2025-Q1, +0.55%/quarter drift) | 433.65480 | +0.048 | honest-anchor cost +4.6 locally (Experiment 6) |
| Metul's AutoGluon with two weighted residual layers (user-reported) | 436.28615 | +2.68 | validation 468.057 for the one-layer stage (`report/metul/model_experiments.md`) |
| expt-3 registry lookup | 173.19 | n/a | a label lookup, not a model; excluded |

Reading: only the V1 to V2 step (block-level encodings, about 9 points) is visible on the public score.
Local gains of 0.5 to 3 points have not transferred in three tries; the public score covers half the
test rows and unit-level noise dominates, so differences this small cannot be seen there. V2 is the
plateau for these methods. Where the competition allows several final submissions, V2 and
`v2_cbweights` are a reasonable pair (same public score, mean absolute prediction difference 16).

## Experiment 4: trimming both tails of the training data (`src/exp4_tail_trimming.py`)
Rows cut from both ends, ranked within flat type, thresholds from training rows only, validation left
complete, encodings rebuilt on the trimmed rows. Two definitions of "extreme": `anchor` (rent minus the
median-rent benchmark) and `oof` (out-of-fold residual of a first-pass LightGBM). V2's recipe, blend of
0.2 LightGBM / 0.8 CatBoost, early stopping on the held-out window for every variant alike.

| trim per end (rows removed) | anchor H1 / H2 | oof H1 / H2 |
|---|---|---|
| none (base) | 450.30 / 482.53 | same |
| 0.5% (1%) | 452.84 / 485.13 (+2.5 / +2.6) | 452.05 / 484.27 (+1.8 / +1.7) |
| 1.0% (2%) | 454.75 / 487.50 (+4.5 / +5.0) | 452.43 / 485.06 (+2.1 / +2.5) |
| 2.5% (5%) | 459.13 / 493.09 (+8.8 / +10.6) | 455.07 / 487.80 (+4.8 / +5.3) |

All twelve differences are worse with intervals excluding zero, rising steadily with the share removed.
The mechanism is visible in the split by validation row: RMSE on the central 90% improves (H1 328.06 to
318.93 at the heaviest anchor trim) but RMSE on the outer 10% worsens by more (1033 to 1096), and the
outer 10% carries about half of the total squared error. Removing the tails teaches the model a
narrower spread, and the test set has the same tails. Not adopted. Consistent with the root pipeline's
earlier outlier-exclusion test (457.37 to 458.22) and with the robust-loss results.

## Experiment 5: robust training losses inside V2's recipe (`src/exp5_robust_loss.py`)
Residual target (rent minus anchor), nested encodings, early stopping on validation RMSE, paired
bootstrap against the squared-loss fit of the same model. "+meanfix" shifts predictions by the mean
training residual (removes the location bias a robust loss adds under skewed noise). The training
residual is nearly symmetric here (skew -0.25 on H1, -0.10 on H2), so this is a fair setting for Huber.

| loss (RMSE difference vs squared, H1 / H2) | raw | mean-corrected |
|---|---|---|
| LightGBM Huber delta 300 | +2.40 / +4.71 | +0.64 / +3.54 |
| LightGBM Huber delta 600 | +0.74 / +2.13 | -0.05 / +1.18 |
| LightGBM Huber delta 1000 | -0.40 / +0.28 | -0.50 (CI -1.09..-0.04) / +0.04 (n.s.) |
| LightGBM Fair c 200 / c 500 | +2.39 / +3.09 and +1.13 / +2.36 | +1.19 / +2.60 and +0.31 / +1.93 |
| LightGBM L1 | +3.15 / +4.89 | +1.25 / +4.29 |
| CatBoost Huber delta 600 | +1.32 / +1.80 | +0.36 / +1.00 |

Squared-loss references: LightGBM 452.48 / 485.89, CatBoost 450.02 / 482.52. Verdict: robust losses do
not beat squared loss. The more robust the loss, the worse the RMSE. Mean-correction recovers some of
the damage but not all, so bias is only part of the problem. The best case (Huber delta 1000, about
2.2 residual SDs, nearly squared loss) is a tie: -0.5 on H1, 0.0 on H2. Large residuals here carry
signal (block premiums, high-rent segments), so down-weighting them loses information.

## Experiment 6: what an honest (forecast-time) anchor costs V2's recipe (`src/exp6_honest_anchor_validation.py`, `src/honest_lib.py`)
Motivation: Beina's report (`report/beina/model_experiment.md` section 6) shows that the median-rent file's values for
quarters after the last labelled quarter are aggregates of the very rents being predicted. V2, and every holdout result in
this folder before this section, used those published values. Honest variants (cutoff = last quarter fully before the
validation window: H1 2024-Q2, H2 2023-Q3): `nodrift` freezes each town x flat-type benchmark at the cutoff and carries it
forward; `drift4` also multiplies by exp(g x quarters ahead), g = mean quarterly log growth of a national index
(geometric mean of the 3-room to executive national medians) over the 4 quarters to the cutoff. My g reproduces Beina's
reported values (+0.78% at 2024-Q2, +0.55% at 2025-Q1). Everything else is V2's recipe, blend 0.2 LightGBM / 0.8 CatBoost.

| holdout | published (leaky) | honest, no drift | honest, drift4 | g at cutoff |
|---|---|---|---|---|
| H1 (validate 2024-07..2025-03) | 450.30 | 456.57 (+6.27, CI +5.28..+7.35) | 454.85 (+4.55, CI +3.65..+5.43) | +0.78%/quarter |
| H2 (validate 2023-12..2025-03) | 482.53 | 487.48 (+4.95, CI +4.05..+5.97) | 764.71 (+282.2) | +5.05%/quarter |

Reading: (1) the published medians flatter our earlier local numbers by about 5 RMSE, matching Beina's measurement
(+5 locally, +0.5 on the leaderboard). (2) Drift helps when the recent market is flat (H1: -1.72 vs no drift; mean
error goes from +47.6 to +1.8) and is catastrophic when the trend at the cutoff is steep (H2, 2023-Q3 followed the
2022-23 surge: extrapolating +5.05%/quarter over 6 quarters over-predicts by 885 by 2025-Q1). The drift rule is
regime-dependent, consistent with Punyaa's finding that projected indices are fragile (README, moving-market
backtests). At the real test cutoff (2025-Q1) g is +0.55%/quarter, the flat-market case.

## Honest V2 submission (`src/honest_v2_submission.py`)
`outputs/submission_v2_honest.csv`: expt-2 V2 with ONE change, the test rows' median_rent_benchmark,
nat_median_rent_benchmark and median_rent_ratio replaced by the benchmark frozen at 2025-Q1 and drifted by +0.55% per
quarter (see Experiment 6). Same features parquet, folds, seeds, hyperparameters and V2 blend weights
(0.545 / 0.075 / 0.380) otherwise. 50,000 rows, finite. Against V2's predictions: correlation 0.99057, mean difference
+26.1, mean absolute difference 50.5, max 1,208; the mean difference is +0.0 in 2025-Q1 (unchanged rows), +4.7 and +7.1
in 2025-Q2 and Q3, and about +36 to +42 from 2025-Q4 on (the drifted benchmark is higher than the published medians in
later quarters: mean 3,310.6 vs 3,280.9 in 2026-Q2). V2's file and the root `submission.csv` are unchanged (checksums
verified). Public leaderboard score: not yet submitted.

## Final submission decision
The team's final submission is `expt-5/outputs/submission_v2_honest.csv` (public score 433.65480): expt-2 V2 with a
forecast-time (honest) test-quarter anchor. It scored 0.048 worse than V2's 433.60708, which is inside the noise of the
half-test public score, and it removes the use of test-period aggregates.

## Error decomposition of the root pipeline (validation fold, shipped address-feature run; used in the final paper)
Rows with rent <= 2,100: 1,096 of 16,961 (6.46%), RMSE 1,090.19; all other rows (93.54%): RMSE 369.94; overall RMSE 452.56.
Shares of squared error: 0.0646 x 1,090.19^2 = 76,800 (37.5%) and 0.9354 x 369.94^2 = 128,000 (62.5%), total 204,800 = 452.56^2.
(An earlier prototype of the address feature gave 368.82 for the normal-rent bucket; progress-report.md and main.ipynb were
corrected to the shipped run's 369.94.)
