# Model Experiments

This document records every model trained, how it was validated, and the results. It covers the scripts `model_training.R` (main models), `predict_test.R` (final submission), `paper_ann.R` (reference-paper models) and the experiment scripts listed in section 13. RMSE is in SGD of monthly rent throughout; lower is better.

## 1. Validation rules

### 1.1 Splits

| Split | Fit on | Evaluate on | Rows (fit / evaluate) | Used for |
|---|---|---|---|---|
| **Out-of-time (OOT) holdout** | 2021-01 to 2024-06 | **2024-07 to 2025-03** | 124,016 / 25,984 | Main score. It mirrors the test set, which lies in the future (test is 0–16 months after train; OOT is 1–9 months after its fitting window). |
| **Tuning window** | 2021-01 to 2023-12 | 2024-01 to 2024-06 | 105,837 / 18,179 | Hyperparameter search, early stopping (number of rounds), blend weights, network size |
| **5-fold CV** | 4/5 of all train rows | the other 1/5 | 120,000 / 30,000 per fold | Stability check. Folds are random, stratified by rent decile, so this is **not** a forecast and its scores are optimistic about time drift. |

### 1.2 Rules that keep the holdout honest

1. **The OOT months are never used to choose anything.** Hyperparameters, rounds, network size and blend weights all come from the tuning window, which lies entirely before the OOT period. *Exception:* the benchmark drift window (section 6.3) was compared on the OOT holdout, among 4 options.
2. **Target encodings are fitted inside each split** on its fitting rows only. Fitting rows get inner 5-fold out-of-fold encodings; evaluation rows get encodings from all fitting rows (`data_process.md`, section 9).
3. **Final fits** retrain on the whole fitting window with 1.1 × the early-stopped number of rounds, since they have more rows than the tuning fit.
4. **Forecast-time information only (honest evaluation, from section 6 on).** Each quarter's median-rent benchmark is computed from that quarter's rents, so the OOT holdout uses medians up to 2024-Q2 only, carried forward with drift, exactly as the test set uses medians up to 2025-Q1. Results before section 6 used the published OOT-quarter medians and are **optimistic by about 5 RMSE**.
5. The same seed (5228) fixes the CV folds in every script, so all models are compared on identical folds.

## 2. Modelling approach

All main models predict the **gap between rent and the town × flat-type × quarter median rent benchmark** (`rent − median_rent_benchmark`); predictions add the benchmark back. The benchmark absorbs the market level (rents rose 53% over train, `exploratory_data_analysis.md`, section 4), which tree models cannot extrapolate.

Features are all numeric and categorical columns of `train_feature`, plus 7 in-fold target encodings (block × room, block, street, subzone, town, flat model, town × flat model). Normalisation: none for the tree models (scale-invariant); Ridge standardises internally (`glmnet`).

## 3. Main models (`model_training.R`)

### 3.1 Models and tuning

| Model | Implementation | Tuning |
|---|---|---|
| Median rent benchmark only | the benchmark itself | — |
| Ridge | `glmnet`, alpha = 0, numeric + encodings + one-hot town, flat model, region | λ by `cv.glmnet` internal 5-fold CV |
| LightGBM | `lightgbm` 4.7, native categoricals | 12-trial random search (learning rate 0.05; leaves, depth, min data in leaf, feature/bagging fraction, L2); early stopping at 100 rounds |
| XGBoost | `xgboost` 3.2, `hist`, integer-coded categoricals | 10-trial random search (eta 0.05; depth, min child weight, subsample, column sample, λ) |
| CatBoost | `catboost` 1.2.10, native categoricals | 6-trial random search (learning rate 0.08; depth, L2, random strength, bagging temperature) |
| Weighted blend | non-negative weights summing to 1 (softmax), minimising RMSE | OOT: weights fitted on tuning-window predictions; CV: weights fitted on out-of-fold predictions |

Tuning spread was small (tuning-window RMSE range): LightGBM 513.67–515.57, XGBoost 512.10–515.55, CatBoost 511.65–512.67. Early stopping chose small models.

| Model | Selected parameters | Final rounds |
|---|---|---|
| LightGBM | 63 leaves, max depth 8, min data in leaf 20, feature fraction 0.64, bagging fraction 0.66, L2 0.71 | 106 |
| XGBoost | max depth 6, min child weight 50, subsample 0.78, column sample 0.71, λ 0.93 | 244 |
| CatBoost | depth 6, L2 4.8, random strength 1.26, bagging temperature 0.31 | 419 |

### 3.2 Results (published OOT-quarter medians, COE and stock features included)

| Model | Tuning window | OOT holdout | 5-fold CV |
|---|---|---|---|
| Median rent benchmark only | 554.49 | 494.99 | 509.23 |
| Ridge | 514.66 | 451.95 | 474.45 |
| LightGBM | 513.67 | 452.29 | 472.94 |
| XGBoost | 512.10 | 450.97 | 471.68 |
| CatBoost | **511.65** | 450.27 | 471.48 |
| **Weighted blend** | — | **449.65** | **471.16** |

Blend weights: OOT ridge 0.18, LightGBM 0.04, XGBoost 0.35, CatBoost 0.44; CV 0.15 / 0.01 / 0.40 / 0.44. The tuning window scores higher (worse) than OOT because January to June 2024 was a harder period. The CV blend is slightly optimistic because its weights are fitted on the same out-of-fold predictions.

## 4. Reference paper models (`paper_ann.R`)

Wang, Chan, Wang & Chang (2016), *Predicting Public Housing Prices Using Delayed Neural Networks*.

| Part | Paper | Implementation here |
|---|---|---|
| 1. Static ANN | 6-h-1 feed-forward network (h = 8, 10, 15), tanh hidden layer, Levenberg-Marquardt, 60:20:20 split, several restarts, lowest MSE kept | Same; Levenberg-Marquardt coded with MATLAB `trainlm` defaults (mu 1e-3, ×0.1 / ×10, stop after 6 validation failures), inputs and target scaled to [−1, 1]. 6 inputs: rooms, floor (block `max_floor` stands in; unit storey is not in the data), floor area, elapsed lease, distance to the nearest school, distance to the nearest MRT. Two targets: rent as in the paper, and rent − benchmark. |
| 2. Delayed NN | time-series network forecasting the HDB resale price index from lagged values and macro inputs | Autoregressive network (delays 1:2, 10 hidden units) on quarterly log changes of a national rent index (2005–2024). Macro inputs are unavailable before 2021, and log changes are used because a tanh network cannot extrapolate above record levels. |

Results. The full run was stopped at the user's request after the tuning stage; OOT and CV figures come from a quick run (5 epochs):

| Model | Tuning window (full training) | OOT (quick run) | 5-fold CV (quick run) |
|---|---|---|---|
| Paper ANN, rent target (6-15-1) | 712.13 | ~717 | ~669 |
| Paper ANN, rent − benchmark target (6-15-1) | 536.30 | ~480 | ~498 |
| *CatBoost (section 3)* | *511.65* | *450.27* | *471.48* |

| Benchmark for the OOT quarters, forecast from data up to 2024-Q2 | Benchmark-only OOT RMSE |
|---|---|
| Published medians (not available at forecast time) | 494.99 |
| Delayed NN forecast (part 2) | 502.40 |
| Naive: last quarter carried forward | 498.67 |
| Town × flat-type linear trend (last 8 quarters) | 536.81 |

**Conclusion:** the paper's static network lacks a time input (its data covered 15 months), so with rent as the target it cannot follow the rent trend. With the residual target it lacks location detail and trails CatBoost by about 25 RMSE on the tuning window. The delayed network's forecast is no better than carrying the last quarter forward.

## 5. Two-stage model: national time series + residual model

The idea: stage 1 forecasts the national average monthly rent from time, COE and stocks; stage 2 (LightGBM) predicts rent − national average from location and flat features (no time, COE, stock or median columns). Same OOT split and LightGBM settings.

| Stage 1 (national level) | Stage-1 error (monthly average) | Stage 1 alone | Two-stage | + median features in stage 2 |
|---|---|---|---|---|
| *Current: town × flat-type median as baseline* | — | 494.99 | **452.19** | (built in) |
| Linear time trend | 344.3 | 736.43 | 556.33 | — |
| Ridge on time + COE + stocks | 320.8 | 726.17 | 559.73 | 569.47 |
| Last 3 months carried forward | 46.7 | 652.65 | 461.99 | 455.56 |
| Actual national average (oracle) | 0 | 650.91 | 459.07 | 453.81 |

**Training on 2023-01 onwards only** (skipping the 2021–22 surge; 53,833 fitting rows):

| Model | Stage-1 error | Two-stage OOT |
|---|---|---|
| Current approach | — | 459.12 (vs 452.19 on full data) |
| Linear trend | **37.2** | 463.69 |
| Ridge on time + COE + stocks | 54.6–174.7 (unstable with 18 points) | 462.20–464.87 |
| Last 3 months carried forward | 46.7 | 465.75 |
| Oracle + median features | 0 | 460.81 |

**Conclusion:**
- Fitted over 2021–24, the trend forecasts keep projecting the 2022–23 surge: they predict S$3,600 for 2025-03 against an actual S$3,163.
- Restricting training to the flat period fixes stage 1, but halves the data, so stage 2 loses 3–7 RMSE.
- Even with a perfect stage 1, a single national level misses town and flat-type differences. The town × flat-type median is the better baseline.

## 6. Honest evaluation: median-rent leakage

### 6.1 The issue

`sg-hdb-median-rent.csv` runs to 2026-Q2, overlapping the test period. A quarter's median is an aggregate of that quarter's rents, so using it for future rows uses information unavailable at forecast time. From here on the OOT holdout uses only medians up to 2024-Q2, and the test set only medians up to 2025-Q1 (`data_process.md`, section 8).

### 6.2 Carried forward (no drift), cutoff 2024-Q2

| Model | Published medians | Carried forward |
|---|---|---|
| Benchmark only | 494.99 | 498.67 |
| Ridge | 451.95 | 457.89 |
| LightGBM | 452.29 | 458.99 |
| XGBoost | 450.97 | 457.36 |
| CatBoost | 450.27 | 458.68 |
| **Blend** | 449.65 | **456.84** |

The carried-forward benchmark goes stale. Mean error (actual − predicted) is +S$37 in 2024-Q3, +S$46 in Q4 and +S$59 in 2025-Q1: the model under-predicts as rents keep rising slowly.

### 6.3 Drift adjustment

The carried-forward benchmark is multiplied by the national rent index's average quarterly growth over the last N quarters before the cutoff.

| N (quarters) | Drift per quarter | Blend OOT | Mean error 2024-Q3 / Q4 / 2025-Q1 |
|---|---|---|---|
| 0 (no drift) | 0% | 456.84 | +37 / +46 / +59 |
| 2 | −0.35% | 459.62 | +46 / +65 / +88 |
| **4** | **+0.78%** | **454.33** | **+15 / +2 / −7** |
| 8 | +4.00% | 498.24 | −75 / −182 / −289 |

N = 4 (one year) removes the bias and is the adopted setting. N = 8 reaches back into the 2022–23 surge.

### 6.4 Kaggle check

| Submission | Benchmark for test quarters | Public leaderboard RMSE |
|---|---|---|
| Blend, published medians (leaky) | published 2025-Q1 to 2026-Q2 medians | 433.36655 |
| Blend, cutoff 2025-Q1 + 4-quarter drift (honest) | 2025-Q1 carried forward, +0.55% per quarter | 433.89670 |

The honest version costs only 0.5 RMSE on the leaderboard and is the one adopted.

## 7. Target-encoding ablation (OOT, published medians)

| Model | With encodings | Without | Worse by |
|---|---|---|---|
| Ridge | 451.89 | 471.37 | +19.47 |
| LightGBM | 452.19 | 460.48 | +8.29 |
| XGBoost | 450.93 | 459.82 | +8.88 |
| CatBoost | 450.99 | 460.73 | +9.74 |
| **Blend** | **449.90** | **458.58** | **+8.68** |

Block-level encodings carry information no other feature has. Without them, the number of rounds early stopping picks rises 2–3 × (CatBoost 371 → 1,200) and accuracy still falls.

## 8. Feature importance (CatBoost, OOT)

Permutation importance: the increase in OOT RMSE (base 451.07) when a feature or group is shuffled, averaged over 3 shuffles.

| Feature group | RMSE increase |
|---|---|
| All 7 target encodings | **+40.4** |
| Location encodings (block × room, block, street, subzone) | +34.2 |
| Median-benchmark features | +13.1 |
| Flat age and lease year | +6.2 |
| CBD distance and interactions | +5.7 |
| Flat model | +2.1 |
| MRT | +1.4 |
| Flat size | +1.1 |
| Town, region, planning area, subzone | +0.8 |
| Building height, malls, schools | +0.2 to +0.4 |
| Time | +0.1 |
| **Stocks** | **−0.06** |
| **COE** | **−0.08** |

Top single features: `te_address_room` +33.8 (24–30% of gain in all three tree models), `median_rent_ratio` +5.5, `dist_to_cbd` +2.9, `area_to_benchmark_ratio` +2.4.

**Building age check:** every block has a single lease year, so age varies only between blocks and the block encoding already contains it. Shuffling the 3 age columns costs +4.2 (+20.7 without encodings), but **refitting without them costs only +0.1** (+0.6 without encodings): location features substitute. Rent vs age is non-monotone. Flats 0–5 years old rent +S$307 above same-month, same-type flats, 21–40-year-old flats rent S$58–71 below, but 41–50-year-old flats rent +S$26 above, because the oldest estates are near the CBD.

## 9. Improvement experiments (honest OOT, 3 seeds each, run one at a time)

Baseline: cutoff 2024-Q2 + 4-quarter drift, all features. Each idea changes one thing; rounds and blend weights are re-chosen on the tuning window.

| Idea | Blend OOT (mean of 3 seeds) | Range | Change |
|---|---|---|---|
| Baseline | 454.47 | 454.28–454.57 | — |
| 1. Average the 3 seed runs | 454.26 | — | −0.20 (single models −0.4 to −0.9) |
| 2. Ratio target, rent / benchmark − 1 | 454.44 | 454.24–454.56 | −0.02 |
| 3. Recency weights (24-month half-life) | 455.60 | 455.42–455.75 | **+1.14** |
| 5. Drop COE, stock and time columns | 454.58 | 454.37–454.83 | +0.11 |

None changes the result meaningfully; the seed-to-seed spread is about 0.3. Recency weighting is reliably worse: it discards block information without fixing any drift that the benchmark does not already handle.

## 10. Noise floor

Rentals of the same unit (same block, street, floor area, flat model and lease year) in the same quarter still differ, with a standard deviation of about **S$461**. The missing drivers are storey, renovation and furnishing, and 87.5% of rents are rounded to S$100. Current errors (honest OOT about 454, Kaggle 434) are at this level, so large further gains are not expected from these features.

## 11. Final model

| Item | Choice |
|---|---|
| Model | Weighted blend of Ridge, LightGBM, XGBoost and CatBoost (section 3 hyperparameters and rounds) |
| Target | rent − town × flat-type median benchmark |
| Benchmark for test quarters | median rent up to 2025-Q1, carried forward with 4-quarter drift (+0.55% per quarter) |
| Features | all `train_feature` columns **except COE and stocks**, plus 7 in-fold target encodings |
| Blend weights | ridge 0.150, LightGBM 0.010, XGBoost 0.401, CatBoost 0.439 (from 5-fold out-of-fold predictions) |
| Training data | all 150,000 train rows |
| Honest OOT RMSE | **454.72** (mean of 3 seeds: 454.69, 454.95, 454.52; cutoff 2024-Q2 + 4-quarter drift) |
| Submission | `submission_1027.csv` (`predict_test.R`) |

COE and stocks are removed because they share only the rent trend (`exploratory_data_analysis.md`, section 4.4), their permutation importance is negative, and removing them leaves the honest OOT unchanged (section 9). The time columns are kept; dropping them as well made no measurable difference either way (section 9, idea 5).

| Model (honest OOT, mean of 3 seeds) | With COE and stocks | Final (without) |
|---|---|---|
| Ridge | 456.64 | 456.52 |
| LightGBM | 456.47 | 456.75 |
| XGBoost | 455.18 | 455.60 |
| CatBoost | 455.36 | 455.71 |
| **Blend** | **454.47** | **454.72** |

The +0.25 difference is within the seed-to-seed spread (about 0.3). The final submission (mean prediction S$3,197, range S$1,584–5,265) has not yet been scored on Kaggle; the leaderboard score of the same setup with COE and stocks was 433.90.

## 12. Summary of findings

1. The town × flat-type median benchmark plus a gradient-boosted gap model is the best design tested. National time-series baselines and the reference paper's networks are 5–270 RMSE worse.
2. Block-level target encoding is the most important component (+8.7 RMSE for the blend).
3. Tuning, target transforms, recency weighting and feature pruning each move results by under 1.2 RMSE. The model is at the data's noise floor.
4. Honest forecasting (no future medians) costs about 5 RMSE on OOT and 0.5 on Kaggle. A 4-quarter drift adjustment recovers about half of the OOT cost.

## 13. Scripts

| Script | Content |
|---|---|
| `data_cleaning.R` | feature building for train and test (`data_process.md`) |
| `eda.R` | EDA figures and statistics (`exploratory_data_analysis.md`) |
| `model_training.R` | tuning, OOT, 5-fold CV, blend (section 3); saves `model_results.csv`, `model_tuning_log.csv`, `model_best_params.rds` |
| `predict_test.R` | final fit on all train rows and the submission (section 11) |
| `paper_ann.R` | reference-paper models (section 4) |
| `experiments/two_stage_ablation*.R` | section 5 |
| `experiments/honest_oot.R`, `experiments/drift_oot.R` | section 6 |
| `experiments/te_ablation.R` | section 7 |
| `experiments/importance.R`, `experiments/age_importance.R` | section 8 |
| `experiments/improve_exp.R`, `experiments/improve_summary.R` | sections 9 and 11 |

Run the experiment scripts from the project root, with `OUTDIR` set to an output folder (and `BENCHMARK_CUTOFF=2024-Q2 BENCHMARK_GROWTH_QUARTERS=4` for the honest runs; `improve_exp.R` also takes `VARIANT` and `SEED`). They reuse sections 1–4 of `model_training.R`, which now excludes COE and stocks. Rerunning sections 3 and 5–9 therefore gives the no-COE/stock versions of those results; the tables above were produced with COE and stocks included.
