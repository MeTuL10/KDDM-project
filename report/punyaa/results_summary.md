# Results Summary

This chapter collects every experiment in one place, states what the evidence
supports, and lists what it cannot show. Protocol, windows (W1 and W2), and the
selection caveats are defined in [`problem_and_protocol.md`](problem_and_protocol.md).
All RMSE values are in Singapore dollars; lower is better.

## A. Experiment ledger

### Stage 1: baseline models

| # | Experiment | Result (W1 RMSE) | Verdict | Detail |
| ---: | --- | ---: | --- | --- |
| 1 | Direct versus residual-over-24-month-anchor prediction | Direct 464.154 to 470.820; residual 646.597 to 662.511 | Residual models rejected, zero blend weight | [`model_experiments.md`](model_experiments.md) B |
| 2 | LightGBM capacity (31, 63, 127 leaves) | 470.282, 469.112, 468.262 | Detailed best; best in all three rolling windows | [`model_experiments.md`](model_experiments.md) D, E |
| 3 | CatBoost, three seeds | 464.154, 463.883, 463.549 | Strongest single family | [`model_experiments.md`](model_experiments.md) D |
| 4 | Bounded blend of six models (v2) | **463.155**, MAE 334.492 | Adopted; Kaggle public 434.81 | [`model_experiments.md`](model_experiments.md) D, F |
| 5 | Extra 6, 12, 24-month history windows (quick run) | 468.494 against 465.295 for the core profile | Not adopted | [`data_processing.md`](data_processing.md) E |

### Stage 2: improvements to the v2 ensemble

| # | Experiment | Result (W1 RMSE) | Change | Verdict |
| ---: | --- | ---: | ---: | --- |
| 6 | Recency weighting, LightGBM, 12 and 24-month half-lives | 470.239, 469.471 | +1.977, +1.209 | No gain |
| 7 | Recency weighting, CatBoost, 12 and 24-month half-lives | 464.447, 464.159 | +0.898, +0.610 | No gain |
| 8 | Higher-capacity LightGBM (255 leaves) | 468.994 | +0.732 | No gain |
| 9 | Property-level history removed | 472.649 | **+4.388** | Feature group retained |
| 10 | XGBoost, categorical histogram | 477.348 | +9.086 | Rejected |
| 11 | Standard GBM, one-hot encoding | 470.333 | +2.071 | Rejected |
| 12 | LightGBM on rent per square metre, blended at 16.1% | blend **462.906**, MAE 334.961 | **-0.249** | Adopted as v3; Kaggle public 434.70 |

Changes in rows 6 to 11 are measured against the matching reference model
(detailed LightGBM 468.262, or CatBoost seed 5228 at 463.549 for row 7). Row 12 is
measured against the v2 blend. Detail: [`improvement_experiments.md`](improvement_experiments.md).

### Stage 3: can a different leaf model extrapolate better?

Standalone RMSE against the detailed LightGBM control (W1 468.262, W2 448.688).

| # | Experiment | W1 | W2 | Verdict |
| ---: | --- | ---: | ---: | --- |
| 13 | Piecewise-linear leaves, regularized | 470.749 | 448.269 | Reverses sign between windows; treated as noise |
| 14 | Piecewise-linear leaves, unregularized | 494.746 | 449.977 | Rejected |
| 15 | Monotone constraints, six core signs | 473.389 | 451.472 | Rejected |
| 16 | Monotone constraints, nine signs | 473.746 | 452.284 | Rejected |
| 17 | Linear leaves and monotone constraints | 471.735 | 450.016 | Rejected |
| 18 | Level-shift and affine recalibration of the v2 blend (reported in that order) | in-sample gain 0.330 and 0.350 | applied to the other window: +0.318 and +0.415 (worse) | Does not transfer; not applied |

Largest ensemble gain from any row 13 to 17 candidate: 0.0014 RMSE. Detail:
[`extrapolation_experiments.md`](extrapolation_experiments.md).

### Stage 4: diagnostics

| # | Analysis | Result | Reading |
| ---: | --- | --- | --- |
| 19 | Irreducible noise on rows with an exact twin | Estimated SD 454.8 (W1) and 435.2 (W2); v2 scores 452.8 and 436.1 | Consistent with the model sitting at the noise level on those rows |

Detail: [`problem_and_protocol.md`](problem_and_protocol.md) E.

### Stage 5: does the pipeline follow a moving market?

Twelve-month backtests at three forecast origins (2022-01, 2023-01, 2024-01), fitted
on all earlier rows. LightGBM unless stated. Results are listed as 2022 / 2023 / 2024.

| # | Experiment | Result | Verdict |
| ---: | --- | --- | --- |
| 20 | Control model (detailed LightGBM) | RMSE 629.9 / 629.1 / 493.9; bias -312.2 / -152.0 / +14.2 | Lags a rising market |
| 21 | Earlier remedies repeated: drop the 24-month lag, drop rent history, piecewise-linear leaves, monotone constraints | Change in RMSE: lag +1.6 / -2.6 / 0.0; history +2.1 / +14.7 / +9.1; linear leaves -2.3 / -1.6 / +0.8; monotone -27.8 / +2.6 / +3.3 | No remedy works across regimes; the lag does not carry the level |
| 22 | Index-deflated target with the true (oracle) index | RMSE 500.2 / 604.6 / 490.6 | The 2022 error is mostly a level error; the oracle is an unattainable bound |
| 23 | Index-deflated target with flat, linear and damped index projections | Damped 602.0 / 613.0 / 493.2; linear 576.8 / 698.9 / 493.4; flat 642.8 / 650.9 / 492.9 | Damped limits the damage; chosen after seeing the outcomes; not adopted |
| 24 | CatBoost control, damped and oracle variants | Control 629.3 / 676.6 / 490.4; damped 603.0 / 613.2 / 494.0 | Same pattern as LightGBM; damped costs 3.5 in the flat year |
| 25 | Index adjustment implied at the test horizon, by slope window | +9 to +151 SGD (damped, 16 months) | The projection is sensitive to its window; the test-period level is not identifiable from labels |

Detail: [moving-market backtests](README.md#moving-market-backtests) in the README.

## B. What the evidence supports

1. **Rent history carries most of the usable signal.** Removing the two
   property-level history features raised the detailed LightGBM's RMSE by 4.388,
   the largest single effect among the tested changes.
2. **The choice of boosting implementation matters more than any ensemble
   detail.** Standalone RMSE ranges from 463.549 (CatBoost) to 477.348 (XGBoost), a
   spread of 13.8. Everything done afterwards to the ensemble moved the blend by
   less than 0.6.
3. **Model-side changes gave small gains at best in the flat windows.** From the
   first blend (463.716) to v3 (462.906) the total improvement is 0.810. Recency
   weighting, larger trees, different leaf models, monotone constraints, and
   recalibration all failed to improve on it.
4. **On rows that have an exact twin, the model is consistent with the noise
   level.** That suggests little room left on those rows, but it rests on a
   quarter of the rows and an interval about 25 to 33 units wide, and it cannot
   speak for the other rows.
5. **After the baseline ensemble, the only accepted change altered the prediction
   target, not the tuning.** The rent-per-square-metre model was weaker alone
   (469.648) yet lowered the blend, and it improved RMSE in 8 of 11 validation
   months, which indicates partly different errors from the direct model. The
   tuning-style changes (rows 6 to 8, 10, and 11) did not help.
6. **The pipeline lags a rising market, and the lag is a level error.** In the
   2022 backtest the control under-predicts by 312 on average and 630 in the last
   three months. Supplying the true market level cuts RMSE from 629.9 to 500.2, so
   the cross-sectional part of the model is not the main source of error. The
   24-month lag is not what carries the level: removing it changes RMSE by at most
   2.6.

## C. What the evidence does not show

* **Behaviour when the market moves.** Both validation windows, W1 and W2, sit in
  a flat market (mean rent up 2.2% across W1, from 3,096 to 3,163), so every
  result in Stages 1 to 3 says little about how a model behaves when rents move.
  The Stage 5 backtests test this directly and show that the pipeline lags a
  rising market.
* **The level of the test period.** The labels end in 2025-03 and the test months
  run 16 months past them, so the level error actually present in the test set is
  unmeasured. The market was close to flat at the end of the labelled data (the
  index rose 2.8% over the last 12 months), and only the leaderboard can say what
  happened afterwards. The [leaderboard shift probe](README.md#leaderboard-shift-probe)
  describes how to measure it; no probe score has been recorded.
* **A validated mitigation.** The damped index projection was chosen after seeing
  the three backtests, costs 3.5 RMSE for CatBoost in the flat year, and does not
  remove the bias, so it is not used in any submission. The early backtest origins
  also fit on fewer rows (37,619 for 2022) than the final model, and three origins
  are three observations.
* **Independent confirmation of any gain.** W1 selects iterations, candidates, and
  weights, and W2 overlaps it. The v3 gain of 0.249 has a month-resampled 95%
  interval of about [-0.002, 0.486], and its MAE is 0.470 worse. The leaderboard
  moved by 0.11 in the same direction, which is encouraging and small.
* **A private-leaderboard result.** Only the public score is available, and both
  public scores (434.81 and 434.70) were supplied by the author.
* **The cause of the higher error on rows without a twin.** Noise and modelling
  headroom cannot be separated for those rows.
* **A comparison between local and leaderboard RMSE.** Local scores use labelled
  2024 to 2025 months, and the public score uses different rows. The 28-point gap
  between 463.155 and 434.81 is not a measured effect of anything.

## D. Reproducing the results

| Result | Command | Output |
| --- | --- | --- |
| Unit tests (76) | `python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v` | console |
| v2 validation and submission | `python3 -m punyaa.train --feature-profile core --output outputs/punyaa_v2_submission.csv --metrics-output outputs/punyaa_v2_validation_metrics.json` | `outputs/punyaa_v2_*` |
| Rolling LightGBM windows | `python3 -m punyaa.tune_lightgbm --feature-profile core --output outputs/punyaa_lightgbm_rolling_core_metrics.json` | `outputs/punyaa_lightgbm_rolling_core_metrics.json` |
| Stage 2 experiments and v3 | `python3 -u -m punyaa.improve --submission outputs/punyaa_v3_submission.csv` | `outputs/punyaa_improvements/` |
| Stage 3 experiments | `python3 -m punyaa.extrapolation --validation-start 2024-05` (and `2024-10`; add `--cross-window` for the cross-window transfer in row 18) | `outputs/punyaa_extrapolation/` |
| Stage 4 noise analysis | `python3 -m punyaa.noise_floor --validation-start 2024-05` (and `2024-10`) | `outputs/punyaa_noise_floor/` |
| Stage 5 backtests | `python3 -m punyaa.market_regimes --output-dir outputs/punyaa_regimes` (and `--families catboost --output-dir outputs/punyaa_regimes/catboost`) | `outputs/punyaa_regimes/` |
| Stage 5 test-horizon adjustments | `python3 -m punyaa.market_regimes --test-projection --output-dir outputs/punyaa_regimes` | `outputs/punyaa_regimes/test_period_projection.csv` |

The 76 tests cover feature isolation (9), ensemble helpers (3), improvement
utilities (5), extrapolation and calibration (20), the noise estimate (11), and the
moving-market helpers (28).
They check correctness of the code, and they do not establish that any accuracy
difference is significant.
