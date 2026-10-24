# Model Experiments and Validation Results

## A. Scope and evaluation protocol

This section documents Punyaa's reproducible LightGBM–CatBoost pipeline, including the initial ensemble, the updated LightGBM configurations, multi-seed CatBoost, and the final submission. 

The target is listing-level monthly rent. Root mean squared error (RMSE) is the primary metric and mean absolute error (MAE) is secondary; lower values are better. The main chronological split uses **117,988 rows before May 2024** for fitting and **32,012 rows from May 2024 onward** for validation. Both versions below use this split.

The validation period is also used for early stopping and blend-weight selection. Reported holdout results are therefore model-selection scores rather than performance on an untouched final local test set. Rolling-window LightGBM results use their own six-month periods and are not ranked directly against the main holdout or Kaggle score.

## B. Initial direct and lag-residual ensemble

The initial implementation tested LightGBM and CatBoost both as direct rent predictors and as predictors of the residual above the global 24-month rent anchor. Non-negative blend selection retained the direct models and rejected both lag-residual components.

**Table I. Initial production results on the common split**

| Component | Validation RMSE | Blend weight |
| --- | ---: | ---: |
| LightGBM, direct rent | 470.820 | 24.40% |
| CatBoost, direct rent | 464.154 | 75.60% |
| LightGBM, lag residual | 646.597 | 0.00% |
| CatBoost, lag residual | 662.511 | 0.00% |
| **Selected initial blend** | **463.716** | **100.00%** |

The initial blend achieved MAE **335.000**. In this run, the old rent anchor was more useful as an input to a direct model than as the baseline for residual prediction. This finding motivated an updated direct ensemble rather than retaining the two weak residual components.

Source: [`punyaa_validation_metrics.json`](../../outputs/punyaa_validation_metrics.json).

## C. Updated model configuration

The updated pipeline fits three LightGBM variants and three CatBoost seeds on the same core feature representation. Its feature construction is documented in [`data_processing.md`](data_processing.md).

**Table II. Production LightGBM configurations**

| Variant | Leaves | Maximum depth | Minimum child samples | L1 penalty | L2 penalty | Feature fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regularized | 31 | 6 | 80 | 0.8 | 7.0 | 0.85 |
| Balanced | 63 | 9 | 60 | 0.5 | 5.0 | 0.85 |
| Detailed | 127 | 10 | 90 | 1.0 | 9.0 | 0.75 |

All three use squared-error regression, learning rate 0.02, up to 2,600 boosting iterations, row fraction 0.85 with subsampling every iteration, 511 histogram bins, categorical smoothing 20, and categorical L2 regularization 10. Early stopping has patience of 120 iterations. The variants deliberately change several capacity and regularization settings together; they are candidate configurations rather than a one-parameter ablation.

CatBoost uses RMSE loss, depth 8, learning rate 0.035, up to 1,600 iterations, L2 leaf regularization 8, random strength 0.7, and bagging temperature 0.5. It is fitted with seeds **42, 2026, and 5228**, each using early stopping with patience 120.

The blend minimizes validation squared error, normalized by target variance, with a small penalty of 0.002 toward equal weights. Weights sum to one and are bounded between zero and 0.85. A component can consequently be excluded without forcing its predictions into the submission.

Source: [`punyaa/modeling.py`](../../punyaa/modeling.py).

## D. Updated results on the common split

**Table III. Updated production components and selected blend**

| Component | Validation RMSE | Blend weight | Best iteration |
| --- | ---: | ---: | ---: |
| LightGBM, regularized | 470.282 | 14.69% | 1,085 |
| LightGBM, balanced | 469.112 | 0.00% | 357 |
| LightGBM, detailed | 468.262 | 3.78% | 251 |
| CatBoost, seed 42 | 464.154 | 17.91% | 1,215 |
| CatBoost, seed 2026 | 463.883 | 25.09% | 1,352 |
| CatBoost, seed 5228 | 463.549 | 38.54% | 1,229 |
| **Selected updated blend** | **463.155** | **100.00%** | n/a |

Weights are rounded, and the balanced model's numerical weight is effectively zero. The updated blend achieved MAE **334.492**, improving RMSE by **0.561** and MAE by **0.509** over the initial blend. The strongest individual LightGBM improved from RMSE 470.820 to **468.262**, a reduction of **2.559**. CatBoost seed 5228 remained the strongest individual component at 463.549.

The blend assigns approximately 18.47% of its total weight to LightGBM and 81.53% to CatBoost. The weaker regularized LightGBM receives more weight than the detailed variant, showing that standalone RMSE and contribution to a blend can differ. This weighting is a validation-selected result, not proof of a causal improvement from any one hyperparameter or seed.

Source: [`punyaa_v2_validation_metrics.json`](../../outputs/punyaa_v2_validation_metrics.json).

## E. Rolling chronological LightGBM evaluation

The three LightGBM configurations were additionally evaluated over three six-month validation windows. Each fits on rows before the beginning of its window, rebuilds the permitted target histories, and selects iterations by early stopping inside that window.

**Table IV. Rolling-window validation RMSE**

| Validation period | Fitting rows | Validation rows | Regularized | Balanced | Detailed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jan–Jun 2024 | 105,837 | 18,179 | 521.551 | 519.925 | **519.245** |
| May–Oct 2024 | 117,988 | 17,477 | 483.476 | 481.905 | **480.860** |
| Oct 2024–Mar 2025 | 132,802 | 17,198 | 449.203 | 449.087 | **448.688** |
| **Unweighted mean across windows** | n/a | n/a | **484.743** | **483.639** | **482.931** |

The detailed variant had the lowest RMSE in every window. This supports its selection among the tested LightGBM candidates. The windows overlap, including part of the main validation period, and early stopping uses their labels. They provide a stability check across time rather than three independent final-test estimates. The summary is an unweighted mean of window RMSEs, not a pooled row-level RMSE.

Sources: [`tune_lightgbm.py`](../../punyaa/tune_lightgbm.py) and [`punyaa_lightgbm_rolling_core_metrics.json`](../../outputs/punyaa_lightgbm_rolling_core_metrics.json).

## F. Kaggle submission and observed result

After validation, the five components with weights above the implementation's numerical cutoff of `1e-8` were refitted on all **150,000 labelled rows**. Each used its selected iteration count increased by 5%, rounded down to an integer. Historical features were rebuilt from the full labelled reference while preserving the strict earlier-month rule for each test row.

The final predictions were written to [`punyaa_v2_submission.csv`](../../outputs/punyaa_v2_submission.csv), with **50,000 rows**, zero-based sequential `Id` values, and the required `Id,Predicted` columns. Predictions are clipped to SGD 300–9,000 by the export code; the generated values were approximately SGD 1,670–5,376, so the export bounds did not bind. The file passed row-count, identifier, schema, and missing-value checks.

Punyaa reported a **Kaggle public score of 434.81** after submitting this file. The leaderboard figure is not independently stored in the local metrics JSON, and no private-leaderboard result is available here.

The public score and the local RMSE of 463.155 evaluate different rows and time distributions. Their numerical difference should not be treated as a measured improvement from local validation to production.

## G. Reproducibility and evidence

Run the following from the repository root. The production training command reruns validation and then refits the selected ensemble for submission.

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v

python3 -m punyaa.tune_lightgbm \
  --feature-profile core \
  --output outputs/punyaa_lightgbm_rolling_core_metrics.json

python3 -m punyaa.train \
  --feature-profile core \
  --output outputs/punyaa_v2_submission.csv \
  --metrics-output outputs/punyaa_v2_validation_metrics.json
```

All **12 unit tests passed** during implementation. The tests check temporal feature isolation, category cleaning, fallback behaviour, history windows, blend constraints, and submission formatting. They complement the executed production and rolling evaluations; they do not establish statistical significance for the small validation gain.

The principal artifacts are the [`feature builder`](../../punyaa/features.py), [`ensemble implementation`](../../punyaa/modeling.py), [`training command`](../../punyaa/train.py), [`initial metrics`](../../outputs/punyaa_validation_metrics.json), [`updated metrics`](../../outputs/punyaa_v2_validation_metrics.json), [`rolling metrics`](../../outputs/punyaa_lightgbm_rolling_core_metrics.json), and [`submission CSV`](../../outputs/punyaa_v2_submission.csv). Optional recent histories and official median-rent residual modelling are documented as experiments, not as contributors to the submitted result.

## H. Additional executed experiments

The evaluation windows and selection caveats used throughout are defined in
[`problem_and_protocol.md`](problem_and_protocol.md), and every experiment is
tabulated in [`results_summary.md`](results_summary.md).

A subsequent study reproduced v2 and fitted nine additional configurations:
recency-weighted LightGBM and CatBoost, higher-capacity LightGBM, property-history
removal, rent-per-square-metre prediction, XGBoost, and standard GBM. The selected
square-metre blend reduced common-split RMSE to 462.906 while increasing MAE to
334.961. The detailed positive and negative findings, selection limitations,
and reproducible commands are in
[`improvement_experiments.md`](improvement_experiments.md). Punyaa subsequently
reported a Kaggle public score of **434.70** for v3, improving on v2's **434.81**
by **0.11 RMSE**. The leaderboard results are user-reported, and the local MAE
tradeoff remains part of the comparison.
