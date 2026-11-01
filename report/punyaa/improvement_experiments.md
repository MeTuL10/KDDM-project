# Executed Model-Improvement Experiments

## A. Purpose and comparison protocol

This study tests concrete changes to the submitted LightGBM–CatBoost ensemble. The reference is v2, which achieved local RMSE **463.155**, MAE **334.492**, and a user-reported Kaggle public score of **434.81**. The new experiments use the same 117,988 fitting rows before May 2024 and 32,012 validation rows from May 2024 onward, with the same 21 core inputs unless a feature group is explicitly removed.

All six v2 component models were reproduced at their previously selected iteration counts. Applying their original fixed weights reproduced RMSE **463.154913**, matching the saved reference. Nine additional configurations were then fitted. Target histories continued to exclude the current month and later labels.

The chronological holdout is used for early stopping and candidate/blend selection. These are model-selection results, not an untouched test-set evaluation. Only measured results are reported below; an unsuccessful candidate remains a useful experimental finding.

## B. Implemented changes

1. **Recency weighting:** LightGBM and CatBoost were each trained with exponential calendar-month weights using half-lives of 12 and 24 months. Weights were normalized to mean one and used only for fitting; validation errors remained unweighted.
2. **Increased LightGBM capacity:** the detailed model was expanded from 127 to 255 leaves with unrestricted depth, minimum child size 80, L1 penalty 0.5, L2 penalty 8, and feature fraction 0.8. This is a joint candidate configuration, not a single-parameter ablation.
3. **Property-history ablation:** the property-level historical mean and count were removed while the detailed LightGBM settings were retained. Both models selected their stopping iterations on the same validation period.
4. **Rent-per-square-metre target:** LightGBM predicted `MONTHLY_RENT / FLOOR_AREA_SQM`; predictions were multiplied by floor area to recover rent. A custom validation metric selected iterations using RMSE in original rent units, rather than RMSE per square metre.
5. **XGBoost:** a categorical histogram model used depth 7, learning rate 0.035, minimum child weight 30, L1/L2 penalties 1/10, row fraction 0.85, and feature fraction 0.8. Early stopping selected from at most 2,200 trees.
6. **Standard GBM:** scikit-learn's `GradientBoostingRegressor` used depth 3, learning rate 0.05, minimum leaf size 30, and row fraction 0.85. Categories were one-hot encoded, unknown categories ignored, and numeric missing values filled using fitting-partition medians. Validation selected the best staged prediction from 400 trees.

The new LightGBM models otherwise used the existing detailed configuration with up to 2,600 iterations and learning rate 0.02. Recency-weighted CatBoost retained the seed-5228 depth-8 configuration with up to 1,600 iterations.

## C. Standalone results

| Model or change | Validation RMSE | Validation MAE | Selected iterations |
| --- | ---: | ---: | ---: |
| Reference detailed LightGBM | **468.262** | 339.006 | 251 |
| LightGBM, 12-month recency half-life | 470.239 | 342.035 | 174 |
| LightGBM, 24-month recency half-life | 469.471 | 340.753 | 212 |
| LightGBM, higher capacity | 468.994 | 340.744 | 193 |
| LightGBM, without property history | 472.649 | 346.516 | 860 |
| LightGBM, rent per square metre | 469.648 | 345.289 | 367 |
| Reference CatBoost, seed 5228 | **463.549** | 335.213 | 1,229 |
| CatBoost, 12-month recency half-life | 464.447 | 335.193 | 512 |
| CatBoost, 24-month recency half-life | 464.159 | **335.091** | 913 |
| XGBoost, categorical histogram | 477.348 | 347.112 | 238 |
| Standard GBM, one-hot encoding | 470.333 | 337.868 | 393 |

Removing property history increased the detailed LightGBM's RMSE by **4.388**, supporting retention of those features in this configuration. Recency weighting did not improve either family's standalone RMSE. Increasing tree capacity also failed to improve the detailed reference. Standard GBM learned a competitive baseline but remained behind detailed LightGBM and CatBoost; the tested XGBoost configuration was weaker. These findings concern the recorded configurations and training budgets, not every possible implementation or tuning of each family.

## D. Candidate blending result

Each new model was separately blended with the frozen v2 prediction. The candidate weight was chosen by least squares between zero and 0.5, preserving at least 50% of the reference prediction. Candidate blends were compared using validation RMSE, with a minimum 0.05 RMSE improvement required to replace the incumbent during selection.

The selected addition was the rent-per-square-metre LightGBM, with weight **0.160657**; the original v2 ensemble retained **0.839343**. Although the square-metre model was weaker alone, its different errors provided a small improvement when blended.

| Result | RMSE | MAE | Mean prediction minus actual |
| --- | ---: | ---: | ---: |
| Reproduced v2 ensemble | 463.155 | **334.492** | +17.481 |
| v2 + rent-per-square-metre LightGBM | **462.906** | 334.961 | +10.697 |
| Change from v2 | **−0.249** | +0.470 | −6.784 |

The selected blend improved RMSE in eight of the 11 validation months and worsened it in three. Paired resampling of complete months, using 3,000 draws with seed 5228, produced an exploratory 95% percentile interval for the RMSE reduction of approximately **[−0.002, 0.486]**. The interval includes zero and does not account for candidate-selection uncertainty or dependence between months. The observed local reduction is therefore modest; statistical significance is not established. The subsequently reported Kaggle result is recorded separately below.

## E. Supplemental chronological stability check

The six reference components and the selected square-metre LightGBM were also
refitted using **132,802 rows before October 2024**, then evaluated on **17,198
rows from October 2024 through March 2025**. Reference components kept their v2
iteration counts; the square-metre candidate selected 421 iterations. The main
run's candidate blend weight of 0.160657 was held fixed for the comparison below.

| October-2024-forward result | RMSE | MAE |
| --- | ---: | ---: |
| Refit reference ensemble, original component weights | 443.304 | **318.105** |
| Same ensemble plus square-metre candidate, fixed main-run weight | **443.267** | 318.373 |

The fixed blend reduced RMSE by **0.036**, while MAE increased by **0.268**.
Selecting a new weight on this period would yield 0.094916 and RMSE 443.234, but
that locally selected figure is not used as the fixed-weight stability result.
This period overlaps the primary selection holdout, and the candidate still
uses its labels for early stopping. The check supports only a small retrospective
stability finding, not independent confirmation of generalization or a claim of
statistical significance.

Sources: [`later-period fits`](../../outputs/punyaa_improvements/2024-10/metrics.json)
and the [`frozen-weight analysis`](../../outputs/punyaa_improvements/analysis.json).

## F. Reported Kaggle result

After submitting [`punyaa_v3_submission.csv`](../../outputs/punyaa_v3_submission.csv),
Punyaa reported a Kaggle public score of **434.70**, compared with **434.81** for
v2. The square-metre blend therefore achieved a reported public-score reduction
of **0.11 RMSE**, alongside the local holdout reduction of 0.249 RMSE.

| Submission | Local validation RMSE | Reported Kaggle public score |
| --- | ---: | ---: |
| v2 LightGBM–CatBoost ensemble | 463.155 | 434.81 |
| v3 with square-metre LightGBM blend | **462.906** | **434.70** |

Both Kaggle scores were supplied by Punyaa and have not been independently
retrieved from Kaggle. They measure the public leaderboard only; no private
leaderboard result or statistical-significance claim is made. Local MAE remained
slightly worse for v3, as shown in Section D.

## G. Reproduction and artifacts

Run these commands from the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v

python3 -u -m punyaa.improve \
  --submission outputs/punyaa_v3_submission.csv

python3 -u -m punyaa.improve \
  --validation-start 2024-10 \
  --experiments lightgbm_regularized lightgbm_balanced lightgbm_detailed \
    catboost_seed_42 catboost_seed_2026 catboost_seed_5228 lightgbm_rent_per_sqm \
  --threads 3

python3 -m punyaa.analyze_improvements
```

The runner records configurations, dependency versions, data/code fingerprints, selected iterations, timings, and per-row validation predictions. Completed fits are reused only when their fingerprint and configuration match. A full-data submission refit uses 105% of each selected iteration count, matching the existing export convention.

All 17 unit tests passed, including additional checks of calendar-based weight
decay and constrained blending. The generated
[`v3 candidate CSV`](../../outputs/punyaa_v3_submission.csv) passed row-count,
schema, sequential-ID, finite-value, and prediction-bound checks; its hash and
prediction range are recorded in the experiment metrics. The original v2 file's
hash was unchanged.

The main artifacts are the [`experiment runner`](../../punyaa/improve.py), [`diagnostic script`](../../punyaa/analyze_improvements.py), [`experiment metrics`](../../outputs/punyaa_improvements/2024-05/metrics.json), [`validation predictions`](../../outputs/punyaa_improvements/2024-05/validation_predictions.csv), [`segment diagnostics`](../../outputs/punyaa_improvements/2024-05/segment_diagnostics.csv), and [`uncertainty analysis`](../../outputs/punyaa_improvements/analysis.json). The original v2 submission and its reported Kaggle result remain a separate baseline. The user-reported v3 public score of 434.70 is documented in Section F rather than treated as a locally computed metric.
