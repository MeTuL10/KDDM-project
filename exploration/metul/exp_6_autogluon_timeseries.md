# Experiment 6: AutoGluon time-series market trend + tabular residual model

## Goal

The test set extends beyond most of the labelled history. A plain tree-based regression ensemble is very effective at learning how a listing's size, location, lease age, furnishing, and flat type affect rent, but it does not reliably extrapolate an increasing rent trend into new months.

This experiment separates those two jobs:

```text
predicted rent = forecasted market rent for the segment × property-level adjustment
```

The market forecast supplies the time extrapolation. The property-level adjustment captures why a particular listing is above or below its segment's typical rent.

## Step-by-step process

1. Load the raw train and test CSVs, then convert `RENT_APPROVAL_DATE` to a monthly timestamp called `period`.

2. Define the forecast origin as the earliest test month. Only training rows strictly before that date are used by the time-series market model. This avoids accidentally using labelled rents from a month that is being forecast.

3. Create a market segment from normalized `TOWN` and `FLAT_TYPE`. Normalization makes values such as `3 room` in train and `3-room` in test refer to the same segment.

4. Analyse candidate monthly covariates before fitting. The notebook writes `exp_6_covariate_correlations.csv`, reporting correlation with both the rent level and month-to-month log-rent change at lags 0, 1, 3, and 6 months. It retains COE bid volume, COE premium, and cyclic calendar features. It excludes stock aggregates because their relationship to rent changes is weak/unstable, and excludes COE quota because it is almost collinear with bids.

5. Aggregate labelled listing rents to one average rent per `market_segment × month`. Segments with fewer than 24 historical monthly observations are excluded from the time-series model because a very short series is not a credible basis for a multi-month forecast.

6. Tune the market forecaster with 25 outer-loop random-search trials. Each trial fits AutoGluon's Direct and Recursive Tabular forecasters (LightGBM) plus ETS and Theta baselines, and evaluates the same 12-month future holdout. The search samples `num_leaves`, `num_boost_round` (iterations), `max_depth`, `learning_rate`, `min_data_in_leaf`, and feature/bagging fractions. The results are written to `exp_6_timeseries_hpo_results.csv` and the lowest market-RMSE configuration is refit.

7. Create the tabular-model label in log space:

   ```text
   log_rent_residual = log(listing rent) - log(observed segment mean rent)
   ```

   This label expresses each listing's premium or discount relative to its town/flat-type market. The observed segment mean is used only to construct a training label; it is not available as a test feature.

8. Use the project `RentalPreprocessor` to produce the property, location, spatial, and auxiliary economic features. The preprocessor is fitted only on the fitting partition for validation, then refitted on all history before the forecast origin for the final model.

9. Tune AutoGluon `TabularPredictor` to predict the log-rent residual. Random HPO requests 25 trials per model family for LightGBM and CatBoost. LightGBM searches leaves, boosting rounds, depth, learning rate, minimum leaf size, and sampling fractions; CatBoost searches iterations, depth, learning rate, and L2 regularization. This model learns stable property-specific effects while the time-series model handles the market level.

10. Combine the components for every test listing:

    ```text
    predicted monthly rent = market_rent_forecast × exp(predicted log-rent residual)
    ```

    If a test segment has no eligible time-series history, the notebook uses the historical global mean as a safe fallback.

11. Save two outputs:

    - `exp_6_autogluon_timeseries_predictions.csv`: diagnostic output containing the original test fields plus `PREDICTED_MONTHLY_RENT`.
    - `exp_6_autogluon_timeseries_submission.csv`: competition output with exactly `Id,Predicted` columns.

## Validation interpretation

The validation RMSE, MAE, and R² are computed only on rows that receive a segment forecast. Compare these metrics with `exp_5_autogluon.ipynb` using the same chronological holdout. If the hybrid model wins on later validation windows, it is evidence that explicitly forecasting the market trend helps future-period predictions.

The first run is intentionally expensive: 25 time-series fits plus 25 tabular trials for each requested tree family. Set `EXP6_TS_TRIAL_TIME_LIMIT`, `EXP6_TS_FINAL_TIME_LIMIT`, `EXP6_TABULAR_HPO_TIME_LIMIT`, or `EXP6_TABULAR_FINAL_TIME_LIMIT` before running if the available runtime is different from the defaults.

## Important limitation

No forecasting approach can guarantee that the rent trend continues at its historical rate. The time-series component makes that assumption explicit and testable through chronological backtests; it is still preferable to expecting a tree model alone to extrapolate a new price regime.
