# Punyaa rental ensemble

This package is a reproducible alternative to the exploratory notebooks. It
uses the model families that performed most consistently in the project report:
LightGBM and CatBoost.

The production `core` pipeline combines:

- the report's compact structural and location feature set;
- native categorical handling;
- nearest MRT, attraction, and monument names;
- a 24-month global rent anchor;
- strictly prior-month town/flat-type and property/flat-type histories; and
- three regularized LightGBM variants at different tree capacities;
- three CatBoost fits using different random seeds; and
- a constrained validation-selected blend that can reject weak components.

Historical target features never use a row's own month or a later month. This
also applies to the March 2025 overlap between `train.csv` and `test.csv`.

## Run the tests

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v
```

An experimental `recent` profile also adds leakage-safe 6-, 12-, and 24-month
group histories. It is available for further experiments, but is not the
production default because it did not improve the main chronological holdout.
If `data/auxiliary/sg-hdb-median-rent.csv` is present, the pipeline additionally
tests the official quarterly median rent as a feature and residual baseline.

## Fast validation smoke run

```bash
python3 -m punyaa.train --quick --validation-only
```

To smoke-test the experimental history windows:

```bash
python3 -m punyaa.train --quick --validation-only --feature-profile recent
```

## Rolling LightGBM evaluation

```bash
python3 -m punyaa.tune_lightgbm \
  --feature-profile core \
  --output outputs/punyaa_lightgbm_rolling_core_metrics.json
```

## Production validation and Kaggle submission

```bash
python3 -m punyaa.train \
  --feature-profile core \
  --output outputs/punyaa_v2_submission.csv \
  --metrics-output outputs/punyaa_v2_validation_metrics.json
```

The validation split starts in May 2024, matching the common split used in the
latest report. The executed production validation achieved RMSE **463.155** and
MAE **334.492**, improving the earlier clean pipeline's RMSE of 463.716. The
three-window LightGBM evaluation selected the `detailed` variant, with mean RMSE
482.931 versus 483.639 for `balanced` and 484.743 for `regularized`. Punyaa
reported a Kaggle public score of 434.81 for this v2 submission; it is separate
from the local validation RMSE.

## Executed improvement study

The separate experiment runner reproduces v2 and tests nine additional
configurations, including standard GBM, XGBoost, recency weighting, a
property-history ablation, and rent-per-square-metre prediction. It checkpoints
each completed validation fit with its configuration and data/code fingerprint.

```bash
python3 -u -m punyaa.improve \
  --submission outputs/punyaa_v3_submission.csv

python3 -m punyaa.analyze_improvements
```

The common-split study selected a blend of 83.9343% v2 and 16.0657%
rent-per-square-metre LightGBM, reducing RMSE from 463.155 to 462.906 while
increasing MAE from 334.492 to 334.961. Punyaa reported a Kaggle public score of
**434.70** for v3, an improvement of **0.11** over v2's 434.81. The local gain is
small and comes with an MAE tradeoff. Detailed positive and negative results are documented in
[`improvement_experiments.md`](../report/punyaa/improvement_experiments.md).

## Extrapolation and noise analyses

```bash
python3 -m punyaa.extrapolation --validation-start 2024-05   # also 2024-10
python3 -m punyaa.noise_floor --validation-start 2024-05     # also 2024-10
```

`extrapolation.py` tests piecewise-linear leaves, monotone constraints, and
recalibration against the frozen v2 blend. `noise_floor.py` estimates the rent
variation among listings that share every model input. Both reuse the stored v2
component predictions and do not refit CatBoost. Results are written to
`outputs/punyaa_extrapolation/` and `outputs/punyaa_noise_floor/`, and are
discussed in `report/punyaa/`.

## Moving-market backtests

```bash
python3 -m punyaa.market_regimes --output-dir outputs/punyaa_regimes
python3 -m punyaa.market_regimes --families catboost --output-dir outputs/punyaa_regimes/catboost
python3 -m punyaa.market_regimes --test-projection --output-dir outputs/punyaa_regimes
```

`market_regimes.py` repeats the evaluation at three twelve-month forecast origins
(2022-01, 2023-01, 2024-01) to test how the pipeline behaves when rents move. It fits
a hedonic monthly price index, deflates the target by it, projects the index forward
(flat, linear or damped), and compares against a control and an oracle index. The
third command tabulates what a projection would add at the test horizons. Results
are written to `outputs/punyaa_regimes/` and discussed in
`report/punyaa/market_regimes.md`.

