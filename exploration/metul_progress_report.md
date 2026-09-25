# Progress report — rental-price modelling

## Executive summary

The work has progressed from a broad, joined-feature LightGBM baseline to a focused, leakage-aware gradient-boosted-tree ensemble. The strongest **listing-level** result from the current Experiment 9 notebook on the common chronological validation split is **RMSE 467.122**, MAE 340.293, and R² 0.4945. It uses a 24-month global rent lag, historical town × flat-type mean rent, and nearest-place names on top of the baseline feature set listed below.

An earlier recorded version of this same 18-feature idea, with a 17-month global lag, achieved **RMSE 466.992** (MAE 340.197, R² 0.4948), the best number recorded so far. The current Experiment 9 notebook deliberately uses a 24-month lag to align with the two-year history available through the 17-month test horizon.

All model-selection results below use a chronological split: train rows before 2024-05 and validation rows from 2024-05 onwards (117,988 / 32,012 rows). This is more representative of the future test period than random cross-validation. Scores produced with other splits or at segment-month level are identified separately and should not be ranked directly against these results.

## Data and exploratory analysis

### Data sources explored

The rental listings were combined with HDB block attributes, MRT stations, shopping malls, schools, monthly COE data, monthly stock aggregates, and—in Experiment 9—MRT, tourist-attraction, and monument geospatial files. The main feature families explored were:

| Family | Features explored | Main finding |
| --- | --- | --- |
| Time / market regime | approval year, month, sequential month number, COE premium and bids, stock close and volume | Time is a major driver of rent. `approval_date_num` (r = 0.535), `approval_year` (r = 0.517), and mean COE premium (r = 0.511) have the strongest numeric correlations with rent. These variables partly capture the same rising market regime and must be evaluated chronologically. |
| Dwelling structure | flat type, floor area, lease commencement / age, flat model | Larger and higher-category flats cost more: `flat_type_num` (r = 0.345) and floor area (r = 0.329) are strong structural signals. Lease age is negatively associated with rent (r = -0.178); newer lease commencement is positive (r = 0.229). |
| Location / HDB attributes | town, block, street, coordinates, region, year completed, maximum floor | Locality is valuable, especially the high-cardinality town/block/street identifiers. `MAX_FLOOR` has a positive association (r = 0.292); town, street, and block collectively capture location effects that a single coordinate cannot. |
| Accessibility | nearest MRT/mall/school distance; numbers within 1 km and 2 km | MRT and mall proximity have small but plausible marginal associations: nearest MRT r = -0.085, MRTs within 2 km r = 0.100, nearest mall r = -0.083. School distance/count features show little linear association and did not survive baseline feature selection. |
| Target history / named landmarks | global and town × flat-type lagged rent; prior-only group mean; nearest MRT, attraction, and monument names | These were tested only with leakage safeguards. They yield a small but measurable improvement over the baseline feature set when added selectively. |

### Correlation, redundancy, and noise

- Correlation was used to generate hypotheses, not as the selection rule. High-cardinality categorical fields can look very strong in-sample, and time-related predictors are correlated with the target because rents rose across the labelled period. Every feature-set decision was consequently confirmed on a future holdout.
- Several variables are redundant or highly collinear: lease commencement and lease age have r = -0.995, while approval year and sequential approval month have r = 0.973. COE quota is near-collinear with COE bids, so the later time-series work kept COE premium/bids rather than all COE fields.
- Stock aggregates were weak or unstable as market-change covariates. In particular, stock close has near-zero Pearson correlation with rent (-0.013; its rank association differs), and stock volume is modestly negative (-0.226). They were not retained in the baseline feature set.
- Native categorical modelling is preferable to broad one-hot expansion for high-cardinality location fields. Earlier random 5-fold comparisons found native LightGBM categorical handling competitive or better in 6 of 7 configurations, while using far fewer features. Those random-fold scores are exploratory rather than the final selection metric.
- An outlier audit found 130 floor-area values and 752 rental values outside IQR fences. These were treated as review candidates, not automatically deleted: large flats and high rents can be genuine, relevant future cases. No blanket outlier removal was adopted.

### EDA benchmarks

The initial `feature_relations` notebook showed that a seven-feature LightGBM model achieved mean random 5-fold RMSE 500.67. Adding town one-hot variables reduced this to 482.73; later tuning reached roughly 480.09 in that random-fold setting. These figures established that locality and non-linear tree models matter, but are not directly comparable with the final chronological scores.

## Data processing completed so far

The overall preprocessing and feature-engineering flow is:

1. **Clean categorical strings.** Rental and HDB categories are lowercased, trimmed, whitespace-normalised, and have hyphen/underscore/slash separators folded to spaces. This reduced rental `STREET` values from 1,146 to 576 and `FLAT_TYPE` values from 11 to 6, and resolved six otherwise unseen test street spellings. Unseen test categories are retained rather than forced into a training vocabulary.
2. **Create base date and dwelling variables.** `RENT_APPROVAL_DATE` becomes approval year, approval month, sequential `approval_date_num`, and lease age; flat type is also represented numerically. `FURNISHED` and `FEE` are removed because they are constant in training (`yes` and 0 respectively).
3. **Join HDB data and address missing attributes.** Cleaned block/street keys provide latitude, longitude, maximum floor, year completed, and region. Missing numeric values are filled hierarchically from street/town/global statistics; region uses town/global modes.
4. **Engineer spatial accessibility features.** Latitude/longitude are converted to approximate kilometre coordinates. KD-tree queries derive nearest distance and counts within 1 km and 2 km for MRT stations, malls, and schools.
5. **Add external monthly aggregates.** COE premiums/bids and stock close/volume are aggregated by year/month and joined using the approval period.
6. **Construct historical target features.** Global monthly lags and town × flat-type historical means are calculated only from prior labelled months; missing lags fall back to available historical/global means. Validation and test targets are never used as features.

## Modelling experiments

### Algorithms evaluated and tabular-model progression

The experiments have evaluated LightGBM, CatBoost, Ridge regression, ETS, Theta, direct and recursive tabular time-series models, and a Temporal Fusion Transformer (TFT). The final approach is an ensemble of the two strongest listing-level gradient-boosted tree models—LightGBM and CatBoost—not a new automated-model-discovery approach.

The implementation uses AutoGluon only as a convenient training and weighted-ensemble wrapper. In the final Experiment 9 fits, the model list is explicitly restricted to `GBM` and `CAT`, and the `ignore_text` preset disables automatic feature generation that would treat categorical strings as text. This keeps feature design under project control: the models receive the hand-engineered feature set documented here rather than an automatically expanded categorical-text representation.

| Experiment | Algorithm / feature strategy | Validation protocol | RMSE | MAE | R² | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Exp. 2 | LightGBM, broad joined feature set; Optuna tuning | Earlier fixed holdout | 478.603 | — | — | Strong initial tree baseline; best Optuna configuration used shallow trees. |
| Exp. 4 feature subset | Tuned LightGBM with flat, location, floor area, lease, COE and selected stocks | Earlier holdout | 481.076 | 362.906 | 0.5887 | Block/street were most important; selected stock additions did not establish a better final approach. |
| Exp. 5 | LightGBM + CatBoost ensemble, full shared preprocessor | Common chronological split | 469.234 | 348.510 | 0.4899 | Established a strong listing-level GBM baseline. |
| Exp. 7 categorical clean | Tuned LightGBM + CatBoost ensemble after string cleaning | Common chronological split | 466.432 | 345.732 | 0.4960 | Best broad-feature ensemble observed; later subset testing favoured a simpler, faster set on the same protocol. |
| Exp. 7 subset baseline | LightGBM + CatBoost ensemble, 13-feature baseline set | Common chronological split | 467.230 | 345.659 | 0.4942 | Selected as the base for Experiment 9. |
| Exp. 9 target features | Same ensemble, baseline features + 24-month global lag + historical group mean + three nearest-place names | Common chronological split | **467.122** | **340.293** | **0.4945** | Best two-year-lag configuration. |

The Exp. 2 and Exp. 4 values were produced with older validation setups, so they show the investigation history rather than a head-to-head ranking with Exp. 7/9.

### Baseline feature set and subset comparison

The baseline consists of these 13 features: `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL`, `FLOOR_AREA_SQM`, `approval_date_num`, `approval_month`, `coe_quota_premium_mean`, `YEAR_COMPLETED`, `MAX_FLOOR`, `mrt_dist_nearest_km`, and `mall_dist_nearest_km`. It retains the most useful property, locality, time, market, and accessibility signals while omitting raw latitude/longitude and weaker auxiliary variables.

The same LightGBM + CatBoost ensemble was used for the following comparison. The result supports parsimony: adding many weak, redundant, or noisy features did not beat this baseline set.

| Feature set | Features | RMSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| Baseline 13-feature set | 13 | 467.230 | 345.659 | 0.4942 |
| 15-feature non-redundant set | 15 | 467.480 | 345.786 | 0.4937 |
| Full preprocessed table | 31 | 467.507 | 346.647 | 0.4936 |
| 25-feature candidate expansion | 25 | 468.304 | 348.314 | 0.4919 |

## Extrapolation problem and time-series attempts

### Why extrapolation is a concern

The test data run from 2025-03 to 2026-07, 17 months beyond much of the labelled history. Tree models learn piecewise predictions from observed splits; they do not naturally extend a sustained upward time trend outside the range seen during fitting. Strong interpolation validation alone therefore does not guarantee reliable future-period forecasts.

### Approaches tested

1. **Two-layer market + structural model (Exp. 4).** A monthly aggregate market model was combined with a time-independent structural model. On its final chronological fold, structural-only RMSE was 548.904, time-only was 674.200, and the tuned blend was 538.165. The market component received only 6.1% blend weight. This was not competitive with later direct listing models.
2. **Property-group time-series model (Exp. 5).** A time-series ensemble was fitted to monthly block × street × flat-type × flat-model groups with COE/stock covariates and historical-mean fallbacks. On the recorded observed item-month evaluation it scored RMSE 477.592, MAE 367.880, R² 0.3207. Fine-grained series were sparse, limiting forecast reliability.
3. **Market-segment forecast plus listing residual (Exp. 6).** The market series was aggregated to town × flat-type; direct/recursive tabular models, ETS, and Theta forecast the segment market, while a GBM residual model predicted listing-level log-rent adjustment. The best segment-level trial gave RMSE 273.240, but the combined listing-level validation was RMSE 486.052 (35,207 scored listings), worse than the direct tabular baseline family.
4. **Explicit trend + residual model (Exp. 7).** Ridge regression learned a global log-rent time slope plus regularised town × flat-type deviations, followed by a GBM residual model. It did not improve the preferred rolling-origin result: pooled RMSE was 571.673 for trend + residual versus 567.562 for the direct GBM baseline. It was somewhat better on the separate calendar-year protocol (606.495 vs. 613.928), but that was not the production selection criterion.
5. **Temporal Fusion Transformer (TFT) market forecast and listing hybrid (Exp. 9).** A global TFT forecasted town × flat-type monthly mean log rent. It achieved RMSE 258.453, MAE 182.479, R² 0.84 across 1,340 observed **segment-months**—a different target and unit from listing-level rent. When converted into a listing baseline and combined with a tabular residual model, the listing hybrid gave RMSE 499.791, MAE 391.545, R² 0.425 on the earlier 2024-04-forward holdout, compared with 471.209 / 349.000 / 0.489 for the direct model using the baseline feature set. The validation-selected TFT blend weight was 0, so it added no value in that experiment.

### Conclusion from the hybrid work

The time-series models can forecast aggregate market segments reasonably well, but their forecasts have not translated into a better per-listing prediction after adding property-level residual models. Sparse group histories, mismatch between segment averages and individual listings, and compounded market-forecast/residual error are likely contributors. The production direction remains the direct tree ensemble with safe time-related and target-history features, while extrapolation should continue to be assessed using rolling or future chronological holds.

## Best result: Experiment 9 target features

Experiment 9 starts with the 13 baseline features listed above and adds five features:

- `monthly_target_lag`: global mean rent from 24 months earlier;
- `town_flat_type_target_mean`: prior-only mean rent for the same town and flat type;
- `nearest_mrt_name`;
- `nearest_tourist_attraction_name`; and
- `nearest_monument_name`.

The alternative 24-month town × flat-type lag was worse than the global lag:

| Exp. 9 variant | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Global 24-month lag + historical group mean + nearest MRT/attraction/monument names | **467.122** | **340.293** | **0.4945** |
| Town × flat-type 24-month lag + historical group mean + same names | 468.986 | 342.419 | 0.4904 |
| Baseline 13-feature set only (Exp. 7 reference) | 467.230 | 345.659 | 0.4942 |

The global-lag Exp. 9 feature set reduces RMSE by 0.108 and MAE by 5.366 versus the baseline reference. The RMSE improvement is small, so it should be confirmed by additional rolling origins before treating it as definitive; its larger MAE improvement is nevertheless encouraging.

## Next steps

1. Search for additional external features that may add information beyond the present property, location, accessibility, COE, and historical-target features.
2. Confirm the leading feature set over multiple chronological/rolling validation windows, including a comparison of the 24-month and earlier 17-month lag variants.
3. Once the best model and feature set are selected, perform hyperparameter tuning with a substantially larger number of trials.
4. Explore deep-learning approaches further, including stronger TFT variants or other sequence/tabular architectures, and assess them on the same listing-level chronological validation protocol.
