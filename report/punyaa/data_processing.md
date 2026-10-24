# Data Preprocessing and Historical Feature Construction

## A. Scope and evaluation protocol

This section documents the feature pipeline implemented in [`punyaa/features.py`](../../punyaa/features.py). 

The common chronological split contains 117,988 fitting rows before May 2024 and 32,012 validation rows from May 2024 onward. Validation history is calculated from the fitting partition only. Final test history uses the complete labelled training set, subject to the same strict calendar-month cutoff for each query row.

## B. Static preprocessing

### 1. Category normalization

Town, block, street, flat type, flat model, and nearest-place names are lowercased and trimmed. Repeated whitespace is collapsed, and hyphens, underscores, and slashes are replaced with spaces. Equivalent strings such as `4-room`, `4_room`, and `4/room` therefore become `4 room`. Missing categories are represented by `unknown`.

LightGBM receives pandas categorical columns whose vocabulary is established from the fitting partition; unseen categories become missing category values. CatBoost receives normalized strings and categorical-column indices. Neither path expands the location fields into a large one-hot matrix.

### 2. Dates and HDB attributes

Approval date is represented by calendar month and `approval_date_num = 12 × year + month`. Normalized town, block, and street values join each rental listing to the auxiliary HDB block table, supplying building coordinates, completion year, and maximum floor. Missing HDB values use town medians and then global medians from that auxiliary table. These fallbacks do not use rental targets.

### 3. Geographic and monthly context

KD-tree nearest-neighbour queries use approximately kilometre-scaled latitude and longitude to obtain distance to the nearest MRT station and shopping mall. The nearest MRT, tourist attraction, and monument names provide categorical geographic context. These are straight-line proximity estimates, not walking or travel times.

COE quota premiums are converted to numeric values and averaged by year and month before joining to the rental approval date. Missing months are interpolated from the supplied auxiliary series, with endpoint values used outside its range. These external observations are used as supplied; their publication dates are not reconstructed, so the pipeline should not be described as a fully publication-time-audited historical forecast.

## C. Selected feature representation

The submitted `core` configuration uses **21 inputs**: 16 static features and five historical features. Eight inputs are categorical.

| Feature group | Inputs |
| --- | --- |
| Property and location categories | `TOWN`, `BLOCK`, `STREET`, `FLAT_TYPE`, `FLAT_MODEL` |
| Structure | `FLOOR_AREA_SQM`, `YEAR_COMPLETED`, `MAX_FLOOR` |
| Time and market context | `approval_date_num`, `approval_month`, `coe_quota_premium_mean` |
| Proximity | `mrt_dist_nearest_km`, `mall_dist_nearest_km` |
| Named geographic context | `nearest_mrt_name`, `nearest_tourist_attraction_name`, `nearest_monument_name` |
| Global target history | `monthly_target_lag` |
| Town × flat-type history | `town_flat_type_history_mean`, `town_flat_type_history_count` |
| Property × flat-type history | `property_flat_type_history_mean`, `property_flat_type_history_count` |

Raw coordinates support distance calculations but are not included as final model inputs. No numeric scaling or dimensionality reduction is applied.

## D. Historical target features and leakage controls

For a listing in month `t`, eligible rental labels must have a month strictly smaller than `t`. All listings from the same calendar month are excluded together. This also handles the March 2025 overlap between the labelled training data and the test data: March labels cannot enter March test-history features.

The global lag is the mean rent exactly 24 calendar months earlier. When unavailable, it falls back to the mean of eligible earlier labels. Town × flat-type history groups by normalized town and flat type. Property × flat-type history additionally identifies the block and street.

Group means are smoothed toward the eligible global mean:

`smoothed mean = (historical rent sum + α × prior global mean) / (historical count + α)`.

The smoothing strengths are 30 for town × flat-type groups and 8 for property × flat-type groups. Counts indicate how much observed history supports each estimate. An unseen group falls back to the eligible global mean; rows without any earlier labelled history retain missing mean values for the tree learners.

Training features are generated from strictly prior-month labels. Validation rows use only pre-May-2024 labels, including when later validation months would otherwise have access to earlier validation targets. Test features similarly use only permitted labelled training history, without appending predicted rents or test labels.

## E. Experimental feature profiles

The optional `recent` profile adds 18 inputs to the core representation, giving **39 features** when the official median-rent file is absent. It includes global rent means over the previous 6, 12, and 24 calendar months; smoothed group means and counts over those windows; and differences between six-month and 24-month means. These are calendar windows rather than a fixed number of transactions.

The saved quick runs recorded RMSE **465.295** for `core` and **468.494** for the expanded-history run. The latter artifact predates the explicit `feature_profile` JSON field; its profile is identified from the recorded execution context. These short-budget results supported retaining `core` as the production default, but do not establish that recent histories are inferior under every tuning budget.

An optional loader also supports `data/auxiliary/sg-hdb-median-rent.csv`, normalizing town and flat-type keys and matching quarterly benchmarks. When usable, the benchmark can enter the feature matrix and support a residual LightGBM component. This file was absent for the reported production run; neither its feature nor its residual component contributed to the 434.81 submission. Its timing and predictive value require separate evaluation before use in a future version.

## F. Verification and evidence

The implementation has 12 unit tests across [`test_punyaa_features.py`](../../tests/test_punyaa_features.py) and [`test_punyaa_modeling.py`](../../tests/test_punyaa_modeling.py). They cover category normalization, same-month exclusion, future-target independence, overlapping-month transformation, unseen-group fallback, calendar-window boundaries, optional benchmark preparation, blend constraints, model capacities, and submission schema. All 12 passed during implementation.

The main evidence is [`features.py`](../../punyaa/features.py), the [`training entry point`](../../punyaa/train.py), the [`core quick metrics`](../../outputs/punyaa_v2_core_quick_validation_metrics.json), and the [`expanded-history quick metrics`](../../outputs/punyaa_v2_quick_validation_metrics.json). Model performance and the final Kaggle result are discussed in [`model_experiments.md`](model_experiments.md).
