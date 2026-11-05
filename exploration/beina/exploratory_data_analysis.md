# Exploratory Data Analysis

All figures and numbers come from `eda.R` (figures in `eda/`, numbers in `eda/eda_stats.txt`), run on the cleaned train table `train_feature` (150,000 rentals, 2021-01 to 2025-03) described in `data_process.md`. The test set (50,000 rentals) covers 2025-03 to 2026-07, so every test month is in the future relative to train.

## 1. Data overview

| Item | Value |
|---|---|
| Train / test rows | 150,000 / 50,000 |
| Missing values in raw train / test | 0 / 0 |
| Constant columns | `FURNISHED` (always `yes`), `FEE` (always 0), both dropped |
| `FLAT_TYPE` spellings | 11 values for 6 types (`"4 room"` and `"4-room"` etc.); unified as `room_count` |
| Rentals by flat type | 1-room 42, 2-room 3,098, 3-room 49,024, 4-room 54,079, 5-room 35,523, executive 8,234 |
| Rent rounding | 87.5% of rents end in 00, 98.7% in 50 |

## 2. Distributions (histograms)

### Target: monthly rent

![Distribution of monthly rent](eda/hist_monthly_rent.png)

- Range S$300 to S$7,600; median S$2,700, mean S$2,719, standard deviation S$750.
- Roughly symmetric (skewness < 1) with a long right tail from large flats. No transformation is needed for an RMSE target.
- Rents are quoted in round amounts, so the histogram is spiky at S$100 steps. This rounding is part of the irreducible noise.

### Numeric features

![Distributions of numeric features](eda/hist_numeric_features.png)

| Pattern | Features | Implication |
|---|---|---|
| Strong right skew (skewness 1.5 to 2.8) | MRT and mall counts, the three proximity scores, distances to the nearest MRT and schools, `max_floor` | Most blocks are in ordinary settings, with a few very well-connected ones. Tree models are unaffected by skew; Ridge sees standardised but unlogged values. |
| Multi-modal | `floor_area_sqm` (peaks at about 67 and 93 sqm: 3-room and 4-room), `lease_commence_date` and `year_completed` (building waves around 1980–85 and 1995–2000) | Reflects HDB's construction history and standard flat designs. |
| Discrete | `room_count`, `approval_month`, `month_sin/cos`, the counts | |
| Near-uniform | `time` | Rentals are spread evenly across months (section 4.2). |
| Narrow | `sqm_per_room` (17–42, IQR 22–25), `median_rent_ratio` (0.79–1.35) | |

Ranges are plausible after cleaning: building age 0–76 years, `max_floor` 2–50, all distances between 0 and 20 km.

## 3. Correlation check

### 3.1 Correlation with the target

![Correlation with the target](eda/corr_with_target.png)

Blue points show the Pearson correlation with monthly rent; orange points show it with the modelling target, rent − median benchmark.

| Feature | r with rent | Spearman with rent | r with rent − benchmark |
|---|---|---|---|
| `median_rent_benchmark` | **0.737** | 0.742 | −0.086 |
| `nat_median_rent_benchmark` | 0.683 | 0.700 | −0.078 |
| `time` | 0.535 | 0.564 | −0.091 |
| `coe_qp_mean` | 0.512 | 0.514 | −0.075 |
| `stock_grab_holdings_adjclose` | −0.452 | −0.345 | 0.050 |
| `floor_area_interaction` | 0.395 | 0.362 | 0.105 |
| `room_count` | 0.345 | 0.342 | 0.027 |
| `floor_area_sqm` | 0.329 | 0.318 | 0.032 |
| `max_floor` | 0.291 | 0.227 | 0.112 |
| `flat_age_at_approval` | −0.179 | −0.168 | −0.119 |
| `dist_to_cbd` | −0.093 | −0.060 | −0.019 |
| `dist_nearest_mrt` | −0.085 | −0.086 | −0.079 |

- The **median benchmark** is the strongest single predictor of rent (r = 0.74): it already contains town, flat type and the market level.
- **Time, COE and stocks** correlate with rent only because they share the 2021–2023 upward trend. Once the benchmark is subtracted, they correlate at −0.09 to +0.05. Section 4.4 shows this is a trend coincidence, not a causal link.
- With the benchmark removed, the remaining linear signals are weak (|r| ≤ 0.12): building height, age and MRT proximity. The residual is driven by block-level effects that no single numeric feature captures, which is why the block target encodings matter most in the models (`model_experiment.md`, section 8).
- Pearson and Spearman agree closely, so the relationships are roughly monotone.

### 3.2 Correlation between features

![Correlation between numeric features](eda/corr_heatmap.png)

Pairs with |r| ≥ 0.9:

| Feature A | Feature B | r |
|---|---|---|
| `year_completed` | `flat_age_at_approval` | −0.995 |
| `lease_commence_date` | `year_completed` | 0.989 |
| `cbd_room_interaction` | `cbd_area_interaction` | 0.988 |
| `lease_commence_date` | `flat_age_at_approval` | −0.984 |
| `approval_year` | `time` | 0.973 |
| `floor_area_sqm` | `room_count` | 0.952 |
| `median_rent_benchmark` | `nat_median_rent_benchmark` | 0.926 |

Other clear clusters: each distance and its proximity score (strongly negative, by construction); the MRT and mall counts at different radii; and `dist_to_cbd` with its two interactions. These features are kept, because tree models tolerate collinearity and Ridge's penalty spreads weight across correlated columns (`data_process.md`, section 10).

## 4. Time trend check

### 4.1 Rent over time

| Year | Rentals | Median rent | Mean rent | Median rent per sqm |
|---|---|---|---|---|
| 2021 | 37,619 | 2,100 | 2,121 | 22.8 |
| 2022 | 32,564 | 2,450 | 2,524 | 27.2 |
| 2023 | 35,654 | 3,000 | 3,036 | 33.0 |
| 2024 | 35,239 | 3,100 | 3,110 | 34.3 |
| 2025 (Q1) | 8,924 | 3,150 | 3,142 | 34.4 |

![Median rent by flat type](eda/trend_rent_by_flat_type.png)

- Mean rent rose **53%**, from S$2,055 (first 3 months) to S$3,142 (last 3 months).
- Nearly all of the rise happened from mid-2021 to mid-2023; since mid-2023 rents have been **almost flat**. Median 4-room rent went 2,200 → 2,500 → 3,200 → 3,300 → 3,300 from 2021 to 2025.
- All flat types moved together, so the trend is market-wide, not specific to one segment.
- **Modelling implication:** test months are 0–16 months after the last train month. Tree models cannot extrapolate a trend, and a straight line fitted over 2021–2024 over-predicts the flat 2024–25 period by up to S$440 a month (`model_experiment.md`, section 5). Hence the design: predict around a market benchmark that tracks the level.

### 4.2 Rental volume

![Rentals per month](eda/trend_rental_volume.png)

Volume is stable at 2,362–3,705 rentals a month (80% of months between 2,633 and 3,218) with no gaps, so no period is under-represented.

### 4.3 Rent vs the median rent benchmark

![Rent vs benchmark](eda/trend_rent_vs_benchmark.png)

The town × flat-type median follows the market level closely, so subtracting it removes the trend. The mean gap (rent − benchmark) is small but not constant: +S$25 (2021), +S$47 (2022), then about −S$75 from 2023 onwards. This shift is one reason the gap model learns from the most recent data patterns via the target encodings, and why the honest forecast adds a small drift (`data_process.md`, section 8.2).

### 4.4 COE and stock prices

![Rent, COE and stocks indexed](eda/trend_macro_indexed.png)

| Series vs monthly mean rent | r (levels) | r (month-on-month changes) |
|---|---|---|
| `coe_qp_mean` | **0.898** | **0.015** |
| `stock_ocbc_bank_adjclose` | 0.755 | −0.040 |
| `stock_golden_agri_resources_adjclose` | 0.759 | −0.159 |
| `stock_grab_holdings_adjclose` | −0.783 | −0.048 |
| `stock_singtel_adjclose` | 0.642 | 0.079 |

High correlation in levels but essentially none in changes is the signature of two series that merely both trend: a **spurious correlation**. COE peaked in late 2023 and fell 30%, while rent did not move. Combined with zero permutation importance, this is why COE and stocks are **excluded from the final model**. See also `rent_coe_over_time.png`.

## 5. Outlier check

### 5.1 Rent

![Rent by flat type](eda/outlier_box_flat_type.png)

| Rule | Rows flagged | Share |
|---|---|---|
| 1.5 × IQR, all rows | 752 | 0.50% |
| 1.5 × IQR within flat type | 281 | 0.19% |
| 1.5 × IQR within flat type × quarter (accounts for the trend) | 3,007 | 2.00% |
| Robust \|z\| > 3.5 on rent − benchmark | 162 | **0.11%** |

![Rent minus benchmark](eda/outlier_residual_hist.png)

**High rents** (top 10: S$6,300–7,600) are almost all 5-room and executive **adjoined flats** or **premium maisonettes** of 129–215 sqm in central or mature estates (Tanjong Pagar Plaza, Holland Drive, Whampoa). These are real, unusually large units.

**Low rents** (bottom 10: S$300–450) are all **2-room** flats of 37–54 sqm, with benchmarks of S$1,300–2,550. They are probably subsidised or partial (room) rentals, and some repeat at the same block, e.g. 463A Sembawang Drive at S$400 three times.

2-room has the highest share of within-quarter outliers (4.97%), matching its sparse benchmark coverage.

**Decision: no rows are removed.** The outliers are plausible transactions, only 0.11% are extreme relative to their benchmark, and similar units will appear in the test set. Removing them would bias predictions for those segments while RMSE still counts their errors.

### 5.2 Features and auxiliary data

| Check | Finding | Treatment |
|---|---|---|
| `floor_area_sqm` by flat type | 1-room 31, 2-room 34–67, 3-room 52–126, 4-room 70–145, 5-room 99–157, executive 125–215 sqm; ranges overlap but are consistent with HDB designs. 72 rows have extreme `sqm_per_room` (3 × IQR). | Kept |
| `sg-hdb-block.csv` sentinels | 11 blocks with `max_floor = -1` and `year_completed = -1` (159 train rows) | Imputed (`data_process.md`, section 5.1) |
| `sg-hdb-median-rent.csv` entry error | Bukit Merah 5-room 2022-Q1 = 300, against about 3,000 elsewhere | Removed (`data_process.md`, section 8.1) |
| Building age after cleaning | 0–76 years; before fixing the sentinels the maximum was 2,024 | Fixed |

## 6. Summary for modelling

1. **Market level:** rents rose 53% and then flattened. The model predicts around a town × flat-type median benchmark rather than raw rent, and test quarters use a carried-forward, drift-adjusted benchmark.
2. **Location at block level** drives what remains. Numeric features correlate weakly with rent − benchmark, so block and street target encodings are needed.
3. **COE, stocks and time** only share the trend (spurious correlation), so they are excluded from the final model.
4. **Outliers** are genuine large or subsidised units and are kept.
5. **Collinear features** are kept, since the models tolerate them.
6. **Noise floor:** rents are rounded, and near-identical flats in the same quarter differ by about S$461 (standard deviation; `model_experiment.md`, section 10). This bounds achievable RMSE.
