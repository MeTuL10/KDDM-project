# Data Processing

This document describes every processing step applied before modelling, and the reason for it. All steps are in `data_cleaning.R`; `model_training.R` adds the in-fold target encodings (section 9). Running `source("data_cleaning.R")` builds two tables with identical processing:

| Table | Rows | Columns | Notes |
|---|---|---|---|
| `train_feature` | 150,000 | 59 | rent approvals 2021-01 to 2025-03 |
| `test_feature` | 50,000 | 58 | rent approvals 2025-03 to 2026-07 (no `monthly_rent`) |

## 0. Input data

| File | Rows | Used for |
|---|---|---|
| `Data/train.csv`, `Data/test.csv` | 150,000 / 50,000 | rental transactions (target: `MONTHLY_RENT`) |
| `sg-hdb-block.csv` | 9,464 blocks | coordinates, max floor, year completed, subzone, planning area, region |
| `sg-mrt-stations.csv` | 243 stations (231 open, 12 planned) | MRT distances and counts |
| `sg-shopping-malls.csv` | 113 malls | mall distances and counts |
| `sg-schools.csv` | 337 schools | school distances and counts |
| `sg-coe-prices.csv` | 2021-01 to 2026-08 | COE features (EDA only; not used by the final model) |
| `sg-stock-prices.csv` | 2021-01-04 to 2026-07-30 | stock features (EDA only; not used by the final model) |
| `sg-hdb-median-rent.csv` | 2005-Q2 to 2026-Q2 | town × flat-type median rent benchmark (used up to 2025-Q1 only, see section 8) |

The raw train and test files have no missing values.

## 1. Cleaning the rental records (`clean_flat_data`)

| Step | Reason |
|---|---|
| Lower-case all column names | One naming convention across all tables. |
| Drop `FURNISHED` and `FEE` | Both are constant: every row is `FURNISHED = yes` and `FEE = 0`, so they carry no information. |
| `room_count` from the first character of `flat_type` (executive → 6) | `flat_type` is spelled two ways in train (`"4 room"` and `"4-room"`); the first character is the same in both spellings, so `room_count` is a clean numeric flat type. `flat_type` itself is dropped later (section 7). |
| `approval_year`, `approval_month`, `approval_quarter` from `rent_approval_date` | Calendar parts for seasonality and for joining the quarterly median rent. |
| `time` = months since 2021-01 (the earliest month of train + test) | A single trend index that stays consistent when future test months are scored. |
| `flat_model_modified`: 20 flat models grouped into 6 families (old model, maisonette, apartment, DBSS, special, compact) | Rare models (e.g. `premium apartment loft`, `terrace`) have few rows; the families give them a reliable shared effect. The original `flat_model` is kept too. |

## 2. COE prices

- Parse `$` and `,` out of `quota_premium` and `bids_received`; map month names to numbers to build `yearmonth`.
- Keep categories A and B (cars), which reflect household spending; C–E are goods vehicles, motorcycles and open category.
- Per month: `coe_qp_mean` (mean quota premium of A and B across both bidding rounds), `coe_qp_spread` (max − min), `coe_bids_mean`.

**Status:** merged into `train_feature` / `test_feature` for EDA but **excluded from the final model**. EDA shows COE only shares the market trend with rent (level correlation 0.90, month-on-month change correlation 0.02), and its permutation importance is ≈ 0 (`model_experiment.md`).

## 3. Stock prices

- The raw file has no sector field, so each of the 77 companies is assigned one of 8 sectors by hand (financial, real estate, telecom, transport, industrials, consumer, healthcare, technology).
- Per sector, keep the company with the largest average trading volume (8 stocks: CapitaLand Mall Trust, ComfortDelGro, Genting Singapore, Golden Agri-Resources, Grab, OCBC, Riverstone, Singtel), so each sector contributes one liquid, representative series.
- Monthly mean adjusted close per stock (`stock_<name>_adjclose`); adjusted close corrects for dividends and splits.

**Status:** like COE, kept for EDA but **excluded from the final model** (trend-only correlation, zero importance).

## 4. Schools, MRT and malls

- Schools: `mainlevel_code` grouped into primary / secondary / post-secondary / mixed; a mixed-level P1–S4 school is counted as both primary and secondary, since it serves both age groups. (`type_code` and `nature_code` are also grouped but not used downstream.)
- MRT stations split into open and planned. Planned stations are kept separately because an announced line can raise rents before it opens.

## 5. HDB block attributes and spatial features

### 5.1 Sentinel values in `sg-hdb-block.csv`

11 blocks store `-1` for both `max_floor` and `year_completed` (159 train rows). Left as is, they produce a building age of about 2,024 years and negative floor-area interactions.

| Column | Treatment | Reason |
|---|---|---|
| `max_floor` | `-1` → missing, filled with the block/street median, then the town median | Neighbouring blocks of the same era have similar heights. In practice each block/street group is a single block, so the town median is what fills these 11 blocks. |
| `year_completed` | `-1` → missing, filled per rental row as `lease_commence_date − 1` (section 7) | Across all other rows the median of `year_completed − lease_commence_date` is −1: a block completes about a year before its lease starts. |

### 5.2 Distances and counts

Coordinates are projected from WGS84 (EPSG:4326) to SVY21 (EPSG:3414), Singapore's metre-based projection, so Euclidean distances are in metres. Then, per HDB block:

| Feature | Definition |
|---|---|
| `dist_nearest_mrt`, `dist_nearest_planned_mrt` | km to the nearest open / planned MRT station |
| `dist_nearest_mall` | km to the nearest mall |
| `dist_nearest_primary_school`, `dist_nearest_secondary_school` | km to the nearest primary / secondary school |
| `dist_to_cbd` | km to Raffles Place (1.2840, 103.8510) |
| `mrt_count_500m/1000m/1500m` | distinct open MRT station codes within the radius |
| `shopping_count_500m/1000m/1500m` | distinct malls within the radius |
| `school_count_1km/3km/5km` | distinct schools within the radius |
| `mrt_proximity_score`, `mall_proximity_score`, `school_proximity_score` | `1 / (distance_km + 0.1)`: an inverse-distance decay, so the first few hundred metres matter most. The 0.1 km floor caps the score at 10 and avoids dividing by near-zero for blocks next to a station or mall. |

Features are computed once per block (9,464 blocks), not per rental row, and joined in section 6.

## 6. Merging rentals with block, COE and stock data (`merge_auxiliary`)

- **Address normalisation:** `town`, `block` and `street` are lower-cased, trimmed and whitespace-collapsed in both the rental and block tables. About 20% of rental rows spell `street` in title case (`"Ang Mo Kio Avenue 3"`) while the block file is lower case; without this step 30,781 train rows (20.5%) and 10,482 test rows (21.0%) fail to match.
- **Join key:** `town + block + street` (left join). After normalisation every train and test row matches exactly one block (0 unmatched). `block + street` is unique in the block file, so the join never duplicates rows, and town never conflicts.
- COE and stock values are joined by `rent_approval_date = yearmonth`; every train and test month is covered.
- The function stops if a join changes the row count and prints the number of unmatched rows for each source (all 0).

## 7. Derived features (`add_derived_features`)

| Feature | Formula | Reason |
|---|---|---|
| `year_completed` gaps | `coalesce(year_completed, lease_commence_date − 1)` | Fill the 11 sentinel blocks (section 5.1). |
| `flat_age_at_approval` | `approval_year − year_completed` | Building age when the lease was signed. |
| `month_sin`, `month_cos` | `sin/cos(2π · month / 12)` | Places December next to January for seasonality. |
| `sqm_per_room` | `floor_area_sqm / room_count` | Spaciousness within a flat type. |
| `cbd_room_interaction` | `dist_to_cbd × room_count` | The CBD premium differs by flat size. |
| `cbd_area_interaction` | `dist_to_cbd × floor_area_sqm` | Same, by floor area. |
| `floor_area_interaction` | `max_floor × floor_area_sqm` | Large flats in tall blocks (views, newer towers). |

Dropped columns: `flat_type` (replaced by `room_count`; its spelling is inconsistent), `postal_code` (an identifier; one block file value is `"NIL"`), `latitude` and `longitude` (location is represented by the distance features, subzone and the target encodings).

## 8. Median rent benchmark (`add_median_rent`)

The official HDB median rent per town × flat type × quarter is the baseline the models predict around (`model_experiment.md`).

### 8.1 Cleaning the file

| Step | Reason |
|---|---|
| `na` and `-` read as missing | 6,892 of 13,305 values (51.8%) are missing, 46.4% since 2021. Most are flat types that are rare in a town: since 2021, 1-room is 100% missing, 2-room 89.5%, executive 59.3%, 3-room 14.3%, 5-room 11.4%, 4-room 3.8%. |
| Town names trimmed and lower-cased; duplicate keys averaged | Some names carry trailing spaces (`"QUEENSTOWN "`), which creates duplicate keys once trimmed. |
| `EXEC` → `room_count` 6 | Matches `clean_flat_data`. |
| Values below half the cross-town median for that flat type and quarter are removed | Removes entry errors; only Bukit Merah 5-room 2022-Q1 = 300 (other towns ≈ 3,000) is caught. |
| **Only quarters up to the last train quarter (2025-Q1) are used** | Each quarter's median is computed from that quarter's rents. Using published medians for 2025-Q2 to 2026-Q2 would anchor test predictions on aggregates of the test rents themselves, which is information unavailable at forecast time. |

### 8.2 Filling a row's benchmark

Each rental row gets the first available value in this order:

| Step | Benchmark | Train rows | Test rows |
|---|---|---|---|
| 1 | Own town × flat type × quarter | 96.6% | — (all test quarters are after the cutoff) |
| 2 | National median for that flat type and quarter (median across towns) | 3.3% | |
| 3 | No town published that flat type that quarter: national median from the adjacent quarter | 0.1% | |
| 4 | Still missing (1-room has no medians since 2021): national median of the next larger flat type | (in step 3) | |
| Forecast | Quarters after the cutoff use the cutoff quarter's (2025-Q1) value from steps 1–4, multiplied by a **drift**: `exp(g × quarters_after_cutoff)` | — | 100% |

`g` is the average quarterly log growth of a national rent index over the last 4 quarters up to the cutoff. The index is the geometric mean of the 3-room to executive national medians, the only types published in every quarter. At the 2025-Q1 cutoff, `g` = +0.55% per quarter, so 2026-Q3 (6 quarters ahead) is scaled by +3.3%. The 4-quarter window was chosen on the honest holdout (`model_experiment.md`, section 6).

Both settings are environment variables: `BENCHMARK_CUTOFF` (default: the last train quarter) and `BENCHMARK_GROWTH_QUARTERS` (default 4). Setting `BENCHMARK_CUTOFF=2024-Q2` reproduces the same forecast setup on the OOT holdout.

### 8.3 Benchmark features

| Feature | Definition |
|---|---|
| `median_rent_benchmark` | the town × flat-type benchmark from 8.2 |
| `nat_median_rent_benchmark` | national median for that flat type and quarter |
| `median_rent_ratio` | `median_rent_benchmark / nat_median_rent_benchmark`: how expensive the town is for that flat type |
| `area_to_benchmark_ratio` | `floor_area_sqm / median_rent_benchmark` |

After section 8 no train or test row has a missing value.

## 9. Model-stage processing (`model_training.R`)

| Step | Reason |
|---|---|
| **Target** = `monthly_rent − median_rent_benchmark` | Removes the market level (rents rose 54% over the train period), which tree models cannot extrapolate; predictions add the benchmark back. |
| **COE and stock columns excluded** (`macro_cols`) | No predictive value beyond the benchmark (sections 2–3). |
| Categorical columns as factors: `town`, `flat_model`, `flat_model_modified`, `subzone`, `planning_area`, `region` | Native categorical handling in LightGBM and CatBoost; integer codes for XGBoost; one-hot (town, flat model, region) for Ridge. |
| **In-fold hierarchical Bayesian target encoding** of 7 keys | `block` and `street` have thousands of levels, too many to use directly. See below. |
| No numeric normalisation for the tree models | Splits are invariant to monotone rescaling. Ridge standardises internally (`glmnet`, `standardize = TRUE`). |

### Target encoding

Each key's mean residual is shrunk towards its parent's encoding:

    encoding = (n × mean residual of the key + M × parent encoding) / (n + M)

| Feature | Key | Parent | M |
|---|---|---|---|
| `te_town` | town | overall mean | 30 |
| `te_street` | town + street | `te_town` | 15 |
| `te_address` | block + street | `te_street` | 10 |
| `te_address_room` | block + street + room count | `te_address` | 5 |
| `te_subzone` | subzone | `te_town` | 20 |
| `te_flat_model` | flat model | overall mean | 20 |
| `te_town_model` | town + flat model | `te_flat_model` | 10 |

Leakage control: encodings are learned only from the fitting rows of each split (tuning split, OOT split, each CV fold, or all of train for test). Fitting rows get inner 5-fold out-of-fold encodings, so a row's own rent never enters its own feature. Unseen keys fall back to their parent. Nothing is saved to disk; encodings are recomputed for every split.

## 10. Processing decisions not taken

| Option | Decision | Reason |
|---|---|---|
| Remove rent outliers | Kept all rows | Extreme rents are plausible, e.g. 5-room adjoined flats at S$6,000–7,600, and 2-room flats at S$300–450 that are likely subsidised or room rentals. Only 0.11% of rows are beyond a robust \|z\| of 3.5 on rent − benchmark, and the test set will contain similar cases (`exploratory_data_analysis.md`, section 5). |
| Log-transform rent | Not used | RMSE is scored on raw rent. A ratio target (rent / benchmark − 1) was tested and gave no gain (`model_experiment.md`). |
| Drop highly correlated features (e.g. `year_completed` vs `flat_age_at_approval`, r = −0.995) | Kept | Tree models are robust to collinearity and Ridge's penalty handles it. Dropping the age columns changed OOT RMSE by only +0.1. |
| Use median rents published after the last train quarter | Not used | Leakage (section 8.1). |
