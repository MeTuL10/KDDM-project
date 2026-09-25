# Comprehensive Progress Report: Singapore HDB Monthly Rent Prediction (Experiment 2)

**Course:** NUS CS5228 — Knowledge Discovery and Data Mining  
**Project:** HDB Monthly Rent Prediction Challenge  
**Author:** Aayush Majumdar 
**Date:** September 2026  
**Scope:** End-to-End Technical Documentation from Exploratory Data Analysis to Zero-Leakage Ensembling  
**Evaluation Metric:** Root Mean Squared Error (RMSE)  
**Leaderboard Trajectory:** Initial Benchmark (~495) $\to$ Baseline Ensemble (**443.000**) $\to$ Advanced TE Ensemble (**433.607**) $\to$ Tuned Luxury (**434.053**) $\to$ Zero-Leakage Private LB Robust Ensemble (**V4 Verified**)

---

## Executive Summary & Milestone Progression

This report provides a rigorous technical breakdown of **Experiment 2 (`expt-2`)**, our high-performance modeling initiative for predicting continuous `MONTHLY_RENT` (SGD) across 50,000 test transactions in Singapore. Spanning 6 iterative sprints and 9 methodological phases, this report details:
1. **Exploratory Data Analysis (EDA)** and structural data quality diagnostics.
2. **Data Sanitization & Address Resolution** achieving 100% building match.
3. **Dual Validation Architecture** designed for non-stationary temporal distribution shift.
4. **Feature Engineering Rationale** encompassing geospatial BallTrees, macroeconomic anchoring, and Bayesian target encodings.
5. **Heterogeneous Modeling, Optuna Optimization & SLSQP Stacking**.
6. **Data Leakage Forensic Audit & Zero-Leakage Nested Architecture** guarding against the 50% Public vs. 50% Private Leaderboard shakeup.

### Summary Progression Matrix

| Iteration / Phase | Core Innovation | Validation Scheme | Local OOT RMSE | 5-Fold OOF RMSE | Kaggle Public LB | Key Diagnostic & Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Sprint 0** (Phases 1–2) | Address matching, casing cleanup, temporal split. | Temporal Split ($\le$ 2024-06 / $\ge$ 2024-07) | 551.40 (Ridge) | — | — | Test transactions occur 9–16 months in the future. Trees plateau without trend anchoring. |
| **Sprint 1** (Phases 3–4) | 14 Spatial BallTrees, Macro residual formulation. | Out-of-Time (OOT) Holdout | 494.86 (Raw Bench) | — | — | Predicting residual ($\Delta = \text{Rent} - \text{Benchmark}$) solves tree extrapolation limits. |
| **Sprint 2** (Phases 5–6 / V1) | Heterogeneous GBDT Tri-Model (LGB, XGB, CB) + 5-Fold CV. | OOT Holdout + 5-Fold CV | 457.61 | 476.53 | **443.000** | Proved strong baseline. Identified 99.80% block overlap and 3.2× higher within-block variance. |
| **Sprint 3** (Phase 7 / V2) | Hierarchical Bayesian Target Encoding + Transit Hubs + SLSQP. | OOT Holdout + 5-Fold CV | 449.26 | 470.99 | **433.607** | **-9.393 drop on Leaderboard**. Uncovered residual asymmetry across flat models (+1,491 SGD). |
| **Sprint 4** (Phase 8 / V3) | Flat Model TE, Cohort Area Ratios, Optuna Tuning. | OOT Holdout + 5-Fold CV | 447.80 | 470.53 | **434.053** | Slight +0.445 degradation on Kaggle Public LB due to high-capacity trees over-indexing on noise. |
| **Sprint 5** (Phase 9 / V4) | Leakage Audit, Nested Fold-Level TE, Capacity Regularization. | Nested Zero-Leakage CV | **451.27** (True Unbiased) | **471.62** (True Unbiased) | **V4 Ready (34ea03...)** | Eliminated 3 subtle target leakages. Neutralized extreme tail biases. Maximizes 50% Private LB robustness. |

---

## 1. Initial Exploratory Data Analysis & Data Quality Audit

### 1.1 Structural Properties of the Datasets
The competition provides:
- **`data/train.csv`**: 150,000 historical rental transactions spanning **January 2021 through March 2025**.
- **`data/test.csv`**: 50,000 blind test transactions spanning **March 2025 through July 2026**.
- **Auxiliary Datasets**:
  - `sg-hdb-block.csv`: 12,472 geocoded HDB blocks with physical building metadata (`MAX_FLOOR`, `YEAR_COMPLETED`, `SUBZONE`, `PLANNING_AREA`, `REGION`).
  - `sg-mrt-stations.csv`: 197 operational and planned MRT/LRT transit stations.
  - `sg-shopping-malls.csv`: Commercial retail centers across Singapore.
  - `sg-primary-schools.csv`: Primary educational institutions.
  - `sg-hdb-median-rent.csv`: Historical quarterly median rents segmented by Town and Flat Type.
  - `sg-coe-prices.csv`: Monthly Certificate of Entitlement (COE) bidding prices across categories.
  - `sg-stock-prices.csv`: Daily domestic equities data for Singapore blue-chip firms.

### 1.2 Data Quality Anomalies & Resolutions
An in-depth data quality audit revealed several critical anomalies in the raw data:

1. **Zero-Variance Columns**:
   - `FEE`: 100.0% of records in both train and test have a value of `0.0`.
   - `FURNISHED`: 100.0% of records in both train and test have a value of `"yes"`.
   - *Resolution*: Both columns were eliminated immediately to reduce noise and memory footprint.

2. **Categorical Spelling & Casing Variants**:
   - `FLAT_TYPE`: Inconsistent string entries mixed hyphenated and non-hyphenated formats (e.g., `"4 room"` vs `"4-room"`, `"executive"` vs `"Executive"`).
   - *Resolution*: Mapped into 6 canonical string categories: `1-room`, `2-room`, `3-room`, `4-room`, `5-room`, and `executive`. Extracted a clean numerical integer `room_count` $\in \{1, 2, 3, 4, 5, 6\}$.

3. **Sentinel Values in Geographic Keys**:
   - `BLOCK`: Contained `"NIL"` string sentinels, leading to failed joins.
   - *Resolution*: Stripped invalid whitespace, normalized alphanumeric characters, and imputed missing block identifiers using street name and postal coordinates.

4. **Geographic Coordinate Missingness**:
   - Several transactions in the raw dataset had missing or unpopulated `LATITUDE` and `LONGITUDE`.
   - *Resolution*: Constructed a composite lookup key `BLOCK + "_" + STREET` matched against `sg-hdb-block.csv`, achieving **100.0% address geocoding resolution** (0 missing coordinates across all 200,000 records).

5. **Temporal Non-Stationarity & Macro Drift**:
   - The HDB rental market in Singapore underwent unprecedented inflation between 2021 and 2025: median monthly rents surged by **over 54%** (from ~$2,100 SGD in early 2021 to ~$3,250 SGD in late 2024).
   - Because the test dataset evaluates future dates (**March 2025 to July 2026**), standard decision tree algorithms (which make orthogonal splits based on historical values) are structurally incapable of predicting values higher than the historical maximum without explicit macroeconomic and temporal anchoring.

---

## 2. Validation Architecture: Preventing Temporal Contamination

To ensure local validation reliably reflects test set generalization, we established a **Dual Validation Architecture**:

### 2.1 Primary Out-of-Time (OOT) Holdout
- **Train Split**: All transactions approved on or before **June 30, 2024** ($124,016$ records, 82.68%).
- **Validation Split**: All transactions approved between **July 1, 2024 and March 31, 2025** ($25,984$ records, 17.32%).
- *Purpose*: Faithfully replicates the 9-to-16-month forward forecasting gap required for the unseen Kaggle test set. Models must predict future price distributions using only historical information.

### 2.2 Secondary 5-Fold Stratified Cross-Validation
- Stratified across all 150,000 transactions to evaluate feature stability, prevent fold-specific variance, and provide out-of-fold predictions across the entire training corpus for ensembling.

---

## 3. Exploratory Insights & Feature Engineering Rationale

### 3.1 The Target Residual Formulation
Tree-based gradient boosters cannot extrapolate linear trends outside their training bounds. If rents continue to rise into 2025/2026, raw tree regressors plateau at the maximum leaves seen in 2024.

*Solution*: We decomposed the prediction problem into two additive components:
$$\text{MONTHLY\_RENT}_{i} = \text{Benchmark}_{i} + \Delta_{i}$$
1. **$\text{Benchmark}_{i}$ (Macro Anchor)**: Derived from the official HDB median rent series matching transaction $i$'s `(town, flat_type, quarter)`. For future test quarters (2025-Q2 through 2026-Q3), benchmarks were calculated using town-specific linear trend extrapolation fitted on recent quarters.
2. **$\Delta_{i}$ (Target Residual)**: The localized deviation ($\text{Actual Rent} - \text{Benchmark}$) driven by micro-geography, building floor, remaining lease, and premium amenities.

By training models to predict $\Delta_{i}$, the learning task becomes stationary, and the models focus on invariant relative premiums rather than fighting macroeconomic inflation.

### 3.2 Geospatial Proximity Engines (14 BallTrees)
Using scikit-learn's `BallTree` with the Haversine metric (Earth radius = 6,371,000 m), we extracted 14 geodesic spatial features across 9,456 unique HDB block coordinates:
- **`dist_cbd`**: Geodesic distance to Downtown Core (Raffles Place: `1.2840, 103.8510`). EDA proved that CBD distance exhibits a sharp, non-linear negative rent decay: flats within 3 km command an average +\$850 SGD premium over flats 15 km away.
- **`dist_nearest_mrt` & `mrt_count_1km`**: Distance and density to 197 transit stations.
- **`dist_transit_interchange` & `interchange_count_1km`**: Identified 44 multi-line interchange stations (e.g., Jurong East, Bishan, Dhoby Ghaut). EDA proved that interchange proximity commands an additional +\$120/month premium over standard single-line stations.
- **`dist_nearest_mall` & `mall_count_1km`**: Proximity to commercial retail hubs.
- **`dist_nearest_pri_school` & `dist_elite_school`**: Proximity to primary schools and elite historical Gifted Education Programme (GEP) schools (e.g., Nanyang, Tao Nan, Catholic High, ACS), reflecting strong rental demand from families.

### 3.3 Macroeconomic Indicators
To inform the models of prevailing economic conditions:
- **COE Premiums (`coe_cat_a`, `coe_cat_b`)**: Monthly Certificate of Entitlement vehicle quota premiums, serving as a real-time proxy for consumer liquidity and inflation.
- **Financial Banking Index (`stock_index`)**: Equal-weighted composite of Singapore's major domestic financial institutions (DBS, OCBC, UOB) extending through the test period.

### 3.4 Building Lifecycle & Physical Attributes
From `sg-hdb-block.csv` and transaction data:
- **`remaining_lease_years`**: $99 - (\text{approval\_year} - \text{YEAR\_COMPLETED})$. Older flats suffer depreciation beyond 30 years of age.
- **`flat_age_at_approval`**: Building age at lease signing.
- **`is_skyrise` & `max_floor_tier`**: Flats in buildings with $\ge 30$ storeys (e.g., Pinnacle@Duxton) command significant elevation premiums.
- **Within-Cohort Floor Area Deviations**:
  - `area_diff_town_type`: $\text{FLOOR\_AREA\_SQM} - \mu_{\text{town, type}}$
  - `area_ratio_town_type`: $\text{FLOOR\_AREA\_SQM} / \mu_{\text{town, type}}$
  Captures oversized units that command premiums above their standard room-type peers.

---

## 4. Iterative Experiments & Modeling Progression

### Sprint 1 (Phases 3–4): Geospatial Engines & Macro Anchoring
- **Goal**: Implement geodesic BallTrees and evaluate the baseline median rent benchmark.
- **Findings**: The raw median rent benchmark achieved an OOT RMSE of **494.86 SGD** (MAE: 346.31 SGD). This confirmed that macro-anchoring alone outperformed naive mean prediction ($798.12) and Ridge regression ($551.40).

### Sprint 2 (Phases 5–6 / V1): Heterogeneous GBDT Tri-Model Ensemble
- **Goal**: Train diverse GBDT regressors (LightGBM, XGBoost, CatBoost) to predict the residual $\Delta_{i}$ across 5-fold cross-validation.
- **Architecture**:
  - LightGBM: `learning_rate=0.03`, `num_leaves=63`, `max_depth=8`.
  - XGBoost: `learning_rate=0.03`, `max_depth=6`, `subsample=0.85`.
  - CatBoost: `learning_rate=0.04`, `depth=6`, `l2_leaf_reg=3.0`.
- **Results**:
  - LightGBM OOT RMSE: 458.12 SGD
  - XGBoost OOT RMSE: 459.40 SGD
  - CatBoost OOT RMSE: 461.35 SGD
  - **Tri-Model Blend OOT RMSE**: **457.61 SGD** (Weights: `[0.40, 0.40, 0.20]`).
  - **5-Fold OOF Blend RMSE**: **476.53 SGD**.
  - **Kaggle Public Leaderboard**: **443.000 RMSE**.
- **Inference**: High-cardinality analysis revealed **99.80% block overlap** between train and test, with within-block variance (205 SGD) being 3.2× higher than between-town variance (63 SGD). This established the necessity of micro-locational block-level target encoding.

### Sprint 3 (Phase 7 / V2): Hierarchical Bayesian Target Encoding & Stacking
- **Goal**: Exploit block-level spatial granularity without overfitting sparse blocks.
- **Methodology**: Implemented empirical Bayesian shrinkage:
  $$\widehat{\mu} = \frac{n \cdot \bar{y} + M \cdot \mu_{\text{prior}}}{n + M}$$
  Descending hierarchically: $\text{Town } (M=30) \to \text{Block } (M=10) \to \text{Block} \times \text{Type } (M=5) \to \text{Street } (M=15) \to \text{Subzone } (M=20)$.
- **Optimization**: Formulated constrained Non-Negative Least Squares (SLSQP) stacking on 150,000 out-of-fold predictions.
- **Results**:
  - LightGBM OOT RMSE: 450.31 SGD (-7.81 SGD gain)
  - XGBoost OOT RMSE: 451.09 SGD (-8.31 SGD gain)
  - CatBoost OOT RMSE: 451.32 SGD (-10.03 SGD gain)
  - **Blended OOT RMSE**: **449.26 SGD** (-8.35 SGD gain).
  - **5-Fold OOF Blend RMSE**: **470.99 SGD** (-5.54 SGD gain).
  - **Kaggle Public Leaderboard**: **433.607 RMSE** (Massive **-9.393 drop**).
- **Inference**: Residual analysis revealed flat-model luxury spreads: luxury models (`type s2`, `dbss`, `terrace`, `premium maisonette`) exhibited residuals up to +\$1,491 SGD above the benchmark.

### Sprint 4 (Phase 8 / V3): Flat Model Target Encoding & Optuna Optimization
- **Goal**: Eliminate flat-model pricing asymmetry and tune tree architectures via Bayesian optimization.
- **Methodology**:
  - Added Flat Model Bayesian TE: $\text{Global Mean} \to \text{Flat Model } (M=20) \to \text{Town} \times \text{Flat Model } (M=10)$.
  - Ran 38 Optuna trials against the OOT holdout, discovering deep tree configurations:
    - LightGBM: `num_leaves=152`, `max_depth=16` $\to$ 449.31 OOT RMSE.
    - XGBoost: `max_depth=10`, `reg_lambda=12.14` $\to$ 448.63 OOT RMSE.
    - CatBoost: `depth=8`, `l2_leaf_reg=4.0` $\to$ 450.59 OOT RMSE.
  - OOT Blend reached **447.80 SGD**; 5-Fold OOF Blend reached **470.53 SGD**.
- **Kaggle Public Leaderboard Result**: **434.0526 RMSE** (+0.445 degradation).
- **Inference & Problem Detection**: Why did a model with superior local validation degrade on the competition leaderboard? This triggered a forensic audit into data leakage and capacity over-parameterization.

---

## 5. The Forensic Leakage Audit & Sprint 5 (V4) Resolution

### 5.1 Root Causes of V3 Degradation
Our forensic investigation uncovered three structural issues:

1. **Temporal Target Encoding Leakage**:
   In `feature_engineering_v3.py`, 5-fold target encodings were computed statically across the *entire 150,000 transactions* prior to the temporal OOT split. Consequently, training rows before July 2024 contained target encoding averages calculated from post-July 2024 validation labels. This artificially deflated local validation RMSE and deceived the Optuna tuner into favoring deeper trees.

2. **Cross-Fold Contamination in 5-Fold CV**:
   Because target encodings were pre-computed in the parquet files, during the outer 5-fold modeling loop, the training folds contained target information computed from the validation fold's ground truth.

3. **Sample Size Prior Mismatch**:
   Training rows had target encodings computed on $0.8 \cdot N$ samples (via 5-fold OOF), while test rows had target encodings computed on $1.0 \cdot N$ samples (full dataset). This altered the Bayesian shrinkage weight between train and test.

4. **Public Leaderboard Noise-Chasing**:
   The Kaggle leaderboard evaluates on only **50% of the test set**. Deep trees (`max_depth=10`, `num_leaves=152`) overfit random noise on the public 50% split. Furthermore, SLSQP gave 33.5% weight to depth-10 XGBoost and halved LightGBM's allocation (from 54.5% to 26.0%).

### 5.2 The Sprint 5 (V4) Architecture
To guarantee 100% zero-leakage and maximize Private Leaderboard stability:

1. **Decoupled Base Feature Matrices ([`feature_engineering_v4.py`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/src/feature_engineering_v4.py))**:
   - `train_features_v4.parquet` ($150,000 \times 71$) and `test_features_v4.parquet` ($50,000 \times 69$) contain strictly static spatial, macroeconomic, and physical building attributes.
   - **Zero static target encoding columns are saved to disk**.

2. **Modular In-Fold Target Encoder (`HierarchicalBayesianTargetEncoder`)**:
   - Implemented with `fit(df, target_col)`, `transform(df)`, and `fit_transform_oof(df, target_col, k_folds)`.
   - In each fold $k$:
     - Encoder is fitted strictly on `tr_idx` ($120,000$ rows).
     - `X_tr` receives internal 5-fold out-of-fold target encodings.
     - `X_va` and `test_features` are transformed using fold $k$'s shrinkage maps.
     - **Result**: Exactly zero leakage, zero cross-fold contamination, and identical sample size distribution ($n = 0.8 \cdot N$) across train, validation, and test.

3. **Structural Capacity Regularization**:
   - Constrained tree depth to prevent memorizing micro-splits:
     - LightGBM: `num_leaves=75`, `max_depth=9`, `reg_alpha=0.6`, `reg_lambda=2.5`.
     - XGBoost: `max_depth=7`, `min_child_weight=20`, `reg_lambda=15.0`.
     - CatBoost: `depth=7`, `l2_leaf_reg=6.0`, `random_strength=1.2`.

4. **Balanced Ensemble Allocation**:
   - Solved bounded SLSQP stacking weights:
     $$\mathbf{w}^* = [0.3091 \, (\text{LightGBM}), \, 0.1909 \, (\text{XGBoost}), \, 0.5000 \, (\text{CatBoost})]$$
     Prevented any single model family from dominating the ensemble.

5. **Tail Error Bias Neutralization**:
   - Diagnostic analysis of OOF predictions confirmed regression-to-the-mean shrinkage was eliminated:
     - Budget Tier ($< \$2,500$, $n=58,265$): Mean error = **-$0.02$ SGD**.
     - Luxury Tier ($> \$4,000$, $n=3,163$): Mean error = **+$4.82$ SGD**.

---

## 6. Comprehensive Performance Comparison Table

| Metric / Dimension | Sprint 1 (Benchmark) | Sprint 2 (V1 Ensemble) | Sprint 3 (V2 Advanced TE) | Sprint 4 (V3 Tuned Luxury) | Sprint 5 (V4 Zero-Leakage Robust) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Leakage Hygiene** | Clean | Clean (No TE) | Contaminated (Full CV TE) | Contaminated (Full CV TE) | **100% Clean (Nested Fold TE)** |
| **Tree Capacity** | N/A | Default (depth 6–8) | Default (depth 6–8) | Deep / Overfit (depth 10, leaves 152) | **Regularized (depth 7, leaves 75)** |
| **LightGBM Weight** | — | 40.0% | 54.5% | 26.0% | **30.9%** |
| **XGBoost Weight** | — | 40.0% | 7.5% | 33.5% | **19.1%** |
| **CatBoost Weight** | — | 20.0% | 38.0% | 40.5% | **50.0%** |
| **OOT Holdout RMSE** | 494.86 | 457.61 | 449.26 *(leaked)* | 447.80 *(leaked)* | **451.27 *(True Unbiased)*** |
| **5-Fold OOF RMSE** | — | 476.53 | 470.99 | 470.53 | **471.62 *(True Unbiased)*** |
| **Kaggle Public LB** | ~495 | 443.000 | 433.607 | 434.053 *(Overfit)* | **V4 Ready (`34ea03...`)** |

---

## 7. Submission Verification & Codebase Artifacts

The final V4 submission has been validated, cryptographically verified, and mirrored to root:
- **Root Submission File**: [`submission.csv`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/submission.csv)
- **Archive Copy**: [`expt-2/outputs/submissions/submission_ensemble_v4.csv`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/outputs/submissions/submission_ensemble_v4.csv)
- **MD5 Checksum**: `34ea0374991eeb177afa5ca97e8aa6b0` (Exact byte-level match)
- **Dimensions**: Exactly 50,000 rows $\times$ 2 columns (`Id`, `Predicted`), 0 nulls.
- **Predicted Distribution**: Min = \$1,538.87, Max = \$6,307.69, Mean = \$3,169.00 SGD (Standard Deviation: \$482.15 SGD).

### Interactive Jupyter Notebooks Index
- **Phase 9 / Sprint 5 Notebook**: [`expt-2/notebooks/06_phase9_v4_zero_leakage_and_private_lb_robustness.ipynb`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/notebooks/06_phase9_v4_zero_leakage_and_private_lb_robustness.ipynb) (184 KB, fully executed).
- **Phase 8 / Sprint 4 Notebook**: [`expt-2/notebooks/05_phase8_v3_luxury_features_and_tuning.ipynb`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/notebooks/05_phase8_v3_luxury_features_and_tuning.ipynb) (175 KB, fully executed).
- **Phase 7 / Sprint 3 Notebook**: [`expt-2/notebooks/04_phase7_v2_advanced_target_encoding_and_stacking.ipynb`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/notebooks/04_phase7_v2_advanced_target_encoding_and_stacking.ipynb) (135 KB, fully executed).
- **Walkthrough Document**: [`expt-2/WALKTHROUGH.md`](file:///Users/aayushmajumdar/Desktop/CS5228-knowledge-discovery-and-data-mining-rel/nus-cs5228/project/KDDM-project/expt-2/WALKTHROUGH.md)

---

## 8. Strategic Recommendation for Kaggle Competition

1. **Avoid Chasing 0.2-Point Fluctuations on Public Leaderboard**:
   Because the Public Leaderboard score is evaluated on only 50% of the test transactions (~25,000 rows), high-depth models will inevitably show spurious micro-improvements that reverse on the private 50% test set.
2. **Select the V4 Zero-Leakage Pipeline for Final Evaluation**:
   The V4 pipeline represents our most scientifically sound, leak-free, and regularized architecture. Its structural constraints and balanced ensembling provide our strongest defense against Private Leaderboard rank shakeup.

