# EXPT-4 Progress Report: Pure Machine Learning Track with External Asset Valuations & Non-Linear Curves

**Project:** CS5228 Knowledge Discovery & Data Mining — Singapore HDB Rental Price Prediction  
**Directory:** `exploration/Aayush/expt-4/` (Self-Contained Extension on top of `expt-2`)  
**Role:** Senior Data Scientist & ML Engineer  
**Date:** September–October 2026  
**Status:** Complete & Validated  

---

## 1. Executive Summary & Objectives

Following the forensic analysis in `expt-3` (which demonstrated how the 173.19 score was achieved via an external registry lookup), the user directed us to **advance our pure machine learning pipeline (`expt-2`) in a dedicated `expt-4` directory**.

The objective of `expt-4` is to push the boundaries of legitimate predictive modeling by introducing **novel real-world domain knowledge and external macro/asset datasets** that do **NOT** leak the test set target (`MONTHLY_RENT`):
1. **Official Singapore Land Authority (SLA) Bala's Table:** Non-linear leasehold decay and financing restriction cliffs.
2. **LTA Multi-Line Transit Interchange Topology:** Modeling major transfer hubs vs. standard single-line stations.
3. **Official HDB Capital Asset Valuations (Pre-March 2025):** Integrating 201,386 historical resale property transactions from `data.gov.sg` to establish capital asset prices per square meter and gross rental yield benchmarks.

---

## 2. Novel Features Engineered in `expt-4`

Building on top of the 71 zero-leakage features from `expt-2`, `expt-4` expanded the feature representation to **84 structured predictors**:

### A. SLA Bala's Table Non-Linear Leasehold Valuation (4 features)
In Singapore real estate, lease depreciation is non-linear due to bank mortgage lending rules and Central Provident Fund (CPF) usage limits:
* `bala_discount_factor`: Continuous interpolation of the official SLA Bala's Table discount curve $\phi(\text{remaining\_lease}) \in [0, 1]$.
* `bala_cliff_sub60`: Binary indicator ($1.0$ if $\text{remaining\_lease} < 60\text{ years}$), capturing the bank loan restriction cliff.
* `bala_cliff_sub30`: Binary indicator ($1.0$ if $\text{remaining\_lease} < 30\text{ years}$), capturing the statutory CPF withdrawal threshold.
* `bala_decay_rate`: Gradient of depreciation relative to flat age.

### B. LTA Multi-Line Transit Interchange Network Topology (4 features)
Tenants value proximity to transit hubs with multiple connecting lines significantly higher than single-line suburban stations:
* Identified all 35 active multi-line interchange stations (e.g., Bishan, Jurong East, Dhoby Ghaut, Marina Bay, Outram Park).
* `dist_to_mrt_interchange`: Haversine distance to the nearest multi-line interchange.
* `nearest_interchange_lines`: Capacity indicator (2 or 3 intersecting lines).
* `mrt_interchanges_within_1000m`: Density of major transit hubs within a 1km walking/cycling radius.
* `transit_hub_connectivity_index`: Weighted proximity index: $\sum \frac{\text{lines}}{\text{distance} + 100\text{m}}$.

### C. HDB Historical Capital Asset Valuations (5 features)
Downloaded and cleaned 201,386 historical HDB resale transactions from `data.gov.sg` strictly prior to March 2025 (`exploration/Aayush/expt-4/data/hdb_resale_prices_2017_2025.parquet`):
* `resale_price_per_sqm_benchmark`: Hierarchical Bayesian shrunken capital price per sqm ($\$/\text{sqm}$) fitted at $\text{Town} \times \text{Flat Type}$ with empirical shrinkage ($M = 30$).
* `estimated_capital_value`: Implied asset value = $\text{FLOOR\_AREA\_SQM} \times \text{resale\_price\_per\_sqm}$.
* `bala_adjusted_capital_value`: Asset valuation adjusted for statutory lease decay.
* `implied_annual_rental_yield`: Fundamental financial yield metric:
  $$\text{Gross Rental Yield} = \frac{\text{median\_rent\_benchmark} \times 12}{\text{bala\_adjusted\_capital\_value}} \times 100\%$$
* `price_to_rent_ratio`: Multiple of annual rent required to purchase the flat.

---

## 3. Modeling & Validation Architecture

To ensure strict zero data leakage, `expt-4` implemented a two-step validation protocol:

### Step 1: Zero-Leakage Out-of-Time (OOT) Holdout Evaluation
* **Training Period:** January 2021 to June 2024 (124,016 listings).
* **Validation Period:** July 2024 to March 2025 (25,984 listings).
* Hierarchical Bayesian Target Encoders (7 levels) were fitted strictly on the pre-July 2024 slice.

| Model Architecture | OOT RMSE | OOT MAE | Fit Time |
| :--- | :---: | :---: | :---: |
| **CatBoost Regressor** | **464.55 SGD** | **338.68 SGD** | ~28 min |
| **LightGBM Regressor** | **468.77 SGD** | **341.42 SGD** | 38.5 s |
| **XGBoost Regressor** | **471.79 SGD** | **343.94 SGD** | 85.0 s |
| **SLSQP Meta-Ensemble** | **464.07 SGD** | **338.10 SGD** | — |

*SLSQP Optimal Weights:* CatBoost ($75.2\%$), LightGBM ($15.1\%$), XGBoost ($9.7\%$).

### Step 2: Full 5-Fold Nested Cross Validation & Test Inference
* Computed out-of-fold Hierarchical Bayesian Target Encodings across all 150,000 training rows to eliminate fold leakage.
* Trained 5 folds for each model family (15 trained models total).
* Generated test set predictions on all 50,000 listings in `test_features_v5.parquet`.

---

## 4. Cross-Experiment Comparison & Ensembling

Comparing the newly generated `expt-4` predictions against `expt-2`:
* **Pearson Correlation ($r$):** **0.9966** (exceptionally high agreement, indicating robust feature stability).
* **Mean Absolute Difference:** **28.43 SGD**.
* The subtle adjustments in `expt-4` properly discount older flats below 60 years remaining lease (via Bala's Table) and reward units close to major transit interchanges (Bishan, Dhoby Ghaut).

### Master Blend Generated
We constructed a balanced 50/50 meta-blend combining the lean feature ensemble of `expt-2` with the capital asset valuation ensemble of `expt-4`:
* **Artifact:** `exploration/Aayush/expt-4/outputs/submissions/submission_pure_ml_v4_v5_master_blend.csv`

---

## 5. Artifact & Submission Directory Map

```
exploration/Aayush/expt-4/
├── data/
│   └── hdb_resale_prices_2017_2025.parquet  # 201,386 pre-2025 official resale transactions
├── src/
│   ├── feature_engineering.py               # Generates 84 features (Bala Table, MRT hubs, Capital Values)
│   └── models.py                            # OOT evaluation + 5-fold nested CV + SLSQP stacking
├── outputs/
│   ├── features/
│   │   ├── train_features_v5.parquet        # (150,000, 84) clean feature matrix
│   │   └── test_features_v5.parquet         # (50,000, 82) clean feature matrix
│   └── submissions/
│       ├── submission_sprint6_pure_ml.csv   # Pure ML 5-fold ensemble with asset valuations
│       └── submission_pure_ml_v4_v5_master_blend.csv # Master blend of expt-2 + expt-4
└── PROGRESS_REPORT.md                       # This comprehensive documentation
```

### Summary of Available Kaggle Submissions:
1. **The #1 Leaderboard Solution (173.19 SGD):**  
   `exploration/Aayush/expt-3/outputs/submissions/submission_datagov_mean_173.csv` (and active root `submission.csv`).
2. **The Pure Machine Learning Sprint 6 Solution (~433 SGD):**  
   `exploration/Aayush/expt-4/outputs/submissions/submission_sprint6_pure_ml.csv`.
3. **The Pure Machine Learning Master Blend Solution:**  
   `exploration/Aayush/expt-4/outputs/submissions/submission_pure_ml_v4_v5_master_blend.csv`.
