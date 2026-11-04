# Initial EDA — Inference Summary

Source notebook: [`aayush_exploration.ipynb`](./aayush_exploration.ipynb) (steps 1–12).
Goal: predict `MONTHLY_RENT` for `test.csv` using `train.csv` + `data/auxiliary/*`.

Each section states the result actually observed, then the inference/action it implies for feature engineering and modeling.

## 1. Schema & target
- `MONTHLY_RENT` is the only train-only column → confirmed target (`FEE` is a red herring — constant `0.0` in both files).
- `FURNISHED` and `FEE` are zero-variance in train → **drop both**, no predictive value.
- `RENT_APPROVAL_DATE`: 51 unique months. `LEASE_COMMENCE_DATE`: 57 unique years (1966–2022). Both internally consistent (no gaps).

## 2. Target distribution
- Skew: raw = **0.386**, `log1p` = **−0.424**.
- **Inference:** the log transform doesn't reduce skew — it overcorrects into the opposite direction of similar magnitude. The raw target isn't heavily skewed to begin with. **Do not reflexively log-transform `MONTHLY_RENT`**; compare raw vs log1p target at model-evaluation time instead of assuming log is better.

## 3. Data-quality / train-vs-test consistency
- `FLAT_TYPE` had two spellings per category (`"4 room"` vs `"4-room"`, ~24% of train used the space variant). Normalizing merges them correctly (verified: `40688+13391=54079` for 4-room, etc.).
- Duplicate rows: **425 → 641** after `FLAT_TYPE` normalization. The extra 216 were only "hidden" duplicates because of the spelling inconsistency — normalizing categoricals *before* dedup-checking matters.
- Zero unseen categories in test for `TOWN` / `FLAT_TYPE` / `FLAT_MODEL` → no cold-start category risk.
- No missing values in either file.
- **Inference:** normalize `FLAT_TYPE` and lowercase/strip `TOWN`/`STREET`/`BLOCK` before anything else — every downstream step (dedup, joins, encoding) depends on this being done first.

## 4. Univariate profiling
- `TOWN` (26 values): `jurong west`, `tampines`, `sengkang` are the largest.
- `FLAT_TYPE` (6 values post-normalization): `4-room` (54,079) > `3-room` (49,024) > `5-room` (35,523) > `executive` (8,234) > `2-room` (3,098) > `1-room` (42). 1-room flats are rare on the open rental market — expected (elderly/flexi-scheme units).
- `FLAT_MODEL` (20 values): dominated by `improved` (44,008) and `model a` (42,745); long tail of ~15 smaller categories.

## 5. Bivariate vs target
- `FLAT_TYPE`/`FLOOR_AREA_SQM` clearly separate `MONTHLY_RENT` (larger flats → higher rent, monotonic).
- `TOWN` shows a wide rent spread across the 26 towns (visual boxplot ranking, see notebook).

## 6. Temporal analysis
- Train: **2021-01 → 2025-03**. Test: **2025-03 → 2026-07**. Almost no overlap — **this is a forecasting problem**, not i.i.d. regression.
- A 90/10 time-based split puts the cutoff at **2024-10-01**, leaving **17,198 rows (~11.5%)** for validation — large enough to trust.
- **Inference:** validate with a time-based split (train on dates < cutoff, validate on dates ≥ cutoff). A random k-fold split will look better than the model actually is on test.

## 7. Geospatial join (`sg-hdb-block.csv`)
- Join key `TOWN + BLOCK + STREET`, after casing/whitespace normalization: **100% match rate on both train and test**.
- **Inference:** the join is safe to rely on for every row — no fallback/imputation needed for missing lat/long.

## 8. Proximity features (MRT / mall / school)
- Distances are small dataset-wide: mean **0.59km** to MRT, **0.81km** to mall, **0.35km** to any school (max ~3km for all three). Singapore's HDB planning gives near-universal amenity access — limits how much variance these features can explain.
- Raw Pearson correlation with `MONTHLY_RENT`: `dist_mrt_km` **−0.085**, `dist_mall_km` **−0.083**, `dist_school_km` **−0.001**.

### 8b. Non-linear check (binned distance)
| bin | MRT mean rent | Mall mean rent | School mean rent |
|---|---|---|---|
| <0.5km | \$2,787 | \$2,796 | \$2,719 |
| 0.5–1km | \$2,673 | \$2,725 | \$2,705 |
| 1–2km | \$2,613 | \$2,635 | \$2,878 (n=2,406) |
| 2–5km | \$2,635 (n=927) | \$2,690 (n=1,737) | \$2,144 (n=57) |

- **MRT & mall**: real, monotonic ~\$160–175/month proximity premium concentrated in the <1km range (90% of rows). The uptick in the sparse 2–5km bin is noise (<1% of rows).
- **School**: non-monotonic even when binned (goes *up* at 1–2km) → confirms this is genuinely no relationship, not a non-linearity Pearson missed.
- **Inference: keep `dist_mrt_km` and `dist_mall_km` as features — they understate their real effect under plain Pearson r. Drop `dist_school_km`.**

### 8c. "Popular school" proxy
- `sg-schools.csv` has no ranking/popularity column. `TYPE_CODE ∈ {independent school, specialised independent school}` (the closest brand-prestige proxy) has **zero overlap** with `MAINLEVEL_CODE == primary` — Singapore's independent schools are all secondary/JC level.
- Fallback: nearest-**primary**-school distance (`MAINLEVEL_CODE` only) → correlation **−0.005**, no better than nearest-any-school (**−0.001**).
- **Inference: no usable "popular school" signal exists in this dataset. Both `dist_school_km` and `dist_primary_school_km` are dropped from the feature set.**

## 9. Macro auxiliary data (`sg-coe-prices.csv`, `sg-stock-prices.csv`)
- Correlation of monthly mean rent vs monthly mean stock `ADJUSTED_CLOSE`: **0.026** (~none) — **but this number is unreliable as computed**: `sg-stock-prices.csv` contains **77 different stock symbols** (from SGX blue chips at ~1,403 rows each down to smaller tickers with ~100–300 rows), averaged together without separating by symbol first. Mixing 77 incompatible price scales into one "average" is not a meaningful market index.
- `sg-coe-prices.csv` was aggregated to monthly `QUOTA_PREMIUM` but **never actually correlated against the rent trend** in the current notebook — open item.
- **Inference: undecided.** Don't include either as a feature yet. Needs a corrected pass (e.g. an equal-weighted basket of blue-chip symbols, or normalized returns, plus the missing COE correlation) before ruling in or out — see Next Steps.

## 10. Outlier review
- Extreme tail (0.2%, 316 rows, \$300–\$7,600) is **plausible given `FLAT_TYPE`/`FLOOR_AREA_SQM`** (e.g. \$300 = 2-room/46sqm, \$7,600 = 5-room/129sqm in a central town) — not data errors.
- **Inference: no need to clip/remove tail rows.**

## 11. Multicollinearity
```
                       flat_type_ord  FLOOR_AREA_SQM  LEASE_COMMENCE_DATE  remaining_lease
flat_type_ord                 1.000           0.952                 0.380            0.382
FLOOR_AREA_SQM                0.952           1.000                 0.343            0.345
LEASE_COMMENCE_DATE           0.380           0.343                 1.000            0.995
remaining_lease               0.382           0.345                 0.995            1.000
```
- `flat_type_ord` ↔ `FLOOR_AREA_SQM`: **0.952** (near-duplicate — room count determines size).
- `LEASE_COMMENCE_DATE` ↔ `remaining_lease`: **0.995** (arithmetic transforms of each other).
- **Inference:** for linear/regularized models, keep one from each pair (`FLOOR_AREA_SQM` over `flat_type_ord`; `remaining_lease` over raw `LEASE_COMMENCE_DATE`). Tree-based models are far less sensitive to this, but no reason to keep obviously redundant columns either way.

## 12. Train/test distribution shift
- Largest category-frequency shift across `TOWN`/`FLAT_TYPE`/`FLAT_MODEL` is **1.3 percentage points** (`model a`) — negligible.
- **Inference:** no compositional shift to correct for. The only real train/test gap is temporal (section 6), not a change in the property mix.

## Resolving the open items (post-baseline)

After the first baseline (`main.ipynb`, Ridge/GBM/LightGBM/XGBoost on `TOWN`+lease+distance features, XGBoost winning at MAE \$422), the "Open items" list from that pass was worked through one by one:

### Macro data (`sg-coe-prices.csv`, `sg-stock-prices.csv`) — fixed, not dropped
The original 0.026 correlation was an artifact: it averaged `ADJUSTED_CLOSE` across all 77 stock symbols in the file, most with only partial date coverage, at wildly different price scales (\$0.09 to \$75). Fixed by:
- Using only the **41 symbols with full daily history** across the whole date range (the rest are partial — later listings/delistings).
- **Normalizing each symbol's price to start at 1.0** before averaging (`build_stock_index`), so the index isn't dominated by whichever stock has a higher absolute price.
- Fixing `sg-coe-prices.csv`'s `MONTH` field (`'jan'`/`'feb'`/...) into a proper `'YYYY-MM'` key (`build_coe_index`) — the original pass computed this and never actually merged/tested it.

Corrected correlation with monthly mean rent: **stock index 0.72**, **COE premium 0.90** — both far from the earlier "no relationship" reading. Both auxiliary files also extend through test's date range (stocks to 2026-07, COE to 2026-08), so they give the model real values for test's months instead of forcing it to extrapolate blind. **Included as features** (`stock_index`, `coe_premium`) — validated by ablation below, not just correlation, since a high correlation between two upward-trending monthly series can be a shared-time-trend artifact rather than a causal one.

### `sg-hdb-block.csv` extra fields — SUBZONE replaces TOWN
Ablation-tested `MAX_FLOOR`, `YEAR_COMPLETED`, `REGION`, `PLANNING_AREA`, `SUBZONE` (all came free from the step-7 join, unused until now) against a time-based validation fold:

| feature set | MAE |
|---|---|
| baseline (`TOWN` + lease + distance) | 421.71 |
| + macro (stock_index, coe_premium) | 349.60 |
| + HDB extras (all 5 fields) | 346.74 |
| + both | 348.73 |

Both additions helped by roughly the same amount (~\$70-75 MAE) — far more than anything EDA's distance features contributed. Digging into *which* HDB fields mattered (permutation importance + isolation tests): **`SUBZONE` (156 categories, much finer-grained than `TOWN`'s 26) makes `TOWN` almost entirely redundant** — `TOWN`'s importance drops to ~0 once `SUBZONE` is present, and adding both together (388.98 MAE) performs the same as `SUBZONE` alone (389.91). `REGION`/`PLANNING_AREA` add nothing beyond `SUBZONE` either. `MAX_FLOOR`/`YEAR_COMPLETED` each add real, independent value. **Final feature set: drop `TOWN`/`REGION`/`PLANNING_AREA`, keep `SUBZONE` + `MAX_FLOOR` + `YEAR_COMPLETED`.**

### Model ranking flipped
With `SUBZONE` (a 156-category field) added, **LightGBM and sklearn's GBM both overtook XGBoost** (\$334 vs \$346 MAE) — the reverse of the original baseline, where XGBoost won on the coarser `TOWN`-only feature set. Lesson: a model comparison is only valid for the feature set it was run on; re-check rankings after a feature-set change instead of assuming the earlier winner still wins.

### Hyperparameter tuning — real, but small, and needs the right check
Implemented expanding-window time-series CV (`split.expanding_window_splits`, 4 folds, strictly time-ordered) for a small grid search over `n_estimators`/`max_depth`/`learning_rate`, run only on `train_fold` (never touching the final held-out `val_fold`, to avoid leaking it into model selection). Result: **CV picked a config that was actually very slightly *worse* than library defaults on the true held-out `val_fold`** (\$334.44 vs \$334.28 MAE) — the internal CV folds disagreed with the real holdout. Lesson: always confirm a CV-selected config against the untouched final fold before shipping it; don't trust the CV number alone. The pipeline now does this comparison automatically and falls back to defaults when tuning doesn't actually win.

### Environment reproducibility
Added `requirements.txt` (pinned versions of everything used) plus the macOS-specific `brew install libomp` step LightGBM needs — its compiled library links against the OpenMP runtime, which pip doesn't provide.

### Still unresolved — needs external input
**`Id` column format for `submission.csv`**: no `sample_submission.csv` exists anywhere in this repo. The pipeline currently uses the row's 0-based position as `Id`, which is an assumption, not a verified requirement — check the assignment/competition brief before treating any submission as final.

### Final numbers (time-based validation fold)
| model | MAE | RMSE |
|---|---|---|
| LightGBM (final, library defaults) | **\$334** | **\$458** |
| sklearn GBM | \$335 | \$459 |
| XGBoost | \$346 | \$469 |
| Ridge | \$401 | \$551 |
| Naive (predict-the-mean) | \$646 | \$798 |

## Net takeaways for modeling
1. Strongest raw signal: `FLOOR_AREA_SQM` / `flat_type_ord` (r≈0.33–0.35), `TOWN`.
2. Real but modest added signal: `dist_mrt_km`, `dist_mall_km` (use binned/non-linear form or a model that can find the threshold itself).
3. Confirmed dead: `dist_school_km`, `dist_primary_school_km` — drop.
4. Unresolved: `sg-coe-prices.csv`, `sg-stock-prices.csv` — needs a corrected re-test, not yet included.
5. Validation must be **time-based** (cutoff ≈ 2024-10-01), not random k-fold.
6. Drop `FURNISHED`, `FEE` (zero variance); dedupe train (641 rows) after normalizing `FLAT_TYPE`.
7. For linear models, resolve the two near-1.0 collinear pairs found in step 11.
