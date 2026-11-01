> **Note:** this report covers the earlier single-LightGBM pipeline, now in `archived_modules/`, `baseline_main.py` and `baseline_main.ipynb`. The final model is `main.py` + `modules/` (see `README.md`).

# Progress Report — HDB Monthly Rent Prediction

Goal: predict `MONTHLY_RENT` for the 50,000 rows in `data/test.csv`, using `data/train.csv`
(150,000 rows) and the auxiliary Singapore datasets under `data/auxiliary/`.

Full detail behind every decision below lives in [exploration/initial-eda-inference.md](../initial-eda-inference.md)
(EDA findings + the "Resolving the open items" section) and as inline comments in
`main.ipynb` / `archived_modules/*.py`. This file is the current-state summary.

**Two ways to run this**: `main.ipynb` is the full exploratory record — every model
comparison and the encoding/ensembling/momentum/data-quality investigations below,
each with reasoning — but re-running it in full fits dozens of models and takes 1-2
minutes. **`python main.py`** is a lean ~10s script (preprocessing → validation split
→ train only the current best model → verify → refit on all of train.csv → write
`submission.csv`) for routine re-verification — e.g. after a `archived_modules/*.py` change —
without re-running the whole notebook. It's a snapshot of the current best
model/hyperparameters, not a re-search; see its docstring for when to re-derive from
`main.ipynb` instead.

## Current result

**Correction: Kaggle scores this task on RMSE, not MAE.** Every "best model" decision
in the pipeline (the model comparison, the CV grid search, and the tuning-vs-defaults
check) is now sorted/decided by RMSE; MAE is kept alongside for interpretability only
(it's a more intuitive "average dollar error"). This is a real correction, not
cosmetic — see the tuning note below.

| model | RMSE | MAE |
|---|---|---|
| **LightGBM + address feature + median-rent anchor, residual target (shipping)** | **$451.87** | **$326.06** |
| LightGBM, tuned round 2 + address-level feature | $452.56 | $329.80 |
| LightGBM, tuned round 2 | $456.70 | $333.42 |
| LightGBM, tuned round 1 | $457.37 | $334.31 |
| LightGBM, library defaults | $458 | $334 |
| sklearn `HistGradientBoostingRegressor` | $459 | $335 |
| XGBoost | $469 | $346 |
| Ridge (linear) | $551 | $401 |
| Naive (predict the train mean) | $798 | $646 |

LightGBM's top three rows reflect the current pipeline (round 1 = data-quality
sentinel fix, round 2 = wider hyperparameter search below, address-level feature =
below in "Address-level rent-level feature"); the bottom four rows predate all three
and haven't been re-benchmarked (each fix/change is small in scale — sub-5-point RMSE
— so unlikely to change their relative ranking, but not re-verified).

**Live Kaggle leaderboard context**: this validation RMSE (~457) has tracked the
actual competition score closely — the live score (451.42, as of this round) came in
*better* than the local validation estimate, a reassuring sign the model isn't
overfitting to `val_fold`. Top team on the leaderboard: 432.82 — an 18.6-point
(~4.1%) gap, addressed below.

Measured on a **time-based validation split** (`archived_modules/split.py::time_based_split`),
not random k-fold — see "Why time-based validation" below. `submission.csv` (format
`Id,Predicted`, confirmed against the assignment's stated format) was generated via
`main.py` (see "Two ways to run this" above), from the current best LightGBM
hyperparameters (see `main.py`'s `BEST_PARAMS`) refit on all of `train.csv`.

LightGBM currently wins, reversing an earlier baseline where XGBoost won — see
"Feature set" below for why the ranking flipped. **The RMSE correction itself flipped
the tuning decision**: under MAE, the CV-selected hyperparameters scored slightly
*worse* than library defaults on the true held-out fold ($334.44 vs $334.28 MAE) and
were rejected; under RMSE, the same CV-selected hyperparameters score *better*
($457.47 vs $458.18 RMSE) and are accepted. Same grid search, same CV winner, opposite
shipping decision — purely from using the metric that actually matters. This is
checked automatically in `main.ipynb` (compares tuned vs. default on `val_fold` by
RMSE), not assumed either way.

## Features used, and why

All feature engineering lives in `archived_modules/preprocessing.py::build_features`; the final
selection for modeling lives in `archived_modules/modeling.py` (`GBM_NUMERIC_FEATURES` /
`GBM_CATEGORICAL_FEATURES`, plus `addr_level_feat` — see below — layered on via the
`numeric_features` parameter rather than folded into that constant, so earlier
notebook cells keep working unmodified; `main.py`'s `NUMERIC_FEATURES` is the actual
shipping list). Tree models (LightGBM/XGBoost/sklearn GBM) use native categorical
support — no one-hot/label/target encoding anywhere in the pipeline.

### Numeric (9)
| feature | source | why it's in |
|---|---|---|
| `FLOOR_AREA_SQM` | `train.csv` | strongest raw predictor found in EDA (r≈0.33 with rent) |
| `remaining_lease` | derived (`99 - (approval_year - LEASE_COMMENCE_DATE)`) | more directly meaningful than raw `LEASE_COMMENCE_DATE`; the two are collinear at r=0.995, only one is kept |
| `dist_mrt_km` | haversine distance to nearest `sg-mrt-stations.csv` entry | EDA found a real, monotonic ~$174/month proximity premium once binned — a plain correlation understated it |
| `dist_mall_km` | haversine distance to nearest `sg-shopping-malls.csv` entry | same pattern as MRT, ~$161/month premium when binned |
| `MAX_FLOOR` | joined from `sg-hdb-block.csv` | ablation-tested; helps beyond `SUBZONE` alone |
| `YEAR_COMPLETED` | joined from `sg-hdb-block.csv` | ablation-tested; helps beyond `SUBZONE` alone |
| `stock_index` | equal-weighted, return-normalized index built from `sg-stock-prices.csv` (41 full-history symbols only) | corrected version of an earlier broken calculation; extends through test's date range, giving the model real (not extrapolated) economic context for test's months |
| `coe_premium` | monthly mean from `sg-coe-prices.csv` | same rationale as `stock_index`; also extends through test's date range |
| `addr_level_feat` | derived (`archived_modules/preprocessing.py::build_address_momentum`/`add_address_feature`) | address-level (finer than `SUBZONE`) trailing rent-per-sqm, m-estimate shrunk toward `SUBZONE` mean — real, independent RMSE gain (see "Address-level rent-level feature" below); unlike the failed `SUBZONE`-level momentum, doesn't cannibalize `SUBZONE`'s own importance |

### Categorical (3, native dtype — no encoding)
| feature | levels | why it's in |
|---|---|---|
| `FLAT_TYPE` | 6 (after normalizing `"4 room"`/`"4-room"` spelling variants) | single strongest feature in every permutation-importance check |
| `FLAT_MODEL` | 20 | consistently useful, moderate importance |
| `SUBZONE` | 156 | joined from `sg-hdb-block.csv`; finer-grained than `TOWN` (26) — see below |

### Deliberately excluded, with reasoning
| feature | why dropped |
|---|---|
| `TOWN`, `REGION`, `PLANNING_AREA` | ablation-tested against `SUBZONE`: `TOWN`'s permutation importance drops to ~0 once `SUBZONE` is present (`SUBZONE`+`TOWN` together performs identically to `SUBZONE` alone), and `REGION`/`PLANNING_AREA` add nothing beyond it either — all three are strictly coarser versions of the same location signal |
| `dist_school_km`, `dist_primary_school_km` | tested in EDA: near-zero correlation (−0.001 / −0.005), non-monotonic even when binned into ranges — genuinely no relationship, not a linearity artifact. `sg-schools.csv` also has no popularity/ranking field to build a better proxy from (checked: `TYPE_CODE`'s "independent school" flag never overlaps with `MAINLEVEL_CODE == primary` in this data) |
| `flat_type_ord` (ordinal encoding of `FLAT_TYPE`) | collinear with `FLOOR_AREA_SQM` at r=0.952 — redundant once both are candidates; `FLOOR_AREA_SQM` is the more granular of the two |
| raw `LEASE_COMMENCE_DATE` | collinear with `remaining_lease` at r=0.995 — see above |
| `FEE`, `FURNISHED` | constant columns in both train and test (zero variance) — `FEE` is in fact a red herring, not the prediction target (`MONTHLY_RENT` is) |

## Why this feature set, in aggregate

Each addition was validated by actually measuring error on a held-out fold, not by
correlation alone. These ablation numbers were measured in MAE, before the RMSE
correction below — the ordering is the same under RMSE (verified for the final row:
$334 MAE ⇔ $458 RMSE), so the conclusion holds, but treat the MAE figures here as
approximate for the intermediate rows (they weren't re-measured in RMSE):

| feature set | MAE |
|---|---|
| `TOWN` + lease + distance only (first baseline) | $422 |
| + corrected macro data (`stock_index`, `coe_premium`) | $350 |
| + `sg-hdb-block.csv` extras (`SUBZONE`/`MAX_FLOOR`/`YEAR_COMPLETED`/`REGION`/`PLANNING_AREA`) | $347 |
| **final (`SUBZONE`+`MAX_FLOOR`+`YEAR_COMPLETED`+macro, `TOWN`/`REGION`/`PLANNING_AREA` dropped)** | **$334** (LightGBM, untuned — $458 RMSE; tuning improves this to $457 RMSE, see above) |

Feature engineering moved MAE by ~$88 (422 → 334); hyperparameter tuning moved it by
$0 (tuning didn't beat defaults on the true holdout) — feature choice mattered far more
than model tuning for this problem.

## Why time-based validation, not random k-fold

`train.csv` spans 2021-01 to 2025-03; `test.csv` spans 2025-03 to 2026-07 — almost no
overlap. This is a forecasting problem, not an i.i.d. regression one. Validating with a
random split would let the model "see" rows from dates later than its training rows,
producing a validation score that looks better than what the model can actually achieve
on genuinely future data. `time_based_split` holds out the most recent ~10% of `train.csv`
by date (cutoff ≈ 2024-10-01) to mimic the real train→test time gap; the same reasoning
drove `expanding_window_splits` for hyperparameter tuning (each CV fold's validation
slice is strictly later than its training slice).

## Encoding experiment — tested, did not beat the benchmark

Tested whether explicit categorical encoding (replacing native tree-model categorical
handling, and giving Ridge access to categoricals for the first time) beats the
**untuned** LightGBM baseline ($458 RMSE — this experiment predates the tuning-decision
fix above, which brought the actual shipping model to $457 RMSE). The gap between
every encoded variant and native categorical handling is well over $1 RMSE, so the
conclusion — no encoding beats native handling — is unaffected by which LightGBM
config is the exact reference point. Implementation in `archived_modules/encoding.py`.

**Encodings used:**
- `FLAT_TYPE` → **ordinal** (`flat_type_ord`; room count has a real order — already computed in `preprocessing.py`).
- `FLAT_MODEL` / `SUBZONE` → **K-fold out-of-fold target encoding** (mean rent per category, smoothed toward the global mean by an m-estimate) — nominal, no natural order, and `SUBZONE`'s 156 levels are too high-cardinality for one-hot on a linear model.
- `SUBZONE` alternative → **`LATITUDE`/`LONGITUDE` directly** (already available from the `sg-hdb-block.csv` join) — `SUBZONE` is fundamentally a discretized proxy for location.

**Results** (time-based validation fold):

| model | MAE | RMSE |
|---|---|---|
| **LightGBM, native categorical (current benchmark, shipping)** | **$334.28** | **$458.18** |
| LightGBM + encoded (lat/long variant) | $336.21 | $459.11 |
| XGBoost + encoded (lat/long variant) | $339.55 | $462.50 |
| LightGBM + encoded (ordinal+target, no native categorical) | $340.86 | $466.30 |
| XGBoost, native categorical | $346.44 | $469.10 |
| XGBoost + encoded (ordinal+target, no native categorical) | $347.07 | $469.77 |
| Ridge + one-hot(`FLAT_TYPE`,`FLAT_MODEL`) + `SUBZONE_te` | $373.16 | $522.98 |
| Ridge + encoded (lat/long instead of `SUBZONE_te`) | $379.19 | $529.31 |
| Ridge + encoded (ordinal + target-encoded) | $386.07 | $535.38 |
| Ridge, numeric-only (current, no categoricals) | $401.39 | $551.11 |

**Conclusion: no pipeline change.** LightGBM's native categorical handling remains
the best result overall. Three takeaways worth keeping:
- **Ridge improves substantially with encoding** ($401 → $373, ~7% better) — confirms
  the categoricals carry real linearly-usable signal, not just tree-exploitable
  non-linear signal, but Ridge still trails the tree models by a wide margin.
- **XGBoost specifically improves with the lat/long variant** ($346 → $340) — a real
  gain, though still short of LightGBM's result. Worth revisiting if XGBoost is ever
  preferred again for another reason (the model ranking already flipped once before,
  when `SUBZONE` was added — see above).
- **LightGBM gets slightly worse with any encoding tried** ($334 → $336-341) — its
  native categorical splitting is already effective here; compressing `FLAT_MODEL`/
  `SUBZONE` into one or two numeric columns discards information their full
  categorical structure preserves.

`final_model`, `best_model_name`, and `submission.csv` are all unaffected by this
experiment — it's a documented negative result, run for due diligence, not a change.

## Ensembling — investigated, no pipeline change

Tested whether blending the 3 already-fitted tree models (LightGBM tuned, sklearn
GBM, XGBoost), or adding genuinely different model types for diversity, beats
single-model LightGBM. Implementation in `archived_modules/ensemble.py`. Blend weights are
fit on **out-of-fold predictions from `train_fold`** (same expanding-window CV as
hyperparameter tuning) — never on `val_fold` directly, avoiding the same leakage
mistake fitting hyperparameters on `val_fold` would be.

**Results** (`val_fold` RMSE, honest check — blend weights fit on `train_fold` OOF only):

| approach | RMSE | inference |
|---|---|---|
| **LightGBM alone (shipping)** | **457.47** | reference point |
| Simple average (LightGBM + sklearn GBM + XGBoost) | 457.71 | worse — equal-weighting drags in weaker models |
| OOF-fit weighted blend (LightGBM 0.80 / sklearn GBM 0.20 / XGBoost ≈0) | 457.23 | real but tiny gain (~0.05%) — grid search correctly zeroed out XGBoost |
| Ridge stacking (OOF meta-learner, 3 models) | 478.40 | worse — meta-learner overfits given how correlated the 3 inputs are |
| Random Forest alone (diversity candidate) | 467.94 | individually weaker than any boosting model, as expected |
| k-NN alone (diversity candidate) | 485.74 | individually weakest, as expected |
| 5-model NNLS blend (+ Random Forest, k-NN) | 522.04 | far worse — NNLS put 75% of its weight on the two *weakest* models, overfitting the OOF sample |

**Why the gains are so small**: prediction correlation between the 3 boosting models
is very high (0.994 LightGBM↔sklearn GBM, ~0.97 vs XGBoost) — expected, since all
three train on identical features/data with a similar boosting mechanism, leaving
little room for ensembling to help. Adding Random Forest/k-NN *did* produce real
diversity (k-NN's error correlation with LightGBM drops to 0.93 vs 0.99 among the
boosting trio), but they're individually too weak, and letting an unconstrained
optimizer (NNLS) pick weights across 5 differently-strong models overfits the OOF
sample rather than finding generalizable signal — a useful general lesson: model
diversity only helps ensembling when the diverse models are also reasonably strong
individually, and when the weight-fitting procedure is constrained enough not to chase
sample noise (the simplex-constrained grid search stayed sane; unconstrained NNLS
did not).

**Conclusion: no ensemble ships.** The 3-model blend's gain (0.24 RMSE, ~0.05%) is
smaller than the tuning-decision fix's gain and doesn't justify tripling the models
the pipeline must train/maintain for a possibly-noise-level improvement. The
diversity-seeking attempt (5-model) actively backfired. `final_model` stays
single-model tuned LightGBM. Infrastructure (`archived_modules/ensemble.py`) is kept for reuse
if a future feature-set change reduces model correlation enough to revisit this.

## Residual diagnostics, log-target training, rare-category smoothing — done

Implemented in that order (diagnose → cheap independent test → targeted test),
matching the sequencing reasoning from chat: diagnostics first since it's nearly free
and determines whether the rest are worth pursuing; log-target next since it's cheap
and independent of any new feature; rare-category smoothing last since diagnostics
could (and did) predict its outcome in advance. Full code in `main.ipynb`'s
"Residual diagnostics" / "Log-target training" / "Rare-category smoothing" sections.

### 1. Residual diagnostics — the key finding

Broke `val_fold` residuals (`actual − predicted`, tuned LightGBM) down by time,
`FLAT_TYPE`, `SUBZONE`, and rent magnitude:

- **Overall bias**: mean residual ≈ +$28 (slight under-prediction on average).
- **No strong time-degradation** within `val_fold`'s 6-month window (RMSE stays in a
  441-474 band) — weakens, but doesn't rule out, the case for near-term extrapolation
  being the main problem (`test.csv` extends much further, to 2026-07, than
  `val_fold` does).
- **`executive` flats run notably worse** (RMSE 536 vs low-400s for other types).
- **`SUBZONE` sparsity does NOT correlate with error** — correlation(train row count,
  RMSE) ≈ −0.19 (weak), sparse (<100 rows) vs well-populated (≥100 rows) subzones have
  essentially identical mean RMSE. Directly predicts item 3's negative result below.
- **The dominant finding**: rent ≤$2,100 (~6.5% of `val_fold`) has RMSE ≈**1,097** —
  by far the worst segment — driven by systematic **over-prediction** (mean residual
  ≈ −966). The model predicts a "typical" rent for these units' features, but actual
  rent is far below that. This single segment is the largest contributor to the
  aggregate RMSE, and directly motivated testing log-target training next.

**Inference:** the aggregate RMSE number is misleading on its own — it looks like
diffuse, evenly-spread error, but it's actually dominated by one narrow, identifiable
segment (cheap properties) where the model has a specific, systematic failure mode
(over-prediction), not random noise. That reframes "improve the score" from "add more
features broadly" to "find what's different about anomalously-cheap listings that the
current features don't capture" — a targeted problem, not a diffuse one. `SUBZONE`
sparsity and short-term time drift are both ruled out as the cause.

### 2. Log-target training — real trade-off, net negative, not adopted

Trained on `log1p(MONTHLY_RENT)`, back-transformed with `expm1`, same tuned
hyperparameters/features. Result: **RMSE 464.67 vs 457.47 raw-target — worse overall**
(MAE also worse: 347.95 vs 334.44). Broken down by the rent bucket diagnostics
flagged:

| rent bucket | raw-target RMSE | log-target RMSE |
|---|---|---|
| ≤$2,100 | 1096.6 | 1055.3 (improved) |
| $2,100–2,700 | 436.0 | 418.8 (improved) |
| $2,700–3,200 | 303.8 | 316.2 (worse) |
| $3,200+ | 400.4 | 434.5 (worse) |

Log-target genuinely fixes the flagged low-rent segment, but systematically
over-corrects (pushes predictions up) across the mid/high buckets — which have more
rows and dominate the aggregate metric — netting an overall regression. **Not
adopted**; `final_model` stays trained on raw `MONTHLY_RENT`.

**Inference:** this confirms the diagnostic's root cause (a scale-dependent error,
exactly what `log1p` targets) but also proves a *global* fix is the wrong tool for a
*local* problem — reshaping the entire loss surface to help 6.5% of rows damages the
other 93.5%. The right fix has to act only where the problem is (the low-rent
segment), not everywhere. This directly rules out "just change the loss/target" as a
solution and rules *in* a segment-specific approach or a feature that gives the model
information it's currently missing about why these specific units are cheap.

### 3. Rare-category smoothing for `SUBZONE` — confirmed negative, as predicted

- **(a) Grouping sparse subzones** into `"other_subzone"` for native categorical
  handling: RMSE moves by <1 point either direction across thresholds (0/100/200
  training-row cutoffs), non-monotonically — noise, not a real effect.
- **(b) Increasing target-encoding smoothing** (`m` in `archived_modules/encoding.py`) from 10
  to 200: RMSE stays flat (~465.7–466.4), nowhere near closing the ~9-point gap to
  native categorical handling.

`SUBZONE` data-sparsity is genuinely not an error driver here, exactly as diagnostics
predicted. No pipeline change.

**Inference:** the diagnostic's correlation check (−0.19, weak) was a correct
prediction of a null result, confirmed properly rather than assumed — worth noting as
a case where a cheap diagnostic saved the cost of a deeper investigation (both the
grouping test and the smoothing sweep) turning out negative, without having to guess
in advance which one to skip. `SUBZONE` is already well-populated enough on average
(most of its 156 categories have well over 100 training rows) that the model doesn't
need help generalizing across it.

## Applying the top two suggested next steps — results

Implementation in `archived_modules/preprocessing.py` (`build_subzone_momentum`,
`add_momentum_feature`) and `main.ipynb`'s "Applying the top two suggested next
steps" section.

### 1. Neighborhood recent-rent momentum — tested, not adopted

Trailing rolling-average rent per `SUBZONE`, built strictly from **past** data
relative to each row's date (`train_fold` only when validating, mirroring what's
genuinely available at prediction time — for `test.csv`, most of it forecasts past
train's last known month given the 16-month horizon, where the feature goes stale and
forward-fills the last known value).

- Correlates with rent at ~0.58–0.59 (window 1/3/6 months all similar) — strong on
  its own.
- Overall RMSE gain: marginal at best (window=6: 457.47 → 457.23, ~0.05%; window=1
  actually worse at 458.63).
- **Does not fix the flagged low-rent segment** — that bucket's RMSE is essentially
  unchanged (very slightly worse: 1096.6 → 1101.7).
- Permutation importance: `momentum` gets modest importance (~0.02), but `SUBZONE`'s
  own importance *drops* (0.30 → 0.23) rather than both staying strong — the two are
  substitutes, not complements.

**Inference:** the correlation number was misleading in isolation — `momentum` is
highly correlated with rent because it's essentially a smoothed, time-lagged version
of what `SUBZONE`'s native categorical splits already encode (a tree model with
enough per-category data can already learn "this subzone's typical rent level"
directly). Adding it doesn't give the model new information, it gives the model a
second way to express information it already had. More importantly, since it doesn't
touch the low-rent segment at all, this **rules out "the model doesn't know the
neighborhood's recent price level"** as the explanation for that segment — the
investigation needed to look elsewhere, which motivated the next step.

### 2. Diagnosing the low-rent tail — a rare, likely irreducible anomaly, not a feature gap

Compared the worst 200 over-predicted rows (all in the low-rent segment) against the
rest of `val_fold`, and inspected the extreme cases directly. The samples were
striking: large, decent-condition units (5-room/executive, 100+ sqm, 50-90 years
remaining lease) renting for **$600–$1,500** — implausible for the unit, not merely
"a cheap property" (the model's $3,000–4,600 predictions look like the *plausible*
values here). Quantified across all of `train.csv`: flats of type `4-room`/`5-room`/
`executive` renting under $1,000 — **99 rows (0.066% of all data)**, spread evenly
across `FLAT_MODEL` (10+ different models), `TOWN` (20+ different towns), and time
(nearly every month) with no concentration in any one dimension.

Quantified their actual contribution: these 12 rows (in `val_fold`) are **1.1% of the
low-rent bucket's row count but 6.1% of its squared error** — disproportionate, but
not the dominant driver of that bucket's poor RMSE. Excluding all 87 such rows from
`train_fold` (evaluated against the full, unmodified `val_fold` — can't drop them from
evaluation since `test.csv` will contain unidentifiable ones too) moves RMSE by
457.47 → 457.34, under a point.

**Inference:** these are real, but rare and evenly-scattered — the hallmark of
irreducible noise (likely non-market-rate transactions: family arrangements, unusual
listings, or similar) rather than a learnable pattern any feature could capture. No
single `FLAT_MODEL`/`TOWN`/time cluster to build a targeted feature around, and even
perfect handling of all of them barely moves the aggregate metric. **The low-rent
segment's poor RMSE is not one fixable problem** — it's mostly the ordinary
difficulty of predicting the tail of a skewed distribution (the same mechanism
log-target partially addressed, at too high a cost elsewhere), plus this small,
likely-irreducible noise floor. Further feature engineering aimed specifically at
this segment is probably diminishing-returns territory. `final_model` and
`submission.csv` are unchanged by both investigations.

## Data quality analysis — one fix shipped

Full writeup in [data-quality-report.md](data-quality-report.md) (code in
`archived_modules/data_quality.py`, seven check categories A-G). One concrete fix applied:
`MAX_FLOOR`/`YEAR_COMPLETED` had a `-1` sentinel value hiding in `sg-hdb-block.csv`
for 159 rows, undetected by any `isna()` check, feeding literal `-1` into two active
numeric model features. Imputed via `SUBZONE` median
(`archived_modules/preprocessing.py::impute_sentinel_values`) — **RMSE 457.47 → 457.37**.

The report's other actionable lead — excluding price-per-sqm outlier rows from
training, generalized from the earlier 99-row low-rent-tail anomaly to a principled
1,444-row rule (percentile-within-`FLAT_TYPE`) — was tested properly and **not
adopted**: it makes RMSE worse (457.37 → 458.22) at full scale, and re-testing the
original narrower rule against the new baseline is now a wash too (457.37 → 457.53).
The wider rule flags genuine market-variance rows, not just errors — removing them
costs the model real signal about the distribution's tails. Five of the seven check
categories (cross-field consistency beyond one trivial row, categorical duplicates,
auxiliary coordinates, temporal coverage, same-address variance) came back clean or
were informational only.

## Closing the Kaggle leaderboard gap (round 2): hyperparameter search + interaction feature

Prompted by real competition context: live score 451.42 vs. leaderboard-leading
432.82 (18.6 points, ~4.1% gap). The original tuning grid was only 8 combinations over
3 params (`n_estimators`, `max_depth`, `learning_rate`) — never touched `num_leaves`,
regularization, or the classic "many trees at a low learning rate" GBM lever.

**Wider randomized search** (30 trials, expanding-window CV, 8 params:
`n_estimators` up to 2000, `learning_rate` down to 0.01, `num_leaves`,
`min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`):
CV `mean_RMSE` across trials ranged 582–626 with ±114-116 std per trial — i.e. **the
CV signal itself is too noisy to discriminate between these configs**, confirming
(not just assuming) that hyperparameter tuning has very little headroom left for this
feature set, not that the earlier grid was too narrow to have found it. Still,
confirmed the CV winner (`n_estimators=600, learning_rate=0.03, num_leaves=15,
min_child_samples=100, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1,
reg_lambda=0.1`) against the true held-out `val_fold`, the metric that actually
matters: **RMSE 457.37 → 456.70 — real, adopted.**

**`SUBZONE` × `FLAT_TYPE` interaction feature** (out-of-fold target-encoded, same
leakage discipline as `archived_modules/encoding.py`; 598 combinations in `train_fold`, ~221
rows each): standalone, **RMSE 457.37 → 456.68** — essentially the same-sized gain as
the hyperparameter tuning, independently. But **combined, the two do not stack**
(456.88 — worse than either alone) — the tuned hyperparameters were selected via CV on
the feature set *without* this interaction, so they're not necessarily right once it's
added; combining properly would need its own re-tuning pass. **Not adopted this
round** — shipped the hyperparameter tuning alone (equal-sized gain, no added
pipeline complexity). Worth revisiting: re-run the hyperparameter search with the
interaction feature included from the start, rather than bolting it onto
already-tuned params.

`main.py`'s `BEST_PARAMS` updated; `submission.csv` regenerated and confirmed to
change (`Id 0`: \$2,670.93 → \$2,615.16).

## Amenity density & MRT connectivity — tested, not adopted

Tested before the joint hyperparameter re-tune below, using the round-2 tuned
hyperparameters (`main.py`'s current `BEST_PARAMS`) as the base. Implementation in
`archived_modules/preprocessing.py` (`add_density_within_radius`, `add_nearest_poi_attribute`,
`add_mrt_line_count`).

- **Amenity density** (count of MRT stations/malls within a radius, tested at
  0.5/1.0/1.5/2.0km): RMSE ranged **456.62–456.99** against a 456.70 baseline — all
  within noise, non-monotonic across radii, no real signal at any threshold tried.
- **MRT connectivity** (line count — 1/2/3 — at the nearest station, derived from
  `sg-mrt-stations.csv` having one row per line at each of its 243 entries, 192
  unique physical stations): **RMSE 456.70 → 457.14 — a real regression**, not just
  noise.
- **Combined**: 456.70 → 456.81 — also net negative.

**Inference:** same pattern as `momentum` in round 1 — `SUBZONE`'s native categorical
splits already capture most of the location-quality signal these geo-derived
features approximate (an interchange station's presence correlates with which
`SUBZONE` it's in), so they add redundant noise rather than genuinely new
information. **Neither adopted.** `main.py`/`submission.csv` unchanged by this
section.

## Joint re-tune: hyperparameters + interaction feature together — conclusive negative

Ran a fresh 15-trial randomized hyperparameter search (same expanding-window CV, same
8-param space as round 2) on the feature set *including* the `SUBZONE`×`FLAT_TYPE`
interaction, rebuilt fresh per fold with the same out-of-fold discipline
(`archived_modules/encoding.py`). **Result: the CV winner converged to nearly the same
hyperparameters as round 2** (`n_estimators=600, learning_rate=0.03, num_leaves=15,
min_child_samples=100, subsample=0.8, colsample_bytree=0.6, reg_lambda=0.1` — only
`reg_alpha` differs, 0 vs 0.1), and confirmed on the true held-out `val_fold` at
**RMSE 456.88 — identical to the earlier mismatched-hyperparameter combination**, not
an improvement.

| approach | RMSE |
|---|---|
| **Round-2 hyperparameters alone (shipping)** | **456.70** |
| Interaction feature alone | 456.68 |
| Jointly re-tuned hyperparameters + interaction | 456.88 |
| Mismatched hyperparameters + interaction | 456.88 |

**Inference:** this is a conclusive result, not an inconclusive one — the search
genuinely explored the joint space and consistently landed back at essentially the
same configuration, meaning the interaction feature and hyperparameter tuning are
capturing the **same underlying signal**, not complementary ones; there is no
"better joint configuration" waiting to be found here. **No pipeline change** — the
simpler option (round-2 hyperparameters alone) is equally good and stays shipping.
This line of investigation (round 1 + round 2 + this joint re-tune) is now closed.

## CatBoost — tested properly, close but doesn't beat LightGBM

Genuinely different mechanism (ordered target statistics for categoricals, automatic
feature combinations, symmetric trees, ordered boosting — see the CatBoost
explanation earlier in this session) — worth testing on its own merits, not assumed
better or worse. Two rounds:

- **First attempt used early stopping on a carved-out 10% slice of `train_fold`**:
  performed notably worse (best 472.64 RMSE) — but this conflated two effects
  (less training data *and* a stopping point tuned to the wrong time period), not a
  genuine CatBoost weakness. Caught and re-tested properly rather than reported as-is.
- **Re-tested on the full `train_fold` with fixed iteration counts** (no carve-out):
  consistently landed in the **457.75–458.88 RMSE range** across `depth` (4/6/8),
  `learning_rate` (0.03/0.05), and `l2_leaf_reg` (3/10) — best config `depth=8,
  learning_rate=0.03, iterations=1500` → **457.75**, still ~1.05 points (0.23%) worse
  than LightGBM's tuned 456.70, and consistently so across every configuration
  tried, not a fluke of one run.

**Inference:** close, but a real and consistent gap, not noise-level (compare to the
±0.02 differences that *were* noise-level in round 2's hyperparameter search).
CatBoost's specific mechanisms (ordered target statistics, automatic combinations)
don't translate into an advantage on this particular feature set — plausibly because
`SUBZONE`'s native histogram-based splitting in LightGBM already extracts most of
what target statistics would offer here, consistent with every other categorical-
encoding variant tested this session landing at or below the native-handling
baseline. Also markedly slower to train (25–55s per fit vs. LightGBM's near-instant
fits), a secondary reason not to switch even if the gap were smaller. **Not
adopted** — `main.py`/`submission.csv` unchanged.

## Address-level rent-level feature — real gain, adopted

Prompted by digging further into the low-rent-tail question (rent ≤$2,100 in
`val_fold`, RMSE ≈1,102 — see "Residual diagnostics" above). Confirmed concretely
what `data-quality-report.md` §D already quantified in aggregate: e.g. one address
(Ang Mo Kio blk 624 Ave 4, 68 sqm, `new generation`) shows rent of $2,900 / $2,950 /
$2,500 / **$1,650** / $2,800 / $2,500 across Oct 2024–Mar 2025 — same address, same
`FLAT_TYPE`, same `FLOOR_AREA_SQM`, same `FLAT_MODEL`, adjacent months, and no column
in the dataset distinguishes the $1,650 outlier from its neighbors. This is the tail's
dominant mechanism: genuine per-unit noise (floor, condition, negotiated terms — none
of it recorded here), not a missing feature. Also re-confirmed `FEE`/`FURNISHED` are
truly constant (no hidden signal) and checked for a round-number/subsidized-rent
pattern — found nothing there either.

Two ideas were tested against that finding:

- **Robust loss functions** (LightGBM `huber`/`quantile`(median)/`fair` objectives —
  hypothesis: fix the segment's over-prediction more surgically than log-target's
  whole-distribution reshape did). **Negative.** `huber`'s delta was swept 50→1500 to
  rule out a badly-scaled default; every variant landed at or above the squared-loss
  baseline on both overall and low-bucket RMSE (best: `huber` alpha=1500 at 456.90
  overall / 1,105.0 low-bucket, vs. squared loss's 456.70 / 1,101.9). **Closed, not
  adopted.**
- **Address-level (TOWN+BLOCK+STREET — finer than `SUBZONE`) trailing rent-per-sqm
  feature**, shrunk toward the `SUBZONE` mean via an m-estimate (same smoothing idea
  as `archived_modules/encoding.py`'s target encoding). Unlike `SUBZONE`-level momentum
  (tested above — NOT adopted, redundant with `SUBZONE`'s own categorical splits), a
  per-building signal captures something a subzone-wide category boundary can't (one
  block running consistently cheaper/pricier than its subzone's average).

**Result: real gain — val_fold RMSE 456.70 → 452.56** (MAE 333.42 → 329.80), held up
across 3 of 4 expanding-window CV folds (mean 582.27 → 578.18, one fold essentially
flat). Feature importance lands comparably to `stock_index`/`FLOOR_AREA_SQM`, and
`SUBZONE`'s own importance is unaffected — genuinely new signal, not a redundant copy
the way `SUBZONE` momentum was.

**Important: this does NOT specifically fix the low-rent tail.** The gain is
concentrated in the normal-rent bucket (372.96 → 369.94 RMSE); the low-rent bucket
itself only improves marginally (1,101.9 → 1,090.2, ~1%). Given the concrete
same-address/type/size/month evidence above, the tail itself still looks structurally
close to its ceiling — this is a good general-purpose feature found *while*
investigating the tail, not a targeted fix for it.

**Implementation**: `archived_modules/preprocessing.py::build_address_momentum` (leakage-safe
trailing mean, same discipline as `build_subzone_momentum` — aggregates to
address+month first, then an expanding shift(1) mean, so a month never includes
itself even within its own source data) and `add_address_feature` (merges the trailing
value onto a frame, m-estimate shrinkage toward the `SUBZONE` mean, forward-fills a
address's last known state only for months genuinely absent from the source, e.g. all
of `test.csv`'s horizon). `archived_modules/modeling.py`'s `_prep_gbm_frame`/`fit_*`/
`predict_test` gained an optional `numeric_features` parameter (default: the
unchanged `GBM_NUMERIC_FEATURES`) rather than mutating that constant directly, so
every earlier notebook cell keeps working if `main.ipynb` is re-run top-to-bottom
from a blank kernel.

**Adopted.** `main.py`'s `NUMERIC_FEATURES = GBM_NUMERIC_FEATURES +
['addr_level_feat']` now ships, wired through both the `val_fold` check and the final
refit-on-`train_feat`/predict-`test_feat` path (each with its own `build_address_momentum`
call — `train_fold`'s table is never reused for the final fit, and vice versa).
`submission.csv` regenerated via `main.py` and confirmed to actually change (`Id 0`:
\$2,615.16 → \$2,677.50).

**Worth revisiting**, same lesson as the `SUBZONE`×`FLAT_TYPE` interaction episode:
`BEST_PARAMS` was tuned without this feature present, so a joint hyperparameter
re-tune with it included from the start might do better still — not done yet since
the gain already clears the bar with the untouched hyperparameters.

## Optuna hyperparameter search — tested, does not beat the current shipping config

Per this file's earlier "Known limitations" note that a formal Bayesian optimizer
"hasn't been tried as a distinct method" (only randomized/grid search had) — tested
properly rather than left as a hunch. `archived_modules/modeling.py::optuna_search_lgbm`
(TPE sampler) run for 100 trials over the same 8-param space and the same
expanding-window CV discipline as the earlier 30-trial/15-trial randomized searches
(`n_estimators` up to 2000, `learning_rate` down to 0.01, `num_leaves`,
`min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`),
against `train_fold` with the address-level feature included (the current shipping
feature set). Took 71.8 minutes for all 400 fits (100 trials × 4 folds).

**Result: negative.** Optuna's CV-selected best trial (`n_estimators=424,
learning_rate=0.024, num_leaves=162, min_child_samples=115, subsample=0.976,
colsample_bytree=0.639, reg_alpha=0.0048, reg_lambda=0.0002`, CV mean_RMSE 579.89)
confirmed on the true held-out `val_fold` at **RMSE 453.71 — worse than the current
456.70-tuned/452.56-with-address-feature shipping config's 452.56**, by about 1.15
points (0.25%). The top 10 trials span quite different hyperparameter values
(`n_estimators` 291–480, `num_leaves` 138–185, `learning_rate` 0.022–0.033) yet their
CV `mean_RMSE` is essentially flat (579.89–580.53) — the same noise-floor pattern the
30-trial/15-trial randomized searches found, now confirmed with a genuinely different
search method rather than just a wider/different sample of the same one.

**Inference:** this empirically confirms, rather than merely repeats, the earlier
"Known limitations" note — the CV signal in this region is too noisy for *any*
search strategy (grid, random, or Bayesian) to reliably find a config that beats
what's already shipping. **Not adopted** — current `BEST_PARAMS` stay shipping;
`main.py`/`submission.csv` unchanged. Implementation code and results are recorded in
`main.ipynb`'s "Optuna hyperparameter search" section (not re-run there — the 71.8
minute runtime isn't worth repeating for a result already confirmed by script).

## Median-rent anchor and residual target — large long-horizon gain, wired into main.py

Roadmap item 1 from the RMSE bottleneck report. The root pipeline never used
`sg-hdb-median-rent.csv` (official quarterly median rent by town and flat type, populated
through 2026-Q2), while `expt-2` built its market anchor from it. Implemented as
`archived_modules/preprocessing.py::build_median_anchor_table` / `fit_anchor_town_ratios` /
`add_median_rent_anchor`, with a fallback chain (exact cell, then interpolation within the
cell for up to 2 quarters, then the cross-town median for that flat type times the town's
own historical ratio, then the cross-town median, then a global mean) and an
`anchor_level` flag recording which rule fired. Data notes: the file spells
`QUEENSTOWN ` with a trailing space in recent quarters, so town names must be stripped
before matching (an unstripped ad hoc check wrongly suggested Queenstown had no recent
data); it repeats 320 rows; and it fills only ~55% of its town x flat-type cells per
quarter. After cleaning, 96.6% of train rows and 96.9% of test rows match an exact cell
(test: 1.1% interpolated, 2.0% via the ratio fallback).

Three variants of the shipping LightGBM were scored on two holdouts
(`exploration/median_anchor_experiment.py`; every encoder fit on that holdout's training
rows only): **A** current model, **B** anchor and `anchor_level` as extra features, **C**
residual target (rent minus anchor, anchor added back to predictions). The second holdout
is horizon-matched: train through 2023-11, validate 2023-12 to 2025-03 (16 months, the
test set's distance from training). Differences are paired-bootstrap intervals resampled
in address-month blocks.

| Holdout | A current | B anchor features | C residual target |
|---|---|---|---|
| 6-month fold, RMSE | 452.56 | 451.62 (-0.94, CI -1.88 to -0.03) | 451.87 (-0.70, CI -1.98 to +0.52) |
| 6-month fold, MAE | 329.80 | 327.47 | 326.06 |
| 16-month holdout, RMSE | 600.72 | 503.12 (-97.6, CI -99.9 to -95.2) | **492.77** (-108.0, CI -111.4 to -104.3) |
| 16-month holdout, MAE | 489.90 | 386.01 | 360.07 |

By months ahead of the cutoff on the long holdout, A swells to 720-786 RMSE in months
2-4 while B and C stay at 515-570, and C leads B through months 2-8. On the 6-month fold
the tail bucket (rent <= 2,100) gets slightly worse (1,090 to 1,096 for B, 1,108 for C)
while the rest improves (369.9 to 367.5 and 365.4), so the anchor helps the market level,
not the tail.

**Decision: C wired into `main.py`** (`NUMERIC_FEATURES` gains `anchor` and
`anchor_level`; the model fits rent minus anchor and the anchor is added back; the
validation check is `{'MAE': 326.06, 'RMSE': 451.87}`). It passes the adoption rule set
beforehand: a decisive win on the long holdout, no loss on the 6-month fold. On that fold
alone the gain is not significant for C; **B is the conservative alternative** (-0.94,
borderline significant) if the residual formulation proves fragile.

**Caveats.** (1) The long holdout trains on only 34 months that end in a steep inflation
phase, so A's collapse there probably overstates what the full-data model would suffer on
test (its last recorded leaderboard score was only ~5 points better than its local
validation, i.e. test was kinder); what it does show is that the pre-anchor model's
handling of time is fragile. (2) A validation row's anchor is the official median of its
own town, flat type and quarter, which aggregates actual rents including that row's. That
mirrors the test setting only while the file's test-quarter values are legitimately
usable; this stays a course-rules question, as does the external-data question for
expt-4. (3) Hyperparameters were tuned for the direct target and not re-tuned.
(4) Not yet verified on the leaderboard.

A candidate submission was written to `submission_residual_anchor.csv` (50,000 rows, no
nulls, mean 3,163, test anchor-level share above). **The root `submission.csv` was not
touched** and is still byte-identical to expt-3's registry-lookup file; replacing it is
left to the user.

## Remaining next steps — feature engineering

Re-ranked after the median-rent anchor; CatBoost, the interaction feature, robust losses
and Optuna remain exhausted:

1. **Score the candidate on the leaderboard** (`submission_residual_anchor.csv`) and log
   local against public score, then decide what to do about the root `submission.csv`
   (still the registry-lookup file).
2. **Hierarchical block by flat-type target encoding, fit inside each fold** — expt-2's
   second winning idea; test that it stacks with `addr_level_feat` and the anchor.
3. **Express the address feature relative to the anchor** (a premium, not absolute rent
   per sqm), so it stays valid as rents move; the anchor now makes this straightforward.
4. **Verify the CatBoost blend on the new setup** (4 CV folds plus paired bootstrap,
   seed bagging); the single-check 0.84 gain predates the anchor.
5. **Re-tune hyperparameters for the residual target** — low expectations, since CV
   cannot discriminate configs in this region.
6. **Recency weights, rent-per-sqm target, seasonality** — cheap, low confidence.
7. **Closed, do not repeat unless the feature set changes**: momentum, low-rent-tail
   tricks, encodings, density/connectivity, interaction feature, robust losses, Optuna,
   plain GBM-family blends.

## Known limitations / not yet done

- Hyperparameter search has now been tried via three distinct strategies — grid
  search, randomized search (30-trial and 15-trial, 8 params each), and a 100-trial
  Optuna/TPE search (see "Optuna hyperparameter search" above) — all confirmed
  against the true held-out fold, all finding the same noisy, indiscriminate CV
  signal in this region. This is now a well-established property of this feature
  set, not an untested hunch; further search-strategy changes alone are unlikely to
  help without a feature-set change first.
- `sg-stock-prices.csv`'s equal-weighted index uses simple price normalization, not
  proper return-weighting or an official index (e.g. STI) — a reasonable proxy, not a
  verified-accurate one.
- Duplicate rows in `train.csv` (641, found after fixing the `FLAT_TYPE` spelling
  inconsistency) are dropped; not deeply investigated for whether they're genuine
  re-listings vs scraping artifacts.
