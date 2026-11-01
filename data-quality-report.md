# Data Quality Report

Run before further feature engineering, on top of the full `archived_modules/preprocessing.py`
pipeline (`train_feat`/`test_feat` — i.e., after the FLAT_TYPE normalization, casing
fixes, and `sg-hdb-block.csv`/macro joins already established in EDA). Organized by
check category: what it verifies, why it matters, what was found, and the
recommended action. Findings that change what ships are called out explicitly — none
have been applied yet, this is the analysis pass only.

**Code**: every check here is implemented in `archived_modules/data_quality.py` (functions
`check_numeric_ranges`, `find_sentinel_rows`, `check_sg_bounds`,
`check_cross_field_consistency`, `check_lease_completion_divergence`,
`flag_rent_per_sqm_outliers`, `check_same_key_rent_variance`,
`check_categorical_values`, `check_temporal_coverage`, and `run_full_report` which
runs all of them), and actually executed in `main.ipynb`'s "Data quality analysis"
section — this file summarizes results that come from running that code, not from
one-off analysis that isn't reproducible in the repo.

## A. Range / plausibility checks on numeric fields

**Why:** the fastest way to catch encoding errors, sentinel/placeholder values, and
unit mistakes before they silently corrupt a model — a `-1` or `9999` fed as a real
number is far more dangerous than a `NaN`, because it passes every `isna()` check.

**Checked:** min/max/zero-count/negative-count for every numeric feature
(`MONTHLY_RENT`, `FLOOR_AREA_SQM`, `LEASE_COMMENCE_DATE`, `remaining_lease`,
`MAX_FLOOR`, `YEAR_COMPLETED`, `LATITUDE`, `LONGITUDE`, `dist_mrt_km`,
`dist_mall_km`), plus a Singapore bounding-box check on coordinates.

**Found:**
- `MONTHLY_RENT`, `FLOOR_AREA_SQM`: clean ranges (\$300–7,600, 31–215 sqm), no
  zeros/negatives.
- `LATITUDE`/`LONGITUDE`: 100% within Singapore's actual bounds — the geo join isn't
  producing garbage coordinates.
- **`MAX_FLOOR` and `YEAR_COMPLETED` both have a `-1` sentinel value** — 159 rows in
  `train_feat` (0 in `test_feat`). Traced to source: 11 of 9,464 blocks in
  `sg-hdb-block.csv` carry `-1` in both fields (confirmed in the raw file, not a bug
  introduced by the join). These 159 rows currently feed `-1` into the model's
  **numeric** feature set as if it were a real max-floor/year value.

**Recommended action:** treat `-1` as missing for both columns and impute (e.g.
`TOWN`/`SUBZONE` median) rather than leave it as a literal numeric value — a fix,
not just a flag, since both columns are actively used features. Low effort, affects
only 159 train rows and 0 test rows, so bounded but real risk currently unaddressed.

**Applied.** `archived_modules/preprocessing.py::impute_sentinel_values` replaces `-1` with the
`SUBZONE` median computed from the other, non-sentinel blocks (confirmed every
affected `SUBZONE` — `tanglin halt`, `macpherson`, `clementi west` — has 25+ other
valid blocks, so no fallback-to-global-median case actually triggers). Applied on the
`sg-hdb-block.csv` reference table itself, before the join, so `train_feat`/`test_feat`
inherit identical corrected values. **Result: RMSE 457.47 → 457.37** (small but real —
this was 159 rows feeding a literal `-1` into two active numeric features).
`submission.csv` regenerated via `main.py` (verified changed, not just re-run).

## B. Cross-field logical consistency

**Why:** individually-valid values can still be jointly impossible — a rental
transaction can't predate the building it's in, and a 99-year lease can't have more
than 99 years remaining. These catch errors range checks miss.

**Checked:** `RENT_APPROVAL_DATE` year vs `YEAR_COMPLETED` (renting before built?),
`RENT_APPROVAL_DATE` year vs `LEASE_COMMENCE_DATE` (renting before lease starts?),
`remaining_lease` bounds, and `LEASE_COMMENCE_DATE` vs `YEAR_COMPLETED` divergence.

**Found:**
- Renting-before-built: **0 rows** — clean.
- Renting-before-lease-start: **1 row** (`RENT_APPROVAL_DATE` 2021-11,
  `LEASE_COMMENCE_DATE` 2022) → `remaining_lease` computes to exactly 100, one over
  the valid [0, 99] bound. Trivial in scale but a real off-by-one/rounding case.
- `LEASE_COMMENCE_DATE` vs `YEAR_COMPLETED` diverging by >2 years: **10,426 rows
  (~7% of data)**, even after excluding the `-1` sentinel — investigated and this is
  **not an error**: it reflects a genuine Singapore HDB phenomenon (lease
  commencement can be registered later than physical construction for older
  estates). Ruling this in as legitimate, not a data problem, avoided over-reacting
  to a large-looking number.

**Recommended action:** clip the 1 out-of-bounds `remaining_lease` row to 99 (trivial,
low-priority given it's a single row). No action needed on the lease/completion
divergence — confirmed legitimate.

## C. Systematic price-per-square-metre outlier detection

**Why:** the low-rent-tail investigation found implausible rent for a unit's size by
eyeballing a narrow, ad hoc rule (`FLAT_TYPE` in {4/5-room, executive} AND rent <
\$1,000). That rule is neither principled nor complete — it can't catch small units
renting implausibly *high*, or borderline cases just outside its arbitrary cutoff. A
price-per-sqm outlier check, computed *within* each `FLAT_TYPE` (so a 1-room and a
5-room aren't compared on the same absolute scale), generalizes this properly.

**Checked:** `MONTHLY_RENT / FLOOR_AREA_SQM`, flagged below the 0.5th or above the
99.5th percentile *within its own* `FLAT_TYPE` group. (A first attempt using a
3×IQR threshold caught nothing on the cheap side — mathematically, 3×IQR pushed the
lower bound negative given the spread in this data, silently defeating the check
entirely; the percentile-based version replaced it.)

**Found:** **718 rows below the 0.5th percentile, 726 above the 99.5th** (≈1.0% of
`train_feat` combined). The recalibrated check recovers **all 99/99** of the
previously-found anomaly rows as a strict subset, and roughly 7× more besides —
i.e., the earlier ad hoc rule was only seeing a fraction of this same underlying
issue.

**Recommended action:** this is the concrete, actionable output of this whole
report — a general-purpose anomaly flag (`rent_per_sqm` percentile within
`FLAT_TYPE`) ready to test the same way the 99-row rule was tested (exclude from
training only, never from evaluation, since `test.csv` will contain unidentifiable
equivalents). Worth re-running that ablation with this larger, more complete set
before concluding (as the narrower 99-row version did) that exclusion isn't worth
it — a ~1,444-row set is over 10× larger and could plausibly cross the threshold
where it matters.

**Tested, not adopted.** Re-ran the exclusion at full scale (1,297 rows in
`train_fold`): **RMSE 457.37 → 458.22 — worse**, the opposite direction from the
narrower rule's earlier marginal improvement. Re-tested the original 99-row-style
rule too, now against the sentinel-fixed baseline: also net-negative/flat (457.37 →
457.53). **Conclusion: the wider percentile-based rule over-flags genuine market
variance as "anomalous"** — many of the newly-caught rows are legitimate extreme
transactions (real premium or genuinely low-demand units), not data errors like the
original 99-row set, and removing them removes real training signal about the
distribution's tails that the model needs to predict similarly extreme rows in
`test.csv`. Excluding rows from training is **not adopted at any scale tested**; the
`-1` sentinel fix above is the only change that shipped from this report.

## D. Same-address/type/month rent variance — not a data error, a resolution limit

**Why:** if many rows share `TOWN`+`BLOCK`+`STREET`+`FLAT_TYPE`+`RENT_APPROVAL_DATE`
but have different `MONTHLY_RENT`, that could mean duplicate/conflicting records
(an error) — or it could mean the features simply can't distinguish individual units
within a block, and the "conflict" is real, unit-level market variance.

**Checked:** grouped by that 5-column key; measured group counts and within-group
`MONTHLY_RENT` standard deviation.

**Found:** **17,900 groups (39,233 rows, ~26% of `train_feat`)** have more than one
distinct rent value for the same address+type+month. Investigated rather than
assumed: median within-group standard deviation is only ~11% of the within-group
mean, and the largest single group (7 identical-`FLOOR_AREA_SQM` units, same
address/type/month) has rent ranging \$2,000–2,400 (std \$144, ~6.5% of its mean) —
consistent with genuine unit-to-unit variance (floor level, condition, negotiated
terms) that no available feature captures (there's no unit/floor-number column),
not a recording error.

**Inference — sets a realistic performance ceiling, doesn't call for a fix:** this
quantifies an *irreducible* noise floor distinct from the low-rent-tail anomalies
found earlier. Even a hypothetically perfect model using every available feature
still can't resolve rent differences between two otherwise-identical units in the
same block, because that information isn't in this dataset at all. Worth keeping in
mind when judging how much further RMSE can realistically drop — this is likely a
meaningful component of the ~\$457 current RMSE that no amount of feature
engineering on the *existing* columns can close.

## E. Categorical near-duplicate labels

**Why:** the `FLAT_TYPE` `"4 room"`/`"4-room"` issue from early EDA is exactly the
kind of bug this check is designed to catch elsewhere — two spellings of the same
category silently splitting a model's signal.

**Checked:** manual inspection of all `FLAT_MODEL` (20), `TOWN` (26) values for
near-duplicates beyond what normalization already fixed.

**Found:** **clean** — `FLAT_MODEL` values that look superficially similar
(`"model a"` / `"model a2"` / `"model a maisonette"`, `"improved"` /
`"improved maisonette"`, `"premium apartment"` / `"premium apartment loft"` /
`"premium maisonette"`) are genuinely distinct official HDB flat model designations,
not typos or spelling variants — no merging needed.

**Recommended action:** none — confirms the categorical cleanup from EDA was
sufficient, checked rather than assumed.

## F. Auxiliary dataset coordinate sanity

**Why:** `dist_mrt_km`/`dist_mall_km` and the geo join all depend on
`sg-mrt-stations.csv`/`sg-shopping-malls.csv`/`sg-schools.csv` having valid
coordinates — a single wrong lat/long in a reference file could silently corrupt
every distance calculated against it.

**Checked:** Singapore bounding-box check on all three auxiliary files.

**Found:** **clean** — 0 rows outside Singapore's bounds in any of the three files.

**Recommended action:** none.

## G. Temporal coverage

**Why:** a sudden gap or spike in monthly row counts could indicate a data
collection change (a new source merged in, a scraping outage) that shifts what the
model sees over time in ways unrelated to the actual rental market.

**Checked:** row count per `RENT_APPROVAL_DATE` month across all 51 months.

**Found:** **clean** — counts range smoothly between ~2,362 and ~3,705/month, no
dramatic gaps or spikes.

**Recommended action:** none.

## Summary

| check | finding | action taken |
|---|---|---|
| A. Numeric range/sentinel | `-1` sentinel in `MAX_FLOOR`/`YEAR_COMPLETED`, 159 rows | **Applied — imputed, RMSE 457.47→457.37** |
| B. Cross-field consistency | 1 row over `remaining_lease` bound; lease/completion divergence is legitimate | Not applied (trivial, single row); no action needed otherwise |
| C. Price-per-sqm outliers | 1,444 rows (0.5th/99.5th percentile within `FLAT_TYPE`), supersedes the earlier 99-row rule | **Tested — not adopted, made RMSE worse (457.37→458.22)** |
| D. Same-address rent variance | 39,233 rows, confirmed genuine unit-level variance | None — informational (sets an RMSE floor expectation) |
| E. Categorical near-duplicates | None found beyond already-fixed `FLAT_TYPE` | None |
| F. Auxiliary coordinate sanity | Clean | None |
| G. Temporal coverage | Clean | None |

## Outcome

One fix shipped: the `-1` sentinel imputation (`archived_modules/preprocessing.py`), moving
the validated benchmark from **RMSE 457.47 → 457.37**. `submission.csv` regenerated
via `main.py` and confirmed to actually change, not just re-run. The price-per-sqm
exclusion was tested properly (both at the full 1,444-row scale and the original
99-row rule re-checked against the new baseline) and confirmed **not worth
adopting at any scale** — the wider rule flags genuine market-variance rows as
"anomalous," and removing them costs the model real signal about the distribution's
tails. See `archived_modules/data_quality.py` for the reusable check implementations and
`main.py`'s docstring for the current reference metrics.
