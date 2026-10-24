# Problem Definition and Evaluation Protocol

## A. Task

The task is to predict the monthly rent of an HDB flat from its attributes. The
labelled file holds **150,000 rental approvals from 2021-01 to 2025-03**. The test
file holds **50,000 approvals from 2025-03 to 2026-07**, scored by root mean
squared error (RMSE) on the public leaderboard. A submission contains one row per
test listing, with columns `Id,Predicted` and zero-based sequential identifiers.

Each listing gives the approval month, town, block, street, flat type, flat model,
floor area, and lease commencement year. Two columns, `FURNISHED` and `FEE`, are
constant in the labelled data (`yes` and `0.0`), so they cannot separate listings
and are not used. Auxiliary tables supply building coordinates, MRT stations,
shopping malls, schools, COE quota premiums, and stock prices.

## B. The structure of the problem

Three properties of the data shape every later decision.

1. **The test set is the future.** Test months run from 2025-03 to 2026-07, so 16 of
   the 17 test months lie wholly beyond the last labelled month. Only March 2025
   appears in both files: 380 test rows, or 0.76% of the test set. Validation must
   therefore be chronological, and a random split would overstate performance.
2. **Rents trended, then flattened.** The monthly mean rent rose from about
   SGD 2,049 in 2021-01 to 2,924 in 2022-12, an increase of 42.7%, and then moved
   only from 2,922 in 2023-01 to 3,163 in 2025-03, an increase of 8.2%.
3. **Most listings are one of several similar flats.** Listings are
   spread across blocks and flat types, and about a quarter of validation rows
   have an exact twin that shares every model input (Section E).

## C. Validation design

All model comparisons fit on earlier months and evaluate on later ones.

| Window | Fitting rows | Fitting period | Validation rows | Validation period | Role |
| --- | ---: | --- | ---: | --- | --- |
| W1 | 117,988 | 2021-01 to 2024-04 | 32,012 | 2024-05 to 2025-03 | Main comparison, early stopping, and blend weights |
| W2 | 132,802 | 2021-01 to 2024-09 | 17,198 | 2024-10 to 2025-03 | Stability check with a later cutoff |

Three six-month rolling windows are used only to compare the three LightGBM
capacities:

| Window | Fitting rows | Validation rows | Validation period |
| --- | ---: | ---: | --- |
| R1 | 105,837 | 18,179 | 2024-01 to 2024-06 |
| R2 | 117,988 | 17,477 | 2024-05 to 2024-10 |
| R3 | 132,802 | 17,198 | 2024-10 to 2025-03 |

The three twelve-month backtest windows used only in the
[moving-market backtests](README.md#moving-market-backtests) start at 2022-01,
2023-01 and 2024-01 and fit on all earlier rows.

RMSE is the primary metric because it is the competition metric. Mean absolute
error (MAE) and the signed mean error (prediction minus actual) are reported as
secondary diagnostics, since a change can lower RMSE while raising MAE.

**Selection effect.** Early stopping, candidate selection, and blend weights all
use the labels of W1. Every W1 score is therefore a model-selection score, not an
estimate of performance on untouched data. W2 overlaps W1 (the last six months
of W1) and also contributes its labels to early stopping, so it is a stability
check rather than an independent test. The leaderboard is the only evaluation on
data that none of these choices have seen.

## D. Leakage controls

Rent history is the strongest signal in the feature set, and it is also the
easiest way to leak the target. Every target-derived feature for a listing in
month `t` uses only labels from months strictly before `t`; all listings from the
same calendar month are excluded together, which also handles the March 2025
overlap. Unit tests assert same-month exclusion, independence from future labels,
and fallback behaviour. The construction is documented in
[`data_processing.md`](data_processing.md).

## E. How much error is irreducible?

Two listings that share every model input receive the same prediction from any
deterministic model, because every engineered feature is a function of those
inputs and of the labelled history. The spread of rents inside such a group is
error that no model of these inputs can remove. This gives a measurable reference
for how far a model could still improve. The analysis is implemented in
[`punyaa/noise_floor.py`](../../punyaa/noise_floor.py).

Rows are grouped by all eight non-constant raw inputs: approval month, town,
block, street, flat type, flat model, floor area, and lease commencement year.
Two quantities are computed on rows that have at least one twin.

* The **estimated noise SD** divides the within-group sum of squares by its
  degrees of freedom (rows minus groups). It estimates the standard deviation of
  rent among identical listings, and it is the figure to compare a model with.
* The **strict lower bound** divides by the row count instead. It is a valid bound
  on the RMSE of any deterministic predictor for those exact rows, but a loose
  one: groups average about 2.2 rows, so the group's own mean is fitted to the
  rows it is scored on. For pairs it equals the estimate divided by the square
  root of two.

| Rows | With a twin | Estimated noise SD | 95% interval | Strict bound |
| --- | ---: | ---: | --- | ---: |
| All labelled months | 26.8% | 464.0 | not computed | 343.1 |
| W1 validation | 27.1% | 454.8 | 442.2 to 466.8 | 338.8 |
| W2 validation | 25.9% | 435.2 | 418.3 to 451.5 | 324.0 |

The interval resamples whole groups, 2,000 draws with seed 5228, because rows in
one group are not independent.

The frozen v2 ensemble scores as follows on the same rows:

| Window | Rows with a twin | Rows without a twin | All rows |
| --- | ---: | ---: | ---: |
| W1 | 452.8 | 466.9 | 463.2 |
| W2 | 436.1 | 445.8 | 443.3 |

**What this shows.**

1. On rows with a twin, the v2 RMSE (452.8 and 436.1) lies inside the interval
   for the noise estimate (442.2 to 466.8 and 418.3 to 451.5). The data are
   consistent with the model sitting at the irreducible level for those rows.
2. The intervals are 25 and 33 RMSE units wide, so the data cannot separate
   "at the floor" from a remaining gap smaller than that. The result is
   consistent with little headroom, and it does not prove it.
3. Rows without a twin score higher than rows with one, by 14.1 in W1 and 9.7 in
   W2. That could be more noise in sparse properties or modelling headroom for
   properties with little history. The two cannot be separated, because noise
   cannot be estimated for a row with no twin. This is a hypothesis for further
   work, not a finding.
4. The all-months figure of 464.0 is **not** a floor for the validation windows.
   The windows are quieter than the full period: the v2 RMSE in W2 is 443.3, below
   464.0. Comparisons must use the window-specific estimate.

**Limits.** Rows with a twin are about a quarter of the data and may differ from
the rest, for example in blocks with many identical units. The estimate assumes
the noise level is similar across properties, and it measures only variation
among identical listings, such as unrecorded floor level or condition. It is
computed on labelled months and says nothing about test-period noise.

## F. How results are read

* Differences below about 0.5 RMSE are not interpreted as improvements. The
  largest accepted ensemble gain (v3, 0.249 RMSE) has a month-resampled 95%
  interval of about [-0.002, 0.486], which includes zero. v3 was adopted below
  this threshold because it was the only candidate that lowered the blend, and
  the public score moved the same way (by 0.11); it is not an established gain.
* A candidate that improves one window and worsens the other is treated as noise.
* A candidate that improves RMSE and worsens MAE is reported with both numbers.
* Public leaderboard scores were supplied by the author. They are not stored in
  the metric files, and no private-leaderboard result is available.

## G. Reproduction

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v
python3 -m punyaa.noise_floor --validation-start 2024-05
python3 -m punyaa.noise_floor --validation-start 2024-10
```

Results are written to `outputs/punyaa_noise_floor/<validation-start>.json`.
