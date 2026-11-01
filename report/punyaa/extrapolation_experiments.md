# Extrapolation and Recalibration Experiments

## A. Motivation

The labelled data ends at 2025-03 and the Kaggle test months run from 2025-03 to
2026-07, so most scored rows lie beyond the last month the model can observe.
Gradient-boosted trees are piecewise constant, so for those rows every model in
the preceding sections predicts a price level it has already seen. None of the
experiments in [`model_experiments.md`](model_experiments.md) or
[`improvement_experiments.md`](improvement_experiments.md) isolates that
limitation: all of them vary the feature set, the model family, or the blend
weights, and none changes what a leaf is able to represent.

Two candidate remedies were tested, both of which change leaf behaviour rather
than the inputs:

1. **Piecewise-linear leaves** (`linear_tree=True`). Each leaf fits a linear
   model instead of a constant, so the ensemble can continue a trend past the
   observed range of `approval_date_num` and `FLOOR_AREA_SQM`.
2. **Monotone constraints.** Six feature-to-target signs that a domain expert can
   assert without inspecting the target (larger floor area, later approval date,
   and higher historical rent all raise rent) are imposed as hard constraints.
   An extended set adds amenity proximity, where the sign is defensible but
   confounded by region.

A third question followed from the diagnostics: whether the v2 ensemble is
**miscalibrated in level or spread**, which a single affine correction would fix
more cheaply than any model change.

Both the v2 component predictions and the fitted iteration counts were held
fixed, so no reported difference comes from changing the baseline.

## B. Protocol

The two chronological windows from
[`improvement_experiments.md`](improvement_experiments.md) were reused unchanged:
117,988 fitting rows before 2024-05 against 32,012 validation rows, and the
later 2024-10 cutoff against 17,198 validation rows. The frozen v2 blend is
rebuilt from the stored component predictions using the v2 weights, so the
baseline is bit-identical to the one in the earlier document.

The control, `lightgbm_detailed` at its stored 251 iterations, reproduced its
recorded **RMSE 468.2615** against the previously recorded 468.26152. That
confirms the feature pipeline, split, and target vector are unchanged before any
candidate is compared.

The experiments live in [`punyaa/extrapolation.py`](../../punyaa/extrapolation.py)
rather than in `improve.py`, because that module's checkpoint fingerprint hashes
its own source: editing it would invalidate every cached v2 and v3 fit.

## C. Standalone results

| Candidate | 2024-05 RMSE | 2024-10 RMSE | Change vs control (2024-05) |
| --- | ---: | ---: | ---: |
| `lightgbm_detailed` (control) | **468.262** | **448.688** | n/a |
| `lightgbm_linear_tree_reg` | 470.749 | 448.269 | +2.487 |
| `lightgbm_linear_tree_monotone` | 471.735 | 450.016 | +3.473 |
| `lightgbm_monotone_core` | 473.389 | 451.472 | +5.127 |
| `lightgbm_monotone_extended` | 473.746 | 452.284 | +5.484 |
| `lightgbm_linear_tree` | 494.746 | 449.977 | +26.484 |

Every candidate is worse than the control on the primary window. Unregularized
piecewise-linear leaves are worse by 26.5 RMSE and stop early at 101 iterations,
which is the signature of added variance rather than added signal. Constraining
the six core signs costs 5.1 RMSE, and extending the constraint set to amenity
distances costs a further 0.4, consistent with those signs being confounded by
region rather than genuinely monotone.

`lightgbm_linear_tree_reg` is the single exception: it beats the control by 0.419
on the 2024-10 window while losing by 2.487 on 2024-05. A result that reverses
sign between windows is treated here as noise, not signal, following the
selection rule used for the rolling LightGBM comparison.

## D. Contribution to the ensemble

| Candidate | Window | Selected weight | Blend RMSE | Gain over frozen v2 |
| --- | --- | ---: | ---: | ---: |
| `lightgbm_linear_tree` | 2024-05 | 0.0066 | 463.1535 | 0.0014 |
| `lightgbm_linear_tree_reg` | 2024-10 | 0.0065 | 443.3034 | 0.0002 |
| all other candidates | both | 0.0000 | unchanged | 0.0000 |

The bounded least-squares blend assigns essentially zero weight to every
candidate on both windows. The largest gain is 0.0014 RMSE, about 180 times
smaller than the 0.249 improvement recorded for the v3 square-metre blend, and far
below the 0.5 RMSE threshold used to interpret differences (see
[`problem_and_protocol.md`](problem_and_protocol.md) Section F). None of these
candidates is worth adopting.

## E. Error by forecast horizon

The horizon table tests the idea directly. If limited extrapolation were the
binding constraint, a candidate able to extrapolate would start behind the
control at short horizons and close the gap as the horizon grows.

| Horizon | Month | Control RMSE | `linear_tree_reg` gap | `monotone_core` gap |
| ---: | --- | ---: | ---: | ---: |
| 1 | 2024-05 | 502.1 | +6.3 | +8.2 |
| 2 | 2024-06 | 503.6 | −0.1 | +0.8 |
| 3 | 2024-07 | 492.2 | +0.3 | +1.3 |
| 4 | 2024-08 | 462.0 | +2.8 | +7.5 |
| 5 | 2024-09 | 476.3 | +4.7 | +6.7 |
| 6 | 2024-10 | 438.3 | +0.7 | +2.2 |
| 7 | 2024-11 | 462.5 | +2.0 | +3.9 |
| 8 | 2024-12 | 442.6 | +2.5 | +6.2 |
| 9 | 2025-01 | 466.7 | +4.2 | +8.5 |
| 10 | 2025-02 | 447.8 | +3.1 | +7.2 |
| 11 | 2025-03 | 444.4 | +1.1 | +4.3 |

The gap does not close. Its correlation with horizon is −0.077 for the
piecewise-linear model and **+0.238** for the monotone model, so the constrained
model falls further behind as the horizon lengthens. Averaged over the first and
last three months the piecewise-linear gap moves only from +2.15 to +2.79, and
the monotone gap widens from +3.46 to +6.67.

It is tempting to attribute this to the 24-month lag, which returns the mean rent
recorded two years earlier and so gives a test row in 2026-07 an observed value from
2024-07. The moving-market backtests show that this is not the mechanism: removing
the lag changes RMSE by at most 2.6 at any of three forecast origins, including a
rising market (finding 3 of the [moving-market backtests](README.md#moving-market-backtests)). What the horizon
analysis does establish is narrower: in these windows, the candidates that can
extrapolate do not recover ground as the horizon grows.

## F. Recalibration: an in-sample gain that does not transfer

The diagnostics exposed a signed bias in the frozen v2 blend, which looked at
first like a cheap win.

| Cutoff | Rows | RMSE | Bias (pred − actual) | Affine slope | SD(pred) | SD(actual) | RMSE after level shift | RMSE after affine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 32,012 | 463.155 | **+17.48** | 1.0092 | 462.1 | 657.0 | 462.825 | **462.805** |
| 2024-10 | 17,198 | 443.304 | +0.68 | 0.9840 | 469.5 | 640.2 | 443.303 | 443.239 |

On the primary window the blend overpredicts by 17.5 SGD per month and is
positively biased in 10 of 11 validation months, and removing that bias is worth
0.330 RMSE while a full affine fit is worth 0.350. Both exceed the 0.249 gain
that v3 delivered. The prediction spread is also about 30 per cent narrower than
the actual spread, the usual under-dispersion of an averaged tree ensemble, which
is why the fitted slope exceeds one.

The correction does not survive validation:

| Test | Result |
| --- | --- |
| Fit on 2024-05, apply to 2024-10 | affine **+0.415**, level shift **+0.318** (both worse) |
| Fit on 2024-10, apply to 2024-05 | affine +0.096 worse, level shift 0.025 better |
| Fit on early months of 2024-05, apply to its later months | affine 0.297 better, level shift 0.334 better |
| Fit on early months of 2024-10, apply to its later months | affine **0.612 worse**, level shift **0.361 worse** |
| Best spread-only rescaling, fitted per window | 0.019 and 0.063 better |

The bias is +17.48 at one cutoff and +0.68 at the other, and the fitted slope
moves from above one to below one. The forward-transfer test passes on the window
whose bias is large and fails on the window whose bias is near zero. So the
apparent miscalibration is a property of a particular fitting cutoff, not a
systematic defect of the model, and a correction estimated on one period cannot
be trusted on another. **No recalibration is applied to any submission.** Had
only the primary window been examined, this would have looked like a 0.35 RMSE
improvement and would have been adopted.

## G. Conclusions

1. Piecewise-linear leaves and monotone constraints both **reduce** accuracy on
   this feature set, by 2.5 to 26.5 RMSE standalone, and contribute at most
   0.0014 RMSE to the ensemble. Neither is recommended.
2. The error gap does not narrow with forecast horizon, so in these windows
   limited extrapolation is **not** the binding constraint. The scope of these
   windows is stated in [`results_summary.md`](results_summary.md) Section C.
3. The v2 blend is under-dispersed and, at the 2024-05 cutoff, biased upward by
   17.5 SGD. Correcting either is worth more in sample than the v3 blend gain,
   but neither correction transfers across cutoffs, so neither is usable.
4. The remaining headroom is not in the leaf model or in post-hoc calibration. On
   validation rows that have an exact twin, the v2 RMSE (452.8 in W1, 436.1 in W2)
   lies inside the interval for the estimated noise level (442.2 to 466.8 and
   418.3 to 451.5), so the data are consistent with little room left on those
   rows. The intervals are wide, and the analysis covers about a quarter of the
   rows (see [`problem_and_protocol.md`](problem_and_protocol.md) Section E).

## H. Reproduction

```bash
python3 -m unittest discover -s tests -p 'test_punyaa_*.py' -v

python3 -m punyaa.extrapolation --validation-start 2024-05
python3 -m punyaa.extrapolation --validation-start 2024-10
python3 -m punyaa.extrapolation --calibration-only --validation-start 2024-05
python3 -m punyaa.extrapolation --cross-window
```

The `--cross-window` command writes
`outputs/punyaa_extrapolation/cross_window_calibration.json`, which holds the
cross-window and spread-rescaling rows of the Section F table. Each other run writes
`metrics.json`, `horizon_diagnostics.csv`, `blend_table.csv`, and
the per-candidate prediction vectors to
`outputs/punyaa_extrapolation/<validation-start>/`. The commands reuse the stored
v2 component predictions in `outputs/punyaa_improvements/` and do not refit
CatBoost.

A latent defect was fixed in passing: `_nearest` in
[`punyaa/features.py`](../../punyaa/features.py) multiplied a NumPy view
returned by `to_numpy()` in place, which raises `ValueError: output array is
read-only` on pandas 3.x. The fix requests a copy and changes no feature values.
