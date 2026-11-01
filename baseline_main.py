"""
BASELINE (superseded): the earlier single-LightGBM pipeline, kept for the progress report. The final model is
main.py; this file and baseline_baseline_main.ipynb are not needed to reproduce it. Run from the repo root.

Lean production pipeline: preprocessing -> validation split -> train the current best
model -> verify against the held-out fold -> refit on all of train.csv -> predict
test.csv -> write submission.csv.

Run this for routine verification (e.g. confirming submission.csv still reproduces
after a archived_archived_modules/*.py change) instead of re-executing baseline_main.ipynb. baseline_main.ipynb stays the
full exploratory record — every model comparison, the encoding/ensembling/momentum/
data-quality investigations, all with reasoning — but it fits dozens of models across
50+ cells and takes 1-2 minutes to re-run. This script fits exactly two: the current
best model on the validation split (to verify), and again on the full data (to ship).

The model/hyperparameters below are a snapshot, not re-derived here — they were
selected in baseline_main.ipynb's model-comparison and hyperparameter-tuning sections (see
exploration/Aayush/progress-report.md "Current result"). If the feature set in archived_modules/modeling.py
changes enough that a different model or hyperparameters might now win, re-run that
comparison in baseline_main.ipynb and update BEST_MODEL/BEST_PARAMS below to match — this
script intentionally does not re-search for "best", only reproduces a known one.

Usage: python baseline_main.py
"""
import sys

sys.path.insert(0, 'archived_modules')

import pandas as pd
import preprocessing as prep
import split as spl
import modeling as mdl

# Established in baseline_main.ipynb (expanding-window CV, confirmed against the true
# held-out val_fold by RMSE — the metric that actually matters here, not MAE).
# Superseded by a wider randomized hyperparameter search (30 trials over
# n_estimators/learning_rate/num_leaves/min_child_samples/subsample/colsample_bytree/
# reg_alpha/reg_lambda, see exploration/Aayush/progress-report.md "Hyperparameter search, round 2") —
# confirmed on the true held-out val_fold: RMSE 457.37 -> 456.70. A SUBZONE x
# FLAT_TYPE interaction feature gave a near-identical standalone gain (456.68) but the
# two do NOT stack (456.88 combined, worse than either alone — the hyperparameters
# below were tuned for the feature set without that interaction, and combining both
# needs its own re-tuning pass, not yet done) — shipping hyperparameters only.
BEST_MODEL = 'LightGBM'
BEST_PARAMS = {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 15,
                'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.6,
                'reg_alpha': 0.1, 'reg_lambda': 0.1}
BEST_FIT_BASELINE = mdl.fit_lgbm_baseline
BEST_FIT_FULL = mdl.fit_lgbm_full

# Address-level (TOWN+BLOCK+STREET — finer than SUBZONE) trailing rent-per-sqm
# feature, shrunk toward the SUBZONE mean (archived_modules/preprocessing.py::
# build_address_momentum/add_address_feature). Tested and ADOPTED — see
# exploration/Aayush/progress-report.md "Address-level rent-level feature": RMSE 456.70 -> 452.56 on
# val_fold, held up across 3/4 expanding-window CV folds. Note this hyperparameter
# set was tuned WITHOUT this feature present — the SUBZONE x FLAT_TYPE interaction
# episode (exploration/Aayush/progress-report.md "Joint re-tune") showed a new feature and existing
# tuned hyperparameters don't always stack additively, so a joint re-tune is a
# worthwhile follow-up, not yet done (the gain here already clears the bar on its
# own with the untouched hyperparameters).
# Median-rent anchor (archived_modules/preprocessing.py::build_median_anchor_table/
# add_median_rent_anchor): the official quarterly town x flat-type median rent from
# sg-hdb-median-rent.csv, with a fallback chain for the ~3% of rows whose cell is
# empty (town names must be stripped when matching: the file has 'QUEENSTOWN ' with a
# trailing space). The model predicts the RESIDUAL
# (rent - anchor) and anchor is added back, so trees no longer have to extrapolate the
# market level. Tested on a 16-month horizon-matched holdout (train through 2023-11):
# RMSE 600.72 -> 492.77 (paired-bootstrap CI -111 to -104), and 452.56 -> 451.87 on the
# usual 6-month fold (not significant there: CI -1.98 to +0.52). See
# exploration/median_anchor_experiment.py and exploration/Aayush/progress-report.md "Median-rent anchor".
# Hyperparameters were tuned for the direct target and are unchanged here.
NUMERIC_FEATURES = mdl.GBM_NUMERIC_FEATURES + ['addr_level_feat', 'anchor', 'anchor_level']

# Reference numbers for the eyeball comparison printed below — not asserted against,
# since minor float noise across environments is expected; this is a sanity check for
# a human to glance at, not a test suite. History: {'MAE': 334.44, 'RMSE': 457.47} pre
# -1-sentinel-fix -> {'MAE': 334.31, 'RMSE': 457.37} post-fix -> {'MAE': 333.42,
# 'RMSE': 456.70} post wider hyperparameter search -> current, post address-level
# feature (452.56 / 329.80) -> current, with the median-rent anchor and residual
# target (see NUMERIC_FEATURES comment above).
EXPECTED_VAL_METRICS = {'MAE': 326.06, 'RMSE': 451.87}


def add_anchor(source_df, *frames, town_table, cross_table):
    """Fit the anchor's town ratios and global fallback on source_df's rents only, then
    add anchor/anchor_level to source_df and every frame in frames (same leakage rule
    as the address feature: train_fold when validating, train_feat for the final fit)."""
    ratios = prep.fit_anchor_town_ratios(source_df, cross_table)
    fallback = source_df['MONTHLY_RENT'].mean()
    return [prep.add_median_rent_anchor(f, town_table, cross_table, ratios, fallback)
            for f in (source_df, *frames)]


def residual(df):
    """Copy of df whose MONTHLY_RENT is rent minus anchor. Errors on this target equal
    errors on rent once the anchor is added back, so metrics computed against it are
    the real RMSE/MAE."""
    return df.assign(MONTHLY_RENT=df['MONTHLY_RENT'] - df['anchor'])


def main(data_dir: str = 'data', submission_path: str = 'submission_baseline.csv'):
    # 1. Preprocessing (archived_modules/preprocessing.py) — cleaning, joins, macro/geo features.
    train_feat, test_feat = prep.load_and_build(data_dir=data_dir)
    print(f'train_feat: {train_feat.shape}  test_feat: {test_feat.shape}')

    # 2. Time-based validation split (archived_modules/split.py) — NOT random, see split.py's
    # docstring. Used here only to verify performance; the shipped model below is
    # refit on all of train_feat afterward.
    train_fold, val_fold, cutoff = spl.time_based_split(train_feat)
    categories = mdl.get_categories(train_feat)

    # 2b. Address-level trailing rent feature (archived_modules/preprocessing.py) — built from
    # train_fold ONLY here (never val_fold), so val_fold's own rows can never leak
    # into the feature applied to it. See NUMERIC_FEATURES comment above.
    momentum_table = prep.build_address_momentum(train_fold)
    subzone_mean, global_mean = prep.rent_per_sqm_priors(train_fold)
    train_fold = prep.add_address_feature(train_fold, momentum_table, subzone_mean, global_mean)
    val_fold = prep.add_address_feature(val_fold, momentum_table, subzone_mean, global_mean)

    # 2c. Median-rent anchor, ratios fit on train_fold only.
    median_df = pd.read_csv(f'{data_dir}/auxiliary/sg-hdb-median-rent.csv')
    town_table, cross_table = prep.build_median_anchor_table(median_df)
    train_fold, val_fold = add_anchor(train_fold, val_fold, town_table=town_table, cross_table=cross_table)

    # 3. Train the current best model on train_fold, verify against the true held-out
    # val_fold — the one honest check, same discipline as baseline_main.ipynb throughout.
    _, val_metrics, _ = BEST_FIT_BASELINE(residual(train_fold), residual(val_fold), categories,
                                           numeric_features=NUMERIC_FEATURES, **BEST_PARAMS)
    print(f'Validation check ({BEST_MODEL}, train_fold -> val_fold): {val_metrics}')
    print(f'Reference from baseline_main.ipynb (compare, do not assume): {EXPECTED_VAL_METRICS}')

    # 4. Refit on ALL of train.csv, predict test.csv, write the submission file.
    # Rebuilt from train_feat (not train_fold) so the final model's feature uses
    # every known transaction, not just the pre-cutoff slice used for validation.
    momentum_table_full = prep.build_address_momentum(train_feat)
    subzone_mean_full, global_mean_full = prep.rent_per_sqm_priors(train_feat)
    train_feat = prep.add_address_feature(train_feat, momentum_table_full, subzone_mean_full, global_mean_full)
    test_feat = prep.add_address_feature(test_feat, momentum_table_full, subzone_mean_full, global_mean_full)

    train_feat, test_feat = add_anchor(train_feat, test_feat, town_table=town_table, cross_table=cross_table)
    print('test anchor level share:', test_feat['anchor_level'].value_counts(normalize=True).sort_index().round(3).to_dict())

    final_model = BEST_FIT_FULL(residual(train_feat), categories, numeric_features=NUMERIC_FEATURES, **BEST_PARAMS)
    test_preds = mdl.predict_test(final_model, test_feat, is_gbm=True, categories=categories,
                                   numeric_features=NUMERIC_FEATURES) + test_feat['anchor']
    submission = pd.DataFrame({'Id': test_feat.index, 'Predicted': test_preds.values})
    submission.to_csv(submission_path, index=False)
    print(f'{submission_path} written: {len(submission)} rows')

    return val_metrics, submission


if __name__ == '__main__':
    main()
