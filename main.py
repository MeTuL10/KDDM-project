"""
Lean production pipeline: preprocessing -> validation split -> train the current best
model -> verify against the held-out fold -> refit on all of train.csv -> predict
test.csv -> write submission.csv.

Run this for routine verification (e.g. confirming submission.csv still reproduces
after a modules/*.py change) instead of re-executing main.ipynb. main.ipynb stays the
full exploratory record — every model comparison, the encoding/ensembling/momentum/
data-quality investigations, all with reasoning — but it fits dozens of models across
50+ cells and takes 1-2 minutes to re-run. This script fits exactly two: the current
best model on the validation split (to verify), and again on the full data (to ship).

The model/hyperparameters below are a snapshot, not re-derived here — they were
selected in main.ipynb's model-comparison and hyperparameter-tuning sections (see
progress-report.md "Current result"). If the feature set in modules/modeling.py
changes enough that a different model or hyperparameters might now win, re-run that
comparison in main.ipynb and update BEST_MODEL/BEST_PARAMS below to match — this
script intentionally does not re-search for "best", only reproduces a known one.

Usage: python main.py
"""
import sys

sys.path.insert(0, 'modules')

import pandas as pd
import preprocessing as prep
import split as spl
import modeling as mdl

# Established in main.ipynb (expanding-window CV, confirmed against the true
# held-out val_fold by RMSE — the metric that actually matters here, not MAE).
# Superseded by a wider randomized hyperparameter search (30 trials over
# n_estimators/learning_rate/num_leaves/min_child_samples/subsample/colsample_bytree/
# reg_alpha/reg_lambda, see progress-report.md "Hyperparameter search, round 2") —
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

# Reference numbers for the eyeball comparison printed below — not asserted against,
# since minor float noise across environments is expected; this is a sanity check for
# a human to glance at, not a test suite. History: {'MAE': 334.44, 'RMSE': 457.47} pre
# -1-sentinel-fix -> {'MAE': 334.31, 'RMSE': 457.37} post-fix -> current, post wider
# hyperparameter search (see BEST_PARAMS comment above).
EXPECTED_VAL_METRICS = {'MAE': 333.42, 'RMSE': 456.70}


def main(data_dir: str = 'data', submission_path: str = 'submission.csv'):
    # 1. Preprocessing (modules/preprocessing.py) — cleaning, joins, macro/geo features.
    train_feat, test_feat = prep.load_and_build(data_dir=data_dir)
    print(f'train_feat: {train_feat.shape}  test_feat: {test_feat.shape}')

    # 2. Time-based validation split (modules/split.py) — NOT random, see split.py's
    # docstring. Used here only to verify performance; the shipped model below is
    # refit on all of train_feat afterward.
    train_fold, val_fold, cutoff = spl.time_based_split(train_feat)
    categories = mdl.get_categories(train_feat)

    # 3. Train the current best model on train_fold, verify against the true held-out
    # val_fold — the one honest check, same discipline as main.ipynb throughout.
    _, val_metrics, _ = BEST_FIT_BASELINE(train_fold, val_fold, categories, **BEST_PARAMS)
    print(f'Validation check ({BEST_MODEL}, train_fold -> val_fold): {val_metrics}')
    print(f'Reference from main.ipynb (compare, do not assume): {EXPECTED_VAL_METRICS}')

    # 4. Refit on ALL of train.csv, predict test.csv, write the submission file.
    final_model = BEST_FIT_FULL(train_feat, categories, **BEST_PARAMS)
    test_preds = mdl.predict_test(final_model, test_feat, is_gbm=True, categories=categories)
    submission = pd.DataFrame({'Id': test_feat.index, 'Predicted': test_preds.values})
    submission.to_csv(submission_path, index=False)
    print(f'{submission_path} written: {len(submission)} rows')

    return val_metrics, submission


if __name__ == '__main__':
    main()
