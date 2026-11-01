"""
Final model, end to end: raw CSVs in data/ -> cleaning -> V2 features -> honest test-quarter anchor ->
LightGBM/XGBoost/CatBoost residual blend -> submission.csv.

Reproduces the submitted file (public leaderboard 433.65480): the expt-2 V2 recipe with one change, the test
rows' median-rent benchmark is frozen at the last labelled quarter (2025-Q1) and drifted, instead of read from
the published medians that already contain test-period rents (modules/anchor.py).

All logic lives in modules/; main.ipynb walks through the same stages with commentary.
The previous pipeline (single LightGBM, validation RMSE 451.87) is kept as baseline_main.py / baseline_main.ipynb.

Usage:   python main.py [--out submission.csv] [--compare reference.csv]
Runtime: about 15-25 minutes on 8 cores (three models x 5 folds, plus one out-of-time fit for the blend weights).
Seeds are fixed (42, fold k uses 42+k) and every model uses 4 threads, as in the submitted run; across machines
or library versions expect predictions to agree to within a few SGD, not to the last digit.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'modules'))

import numpy as np
import pandas as pd
import pipeline as pipe


def compare(sub: pd.DataFrame, reference_path: str):
    """Agreement with a reference submission (e.g. the file that scored 433.65480)."""
    ref = pd.read_csv(reference_path)
    assert len(ref) == len(sub) and (ref['Id'].values == sub['Id'].values).all(), 'Id mismatch'
    d = sub['Predicted'].values - ref['Predicted'].values
    print(f'vs {reference_path}: corr {np.corrcoef(sub["Predicted"], ref["Predicted"])[0, 1]:.6f}, '
          f'mean abs diff {np.abs(d).mean():.3f}, max abs diff {np.abs(d).max():.3f}')


def main(submission_path: str = os.path.join(ROOT, 'submission.csv'), compare_to: str = None):
    sub, info = pipe.run(submission_path)
    print('blend weights (lgb, xgb, cb):', np.round(info['weights'], 4))
    print('out-of-time RMSE:', {k: round(v, 2) for k, v in info['oot_rmse'].items()})
    print('5-fold OOF RMSE:', {k: round(v, 2) for k, v in info['oof_rmse'].items()})
    print(f'{submission_path} written: {len(sub)} rows')
    if compare_to:
        compare(sub, compare_to)
    return sub, info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'submission.csv'))
    ap.add_argument('--compare', default=None, help='reference submission csv to compare against')
    a = ap.parse_args()
    main(a.out, a.compare)
