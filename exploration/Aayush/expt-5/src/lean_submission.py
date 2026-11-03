"""
expt-5: the final pipeline (modules/pipeline.py) with the nine COE and stock columns removed, as a candidate only.

Same cleaning, features, honest anchor (frozen at 2025-Q1 plus drift), blend-weight fit and 5-fold bagging as main.py;
the only change is dropping coe_*, stock_* and sg_financial_index from train and test before modelling.
Writes exploration/Aayush/expt-5/outputs/submission_v2_honest_lean.csv and compares it with the scored file
(submission_v2_honest.csv, public 433.65480). Never touches the repository's submission.csv.
Usage (repo root): python exploration/Aayush/expt-5/src/lean_submission.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, 'modules'))
import numpy as np, pandas as pd
import pipeline as pipe, models

OUT = os.path.join(os.path.dirname(HERE), 'outputs')
train, test_pub = pipe.build_features()
test = pipe.apply_honest_anchor(test_pub)
macro = [c for c in train.columns if c.startswith(('coe_', 'stock_')) or c == 'sg_financial_index']
assert len(macro) == 9, macro
train, test = train.drop(columns=macro), test.drop(columns=macro)
print('features:', len(models.get_feature_columns(train)[0]), '(63 in the scored model)', flush=True)
weights, oot = models.fit_blend_weights(train)
print('blend weights:', np.round(weights, 4), '| out-of-time RMSE', {k: round(v, 2) for k, v in oot.items()}, flush=True)
out = models.train_five_fold(train, test, weights)
print('5-fold OOF RMSE:', {k: round(v, 2) for k, v in out['oof_rmse'].items()}, flush=True)
sub = pipe.write_submission(out['blend'], os.path.join(OUT, 'submission_v2_honest_lean.csv'))
ref = pd.read_csv(os.path.join(OUT, 'submission_v2_honest.csv'))['Predicted'].values
d = sub['Predicted'].values - ref
print(f'vs scored file: corr {np.corrcoef(sub["Predicted"], ref)[0, 1]:.5f}, mean diff {d.mean():+.2f}, '
      f'mean abs diff {np.abs(d).mean():.2f}, max abs diff {np.abs(d).max():.1f}')
