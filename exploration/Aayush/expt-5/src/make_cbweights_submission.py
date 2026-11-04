"""
expt-5: V2 with ONLY the blend weights changed.

Uses V2's exactly reproduced per-model test predictions (v2_reproduce.py) and the weights chosen
on the two time holdouts (blend_weights_evidence.py: argmin of mean RMSE over the simplex, step
0.05). Nothing else about V2 changes, so a leaderboard difference isolates the weights.
Output: exploration/Aayush/expt-5/outputs/submission_v2_cbweights.csv
"""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'expt-5', 'outputs')
w = json.load(open(os.path.join(OUT, 'blend_weights_evidence.json')))['chosen_weights']
p = pd.read_csv(os.path.join(OUT, 'v2_test_preds_by_model.csv'))
blend = w[0] * p['lgb'] + w[1] * p['xgb'] + w[2] * p['cb']
v2 = pd.read_csv(os.path.join(ROOT, 'expt-2', 'outputs', 'submissions', 'submission_ensemble_v2.csv'))
assert len(blend) == len(v2) == 50000 and blend.notna().all()
out = pd.DataFrame({'Id': v2['Id'].values, 'Predicted': blend.values})
out.to_csv(os.path.join(OUT, 'submission_v2_cbweights.csv'), index=False)
d = out['Predicted'].values - v2['Predicted'].values
print(f'weights (lgb, xgb, cb): {w}')
print(f'vs V2: corr {np.corrcoef(out["Predicted"], v2["Predicted"])[0, 1]:.5f}, mean abs diff {np.abs(d).mean():.2f}, '
      f'mean diff {d.mean():+.2f}, max abs diff {np.abs(d).max():.1f}')
print(f'predictions: mean {out["Predicted"].mean():.1f}, std {out["Predicted"].std():.1f}, min {out["Predicted"].min():.0f}, max {out["Predicted"].max():.0f}')
