"""
expt-5: expt-2 V2 with ONE change, an honest test-quarter anchor.

Identical to src/v2_reproduce.py (V2's 5-fold LightGBM/XGBoost/CatBoost on train_features_v2.parquet, same seeds,
hyperparameters and V2's blend weights) except that the test rows' median_rent_benchmark, nat_median_rent_benchmark and
median_rent_ratio are replaced by the frozen-and-drifted benchmark of honest_lib (cutoff 2025-Q1, the last
labelled quarter; drift +0.55%/quarter). Train features are untouched (their quarters are all <= 2025-Q1).
Output: exploration/Aayush/expt-5/outputs/submission_v2_honest.csv (+ per-model predictions). Does not touch expt-2 or the root submission.
Usage (repo root): python exploration/Aayush/expt-5/src/honest_v2_submission.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'expt-2', 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import config, models_v2, honest_lib as hl

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'honest_v2_submission.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

FEAT = os.path.join(ROOT, 'expt-2', 'outputs', 'features')
train = pd.read_parquet(os.path.join(FEAT, 'train_features_v2.parquet'))
test_pub = pd.read_parquet(os.path.join(FEAT, 'test_features_v2.parquet'))
assert train['approval_quarter'].max() == '2025-Q1'
test = hl.make_variant(test_pub, '2025-Q1', 'drift4')
changed = (test['median_rent_benchmark'] != test_pub['median_rent_benchmark'])
say(f'test rows with a changed benchmark: {changed.sum():,} of {len(test):,}; mean benchmark published {test_pub["median_rent_benchmark"].mean():.1f} -> honest {test["median_rent_benchmark"].mean():.1f}')
for q, g in test.groupby('approval_quarter'):
    say(f'  {q}: n={len(g):5d}  mean benchmark published {test_pub.loc[g.index, "median_rent_benchmark"].mean():7.1f}  honest {g["median_rent_benchmark"].mean():7.1f}')

feat_cols, cat_cols = models_v2.get_feature_columns(train)
y_res = train['MONTHLY_RENT'].values - train['median_rent_benchmark'].values
y_bench = train['median_rent_benchmark'].values
test_bench = test['median_rent_benchmark'].values
df_lgb, df_te_lgb, df_cb, df_te_cb = train.copy(), test.copy(), train.copy(), test.copy()
for c in cat_cols:
    df_lgb[c], df_te_lgb[c] = df_lgb[c].astype('category'), df_te_lgb[c].astype('category')
    df_cb[c], df_te_cb[c] = df_cb[c].astype(str), df_te_cb[c].astype(str)
tst = {k: np.zeros(len(test)) for k in ['lgb', 'xgb', 'cb']}
kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
    t0 = time.time()
    X_tr, y_tr = df_lgb.iloc[tr_idx][feat_cols], y_res[tr_idx]
    X_va, y_va = df_lgb.iloc[va_idx][feat_cols], y_res[va_idx]
    m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                          random_state=config.RANDOM_SEED + fold, n_jobs=4)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
    tst['lgb'] += (test_bench + m.predict(df_te_lgb[feat_cols])) / config.N_FOLDS
    xm = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.75,
                          enable_categorical=True, random_state=config.RANDOM_SEED + fold, n_jobs=4, early_stopping_rounds=40)
    xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    tst['xgb'] += (test_bench + xm.predict(df_te_lgb[feat_cols])) / config.N_FOLDS
    cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=cat_cols,
                           random_seed=config.RANDOM_SEED + fold, thread_count=4, verbose=False, early_stopping_rounds=40)
    cm.fit(df_cb.iloc[tr_idx][feat_cols], y_tr, eval_set=(df_cb.iloc[va_idx][feat_cols], y_va))
    tst['cb'] += (test_bench + cm.predict(df_te_cb[feat_cols])) / config.N_FOLDS
    say(f'fold {fold} done ({time.time() - t0:.0f}s)')
pd.DataFrame(tst).to_csv(os.path.join(OUT, 'honest_v2_test_preds_by_model.csv'), index=False)
w = json.load(open(os.path.join(ROOT, 'expt-2', 'outputs', 'models', 'cv_metrics_v2.json')))['weights']
blend = w[0] * tst['lgb'] + w[1] * tst['xgb'] + w[2] * tst['cb']
v2 = pd.read_csv(os.path.join(ROOT, 'expt-2', 'outputs', 'submissions', 'submission_ensemble_v2.csv'))
assert len(blend) == len(v2) == 50000 and np.isfinite(blend).all()
out = pd.DataFrame({'Id': v2['Id'].values, 'Predicted': blend})
out.to_csv(os.path.join(OUT, 'submission_v2_honest.csv'), index=False)
d = blend - v2['Predicted'].values
say(f'\nhonest vs V2 predictions: corr {np.corrcoef(blend, v2["Predicted"])[0, 1]:.5f}, mean diff {d.mean():+.1f}, mean abs diff {np.abs(d).mean():.1f}, max abs diff {np.abs(d).max():.0f}')
for q in sorted(test['approval_quarter'].unique()):
    mk = (test['approval_quarter'] == q).values
    say(f'  {q}: mean diff {d[mk].mean():+7.1f}  (n={mk.sum()})')
say('written', os.path.join(OUT, 'submission_v2_honest.csv'))
