"""
expt-5: reproduce expt-2 V2's 5-fold training EXACTLY as run, saving each model's test and
out-of-fold predictions so blend weights can be changed without retraining.

Mirrors exploration/Aayush/expt-2/src/models_v2.py::train_and_predict_5fold_v2 (same features parquet, folds,
seeds, hyperparameters, early stopping on each fold's validation split). Unlike that function it
does NOT write the root submission.csv or anything under expt-2; it writes only to exploration/Aayush/expt-5/outputs.
Verification: rebuilds the blend with V2's weights and compares with V2's own submission file.
Usage (repo root): python exploration/Aayush/expt-5/src/v2_reproduce.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'expt-2', 'src'))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import config, models_v2

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'v2_reproduce.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

FEAT = os.path.join(ROOT, 'expt-2', 'outputs', 'features')
train = pd.read_parquet(os.path.join(FEAT, 'train_features_v2.parquet'))
test = pd.read_parquet(os.path.join(FEAT, 'test_features_v2.parquet'))
feat_cols, cat_cols = models_v2.get_feature_columns(train)
y_actual = train['MONTHLY_RENT'].values
y_bench = train['median_rent_benchmark'].values
y_res = y_actual - y_bench
test_bench = test['median_rent_benchmark'].values

df_lgb, df_te_lgb = train.copy(), test.copy()
df_cb, df_te_cb = train.copy(), test.copy()
for c in cat_cols:
    df_lgb[c], df_te_lgb[c] = df_lgb[c].astype('category'), df_te_lgb[c].astype('category')
    df_cb[c], df_te_cb[c] = df_cb[c].astype(str), df_te_cb[c].astype(str)

oof = {k: np.zeros(len(train)) for k in ['lgb', 'xgb', 'cb']}
tst = {k: np.zeros(len(test)) for k in ['lgb', 'xgb', 'cb']}
kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
    t0 = time.time()
    X_tr, y_tr = df_lgb.iloc[tr_idx][feat_cols], y_res[tr_idx]
    X_va, y_va = df_lgb.iloc[va_idx][feat_cols], y_res[va_idx]
    va_bench = y_bench[va_idx]
    m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                          random_state=config.RANDOM_SEED + fold, n_jobs=4)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof['lgb'][va_idx] = va_bench + m.predict(X_va)
    tst['lgb'] += (test_bench + m.predict(df_te_lgb[feat_cols])) / config.N_FOLDS
    xm = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.75,
                          enable_categorical=True, random_state=config.RANDOM_SEED + fold, n_jobs=4, early_stopping_rounds=40)
    xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof['xgb'][va_idx] = va_bench + xm.predict(X_va)
    tst['xgb'] += (test_bench + xm.predict(df_te_lgb[feat_cols])) / config.N_FOLDS
    cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=cat_cols,
                           random_seed=config.RANDOM_SEED + fold, thread_count=4, verbose=False, early_stopping_rounds=40)
    cm.fit(df_cb.iloc[tr_idx][feat_cols], y_tr, eval_set=(df_cb.iloc[va_idx][feat_cols], y_va))
    oof['cb'][va_idx] = va_bench + cm.predict(df_cb.iloc[va_idx][feat_cols])
    tst['cb'] += (test_bench + cm.predict(df_te_cb[feat_cols])) / config.N_FOLDS
    say(f'fold {fold}: lgb {root_mean_squared_error(y_actual[va_idx], oof["lgb"][va_idx]):.2f}  '
        f'xgb {root_mean_squared_error(y_actual[va_idx], oof["xgb"][va_idx]):.2f}  '
        f'cb {root_mean_squared_error(y_actual[va_idx], oof["cb"][va_idx]):.2f}  ({time.time()-t0:.0f}s)')

pd.DataFrame(oof).assign(y=y_actual).to_csv(os.path.join(OUT, 'v2_oof_preds.csv'), index=False)
pd.DataFrame(tst).to_csv(os.path.join(OUT, 'v2_test_preds_by_model.csv'), index=False)
w = json.load(open(os.path.join(ROOT, 'expt-2', 'outputs', 'models', 'cv_metrics_v2.json')))['weights']
blend = w[0] * tst['lgb'] + w[1] * tst['xgb'] + w[2] * tst['cb']
ref = pd.read_csv(os.path.join(ROOT, 'expt-2', 'outputs', 'submissions', 'submission_ensemble_v2.csv'))['Predicted'].values
say(f'\nreproduction check vs V2 submission file: corr {np.corrcoef(blend, ref)[0, 1]:.6f}, '
    f'mean abs diff {np.abs(blend - ref).mean():.3f}, max abs diff {np.abs(blend - ref).max():.2f}, mean diff {(blend - ref).mean():+.3f}')
say('overall OOF RMSE with V2 weights:', round(root_mean_squared_error(y_actual, w[0] * oof['lgb'] + w[1] * oof['xgb'] + w[2] * oof['cb']), 2),
    '(V2 reported 470.99)')
say('done')
