"""
expt-5 experiment 2: bigger levers on the clean V2 recipe.

  Part A  smoothed anchor (as features / as the residual base) on top of 'both'
  Part B  LightGBM regularisation sweep on the best feature set, trees by early stopping
          per holdout (optimistic but equal for every config)
  Part C  seed bagging and an XGBoost + CatBoost blend on the best config

Usage (repo root): python exploration/Aayush/expt-5/src/v2_improvements.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
import xgboost as xgb
from catboost import CatBoostRegressor

log = open(os.path.join(ROOT, 'expt-5', 'outputs', 'v2_improvements.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

HOLD = [('H1', '2024-07'), ('H2', '2023-12')]

def frames(cutoff, smooth=None):
    tr, va = build(cutoff, anchor=True, trailing=True)
    if smooth:
        tr, va = add_smooth(tr, va, as_target=(smooth == 'target'))
    return tr, va

# ---------------- Part A: smoothed anchor ----------------
say('=== Part A: smoothed anchor (all on top of anchor+trailing, V2 params, early stopping on val) ===')
best_iter = {}
for h, cutoff in HOLD:
    res = {}
    for name, sm in [('both', None), ('smooth feature', 'feature'), ('smooth target', 'target')]:
        tr, va = frames(cutoff, sm)
        p, m = fit_predict(tr, va, V2_PARAMS, 1200, early_stop=True)
        res[name] = (ev.rmse(va['MONTHLY_RENT'], p), int(m.best_iteration_))
        say(f'  {h} {name:15s} RMSE {res[name][0]:7.2f}  best_iter {res[name][1]}')

# ---------------- Part B: regularisation sweep ----------------
say('\n=== Part B: LightGBM regularisation sweep (anchor+trailing features, early stopping on val) ===')
configs = {
  'v2':        V2_PARAMS,
  'reg':       REG_PARAMS,
  'l31_mc100': dict(REG_PARAMS, num_leaves=31, min_child_samples=100),
  'l15_mc100': dict(REG_PARAMS, num_leaves=15, min_child_samples=100),
  'l31_lam20': dict(REG_PARAMS, num_leaves=31, min_child_samples=100, reg_lambda=20.0),
  'l31_lr02':  dict(REG_PARAMS, num_leaves=31, min_child_samples=100, learning_rate=0.02),
  'l31_cs5':   dict(REG_PARAMS, num_leaves=31, min_child_samples=100, colsample_bytree=0.5),
  'l63_mc200': dict(REG_PARAMS, num_leaves=63, min_child_samples=200, reg_lambda=10.0),
}
sweep = {}
for h, cutoff in HOLD:
    tr, va = frames(cutoff)
    for name, params in configs.items():
        p, m = fit_predict(tr, va, params, 3000, early_stop=True)
        sweep[(h, name)] = (ev.rmse(va['MONTHLY_RENT'], p), int(m.best_iteration_))
        say(f'  {h} {name:10s} RMSE {sweep[(h, name)][0]:7.2f}  best_iter {sweep[(h, name)][1]}')
table = pd.DataFrame({h: {n: sweep[(h, n)][0] for n in configs} for h, _ in HOLD})
table['mean'] = table.mean(axis=1)
say('\nsweep summary (lower is better):'); say(table.round(2).sort_values('mean').to_string())
best_name = table['mean'].idxmin()
best_params = configs[best_name]
say(f'\nbest config by mean of the two holdouts: {best_name}')

# ---------------- Part C: bagging and blend ----------------
say('\n=== Part C: seed bagging and XGBoost + CatBoost blend on the best config ===')
for h, cutoff in HOLD:
    tr, va = frames(cutoff)
    n_best = sweep[(h, best_name)][1]
    y = va['MONTHLY_RENT']
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    preds = {}
    bag = []
    for seed in range(5):
        m = lgb.LGBMRegressor(n_estimators=n_best, **dict(best_params, random_state=seed)).fit(Xtr, ytr)
        bag.append(m.predict(Xva))
    preds['lgb single'] = pd.Series(bag[0] + bench, index=va.index)
    preds['lgb 5-seed bag'] = pd.Series(np.mean(bag, axis=0) + bench, index=va.index)
    xm = xgb.XGBRegressor(n_estimators=n_best, learning_rate=0.04, max_depth=6, subsample=0.8,
                          colsample_bytree=0.75, min_child_weight=50, reg_lambda=10.0, enable_categorical=True,
                          random_state=0, n_jobs=-1, tree_method='hist').fit(Xtr, ytr)
    preds['xgb'] = pd.Series(xm.predict(Xva) + bench, index=va.index)
    ctr, cva = Xtr.copy(), Xva.copy()
    for c in CAT:
        ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
    cm = CatBoostRegressor(iterations=max(n_best, 300), learning_rate=0.05, depth=7, l2_leaf_reg=6.0, cat_features=CAT,
                           random_seed=0, thread_count=-1, verbose=False).fit(ctr, ytr)
    preds['catboost'] = pd.Series(cm.predict(cva) + bench, index=va.index)
    preds['blend lgb-bag/xgb/cb (0.4/0.2/0.4)'] = 0.4 * preds['lgb 5-seed bag'] + 0.2 * preds['xgb'] + 0.4 * preds['catboost']
    preds['blend lgb-bag/cb (0.5/0.5)'] = 0.5 * preds['lgb 5-seed bag'] + 0.5 * preds['catboost']
    for n, p in preds.items():
        say(f'  {h} {n:38s} RMSE {ev.rmse(y, p):7.2f}')
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    for n in ['lgb 5-seed bag', 'blend lgb-bag/xgb/cb (0.4/0.2/0.4)']:
        r = ev.paired_bootstrap_rmse_diff(y, preds['lgb single'], preds[n], groups, n_boot=300)
        say(f'  {h} {n} minus single lgb: {r["diff"]:+.2f}  CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
