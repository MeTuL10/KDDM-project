"""
expt-5: temporal evidence for V2's blend weights.

V2's exact model settings (LightGBM / XGBoost / CatBoost, residual target, early stopping 40 on
the held-out window) with the hierarchical target encodings recomputed inside each split, on the
two time holdouts. Searches the weight simplex and compares against V2's own weights
(0.545 / 0.075 / 0.380, fit on its leaky OOT run).
Usage (repo root): python expt-5/src/blend_weights_evidence.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
import xgboost as xgb
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'blend_weights_evidence.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

V2_W = np.array([0.5449606385644811, 0.07532980214668984, 0.37970955928882905])
grid = [(a / 20, b / 20, (20 - a - b) / 20) for a in range(21) for b in range(21 - a)]
P, Y, G, scores = {}, {}, {}, {}

for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    tr, va = build(cutoff)                       # V2 benchmark + features, nested encodings
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                          random_state=42, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    p_lgb = m.predict(Xva) + bench
    xm = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.75,
                          enable_categorical=True, random_state=42, n_jobs=4, early_stopping_rounds=40)
    xm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_xgb = xm.predict(Xva) + bench
    ctr, cva = Xtr.copy(), Xva.copy()
    for c in CAT:
        ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
    cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=CAT, random_seed=42,
                           thread_count=4, verbose=False, early_stopping_rounds=40)
    cm.fit(ctr, ytr, eval_set=(cva, yva))
    p_cb = cm.predict(cva) + bench
    y = va['MONTHLY_RENT']
    P[h] = np.column_stack([p_lgb, p_xgb, p_cb]); Y[h] = y.values
    G[h] = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    say(f'{h}: single models RMSE  lgb {ev.rmse(y, p_lgb):.2f}  xgb {ev.rmse(y, p_xgb):.2f}  cb {ev.rmse(y, p_cb):.2f}'
        f'   (trees: lgb {m.best_iteration_}, xgb {xm.best_iteration}, cb {cm.get_best_iteration()})')
    say(f'{h}: V2 weights {np.round(V2_W, 3).tolist()} -> RMSE {ev.rmse(y, P[h] @ V2_W):.2f}')
    pd.DataFrame({'lgb': p_lgb, 'xgb': p_xgb, 'cb': p_cb, 'y': y.values}, index=va.index).to_csv(os.path.join(OUT, f'blend_evidence_preds_{h}.csv'))

rows = []
for w in grid:
    w_ = np.array(w)
    r = {h: ev.rmse(Y[h], P[h] @ w_) for h in P}
    rows.append((w, r['H1'], r['H2'], (r['H1'] + r['H2']) / 2))
tab = pd.DataFrame(rows, columns=['w', 'H1', 'H2', 'mean']).sort_values('mean')
say('\nbest 8 weight sets by mean RMSE over both holdouts (lgb, xgb, cb):')
say(tab.head(8).round(2).to_string(index=False))
best = np.array(tab.iloc[0]['w'])
say(f'\nchosen weights: {best.tolist()}   (V2 weights mean RMSE: '
    f'{np.mean([ev.rmse(Y[h], P[h] @ V2_W) for h in P]):.2f}, chosen: {tab.iloc[0]["mean"]:.2f})')
for h in P:
    r = ev.paired_bootstrap_rmse_diff(Y[h], P[h] @ V2_W, P[h] @ best, G[h], n_boot=300)
    say(f'{h}: chosen minus V2 weights {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
json.dump({'chosen_weights': best.tolist(), 'v2_weights': V2_W.tolist(),
           'table_top': tab.head(15).assign(w=tab.head(15)['w'].map(list)).to_dict('records')},
          open(os.path.join(OUT, 'blend_weights_evidence.json'), 'w'), indent=1)
say('done')
