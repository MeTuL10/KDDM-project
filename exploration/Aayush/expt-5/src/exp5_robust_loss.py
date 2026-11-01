"""
expt-5 experiment 5: robust training losses inside V2's recipe (residual target, nested encodings).

LightGBM: squared (reference), Huber (delta 300 / 600 / 1000 SGD), Fair (c 200 / 500), L1.
CatBoost: squared (reference), Huber (delta 600).
Each fit is scored raw and "mean-corrected": predictions shifted by the mean training residual, which
removes the location bias a robust loss introduces under skewed noise (keeps any variance benefit).
Early stopping on validation RMSE for every variant alike. Paired bootstrap vs the squared-loss fit of
the same model. Usage (repo root): python exploration/Aayush/expt-5/src/exp5_robust_loss.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp5_robust_loss.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

BASE = dict(n_estimators=2000, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
            random_state=42, n_jobs=-1, verbose=-1, metric='rmse')
LGB_CONFIGS = {'lgb squared': dict(objective='regression'),
               'lgb huber 300': dict(objective='huber', alpha=300.0),
               'lgb huber 600': dict(objective='huber', alpha=600.0),
               'lgb huber 1000': dict(objective='huber', alpha=1000.0),
               'lgb fair 200': dict(objective='fair', fair_c=200.0),
               'lgb fair 500': dict(objective='fair', fair_c=500.0),
               'lgb l1': dict(objective='regression_l1')}
CB_CONFIGS = {'cb squared': 'RMSE', 'cb huber 600': 'Huber:delta=600'}
summary = {}
for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    say(f'\n===== {h}: validate from {cutoff} =====')
    tr, va = build(cutoff)
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    y = va['MONTHLY_RENT']
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    say(f'  training residual: mean {ytr.mean():.1f}, median {ytr.median():.1f}, skew {ytr.skew():.2f}')
    preds = {}
    for name, kw in LGB_CONFIGS.items():
        m = lgb.LGBMRegressor(**BASE, **kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='rmse', callbacks=[lgb.early_stopping(60, verbose=False)])
        shift = float((ytr - m.predict(Xtr)).mean())
        raw = pd.Series(m.predict(Xva) + bench, index=va.index)
        preds[name], preds[name + ' +meanfix'] = raw, raw + shift
        say(f'  {name:16s} trees {m.best_iteration_:4d}  RMSE raw {ev.rmse(y, raw):7.2f}   mean-corrected {ev.rmse(y, raw + shift):7.2f}  (shift {shift:+.1f})')
    ctr, cva = Xtr.copy(), Xva.copy()
    for c in CAT:
        ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
    for name, loss in CB_CONFIGS.items():
        cm = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=7, cat_features=CAT, loss_function=loss,
                               eval_metric='RMSE', random_seed=42, thread_count=-1, verbose=False, early_stopping_rounds=60)
        cm.fit(ctr, ytr, eval_set=(cva, yva))
        shift = float((ytr - cm.predict(ctr)).mean())
        raw = pd.Series(cm.predict(cva) + bench, index=va.index)
        preds[name], preds[name + ' +meanfix'] = raw, raw + shift
        say(f'  {name:16s} trees {cm.get_best_iteration():4d}  RMSE raw {ev.rmse(y, raw):7.2f}   mean-corrected {ev.rmse(y, raw + shift):7.2f}  (shift {shift:+.1f})')
    for fam, ref in (('lgb', 'lgb squared'), ('cb', 'cb squared')):
        for name in [n for n in preds if n.startswith(fam) and n != ref and not n.startswith(ref + ' ')]:
            r = ev.paired_bootstrap_rmse_diff(y, preds[ref], preds[name], groups, n_boot=300)
            say(f'  {name:26s} minus {ref}: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    summary[h] = {n: ev.rmse(y, p) for n, p in preds.items()}
json.dump(summary, open(os.path.join(OUT, 'exp5_robust_loss.json'), 'w'), indent=1)
say('\ndone')
