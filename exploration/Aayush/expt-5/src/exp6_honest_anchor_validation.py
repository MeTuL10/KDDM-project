"""
expt-5 experiment 6: what does an honest (forecast-time) anchor cost V2's recipe on our two holdouts?

Variants of the benchmark for every row in a quarter after the cutoff quarter (cutoff = last quarter fully
before the validation window: H1 2024-Q2, H2 2023-Q3):
  published  the file's own medians (leaks the validation quarters' rents; what V2 and our earlier results used)
  nodrift    frozen at the cutoff quarter and carried forward
  drift4     carried forward x exp(g * quarters ahead), g = 4-quarter national log growth at the cutoff
Everything else is V2's recipe (benchmark residual target, nested hierarchical encodings, LightGBM and
CatBoost with V2's settings, early stopping on the held-out window for every variant alike, blend 0.2/0.8).
Usage (repo root): python exploration/Aayush/expt-5/src/exp6_honest_anchor_validation.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *            # full, matrix, DROP, CAT, fe2, lgb, ev, ROOT
import honest_lib as hl
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp6_honest_anchor_validation.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

summary = {}
for h, cutoff, cq in [('H1', '2024-07', '2024-Q2'), ('H2', '2023-12', '2023-Q3')]:
    say(f'\n===== {h}: validate from {cutoff}; anchor cutoff quarter {cq}; drift g = {100 * hl.drift(cq):+.2f}%/quarter =====')
    preds, y, groups, vaq = {}, None, None, None
    for variant in ['published', 'nodrift', 'drift4']:
        df = hl.make_variant(full, cq, variant)
        tr = df[df['RENT_APPROVAL_DATE'] < cutoff].copy()
        va = df[df['RENT_APPROVAL_DATE'] >= cutoff].copy()
        tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
        oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')
        tr, va = pd.concat([tr, oof], axis=1), pd.concat([va, fe2.apply_hierarchical_te(va, maps, gm)], axis=1)
        cols = [c for c in tr.columns if c not in DROP]
        Xtr, Xva = matrix(tr, cols), matrix(va, cols)
        ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
        yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
        bench = va['median_rent_benchmark'].values
        m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                              random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
        p_l = m.predict(Xva) + bench
        ctr, cva = Xtr.copy(), Xva.copy()
        for c in CAT:
            ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
        cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=CAT, random_seed=42,
                               thread_count=-1, verbose=False, early_stopping_rounds=40)
        cm.fit(ctr, ytr, eval_set=(cva, yva))
        p_c = cm.predict(cva) + bench
        y = va['MONTHLY_RENT']
        groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
        vaq = va['approval_quarter'].values
        p_b = pd.Series(0.2 * p_l + 0.8 * p_c, index=va.index)
        preds[variant] = p_b
        say(f'  {variant:9s} lgb {ev.rmse(y, p_l):7.2f}   cb {ev.rmse(y, p_c):7.2f}   blend {ev.rmse(y, p_b):7.2f}'
            f'   mean error (actual - predicted) {float((y - p_b).mean()):+6.1f}')
    for v in ['nodrift', 'drift4']:
        r = ev.paired_bootstrap_rmse_diff(y, preds['published'], preds[v], groups, n_boot=300)
        say(f'  {v:9s} minus published: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    r = ev.paired_bootstrap_rmse_diff(y, preds['nodrift'], preds['drift4'], groups, n_boot=300)
    say(f'  drift4    minus nodrift:   {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    say('  mean error by validation quarter (published / nodrift / drift4):')
    for q in sorted(set(vaq)):
        mk = vaq == q
        say(f'    {q}  n={mk.sum():5d}   ' + ' / '.join(f'{float((y[mk] - preds[v][mk]).mean()):+6.1f}' for v in ['published', 'nodrift', 'drift4']))
    summary[h] = {v: ev.rmse(y, p) for v, p in preds.items()}
json.dump(summary, open(os.path.join(OUT, 'exp6_honest_anchor_validation.json'), 'w'), indent=1)
say('\ndone')
