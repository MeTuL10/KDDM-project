"""
expt-5 experiment 3: external resale signals (E1) and neighbour-block priors (E2) on V2's recipe.
Base = V2's benchmark and features with nested encodings. LightGBM and CatBoost with V2's settings,
early stopping on the held-out window for every variant alike (optimistic but equal). Paired bootstrap
vs base. Usage (repo root): python exploration/Aayush/expt-5/src/exp3_external_neighbours.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_extra import *
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp3_external_neighbours.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

NEW = ['resale_prem_block', 'resale_n_block', 'resale_mean_storey', 'resale_prem_bt', 'nb_mu5', 'nb_mu15', 'nb_dist5']
VARIANTS = {'base': (False, False), '+resale': (True, False), '+neighbours': (False, True), '+resale+neighbours': (True, True)}
summary = {}
for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    say(f'\n===== {h}: validate from {cutoff} =====')
    tr0, va0 = build(cutoff)
    g, bt = resale_block_features(cutoff)
    cov = add_resale(va0, g, bt)['resale_prem_block'].notna().mean()
    say(f'  resale rows used: {(RESALE["month"] < cutoff).sum():,}; validation rows with block resale history: {100 * cov:.1f}%')
    preds, y = {}, va0['MONTHLY_RENT']
    groups = (va0['block_id'] + '|' + va0['RENT_APPROVAL_DATE']).values
    for name, (use_resale, use_nb) in VARIANTS.items():
        tr, va = tr0, va0
        if use_resale:
            tr, va = add_resale(tr, g, bt), add_resale(va, g, bt)
        if use_nb:
            tr, va = neighbour_features(tr, va)
        cols = [c for c in tr.columns if c not in DROP]
        Xtr, Xva = matrix(tr, cols), matrix(va, cols)
        ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
        yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
        bench = va['median_rent_benchmark'].values
        m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                              random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
        p_lgb = pd.Series(m.predict(Xva) + bench, index=va.index)
        ctr, cva = Xtr.copy(), Xva.copy()
        for c in CAT:
            ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
        cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=CAT, random_seed=42,
                               thread_count=-1, verbose=False, early_stopping_rounds=40)
        cm.fit(ctr, ytr, eval_set=(cva, yva))
        p_cb = pd.Series(cm.predict(cva) + bench, index=va.index)
        preds[name] = {'lgb': p_lgb, 'cb': p_cb, 'blend': 0.2 * p_lgb + 0.8 * p_cb}
        say(f'  {name:20s} lgb {ev.rmse(y, p_lgb):7.2f} (trees {m.best_iteration_})   cb {ev.rmse(y, p_cb):7.2f} (trees {cm.get_best_iteration()})'
            f'   blend(0.2/0.8) {ev.rmse(y, preds[name]["blend"]):7.2f}')
        if name == '+resale+neighbours':
            imp = pd.Series(m.booster_.feature_importance('gain'), index=cols)
            say('  LightGBM gain share of the new features:', {c: f'{100 * imp[c] / imp.sum():.1f}%' for c in NEW if c in imp.index})
    for name in list(VARIANTS)[1:]:
        for model in ('lgb', 'cb', 'blend'):
            r = ev.paired_bootstrap_rmse_diff(y, preds['base'][model], preds[name][model], groups, n_boot=300)
            say(f'  {name:20s} {model:5s} minus base: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    summary[h] = {n: {m_: ev.rmse(y, p) for m_, p in d.items()} for n, d in preds.items()}
    pd.concat({f'{n}|{m_}': p for n, d in preds.items() for m_, p in d.items()}, axis=1).assign(y=y).to_csv(os.path.join(OUT, f'exp3_preds_{h}.csv'))
json.dump(summary, open(os.path.join(OUT, 'exp3_external_neighbours.json'), 'w'), indent=1)
say('\ndone')
