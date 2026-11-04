"""
expt-5 experiment 4: remove extreme rows from BOTH ends of the training data, then train V2's recipe.

"Extreme" is market-adjusted and ranked within flat type (so each type loses the same share):
  anchor   rent - median-rent benchmark (V2's own residual target)
  oof      out-of-fold residual of a first-pass LightGBM (V2 features), a stricter, model-based cut
Trim levels: q = 0.5%, 1%, 2.5% off EACH end. Thresholds come from training rows only. Validation is
left complete (the test set has tails too). Hierarchical encodings are recomputed on the trimmed rows.
LightGBM and CatBoost with V2's settings, early stopping on the held-out window for every variant alike.
Usage (repo root): python exploration/Aayush/expt-5/src/exp4_tail_trimming.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp4_tail_trimming.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

QS = [0.005, 0.01, 0.025]


def with_te(tr, va):
    tr, va = tr.copy(), va.copy()
    tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')
    return pd.concat([tr, oof], axis=1), pd.concat([va, fe2.apply_hierarchical_te(va, maps, gm)], axis=1)


def keep_mask(score, ft, q):
    """True = keep: rank of `score` within flat type must lie in (q, 1-q]."""
    pct = pd.Series(np.asarray(score), index=ft.index).groupby(ft.values).rank(pct=True)
    return ((pct > q) & (pct <= 1 - q)).values


def oof_residual(tr_te):
    cols = [c for c in tr_te.columns if c not in DROP]
    X = matrix(tr_te, cols)
    y = (tr_te['MONTHLY_RENT'] - tr_te['median_rent_benchmark']).values
    e = np.zeros(len(tr_te))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        m = lgb.LGBMRegressor(n_estimators=106, **V2_PARAMS).fit(X.iloc[a], y[a])
        e[b] = y[b] - m.predict(X.iloc[b])
    return e


def fit_eval(tr, va):
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                          random_state=42, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    p_lgb = m.predict(Xva) + bench
    ctr, cva = Xtr.copy(), Xva.copy()
    for c in CAT:
        ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
    cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=CAT, random_seed=42,
                           thread_count=-1, verbose=False, early_stopping_rounds=40)
    cm.fit(ctr, ytr, eval_set=(cva, yva))
    p_cb = cm.predict(cva) + bench
    return pd.Series(0.2 * p_lgb + 0.8 * p_cb, index=va.index), p_lgb, p_cb


summary = {}
for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    say(f'\n===== {h}: validate from {cutoff} =====')
    tr_all = full[full['RENT_APPROVAL_DATE'] < cutoff]
    va = full[full['RENT_APPROVAL_DATE'] >= cutoff]
    y = va['MONTHLY_RENT']
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    a = (va['MONTHLY_RENT'] - va['median_rent_benchmark'])
    central = ((a.groupby(va['flat_type_clean'].values).rank(pct=True)) > 0.05).values & \
              ((a.groupby(va['flat_type_clean'].values).rank(pct=True)) <= 0.95).values
    # scores for trimming, from training rows only
    r_anchor = (tr_all['MONTHLY_RENT'] - tr_all['median_rent_benchmark']).values
    tr0, _ = with_te(tr_all, va)
    r_oof = oof_residual(tr0)
    preds = {}
    plan = [('base', None, None)] + [(f'{m}@{q:.1%}', m, q) for m in ('anchor', 'oof') for q in QS]
    for name, method, q in plan:
        if method is None:
            keep = np.ones(len(tr_all), dtype=bool)
        else:
            score = r_anchor if method == 'anchor' else r_oof
            keep = keep_mask(score, tr_all['flat_type_clean'], q)
        tr, vv = with_te(tr_all[keep], va)
        p, p_lgb, p_cb = fit_eval(tr, vv)
        preds[name] = p
        c, o = ~central, central
        say(f'  {name:12s} removed {100 * (1 - keep.mean()):4.1f}% ({(~keep).sum():5d} rows)   RMSE blend {ev.rmse(y, p):7.2f}'
            f'  (lgb {ev.rmse(y, p_lgb):.2f}, cb {ev.rmse(y, p_cb):.2f})   central-90% {ev.rmse(y[central], p[central]):7.2f}'
            f'   outer-10% {ev.rmse(y[~central], p[~central]):7.2f}')
    for name in list(preds)[1:]:
        r = ev.paired_bootstrap_rmse_diff(y, preds['base'], preds[name], groups, n_boot=300)
        say(f'  {name:12s} minus base: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    summary[h] = {n: ev.rmse(y, p) for n, p in preds.items()}
json.dump(summary, open(os.path.join(OUT, 'exp4_tail_trimming.json'), 'w'), indent=1)
say('\ndone')
