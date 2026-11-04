"""
expt-5 experiment 1 (builds on experiment 2): CatBoost with raw categoricals, and shrinkage
constants derived from experiment 2's learned variance components.

(a) Tuned shrinkage. V2's target encodings shrink each level toward its parent with fixed
    constants M (town 30, block 10, block x type 5, street 15, subzone 20). A random-effects
    model implies M_level = sigma2_error / sigma2_level. Two sources of those variances:
      cond  experiment 2's GPBoost fit (variances AFTER the trees explain the other features)
      marg  an intercept-only random-effects fit on the raw residual (the scale V2's encodings
            actually operate on, since they see no trees)
    Each M set feeds V2's own encoding code + LightGBM, vs V2's constants.
(b) CatBoost with raw block / block x type / street categoricals (ordered target statistics),
    without ('cb_raw') and with ('cb_raw_te') the tuned encodings, vs CatBoost on V2's
    encodings ('cb_te').
(c) Blends of experiment 2's GPBoost with the CatBoost variants.

Everything uses V2's original benchmark and nested encodings; trees/iterations by best round on
the held-out window (optimistic, but identical for every variant). Holdouts H1/H2 as before.
Usage (repo root): python exploration/Aayush/expt-5/src/exp1_catboost_tuned_te.py   (needs exp2 outputs)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
from sklearn.model_selection import KFold
import gpboost as gpb
from catboost import CatBoostRegressor

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp1_catboost_tuned_te.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

exp2 = json.load(open(os.path.join(OUT, 'exp2_gpboost.json')))
full['block_ft_id'] = full['block_id'] + '_' + full['flat_type_clean']
GROUPS = ['town_clean', 'block_id', 'block_ft_id', 'street_clean', 'subzone_clean']
LEVEL_COL = dict(town='town_clean', block='block_id', bft='block_ft_id', street='street_clean', subzone='subzone_clean')
V2_M = dict(town=30.0, block=10.0, bft=5.0, street=15.0, subzone=20.0)
RAW_CATS = ['block_id', 'block_ft_id', 'street_clean']
LGB_PARAMS = dict(V2_PARAMS, learning_rate=0.1)
HOLD = [('H1', '2024-07'), ('H2', '2023-12')]


def m_from_variances(v):
    e = v['Error_var']
    return {k: float(np.clip(e / v[col], 1.0, 1000.0)) for k, col in LEVEL_COL.items()}


def te_maps(tr, M, gm, col='target_res'):
    t = tr.groupby('town_clean')[col].agg(['count', 'mean'])
    t_map = ((t['count'] * t['mean'] + M['town'] * gm) / (t['count'] + M['town'])).to_dict()
    b = tr.groupby('block_id').agg(count=(col, 'count'), mean=(col, 'mean'), town=('town_clean', 'first'))
    b['prior'] = b['town'].map(t_map).fillna(gm)
    b_map = ((b['count'] * b['mean'] + M['block'] * b['prior']) / (b['count'] + M['block'])).to_dict()
    f = tr.groupby('block_ft_id').agg(count=(col, 'count'), mean=(col, 'mean'), block=('block_id', 'first'))
    f['prior'] = f['block'].map(b_map).fillna(gm)
    bft_map = ((f['count'] * f['mean'] + M['bft'] * f['prior']) / (f['count'] + M['bft'])).to_dict()
    s = tr.groupby('street_clean').agg(count=(col, 'count'), mean=(col, 'mean'), town=('town_clean', 'first'))
    s['prior'] = s['town'].map(t_map).fillna(gm)
    s_map = ((s['count'] * s['mean'] + M['street'] * s['prior']) / (s['count'] + M['street'])).to_dict()
    z = tr.groupby('subzone_clean')[col].agg(['count', 'mean'])
    sub_map = ((z['count'] * z['mean'] + M['subzone'] * gm) / (z['count'] + M['subzone'])).to_dict()
    return t_map, b_map, bft_map, s_map, sub_map


def te_oof(tr, M, col='target_res'):
    """Same out-of-fold scheme as expt-2 V2's compute_hierarchical_te, with M as a parameter."""
    gm = tr[col].mean()
    oof = pd.DataFrame(index=tr.index, columns=TE_COLS, dtype=float)
    for a, b in KFold(5, shuffle=True, random_state=42).split(tr):
        t_map, b_map, bft_map, s_map, sub_map = te_maps(tr.iloc[a], M, gm, col)
        v = tr.iloc[b]
        town = v['town_clean'].map(t_map).fillna(gm)
        block = v['block_id'].map(b_map).fillna(town)
        bft = v['block_ft_id'].map(bft_map).fillna(block)
        street = v['street_clean'].map(s_map).fillna(town)
        sub = v['subzone_clean'].map(sub_map).fillna(gm)
        for i, s_ in enumerate([town, block, bft, street, sub]):
            oof.iloc[b, i] = s_
    return oof


def build_te(cutoff, M, with_te=True):
    tr = full[full['RENT_APPROVAL_DATE'] < cutoff].copy()
    va = full[full['RENT_APPROVAL_DATE'] >= cutoff].copy()
    if not with_te:
        return tr, va
    tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    gm = tr['target_res'].mean()
    oof = te_oof(tr, M)
    te_va = fe2.apply_hierarchical_te(va, te_maps(tr, M, gm), gm)
    return pd.concat([tr, oof], axis=1), pd.concat([va, te_va], axis=1)


def marginal_variances(tr):
    y = (tr['MONTHLY_RENT'] - tr['median_rent_benchmark']).values
    gp = gpb.GPModel(group_data=tr[GROUPS].astype(str).reset_index(drop=True), likelihood='gaussian')
    gp.fit(y=y, X=np.ones((len(y), 1)))
    cov = gp.get_cov_pars()
    return {c: float(cov[c].iloc[0]) for c in cov.columns}


def lgb_run(tr, va, max_rounds=150):
    cols = [c for c in tr.columns if c not in DROP]
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    m = lgb.LGBMRegressor(n_estimators=max_rounds, **LGB_PARAMS)
    m.fit(matrix(tr, cols), ytr, eval_set=[(matrix(va, cols), yva)], eval_metric='rmse')
    k = int(np.argmin(m.evals_result_['valid_0']['rmse'])) + 1
    return pd.Series(m.predict(matrix(va, cols), num_iteration=k) + va['median_rent_benchmark'].values, index=va.index), k


def cb_run(tr, va, raw=False, iterations=500):
    cols = [c for c in tr.columns if c not in DROP] + (RAW_CATS if raw else [])
    cats = CAT + (RAW_CATS if raw else [])
    def mat(d):
        X = d[cols].copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return X
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    m = CatBoostRegressor(iterations=iterations, learning_rate=0.05, depth=7, l2_leaf_reg=6.0, cat_features=cats,
                          random_seed=0, thread_count=-1, verbose=False, use_best_model=False)
    m.fit(mat(tr), ytr, eval_set=(mat(va), yva))
    k = int(np.argmin(m.get_evals_result()['validation']['RMSE'])) + 1
    return pd.Series(m.predict(mat(va), ntree_end=k) + va['median_rent_benchmark'].values, index=va.index), k


preds, Y, Ms = {}, {}, {}
for h, cutoff in HOLD:
    say(f'\n===== {h}: validate from {cutoff} =====')
    ref = pd.read_csv(os.path.join(OUT, f'exp2_preds_{h}.csv'), index_col=0)
    y = Y[h] = ref['y']
    preds[(h, 'lgb_te_v2 (exp2 baseline)')] = ref['base_lgb']
    preds[(h, 'gpboost (exp2 base)')] = ref['gpboost']
    tr0, _ = build_te(cutoff, V2_M, with_te=False)
    t0 = time.time()
    marg = marginal_variances(tr0)
    cond = exp2[h]['variance_components']
    Ms[h] = {'v2': V2_M, 'cond': m_from_variances(cond), 'marg': m_from_variances(marg)}
    say(f'  marginal random-effects fit took {time.time()-t0:.0f}s')
    say('  variances cond:', {k: round(v, 1) for k, v in cond.items()})
    say('  variances marg:', {k: round(v, 1) for k, v in marg.items()})
    for name, M in Ms[h].items():
        say(f'  shrinkage M [{name}]:', {k: round(v, 1) for k, v in M.items()})
    for name in ['cond', 'marg']:
        tr, va = build_te(cutoff, Ms[h][name])
        p, k = lgb_run(tr, va)
        preds[(h, f'lgb_te_{name}')] = p
        say(f'  lgb with TE M={name}: RMSE {ev.rmse(y, p):.2f} (round {k})')

# choose the tuned M set by mean RMSE across both holdouts (not per holdout)
mean_by = {n: float(np.mean([ev.rmse(Y[h], preds[(h, f'lgb_te_{n}')]) for h, _ in HOLD])) for n in ['cond', 'marg']}
best_M = min(mean_by, key=mean_by.get)
say(f'\ntuned-M set carried into CatBoost: {best_M}  (mean RMSE {mean_by})')

for h, cutoff in HOLD:
    say(f'\n===== {h}: CatBoost variants =====')
    y = pd.read_csv(os.path.join(OUT, f'exp2_preds_{h}.csv'), index_col=0)['y']
    for name, kw in [('cb_te (V2 encodings)', dict(raw=False)), ('cb_raw', dict(raw=True)), ('cb_raw_te', dict(raw=True))]:
        t0 = time.time()
        if name == 'cb_te (V2 encodings)':
            tr, va = build_te(cutoff, V2_M)
        elif name == 'cb_raw':
            tr, va = build_te(cutoff, V2_M, with_te=False)
        else:
            tr, va = build_te(cutoff, Ms[h][best_M])
        p, k = cb_run(tr, va, **kw)
        preds[(h, name)] = p
        say(f'  {name:22s} RMSE {ev.rmse(y, p):.2f} (iteration {k}, {time.time()-t0:.0f}s)')
    for cbn in ['cb_te (V2 encodings)', 'cb_raw', 'cb_raw_te']:
        preds[(h, f'blend gpboost+{cbn}')] = 0.5 * preds[(h, 'gpboost (exp2 base)')] + 0.5 * preds[(h, cbn)]

say('\n===== summary: RMSE and paired-bootstrap differences =====')
names = ['lgb_te_v2 (exp2 baseline)', 'gpboost (exp2 base)', 'lgb_te_cond', 'lgb_te_marg', 'cb_te (V2 encodings)',
         'cb_raw', 'cb_raw_te', 'blend gpboost+cb_te (V2 encodings)', 'blend gpboost+cb_raw', 'blend gpboost+cb_raw_te']
summary = {}
for h, cutoff in HOLD:
    ref = pd.read_csv(os.path.join(OUT, f'exp2_preds_{h}.csv'), index_col=0)
    y = ref['y']
    groups = (full.loc[y.index, 'block_id'] + '|' + full.loc[y.index, 'RENT_APPROVAL_DATE']).values
    say(f'\n{h}:')
    for n in names:
        p = preds[(h, n)]
        r_base = ev.paired_bootstrap_rmse_diff(y, preds[(h, names[0])], p, groups, n_boot=300) if n != names[0] else None
        r_gp = ev.paired_bootstrap_rmse_diff(y, preds[(h, names[1])], p, groups, n_boot=300) if n != names[1] else None
        fmt = lambda r: '      -        ' if r is None else f'{r["diff"]:+6.2f} [{r["ci_low"]:+.2f},{r["ci_high"]:+.2f}]'
        say(f'  {n:38s} RMSE {ev.rmse(y, p):7.2f}   vs lgb_te_v2 {fmt(r_base)}   vs gpboost {fmt(r_gp)}')
        summary[f'{h}|{n}'] = ev.rmse(y, p)
json.dump(dict(M=Ms, best_M=best_M, rmse=summary), open(os.path.join(OUT, 'exp1_catboost_tuned_te.json'), 'w'), indent=1)
say('\ndone')
