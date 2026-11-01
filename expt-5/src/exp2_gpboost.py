"""
expt-5 experiment 2 (the base): GPBoost replaces V2's hand-built target encodings.

Change under test (one only): V2's static hierarchical target encodings (te_town, te_block,
te_bft, te_street, te_subzone, fixed shrinkage constants) are removed and replaced by grouped
random effects for the same five levels, fit jointly with the boosted trees by likelihood
(Sigrist 2022, "Gaussian Process Boosting", JMLR 23(232)). Everything else is V2's original:
its benchmark, its features, LightGBM-style tree settings, residual target (rent - benchmark).

Because GPBoost re-estimates the variance components every round (~10s/round here), both
models use learning_rate 0.1 and the baseline is run at the same rate; the validation curves
(RMSE per boosting round, on the held-out window) are compared directly.

Outputs: expt-5/outputs/exp2_gpboost.json (curves, variance components, scores) and
exp2_preds_<H>.csv (validation predictions, reused by experiment 1).
Usage (repo root): python expt-5/src/exp2_gpboost.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
import gpboost as gpb

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp2_gpboost.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

full['block_ft_id'] = full['block_id'] + '_' + full['flat_type_clean']
GROUPS = ['town_clean', 'block_id', 'block_ft_id', 'street_clean', 'subzone_clean']
LR, ROUNDS = 0.1, 50
LGB_PARAMS = dict(V2_PARAMS, learning_rate=LR)
GP_PARAMS = dict(objective='regression_l2', learning_rate=LR, num_leaves=75, feature_fraction=0.75,
                 metric='rmse', verbose=-1)
FEATS_DROP = DROP + ['block_ft_id']


def gp_frames(cutoff):
    tr = full[full['RENT_APPROVAL_DATE'] < cutoff]
    va = full[full['RENT_APPROVAL_DATE'] >= cutoff]
    cols = [c for c in tr.columns if c not in FEATS_DROP]
    return tr, va, cols


def train_gp(tr, va, cols, rounds, with_valid):
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = (tr['MONTHLY_RENT'] - tr['median_rent_benchmark']).values
    yva = (va['MONTHLY_RENT'] - va['median_rent_benchmark']).values
    Gtr = tr[GROUPS].astype(str).reset_index(drop=True)
    Gva = va[GROUPS].astype(str).reset_index(drop=True)
    gp = gpb.GPModel(group_data=Gtr, likelihood='gaussian')
    dtr = gpb.Dataset(Xtr, label=ytr)
    kw = {}
    if with_valid:
        gp.set_prediction_data(group_data_pred=Gva)
        kw = dict(valid_sets=[gpb.Dataset(Xva, label=yva, reference=dtr)], valid_names=['valid'],
                  use_gp_model_for_validation=True)
    evals = {}
    bst = gpb.train(params=GP_PARAMS, train_set=dtr, num_boost_round=rounds, gp_model=gp,
                    evals_result=evals, verbose_eval=False, **kw)
    return bst, gp, Xva, Gva, evals


results = {}
for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    say(f'\n===== {h}: validate from {cutoff} =====')
    # --- baseline: V2's recipe, nested TE, LightGBM at the same learning rate, curve per round
    tr_b, va_b = build(cutoff)
    cols_b = [c for c in tr_b.columns if c not in DROP]
    m = lgb.LGBMRegressor(n_estimators=150, **LGB_PARAMS)
    m.fit(matrix(tr_b, cols_b), tr_b['MONTHLY_RENT'] - tr_b['median_rent_benchmark'],
          eval_set=[(matrix(va_b, cols_b), va_b['MONTHLY_RENT'] - va_b['median_rent_benchmark'])],
          eval_metric='rmse')
    base_curve = [float(v) for v in m.evals_result_['valid_0']['rmse']]
    kb = int(np.argmin(base_curve)) + 1
    say(f'  baseline LightGBM+TE (lr {LR}): best round {kb}, RMSE {base_curve[kb-1]:.2f}; at 20/40/50 rounds: '
        f'{base_curve[19]:.2f} / {base_curve[39]:.2f} / {base_curve[49]:.2f}')

    # --- GPBoost with validation curve
    tr, va, cols = gp_frames(cutoff)
    t0 = time.time()
    bst, gp, Xva, Gva, evals = train_gp(tr, va, cols, ROUNDS, with_valid=True)
    gp_curve = [float(v) for v in evals['valid']['rmse']]
    kg = int(np.argmin(gp_curve)) + 1
    cov = gp.get_cov_pars()
    say(f'  GPBoost ({time.time()-t0:.0f}s): best round {kg}, RMSE {gp_curve[kg-1]:.2f}; at 20/40/50 rounds: '
        f'{gp_curve[19]:.2f} / {gp_curve[39]:.2f} / {gp_curve[49]:.2f}')
    say('  learned variance components (SGD^2):', {c: round(float(cov[c].iloc[0]), 1) for c in cov.columns})

    # --- refit with exactly the best round count to get predictions for bootstrap/blending
    t0 = time.time()
    bst2, gp2, _, _, _ = train_gp(tr, va, cols, kg, with_valid=False)
    pr = bst2.predict(data=Xva, group_data_pred=Gva, pred_latent=False)
    gp_pred = pd.Series(np.asarray(pr['response_mean']) + va['median_rent_benchmark'].values, index=va.index)
    base_pred_model = lgb.LGBMRegressor(n_estimators=kb, **LGB_PARAMS).fit(
        matrix(tr_b, cols_b), tr_b['MONTHLY_RENT'] - tr_b['median_rent_benchmark'])
    base_pred = pd.Series(base_pred_model.predict(matrix(va_b, cols_b)) + va_b['median_rent_benchmark'].values, index=va_b.index)
    y = va['MONTHLY_RENT']
    say(f'  refit at best rounds ({time.time()-t0:.0f}s): GPBoost RMSE {ev.rmse(y, gp_pred):.2f} vs baseline {ev.rmse(y, base_pred):.2f}')
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    r = ev.paired_bootstrap_rmse_diff(y, base_pred, gp_pred, groups, n_boot=300)
    say(f'  GPBoost minus baseline: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    pd.DataFrame({'gpboost': gp_pred, 'base_lgb': base_pred, 'y': y}).to_csv(os.path.join(OUT, f'exp2_preds_{h}.csv'))
    results[h] = dict(base_curve=base_curve, gp_curve=gp_curve, base_best_round=kb, gp_best_round=kg,
                      variance_components={c: float(cov[c].iloc[0]) for c in cov.columns},
                      base_rmse=ev.rmse(y, base_pred), gp_rmse=ev.rmse(y, gp_pred), diff=r)
    json.dump(results, open(os.path.join(OUT, 'exp2_gpboost.json'), 'w'), indent=1)
say('\ndone')
