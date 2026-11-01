"""
expt-5 final candidate: expt-2 V2's pipeline (public LB 433.607) plus the improvements that
held up on the clean time holdouts.

Recipe: V2 features with target encodings recomputed nested; anchor with town-ratio
fallback; time-aware address premium (trailing mean of rent - anchor per block); residual
target; regularised LightGBM bagged over 5 seeds blended 50/50 with CatBoost.

The recipe is first scored on both holdouts (train on the earlier rows, predict the later
ones, trees fixed), then refit on ALL training rows to predict test.csv.
Output: expt-5/outputs/submission_v2_improved.csv  (V2's own file is untouched).

Usage (repo root): python expt-5/src/final_candidate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
from catboost import CatBoostRegressor

log = open(os.path.join(ROOT, 'expt-5', 'outputs', 'final_candidate.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

TEST = os.path.join(ROOT, 'expt-2', 'outputs', 'features', 'test_features_v2.parquet')
SEEDS = range(5)
CB_PARAMS = dict(learning_rate=0.05, depth=7, l2_leaf_reg=6.0, random_seed=0, thread_count=-1, verbose=False)


def recipe(tr, va, n_lgb, n_cb):
    """LightGBM (5-seed bag, REG_PARAMS) and CatBoost on the residual; 50/50 blend."""
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    lgb_pred = np.mean([lgb.LGBMRegressor(n_estimators=n_lgb, **dict(REG_PARAMS, random_state=s)).fit(Xtr, ytr).predict(Xva)
                        for s in SEEDS], axis=0)
    ctr, cva = Xtr.copy(), Xva.copy()
    for c in CAT:
        ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
    cb_pred = CatBoostRegressor(iterations=n_cb, cat_features=CAT, **CB_PARAMS).fit(ctr, ytr).predict(cva)
    return pd.Series(0.5 * lgb_pred + 0.5 * cb_pred + bench, index=va.index)


# 1. honest check of the exact recipe on both holdouts (trees as in the experiments)
for h, cutoff in [('H1', '2024-07'), ('H2', '2023-12')]:
    tr, va = build(cutoff, anchor=True, trailing=True)
    p = recipe(tr, va, n_lgb=165, n_cb=300)
    say(f'{h} final recipe RMSE {ev.rmse(va["MONTHLY_RENT"], p):.2f}   (clean V2 baseline: {"452.48" if h == "H1" else "485.96"})')

# 2. refit on all training rows, predict test.csv
test = pd.read_parquet(TEST).drop(columns=TE_COLS)
test['postal_code_clean'] = test['postal_code_clean'].astype(str)
raw = pd.read_csv(os.path.join(ROOT, 'data', 'test.csv'))
assert len(test) == len(raw) == 50000
assert np.allclose(test['FLOOR_AREA_SQM'].values, raw['FLOOR_AREA_SQM'].values), 'test row order differs from test.csv'
assert (test['RENT_APPROVAL_DATE'].values == raw['RENT_APPROVAL_DATE'].values).all(), 'test dates differ from test.csv'
for c in CAT:
    cat_levels[c] = sorted(set(cat_levels[c]) | set(test[c].astype(str).unique()))

tr = full.copy()
tr, te = swap_anchor(tr, test)
tr, te = add_trailing(tr, te)
tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')
te_te = fe2.apply_hierarchical_te(te, maps, gm)
tr = pd.concat([tr, oof], axis=1)
te = pd.concat([te, te_te], axis=1)
te['MONTHLY_RENT'] = np.nan   # absent in test; recipe() never reads it for va
preds = recipe(tr, te, n_lgb=200, n_cb=350)

v2 = pd.read_csv(os.path.join(ROOT, 'expt-2', 'outputs', 'submissions', 'submission_ensemble_v2.csv'))['Predicted'].values
say(f'\ntest predictions: n={len(preds)} nulls={int(preds.isna().sum())} mean={preds.mean():.1f} std={preds.std():.1f} min={preds.min():.0f} max={preds.max():.0f}')
say(f'vs V2 submission: corr {np.corrcoef(preds.values, v2)[0, 1]:.4f}, mean abs diff {np.abs(preds.values - v2).mean():.1f}, mean diff {(preds.values - v2).mean():+.1f}')
out = os.path.join(ROOT, 'expt-5', 'outputs', 'submission_v2_improved.csv')
pd.DataFrame({'Id': np.arange(len(preds)), 'Predicted': preds.values}).to_csv(out, index=False)
say('written', out)
