"""
expt-5 experiment 7: does V2's recipe need the COE and stock columns?

The final V2 model keeps nine macro columns (coe_a, coe_b, coe_c, coe_e, coe_overall_mean, three bank closing prices and
sg_financial_index). The R pipeline dropped them after an ablation (report/beina/model_experiment.md section 11); this
measures the same thing inside V2's recipe. Everything else is exp6's setup: benchmark residual target, nested
hierarchical encodings, LightGBM, XGBoost and CatBoost with V2's settings, early stopping on the held-out window for
every variant alike, paired block-by-month bootstrap (300 draws). Two blends are reported: the final model's weights
(0.545 / 0.075 / 0.380 for LightGBM / XGBoost / CatBoost) and exp6's LightGBM + CatBoost 0.2 / 0.8, for comparison
with the honest-anchor table.

  H1  validate 2024-07..2025-03, honest anchor with drift (cutoff 2024-Q2: the variant the final model uses)
  H2  validate 2023-12..2025-03, honest anchor frozen at 2023-Q3 (no drift: at that cutoff drift is +5%/quarter and
      extrapolates badly, exp6, so a drifted run would measure the drift rule, not the columns)
Three seeds per cell (42, 43, 44) because the R pipeline's effect was inside its seed spread.
Usage (repo root): python exploration/Aayush/expt-5/src/exp7_coe_stocks_ablation.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *            # full, matrix, DROP, CAT, fe2, lgb, ev, ROOT
import honest_lib as hl
from catboost import CatBoostRegressor
import xgboost as xgb

OUT = os.path.join(ROOT, 'expt-5', 'outputs')
log = open(os.path.join(OUT, 'exp7_coe_stocks_ablation.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

SEEDS = [42, 43, 44]
W3 = np.array([0.545, 0.0753, 0.3797])   # final model's blend weights (lgb, xgb, cb), from the out-of-time fit
MACRO = [c for c in full.columns if c.startswith(('coe_', 'stock_')) or c == 'sg_financial_index']
assert len(MACRO) == 9, MACRO
say('dropped in the ablation:', MACRO)

summary = {}
for h, cutoff, cq, variant in [('H1', '2024-07', '2024-Q2', 'drift4'), ('H2', '2023-12', '2023-Q3', 'nodrift')]:
    say(f'\n===== {h}: validate from {cutoff}; anchor {variant} at {cq} =====')
    df = hl.make_variant(full, cq, variant)
    tr = df[df['RENT_APPROVAL_DATE'] < cutoff].copy()
    va = df[df['RENT_APPROVAL_DATE'] >= cutoff].copy()
    tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')
    tr, va = pd.concat([tr, oof], axis=1), pd.concat([va, fe2.apply_hierarchical_te(va, maps, gm)], axis=1)
    all_cols = [c for c in tr.columns if c not in DROP]
    cols_by = {'all features': all_cols, 'without COE/stocks': [c for c in all_cols if c not in MACRO]}
    say(f'  features: {len(all_cols)} with, {len(cols_by["without COE/stocks"])} without')
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    bench = va['median_rent_benchmark'].values
    y = va['MONTHLY_RENT']
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    per_seed, blends = {}, {}
    for name, cols in cols_by.items():
        Xtr, Xva = matrix(tr, cols), matrix(va, cols)
        ctr, cva = Xtr.copy(), Xva.copy()
        for c in CAT:
            ctr[c], cva[c] = ctr[c].astype(str), cva[c].astype(str)
        seed_final, seed_two = [], []
        for s in SEEDS:
            m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                                  random_state=s, n_jobs=-1, verbose=-1)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
            xm = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.75,
                                  enable_categorical=True, random_state=s, n_jobs=-1, early_stopping_rounds=40)
            xm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            cm = CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=7, cat_features=CAT, random_seed=s,
                                   thread_count=-1, verbose=False, early_stopping_rounds=40, allow_writing_files=False)
            cm.fit(ctr, ytr, eval_set=(cva, yva))
            pl, px, pc = m.predict(Xva) + bench, xm.predict(Xva) + bench, cm.predict(cva) + bench
            p3 = W3[0] * pl + W3[1] * px + W3[2] * pc
            p2 = 0.2 * pl + 0.8 * pc
            seed_final.append(p3); seed_two.append(p2)
            say(f'  {name:19s} seed {s}: final-weights blend {ev.rmse(y, p3):.2f}   lgb+cb 0.2/0.8 blend {ev.rmse(y, p2):.2f}')
        for key, sp in (('final weights', seed_final), ('lgb+cb 0.2/0.8', seed_two)):
            rm = [ev.rmse(y, p) for p in sp]
            blends[(name, key)] = pd.Series(np.mean(sp, axis=0), index=va.index)
            per_seed[(name, key)] = rm
            say(f'  {name:19s} [{key}] mean of seeds {np.mean(rm):.2f} (sd {np.std(rm, ddof=1):.2f}); seed-averaged blend {ev.rmse(y, blends[(name, key)]):.2f}')
    a, b = 'all features', 'without COE/stocks'
    res = {}
    for key in ('final weights', 'lgb+cb 0.2/0.8'):
        r = ev.paired_bootstrap_rmse_diff(y, blends[(a, key)], blends[(b, key)], groups, n_boot=300)
        d_mean = float(np.mean(per_seed[(b, key)]) - np.mean(per_seed[(a, key)]))
        say(f'  [{key}] without minus all: mean of seeds {d_mean:+.2f}; seed-averaged blend, paired bootstrap {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
        res[key] = dict(mean_diff=d_mean, boot=r, per_seed={n: per_seed[(n, key)] for n in (a, b)})
    summary[h] = dict(variant=variant, results=res)
json.dump(summary, open(os.path.join(OUT, 'exp7_coe_stocks_ablation.json'), 'w'), indent=1, default=float)
say('\ndone')
