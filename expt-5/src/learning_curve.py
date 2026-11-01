"""
expt-5: would more training rows, or rows closer to the target window, help?

Validation window fixed at 2024-10..2025-03. V2's recipe (benchmark residual, hierarchical
encodings recomputed inside each training set, LightGBM with V2's settings, 106 trees fixed so no
variant gets validation-chosen trees).
 Part 1  learning curve: random subsamples of all rows before 2024-10
 Part 2  recency: training ends 0..15 months before the window, vs a same-size random control
         drawn from all rows before 2024-10 (same row count, but includes recent months)
Usage (repo root): python expt-5/src/learning_curve.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *

log = open(os.path.join(ROOT, 'expt-5', 'outputs', 'learning_curve.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

VAL = full[(full['RENT_APPROVAL_DATE'] >= '2024-10') & (full['RENT_APPROVAL_DATE'] <= '2025-03')].copy()


def score(tr, n_trees=106):
    tr = tr.copy()
    va = VAL.copy()
    tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')
    te_va = fe2.apply_hierarchical_te(va, maps, gm)
    tr, va = pd.concat([tr, oof], axis=1), pd.concat([va, te_va], axis=1)
    cols = [c for c in tr.columns if c not in DROP]
    m = lgb.LGBMRegressor(n_estimators=n_trees, **V2_PARAMS)
    m.fit(matrix(tr, cols), tr['MONTHLY_RENT'] - tr['median_rent_benchmark'])
    return ev.rmse(va['MONTHLY_RENT'], m.predict(matrix(va, cols)) + va['median_rent_benchmark'].values)


pool = full[full['RENT_APPROVAL_DATE'] < '2024-10']
say(f'validation rows: {len(VAL):,}; all earlier rows: {len(pool):,}')

say('\nPart 1: learning curve (random subsamples of all earlier rows)')
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    seeds = [0] if frac == 1.0 else [0, 1, 2]
    r = [score(pool.sample(frac=frac, random_state=s) if frac < 1 else pool) for s in seeds]
    say(f'  {frac:4.0%}  {int(len(pool) * frac):7,} rows   RMSE {np.mean(r):7.2f}   (spread {np.ptp(r):.2f} over {len(seeds)} draws)')

say('\nPart 2: recency (training ends N months before the window) vs same-size random control')
for gap, end in [(0, '2024-10'), (3, '2024-07'), (6, '2024-04'), (9, '2024-01'), (12, '2023-10'), (15, '2023-07')]:
    older = full[full['RENT_APPROVAL_DATE'] < end]
    r_old = score(older)
    r_ctl = np.mean([score(pool.sample(n=len(older), random_state=s)) for s in (0, 1)]) if gap else r_old
    say(f'  gap {gap:2d} mo  {len(older):7,} rows   ends-earlier RMSE {r_old:7.2f}   same-size control {r_ctl:7.2f}   '
        f'cost of being older: {r_old - r_ctl:+.2f}')
say('done')
