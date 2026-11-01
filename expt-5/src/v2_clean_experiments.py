"""
expt-5 experiment 1: V2 recipe rebuilt leak-free on two time holdouts, one improvement at
a time (see expt5_lib.py for the setup). Usage (repo root): python expt-5/src/v2_clean_experiments.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *

results = {}
log = open(os.path.join(ROOT, 'expt-5', 'outputs', 'v2_clean_experiments.log'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# n_estimators for all variants: best_iteration of the clean baseline on H1 (one shared
# setting, so no variant is favoured; V2's own early-stopped count is not recorded).
tr, va = build('2024-07')
p, m = fit_predict(tr, va, V2_PARAMS, 1200, early_stop=True)
N_EST = int(m.best_iteration_)
say(f'baseline early-stopped best_iteration on H1: {N_EST}  (RMSE with early stopping on val, optimistic: {ev.rmse(va["MONTHLY_RENT"], p):.2f})')

for hname, cutoff in [('H1 (train<=2024-06, 9 months)', '2024-07'), ('H2 (train<=2023-11, 16 months)', '2023-12')]:
    say(f'\n===== {hname} =====')
    cut_ts = pd.Timestamp(cutoff + '-01')
    preds = {}
    for name, (anchor, trailing, params) in {
            'base': (False, False, V2_PARAMS), 'anchor': (True, False, V2_PARAMS),
            'trailing': (False, True, V2_PARAMS), 'both': (True, True, V2_PARAMS),
            'reg': (False, False, REG_PARAMS), 'both+reg': (True, True, REG_PARAMS)}.items():
        t0 = time.time()
        tr, va = build(cutoff, anchor, trailing)
        preds[name], _ = fit_predict(tr, va, params, N_EST)
        say(f'  {name:9s} RMSE {ev.rmse(va["MONTHLY_RENT"], preds[name]):7.2f}   ({time.time()-t0:.0f}s)')
    y = va['MONTHLY_RENT']
    groups = (va['block_id'] + '|' + va['RENT_APPROVAL_DATE']).values
    for name in preds:
        if name == 'base': continue
        r = ev.paired_bootstrap_rmse_diff(y, preds['base'], preds[name], groups, n_boot=300)
        say(f'  {name:9s} minus base: {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]')
    by_month = pd.DataFrame({n: ev.rmse_by_month_ahead(va, p, cut_ts) for n, p in preds.items()})
    say('  RMSE by months ahead (base / both / both+reg):')
    say(by_month[['base', 'both', 'both+reg']].round(1).to_string())
    results[hname] = {n: ev.rmse(y, p) for n, p in preds.items()}
say('\nsummary:', results)
