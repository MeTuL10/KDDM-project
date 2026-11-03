"""
expt-5 experiment 8: where do the COE and stock columns of V2's features sit in the test months, relative to training?

For each of the nine macro columns, compares the test months' values with the training range and counts test rows
outside it. A tree cannot extrapolate: a value above the training maximum is treated like the largest training value,
so such a column acts as a (noisy) marker of the period rather than as information about rent.
Usage (repo root): python exploration/Aayush/expt-5/src/exp8_macro_range_shift.py
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # exploration/Aayush
FEAT = os.path.join(ROOT, 'expt-2', 'outputs', 'features')
OUT = os.path.join(ROOT, 'expt-5', 'outputs')
tr = pd.read_parquet(os.path.join(FEAT, 'train_features_v2.parquet'))
te = pd.read_parquet(os.path.join(FEAT, 'test_features_v2.parquet'))
cols = ['coe_a', 'coe_b', 'coe_c', 'coe_e', 'coe_overall_mean', 'stock_dbs_close', 'stock_ocbc_close', 'stock_uob_close', 'sg_financial_index']
lines = [f'{"column":20s}{"train min":>11s}{"train max":>11s}{"test min":>11s}{"test max":>11s}{"test rows outside train range":>32s}']
outside = []
for c in cols:
    lo, hi = tr[c].min(), tr[c].max()
    o = (te[c] < lo) | (te[c] > hi)
    outside.append(o.values)
    lines.append(f'{c:20s}{lo:11.1f}{hi:11.1f}{te[c].min():11.1f}{te[c].max():11.1f}{100 * o.mean():30.1f} %')
any_out = np.column_stack(outside).any(axis=1)
lines.append(f'\ntest rows with at least one macro column outside its training range: {100 * any_out.mean():.1f}%')
# the same count inside our two time holdouts: validation rows against the range of the rows before the cutoff
lines.append('\nthe same check inside the local validation windows (rows with at least one macro column outside the earlier rows\' range):')
for h, cutoff in [('P2', '2024-07'), ('P4', '2023-12')]:
    a_, b_ = tr[tr['RENT_APPROVAL_DATE'] < cutoff], tr[tr['RENT_APPROVAL_DATE'] >= cutoff]
    o = np.column_stack([((b_[c] < a_[c].min()) | (b_[c] > a_[c].max())).values for c in cols]).any(axis=1)
    lines.append(f'  {h} (validate from {cutoff}): {100 * o.mean():.1f}% of {len(b_):,} rows')
text = '\n'.join(lines)
print(text)
open(os.path.join(OUT, 'exp8_macro_range_shift.log'), 'w').write(text + '\n')
