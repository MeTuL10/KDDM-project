"""
Shared setup for expt-5 (library; see v2_clean_experiments.py for the first experiment).

expt-5: improve on expt-2 V2 (public leaderboard 433.607).

V2 recipe: official median-rent benchmark per (town, flat type, quarter), residual target
(rent - benchmark), 70 features incl. hierarchical Bayesian target encodings
(town/block/block x type/street/subzone), LightGBM + XGBoost + CatBoost, 5-fold bagged.

This script rebuilds V2's feature recipe LEAK-FREE on two time holdouts (target encodings
recomputed inside each training split) and tests one improvement at a time against it.

  base       V2 recipe, target encodings nested inside the split
  anchor     base + fallback for empty median cells = cross-town median x town ratio
  trailing   base + time-aware address premium (trailing mean of rent - anchor per block)
  both       anchor + trailing
  reg        base with more regularised LightGBM (active bagging, min_child_samples, L2)

Holdouts: H1 train <= 2024-06 / validate 2024-07..2025-03 (expt-2's OOT split);
H2 train <= 2023-11 / validate 2023-12..2025-03 (16 months, the test set's distance).
Usage (repo root): python exploration/Aayush/expt-5/src/v2_clean_experiments.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # exploration/Aayush
REPO = os.path.dirname(os.path.dirname(ROOT))  # repository root (data/, archived_modules/)
sys.path.insert(0, os.path.join(ROOT, 'expt-2', 'src'))
sys.path.insert(0, os.path.join(REPO, 'archived_modules'))

import numpy as np
import pandas as pd
import lightgbm as lgb
import feature_engineering_v2 as fe2
import preprocessing as prep
import evaluation as ev

FEAT = os.path.join(ROOT, 'expt-2', 'outputs', 'features', 'train_features_v2.parquet')
MEDIAN = os.path.join(REPO, 'data', 'auxiliary', 'sg-hdb-median-rent.csv')
CAT = ['town_clean', 'flat_type_clean', 'flat_model_clean', 'subzone_clean', 'planning_area_clean', 'region_clean']
DROP = ['MONTHLY_RENT', 'RENT_APPROVAL_DATE', 'approval_quarter', 'block_clean', 'street_clean',
        'postal_code_clean', 'block_id', 'block_ft_id', 'target_res']
TE_COLS = ['te_town', 'te_block', 'te_bft', 'te_street', 'te_subzone']

V2_PARAMS = dict(learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75,
                 random_state=42, n_jobs=-1, verbose=-1)
REG_PARAMS = dict(learning_rate=0.04, num_leaves=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.75, min_child_samples=50, reg_lambda=5.0,
                  random_state=42, n_jobs=-1, verbose=-1)

full = pd.read_parquet(FEAT).drop(columns=TE_COLS)      # static, leaky TEs removed
full['postal_code_clean'] = full['postal_code_clean'].astype(str)
cat_levels = {c: sorted(full[c].astype(str).unique()) for c in CAT}
town_table, cross_table = prep.build_median_anchor_table(pd.read_csv(MEDIAN))


def swap_anchor(tr, va):
    """Replace V2's benchmark by anchor with town-ratio fallback (fit on tr rents only)."""
    def view(d): return d.assign(TOWN=d['town_clean'], FLAT_TYPE=d['flat_type_clean'])
    ratios = prep.fit_anchor_town_ratios(view(tr), cross_table)
    fb = tr['MONTHLY_RENT'].mean()
    out = []
    for d in (tr, va):
        a = prep.add_median_rent_anchor(view(d), town_table, cross_table, ratios, fb)
        d = d.copy()
        d['median_rent_benchmark'] = a['anchor'].values
        d['median_rent_ratio'] = d['median_rent_benchmark'] / d['nat_median_rent_benchmark']
        out.append(d)
    return out


def add_trailing(tr, va):
    """Trailing (prior-months-only) mean of rent - benchmark per block, shrunk to the
    subzone mean; train rows see only earlier months, validation rows see the state frozen
    at the training cutoff (the same situation test.csv is in)."""
    def view(d):
        return pd.DataFrame({'TOWN': d['town_clean'], 'BLOCK': d['block_clean'], 'STREET': d['street_clean'],
                             'SUBZONE': d['subzone_clean'], 'RENT_APPROVAL_DATE': d['RENT_APPROVAL_DATE'],
                             'MONTHLY_RENT': (d['MONTHLY_RENT'] if 'MONTHLY_RENT' in d else 0.0) - d['median_rent_benchmark'],
                             'FLOOR_AREA_SQM': 1.0}, index=d.index)
    vt, vv = view(tr), view(va)
    mt = prep.build_address_momentum(vt)
    sz, gm = prep.rent_per_sqm_priors(vt)
    tr, va = tr.copy(), va.copy()
    tr['addr_resid_trailing'] = prep.add_address_feature(vt, mt, sz, gm)['addr_level_feat'].values
    va['addr_resid_trailing'] = prep.add_address_feature(vv, mt, sz, gm)['addr_level_feat'].values
    return tr, va


def build(cutoff, anchor=False, trailing=False):
    tr = full[full['RENT_APPROVAL_DATE'] < cutoff].copy()
    va = full[full['RENT_APPROVAL_DATE'] >= cutoff].copy()
    if anchor:
        tr, va = swap_anchor(tr, va)
    if trailing:
        tr, va = add_trailing(tr, va)
    tr['target_res'] = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    oof, maps, gm = fe2.compute_hierarchical_te(tr, 'target_res')       # OOF inside the split
    te_va = fe2.apply_hierarchical_te(va, maps, gm)
    tr = pd.concat([tr, oof], axis=1)
    va = pd.concat([va, te_va], axis=1)
    return tr, va


def matrix(d, cols):
    X = d[cols].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
    return X


def fit_predict(tr, va, params, n_estimators, early_stop=False):
    cols = [c for c in tr.columns if c not in DROP]
    Xtr, Xva = matrix(tr, cols), matrix(va, cols)
    ytr = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    yva = va['MONTHLY_RENT'] - va['median_rent_benchmark']
    m = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
    if early_stop:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return pd.Series(m.predict(Xva) + va['median_rent_benchmark'].values, index=va.index), m




# --- smoothed anchor: quarterly cell medians from small cells are noisy, and that noise
# sits inside the residual target. Centered 3-quarter mean within (town, flat type).
_st = town_table.sort_values(['TOWN', 'FLAT_TYPE', 'quarter']).copy()
_st['anchor_smooth'] = _st.groupby(['TOWN', 'FLAT_TYPE'])['anchor_town'].transform(
    lambda s: s.rolling(3, center=True, min_periods=1).mean())
SMOOTH = _st.set_index(['TOWN', 'FLAT_TYPE', 'quarter'])['anchor_smooth']
MAXQ = cross_table['quarter'].max()


def smooth_values(d):
    q = d['approval_quarter'].astype(str)
    q = q.where(q <= MAXQ, MAXQ)
    idx = pd.MultiIndex.from_arrays([d['town_clean'], d['flat_type_clean'], q])
    return pd.Series(SMOOTH.reindex(idx).values, index=d.index).fillna(d['median_rent_benchmark'])


def add_smooth(tr, va, as_target=False):
    out = []
    for d in (tr, va):
        d = d.copy()
        sm = smooth_values(d)
        if as_target:
            d['bench_raw'] = d['median_rent_benchmark']
            d['median_rent_benchmark'] = sm
            d['median_rent_ratio'] = d['median_rent_benchmark'] / d['nat_median_rent_benchmark']
        else:
            d['bench_smooth'] = sm
            d['bench_smooth_gap'] = sm - d['median_rent_benchmark']
        out.append(d)
    return out
