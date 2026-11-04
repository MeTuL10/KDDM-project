"""
Median-rent anchor experiment (RMSE bottleneck report, roadmap item 1).

Three variants of the shipping LightGBM, each scored on two holdouts:
  A  current model (address feature, no anchor)
  B  A + anchor and anchor_level as features, direct rent target
  C  A + anchor features, residual target (rent - anchor); prediction = anchor + model

Holdouts: the usual 6-month time_based_split, and a 16-month horizon-matched split
(train through 2023-11, validate 2023-12 to 2025-03) that mimics the test set's distance
from training. Every encoder (address feature, anchor town ratios) is fit on that
holdout's training rows only.

Usage: python exploration/Aayush/median_anchor_experiment.py   (from the repo root)
"""
import sys

sys.path.insert(0, 'exploration/Aayush/archived_modules')

import pandas as pd
import evaluation as ev
import modeling as mdl
import preprocessing as prep
import split as spl

BEST_PARAMS = {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 15,
               'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.6,
               'reg_alpha': 0.1, 'reg_lambda': 0.1}
BASE_FEATS = mdl.GBM_NUMERIC_FEATURES + ['addr_level_feat']
ANCHOR_FEATS = BASE_FEATS + ['anchor', 'anchor_level']

train_feat, _ = prep.load_and_build(data_dir='data')
town_table, cross_table = prep.build_median_anchor_table(pd.read_csv('data/auxiliary/sg-hdb-median-rent.csv'))
categories = mdl.get_categories(train_feat)


def prepare(train_df, val_df):
    mt = prep.build_address_momentum(train_df)
    sz, gm = prep.rent_per_sqm_priors(train_df)
    tr = prep.add_address_feature(train_df, mt, sz, gm)
    va = prep.add_address_feature(val_df, mt, sz, gm)
    ratios = prep.fit_anchor_town_ratios(train_df, cross_table)
    fallback = train_df['MONTHLY_RENT'].mean()
    tr = prep.add_median_rent_anchor(tr, town_table, cross_table, ratios, fallback)
    va = prep.add_median_rent_anchor(va, town_table, cross_table, ratios, fallback)
    return tr, va


def fit_predict(tr, va, feats, residual=False):
    if residual:
        tr = tr.assign(MONTHLY_RENT=tr['MONTHLY_RENT'] - tr['anchor'])
        va_fit = va.assign(MONTHLY_RENT=va['MONTHLY_RENT'] - va['anchor'])
    else:
        va_fit = va
    model, _, X = mdl.fit_lgbm_baseline(tr, va_fit, categories, numeric_features=feats, **BEST_PARAMS)
    preds = pd.Series(model.predict(X), index=va.index)
    return preds + va['anchor'] if residual else preds


def run(label, train_df, val_df, cutoff):
    tr, va = prepare(train_df, val_df)
    print(f'\n===== {label}: train {len(tr):,} rows, validate {len(va):,} rows =====')
    print('anchor level share, train:', tr['anchor_level'].value_counts(normalize=True).sort_index().round(3).to_dict())
    print('anchor level share, val:  ', va['anchor_level'].value_counts(normalize=True).sort_index().round(3).to_dict())
    preds = {'A current': fit_predict(tr, va, BASE_FEATS),
             'B anchor features': fit_predict(tr, va, ANCHOR_FEATS),
             'C residual target': fit_predict(tr, va, ANCHOR_FEATS, residual=True)}
    y = va['MONTHLY_RENT']
    groups = (va['TOWN'] + '|' + va['BLOCK'] + '|' + va['STREET'] + '|' + va['RENT_APPROVAL_DATE']).values
    rows = []
    for name, p in preds.items():
        low = y <= 2100
        rows.append({'variant': name, 'RMSE': ev.rmse(y, p), 'MAE': float((y - p).abs().mean()),
                     'RMSE (rent<=2100)': ev.rmse(y[low], p[low]) if low.any() else float('nan'),
                     'RMSE (rent>2100)': ev.rmse(y[~low], p[~low])})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    for name in ['B anchor features', 'C residual target']:
        r = ev.paired_bootstrap_rmse_diff(y, preds['A current'], preds[name], groups)
        print(f'{name} minus A: diff {r["diff"]:+.2f}  95% CI [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}]'
              f'  share of resamples better {r["share_b_better"]:.2f}')
    by_month = pd.DataFrame({n: ev.rmse_by_month_ahead(va, p, cutoff) for n, p in preds.items()})
    print('\nRMSE by months ahead of the cutoff:')
    print(by_month.round(1).to_string())


train_fold, val_fold, cutoff1 = spl.time_based_split(train_feat)
run('6-month fold', train_fold, val_fold, cutoff1)
long_train, long_val, cutoff2 = spl.time_split_at(train_feat, '2023-12')
run('16-month horizon holdout', long_train, long_val, cutoff2)
