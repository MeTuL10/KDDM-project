"""
Residual models for the final pipeline: LightGBM, XGBoost and CatBoost predict rent minus the median-rent anchor,
the anchor is added back, and the three are blended.

1. fit_blend_weights: one out-of-time fit (train <= 2024-06, validate >= 2024-07), SLSQP on the blend RMSE,
   weights >= 0 summing to 1.
2. train_five_fold: shuffled 5-fold over all of train; each fold's three models predict the full test set and
   the test predictions are averaged over folds. Early stopping uses each fold's validation split.
Seeds, hyperparameters and thread counts are those of the submitted run (ported from expt-2/src/models_v2.py).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import config

LGB_PARAMS = dict(n_estimators=1200, learning_rate=0.04, num_leaves=75, subsample=0.8, colsample_bytree=0.75, n_jobs=4)
XGB_PARAMS = dict(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.8, colsample_bytree=0.75,
                  enable_categorical=True, n_jobs=4, early_stopping_rounds=40)
CB_PARAMS = dict(iterations=1200, learning_rate=0.05, depth=7, thread_count=4, verbose=False, early_stopping_rounds=40)
MODEL_NAMES = ['lgb', 'xgb', 'cb']


def get_feature_columns(df: pd.DataFrame):
    """Predictor columns and the categorical subset."""
    drop_cols = ["MONTHLY_RENT", "RENT_APPROVAL_DATE", "approval_quarter",
                 "block_clean", "street_clean", "postal_code_clean", "block_id", "target_res"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = ["town_clean", "flat_type_clean", "flat_model_clean", "subzone_clean", "planning_area_clean", "region_clean"]
    return feat_cols, cat_cols


def _as_category(df, cat_cols):
    d = df.copy()
    for c in cat_cols:
        d[c] = d[c].astype("category")
    return d


def _as_string(df, cat_cols):
    d = df.copy()
    for c in cat_cols:
        d[c] = d[c].astype(str)
    return d


def _fit_three(X_tr, y_tr, X_va, y_va, cb_tr, cb_va, cat_cols, seed):
    """Fit LightGBM, XGBoost and CatBoost on the residual target; return the three fitted models."""
    m_lgb = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
    m_xgb = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    m_cb = CatBoostRegressor(cat_features=cat_cols, random_seed=seed, **CB_PARAMS)
    m_cb.fit(cb_tr, y_tr, eval_set=(cb_va, y_va))
    return m_lgb, m_xgb, m_cb


def fit_blend_weights(train: pd.DataFrame):
    """Blend weights (lgb, xgb, cb) from the out-of-time fit. Returns (weights, dict of per-model and blend RMSE)."""
    val_mask = (train["RENT_APPROVAL_DATE"] >= config.OOT_CUTOFF_DATE).values
    tr, va = train[~val_mask], train[val_mask]
    feat_cols, cat_cols = get_feature_columns(train)
    y_tr = (tr["MONTHLY_RENT"] - tr["median_rent_benchmark"]).values
    y_va = (va["MONTHLY_RENT"] - va["median_rent_benchmark"]).values
    y_val_actual, y_val_bench = va["MONTHLY_RENT"].values, va["median_rent_benchmark"].values

    tr_c, va_c = _as_category(tr, cat_cols), _as_category(va, cat_cols)
    tr_s, va_s = _as_string(tr, cat_cols), _as_string(va, cat_cols)
    models = _fit_three(tr_c[feat_cols], y_tr, va_c[feat_cols], y_va, tr_s[feat_cols], va_s[feat_cols],
                        cat_cols, config.RANDOM_SEED)
    preds = [y_val_bench + models[0].predict(va_c[feat_cols]),
             y_val_bench + models[1].predict(va_c[feat_cols]),
             y_val_bench + models[2].predict(va_s[feat_cols])]
    pred_mat = np.column_stack(preds)
    res = minimize(lambda w: root_mean_squared_error(y_val_actual, pred_mat @ w), [0.5, 0.1, 0.4],
                   bounds=[(0, 1)] * 3, constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
    rmse = {n: root_mean_squared_error(y_val_actual, p) for n, p in zip(MODEL_NAMES, preds)}
    rmse['blend'] = root_mean_squared_error(y_val_actual, pred_mat @ res.x)
    return res.x, rmse


def train_five_fold(train: pd.DataFrame, test: pd.DataFrame, weights, verbose=True):
    """
    5-fold residual training. `test` must already carry the benchmark to be used (honest or published).
    Returns dict: oof (per model, rent scale), test_preds (per model), oof_rmse (per model + blend), blend (test).
    """
    feat_cols, cat_cols = get_feature_columns(train)
    y_actual = train["MONTHLY_RENT"].values
    y_bench = train["median_rent_benchmark"].values
    y_res = y_actual - y_bench
    test_bench = test["median_rent_benchmark"].values

    d_lgb, d_te_lgb = _as_category(train, cat_cols), _as_category(test, cat_cols)
    d_cb, d_te_cb = _as_string(train, cat_cols), _as_string(test, cat_cols)

    oof = {k: np.zeros(len(train)) for k in MODEL_NAMES}
    tst = {k: np.zeros(len(test)) for k in MODEL_NAMES}
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        X_tr, X_va = d_lgb.iloc[tr_idx][feat_cols], d_lgb.iloc[va_idx][feat_cols]
        m_lgb, m_xgb, m_cb = _fit_three(X_tr, y_res[tr_idx], X_va, y_res[va_idx],
                                        d_cb.iloc[tr_idx][feat_cols], d_cb.iloc[va_idx][feat_cols],
                                        cat_cols, config.RANDOM_SEED + fold)
        va_bench = y_bench[va_idx]
        oof['lgb'][va_idx] = va_bench + m_lgb.predict(X_va)
        oof['xgb'][va_idx] = va_bench + m_xgb.predict(X_va)
        oof['cb'][va_idx] = va_bench + m_cb.predict(d_cb.iloc[va_idx][feat_cols])
        tst['lgb'] += (test_bench + m_lgb.predict(d_te_lgb[feat_cols])) / config.N_FOLDS
        tst['xgb'] += (test_bench + m_xgb.predict(d_te_lgb[feat_cols])) / config.N_FOLDS
        tst['cb'] += (test_bench + m_cb.predict(d_te_cb[feat_cols])) / config.N_FOLDS
        if verbose:
            print(f"fold {fold}/{config.N_FOLDS}: " + "  ".join(
                f"{k} {root_mean_squared_error(y_actual[va_idx], oof[k][va_idx]):.2f}" for k in MODEL_NAMES), flush=True)

    w = np.asarray(weights)
    oof_rmse = {k: root_mean_squared_error(y_actual, oof[k]) for k in MODEL_NAMES}
    oof_rmse['blend'] = root_mean_squared_error(y_actual, sum(w[i] * oof[k] for i, k in enumerate(MODEL_NAMES)))
    blend = sum(w[i] * tst[k] for i, k in enumerate(MODEL_NAMES))
    return dict(oof=oof, test_preds=tst, oof_rmse=oof_rmse, blend=blend)
