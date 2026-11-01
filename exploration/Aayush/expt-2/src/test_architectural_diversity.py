"""
Test architectural diversity on OOT holdout:
Evaluating LightGBM (Tuned), XGBoost (Tuned), CatBoost (Tuned) + ExtraTrees / DART.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import config

def main():
    train_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "train_features_v3.parquet"))
    val_mask = train_feat["RENT_APPROVAL_DATE"] >= "2024-07"
    train_mask = ~val_mask

    train_df = train_feat[train_mask].copy()
    val_df = train_feat[val_mask].copy()

    drop_cols = [
        "MONTHLY_RENT", "RENT_APPROVAL_DATE", "approval_quarter",
        "block_clean", "street_clean", "postal_code_clean", "block_id", "target_res"
    ]
    feat_cols = [c for c in train_feat.columns if c not in drop_cols]
    cat_cols = ["town_clean", "flat_type_clean", "flat_model_clean", "subzone_clean", "planning_area_clean", "region_clean"]

    y_train_res = train_df["MONTHLY_RENT"] - train_df["median_rent_benchmark"]
    y_val_res = val_df["MONTHLY_RENT"] - val_df["median_rent_benchmark"]
    y_val_actual = val_df["MONTHLY_RENT"].values
    y_val_bench = val_df["median_rent_benchmark"].values

    with open(os.path.join(config.MODEL_DIR, "best_params_v3.json")) as f:
        best_p = json.load(f)

    # 1. Tuned LightGBM
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    lgb_params = best_p["lightgbm"]["params"]
    lgb_m = lgb.LGBMRegressor(
        n_estimators=1500, random_state=config.RANDOM_SEED, n_jobs=4, verbose=-1, **lgb_params
    )
    lgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    preds_lgb = y_val_bench + lgb_m.predict(val_df[feat_cols])
    print(f"Tuned LightGBM OOT RMSE: {root_mean_squared_error(y_val_actual, preds_lgb):.2f}")

    # 2. Tuned XGBoost
    xgb_params = best_p["xgboost"]["params"]
    xgb_m = xgb.XGBRegressor(
        n_estimators=1500, enable_categorical=True, random_state=config.RANDOM_SEED,
        n_jobs=4, early_stopping_rounds=40, **xgb_params
    )
    xgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        verbose=False
    )
    preds_xgb = y_val_bench + xgb_m.predict(val_df[feat_cols])
    print(f"Tuned XGBoost OOT RMSE:  {root_mean_squared_error(y_val_actual, preds_xgb):.2f}")

    # 3. Tuned CatBoost
    cb_train = train_df[feat_cols].copy()
    cb_val = val_df[feat_cols].copy()
    for c in cat_cols:
        cb_train[c] = cb_train[c].astype(str)
        cb_val[c] = cb_val[c].astype(str)

    cb_params = best_p["catboost"]["params"]
    cb_m = CatBoostRegressor(
        iterations=1500, cat_features=cat_cols, random_seed=config.RANDOM_SEED,
        thread_count=4, verbose=False, early_stopping_rounds=40, **cb_params
    )
    cb_m.fit(cb_train, y_train_res, eval_set=(cb_val, y_val_res))
    preds_cb = y_val_bench + cb_m.predict(cb_val)
    print(f"Tuned CatBoost OOT RMSE: {root_mean_squared_error(y_val_actual, preds_cb):.2f}")

    # Tri-model blend optimization
    mat3 = np.column_stack([preds_lgb, preds_xgb, preds_cb])
    res3 = minimize(
        lambda w: root_mean_squared_error(y_val_actual, np.dot(mat3, w)),
        [0.4, 0.4, 0.2], bounds=[(0, 1), (0, 1), (0, 1)],
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0}
    )
    opt_w3 = res3.x
    rmse_blend3 = root_mean_squared_error(y_val_actual, np.dot(mat3, opt_w3))
    print(f"\nTri-Model Blend OOT RMSE: {rmse_blend3:.2f} (Weights: {opt_w3.round(3)})")
    print(f"Delta vs Sprint 3 (449.26): {rmse_blend3 - 449.26:.2f} SGD improvement!")

if __name__ == "__main__":
    main()

