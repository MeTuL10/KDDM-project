"""
Scratch evaluation script for v3 features on Out-of-Time Holdout.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import config

def main():
    train_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "train_features_v3.parquet"))
    val_mask = train_feat["RENT_APPROVAL_DATE"] >= "2024-07"
    train_mask = ~val_mask

    train_df = train_feat[train_mask].copy()
    val_df = train_feat[val_mask].copy()

    y_val_actual = val_df["MONTHLY_RENT"].values
    y_val_bench = val_df["median_rent_benchmark"].values

    drop_cols = [
        "MONTHLY_RENT", "RENT_APPROVAL_DATE", "approval_quarter",
        "block_clean", "street_clean", "postal_code_clean", "block_id", "target_res"
    ]
    feat_cols = [c for c in train_feat.columns if c not in drop_cols]
    cat_cols = ["town_clean", "flat_type_clean", "flat_model_clean", "subzone_clean", "planning_area_clean", "region_clean"]

    y_train_res = train_df["MONTHLY_RENT"] - train_df["median_rent_benchmark"]
    y_val_res = val_df["MONTHLY_RENT"] - val_df["median_rent_benchmark"]

    print(f"Number of predictor features in v3: {len(feat_cols)}")
    print("New v3 features present:", [c for c in feat_cols if "flat_model" in c or "cohort" in c or "area" in c or "luxury" in c or "skyrise" in c or "tier" in c])

    # 1. LightGBM
    print("\n--- Training LightGBM on v3 OOT ---")
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    lgb_m = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.035, num_leaves=95,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0,
        random_state=config.RANDOM_SEED, n_jobs=4
    )
    lgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    preds_lgb = y_val_bench + lgb_m.predict(val_df[feat_cols])
    rmse_lgb = root_mean_squared_error(y_val_actual, preds_lgb)
    print(f"   -> LightGBM v3 OOT RMSE: {rmse_lgb:.2f} (v2 was 450.31)")

    # 2. XGBoost
    print("\n--- Training XGBoost on v3 OOT ---")
    xgb_m = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.035, max_depth=8,
        subsample=0.8, colsample_bytree=0.70, reg_alpha=1.0, reg_lambda=3.0,
        enable_categorical=True, random_state=config.RANDOM_SEED, n_jobs=4, early_stopping_rounds=40
    )
    xgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        verbose=False
    )
    preds_xgb = y_val_bench + xgb_m.predict(val_df[feat_cols])
    rmse_xgb = root_mean_squared_error(y_val_actual, preds_xgb)
    print(f"   -> XGBoost v3 OOT RMSE:  {rmse_xgb:.2f} (v2 was 451.09)")

    # 3. CatBoost
    print("\n--- Training CatBoost on v3 OOT ---")
    cb_train = train_df[feat_cols].copy()
    cb_val = val_df[feat_cols].copy()
    for c in cat_cols:
        cb_train[c] = cb_train[c].astype(str)
        cb_val[c] = cb_val[c].astype(str)

    cb_m = CatBoostRegressor(
        iterations=1200, learning_rate=0.05, depth=7, l2_leaf_reg=5.0,
        cat_features=cat_cols, random_seed=config.RANDOM_SEED,
        thread_count=4, verbose=False, early_stopping_rounds=40
    )
    cb_m.fit(cb_train, y_train_res, eval_set=(cb_val, y_val_res))
    preds_cb = y_val_bench + cb_m.predict(cb_val)
    rmse_cb = root_mean_squared_error(y_val_actual, preds_cb)
    print(f"   -> CatBoost v3 OOT RMSE: {rmse_cb:.2f} (v2 was 451.32)")

    # Optimal Blending
    pred_mat = np.column_stack([preds_lgb, preds_xgb, preds_cb])
    res = minimize(
        lambda w: root_mean_squared_error(y_val_actual, np.dot(pred_mat, w)),
        [0.5, 0.1, 0.4], bounds=[(0, 1), (0, 1), (0, 1)],
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0}
    )
    opt_w = res.x
    preds_blend = np.dot(pred_mat, opt_w)
    rmse_blend = root_mean_squared_error(y_val_actual, preds_blend)
    print(f"\n[Optimal Ensemble Blend v3] Weights: {opt_w.round(3)}")
    print(f"   -> Blended v3 OOT RMSE: {rmse_blend:.2f} (v2 was 449.26) | Delta: {rmse_blend - 449.26:.2f}")

if __name__ == "__main__":
    main()

