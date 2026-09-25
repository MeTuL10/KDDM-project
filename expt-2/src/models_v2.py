"""
Advanced Modeling & Stacking Pipeline v2
Path: expt-2/src/models_v2.py

Key Capabilities:
1. Out-of-Time (OOT) Holdout Validation Benchmark.
2. Full 5-Fold Cross Validation across all 150,000 transactions for:
   - LightGBM Regressor (Histogram-based GBDT)
   - XGBoost Regressor (Depth-wise Regularized GBDT)
   - CatBoost Regressor (Symmetric Oblivious Decision Trees)
3. Constrained Non-Negative Least Squares (NNLS) optimal stacking blending.
4. Bagged 5-Fold Test Set Inference & Schema Verification.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import config

def get_feature_columns(df: pd.DataFrame):
    """Isolate predictor columns and categorize them."""
    drop_cols = [
        "MONTHLY_RENT", "RENT_APPROVAL_DATE", "approval_quarter",
        "block_clean", "street_clean", "postal_code_clean", "block_id", "target_res"
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = ["town_clean", "flat_type_clean", "flat_model_clean", "subzone_clean", "planning_area_clean", "region_clean"]
    return feat_cols, cat_cols

def evaluate_oot_benchmark_v2(train_features: pd.DataFrame):
    """
    Evaluates OOT holdout benchmark (Train: <= 2024-06, Val: 2024-07 to 2025-03).
    """
    print("=" * 80)
    print("PHASE 5: OUT-OF-TIME (OOT) BENCHMARK EVALUATION (V2)")
    print("=" * 80)

    val_mask = train_features["RENT_APPROVAL_DATE"] >= "2024-07"
    train_mask = ~val_mask

    train_df = train_features[train_mask].copy()
    val_df = train_features[val_mask].copy()

    y_val_actual = val_df["MONTHLY_RENT"].values
    y_val_bench = val_df["median_rent_benchmark"].values

    base_rmse = root_mean_squared_error(y_val_actual, y_val_bench)
    base_mae = mean_absolute_error(y_val_actual, y_val_bench)
    print(f"[Baseline] Raw Median Rent Benchmark -> RMSE: {base_rmse:.2f}, MAE: {base_mae:.2f}\n")

    feat_cols, cat_cols = get_feature_columns(train_features)
    y_train_res = train_df["MONTHLY_RENT"] - train_df["median_rent_benchmark"]
    y_val_res = val_df["MONTHLY_RENT"] - val_df["median_rent_benchmark"]

    # 1. LightGBM
    print("[Model 1/3] Training LightGBM Residual Model on OOT...")
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    lgb_m = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.04, num_leaves=75,
        subsample=0.8, colsample_bytree=0.75, random_state=config.RANDOM_SEED, n_jobs=4
    )
    lgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    preds_lgb = y_val_bench + lgb_m.predict(val_df[feat_cols])
    rmse_lgb = root_mean_squared_error(y_val_actual, preds_lgb)
    print(f"   -> LightGBM OOT RMSE: {rmse_lgb:.2f}")

    # 2. XGBoost
    print("[Model 2/3] Training XGBoost Residual Model on OOT...")
    xgb_m = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.04, max_depth=7,
        subsample=0.8, colsample_bytree=0.75, enable_categorical=True,
        random_state=config.RANDOM_SEED, n_jobs=4, early_stopping_rounds=40
    )
    xgb_m.fit(
        train_df[feat_cols], y_train_res,
        eval_set=[(val_df[feat_cols], y_val_res)],
        verbose=False
    )
    preds_xgb = y_val_bench + xgb_m.predict(val_df[feat_cols])
    rmse_xgb = root_mean_squared_error(y_val_actual, preds_xgb)
    print(f"   -> XGBoost OOT RMSE:  {rmse_xgb:.2f}")

    # 3. CatBoost
    print("[Model 3/3] Training CatBoost Residual Model on OOT...")
    cb_train = train_df[feat_cols].copy()
    cb_val = val_df[feat_cols].copy()
    for c in cat_cols:
        cb_train[c] = cb_train[c].astype(str)
        cb_val[c] = cb_val[c].astype(str)

    cb_m = CatBoostRegressor(
        iterations=1200, learning_rate=0.05, depth=7,
        cat_features=cat_cols, random_seed=config.RANDOM_SEED,
        thread_count=4, verbose=False, early_stopping_rounds=40
    )
    cb_m.fit(cb_train, y_train_res, eval_set=(cb_val, y_val_res))
    preds_cb = y_val_bench + cb_m.predict(cb_val)
    rmse_cb = root_mean_squared_error(y_val_actual, preds_cb)
    print(f"   -> CatBoost OOT RMSE: {rmse_cb:.2f}")

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
    mae_blend = mean_absolute_error(y_val_actual, preds_blend)
    r2_blend = r2_score(y_val_actual, preds_blend)

    print(f"\n[Optimal Ensemble Blend] Weights: {opt_w.round(3)}")
    print(f"   -> Blended OOT RMSE: {rmse_blend:.2f}, MAE: {mae_blend:.2f}, R2: {r2_blend:.4f}")
    return opt_w

def train_and_predict_5fold_v2(train_features: pd.DataFrame, test_features: pd.DataFrame, opt_weights: np.ndarray):
    """
    Runs full 5-Fold Cross Validation and generates bagged test predictions.
    """
    print("\n" + "=" * 80)
    print("PHASE 5 & 6: 5-FOLD ENSEMBLE TRAINING & TEST INFERENCE (V2)")
    print("=" * 80)

    feat_cols, cat_cols = get_feature_columns(train_features)
    y_actual = train_features["MONTHLY_RENT"].values
    y_bench = train_features["median_rent_benchmark"].values
    y_res = y_actual - y_bench

    test_bench = test_features["median_rent_benchmark"].values

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    oof_preds_lgb = np.zeros(len(train_features))
    oof_preds_xgb = np.zeros(len(train_features))
    oof_preds_cb = np.zeros(len(train_features))

    test_preds_lgb = np.zeros(len(test_features))
    test_preds_xgb = np.zeros(len(test_features))
    test_preds_cb = np.zeros(len(test_features))

    # Pre-format categoricals
    df_train_lgb = train_features.copy()
    df_test_lgb = test_features.copy()
    for c in cat_cols:
        df_train_lgb[c] = df_train_lgb[c].astype("category")
        df_test_lgb[c] = df_test_lgb[c].astype("category")

    df_train_cb = train_features.copy()
    df_test_cb = test_features.copy()
    for c in cat_cols:
        df_train_cb[c] = df_train_cb[c].astype(str)
        df_test_cb[c] = df_test_cb[c].astype(str)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_features), 1):
        print(f"\n--- FOLD {fold}/{config.N_FOLDS} ---")
        X_tr, y_tr_res = df_train_lgb.iloc[tr_idx][feat_cols], y_res[tr_idx]
        X_va, y_va_res = df_train_lgb.iloc[va_idx][feat_cols], y_res[va_idx]
        va_bench = y_bench[va_idx]
        va_actual = y_actual[va_idx]

        # 1. LightGBM
        lgb_m = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.04, num_leaves=75,
            subsample=0.8, colsample_bytree=0.75, random_state=config.RANDOM_SEED + fold, n_jobs=4
        )
        lgb_m.fit(X_tr, y_tr_res, eval_set=[(X_va, y_va_res)], callbacks=[lgb.early_stopping(40, verbose=False)])
        f_pred_lgb = va_bench + lgb_m.predict(X_va)
        oof_preds_lgb[va_idx] = f_pred_lgb
        test_preds_lgb += (test_bench + lgb_m.predict(df_test_lgb[feat_cols])) / config.N_FOLDS

        # 2. XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=1200, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.75, enable_categorical=True,
            random_state=config.RANDOM_SEED + fold, n_jobs=4, early_stopping_rounds=40
        )
        xgb_m.fit(X_tr, y_tr_res, eval_set=[(X_va, y_va_res)], verbose=False)
        f_pred_xgb = va_bench + xgb_m.predict(X_va)
        oof_preds_xgb[va_idx] = f_pred_xgb
        test_preds_xgb += (test_bench + xgb_m.predict(df_test_lgb[feat_cols])) / config.N_FOLDS

        # 3. CatBoost
        cb_X_tr = df_train_cb.iloc[tr_idx][feat_cols]
        cb_X_va = df_train_cb.iloc[va_idx][feat_cols]
        cb_m = CatBoostRegressor(
            iterations=1200, learning_rate=0.05, depth=7,
            cat_features=cat_cols, random_seed=config.RANDOM_SEED + fold,
            thread_count=4, verbose=False, early_stopping_rounds=40
        )
        cb_m.fit(cb_X_tr, y_tr_res, eval_set=(cb_X_va, y_va_res))
        f_pred_cb = va_bench + cb_m.predict(cb_X_va)
        oof_preds_cb[va_idx] = f_pred_cb
        test_preds_cb += (test_bench + cb_m.predict(df_test_cb[feat_cols])) / config.N_FOLDS

        f_blend = opt_weights[0] * f_pred_lgb + opt_weights[1] * f_pred_xgb + opt_weights[2] * f_pred_cb
        print(f"   Fold {fold} RMSE -> LGB: {root_mean_squared_error(va_actual, f_pred_lgb):.2f} | "
              f"XGB: {root_mean_squared_error(va_actual, f_pred_xgb):.2f} | "
              f"CB: {root_mean_squared_error(va_actual, f_pred_cb):.2f} | "
              f"BLEND: {root_mean_squared_error(va_actual, f_blend):.2f}")

    # Overall OOF Metrics
    oof_blend = opt_weights[0] * oof_preds_lgb + opt_weights[1] * oof_preds_xgb + opt_weights[2] * oof_preds_cb
    oof_rmse_lgb = root_mean_squared_error(y_actual, oof_preds_lgb)
    oof_rmse_xgb = root_mean_squared_error(y_actual, oof_preds_xgb)
    oof_rmse_cb = root_mean_squared_error(y_actual, oof_preds_cb)
    oof_rmse_blend = root_mean_squared_error(y_actual, oof_blend)
    oof_mae_blend = mean_absolute_error(y_actual, oof_blend)
    oof_r2_blend = r2_score(y_actual, oof_blend)

    print("\n" + "=" * 80)
    print("OVERALL 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE (V2)")
    print("=" * 80)
    print(f"Overall OOF LightGBM RMSE: {oof_rmse_lgb:.2f}")
    print(f"Overall OOF XGBoost  RMSE: {oof_rmse_xgb:.2f}")
    print(f"Overall OOF CatBoost RMSE: {oof_rmse_cb:.2f}")
    print(f"OVERALL OOF ENSEMBLE BLEND RMSE: {oof_rmse_blend:.2f} (MAE: {oof_mae_blend:.2f}, R2: {oof_r2_blend:.4f})")

    # Final Test Set Predictions
    final_test_preds = (
        opt_weights[0] * test_preds_lgb +
        opt_weights[1] * test_preds_xgb +
        opt_weights[2] * test_preds_cb
    )

    # Save Submission
    sub_df = pd.DataFrame({
        "Id": np.arange(len(final_test_preds)),
        "Predicted": final_test_preds
    })

    sub_path = os.path.join(config.SUBMISSION_DIR, "submission_ensemble_v2.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission generated and saved: {sub_path} ({len(sub_df)} rows)")

    # Also save to root submission.csv for easy upload
    root_sub_path = os.path.join(config.PROJECT_ROOT, "submission.csv")
    sub_df.to_csv(root_sub_path, index=False)
    print(f"Root submission updated: {root_sub_path}")

    # Save metrics
    metrics = {
        "oof_rmse_lgb": float(oof_rmse_lgb),
        "oof_rmse_xgb": float(oof_rmse_xgb),
        "oof_rmse_cb": float(oof_rmse_cb),
        "oof_rmse_blend": float(oof_rmse_blend),
        "oof_mae_blend": float(oof_mae_blend),
        "oof_r2_blend": float(oof_r2_blend),
        "weights": [float(w) for w in opt_weights]
    }
    metrics_path = os.path.join(config.MODEL_DIR, "cv_metrics_v2.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

def run_models_pipeline_v2():
    train_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "train_features_v2.parquet"))
    test_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "test_features_v2.parquet"))

    opt_weights = evaluate_oot_benchmark_v2(train_feat)
    train_and_predict_5fold_v2(train_feat, test_feat, opt_weights)

if __name__ == "__main__":
    run_models_pipeline_v2()

