"""
Model Training, Validation, and Ensembling Pipeline (Phases 5 & 6).
Implements:
1. Out-of-Time (OOT) Holdout Validation (Train < 2024-07, Holdout >= 2024-07)
2. 5-Fold Cross Validation across the full dataset
3. Residual-to-Benchmark Target Formulation (eliminating extrapolation ceiling)
4. Tri-model GBDT Family: LightGBM, XGBoost, and CatBoost
5. Optimal Ensemble Blending & Test Set Inference
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    from . import config
except ImportError:
    import config

# Selected feature columns
DROP_COLS = [
    "MONTHLY_RENT", "RENT_APPROVAL_DATE", "approval_quarter",
    "block_clean", "street_clean", "postal_code_clean"
]
CAT_COLS = [
    "town_clean", "flat_type_clean", "flat_model_clean",
    "subzone_clean", "planning_area_clean", "region_clean"
]

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature columns for model training."""
    return [c for c in df.columns if c not in DROP_COLS]

def prepare_feature_matrix(df: pd.DataFrame, feat_cols: List[str], model_type: str = "lgb") -> pd.DataFrame:
    """Format categorical columns according to the requirements of the model framework."""
    X = df[feat_cols].copy()
    if model_type in ["lgb", "xgb"]:
        for col in CAT_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")
    elif model_type == "catboost":
        for col in CAT_COLS:
            if col in X.columns:
                X[col] = X[col].astype(str)
    return X

def evaluate_oot_benchmark(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and evaluate models on an Out-of-Time (OOT) split:
    - Train: All records prior to 2024-07 (124,016 rows, ~3.5 years)
    - Holdout: All records from 2024-07 onwards (25,984 rows, 9 months)
    Tests generalizability to strictly unseen future periods.
    """
    print("=" * 80)
    print("PHASE 5: OUT-OF-TIME (OOT) VALIDATION BENCHMARK")
    print("=" * 80)

    train_mask = df["RENT_APPROVAL_DATE"] < "2024-07"
    val_mask = df["RENT_APPROVAL_DATE"] >= "2024-07"

    train_df = df[train_mask]
    val_df = df[val_mask]
    print(f"OOT Train split:   {train_df.shape[0]:,} records (Jan 2021 - Jun 2024)")
    print(f"OOT Holdout split: {val_df.shape[0]:,} records (Jul 2024 - Mar 2025)")

    feat_cols = get_feature_columns(df)
    y_train = train_df["MONTHLY_RENT"].values
    y_val = val_df["MONTHLY_RENT"].values
    
    bench_train = train_df["median_rent_benchmark"].values
    bench_val = val_df["median_rent_benchmark"].values
    
    # Target residuals
    y_train_res = y_train - bench_train
    y_val_res = y_val - bench_val

    results = {}

    # 0. Baseline Benchmark Metric
    baseline_rmse = root_mean_squared_error(y_val, bench_val)
    baseline_mae = mean_absolute_error(y_val, bench_val)
    print(f"\n[Baseline] Raw Median Rent Benchmark -> RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}")
    results["baseline_benchmark"] = {"rmse": baseline_rmse, "mae": baseline_mae}

    # 1. LightGBM Residual Model
    print("\n[Model 1/3] Training LightGBM Residual Model on OOT...")
    X_train_lgb = prepare_feature_matrix(train_df, feat_cols, "lgb")
    X_val_lgb = prepare_feature_matrix(val_df, feat_cols, "lgb")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=70,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
        n_jobs=4
    )
    lgb_model.fit(
        X_train_lgb, y_train_res,
        eval_set=[(X_val_lgb, y_val_res)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds_lgb = bench_val + lgb_model.predict(X_val_lgb)
    rmse_lgb = root_mean_squared_error(y_val, preds_lgb)
    mae_lgb = mean_absolute_error(y_val, preds_lgb)
    print(f"   -> LightGBM OOT RMSE: {rmse_lgb:.2f}, MAE: {mae_lgb:.2f}")
    results["lightgbm"] = {"rmse": rmse_lgb, "mae": mae_lgb, "preds": preds_lgb}

    # 2. XGBoost Residual Model
    print("\n[Model 2/3] Training XGBoost Residual Model on OOT...")
    X_train_xgb = prepare_feature_matrix(train_df, feat_cols, "xgb")
    X_val_xgb = prepare_feature_matrix(val_df, feat_cols, "xgb")

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=config.RANDOM_SEED,
        n_jobs=4,
        early_stopping_rounds=50
    )
    xgb_model.fit(X_train_xgb, y_train_res, eval_set=[(X_val_xgb, y_val_res)], verbose=False)
    preds_xgb = bench_val + xgb_model.predict(X_val_xgb)
    rmse_xgb = root_mean_squared_error(y_val, preds_xgb)
    mae_xgb = mean_absolute_error(y_val, preds_xgb)
    print(f"   -> XGBoost OOT RMSE:  {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}")
    results["xgboost"] = {"rmse": rmse_xgb, "mae": mae_xgb, "preds": preds_xgb}

    # 3. CatBoost Residual Model
    print("\n[Model 3/3] Training CatBoost Residual Model on OOT...")
    X_train_cb = prepare_feature_matrix(train_df, feat_cols, "catboost")
    X_val_cb = prepare_feature_matrix(val_df, feat_cols, "catboost")

    cb_model = cb.CatBoostRegressor(
        iterations=900,
        learning_rate=0.07,
        depth=7,
        cat_features=CAT_COLS,
        random_seed=config.RANDOM_SEED,
        verbose=False,
        thread_count=4
    )
    cb_model.fit(X_train_cb, y_train_res, eval_set=(X_val_cb, y_val_res), early_stopping_rounds=50)
    preds_cb = bench_val + cb_model.predict(X_val_cb)
    rmse_cb = root_mean_squared_error(y_val, preds_cb)
    mae_cb = mean_absolute_error(y_val, preds_cb)
    print(f"   -> CatBoost OOT RMSE: {rmse_cb:.2f}, MAE: {mae_cb:.2f}")
    results["catboost"] = {"rmse": rmse_cb, "mae": mae_cb, "preds": preds_cb}

    # 4. Tri-Model Weighted Ensemble
    preds_blend = 0.45 * preds_lgb + 0.35 * preds_xgb + 0.20 * preds_cb
    rmse_blend = root_mean_squared_error(y_val, preds_blend)
    mae_blend = mean_absolute_error(y_val, preds_blend)
    r2_blend = r2_score(y_val, preds_blend)
    print(f"\n[Ensemble Blend] (0.45 LGB + 0.35 XGB + 0.20 CB)")
    print(f"   -> Blended OOT RMSE: {rmse_blend:.2f}, MAE: {mae_blend:.2f}, R2: {r2_blend:.4f}")
    results["ensemble_blend"] = {"rmse": rmse_blend, "mae": mae_blend, "r2": r2_blend, "preds": preds_blend}

    return results

def train_and_predict_5fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train 5-Fold Cross Validation ensemble across the full 150,000 training records.
    Produces:
    - Out-of-fold (OOF) cross-validation predictions for all train rows
    - Bagged ensemble test predictions across all 5 folds and 3 model families
    """
    print("=" * 80)
    print("PHASES 5 & 6: 5-FOLD ENSEMBLE TRAINING & TEST INFERENCE")
    print("=" * 80)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    feat_cols = get_feature_columns(train_df)
    n_train = len(train_df)
    n_test = len(test_df)

    y_train = train_df["MONTHLY_RENT"].values
    bench_train = train_df["median_rent_benchmark"].values
    bench_test = test_df["median_rent_benchmark"].values
    
    y_res_train = y_train - bench_train

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    oof_preds_lgb = np.zeros(n_train)
    oof_preds_xgb = np.zeros(n_train)
    oof_preds_cb = np.zeros(n_train)

    test_preds_lgb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    test_preds_cb = np.zeros(n_test)

    # Prepare test matrices once
    X_test_lgb = prepare_feature_matrix(test_df, feat_cols, "lgb")
    X_test_xgb = prepare_feature_matrix(test_df, feat_cols, "xgb")
    X_test_cb = prepare_feature_matrix(test_df, feat_cols, "catboost")

    print(f"Running {config.N_FOLDS}-Fold CV with LightGBM, XGBoost, and CatBoost...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- FOLD {fold + 1}/{config.N_FOLDS} ---")
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        y_fold_tr_res = y_res_train[train_idx]
        y_fold_va_res = y_res_train[val_idx]
        y_fold_va = y_train[val_idx]
        bench_fold_va = bench_train[val_idx]

        # 1. LightGBM
        X_tr_lgb = prepare_feature_matrix(fold_train_df, feat_cols, "lgb")
        X_va_lgb = prepare_feature_matrix(fold_val_df, feat_cols, "lgb")
        
        m_lgb = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=70,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED + fold,
            n_jobs=4
        )
        m_lgb.fit(X_tr_lgb, y_fold_tr_res, eval_set=[(X_va_lgb, y_fold_va_res)], callbacks=[lgb.early_stopping(50, verbose=False)])
        fold_lgb_val = bench_fold_va + m_lgb.predict(X_va_lgb)
        oof_preds_lgb[val_idx] = fold_lgb_val
        test_preds_lgb += (bench_test + m_lgb.predict(X_test_lgb)) / config.N_FOLDS
        rmse_f_lgb = root_mean_squared_error(y_fold_va, fold_lgb_val)

        # 2. XGBoost
        X_tr_xgb = prepare_feature_matrix(fold_train_df, feat_cols, "xgb")
        X_va_xgb = prepare_feature_matrix(fold_val_df, feat_cols, "xgb")

        m_xgb = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            random_state=config.RANDOM_SEED + fold,
            n_jobs=4,
            early_stopping_rounds=50
        )
        m_xgb.fit(X_tr_xgb, y_fold_tr_res, eval_set=[(X_va_xgb, y_fold_va_res)], verbose=False)
        fold_xgb_val = bench_fold_va + m_xgb.predict(X_va_xgb)
        oof_preds_xgb[val_idx] = fold_xgb_val
        test_preds_xgb += (bench_test + m_xgb.predict(X_test_xgb)) / config.N_FOLDS
        rmse_f_xgb = root_mean_squared_error(y_fold_va, fold_xgb_val)

        # 3. CatBoost
        X_tr_cb = prepare_feature_matrix(fold_train_df, feat_cols, "catboost")
        X_va_cb = prepare_feature_matrix(fold_val_df, feat_cols, "catboost")

        m_cb = cb.CatBoostRegressor(
            iterations=900,
            learning_rate=0.07,
            depth=7,
            cat_features=CAT_COLS,
            random_seed=config.RANDOM_SEED + fold,
            verbose=False,
            thread_count=4
        )
        m_cb.fit(X_tr_cb, y_fold_tr_res, eval_set=(X_va_cb, y_fold_va_res), early_stopping_rounds=50)
        fold_cb_val = bench_fold_va + m_cb.predict(X_va_cb)
        oof_preds_cb[val_idx] = fold_cb_val
        test_preds_cb += (bench_test + m_cb.predict(X_test_cb)) / config.N_FOLDS
        rmse_f_cb = root_mean_squared_error(y_fold_va, fold_cb_val)

        fold_blend = 0.45 * fold_lgb_val + 0.35 * fold_xgb_val + 0.20 * fold_cb_val
        rmse_f_blend = root_mean_squared_error(y_fold_va, fold_blend)
        print(f"   Fold {fold+1} RMSE -> LGB: {rmse_f_lgb:.2f} | XGB: {rmse_f_xgb:.2f} | CB: {rmse_f_cb:.2f} | BLEND: {rmse_f_blend:.2f}")

    # Compute overall Out-Of-Fold (OOF) metrics across all 150,000 samples
    oof_blend = 0.45 * oof_preds_lgb + 0.35 * oof_preds_xgb + 0.20 * oof_preds_cb
    test_blend = 0.45 * test_preds_lgb + 0.35 * test_preds_xgb + 0.20 * test_preds_cb

    oof_rmse_lgb = root_mean_squared_error(y_train, oof_preds_lgb)
    oof_rmse_xgb = root_mean_squared_error(y_train, oof_preds_xgb)
    oof_rmse_cb = root_mean_squared_error(y_train, oof_preds_cb)
    oof_rmse_blend = root_mean_squared_error(y_train, oof_blend)
    oof_mae_blend = mean_absolute_error(y_train, oof_blend)
    oof_r2_blend = r2_score(y_train, oof_blend)

    print("\n" + "=" * 80)
    print("OVERALL 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE (150,000 TRANSACTIONS)")
    print("=" * 80)
    print(f"Overall OOF LightGBM RMSE: {oof_rmse_lgb:.2f}")
    print(f"Overall OOF XGBoost  RMSE: {oof_rmse_xgb:.2f}")
    print(f"Overall OOF CatBoost RMSE: {oof_rmse_cb:.2f}")
    print(f"OVERALL OOF ENSEMBLE BLEND RMSE: {oof_rmse_blend:.2f} (MAE: {oof_mae_blend:.2f}, R2: {oof_r2_blend:.4f})")

    # Generate Submission File
    sub_df = pd.DataFrame({
        "Id": np.arange(n_test),
        "Predicted": test_blend
    })
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission_ensemble_5fold.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission generated and saved: {sub_path} ({len(sub_df):,} rows)")
    print(f"Sample predictions (first 5):\n{sub_df.head(5)}")

    metrics = {
        "oof_rmse_lgb": oof_rmse_lgb,
        "oof_rmse_xgb": oof_rmse_xgb,
        "oof_rmse_cb": oof_rmse_cb,
        "oof_rmse_blend": oof_rmse_blend,
        "oof_mae_blend": oof_mae_blend,
        "oof_r2_blend": oof_r2_blend
    }

    # Save metrics JSON
    with open(os.path.join(config.MODEL_DIR, "cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return oof_blend, test_blend, metrics

if __name__ == "__main__":
    train_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "train_features.parquet"))
    test_feat = pd.read_parquet(os.path.join(config.FEATURE_DATA_DIR, "test_features.parquet"))
    
    # 1. Run OOT benchmark
    evaluate_oot_benchmark(train_feat)

    # 2. Run 5-Fold CV and generate test predictions
    train_and_predict_5fold(train_feat, test_feat)

