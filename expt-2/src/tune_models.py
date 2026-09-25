"""
Bayesian Hyperparameter Optimization Module (Sprint 4: Sub-430 Frontier)
Path: expt-2/src/tune_models.py

Uses Optuna TPE sampler to find optimal hyperparameters on the Out-of-Time (OOT) holdout:
1. LightGBM (num_leaves, max_depth, learning_rate, colsample, subsample, reg_alpha, reg_lambda)
2. XGBoost (max_depth, learning_rate, colsample_bytree, subsample, min_child_weight, reg_alpha, reg_lambda)
3. CatBoost (depth, learning_rate, l2_leaf_reg, random_strength)

Saves best hyperparameters to: expt-2/outputs/models/best_params_v3.json
"""

import os
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
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

    return train_df, val_df, feat_cols, cat_cols, y_train_res, y_val_res, y_val_actual, y_val_bench

def tune_lightgbm(n_trials=20):
    print("=" * 80)
    print(f"OPTUNA TUNING: LightGBM ({n_trials} trials)")
    print("=" * 80)
    train_df, val_df, feat_cols, cat_cols, y_train_res, y_val_res, y_val_actual, y_val_bench = load_data()

    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    def objective(trial):
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.045, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 70, 160),
            "max_depth": trial.suggest_int("max_depth", 8, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 0.90),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
            "random_state": config.RANDOM_SEED,
            "n_jobs": 4,
            "verbose": -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_df[feat_cols], y_train_res,
            eval_set=[(val_df[feat_cols], y_val_res)],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
        preds = y_val_bench + model.predict(val_df[feat_cols])
        return root_mean_squared_error(y_val_actual, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[LightGBM Best Trial] RMSE: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value

def tune_xgboost(n_trials=15):
    print("\n" + "=" * 80)
    print(f"OPTUNA TUNING: XGBoost ({n_trials} trials)")
    print("=" * 80)
    train_df, val_df, feat_cols, cat_cols, y_train_res, y_val_res, y_val_actual, y_val_bench = load_data()

    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    def objective(trial):
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.045, log=True),
            "max_depth": trial.suggest_int("max_depth", 7, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.65, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
            "enable_categorical": True,
            "random_state": config.RANDOM_SEED,
            "n_jobs": 4,
            "early_stopping_rounds": 40
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            train_df[feat_cols], y_train_res,
            eval_set=[(val_df[feat_cols], y_val_res)],
            verbose=False
        )
        preds = y_val_bench + model.predict(val_df[feat_cols])
        return root_mean_squared_error(y_val_actual, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[XGBoost Best Trial] RMSE: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value

def tune_catboost(n_trials=10):
    print("\n" + "=" * 80)
    print(f"OPTUNA TUNING: CatBoost ({n_trials} trials)")
    print("=" * 80)
    train_df, val_df, feat_cols, cat_cols, y_train_res, y_val_res, y_val_actual, y_val_bench = load_data()

    cb_train = train_df[feat_cols].copy()
    cb_val = val_df[feat_cols].copy()
    for c in cat_cols:
        cb_train[c] = cb_train[c].astype(str)
        cb_val[c] = cb_val[c].astype(str)

    def objective(trial):
        params = {
            "iterations": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.06),
            "depth": trial.suggest_int("depth", 6, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 12.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
            "cat_features": cat_cols,
            "random_seed": config.RANDOM_SEED,
            "thread_count": 4,
            "verbose": False,
            "early_stopping_rounds": 40
        }
        model = CatBoostRegressor(**params)
        model.fit(cb_train, y_train_res, eval_set=(cb_val, y_val_res))
        preds = y_val_bench + model.predict(cb_val)
        return root_mean_squared_error(y_val_actual, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[CatBoost Best Trial] RMSE: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value

def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_lgb_params, best_lgb_rmse = tune_lightgbm(n_trials=18)
    best_xgb_params, best_xgb_rmse = tune_xgboost(n_trials=12)
    best_cb_params, best_cb_rmse = tune_catboost(n_trials=8)

    all_best = {
        "lightgbm": {
            "best_rmse": float(best_lgb_rmse),
            "params": best_lgb_params
        },
        "xgboost": {
            "best_rmse": float(best_xgb_rmse),
            "params": best_xgb_params
        },
        "catboost": {
            "best_rmse": float(best_cb_rmse),
            "params": best_cb_params
        }
    }

    out_json = os.path.join(config.MODEL_DIR, "best_params_v3.json")
    with open(out_json, "w") as f:
        json.dump(all_best, f, indent=2)
    print(f"\nAll optimal hyperparameters saved to: {out_json}")

if __name__ == "__main__":
    main()

