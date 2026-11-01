"""
Baseline models for MONTHLY_RENT prediction.

Five baselines, weakest to strongest expected:

- Naive (predict-the-mean): the floor. Any real model's MAE/RMSE should be
  judged against this, not in isolation — it's the reference point for
  "how much is the model actually learning?"

- Ridge: a plain linear model on a small, non-redundant numeric feature set
  (drops the collinear pairs found in EDA step 11: flat_type_ord vs
  FLOOR_AREA_SQM, LEASE_COMMENCE_DATE vs remaining_lease).

- HistGradientBoostingRegressor (sklearn): native gradient boosting, no
  extra install, accepts categoricals directly. Can represent the
  non-linear, threshold-like MRT/mall proximity effect found in EDA step
  8b that Ridge cannot.

- LGBMRegressor (lightgbm): histogram-based boosting like sklearn's GBM,
  typically faster and often a bit more accurate out of the box. Needs
  categorical columns declared explicitly via `categorical_feature=`.

- XGBRegressor (xgboost): boosting with a different tree-growth strategy
  (level-wise by default) than LightGBM's leaf-wise growth — comparing
  the two is a cheap way to see if one generalizes better on this data.
  Uses native categorical support (`enable_categorical=True`).

Install (macOS): pip install lightgbm xgboost, plus `brew install libomp`
— LightGBM's compiled binary needs the OpenMP runtime, which isn't a
Python package and pip won't install it for you.

Shared category sets: `get_categories(train_feat)` must be computed once
on the FULL feature-built train set and passed to every fit/predict call
below. Without it, each split infers its own categories independently —
e.g. FLAT_MODEL's "improved maisonette" appears in train's later months
(val_fold) but not its earlier months (train_fold); XGBoost's native
categorical handling then correctly errors on an "unseen" category at
predict time instead of silently guessing (sklearn's GBM and LightGBM
tolerate it, but the fix belongs here, not per-model).
"""
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

TARGET = 'MONTHLY_RENT'

# Established in the "CatBoost" investigation (progress-report.md) — best of depth
# (4/6/8) x learning_rate (0.03/0.05) x l2_leaf_reg (3/10), on the feature set BEFORE
# addr_level_feat existed. Reused as-is by fit_catboost_baseline/_full below (not
# re-tuned) when the question is "does blending with LightGBM help", not "what's
# CatBoost's best config for the current feature set".
CATBOOST_BEST_PARAMS = {'depth': 8, 'learning_rate': 0.03, 'iterations': 1500}

# Numeric features only, with the EDA step-11 collinear duplicates removed.
LINEAR_FEATURES = ['FLOOR_AREA_SQM', 'remaining_lease', 'dist_mrt_km', 'dist_mall_km',
                    'MAX_FLOOR', 'YEAR_COMPLETED', 'stock_index', 'coe_premium']

# Ablation-tested (see chat / initial-eda-inference.md "Resolving the open items"):
# - SUBZONE replaces TOWN entirely — TOWN's permutation importance drops to ~0 once
#   SUBZONE is present (finer-grained location signal makes TOWN redundant), and
#   REGION/PLANNING_AREA add nothing beyond SUBZONE either, so all three are dropped.
# - MAX_FLOOR/YEAR_COMPLETED (from sg-hdb-block.csv) and stock_index/coe_premium
#   (corrected macro indicators) each independently improved MAE by ~$70-75 in
#   isolation; combined with SUBZONE the best validated set below gets to $346 MAE
#   (vs $422 for the original TOWN-only feature set).
GBM_NUMERIC_FEATURES = ['FLOOR_AREA_SQM', 'remaining_lease', 'dist_mrt_km', 'dist_mall_km',
                         'MAX_FLOOR', 'YEAR_COMPLETED', 'stock_index', 'coe_premium']
GBM_CATEGORICAL_FEATURES = ['FLAT_TYPE', 'FLAT_MODEL', 'SUBZONE']


def evaluate(y_true, y_pred) -> dict:
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
    }


def fit_naive_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str = TARGET):
    """Predicts the train mean for every row. Not a real model — a reference point."""
    mean_pred = train_df[target].mean()
    preds = pd.Series(mean_pred, index=val_df.index)
    metrics = evaluate(val_df[target], preds)
    return mean_pred, metrics


def fit_ridge_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str = TARGET):
    X_train, X_val = train_df[LINEAR_FEATURES], val_df[LINEAR_FEATURES]
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_df[target])
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics


def get_categories(df: pd.DataFrame) -> dict:
    """Fixed category sets, computed once on the full train_feat. Pass the result to
    every _prep_gbm_frame/fit_*/predict_test call below so train_fold, val_fold and
    test_feat all share one consistent category->code mapping."""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in GBM_CATEGORICAL_FEATURES}


def _prep_gbm_frame(df: pd.DataFrame, categories: dict, numeric_features: list = None) -> pd.DataFrame:
    numeric_features = GBM_NUMERIC_FEATURES if numeric_features is None else numeric_features
    X = df[numeric_features + GBM_CATEGORICAL_FEATURES].copy()
    for c in GBM_CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def fit_gbm_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                      target: str = TARGET, random_state: int = 0, numeric_features: list = None):
    X_train = _prep_gbm_frame(train_df, categories, numeric_features)
    X_val = _prep_gbm_frame(val_df, categories, numeric_features)
    model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=random_state)
    model.fit(X_train, train_df[target])
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def fit_lgbm_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                       target: str = TARGET, random_state: int = 0, numeric_features: list = None,
                       **params):
    X_train = _prep_gbm_frame(train_df, categories, numeric_features)
    X_val = _prep_gbm_frame(val_df, categories, numeric_features)
    model = LGBMRegressor(random_state=random_state, verbose=-1, **params)
    model.fit(X_train, train_df[target], categorical_feature=GBM_CATEGORICAL_FEATURES)
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def fit_xgb_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                      target: str = TARGET, random_state: int = 0, numeric_features: list = None,
                      **params):
    X_train = _prep_gbm_frame(train_df, categories, numeric_features)
    X_val = _prep_gbm_frame(val_df, categories, numeric_features)
    model = XGBRegressor(random_state=random_state, enable_categorical=True, tree_method='hist', **params)
    model.fit(X_train, train_df[target])
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def _grid_search_tree(model_ctor, fixed_kwargs: dict, fit_kwargs: dict, train_df: pd.DataFrame,
                       categories: dict, splits: list, param_grid: dict,
                       target: str = TARGET) -> pd.DataFrame:
    """Shared grid-search loop behind grid_search_xgb/grid_search_lgbm. Scores every
    param_grid combination by mean RMSE across the given expanding-window CV folds
    (see split.expanding_window_splits) — RMSE, not MAE, because that's the actual
    Kaggle scoring metric for this task; MAE is tracked alongside for reference only.
    splits must come from train_fold only (never train_feat) — see that function's
    docstring for why. Returns every combination's mean/std RMSE and mean MAE, sorted
    best-first by RMSE, so you can see fold-to-fold sensitivity, not just the single
    best number."""
    from itertools import product
    keys = list(param_grid.keys())
    rows = []
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        fold_maes, fold_rmses = [], []
        for train_idx, val_idx in splits:
            fold_train, fold_val = train_df.loc[train_idx], train_df.loc[val_idx]
            X_train = _prep_gbm_frame(fold_train, categories)
            X_val = _prep_gbm_frame(fold_val, categories)
            model = model_ctor(random_state=0, **fixed_kwargs, **params)
            model.fit(X_train, fold_train[target], **fit_kwargs)
            preds = model.predict(X_val)
            fold_maes.append(mean_absolute_error(fold_val[target], preds))
            fold_rmses.append(mean_squared_error(fold_val[target], preds) ** 0.5)
        rows.append({**params,
                      'mean_RMSE': pd.Series(fold_rmses).mean(), 'std_RMSE': pd.Series(fold_rmses).std(),
                      'mean_MAE': pd.Series(fold_maes).mean(), 'std_MAE': pd.Series(fold_maes).std()})
    return pd.DataFrame(rows).sort_values('mean_RMSE').reset_index(drop=True)


def grid_search_xgb(train_df: pd.DataFrame, categories: dict, splits: list, param_grid: dict,
                     target: str = TARGET) -> pd.DataFrame:
    return _grid_search_tree(XGBRegressor, {'enable_categorical': True, 'tree_method': 'hist'}, {},
                              train_df, categories, splits, param_grid, target)


def grid_search_lgbm(train_df: pd.DataFrame, categories: dict, splits: list, param_grid: dict,
                      target: str = TARGET) -> pd.DataFrame:
    return _grid_search_tree(LGBMRegressor, {'verbose': -1}, {'categorical_feature': GBM_CATEGORICAL_FEATURES},
                              train_df, categories, splits, param_grid, target)


def optuna_search_lgbm(train_df: pd.DataFrame, categories: dict, splits: list, param_space: dict,
                        n_trials: int = 50, numeric_features: list = None, target: str = TARGET,
                        seed: int = 0):
    """TPE-based hyperparameter search over the given expanding-window CV folds (see
    split.expanding_window_splits) — same evaluation discipline as grid_search_lgbm
    (mean RMSE across folds; splits must come from train_fold only, never train_feat,
    for the same reason grid_search_lgbm's docstring gives), swapping grid/random
    search for Optuna's Tree-structured Parzen Estimator sampler. Tried per
    progress-report.md's "Known limitations" note that a formal Bayesian optimizer
    hadn't been tested as a distinct method (only randomized/grid search had).

    param_space: {name: (low, high, log)} for float/int params (log=True samples on
    a log scale, e.g. learning_rate) or {name: [choices]} for categorical ones.

    Returns (study, trials_df) — trials_df matches grid_search_lgbm's output shape
    (one row per trial, sorted best-first by mean_RMSE) so results are directly
    comparable to the existing 30-trial/15-trial randomized-search results in
    progress-report.md.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for name, spec in param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            else:
                low, high, log = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high, log=log)
                else:
                    params[name] = trial.suggest_float(name, low, high, log=log)

        fold_rmses = []
        for train_idx, val_idx in splits:
            fold_train, fold_val = train_df.loc[train_idx], train_df.loc[val_idx]
            X_train = _prep_gbm_frame(fold_train, categories, numeric_features)
            X_val = _prep_gbm_frame(fold_val, categories, numeric_features)
            model = LGBMRegressor(random_state=0, verbose=-1, **params)
            model.fit(X_train, fold_train[target], categorical_feature=GBM_CATEGORICAL_FEATURES)
            preds = model.predict(X_val)
            fold_rmses.append(mean_squared_error(fold_val[target], preds) ** 0.5)
        return pd.Series(fold_rmses).mean()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.rename(columns={'value': 'mean_RMSE'})
    param_cols = [c for c in trials_df.columns if c.startswith('params_')]
    trials_df = trials_df[param_cols + ['mean_RMSE']].rename(
        columns={c: c.replace('params_', '') for c in param_cols})
    return study, trials_df.sort_values('mean_RMSE').reset_index(drop=True)


def _prep_catboost_frame(df: pd.DataFrame, numeric_features: list = None) -> pd.DataFrame:
    """CatBoost takes categoricals as plain strings (no pandas Categorical dtype /
    fixed category codes needed — unlike LightGBM/XGBoost, it doesn't require
    train/val to share one category->code mapping across folds)."""
    numeric_features = GBM_NUMERIC_FEATURES if numeric_features is None else numeric_features
    X = df[numeric_features + GBM_CATEGORICAL_FEATURES].copy()
    for c in GBM_CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return X


def fit_catboost_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict = None,
                           target: str = TARGET, random_state: int = 0, numeric_features: list = None,
                           **params):
    """Same (train_df, val_df, categories, **params) -> (model, metrics, X_val) shape
    as fit_lgbm_baseline/fit_xgb_baseline, so this plugs directly into
    archived_modules/ensemble.py::oof_predictions unmodified. categories is accepted but
    unused (interface compatibility only) — CatBoost doesn't need LightGBM/XGBoost's
    shared category->code map."""
    X_train = _prep_catboost_frame(train_df, numeric_features)
    X_val = _prep_catboost_frame(val_df, numeric_features)
    model = CatBoostRegressor(random_state=random_state, verbose=False, **params)
    model.fit(X_train, train_df[target], cat_features=GBM_CATEGORICAL_FEATURES)
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def fit_catboost_full(train_df: pd.DataFrame, categories: dict = None, target: str = TARGET,
                       random_state: int = 0, numeric_features: list = None, **params):
    model = CatBoostRegressor(random_state=random_state, verbose=False, **params)
    model.fit(_prep_catboost_frame(train_df, numeric_features), train_df[target],
               cat_features=GBM_CATEGORICAL_FEATURES)
    return model


def fit_gbm_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                  random_state: int = 0, numeric_features: list = None):
    """Fit the sklearn GBM on the entire dataset (no held-out split) — for the final
    test-time model, once the split-based metrics above have established trust."""
    model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=random_state)
    model.fit(_prep_gbm_frame(train_df, categories, numeric_features), train_df[target])
    return model


def fit_lgbm_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                   random_state: int = 0, numeric_features: list = None, **params):
    model = LGBMRegressor(random_state=random_state, verbose=-1, **params)
    model.fit(_prep_gbm_frame(train_df, categories, numeric_features), train_df[target],
               categorical_feature=GBM_CATEGORICAL_FEATURES)
    return model


def fit_xgb_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                  random_state: int = 0, numeric_features: list = None, **params):
    """**params forwards tuned hyperparameters from grid_search_xgb's best row, e.g.
    fit_xgb_full(train_feat, categories, **best_params) where best_params is
    grid_search_xgb(...).iloc[0][param_grid.keys()].to_dict()."""
    model = XGBRegressor(random_state=random_state, enable_categorical=True, tree_method='hist', **params)
    model.fit(_prep_gbm_frame(train_df, categories, numeric_features), train_df[target])
    return model


def feature_importance(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5,
                        random_state: int = 0) -> pd.Series:
    """Permutation importance — validates (or contradicts) the EDA feature-selection calls."""
    result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats,
                                     random_state=random_state, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X_val.columns).sort_values(ascending=False)


def predict_test(model, test_feat: pd.DataFrame, is_gbm: bool, categories: dict = None,
                  numeric_features: list = None) -> pd.Series:
    """Apply a fitted model to build_features(test) output and return predictions.
    is_gbm=True covers all three tree models (sklearn GBM, LightGBM, XGBoost) — they
    share the same categorical-frame input format. categories is required when
    is_gbm=True (use the same dict passed to the corresponding fit_*_full call).
    numeric_features must match whatever was passed to that fit_*_full call too (e.g.
    GBM_NUMERIC_FEATURES + ['addr_level_feat'])."""
    X_test = _prep_gbm_frame(test_feat, categories, numeric_features) if is_gbm else test_feat[LINEAR_FEATURES]
    return pd.Series(model.predict(X_test), index=test_feat.index, name='MONTHLY_RENT')
