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
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

TARGET = 'MONTHLY_RENT'

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


def _prep_gbm_frame(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    X = df[GBM_NUMERIC_FEATURES + GBM_CATEGORICAL_FEATURES].copy()
    for c in GBM_CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def fit_gbm_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                      target: str = TARGET, random_state: int = 0):
    X_train, X_val = _prep_gbm_frame(train_df, categories), _prep_gbm_frame(val_df, categories)
    model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=random_state)
    model.fit(X_train, train_df[target])
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def fit_lgbm_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                       target: str = TARGET, random_state: int = 0, **params):
    X_train, X_val = _prep_gbm_frame(train_df, categories), _prep_gbm_frame(val_df, categories)
    model = LGBMRegressor(random_state=random_state, verbose=-1, **params)
    model.fit(X_train, train_df[target], categorical_feature=GBM_CATEGORICAL_FEATURES)
    metrics = evaluate(val_df[target], model.predict(X_val))
    return model, metrics, X_val


def fit_xgb_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, categories: dict,
                      target: str = TARGET, random_state: int = 0, **params):
    X_train, X_val = _prep_gbm_frame(train_df, categories), _prep_gbm_frame(val_df, categories)
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


def fit_gbm_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                  random_state: int = 0):
    """Fit the sklearn GBM on the entire dataset (no held-out split) — for the final
    test-time model, once the split-based metrics above have established trust."""
    model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=random_state)
    model.fit(_prep_gbm_frame(train_df, categories), train_df[target])
    return model


def fit_lgbm_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                   random_state: int = 0, **params):
    model = LGBMRegressor(random_state=random_state, verbose=-1, **params)
    model.fit(_prep_gbm_frame(train_df, categories), train_df[target],
               categorical_feature=GBM_CATEGORICAL_FEATURES)
    return model


def fit_xgb_full(train_df: pd.DataFrame, categories: dict, target: str = TARGET,
                  random_state: int = 0, **params):
    """**params forwards tuned hyperparameters from grid_search_xgb's best row, e.g.
    fit_xgb_full(train_feat, categories, **best_params) where best_params is
    grid_search_xgb(...).iloc[0][param_grid.keys()].to_dict()."""
    model = XGBRegressor(random_state=random_state, enable_categorical=True, tree_method='hist', **params)
    model.fit(_prep_gbm_frame(train_df, categories), train_df[target])
    return model


def feature_importance(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5,
                        random_state: int = 0) -> pd.Series:
    """Permutation importance — validates (or contradicts) the EDA feature-selection calls."""
    result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats,
                                     random_state=random_state, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X_val.columns).sort_values(ascending=False)


def predict_test(model, test_feat: pd.DataFrame, is_gbm: bool, categories: dict = None) -> pd.Series:
    """Apply a fitted model to build_features(test) output and return predictions.
    is_gbm=True covers all three tree models (sklearn GBM, LightGBM, XGBoost) — they
    share the same categorical-frame input format. categories is required when
    is_gbm=True (use the same dict passed to the corresponding fit_*_full call)."""
    X_test = _prep_gbm_frame(test_feat, categories) if is_gbm else test_feat[LINEAR_FEATURES]
    return pd.Series(model.predict(X_test), index=test_feat.index, name='MONTHLY_RENT')
