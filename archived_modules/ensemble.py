"""
Blend LightGBM / sklearn GBM / XGBoost predictions. See progress-report.md
"Ensembling" section for rationale and results.

Leakage discipline, same principle as archived_modules/modeling.py's tuning-vs-defaults check:
blend weights are fit on OUT-OF-FOLD predictions generated from train_fold via the
same expanding-window CV used for hyperparameter tuning (archived_modules/split.py) — never on
val_fold directly, which would leak val_fold into the blend the same way fitting
hyperparameters on it would. val_fold is reserved for the one honest final check.
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

TARGET = 'MONTHLY_RENT'


def oof_predictions(model_specs: dict, train_df: pd.DataFrame, splits: list, categories: dict,
                     target: str = TARGET) -> pd.DataFrame:
    """model_specs: {name: (fit_fn, params)} where fit_fn matches archived_modules/modeling.py's
    fit_*_baseline signature: fit_fn(fold_train, fold_val, categories, **params) ->
    (model, metrics, X_val). Returns a frame indexed like train_df, one out-of-fold
    prediction column per model — a row's own fold never contributes to its own
    prediction, so these are safe to fit blend weights on."""
    oof = pd.DataFrame(index=train_df.index, columns=list(model_specs.keys()), dtype=float)
    for name, (fit_fn, params) in model_specs.items():
        for train_idx, val_idx in splits:
            fold_train, fold_val = train_df.loc[train_idx], train_df.loc[val_idx]
            model, _, X_val = fit_fn(fold_train, fold_val, categories, **params)
            oof.loc[val_idx, name] = model.predict(X_val)
    return oof


def grid_search_blend_weights(oof_df: pd.DataFrame, y: pd.Series, step: float = 0.05):
    """Coarse grid search over the weight simplex (weights >= 0, sum to 1) for a 2 or
    3-model blend, scored by RMSE on the out-of-fold predictions. Returns
    (best_weights_dict, results_df sorted best-first)."""
    names = list(oof_df.columns)
    valid_idx = oof_df.dropna().index
    preds, y_valid = oof_df.loc[valid_idx], y.loc[valid_idx]
    grid = np.round(np.arange(0, 1 + 1e-9, step), 10)
    rows = []
    if len(names) == 2:
        for w0 in grid:
            w1 = round(1 - w0, 10)
            blend = w0 * preds[names[0]] + w1 * preds[names[1]]
            rows.append({names[0]: w0, names[1]: w1,
                         'RMSE': mean_squared_error(y_valid, blend) ** 0.5})
    elif len(names) == 3:
        for w0, w1 in product(grid, grid):
            w2 = round(1 - w0 - w1, 10)
            if w2 < -1e-9:
                continue
            w2 = max(w2, 0.0)
            blend = w0 * preds[names[0]] + w1 * preds[names[1]] + w2 * preds[names[2]]
            rows.append({names[0]: w0, names[1]: w1, names[2]: w2,
                         'RMSE': mean_squared_error(y_valid, blend) ** 0.5})
    else:
        raise NotImplementedError('grid_search_blend_weights only supports 2 or 3 models')
    results = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    return results.iloc[0][names].to_dict(), results


def fit_nnls_weights(oof_df: pd.DataFrame, y: pd.Series) -> dict:
    """Non-negative least squares blend weights (scipy.optimize.nnls) — scales to any
    number of models, unlike grid_search_blend_weights' 2-3 model limit. Doesn't force
    weights to sum to 1 (unnecessary for RMSE; only the blended prediction's accuracy
    matters), but in practice lands close to 1 when every base model is a reasonably
    unbiased predictor of the target.

    CAUTION (see progress-report.md "Ensembling" section): tested with 5 base models
    of very different individual strength (3 similar-strength boosting models +
    Random Forest + kNN, the latter two considerably weaker on their own). NNLS put
    most of its weight on the two WEAKEST individual models and the resulting blend
    scored far worse on the true held-out fold than either the best single model or a
    same-strength-model blend — it was fitting noise in the OOF sample, not real
    signal. Prefer this only when candidate models are of comparable individual
    strength; grid_search_blend_weights (which respects the simplex and is easier to
    sanity-check by eye) is safer otherwise.
    """
    from scipy.optimize import nnls
    valid_idx = oof_df.dropna().index
    weights, _ = nnls(oof_df.loc[valid_idx].to_numpy(), y.loc[valid_idx].to_numpy())
    return dict(zip(oof_df.columns, weights))


def fit_stacking_meta(oof_df: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Ridge:
    """Alternative to grid_search_blend_weights: a Ridge meta-learner on the OOF
    predictions, coefficients constrained non-negative (positive=True) so it stays
    interpretable as a blend rather than a model that subtracts one base learner."""
    valid_idx = oof_df.dropna().index
    meta = Ridge(alpha=alpha, positive=True)
    meta.fit(oof_df.loc[valid_idx], y.loc[valid_idx])
    return meta


def apply_weights(preds: dict, weights: dict) -> pd.Series:
    """preds: {name: prediction Series/array} for the SAME rows. weights: {name: float}."""
    names = list(weights.keys())
    return sum(weights[n] * pd.Series(preds[n]).reset_index(drop=True) for n in names)
