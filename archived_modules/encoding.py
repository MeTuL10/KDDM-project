"""
Explicit categorical encodings, tried as an alternative to the tree models' native
categorical handling (archived_modules/modeling.py's default). See chat / exploration/Aayush/progress-report.md
"Encoding experiment" for the reasoning behind each choice and the results.

- FLAT_TYPE: ordinal (room count has a real order) — reuses
  preprocessing.FLAT_TYPE_RANK, already computed as `flat_type_ord`.
- FLAT_MODEL / SUBZONE: K-fold out-of-fold target encoding (mean rent per category,
  smoothed toward the global mean) — nominal, no natural order, and SUBZONE is too
  high-cardinality (156 levels) for one-hot to be practical for a linear model.
- SUBZONE alternative: LATITUDE/LONGITUDE directly, since SUBZONE is fundamentally a
  discretized proxy for location and the exact coordinates are already available from
  the sg-hdb-block.csv join.

Leakage discipline: kfold_target_encode() computes OUT-OF-FOLD means for the frame a
model will be *trained* on (so no row ever sees its own target), and separately
encodes any other frames (a validation/test frame) using means fit on the FULL
training frame — safe because those frames are never used to fit the model.
"""
import pandas as pd
from sklearn.model_selection import KFold

TARGET = 'MONTHLY_RENT'


def _smoothed_means(df: pd.DataFrame, col: str, target: str, m: float) -> pd.Series:
    """Category -> smoothed mean target, shrunk toward the global mean by an m-estimate
    (m acts like a pseudo-count of "prior" global-mean observations per category) —
    keeps rare categories from getting a noisy, overfit mean."""
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    return (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)


def kfold_target_encode(fit_df: pd.DataFrame, apply_dfs: list, col: str, target: str = TARGET,
                         n_splits: int = 5, m: float = 10.0, random_state: int = 0):
    """Target-encode `col` by mean of `target`.

    Returns (fit_encoded, [apply_df_encoded, ...]):
    - fit_encoded: out-of-fold encoding for fit_df's own rows — safe to train on.
    - apply_dfs' encodings: use means fit on ALL of fit_df (no leakage risk, these
      frames aren't used to fit the model).
    Unseen categories (in an apply_df but not fit_df) fall back to fit_df's global mean.
    """
    global_mean = fit_df[target].mean()
    fit_encoded = pd.Series(index=fit_df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fit_pos, hold_pos in kf.split(fit_df):
        fold_fit, fold_hold = fit_df.iloc[fit_pos], fit_df.iloc[hold_pos]
        means = _smoothed_means(fold_fit, col, target, m)
        fit_encoded.iloc[hold_pos] = fold_hold[col].map(means).fillna(global_mean).values

    full_means = _smoothed_means(fit_df, col, target, m)
    apply_encoded = [df[col].map(full_means).fillna(global_mean) for df in apply_dfs]
    return fit_encoded, apply_encoded


def add_encoded_features(fit_df: pd.DataFrame, apply_dfs: list, columns: list = ('FLAT_MODEL', 'SUBZONE'),
                          target: str = TARGET, n_splits: int = 5, m: float = 10.0,
                          random_state: int = 0):
    """Adds a '<col>_te' target-encoded column (see kfold_target_encode) to a COPY of
    fit_df and each of apply_dfs, for every column in `columns`. Returns
    (fit_df_copy, [apply_df_copy, ...])."""
    fit_df = fit_df.copy()
    apply_dfs = [df.copy() for df in apply_dfs]
    for col in columns:
        fit_encoded, applied = kfold_target_encode(fit_df, apply_dfs, col, target, n_splits, m, random_state)
        fit_df[f'{col}_te'] = fit_encoded
        for df, enc in zip(apply_dfs, applied):
            df[f'{col}_te'] = enc
    return fit_df, apply_dfs
