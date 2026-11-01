"""
Time-based train/validation split.

EDA (initial-eda-inference.md, section 6) showed train and test are almost
non-overlapping in time (train: 2021-01 to 2025-03, test: 2025-03 to
2026-07) — this is a forecasting problem, not an i.i.d. regression one. A
random k-fold split would leak "future" rows into training and produce an
overly optimistic validation score. Split by date instead so validation
mimics the real train/test gap.
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def time_based_split(df: pd.DataFrame, date_col: str = 'RENT_APPROVAL_DATE', val_frac: float = 0.1):
    """Split df into (train_fold, val_fold, cutoff) using the last val_frac of rows by date.

    date_col is expected in 'YYYY-MM' string form (as in RENT_APPROVAL_DATE).
    Default val_frac=0.1 reproduces the EDA cutoff of ~2024-10-01 (~17,198 rows).
    """
    dates = pd.to_datetime(df[date_col] + '-01')
    cutoff = dates.quantile(1 - val_frac)
    train_mask = dates < cutoff
    return df[train_mask].copy(), df[~train_mask].copy(), cutoff


def time_split_at(df: pd.DataFrame, cutoff_month: str, date_col: str = 'RENT_APPROVAL_DATE'):
    """Split df into (train, val, cutoff) at an explicit 'YYYY-MM' month: train is every
    row strictly before cutoff_month, val is cutoff_month onward.

    Used for the horizon-matched holdout (progress-report "Validation upgrades"): the
    test set runs 16 months past train's last date, so cutoff_month='2023-12' leaves a
    16-month validation window (2023-12 to 2025-03) that stresses time-sensitive
    features the way the test set will, which the default 6-month time_based_split
    cannot.
    """
    dates = pd.to_datetime(df[date_col] + '-01')
    cutoff = pd.Timestamp(cutoff_month + '-01')
    train_mask = dates < cutoff
    return df[train_mask].copy(), df[~train_mask].copy(), cutoff


def expanding_window_splits(df: pd.DataFrame, n_splits: int = 4, date_col: str = 'RENT_APPROVAL_DATE'):
    """Yields (train_idx, val_idx) for n_splits expanding-window folds, ordered by date_col.

    For hyperparameter tuning ONLY on data that already EXCLUDES the final held-out
    validation fold — i.e. pass train_fold (from time_based_split), never train_feat.
    Tuning against data that includes the final holdout would leak "future" rows into
    model selection, defeating the point of holding them out in the first place.

    Materialize this (list(...)) before reusing it across multiple hyperparameter
    combinations — it's a generator, and TimeSeriesSplit.split re-runs per call anyway,
    but callers that build the list once avoid recomputing the date sort repeatedly.
    """
    sorted_df = df.sort_values(date_col)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_pos, val_pos in tscv.split(sorted_df):
        yield sorted_df.index[train_pos], sorted_df.index[val_pos]
