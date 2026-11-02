"""
Evaluation helpers behind the "Validation upgrades" in the RMSE bottleneck report:
error by months ahead of the training cutoff, and a paired bootstrap that puts an
interval on an RMSE difference between two models scored on the same rows.

Why: the project's decisions kept landing on sub-1-point differences (Optuna 1.15,
CatBoost blend 0.84) that a single holdout cannot resolve, and the test set sits up to
16 months past the training data while the default validation fold sees only 6.
"""
import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def rmse_by_month_ahead(val_df: pd.DataFrame, preds: pd.Series, cutoff: pd.Timestamp,
                         target: str = 'MONTHLY_RENT',
                         date_col: str = 'RENT_APPROVAL_DATE') -> pd.Series:
    """RMSE per month since the training cutoff (1 = the cutoff month itself). Shows
    whether error grows with distance from the training data, which a single
    aggregate RMSE averages away."""
    dates = pd.to_datetime(val_df[date_col] + '-01')
    ahead = (dates.dt.year - cutoff.year) * 12 + (dates.dt.month - cutoff.month) + 1
    err2 = (val_df[target] - preds.loc[val_df.index]) ** 2
    return np.sqrt(err2.groupby(ahead.values).mean()).rename('rmse')


def paired_bootstrap_rmse_diff(y_true, pred_a, pred_b, groups, n_boot: int = 500,
                                seed: int = 0) -> dict:
    """Bootstrap the RMSE difference (b minus a; negative means b is better) by
    resampling whole groups with replacement (e.g. address-month blocks), since rows in
    one block share unit-level noise and are not independent. Returns the observed
    difference, a 95% interval, and the share of resamples where b beats a."""
    y = np.asarray(y_true, dtype=float)
    ea = (y - np.asarray(pred_a, dtype=float)) ** 2
    eb = (y - np.asarray(pred_b, dtype=float)) ** 2
    frame = pd.DataFrame({'g': np.asarray(groups), 'ea': ea, 'eb': eb})
    agg = frame.groupby('g').agg(sa=('ea', 'sum'), sb=('eb', 'sum'), n=('ea', 'size'))
    sa, sb, n = agg['sa'].to_numpy(), agg['sb'].to_numpy(), agg['n'].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(agg), len(agg))
        diffs[i] = np.sqrt(sb[idx].sum() / n[idx].sum()) - np.sqrt(sa[idx].sum() / n[idx].sum())
    observed = np.sqrt(eb.mean()) - np.sqrt(ea.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'diff': float(observed), 'ci_low': float(lo), 'ci_high': float(hi),
            'share_b_better': float((diffs < 0).mean())}
