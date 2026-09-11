"""
Data quality checks for the HDB rental pipeline, run on the output of
preprocessing.load_and_build(). See data-quality-report.md for the full writeup of
what each check found and why it matters — this module is the reusable code behind
that report, not a one-off script.

Categories (A-G), matching the report's structure:
  A. numeric range / sentinel-value checks
  B. cross-field logical consistency
  C. systematic price-per-sqm outlier detection (percentile-within-FLAT_TYPE)
  D. same-address/type/month rent variance (data error vs. real resolution limit)
  E. categorical near-duplicate labels
  F. auxiliary dataset coordinate sanity
  G. temporal coverage
"""
import pandas as pd

SG_LAT_BOUNDS = (1.1, 1.5)
SG_LON_BOUNDS = (103.5, 104.2)


def check_numeric_ranges(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """(A) min/max/zero-count/negative-count/null-count per numeric column — the
    fastest way to spot a sentinel value (e.g. -1) hiding as a 'valid' number,
    which no isna() check would ever catch."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({'column': col, 'min': s.min(), 'max': s.max(),
                     'zeros': (s == 0).sum(), 'negatives': (s < 0).sum(), 'nulls': s.isna().sum()})
    return pd.DataFrame(rows).set_index('column')


def find_sentinel_rows(df: pd.DataFrame, column: str, sentinel=-1) -> pd.DataFrame:
    """(A) Rows where `column` equals a suspected sentinel/placeholder value."""
    return df[df[column] == sentinel]


def check_sg_bounds(df: pd.DataFrame, lat_col: str = 'LATITUDE', lon_col: str = 'LONGITUDE') -> pd.DataFrame:
    """(A/F) Rows whose coordinates fall outside Singapore's actual bounding box —
    catches a bad geocode in any lat/long-bearing table (train/test or an auxiliary
    reference file like sg-mrt-stations.csv)."""
    lat_lo, lat_hi = SG_LAT_BOUNDS
    lon_lo, lon_hi = SG_LON_BOUNDS
    return df[(df[lat_col] < lat_lo) | (df[lat_col] > lat_hi) |
              (df[lon_col] < lon_lo) | (df[lon_col] > lon_hi)]


def check_cross_field_consistency(df: pd.DataFrame) -> dict:
    """(B) Jointly-impossible combinations that individually-valid ranges can't catch:
    renting before a building was completed / before its lease commenced, and
    remaining_lease outside a valid 99-year lease's [0, 99] bound."""
    return {
        'rented_before_built': df[df['approval_year'] < df['YEAR_COMPLETED']],
        'rented_before_lease_commence': df[df['approval_year'] < df['LEASE_COMMENCE_DATE']],
        'remaining_lease_out_of_bounds': df[(df['remaining_lease'] < 0) | (df['remaining_lease'] > 99)],
    }


def check_lease_completion_divergence(df: pd.DataFrame, threshold_years: int = 2,
                                       sentinel=-1) -> pd.DataFrame:
    """(B) LEASE_COMMENCE_DATE vs YEAR_COMPLETED diverging by more than threshold_years,
    excluding sentinel rows (see find_sentinel_rows) so the sentinel doesn't produce a
    false-alarm divergence signal. A real, large divergence here is expected for HDB
    (lease can be registered later than physical construction for older estates) —
    this check is for spotting it and confirming that, not assuming it's an error."""
    clean = df[df['YEAR_COMPLETED'] != sentinel]
    return clean[(clean['LEASE_COMMENCE_DATE'] - clean['YEAR_COMPLETED']).abs() > threshold_years]


def flag_rent_per_sqm_outliers(df: pd.DataFrame, low_q: float = 0.005, high_q: float = 0.995,
                                group_col: str = 'FLAT_TYPE') -> pd.DataFrame:
    """(C) Returns df with an added 'rent_per_sqm' and 'rent_per_sqm_outlier' column.
    Outliers are flagged by percentile WITHIN group_col (so a 1-room and a 5-room
    aren't compared on the same absolute rent/sqm scale) — NOT by a fixed multiple of
    IQR, which was tried first and mathematically failed here: with typical spread in
    this data, low_bound = Q1 - 3*IQR goes negative for a strictly-positive ratio,
    silently flagging nothing on the cheap side no matter how extreme.
    """
    df = df.copy()
    df['rent_per_sqm'] = df['MONTHLY_RENT'] / df['FLOOR_AREA_SQM']
    lo = df.groupby(group_col)['rent_per_sqm'].transform(lambda s: s.quantile(low_q))
    hi = df.groupby(group_col)['rent_per_sqm'].transform(lambda s: s.quantile(high_q))
    df['rent_per_sqm_outlier'] = (df['rent_per_sqm'] < lo) | (df['rent_per_sqm'] > hi)
    return df


def check_same_key_rent_variance(df: pd.DataFrame,
                                  key: list = ('TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'RENT_APPROVAL_DATE')
                                  ) -> pd.DataFrame:
    """(D) Groups by an address+type+month key and reports within-group MONTHLY_RENT
    count/std/mean for groups with >1 row. A LOW relative std (std/mean) across such
    groups indicates genuine unit-level market variance (different floors/condition
    within the same block — information no available feature captures), not a
    recording error; a systematically HIGH relative std would instead suggest data
    problems worth investigating further."""
    key = list(key)
    grp = df.groupby(key)['MONTHLY_RENT'].agg(['nunique', 'count', 'std', 'mean'])
    return grp[grp['count'] > 1]


def check_categorical_values(df: pd.DataFrame, columns: list) -> dict:
    """(E) Returns the sorted unique values for each categorical column, for manual
    (or fuzzy-match) inspection of near-duplicate labels beyond what normalization
    already fixed (the FLAT_TYPE '4 room'/'4-room' bug is exactly this failure mode
    elsewhere)."""
    return {col: sorted(df[col].dropna().unique()) for col in columns if col in df.columns}


def check_temporal_coverage(df: pd.DataFrame, date_col: str = 'RENT_APPROVAL_DATE') -> pd.Series:
    """(G) Row count per month — a sudden gap or spike can indicate a data-collection
    change (a new source merged in, a scraping outage) unrelated to the real market."""
    return df[date_col].value_counts().sort_index()


def run_full_report(train_feat: pd.DataFrame, test_feat: pd.DataFrame, aux_dir: str = 'data/auxiliary') -> dict:
    """Runs every check above and returns a dict of results, matching
    data-quality-report.md's structure. Doesn't print — caller decides how to
    present results (see main.ipynb's "Data quality analysis" section)."""
    numeric_cols = ['MONTHLY_RENT', 'FLOOR_AREA_SQM', 'LEASE_COMMENCE_DATE', 'remaining_lease',
                    'MAX_FLOOR', 'YEAR_COMPLETED', 'LATITUDE', 'LONGITUDE', 'dist_mrt_km', 'dist_mall_km']
    results = {
        'A_numeric_ranges': check_numeric_ranges(train_feat, numeric_cols),
        'A_sentinel_max_floor': find_sentinel_rows(train_feat, 'MAX_FLOOR'),
        'A_sentinel_max_floor_test': find_sentinel_rows(test_feat, 'MAX_FLOOR'),
        'A_bad_coords': check_sg_bounds(train_feat),
        'B_cross_field': check_cross_field_consistency(train_feat),
        'B_lease_completion_divergence': check_lease_completion_divergence(train_feat),
        'C_rent_per_sqm': flag_rent_per_sqm_outliers(train_feat),
        'D_same_key_variance': check_same_key_rent_variance(train_feat),
        'E_categorical_values': check_categorical_values(train_feat, ['TOWN', 'FLAT_MODEL']),
        'G_temporal_coverage': check_temporal_coverage(train_feat),
    }
    for fname in ['sg-mrt-stations.csv', 'sg-shopping-malls.csv', 'sg-schools.csv']:
        aux = pd.read_csv(f'{aux_dir}/{fname}')
        results[f'F_bad_coords_{fname}'] = check_sg_bounds(aux)
    return results
