"""
Data cleaning and feature engineering for the HDB rental prediction task.

Consolidates the fixes and features validated during EDA (see
exploration/aayush_exploration.ipynb and exploration/initial-eda-inference.md)
into one place, so train and test always go through *identical* transforms:

- FLAT_TYPE has two spellings ("4 room" / "4-room") that must be merged.
- TOWN / STREET / BLOCK casing must match sg-hdb-block.csv (lowercase) to join.
- dist_mrt_km / dist_mall_km are validated, real (if modest) predictors.
- dist_school_km / dist_primary_school_km tested as ~zero correlation in EDA
  and are intentionally NOT produced here.
"""
import numpy as np
import pandas as pd

FLAT_TYPE_ORDER = ['1-room', '2-room', '3-room', '4-room', '5-room', 'executive']
FLAT_TYPE_RANK = {t: i for i, t in enumerate(FLAT_TYPE_ORDER)}

JOIN_KEYS = ['TOWN', 'BLOCK', 'STREET']


def normalize_flat_type(series: pd.Series) -> pd.Series:
    """Collapse '4 room' / '4-room' style variants into one spelling."""
    return series.str.strip().str.lower().str.replace(' ', '-', regex=False)


def clean_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/strip TOWN, STREET, BLOCK so they match sg-hdb-block.csv for joining."""
    df = df.copy()
    df['TOWN'] = df['TOWN'].str.strip().str.lower()
    df['STREET'] = df['STREET'].str.strip().str.lower()
    df['BLOCK'] = df['BLOCK'].astype(str).str.strip().str.lower()
    return df


def add_lease_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive approval_year and remaining_lease (99-year HDB lease minus age at approval)."""
    df = df.copy()
    df['approval_year'] = df['RENT_APPROVAL_DATE'].str.slice(0, 4).astype(int)
    df['remaining_lease'] = 99 - (df['approval_year'] - df['LEASE_COMMENCE_DATE'])
    return df


def add_flat_type_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """FLAT_TYPE has a natural room-count order; encode it as an ordinal 0..5."""
    df = df.copy()
    df['flat_type_ord'] = df['FLAT_TYPE'].map(FLAT_TYPE_RANK)
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between (lat1, lon1) and (lat2, lon2). Inputs in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def add_nearest_distance(df: pd.DataFrame, poi_df: pd.DataFrame, out_col: str,
                          poi_lat: str = 'LATITUDE', poi_lon: str = 'LONGITUDE') -> pd.DataFrame:
    """For each row (needs LATITUDE/LONGITUDE), distance in km to the nearest point in poi_df."""
    df = df.copy()
    dists = np.full(len(df), np.nan)
    valid = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    lat_arr = poi_df[poi_lat].to_numpy()
    lon_arr = poi_df[poi_lon].to_numpy()
    for i, (lat, lon) in zip(valid.index, zip(valid['LATITUDE'], valid['LONGITUDE'])):
        d = haversine_km(lat, lon, lat_arr, lon_arr)
        dists[df.index.get_loc(i)] = d.min()
    df[out_col] = dists
    return df


def add_density_within_radius(df: pd.DataFrame, poi_df: pd.DataFrame, out_col: str,
                               radius_km: float = 1.0, poi_lat: str = 'LATITUDE',
                               poi_lon: str = 'LONGITUDE') -> pd.DataFrame:
    """For each row, count of poi_df entries within radius_km — a neighborhood-richness
    signal distinct from nearest-point distance (EDA step 8b found the MRT/mall
    proximity effect saturates past ~1km, where distance alone stops discriminating;
    density can still vary among rows that are all 'close enough')."""
    df = df.copy()
    counts = np.full(len(df), np.nan)
    valid = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    lat_arr = poi_df[poi_lat].to_numpy()
    lon_arr = poi_df[poi_lon].to_numpy()
    for i, (lat, lon) in zip(valid.index, zip(valid['LATITUDE'], valid['LONGITUDE'])):
        d = haversine_km(lat, lon, lat_arr, lon_arr)
        counts[df.index.get_loc(i)] = (d <= radius_km).sum()
    df[out_col] = counts
    return df


def add_nearest_poi_attribute(df: pd.DataFrame, poi_df: pd.DataFrame, attribute_col: str,
                               out_col: str, poi_lat: str = 'LATITUDE',
                               poi_lon: str = 'LONGITUDE') -> pd.DataFrame:
    """For each row, the value of attribute_col belonging to the NEAREST poi_df entry
    (not the distance itself) — e.g. how many MRT lines serve the closest station, a
    connectivity signal distinct from raw distance (an interchange a bit farther away
    can matter more than a single-line station that's closer)."""
    df = df.copy()
    values = np.full(len(df), np.nan)
    valid = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    lat_arr = poi_df[poi_lat].to_numpy()
    lon_arr = poi_df[poi_lon].to_numpy()
    attr_arr = poi_df[attribute_col].to_numpy()
    for i, (lat, lon) in zip(valid.index, zip(valid['LATITUDE'], valid['LONGITUDE'])):
        d = haversine_km(lat, lon, lat_arr, lon_arr)
        values[df.index.get_loc(i)] = attr_arr[d.argmin()]
    df[out_col] = values
    return df


def add_mrt_line_count(mrt: pd.DataFrame) -> pd.DataFrame:
    """sg-mrt-stations.csv has one row PER LINE at a station (an interchange serving
    3 lines appears as 3 rows, confirmed sharing identical LATITUDE/LONGITUDE) — this
    adds a 'line_count' column (rows sharing the same NAME) for use with
    add_nearest_poi_attribute to build an MRT connectivity feature."""
    mrt = mrt.copy()
    mrt['line_count'] = mrt.groupby('NAME')['NAME'].transform('size')
    return mrt


_MONTH_NUM = {m: f'{i:02d}' for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start=1)}


def build_stock_index(stocks: pd.DataFrame) -> pd.Series:
    """Equal-weighted, return-normalized monthly index from sg-stock-prices.csv.

    Only uses symbols with the maximum row count (full daily history across the whole
    date range, 41 of the file's 77 symbols) — the rest have partial coverage (later
    IPOs/listings, delistings, etc.) and mixing them into one average, as an earlier
    pass did, is not a meaningful index (that earlier version got a near-zero, useless
    correlation). Each symbol's price is normalized to start at 1.0 before averaging,
    so the index isn't dominated by whichever stock happens to have a higher absolute
    price (e.g. DBS at ~$30 vs HPH Trust at ~$0.15).
    """
    counts = stocks['SYMBOL'].value_counts()
    full_history_symbols = counts[counts == counts.max()].index
    sub = stocks[stocks['SYMBOL'].isin(full_history_symbols)].copy()
    sub['DATE'] = pd.to_datetime(sub['DATE'])
    sub = sub.sort_values(['SYMBOL', 'DATE'])
    first_price = sub.groupby('SYMBOL')['ADJUSTED_CLOSE'].transform('first')
    sub['norm_price'] = sub['ADJUSTED_CLOSE'] / first_price
    daily_index = sub.groupby('DATE')['norm_price'].mean()
    monthly_index = daily_index.resample('MS').mean()
    monthly_index.index = monthly_index.index.strftime('%Y-%m')
    return monthly_index.rename('stock_index')


def build_coe_index(coe: pd.DataFrame) -> pd.Series:
    """Monthly mean COE quota premium from sg-coe-prices.csv, keyed to 'YYYY-MM'.

    An earlier pass computed this and never actually joined/correlated it against
    rent — this is that missing step, plus fixing MONTH ('jan'/'feb'/...) into a
    proper zero-padded numeric month for the 'YYYY-MM' key.
    """
    coe = coe.copy()
    coe['QUOTA_PREMIUM'] = coe['QUOTA_PREMIUM'].replace('[$,]', '', regex=True).astype(float)
    coe['year_month'] = coe['YEAR'].astype(str) + '-' + coe['MONTH'].str.lower().map(_MONTH_NUM)
    return coe.groupby('year_month')['QUOTA_PREMIUM'].mean().rename('coe_premium')


def add_macro_features(df: pd.DataFrame, stock_index: pd.Series, coe_index: pd.Series) -> pd.DataFrame:
    """Merge monthly macro indicators onto df by RENT_APPROVAL_DATE.

    Unlike every other feature here (all trailing/backward-looking from train's own
    history), sg-stock-prices.csv and sg-coe-prices.csv both extend through test's
    date range (stocks to 2026-07, COE to 2026-08) — so these give the model real
    economic conditions for test's months instead of forcing it to blindly
    extrapolate a trend from training data alone.
    """
    df = df.copy()
    df = df.merge(stock_index, left_on='RENT_APPROVAL_DATE', right_index=True, how='left')
    df = df.merge(coe_index, left_on='RENT_APPROVAL_DATE', right_index=True, how='left')
    return df


def build_subzone_momentum(source_df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Trailing rolling-average rent per SUBZONE, computed from source_df only, for
    use as a 'recent local market level' feature. See progress-report.md "Applying
    the top two suggested next steps" for the full result — TESTED, NOT ADOPTED
    (redundant with SUBZONE's own native categorical signal; doesn't fix the
    low-rent over-prediction problem it was built to address).

    Monthly mean rent per SUBZONE, then shift(1) before the rolling window so a
    month's value never includes itself (avoids circularity even within source_df,
    e.g. a sparse subzone/month where the row IS most of that month's average).

    IMPORTANT for leakage-safe use: source_df must be only the data actually known
    at prediction time — train_fold when validating (so val_fold's own rows are
    never used to build the feature applied to it), train_feat (all of train.csv)
    when building the feature for test_feat. Passing val_fold or test_feat as
    source_df would leak unknown-at-prediction-time information.
    """
    monthly = source_df.groupby(['SUBZONE', 'RENT_APPROVAL_DATE'])['MONTHLY_RENT'].mean().reset_index()
    monthly = monthly.sort_values(['SUBZONE', 'RENT_APPROVAL_DATE'])
    monthly['momentum'] = (monthly.groupby('SUBZONE')['MONTHLY_RENT']
                            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))
    return monthly[['SUBZONE', 'RENT_APPROVAL_DATE', 'momentum']]


def add_momentum_feature(df: pd.DataFrame, momentum_table: pd.DataFrame, global_fallback: float) -> pd.DataFrame:
    """Merge build_subzone_momentum's trailing values onto df by (SUBZONE, month).
    For months beyond momentum_table's coverage (forecasting past known history —
    the norm for most of test.csv, whose horizon extends 16 months past train's last
    date), forward-fills the last known value per subzone; global_fallback covers a
    subzone with no history in momentum_table at all."""
    df = df.copy()
    mt = momentum_table.sort_values(['SUBZONE', 'RENT_APPROVAL_DATE'])
    merged = df.merge(mt, on=['SUBZONE', 'RENT_APPROVAL_DATE'], how='left')
    last_known = mt.groupby('SUBZONE').last()['momentum']
    merged['momentum'] = merged['momentum'].fillna(merged['SUBZONE'].map(last_known))
    merged['momentum'] = merged['momentum'].fillna(global_fallback)
    df['momentum'] = merged['momentum'].values
    return df


def impute_sentinel_values(df: pd.DataFrame, columns: list, sentinel=-1,
                            group_col: str = 'SUBZONE') -> pd.DataFrame:
    """Replace a sentinel/placeholder value (e.g. -1, found in sg-hdb-block.csv's
    MAX_FLOOR/YEAR_COMPLETED for 11 of 9,464 blocks — data-quality-report.md section
    A; these were being fed as literal numeric values into the model, undetected by
    any isna() check) with the group_col median computed from the OTHER, non-sentinel
    rows. Falls back to the global median if a group has no other valid rows at all
    (doesn't happen for the known sentinel blocks — every affected SUBZONE has 25+
    other valid blocks, confirmed in data-quality-report.md — but included for
    robustness against future data)."""
    df = df.copy()
    for col in columns:
        is_sentinel = df[col] == sentinel
        if not is_sentinel.any():
            continue
        valid = df.loc[~is_sentinel]
        group_median = valid.groupby(group_col)[col].median()
        global_median = valid[col].median()
        df.loc[is_sentinel, col] = df.loc[is_sentinel, group_col].map(group_median).fillna(global_median)
    return df


def join_hdb_geo(df: pd.DataFrame, hdb_block: pd.DataFrame) -> pd.DataFrame:
    """Attach LATITUDE/LONGITUDE/MAX_FLOOR/YEAR_COMPLETED/... from sg-hdb-block.csv.

    df must already have normalized TOWN/BLOCK/STREET (see clean_join_keys).
    EDA confirmed a 100% match rate on both train and test after normalization.

    Imputes the MAX_FLOOR/YEAR_COMPLETED sentinel (see impute_sentinel_values) on the
    reference table itself, BEFORE the join — so train_feat and test_feat both
    inherit the same corrected values for any given block, rather than each
    recomputing (and potentially disagreeing on) their own imputation.
    """
    hdb_block = hdb_block.copy()
    for c in JOIN_KEYS:
        hdb_block[c] = hdb_block[c].astype(str).str.strip().str.lower()
    hdb_block = hdb_block.drop_duplicates(subset=JOIN_KEYS)
    hdb_block = impute_sentinel_values(hdb_block, ['MAX_FLOOR', 'YEAR_COMPLETED'])
    return df.merge(hdb_block, on=JOIN_KEYS, how='left', suffixes=('', '_hdb'))


def build_features(df: pd.DataFrame, hdb_block: pd.DataFrame, mrt: pd.DataFrame,
                    malls: pd.DataFrame, stock_index: pd.Series = None,
                    coe_index: pd.Series = None) -> pd.DataFrame:
    """Full, EDA-validated feature pipeline. Apply identically to train and test.

    Order matters: FLAT_TYPE/casing must be normalized before the join/dedup
    steps that depend on them (see initial-eda-inference.md, section 3).

    stock_index/coe_index are optional (pass both, or neither) — see
    add_macro_features for why they're worth including.
    """
    df = df.copy()
    df['FLAT_TYPE'] = normalize_flat_type(df['FLAT_TYPE'])
    df = clean_join_keys(df)
    df = add_lease_features(df)
    df = add_flat_type_ordinal(df)
    df = join_hdb_geo(df, hdb_block)
    df = add_nearest_distance(df, mrt, out_col='dist_mrt_km')
    df = add_nearest_distance(df, malls, out_col='dist_mall_km')
    if stock_index is not None and coe_index is not None:
        df = add_macro_features(df, stock_index, coe_index)
    return df


def load_and_build(data_dir: str = '../data', include_macro: bool = True):
    """Convenience loader: reads train/test + auxiliary files and runs build_features on both."""
    aux_dir = f'{data_dir}/auxiliary'
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    hdb_block = pd.read_csv(f'{aux_dir}/sg-hdb-block.csv')
    mrt = pd.read_csv(f'{aux_dir}/sg-mrt-stations.csv')
    malls = pd.read_csv(f'{aux_dir}/sg-shopping-malls.csv')

    stock_index = coe_index = None
    if include_macro:
        stocks = pd.read_csv(f'{aux_dir}/sg-stock-prices.csv')
        coe = pd.read_csv(f'{aux_dir}/sg-coe-prices.csv')
        stock_index = build_stock_index(stocks)
        coe_index = build_coe_index(coe)

    train_feat = build_features(train, hdb_block, mrt, malls, stock_index, coe_index)
    test_feat = build_features(test, hdb_block, mrt, malls, stock_index, coe_index)

    # EDA step 3: 216 duplicate rows only surface after FLAT_TYPE normalization.
    train_feat = train_feat.drop_duplicates()

    return train_feat, test_feat
