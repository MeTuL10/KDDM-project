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


def build_address_momentum(source_df: pd.DataFrame) -> pd.DataFrame:
    """Trailing, address-level (TOWN+BLOCK+STREET — finer than SUBZONE) rent_per_sqm
    history, for use as a per-building 'usual price level' signal.

    Tested and ADOPTED — see progress-report.md "Address-level rent-level feature".
    Unlike build_subzone_momentum (tested, NOT adopted — redundant with SUBZONE's own
    categorical splits), a per-building signal captures something a SUBZONE-wide
    category boundary can't (e.g. one block running consistently cheaper/pricier than
    its subzone's average), and gave a real, independent RMSE gain.

    Same leakage discipline as build_subzone_momentum: source_df must be only data
    actually known at prediction time (train_fold when validating, train_feat for the
    final fit). Aggregates to (address, month) first, then takes an EXPANDING (not a
    fixed rolling window — most addresses see too few transactions/month for a window
    to be meaningful) trailing mean, shift(1) so a month never includes itself even
    within source_df — the same "circularity" concern build_subzone_momentum's
    docstring describes, at a sparser grain where it would matter even more (an
    address can BE its own month's entire average when it has only one transaction).

    Returns one row per (JOIN_KEYS..., RENT_APPROVAL_DATE) with the trailing mean and
    trailing transaction count (the count feeds add_address_feature's shrinkage).
    """
    df = source_df.copy()
    df['rent_per_sqm'] = df['MONTHLY_RENT'] / df['FLOOR_AREA_SQM']
    monthly = (df.groupby(JOIN_KEYS + ['RENT_APPROVAL_DATE'])['rent_per_sqm']
                 .agg(['sum', 'count']).reset_index())
    monthly = monthly.sort_values(JOIN_KEYS + ['RENT_APPROVAL_DATE'])
    grp = monthly.groupby(JOIN_KEYS)
    prior_sum = grp['sum'].transform(lambda s: s.shift(1).cumsum())
    prior_n = grp['count'].transform(lambda s: s.shift(1).cumsum())
    monthly['addr_trailing_mean'] = prior_sum / prior_n
    monthly['addr_trailing_n'] = prior_n.fillna(0)
    return monthly[JOIN_KEYS + ['RENT_APPROVAL_DATE', 'addr_trailing_mean', 'addr_trailing_n']]


def rent_per_sqm_priors(source_df: pd.DataFrame):
    """SUBZONE-mean and global-mean rent_per_sqm from source_df — the shrinkage
    target add_address_feature falls back to when an address has little/no trailing
    history of its own. Same source_df leakage rule as build_address_momentum
    (train_fold when validating, train_feat for the final fit)."""
    rent_per_sqm = source_df['MONTHLY_RENT'] / source_df['FLOOR_AREA_SQM']
    subzone_mean = rent_per_sqm.groupby(source_df['SUBZONE']).mean()
    return subzone_mean, rent_per_sqm.mean()


def add_address_feature(df: pd.DataFrame, momentum_table: pd.DataFrame, subzone_mean: pd.Series,
                         global_fallback: float, m: float = 5.0) -> pd.DataFrame:
    """Merge build_address_momentum's trailing values onto df by address+month, then
    shrink toward the SUBZONE mean via an m-estimate (same smoothing idea as
    archived_modules/encoding.py's target encoding) — most addresses see only a handful of
    transactions across the whole dataset (median ~13 over ~4 years), so the raw
    trailing mean alone would be noisy for anything but the most-transacted blocks.

    m is pseudo-observations pulled toward the SUBZONE prior; tested robust across
    m=1..100 (all beat the no-feature baseline on val_fold) — m=5 ships as a
    reasonable middle ground, not squeezed to whichever value looked best once.

    Forward-fill rule, and why it's leakage-safe: a row's exact (address, month) is
    looked up in momentum_table first. If that exact key exists (df IS source_df, or
    shares one of its months), its value is used as-is *even if NaN* — NaN there
    means "this address's genuine first-ever appearance," which must NOT be
    forward-filled from a later month of the same address (that would leak future
    information backwards). Only when the key is genuinely absent (df has a month
    momentum_table never saw at all — test.csv's whole date range past train's last
    known month, the norm for most of it, same situation add_momentum_feature's
    docstring describes) does it fall back to that address's LAST known state, which
    is safe precisely because it only happens for months later than anything in
    momentum_table.
    """
    df = df.copy()
    merged = df.merge(momentum_table, on=JOIN_KEYS + ['RENT_APPROVAL_DATE'],
                       how='left', indicator=True)

    last_known = (momentum_table.sort_values(JOIN_KEYS + ['RENT_APPROVAL_DATE'])
                                 .groupby(JOIN_KEYS)[['addr_trailing_mean', 'addr_trailing_n']]
                                 .last().reset_index()
                                 .rename(columns={'addr_trailing_mean': 'addr_last_mean',
                                                   'addr_trailing_n': 'addr_last_n'}))
    merged = merged.merge(last_known, on=JOIN_KEYS, how='left')

    no_exact_month = merged['_merge'] == 'left_only'
    merged.loc[no_exact_month, 'addr_trailing_mean'] = merged.loc[no_exact_month, 'addr_last_mean']
    merged.loc[no_exact_month, 'addr_trailing_n'] = merged.loc[no_exact_month, 'addr_last_n']
    merged['addr_trailing_n'] = merged['addr_trailing_n'].fillna(0)

    prior = merged['SUBZONE'].map(subzone_mean).fillna(global_fallback)
    own = merged['addr_trailing_mean'].fillna(prior)
    n = merged['addr_trailing_n']
    df['addr_level_feat'] = ((n * own + m * prior) / (n + m)).values
    return df


MEDIAN_RENT_FLAT_TYPE = {'1-rm': '1-room', '2-rm': '2-room', '3-rm': '3-room',
                          '4-rm': '4-room', '5-rm': '5-room', 'exec': 'executive'}


def to_quarter(date_str: str) -> str:
    """'2025-03' -> '2025-Q1', the quarter key used by sg-hdb-median-rent.csv."""
    year, month = date_str.split('-')
    return f'{year}-Q{(int(month) - 1) // 3 + 1}'


def build_median_anchor_table(median_df: pd.DataFrame, interp_limit: int = 2):
    """Turn sg-hdb-median-rent.csv (official quarterly median rent by town and flat
    type) into two lookup tables for add_median_rent_anchor.

    The root pipeline never used this file; expt-2 built its market anchor from it and
    its leaderboard score ran 8-18 points below the root's at similar local scores (see
    the RMSE bottleneck report). Data notes: the file is populated through 2026-Q2 but
    only ~55% of its 156 town x flat-type cells per quarter hold a value; it spells
    'QUEENSTOWN ' with a trailing space in recent quarters, so town names MUST be
    stripped before matching (an unstripped check wrongly suggested Queenstown had no
    data after 2016); and it repeats some rows. With that cleaning, 96.6% of train rows
    and 96.9% of test rows match an exact cell, so the fallbacks cover ~3%:

      level 0  exact town x flat type x quarter median
      level 1  linear interpolation within the same cell, up to interp_limit quarters
               (edges are held flat for the same limit)
      level 2  cross-town median for that flat type and quarter x the town's own
               historical ratio to it (fit_anchor_town_ratios)
      level 3  cross-town median alone (a town/flat-type pair with no ratio)
      level 4  caller-supplied global fallback (add_median_rent_anchor)

    Returns (town_table, cross_table). Column names match the model frames (TOWN,
    FLAT_TYPE) so they merge directly.
    """
    m = median_df.copy()
    m['TOWN'] = m['town'].str.strip().str.lower()
    m['FLAT_TYPE'] = m['flat_type'].str.strip().str.lower().map(MEDIAN_RENT_FLAT_TYPE)
    m['median_rent'] = pd.to_numeric(m['median_rent'], errors='coerce')
    m = m.dropna(subset=['FLAT_TYPE'])
    # The file repeats some (town, flat type, quarter) rows (320 duplicate rows; the
    # ones inspected were exact repeats, e.g. 2020-Q1). Averaging is a no-op for exact
    # repeats and keeps the pivot below well-defined.
    m = m.groupby(['TOWN', 'FLAT_TYPE', 'quarter'], as_index=False)['median_rent'].mean()
    exact = m.pivot(index=['TOWN', 'FLAT_TYPE'], columns='quarter', values='median_rent')
    exact = exact.reindex(sorted(exact.columns), axis=1)
    filled = exact.interpolate(axis=1, limit=interp_limit, limit_direction='both')

    town_table = (filled.stack().rename('anchor_town').reset_index()
                  .merge(exact.stack().rename('exact_value').reset_index(),
                         on=['TOWN', 'FLAT_TYPE', 'quarter'], how='left'))
    town_table['anchor_level'] = town_table['exact_value'].notna().map({True: 0, False: 1})
    town_table = town_table.drop(columns='exact_value')

    cross = filled.groupby(level='FLAT_TYPE').median()
    cross = cross.ffill(axis=1).bfill(axis=1)
    cross_table = cross.stack().rename('cross_town').reset_index()
    return town_table, cross_table


def _quarter_keys(df: pd.DataFrame, max_quarter: str) -> pd.Series:
    """Per-row quarter, clamped to the last quarter the median file covers (test.csv
    runs into 2026-Q3; those rows borrow 2026-Q2's level)."""
    q = df['RENT_APPROVAL_DATE'].map(to_quarter)
    return q.where(q <= max_quarter, max_quarter)


def fit_anchor_town_ratios(source_df: pd.DataFrame, cross_table: pd.DataFrame) -> pd.Series:
    """Median of (rent / cross-town median) per TOWN x FLAT_TYPE, from source_df only —
    the level-2 fallback multiplier for cells the official file leaves empty. Same
    leakage rule as build_address_momentum: train_fold when validating, train_feat for
    the final fit."""
    k = pd.DataFrame({'TOWN': source_df['TOWN'].values, 'FLAT_TYPE': source_df['FLAT_TYPE'].values,
                      'quarter': _quarter_keys(source_df, cross_table['quarter'].max()).values,
                      'rent': source_df['MONTHLY_RENT'].values})
    k = k.merge(cross_table, on=['FLAT_TYPE', 'quarter'], how='left')
    k['ratio'] = k['rent'] / k['cross_town']
    return k.groupby(['TOWN', 'FLAT_TYPE'])['ratio'].median().rename('town_ratio')


def add_median_rent_anchor(df: pd.DataFrame, town_table: pd.DataFrame, cross_table: pd.DataFrame,
                            town_ratios: pd.Series, global_fallback: float) -> pd.DataFrame:
    """Adds 'anchor' (SGD, the market-level rent for the row's town, flat type and
    quarter) and 'anchor_level' (0-4, which rule produced it; see
    build_median_anchor_table). The anchor is provided auxiliary data, not derived from
    the row's own rent, except via town_ratios, which come from source_df's rents only."""
    max_q = cross_table['quarter'].max()
    k = pd.DataFrame({'TOWN': df['TOWN'].values, 'FLAT_TYPE': df['FLAT_TYPE'].values,
                      'quarter': _quarter_keys(df, max_q).values})
    k = (k.merge(town_table, on=['TOWN', 'FLAT_TYPE', 'quarter'], how='left')
          .merge(cross_table, on=['FLAT_TYPE', 'quarter'], how='left')
          .merge(town_ratios.reset_index(), on=['TOWN', 'FLAT_TYPE'], how='left'))
    anchor = k['anchor_town']
    level = k['anchor_level']
    by_ratio = k['cross_town'] * k['town_ratio']
    use_ratio = anchor.isna() & by_ratio.notna()
    anchor, level = anchor.where(~use_ratio, by_ratio), level.where(~use_ratio, 2)
    use_cross = anchor.isna() & k['cross_town'].notna()
    anchor, level = anchor.where(~use_cross, k['cross_town']), level.where(~use_cross, 3)
    use_global = anchor.isna()
    anchor, level = anchor.where(~use_global, global_fallback), level.where(~use_global, 4)
    out = df.copy()
    out['anchor'] = anchor.values
    out['anchor_level'] = level.astype(int).values
    return out


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
