"""
expt-5 extras: block-level resale signals (external data) and neighbour-block priors.

E1  resale: dense resale market per block (expt-4's data.gov.sg download, 2017-01..2025-02), independent of
    rental labels. Per block: premium of price per sqm over its town / flat type / quarter norm (shrunk),
    block x type premium, mean storey, transaction count. Only resale rows BEFORE the holdout cutoff are used.
E2  neighbours: each block x flat type gets the rental residual (rent - benchmark) of its nearest OTHER
    blocks of the same flat type (distance-weighted mean of the 5 and 15 nearest, mean distance to the 5
    nearest), computed from the training rows only and excluding the row's own block.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt5_lib import *
from sklearn.neighbors import BallTree

ABBR = {'AVE': 'AVENUE', 'ST': 'STREET', 'RD': 'ROAD', 'DR': 'DRIVE', 'CRES': 'CRESCENT', 'CL': 'CLOSE', 'CTRL': 'CENTRAL',
        'NTH': 'NORTH', 'STH': 'SOUTH', 'BT': 'BUKIT', 'JLN': 'JALAN', 'LOR': 'LORONG', 'TER': 'TERRACE', 'PL': 'PLACE',
        'GDNS': 'GARDENS', 'HTS': 'HEIGHTS', 'PK': 'PARK', 'UPP': 'UPPER', 'TG': 'TANJONG', 'KG': 'KAMPONG',
        "C'WEALTH": 'COMMONWEALTH', 'MKT': 'MARKET', 'LK': 'LINK', 'BLVD': 'BOULEVARD'}


def _street(s):
    return ' '.join(ABBR.get(w, w) for w in str(s).strip().upper().split()).lower()


def load_resale():
    r = pd.read_parquet(os.path.join(ROOT, 'expt-4', 'data', 'hdb_resale_prices_2017_2025.parquet'))
    r['town_c'] = r['town'].str.strip().str.lower().replace({'central area': 'central'})
    r['key'] = r['town_c'] + '|' + r['block'].astype(str).str.strip().str.lower() + '|' + r['street_name'].map(_street)
    r['ft'] = r['flat_type'].str.strip().str.lower().str.replace(' ', '-')
    mid = r['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)').astype(float)
    r['storey_mid'] = (mid[0] + mid[1]) / 2
    r['q'] = r['month'].str.slice(0, 4) + '-Q' + ((r['month'].str.slice(5, 7).astype(int) - 1) // 3 + 1).astype(str)
    r['lpps'] = np.log(r['price_per_sqm'])
    return r


RESALE = load_resale()


def resale_block_features(cutoff):
    r = RESALE[RESALE['month'] < cutoff].copy()
    r['resid'] = r['lpps'] - r.groupby(['town_c', 'ft', 'q'])['lpps'].transform('mean')
    g = r.groupby('key').agg(n=('resid', 'size'), s=('resid', 'sum'), resale_mean_storey=('storey_mid', 'mean'))
    g['resale_prem_block'] = g['s'] / (g['n'] + 10.0)
    g['resale_n_block'] = np.log1p(g['n'])
    bt = r.groupby(['key', 'ft']).agg(n=('resid', 'size'), s=('resid', 'sum')).reset_index()
    bt = bt.merge(g[['resale_prem_block']], left_on='key', right_index=True)
    bt['resale_prem_bt'] = (bt['s'] + 5.0 * bt['resale_prem_block']) / (bt['n'] + 5.0)
    return g[['resale_prem_block', 'resale_n_block', 'resale_mean_storey']], bt[['key', 'ft', 'resale_prem_bt']]


def add_resale(df, g, bt):
    key = (df['town_clean'] + '|' + df['block_clean'] + '|' + df['street_clean']).values
    out = df.copy()
    for c in g.columns:
        out[c] = g[c].reindex(key).values
    k = pd.DataFrame({'key': key, 'ft': df['flat_type_clean'].values}).merge(bt, on=['key', 'ft'], how='left')
    out['resale_prem_bt'] = k['resale_prem_bt'].values
    return out


def neighbour_features(tr, va):
    """Distance-weighted rental residuals of the nearest other blocks (same flat type), from tr only."""
    res = tr['MONTHLY_RENT'] - tr['median_rent_benchmark']
    gm = res.mean()
    bt = (tr.assign(res=res).groupby(['block_id', 'flat_type_clean'])
            .agg(n=('res', 'size'), s=('res', 'sum'), lat=('LATITUDE', 'first'), lon=('LONGITUDE', 'first'))
            .reset_index())
    bt['mu'] = (bt['s'] + 5.0 * gm) / (bt['n'] + 5.0)
    keys = pd.concat([tr, va])[['block_id', 'flat_type_clean', 'LATITUDE', 'LONGITUDE']].drop_duplicates(['block_id', 'flat_type_clean'])
    rows = []
    for ft, pts in bt.groupby('flat_type_clean'):
        q = keys[keys['flat_type_clean'] == ft]
        tree = BallTree(np.radians(pts[['lat', 'lon']].values), metric='haversine')
        kq = min(30, len(pts))
        dist, idx = tree.query(np.radians(q[['LATITUDE', 'LONGITUDE']].values), k=kq)
        dist = dist * 6371.0
        pts_bid = pts['block_id'].to_numpy(dtype=object)      # plain numpy: Arrow-backed strings reject 2-D indexing
        valid = pts_bid[idx] != q['block_id'].to_numpy(dtype=object)[:, None]
        rank = np.cumsum(valid, axis=1)
        mu = pts['mu'].values[idx]
        feats = {}
        for k in (5, 15):
            sel = valid & (rank <= k)
            w = sel / (dist + 0.05)
            feats[f'nb_mu{k}'] = np.where(w.sum(1) > 0, (w * mu).sum(1) / np.maximum(w.sum(1), 1e-9), np.nan)
            if k == 5:
                feats['nb_dist5'] = np.where(sel.sum(1) > 0, (dist * sel).sum(1) / np.maximum(sel.sum(1), 1), np.nan)
        rows.append(pd.DataFrame({'block_id': q['block_id'].values, 'flat_type_clean': ft, **feats}))
    nb = pd.concat(rows, ignore_index=True)
    out = []
    for d in (tr, va):
        m = d[['block_id', 'flat_type_clean']].merge(nb, on=['block_id', 'flat_type_clean'], how='left')
        d = d.copy()
        for c in ['nb_mu5', 'nb_mu15', 'nb_dist5']:
            d[c] = m[c].values
        out.append(d)
    return out
