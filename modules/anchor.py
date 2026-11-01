"""
Honest (forecast-time) median-rent anchor.

The official median-rent file runs to 2026-Q2, so its 2025-Q2..2026-Q2 values are aggregates of test-period
rents: information that would not exist at forecast time. Honest anchor: freeze every (town, flat type)
benchmark at the cutoff quarter and carry it forward, multiplied by exp(g * quarters_after_cutoff), where g is
the mean quarterly log growth of a national rent index over the 4 quarters up to the cutoff (index = geometric
mean of the 3-room..executive national medians, the only types published every quarter). Quarters up to the
cutoff keep their published values, so train rows are never touched.
(Ported from expt-5/src/honest_lib.py.)
"""

from functools import lru_cache

import numpy as np
import pandas as pd

import features as fe

INDEX_TYPES = ['3-room', '4-room', '5-room', 'executive']


@lru_cache(maxsize=1)
def benchmark_table() -> pd.DataFrame:
    """The V2 quarterly benchmark table (fallbacks included), built once."""
    return fe.build_median_rent_benchmark_v2()


def qnum(q):
    y, n = str(q).split('-Q')
    return int(y) * 4 + int(n) - 1


def qstr(num):
    return f'{num // 4}-Q{num % 4 + 1}'


def national_index() -> pd.Series:
    nat = benchmark_table().drop_duplicates(['quarter', 'flat_type_clean']).pivot(
        index='quarter', columns='flat_type_clean', values='nat_median_rent_benchmark')
    return np.exp(np.log(nat[INDEX_TYPES]).mean(axis=1))


def drift(cutoff_q, n=4) -> float:
    """Mean quarterly log growth of the national index over the n quarters up to cutoff_q."""
    idx = national_index()
    return float((np.log(idx[cutoff_q]) - np.log(idx[qstr(qnum(cutoff_q) - n)])) / n)


def make_honest(df: pd.DataFrame, cutoff_q: str) -> pd.DataFrame:
    """Replace the benchmark of rows after cutoff_q by the cutoff-quarter value times exp(drift * k)."""
    d = df.copy()
    k = d['approval_quarter'].map(qnum) - qnum(cutoff_q)
    mask = (k > 0).values
    c = benchmark_table()
    c = c[c['quarter'] == cutoff_q][['town_clean', 'flat_type_clean', 'median_rent_benchmark', 'nat_median_rent_benchmark']]
    c = c.rename(columns={'median_rent_benchmark': 'b_c', 'nat_median_rent_benchmark': 'n_c'})
    m = d[['town_clean', 'flat_type_clean']].merge(c, on=['town_clean', 'flat_type_clean'], how='left')
    assert not m.loc[mask, ['b_c', 'n_c']].isna().any().any(), 'missing cutoff-quarter benchmark'
    f = np.exp(drift(cutoff_q) * k.values[mask])
    d.loc[mask, 'median_rent_benchmark'] = m.loc[mask, 'b_c'].values * f
    d.loc[mask, 'nat_median_rent_benchmark'] = m.loc[mask, 'n_c'].values * f
    d.loc[mask, 'median_rent_ratio'] = d.loc[mask, 'median_rent_benchmark'] / d.loc[mask, 'nat_median_rent_benchmark']
    return d
