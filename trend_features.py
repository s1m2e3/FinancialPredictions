"""Trend features from moving averages, as known at the OPEN of each day (closes up to the
day before), for the main universe (data/trend_features.npz) and the growth pool.

"DIFFED" MOVING AVERAGES -- short against long, in logs so they compare across stocks:
    ma_5_21, ma_21_63, ma_50_200   log(MA_short / MA_long) of the adjusted close; > 0 when
                                   the short-term trend runs above the long-term one (a
                                   crossover is this changing sign)
    px_ma_50, px_ma_200            log(close / MA): how far the price stands above its trend
    ma_slope_21, ma_slope_63       log(MA_t / MA_{t-k}) with k = 5 and 21 days: is the trend
                                   itself rising, and how fast
and each one's percentile rank among that day's universe (rank_*): the level of a crossover
drifts with the market, its rank against the other stocks does not. A moving average needs
its full window of prices, so a young stock has none until it has the history (NaN).

    python trend_features.py       builds data/trend_features.npz from the panel
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "trend_features.npz")
CROSS = ((5, 21), (21, 63), (50, 200))
LEVEL = (50, 200)
SLOPE = ((21, 5), (63, 21))                      # (moving-average window, days of change)
RAW = ([f"ma_{s}_{l}" for s, l in CROSS] + [f"px_ma_{w}" for w in LEVEL]
       + [f"ma_slope_{w}" for w, _ in SLOPE])
NAMES = RAW + ["rank_" + n for n in RAW]


def trend(close, universe=None):
    """{name: (T, N)} from adjusted closes (T, N) (NaN where a stock has no price), shifted a
    day so row t uses closes up to t-1; ranks within `universe` (T, N) bool when given."""
    import pandas as pd
    c = pd.DataFrame(np.where(close > 0, close, np.nan))
    ma = {w: c.rolling(w, min_periods=w).mean() for w in sorted({w for p in CROSS for w in p} | set(LEVEL)
                                                                 | {w for w, _ in SLOPE})}
    out = {}
    for s, l in CROSS:
        out[f"ma_{s}_{l}"] = np.log(ma[s] / ma[l])
    for w in LEVEL:
        out[f"px_ma_{w}"] = np.log(c / ma[w])
    for w, k in SLOPE:
        out[f"ma_slope_{w}"] = np.log(ma[w] / ma[w].shift(k))
    out = {n: v.shift(1).to_numpy() for n, v in out.items()}                    # known at the open
    for n in RAW:
        x = out[n] if universe is None else np.where(universe, out[n], np.nan)
        out["rank_" + n] = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()
    return out


def build(panel=None):
    from stocks_data import load_panel
    p = panel or load_panel()
    f = trend(p.close, p.universe)
    F = np.stack([f[n] for n in NAMES], axis=2).astype(np.float32)
    np.savez_compressed(CACHE, F=F, names=np.array(NAMES), tickers=np.array(p.tickers),
                        dates=p.dates.values.astype("datetime64[D]"))
    ok = np.isfinite(F[:, :, 0]) & p.universe
    print(f"trend features {F.shape} -> {CACHE}; share of the universe with ma_5_21: "
          f"{ok.sum() / max(p.universe.sum(), 1):.1%}, with ma_50_200: "
          f"{(np.isfinite(F[:, :, NAMES.index('ma_50_200')]) & p.universe).sum() / max(p.universe.sum(), 1):.1%}")
    return F


def load():
    with np.load(CACHE) as z:
        return z["F"], [str(n) for n in z["names"]]


if __name__ == "__main__":
    build()
