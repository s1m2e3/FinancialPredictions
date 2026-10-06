"""The hourly multi-asset panel on one UTC hourly grid, with the costs, carry and features the
hourly world (macro_env.py) trades on. Universes (dukascopy_prices.UNIVERSES, --universe):
"fx" (default, 43 instruments: 32 G10 pairs, 9 emerging-market pairs, gold, silver -- hourly
bid and ask from dukascopy_feed.py) and "macro12" (the first plan, with the index and
commodity CFDs). The panel holds the instruments whose hourly bid AND ask are merged
(`available`), so a test can start before a download finishes.

GRID. The hours in which EUR/USD has a bar (Sunday evening to Friday evening, UTC). An
instrument without a bar in an hour cannot be traded in it; its last price carries its value
until it trades again, so a gap at the reopen is a real move.

PRICES AND COSTS. Every bar has a bid and an ask. The simulator trades at the OPEN of a bar at
the mid plus or minus half the spread at that open -- the spread actually quoted in that hour,
which widens at the daily rollover and in thin hours -- and marks positions at mid opens.

CARRY. Holding a position overnight earns or pays interest: a currency pair the 3-month rate of
the currency held long minus that of the one held short (dollar / offshore yuan: China's rate),
gold, silver and CFDs minus the US rate (they are financed at it). Monthly OECD rates on FRED,
used two months late so nothing is used before it would have been published.

FEATURES, each known at the decision -- from bars that CLOSED before the bar the order executes at:
  hourly  ret_1h, ret_4h, ret_24h, ret_120h   log return over the last k hours / (hourly vol sqrt k)
          vol_ratio_24h                       realised vol of the last 24 hours / the long hourly vol
          range_pos_120h                      where the price sits in its last week's range, 0..1
          spread_ratio                        the last quoted spread / its median of the last 20 days
          hour_utc, weekday                   the clock (sessions: Asia, Europe, US)
  daily   trend_20d, trend_60d, trend_250d    log(price / daily close k days ago) / (daily vol sqrt k),
                                              daily closes = the last hourly mid close of each UTC day
          vol_ann                             the daily-return volatility forecast (EWMA, 20-day half-life), annualised
          carry                               per year the LONG position earns from financing (short: minus it)
  fx universe:
    kind    is_metal, is_em (an emerging-market currency in the pair), usd_side (+1 long the dollar
            when long the pair, -1 short it, 0 a cross)
    ranks   rank_carry, rank_trend_60d, rank_trend_250d, rank_ret_24h, rank_ret_120h: the
            instrument's percentile among those tradable in that hour -- carry and momentum
            across currencies are the documented currency factors
    market  usd_ret_24h, usd_trend_20d (the dollar against the pairs that hold it), gold_ret_24h,
            us500_ret_1d, us500_trend_20d (the S&P's last complete day, Dukascopy's daily bars)
  macro12: is_fx, is_equity, is_metal, is_energy and us500_ret_24h, us500_trend_20d,
           us500_vol_ratio, usd_ret_24h, usd_trend_20d
"""
import os

import numpy as np
import pandas as pd

from dukascopy_prices import DAILY_EXTRA, DIR, UNIVERSE, UNIVERSES

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID_START = "2010-01-01"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
VAL_END = "2021-12-31"
HOURLY = ("ret_1h", "ret_4h", "ret_24h", "ret_120h", "vol_ratio_24h", "range_pos_120h", "spread_ratio",
          "hour_utc", "weekday")
DAILY = ("trend_20d", "trend_60d", "trend_250d", "vol_ann", "carry")
RANKED = ("carry", "trend_60d", "trend_250d", "ret_24h", "ret_120h")
EM = {"HUF", "MXN", "PLN", "ZAR", "CNH"}
FEATURES_OF = {
    "fx": HOURLY + DAILY + ("is_metal", "is_em", "usd_side") + tuple(f"rank_{k}" for k in RANKED)
          + ("usd_ret_24h", "usd_trend_20d", "gold_ret_24h", "us500_ret_1d", "us500_trend_20d"),
    "macro12": HOURLY + DAILY + ("is_fx", "is_equity", "is_metal", "is_energy")
               + ("us500_ret_24h", "us500_trend_20d", "us500_vol_ratio", "usd_ret_24h", "usd_trend_20d"),
}
FEATURES = FEATURES_OF[UNIVERSE]
VOL_HALFLIFE_H = 240          # hourly vol: EWMA over the traded hours, half-life ~2 weeks
VOL_HALFLIFE_D = 20           # daily vol: EWMA, half-life 20 trading days
RATE_LAG_MONTHS = 2           # the OECD monthly average of month m is used from month m + 2
MAX_REL_SPREAD = 0.02         # a quote with the ask above the bid by more than 2% is a bad print: no trade
Z_CLIP = 10.0


def legs(ident):
    """(currency held long, currency held short) by a LONG position; gold / silver / CFDs: (None, "USD")."""
    base, quote = ident[:3].upper(), ident[3:6].upper()
    if base in ("XAU", "XAG") or len(ident) != 6:
        return None, "USD"
    return base, quote


def _checked(ident, side):
    """The hourly series is merged AND its coverage was checked (marker v2: dukascopy_prices, v3:
    dukascopy_feed); an older marker belongs to a file that may stop early (EUR/USD's first
    download ended 2016-08-31 and said nothing)."""
    mk = os.path.join(DIR, f"{ident}_h1_{side}.merged")
    if not (os.path.exists(os.path.join(DIR, f"{ident}_h1_{side}.csv")) and os.path.exists(mk)):
        return False
    with open(mk) as fh:
        return fh.read().startswith(("v2", "v3"))


def available(universe=UNIVERSE):
    """The universe's instruments whose hourly bid and ask are both merged and checked, in universe order."""
    return [n for n, (i, _) in UNIVERSES[universe].items() if _checked(i, "bid") and _checked(i, "ask")]


def _read(ident, tf, side):
    """One merged CSV (timestamp ms, open, high, low, close[, volume]) on naive-UTC times."""
    df = pd.read_csv(os.path.join(DIR, f"{ident}_{tf}_{side}.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    ts = df.columns[0]
    t = (pd.to_datetime(df[ts], unit="ms", utc=True) if np.issubdtype(df[ts].dtype, np.number)
         else pd.to_datetime(df[ts], utc=True))
    t = t.dt.tz_convert(None)
    df.index = t.dt.floor("h") if tf == "h1" else t.dt.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close"] + (["volume"] if "volume" in df else [])]


def _rates():
    r = pd.read_csv(os.path.join(DIR, "rates.csv"), index_col=0, parse_dates=True).sort_index()
    r = r.ffill()
    r.index = r.index + pd.DateOffset(months=RATE_LAG_MONTHS)         # usable from here
    return r / 100.0


def build(refresh=False, names=None, universe=UNIVERSE):
    """The panel (arrays on the hourly grid) of `names` (default: every available instrument),
    cached to data/macro_panel_<universe>.npz and rebuilt when the instrument list changes."""
    names = list(names or available(universe))
    if "EURUSD" not in names:
        raise RuntimeError("EUR/USD (the grid) is not downloaded yet")
    cache = os.path.join(ROOT, "data", f"macro_panel_{universe}.npz")
    if os.path.exists(cache) and not refresh:
        with np.load(cache, allow_pickle=False) as z:
            if list(z["names"]) == names:
                return {k: z[k] for k in z.files}
    inst = UNIVERSES[universe]
    ident = {n: inst[n][0] for n in names}
    cls = [inst[n][1] for n in names]
    grid = _read(ident["EURUSD"], "h1", "bid").index
    grid = grid[grid >= pd.Timestamp(GRID_START)]
    T, N = len(grid), len(names)
    arr = {k: np.full((T, N), np.nan) for k in ("bo", "ao", "bc", "ac")}
    for i, n in enumerate(names):
        bid, ask = _read(ident[n], "h1", "bid"), _read(ident[n], "h1", "ask")
        both = bid.index.intersection(ask.index)
        bid, ask = bid.loc[both].reindex(grid), ask.loc[both].reindex(grid)
        arr["bo"][:, i], arr["ao"][:, i] = bid["open"].to_numpy(), ask["open"].to_numpy()
        arr["bc"][:, i], arr["ac"][:, i] = bid["close"].to_numpy(), ask["close"].to_numpy()
    for side in ("o", "c"):                 # bad prints: crossed or absurdly wide quotes are no bar
        b, a = arr["b" + side], arr["a" + side]
        with np.errstate(invalid="ignore"):
            bad = ~((a >= b) & ((a - b) / ((a + b) / 2) <= MAX_REL_SPREAD) & (b > 0))
        b[bad], a[bad] = np.nan, np.nan
    bar = ~np.isnan(arr["bo"]) & ~np.isnan(arr["bc"])
    for k in arr:
        arr[k][~bar] = np.nan
    # daily closes: the last hourly mid close of each UTC day the instrument traded
    mc = pd.DataFrame((arr["bc"] + arr["ac"]) / 2.0, index=grid)
    daily = mc.groupby(mc.index.normalize()).last()
    extra = {}
    for n, i in DAILY_EXTRA.get(universe, {}).items():
        path = os.path.join(DIR, f"{i}_d1_bid.csv")
        if os.path.exists(path):
            extra[n] = _read(i, "d1", "bid")["close"]
    ext = pd.DataFrame(extra).sort_index() if extra else pd.DataFrame(index=daily.index)
    rates = _rates()
    out = dict(times=grid.values.astype("datetime64[h]"), names=np.array(names), classes=np.array(cls),
               idents=np.array([ident[n] for n in names]), bar=bar,
               daily_dates=daily.index.values.astype("datetime64[D]"), daily_close=daily.to_numpy(),
               extra_names=np.array(list(ext.columns)), extra_dates=ext.index.values.astype("datetime64[D]"),
               extra_close=ext.to_numpy(dtype=float).reshape(len(ext), -1),
               rate_dates=rates.index.values.astype("datetime64[D]"), rate_ccy=np.array(list(rates.columns)),
               rates=rates.to_numpy(), universe=np.array(universe), **arr)
    np.savez_compressed(cache, **out)
    return out


def _ewm_vol(r, halflife, min_periods):
    """sqrt of the EWMA of r**2 over its non-NaN entries, carried over NaNs."""
    return np.sqrt(pd.DataFrame(r ** 2).ewm(halflife=halflife, min_periods=min_periods, ignore_na=True)
                   .mean().ffill().to_numpy())


def _prev_day(times, dates):
    """For each bar, the index in `dates` of the last day strictly before the bar's own UTC day (-1: none)."""
    return np.searchsorted(dates, times.normalize().values.astype("datetime64[D]")) - 1


def _daily_block(close, dates, times, lr_now):
    """(trend_k for k in 20, 60, 250 as a dict, annualised vol) at every bar from daily closes
    through the previous UTC day; lr_now is the log of the latest price known at the bar."""
    dlog = np.log(close)
    dret = np.r_[np.full((1, close.shape[1]), np.nan), np.diff(dlog, axis=0)]
    sig_d = _ewm_vol(dret, VOL_HALFLIFE_D, 20)
    j = _prev_day(times, dates)
    ok = j >= 0
    jj = np.maximum(j, 0)
    sd = np.where(ok[:, None], sig_d[jj], np.nan)
    out = {}
    for k in (20, 60, 250):
        jk = jj - k
        past = np.where((ok & (jk >= 0))[:, None], dlog[np.maximum(jk, 0)], np.nan)
        out[f"trend_{k}d"] = (lr_now - past) / (sd * np.sqrt(k))
    return out, sd * np.sqrt(252.0), (j, dlog, sig_d)


def features(p):
    """(X (T, N, F) float32 with the universe's FEATURES, arrays the simulator reads). Everything
    at bar t is known before bar t opens."""
    universe = str(p["universe"]) if "universe" in p else UNIVERSE
    feats = FEATURES_OF[universe]
    times = pd.DatetimeIndex(p["times"])
    names, cls = list(p["names"]), list(p["classes"])
    idents = list(p["idents"]) if "idents" in p else [UNIVERSES[universe][n][0] for n in names]
    T, N = p["bar"].shape
    mo = (p["bo"] + p["ao"]) / 2.0
    mc = (p["bc"] + p["ac"]) / 2.0
    # the last price known at the decision: the mid CLOSE of the latest bar before t
    lc = pd.DataFrame(mc).shift(1).ffill().to_numpy()
    traded_prev = np.r_[np.zeros((1, N), bool), p["bar"][:-1]]
    lr = np.log(lc)
    ret = np.where(traded_prev, np.r_[np.full((1, N), np.nan), np.diff(lr, axis=0)], np.nan)
    sig_h = _ewm_vol(ret, VOL_HALFLIFE_H, 240)
    F = {}
    for k, lab in ((1, "ret_1h"), (4, "ret_4h"), (24, "ret_24h"), (120, "ret_120h")):
        past = np.r_[np.full((k, N), np.nan), lr[:-k]]
        F[lab] = (lr - past) / (sig_h * np.sqrt(k))
    vol24 = np.sqrt(pd.DataFrame(ret ** 2).rolling(24, min_periods=12).mean().to_numpy())
    F["vol_ratio_24h"] = vol24 / sig_h
    lcd = pd.DataFrame(lc)
    hi, lo = lcd.rolling(120, min_periods=60).max().to_numpy(), lcd.rolling(120, min_periods=60).min().to_numpy()
    F["range_pos_120h"] = np.where(hi > lo, (lc - lo) / np.where(hi > lo, hi - lo, 1.0), 0.5)
    rel = pd.DataFrame((p["ac"] - p["bc"]) / mc).shift(1)                      # bar t-1's closing spread
    med = rel.rolling(480, min_periods=120).median().ffill()
    F["spread_ratio"] = (rel.ffill() / med).to_numpy()
    F["hour_utc"] = np.broadcast_to(times.hour.to_numpy()[:, None], (T, N)).astype(float)
    F["weekday"] = np.broadcast_to(times.weekday.to_numpy()[:, None], (T, N)).astype(float)
    trends, vol_ann, _ = _daily_block(p["daily_close"], p["daily_dates"], times, lr)
    F.update(trends)
    F["vol_ann"] = vol_ann
    # carry, a fraction per year, for a LONG position
    rd = pd.DatetimeIndex(p["rate_dates"])
    days = times.normalize()
    R = pd.DataFrame(p["rates"], index=rd, columns=list(p["rate_ccy"])).reindex(rd.union(days.unique())).ffill()
    R = R.reindex(days).to_numpy()
    col = {c: k for k, c in enumerate(p["rate_ccy"])}
    carry = np.empty((T, N))
    for i, idt in enumerate(idents):
        lg, sh = legs(idt)
        carry[:, i] = (R[:, col[lg]] if lg else 0.0) - R[:, col[sh]]
    F["carry"] = carry
    sides = [legs(i) for i in idents]
    usd_sign = np.array([1.0 if lg == "USD" else (-1.0 if sh == "USD" and lg else 0.0) for lg, sh in sides])
    fx = usd_sign != 0
    const = lambda v: np.broadcast_to(np.asarray(v, float)[None, :], (T, N))
    if universe == "fx":
        F["is_metal"] = const([c == "metal" for c in cls])
        F["is_em"] = const([bool({lg, sh} & EM) for lg, sh in sides])
        F["usd_side"] = const([1.0 if lg == "USD" else (-1.0 if sh == "USD" else 0.0) for lg, sh in sides])
        live = p["bar"]
        for k in RANKED:
            v = pd.DataFrame(np.where(live, F[k], np.nan))
            F[f"rank_{k}"] = v.rank(axis=1, pct=True).fillna(0.5).to_numpy()
        g = names.index("GOLD") if "GOLD" in names else None
        mk = {"usd_ret_24h": np.nanmean(F["ret_24h"][:, fx] * usd_sign[fx], axis=1),
              "usd_trend_20d": np.nanmean(F["trend_20d"][:, fx] * usd_sign[fx], axis=1),
              "gold_ret_24h": F["ret_24h"][:, g] if g is not None else np.zeros(T)}
        ex = list(p["extra_names"]) if "extra_names" in p else []
        if "US500" in ex:                            # the S&P's last complete day
            c = p["extra_close"][:, ex.index("US500")][:, None]
            dlog = np.log(c)
            dret = np.r_[np.full((1, 1), np.nan), np.diff(dlog, axis=0)]
            sd = _ewm_vol(dret, VOL_HALFLIFE_D, 20)
            j = _prev_day(times, p["extra_dates"])
            ok, jj = j >= 1, np.maximum(j, 1)
            mk["us500_ret_1d"] = np.where(ok, dret[jj, 0] / sd[jj, 0], np.nan)
            j20 = np.maximum(jj - 20, 0)
            mk["us500_trend_20d"] = np.where(ok & (jj >= 20), (dlog[jj, 0] - dlog[j20, 0]) / (sd[jj, 0] * np.sqrt(20)), np.nan)
        else:
            mk["us500_ret_1d"] = mk["us500_trend_20d"] = np.zeros(T)
    else:
        for c in ("fx", "equity", "metal", "energy"):
            F[f"is_{c}"] = const([x == c for x in cls])
        u = names.index("US500")
        mk = {"us500_ret_24h": F["ret_24h"][:, u], "us500_trend_20d": F["trend_20d"][:, u],
              "us500_vol_ratio": F["vol_ratio_24h"][:, u],
              "usd_ret_24h": np.nanmean(F["ret_24h"][:, fx] * usd_sign[fx], axis=1),
              "usd_trend_20d": np.nanmean(F["trend_20d"][:, fx] * usd_sign[fx], axis=1)}
    for k, v in mk.items():
        F[k] = np.broadcast_to(np.nan_to_num(v, nan=0.0)[:, None], (T, N))
    X = np.stack([np.asarray(F[k], dtype=np.float64) for k in feats], axis=2)
    zcols = [feats.index(k) for k in feats if k.startswith(("ret_", "trend_", "us500_ret", "us500_trend", "usd_", "gold_"))]
    X[:, :, zcols] = np.clip(X[:, :, zcols], -Z_CLIP, Z_CLIP)
    X[:, :, feats.index("vol_ratio_24h")] = np.clip(X[:, :, feats.index("vol_ratio_24h")], 0, Z_CLIP)
    X[:, :, feats.index("spread_ratio")] = np.clip(X[:, :, feats.index("spread_ratio")], 0, 50.0)
    finite = np.isfinite(X).all(axis=2)
    avail = p["bar"] & finite & np.isfinite(vol_ann) & (vol_ann > 0) & np.isfinite(carry)
    hours = np.r_[np.diff(p["times"]).astype(np.int64).astype(float), 1.0]
    half = np.where(p["bar"], (p["ao"] - p["bo"]) / (2.0 * mo), np.nan)
    sim = dict(times=p["times"], names=names, classes=cls, avail=avail, features=list(feats),
               mid=pd.DataFrame(mo).ffill().to_numpy(), half_spread=half, carry=carry,
               vol=vol_ann, hours=hours, last=lc)
    return np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float32), sim


def synthetic(T=24 * 5 * 52 * 3, N=6, seed=0, universe=UNIVERSE):
    """A random-walk panel in the shape `features` returns, for tests before the data exists:
    hourly bars five days a week from 2012, fat-tailed returns, spreads, carry and closures."""
    feats = FEATURES_OF[universe]
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2012-01-02T00", "h")
    hrs, t = [], t0
    while len(hrs) < T:
        wd = int(((t.astype("datetime64[D]").astype(int)) + 3) % 7)          # 0 Monday
        if wd < 5:
            hrs.append(t)
        t = t + np.timedelta64(1, "h")
    times = np.array(hrs)
    vol = rng.uniform(0.06, 0.35, N)
    r = rng.standard_t(4, (T, N)) / np.sqrt(2.0) * vol / np.sqrt(6240.0)
    mid = 100.0 * np.exp(np.cumsum(r, axis=0))
    avail = rng.random((T, N)) > 0.03
    avail[:, 0] = True
    names = [f"A{i}" for i in range(N)]
    classes = ["fx" if i % 2 == 0 else ("metal" if universe == "fx" else "equity") for i in range(N)]
    X = rng.normal(0, 1, (T, N, len(feats))).astype(np.float32)
    X[:, :, feats.index("hour_utc")] = pd.DatetimeIndex(times).hour.to_numpy()[:, None]
    for k, f in enumerate(feats):
        if f.startswith("is_"):
            X[:, :, k] = np.array([c == f[3:] for c in classes], np.float32)[None, :]
    sim = dict(times=times, names=names, classes=classes, avail=avail, mid=mid, features=list(feats),
               half_spread=np.where(avail, rng.uniform(0.2e-4, 3e-4, (T, N)), np.nan),
               carry=np.broadcast_to(rng.uniform(-0.03, 0.03, N)[None, :], (T, N)).copy(),
               vol=np.broadcast_to(vol[None, :], (T, N)).copy() * rng.uniform(0.8, 1.2, (T, N)),
               hours=np.r_[np.diff(times).astype(np.int64).astype(float), 1.0], last=mid)
    return X, sim
