"""Do any of the cheap features pick which NEW companies will go on to grow? A signal audit
inside the growth pool's fair sample (growth_pool.py --sample: companies that first reached
$2B in 2013 or later, listed and gone alike).

POINT IN TIME, both ways:
    membership   a company is in the pool on day t only if its market value that day was at
                 least FLOOR: shares outstanding AS FILED with the SEC before t (EDGAR,
                 dei:EntityCommonStockSharesOutstanding, all classes) x the price it was
                 quoted at the day before -- never the size it reached later
    features     computed from prices up to the day before (trend_features.py and below)
    outcomes     open-to-open returns over the next h days; a stock that stops trading
                 inside the window keeps its last price (a buy-out pays about the deal price,
                 a failure about nothing), so the dead are never dropped from the outcome

Prices: yfinance for the listed (data/yf, yf_prices.py), Tiingo for the gone (data/tiingo,
tiingo_prices.py), adjusted for splits and dividends; Tiingo's stale rows after a delisting
(no volume) are cut. The rank IC, its Newey-West t and the Bonferroni bar are signal_audit's.

    python growth_audit.py [--sample N]   writes results/growth_audit/growth_audit[N].md
    python growth_audit.py --sample 1+2   both samples pooled
"""
import json
import os
import time

import numpy as np
import pandas as pd

import fundamentals as fu
import signal_audit as sa
import trend_features as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARES = os.path.join(ROOT, "data", "edgar_shares")
OUT = os.path.join(ROOT, "results", "growth_audit")
FLOOR = 2e9
START = "2013-01-01"
CAL_START = "2011-01-01"          # two years of history before the pool starts, for the 252-day windows
# which sample: --sample N reads data/growth_sample<N>.csv and writes growth_audit<N>.md;
# --sample 1+2 pools samples ("1" is the first, data/growth_sample.csv) into growth_audit1+2.md
SAMPLE_NAME = __import__("sys").argv[__import__("sys").argv.index("--sample") + 1] if "--sample" in __import__("sys").argv else ""


def sample_table():
    parts = [("" if n == "1" else n) for n in SAMPLE_NAME.split("+")] if "+" in SAMPLE_NAME else [SAMPLE_NAME]
    s = pd.concat([pd.read_csv(os.path.join(ROOT, "data", f"growth_sample{n}.csv")) for n in parts])
    return s.drop_duplicates("cik")


def prices(ticker, source):
    """Daily adjusted open/high/low/close, the quoted close and volume, or None."""
    if source == "yfinance":
        p = os.path.join(ROOT, "data", "yf", f"{ticker}.csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        k = d["Adj Close"] / d["Close"]
        split = d["Stock Splits"].replace(0, 1.0).fillna(1.0)
        later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)           # splits after each day
        return pd.DataFrame({"open": d["Open"] * k, "high": d["High"] * k, "low": d["Low"] * k,
                             "close": d["Adj Close"], "quoted": d["Close"] * later, "volume": d["Volume"],
                             "raw_volume": d["Volume"] / later})
    p = os.path.join(ROOT, "data", "tiingo", f"{ticker}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    traded = d.index[d["volume"] > 0]
    if len(traded):
        d = d[d.index <= traded[-1]]          # Tiingo repeats the last price with no volume after a delisting
    return pd.DataFrame({"open": d["adjOpen"], "high": d["adjHigh"], "low": d["adjLow"], "close": d["adjClose"],
                         "quoted": d["close"], "volume": d["adjVolume"], "raw_volume": d["volume"]})


def shares(cik):
    """Shares outstanding as known after each filing (all classes summed), by filing date."""
    os.makedirs(SHARES, exist_ok=True)
    path = os.path.join(SHARES, f"{int(cik)}.json")
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
    else:
        raw = fu._get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{int(cik):010d}/dei/"
                      f"EntityCommonStockSharesOutstanding.json") or {}
        with open(path, "w") as fh:
            json.dump(raw, fh)
        time.sleep(0.12)
    rows = [(u["filed"], u["end"], u.get("accn"), u["val"]) for u in raw.get("units", {}).get("shares", [])]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["filed", "end", "accn", "val"])
    per = df.groupby(["accn", "end"]).agg(filed=("filed", "min"), val=("val", "sum")).reset_index()
    per = per.sort_values("filed").groupby("filed")["val"].last()
    per.index = pd.to_datetime(per.index)
    return per


def calendar():
    """(trading days from CAL_START, SPY's dividend-adjusted open): from a small SPY file, so
    the audit never loads the main panel (memory)."""
    path = os.path.join(ROOT, "data", "yf", "SPY.csv")
    if not os.path.exists(path):
        import yfinance as yf
        os.makedirs(os.path.dirname(path), exist_ok=True)
        yf.download("SPY", start="2004-01-01", auto_adjust=False, actions=True, progress=False,
                    multi_level_index=False).to_csv(path)
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d = d[d.index >= CAL_START]
    return pd.DatetimeIndex(d.index), (d["Open"] * d["Adj Close"] / d["Close"]).to_numpy()


FACTS = os.path.join(ROOT, "data", "edgar_facts_pool")
FACT_TAGS = dict(fu.TAGS, gross=["GrossProfit"], opinc=["OperatingIncomeLoss"],
                 cogs=["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"])
FUND_NAMES = ["rev_growth (TTM, y/y)", "rev_accel (growth - growth a quarter ago)", "gross_margin (TTM)",
              "op_margin (TTM)", "profit_margin (TTM)", "asset_growth (y/y)", "sue (earnings surprise)",
              "rev_sue (revenue surprise)"]
STALE_DAYS = 200                     # a figure older than this (no new filing: gone, or late) is dropped


def facts(cik):
    """{concept: {tag: [facts]}} of FACT_TAGS for one company (EDGAR companyfacts), cached."""
    os.makedirs(FACTS, exist_ok=True)
    path = os.path.join(FACTS, f"{int(cik)}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    f = (fu._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json") or {}).get("facts", {})
    taxonomies = [(f.get("us-gaap", {}), FACT_TAGS), (f.get("ifrs-full", {}), fu.IFRS_TAGS)]
    cur = fu._currency([tx for tx, _ in taxonomies])
    out = {c: {} for c in FACT_TAGS}
    for tx, tagset in taxonomies:
        for c, tags in tagset.items():
            for tag in tags:
                if tag in tx and cur in tx[tag]["units"]:
                    out.setdefault(c, {}).setdefault(tag, []).extend(tx[tag]["units"][cur])
    with open(path, "w") as fh:
        json.dump(out, fh)
    time.sleep(0.12)
    return out


def fundamentals(cik, dates):
    """{name: (T,) as known before each day} from one company's filings."""
    import earnings
    f = facts(cik)
    per = {c: fu._periods(f.get(c, {})) for c in FACT_TAGS}
    ttm = lambda c: {e: (v, fd) for e, v, fd in fu._ttm(per[c])}
    rev, gross, cogs, opinc, profit = ttm("revenue"), ttm("gross"), ttm("cogs"), ttm("opinc"), ttm("profit")
    rec = {n: [] for n in FUND_NAMES}                # (available, period end, value)
    ends = sorted(e for e in rev if rev[e][0] > 0)

    def growth(e):
        ago = [x for x in ends if abs((e - x).days - 365) <= 20]
        return (np.log(rev[e][0] / rev[ago[0]][0]), max(rev[e][1], rev[ago[0]][1])) if ago else None
    for e in ends:
        R, fr = rev[e]
        g = growth(e)
        if g:
            rec["rev_growth (TTM, y/y)"].append((g[1], e, g[0]))
            prev = [x for x in ends if 80 <= (e - x).days <= 100]
            gp = growth(prev[-1]) if prev else None
            if gp:
                rec["rev_accel (growth - growth a quarter ago)"].append((max(g[1], gp[1]), e, g[0] - gp[0]))
        gpv = gross.get(e) or ((R - cogs[e][0], max(fr, cogs[e][1])) if e in cogs else None)
        if gpv:
            rec["gross_margin (TTM)"].append((max(fr, gpv[1]), e, gpv[0] / R))
        if e in opinc:
            rec["op_margin (TTM)"].append((max(fr, opinc[e][1]), e, np.clip(opinc[e][0] / R, -5, 5)))
        if e in profit:
            rec["profit_margin (TTM)"].append((max(fr, profit[e][1]), e, np.clip(profit[e][0] / R, -5, 5)))
    assets = {k[1]: v for k, v in per["assets"].items() if k[0] is None and v[0] > 0}
    for e, (a, fa) in assets.items():
        ago = [x for x in assets if abs((e - x).days - 365) <= 20]
        if ago:
            rec["asset_growth (y/y)"].append((max(fa, assets[ago[0]][1]), e, np.log(a / assets[ago[0]][0])))
    for n, c in (("sue (earnings surprise)", "profit"), ("rev_sue (revenue surprise)", "revenue")):
        rec[n] = [(d, d, v) for d, v in earnings.surprises(per[c])]      # dated by the 10-Q
    return {n: fu._as_of(dates, r, STALE_DAYS) for n, r in rec.items()}


def build():
    dates, spy = calendar()
    s = sample_table()
    s = s[s["prices_from"] != "none"]
    keys = ("open", "high", "low", "close", "quoted", "volume", "raw_volume", "mv", "shares") + tuple(FUND_NAMES)
    cols, meta = {k: [] for k in keys}, []
    for r in s.itertuples():
        d = prices(r.ticker, r.prices_from)
        if d is None or d.empty:
            continue
        d = d[~d.index.duplicated()].reindex(dates)
        sh = shares(r.cik)
        known = sh.reindex(sh.index.union(dates)).ffill().shift(1).reindex(dates) if len(sh) else pd.Series(np.nan, dates)
        d["mv"] = known.to_numpy() * d["quoted"].shift(1).to_numpy()                # shares as filed x yesterday's price
        d["shares"] = known.to_numpy()
        for n, x in fundamentals(r.cik, dates).items():                              # as filed, known the day after
            d[n] = x
        for k in cols:
            cols[k].append(d[k].to_numpy(dtype=float))
        meta.append((r.ticker, r.prices_from))
    A = {k: np.column_stack(v) for k, v in cols.items()}
    return spy, dates, A, meta


def features(A, pool):
    c, h, l, v = A["close"], A["high"], A["low"], A["volume"]
    C, V = pd.DataFrame(c), pd.DataFrame(np.where(v > 0, v, np.nan))
    lag = lambda x: pd.DataFrame(x).shift(1).to_numpy()
    f = tf.trend(c, pool)

    class F32(dict):                        # every feature stored in 32 bits as it is made (memory)
        def __setitem__(self, k, v):
            super().__setitem__(k, np.asarray(v, np.float32))
    out = F32()
    for n in ("ma_5_21", "ma_21_63", "ma_50_200", "px_ma_50", "px_ma_200", "ma_slope_21", "ma_slope_63"):
        out[n] = f[n]
    s63 = pd.DataFrame(f["ma_slope_63"])
    out["slope_acc_63 (2 diff)"] = (s63 - s63.shift(21)).to_numpy()
    out["near_52w_high"] = lag(np.log(C / C.rolling(252, min_periods=200).max()))
    out["above_52w_low"] = lag(np.log(C / C.rolling(252, min_periods=200).min()))
    out["dd_from_63d_high"] = lag(np.log(C / C.rolling(63, min_periods=50).max()))
    vwap = (C * V).rolling(252, min_periods=200).sum() / V.rolling(252, min_periods=200).sum()
    out["overhang (P / 1y VWAP)"] = lag(np.log(C / vwap))
    out["mom_12_1"] = np.log(C.shift(21) / C.shift(252)).shift(1).to_numpy()
    out["rev_1m"] = lag(-np.log(C / C.shift(21)))
    first = C.apply(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
    out["gain_since_first_trade"] = lag(np.log(C / first))
    r = np.log(C / C.shift(1))
    out["vol_21 (close to close)"] = lag(r.rolling(21, min_periods=15).std() * np.sqrt(252))
    gk = 0.5 * np.log(pd.DataFrame(h) / pd.DataFrame(l)) ** 2 - (2 * np.log(2) - 1) * np.log(C / pd.DataFrame(A["open"])) ** 2
    out["vol_21 (range, Garman-Klass)"] = lag(np.sqrt(gk.rolling(21, min_periods=15).mean() * 252))
    age = C.notna().cumsum()
    out["log_age (days traded)"] = lag(np.log(age.where(age > 0)))
    mv = pd.DataFrame(A["mv"])
    out["log_market_value"] = np.log(mv).to_numpy()
    out["mv_growth_1y"] = np.log(mv / mv.shift(252)).to_numpy()
    # VOLUME: a surge against normal (the high-volume premium), turnover against the shares
    # outstanding as filed (low turnover has earned more), liquidity, and the volume trend
    vm = lambda w: V.rolling(w, min_periods=int(0.7 * w)).mean()
    out["abn_volume (21d / 252d)"] = lag(np.log(vm(21) / vm(252)))
    out["abn_volume (5d / 63d)"] = lag(np.log(vm(5) / vm(63)))
    out["volume_trend (63d / 252d)"] = lag(np.log(vm(63) / vm(252)))
    raw = pd.DataFrame(np.where(A["raw_volume"] > 0, A["raw_volume"], np.nan))
    out["turnover (63d, of shares)"] = lag(np.log(raw.rolling(63, min_periods=44).mean() / pd.DataFrame(A["shares"])))
    out["log_dollar_volume (63d)"] = lag(np.log((C * V).rolling(63, min_periods=44).mean()))
    # FUNDAMENTALS, as filed with the SEC (fundamentals(): TTM, known the day after filing)
    for n in FUND_NAMES:
        out[n] = A[n]
    return out


def main():
    from types import SimpleNamespace
    from scipy import stats
    spy, dates, A, meta = build()
    pool = (A["mv"] >= FLOOR) & np.isfinite(A["open"]) & (dates >= pd.Timestamp(START))[:, None]
    n_pool = pd.Series(pool.sum(1), index=dates)
    print(f"{len(meta)} companies priced ({sum(m[1] == 'tiingo' for m in meta)} gone, from Tiingo); pool size by year "
          f"(mean per day): " + ", ".join(f"{y}: {v:.0f}" for y, v in n_pool[n_pool.index >= START].groupby(
              n_pool[n_pool.index >= START].index.year).mean().items()), flush=True)
    feats = features(A, pool)
    names = list(feats)
    X = np.empty(feats[names[0]].shape + (len(names),), np.float32)      # 32-bit, filled one at a time
    for k, n in enumerate(names):
        X[:, :, k] = feats.pop(n)
    opn = pd.DataFrame(A["open"]).ffill().to_numpy()               # the dead keep their last price
    panel = SimpleNamespace(dates=dates.values, open=opn, universe=pool, bench_open=spy)
    sa.MIN_STOCKS = 15
    M = np.zeros((len(dates), 0))
    periods = {"2014-2019": ("2014-01-01", "2019-12-31"), "2020-2026": ("2020-01-01", "2026-09-30")}
    rows = []
    for h in (21, 63, 252):
        res = {k: sa.audit(panel, X, M, names, [], a, b, h, 21)[0] for k, (a, b) in periods.items()}
        for n in names:
            r0, r1 = res["2014-2019"][n], res["2020-2026"][n]
            rows.append(dict(horizon=f"{h}d", input=n, IC_1419=r0["ic"], t_1419=r0["t"], days_1419=r0["n"],
                             IC_2026=r1["ic"], t_2026=r1["t"], days_2026=r1["n"]))
    df = pd.DataFrame(rows)
    bar = float(stats.norm.ppf(1 - 0.025 / (3 * len(names))))
    pd.set_option("display.width", 220)
    text = df.round(3).to_string(index=False)
    print(text)
    print(f"\nBonferroni bar for {len(names)} inputs x 3 horizons: |t| > {bar:.2f}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"growth_audit{SAMPLE_NAME}.md"), "w", encoding="utf-8") as fh:
        fh.write("# Growth-pool signal audit (fair sample)\n\n" + __doc__.split("    python")[0].strip()
                 + "\n\n```\n" + text + f"\n```\n\nBonferroni bar |t| > {bar:.2f}\n")
    # THE PORTFOLIO CHECK: an IC is not a return. Equal weights, rebalanced every 21 trading
    # days at the open, 5 bp per traded dollar; the score is the input strongest on 2014-2019
    rows = []
    for lab, (a, b) in periods.items():
        for sname in ("near_52w_high", "mom_12_1", "rev_growth (TTM, y/y)"):
            score = X[:, :, names.index(sname)]
            for pick in ("top", "all", "bottom"):
                if pick == "all" and sname != "near_52w_high":
                    continue
                r, rs = portfolio(opn, pool, score, spy, dates, a, b, pick)
                rows.append(dict(period=lab, portfolio="whole pool" if pick == "all" else f"{pick} third by {sname}",
                                 **perf(r, rs)))
            if sname == "near_52w_high":
                rows.append(dict(period=lab, portfolio="S&P 500 (SPY)", **perf(rs, rs)))
            # LET IT RIDE: enter on the signal, never trim a winner, exit only when the trend
            # breaks (RIDE_EXIT below the 52-week high) -- the right tail an equal-weight
            # monthly rebalance sells off a little every month
            r, rs = ride(opn, pool, score, X[:, :, names.index("near_52w_high")], spy, dates, a, b)
            rows.append(dict(period=lab, portfolio=f"RIDE: enter top third by {sname}, exit 30% off high",
                             **perf(r, rs)))
    port = pd.DataFrame(rows).round(2).to_string(index=False)
    print("\n" + port)
    with open(os.path.join(OUT, f"growth_audit{SAMPLE_NAME}.md"), "a", encoding="utf-8") as fh:
        fh.write("\n## Portfolio check (21-day decisions, 5 bp costs; the thirds equal-weighted and rebalanced; "
                 "RIDE never trims a winner and sells only 30% below the 52-week high)\n\n```\n" + port + "\n```\n")


RIDE_EXIT = np.log(0.7)            # sell only when the price is 30% or more below its 52-week high


def ride(opn, pool, score, near_high, spy, dates, a, b, step=21, frac=1 / 3, cost=5e-4):
    """(net 21-day returns, SPY's) of LET IT RIDE: every decision, holdings that fell RIDE_EXIT
    below their 52-week high are sold; pool stocks newly in the top third by `score` are bought,
    each at 1 / (number held), paid for by trimming every holding in proportion -- so a winner
    keeps its size relative to the rest and is never cut back to an equal weight."""
    idx = np.where((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))[0]
    days = [d for d in idx[::step] if d + step < len(dates)]
    w_drift = np.zeros(opn.shape[1])
    out, bench = [], []
    for d in days:
        w = w_drift.copy()
        w[(w > 0) & np.isfinite(near_high[d]) & (near_high[d] < RIDE_EXIT)] = 0.0       # the trend broke
        w[~np.isfinite(opn[d])] = 0.0
        ok = pool[d] & np.isfinite(score[d]) & np.isfinite(opn[d])
        if ok.sum() >= 15:
            q = np.nanquantile(np.where(ok, score[d], np.nan), 1 - frac)
            new = ok & (score[d] >= q) & (w == 0)
        else:
            new = np.zeros_like(ok)
        n_keep, n_new = int((w > 0).sum()), int(new.sum())
        if n_keep + n_new == 0:
            out.append(0.0)
            bench.append(spy[d + step] / spy[d] - 1.0)
            w_drift = w
            continue
        if n_keep:
            w *= (n_keep / (n_keep + n_new)) / w.sum()
        w[new] = 1.0 / (n_keep + n_new)
        r = np.where(w > 0, opn[d + step] / opn[d] - 1.0, 0.0)
        gross = float(w @ r)
        out.append(gross - cost * float(np.abs(w - w_drift).sum()))
        w_drift = w * (1 + r) / (1 + gross)
        bench.append(spy[d + step] / spy[d] - 1.0)
    return np.array(out), np.array(bench)


def portfolio(opn, pool, score, spy, dates, a, b, pick, step=21, frac=1 / 3, cost=5e-4):
    """(net 21-day returns of the pick, SPY's over the same windows)."""
    idx = np.where((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))[0]
    days = [d for d in idx[::step] if d + step < len(dates)]
    w_drift = np.zeros(opn.shape[1])
    out, bench = [], []
    for d in days:
        ok = pool[d] & np.isfinite(score[d]) & np.isfinite(opn[d])
        if ok.sum() < 15:
            continue
        if pick == "all":
            sel = ok
        else:
            q = np.nanquantile(np.where(ok, score[d], np.nan), 1 - frac if pick == "top" else frac)
            sel = ok & ((score[d] >= q) if pick == "top" else (score[d] <= q))
        w = sel / sel.sum()
        r = np.where(sel, opn[d + step] / opn[d] - 1.0, 0.0)
        gross = float(w @ r)
        out.append(gross - cost * float(np.abs(w - w_drift).sum()))
        w_drift = w * (1 + r) / (1 + gross)
        bench.append(spy[d + step] / spy[d] - 1.0)
    return np.array(out), np.array(bench)


def perf(r, rs, per_year=252 / 21):
    if len(r) < 3:                                  # too few decision periods to say anything
        return {k: np.nan for k in ("return %/yr", "vol %", "max DD %", "beta", "alpha %/yr", "t(alpha)")} | {"periods": len(r)}
    w = np.r_[1.0, np.cumprod(1 + r)]
    beta = float(np.cov(r, rs)[0, 1] / np.var(rs, ddof=1))
    alpha = float(np.mean(r - beta * rs) * per_year * 100)
    se = float(np.std(r - beta * rs, ddof=2) / np.sqrt(len(r)) * per_year * 100)
    return {"return %/yr": 100 * (w[-1] ** (per_year / len(r)) - 1), "vol %": 100 * np.std(r, ddof=1) * np.sqrt(per_year),
            "max DD %": 100 * (w / np.maximum.accumulate(w) - 1).min(), "beta": beta, "alpha %/yr": alpha,
            "t(alpha)": alpha / se if se > 0 else np.nan, "periods": len(r)}


if __name__ == "__main__":
    main()
