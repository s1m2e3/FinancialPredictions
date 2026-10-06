"""Do P/E and PEG tell which of the ten largest S&P 500 companies will move up next?

Every month of the last five years (the first trading day of the month), the ten largest
S&P 500 members BY MARKET VALUE THAT DAY, and for each, as known at that open:
    P/E   market value / trailing-12-month net income   (net income <= 0: no P/E)
    PEG   P/E / (EPS growth over the past year, in %)   (EPS fell: no PEG)
          EPS = trailing net income / shares outstanding, both as filed by that day
and what the stock did next: its return over the next 1, 3 and 6 months (21, 63, 126
trading days, open to open) relative to the S&P 500 (SPY, dividends in). The question is
not whether a company stays a winner, only whether valuation tells which ones rise next.

Market values and shares are fundamentals.py's (EDGAR, point in time); net income is the
EDGAR "profit" concept, first-filed values only, NetIncomeLoss before ProfitLoss. "Cheap on
both" means below that month's median P/E AND below its median PEG among the ten -- a fixed
cut-off (P/E < 25) would mean different things in 2021 and 2025.

Below it, the portfolio test: the ten held month by month, weighted by 1/volatility or by market value
(just following the ten), with and without a P/E and PEG preference.

Run from the repository root:  python valuation_top10.py
    (writes results/valuation/valuation_top10.png, valuation_portfolio.png, valuation_vs_top10.png,
     valuation_windows.png -- every entry and exit -- and valuation_6months.png)
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "valuation")
START = "2021-09-01"
HORIZONS = (21, 63, 126)
N_TOP = 10
# the portfolio test: from 2011 (once EDGAR has a year of filings for a PEG), split where the
# five-year look above begins -- the P/E and PEG preference was read from the later part only
PORT_START = "2011-01-01"
COST = 0.0005              # 5 basis points per dollar traded
# the reference palette (dataviz skill): text, surface, two series, diverging poles
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, GREY_MID, RED = "#2a78d6", "#eb6834", "#f0efec", "#e34948"


def net_income_ttm(facts, dates):
    """(T,) trailing-12-month net income as known at each open (first-filed values)."""
    import fundamentals as fu
    tags = facts.get("profit", {})
    ordered = {k: tags[k] for k in ("NetIncomeLoss", "ProfitLoss") if k in tags}
    ttm = fu._ttm(fu._periods(ordered))
    return fu._as_of(dates, [(f, e, v) for e, v, f in ttm], fu.STALE_DAYS + 90)


def build(start=START, n_top=N_TOP):
    import fundamentals as fu
    from stocks_data import load_panel
    p = load_panel()
    dates = pd.DatetimeIndex(p.dates)
    from stocks_data import raw_close
    with open(fu.RAW) as fh:
        facts = json.load(fh)
    S, _, domestic = fu.shares_and_equity(p, facts)
    # members whose share count is missing on most recent days were fetched before every
    # share-count tag was read (multi-class companies): fetch them again
    recent = dates >= pd.Timestamp(START)
    stale = [p.tickers[i] for i in range(len(p.tickers))
             if p.sp500[recent, i].any() and np.isnan(S[recent, i][p.sp500[recent, i]]).mean() > 0.5
             and facts.get(p.tickers[i], {}).get("_share_tags", 0) < len(fu.SHARE_TAGS)]
    if stale:
        print("fetching EDGAR share counts again for", stale, flush=True)
        facts = fu.refresh(stale)
        S, _, domestic = fu.shares_and_equity(p, facts)
    px = raw_close(p)
    prev = np.vstack([np.full((1, px.shape[1]), np.nan), px[:-1]])     # yesterday's close, known at the open
    mcap = np.where(domestic[None, :] & (S * prev > 0), S * prev, np.nan)
    first = np.r_[True, dates.month[1:] != dates.month[:-1]] & (dates >= pd.Timestamp(start))
    days = np.where(first)[0]
    # the ten largest members on each decision day; their net income only where needed
    top = {}
    for t in days:
        ok = p.sp500[t] & np.isfinite(mcap[t]) & (mcap[t] > 0)
        idx = np.where(ok)[0]
        top[t] = idx[np.argsort(-mcap[t, idx])[:n_top]]
    need = sorted({int(i) for v in top.values() for i in v})
    ni = {i: net_income_ttm(facts.get(p.tickers[i], {}), dates) for i in need}
    rows = []
    year = 252
    logret = np.diff(np.log(p.close), axis=0, prepend=np.nan)
    nxt = dict(zip(days, list(days[1:]) + [len(dates) - 1]))       # the next decision day
    for t in days:
        t1 = nxt[t]
        for rank, i in enumerate(top[t]):
            e_now = ni[i][t] / S[t, i] if S[t, i] > 0 else np.nan
            t0 = t - year
            e_ago = ni[i][t0] / S[t0, i] if t0 >= 0 and S[t0, i] > 0 else np.nan
            pe = mcap[t, i] / ni[i][t] if ni[i][t] > 0 else np.nan
            growth = e_now / e_ago - 1.0 if e_ago > 0 and e_now > 0 else np.nan
            peg = pe / (100.0 * growth) if np.isfinite(pe) and growth > 0 else np.nan
            r = dict(date=dates[t], ticker=p.tickers[i], rank=rank + 1, mcap_bn=mcap[t, i] / 1e9,
                     pe=pe, eps_growth=growth, peg=peg,
                     # risk known at the open: the last 63 days' daily returns, annualised
                     vol63=np.nanstd(logret[max(t - 63, 1):t, i]) * np.sqrt(252),
                     # held until the next decision day's open
                     ret=p.open[t1, i] / p.open[t, i] - 1.0, bench_ret=p.bench_open[t1] / p.bench_open[t] - 1.0,
                     rf=np.prod(1.0 + p.rf[t:t1]) - 1.0)
            for h in HORIZONS:
                if t + h < len(dates):
                    rs = p.open[t + h, i] / p.open[t, i]
                    rb = p.bench_open[t + h] / p.bench_open[t]
                    r[f"ex{h}"] = 100.0 * (rs / rb - 1.0)
                else:
                    r[f"ex{h}"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def monthly_ic(df, x, y):
    """Mean over months of the Spearman correlation across that month's ten, with a
    Newey-West t (lag grows with how much the forward windows overlap)."""
    ics = []
    for _, g in df.groupby("date"):
        g = g[[x, y]].dropna()
        if len(g) >= 5:
            ics.append(g[x].rank().corr(g[y].rank()))
    ics = np.array(ics)
    if len(ics) < 3:
        return np.nan, np.nan, len(ics)
    lag = {"ex21": 1, "ex63": 3, "ex126": 6}[y]
    e = ics - ics.mean()
    var = e @ e / len(e)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / len(e)
    return ics.mean(), ics.mean() / np.sqrt(max(var, 1e-12) / len(e)), len(ics)


def draw(df, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    y = "ex63"
    div = LinearSegmentedColormap.from_list("div", ["#9f2a29", RED, GREY_MID, BLUE, "#1c5cab"])

    def scatter(a, x, label, logx):
        d = df[[x, y]].dropna()
        a.scatter(d[x], d[y], s=16, color=BLUE, alpha=0.45, edgecolors="none")
        # medians by fifth of the x values: the shape, if there is one
        q = pd.qcut(d[x].rank(method="first"), 5, labels=False)
        med = d.groupby(q).agg(x=(x, "median"), y=(y, "median"))
        a.plot(med["x"], med["y"], "-o", color=INK, lw=2, ms=6, label="median of each fifth")
        a.axhline(0, color=INK2, lw=0.8)
        if logx:
            a.set_xscale("log")
        ic, tt, n = monthly_ic(df, x, y)
        rho = d[x].rank().corr(d[y].rank())
        a.set_title(f"{label} vs the next 3 months\nwithin-month rank correlation {ic:+.2f} (t {tt:+.1f}, {n} months); "
                    f"pooled {rho:+.2f}", fontsize=10)
        a.set_xlabel(label + (" (log scale)" if logx else ""))
        a.set_ylabel("return vs S&P 500, next 3 months, %")
        a.legend(loc="upper right", fontsize=8, frameon=False)

    scatter(ax[0, 0], "pe", "P/E", True)
    scatter(ax[0, 1], "peg", "PEG", True)

    # P/E against PEG, coloured by what came next
    a = ax[0, 2]
    d = df[["pe", "peg", y]].dropna()
    lim = float(np.nanpercentile(np.abs(d[y]), 95))
    sc = a.scatter(d["pe"], d["peg"], c=d[y], cmap=div, norm=TwoSlopeNorm(0, -lim, lim), s=22,
                   edgecolors="white", linewidths=0.4)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("P/E (log scale)")
    a.set_ylabel("PEG (log scale)")
    rho = d["pe"].rank().corr(d["peg"].rank())
    a.set_title(f"P/E against PEG, colour = return vs S&P next 3 months\n(P/E and PEG rank correlation {rho:+.2f})",
                fontsize=10)
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("return vs S&P 500, next 3 months, %")

    # does either rank the ten, at each horizon?
    a = ax[1, 0]
    xs = np.arange(len(HORIZONS))
    for k, (x, lab, col) in enumerate((("pe", "P/E", BLUE), ("peg", "PEG", ORANGE))):
        vals = [monthly_ic(df, x, f"ex{h}") for h in HORIZONS]
        m = np.array([v[0] for v in vals])
        se = np.array([abs(v[0] / v[1]) if v[1] and np.isfinite(v[1]) else np.nan for v in vals])
        a.errorbar(xs + (k - 0.5) * 0.18, m, yerr=2 * se, fmt="o", color=col, ms=8, lw=2, capsize=4, label=lab)
        for xi, mi in zip(xs, m):
            a.annotate(f"{mi:+.2f}", (xi + (k - 0.5) * 0.18, mi), xytext=(-10 if k == 0 else 10, 0),
                       textcoords="offset points", fontsize=8, color=INK2, va="center",
                       ha="right" if k == 0 else "left")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(xs, [f"next {h // 21} month{'s' if h > 21 else ''}" for h in HORIZONS])
    a.set_ylabel("within-month rank correlation with return vs S&P")
    a.set_title("Does a LOW value go with a HIGHER return? (negative = yes)\nmean over months, bars = 2 standard errors",
                fontsize=10)
    a.legend(loc="upper right", fontsize=8, frameon=False)

    # the cheaper half on both, month by month, against all ten and the S&P
    a = ax[1, 1]
    rets = {"all ten, equal weight": [], "cheaper half on BOTH P/E and PEG": [], "dearer half on both": []}
    months = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=["ex21"])
        if g.empty:
            continue
        mp, mg = g["pe"].median(), g["peg"].median()
        cheap = g[(g["pe"] <= mp) & (g["peg"] <= mg)]
        dear = g[(g["pe"] > mp) & (g["peg"] > mg)]
        months.append(dt)
        rets["all ten, equal weight"].append(g["ex21"].mean())
        rets["cheaper half on BOTH P/E and PEG"].append(cheap["ex21"].mean() if len(cheap) else 0.0)
        rets["dearer half on both"].append(dear["ex21"].mean() if len(dear) else 0.0)
    for (lab, r), col in zip(rets.items(), (INK2, BLUE, ORANGE)):
        cum = 100 * (np.cumprod(1 + np.array(r) / 100) - 1)
        a.plot(months, cum, color=col, lw=2, label=f"{lab}: {cum[-1]:+.0f}%")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("cumulative return vs S&P 500, %")
    a.set_title("Monthly baskets, rebalanced each month (no costs)\n0 = matched the S&P 500", fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False)

    # quadrants: mean next-3-month return by cheap / dear on each
    a = ax[1, 2]
    lab, mean, count, without, who = [], [], [], [], []
    for pe_cheap in (True, False):
        for peg_cheap in (True, False):
            sel = []
            for _, g in df.dropna(subset=["pe", "peg", y]).groupby("date"):
                m = ((g["pe"] <= g["pe"].median()) == pe_cheap) & ((g["peg"] <= g["peg"].median()) == peg_cheap)
                sel.append(g.loc[m, ["ticker", y]])
            s = pd.concat(sel)
            lab.append(f"P/E {'low' if pe_cheap else 'high'}\nPEG {'low' if peg_cheap else 'high'}")
            mean.append(s[y].mean())
            count.append(len(s))
            big = s.groupby("ticker")[y].sum().abs().idxmax()        # the company that moves the mean most
            without.append(s.loc[s["ticker"] != big, y].mean())
            who.append(big)
    cols = [BLUE if v >= 0 else RED for v in mean]
    bars = a.bar(range(4), mean, color=cols, width=0.55)
    a.scatter(range(4), without, marker="D", s=60, color="white", edgecolors=INK, linewidths=1.5, zorder=3,
              label="the same without its biggest contributor")
    for i, (b, v, n) in enumerate(zip(bars, mean, count)):
        a.annotate(f"{v:+.1f}%  (n={n})\nwithout {who[i]}: {without[i]:+.1f}%",
                   (b.get_x() + b.get_width() / 2, max(v, without[i], 0)), xytext=(0, 6),
                   textcoords="offset points", ha="center", fontsize=8, color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    lo, hi = min(min(mean), min(without), 0), max(max(mean), max(without), 0)
    a.set_ylim(lo - 0.3 * (hi - lo), hi + 0.45 * (hi - lo))
    a.set_xticks(range(4), lab)
    a.set_ylabel("mean return vs S&P 500, next 3 months, %")
    a.set_title("By quadrant (each month's medians among the ten)", fontsize=10)
    a.legend(loc="lower left", fontsize=8, frameon=False)

    n_missing_pe = int(df["pe"].isna().sum())
    n_missing_peg = int(df["peg"].isna().sum())
    fig.suptitle(f"The ten largest S&P 500 companies each month, {df['date'].min():%b %Y} - {df['date'].max():%b %Y}: "
                 f"does valuation tell which ones rise next?\n{len(df)} company-months, {df['ticker'].nunique()} companies; "
                 f"no P/E for {n_missing_pe} (losses), no PEG for {n_missing_peg} (EPS fell, or no P/E)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- the portfolio test
# Each month's ten largest, bought at the first open of the month and held to the next
# month's first open. T is just following the ten: weights by market value, like an index
# of the ten. B weighs by 1/volatility instead (the calmer a company, the more of it). The
# valuation tilts multiply a base weight by 0.5 (least preferred of the ten) up to 1.5 (most
# preferred); "only" keeps the names with P/E above the month's median and PEG at or below
# it (all ten in a month with none). A company with no P/E or PEG that month counts as
# middle of the pack.
STRATEGIES = {
    "equal": "A  equal weight",
    "invvol": "B  1/volatility",
    "cheap": "C  B tilted to LOW P/E + LOW PEG",
    "hipe_lopeg": "D  B tilted to HIGH P/E + LOW PEG",
    "quadrant": "E  only HIGH P/E + LOW PEG, by 1/vol",
    "cap_tilt": "F  T tilted to HIGH P/E + LOW PEG",
    "cap_quadrant": "G  only HIGH P/E + LOW PEG, by mkt value",
    # E and G hold ONE company in half the months (few pass both medians): the same
    # preference with at least three names, the minimum the portfolio runs keep
    "top3": "H  3 best on HIGH P/E + LOW PEG, by mkt value",
    # half the money follows T, half G, put back to 50 / 50 each month
    "half": "M  half T, half G",
    "cap": "T  top 10 by market value",
}
# colour follows the strategy in both figures (the reference palette's categorical order)
COLORS = {"equal": "#8a8984", "invvol": BLUE, "cheap": ORANGE, "hipe_lopeg": "#1baf7a", "quadrant": "#e87ba4",
          "cap_tilt": "#eda100", "cap_quadrant": "#4a3aa7", "top3": RED, "half": "#008300", "cap": INK}


def _place(x, high):
    """0..1 place among the month's ten (1 = the highest if high, else the lowest); missing 0.5."""
    n = x.notna().sum()
    r = (x.rank() - 1) / max(n - 1, 1)
    return (r if high else 1 - r).fillna(0.5).to_numpy()


def weights(g, how):
    vol = g["vol63"].fillna(g["vol63"].median()).to_numpy()
    base = g["mcap_bn"].to_numpy() if how.startswith("cap") else 1.0 / vol
    prefer = 0.5 + 0.5 * (_place(g["pe"], True) + _place(g["peg"], False))
    quad = ((g["pe"] > g["pe"].median()) & (g["peg"] <= g["peg"].median())).to_numpy()
    if how == "equal":
        w = np.ones(len(g))
    elif how in ("invvol", "cap"):
        w = base
    elif how == "cheap":
        w = base * (0.5 + 0.5 * (_place(g["pe"], False) + _place(g["peg"], False)))
    elif how in ("hipe_lopeg", "cap_tilt"):
        w = base * prefer
    elif how in ("quadrant", "cap_quadrant"):
        w = base * quad if quad.any() else base
    elif how == "half":
        cap = g["mcap_bn"].to_numpy()
        g_w = cap * quad if quad.any() else cap
        w = 0.5 * cap / cap.sum() + 0.5 * g_w / g_w.sum()
    elif how == "top3":
        # the three highest on P/E place + PEG place (ties: the larger company)
        score = _place(g["pe"], True) + _place(g["peg"], False)
        best = np.lexsort((-base, -score))[:3]
        w = np.zeros(len(g))
        w[best] = base[best]
    return w / w.sum()


def backtest(df, how, drop=()):
    """Monthly (date, net return, S&P return, T-bill return), and the weights and returns
    (months x companies) behind it. drop: companies left out (the next largest takes the
    place)."""
    rows, W, R = [], {}, {}
    held = pd.Series(dtype=float)            # last month's weights after they drifted
    for dt, g in df.groupby("date"):
        g = g[~g["ticker"].isin(drop)].sort_values("rank").head(N_TOP)
        w = pd.Series(weights(g, how), index=g["ticker"].to_numpy())
        r = pd.Series(g["ret"].to_numpy(), index=w.index)
        cost = COST * w.sub(held, fill_value=0.0).abs().sum()
        gross = float((w * r).sum())
        rows.append((dt, (1.0 - cost) * (1.0 + gross) - 1.0, float(g["bench_ret"].iloc[0]), float(g["rf"].iloc[0])))
        W[dt], R[dt] = w, r
        held = w * (1.0 + r) / (1.0 + gross)
    out = pd.DataFrame(rows, columns=["date", "ret", "bench", "rf"]).set_index("date")
    return out, pd.DataFrame(W).T.fillna(0.0), pd.DataFrame(R).T


def edge_by_company(run, base=None):
    """Each company's summed monthly contribution to a portfolio's return over the S&P
    (base None) or over another portfolio (both hold only that month's ten)."""
    m, W, R = run
    if base is None:
        c = W * R.sub(m["bench"], axis=0)
    else:
        cols = W.columns.union(base[1].columns)
        c = (W.reindex(columns=cols, fill_value=0.0) - base[1].reindex(index=W.index, columns=cols, fill_value=0.0)) \
            * R.reindex(columns=cols)
    return c.fillna(0.0).sum().sort_values(ascending=False)


def metrics(m, vs=None):
    """Return, risk and the comparison with `vs` (monthly returns; the S&P by default)."""
    vs = m["bench"] if vs is None else vs
    years = len(m) / 12.0
    cagr = np.prod(1 + m["ret"]) ** (1 / years) - 1
    cagr_b = np.prod(1 + vs) ** (1 / years) - 1
    ex = m["ret"] - m["rf"]
    value = np.cumprod(1 + m["ret"])
    dd = float((1 - value / np.maximum.accumulate(value)).max())
    rel = np.log1p(m["ret"]) - np.log1p(vs)
    t = rel.mean() / rel.std() * np.sqrt(len(rel)) if rel.std() > 0 else np.nan
    won = (rel.rolling(12).sum().dropna() > 0).mean() if len(rel) >= 12 else np.nan
    return dict(cagr=100 * cagr, vol=100 * m["ret"].std() * np.sqrt(12), sharpe=ex.mean() / ex.std() * np.sqrt(12),
                maxdd=100 * dd, vs=100 * (cagr - cagr_b), t=t, won=100 * won)


def run_periods(df):
    """{period label: (that period's company-months, {strategy: backtest})}"""
    cut = pd.Timestamp(START)
    periods = {f"{df['date'].min():%b %Y} - {cut - pd.Timedelta(days=1):%b %Y}  (a period the preference was NOT read from)":
               df[df["date"] < cut],
               f"{cut:%b %Y} - {df['date'].max():%b %Y}  (the period the preference was read from)":
               df[df["date"] >= cut]}
    return {lab: (d, {how: backtest(d, how) for how in STRATEGIES}) for lab, d in periods.items()}


def compare(d, runs, rows, base=None):
    """Metrics of each strategy in rows against base (a strategy; None = the S&P), and the
    same with the company that did most for E over that base left out of every portfolio."""
    vs = runs[base][0]["ret"] if base else None
    big = edge_by_company(runs["quadrant"], runs[base] if base else None).index[0]
    vs_without = backtest(d, base, drop=(big,))[0]["ret"] if base else None
    out = {}
    for how in rows:
        out[how] = (metrics(runs[how][0], vs), metrics(backtest(d, how, drop=(big,))[0], vs_without))
    return out, big


def draw_portfolio(res, png, lines, rows, base=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    ref = "the S&P 500" if base is None else "T (just following the top ten)"
    short = "S&P" if base is None else "T"
    fig, ax = plt.subplots(2, 2, figsize=(17, 11.5), gridspec_kw={"height_ratios": [1.45, 1]})
    for k, (lab, (d, runs)) in enumerate(res.items()):
        a = ax[0, k]
        vs = runs[base][0]["ret"] if base else runs["equal"][0]["bench"]
        ends, tags = {}, {how: STRATEGIES[how][0] for how in lines}
        first = runs["equal"][0].index[0] - pd.Timedelta(days=1)
        # the reference every line is divided by: flat at 1.00 by construction
        a.axhline(1.0, color=INK if base else INK2, lw=1.4, ls="--",
                  label=f"{STRATEGIES[base] if base else 'S&P 500 (SPY)'}: the 1.00 line, every line is divided by it")
        ends["_base"], tags["_base"] = (runs["equal"][0].index[-1], 1.0), "T" if base else "S&P"
        if base is not None:
            # the S&P against T: how far just following the ten stayed ahead of the index
            sp = np.cumprod(1 + runs["equal"][0]["bench"]) / np.cumprod(1 + vs)
            sp = pd.concat([pd.Series([1.0], index=[first]), sp])
            a.plot(sp.index, sp.to_numpy(), color=INK2, lw=1.4, ls=":", label="S&P 500 (SPY)")
            ends["_sp"], tags["_sp"] = (sp.index[-1], float(sp.iloc[-1])), "S&P"
        for how in lines:
            m = runs[how][0]
            rel = np.cumprod(1 + m["ret"]) / np.cumprod(1 + vs)
            rel = pd.concat([pd.Series([1.0], index=[first]), rel])
            a.plot(rel.index, rel.to_numpy(), color=COLORS[how], lw=2.4 if how in ("hipe_lopeg", "quadrant") else 1.6,
                   label=STRATEGIES[how])
            ends[how] = (rel.index[-1], float(rel.iloc[-1]))
        # end labels, pushed apart where lines finish close together
        lo, hi = a.get_ylim()
        placed = []
        for how, (x, v) in sorted(ends.items(), key=lambda kv: kv[1][1]):
            placed.append(max(v, placed[-1] + 0.035 * (hi - lo)) if placed else v)
            a.annotate(f"{tags[how]} {v:.2f}", (x, v), xytext=(x + pd.Timedelta(days=25), placed[-1]),
                       textcoords="data", fontsize=8, color=INK, va="center",
                       fontweight="bold" if how in ("_base", "cap") else "normal",
                       bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
        a.set_xlabel(f"1.00 = matched {ref}" + (" (SPY, dividends in)" if base is None else ""), color=INK2)
        a.set_ylabel(f"portfolio value / {'S&P 500' if base is None else 'T'} value")
        a.set_title(lab, fontsize=10)
        a.legend(loc="best", fontsize=8, frameon=False)
        a.margins(x=0.1)

        # the numbers
        a = ax[1, k]
        a.axis("off")
        cmp, big = compare(d, runs, rows, base)
        cols = ["", "return\n/ year", "volatility\n/ year", "Sharpe", "worst\ndrawdown", f"vs {short}\n/ year",
                f"t vs\n{short}", f"12-month\nspans ahead", f"vs {short} / yr\nwithout {big}"]
        cells = []
        for how in rows:
            mt, m2 = cmp[how]
            cells.append([STRATEGIES[how], f"{mt['cagr']:.1f}%", f"{mt['vol']:.1f}%", f"{mt['sharpe']:.2f}",
                          f"{mt['maxdd']:.0f}%", f"{mt['vs']:+.1f}%", f"{mt['t']:+.1f}", f"{mt['won']:.0f}%",
                          f"{m2['vs']:+.1f}%"])
        refs = [] if base is None else [(STRATEGIES[base], runs[base][0]["ret"])]
        refs.append(("S&P 500 (SPY)", runs["equal"][0]["bench"]))
        for name, r in refs:
            b = metrics(runs["equal"][0].assign(ret=r), vs)
            cells.append([name, f"{b['cagr']:.1f}%", f"{b['vol']:.1f}%", f"{b['sharpe']:.2f}", f"{b['maxdd']:.0f}%",
                          *([""] * 4 if name == refs[0][0] else [f"{b['vs']:+.1f}%", f"{b['t']:+.1f}",
                                                                 f"{b['won']:.0f}%", ""])])
        tb = a.table(cellText=cells, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[0.316] + [0.0855] * 8)
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(GRID)
            if i == 0:
                c.set_text_props(color=INK2, fontsize=7.5)
            if j == 0:
                c._loc = "left"
        a.set_title(f"monthly, 5 bp per dollar traded; t = mean monthly return over {short} / its standard error\n"
                    f"last column: {big} (the company that did most for E over {short}) left out of every portfolio, "
                    f"the 11th largest in its place", fontsize=9, color=INK2)
    what = ("weighted by 1/volatility, with and without a P/E and PEG preference" if base is None else
            "does a P/E and PEG preference beat just holding them by market value (T)?")
    fig.suptitle(f"The ten largest S&P 500 companies each month: {what}\nvolatility = the last 63 days, P/E and PEG "
                 f"as filed by that day; rebalanced at each month's first open", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- every entry and exit
# A result read at one end date depends on where the measuring stopped, and no one knows
# their entry or exit in advance. So: every pair (entry month, exit month) of 2011-2026,
# each weighted the same -- the double integral over when one starts and how long one holds.
def windows(ra, rb):
    """(M, M) annualised log return of ra over rb from the start of entry month s to the end
    of exit month e (NaN where e < s)."""
    la = np.r_[0.0, np.cumsum(np.log1p(np.asarray(ra)))]
    lb = np.r_[0.0, np.cumsum(np.log1p(np.asarray(rb)))]
    n = len(la) - 1
    s, e = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    months = (e - s + 1).astype(float)
    d = (la[e + 1] - la[s]) - (lb[e + 1] - lb[s])
    return np.where(e >= s, d / (months / 12.0), np.nan), np.where(e >= s, months, np.nan)


def window_stats(d, months, min_months=1):
    m = np.isfinite(d) & (months >= min_months)
    return dict(won=100 * (d[m] > 0).mean(), mean=100 * d[m].mean(), median=100 * np.median(d[m]), n=int(m.sum()))


WINDOW_LINES = ["hipe_lopeg", "quadrant", "cap_tilt", "cap_quadrant", "top3"]


def draw_windows(full, runs, png, base="cap", drop="NVDA"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    div = LinearSegmentedColormap.from_list("div", ["#9f2a29", RED, GREY_MID, BLUE, "#1c5cab"])
    dates = runs[base][0].index
    rb = runs[base][0]["ret"]
    rb_drop = backtest(full, base, drop=(drop,))[0]["ret"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # E against T for every entry and exit
    for a, how in ((ax[0, 0], "quadrant"), (ax[0, 1], "cap_quadrant")):
        d, months = windows(runs[how][0]["ret"], rb)
        st = window_stats(d, months)
        lim = 25.0
        edges = np.r_[dates.to_numpy(), dates[-1] + pd.Timedelta(days=31)]
        pc = a.pcolormesh(edges, edges, 100 * d.T, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), shading="flat")
        a.contour(dates, dates, np.where(np.isfinite(d.T), d.T, np.nan), levels=[0.0], colors=INK, linewidths=0.6)
        a.set_xlabel("entry month")
        a.set_ylabel("exit month")
        a.grid(False)
        a.set_title(f"{STRATEGIES[how]}  against T, every entry and exit\n"
                    f"wins {st['won']:.0f}% of the {st['n']:,} (entry, exit) pairs; mean {st['mean']:+.1f}%/yr, "
                    f"median {st['median']:+.1f}%/yr", fontsize=10)
        cb = fig.colorbar(pc, ax=a, fraction=0.046, pad=0.02, extend="both")
        cb.set_label("return over T, % per year (blue: ahead of T)")

    # share of pairs won, by how long one holds
    a = ax[1, 0]
    horizons = np.arange(1, len(dates) + 1)
    for how in WINDOW_LINES:
        d, months = windows(runs[how][0]["ret"], rb)
        won = [100 * (d[months == h] > 0).mean() for h in horizons]
        a.plot(horizons / 12, won, color=COLORS[how], lw=2.4 if how == "quadrant" else 1.6, label=STRATEGIES[how])
    d, months = windows(runs["quadrant"][0]["bench"], rb)
    a.plot(horizons / 12, [100 * (d[months == h] > 0).mean() for h in horizons], color=INK2, ls=":", lw=1.4,
           label="S&P 500 (SPY)")
    a.axhline(50, color=INK, lw=1.2, ls="--")
    a.set_xlim(0, 10)
    a.set_ylim(0, 100)
    a.set_xlabel("years held")
    a.set_ylabel("% of entry months that ended ahead of T")
    a.set_title("How often each ends ahead of T, by how long it is held\n(every entry month 2011-2026 with that much "
                "history after it; 50 = a coin toss)", fontsize=10)
    a.legend(loc="lower left", fontsize=8, frameon=False)

    # the same with the biggest winner out
    a = ax[1, 1]
    for how in WINDOW_LINES:
        d, months = windows(backtest(full, how, drop=(drop,))[0]["ret"], rb_drop)
        won = [100 * (d[months == h] > 0).mean() for h in horizons]
        a.plot(horizons / 12, won, color=COLORS[how], lw=2.4 if how == "quadrant" else 1.6, label=STRATEGIES[how])
    a.axhline(50, color=INK, lw=1.2, ls="--")
    a.set_xlim(0, 10)
    a.set_ylim(0, 100)
    a.set_xlabel("years held")
    a.set_ylabel("% of entry months that ended ahead of T")
    a.set_title(f"The same with {drop} left out of every portfolio (T too; the 11th largest in its place)",
                fontsize=10)
    a.legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"Every entry and exit, {dates[0]:%b %Y} - {dates[-1]:%b %Y}: does a HIGH P/E + LOW PEG preference "
                 f"beat just following the top ten (T)?\nmonthly, 5 bp per dollar traded", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- six months at a time
# The same strategies (still rebalanced monthly) judged over each calendar half-year, one
# after another, and over every 6-month hold starting in any month.
HALF_ROWS = ["equal", "invvol", "hipe_lopeg", "quadrant", "cap_tilt", "cap_quadrant", "top3", "cap", "_sp"]
HALF_BARS = ["quadrant", "cap_quadrant", "top3", "cap", "_sp"]


def _monthly(runs, how):
    return runs["equal"][0]["bench"] if how == "_sp" else runs[how][0]["ret"]


def half_years(runs):
    """(half-years, strategies) compounded return of each complete calendar half-year."""
    r = pd.DataFrame({how: _monthly(runs, how) for how in HALF_ROWS})
    key = r.index.year.astype(str) + np.where(r.index.month <= 6, " H1", " H2")
    out = (1 + r).groupby(key).prod() - 1
    return out[r.groupby(key).size() == 6]


def draw_half_years(full, runs, png, drop="NVDA"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    names = {**STRATEGIES, "_sp": "S&P 500 (SPY)"}
    colors = {**COLORS, "_sp": "#b8b7b1"}
    h = half_years(runs)
    runs_drop = {how: backtest(full, how, drop=(drop,)) for how in HALF_ROWS if how != "_sp"}
    h_drop = half_years(runs_drop)
    fig = plt.figure(figsize=(17, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1, 1.35])

    # every half-year, side by side
    a = fig.add_subplot(gs[0, :])
    x = np.arange(len(h))
    wdt = 0.8 / len(HALF_BARS)
    for k, how in enumerate(HALF_BARS):
        a.bar(x + (k - (len(HALF_BARS) - 1) / 2) * wdt, 100 * h[how], width=wdt * 0.9, color=colors[how],
              label=names[how])
    ahead = h["quadrant"] > h["cap"]
    top =100 * h[HALF_BARS].max(axis=1)
    for xi, won, tp in zip(x, ahead, top):
        a.annotate("E>T" if won else "", (xi, tp), xytext=(0, 3), textcoords="offset points", ha="center",
                   fontsize=7, color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x, [k.replace(" ", "\n") for k in h.index], fontsize=8)
    a.set_xlim(-0.6, len(h) - 0.4)
    a.set_ylabel("return over the half-year, %")
    a.set_title(f"Every calendar half-year, {h.index[0]} - {h.index[-1]}: E ahead of T in {int(ahead.sum())} of "
                f"{len(h)}, G ahead of T in {int((h['cap_quadrant'] > h['cap']).sum())} of {len(h)} "
                f"(\"E>T\" marks E's)", fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    # every 6-month hold, starting in any month: how far ahead of T
    a = fig.add_subplot(gs[1, 0])
    bins = np.arange(-40, 42.5, 2.5)
    for how in ("quadrant", "cap_quadrant"):
        d, months = windows(_monthly(runs, how), runs["cap"][0]["ret"])
        six = 100 * d[months == 6] / 2.0           # back from annualised to the six months' own return
        a.hist(np.clip(six, bins[0], bins[-1]), bins=bins, histtype="step", lw=2.2, color=colors[how],
               label=f"{names[how]}: ahead in {100 * (six > 0).mean():.0f}% of {len(six)}, median {np.median(six):+.1f}%")
    a.axvline(0, color=INK, lw=1.2, ls="--")
    a.set_xlabel("6-month return over T, %  (ends clipped at +-40)")
    a.set_ylabel("number of 6-month holds")
    a.set_title("Every 6-month hold, entered in any month 2011-2026", fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False)

    # the numbers
    a = fig.add_subplot(gs[1, 1])
    a.axis("off")
    cols = ["", "half-years\nahead of T", "half-years\nahead of S&P", "mean\nhalf-year", "worst\nhalf-year",
            "best\nhalf-year", "any-month 6-mo\nholds ahead of T", f"ahead of T\nwithout {drop}"]
    cells = []
    for how in HALF_ROWS:
        v = h[how]
        d, months = windows(_monthly(runs, how), runs["cap"][0]["ret"])
        roll = 100 * (d[months == 6] > 0).mean()
        cells.append([names[how],
                      "" if how == "cap" else f"{int((v > h['cap']).sum())} / {len(h)}",
                      "" if how == "_sp" else f"{int((v > h['_sp']).sum())} / {len(h)}",
                      f"{100 * v.mean():+.1f}%", f"{100 * v.min():+.1f}%", f"{100 * v.max():+.1f}%",
                      "" if how == "cap" else f"{roll:.0f}%",
                      "" if how == "cap" else f"{int((h_drop[how] > h_drop['cap']).sum())} / {len(h)}"])
    tb = a.table(cellText=cells, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1],
                 colWidths=[0.32] + [0.088] * 5 + [0.12, 0.12])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(color=INK2, fontsize=7.5)
        if j == 0:
            c._loc = "left"
    a.set_title(f"{len(h)} half-years; monthly rebalancing inside each, 5 bp per dollar traded\nlast column: {drop} left "
                f"out of every portfolio (T too), the 11th largest in its place", fontsize=9, color=INK2)
    fig.suptitle("Six months at a time: the HIGH P/E + LOW PEG portfolios against just following the top ten (T) and "
                 "the S&P 500", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return h, h_drop


# ---------------------------------------------------------------- the area between, year by year
# Day by day: the portfolios still trade once a month, but their value moves every day. The
# gap E - T is E's value over T's in % with both restarted equal at each year's first open
# (so a year's area is that year's alone, not an early jump carried forward), smoothed by a
# one-week (5-day) rolling mean; its integral over the year is in %-years: +5 means E stood
# 5% above T on average through a full year.
_PANEL = {}


def _panel():
    if "p" not in _PANEL:
        from stocks_data import load_panel
        _PANEL["p"] = load_panel()
    return _PANEL["p"]


def daily_values(run):
    """Value at each day's open of a portfolio set monthly by backtest (1.0 at its first open);
    between decision days it drifts with the prices; each month's trading cost is paid at its
    first open, so the month ends exactly at backtest's net return."""
    p = _panel()
    dates = pd.DatetimeIndex(p.dates)
    col = {t: i for i, t in enumerate(p.tickers)}
    m, W, _ = run
    starts = [dates.get_loc(d) for d in m.index] + [len(dates) - 1]
    out, value = np.full(len(dates), np.nan), 1.0
    for j, dt in enumerate(m.index):
        t, t1 = starts[j], starts[j + 1]
        w = W.loc[dt]
        w = w[w > 0]
        px = pd.DataFrame(p.open[t:t1 + 1, [col[k] for k in w.index]]).ffill().to_numpy()
        rel = (px / px[0]) @ w.to_numpy()
        path = value * rel * (1.0 + m.loc[dt, "ret"]) / rel[-1]
        out[t:t1 + 1] = path
        value = path[-1]
    return pd.Series(out, index=dates).dropna()


def yearly_gaps(a, b, smooth=5):
    """{year: smoothed daily gap of a over b, % (both restarted at the year's first open)} and
    {year: its integral over the year, %-years}."""
    gaps, area = {}, {}
    for y in sorted(set(a.index.year)):
        va, vb = a[a.index.year == y], b[b.index.year == y]
        gap = 100.0 * ((va / va.iloc[0]) / (vb / vb.iloc[0]) - 1.0)
        gap = gap.rolling(smooth, min_periods=1).mean()
        gaps[y] = gap
        area[y] = float(gap.sum() / 252.0)
    return gaps, area


def integral_pairs(runs):
    """{E, G: yearly_gaps against T}"""
    t = daily_values(runs["cap"])
    return {how: yearly_gaps(daily_values(runs[how]), t) for how in ("quadrant", "cap_quadrant")}


# the same areas brought to one date: each day's gap weighted by (1 + RATE)^-t, t in years
# from the first open (the day the money goes in) -- the integral of gap(t) (1 + RATE)^-t dt.
# Valuing at today instead multiplies every figure by the same (1 + RATE)^(years run)
DISCOUNT_RATE = 0.10


def discounted_areas(gaps, start, rate=DISCOUNT_RATE):
    """{year: that year's area, each day discounted to `start` at `rate` a year}"""
    out = {}
    for y, g in gaps.items():
        t = (g.index - start).days.to_numpy() / 365.25
        out[y] = float((g.to_numpy() * (1.0 + rate) ** -t).sum() / 252.0)
    return out


def draw_present_value(pairs, png, rate=DISCOUNT_RATE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    years = sorted(pairs["quadrant"][1])
    start = pairs["quadrant"][0][years[0]].index[0]
    last = pairs["quadrant"][0][years[-1]].index[-1]
    run_years = (last - start).days / 365.25
    pv = {how: discounted_areas(gaps, start, rate) for how, (gaps, _) in pairs.items()}
    fig, ax = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1.2, 1]})

    # each year: the area as measured (outline) and its present value (filled)
    a = ax[0]
    x = np.arange(len(years))
    wdt = 0.38
    for k, (how, (_, area)) in enumerate(pairs.items()):
        raw = np.array([area[y] for y in years])
        dis = np.array([pv[how][y] for y in years])
        xs = x + (k - 0.5) * wdt
        a.bar(xs, raw, width=wdt * 0.92, facecolor="none", edgecolor=COLORS[how], lw=1.0, ls="--")
        a.bar(xs, dis, width=wdt * 0.92, color=COLORS[how],
              label=f"{STRATEGIES[how]}: present value {dis.sum():+.1f} %-years (undiscounted {raw.sum():+.1f})")
        for xi, v in zip(xs, dis):
            a.annotate(f"{v:+.1f}", (xi, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                       ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK)
    from matplotlib.patches import Patch
    handles, labels = a.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor=INK2, linestyle="--"))
    labels.append("dashed outline: the same year undiscounted")
    a.axhline(0, color=INK, lw=1.0)
    a.set_xticks(x, [f"{y}\nx{(1 + rate) ** -((pd.Timestamp(f'{y}-07-01') - start).days / 365.25):.2f}"
                     + (f"\n(to {last:%b})" if y == years[-1] else "") for y in years])
    a.set_ylabel("area over T, %-years, in Jan-2011 value")
    a.set_title(f"Each year's area over T, discounted day by day at {100 * rate:.0f}% a year to the first open "
                f"({start:%d %b %Y}); under each year: its mid-year discount factor", fontsize=10)
    lo, hi = a.get_ylim()
    a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    a.legend(handles, labels, loc="upper left", fontsize=8, frameon=False)

    # the running sum of the discounted areas
    a = ax[1]
    for how in pairs:
        cum = np.cumsum([pv[how][y] for y in years])
        a.plot(x, cum, "-o", color=COLORS[how], lw=2.2, ms=5, label=STRATEGIES[how])
        a.annotate(f"{STRATEGIES[how][0]} {cum[-1]:+.1f}  (today's value {cum[-1] * (1 + rate) ** run_years:+.1f})",
                   (x[-1], cum[-1]), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    a.axhline(0, color=INK, lw=1.0)
    a.set_xticks(x, [str(y) for y in years])
    a.set_xlim(-0.5, len(years) + 2.5)
    a.set_ylabel("running sum, %-years, in Jan-2011 value")
    a.set_title(f"Running sum of the discounted areas (valued at today, {last:%b %Y}, every figure is "
                f"x{(1 + rate) ** run_years:.2f})", fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(f"The area of E and G over just following the top ten (T), brought to present value at "
                 f"{100 * rate:.0f}% a year\n(daily gap with each year restarted at 0, 1-week mean; +1 %-year = 1% "
                 f"ahead of T held for a whole year)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return pv


# ---------------------------------------------------------------- average return, discounted
# The average yearly return with the same weighting through time: each day's log return
# weighted by (1 + RATE)^-t, divided by the equally weighted time -- the integral of the
# return rate r(t) (1 + RATE)^-t dt over the integral of (1 + RATE)^-t dt. With no discount
# it is the compound annual rate.
RATE_ROWS = ["quadrant", "cap_quadrant", "half", "cap", "_sp"]


def sp_daily(start):
    """The S&P 500 (SPY, dividends in) at each open from start, 1.0 there."""
    p = _panel()
    s = pd.Series(p.bench_open, index=pd.DatetimeIndex(p.dates))
    s = s[s.index >= start]
    return s / s.iloc[0]


def average_rates(v, start, rate=DISCOUNT_RATE):
    """(discount-weighted, plain) average yearly return in %, from daily values v."""
    v = v[v.index >= start]
    lr = np.log(v.to_numpy())
    step = np.diff(lr)
    dt = np.diff(v.index.to_numpy()).astype("timedelta64[D]").astype(float) / 365.25
    t = (v.index[1:] - start).days.to_numpy() / 365.25
    w = (1.0 + rate) ** -t
    return 100 * (np.exp((step * w).sum() / (dt * w).sum()) - 1), 100 * (np.exp(step.sum() / dt.sum()) - 1)


def calendar_returns(v):
    """{year: % from the year's first open to the next year's first open (or the last day)}"""
    out = {}
    for y in sorted(set(v.index.year)):
        nxt = v[v.index.year == y + 1]
        out[y] = 100 * ((nxt.iloc[0] if len(nxt) else v.iloc[-1]) / v[v.index.year == y].iloc[0] - 1)
    return out


def draw_rates(runs, png, rate=DISCOUNT_RATE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    names = {**STRATEGIES, "_sp": "S&P 500 (SPY)"}
    colors = {**COLORS, "_sp": "#b8b7b1"}
    vals = {how: daily_values(runs[how]) for how in RATE_ROWS if how != "_sp"}
    start = vals["cap"].index[0]
    vals["_sp"] = sp_daily(start)
    rates = {how: average_rates(v, start, rate) for how, v in vals.items()}
    years_ret = {how: calendar_returns(v) for how, v in vals.items()}
    years = sorted(years_ret["cap"])
    last = vals["cap"].index[-1]
    fig, ax = plt.subplots(2, 1, figsize=(16, 11.5), gridspec_kw={"height_ratios": [1.3, 1]})

    # every calendar year
    a = ax[0]
    x = np.arange(len(years))
    wdt = 0.8 / len(RATE_ROWS)
    for k, how in enumerate(RATE_ROWS):
        a.bar(x + (k - (len(RATE_ROWS) - 1) / 2) * wdt, [years_ret[how][y] for y in years], width=wdt * 0.9,
              color=colors[how], label=names[how])
    a.axhline(0, color=INK, lw=1.0)
    a.axhline(100 * rate, color=INK2, lw=1.0, ls=":", label=f"{100 * rate:.0f}% a year (the discount rate)")
    a.set_xticks(x, [f"{y}\nx{(1 + rate) ** -((pd.Timestamp(f'{y}-07-01') - start).days / 365.25):.2f}"
                     + (f"\n(to {last:%b})" if y == years[-1] else "") for y in years])
    a.set_ylabel("return over the calendar year, %")
    a.set_title("Every calendar year's return (under each year: the weight its days get, the mid-year discount factor)",
                fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False, ncol=5)

    # the averages
    a = ax[1]
    y = np.arange(len(RATE_ROWS))[::-1]
    sp_d, sp_p = rates["_sp"]
    for yi, how in zip(y, RATE_ROWS):
        d, pl = rates[how]
        a.barh(yi, pl, height=0.62, facecolor="none", edgecolor=colors[how] if how != "_sp" else INK2, ls="--", lw=1.2)
        a.barh(yi, d, height=0.42, color=colors[how])
        extra = "" if how == "_sp" else f";  over the S&P {d - sp_d:+.1f} (plain {pl - sp_p:+.1f})"
        a.annotate(f"{d:.1f}% a year discounted average,  {pl:.1f}% plain{extra}", (max(d, pl), yi), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=9, color=INK)
    a.axvline(100 * rate, color=INK2, lw=1.0, ls=":")
    a.set_yticks(y, [names[h] for h in RATE_ROWS])
    a.set_xlim(0, max(max(r) for r in rates.values()) * 1.9)
    a.set_xlabel("average yearly return, %")
    a.grid(axis="y", visible=False)
    a.set_axisbelow(True)
    a.legend([Patch(color=INK2), Patch(facecolor="none", edgecolor=INK2, linestyle="--")],
             [f"discounted average: each day's return weighted by {1 + rate:.2f}^-t (t in years from {start:%b %Y})",
              "plain average: the compound annual rate"], loc="lower right", fontsize=8, frameon=False)
    a.set_title(f"Average yearly return, {start:%b %Y} - {last:%b %Y}", fontsize=10)
    fig.suptitle(f"The average yearly gain with the {100 * rate:.0f}% discount: E and G against just following the top "
                 f"ten (T) and the S&P 500\nmonthly rebalancing, 5 bp per dollar traded; values at each open, "
                 f"dividends in", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return rates, years_ret


# ---------------------------------------------------------------- IRR by how long one stays in
# One investment at the first open, everything carried forward (no money in or out after):
# the internal rate of return up to a date is the rate that turns the starting value into the
# value then, (V / V0)^(1 / years) - 1. It changes as each year is added. Next to it the net
# present value at DISCOUNT_RATE of INVESTED put in at the start: V / (1 + rate)^years - 1.
INVESTED = 100_000


def irr_by_horizon(v, rate=DISCOUNT_RATE):
    """(date, years held, IRR %, NPV of INVESTED at rate) at each year's first open after the
    start and at the last day."""
    start = v.index[0]
    ends = [v[v.index.year == y].index[0] for y in sorted(set(v.index.year))[1:]] + [v.index[-1]]
    rows = []
    for d in ends:
        yrs = (d - start).days / 365.25
        growth = v[d] / v.iloc[0]
        rows.append((d, yrs, 100 * (growth ** (1 / yrs) - 1), INVESTED * (growth / (1 + rate) ** yrs - 1)))
    return pd.DataFrame(rows, columns=["date", "years", "irr", "npv"]).set_index("date")


def draw_irr(runs, png, rate=DISCOUNT_RATE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    names = {**STRATEGIES, "_sp": "S&P 500 (SPY)"}
    colors = {**COLORS, "_sp": "#8a8984"}
    vals = {how: daily_values(runs[how]) for how in RATE_ROWS if how != "_sp"}
    start = vals["cap"].index[0]
    vals["_sp"] = sp_daily(start)
    tabs = {how: irr_by_horizon(v, rate) for how, v in vals.items()}
    fig, ax = plt.subplots(2, 1, figsize=(16, 11.5), gridspec_kw={"height_ratios": [1.2, 1]})
    for a, col, unit in ((ax[0], "irr", "%"), (ax[1], "npv", "$")):
        ends = {}
        for how in RATE_ROWS:
            tb = tabs[how]
            a.plot(tb["years"], tb[col], "-o", color=colors[how], lw=2.4 if how in ("quadrant", "cap_quadrant") else 1.8,
                   ms=4.5, ls="--" if how == "_sp" else "-", label=names[how])
            ends[how] = (tb["years"].iloc[-1], float(tb[col].iloc[-1]))
        lo, hi = a.get_ylim()
        placed = []
        for how, (x, v) in sorted(ends.items(), key=lambda kv: kv[1][1]):
            placed.append(max(v, placed[-1] + 0.045 * (hi - lo)) if placed else v)
            txt = f"{STRATEGIES.get(how, 'S&P')[0] if how != '_sp' else 'S&P'} " + \
                (f"{v:.1f}%" if unit == "%" else f"${v / 1e3:,.0f}k")
            a.annotate(txt, (x, v), xytext=(x + 0.25, placed[-1]), textcoords="data", fontsize=8.5, color=INK,
                       va="center", fontweight="bold" if how in ("quadrant", "cap_quadrant") else "normal")
        a.axhline(100 * rate if col == "irr" else 0, color=INK, lw=1.0, ls=":")
        ticks = tabs["cap"]
        a.set_xticks(ticks["years"], [f"{d:%b %Y}\n{y:.0f}y" if i < len(ticks) - 1 else f"{d:%b %Y}\n{y:.1f}y"
                                     for i, (d, y) in enumerate(zip(ticks.index, ticks["years"]))], fontsize=8)
        a.set_xlim(0.6, ticks["years"].iloc[-1] + 1.6)
        a.legend(loc="lower center" if col == "irr" else "upper left", fontsize=8, frameon=False,
                 ncol=2 if col == "irr" else 1)
    ax[0].set_ylabel("IRR, % a year")
    ax[0].set_title(f"IRR of one investment made {start:%d %b %Y} and held to each date (the dotted line: "
                    f"{100 * rate:.0f}%, the discount rate)", fontsize=10)
    ax[1].set_ylabel(f"net present value at {100 * rate:.0f}%, $")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1e3:,.0f}k"))
    ax[1].set_title(f"Net present value at {100 * rate:.0f}% a year of ${INVESTED:,} put in on {start:%d %b %Y}, "
                    f"held to each date (0 = earned exactly {100 * rate:.0f}% a year)", fontsize=10)
    ax[1].set_xlabel("held until (years since the investment)")
    fig.suptitle("How the IRR evolves as the investment is held longer: E and G against just following the top ten "
                 "(T) and the S&P 500\nmonthly rebalancing, 5 bp per dollar traded, dividends reinvested, nothing "
                 "taken out", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return tabs


def draw_integrals(pairs, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    years = sorted(pairs["quadrant"][1])
    last = pairs["quadrant"][0][years[-1]].index[-1]
    fig, ax = plt.subplots(3, 1, figsize=(17, 13), gridspec_kw={"height_ratios": [1, 1, 1.15]})
    for a, (how, (gaps, area)) in zip(ax[:2], pairs.items()):
        for y in years:
            g = gaps[y]
            a.fill_between(g.index, g.to_numpy(), 0, where=g.to_numpy() >= 0, color=BLUE, alpha=0.28, lw=0,
                           interpolate=True)
            a.fill_between(g.index, g.to_numpy(), 0, where=g.to_numpy() < 0, color=RED, alpha=0.28, lw=0,
                           interpolate=True)
            a.plot(g.index, g.to_numpy(), color=COLORS[how], lw=1.3)
            a.axvline(g.index[0], color=GRID, lw=1.0, zorder=0)
        a.axhline(0, color=INK, lw=1.0)
        pos = sum(v for v in area.values() if v > 0)
        neg = sum(v for v in area.values() if v < 0)
        a.set_ylabel(f"{STRATEGIES[how][0]} over T, %\n(1-week mean)")
        a.set_title(f"{STRATEGIES[how]}  minus  T (top 10 by market value), each year restarted at 0; blue: ahead of T, "
                    f"red: behind.  Area ahead {pos:+.1f}, behind {neg:+.1f}, total {pos + neg:+.1f} %-years",
                    fontsize=10)
        a.set_xlim(gaps[years[0]].index[0], last)

    # the integrals, year by year
    a = ax[2]
    x = np.arange(len(years))
    wdt = 0.38
    for k, (how, (gaps, area)) in enumerate(pairs.items()):
        vals = np.array([area[y] for y in years])
        a.bar(x + (k - 0.5) * wdt, vals, width=wdt * 0.92, color=COLORS[how],
              label=f"{STRATEGIES[how]}: ahead in {int((vals > 0).sum())} of {len(vals)} years, "
                    f"total {vals.sum():+.1f} %-years")
        for xi, v in zip(x, vals):
            a.annotate(f"{v:+.1f}", (xi + (k - 0.5) * wdt, v), xytext=(0, 3 if v >= 0 else -3),
                       textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
                       color=INK)
    a.axhline(0, color=INK, lw=1.0)
    a.set_xticks(x, [f"{y}" + (f"\n(to {last:%b})" if y == years[-1] else "") for y in years])
    a.set_ylabel("integral of the gap over the year, %-years")
    lo, hi = a.get_ylim()
    a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    a.set_title("Each year's area between the line and 0 (+5 = on average 5% ahead of T through the whole year)",
                fontsize=10)
    a.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("How far E and G stood above or below just following the top ten (T), day by day, and its integral "
                 "per year\nmonthly rebalancing, 5 bp per dollar traded; daily values at each open, smoothed by a 1-week "
                 "rolling mean", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return {how: area for how, (_, area) in pairs.items()}


if __name__ == "__main__":
    full = build(PORT_START, N_TOP + 1)       # the 11th largest stands in when one is left out
    df = full[(full["date"] >= pd.Timestamp(START)) & (full["rank"] <= N_TOP)].reset_index(drop=True)
    png = os.path.join(OUT, "valuation_top10.png")
    draw(df, png)
    print(f"{len(df)} company-months, companies: {sorted(df['ticker'].unique())}")
    for x in ("pe", "peg"):
        for h in HORIZONS:
            ic, t, n = monthly_ic(df, x, f"ex{h}")
            print(f"  {x:3s} vs next {h:3d} days: within-month rank correlation {ic:+.3f} (t {t:+.2f}, {n} months)")
    last = df[df["date"] == df["date"].max()].sort_values("rank")
    print(f"\nlatest month ({df['date'].max():%Y-%m-%d}):")
    print(last[["rank", "ticker", "mcap_bn", "pe", "eps_growth", "peg"]].round(2).to_string(index=False))
    print("wrote", png)

    print("\nthe ten largest on the first decision day of each year:")
    for y, g in full[full["rank"] <= N_TOP].groupby(full["date"].dt.year):
        g = g[g["date"] == g["date"].min()]
        print(f"  {y}: {' '.join(g.sort_values('rank')['ticker'])}   (no P/E {g['pe'].isna().sum()}, "
              f"no PEG {g['peg'].isna().sum()})")
    res = run_periods(full)
    res_all = {"all": (full, {how: backtest(full, how) for how in STRATEGIES})}
    for lab, (d, runs) in {**res, f"{full['date'].min():%b %Y} - {full['date'].max():%b %Y} (both)": res_all["all"]}.items():
        print(f"\n{lab}")
        for base in (None, "cap"):
            rows = [h for h in STRATEGIES if h != base]
            cmp, big = compare(d, runs, rows, base)
            short = "S&P" if base is None else "T"
            print(f"  against {'the S&P 500' if base is None else 'T, the top 10 by market value'}:")
            for how in rows:
                mt, m2 = cmp[how]
                print(f"    {STRATEGIES[how]:42s} {mt['cagr']:5.1f}%/yr  vol {mt['vol']:4.1f}%  Sharpe {mt['sharpe']:.2f}  "
                      f"worst dd {mt['maxdd']:2.0f}%  vs {short} {mt['vs']:+5.1f}%/yr (t {mt['t']:+.1f}, ahead in "
                      f"{mt['won']:3.0f}% of 12-month spans); without {big} {m2['vs']:+5.1f}%/yr")
            c = edge_by_company(runs["quadrant"], runs[base] if base else None)
            print(f"    E over {short} by company (summed monthly, %):",
                  ", ".join(f"{k} {100 * v:+.0f}" for k, v in c.head(5).items()), "...",
                  ", ".join(f"{k} {100 * v:+.0f}" for k, v in c.tail(3).items()))
    png2 = os.path.join(OUT, "valuation_portfolio.png")
    draw_portfolio(res, png2, lines=["equal", "invvol", "cheap", "hipe_lopeg", "quadrant", "cap"],
                   rows=["equal", "invvol", "cheap", "hipe_lopeg", "quadrant", "cap"])
    png3 = os.path.join(OUT, "valuation_vs_top10.png")
    draw_portfolio(res, png3, lines=["invvol", "hipe_lopeg", "quadrant", "cap_tilt", "cap_quadrant"],
                   rows=["equal", "invvol", "hipe_lopeg", "quadrant", "cap_tilt", "cap_quadrant"], base="cap")
    print("wrote", png2, "and", png3)

    runs = res_all["all"][1]
    print(f"\nevery (entry month, exit month) pair, {full['date'].min():%b %Y} - {full['date'].max():%b %Y}, against T:")
    rb = runs["cap"][0]["ret"]
    rb_drop = backtest(full, "cap", drop=("NVDA",))[0]["ret"]
    for how in [h for h in STRATEGIES if h != "cap"] + ["_sp"]:
        ra = runs["equal"][0]["bench"] if how == "_sp" else runs[how][0]["ret"]
        d, months = windows(ra, rb)
        parts = [window_stats(d, months, k) for k in (1, 12, 36, 60)]
        if how == "_sp":
            d2 = windows(runs["equal"][0]["bench"], rb_drop)[0]
        else:
            d2 = windows(backtest(full, how, drop=("NVDA",))[0]["ret"], rb_drop)[0]
        w2 = window_stats(d2, months, 1)
        print(f"  {('S&P 500' if how == '_sp' else STRATEGIES[how]):42s} wins {parts[0]['won']:3.0f}% of pairs "
              f"(held 1y+ {parts[1]['won']:3.0f}%, 3y+ {parts[2]['won']:3.0f}%, 5y+ {parts[3]['won']:3.0f}%); "
              f"mean {parts[0]['mean']:+5.1f}%/yr, median {parts[0]['median']:+5.1f}%/yr | without NVDA: wins "
              f"{w2['won']:3.0f}%, median {w2['median']:+5.1f}%/yr")
    png4 = os.path.join(OUT, "valuation_windows.png")
    draw_windows(full, runs, png4)
    print("wrote", png4)

    png5 = os.path.join(OUT, "valuation_6months.png")
    h, h_drop = draw_half_years(full, runs, png5)
    print(f"\nevery calendar half-year {h.index[0]} - {h.index[-1]} ({len(h)}), return %:")
    print((100 * h.rename(columns={k: (STRATEGIES.get(k, "S&P")[:2].strip()) for k in h.columns})).round(1).to_string())
    for how in HALF_ROWS:
        if how != "cap":
            print(f"  {STRATEGIES.get(how, 'S&P 500'):42s} ahead of T in {int((h[how] > h['cap']).sum()):2d} of {len(h)} "
                  f"half-years (without NVDA {int((h_drop[how] > h_drop['cap']).sum()):2d}); mean over T "
                  f"{100 * (h[how] - h['cap']).mean():+.1f}% per half-year")
    print("wrote", png5)

    png6 = os.path.join(OUT, "valuation_integrals.png")
    pairs = integral_pairs(runs)
    areas = draw_integrals(pairs, png6)
    print("\nintegral of the daily gap over T per year, %-years (each year restarted at 0, 1-week mean):")
    print(pd.DataFrame({STRATEGIES[k][0] + "-T": v for k, v in areas.items()}).round(1).T.to_string())
    print("wrote", png6)

    png7 = os.path.join(OUT, "valuation_present_value.png")
    pv = draw_present_value(pairs, png7)
    print(f"\nthe same, discounted day by day at {100 * DISCOUNT_RATE:.0f}% a year to the first open (%-years):")
    tab = pd.DataFrame({STRATEGIES[k][0] + "-T": v for k, v in pv.items()})
    print(tab.round(2).T.to_string())
    print("  totals:", ", ".join(f"{c} {tab[c].sum():+.2f} (undiscounted {sum(areas[k].values()):+.1f})"
                                 for c, k in zip(tab.columns, pv)))
    print("wrote", png7)

    png8 = os.path.join(OUT, "valuation_rates.png")
    rates, years_ret = draw_rates(runs, png8)
    print(f"\naverage yearly return, discounted at {100 * DISCOUNT_RATE:.0f}% (and plain):")
    for how, (d, pl) in rates.items():
        print(f"  {STRATEGIES.get(how, 'S&P 500'):42s} {d:5.1f}% discounted average   {pl:5.1f}% plain;  over the S&P "
              f"{d - rates['_sp'][0]:+.1f} (plain {pl - rates['_sp'][1]:+.1f})")
    print(pd.DataFrame({STRATEGIES.get(k, "S&P")[:2].strip(): v for k, v in years_ret.items()}).round(1).T.to_string())
    print("wrote", png8)

    png9 = os.path.join(OUT, "valuation_irr.png")
    tabs = draw_irr(runs, png9)
    short = {k: STRATEGIES.get(k, "S&P")[:2].strip() for k in tabs}
    print(f"\nIRR (% a year) of one investment at the first open, held to each date:")
    print(pd.DataFrame({short[k]: t["irr"] for k, t in tabs.items()}).round(1).T.to_string())
    print(f"net present value at {100 * DISCOUNT_RATE:.0f}% of ${INVESTED:,} ($k):")
    print(pd.DataFrame({short[k]: t["npv"] / 1e3 for k, t in tabs.items()}).round(0).T.to_string())
    print("wrote", png9)

    # how often each one actually beat the S&P, every way we have measured it
    print("\nagainst the S&P 500, Jan 2011 - Sep 2026:")
    sp = runs["cap"][0]["bench"]
    sp_drop = backtest(full, "cap", drop=("NVDA",))[0]["bench"]
    for how in ("cap", "cap_quadrant", "half", "quadrant"):
        m, W, _ = runs[how]
        d, months = windows(m["ret"], sp)
        d2 = windows(backtest(full, how, drop=("NVDA",))[0]["ret"], sp_drop)[0]
        mt = metrics(m)
        yr = years_ret[how] if how in years_ret else calendar_returns(daily_values(runs[how]))
        behind_years = [y for y in yr if yr[y] < years_ret["_sp"][y]]
        behind_h = [f"{t:.0f}y" for t, a, b in zip(tabs["cap"]["years"], tabs[how]["irr"], tabs["_sp"]["irr"]) if a < b] \
            if how in tabs else []
        print(f"  {STRATEGIES[how]:26s} IRR {tabs[how]['irr'].iloc[-1] if how in tabs else float('nan'):.1f}% | "
              f"ahead of the S&P in {100 * (d[np.isfinite(d)] > 0).mean():.0f}% of (entry, exit) pairs, "
              f"{100 * (d[months >= 36] > 0).mean():.0f}% of holds of 3y+ (without NVDA {100 * (d2[np.isfinite(d2)] > 0).mean():.0f}%) | "
              f"behind it in {len(behind_years)} of {len(yr)} calendar years {behind_years} | IRR from Jan 2011 behind at "
              f"{behind_h or 'no horizon'} | vol {mt['vol']:.1f}%, worst drawdown {mt['maxdd']:.0f}% | largest single "
              f"weight: median {100 * W.max(axis=1).median():.0f}%, max {100 * W.max(axis=1).max():.0f}%")
