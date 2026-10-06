"""Signal audit: does any input the trees read predict what they are trying to predict?

The tree search judges every idea by one number per portfolio episode, a few hundred
overlapping episodes that hold ~10-20 independent half-years; only a huge effect could pass.
The data hold far more: every month ~116 stocks each have a next-month return. This asks,
input by input, the question the standard way (information coefficients), with the power
those ~19,000 stock-months give:

    STOCK SELECTION   on every decision day (every DECIDE_EVERY trading days) the Spearman
                      rank correlation, across that day's tradable universe, between an input
                      and the stocks' return over the next DECIDE_EVERY days (open to open) --
                      the rank IC; its mean over decision days, and a t-statistic with a
                      Newey-West variance (lag 2: the windows do not overlap, the lag guards
                      against slow regimes)
    MARKET TIMING     for every market input, the correlation over decision days between its
                      value and the S&P 500's (SPY, dividends in) next-window return, with the
                      same t-statistic -- a few hundred observations, so far less power
    MULTIPLE TESTING  some 70 inputs are tested: an input "passes" only if |t| clears the
                      Bonferroni bar (two-sided 5% over all of them), not just 2

Both periods are shown: 2006-2019 (what the trees are trained on) and 2020-2026. A real
signal has the same sign in both. An IC of 0.02-0.05 that holds is what the literature calls
a usable factor; an IC indistinguishable from 0 in 2006-2019 is something no method can learn.

EARNINGS SURPRISES (earnings.py) are audited with the rest when data/earnings.npz exists: they
are candidates, not yet inputs of the trees. They start in 2010-11, so on 2006-2019 their
ICs rest on 2011-2019.

HORIZON. --h 63 asks about the next three months instead of the next window (fundamentals and
momentum act over quarters); decision days stay one window apart, so the three-month returns
overlap and the Newey-West lag grows with the overlap.

    python signal_audit.py [--h 63]   writes results/signal_audit/signal_audit_h<h>.md and .png
"""
import os

import numpy as np

import portfolio_bt as pb

OUT = os.path.join(pb.ROOT, "results", "signal_audit")
PERIODS = {"2006-2019": ("2006-01-01", "2019-12-31"), "2020-2026": ("2020-01-01", pb.END)}
MIN_STOCKS = 20


def nw_t(x, lag=2):
    """t-statistic of mean(x) with a Newey-West (Bartlett) variance."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    v = d @ d / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (d[j:] @ d[:-j]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def rank(a):
    """Ranks along axis 1 (NaN stays NaN), for Spearman correlations."""
    import pandas as pd
    return pd.DataFrame(a).rank(axis=1).to_numpy()


def audit(panel, F, M, sn, mn, start, end, h, step):
    dates = panel.dates
    a = np.searchsorted(dates, np.datetime64(start))
    b = np.searchsorted(dates, np.datetime64(end), "right") - h - 1
    days = np.arange(a, b, step)
    lag = max(2, int(np.ceil(h / step)) + 1)          # overlapping returns: a longer NW lag
    opn = panel.open
    fwd = opn[days + h] / opn[days] - 1.0                                  # (D, N)
    ok = panel.universe[days] & np.isfinite(fwd)
    r_rank = rank(np.where(ok, fwd, np.nan))
    stock = {}
    for j, name in enumerate(sn):
        x = np.where(ok, F[days, :, j], np.nan)
        x_rank = rank(x)
        ics = []
        for d in range(len(days)):
            m = np.isfinite(x_rank[d]) & np.isfinite(r_rank[d])
            if m.sum() >= MIN_STOCKS and np.std(x_rank[d, m]) > 0:
                ics.append(np.corrcoef(x_rank[d, m], r_rank[d, m])[0, 1])
        ics = np.array(ics)
        stock[name] = dict(ic=float(np.nanmean(ics)) if len(ics) else np.nan, t=nw_t(ics, lag), n=len(ics),
                           hit=float(np.mean(ics > 0)) if len(ics) else np.nan)
    bench = panel.bench_open
    spx_fwd = bench[days + h] / bench[days] - 1.0
    market = {}
    for j, name in enumerate(mn):
        x = M[days, j]
        m = np.isfinite(x) & np.isfinite(spx_fwd)
        if m.sum() < 20 or np.std(x[m]) == 0:
            market[name] = dict(corr=np.nan, t=np.nan, n=int(m.sum()))
            continue
        # the t of a regression slope = t of mean(z_x * z_y) with the same NW variance
        zx = (x[m] - x[m].mean()) / x[m].std()
        zy = (spx_fwd[m] - spx_fwd[m].mean()) / spx_fwd[m].std()
        market[name] = dict(corr=float(np.corrcoef(x[m], spx_fwd[m])[0, 1]), t=nw_t(zx * zy, lag), n=int(m.sum()))
    return stock, market


def main():
    import pandas as pd
    from scipy import stats
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    import earnings
    if os.path.exists(earnings.CACHE):
        with np.load(earnings.CACHE) as z:
            if list(z["tickers"]) == list(panel.tickers) and len(z["dates"]) == len(panel.dates):
                F, sn = np.concatenate([F, z["F"]], axis=2), sn + [str(n) for n in z["names"]]
    # THE COMPOSITE, stated before its result was seen: the mean rank of the weak signals that
    # kept their sign in both periods (the earnings and revenue surprises, release-dated, the
    # model's mean-forecast rank, 12-month momentum), re-ranked within the day's universe.
    # Its members were picked from this audit's 2006-2019 numbers, so 2020-2026 is its test.
    members = [m for m in ("rank_sue", "rank_rev_sue", "rank_mu_21d", "rank_mom_12_1") if m in sn]
    if len(members) >= 3:
        X = np.stack([F[:, :, sn.index(m)] for m in members], axis=2)
        n_ok = np.isfinite(X).sum(2)
        avg = np.where(n_ok >= 2, np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=2), np.nan)
        comp = pd.DataFrame(np.where(panel.universe, avg, np.nan)).rank(axis=1, pct=True).to_numpy()
        F, sn = np.concatenate([F, comp[:, :, None].astype(F.dtype)], axis=2), sn + ["COMPOSITE (pre-stated)"]
    step = pb.DECIDE_EVERY
    h = int(pb._flag_value("--h") or step)
    res = {p: audit(panel, F, M, sn, mn, a, b, h, step) for p, (a, b) in PERIODS.items()}
    n_tests = len(sn) + len(mn)
    bar = float(stats.norm.ppf(1 - 0.025 / n_tests))                  # Bonferroni, two-sided 5%
    p0, p1 = list(PERIODS)
    rows = []
    for name in sn:
        s0, s1 = res[p0][0][name], res[p1][0][name]
        rows.append(dict(input=name, **{f"IC {p0}": s0["ic"], f"t {p0}": s0["t"], f"hit {p0}": s0["hit"],
                                        f"IC {p1}": s1["ic"], f"t {p1}": s1["t"]},
                         passes="YES" if abs(s0["t"]) > bar else ("|t|>2" if abs(s0["t"]) > 2 else ""),
                         same_sign="yes" if np.sign(s0["ic"]) == np.sign(s1["ic"]) else "NO"))
    st = pd.DataFrame(rows).set_index("input")
    st = st.reindex(st[f"t {p0}"].abs().sort_values(ascending=False).index).round(3)
    rows = []
    for name in mn:
        s0, s1 = res[p0][1][name], res[p1][1][name]
        rows.append(dict(input=name, **{f"corr {p0}": s0["corr"], f"t {p0}": s0["t"], f"corr {p1}": s1["corr"],
                                        f"t {p1}": s1["t"]},
                         passes="YES" if abs(s0["t"]) > bar else ("|t|>2" if abs(s0["t"]) > 2 else ""),
                         same_sign="yes" if np.sign(s0["corr"]) == np.sign(s1["corr"]) else "NO"))
    mk = pd.DataFrame(rows).set_index("input")
    mk = mk.reindex(mk[f"t {p0}"].abs().sort_values(ascending=False).index).round(3)
    n_months = res[p0][0][sn[0]]["n"]
    lines = ["# Signal audit", "", __doc__.split("    python")[0].strip(), "",
             f"Return horizon {h} trading days, decision days every {step}; {n_months} decision days in {p0}; "
             f"{n_tests} inputs tested, "
             f"Bonferroni bar |t| > {bar:.2f}.", "",
             "## Stock selection: rank IC (input vs next-window return, across the day's universe)", "",
             "```", st.to_string(), "```", "",
             "## Market timing: correlation (input vs the S&P 500's next-window return)", "",
             "```", mk.to_string(), "```", ""]
    n_pass = int((st["passes"] == "YES").sum() + (mk["passes"] == "YES").sum())
    both = st[(st["passes"] != "") & (st["same_sign"] == "yes")].index.tolist()
    lines.append(f"**{n_pass} inputs clear the Bonferroni bar on {p0}; stock inputs with |t| > 2 there and the "
                 f"same sign on {p1}: {both or 'none'}.**")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"signal_audit_h{h}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[3:]))
    plot(st, mk, p0, p1, bar, h)


def plot(st, mk, p0, p1, bar, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.24 * len(st))))
    for ax, df, title in ((axes[0], st, "stock selection: rank IC t-statistic"),
                          (axes[1], mk, "market timing: correlation t-statistic")):
        df = df.iloc[::-1]
        y = np.arange(len(df))
        ax.barh(y - 0.2, df[f"t {p0}"], height=0.4, color="#1f77b4", label=p0)
        ax.barh(y + 0.2, df[f"t {p1}"], height=0.4, color="#ff7f0e", label=p1)
        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=7)
        for v, ls in ((bar, "--"), (2, ":")):
            ax.axvline(v, color="black", lw=0.8, ls=ls)
            ax.axvline(-v, color="black", lw=0.8, ls=ls)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{title}\n(dashed: Bonferroni bar {bar:.2f}; dotted: |t| = 2)", fontsize=10)
        ax.grid(True, axis="x", color="0.9")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"return horizon {h} trading days", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"signal_audit_h{h}.png"), dpi=110)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"signal_audit_h{h}.md"), f"and signal_audit_h{h}.png")


if __name__ == "__main__":
    main()
