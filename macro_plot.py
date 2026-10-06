"""What the hourly tree DOES, drawn: one continuous account from 2012 (macro_bt.py redraws it to
<results dir>/learned.png as the search adopts trees and at every stage's end; the report adds
learned_final.png, through the test years).

    value over cash (log)      the tree against holding the basket, the 60-day trend rule
                               (decided daily) and the US 500 at 1x -- every account's return
                               over cash, so a flat line is cash
    positions                  each instrument's position through time (daily mean of the hourly
                               position: +1 long, -1 short): what the tree actually holds
    drawdown                   the fall from the running high
    the tree, and per period   CER (the score, % per year over cash), return, volatility, Sharpe,
                               max drawdown, turnover and trades per week

Years are shaded by their role: held-out (the current stage accepts on them), check,
validation (never read by the search) and test. One path from one start: what the tree does,
not how sure one can be of it (the report's bootstrap is for that).
"""
import os
import time

import numpy as np

SPANS = [("2012-2019", "2012-01-01", "2019-12-31"), ("2020-2021", "2020-01-01", "2021-12-31"),
         ("2022-2026", "2022-01-01", "2026-12-31")]
ROLE_COLOURS = {"held-out": "#cfe3f5", "check": "#fbe0c3", "validation": "#d5efd0", "test": "#f6d0d0"}


def metrics(x, A, gamma, pos=None, turnover=None):
    """Per-bar excess returns x -> {cer, ret, vol, sharpe, dd, turnover/yr, trades/wk, in market}."""
    w = np.r_[1.0, np.cumprod(1 + x)]
    sd = x.std(ddof=1)
    out = dict(cer=100 * A * (x.mean() - 0.5 * gamma * x.var(ddof=1)),
               ret=100 * (w[-1] ** (A / len(x)) - 1), vol=100 * sd * np.sqrt(A),
               sharpe=x.mean() / sd * np.sqrt(A) if sd > 0 else 0.0,
               dd=100 * (w / np.maximum.accumulate(w) - 1).min())
    if pos is not None:
        ch = (np.diff(pos.astype(np.int16), axis=0) != 0).sum()
        out.update(trades_wk=ch / (len(x) / 120.0), in_mkt=100 * float(np.mean(pos != 0)))
    if turnover is not None:
        out["turn_yr"] = turnover * A / len(x)
    return out


def describe(bank, names, max_lines=12, width=100):
    import textwrap
    from btind.memory import emit
    lines = []
    for ln in emit(bank, names).splitlines():
        lead = len(ln) - len(ln.lstrip(" |\\-"))
        lines += textwrap.wrap(ln, width, subsequent_indent=" " * (lead + 6), break_long_words=True) or [""]
    return lines[:max_lines] + ([f"... ({len(lines) - max_lines} more lines: see the report)"]
                                if len(lines) > max_lines else [])


def draw(env, bank, baselines, out_png, end, title, final=False):
    """`baselines` {name: callable(start, end) -> per-bar excess returns on the same bars}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    start = SPANS[0][1]
    tree = env.path(bank, start, end)
    times = tree["times"].values.astype("datetime64[ns]")
    series = [(n, f(start, end)) for n, f in baselines.items()] + [("the tree", tree["ret"])]
    colours = iter(["#8c8c8c", "#e07b00", "#2ca02c", "#9467bd", "#8c564b"])

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.0, 0.7, 0.75], hspace=0.3)
    ax_g, ax_p, ax_d = (fig.add_subplot(gs[i, 0]) for i in range(3))
    yrs = lambda s: [int(y) for y in s.split(",")] if s else []
    roles = {} if final else {"held-out": yrs(env.holdout)}
    roles["check"] = yrs(env.check)
    roles["validation"] = [2020, 2021]
    if final:
        roles["test"] = list(range(2022, 2027))
    for ax in (ax_g, ax_d):
        for role, years in roles.items():
            for y in years:
                ax.axvspan(np.datetime64(f"{y}-01-01"), np.datetime64(f"{y + 1}-01-01"), color=ROLE_COLOURS[role],
                           lw=0, zorder=0)
    for ax in (ax_g, ax_p, ax_d):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(times[0], times[-1])
    for ax in (ax_g, ax_d):
        ax.grid(True, color="0.8", lw=0.6, zorder=1)
    for name, x in series:
        c = "#1f5fbf" if name == "the tree" else next(colours)
        lw = 2.2 if name == "the tree" else 1.2
        w = 100_000 * np.cumprod(1 + x)
        k = slice(None, None, 6)                                   # every 6th bar is plenty to draw
        ax_g.plot(times[k], w[k], color=c, lw=lw, label=f"{name}  (${w[-1] / 1000:,.0f}k)", zorder=3)
        dd = 100 * (w / np.maximum.accumulate(np.r_[100_000, w])[1:] - 1)
        ax_d.plot(times[k], dd[k], color=c, lw=lw, zorder=3)
    ax_g.set_yscale("log")
    ax_g.set_ylabel("$100k over cash (log scale)")
    ax_g.legend(loc="upper left", fontsize=9)
    ax_g.set_title(title, fontsize=12)
    ax_d.set_ylabel("drawdown, %")
    ax_d.axhline(0, color="black", lw=0.8)
    ax_d.legend(handles=[Patch(color=ROLE_COLOURS[r], label=r + " years") for r in roles], loc="lower left",
                fontsize=8, ncol=len(roles))
    # positions: daily mean per instrument
    import pandas as pd
    P = pd.DataFrame(tree["pos"].astype(float), index=tree["times"]).resample("D").mean().dropna(how="all")
    ax_p.imshow(P.to_numpy().T, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest",
                extent=(mdates.date2num(P.index[0]), mdates.date2num(P.index[-1] + pd.Timedelta(days=1)),
                        len(env.asset_names) - 0.5, -0.5))
    ax_p.set_yticks(range(len(env.asset_names)))
    ax_p.set_yticklabels(env.asset_names, fontsize=8)
    ax_p.set_title("position held (daily mean of the hourly position): blue long, red short, white flat", fontsize=10)

    ax_t = fig.add_subplot(gs[3, 0])
    ax_t.axis("off")
    rows = []
    A, g = env.bars_per_year, env.risk_aversion
    tt = tree["times"]
    for label, a_, b_ in SPANS:
        m = np.asarray((tt >= a_) & (tt <= pd.Timestamp(b_) + pd.Timedelta(days=1)))
        if m.sum() < 500:
            continue
        for name, x in series:
            s = metrics(x[m], A, g, tree["pos"][m] if name == "the tree" else None)
            rows.append(f"{label}  {name[:28]:<28} {s['cer']:+7.2f} {s['ret']:+7.2f} {s['vol']:6.1f} {s['sharpe']:+6.2f} "
                        f"{s['dd']:7.1f}" + (f" {s['trades_wk']:7.1f} {s['in_mkt']:5.0f}%" if 'trades_wk' in s
                                             else f" {'':>7} {'':>6}"))
        rows.append("")
    head = (f"{'period':<10} {'':<28} {'CER':>7} {'return':>7} {'vol':>6} {'Sharpe':>6} {'max DD':>7} "
            f"{'trd/wk':>7} {'in mkt':>6}")
    ax_t.text(0.0, 1.0, "\n".join(["THE TREE (one row per instrument, every hour)"] + describe(bank, env.names, 14)),
              family="monospace", fontsize=7.8, va="top", ha="left", transform=ax_t.transAxes)
    ax_t.text(1.0, 1.0, "\n".join([head, ""] + rows[:-1]), family="monospace", fontsize=7.8, va="top", ha="right",
              transform=ax_t.transAxes)
    fig.text(0.99, 0.005, f"drawn {time.strftime('%Y-%m-%d %H:%M')}; % per year over cash; CER at gamma {g:.2f}",
             fontsize=7, ha="right", color="0.4")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    tmp = f"{out_png}.{os.getpid()}.tmp.png"
    fig.savefig(tmp, dpi=100, bbox_inches="tight")
    plt.close(fig)
    for _ in range(40):
        try:
            os.replace(tmp, out_png)
            break
        except PermissionError:
            time.sleep(0.05)
    return rows
