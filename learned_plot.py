"""What the trees DO, drawn: one continuous $100k portfolio from 2006 to the end of the
validation years (2020-2021), next to the S&P 500, buy-all and the volatility target.

portfolio_bt.py redraws it to <results dir>/learned.png every time the search adopts a tree
and at the end of every stage (a copy per stage in <run dir>/learned_stage<k>.png); the
report adds learned_final.png, which runs on through the test years 2022-2026. Panels:

    growth of $100k (log scale)       the trees against the S&P 500 (SPY, dividends in), buy
                                      all (100% invested) and buy all + volatility target
    invested share                    the exposure level chosen at every decision -- the trees'
                                      and the volatility target's -- over the S&P 500's forecast
                                      21-day volatility (spx_sig_21d, what the target reads)
    drawdown                          the fall from the running high: where risk control pays
    the trees, and per period         CAGR, volatility, max drawdown, Calmar (CAGR / |max DD|),
                                      and the CER gap vs the S&P 500 (%/yr, real, the score's gamma)

Years are shaded by their role in the current stage: held-out (what moves are accepted on),
check years (the stage gate), validation (never read by the search) and test. It is ONE path
from one start: it shows what a tree does, not how sure one can be of it (training_budget.png
and the report's bootstrap are for that). Paths run on the numpy reference simulator
(PortfolioWorld.run), which checks/verify_portfolio_kernel.py holds to the compiled one.
"""
import os
import time

import numpy as np

SPANS = [("2006-2019", "2006-01-01", "2019-12-31"), ("2020-2021", "2020-01-01", "2021-12-31"),
         ("2022-2026", "2022-01-01", "2026-12-31")]
ROLE_COLOURS = {"held-out": "#cfe3f5", "check": "#fbe0c3", "validation": "#d5efd0", "test": "#f6d0d0"}
_BASE = {}          # the baselines' paths: they never change within a run, computed once


def path(env, stock_bank, exposure_bank, start, end):
    """One continuous run from the first trading day >= start to the last <= end."""
    a = int(np.searchsorted(env._dates, np.datetime64(start)))
    b = int(np.searchsorted(env._dates, np.datetime64(end), "right"))
    T = b - a - 1                                   # the run reads the open of day a + T
    o = env.run(stock_bank, exposure_bank, np.array([a]), T, full=True)
    dec = a + np.arange(o["exposure"].shape[1]) * env.decide_every
    return dict(dates=env._dates[a + 1:a + 1 + T], daily=o["daily"][0], bench=o["bench"][0],
                infl=o["infl"][0], rf=o["rf"][0], dec=env._dates[dec], f=o["exposure"][0])


def metrics(r, bench, infl, gamma, rf=None):
    """CAGR %, volatility %, max drawdown %, Calmar, CER gap vs the S&P 500 (%/yr, real), and
    beta and alpha (%/yr) against the S&P 500 on excess returns over cash."""
    real = lambda x: (1 + x) / (1 + infl) - 1
    cer = lambda x: 100 * 252 * (x.mean() - 0.5 * gamma * x.var(ddof=1))
    w = np.r_[1.0, np.cumprod(1 + r)]
    cagr = 100 * (w[-1] ** (252 / len(r)) - 1)
    dd = 100 * (w / np.maximum.accumulate(w) - 1).min()
    rf = np.zeros_like(r) if rf is None else rf
    xp, xb = r - rf, bench - rf
    beta = float(np.cov(xp, xb)[0, 1] / np.var(xb, ddof=1))
    return dict(cagr=cagr, vol=100 * np.sqrt(252) * r.std(ddof=1), dd=dd, calmar=cagr / abs(dd) if dd < 0 else np.nan,
                cer_gap=cer(real(r)) - cer(real(bench)), beta=beta, alpha=100 * 252 * (xp.mean() - beta * xb.mean()))


def _describe(bank, names, max_lines=9, width=92):
    """The tree as text (btind's emit), long lines wrapped: the metrics table sits to the right."""
    import textwrap
    from btind.memory import emit
    lines = []
    for ln in emit(bank, names).splitlines():
        lead = len(ln) - len(ln.lstrip(" |\\-"))
        lines += textwrap.wrap(ln, width, subsequent_indent=" " * (lead + 6), break_long_words=True) or [""]
    return lines[:max_lines] + ([f"... ({len(lines) - max_lines} more lines: see the run's report)"]
                                if len(lines) > max_lines else [])


def _roles(env, final):
    """{role: [years]} for the shading."""
    yrs = lambda s: [int(y) for y in s.split(",")] if s else []
    roles = {} if final else {"held-out": yrs(env.holdout)}
    roles["check"] = yrs(env.check)
    roles["validation"] = [2020, 2021]
    if final:
        roles["test"] = list(range(2022, 2027))
    return roles


def draw(env, stock_bank, exposure_bank, out_png, end, baselines, title, final=False):
    """Draw the trees' continuous path from 2006 to `end` with `baselines` {name: (stock
    bank, exposure bank)} to `out_png`; returns the trees' metrics per period."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    start = SPANS[0][1]
    runs = {}
    for name, (sb, eb) in baselines.items():
        key = (id(env), name, str(end))
        if key not in _BASE:
            _BASE[key] = path(env, sb, eb, start, end)
        runs[name] = _BASE[key]
    runs["the trees"] = path(env, stock_bank, exposure_bank, start, end)
    tree = runs["the trees"]
    dates = tree["dates"].astype("datetime64[ns]")
    colours = {"S&P 500": "black", "the trees": "#1f5fbf"}
    other = iter(["#8c8c8c", "#e07b00", "#2ca02c", "#9467bd"])

    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.35, 0.8, 0.8, 0.6], hspace=0.28)
    ax_g, ax_f, ax_d = (fig.add_subplot(gs[i, 0]) for i in range(3))
    roles = _roles(env, final)
    for ax in (ax_g, ax_f, ax_d):
        for role, years in roles.items():
            for y in years:
                ax.axvspan(np.datetime64(f"{y}-01-01"), np.datetime64(f"{y + 1}-01-01"), color=ROLE_COLOURS[role],
                           lw=0, zorder=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, color="0.8", lw=0.6, zorder=1)
        ax.set_xlim(dates[0], dates[-1])

    series = [("S&P 500", tree["bench"])] + [(n, runs[n]["daily"]) for n in baselines] + [("the trees", tree["daily"])]
    for name, r in series:
        c = colours.get(name) or next(other)
        colours[name] = c
        w = env.budget * np.cumprod(1 + r)
        lw = 2.4 if name == "the trees" else 1.3
        ls = "--" if "volatility" in name else "-"
        ax_g.plot(dates, w, color=c, lw=lw, ls=ls, label=f"{name}  (${w[-1] / 1000:,.0f}k)", zorder=3)
        dd = 100 * (w / np.maximum.accumulate(np.r_[env.budget, w])[1:] - 1)      # the start is a high too
        ax_d.plot(dates, dd, color=c, lw=lw, ls=ls, zorder=3)
    ax_g.set_yscale("log")
    ax_g.set_ylabel("value of $100k (log scale)")
    ax_g.legend(loc="upper left", fontsize=9)
    ax_g.set_title(title, fontsize=12)
    ax_d.set_ylabel("drawdown from the running high, %")
    ax_d.axhline(0, color="black", lw=0.8)

    for name, r in [(n, runs[n]) for n in baselines if "volatility" in n] + [("the trees", tree)]:
        ax_f.step(r["dec"].astype("datetime64[ns]"), 100 * r["f"], where="post", color=colours[name],
                  lw=2.2 if name == "the trees" else 1.2, ls="--" if "volatility" in name else "-", label=name, zorder=3)
    ax_f.set_ylim(-5, 108)
    ax_f.set_ylabel("invested, % (exposure level)")
    ax_f.legend(loc="lower left", fontsize=8)
    if "spx_sig_21d" in env.exposure_names:
        tw = ax_f.twinx()
        a = int(np.searchsorted(env._dates, tree["dates"][0]))
        sig = env._M[a:a + len(dates), env.exposure_names.index("spx_sig_21d")]
        tw.plot(dates, sig, color="#b03030", lw=0.7, alpha=0.6)
        tw.set_ylabel("S&P forecast 21-day vol (spx_sig_21d)", color="#b03030", fontsize=8)
        tw.tick_params(axis="y", colors="#b03030", labelsize=8)
    ax_f.set_title("invested share chosen at every decision (step), and the S&P 500's forecast volatility (thin red)",
                   fontsize=10)
    handles = [Patch(color=ROLE_COLOURS[r], label=r + " years") for r in roles]
    ax_d.legend(handles=handles, loc="lower left", fontsize=8, ncol=len(handles))

    # ---- the trees as text, and the metrics per period
    ax_t = fig.add_subplot(gs[3, 0])
    ax_t.axis("off")
    rows = []
    gamma = env.risk_aversion
    for label, a_, b_ in SPANS:
        m = (tree["dates"] >= np.datetime64(a_)) & (tree["dates"] <= np.datetime64(b_))
        if m.sum() < 60:
            continue
        for name, r in series:
            x = metrics(r[m], tree["bench"][m], tree["infl"][m], gamma, tree["rf"][m])
            rows.append(f"{label}  {name[:30]:<30} {x['cagr']:6.1f} {x['vol']:6.1f} {x['dd']:7.1f} "
                        f"{x['calmar']:6.2f} {x['cer_gap']:+7.2f} {x['beta']:5.2f} {x['alpha']:+6.2f}")
        rows.append("")
    head = (f"{'period':<10} {'':<30} {'CAGR':>6} {'vol':>6} {'max DD':>7} {'Calmar':>6} {'CER gap':>7} "
            f"{'beta':>5} {'alpha':>6}")
    text = (["STOCK TREE"] + _describe(stock_bank, env.stock_names, 5) + ["", "EXPOSURE TREE"]
            + _describe(exposure_bank, env.exposure_names, 11))
    ax_t.text(0.0, 1.0, "\n".join(text), family="monospace", fontsize=8.2, va="top", ha="left", transform=ax_t.transAxes)
    ax_t.text(1.0, 1.0, "\n".join([head, ""] + rows[:-1]), family="monospace", fontsize=8.2, va="top", ha="right",
              transform=ax_t.transAxes)
    fig.text(0.99, 0.005, f"drawn {time.strftime('%Y-%m-%d %H:%M')}; % per year; CER gap real, gamma {gamma:.2f}",
             fontsize=7, ha="right", color="0.4")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    tmp = f"{out_png}.{os.getpid()}.tmp.png"
    fig.savefig(tmp, dpi=100, bbox_inches="tight")
    plt.close(fig)
    for _ in range(40):
        try:
            os.replace(tmp, out_png)
            break
        except PermissionError:                          # an image viewer holding it (Windows)
            time.sleep(0.05)
    return rows
