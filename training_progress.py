"""The portfolio against the S&P 500 over training: mean and quartiles, tree by tree.

While portfolio_bt.py trains, every tree btind ADOPTS (a grown arm, a refitted law, a
simplification or pruning, a round's end -- `structure.adopted` on the portfolio-env branch;
a candidate that passed the test but lost to a better variant is not one) is replayed with
its partner tree on the same N_EP
one-year windows drawn once from the training period, next to the S&P 500 on the same
windows (SPY with dividends reinvested, as the trees are scored).

RELATIVE, because the start date dominates anything absolute: every tree loses money in a
window that starts in March 2008 and makes it in one that starts in March 2009, and no
tree could have known which. What a tree controls is how it does AGAINST THE MARKET over
the same year, so each window is measured as

    % ahead of the S&P 500 = 100 (V_portfolio(t) / V_S&P(t) - 1),   both starting at $100k

0 is "matched the market"; +5 is "5% more money than the S&P 500 made". Kept per tree: the
end-of-year value of every window, and day by day the mean and the 25th / 75th percentile
over the windows (the middle half of the start dates: best and worst windows are single
dates, mostly the 2008 crash and the 2009 rebound, and say more about those dates than
about the tree). Best, worst and the 10-90% band are stored too. The first point of every
stage is the pair the stage starts from. The figure is raw money, not risk-adjusted; the
training score (the real certainty-equivalent gap, portfolio_env.py) also charges
variance, so a tree may give up a little of this for a smoother path.

Saved to <run dir>/progress.npz and redrawn to <results dir>/training_budget.png every
time a new tree is adopted and at the end of every stage: each costs about a second, and
adoptions are minutes apart. The windows overlap (200 one-year windows from 13 years),
so they show how the result depends on the start date, not 200 independent years; and they
are SEARCH windows, which the search has fitted. So each tree is also replayed on the
held-out years' windows (what btind accepts moves on) and on one-year windows starting in
2020 (the validation years, never used by the search at all), and the three are drawn side
by side: a tree that pulls ahead on the search windows and not on the others is fitting
history. Windows differ in length (search 6 months, held-out 3, validation 12), so the top
panel shows every result ANNUALISED: 100 ((V / V_S&P)^(252 / days) - 1). The test period
(2022-2026) is never touched here.

Run from the repository root, any time, also while training runs:
    python training_progress.py [--dd W]            draw once and summarise the run of weight W
    python training_progress.py [--dd W] --watch    a live window that redraws, and a console
                                                    line, each time a tree is adopted
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "portfolio_bt")
N_EP, SEED = 200, 4242
KEYS = ("final", "rel", "rel_mean", "rel_p25", "rel_p75", "rel_p10", "rel_p90", "rel_best",
        "rel_worst", "hold_rel", "val_rel", "stage", "agent_is_stocks", "wall")


class Recorder:
    """Replays each adopted tree pair on fixed training windows; appends to progress.npz."""

    def __init__(self, env, run_dir, out_png, n_ep=N_EP, seed=SEED, val_env=None, archive=None):
        self.env, self.out_png, self.val_env, self.archive = env, out_png, val_env, archive
        self.val_starts = val_env._starts[::2] if val_env is not None else None
        self.hold_starts = (np.random.default_rng(seed + 1).choice(env._starts_hold, n_ep).astype(np.int64)
                            if getattr(env, "holdout", "") else None)
        self.path = os.path.join(run_dir, "progress.npz")
        self.starts = env.sample_starts(n_ep, np.random.default_rng(seed))
        self.rows = {k: [] for k in KEYS}
        if os.path.exists(self.path):                        # a resumed run keeps its history
            z = _read(self.path)
            if np.array_equal(z["starts"], self.starts) and all(k in z for k in KEYS):
                self.rows = {k: list(z[k]) for k in KEYS}
        self.stage = 0
        self.spx = None
        # called with (stock bank, exposure bank) after every record: another picture of the
        # adopted pair (portfolio_bt: learned_plot.py); it may never stop training
        self.on_record = None

    def record(self, stock_bank, exposure_bank):
        env = self.env
        out = env.rollout(stock_bank, exposure_bank, self.starts, env.T, daily=True)
        V = env.budget * np.cumprod(1.0 + out["daily"], axis=1)          # (E, T) dollars
        if self.spx is None:
            self.spx = env.budget * np.cumprod(1.0 + out["bench"], axis=1)
        R = 100.0 * (V / self.spx - 1.0)                                  # % ahead of the S&P 500
        end = _annual(R[:, -1], env.T)
        both = {"search": (float(out["cer_gap"].mean()), float(out["dd_gap"].mean()))}
        hold = np.full(1, np.nan)
        if self.hold_starts is not None:
            o = env.rollout(stock_bank, exposure_bank, self.hold_starts, env.T_hold, daily=True)
            hold = _annual(100.0 * (np.prod(1.0 + o["daily"], 1) / np.prod(1.0 + o["bench"], 1) - 1.0), env.T_hold)
            both["held_out"] = (float(o["cer_gap"].mean()), float(o["dd_gap"].mean()))
        if len(getattr(env, "_starts_check", ())):     # the check years: what elite.py chooses on
            o = env.rollout(stock_bank, exposure_bank, env._starts_check.astype(np.int64), env.T_hold)
            both["check"] = (float(o["cer_gap"].mean()), float(o["dd_gap"].mean()))
        val = np.full(1, np.nan)
        if self.val_env is not None:
            ve = self.val_env
            ve.agent, ve.partner = env.agent, env.partner
            o = ve.rollout(stock_bank, exposure_bank, self.val_starts, ve.T, daily=True)
            val = _annual(100.0 * (np.prod(1.0 + o["daily"], 1) / np.prod(1.0 + o["bench"], 1) - 1.0), ve.T)
            both["validation"] = (float(o["cer_gap"].mean()), float(o["dd_gap"].mean()))
        for k, v in (("final", V[:, -1]), ("rel", end), ("rel_mean", R.mean(0)),
                     ("rel_p25", np.quantile(R, 0.25, axis=0)), ("rel_p75", np.quantile(R, 0.75, axis=0)),
                     ("rel_p10", np.quantile(R, 0.1, axis=0)), ("rel_p90", np.quantile(R, 0.9, axis=0)),
                     ("rel_best", R[np.argmax(end)]), ("rel_worst", R[np.argmin(end)]), ("hold_rel", hold),
                     ("val_rel", val),
                     ("stage", self.stage), ("agent_is_stocks", env.agent == "stocks"), ("wall", time.time())):
            self.rows[k].append(v)
        np.savez(self.path + ".tmp.npz", starts=self.starts, spx=self.spx, dates=env._dates[self.starts],
                 **{k: np.asarray(v) for k, v in self.rows.items()})
        for _ in range(40):              # a viewer reading the file blocks the replace on Windows
            try:
                os.replace(self.path + ".tmp.npz", self.path)
                break
            except PermissionError:
                time.sleep(0.05)         # still blocked: the next record rewrites everything
        if self.archive:                 # every adopted pair with both scores, for elite.py
            import json
            from btind.runlog import bank_json
            with open(self.archive, "a") as fh:
                fh.write(json.dumps(dict(
                    dd_weight=env.dd_weight, weighting=env.weighting, stage=int(self.stage), agent=env.agent,
                    wall=time.time(),
                    **{f"{k}_{m}": v[i] for k, v in both.items() for i, m in enumerate(("gap", "dd"))},
                    stock_bank=bank_json(stock_bank, env.stock_names),
                    exposure_bank=bank_json(exposure_bank, env.exposure_names))) + "\n")
        self.draw()
        if self.on_record is not None:
            try:
                self.on_record(stock_bank, exposure_bank)
            except Exception as e:                           # a picture must never stop training
                print("  (learned-path picture skipped:", e, ")", flush=True)

    def draw(self):
        try:
            draw(self.path, self.out_png)
        except Exception as e:                               # a plot must never stop training
            print("  (training plot skipped:", e, ")", flush=True)


def _annual(pct, days):
    """% ahead of the S&P over a window of `days`, as % per year."""
    return 100.0 * ((1.0 + np.asarray(pct) / 100.0) ** (252.0 / days) - 1.0)


def _read(path):
    """The arrays of an .npz, file closed at once (an open one blocks the atomic replace on Windows)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def draw(path, out_png):
    """Render the progress file to a PNG, off screen (what training calls)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    K = render(fig, _read(path))
    _save(fig, out_png)
    plt.close(fig)
    return K


def _save(fig, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    tmp = f"{out_png}.{os.getpid()}.tmp.png"                 # training and a viewer may both draw
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    for _ in range(40):
        try:
            os.replace(tmp, out_png)
            return
        except PermissionError:                              # an image viewer holding it (Windows)
            time.sleep(0.05)
    os.remove(tmp)


REF_LINES = (5, 10)          # % lines drawn and labelled on both panels (above and below 0)


def _grid(ax, y_lo, y_hi):
    """Major and minor grid on both axes, and labelled +-REF_LINES reference lines."""
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", color="0.75", lw=0.7, zorder=0)
    ax.grid(True, which="minor", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for r in REF_LINES:
        for v in (r, -r):
            if y_lo < v < y_hi:
                ax.axhline(v, color="0.35", lw=0.9, ls=(0, (6, 3)), zorder=1)
                ax.text(1.002, v, f"{v:+d}%", transform=ax.get_yaxis_transform(), fontsize=8,
                        va="center", ha="left", color="0.25")


def render(fig, z):
    """Draw the progress figure into `fig`, whatever its backend (a PNG or a live window).
    Every adopted tree has one colour (dark = early, light = late) in both panels."""
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    rel, stage = z["rel"], z["stage"]
    K, E = rel.shape
    pct = FuncFormatter(lambda v, _: f"{v:+.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}%")
    norm = Normalize(0, max(K - 1, 1))
    palette = ListedColormap(cm.viridis(np.linspace(0, 0.9, 256)))   # stop before viridis' pale yellow
    cmap = lambda k: palette(norm(k))
    # wspace leaves room for the +-5% / +-10% labels between the panels and the colour bar
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1, 0.015], hspace=0.3, wspace=0.07)

    # ---- top: the end-of-year result of every tree, one colour each
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(K)
    q25, q75 = np.quantile(rel, 0.25, axis=1), np.quantile(rel, 0.75, axis=1)
    # each tree's three markers straddle ITS number: search years just left, held-out years
    # on it, validation just right -- never reaching the next tree or a stage divider
    XS, XH, XV = -0.25, 0.0, 0.25
    ax.plot(x + XS, rel.mean(1), color="grey", lw=1.0, zorder=1)
    for k in range(K):
        c = cmap(k)
        ax.vlines(k + XS, q25[k], q75[k], color=c, lw=7, alpha=0.55, zorder=2)
        ax.plot(k + XS, rel[k].mean(), "o", color=c, ms=8, mec="black", mew=0.7, zorder=3)
    vr = z["val_rel"] if "val_rel" in z else None
    has_val = vr is not None and np.isfinite(vr).any()
    hr = z["hold_rel"] if "hold_rel" in z else None
    has_hold = hr is not None and np.isfinite(hr).any()
    if has_hold:
        h25, h75 = np.nanquantile(hr, 0.25, axis=1), np.nanquantile(hr, 0.75, axis=1)
        ax.plot(x + XH, np.nanmean(hr, 1), color="grey", lw=1.0, ls="--", zorder=1)
        for k in range(K):
            ax.vlines(k + XH, h25[k], h75[k], color=cmap(k), lw=2.5, alpha=0.9, zorder=2)
            ax.plot(k + XH, np.nanmean(hr[k]), "s", color=cmap(k), ms=7, mec="black", mew=0.7, zorder=3)
    if has_val:
        v25, v75 = np.nanquantile(vr, 0.25, axis=1), np.nanquantile(vr, 0.75, axis=1)
        ax.plot(x + XV, np.nanmean(vr, 1), color="grey", lw=1.0, ls=":", zorder=1)
        for k in range(K):
            ax.vlines(k + XV, v25[k], v75[k], color=cmap(k), lw=2.5, alpha=0.9, zorder=2)
            ax.plot(k + XV, np.nanmean(vr[k]), "D", color="white", ms=7, mec=cmap(k), mew=2.0, zorder=3)
    ax.axhline(0, color="black", lw=1.4)
    x_right = max(K - 0.5, 1.5)
    htxt = (f"\nheld-out years: mean {np.nanmean(hr[-1]):+.1f}%, median {np.nanmedian(hr[-1]):+.1f}%, "
            f"ahead in {100 * np.nanmean(hr[-1] > 0):.0f}%" if has_hold else "")
    vtxt = (f"\nvalidation 2020-21: mean {np.nanmean(vr[-1]):+.1f}%, median {np.nanmedian(vr[-1]):+.1f}%, "
            f"ahead in {100 * np.nanmean(vr[-1] > 0):.0f}%" if has_val else "")
    ax.text(0.995, 0.97, f"latest tree #{K - 1}, search windows: mean {rel[-1].mean():+.1f}%, median "
                         f"{np.median(rel[-1]):+.1f}%, ahead in {100 * np.mean(rel[-1] > 0):.0f}%" + htxt + vtxt,
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec=cmap(K - 1), lw=1.5))
    for s in np.unique(stage):
        k0 = np.argmax(stage == s)
        if k0 > 0:
            ax.axvline(k0 - 0.5, color="grey", lw=0.8, ls="--")
        agent = "stocks" if z["agent_is_stocks"][k0] else "exposure"
        ax.text(k0 - 0.4, 1.01, f"stage {s}: {agent}", transform=ax.get_xaxis_transform(), fontsize=8.5,
                va="bottom")
    ax.set_xlim(-0.5, x_right)
    y0, y1 = min(q25.min(), rel.mean(1).min(), 0.0), max(q75.max(), rel.mean(1).max(), 0.0)
    if has_hold:
        y0, y1 = min(y0, np.nanmin(h25)), max(y1, np.nanmax(h75))
    if has_val:
        y0, y1 = min(y0, np.nanmin(v25)), max(y1, np.nanmax(v75))
    ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.30 * (y1 - y0))      # headroom for the legend
    # a grid line at EVERY tree, labels as many as fit
    from matplotlib.ticker import MultipleLocator
    # every tree labelled while they fit (30), then as many as fit, a grid line at each
    ax.xaxis.set_major_locator(MultipleLocator(1) if K <= 30 else MaxNLocator(integer=True, nbins=30))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(pct)
    _grid(ax, *ax.get_ylim())
    ax.grid(True, which="minor", axis="x", color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("tree # (every tree the search adopted, in order)")
    ax.set_ylabel("ahead of the S&P 500,\n% per year (0 = matched it)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="grey", mec="black", ms=8,
                              label="search years (fitted on): mean, bar 25th-75th"),
                       Line2D([], [], marker="s", ls="", color="grey", mec="black", ms=7,
                              label="held-out years (accepted on): mean"),
                       Line2D([], [], marker="D", ls="", color="white", mec="grey", mew=2, ms=7,
                              label="validation 2020-21 (never used): mean")],
              loc="upper left", fontsize=8, ncol=1)
    years = z["dates"].astype("datetime64[Y]").astype(int) + 1970
    ax.set_title(f"Portfolio vs the S&P 500 (SPY, dividends reinvested), both from \\$100k: {E} search windows "
                 f"({years.min()}-{years.max()} starts), held-out years and validation 2020-21, annualised",
                 fontsize=11, pad=18)

    # ---- bottom: every tree's day-by-day path in its colour, the latest drawn last and thickest
    b = fig.add_subplot(gs[1, 0])
    d = np.arange(z["rel_mean"].shape[1])
    for k in range(K):
        c, last = cmap(k), k == K - 1
        alpha = 1.0 if last else 0.3 + 0.5 * k / max(K - 1, 1)
        b.plot(d, z["rel_mean"][k], color=c, lw=3.0 if last else 1.3, alpha=alpha, zorder=2 + last)
        for q in ("rel_p25", "rel_p75"):
            if q in z:                   # a run recorded before the quartiles were kept has none
                b.plot(d, z[q][k], color=c, lw=1.6 if last else 0.8, ls="--", alpha=alpha, zorder=2 + last)
    b.axhline(0, color="black", lw=1.2)
    b.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    b.yaxis.set_major_formatter(pct)
    # time: a major line every 21 trading days (a month), labelled in months, a minor one weekly
    from matplotlib.ticker import FuncFormatter as _Fmt, MultipleLocator
    b.set_xlim(0, max(len(d) - 1, 1))
    b.xaxis.set_major_locator(MultipleLocator(21))
    b.xaxis.set_minor_locator(MultipleLocator(5))
    b.xaxis.set_major_formatter(_Fmt(lambda v, _: f"{int(v)}\n({v / 21:.0f} mo)" if v > 0 else "0"))
    _grid(b, *b.get_ylim())
    b.grid(True, which="minor", axis="x", color="0.9", lw=0.5, zorder=0)
    b.set_xlabel("trading day of the search window (21 trading days = 1 month)")
    b.set_ylabel("ahead of the S&P 500")
    b.set_title(f"Day by day, every tree: mean over the windows (solid), 25th and 75th percentile "
                f"(dashed{'' if 'rel_p25' in z else ': not recorded by this run'}); latest tree #{K - 1} "
                f"thickest", fontsize=10)
    b.legend(handles=[Line2D([], [], color="black", lw=2, label="mean"),
                      Line2D([], [], color="black", lw=1, ls="--", label="25th / 75th percentile")],
             loc="upper left", fontsize=8)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=palette),
                      cax=fig.add_subplot(gs[:, 1]))
    cb.set_label("tree #")
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()
    return K


def _line(z, k):
    r = z["rel"][k]
    agent = "stocks  " if z["agent_is_stocks"][k] else "exposure"
    q25, q50, q75 = np.quantile(r, [0.25, 0.5, 0.75])
    part = lambda key, lab: (f"  | {lab}: mean {np.nanmean(z[key][k]):+6.1f}%  median {np.nanmedian(z[key][k]):+6.1f}%"
                             if key in z and np.isfinite(z[key][k]).any() else "")
    return (f"  tree #{k:<4} stage {z['stage'][k]} {agent}  %/yr vs S&P 500, search: mean {r.mean():+6.1f}%   "
            f"p25 {q25:+6.1f}%   median {q50:+6.1f}%   p75 {q75:+6.1f}%   ahead in {100 * np.mean(r > 0):3.0f}%"
            + part("hold_rel", "held-out") + part("val_rel", "validation 2020-21"))


def _summary(path):
    z = _read(path)
    print(f"{len(z['rel'])} trees on {z['rel'].shape[1]} fixed training windows")
    for k in sorted({0, len(z["rel"]) // 2, len(z["rel"]) - 1}):
        print(_line(z, k))


def watch(path, png):
    """A live window that redraws in place whenever training accepts a new tree (and writes the
    PNG and a console line for each); closing the window ends it."""
    import matplotlib.pyplot as plt                          # the interactive backend (TkAgg)
    plt.ion()
    fig = plt.figure("BT training vs the S&P 500", figsize=(14, 9))
    fig.text(0.5, 0.5, "waiting for the first tree ...", ha="center", fontsize=12, color="dimgrey")
    shown, stamp = 0, None
    print("watching", path, "- the window redraws when a tree is accepted; close it or Ctrl-C to stop",
          flush=True)
    while plt.fignum_exists(fig.number):                     # react to a new record, not to the clock
        try:
            m = os.path.getmtime(path) if os.path.exists(path) else None
            if m is not None and m != stamp:
                z = _read(path)
                if len(z["rel"]) < shown:                    # a new run started: show it from its beginning
                    shown = 0
                for k in range(shown, len(z["rel"])):
                    print(time.strftime("%H:%M:%S"), _line(z, k), flush=True)
                shown, stamp = len(z["rel"]), m
                fig.clf()
                render(fig, z)
                _save(fig, png)
                fig.canvas.draw_idle()
        except (OSError, KeyError, ValueError):
            pass                                             # caught mid-write: the next poll reads it whole
        plt.pause(2.0)                                       # keeps the window responsive between polls


if __name__ == "__main__":
    dd = float(sys.argv[sys.argv.index("--dd") + 1]) if "--dd" in sys.argv else 0.0
    OUT = os.path.join(OUT, f"dd{dd:g}")          # the run of that drawdown weight (portfolio_bt.py --dd)
    path = os.path.join(OUT, "run", "progress.npz")
    png = os.path.join(OUT, "training_budget.png")
    if not os.path.exists(path) and "--watch" not in sys.argv:
        sys.exit(f"no progress yet: {path} (it appears when training starts)")
    if "--watch" not in sys.argv:
        draw(path, png)
        _summary(path)
        print("wrote", png)
        sys.exit()
    watch(path, png)
