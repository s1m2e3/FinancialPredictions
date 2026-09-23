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
are TRAINING windows, which the search has seen (the report's validation and test periods
are the out-of-sample check).

Run from the repository root, any time, also while training runs:
    python training_progress.py              draw once and summarise
    python training_progress.py --watch      a live window that redraws, and a console line,
                                             each time a tree is adopted; close it to stop
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "portfolio_bt")
N_EP, SEED = 200, 4242
KEYS = ("final", "rel", "rel_mean", "rel_p25", "rel_p75", "rel_p10", "rel_p90", "rel_best",
        "rel_worst", "stage", "agent_is_stocks", "wall")


class Recorder:
    """Replays each adopted tree pair on fixed training windows; appends to progress.npz."""

    def __init__(self, env, run_dir, out_png, n_ep=N_EP, seed=SEED):
        self.env, self.out_png = env, out_png
        self.path = os.path.join(run_dir, "progress.npz")
        self.starts = env.sample_starts(n_ep, np.random.default_rng(seed))
        self.rows = {k: [] for k in KEYS}
        if os.path.exists(self.path):                        # a resumed run keeps its history
            z = _read(self.path)
            if np.array_equal(z["starts"], self.starts) and all(k in z for k in KEYS):
                self.rows = {k: list(z[k]) for k in KEYS}
        self.stage = 0
        self.spx = None

    def record(self, stock_bank, exposure_bank):
        env = self.env
        out = env.rollout(stock_bank, exposure_bank, self.starts, env.T, daily=True)
        V = env.budget * np.cumprod(1.0 + out["daily"], axis=1)          # (E, T) dollars
        if self.spx is None:
            self.spx = env.budget * np.cumprod(1.0 + out["bench"], axis=1)
        R = 100.0 * (V / self.spx - 1.0)                                  # % ahead of the S&P 500
        end = R[:, -1]
        for k, v in (("final", V[:, -1]), ("rel", end), ("rel_mean", R.mean(0)),
                     ("rel_p25", np.quantile(R, 0.25, axis=0)), ("rel_p75", np.quantile(R, 0.75, axis=0)),
                     ("rel_p10", np.quantile(R, 0.1, axis=0)), ("rel_p90", np.quantile(R, 0.9, axis=0)),
                     ("rel_best", R[np.argmax(end)]), ("rel_worst", R[np.argmin(end)]),
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
        self.draw()

    def draw(self):
        try:
            draw(self.path, self.out_png)
        except Exception as e:                               # a plot must never stop training
            print("  (training plot skipped:", e, ")", flush=True)


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
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1, 0.015], hspace=0.3, wspace=0.03)

    # ---- top: the end-of-year result of every tree, one colour each
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(K)
    q25, q75 = np.quantile(rel, 0.25, axis=1), np.quantile(rel, 0.75, axis=1)
    ax.plot(x, rel.mean(1), color="grey", lw=1.0, zorder=1)
    for k in range(K):
        c = cmap(k)
        ax.vlines(k, q25[k], q75[k], color=c, lw=9, alpha=0.55, zorder=2)
        ax.plot(k, rel[k].mean(), "o", color=c, ms=8, mec="black", mew=0.7, zorder=3)
    ax.axhline(0, color="black", lw=1.4)
    x_right = max(K - 0.5, 5.5)
    ax.text(0.995, 0.97, f"latest tree #{K - 1}: mean {rel[-1].mean():+.1f}%, median {np.median(rel[-1]):+.1f}%, "
                         f"ahead of the S&P 500 in {100 * np.mean(rel[-1] > 0):.0f}% of windows",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec=cmap(K - 1), lw=1.5))
    for s in np.unique(stage):
        k0 = np.argmax(stage == s)
        if k0 > 0:
            ax.axvline(k0 - 0.5, color="grey", lw=0.8, ls="--")
        agent = "stock tree" if z["agent_is_stocks"][k0] else "exposure tree"
        ax.text(k0, 1.01, f" stage {s}: {agent}", transform=ax.get_xaxis_transform(), fontsize=9, va="bottom")
    ax.set_xlim(-0.5, x_right)
    y0, y1 = min(q25.min(), rel.mean(1).min(), 0.0), max(q75.max(), rel.mean(1).max(), 0.0)
    ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.30 * (y1 - y0))      # headroom for the legend
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("tree # (every tree the search adopted, in order)")
    ax.set_ylabel("ahead of the S&P 500 after\none year (0 = matched it)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="grey", mec="black", ms=8, label="mean"),
                       Line2D([], [], color="grey", lw=9, alpha=0.55, label="25th to 75th percentile of windows")],
              loc="upper left", fontsize=8, ncol=2)
    years = z["dates"].astype("datetime64[Y]").astype(int) + 1970
    ax.set_title(f"Portfolio vs the S&P 500 (SPY, dividends reinvested) on the same {E} one-year training "
                 f"windows ({years.min()}-{years.max()} starts), both from \\$100k", fontsize=11, pad=18)

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
    b.set_xlabel("trading day of the one-year window")
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
    return (f"  tree #{k:<4} stage {z['stage'][k]} {agent}  vs S&P 500: mean {r.mean():+6.1f}%   "
            f"p25 {q25:+6.1f}%   median {q50:+6.1f}%   p75 {q75:+6.1f}%   ahead in {100 * np.mean(r > 0):3.0f}%")


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
