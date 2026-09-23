"""Simulated Nasdaq price paths from the walk-forward models, against the real index.

The models are the ones scored by run_armagarch.py: every 63 trading days of the test
window each arm is refitted on everything before the block (fits are deterministic, so
refitting here reproduces the evaluated parameters; they are cached to
walk_forward_params.npz next to the run's results). From an origin t, paths are simulated
21 trading days ahead with information up to day t-1 and turned into index levels.

  sim_paths_test.png   a fresh one-month forecast every 21 trading days across 2023-2025
  sim_paths_zoom.png   four origins in detail: calm, before and after large sell-offs

Run from the repository root:  python plot_paths.py [--criterion AICc|BIC]
"""
import argparse
import json
import os
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter          # noqa: E402

from armagarch import ARMS, Fit, arm_spec, difference, fit_arms, simulate  # noqa: E402
from data import build_dataset, load_prices                         # noqa: E402
from run_armagarch import (BLOCK, COLOR, INK, INK_2, LABEL, ORDER_SEARCH, ROOT,  # noqa: E402
                           SPEC_FILE, spec_from_label)

warnings.filterwarnings("ignore")

SHOW = ("sarimax", "full", "jump")   # plain SARIMAX, SARIMAX + GARCH-M skewed-t, SARIMAX + GARCH-jump
H = 21                          # one month of trading days
N_SIM = 2000                    # paths per origin used for the interval statistics
N_DRAW = 30                     # paths drawn per origin in the test-window figure
N_DRAW_ZOOM = 100               # paths drawn per origin in the zoomed figure
PATH_ALPHA = 0.4
REAL_WIDTH = 2.6
ZOOM = {"2023-06-01": "calm bull market",
        "2024-07-15": "before the July-August 2024 sell-off",
        "2025-04-01": "just before the April 2025 tariff crash",
        "2025-04-10": "after the crash and the 9 April rebound"}


def walk_forward_fits(base, w, X, n_tr, n_va, n, cache):
    """Per-block fits of every arm, exactly as in run_armagarch.walk_forward."""
    starts = np.arange(n_va, n, BLOCK)
    if os.path.exists(cache):
        z = np.load(cache)
        if (np.array_equal(z["starts"], starts) and str(z["label"]) == base.label()
                and all(a in z.files for a in ARMS)):
            specs = {a: arm_spec(a, base) for a in ARMS}
            return starts, [{a: Fit(spec=specs[a], u=z[a][b], t0=specs[a].t0, t_end=s) for a in ARMS}
                            for b, s in enumerate(starts)]
    t = time.time()
    warm = fit_arms(base, w, X, n_tr, val_end=n_va)
    blocks = []
    for s in starts:
        warm = fit_arms(base, w, X, t_end=s, warm=warm)
        blocks.append(warm)
        print(f"  refit for block starting {s}: {time.time() - t:.0f}s elapsed")
    np.savez(cache, starts=starts, label=base.label(), **{a: np.stack([b[a].u for b in blocks]) for a in ARMS})
    return starts, blocks


def simulate_origins(origins, starts, blocks, w, y, X, stochastic, n_paths, seed):
    """Cumulative log-return paths (n_origins, n_paths, H) for each shown arm."""
    out = {a: np.empty((len(origins), n_paths, H)) for a in SHOW}
    block_of = np.searchsorted(starts, origins, side="right") - 1
    for b in np.unique(block_of):
        rows = np.where(block_of == b)[0]
        for a in SHOW:
            f = blocks[b][a]
            out[a][rows] = simulate(f, f.filter(w, X), y, X, stochastic, origins[rows],
                                    list(range(1, H + 1)), n_paths, seed + int(b))
    return out


def ordinal(k):
    k = int(round(k))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def price_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())


def price_ticks(ax, n=6):
    """Round index levels as ticks on a log axis (call once the limits are final)."""
    lo, hi = ax.get_ylim()
    raw = (hi - lo) / n
    step = min((m * 10 ** np.floor(np.log10(raw)) for m in (1, 2, 2.5, 5, 10)), key=lambda v: abs(v - raw))
    ax.set_yticks(np.arange(np.ceil(lo / step) * step, hi, step))
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--criterion", choices=["AICc", "BIC"], default="AICc")
    args = parser.parse_args()
    out_dir = os.path.join(ROOT, "results", f"sarimax_garch_{args.criterion.lower()}")
    os.makedirs(out_dir, exist_ok=True)

    with open(SPEC_FILE) as fh:
        ident = json.load(fh)
    d = build_dataset(load_prices())
    y, dates, n_tr, n_va, n = d.y, d.dates, d.n_train, d.n_val, d.n
    X = d.X(ident["regressors"])
    stochastic = np.array(ident["stochastic"], bool)
    grid = pd.read_csv(ORDER_SEARCH, index_col=0)
    base = spec_from_label(grid[args.criterion].idxmin(), ident["d"], len(ident["regressors"]))
    w = difference(y, base)
    price = np.exp(y / 100)
    print(f"{base.label()}: walk-forward fits")
    starts, blocks = walk_forward_fits(base, w, X, n_tr, n_va, n,
                                       os.path.join(out_dir, "walk_forward_params.npz"))

    # ---------------------------------------------------------------- whole test window
    origins = np.arange(n_va, n - H + 1, H)
    sims = simulate_origins(origins, starts, blocks, w, y, X, stochastic, N_SIM, seed=7)
    fig, axes = plt.subplots(len(SHOW), 1, figsize=(15, 5 * len(SHOW)), sharex=True, sharey=True)
    lo_hi = {}
    for ax, a in zip(axes, SHOW):
        paths = price[origins - 1][:, None, None] * np.exp(sims[a] / 100)      # (n_o, N_SIM, H)
        end_real = price[origins + H - 1]
        q05, q95 = np.quantile(paths[:, :, -1], [0.05, 0.95], axis=1)
        inside = (end_real >= q05) & (end_real <= q95)
        width = np.mean((q95 - q05) / price[origins - 1]) * 100
        lo_hi[a] = (inside.mean(), width)
        for i, t in enumerate(origins):
            x = dates[t - 1:t + H]
            for p in paths[i, :N_DRAW]:
                ax.plot(x, np.r_[price[t - 1], p], color=COLOR[a], alpha=PATH_ALPHA, lw=0.6)
        ax.plot(dates[n_va - 20:], price[n_va - 20:], color=INK, lw=REAL_WIDTH, zorder=5,
                label="Nasdaq Composite (real)")
        ax.plot([], [], color=COLOR[a], alpha=PATH_ALPHA, lw=1.5, label=f"simulated paths, {N_DRAW} per origin")
        ax.scatter(dates[origins - 1], price[origins - 1], s=14, color=INK, zorder=6, label="forecast origin")
        price_axis(ax)
        ax.set_ylabel("Nasdaq Composite (log scale)")
        ax.set_title(f"{LABEL[a]}: real 1-month-ahead level inside the 90% simulated range at "
                     f"{inside.sum()} of {len(origins)} origins; average range width {width:.1f}% of the index")
        ax.legend(loc="upper left")
    for ax in axes:
        price_ticks(ax)
    fig.suptitle(f"One-month simulated paths from every 21st test day vs the real Nasdaq, {base.label()} "
                 "(walk-forward, out of sample)", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sim_paths_test.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- zoomed origins
    zoom_t = np.array([int(np.searchsorted(dates, pd.Timestamp(s))) for s in ZOOM])
    zsims = simulate_origins(zoom_t, starts, blocks, w, y, X, stochastic, N_SIM, seed=11)
    fig, axes = plt.subplots(len(zoom_t), len(SHOW), figsize=(7.5 * len(SHOW), 4.2 * len(zoom_t)), sharey="row")
    for r, (t, why) in enumerate(zip(zoom_t, ZOOM.values())):
        window = slice(t - 30, t + H)
        for c, a in enumerate(SHOW):
            ax = axes[r, c]
            paths = price[t - 1] * np.exp(zsims[a][r] / 100)
            x = dates[t - 1:t + H]
            for p in paths[:N_DRAW_ZOOM]:
                ax.plot(x, np.r_[price[t - 1], p], color=COLOR[a], alpha=PATH_ALPHA, lw=0.7)
            ax.plot(dates[window], price[window], color=INK, lw=REAL_WIDTH, zorder=5)
            ax.axvline(dates[t - 1], color=INK_2, lw=1, ls="--")
            ret_real = 100 * (price[t + H - 1] / price[t - 1] - 1)
            pct = 100 * np.mean(paths[:, -1] <= price[t + H - 1])
            q05, q95 = 100 * (np.quantile(paths[:, -1], [0.05, 0.95]) / price[t - 1] - 1)
            ax.set_title(f"real 1-month return {ret_real:+.1f}%  |  90% range {q05:+.1f}% to {q95:+.1f}%  |  "
                         f"real outcome at the {ordinal(pct)} percentile", fontsize=9, fontweight="normal")
            if r == 0:
                ax.text(0, 1.16, LABEL[a], transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
            if c == 0:
                ax.set_ylabel(f"origin {dates[t - 1].date()}\n{why}", fontsize=10, color=INK)
            price_axis(ax)
            ax.tick_params(axis="x", labelrotation=30)
    for ax in axes.flat:
        price_ticks(ax, n=5)
    fig.suptitle(f"Simulated paths ({N_DRAW_ZOOM} per origin, alpha {PATH_ALPHA}) vs the real Nasdaq (thick black), "
                 f"{base.label()}", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5)
    fig.savefig(os.path.join(out_dir, "sim_paths_zoom.png"), dpi=150)
    plt.close(fig)

    for a in SHOW:
        print(f"{LABEL[a]}: 1-month level inside 90% range {lo_hi[a][0]:.0%} of origins, "
              f"mean range width {lo_hi[a][1]:.1f}%")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
