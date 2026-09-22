"""SARIMAX vs SARIMAX + GARCH, one day to one month ahead, on the Nasdaq Composite.

The SARIMAX mean equation comes from identification.py (Box-Jenkins on the training
window, saved to results/identification/spec.json). Six arms share it; each adds one
thing to the one before:

  sarimax      constant variance, Gaussian              (the classic SARIMAX)
  sarimax_t    constant variance, Student-t             (fat tails)
  two_step     SARIMAX mean frozen, GJR-GARCH-t on its residuals
  joint        SARIMAX + GJR-GARCH-t estimated together (coupling)
  joint_skewt  + skewed-t innovations
  full         + GARCH-in-mean

Protocol (chronological, nothing shuffled):
  1. Fit all arms on train (2013-2020); loss curves record the training NLL and, with
     the same parameters, the validation NLL (2021-2022) at every iteration.
  2. Test (2023-2025), expanding-window walk-forward: every BLOCK trading days every arm is
     refitted on everything before the block. Within the block, parameters are frozen:
       - one-day-ahead densities are exact and scored by log-likelihood;
       - from every origin, N_PATHS paths are simulated 21 days ahead and the cumulative
         log return is scored at 1 day, 3 days, 1 week, 2 weeks and 1 month.

Run from the repository root (after identification.py):
    python run_armagarch.py                    # SARIMA order chosen by AICc (spec.json)
    python run_armagarch.py --criterion BIC    # the BIC choice from the same order search
Outputs go to results/sarimax_garch_<criterion>/, including scores.npz with the per-origin
scores that compare_specs.py uses to test the two orders against each other.
"""
import argparse
import json
import os
import re
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from statsmodels.tsa.stattools import acf                          # noqa: E402

import evaluation as ev                                             # noqa: E402
from armagarch import ARMS, Spec, difference, fit_arms, walk_forward             # noqa: E402
from data import TRAIN_END, VAL_END, build_dataset, load_prices     # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC_FILE = os.path.join(ROOT, "results", "identification", "spec.json")
ORDER_SEARCH = os.path.join(ROOT, "results", "identification", "order_search.csv")
OUT = None                     # results/sarimax_garch_<criterion>, set in main()
BLOCK = 63                     # refit once a quarter in the walk-forward test
N_PATHS = 5000                 # simulated paths per origin
HORIZONS = {1: "1 day", 3: "3 days", 5: "1 week", 10: "2 weeks", 21: "1 month"}

LABEL = {
    "sarimax": "SARIMAX (constant variance, normal)",
    "sarimax_t": "SARIMAX-t (constant variance, Student-t)",
    "two_step": "SARIMAX then GARCH-t (two-step)",
    "joint": "SARIMAX + GARCH-t (joint)",
    "joint_skewt": "SARIMAX + GARCH, skewed-t (joint)",
    "full": "SARIMAX + GARCH-in-mean, skewed-t (joint)",
}
SHORT = {"sarimax": "SARIMAX", "sarimax_t": "SARIMAX-t", "two_step": "Two-step",
         "joint": "Joint", "joint_skewt": "Joint skew-t", "full": "Full (GARCH-M)"}
STEP = {"sarimax_t": "fat tails", "two_step": "GARCH variance", "joint": "joint estimation",
        "joint_skewt": "skewed errors", "full": "GARCH-in-mean"}
LADDER = {"sarimax_t": "sarimax", "two_step": "sarimax_t", "joint": "two_step",
          "joint_skewt": "joint", "full": "joint_skewt"}     # the rung each arm is compared with

# Categorical slots in fixed order; the full model takes slot 1.
COLOR = {"full": "#2a78d6", "joint_skewt": "#eb6834", "joint": "#1baf7a", "two_step": "#eda100",
         "sarimax_t": "#e87ba4", "sarimax": "#008300"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 9,
})


def md_table(df, fmt="{:.4f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def param_table(fits, regressors):
    rows = {}
    for a in ARMS:
        P = fits[a].params
        row = {"c": P["c"]}
        row.update({f"beta_{n}": b for n, b in zip(regressors, P["beta"])})
        row.update({f"phi{i + 1}": v for i, v in enumerate(P["phi"])})
        row.update({f"Phi{i + 1}": v for i, v in enumerate(P["Phi"])})
        row.update({f"theta{i + 1}": v for i, v in enumerate(P["theta"])})
        row.update({f"Theta{i + 1}": v for i, v in enumerate(P["Theta"])})
        row.update({"delta (in-mean)": P["delta"], "sigma2 (uncond.)": P["sbar2"], "omega": P["omega"],
                    "alpha": P["alpha"], "gamma": P["gamma"], "b": P["b"],
                    "persistence": P["persistence"], "nu": P["nu"], "lambda (skew)": P["lam"]})
        rows[SHORT[a]] = row
    df = pd.DataFrame(rows)
    df.index.name = "parameter (train fit)"
    return df


def spec_from_label(label, d, k):
    """Spec for a label such as 'SARIMA(2,1,2)(0,0,1)[5]' or 'SARIMA(3,1,3)'."""
    m = re.fullmatch(r"SARIMA\((\d+),(\d+),(\d+)\)(?:\((\d+),(\d+),(\d+)\)\[(\d+)\])?", label)
    p, dd, q, P, D, Q, s = (int(g) if g is not None else None for g in m.groups())
    assert dd == d, "differencing order must match identification"
    return Spec(p=p, d=d, q=q, P=P or 0, D=D or 0, Q=Q or 0, s=s or 1, k=k)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--criterion", choices=["AICc", "BIC"], default="AICc",
                        help="which information criterion's order to use (default: AICc, as auto_arima)")
    args = parser.parse_args()
    OUT = os.path.join(ROOT, "results", f"sarimax_garch_{args.criterion.lower()}")
    os.makedirs(OUT, exist_ok=True)

    with open(SPEC_FILE) as fh:
        ident = json.load(fh)
    d = build_dataset(load_prices())
    y, dates, n_tr, n_va, n = d.y, d.dates, d.n_train, d.n_val, d.n
    regressors = ident["regressors"]
    X = d.X(regressors)
    stochastic = np.array(ident["stochastic"], bool)
    grid = pd.read_csv(ORDER_SEARCH, index_col=0)
    base = spec_from_label(grid[args.criterion].idxmin(), ident["d"], len(regressors))
    assert args.criterion != "AICc" or base.label() == ident["label"], "spec.json and order search disagree"
    w = difference(y, base)
    title = base.label() + (f" + {len(regressors)} regressors" if regressors else "")
    print(f"mean equation {title}; train {n_tr}, val {n_va - n_tr}, test {n - n_va} days")

    # 1. train-window fits with validation monitoring -----------------------------------
    t = time.time()
    fits = fit_arms(base, w, X, n_tr, val_end=n_va)
    print(f"train-window fits: {time.time() - t:.0f}s")
    filt = {a: fits[a].filter(w, X) for a in ARMS}
    val = pd.DataFrame({SHORT[a]: ev.one_step_metrics(w[n_tr:n_va], filt[a].mu[n_tr:n_va], filt[a].s2[n_tr:n_va],
                                                      filt[a].ll[n_tr:n_va])[0] for a in ARMS})
    val.index.name = "validation, 1 day ahead (fixed origin)"

    # 2. walk-forward test, all horizons --------------------------------------------------
    t = time.time()
    summarize = ev.make_summarizer(y, HORIZONS)
    one, multi, refits = walk_forward(base, w, y, X, stochastic, n_va, n, BLOCK, list(HORIZONS),
                                      N_PATHS, summarize, warm=fits, seed=2026)
    print(f"walk-forward: {len(refits)} refits, {N_PATHS} paths per origin, {time.time() - t:.0f}s")
    summ = {a: ev.concat(multi[a]) for a in ARMS}
    np.savez(os.path.join(OUT, "scores.npz"), label=base.label(), horizons=list(HORIZONS),
             origins=summ["sarimax"]["origins"], test_start=n_va,
             **{f"crps_{a}": summ[a]["crps"] for a in ARMS},
             **{f"ll_{a}": one[a]["ll"][n_va:] for a in ARMS},
             **{f"mean_{a}": summ[a]["mean"] for a in ARMS}, truth=summ["sarimax"]["truth"])

    one_test, z_test = {}, {}
    for a in ARMS:
        o = {key: v[n_va:] for key, v in one[a].items()}
        one_test[a], z_test[a] = ev.one_step_metrics(w[n_va:], o["mu"], o["s2"], o["ll"])
    one_df = pd.DataFrame({SHORT[a]: one_test[a] for a in ARMS})
    one_df.index.name = "test, 1 day ahead (exact density)"

    by_h = {h: pd.DataFrame({SHORT[a]: ev.horizon_metrics(summ[a], hi, h) for a in ARMS})
            for hi, h in enumerate(HORIZONS)}
    for h, df in by_h.items():
        df.index.name = f"test, {HORIZONS[h]} ahead"

    # 3. what each step adds: log score (1 day, exact) and CRPS (every horizon) -----------
    steps = [(STEP[a], a, LADDER[a]) for a in ARMS[1:]] + [("total", "full", "sarimax")]
    rows, crps_gain, crps_p = [], {}, {}
    for name, a, b in steps:
        g1, p1 = ev.diebold_mariano(one[a]["ll"][n_va:], one[b]["ll"][n_va:])
        row = {"step": name, "comparison": f"{SHORT[a]} vs {SHORT[b]}", "1d log-score gain / day": g1, "p": p1}
        for hi, h in enumerate(HORIZONS):
            ca, cb = summ[a]["crps"][:, hi], summ[b]["crps"][:, hi]
            gain, p = ev.diebold_mariano(cb, ca, lag=max(h, int(len(ca) ** (1 / 3))))   # positive = a better
            crps_gain[(name, h)] = 100 * gain / cb.mean()
            crps_p[(name, h)] = p
            row[f"CRPS gain {HORIZONS[h]} (%)"] = 100 * gain / cb.mean()
            row[f"p {HORIZONS[h]}"] = p
        rows.append(row)
    steps_df = pd.DataFrame(rows).set_index("step")
    print("\n" + steps_df.round(3).to_string())

    # report ------------------------------------------------------------------------------
    report = [f"# SARIMAX + GARCH, 1 day to 1 month ahead: results", "",
              f"Mean equation from identification.py, lowest {args.criterion}: **{title}**"
              + (f" ({', '.join(regressors)})" if regressors else "") + ".", "",
              "| window | dates | days |", "|---|---|---|",
              f"| train | {dates[0].date()} .. {TRAIN_END} | {n_tr} |",
              f"| validation | .. {VAL_END} | {n_va - n_tr} |",
              f"| test | .. {dates[-1].date()} | {n - n_va} (walk-forward, refit every {BLOCK} days, "
              f"{N_PATHS} simulated paths per origin) |", "",
              "## Parameters (train fit)", "", md_table(param_table(fits, regressors)), "",
              "## Validation, 1 day ahead", "", md_table(val, "{:.4g}"), "",
              "## Test, 1 day ahead (exact density)", "", md_table(one_df, "{:.4g}"), ""]
    for h, df in by_h.items():
        report += [f"## Test, {HORIZONS[h]} ahead (cumulative log return, %)", "", md_table(df, "{:.4g}"), ""]
    report += ["## What each step adds", "",
               "1-day column: mean log-score gain per day (exact densities). CRPS columns: percentage "
               "reduction in CRPS against the previous rung (positive = better). p: Diebold-Mariano "
               "with HAC variance (at least h lags for overlapping h-day windows).", "",
               md_table(steps_df, "{:.3g}"), ""]
    with open(os.path.join(OUT, "metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print("\n" + pd.concat({h: df.loc[["CRPS", "cover 90%", "VaR 5% hit rate"]] for h, df in by_h.items()})
          .round(4).to_string())

    # figures -------------------------------------------------------------------------------
    plot_training_loss(fits, title)
    s2_all = {a: np.where(np.arange(n) < n_va, filt[a].s2, one[a]["s2"]) for a in ARMS}
    plot_volatility(w, dates, n_tr, n_va, s2_all, title)
    plot_horizons(summ, title)
    plot_step_heatmap(crps_gain, crps_p, steps, title)
    plot_fan(summ, dates, title)
    plot_pit(summ, title)
    plot_diagnostics(z_test, one_test)
    print(f"\nwrote {OUT}")


# --------------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def plot_training_loss(fits, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, a in zip(axes.flat, ARMS):
        h = fits[a].history
        for name, hr in h["runs"].items():
            kept = name == "" or name == h.get("kept")
            start = name.replace(" start", "")
            suffix = f" (from {SHORT.get(start, start)} start)" if name else ""
            ax.plot(hr["train"], color=COLOR[a], alpha=1 if kept else 0.35, label="train" + suffix)
            ax.plot(hr["val"], color=COLOR[a], alpha=1 if kept else 0.35, ls="--", label="validation" + suffix)
        ax.set_title(LABEL[a], fontsize=10)
        ax.set_xlabel("L-BFGS iteration")
        ax.legend(loc="center right", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("negative log-likelihood per day")
    fig.suptitle(f"Training loss, {title}: train and validation NLL per optimiser iteration",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "training_loss.png")


def plot_volatility(w, dates, n_tr, n_va, s2_all, title):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, np.abs(w), color=AXIS, lw=0.6, label="|daily return|")
    for a in ("sarimax", "joint", "full"):
        ax.plot(dates, np.sqrt(s2_all[a]), color=COLOR[a], lw=2.0 if a == "sarimax" else 1.2,
                label=f"{SHORT[a]} sigma_t")
    ax.set_ylim(0, np.percentile(np.abs(w), 99.8))
    ax.axvspan(dates[n_tr], dates[n_va - 1], color="#f0efec", zorder=0, lw=0)
    ax.axvspan(dates[n_va], dates[-1], color="#e6e5e0", zorder=0, lw=0)
    top = ax.get_ylim()[1]
    for x, text in ((dates[0], "train"), (dates[n_tr], "validation"), (dates[n_va], "test (walk-forward)")):
        ax.text(x, top, "  " + text, va="top", color=INK_2, fontsize=9)
    ax.set_ylabel("% per day")
    ax.set_title(f"One-day-ahead volatility, {title}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    fig.tight_layout()
    save(fig, "conditional_volatility.png")


def plot_horizons(summ, title):
    hs = list(HORIZONS)
    x = np.arange(len(hs))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = {a: [ev.horizon_metrics(summ[a], hi, h) for hi, h in enumerate(hs)] for a in ARMS}
    base = np.array([m["CRPS"] for m in metrics["sarimax"]])
    panels = [
        (axes[0], lambda m: 100 * (1 - np.array([v["CRPS"] for v in m]) / base),
         "CRPS skill vs SARIMAX (%), higher is better"),
        (axes[1], lambda m: 100 * (np.array([v["cover 90%"] for v in m]) - 0.90),
         "90% interval coverage minus nominal (pp)"),
        (axes[2], lambda m: 100 * (np.array([v["VaR 5% hit rate"] for v in m]) - 0.05),
         "5% VaR hit rate minus nominal (pp)"),
    ]
    for ax, fn, ttl in panels:
        ax.axhline(0, color=INK, lw=1, ls="--")
        for a in ARMS:
            ax.plot(x, fn(metrics[a]), color=COLOR[a], marker="o", ms=7, mec=SURFACE, mew=1.5, label=SHORT[a])
        ax.set_xticks(x, [HORIZONS[h] for h in hs])
        ax.set_title(ttl)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ARMS))
    fig.suptitle(f"Test-window forecast quality by horizon, {title}",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "horizon_scores.png")


def plot_step_heatmap(gain, pval, steps, title):
    names = [s[0] for s in steps]
    hs = list(HORIZONS)
    M = np.array([[gain[(n, h)] for h in hs] for n in names])
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for i, n in enumerate(names):
        for j, h in enumerate(hs):
            p = pval[(n, h)]
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(j, i, f"{M[i, j]:+.2f}%{star}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(hs)), [HORIZONS[h] for h in hs])
    ax.set_yticks(range(len(names)), [f"{n}\n({SHORT[a]} vs {SHORT[b]})" for n, a, b in steps], fontsize=9)
    fig.colorbar(im, ax=ax, label="CRPS reduction vs previous rung (%)", shrink=0.8)
    ax.set_title(f"What each modelling step adds, by horizon (test; * p<0.05, ** p<0.01), {title}")
    fig.tight_layout()
    save(fig, "step_gains.png")


def plot_fan(summ, dates, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for col, (hi, h) in enumerate(((0, 1), (4, 21))):
        for row, a in enumerate(("sarimax", "full")):
            ax = axes[row, col]
            s = summ[a]
            dt = dates[s["origins"]]
            q = s["q"][:, hi, :]
            y = s["truth"][:, hi]
            ax.fill_between(dt, q[:, 1], q[:, 4], color=COLOR[a], alpha=0.18, lw=0, label="95% interval")
            ax.fill_between(dt, q[:, 2], q[:, 3], color=COLOR[a], alpha=0.30, lw=0, label="90% interval")
            ax.plot(dt, y, color=INK_2, lw=0.8, label="realised")
            out = (y < q[:, 1]) | (y > q[:, 4])
            ax.scatter(dt[out], y[out], s=12, color=INK, zorder=3, label=f"outside 95% ({out.sum()})")
            ax.set_title(f"{SHORT[a]}: {HORIZONS[h]} ahead")
            ax.set_ylabel("cumulative log return, %")
            ax.legend(loc="lower left", ncol=4, fontsize=8)
    fig.suptitle(f"Predictive intervals from each test origin, {title}",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "fan_charts.png")


def plot_pit(summ, title):
    fig, axes = plt.subplots(len(ARMS), len(HORIZONS), figsize=(15, 13), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for i, a in enumerate(ARMS):
        for j, (hi, h) in enumerate(zip(range(len(HORIZONS)), HORIZONS)):
            ax = axes[i, j]
            ax.hist(summ[a]["pit"][:, hi], bins=bins, density=True, color=COLOR[a], edgecolor=SURFACE, lw=1.5)
            ax.axhline(1, color=INK, lw=1, ls="--")
            ax.set_ylim(0, 2.2)
            if i == 0:
                ax.set_title(HORIZONS[h])
            if j == 0:
                ax.set_ylabel(SHORT[a], fontsize=9)
    fig.suptitle(f"PIT histograms on the test window (flat = calibrated; U = too narrow, hump = too wide), {title}",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "pit_by_horizon.png")


def plot_diagnostics(z_test, one_test):
    fig, axes = plt.subplots(1, len(ARMS), figsize=(16, 3.8), sharey=True)
    lags = np.arange(1, 21)
    for ax, a in zip(axes, ARMS):
        z2 = z_test[a] ** 2
        rho = acf(z2, nlags=20)[1:]
        band = 1.96 / np.sqrt(len(z2))
        ax.axhspan(-band, band, color="#f0efec", lw=0)
        ax.vlines(lags, 0, rho, color=COLOR[a], lw=2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(f"{SHORT[a]}\nLjung-Box p = {one_test[a]['Ljung-Box z^2 (10) p']:.3g}", fontsize=10)
        ax.set_xlabel("lag (days)")
    axes[0].set_ylabel("ACF of z_t^2")
    fig.suptitle("Leftover volatility clustering in one-day standardised test residuals",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "residual_diagnostics.png")


if __name__ == "__main__":
    main()
