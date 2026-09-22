"""Head-to-head test of the AICc and BIC SARIMA orders, model by model and horizon by horizon.

Both runs of run_armagarch.py score the same test origins, so for every arm the two
mean equations can be compared on identical days with a Diebold-Mariano test:

  CRPS         h-day predictive distributions, HAC variance with at least h lags
  log score    exact one-day densities
  MSE          of the predictive mean, next to the zero-return forecast

Run from the repository root after both
    python run_armagarch.py
    python run_armagarch.py --criterion BIC
Writes results/spec_comparison/{spec_comparison.md, spec_comparison.png}.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap            # noqa: E402

import evaluation as ev                                           # noqa: E402
from armagarch import ARMS                                        # noqa: E402
from run_armagarch import HORIZONS, SHORT, md_table               # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = {"AICc": os.path.join(ROOT, "results", "sarimax_garch_aicc", "scores.npz"),
        "BIC": os.path.join(ROOT, "results", "sarimax_garch_bic", "scores.npz")}
OUT = os.path.join(ROOT, "results", "spec_comparison")
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
INK, INK_2 = "#0b0b0b", "#52514e"


def main():
    os.makedirs(OUT, exist_ok=True)
    A, B = (np.load(RUNS[k]) for k in ("AICc", "BIC"))
    assert np.array_equal(A["origins"], B["origins"]), "the two runs must score the same origins"
    la, lb = str(A["label"]), str(B["label"])
    hs = list(HORIZONS)
    n = len(A["origins"])

    rows, gain, pval = [], {}, {}
    for a in ARMS:
        g1, p1 = ev.diebold_mariano(B[f"ll_{a}"], A[f"ll_{a}"])
        row = {"model": SHORT[a], "1d log-score gain / day": g1, "p (log score)": p1}
        for hi, h in enumerate(hs):
            ca, cb = A[f"crps_{a}"][:, hi], B[f"crps_{a}"][:, hi]
            diff, p = ev.diebold_mariano(ca, cb, lag=max(h, int(n ** (1 / 3))))   # positive = BIC better
            gain[(a, h)], pval[(a, h)] = 100 * diff / ca.mean(), p
            row[f"CRPS {HORIZONS[h]} (%)"] = 100 * diff / ca.mean()
            row[f"p {HORIZONS[h]}"] = p
        rows.append(row)
    crps = pd.DataFrame(rows).set_index("model")

    truth = A["truth"]
    mse = {}
    for a in ARMS:
        for label, run in (("AICc", A), ("BIC", B)):
            mse[(SHORT[a], label)] = {HORIZONS[h]: np.mean((truth[:, hi] - run[f"mean_{a}"][:, hi]) ** 2)
                                      for hi, h in enumerate(hs)}
    mse = pd.DataFrame(mse).T
    mse.loc[("zero forecast", ""), :] = [np.mean(truth[:, hi] ** 2) for hi in range(len(hs))]
    mse.index = [f"{m} ({c})" if c else m for m, c in mse.index]
    mse.index.name = "MSE of the predictive mean"
    print(crps.round(3).to_string(), "\n\n", mse.round(3).to_string())

    report = ["# AICc order vs BIC order", "",
              f"AICc: **{la}**, BIC: **{lb}**. Same six models, same {n} test origins "
              "(2023-2025, walk-forward).", "",
              "Positive = the BIC order forecasts better. CRPS columns are percentage reductions in CRPS; "
              "p-values are Diebold-Mariano with HAC variance (at least h lags).", "",
              md_table(crps, "{:.3g}"), "",
              "## Mean forecasts", "", md_table(mse, "{:.3f}"), ""]
    with open(os.path.join(OUT, "spec_comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))

    M = np.array([[gain[(a, h)] for h in hs] for a in ARMS])
    lim = max(np.abs(M).max(), 0.5)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for i, a in enumerate(ARMS):
        for j, h in enumerate(hs):
            p = pval[(a, h)]
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(j, i, f"{M[i, j]:+.2f}%{star}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(hs)), [HORIZONS[h] for h in hs])
    ax.set_yticks(range(len(ARMS)), [SHORT[a] for a in ARMS])
    ax.tick_params(colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="CRPS reduction, BIC order vs AICc order (%)", shrink=0.8)
    ax.set_title(f"{lb} (BIC) vs {la} (AICc), test window; * p<0.05, ** p<0.01",
                 loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spec_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
