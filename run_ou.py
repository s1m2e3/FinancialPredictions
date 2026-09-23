"""Does mean reversion in the price level help? OU-with-jumps vs the SARIMAX + GARCH models.

The OU SDE on X = 100 log P,

    dX_t = kappa (theta_t - X_t) dt + sigma dW_t + J dN_t,

sampled once a day is exactly an AR(1) in levels with phi = exp(-kappa), plus the jumps:

    X_t - theta_t = phi (X_{t-1} - theta_{t-1}) + eps_t,   half-life = ln 2 / kappa days.

Three versions, all with Poisson-normal jumps:

  ou_jump              constant target theta, constant diffusion variance (the SDE as written)
  ou_trend_jump        linear-trend target theta + b t
  ou_trend_garch_jump  linear-trend target with the GJR-GARCH diffusion: the same variance and
                       jump model as the SARIMAX "jump" arm, so the two differ only in whether
                       the level reverts (OU) or the returns carry the dynamics (SARIMAX)

Same test window, quarterly walk-forward refits, horizons, path count and seeds as
run_armagarch.py, scored against its saved per-origin scores (scores.npz) with
Diebold-Mariano tests. One-day log scores are comparable because the density of the level
given the past equals the density of the return given the past.

Run from the repository root after run_armagarch.py:  python run_ou.py [--criterion AICc|BIC]
Writes results/ou/.
"""
import argparse
import os
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

import evaluation as ev                                             # noqa: E402
from armagarch import Spec, fit, fit_multistart, simulate          # noqa: E402
from data import build_dataset, load_prices                         # noqa: E402
from run_armagarch import (BLOCK, COLOR, HORIZONS, INK, LABEL, N_PATHS, ROOT, SHORT,  # noqa: E402
                           md_table)

warnings.filterwarnings("ignore")

OU = {
    "ou_jump": ("OU + jumps, constant target", Spec(p=1, d=0, jumps=True)),
    "ou_trend_jump": ("OU + jumps, trend target", Spec(p=1, d=0, k=1, jumps=True)),
    "ou_trend_garch_jump": ("OU + GARCH + jumps, trend target", Spec(p=1, d=0, k=1, garch=True, jumps=True)),
}
OU_STYLE = {"ou_jump": ":", "ou_trend_jump": "--", "ou_trend_garch_jump": "-"}
OU_COLOR = "#e34948"
BENCH = ("sarimax", "full", "jump")          # arms from the saved SARIMAX run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--criterion", choices=["AICc", "BIC"], default="AICc",
                        help="which SARIMAX run to compare against")
    args = parser.parse_args()
    out_dir = os.path.join(ROOT, "results", "ou")
    os.makedirs(out_dir, exist_ok=True)
    saved = np.load(os.path.join(ROOT, "results", f"sarimax_garch_{args.criterion.lower()}", "scores.npz"))

    d = build_dataset(load_prices())
    y, dates, n_tr, n_va, n = d.y, d.dates, d.n_train, d.n_val, d.n
    w = y.copy()                                           # d = 0: the model is on the level itself
    trend = (np.arange(n) / 252.0)[:, None]                # years since the start: theta_t = c + b t
    no_x = np.zeros((n, 0))
    H = max(HORIZONS)
    summarize = ev.make_summarizer(y, HORIZONS)

    train_rows, one, summ = {}, {}, {}
    for name, (label, spec) in OU.items():
        X = trend if spec.k else no_x
        stochastic = np.zeros(spec.k, bool)
        t = time.time()
        f = fit_multistart(spec, w, X, n_tr)
        P = f.params
        kappa = -np.log(P["phi"][0])
        train_rows[label] = {"phi": P["phi"][0], "kappa per day": kappa, "half-life (days)": np.log(2) / kappa,
                             "target at start (level)": np.exp(P["c"] / 100),
                             "trend (% per year)": P["beta"][0] if spec.k else 0.0,
                             "diffusion variance": P["sbar2"], "GARCH persistence": P["persistence"],
                             "jumps per day": P["lam_j"], "mean jump (%)": P["mu_j"], "jump s.d. (%)": P["sig_j"],
                             "train NLL / day": f.train_nll}

        ll = np.full(n, np.nan)
        parts = []
        for bi, s in enumerate(range(n_va, n, BLOCK)):
            e = min(s + BLOCK, n)
            f = fit(spec, w, X, s, u0=f.u)                  # warm start from the previous window
            filt = f.filter(w, X)
            ll[s:e] = filt.ll[s:e]
            origins = np.arange(s, min(e, n - H + 1))
            if len(origins):
                parts.append(summarize(simulate(f, filt, y, X, stochastic, origins, list(HORIZONS),
                                                N_PATHS, 2026 + bi), origins))
        one[name], summ[name] = ll[n_va:], ev.concat(parts)
        assert np.array_equal(summ[name]["origins"], saved["origins"]), "origins must match the saved run"
        print(f"{label}: {time.time() - t:.0f}s, train half-life {train_rows[label]['half-life (days)']:,.0f} days")

    # head-to-head against the saved SARIMAX-family arms ------------------------------------
    pairs = [(o, b) for o in OU for b in BENCH]
    rows, crps_skill = [], {}
    for o, b in pairs:
        g1, p1 = ev.diebold_mariano(one[o], saved[f"ll_{b}"])
        row = {"comparison": f"{OU[o][0]} vs {SHORT[b]}", "1d log-score gain / day": g1, "p": p1}
        for hi, h in enumerate(HORIZONS):
            co, cb = summ[o]["crps"][:, hi], saved[f"crps_{b}"][:, hi]
            gain, p = ev.diebold_mariano(cb, co, lag=max(h, int(len(co) ** (1 / 3))))   # positive = OU better
            row[f"CRPS {HORIZONS[h]} (%)"] = 100 * gain / cb.mean()
            row[f"p {HORIZONS[h]}"] = p
        rows.append(row)
    h2h = pd.DataFrame(rows).set_index("comparison")

    truth = saved["truth"]
    base_crps = saved["crps_sarimax"].mean(axis=0)
    for a in BENCH:
        crps_skill[a] = 100 * (1 - saved[f"crps_{a}"].mean(axis=0) / base_crps)
    for o in OU:
        crps_skill[o] = 100 * (1 - summ[o]["crps"].mean(axis=0) / base_crps)
    mse = {SHORT[a]: np.mean((truth - saved[f"mean_{a}"]) ** 2, axis=0) for a in BENCH}
    mse.update({OU[o][0]: np.mean((truth - summ[o]["mean"]) ** 2, axis=0) for o in OU})
    mse["zero forecast"] = np.mean(truth ** 2, axis=0)
    mse = pd.DataFrame(mse, index=[HORIZONS[h] for h in HORIZONS]).T
    mse.index.name = "MSE of the predictive mean"
    cover = {}
    for o in OU:
        cover[OU[o][0]] = {HORIZONS[h]: ev.horizon_metrics(summ[o], hi, h)["cover 90%"]
                           for hi, h in enumerate(HORIZONS)}
    cover = pd.DataFrame(cover).T
    cover.index.name = "90% interval coverage"
    params = pd.DataFrame(train_rows)
    params.index.name = "parameter (train fit, 2013-2020)"
    print(params.round(4).to_string(), "\n\n", h2h.round(3).to_string(), "\n\n", mse.round(3).to_string())

    report = ["# OU with jumps vs SARIMAX + GARCH", "",
              f"Compared with the {args.criterion} SARIMAX run on the same {len(saved['origins'])} test origins "
              "(2023-2025, walk-forward, quarterly refits).", "",
              "## OU parameters (train fit)", "", md_table(params, "{:.4g}"), "",
              "## Head to head", "",
              "Positive = the OU model is better. CRPS columns: percentage reduction in CRPS; p: Diebold-Mariano "
              "with HAC variance (at least h lags).", "", md_table(h2h, "{:.3g}"), "",
              "## Mean forecasts", "", md_table(mse, "{:.3f}"), "",
              "## OU interval coverage (nominal 90%)", "", md_table(cover, "{:.3f}"), ""]
    with open(os.path.join(out_dir, "ou_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(HORIZONS))
    ax.axhline(0, color=INK, lw=1, ls="--")
    for a in ("full", "jump"):
        ax.plot(x, crps_skill[a], color=COLOR[a], marker="o", ms=7, label=LABEL[a])
    for o in OU:
        ax.plot(x, crps_skill[o], color=OU_COLOR, ls=OU_STYLE[o], marker="s", ms=6, label=OU[o][0])
    ax.set_xticks(x, list(HORIZONS.values()))
    ax.set_ylabel("CRPS skill vs plain SARIMAX (%), higher is better")
    ax.set_title("Mean reversion in the level (OU) vs SARIMAX + GARCH, test window 2023-2025")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ou_vs_sarimax_garch.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
