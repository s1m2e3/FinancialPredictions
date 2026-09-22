"""Box-Jenkins identification of the SARIMAX mean equation, on the training window only.

  1. Regular differencing d: ADF, KPSS and Phillips-Perron on y = 100 log P and on its
     first difference; d is pmdarima's ndiffs (KPSS, the auto_arima default).
  2. Seasonal differencing D for each candidate season s: OCSB test (auto_arima's default,
     decides D), Canova-Hansen test and STL seasonal strength (reported for reference).
  3. Evidence of seasonal autocorrelation in w: Ljung-Box at lags s, 2s, 3s (classic and
     heteroskedasticity-robust) and the periodogram with Fisher's g test.
  4. Calendar regressors (the X): day-of-week, month-of-year and turn-of-month dummies
     and lagged constituent returns, one regression with HAC standard errors; a block
     enters the model if its Wald test rejects at 5%.
  5. Orders: every SARIMA(p,d,q)(P,D,Q)[s] with p, q <= 5, P, Q <= 2, s in {5, 21, 63}
     is fitted by Gaussian maximum likelihood on a common sample and ranked by AICc
     (auto_arima's default criterion); BIC is reported alongside.
  6. Diagnostics of the chosen model: residual autocorrelation, squared-residual
     autocorrelation (ARCH effects) and normality.

The annual season (s = 252) is tested but not used as a SARMA period: eight years of
training data cannot support lag-252 terms; month-of-year dummies carry annual effects.

Run from the repository root:  python identification.py [--reuse-grid]
(--reuse-grid reads results/identification/order_search.csv instead of refitting the
900 candidate models, which takes about 15 minutes.)
Writes results/identification/{identification.md, spec.json, *.png}.
"""
import itertools
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from armagarch import Spec, difference, fit_multistart

# The order search runs in worker processes, which on Windows re-import this module.
# Only numpy and the model are imported at the top so each worker stays small; the
# statistics and plotting stacks are imported inside main().

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "identification")
SEASONS = {5: "week", 21: "month", 63: "quarter", 252: "year"}
SARMA_SEASONS = (5, 21, 63)
MAX_P, MAX_Q, MAX_SP, MAX_SQ = 5, 5, 2, 2    # auto_arima defaults
WORKERS = 4                     # each worker holds its own copy of the data and numba code

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _import_stack():
    """Heavy imports, bound as module globals, only in the parent process."""
    global plt, pd, sm, PhillipsPerron, CHTest, OCSBTest, ndiffs, signal, stats, gammaln
    global acorr_ljungbox, het_arch, STL, acf, adfuller, kpss, pacf
    global REGRESSOR_BLOCKS, STOCHASTIC, build_dataset, load_prices
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from arch.unitroot import PhillipsPerron
    from pmdarima.arima import CHTest, OCSBTest, ndiffs
    from scipy import signal, stats
    from scipy.special import gammaln
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
    from data import REGRESSOR_BLOCKS, STOCHASTIC, build_dataset, load_prices
    warnings.filterwarnings("ignore")
    _style(plt)


def _style(plt):
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


def robust_acf_se(x, nlags):
    """Standard error of the sample autocorrelation that stays valid under
    heteroskedasticity: se_k = sqrt(sum (x_t x_{t-k})^2) / sum x_t^2."""
    x = x - x.mean()
    denom = x @ x
    return np.array([np.sqrt(np.sum((x[k:] * x[:-k]) ** 2)) / denom for k in range(1, nlags + 1)])


def seasonal_ljung_box(x, s, n_seasonal_lags=3):
    """Ljung-Box restricted to lags s, 2s, 3s: classic and heteroskedasticity-robust p-values."""
    x = x - x.mean()
    n = len(x)
    lags = [s * j for j in range(1, n_seasonal_lags + 1)]
    rho = acf(x, nlags=max(lags), fft=True)
    q = n * (n + 2) * sum(rho[k] ** 2 / (n - k) for k in lags)
    q_rob = sum((x[k:] @ x[:-k]) ** 2 / np.sum((x[k:] * x[:-k]) ** 2) for k in lags)
    return stats.chi2.sf(q, len(lags)), stats.chi2.sf(q_rob, len(lags))


def robust_ljung_box(x, n_lags, model_df=0):
    """Ljung-Box over lags 1..n_lags with heteroskedasticity-robust autocorrelation
    variances; chi-square with n_lags - model_df degrees of freedom."""
    x = x - x.mean()
    q = sum((x[k:] @ x[:-k]) ** 2 / np.sum((x[k:] * x[:-k]) ** 2) for k in range(1, n_lags + 1))
    return stats.chi2.sf(q, n_lags - model_df)


def fisher_g(periodogram):
    """Fisher's exact test that the largest periodogram ordinate is noise."""
    I = periodogram[1:]                       # drop the zero frequency
    m = len(I)
    g = I.max() / I.sum()
    # P(G > g) = sum_{j=1}^{floor(1/g)} (-1)^(j-1) C(m, j) (1 - j g)^(m-1)
    j = np.arange(1, int(np.floor(1 / g)) + 1)
    log_comb = gammaln(m + 1) - gammaln(j + 1) - gammaln(m - j + 1)
    terms = (-1.0) ** (j - 1) * np.exp(log_comb + (m - 1) * np.log1p(-j * g))
    return g, float(np.clip(terms.sum(), 0, 1))


def stl_strength(y, period):
    res = STL(y, period=period, robust=True).fit()
    return max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))


def _fit_candidate(args):
    spec, w, X, t_end, t0 = args
    f = fit_multistart(spec, w, X, t_end, t0=t0)
    return spec, f.train_nll, f.info_criteria(), f.u


def main(reuse_grid=False):
    _import_stack()
    os.makedirs(OUT, exist_ok=True)
    d = build_dataset(load_prices())
    n_tr = d.n_train
    y_tr = d.y[:n_tr]
    report = ["# SARIMAX identification (Box-Jenkins), training window "
              f"{d.dates[0].date()} .. {d.dates[n_tr - 1].date()}, {n_tr} days", ""]

    # 1. regular differencing ------------------------------------------------------------
    rows = []
    for name, series, reg in (("level: 100 log P", y_tr, "ct"), ("first difference", np.diff(y_tr), "c")):
        rows.append({"series": name,
                     "ADF p (H0 unit root)": adfuller(series, regression=reg, autolag="AIC")[1],
                     "Phillips-Perron p (H0 unit root)": PhillipsPerron(series, trend=reg).pvalue,
                     "KPSS p (H0 stationary)": kpss(series, regression=reg, nlags="auto")[1]})
    unit = pd.DataFrame(rows).set_index("series")
    d_by = {test: ndiffs(y_tr, test=test, max_d=2) for test in ("kpss", "adf", "pp")}
    d_order = d_by["kpss"]
    print(unit.round(4).to_string(), "\nndiffs:", d_by)
    report += ["## 1. Regular differencing", "", md_table(unit), "",
               f"pmdarima `ndiffs`: KPSS {d_by['kpss']}, ADF {d_by['adf']}, PP {d_by['pp']}. "
               f"KPSS p-values are truncated to [0.01, 0.10] by the published tables. "
               f"**d = {d_order}**.", ""]

    w = difference(d.y, Spec(d=d_order))
    w_tr = w[d_order:n_tr]

    # 2. seasonal differencing and 3. seasonal autocorrelation ------------------------------
    rows = []
    for s, name in SEASONS.items():
        t = time.time()
        try:
            ocsb = OCSBTest(m=s).estimate_seasonal_differencing_term(y_tr)
        except Exception as exc:                       # OCSB can fail for very long seasons
            ocsb = f"failed ({type(exc).__name__})"
        try:
            ch = CHTest(m=s).estimate_seasonal_differencing_term(y_tr)
        except Exception as exc:
            ch = f"failed ({type(exc).__name__})"
        lb, lb_rob = seasonal_ljung_box(w_tr, s)
        rows.append({"season": f"{s} ({name})", "OCSB D": ocsb, "Canova-Hansen D": ch,
                     "STL strength, log price": stl_strength(y_tr, s),
                     "STL strength, differenced": stl_strength(w_tr, s),
                     "Ljung-Box s,2s,3s p": lb, "robust Ljung-Box p": lb_rob})
        print(f"season {s}: {time.time() - t:.1f}s")
    seas = pd.DataFrame(rows).set_index("season")
    D_by_s = {s: int(seas.loc[f"{s} ({SEASONS[s]})", "OCSB D"]) if isinstance(
        seas.loc[f"{s} ({SEASONS[s]})", "OCSB D"], (int, np.integer)) else 0 for s in SARMA_SEASONS}
    freqs, power = signal.periodogram(w_tr - w_tr.mean())
    g, g_p = fisher_g(power)
    peak_period = 1 / freqs[1:][np.argmax(power[1:])]
    print(seas.to_string())
    report += ["## 2-3. Seasonality", "", md_table(seas), "",
               "STL strength above 0.64 is Hyndman's threshold for seasonal differencing. "
               "The robust Ljung-Box uses heteroskedasticity-robust autocorrelation variances; "
               "the classic version over-rejects when volatility clusters.", "",
               f"Periodogram of the differenced series: largest peak at a period of {peak_period:.1f} days, "
               f"Fisher's g = {g:.4f}, p = {g_p:.3f} (H0: no periodic component).", "",
               f"Seasonal differencing used (OCSB): " +
               ", ".join(f"s={s}: D={D}" for s, D in D_by_s.items()) + ".", ""]

    # 4. calendar and exogenous regressors ----------------------------------------------------
    blocks = {name: cols for name, cols in REGRESSOR_BLOCKS.items()}
    all_cols = [c for cols in blocks.values() for c in cols]
    Xr = d.regressors[all_cols].to_numpy()[d_order:n_tr]
    ols = sm.OLS(w_tr, sm.add_constant(Xr)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    rows, chosen = [], []
    for name, cols in blocks.items():
        idx = [1 + all_cols.index(c) for c in cols]
        R = np.zeros((len(idx), Xr.shape[1] + 1))
        R[np.arange(len(idx)), idx] = 1
        wald = ols.wald_test(R, scalar=True)
        p = float(wald.pvalue)
        rows.append({"block": name, "regressors": len(cols), "HAC Wald p": p, "included": p < 0.05})
        if p < 0.05:
            chosen += cols
    exog = pd.DataFrame(rows).set_index("block")
    print(exog.to_string())
    report += ["## 4. Regressors (one OLS with all blocks, Newey-West HAC, 10 lags)", "",
               md_table(exog), "", f"Regressors carried into the model: {chosen or 'none'}.", ""]

    # 5. order search ----------------------------------------------------------------------------
    X = d.X(chosen)
    k = len(chosen)
    candidates = [Spec(p=p, d=d_order, q=q, k=k) for p in range(MAX_P + 1) for q in range(MAX_Q + 1)]
    for s in SARMA_SEASONS:
        for p, q, P, Q in itertools.product(range(MAX_P + 1), range(MAX_Q + 1), range(MAX_SP + 1), range(MAX_SQ + 1)):
            if P or Q:
                candidates.append(Spec(p=p, d=d_order, q=q, P=P, D=D_by_s[s], Q=Q, s=s, k=k))
    t0 = max(c.t0 for c in candidates)            # common sample for every candidate
    print(f"order search: {len(candidates)} candidates, common sample from index {t0} "
          f"({d.dates[t0].date()}), {n_tr - t0} days")
    cached = os.path.join(OUT, "order_search.csv")
    if reuse_grid and os.path.exists(cached):
        grid = pd.read_csv(cached, index_col=0)
        print(f"order search: reusing {cached}")
    else:
        t = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(_fit_candidate, [(c, w, X, n_tr, t0) for c in candidates], chunksize=4))
        print(f"order search took {time.time() - t:.0f}s")
        grid = pd.DataFrame([{"model": s.label(), "params": s.n_params, "NLL / day": nll, **ic}
                             for s, nll, ic, _ in results]).set_index("model")
        grid["dAICc"] = grid["AICc"] - grid["AICc"].min()
        grid["dBIC"] = grid["BIC"] - grid["BIC"].min()
    by_label = {c.label(): c for c in candidates}
    best_spec = by_label[grid["AICc"].idxmin()]
    best_bic = grid["BIC"].idxmin()
    best_nonseasonal = grid[~grid.index.str.contains(r"\[")]["AICc"].idxmin()
    print(grid.sort_values("AICc").head(10).round(2).to_string())
    report += ["## 5. Order search", "",
               f"{len(candidates)} Gaussian SARIMA models, common sample of {n_tr - t0} days "
               f"({d.dates[t0].date()} on). Ten best by AICc:", "",
               md_table(grid.sort_values("AICc").head(10), "{:.2f}"), "",
               "Ten best by BIC:", "", md_table(grid.sort_values("BIC").head(10), "{:.2f}"), "",
               f"**Selected by AICc: {best_spec.label()}**"
               + (f" with {k} regressors" if k else "") + f". Best by BIC: {best_bic}. "
               f"Best non-seasonal by AICc: {best_nonseasonal} "
               f"(dAICc {grid.loc[best_nonseasonal, 'dAICc']:.2f}).", ""]

    # 6. diagnostics of the chosen model -------------------------------------------------------
    f = fit_multistart(best_spec, w, X, n_tr)
    filt = f.filter(w, X)
    e = filt.eps[f.t0:n_tr]
    n_arma = best_spec.p + best_spec.q + best_spec.P + best_spec.Q
    lb_lags = [L for L in (10, 21, 42, 63) if L > n_arma]     # degrees of freedom = lags - n_arma
    lb = acorr_ljungbox(e, lags=lb_lags, model_df=n_arma)
    rows = [{"test": f"Ljung-Box residuals, {L} lags", "p-value": lb.loc[L, "lb_pvalue"]} for L in lb_lags]
    rows += [{"test": f"robust Ljung-Box residuals, {L} lags", "p-value": robust_ljung_box(e, L, n_arma)}
             for L in lb_lags]
    rows += [{"test": f"robust Ljung-Box s={s} lags s,2s,3s", "p-value": seasonal_ljung_box(e, s)[1]}
             for s in SARMA_SEASONS]
    lb2 = acorr_ljungbox(e ** 2, lags=[10])
    rows += [{"test": "Ljung-Box squared residuals, 10 lags (ARCH)", "p-value": lb2.loc[10, "lb_pvalue"]},
             {"test": "ARCH-LM, 5 lags", "p-value": het_arch(e, nlags=5)[1]},
             {"test": "Jarque-Bera (normality)", "p-value": stats.jarque_bera(e).pvalue}]
    diag = pd.DataFrame(rows).set_index("test")
    print(diag.to_string())
    P = f.params
    mean_white = all(diag.loc[f"robust Ljung-Box residuals, {L} lags", "p-value"] > 0.05 for L in lb_lags)
    arch = diag.loc["ARCH-LM, 5 lags", "p-value"] < 0.05
    report += ["## 6. Diagnostics of the selected model (training residuals)", "",
               md_table(diag, "{:.3g}"), "",
               f"Residual skewness {stats.skew(e):.2f}, excess kurtosis {stats.kurtosis(e):.2f}.", "",
               f"Residual autocorrelation (robust Ljung-Box, valid under ARCH): "
               f"{'none left at 5%' if mean_white else 'still present at 5%'}. "
               f"ARCH effects in the residuals: {'yes' if arch else 'no'}; normality "
               f"{'rejected' if diag.loc['Jarque-Bera (normality)', 'p-value'] < 0.05 else 'not rejected'}.", "",
               "Fitted coefficients: " + ", ".join(
                   f"{n}={np.round(P[n], 4).tolist()}" for n in ("c", "phi", "Phi", "theta", "Theta") if np.size(P[n]))
               + (", beta=" + str(dict(zip(chosen, np.round(P["beta"], 4).tolist()))) if k else ""), ""]

    spec_out = dict(p=best_spec.p, d=best_spec.d, q=best_spec.q, P=best_spec.P, D=best_spec.D,
                    Q=best_spec.Q, s=best_spec.s, regressors=chosen,
                    stochastic=[c in STOCHASTIC for c in chosen], label=best_spec.label())
    with open(os.path.join(OUT, "spec.json"), "w") as fh:
        json.dump(spec_out, fh, indent=2)
    with open(os.path.join(OUT, "identification.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    grid.sort_values("AICc").to_csv(os.path.join(OUT, "order_search.csv"))

    plot_identification(d, n_tr, d_order, w_tr, freqs, power, ols, all_cols, e, best_spec)
    print("selected", spec_out)


def plot_identification(d, n_tr, d_order, w_tr, freqs, power, ols, all_cols, resid, spec):
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 1, 1, 1])
    dates = d.dates[:n_tr]

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(dates, d.y[:n_tr] / 100, color=BLUE, lw=1.2)
    ax.set_title("log price, training window")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(dates[d_order:], w_tr, color=BLUE, lw=0.6)
    ax.set_title(f"differenced series (d = {d_order}), % per day")

    nl = 130
    lags = np.arange(1, nl + 1)
    rho = acf(w_tr, nlags=nl, fft=True)[1:]
    classic = 1.96 / np.sqrt(len(w_tr))
    robust = 1.96 * robust_acf_se(w_tr, nl)
    for row, (vals, title) in enumerate(((rho, "ACF of the differenced series"),
                                         (pacf(w_tr, nlags=nl)[1:], "PACF of the differenced series"))):
        ax = fig.add_subplot(gs[1 + row, :])
        ax.fill_between(lags, -robust, robust, color="#e6e5e0", lw=0, label="95% band, heteroskedasticity-robust")
        ax.axhline(classic, color=INK_2, lw=0.8, ls="--", label="95% band, classic (iid)")
        ax.axhline(-classic, color=INK_2, lw=0.8, ls="--")
        ax.vlines(lags, 0, vals, color=BLUE, lw=1.6)
        for s in SARMA_SEASONS:
            for m in range(1, nl // s + 1):
                ax.axvline(s * m, color=ORANGE, lw=0.6, alpha=0.5 if s != 63 else 0.9, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlim(0, nl + 1)
        ax.set_title(title + " (orange: seasonal lags of 5, 21, 63 days)")
        ax.set_xlabel("lag (business days)")
        ax.legend(loc="upper right", ncol=2)

    ax = fig.add_subplot(gs[3, 0])
    period = 1 / freqs[1:]
    smooth = pd.Series(power[1:]).rolling(9, center=True, min_periods=1).mean().to_numpy()
    ax.plot(period, power[1:], color=AXIS, lw=0.5, label="periodogram")
    ax.plot(period, smooth, color=BLUE, lw=1.4, label="smoothed")
    for s in SEASONS:
        ax.axvline(s, color=ORANGE, lw=1, ls="--")
        ax.text(s, ax.get_ylim()[1], f" {s}d", color=ORANGE, va="top", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("period (business days, log scale)")
    ax.set_title("Periodogram of the differenced series")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[3, 1])
    r2 = acf(resid ** 2, nlags=40, fft=True)[1:]
    ax.axhspan(-1.96 / np.sqrt(len(resid)), 1.96 / np.sqrt(len(resid)), color="#e6e5e0", lw=0)
    ax.vlines(np.arange(1, 41), 0, r2, color=ORANGE, lw=2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title(f"ACF of squared {spec.label()} residuals")
    ax.set_xlabel("lag (business days)")

    fig.suptitle("Box-Jenkins identification on the training window (2013-2020)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "identification.png"), dpi=150)
    plt.close(fig)

    # calendar effects with HAC confidence intervals
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 2.2]}, sharey=True)
    params, se = ols.params, ols.bse
    for ax, prefix, labels, base in ((axes[0], "dow_", ["Tue", "Wed", "Thu", "Fri"], "Mon"),
                                     (axes[1], "month_", ["Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                                          "Aug", "Sep", "Oct", "Nov", "Dec"], "Jan")):
        idx = [1 + all_cols.index(prefix + l) for l in labels]
        ax.errorbar(range(len(labels)), params[idx], yerr=1.96 * se[idx], fmt="o", color=BLUE,
                    ecolor=AXIS, elinewidth=2, capsize=0, ms=7)
        ax.axhline(0, color=INK_2, lw=1, ls="--")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(f"{'Day of week' if prefix == 'dow_' else 'Month'} effect vs {base}, % per day (HAC 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "calendar_effects.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    main(reuse_grid="--reuse-grid" in sys.argv)
