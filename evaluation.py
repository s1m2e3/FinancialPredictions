"""Out-of-sample scoring.

One day ahead the predictive density is exact, so it is scored by its log-likelihood.
For h-day cumulative returns (h = 1, 3, 5, 10, 21 business days) the predictive
distribution comes from simulated paths and is scored from the samples:

  CRPS      continuous ranked probability score, a proper score for the whole
            distribution in the units of the target (lower is better)
  MSE       of the predictive mean
  coverage  of the central 90% and 95% intervals
  VaR hits  how often the realised return falls below the 5% / 1% quantile
  PIT       predictive CDF at the realised value (uniform if calibrated)

Consecutive h-day windows overlap, so their scores are autocorrelated; the
Diebold-Mariano test uses a HAC variance with at least h lags.
"""
import numpy as np
from scipy import stats
from scipy.special import xlogy
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

QUANTILES = (0.01, 0.025, 0.05, 0.95, 0.975)


def make_summarizer(y, horizons):
    """Reduce simulated samples (n_origins, M, n_horizons) to per-origin statistics."""
    horizons = list(horizons)

    def summarize(samples, origins):
        truth = np.stack([y[origins + h - 1] - y[origins - 1] for h in horizons], axis=1)
        xs = np.sort(samples, axis=1)
        M = xs.shape[1]
        rank_weight = (2 * np.arange(1, M + 1) - M - 1)[None, :, None]
        spread = 2 * (rank_weight * xs).sum(axis=1) / M ** 2           # E|X - X'|
        crps = np.abs(xs - truth[:, None, :]).mean(axis=1) - 0.5 * spread
        return {
            "origins": origins,
            "truth": truth,
            "mean": xs.mean(axis=1),
            "crps": crps,
            "pit": (xs <= truth[:, None, :]).mean(axis=1),
            "q": np.moveaxis(np.quantile(xs, QUANTILES, axis=1), 0, -1),   # (n_o, n_h, n_q)
        }

    return summarize


def concat(parts):
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}


def kupiec(hits, level):
    """Unconditional coverage LR test: is the violation rate equal to `level`?"""
    n, x = len(hits), hits.sum()
    pi = x / n
    lr = -2 * (xlogy(n - x, 1 - level) + xlogy(x, level) - xlogy(n - x, 1 - pi) - xlogy(x, pi))
    return stats.chi2.sf(lr, 1)


def christoffersen(hits):
    """Independence LR test: do violations cluster (a hit makes the next hit likelier)?"""
    h = hits.astype(int)
    prev, nxt = h[:-1], h[1:]
    n00, n01 = np.sum((prev == 0) & (nxt == 0)), np.sum((prev == 0) & (nxt == 1))
    n10, n11 = np.sum((prev == 1) & (nxt == 0)), np.sum((prev == 1) & (nxt == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    return stats.chi2.sf(-2 * (l0 - l1), 1)


def diebold_mariano(score_a, score_b, lag=None):
    """Equal predictive accuracy test with a HAC (Newey-West, Bartlett) variance.

    Returns (mean of score_a - score_b, two-sided p-value). The sign convention is up
    to the caller: pass log scores for "higher is better", CRPS for "lower is better".
    """
    d = score_a - score_b
    n = len(d)
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    dc = d - d.mean()
    lrv = dc @ dc / n
    for j in range(1, lag + 1):
        lrv += 2 * (1 - j / (lag + 1)) * (dc[j:] @ dc[:-j]) / n
    return d.mean(), 2 * stats.norm.sf(abs(d.mean() / np.sqrt(lrv / n)))


def horizon_metrics(summary, hi, h):
    """Metrics at horizon index hi (h days) from a concatenated summary."""
    y, mean, q = summary["truth"][:, hi], summary["mean"][:, hi], summary["q"][:, hi, :]
    q01, q025, q05, q95, q975 = (q[:, i] for i in range(len(QUANTILES)))
    out = {
        "CRPS": summary["crps"][:, hi].mean(),
        "MSE": np.mean((y - mean) ** 2),
        "MSE, zero forecast": np.mean(y ** 2),
        "90% interval width": np.mean(q95 - q05),
        "cover 90%": np.mean((y >= q05) & (y <= q95)),
        "cover 95%": np.mean((y >= q025) & (y <= q975)),
        "VaR 5% hit rate": np.mean(y < q05),
        "VaR 1% hit rate": np.mean(y < q01),
    }
    if h == 1:   # windows do not overlap, so the classic backtests apply
        out["VaR 5% Kupiec p"] = kupiec(y < q05, 0.05)
        out["VaR 5% indep. p"] = christoffersen(y < q05)
        out["VaR 1% Kupiec p"] = kupiec(y < q01, 0.01)
    return out


def one_step_metrics(w, mu, s2, ll):
    """Exact one-day-ahead density scores and residual diagnostics."""
    z = (w - mu) / np.sqrt(s2)
    return {
        "NLL / day": -ll.mean(),
        "Ljung-Box z (10) p": acorr_ljungbox(z, lags=[10])["lb_pvalue"].iloc[0],
        "Ljung-Box z^2 (10) p": acorr_ljungbox(z ** 2, lags=[10])["lb_pvalue"].iloc[0],
        "ARCH-LM z (5) p": het_arch(z, nlags=5)[1],
    }, z
