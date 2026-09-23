"""Walk-forward features for every stock, as known at the OPEN of each trading day.

Row t describes stock i with information up to the close of day t-1 plus the open of
day t (the overnight gap). Nothing in row t uses a later price.

  GARCH block, per horizon h in (1, 3, 5, 10, 21) trading days
      AR(1)-GJR-GARCH(1,1) with skewed-t errors on 100 x daily log close returns, refitted every 63 days on
      an expanding window (first fit after 504 days of history). For the h-day
      cumulative return from the close of t-1:
        mu_h    closed-form mean:  sum_k (c + phi^(k+1) u_{t-1})
        sig_h   closed-form s.d.:  sqrt(sum_j Psi_{h-1-j}^2 sigma^2_{t+j}),  Psi the
                cumulative AR weights and sigma^2 the GARCH variance path, mean-reverting
                at rate alpha + gamma/2 + b
        q05_h   5% quantile, P(up)_h: skewed-t for h = 1 (exact), normal for h > 1
                (a sum of fat-tailed GARCH shocks, approximated)
      One fixed specification for every stock: the identification work showed the mean
      equation forecasts almost nothing, so the value is in sigma, and 900 candidate
      models per stock would buy nothing.
  price block
      gap (open / previous close), 1-month reversal, 12-1 month momentum, 21-day realised
      volatility, 252-day beta to the S&P 500, distance from the 52-week high,
      log 21-day dollar volume
  cross-sectional block
      the percentile rank of the main features among the stocks in that day's universe
  index membership
      in_sp500: 1 for a member of the S&P 500 that day, 0 for a stock of the outside pool
      (Nasdaq-100 members the S&P 500 has not admitted, stocks_data.py)
  market block (same for every stock)
      the S&P 500 through the full model -- SARIMA(2,1,2)(0,0,1)[5] + GJR-GARCH-in-mean with
      skewed-t errors -- with every horizon's mean, s.d., 5% quantile and P(up) from 1000
      simulated paths per day; VIX, the S&P gap and its distance from the 200-day average

Run from the repository root:  python stock_features.py   (cached to data/stock_features.npz)
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "stock_features.npz")
HORIZONS = (1, 3, 5, 10, 21)
BLOCK = 63
BURN_IN = 504
WORKERS = 4

GARCH_NAMES = [f"{k}_{h}d" for h in HORIZONS for k in ("mu", "sig", "q05", "pup")]
PRICE_NAMES = ["gap", "rev_1m", "mom_12_1", "rv_21", "beta_252", "dist_52w_high", "log_dollar_vol"]
RANKED = ["mu_21d", "sig_21d", "pup_5d", "gap", "rev_1m", "mom_12_1", "rv_21", "beta_252", "dist_52w_high"]
RANK_NAMES = [f"rank_{n}" for n in RANKED]
MARKET_NAMES = [f"spx_{n}" for n in GARCH_NAMES] + ["vix", "spx_gap", "spx_dist_200d"]
MARKET_ORDER = dict(p=2, d=1, q=2, Q=1, s=5)   # the BIC order identified on the Nasdaq Composite
STOCK_NAMES = GARCH_NAMES + PRICE_NAMES + RANK_NAMES + ["in_sp500"]


def garch_features(close):
    """Walk-forward GARCH features for one close-price series; NaN before the burn-in."""
    from arch.univariate import SkewStudent
    from scipy import stats
    from armagarch import Spec, fit

    T = len(close)
    out = np.full((T, len(GARCH_NAMES)), np.nan)
    valid = np.where(~np.isnan(close))[0]
    if len(valid) < BURN_IN + 10:
        return out
    a, b = valid[0], valid[-1] + 1
    y = pd_ffill(100 * np.log(close[a:b]))                 # carry a missing day's price forward
    n = len(y)
    w = np.r_[0.0, np.diff(y)]                              # w[t] = log return of day t (close to close)
    spec = Spec(p=1, d=1, garch=True, dist="skewt")
    X = np.zeros((n, 0))
    u0 = None
    H = np.array(HORIZONS)
    for s in range(BURN_IN, n, BLOCK):
        f = fit(spec, w, X, s, u0=u0)
        u0 = f.u
        e = min(s + BLOCK, n)
        filt = f.filter(w, X)
        P = f.params
        phi, c, nu, lam = P["phi"][0], P["c"], P["nu"], P["lam"]
        sbar2, pers = P["sbar2"], P["persistence"]
        # the forecast made at the open of day t uses returns up to day t-1: filt.mu[t],
        # filt.s2[t] are exactly the one-step forecasts for day t
        rows = np.arange(s, e)
        u_prev = filt.u[rows - 1]
        s2_1 = filt.s2[rows]
        kmax = H.max()
        k = np.arange(kmax)
        mean_path = c + np.outer(u_prev, phi ** (k + 1))                           # (m, kmax)
        var_path = sbar2 + np.outer(s2_1 - sbar2, pers ** k)                       # sigma^2_{t+k}
        cum_psi = np.cumsum(phi ** np.arange(kmax))                                # Psi_0..Psi_{kmax-1}
        cols = []
        for h in HORIZONS:
            mu_h = mean_path[:, :h].sum(axis=1)
            var_h = (var_path[:, :h] * cum_psi[:h][::-1] ** 2).sum(axis=1)
            sd_h = np.sqrt(var_h)
            if h == 1:                          # exact: Hansen's standardised skewed-t
                sk = SkewStudent()              # arch accepts nu <= 300; beyond it the tails are normal
                par = np.array([min(nu, 299.0), lam])
                q05 = mu_h + sd_h * sk.ppf(0.05, par)
                pup = 1.0 - sk.cdf(-mu_h / sd_h, par)
            else:
                q05 = mu_h + sd_h * stats.norm.ppf(0.05)
                pup = stats.norm.cdf(mu_h / sd_h)
            cols += [mu_h, sd_h, q05, pup]
        out[a + rows] = np.column_stack(cols)
    return out


def pd_ffill(x):
    import pandas as pd
    return pd.Series(x).ffill().to_numpy()


GARCH_CACHE = os.path.join(ROOT, "data", "garch_cache")


def _garch_job(args):
    """One stock's walk-forward GARCH block, cached per ticker. The key is the price series
    itself (and the refit schedule), so new or revised prices recompute and a stopped build
    resumes from the stocks it already finished."""
    import hashlib
    i, ticker, close = args
    key = hashlib.sha1(np.ascontiguousarray(close).tobytes() + f"{BLOCK}-{BURN_IN}".encode()).hexdigest()[:16]
    path = os.path.join(GARCH_CACHE, f"{ticker}_{key}.npy")
    if os.path.exists(path):
        return i, np.load(path)
    g = garch_features(close)
    os.makedirs(GARCH_CACHE, exist_ok=True)
    np.save(path + ".tmp.npy", g)
    os.replace(path + ".tmp.npy", path)
    return i, g


def rolling(x, window, fn):
    import pandas as pd
    return getattr(pd.DataFrame(x).rolling(window, min_periods=int(window * 0.8)), fn)().to_numpy()


def price_features(p):
    """(T, N, 7) price block; row t uses closes up to t-1 and the open of t."""
    lc = np.log(p.close)
    r = np.vstack([np.full((1, lc.shape[1]), np.nan), np.diff(lc, axis=0)])          # r[t]: close t-1 -> t
    lag = lambda a, k: np.vstack([np.full((k,) + a.shape[1:], np.nan), a[:-k]])
    spx_lc = np.log(p.spx_close)
    spx_r = np.r_[np.nan, np.diff(spx_lc)]
    gap = np.log(p.open) - lag(lc, 1)
    rev = lag(lc, 1) - lag(lc, 22)
    mom = lag(lc, 22) - lag(lc, 253)
    rv = 100 * lag(rolling(r, 21, "std"), 1)
    # beta: rolling cov / var over 252 days, lagged a day
    import pandas as pd
    R = pd.DataFrame(r)
    S = pd.Series(spx_r)
    cov = R.rolling(252, min_periods=200).cov(S)
    var = S.rolling(252, min_periods=200).var()
    beta = lag(cov.div(var, axis=0).to_numpy(), 1)
    dist = lag(lc - np.log(rolling(p.close, 252, "max")), 1)
    dvol = lag(np.log(rolling(p.close * p.volume, 21, "mean") + 1.0), 1)
    return np.stack([gap, rev, mom, rv, beta, dist, dvol], axis=2), spx_r


def cross_sectional_ranks(F, names, universe):
    """Percentile rank (0-1) of selected columns among the stocks IN THAT DAY'S UNIVERSE:
    a stock outside it gets no rank, and does not shift the ranks of those inside."""
    import pandas as pd
    out = np.full(F.shape[:2] + (len(RANKED),), np.nan)
    for j, nm in enumerate(RANKED):
        col = np.where(universe, F[:, :, names.index(nm)], np.nan)
        out[:, :, j] = pd.DataFrame(col).rank(axis=1, pct=True).to_numpy()
    return out


def spx_full_features(close, n_paths=1000):
    """The S&P 500 through the full model: SARIMA(2,1,2)(0,0,1)[5] + GJR-GARCH-in-mean with
    skewed-t innovations (the "full" arm, BIC order), walk-forward with quarterly refits;
    the h-day distributions come from simulated paths, so they are exact for this model."""
    from armagarch import Spec, arm_spec, difference, fit, fit_multistart, simulate
    spec = arm_spec("full", Spec(**MARKET_ORDER))
    y = pd_ffill(100 * np.log(close))
    n = len(y)
    w = difference(y, spec)
    X = np.zeros((n, 0))
    out = np.full((n, len(GARCH_NAMES)), np.nan)
    u0 = None
    for bi, s in enumerate(range(BURN_IN, n, BLOCK)):
        f = fit_multistart(spec, w, X, s) if u0 is None else fit(spec, w, X, s, u0=u0)
        u0 = f.u
        origins = np.arange(s, min(s + BLOCK, n))
        sims = simulate(f, f.filter(w, X), y, X, np.zeros(0, bool), origins, list(HORIZONS), n_paths, 7 + bi)
        cols = []
        for hi in range(len(HORIZONS)):
            x = sims[:, :, hi]
            cols += [x.mean(1), x.std(1), np.quantile(x, 0.05, axis=1), (x > 0).mean(1)]
        out[origins] = np.column_stack(cols)
    return out


def market_features(p, spx):
    """(T, 23): the S&P 500's full-model forecasts at every horizon, VIX, its gap and trend."""
    import pandas as pd
    lc = np.log(p.spx_close)
    lag1 = lambda a: np.r_[np.nan, a[:-1]]
    gap = np.log(p.spx_open) - lag1(lc)
    dist200 = lag1(lc - np.log(pd.Series(p.spx_close).rolling(200, min_periods=150).mean().to_numpy()))
    return np.column_stack([spx, lag1(p.vix), gap, dist200])


def build(panel=None, workers=WORKERS):
    from stocks_data import load_panel
    p = panel or load_panel()
    T, N = p.close.shape
    t0 = time.time()
    jobs = [(i, p.tickers[i], p.close[:, i]) for i in range(N)]
    G = np.full((T, N, len(GARCH_NAMES)), np.nan)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        spx_job = pool.submit(spx_full_features, p.spx_close)
        for i, g in pool.map(_garch_job, jobs, chunksize=2):
            G[:, i] = g
        spx_g = spx_job.result()
    print(f"GARCH features for {N} stocks + the S&P 500 full model: {time.time() - t0:.0f}s")
    PF, _ = price_features(p)
    F = np.concatenate([G, PF], axis=2)
    F = np.concatenate([F, cross_sectional_ranks(F, GARCH_NAMES + PRICE_NAMES, p.universe),
                        p.sp500[:, :, None].astype(np.float64)], axis=2)
    M = market_features(p, spx_g)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, F=F.astype(np.float32), M=M.astype(np.float32),
                        stock_names=np.array(STOCK_NAMES), market_names=np.array(MARKET_NAMES),
                        tickers=np.array(p.tickers), dates=p.dates.values.astype("datetime64[D]"))
    return F, M


def load(fundamentals=False):
    """(F, M, stock names, market names); with `fundamentals` the EDGAR block of
    fundamentals.py is appended to F (built on the same panel: dates and tickers must match)."""
    z = np.load(CACHE, allow_pickle=False)
    F, sn = z["F"], list(z["stock_names"])
    if fundamentals:
        import fundamentals as fu
        zf = np.load(fu.CACHE, allow_pickle=False)
        if not (np.array_equal(zf["dates"], z["dates"]) and np.array_equal(zf["tickers"], z["tickers"])):
            raise RuntimeError("fundamentals.npz is from another panel: rerun python fundamentals.py")
        F, sn = np.concatenate([F, zf["F"]], axis=2), sn + list(zf["names"])
    return F, z["M"], sn, list(z["market_names"])


if __name__ == "__main__":
    F, M = build()
    print("features:", F.shape, "market:", M.shape,
          "| share of stock-days with GARCH features:", np.mean(~np.isnan(F[:, :, 1])).round(3))
