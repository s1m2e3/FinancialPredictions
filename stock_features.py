"""Walk-forward features for every stock, as known at the OPEN of each trading day.

Row t describes stock i with information up to the close of day t-1 plus the open of
day t (the overnight gap). Nothing in row t uses a later price.

  GARCH block, per horizon h in (1, 3, 5, 10, 21, 63, 126) trading days
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
      simulated paths per day (the 63- and 126-day horizons every SPX_LONG_STEP days, carried
      forward); the mean and P(up) as z-scores against the model's own past year of forecasts
      (spxz_*); VIX, the S&P gap and its distance from the 200-day average; the regime block
      (distance below the 52-week high, 12-month return, 6-month realised volatility)
  Both builds cache per series (data/garch_cache/), so a stopped build resumes.

  --market-only rebuilds just the market block from the cached S&P forecasts.

Run from the repository root:  python stock_features.py   (cached to data/stock_features.npz)
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "stock_features.npz")
# SHORT horizons (days to a month) and LONG ones (3 and 6 months): over a quarter or two the
# GARCH variance has mean-reverted and the mean equation's drift has accumulated, which is
# where the models say something the one-month forecasts do not
SHORT, LONG = (1, 3, 5, 10, 21), (63, 126)
HORIZONS = SHORT + LONG
BLOCK = 63
BURN_IN = 504
WORKERS = 3                  # each worker is a full Python with numpy / pandas / arch: --workers N
# the S&P full model's LONG horizons are simulated at every SPX_LONG_STEP-th day and carried
# forward to the days between (a 3-6 month forecast barely moves within a week, and 126-day
# paths cost six times the 21-day ones); carrying forward only ever uses an earlier forecast
SPX_LONG_STEP = 5

GARCH_NAMES = [f"{k}_{h}d" for h in HORIZONS for k in ("mu", "sig", "q05", "pup")]
PRICE_NAMES = ["gap", "rev_1m", "mom_12_1", "rv_21", "beta_252", "dist_52w_high", "log_dollar_vol"]
# ranks of the mean and direction forecasts too: a refit moves every stock's level, and a
# rank against the same day's universe does not care (the raw mu / pup stay out, see
# portfolio_bt.DROPPED)
RANKED = ["mu_21d", "mu_63d", "mu_126d", "sig_21d", "sig_126d", "pup_5d", "pup_21d", "pup_63d",
          "gap", "rev_1m", "mom_12_1", "rv_21", "beta_252", "dist_52w_high"]
RANK_NAMES = [f"rank_{n}" for n in RANKED]
# REGIME: what the market has been doing for MONTHS, which the GARCH forecasts (a memory of
# a few weeks) and VIX (today) do not say: how far below its 52-week high the S&P is, its
# 12-month return and its realised volatility over 6 months. 2022 fell for ten months with
# no volatility spike; these are the inputs that describe it.
REGIME_NAMES = ["spx_dd_52w", "spx_ret_12m", "spx_rv_126"]
# the S&P model's mean and P(up) as a z-score against its own forecasts of the past year:
# the level drifts with every refit (why the raw ones were calendars), the deviation from
# what the model has been saying lately does not -- "unusually bullish / bearish right now"
SPXZ_H = (5, 21, 63, 126)
SPXZ_NAMES = [f"spxz_{k}_{h}d" for h in SPXZ_H for k in ("mu", "pup")]
MARKET_NAMES = ([f"spx_{n}" for n in GARCH_NAMES] + ["vix", "spx_gap", "spx_dist_200d"]
                + REGIME_NAMES + SPXZ_NAMES)
MARKET_ORDER = dict(p=2, d=1, q=2, Q=1, s=5)   # the BIC order identified on the Nasdaq Composite
# YOUNG STOCKS. The walk-forward GARCH needs BURN_IN (504) days, so a stock listed two years
# ago was invisible -- the fast growers this universe exists to find, while they are new. From
# its YOUNG_MIN_DAYS-th day until GARCH takes over, a stock's risk forecasts come from an
# EWMA of its squared returns (RiskMetrics, lambda 0.94, returns up to the previous close):
# sig_h = sigma sqrt(h), q05_h = -1.645 sig_h; its mean and P(up) stay empty (no mean model).
# `young` is 1 on those days, so a tree knows which estimate it is reading.
YOUNG_MIN_DAYS, YOUNG_LAMBDA = 63, 0.94
STOCK_NAMES = GARCH_NAMES + PRICE_NAMES + RANK_NAMES + ["in_sp500", "young"]


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


def young_stock_risk(close, G):
    """Fill the sig_h / q05_h columns of G where GARCH has no forecast yet but the stock has
    YOUNG_MIN_DAYS of returns (see YOUNG_MIN_DAYS); returns the (T, N) mask of filled days."""
    import pandas as pd
    lc = 100 * np.log(pd.DataFrame(close).ffill())
    r = lc.diff()
    # the variance forecast for day t from returns up to day t-1, as GARCH's filt.s2[t] is
    var = (r ** 2).ewm(alpha=1 - YOUNG_LAMBDA, min_periods=YOUNG_MIN_DAYS).mean().shift(1).to_numpy()
    sig1 = np.sqrt(var)
    young = np.isnan(G[:, :, GARCH_NAMES.index("sig_21d")]) & np.isfinite(sig1) & ~np.isnan(close)
    for h in HORIZONS:
        s = sig1 * np.sqrt(h)
        for name, val in ((f"sig_{h}d", s), (f"q05_{h}d", -1.645 * s)):
            j = GARCH_NAMES.index(name)
            G[:, :, j] = np.where(young, val, G[:, :, j])
    return young


def pd_ffill(x):
    import pandas as pd
    return pd.Series(x).ffill().to_numpy()


GARCH_CACHE = os.path.join(ROOT, "data", "garch_cache")


def _garch_path(ticker, close):
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(close).tobytes()
                       + f"{BLOCK}-{BURN_IN}-{HORIZONS}".encode()).hexdigest()[:16]
    return os.path.join(GARCH_CACHE, f"{ticker}_{key}.npy")


def _garch_job(args):
    """One stock's walk-forward GARCH block, cached per ticker. The key is the price series
    itself (and the refit schedule), so new or revised prices recompute and a stopped build
    resumes from the stocks it already finished. Returns (i, block, seconds, from cache)."""
    i, ticker, close = args
    t0 = time.time()
    path = _garch_path(ticker, close)
    if os.path.exists(path):
        return i, np.load(path), time.time() - t0, True
    g = garch_features(close)
    os.makedirs(GARCH_CACHE, exist_ok=True)
    np.save(path + ".tmp.npy", g)
    os.replace(path + ".tmp.npy", path)
    return i, g, time.time() - t0, False


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
    out = np.full(F.shape[:2] + (len(RANKED),), np.nan, dtype=np.float32)
    for j, nm in enumerate(RANKED):
        col = np.where(universe, F[:, :, names.index(nm)], np.nan)
        out[:, :, j] = pd.DataFrame(col).rank(axis=1, pct=True).to_numpy()
    return out


def spx_full_features(close, horizons=SHORT, step=1, n_paths=1000):
    """The S&P 500 through the full model: SARIMA(2,1,2)(0,0,1)[5] + GJR-GARCH-in-mean with
    skewed-t innovations (the "full" arm, BIC order), walk-forward with quarterly refits;
    the h-day distributions come from simulated paths, so they are exact for this model.
    (n, 4 x len(horizons)): mean, s.d., 5% quantile, P(up) per horizon. With step > 1 only
    every step-th day is simulated and each forecast is carried forward to the next one."""
    import pandas as pd
    from armagarch import Spec, arm_spec, difference, fit, fit_multistart, simulate
    spec = arm_spec("full", Spec(**MARKET_ORDER))
    y = pd_ffill(100 * np.log(close))
    n = len(y)
    w = difference(y, spec)
    X = np.zeros((n, 0))
    out = np.full((n, 4 * len(horizons)), np.nan)
    u0 = None
    for bi, s in enumerate(range(BURN_IN, n, BLOCK)):
        f = fit_multistart(spec, w, X, s) if u0 is None else fit(spec, w, X, s, u0=u0)
        u0 = f.u
        origins = np.arange(s, min(s + BLOCK, n))
        origins = origins[origins % step == 0]
        if not len(origins):
            continue
        sims = simulate(f, f.filter(w, X), y, X, np.zeros(0, bool), origins, list(horizons), n_paths, 7 + bi)
        cols = []
        for hi in range(len(horizons)):
            x = sims[:, :, hi]
            cols += [x.mean(1), x.std(1), np.quantile(x, 0.05, axis=1), (x > 0).mean(1)]
        out[origins] = np.column_stack(cols)
    if step > 1:
        first = BURN_IN + (-BURN_IN) % step
        out[first:] = pd.DataFrame(out[first:]).ffill().to_numpy()   # only earlier forecasts
    return out


def _spx_job(args):
    """The S&P block for `horizons`, cached like the stocks' (keyed by the price series)."""
    import hashlib
    close, horizons, step = args
    key = hashlib.sha1(np.ascontiguousarray(close).tobytes()
                       + f"{BLOCK}-{BURN_IN}-{horizons}-{step}".encode()).hexdigest()[:16]
    path = os.path.join(GARCH_CACHE, f"SPX_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    out = spx_full_features(close, horizons, step)
    os.makedirs(GARCH_CACHE, exist_ok=True)
    np.save(path + ".tmp.npy", out)
    os.replace(path + ".tmp.npy", path)
    return out


def _seed_spx_cache(p):
    """The short-horizon S&P block of the cached features, filed where _spx_job looks for
    it: those forecasts were simulated for this very price series, so they are reused."""
    import hashlib
    if not os.path.exists(CACHE):
        return
    with np.load(CACHE, allow_pickle=False) as z:
        if not np.array_equal(z["dates"], p.dates.values.astype("datetime64[D]")):
            return
        mn = list(z["market_names"])
        want = [f"spx_{k}_{h}d" for h in SHORT for k in ("mu", "sig", "q05", "pup")]
        if not all(c in mn for c in want):
            return
        block = z["M"][:, [mn.index(c) for c in want]].astype(np.float64)
    key = hashlib.sha1(np.ascontiguousarray(p.spx_close).tobytes()
                       + f"{BLOCK}-{BURN_IN}-{SHORT}-1".encode()).hexdigest()[:16]
    path = os.path.join(GARCH_CACHE, f"SPX_{key}.npy")
    if not os.path.exists(path):
        os.makedirs(GARCH_CACHE, exist_ok=True)
        np.save(path, block)
        print("reused the cached short-horizon S&P forecasts", flush=True)


def market_features(p, spx):
    """(T, 26): the S&P 500's full-model forecasts at every horizon, VIX, its gap and trend,
    and the regime block (REGIME_NAMES). Every column uses closes up to day t-1."""
    import pandas as pd
    lc = np.log(p.spx_close)
    lag1 = lambda a: np.r_[np.nan, a[:-1]]
    close = pd.Series(p.spx_close)
    gap = np.log(p.spx_open) - lag1(lc)
    dist200 = lag1(lc - np.log(close.rolling(200, min_periods=150).mean().to_numpy()))
    dd52 = lag1(p.spx_close / close.rolling(252, min_periods=200).max().to_numpy() - 1.0)
    ret12 = lag1(lc - np.r_[np.full(252, np.nan), lc[:-252]])
    rv126 = lag1(100 * np.sqrt(252) * pd.Series(np.r_[np.nan, np.diff(lc)]).rolling(126, min_periods=100).std().to_numpy())
    # row t of the model block is already the forecast made at t's open, so a trailing window
    # that ends at t uses no later forecast
    zs = []
    for h in SPXZ_H:
        for k in ("mu", "pup"):
            x = pd.Series(spx[:, GARCH_NAMES.index(f"{k}_{h}d")])
            m, s = x.rolling(252, min_periods=126).mean(), x.rolling(252, min_periods=126).std()
            zs.append(((x - m) / s.where(s > 0)).to_numpy())
    return np.column_stack([spx, lag1(p.vix), gap, dist200, dd52, ret12, rv126] + zs)


def rebuild_market():
    """Recompute only the market block M of the cached features (the regime columns are
    new; the S&P full-model forecasts are reused, so nothing is re-simulated)."""
    from stocks_data import load_panel
    p = load_panel()
    z = dict(np.load(CACHE, allow_pickle=False))
    if not np.array_equal(z["dates"], p.dates.values.astype("datetime64[D]")):
        raise RuntimeError("stock_features.npz is from another panel: run python stock_features.py")
    want = [f"spx_{n}" for n in GARCH_NAMES]
    if list(z["market_names"][:len(want)]) != want:
        raise RuntimeError("the cached market block has other horizons: run python stock_features.py")
    spx = z["M"][:, :len(GARCH_NAMES)]
    z["M"] = market_features(p, spx.astype(np.float64)).astype(np.float32)
    z["market_names"] = np.array(MARKET_NAMES)
    np.savez_compressed(CACHE, **z)
    return z["M"]


def build(panel=None, workers=WORKERS):
    from stocks_data import load_panel
    print(f"[{time.strftime('%H:%M:%S')}] loading the panel", flush=True)
    p = panel or load_panel()
    T, N = p.close.shape
    t0 = time.time()
    jobs = [(i, p.tickers[i], p.close[:, i]) for i in range(N)]
    # float32, as it is saved: the (days x stocks x 28) block is the build's largest array
    G = np.full((T, N, len(GARCH_NAMES)), np.nan, dtype=np.float32)
    stamp = lambda: f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:5.0f}s]"
    n_cached = sum(os.path.exists(_garch_path(t, c)) for _, t, c in jobs)
    print(f"{stamp()} panel {T} days x {N} stocks; GARCH for {N} stocks ({n_cached} already cached, "
          f"{N - n_cached} to fit) on {workers} workers, and the S&P 500 full model", flush=True)
    _seed_spx_cache(p)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        spx_jobs = {pool.submit(_spx_job, (p.spx_close, SHORT, 1)): "S&P short horizons (1-21 days)",
                    pool.submit(_spx_job, (p.spx_close, LONG, SPX_LONG_STEP)): "S&P long horizons (63, 126 days)"}
        for fut, label in spx_jobs.items():
            fut.add_done_callback(lambda f, label=label: print(
                f"{stamp()} {label}: done" + ("" if f.exception() is None else f" -- FAILED: {f.exception()}"),
                flush=True))
        futs = {pool.submit(_garch_job, j): j for j in jobs}
        fitted, fit_s, k = 0, 0.0, 0
        failed = []
        for fut in as_completed(futs):
            try:
                i, g, sec, cached = fut.result()
            except Exception as exc:        # a worker that died (e.g. out of memory): retried below
                failed.append(futs[fut])
                print(f"{stamp()} {futs[fut][1]:6s} FAILED in its worker ({type(exc).__name__}: {exc}); "
                      f"will retry", flush=True)
                continue
            k += 1
            G[:, i] = g
            if not cached:
                fitted, fit_s = fitted + 1, fit_s + sec
            left = N - k
            # time left from the stocks fitted so far (cached ones cost nothing), per worker
            eta = (fit_s / fitted) * min(left, N - n_cached) / workers if fitted else 0.0
            print(f"{stamp()} [{k:3d}/{N}] {p.tickers[i]:6s} {'cached' if cached else f'fitted in {sec:4.0f}s'}"
                  + (f"   ~{eta / 60:.0f} min left" if fitted and left else ""), flush=True)
        spx_out = {}
        for fut, label in spx_jobs.items():
            try:
                spx_out[label] = fut.result()
            except Exception as exc:
                print(f"{stamp()} {label} FAILED in its worker ({type(exc).__name__}); will retry", flush=True)
    # RETRIES: first in a fresh pool (a broken worker never comes back), then one by one here
    for attempt in ("a fresh pool", "the main process"):
        if not failed:
            break
        print(f"{stamp()} retrying {len(failed)} stocks in {attempt}", flush=True)
        again = []
        if attempt == "a fresh pool":
            with ProcessPoolExecutor(max_workers=max(1, workers - 1)) as pool:
                futs = {pool.submit(_garch_job, j): j for j in failed}
                for fut in as_completed(futs):
                    try:
                        i, g, sec, cached = fut.result()
                        G[:, i] = g
                        print(f"{stamp()} {p.tickers[i]:6s} done on retry ({sec:.0f}s)", flush=True)
                    except Exception:
                        again.append(futs[fut])
        else:
            for j in failed:
                i, g, sec, cached = _garch_job(j)
                G[:, i] = g
                print(f"{stamp()} {p.tickers[i]:6s} done in the main process ({sec:.0f}s)", flush=True)
        failed = again
    labels = {"S&P short horizons (1-21 days)": (SHORT, 1), "S&P long horizons (63, 126 days)": (LONG, SPX_LONG_STEP)}
    for label, (h, step) in labels.items():
        if label not in spx_out:
            print(f"{stamp()} {label}: running in the main process", flush=True)
            spx_out[label] = _spx_job((p.spx_close, h, step))
    print(f"{stamp()} all stocks done", flush=True)
    spx_g = np.concatenate([spx_out[l] for l in labels], axis=1)
    print(f"{stamp()} GARCH features for {N} stocks + the S&P 500 full model done", flush=True)
    young = young_stock_risk(p.close, G)
    print(f"{stamp()} young-stock risk filled on {young.sum()} stock-days "
          f"({(young & p.universe).sum()} of them in the universe)", flush=True)
    PF, _ = price_features(p)
    print(f"{stamp()} price features done", flush=True)
    F = np.concatenate([G, PF.astype(np.float32)], axis=2)       # float32 throughout (memory)
    del G, PF
    F = np.concatenate([F, cross_sectional_ranks(F, GARCH_NAMES + PRICE_NAMES, p.universe).astype(np.float32),
                        p.sp500[:, :, None].astype(np.float32), young[:, :, None].astype(np.float32)], axis=2)
    print(f"{stamp()} cross-sectional ranks done", flush=True)
    M = market_features(p, spx_g)
    print(f"{stamp()} market block done ({M.shape[1]} columns); saving {CACHE}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, F=F.astype(np.float32), M=M.astype(np.float32),
                        stock_names=np.array(STOCK_NAMES), market_names=np.array(MARKET_NAMES),
                        tickers=np.array(p.tickers), dates=p.dates.values.astype("datetime64[D]"))
    print(f"{stamp()} saved", flush=True)
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
    import sys
    if "--market-only" in sys.argv:
        M = rebuild_market()
        print("market block rebuilt:", M.shape, MARKET_NAMES[-len(REGIME_NAMES):])
        raise SystemExit
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else WORKERS
    F, M = build(workers=workers)
    print("features:", F.shape, "market:", M.shape,
          "| share of stock-days with GARCH features:", np.mean(~np.isnan(F[:, :, 1])).round(3))
