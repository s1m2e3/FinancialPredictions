"""SARIMAX mean + GJR-GARCH(-in-mean) variance with normal, Student-t or skewed-t errors.

Notation: y_t = 100 * log P_t, B the backshift operator.

  differencing   w_t = (1 - B)^d (1 - B^s)^D y_t
  regression     w_t = c + beta' x_t + delta * sigma_t + u_t          (delta: GARCH-in-mean)
  SARMA errors   phi(B) Phi(B^s) u_t = theta(B) Theta(B^s) eps_t
  innovations    eps_t = sigma_t z_t,  z_t iid with mean 0 and variance 1
  variance       sigma_t^2 = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2 + b sigma_{t-1}^2

The conditional mean of w_t is mu_t = c + beta' x_t + delta sigma_t + [SARMA part]. The
regression-with-SARMA-errors form is the one statsmodels' SARIMAX uses. z_t is standard
normal, standardised Student-t, or Hansen's (1994) standardised skewed-t with tail nu and
skew lambda in (-1, 1) (lambda < 0: longer left tail). With garch=False the variance is a
constant sigma^2 and the model is a plain (S)ARIMAX.

Mean and variance are coupled three ways: the variance recursion runs on the mean model's
residuals eps_t; the likelihood divides each squared residual by sigma_t^2; and with
in_mean the variance enters the mean directly.

Parameters are optimised unconstrained. The transforms keep the AR and seasonal-AR
polynomials stationary, the MA and seasonal-MA polynomials invertible, omega, alpha,
gamma, b > 0 with alpha + gamma/2 + b < 1, nu > 2.05 and |lambda| < 1.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit

LOG_2PI = np.log(2 * np.pi)
MAX_PERSISTENCE = 0.999
DIST_CODE = {"normal": 0, "t": 1, "skewt": 2}


@dataclass(frozen=True)
class Spec:
    p: int = 0
    d: int = 1
    q: int = 0
    P: int = 0
    D: int = 0
    Q: int = 0
    s: int = 1
    k: int = 0               # number of regressors in x_t
    garch: bool = False      # False: constant variance
    dist: str = "normal"     # "normal", "t" or "skewt"
    in_mean: bool = False    # GARCH-in-mean term delta * sigma_t

    def __post_init__(self):
        if (self.P or self.Q or self.D) and self.s < 2:
            raise ValueError("seasonal terms need a season length s >= 2")
        if self.in_mean and not self.garch:
            raise ValueError("in_mean needs a GARCH variance")

    @property
    def n_diff(self):
        """Observations lost to differencing; w_t exists from this index on."""
        return self.d + self.s * self.D

    @property
    def ar_order(self):
        return self.p + self.s * self.P

    @property
    def ma_order(self):
        return self.q + self.s * self.Q

    @property
    def t0(self):
        """Default first index of the likelihood: every lag of w exists from here on."""
        return self.n_diff + max(self.ar_order, self.ma_order, 1)

    def blocks(self):
        return [("c", 1), ("beta", self.k), ("delta", int(self.in_mean)),
                ("ar", self.p), ("sar", self.P), ("ma", self.q), ("sma", self.Q),
                ("logvar", 1), ("garch", 3 if self.garch else 0),
                ("nu", int(self.dist in ("t", "skewt"))), ("skew", int(self.dist == "skewt"))]

    def slices(self):
        out, i = {}, 0
        for name, size in self.blocks():
            out[name] = slice(i, i + size)
            i += size
        return out

    @property
    def n_params(self):
        return sum(size for _, size in self.blocks())

    def mean_mask(self):
        mask = np.zeros(self.n_params, dtype=bool)
        for name, sl in self.slices().items():
            if name in ("c", "beta", "delta", "ar", "sar", "ma", "sma"):
                mask[sl] = True
        return mask

    def label(self):
        seasonal = f"({self.P},{self.D},{self.Q})[{self.s}]" if (self.P or self.D or self.Q) else ""
        return f"SARIMA({self.p},{self.d},{self.q}){seasonal}"


# ------------------------------------------------------------------ polynomials
def diff_poly(d, D, s):
    """Coefficients delta_0..delta_L of (1 - B)^d (1 - B^s)^D, delta_0 = 1."""
    poly = np.array([1.0])
    for _ in range(d):
        poly = np.convolve(poly, [1.0, -1.0])
    seasonal = np.zeros(s + 1)
    seasonal[0], seasonal[-1] = 1.0, -1.0
    for _ in range(D):
        poly = np.convolve(poly, seasonal)
    return poly


def difference(y, spec):
    """w_t = (1 - B)^d (1 - B^s)^D y_t; the first n_diff entries are set to 0 and unused."""
    poly = diff_poly(spec.d, spec.D, spec.s)
    w = np.convolve(y, poly)[:len(y)]
    w[:spec.n_diff] = 0.0
    return w


def _pacf_to_coef(u):
    """Unconstrained values -> coefficients of a stationary AR polynomial.

    tanh(u) are partial autocorrelations in (-1, 1); the Durbin-Levinson recursion turns
    them into phi with every root of 1 - sum phi_j z^j outside the unit circle.
    """
    phi = np.zeros(0)
    for rk in np.tanh(u):
        phi = np.append(phi - rk * phi[::-1], rk)
    return phi


def _seasonal(coef, s):
    """Coefficients at lags s, 2s, ... laid out on a dense lag axis 1..len(coef)*s."""
    out = np.zeros(len(coef) * s)
    out[s - 1::s] = coef
    return out


def _ar_expand(phi, Phi, s):
    """a_k with phi(B) Phi(B^s) = 1 - sum_k a_k B^k."""
    poly = np.convolve(np.r_[1.0, -phi], np.r_[1.0, -_seasonal(Phi, s)])
    return -poly[1:]


def _ma_expand(theta, Theta, s):
    """b_k with theta(B) Theta(B^s) = 1 + sum_k b_k B^k."""
    poly = np.convolve(np.r_[1.0, theta], np.r_[1.0, _seasonal(Theta, s)])
    return poly[1:]


def skewt_constants(nu, lam):
    """Hansen's a, b, c for the standardised skewed-t."""
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


# ------------------------------------------------------------------ parameters
def unpack(u, spec):
    """Unconstrained parameter vector -> named model parameters."""
    sl = spec.slices()
    s = max(spec.s, 1)
    phi, Phi = _pacf_to_coef(u[sl["ar"]]), _pacf_to_coef(u[sl["sar"]])
    theta, Theta = -_pacf_to_coef(u[sl["ma"]]), -_pacf_to_coef(u[sl["sma"]])
    sbar2 = np.exp(u[sl["logvar"]][0])

    if spec.garch:
        g = u[sl["garch"]]
        persistence = MAX_PERSISTENCE * expit(g[0])
        wts = np.exp([g[1], g[2], 0.0])
        wts /= wts.sum()                           # shares of alpha, gamma/2, b in the persistence
        alpha, gamma, b = persistence * wts[0], 2 * persistence * wts[1], persistence * wts[2]
        omega = sbar2 * (1 - persistence)
    else:
        persistence, alpha, gamma, b, omega = 0.0, 0.0, 0.0, 0.0, sbar2

    nu = 2.05 + np.exp(u[sl["nu"]][0]) if spec.dist in ("t", "skewt") else np.inf
    lam = np.tanh(u[sl["skew"]][0]) if spec.dist == "skewt" else 0.0
    return dict(c=u[sl["c"]][0], beta=u[sl["beta"]].copy(),
                delta=u[sl["delta"]][0] if spec.in_mean else 0.0,
                phi=phi, Phi=Phi, theta=theta, Theta=Theta,
                ar=_ar_expand(phi, Phi, s), ma=_ma_expand(theta, Theta, s),
                sbar2=sbar2, omega=omega, alpha=alpha, gamma=gamma, b=b,
                persistence=persistence, nu=nu, lam=lam)


def initial_params(spec, w_train):
    u = np.zeros(spec.n_params)
    sl = spec.slices()
    u[sl["c"]] = w_train.mean()
    u[sl["logvar"]] = np.log(w_train.var())
    if spec.garch:
        # persistence 0.95 split as alpha 0.03, gamma/2 0.04, b 0.88
        u[sl["garch"]] = [logit(0.95 / MAX_PERSISTENCE), np.log(0.03 / 0.88), np.log(0.04 / 0.88)]
    if spec.dist in ("t", "skewt"):
        u[sl["nu"]] = np.log(8.0 - 2.05)
    return u


def embed(u_from, spec_from, spec_to, w_train):
    """Start spec_to from spec_from's values wherever the two share a parameter block.

    New blocks start at zero skew, zero in-mean effect and the default variance
    parameters, so a richer spec started this way reproduces the simpler fit.
    """
    u = initial_params(spec_to, w_train)
    src, dst = spec_from.slices(), spec_to.slices()
    for name, sl in dst.items():
        if name in src and (sl.stop - sl.start) == (src[name].stop - src[name].start):
            u[sl] = u_from[src[name]]
    return u


# ------------------------------------------------------------------ filter
@njit(cache=True)
def _filter(w, reg, ar, ma, garch, in_mean, delta, sbar2, omega, alpha, gamma, b, start, t0):
    """Mean and variance recursions over the whole series.

    mu[t] and s2[t] only use information up to t-1 (reg[t] only holds regressors known
    before day t), so past the fitting window they are genuine one-step-ahead forecasts.
    Pre-sample residuals are zero (conditional likelihood); the likelihood starts at t0.
    """
    n = w.shape[0]
    mu = np.zeros(n)
    u = np.zeros(n)
    eps = np.zeros(n)
    s2 = np.full(n, sbar2)
    for t in range(start, n):
        if garch and t > t0:
            e = eps[t - 1]
            a = alpha + gamma if e < 0.0 else alpha
            s2[t] = omega + a * e * e + b * s2[t - 1]
        m = reg[t] + (delta * np.sqrt(s2[t]) if in_mean else 0.0)
        arma = 0.0
        for i in range(ar.shape[0]):
            j = t - 1 - i
            if j >= start:
                arma += ar[i] * u[j]
        for i in range(ma.shape[0]):
            j = t - 1 - i
            if j >= start:
                arma += ma[i] * eps[j]
        mu[t] = m + arma
        u[t] = w[t] - m
        if t >= t0:
            eps[t] = w[t] - mu[t]
    return mu, u, eps, s2


def log_density(z, dist, nu, lam):
    """log f(z) for the standardised (mean 0, variance 1) innovation distribution."""
    if dist == "normal":
        return -0.5 * (LOG_2PI + z ** 2)
    if dist == "t":
        return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
                - (nu + 1) / 2 * np.log1p(z ** 2 / (nu - 2)))
    a, b, c = skewt_constants(nu, lam)
    v = b * z + a
    scale = np.where(v < 0, 1 - lam, 1 + lam)
    return np.log(b) + np.log(c) - (nu + 1) / 2 * np.log1p((v / scale) ** 2 / (nu - 2))


@dataclass
class Filtered:
    ll: np.ndarray    # per-observation log-likelihood
    mu: np.ndarray    # one-step-ahead conditional mean of w_t
    s2: np.ndarray    # one-step-ahead conditional variance
    u: np.ndarray     # regression residuals (SARMA process)
    eps: np.ndarray   # innovations
    P: dict           # the parameters used


def filter_series(u, spec, w, X, t0=None):
    P = unpack(u, spec)
    t0 = spec.t0 if t0 is None else t0
    reg = P["c"] + (X[:, :spec.k] @ P["beta"] if spec.k else 0.0) + np.zeros(len(w))
    mu, uu, eps, s2 = _filter(w, reg, P["ar"], P["ma"], spec.garch, spec.in_mean, P["delta"],
                              P["sbar2"], P["omega"], P["alpha"], P["gamma"], P["b"], spec.n_diff, t0)
    ll = log_density(eps / np.sqrt(s2), spec.dist, P["nu"], P["lam"]) - 0.5 * np.log(s2)
    return Filtered(ll=ll, mu=mu, s2=s2, u=uu, eps=eps, P=P)


# ------------------------------------------------------------------ estimation
@dataclass
class Fit:
    spec: Spec
    u: np.ndarray
    t0: int                                      # likelihood over w[t0:t_end]
    t_end: int
    history: dict = field(default_factory=dict)  # per-iteration mean NLL: "train", "val"
    message: str = ""

    @property
    def params(self):
        return unpack(self.u, self.spec)

    @property
    def train_nll(self):
        return self.history["train"][-1]

    @property
    def n_obs(self):
        return self.t_end - self.t0

    def filter(self, w, X):
        return filter_series(self.u, self.spec, w, X, self.t0)

    def info_criteria(self):
        n, k = self.n_obs, self.spec.n_params
        aic = 2 * n * self.train_nll + 2 * k
        return {"AIC": aic, "AICc": aic + 2 * k * (k + 1) / (n - k - 1), "BIC": 2 * n * self.train_nll + k * np.log(n)}


def fit(spec, w, X, t_end, t0=None, u0=None, fixed=None, val_end=None, maxiter=3000):
    """Minimise the mean negative log-likelihood over w[t0:t_end] with L-BFGS.

    fixed: boolean mask of parameters held at their u0 value (the two-step arm).
    val_end: also record the NLL on w[t_end:val_end] at every iteration (monitoring only;
    nothing after t_end enters the objective).
    t0: common likelihood start, so that models with different lag orders are compared
    on the same observations.
    """
    t0 = spec.t0 if t0 is None else t0
    u0 = initial_params(spec, w[t0:t_end]) if u0 is None else np.asarray(u0, float).copy()
    free = np.ones(spec.n_params, bool) if fixed is None else ~fixed

    def full(z):
        u = u0.copy()
        u[free] = z
        return u

    def objective(z):
        v = -filter_series(full(z), spec, w, X, t0).ll[t0:t_end].mean()
        return v if np.isfinite(v) else 1e10

    history = {"train": [], "val": []}

    def record(z):
        ll = filter_series(full(z), spec, w, X, t0).ll
        history["train"].append(-ll[t0:t_end].mean())
        if val_end is not None:
            history["val"].append(-ll[t_end:val_end].mean())

    z0 = u0[free]
    record(z0)
    res = minimize(objective, z0, method="L-BFGS-B", callback=record,
                   options=dict(maxiter=maxiter, maxfun=200000, gtol=1e-7))
    if not np.isclose(history["train"][-1], res.fun):
        record(res.x)
    return Fit(spec=spec, u=full(res.x), t0=t0, t_end=t_end, history=history, message=res.message)


def mean_starts(spec, w_train, pacf=0.8):
    """Starting points for the (S)ARMA coefficients.

    ARMA likelihoods have a ridge where an AR factor nearly cancels an MA factor
    (phi ~ -theta); started at zero the optimiser can stop on the ridge's low side. Besides
    the zero start, each AR/MA pair is also started at (pacf, -pacf) and (-pacf, pacf).
    """
    base = initial_params(spec, w_train)
    sl = spec.slices()
    a = np.arctanh(pacf)
    starts = [base]
    for ar_name, ma_name in (("sar", "sma"), ("ar", "ma")):
        if (sl[ar_name].stop - sl[ar_name].start) + (sl[ma_name].stop - sl[ma_name].start) == 0:
            continue
        for sign in (1.0, -1.0):
            u = base.copy()
            for name in (ar_name, ma_name):       # same sign in u-space: phi = tanh, theta = -tanh
                if sl[name].stop > sl[name].start:
                    u[sl[name].start] = sign * a
            starts.append(u)
    return starts


def fit_multistart(spec, w, X, t_end, t0=None, val_end=None):
    """fit() from every mean_starts() point; returns the run with the lowest training NLL."""
    t0 = spec.t0 if t0 is None else t0
    runs = [fit(spec, w, X, t_end, t0=t0, u0=u0, val_end=val_end) for u0 in mean_starts(spec, w[t0:t_end])]
    return min(runs, key=lambda f: f.train_nll)


# ------------------------------------------------------------------ the comparison ladder
# Every arm shares the same SARIMAX mean equation; each rung adds one thing.
#   sarimax      constant variance, Gaussian              -- the classic SARIMAX
#   sarimax_t    constant variance, Student-t             -- fat tails only
#   two_step     SARIMAX mean frozen, GJR-GARCH-t fitted on its residuals
#   joint        SARIMAX + GJR-GARCH-t, all estimated together (coupling)
#   joint_skewt  + skewed-t innovations
#   full         + GARCH-in-mean
ARMS = ["sarimax", "sarimax_t", "two_step", "joint", "joint_skewt", "full"]
PREVIOUS = {"sarimax_t": "sarimax", "joint": "two_step", "joint_skewt": "joint", "full": "joint_skewt"}


def arm_spec(arm, base):
    kw = dict(p=base.p, d=base.d, q=base.q, P=base.P, D=base.D, Q=base.Q, s=base.s, k=base.k)
    return {
        "sarimax": Spec(**kw, garch=False, dist="normal"),
        "sarimax_t": Spec(**kw, garch=False, dist="t"),
        "two_step": Spec(**kw, garch=True, dist="t"),
        "joint": Spec(**kw, garch=True, dist="t"),
        "joint_skewt": Spec(**kw, garch=True, dist="skewt"),
        "full": Spec(**kw, garch=True, dist="skewt", in_mean=True),
    }[arm]


def fit_arms(base, w, X, t_end, val_end=None, warm=None):
    """Fit every arm on w[:t_end].

    Each arm except the two-step one is started twice: from its own start (the previous
    window's fit when warm is given) and from the rung below it, which is nested in it, so
    the richer model can never end with a worse training likelihood. The better run is
    kept; both loss curves are stored.
    """
    specs = {arm: arm_spec(arm, base) for arm in ARMS}
    t0 = specs["sarimax"].t0
    w_train = w[t0:t_end]
    fits = {}
    for arm in ARMS:
        s = specs[arm]
        own = warm[arm].u if warm else None
        if arm == "sarimax":
            fits[arm] = (fit(s, w, X, t_end, u0=own, val_end=val_end) if warm
                         else fit_multistart(s, w, X, t_end, val_end=val_end))
            continue
        if arm == "two_step":
            u0 = embed(fits["sarimax"].u, specs["sarimax"], s, w_train)   # mean from SARIMAX
            if warm:
                u0[~s.mean_mask()] = own[~s.mean_mask()]                    # variance from last window
            fits[arm] = fit(s, w, X, t_end, u0=u0, fixed=s.mean_mask(), val_end=val_end)
            fits[arm].history["runs"] = {"": dict(fits[arm].history)}
            continue
        prev = PREVIOUS[arm]
        runs = {"own start": fit(s, w, X, t_end, u0=own, val_end=val_end),
                f"{prev} start": fit(s, w, X, t_end, u0=embed(fits[prev].u, specs[prev], s, w_train),
                                     val_end=val_end)}
        kept = min(runs, key=lambda name: runs[name].train_nll)
        fits[arm] = runs[kept]
        fits[arm].history = {"train": runs[kept].history["train"], "val": runs[kept].history["val"],
                             "kept": kept, "runs": {name: dict(f.history) for name, f in runs.items()}}
    fits["sarimax"].history["runs"] = {"": dict(fits["sarimax"].history)}
    return fits


# ------------------------------------------------------------------ multi-horizon simulation
@njit(cache=True)
def _simulate(origins, horizon_k, H, M, seed, y, u, eps, s2, reg_paths, ar, ma, diffc,
              garch, in_mean, delta, sbar2, omega, alpha, gamma, b, dist, nu, lam, ska, skb):
    """Simulate M paths of H days ahead from each origin t (information up to day t-1).

    Returns the cumulative change y_{t+h-1} - y_{t-1} at each requested horizon, i.e. the
    h-day log return in %, shape (n_origins, M, n_horizons).
    """
    La, Lm, Ld = ar.shape[0], ma.shape[0], diffc.shape[0]
    n_h = horizon_k.shape[0]
    out = np.empty((origins.shape[0], M, n_h))
    ub = np.empty(La + H)
    eb = np.empty(Lm + H)
    yb = np.empty(Ld + H)
    tscale = np.sqrt((nu - 2.0) / nu) if dist > 0 else 1.0
    np.random.seed(seed)
    for o in range(origins.shape[0]):
        t = origins[o]
        for path in range(M):
            for i in range(La):
                ub[i] = u[t - La + i]
            for i in range(Lm):
                eb[i] = eps[t - Lm + i]
            for i in range(Ld):
                yb[i] = y[t - Ld + i]
            e_prev = eps[t - 1]
            sig2 = s2[t] if garch else sbar2
            hi = 0
            for k in range(H):
                if garch and k > 0:
                    a = alpha + gamma if e_prev < 0.0 else alpha
                    sig2 = omega + a * e_prev * e_prev + b * sig2
                m = reg_paths[o, k] + (delta * np.sqrt(sig2) if in_mean else 0.0)
                arma = 0.0
                for i in range(La):
                    arma += ar[i] * ub[La + k - 1 - i]
                for i in range(Lm):
                    arma += ma[i] * eb[Lm + k - 1 - i]
                if dist == 0:
                    z = np.random.standard_normal()
                elif dist == 1:
                    z = np.random.standard_t(nu) * tscale
                else:
                    S = abs(np.random.standard_t(nu)) * tscale
                    v = -(1.0 - lam) * S if np.random.random() < (1.0 - lam) / 2.0 else (1.0 + lam) * S
                    z = (v - ska) / skb
                e = np.sqrt(sig2) * z
                wv = m + arma + e
                ub[La + k] = wv - m
                eb[Lm + k] = e
                yv = wv
                for i in range(Ld):
                    yv -= diffc[i] * yb[Ld + k - 1 - i]
                yb[Ld + k] = yv
                e_prev = e
                if hi < n_h and horizon_k[hi] == k:
                    out[o, path, hi] = yv - y[t - 1]
                    hi += 1
    return out


def simulate(fit_, filt, y, X, stochastic, origins, horizons, n_paths, seed):
    """h-day-ahead predictive samples from each origin, for every h in horizons.

    Regressors that are themselves random (lagged constituent returns) are only known one
    day ahead; beyond that they are replaced by their mean over the fitting window.
    """
    spec, P = fit_.spec, filt.P
    H = max(horizons)
    idx = origins[:, None] + np.arange(H)[None, :]
    Xf = X[:, :spec.k][idx] if spec.k else np.zeros((len(origins), H, 0))
    if spec.k and stochastic.any():
        means = X[fit_.t0:fit_.t_end, :spec.k].mean(axis=0)
        Xf[:, 1:, stochastic] = means[stochastic]
    reg_paths = np.ascontiguousarray(P["c"] + (Xf @ P["beta"] if spec.k else 0.0) + np.zeros((len(origins), H)))
    ska, skb, _ = skewt_constants(P["nu"], P["lam"]) if spec.dist == "skewt" else (0.0, 1.0, 1.0)
    nu = P["nu"] if np.isfinite(P["nu"]) else 0.0
    diffc = diff_poly(spec.d, spec.D, spec.s)[1:]
    return _simulate(origins.astype(np.int64), np.asarray(horizons, np.int64) - 1, H, n_paths, seed,
                     y, filt.u, filt.eps, filt.s2, reg_paths, P["ar"], P["ma"], diffc,
                     spec.garch, spec.in_mean, P["delta"], P["sbar2"], P["omega"], P["alpha"],
                     P["gamma"], P["b"], DIST_CODE[spec.dist], nu, P["lam"], ska, skb)


def walk_forward(base, w, y, X, stochastic, start, stop, block, horizons, n_paths, summarize,
                 warm=None, seed=0):
    """Expanding-window evaluation on [start, stop).

    Every `block` days all arms are refitted on everything before the block, then used with
    frozen parameters for that block: one-step-ahead filtering for every day, and
    n_paths simulated paths from every origin t with t + max(horizons) - 1 < len(y).
    `summarize(samples, origins)` reduces each block's samples so they are not all kept.
    """
    n, H = len(w), max(horizons)
    one = {arm: {key: np.full(n, np.nan) for key in ("ll", "mu", "s2")} for arm in ARMS}
    multi = {arm: [] for arm in ARMS}
    refits = []
    for bi, s in enumerate(range(start, stop, block)):
        e = min(s + block, stop)
        warm = fit_arms(base, w, X, t_end=s, warm=warm)
        origins = np.arange(s, min(e, n - H + 1))
        for arm, f in warm.items():
            filt = f.filter(w, X)
            for key in ("ll", "mu", "s2"):
                one[arm][key][s:e] = getattr(filt, key)[s:e]
            if len(origins):
                samples = simulate(f, filt, y, X, stochastic, origins, horizons, n_paths, seed + bi)
                multi[arm].append(summarize(samples, origins))
        refits.append((s, warm))
    return one, multi, refits
