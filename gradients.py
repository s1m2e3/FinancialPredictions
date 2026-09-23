"""Exact gradient of the SARIMAX-GARCH negative log-likelihood, in reverse mode.

    L(u) = -(1/N) sum_{t0 <= t < t_end} ll_t,    ll_t = log f(eps_t | s2_t, distribution parameters)

computed in three chained pieces:

  1. per-observation derivatives of ll_t with respect to eps_t, s2_t and the distribution
     parameters: analytic, for the normal, Student-t, Hansen skewed-t and jump-mixture cases;
  2. an adjoint (backpropagation) pass through the filter recursion in armagarch._filter, which
     turns those into derivatives with respect to the regression part reg_t = c + beta' x_t, the
     expanded AR and MA coefficients, delta, the pre-sample variance and omega, alpha, gamma, b;
  3. the Jacobian of those constrained parameters with respect to the unconstrained vector u:
     closed form for the polynomial products (linear in each factor), the GARCH softmax and
     the exp / logistic / tanh maps; complex-step differentiation, exact to machine precision,
     for the small partial-autocorrelation recursions.

One forward and one backward pass replace the n_params + 1 filter passes of finite differences.
checks/verify_against_libraries.py compares the gradient with central differences and the
Jacobian with a full complex-step Jacobian.
"""
import numpy as np
from numba import njit
from scipy.special import digamma, gammaln

from armagarch import MAX_PERSISTENCE, N_MAX_JUMPS, LOG_2PI, _filter, skewt_constants, unpack


# ------------------------------------------------------------------ 3. parameter transforms
def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pacf_to_coef(u):
    phi = np.zeros(0, dtype=u.dtype)
    for rk in np.tanh(u):
        phi = np.append(phi - rk * phi[::-1], rk)
    return phi


def _seasonal(coef, s):
    out = np.zeros(len(coef) * s, dtype=coef.dtype)
    out[s - 1::s] = coef
    return out


def _theta(u, spec):
    """Constrained parameters that the filter and density use, as one vector (complex-safe).

    Layout: c, beta (k), delta, ar (p + s P), ma (q + s Q),
            sbar2, omega, alpha, gamma, b, nu, lam, lam_j, mu_j, sig_j.
    Entries a spec does not use are constant zeros.
    """
    sl = spec.slices()
    s = max(spec.s, 1)
    zero = u[0] * 0
    phi, Phi = _pacf_to_coef(u[sl["ar"]]), _pacf_to_coef(u[sl["sar"]])
    theta, Theta = -_pacf_to_coef(u[sl["ma"]]), -_pacf_to_coef(u[sl["sma"]])
    ar = -np.convolve(np.r_[1.0, -phi], np.r_[1.0, -_seasonal(Phi, s)])[1:]
    ma = np.convolve(np.r_[1.0, theta], np.r_[1.0, _seasonal(Theta, s)])[1:]
    sbar2 = np.exp(u[sl["logvar"]][0])
    if spec.garch:
        g = u[sl["garch"]]
        pers = MAX_PERSISTENCE * _expit(g[0])
        e = np.exp(np.array([g[1], g[2], zero]))
        wts = e / e.sum()
        alpha, gamma, b = pers * wts[0], 2 * pers * wts[1], pers * wts[2]
        omega = sbar2 * (1 - pers)
    else:
        alpha = gamma = b = zero
        omega = sbar2
    nu = 2.05 + np.exp(u[sl["nu"]][0]) if spec.dist in ("t", "skewt") else zero
    lam = np.tanh(u[sl["skew"]][0]) if spec.dist == "skewt" else zero
    if spec.jumps:
        j = u[sl["jump"]]
        lam_j, mu_j, sig_j = _expit(j[0]), j[1], np.exp(j[2])
    else:
        lam_j = mu_j = sig_j = zero
    delta = u[sl["delta"]][0] if spec.in_mean else zero
    return np.concatenate([[u[sl["c"]][0]], u[sl["beta"]], [delta], ar, ma,
                           [sbar2, omega, alpha, gamma, b, nu, lam, lam_j, mu_j, sig_j]])


def theta_jacobian_complex_step(u, spec, h=1e-30):
    """d theta / d u by the complex step, Im theta(u + i h e_j) / h: the reference for tests."""
    cols = []
    for j in range(len(u)):
        uc = u.astype(complex)
        uc[j] += 1j * h
        cols.append(_theta(uc, spec).imag / h)
    return np.array(cols).T


def _pacf_jacobian(u, h=1e-30):
    """d coef / d u of the partial-autocorrelation map (at most a few coefficients)."""
    J = np.zeros((len(u), len(u)))
    for j in range(len(u)):
        uc = u.astype(complex)
        uc[j] += 1j * h
        J[:, j] = _pacf_to_coef(uc).imag / h
    return J


def _poly_factor_jacobian(own, other, s_own, s_other, sign):
    """d/d own_i of the coefficients (lags 1..) of (1 + sign sum own_i B^{s_own i}) (1 + sign sum other_j B^{s_other j}).

    The product is linear in each factor: d/d own_i = sign B^{s_own i} (1 + sign sum_j other_j B^{s_other j}).
    """
    L = len(own) * s_own + len(other) * s_other
    D = np.zeros((L, len(own)))
    for i in range(len(own)):
        lag = s_own * (i + 1)
        D[lag - 1, i] += sign
        for j, o in enumerate(other):
            D[lag + s_other * (j + 1) - 1, i] += sign * sign * o
    return D


def theta_jacobian(u, spec):
    """d theta / d u, analytic except for the tiny partial-autocorrelation maps."""
    sl = spec.slices()
    s = max(spec.s, 1)
    P = unpack(u, spec)
    k, La, Lm = spec.k, len(P["ar"]), len(P["ma"])
    J = np.zeros((1 + k + 1 + La + Lm + 10, spec.n_params))
    J[0, sl["c"].start] = 1.0
    J[1:1 + k, sl["beta"]] = np.eye(k)
    if spec.in_mean:
        J[1 + k, sl["delta"].start] = 1.0
    r = 2 + k
    # ar = -(A[1:]) with A = (1 - sum phi B^i)(1 - sum Phi B^{s j}): d ar = d of the product with sign +1
    if La:
        if spec.p:
            J[r:r + La, sl["ar"]] = -_poly_factor_jacobian(P["phi"], P["Phi"], 1, s, -1.0) @ _pacf_jacobian(u[sl["ar"]])
        if spec.P:
            J[r:r + La, sl["sar"]] = -_poly_factor_jacobian(P["Phi"], P["phi"], s, 1, -1.0) @ _pacf_jacobian(u[sl["sar"]])
    r += La
    # ma = M[1:] with M = (1 + sum theta B^i)(1 + sum Theta B^{s j}); theta = -pacf(u)
    if Lm:
        if spec.q:
            J[r:r + Lm, sl["ma"]] = _poly_factor_jacobian(P["theta"], P["Theta"], 1, s, 1.0) @ -_pacf_jacobian(u[sl["ma"]])
        if spec.Q:
            J[r:r + Lm, sl["sma"]] = _poly_factor_jacobian(P["Theta"], P["theta"], s, 1, 1.0) @ -_pacf_jacobian(u[sl["sma"]])
    r += Lm
    i_sbar2, i_omega, i_alpha, i_gamma, i_b, i_nu, i_lam, i_lj, i_mj, i_sj = range(r, r + 10)
    lv = sl["logvar"].start
    J[i_sbar2, lv] = P["sbar2"]
    J[i_omega, lv] = P["omega"]
    if spec.garch:
        g0, g1, g2 = sl["garch"].start, sl["garch"].start + 1, sl["garch"].start + 2
        pers = P["persistence"]
        dpers = pers * (1 - pers / MAX_PERSISTENCE)                 # d/dg0 of 0.999 * expit(g0)
        wts = np.array([P["alpha"], P["gamma"] / 2, P["b"]]) / pers  # softmax shares
        for row, i, mult in ((i_alpha, 0, 1.0), (i_gamma, 1, 2.0), (i_b, 2, 1.0)):
            J[row, g0] = mult * wts[i] * dpers
            J[row, g1] = mult * pers * wts[i] * ((i == 0) - wts[0])
            J[row, g2] = mult * pers * wts[i] * ((i == 1) - wts[1])
        J[i_omega, g0] = -P["sbar2"] * dpers
    if spec.dist in ("t", "skewt"):
        J[i_nu, sl["nu"].start] = P["nu"] - 2.05
    if spec.dist == "skewt":
        J[i_lam, sl["skew"].start] = 1 - P["lam"] ** 2
    if spec.jumps:
        j0 = sl["jump"].start
        J[i_lj, j0] = P["lam_j"] * (1 - P["lam_j"])
        J[i_mj, j0 + 1] = 1.0
        J[i_sj, j0 + 2] = P["sig_j"]
    return J


# ------------------------------------------------------------------ 1. density derivatives
def density_grads(eps, s2, spec, P):
    """d ll_t / d(eps_t, s2_t) and d ll_t / d(nu, lam, lam_j, mu_j, sig_j), per observation."""
    zeros = np.zeros_like(eps)
    out = {"nu": zeros, "lam": zeros, "lam_j": zeros, "mu_j": zeros, "sig_j": zeros}
    if spec.jumps:
        lj, mj, sj = P["lam_j"], P["mu_j"], P["sig_j"]
        n = np.arange(N_MAX_JUMPS + 1)[:, None]
        V = s2[None, :] + n * sj ** 2
        M = (n - lj) * mj
        D = (eps[None, :] - M) / V
        comp = -lj + n * np.log(lj) - gammaln(n + 1) - 0.5 * (LOG_2PI + np.log(V) + (eps[None, :] - M) * D)
        r = np.exp(comp - np.logaddexp.reduce(comp, axis=0))              # posterior jump counts
        dV = -0.5 / V + 0.5 * D ** 2
        out.update(eps=-(r * D).sum(0), s2=(r * dV).sum(0), sig_j=(r * dV * n).sum(0) * 2 * sj,
                   mu_j=(r * D * (n - lj)).sum(0), lam_j=(r * ((n / lj - 1) - D * mj)).sum(0))
        return out
    if spec.dist == "normal":
        out.update(eps=-eps / s2, s2=-0.5 / s2 + 0.5 * eps ** 2 / s2 ** 2)
        return out
    nu = P["nu"]
    dlogc = 0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / (nu - 2)
    if spec.dist == "t":
        q = eps ** 2 / ((nu - 2) * s2)
        out.update(eps=-(nu + 1) * eps / ((nu - 2) * s2 * (1 + q)),
                   s2=-0.5 / s2 + 0.5 * (nu + 1) * q / (s2 * (1 + q)),
                   nu=dlogc - 0.5 * np.log1p(q) + 0.5 * (nu + 1) * q / ((nu - 2) * (1 + q)))
        return out
    lam = P["lam"]
    a, b, c = skewt_constants(nu, lam)
    sd = np.sqrt(s2)
    z = eps / sd
    v = b * z + a
    sc = np.where(v < 0, 1 - lam, 1 + lam)
    r = (v / sc) ** 2 / (nu - 2)
    d_v = -(nu + 1) * v / (sc ** 2 * (nu - 2) * (1 + r))
    d_sc = (nu + 1) * v ** 2 / (sc ** 3 * (nu - 2) * (1 + r))
    da_dlam = 4 * c * (nu - 2) / (nu - 1)
    da_dnu = 4 * lam * (c * dlogc * (nu - 2) / (nu - 1) + c / (nu - 1) ** 2)
    db_dlam = (3 * lam - a * da_dlam) / b
    db_dnu = -a * da_dnu / b
    out.update(eps=d_v * b / sd,
               s2=d_v * b * (-0.5 * z / s2) - 0.5 / s2,
               lam=db_dlam / b + d_v * (z * db_dlam + da_dlam) + d_sc * np.where(v < 0, -1.0, 1.0),
               nu=db_dnu / b + dlogc + d_v * (z * db_dnu + da_dnu)
               - 0.5 * np.log1p(r) + 0.5 * (nu + 1) * r / ((nu - 2) * (1 + r)))
    return out


# ------------------------------------------------------------------ 2. adjoint of the filter
@njit(cache=True)
def _adjoint(reg, ar, ma, garch, in_mean, delta, alpha, gamma, b, start, t0,
             u, eps, s2, g_eps, g_s2):
    """Backpropagate dL/d eps_t and dL/d s2_t through armagarch._filter (same arguments).

    Walks t from the end back to `start`, mirroring the forward loop step by step:
        s2_t  = omega + (alpha + gamma 1[eps_{t-1}<0]) eps_{t-1}^2 + b s2_{t-1}   (t > t0), else sbar2
        m_t   = reg_t + delta sqrt(s2_t)
        mu_t  = m_t + sum_i ar_i u_{t-1-i} + sum_i ma_i eps_{t-1-i}
        u_t   = w_t - m_t
        eps_t = w_t - mu_t   (t >= t0), else 0
    """
    n = reg.shape[0]
    La, Lm = ar.shape[0], ma.shape[0]
    eb = g_eps.copy()
    sb = g_s2.copy()
    ub = np.zeros(n)
    g_reg = np.zeros(n)
    g_ar = np.zeros(La)
    g_ma = np.zeros(Lm)
    g_delta = g_sbar2 = g_omega = g_alpha = g_gamma = g_b = 0.0
    for t in range(n - 1, start - 1, -1):
        if t >= t0:
            arma_bar = -eb[t]
            m_bar = -eb[t] - ub[t]
        else:                       # before t0 eps is fixed at 0 and mu_t is unused
            arma_bar = 0.0
            m_bar = -ub[t]
        if arma_bar != 0.0:
            for i in range(La):
                j = t - 1 - i
                if j >= start:
                    g_ar[i] += arma_bar * u[j]
                    ub[j] += arma_bar * ar[i]
            for i in range(Lm):
                j = t - 1 - i
                if j >= start:
                    g_ma[i] += arma_bar * eps[j]
                    eb[j] += arma_bar * ma[i]
        g_reg[t] += m_bar
        if in_mean:
            sq = np.sqrt(s2[t])
            g_delta += m_bar * sq
            sb[t] += m_bar * delta / (2.0 * sq)
        if garch and t > t0:
            e = eps[t - 1]
            neg = e < 0.0
            a = alpha + gamma if neg else alpha
            g_omega += sb[t]
            g_alpha += sb[t] * e * e
            if neg:
                g_gamma += sb[t] * e * e
            g_b += sb[t] * s2[t - 1]
            eb[t - 1] += sb[t] * 2.0 * a * e
            sb[t - 1] += sb[t] * b
        else:
            g_sbar2 += sb[t]
    return g_reg, g_ar, g_ma, g_delta, g_sbar2, g_omega, g_alpha, g_gamma, g_b


# ------------------------------------------------------------------ assembled
def value_and_grad(u, spec, w, X, t0, t_end):
    """Mean negative log-likelihood over w[t0:t_end] and its exact gradient w.r.t. u."""
    P = unpack(u, spec)
    w = w[:t_end]
    Xk = X[:t_end, :spec.k]
    reg = P["c"] + (Xk @ P["beta"] if spec.k else 0.0) + np.zeros(t_end)
    mu, uu, eps, s2 = _filter(w, reg, P["ar"], P["ma"], spec.garch, spec.in_mean, P["delta"], P["sbar2"],
                              P["omega"], P["alpha"], P["gamma"], P["b"], spec.n_diff, t0)
    e, v = eps[t0:], s2[t0:]
    N = t_end - t0
    from armagarch import jump_mixture_loglik, log_density
    if spec.jumps:
        ll = jump_mixture_loglik(e, v, P["lam_j"], P["mu_j"], P["sig_j"])
    else:
        ll = log_density(e / np.sqrt(v), spec.dist, P["nu"], P["lam"]) - 0.5 * np.log(v)
    L = -ll.mean()
    if not np.isfinite(L):
        return 1e10, np.zeros_like(u)

    dg = density_grads(e, v, spec, P)
    g_eps = np.zeros(t_end)
    g_s2 = np.zeros(t_end)
    g_eps[t0:] = -dg["eps"] / N
    g_s2[t0:] = -dg["s2"] / N
    g_reg, g_ar, g_ma, g_delta, g_sbar2, g_omega, g_alpha, g_gamma, g_b = _adjoint(
        reg, P["ar"], P["ma"], spec.garch, spec.in_mean, P["delta"], P["alpha"], P["gamma"], P["b"],
        spec.n_diff, t0, uu, eps, s2, g_eps, g_s2)
    g_theta = np.concatenate([[g_reg.sum()], Xk.T @ g_reg if spec.k else [], [g_delta], g_ar, g_ma,
                              [g_sbar2, g_omega, g_alpha, g_gamma, g_b,
                               -dg["nu"].sum() / N, -dg["lam"].sum() / N,
                               -dg["lam_j"].sum() / N, -dg["mu_j"].sum() / N, -dg["sig_j"].sum() / N]])
    return L, theta_jacobian(u, spec).T @ g_theta
