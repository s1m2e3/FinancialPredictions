"""Cross-check armagarch.py against statsmodels, arch and closed-form results.

Our estimator is conditional MLE; statsmodels uses the exact (Kalman) likelihood and
arch initialises the variance with a backcast, so estimates should agree closely but not
to machine precision. Monte Carlo checks agree up to simulation noise.

Run from the repository root:  python checks/verify_against_libraries.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armagarch import Spec, difference, fit, fit_multistart, log_density, simulate  # noqa: E402
from data import build_dataset, load_prices                                          # noqa: E402

warnings.filterwarnings("ignore")
d = build_dataset(load_prices())
n_tr = d.n_train


def row(name, ours, theirs):
    print(f"  {name:<10} ours {ours:>10.4f}   reference {theirs:>10.4f}   diff {ours - theirs:>+9.4f}")


w = difference(d.y, Spec(d=1))
assert np.allclose(w[1:], np.diff(d.y)), "first difference"
w_tr = w[1:n_tr]

# 1. Seasonal SARMA(1,1)(1,1)[5] + regressor vs statsmodels SARIMAX (regression with SARMA errors)
from statsmodels.tsa.statespace.sarimax import SARIMAX   # noqa: E402

cols = ["turn_of_month"]
X = d.X(cols)
spec = Spec(p=1, d=1, q=1, P=1, Q=1, s=5, k=1)
ours = fit_multistart(spec, w, X, n_tr).params    # multi-start: the seasonal likelihood has a ridge
sm = SARIMAX(w_tr, exog=X[1:n_tr], order=(1, 0, 1), seasonal_order=(1, 0, 1, 5), trend="c").fit(disp=False)
smp = dict(zip(sm.model.param_names, sm.params))
print("SARMA(1,1)(1,1)[5] + turn-of-month regressor  vs  statsmodels SARIMAX")
# statsmodels puts the constant in the SARMA equation: its intercept is c * phi(1) * Phi(1)
row("c", ours["c"], smp["intercept"] / ((1 - smp["ar.L1"]) * (1 - smp["ar.S.L5"])))
row("beta_tom", ours["beta"][0], smp["x1"])
row("phi1", ours["phi"][0], smp["ar.L1"])
row("theta1", ours["theta"][0], smp["ma.L1"])
row("Phi1 (s=5)", ours["Phi"][0], smp["ar.S.L5"])
row("Theta1", ours["Theta"][0], smp["ma.S.L5"])
row("sigma2", ours["sbar2"], smp["sigma2"])

# 2. AR(1)-GJR-GARCH(1,1)-t vs arch ------------------------------------------------------
from arch import arch_model                                 # noqa: E402
from arch.univariate import SkewStudent                     # noqa: E402

ours = fit(Spec(p=1, d=1, garch=True, dist="t"), w, np.zeros((d.n, 0)), n_tr).params
am = arch_model(w_tr, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").params
print("\nAR(1)-GJR-GARCH(1,1)-t  vs  arch")
row("c", ours["c"] * (1 - ours["phi"][0]), am["Const"])   # arch's constant is c * (1 - phi)
row("phi1", ours["phi"][0], am["y[1]"])
row("omega", ours["omega"], am["omega"])
row("gamma", ours["gamma"], am["gamma[1]"])
row("beta", ours["b"], am["beta[1]"])
row("nu", ours["nu"], am["nu"])

# 3. Skewed-t density vs arch's SkewStudent (same Hansen parameterisation) -----------------
nu, lam = 5.0, -0.2
z = np.linspace(-6, 6, 41)
ref = np.array([SkewStudent().loglikelihood(np.array([nu, lam]), np.array([zi]), np.array([1.0]),
                                            individual=True)[0] for zi in z])
print(f"\nskewed-t log-density, nu={nu}, lambda={lam}: max |ours - arch| = "
      f"{np.max(np.abs(log_density(z, 'skewt', nu, lam) - ref)):.2e}")

# 4. Simulator: constant-variance Gaussian ARIMA(1,1,1) h-step mean and variance vs closed form
f = fit(Spec(p=1, d=1, q=1), w, np.zeros((d.n, 0)), n_tr)
filt = f.filter(w, np.zeros((d.n, 0)))
P = f.params
horizons = [1, 5, 21]
origin = np.array([n_tr])
sims = simulate(f, filt, d.y, np.zeros((d.n, 0)), np.zeros(0, bool), origin, horizons, 200_000, 1)[0]
psi = np.zeros(21)                        # MA(infinity) weights of the ARMA(1,1) for w
psi[0] = 1.0
psi[1] = P["phi"][0] + P["theta"][0]
for j in range(2, 21):
    psi[j] = P["phi"][0] * psi[j - 1]
cum = np.cumsum(psi)                      # weights of the cumulative sum of w
print("\nsimulator vs closed form, ARIMA(1,1,1) with constant variance")
for hi, h in enumerate(horizons):
    var_exact = P["sbar2"] * np.sum(cum[:h] ** 2)
    row(f"var h={h}", sims[:, hi].var(), var_exact)

# 5. Simulator: one-day-ahead quantiles vs the exact skewed-t quantiles
f = fit(Spec(p=1, d=1, garch=True, dist="skewt"), w, np.zeros((d.n, 0)), n_tr)
filt = f.filter(w, np.zeros((d.n, 0)))
P = f.params
sims = simulate(f, filt, d.y, np.zeros((d.n, 0)), np.zeros(0, bool), origin, [1], 400_000, 2)[0, :, 0]
t = n_tr
levels = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
exact = filt.mu[t] + np.sqrt(filt.s2[t]) * SkewStudent().ppf(levels, np.array([P["nu"], P["lam"]]))
print(f"\nsimulator vs exact one-day skewed-t quantiles (nu={P['nu']:.2f}, lambda={P['lam']:.3f})")
for lv, e_, s_ in zip(levels, exact, np.quantile(sims, levels)):
    row(f"q{lv:.2f}", s_, e_)

# 6. GARCH-jump: the Poisson-mixture density is a proper, centred density, and the
#    simulator reproduces its exact one-day quantiles
from scipy import integrate, optimize, stats                  # noqa: E402

from armagarch import jump_mixture_loglik                     # noqa: E402

f = fit(Spec(p=1, d=1, garch=True, jumps=True), w, np.zeros((d.n, 0)), n_tr)
filt = f.filter(w, np.zeros((d.n, 0)))
P = f.params
s2t = filt.s2[t]
dens = lambda x: np.exp(jump_mixture_loglik(np.atleast_1d(x), np.array([s2t]), P["lam_j"], P["mu_j"], P["sig_j"]))[0]
mass = integrate.quad(dens, -60, 60, limit=400)[0]
mean = integrate.quad(lambda x: x * dens(x), -60, 60, limit=400)[0]
var = integrate.quad(lambda x: x * x * dens(x), -60, 60, limit=400)[0]
print(f"\nGARCH-jump (lambda_J={P['lam_j']:.4f}/day, mu_J={P['mu_j']:.3f}, sigma_J={P['sig_j']:.3f})")
row("mass", mass, 1.0)
row("mean", mean, 0.0)
row("variance", var, s2t + P["lam_j"] * (P["sig_j"] ** 2 + P["mu_j"] ** 2))
n = np.arange(11)
w_n = stats.poisson.pmf(n, P["lam_j"])
cdf = lambda x: np.sum(w_n * stats.norm.cdf(x, (n - P["lam_j"]) * P["mu_j"], np.sqrt(s2t + n * P["sig_j"] ** 2)))
sims = simulate(f, filt, d.y, np.zeros((d.n, 0)), np.zeros(0, bool), origin, [1], 400_000, 3)[0, :, 0]
for lv, s_ in zip(levels, np.quantile(sims, levels)):
    row(f"q{lv:.2f}", s_, filt.mu[t] + optimize.brentq(lambda x: cdf(x) - lv, -60, 60))

# 7. Exact gradient (gradients.py) vs central finite differences, every model type
from armagarch import ARMS, arm_spec, filter_series, initial_params                  # noqa: E402
from gradients import theta_jacobian, theta_jacobian_complex_step, value_and_grad    # noqa: E402

rng = np.random.default_rng(0)
cases = [(f"{a}", arm_spec(a, Spec(p=2, d=1, q=2, Q=1, s=5)), w, np.zeros((d.n, 0))) for a in ARMS]
cases += [("seasonal + regressor, GARCH-M skew-t", Spec(p=1, d=1, q=1, P=1, Q=1, s=21, k=1, garch=True,
                                                         dist="skewt", in_mean=True), w, X),
          ("OU level + trend, GARCH + jumps", Spec(p=1, d=0, k=1, garch=True, jumps=True), d.y.copy(),
           (np.arange(d.n) / 252.0)[:, None])]
print("\nexact gradient vs central differences (max relative error), Jacobian vs complex step")
for name, spec, ww, XX in cases:
    u = initial_params(spec, ww[spec.t0:n_tr]) + 0.3 * rng.standard_normal(spec.n_params)
    _, g = value_and_grad(u, spec, ww, XX, spec.t0, n_tr)
    f0 = lambda v: -filter_series(v, spec, ww, XX, spec.t0).ll[spec.t0:n_tr].mean()
    gf = np.array([(f0(u + 1e-6 * e) - f0(u - 1e-6 * e)) / 2e-6 for e in np.eye(len(u))])
    jac = np.max(np.abs(theta_jacobian(u, spec) - theta_jacobian_complex_step(u, spec)))
    print(f"  {name:<40} gradient {np.max(np.abs(g - gf)) / np.max(np.abs(gf)):.1e}   Jacobian {jac:.1e}")
