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
