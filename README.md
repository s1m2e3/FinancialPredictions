# FinancialPredictions

Can a validated statistical time-series model teach a neural network? A
feed-forward network predicting the NASDAQ Composite is regularised toward a
fitted SARIMAX forecast, in the same spirit as a physics-informed neural network —
except the "physics" here is an autoregressive model rather than a differential
equation.

## What the project is about

Financial time series are noisy and short. A neural network given three years of
daily closes has plenty of capacity to fit the noise, and classical
autoregressive models (ARIMA/SARIMAX) often match or beat it out of sample
precisely because their structure is a strong prior.

The idea tested here is to stop treating that as a competition. A SARIMAX model is
fitted on the training window first, and its one-step forecast is then used as a
**soft constraint on the neural network's own one-day-ahead output**:

```
L  =  MSE(network output, ground truth)                        the usual fit term
   +  lambda * MSE(network linear-head output, SARIMAX forecast)   the prior term
```

with `lambda = 0.001`. The prior term is attached to `forward_linear` — the
network's first (linear) layer output — not to the full non-linear forward pass.
That is deliberate: the linear head is the part of the network whose role is
closest to what an autoregressive model does, so the SARIMAX forecast constrains
that subspace while the deeper layers stay free to learn the residual structure.

The experiment is a paired A/B: the identical architecture, initialisation scheme,
optimiser, learning rate and epoch count are trained twice — once with the prior
term (`autocorr=True`) and once without (`autocorr=False`) — and compared on a
held-out 30% tail.

## Setup

| component | choice |
|---|---|
| target | NASDAQ Composite (`^IXIC`) adjusted close, min-max normalised |
| features | the 10 heaviest Composite constituents (AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, GOOG, META, AVGO, PEP) — adjusted close **and** volume |
| window | 36 months of daily bars, resampled to calendar days and forward-filled |
| input | 5 days x 10 tickers x 2 channels, flattened to 50 features |
| output | the next 3 days of the Composite (a 3-step direct forecast) |
| split | first 70% train, last 30% test |
| network | 50 -> 3 -> 512 -> 3 feed-forward, ReLU, plain SGD, lr 0.01, 10 000 epochs |
| statistical model | SARIMAX(4, 1, 5) x (1, 1, 1, 7) — weekly seasonality, differencing order chosen by an ADF test on the training window |

The SARIMAX order is not searched per run: `p=4`, `q=5` are fixed, and only the
differencing order is selected, by comparing ADF p-values at `d=0` and `d=1`. The
`d` chosen this way is reported but the model is instantiated with `order=(p,1,q)`.

## Layout

```
main.py     the whole experiment: download, fit SARIMAX, train both networks,
            score, and write all three figures
model.py    NN (the feed-forward model with the SARIMAX prior term in .train()),
            plus LSTM and RNN variants kept for reference
*.png       committed outputs of a previous run
```

`model.NN.train(..., autocorr=True)` is where the method lives: it walks the
training sequences, appends each to the fitted SARIMAX state
(`sarimax_pred` -> `statsmodels` `append` + `forecast`), stacks the resulting
one-step forecasts, and adds the weighted penalty term.

## Setup and running

There is no `requirements.txt`; the dependencies are:

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install "yfinance<0.2.51" statsmodels pmdarima torch numpy pandas \
            scipy matplotlib seaborn
```

Then, from the repository root:

```bash
python main.py
```

One run does everything: pulls the price history from Yahoo Finance, runs the ADF
test, fits SARIMAX, trains both networks for 10 000 epochs each, evaluates on the
held-out tail and writes the figures. CUDA is used automatically if available;
CPU is fine at this size.

## Outputs

| file | what it shows |
|---|---|
| `feedforward_autocorr.png` | two stacked panels — FFN+SARIMAX (top) vs plain FFN (bottom) — with the 1-, 2- and 3-day-ahead predictions scattered over the ground-truth line |
| `feedforward_autocorr_justones.png` | both models' 1-day-ahead predictions over a filled ground-truth area, the clearest single comparison |
| `lossescomparison.png` | held-out MSE broken down by forecast horizon for each model, plus a head-to-head total |

The committed PNGs are from a previous run and are what the figures should look
like.

## Reproducibility caveats

Read these before comparing your numbers to the committed figures.

- **The data window moves.** `yf.download(..., period="36mo")` is relative to
  today, so every run trains on a different 36 months and the absolute losses
  will differ. Pin `start=`/`end=` dates to get a fixed dataset.
- **No seeds are set.** Neither `torch` nor `numpy` is seeded, so the two arms of
  the comparison start from different initialisations on every run. Seed both
  before drawing conclusions from a single run.
- **`yfinance` API drift.** The script reads the `Adj Close` column, which
  `yfinance` stopped returning by default in 0.2.51 (it now auto-adjusts and
  drops the column). Either pin `yfinance<0.2.51` as above, or pass
  `auto_adjust=False` to `yf.download`.
- **pandas deprecation.** `.fillna(method="ffill")` is removed in pandas 3.x;
  replace with `.ffill()` if you hit it.
- `predict_arima` in `main.py` is a hand-rolled ARIMA evaluator that the final
  experiment does not call — the `statsmodels` SARIMAX object is used instead. It
  is left in place as a reference implementation.

## SARIMAX–GARCH experiment (statistical models only)

A separate experiment, independent of the neural network, asks what a GARCH
conditional variance adds to a properly identified SARIMAX conditional mean, from
one day to one month ahead. It works on `y = 100 log P` of `^IXIC`, 2013–2025.

```
differencing  w_t = (1-B)^d (1-B^s)^D y_t
regression    w_t = c + beta' x_t + delta sigma_t + u_t              (delta: GARCH-in-mean)
SARMA errors  phi(B) Phi(B^s) u_t = theta(B) Theta(B^s) eps_t
variance      sigma_t^2 = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2 + b sigma_{t-1}^2
innovations   eps_t = sigma_t z_t,  z_t ~ normal, Student-t or Hansen skewed-t
```

**1. Identification** (`identification.py`, training window only, Box–Jenkins):
unit-root tests (ADF, Phillips–Perron, KPSS) for d; OCSB, Canova–Hansen and STL
seasonal strength for D at seasons of 5, 21, 63 and 252 days; seasonal Ljung–Box
(classic and heteroskedasticity-robust) and a periodogram with Fisher's g test;
calendar regressors (day of week, month, turn of month) and lagged constituent
returns tested with HAC standard errors; then an AICc search over 400
SARIMA(p,d,q)(P,D,Q)[s] models on a common sample, and residual diagnostics.
The chosen specification is written to `results/identification/spec.json`.

**2. Comparison** (`run_armagarch.py`): six models share that mean equation, each
adding one thing: `sarimax` (constant variance, normal) → `sarimax_t` (fat tails)
→ `two_step` (GJR-GARCH-t fitted on the SARIMAX residuals) → `joint` (estimated
together) → `joint_skewt` (skewed-t) → `full` (GARCH-in-mean).

The split is chronological: train 2013–2020, validation 2021–2022, test 2023–2025.
The test window is walk-forward: every 63 trading days every model is refitted on
everything before the block, then forecasts with frozen parameters. One-day
densities are exact; for 1 day, 3 days, 1 week, 2 weeks and 1 month ahead, 5,000
paths are simulated from every origin and the cumulative log return is scored by
CRPS, interval coverage and VaR hit rates.

```
conda create -n financialPredictions python=3.11
conda activate financialPredictions
pip install -r requirements.txt
python identification.py                     # Box-Jenkins; writes the spec (~15 min order search)
python run_armagarch.py                      # six-model comparison, AICc order (~11 min)
python run_armagarch.py --criterion BIC      # same with the BIC order
python compare_specs.py                      # AICc order vs BIC order, head to head
python checks/verify_against_libraries.py    # estimator and simulator checks
```

| file | role |
|---|---|
| `data.py` | download (cached to `data/prices.csv`), log prices, candidate regressors, split dates |
| `armagarch.py` | model, numba filter and path simulator, L-BFGS fit, the six arms, walk-forward |
| `identification.py` | Box–Jenkins identification; `results/identification/` |
| `evaluation.py` | log score, CRPS, coverage, VaR backtests, Ljung–Box, Diebold–Mariano |
| `run_armagarch.py` | the comparison; `results/sarimax_garch_<criterion>/metrics.md`, seven figures, `scores.npz` |
| `compare_specs.py` | Diebold–Mariano tests of the AICc vs BIC orders; `results/spec_comparison/` |
| `checks/verify_against_libraries.py` | vs statsmodels SARIMAX, arch GARCH and skewed-t, closed-form forecasts |

## Context

Listed under Projects as *Physics-Informed Financial Time-Series Prediction*. The prior-as-regulariser pattern is the same one used in
[PINN-Bicycle-Model](https://github.com/s1m2e3/PINN-Bicycle-Model), where the
prior is vehicle dynamics rather than an autoregressive model.

Author: Samuel Cornejo (<samuelcornejo@arizona.edu>)
