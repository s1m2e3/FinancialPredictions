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
              or  sigma_t z_t + sum_{k<=N_t} Y_k - lambda_J mu_J,  N_t ~ Poisson(lambda_J), Y_k ~ N(mu_J, sigma_J^2)
```

**1. Identification** (`identification.py`, training window only, Box–Jenkins):
unit-root tests (ADF, Phillips–Perron, KPSS) for d; OCSB, Canova–Hansen and STL
seasonal strength for D at seasons of 5, 21, 63 and 252 days; seasonal Ljung–Box
(classic and heteroskedasticity-robust) and a periodogram with Fisher's g test;
calendar regressors (day of week, month, turn of month) and lagged constituent
returns tested with HAC standard errors; then an AICc search over 900
SARIMA(p,d,q)(P,D,Q)[s] models (p, q <= 5, P, Q <= 2) on a common sample, and
residual diagnostics.
The chosen specification is written to `results/identification/spec.json`.

**2. Comparison** (`run_armagarch.py`): seven models share that mean equation. Six
form a ladder, each adding one thing: `sarimax` (constant variance, normal) →
`sarimax_t` (fat tails) → `two_step` (GJR-GARCH-t fitted on the SARIMAX residuals)
→ `joint` (estimated together) → `joint_skewt` (skewed-t) → `full` (GARCH-in-mean).
The seventh, `jump`, swaps the fat-tailed errors for a normal diffusion plus
Poisson jumps whose size does not scale with the GARCH volatility.

**3. Mean reversion** (`run_ou.py`): the OU-with-jumps SDE on the log price, which
sampled daily is an AR(1) in levels, with a constant or a linear-trend target and
with or without the GARCH variance, scored against the SARIMAX models on the same
test days.

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
python run_ou.py                             # OU-with-jumps models vs the SARIMAX run
python plot_paths.py                         # simulated price paths vs the real index
python checks/verify_against_libraries.py    # estimator and simulator checks
```

| file | role |
|---|---|
| `data.py` | download (cached to `data/prices.csv`), log prices, candidate regressors, split dates |
| `armagarch.py` | model, numba filter and path simulator, L-BFGS fit, the seven arms, walk-forward |
| `identification.py` | Box–Jenkins identification; `results/identification/` |
| `evaluation.py` | log score, CRPS, coverage, VaR backtests, Ljung–Box, Diebold–Mariano |
| `run_armagarch.py` | the comparison; `results/sarimax_garch_<criterion>/metrics.md`, figures, `scores.npz` |
| `compare_specs.py` | Diebold–Mariano tests of the AICc vs BIC orders; `results/spec_comparison/` |
| `run_ou.py` | OU-with-jumps (mean reversion in the level) vs SARIMAX + GARCH; `results/ou/` |
| `plot_paths.py` | simulated one-month price paths from the walk-forward models vs the real index |
| `checks/verify_against_libraries.py` | vs statsmodels SARIMAX, arch GARCH and skewed-t, closed-form forecasts |

## Stock portfolio with behaviour trees (in progress)

A stock portfolio run by two behaviour trees **discovered** with
[btind](https://github.com/s1m2e3/btind), judged against the S&P 500 (dividends
reinvested) on risk-adjusted **real** return.

- **Universe** (`stocks_data.py`), point-in-time, so the backtest never trades a company
  it could not have known about: each month the 100 largest S&P 500 members of that day
  plus the 20 largest Nasdaq-100 members *outside* the S&P 500 (the growth names the S&P
  admits late: Tesla 2013-2020, Mercado Libre, ASML, PDD ...), sized by trailing dollar
  volume. Membership from the S&P 500's change history
  ([fja05680/sp500](https://github.com/fja05680/sp500)) and the Nasdaq-100 Wikipedia
  article's revision history, one snapshot a quarter; renamed symbols are mapped to the
  company's current one, symbols now held by another company are dropped.
- **Information layer**: the statistical models are the inputs. `stock_features.py` gives
  every stock a walk-forward AR(1)-GJR-GARCH with skewed-t errors and its 1, 3, 5, 10 and
  21-day mean, volatility, 5% quantile and P(up) in closed form; the S&P 500 goes through
  the full model (SARIMA + GJR-GARCH-in-mean, skewed-t) with those horizons simulated;
  plus the overnight gap, momentum, reversal, beta, realised volatility, their
  cross-sectional ranks, VIX, trend and index membership. `fundamentals.py` adds SEC
  EDGAR XBRL fundamentals as first filed (never restated): EBTDA / assets, margin, 3-year
  record and change ranked within sector, and revenue growth, including foreign filers
  (IFRS, non-USD); a missing value is a random rank, so the years before EDGAR's XBRL
  (2006-2009) cannot act as a hidden date flag. Each row uses information up to the previous close plus that day's open.
- **Two trees** (`portfolio_env.py`), trained in alternation as each other's fixed
  partner: a **stock tree**, one row per stock, choosing exit / hold / buy at the open
  every 5 trading days; and an **exposure tree**, one row per portfolio, choosing how much
  is invested (25-100%, the rest in T-bills). Buys share the invested budget by inverse
  volatility; long-only, whole shares, $100k, 5 bp per trade.
- **Score**: certainty-equivalent real return, portfolio minus the S&P 500 (SPY with
  dividends reinvested, priced like the stocks), % per year: every return deflated by
  CPI-U (BLS), so idle cash loses what inflation takes, and variance charged at the risk
  aversion that made 100% S&P optimal on the training period.
- **Search** (`portfolio_bt.py`): btind's pure-RL loop on one-year episodes from 2006-2019,
  every move accepted only by a paired rollout test (z = 2) that no market regime may lose;
  then a continuous run over the validation (2020-2021) and test (2022-2026) periods
  against the S&P 500 and simple strategies through the same simulator, with a
  block-bootstrap p-value on the gap. Checkpointed; rerunning the command resumes.
- **Training curves** (`training_progress.py`): every tree the search adopts is replayed
  on 200 fixed training windows and drawn as % ahead of the S&P 500 (mean and quartiles),
  live with `--watch`.
- **Speed**: the portfolio rollout is a numba kernel built on btind's own compiled
  arbitration, held to 1e-12 agreement with the numpy reference by
  `checks/verify_portfolio_kernel.py`.
- btind is used from the `portfolio-env` branch of s1m2e3/btind (hooks for a world that
  scores its own episodes, keeps its own store and watches the trees the search adopts),
  cloned next to this repository.

**Caveat:** yfinance has no prices for most delisted stocks, so companies acquired or
bankrupt before today mostly cannot be traded. Measured against RSP (the equal-weight
S&P 500) the priced members still carry +0.7 to +3 %/yr, and the outside pool a similar
edge against QQEW; the "buy all" baselines carry the same edge, so the trees are judged
against them. `stocks_data.load_crsp` (CRSP via WRDS, delisted stocks included) is the fix.

```
python stock_features.py                          # information layer
SEC_USER_AGENT="Name email" python fundamentals.py  # EDGAR fundamentals (the SEC asks for a contact)
python checks/verify_portfolio_kernel.py          # kernel vs reference
python portfolio_bt.py 3                          # both trees, 4 stages x 3 rounds, then the report
python training_progress.py --watch               # training curves, live
```

## Context

Listed under Projects as *Physics-Informed Financial Time-Series Prediction*. The prior-as-regulariser pattern is the same one used in
[PINN-Bicycle-Model](https://github.com/s1m2e3/PINN-Bicycle-Model), where the
prior is vehicle dynamics rather than an autoregressive model.

Author: Samuel Cornejo (<samuelcornejo@arizona.edu>)
