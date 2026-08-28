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

## Context

Listed under Projects as *Physics-Informed Financial Time-Series Prediction*. The prior-as-regulariser pattern is the same one used in
[PINN-Bicycle-Model](https://github.com/s1m2e3/PINN-Bicycle-Model), where the
prior is vehicle dynamics rather than an autoregressive model.

Author: Samuel Cornejo (<samuelcornejo@arizona.edu>)
