"""Price download, caching, candidate regressors and the chronological split.

The model works on y_t = 100 * log P_t of the Nasdaq Composite, in levels:
differencing is chosen by the identification step, not assumed here. Nothing is
normalised on the full sample.

Candidate exogenous regressors (the X of SARIMAX), aligned so that row t only
holds information available before day t's close:
  - calendar dummies for day t (deterministic, known in advance):
      day of week (Monday is the base), month of year (January is the base),
      turn of month (last trading day of a month and the first three of the next)
  - constituent log returns of day t-1 (stochastic: unknown beyond one day ahead)
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

INDEX = "^IXIC"
# Heaviest Nasdaq Composite constituents. GOOG is left out on purpose: it is
# almost perfectly collinear with GOOGL, which makes both coefficients unidentified.
CONSTITUENTS = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL", "META", "AVGO", "PEP"]

# Pinned window so every run sees the same data (yfinance `end` is exclusive).
START = "2013-01-01"
END = "2026-01-01"

# Chronological split by calendar date:
#   train  2013-2020  (includes the 2020 COVID crash)
#   val    2021-2022  (includes the 2022 bear market)
#   test   2023-2025  (never touched until the final evaluation)
TRAIN_END = "2020-12-31"
VAL_END = "2022-12-31"

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prices.csv")

DOW = ["Tue", "Wed", "Thu", "Fri"]
MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REGRESSOR_BLOCKS = {
    "day of week": [f"dow_{d}" for d in DOW],
    "month of year": [f"month_{m}" for m in MONTHS],
    "turn of month": ["turn_of_month"],
    "lagged constituents": [f"lag_{c}" for c in CONSTITUENTS],
}
STOCHASTIC = set(REGRESSOR_BLOCKS["lagged constituents"])


def load_prices(refresh=False):
    """Adjusted closes for the index and constituents, cached to data/prices.csv."""
    if os.path.exists(CACHE) and not refresh:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download([INDEX] + CONSTITUENTS, start=START, end=END,
                      auto_adjust=True, progress=False)
    # Select by name: yfinance orders the columns alphabetically, not as requested.
    prices = raw["Close"][[INDEX] + CONSTITUENTS].dropna()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    prices.to_csv(CACHE)
    return prices


def calendar_regressors(dates):
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=dates)
    for i, name in enumerate(DOW, start=1):
        out[f"dow_{name}"] = (dates.dayofweek == i).astype(float)
    for i, name in enumerate(MONTHS, start=2):
        out[f"month_{name}"] = (dates.month == i).astype(float)
    period = dates.to_period("M")
    new_month = np.r_[True, period[1:] != period[:-1]]
    day_in_month = pd.Series(1, index=dates).groupby(period).cumsum().to_numpy()
    last_of_month = np.r_[new_month[1:], False]
    out["turn_of_month"] = ((day_in_month <= 3) | last_of_month).astype(float)
    return out


@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    y: np.ndarray            # (n,) 100 * log price of the index
    regressors: pd.DataFrame  # (n, all candidates), row t known before day t closes
    n_train: int             # y[:n_train] is train
    n_val: int               # y[n_train:n_val] is validation, y[n_val:] is test

    @property
    def n(self):
        return len(self.y)

    def X(self, columns):
        """Contiguous float array of the chosen regressor columns."""
        if not columns:
            return np.zeros((self.n, 0))
        return np.ascontiguousarray(self.regressors[list(columns)].to_numpy(dtype=np.float64))


def build_dataset(prices):
    logp = 100 * np.log(prices)
    lagged = logp[CONSTITUENTS].diff().shift(1)
    lagged.columns = [f"lag_{c}" for c in CONSTITUENTS]
    frame = pd.concat([logp[INDEX].rename("y"), lagged], axis=1).dropna()
    dates = frame.index
    regressors = pd.concat([calendar_regressors(dates), frame[lagged.columns]], axis=1)
    return Dataset(
        dates=dates,
        y=frame["y"].to_numpy(dtype=np.float64),
        regressors=regressors,
        n_train=int((dates <= TRAIN_END).sum()),
        n_val=int((dates <= VAL_END).sum()),
    )
