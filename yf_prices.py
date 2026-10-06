"""Daily prices of the growth pool's still-listed candidates from yfinance, in resumable batches:
data/yf/<TICKER>.csv with Open, High, Low, Close (split-adjusted), Adj Close (splits and
dividends), Volume and Stock Splits -- the splits give back the price as it was quoted that
day, which market value (shares as filed x price) needs.

    python yf_prices.py --sample [N]   the fair sample's listed names (growth_pool.py --sample [--name N])
    python yf_prices.py                every listed candidate (data/growth_candidates.csv)
"""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "data", "yf")


def _sample_file():
    """data/growth_sample<name>.csv for --sample [name] (no name: the first sample)."""
    i = sys.argv.index("--sample") + 1
    name = sys.argv[i] if i < len(sys.argv) and not sys.argv[i].startswith("--") else ""
    return os.path.join(ROOT, "data", f"growth_sample{name}.csv")


def main():
    import yfinance as yf
    os.makedirs(DIR, exist_ok=True)
    if "--sample" in sys.argv:
        s = pd.read_csv(_sample_file())
        todo = s[s["prices_from"] == "yfinance"]["ticker"].dropna().tolist()
    else:
        g = pd.read_csv(os.path.join(ROOT, "data", "growth_candidates.csv"))
        todo = g[g["listed_now"] | g["ticker_trades_now"]]["ticker"].dropna().tolist()
    todo = [t for t in dict.fromkeys(todo) if not os.path.exists(os.path.join(DIR, f"{t}.csv"))]
    print(f"{len(todo)} tickers to fetch", flush=True)
    missing = []
    for i in range(0, len(todo), 25):
        batch = todo[i:i + 25]
        yt = [t.replace(".", "-") for t in batch]
        raw = yf.download(yt, start="2004-01-01", auto_adjust=False, actions=True, group_by="ticker",
                          threads=False, progress=False)
        for t, y in zip(batch, yt):
            try:
                d = raw[y].dropna(subset=["Close"])
            except KeyError:
                d = pd.DataFrame()
            if d.empty:
                missing.append(t)
                continue
            d.to_csv(os.path.join(DIR, f"{t}.csv"))
        print(f"  {min(i + 25, len(todo))}/{len(todo)}", flush=True)
        time.sleep(1.0)
    print(f"done; no prices for {len(missing)}: {missing}" if missing else "done", flush=True)


if __name__ == "__main__":
    main()
