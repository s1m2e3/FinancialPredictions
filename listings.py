"""Every US listing, active and DELISTED, with its IPO and delisting dates (Alpha Vantage
LISTING_STATUS): the point-in-time list of what could be traded on any day since 2010 --
a stock is listed on day t if ipoDate <= t < delistingDate. Unlike yfinance, the delisted
names (bankruptcies, buy-outs, failed IPOs) are kept, so a wider universe built on it is
not a list of survivors.

Two requests (the free key allows 25 a day). The key is read from the environment and
never written anywhere; the CSVs are saved as they come:
    data/listings_active.csv      symbol, name, exchange, assetType, ipoDate, delistingDate, status
    data/listings_delisted.csv    the same for every symbol delisted since 2010

    set "ALPHAVANTAGE_API_KEY=<your key>" && python listings.py
"""
import io
import os
import sys
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "https://www.alphavantage.co/query?function=LISTING_STATUS&state={state}&apikey={key}"


def fetch(state, key):
    with urllib.request.urlopen(URL.format(state=state, key=key), timeout=60) as r:
        text = r.read().decode("utf-8")
    if not text.startswith("symbol"):                 # an error or a rate-limit note comes as JSON
        raise SystemExit(f"Alpha Vantage did not return a listing for state={state}: {text[:300]}")
    return pd.read_csv(io.StringIO(text), keep_default_na=False, na_values=["", "null"])


def main():
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        sys.exit('no key: set "ALPHAVANTAGE_API_KEY=<your key>" first (https://www.alphavantage.co/support/#api-key)')
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out = {}
    for state in ("active", "delisted"):
        df = fetch(state, key)
        path = os.path.join(ROOT, "data", f"listings_{state}.csv")
        df.to_csv(path, index=False)
        out[state] = df
        print(f"{state}: {len(df)} symbols -> {path}", flush=True)
        time.sleep(15)                                # the free tier also limits requests per minute
    for state, df in out.items():
        stocks = df[df["assetType"] == "Stock"]
        print(f"\n{state}: {len(stocks)} stocks, {len(df) - len(stocks)} ETFs and others; by exchange:")
        print(stocks["exchange"].value_counts().to_string())
    d = out["delisted"]
    d = d[d["assetType"] == "Stock"]
    years = pd.to_datetime(d["delistingDate"], errors="coerce").dt.year.value_counts().sort_index()
    print("\ndelisted stocks by year:\n" + years.to_string())


if __name__ == "__main__":
    main()
