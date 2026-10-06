"""Daily prices of the growth pool's DELISTED candidates from Tiingo (growth_pool.py lists
them: data/growth_candidates.csv, need_tiingo), in small resumable batches.

Each run fetches at most --limit tickers it does not have yet, one request each, --pause
seconds apart (Tiingo's free tier allows about 50 requests an hour and 500 new tickers a
month; check your plan), and saves each as data/tiingo/<TICKER>.csv as it arrives: stop it
any time and run it again to go on. It uses almost no memory or CPU. A rate-limit answer
stops the run cleanly.

The token is read from the environment and sent in a request header, never written anywhere:

    set "TIINGO_API_KEY=<your token>" && python tiingo_prices.py --tickers NKLA,TWTR,ATVI     a first test
    set "TIINGO_API_KEY=<your token>" && python tiingo_prices.py --limit 40                   the next batch
    set "TIINGO_API_KEY=<your token>" && python tiingo_prices.py --sample --limit 50          the fair sample (growth_pool.py --sample)
    set "TIINGO_API_KEY=<your token>" && python tiingo_prices.py --sample 2 --limit 50        the second sample (--name 2)
"""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "data", "tiingo")
URL = "https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2004-01-01&format=csv&resampleFreq=daily"


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _sample_file():
    """data/growth_sample<name>.csv for --sample [name] (no name: the first sample)."""
    i = sys.argv.index("--sample") + 1
    name = sys.argv[i] if i < len(sys.argv) and not sys.argv[i].startswith("--") else ""
    return os.path.join(ROOT, "data", f"growth_sample{name}.csv")


def main():
    import requests
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        sys.exit('no token: set "TIINGO_API_KEY=<your token>" first (tiingo.com -> Account -> API)')
    os.makedirs(DIR, exist_ok=True)
    if _arg("--tickers"):
        todo = [t.strip().upper() for t in _arg("--tickers").split(",") if t.strip()]
    elif "--sample" in sys.argv:                      # the fair sample's gone companies (growth_pool.py --sample)
        s = pd.read_csv(_sample_file())
        todo = s[s["prices_from"] == "tiingo"].sort_values("max_float_bn", ascending=False)["ticker"].tolist()
    else:
        g = pd.read_csv(os.path.join(ROOT, "data", "growth_candidates.csv"))
        todo = g[g["need_tiingo"]].sort_values("max_float_bn", ascending=False)["ticker"].tolist()
    missing_log = os.path.join(DIR, "_not_found.txt")
    skip = set(open(missing_log).read().split()) if os.path.exists(missing_log) else set()
    todo = [t for t in todo if t not in skip and not os.path.exists(os.path.join(DIR, f"{t}.csv"))]
    limit, pause = int(_arg("--limit", 40)), float(_arg("--pause", 75))
    print(f"{len(todo)} tickers still to fetch; this run: {min(limit, len(todo))}, {pause:g}s apart", flush=True)
    headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
    for n, t in enumerate(todo[:limit]):
        if n:
            time.sleep(pause)
        r = requests.get(URL.format(t=t.lower()), headers=headers, timeout=60)
        if r.status_code in (401, 403):
            sys.exit(f"Tiingo refused the token ({r.status_code}): check TIINGO_API_KEY")
        if r.status_code == 429 or "rate limit" in r.text[:300].lower():
            sys.exit(f"rate limit reached after {n} tickers: run again later, it resumes where it stopped")
        if r.status_code == 404 or not r.text.startswith("date"):
            with open(missing_log, "a") as fh:
                fh.write(t + "\n")
            print(f"  {t}: not found ({r.status_code}) {r.text[:120]!r}", flush=True)
            continue
        path = os.path.join(DIR, f"{t}.csv")
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(r.text)
        os.replace(path + ".tmp", path)
        rows = r.text.count("\n") - 1
        first, last = r.text.split("\n")[1][:10], r.text.strip().split("\n")[-1][:10]
        print(f"  {t}: {rows} days, {first} .. {last}  ({n + 1}/{min(limit, len(todo))})", flush=True)
    left = len(todo) - min(limit, len(todo))
    print(f"done; {left} tickers left for later runs" if left else "all tickers fetched", flush=True)


if __name__ == "__main__":
    main()
