"""Hourly bid and ask bars straight from Dukascopy's public datafeed (datafeed.dukascopy.com, the
server JForex and tools like Tickstory download from) -- no dukascopy-node, whose chart API
(jetta.dukascopy.com) throttles after a few hundred files and then cuts its files off silently.

One file per instrument, side and calendar month: <SYMBOL>/<YYYY>/<MM-1>/<SIDE>_candles_hour_1.bi5,
LZMA-compressed records of 24 bytes, big-endian: seconds from the month's start, open, close,
low, high (integers in points) and volume (float); an hour the market was shut is a flat bar with
zero volume and is dropped. Prices are points / POINT (1000 for yen, forint, gold and silver
quotes, 100000 otherwise), checked against a rough price level per instrument before anything
is written.

    data/dukascopy/feed/<SYMBOL>/<YYYYMM>_<SIDE>.bi5   the raw monthly files (kept; a month with
                                                        no file on the server is <..>.none)
    data/dukascopy/<id>_h1_bid.csv, <id>_h1_ask.csv     merged: timestamp (ms, UTC), open, high,
                                                        low, close, volume -- what macro_data reads

Resumable: every monthly file is saved as it arrives and skipped on a rerun. Requests go
WORKERS at a time; a refused or failed one (HTTP 429 / 5xx / timeout) waits and retries, and
all requests slow down while the server refuses. Speed is printed as it goes. Completed months
only (the current month's file does not exist yet). Also fetches the interest rates
(dukascopy_prices.fetch_rates). The universe is dukascopy_prices.UNIVERSES["fx"] unless
--universe says otherwise.

    python dukascopy_feed.py [--universe fx|macro12] [--workers N] [--from YYYY-MM]
"""
import datetime as dt
import lzma
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import dukascopy_prices as dp

ROOT = os.path.dirname(os.path.abspath(__file__))
FEED = os.path.join(dp.DIR, "feed")
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{side}_candles_hour_1.bi5"
START = "2010-06"            # a year of hourly history before training starts in 2012 (250-day trends)
# rough price level (2020) per instrument: a decoded series whose median is not within 3x of it
# has the wrong point size, and nothing is written
LEVEL = {"eurusd": 1.1, "usdjpy": 108, "gbpusd": 1.28, "audusd": 0.69, "usdcad": 1.34, "usdchf": 0.94,
         "nzdusd": 0.65, "usdnok": 9.4, "usdsek": 9.2, "eurgbp": 0.89, "eurjpy": 121, "eurchf": 1.07,
         "euraud": 1.65, "eurcad": 1.53, "eurnzd": 1.75, "eurnok": 10.7, "eursek": 10.5, "gbpjpy": 137,
         "gbpchf": 1.2, "gbpaud": 1.87, "gbpcad": 1.72, "gbpnzd": 1.98, "audjpy": 74, "audchf": 0.65,
         "audcad": 0.93, "audnzd": 1.06, "nzdjpy": 70, "nzdchf": 0.61, "nzdcad": 0.88, "cadjpy": 80,
         "cadchf": 0.7, "chfjpy": 114, "usdhuf": 310, "eurhuf": 350, "usdmxn": 21, "eurmxn": 24,
         "usdpln": 3.9, "eurpln": 4.44, "usdzar": 16, "eurzar": 18.8, "usdcnh": 6.9, "xauusd": 1770,
         "xagusd": 20, "usa500idxusd": 3200, "usatechidxusd": 10000, "lightcmdusd": 50,
         "brentcmdusd": 55, "coppercmdusd": 2.8}


def point(ident):
    """Points per price unit: 1000 for quotes in yen or forint and for gold and silver, else 100000."""
    return 1e3 if ident[3:] in ("jpy", "huf") or ident[:3] in ("xau", "xag") else 1e5


def _flag(name, default):
    i = sys.argv.index(name) + 1 if name in sys.argv else 0
    return sys.argv[i] if i else default


class Pace:
    """A steady request rate shared by all workers: requests START at least `interval` seconds
    apart. The server's rate limiter answers HTTP 503 to bursts (a few dozen quick requests
    were enough); each refusal widens the interval by half (up to 10 s), each success narrows
    it by 1% (down to MIN_INTERVAL)."""
    MIN_INTERVAL = 0.5

    def __init__(self, interval=1.0):
        self.lock = threading.Lock()
        self.interval, self.next_t = float(interval), 0.0
        self.done = self.bytes = self.refused = self.reconnects = 0
        self.t0 = time.time()

    @property
    def pause(self):
        return self.interval

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)

    def ok(self, n):
        with self.lock:
            self.done += 1
            self.bytes += n
            self.interval = max(self.MIN_INTERVAL, self.interval * 0.99)

    def refuse(self):
        with self.lock:
            self.refused += 1
            self.interval = min(10.0, self.interval * 1.5)

    def line(self, total):
        el = time.time() - self.t0
        return (f"{self.done}/{total} files, {self.bytes / 1e6:.1f} MB, {self.done / max(el, 1) * 60:.0f} files/min, "
                f"{self.refused} refused, {self.reconnects} reconnects, interval {self.interval:.2f}s, {el / 60:.1f} min")


def fetch_month(session, pace, ident, y, m, side):
    """One monthly file into FEED; returns the bytes it saved (0 for a month the server has none)."""
    sym = ident.upper()
    d = os.path.join(FEED, sym)
    path = os.path.join(d, f"{y}{m:02d}_{side}.bi5")
    if os.path.exists(path) or os.path.exists(path[:-4] + ".none"):
        return None
    url = URL.format(sym=sym, y=y, m=m - 1, side=side)
    for attempt in range(12):
        pace.wait()
        try:
            # a NEW connection to this server can take 20-30 s, a request on an open one ~0.3 s:
            # connection trouble is retried at once on the same session (it reconnects) and does
            # not slow the other workers -- a pause longer than the server's keep-alive would make
            # every request pay for a new connection
            r = session.get(url, timeout=(60, 60))
        except Exception as e:
            with pace.lock:
                pace.reconnects += 1
                n = pace.reconnects
            if n <= 5 or n % 50 == 0:
                print(f"    reconnecting: {type(e).__name__} for {sym} {y}-{m:02d} {side} ({n} so far)", flush=True)
            time.sleep(random.uniform(1, 4))
            continue
        if r.status_code == 200:
            os.makedirs(d, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            os.replace(tmp, path)
            pace.ok(len(r.content))
            return len(r.content)
        if r.status_code == 404:
            os.makedirs(d, exist_ok=True)
            open(path[:-4] + ".none", "w").close()
            pace.ok(0)
            return 0
        pace.refuse()                                    # 429, 5xx, anything else: wait and retry
        if pace.refused <= 5 or pace.refused % 50 == 0:
            print(f"    refused: HTTP {r.status_code} for {sym} {y}-{m:02d} {side} (refusal {pace.refused})", flush=True)
        time.sleep(min(120, 15 * (attempt + 1)) * random.uniform(0.8, 1.2))   # this request waits; the others keep the pace
    raise RuntimeError(f"{url}: still refused after 12 tries")


def decode(ident, side, months):
    """The months' files -> (T, 6) float array: timestamp ms, open, high, low, close, volume;
    closed hours (zero volume) dropped."""
    rows = []
    for y, m in months:
        path = os.path.join(FEED, ident.upper(), f"{y}{m:02d}_{side}.bi5")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        raw = lzma.decompress(open(path, "rb").read())
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6).astype(np.int64)
        v = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5].astype(np.float64)
        t0 = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        p = point(ident)
        o, c, lo, hi = a[:, 1] / p, a[:, 2] / p, a[:, 3] / p, a[:, 4] / p
        keep = v > 0
        rows.append(np.column_stack([t0 + a[:, 0] * 1000, o, hi, lo, c, v])[keep])
    return np.concatenate(rows) if rows else np.zeros((0, 6))


def merge(ident, side, months):
    """Write data/dukascopy/<id>_h1_<side>.csv from the monthly files, after the price check."""
    import pandas as pd
    x = decode(ident, side, months)
    name = f"{ident}_h1_{side.lower()}"
    if not len(x):
        print(f"  {name}: no bars", flush=True)
        return False
    med = float(np.median(x[:, 4]))
    ref = LEVEL.get(ident)
    if ref and not (ref / 3 <= med <= ref * 3):
        print(f"  {name}: median price {med:.5g} is not near {ref} -- wrong point size, NOT written", flush=True)
        return False
    have = {(int(t.year), int(t.month)) for t in pd.to_datetime(x[:, 0], unit="ms")}
    missing = [f"{y}{m:02d}" for y, m in months if (y, m) not in have]
    df = pd.DataFrame(x, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    out, mk = os.path.join(dp.DIR, name + ".csv"), os.path.join(dp.DIR, name + ".merged")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        old = open(mk).read() if os.path.exists(mk) else ""
        if not old.startswith("v3"):                        # a dukascopy-node file: kept aside, never deleted
            os.makedirs(dp.PARTS, exist_ok=True)
            os.replace(out, os.path.join(dp.PARTS, f"{name}_prev{time.strftime('%H%M%S')}.csv"))
    df.to_csv(out, index=False)
    with open(mk, "w") as fh:
        print(f"v3 datafeed; months with no bars: {', '.join(missing) or 'none'}", file=fh)
    t = pd.to_datetime(df["timestamp"].iloc[[0, -1]], unit="ms")
    print(f"  {name}: {len(df)} bars {t.iloc[0].date()} .. {t.iloc[1].date()}"
          + (f", no bars in {len(missing)} months" if missing else ""), flush=True)
    return True


def main():
    import requests
    workers = int(_flag("--workers", "1"))
    y0, m0 = map(int, _flag("--from", START).split("-"))
    today = dt.datetime.now(dt.timezone.utc).date()
    months, y, m = [], y0, m0
    while (y, m) < (today.year, today.month):             # completed months only
        months.append((y, m))
        y, m = y + (m == 12), m % 12 + 1
    inst = [(name, ident) for name, (ident, _) in dp.INSTRUMENTS.items()]
    jobs = [(ident, yy, mm, side) for _, ident in inst for side in ("BID", "ASK") for yy, mm in months]
    todo = [j for j in jobs if not (os.path.exists(os.path.join(FEED, j[0].upper(), f"{j[1]}{j[2]:02d}_{j[3]}.bi5"))
                                    or os.path.exists(os.path.join(FEED, j[0].upper(), f"{j[1]}{j[2]:02d}_{j[3]}.none")))]
    print(f"universe {dp.UNIVERSE}: {len(inst)} instruments x 2 sides x {len(months)} months "
          f"({months[0][0]}-{months[0][1]:02d} .. {months[-1][0]}-{months[-1][1]:02d}) = {len(jobs)} files, "
          f"{len(todo)} to fetch, {workers} at a time", flush=True)
    pace = Pace()
    local = threading.local()

    failed = []

    def job(j):
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers["User-Agent"] = "Mozilla/5.0"
        try:
            return fetch_month(local.s, pace, *j)
        except Exception as e:                           # still refused: reported, the run goes on
            failed.append(j)
            print(f"  {j[0]} {j[3]} {j[1]}-{j[2]:02d}: {e}", flush=True)
            return None

    last = time.time()
    by_series = {}
    for ident, yy, mm, side in todo:
        by_series.setdefault((ident, side), 0)
        by_series[(ident, side)] += 1
    left = dict(by_series)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, j): j for j in todo}
        for f in as_completed(futs):
            ident, yy, mm, side = futs[f]
            f.result()
            left[(ident, side)] -= 1
            if left[(ident, side)] == 0 and not any(x[0] == ident and x[3] == side for x in failed):
                merge(ident, side, months)                 # a series complete: merged now
            if time.time() - last > 30:
                print("   ", pace.line(len(todo)), flush=True)
                last = time.time()
    for ident, side in [(i, s) for _, i in inst for s in ("BID", "ASK")]:
        if (ident, side) not in by_series:                 # fetched on an earlier run: merge if not yet
            mk = os.path.join(dp.DIR, f"{ident}_h1_{side.lower()}.merged")
            if not (os.path.exists(mk) and open(mk).read().startswith("v3")):
                merge(ident, side, months)
    print("   ", pace.line(len(todo)), flush=True)
    try:
        dp.fetch_rates()
    except Exception as e:
        print("  rates: FAILED, rerun for them:", e, flush=True)
    print("DONE" if not failed else f"NOT COMPLETE: {len(failed)} files still refused -- rerun to fetch just those",
          flush=True)


if __name__ == "__main__":
    main()
