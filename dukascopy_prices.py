"""Hourly bid and ask bars of the multi-asset universe from Dukascopy (free, via the MIT-licensed
dukascopy-node CLI run through npx), daily bars for the long-window features, and the 3-month
interbank rates that give each currency pair its carry (FRED / OECD, monthly).

    data/dukascopy/<id>_h1_bid.csv, <id>_h1_ask.csv   hourly OHLC + volume, UTC, from 2010
    data/dukascopy/<id>_d1_bid.csv                    daily OHLC, UTC days, from 2000
    data/dukascopy/rates.csv                          3-month rates, % per year, monthly
    data/dukascopy/parts/                             the hourly bars by year or month, merged above

WHAT ARRIVED IS CHECKED, NOT WHAT THE TOOL SAYS. dukascopy-node ends a file silently when a
batch is throttled (EUR/USD came back "ok" ending 2016-08-31) and fails a whole request when one
month in it has no data (the US 500 has such months in 2011-2013). So each series: the whole
range in one request, continued from its last bar where it stops early; then every calendar
month must hold its bars, and a short one is fetched on its own (hourly: once more as 30-minute
bars, Dukascopy's per-day files, made hourly here). A month the tool reports no data for every
way is a HOLE in Dukascopy's history: listed in <series>.merged and left out. A request refused
for load (HTTP 429, network errors, no file and no message) waits and retries; a series with a
month still refused is not merged -- rerun, and only what is missing is fetched.

    python dukascopy_prices.py [--universe fx|macro12] [--workers N]     (default: fx, 1 series at a time)
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "data", "dukascopy")
PARTS = os.path.join(DIR, "parts")
PKG = "dukascopy-node@1.50.0"          # pinned: the version checked on npm (MIT, Leo4815162342/dukascopy-node)
H1_FROM, D1_FROM = "2010-01-01", "2000-01-01"

# UNIVERSES: name -> (Dukascopy id, asset class). Hourly history (dukascopy-node's instrument
# metadata, startMonthForHourlyCandles):
#   fx       (default) the 32 G10 currency pairs with hourly bars since 2003-2006, eight
#            emerging-market pairs since 2007 (forint, peso, zloty, rand vs the dollar and the
#            euro), dollar / offshore yuan since June 2012, gold and silver: 43 instruments with
#            no empty months. Left out: the lira and the Singapore dollar (no 3-month rate on
#            FRED for them since 2008 -- their carry, 15-40%/yr for the lira, cannot be priced),
#            pegged pairs (HKD, DKK) and the 13 G10 crosses that start in 2023-2024.
#   macro12  the first plan: US 500, US Tech 100, gold, silver, WTI, Brent, copper, five dollar
#            pairs (the index CFDs have empty months in 2011-2013)
G10_PAIRS = ["eurusd", "usdjpy", "gbpusd", "audusd", "usdcad", "usdchf", "nzdusd", "usdnok", "usdsek",
             "eurgbp", "eurjpy", "eurchf", "euraud", "eurcad", "eurnzd", "eurnok", "eursek",
             "gbpjpy", "gbpchf", "gbpaud", "gbpcad", "gbpnzd", "audjpy", "audchf", "audcad", "audnzd",
             "nzdjpy", "nzdchf", "nzdcad", "cadjpy", "cadchf", "chfjpy"]
EM_PAIRS = ["usdhuf", "eurhuf", "usdmxn", "eurmxn", "usdpln", "eurpln", "usdzar", "eurzar", "usdcnh"]
UNIVERSES = {
    "fx": {"GOLD": ("xauusd", "metal"), "SILVER": ("xagusd", "metal"),       # in download order: metals,
           **{p.upper(): (p, "fx") for p in G10_PAIRS + EM_PAIRS}},       # majors, crosses, emerging
    "macro12": {"US500": ("usa500idxusd", "equity"), "USTECH": ("usatechidxusd", "equity"),
                "GOLD": ("xauusd", "metal"), "SILVER": ("xagusd", "metal"), "WTI": ("lightcmdusd", "energy"),
                "BRENT": ("brentcmdusd", "energy"), "COPPER": ("coppercmdusd", "metal"),
                "EURUSD": ("eurusd", "fx"), "USDJPY": ("usdjpy", "fx"), "GBPUSD": ("gbpusd", "fx"),
                "AUDUSD": ("audusd", "fx"), "USDCAD": ("usdcad", "fx")},
}
# daily bars only, for a universe's market features (the fx universe reads the US 500's day)
DAILY_EXTRA = {"fx": {"US500": "usa500idxusd"}, "macro12": {}}


def _universe_arg():
    i = sys.argv.index("--universe") + 1 if "--universe" in sys.argv else 0
    name = sys.argv[i] if i else "fx"
    if name not in UNIVERSES:
        raise SystemExit(f"--universe must be one of {', '.join(UNIVERSES)}")
    return name


UNIVERSE = _universe_arg()
INSTRUMENTS = UNIVERSES[UNIVERSE]
# the first hourly bar where it is after H1_FROM
H1_START = {"usa500idxusd": "2011-09-18", "usatechidxusd": "2011-09-18", "lightcmdusd": "2011-09-23",
            "brentcmdusd": "2010-12-02", "coppercmdusd": "2012-03-02", "usdcnh": "2012-06-26"}
# OECD 3-month interbank rates on FRED, % per year, monthly: a currency pair's carry is the
# rate of the currency held long minus the rate of the one held short (CNH: China's rate)
RATES = {"USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
         "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
         "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N", "SEK": "IR3TIB01SEM156N",
         "NOK": "IR3TIB01NOM156N", "HUF": "IR3TIB01HUM156N", "MXN": "IR3TIB01MXM156N",
         "PLN": "IR3TIB01PLM156N", "ZAR": "IR3TIB01ZAM156N", "CNH": "IR3TIB01CNM156N"}
# refused for load, not for missing data: wait and retry
BUSY = ("429", "status 5", "ECONNRESET", "ETIMEDOUT", "EAI_AGAIN", "socket hang up", "ENOTFOUND")
WAITS = (120, 300, 600, 900)           # seconds before each retry of a refused request (the API throttles by IP)
M30_LOCK = threading.Lock()            # 30-minute requests (~30 files each): one at a time


def _npx():
    exe = shutil.which("npx") or shutil.which("npx.cmd")
    if exe is None:
        raise SystemExit("npx not found: Node.js is needed (https://nodejs.org)")
    return exe


def _run(ident, timeframe, side, start, end, name, where):
    """One dukascopy-node request into where/<name>.csv: "ok" (a non-empty file and no error --
    judged by the file, not the exit code: Node on Windows can crash on exit after writing it),
    "none" (the tool reported an error: no data there) or "busy" (refused for load, or no file
    and no error at all, after the retries). An "ok" file may still END EARLY: a throttled batch
    ends it silently -- fetch_range checks where it stopped."""
    out = os.path.join(where, name + ".csv")
    # gentle: Dukascopy's chart API (jetta.dukascopy.com, what dukascopy-node calls) answers 429
    # to bursts -- 3 series of 30-minute requests at 10 files a batch were enough
    gentle = ["-bs", "3", "-bp", "3000"] if timeframe.startswith("m") else ["-bs", "5", "-bp", "1500"]
    cmd = [_npx(), "-y", PKG, "-i", ident, "-from", start, "-to", end, "-t", timeframe, "-p", side,
           "-f", "csv", "-v", "-dir", where, "-fn", name, "-r", "3", "-rp", "2000"] + gentle
    for attempt in range(len(WAITS) + 1):
        res = subprocess.run(cmd, capture_output=True, text=True)
        text = (res.stdout or "") + (res.stderr or "")
        if os.path.exists(out) and os.path.getsize(out) > 0 and "Something went wrong" not in text:
            return "ok"
        why = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(">")]
        why = why[-1] if why else "no file, no message"
        if "Something went wrong" in text and not any(b in text for b in BUSY):
            print(f"      {name}: no data ({why})", flush=True)
            return "none"
        if attempt < len(WAITS):
            print(f"      {name}: refused ({why}) -- waiting {WAITS[attempt]}s", flush=True)
            time.sleep(WAITS[attempt])
    print(f"      {name}: still refused -- left for a rerun", flush=True)
    return "busy"


def _month_iter(first, today):
    """(YYYYMM, start, end) of every calendar month overlapping [first, today)."""
    import datetime as dt
    y, m = first.year, first.month
    while dt.date(y, m, 1) < today:
        nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
        yield f"{y}{m:02d}", max(dt.date(y, m, 1), first), min(nxt, today), (nxt - dt.date(y, m, 1)).days
        y, m = y + (m == 12), m % 12 + 1


def _to_hourly(df):
    """30-minute bars -> hourly bars (open of the first, high, low, close of the last, volumes summed)."""
    import pandas as pd
    ts = df.columns[0]
    t = (pd.to_datetime(df[ts], unit="ms") if pd.api.types.is_numeric_dtype(df[ts]) else pd.to_datetime(df[ts]))
    df = df.assign(_h=t.dt.floor("h")).sort_values(ts)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    out = df.groupby("_h").agg(agg).reset_index()
    stamp = (((out["_h"] - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64")
             if pd.api.types.is_numeric_dtype(df[ts]) else out["_h"])
    return pd.concat([stamp.rename(ts), out[list(agg)]], axis=1)


def _read(path):
    """A piece as bars (30-minute pieces made hourly); an empty file is no bars."""
    import pandas as pd
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path)
    return _to_hourly(df) if "_m30" in os.path.basename(path) else df


def _stamps(df):
    """A piece's bar times, integer milliseconds UTC."""
    import numpy as np
    import pandas as pd
    if not len(df):
        return np.zeros(0, np.int64)
    c = df.columns[0]
    if pd.api.types.is_numeric_dtype(df[c]):
        return df[c].to_numpy(np.int64)
    return ((pd.to_datetime(df[c]) - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy(np.int64)


def _ms(d):
    import datetime as dt
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


MIN_BARS = {"h1": 300, "d1": 15}       # bars a full calendar month must hold (a currency month: ~500 / ~22)
SLACK_DAYS = 5                         # a piece ending this close to its requested end is complete


def fetch_series(ident, tf, side, first, today):
    """One series into DIR/<id>_<tf>_<side>.csv, checked by what actually arrived:

    1. the whole range in one request; where it stops early (a throttled batch ends the file
       silently -- EUR/USD came back "ok" ending 2016-08-31), a new request from its last bar,
       until it reaches the end or stops moving;
    2. every calendar month must then hold MIN_BARS (scaled for a partial first or last month);
       a short month is fetched on its own, and for hourly bars once more as 30-minute bars
       (Dukascopy's per-day files instead of the per-month one), made hourly here;
    3. a month still short after that, with the tool reporting no data, is a HOLE in Dukascopy's
       history (the US 500 has some in 2011-2013): listed and merged without it. A month refused
       for load is not a hole: the series is then NOT merged, and a rerun fetches only what is
       missing.

    Pieces live in PARTS and are kept; a merged file whose marker is not this version's, and a
    cut-off file from an earlier run (first_run_partial/), count as pieces too. Returns what is
    still refused ([] once merged)."""
    import datetime as dt
    import numpy as np
    import pandas as pd
    name = f"{ident}_{tf}_{side}"
    out = os.path.join(DIR, name + ".csv")
    mark = os.path.join(DIR, name + ".merged")
    if os.path.exists(out) and os.path.exists(mark) and open(mark).read().startswith("v2"):
        print(f"  {name}: already there", flush=True)
        return []
    if os.path.exists(out) and os.path.getsize(out) > 0:       # an earlier version's file: one more piece
        os.replace(out, os.path.join(PARTS, f"{name}_prev{time.strftime('%H%M%S')}.csv"))
    t0 = time.time()
    first = dt.date.fromisoformat(first) if isinstance(first, str) else first

    def all_pieces():
        """Every piece of this series, in merge order: later wins a duplicate bar."""
        return (sorted(glob.glob(os.path.join(DIR, "first_run_partial", f"{name}*.csv")))
                + sorted(glob.glob(os.path.join(PARTS, f"{name}_*.failed")))
                + sorted(p for p in glob.glob(os.path.join(PARTS, f"{name}_*.csv")) if "_m30" in p)
                + sorted(p for p in glob.glob(os.path.join(PARTS, f"{name}_*.csv")) if "_m30" not in p))

    stamps = set()
    for p in all_pieces():
        stamps.update(_stamps(_read(p)).tolist())

    def get(tf_, a, b, piece):
        """(status, bar times) of a piece over [a, b); a piece that stops early is continued
        from its last bar (piece_c1, piece_c2, ...)."""
        got, cur, k, last_end = set(), a, 0, None
        while cur < b:
            p = piece + (f"_c{k}" if k else "")
            path = os.path.join(PARTS, p + ".csv")
            if not (os.path.exists(path) and os.path.getsize(path) > 0):
                if tf_ == "m30":
                    with M30_LOCK:
                        st = _run(ident, tf_, side, cur.isoformat(), b.isoformat(), p, PARTS)
                else:
                    st = _run(ident, tf_, side, cur.isoformat(), b.isoformat(), p, PARTS)
                if st == "busy":
                    return st, got
                if st == "none":
                    # the tool died at a month with no data: keep what came, go on AFTER that month
                    s = _stamps(_read(path)) if os.path.exists(path) else np.zeros(0, np.int64)
                    got.update(s.tolist())
                    dead = (dt.datetime.fromtimestamp(s.max() / 1000, dt.timezone.utc).date() + dt.timedelta(days=1)
                            if len(s) else cur)
                    nxt = dt.date(dead.year + (dead.month == 12), dead.month % 12 + 1, 1)
                    if nxt >= b or k >= 60:
                        return ("ok" if got else "none"), got
                    print(f"      {p}: no data from {dead:%Y-%m} -- going on from {nxt}", flush=True)
                    cur, k = nxt, k + 1
                    continue
            s = _stamps(_read(path))
            got.update(s.tolist())
            last = dt.datetime.fromtimestamp(s.max() / 1000, dt.timezone.utc).date() if len(s) else cur
            if last >= b - dt.timedelta(days=SLACK_DAYS) or (last_end is not None and last <= last_end):
                return "ok", got
            print(f"      {p}: stops at {last} -- continuing from there", flush=True)
            last_end, cur, k = last, last, k + 1
        return "ok", got

    # 1. the whole range -- or, with pieces already here (a cut-off earlier download), from their
    # last bar on -- in large requests, continued where each stops
    last = dt.datetime.fromtimestamp(max(stamps) / 1000, dt.timezone.utc).date() if stamps else None
    start = first if last is None else max(first, last)
    if start < today - dt.timedelta(days=SLACK_DAYS):
        st, got = get(tf, start, today, f"{name}_all" if last is None else f"{name}_from{start:%Y%m%d}")
        stamps |= got
        if st == "busy":
            print(f"  {name}: NOT merged -- refused; rerun", flush=True)
            return [name]
    if tf == "d1" and stamps:                    # a daily series starts where Dukascopy's does
        first = max(first, dt.datetime.fromtimestamp(min(stamps) / 1000, dt.timezone.utc).date())
    # 2. every month must hold its bars
    arr = np.array(sorted(stamps), np.int64)
    in_month = lambda a, b: int(np.searchsorted(arr, _ms(b)) - np.searchsorted(arr, _ms(a)))
    holes, busy = [], []
    for tag, a, b, n_days in _month_iter(first, today):
        share = (b - a).days / n_days
        need = MIN_BARS[tf] * share
        if share < 0.3 or in_month(a, b) >= need:     # a sliver of a month at either end is not judged
            continue
        status = []
        for tf_, piece in [(tf, f"{name}_{tag}")] + ([("m30", f"{name}_{tag}_m30")] if tf == "h1" else []):
            st, got = get(tf_, a, b, piece)
            status.append(st)
            if got:
                stamps |= got
                arr = np.array(sorted(stamps), np.int64)
            if in_month(a, b) >= need:
                break
        else:
            (busy if "busy" in status else holes).append(tag)
    if busy:
        print(f"  {name}: NOT merged -- {len(busy)} months refused ({', '.join(busy)}); rerun", flush=True)
        return busy
    # 3. merge
    frames = [f for f in (_read(p) for p in all_pieces()) if len(f)]
    if not frames:
        print(f"  {name}: FAILED, nothing downloaded", flush=True)
        return [name]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=df.columns[0], keep="last").sort_values(df.columns[0])
    df.to_csv(out, index=False)
    with open(mark, "w") as fh:
        print("v2 coverage-checked; no data in: " + (", ".join(holes) or "none"), file=fh)
    s = _stamps(df)
    span = [dt.datetime.fromtimestamp(x / 1000, dt.timezone.utc).date() for x in (s.min(), s.max())]
    print(f"  {name}: ok, {len(df)} bars {span[0]} .. {span[1]} in {time.time() - t0:.0f}s"
          + (f" -- no data in {len(holes)} months: {', '.join(holes)}" if holes else ""), flush=True)
    return []


def fetch_rates():
    import pandas as pd
    out = os.path.join(DIR, "rates.csv")
    if os.path.exists(out):
        if set(RATES) <= set(pd.read_csv(out, index_col=0, nrows=1).columns):
            print("  rates.csv: already there", flush=True)
            return True
        os.replace(out, os.path.join(DIR, f"rates_{time.strftime('%Y%m%d_%H%M%S')}.csv"))   # fewer currencies: kept aside
    cols = {}
    for ccy, sid in RATES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as fh:
            df = pd.read_csv(fh, index_col=0, parse_dates=True, na_values=".")
        cols[ccy] = df.iloc[:, 0]
        print(f"  rate {ccy} ({sid}): {df.index[0].date()} .. {df.index[-1].date()}", flush=True)
    pd.DataFrame(cols).to_csv(out)
    return True


def main(workers=1):
    """Every series, `workers` at a time (--workers N): each is its own sequence of requests."""
    import datetime as dt
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(PARTS, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()             # requests end before today (UTC)
    h1_first = lambda i: max(H1_FROM, H1_START.get(i, H1_FROM))
    print(f"universe {UNIVERSE}: {len(INSTRUMENTS)} instruments, {workers} series at a time", flush=True)
    jobs = []
    for name, (ident, _) in INSTRUMENTS.items():
        jobs += [lambda i=ident, s=side: fetch_series(i, "h1", s, h1_first(i), today) for side in ("bid", "ask")]
        jobs.append(lambda i=ident: fetch_series(i, "d1", "bid", D1_FROM, today))
    jobs += [lambda i=ident: fetch_series(i, "d1", "bid", D1_FROM, today) for ident in DAILY_EXTRA[UNIVERSE].values()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [x for res in pool.map(lambda job: job(), jobs) for x in res]
    try:
        fetch_rates()
    except Exception as e:                       # the prices are still usable; rerun for the rates
        failed.append(f"rates ({e})")
    print("\nDONE" if not failed else f"\nNOT COMPLETE (rerun to retry just these): {', '.join(failed)}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    w = sys.argv.index("--workers") + 1 if "--workers" in sys.argv else 0
    sys.exit(main(int(sys.argv[w]) if w else 1))
