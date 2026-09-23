"""Point-in-time stock universe and daily prices for the portfolio experiment.

UNIVERSE. On every day the model may only trade stocks that were large and in an index ON
THAT DAY, never a company that was not yet listed, not yet large or not yet in the index,
and never with the knowledge of which ones later won. Two pools:
  S&P 500      the 100 largest S&P 500 members of the day (UNIVERSE_SIZE, EXIT_RANK)
  outside      the 20 largest Nasdaq-100 members of the day that are NOT in the S&P 500
               (OUTSIDE_SIZE, OUTSIDE_EXIT): the growth names the S&P admits late or
               never -- Tesla 2013-2020, Mercado Libre, ASML, Baidu, JD, PDD, Atlassian,
               Monster Beverage before 2012, CrowdStrike and Datadog before 2024-25 ...
               `panel.sp500` (the feature `in_sp500`) tells a tree which pool a stock is in
  membership   S&P 500: `universe/sp500_ticker_start_end.csv`, the index's membership
               spells (github.com/fja05680/sp500, reconstructed from S&P's change
               announcements; bankrupt members carry a Q suffix, e.g. LEHMQ).
               Nasdaq-100: `universe/ndx_snapshots.csv`, the component list of Wikipedia's
               Nasdaq-100 article as it stood on the first day of every quarter since 2005
               (its revision history: what was published then, not rewritten later); a
               snapshot holds until the next one
  symbols      both sources name some companies by a symbol they later dropped (FB, ANTM,
               UTX, BK, RIMM, HANS ...), which yfinance does not know: RENAMED maps each to
               the company's current symbol, whose yfinance history is the same company's.
               A few old symbols now belong to a DIFFERENT company whose prices yfinance
               would return (REUSED); those are dropped, not priced with the wrong series
  size         the trailing 63-day mean dollar volume (close x volume), lagged a day: the
               size and liquidity a trader could see then. Market capitalisation needs
               shares outstanding for delisted members, which no free source has
  selection    on the first trading day of every month the top UNIVERSE_SIZE members by
               size enter; a member already in stays until it falls below rank EXIT_RANK
               (so a name on the border does not churn in and out every month). Between
               rebalances a stock leaves the day it leaves its index or stops trading. The
               outside pool is ranked the same way among its own members; a stock that
               joins the S&P 500 moves to that pool
  dual classes GOOG, FOX, NWS, LBTYK ... are dropped: the same company as GOOGL, FOXA ...

REMAINING BIAS. yfinance drops a ticker once it is delisted, so most members that were
acquired or went bankrupt have no prices (about 32% of the member-days of departed members
are priced; 4 of 20 bankruptcies). Those stocks cannot be traded here, and they were worse
than average, so the tradable set still has a free edge. Measured against RSP (the real
equal-weight S&P 500, which held every member): equal-weighted priced members beat it by
+3.0 %/yr on 2006-2019, +1.4 on 2020-2021 and +0.7 on 2022-2026, against +6.9, +8.4 and
+10.6 for a fixed list of today's S&P 100. The "buy all" baseline carries the same edge, so
trees are judged against it. The complete fix is CRSP (`load_crsp`).
The outside pool: 53-63% of the member-days of Nasdaq-100 members outside the S&P 500 are
priced in 2006-2012 (Yahoo, BEA, Sun, Pixar, NII ... are gone), 73-80% in 2016-2018, 89-99%
from 2019. Equal-weighted priced Nasdaq-100 members beat QQEW (the equal-weight Nasdaq-100
ETF) by +2.9 %/yr on 2006-2019, +3.2 on 2020-2021 and +2.6 on 2022-2026; with nearly every
member priced in the last period, most of that gap is QQEW's fee (~0.6) and its quarterly
rather than daily rebalancing, so the survivorship edge is of the S&P part's order.

Prices are yfinance's split- and dividend-adjusted open, close and volume. Integer share
counts computed on adjusted prices are an approximation of the shares actually traded.
Sectors come from each company's SIC code on EDGAR (the same source for current and former
members), mapped to GICS-like sector names; "Unknown" where EDGAR has no match.

Split (calendar dates), chosen so each period holds a different kind of stress:
    train       2006-2019   (the 2008 crash, 2011, 2015-16, Q4 2018; 2 years of history
                            before it warm the GARCH features up)
    validation  2020-2021   (the COVID crash and rebound)
    test        2022-2026   (the 2022 bear market, the April 2025 drop, the rallies between)
EDGAR fundamentals (fundamentals.py) only start in 2010, so the first training years have
none: the trees can use them from then on.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MEMBERSHIP_FILE = os.path.join(ROOT, "universe", "sp500_ticker_start_end.csv")
MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
SECTOR_FILE = os.path.join(ROOT, "universe", "sectors.csv")
CACHE = os.path.join(ROOT, "data", "pit_ohlcv.pkl")
START, END = "2004-01-01", "2026-09-19"
TRAIN_START, TRAIN_END, VAL_END = "2006-01-01", "2019-12-31", "2021-12-31"
UNIVERSE_START = "2005-01-01"        # the first monthly selection (a year before training)
UNIVERSE_SIZE, EXIT_RANK, SIZE_WINDOW = 100, 120, 63
OUTSIDE_SIZE, OUTSIDE_EXIT = 20, 30
# ^GSPC is the S&P 500 PRICE index (the features model it); the benchmark every portfolio
# is scored against is SPY with dividends reinvested (yfinance's adjusted prices), because the
# stocks are priced the same way: comparing dividend-adjusted stocks with a price index would
# hand every portfolio the index's ~2 %/yr dividend yield for free
MARKET = {"spx": "^GSPC", "vix": "^VIX", "tbill": "^IRX", "bench": "SPY"}
DROP = {"GOOG", "FOX", "NWS", "UA", "DISCK", "DISCB", "BF-A", "LEN-B", "HEI-A",
        "LBTYK", "LBTYB", "LILAK", "BATRK", "LMCK", "TFCF"}
# old symbol -> the same company's current symbol (renames, and the survivor of a merger
# that kept the surviving company's history: UTX -> RTX, BBT -> TFC, DWDP -> DD)
RENAMED = {
    # S&P 500 membership file
    "FB": "META", "ANTM": "ELV", "ABC": "COR", "BHGE": "BKR", "BK": "BNY", "BLL": "BALL",
    "CTL": "LUMN", "DISCA": "WBD", "FLT": "CPAY", "HRS": "LHX", "JEC": "J", "LB": "BBWI",
    "MMC": "MRSH", "MYL": "VTRS", "NLOK": "GEN", "SYMC": "GEN", "PEAK": "DOC", "HCP": "DOC",
    "PKI": "RVTY", "RE": "EG", "TMK": "GL", "UTX": "RTX", "WLTW": "WTW", "CBS": "PSKY",
    "VIAC": "PSKY", "PARA": "PSKY", "BBT": "TFC", "FI": "FISV", "ARNC": "HWM", "DWDP": "DD",
    # Nasdaq-100 snapshots
    "RIMM": "BB", "HANS": "MNST", "PCLN": "BKNG", "ERTS": "EA", "CTRP": "TCOM", "VIP": "VEON",
    "ERICY": "ERIC", "JDSU": "VIAV", "UAUA": "UAL", "AEOS": "AEO", "WFMI": "WFM", "KFT": "MDLZ",
}
# old symbols yfinance now prices as another company (Career Education -> CECO Environmental,
# Smurfit-Stone, Randgold -> Gold.com, Millicom -> Magnum Ice Cream, Genzyme -> a 2008 listing)
REUSED = {"CECO", "SSCC", "GOLD", "MICC", "GENZ"}


def _symbols(s):
    return s.str.replace(".", "-", regex=False).replace(RENAMED)


def membership(refresh=False):
    """S&P 500 membership spells (ticker, start, end); end is NaT while still a member.
    A renamed company's spells under its old and new symbols join into one."""
    if refresh or not os.path.exists(MEMBERSHIP_FILE):
        import requests
        os.makedirs(os.path.dirname(MEMBERSHIP_FILE), exist_ok=True)
        with open(MEMBERSHIP_FILE, "wb") as fh:
            fh.write(requests.get(MEMBERSHIP_URL, timeout=60).content)
    m = pd.read_csv(MEMBERSHIP_FILE, parse_dates=["start_date", "end_date"])
    m["ticker"] = _symbols(m["ticker"])
    return m[~m["ticker"].isin(DROP | REUSED)]


NDX_FILE = os.path.join(ROOT, "universe", "ndx_snapshots.csv")
NDX_WIKI = os.path.join(ROOT, "data", "ndx_wiki")


def _ndx_components(text):
    """[(ticker, name)] from one revision of Wikipedia's Nasdaq-100 article: the bulleted
    "Name (TICKER)" list of the early years or the later constituents table."""
    import re
    head = re.search(r"^==\s*(?:Current )?[Cc]omponents\s*==.*$", text, re.M)
    sec = text[head.end():] if head else text
    nxt = re.search(r"^==", sec, re.M) if head else None
    sec = sec[:nxt.start()] if nxt else sec
    tick = re.compile(r"^[A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z])?$")
    clean = lambda s: re.sub(r"\{\{[^}]*\}\}|<[^>]*>", "", s).strip()
    name = lambda s: (lambda m: m.group(1) if m else clean(s))(re.search(r"\[\[([^|\]]+)", s)).strip()
    out = []
    if "{|" in sec:
        for row in re.split(r"^\|-.*$", sec, flags=re.M)[1:]:
            row = row.strip()
            if not row.startswith("|") or row.startswith("|}"):
                continue
            cells = [c.strip() for c in re.split(r"\|\||\n\|", row.lstrip("|"))]
            tk = [clean(c) for c in cells if tick.match(clean(c))]
            nm = [c for c in cells if "[[" in c]
            if tk and nm:
                out.append((tk[0], name(nm[0])))
    else:
        for line in sec.splitlines():
            if line[:1] in "#*":
                plain = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", lambda m: m.group(1), line)
                tk = re.findall(r"\(([A-Z][A-Z0-9.\-]{0,6})\)", plain)
                if tk:
                    out.append((tk[-1], name(line)))
    return out


def ndx_snapshots(refresh=False):
    """(date, ticker, name): the Nasdaq-100 as Wikipedia listed it at the first day of every
    quarter, from the article's last revision before that day; cached to universe/."""
    if os.path.exists(NDX_FILE) and not refresh:
        return pd.read_csv(NDX_FILE, parse_dates=["date"])
    import json
    import requests
    os.makedirs(NDX_WIKI, exist_ok=True)
    rows = []
    for d in pd.date_range(UNIVERSE_START, END, freq="QS"):
        path = os.path.join(NDX_WIKI, f"{d:%Y-%m-%d}.json")
        if not os.path.exists(path):
            p = dict(action="query", prop="revisions", titles="Nasdaq-100", rvlimit=1, rvdir="older",
                     rvstart=f"{d:%Y-%m-%d}T00:00:00Z", rvprop="content|timestamp|ids", rvslots="main",
                     format="json", formatversion=2)
            rev = requests.get("https://en.wikipedia.org/w/api.php", params=p, timeout=60,
                               headers={"User-Agent": "FinancialPredictions research"}).json()
            rev = rev["query"]["pages"][0]["revisions"][0]
            with open(path, "w") as fh:
                json.dump({"timestamp": rev["timestamp"], "revid": rev["revid"],
                           "text": rev["slots"]["main"]["content"]}, fh)
        with open(path) as fh:
            rev = json.load(fh)
        rows += [(d, rev["timestamp"][:10], t, n) for t, n in _ndx_components(rev["text"])]
    out = pd.DataFrame(rows, columns=["date", "revision", "ticker", "name"])
    os.makedirs(os.path.dirname(NDX_FILE), exist_ok=True)
    out.to_csv(NDX_FILE, index=False)
    return out


def ndx_member(dates, tickers):
    """(T, N) bool Nasdaq-100 membership. A snapshot holds from its date to the next one; a
    revision that lists no components (the list briefly lived on another page) keeps the
    previous snapshot."""
    s = ndx_snapshots()
    s["ticker"] = _symbols(s["ticker"])
    col = {t: i for i, t in enumerate(tickers)}
    snaps = [(d, set(g["ticker"])) for d, g in s.groupby("date")]
    member = np.zeros((len(dates), len(tickers)), bool)
    for k, (d, names) in enumerate(snaps):
        end = snaps[k + 1][0] if k + 1 < len(snaps) else pd.Timestamp(END) + pd.Timedelta(days=1)
        rows = (dates >= d) & (dates < end)
        for t in names:
            if t in col:
                member[rows, col[t]] = True
    return member


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list
    open: np.ndarray          # (T, N) adjusted open, NaN where the stock has no price
    close: np.ndarray         # (T, N)
    volume: np.ndarray        # (T, N)
    universe: np.ndarray      # (T, N) bool: in that day's point-in-time universe (both pools)
    sp500: np.ndarray         # (T, N) bool: an S&P 500 member that day
    spx_open: np.ndarray      # (T,)
    spx_close: np.ndarray     # (T,)
    bench_open: np.ndarray    # (T,) SPY open, dividend-adjusted: the total-return benchmark
    vix: np.ndarray           # (T,) close
    rf: np.ndarray            # (T,) daily risk-free rate (13-week T-bill), decimal
    infl: np.ndarray          # (T,) daily CPI-U inflation, decimal (for real returns; not a feature)
    sectors: list

    def index_of(self, date):
        return int(np.searchsorted(self.dates, pd.Timestamp(date)))


def select_universe(dates, close, volume, member, size=UNIVERSE_SIZE, exit_rank=EXIT_RANK):
    """(T, N) bool point-in-time universe: monthly top `size` by trailing dollar volume
    among that day's priced members, kept until they fall below `exit_rank`."""
    dv = pd.DataFrame(close * volume).rolling(SIZE_WINDOW, min_periods=int(0.6 * SIZE_WINDOW)).mean()
    dv = dv.shift(1).to_numpy()                      # known at the open: up to yesterday
    priced = ~np.isnan(close)
    month = pd.DatetimeIndex(dates).to_period("M")
    first = np.r_[True, month[1:] != month[:-1]] & (dates >= pd.Timestamp(UNIVERSE_START))
    U = np.zeros(close.shape, bool)
    cur = np.zeros(close.shape[1], bool)
    for t in range(len(dates)):
        if first[t]:
            ok = member[t] & ~np.isnan(dv[t])
            order = np.argsort(-np.where(ok, dv[t], -np.inf), kind="stable")
            rank = np.empty(len(order))
            rank[order] = np.arange(1, len(order) + 1)
            keep = cur & ok & (rank <= exit_rank)
            add = [i for i in order if ok[i] and not keep[i]]
            cur = keep.copy()
            cur[add[:max(0, size - int(keep.sum()))]] = True
        U[t] = cur
    return U & member & priced


def sectors(tickers, refresh=False):
    """{ticker: sector} from each company's SIC code on EDGAR, cached to universe/sectors.csv."""
    if os.path.exists(SECTOR_FILE) and not refresh:
        cached = pd.read_csv(SECTOR_FILE)
        if set(tickers) <= set(cached["ticker"]):          # re-mapped, so rule changes apply
            return {r.ticker: sic_sector(r.sic, r.ticker) for r in cached.itertuples()}
    from fundamentals import _get
    cik = {r["ticker"].replace(".", "-"): r["cik_str"]
           for r in _get("https://www.sec.gov/files/company_tickers.json").values()}
    rows = []
    for t in tickers:
        sic = None
        if t in cik:
            sub = _get(f"https://data.sec.gov/submissions/CIK{cik[t]:010d}.json") or {}
            sic = int(sub["sic"]) if str(sub.get("sic", "")).isdigit() else None
        rows.append(dict(ticker=t, sic=sic, sector=sic_sector(sic, t)))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SECTOR_FILE), exist_ok=True)
    out.to_csv(SECTOR_FILE, index=False)
    return out.set_index("ticker")["sector"].to_dict()


# SIC 7389 ("business services, not elsewhere classified") holds payment networks, IT
# consultancies and internet platforms alike, and a few large names sit in an SIC far from
# their business; these are named instead of ruled
SECTOR_OVERRIDE = {
    **dict.fromkeys(["V", "MA", "PYPL", "FIS", "FISV", "GPN", "FI"], "Financials"),
    **dict.fromkeys(["ACN", "AKAM", "LRCX", "KLAC"], "Information Technology"),
    **dict.fromkeys(["UBER", "DASH", "EBAY", "ETSY", "ABNB"], "Consumer Discretionary"),
    **dict.fromkeys(["GOOGL", "META"], "Communication Services"),
}


def sic_sector(sic, ticker=None):
    """A GICS-like sector for a 4-digit SIC code: coarse, but one rule for every company."""
    if ticker in SECTOR_OVERRIDE:
        return SECTOR_OVERRIDE[ticker]
    if sic is None or pd.isna(sic):
        return "Unknown"
    sic = int(sic)
    d2 = sic // 100
    if d2 == 15 or sic in (3021, 4400, 4700, 7340):          # homebuilders, footwear, cruises, travel
        return "Consumer Discretionary"
    if sic == 7320:                                             # credit ratings and data
        return "Financials"
    if sic in (2834, 2835, 2836, 5122, 6324) or 3841 <= sic <= 3851 or d2 == 80 or sic == 8731:
        return "Health Care"
    if 3570 <= sic <= 3579 or 3660 <= sic <= 3679 or 7370 <= sic <= 7379 or 3812 <= sic <= 3829:
        return "Information Technology"
    if sic == 6798:
        return "Real Estate"
    if 60 <= d2 <= 67:
        return "Financials"
    if d2 in (13, 29) or sic in (4610, 4612, 4613, 4922, 4923):
        return "Energy"
    if d2 == 49:
        return "Utilities"
    if d2 in (27, 48) or 7810 <= sic <= 7841 or sic == 7900:
        return "Communication Services"
    if d2 in (20, 21) or sic in (2844, 5140, 5141, 5411, 5912, 5331, 5399):
        return "Consumer Staples"
    if d2 in (1, 2, 8, 9, 10, 12, 14, 24, 26, 28, 32, 33):
        return "Materials"
    if d2 in (23, 25, 31, 39, 55, 56, 57, 58, 59, 70, 72, 79) or 5200 <= sic <= 5399 \
            or sic in (3711, 3714, 3751, 5961):
        return "Consumer Discretionary"
    return "Industrials"


def _prices(tickers, refresh=False):
    """yfinance daily OHLCV for `tickers`, cached to data/pit_ohlcv.pkl; only the tickers
    the cache lacks are downloaded (a symbol yfinance does not know is kept as an empty
    column, so it is not asked for again)."""
    import yfinance as yf
    raw = None if refresh or not os.path.exists(CACHE) else pd.read_pickle(CACHE)
    have = set() if raw is None else set(raw.columns.get_level_values(1))
    missing = sorted(set(tickers) - have)
    if missing:
        print(f"downloading {len(missing)} tickers from yfinance", flush=True)
        new = yf.download(missing, start=START, end=END, auto_adjust=True, progress=False, threads=True)
        fields = ["Close", "High", "Low", "Open", "Volume"]
        new = new.reindex(columns=pd.MultiIndex.from_product([fields, missing]))
        raw = new if raw is None else pd.concat([raw, new], axis=1).sort_index(axis=1)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        raw.to_pickle(CACHE)
    return raw


def load_panel(refresh=False):
    """Daily point-in-time panel on the S&P 500's trading calendar: every stock that is
    ever in either pool of the universe."""
    m = membership()
    live = m[m["end_date"].isna() | (m["end_date"] >= pd.Timestamp(UNIVERSE_START))]
    ndx = set(_symbols(ndx_snapshots()["ticker"])) - DROP - REUSED
    raw = _prices(sorted(set(live["ticker"]) | ndx) + list(MARKET.values()), refresh)
    spx = raw["Close"][MARKET["spx"]]
    dates = spx.dropna().index                       # the exchange calendar
    names = set(m["ticker"]) | ndx
    cols = [t for t in raw["Close"].columns if t in names and raw["Close"][t].notna().any()]
    get = lambda field: raw[field].reindex(dates)[cols].to_numpy(dtype=np.float64)
    close, volume = get("Close"), get("Volume")
    sp = np.zeros(close.shape, bool)
    col = {t: i for i, t in enumerate(cols)}
    for r in m.itertuples():
        if r.ticker in col:
            e = r.end_date if pd.notna(r.end_date) else pd.Timestamp(END)
            sp[(dates >= r.start_date) & (dates <= e), col[r.ticker]] = True
    outside = ndx_member(dates, cols) & ~sp
    U = (select_universe(dates, close, volume, sp)
         | select_universe(dates, close, volume, outside, OUTSIDE_SIZE, OUTSIDE_EXIT))
    keep = U.any(axis=0)                             # only stocks that are ever tradable
    tickers = [t for t, k in zip(cols, keep) if k]
    tbill = raw["Close"][MARKET["tbill"]].reindex(dates).ffill().bfill().to_numpy()
    sec = sectors(tickers)
    return Panel(
        dates=dates, tickers=tickers,
        open=get("Open")[:, keep], close=close[:, keep], volume=volume[:, keep],
        universe=U[:, keep], sp500=sp[:, keep],
        spx_open=raw["Open"][MARKET["spx"]].reindex(dates).to_numpy(),
        spx_close=spx.reindex(dates).to_numpy(),
        bench_open=raw["Open"][MARKET["bench"]].reindex(dates).ffill().to_numpy(),
        vix=raw["Close"][MARKET["vix"]].reindex(dates).ffill().to_numpy(),
        rf=tbill / 100.0 / 252.0,
        infl=daily_inflation(dates, load_cpi()),
        sectors=[sec.get(t, "Unknown") for t in tickers],
    )


CPI_CACHE = os.path.join(ROOT, "data", "cpi_u.csv")


def load_cpi(refresh=False):
    """US CPI-U, all items, not seasonally adjusted (BLS series CUUR0000SA0), monthly.

    From the BLS public API v1 (no key; 10 years per request), cached to data/cpi_u.csv.
    Not seasonally adjusted because it deflates realised returns: the price level that
    actually prevailed each month.
    """
    import pandas as pd
    if os.path.exists(CPI_CACHE) and not refresh:
        return pd.read_csv(CPI_CACHE, index_col=0, parse_dates=True)["cpi"]
    import requests
    rows = []
    for y0 in range(int(START[:4]) - 1, int(END[:4]), 10):
        r = requests.post("https://api.bls.gov/publicAPI/v1/timeseries/data/",
                          json={"seriesid": ["CUUR0000SA0"], "startyear": str(y0),
                                "endyear": str(min(y0 + 9, int(END[:4]) - 1))}, timeout=60).json()
        if r["status"] != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API: {r.get('message')}")
        # "-" marks a month BLS did not publish (October 2025 was never collected during
        # the federal shutdown); it is filled by log-linear interpolation below
        rows += [(pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), float(d["value"]))
                 for d in r["Results"]["series"][0]["data"]
                 if d["period"] != "M13" and d["value"] not in ("-", "")]
    cpi = pd.Series(dict(rows), name="cpi").sort_index()
    months = pd.date_range(cpi.index[0], cpi.index[-1], freq="MS")
    cpi = np.exp(np.log(cpi).reindex(months).interpolate()).rename("cpi")
    os.makedirs(os.path.dirname(CPI_CACHE), exist_ok=True)
    cpi.to_frame().to_csv(CPI_CACHE)
    return cpi


def daily_inflation(dates, cpi):
    """Each month's CPI change spread evenly over its trading days: (CPI_m / CPI_m-1)^(1/n_m) - 1."""
    import pandas as pd
    monthly = cpi / cpi.shift(1)
    period = pd.DatetimeIndex(dates).to_period("M")
    n_days = pd.Series(1, index=period).groupby(level=0).transform("count").to_numpy()
    growth = monthly.reindex(period.to_timestamp()).to_numpy()
    growth = np.where(np.isnan(growth), 1.0, growth)
    return growth ** (1.0 / n_days) - 1.0


def load_crsp(*_, **__):
    """Survivorship-free replacement for load_panel (not implemented yet).

    With WRDS access: daily prices and returns from crsp.dsf (prc, openprc, ret, vol,
    cfacpr for adjustment), S&P 500 membership history from crsp.dsp500list, the S&P 100
    approximated point-in-time as the 100 largest members by market cap (prc * shrout) at
    each rebalance date, and delisting returns from crsp.dsedelist. It must return a Panel
    with the same fields so the features, environment and baselines run unchanged.
    """
    raise NotImplementedError("CRSP loader pending WRDS access; see the docstring")
