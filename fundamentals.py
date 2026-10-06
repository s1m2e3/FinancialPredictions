"""Point-in-time fundamentals from SEC EDGAR, as known at the OPEN of each trading day.

SOURCE. The XBRL "company facts" API (data.sec.gov/api/xbrl/companyfacts), free and
keyless. The SEC requires a User-Agent that names the requester and a contact address:
set SEC_USER_AGENT, e.g.  "Jane Doe jane@example.edu". XBRL is mandatory for large filers
from mid-2009, so nothing here exists before then; the first trailing-twelve-month (TTM)
values appear in 2010.

POINT IN TIME. Every XBRL value carries the date its filing reached EDGAR (`filed`). A
value enters the panel on the first trading day AFTER that date (filings often land after
the close), and for each reporting period only the FIRST filed value is used: later
restatements are what the company said afterwards, not what investors knew then. A value
older than STALE_DAYS (no new filing) is dropped.

THE MEASURE. EBTDA -- earnings after interest, before taxes, depreciation and
amortisation:
    EBTDA = pre-tax income + depreciation & amortisation     (= EBITDA - interest)
TTM from any period that ends at e, with durations matched to within a few days:
    annual report           TTM(e) = FY(e)
    3, 6 or 9 months YTD    TTM(e) = YTD(e) + FY(prior year) - YTD(e - 1 year)
available once every one of those values has been filed.

NORMALISED, because raw EBTDA mostly measures size, and a level that is good in one
sector or one decade is ordinary in another:
    ebtda_roa       TTM EBTDA / total assets at e
    ebtda_margin    TTM EBTDA / TTM revenue
    ebtda_roa_3y    mean of ebtda_roa over the last 3 years of reports (the track record)
    ebtda_roa_chg   ebtda_roa minus its value a year earlier (improving or not)
    rev_growth      TTM revenue / TTM revenue a year earlier - 1 (the growth signal)
    ebtda_yield     TTM EBTDA / market value            (value: what a dollar of the company
    book_to_market  book equity / market value           buys, the classic cheapness measures)
where market value = shares outstanding from the latest filing's cover page (every share
class summed) x the previous day's REAL closing price (stocks_data.raw_close). Companies
that file 20-F / 40-F reports trade here as depositary receipts whose price covers a
different number of shares than the filing reports, so they get no value measures (a
random rank, like any missing value).
each turned into a percentile rank WITHIN THE STOCK'S SECTOR among the stocks that have it
that day (a bank's EBTDA / assets is not comparable with a software company's), falling
back to the whole universe when fewer than MIN_PEERS sector peers have the value; revenue
growth is ranked across the whole universe instead, since a growth rate means the same in
every sector and "the fastest growers" is the question it answers. Ranks are in (0, 1].

MISSING VALUES ARE RANDOM RANKS, drawn once (MISSING_SEED), not 0 or a flag. EDGAR has
nothing before 2010, so before then EVERY stock lacks fundamentals: a 0 (or a has-data
flag) is a calendar in disguise, and the search used it -- "no fundamentals -> exit" sold
everything through 2006-2009 and scored the 2008 crash as skill. A uniform draw has the
same distribution as a real rank and says nothing, so a rule on fundamentals acts at
random where there are none (before 2010, foreign filers without XBRL) and can only earn
where they exist.

FOREIGN FILERS (the Nasdaq-100 pool holds ASML, Baidu, JD, PDD, AstraZeneca ...): their
20-F / 40-F reports are read under IFRS tags (ifrs-full) as well as US GAAP, in the
company's reporting currency (every measure is a ratio of one company's own numbers, so
the currency cancels). They report once a year, so their values may stay in use until the
next annual report is filed (ANNUAL_STALE_DAYS) instead of STALE_DAYS.

UNIVERSE. Ranks are taken among the stocks in THAT DAY'S point-in-time universe
(stocks_data.py); EDGAR facts are cached per ticker in data/edgar_facts.json and only the
tickers the cache lacks are fetched (--refresh fetches everything again).

Run from the repository root:  python fundamentals.py [--refresh]   (cached to data/fundamentals.npz)
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "edgar_facts.json")
CACHE = os.path.join(ROOT, "data", "fundamentals.npz")
STALE_DAYS = 200
ANNUAL_STALE_DAYS = 420            # companies that file only annual reports (20-F filers)
MIN_PEERS = 5
MISSING_SEED = 2010
# XBRL tags per concept in priority order: companies switch tags over the years (revenue
# moved to RevenueFromContractWithCustomer... under ASC 606 in 2018), so every tag is read
# and a period takes its value from the first tag that reports it
TAGS = {
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
           "DepreciationAndAmortization", "DepreciationAmortizationAndOther", "Depreciation",
           "DepreciationNonproduction"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense"],
    "assets": ["Assets"],
    # pre-tax income rebuilt as net income + income tax where no pre-tax tag reports it
    "profit": ["ProfitLoss", "NetIncomeLoss"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # the LAST resort for D&A, used only for a period with no D&A total at all: some software
    # companies tag only the amortisation of intangibles (Zscaler before 2024) -- a partial
    # D&A, but closer than none, which dropped their EBTDA entirely
    "amort": ["AmortizationOfIntangibleAssets"],
}
# share counts, all read, the latest-ended one used at each date: the cover page, the
# balance sheet, and the diluted weighted average of the period. Companies with several
# share classes (Alphabet, Meta, Mastercard ...) report the cover page per class, which the
# company-facts feed leaves out, and some only the weighted average carries in total
SHARE_TAGS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
              ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]
# no total share count on EDGAR at all in recent years (every figure is per class): Yahoo's
# share-count history (yfinance get_shares_full, dated when Yahoo recorded it), cached to
# data/yahoo_shares.json
YAHOO_SHARES = ("BRK-B", "V")
YAHOO_CACHE = os.path.join(ROOT, "data", "yahoo_shares.json")
IFRS_TAGS = {
    "pretax": ["ProfitLossBeforeTax"],
    "da": ["DepreciationAndAmortisationExpense",
           "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
           "AdjustmentsForDepreciationAndAmortisationExpense",
           "AdjustmentsForDepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss"],
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "assets": ["Assets"],
    "profit": ["ProfitLoss"],
    "tax": ["IncomeTaxExpenseContinuingOperations"],
    "equity": ["EquityAttributableToOwnersOfParent", "Equity"],
}
# companies whose history sits under an earlier registrant: Google Inc. before the 2015
# Alphabet holding company, Exxon Mobil Corp before the ExxonMobil Holdings reorganisation,
# Marvell Technology Group Ltd (Bermuda) before its 2021 re-domicile as Marvell Technology Inc
EXTRA_CIKS = {"GOOGL": [1288776], "XOM": [34088], "MRVL": [1058057]}
RAW_NAMES = ["ebtda_roa", "ebtda_margin", "ebtda_roa_3y", "ebtda_roa_chg", "rev_growth",
             "ebtda_yield", "book_to_market"]
BY_SECTOR = [True, True, True, True, False, True, True]
FUND_NAMES = [f"rank_{n}" for n in RAW_NAMES]


def _get(url):
    import requests
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("set SEC_USER_AGENT to 'Your Name your@email' (the SEC's access rule)")
    for attempt in range(5):
        r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt)            # 429 / 5xx: back off (the SEC allows 10 requests/s)
    r.raise_for_status()


def _currency(taxonomies):
    """The company's reporting currency: the one with the most Assets values. Not "USD
    whenever present": Chinese ADRs add a USD convenience translation of the latest year
    only, at each year's exchange rate, which would make growth partly a currency move."""
    counts = {}
    for tax in taxonomies:
        for u, rows in tax.get("Assets", {}).get("units", {}).items():
            if len(u) == 3 and u.isupper():
                counts[u] = counts.get(u, 0) + len(rows)
    return max(counts, key=counts.get) if counts else "USD"


def fetch(tickers):
    """The TAGS / IFRS_TAGS facts of every ticker in its reporting currency, cached to
    data/edgar_facts.json."""
    cik = {row["ticker"].replace(".", "-"): row["cik_str"]
           for row in _get("https://www.sec.gov/files/company_tickers.json").values()}
    out = {}
    for t in tickers:
        if t not in cik:
            print(f"  {t}: no CIK in the SEC's ticker list, skipped")
            continue
        out[t] = {c: {} for c in TAGS}
        out[t]["_share_tags"] = len(SHARE_TAGS)     # fetched with every share-count tag
        for k in [cik[t]] + EXTRA_CIKS.get(t, []):
            facts = (_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{k:010d}.json") or {}).get("facts", {})
            taxonomies = [(facts.get("us-gaap", {}), TAGS), (facts.get("ifrs-full", {}), IFRS_TAGS)]
            cur = _currency([tx for tx, _ in taxonomies])
            for tx, tagset in taxonomies:
                for c, tags in tagset.items():
                    for tag in tags:
                        if tag in tx and cur in tx[tag]["units"]:
                            out[t].setdefault(c, {}).setdefault(tag, []).extend(tx[tag]["units"][cur])
            for tax, tag in SHARE_TAGS:
                rows = facts.get(tax, {}).get(tag, {}).get("units", {}).get("shares")
                if rows:
                    out[t].setdefault("shares", {}).setdefault(f"{tax}:{tag}", []).extend(rows)
            time.sleep(0.15)
    return out


def _periods(tag_facts):
    """{(start, end): (value, filed)} over a concept's tags: per period the value filed
    FIRST under any tag (what investors could read then), ties to the higher-priority tag.
    Taking the priority tag's value regardless of date would date a period by whenever that
    tag first carried it, sometimes years after another tag reported it."""
    import pandas as pd
    out = {}
    for rows in tag_facts.values():             # dict order = priority order
        first = {}
        for r in rows:
            if r.get("form", "").split("/")[0] not in ("10-K", "10-Q", "10-KT", "20-F", "40-F"):
                continue
            key = (pd.Timestamp(r["start"]) if "start" in r else None, pd.Timestamp(r["end"]))
            filed = pd.Timestamp(r["filed"])
            if key not in first or filed < first[key][1]:
                first[key] = (float(r["val"]), filed)
        for k, v in first.items():
            if k not in out or v[1] < out[k][1]:
                out[k] = v
    return out


def _ttm(periods):
    """[(end, TTM value, available date)] for every duration period that yields a TTM."""
    days = lambda k: (k[1] - k[0]).days
    dur = {k: v for k, v in periods.items() if k[0] is not None}
    annual = {k: v for k, v in dur.items() if 350 <= days(k) <= 380}
    ytd = {k: v for k, v in dur.items() if 80 <= days(k) <= 285}
    out = {}
    for k, (v, f) in annual.items():
        out[k[1]] = (v, f)
    for (s, e), (v, f) in ytd.items():
        if e in out:
            continue
        d = (e - s).days
        prev_fy = [(k, x) for k, x in annual.items() if 0 < (e - k[1]).days < 366]
        prev_ytd = [(k, x) for k, x in ytd.items()
                    if abs((e - k[1]).days - 365) <= 10 and abs(days(k) - d) <= 10]
        if not prev_fy or not prev_ytd:
            continue
        (_, (fy, ffy)) = max(prev_fy, key=lambda kx: kx[0][1])
        (_, (py, fpy)) = prev_ytd[0]
        out[e] = (v + fy - py, max(f, ffy, fpy))
    return sorted((e, v, f) for e, (v, f) in out.items())


def company_series(facts):
    """{end: (ebtda_roa, ebtda_margin, rev_growth, available date)} for one company."""
    per = {c: _periods(facts.get(c, {})) for c in TAGS}
    for k, (v, f) in per["profit"].items():
        if k not in per["pretax"] and k in per["tax"]:
            per["pretax"][k] = (v + per["tax"][k][0], max(f, per["tax"][k][1]))
    pretax = {e: (v, f) for e, v, f in _ttm(per["pretax"])}
    da = {e: (v, f) for e, v, f in _ttm(per["da"])}
    # some companies tag D&A only in the annual report: a quarter without a D&A TTM takes the
    # latest annual one that ended within the previous year (D&A moves slowly)
    annual_da = sorted(e for e in da)
    for e in pretax:
        if e not in da:
            prior = [x for x in annual_da if 0 < (e - x).days < 366]
            if prior:
                da[e] = da[prior[-1]]
    # last resort (TAGS["amort"]): amortisation of intangibles where no D&A total exists
    amort = {e: (v, f) for e, v, f in _ttm(per.get("amort", {}))}
    for e in pretax:
        if e not in da and e in amort:
            da[e] = amort[e]
    rev = {e: (v, f) for e, v, f in _ttm(per["revenue"])}
    assets = {k[1]: v for k, v in per["assets"].items() if k[0] is None}
    rows = {}
    for e, (p, fp) in pretax.items():
        if e not in da or e not in assets or assets[e][0] <= 0:
            continue
        ebtda = p + da[e][0]
        avail = max(fp, da[e][1], assets[e][1])
        margin = growth = np.nan
        if e in rev and rev[e][0] > 0:
            margin, avail = ebtda / rev[e][0], max(avail, rev[e][1])
            ago = [x for x in rev if abs((e - x).days - 365) <= 20 and rev[x][0] > 0]
            if ago:
                growth = rev[e][0] / rev[ago[0]][0] - 1.0
        rows[e] = (ebtda / assets[e][0], margin, growth, avail, ebtda)
    return rows


def raw_panel(panel, facts):
    """(T, N, 6): the five ratio features and the TTM EBTDA level, as known at day t's open."""
    import pandas as pd
    dates = pd.DatetimeIndex(panel.dates)
    T, N = len(dates), len(panel.tickers)
    out = np.full((T, N, 6), np.nan)
    for i, t in enumerate(panel.tickers):
        rows = company_series(facts.get(t, {}))
        if not rows:
            continue
        ends = sorted(rows)
        roa = pd.Series({e: rows[e][0] for e in ends})
        gaps = [(b - a).days for a, b in zip(ends[:-1], ends[1:])]
        annual_only = len(gaps) > 0 and np.median(gaps) > 200
        stale = (ANNUAL_STALE_DAYS if annual_only else STALE_DAYS) + 90
        records = []
        for e in ends:
            hist = roa[(roa.index > e - pd.Timedelta(days=3 * 365 + 20)) & (roa.index <= e)]
            year_ago = roa[(roa.index - (e - pd.Timedelta(days=365))).map(abs) <= pd.Timedelta(days=20)]
            track = hist.mean() if len(hist) >= (3 if annual_only else 8) else np.nan
            records.append((rows[e][3], e, rows[e][0], rows[e][1], track,
                            rows[e][0] - year_ago.iloc[0] if len(year_ago) else np.nan, rows[e][2],
                            rows[e][4]))
        # at each trading day: the latest-ending period whose values were filed BEFORE that day
        records.sort()
        latest_end, vals = None, None
        j = 0
        for ti, d in enumerate(dates):
            while j < len(records) and records[j][0] < d:
                if latest_end is None or records[j][1] >= latest_end:
                    latest_end, vals = records[j][1], records[j][2:]
                j += 1
            if vals is not None and (d - latest_end).days <= stale:
                out[ti, i] = vals
    return out


def _as_of(dates, records, stale_days):
    """(T,) the value of the latest record (available, end, value) available BEFORE each
    day, NaN once its period ended more than stale_days ago."""
    out = np.full(len(dates), np.nan)
    records = sorted(records)
    j, end, val = 0, None, np.nan
    for ti, d in enumerate(dates):
        while j < len(records) and records[j][0] < d:
            if end is None or records[j][1] >= end:
                end, val = records[j][1], records[j][2]
            j += 1
        if end is not None and (d - end).days <= stale_days:
            out[ti] = val
    return out


def yahoo_shares(ticker, refresh=False):
    """[(date, shares)] from Yahoo's share-count history, cached to data/yahoo_shares.json."""
    import pandas as pd
    cache = {}
    if os.path.exists(YAHOO_CACHE):
        with open(YAHOO_CACHE) as fh:
            cache = json.load(fh)
    if ticker not in cache or refresh:
        import yfinance as yf
        s = yf.Ticker(ticker).get_shares_full(start="2005-01-01")
        s = s[~s.index.duplicated(keep="last")].sort_index() if s is not None else []
        cache[ticker] = [[str(pd.Timestamp(k).date()), float(v)] for k, v in s.items()] if len(s) else []
        with open(YAHOO_CACHE + ".tmp", "w") as fh:
            json.dump(cache, fh)
        os.replace(YAHOO_CACHE + ".tmp", YAHOO_CACHE)
    return [(pd.Timestamp(d), v) for d, v in cache[ticker]]


def refresh(tickers):
    """Fetch these tickers' EDGAR facts again and merge them into data/edgar_facts.json."""
    with open(RAW) as fh:
        facts = json.load(fh)
    facts.update(fetch(list(tickers)))
    with open(RAW + ".tmp", "w") as fh:
        json.dump(facts, fh)
    os.replace(RAW + ".tmp", RAW)
    return facts


MAX_SHARES = 5e10     # no US company has had 50 billion shares (Apple peaked near 26 billion)
MIN_SHARES = 1e5      # nor a listed one fewer than 100,000 (a shell before a merger reports 100)


def splits_after(panel):
    """(T, N) the product of every stock split strictly after each day (Nvidia on
    2024-06-07: 10; on 2024-06-10: 1), read off the jumps of the quoted close over the
    panel's adjusted close: a dividend moves that ratio by a few percent, a split by a
    quarter or more."""
    from stocks_data import raw_close
    with np.errstate(invalid="ignore", divide="ignore"):
        q = raw_close(panel) / panel.close
        jump = q[:-1] / q[1:]
    jump = np.where(np.isfinite(jump) & (np.abs(np.log(np.where(jump > 0, jump, 1.0))) > np.log(1.2)), jump, 1.0)
    after = np.ones(q.shape)
    after[:-1] = np.cumprod(jump[::-1], axis=0)[::-1]
    return after


def _drop_scale_errors(recs, days=400, factor=8.0):
    """Share-count records [(key, filed, end, value)], values on one split basis, without the
    ones filed with the wrong scale: some filers tag a count a thousand or a million times
    too big (Qualcomm's cover page of 2011-10, Garmin's of 2016-18, PG&E's balance sheets of
    2015-17). A record more than `factor` away from the median of the company's other records
    ending within `days` of it is dropped."""
    if len(recs) < 3:
        return recs
    import pandas as pd
    ends = pd.DatetimeIndex([r[2] for r in recs]).to_numpy()
    vals = np.array([r[3] for r in recs])
    near = np.abs(ends[:, None] - ends[None, :]) <= np.timedelta64(days, "D")
    np.fill_diagonal(near, False)
    keep = []
    for j, r in enumerate(recs):
        if near[j].sum() >= 2:
            ratio = vals[j] / np.median(vals[near[j]])
            if not 1.0 / factor < ratio < factor:
                continue
        keep.append(r)
    return keep


def shares_and_equity(panel, facts):
    """(T, N) shares outstanding and book equity as known at each open, and (N,) whether
    the company files 10-K / 10-Q (a domestic filer, priced per share here). The share count
    is on the split basis of the PREVIOUS close, the price a market value at the open is
    computed with: a count filed before a split is multiplied by it once that close is past
    the split (a filing restates every period for the splits before it)."""
    import pandas as pd
    dates = pd.DatetimeIndex(panel.dates)
    after = splits_after(panel)
    S = np.full((len(dates), len(panel.tickers)), np.nan)
    Q = np.full_like(S, np.nan)
    domestic = np.zeros(len(panel.tickers), bool)
    for i, t in enumerate(panel.tickers):
        f = facts.get(t, {})
        forms = {r.get("form", "") for rows in f.get("assets", {}).values() for r in rows}
        domestic[i] = any(x.startswith("10-") for x in forms)
        # shares: every tag, one value per filing and period end (a 10-Q repeats a weighted
        # average for the quarter and the year to date); at each date the latest-ended
        # count filed before it, whichever tag carried it
        raw = [((r["accn"], r["end"]), pd.Timestamp(r["filed"]), pd.Timestamp(r["end"]), float(r["val"]))
               for tax, tag in SHARE_TAGS for r in f.get("shares", {}).get(f"{tax}:{tag}", [])
               if "accn" in r and "filed" in r and MIN_SHARES <= float(r["val"]) < MAX_SHARES]
        if t in YAHOO_SHARES:
            raw += [(("yahoo", d), d + pd.Timedelta(days=1), d, v) for d, v in yahoo_shares(t)]
        # every count on today's basis (the splits after the day it was filed), so records from
        # either side of a split compare, and the as-of below carries across one
        day = np.clip(dates.searchsorted(pd.DatetimeIndex([r[1] for r in raw]), side="right") - 1, 0, None)             if raw else np.zeros(0, int)
        raw = [(k, fd, e, v * after[d, i]) for (k, fd, e, v), d in zip(raw, day)]
        recs = {}
        for key, filed, end, val in _drop_scale_errors(raw):
            old = recs.get(key)
            recs[key] = (filed, end, max(val, old[2] if old else 0.0))
        S[:, i] = _as_of(dates, list(recs.values()), 400) / np.r_[after[0, i], after[:-1, i]]
        eq = {k[1]: v for k, v in _periods(f.get("equity", {})).items() if k[0] is None}
        Q[:, i] = _as_of(dates, [(v[1], e, v[0]) for e, v in eq.items()], ANNUAL_STALE_DAYS + 90)
    return S, Q, domestic


def sector_ranks(X, sectors, universe, by_sector=None):
    """Percentile rank (0, 1] within the sector among THAT DAY'S UNIVERSE; the whole
    universe as fallback, and for the columns with by_sector False. Stocks outside the
    universe get no rank and move no one's."""
    import pandas as pd
    T, N, K = X.shape
    sectors = np.asarray(sectors)
    out = np.full_like(X, np.nan)
    for k in range(K):
        col = pd.DataFrame(np.where(universe, X[:, :, k], np.nan))
        overall = col.rank(axis=1, pct=True).to_numpy()
        res = overall.copy()
        for s in (np.unique(sectors) if by_sector is None or by_sector[k] else []):
            m = sectors == s
            sub = col.loc[:, m]
            rk = sub.rank(axis=1, pct=True).to_numpy()
            enough = (sub.notna().sum(axis=1) >= MIN_PEERS).to_numpy()[:, None]
            res[:, m] = np.where(enough, rk, overall[:, m])
        out[:, :, k] = res
    return out


def build(panel=None, refresh=False):
    from stocks_data import load_panel
    p = panel or load_panel()
    facts = {}
    if os.path.exists(RAW) and not refresh:
        with open(RAW) as fh:
            facts = json.load(fh)
    # new tickers, and cached ones with no balance sheet (read before foreign filers were)
    missing = [t for t in p.tickers if t not in facts or "shares" not in facts[t]]
    if missing:                                  # only what the cache lacks (a new universe)
        print(f"fetching EDGAR facts for {len(missing)} tickers", flush=True)
        facts.update(fetch(missing))
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        with open(RAW + ".tmp", "w") as fh:
            json.dump(facts, fh)
        os.replace(RAW + ".tmp", RAW)
    from stocks_data import raw_close
    X6 = raw_panel(p, facts)
    S, Q, domestic = shares_and_equity(p, facts)
    px = raw_close(p)
    prev = np.vstack([np.full((1, px.shape[1]), np.nan), px[:-1]])      # yesterday's close, known at the open
    mcap = np.where(domestic[None, :], S * prev, np.nan)
    mcap = np.where(mcap > 0, mcap, np.nan)
    X = np.concatenate([X6[:, :, :5], (X6[:, :, 5] / mcap)[:, :, None], (Q / mcap)[:, :, None]], axis=2)
    R = sector_ranks(X, p.sectors, p.universe, BY_SECTOR)
    fill = np.random.default_rng(MISSING_SEED).uniform(0.0, 1.0, R.shape)
    Fd = np.where(np.isnan(R), fill, R)                 # missing = uninformative, not a date flag
    np.savez_compressed(CACHE, F=Fd.astype(np.float32), raw=X.astype(np.float32), mcap=mcap.astype(np.float32),
                        names=np.array(FUND_NAMES), raw_names=np.array(RAW_NAMES),
                        tickers=np.array(p.tickers), dates=p.dates.values.astype("datetime64[D]"))
    return Fd, X


def load():
    z = np.load(CACHE, allow_pickle=False)
    return z["F"], list(z["names"])


if __name__ == "__main__":
    import pandas as pd
    from stocks_data import load_panel
    p = load_panel()
    Fd, X = build(p, refresh="--refresh" in sys.argv)
    dates = pd.DatetimeIndex(p.dates)
    has = ~np.isnan(X[:, :, 0]) & p.universe
    cover = pd.Series(has.sum(1) / np.maximum(p.universe.sum(1), 1), index=dates)
    print("share of the day's universe with ebtda_roa, by year:")
    print(cover.groupby(dates.year).mean().round(2).to_string())
    missing = [t for i, t in enumerate(p.tickers) if np.isnan(X[:, i, 0]).all()]
    print("never covered:", missing)
