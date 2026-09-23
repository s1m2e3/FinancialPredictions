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
}
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
}
# companies whose history sits under an earlier registrant: Google Inc. before the 2015
# Alphabet holding company, Exxon Mobil Corp before the ExxonMobil Holdings reorganisation
EXTRA_CIKS = {"GOOGL": [1288776], "XOM": [34088]}
RAW_NAMES = ["ebtda_roa", "ebtda_margin", "ebtda_roa_3y", "ebtda_roa_chg", "rev_growth"]
BY_SECTOR = [True, True, True, True, False]
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
        for k in [cik[t]] + EXTRA_CIKS.get(t, []):
            facts = (_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{k:010d}.json") or {}).get("facts", {})
            taxonomies = [(facts.get("us-gaap", {}), TAGS), (facts.get("ifrs-full", {}), IFRS_TAGS)]
            cur = _currency([tx for tx, _ in taxonomies])
            for tx, tagset in taxonomies:
                for c, tags in tagset.items():
                    for tag in tags:
                        if tag in tx and cur in tx[tag]["units"]:
                            out[t][c].setdefault(tag, []).extend(tx[tag]["units"][cur])
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
        rows[e] = (ebtda / assets[e][0], margin, growth, avail)
    return rows


def raw_panel(panel, facts):
    """(T, N, 5) raw features on the trading calendar, each as known at day t's open."""
    import pandas as pd
    dates = pd.DatetimeIndex(panel.dates)
    T, N = len(dates), len(panel.tickers)
    out = np.full((T, N, len(RAW_NAMES)), np.nan)
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
                            rows[e][0] - year_ago.iloc[0] if len(year_ago) else np.nan, rows[e][2]))
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
    missing = [t for t in p.tickers if t not in facts or not facts[t].get("assets")]
    if missing:                                  # only what the cache lacks (a new universe)
        print(f"fetching EDGAR facts for {len(missing)} tickers", flush=True)
        facts.update(fetch(missing))
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        with open(RAW + ".tmp", "w") as fh:
            json.dump(facts, fh)
        os.replace(RAW + ".tmp", RAW)
    X = raw_panel(p, facts)
    R = sector_ranks(X, p.sectors, p.universe, BY_SECTOR)
    fill = np.random.default_rng(MISSING_SEED).uniform(0.0, 1.0, R.shape)
    Fd = np.where(np.isnan(R), fill, R)                 # missing = uninformative, not a date flag
    np.savez_compressed(CACHE, F=Fd.astype(np.float32), raw=X.astype(np.float32),
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
