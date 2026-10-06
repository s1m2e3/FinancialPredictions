"""Earnings surprises from EDGAR, as known at the OPEN of each trading day.

POST-EARNINGS-ANNOUNCEMENT DRIFT. Stocks whose earnings beat what their own history led one
to expect keep drifting up for weeks after the news, and those that miss keep drifting down
(Ball and Brown 1968, Bernard and Thomas 1989; among the most replicated effects in finance).
The surprise is measured the standard way, against a SEASONAL RANDOM WALK:

    SUE_q = (X_q - X_{q-4}) / sd(X_k - X_{k-4} over the 8 quarters before q)

X_q is quarterly net income (ProfitLoss / NetIncomeLoss, fundamentals.py's "profit"), or
revenue for the revenue surprise. Quarters come straight from 10-Q three-month values; a
fiscal fourth quarter is the annual report minus the nine-month year-to-date value, and a
missing second or third quarter is a year-to-date difference. At least 4 prior changes are
required for the scale.

POINT IN TIME. A quarter's surprise is dated by its EARNINGS RELEASE: the company's first 8-K
with item 2.02 ("Results of Operations and Financial Condition") filed after the quarter ended
and no later than its 10-Q (EDGAR's submissions data, cached in data/edgar_8k.json); without
one, by the 10-Q itself. It enters the panel on the first trading day AFTER that date and is
kept for 100 calendar days (roughly until the next quarter's). The figures are the 10-Q's,
which are the ones the release announced -- the report-date convention of the academic
literature (Compustat's rdq) -- so the only thing taken from the later filing is the number
the market already had. The 10-Q usually lands 2-5 weeks after the release, when most of the
drift is gone; the 10-Q-dated versions are kept (`_10q`) to show the difference.

FEATURES (data/earnings.npz), all among the stocks in that day's universe:
    rank_sue            percentile rank of the earnings surprise, release-dated
    rank_rev_sue        percentile rank of the revenue surprise, release-dated
    sue_age             trading days since the release / 63 (the drift is strongest early)
    rank_sue_10q, rank_rev_sue_10q    the same, dated by the 10-Q

    python earnings.py      builds data/earnings.npz from data/edgar_facts.json (fundamentals.py)
"""
import json
import os

import numpy as np

import fundamentals as fu

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "earnings.npz")
RELEASES = os.path.join(ROOT, "data", "edgar_8k.json")
NAMES = ["rank_sue", "rank_rev_sue", "sue_age", "rank_sue_10q", "rank_rev_sue_10q"]
KEEP_DAYS = 100


def fetch_releases(tickers):
    """{ticker: sorted filing dates of its 8-Ks with item 2.02}, from EDGAR's submissions data
    (the recent filings and every older page), cached to data/edgar_8k.json; only tickers the
    cache lacks are fetched."""
    import time
    have = {}
    if os.path.exists(RELEASES):
        with open(RELEASES) as fh:
            have = json.load(fh)
    missing = [t for t in tickers if t not in have]
    if not missing:
        return have
    cik = {r["ticker"].replace(".", "-"): r["cik_str"]
           for r in fu._get("https://www.sec.gov/files/company_tickers.json").values()}
    for n, t in enumerate(missing):
        dates = set()
        for k in ([cik[t]] if t in cik else []) + fu.EXTRA_CIKS.get(t, []):
            sub = fu._get(f"https://data.sec.gov/submissions/CIK{k:010d}.json") or {}
            pages = [sub.get("filings", {}).get("recent", {})]
            for extra in sub.get("filings", {}).get("files", []):
                pages.append(fu._get(f"https://data.sec.gov/submissions/{extra['name']}") or {})
                time.sleep(0.12)
            for pg in pages:
                for form, day, items in zip(pg.get("form", []), pg.get("filingDate", []), pg.get("items", [])):
                    if form in ("8-K", "8-K/A") and "2.02" in str(items).split(","):
                        dates.add(day)
            time.sleep(0.12)
        have[t] = sorted(dates)
        if (n + 1) % 25 == 0 or n + 1 == len(missing):
            print(f"  earnings releases: {n + 1}/{len(missing)} companies", flush=True)
            with open(RELEASES + ".tmp", "w") as fh:
                json.dump(have, fh)
            os.replace(RELEASES + ".tmp", RELEASES)
    return have


def release_dater(release_dates):
    """A function (quarter end, 10-Q filed) -> the date the quarter's figures became public:
    the first 8-K 2.02 after the quarter end and no later than the 10-Q, else the 10-Q."""
    import pandas as pd
    rel = pd.DatetimeIndex(sorted(pd.to_datetime(release_dates))) if release_dates else pd.DatetimeIndex([])

    def date(end, filed):
        i = rel.searchsorted(end, side="right")
        if i < len(rel) and rel[i] <= filed:
            return rel[i]
        return filed
    return date


def quarters(periods):
    """{quarter end: (value, filed)} of discrete fiscal quarters from a concept's periods."""
    days = lambda k: (k[1] - k[0]).days
    dur = {k: v for k, v in periods.items() if k[0] is not None}
    q = {k[1]: v for k, v in dur.items() if 80 <= days(k) <= 100}
    ytd = {k: v for k, v in dur.items() if 170 <= days(k) <= 290}
    fy = {k: v for k, v in dur.items() if 350 <= days(k) <= 380}
    near = lambda a, b, tol=6: abs((a - b).days) <= tol
    # fiscal Q4 = annual - nine months with the same start
    for (s, e), (v, f) in fy.items():
        if any(near(e, x) for x in q):
            continue
        nine = [(k, x) for k, x in ytd.items() if near(k[0], s) and 260 <= days(k) <= 290]
        if nine:
            (k9, (v9, f9)) = nine[0]
            q[e] = (v - v9, max(f, f9))
    # Q2 = six months - Q1, Q3 = nine months - six months (same start), where not reported
    for (s, e), (v, f) in ytd.items():
        if any(near(e, x) for x in q):
            continue
        prior = [(k, x) for k, x in ytd.items() if near(k[0], s) and 80 <= (e - k[1]).days <= 100]
        first = [(k, x) for k, x in q.items() if 80 <= (k - s).days <= 100]      # Q1 ending ~3 months after s
        if prior:
            (_, (vp, fp)) = prior[0]
            q[e] = (v - vp, max(f, fp))
        elif 170 <= days((s, e)) <= 190 and first:
            (_, (vp, fp)) = first[0]
            q[e] = (v - vp, max(f, fp))
    return dict(sorted(q.items()))


def surprises(periods, dater=None):
    """[(available date, SUE)] of every quarter with a year-ago value and 4+ prior changes;
    `dater(end, filed)` moves each quarter's date to its earnings release (release_dater)."""
    q = quarters(periods)
    if dater is not None:
        q = {e: (v, dater(e, f)) for e, (v, f) in q.items()}
    ends = sorted(q)
    change, out = {}, []
    for e in ends:
        ago = [x for x in ends if abs((e - x).days - 365) <= 20]
        if ago:
            change[e] = (q[e][0] - q[ago[0]][0], max(q[e][1], q[ago[0]][1]))
    ch_ends = sorted(change)
    for i, e in enumerate(ch_ends):
        past = [change[x][0] for x in ch_ends[max(0, i - 8):i]]
        if len(past) < 4:
            continue
        sd = float(np.std(past, ddof=1))
        if sd <= 0:
            continue
        out.append((change[e][1], float(np.clip(change[e][0] / sd, -10, 10))))
    return out


def panel_series(dates, records):
    """(T,) value of the latest record known before each day's open (NaN after KEEP_DAYS),
    and (T,) trading days since it became known."""
    import pandas as pd
    val, age = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
    if not records:
        return val, age
    records = sorted(records)
    avail = pd.DatetimeIndex([r[0] for r in records])
    idx = avail.searchsorted(dates, side="left") - 1              # latest filed strictly before the day
    for t in np.where(idx >= 0)[0]:
        f, v = records[idx[t]]
        if (dates[t] - f).days <= KEEP_DAYS:
            val[t] = v
            age[t] = np.searchsorted(dates, f, side="right")
    known = np.isfinite(age)
    age[known] = np.arange(len(dates))[known] - age[known]
    return val, age


def build(panel=None):
    import pandas as pd
    from stocks_data import load_panel
    p = panel or load_panel()
    with open(fu.RAW) as fh:
        facts = json.load(fh)
    dates = pd.DatetimeIndex(p.dates)
    T, N = len(dates), len(p.tickers)
    releases = fetch_releases(p.tickers)
    blank = lambda: np.full((T, N), np.nan)
    sue, rsue, age, sue_q, rsue_q = blank(), blank(), blank(), blank(), blank()
    n_quarters, lead = 0, []
    for i, t in enumerate(p.tickers):
        f = facts.get(t)
        if not f:
            continue
        dater = release_dater(releases.get(t, []))
        pe, pr = fu._periods(f.get("profit", {})), fu._periods(f.get("revenue", {}))
        s_e, s_r = surprises(pe, dater), surprises(pr, dater)
        n_quarters += len(s_e)
        sue[:, i], age[:, i] = panel_series(dates, s_e)
        rsue[:, i], _ = panel_series(dates, s_r)
        sue_q[:, i], _ = panel_series(dates, surprises(pe))
        rsue_q[:, i], _ = panel_series(dates, surprises(pr))
        lead += [(fq - dater(e, fq)).days for e, (_, fq) in quarters(pe).items()]
    U = p.universe
    rank = lambda x: pd.DataFrame(np.where(U, x, np.nan)).rank(axis=1, pct=True).to_numpy()
    F = np.stack([rank(sue), rank(rsue), np.where(np.isfinite(sue), age / 63.0, np.nan),
                  rank(sue_q), rank(rsue_q)], axis=2)
    np.savez_compressed(CACHE, F=F.astype(np.float32), names=np.array(NAMES), raw_sue=sue.astype(np.float32),
                        tickers=np.array(p.tickers), dates=p.dates.values.astype("datetime64[D]"))
    lead = np.array(lead)
    print(f"earnings releases found for {sum(bool(releases.get(t)) for t in p.tickers)} of {N} stocks; the release "
          f"came before the 10-Q by a median {np.median(lead):.0f} days (mean {lead.mean():.0f}; "
          f"{np.mean(lead == 0):.0%} of quarters dated by the 10-Q itself)")
    cover = pd.Series((np.isfinite(sue) & U).sum(1) / np.maximum(U.sum(1), 1), index=dates)
    print(f"earnings surprises for {N} stocks: {n_quarters} quarters; share of the day's universe with a "
          f"surprise known in the last {KEEP_DAYS} days, by year:")
    print(cover.groupby(dates.year).mean().round(2).to_string())
    return F


def load():
    z = np.load(CACHE, allow_pickle=False)
    return z["F"], list(z["names"])


if __name__ == "__main__":
    build()
