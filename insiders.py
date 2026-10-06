"""Insider buying from the SEC's Form 3/4/5 data sets, as known at the OPEN of each day.

SOURCE. The SEC's "Insider Transactions Data Sets" (www.sec.gov/dera/data/form-345), one
zip per quarter since 2006: every Form 4 filed, with its filing date, the reporting
owners and their relationship to the company, and every non-derivative transaction. Set
SEC_USER_AGENT as for fundamentals.py; the zips are cached in data/insider/.

THE SIGNAL. Open-market PURCHASES (transaction code P) by officers and directors. Sales
are mostly compensation and diversification and say little; an insider spending their
own cash on the stock is one of the better-documented signals (Lakonishok and Lee 2001,
Cohen, Malloy and Pomorski 2012). Per stock and day:
    insider_buyers_90d   distinct officers / directors whose Form 4 reporting an
                         open-market purchase was FILED in the last 90 calendar days
                         (filed before that day's open), log(1 + count)
    insider_buy_90d      1 if there is at least one such buyer
Companies are matched by the issuer's CIK, which survives ticker changes.

Run from the repository root:  python insiders.py   (cached to data/insiders.npz)
"""
import io
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIPS = os.path.join(ROOT, "data", "insider")
CACHE = os.path.join(ROOT, "data", "insiders.npz")
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
WINDOW_DAYS = 90
INSIDER_NAMES = ["insider_buyers_90d", "insider_buy_90d"]


def quarters(first=2005, last=None):
    import pandas as pd
    last = last or pd.Timestamp.today().year
    return [f"{y}q{q}" for y in range(first, last + 1) for q in range(1, 5)]


def download(qs=None):
    """Every quarter's zip not yet in data/insider/; a quarter the SEC has not published
    yet (404) is skipped."""
    import requests
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("set SEC_USER_AGENT to 'Your Name your@email' (the SEC's access rule)")
    os.makedirs(ZIPS, exist_ok=True)
    for q in qs or quarters():
        path = os.path.join(ZIPS, f"{q}_form345.zip")
        if os.path.exists(path):
            continue
        for attempt in range(5):
            r = requests.get(URL.format(q=q), headers={"User-Agent": ua}, timeout=120)
            if r.status_code in (200, 404):
                break
            time.sleep(2 ** attempt)
        if r.status_code == 404:
            print(f"  {q}: not published", flush=True)
            continue
        r.raise_for_status()
        with open(path + ".tmp", "wb") as fh:
            fh.write(r.content)
        os.replace(path + ".tmp", path)
        print(f"  {q}: {len(r.content) / 1e6:.1f} MB", flush=True)
        time.sleep(0.3)


def purchases():
    """DataFrame (cik, owner, filed) of open-market purchases by officers and directors."""
    import pandas as pd
    rows = []
    for name in sorted(os.listdir(ZIPS)):
        if not name.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(ZIPS, name)) as z:
            rd = lambda t, cols: pd.read_csv(io.BytesIO(z.read(t)), sep="\t", usecols=cols, dtype=str,
                                             on_bad_lines="skip", low_memory=False)
            sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "DOCUMENT_TYPE"])
            own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
            tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_CODE"])
        buys = tr[tr["TRANS_CODE"] == "P"][["ACCESSION_NUMBER"]].drop_duplicates()
        rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
        own = own[rel.str.contains("Officer|Director", case=False, regex=True)]
        df = (buys.merge(sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])], on="ACCESSION_NUMBER")
              .merge(own, on="ACCESSION_NUMBER"))
        rows.append(df[["ISSUERCIK", "RPTOWNERCIK", "FILING_DATE"]])
    out = pd.concat(rows, ignore_index=True)
    out.columns = ["cik", "owner", "filed"]
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce")
    out["filed"] = pd.to_datetime(out["filed"], format="mixed", dayfirst=False, errors="coerce")
    return out.dropna().drop_duplicates()


def build(panel=None):
    """(T, N, 2) insider features on the panel's calendar, saved to data/insiders.npz."""
    import pandas as pd
    from fundamentals import EXTRA_CIKS, _get
    from stocks_data import load_panel
    p = panel or load_panel()
    cik_of = {r["ticker"].replace(".", "-"): int(r["cik_str"])
              for r in _get("https://www.sec.gov/files/company_tickers.json").values()}
    buys = purchases()
    dates = pd.DatetimeIndex(p.dates)
    X = np.zeros((len(dates), len(p.tickers), len(INSIDER_NAMES)))
    for i, t in enumerate(p.tickers):
        ciks = [cik_of[t]] + EXTRA_CIKS.get(t, []) if t in cik_of else []
        b = buys[buys["cik"].isin(ciks)]
        if b.empty:
            continue
        # a filing counts from the first day AFTER it was filed (known at that open) for
        # WINDOW_DAYS calendar days; an owner counts once however often they bought
        count = np.zeros(len(dates))
        for _, g in b.groupby("owner"):
            active = np.zeros(len(dates), bool)
            lo = dates.searchsorted(g["filed"].values, side="right")
            hi = dates.searchsorted((g["filed"] + pd.Timedelta(days=WINDOW_DAYS)).values, side="right")
            for a, z in zip(lo, hi):
                active[a:z] = True
            count += active
        X[:, i, 0] = np.log1p(count)
        X[:, i, 1] = count > 0
    # past the last published quarter the data sets are silent, which would read as "no
    # insider bought": the features hold their last known values there instead
    last = dates.searchsorted(buys["filed"].max(), side="right")
    X[last:] = X[last - 1]
    np.savez_compressed(CACHE, F=X.astype(np.float32), names=np.array(INSIDER_NAMES),
                        tickers=np.array(p.tickers), dates=p.dates.values.astype("datetime64[D]"))
    return X


if __name__ == "__main__":
    download()
    if "--download-only" not in sys.argv:
        X = build()
        print("stock-days with an insider buyer in the last 90 days:", round(float(X[:, :, 1].mean()), 4))
