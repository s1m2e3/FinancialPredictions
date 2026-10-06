"""Survivorship-free prices from CRSP (through WRDS) for the point-in-time universe.

WHY. yfinance drops a ticker once it is delisted, so most S&P 500 members that went
bankrupt or were acquired have no prices: the tradable set silently loses, AHEAD of time,
many of the companies that were about to fail (stocks_data.py measured the free edge this
leaves at +3.0 / +1.4 / +0.7 %/yr on 2006-2019 / 2020-2021 / 2022-2026). CRSP keeps every
US common stock, delisted or not, with its delisting return.

WHAT THIS BUILDS (data/crsp_raw.pkl), everything keyed by CRSP's PERMNO, the permanent
security id, so a reused ticker can never mix two companies:
    membership   S&P 500 spells from CRSP's own constituent table: every member has prices
                 by construction, no ticker matching
    prices       daily open / close / volume, adjusted to a TOTAL-RETURN basis like the
                 yfinance prices (dividends reinvested): the close is a cumulative product
                 of CRSP's daily returns, the open scaled by that day's open / close
    delisting    the day after a stock's last trading day carries its CRSP delisting return
                 (a bankruptcy is about -100%), so a tree holding it takes the loss; after
                 that the stock has no price and the simulator sells it
    names        the ticker each permno traded under over time, and the Nasdaq-100 snapshots
                 (stocks_data.ndx_snapshots) mapped to permnos by the ticker AS OF each
                 snapshot's date, for the outside pool
    labels       each permno's column label: its latest ticker, with ".<permno>" appended
                 when two permnos share one, so fundamentals and insiders (matched by ticker
                 on EDGAR) keep working for companies that still file; delisted companies
                 have no EDGAR match and their fundamentals are missing

ACCESS. A WRDS account (the University of Arizona library provides them) and the `wrds`
package (pip install wrds). You log in yourself when it asks -- the password goes to WRDS
and to your own ~/.pgpass if you choose to save it, never into this code.

TABLES. CRSP's current format (CIZ, the `_v2` / `stk*` tables) is used; the legacy format
(crsp.dsf, dsedelist, dsp500list, stocknames) stopped with December 2024 data, which would
leave 2025-2026 without delisted stocks. The column names below follow WRDS's CIZ
documentation; `--check` verifies every one of them exists BEFORE anything is downloaded,
and stops with the table's actual columns if WRDS names them differently.

    python crsp_data.py --check      connect, verify the tables, pull Lehman Brothers' 2008
    python crsp_data.py --download   pull everything (writes data/crsp_raw.pkl)

stocks_data.load_panel uses it once PRICE_SOURCE is set to "crsp" there.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "crsp_raw.pkl")
START, END = "2004-01-01", "2026-09-19"

# CIZ tables and the columns this module reads from each
TABLES = {
    "prices": ("crsp", "dsf_v2", ["permno", "dlycaldt", "dlyprc", "dlyopen", "dlyvol", "dlyret"]),
    "delist": ("crsp", "stkdelists", ["permno", "delistingdt", "delret"]),
    "sp500": ("crsp", "dsp500list_v2", ["permno", "mbrstartdt", "mbrenddt"]),
    "names": ("crsp", "stksecurityinfohist", ["permno", "secinfostartdt", "secinfoenddt", "ticker",
                                              "issuernm", "sharetype", "securitytype"]),
}


def connect():
    import wrds
    user = os.environ.get("WRDS_USERNAME")
    print("connecting to WRDS" + (f" as {user}" if user else "") + " (it will ask for your password)", flush=True)
    return wrds.Connection(wrds_username=user) if user else wrds.Connection()


def verify(db):
    """Stop with the real column list if any table or column is not what this module reads."""
    ok = True
    for key, (lib, table, cols) in TABLES.items():
        try:
            have = set(db.describe_table(lib, table)["name"].str.lower())
        except Exception as exc:
            print(f"  {lib}.{table}: NOT AVAILABLE ({type(exc).__name__}: {exc})")
            ok = False
            continue
        missing = [c for c in cols if c not in have]
        print(f"  {lib}.{table}: " + ("ok" if not missing else f"MISSING {missing}; it has {sorted(have)}"))
        ok &= not missing
    if not ok:
        raise SystemExit("the CRSP tables differ from what crsp_data.py expects: send these lines to fix the names")


def _q(db, sql):
    return db.raw_sql(sql, date_cols=None)


def sp500_spells(db):
    import pandas as pd
    lib, t, _ = TABLES["sp500"]
    m = _q(db, f"select permno, mbrstartdt, mbrenddt from {lib}.{t} where mbrenddt >= '{START}'")
    m["mbrstartdt"], m["mbrenddt"] = pd.to_datetime(m["mbrstartdt"]), pd.to_datetime(m["mbrenddt"])
    return m.astype({"permno": int})


def names(db, permnos=None):
    import pandas as pd
    lib, t, _ = TABLES["names"]
    where = f"where secinfoenddt >= '{START}'" + (f" and permno in ({','.join(map(str, permnos))})" if permnos else "")
    n = _q(db, f"select permno, secinfostartdt, secinfoenddt, ticker, issuernm, sharetype, securitytype "
               f"from {lib}.{t} {where}")
    n["secinfostartdt"], n["secinfoenddt"] = pd.to_datetime(n["secinfostartdt"]), pd.to_datetime(n["secinfoenddt"])
    return n.astype({"permno": int})


def prices(db, permnos, chunk=150):
    """Daily rows for these permnos, in chunks (a few hundred thousand rows each)."""
    import pandas as pd
    lib, t, _ = TABLES["prices"]
    out = []
    permnos = sorted(set(int(p) for p in permnos))
    for i in range(0, len(permnos), chunk):
        part = permnos[i:i + chunk]
        out.append(_q(db, f"select permno, dlycaldt, dlyprc, dlyopen, dlyvol, dlyret from {lib}.{t} "
                          f"where permno in ({','.join(map(str, part))}) "
                          f"and dlycaldt between '{START}' and '{END}'"))
        print(f"  prices: {min(i + chunk, len(permnos))}/{len(permnos)} securities", flush=True)
    p = pd.concat(out, ignore_index=True)
    p["dlycaldt"] = pd.to_datetime(p["dlycaldt"])
    return p.astype({"permno": int})


def delistings(db, permnos):
    import pandas as pd
    lib, t, _ = TABLES["delist"]
    d = _q(db, f"select permno, delistingdt, delret from {lib}.{t} "
               f"where permno in ({','.join(map(str, sorted(set(map(int, permnos)))))})")
    d["delistingdt"] = pd.to_datetime(d["delistingdt"])
    return d.astype({"permno": int})


def adjusted(p, delist):
    """{field: DataFrame(date x permno)} on a total-return basis, with each delisting return
    applied on the day after the last trade."""
    import pandas as pd
    p = p.sort_values(["permno", "dlycaldt"]).copy()
    p["dlyprc"] = p["dlyprc"].abs()                 # CRSP marks a bid/ask midpoint with a minus sign
    p["dlyopen"] = p["dlyopen"].abs()
    p["dlyret"] = pd.to_numeric(p["dlyret"], errors="coerce")
    rows = []
    for permno, g in p.groupby("permno"):
        g = g.dropna(subset=["dlyprc"])
        if g.empty:
            continue
        r = g["dlyret"].fillna(0.0).to_numpy()
        tr = g["dlyprc"].iloc[0] * np.cumprod(1.0 + np.r_[0.0, r[1:]])      # total-return close
        opn = tr * (g["dlyopen"] / g["dlyprc"]).to_numpy()                     # same-day ratio
        df = pd.DataFrame({"Close": tr, "Open": np.where(np.isfinite(opn) & (opn > 0), opn, np.nan),
                           "Volume": g["dlyvol"].to_numpy()}, index=g["dlycaldt"].to_numpy())
        dl = delist[delist["permno"] == permno]
        if len(dl) and pd.notna(dl["delret"].iloc[0]):
            last = df.index[-1]
            end = last + pd.tseries.offsets.BDay(1)
            v = df["Close"].iloc[-1] * (1.0 + float(dl["delret"].iloc[0]))
            df.loc[end] = [v, v, 0.0]                  # the delisting value, then no more prices
        df["permno"] = permno
        rows.append(df)
    long = pd.concat(rows).rename_axis("date").reset_index()
    return {f: long.pivot_table(index="date", columns="permno", values=f, aggfunc="last") for f in ("Open", "Close", "Volume")}


def labels(nm):
    """{permno: column label}: its latest ticker, ".<permno>" added where tickers collide."""
    last = nm.sort_values("secinfoenddt").groupby("permno")["ticker"].last().fillna("").str.replace(".", "-", regex=False)
    dup = last[last.duplicated(keep=False)].index
    return {p: (f"{t}.{p}" if p in dup or not t else t) for p, t in last.items()}


def ndx_permnos(nm):
    """The Nasdaq-100 snapshots (stocks_data) with each ticker mapped to the permno that
    traded under it ON THE SNAPSHOT'S DATE."""
    from stocks_data import ndx_snapshots
    s = ndx_snapshots()
    rows = []
    for r in s.itertuples():
        hit = nm[(nm["ticker"] == str(r.ticker).replace("-", ".")) | (nm["ticker"] == str(r.ticker))]
        hit = hit[(hit["secinfostartdt"] <= r.date) & (hit["secinfoenddt"] >= r.date)]
        if len(hit):
            rows.append((r.date, int(hit["permno"].iloc[0])))
    import pandas as pd
    return pd.DataFrame(rows, columns=["date", "permno"])


def download():
    import pandas as pd
    db = connect()
    verify(db)
    sp = sp500_spells(db)
    print(f"S&P 500 spells since {START}: {len(sp)}, securities: {sp.permno.nunique()}", flush=True)
    nm = names(db)
    ndx = ndx_permnos(nm)
    print(f"Nasdaq-100 snapshot rows mapped to permnos: {len(ndx)} ({ndx.permno.nunique()} securities)", flush=True)
    universe = sorted(set(sp.permno) | set(ndx.permno))
    p = prices(db, universe)
    d = delistings(db, universe)
    fields = adjusted(p, d)
    out = dict(fields=fields, sp500=sp, ndx=ndx, names=nm[nm.permno.isin(universe)], labels=labels(nm[nm.permno.isin(universe)]),
               delist=d, start=START, end=END)
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    pd.to_pickle(out, RAW + ".tmp")
    os.replace(RAW + ".tmp", RAW)
    print(f"wrote {RAW}: {len(universe)} securities, {fields['Close'].shape[0]} days, "
          f"{len(d)} delisting records", flush=True)


def check():
    """Verify the tables and pull one known case: Lehman Brothers through its 2008 failure."""
    db = connect()
    verify(db)
    nm = names(db)
    leh = nm[nm["ticker"] == "LEH"]
    print("LEH permnos:", sorted(leh.permno.unique()), leh[["issuernm", "secinfostartdt", "secinfoenddt"]].tail(3).to_string())
    if leh.empty:
        raise SystemExit("no LEH in the names table: send this output")
    pn = int(leh.permno.iloc[-1])
    p = prices(db, [pn])
    d = delistings(db, [pn])
    f = adjusted(p[p.dlycaldt >= "2008-06-01"], d)
    c = f["Close"][pn].dropna()
    print(f"Lehman (permno {pn}): {len(c)} days from {c.index[0].date()} to {c.index[-1].date()}, "
          f"close {c.iloc[0]:.2f} -> {c.iloc[-1]:.4f}; delisting records:\n{d.to_string()}")
    sp = sp500_spells(db)
    print("in CRSP's S&P 500 list:", sp[sp.permno == pn].to_string(index=False))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--download" in sys.argv:
        download()
    else:
        raise SystemExit(__doc__)
