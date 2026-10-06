"""The GROWTH POOL's candidates: every US company whose publicly held stock was worth at least
FLOOR ($2B) at some mid-year since 2010 -- including those that were later bought out or
failed -- and where each one's prices can come from.

WHY. To reward the model for finding growers early (Palantir at its IPO, Tesla in 2013), the
pool must also hold the growers that failed (Nikola, a $30B company in 2020), or a rule that
buys young fast-growing stocks is paid for every winner and never charged for a loser.
yfinance has almost no delisted stocks, so the delisted names come from Tiingo (tiingo_prices.py).

SOURCES, all free:
    EDGAR frames      dei:EntityPublicFloat of EVERY filer, one request per year (CYyyyyQ2I);
                      a scale typo (a float reported in thousands as dollars) is dropped when a
                      value is 50x the company's own median of the other years
    EDGAR tickers     company_tickers.json: the companies with a ticker today (yfinance prices)
    Alpha Vantage     data/listings_{active,delisted}.csv (listings.py): names -> old tickers
    Tiingo            data/tiingo_supported_tickers.csv: each ticker's first and last price day

A company gone from EDGAR's ticker list is matched to its old ticker by normalized name
(Alpha Vantage); names that do not match are reported, not guessed.

A FAIR SAMPLE for quick tests (--sample P): the "new guys" -- companies that first reached
FLOOR in 2013 or later -- drawn at the SAME rate P whether they are still listed or gone,
with a fixed seed and no hand-picking, so the sample is as fair as the whole list, only
smaller. Gone companies without a Tiingo history are drawn too and reported: they are the
part of the sample that cannot be priced.

    python growth_pool.py                 writes data/growth_candidates.csv and prints the coverage
    python growth_pool.py --sample 0.125  also writes data/growth_sample.csv
    python growth_pool.py --sample 0.143 --name 2 --seed 2014
                                          a SECOND sample (data/growth_sample2.csv) of the new guys
                                          no earlier sample holds: a test on different companies
"""
import json
import os
import re
import time
import unicodedata

import pandas as pd

import fundamentals as fu

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
FRAMES = os.path.join(DATA, "edgar_float_frames.json")
OUT = os.path.join(DATA, "growth_candidates.csv")
FLOOR = 2e9
YEARS = range(2010, 2026)
US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSE ARCA", "NYSE MKT", "BATS"}
_SUFFIX = re.compile(r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LLC|LP|HOLDINGS?|GROUP|"
                     r"THE|NV|SA|AG|SE|NEW|CLASS [A-Z]|COMMON STOCK|ORDINARY SHARES?|ADR|ADS|SPONSORED)\b")


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().upper().replace("&", " AND ")
    s = re.sub(r"/[A-Z]{2,3}/?", " ", s)                     # EDGAR's state tags: AETNA INC /PA/
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", _SUFFIX.sub(" ", s)).strip()


def frames():
    """{year: DataFrame(cik, entityName, val)} of every filer's public float, cached. A company
    measures it at the end of ITS second fiscal quarter -- June 30 for a December year, Dec 31
    for a June year (Peloton), Sept 30 for a March year (Xilinx) -- so all four calendar
    quarters' frames are read and each company keeps its value of the year."""
    raw = {}
    if os.path.exists(FRAMES):
        with open(FRAMES) as fh:
            raw = json.load(fh)
    todo = [f"{y}Q{q}" for y in YEARS for q in (1, 2, 3, 4) if f"{y}Q{q}" not in raw]
    for k in todo:
        fr = fu._get(f"https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/CY{k}I.json")
        raw[k] = (fr or {}).get("data", [])
        time.sleep(0.15)
    if todo:
        with open(FRAMES, "w") as fh:
            json.dump(raw, fh)
    out = {}
    for k, d in raw.items():
        if d and "Q" in k:
            out.setdefault(int(k[:4]), []).append(pd.DataFrame(d)[["cik", "entityName", "val"]])
    return {y: pd.concat(v).sort_values("val").drop_duplicates("cik", keep="last") for y, v in out.items()}


def floats():
    """(cik x year) public float, scale typos removed."""
    long = pd.concat([f.assign(year=y) for y, f in frames().items()])
    wide = long.pivot_table(index="cik", columns="year", values="val", aggfunc="max")
    wide = wide.where(wide <= 4e12)
    med = wide.median(axis=1)
    return wide.where(wide.le(50 * med, axis=0) | wide.count(axis=1).lt(3).to_frame().values), long


def main():
    fl, long = floats()
    names = long.sort_values("year").groupby("cik")["entityName"].last()
    big = fl[(fl >= FLOOR).any(axis=1)]
    now = {int(r["cik_str"]): r["ticker"] for r in fu._get("https://www.sec.gov/files/company_tickers.json").values()}
    av = pd.concat([pd.read_csv(os.path.join(DATA, f"listings_{s}.csv"), keep_default_na=False, na_values=["", "null"])
                    for s in ("active", "delisted")])
    av = av[(av["assetType"] == "Stock") & ~av["symbol"].str.contains(r"[-.^]", regex=True)
            & ~av["name"].str.contains(r"Warrant|Unit|Right", case=False, na=False)]
    av["key"] = av["name"].map(norm)
    ti = pd.read_csv(os.path.join(DATA, "tiingo_supported_tickers.csv"), keep_default_na=False, na_values=[""])
    ti = ti[ti["exchange"].isin(US_EXCHANGES) & (ti["assetType"] == "Stock")].set_index("ticker")
    rows = []
    for cik, r in big.iterrows():
        yrs = r.index[r >= FLOOR]
        last = int(r.last_valid_index())
        row = dict(cik=int(cik), name=names.get(cik, ""), max_float_bn=round(float(r.max()) / 1e9, 2),
                   first_2B=int(yrs.min()), last_float_year=last, listed_now=int(cik) in now,
                   ticker=now.get(int(cik)), ticker_source="sec" if int(cik) in now else None)
        if not row["listed_now"]:
            m = av[av["key"] == norm(row["name"])]
            if len(m):
                m = m.assign(d=pd.to_datetime(m["delistingDate"], errors="coerce"))
                ok = m[m["d"].isna() | (m["d"].dt.year >= last)]
                pick = (ok if len(ok) else m).sort_values("d").iloc[0]
                row.update(ticker=pick["symbol"], ticker_source="alphavantage", delisted=pick["delistingDate"])
        t = row["ticker"]
        if t in ti.index:
            tr = ti.loc[[t]].iloc[-1]
            row.update(tiingo_start=tr["startDate"], tiingo_end=tr["endDate"])
        rows.append(row)
    df = pd.DataFrame(rows)
    gone = df[~df["listed_now"]]
    # a reused ticker (LZ: Lubrizol to 2011, LegalZoom since 2021) is caught by its dates: Tiingo's
    # history must cover the years the company was worth FLOOR or more
    ts, te = pd.to_datetime(df.get("tiingo_start"), errors="coerce"), pd.to_datetime(df.get("tiingo_end"), errors="coerce")
    covers = (ts.dt.year <= df["first_2B"]) & (te.dt.year >= df["last_float_year"])
    # an old EDGAR id whose ticker still trades (a re-organisation: Broadcom 2018, Disney 2019) is
    # priced by yfinance like any listed stock
    df["ticker_trades_now"] = df["ticker"].isin(set(now.values()))
    df["need_tiingo"] = ~df["listed_now"] & ~df["ticker_trades_now"] & covers
    df.to_csv(OUT, index=False)
    print(f"companies with public float >= ${FLOOR / 1e9:.0f}B at some mid-year {min(YEARS)}-{max(YEARS)}: {len(df)}")
    print(f"  listed today (yfinance can price them): {int(df['listed_now'].sum())}")
    print(f"  gone from EDGAR's ticker list: {len(gone)}; matched to an old ticker: {int(gone['ticker'].notna().sum())}; "
          f"of those with Tiingo prices: {int(df['need_tiingo'].sum())}")
    by = gone.groupby("last_float_year").agg(gone=("cik", "size"), matched=("ticker", lambda s: s.notna().sum()))
    by["with_tiingo"] = df[df["need_tiingo"]].groupby("last_float_year").size()
    print("\ngone companies by the last year they reported a float:\n" + by.fillna(0).astype(int).to_string())
    print("\nlargest gone companies without a Tiingo match:")
    miss = gone[~df.loc[gone.index, "need_tiingo"]].sort_values("max_float_bn", ascending=False)
    print(miss[["name", "max_float_bn", "first_2B", "last_float_year", "ticker"]].head(15).to_string(index=False))
    print("\nwrote", OUT)
    return df


def sample_path(name=""):
    return os.path.join(DATA, f"growth_sample{name}.csv")


def sample(df, p, seed=2013, first=2013, name=""):
    """The new guys (first at FLOOR in `first` or later) that no other sample holds, each drawn
    with probability p, listed or gone alike."""
    import glob
    import numpy as np
    new = df[df["first_2B"] >= first].copy()
    others = [f for f in glob.glob(os.path.join(DATA, "growth_sample*.csv")) if f != sample_path(name)]
    taken = set().union(*[set(pd.read_csv(f)["cik"]) for f in others]) if others else set()
    new = new[~new["cik"].isin(taken)]
    new["drawn"] = np.random.default_rng(seed).random(len(new)) < p
    s = new[new["drawn"]].drop(columns="drawn")
    s["prices_from"] = np.where(s["listed_now"] | s["ticker_trades_now"], "yfinance",
                                np.where(s["need_tiingo"], "tiingo", "none"))
    s.to_csv(sample_path(name), index=False)
    c = s["prices_from"].value_counts()
    print(f"\nsample of the new guys at {p:.1%} (seed {seed}): {len(s)} companies -- {c.get('yfinance', 0)} still listed "
          f"(yfinance), {c.get('tiingo', 0)} gone with Tiingo history, {c.get('none', 0)} gone without prices")
    print(f"wrote {sample_path(name)}" + (f" (excluding {len(taken)} companies of earlier samples)" if taken else ""))
    return s


if __name__ == "__main__":
    import sys
    frame = main()
    if "--sample" in sys.argv:
        opt = lambda f, d: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
        sample(frame, float(opt("--sample", 0.125)), seed=int(opt("--seed", 2013)), name=opt("--name", ""))
