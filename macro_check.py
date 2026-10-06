"""Checks the hourly panel before anything trains on it (macro_data.build): coverage per
instrument, spreads by hour of day, jumps (a price that moves 10+ hourly standard deviations
in one bar: a bad print, or a CFD rolling to the next futures contract -- a move the account
never earned), the carry each pair's long side earns, and how many instruments are tradable.

    python macro_check.py [--universe fx|macro12]
"""
import numpy as np
import pandas as pd

import macro_data as md


def main():
    p = md.build()
    X, sim = md.features(p)
    times = pd.DatetimeIndex(p["times"])
    names = list(p["names"])
    mo = (p["bo"] + p["ao"]) / 2
    mc = (p["bc"] + p["ac"]) / 2
    print(f"universe {md.UNIVERSE}: {len(names)} instruments available; grid {len(times)} hours, "
          f"{times[0]} .. {times[-1]}")
    tr = (times >= "2013-01-01") & (times < "2020-01-01")
    weeks = (times[tr][-1] - times[tr][0]).days / 7
    carry = sim["carry"]
    print(f"\n{'':8s} {'first bar':>11s} {'last bar':>11s} {'bars/wk':>7s} {'tradable':>8s} {'half-spread bp':>14s} "
          f"{'best hr':>8s} {'worst hr':>9s} {'jumps':>5s} {'carry %/yr 2013 / 2023':>22s}")
    for i, n in enumerate(names):
        bar = p["bar"][:, i]
        hs = 1e4 * (p["ao"][:, i] - p["bo"][:, i]) / (2 * mo[:, i])
        byh = pd.Series(hs[bar]).groupby(times[bar].hour).median()
        lr = np.log(pd.Series(mc[:, i]).ffill().to_numpy())
        r = np.r_[np.nan, np.diff(lr)]
        sd = pd.Series(r).rolling(480, min_periods=100).std().shift(1).to_numpy()
        with np.errstate(invalid="ignore"):
            jump = np.abs(r) > 10 * sd
        c13 = 100 * np.nanmean(carry[(times.year == 2013), i])
        c23 = 100 * np.nanmean(carry[(times.year == 2023), i])
        print(f"{n:8s} {str(times[bar][0])[:10]:>11s} {str(times[bar][-1])[:10]:>11s} {bar[tr].sum() / weeks:7.1f} "
              f"{100 * sim['avail'][times >= '2012-01-01', i].mean():7.1f}% {np.nanmedian(hs[bar]):14.2f} "
              f"{byh.min():5.2f}@{byh.idxmin():02d} {byh.max():6.2f}@{byh.idxmax():02d} {int(jump.sum()):5d} "
              f"{c13:+10.2f} / {c23:+6.2f}")
        if jump.sum():
            k = np.argsort(-np.abs(np.where(jump, r, 0)))[:3]
            print("           largest: " + ", ".join(f"{str(times[j])[:13]} {100 * r[j]:+.2f}% ({abs(r[j]) / sd[j]:.0f} sd)"
                                                      for j in k if jump[j]))
    av = sim["avail"]
    per_year = pd.Series(av.sum(1), index=times).groupby(times.year).mean()
    print("\ninstruments tradable in an average hour, by year:", {y: round(v, 1) for y, v in per_year.items()})
    # month coverage: any month an instrument misses (fewer than 300 bars) inside its span
    print("\nmonths with fewer than 300 bars inside an instrument's span:")
    for i, n in enumerate(names):
        s = pd.Series(p["bar"][:, i], index=times).resample("MS").sum()
        s = s[(s.index >= s[s > 0].index[0]) & (s.index < s.index[-1])]
        short = s[s < 300]
        print(f"  {n:8s} " + (", ".join(f"{d:%Y-%m} ({int(v)})" for d, v in short.items()) if len(short) else "none"))
    print("\nfeatures (share of non-zero values over tradable rows, 2012 on):")
    m = av & (times >= "2012-01-01")[:, None]
    nz = {f: round(float(np.mean(X[:, :, k][m] != 0)), 3) for k, f in enumerate(sim["features"])}
    print(" ", nz)


if __name__ == "__main__":
    main()
