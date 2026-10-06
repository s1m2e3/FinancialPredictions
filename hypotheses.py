"""A few PRE-STATED rules, tested head to head instead of searched for.

The tree search tries hundreds of rules on 14 years, and with honest (cluster-robust)
standard errors none of them can be confirmed: the data are too few for that many looks.
Here a handful of rules are FIXED in advance and each is compared with one reference on
the same days, so the only multiple comparison is the short list below, corrected for
with Holm's method.

    H1  EBTDA track record   buy the 30% with the best 3-year EBTDA / assets in their
                             sector (100% invested)                vs buy all (by size)
    H2  contrarian exposure  buy all, but invest 50% while the S&P model's 6-month P(up)
                             is unusually HIGH (spxz_pup_126d > 1)   vs buy all (by size)
    H3  volatility target    buy all, 50% / 20% invested in the top quarter / tenth of the
                             S&P's forecast volatility              vs buy all (by size)
    H4  the outside pool     buy all (both pools)                   vs buy all S&P members
    H5  revenue growth       buy the 30% fastest growers            vs buy all (by size)

WHICH PERIOD IS CLEAN FOR WHICH RULE. H1 was chosen because it did well on 2020-2026 (it
became the search's seed after those results were seen), so for H1 the TRAINING period
2006-2019 is the untouched test and 2020-2026 is where it was found. H2 comes from the S&P
model's forecasts being contrarian, measured over 2006-2026, so no period is clean for it:
its result here is descriptive. H3-H5 were fixed as baselines before any result was read.
All periods are shown; read each rule on its clean period.

THE TEST. Each rule and its reference run through the simulator as one continuous
portfolio over the period (same costs, allocation and universe as portfolio_bt.py). The
difference in real certainty-equivalent return is bootstrapped PAIRED, in 126-day blocks
(6 months: rules that act on regimes are persistent, and 21-day blocks would understate
the uncertainty the way the naive standard error did); p is one-sided (difference > 0),
and the Holm-adjusted p is the one to read.

Run from the repository root:  python hypotheses.py   (writes results/portfolio_bt/hypotheses.md)
"""
import os

import numpy as np

BLOCK = 126
PERIODS = {"training 2006-2019": ("2006-01-01", "2019-12-31"),
           "validation 2020-2021": ("2020-01-01", "2021-12-31"),
           "test 2022-2026": ("2022-01-01", None)}


def holm(p):
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def rules(env, pb):
    """{name: (rule's (stock bank, exposure bank), reference's)}."""
    from portfolio_env import EXPOSURE_ACTIONS, constant_bank, level_index
    base = dict(pb.baselines(env))
    buy_all = base["buy all (by size)"]
    en = env.exposure_names
    contrarian = constant_bank(en, len(EXPOSURE_ACTIONS), level_index(1.0), EXPOSURE_ACTIONS)
    law50 = np.zeros_like(contrarian["default"])
    law50[-1, level_index(0.5)] = 1.0
    contrarian["clauses"].append([[en.index("spxz_pup_126d"), 1.0, False]])      # > 1: 50%
    contrarian["laws"].append(law50)
    return {
        "H1 EBTDA track record": (base["EBTDA track record (top 30%)"], buy_all),
        "H2 contrarian exposure": ((buy_all[0], contrarian), buy_all),
        "H3 volatility target": (base["buy all + volatility target"], buy_all),
        "H4 the outside pool": (buy_all, base["buy all S&P members"]),
        "H5 revenue growth": (base["revenue growth (top 30%)"], buy_all),
    }


def main():
    import pandas as pd
    import portfolio_bt as pb
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                   holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
    H = rules(env, pb)
    lines = ["# Pre-stated hypotheses", "", __doc__.split("THE TEST.")[0].strip(), "",
             f"Paired bootstrap of the real CER difference, blocks of up to {BLOCK} days (at least ~10 per "
             f"period: 126 on 2006-2019 and 2022-2026, 50 on 2020-2021); Holm-adjusted over the "
             f"{len(H)} hypotheses within each period.", ""]
    for label, (a, b) in PERIODS.items():
        b = b or pb.END
        i0 = np.searchsorted(panel.dates, np.datetime64(a))
        i1 = np.searchsorted(panel.dates, np.datetime64(b), "right")
        pe = pb.world(panel, F, M, sn, mn, a, b, T=int(i1 - i0 - 2), gamma=env.risk_aversion)
        s = pe.sample_starts(1, np.random.default_rng(0))
        real = {}
        def series(pair):
            key = id(pair[0]), id(pair[1])
            if key not in real:
                out = pe.rollout(*pair, s, pe.T, daily=True)
                real[key] = (1 + out["daily"][0]) / (1 + out["infl"][0]) - 1
            return real[key]
        rows = []
        for name, (rule, ref) in H.items():
            # up to BLOCK days, but at least ~10 blocks per period (bt_eval.block_len)
            import bt_eval
            blk = bt_eval.block_len(len(series(rule)), BLOCK)
            gap, p = pb.gap_bootstrap(series(rule), series(ref), lambda r: pb.cer(r, env.risk_aversion), block=blk)
            rows.append(dict(hypothesis=name, **{"CER difference %/yr": round(gap, 3), "p": round(p, 3)}))
        df = pd.DataFrame(rows)
        df["p (Holm)"] = holm(df["p"]).round(3)
        print(f"\n{label}\n" + df.to_string(index=False), flush=True)
        lines += [f"## {label}", "", "```", df.to_string(index=False), "```", ""]
    path = os.path.join(pb.ROOT, "results", "portfolio_bt", "hypotheses.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
