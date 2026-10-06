"""Evaluate a stock tree and an exposure tree from DIFFERENT runs together.

The two trees of one run were grown as partners, but nothing stops pairing the stock tree
of one drawdown weight with the exposure tree of another (they read the same features; the
exposure tree sees the stock tree only through buy_frac and the portfolio state). This runs
the pair through the same simulator, costs and periods as portfolio_bt.py's report, next to
each run's own pair and the baselines, and tests the differences that matter with the same
21-day block bootstrap, PAIRED (the same days, so the market's own moves cancel):
    the pair vs the stock tree fully invested   what the borrowed exposure tree adds
    the pair vs the EBTDA track-record rule     against the best simple rule
    the pair vs each run's own pair
The test period (2022-2026) is reported, but choosing between pairs on it spends it: say
which pairing you would pick from validation first.

    python pair_trees.py --stocks DIR --exposure DIR [--legacy]

DIR is a run's results directory (holding stock_tree.json and exposure_tree.json).
--legacy evaluates trees grown before the universe change (a 20-name outside pool, no
separate outside cap), on the data files built for them: run it BEFORE rebuilding the data.
Writes <stocks DIR>/../pairs/<stocks>+<exposure>.md
"""
import json
import os
import sys

import numpy as np


def _arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def main():
    s_dir, e_dir = _arg("--stocks"), _arg("--exposure")
    if not (s_dir and e_dir):
        raise SystemExit(__doc__)
    import stocks_data
    if "--legacy" in sys.argv:
        stocks_data.OUTSIDE_SIZE, stocks_data.OUTSIDE_EXIT = 20, 30
    import pandas as pd
    import portfolio_bt as pb
    if "--legacy" in sys.argv:
        pb.OUTSIDE_MAX_WEIGHT = pb.MAX_WEIGHT
    from btind.runlog import bank_from_json

    panel = stocks_data.load_panel()
    F, M, sn, mn = pb.load_features()
    # the training world exactly as portfolio_bt.py builds it (the volatility-target baseline
    # takes its thresholds from its search days, so the pools must match)
    env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                   holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
    tree = lambda d, f: bank_from_json(json.load(open(os.path.join(d, f))))
    trees = {"S": (tree(s_dir, "stock_tree.json"), tree(s_dir, "exposure_tree.json")),
             "E": (tree(e_dir, "stock_tree.json"), tree(e_dir, "exposure_tree.json"))}
    for k, (sb, eb) in trees.items():
        if list(sb["names"]) != env.stock_names or list(eb["names"]) != env.exposure_names:
            raise SystemExit(f"the trees in {s_dir if k == 'S' else e_dir} were grown on other inputs than "
                             "these data files provide (rebuilt data? try --legacy before rebuilding)")
    sname, ename = os.path.basename(os.path.normpath(s_dir)), os.path.basename(os.path.normpath(e_dir))
    full = env.default_bank("exposure")
    pairs = dict(pb.baselines(env))
    pairs[f"{sname} stock tree, 100% invested"] = (trees["S"][0], full)
    pairs[f"{sname} own pair"] = trees["S"]
    pairs[f"{ename} own pair"] = trees["E"]
    pair = f"PAIR: {sname} stocks + {ename} exposure"
    pairs[pair] = (trees["S"][0], trees["E"][1])
    tests = [(pair, f"{sname} stock tree, 100% invested"), (pair, "EBTDA track record (top 30%)"),
             (pair, f"{sname} own pair"), (pair, f"{ename} own pair")]

    lines = [f"# {pair}", "", f"stock tree from `{s_dir}`, exposure tree from `{e_dir}`"
             + (" (legacy universe)" if "--legacy" in sys.argv else ""), ""]
    for label, (a, b) in pb.PERIODS.items():
        df = pd.DataFrame(pb.report(pairs, panel, F, M, sn, mn, a, b)).T.round(3)
        print(f"\n{label}\n" + df.to_string(), flush=True)
        lines += [f"## {label}", "", "```", df.to_string(), "```", ""]
        # paired differences, on the same days
        i0 = np.searchsorted(panel.dates, np.datetime64(a))
        i1 = np.searchsorted(panel.dates, np.datetime64(b), "right")
        pe = pb.world(panel, F, M, sn, mn, a, b, T=int(i1 - i0 - 2), gamma=env.risk_aversion)
        s = pe.sample_starts(1, np.random.default_rng(0))
        daily = {}
        for name in {x for t in tests for x in t}:
            out = pe.rollout(*pairs[name], s, pe.T, daily=True)
            infl = out["infl"][0]
            daily[name] = (1 + out["daily"][0]) / (1 + infl) - 1
        rows = ["| comparison | CER difference (%/yr) | p (difference > 0) |", "|---|---|---|"]
        for x, y in tests:
            gap, p = pb.gap_bootstrap(daily[x], daily[y], lambda r: pb.cer(r, env.risk_aversion))
            rows.append(f"| {x} vs {y} | {gap:+.3f} | {p:.3f} |")
            print(f"  {x}  vs  {y}:  {gap:+.3f} %/yr   p {p:.3f}", flush=True)
        lines += ["paired, on the same days (21-day block bootstrap):", ""] + rows + [""]
    out_dir = os.path.join(os.path.dirname(os.path.normpath(s_dir)), "pairs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sname}+{ename}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
