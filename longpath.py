"""Drawdowns on long, CONTIGUOUS paths.

The search scores episodes of 6-9 months, so its drawdown term never sees a fall that
lasts longer -- 2008-2009 is two unrelated half-year windows to it, and the drawdown weight
cannot reward a tree for getting through a whole bear market. This runs tree pairs and the
baselines as ONE continuous portfolio over long periods and reports what an investor
holding them would have lived through:
    max drawdown        the deepest fall from a running high, %
    longest underwater  the longest stretch below a previous high, trading days
    worst 12 months     the worst rolling 252-day return, %
    Calmar              annual return / |max drawdown|
all against the S&P 500 (SPY, dividends in), plus underwater curves (longpath.png).

    python longpath.py [DIR ...]      tree pairs from portfolio_bt results directories
                                      (default: every results/portfolio_bt/dd*)

Evaluation only: nothing is trained. The training periods are shown for completeness; the
trees were chosen on them, so read 2020-2026 for anything out of sample.
"""
import glob
import json
import os
import sys

import numpy as np

import bt_eval as ev
import portfolio_bt as pb

PERIODS = {"2006-2019 (training)": ("2006-01-01", "2019-12-31"),
           "2020-2026 (out of sample)": ("2020-01-01", None),
           "2006-2026": ("2006-01-01", None)}


def stats(daily):
    w = np.cumprod(1 + np.asarray(daily))
    peak = np.maximum.accumulate(w)
    under = w < peak
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    ann = w[-1] ** (252 / len(w)) - 1
    mdd = float((w / peak - 1).min())
    r12 = w[252:] / w[:-252] - 1 if len(w) > 252 else np.array([w[-1] - 1])
    return {"annual %": 100 * ann, "max drawdown %": 100 * mdd, "longest underwater (days)": longest,
            "worst 12 months %": 100 * float(r12.min()), "Calmar": ann / abs(mdd) if mdd < 0 else np.nan}


def main(dirs):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from btind.runlog import bank_from_json
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                   holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
    base = dict(pb.baselines(env))
    pairs = {k: base[k] for k in ("buy all (by size)", "EBTDA track record (top 30%)", "buy all + volatility target")}
    for d in dirs:
        try:
            tree = lambda f: bank_from_json(json.load(open(os.path.join(d, f))))
            sb, eb = tree("stock_tree.json"), tree("exposure_tree.json")
        except FileNotFoundError:
            print("skipped (no trees):", d)
            continue
        if list(sb["names"]) != env.stock_names or list(eb["names"]) != env.exposure_names:
            print("skipped (grown on other inputs):", d)
            continue
        pairs[f"trees {os.path.basename(os.path.normpath(d))}"] = (sb, eb)
    lines = ["# Long contiguous paths", "", __doc__.split("    python")[0].strip(), ""]
    fig, axes = plt.subplots(len(PERIODS), 1, figsize=(13, 3.6 * len(PERIODS)))
    for ax, (label, (a, b)) in zip(axes, PERIODS.items()):
        paths = ev.period_paths(panel, F, M, sn, mn, pairs, a, b or pb.END, env.risk_aversion)
        df = pd.DataFrame({k: stats(v["daily"]) for k, v in paths.items()}).T.round(2)
        print(f"\n{label}\n" + df.to_string(), flush=True)
        lines += [f"## {label}", "", "```", df.to_string(), "```", ""]
        for k, v in paths.items():
            w = np.cumprod(1 + v["daily"])
            ax.plot(v["dates"], 100 * (w / np.maximum.accumulate(w) - 1), lw=2.2 if k == "S&P 500" else 1.1,
                    color="black" if k == "S&P 500" else None, label=k)
        ax.set_title(f"{label}: below the running high", fontsize=10)
        ax.set_ylabel("drawdown %")
        ax.grid(True, color="0.88")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    out = os.path.join(pb.ROOT, "results", "portfolio_bt")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "longpath.png"), dpi=110)
    with open(os.path.join(out, "longpath.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nwrote", os.path.join(out, "longpath.md"), "and longpath.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or sorted(glob.glob(os.path.join(pb.ROOT, "results", "portfolio_bt", "dd*"))))
