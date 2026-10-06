"""The elite front: the trees no other tree beats on both objectives.

Every tree pair a portfolio_bt.py run adopts is archived with its scores (run/adopted.jsonl
of each drawdown weight's directory): the risk-adjusted real return gap to the S&P 500
(%/yr, higher is better) and the maximum-drawdown gap (points, lower is better: negative
means shallower falls than the market), on the held-out years, the CHECK years and
validation 2020-2021.

A tree is ELITE when no other tree has a gap at least as high AND a drawdown gap at least
as low, one of them strictly: the front of the trade-off, whatever weight produced it. It is
chosen on the CHECK years (portfolio_bt.CHECK_YEARS), which no search reads. Not on the
held-out years: every move of the search was accepted on them, and a front chosen there
picked exactly the trees that had fitted them -- the first two sweeps' "elite" trees scored
+5 to +22 %/yr on the held-out years and -2.6 to -14.7 on 2020-2021. Runs recorded before
the check years existed have no check scores and fall back to the held-out years, with a
warning. The validation column says how each elite tree held up on years nothing was chosen
on; the test period (2022-2026) is not read here.

ONE WORLD AT A TIME. Only the runs made with the same flags are compared -- the same run
directory suffix as portfolio_bt.py's RUN_TAG after dd<W> (--max-stocks N, --voltarget,
--every N, --inverse-vol): a 20-stock tree and an unlimited one are scored in different
worlds. sweep.py passes its flags on.

Run from the repository root:  python elite.py [the sweep's flags, e.g. --max-stocks 20]
(writes results/portfolio_bt/elite<suffix>.json and elite_front<suffix>.png)
"""
import glob
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "results", "portfolio_bt")


def suffix():
    """The run-directory suffix of the flags on the command line (portfolio_bt.RUN_TAG
    after dd<W>): "" for the plain runs, "_top20" for --max-stocks 20, ..."""
    import portfolio_bt as pb
    return pb.RUN_TAG[len(f"dd{pb.DD_WEIGHT:g}"):]


def load(tag=""):
    """Every archived tree of the runs with directory suffix `tag` once (identical pairs
    from several runs or resumes collapse)."""
    rows, seen = [], set()
    paths = [p for p in sorted(glob.glob(os.path.join(BASE, "dd*", "run", "adopted.jsonl")))
             if re.fullmatch(r"dd[-0-9.e]+" + re.escape(tag), os.path.basename(os.path.dirname(os.path.dirname(p))))]
    for path in paths:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                key = hashlib.sha1(json.dumps([r["stock_bank"], r["exposure_bank"]],
                                              sort_keys=True).encode()).hexdigest()
                if key in seen or "held_out_gap" not in r:
                    continue
                seen.add(key)
                r["key"] = key[:10]
                rows.append(r)
    return rows


def front(rows, gap, dd):
    """Indices of the non-dominated rows: maximise gap, minimise dd."""
    keep = []
    for i, a in enumerate(rows):
        dominated = any(b[gap] >= a[gap] and b[dd] <= a[dd] and (b[gap] > a[gap] or b[dd] < a[dd])
                        for j, b in enumerate(rows) if j != i)
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: rows[i][dd])


def draw(rows, elite, png, on):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    weights = sorted({r["dd_weight"] for r in rows})
    cm = plt.get_cmap("viridis", max(len(weights), 2))
    chosen = {"check": "check years (chosen on, never searched)",
              "held_out": "held-out years (chosen on -- and searched on)"}[on]
    for ax, (gap, dd, title) in zip(axes, ((f"{on}_gap", f"{on}_dd", chosen),
                                           ("validation_gap", "validation_dd", "validation 2020-2021 (never used)"))):
        for wi, w in enumerate(weights):
            pts = [r for r in rows if r["dd_weight"] == w and dd in r]
            ax.scatter([r[dd] for r in pts], [r[gap] for r in pts], s=18, color=cm(wi), alpha=0.5,
                       label=f"trees of the run with drawdown weight {w:g}")
        e = [rows[i] for i in elite if dd in rows[i]]
        ax.plot([r[dd] for r in e], [r[gap] for r in e], "k-", lw=1, alpha=0.6)
        ax.scatter([r[dd] for r in e], [r[gap] for r in e], s=90, facecolors="none", edgecolors="black",
                   lw=1.8, label=f"elite (non-dominated on the {on.replace('_', '-')} years)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        ax.grid(True, color="0.88", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("max drawdown vs the S&P 500, points  (left = shallower falls)")
        ax.set_ylabel("risk-adjusted real return vs the S&P 500, %/yr  (up = better)")
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Elite front: every adopted tree, both objectives; (0, 0) is the S&P 500 itself", fontsize=12)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    tag = suffix()
    rows = load(tag)
    if not rows:
        raise SystemExit(f"no archived trees yet in results/portfolio_bt/dd*{tag}: run portfolio_bt.py first")
    with_check = [r for r in rows if "check_gap" in r]
    if with_check:
        on = "check"
        if len(with_check) < len(rows):
            print(f"note: {len(rows) - len(with_check)} trees from runs without check years are left out")
        rows = with_check
    else:
        on = "held_out"
        print("WARNING: no run recorded check-year scores; choosing on the held-out years, which the "
              "search was accepted on -- this front favours trees that fitted them")
    elite = front(rows, f"{on}_gap", f"{on}_dd")
    out = [dict({k: v for k, v in rows[i].items()}, rank=n, chosen_on=on) for n, i in enumerate(elite)]
    with open(os.path.join(BASE, f"elite{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    draw(rows, elite, os.path.join(BASE, f"elite_front{tag}.png"), on)
    lab = on.replace("_", "-")
    print(f"{len(rows)} distinct trees, {len(elite)} elite ({lab} years):")
    print(f"  {'weight':>6s} {'stage':>5s} {lab + ' gap':>13s} {lab + ' dd':>12s} {'val gap':>8s} {'val dd':>7s}")
    for i in elite:
        r = rows[i]
        print(f"  {r['dd_weight']:6g} {r['stage']:5d} {r[f'{on}_gap']:+13.2f} {r[f'{on}_dd']:+12.2f} "
              f"{r.get('validation_gap', np.nan):+8.2f} {r.get('validation_dd', np.nan):+7.2f}   [{r['key']}]")
    print("wrote", os.path.join(BASE, f"elite{tag}.json"), f"and elite_front{tag}.png")
