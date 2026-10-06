"""Stability selection: which rules come back when the search is rerun on other data splits.

One greedy search through a noisy landscape is fragile: every sweep so far grew different
arms, and the few ideas that kept coming back (the exposure tree cutting back when the S&P
model is unusually bullish) were the most believable. This makes that systematic. Each
replicate reruns the full four-stage search with
    a different split of the six two-year blocks of non-check training years into the
    held-out folds A and B (three blocks each: 20 possible splits, balanced or not), and
    different proposal seeds,
and records what the kept trees USE: every guard literal (input, direction) of every arm, and
the inputs each law weights (btind's sparse laws keep two). A rule that recurs in most
replicates is structure; one that appears once is the search's luck.

    python stability.py --reps 12 [--dd W]    run (resumable)
    python stability.py --summary [--dd W]    how often each guard and law input recurs

Writes results/stability/dd<W>/.
"""
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np

import bt_eval as ev
import portfolio_bt as pb

OUT = os.path.join(pb.ROOT, "results", "stability", pb.RUN_TAG)
BLOCKS = [[2006, 2007], [2008, 2009], [2010, 2011], [2013, 2014], [2016, 2017], [2018, 2019]]


def split(r):
    """Replicate r's folds: the r-th of the 20 ways to put three of the six blocks in A."""
    combos = list(itertools.combinations(range(len(BLOCKS)), 3))
    a = combos[(r * 7) % len(combos)]                 # stride through them, not in lexicographic order
    A = [y for i in a for y in BLOCKS[i]]
    B = [y for i in range(len(BLOCKS)) if i not in a for y in BLOCKS[i]]
    return {"A": A, "B": B}


def uses(bank, names):
    """(guard literals, law inputs) of a tree: {"input > / <=": arms}, {"agent: input": laws}."""
    from btind.memory import mem_names
    z = mem_names(names, None)
    guards = Counter(f"{names[j]} {'<=' if neg else '>'}" for cl in bank["clauses"] for j, _, neg in cl)
    law_in = Counter()
    for th in bank["laws"] + [bank["default"]]:
        th = np.asarray(th)
        d = th[:-1] - th[:-1].mean(axis=1, keepdims=True) if th.shape[1] > 1 else th[:-1]
        for j in np.where(np.abs(d).max(axis=1) > 1e-9)[0]:
            if j < len(z):
                law_in[z[j]] += 1
    return guards, law_in


def run(reps):
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    os.makedirs(OUT, exist_ok=True)
    for r in range(reps):
        done = os.path.join(OUT, f"rep{r:02d}.json")
        if os.path.exists(done):
            continue
        folds = split(r)
        print(f"\n##### [{ev.stamp()}] stability replicate {r}: A {folds['A']}  B {folds['B']}", flush=True)
        env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                       holdout=folds["A"], check=pb.CHECK_YEARS)
        # from buy all, so what recurs is what the search finds, not what the seed brought
        banks, state = ev.run_stages(env, os.path.join(OUT, f"rep{r:02d}"), folds=folds, run_seed=11 + 100 * r)
        rec = dict(rep=r, folds=folds, reverted=[bool((s.get("check") or {}).get("reverted")) for s in state["stages"]])
        for agent, names in (("stocks", env.stock_names), ("exposure", env.exposure_names)):
            g, l = uses(banks[agent], names)
            rec[agent] = dict(arms=len(banks[agent]["clauses"]), guards=dict(g), law_inputs=dict(l))
        with open(done, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"  replicate {r}: stocks {rec['stocks']['arms']} arms, exposure {rec['exposure']['arms']} arms, "
              f"stages reverted {sum(rec['reverted'])}", flush=True)
        del env


def summary():
    reps = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")]
    if not reps:
        raise SystemExit(f"no replicates in {OUT} yet: run python stability.py --reps N")
    n = len(reps)
    lines = [f"# Stability selection ({n} replicates, drawdown weight {pb.DD_WEIGHT:g})", "",
             "Share of replicates whose kept tree uses each guard (input and direction) or law input.", ""]
    for agent in ("stocks", "exposure"):
        for kind, title in (("guards", "guards"), ("law_inputs", "law inputs")):
            c = Counter()
            for r in reps:
                c.update({k: 1 for k in r[agent][kind]})       # once per replicate
            lines += [f"## {agent} tree: {title}", "", "| rule | replicates | share |", "|---|---|---|"]
            lines += [f"| {k} | {v} | {v / n:.0%} |" for k, v in c.most_common()] or ["| (none) | 0 | 0% |"]
            lines.append("")
        arms = [r[agent]["arms"] for r in reps]
        lines.append(f"{agent}: arms per replicate {arms}; empty in {sum(a == 0 for a in arms)} of {n}\n")
    lines.append("A rule used in at least half the replicates is worth believing; one used once is noise.")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        run(int(pb._flag_value("--reps") or 12))
