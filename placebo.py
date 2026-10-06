"""The null distribution of the whole pipeline: the same search on data with NO signal.

With hundreds of candidate rules and one history, some rule always looks good. The question
"did it learn something?" is answered by running the ENTIRE pipeline -- the same stages,
folds, acceptance test and check gate as portfolio_bt.py -- on placebo data where no rule can
have an edge, many times, and asking where the real run falls in that distribution.

THE PLACEBO. Prices, the universe, company sizes, costs and the benchmark stay REAL, so buy-
all is exactly the real buy-all. Only what the trees read is broken:
    stock features   on each day the feature rows of the tradable stocks are permuted among
                     them: stock i is shown stock j's features, so no selection rule can know
                     which stock will do well, while every feature's cross-section, and each
                     day's market conditions, keep their real distribution
    market features  the whole block is shifted in time by a random 2-10 years (circularly,
                     within its valid span): each series keeps its persistence and its
                     regimes, but no longer lines up with what the market did next
Anything the pipeline reports on this data -- held-out gains, check-year gains, validation and
test gaps to buy-all -- is what selection alone produces.

    python placebo.py --reps 20 [--dd W]      run (resumable: finished replicates are kept)
    python placebo.py --summary --real DIR    the null distribution, and where the real run
                                              (a portfolio_bt results directory) falls in it

Each replicate is a full four-stage search on the placebo data (as long as one sweep weight)
and holds one extra copy of the feature panel in memory. Writes results/placebo/dd<W>/.
"""
import json
import os
import sys

import numpy as np

import bt_eval as ev
import portfolio_bt as pb

OUT = os.path.join(pb.ROOT, "results", "placebo", pb.RUN_TAG)
METRICS = ("validation gap vs buy all", "test gap vs buy all", "check gain (sum over stages)",
           "arms (stocks + exposure)")


def placebo_features(panel, F, M, sn, seed):
    """(F, M) with the stock rows permuted within each day's tradable set and the market
    block shifted in time (see the module docstring)."""
    rng = np.random.default_rng(seed)
    sig = sn.index("sig_21d")
    tradable = panel.universe & np.isfinite(F[:, :, sig]) & np.isfinite(panel.open)
    F2 = F.copy()
    for t in np.where(tradable.any(1))[0]:
        idx = np.where(tradable[t])[0]
        if len(idx) > 1:
            F2[t, idx] = F[t, rng.permutation(idx)]
    M2 = M.copy()
    valid = np.where(np.isfinite(M).all(1))[0]
    if len(valid):
        v0 = valid[0]
        span = len(M) - v0
        shift = int(rng.integers(504, max(505, min(2520, span - 504))))
        M2[v0:] = np.roll(M[v0:], shift, axis=0)
    return F2, M2


def evaluate(panel, F, M, sn, mn, banks, gamma, state):
    """The replicate's numbers, the same way as for the real run."""
    buy_all = (pb.small(banks["_env"].default_bank("stocks")), banks["_env"].default_bank("exposure"))
    pair = (banks["stocks"], banks["exposure"])
    out = {}
    for label, (a, b) in (("validation", ("2020-01-01", pb.VAL_END)), ("test", ("2022-01-01", pb.END))):
        paths = ev.period_paths(panel, F, M, sn, mn, {"trees": pair, "buy all": buy_all}, a, b, gamma)
        out[f"{label} gap vs buy all"] = ev.paired(paths["trees"]["real"], paths["buy all"]["real"], gamma)[0]
    out["check gain (sum over stages)"] = float(sum((s.get("check") or {}).get("diff", 0.0)
                                                    for s in state["stages"] if not (s.get("check") or {}).get("reverted")))
    out["arms (stocks + exposure)"] = len(banks["stocks"]["clauses"]) + len(banks["exposure"]["clauses"])
    return out


def run(reps):
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    os.makedirs(OUT, exist_ok=True)
    for r in range(reps):
        done = os.path.join(OUT, f"rep{r:02d}.json")
        if os.path.exists(done):
            continue
        print(f"\n##### [{ev.stamp()}] placebo replicate {r} of {reps}", flush=True)
        F2, M2 = placebo_features(panel, F, M, sn, seed=1000 + r)
        env = pb.world(panel, F2, M2, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                       holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
        # from buy all (the search's neutral start): compare with a real run started the same
        # way, python portfolio_bt.py 3 --seed none
        banks, state = ev.run_stages(env, os.path.join(OUT, f"rep{r:02d}"), run_seed=11 + 100 * r)
        banks["_env"] = env
        res = evaluate(panel, F2, M2, sn, mn, banks, env.risk_aversion, state)
        with open(done, "w") as fh:
            json.dump(dict(rep=r, seed=1000 + r, **res), fh, indent=1)
        print(f"  replicate {r}: " + ", ".join(f"{k} {v:+.2f}" for k, v in res.items()), flush=True)
        del env, F2, M2


def summary(real_dir):
    """The null distribution of each metric and the real run's empirical p-value in it."""
    reps = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")]
    if not reps:
        raise SystemExit(f"no replicates in {OUT} yet: run python placebo.py --reps N")
    real = None
    if real_dir:
        from btind.runlog import bank_from_json
        panel = pb.load_panel()
        F, M, sn, mn = pb.load_features()
        env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                       holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
        tree = lambda f: bank_from_json(json.load(open(os.path.join(real_dir, f))))
        with open(os.path.join(real_dir, "run", "state.json")) as fh:
            state = json.load(fh)
        real = evaluate(panel, F, M, sn, mn, dict(stocks=tree("stock_tree.json"), exposure=tree("exposure_tree.json"), _env=env),
                        env.risk_aversion, state)
    lines = [f"# Placebo null distribution ({len(reps)} replicates, drawdown weight {pb.DD_WEIGHT:g})", "",
             "| metric | null mean | null 90th pct | null max | real run | p (null >= real) |", "|---|---|---|---|---|---|"]
    for m in METRICS:
        v = np.array([r[m] for r in reps], float)
        rv = real[m] if real else np.nan
        p = (1 + np.sum(v >= rv)) / (1 + len(v)) if real else np.nan
        lines.append(f"| {m} | {v.mean():+.2f} | {np.quantile(v, 0.9):+.2f} | {v.max():+.2f} | {rv:+.2f} | {p:.3f} |")
    lines += ["", "p is the share of placebo replicates doing at least as well as the real run (with the usual +1): "
              "below 0.05 the real run did better than selection luck alone explains."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary(pb._flag_value("--real"))
    else:
        run(int(pb._flag_value("--reps") or 20))
