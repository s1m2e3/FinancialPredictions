"""Grow the stock-picking and exposure behaviour trees with btind; test them against the S&P 500.

btind comes from the `portfolio-env` branch of s1m2e3/btind, cloned next to this
repository (BTIND_PATH overrides the location); that branch adds the hooks this world
needs (`env.score_bank`, `env.store_root`, held-out scoring through `score_bank`).

SCORE: certainty-equivalent REAL return gap vs the S&P 500 with dividends reinvested (SPY), % per
year (portfolio_env.py):
returns deflated by CPI-U, so idle cash loses what inflation takes; variance charged at the
risk aversion that made 100% S&P optimal on 2006-2019 (gamma, calibrated once, reused on
the later periods).

TRAINING alternates the two trees, each searched with the other fixed as its partner
(btind's vehicle / signal cycle): stocks (exposure fixed at 100%), exposure (with that
stock tree), then both once more. Every move is accepted only by btind's paired rollout
test on 2006-2019 episodes; 2020-2021 and 2022-2026 are never seen by the search.

ONE MODEL: the two trees see everything -- the walk-forward GARCH and price features, their
cross-sectional ranks, the S&P 500 full-model forecasts, the EDGAR fundamentals
(fundamentals.py: sector ranks of EBTDA / assets, margin, 3-year record and change, the
universe rank of revenue growth; random where missing) and `in_sp500`, which tells the tree
whether a stock is an S&P 500 member or one of the Nasdaq-100 growth names outside it
(stocks_data.py), so it can decide how much of the outside pool to hold.

BASELINES are trees too, so they run through the same simulator, costs and allocation:
    S&P 500 (SPY, dividends in)  the reference every score is measured against
    buy all                       every stock by inverse volatility, always 100% invested
    buy all S&P members           the same without the outside pool: what the pool adds
    low volatility / momentum     the 30% least volatile / 20% strongest, 100% invested
    high volatility               the 30% MOST volatile: the survivorship check (see baselines)
    buy all + volatility target   invested 50% when the S&P's forecast 21-day volatility is
                                  in its top quarter of 2006-2019, 25% in its top tenth
    EBTDA track record            the 30% with the best 3-year EBTDA / assets in their sector
    revenue growth                the 30% fastest growers (TTM revenue, year on year)

TRAINING CURVES. Every tree btind accepts is replayed on 200 fixed one-year training
episodes and the $100k budget's best / mean / worst path is drawn to
results/portfolio_bt/training_budget.png while the run goes (training_progress.py, which
also redraws it on demand).

CHECKPOINTS. Every finished stage and every round's best tree are saved under
<results dir>/run/ as they happen. Rerunning the same command after a stop resumes from
there (see main); --fresh archives the run and starts over; --report writes the report
from the checkpoint without training.

SEED. --seed tree.json starts stage 0 from that stock tree (btind's bank_json) instead of
btind's cold start, e.g. arms an earlier run discovered; the search then continues from it
and only accepts moves that beat it. A resumed stage 0 ignores it (its snapshot is newer).

Run from the repository root:  python portfolio_bt.py [rounds per stage] [--seed tree.json] [--fresh | --report]
"""
import json
import os
import sys
import time

import numpy as np

from portfolio_env import (BUY, EXIT, EXPOSURE_ACTIONS, EXPOSURE_LEVELS, STOCK_ACTIONS,
                           PortfolioWorld, calibrate_risk_aversion, constant_bank)
from stock_features import load as _load_features
from stocks_data import END, TRAIN_END, TRAIN_START, VAL_END, load_panel

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "portfolio_bt")
PERIODS = {"validation 2020-2021": ("2020-01-01", VAL_END), "test 2022-2026": ("2022-01-01", END)}


def load_features():
    return _load_features(fundamentals=True)

# With the compiled rollout a 400-episode score costs ~0.1-0.2 s, so btind's budgets can be
# near its defaults. Stages that need btind's own traces or kernels (exploration, memory,
# termination, steps, subtrees) stay off. z = 2: a move must beat the incumbent by two
# standard errors on paired episodes, and lose significantly in no market regime. Gains are
# in percentage points of real certainty-equivalent return per year.
CFG = dict(n_ep=400, T=252, z=2.0, min_gain=0.1, grow_min_gain=0.2,
           grow_pool=60, grow_arms=4, cem_top=6, n_pos=2, max_arity=2, min_n=300,
           cem_iter=8, cem_K=48, cem_sigma=0.35, n_cover=6000, cover_ep=80,
           explore_ep=0, value_laws=False, mem_at=999, beta_at=999, steps_at=999,
           subtree_at=(), kern_at=(), val_ep=1000, val_seed=90210)


def world(panel, F, M, sn, mn, start, end, T=252, agent="stocks", gamma=None):
    gamma = calibrate_risk_aversion(panel, TRAIN_START, TRAIN_END) if gamma is None else gamma
    # the source is part of the world's signature in btind's store: trees grown on another
    # universe or feature set are never reused
    return PortfolioWorld(panel, F, M, sn, mn, start, end, T=T, agent=agent, risk_aversion=gamma,
                          source="yfinance-sp500top100+ndx20+edgar")


def rule(names, n_act, column, threshold, above, action, default):
    """One guarded arm: `action` where column > threshold (above) or <= it, else `default`."""
    bank = constant_bank(names, n_act, default, STOCK_ACTIONS if n_act == 3 else EXPOSURE_ACTIONS)
    law = np.zeros_like(bank["default"])
    law[-1, action] = 1.0
    bank["clauses"].append([(names.index(column), float(threshold), not above)])
    bank["laws"].append(law)
    return bank


def baselines(env):
    """{name: (stock bank, exposure bank)} run through the same simulator."""
    sn, en = env.stock_names, env.exposure_names
    full = env.default_bank("exposure")
    sig = env._M[env._starts, en.index("spx_sig_21d")]
    q75, q90 = np.quantile(sig, [0.75, 0.90])
    vol_target = rule(en, 4, "spx_sig_21d", q90, True, 0, 3)              # top tenth: 25%
    law50 = np.zeros_like(vol_target["default"])
    law50[-1, 1] = 1.0
    vol_target["clauses"].append([(en.index("spx_sig_21d"), float(q75), False)])   # top quarter: 50%
    vol_target["laws"].append(law50)
    return {"buy all (inverse vol)": (env.default_bank("stocks"), full),
            "buy all S&P members": (rule(sn, 3, "in_sp500", 0.5, True, BUY, EXIT), full),
            "low volatility (30%)": (rule(sn, 3, "rank_sig_21d", 0.30, False, BUY, EXIT), full),
            "momentum (top 20%)": (rule(sn, 3, "rank_mom_12_1", 0.80, True, BUY, EXIT), full),
            # the survivorship check: on a list of today's members the most volatile stocks
            # are the ones that became mega-winners; on a point-in-time universe they lag
            "high volatility (top 30%)": (rule(sn, 3, "rank_sig_21d", 0.70, True, BUY, EXIT), full),
            "buy all + volatility target": (env.default_bank("stocks"), vol_target),
            "EBTDA track record (top 30%)": (rule(sn, 3, "rank_ebtda_roa_3y", 0.70, True, BUY, EXIT), full),
            "revenue growth (top 30%)": (rule(sn, 3, "rank_rev_growth", 0.70, True, BUY, EXIT), full)}


def train_stage(env, agent, partner, rounds, seed, cfg=None, init_bank=None, tag=None):
    """Search one agent's tree with `partner` fixed. `init_bank` is the tree this agent left
    at its previous stage: the search continues from it instead of from an empty tree, and
    btind only accepts moves that beat it (against the new partner)."""
    from btind.rlfit import fit
    env.agent, env.partner = agent, partner
    bank, log, held = fit(env, env.names, rounds=rounds, warm=False, cfg=dict(CFG, **(cfg or {})),
                          tag=tag or f"portfolio-{agent}", run_seed=seed, init_bank=init_bank)
    return bank, held


def cer(real, gamma):
    """Certainty-equivalent real return, % per year."""
    return 100 * 252 * (real.mean() - 0.5 * gamma * real.var(ddof=1))


def metrics(daily, rf, infl, gamma, budget=100_000.0):
    real = (1 + daily) / (1 + infl) - 1
    ex = daily - rf
    wealth = np.cumprod(1 + daily)
    dd = wealth / np.maximum.accumulate(wealth) - 1
    return {"real annual return %": 100 * (np.prod(1 + real) ** (252 / len(real)) - 1),
            "real value of $100k": budget * np.prod(1 + real),
            "CER real %/yr": cer(real, gamma),
            "annual vol %": 100 * daily.std() * np.sqrt(252),
            "Sharpe": np.sqrt(252) * ex.mean() / ex.std(),
            "max drawdown %": 100 * dd.min()}


def gap_bootstrap(a, b, stat, n=2000, block=21, seed=0):
    """Circular block bootstrap of stat(a) - stat(b) on paired daily series: (gap, one-sided p)."""
    rng = np.random.default_rng(seed)
    T = len(a)
    gap = stat(a) - stat(b)
    boots = np.empty(n)
    for k in range(n):
        idx = (rng.integers(0, T, int(np.ceil(T / block)))[:, None] + np.arange(block)).ravel()[:T] % T
        boots[k] = stat(a[idx]) - stat(b[idx])
    return gap, float(np.mean(boots - gap >= gap))         # H0: gap <= 0, centred bootstrap


def report(pairs, panel, F, M, sn, mn, start, end):
    """Full-period metrics of each (stock bank, exposure bank) pair and of the S&P 500."""
    a = np.searchsorted(panel.dates, np.datetime64(start))
    b = np.searchsorted(panel.dates, np.datetime64(end), "right")
    env = world(panel, F, M, sn, mn, start, end, T=int(b - a - 2))
    g = env.risk_aversion
    s = env.sample_starts(1, np.random.default_rng(0))
    rows = {}
    for name, (sb, eb) in pairs.items():
        out = env.rollout(sb, eb, s, env.T, daily=True)
        infl = out["infl"][0]
        real = lambda x: (1 + x) / (1 + infl) - 1
        if not rows:
            rows["S&P 500 (SPY, dividends reinvested)"] = dict(metrics(out["bench"][0], out["rf"][0], infl, g),
                                                **{"turnover / yr": 0.0, "CER gap": 0.0, "p (CER gap > 0)": np.nan})
        gap, p = gap_bootstrap(real(out["daily"][0]), real(out["bench"][0]), lambda r: cer(r, g))
        rows[name] = dict(metrics(out["daily"][0], out["rf"][0], infl, g),
                          **{"turnover / yr": out["turnover"][0] * 252 / env.T, "CER gap": gap, "p (CER gap > 0)": p})
    return rows


STAGES = ("stocks", "exposure", "stocks", "exposure")
RUN_DIR = os.path.join(OUT, "run")       # the current run: state.json + its own btind store


# ------------------------------------------------------------------ checkpointing
# A run can be stopped at any moment and restarted with the same command. What survives:
#   state.json       every FINISHED stage's tree and held-out score, written atomically
#                    the moment the stage ends
#   store/           btind's store for THIS run only (env.store_root): the best tree of
#                    every round, tagged by stage and attempt. btind's shared store keeps
#                    only the top 20 trees per world across all runs, so an early round of
#                    a new run could be evicted there; here nothing else competes
# On restart the finished stages are reloaded, and the unfinished stage continues from the
# best round snapshot it reached (same partner, same evaluation seeds, so the scores are
# comparable), with only the rounds it has not run yet; with no snapshot it starts from the
# tree its agent left at its previous stage. A different configuration never resumes: the
# old run directory is archived and a new run starts.

def _config(env, rounds):
    from btind.runlog import env_signature
    sig = {k: v for k, v in env_signature(env).items() if k != "agent"}
    return dict(world=sig, rounds=rounds, cfg=CFG, stock_names=env.stock_names,
                exposure_names=env.exposure_names, n_stocks=int(env._N))


def _save_state(state):
    tmp = os.path.join(RUN_DIR, "state.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, os.path.join(RUN_DIR, "state.json"))     # atomic: never half-written


def _open_run(env, rounds, fresh):
    path = os.path.join(RUN_DIR, "state.json")
    cfg = json.loads(json.dumps(_config(env, rounds)))      # through JSON, as it is stored
    if os.path.exists(path) and not fresh:
        with open(path) as fh:
            state = json.load(fh)
        if state["config"] == cfg:
            print(f"resuming run {state['run_id']}: {len(state['stages'])} of {len(STAGES)} stages done",
                  flush=True)
            return state
        print("the checkpoint in", RUN_DIR, "is from a different configuration", flush=True)
    if os.path.exists(RUN_DIR):                             # archived, never deleted
        old = RUN_DIR + "_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(RUN_DIR)))
        os.replace(RUN_DIR, old)
        print("archived the previous run to", old, flush=True)
    os.makedirs(RUN_DIR)
    state = dict(run_id=time.strftime("%Y%m%d_%H%M%S"), config=cfg, stages=[], complete=False)
    _save_state(state)
    return state


def _stage_snapshot(env, agent, stage):
    """(best round snapshot of an unfinished stage or None, its G, rounds it already ran)."""
    from btind import store
    from btind.runlog import bank_from_json
    env.agent = agent
    p = store._path(env)
    if not os.path.exists(p):
        return None, None, 0
    with open(p) as fh:
        entries = [e for e in json.load(fh)["banks"]
                   if e["tag"].startswith(f"portfolio-{agent}-s{stage}-") and "@r" in e["tag"]]
    if not entries:
        return None, None, 0
    e = max(entries, key=lambda e: e["G"])
    return bank_from_json(e["bank"]), e["G"], len(entries)


def main(rounds=3, fresh=False, seed=None):
    from btind.runlog import bank_from_json, bank_json
    from training_progress import Recorder
    os.makedirs(OUT, exist_ok=True)
    panel = load_panel()
    F, M, sn, mn = load_features()
    env = world(panel, F, M, sn, mn, TRAIN_START, TRAIN_END)
    env._store_root = RUN_DIR
    state = _open_run(env, rounds, fresh)
    # every accepted tree is replayed on fixed training episodes (training_budget.png)
    rec = env._recorder = Recorder(env, RUN_DIR, os.path.join(OUT, "training_budget.png"))
    t0 = time.time()
    banks = {"stocks": None, "exposure": None}
    seed_bank = None
    if seed is not None:
        with open(seed) as fh:
            seed_bank = bank_from_json(json.load(fh))
    for s in state["stages"]:
        banks[s["agent"]] = bank_from_json(s["bank"])
        print(f"  stage {s['stage']} {s['agent']}: {s['arms']} arms, held-out G {s['held_out']:+.3f} (from checkpoint)",
              flush=True)
    for stage in range(len(state["stages"]), len(STAGES)):
        agent = STAGES[stage]
        partner = banks["exposure" if agent == "stocks" else "stocks"]
        snap, snap_G, done = _stage_snapshot(env, agent, stage)
        init = snap if snap is not None else banks[agent]        # None the first time
        if init is None and stage == 0 and seed_bank is not None:
            init = seed_bank
            print(f"  starting from the seed tree {seed} ({len(seed_bank['clauses'])} arms)", flush=True)
        left = max(rounds - done, 0)
        # each (re)start of a stage gets its own tag, so its round snapshots never collide
        attempt = state.setdefault("attempts", {}).get(str(stage), 0) + 1
        state["attempts"][str(stage)] = attempt
        _save_state(state)
        print(f"\n=== stage {stage}: {agent} tree ({time.time() - t0:.0f}s) ===", flush=True)
        if snap is not None:
            print(f"  continuing from its best round snapshot (G {snap_G:+.3f}, {done} rounds done, {left} left)",
                  flush=True)
        rec.stage = stage
        env.agent, env.partner = agent, partner
        start = banks[agent] if banks[agent] is not None else env.default_bank(agent)
        if snap is not None and stage not in rec.rows["stage"]:
            env.on_adopt(start)            # resumed with no history: where the stage began
        env.on_adopt(init if init is not None else start)                            # where it starts
        bank, held = train_stage(env, agent, partner, left, seed=11 + stage, init_bank=init,
                                 tag=f"portfolio-{agent}-s{stage}-a{attempt}")
        env.agent, env.partner = agent, partner
        env.on_adopt(bank)                             # the tree the stage keeps (if new)
        rec.draw()
        banks[agent] = bank
        state["stages"].append(dict(stage=stage, agent=agent, bank=bank_json(bank, env.names),
                                    held_out=held["G"], ci=held["ci"], arms=len(bank["clauses"])))
        _save_state(state)
        print(f"stage {stage} done: {agent} tree with {len(bank['clauses'])} arms, held-out "
              f"(training period) CER gap {held['G']:+.3f} +- {held['ci']:.3f} %/yr  [checkpointed]", flush=True)
    write_report(env, panel, F, M, sn, mn, banks["stocks"], banks["exposure"], _summary(state))
    state["complete"] = True
    _save_state(state)


def _summary(state):
    rows = ["| stage | tree | arms | held-out CER gap (%/yr), training period |", "|---|---|---|---|"]
    rows += [f"| {s['stage']} | {s['agent']} | {s['arms']} | {s['held_out']:+.3f} +- {s['ci']:.3f} |"
             for s in state["stages"]]
    return [f"Run {state['run_id']}: {len(state['stages'])} of {len(STAGES)} stages x "
            f"{state['config']['rounds']} rounds.", ""] + rows


def write_report(env, panel, F, M, sn, mn, stock_bank, exp_bank, summary):
    import pandas as pd
    from btind.memory import emit
    from btind.runlog import bank_json
    t0 = time.time()
    for name, b, names in (("stock_tree", stock_bank, env.stock_names), ("exposure_tree", exp_bank, env.exposure_names)):
        with open(os.path.join(OUT, name + ".json"), "w") as fh:
            json.dump(bank_json(b, names), fh, indent=1)
    pairs = dict(baselines(env))
    pairs["stock tree, 100% invested"] = (stock_bank, env.default_bank("exposure"))
    pairs["stock tree + exposure tree"] = (stock_bank, exp_bank)
    lines = ["# Behaviour-tree portfolio vs the S&P 500", "",
             f"Score: certainty-equivalent REAL return (CPI-deflated), risk aversion gamma = "
             f"{env.risk_aversion:.2f} calibrated on {TRAIN_START[:4]}-{TRAIN_END[:4]} (100% S&P optimal); gaps are portfolio "
             "minus S&P 500 in % per year, p from a 21-day block bootstrap.", "",
             "Universe: each month's 100 largest S&P 500 members of that day plus the 20 largest "
             "Nasdaq-100 members outside the S&P 500, by trailing dollar volume (point-in-time "
             "membership, stocks_data.py). Stocks delisted before today mostly have no yfinance "
             "prices and cannot be traded, a remaining upward bias of roughly +0.7 to +3 %/yr for an "
             "equal-weight portfolio of the S&P part, which the 'buy all' baselines share; "
             f"decisions at the open every {env.decide_every} trading days; long-only, whole shares, "
             f"$100k, {env.cost_bps:g} bp per trade, cash at the T-bill rate. Both trees were grown on "
             f"{TRAIN_START[:4]}-{TRAIN_END[:4]} only; nothing below was used by the search.", ""] + summary
    lines += ["", "## Stock tree", "", "```", emit(stock_bank, env.stock_names), "```", "",
              "## Exposure tree", "", "```", emit(exp_bank, env.exposure_names), "```", ""]
    for label, (a, b) in PERIODS.items():
        df = pd.DataFrame(report(pairs, panel, F, M, sn, mn, a, b)).T.round(3)
        print(f"\n{label}\n" + df.to_string(), flush=True)
        lines += [f"## {label}", "", "```", df.to_string(), "```", ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")


def report_from_checkpoint():
    """Report the latest trees of the current run's checkpoint, finished or not, without
    training; an agent with no finished stage yet plays its constant default."""
    from btind.runlog import bank_from_json
    with open(os.path.join(RUN_DIR, "state.json")) as fh:
        state = json.load(fh)
    panel = load_panel()
    F, M, sn, mn = load_features()
    env = world(panel, F, M, sn, mn, TRAIN_START, TRAIN_END)
    banks = {a: env.default_bank(a) for a in ("stocks", "exposure")}
    for s in state["stages"]:
        banks[s["agent"]] = bank_from_json(s["bank"])
    write_report(env, panel, F, M, sn, mn, banks["stocks"], banks["exposure"], _summary(state))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--report" in sys.argv:
        report_from_checkpoint()
    else:
        # restarting the same command resumes; --fresh archives the current run and starts over
        seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else None
        args = [a for a in args if a != seed]
        main(int(args[0]) if args else 3, fresh="--fresh" in sys.argv, seed=seed)
