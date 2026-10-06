"""Shared machinery for the research scripts -- placebo.py, walkforward.py, stability.py,
longpath.py and portfolio_score.py: portfolio_bt.main's stage loop without its report and
training plot, the evaluation of a tree pair over a calendar period, and the paired test.

Nothing here changes portfolio_bt.py; each script builds its own worlds (possibly on
transformed features, possibly of another world class) and hands them in.

STAGES. `run_stages` runs the same four stages as portfolio_bt.main -- stocks, exposure,
stocks, exposure; each accepted on its fold of held-out years, each judged by the check-year
gate against the tree it started from -- and checkpoints every finished stage to
<run_dir>/state.json. A stopped script resumes at the first unfinished stage; unlike
portfolio_bt.main it restarts that stage from its start rather than from a round snapshot
(these scripts run many short searches, where the simpler rule costs little).
"""
import json
import os
import time

import numpy as np

import portfolio_bt as pb


def run_stages(env, run_dir, rounds=3, seed_bank=None, folds=None, stage_folds=None,
               stages=pb.STAGES, agent_cfg=None, run_seed=11, on_stage=None):
    """Search both trees on `env` (portfolio_bt.main's loop); returns (banks, state).
    `folds` {name: held-out years} and `stage_folds` (one name per stage) default to
    portfolio_bt's; `run_seed` shifts the proposal seeds (stage k uses run_seed + k);
    `on_stage(state)` is called after every finished stage (a progress picture, say) and
    may never stop the search."""
    from btind.runlog import bank_from_json, bank_json
    folds = folds or pb.HOLDOUT_FOLDS
    stage_folds = stage_folds or pb.STAGE_FOLDS
    agent_cfg = agent_cfg or pb.AGENT_CFG
    os.makedirs(run_dir, exist_ok=True)
    env._store_root = run_dir
    path = os.path.join(run_dir, "state.json")
    state = dict(stages=[], complete=False)
    if os.path.exists(path):
        with open(path) as fh:
            state = json.load(fh)
    banks = {"stocks": None, "exposure": None}
    for s in state["stages"]:
        banks[s["agent"]] = bank_from_json(s["bank"])
    for stage in range(len(state["stages"]), len(stages)):
        agent = stages[stage]
        partner = banks["exposure" if agent == "stocks" else "stocks"]
        env.set_holdout(folds[stage_folds[stage]])
        origin = banks[agent] if banks[agent] is not None else (
            seed_bank if (stage == 0 and seed_bank is not None) else pb.small(env.default_bank(agent)))
        print(f"\n=== stage {stage}: {agent} tree, held-out {env.holdout} ===", flush=True)
        bank, held = pb.train_stage(env, agent, partner, rounds, seed=run_seed + stage, init_bank=origin,
                                    tag=f"{agent}-s{stage}", cfg=agent_cfg[agent])
        bank, check = pb.check_gate(env, agent, partner, bank, origin)
        banks[agent] = bank
        state["stages"].append(dict(stage=stage, agent=agent, bank=bank_json(bank, env.names), fold=stage_folds[stage],
                                    held_out=held["G"], ci=held["ci"], arms=len(bank["clauses"]), check=check))
        _save(path, state)
        if on_stage is not None:
            try:
                on_stage(state)
            except Exception as exc:                     # a picture must never stop training
                print("  (progress picture skipped:", exc, ")", flush=True)
    state["complete"] = True
    _save(path, state)
    return {a: (b if b is not None else env.default_bank(a)) for a, b in banks.items()}, state


def _save(path, state):
    with open(path + ".tmp", "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(path + ".tmp", path)


def period_paths(panel, F, M, sn, mn, pairs, start, end, gamma, world_fn=None):
    """{name: dict(real, daily, bench_real, dates)} for every (stock bank, exposure bank) of
    `pairs`, run as ONE continuous portfolio from `start` to `end` (as portfolio_bt.report
    does), plus "S&P 500" for the benchmark. `world_fn` builds the world (portfolio_bt.world
    by default)."""
    import pandas as pd
    world_fn = world_fn or pb.world
    a = np.searchsorted(panel.dates, np.datetime64(start))
    b = np.searchsorted(panel.dates, np.datetime64(end), "right")
    env = world_fn(panel, F, M, sn, mn, start, end, T=int(b - a - 2), gamma=gamma)
    s = env.sample_starts(1, np.random.default_rng(0))
    out = {}
    for name, (sb, eb) in pairs.items():
        o = env.rollout(sb, eb, s, env.T, daily=True)
        infl = o["infl"][0]
        out[name] = dict(daily=o["daily"][0], real=(1 + o["daily"][0]) / (1 + infl) - 1,
                         dates=pd.DatetimeIndex(panel.dates[s[0]:s[0] + env.T]))
        if "S&P 500" not in out:
            out["S&P 500"] = dict(daily=o["bench"][0], real=(1 + o["bench"][0]) / (1 + infl) - 1,
                                  dates=out[name]["dates"])
    del env
    return out


def block_len(n, longest=126):
    """Bootstrap block length for a path of n days: up to `longest` (regime-scale persistence)
    but never fewer than ~10 blocks -- a 2-year period cut into 126-day blocks has four, and
    resampling four blocks badly understates the spread (a validation p of 0.014 against 0.10
    with 21-day blocks)."""
    return int(min(longest, max(21, n // 10)))


def paired(real_a, real_b, gamma, block=None):
    """(CER difference %/yr, one-sided p that it is > 0), paired block bootstrap (block_len)."""
    return pb.gap_bootstrap(real_a, real_b, lambda r: pb.cer(r, gamma), block=block or block_len(len(real_a)))


def describe_tree(bank, top=2):
    """Lines of plain text for a tree: every arm's condition, then what its law does -- for a
    stock tree the inputs that push toward buy / hold rather than exit, for an exposure tree
    the level it prefers by default and the inputs that move it, for a scoring tree the
    inputs that raise or lower the score. The default law comes last ("otherwise")."""
    from btind.memory import mem_names
    names = list(bank["names"])
    z = mem_names(names, None) + ["bias"]
    acts = list(bank.get("actions") or [])
    lines = []
    for k, th in enumerate(list(bank["laws"]) + [bank["default"]]):
        th = np.asarray(th, float)
        if k < len(bank["clauses"]):
            cond = " AND ".join(f"{names[j]} {'<=' if neg else '>'} {t:.3g}" for j, t, neg in bank["clauses"][k])
            lines.append(f"arm {k}: IF {cond}")
        else:
            lines.append("otherwise:")
        coef = lambda col: [(z[i], col[i]) for i in np.argsort(-np.abs(col[:-1]))[:top] if abs(col[i]) > 1e-9]
        fmt = lambda pairs: ", ".join(f"{n} {v:+.2f}" for n, v in pairs) or "nothing"
        if th.shape[1] == 1:                                      # scoring head
            lines.append(f"    score {th[-1, 0]:.2f} at the average; " + fmt(coef(th[:, 0])))
        elif th.shape[1] == 3 and acts[:1] == ["exit"]:           # exit / hold / buy
            for a in (2, 1):
                d = th[:, a] - th[:, 0]
                lines.append(f"    {acts[a]} vs exit: bias {d[-1]:+.2f}; " + fmt(coef(d)))
        else:                                                     # exposure levels
            best = int(np.argmax(th[-1]))
            spread = th - th.mean(axis=1, keepdims=True)
            strength = np.abs(spread[:-1]).max(axis=1)
            idx = [i for i in np.argsort(-strength)[:top] if strength[i] > 1e-9]
            moves = ", ".join(f"{z[i]} (favours {acts[int(np.argmax(th[i]))]} when high)" for i in idx) or "nothing"
            lines.append(f"    by default {acts[best] if acts else best}; moved by: {moves}")
    return lines


def max_drawdown(daily):
    w = np.cumprod(1 + np.asarray(daily))
    return float((w / np.maximum.accumulate(w) - 1).min())


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
