"""Grow the hourly multi-asset behaviour tree with btind (macro_env.py); test it on untouched years.

ONE TREE, one row per instrument -- the "fx" universe (default): 32 G10 currency pairs, 9
emerging-market pairs, gold and silver, 43 in all; or --universe macro12 (indices, oil, copper,
metals, five dollar pairs) -- deciding every hour, or once a day with --daily (the comparison
that says what trading within the day adds), whether to be long, flat or short, at risk-parity
size, after the real bid-ask spread, commission, carry and financing (macro_env.py). It trades
the instruments downloaded so far (macro_data.available); the run's directory names how many.

SPLIT. Training 2012-2019,
validation 2020-2021, test 2022-2026 -- the same untouched years as the stock model. Inside
training, as portfolio_bt.py:
    check years 2014, 2017      read by nothing in the search; a stage whose tree does not beat
                                its start there by CHECK_Z standard errors is reverted
    fold A 2012, 2015, 2018     the held-out years stage 0 accepts on (the rest: search years)
    fold B 2013, 2016, 2019     stage 1's
Each fold holds one year of each of the acceptance test's three periods (2012-14, 2015-17,
2018-19), and each stage is accepted on the years the previous one searched on.

START: all flat (cash, CER 0). Every move must pass btind's paired test on held-out episodes;
nothing is handed to the tree.

BASELINES, fixed rules through the same simulator and costs: the carry trade (long the pairs
whose long side earns the higher rate, short the rest), the carry and 60-day-momentum ranks
across currencies (long the top third, short the bottom third), trend following at 60 and 250
days (long if up, short if down; decided daily) and gold held at 1x -- for macro12 holding the
basket and the US 500 instead of the currency rules.

PICTURES, redrawn as the run goes: learned.png (the current tree 2012-2021 with the baselines),
run/learned_stage<k>.png, and with the report learned_final.png through the test years.

CHECKPOINTS: every finished stage and every round's best tree under <results dir>/run/;
rerunning the same command resumes, --fresh archives the run (never deletes it), --report
writes the report from the checkpoint.

    python macro_bt.py [rounds per stage] [--universe fx|macro12] [--instruments N] [--daily] [--dd W] [--explore]
                       [--fresh | --report]
"""
import json
import os
import sys
import time

import numpy as np

import macro_data as md
from macro_env import FLAT, LONG, SHORT, MacroWorld

ROOT = os.path.dirname(os.path.abspath(__file__))


def _flag_value(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag) + 1
    if i >= len(sys.argv) or sys.argv[i].startswith("--"):
        raise SystemExit(f"{flag} needs a value, e.g.  python macro_bt.py 3 {flag} 0.25")
    return sys.argv[i]


DD_WEIGHT = float(_flag_value("--dd") or 0.25)
DAILY = "--daily" in sys.argv
EXPLORE = "--explore" in sys.argv
DECIDE = "daily" if DAILY else "hourly"
# episodes: 2 weeks of hourly decisions (240 bars); daily decisions get 6 weeks (720 bars, 30
# decisions), so an episode holds enough of them to judge
T_EP = 720 if DAILY else 240
# --instruments N: the first N of the universe (dukascopy_prices order: gold, silver, the majors,
# the crosses, emerging markets), all of which must be downloaded; default: every one downloaded.
# Pinned, so a run keeps its instruments while the download goes on
_pick = int(_flag_value("--instruments") or 0)
if _pick:
    NAMES = list(md.UNIVERSES[md.UNIVERSE])[:_pick]
    _missing = [n for n in NAMES if n not in md.available()]
    if _missing:
        raise SystemExit(f"--instruments {_pick}: not downloaded yet: {', '.join(_missing)}")
else:
    NAMES = md.available()
N_INST = len(NAMES)
RUN_TAG = f"{md.UNIVERSE}{N_INST}_dd{DD_WEIGHT:g}_{DECIDE}" + ("_explore" if EXPLORE else "")
OUT = os.path.join(ROOT, "results", "macro_bt", RUN_TAG)
RUN_DIR = os.path.join(OUT, "run")
PERIODS = {"training 2012-2019": (md.TRAIN_START, md.TRAIN_END), "validation 2020-2021": ("2020-01-01", md.VAL_END),
           "test 2022-2026": ("2022-01-01", "2026-12-31")}
HOLDOUT_FOLDS = {"A": [2012, 2015, 2018], "B": [2013, 2016, 2019]}
STAGE_FOLDS = ("A", "B")
CHECK_YEARS = [2014, 2017]
CHECK_Z = 1.0
GAMMA = 2.76            # the stock world's investor: 100% S&P 500 optimal on 2006-2019 (its report)
LAW_INPUTS = 2          # sparse laws, as the stock world: an intercept and the 2 strongest inputs
# btind's search, as portfolio_bt.CFG (see there) at this world's episode length
CFG = dict(n_ep=400, screen_ep=100, cem_ep=150, T=T_EP, z=1.5, min_gain=0.1, grow_min_gain=0.4,
           grow_pool=90, grow_arms=4, cem_top=6, n_pos=2, max_arity=3, min_n=300,
           cem_iter=8, cem_K=48, cem_sigma=0.35, n_cover=6000, cover_ep=80,
           explore_ep=0, value_laws=False, mem_at=999, beta_at=999, steps_at=999,
           subtree_at=(2,), kern_at=(), val_ep=1000, val_seed=90210)
if EXPLORE:
    CFG.update(eps0=0.5, eps_decay=0.5, grow_pool=120, cem_sigma=0.5, cem_K=64)
ACCEPT_N = CFG["n_ep"]
assert max(CFG["screen_ep"], CFG["cem_ep"], CFG["cover_ep"], 300) < ACCEPT_N <= CFG["val_ep"], \
    "the search budgets must stay below the acceptance budget, or they would read the held-out years"


def small(bank):
    return dict(bank, prior="sparse", prior_k=LAW_INPUTS)


def load():
    X, sim = md.features(md.build(names=NAMES))
    return X, sim


def world(X, sim, start, end, decide=DECIDE, holdout=None, check=None, T=T_EP):
    return MacroWorld(X, sim, start, end, T=T, T_hold=T, decide=decide, holdout_years=holdout,
                      check_years=check, accept_n=ACCEPT_N, dd_weight=DD_WEIGHT, risk_aversion=GAMMA,
                      source=f"dukascopy-h1-{md.UNIVERSE}-" + "-".join(sim["names"]))


HOLD_1X = "GOLD" if md.UNIVERSE == "fx" else "US500"      # the plain buy-and-hold reference


def baseline_banks(env):
    """{name: (tree, decided daily?)}: fixed rules through the same simulator and costs."""
    trend = {"trend 60d (long if up, short if down)": (env.rule("trend_60d", 0.0, True, LONG, SHORT), True),
             "trend 250d": (env.rule("trend_250d", 0.0, True, LONG, SHORT), True)}
    if "is_fx" in env.names:                                         # macro12
        return {"hold the basket (long all but FX)": (env.rule("is_fx", 0.5, True, FLAT, LONG), False), **trend}
    return {"carry trade (long if carry > 0, else short)":
                (env.rules([("is_metal", 0.5, True, FLAT), ("carry", 0.0, True, LONG)], SHORT), True),
            "carry rank (top third long, bottom third short)":
                (env.rules([("is_metal", 0.5, True, FLAT), ("rank_carry", 2 / 3, True, LONG),
                            ("rank_carry", 1 / 3, True, FLAT)], SHORT), True),
            "momentum rank 60d (top third long, bottom short)":
                (env.rules([("rank_trend_60d", 2 / 3, True, LONG), ("rank_trend_60d", 1 / 3, True, FLAT)], SHORT), True),
            **trend}


def picture_baselines(env, env_daily):
    """{name: callable(start, end) -> per-bar excess returns} for macro_plot.draw."""
    b = baseline_banks(env)
    first = "carry trade (long if carry > 0, else short)" if "is_metal" in env.names and "is_fx" not in env.names         else "hold the basket (long all but FX)"
    tree, daily = b[first]
    return {first.split(" (")[0] + (" (decided daily)" if daily else ""):
                lambda s, e: (env_daily if daily else env).path(tree, s, e)["ret"],
            "trend 60d (decided daily)": lambda s, e: env_daily.path(b["trend 60d (long if up, short if down)"][0], s, e)["ret"],
            f"{HOLD_1X.lower()} at 1x": lambda s, e: env.hold_1x(HOLD_1X, s, e)}


def check_gate(env, bank, origin):
    """Keep a stage's tree only if it beats its start on the CHECK years by CHECK_Z cluster-robust SEs."""
    from btind.structure import mean_se
    g_new, g_old = env.check_scores(bank), env.check_scores(origin)
    d = g_new - g_old
    diff = float(d.mean())
    se = float(mean_se(env, d, env._starts_check)) if len(d) > 1 else 0.0
    same = bool(np.all(d == 0))
    keep = same or diff > CHECK_Z * se
    rec = dict(new_G=float(g_new.mean()), origin_G=float(g_old.mean()), diff=diff, se=se,
               episodes=int(len(d)), reverted=not keep, kept_G=float((g_new if keep else g_old).mean()))
    verdict = "no change" if same else ("kept" if keep else "REVERTED to the stage's starting tree")
    print(f"  check years {env.check}: stage tree {rec['new_G']:+.3f}, its start {rec['origin_G']:+.3f}, "
          f"paired {diff:+.3f} +- {se:.3f} on {len(d)} episodes -- {verdict}", flush=True)
    return (bank if keep else origin), rec


# ------------------------------------------------------------------ checkpoints (as portfolio_bt.py)
def _config(env, rounds):
    from btind.runlog import env_signature
    return dict(world=env_signature(env), rounds=rounds, cfg=CFG, check_z=CHECK_Z, holdout_folds=HOLDOUT_FOLDS,
                stage_folds=list(STAGE_FOLDS), names=env.names, assets=env.asset_names)


def _save_state(state):
    tmp = os.path.join(RUN_DIR, "state.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, os.path.join(RUN_DIR, "state.json"))


def _open_run(env, rounds, fresh):
    path = os.path.join(RUN_DIR, "state.json")
    cfg = json.loads(json.dumps(_config(env, rounds)))
    if os.path.exists(path) and not fresh:
        with open(path) as fh:
            state = json.load(fh)
        if state["config"] == cfg:
            print(f"resuming run {state['run_id']}: {len(state['stages'])} of {len(STAGE_FOLDS)} stages done", flush=True)
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


def _stage_snapshot(env, stage):
    from btind import store
    from btind.runlog import bank_from_json
    p = store._path(env)
    if not os.path.exists(p):
        return None, None, 0
    with open(p) as fh:
        entries = [e for e in json.load(fh)["banks"] if e["tag"].startswith(f"macro-s{stage}-") and "@r" in e["tag"]]
    if not entries:
        return None, None, 0
    e = max(entries, key=lambda e: e["G"])
    return bank_from_json(e["bank"]), e["G"], len(entries)


def main(rounds=3, fresh=False):
    import shutil
    import btind.structure
    import macro_plot
    from btind.rlfit import fit
    from btind.runlog import bank_from_json, bank_json
    os.makedirs(OUT, exist_ok=True)
    X, sim = load()
    env = world(X, sim, md.TRAIN_START, md.TRAIN_END, holdout=HOLDOUT_FOLDS["A"], check=CHECK_YEARS)
    env._store_root = RUN_DIR
    state = _open_run(env, rounds, fresh)
    full = world(X, sim, md.TRAIN_START, "2026-12-31")                    # the pictures' continuous paths
    full_daily = world(X, sim, md.TRAIN_START, "2026-12-31", decide="daily")
    pic_base = picture_baselines(full, full_daily)
    learned_png = os.path.join(OUT, "learned.png")
    last_draw = [0.0]

    def picture(bank, force=False):
        if not force and time.time() - last_draw[0] < 120:     # at most every 2 minutes while searching
            return
        last_draw[0] = time.time()
        try:
            full.holdout, full.check = env.holdout, env.check                   # the shading
            macro_plot.draw(full, bank, pic_base, learned_png, md.VAL_END,
                            f"{RUN_TAG}, stage {len(state['stages'])}: one continuous account, training 2012-2019 "
                            f"and validation 2020-2021 (never read by the search)")
        except Exception as e:                                   # a picture must never stop training
            print("  (picture skipped:", e, ")", flush=True)
    env._recorder = picture
    btind.structure.LOG_STEPS = True
    t0 = time.time()
    bank = None
    for s in state["stages"]:
        bank = bank_from_json(s["bank"])
        print(f"  stage {s['stage']}: {s['arms']} arms, held-out G {s['held_out']:+.3f} (from checkpoint)", flush=True)
    for stage in range(len(state["stages"]), len(STAGE_FOLDS)):
        env.set_holdout(HOLDOUT_FOLDS[STAGE_FOLDS[stage]])
        print(f"\n=== stage {stage} ({time.time() - t0:.0f}s): accepts on fold {STAGE_FOLDS[stage]}, held-out years "
              f"{env.holdout}, check years {env.check} ===", flush=True)
        snap, snap_G, done = _stage_snapshot(env, stage)
        origin = bank if bank is not None else small(env.default_bank())
        init = snap if snap is not None else origin
        if snap is not None:
            print(f"  continuing from its best round snapshot (G {snap_G:+.3f}, {done} rounds done)", flush=True)
        attempt = state.setdefault("attempts", {}).get(str(stage), 0) + 1
        state["attempts"][str(stage)] = attempt
        _save_state(state)
        new, log, held = fit(env, env.names, rounds=max(rounds - done, 0), warm=False, cfg=CFG,
                             tag=f"macro-s{stage}-a{attempt}", run_seed=11 + stage, init_bank=init)
        new, check = check_gate(env, new, origin)
        bank = new
        picture(bank, force=True)
        shutil.copyfile(learned_png, os.path.join(RUN_DIR, f"learned_stage{stage}.png")) if os.path.exists(learned_png) else None
        state["stages"].append(dict(stage=stage, bank=bank_json(bank, env.names), held_out=held["G"], ci=held["ci"],
                                    arms=len(bank["clauses"]), check=check))
        _save_state(state)
        print(f"stage {stage} done: {len(bank['clauses'])} arms, held-out G {held['G']:+.3f} +- {held['ci']:.3f}, "
              f"check years {check['kept_G']:+.3f}  [checkpointed]", flush=True)
    env._recorder = None
    write_report(X, sim, bank if bank is not None else env.default_bank(), state)
    state["complete"] = True
    _save_state(state)


def gap_p(x, stat, n=1000, block=120, seed=0):
    """Circular block bootstrap (1-week blocks) of stat(x): one-sided p of stat <= 0."""
    rng = np.random.default_rng(seed)
    T = len(x)
    s0 = stat(x)
    b = np.empty(n)
    for k in range(n):
        idx = (rng.integers(0, T, int(np.ceil(T / block)))[:, None] + np.arange(block)).ravel()[:T] % T
        b[k] = stat(x[idx])
    return float(np.mean(b - s0 >= s0))


def write_report(X, sim, bank, state):
    import pandas as pd
    import macro_plot
    from btind.memory import emit
    from btind.runlog import bank_json
    env = world(X, sim, md.TRAIN_START, "2026-12-31")
    env_d = world(X, sim, md.TRAIN_START, "2026-12-31", decide="daily")
    with open(os.path.join(OUT, "tree.json"), "w") as fh:
        json.dump(bank_json(bank, env.names), fh, indent=1)
    A, g = env.bars_per_year, env.risk_aversion
    cer = lambda x: 100 * A * (x.mean() - 0.5 * g * x.var(ddof=1))
    rows_md = []
    for label, (a, b) in PERIODS.items():
        rows = {}
        runs = {"the tree": (env, bank, True), "the tree, no spread or commission": (env, bank, False)}
        for name, (bb, daily) in baseline_banks(env).items():
            runs[name + (" (daily)" if daily else "")] = ((env_d if daily else env), bb, True)
        for name, (w, bb, costs) in runs.items():
            p = w.path(bb, a, b, costs=costs)
            m = macro_plot.metrics(p["ret"], A, g, p["pos"], p["turnover"])
            rows[name] = dict(m, p=gap_p(p["ret"], cer))
        x = env.hold_1x(HOLD_1X, a, b)
        rows[f"{HOLD_1X.lower()} at 1x (financed)"] = dict(macro_plot.metrics(x, A, g), p=gap_p(x, cer))
        df = pd.DataFrame(rows).T[["cer", "p", "ret", "vol", "sharpe", "dd", "turn_yr", "trades_wk", "in_mkt"]]
        df.columns = ["CER %/yr", "p (CER<=0)", "return %/yr", "vol %", "Sharpe", "max DD %", "turnover x/yr",
                      "trades/wk", "in market %"]
        txt = df.round(3).to_string()
        print(f"\n{label}\n{txt}", flush=True)
        rows_md += [f"## {label}", "", "```", txt, "```", ""]
    stages = ["| stage | fold | arms | held-out G | check years: kept | vs start | |", "|---|---|---|---|---|---|---|"]
    for s in state["stages"]:
        c = s["check"]
        stages.append(f"| {s['stage']} | {STAGE_FOLDS[s['stage']]} | {s['arms']} | {s['held_out']:+.3f} +- {s['ci']:.3f} | "
                      f"{c['kept_G']:+.3f} | {c['diff']:+.3f} +- {c['se']:.3f} | {'reverted' if c['reverted'] else 'kept'} |")
    lines = ["# Hourly multi-asset behaviour tree", "",
             f"Decisions {DECIDE}, one tree over {len(env.asset_names)} instruments ({', '.join(env.asset_names)}): "
             f"long / flat / short at risk parity ({env.risk_per_asset:.1%} volatility per position), real Dukascopy "
             f"bid-ask spreads, {env.commission * 1e4:g} bp commission per side, carry and a {env.fin_markup:.0%} "
             f"financing margin. Score: CER over cash, % per year, gamma {g:.2f}, minus {DD_WEIGHT:g} x max drawdown. "
             "Trained on 2012-2019 only; 2020-2026 was never read by the search. p: one-sided, 1-week block bootstrap.",
             "", f"Run {state['run_id']}, {state['config']['rounds']} rounds per stage.", ""] + stages + [
             "", "## The tree", "", "```", emit(bank, env.names), "```", ""] + rows_md
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    try:
        env.check = ",".join(map(str, CHECK_YEARS))
        macro_plot.draw(env, bank, picture_baselines(env, env_d), os.path.join(OUT, "learned_final.png"), "2026-12-31",
                        f"{RUN_TAG}, final tree: one continuous account, training 2012-2019, validation 2020-2021, "
                        "test 2022-2026", final=True)
    except Exception as e:
        print("  (final picture skipped:", e, ")", flush=True)
    print("wrote", OUT, flush=True)


def report_from_checkpoint():
    from btind.runlog import bank_from_json
    with open(os.path.join(RUN_DIR, "state.json")) as fh:
        state = json.load(fh)
    X, sim = load()
    env = world(X, sim, md.TRAIN_START, md.TRAIN_END)
    bank = bank_from_json(state["stages"][-1]["bank"]) if state["stages"] else env.default_bank()
    write_report(X, sim, bank, state)


class _Tee:
    def __init__(self, stream, files):
        self.stream, self.files = stream, files

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        for fh in self.files:
            fh.write(s)
            fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        for fh in self.files:
            fh.flush()

    def isatty(self):
        return self.stream.isatty()


if __name__ == "__main__":
    args, rest = [], sys.argv[1:]
    while rest:
        a = rest.pop(0)
        if a in ("--dd", "--universe", "--instruments"):
            rest = rest[1:]
        elif not a.startswith("--"):
            args.append(a)
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "train_log.txt"), "a", encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, [log]), _Tee(sys.__stderr__, [log])
    print(f"\n===== [{time.strftime('%Y-%m-%d %H:%M:%S')}] python {' '.join(sys.argv)}", flush=True)
    if "--report" in sys.argv:
        report_from_checkpoint()
    else:
        main(int(args[0]) if args else 3, fresh="--fresh" in sys.argv)
    print(f"===== [{time.strftime('%Y-%m-%d %H:%M:%S')}] finished", flush=True)
