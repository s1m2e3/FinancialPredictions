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
                                  in its top quarter of 2006-2019, 20% in its top tenth
                                  (25% before the 10%-step exposure levels)
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

SEED. Stage 0 starts from the EBTDA track-record rule by default (ebtda_rule: the best
simple rule on both 2020-2021 and 2022-2026), so the search can only improve on it;
--seed tree.json starts from another stock tree (btind's bank_json), --seed none from
buy-everything. A resumed stage 0 ignores it (its snapshot is newer).

CHECK YEARS (CHECK_YEARS, 2012 and 2015) are read by nothing in the search; a stage whose
tree does not beat its starting tree there is reverted (check_gate). The report's stage
table shows both.

VOLATILITY TARGET (--voltarget): only the exposure tree is searched, in two stages (folds A,
B), starting from the volatility-target baseline; the stock tree stays buy-all (or the
--seed tree, held fixed). See VOL_TARGET below.

AT MOST N STOCKS (--max-stocks N): the portfolio holds at most N names, the N largest of
those the stock tree buys or holds; buy-everything is "the N largest" and stage 0 starts
from it. See MAX_STOCKS below.

PICTURES, all redrawn while the run goes:
    training_budget.png   every adopted tree on fixed training windows vs the S&P 500
                          (training_progress.py)
    learned.png           the current pair on ONE continuous path 2006-2021: growth, the
                          invested share it chose, drawdown, the trees as text and per-period
                          metrics (learned_plot.py); run/learned_stage<k>.png per stage
    learned_final.png     the final pair through the test years, with the report

Run from the repository root:  python portfolio_bt.py [rounds per stage] [--dd W] [--every N] [--seed tree.json|none] [--voltarget] [--max-stocks N] [--objective cer|alpha] [--weighted] [--stocks-only] [--core-cap F [--core-n N]] [--explore] [--fresh | --report]
"""
import json
import os
import sys
import time

import numpy as np

from portfolio_env import (BUY, EXIT, EXPOSURE_ACTIONS, EXPOSURE_LEVELS, STOCK_ACTIONS, level_index,
                           PortfolioWorld, calibrate_risk_aversion, constant_bank)
from stock_features import load as _load_features
from stocks_data import END, TRAIN_END, TRAIN_START, VAL_END, load_panel

ROOT = os.path.dirname(os.path.abspath(__file__))
# TWO OBJECTIVES. Each run scores episodes as  gap - DD_WEIGHT * drawdown gap
# (portfolio_env.py); runs at several weights (--dd) each get their own directory, and
# elite.py keeps the trees no other tree beats on both the gap and the drawdown.
def _flag_value(flag):
    """The value after `flag` on the command line, or None; a flag with no value is an error."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag) + 1
    if i >= len(sys.argv) or sys.argv[i].startswith("--"):
        raise SystemExit(f"{flag} needs a value, e.g.  python portfolio_bt.py 3 {flag} 0.25")
    return sys.argv[i]


DD_WEIGHT = float(_flag_value("--dd") or 0.0)
# SIZE WEIGHTS (the default; --inverse-vol for the old ones): buys share the invested money
# by company size, so "buy everything" is about the index and a tree only has to tilt it
# (company_size); inverse-volatility runs get their own directories (dd<W>_invvol)
SIZE_WEIGHTS = "--inverse-vol" not in sys.argv
# DECISIONS every DECIDE_EVERY trading days: 21 (monthly) by default -- factor premia are slow,
# and weekly re-deciding was noise and cost when first tried; --every 5 re-decides weekly. It
# is part of the world's signature, and a non-default interval gets its own directories
# (dd<W>_every<N>), so weekly and monthly runs never mix
DECIDE_EVERY = int(_flag_value("--every") or 21)
# VOLATILITY TARGET (--voltarget): risk control instead of stock picking. The signal audit
# (signal_audit.py) found no input that predicts which of these stocks will do better, while
# volatility is predictable. So the stock tree stays buy-all (or the --seed tree, held fixed)
# and only the exposure tree is searched, starting FROM the volatility-target baseline (50%
# invested when the S&P's forecast 21-day volatility is in its top quarter of 2006-2019, 20%
# in its top tenth): two exposure stages, accepted on fold A then fold B. Run it with a
# drawdown weight (--dd), so a smoother path is worth something. Its own directories
# (dd<W>_voltarget)
VOL_TARGET = "--voltarget" in sys.argv
# AT MOST MAX_STOCKS NAMES (--max-stocks N; 0 = no limit): of the stocks the stock tree buys
# or holds, the N largest by company size are kept and the rest sold (PortfolioWorld's
# max_names). Buying everything becomes "the N largest" -- an S&P N -- and the stock tree
# changes the portfolio by what it leaves out: excluding a stock lets the next largest in.
# Stage 0 starts from buy-everything unless --seed is given. Its own directories (..._top<N>)
MAX_STOCKS = int(_flag_value("--max-stocks") or 0)
# THE SCORE (--objective; portfolio_env.py): "cer" (the default) the risk-adjusted real gap to
# the S&P 500, "alpha" the return left after the portfolio's own beta to the S&P -- stock
# selection judged apart from the market exposure it carries. Its own directories (..._alpha)
OBJECTIVE = _flag_value("--objective") or "cer"
# CONDITIONAL WEIGHTS in the paired test (--weighted; PortfolioWorld.episode_weights): the stock
# tree's held-out episodes weighted by the inverse of their expected noise, from the S&P's
# forecast volatility at each start; the t critical value uses the effective number of
# clusters (btind.structure.weighted_mean_se). Its own directories (..._w)
WEIGHTED = "--weighted" in sys.argv
# STOCK SELECTION ONLY (--stocks-only): the exposure tree is never searched and stays at 100%
# invested; the stock tree gets both stages (folds A, B) and reads NO market features (stock
# features and its portfolio state only), so it cannot time the market through its arms. The first alpha run's only kept rule
# was an exposure rule (cash when the S&P ran >6.3% above its 200-day average) that passed the
# held-out and check years and lost 17 points a year on 2020-2021. Its own directories (..._stocks)
STOCKS_ONLY = "--stocks-only" in sys.argv
# THE CORE CAP (--core-cap F [--core-n N]; PortfolioWorld.core_names / core_frac): the N (10)
# largest companies of the day together hold at most F of the invested money, so "hold the
# giants" -- the basin every search so far ended in -- is no longer available and the rest must
# be found among the other stocks. Its own directories (..._core<N>x<F%>)
CORE_FRAC = float(_flag_value("--core-cap") or 1.0)
CORE_N = int(_flag_value("--core-n") or 10) if CORE_FRAC < 1.0 else 0
# EPSILON-GREEDY SEARCH (--explore; btind rlfit eps0 / eps_decay): round r explores with
# eps = 0.5 * 0.5**r -- that share of proposal columns drawn uniformly, and that chance of an
# exploratory kick (a random new arm, or a structural kick) from the incumbent -- plus wider
# candidate pools and law tuning; acceptance is unchanged. Its own directories (..._explore)
EXPLORE = "--explore" in sys.argv
VT_NAME = f"the {MAX_STOCKS} largest + volatility target" if MAX_STOCKS else "buy all + volatility target"
RUN_TAG = (f"dd{DD_WEIGHT:g}" + ("" if SIZE_WEIGHTS else "_invvol")
           + ("" if DECIDE_EVERY == 21 else f"_every{DECIDE_EVERY}") + ("_voltarget" if VOL_TARGET else "")
           + (f"_top{MAX_STOCKS}" if MAX_STOCKS else "") + ("" if OBJECTIVE == "cer" else f"_{OBJECTIVE}")
           + ("_w" if WEIGHTED else "") + ("_stocks" if STOCKS_ONLY else "")
           + (f"_core{CORE_N}x{round(100 * CORE_FRAC)}" if CORE_N else "") + ("_explore" if EXPLORE else ""))
OUT = os.path.join(ROOT, "results", "portfolio_bt", RUN_TAG)
PERIODS = {"validation 2020-2021": ("2020-01-01", VAL_END), "test 2022-2026": ("2022-01-01", END)}


# The models' MEAN and DIRECTION forecasts (mu_h, pup_h, for every stock and the S&P 500)
# are not inputs. The identification found no directional skill, and their levels drift
# with every walk-forward refit, so to the search they were calendars: arms on spx_mu_10d
# fired in whole stretches of particular years and in none of 2008 or 2022, and arms on
# pup_1d helped in 2010-2019 and hurt in 2006-2009 and on validation. The risk forecasts
# (sig_h, q05_h) -- where the models do have skill -- stay. The mean and direction come
# back in DRIFT-FREE forms (stock_features.py): each stock's rank against the same day's
# universe (rank_mu_*, rank_pup_*, horizons to 6 months) and the S&P model's deviation from
# its own past year of forecasts (spxz_*); the check years judge whether they carry anything.
# The OPEN GAPS (gap, rank_gap, spx_gap) are out too: they use the day's opening price and
# the simulator trades at that same open, but a market-on-open order is placed before the
# open is known -- a small look-ahead (early trees split on rank_gap).
DROPPED = ("mu_", "pup_", "spx_mu_", "spx_pup_", "gap", "rank_gap", "spx_gap")


def load_features():
    import insiders
    F, M, sn, mn = _load_features(fundamentals=True)
    with np.load(insiders.CACHE) as zi, np.load(os.path.join(ROOT, "data", "stock_features.npz")) as zs:
        if not (np.array_equal(zi["tickers"], zs["tickers"]) and np.array_equal(zi["dates"], zs["dates"])):
            raise RuntimeError("insiders.npz is from another panel: rerun python insiders.py")
        F, sn = np.concatenate([F, zi["F"]], axis=2), sn + list(zi["names"])
    ks = [i for i, n in enumerate(sn) if not n.startswith(DROPPED)]
    km = [i for i, n in enumerate(mn) if not n.startswith(DROPPED)]
    return F[:, :, ks], M[:, km], [sn[i] for i in ks], [mn[i] for i in km]

# With the compiled rollout a 400-episode score costs ~0.1-0.2 s, so btind's budgets can be
# near its defaults. Stages that need btind's own traces or kernels (exploration, memory,
# termination, steps, subtrees) stay off. z = 2: a move must beat the incumbent by two
# standard errors on paired episodes, and lose significantly in no market regime. Gains are
# in percentage points of real certainty-equivalent return per year.
# SCREEN CHEAP, CONFIRM FULL: candidates are ranked, and laws tuned by CEM, on fewer
# episodes (screen_ep, cem_ep) from the SEARCH years; every move is ACCEPTED on n_ep
# episodes from the HELD-OUT years (portfolio_env.py, "search and held-out years").
# z = 1.5 (was 2): with cluster-robust standard errors (btind.structure.mean_se) z = 2 let
# almost nothing through -- 1 accepted move in 165 over a whole run; the check gate still
# filters every stage's result on years no search reads.
# GUARDS of up to 3 literals (max_arity, was 2; btind's own default) from a pool of 90
# candidates per arm (was 60): btind draws 60% single-literal guards and the rest of 1-3
# literals, each literal's threshold from the quantile alphabet of the tree's own rows and
# the rows before its worst decisions; accepted thresholds are then polished. A longer
# guard is one more way to fit noise, which the acceptance test and the check gate price.
CFG = dict(n_ep=400, screen_ep=100, cem_ep=150, T=189, z=1.5, min_gain=0.1, grow_min_gain=0.4,
           grow_pool=90, grow_arms=4, cem_top=6, n_pos=2, max_arity=3, min_n=300,
           cem_iter=8, cem_K=48, cem_sigma=0.35, n_cover=6000, cover_ep=80,
           explore_ep=0, value_laws=False, mem_at=999, beta_at=999, steps_at=999,
           subtree_at=(), kern_at=(), val_ep=1000, val_seed=90210)
if EXPLORE:                   # --explore: epsilon-greedy proposals and kicks, wider pools and law tuning
    CFG.update(eps0=0.5, eps_decay=0.5, grow_pool=120, cem_sigma=0.5, cem_K=64)


# PER TREE, on top of CFG. A new arm must add 0.4 %/yr (grow_min_gain, was 0.2): with this
# little signal every extra arm is likelier noise than knowledge.
#   exposure  TERMINATION conditions from round 1 (btind's search_beta): an arm can become
#             sticky -- "once defensive, stay defensive until <condition>" -- the hysteresis
#             a monthly flip between 100% and 50% lacks
#   both      NESTED SUBTREES in the last round (btind's search_subtrees): children grown
#             inside an existing arm, on that arm's own rows
# Each is one more kind of move, and every move still has to pass the paired test.
AGENT_CFG = {"stocks": dict(subtree_at=(2,)), "exposure": dict(beta_at=1, subtree_at=(2,))}

MAX_WEIGHT = 1 / 3            # no stock above a third of the invested money: at least 3 names
OUTSIDE_MAX_WEIGHT = 0.10     # an outside-pool (non-S&P) name: at most a tenth -- explore, don't bet the book
# DECIDE_EVERY: set with the command-line flags above (--every)
T_HOLD = 126                  # held-out episodes: 6 months (search episodes: CFG["T"], 9 months)
# SMALL LAWS: every law keeps its intercept and its LAW_INPUTS strongest inputs (btind's
# sparse prior, applied to every candidate before it is scored). A dense law over ~45
# inputs was what memorised: tuned to +4 on search years, -9 to -13 on held-out ones.
LAW_INPUTS = 2


def small(bank):
    return dict(bank, prior="sparse", prior_k=LAW_INPUTS)
ACCEPT_N = CFG["n_ep"]        # requests of this many episodes or more come from the held-out years
assert max(CFG["screen_ep"], CFG["cem_ep"], CFG["cover_ep"], 300) < ACCEPT_N <= CFG["val_ep"], \
    "the search budgets must stay below the acceptance budget, or they would read the held-out years"


def holdout_years():
    """Five of the fourteen training years, BALANCED: each side holds a fall and a rally.
    Held out: 2008 (the crash), 2010, 2013, 2016, 2019. Searched: 2006, 2007, 2009 (the
    rebound), 2011 and 2018 (the two sharp falls), 2012, 2014, 2015, 2017. A defensive
    rule is learned from 2011, 2018 and the 2009 rebound and judged on whether it gets
    through 2008 without giving up the rallies of 2013 and 2019. The first split held the
    crash in the search years and mostly rallies out, so any insurance was judged only on
    what it costs, never on what it pays for. Each era keeps held-out years (2008;
    2010, 2013; 2016, 2019), which the period check needs.

    Superseded by HOLDOUT_FOLDS: this returns fold A, the held-out years of stages 0-1 and
    of every world built outside the stage loop (reports, pair_trees, hypotheses)."""
    return HOLDOUT_FOLDS["A"]


# ROTATING HELD-OUT YEARS. One fixed set of held-out years judged every move of every stage,
# and trees fitted it (a stage's held-out score rose to +14 to +22 %/yr while the check years
# fell 6 to 11). Now the twelve non-check training years form two halves of consecutive
# two-year blocks, each with a fall, a rally and a year in each of the acceptance test's three
# periods (2006-09, 2010-14, 2015-19):
#     A  2008-09 (crash, rebound)   2013-14 (rally, calm)   2018-19 (Q4 fall, rally)
#     B  2006-07 (calm, the top)    2010-11 (flash crash, euro crisis)   2016-17 (choppy, calm)
# Stages accept on A, A, B, B: each tree's SECOND stage is accepted on the years its first
# stage searched on, so what one stage fitted, the next one is judged against. Blocks of two
# years because a 9-month search episode must fit inside consecutive search years: in
# isolated single years every episode would have started in January-March.
HOLDOUT_FOLDS = {"A": [2008, 2009, 2013, 2014, 2018, 2019], "B": [2006, 2007, 2010, 2011, 2016, 2017]}
STAGE_FOLDS = ("A", "B") if (VOL_TARGET or STOCKS_ONLY) else ("A", "A", "B", "B")   # one tree: folds A, B


# CHECK YEARS: taken out of the search years and read by NOTHING in btind's search --
# screening, law tuning and the paired acceptance test all run on the search and held-out
# years. The held-out years judge hundreds of moves per stage, so they get fitted too:
# the first sweep's "elite" trees scored +5 to +10 %/yr there and -2.6 to -10.9 on
# 2020-2021. After a stage, its tree and the tree it started from are compared on the
# check years (paired, the same episodes); a stage that does not beat its start there is
# REVERTED to its start. 2012 (a rally with a mid-year scare) and 2015 (choppy, a
# correction): normal markets, where the failed trees gave up the most. The sharp falls
# of 2011 and 2018 stay in the search years, to be learned from.
CHECK_YEARS = [2012, 2015]


_SIZE = {}


def company_size(panel):
    """(T, N) each company's size as known at the open. Market value (fundamentals.py:
    shares from the latest filing x yesterday's real close) where EDGAR has the shares --
    domestic filers, from 2010; elsewhere the trailing 63-day mean dollar volume, lagged a
    day, turned into market-value units by that day's median market value / dollar volume
    over the stocks that have both. Before 2010 no stock has a market value and dollar
    volume alone is used: weights only compare the stocks of one day. Nothing here uses a
    later share count."""
    key = id(panel)
    if key not in _SIZE:
        import pandas as pd
        import fundamentals as fu
        with np.load(fu.CACHE) as z:
            if list(z["tickers"]) != list(panel.tickers):
                raise RuntimeError("fundamentals.npz is from another panel: rerun python fundamentals.py")
            mcap = z["mcap"].astype(np.float64)
        dv = pd.DataFrame(panel.close * panel.volume).rolling(63, min_periods=38).mean().shift(1).to_numpy()
        both = np.isfinite(mcap) & np.isfinite(dv) & (dv > 0)
        ratio = np.full(len(mcap), np.nan)
        for t in np.where(both.any(1))[0]:
            ratio[t] = np.median(mcap[t, both[t]] / dv[t, both[t]])
        _SIZE[key] = np.where(np.isfinite(mcap), mcap, dv * np.where(np.isfinite(ratio), ratio, 1.0)[:, None])
    return _SIZE[key]


def world(panel, F, M, sn, mn, start, end, T=252, agent="stocks", gamma=None, holdout=None, check=None):
    gamma = calibrate_risk_aversion(panel, TRAIN_START, TRAIN_END) if gamma is None else gamma
    by_size = dict(size=company_size(panel), weighting="company size") if SIZE_WEIGHTS else {}
    # the source is part of the world's signature in btind's store: trees grown on another
    # universe or feature set are never reused
    return PortfolioWorld(panel, F, M, sn, mn, start, end, T=T, agent=agent, risk_aversion=gamma,
                          fractional=True, max_weight=MAX_WEIGHT, outside_max_weight=OUTSIDE_MAX_WEIGHT,
                          holdout_years=holdout,
                          T_hold=T_HOLD, accept_n=ACCEPT_N, decide_every=DECIDE_EVERY, dd_weight=DD_WEIGHT,
                          check_years=check, max_names=MAX_STOCKS, objective=OBJECTIVE, weighted_accept=WEIGHTED,
                          core_names=CORE_N, core_frac=CORE_FRAC, stock_market=not STOCKS_ONLY,
                          source="yfinance-sp500top100+ndx40+edgar-longh-regime-young", **by_size)


def rule(names, n_act, column, threshold, above, action, default):
    """One guarded arm: `action` where column > threshold (above) or <= it, else `default`."""
    bank = constant_bank(names, n_act, default, STOCK_ACTIONS if n_act == 3 else EXPOSURE_ACTIONS)
    law = np.zeros_like(bank["default"])
    law[-1, action] = 1.0
    # a list, not a tuple: btind edits literals in place (threshold polishing) when a rule
    # is the starting tree of a search
    bank["clauses"].append([[names.index(column), float(threshold), not above]])
    bank["laws"].append(law)
    return bank


def vol_target_rule(env):
    """The volatility target as an exposure tree: invest 20% when the S&P 500's forecast
    21-day volatility is above its 90th percentile over the search years' episode starts,
    50% above its 75th, 100% otherwise (the baseline, and --voltarget's starting tree)."""
    en = env.exposure_names
    sig = env._M[env._starts, en.index("spx_sig_21d")]
    q75, q90 = np.quantile(sig, [0.75, 0.90])
    vol_target = rule(en, len(EXPOSURE_LEVELS), "spx_sig_21d", q90, True, level_index(0.2),
                      level_index(1.0))                                            # top tenth: 20%
    law50 = np.zeros_like(vol_target["default"])
    law50[-1, level_index(0.5)] = 1.0
    vol_target["clauses"].append([[en.index("spx_sig_21d"), float(q75), False]])   # top quarter: 50%
    vol_target["laws"].append(law50)
    return vol_target


def baselines(env):
    """{name: (stock bank, exposure bank)} run through the same simulator."""
    sn = env.stock_names
    full = env.default_bank("exposure")
    vol_target = vol_target_rule(env)
    everything = (f"the {MAX_STOCKS} largest (by size)" if MAX_STOCKS
                  else "buy all (by size)" if SIZE_WEIGHTS else "buy all (inverse vol)")
    return {everything: (env.default_bank("stocks"), full),
            "buy all S&P members": (rule(sn, 3, "in_sp500", 0.5, True, BUY, EXIT), full),
            "low volatility (30%)": (rule(sn, 3, "rank_sig_21d", 0.30, False, BUY, EXIT), full),
            "momentum (top 20%)": (rule(sn, 3, "rank_mom_12_1", 0.80, True, BUY, EXIT), full),
            # the survivorship check: on a list of today's members the most volatile stocks
            # are the ones that became mega-winners; on a point-in-time universe they lag
            "high volatility (top 30%)": (rule(sn, 3, "rank_sig_21d", 0.70, True, BUY, EXIT), full),
            VT_NAME: (env.default_bank("stocks"), vol_target),
            "EBTDA track record (top 30%)": (rule(sn, 3, "rank_ebtda_roa_3y", 0.70, True, BUY, EXIT), full),
            "revenue growth (top 30%)": (rule(sn, 3, "rank_rev_growth", 0.70, True, BUY, EXIT), full)}


def train_stage(env, agent, partner, rounds, seed, cfg=None, init_bank=None, tag=None):
    """Search one agent's tree with `partner` fixed. `init_bank` is the tree this agent left
    at its previous stage: the search continues from it instead of from an empty tree, and
    btind only accepts moves that beat it (against the new partner)."""
    import btind.structure
    from btind.rlfit import fit
    btind.structure.LOG_STEPS = True     # every paired test in the log: its held-out delta +- se
    env.agent, env.partner = agent, partner
    bank, log, held = fit(env, env.names, rounds=rounds, warm=False, cfg=dict(CFG, **(cfg or {})),
                          tag=tag or f"portfolio-{agent}", run_seed=seed, init_bank=init_bank)
    return bank, held


def ebtda_rule(names):
    """The EBTDA track-record baseline as a tree: buy the 30% with the best 3-year EBTDA /
    assets in their sector, exit the rest (stage 0's default starting point)."""
    return rule(names, 3, "rank_ebtda_roa_3y", 0.70, True, BUY, EXIT)


def _seeds(env, seed):
    """{agent: the tree its first stage starts from, or None (its constant default)}. The
    stock tree: `seed` ("ebtda", a bank_json file, or "none"); under --voltarget it is never
    searched and plays this tree throughout. The exposure tree: the volatility target under
    --voltarget."""
    from btind.runlog import bank_from_json
    seeds = {"stocks": None, "exposure": small(vol_target_rule(env)) if VOL_TARGET else None}
    if seed == "ebtda":
        # the best simple rule on both 2020-2021 and 2022-2026 when first tried -- buy the 30%
        # with the best 3-year EBTDA / assets in their sector, exit the rest
        seeds["stocks"] = small(ebtda_rule(env.stock_names))
    elif seed not in (None, "none"):
        with open(seed) as fh:
            seeds["stocks"] = small(bank_from_json(json.load(fh)))
    return seeds


def picture_baselines(env):
    """The baselines learned_plot.py draws next to the trees."""
    everything = f"the {MAX_STOCKS} largest (100% invested)" if MAX_STOCKS else "buy all (100% invested)"
    return {everything: (env.default_bank("stocks"), env.default_bank("exposure")),
            VT_NAME: (env.default_bank("stocks"), vol_target_rule(env))}


CHECK_Z = 1.0     # a stage's check-year gain must beat CHECK_Z cluster-robust standard errors


def check_gate(env, agent, partner, bank, origin):
    """Keep a stage's tree only if it beats the tree the stage started from on the CHECK
    years by more than CHECK_Z standard errors (paired: identical episodes, the same
    partner); otherwise revert to that start. The SE is cluster-robust by calendar
    half-year (btind.structure.mean_se): the 248 check episodes come from two years, so a
    naive std / sqrt(248) claimed a precision they do not have -- dd0.5's stock tree passed
    at +3.4 +- 0.3 and then lost to its start on 2020-2026. Returns (the tree kept, the record)."""
    from btind.structure import mean_se
    env.agent, env.partner = agent, partner
    g_new = env.check_scores(*env.banks(bank))
    g_old = env.check_scores(*env.banks(origin))
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


def alpha_beta(daily, bench, rf):
    """{beta, alpha %/yr} of daily returns against the S&P 500, on excess returns over cash."""
    xp, xb = np.asarray(daily) - rf, np.asarray(bench) - rf
    beta = float(np.cov(xp, xb)[0, 1] / np.var(xb, ddof=1))
    return {"beta": beta, "alpha %/yr": float(100 * 252 * (xp.mean() - beta * xb.mean()))}


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
                                                **{"turnover / yr": 0.0, "CER gap": 0.0, "p (CER gap > 0)": np.nan,
                                                   "beta": 1.0, "alpha %/yr": 0.0})
        gap, p = gap_bootstrap(real(out["daily"][0]), real(out["bench"][0]), lambda r: cer(r, g))
        rows[name] = dict(metrics(out["daily"][0], out["rf"][0], infl, g),
                          **{"turnover / yr": out["turnover"][0] * 252 / env.T, "CER gap": gap, "p (CER gap > 0)": p},
                          **alpha_beta(out["daily"][0], out["bench"][0], out["rf"][0]))
    return rows


STAGES = (("exposure", "exposure") if VOL_TARGET else ("stocks", "stocks") if STOCKS_ONLY
          else ("stocks", "exposure", "stocks", "exposure"))
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

def _config(env, rounds, seed=None):
    from btind.runlog import env_signature
    sig = {k: v for k, v in env_signature(env).items() if k != "agent"}
    # acceptance: btind's paired test with cluster-robust SEs (env.cluster_ids), the check
    # gate at CHECK_Z of them -- a run under other rules never resumes into this one
    cfg = dict(world=sig, rounds=rounds, cfg=CFG, agent_cfg=AGENT_CFG, check_z=CHECK_Z,
               exposure_levels=list(EXPOSURE_LEVELS), holdout_folds=HOLDOUT_FOLDS,
               stage_folds=list(STAGE_FOLDS), dropped=list(DROPPED),
               accept_se="cluster-robust, calendar half-years; t(G-1) critical value",
               min_clusters=int(env.min_clusters), stock_names=env.stock_names,
               exposure_names=env.exposure_names, n_stocks=int(env._N))
    if STOCKS_ONLY:
        cfg.update(stages=list(STAGES), exposure_tree="invest 100% (fixed)")
    if VOL_TARGET:        # the stages, the exposure tree's start and the fixed stock tree
        cfg.update(stages=list(STAGES), exposure_seed="volatility target (vol_target_rule)",
                   stock_tree=str(seed or "none"))
    return cfg


def _save_state(state):
    tmp = os.path.join(RUN_DIR, "state.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, os.path.join(RUN_DIR, "state.json"))     # atomic: never half-written


def _open_run(env, rounds, fresh, seed=None):
    path = os.path.join(RUN_DIR, "state.json")
    cfg = json.loads(json.dumps(_config(env, rounds, seed)))      # through JSON, as it is stored
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
    env = world(panel, F, M, sn, mn, TRAIN_START, TRAIN_END, T=CFG["T"], holdout=holdout_years(),
                check=CHECK_YEARS)
    env._store_root = RUN_DIR
    state = _open_run(env, rounds, fresh, seed)
    # every accepted tree is replayed on fixed training episodes (training_budget.png)
    # every adopted tree is also replayed on the validation years (2020-2021), for the plot
    # only: nothing in the search reads them
    val_env = world(panel, F, M, sn, mn, "2020-01-01", VAL_END, gamma=env.risk_aversion)
    rec = env._recorder = Recorder(env, RUN_DIR, os.path.join(OUT, "training_budget.png"), val_env=val_env,
                                   archive=os.path.join(RUN_DIR, "adopted.jsonl"))
    # WHAT THE TREES DO, drawn (learned_plot.py): one continuous path 2006 -> 2021 of every
    # adopted pair next to the S&P 500, buy-all and the volatility target, to learned.png;
    # a copy per finished stage in the run directory
    import shutil
    import learned_plot
    learned_png = os.path.join(OUT, "learned.png")
    pic_base = picture_baselines(env)

    def picture(sb, eb):
        learned_plot.draw(env, sb, eb, learned_png, VAL_END, pic_base,
                          f"{RUN_TAG}, stage {rec.stage} ({env.agent} tree searched): one continuous $100k path, "
                          f"training 2006-2019 and validation 2020-2021 (never read by the search)")
    rec.on_record = picture
    t0 = time.time()
    banks = {"stocks": None, "exposure": None}
    seeds = _seeds(env, seed)
    for s in state["stages"]:
        banks[s["agent"]] = bank_from_json(s["bank"])
        print(f"  stage {s['stage']} {s['agent']}: {s['arms']} arms, held-out G {s['held_out']:+.3f} (from checkpoint)",
              flush=True)
    for stage in range(len(state["stages"]), len(STAGES)):
        agent = STAGES[stage]
        other = "exposure" if agent == "stocks" else "stocks"
        # the partner: the other tree as its last stage left it, else its seed (the fixed stock
        # tree under --voltarget), else None -- its constant default
        partner = banks[other] if banks[other] is not None else seeds[other]
        # this stage's held-out half (HOLDOUT_FOLDS): set BEFORE the snapshot lookup, since the
        # held-out years are part of the store's world signature
        env.set_holdout(HOLDOUT_FOLDS[STAGE_FOLDS[stage]])
        print(f"  stage {stage} accepts on fold {STAGE_FOLDS[stage]}: held-out years {env.holdout}", flush=True)
        snap, snap_G, done = _stage_snapshot(env, agent, stage)
        init = snap if snap is not None else banks[agent]        # None the first time
        if init is None and seeds[agent] is not None:            # the agent's first stage
            init = seeds[agent]
            label = "the volatility target" if agent == "exposure" else seed
            print(f"  starting from the seed tree {label} ({len(init['clauses'])} arms)", flush=True)
        if init is None:
            # never btind's cold start: it tunes a default law on n_ep episodes, and those are
            # the held-out years' (portfolio_env.py) -- fitting on what acceptance is judged
            # on. A stage with nothing to continue starts from its constant default instead
            # (buy everything / invest 100%), and every change to it must pass the test.
            init = small(env.default_bank(agent))
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
        # the tree this stage is judged against on the check years: what the agent had before
        # it (the seed or the constant default the first time), never a mid-stage snapshot
        origin = banks[agent] if banks[agent] is not None else (
            seeds[agent] if seeds[agent] is not None else small(env.default_bank(agent)))
        if snap is not None and stage not in rec.rows["stage"]:
            env.on_adopt(origin)           # resumed with no history: where the stage began
        env.on_adopt(init if init is not None else origin)                           # where it starts
        bank, held = train_stage(env, agent, partner, left, seed=11 + stage, init_bank=init,
                                 tag=f"portfolio-{agent}-s{stage}-a{attempt}", cfg=AGENT_CFG[agent])
        env.agent, env.partner = agent, partner
        bank, check = check_gate(env, agent, partner, bank, origin)
        env.on_adopt(bank)                             # the tree the stage keeps (if new)
        rec.draw()
        try:                                           # the kept pair, and a copy for this stage
            picture(*env.banks(bank))
            shutil.copyfile(learned_png, os.path.join(RUN_DIR, f"learned_stage{stage}.png"))
        except Exception as e:                         # a picture must never stop training
            print("  (learned-path picture skipped:", e, ")", flush=True)
        banks[agent] = bank
        state["stages"].append(dict(stage=stage, agent=agent, bank=bank_json(bank, env.names),
                                    held_out=held["G"], ci=held["ci"], arms=len(bank["clauses"]),
                                    check=check))
        _save_state(state)
        print(f"stage {stage} done: {agent} tree with {len(bank['clauses'])} arms, held-out "
              f"(training period) {'alpha' if OBJECTIVE == 'alpha' else 'CER gap'} {held['G']:+.3f} +- {held['ci']:.3f} %/yr, check years "
              f"{check['kept_G']:+.3f}  [checkpointed]", flush=True)
    import gc
    env.set_holdout(holdout_years())            # the report's baselines on the fold every report uses
    env._recorder = rec.val_env = None          # the validation world is not needed for the report
    del val_env
    gc.collect()
    final = {a: banks[a] if banks[a] is not None else (seeds[a] or env.default_bank(a)) for a in banks}
    write_report(env, panel, F, M, sn, mn, final["stocks"], final["exposure"], _summary(state))
    state["complete"] = True
    _save_state(state)


def _summary(state):
    rows = ["| stage | tree | arms | held-out gap (%/yr) | check years: kept tree | vs its start | |",
            "|---|---|---|---|---|---|---|"]
    for s in state["stages"]:
        c = s.get("check") or {}
        rows.append(f"| {s['stage']} | {s['agent']} | {s['arms']} | {s['held_out']:+.3f} +- {s['ci']:.3f} | "
                    + (f"{c['kept_G']:+.3f} | {c['diff']:+.3f} +- {c['se']:.3f} | "
                       f"{'reverted' if c['reverted'] else 'kept'} |" if c else "- | - | |"))
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
             ("The trees were TRAINED on ALPHA against the S&P 500 (the return left after the portfolio's own "
              "beta, OLS within each episode, %/yr)" + (", with the paired test weighted by each episode's "
              "forecast volatility" if WEIGHTED else "") + "; the tables below also report beta and alpha over "
              "each whole period. " if OBJECTIVE == "alpha" else "")
             + f"Score: certainty-equivalent REAL return (CPI-deflated), risk aversion gamma = "
             f"{env.risk_aversion:.2f} calibrated on {TRAIN_START[:4]}-{TRAIN_END[:4]} (100% S&P optimal); gaps are portfolio "
             "minus S&P 500 in % per year, p from a 21-day block bootstrap.", "",
             "Universe: each month's 100 largest S&P 500 members of that day plus the 20 largest "
             "Nasdaq-100 members outside the S&P 500, by trailing dollar volume (point-in-time "
             "membership, stocks_data.py). Stocks delisted before today mostly have no yfinance "
             "prices and cannot be traded, a remaining upward bias of roughly +0.7 to +3 %/yr for an "
             "equal-weight portfolio of the S&P part, which the 'buy all' baselines share; "
             + (f"AT MOST {MAX_STOCKS} STOCKS at a time, every baseline and tree alike: the {MAX_STOCKS} largest "
                f"by company size of those it buys or holds; " if MAX_STOCKS else "")
             + f"decisions at the open every {env.decide_every} trading days; long-only, "
             f"{'fractional' if env.fractional else 'whole'} shares, "
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
    try:                         # the whole history in one picture, test years included
        import learned_plot
        learned_plot.draw(env, stock_bank, exp_bank, os.path.join(OUT, "learned_final.png"), END,
                          picture_baselines(env), f"{RUN_TAG}, final trees: one continuous $100k path, training "
                          f"2006-2019, validation 2020-2021, test 2022-{END[:4]}", final=True)
        print("wrote", os.path.join(OUT, "learned_final.png"), flush=True)
    except Exception as e:
        print("  (final picture skipped:", e, ")", flush=True)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")


def report_from_checkpoint():
    """Report the latest trees of the current run's checkpoint, finished or not, without
    training; an agent with no finished stage yet plays its seed (--voltarget: the fixed stock
    tree, the volatility target), else its constant default."""
    from btind.runlog import bank_from_json
    with open(os.path.join(RUN_DIR, "state.json")) as fh:
        state = json.load(fh)
    panel = load_panel()
    F, M, sn, mn = load_features()
    env = world(panel, F, M, sn, mn, TRAIN_START, TRAIN_END, T=CFG["T"], holdout=holdout_years(),
                check=CHECK_YEARS)
    seeds = _seeds(env, state["config"].get("stock_tree")) if VOL_TARGET else {}
    banks = {a: seeds.get(a) or env.default_bank(a) for a in ("stocks", "exposure")}
    for s in state["stages"]:
        banks[s["agent"]] = bank_from_json(s["bank"])
    write_report(env, panel, F, M, sn, mn, banks["stocks"], banks["exposure"], _summary(state))


class _Tee:
    """A stream that writes to the terminal and to log files at once, flushing every write,
    so the log is complete up to the moment a run is stopped."""

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


def log_to_files():
    """Everything printed (errors and tracebacks too) also goes to <run dir>/train_log.txt
    and to results/portfolio_bt/train_log.txt, appended, each run headed by its command."""
    os.makedirs(OUT, exist_ok=True)
    paths = [os.path.join(OUT, "train_log.txt"), os.path.join(ROOT, "results", "portfolio_bt", "train_log.txt")]
    files = [open(p, "a", encoding="utf-8") for p in paths]
    sys.stdout = _Tee(sys.__stdout__, files)
    sys.stderr = _Tee(sys.__stderr__, files)
    print(f"\n===== [{time.strftime('%Y-%m-%d %H:%M:%S')}] python {' '.join(sys.argv)}", flush=True)
    return paths


if __name__ == "__main__":
    args, rest = [], sys.argv[1:]
    while rest:                             # positional arguments: not flags, not flag values
        a = rest.pop(0)
        if a in ("--seed", "--dd", "--every", "--max-stocks", "--objective", "--core-cap", "--core-n"):
            rest = rest[1:]
        elif not a.startswith("--"):
            args.append(a)
    seed = _flag_value("--seed")
    # --seed none: start stage 0 from buy-everything; under --voltarget the stock tree is
    # buy-everything, and under --max-stocks stage 0 starts from it, unless a --seed is given
    seed = seed or ("none" if (VOL_TARGET or MAX_STOCKS) else "ebtda")
    logs = log_to_files()
    print("logging to", " and ".join(logs), flush=True)
    if "--report" in sys.argv:
        report_from_checkpoint()
    else:
        # restarting the same command resumes; --fresh archives the current run and starts over
        main(int(args[0]) if args else 3, fresh="--fresh" in sys.argv, seed=seed)
    print(f"===== [{time.strftime('%Y-%m-%d %H:%M:%S')}] finished: python {' '.join(sys.argv)}", flush=True)
