"""PortfolioWorld: a btind world that trades a point-in-time stock universe at the open against the S&P 500.

UNIVERSE. A stock is tradable on day t only if it is in that day's point-in-time universe
(`panel.universe`, stocks_data.py), has a price and has its GARCH features. An untradable
stock is sold, the tree is not evaluated on it and its latch is cleared: if it comes back
it starts with no memory. `n_held_frac` and `buy_frac` are shares of the stocks tradable
that day, not of the panel's columns (the panel holds every stock that is EVER tradable).

TWO TREES, trained in alternation (btind's vehicle / signal pattern), each the fixed
partner of the other while one is searched:

  stocks    one shared tree, one row per stock: its walk-forward features, the market
            features and the portfolio's state. Per stock:  0 exit  1 hold  2 buy
  exposure  one row per portfolio: the S&P 500's full-model forecasts, VIX, drawdown,
            current exposure and the share of stocks the stock tree wants to buy.
            Chooses the invested fraction: 0%, 10%, ..., 100% (EXPOSURE_LEVELS; the rest in T-bills)

ALLOCATION (fixed rules, not learned). With exposure f and portfolio value V the stocks
may hold f * V. "hold" positions keep their shares; if they alone exceed f * V they are
trimmed proportionally. What is left of f * V goes to the "buy" stocks by inverse
forecast volatility (1 / sig_21d) -- or, with `size` given, by company size, so that
buying everything is about the index and a tree only has to tilt it. No position may exceed `max_weight` of f * V (1/3: at
least three stocks to be fully invested); a buy that would is capped and its excess goes to
the other buys, and what no stock can take stays in cash -- nothing is bought on the tree's
behalf, so a tree that picks fewer than 1 / max_weight stocks pays for the idle cash. With
`max_names` (portfolio_bt.py --max-stocks) at most that many stocks are bought or held: the
largest of those the tree wants, the rest sold. Long-only, no leverage, `cost_bps` on every traded
dollar, cash at the 13-week T-bill rate. Shares are fractional (`fractional`, as Fidelity,
Schwab, Robinhood or IBKR fill them) or whole: with $100k over ~115 stocks, whole shares
leave 4-7% of the money idle and cannot buy a stock priced above its slice at all.

TIMING. Decisions at the open every `decide_every` trading days, positions marked at
every open in between: daily open-to-open returns. The benchmark is the S&P 500 WITH
DIVIDENDS REINVESTED (SPY's dividend-adjusted open, open to open), priced like the stocks.

SCORE of an episode, in REAL terms (every daily return deflated by CPI-U inflation, so
idle cash earns the T-bill rate minus inflation and is no longer free):

    objective "cer" (default)  certainty-equivalent real return, % per year,
                               CER = 252 mean(r_real) - (gamma/2) 252 var(r_real),
                               portfolio minus S&P 500. It is the guaranteed real return
                               an investor with risk aversion gamma would take instead of
                               the strategy -- its risk-adjusted discounted value per year.
                               gamma is calibrated on the training period as the risk
                               aversion at which holding 100% S&P 500 was optimal
                               (mean real excess return / variance), so the S&P is the
                               fair benchmark and leaving money idle has a real cost.
    objective "sharpe"         annualised Sharpe ratio gap on excess returns (blind to
                               inflation and to how much is invested)
    objective "alpha"          ALPHA against the S&P 500, % per year: the portfolio's daily
                               return over cash minus its own beta (OLS within the episode)
                               times the S&P's -- what the stocks earned beyond the market
                               exposure they carried, whatever that exposure was

DRAWDOWN. Every episode also measures its maximum drawdown -- the deepest fall of the
account from its running high, from the $100k it starts with -- for the portfolio and the
S&P 500, and the score is  G = gap - dd_weight * dd_gap,  dd_gap = 100 (maxDD_portfolio -
maxDD_S&P) in points (negative: shallower falls than the market). The two parts come back
separately (`cer_gap`, `dd_gap`), so trees trained at any dd_weight can be compared on
both: runs at several weights trace the trade-off, and the non-dominated trees are kept
(elite.py).

`score_bank` returns one number per episode (btind's `structure.score` hook on the
`portfolio-env` branch).

SEARCH AND HELD-OUT YEARS (`holdout_years`). One history can be memorised: a condition
like "VIX below 12.5" picks out 2006 and 2017, and a rule fitted there wins on those years
and nowhere else. So the training period is split BY WHOLE YEARS into search years and
held-out years, and an episode never crosses from one to the other:
    search      episodes of `T` days (6 months) inside search years: what btind screens
                candidates and tunes laws on
    held-out    episodes of `T_hold` days (3 months) inside held-out years: what btind
                ACCEPTS moves on, and scores rounds and stages on
btind asks for episodes by count, and its budgets are distinct: screening (screen_ep),
law tuning (cem_ep) and coverage (cover_ep) ask for fewer than `accept_n`, the paired
test and its reference (n_ep), the round check (val_ep) and the held-out score ask for
`accept_n` or more. So `sample_starts` serves the held-out pool to requests of `accept_n`
or more and the search pool to the rest -- the candidate and the incumbent of every
paired test land on the same held-out episodes -- and `score_bank` runs each pool at its
own length. portfolio_bt.py asserts the budgets keep that order.

TWO IMPLEMENTATIONS OF ONE SIMULATION, on purpose (btind keeps the same discipline):
`run` in numpy with btind's `MemBank` is the reference and produces traces; `_rollout`
is the numba kernel the search uses, built on btind's own compiled arbitration
(`btind.tick._tick`). `checks/verify_portfolio_kernel.py` holds them to agreement.
"""
import os
import sys

import numpy as np
from numba import njit, prange

ROOT = os.path.dirname(os.path.abspath(__file__))
# btind (the `portfolio-env` branch of s1m2e3/btind) next to this repository, or BTIND_PATH
sys.path.insert(0, os.environ.get("BTIND_PATH", os.path.join(os.path.dirname(ROOT), "btind-portfolio")))
from btind.tick import _tick  # noqa: E402  (btind's compiled arbitration, shared with its kernels)
PORTFOLIO_NAMES = ["held", "weight", "days_held", "cash_frac", "n_held_frac", "drawdown"]
# exposure_days: years since the invested level last changed (0 at an episode's first
# decision), so a tree can tell a fresh cut from one it has held for months
EXPOSURE_STATE = ["cash_frac", "n_held_frac", "drawdown", "exposure", "buy_frac", "exposure_days"]
STOCK_ACTIONS = ["exit", "hold", "buy"]
# the invested fraction the exposure tree chooses from: 0% (all in T-bills) to 100% in steps
# of 10% (was 25-100% in steps of 25%: no way to go fully to cash, and coarse). Discrete on
# purpose -- a continuous fraction would move every month and trade on every move
EXPOSURE_LEVELS = tuple(round(0.1 * k, 1) for k in range(11))
EXPOSURE_ACTIONS = [f"invest {round(100 * x)}%" for x in EXPOSURE_LEVELS]


def level_index(x):
    """The action index of exposure level x (a fraction, e.g. 0.5)."""
    return EXPOSURE_LEVELS.index(round(x, 1))
EXIT, HOLD, BUY = 0, 1, 2
# the episode score before the drawdown term: 1 the CER gap (default), 0 the Sharpe gap, 2 ALPHA --
# the portfolio's return over cash left after its own beta to the S&P (daily OLS within the
# episode), % per year: what the stocks earned beyond the market exposure they carried
OBJECTIVES = {"sharpe": 0, "cer": 1, "alpha": 2}
# smallest position, dollars: brokers' minimum fractional order is $1-5, and without a floor
# the budget left after trimming (zero up to rounding, +-1e-10) buys dust that then counts
# as a holding
MIN_POSITION = 1.0


def calibrate_risk_aversion(panel, start, end):
    """gamma at which 100% in the S&P 500 was mean-variance optimal over [start, end]:
    mean real excess return over cash divided by the variance of the real S&P return."""
    dates = panel.dates
    a = np.searchsorted(dates, np.datetime64(start))
    b = np.searchsorted(dates, np.datetime64(end), "right") - 1
    r = panel.bench_open[a + 1:b + 1] / panel.bench_open[a:b] - 1.0
    real = (1 + r) / (1 + panel.infl[a:b]) - 1.0
    cash = (1 + panel.rf[a:b]) / (1 + panel.infl[a:b]) - 1.0
    return float(np.mean(real - cash) / np.var(real))


def constant_bank(names, n_act, action, actions):
    """A tree with no arms whose default always prefers `action`."""
    from btind.memory import mem_names
    th = np.zeros((len(mem_names(names, None)) + 1, n_act))
    th[-1, action] = 1.0
    return dict(clauses=[], laws=[], default=th, names=list(names), laws_on_z=True,
                head="argmax", n_act=n_act, actions=list(actions))


class PortfolioWorld:
    head = "argmax"
    gamma = 1.0
    # btind.structure.accept: a gain is accepted only on held-out years holding at least this
    # many calendar half-years (cluster_ids) -- 4 years; one 2-year crash is not evidence
    min_clusters = 8
    # AT MOST max_names STOCKS held at a time (0: no limit; set per world, see __init__)
    max_names = 0
    # CONDITIONAL WEIGHTS in btind's paired test (episode_weights; off unless set per world)
    weighted_accept = False
    # A CAP ON THE CORE (off at 0): the core_names largest companies of the day together hold
    # at most core_frac of the invested money (set per world, see __init__)
    core_names = 0
    core_frac = 1.0
    # the stock tree's rows carry the market features too (off: stock features and portfolio state only)
    stock_market = True

    def __init__(self, panel, F, M, stock_names, market_names, start, end, T=252,
                 decide_every=5, budget=100_000.0, cost_bps=5.0, source="yfinance-sp100",
                 agent="stocks", objective="cer", risk_aversion=None, fractional=False,
                 max_weight=1.0, holdout_years=None, T_hold=63, accept_n=400, dd_weight=0.0,
                 size=None, weighting="inverse volatility", check_years=None, outside_max_weight=None,
                 max_names=0, weighted_accept=False, core_names=0, core_frac=1.0, stock_market=True,
                 store_root=os.path.join(ROOT, "runs_bt")):
        # scalar attributes are the world's signature in btind's store
        self.split_start, self.split_end = str(start), str(end)
        self.T, self.decide_every = int(T), int(decide_every)
        self.budget, self.cost_bps, self.source = float(budget), float(cost_bps), str(source)
        self.agent = str(agent)
        self.objective = str(objective)
        self.benchmark = "SPY total return"
        self.fractional = bool(fractional)
        self.dd_weight = float(dd_weight)
        self.weighting = str(weighting) if size is not None else "inverse volatility"
        # scalars, so the split is part of the world's signature in btind's store
        self.holdout = ",".join(str(y) for y in sorted(holdout_years)) if holdout_years else ""
        self.check = ",".join(str(y) for y in sorted(check_years)) if check_years else ""
        self.T_hold, self.accept_n = int(T_hold), int(accept_n)
        self.max_weight = float(max_weight)
        # the OUTSIDE pool (Nasdaq-100 names not in the S&P 500, `panel.sp500` False) may have
        # its own, smaller cap: a tree can explore younger growth names without one of them
        # becoming a third of the portfolio. Scalar, so it is part of the world's signature
        self.outside_max_weight = float(outside_max_weight if outside_max_weight is not None else max_weight)
        ratio = min(1.0, self.outside_max_weight / self.max_weight)
        self._cap_scale = np.ascontiguousarray(np.where(panel.sp500, 1.0, ratio), dtype=np.float64)
        # AT MOST max_names STOCKS: of the stocks the stock tree buys or holds, the max_names
        # largest by company size (inverse volatility in a world without sizes) are kept and
        # the rest sold, so buy-everything becomes "the max_names largest" and a tree changes
        # the portfolio by what it leaves out. The exposure tree's buy_frac still counts every
        # stock the stock tree wanted. Set on the instance only when used, so the store
        # signature of worlds without a limit is what it was
        if max_names:
            self.max_names = int(max_names)
        if weighted_accept:              # on the instance only when used: the signature of the rest is unchanged
            self.weighted_accept = True
        # THE CORE CAP (portfolio_bt.py --core-cap): the core_names largest companies of the
        # day (by company size, among the tradable) together get at most core_frac of the
        # invested money -- held core positions are trimmed to it in proportion, and the buy
        # budget is split between core and non-core buys by their size weights with the core's
        # share capped at the room left; each group is then water-filled under the per-stock
        # caps, and what the non-core buys cannot take stays in cash. It needs company sizes.
        if core_names and core_frac < 1.0:
            if size is None:
                raise ValueError("a core cap needs company sizes (size-weighted buys)")
            self.core_names, self.core_frac = int(core_names), float(core_frac)
        self.groups = "period"
        # the acceptance test's groups: the calendar block of the episode's start (2006-09,
        # 2010-14, 2015-19 on the training split), so a move that clearly loses over one
        # stretch of history is rejected. Not the S&P regime: a rule may lose in strong
        # rallies -- the price of backing off before a fall -- if it pays over each period
        # calibrated on THIS world's period unless given: pass the training value to the
        # validation and test worlds so nothing is calibrated on the future
        self.risk_aversion = float(risk_aversion if risk_aversion is not None
                                   else calibrate_risk_aversion(panel, start, end))
        self._store_root = store_root
        self._recorder, self._last_adopted = None, None
        # STOCK ROWS WITHOUT THE MARKET (stock_market=False, portfolio_bt.py --stocks-only): a stock
        # tree that reads the S&P's features can act on every stock at once -- market timing
        # through the stock tree. A placebo with the market block shifted by years accepted such
        # an arm (spx_sig_1d > 0.624, laws on spx_ret_12m) at +3.2 +- 1.7 on its held-out years;
        # without the market block a stock rule can only win by choosing WHICH stocks
        if not stock_market:
            self.stock_market = False
        self.stock_names = (list(stock_names) + (list(market_names) if stock_market else [])
                            + PORTFOLIO_NAMES)
        self.exposure_names = list(market_names) + EXPOSURE_STATE
        self._sig = stock_names.index("sig_21d")

        import pandas as pd
        dates = panel.dates
        a = np.searchsorted(dates, np.datetime64(start))
        b = np.searchsorted(dates, np.datetime64(end), "right")
        T_all, N = F.shape[:2]
        self._N = N
        self._dates = np.asarray(pd.DatetimeIndex(dates).values.astype("datetime64[D]"))
        self._open = np.ascontiguousarray(pd.DataFrame(panel.open).ffill().to_numpy())
        self._avail = np.ascontiguousarray(~np.isnan(panel.open) & ~np.isnan(F[:, :, self._sig])
                                           & panel.universe)
        X = (np.concatenate([F, np.broadcast_to(M[:, None, :], (T_all, N, M.shape[1]))], axis=2)
             if stock_market else F)
        # float32: the (days x stocks x features) block is the largest array of a world; every
        # value the trees and the allocation read is widened to float64 where it is used
        self._X = np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float32)
        self._M = np.ascontiguousarray(np.nan_to_num(M, nan=0.0), dtype=np.float64)
        self._rf = np.ascontiguousarray(panel.rf, dtype=np.float64)
        self._infl = np.ascontiguousarray(panel.infl, dtype=np.float64)
        self._spx = panel.bench_open     # SPY with dividends reinvested (stocks_data.MARKET)
        self._spx_ret = np.ascontiguousarray(np.r_[self._spx[1:] / self._spx[:-1] - 1.0, 0.0])
        # 0 search, 1 held-out, 2 check, -1 outside the period; episodes stay inside one pool.
        # CHECK years are read by nothing in btind's search -- not screening, not tuning, not
        # the paired acceptance test -- only by `check_scores`, after a stage has finished
        self._span, self._years = (a, b, T_all), pd.DatetimeIndex(dates).year
        self._check_years = list(check_years or [])
        self._build_pools(holdout_years)
        self._levels = np.array(EXPOSURE_LEVELS)
        # buy weights: company size where given (a positive number per stock and day), else
        # the kernel falls back to 1 / sig_21d; an empty array means "inverse volatility"
        self._size = (np.ascontiguousarray(np.nan_to_num(size, nan=0.0), dtype=np.float64)
                      if size is not None else np.zeros((0, 0)))
        self._blocks = np.searchsorted(dates, [np.datetime64("2010-01-01"), np.datetime64("2015-01-01")])
        self.partner = None       # the other agent's bank; None = its constant default

    # ---------------------------------------------------------------- which agent is searched
    @property
    def names(self):
        return self.stock_names if self.agent == "stocks" else self.exposure_names

    @property
    def n_act(self):
        return 3 if self.agent == "stocks" else len(EXPOSURE_LEVELS)

    @property
    def actions(self):
        return STOCK_ACTIONS if self.agent == "stocks" else EXPOSURE_ACTIONS

    @property
    def store_root(self):
        return self._store_root

    def default_bank(self, agent):
        if agent == "stocks":
            return constant_bank(self.stock_names, 3, BUY, STOCK_ACTIONS)
        return constant_bank(self.exposure_names, len(EXPOSURE_LEVELS), len(EXPOSURE_LEVELS) - 1,
                             EXPOSURE_ACTIONS)

    def on_adopt(self, bank):
        """btind (portfolio-env branch, `structure.adopted`) calls this whenever the search
        HOLDS a new tree -- a grown arm, a refitted law, a simplification, a round's end;
        a recorder (training_progress.py) may listen. btind reports after every phase, so a
        tree identical to the last one recorded is skipped."""
        if self._recorder is None:
            return
        import hashlib
        import json
        from btind.runlog import bank_json
        key = hashlib.sha1(json.dumps([self.agent, bank_json(bank)], sort_keys=True,
                                      default=str).encode()).hexdigest()
        if key != self._last_adopted:
            self._last_adopted = key
            self._recorder.record(*self.banks(bank))

    def banks(self, bank):
        """(stock bank, exposure bank) with `bank` in the searched agent's seat."""
        other = self.partner or self.default_bank("exposure" if self.agent == "stocks" else "stocks")
        return (bank, other) if self.agent == "stocks" else (other, bank)

    def _build_pools(self, holdout_years):
        a, b, T_all = self._span
        pool = np.full(T_all, -1)
        pool[a:b] = 0
        if holdout_years:
            pool[(pool == 0) & np.isin(self._years, list(holdout_years))] = 1
        if self._check_years:
            pool[(pool == 0) & np.isin(self._years, self._check_years)] = 2
        self._pool = pool
        self._starts = self._pool_starts(0, self.T)
        self._starts_hold = self._pool_starts(1, self.T_hold) if holdout_years else self._starts
        self._starts_check = (self._pool_starts(2, self.T_hold) if self._check_years
                              else np.zeros(0, np.int64))

    def set_holdout(self, holdout_years):
        """Make `holdout_years` the held-out (acceptance) years and the rest of the period,
        check years aside, the search years -- between stages, so each tree is accepted on a
        different half of history (portfolio_bt.HOLDOUT_FOLDS). The held-out years are part
        of the world's signature, so btind's store files differ per fold, and btind's cache
        of episode starts (structure.starts, on the world) is cleared: it would otherwise
        hand out the previous fold's episodes."""
        self.holdout = ",".join(str(y) for y in sorted(holdout_years)) if holdout_years else ""
        self._build_pools(holdout_years)
        if hasattr(self, "_starts_cache"):
            del self._starts_cache

    # ---------------------------------------------------------------- btind world API
    def _pool_starts(self, k, T):
        """Days an episode of T days can start on and stay inside pool k (it reads the open
        of day s + T, so T + 1 days)."""
        inside = np.r_[0, np.cumsum(self._pool == k)]
        s = np.arange(len(self._pool) - T - 1)
        return s[inside[s + T + 2] - inside[s] == T + 2]

    def sample_starts(self, n, rng):
        src = self._starts_hold if (self.holdout and n >= self.accept_n) else self._starts
        return rng.choice(src, size=n, replace=True).astype(np.int64)

    def episode_length(self, starts, T=None):
        """The length a pool's episodes run for: T_hold for held-out starts, T otherwise."""
        if self.holdout and len(starts) and self._pool[int(starts[0])] in (1, 2):
            return self.T_hold
        return self.T if self.holdout or T is None else int(T)

    def cluster_ids(self, starts):
        """The calendar half-year each episode starts in (btind.structure.mean_se): episodes
        that start in the same half-year share most of their history, so btind's paired test
        and its held-out interval treat them as one piece of evidence, not as independent."""
        d = self._dates[np.asarray(starts, np.int64)]
        year = d.astype("datetime64[Y]").astype(int)
        month = d.astype("datetime64[M]").astype(int) % 12
        return year * 2 + (month >= 6)

    WEIGHT_CLIP = (0.25, 4.0)             # no episode counts for more than 4x, or less than a quarter, of another
    # MEASURED on the 10-largest world (12 random one-arm stock trees vs buy-all, 400 held-out
    # episodes): log|paired difference| rises with log(forecast vol at the start) with a median
    # slope of +0.51 (positive for 10 of 12, R2 ~0.04) -- noise ~ vol^0.5, so the GLS weight is
    # 1 / variance ~ vol^-1, not vol^-2
    WEIGHT_POWER = 1.0

    def episode_weights(self, starts):
        """(E,) weights of btind's paired test (structure.accept): the inverse of each
        episode's expected noise, from the S&P 500's forecast 21-day volatility on its START
        day -- known before the episode, so nothing later leaks in -- as (median / sigma)^WEIGHT_POWER,
        clipped to WEIGHT_CLIP and normalised to mean 1. Only for the stock tree and only
        with `weighted_accept`: down-weighting volatile episodes is right for a selection
        rule whose noise grows with volatility (measured, checks in the log), and wrong for
        an exposure rule, whose whole value may lie in the volatile episodes. None: unweighted."""
        if not self.weighted_accept or self.agent != "stocks":
            return None
        sig = self._M[np.asarray(starts, np.int64), self.exposure_names.index("spx_sig_21d")]
        sig = np.where(sig > 0, sig, np.nanmedian(sig[sig > 0]))
        w = np.clip((np.median(sig) / sig) ** self.WEIGHT_POWER, *self.WEIGHT_CLIP)
        return w / w.mean()

    def check_scores(self, stock_bank, exposure_bank):
        """(E,) G of every episode that fits in the check years, in a fixed order, so two
        trees scored here are compared on identical episodes (a paired test)."""
        starts = self._starts_check.astype(np.int64)
        return self.rollout(stock_bank, exposure_bank, starts, self.T_hold)["G"]

    def seed_kernels(self, seed):
        pass

    def condition_groups(self, starts):
        """The period an episode starts in: 0 before 2010, 1 in 2010-2014, 2 from 2015.
        btind rejects a move that loses significantly in any group: a rule that pays in one
        stretch of history and clearly loses in another is fitting that history (stage 0 of
        the first run: +16 %/yr in training, -15 on 2020-2021). Losing windows, years and
        rallies are allowed; losing over a whole period is not."""
        return np.searchsorted(self._blocks, np.asarray(starts), side="right")

    # PER-STOCK CREDIT, the portfolio's critic. Buys share the money in proportion, so what
    # holding stock i through one decision period added is, to first order, its weight times
    # its return AGAINST the day's tradable universe: that relative return is the credit of
    # "hold i" vs "exit i" at that decision, exact enough and free to compute for every
    # stock-decision of an episode -- some 20,000 per 150 episodes, where the paired test
    # sees one number per episode. It only AIMS the search (anchor rows, the column prior);
    # every move is still accepted by btind's paired test on the held-out years. It is
    # measured on SEARCH episodes (the requests stay below accept_n).
    CREDIT_TAIL = 0.05             # the best and the worst 5% of stock-decisions anchor thresholds
    IC_FULL = 0.10                 # a column whose credit IC reaches this gets the full prior

    def _credit(self, out, starts):
        """(E, D, N) each stock's return over the decision period after each decision, minus
        the mean over that day's tradable stocks; NaN where it was not tradable."""
        de = self.decide_every
        D = out["avail"].shape[1]
        s = np.asarray(starts, np.int64)[:, None]
        t = s + de * np.arange(D)[None, :]                                                # (E, D)
        t1 = np.minimum(t + de, s + self.T)          # never past the episode's own last open
        r = self._open[t1] / np.where(self._open[t] > 0, self._open[t], np.nan) - 1.0      # (E, D, N)
        r = np.where(out["avail"], r, np.nan)
        return r - np.nanmean(r, axis=2, keepdims=True)

    def coverage_rows(self, bank, n_ep=150, seed=0):
        """Rows of the searched agent for proposing guards. Stocks: on-policy rows, and the
        stock-decisions of the best and worst CREDIT_TAIL of per-stock credit -- where acting
        differently on one stock paid or cost most (btind's "hot rows"). They replace the rows
        of each episode's two worst periods, which held every stock of those days, winners and
        losers alike. Exposure: on-policy rows and the decisions before the two worst periods."""
        rng = np.random.default_rng(seed)
        sb, eb = self.banks(bank)
        starts = self.sample_starts(n_ep, rng)
        out = self.run(sb, eb, starts, self.T, trace=True)
        if self.agent == "stocks":
            obs, ok = out["obs"], out["avail"]
            c = self._credit(out, starts)
            lo, hi = np.nanquantile(c, [self.CREDIT_TAIL, 1 - self.CREDIT_TAIL])
            tails = ok & np.isfinite(c) & ((c <= lo) | (c >= hi))
            return obs[ok], obs[tails]
        worst = np.argsort(out["decision_ret"], axis=1)[:, :2]
        ep = np.repeat(np.arange(n_ep), 2)
        obs = out["obs_e"]
        return obs.reshape(-1, obs.shape[-1]), obs[ep, worst.ravel()]

    def column_prior(self, bank, n_ep=60, seed=4711):
        """(n_stock_columns,) in [0, 1] for btind's proposal weights (rlfit, once per world): each
        column's mean CROSS-SECTIONAL rank IC with per-stock credit -- one Spearman correlation
        per decision of n_ep SEARCH episodes, across that day's tradable stocks, averaged (as the
        signal audit measures inputs; a correlation pooled over all rows would mix in what
        differs between days) -- on an ABSOLUTE scale, IC_FULL gets 1: a percentile would spread
        pure noise over [0, 1]. The exposure tree gets none."""
        if self.agent != "stocks":
            return None
        from scipy.stats import rankdata
        rng = np.random.default_rng(seed)
        sb, eb = self.banks(bank)
        starts = self.sample_starts(n_ep, rng)
        out = self.run(sb, eb, starts, self.T, trace=True)
        c = self._credit(out, starts)
        obs, keep = out["obs"], out["avail"] & np.isfinite(c)
        del out
        E, D = keep.shape[:2]
        ic_sum, n_sl = np.zeros(obs.shape[-1]), 0
        for e in range(E):
            for k in range(D):
                m = keep[e, k]
                if m.sum() < 20:
                    continue
                ry = rankdata(c[e, k][m])
                ry = (ry - ry.mean()) / ry.std()
                X = obs[e, k][m]
                R = np.apply_along_axis(rankdata, 0, X)
                sd = R.std(0)
                ic = np.where(sd > 0, ((R - R.mean(0)) / np.where(sd > 0, sd, 1) * ry[:, None]).mean(0), 0.0)
                ic_sum += ic
                n_sl += 1
        if n_sl < 20:
            return None
        return np.clip(np.abs(ic_sum / n_sl) / self.IC_FULL, 0.0, 1.0)

    def arm_trace(self, bank, pol_fn, n_ep=200, seed=11):
        """(arm per row and decision, row alive) for btind's `churn` (termination search),
        which otherwise steps a world one tick at a time through observe / step. The
        reference run records every decision's rows; they are replayed through the searched
        tree in order, so latches evolve as in the run (an untradable stock's is cleared)."""
        rng = np.random.default_rng(seed)
        sb, eb = self.banks(bank)
        out = self.run(sb, eb, self.sample_starts(n_ep, rng), self.T, trace=True)
        pol = pol_fn(bank)
        if self.agent == "stocks":
            obs, ok = out["obs"], out["avail"]                 # (E, K, N, W), (E, K, N)
            E, K, N = ok.shape
            rows = lambda k: obs[:, k].reshape(E * N, -1)
            alive = lambda k: ok[:, k].reshape(-1)
            pol.reset(E * N)
        else:
            obs = out["obs_e"]                                  # (E, K, W)
            K = obs.shape[1]
            rows = lambda k: obs[:, k]
            alive = lambda k: np.ones(len(obs), bool)
            pol.reset(len(obs))
        A, AL = [], []
        for k in range(K):
            o = rows(k)
            A.append(np.asarray(pol.arbitrate(pol.z(o))).copy())
            pol.act(o)
            off = ~alive(k)
            pol.latch[off], pol.step[off] = -1, 0
            AL.append(alive(k))
        return np.array(A).T, np.array(AL).T

    def score_bank(self, bank, pol_fn, starts, T):
        sb, eb = self.banks(bank)
        return self.rollout(sb, eb, starts, self.episode_length(starts, T))["G"]

    # ---------------------------------------------------------------- numba path
    def rollout(self, stock_bank, exposure_bank, starts, T, daily=False):
        from btind.tick import flatten, tick_args
        fs = flatten(stock_bank, len(self.stock_names))
        fe = flatten(exposure_bank, len(self.exposure_names))
        if fs["has_kern"] or fe["has_kern"] or (stock_bank.get("mem") or exposure_bank.get("mem")):
            return self.run(stock_bank, exposure_bank, starts, T, full=daily)
        starts = np.asarray(starts, np.int64)
        E = len(starts)
        G = np.empty(E)
        CG = np.empty(E)
        DG = np.empty(E)
        D = np.empty((E, T)) if daily else np.empty((0, 0))
        turn = np.empty(E)
        _rollout(starts, int(T), self.decide_every, self._X, self._M, self._open, self._avail,
                 self._sig, self._rf, self._infl, self._spx_ret, self.budget, self.cost_bps * 1e-4,
                 self._levels, OBJECTIVES[self.objective], self.risk_aversion, self.fractional,
                 self.max_weight, self._cap_scale, self.dd_weight, self._size,
                 tick_args(fs), np.ascontiguousarray(fs["laws"]),
                 tick_args(fe), np.ascontiguousarray(fe["laws"]), G, CG, DG, D, turn, daily,
                 int(self.max_names), int(self.core_names), float(self.core_frac))
        out = {"G": G, "turnover": turn, "cer_gap": CG, "dd_gap": DG}
        if daily:
            days = starts[:, None] + np.arange(T)[None, :]
            out.update(daily=D, bench=self._spx_ret[days], rf=self._rf[days], infl=self._infl[days])
        return out

    def _core_mask(self, t, avail):
        """(E, N) the core_names largest tradable companies at t (ties: the earlier stock)."""
        size = np.where(avail, np.maximum(self._size[t], 0.0), -np.inf)
        out = np.zeros(avail.shape, bool)
        for e in range(avail.shape[0]):
            idx = np.where(avail[e])[0]
            order = idx[np.lexsort((idx, -size[e, idx]))]
            out[e, order[:self.core_names]] = True
        return out

    def _limit_names(self, act, held, t):
        """(E, N) actions with at most `max_names` stocks bought or held per portfolio: the
        largest by company size (inverse volatility without sizes) stay, the rest exit; ties
        keep the earlier stock (as the kernel)."""
        want = (act == BUY) | (held & (act == HOLD))
        if self._size.size:
            key = np.maximum(self._size[t], 0.0)
        else:
            key = 1.0 / np.maximum(self._X[t][:, :, self._sig].astype(np.float64), 1e-6)
        out = act.copy()
        for e in range(act.shape[0]):
            idx = np.where(want[e])[0]
            if len(idx) > self.max_names:
                order = idx[np.lexsort((idx, -key[e, idx]))]           # largest first, then earlier
                out[e, order[self.max_names:]] = EXIT
        return out

    # ---------------------------------------------------------------- numpy reference
    def run(self, stock_bank, exposure_bank, starts, T, trace=False, full=False):
        """The reference simulation with btind's MemBank; traces for coverage rows. Also
        returns `exposure` (E, decisions): the level the exposure tree chose at each decision."""
        from btind.memory import MemBank
        starts = np.asarray(starts, np.int64)
        E, N, cost = len(starts), self._N, self.cost_bps * 1e-4
        ps = MemBank(stock_bank, len(self.stock_names))
        pe = MemBank(exposure_bank, len(self.exposure_names))
        ps.reset(E * N)
        pe.reset(E)
        cash = np.full(E, self.budget)
        shares = np.zeros((E, N))
        days_held = np.zeros((E, N))
        peak = np.full(E, self.budget)
        n_dec = int(np.ceil(T / self.decide_every))
        daily_p = np.zeros((E, T))
        traded = np.zeros(E)
        f_prev = np.full(E, -1)                  # the exposure level chosen last time, -1 at the start
        f_days = np.zeros(E)                     # trading days at that level
        obs_tr, obse_tr, avail_tr, dec_ret, f_tr = [], [], [], np.zeros((E, n_dec)), []
        for k in range(n_dec):
            t = starts + k * self.decide_every
            px = self._open[t]
            val = np.where(shares > 0, shares * np.nan_to_num(px), 0.0)
            V = cash + val.sum(1)
            peak = np.maximum(peak, V)
            held = shares > 0
            n_held = held.sum(1)
            avail = self._avail[t]
            n_av = np.maximum(avail.sum(1), 1)
            port = np.stack([held.astype(float), val / V[:, None], days_held / 252.0,
                             np.broadcast_to((cash / V)[:, None], (E, N)),
                             np.broadcast_to((n_held / n_av)[:, None], (E, N)),
                             np.broadcast_to((V / peak - 1.0)[:, None], (E, N))], axis=2)
            obs = np.concatenate([self._X[t], port], axis=2)
            act = np.asarray(ps.act(obs.reshape(E * N, -1))).reshape(E, N)
            # an untradable stock is sold and forgets its latch (the kernel skips it)
            off = ~avail.reshape(-1)
            ps.latch[off], ps.step[off] = -1, 0
            act = np.where(avail, act, EXIT)
            buy = act == BUY
            obs_e = np.column_stack([self._M[t], cash / V, n_held / n_av, V / peak - 1.0,
                                     (V - cash) / V, buy.sum(1) / n_av, f_days / 252.0])
            f_idx = np.asarray(pe.act(obs_e))
            f_days = np.where(f_idx == f_prev, f_days + self.decide_every, 0.0)
            f_prev = f_idx
            f = self._levels[f_idx]
            f_tr.append(f)
            if self.max_names:
                act = self._limit_names(act, held, t)
                buy = act == BUY
            if trace:
                obs_tr.append(obs)
                obse_tr.append(obs_e)
                avail_tr.append(avail)
            # --- allocation
            keep = held & (act == HOLD)
            K = np.where(keep, val, 0.0).sum(1)
            S = f * V
            over = K > S
            scale = np.where(over, S / np.maximum(K, 1e-12), 1.0)
            cap = self.max_weight * S[:, None] * self._cap_scale[t]   # largest position per stock, dollars
            kept = shares * np.minimum(scale[:, None], cap / np.maximum(val, 1e-12))
            if self.core_names:
                core = self._core_mask(t, avail)
                lim = self.core_frac * S
                Kcore = np.where(core & keep, kept * np.nan_to_num(px), 0.0).sum(1)
                cs = np.where(Kcore > lim, lim / np.maximum(Kcore, 1e-12), 1.0)
                kept = np.where(core & keep, kept * cs[:, None], kept)
            B = np.maximum(S - np.where(keep, kept * np.nan_to_num(px), 0.0).sum(1), 0.0)
            if self._size.size:
                inv_vol = np.where(buy, np.maximum(self._size[t], 0.0), 0.0)
            else:
                inv_vol = np.where(buy, 1.0 / np.maximum(self._X[t][:, :, self._sig].astype(np.float64), 1e-6), 0.0)
            if self.core_names:
                room = np.maximum(lim - np.minimum(Kcore, lim), 0.0)
                inv_c = np.where(core, inv_vol, 0.0).sum(1)
                inv_a = inv_vol.sum(1)
                Bc = np.minimum(np.where(inv_a > 0, B * inv_c / np.maximum(inv_a, 1e-300), 0.0), room)
                alloc = (_capped_split(Bc, np.where(core, inv_vol, 0.0), cap)
                         + _capped_split(B - Bc, np.where(core, 0.0, inv_vol), cap))
            else:
                alloc = _capped_split(B, inv_vol, cap)
            px0 = np.where(np.isnan(px), np.inf, px)
            target = alloc / (px0 * (1 + 2 * cost))
            if not self.fractional:
                target, kept = np.floor(target), np.floor(kept)
            new = np.where(keep, kept, np.where(buy, target, 0.0))
            pxn = np.nan_to_num(px)
            new = np.where(new * pxn < MIN_POSITION, 0.0, new)
            trade = (np.abs(new - shares) * pxn).sum(1)
            traded += trade / V
            cash = V - (new * pxn).sum(1) - trade * cost
            days_held = np.where(new > 0, np.where(held, days_held + self.decide_every, 0.0), 0.0)
            shares = new
            # --- mark to market at every open until the next decision
            v_prev = V.copy()
            for j in range(self.decide_every):
                d = k * self.decide_every + j
                if d >= T:
                    break
                tt = t + j + 1
                cash = cash * (1.0 + self._rf[tt - 1])
                v_new = cash + np.where(shares > 0, shares * np.nan_to_num(self._open[tt]), 0.0).sum(1)
                daily_p[:, d] = v_new / v_prev - 1.0
                v_prev = v_new
            dec_ret[:, k] = v_prev / V - 1.0
        days = starts[:, None] + np.arange(T)[None, :]
        rf, bench, infl = self._rf[days], self._spx_ret[days], self._infl[days]
        if self.objective == "cer":
            cer = lambda x: 100.0 * 252.0 * (x.mean(1) - 0.5 * self.risk_aversion * x.var(1, ddof=1))
            real = lambda x: (1.0 + x) / (1.0 + infl) - 1.0
            gap = cer(real(daily_p)) - cer(real(bench))
        elif self.objective == "alpha":
            # the episode's own beta against the S&P (OLS on daily excess returns), and the
            # return left after it, % per year
            xp, xb = daily_p - rf, bench - rf
            xpc, xbc = xp - xp.mean(1, keepdims=True), xb - xb.mean(1, keepdims=True)
            beta = (xpc * xbc).sum(1) / np.maximum((xbc * xbc).sum(1), 1e-18)
            gap = 100.0 * 252.0 * (xp.mean(1) - beta * xb.mean(1))
        else:
            sharpe = lambda x: np.sqrt(252.0) * x.mean(1) / np.maximum(x.std(1, ddof=1), 1e-12)
            gap = sharpe(daily_p - rf) - sharpe(bench - rf)
        dd_gap = 100.0 * (max_drawdown(daily_p) - max_drawdown(bench))
        G = gap - self.dd_weight * dd_gap
        out = {"G": G, "turnover": traded, "cer_gap": gap, "dd_gap": dd_gap, "exposure": np.stack(f_tr, 1)}
        if trace:
            out.update(obs=np.stack(obs_tr, 1), obs_e=np.stack(obse_tr, 1), avail=np.stack(avail_tr, 1),
                       decision_ret=dec_ret)
        if full:
            out.update(daily=daily_p, bench=bench, rf=rf, infl=infl)
        return out


def max_drawdown(daily):
    """(E,) the deepest fall from the running high of each row's value path, a fraction;
    the path starts at 1 (the budget), so a fall on the first day counts."""
    w = np.cumprod(1.0 + daily, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(w), 1)), w], axis=1), axis=1)[:, 1:]
    return (1.0 - w / peak).max(axis=1)


def _capped_split(B, weights, cap):
    """Split B[e] over the positive weights[e, :] in proportion, none above cap[e, i]
    (water-filling): a share that would pass its cap is fixed at it and the rest re-split
    among the others, each pass judged on that pass's budget; what no one can take is left."""
    N = weights.shape[1]
    live = weights > 0
    capped = np.zeros(weights.shape, bool)
    rem = np.asarray(B, float).copy()
    for _ in range(N):
        free = live & ~capped
        wsum = np.where(free, weights, 0.0).sum(1)
        share = np.where(free, rem[:, None] * weights / np.maximum(wsum, 1e-300)[:, None], 0.0)
        new = free & (share > cap)
        if not new.any():
            break
        capped |= new
        rem = rem - np.where(new, cap, 0.0).sum(1)
    free = live & ~capped
    wsum = np.where(free, weights, 0.0).sum(1)
    share = np.where(free, np.maximum(rem, 0.0)[:, None] * weights / np.maximum(wsum, 1e-300)[:, None], 0.0)
    return np.where(capped, cap, share)


# ------------------------------------------------------------------ the compiled rollout
@njit(cache=True, inline="always")
def _tick_tree(z, latch, step, a):
    """btind's `_tick` with its 23 arrays passed from the tuple `tick_args` built
    (numba cannot star-unpack a tuple into an inlined call)."""
    return _tick(z, latch, step, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
                 a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20],
                 a[21], a[22])


@njit(cache=True, inline="always")
def _argmax_law(z, width, laws, law):
    """argmax over actions of [z, 1] @ laws[law]; the first maximum wins, as np.argmax."""
    n_out = laws.shape[2]
    best, best_v = 0, -np.inf
    for o in range(n_out):
        v = laws[law, width, o]
        for j in range(width):
            v += z[j] * laws[law, j, o]
        if v > best_v:
            best, best_v = o, v
    return best


@njit(cache=True, inline="always")
def _fill(inv, cap, cscale_t, budget, core, want_core, capped):
    """Water-filling of `budget` over the buys of one group (core == want_core), none above
    cap * cscale_t[i] -- the ungrouped loop of _rollout, restricted: (budget left, weight sum)."""
    N = inv.shape[0]
    rem = budget
    for _ in range(N):
        wsum = 0.0
        for i in range(N):
            if core[i] == want_core and inv[i] > 0 and not capped[i]:
                wsum += inv[i]
        if wsum <= 0.0:
            break
        n_new = 0
        paid = 0.0
        for i in range(N):
            if core[i] == want_core and inv[i] > 0 and not capped[i] and rem * inv[i] / wsum > cap * cscale_t[i]:
                capped[i] = True
                n_new += 1
                paid += cap * cscale_t[i]
        if n_new == 0:
            break
        rem -= paid
    rem = max(rem, 0.0)
    wsum = 0.0
    for i in range(N):
        if core[i] == want_core and inv[i] > 0 and not capped[i]:
            wsum += inv[i]
    return rem, wsum


@njit(cache=True, parallel=True)
def _rollout(starts, T, de, X, M, open_, avail, sig_col, rf, infl, spx_ret, budget, cost, levels,
             objective, gamma, frac, max_w, cscale, dd_w, size, st, s_laws, et, e_laws, G, CG, DG, D,
             turnover, want_daily, max_n, core_n, core_frac):
    E = starts.shape[0]
    Tall, N, F = X.shape
    Mw = M.shape[1]
    n_dec = (T + de - 1) // de
    for e in prange(E):
        s = starts[e]
        cash = budget
        peak = budget
        shares = np.zeros(N)
        days_held = np.zeros(N)
        latch_s = np.full(N, -1, np.int64)
        step_s = np.zeros(N, np.int64)
        latch_e, step_e = -1, 0
        act = np.zeros(N, np.int64)
        inv = np.zeros(N)
        capped = np.zeros(N, np.bool_)
        core = np.zeros(N, np.bool_)
        use_core = core_n > 0 and size.shape[0] > 0
        z = np.zeros(F + 8)                      # obs (F + 6 portfolio columns), V_hat, leverage
        ze = np.zeros(Mw + 8)                    # market (Mw + 6 state columns), V_hat, leverage
        daily = np.zeros(T)
        traded = 0.0
        f_prev, f_days = -1, 0.0                 # last exposure level chosen, days at it
        for k in range(n_dec):
            t = s + k * de
            V = cash
            n_held = 0
            for i in range(N):
                if shares[i] > 0:
                    V += shares[i] * open_[t, i]
                    n_held += 1
            if V > peak:
                peak = V
            dd = V / peak - 1.0
            n_av = 0
            for i in range(N):
                if avail[t, i]:
                    n_av += 1
            n_av = max(n_av, 1)
            # --- stock tree, one row per TRADABLE stock; the rest are sold and forget
            n_buy = 0
            for i in range(N):
                if not avail[t, i]:
                    act[i] = 0
                    latch_s[i] = -1
                    step_s[i] = 0
                    continue
                for j in range(F):
                    z[j] = X[t, i, j]
                held = 1.0 if shares[i] > 0 else 0.0
                z[F] = held
                z[F + 1] = shares[i] * open_[t, i] / V if shares[i] > 0 else 0.0
                z[F + 2] = days_held[i] / 252.0
                z[F + 3] = cash / V
                z[F + 4] = n_held / n_av
                z[F + 5] = dd
                law, latch_s[i], step_s[i] = _tick_tree(z, latch_s[i], step_s[i], st)
                a = _argmax_law(z, F + 8, s_laws, law)
                act[i] = a
                if a == 2:
                    n_buy += 1
            # --- exposure tree, one row per portfolio
            for j in range(Mw):
                ze[j] = M[t, j]
            ze[Mw] = cash / V
            ze[Mw + 1] = n_held / n_av
            ze[Mw + 2] = dd
            ze[Mw + 3] = (V - cash) / V
            ze[Mw + 4] = n_buy / n_av
            ze[Mw + 5] = f_days / 252.0
            law, latch_e, step_e = _tick_tree(ze, latch_e, step_e, et)
            f_idx = _argmax_law(ze, Mw + 8, e_laws, law)
            f_days = f_days + de if f_idx == f_prev else 0.0
            f_prev = f_idx
            f = levels[f_idx]
            # --- at most max_n names: sell the smallest of those wanted until max_n are left
            if max_n > 0:
                n_want = 0
                for i in range(N):
                    if act[i] == 2 or (act[i] == 1 and shares[i] > 0):
                        n_want += 1
                while n_want > max_n:
                    worst, worst_v = -1, np.inf
                    for i in range(N):
                        if act[i] == 2 or (act[i] == 1 and shares[i] > 0):
                            v = max(size[t, i], 0.0) if size.shape[0] > 0 else 1.0 / max(X[t, i, sig_col], 1e-6)
                            if v < worst_v or (v == worst_v and i > worst):     # ties: the later stock goes
                                worst, worst_v = i, v
                    act[worst] = 0
                    n_want -= 1
            # --- allocation
            K = 0.0
            for i in range(N):
                if act[i] == 1 and shares[i] > 0:
                    K += shares[i] * open_[t, i]
            S = f * V
            cap = max_w * S
            scale = S / max(K, 1e-12) if K > S else 1.0
            cs, lim, Kcore = 1.0, core_frac * S, 0.0
            if use_core:                              # the core: the core_n largest tradable
                for i in range(N):
                    core[i] = False
                for c in range(core_n):
                    best, best_v = -1, -np.inf
                    for i in range(N):
                        if avail[t, i] and not core[i] and max(size[t, i], 0.0) > best_v:
                            best, best_v = i, max(size[t, i], 0.0)
                    if best < 0:
                        break
                    core[best] = True
                for i in range(N):
                    if core[i] and act[i] == 1 and shares[i] > 0:
                        Kcore += min(shares[i] * open_[t, i] * scale, cap * cscale[t, i])
                if Kcore > lim:
                    cs = lim / max(Kcore, 1e-12)
            Kc = 0.0                                  # kept value after the trim and the cap
            for i in range(N):
                if act[i] == 1 and shares[i] > 0:
                    v = min(shares[i] * open_[t, i] * scale, cap * cscale[t, i])
                    Kc += v * cs if (use_core and core[i]) else v
            B = max(S - Kc, 0.0)
            # buys by inverse volatility, none above the cap: water-filling
            for i in range(N):
                if act[i] != 2:
                    inv[i] = 0.0
                elif size.shape[0] > 0:
                    inv[i] = max(size[t, i], 0.0)
                else:
                    inv[i] = 1.0 / max(X[t, i, sig_col], 1e-6)
                capped[i] = False
            remC, wsumC = 0.0, 0.0
            if not use_core:
                rem = B
                for _ in range(N):
                    wsum = 0.0
                    for i in range(N):
                        if inv[i] > 0 and not capped[i]:
                            wsum += inv[i]
                    if wsum <= 0.0:
                        break
                    n_new = 0
                    paid = 0.0
                    for i in range(N):              # judged on this pass's budget, then paid
                        if inv[i] > 0 and not capped[i] and rem * inv[i] / wsum > cap * cscale[t, i]:
                            capped[i] = True
                            n_new += 1
                            paid += cap * cscale[t, i]
                    if n_new == 0:
                        break
                    rem -= paid
                rem = max(rem, 0.0)
                wsum = 0.0
                for i in range(N):
                    if inv[i] > 0 and not capped[i]:
                        wsum += inv[i]
            else:                                   # two groups: the core at most its room
                inv_c, inv_a = 0.0, 0.0
                for i in range(N):
                    inv_a += inv[i]
                    if core[i]:
                        inv_c += inv[i]
                room = max(lim - min(Kcore, lim), 0.0)
                Bc = min(B * inv_c / max(inv_a, 1e-300), room) if inv_a > 0 else 0.0
                remC, wsumC = _fill(inv, cap, cscale[t], Bc, core, True, capped)
                rem, wsum = _fill(inv, cap, cscale[t], B - Bc, core, False, capped)
            spent = 0.0
            trade = 0.0
            for i in range(N):
                px = open_[t, i]
                if act[i] == 1 and shares[i] > 0:
                    new = shares[i] * min(scale, cap * cscale[t, i] / max(shares[i] * px, 1e-12))
                    if use_core and core[i]:
                        new *= cs
                elif act[i] == 2:
                    if use_core and core[i]:
                        a = cap * cscale[t, i] if capped[i] else (remC * inv[i] / wsumC if wsumC > 0.0 else 0.0)
                    else:
                        a = cap * cscale[t, i] if capped[i] else (rem * inv[i] / wsum if wsum > 0.0 else 0.0)
                    new = a / (px * (1 + 2 * cost))
                else:
                    new = 0.0
                if not frac:
                    new = np.floor(new)
                if new * px < MIN_POSITION:
                    new = 0.0
                if new > 0 or shares[i] > 0:
                    trade += abs(new - shares[i]) * px
                    spent += new * px
                if new > 0:
                    days_held[i] = days_held[i] + de if shares[i] > 0 else 0.0
                else:
                    days_held[i] = 0.0
                shares[i] = new
            traded += trade / V
            cash = V - spent - trade * cost
            # --- mark to market at every open until the next decision
            v_prev = V
            for j in range(de):
                d = k * de + j
                if d >= T:
                    break
                tt = t + j + 1
                cash = cash * (1.0 + rf[tt - 1])
                v_new = cash
                for i in range(N):
                    if shares[i] > 0:
                        v_new += shares[i] * open_[tt, i]
                daily[d] = v_new / v_prev - 1.0
                v_prev = v_new
        # --- score: portfolio minus S&P 500, see the module docstring
        mp, mb = 0.0, 0.0
        for d in range(T):
            if objective == 1:
                mp += (1.0 + daily[d]) / (1.0 + infl[s + d]) - 1.0
                mb += (1.0 + spx_ret[s + d]) / (1.0 + infl[s + d]) - 1.0
            else:
                mp += daily[d] - rf[s + d]
                mb += spx_ret[s + d] - rf[s + d]
        mp /= T
        mb /= T
        vp, vb, cpb = 0.0, 0.0, 0.0
        for d in range(T):
            if objective == 1:
                xp = (1.0 + daily[d]) / (1.0 + infl[s + d]) - 1.0 - mp
                xb = (1.0 + spx_ret[s + d]) / (1.0 + infl[s + d]) - 1.0 - mb
            else:
                xp = daily[d] - rf[s + d] - mp
                xb = spx_ret[s + d] - rf[s + d] - mb
            vp += xp * xp
            vb += xb * xb
            cpb += xp * xb
        vp /= T - 1
        vb /= T - 1
        cpb /= T - 1
        if objective == 1:
            gap = 100.0 * 252.0 * ((mp - 0.5 * gamma * vp) - (mb - 0.5 * gamma * vb))
        elif objective == 2:
            gap = 100.0 * 252.0 * (mp - cpb / max(vb, 1e-18) * mb)                # alpha after beta
        else:
            gap = np.sqrt(252.0) * (mp / max(np.sqrt(vp), 1e-12) - mb / max(np.sqrt(vb), 1e-12))
        wp, wb, pp, pb, ddp, ddb = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
        for d in range(T):
            wp *= 1.0 + daily[d]
            wb *= 1.0 + spx_ret[s + d]
            pp = max(pp, wp)
            pb = max(pb, wb)
            ddp = max(ddp, 1.0 - wp / pp)
            ddb = max(ddb, 1.0 - wb / pb)
        dd_gap = 100.0 * (ddp - ddb)
        CG[e] = gap
        DG[e] = dd_gap
        G[e] = gap - dd_w * dd_gap
        turnover[e] = traded
        if want_daily:
            for d in range(T):
                D[e, d] = daily[d]
