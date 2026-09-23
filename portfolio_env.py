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
            Chooses the invested fraction: 25%, 50%, 75% or 100% (the rest in T-bills)

ALLOCATION (fixed rules, not learned). With exposure f and portfolio value V the stocks
may hold f * V. "hold" positions keep their shares; if they alone exceed f * V they are
trimmed proportionally. What is left of f * V goes to the "buy" stocks by inverse
forecast volatility (1 / sig_21d). No position may exceed `max_weight` of f * V (1/3: at
least three stocks to be fully invested); a buy that would is capped and its excess goes to
the other buys, and what no stock can take stays in cash -- nothing is bought on the tree's
behalf, so a tree that picks fewer than 1 / max_weight stocks pays for the idle cash. Long-only, no leverage, `cost_bps` on every traded
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

`score_bank` returns one number per episode (btind's `structure.score` hook on the
`portfolio-env` branch).

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
EXPOSURE_STATE = ["cash_frac", "n_held_frac", "drawdown", "exposure", "buy_frac"]
STOCK_ACTIONS = ["exit", "hold", "buy"]
EXPOSURE_LEVELS = (0.25, 0.50, 0.75, 1.00)
EXPOSURE_ACTIONS = [f"invest {int(100 * x)}%" for x in EXPOSURE_LEVELS]
EXIT, HOLD, BUY = 0, 1, 2
OBJECTIVES = {"sharpe": 0, "cer": 1}
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

    def __init__(self, panel, F, M, stock_names, market_names, start, end, T=252,
                 decide_every=5, budget=100_000.0, cost_bps=5.0, source="yfinance-sp100",
                 agent="stocks", objective="cer", risk_aversion=None, fractional=False,
                 max_weight=1.0, store_root=os.path.join(ROOT, "runs_bt")):
        # scalar attributes are the world's signature in btind's store
        self.split_start, self.split_end = str(start), str(end)
        self.T, self.decide_every = int(T), int(decide_every)
        self.budget, self.cost_bps, self.source = float(budget), float(cost_bps), str(source)
        self.agent = str(agent)
        self.objective = str(objective)
        self.benchmark = "SPY total return"
        self.fractional = bool(fractional)
        self.max_weight = float(max_weight)
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
        self.stock_names = list(stock_names) + list(market_names) + PORTFOLIO_NAMES
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
        X = np.concatenate([F, np.broadcast_to(M[:, None, :], (T_all, N, M.shape[1]))], axis=2)
        self._X = np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float64)
        self._M = np.ascontiguousarray(np.nan_to_num(M, nan=0.0), dtype=np.float64)
        self._rf = np.ascontiguousarray(panel.rf, dtype=np.float64)
        self._infl = np.ascontiguousarray(panel.infl, dtype=np.float64)
        self._spx = panel.bench_open     # SPY with dividends reinvested (stocks_data.MARKET)
        self._spx_ret = np.ascontiguousarray(np.r_[self._spx[1:] / self._spx[:-1] - 1.0, 0.0])
        self._starts = np.arange(a, b - self.T - 1)
        self._levels = np.array(EXPOSURE_LEVELS)
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

    # ---------------------------------------------------------------- btind world API
    def sample_starts(self, n, rng):
        return rng.choice(self._starts, size=n, replace=True).astype(np.int64)

    def seed_kernels(self, seed):
        pass

    def condition_groups(self, starts):
        """The period an episode starts in: 0 before 2010, 1 in 2010-2014, 2 from 2015.
        btind rejects a move that loses significantly in any group: a rule that pays in one
        stretch of history and clearly loses in another is fitting that history (stage 0 of
        the first run: +16 %/yr in training, -15 on 2020-2021). Losing windows, years and
        rallies are allowed; losing over a whole period is not."""
        return np.searchsorted(self._blocks, np.asarray(starts), side="right")

    def coverage_rows(self, bank, n_ep=150, seed=0):
        """Rows of the searched agent for proposing guards: on-policy rows, and the rows of
        the decisions that preceded each episode's two worst periods."""
        rng = np.random.default_rng(seed)
        sb, eb = self.banks(bank)
        out = self.run(sb, eb, self.sample_starts(n_ep, rng), self.T, trace=True)
        worst = np.argsort(out["decision_ret"], axis=1)[:, :2]
        ep = np.repeat(np.arange(n_ep), 2)
        if self.agent == "stocks":
            obs, ok = out["obs"], out["avail"]
            return obs[ok], obs[ep, worst.ravel()][ok[ep, worst.ravel()]]
        obs = out["obs_e"]
        return obs.reshape(-1, obs.shape[-1]), obs[ep, worst.ravel()]

    def score_bank(self, bank, pol_fn, starts, T):
        sb, eb = self.banks(bank)
        return self.rollout(sb, eb, starts, T)["G"]

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
        D = np.empty((E, T)) if daily else np.empty((0, 0))
        turn = np.empty(E)
        _rollout(starts, int(T), self.decide_every, self._X, self._M, self._open, self._avail,
                 self._sig, self._rf, self._infl, self._spx_ret, self.budget, self.cost_bps * 1e-4,
                 self._levels, OBJECTIVES[self.objective], self.risk_aversion, self.fractional,
                 self.max_weight,
                 tick_args(fs), np.ascontiguousarray(fs["laws"]),
                 tick_args(fe), np.ascontiguousarray(fe["laws"]), G, D, turn, daily)
        out = {"G": G, "turnover": turn}
        if daily:
            days = starts[:, None] + np.arange(T)[None, :]
            out.update(daily=D, bench=self._spx_ret[days], rf=self._rf[days], infl=self._infl[days])
        return out

    # ---------------------------------------------------------------- numpy reference
    def run(self, stock_bank, exposure_bank, starts, T, trace=False, full=False):
        """The reference simulation with btind's MemBank; traces for coverage rows."""
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
        obs_tr, obse_tr, avail_tr, dec_ret = [], [], [], np.zeros((E, n_dec))
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
                                     (V - cash) / V, buy.sum(1) / n_av])
            f = self._levels[np.asarray(pe.act(obs_e))]
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
            cap = self.max_weight * S                          # largest position, dollars
            kept = shares * np.minimum(scale[:, None], cap[:, None] / np.maximum(val, 1e-12))
            B = np.maximum(S - np.where(keep, kept * np.nan_to_num(px), 0.0).sum(1), 0.0)
            inv_vol = np.where(buy, 1.0 / np.maximum(self._X[t][:, :, self._sig], 1e-6), 0.0)
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
            G = cer(real(daily_p)) - cer(real(bench))
        else:
            sharpe = lambda x: np.sqrt(252.0) * x.mean(1) / np.maximum(x.std(1, ddof=1), 1e-12)
            G = sharpe(daily_p - rf) - sharpe(bench - rf)
        out = {"G": G, "turnover": traded}
        if trace:
            out.update(obs=np.stack(obs_tr, 1), obs_e=np.stack(obse_tr, 1), avail=np.stack(avail_tr, 1),
                       decision_ret=dec_ret)
        if full:
            out.update(daily=daily_p, bench=bench, rf=rf, infl=infl)
        return out


def _capped_split(B, weights, cap):
    """Split B[e] over the positive weights[e, :] in proportion, none above cap[e]
    (water-filling): a share that would pass the cap is fixed at it and the rest re-split
    among the others, each pass judged on that pass's budget; what no one can take is left."""
    N = weights.shape[1]
    live = weights > 0
    capped = np.zeros(weights.shape, bool)
    rem = np.asarray(B, float).copy()
    for _ in range(N):
        free = live & ~capped
        wsum = np.where(free, weights, 0.0).sum(1)
        share = np.where(free, rem[:, None] * weights / np.maximum(wsum, 1e-300)[:, None], 0.0)
        new = free & (share > cap[:, None])
        if not new.any():
            break
        capped |= new
        rem = rem - new.sum(1) * cap
    free = live & ~capped
    wsum = np.where(free, weights, 0.0).sum(1)
    share = np.where(free, np.maximum(rem, 0.0)[:, None] * weights / np.maximum(wsum, 1e-300)[:, None], 0.0)
    return np.where(capped, cap[:, None], share)


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


@njit(cache=True, parallel=True)
def _rollout(starts, T, de, X, M, open_, avail, sig_col, rf, infl, spx_ret, budget, cost, levels,
             objective, gamma, frac, max_w, st, s_laws, et, e_laws, G, D, turnover, want_daily):
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
        z = np.zeros(F + 8)                      # obs (F + 6 portfolio columns), V_hat, leverage
        ze = np.zeros(Mw + 7)                    # market (Mw + 5 state columns), V_hat, leverage
        daily = np.zeros(T)
        traded = 0.0
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
            law, latch_e, step_e = _tick_tree(ze, latch_e, step_e, et)
            f = levels[_argmax_law(ze, Mw + 7, e_laws, law)]
            # --- allocation
            K = 0.0
            for i in range(N):
                if act[i] == 1 and shares[i] > 0:
                    K += shares[i] * open_[t, i]
            S = f * V
            cap = max_w * S
            scale = S / max(K, 1e-12) if K > S else 1.0
            Kc = 0.0                                  # kept value after the trim and the cap
            for i in range(N):
                if act[i] == 1 and shares[i] > 0:
                    Kc += min(shares[i] * open_[t, i] * scale, cap)
            B = max(S - Kc, 0.0)
            # buys by inverse volatility, none above the cap: water-filling
            for i in range(N):
                inv[i] = 1.0 / max(X[t, i, sig_col], 1e-6) if act[i] == 2 else 0.0
                capped[i] = False
            rem = B
            for _ in range(N):
                wsum = 0.0
                for i in range(N):
                    if inv[i] > 0 and not capped[i]:
                        wsum += inv[i]
                if wsum <= 0.0:
                    break
                n_new = 0
                for i in range(N):              # judged on this pass's budget, then paid
                    if inv[i] > 0 and not capped[i] and rem * inv[i] / wsum > cap:
                        capped[i] = True
                        n_new += 1
                if n_new == 0:
                    break
                rem -= n_new * cap
            rem = max(rem, 0.0)
            wsum = 0.0
            for i in range(N):
                if inv[i] > 0 and not capped[i]:
                    wsum += inv[i]
            spent = 0.0
            trade = 0.0
            for i in range(N):
                px = open_[t, i]
                if act[i] == 1 and shares[i] > 0:
                    new = shares[i] * min(scale, cap / max(shares[i] * px, 1e-12))
                elif act[i] == 2:
                    a = cap if capped[i] else (rem * inv[i] / wsum if wsum > 0.0 else 0.0)
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
        vp, vb = 0.0, 0.0
        for d in range(T):
            if objective == 1:
                xp = (1.0 + daily[d]) / (1.0 + infl[s + d]) - 1.0 - mp
                xb = (1.0 + spx_ret[s + d]) / (1.0 + infl[s + d]) - 1.0 - mb
            else:
                xp = daily[d] - rf[s + d] - mp
                xb = spx_ret[s + d] - rf[s + d] - mb
            vp += xp * xp
            vb += xb * xb
        vp /= T - 1
        vb /= T - 1
        if objective == 1:
            G[e] = 100.0 * 252.0 * ((mp - 0.5 * gamma * vp) - (mb - 0.5 * gamma * vb))
        else:
            G[e] = np.sqrt(252.0) * (mp / max(np.sqrt(vp), 1e-12) - mb / max(np.sqrt(vb), 1e-12))
        turnover[e] = traded
        if want_daily:
            for d in range(T):
                D[e, d] = daily[d]
