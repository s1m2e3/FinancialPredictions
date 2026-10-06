"""MacroWorld: a btind world that trades a universe of currency pairs and metals (macro12: also
index and commodity CFDs) HOUR BY HOUR, long, flat or short, with one shared behaviour tree --
one row per instrument.

ACTIONS per instrument:  0 short  1 flat  2 long  3 keep (whatever it holds now)
A position is entered at the risk-parity size -- signed notional = V * risk_per_asset /
(its annualised volatility forecast, floored at VOL_FLOOR), risk_per_asset = RISK_TOTAL / sqrt(N)
for N instruments -- and then held in UNITS: its
notional moves with its price, and nothing is traded while the tree keeps it. Every Monday at
12:00 UTC (`resize`) positions still held are brought back to their risk-parity size, a
fixed rule applied to every tree and baseline alike. N uncorrelated positions then carry
RISK_TOTAL (10%) together whatever N is -- 43 instruments: 1.5% each, about 8x gross notional
on currencies (a margin account, inside retail leverage limits); positions that lean the same
way (pairs share the dollar and the euro) carry more, and the drawdown term prices that.

WHAT A TRADE COSTS. Orders execute at the OPEN of the bar, at the mid plus or minus HALF THE
SPREAD quoted at that open (the actual Dukascopy bid and ask of that hour), plus COMMISSION
per side. A position held pays or earns its carry every hour (macro_data: the interest
differential of a currency pair; minus the US rate for an index or commodity CFD) and
FIN_MARKUP per year on its notional (the broker's financing margin), by the calendar hours to
the next bar, so a position held over a weekend pays the weekend.

TIMING. A decision at bar t reads features from bars that closed BEFORE t and trades at t's
open; positions are marked at every bar's mid open. Decisions every bar (`decide="hourly"`) or
once a day at 20:00 UTC (`decide="daily"`, the comparison: what does trading within the day
add). An instrument without a bar in an hour (outside its session) is neither evaluated nor
traded in it; it keeps its position and the tree's latch on it.

SCORE of an episode: certainty-equivalent EXCESS return over cash, % per year,
    CER = A * (mean(x) - gamma / 2 * var(x)),   x = the account's return per bar over cash,
A the grid's bars per year, gamma = 2.76, the risk aversion at which holding 100% S&P 500 was
optimal on 2006-2019 (the stock world's investor). Cash is the benchmark -- CER 0 -- so a
positive score is money made by trading after every cost; the report sets it against holding
the instruments. G = CER - dd_weight * 100 * max drawdown of the episode.

HELD-OUT AND CHECK YEARS, as the stock world: search episodes in the search years, the paired
acceptance test on the held-out years, the check years read by nothing in the search. The
acceptance test's clusters are calendar QUARTERS (episodes last 2 weeks; half-years would
leave a 3-year fold 6 clusters, fewer than min_clusters).

TWO IMPLEMENTATIONS OF ONE SIMULATION: `run` (numpy, btind's MemBank; traces) is the
reference, `_rollout` (numba, btind's compiled `_tick`) the kernel the search uses;
checks/verify_macro_kernel.py holds them to agreement.
"""
import os

import numpy as np
from numba import njit, prange

from portfolio_env import _argmax_law, _tick_tree, constant_bank, max_drawdown   # also puts btind on the path
from macro_data import FEATURES

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["short", "flat", "long", "keep"]
SHORT, FLAT, LONG, KEEP = 0, 1, 2, 3
STATE_NAMES = ["position", "weeks_held", "trade_pnl", "drawdown", "gross"]
RISK_TOTAL = 0.10             # annual volatility of N uncorrelated positions together: each gets RISK_TOTAL / sqrt(N)
VOL_FLOOR = 0.04              # sizing never assumes less volatility than this
COMMISSION = 0.25e-4          # per unit of notional traded, per side (0.25 bp; Dukascopy ~0.35 bp FX)
FIN_MARKUP = 0.01             # per year on |notional|: the broker's financing margin
HOURS_PER_YEAR = 8760.0
BARS_PER_WEEK = 120.0         # weeks_held: bars held / 120 (a trading week of the currency grid)


class MacroWorld:
    head = "argmax"
    gamma = 1.0
    min_clusters = 8
    agent = "assets"

    def __init__(self, X, sim, start, end, T=240, T_hold=240, decide="hourly", holdout_years=None,
                 check_years=None, accept_n=400, dd_weight=0.25, risk_aversion=2.76,
                 source="dukascopy-h1-12", store_root=os.path.join(ROOT, "runs_macro")):
        import pandas as pd
        # scalar attributes are the world's signature in btind's store
        self.split_start, self.split_end = str(start), str(end)
        self.T, self.T_hold, self.accept_n = int(T), int(T_hold), int(accept_n)
        self.decide = str(decide)
        self.dd_weight, self.risk_aversion, self.source = float(dd_weight), float(risk_aversion), str(source)
        self.risk_per_asset, self.vol_floor = RISK_TOTAL / np.sqrt(X.shape[1]), VOL_FLOOR
        self.commission, self.fin_markup = COMMISSION, FIN_MARKUP
        self.holdout = ",".join(str(y) for y in sorted(holdout_years)) if holdout_years else ""
        self.check = ",".join(str(y) for y in sorted(check_years)) if check_years else ""
        self.names = list(sim.get("features", FEATURES)) + STATE_NAMES       # the universe's features (macro_data)
        self.n_act = len(ACTIONS)
        self.actions = ACTIONS
        self.asset_names = list(sim["names"])
        self._store_root = store_root
        self._recorder = None
        times = pd.DatetimeIndex(sim["times"])
        self._times = times
        self._X = np.ascontiguousarray(X, dtype=np.float32)
        self._N = X.shape[1]
        self._avail = np.ascontiguousarray(sim["avail"], dtype=np.bool_)
        self._mid = np.ascontiguousarray(np.nan_to_num(sim["mid"], nan=1.0), dtype=np.float64)
        self._last = np.ascontiguousarray(np.nan_to_num(sim["last"], nan=1.0), dtype=np.float64)
        self._half = np.ascontiguousarray(np.nan_to_num(sim["half_spread"], nan=0.0), dtype=np.float64)
        self._carry = np.ascontiguousarray(np.nan_to_num(sim["carry"], nan=0.0), dtype=np.float64)
        self._vol = np.ascontiguousarray(np.nan_to_num(sim["vol"], nan=1.0), dtype=np.float64)
        self._hours = np.ascontiguousarray(sim["hours"], dtype=np.float64)
        self._decide = np.ascontiguousarray((times.hour == 20) if decide == "daily"
                                            else np.ones(len(times), bool), dtype=np.bool_)
        self._resize = np.ascontiguousarray((times.weekday == 0) & (times.hour == 12), dtype=np.bool_)
        a = np.searchsorted(times.values, np.datetime64(start))
        b = np.searchsorted(times.values, np.datetime64(pd.Timestamp(end) + pd.Timedelta(days=1)))
        span = times[a:b]
        # bars per year of the grid over this world's period: the annualisation of the score
        self.bars_per_year = float(round(len(span) / max((span[-1] - span[0]).days / 365.25, 1e-9)))
        self._span, self._years = (a, b), times.year.values
        self._check_years = list(check_years or [])
        self._blocks = np.searchsorted(times.values, [np.datetime64("2015-01-01"), np.datetime64("2018-01-01")])
        self._build_pools(holdout_years)
        self.partner = None

    # ---------------------------------------------------------------- banks
    @property
    def store_root(self):
        return self._store_root

    def default_bank(self, agent=None):
        """All flat: cash, the benchmark."""
        return constant_bank(self.names, self.n_act, FLAT, ACTIONS)

    def rule(self, column, threshold, above, action, default):
        """One guarded arm: `action` where column > threshold (above) or <= it, else `default`."""
        bank = constant_bank(self.names, self.n_act, default, ACTIONS)
        law = np.zeros_like(bank["default"])
        law[-1, action] = 1.0
        bank["clauses"].append([[self.names.index(column), float(threshold), not above]])
        bank["laws"].append(law)
        return bank

    def rules(self, arms, default):
        """A tree of one-literal arms, first match wins: arms = [(column, threshold, above, action), ...]."""
        bank = constant_bank(self.names, self.n_act, default, ACTIONS)
        for column, threshold, above, action in arms:
            law = np.zeros_like(bank["default"])
            law[-1, action] = 1.0
            bank["clauses"].append([[self.names.index(column), float(threshold), not above]])
            bank["laws"].append(law)
        return bank

    def banks(self, bank):
        return (bank,)

    def on_adopt(self, bank):
        if self._recorder is not None:
            self._recorder(bank)

    # ---------------------------------------------------------------- pools and episodes
    def _build_pools(self, holdout_years):
        a, b = self._span
        pool = np.full(len(self._times), -1)
        pool[a:b] = 0
        if holdout_years:
            pool[(pool == 0) & np.isin(self._years, list(holdout_years))] = 1
        if self._check_years:
            pool[(pool == 0) & np.isin(self._years, self._check_years)] = 2
        self._pool = pool
        self._starts = self._pool_starts(0, self.T)
        self._starts_hold = self._pool_starts(1, self.T_hold) if holdout_years else self._starts
        self._starts_check = self._pool_starts(2, self.T_hold) if self._check_years else np.zeros(0, np.int64)

    def set_holdout(self, holdout_years):
        self.holdout = ",".join(str(y) for y in sorted(holdout_years)) if holdout_years else ""
        self._build_pools(holdout_years)
        if hasattr(self, "_starts_cache"):
            del self._starts_cache

    def _pool_starts(self, k, T):
        """Bars an episode of T bars can start on and stay inside pool k (it marks at s + T)."""
        inside = np.r_[0, np.cumsum(self._pool == k)]
        s = np.arange(len(self._pool) - T - 1)
        return s[inside[s + T + 1] - inside[s] == T + 1].astype(np.int64)

    def sample_starts(self, n, rng):
        src = self._starts_hold if (self.holdout and n >= self.accept_n) else self._starts
        return rng.choice(src, size=n, replace=True).astype(np.int64)

    def episode_length(self, starts, T=None):
        if self.holdout and len(starts) and self._pool[int(starts[0])] in (1, 2):
            return self.T_hold
        return self.T if self.holdout or T is None else int(T)

    def cluster_ids(self, starts):
        """The calendar quarter each episode starts in."""
        t = self._times[np.asarray(starts, np.int64)]
        return t.year.values * 4 + (t.month.values - 1) // 3

    def condition_groups(self, starts):
        """0: 2012-2014, 1: 2015-2017, 2: 2018 on -- a move that loses significantly over one
        stretch of history is rejected."""
        return np.searchsorted(self._blocks, np.asarray(starts), side="right")

    def check_scores(self, bank, *_):
        starts = self._starts_check.astype(np.int64)
        return self.rollout(bank, starts, self.T_hold)["G"]

    def seed_kernels(self, seed):
        pass

    def score_bank(self, bank, pol_fn, starts, T):
        return self.rollout(bank, starts, self.episode_length(starts, T))["G"]

    # ---------------------------------------------------------------- the critic
    CREDIT_H = 24                 # a row's credit: the instrument's next 24 bars, in volatility units
    CREDIT_TAIL = 0.05
    IC_FULL = 0.05                # hourly rank ICs are small; 0.05 gets the full prior

    def _credit(self, starts, T):
        """(E, T, N) log(mid[t + H] / mid[t]) / (vol sqrt(H / A)), never past the episode's end."""
        s = np.asarray(starts, np.int64)[:, None]
        t = s + np.arange(T)[None, :]
        t1 = np.minimum(t + self.CREDIT_H, s + T)
        r = np.log(self._mid[t1] / self._mid[t])
        v = np.maximum(self._vol[t], VOL_FLOOR) * np.sqrt(self.CREDIT_H / self.bars_per_year)
        return r / v

    def coverage_rows(self, bank, n_ep=80, seed=0):
        """On-policy rows, and the rows whose next day moved most (either way): where being
        long, flat or short mattered most."""
        rng = np.random.default_rng(seed)
        starts = self.sample_starts(n_ep, rng)
        out = self.run(bank, starts, self.T, trace=True)
        obs, ok = out["obs"], out["tick"]
        c = self._credit(starts, self.T)
        lo, hi = np.nanquantile(c[ok], [self.CREDIT_TAIL, 1 - self.CREDIT_TAIL])
        tails = ok & ((c <= lo) | (c >= hi))
        return obs[ok], obs[tails]

    def column_prior(self, bank, n_ep=60, seed=4711):
        """(n_columns,) in [0, 1]: each column's mean over instruments of its time-series rank
        IC with the credit on n_ep search episodes, |IC| / IC_FULL, clipped."""
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        starts = self.sample_starts(n_ep, rng)
        out = self.run(bank, starts, self.T, trace=True)
        obs, ok = out["obs"], out["tick"]
        c = self._credit(starts, self.T)
        W = obs.shape[-1]
        ic = np.zeros(W)
        n = 0
        for i in range(self._N):
            m = ok[:, :, i]
            if m.sum() < 200:
                continue
            X, y = obs[:, :, i][m], c[:, :, i][m]
            for j in range(W):
                if np.std(X[:, j]) > 0:
                    r = spearmanr(X[:, j], y)[0]
                    ic[j] += abs(r) if np.isfinite(r) else 0.0
            n += 1
        if n == 0:
            return None
        return np.clip(ic / n / self.IC_FULL, 0.0, 1.0)

    # ---------------------------------------------------------------- numba path
    def rollout(self, bank, starts, T, path=False, costs=True):
        """(E,) scores on the compiled kernel; `path` adds each bar's return and positions,
        `costs=False` trades at the mid with no commission (what the costs took; carry and
        the financing margin are kept)."""
        from btind.tick import flatten, tick_args
        fb = flatten(bank, len(self.names))
        if fb["has_kern"] or bank.get("mem"):
            if not costs:
                raise ValueError("costs=False needs the compiled kernel")
            return self.run(bank, starts, T, full=path)
        starts = np.asarray(starts, np.int64)
        E = len(starts)
        G, CG, DG, turn = np.empty(E), np.empty(E), np.empty(E), np.empty(E)
        R = np.empty((E, T)) if path else np.empty((0, 0))
        P = np.empty((E, T, self._N), np.int8) if path else np.empty((0, 0, 0), np.int8)
        if not costs and not hasattr(self, "_no_half"):
            self._no_half = np.zeros_like(self._half)
        _rollout(starts, int(T), self._X, self._avail, self._mid, self._last,
                 self._half if costs else self._no_half, self._carry,
                 self._vol, self._hours, self._decide, self._resize, COMMISSION if costs else 0.0, FIN_MARKUP,
                 self.risk_per_asset, VOL_FLOOR, self.risk_aversion, self.dd_weight, self.bars_per_year,
                 tick_args(fb), np.ascontiguousarray(fb["laws"]), G, CG, DG, turn, R, P, path)
        out = {"G": G, "cer": CG, "dd": DG, "turnover": turn}
        if path:
            out.update(ret=R, pos=P)
        return out

    def path(self, bank, start, end, costs=True):
        """One continuous run from the first bar >= start to the last bar of `end`'s day."""
        import pandas as pd
        a = int(np.searchsorted(self._times.values, np.datetime64(start)))
        b = int(np.searchsorted(self._times.values, np.datetime64(pd.Timestamp(end) + pd.Timedelta(days=1))))
        T = b - a - 1                                  # the run marks at bar a + T
        o = self.rollout(bank, np.array([a]), T, path=True, costs=costs)
        return dict(times=self._times[a + 1:a + 1 + T], ret=o["ret"][0], pos=o["pos"][0],
                    turnover=float(o["turnover"][0]), a=a, T=T)

    def hold_1x(self, name, start, end):
        """Per-bar excess return of holding `name` at 1x notional, financed (carry and margin),
        rebalanced every bar: the plain buy-and-hold reference."""
        import pandas as pd
        i = self.asset_names.index(name)
        a = int(np.searchsorted(self._times.values, np.datetime64(start)))
        b = int(np.searchsorted(self._times.values, np.datetime64(pd.Timestamp(end) + pd.Timedelta(days=1))))
        t = np.arange(a, b - 1)
        r = self._mid[t + 1, i] / self._mid[t, i] - 1.0
        return r + (self._carry[t, i] - FIN_MARKUP) * self._hours[t] / HOURS_PER_YEAR

    # ---------------------------------------------------------------- numpy reference
    def run(self, bank, starts, T, trace=False, full=False):
        from btind.memory import MemBank
        starts = np.asarray(starts, np.int64)
        E, N, Fx = len(starts), self._N, self._X.shape[2]
        pb = MemBank(bank, len(self.names))
        pb.reset(E * N)
        V = np.ones(E)
        peak = np.ones(E)
        e = np.zeros((E, N))
        state = np.zeros((E, N), int)
        held = np.zeros((E, N))
        entry = np.ones((E, N))
        ret = np.zeros((E, T))
        pos = np.zeros((E, T, N), np.int8)
        traded = np.zeros(E)
        obs_tr = np.zeros((E, T, N, len(self.names))) if trace else None
        tick_tr = np.zeros((E, T, N), bool) if trace else None
        rows = np.arange(E)
        for d in range(T):
            t = starts + d
            V0 = V.copy()
            cost = np.zeros(E)
            dec = self._decide[t]
            if dec.any():
                av = self._avail[t] & dec[:, None]
                vol = np.maximum(self._vol[t], VOL_FLOOR)
                pnl_z = np.where(state != 0, state * np.log(self._last[t] / entry)
                                 / (vol * np.sqrt(np.maximum(held, 1.0) / self.bars_per_year)), 0.0)
                gross = np.abs(e).sum(1) / V
                st = np.stack([state.astype(float), held / BARS_PER_WEEK, pnl_z,
                               np.broadcast_to((V / peak - 1.0)[:, None], (E, N)),
                               np.broadcast_to(gross[:, None], (E, N))], axis=2)
                obs = np.concatenate([self._X[t].astype(np.float64), st], axis=2)
                lat, stp = pb.latch.copy(), pb.step.copy()
                act = np.asarray(pb.act(obs.reshape(E * N, -1))).reshape(E, N)
                off = ~av.reshape(-1)                  # not evaluated: latch and step as they were
                pb.latch[off], pb.step[off] = lat[off], stp[off]
                if trace:
                    obs_tr[:, d] = obs
                    tick_tr[:, d] = av
                new = np.where(act == KEEP, state, act - 1)
                ch = av & (new != state)
                target = new * V[:, None] * self.risk_per_asset / vol
                dq = np.where(ch, np.abs(target - e), 0.0)
                cost += (dq * (self._half[t] + COMMISSION)).sum(1)
                traded += dq.sum(1) / V
                e = np.where(ch, target, e)
                state = np.where(ch, new, state)
                held = np.where(ch, 0.0, held)
                entry = np.where(ch, self._mid[t], entry)
            rz = self._resize[t]
            if rz.any():
                vol = np.maximum(self._vol[t], VOL_FLOOR)
                ok = rz[:, None] & self._avail[t] & (state != 0) & (held > 0)
                target = state * V[:, None] * self.risk_per_asset / vol
                dq = np.where(ok, np.abs(target - e), 0.0)
                cost += (dq * (self._half[t] + COMMISSION)).sum(1)
                traded += dq.sum(1) / V
                e = np.where(ok, target, e)
            V = V - cost
            r = self._mid[t + 1] / self._mid[t] - 1.0
            h = self._hours[t][:, None] / HOURS_PER_YEAR
            V = V + (e * r + (e * self._carry[t] - np.abs(e) * FIN_MARKUP) * h).sum(1)
            V = np.maximum(V, 1e-9)
            e = e * (1.0 + r)
            held = np.where(state != 0, held + 1.0, held)
            peak = np.maximum(peak, V)
            ret[:, d] = V / V0 - 1.0
            pos[:, d] = state
        x = ret
        cer = 100.0 * self.bars_per_year * (x.mean(1) - 0.5 * self.risk_aversion * x.var(1, ddof=1))
        dd = 100.0 * max_drawdown(x)
        out = {"G": cer - self.dd_weight * dd, "cer": cer, "dd": dd, "turnover": traded}
        if trace:
            out.update(obs=obs_tr, tick=tick_tr)
        if full:
            out.update(ret=ret, pos=pos)
        return out


@njit(cache=True, parallel=True)
def _rollout(starts, T, X, avail, mid, last, half, carry, vol_, hours, decide, resize, comm, markup,
             risk, vol_floor, gamma, dd_w, A, tb, laws, G, CG, DG, turnover, R, P, want_path):
    E = starts.shape[0]
    Tall, N, F = X.shape
    W = F + 5                                    # observation width: features + the state columns
    for ep in prange(E):
        s = starts[ep]
        V = 1.0
        peak = 1.0
        e = np.zeros(N)
        state = np.zeros(N, np.int64)
        held = np.zeros(N)
        entry = np.ones(N)
        latch = np.full(N, -1, np.int64)
        step = np.zeros(N, np.int64)
        z = np.zeros(W + 2)                      # obs, V_hat, leverage (btind's layout)
        x = np.zeros(T)
        traded = 0.0
        for d in range(T):
            t = s + d
            V0 = V
            cost = 0.0
            if decide[t]:
                gross = 0.0
                for i in range(N):
                    gross += abs(e[i])
                gross /= V
                dd = V / peak - 1.0
                for i in range(N):
                    if not avail[t, i]:
                        continue
                    vol = max(vol_[t, i], vol_floor)
                    for j in range(F):
                        z[j] = X[t, i, j]
                    z[F] = float(state[i])
                    z[F + 1] = held[i] / 120.0
                    if state[i] != 0:
                        z[F + 2] = state[i] * np.log(last[t, i] / entry[i]) / (vol * np.sqrt(max(held[i], 1.0) / A))
                    else:
                        z[F + 2] = 0.0
                    z[F + 3] = dd
                    z[F + 4] = gross
                    law, latch[i], step[i] = _tick_tree(z, latch[i], step[i], tb)
                    a = _argmax_law(z, W + 2, laws, law)
                    new = state[i] if a == 3 else a - 1
                    if new != state[i]:
                        target = new * V * risk / vol
                        dq = abs(target - e[i])
                        cost += dq * (half[t, i] + comm)
                        traded += dq / V
                        e[i] = target
                        state[i] = new
                        held[i] = 0.0
                        entry[i] = mid[t, i]
            if resize[t]:
                for i in range(N):
                    if avail[t, i] and state[i] != 0 and held[i] > 0:
                        target = state[i] * V * risk / max(vol_[t, i], vol_floor)
                        dq = abs(target - e[i])
                        cost += dq * (half[t, i] + comm)
                        traded += dq / V
                        e[i] = target
            V -= cost
            h = hours[t] / 8760.0
            pnl = 0.0
            for i in range(N):
                if e[i] != 0.0:
                    r = mid[t + 1, i] / mid[t, i] - 1.0
                    pnl += e[i] * r + (e[i] * carry[t, i] - abs(e[i]) * markup) * h
                    e[i] *= 1.0 + r
                if state[i] != 0:
                    held[i] += 1.0
            V = max(V + pnl, 1e-9)
            if V > peak:
                peak = V
            x[d] = V / V0 - 1.0
            if want_path:
                R[ep, d] = x[d]
                for i in range(N):
                    P[ep, d, i] = state[i]
        m = 0.0
        for d in range(T):
            m += x[d]
        m /= T
        v = 0.0
        for d in range(T):
            v += (x[d] - m) ** 2
        v /= T - 1
        cer = 100.0 * A * (m - 0.5 * gamma * v)
        w, pk, mdd = 1.0, 1.0, 0.0
        for d in range(T):
            w *= 1.0 + x[d]
            pk = max(pk, w)
            mdd = max(mdd, 1.0 - w / pk)
        CG[ep] = cer
        DG[ep] = 100.0 * mdd
        G[ep] = cer - dd_w * 100.0 * mdd
        turnover[ep] = traded
