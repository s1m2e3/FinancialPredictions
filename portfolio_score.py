"""The scoring head: the stock tree SCORES each stock and the portfolio is weighted by score.

portfolio_bt.py's stock tree answers exit / hold / buy for every stock on its own, and the
allocation then splits the money over the "buy" rows. Stock selection is a RANKING problem --
which of these ~116 names deserve more money than the others -- and the ranks among the
inputs were a workaround for a head that cannot express it. Here:

    score_i   = clip(law(x_i), 0, 1), from btind's continuous ("scalar") head: the tree's
                arms still choose WHICH law applies to a stock, the law's affine output IS
                the score; below SCORE_MIN a stock is not held
    weight_i  proportional to score_i x company size (1 / sig_21d without sizes), no stock
                above its cap (a third of the invested money; a tenth for the outside pool),
                water-filled exactly as portfolio_env's allocation
    band      a held stock whose target is within SCORE_BAND of its current value is left
                alone (no trade), so scores that wobble month to month do not churn the book
    exposure  the exposure tree and its 0-100% levels are unchanged; its `buy_frac` input
                is now the stock tree's MEAN score over the tradable stocks (how much it likes
                the cross-section)

A constant score is "buy all by size", so every search starts from the benchmark to beat.

TWO IMPLEMENTATIONS, as portfolio_env: `run` in numpy with btind's MemBank is the reference,
`_score_rollout` the numba kernel the search uses; checks/verify_score_kernel.py holds them to
agreement. Run that check before trusting a search here.

    python portfolio_score.py [rounds] [--dd W]   search both trees with the scoring head
                                                  (resumable) and write
                                                  results/portfolio_score/dd<W>/report.md
"""
import json
import os
import sys

import numpy as np
from numba import njit, prange

import portfolio_bt as pb
from portfolio_env import (MIN_POSITION, OBJECTIVES, PortfolioWorld, _capped_split, _tick_tree,
                           max_drawdown)

SCORE_RANGE = (0.0, 1.0)
SCORE_MIN = 0.05                 # below this a stock is not held
SCORE_BAND = 0.25                # no trade while a holding is within 25% of its target
OUT = os.path.join(pb.ROOT, "results", "portfolio_score", pb.RUN_TAG)


def score_bank(names, value=0.5):
    """A stock tree with no arms that gives every stock the same score: buy all by size."""
    from btind.memory import mem_names
    th = np.zeros((len(mem_names(names, None)) + 1, 1))
    th[-1, 0] = value
    return dict(clauses=[], laws=[], default=th, names=list(names), laws_on_z=True, head="scalar",
                n_act=1, actions=["score"], u_range=SCORE_RANGE)


def score_rule(names, column, threshold, above=True, hi=1.0, lo=0.0):
    """Score `hi` where column > threshold (above) or <= it, else `lo`."""
    b = score_bank(names, lo)
    law = np.zeros_like(b["default"])
    law[-1, 0] = hi
    b["clauses"].append([[names.index(column), float(threshold), not above]])
    b["laws"].append(law)
    return b


class ScoreWorld(PortfolioWorld):
    u_range = SCORE_RANGE
    u_null = 0.5

    def __init__(self, *args, score_min=SCORE_MIN, score_band=SCORE_BAND, **kwargs):
        super().__init__(*args, **kwargs)
        # scalars: part of the world's signature in btind's store
        self.stock_head = "score"
        self.score_min, self.score_band = float(score_min), float(score_band)

    # the head depends on which tree is searched: btind reads it when a stage starts
    @property
    def head(self):
        return "scalar" if self.agent == "stocks" else "argmax"

    @property
    def n_act(self):
        return 1 if self.agent == "stocks" else super().n_act

    @property
    def actions(self):
        return ["score"] if self.agent == "stocks" else super().actions

    def default_bank(self, agent):
        return score_bank(self.stock_names) if agent == "stocks" else super().default_bank(agent)

    # ------------------------------------------------------------ numba path
    def rollout(self, stock_bank, exposure_bank, starts, T, daily=False):
        from btind.tick import flatten, tick_args
        fs = flatten(stock_bank, len(self.stock_names))
        fe = flatten(exposure_bank, len(self.exposure_names))
        if fs["has_kern"] or fe["has_kern"] or (stock_bank.get("mem") or exposure_bank.get("mem")):
            return self.run(stock_bank, exposure_bank, starts, T, full=daily)
        starts = np.asarray(starts, np.int64)
        E = len(starts)
        G, CG, DG, turn = np.empty(E), np.empty(E), np.empty(E), np.empty(E)
        D = np.empty((E, T)) if daily else np.empty((0, 0))
        lo, hi = SCORE_RANGE
        _score_rollout(starts, int(T), self.decide_every, self._X, self._M, self._open, self._avail,
                       self._sig, self._rf, self._infl, self._spx_ret, self.budget, self.cost_bps * 1e-4,
                       self._levels, OBJECTIVES[self.objective], self.risk_aversion, self.fractional,
                       self.max_weight, self._cap_scale, self.dd_weight, self._size, lo, hi,
                       self.score_min, self.score_band,
                       tick_args(fs), np.ascontiguousarray(fs["laws"]),
                       tick_args(fe), np.ascontiguousarray(fe["laws"]), G, CG, DG, D, turn, daily)
        out = {"G": G, "turnover": turn, "cer_gap": CG, "dd_gap": DG}
        if daily:
            days = starts[:, None] + np.arange(T)[None, :]
            out.update(daily=D, bench=self._spx_ret[days], rf=self._rf[days], infl=self._infl[days])
        return out

    # ------------------------------------------------------------ numpy reference
    def run(self, stock_bank, exposure_bank, starts, T, trace=False, full=False):
        from btind.memory import MemBank
        starts = np.asarray(starts, np.int64)
        E, N, cost = len(starts), self._N, self.cost_bps * 1e-4
        lo, hi = SCORE_RANGE
        ps = MemBank(stock_bank, len(self.stock_names))
        pe = MemBank(exposure_bank, len(self.exposure_names))
        ps.reset(E * N)
        pe.reset(E)
        cash, peak = np.full(E, self.budget), np.full(E, self.budget)
        shares, days_held = np.zeros((E, N)), np.zeros((E, N))
        n_dec = int(np.ceil(T / self.decide_every))
        daily_p, traded = np.zeros((E, T)), np.zeros(E)
        f_prev, f_days = np.full(E, -1), np.zeros(E)
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
            sc = np.asarray(ps.act(obs.reshape(E * N, -1)), float).reshape(E, N)
            off = ~avail.reshape(-1)
            ps.latch[off], ps.step[off] = -1, 0
            sc = np.clip(sc, lo, hi)
            sc = np.where(avail & (sc >= self.score_min), sc, 0.0)
            obs_e = np.column_stack([self._M[t], cash / V, n_held / n_av, V / peak - 1.0,
                                     (V - cash) / V, sc.sum(1) / n_av, f_days / 252.0])
            f_idx = np.asarray(pe.act(obs_e))
            f_days = np.where(f_idx == f_prev, f_days + self.decide_every, 0.0)
            f_prev = f_idx
            f = self._levels[f_idx]
            if trace:
                obs_tr.append(obs)
                obse_tr.append(obs_e)
                avail_tr.append(avail)
            # --- allocation: every position re-targeted by score x size, capped, banded
            S = f * V
            cap = self.max_weight * S[:, None] * self._cap_scale[t]
            if self._size.size:
                base = np.maximum(self._size[t], 0.0)
            else:
                base = 1.0 / np.maximum(self._X[t][:, :, self._sig].astype(np.float64), 1e-6)
            alloc = _capped_split(S, np.where(sc > 0, sc * base, 0.0), cap)
            pxn = np.nan_to_num(px)
            px0 = np.where(np.isnan(px), np.inf, px)
            keep = (shares > 0) & (alloc > 0) & (np.abs(shares * pxn - alloc) <= self.score_band * alloc)
            nv = np.where(alloc > 0, np.where(keep, shares, alloc / (px0 * (1 + 2 * cost))), 0.0)
            tot = (nv * pxn).sum(1)
            scale = np.where(tot > S, S / np.where(tot > 0, tot, 1.0), 1.0)
            new = nv * scale[:, None]
            if not self.fractional:
                new = np.floor(new)
            new = np.where(new * pxn < MIN_POSITION, 0.0, new)
            trade = (np.abs(new - shares) * pxn).sum(1)
            traded += trade / V
            cash = V - (new * pxn).sum(1) - trade * cost
            days_held = np.where(new > 0, np.where(held, days_held + self.decide_every, 0.0), 0.0)
            shares = new
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
        else:
            sharpe = lambda x: np.sqrt(252.0) * x.mean(1) / np.maximum(x.std(1, ddof=1), 1e-12)
            gap = sharpe(daily_p - rf) - sharpe(bench - rf)
        dd_gap = 100.0 * (max_drawdown(daily_p) - max_drawdown(bench))
        out = {"G": gap - self.dd_weight * dd_gap, "turnover": traded, "cer_gap": gap, "dd_gap": dd_gap}
        if trace:
            out.update(obs=np.stack(obs_tr, 1), obs_e=np.stack(obse_tr, 1), avail=np.stack(avail_tr, 1),
                       decision_ret=dec_ret)
        if full:
            out.update(daily=daily_p, bench=bench, rf=rf, infl=infl)
        return out


@njit(cache=True, inline="always")
def _scalar_law(z, width, laws, law):
    """[z, 1] @ laws[law][:, 0]: the continuous head's command."""
    v = laws[law, width, 0]
    for j in range(width):
        v += z[j] * laws[law, j, 0]
    return v


@njit(cache=True, inline="always")
def _argmax_law(z, width, laws, law):
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
def _score_rollout(starts, T, de, X, M, open_, avail, sig_col, rf, infl, spx_ret, budget, cost, levels,
                   objective, gamma, frac, max_w, cscale, dd_w, size, lo, hi, s_min, band,
                   st, s_laws, et, e_laws, G, CG, DG, D, turnover, want_daily):
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
        score = np.zeros(N)
        inv = np.zeros(N)
        newv = np.zeros(N)
        capped = np.zeros(N, np.bool_)
        z = np.zeros(F + 8)
        ze = np.zeros(Mw + 8)
        daily = np.zeros(T)
        traded = 0.0
        f_prev, f_days = -1, 0.0
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
            # --- stock tree: a score per TRADABLE stock; the rest score 0 and forget
            ssum = 0.0
            for i in range(N):
                if not avail[t, i]:
                    score[i] = 0.0
                    latch_s[i] = -1
                    step_s[i] = 0
                    continue
                for j in range(F):
                    z[j] = X[t, i, j]
                z[F] = 1.0 if shares[i] > 0 else 0.0
                z[F + 1] = shares[i] * open_[t, i] / V if shares[i] > 0 else 0.0
                z[F + 2] = days_held[i] / 252.0
                z[F + 3] = cash / V
                z[F + 4] = n_held / n_av
                z[F + 5] = dd
                law, latch_s[i], step_s[i] = _tick_tree(z, latch_s[i], step_s[i], st)
                v = _scalar_law(z, F + 8, s_laws, law)
                v = min(max(v, lo), hi)
                if v < s_min:
                    v = 0.0
                score[i] = v
                ssum += v
            # --- exposure tree, one row per portfolio
            for j in range(Mw):
                ze[j] = M[t, j]
            ze[Mw] = cash / V
            ze[Mw + 1] = n_held / n_av
            ze[Mw + 2] = dd
            ze[Mw + 3] = (V - cash) / V
            ze[Mw + 4] = ssum / n_av
            ze[Mw + 5] = f_days / 252.0
            law, latch_e, step_e = _tick_tree(ze, latch_e, step_e, et)
            f_idx = _argmax_law(ze, Mw + 8, e_laws, law)
            f_days = f_days + de if f_idx == f_prev else 0.0
            f_prev = f_idx
            f = levels[f_idx]
            # --- allocation: targets by score x size, water-filled under per-stock caps
            S = f * V
            cap = max_w * S
            for i in range(N):
                if score[i] > 0.0:
                    b = max(size[t, i], 0.0) if size.shape[0] > 0 else 1.0 / max(X[t, i, sig_col], 1e-6)
                    inv[i] = score[i] * b
                else:
                    inv[i] = 0.0
                capped[i] = False
            rem = S
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
            # --- the no-trade band, then everything scaled to fit the invested budget
            tot = 0.0
            for i in range(N):
                px = open_[t, i]
                if capped[i]:
                    a = cap * cscale[t, i]
                elif inv[i] > 0 and wsum > 0.0:
                    a = rem * inv[i] / wsum
                else:
                    a = 0.0
                if a <= 0.0:
                    nv = 0.0
                elif shares[i] > 0 and abs(shares[i] * px - a) <= band * a:
                    nv = shares[i]
                else:
                    nv = a / (px * (1 + 2 * cost))
                newv[i] = nv
                if nv > 0.0:                    # a stock with no price yet has a NaN open: 0 x NaN
                    tot += nv * px              # would make the total NaN and skip the scaling
            scale = S / tot if tot > S else 1.0
            spent = 0.0
            trade = 0.0
            for i in range(N):
                px = open_[t, i]
                new = newv[i] * scale
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
        # --- score: portfolio minus S&P 500, as portfolio_env's kernel
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
            gap = 100.0 * 252.0 * ((mp - 0.5 * gamma * vp) - (mb - 0.5 * gamma * vb))
        else:
            gap = np.sqrt(252.0) * (mp / max(np.sqrt(vp), 1e-12) - mb / max(np.sqrt(vb), 1e-12))
        wp, wb, pp, pb_, ddp, ddb = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
        for d in range(T):
            wp *= 1.0 + daily[d]
            wb *= 1.0 + spx_ret[s + d]
            pp = max(pp, wp)
            pb_ = max(pb_, wb)
            ddp = max(ddp, 1.0 - wp / pp)
            ddb = max(ddb, 1.0 - wb / pb_)
        dd_gap = 100.0 * (ddp - ddb)
        CG[e] = gap
        DG[e] = dd_gap
        G[e] = gap - dd_w * dd_gap
        turnover[e] = traded
        if want_daily:
            for d in range(T):
                D[e, d] = daily[d]


def score_world(panel, F, M, sn, mn, start, end, T=252, agent="stocks", gamma=None, holdout=None, check=None):
    """portfolio_bt.world with the scoring head."""
    gamma = pb.calibrate_risk_aversion(panel, pb.TRAIN_START, pb.TRAIN_END) if gamma is None else gamma
    by_size = dict(size=pb.company_size(panel), weighting="company size") if pb.SIZE_WEIGHTS else {}
    return ScoreWorld(panel, F, M, sn, mn, start, end, T=T, agent=agent, risk_aversion=gamma,
                      fractional=True, max_weight=pb.MAX_WEIGHT, outside_max_weight=pb.OUTSIDE_MAX_WEIGHT,
                      holdout_years=holdout, T_hold=pb.T_HOLD, accept_n=pb.ACCEPT_N,
                      decide_every=pb.DECIDE_EVERY, dd_weight=pb.DD_WEIGHT, check_years=check,
                      source="yfinance-sp500top100+ndx40+edgar-longh-regime-young+score", **by_size)


def main(rounds):
    import pandas as pd
    import bt_eval as ev
    from btind.runlog import bank_json
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    env = score_world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=pb.CFG["T"],
                      holdout=pb.holdout_years(), check=pb.CHECK_YEARS)
    banks, state = ev.run_stages(env, os.path.join(OUT, "run"), rounds=rounds)
    os.makedirs(OUT, exist_ok=True)
    for name, b, names in (("stock_tree", banks["stocks"], env.stock_names),
                           ("exposure_tree", banks["exposure"], env.exposure_names)):
        with open(os.path.join(OUT, name + ".json"), "w") as fh:
            json.dump(bank_json(b, names), fh, indent=1)
    full = env.default_bank("exposure")
    pairs = {"buy all (by size)": (score_bank(env.stock_names), full),
             "EBTDA score (top 30%: 1, else 0)": (score_rule(env.stock_names, "rank_ebtda_roa_3y", 0.70), full),
             "score tree, 100% invested": (banks["stocks"], full),
             "score tree + exposure tree": (banks["stocks"], banks["exposure"])}
    g = env.risk_aversion
    lines = ["# Scoring head", "", __doc__.split("TWO IMPLEMENTATIONS")[0].strip(), "",
             "| stage | tree | arms | held-out | check kept | reverted |", "|---|---|---|---|---|---|"]
    lines += [f"| {s['stage']} | {s['agent']} | {s['arms']} | {s['held_out']:+.2f} +- {s['ci']:.2f} | "
              f"{s['check']['kept_G']:+.2f} | {s['check']['reverted']} |" for s in state["stages"]]
    for label, (a, b) in pb.PERIODS.items():
        paths = ev.period_paths(panel, F, M, sn, mn, pairs, a, b, g, world_fn=score_world)
        rows = {}
        for k, v in paths.items():
            d_b, p_b = ev.paired(v["real"], paths["buy all (by size)"]["real"], g) if k != "buy all (by size)" else (0.0, np.nan)
            d_s, p_s = ev.paired(v["real"], paths["S&P 500"]["real"], g) if k != "S&P 500" else (0.0, np.nan)
            rows[k] = {"real %/yr": 100 * (np.prod(1 + v["real"]) ** (252 / len(v["real"])) - 1),
                       "vol %": 100 * np.std(v["daily"]) * np.sqrt(252), "max DD %": 100 * ev.max_drawdown(v["daily"]),
                       "CER %/yr": pb.cer(v["real"], g), "vs buy all": d_b, "p": p_b, "vs S&P": d_s, "p ": p_s}
        df = pd.DataFrame(rows).T.round(3)
        print(f"\n{label}\n" + df.to_string(), flush=True)
        lines += ["", f"## {label}", "", "```", df.to_string(), "```"]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nwrote", os.path.join(OUT, "report.md"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != pb._flag_value("--dd")]
    main(int(args[0]) if args else 3)
