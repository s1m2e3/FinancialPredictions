"""The scoring head's numba rollout against its numpy reference (portfolio_score.py), on random trees.

Both simulate the same trades from the same trees; they may differ only where a floating-point
sum in a different order moves a share count across an integer boundary (whole shares) or a
holding across the no-trade band's edge. Run it before trusting a search with the scoring head.

Run from the repository root:  python checks/verify_score_kernel.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import portfolio_bt as pb                                     # noqa: E402
from portfolio_env import EXPOSURE_ACTIONS, EXPOSURE_LEVELS  # noqa: E402
from portfolio_score import score_bank, score_world          # noqa: E402


def random_tree(env, names, rng, scalar, n_arms=4):
    """Random guards on random columns at in-sample quantiles, random affine laws."""
    from btind.memory import mem_names
    d = len(mem_names(names, None)) + 1
    rows = (env._X[env._starts[:200]].reshape(-1, env._X.shape[2]) if names is env.stock_names
            else env._M[env._starts[:2000]])
    clauses = []
    for _ in range(n_arms):
        lits = []
        for _ in range(int(rng.integers(1, 3))):
            j = int(rng.integers(0, rows.shape[1]))
            lits.append([j, float(np.quantile(rows[:, j], rng.uniform(0.2, 0.8))), bool(rng.integers(0, 2))])
        clauses.append(lits)
    if scalar:
        b = score_bank(names)
        # scores spread around the clip range, so clipping, the minimum and the band all act
        b.update(clauses=clauses, laws=[rng.normal(0, 0.3, (d, 1)) + np.r_[np.zeros(d - 1), 0.5][:, None]
                                        for _ in clauses], default=rng.normal(0, 0.3, (d, 1)))
        return b
    n = len(EXPOSURE_LEVELS)
    return dict(clauses=clauses, laws=[rng.normal(0, 1, (d, n)) for _ in clauses], default=rng.normal(0, 1, (d, n)),
                names=list(names), laws_on_z=True, head="argmax", n_act=n, actions=list(EXPOSURE_ACTIONS))


panel = pb.load_panel()
F, M, sn, mn = pb.load_features()
env = score_world(panel, F, M, sn, mn, pb.TRAIN_START, pb.TRAIN_END, T=189)
rng = np.random.default_rng(0)
starts = env.sample_starts(40, rng)
env.rollout(score_bank(env.stock_names), env.default_bank("exposure"), starts[:2], env.T)   # compile
# (fractional, max_weight, outside cap, band, dd_weight, decide_every, size-weighted)
SETTINGS = [(True, 1 / 3, 0.10, 0.25, 0.0, 21, True), (True, 1 / 3, 0.10, 0.0, 0.5, 21, True),
            (False, 1 / 3, 0.10, 0.25, 0.0, 21, True), (True, 1.0, 1.0, 0.25, 1.0, 5, False),
            (False, 1 / 3, 0.05, 0.10, 0.0, 5, False)]
SIZE = env._size.copy()
worst, t_ref, t_ker = 0.0, 0.0, 0.0
for trial in range(2 * len(SETTINGS)):
    env.objective = ("cer", "sharpe")[trial % 2]
    env.fractional, env.max_weight, out_w, env.score_band, env.dd_weight, env.decide_every, by_size = SETTINGS[trial // 2]
    env._size = SIZE if by_size else np.zeros((0, 0))
    env._cap_scale = np.ascontiguousarray(np.where(panel.sp500, 1.0, min(1.0, out_w / env.max_weight)), dtype=np.float64)
    sb = random_tree(env, env.stock_names, rng, scalar=True)
    eb = random_tree(env, env.exposure_names, rng, scalar=False)
    t = time.perf_counter(); r = env.run(sb, eb, starts, env.T); t_ref += time.perf_counter() - t
    t = time.perf_counter(); k = env.rollout(sb, eb, starts, env.T); t_ker += time.perf_counter() - t
    diff = np.maximum(np.abs(r["G"] - k["G"]), np.abs(r["dd_gap"] - k["dd_gap"]))
    worst = max(worst, diff.max())
    print(f"trial {trial:2d} ({env.objective}, {'fractional' if env.fractional else 'whole'} shares, cap "
          f"{env.max_weight:.2f}, outside {out_w:.2f}, band {env.score_band:.2f}, dd {env.dd_weight:g}, every "
          f"{env.decide_every}d{', by size' if by_size else ''}): G reference {r['G'].mean():+.4f} kernel "
          f"{k['G'].mean():+.4f}  max |diff| {diff.max():.2e}  episodes differing > 1e-9: {(diff > 1e-9).sum()}/{len(diff)}"
          f"  turnover {k['turnover'].mean():.2f}")
print(f"worst difference {worst:.2e}; time per 40-episode score: reference {t_ref / (2 * len(SETTINGS)):.2f}s, "
      f"kernel {t_ker / (2 * len(SETTINGS)):.3f}s")
