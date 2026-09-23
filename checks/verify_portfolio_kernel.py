"""The numba portfolio rollout against the numpy reference (btind's MemBank), on random trees.

Both simulate the same trades from the same trees; they may differ only where a floating-
point sum in a different order moves a share count across an integer boundary.

Run from the repository root:  python checks/verify_portfolio_kernel.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio_env import PortfolioWorld, constant_bank     # noqa: E402
from stock_features import load                               # noqa: E402
from stocks_data import TRAIN_END, TRAIN_START, load_panel   # noqa: E402


def random_bank(env, names, n_act, rng, n_arms=4):
    """Random guards on random columns at in-sample quantiles, random affine laws."""
    from btind.memory import mem_names
    d = len(mem_names(names, None)) + 1
    rows = env._X[env._starts[:200]].reshape(-1, env._X.shape[2]) if names is env.stock_names else env._M[env._starts[:2000]]
    clauses = []
    for _ in range(n_arms):
        cl = []
        for _ in range(rng.integers(1, 3)):
            j = int(rng.integers(0, rows.shape[1]))
            cl.append((j, float(np.quantile(rows[:, j], rng.uniform(0.2, 0.8))), bool(rng.integers(0, 2))))
        clauses.append(cl)
    bank = constant_bank(names, n_act, 0, [str(a) for a in range(n_act)])
    bank.update(clauses=clauses, laws=[rng.normal(0, 1, (d, n_act)) for _ in clauses],
                default=rng.normal(0, 1, (d, n_act)))
    return bank


panel = load_panel()
F, M, sn, mn = load(fundamentals=True)
env = PortfolioWorld(panel, F, M, sn, mn, TRAIN_START, TRAIN_END)
print(f"risk aversion calibrated on {TRAIN_START[:4]}-{TRAIN_END[:4]}: gamma = {env.risk_aversion:.3f}")
rng = np.random.default_rng(0)
starts = env.sample_starts(40, rng)
worst = 0.0
env.rollout(env.default_bank("stocks"), env.default_bank("exposure"), starts[:2], env.T)   # compile
t_ref = t_ker = 0.0
SETTINGS = [(False, 1.0), (True, 1.0), (True, 1 / 3), (False, 1 / 3)]   # (fractional, max_weight)
for trial in range(2 * len(SETTINGS)):
    env.objective = ("cer", "sharpe")[trial % 2]         # both scores, same trades
    env.fractional, env.max_weight = SETTINGS[trial // 2]
    sb = random_bank(env, env.stock_names, 3, rng)
    eb = random_bank(env, env.exposure_names, 4, rng)
    t = time.perf_counter(); ref = env.run(sb, eb, starts, env.T)["G"]; t_ref += time.perf_counter() - t
    t = time.perf_counter(); ker = env.rollout(sb, eb, starts, env.T)["G"]; t_ker += time.perf_counter() - t
    diff = np.abs(ref - ker)
    worst = max(worst, diff.max())
    print(f"trial {trial} ({env.objective}, {'fractional' if env.fractional else 'whole'} shares, "
          f"cap {env.max_weight:.2f}): mean G reference {ref.mean():+.4f} kernel {ker.mean():+.4f}  "
          f"max |diff| {diff.max():.2e}  episodes differing > 1e-9: {(diff > 1e-9).sum()}/{len(diff)}")
print(f"worst difference {worst:.2e}; time per 40-episode score: reference {t_ref / (2 * len(SETTINGS)):.2f}s, "
      f"kernel {t_ker / (2 * len(SETTINGS)):.3f}s")
