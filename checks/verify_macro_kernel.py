"""The hourly world's compiled rollout against its numpy reference (macro_env.py): random trees
-- guards on features and on the portfolio state, random laws -- on hourly and daily decisions,
must give the same score, CER, drawdown and turnover on every episode.

    python checks/verify_macro_kernel.py            synthetic bars (no data needed)
    python checks/verify_macro_kernel.py --real     the Dukascopy panel (macro_data.build)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import macro_data as md
from macro_env import ACTIONS, MacroWorld


def random_bank(env, rng, n_arms=4):
    from btind.memory import mem_names
    from portfolio_env import constant_bank
    d = len(mem_names(env.names, None)) + 1
    s = env._starts[:300]
    feat = env._X[s].reshape(-1, env._X.shape[2]).astype(np.float64)
    clauses = []
    for _ in range(n_arms):
        cl = []
        for _ in range(rng.integers(1, 3)):
            j = int(rng.integers(0, len(env.names)))
            if j < feat.shape[1]:
                thr = float(np.quantile(feat[:, j], rng.uniform(0.2, 0.8)))
            else:                                   # a state column: position, weeks held, pnl, drawdown, gross
                thr = float(rng.choice([-0.5, 0.5, 0.05, 0.0, -0.01, 0.5]))
            cl.append((j, thr, bool(rng.integers(0, 2))))
        clauses.append(cl)
    bank = constant_bank(env.names, len(ACTIONS), 1, ACTIONS)
    bank.update(clauses=clauses, laws=[rng.normal(0, 1, (d, len(ACTIONS))) for _ in clauses],
                default=rng.normal(0, 1, (d, len(ACTIONS))))
    return bank


def main():
    real = "--real" in sys.argv
    if real:
        X, sim = md.features(md.build())
        start, end, hold = md.TRAIN_START, md.TRAIN_END, [2013, 2016, 2019]
    else:
        X, sim = md.synthetic()
        start, end, hold = "2012-01-01", "2014-12-31", [2013]
    worst = 0.0
    rng = np.random.default_rng(5)
    for decide in ("hourly", "daily"):
        env = MacroWorld(X, sim, start, end, T=240 if decide == "hourly" else 720,
                         T_hold=240 if decide == "hourly" else 720, decide=decide, holdout_years=hold)
        starts = env.sample_starts(40, rng)
        for trial in range(5):
            bank = env.default_bank() if trial == 0 else random_bank(env, rng)
            r, k = env.run(bank, starts, env.T), env.rollout(bank, starts, env.T)
            diff = max(np.abs(r[key] - k[key]).max() for key in ("G", "cer", "dd", "turnover"))
            worst = max(worst, diff)
            print(f"{decide:6s} trial {trial} ({'all flat' if trial == 0 else 'random tree'}): "
                  f"mean G ref {r['G'].mean():+9.4f} kernel {k['G'].mean():+9.4f}  "
                  f"turnover {k['turnover'].mean():7.2f}  max |diff| {diff:.2e}", flush=True)
    print(f"worst difference {worst:.2e}  ->  {'OK' if worst < 1e-8 else 'MISMATCH'}")
    return 0 if worst < 1e-8 else 1


if __name__ == "__main__":
    sys.exit(main())
