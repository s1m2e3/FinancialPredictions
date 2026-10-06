"""Run portfolio_bt.py at several drawdown weights, one after another, then elite.py.

Each run resumes from its own checkpoint (results/portfolio_bt/dd<W>/run/), so stopping
the sweep and starting the same command again loses at most the round in progress; a
weight whose run is already complete only rewrites its report. Every run writes its log
to results/portfolio_bt/dd<W>/train_log.txt and results/portfolio_bt/train_log.txt while
printing it here.

Works in any shell (cmd, PowerShell, bash), from the repository root:
    python sweep.py                    the default weights 0 0.25 0.5 1, 3 rounds
    python sweep.py 0.25 0.5 1         only these weights
    python sweep.py 0.5 --rounds 2     other options are passed on to portfolio_bt.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = ["0", "0.25", "0.5", "1"]
VALUE_FLAGS = ("--every", "--seed", "--max-stocks", "--objective", "--core-cap", "--core-n")


def main(argv):
    rounds = "3"
    if "--rounds" in argv:
        i = argv.index("--rounds")
        rounds = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    # flags that take a value travel as pairs (--every 5, --seed none): their values are not weights
    extra, rest, i = [], [], 0
    while i < len(argv):
        if argv[i] in VALUE_FLAGS and i + 1 < len(argv):
            extra += argv[i:i + 2]
            i += 2
        else:
            (extra if argv[i].startswith("--") else rest).append(argv[i])
            i += 1
    weights = rest or DEFAULT_WEIGHTS
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    failed = []
    for w in weights:
        print(f"\n##### [{stamp()}] sweep: drawdown weight {w}", flush=True)
        code = subprocess.call([sys.executable, "-u", "portfolio_bt.py", rounds, "--dd", w] + extra, cwd=ROOT)
        if code != 0:
            failed.append(w)
            print(f"##### [{stamp()}] weight {w} exited with code {code}; going on", flush=True)
    print(f"\n##### [{stamp()}] sweep: elite.py", flush=True)
    subprocess.call([sys.executable, "-u", "elite.py"] + extra, cwd=ROOT)     # the same world's runs only
    print(f"##### [{stamp()}] sweep done" + (f"; failed weights: {failed}" if failed else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
