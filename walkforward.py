"""Walk-forward: judge the PROCEDURE, not one tree.

portfolio_bt.py grows one pair of trees on 2006-2019 and scores it on 2020-2026, a single
out-of-sample draw. Here the same search is rerun every year on everything known by then and
the trees it produces trade only the NEXT year:

    for Y in 2014 ... 2026:
        train on 2006 .. Y-1   the two years before Y are that window's CHECK years; the rest,
                               in consecutive two-year blocks from 2006, alternate between the
                               held-out folds A and B (a trailing single year joins the last
                               block's fold); the risk aversion is calibrated on the window
        trade Y                the window's trees, next to buy all (by size), the EBTDA rule
                               and the S&P 500, from $100k on Jan 1 (every portfolio restarts
                               each year, so the yearly reset costs them all alike)

The years chained together are a track record every year of which was out of sample for the
trees that traded it: 2014-2026, 13 years, against the one validation window of the fixed
split. The paired test against buy-all uses 126-day blocks.

WHAT IS STILL NOT CLEAN. The features, the universe rules and the search settings were chosen
while looking at 2020-2026 results; walk-forward removes the fitted trees' look-ahead, not the
designer's. Only data after 2026-09-19 is untouched by both. (Every window's search starts
from buy all rather than from the EBTDA rule, which was picked after seeing 2020-2026; the
EBTDA rule is still traded alongside, as a comparison.)

    python walkforward.py [--dd W] [--from 2014]    run (resumable: finished years are kept)
    python walkforward.py --summary [--dd W]        the chained track record and its tests

Each year is one four-stage search (the early windows are short and fast). Writes
results/walkforward/dd<W>/.
"""
import os
import sys

import numpy as np

import bt_eval as ev
import portfolio_bt as pb
from portfolio_env import calibrate_risk_aversion

OUT = os.path.join(pb.ROOT, "results", "walkforward", pb.RUN_TAG)
FIRST, LAST = 2014, int(pb.END[:4])


def window_split(Y):
    """(check years, {"A": years, "B": years}, stage folds) of the window 2006 .. Y-1."""
    check = [Y - 2, Y - 1]
    years = list(range(int(pb.TRAIN_START[:4]), Y - 2))
    blocks = [years[i:i + 2] for i in range(0, len(years), 2)]
    if len(blocks) > 1 and len(blocks[-1]) == 1:                 # a lone trailing year
        last = blocks.pop()             # (not blocks[-2] += blocks.pop(): the target index is
        blocks[-1] = blocks[-1] + last  # resolved before the pop and assigned after it)
    A = [y for b in blocks[0::2] for y in b]
    B = [y for b in blocks[1::2] for y in b]
    stage_folds = ("A", "A", "B", "B") if B else ("A", "A", "A", "A")
    if len(pb.STAGES) == 2:             # one tree searched (--stocks-only, --voltarget): folds A then B
        stage_folds = ("A", "B") if B else ("A", "A")
    return check, {"A": A, "B": B or A}, stage_folds


def run(first):
    panel = pb.load_panel()
    F, M, sn, mn = pb.load_features()
    os.makedirs(OUT, exist_ok=True)
    for Y in range(first, LAST + 1):
        done = os.path.join(OUT, f"y{Y}.npz")
        if os.path.exists(done):
            continue
        check, folds, stage_folds = window_split(Y)
        end = f"{Y - 1}-12-31"
        gamma = calibrate_risk_aversion(panel, pb.TRAIN_START, end)
        print(f"\n##### [{ev.stamp()}] walk-forward {Y}: train {pb.TRAIN_START[:4]}-{Y - 1}, check {check}, "
              f"folds A {folds['A']} B {folds['B']}", flush=True)
        env = pb.world(panel, F, M, sn, mn, pb.TRAIN_START, end, T=pb.CFG["T"], gamma=gamma,
                       holdout=folds["A"], check=check)
        # started from buy all, not the EBTDA rule: that rule was chosen after seeing
        # 2020-2026, and seeding every window with it would carry that hindsight in
        banks, state = ev.run_stages(env, os.path.join(OUT, f"y{Y}"), folds=folds, stage_folds=stage_folds,
                                     on_stage=lambda st, Y=Y: year_plot(Y))      # y<Y>.png after every stage
        pairs = {"trees": (banks["stocks"], banks["exposure"]),
                 "buy all": (env.default_bank("stocks"), env.default_bank("exposure")),
                 "EBTDA": (pb.ebtda_rule(env.stock_names), env.default_bank("exposure"))}
        year_end = pb.END if Y == LAST else f"{Y}-12-31"
        paths = ev.period_paths(panel, F, M, sn, mn, pairs, f"{Y}-01-01", year_end, gamma)
        np.savez(done, gamma=gamma, dates=paths["trees"]["dates"].values.astype("datetime64[D]"),
                 **{f"{k}_real": v["real"] for k, v in paths.items()},
                 **{f"{k}_daily": v["daily"] for k, v in paths.items()},
                 arms=len(banks["stocks"]["clauses"]) + len(banks["exposure"]["clauses"]),
                 reverted=sum(bool((s.get("check") or {}).get("reverted")) for s in state["stages"]))
        yr = lambda k: 100 * (np.prod(1 + paths[k]["daily"]) - 1)
        print(f"  {Y}: trees {yr('trees'):+.1f}%  buy all {yr('buy all'):+.1f}%  EBTDA {yr('EBTDA'):+.1f}%  "
              f"S&P {yr('S&P 500'):+.1f}%", flush=True)
        del env
        try:                                   # the pictures and the chained record, as it grows
            summary(quiet=True)
        except Exception as exc:
            print("  (summary skipped:", exc, ")", flush=True)


def summary(quiet=False):
    import pandas as pd
    files = sorted(f for f in os.listdir(OUT) if f.startswith("y") and f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no walk-forward years in {OUT} yet: run python walkforward.py")
    rows, chain = [], {k: [] for k in ("trees", "buy all", "EBTDA", "S&P 500")}
    gammas = []
    for f in files:
        z = np.load(os.path.join(OUT, f))
        Y = int(f[1:5])
        gammas.append(float(z["gamma"]))
        row = {"year": Y, "arms": int(z["arms"]), "stages reverted": int(z["reverted"])}
        for k in chain:
            chain[k].append(z[f"{k}_real"])
            row[f"{k} real %"] = 100 * (np.prod(1 + z[f"{k}_real"]) - 1)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("year").round(1)
    real = {k: np.concatenate(v) for k, v in chain.items()}
    g = float(np.mean(gammas))            # one risk aversion for the chained comparison
    lines = [f"# Walk-forward track record {files[0][1:5]}-{files[-1][1:5]} (drawdown weight {pb.DD_WEIGHT:g})", "",
             "Every year traded by trees grown only on the years before it.", "", "```", df.to_string(), "```", "",
             "| chained, real | CER %/yr | max drawdown % | vs buy all: CER diff | p | vs S&P: CER diff | p |",
             "|---|---|---|---|---|---|---|"]
    for k in ("trees", "EBTDA", "buy all", "S&P 500"):
        d_b, p_b = ev.paired(real[k], real["buy all"], g) if k != "buy all" else (0.0, np.nan)
        d_s, p_s = ev.paired(real[k], real["S&P 500"], g) if k != "S&P 500" else (0.0, np.nan)
        lines.append(f"| {k} | {pb.cer(real[k], g):+.2f} | {100 * ev.max_drawdown(real[k]):.1f} | "
                     f"{d_b:+.2f} | {p_b:.3f} | {d_s:+.2f} | {p_s:.3f} |")
    for f in files:                       # one picture per year: its path and the trees that traded it
        year_plot(int(f[1:5]))
    dates = np.concatenate([np.load(os.path.join(OUT, f))["dates"] for f in files])
    png = os.path.join(OUT, "walkforward.png")
    plot(dates, real, df, png)
    lines += ["", f"![walk-forward]({os.path.basename(png)})"]
    text = "\n".join(lines)
    if not quiet:
        print(text)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote", os.path.join(OUT, "summary.md"), "and", png, "(and one y<YEAR>.png per year)")


COLORS = {"trees": "#1f77b4", "buy all": "#7f7f7f", "EBTDA": "#2ca02c", "S&P 500": "black"}


def year_plot(Y):
    """y<Y>.png: the year's $100k paths and, in words, the trees that traded it (the last
    stock and exposure stage of that year's search) with what the check gate did."""
    import json
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    npz = os.path.join(OUT, f"y{Y}.npz")
    z = np.load(npz) if os.path.exists(npz) else None          # None while the year is training
    with open(os.path.join(OUT, f"y{Y}", "state.json")) as fh:
        state = json.load(fh)
    last = {s["agent"]: s for s in state["stages"]}
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.08)
    a = fig.add_subplot(gs[0, 0])
    if z is None:
        a.text(0.5, 0.5, f"{Y} is still training: stage {len(state['stages'])} of {len(pb.STAGES)} done\n"
               f"(the year is traded once all stages finish)", ha="center", va="center", fontsize=11,
               transform=a.transAxes)
        a.set_xticks([]), a.set_yticks([])
    else:
        d = pd.DatetimeIndex(z["dates"])
        for k in COLORS:
            w = 100_000 * np.cumprod(1 + z[f"{k}_daily"])
            a.plot(d, w, color=COLORS[k], lw=2.4 if k == "trees" else 1.2, ls="--" if k == "S&P 500" else "-",
                   label=f"{k}: {100 * (w[-1] / 1e5 - 1):+.1f}%")
        a.legend(fontsize=9, loc="upper left")
    a.set_title(f"{Y}: traded by trees grown on {pb.TRAIN_START[:4]}-{Y - 1}", fontsize=11)
    a.set_ylabel("value of $100k (nominal)")
    a.grid(True, color="0.88", lw=0.6)
    t = fig.add_subplot(gs[0, 1])
    t.axis("off")
    text = [f"STAGES (held-out fold, check years {Y - 2}-{Y - 1})"]
    for s in state["stages"]:
        c = s.get("check") or {}
        verdict = "reverted" if c.get("reverted") else ("kept" if c.get("diff") else "no change")
        text.append(f"  {s['stage']} {s['agent']:8s} fold {s.get('fold', '?')}: {s['arms']} arms, held-out "
                    f"{s['held_out']:+.1f}, check {c.get('diff', 0.0):+.2f} +- {c.get('se', 0.0):.2f} -> {verdict}")
    for agent in ("stocks", "exposure"):
        text += ["", f"{agent.upper()} TREE ({last[agent]['arms']} arms)" if agent in last else f"{agent.upper()} TREE"]
        if agent in last:
            text += ["  " + ln for ln in ev.describe_tree(last[agent]["bank"])]
    wrapped = [w for ln in text for w in (textwrap.wrap(ln, 105, subsequent_indent="        ") or [""])]
    t.text(0, 1, "\n".join(wrapped[:58]), va="top", family="monospace", fontsize=7.6, transform=t.transAxes)
    fig.savefig(os.path.join(OUT, f"y{Y}.png"), dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot(dates, real, df, png):
    """$100k chained over every out-of-sample year (real), each year's gap to buy all, and the
    chained drawdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    d = pd.DatetimeIndex(dates)
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw=dict(height_ratios=[1.6, 1, 0.9]))
    for k, r in real.items():
        w = 100_000 * np.cumprod(1 + r)
        a.plot(d, w, color=COLORS[k], lw=2.4 if k == "trees" else 1.3, ls="--" if k == "S&P 500" else "-",
               label=f"{k}: ${w[-1] / 1000:,.0f}k")
    a.set_yscale("log")
    a.set_ylabel("real value of $100k (log scale)")
    a.set_title("Walk-forward: every year traded by trees grown only on the years before it", fontsize=11)
    a.legend(fontsize=9, loc="upper left")
    years = df.index.to_numpy()
    for off, k in ((-0.2, "trees"), (0.2, "EBTDA")):
        gap = df[f"{k} real %"] - df["buy all real %"]
        b.bar(years + off, gap, width=0.38, color=COLORS[k], label=f"{k} minus buy all")
    for y, n in zip(years, df["arms"]):
        b.text(y - 0.2, 0, f"{n}", ha="center", va="bottom", fontsize=7, color=COLORS["trees"])
    b.axhline(0, color="black", lw=1)
    b.set_xticks(years)
    b.set_ylabel("real return gap, % points")
    b.set_title("Each year against buy all (numbers: arms in that year's trees)", fontsize=10)
    b.legend(fontsize=8, loc="upper left")
    for k, r in real.items():
        w = np.cumprod(1 + r)
        c.plot(d, 100 * (w / np.maximum.accumulate(w) - 1), color=COLORS[k], lw=2.0 if k == "trees" else 1.0,
               ls="--" if k == "S&P 500" else "-", label=k)
    c.set_ylabel("below running high, %")
    c.set_title("Drawdown of the chained path", fontsize=10)
    c.legend(fontsize=8, loc="lower left", ncol=4)
    for ax in (a, b, c):
        ax.grid(True, color="0.88", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        run(int(pb._flag_value("--from") or FIRST))
