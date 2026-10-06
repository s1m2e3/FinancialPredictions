"""One page over a set of portfolio_bt runs: each run's kept trees, its stage table and its
validation (2020-2021) and test (2022-2026) rows -- the S&P 500, the run's buy-all baseline
and the learned trees -- with alpha and beta; plus the placebo and stability summaries when
they exist. Reads only what the runs wrote (report.md, stock/exposure_tree.json).

    python night_summary.py RUN_DIR [RUN_DIR ...]     writes results/night_summary.md
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "results", "portfolio_bt")
KEEP = re.compile(r"^(S&P 500|buy all \(by size\)|the \d+ largest \(by size\)|stock tree, 100% invested|stock tree \+ exposure tree)")


def section(report, title):
    """The ``` block under '## title' of a report, as lines."""
    lines = report.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == f"## {title}")
    except StopIteration:
        return []
    out, inside = [], False
    for ln in lines[i + 1:]:
        if ln.startswith("```"):
            if inside:
                break
            inside = True
            continue
        if inside:
            out.append(ln)
    return out


def main(runs):
    page = ["# Night summary", ""]
    for run in runs:
        d = run if os.path.isabs(run) else os.path.join(BASE, run)
        rep = os.path.join(d, "report.md")
        page += [f"## {os.path.basename(d)}", ""]
        if not os.path.exists(rep):
            page += ["(no report yet)", ""]
            continue
        text = open(rep, encoding="utf-8").read()
        stage = [ln for ln in text.splitlines() if re.match(r"^\|( |---\|)", ln)]
        page += stage + [""]
        for t in ("Stock tree", "Exposure tree"):
            page += [f"**{t}**", "", "```"] + section(text, t) + ["```", ""]
        for per in ("validation 2020-2021", "test 2022-2026"):
            rows = section(text, per)
            if rows:
                page += [f"**{per}**", "", "```", rows[0]] + [r for r in rows[1:] if KEEP.match(r)] + ["```", ""]
    for extra in ("placebo", "stability"):
        for root, _, files in os.walk(os.path.join(ROOT, "results", extra)):
            if "summary.md" in files:
                page += [f"## {extra}: {os.path.relpath(root, ROOT)}", "", open(os.path.join(root, "summary.md"),
                                                                              encoding="utf-8").read(), ""]
    out = os.path.join(ROOT, "results", "night_summary.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(page))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
