#!/usr/bin/env python3
"""Infrastructure-configuration benchmark: energy per *solved* task.

Stand-alone. Writes only files nothing else in the paper uses, so this can be developed in
parallel with the other sections:

    figures/fig_infra.pdf
    tables/tab_infra.tex
    tables/infra_macros.tex     <- every number the prose quotes, as \\newcommand

It must not regenerate fig_comm / fig_amortization / fig_latency / tab_dimensioning, which are
shared with the earlier manuscripts.

WHY THIS SECTION EXISTS. The metric of Eq. (1) divides by useful output *tokens*, which is a
proxy: a workflow that fails its task still emits tokens and still scores. This benchmark
supplies a task-completion denominator, so energy can be charged per *solved* task. Because
workflow energy grows super-linearly in team size (Eq. 7) while task success saturates, joules
per solved task has a minimum at finite N -- a cost-optimal team size, which the token-based
metric cannot locate.

INPUT SCHEMA -- results/infra_runs.csv, one row per (topology, N, run, task):

    topology        str   e.g. single-thinking | proposer-synth
                          | inspection-proposer-synth-verifier-actuator
                          | state-proposer-synth-verifier-actuator
    n_proposers     int   the paper's N; 1 for the single-model baseline
    run_id          int   repeat index
    task_id         str   benchmark task identifier
    difficulty      str   easy | medium | hard        (optional)
    success         int   1 if the task completed successfully, else 0
    prefill_tokens  int   summed over every LLM call in the run
    decode_tokens   int   summed over every LLM call in the run
    wall_s          float wall-clock seconds                (optional)

Usage:  python3 make_figure_infra.py [path/to/infra_runs.csv]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import agentic_ecal as ae

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR, TABDIR = os.path.join(ROOT, "figures"), os.path.join(ROOT, "tables")
DEFAULT_CSV = os.path.join(HERE, "results", "infra_runs.csv")

# CALIBRATION STATUS: the runs use Qwen3.5-9B, for which we have no measured two-rate
# coefficients. We price with the Qwen2.5-7B calibration and say so in the text; the ordering
# across topologies is insensitive to the choice because every topology is priced identically.
CALIB = ae.TWO_RATE["qwen2_5_7b"]
BATCH = 64

ORDER = ["single-thinking", "proposer-synth",
         "inspection-proposer-synth-verifier-actuator",
         "state-proposer-synth-verifier-actuator"]
SHORT = {"single-thinking": "single + thinking",
         "proposer-synth": "proposer$\\to$synth",
         "inspection-proposer-synth-verifier-actuator": "+ verifier/actuator (inspection)",
         "state-proposer-synth-verifier-actuator": "+ verifier/actuator (state)"}


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the small run counts this campaign has."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(path):
    d = pd.read_csv(path)
    need = {"topology", "n_proposers", "success", "prefill_tokens", "decode_tokens"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{path}: missing required columns {sorted(missing)}")
    return d


def summarise(d):
    """Per (topology, N): success rate with CI, energy per task, and energy per solved task."""
    rows = []
    for (topo, n), g in d.groupby(["topology", "n_proposers"]):
        k, tot = int(g.success.sum()), len(g)
        rate = k / tot
        lo, hi = wilson(k, tot)
        e = CALIB.c_pre * g.prefill_tokens.mean() + CALIB.c_dec(BATCH) * g.decode_tokens.mean()
        rows.append(dict(topology=topo, n=int(n), tasks=tot, solved=k, rate=rate,
                         lo=lo, hi=hi, e_task=e,
                         e_solved=(e / rate if rate > 0 else float("nan")),
                         runs=g.run_id.nunique() if "run_id" in g else 1))
    return pd.DataFrame(rows).sort_values(["topology", "n"])


def figure(s):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))
    topos = [t for t in ORDER if t in set(s.topology)] or sorted(set(s.topology))
    for i, topo in enumerate(topos):
        g = s[s.topology == topo].sort_values("n")
        c, lab = f"C{i}", SHORT.get(topo, topo)
        ax[0].plot(g.n, 100 * g.rate, "-o", color=c, ms=3, lw=1.3, label=lab)
        ax[0].fill_between(g.n, 100 * g.lo, 100 * g.hi, color=c, alpha=0.15, lw=0)
        ax[1].plot(g.n, g.e_task, "-o", color=c, ms=3, lw=1.3)
        ax[2].plot(g.n, g.e_solved, "-o", color=c, ms=3, lw=1.3)
        if g.e_solved.notna().any():          # mark the cost-optimal team size
            b = g.loc[g.e_solved.idxmin()]
            ax[2].plot([b.n], [b.e_solved], "v", color=c, ms=6)
    ax[0].set_ylabel("tasks solved (\\%)"); ax[0].set_title("(a) benefit saturates", fontsize=8)
    ax[1].set_ylabel("energy per task [J]"); ax[1].set_title("(b) cost is super-linear", fontsize=8)
    ax[2].set_ylabel("energy per solved task [J]")
    ax[2].set_title("(c) the optimum", fontsize=8)
    for a in ax:
        a.set_xlabel("proposers $N$"); a.grid(alpha=0.25, lw=0.4); a.tick_params(labelsize=7)
    ax[1].set_yscale("log"); ax[2].set_yscale("log")
    ax[0].legend(fontsize=5, frameon=False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_infra.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def tables(s, d):
    os.makedirs(TABDIR, exist_ok=True)
    body = []
    for topo in [t for t in ORDER if t in set(s.topology)] or sorted(set(s.topology)):
        g = s[s.topology == topo]
        b = g.loc[g.e_solved.idxmin()]
        body.append(f"{SHORT.get(topo, topo)} & {int(b.n)} & {100*b.rate:.0f}\\% & "
                    f"{b.e_task:,.0f} & {b.e_solved:,.0f} \\\\")
    tex = f"""% Generated by scripts/clean/make_figure_infra.py -- do not edit by hand.
\\begin{{table}}[t]
\\ROOF{{}}
\\caption{{Cost-optimal operating point per topology on the infrastructure-configuration
benchmark. $N^\\star$ minimises energy per \\emph{{solved}} task.}}
\\label{{tab:infra}}
\\centering
\\small
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Topology & $N^\\star$ & solved & J/task & J/solved \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(os.path.join(TABDIR, "tab_infra.tex"), "w") as fh:
        fh.write(tex)

    # every number the prose quotes, so the text never needs editing when runs land
    best = s.loc[s.e_solved.idxmin()]
    worst_n = s.n.max()
    at_worst = s[(s.topology == best.topology) & (s.n == worst_n)]
    ratio = (float(at_worst.e_solved.iloc[0]) / best.e_solved) if len(at_worst) else float("nan")
    m = f"""% Generated by scripts/clean/make_figure_infra.py -- do not edit by hand.
\\newcommand{{\\infraTasks}}{{{d.task_id.nunique() if 'task_id' in d else len(d)}}}
\\newcommand{{\\infraRuns}}{{{int(s.runs.max())}}}
\\newcommand{{\\infraTopos}}{{{s.topology.nunique()}}}
\\newcommand{{\\infraBestTopo}}{{{SHORT.get(best.topology, best.topology)}}}
\\newcommand{{\\infraBestN}}{{{int(best.n)}}}
\\newcommand{{\\infraBestRate}}{{{100*best.rate:.0f}}}
\\newcommand{{\\infraBestJ}}{{{best.e_solved:,.0f}}}
\\newcommand{{\\infraMaxN}}{{{int(worst_n)}}}
\\newcommand{{\\infraOverrun}}{{{ratio:.1f}}}
"""
    with open(os.path.join(TABDIR, "infra_macros.tex"), "w") as fh:
        fh.write(m)
    print("wrote", os.path.join(TABDIR, "tab_infra.tex"), "and infra_macros.tex")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv):
        raise SystemExit(
            f"no results yet at {csv}\n"
            "Expected columns: topology, n_proposers, run_id, task_id, difficulty, success,\n"
            "prefill_tokens, decode_tokens, wall_s. See the module docstring.")
    d = load(csv)
    s = summarise(d)
    print(s.to_string(index=False))
    figure(s)
    tables(s, d)
