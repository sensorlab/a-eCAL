#!/usr/bin/env python3
"""Figure and table for model placement under inter-agent communication.

Kept separate from ``make_figures.py`` so that regenerating this material cannot alter any
figure the frozen manuscript builds from. Writes:

    figures/fig_model_placement.pdf
    tables/tab_model_placement.tex

Measured basis: results/steiner_tokens_by_cell.csv, summarised by
``fit_peer_law.py`` from ~/DATA/data_steiner_Aug26 (6.69M team-item records, 16 open-weight
models, team sizes 1..30 at full mesh, 3 debate rounds, peer_token_budget=120). Token counts
only; task accuracy is not analysed.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _pkg_path  # noqa: F401  (puts repo root on sys.path)
import agentic_ecal_pkg as ae
import osi
import model_placement as mp

HERE = os.path.dirname(os.path.abspath(__file__))

def _paper_figures(default="figures"):
    """Newest vN/figures alongside the repo, or repo-root figures/ if none exists."""
    root = os.path.dirname(HERE)
    vs = sorted((d for d in os.listdir(root)
                 if re.fullmatch(r"v\d+", d) and os.path.isdir(os.path.join(root, d))),
                key=lambda d: int(d[1:]))
    return os.path.join(root, vs[-1] if vs else "", default)

ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(os.path.dirname(HERE), "results", "steiner_tokens_by_cell.csv")
FIGDIR = _paper_figures()
TABDIR = _paper_figures("tables")

CASE = "qwen2.5-7b"          # the model the manuscript already uses
CASE_KEY = "qwen2_5_7b"
CASE_BATCH = 64
# Span the KV-width axis, and include a model small enough to reach the far edge.
HIGHLIGHT = ["qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b", "olmo2-7b"]
TIERS = ["orin_nano", "agx_orin", "l4", "a100"]
TEAM = 30                    # the largest measured team


def load() -> pd.DataFrame:
    """Per-agent token means, pooled over the eight reasoning tasks.

    The cached summary is resolved by task; that axis must be collapsed by a count-weighted
    sum before dividing, or every per-team quantity comes out eightfold too large.
    """
    d = pd.read_csv(RESULTS)
    d = d[d.condition.str.startswith("comm_true")]
    keys = ["model_short", "condition", "team_size", "comm_k", "comm_round"]
    g = d.groupby(keys, as_index=False)[["n", "p_sum", "o_sum"]].sum()
    g["p_ag"] = g.p_sum / g.n / g.team_size
    g["o_ag"] = g.o_sum / g.n / g.team_size
    return g


def panel_a(ax, d):
    """Per-agent prefill tokens against peer count: the measured communication law."""
    r2 = d[d.comm_round == 2]
    for m, g in r2.groupby("model_short"):
        g = g.sort_values("comm_k")
        c = "C0" if m == CASE else "0.75"
        z = 3 if m == CASE else 1
        ax.plot(g.comm_k, g.p_ag, "-o", color=c, ms=2.5, lw=1.4 if m == CASE else 0.7,
                zorder=z, alpha=1.0 if m == CASE else 0.6)
    law = mp.PEER_LAW[CASE]
    k = np.linspace(0, 29, 50)
    # b is the serving batch everywhere else in the paper; the slope is tau
    ax.plot(k, law.a + law.b * k, "--", color="k", lw=1.0, zorder=4,
            label=f"$p_0+\\tau k$, $\\tau={law.b:.0f}$")
    ax.axhline(law.solo, color="C3", lw=1.0, ls=":", zorder=4)
    ax.text(14.5, law.solo * 1.35, "no communication", color="C3", fontsize=6.5, ha="center")
    ax.set_xlabel("peers read, $k=N-1$")
    ax.set_ylabel("prefill tokens per agent")
    ax.set_title("(a) the communication law", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")


def panel_b(ax, d):
    """Team prefill: linear without communication, quadratic with it."""
    q = d[d.model_short == CASE]
    Ns, solo, deb = [], [], []
    for N, g in q.groupby("team_size"):
        r1 = g[g.comm_round == 1]
        if r1.empty:
            continue
        Ns.append(N)
        solo.append(float(r1.p_ag.iloc[0]) * N)
        deb.append(float((g.p_ag * N).sum()))
    Ns = np.array(Ns, float); solo = np.array(solo); deb = np.array(deb)
    ax.loglog(Ns, solo, "-o", color="C3", ms=3, lw=1.4, label="no communication")
    ax.loglog(Ns, deb, "-o", color="C0", ms=3, lw=1.4, label="3-round debate")
    hi = Ns >= 10
    s = np.polyfit(np.log(Ns[hi]), np.log(deb[hi]), 1)[0]
    ax.text(0.05, 0.62, f"slope $\\to$ {s:.2f}", transform=ax.transAxes, fontsize=7, color="C0")
    ax.text(0.42, 0.05, "slope $1$", transform=ax.transAxes, fontsize=7, color="C3")
    ax.annotate("", xy=(Ns[-1], deb[-1]), xytext=(Ns[-1], solo[-1]),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.8))
    ax.text(Ns[-1] * 0.92, (deb[-1] * solo[-1]) ** 0.5,
            f"{deb[-1]/solo[-1]:.0f}$\\times$", fontsize=7.5, color="0.2", ha="right")
    ax.set_ylim(top=deb.max() * 3.5)
    ax.set_xlabel("team size $N$")
    ax.set_ylabel("team prefill tokens")
    ax.set_title("(b) quadratic in team size", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")


def panel_c(ax):
    """Agents a device can hold: communication is a memory constraint."""
    x = np.arange(len(TIERS)); w = 0.19
    for i, key in enumerate(HIGHLIGHT):
        m = mp.FLEET[key]
        s = [mp.max_agents(m, mp.DEVICES[t], mp.context_tokens(key, TEAM, False)) for t in TIERS]
        b = [mp.max_agents(m, mp.DEVICES[t], mp.context_tokens(key, TEAM, True)) for t in TIERS]
        off = (i - 1.5) * w
        ax.bar(x + off, s, w, color=f"C{i}", alpha=0.30,
               label=f"{key} $\\cdot$ {mp.kv_bytes_per_token(m)/1024:.0f} KB")
        ax.bar(x + off, b, w, color=f"C{i}")
    ax.axhline(TEAM, color="k", ls="--", lw=1.0)
    ax.text(len(TIERS) - 0.6, TEAM * 1.3, f"$N={TEAM}$ team", fontsize=6.5, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(top=3e6)
    ax.set_xticks(x)
    ax.set_xticklabels([mp.DEVICES[t].label for t in TIERS], fontsize=6.5)
    ax.set_ylabel("concurrent agents hosted")
    ax.set_title("(c) placement: solid = debate, pale = solo", fontsize=8)
    ax.text(0.02, 0.60, "KV per token", transform=ax.transAxes, fontsize=5.0,
            color="0.35")
    ax.legend(fontsize=5.0, frameon=False, loc="upper left", ncol=2,
              handlelength=0.9, columnspacing=0.6, borderpad=0.1, handletextpad=0.4)


def figure(d):
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6))
    panel_a(axes[0], d); panel_b(axes[1], d); panel_c(axes[2])
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_model_placement.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def table(d):
    """Placement feasibility, ordered by KV width rather than parameter count."""
    rows = []
    order = sorted(mp.FLEET, key=lambda k: mp.kv_bytes_per_token(mp.FLEET[k]))
    for key in order:
        m = mp.FLEET[key]
        kv = mp.kv_bytes_per_token(m) / 1024.0
        w = mp.weight_bytes(m) / mp.GB
        cells = []
        for t in TIERS:
            dev = mp.DEVICES[t]
            if mp.weight_bytes(m) > dev.usable:
                cells.append("--")
            else:
                a = mp.max_agents(m, dev, mp.context_tokens(key, TEAM, True))
                cells.append(f"\\textbf{{{a}}}" if a < TEAM else f"{a}")
        rows.append(f"{m.name} & {w:.1f} & {kv:.0f} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    hdr = " & ".join(mp.DEVICES[t].label.replace("\n", " ") for t in TIERS)
    tex = f"""% Generated by figure_scripts/make_figures_placement.py -- do not edit by hand.
\\begin{{table*}}[t]
\\ROOF{{}}
\\caption{{Agents a device can host concurrently at the measured $N={TEAM}$ full-mesh debate
context, ordered by KV width. Entries in bold denote devices that cannot host the team in which their agents
participate; ``--'' denotes that the weights alone exceed the device.}}
\\label{{tab:modelplacement}}
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\\toprule
Model & $W$ [GB] & KV [KB/tok] & {hdr} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    os.makedirs(TABDIR, exist_ok=True)
    out = os.path.join(TABDIR, "tab_model_placement.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    print("wrote", out)


def numbers(d):
    """The quantities quoted in the manuscript text."""
    q = d[d.model_short == CASE]
    cal = ae.TWO_RATE[CASE_KEY]
    dec = ae.decode_rate(ae.LLMS[CASE_KEY], ae.HW["a100"], batch=CASE_BATCH, context=1500)
    print("\n--- numbers quoted in v2 ---")
    b = [mp.PEER_LAW[k].b for k in mp.PEER_LAW]
    print(f"tokens/peer: min {min(b):.0f}  median {np.median(b):.0f}  max {max(b):.0f}")
    for N in (10, TEAM):
        g = q[q.team_size == N]
        r1 = g[g.comm_round == 1]
        Ps = float(r1.p_ag.iloc[0]) * N; Os = float(r1.o_ag.iloc[0]) * N
        P = float((g.p_ag * N).sum()); O = float((g.o_ag * N).sum())
        Es, Ed = cal.c_pre * Ps + dec * Os, cal.c_pre * P + dec * O
        k = N - 1
        bits = N * k * mp.PEER_LAW[CASE].b * 2 * ae.BITS_PER_TOKEN
        etx = osi.segment_energy(bits, 1e-6, 0.0) + osi.endpoint_stack_energy(bits)
        print(f"N={N}: prefill {Ps:,.0f} -> {P:,.0f} tok ({P/Ps:.0f}x), "
              f"E {Es:.0f} -> {Ed:.0f} J ({Ed/Es:.1f}x), prefill/E_tx = {cal.c_pre*P/etx:.0f}x")
    for key in HIGHLIGHT:
        m = mp.FLEET[key]
        dv = mp.DEVICES["a100"]
        print(f"  {key:<18} KV {mp.kv_bytes_per_token(m)/1024:>5.0f} KB/tok  A100 hosts "
              f"{mp.max_agents(m, dv, mp.context_tokens(key, TEAM, True)):>4} debating "
              f"(solo {mp.max_agents(m, dv, mp.context_tokens(key, TEAM, False))})")


if __name__ == "__main__":
    d = load()
    figure(d); table(d); numbers(d)
