#!/usr/bin/env python3
"""Framing figure: an agentic workflow mapped onto a network substrate.

The paper's terms all arise from one picture. A workflow is a logical graph whose nodes are LLM
calls and whose edges carry context between them. Deployment maps that graph onto a physical
substrate of accelerators at successive network tiers. Where an edge stays inside a tier its
context is prompt tokens and is charged to E_call; where it crosses a tier boundary the same
context becomes traffic and is charged additionally to E_tx at the bearer's intensity. Placement is
that mapping, and nothing else.

The two layers are drawn as ONE diagram -- tier as the vertical axis, workflow order as the
horizontal -- rather than as a graph above a substrate joined by projection lines. The earlier
two-layer form spent most of its area on those lines and on the gap between the layers, and was
authored 7.1 in wide while the manuscript includes it at \\columnwidth, so its labels reached the
page below 3 pt. Everything here is sized for the column it is actually placed in.

Writes figures/fig_overview.pdf (default ../../v4/figures).
"""
from __future__ import annotations

import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

import agentic_ecal as ae
import osi

HERE = os.path.dirname(os.path.abspath(__file__))

def _paper_figures(default="figures"):
    """Newest vN/figures alongside the repo, or repo-root figures/ if none exists."""
    root = os.path.dirname(os.path.dirname(HERE))
    vs = sorted((d for d in os.listdir(root)
                 if re.fullmatch(r"v\d+", d) and os.path.isdir(os.path.join(root, d))),
                key=lambda d: int(d[1:]))
    return os.path.join(root, vs[-1] if vs else "", default)

OUT_DEFAULT = _paper_figures()

# Equal data aspect, so the agent markers are round: the y-range is the x-range scaled by the
# figure's own aspect. Everything below is in those units.
XR, YR = 12.0, 8.0
X0, X1 = 2.35, 11.75                       # band extent; the tier labels sit to the left of X0
BH, GAP = 1.95, 0.40                       # band height, and the gap that IS the bearer

# tier bands, bottom to top: (label, memory, y0)
TIERS = [("far edge", "8 GB", 0.20), ("edge", "64 GB", 2.55), ("core DC", "80 GB", 4.90)]
BEARERS = [("5G", "10^{-6}"), ("metro", "10^{-8}")]   # far/edge boundary, edge/core boundary

# agents: (label, x, tier, dy within the band)
AGENTS = [("$a_1$", 3.15, 0, 1.35), ("$a_2$", 4.75, 0, 1.35), ("$a_3$", 6.55, 1, 1.02),
          ("$a_4$", 8.60, 2, 1.42), ("$a_5$", 8.60, 2, 0.53), ("$a_6$", 10.65, 2, 0.97)]
EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)]
R = 0.36                                   # agent radius

HANDOFF = 2700          # tokens on one hand-off, the case study's largest


def crossing_ratio(tokens=HANDOFF, eps=1e-6):
    """Prefill energy of a hand-off against the energy of transporting it. The figure's point."""
    bits = tokens * ae.BITS_PER_TOKEN
    tx = osi.segment_energy(bits, eps, 0.0) + osi.endpoint_stack_energy(bits)
    return ae.TWO_RATE["qwen2_5_7b"].c_pre * tokens / tx


def draw(ax):
    ax.set_xlim(0, XR); ax.set_ylim(0, YR); ax.set_aspect("equal"); ax.axis("off")

    for name, mem, y0 in TIERS:
        ax.add_patch(FancyBboxPatch((X0, y0), X1 - X0, BH, boxstyle="round,pad=0.04",
                                    fc="#eef2f7", ec="0.62", lw=0.8, zorder=0))
        ax.text(X0 - 0.24, y0 + BH / 2 + 0.24, name, ha="right", va="center",
                fontsize=6.2, weight="bold")
        ax.text(X0 - 0.24, y0 + BH / 2 - 0.36, mem, ha="right", va="center",
                fontsize=5.2, color="0.45")

    # the bearers ARE the gaps between the bands; the label punches through on a white ground
    for i, (lab, eps) in enumerate(BEARERS):
        yb = TIERS[i][2] + BH + GAP / 2
        ax.plot([X0, X1 - 2.55], [yb, yb], "-", color="C0", lw=0.9, alpha=0.6, zorder=1)
        ax.text(X1 - 2.42, yb, f"{lab}, $\\varepsilon={eps}$", ha="left", va="center",
                fontsize=5.2, color="C0", zorder=2)

    pos = {}
    for i, (lab, x, t, dy) in enumerate(AGENTS):
        pos[i] = (x, TIERS[t][2] + dy)
        ax.add_patch(Circle(pos[i], R, fc="#cfe0f5", ec="C0", lw=1.1, zorder=3))
        ax.text(*pos[i], lab, ha="center", va="center", fontsize=6.0, zorder=4)

    tier_of = {i: a[2] for i, a in enumerate(AGENTS)}
    for u, v in EDGES:
        cross = tier_of[u] != tier_of[v]
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=7,
                                     shrinkA=11, shrinkB=11, lw=1.5 if cross else 0.8,
                                     color="C3" if cross else "0.55",
                                     connectionstyle="arc3,rad=0.10", zorder=2))

    # the quantitative claim, in the space the far edge leaves empty, on the crossing it describes
    mid = ((pos[1][0] + pos[2][0]) / 2, (pos[1][1] + pos[2][1]) / 2)
    ax.annotate(f"the same context, now also traffic:\n"
                f"$E_{{\\mathrm{{tx}}}}$ is $1/{crossing_ratio():.0f}$ of the prefill it feeds",
                xy=mid, xytext=(6.95, 1.02), fontsize=5.0, color="C3", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.7,
                                shrinkA=3, shrinkB=6, connectionstyle="arc3,rad=0.18"))

    for j, (col, lw, txt) in enumerate([("0.55", 0.8, "inside a tier: $E_{\\mathrm{call}}$"),
                                        ("C3", 1.5, "across a tier: "
                                                    "$E_{\\mathrm{call}}+E_{\\mathrm{tx}}$")]):
        y = YR - 0.22 - j * 0.44
        ax.plot([X0 + 0.05, X0 + 0.75], [y, y], "-", color=col, lw=lw)
        ax.text(X0 + 0.88, y, txt, fontsize=5.2, color=col if col == "C3" else "0.4",
                va="center")


def main(out_dir=OUT_DEFAULT):
    fig, ax = plt.subplots(figsize=(3.45, 3.45 * YR / XR))
    draw(ax)
    fig.tight_layout(pad=0.12)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fig_overview.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    tier_of = {i: a[2] for i, a in enumerate(AGENTS)}
    cross = sum(1 for u, v in EDGES if tier_of[u] != tier_of[v])
    print(f"  {len(AGENTS)} agents, {len(EDGES)} edges, {cross} crossing a tier boundary")
    print(f"  {HANDOFF}-token hand-off on 5G: prefill / E_tx = {crossing_ratio():.0f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
