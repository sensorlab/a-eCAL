#!/usr/bin/env python3
"""Schematic of the eight orchestration architectures (fig:topologies).

Each panel is a small DAG. Nodes are colored by functional stage:
generate (first answer-producing tier, the L proposals), transform
(critics / refiners / synthesizers / duelists), the manager, and a
non-answering planner (Diamond only). Illustrative small N; the caption
notes N scales. Writes figures/topologies_v1.{pdf,png}.
"""
from __future__ import annotations
import re
from pathlib import Path

def _paper_figures():
    """Newest vN/figures alongside the repo; the manuscript is maintained outside it."""
    root = Path(__file__).resolve().parents[1]
    vs = sorted((d for d in root.iterdir()
                 if d.is_dir() and re.fullmatch(r"v\d+", d.name)),
                key=lambda d: int(d.name[1:]))
    return (vs[-1] if vs else root) / "figures"


FIG_DIR = _paper_figures()

# Colours come from roles.py so that this figure, the case study and the benchmark figure
# classify their agents the same way even though their concrete role names differ.
import roles

GEN = roles.GENERATE   # proposals, the L nodes, and Diamond's planner
TRA = roles.TRANSFORM  # refines one line of work: critics, chain refiners
MAN = roles.AGGREGATE  # reduces competing candidates: managers, synthesizers, duelists
STAGE = {"g": GEN, "t": TRA, "m": MAN}

# nodes: name -> (x, y, stage); edges: (from, to)
def _defs():
    return {
        "Star": (
            {"w0": (0, 0, "g"), "w1": (1, 0, "g"), "w2": (2, 0, "g"), "w3": (3, 0, "g"),
             "M": (1.5, 1.6, "m")},
            [("w0", "M"), ("w1", "M"), ("w2", "M"), ("w3", "M")],
            None,
        ),
        "Persona-Star": (
            {"p0": (0, 0, "g"), "p1": (1, 0, "g"), "p2": (2, 0, "g"), "p3": (3, 0, "g"),
             "M": (1.5, 1.6, "m")},
            [("p0", "M"), ("p1", "M"), ("p2", "M"), ("p3", "M")],
            "same graph, 6 role prompts",
        ),
        "Proposer-Critic": (
            {"p0": (0, 0, "g"), "p1": (1.5, 0, "g"), "p2": (3, 0, "g"),
             "c0": (0, 0.9, "t"), "c1": (1.5, 0.9, "t"), "c2": (3, 0.9, "t"),
             "M": (1.5, 1.8, "m")},
            [("p0", "c0"), ("p1", "c1"), ("p2", "c2"), ("c0", "M"), ("c1", "M"), ("c2", "M")],
            None,
        ),
        "Tournament": (
            {"p0": (0, 0, "g"), "p1": (1, 0, "g"), "p2": (2, 0, "g"), "p3": (3, 0, "g"),
             "d0": (0.5, 0.8, "m"), "d1": (2.5, 0.8, "m"), "d2": (1.5, 1.6, "m"),
             "M": (1.5, 2.4, "m")},
            [("p0", "d0"), ("p1", "d0"), ("p2", "d1"), ("p3", "d1"),
             ("d0", "d2"), ("d1", "d2"), ("d2", "M")],
            "pairwise duels, select",
        ),
        "Tree": (
            {"w0": (0, 0, "g"), "w1": (1, 0, "g"), "w2": (2, 0, "g"),
             "w3": (3, 0, "g"), "w4": (4, 0, "g"), "w5": (5, 0, "g"),
             "s0": (1, 0.9, "m"), "s1": (4, 0.9, "m"), "M": (2.5, 1.8, "m")},
            [("w0", "s0"), ("w1", "s0"), ("w2", "s0"), ("w3", "s1"), ("w4", "s1"),
             ("w5", "s1"), ("s0", "M"), ("s1", "M")],
            "fan-in-3, synthesize",
        ),
        "Chain": (
            {"w": (0, 0, "g"), "r1": (0, 0.75, "t"), "r2": (0, 1.5, "t"), "M": (0, 2.25, "m")},
            [("w", "r1"), ("r1", "r2"), ("r2", "M")],
            None,
        ),
        "Cascading-Chain": (
            {"w": (0, 0, "g"), "r1": (0, 0.75, "t"), "r2": (0, 1.5, "t"), "M": (0, 2.25, "m")},
            [("w", "r1"), ("r1", "r2"), ("r2", "M"), ("w", "r2"), ("w", "M"), ("r1", "M")],
            "manager sees all preceding",
        ),
        "Diamond": (
            {"P": (1.5, 0, "g"), "s0": (0, 0.9, "g"), "s1": (1.5, 0.9, "g"),
             "s2": (3, 0.9, "g"), "M": (1.5, 1.8, "m")},
            [("P", "s0"), ("P", "s1"), ("P", "s2"), ("s0", "M"), ("s1", "M"), ("s2", "M")],
            None,
        ),
    }


def _draw(ax, nodes, edges, note, title):
    from matplotlib.patches import FancyArrowPatch, Circle
    r = 0.16
    for a, b in edges:
        xa, ya, _ = nodes[a]
        xb, yb, _ = nodes[b]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                     mutation_scale=6, lw=0.7, color="#888888",
                     shrinkA=5.5, shrinkB=5.5, zorder=1))
    for name, (x, y, stg) in nodes.items():
        ax.add_patch(Circle((x, y), r, facecolor=STAGE[stg], edgecolor="white",
                     lw=0.6, zorder=2))
    xs = [x for x, _, _ in nodes.values()]
    ys = [y for _, y, _ in nodes.values()]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(-0.35, max(ys) + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8, pad=2)
    if note:
        ax.text((min(xs) + max(xs)) / 2, -0.3, note, ha="center", va="top",
                fontsize=5.6, style="italic", color="#555555")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    defs = _defs()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.25))
    for ax, (name, (nodes, edges, note)) in zip(axes.ravel(), defs.items()):
        _draw(ax, nodes, edges, note, name)
    handles = [Patch(facecolor=GEN, label="generate (proposals $L$, planner)"),
               Patch(facecolor=TRA, label="transform (critic, refiner)"),
               Patch(facecolor=MAN, label="aggregate (manager, synth., duel)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.13, wspace=0.05, hspace=0.42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "topologies_v1.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "topologies_v1.png", dpi=220, bbox_inches="tight")
    print("wrote", FIG_DIR / "topologies_v1.pdf")


if __name__ == "__main__":
    main()
