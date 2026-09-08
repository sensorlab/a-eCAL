#!/usr/bin/env python3
"""Placement capacity maps: what fits where, and what it costs to make them talk.

Panel (a)  accelerator memory x model weight, coloured by how many of the 16 measured
           open-weight models will load at all.
Panel (b)  the same grid, coloured by how many agents fit alongside the weights at the
           measured N=30 full-mesh debate context. An agent is a resident context, not a
           model copy: all agents on a device share one set of weights.
Panel (c)  orchestration architecture x transport technology, coloured by inter-agent
           transmission energy at N=30 calls.

ARCHITECTURES AND EDGE COUNTS. The eight architectures are those of the orchestration study,
whose first-tier proposal counts at N=30 it states directly: L=29 for both stars, 15 for
Proposer-Critic and Tournament, 22 for Tree, 23 for Diamond, and 1 for both chains. Edge counts
below follow from those L values and each architecture's stated wiring; they are DERIVED, not
measured, and the two marked "approx" involve a judgement about the transform tier.

Writes fig_capacity.pdf (default ../../v2/figures).
"""
from __future__ import annotations

import math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import agentic_ecal as ae
import model_placement as mp
import placement as pl
import osi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "v2", "figures"))

TIERS = ["orin_nano", "l4", "rtx4090", "agx_orin", "a100"]      # ascending usable memory
TRANSPORTS = ["metro", "ftth", "5g", "edgecell"]
MSG_TOKENS = 120        # per-message budget of the measured campaign

# (name, inter-agent edges at N=30, how the count follows from the stated wiring)
# Inter-agent edges as a function of team size, from each architecture's stated construction.
# Total prefill is N*p0 + E*tau, so a DAG's shape does not affect cost; only its edge count does.
def edges(t, N):
    if t in ("Star", "Persona-Star", "Chain"):
        return N - 1
    if t == "Proposer-Critic":
        L = math.ceil((N - 1) / 2); return L + (N - 1 - L)
    if t == "Tournament":
        L = (N + 1) // 2; return 2 * (L - 1)
    if t == "Tree":
        L = max(1, round(N * 0.73)); e = 0; n = L
        while n > 1:
            e += n; n = math.ceil(n / 3)
        return e
    if t == "Diamond":
        P = max(1, N // 5); return 2 * (N - P - 1)
    if t == "Cascading-Chain":
        return sum(min(5, i) for i in range(1, N))
    raise KeyError(t)


ARCH = ["Star", "Persona-Star", "Proposer-Critic", "Tournament", "Tree", "Chain",
        "Diamond", "Cascading-Chain"]
SWEEP_N = [5, 7, 10, 15, 20, 25, 30]
BINS = [(0, 8, "$<$8"), (8, 16, "8--16"), (16, 32, "16--32"), (32, 64, "32--64")]
TEAM = 10     # mid-range operating point; all eight architectures are defined for N>=5

# The full measured set, quantised checkpoints included. A four-bit 70B occupies a light weight
# class while carrying the widest KV cache in the set, which is precisely the case that separates
# panel (a) from panel (b).
FLEET = mp.FLEET


def grids():
    """Matrices over (tier, weight bin): models that load, agents they hold, and class size.

    The class populations are the denominator panel (a) needs. They are 3, 7, 3, 3 -- not equal --
    so a raw count of 3 and a raw count of 7 can both mean "every model in the class".
    """
    fits = np.zeros((len(TIERS), len(BINS)))
    agents = np.full((len(TIERS), len(BINS)), np.nan)
    pops = np.zeros(len(BINS))
    for i, tk in enumerate(TIERS):
        dev = mp.DEVICES[tk]
        for j, (lo, hi, _) in enumerate(BINS):
            inbin = [k for k, m in FLEET.items()
                     if lo <= mp.weight_bytes(m) / mp.GB < hi]
            pops[j] = len(inbin)
            ok = [k for k in inbin if mp.weight_bytes(FLEET[k]) <= dev.usable]
            fits[i, j] = len(ok)
            if ok:
                # an operator runs many concurrent teams, so report sessions, not agents
                caps = [mp.max_agents(FLEET[k], dev, mp.context_tokens(k, TEAM, True)) // TEAM
                        for k in ok]
                caps = [c for c in caps if c > 0]
                if caps:
                    agents[i, j] = float(np.median(caps))
    return fits, agents, pops


def heat(ax, M, title, cbar_label, fmt, log=False, cmap="viridis", annot=True, denoms=None):
    """pcolormesh rather than imshow: imshow resamples in vector output and stripes the cells.

    ``denoms`` gives the per-column population. Supplied, cells are coloured by the FRACTION n/N
    and labelled "n/N" rather than by the raw count. Colouring counts on a shared scale made the
    3-model classes render permanently fainter than the 7-model one even where both were fully
    satisfied, which reads as though fewer light models fit than medium ones -- the opposite of
    the truth. With a fraction the four columns are directly comparable.
    """
    C = M if denoms is None else M / np.asarray(denoms, float)[None, :]
    D = np.ma.masked_invalid(C)
    norm = matplotlib.colors.LogNorm(vmin=max(np.nanmin(C), 1), vmax=np.nanmax(C)) if log else None
    ny, nx = M.shape
    mesh = ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), D, cmap=cmap, norm=norm,
                         shading="flat", edgecolors="white", linewidth=0.4)
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_xticklabels([b[2] for b in BINS], fontsize=5.6, rotation=28, ha="right")
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_yticklabels([f"{mp.DEVICES[t].label.splitlines()[0]}\n{mp.DEVICES[t].memory/mp.GB:.0f} GB"
                        for t in TIERS], fontsize=5.6)
    ax.set_xlabel("model weight [GB]", fontsize=7)
    ax.set_title(title, fontsize=8)
    if annot:
        hi = np.nanmax(C)
        for i in range(ny):
            for j in range(nx):
                v, c = M[i, j], C[i, j]
                # With a denominator, 0 is informative ("0/7"), so only NaN is blank.
                if np.isnan(v) or (v == 0 and denoms is None):
                    ax.text(j + .5, i + .5, "--", ha="center", va="center",
                            fontsize=6, color="0.45")
                else:
                    shade = (np.log10(max(c, 1)) / np.log10(hi)) if log else (c / hi)
                    label = f"{v:.0f}/{denoms[j]:.0f}" if denoms is not None else fmt(v)
                    ax.text(j + .5, i + .5, label, ha="center", va="center", fontsize=5.8,
                            color="white" if shade < 0.55 else "black")
    cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=6)
    # tick_params defaults to which="major", so under LogNorm the minor labels (2x10^0, 3x10^0, ...)
    # kept the default font size and overprinted each other. Label decades only.
    cb.ax.tick_params(labelsize=5.5, which="both")
    if log:
        cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def tx_energy(edges, transport):
    """Transmission energy of one session's inter-agent messages over one bearer."""
    bits = edges * MSG_TOKENS * ae.BITS_PER_TOKEN
    b = pl.BEARERS[transport]
    return osi.segment_energy(bits, b.eps, b.failure_rate) + osi.endpoint_stack_energy(bits)


def main(out_dir=OUT_DEFAULT):
    fits, agents, pops = grids()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    heat(ax[0], fits, "(a) models that load", "fraction of weight class",
         lambda v: f"{v:.0f}", denoms=pops)
    heat(ax[1], agents, f"(b) concurrent $N={TEAM}$ sessions", "sessions",
         lambda v: f"{v:,.0f}", log=True, cmap="magma")
    for i, t in enumerate(ARCH):
        e = [edges(t, n) for n in SWEEP_N]
        cls = 2 if t == "Cascading-Chain" else (1 if t == "Diamond" else 0)
        ax[2].plot(SWEEP_N, e, marker="os^"[cls], ms=3.4, lw=1.5 if cls else 0.9,
                   color=["0.55", "C1", "C3"][cls], alpha=1.0 if cls else 0.75,
                   label=t if cls or i == 0 else None, zorder=3 - cls)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xticks(SWEEP_N); ax[2].set_xticklabels(SWEEP_N, fontsize=6)
    ax[2].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_xlabel("team size $N$", fontsize=7)
    ax[2].set_ylabel("inter-agent edges", fontsize=7)
    ax[2].set_title("(c) topology as a traffic lever", fontsize=8)
    ax[2].tick_params(labelsize=6.5); ax[2].grid(alpha=0.25, lw=0.4)
    ax[2].annotate("six architectures\ncoincide at $N-1$", xy=(15, 14), xytext=(5.2, 30),
                   fontsize=5.2, color="0.4",
                   arrowprops=dict(arrowstyle="->", color="0.55", lw=0.7))
    ax[2].legend(fontsize=5.2, frameon=False, loc="upper left")
    ax[2].axvline(TEAM, color="0.75", ls=":", lw=0.8, zorder=0)

    ax[0].set_ylabel("accelerator", fontsize=7)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fig_capacity.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    print("\n(a) models that load / (b) median agents hosted")
    print(f"{'tier':<22}" + "".join(f"{b[2] + f' (n/{int(pops[j])})':>14}"
                                      for j, b in enumerate(BINS)))
    for i, t in enumerate(TIERS):
        row = ""
        for j in range(len(BINS)):
            sess = "--" if np.isnan(agents[i, j]) else f"{round(agents[i,j])}"
            row += f"{int(fits[i,j])}/{int(pops[j])}  {sess}".rjust(14)
        print(f"{mp.DEVICES[t].name:<22}{row}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
