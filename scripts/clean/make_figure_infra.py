#!/usr/bin/env python3
"""Energy per solved task on the Infra-Bench Kubernetes repair suite.

Measured token counts come from the Infra-Bench pilot (agent-infra/infra_bench_summary.pdf):
Kubeply kubernetes-core tasks on ephemeral k3s clusters through the Harbor harness, 15 tasks
(five easy, five medium, five hard), 60-minute deadline and 50-turn cap per task, binary reward
graded on final cluster state.

CALIBRATION STATUS. The runs use Qwen3.5-9B, for which no measured two-rate coefficients exist;
calls are priced with the Qwen2.5-7B calibration of Table II. Every topology is priced identically,
so the ordering is insensitive to that choice while the absolute joules are not. Cached input is
served from the prefix cache and is not charged prefill; only uncached input is.

Writes figures/fig_infra.pdf (default ../../v2/figures).
"""
from __future__ import annotations

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import agentic_ecal as ae

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "v2", "figures"))

CALIB, BATCH, TASKS = ae.TWO_RATE["qwen2_5_7b"], 64, 15

# (label, score %, cached Mtok, uncached Mtok, output Mtok), Qwen3.5-9B, five runs per topology
MAIN = [("Single",           48.0,  6.488, 0.320, 0.153),
        ("Single+V",         49.3, 12.180, 0.580, 0.345),
        ("PS",               54.7,  6.785, 0.539, 0.315),
        ("PS+V",             44.0, 10.594, 0.776, 0.486),
        ("I+PS+V",           48.0, 18.921, 2.405, 0.286)]
# the earlier proposer sweep, N in {1,3,5}
SWEEP = {"PS":     [(1, 53.3, 0.504, 0.283), (3, 53.3, 0.743, 0.637), (5, 46.7, 1.011, 0.949)],
         "I+PS+V": [(1, 46.7, 2.194, 0.401), (3, 46.7, 2.868, 0.589), (5, 46.7, 2.996, 0.655)]}


def energy(uncached_m, output_m):
    """Joules per run: prefill on uncached input, decode on generated output."""
    return CALIB.c_pre * uncached_m * 1e6 + CALIB.c_dec(BATCH) * output_m * 1e6


def main(out_dir=OUT_DEFAULT):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))

    # (a) the trade itself: score against the energy spent reaching it
    for i, (lab, sc, _c, u, o) in enumerate(MAIN):
        e = energy(u, o) / 1e3
        ax[0].scatter(e, sc, s=34, color=f"C{i}", zorder=3, label=lab)
        ax[0].annotate(lab, (e, sc), xytext=(4, 4), textcoords="offset points", fontsize=5)
    ax[0].set_xlabel("energy per run [kJ]"); ax[0].set_ylabel("tasks solved (%)")
    ax[0].set_title("(a) score against cost", fontsize=8)
    ax[0].set_xlim(0, 90); ax[0].set_ylim(40, 60)

    # (b) the quotient: what one solved task costs
    lab = [m[0] for m in MAIN]
    per = [energy(m[3], m[4]) / 1e3 / (m[1] / 100 * TASKS) for m in MAIN]
    ax[1].bar(range(len(lab)), per, 0.6, color=[f"C{i}" for i in range(len(lab))])
    for x, v in enumerate(per):
        ax[1].annotate(f"{v:.1f}", (x, v), xytext=(0, 2), textcoords="offset points",
                       ha="center", fontsize=5.4)
    ax[1].set_xticks(range(len(lab)))
    ax[1].set_xticklabels(lab, fontsize=5.4, rotation=25, ha="right")
    ax[1].set_ylabel("energy per solved task [kJ]")
    ax[1].set_title("(b) cost of a solved task", fontsize=8)

    # (c) replication within a role does not pay for itself
    for j, (name, rows) in enumerate(SWEEP.items()):
        ns = [r[0] for r in rows]
        pt = [energy(r[2], r[3]) / 1e3 / (r[1] / 100 * TASKS) for r in rows]
        ax[2].plot(ns, pt, "-o", color=f"C{j*3}", ms=4, lw=1.4, label=name)
        for n, p, r in zip(ns, pt, rows):
            ax[2].annotate(f"{r[1]:.0f}%", (n, p), xytext=(3, -7),
                           textcoords="offset points", fontsize=4.8, color="0.35")
    ax[2].set_xticks([1, 3, 5]); ax[2].set_xlabel("proposers $N$")
    ax[2].set_ylabel("energy per solved task [kJ]")
    ax[2].set_title("(c) replication does not pay", fontsize=8)
    ax[2].legend(frameon=False, fontsize=5.5)

    for a in ax:
        a.tick_params(labelsize=6.5); a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fig_infra.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    for l, s_, _c, u, o in MAIN:
        e = energy(u, o)
        print(f"  {l:<10} {s_:>5.1f}%  {e/1e3:>6.1f} kJ/run  {e/1e3/(s_/100*TASKS):>6.1f} kJ/solved")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
