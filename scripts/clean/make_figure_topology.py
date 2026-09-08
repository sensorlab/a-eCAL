#!/usr/bin/env python3
"""Three network topologies, with an agent team placed on each.

Draws the deployment as network segments -- the view a networking reader expects -- and annotates
each with what the placement model of Section V-B computes: which device hosts how many agents,
how many devices the team needs, what fraction of the all-to-all traffic therefore leaves the box,
and what that costs against the compute it accompanies.

The three topologies are the ones the analysis actually distinguishes:

  (a) core-centralised   all agents in the datacentre. Every model fits, nothing crosses.
  (b) edge-hosted        all agents on one edge accelerator. Inter-agent traffic stays local.
  (c) far-edge split     device memory cannot hold the team, so it fragments across several
                         far-edge boxes and the all-to-all traffic crosses the radio.

An "agent" here is a resident context, not a model instance: every agent on a device shares one
copy of the weights, so a device holds (memory - weights) / (context x KV-per-token) of them.
That matches the measured campaign, whose teams are homogeneous. A heterogeneous team would need
one weight copy per distinct model and the capacities below would fall sharply.

Writes figures/fig_topology.pdf (default: ../../v2/figures).
"""
from __future__ import annotations

import math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

import agentic_ecal as ae
import model_placement as mp
import placement as pl
import osi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "v2", "figures"))

N, ROUNDS = 30, 2
CALIB = ae.TWO_RATE["qwen2_5_7b"]

# (label, hosting device, bearer that carries any inter-agent traffic, model)
TOPOS = [("(a) core-centralised", "a100",     "backbone", "qwen2.5-7b"),
         ("(b) edge-hosted",      "agx_orin", "ftth",     "qwen2.5-7b"),
         ("(c) far-edge split",   "orin_nano","edgecell", "qwen2.5-3b")]

# the segment chain every panel draws, outermost first
CHAIN = [("UE", None), ("far edge", "orin_nano"), ("edge", "agx_orin"), ("core DC", "a100")]
LINKS = [("5G", "5g"), ("FTTH", "ftth"), ("metro", "metro")]


def analyse(dev_key, bearer_key, model_key):
    """Placement outcome for one topology: devices, traffic split, and its cost."""
    m, law = mp.FLEET[model_key], mp.PEER_LAW[model_key]
    dev, bear = mp.DEVICES[dev_key], pl.BEARERS[bearer_key]
    ctx = mp.context_tokens(model_key, N, True)
    if mp.weight_bytes(m) > dev.usable:
        return dict(fits=False, model=model_key)
    per_dev = mp.max_agents(m, dev, ctx)
    devices = math.ceil(N / per_dev) if per_dev else None
    if not per_dev:
        return dict(fits=False, model=model_key)
    grouped = N // devices
    on_box = devices * grouped * (grouped - 1)
    total = N * (N - 1)
    crossing = max(0, total - on_box) * ROUNDS
    bits = crossing * law.b * ae.BITS_PER_TOKEN
    e_tx = (osi.segment_energy(bits, bear.eps, bear.failure_rate)
            + osi.endpoint_stack_energy(bits)) if bits else 0.0
    e_call = CALIB.c_pre * N * (law.solo + ROUNDS * (law.a + law.b * (N - 1)))
    dt = (osi.cascade_bits(bits) / bear.rate + bear.latency) if bits else 0.0
    return dict(fits=True, model=model_key, per_dev=per_dev, devices=devices,
                cross_frac=crossing / (total * ROUNDS), bits=bits,
                e_tx=e_tx, e_call=e_call, dt=dt, bearer=bear, dev=dev)


def panel(ax, title, dev_key, bearer_key, model_key):
    r = analyse(dev_key, bearer_key, model_key)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.set_title(title, fontsize=8, pad=2)

    w, y = 1.62, 3.30
    xs = [0.30 + i * 2.55 for i in range(4)]
    host_i = [i for i, (_, k) in enumerate(CHAIN) if k == dev_key][0]

    for i, (name, key) in enumerate(CHAIN):
        on = (i == host_i)
        ax.add_patch(FancyBboxPatch((xs[i], y), w, 0.80, boxstyle="round,pad=0.05",
                                    fc="#e8f0fb" if on else "white",
                                    ec="C0" if on else "0.65", lw=1.5 if on else 0.9))
        ax.text(xs[i] + w / 2, y + 0.53, name, ha="center", fontsize=6.6,
                weight="bold" if on else "normal")
        if key:
            ax.text(xs[i] + w / 2, y + 0.20, f"{mp.DEVICES[key].memory/mp.GB:.0f} GB",
                    ha="center", fontsize=5.2, color="0.45")

    # links drawn in the gaps, labelled above the boxes so nothing overprints
    for i, (lab, key) in enumerate(LINKS):
        b = pl.BEARERS[key]
        x0, x1 = xs[i] + w, xs[i + 1]
        # the query travels from the UE inward to the hosting tier, so the links it
        # traverses are those *before* that tier, not after it
        carries = (i < host_i)
        ax.plot([x0, x1], [y + 0.40] * 2, "-", lw=1.7 if carries else 0.8,
                color="C0" if carries else "0.78", zorder=0)
        ax.text((x0 + x1) / 2, y + 1.12, lab, ha="center", fontsize=5.6, color="0.3")
        ax.text((x0 + x1) / 2, y + 0.92, f"$\\varepsilon${b.eps:.0e}".replace("e-0", "e-"),
                ha="center", fontsize=4.9, color="0.55")

    if not r["fits"]:
        ax.text(xs[host_i] + w / 2, y - 0.60, "model does not fit",
                ha="center", fontsize=7, color="C3", weight="bold")
        return r

    ax.plot([xs[host_i] + w / 2] * 2, [y, y - 0.36], ":", color="0.6", lw=0.8, zorder=0)
    shown = min(r["devices"], 3)
    bw = w / shown
    for d in range(shown):
        bx = xs[host_i] + d * bw
        ax.add_patch(FancyBboxPatch((bx + 0.03, y - 1.42), bw - 0.06, 1.06,
                                    boxstyle="round,pad=0.02", fc="white", ec="C1", lw=0.9))
        for a in range(min(6, r["per_dev"])):
            ax.add_patch(Circle((bx + bw / 2 + 0.17 * ((a % 3) - 1),
                                 y - 0.68 - 0.30 * (a // 3)), 0.075, fc="C1", ec="none"))
    ax.text(xs[host_i] + w / 2, y - 1.86,
            f"{r['devices']}x {r['dev'].label.splitlines()[0]}, {r['per_dev']}/device",
            ha="center", fontsize=5.4, color="0.35")

    if r["cross_frac"] > 0:
        ax.annotate("", xy=(xs[host_i] + w - 0.10, y - 1.58),
                    xytext=(xs[host_i] + 0.10, y - 1.58),
                    arrowprops=dict(arrowstyle="<->", color="C3", lw=1.0))
        ax.text(xs[host_i] + w / 2, y - 2.28,
                f"{100*r['cross_frac']:.0f}% of peer traffic\ncrosses the "
                f"{r['bearer'].name.split()[0].lower()} link",
                ha="center", fontsize=5.3, color="C3", linespacing=1.35)
    else:
        ax.text(xs[host_i] + w / 2, y - 2.20, "all peer traffic stays on-box",
                ha="center", fontsize=5.6, color="C2")

    ax.text(0.30, 0.10, f"$E_{{\\rm tx}}$/compute {100*r['e_tx']/r['e_call']:.3f}%"
                        f"   added latency {r['dt']*1e3:.0f} ms",
            fontsize=5.8, color="0.2", va="bottom")
    return r


def main(out_dir=OUT_DEFAULT):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    res = [panel(ax, *t) for ax, t in zip(axes, TOPOS)]
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fig_topology.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    for (title, dk, bk, mk), r in zip(TOPOS, res):
        if not r["fits"]:
            print(f"  {title:<22} {mk}: does not fit"); continue
        print(f"  {title:<22} {mk:<12} {r['devices']}x{dk:<10} {r['per_dev']:>4}/dev  "
              f"cross {100*r['cross_frac']:>3.0f}%  E_tx {100*r['e_tx']/r['e_call']:.4f}%  "
              f"dt {r['dt']*1e3:.0f} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
