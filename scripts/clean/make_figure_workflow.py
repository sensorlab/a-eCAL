#!/usr/bin/env python3
"""Schematic and token composition of the four-agent RAG case study.

Kept separate from ``make_figures.py`` so regenerating it cannot alter any figure the frozen
manuscript builds from. Writes figures/fig_workflow.pdf.

The manuscript refers to "the four-agent RAG workflow" throughout but never shows it. This
figure supplies the missing definition: who calls whom, what each step contributes to the
prompt, and where the accumulating transcript enters.

The composition is recomputed here exactly as ``Workflow.run`` computes it --
p_in = sys_tokens + p_in_local + retrieved + carried -- so the panel cannot drift from the
model that prices it.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import agentic_ecal as ae
import osi
import placement as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(ROOT, "figures")

# colours for the four prompt sources, in the order they are stacked
SRC = [("system",    "0.80"),
       ("local",     "C0"),
       ("retrieved", "C2"),
       ("history",   "C1")]

CALIB = ae.TWO_RATE["llama3_8b"]     # the case study's model, measured coefficients
# b=1 lies below the fitted range b in [2,256]; Eq. (5) is derived rather than fitted, so
# extrapolating is principled, but the caption says so.
REGIMES = [(1, "$b=1$", "0.62"), (64, "$b=64$", "C0"), (256, "$b=256$", "C2")]
# every bearer of Table III except the NB-IoT uplink, which the manuscript excludes
BEARERS = ("metro", "ftth", "5g", "edgecell")
SHORT_BEARER = {"metro": "metro", "ftth": "FTTH", "5g": "5G", "edgecell": "edge cell"}


def decompose(wf: ae.Workflow):
    """Per-step prompt composition, mirroring Workflow.run exactly."""
    rs = max(wf.round_size, 1)
    transcript, rows = [], []
    for k, st in enumerate(wf.steps):
        rnd = k // rs
        added = st.retrieval.added_context if st.retrieval is not None else 0
        carried = 0
        if wf.carry_history:
            lo = 0 if wf.history_rounds is None else rnd - wf.history_rounds
            carried = sum(tok for (r, ag, tok) in transcript
                          if lo <= r < rnd and not (wf.exclude_self and ag == st.agent))
        rows.append(dict(agent=st.agent, system=wf.sys_tokens, local=st.p_in_local,
                         retrieved=added, history=carried,
                         p_in=wf.sys_tokens + st.p_in_local + added + carried,
                         p_out=st.p_out,
                         obs=st.tool.obs_tokens if st.tool else 0))
        if st.tool is not None:
            transcript.append((rnd, st.agent, st.tool.obs_tokens))
        transcript.append((rnd, st.agent, st.p_out))
    return rows


def step_energies(rows, batch):
    """Per-step prefill/decode energy [J] at a given serving batch, via Eq. (3)."""
    out = []
    for r in rows:
        pre = CALIB.c_pre * r["p_in"]
        dec = CALIB.c_dec(batch) * r["p_out"]
        out.append(dict(agent=r["agent"], pre=pre, dec=dec, e=pre + dec,
                        jpb=(pre + dec) / (r["p_out"] * ae.BITS_PER_TOKEN)))
    return out


def workflow_energy(batch):
    """E_W and the workflow metric at a given batch, from the model itself."""
    wf = ae.four_agent_rag_workflow(hw=ae.HW["a100"])
    wf.serving_batch = batch
    wf.calib = CALIB
    r = wf.run()
    return r["e_total"], r["e_total"] / (r["useful_output_tokens"] * ae.BITS_PER_TOKEN)


def ecal_panel(ax, rows):
    """agentic-eCAL per agent. The workflow total is quoted in the text, not plotted:
    it is not comparable to a per-agent bar, carrying tool, retrieval and orchestration."""
    labels = [r["agent"] for r in rows]
    x = list(range(len(labels))); w = 0.27
    for i, (batch, lab, col) in enumerate(REGIMES):
        vals = [d["jpb"] for d in step_energies(rows, batch)]
        ax.bar([xx + (i - 1) * w for xx in x], vals, w, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=20, ha="right")
    ax.set_ylabel("agentic-eCAL [J/bit]", fontsize=7.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=3,
              handlelength=0.9, columnspacing=0.6)
    ax.set_ylim(top=2.0)
    # the retriever is the outlier, and only once the workflow is batched
    d = step_energies(rows, 64)
    ratio = d[1]["jpb"] / d[0]["jpb"]
    ax.annotate(f"{ratio:.1f}$\\times$", xy=(1, d[1]["jpb"]), xytext=(0, 5),
                textcoords="offset points", fontsize=6.5, color="C0", ha="center")


def transmission_panel(ax, wf):
    """Hand-off energy over each bearer of Table III, against the compute it accompanies."""
    toks = pl.handoff_tokens(wf)
    names, vals = [], []
    for key in BEARERS:
        b = pl.BEARERS[key]
        e = 0.0
        for t in toks:
            bits = ae.BITS_PER_TOKEN * t
            e += osi.segment_energy(bits, b.eps, b.failure_rate) + osi.endpoint_stack_energy(bits)
        names.append(SHORT_BEARER[key]); vals.append(e)
    ax.bar(range(len(vals)), vals, 0.6, color="C2")
    # b=64 and b=256 differ by only 18%, so their reference lines would overprint; bracket
    # the range with the single-stream extreme and the production operating point instead.
    for batch, lab, col in REGIMES[:2]:
        ew = workflow_energy(batch)[0]
        ax.axhline(ew, color=col, ls="--", lw=1.0)
        ax.text(-0.42, ew * 1.5, f"$E_W$, {lab}", fontsize=5.8, color=col, ha="left")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6.4, rotation=20, ha="right")
    ax.set_ylabel("energy [J]", fontsize=7.5)
    ax.set_ylim(min(vals) * 0.25, workflow_energy(1)[0] * 60)
    worst = max(vals) / workflow_energy(64)[0]
    ax.annotate(f"$\\leq{100*worst:.2f}\\%$ of $E_W$", xy=(1.5, max(vals) * 2.2),
                fontsize=6.2, color="C2", ha="center")


def schematic(ax, rows):
    """Boxes left to right; both attachments on an upper row, the transcript arc below."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.75); ax.axis("off")
    w, h, y = 1.9, 0.72, 1.42
    y_att = 2.55                      # attachment row, clear of the boxes
    xs = [0.35 + i * 2.42 for i in range(len(rows))]
    for x, r in zip(xs, rows):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc="white", ec="C0", lw=1.1))
        ax.text(x + w / 2, y + h * 0.62, r["agent"], ha="center", va="center",
                fontsize=7.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.22, f"{r['p_in']}$\\to${r['p_out']}",
                ha="center", va="center", fontsize=6.3, color="0.35")
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + 2.42, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color="0.45", lw=0.9))

    def attach(i, label, col, face):
        ax.add_patch(FancyBboxPatch((xs[i], y_att), w, 0.60, boxstyle="round,pad=0.04",
                                    fc=face, ec=col, lw=0.9))
        ax.text(xs[i] + w / 2, y_att + 0.30, label, ha="center", va="center",
                fontsize=6.0, color=col, linespacing=1.25)
        ax.add_patch(FancyArrowPatch((xs[i] + w / 2, y_att), (xs[i] + w / 2, y + h),
                                     arrowstyle="-|>", mutation_scale=7, color=col, lw=0.9))

    attach(1, "index\n$5\\times256$ tok", "C2", "#eaf5ea")
    attach(2, "tool\n$150$ tok", "C1", "#fdf0e3")

    # every later step re-reads what the earlier ones produced
    ax.add_patch(FancyArrowPatch((xs[0] + w / 2, y), (xs[3] + w / 2, y),
                                 connectionstyle="arc3,rad=0.13", arrowstyle="-|>",
                                 mutation_scale=7, color="0.55", lw=0.9, ls="--"))
    ax.text(5.2, 0.46, "accumulated transcript", fontsize=6.3, color="0.45", ha="center",
            va="center", bbox=dict(fc="white", ec="none", pad=1.2))


def composition(ax, rows):
    xs = range(len(rows))
    bottom = [0.0] * len(rows)
    for key, col in SRC:
        vals = [r[key] for r in rows]
        ax.bar(xs, vals, 0.62, bottom=bottom, color=col, label=key,
               edgecolor="white", lw=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.plot(list(xs), [r["p_out"] for r in rows], "o--", color="C3", ms=3.2, lw=1.0,
            label="output")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["agent"] for r in rows], fontsize=6.4, rotation=20, ha="right")
    ax.set_ylabel("tokens", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.6, frameon=False, ncol=3, loc="upper center",
              handlelength=1.0, columnspacing=0.8, borderpad=0.1)
    ax.set_ylim(top=max(bottom) * 1.55)


def main():
    wf = ae.four_agent_rag_workflow()
    rows = decompose(wf)
    r = wf.run()
    assert sum(x["p_in"] for x in rows) == r["prefill_tokens"], "composition drifted from run()"

    fig = plt.figure(figsize=(7.1, 4.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.0], hspace=0.55, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    schematic(ax_a, rows)
    composition(ax_b, rows)
    ecal_panel(ax_c, rows)
    transmission_panel(ax_d, wf)

    ax_a.set_title("(a) the four-agent RAG workflow", fontsize=8)
    ax_b.set_title("(b) prompt composition", fontsize=8)
    ax_c.set_title("(c) energy per useful bit", fontsize=8)
    ax_d.set_title("(d) hand-off energy by bearer", fontsize=8)
    for a in (ax_b, ax_c, ax_d):
        a.tick_params(labelsize=6.6)
        a.grid(axis="y", alpha=0.25, lw=0.4)

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_workflow.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    print(f"prefill {r['prefill_tokens']} tok, decode {r['decode_tokens']} tok")
    for x in rows:
        print(f"  {x['agent']:<10} p_in={x['p_in']:>5} (sys {x['system']}, local {x['local']}, "
              f"ret {x['retrieved']}, hist {x['history']})  p_out={x['p_out']}")
    for batch, lab, _ in REGIMES:
        ew, jpb = workflow_energy(batch)
        d = step_energies(rows, batch)
        per = "  ".join(f"{x['agent'][:4]} {x['jpb']:.4f}" for x in d)
        print(f"  {lab:<14} E_W={ew:>7.1f} J  eCAL={jpb:.4f} J/bit   per-agent: {per}")
    toks = pl.handoff_tokens(wf)
    print(f"  hand-off tokens {toks}")
    for key in BEARERS:
        b = pl.BEARERS[key]
        e = sum(osi.segment_energy(ae.BITS_PER_TOKEN * t, b.eps, b.failure_rate)
                + osi.endpoint_stack_energy(ae.BITS_PER_TOKEN * t) for t in toks)
        print(f"    {b.name:<22} E_tx={e:.4f} J  "
              f"({100*e/workflow_energy(64)[0]:.4f}% of E_W at b=64)")


if __name__ == "__main__":
    main()
