#!/usr/bin/env python3
"""Placement analysis for agentic workflows: where the agents live, and what it costs.

Companion to ``agentic_ecal.py``. Four things the per-invocation percentage table cannot say:

  1. STATE MIGRATION. What a distributed agent actually needs from its predecessor is the KV
     cache, not the transcript. It can be shipped or recomputed, and unlike the transcript
     hand-off that trade has a crossover inside the range of deployed bearers.
  2. HETEROGENEOUS PATHS. A far-edge-to-core path is a concatenation of segments -- radio, then
     copper or fibre access, then metro and core fibre -- not one bearer repeated per hop.
     Charging the worst segment to every hop overstates the cost several-fold.
  3. OFFERED LOAD. An operator sizes links in bit/s at a site, not in percent of one query.
  4. INCREMENTAL vs ATTRIBUTIONAL. A lit router draws essentially the same power whether or not
     it forwards the packet; a radio does not. The two accountings differ by orders of magnitude
     and the distinction belongs in the table, not in a footnote.

Energies are Joules, rates bit/s, latencies seconds unless named otherwise.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import agentic_ecal as ae
import osi

__all__ = ["Bearer", "BEARERS", "ARCHETYPES", "PLACEMENTS", "PathCost",
           "path_cost", "offered_load",
           "handoff_tokens", "deep_path_energy"]


@dataclass(frozen=True)
class Bearer:
    """One network segment.

    ``eps`` is the attributional energy intensity in J/bit -- link power divided by achieved
    throughput -- which is the convention of the eCAL transmission model and of the sources cited
    in the manuscript.

    ``lit`` marks a segment whose power is essentially load-independent: an already-powered router
    or line card forwards one more packet for close to nothing, so its *incremental* intensity is
    far below ``eps``. Radio segments carry no such slack, since a transmitter draws power while
    it transmits. ``eps_incremental`` is therefore 0 for lit segments and ``eps`` otherwise.

    CITATION STATUS: ``eps`` values are those already cited in the manuscript. ``rate`` and
    ``latency`` are representative planning figures used only for the latency axis and the
    link-sizing view; they are not load-bearing for any energy claim and should be replaced with
    cited values before submission.
    """
    name: str
    eps: float             # J/bit, attributional
    rate: float            # bit/s, usable one-way throughput
    latency: float         # s, one-way traversal (propagation + typical queueing)
    lit: bool = True       # load-independent power (fixed/optical) vs load-proportional (radio)
    failure_rate: float = 0.0   # expected retransmissions = 1/(1-failure_rate), Eq. (3) repo

    @property
    def eps_incremental(self) -> float:
        return 0.0 if self.lit else self.eps


# eps: manuscript-cited. rate/latency: representative, see CITATION STATUS above.
BEARERS: Dict[str, Bearer] = {
    "backbone":  Bearer("Backbone fibre",     1e-8, 100e9, 5.0e-3,  lit=True),
    "metro":     Bearer("Metro fibre",        1e-8,  10e9, 2.0e-3,  lit=True),
    "ftth":      Bearer("FTTH / copper access", 1e-7, 1e9, 3.0e-3,  lit=True),
    "5g":        Bearer("5G RAN",             1e-6, 100e6, 10.0e-3, lit=False),
    "edgecell":  Bearer("Loaded edge cell",   1e-5,  20e6, 25.0e-3, lit=False),
    "nbiot":     Bearer("NB-IoT uplink",      1e-3, 100e3, 1.0,     lit=False),
}

# A deployment archetype assigns one bearer to each agent boundary, ordered outward from the far
# edge. The homogeneous rows of the original table are the special case of a repeated bearer.
ARCHETYPES: Dict[str, Tuple[str, str, str]] = {
    "All-fibre (fixed CPE)":   ("ftth",     "metro", "backbone"),
    "5G consumer":             ("5g",       "ftth",  "metro"),
    "Loaded cell":             ("edgecell", "ftth",  "metro"),
    "NB-IoT industrial":       ("nbiot",    "ftth",  "metro"),
    "Core-only (all in DC)":   ("metro",    "metro", "backbone"),
}

# Which agent boundaries actually cross a network segment under each placement.
PLACEMENTS: Dict[str, Tuple[bool, bool, bool]] = {
    "co-located":        (False, False, False),
    "split once":        (False, True,  False),
    "fully distributed": (True,  True,  True),
}


@dataclass
class PathCost:
    e_tx: float                 # J, attributional
    e_tx_incremental: float     # J, incremental (lit segments free)
    latency: float              # s added by transmission
    per_segment: Sequence[float]   # attributional J per boundary
    bits: Sequence[float]          # bits crossing each boundary


def handoff_tokens(wf: ae.Workflow, cumulative: bool = True) -> Sequence[int]:
    """Tokens crossing each agent boundary.

    Two protocol conventions, and the choice between them is the dominant factor in whether
    communication matters for deep loops:

      ``cumulative`` -- each hand-off re-ships the whole accumulated transcript, as a stateless
        agent endpoint must. Volume grows with the square of the loop depth.
      otherwise -- each hand-off ships only the tokens produced since the recipient last saw the
        conversation. Volume grows linearly.

    They coincide for a single round, which is why the placement table is insensitive to the
    distinction and the depth sweep is not.
    """
    history, cumulative_out, delta_out = 0, [], []
    for st in wf.steps:
        added = st.p_out + (st.tool.obs_tokens if st.tool else 0)
        cumulative_out.append(history)
        delta_out.append(added)
        history += added
    return cumulative_out[1:] if cumulative else delta_out[:-1]


def deep_path_energy(wf: ae.Workflow, archetype: str, cumulative: bool = True,
                     bits_per_token: float = ae.BITS_PER_TOKEN) -> float:
    """Transmission energy [J] for a fully distributed team run over several rounds.

    The four agents occupy successive sites, so within a round the three hand-offs traverse the
    access, aggregation and core segments in turn; the hand-off from the last agent back to the
    first at the start of the next round retraces the whole path, and is charged as the sum of the
    three. ``path_cost`` covers the single-round case and is left untouched.
    """
    segs = [BEARERS[k] for k in ARCHETYPES[archetype]]
    cycle = [(s.eps, s.failure_rate) for s in segs] + [(sum(s.eps for s in segs),
                                                       max(s.failure_rate for s in segs))]
    total = 0.0
    for j, tok in enumerate(handoff_tokens(wf, cumulative)):
        b = bits_per_token * tok
        eps, fr = cycle[j % 4]
        total += osi.segment_energy(b, eps, fr) + osi.endpoint_stack_energy(b)
    return total


def path_cost(wf: ae.Workflow, archetype: str, placement: str,
              bits_per_token: float = ae.BITS_PER_TOKEN) -> PathCost:
    """Transmission energy and added latency for one (archetype, placement) pair."""
    segs = [BEARERS[k] for k in ARCHETYPES[archetype]]
    crossed = PLACEMENTS[placement]
    toks = handoff_tokens(wf)

    e_attr, e_incr, lat, per_seg, bits = 0.0, 0.0, 0.0, [], []
    for seg, tok, does_cross in zip(segs, toks, crossed):
        b = bits_per_token * tok if does_cross else 0.0
        bits.append(b)
        # eCAL Eq. (3): the link carries the CASCADED bit count, not the payload, and the two
        # hosts terminating the hand-off run the L2..L7 stack over it. Endpoint processing is
        # charged once per crossing and is real computation, so it survives incremental
        # accounting even where the link itself is load-independent.
        e_link = osi.segment_energy(b, seg.eps, seg.failure_rate)
        e_link_incr = osi.segment_energy(b, seg.eps_incremental, seg.failure_rate)
        e_host = osi.endpoint_stack_energy(b) if does_cross else 0.0
        per_seg.append(e_link + e_host)
        e_attr += e_link + e_host
        e_incr += e_link_incr + e_host
        if does_cross:
            lat += seg.latency + osi.cascade_bits(b) / seg.rate
    return PathCost(e_attr, e_incr, lat, per_seg, bits)


def offered_load(wf: ae.Workflow, archetype: str, placement: str,
                 sessions_per_s: float) -> dict:
    """Aggregate hand-off traffic a site must carry, in bit/s per segment."""
    pc = path_cost(wf, archetype, placement)
    return {
        "per_segment_bps": [b * sessions_per_s for b in pc.bits],
        "total_bps": sum(pc.bits) * sessions_per_s,
        "power_w": pc.e_tx * sessions_per_s,
        "power_w_incremental": pc.e_tx_incremental * sessions_per_s,
    }


# ---------------------------------------------------------------------------
# LaTeX table generation, so the manuscript's placement numbers are produced
# rather than asserted
# ---------------------------------------------------------------------------

_SHORT = {"backbone": "backbone", "metro": "metro", "ftth": "FTTH", "5g": "5G",
          "edgecell": "cell", "nbiot": "NB-IoT"}


def _pct(v: float) -> str:
    if v == 0:
        return "$0$"
    if v < 1e-3:
        return "$<\\!0.001$"
    return f"${v:.3f}$" if v < 1 else f"${v:.2f}$"


def latex_table(wf: ae.Workflow, path: str) -> str:
    """Write the placement table. Rows are deployment archetypes, not single bearers."""
    e_w = wf.run()["e_total"]
    rows = []
    for arch, segs in ARCHETYPES.items():
        one = path_cost(wf, arch, "split once")
        full = path_cost(wf, arch, "fully distributed")
        access = 100.0 * full.per_segment[0] / full.e_tx if full.e_tx else 0.0
        rows.append(
            f"{arch} & {' / '.join(_SHORT[k] for k in segs)} & "
            f"{_pct(100 * one.e_tx / e_w)} & {_pct(100 * full.e_tx / e_w)} & "
            f"{_pct(100 * full.e_tx_incremental / e_w)} & {access:.0f}\\% & "
            f"{full.latency * 1e3:.0f} \\\\")

    body = "\n".join(rows)
    tex = f"""% Generated by scripts/clean/placement.py -- do not edit by hand.
\\begin{{table*}}[t]
\\caption{{Cost of distributing the four-agent workflow over composed network paths.}}
\\label{{tab:placement}}
\\centering
\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\\toprule
 &  & \\multicolumn{{3}}{{c}}{{$E_{{\\mathrm{{tx}}}}$ as \\% of $E_W$}} & & \\\\
\\cmidrule(lr){{3-5}}
Deployment & Path (access$\\to$core) & 1 hop, attr. & 3 hops, attr. & 3 hops, incr. & access share & $\\Delta t$ [ms] \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
\\parbox{{\\textwidth}}{{\\footnotesize Co-located placement is $0$ in every row and is omitted.
Attributional accounting charges link power over achieved throughput; incremental charges only
load-proportional segments, since an already-lit router forwards one further packet for close to
nothing. ``Access share'' is the first segment's fraction of $E_{{\\mathrm{{tx}}}}$.
$E_W = {e_w:.0f}$\\,J at the single-stream operating point.}}
\\end{{table*}}
"""
    with open(path, "w") as fh:
        fh.write(tex)
    return tex


if __name__ == "__main__":
    import os
    _hw = dataclasses.replace(ae.HW["h100"], mfu=ae.MFU_LATENCY)
    _wf = ae.four_agent_rag_workflow(hw=_hw)
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _out = os.path.join(_root, "tables")
    os.makedirs(_out, exist_ok=True)
    latex_table(_wf, os.path.join(_out, "tab_placement.tex"))
    print("wrote", os.path.join(_out, "tab_placement.tex"))
    _r = _wf.run()
    print(f"E_W={_r['e_total']:.0f} J")
