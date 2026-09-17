#!/usr/bin/env python3
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

def build_figure(out_pdf=None):
    if out_pdf is None:
        if len(sys.argv) > 1:
            out_pdf = sys.argv[1]
        else:
            import re
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vs = sorted((d for d in os.listdir(base_dir)
                         if re.fullmatch(r"v\d+", d)
                         and os.path.isdir(os.path.join(base_dir, d, "figures"))),
                        key=lambda d: int(d[1:]))
            sub = os.path.join(vs[-1], "figures") if vs else "figures"
            out_pdf = os.path.join(base_dir, sub, "fig_overview.pdf")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # Exact IEEE single column: width 3.45 in, height 2.85 in
    fig, ax = plt.subplots(figsize=(3.45, 2.85), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 82.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors with matching contour = fill (ec = fc)
    C_TIER_BG     = "#f8fafc"
    C_TIER_HEAD   = "#edf2f7"
    C_TIER_BORDER = "#cbd5e1"
    
    C_AGENT_FC    = "#bae6fd" # sky-200
    C_AGENT_TXT   = "#0369a1" # sky-700
    
    C_LLM_FC      = "#e9d5ff" # purple-200
    C_LLM_TXT     = "#7e22ce" # purple-700
    
    C_TOOL_FC     = "#bbf7d0" # green-200
    C_TOOL_TXT    = "#15803d" # green-700
    
    C_RAG_FC      = "#fed7aa" # orange-200
    C_RAG_TXT     = "#c2410c" # orange-700
    
    C_RED         = "#dc2626"
    C_GRAY        = "#64748b"
    C_TEXT_DARK   = "#0f172a"
    C_SLATE       = "#475569"

    # -------------------------------------------------------------------------
    # 1. Legend Bar at Top (y: 76.2 - 82.0) - STACKED Intra-tier & Cross-tier
    # -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((1.0, 76.2), 98.0, 5.8, boxstyle="round,pad=0.25",
                                fc="#ffffff", ec="#cbd5e1", lw=0.7, zorder=2))

    # Agent circle
    ax.add_patch(Circle((4.2, 79.1), 1.7, fc=C_AGENT_FC, ec=C_AGENT_FC, lw=0, zorder=3))
    ax.text(6.6, 79.1, "Agent ($a_i, c_a$)", va="center", fontsize=4.8, weight="bold", color=C_TEXT_DARK, zorder=4)

    # LLM box
    ax.add_patch(FancyBboxPatch((23.0, 77.6), 2.9, 2.9, boxstyle="round,pad=0.12", fc=C_LLM_FC, ec=C_LLM_FC, lw=0, zorder=3))
    ax.text(26.7, 79.1, "LLM ($E_{\\mathrm{call}}$)", va="center", fontsize=4.8, weight="bold", color=C_TEXT_DARK, zorder=4)

    # Tool box
    ax.add_patch(FancyBboxPatch((42.0, 77.6), 2.9, 2.9, boxstyle="round,pad=0.12", fc=C_TOOL_FC, ec=C_TOOL_FC, lw=0, zorder=3))
    ax.text(45.7, 79.1, "Tool ($E_{\\mathrm{tool}}$)", va="center", fontsize=4.8, weight="bold", color=C_TEXT_DARK, zorder=4)

    # RAG box
    ax.add_patch(FancyBboxPatch((60.5, 77.6), 2.9, 2.9, boxstyle="round,pad=0.12", fc=C_RAG_FC, ec=C_RAG_FC, lw=0, zorder=3))
    ax.text(64.2, 79.1, "RAG ($E_{\\mathrm{ret}}$)", va="center", fontsize=4.8, weight="bold", color=C_TEXT_DARK, zorder=4)

    # Stacked Intra-tier (top) and Cross-tier (bottom)
    ax.plot([79.5, 83.2], [80.3, 80.3], "-", color=C_GRAY, lw=1.3, zorder=3)
    ax.text(84.3, 80.3, "Intra-tier", va="center", fontsize=4.4, color=C_GRAY, zorder=4)

    ax.plot([79.5, 83.2], [77.8, 77.8], "-", color=C_RED, lw=1.6, zorder=3)
    ax.text(84.3, 77.8, "Cross-tier", va="center", fontsize=4.4, weight="bold", color=C_RED, zorder=4)

    # -------------------------------------------------------------------------
    # 2. Three Network Tiers
    # -------------------------------------------------------------------------
    tiers = [
        {"name": "Core Cloud / DC", "spec": "M_u \\geq 80\\,\\mathrm{GB}", "desc": "e.g., A100 / H100 (models $\\geq$ 70B, 27B)", "y0": 54.0, "h": 20.0},
        {"name": "Telco Edge / MEC", "spec": "M_u \\approx 24-64\\,\\mathrm{GB}", "desc": "e.g., AGX Orin / L4 (models 7B–14B)", "y0": 28.5, "h": 20.5},
        {"name": "Far Edge / Device", "spec": "M_u \\leq 8\\,\\mathrm{GB}", "desc": "e.g., Orin Nano (models $\\leq$ 3B–4B)", "y0": 2.0, "h": 21.5}
    ]

    for t in tiers:
        ax.add_patch(FancyBboxPatch((1.0, t["y0"]), 98.0, t["h"], boxstyle="round,pad=0.25",
                                    fc=C_TIER_BG, ec=C_TIER_BORDER, lw=0.75, zorder=1))
        # Header bar
        ax.add_patch(FancyBboxPatch((1.0, t["y0"] + t["h"] - 3.6), 98.0, 3.6, boxstyle="round,pad=0.1",
                                    fc=C_TIER_HEAD, ec=C_TIER_BORDER, lw=0.5, zorder=2))
        ax.text(2.8, t["y0"] + t["h"] - 1.8, t["name"], fontsize=5.6, weight="bold", color="#1e293b", zorder=3, va="center")
        ax.text(32.0, t["y0"] + t["h"] - 1.8, f"${t['spec']}$", fontsize=4.9, color="#475569", zorder=3, va="center")
        ax.text(97.5, t["y0"] + t["h"] - 1.8, t["desc"], fontsize=4.3, style="italic", color="#64748b", zorder=3, va="center", ha="right")

    # Bearer lines and badges
    # Bearer 1: Far Edge <-> Telco Edge (midpoint y = 26.0)
    ax.plot([1.0, 99.0], [26.0, 26.0], color="#0284c7", lw=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.text(74.0, 26.0, "Wireless / 5G RAN Bearer ($\\varepsilon \\approx 10^{-6}\\,\\mathrm{J/b}$)", fontsize=4.5,
            ha="center", va="center", color="#0369a1", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#0284c7", lw=0.5), zorder=3)

    # Bearer 2: Telco Edge <-> Core Cloud (midpoint y = 51.5)
    ax.plot([1.0, 99.0], [51.5, 51.5], color="#0284c7", lw=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.text(21.0, 51.5, "Metro / Optical Bearer ($\\varepsilon \\leq 10^{-7}\\,\\mathrm{J/b}$)", fontsize=4.5,
            ha="center", va="center", color="#0369a1", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#0284c7", lw=0.5), zorder=3)

    # -------------------------------------------------------------------------
    # 3. Agent Nodes & Workflows
    # -------------------------------------------------------------------------
    R_NODE = 2.8

    # --- FAR EDGE TIER ---
    pos_a1 = (7.5, 14.5)
    pos_a2 = (23.0, 14.5)
    
    # Render a1, a2
    ax.add_patch(Circle(pos_a1, R_NODE, fc=C_AGENT_FC, ec=C_AGENT_FC, lw=0, zorder=4))
    ax.text(pos_a1[0], pos_a1[1] + 0.3, "$a_1$", ha="center", va="center", fontsize=5.6, weight="bold", color=C_AGENT_TXT, zorder=5)
    ax.text(pos_a1[0], pos_a1[1] - 1.5, "$c_1$", ha="center", va="center", fontsize=3.7, color="#0284c7", zorder=5)

    ax.add_patch(Circle(pos_a2, R_NODE, fc=C_AGENT_FC, ec=C_AGENT_FC, lw=0, zorder=4))
    ax.text(pos_a2[0], pos_a2[1] + 0.3, "$a_2$", ha="center", va="center", fontsize=5.6, weight="bold", color=C_AGENT_TXT, zorder=5)
    ax.text(pos_a2[0], pos_a2[1] - 1.5, "$c_2$", ha="center", va="center", fontsize=3.7, color="#0284c7", zorder=5)

    # Net-CLI Tool below a1 (ec = fc)
    ax.add_patch(FancyBboxPatch((1.0, 3.6), 13.0, 4.8, boxstyle="round,pad=0.15", fc=C_TOOL_FC, ec=C_TOOL_FC, lw=0, zorder=4))
    ax.text(7.5, 6.6, "Net-CLI Tool", ha="center", va="center", fontsize=4.2, weight="bold", color=C_TOOL_TXT, zorder=5)
    ax.text(7.5, 4.8, "Tool ($E_{\\mathrm{tool}}$)", ha="center", va="center", fontsize=3.7, weight="bold", color=C_TOOL_TXT, zorder=5)
    ax.plot([7.5, 7.5], [8.4, 11.7], color=C_TOOL_TXT, lw=0.8, linestyle=":", zorder=3)

    # Device LLM below a2 (ec = fc)
    ax.add_patch(FancyBboxPatch((15.5, 3.6), 15.0, 4.8, boxstyle="round,pad=0.15", fc=C_LLM_FC, ec=C_LLM_FC, lw=0, zorder=4))
    ax.text(23.0, 6.6, "Device LLM ($E_{\\mathrm{call}}$)", ha="center", va="center", fontsize=3.9, weight="bold", color=C_LLM_TXT, zorder=5)
    ax.text(23.0, 4.8, "Weights $\\leq$ 3B–4B", ha="center", va="center", fontsize=3.5, color=C_LLM_TXT, zorder=5)
    
    # Call link from a2 to Device LLM (vertical dotted line)
    ax.plot([23.0, 23.0], [8.4, 11.7], color=C_LLM_TXT, lw=0.8, linestyle=":", zorder=3)

    # Call link from a1 to Device LLM (dotted line WITHOUT arrow end, as requested)
    ax.plot([9.5, 17.5], [12.5, 8.4], color=C_LLM_TXT, lw=0.8, linestyle=":", zorder=3)

    # Intra-tier hand-off a1 -> a2
    ax.add_patch(FancyArrowPatch(pos_a1, pos_a2, arrowstyle="-|>", mutation_scale=6,
                                 shrinkA=8, shrinkB=8, lw=1.0, color=C_GRAY, zorder=3))

    # --- TELCO EDGE TIER ---
    pos_a3 = (25.0, 37.0)
    ax.add_patch(Circle(pos_a3, R_NODE, fc=C_AGENT_FC, ec=C_AGENT_FC, lw=0, zorder=4))
    ax.text(pos_a3[0], pos_a3[1] + 0.3, "$a_3$", ha="center", va="center", fontsize=5.6, weight="bold", color=C_AGENT_TXT, zorder=5)
    ax.text(pos_a3[0], pos_a3[1] - 1.5, "$c_3$", ha="center", va="center", fontsize=3.7, color="#0284c7", zorder=5)

    # Cross-tier hand-off a2 -> a3 (unobstructed vertical arrow across 5G RAN)
    ax.add_patch(FancyArrowPatch(pos_a2, pos_a3, arrowstyle="-|>", mutation_scale=8,
                                 shrinkA=9, shrinkB=9, lw=1.5, color=C_RED, connectionstyle="arc3,rad=0.02", zorder=3))

    # Telco RAG on left of a3 (ec = fc)
    ax.add_patch(FancyBboxPatch((3.0, 39.5), 16.5, 4.4, boxstyle="round,pad=0.15", fc=C_RAG_FC, ec=C_RAG_FC, lw=0, zorder=4))
    ax.text(11.25, 42.2, "Telco RAG ($E_{\\mathrm{ret}}$)", ha="center", va="center", fontsize=4.2, weight="bold", color=C_RAG_TXT, zorder=5)
    ax.text(11.25, 40.6, "Vector Index Search", ha="center", va="center", fontsize=3.6, color=C_RAG_TXT, zorder=5)
    ax.plot([19.5, 22.2], [41.7, 38.0], color=C_RAG_TXT, lw=0.8, linestyle=":", zorder=3)

    # Edge LLM on left of a3 (ec = fc)
    ax.add_patch(FancyBboxPatch((3.0, 31.5), 16.5, 4.4, boxstyle="round,pad=0.15", fc=C_LLM_FC, ec=C_LLM_FC, lw=0, zorder=4))
    ax.text(11.25, 34.2, "Edge LLM ($E_{\\mathrm{call}}$)", ha="center", va="center", fontsize=4.2, weight="bold", color=C_LLM_TXT, zorder=5)
    ax.text(11.25, 32.6, "Model Weights 8B–14B", ha="center", va="center", fontsize=3.6, color=C_LLM_TXT, zorder=5)
    ax.plot([19.5, 22.2], [33.7, 36.0], color=C_LLM_TXT, lw=0.8, linestyle=":", zorder=3)

    # --- TELCO EDGE EMPTY SPACE NOTE ---
    # Note on right side: Example Chain Topology
    ax.add_patch(FancyBboxPatch((38.0, 31.0), 59.0, 12.5, boxstyle="round,pad=0.25",
                                fc="#ffffff", ec="#cbd5e1", lw=0.8, zorder=3))
    ax.text(67.5, 41.2, "Example Chain Topology",
            ha="center", va="center", fontsize=4.9, weight="bold", color="#0f172a", zorder=4)
    ax.text(67.5, 38.5, "$a_1 \\longrightarrow a_2 \\longrightarrow a_3 \\longrightarrow a_4 \\longrightarrow a_5 \\longrightarrow a_6$",
            ha="center", va="center", fontsize=4.2, weight="bold", color="#0369a1", zorder=4)
    ax.text(67.5, 35.8, "• Distributed sequential pipeline across Far Edge, Telco MEC, and Core Cloud",
            ha="center", va="center", fontsize=3.6, color=C_SLATE, zorder=4)
    ax.text(67.5, 33.2, "• Context $c_a$ accumulates along sequence across multi-step execution",
            ha="center", va="center", fontsize=3.6, color=C_SLATE, zorder=4)

    # --- CORE CLOUD TIER ---
    # Core LLM Engine on the left (ec = fc)
    ax.add_patch(FancyBboxPatch((3.0, 56.5), 32.0, 11.0, boxstyle="round,pad=0.2",
                                fc=C_LLM_FC, ec=C_LLM_FC, lw=0, zorder=3))
    ax.text(19.0, 64.8, "Core LLM Engine ($E_{\\mathrm{call}}$)", ha="center", va="center",
            fontsize=4.8, weight="bold", color=C_LLM_TXT, zorder=5)
    ax.text(19.0, 61.8, "Model weights $\\beta N_{\\mathrm{params}}$ (70B / 27B)", ha="center", va="center",
            fontsize=4.1, weight="bold", color="#581c87", zorder=5)
    ax.text(19.0, 58.8, "Shared serving pool for $a_4, a_5, a_6$", ha="center", va="center",
            fontsize=3.8, color="#581c87", zorder=5)

    pos_a4 = (47.0, 59.0)
    pos_a5 = (65.0, 59.0)
    pos_a6 = (83.0, 59.0)

    # Agents a4, a5, a6
    for pos, label, c_lbl in [(pos_a4, "a_4", "c_4"), (pos_a5, "a_5", "c_5"), (pos_a6, "a_6", "c_6")]:
        ax.add_patch(Circle(pos, R_NODE, fc=C_AGENT_FC, ec=C_AGENT_FC, lw=0, zorder=4))
        ax.text(pos[0], pos[1] + 0.3, f"${label}$", ha="center", va="center", fontsize=5.6, weight="bold", color=C_AGENT_TXT, zorder=5)
        ax.text(pos[0], pos[1] - 1.5, f"${c_lbl}$", ha="center", va="center", fontsize=3.7, color="#0284c7", zorder=5)

    # Cross-tier hand-off from a3 to a4 (across Metro Bearer)
    ax.add_patch(FancyArrowPatch(pos_a3, pos_a4, arrowstyle="-|>", mutation_scale=8,
                                 shrinkA=9, shrinkB=9, lw=1.5, color=C_RED, connectionstyle="arc3,rad=-0.02", zorder=3))

    # Intra-tier hand-offs in Core Cloud: a4 -> a5 -> a6
    ax.add_patch(FancyArrowPatch(pos_a4, pos_a5, arrowstyle="-|>", mutation_scale=6,
                                 shrinkA=8, shrinkB=8, lw=1.0, color=C_GRAY, zorder=3))
    ax.add_patch(FancyArrowPatch(pos_a5, pos_a6, arrowstyle="-|>", mutation_scale=6,
                                 shrinkA=8, shrinkB=8, lw=1.0, color=C_GRAY, zorder=3))

    # Shared LLM Inference Bus along top of Core Cloud (Y = 66.5)
    ax.plot([35.0, 83.0], [66.5, 66.5], color=C_LLM_TXT, lw=0.8, linestyle=":", zorder=3)
    ax.text(56.0, 68.0, "Inference Calls ($E_{\\mathrm{call}}$)", fontsize=3.8, color=C_LLM_TXT, weight="bold", ha="center", zorder=4)

    # Vertical drops from bus to each agent
    for pos in [pos_a4, pos_a5, pos_a6]:
        ax.plot([pos[0], pos[0]], [66.5, pos[1] + R_NODE], color=C_LLM_TXT, lw=0.8, linestyle=":", zorder=3)

    # -------------------------------------------------------------------------
    # 4. AgECAL Workflow Energy Accounting Callout Box (INCREASED SPACING BELOW EQ. 3)
    # -------------------------------------------------------------------------
    # Box spans Y: [2.3, 19.3] (height 17.0 units)
    ax.add_patch(FancyBboxPatch((34.0, 2.3), 64.0, 17.0, boxstyle="round,pad=0.25",
                                fc="#fffbeb", ec="#fde68a", lw=0.8, zorder=4))

    # Title: top of box
    ax.text(66.0, 18.10, "AgECAL WORKFLOW ENERGY ACCOUNTING", ha="center", va="center",
            fontsize=4.7, weight="bold", color="#92400e", zorder=5)

    # Master Equation (Eq. 3): placed at 15.25 (gap to title: 2.85 units; gap to bullet 1: 3.55 units!)
    ax.text(66.0, 14.35, "$E_W = (1+\\gamma_{\\mathrm{v}}) \\, [\\sum E_{\\mathrm{call}} + \\sum E_{\\mathrm{tool}} + \\sum E_{\\mathrm{ret}} + E_{\\mathrm{tx}}] \\quad \\mathbf{(Eq.\\,3)}$",
            ha="center", va="center", fontsize=4.2, color="#1e3a8a", weight="bold", zorder=5)

    # Component Breakdowns: shifted down to give generous 3.55 units gap below Eq. 3
    ax.text(35.5, 11.10, "• LLM Inference ($E_{\\mathrm{call}}$): compute-bound prefill + memory-bound decode",
            fontsize=3.8, color="#78350f", zorder=5)

    ax.text(35.5, 8.95, "• Diagnostics & RAG ($E_{\\mathrm{tool}}, E_{\\mathrm{ret}}$): local tool exec + vector search",
            fontsize=3.8, color="#78350f", zorder=5)

    ax.text(35.5, 6.80, "• Data Transport ($E_{\\mathrm{tx}}$): 7-layer OSI stack (negligible $E_{\\mathrm{tx}} \\ll E_W$)",
            fontsize=3.8, color="#92400e", zorder=5)

    ax.text(35.5, 4.65, "• Memory Feasibility: weights $\\beta N_{\\mathrm{params}}$ + dynamic context $\\gamma \\sum c_a \\leq M_u$ (Eq. 1)",
            fontsize=3.8, color="#78350f", zorder=5)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("Generated:", out_pdf)

if __name__ == "__main__":
    build_figure()
