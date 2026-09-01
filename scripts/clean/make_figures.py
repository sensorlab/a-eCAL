#!/usr/bin/env python3
"""Generate the agentic-eCAL figures from the single-rate model in ``agentic_ecal.py``.

Five figures: fig_scaling, fig_validation, fig_tokens (measured), fig_components and
fig_amortization (model only). The measured ones need the CSVs; point --data at them or set
AGENTIC_ECAL_DATA, otherwise they are skipped with a message rather than an error.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_ecal as ae  # noqa: E402
import placement as pl  # noqa: E402
import osi  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
# The repo-root figures/ that main.tex includes, so a bare run updates the paper's figures.
OUT_DEFAULT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "figures"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# eCAL overlay: A100 unchanged (400 W nameplate, 312 TFLOP/s), varying only mfu.
# Power and utilisation are interchangeable here since E ~ P/eta, so 400 W at eta equals
# 294 W at 0.735*eta -- 294 W being the board power actually drawn during the measured runs.
# An eta label therefore means "utilisation at nameplate power", the paper's own convention.
HW_OVERLAY = ae.HW["a100"]
BOARD_POWER_MEAS = 296.0   # W, median draw derived from the measurements themselves

# The case study is stated on H100 at eta = 0.05 in Section IV; fig_components and
# fig_amortization share it, so the component split and the amortization floor describe one regime.
# The model-only figures price calls with the SAME measured coefficients as fig_validation, so
# every model estimate in the paper rests on one calibration. Those coefficients were measured on
# the A100, so the case study is an A100 case study.
CASE_HW = ae.HW["a100"]
CASE_BATCH = 64            # b in Eq. (3)
CASE_CALIB = ae.TWO_RATE["llama3_8b"]
ETAS = (0.25, 0.30, 0.60, 0.70)
ETA_GREYS = ("0.80", "0.62", "0.40", "0.10")   # light -> dark with increasing eta

# Amortization: single-rate at one eta for ALL three curves, so the comparison is method-uniform.
AMORT_ETA = 0.05
AMORT_MODELS = (
    ("Llama-3 8B",  "llama3_8b",  "#1f77b4", "-"),
    ("LLaMA-65B",   "llama_65b",  "#2ca02c", "--"),
    ("Llama-3 70B", "llama3_70b", "#d62728", "-."),
)

# Measured models: (label, LLMS key, agents CSV, depth CSVs, colour, marker)
MEASURED = (
    ("Qwen2.5-7B", "qwen2_5_7b", "agents_Qwen2.5-7B-Instruct.csv",
     ("depth_carry_N1_Q256_Qwen2.5-7B-Instruct.csv",
      "depth_free_N1_Q256_Qwen2.5-7B-Instruct.csv"), "#1f77b4", "o"),
    ("Llama-3.1-8B", "llama3_8b", "agents_Llama-3.1-8B-Instruct.csv",
     ("depth_carry_N1_Q256_Llama-3.1-8B-Instruct.csv",
      "depth_free_N1_Q256_Llama-3.1-8B-Instruct.csv"), "#ff7f0e", "s"),
)

REQUIRED_COLS = ("per_query_energy_j", "per_query_in_tok", "per_query_out_tok",
                 "team_size", "rounds", "decode_batch", "task")

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Data location and loading
# ---------------------------------------------------------------------------

def resolve_data_dir(explicit: str | None = None) -> str | None:
    """First readable candidate that actually holds the measured CSVs, else None."""
    root = os.path.dirname(HERE)
    candidates = [explicit, os.environ.get("AGENTIC_ECAL_DATA"),
                  os.path.join(root, "results"),
                  os.path.join(HERE, "results"),
                  os.path.join(os.path.dirname(root), "experimental_results", "data_energy")]
    probe = MEASURED[0][2]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, probe)):
            return c
    return None


def load(data_dir: str, *names: str):
    """Concatenate one or more measured CSVs, validating the columns the figures rely on."""
    import pandas as pd
    frames = []
    for n in names:
        path = os.path.join(data_dir, n)
        d = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in d.columns]
        if missing:
            raise SystemExit(f"{path}: missing expected column(s) {missing}")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _mean_curve(df, xcol, ycol):
    m = df.groupby(xcol)[ycol].mean()
    return m.index.values, m.values


def _loglog_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# The eCAL overlay: model prediction for a measured token workload
# ---------------------------------------------------------------------------

def two_rate_energy_per_query(key, p_in_total, p_out_total, n_calls, batch):
    """Paper Eq. (3) on a measured token workload, using the calibrated coefficients.

    The same ``TWO_RATE`` values fig_validation uses, applied the same way -- to the per-query
    totals, as they were fitted -- so the overlay and the validation figure rest on one
    calibration and no utilisation needs choosing.
    """
    return ae.TWO_RATE[key].energy(p_in_total, p_out_total, batch)


def two_rate_curve(df, xcol, key, batch=None):
    """(x, predicted energy) under Eq. (3); ``batch`` defaults to the frame's own serving batch."""
    g = df.groupby(xcol).agg(pin=("per_query_in_tok", "mean"),
                             pout=("per_query_out_tok", "mean"),
                             N=("team_size", "mean"), R=("rounds", "mean"),
                             b=("decode_batch", "mean"))
    y = np.array([two_rate_energy_per_query(key, r.pin, r.pout, r.N * r.R,
                                            batch if batch is not None else r.b)
                  for r in g.itertuples()])
    return g.index.values, y


def two_rate_accuracy(model, df):
    """(predicted, R2, MAPE) for Eq. (3) on a measured frame, with no fitted parameter."""
    pin = df.per_query_in_tok.values.astype(float)
    pout = df.per_query_out_tok.values.astype(float)
    ncalls = np.maximum(df.team_size.values.astype(float) * df.rounds.values.astype(float), 1.0)
    batch = df.decode_batch.values.astype(float)
    E = df.per_query_energy_j.values.astype(float)
    pred = np.array([two_rate_energy_per_query(model, a_, b_, c_, d_)
                     for a_, b_, c_, d_ in zip(pin, pout, ncalls, batch)])
    r2 = 1.0 - np.sum((E - pred) ** 2) / np.sum((E - E.mean()) ** 2)
    mape = float(np.mean(np.abs(pred - E) / E) * 100.0)
    return pred, float(r2), mape


def measured_rates(df):
    """Per serving batch, regress E = alpha*p_in + beta*p_out. Returns (batches, alpha, beta).

    alpha is the measured prefill rate, beta the measured decode rate, both in J/token. No model
    is fitted here -- these are a description of the data.
    """
    bs, alpha, beta = [], [], []
    for b, g in df.groupby("decode_batch"):
        A = np.column_stack([g.per_query_in_tok.values.astype(float),
                             g.per_query_out_tok.values.astype(float)])
        (al, be), *_ = np.linalg.lstsq(A, g.per_query_energy_j.values.astype(float), rcond=None)
        bs.append(float(b)); alpha.append(float(al)); beta.append(float(be))
    o = np.argsort(bs)
    return np.array(bs)[o], np.array(alpha)[o], np.array(beta)[o]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_scaling(data_dir, out_dir):
    """Measured scaling in three panels, with Eq. (3) overlaid on (a) and (b).

    Throughout: markers joined by a heavy line are MEASURED; thin unmarked lines are the MODEL,
    Eq. (3) evaluated on the same measured token workload at the same serving batch.
    """
    qwen = ae.LLMS["qwen2_5_7b"]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))

    # (a) reasoning depth: carry vs free, both models, at the high query budget where the gap shows
    for tag, fname, style, col in [
            ("Qwen carry (meas.)",  MEASURED[0][3][0], "o-",  "#1f77b4"),
            ("Qwen free (meas.)",   MEASURED[0][3][1], "o--", "#1f77b4"),
            ("Llama carry (meas.)", MEASURED[1][3][0], "s-",  "#ff7f0e"),
            ("Llama free (meas.)",  MEASURED[1][3][1], "s--", "#ff7f0e")]:
        x, y = _mean_curve(load(data_dir, fname), "rounds", "per_query_energy_j")
        ax[0].plot(x, y, style, color=col, label=tag, ms=3.5, lw=1.2, zorder=3)
    # Eq. (3) evaluated on each measured workload at its own serving batch, one curve per model
    # and condition, in the model's own colour. Drawn pale and unmarked so that the eye separates
    # model from measurement without needing four more legend entries.
    for label, key, _agents, depth, col, _mk in MEASURED:
        for fname, ls in ((depth[0], "-"), (depth[1], "--")):
            xa, ya = two_rate_curve(load(data_dir, fname), "rounds", key)
            ax[0].plot(xa, ya, ls, color=col, lw=1.0, alpha=0.55, zorder=1)
            meas = load(data_dir, fname).groupby("rounds").per_query_energy_j.mean().values
            err = 100 * (ya - meas) / meas
            print(f"  [depth {label:12s} {'carry' if ls == '-' else 'free ':5s}] "
                  f"Eq.(3) vs measured: {err.min():+.0f}%..{err.max():+.0f}%")
    ax[0].set_xlabel("reasoning depth $K$ (rounds)")
    ax[0].set_ylabel("energy / query (J)")
    ax[0].set_title("(a) depth: carry vs free")
    h0, l0 = ax[0].get_legend_handles_labels()
    h0.append(Line2D([0], [0], color="0.45", lw=1.0, alpha=0.6))
    l0.append("Eq. (3), model")
    ax[0].legend(h0, l0, frameon=False, fontsize=5)

    # (b) agent count at a small and a large serving batch (Qwen)
    d = load(data_dir, MEASURED[0][2])
    for b, style in [(16, "o-"), (256, "^-")]:
        g = d[d.decode_batch == b].groupby("team_size").per_query_energy_j.mean()
        ax[1].plot(g.index.values, g.values, style, label=f"$b$={b} (meas.)", ms=4, zorder=3)
    # Eq. (3) carries a batch term, so it yields one curve per batch rather than one for both.
    for bb, col in ((16, "C0"), (256, "C1")):
        xb, yb = two_rate_curve(d[d.decode_batch == bb], "team_size",
                                MEASURED[0][1], batch=bb)
        ax[1].plot(xb, yb, ":", color=col, lw=1.4, zorder=1,
                   label=f"Eq. (3), $b$={bb}")
        meas = d[d.decode_batch == bb].groupby("team_size").per_query_energy_j.mean().values
        err = 100 * (yb - meas) / meas
        print(f"  [scaling b={bb:>3}] Eq.(3) vs measured: {err.min():+.0f}%..{err.max():+.0f}%")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    # A log axis spanning only 2..30 makes matplotlib label the MINOR ticks in scientific notation
    # (2x10^0, 3x10^0, ...), which collides on a panel this narrow, and puts a major tick at 10^2
    # outside the data. Tick the actual agent counts instead.
    _decade_free_ticks(ax[1], sorted(d.team_size.unique()))
    ax[1].set_xlabel("agent count $N$"); ax[1].set_ylabel("energy / query (J)")
    ax[1].set_title("(b) agent count")
    h1, l1 = ax[1].get_legend_handles_labels()

    ax[1].legend(h1, l1, frameon=False, fontsize=5)

    # (c) exponent a vs batch. The single-rate model cannot appear here at all: eta is a constant
    # scale factor and cancels exactly from a log-log slope, so every eta gives the same flat
    # line. Eq. 5 has a batch term, so it predicts the rise the measurements show.
    # No model line: eta is a constant scale factor and cancels exactly from a log-log slope,
    # so every utilisation predicts the same, batch-independent exponent.
    batches = []
    for label, key, fname, _depth, col, _mk in MEASURED:
        d = load(data_dir, fname)
        bb, aa = [], []
        batches = sorted(float(x) for x in d.decode_batch.unique())
        for b, gb in d.groupby("decode_batch"):
            slopes = [_loglog_slope(*_mean_curve(gt, "team_size", "per_query_energy_j"))
                      for _, gt in gb.groupby("task")]
            bb.append(float(b)); aa.append(float(np.mean(slopes)))
        ax[2].plot(bb, aa, "o-" if label.startswith("Qwen") else "s--",
                   label=f"{label} (meas.)", ms=4, color=col, zorder=3)
        print(f"  [exponent {label:12s}] measured {aa[0]:.2f}->{aa[-1]:.2f}")
    ax[2].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[2].set_xscale("log")
    _decade_free_ticks(ax[2], sorted(batches))
    ax[2].set_xlabel("serving batch $b$")
    ax[2].set_ylabel("exponent $a$  ($E\\sim N^a$)")
    ax[2].set_title("(c) crossover")
    h2, l2 = ax[2].get_legend_handles_labels()
    ax[2].legend(h2, l2, frameon=False, fontsize=5.5, loc="lower right",
                 labelspacing=0.25, handlelength=1.4)

    _save(fig, out_dir, "fig_scaling.pdf")


def _decade_free_ticks(ax, ticks):
    """Tick a log x-axis at the measured values rather than at decades."""
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_validation(data_dir, out_dir):
    """The two-rate form of Eq. (3) against measurement, with measured coefficients.

    Coefficients are the calibrated ``TWO_RATE`` values, so the figure tests the *form* -- that a
    compute-bound prefill rate plus a batch-amortising decode rate describes the data -- separately
    from the hardware derivation of Eqs. (4)-(5), which ``validate_two_rate()`` reports.
    Eq. (3) is applied to the per-query totals, as the coefficients were fitted.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    span = []

    # (a) predicted vs measured energy, every configuration, both models
    for label, key, agents, depth, col, mk in MEASURED:
        d = load(data_dir, agents, *depth)
        cal = ae.TWO_RATE[key]
        pin = d.per_query_in_tok.values.astype(float)
        pout = d.per_query_out_tok.values.astype(float)
        bat = d.decode_batch.values.astype(float)
        E = d.per_query_energy_j.values.astype(float)
        Ep = np.array([cal.energy(i, o, bb) for i, o, bb in zip(pin, pout, bat)])
        r2 = 1.0 - np.sum((E - Ep) ** 2) / np.sum((E - E.mean()) ** 2)
        mape = float(np.mean(np.abs(Ep - E) / E) * 100.0)
        ax[0].scatter(E, Ep, s=9, c=col, marker=mk, alpha=0.45, edgecolors="none",
                      label=f"{label}  ($R^2$={r2:.3f}, {mape:.0f}%)")
        span += [E.min(), E.max(), Ep.min(), Ep.max()]
        print(f"  [{label}] n={len(d)}  R2={r2:.4f}  MAPE={mape:.1f}%")
    lo, hi = min(span), max(span)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ideal")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("measured energy / query (J)")
    ax[0].set_ylabel("predicted energy (J)")
    ax[0].set_title("(a) two-rate model vs measurement")
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")

    # (b) the two rates, extracted per batch by regressing E = alpha p_in + beta p_out. alpha
    #     (prefill) is batch-independent; beta (decode) collapses as a_dec/b + c_0, the
    #     memory-bound law Eq. (5) asserts.
    for label, key, agents, _depth, col, mk in MEASURED:
        d = load(data_dir, agents)
        cal = ae.TWO_RATE[key]
        bs, alpha, beta = measured_rates(d)
        # How much of the measured energy the decode term actually accounts for. Where that
        # share is small, beta is a small residual extracted from regressors correlated at
        # ~0.95, so the marker is weakly determined; those are drawn hollow.
        share = np.array([float(cal.c_dec(b_) * g.per_query_out_tok.sum()
                                / (cal.c_pre * g.per_query_in_tok.sum()
                                   + cal.c_dec(b_) * g.per_query_out_tok.sum()))
                          for b_, g in d.groupby("decode_batch")])
        firm, weak = share >= 0.15, share < 0.15
        bb = np.logspace(np.log10(bs.min()), np.log10(bs.max()), 60)
        ax[1].plot(bs[firm], beta[firm], mk, color=col, ms=5, ls="none",
                   label=f"{label} decode (meas.)")
        if weak.any():
            ax[1].plot(bs[weak], beta[weak], mk, mfc="none", mec=col, mew=1.0, ms=5, ls="none")
        ax[1].plot(bb, [cal.c_dec(x) for x in bb], "-", color=col, lw=1.3,
                   label=f"{label} $a_{{\\rm dec}}/b+c_0$")
        ax[1].plot(bs, alpha, ":", color=col, lw=1.1, marker=mk, ms=3, alpha=0.8)
        print(f"  [{label}] c_pre={cal.c_pre} vs measured prefill {alpha.mean():.4f}; "
              f"a_dec={cal.a_dec}, c_0={cal.kv_floor}; decode share of E "
              f"{100*share[0]:.0f}%->{100*share[-1]:.0f}%; "
              f"{int(weak.sum())} point(s) weakly determined")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("serving batch $b$")
    ax[1].set_ylabel("energy rate (J/token)")
    ax[1].set_title("(b) decode $\\sim\\!1/b$, prefill flat")
    h, lab = ax[1].get_legend_handles_labels()
    h.append(Line2D([0], [0], color="0.45", ls=":", lw=1.2, marker="."))
    lab.append("prefill $c_{\\rm pre}$ (meas.)")
    h.append(Line2D([0], [0], color="0.45", ls="none", marker="o", mfc="none", ms=5))
    lab.append("decode share of $E<15\\%$")
    ax[1].legend(h, lab, frameon=False, fontsize=5.5, loc="upper right")

    _save(fig, out_dir, "fig_validation.pdf")


def _scaffold_tokens(data_dir, key_files):
    """Fixed prompt scaffold [tokens]: the round-1 prompt of the depth sweep, measured."""
    d = load(data_dir, key_files[0])
    return float(d[d.rounds == d.rounds.min()].per_query_in_tok.mean())


def fig_tokens(data_dir, out_dir):
    """Validate the workflow's TOKEN accounting -- the structural claim -- against measurement.

    Prefill and decode token counts contain no energy model and no utilisation, so this isolates
    the claim that history accumulates super-linearly from the eta knob that would otherwise
    absorb any error in it.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # (a) reasoning depth, N = 1: retain every round (carry) vs the previous round only (free).
    for label, key, _agents, depth, col, mk in MEASURED:
        base = _scaffold_tokens(data_dir, depth)
        for fname, tag, hr, ls in [(depth[0], "carry", None, "-"), (depth[1], "free", 1, "--")]:
            d = load(data_dir, fname)
            g = d.groupby("rounds")[["per_query_in_tok", "per_query_out_tok"]].mean()
            per_round_out = np.diff(g.per_query_out_tok.values, prepend=0.0)
            pred = []
            for i, K in enumerate(g.index.values):
                wf = ae.Workflow(model=ae.LLMS[key], hw=HW_OVERLAY, sys_tokens=int(round(base)),
                                 steps=[ae.Step(agent="solo", p_out=int(round(o)))
                                        for o in per_round_out[:i + 1]],
                                 carry_history=True, history_rounds=hr, round_size=1,
                                 exclude_self=False)
                pred.append(wf.run()["prefill_tokens"])
            meas = g.per_query_in_tok.values
            err = 100 * (np.array(pred) - meas) / meas
            ax[0].plot(g.index.values, meas, mk, color=col, ms=4, ls="none", zorder=3)
            ax[0].plot(g.index.values, pred, ls, color=col, lw=1.2,
                       label=f"{label} {tag} ({err[-1]:+.0f}%)")
            print(f"  [depth {tag:5s} {label:12s}] prefill error at K={g.index.max()}: "
                  f"{err[-1]:+.1f}%  (worst {err[np.argmax(np.abs(err))]:+.1f}%)")
    ax[0].set_xlabel("reasoning depth $K$ (rounds)")
    ax[0].set_ylabel("prefill tokens / query")
    ax[0].set_title("(a) depth: markers measured, lines model")
    ax[0].legend(frameon=False, fontsize=5.5, loc="upper left")

    # (b) agent count: the three retention rules against the measured debate.
    RULES = [("peers of prev. round", dict(history_rounds=1, exclude_self=True), "-"),
             ("whole transcript", dict(history_rounds=None, exclude_self=False), "--"),
             ("no history", dict(history_rounds=0, exclude_self=False), ":")]
    for label, key, agents, depth, col, mk in MEASURED:
        base = _scaffold_tokens(data_dir, depth)
        d = load(data_dir, agents)
        g = d.groupby("team_size").agg(pin=("per_query_in_tok", "mean"),
                                       pout=("per_query_out_tok", "mean"),
                                       R=("rounds", "mean"))
        ax[1].plot(g.index.values, g.pin.values, mk, color=col, ms=4, ls="none", zorder=3)
        for rule, kw, ls in RULES:
            pred = np.array([ae.debate_workflow(ae.LLMS[key], HW_OVERLAY, int(N), int(r.R),
                                                r.pout / (N * r.R), int(round(base)),
                                                **kw).run()["prefill_tokens"]
                             for N, r in zip(g.index.values, g.itertuples())])
            err = 100 * (pred - g.pin.values) / g.pin.values
            ax[1].plot(g.index.values, pred, ls, color=col, lw=1.2,
                       label=f"{label}: {rule}" if True else None)
            print(f"  [agents {label:12s}] {rule:22s}: error {err.min():+.0f}%..{err.max():+.0f}%")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    _decade_free_ticks(ax[1], sorted(load(data_dir, MEASURED[0][2]).team_size.unique()))
    ax[1].set_xlabel("agent count $N$")
    ax[1].set_ylabel("prefill tokens / query")
    ax[1].set_title("(b) which transcript an agent re-reads")
    ax[1].legend(frameon=False, fontsize=5, loc="upper left")

    _save(fig, out_dir, "fig_tokens.pdf")


COMPOSITION = (
    ("prefill", "e_prefill", "#4477aa"),
    ("decode", "e_decode", "#88ccee"),
    ("retrieval", "e_retrieval", "#999933"),
    ("tools", "e_tool", "#ddcc77"),
    ("inter-agent", "e_transmission", "#117733"),
    ("orchestration", "e_orchestration", "#bbbbbb"),
)

COMP_BATCHES = (1, 64, 256)     # single-stream, production, and the largest measured batch
COMP_ROUNDS = [1, 2, 3, 4, 5, 6]


def fig_components(out_dir):
    """How the composition of workflow energy moves with loop depth, at three serving batches.

    The four-agent RAG team is re-run over its own transcript for K = 1..6 rounds, the depth range
    the measured sweeps cover. Calls are priced with Eq. (3) using the measured coefficients of
    fig_validation, so the prefill and decode bands are its two terms. The black curve is total
    E_W on the right-hand axis, fixed to one range across panels so the regimes are comparable.
    """
    hw, calib = CASE_HW, CASE_CALIB
    fig, axes = plt.subplots(1, len(COMP_BATCHES), figsize=(7.1, 2.6), sharey=True)

    for col, (ax, batch) in enumerate(zip(axes, COMP_BATCHES)):
        shares = {k: [] for _, k, _ in COMPOSITION}
        totals = []
        for depth in COMP_ROUNDS:
            wf = dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=depth),
                                     serving_batch=batch, calib=calib)
            r = wf.run()
            totals.append(r["e_total"])
            for _, key, _c in COMPOSITION:
                shares[key].append(100.0 * r[key] / r["e_total"])

        ax.stackplot(COMP_ROUNDS, [shares[k] for _, k, _ in COMPOSITION],
                     labels=[lab for lab, _, _ in COMPOSITION],
                     colors=[c for _, _, c in COMPOSITION], lw=0.3, edgecolor="white")
        ax.set_xlim(min(COMP_ROUNDS), max(COMP_ROUNDS)); ax.set_ylim(0, 100)
        ax.set_xticks(COMP_ROUNDS)
        ax.set_title("single-stream" if batch == 1 else f"batch {batch}", fontsize=7.5)
        ax.set_xlabel("rounds $K$ of the four-agent team", fontsize=7)
        ax.grid(False); ax.tick_params(labelsize=6)
        if col:
            ax.tick_params(labelleft=False)

        rhs = ax.twinx()
        rhs.plot(COMP_ROUNDS, totals, "k-", lw=1.4, marker="o", ms=3, zorder=5,
                 path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        rhs.set_yscale("log"); rhs.set_ylim(150, 6e4); rhs.grid(False)
        rhs.tick_params(labelsize=6)
        if col == len(COMP_BATCHES) - 1:
            rhs.set_ylabel("$E_W$ (J)", fontsize=7, labelpad=1)
        else:
            rhs.set_yticklabels([])

        print(f"  batch {batch:>3}: prefill {shares['e_prefill'][0]:5.1f}%->{shares['e_prefill'][-1]:5.1f}%"
              f"  decode {shares['e_decode'][0]:5.1f}%->{shares['e_decode'][-1]:5.1f}%"
              f"  E_W {totals[0]:6.0f}->{totals[-1]:7.0f} J")

    axes[0].set_ylabel("% of workflow energy", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.07), columnspacing=1.2, handlelength=1.1)
    _save(fig, out_dir, "fig_components.pdf")


def fig_amortization(out_dir):
    """Embodied cost amortises over served invocations G; crossover G* = E_emb / E_W.

    Single-rate at one eta for all three curves: the 65B and 70B have no measured calibration of
    their own, so a uniform method is the only comparable one. The crossover RATIOS between models
    are invariant to hardware and eta alike, which is why the text quotes ratios, not absolutes.
    """
    hw = CASE_HW
    G = np.logspace(3, 12, 120)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for label, key, col, ls in AMORT_MODELS:
        model = ae.LLMS[key]
        r = dataclasses.replace(ae.four_agent_rag_workflow(model=model, hw=hw),
                                serving_batch=CASE_BATCH).run()
        e_w, e_emb = r["e_total"], model.e_pretrain
        bits = ae.BITS_PER_TOKEN * r["useful_output_tokens"]
        floor = e_w / bits
        ax.loglog(G, (e_w + e_emb / G) / bits, ls, color=col, lw=1.3, label=label)
        ax.axhline(floor, color=col, ls=":", lw=0.6, alpha=0.6)
        if e_emb > 0:
            gstar = e_emb / e_w
            ax.plot([gstar], [2 * floor], "v", color=col, ms=5, zorder=4)
            print(f"  {label:12s} E_emb={e_emb / ae.GWH_TO_J:5.2f} GWh  E_W={e_w:7.0f} J  "
                  f"G*={gstar:.2e}  floor={floor:.2e} J/bit")
        else:
            print(f"  {label:12s} E_emb unknown (0)      E_W={e_w:7.0f} J  "
                  f"G*=n/a       floor={floor:.2e} J/bit")
    ax.set_xlabel("served invocations $G$")
    ax.set_ylabel("agentic-eCAL (J/bit)")
    ax.set_title(f"Amortization ({hw.name}, $b$={CASE_BATCH})", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    _save(fig, out_dir, "fig_amortization.pdf")



# ---------------------------------------------------------------------------
# Placement: what distribution actually costs, and where the cost lives
# ---------------------------------------------------------------------------


# Every bearer of the placement table except the NB-IoT uplink, which no operator would use to
# carry agent state and is the only one on which transmission is not a small term.
COMM_BEARERS = ("metro", "ftth", "5g", "edgecell")

# How a hand-off can carry the state its successor needs. These are the protocol choices open to
# a distributed agentic deployment, not properties of the network.
PROTOCOLS = (("full context", "#cc6677"),      # re-ship the accumulated transcript
             ("incremental", "#4477aa"),       # only what the sender just produced
             ("summary", "#ddaa33"),           # a bounded running summary
             ("KV migration", "#882255"))      # ship the cache instead of rebuilding it
COMM_N, COMM_K, COMM_POUT, SUMMARY_TOKENS = 4, 3, 250, 400


def _protocol_messages(kind, n_agents=COMM_N, rounds=COMM_K, p_out=COMM_POUT):
    """Tokens carried by each hand-off under one protocol."""
    out, produced = [], 0.0
    for _ in range(rounds):
        for _i in range(n_agents):
            produced += p_out
            if kind == "full context":
                out.append(produced)
            elif kind == "incremental":
                out.append(p_out)
            elif kind == "summary":
                out.append(min(produced, SUMMARY_TOKENS))
    return out


def _protocol_bits(kind, model):
    """Bits a hand-off puts on the wire; the cache is bytes, not tokens."""
    if kind == "KV migration":
        ctx = COMM_N * COMM_K * COMM_POUT / 2.0
        return [model.kv_bytes_per_token * ctx * 8.0] * (COMM_N * COMM_K)
    return [ae.BITS_PER_TOKEN * t for t in _protocol_messages(kind)]


def fig_comm(out_dir):
    """Communication-efficient distributed reasoning: what the hand-off protocol costs.

    The four protocols carry the same conversation between the same agents over the same links.
    What differs is what each hand-off must contain, and that choice spans five orders of
    magnitude in traffic, energy and added latency.
    """
    hw, calib, batch = CASE_HW, CASE_CALIB, CASE_BATCH
    model = ae.LLMS["llama3_8b"]
    e_w = dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=COMM_K),
                              serving_batch=batch, calib=calib).run()["e_total"]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))
    xs = np.arange(len(PROTOCOLS))

    # (a) traffic the protocol puts on the wire
    vols = [sum(_protocol_bits(k, model)) / 1e6 for k, _ in PROTOCOLS]
    ax[0].bar(xs, vols, 0.6, color=[c for _, c in PROTOCOLS])
    for x, v in zip(xs, vols):
        ax[0].annotate(f"{v:,.3g}", xy=(x, v * 1.6), ha="center", fontsize=5)
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-2, 1e6)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([k for k, _ in PROTOCOLS], fontsize=5.5,
                                                rotation=20, ha="right")
    ax[0].set_ylabel("traffic per session (Mbit)")
    ax[0].set_title("(a) what a hand-off carries")
    ax[0].tick_params(labelsize=6)

    # (b) the energy that traffic costs, as a share of the compute it accompanies
    width = 0.2
    for j, (kind, col) in enumerate(PROTOCOLS):
        bits = _protocol_bits(kind, model)
        vals = [100 * sum(osi.segment_energy(x, pl.BEARERS[b].eps)
                          + osi.endpoint_stack_energy(x) for x in bits) / e_w
                for b in COMM_BEARERS]
        ax[1].bar(np.arange(len(COMM_BEARERS)) + (j - 1.5) * width, vals, width,
                  color=col, label=kind)
        print(f"  [{kind:<13}] traffic {sum(bits)/1e6:9.3f} Mb | "
              + ", ".join(f"{pl.BEARERS[b].name.split()[0]} {v:.3g}%"
                          for b, v in zip(COMM_BEARERS, vals)))
    ax[1].axhline(100, color="k", lw=0.7, ls=":")
    ax[1].set_yscale("log"); ax[1].set_xticks(np.arange(len(COMM_BEARERS)))
    ax[1].set_xticklabels([pl.BEARERS[b].name.replace(" / copper access", "")
                           for b in COMM_BEARERS], fontsize=5, rotation=20, ha="right")
    ax[1].set_ylabel("$E_{\\rm tx}$ (% of $E_W$)")
    ax[1].set_title("(b) cost against the compute")
    ax[1].legend(frameon=False, fontsize=4.5, ncol=2); ax[1].tick_params(labelsize=6)

    # (c) added latency. The three text protocols coincide: latency is set by the propagation
    # floor of len(bits) messages, which every protocol shares, not by payload size.
    for (kind, col), mk in zip(PROTOCOLS, ("o", "s", "^", "D")):
        bits = _protocol_bits(kind, model)
        lat = [1e3 * (osi.cascade_bits(sum(bits)) / pl.BEARERS[b].rate
                      + len(bits) * pl.BEARERS[b].latency) for b in COMM_BEARERS]
        ax[2].semilogy(np.arange(len(COMM_BEARERS)), lat, marker=mk, ls="-", color=col,
                       ms=3.2, lw=1.2, mfc="none", mew=0.9, label=kind)
        print(f"  [{kind:<13}] added latency {lat[0]:,.0f}-{lat[-1]:,.0f} ms across the links")
    ax[2].set_xticks(np.arange(len(COMM_BEARERS)))
    ax[2].set_xticklabels([pl.BEARERS[b].name.replace(" / copper access", "")
                           for b in COMM_BEARERS], fontsize=5, rotation=20, ha="right")
    ax[2].text(0.5, 0.40, "text protocols coincide within $11\\%$:\n"
               "latency is set by the $11$ propagation delays, not payload",
               transform=ax[2].transAxes, fontsize=4.2, color="0.35", ha="center")
    ax[2].set_ylabel("added latency (ms)")
    ax[2].set_title("(c) added latency per session")
    ax[2].legend(frameon=False, fontsize=4.5); ax[2].tick_params(labelsize=6)

    _save(fig, out_dir, "fig_comm.pdf")



LAT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def fig_latency(out_dir):
    """The latency budget of an agentic loop, and what actually trades against energy.

    Distributing the agents adds network latency; batching them adds queueing latency. The two
    differ by three orders of magnitude, which decides where a latency-constrained deployment has
    any freedom.
    """
    hw, calib = CASE_HW, CASE_CALIB
    rounds, hops = COMM_K, 3
    n_handoffs = COMM_N * rounds - 1
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))

    # (a) end-to-end budget: generation against the network that carries the hand-offs
    base = dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=rounds),
                               serving_batch=CASE_BATCH, calib=calib).run()
    compute = base["t_wall"]
    nets = [hops * pl.BEARERS[b].latency * n_handoffs for b in COMM_BEARERS]
    xs = np.arange(len(COMM_BEARERS))
    ax[0].bar(xs, [compute] * len(xs), 0.6, color="#4477aa", label="generation")
    ax[0].bar(xs, nets, 0.6, bottom=[compute] * len(xs), color="#cc6677", label="network, 3 hops")
    for x, n in zip(xs, nets):
        ax[0].annotate(f"{100*n/compute:.2f}%", xy=(x, compute + n), ha="center",
                       va="bottom", fontsize=5)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([pl.BEARERS[b].name.replace(" / copper access", "")
                           for b in COMM_BEARERS], fontsize=5, rotation=20, ha="right")
    ax[0].set_ylabel("end-to-end latency (s)")
    ax[0].set_title("(a) the latency budget")
    ax[0].legend(frameon=False, fontsize=5.5); ax[0].tick_params(labelsize=6)
    print(f"  [latency] generation {compute:.0f} s; network 3 hops "
          + ", ".join(f"{pl.BEARERS[b].name.split()[0]} {1e3*n:.0f} ms ({100*n/compute:.2f}%)"
                      for b, n in zip(COMM_BEARERS, nets)))

    # (b) the trade that is actually available: serving batch
    es, ts = [], []
    for b in LAT_BATCHES:
        r = dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=rounds),
                                serving_batch=b, calib=calib).run()
        es.append(r["e_total"]); ts.append(r["t_wall"])
    ax[1].plot(ts, es, "o-", color="#117733", ms=3.5, lw=1.2)
    for b, t, e in zip(LAT_BATCHES, ts, es):
        if b in (1, 16, 64, 512):
            ax[1].annotate(f"$b$={b}", xy=(t, e), xytext=(4, 4), textcoords="offset points",
                           fontsize=5)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("end-to-end latency (s)"); ax[1].set_ylabel("$E_W$ (J)")
    ax[1].set_title("(b) the batch Pareto")
    ax[1].tick_params(labelsize=6)
    print(f"  [pareto] b=1: {es[0]:.0f} J in {ts[0]:.0f} s;  b=512: {es[-1]:.0f} J in {ts[-1]:.0f} s "
          f"({es[0]/es[-1]:.1f}x energy for {ts[-1]/ts[0]:.1f}x latency)")

    # (c) what a latency budget permits, against the workflow that must fit in it
    depths = [1, 2, 3, 4, 5, 6]
    for b, col in ((1, "#cc6677"), (16, "#ddaa33"), (64, "#4477aa"), (256, "#117733")):
        lat = [dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=k),
                                   serving_batch=b, calib=calib).run()["t_wall"] for k in depths]
        ax[2].semilogy(depths, lat, "o-", color=col, ms=3, lw=1.2, label=f"$b$={b}")
    for sla, lab in ((1.0, "1 s"), (10.0, "10 s"), (60.0, "1 min")):
        ax[2].axhline(sla, color="0.75", lw=0.6, ls=":")
        ax[2].annotate(lab, xy=(6.0, sla * 1.15), ha="right", fontsize=4.5, color="0.45")
    ax[2].set_xticks(depths)
    ax[2].set_xlabel("rounds $K$"); ax[2].set_ylabel("end-to-end latency (s)")
    ax[2].set_title("(c) against a latency budget")
    ax[2].legend(frameon=False, fontsize=5); ax[2].tick_params(labelsize=6)

    _save(fig, out_dir, "fig_latency.pdf")



LINK_RATES = ((1e9, "1 GbE"), (1e10, "10 GbE"), (1e11, "100 GbE"))


def write_dimensioning_table(out_dir):
    """Site dimensioning: how much compute fits behind a link before hand-offs saturate it.

    The generated table carries its own \\ROOF{} marker inside the float, because a colour set
    around the \\input never reaches a float: floats are typeset where they land, not where they
    are declared.

    An operator sizes a site in accelerators and its uplink in bit/s. One accelerator serving at
    batch b completes b/t_wall sessions per second, and each session emits a fixed hand-off
    volume, so the offered load per accelerator follows directly and can be compared against a
    link rate. Writes tables/tab_dimensioning.tex.
    """
    hw, calib, batch = CASE_HW, CASE_CALIB, CASE_BATCH
    model = ae.LLMS["llama3_8b"]
    r = dataclasses.replace(ae.four_agent_rag_workflow(hw=hw, rounds=COMM_K),
                            serving_batch=batch, calib=calib).run()
    sessions = batch / r["t_wall"]

    rows = []
    for kind, _col in PROTOCOLS:
        bits = sum(_protocol_bits(kind, model))
        load = sessions * bits
        per_link = [rate / load for rate, _ in LINK_RATES]
        binding = "compute" if per_link[1] > 1 else "network"
        rows.append(f"{kind} & {bits/1e6:,.3g} & {load/1e6:,.3g} & "
                    + " & ".join(f"{v:,.0f}" if v >= 1 else f"{v:.2f}" for v in per_link)
                    + f" & {binding} \\\\")
        print(f"  [dimension] {kind:<13} {bits/1e6:9.3g} Mb/session, {load/1e6:9.3g} Mb/s per "
              f"accelerator, {per_link[1]:,.1f} accelerators per 10 GbE ({binding}-bound)")

    tex = f"""% Generated by scripts/clean/make_figures.py -- do not edit by hand.
\\begin{{table*}}[t]
\\caption{{Site dimensioning for an agentic service. One accelerator serving at batch
${batch}$ completes ${sessions:.2f}$ sessions per second.}}
\\label{{tab:dimensioning}}
\\centering
\\ROOF{{}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrrrrrl@{{}}}}
\\toprule
 & traffic & load & \\multicolumn{{3}}{{c}}{{accelerators per link}} & binding \\\\
\\cmidrule(lr){{4-6}}
Hand-off protocol & [Mbit] & [Mb/s] & 1\\,GbE & 10\\,GbE & 100\\,GbE & resource \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\OLD{{}}
\\end{{table*}}
"""
    path = os.path.join(out_dir, "tab_dimensioning.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.basename(path)}")


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_FIGS = ("scaling", "validation", "tokens", "components", "amortization", "comm",
            "latency")
NEEDS_DATA = {"scaling", "validation", "tokens"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="directory holding the measured CSVs")
    p.add_argument("--out", default=OUT_DEFAULT, help="output directory")
    p.add_argument("--figs", nargs="+", choices=ALL_FIGS, default=list(ALL_FIGS))
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    data_dir = resolve_data_dir(a.data)
    wanted = list(dict.fromkeys(a.figs))

    if data_dir:
        print(f"data: {data_dir}")
    else:
        skipped = [f for f in wanted if f in NEEDS_DATA]
        if skipped:
            print(f"data: not found -- skipping {', '.join(skipped)}. "
                  f"Pass --data <dir> or set AGENTIC_ECAL_DATA; expected to contain "
                  f"{MEASURED[0][2]}.")
        wanted = [f for f in wanted if f not in NEEDS_DATA]
    print(f"out : {a.out}\n")

    for name in wanted:
        print(f"[{name}]")
        if name == "scaling":
            fig_scaling(data_dir, a.out)
        elif name == "validation":
            fig_validation(data_dir, a.out)
        elif name == "tokens":
            fig_tokens(data_dir, a.out)
        elif name == "components":
            fig_components(a.out)
        elif name == "amortization":
            fig_amortization(a.out)
        elif name == "comm":
            fig_comm(a.out)
        elif name == "latency":
            fig_latency(a.out)
        print()


if __name__ == "__main__":
    main()
