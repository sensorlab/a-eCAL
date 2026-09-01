#!/usr/bin/env python3
"""Fit the per-agent communication law to the measured multi-agent debate runs.

Source: ~/DATA/data_steiner_Aug26 -- 16 open-weight models, 8 reasoning tasks, team sizes
1..30 at full mesh (comm_k = N-1), 3 debate rounds, peer_token_budget = 120, vLLM backend.

Only token counts are used; task accuracy is deliberately not analysed here.

The teams_*.csv files carry per-team-per-item token totals and are ~20x smaller than the
agents_*.csv files, so the whole 132 GB tree is summarised without loading any of it into
memory. Round 1 is the pre-communication round -- no peer has spoken -- and its prompt equals
the single-agent ``ensemble`` prompt, which is what makes it the natural no-communication
baseline. Rounds 2 and 3 add peer context, giving

    p_in(k) = a + b*k

with b the tokens imported per peer. Writes the fitted constants for PEER_LAW in
model_placement.py.

Usage:  python3 fit_peer_law.py [outdir]
"""
from __future__ import annotations

import glob, os, sys
import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/DATA/data_steiner_Aug26")
COLS = ["model_short", "task", "condition", "team_size", "comm_k", "comm_round",
        "total_prompt_toks", "total_output_toks"]
KEYS = ["model_short", "condition", "team_size", "comm_k", "comm_round"]


def summarise(data_dir: str = DATA) -> pd.DataFrame:
    """Chunked pass over every teams_*.csv -> per-cell token means."""
    out = []
    for f in sorted(glob.glob(f"{data_dir}/*/teams_*.csv")):
        parts = []
        for ch in pd.read_csv(f, usecols=COLS, chunksize=500_000, low_memory=False):
            ch = ch.dropna(subset=["total_prompt_toks", "total_output_toks"])
            parts.append(ch.groupby(KEYS, dropna=False).agg(
                n=("total_prompt_toks", "size"),
                p_sum=("total_prompt_toks", "sum"),
                o_sum=("total_output_toks", "sum")))
        if parts:
            out.append(pd.concat(parts).groupby(level=list(range(len(KEYS)))).sum())
            print(f"  {os.path.basename(os.path.dirname(f))}: "
                  f"{int(out[-1].n.sum()):,} team-rows", flush=True)
    d = pd.concat(out).reset_index()
    d["p"] = d.p_sum / d.n
    d["o"] = d.o_sum / d.n
    return d


def fit(d: pd.DataFrame) -> pd.DataFrame:
    """Per model: solo prompt, round-2 intercept/slope in k, and mean output tokens."""
    c = d[d.condition.str.startswith("comm_true")].copy()
    c["p_ag"] = c.p / c.team_size          # teams rows are team totals
    c["o_ag"] = c.o / c.team_size
    rows = []
    for m, g in c.groupby("model_short"):
        r1, r2 = g[g.comm_round == 1], g[g.comm_round == 2]
        if len(r2) < 3:
            continue
        k, y = r2.comm_k.values.astype(float), r2.p_ag.values
        b, a = np.polyfit(k, y, 1)
        ss = 1 - ((y - (a + b * k)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(dict(model=m, solo=r1.p_ag.mean(), a=a, b=b,
                         p_out=r1.o_ag.mean(), r2=ss))
    return pd.DataFrame(rows).sort_values("b")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    d = summarise()
    d.to_csv(os.path.join(outdir, "tokens_by_cell.csv"), index=False)
    t = fit(d)
    print(f"\n{'model':<24}{'solo':>7}{'a':>8}{'b tok/peer':>12}{'p_out':>8}{'R2':>9}")
    for _, x in t.iterrows():
        print(f"{x.model:<24}{x.solo:>7.0f}{x.a:>8.0f}{x.b:>12.1f}{x.p_out:>8.0f}{x.r2:>9.4f}")
    print(f"\nmedian b = {t.b.median():.1f} tokens/peer (peer_token_budget = 120)")
    print(f"min R2   = {t.r2.min():.4f}")
