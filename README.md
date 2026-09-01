# a-eCAL — Agentic-eCAL

Analytical model and paper source for **agentic-eCAL**, extending the
[eCAL](https://github.com/sensorlab/eCAL) methodology (Chou et al., IEEE JSAC 2026) from a single
AI-model lifecycle to **agentic workflows on open-weight LLMs**: a graph of LLM calls, tool
invocations, retrievals and inter-agent messages, measured in joules per useful bit.

## The model

A workflow's operational energy is the sum over its steps,

```
E_W = (1 + gamma_v) [ sum_k E_call^(k) + sum_k E_tool^(k) + sum_k E_ret^(k) + E_tx ]
```

and each LLM call is priced by a **two-rate** model, because a call's two phases have different
energy characteristics: prefill is compute-bound and batch-independent, decode is
memory-bandwidth-bound and amortises the weight read over the serving batch.

```
E_call(p_in, p_out; b) = c_pre * p_in + c_dec(b) * p_out
c_pre    = 2 N P / (eta_pre * Pi)
c_dec(b) = P / (eps * B_HBM) * ( N beta / b  +  gamma * cbar )
```

Against 756 measured configurations on an A100 under vLLM, the two-rate form reaches
**R² = 0.997 (Qwen2.5-7B) and 0.996 (Llama-3.1-8B)** with two measured coefficients per model. The
coefficients are also *derivable* from datasheet quantities alone: `c_pre` comes out at 0.0230
against a measured 0.023, and the decode weight term at 4.27 J against a measured 3.2 J.

Inter-agent transmission reuses eCAL's own per-layer OSI model (Eq. 3 of the eCAL paper): the
payload cascades through seven layers, each adding data- and control-plane overhead, and both
sender and receiver are charged.

## Layout

```
scripts/clean/
  agentic_ecal.py             the model: Hardware/LLM descriptors, two-rate call model, Workflow, the metric
  osi.py                      eCAL Eq. (3), the OSI-layer transmission model, for E_tx
  placement.py                bearers, deployment archetypes, and the generated placement table
  make_figures.py             the shared figures + tab_dimensioning
  model_placement.py          device tiers with MEMORY, 16-model fleet geometry, the measured peer law
  fit_peer_law.py             fits the peer law from ~/DATA/data_steiner_Aug26 (see below)
  make_figures_placement.py   fig_model_placement + tab_model_placement          (v2 only)
  make_figure_workflow.py     fig_workflow, the case-study schematic              (v2 only)
  make_figure_infra.py        fig_infra + tab_infra + infra_macros                (v2 only, needs runs)
  results/                    the measured sweeps (A100, vLLM), plus the cached Steiner aggregate
main.tex            the frozen manuscript -- do not edit
v1/, v2/            successive revisions; v2 is the working copy
figures/, tables/   generated artifacts, committed so the paper builds without running the model
```

`main.tex` and `v1/` are frozen. All current work happens in `v2/`, with changes marked
`\ROOF{}` (green) and text flagged for removal marked `\CUT{}` (red). Note that a colour set
inside a group does **not** survive a column break in this two-column layout, so coloured
paragraphs re-assert the marker at each paragraph or clause boundary.

## Reproduce

```bash
pip install -r requirements.txt
cd scripts/clean
python3 agentic_ecal.py              # derived-vs-measured coefficients, reduction to eCAL at K=1
python3 placement.py                 # tables/tab_placement.tex  (generated but currently unused)
python3 make_figures.py              # the shared figures + tab_dimensioning
python3 make_figures_placement.py    # fig_model_placement + tab_model_placement
python3 make_figure_workflow.py      # fig_workflow
cd ../v2 && latexmk -pdf main.tex
```

`make_figures_placement.py` reads the cached aggregate
`scripts/clean/results/steiner_tokens_by_cell.csv`, which is committed, so it runs without the
raw data. To rebuild that aggregate from scratch you need `~/DATA/data_steiner_Aug26` (132 GB,
not in the repo) and `python3 fit_peer_law.py`; it summarises the `teams_*.csv` files in chunks,
which are ~20x smaller than the `agents_*.csv` ones.

`make_figure_infra.py` waits on the infrastructure-benchmark runs. It expects
`scripts/clean/results/infra_runs.csv` with columns `topology, n_proposers, run_id, task_id,
difficulty, success, prefill_tokens, decode_tokens, wall_s`, and exits with that message if the
file is absent. Until then `tables/infra_macros.tex`, `tables/tab_infra.tex` and
`figures/fig_infra.pdf` are committed placeholders that render a conspicuous red `??` or
"pending benchmark runs", so no placeholder can ship unnoticed.

`agentic_ecal.py` is stdlib-only. `make_figures.py` and `placement.py` need numpy, pandas and
matplotlib, and the measured CSVs in `scripts/clean/results/`.

## Status

Research draft. Numeric constants and author lists should be re-verified before submission, and
energy (J) is kept separate from carbon (CO2eq) throughout.
