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
  agentic_ecal.py   the model: Hardware/LLM descriptors, two-rate call model, Workflow, the metric
  osi.py            eCAL Eq. (3), the OSI-layer transmission model, for E_tx
  placement.py      bearers, deployment archetypes, and the generated placement table
  make_figures.py   regenerates every figure and prints the numbers quoted in the paper
  results/          the measured sweeps (A100, vLLM): agent count, reasoning depth, serving batch
main.tex, refs.bib  the paper
figures/, tables/   generated artifacts, committed so the paper builds without running the model
```

## Reproduce

```bash
pip install -r requirements.txt
cd scripts/clean
python3 agentic_ecal.py     # derived-vs-measured coefficients, and the reduction to eCAL at K=1
python3 placement.py        # regenerates tables/tab_placement.tex
python3 make_figures.py     # regenerates all figures from results/
cd ../.. && latexmk -pdf main.tex
```

`agentic_ecal.py` is stdlib-only. `make_figures.py` and `placement.py` need numpy, pandas and
matplotlib, and the measured CSVs in `scripts/clean/results/`.

## Status

Research draft. Numeric constants and author lists should be re-verified before submission, and
energy (J) is kept separate from carbon (CO2eq) throughout.
