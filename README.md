# a-eCAL — Agentic-eCAL

Analytical model for **agentic-eCAL**, extending the
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

The model now lives at the repository root as a modular package, alongside the figure scripts,
the measured data, tests, and examples:

```
agentic_ecal_pkg/     the model, as a modular package  (import agentic_ecal_pkg as ae)
  descriptors.py        Hardware, LLM descriptors
  rates.py              the two-rate call model: prefill/decode rates, call energy & latency
  calib.py              TwoRateCalib (measured coefficients)
  components.py         Tool, Retrieval, Step
  workflow.py           Workflow  (.run() breakdown, .agentic_ecal() J/bit)
  registries.py         HW, LLMS, TWO_RATE presets (Python data)
  builders.py           four_agent_rag_workflow, tree_workflow, debate_workflow
  validation.py         literature-anchor calibration checks
  cli.py                `agentic-ecal` / `python -m agentic_ecal_pkg`  (add --json)
examples/             runnable usage examples
tests/                pytest parity suite: package == frozen reference, bit-exact (~3000 cases)
  agentic_ecal_reference.py   the original single-file model, frozen as the golden master
figure_scripts/       every figure/analysis script (import agentic_ecal_pkg)
  osi.py                eCAL Eq. (3), the OSI-layer transmission model, for E_tx
  placement.py          bearers, deployment archetypes, and the generated placement table
  model_placement.py    device tiers with MEMORY, 16-model fleet geometry, the measured peer law
  make_figures.py       the shared figures + tab_dimensioning
  make_figures_placement.py   fig_model_placement + tab_model_placement
  make_figure_workflow.py     fig_workflow, the case-study schematic
  make_figure_infra.py        fig_infra + tab_infra + infra_macros   (needs benchmark runs)
  fit_peer_law.py             fits the peer law from ~/DATA/data_steiner_Aug26 (see below)
results/              the measured sweeps (A100, vLLM), plus the cached Steiner aggregate
                        (not tracked in git; provided separately)
```

`agentic_ecal_pkg` is a **behaviour-preserving** refactor of the original single-file model (now
frozen as `tests/agentic_ecal_reference.py`): it re-exports the identical public API
(`import agentic_ecal_pkg as ae` is a drop-in) and produces bit-for-bit identical numbers, which
the parity test suite enforces against that reference.

The manuscript no longer lives here. This repository holds the model, the measured data, and the
scripts that generate the paper's figures and tables; the LaTeX sources are maintained separately.
Running the scripts recreates `figures/` and `tables/` from `results/`.

## Setup (uv)

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt       # numpy, pandas, matplotlib + pytest

# optional: install the model as a package (enables the `agentic-ecal` CLI from anywhere)
uv pip install -e .
```

The model itself is **pure standard library**; numpy/pandas/matplotlib are only needed to
regenerate the figures, and pytest only to run the tests.

## Use it

```python
import agentic_ecal_pkg as ae

wf = ae.four_agent_rag_workflow()          # the paper's case study (Llama-3 8B on A100)
print(wf.run()["e_total"], "J")            # 5240.26 J
print(wf.agentic_ecal(), "J/bit")          # 0.3022 J/bit
```

```bash
python examples/basic_call.py        # per-call / per-token energy across batches
python examples/case_study.py         # the four-agent RAG breakdown
python examples/custom_workflow.py    # build a Workflow from Steps by hand
python -m agentic_ecal_pkg            # derived-vs-measured coefficients, reduction to eCAL at K=1
python -m agentic_ecal_pkg --json     # the same, machine-readable
```

## Tests (pytest)

A golden-master / parity suite loads the frozen reference `tests/agentic_ecal_reference.py` and the new package side by
side and asserts bit-exact identical results across every public function, registry, workflow
builder, and the full `Workflow.run()` breakdown; `tests/test_paper_numbers.py` additionally pins
the anchors and case-study figures quoted in the manuscript.

```bash
pytest                                # ~3000 parametrized cases
```

## Reproduce the figures

```bash
python figure_scripts/make_figures.py              # the shared figures + tab_dimensioning
python figure_scripts/make_figures_placement.py    # fig_model_placement + tab_model_placement
python figure_scripts/make_figure_workflow.py      # fig_workflow
```

The scripts write into `figures/` (or the newest `vN/figures/`) and `tables/` at the repository
root, creating them if absent.

The `results/` directory is **not tracked in git** (see `.gitignore`); obtain the measured CSVs
separately and place them there. `make_figures_placement.py` reads the cached aggregate
`results/steiner_tokens_by_cell.csv`, so it runs from that aggregate alone, without the raw data.
To rebuild that aggregate from scratch you need
`~/DATA/data_steiner_Aug26` (132 GB, not in the repo) and `python figure_scripts/fit_peer_law.py`;
it summarises the `teams_*.csv` files in chunks, which are ~20x smaller than the `agents_*.csv` ones.

`make_figure_infra.py` waits on the infrastructure-benchmark runs. It expects
`results/infra_runs.csv` with columns `topology, n_proposers, run_id, task_id, difficulty,
success, prefill_tokens, decode_tokens, wall_s`, and exits with that message if the file is
absent. Until then it emits placeholders that render a conspicuous red `??` or "pending benchmark
runs", so no placeholder can ship unnoticed.

`results/infranet.png` is the infrastructure-benchmark result figure that motivates that section.

`agentic_ecal_pkg` is stdlib-only. The figure scripts need numpy, pandas
and matplotlib, and the measured CSVs in `results/`.

## Status

Research draft. Numeric constants and author lists should be re-verified before submission, and
energy (J) is kept separate from carbon (CO2eq) throughout.
