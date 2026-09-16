# Ringelmann — multi-agent LLM scaling law code

Runtime code for two papers:

- **Foundation** (NeurIPS): *The Ringelmann Effect in Multi-Agent LLM
  Systems: A Scaling Law for Effective Team Size.* Measures the
  Ringelmann scaling law `R(N) = N_eff / N = 1 / (1 + c (N - 1) N^{-β})`
  across homogeneous flat-team conditions (self-consistency,
  peer chat, LLM-aggregator) on MMLU-Hard, GSM-Hard, GPQA, ARC,
  GSM8K and several model families.
- **Extension** (ICLR submission): *Topology Beats Talk — Scaling Laws
  of Strict Hierarchical Multi-Agent LLM Systems.* Adds 9 hierarchical
  DAG topologies and asks whether they shift β beyond the foundation
  paper's hard-ceiling regime.

## Layout

```
ringelmann/                  installable package
├── inference/               vLLM OpenAI server, async client, pool, energy, csv writer  [NEW]
├── topology/                hierarchical DAG framework (9 topologies + prompts)         [NEW]
├── curves.py                Ringelmann fit + regime classification
├── load.py                  CSV loaders for the analyses scripts
├── stats.py                 bootstrap CIs, bias-corrected pairwise agreement
├── derive.py                variance-matching / Kish design effect
├── style.py                 matplotlib style + figure helpers
└── adapters/                Gemini and DeepSeek API adapters (optional)

runner/
├── ringelmann_runner.py     foundation flat-debate runner (with --backend vllm)
└── hierarchical_runner.py   hierarchical DAG runner (fully async, vLLM-native)          [NEW]

scripts/
├── smoke_test_vllm.py       inference-layer smoke test (5 generations)                  [NEW]
├── smoke_test_dag.py        DAG-construction smoke test (no LLM)
├── run_hier_pilot.sh        hierarchical pilot (5 items × N=3 × all topologies)         [NEW]
├── run_baseline_pilot.sh    foundation pilot via vLLM backend                           [NEW]
├── slurm_hier_1gpu.sh       full hierarchical sweep, single-GPU topologies              [NEW]
├── slurm_hier_2gpu.sh       Capability Star sweep, dual-GPU                             [NEW]
└── run_{qwen,llama,gemini}.sh    foundation HF-transformers sweep launchers

analyses/                    figure / table generation for the foundation paper
data/                        baseline CSVs from the foundation paper sweep — IGNORED IN GIT
results/                     created on first run; figures + tables produced here — IGNORED IN GIT
```

## Setup

```bash
cd ringelmann_code/
pip install -e .

# Required for the vLLM backend:
pip install vllm openai httpx jinja2 pynvml
```

Or with `uv`:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
uv pip install vllm openai httpx jinja2 pynvml
```

## Backend choices

The foundation runner accepts `--backend {local, vllm, gemini, deepseek}`:

- `local` — original HF transformers path (used in the NeurIPS sweep).
  Required if you need `--store_logits` (full per-token probabilities).
- `vllm` — drop-in via `ringelmann.inference.VLLMRunner`. Launches a
  subprocess vLLM OpenAI server and uses continuous batching for the
  `generate_batch` calls. Much faster on H100; logit storage disabled.
- `gemini` / `deepseek` — API backends (no torch needed).

The hierarchical runner is vLLM-only by design — it's fully async and
fans out across items concurrently.

## Quickstart: hierarchical pilot

```bash
# 1. Launch a vLLM server (example: Qwen3-8B on cuda:0, port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-Instruct --port 8000 --dtype bfloat16

# 2. Smoke-test the layer
VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_MODEL=Qwen/Qwen3-8B-Instruct \
    python3 scripts/smoke_test_vllm.py

# 3. Smoke-test all 9 DAG topologies build correctly (no LLM)
python3 scripts/smoke_test_dag.py

# 4. Run the hierarchical pilot
USE_EXISTING_SERVERS=1 WORKER_MODEL=Qwen/Qwen3-8B-Instruct \
    bash scripts/run_hier_pilot.sh

# 5. Foundation conditions on the same model
bash scripts/run_baseline_pilot.sh
```

## Quickstart: reproduce foundation paper figures (legacy)

This still works if you have the baseline CSVs (regenerate via
`bash scripts/run_qwen.sh` etc., or request from authors). The CSVs are
**not committed** — `*.csv` is in `.gitignore` at the repo root.

```bash
# Drop your baseline CSVs into data/, then:
bash scripts/make_all_figures.sh
```

Figures land in `results/figures/`.

## Hierarchical sweep on Slurm

```bash
# Single-GPU topologies (star, chain, tree, proposer_critic_star,
# persona_star, cascading_chain, tournament_star, diamond):
sbatch scripts/slurm_hier_1gpu.sh

# Capability Star (needs 2 GPUs — worker 7B on cuda:0, manager 32B on cuda:1):
sbatch scripts/slurm_hier_2gpu.sh
```

Override defaults via `sbatch --export=ALL,SAMPLES=100,NUM_RUNS=1 ...`.

## CSV schemas

Both runners write `teams_<model>.csv` and `agents_<model>.csv`. The
hierarchical runner extends the foundation schema additively:

- **Foundation columns** (`TEAM_FIELDS`, `AGENT_FIELDS` in
  `runner/ringelmann_runner.py:1286, 1308`) — 50+ columns per team row,
  34 per agent row. Preserved verbatim for cross-paper analysis
  continuity.
- **Hierarchical extras** (`HIER_TEAM_EXTRA`, `HIER_AGENT_EXTRA` in
  `runner/hierarchical_runner.py`) — adds `hier_topology`, `hier_tiers`,
  per-role counts (`hier_n_workers`, `hier_n_critics`,
  `hier_n_synthesizers`, `hier_n_refiners`, `hier_n_duelists`,
  `hier_n_planners`, `hier_n_plan_solvers`), `manager_answer`,
  `manager_conf`, `manager_model_id`, leaf-aggregate stats, and per-node
  role / tier / persona / parent_ids.

`comm_round = 0` for all hierarchical rows (hierarchies have no debate
rounds). `total_agents = len(dag.nodes)` (may differ from `team_size`
for Tournament Star, which rounds down to a valid bracket size).

## Hard constraints (for AI agents reviewing this code)

- **`runner/ringelmann_runner.py` is the foundation paper's runner.**
  Read-only for the extension paper EXCEPT for the `--backend vllm`
  enablement that was added. Don't restructure its condition functions
  without explicit user instruction; the NeurIPS reviewers verified the
  current implementation.
- **`ringelmann/topology/dag.py` and `prompts.py` are stable.** New
  topologies should be added as new classes registered in
  `TOPOLOGY_REGISTRY`; do not mutate existing ones.
- **CSV schema is additive only.** Do not delete or rename columns from
  the foundation `TEAM_FIELDS` / `AGENT_FIELDS`. Append new columns to
  the `HIER_*_EXTRA` lists.

## Citation

Anonymized for review.

## License

MIT (see `LICENSE`).
