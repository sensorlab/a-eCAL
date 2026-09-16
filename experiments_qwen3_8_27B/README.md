# Qwen3.8-27B energy experiment

Per-query energy for `Qwen/Qwen3.8-27B` on **one A100 80GB PCIe**, measured the same
way as the Qwen2.5-7B / Llama-3.1-8B / Qwen3-8B batteries in `experimental_results/`.

This folder is **self-contained**: it imports nothing from `experimental_results/`
or `scripts/`. The measurement code is copied in (see `CHANGES.md` for the
provenance of each file), so this run can be reproduced from this directory alone
and cannot disturb the existing batteries.

## Run

```bash
cd experiments_qwen3_8_27B
pip install -r requirements.txt
nohup bash run_qwen38.sh > results/run_qwen38.log 2>&1 &
tail -f results/run_qwen38.log
```

Everything is driven from `run_qwen38.sh`, same convention as the other
experiments — env overrides, a smoke gate that aborts, one CSV per arm,
cell-resumable so you can interrupt and re-run:

```bash
REASONING_EFFORT=xhigh DECODE_BATCHES=2,4,8 bash run_qwen38.sh
THINKING_OFF=0 ROUNDS_LIST=1,2,3,4,5,6     bash run_qwen38.sh
SELFCORRECT=1                              bash run_qwen38.sh
EFFORT_SWEEP=1                             bash run_qwen38.sh   # low/medium/xhigh
```

### Hugging Face token

Qwen3.8-27B is Apache-2.0 and ungated, so **no token is needed for it**. Supply one for gated
repos (any Llama), private mirrors, or to lift the anonymous rate limit on the ~56 GB fetch:

```bash
HF_TOKEN_FILE=~/.hf_token bash run_qwen38.sh    # preferred: keeps it out of shell history
HF_TOKEN=hf_xxx bash run_qwen38.sh              # works, but lands in history
hf auth login                                    # cached in ~/.cache/huggingface/token
```

Precedence is `HF_TOKEN_FILE` > `HF_TOKEN` > cached login. The resolved token is exported to
both `preflight.py` and the vLLM subprocess, and the run log only ever prints it masked
(`hf token: set (hf_...ABCD, 22 chars)`).

Check it will work before booking the GPU for a day:

```bash
python3 preflight.py --model Qwen/Qwen3.8-27B --decode-batch 16 --max-model-len 8192
```

## What it produces

`results/Qwen3.8-27B_depth_{carry,free}_think{ON,OFF}.csv`, one row per
(task, decode_batch, team_size, rounds, run):

| column | meaning |
|---|---|
| `per_query_energy_j` | NVML board joules over the generation window, summed over rounds, / Q |
| `per_query_in_tok` / `per_query_out_tok` | vLLM's authoritative token counts |
| `per_query_think_tok` / `per_query_answer_tok` | the output split — cost vs. product |
| `accuracy` | final-round majority answer vs ground truth |
| `trunc_frac` | responses that hit the token cap (a data-validity signal, gated) |
| `reasoning_effort` | `low` / `medium` / `xhigh` — new axis, absent from the 8B battery |

`results/gpulog_Qwen3.8-27B.csv` carries power, temperature, SM clock and throttle
reasons every 2 s. `results/preflight_Qwen3.8-27B.json` records the architecture and
residency the run was admitted under.

## Why this is not just the 8B script with a different `--model`

Qwen3.8-27B breaks three assumptions the existing harness makes, each silently.
`CHANGES.md` lists every fix, the `file:line` it addresses, and what remains an
issue — including the parts of `scripts/clean/` that would misprice this model.
Read it before using the CSVs for anything.
