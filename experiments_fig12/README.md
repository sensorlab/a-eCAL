# Adding Qwen3.5-9B and Qwen3.8-27B to Figures 1 and 2

Measurement runs for the two extra models, using the **fixed** reasoning harness rather
than the one that produced `experiments_qwen35_9B/`.

## Why this folder exists

`experimental_results/scripts/run_qwen3_reasoning.sh` produced the existing Qwen3.5-9B
CSVs and has three defects that make them unusable for the figures:

| defect | evidence in `experiments_qwen35_9B/` |
|---|---|
| think/answer split not recognised | `per_query_think_tok` averages 2.5–3.3 tok in **both** arms while `answer_tok` takes everything — the whole chain of thought is booked as "answer" |
| truncation logged, not gated | `trunc_frac` > 0 in **211 of 216 rows**, mean 0.4 with thinking on, max 0.875 |
| single serving batch | `decode_batch=64` only, so `c_dec(b) = a_dec/b + c₀` is not identifiable |

`experiments_qwen3_8_27B/` already fixed all three (its `CHANGES-upstream.md`, copied here,
documents each with a `file:line`), and its 27B output proves it: the split works
(1246/153 tok with thinking on, 3/2178 off) and `trunc_frac` maxes at 0.047. So the code
here is that folder's, plus three small patches to `run_battery.sh`:

- `OUTDIR` is overridable (was hardcoded `results`)
- `TAG` suffixes every output path, so two axes of the *same* model don't overwrite each other
- `THINKING_ON=0` skips the thinking arms, which the upstream script always ran

Nothing else was touched. `CHANGES-upstream.md` is the provenance record for the rest.

## What each figure panel needs, and which run supplies it

| panel | consumes | run |
|---|---|---|
| Fig. 1(a) | breadth of `(p_in, p_out, b)` | A + B (+ C, D) |
| Fig. 1(b) | a **batch axis**; regresses `E = α·p_in + β·p_out` per batch | **A**, C |
| Fig. 2(a) | depth carry vs free at **b=256** | **B**; D as negative control |
| Fig. 2(b) | energy vs `team_size` at two batches | **A**, C |
| Fig. 2(c) | log-log slope of `E` vs `team_size`, **per batch** | **A**, C |

Fig. 1 needs no team-size axis. Fig. 2(b) and (c) do — that is the extra cost.

## Order

Two scripts, one per model. They share `_runner.sh`, so they cannot drift apart — which is the
whole point, since the value of these runs is that both models are measured identically to each
other and to the earlier battery.

```bash
nohup bash run_9b.sh  > results/run_9b.log  2>&1 &   # stages A, B
nohup bash run_27b.sh > results/run_27b.log 2>&1 &   # stages C, D
```

**Not at the same time** — each stage boots its own vLLM server on `GPU_INDEX`, so on one card
they must run back to back. Each script preflights its own model, derives `EXPECT_FULL_LAYERS`
from the JSON (the two models differ: 8 vs 16), runs its stages, then verifies its own outputs.

```bash
STAGES=A              bash run_9b.sh     # one stage
STAGES=A,B,E          bash run_9b.sh     # + held-out thinking
C_BATCHES=2,4,8,16,64 bash run_27b.sh    # try to extend the 27B batch axis
C_TEAMS=2,5,10        bash run_27b.sh    # if N=20/30 fail residency
GPU_INDEX=1           bash run_9b.sh
```

| stage | script | axis | feeds |
|---|---|---|---|
| A | `run_9b.sh` | agent count, b=2..256 | Fig 1(b), Fig 2(b), Fig 2(c) |
| B | `run_9b.sh` | depth at b=256 | Fig 2(a) |
| C | `run_27b.sh` | agent count, b≤16 | Fig 1(b), Fig 2(b), Fig 2(c) |
| D | `run_27b.sh` | depth to rounds 6 | Fig 2(a), negative control |
| E | either, opt-in | thinking held-out | stronger Fig 1(a); never fit on it |

Everything is **cell-resumable** — re-running fills only the gaps. But if a stage failed because
a *parameter* changed, delete its `results/` subdirectory first: resume skips cells by key, so
rows measured under the old parameter would survive in the middle of the grid.

Per-stage logs land in `results/logs/`. A stage is reported INCOMPLETE if the wrapper exits
non-zero, **or** wrote no rows, **or** the driver printed its own `FAILED` marker — that last one
matters because `run_battery.sh` continues past a failed arm and can exit 0 holding a partial CSV.

## The generation cap had to be raised — and that IS the matching choice

`exp_vllm_debate.py` used `max_new_tokens=1024`. For its two models that cap **never bound**:

```
Qwen2.5-7B     mean 110 output tok/response,  max 144   against a 1024 cap
Llama-3.1-8B   mean 170,                      max 254
Qwen3.5-9B     mean ~255,                     11.5% of responses hit 1024   (measured 2026-09-09)
```

It was a safety limit, not an experimental treatment. It binds hard for Qwen3.5-9B, whose
single answers ran 468–552 tokens with thinking OFF. Holding 1024 would censor the new models'
output while the old models' was untouched — a **larger** deviation than changing the number,
because it alters the experimental condition rather than a slack parameter.

So runs A–D use `MAX_NEW_TOKENS=4096`, `MAX_MODEL_LEN=16384` (which must clear the worst prompt,
2,816 tok on gpqa, plus the 3,856-token revise margin at N=30, plus max_new). `MAX_TRUNC_FRAC=0.10`
stays as the gate that checks the new cap is actually non-binding. Every other parameter still
comes from `_original_battery.env`.

Note this matters much less for Fig. 1 than for Fig. 2: Fig. 1 regresses energy on *measured*
`(p_in, p_out)`, so a truncated response is still internally consistent. Fig. 2(a)'s depth
scaling and any accuracy claim are what a biting cap would distort.

## The truncation gate is off for the thinking-OFF stages

`MAX_TRUNC_FRAC=1.0` in runs A–D, matching the published battery, which had no such gate.
`trunc_frac` is still written on every row and `verify_results.py` flags anything above 0.10 at
the end of a run, so it is a report rather than an abort. Run E keeps `0.10`.

The gate's premise — truncation cuts mid-`<think>`, `strip_thinking_tags()` leaves no answer, the
row scores `answer_tok=0` and counts as wrong — is thinking-specific and false with thinking off:
`think_tok=1`, nothing to strip, `answer_tok` populated, accuracy normal. Measured:

| task | trunc_frac | per-response output |
|---|---|---|
| mmlu_hard | 0.01 – 0.02 | ~348 tok |
| gsmhard | 0.04 – 0.08 | ~613 tok |
| **gpqa** | **0.10 – 0.18** | **~1,173 tok** |

**gpqa truncates 10–18% of responses at 4096 tokens.** That is fine for Figs. 1 and 2, which
regress energy on *measured* `p_out` — a capped response is still a consistent observation — and
it is not a basis for any accuracy claim from these rows. Raising the cap is the wrong lever: the
tail is runaway generations at 5–10× the mean, so 8192 would double the cost of exactly those
and might still trip a finite threshold.

## Check which version is running

`run_battery.sh` prints two config lines. The second one exists because on 2026-09-10 a run
executed for ten hours against a stale copy while the first line looked entirely correct — it
prints only knobs that had not changed:

```
[..] effort=medium mml=16384 max_new=4096 Q=32 runs=3
[..] trunc=1.0 client_timeout=18000 arms=free thinkON=0 seats_advisory=1
```

If the second line says `trunc=0.10`, the copy did not land. Check it before walking away.

## Timeouts

`VLLMClient` defaults to `timeout_s=600` and `generate_batch` fires all Q·N prompts at once with
no semaphore, so at small batch the tail request waits most of a cell. `CLIENT_TIMEOUT` is 18000
(5 h) for the 9B and 43200 (12 h) for the 27B, whose cells are longer.

This is insurance, not a cure: the one `APITimeoutError` observed (2026-09-09, N=30 b=2) did not
recur — the same cell completed in 7151 s the next day under the same 600 s timeout — so it was a
stalled request, not a systematic queue-tail failure.

## Two preflight gates are advisory, on purpose

The 27B fork's guards are calibrated for the 27B. Two of them block runs that are legitimate
here, so both became warnings that still appear in the log and in the preflight JSON.

**`reasoning_effort`.** Qwen3.5-9B's chat template ignores it — low/medium/xhigh render
identically. That is fatal to a run that *sweeps* effort and irrelevant to one that does not.
Stages A–D run thinking OFF and never pass the flag; `exp_qwen38.py:271` already makes exactly
this distinction (`if args.reasoning_effort and not args.skip_template_check`), preflight did
not. Now `--require-reasoning-effort` promotes it back to a failure, and `run_all.sh` passes
that only when stage E is selected. Stage E then omits the flag entirely for a model that
ignores it, rather than writing an effort into the CSV that the model did not use.
`enable_thinking` stays a hard failure — that one is honored by both models, and it is the bug
that produced the unusable `experiments_qwen35_9B/` data.

**The seat budget.** It assumes every sequence fills `max_model_len`, which no sequence does:

```
Qwen2.5-7B (original)   ctx 8192 -> 448 MiB/seq -> 125 seats   worst case
                        ctx 2500 -> 137 MiB/seq -> 410 seats   what the agents run actually uses
Qwen3.5-9B (run A)      ctx 8192 -> 256 MiB/seq -> 208 seats   worst case
                        ctx 2500 ->  78 MiB/seq -> 683 seats   actual
```

The published Qwen2.5-7B `agents_*.csv` has b=256 rows, and the worst-case bound denies b=256
to it too. Putting a new model on that same axis means reproducing that, so runs A and C set
`SEATS_ADVISORY=1`. The CSV records the requested batch, as the original CSVs do. `seats` now
goes into the preflight JSON so the number is on the record either way.

**The smoke gate.** It hardcoded thinking-ON, which breaks a thinking-OFF battery three ways at
once: `--reasoning_effort` trips `assert_template_honors_kwargs` on a model that ignores it, an
empty `TOP_K`/`MIN_P` reaches argparse as `""`, and a reasoning model given `max_new=1024`
truncates past `max_trunc_frac`. It now builds its flags from `common_flags` in whichever mode the
battery is about to run, so the gate tests the configuration that follows it.

## Two things that are physics, not bugs

**The 27B cannot do Fig. 2(a).** One 80 GB A100 caps it near 34 sequences at
`max_model_len=8192`, so b=256 is unreachable and it never enters the prefill-dominated
regime. Measured on the existing data, carry/free never separates:

```
b=  2   carry/free  0.95  1.05  1.13  0.93
b=  4               1.05  1.00  1.00  0.89
b=  8               1.12  1.07  0.93  0.94
b= 16               0.92  0.79  1.04  0.94
Qwen2.5-7B, b=256   0.93  0.99  1.07  1.06  1.18  1.33   <- the reference
```

That is §III-B's own claim — context grows quadratically unconditionally, energy inherits
it only when prefill-dominated — so run D reports it as the negative control. To get a real
27B panel-(a) series instead, run it at TP=2 or on an H100.

**The 27B's `kv_floor` will be weakly constrained** in Fig. 1(b), because its batch axis
stops at 16. Say so in the caption; do not extrapolate the fitted curve past the data.

## After the runs

Three things stand between these CSVs and the figures, all recorded in
`CHANGES-upstream.md` under *"`scripts/clean/` will misprice this model"*:

1. **`scripts/clean/fit_two_rate.py` does not exist.** `TWO_RATE` holds hand-transcribed
   literals and `TwoRateCalib`'s docstring claims a least-squares fit no code performs.
   Write it (per-batch `(α,β)` via `make_figures.py:165 measured_rates()`, then fit
   `β(b) = a_dec/b + kv_floor`) and check it reproduces `0.023/3.2/0.02` for Qwen2.5-7B
   before trusting it on a new model.
2. **Pricing.** `kv_bytes_per_token` (`agentic_ecal.py:116`) assumes every layer caches —
   4× error on a hybrid model; `BITS_PER_TOKEN` (`:46`) is a global 17.0 against 16 in the
   manuscript and 17.92 for the 27B vocab; `HW["a100"]` (`:88`) holds SXM specs while these
   runs are 80 GB PCIe. `runs/00_preflight.sh` prints the first two.
3. **`MEASURED` in `make_figures.py:66`** is a positional tuple assuming one agents CSV and
   a depth *pair* per model. Fig. 2(b) hardcodes index 0; Fig. 2(c) styles on
   `label.startswith("Qwen")`, which three of four models now match.

Fig. 2(c) is the one panel needing no calibration at all — it is a pure log-log slope, and
η cancels from it — so A and C can enter it before any coefficients are fitted.
