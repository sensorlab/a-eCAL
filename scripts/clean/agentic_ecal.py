"""Analytical energy model for agentic workflows of open-weight LLMs (agentic-eCAL).

This module extends the eCAL methodology (Chou et al., IEEE JSAC 2026) from a single AI-model
lifecycle to an *agentic workflow*: a directed graph of LLM calls, tool invocations, retrievals,
and inter-agent messages built on a pre-trained open-weight model.

It is the computational backing for the paper "Agentic-eCAL: The Energy Cost of Agentic AI
Workflows with Open-Weight Models". It contains the model only -- no plotting, no data loading,
no pandas/matplotlib import -- so it can be imported and tested on its own. Figures live in
``make_figures.py``.

The call model is SINGLE-RATE: energy is FLOPs / effective-FLOP/s * power, with utilisation
``eta`` (MFU) as the single calibration knob. It therefore has no serving-batch term; that
limitation is a stated result of the paper, quantified by ``make_figures.fig_validation``.

Numeric constants (J/token, MFU, pre-training energy) are grounded in the literature; see
``docs/research-notes.md`` for the source of each value. All energies are in Joules unless noted.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "Hardware", "LLM", "Tool", "Retrieval", "Step", "Workflow",
    "HW", "LLMS", "MFU_LATENCY", "MFU_THROUGHPUT", "BITS_PER_TOKEN", "GWH_TO_J",
    "prefill_rate", "decode_rate", "llm_call_energy", "call_latency",
    "joules_per_token",
    "embedding_flops", "validate_two_rate", "TwoRateCalib", "TWO_RATE",
    "four_agent_rag_workflow", "tree_workflow", "debate_workflow",
    "validate_against_samsi", "validate_current_gen",
]


# ---------------------------------------------------------------------------
# Token -> bit conversion
# ---------------------------------------------------------------------------
#
# f is the information-theoretic capacity log2(|V|) of a token drawn from a vocabulary of size
# |V|: ~17 bits for the 128k-152k vocabularies of Llama-3 / Qwen-2.5. It is a pure denominator
# scale, so it cancels from every scaling exponent and sets only the absolute J/bit level.

BITS_PER_TOKEN = 17.0


# ---------------------------------------------------------------------------
# Hardware / model descriptors
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    """Accelerator descriptor. ``flops_peak`` in FLOP/s (fp16/bf16), ``power`` in Watts.

    ``mfu`` (model-FLOP utilization, achieved/peak) is the calibration knob. Single-stream,
    latency-bound decode is memory-bound and reaches only ~5% MFU; throughput-optimized batched
    serving reaches ~30-50%. We calibrate it against measured per-token energy (research-notes.md)
    rather than asserting peak FLOP/s, because estimating from nameplate TDP overestimates measured
    energy by up to 4.1x (ML.ENERGY, arXiv:2505.06371).
    """
    name: str
    flops_peak: float          # peak dense FLOP/s at the working precision
    power: float               # board power drawn while busy [W]
    mfu: float = 0.10          # default: mid latency/throughput regime
    bandwidth: float = 0.0     # B_HBM, peak HBM bandwidth [byte/s]
    eta_pre: float = 0.85      # prefill utilisation; compute-bound, so near peak
    bw_efficiency: float = 0.70   # sustained fraction of peak HBM bandwidth

    @property
    def flops_eff(self) -> float:
        """Effective (achieved) FLOP/s = peak * utilisation."""
        return self.flops_peak * self.mfu


# Representative accelerators. Specs from vendor datasheets (see research-notes.md).
# H100 SXM bf16 dense ~989 TFLOP/s (with sparsity ~1979); board power ~700 W.
# A100 80GB bf16 dense ~312 TFLOP/s; board power ~400 W (SXM).
# MFU regimes calibrated so the 65B model reproduces Samsi et al.'s measured 3-4 J/token anchor
# at MFU_LATENCY (single-stream, unbatched), with batched serving at MFU_THROUGHPUT.
MFU_LATENCY = 0.05      # single-stream / latency-bound decode (reproduces 3-4 J/token for 65B, Samsi 2023)
MFU_THROUGHPUT = 0.25   # throughput-optimized batched serving (reproduces ~0.39 J/token Llama-3 70B, 2026)

HW = {
    "h100": Hardware("NVIDIA H100 SXM", flops_peak=989e12, power=700.0, mfu=MFU_LATENCY,
                     bandwidth=3.35e12),
    "a100": Hardware("NVIDIA A100 80GB", flops_peak=312e12, power=400.0, mfu=MFU_LATENCY,
                     bandwidth=2.039e12),
}


@dataclass
class LLM:
    """Open-weight decoder-only LLM descriptor.
    """
    name: str
    n_params: float            # total parameters
    n_layers: int              # transformer blocks
    d_model: int               # hidden size
    e_pretrain: float = 0.0    # embodied (pre-training) energy [J], 0 if unknown

    # Active (non-embedding) parameters per token. For dense models == n_params;
    # for Mixture-of-Experts this is the routed/active subset.
    n_active: Optional[float] = None
    d_kv: Optional[int] = None      # n_kv_heads * head_dim; equals d_model under multi-head
    bytes_per_param: float = 2.0    # beta, element size of weights and cache (bf16 = 2)

    def __post_init__(self) -> None:
        if self.n_active is None:
            self.n_active = self.n_params
        if self.d_kv is None:
            self.d_kv = self.d_model

    @property
    def kv_bytes_per_token(self) -> float:
        """gamma: KV-cache bytes read per decoded token, per token of context."""
        return 2.0 * self.n_layers * self.d_kv * self.bytes_per_param


# Representative open-weight models. Architectures from official model cards; pre-training
# (embodied) energy = GPU-hours x 700 W from the Llama-3 model card (research-notes.md):
#   8B  : 1.3M H100-h x 700 W = 0.91 GWh = 3.28e12 J
#   70B : 6.4M H100-h x 700 W = 4.48 GWh = 1.61e13 J
GWH_TO_J = 3.6e12  # 1 GWh = 3.6e12 J
LLMS = {
    # Llama-3 8B / Llama-3.1-8B-Instruct: 32 layers, d_model 4096. Also the measured model.
    # GQA: 8 KV heads x head_dim 128 -> d_kv = 1024.
    "llama3_8b": LLM("Llama-3 8B", n_params=8.03e9, n_layers=32, d_model=4096,
                     e_pretrain=0.91 * GWH_TO_J, d_kv=1024),
    # Llama-3 70B: 80 layers, d_model 8192.
    "llama3_70b": LLM("Llama-3 70B", n_params=70.6e9, n_layers=80, d_model=8192,
                      e_pretrain=4.48 * GWH_TO_J, d_kv=1024),
    # LLaMA-65B: 80 layers, d_model 8192. Validates eta against Samsi et al. (2023), and carries
    # the first-generation embodied cost in the amortization figure.
    #   65B : 1,022,362 A100-h x 400 W = 0.409 GWh = 1.47e12 J
    # Same convention as the Llama-3 entries above (GPU-hours x TDP, no PUE), which is why this
    # is below the 449 MWh the LLaMA paper reports for the same run.
    "llama_65b": LLM("LLaMA 65B", n_params=65.2e9, n_layers=80, d_model=8192,
                     e_pretrain=0.409 * GWH_TO_J),
    # Qwen2.5-7B-Instruct: 28 layers, d_model 3584 = 28 query heads x head_dim 128. GQA with
    # 4 KV heads, which cuts KV-cache traffic 7x but NOT the attention FLOPs below (every query
    # head still scores against the full key sequence), so d_model is the right width here.
    # The second measured model; the notebook referenced this key but it was missing from the dict.
    # GQA: 4 KV heads x 128 -> d_kv = 512, half of Llama-3.1-8B despite similar size.
    "qwen2_5_7b": LLM("Qwen2.5-7B", n_params=7.615e9, n_layers=28, d_model=3584,
                      e_pretrain=0.0, d_kv=512),
}


# ---------------------------------------------------------------------------
# Two-rate call model (paper Eqs. 3-5)
# ---------------------------------------------------------------------------
#
# A call has two phases with different energy characteristics, so one rate cannot describe both.
#
#   E_call(p_in, p_out; b) = c_pre * p_in + c_dec(b) * p_out                        (Eq. 3)
#
# PREFILL processes the whole prompt in one parallel pass. It is compute-bound at near-peak
# utilisation and essentially batch-independent:
#
#   c_pre = 2 N P / (eta_pre * Pi)                                                  (Eq. 4)
#
# DECODE emits one token at a time and is memory-bandwidth-bound. Each step streams the weight
# bytes from HBM once for the whole batch, so that cost amortises over the b concurrently
# decoding sequences, while every sequence re-reads its own cache and that term does not:
#
#   c_dec(b) = P / (eps B_HBM) * ( N beta / b  +  gamma * cbar )                    (Eq. 5)
#
# eps is the fraction of peak HBM bandwidth production kernels sustain, conventionally
# 0.65-0.85. Omitting it (eps = 1) makes the model under-predict measured energy by 17-66%;
# at eps = 0.70 the fit over 756 measured configurations is R^2 = 0.99 with no free
# parameter. It is a datasheet-adjacent constant, not a calibration knob.
#
# NOTE on the weight term. The paper writes 2N beta, carrying the "2N" of the FLOP convention
# into a byte count. The weight footprint is N beta bytes, not 2N beta: for Qwen2.5-7B at bf16
# that is 15.2 GB, giving P N beta / B_HBM = 2.99 J against a measured a_dec of 3.2 J, whereas
# 2N beta would give 5.98 J. This module uses N beta; see validate_two_rate().

def prefill_rate(model: LLM, hw: Hardware) -> float:
    """c_pre [J per prompt token]: compute-bound, batch-independent (Eq. 4)."""
    return 2.0 * model.n_active * hw.power / (hw.eta_pre * hw.flops_peak)


def decode_rate(model: LLM, hw: Hardware, batch: float = 1.0, context: float = 0.0) -> float:
    """c_dec(b) [J per generated token]: memory-bound, amortising over the batch (Eq. 5)."""
    if hw.bandwidth <= 0.0:
        raise ValueError(f"{hw.name} has no HBM bandwidth set")
    weights = model.n_active * model.bytes_per_param / max(float(batch), 1.0)
    cache = model.kv_bytes_per_token * context
    return hw.power * (weights + cache) / (hw.bandwidth * hw.bw_efficiency)


def call_latency(model: LLM, hw: Hardware, p_in: float, p_out: float,
                 batch: float = 1.0) -> float:
    """Wall-clock time of one call [s], which is not energy divided by power.

    Energy per request divides by the serving batch, because the weight read of a decode step is
    shared by every sequence in flight. Latency does not: the step occupies its full duration for
    all of them. Prefill is one compute-bound pass; each decode step costs the weight read plus
    the cache read of the whole batch.
    """
    b = max(float(batch), 1.0)
    cbar = p_in + p_out / 2.0
    t_pre = 2.0 * model.n_active * p_in / (hw.eta_pre * hw.flops_peak)
    step = (model.n_active * model.bytes_per_param
            + b * model.kv_bytes_per_token * cbar) / (hw.bandwidth * hw.bw_efficiency)
    return t_pre + p_out * step


def llm_call_energy(model: LLM, hw: Hardware, p_in: float, p_out: float,
                    batch: float = 1.0) -> float:
    """Energy [J] of one LLM call under the two-rate model (Eq. 3).

    The cache term is evaluated at the call's mean context, p_in + p_out/2, since the transcript
    grows by one token per decode step.
    """
    cbar = p_in + p_out / 2.0
    return prefill_rate(model, hw) * p_in + decode_rate(model, hw, batch, cbar) * p_out


def joules_per_token(model: LLM, hw: Hardware, p_in: float = 512, p_out: float = 512,
                     batch: float = 1.0) -> float:
    """Convenience: energy per generated token for a representative call."""
    return llm_call_energy(model, hw, p_in, p_out, batch) / p_out


@dataclass(frozen=True)
class TwoRateCalib:
    """Two-rate coefficients [J/token] measured directly on the A100 sweeps.

    ``c_dec(b) = a_dec / b + kv_floor``. These are fitted, not derived: they are what a
    least-squares fit of Eq. (3) to the measured per-query energies returns, averaged over
    GPQA / GSM-Hard / MMLU-Hard under vLLM. They are the calibration counterpart of Eqs. (4)-(5),
    and the validation figure uses them so that the figure tests the two-rate *form* independently
    of the hardware derivation.
    """
    name: str
    c_pre: float
    a_dec: float
    kv_floor: float = 0.0

    def c_dec(self, batch: float) -> float:
        return self.a_dec / max(float(batch), 1.0) + self.kv_floor

    def energy(self, p_in: float, p_out: float, batch: float) -> float:
        """Eq. (3) with measured coefficients."""
        return self.c_pre * p_in + self.c_dec(batch) * p_out


# Qwen shows no measurable cache floor over the contexts swept; Llama's wider KV does.
TWO_RATE = {
    "qwen2_5_7b": TwoRateCalib("Qwen2.5-7B", c_pre=0.023, a_dec=3.2, kv_floor=0.00),
    "llama3_8b": TwoRateCalib("Llama-3.1-8B", c_pre=0.028, a_dec=3.3, kv_floor=0.06),
}


def embedding_flops(n_params: float, n_tokens: float) -> float:
    """Forward FLOPs of a small encoder, used for the retrieval term (2 N p_q)."""
    return 2.0 * n_params * n_tokens


# ---------------------------------------------------------------------------
# Workflow components: tools, retrieval, transmission
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A non-LLM tool invocation (code exec, API, search). ``energy`` in Joules per call."""
    name: str
    energy: float = 0.0
    obs_tokens: int = 0        # tokens the tool result adds to the running context


@dataclass
class Retrieval:
    """RAG retrieval step: query embedding, vector-index search, and injection of k chunks.

    The embedder is a distinct bi-encoder rather than the generator. The default values
    correspond to a BERT-base-class encoder: bge-base-en-v1.5, e5-base and all-mpnet-base-v2 each
    comprise approximately 109M parameters at 768 dimensions. Accordingly, ``dim`` bears no
    relation to the generator's hidden size (4096 for Llama-3 8B, 3584 for Qwen2.5-7B), and
    ``chunk_tokens`` remains within the 512-token input limit of such an encoder.

    The energy of retrieval itself is negligible: 0.199 J, or 0.013% of the case-study workflow.
    The dominant cost is the ``added_context`` injected into each subsequent prompt. Those same
    1280 tokens incur approximately 300 J of prefill in the generator, some three orders of
    magnitude above the retrieval term. ``k_chunks`` and ``chunk_tokens``, expressed in the
    generator's tokenizer, are therefore the only parameters of material consequence:
    substituting bge-large for MiniLM alters the workflow total by less than 0.05%, and
    ``n_vectors`` is inert.

    Two simplifications follow from this insensitivity and are recorded rather than corrected.
    The encoder is priced at the generator's accelerator and utilisation, which a model of 110M
    parameters would not attain. Only the query embedding is charged; embedding the corpus is a
    genuine cost, but one incurred offline and amortised across all queries.
    """
    embed_params: float = 1.1e8     # bi-encoder parameters (BERT-base class, approx. 109M)
    query_tokens: int = 64
    n_vectors: float = 1e6          # index size; immaterial to the result (see above)
    dim: int = 768                  # bi-encoder embedding width, not the generator hidden size
    k_chunks: int = 5               # k_chunks * chunk_tokens is appended to the prompt,
    chunk_tokens: int = 256         # expressed in the generator's tokenizer
    ann: bool = True                # approximate or exhaustive search; both are dominated
                                    # by the embedding term

    def energy(self, hw: Hardware) -> float:
        embed_flops = embedding_flops(self.embed_params, self.query_tokens)
        if self.ann:
            # HNSW-style: ~ dim * log2(n_vectors) distance ops, *2 FLOPs each
            search_flops = 2.0 * self.dim * math.log2(max(self.n_vectors, 2.0))
        else:
            search_flops = 2.0 * self.dim * self.n_vectors
        return (embed_flops + search_flops) / hw.flops_eff * hw.power

    @property
    def added_context(self) -> int:
        return self.k_chunks * self.chunk_tokens


@dataclass
class Step:
    """One workflow step executed by an agent: an LLM call plus optional tool/retrieval."""
    agent: str = "main"
    p_in_local: int = 0        # *new* prompt tokens contributed at this step (excl. carried history)
    p_out: int = 256           # generated tokens
    tool: Optional[Tool] = None
    retrieval: Optional[Retrieval] = None


@dataclass
class Workflow:
    """An agentic workflow over an open-weight model.

    ``carry_history`` toggles whether each step's prompt accumulates the running transcript
    (the dominant driver of super-linear token growth in agent loops).

    Three further fields describe *how much* transcript is retained, because all-or-nothing does
    not describe measured agent protocols. Steps are grouped into rounds of ``round_size``; a step
    sees the outputs of the ``history_rounds`` most recent *completed* rounds, and never those of
    its own round (agents within a round act in parallel).

      ``round_size``     steps per round; 0 or 1 means every step is its own round.
      ``history_rounds`` rounds of transcript retained. None retains all of them, which with
                         ``round_size`` = 1 is the original accumulating behaviour; 1 retains only
                         the previous round; 0 retains none, equivalent to carry_history = False.
      ``exclude_self``   within the retained window, skip outputs the same agent produced. This is
                         the difference between an agent re-reading its own reasoning and a debate
                         participant reading only its N-1 peers.

    The defaults reproduce the original semantics exactly.
    """
    model: LLM
    hw: Hardware
    steps: List[Step] = field(default_factory=list)
    sys_tokens: int = 400              # system / tool-schema prompt prepended to every call
    carry_history: bool = True
    history_rounds: Optional[int] = None   # None = every completed round
    round_size: int = 1                    # steps per round (team width in a debate)
    exclude_self: bool = False             # skip the agent's own output within the window
    serving_batch: float = 1.0         # b in Eq. (5); decode amortises over it
    calib: Optional["TwoRateCalib"] = None   # measured coefficients; None uses Eqs. (4)-(5)
    gamma_v: float = 0.10              # virtualization / orchestration overhead (eCAL's gamma_v)
    tx_energy_per_message: float = 0.0  # inter-agent transmission energy [J] (OSI model)

    def run(self) -> dict:
        """Compute the full energy breakdown of the workflow."""
        rs = max(self.round_size, 1)
        transcript = []      # (round index, agent, tokens) for outputs and tool observations
        e_llm = e_tool = e_ret = e_tx = 0.0
        e_prefill = e_decode = 0.0   # the two terms of Eq. (3)
        t_llm = 0.0          # request's share of device time [s]
        t_wall = 0.0         # wall-clock time of the calls, run back to back [s]
        useful_out = 0
        prefill_tokens = decode_tokens = 0
        prev_agent = None

        for k, st in enumerate(self.steps):
            rnd = k // rs

            # retrieval (RAG) before the call
            added = 0
            if st.retrieval is not None:
                e_ret += st.retrieval.energy(self.hw)
                added += st.retrieval.added_context

            # carried transcript: completed rounds inside the retention window only
            carried = 0
            if self.carry_history:
                lo = 0 if self.history_rounds is None else rnd - self.history_rounds
                carried = sum(tok for (r, ag, tok) in transcript
                              if lo <= r < rnd and not (self.exclude_self and ag == st.agent))

            # prompt = system + carried history + new local prompt + retrieved context
            p_in = self.sys_tokens + st.p_in_local + added + carried

            if self.calib is not None:
                pre = self.calib.c_pre * p_in
                dec = self.calib.c_dec(self.serving_batch) * st.p_out
            else:
                cbar = p_in + st.p_out / 2.0
                pre = prefill_rate(self.model, self.hw) * p_in
                dec = decode_rate(self.model, self.hw, self.serving_batch, cbar) * st.p_out
            e_prefill += pre
            e_decode += dec
            e_call = pre + dec
            e_llm += e_call
            t_llm += e_call / self.hw.power
            t_wall += call_latency(self.model, self.hw, p_in, st.p_out,
                                   self.serving_batch)
            prefill_tokens += p_in
            decode_tokens += st.p_out
            useful_out += st.p_out

            # tool execution after the call
            if st.tool is not None:
                e_tool += st.tool.energy
                transcript.append((rnd, st.agent, st.tool.obs_tokens))

            # inter-agent hand-off transmission
            if prev_agent is not None and st.agent != prev_agent:
                e_tx += self.tx_energy_per_message
            prev_agent = st.agent

            transcript.append((rnd, st.agent, st.p_out))  # joins the transcript

        e_core = e_llm + e_tool + e_ret + e_tx
        e_total = (1.0 + self.gamma_v) * e_core
        return {
            "e_llm": e_llm,
            "e_prefill": e_prefill,
            "e_decode": e_decode,
            "t_llm": t_llm,
            "t_wall": t_wall,
            "e_tool": e_tool,
            "e_retrieval": e_ret,
            "e_transmission": e_tx,
            "e_orchestration": self.gamma_v * e_core,
            "e_total": e_total,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "useful_output_tokens": useful_out,
        }

    def agentic_ecal(self, bits_per_token: float = BITS_PER_TOKEN,
                     e_emb_amortized: float = 0.0) -> float:
        """agentic-eCAL in J/bit: total workflow energy per useful output bit.

        ``e_emb_amortized`` is the share of the open-weight model's embodied (pre-training)
        energy attributed to this single workflow invocation.
        """
        r = self.run()
        useful_bits = bits_per_token * r["useful_output_tokens"]
        return (r["e_total"] + e_emb_amortized) / useful_bits if useful_bits else float("nan")


# ---------------------------------------------------------------------------
# The paper's case-study workflow
# ---------------------------------------------------------------------------

def four_agent_rag_workflow(model: Optional[LLM] = None, hw: Optional[Hardware] = None,
                            rounds: int = 1) -> Workflow:
    """Planner -> retriever -> analyst -> writer, carrying history, repeated ``rounds`` times.

    ``rounds=1`` is the single-pass case study; larger values re-run the same team over the
    accumulated transcript, which is the depth axis of the composition figure.

    Single definition of the case study, so the component breakdown, the amortization curve and
    the numbers quoted in the text cannot drift apart. Callers choose the model and hardware;
    the token budget and topology are fixed.

    This is the only case-study workflow in the codebase, and it settles audit finding C5: the
    competing planner-plus-two-workers topology that produced the manuscript's 2.4e3 J has been
    removed. For Llama-3 8B on H100 at eta = 0.05 this one gives E_W = 1527 J, agentic-eCAL
    8.8e-2 J/bit and crossover G* = 2.1e9 -- quote those everywhere.
    """
    ret = Retrieval(k_chunks=5, chunk_tokens=256)
    tool = Tool(name="code", energy=2.0, obs_tokens=150)
    steps = [
        Step(agent="planner",   p_in_local=300, p_out=200),
        Step(agent="retriever", p_in_local=120, p_out=120, retrieval=ret),
        Step(agent="analyst",   p_in_local=200, p_out=300, tool=tool),
        Step(agent="writer",    p_in_local=200, p_out=400),
    ] * max(int(rounds), 1)
    # round_size stays 1: the four agents run in sequence, so each reads its predecessors'
    # output within the same pass. Grouping them into one parallel round would remove that.
    return Workflow(model=model or LLMS["llama3_8b"], hw=hw or HW["a100"], steps=steps,
                    carry_history=True, gamma_v=0.10, tx_energy_per_message=0.5)


def tree_workflow(model: LLM, hw: Hardware, fanout: int, depth: int,
                  p_out: int = 250, sys_tokens: int = 400, p_in_root: int = 300,
                  batch: float = 1.0, aggregate: bool = True) -> Workflow:
    """A fan-out tree: a root delegates to ``fanout`` children, each of which may delegate again.

    Every node reads only its parent's output, so a node's prompt is bounded by one hand-off
    however wide or deep the tree becomes. That is the structural difference from a chain, where
    each step reads the whole accumulated transcript. With ``aggregate`` the root reads its
    children's outputs back on the way up, which is the only place a prompt grows with fan-out.
    """
    steps: List[Step] = [Step(agent="root", p_in_local=p_in_root, p_out=p_out)]
    nodes = 1
    for level in range(1, depth + 1):
        nodes = fanout ** level
        steps += [Step(agent=f"L{level}_{i}", p_in_local=p_out, p_out=p_out)
                  for i in range(nodes)]
    if aggregate:
        steps.append(Step(agent="root", p_in_local=nodes * p_out, p_out=p_out))
    return Workflow(model=model, hw=hw, steps=steps, sys_tokens=sys_tokens,
                    carry_history=False, gamma_v=0.10, tx_energy_per_message=0.5,
                    serving_batch=batch)


def debate_workflow(model: LLM, hw: Hardware, n_agents: int, rounds: int, p_out: float,
                    sys_tokens: int, history_rounds: Optional[int] = 1,
                    exclude_self: bool = True) -> Workflow:
    """The measured multi-agent protocol: ``n_agents`` agents debating for ``rounds`` rounds.

    Each round is one pass over the team, and by default an agent entering round r reads the
    outputs its n-1 peers produced in round r-1. Setting ``history_rounds=None`` retains the whole
    transcript instead and ``carry_history=False`` retains none; the token-validation figure plots
    all three against measurement.

    ``p_out`` is the generated tokens per call and ``sys_tokens`` the fixed prompt scaffold, both
    taken from the measurements rather than assumed.
    """
    steps = [Step(agent=f"agent{i}", p_in_local=0, p_out=int(round(p_out)))
             for _ in range(rounds) for i in range(n_agents)]
    return Workflow(model=model, hw=hw, steps=steps, sys_tokens=sys_tokens,
                    carry_history=True, history_rounds=history_rounds,
                    round_size=n_agents, exclude_self=exclude_self,
                    gamma_v=0.10, tx_energy_per_message=0.0)


# ---------------------------------------------------------------------------
# Calibration checks against the measured anchors in the literature
# ---------------------------------------------------------------------------

def validate_two_rate() -> dict:
    """Derived rates against the coefficients a two-rate fit obtains from the A100 sweeps.

    Nothing here is fitted: P is the board power, Pi and B_HBM are datasheet, eta_pre is the
    near-peak prefill utilisation. Measured values are c_pre = 0.023 (Qwen) / 0.028 (Llama)
    J per prompt token and a_dec = 3.2 / 3.3 J, from the vLLM sweeps on an A100.
    """
    out = {}
    for key, c_pre_meas, a_dec_meas in (("qwen2_5_7b", 0.023, 3.2), ("llama3_8b", 0.028, 3.3)):
        m, hw = LLMS[key], HW["a100"]
        a_dec = hw.power * m.n_active * m.bytes_per_param / (hw.bandwidth * hw.bw_efficiency)
        out[m.name] = {
            "c_pre": (round(prefill_rate(m, hw), 4), c_pre_meas),
            "a_dec": (round(a_dec, 2), a_dec_meas),
        }
    return out


def implied_batch(model: LLM, hw: Hardware, j_per_token: float,
                  p_in: float = 64, p_out: float = 512) -> float:
    """Serving batch at which the two-rate model reproduces a measured J/output-token anchor.

    A batch-aware call model should not be tuned to hit a literature number; it should say what
    operating point that number corresponds to. Solving Eq. (3) for b is the honest use of it.
    """
    cbar = p_in + p_out / 2.0
    fixed = prefill_rate(model, hw) * p_in / p_out \
        + hw.power * model.kv_bytes_per_token * cbar / (hw.bandwidth * hw.bw_efficiency)
    weight_term = hw.power * model.n_active * model.bytes_per_param \
        / (hw.bandwidth * hw.bw_efficiency)
    residual = j_per_token - fixed
    return weight_term / residual if residual > 0 else float("inf")


def validate_against_samsi() -> dict:
    """Samsi et al. (2023) measured 3-4 J/token for LLaMA-65B on A100s.

    Under Eq. (5) the b = 1 limit is P N beta / B_HBM = 25.6 J/token, so that anchor cannot be a
    single-stream measurement; it implies a serving batch of roughly 7-9. Tensor parallelism does
    not change this, since sharding over n devices divides the bytes each reads by n while
    multiplying board power by n, leaving P W / B invariant.
    """
    m, hw = LLMS["llama_65b"], HW["a100"]
    return {"J/token at b=1": round(joules_per_token(m, hw, 64, 512, 1.0), 1),
            "implied batch for 3 J/token": round(implied_batch(m, hw, 3.0), 1),
            "implied batch for 4 J/token": round(implied_batch(m, hw, 4.0), 1)}


def validate_current_gen() -> dict:
    """2026 anchors: Llama-3 70B FP8 on H100 at ~0.39 J/token (TokenPowerBench)."""
    hw = HW["h100"]
    return {"llama3_70b implied batch at 0.39 J/token":
            round(implied_batch(LLMS["llama3_70b"], hw, 0.39), 1),
            "llama3_8b implied batch at 0.11 J/token":
            round(implied_batch(LLMS["llama3_8b"], hw, 0.11), 1)}


if __name__ == "__main__":
    print("[two-rate] derived vs measured coefficients (A100, nothing fitted):")
    for name, v in validate_two_rate().items():
        print(f"  {name:<13} c_pre {v['c_pre'][0]:.4f} vs {v['c_pre'][1]}   "
              f"a_dec {v['a_dec'][0]} vs {v['a_dec'][1]}")
    print("\n[anchors] what serving batch each literature figure implies:")
    for k, v in {**validate_against_samsi(), **validate_current_gen()}.items():
        print(f"  {k}: {v}")

    m, hw = LLMS["llama3_8b"], HW["h100"]
    wf = Workflow(model=m, hw=hw, steps=[Step(p_in_local=600, p_out=300)],
                  sys_tokens=0, carry_history=True, gamma_v=0.0, serving_batch=64)
    direct = llm_call_energy(m, hw, 600, 300, 64)
    one_shot = wf.run()["e_total"]
    print(f"\n[reduce] one-shot workflow {one_shot:.2f} J == direct call {direct:.2f} J: "
          f"{abs(one_shot - direct) < 1e-6}")
    print(f"  agentic-eCAL (one-shot, b=64): {wf.agentic_ecal():.2e} J/bit")
