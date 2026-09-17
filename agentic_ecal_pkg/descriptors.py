"""Hardware and model descriptors -- the leaf dataclasses the rest of the package builds on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware / model descriptors
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    """Accelerator descriptor for the two-rate energy model.

    ``mfu`` (model-FLOP utilization, achieved/peak) is the calibration knob. Single-stream,
    latency-bound decode is memory-bound and reaches only ~5% MFU; throughput-optimized batched
    serving reaches ~30-50%. We calibrate it against measured per-token energy (research-notes.md)
    rather than asserting peak FLOP/s, because estimating from nameplate TDP overestimates measured
    energy by up to 4.1x (ML.ENERGY, arXiv:2505.06371).

    Attributes:
        name (str): Human-readable accelerator name.
        flops_peak (float): Peak dense FLOP/s at the working precision (fp16/bf16).
        power (float): Board power drawn while busy, in Watts.
        mfu (float): Model-FLOP utilization (achieved/peak); prices the retrieval term. Defaults
            to ``0.10`` (mid latency/throughput regime).
        bandwidth (float): Peak HBM bandwidth ``B_HBM``, in byte/s. Required by the decode rate.
        eta_pre (float): Prefill utilization; compute-bound, so near peak. Back-solving the fitted
            ``c_pre`` of ``TWO_RATE`` gives 0.74 (Llama-3.1-8B) and 0.85 (Qwen2.5-7B) on the A100
            sweeps. Defaults to ``0.85``.
        bw_efficiency (float): ``eps``, the sustained fraction of peak HBM bandwidth (model-
            bandwidth utilization). Measured at ~55-72% at batch 1; degrades with batch. Defaults
            to ``0.70``.
    """
    name: str
    flops_peak: float          # peak dense FLOP/s at the working precision
    power: float               # board power drawn while busy [W]
    mfu: float = 0.10          # default: mid latency/throughput regime
    bandwidth: float = 0.0     # B_HBM, peak HBM bandwidth [byte/s]
    eta_pre: float = 0.85      # prefill utilisation; compute-bound, so near peak. Back-solving
                               # the fitted c_pre of TWO_RATE gives 0.74 (Llama-3.1-8B) and
                               # 0.85 (Qwen2.5-7B) on the A100 sweeps.
    bw_efficiency: float = 0.70   # eps, sustained fraction of peak HBM bandwidth. Reported MBU
                                  # is 72% (pytorch.org/blog/accelerating-generative-ai-2) and
                                  # 55-60% (databricks.com/blog/llm-inference-performance-
                                  # engineering-best-practices), both at batch 1. See the
                                  # two-rate comment block for why it degrades with batch.

    @property
    def flops_eff(self) -> float:
        """Effective (achieved) FLOP/s.

        Returns:
            float: ``flops_peak * mfu``, the utilisation-derated throughput.
        """
        return self.flops_peak * self.mfu


@dataclass
class LLM:
    """Open-weight decoder-only LLM descriptor.

    Attributes:
        name (str): Human-readable model name.
        n_params (float): Total parameter count.
        n_layers (int): Number of transformer blocks.
        d_model (int): Hidden size.
        e_pretrain (float): Embodied (pre-training) energy in Joules; ``0.0`` if unknown.
        n_active (Optional[float]): Active (non-embedding) parameters per token. Defaults to
            ``n_params`` for dense models; for Mixture-of-Experts, the routed/active subset.
        d_kv (Optional[int]): KV width ``n_kv_heads * head_dim``. Defaults to ``d_model``
            (multi-head attention).
        bytes_per_param (float): ``beta``, the element size of weights and cache in bytes
            (bf16 = 2). Defaults to ``2.0``.
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
        """KV-cache traffic coefficient ``gamma``.

        Returns:
            float: KV-cache bytes read per decoded token, per token of context
            (``2 * n_layers * d_kv * bytes_per_param``).
        """
        return 2.0 * self.n_layers * self.d_kv * self.bytes_per_param
