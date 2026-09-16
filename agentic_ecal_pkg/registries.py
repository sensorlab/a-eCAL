"""Representative hardware, model, and measured-calibration presets.

Kept as Python literals (not externalised YAML) so the numbers are bit-exact and the package
stays dependency-free. Specs/architectures are from vendor datasheets and official model cards;
see ``docs/research-notes.md`` for the source of each value.
"""

from __future__ import annotations

from .constants import GWH_TO_J, MFU_LATENCY
from .descriptors import Hardware, LLM
from .calib import TwoRateCalib


# Representative accelerators.
# H100 SXM bf16 dense ~989 TFLOP/s (with sparsity ~1979); board power ~700 W.
# A100 80GB bf16 dense ~312 TFLOP/s; board power ~400 W (SXM).
HW = {
    "h100": Hardware("NVIDIA H100 SXM", flops_peak=989e12, power=700.0, mfu=MFU_LATENCY,
                     bandwidth=3.35e12),
    # H100 NVL, the board TokenPowerBench actually ran on. Their released results record
    # gpu_memory_total_mb = 95830 (94 GB, against the SXM's 80) and a per-GPU draw that caps at
    # 394.5 W over all 185 runs, so the 700 W / 3.35 TB/s SXM entry above misprices every number
    # taken from that source by ~2x. NVL: 94 GB HBM3, 3.9 TB/s, 400 W board cap.
    "h100_nvl": Hardware("NVIDIA H100 NVL 94GB", flops_peak=989e12, power=400.0,
                         mfu=MFU_LATENCY, bandwidth=3.9e12),
    "a100": Hardware(  "NVIDIA A100 80GB", flops_peak=312e12, power=400.0, mfu=MFU_LATENCY,
                     bandwidth=2.039e12),
}


# Representative open-weight models. Architectures from official model cards; pre-training
# (embodied) energy = GPU-hours x 700 W from the Llama-3 model card (research-notes.md):
#   8B  : 1.3M H100-h x 700 W = 0.91 GWh = 3.28e12 J
#   70B : 6.4M H100-h x 700 W = 4.48 GWh = 1.61e13 J
LLMS = {
    # Llama-3 8B / Llama-3.1-8B-Instruct: 32 layers, d_model 4096. Also the measured model.
    # GQA: 8 KV heads x head_dim 128 -> d_kv = 1024.
    "llama3_8b": LLM("Llama-3 8B", n_params=8.03e9, n_layers=32, d_model=4096,
                     e_pretrain=0.91 * GWH_TO_J, d_kv=1024),
    # Llama-3 70B: 80 layers, d_model 8192.
    "llama3_70b": LLM("Llama-3 70B", n_params=70.6e9, n_layers=80, d_model=8192,
                      e_pretrain=4.48 * GWH_TO_J, d_kv=1024),
    # LLaMA-65B: 80 layers, d_model 8192. Retained only to back validate_against_samsi(); it was
    # dropped from Table II and from the amortization figure. No explicit d_kv, so d_kv = d_model
    # = 8192 -- multi-head attention, correct for LLaMA-1 and 8x the 70B's GQA width.
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


# Both models keep a decode floor at large batch, where the weight read has amortised away and
# the per-token KV traffic is what is left. Llama's wider KV puts its floor ~3x above Qwen's.
TWO_RATE = {
    "qwen2_5_7b": TwoRateCalib("Qwen2.5-7B", c_pre=0.023, a_dec=3.2, kv_floor=0.02),
    "llama3_8b": TwoRateCalib("Llama-3.1-8B", c_pre=0.028, a_dec=3.3, kv_floor=0.06),
}
