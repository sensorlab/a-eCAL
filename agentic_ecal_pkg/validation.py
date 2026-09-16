"""Calibration checks against the measured anchors in the literature.

``implied_batch`` is a package-internal helper (not part of the public API) used by the two
current-generation validators; it is the honest inverse of the two-rate model -- given a measured
J/output-token anchor, it reports the serving batch that anchor implies.
"""

from __future__ import annotations

from .descriptors import Hardware, LLM
from .registries import HW, LLMS
from .rates import prefill_rate, joules_per_token


def validate_two_rate() -> dict:
    """Derived rates against the coefficients a two-rate fit obtains from the A100 sweeps.

    Nothing here is fitted: P is the board power, Pi and B_HBM are datasheet, eta_pre is the
    near-peak prefill utilisation. Measured values are c_pre = 0.023 (Qwen) / 0.028 (Llama)
    J per prompt token and a_dec = 3.2 / 3.3 J, from the vLLM sweeps on an A100.

    Returns:
        dict: Mapping model name to ``{"c_pre": (derived, measured), "a_dec": (derived,
        measured)}`` tuples.
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
    operating point that number corresponds to. Solving the two-rate model for b is the honest use of it.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor.
        j_per_token (float): The measured energy anchor, in Joules per output token.
        p_in (float): Prompt tokens assumed for the anchor. Defaults to ``64``.
        p_out (float): Output tokens assumed for the anchor. Defaults to ``512``.

    Returns:
        float: The serving batch that reproduces the anchor, or ``inf`` when the residual is
        non-positive (the anchor is at or below the ``b -> inf`` floor).
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

    Retained as a diagnostic only: the Table II row this used to back was DROPPED, because a 2023
    sweep billing 8-32 shards at 300 W-1 kW aggregate sat far below the bandwidth roofline the
    decode rate assumes. See table2-row1-options.md.

    Under the decode rate the b = 1 limit is P N beta / B_HBM = 25.6 J/token, so that anchor cannot be a
    single-stream measurement; it implies a serving batch of roughly 7-9. Tensor parallelism does
    not change this, since sharding over n devices divides the bytes each reads by n while
    multiplying board power by n, leaving P W / B invariant.

    Returns:
        dict: The b=1 J/token figure and the serving batches implied by the 3 and 4 J/token
        anchors.
    """
    m, hw = LLMS["llama_65b"], HW["a100"]
    return {"J/token at b=1": round(joules_per_token(m, hw, 64, 512, 1.0), 1),
            "implied batch for 3 J/token": round(implied_batch(m, hw, 3.0), 1),
            "implied batch for 4 J/token": round(implied_batch(m, hw, 4.0), 1)}


def validate_current_gen() -> dict:
    """2026 anchors from TokenPowerBench, on the board it actually ran (H100 NVL).

    The 0.70 J/token used here for the 70B and the 0.39 it replaced are BOTH figure reads of that
    paper, from different panels; 0.39 is not fabricated, contrary to an earlier note in
    ``table2-row2-options.md``. 0.70 is corroborated by the paper's own text: it reports energy
    per token rising 7.3x from Llama3-1B to 70B, and its Fig. 4 puts the 1B near 0.10 J/token.
    Which panel 0.39 came from is not yet settled -- the engine comparison is the likely source,
    since the same text reports TensorRT-LLM and vLLM cutting energy per token 25-40% below
    Transformers, and 0.70 x 0.6 = 0.42.

    The 8B anchor below is not a figure read at all: it is the released per-run data, 0.084
    J/token on the active device at batch 32.

    Returns:
        dict: The serving batches implied by the 70B (0.70 J/token) and 8B (0.084 J/token)
        anchors on the H100 NVL.
    """
    hw = HW["h100_nvl"]
    return {"llama3_70b implied batch at 0.70 J/token (Fig. 3b)":
            round(implied_batch(LLMS["llama3_70b"], hw, 0.70), 1),
            "llama3_8b implied batch at 0.084 J/token (released runs, active GPU)":
            round(implied_batch(LLMS["llama3_8b"], hw, 0.084, p_in=50, p_out=382), 1)}
