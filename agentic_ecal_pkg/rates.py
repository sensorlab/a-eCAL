"""The two-rate call model (paper Eqs. 3-5): per-token energy rates, call energy, and latency.

A call has two phases with different energy characteristics, so one rate cannot describe both.

  E_call(p_in, p_out; b) = c_pre * p_in + c_dec(b) * p_out            (the two-rate model)

PREFILL processes the whole prompt in one parallel pass. It is compute-bound at near-peak
utilisation and essentially batch-independent:

  c_pre = 2 N P / (eta_pre * Pi)                                         (the prefill rate)

DECODE emits one token at a time and is memory-bandwidth-bound. Each step streams the weight
bytes from HBM once for the whole batch, so that cost amortises over the b concurrently
decoding sequences, while every sequence re-reads its own cache and that term does not:

  c_dec(b) = P / (eps B_HBM) * ( N beta / b  +  gamma * cbar )            (the decode rate)

eps is the fraction of peak HBM bandwidth a serving stack actually sustains, reported as
model-bandwidth utilisation, (achieved bandwidth) / (peak bandwidth). See ``Hardware.bw_efficiency``
for the measured values and the batch-degradation caveats.
"""

from __future__ import annotations

from .descriptors import Hardware, LLM


def embedding_flops(n_params: float, n_tokens: float) -> float:
    """Forward FLOPs of a small encoder, used for the retrieval term.

    Args:
        n_params (float): Encoder parameter count ``N``.
        n_tokens (float): Tokens embedded ``p_q``.

    Returns:
        float: ``2 * N * p_q``, the forward FLOPs.
    """
    return 2.0 * n_params * n_tokens


def prefill_rate(model: LLM, hw: Hardware) -> float:
    """Prefill energy rate ``c_pre``: compute-bound and batch-independent.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor.

    Returns:
        float: Joules per prompt token, ``2 N P / (eta_pre * Pi)``.
    """
    return 2.0 * model.n_active * hw.power / (hw.eta_pre * hw.flops_peak)


def decode_rate(model: LLM, hw: Hardware, batch: float = 1.0, context: float = 0.0) -> float:
    """Decode energy rate ``c_dec(b)``: memory-bound, amortising over the batch.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor (must have ``bandwidth`` set).
        batch (float): Serving batch size ``b`` (clamped to a minimum of 1). Defaults to ``1.0``.
        context (float): Mean context length the KV term is evaluated at. Defaults to ``0.0``.

    Returns:
        float: Joules per generated token.

    Raises:
        ValueError: If ``hw.bandwidth`` is not set (``<= 0``).
    """
    if hw.bandwidth <= 0.0:
        raise ValueError(f"{hw.name} has no HBM bandwidth set")
    weights = model.n_active * model.bytes_per_param / max(float(batch), 1.0)
    cache = model.kv_bytes_per_token * context
    return hw.power * (weights + cache) / (hw.bandwidth * hw.bw_efficiency)


def call_latency(model: LLM, hw: Hardware, p_in: float, p_out: float,
                 batch: float = 1.0) -> float:
    """Wall-clock time of one call, which is not energy divided by power.

    Energy per request divides by the serving batch, because the weight read of a decode step is
    shared by every sequence in flight. Latency does not: the step occupies its full duration for
    all of them. Prefill is one compute-bound pass; each decode step costs the weight read plus
    the cache read of the whole batch.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor.
        p_in (float): Prompt (prefill) tokens.
        p_out (float): Generated (decode) tokens.
        batch (float): Serving batch size ``b`` (clamped to a minimum of 1). Defaults to ``1.0``.

    Returns:
        float: Wall-clock time of the call, in seconds.
    """
    b = max(float(batch), 1.0)
    cbar = p_in + p_out / 2.0
    t_pre = 2.0 * model.n_active * p_in / (hw.eta_pre * hw.flops_peak)
    step = (model.n_active * model.bytes_per_param
            + b * model.kv_bytes_per_token * cbar) / (hw.bandwidth * hw.bw_efficiency)
    return t_pre + p_out * step


def llm_call_energy(model: LLM, hw: Hardware, p_in: float, p_out: float,
                    batch: float = 1.0) -> float:
    """Energy of one LLM call under the two-rate model, from datasheet quantities.

    The rates come from :func:`prefill_rate` / :func:`decode_rate` rather than from the measured
    ``TWO_RATE`` coefficients, so this is a derivation rather than a calibration. The figures use
    the measured path instead; see ``two_rate_energy_per_query`` in ``make_figures.py``.

    The cache term is evaluated at the call's mean context, ``p_in + p_out/2``, since the
    transcript grows by one token per decode step.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor.
        p_in (float): Prompt (prefill) tokens.
        p_out (float): Generated (decode) tokens.
        batch (float): Serving batch size ``b``. Defaults to ``1.0``.

    Returns:
        float: Energy of the call, in Joules.
    """
    cbar = p_in + p_out / 2.0
    return prefill_rate(model, hw) * p_in + decode_rate(model, hw, batch, cbar) * p_out


def joules_per_token(model: LLM, hw: Hardware, p_in: float = 512, p_out: float = 512,
                     batch: float = 1.0) -> float:
    """Energy per generated token for a representative call.

    Args:
        model (LLM): The model descriptor.
        hw (Hardware): The accelerator descriptor.
        p_in (float): Prompt (prefill) tokens. Defaults to ``512``.
        p_out (float): Generated (decode) tokens. Defaults to ``512``.
        batch (float): Serving batch size ``b``. Defaults to ``1.0``.

    Returns:
        float: ``llm_call_energy(...) / p_out``, in Joules per generated token.
    """
    return llm_call_energy(model, hw, p_in, p_out, batch) / p_out
