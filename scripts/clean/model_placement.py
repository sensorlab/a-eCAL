#!/usr/bin/env python3
"""Model placement: model dimension against edge hardware capability.

Companion to ``agentic_ecal.py`` and ``placement.py``. Those two price a workflow once its
agents have been *assigned* to sites. This module asks the prior question: which sites can host
the model at all, and how many agents fit once they start talking to each other.

The binding constraint at the edge is not FLOP/s but MEMORY. A device can serve a model only if

    weights + concurrent_agents * KV(context)  <=  device memory

and the measured debate data supplies the context term. Communication enters the placement
problem not as transmitted bits but as KV-cache pressure: a peer's message must sit in the
receiver's context window for the whole of the receiver's forward pass, so full-mesh debate
inflates every agent's resident footprint linearly in the number of peers.

MEASURED BASIS (~/DATA/data_steiner_Aug26, 6.69M team-item records, 16 open-weight models,
team sizes 1..30, full mesh, 3 debate rounds, peer_token_budget=120):

    per-agent prefill tokens at round r>=2:   p_in(k) = a + b*k        R^2 >= 0.998, all 16 models
    b (tokens imported per peer) = 79..126,   median 103

so context grows LINEARLY in team size while the team's total prefill grows QUADRATICALLY.

CITATION STATUS: model geometry is transcribed from the published model cards and has not been
re-read from the checkpoints; ``qwen3.6-27b`` in particular is unverified. Device memory and
bandwidth are vendor datasheet figures; peak FLOP/s are dense (non-sparse) figures and are used
only for the prefill-time column, not for any feasibility claim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import agentic_ecal as ae

__all__ = ["Device", "DEVICES", "FLEET", "PeerLaw", "PEER_LAW",
           "weight_bytes", "kv_bytes_per_token", "resident_bytes",
           "max_agents", "fits", "context_tokens"]

GB = 1024.0 ** 3


# ---------------------------------------------------------------------------
# Hardware: the far edge to the core. ``memory`` is the load-bearing field.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Device:
    name: str
    tier: str
    memory: float          # bytes usable for weights + cache
    short: str             # compact label for figures/tables
    bandwidth: float       # byte/s, peak
    flops_peak: float      # FLOP/s, dense at the working precision
    power: float           # W, board/module power while busy
    util: float = 0.90     # fraction of nameplate memory a server may actually claim

    @property
    def usable(self) -> float:
        return self.memory * self.util

    @property
    def label(self) -> str:
        return self.short or self.name


# vLLM in the measured runs used gpu_memory_utilization=0.85; we keep 0.90 as the nominal
# headroom and note that the runs were slightly more conservative.
DEVICES: Dict[str, Device] = {
    "orin_nano":  Device("Jetson Orin Nano 8GB",  "far edge",  8 * GB,  "Orin\n8 GB",
                         102e9,  17e12,  25.0),
    "agx_orin":   Device("Jetson AGX Orin 64GB",  "edge",      64 * GB, "AGX\n64 GB",
                         204.8e9, 85e12,  60.0),
    "l4":         Device("NVIDIA L4 24GB",        "edge site", 24 * GB, "L4\n24 GB",
                         300e9,  121e12,  72.0),
    "rtx4090":    Device("RTX 4090 24GB",         "regional",  24 * GB, "RTX 4090\n24 GB",
                         1008e9, 165e12, 450.0),
    "a100":       Device("A100 80GB",             "core DC",   80 * GB, "A100\n80 GB",
                         2039e9, 312e12, 400.0),
    "h100":       Device("H100 SXM 80GB",         "core DC",   80 * GB, "H100\n80 GB",
                         3350e9, 989e12, 700.0),
}


# ---------------------------------------------------------------------------
# The measured fleet. d_kv = n_kv_heads * head_dim, which is what the cache actually stores.
# ``kv_bytes`` is the CACHE element size and is 2 (fp16) even for 4-bit weights: AWQ quantises
# the weights, not the KV cache.
# ---------------------------------------------------------------------------
def _m(name, n_params, n_layers, d_model, d_kv, bpp=2.0, n_active=None):
    return ae.LLM(name, n_params=n_params, n_layers=n_layers, d_model=d_model,
                  d_kv=d_kv, bytes_per_param=bpp, n_active=n_active)


FLEET: Dict[str, ae.LLM] = {
    "smollm3-3b":          _m("SmolLM3-3B",         3.08e9, 36, 2048,  512),
    "qwen2.5-3b":          _m("Qwen2.5-3B",         3.09e9, 36, 2048,  256),
    "qwen3-4b":            _m("Qwen3-4B",           4.02e9, 36, 2560, 1024),
    "mistral-7b":          _m("Mistral-7B-v0.3",    7.25e9, 32, 4096, 1024),
    "olmo2-7b":            _m("OLMo-2-7B",          7.30e9, 32, 4096, 4096),   # MHA, no GQA
    "qwen2.5-7b":          _m("Qwen2.5-7B",        7.615e9, 28, 3584,  512),
    "llama3.1-8b":         _m("Llama-3.1-8B",       8.03e9, 32, 4096, 1024),
    "marin-8b":            _m("Marin-8B",           8.00e9, 32, 4096, 1024),
    "qwen3-8b":            _m("Qwen3-8B",           8.19e9, 36, 4096, 1024),
    "phi4-14b":            _m("Phi-4 14B",          14.7e9, 40, 5120, 1280),
    "qwen3-14b":           _m("Qwen3-14B",          14.8e9, 40, 5120, 1024),
    "r1-distill-qwen-14b": _m("R1-Distill-Qwen-14B",14.8e9, 48, 5120, 1024),
    "gpt-oss-20b":         _m("GPT-OSS-20B (MoE)",  20.9e9, 24, 2880,  512, bpp=0.6, n_active=3.6e9),
    "qwen3.6-27b":         _m("Qwen3.6-27B",        27.0e9, 48, 5120, 1024),   # UNVERIFIED
    "llama3.3-70b-awq":    _m("Llama-3.3-70B AWQ",  70.6e9, 80, 8192, 1024, bpp=0.5),
    "qwen2.5-72b-awq":     _m("Qwen2.5-72B AWQ",    72.7e9, 80, 8192, 1024, bpp=0.5),
}

KV_ELEM_BYTES = 2.0   # fp16 cache, independent of weight quantisation


# ---------------------------------------------------------------------------
# The measured communication law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeerLaw:
    """Per-agent prefill tokens as a function of peer count, fitted to the measured runs.

    ``solo`` is the round-1 prompt (no peer has spoken yet) and equals the single-agent
    ensemble prompt to within a token. ``a`` is the round-2 intercept -- the solo prompt plus
    the framing the debate template adds -- and ``b`` is the marginal cost of one peer.
    """
    solo: float
    a: float
    b: float
    p_out: float

    def p_in(self, k: float, communicating: bool = True) -> float:
        return self.a + self.b * k if communicating else self.solo


# Fitted on ~/DATA/data_steiner_Aug26; see scripts/clean/fit_peer_law.py.
PEER_LAW: Dict[str, PeerLaw] = {
    "gpt-oss-20b":          PeerLaw(192, 287,  78.6, 172),
    "marin-8b":             PeerLaw(541, 660,  90.5, 118),
    "qwen2.5-72b-awq":      PeerLaw(150, 296,  92.8,  86),
    "qwen2.5-7b":           PeerLaw(163, 281,  93.9, 108),
    "llama3.3-70b-awq":     PeerLaw(146, 266,  95.6, 143),
    "qwen3-4b":             PeerLaw(142, 259,  99.2, 126),
    "phi4-14b":             PeerLaw(135, 259, 101.5, 113),
    "qwen3-14b":            PeerLaw(145, 264, 101.9, 112),
    "qwen3-8b":             PeerLaw(145, 265, 103.3, 156),
    "smollm3-3b":           PeerLaw(198, 326, 106.5, 342),
    "llama3.1-8b":          PeerLaw(164, 294, 108.4, 146),
    "qwen2.5-3b":           PeerLaw(163, 313, 111.0, 188),
    "qwen3.6-27b":          PeerLaw(138, 279, 113.9, 311),
    "olmo2-7b":             PeerLaw(142, 287, 116.5, 191),
    "mistral-7b":           PeerLaw(159, 300, 119.2, 176),
    "r1-distill-qwen-14b":  PeerLaw(139, 278, 125.9, 567),
}


def context_tokens(key: str, team_size: int, communicating: bool = True) -> float:
    """Resident context per agent [tokens]: prompt plus its own generated answer."""
    law = PEER_LAW[key]
    return law.p_in(team_size - 1, communicating) + law.p_out


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------
def weight_bytes(model: ae.LLM) -> float:
    return model.n_params * model.bytes_per_param


def kv_bytes_per_token(model: ae.LLM) -> float:
    """2 (K and V) * layers * d_kv * cache element size."""
    return 2.0 * model.n_layers * model.d_kv * KV_ELEM_BYTES


def resident_bytes(model: ae.LLM, context: float, concurrent: int) -> float:
    return weight_bytes(model) + concurrent * context * kv_bytes_per_token(model)


def fits(model: ae.LLM, dev: Device, context: float = 0.0, concurrent: int = 0) -> bool:
    return resident_bytes(model, context, concurrent) <= dev.usable


def max_agents(model: ae.LLM, dev: Device, context: float) -> int:
    """How many concurrent agent contexts the device can hold alongside the weights."""
    free = dev.usable - weight_bytes(model)
    if free <= 0:
        return 0
    return int(free // (context * kv_bytes_per_token(model)))
