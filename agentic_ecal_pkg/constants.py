"""Grounded numeric constants for the agentic-eCAL model.

Every value here is literature-grounded; see ``docs/research-notes.md`` for the source of each.
All energies are in Joules unless noted.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Token -> bit conversion
# ---------------------------------------------------------------------------
#
# f is the information-theoretic capacity log2(|V|) of a token drawn from a vocabulary of size
# |V|: ~17 bits for the 128k-152k vocabularies of Llama-3 / Qwen-2.5. It is a pure denominator
# scale, so it cancels from every scaling exponent and sets only the absolute J/bit level.

BITS_PER_TOKEN = 17.0

# MFU_LATENCY / MFU_THROUGHPUT are residues of the retired single-rate model. Nothing in the
# two-rate path reads them; they survive only through Hardware.mfu, which prices the retrieval
# term. The throughput value's old comment claimed 0.25 "reproduces ~0.39 J/token Llama-3 70B",
# which was a calibration claim about the model that no longer exists (see validate_current_gen).
MFU_LATENCY = 0.05      # single-stream / latency-bound decode
MFU_THROUGHPUT = 0.25   # throughput-optimized batched serving

GWH_TO_J = 3.6e12  # 1 GWh = 3.6e12 J
