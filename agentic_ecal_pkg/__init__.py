"""agentic-eCAL: analytical two-rate energy model for agentic workflows of open-weight LLMs.

This package is a modular, behaviour-preserving refactor of the single-file ``agentic_ecal.py``.
It extends the eCAL methodology (Chou et al., IEEE JSAC 2026) from a single AI-model lifecycle to
an *agentic workflow*: a directed graph of LLM calls, tool invocations, retrievals, and
inter-agent messages built on a pre-trained open-weight model.

It contains the model only -- no plotting, no data loading, no pandas/matplotlib import -- so it
can be imported and tested on its own. Figures live in ``figure_scripts/``.

The call model is TWO-RATE: prefill is compute-bound and priced at ``eta_pre``, decode is
memory-bandwidth-bound and amortises the weight read over the serving batch. All energies are in
Joules unless noted.

The top-level namespace re-exports the full public API, so ``import agentic_ecal_pkg as ae`` is a
drop-in for the original ``import agentic_ecal as ae``.
"""

from __future__ import annotations

from .constants import BITS_PER_TOKEN, GWH_TO_J, MFU_LATENCY, MFU_THROUGHPUT
from .descriptors import Hardware, LLM
from .calib import TwoRateCalib
from .rates import (
    call_latency,
    decode_rate,
    embedding_flops,
    joules_per_token,
    llm_call_energy,
    prefill_rate,
)
from .components import Retrieval, Step, Tool
from .workflow import Workflow
from .registries import HW, LLMS, TWO_RATE
from .builders import debate_workflow, four_agent_rag_workflow, tree_workflow
from .validation import validate_against_samsi, validate_current_gen, validate_two_rate

__all__ = [
    "Hardware", "LLM", "Tool", "Retrieval", "Step", "Workflow",
    "HW", "LLMS", "MFU_LATENCY", "MFU_THROUGHPUT", "BITS_PER_TOKEN", "GWH_TO_J",
    "prefill_rate", "decode_rate", "llm_call_energy", "call_latency",
    "joules_per_token",
    "embedding_flops", "validate_two_rate", "TwoRateCalib", "TWO_RATE",
    "four_agent_rag_workflow", "tree_workflow", "debate_workflow",
    "validate_against_samsi", "validate_current_gen",
]
