"""Workflow components: tools, retrieval, and a single step executed by an agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .descriptors import Hardware
from .rates import embedding_flops


@dataclass
class Tool:
    """A non-LLM tool invocation (code exec, API, search).

    Attributes:
        name (str): Human-readable tool name.
        energy (float): Energy per call, in Joules. Defaults to ``0.0``.
        obs_tokens (int): Tokens the tool result adds to the running context. Defaults to ``0``.
    """
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

    Attributes:
        embed_params (float): Bi-encoder parameters (BERT-base class, approx. 109M). Defaults to
            ``1.1e8``.
        query_tokens (int): Tokens in the query to embed. Defaults to ``64``.
        n_vectors (float): Index size; immaterial to the result. Defaults to ``1e6``.
        dim (int): Bi-encoder embedding width (not the generator hidden size). Defaults to ``768``.
        k_chunks (int): Number of retrieved chunks appended to the prompt. Defaults to ``5``.
        chunk_tokens (int): Tokens per chunk, in the generator's tokenizer. Defaults to ``256``.
        ann (bool): Approximate (sublinear) search if ``True``, else exhaustive. Both are
            dominated by the embedding term. Defaults to ``True``.
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
        """Energy of the retrieval step (query embedding + index search).

        Args:
            hw (Hardware): The accelerator the encoder is priced on.

        Returns:
            float: Retrieval energy, in Joules (dominated by the embedding term).
        """
        embed_flops = embedding_flops(self.embed_params, self.query_tokens)
        if self.ann:
            # HNSW-style: ~ dim * log2(n_vectors) distance ops, *2 FLOPs each
            search_flops = 2.0 * self.dim * math.log2(max(self.n_vectors, 2.0))
        else:
            search_flops = 2.0 * self.dim * self.n_vectors
        return (embed_flops + search_flops) / hw.flops_eff * hw.power

    @property
    def added_context(self) -> int:
        """Tokens injected into the prompt by this retrieval.

        Returns:
            int: ``k_chunks * chunk_tokens``.
        """
        return self.k_chunks * self.chunk_tokens


@dataclass
class Step:
    """One workflow step executed by an agent: an LLM call plus optional tool/retrieval.

    Attributes:
        agent (str): Name of the agent executing this step. Defaults to ``"main"``.
        p_in_local (int): *New* prompt tokens contributed at this step, excluding carried
            history. Defaults to ``0``.
        p_out (int): Generated (decode) tokens. Defaults to ``256``.
        tool (Optional[Tool]): A tool invoked after the call, if any. Defaults to ``None``.
        retrieval (Optional[Retrieval]): A retrieval performed before the call, if any. Defaults
            to ``None``.
    """
    agent: str = "main"
    p_in_local: int = 0        # *new* prompt tokens contributed at this step (excl. carried history)
    p_out: int = 256           # generated tokens
    tool: Optional[Tool] = None
    retrieval: Optional[Retrieval] = None
