"""The ``Workflow`` aggregator: walk steps, price each call, return the energy breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .constants import BITS_PER_TOKEN
from .descriptors import Hardware, LLM
from .components import Step
from .calib import TwoRateCalib
from .rates import prefill_rate, decode_rate, call_latency


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

    Attributes:
        model (LLM): The model every step runs on.
        hw (Hardware): The accelerator every step runs on.
        steps (List[Step]): The ordered steps of the workflow. Defaults to an empty list.
        sys_tokens (int): System / tool-schema prompt prepended to every call. Defaults to ``400``.
        carry_history (bool): Whether each step's prompt accumulates the running transcript.
            Defaults to ``True``.
        history_rounds (Optional[int]): Rounds of transcript retained; ``None`` retains every
            completed round. Defaults to ``None``.
        round_size (int): Steps per round (team width in a debate). Defaults to ``1``.
        exclude_self (bool): Skip the agent's own output within the retained window. Defaults to
            ``False``.
        serving_batch (float): Decode batch ``b`` the decode rate amortises over. Defaults to
            ``1.0``.
        calib (Optional[TwoRateCalib]): Measured coefficients; ``None`` uses the derived Eqs.
            (4)-(5). Defaults to ``None``.
        gamma_v (float): Virtualization / orchestration overhead (eCAL's ``gamma_v``). Defaults
            to ``0.10``.
        tx_energy_per_message (float): Inter-agent transmission energy per hand-off, in Joules.
            Defaults to ``0.0``.
    """
    model: LLM
    hw: Hardware
    steps: List[Step] = field(default_factory=list)
    sys_tokens: int = 400              # system / tool-schema prompt prepended to every call
    carry_history: bool = True
    history_rounds: Optional[int] = None   # None = every completed round
    round_size: int = 1                    # steps per round (team width in a debate)
    exclude_self: bool = False             # skip the agent's own output within the window
    serving_batch: float = 1.0         # b of the decode rate; decode amortises over it
    calib: Optional["TwoRateCalib"] = None   # measured coefficients; None uses Eqs. (4)-(5)
    gamma_v: float = 0.10              # virtualization / orchestration overhead (eCAL's gamma_v)
    tx_energy_per_message: float = 0.0  # inter-agent transmission energy [J] (OSI model)

    def run(self) -> dict:
        """Compute the full energy breakdown of the workflow.

        Returns:
            dict: A breakdown with keys (all energies in Joules, times in seconds, counts in
            tokens):

            - ``e_llm``: total LLM-call energy.
            - ``e_prefill``: prefill component of ``e_llm``.
            - ``e_decode``: decode component of ``e_llm``.
            - ``t_llm``: the request's share of device time.
            - ``t_wall``: wall-clock time of the calls run back to back.
            - ``e_tool``: total tool energy.
            - ``e_retrieval``: total retrieval energy.
            - ``e_transmission``: total inter-agent transmission energy.
            - ``e_orchestration``: the ``gamma_v`` overhead on the core energy.
            - ``e_total``: ``(1 + gamma_v)`` times the core energy.
            - ``prefill_tokens``: total prompt tokens across all calls.
            - ``decode_tokens``: total generated tokens across all calls.
            - ``useful_output_tokens``: generated tokens counted as useful output.
        """
        rs = max(self.round_size, 1)
        transcript = []      # (round index, agent, tokens) for outputs and tool observations
        e_llm = e_tool = e_ret = e_tx = 0.0
        e_prefill = e_decode = 0.0   # the two terms of the two-rate call model
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

        Args:
            bits_per_token (float): Information content per token. Defaults to
                :data:`~agentic_ecal_pkg.constants.BITS_PER_TOKEN`.
            e_emb_amortized (float): Share of the model's embodied (pre-training) energy
                attributed to this single workflow invocation, in Joules. Defaults to ``0.0``.

        Returns:
            float: ``(e_total + e_emb_amortized) / (bits_per_token * useful_output_tokens)``, in
            Joules per bit; ``nan`` when there is no useful output.
        """
        r = self.run()
        useful_bits = bits_per_token * r["useful_output_tokens"]
        return (r["e_total"] + e_emb_amortized) / useful_bits if useful_bits else float("nan")
