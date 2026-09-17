"""Workflow builders: the paper's case study plus the tree and debate topologies."""

from __future__ import annotations

from typing import List, Optional

from .descriptors import Hardware, LLM
from .components import Step, Tool, Retrieval
from .workflow import Workflow
from .registries import HW, LLMS


def four_agent_rag_workflow(model: Optional[LLM] = None, hw: Optional[Hardware] = None,
                            rounds: int = 1) -> Workflow:
    """Planner -> retriever -> analyst -> writer, carrying history, repeated ``rounds`` times.

    ``rounds=1`` is the single-pass case study; larger values re-run the same team over the
    accumulated transcript, which is the depth axis of the composition figure.

    Single definition of the case study, so the component breakdown, the amortization curve and
    the numbers quoted in the text cannot drift apart. Callers choose the model and hardware;
    the token budget and topology are fixed.

    This is the only case-study workflow in the codebase. A competing planner-plus-two-workers
    topology, which produced the 2.4e3 J and 43.2 kbit the manuscript once quoted, was removed;
    audit finding C5 records that regime choice as still open for the co-authors rather than
    settled here. Energies therefore depend on the operating point and are not quoted in this
    docstring, where they went stale twice; run make_figure_workflow.py for current values.

    Args:
        model (Optional[LLM]): The model to run on. Defaults to ``LLMS["llama3_8b"]``.
        hw (Optional[Hardware]): The accelerator to run on. Defaults to ``HW["a100"]``.
        rounds (int): Number of times to repeat the four-agent team over the accumulated
            transcript. Defaults to ``1`` (the single-pass case study).

    Returns:
        Workflow: The configured case-study workflow.
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
    #
    # tx_energy_per_message = 0: this is the CENTRALIZED case study, four co-located agents, and
    # the co-located column of the placement table is zero by construction. It was 0.5 J, a flat
    # per-message constant, which is the wrong shape as well as the wrong value: osi.segment_energy
    # and osi.endpoint_stack_energy are exactly linear in payload with zero intercept, so there is
    # no fixed per-message cost for a constant to represent, and the three hand-offs here differ
    # 3.85x in size (200/320/770 tokens). The value was also bearer-independent, where the correct
    # figure spans 0.003 J (metro) to 17 J (NB-IoT) per message. Distribution is priced ON TOP of
    # E_W by placement.py/osi.py -- which is what fig_workflow(d) and the placement section do --
    # so carrying a second, flat transmission model inside E_W double-counted the same hand-offs
    # by a different method. debate_workflow already sets 0.0.
    return Workflow(model=model or LLMS["llama3_8b"], hw=hw or HW["a100"], steps=steps,
                    carry_history=True, gamma_v=0.10, tx_energy_per_message=0.0)


def tree_workflow(model: LLM, hw: Hardware, fanout: int, depth: int,
                  p_out: int = 250, sys_tokens: int = 400, p_in_root: int = 300,
                  batch: float = 1.0, aggregate: bool = True) -> Workflow:
    """A fan-out tree: a root delegates to ``fanout`` children, each of which may delegate again.

    Every node reads only its parent's output, so a node's prompt is bounded by one hand-off
    however wide or deep the tree becomes. That is the structural difference from a chain, where
    each step reads the whole accumulated transcript. With ``aggregate`` the root reads its
    children's outputs back on the way up, which is the only place a prompt grows with fan-out.

    Args:
        model (LLM): The model every node runs on.
        hw (Hardware): The accelerator every node runs on.
        fanout (int): Number of children each node delegates to.
        depth (int): Number of delegation levels below the root.
        p_out (int): Generated tokens per node. Defaults to ``250``.
        sys_tokens (int): System prompt prepended to every call. Defaults to ``400``.
        p_in_root (int): Local prompt tokens at the root. Defaults to ``300``.
        batch (float): Serving batch ``b``. Defaults to ``1.0``.
        aggregate (bool): If ``True``, the root reads its children's outputs back. Defaults to
            ``True``.

    Returns:
        Workflow: The configured fan-out tree workflow.
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

    Args:
        model (LLM): The model every agent runs on.
        hw (Hardware): The accelerator every agent runs on.
        n_agents (int): Number of agents (team width, i.e. the round size).
        rounds (int): Number of debate rounds.
        p_out (float): Generated tokens per call (rounded to an int per step).
        sys_tokens (int): Fixed prompt scaffold prepended to every call.
        history_rounds (Optional[int]): Rounds of transcript retained; ``None`` retains all,
            ``0`` retains none. Defaults to ``1`` (previous round only).
        exclude_self (bool): Skip an agent's own output within the retained window. Defaults to
            ``True``.

    Returns:
        Workflow: The configured debate workflow.
    """
    steps = [Step(agent=f"agent{i}", p_in_local=0, p_out=int(round(p_out)))
             for _ in range(rounds) for i in range(n_agents)]
    return Workflow(model=model, hw=hw, steps=steps, sys_tokens=sys_tokens,
                    carry_history=True, history_rounds=history_rounds,
                    round_size=n_agents, exclude_self=exclude_self,
                    gamma_v=0.10, tx_energy_per_message=0.0)
