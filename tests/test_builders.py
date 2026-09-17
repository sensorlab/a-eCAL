"""Parity of the workflow builders: four_agent_rag / tree / debate."""

from __future__ import annotations

import itertools

import pytest

from _dual import NEW, ORIG, same


def _cmp(ow, nw):
    assert len(ow.steps) == len(nw.steps)
    assert same(ow.run(), nw.run())
    assert same(ow.agentic_ecal(), nw.agentic_ecal())


@pytest.mark.parametrize("model_key", [None, "llama3_8b", "qwen2_5_7b"])
@pytest.mark.parametrize("hw_key", [None, "a100", "h100"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_four_agent_rag_workflow(model_key, hw_key, rounds):
    om = ORIG.LLMS[model_key] if model_key else None
    nm = NEW.LLMS[model_key] if model_key else None
    oh = ORIG.HW[hw_key] if hw_key else None
    nh = NEW.HW[hw_key] if hw_key else None
    _cmp(ORIG.four_agent_rag_workflow(om, oh, rounds),
         NEW.four_agent_rag_workflow(nm, nh, rounds))


@pytest.mark.parametrize("fanout,depth", list(itertools.product([1, 2, 3], [0, 1, 2])))
@pytest.mark.parametrize("p_out", [100, 250])
@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("aggregate", [True, False])
def test_tree_workflow(fanout, depth, p_out, batch, aggregate):
    _cmp(ORIG.tree_workflow(ORIG.LLMS["llama3_8b"], ORIG.HW["a100"], fanout, depth,
                            p_out=p_out, batch=batch, aggregate=aggregate),
         NEW.tree_workflow(NEW.LLMS["llama3_8b"], NEW.HW["a100"], fanout, depth,
                           p_out=p_out, batch=batch, aggregate=aggregate))


@pytest.mark.parametrize("n_agents", [1, 2, 3])
@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("p_out", [120.0, 200.4])   # fractional exercises int(round(...))
@pytest.mark.parametrize("history_rounds", [None, 0, 1, 2])
@pytest.mark.parametrize("exclude_self", [True, False])
def test_debate_workflow(n_agents, rounds, p_out, history_rounds, exclude_self):
    _cmp(ORIG.debate_workflow(ORIG.LLMS["llama3_8b"], ORIG.HW["a100"], n_agents, rounds,
                              p_out, 400, history_rounds, exclude_self),
         NEW.debate_workflow(NEW.LLMS["llama3_8b"], NEW.HW["a100"], n_agents, rounds,
                             p_out, 400, history_rounds, exclude_self))
