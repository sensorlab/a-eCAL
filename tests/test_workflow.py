"""Parity of Workflow.run() (all 13 keys) and agentic_ecal() across the parameter factorial."""

from __future__ import annotations

import itertools

import pytest

from _dual import NEW, ORIG, same


def _build_steps(mod, spec):
    steps = []
    for s in spec:
        tool = mod.Tool(**s["tool"]) if s.get("tool") else None
        ret = mod.Retrieval(**s["retrieval"]) if s.get("retrieval") else None
        steps.append(mod.Step(agent=s.get("agent", "main"),
                              p_in_local=s.get("p_in_local", 0),
                              p_out=s.get("p_out", 256),
                              tool=tool, retrieval=ret))
    return steps


EMPTY = []
SINGLE = [{"p_in_local": 600, "p_out": 300}]
MULTI = [
    {"agent": "a", "p_in_local": 300, "p_out": 200},
    {"agent": "b", "p_in_local": 120, "p_out": 120,
     "retrieval": {"k_chunks": 5, "chunk_tokens": 256}},
    {"agent": "a", "p_in_local": 200, "p_out": 300,
     "tool": {"name": "code", "energy": 2.0, "obs_tokens": 150}},
    {"agent": "c", "p_in_local": 200, "p_out": 400},
    {"agent": "c", "p_in_local": 50, "p_out": 128},
]
STEP_SPECS = {"empty": EMPTY, "single": SINGLE, "multi": MULTI}

# A curated factorial over every run() axis (kept representative, not exhaustive).
_AXES = list(itertools.product(
    [True, False],              # carry_history
    [None, "llama3_8b"],        # calib key (None -> derived path)
    [0, 1, 2, 4],               # round_size
    [None, 0, 1, 2],            # history_rounds
    [True, False],              # exclude_self
    [1, 32, 256],               # serving_batch
    [0, 400],                   # sys_tokens
    [0.0, 0.10],                # gamma_v
    [0.0, 0.5],                 # tx_energy_per_message
))
# Sample deterministically to keep the run fast while covering every axis value.
_COMBOS = _AXES[::37]


def _make(mod, spec_name, model_key, hw_key, axis):
    ch, calib_key, rs, hr, ex, sb, sys_t, gv, tx = axis
    calib = mod.TWO_RATE[calib_key] if calib_key else None
    return mod.Workflow(
        model=mod.LLMS[model_key], hw=mod.HW[hw_key],
        steps=_build_steps(mod, STEP_SPECS[spec_name]),
        sys_tokens=sys_t, carry_history=ch, history_rounds=hr, round_size=rs,
        exclude_self=ex, serving_batch=sb, calib=calib, gamma_v=gv,
        tx_energy_per_message=tx)


@pytest.mark.parametrize("spec_name", list(STEP_SPECS))
@pytest.mark.parametrize("axis", _COMBOS)
def test_run_and_agentic_ecal(spec_name, axis):
    ow = _make(ORIG, spec_name, "llama3_8b", "a100", axis)
    nw = _make(NEW, spec_name, "llama3_8b", "a100", axis)
    assert same(ow.run(), nw.run())
    for bpt in (17.0, 8.0):
        for emb in (0.0, 1e12):
            assert same(ow.agentic_ecal(bpt, emb), nw.agentic_ecal(bpt, emb))


@pytest.mark.parametrize("model_key", ["llama3_8b", "llama3_70b", "qwen2_5_7b"])
@pytest.mark.parametrize("hw_key", ["h100", "h100_nvl", "a100"])
def test_run_across_models_and_hw(model_key, hw_key):
    axis = _COMBOS[len(_COMBOS) // 2]
    ow = _make(ORIG, "multi", model_key, hw_key, axis)
    nw = _make(NEW, "multi", model_key, hw_key, axis)
    assert same(ow.run(), nw.run())


def test_reduce_case_one_shot_equals_direct_call():
    """The demo's self-consistency check, verified in the new package."""
    m, hw = NEW.LLMS["llama3_8b"], NEW.HW["h100"]
    wf = NEW.Workflow(model=m, hw=hw, steps=[NEW.Step(p_in_local=600, p_out=300)],
                      sys_tokens=0, carry_history=True, gamma_v=0.0, serving_batch=64)
    direct = NEW.llm_call_energy(m, hw, 600, 300, 64)
    assert abs(wf.run()["e_total"] - direct) < 1e-6
