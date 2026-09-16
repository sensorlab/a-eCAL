"""Parity of the two-rate call model: rates, energy, latency, embedding FLOPs."""

from __future__ import annotations

import pytest

from _dual import HW_KEYS, MODEL_KEYS, NEW, ORIG, same

P_IN = [0, 1, 50, 64, 300, 512, 2048]
P_OUT = [1, 120, 256, 382, 512]
BATCH = [0, 1, 32, 64, 256]
CONTEXT = [0, 64, 512, 4096]


@pytest.mark.parametrize("mk", MODEL_KEYS)
@pytest.mark.parametrize("hk", HW_KEYS)
def test_prefill_rate(mk, hk):
    assert same(ORIG.prefill_rate(ORIG.LLMS[mk], ORIG.HW[hk]),
                NEW.prefill_rate(NEW.LLMS[mk], NEW.HW[hk]))


@pytest.mark.parametrize("mk", MODEL_KEYS)
@pytest.mark.parametrize("hk", HW_KEYS)
@pytest.mark.parametrize("batch", BATCH)
@pytest.mark.parametrize("ctx", CONTEXT)
def test_decode_rate(mk, hk, batch, ctx):
    assert same(ORIG.decode_rate(ORIG.LLMS[mk], ORIG.HW[hk], batch, ctx),
                NEW.decode_rate(NEW.LLMS[mk], NEW.HW[hk], batch, ctx))


def test_decode_rate_no_bandwidth_raises_same_message():
    oh = ORIG.Hardware("no-bw", flops_peak=1e12, power=100.0)   # bandwidth defaults to 0
    nh = NEW.Hardware("no-bw", flops_peak=1e12, power=100.0)
    with pytest.raises(ValueError) as oe:
        ORIG.decode_rate(ORIG.LLMS["llama3_8b"], oh, 1, 0)
    with pytest.raises(ValueError) as ne:
        NEW.decode_rate(NEW.LLMS["llama3_8b"], nh, 1, 0)
    assert str(oe.value) == str(ne.value)


@pytest.mark.parametrize("mk", MODEL_KEYS)
@pytest.mark.parametrize("hk", HW_KEYS)
@pytest.mark.parametrize("p_in", P_IN)
@pytest.mark.parametrize("p_out", P_OUT)
@pytest.mark.parametrize("batch", [0, 1, 32, 256])
def test_call_latency_and_energy(mk, hk, p_in, p_out, batch):
    om, oh = ORIG.LLMS[mk], ORIG.HW[hk]
    nm, nh = NEW.LLMS[mk], NEW.HW[hk]
    assert same(ORIG.call_latency(om, oh, p_in, p_out, batch),
                NEW.call_latency(nm, nh, p_in, p_out, batch))
    assert same(ORIG.llm_call_energy(om, oh, p_in, p_out, batch),
                NEW.llm_call_energy(nm, nh, p_in, p_out, batch))


@pytest.mark.parametrize("mk", MODEL_KEYS)
@pytest.mark.parametrize("hk", HW_KEYS)
def test_joules_per_token_defaults(mk, hk):
    assert same(ORIG.joules_per_token(ORIG.LLMS[mk], ORIG.HW[hk]),
                NEW.joules_per_token(NEW.LLMS[mk], NEW.HW[hk]))


@pytest.mark.parametrize("p_out", [1, 256, 512])
def test_joules_per_token_grid(p_out):
    om, oh = ORIG.LLMS["qwen2_5_7b"], ORIG.HW["a100"]
    nm, nh = NEW.LLMS["qwen2_5_7b"], NEW.HW["a100"]
    assert same(ORIG.joules_per_token(om, oh, 64, p_out, 32),
                NEW.joules_per_token(nm, nh, 64, p_out, 32))


@pytest.mark.parametrize("n_params", [0.0, 1.1e8, 7.615e9])
@pytest.mark.parametrize("n_tokens", [0, 64, 256, 512])
def test_embedding_flops(n_params, n_tokens):
    assert same(ORIG.embedding_flops(n_params, n_tokens),
                NEW.embedding_flops(n_params, n_tokens))
