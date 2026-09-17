"""Parity of Tool / Retrieval / TwoRateCalib behaviour."""

from __future__ import annotations

import itertools

import pytest

from _dual import HW_KEYS, NEW, ORIG, same

# Retrieval parameter grid, incl. the log2(max(n,2)) edge and both search modes.
_RET_GRID = list(itertools.product(
    [True, False],          # ann
    [0, 5],                 # k_chunks
    [0, 256],               # chunk_tokens
    [1, 2, 1e6],            # n_vectors (1 exercises the log2 clamp)
    [1.1e8, 7.615e9],       # embed_params
    [0, 64],                # query_tokens
))


@pytest.mark.parametrize("hk", HW_KEYS)
@pytest.mark.parametrize("ann,k_chunks,chunk_tokens,n_vectors,embed_params,query_tokens", _RET_GRID)
def test_retrieval_energy_and_context(hk, ann, k_chunks, chunk_tokens, n_vectors,
                                      embed_params, query_tokens):
    kw = dict(ann=ann, k_chunks=k_chunks, chunk_tokens=chunk_tokens, n_vectors=n_vectors,
              embed_params=embed_params, query_tokens=query_tokens)
    o = ORIG.Retrieval(**kw)
    n = NEW.Retrieval(**kw)
    assert same(o.energy(ORIG.HW[hk]), n.energy(NEW.HW[hk]))
    assert same(o.added_context, n.added_context)


def test_tool_defaults_and_values():
    for kw in (dict(name="code"), dict(name="api", energy=2.0, obs_tokens=150)):
        o, n = ORIG.Tool(**kw), NEW.Tool(**kw)
        assert same(o.energy, n.energy)
        assert same(o.obs_tokens, n.obs_tokens)


_CALIB = [("Q", 0.023, 3.2, 0.02), ("L", 0.028, 3.3, 0.06), ("z", 0.0, 0.0, 0.0)]


@pytest.mark.parametrize("name,c_pre,a_dec,kv_floor", _CALIB)
@pytest.mark.parametrize("batch", [0, 1, 32, 256])
@pytest.mark.parametrize("p_in,p_out", [(0, 0), (64, 512), (300, 200)])
def test_two_rate_calib(name, c_pre, a_dec, kv_floor, batch, p_in, p_out):
    o = ORIG.TwoRateCalib(name, c_pre, a_dec, kv_floor)
    n = NEW.TwoRateCalib(name, c_pre, a_dec, kv_floor)
    assert same(o.c_dec(batch), n.c_dec(batch))
    assert same(o.energy(p_in, p_out, batch), n.energy(p_in, p_out, batch))
