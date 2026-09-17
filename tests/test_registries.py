"""Parity of the HW / LLMS / TWO_RATE preset registries, field by field."""

from __future__ import annotations

import dataclasses

import pytest

from _dual import NEW, ORIG, same

BATCH = [0, 1, 32, 64, 128, 256]


def _fields_equal(o, n):
    of = {f.name: getattr(o, f.name) for f in dataclasses.fields(o)}
    nf = {f.name: getattr(n, f.name) for f in dataclasses.fields(n)}
    assert of.keys() == nf.keys()
    for k in of:
        assert same(of[k], nf[k]), k


def test_registry_keys_match():
    assert ORIG.HW.keys() == NEW.HW.keys()
    assert ORIG.LLMS.keys() == NEW.LLMS.keys()
    assert ORIG.TWO_RATE.keys() == NEW.TWO_RATE.keys()


@pytest.mark.parametrize("k", list(ORIG.HW.keys()))
def test_hw_entry(k):
    o, n = ORIG.HW[k], NEW.HW[k]
    _fields_equal(o, n)
    assert same(o.flops_eff, n.flops_eff)


@pytest.mark.parametrize("k", list(ORIG.LLMS.keys()))
def test_llm_entry(k):
    o, n = ORIG.LLMS[k], NEW.LLMS[k]
    _fields_equal(o, n)
    assert same(o.kv_bytes_per_token, n.kv_bytes_per_token)


@pytest.mark.parametrize("k", list(ORIG.TWO_RATE.keys()))
def test_two_rate_entry(k):
    o, n = ORIG.TWO_RATE[k], NEW.TWO_RATE[k]
    _fields_equal(o, n)
    for b in BATCH:
        assert same(o.c_dec(b), n.c_dec(b)), b
