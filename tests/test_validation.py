"""Parity of the literature-anchor validators and the internal implied_batch helper."""

from __future__ import annotations

import pytest

from _dual import HW_KEYS, MODEL_KEYS, NEW, ORIG, same

# implied_batch is package-internal (not exported); reach it on each module.
_orig_implied = ORIG.implied_batch
_new_implied = NEW.validation.implied_batch


def test_validate_two_rate():
    assert same(ORIG.validate_two_rate(), NEW.validate_two_rate())


def test_validate_against_samsi():
    assert same(ORIG.validate_against_samsi(), NEW.validate_against_samsi())


def test_validate_current_gen():
    assert same(ORIG.validate_current_gen(), NEW.validate_current_gen())


@pytest.mark.parametrize("mk", MODEL_KEYS)
@pytest.mark.parametrize("hk", HW_KEYS)
@pytest.mark.parametrize("j", [0.084, 0.70, 3.0, 4.0, 25.6, 30.0])
@pytest.mark.parametrize("p_in,p_out", [(64, 512), (50, 382)])
def test_implied_batch(mk, hk, j, p_in, p_out):
    assert same(_orig_implied(ORIG.LLMS[mk], ORIG.HW[hk], j, p_in, p_out),
                _new_implied(NEW.LLMS[mk], NEW.HW[hk], j, p_in, p_out))
