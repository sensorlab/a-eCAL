"""Pin the specific anchors and case-study figures the manuscript reports.

Unlike the parity tests (new == original), these pin literal expected values, so a change that
drifts *both* implementations away from the numbers in the paper is still caught. All values are
computed from the NEW package; each is also asserted equal to the ORIGINAL as a cross-check.
"""

from __future__ import annotations

import pytest

from _dual import NEW, ORIG, same

# Two-rate check (paper Table II): derived c_pre/a_dec vs the measured A100-sweep targets.
EXPECTED_TWO_RATE = {
    "Qwen2.5-7B": {"c_pre": (0.023, 0.023), "a_dec": (4.27, 3.2)},
    "Llama-3 8B": {"c_pre": (0.0242, 0.028), "a_dec": (4.5, 3.3)},
}
EXPECTED_SAMSI = {
    "J/token at b=1": 36.8,
    "implied batch for 3 J/token": 13.3,
    "implied batch for 4 J/token": 9.8,
}
EXPECTED_CURRENT_GEN = {
    "llama3_70b implied batch at 0.70 J/token (Fig. 3b)": 30.6,
    "llama3_8b implied batch at 0.084 J/token (released runs, active GPU)": 30.0,
}


def test_two_rate_anchors():
    got = NEW.validate_two_rate()
    assert got == EXPECTED_TWO_RATE
    assert same(got, ORIG.validate_two_rate())


def test_samsi_anchor():
    got = NEW.validate_against_samsi()
    assert got == EXPECTED_SAMSI
    assert same(got, ORIG.validate_against_samsi())


def test_current_gen_anchor():
    got = NEW.validate_current_gen()
    assert got == EXPECTED_CURRENT_GEN
    assert same(got, ORIG.validate_current_gen())


def test_case_study_breakdown():
    """The single-pass four-agent RAG case study (Llama-3 8B on A100)."""
    r = NEW.four_agent_rag_workflow().run()
    # Token budget and topology are fixed, so these are exact.
    assert r["prefill_tokens"] == 4990
    assert r["decode_tokens"] == 1020
    assert r["useful_output_tokens"] == 1020
    # Energies at the fixed operating point.
    assert r["e_total"] == pytest.approx(5240.261244, rel=1e-9)
    assert r["e_llm"] == pytest.approx(4761.512832, rel=1e-9)
    # And identical to the golden master.
    assert same(r, ORIG.four_agent_rag_workflow().run())


def test_case_study_agentic_ecal():
    got = NEW.four_agent_rag_workflow().agentic_ecal()
    assert got == pytest.approx(0.3022065308190462, rel=1e-12)
    assert same(got, ORIG.four_agent_rag_workflow().agentic_ecal())
