#!/usr/bin/env python3
"""Build a workflow by hand from Steps, a Tool, and a Retrieval, then price it.

Run:  python examples/custom_workflow.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_ecal_pkg as ae


def main() -> None:
    steps = [
        ae.Step(agent="planner", p_in_local=300, p_out=200),
        ae.Step(agent="searcher", p_in_local=120, p_out=150,
                retrieval=ae.Retrieval(k_chunks=4, chunk_tokens=200)),
        ae.Step(agent="coder", p_in_local=200, p_out=350,
                tool=ae.Tool(name="python", energy=3.0, obs_tokens=180)),
        ae.Step(agent="writer", p_in_local=150, p_out=400),
    ]
    wf = ae.Workflow(
        model=ae.LLMS["qwen2_5_7b"],
        hw=ae.HW["h100_nvl"],
        steps=steps,
        carry_history=True,
        serving_batch=32,
        calib=ae.TWO_RATE["qwen2_5_7b"],   # use the measured coefficients
    )
    r = wf.run()
    print(f"Custom 4-step workflow on {wf.model.name} / {wf.hw.name} (batch {int(wf.serving_batch)})")
    print(f"  total energy   : {r['e_total']:.2f} J")
    print(f"  useful output  : {r['useful_output_tokens']} tokens")
    print(f"  agentic-eCAL   : {wf.agentic_ecal():.4f} J/bit")

    # Compare with the derived (datasheet) rates instead of the measured calibration.
    wf.calib = None
    print(f"  agentic-eCAL (derived rates): {wf.agentic_ecal():.4f} J/bit")


if __name__ == "__main__":
    main()
