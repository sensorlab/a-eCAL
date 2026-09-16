#!/usr/bin/env python3
"""Per-call and per-token energy for a preset model on preset hardware.

Run:  python examples/basic_call.py
"""

import os
import sys

# Make the package importable when running from a source checkout without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_ecal_pkg as ae


def main() -> None:
    model = ae.LLMS["llama3_8b"]
    hw = ae.HW["a100"]
    p_in, p_out = 512, 512

    for batch in (1, 32, 256):
        e = ae.llm_call_energy(model, hw, p_in, p_out, batch)
        jpt = ae.joules_per_token(model, hw, p_in, p_out, batch)
        t = ae.call_latency(model, hw, p_in, p_out, batch)
        print(f"{model.name} on {hw.name}, batch={batch:>3}: "
              f"{e:8.2f} J/call  |  {jpt:6.3f} J/token  |  {t:6.3f} s latency")


if __name__ == "__main__":
    main()
