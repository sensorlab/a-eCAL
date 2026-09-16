"""Command-line entry point: the model's self-consistency demo, with an optional ``--json`` mode.

The default (human) output is byte-for-byte identical to running the original
``agentic_ecal.py`` module directly, so it doubles as a reproduction check.
"""

from __future__ import annotations

import argparse
import json

from .components import Step
from .registries import HW, LLMS
from .rates import llm_call_energy
from .validation import validate_two_rate, validate_against_samsi, validate_current_gen
from .workflow import Workflow


def _demo_numbers() -> dict:
    """Compute the values the demo reports, so the human and ``--json`` paths agree.

    Returns:
        dict: ``{"two_rate": ..., "anchors": ..., "reduce": ...}`` with the derived-vs-measured
        coefficients, the literature-anchor implied batches, and the one-shot reduction check.
    """
    m, hw = LLMS["llama3_8b"], HW["h100"]
    wf = Workflow(model=m, hw=hw, steps=[Step(p_in_local=600, p_out=300)],
                  sys_tokens=0, carry_history=True, gamma_v=0.0, serving_batch=64)
    direct = llm_call_energy(m, hw, 600, 300, 64)
    one_shot = wf.run()["e_total"]
    return {
        "two_rate": validate_two_rate(),
        "anchors": {**validate_against_samsi(), **validate_current_gen()},
        "reduce": {
            "one_shot_J": one_shot,
            "direct_J": direct,
            "match": abs(one_shot - direct) < 1e-6,
            "agentic_ecal_J_per_bit": wf.agentic_ecal(),
        },
    }


def _print_human() -> None:
    """Emit the original ``__main__`` output verbatim."""
    print("[two-rate] derived vs measured coefficients (A100, nothing fitted):")
    for name, v in validate_two_rate().items():
        print(f"  {name:<13} c_pre {v['c_pre'][0]:.4f} vs {v['c_pre'][1]}   "
              f"a_dec {v['a_dec'][0]} vs {v['a_dec'][1]}")
    print("\n[anchors] what serving batch each literature figure implies:")
    for k, v in {**validate_against_samsi(), **validate_current_gen()}.items():
        print(f"  {k}: {v}")

    m, hw = LLMS["llama3_8b"], HW["h100"]
    wf = Workflow(model=m, hw=hw, steps=[Step(p_in_local=600, p_out=300)],
                  sys_tokens=0, carry_history=True, gamma_v=0.0, serving_batch=64)
    direct = llm_call_energy(m, hw, 600, 300, 64)
    one_shot = wf.run()["e_total"]
    print(f"\n[reduce] one-shot workflow {one_shot:.2f} J == direct call {direct:.2f} J: "
          f"{abs(one_shot - direct) < 1e-6}")
    print(f"  agentic-eCAL (one-shot, b=64): {wf.agentic_ecal():.2e} J/bit")


def main(argv=None) -> None:
    """Entry point for ``agentic-ecal`` / ``python -m agentic_ecal_pkg``.

    Args:
        argv (Optional[list[str]]): Argument vector to parse. Defaults to ``None`` (uses
            ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        prog="agentic-ecal",
        description="Two-rate energy model for agentic LLM workflows (self-consistency demo).")
    parser.add_argument("--json", action="store_true",
                        help="emit the demo values as machine-readable JSON instead of text")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(_demo_numbers(), indent=2))
    else:
        _print_human()


if __name__ == "__main__":
    main()
