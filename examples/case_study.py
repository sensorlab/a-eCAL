#!/usr/bin/env python3
"""The paper's four-agent RAG case study: full energy breakdown and the agentic-eCAL metric.

Run:  python examples/case_study.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_ecal_pkg as ae


def main() -> None:
    wf = ae.four_agent_rag_workflow()   # Llama-3 8B on A100, single pass
    r = wf.run()

    print("Four-agent RAG case study (planner -> retriever -> analyst -> writer)")
    print(f"  model / hardware : {wf.model.name} / {wf.hw.name}")
    print(f"  prefill tokens   : {r['prefill_tokens']}")
    print(f"  decode tokens    : {r['decode_tokens']}")
    print()
    for key in ("e_llm", "e_prefill", "e_decode", "e_tool", "e_retrieval",
                "e_transmission", "e_orchestration", "e_total"):
        print(f"  {key:<16}: {r[key]:10.3f} J")
    print()
    print(f"  agentic-eCAL     : {wf.agentic_ecal():.4f} J/bit")


if __name__ == "__main__":
    main()
