"""Load the original module and the new package side by side for golden-master comparison.

The original single-file model is frozen alongside the tests as
``tests/agentic_ecal_reference.py`` -- the golden master this suite proves the package against.
It and the package (``agentic_ecal_pkg``) are loaded together; to avoid the name clash they carry
different names: the package keeps its own name, and the reference is loaded by explicit file path
under the alias ``agentic_ecal_orig``.
"""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../a-eCAL/tests
ROOT = HERE.parent                              # .../a-eCAL (repo root)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NEW package (modular refactor).
import agentic_ecal_pkg as NEW  # noqa: E402

# ORIGINAL model (frozen golden-master reference), loaded by path under a unique alias.
ORIG_PATH = HERE / "agentic_ecal_reference.py"
_spec = importlib.util.spec_from_file_location("agentic_ecal_orig", ORIG_PATH)
ORIG = importlib.util.module_from_spec(_spec)
sys.modules["agentic_ecal_orig"] = ORIG
_spec.loader.exec_module(ORIG)


def same(a, b) -> bool:
    """True iff ``a`` and ``b`` are bit-identical, recursing into dict/list/tuple.

    Floats are compared by their IEEE-754 bytes, so ``-0.0`` differs from ``0.0`` and ``inf`` is
    handled; two NaNs count as equal (both implementations should produce NaN in the same places).
    """
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return struct.pack("<d", fa) == struct.pack("<d", fb)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() \
            and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) \
            and all(same(x, y) for x, y in zip(a, b))
    return a == b


MODEL_KEYS = ["llama3_8b", "llama3_70b", "llama_65b", "qwen2_5_7b"]
HW_KEYS = ["h100", "h100_nvl", "a100"]
