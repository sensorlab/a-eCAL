"""Put the repository root on ``sys.path`` so ``import agentic_ecal_pkg`` resolves.

The figure scripts live in ``figure_scripts/`` while the package lives one level up at the
repository root in ``agentic_ecal_pkg/``. Importing this module (before importing the package)
adds the repo root to the import path. Import it for its side effect:

    import _pkg_path  # noqa: F401
    import agentic_ecal_pkg as ae
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
