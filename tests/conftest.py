"""Shared pytest fixtures exposing the dual-loaded modules and the bit-exact comparator."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling helper importable regardless of pytest's import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from _dual import NEW, ORIG  # noqa: E402
from _dual import same as _same  # noqa: E402


@pytest.fixture(scope="session")
def orig():
    """The original ``agentic_ecal.py`` module (golden master)."""
    return ORIG


@pytest.fixture(scope="session")
def new():
    """The new ``agentic_ecal_pkg`` package."""
    return NEW


@pytest.fixture(scope="session")
def same():
    """Bit-exact recursive comparator."""
    return _same
