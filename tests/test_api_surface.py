"""The new package must expose exactly the original public API, in the same order."""

from __future__ import annotations

from _dual import NEW, ORIG


def test_all_identical():
    assert NEW.__all__ == ORIG.__all__


def test_every_name_present():
    for name in ORIG.__all__:
        assert hasattr(NEW, name), f"missing on new package: {name}"


def test_kinds_match():
    import inspect
    for name in ORIG.__all__:
        o, n = getattr(ORIG, name), getattr(NEW, name)
        assert inspect.isclass(o) == inspect.isclass(n), name
        assert inspect.isfunction(o) == inspect.isfunction(n), name
        assert isinstance(o, dict) == isinstance(n, dict), name
