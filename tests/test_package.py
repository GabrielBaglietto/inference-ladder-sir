"""Smoke test: the package imports and reports a version.

Placeholder until the first real module lands in M1; it exists so that CI has
something to run and fails loudly if packaging breaks.
"""

import inferladder


def test_version_is_exposed() -> None:
    assert isinstance(inferladder.__version__, str)
    assert inferladder.__version__
