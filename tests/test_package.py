"""Packaging smoke test.

Cheap, but it catches the two ways packaging silently rots: the version drifting
between the module and the distribution metadata, and the public surface losing an
export that notebooks import by name.
"""

from importlib.metadata import version

import inferladder


def test_module_version_matches_distribution_metadata() -> None:
    assert inferladder.__version__ == version("inferladder")


def test_public_surface_is_importable() -> None:
    for name in inferladder.__all__:
        assert hasattr(inferladder, name), name
