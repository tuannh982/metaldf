"""Test that the public API surface is correct."""

import metaldf


def test_version_exists():
    assert hasattr(metaldf, "__version__")
    assert metaldf.__version__ == "0.1.0"


def test_install_exported():
    assert hasattr(metaldf, "install")


def test_uninstall_exported():
    assert hasattr(metaldf, "uninstall")
