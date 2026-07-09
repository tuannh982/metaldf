"""metaldf — GPU-accelerated DataFrame library for Apple Silicon."""

from metaldf._accelerator import install, uninstall

__all__ = [
    "install",
    "uninstall",
]

__version__ = "0.1.0"
