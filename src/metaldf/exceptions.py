"""metaldf exception hierarchy."""


class MetaldfError(Exception):
    """Base exception for all metaldf errors."""


class MetalNotAvailable(MetaldfError):
    """Raised when Metal acceleration is not available for an operation."""
