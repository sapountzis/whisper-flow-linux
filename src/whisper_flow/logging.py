"""Logging utilities for whisper-flow."""

# Global logging state
_logging_enabled = False


def set_logging_enabled(enabled: bool):
    """Set whether logging is enabled globally."""
    global _logging_enabled
    _logging_enabled = enabled


def log(*args, **kwargs):
    """Conditional print function that only prints when logging is enabled."""
    if _logging_enabled:
        print(*args, **kwargs)
