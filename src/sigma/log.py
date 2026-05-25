"""Sigma logging — thin wrapper around Python logging.

Every Sigma module gets a logger via `get_logger(__name__)`.
Applications configure handlers via `setup_logging()` or standard `logging.basicConfig()`.
"""

import logging
import re
import sys

_logger_registry: dict[str, logging.Logger] = {}

# Patterns that should never appear in log output.
_SECRET_PATTERNS = (
    (re.compile(r'sk-[A-Za-z0-9]{20,}'), '[REDACTED_KEY]'),
    (re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{20,}'), 'Bearer [REDACTED]'),
    (re.compile(r'-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH)\s+PRIVATE KEY-----'), '[REDACTED_KEY]'),
    (re.compile(r'pypi-[A-Za-z0-9_-]{20,}'), '[REDACTED_TOKEN]'),
)


class _SecretRedactionFilter(logging.Filter):
    """Strip API keys, tokens, and private keys from log messages."""

    def filter(self, record):
        msg = str(record.msg)
        for pattern, replacement in _SECRET_PATTERNS:
            msg = pattern.sub(replacement, msg)
        record.msg = msg
        if record.args and isinstance(record.args, dict):
            record.args = {
                k: pattern.sub(replacement, str(v)) if isinstance(v, str) else v
                for k, v in record.args.items()
                for pattern, replacement in _SECRET_PATTERNS
            }
        return True
_initialized = False


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a Sigma module.

    On first call, a default console handler is installed (INFO level, stdout).
    Call `setup_logging()` explicitly to override.
    """
    global _initialized
    if not _initialized:
        _init_defaults()
    if name not in _logger_registry:
        _logger_registry[name] = logging.getLogger(name)
    return _logger_registry[name]


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
    datefmt: str = "%H:%M:%S",
    stream=None,
    file_path: str | None = None,
):
    """Configure Sigma logging globally.

    Args:
        level: Log level (default INFO).
        fmt: Log message format.
        datefmt: Date format for timestamps.
        stream: Output stream (default stdout).
        file_path: If set, also log to this file.
    """
    global _initialized
    logger = logging.getLogger("sigma")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addFilter(_SecretRedactionFilter())

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)

    if file_path:
        from pathlib import Path
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    _initialized = True


def _init_defaults():
    """Install a minimal default handler so logging works out of the box."""
    setup_logging()
