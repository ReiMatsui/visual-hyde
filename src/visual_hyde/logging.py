"""
Centralized logging setup using Rich for pretty console output.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with Rich handler. Call once at startup."""
    global _configured
    if _configured:
        return
    _configured = True

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Auto-configures if needed."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
