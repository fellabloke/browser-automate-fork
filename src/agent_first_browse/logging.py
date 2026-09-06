"""Shared structured logging for the agent runtime.

This is the canonical owner for the logger used by core runtime packages.
"""

from __future__ import annotations

import logging as _logging
import os
import sys
import time
from pathlib import Path

_FORMATTER = _logging.Formatter(
    fmt="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
_FILE_HANDLER: _logging.Handler | bool | None = None


def _file_handler() -> _logging.Handler | None:
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        if "pytest" in sys.modules or os.getenv("AGENT_NO_FILE_LOG"):
            _FILE_HANDLER = False
        else:
            try:
                log_dir = Path(os.getenv("AGENT_LOG_DIR", "logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                handler = _logging.FileHandler(
                    log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
                    encoding="utf-8",
                )
                handler.setFormatter(_FORMATTER)
                _FILE_HANDLER = handler
            except Exception:
                _FILE_HANDLER = False
    return _FILE_HANDLER or None


def get_logger(name: str, level: int = _logging.INFO) -> _logging.Logger:
    """Return the consistently configured logger for a runtime module."""
    logger = _logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = _logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
        file_handler = _file_handler()
        if file_handler is not None:
            logger.addHandler(file_handler)
        logger.propagate = False
    return logger


__all__ = ["get_logger"]
