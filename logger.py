"""
core/logger.py — Centralized Structured Logging
Part of the Be More Agent architecture migration (Phase 0).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional


LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-30s │ %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = "logs/agent.log",
    max_bytes: int = 2 * 1024 * 1024,   # 2 MB
    backup_count: int = 3,
) -> None:
    """
    Configure root logger with console + optional rotating file handler.

    Call once at application startup before importing other modules.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers on re-import
    if root.handlers:
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    # Rotating file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
