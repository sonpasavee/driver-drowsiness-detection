from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.config import Config

_LOGGING_CONFIGURED = False


def _resolve_level(level_name: str) -> int:
    return getattr(logging, level_name.upper(), logging.INFO)


def _configure_root_logger() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    cfg = Config()
    level_name = cfg.get("logging.level", "INFO")
    log_file = Path(cfg.get("logging.file", "logs/driver_monitor.log"))
    max_bytes = int(cfg.get("logging.max_bytes", 5 * 1024 * 1024))
    backup_count = int(cfg.get("logging.backup_count", 3))
    log_level = _resolve_level(level_name)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    stdout_stream = sys.stdout
    if hasattr(stdout_stream, "reconfigure"):
        try:
            stdout_stream.reconfigure(errors="replace")
        except OSError:
            pass

    console_handler = logging.StreamHandler(stdout_stream)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a project logger configured from `configs/system.yaml`."""
    _configure_root_logger()
    return logging.getLogger(name)
