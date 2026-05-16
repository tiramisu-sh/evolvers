"""Shared structlog logger for evolvers.

Human-readable, colorized console output by default. Set `EVOLVERS_LOG_FORMAT=json`
for line-delimited JSON (log aggregation / downstream parsing). `EVOLVERS_LOG_LEVEL`
(default `INFO`) filters by level — set it to `DEBUG` to see per-trial events.

If the host application has already called `structlog.configure()`, evolvers
defers to that configuration rather than clobbering it.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def _build_logger() -> Any:
    if not structlog.is_configured():
        json_mode = os.environ.get("EVOLVERS_LOG_FORMAT", "").lower() == "json"
        level_name = os.environ.get("EVOLVERS_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso" if json_mode else "%H:%M:%S"),
                structlog.processors.JSONRenderer() if json_mode else structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )

    return structlog.get_logger("evolvers")


log = _build_logger()
