"""Structured logging configuration.

Call ``setup_logging()`` once from every entrypoint (CLI / web) before
any other application code runs.  Library modules should simply use
``logging.getLogger(__name__)`` — they inherit the root configuration.
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """Configure the root logger with a consistent format.

    In production (``LOG_FORMAT=json``), emits one-line JSON records
    suitable for Railway / CloudWatch / Datadog.  Otherwise falls back
    to a human-readable format.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    if log_format == "json":
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
