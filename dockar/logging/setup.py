"""Structured logging setup."""

import logging
import sys

import structlog

from dockar.config.models import LoggingConfig


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure standard logging and structlog for JSON-friendly output."""

    logging_config = config or LoggingConfig()
    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if logging_config.json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, logging_config.level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
