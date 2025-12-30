"""Logging configuration using loguru."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

from src.models.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure loguru logger based on configuration.

    Args:
        config: Logging configuration
    """
    # Remove default handler
    logger.remove()

    # Add console handler with optional colorization
    logger.add(
        sys.stderr,
        level=config.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=config.colorize,
        serialize=False,
    )

    # Add file handler with JSON serialization if enabled
    log_path = Path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        config.file_path,
        level=config.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=config.rotation,
        retention=config.retention,
        compression=config.compression,
        serialize=config.serialize,
        enqueue=True,  # Async logging
    )

    logger.info(
        "Logging configured",
        level=config.level,
        file=config.file_path,
        serialize=config.serialize,
    )


def get_logger(name: str) -> "Logger":
    """
    Get a logger instance with a specific name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
