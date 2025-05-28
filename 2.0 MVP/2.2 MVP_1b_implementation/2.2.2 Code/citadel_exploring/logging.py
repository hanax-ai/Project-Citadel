
"""
Logging utilities for Citadel.

This module provides logging functionality for the Citadel project.
"""

import logging
import sys
from typing import Optional, Dict, Any

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_format: str = DEFAULT_FORMAT,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Name of the logger.
        level: Logging level.
        log_format: Format string for log messages.
        log_file: Path to log file. If None, logs to console only.

    Returns:
        Configured logger instance.
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with the specified configuration.

    Args:
        name: Name of the logger.
        config: Configuration dictionary. If None, default configuration is used.

    Returns:
        Configured logger instance.
    """
    if config is None:
        config = {}
    
    level = config.get("level", "INFO")
    log_format = config.get("format", DEFAULT_FORMAT)
    log_file = config.get("file", None)
    
    return setup_logger(name, level, log_format, log_file)
