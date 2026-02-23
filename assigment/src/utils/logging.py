"""Logging utilities for observability."""
import sys
from typing import Optional
from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file logger
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )


def get_logger(name: str) -> "logger":
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
