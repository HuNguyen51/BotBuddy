import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a logger instance.
    
    Args:
        name: The name of the logger. 
              If None, configures the ROOT logger (recommended for the main entry point to capture all logs).
              If string, returns a named logger that inherits handlers from ROOT.
        log_level: The logging level (default: logging.INFO).
        log_file: Optional path to a log file. Only used when configuring ROOT logger (name=None).
    
    Returns:
        logging.Logger: The configured logger.
    
    Note:
        Only the ROOT logger (name=None) should have handlers configured.
        Named loggers will automatically inherit handlers from ROOT through propagation.
        This prevents duplicate log messages.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # For named loggers (child loggers), just return them without adding handlers.
    # They will inherit handlers from the ROOT logger through propagation.
    if name is not None:
        return logger

    # Only configure handlers for ROOT logger (name=None)
    # Check if ROOT logger already has handlers to avoid duplicates
    if logger.handlers:
        return logger

    formatter = logging.Formatter(DEFAULT_FORMAT)

    # 1. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        log_path = Path(log_file)
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler to manage file size (e.g., max 10MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10 * 1024 * 1024, # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
