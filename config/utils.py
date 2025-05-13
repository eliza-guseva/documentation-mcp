import logging
import sys


DEFAULT_LOG_FILE = 'documentation.log'


def setup_logging(
    log_file: str = DEFAULT_LOG_FILE, 
    level: int = logging.INFO
    ) -> None:
    """
    Configures logging to file and console.
    Args:
        log_file (str): The path to the log file.
        level (int): The logging level.
    """
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates if called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance."""
    return logging.getLogger(name) 

