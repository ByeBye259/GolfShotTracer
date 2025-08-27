import logging
from typing import Optional
import os
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    # ANSI color codes
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    
    COLORS = {
        'WARNING': YELLOW,
        'INFO': WHITE,
        'DEBUG': BLUE,
        'CRITICAL': YELLOW,
        'ERROR': RED
    }
    
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            levelname_color = self.COLOR_SEQ % (30 + self.COLORS[levelname]) + levelname + self.RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

def get_logger(name: str, log_level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    # Set default log level if not provided
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Format for console output
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = ColoredFormatter(console_format)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Format for file output
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_formatter = logging.Formatter(file_format)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = get_logger("example", log_level="DEBUG", log_file="example.log")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
