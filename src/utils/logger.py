import logging
import sys
from datetime import datetime
from typing import Optional
import os

class BioInsectiNetLogger:
    """
    Centralized logger for BioInsectiNet with consistent formatting.
    """
    
    def __init__(self, name: str = "BioInsectiNet", log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def success(self, message: str):
        """Log success message (using info level with green color)."""
        colored_message = f"\033[92m{message}\033[0m"
        self.logger.info(colored_message)

_global_logger: Optional[BioInsectiNetLogger] = None

def get_logger(name: str = "BioInsectiNet", log_file: Optional[str] = None) -> BioInsectiNetLogger:
    """
    Get or create a global logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        BioInsectiNetLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = BioInsectiNetLogger(name, log_file)
    return _global_logger

def set_log_file(log_file: str):
    """
    Set the log file for the global logger.
    
    Args:
        log_file: Path to log file
    """
    global _global_logger
    if _global_logger is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        _global_logger.logger.addHandler(file_handler)

def log_info(message: str):
    """Log info message using global logger."""
    get_logger().info(message)

def log_warning(message: str):
    """Log warning message using global logger."""
    get_logger().warning(message)

def log_error(message: str):
    """Log error message using global logger."""
    get_logger().error(message)

def log_success(message: str):
    """Log success message using global logger."""
    get_logger().success(message)

def log_debug(message: str):
    """Log debug message using global logger."""
    get_logger().debug(message)
