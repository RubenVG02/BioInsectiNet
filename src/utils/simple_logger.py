from datetime import datetime

def log_with_timestamp(message: str, level: str = "INFO"):
    """
    Print message with timestamp in consistent format.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, SUCCESS)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    color_codes = {
        "INFO": "",
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "SUCCESS": "\033[92m"   # Green
    }
    
    reset_code = "\033[0m" if level in ["WARNING", "ERROR", "SUCCESS"] else ""
    color = color_codes.get(level, "")
    
    log_message = f"[{timestamp}] [{level}] {color}{message}{reset_code}"
    print(log_message)

def log_info(message: str):
    log_with_timestamp(message, "INFO")

def log_warning(message: str):
    log_with_timestamp(message, "WARNING")

def log_error(message: str):
    log_with_timestamp(message, "ERROR")

def log_success(message: str):
    log_with_timestamp(message, "SUCCESS")
