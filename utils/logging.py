import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir=None, level=logging.INFO, suffix=None):
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [] # No console logging

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True) # Create log directory if it doesn't exist
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{suffix}.log" if suffix else f"{timestamp}.log"
        log_file = Path(log_dir) / filename
        handlers.append(logging.FileHandler(log_file)) # Log to file
    
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress httpx logging
    logging.getLogger("google_genai.models").setLevel(logging.WARNING) # Suppress AFC messages

def get_logger(name):
    return logging.getLogger(name)