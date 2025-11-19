import logging  # Python's built-in logging module for writing logs to files
import os  # Used to create directories and build file paths
from datetime import datetime  # Used to generate timestamped log filenames


# Create a log filename with a timestamp (e.g., network_security_2025-11-19_12-45-01.log)
LOG_FILE = f"network_security_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Build the logs directory path -> <current working directory>/logs/<timestamped_log_filename>
logs_dir = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Ensure that the directory exists; creates it if missing
os.makedirs(logs_dir, exist_ok=True)

# Final log file path inside the created directory
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure Python's logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,              # File where logs will be stored
    format='[%(asctime)s] %(levelname)s - %(message)s',  # Log message format with timestamp & level
    level=logging.INFO                   # Logging level (INFO and above)
)
