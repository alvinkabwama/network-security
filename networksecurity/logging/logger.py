import logging
import os
from datetime import datetime

LOG_FILE=f"network_security_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)