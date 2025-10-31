import sys
from networksecurity.logging import logger
from networksecurity.exception.exception import NetworkSecurityException

if __name__ == "__main__":
    try:
        logger.logging.info("Testing NetworkSecurityException")
        a = 1 / 0
        print("This will not be printed due to exception")
    except Exception as e:
        raise NetworkSecurityException(e, sys)