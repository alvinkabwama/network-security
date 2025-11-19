import sys  # Used to fetch exception info such as traceback details
from networksecurity.logging import logger  # Custom logger for logging exceptions or activity


# Custom exception class for the NetworkSecurity project
class NetworkSecurityException(Exception):
    def __init__(self, error_message: str, error_detail: sys) -> None:
        # Store the raw error message
        self.error_message = error_message

        # Extract traceback details from the current exception context
        _, _, exc_tb = error_detail.exc_info()

        # Capture the line number where the exception occurred
        self.lineno = exc_tb.tb_lineno

        # Capture the filename where the exception occurred
        self.filename = exc_tb.tb_frame.f_code.co_filename

    # Defines how the exception should be displayed when converted to a string
    def __str__(self) -> str:
        return "Error occurened in python script name [{0}] at line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message)
        )


# Test block to verify that the custom exception captures details correctly
if __name__ == "__main__":
    try:
        logger.logging.info("Testing NetworkSecurityException")
        a = 1 / 0  # Force a ZeroDivisionError
        print("This will not be printed due to exception")
    except Exception as e:
        # Raise custom exception with traceback info
        raise NetworkSecurityException(e, sys)
