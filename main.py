import sys
from networksecurity.logging import logger


class NetworkSecurityException(Exception):
    def __init__(self, error_message: str, error_detail: sys) -> None:
        self.error_message = error_message
        _,_,exc_tb = error_detail.exc_info()

        self.lineno=exc_tb.tb_lineno
        self.filename=exc_tb.tb_frame.f_code.co_filename



    def __str__(self) -> str:
        return "Error occurened in python script name [{0}] at line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message))
    

if __name__ == "__main__":
    try:
        logger.logging.info("Testing NetworkSecurityException")
        a = 1 / 0
        print("This will not be printed due to exception")
    except Exception as e:
        raise NetworkSecurityException(e, sys)