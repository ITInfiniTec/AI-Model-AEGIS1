# logger.py

import logging
import sys

def setup_logger():
    """
    Configures and returns a singleton logger for the AEGIS application.
    This ensures consistent, centralized logging across all modules.
    """
    logger = logging.getLogger("AEGIS_CORE")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the function is called more than once.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

log = setup_logger()