import logging

from settings import DATE_FORMAT, FORMAT, LOGGING_LEVEL

logger = logging.getLogger()


def set_log_config():
    """_
    Set log level
    """

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(LOGGING_LEVEL)
    straem_handler = logging.StreamHandler()
    straem_handler.setLevel(LOGGING_LEVEL)

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
    straem_handler.setFormatter(formatter)
    logger.addHandler(straem_handler)
